import argparse
import random
import shutil
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    roc_curve,
    precision_recall_curve,
)

from prott5_graph_dataset import resolve_data_dir, IsolatedProtT5Extractor, ProteinGraphDataset
from gnn_models import ProtT5GNNClassifier, LabelSmoothedFocalLoss


def compute_metrics(y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_probs >= threshold).astype(np.int32)
    return {
        "acc": accuracy_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_probs),
        "auprc": average_precision_score(y_true, y_probs),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


@torch.no_grad()
def evaluate_gnn(model, dataset, indices, criterion_cls, device):
    model.eval()
    probs, labels = [], []
    total_loss = 0.0
    t0 = time.perf_counter()

    for idx in indices:
        x, edge_index, y, _ = dataset[idx]
        x = x.to(device)
        edge_index = edge_index.to(device)
        y_dev = y.to(device).view(1, 1)

        logits, x_recon, _ = model(x, edge_index)
        loss_cls = criterion_cls(logits, y_dev)
        loss_recon = torch.mean(torch.abs(x_recon - x))
        total_loss += (loss_cls + 0.1 * loss_recon).item()

        probs.append(torch.sigmoid(logits).item())
        labels.append(int(y.item()))

    elapsed_ms = ((time.perf_counter() - t0) * 1000.0) / max(len(indices), 1)
    y_true = np.array(labels, dtype=np.int32)
    y_probs = np.array(probs, dtype=np.float32)
    metrics = compute_metrics(y_true, y_probs)
    metrics["loss"] = total_loss / max(len(indices), 1)
    metrics["ms_per_protein"] = elapsed_ms
    return y_true, y_probs, metrics


def train_single_gnn_family(
    gnn_type: str,
    display_name: str,
    train_ds: ProteinGraphDataset,
    extractor: IsolatedProtT5Extractor,
    skf: StratifiedKFold,
    all_labels: np.ndarray,
    args,
    base_dir: Path,
    device: torch.device,
) -> Path:
    print(f"5-FOLD CROSS-VALIDATION TRAINING: {display_name} (gnn_type='{gnn_type}')")

    # Reset seeds before each GNN family so weight initialization & batch shuffling are 100% fair
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    criterion_cls = LabelSmoothedFocalLoss(alpha=0.5, gamma=1.0, smoothing=0.1)
    use_amp = device.type == "cuda"
    best_overall_f1 = -1.0
    final_pt = base_dir / f"model_{gnn_type}.pt"
    fold_metrics = []

    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(np.zeros(len(all_labels)), all_labels), start=1):
        model = ProtT5GNNClassifier(gnn_type=gnn_type, prot_dim=1024, hidden_dim=192, dropout=0.2).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(args.epochs, 1), T_mult=1, eta_min=1e-6
        )
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        train_order = train_idx.tolist()
        fold_pt = base_dir / f"model_{gnn_type}.fold{fold_idx}.pt"
        best_fold_f1 = -1.0
        best_fold_m = None

        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            random.shuffle(train_order)

            for step, idx in enumerate(train_order, start=1):
                x, edge_index, y, _ = train_ds[idx]
                x = x.to(device)
                edge_index = edge_index.to(device)
                y_dev = y.to(device).view(1, 1)

                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits, x_recon, _ = model(x, edge_index)
                    loss_cls = criterion_cls(logits, y_dev)
                    loss_recon = torch.mean(torch.abs(x_recon - x))
                    loss = loss_cls + 0.1 * loss_recon

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()

                if step % 1000 == 0 or step == len(train_order):
                    print(
                        f"  [{display_name} | Fold {fold_idx} | {step}/{len(train_order)}] "
                        f"Avg Train Loss: {running_loss / step:.4f}"
                    )

            scheduler.step()
            _, _, val_m = evaluate_gnn(model, train_ds, valid_idx.tolist(), criterion_cls, device)

            # Once Fold 1 Epoch 1 of the first model finishes caching, free ProtT5 from VRAM
            if fold_idx == 1 and epoch == 1:
                extractor.unload_from_gpu()

            print(
                f"  Fold {fold_idx} Val -> Loss: {val_m['loss']:.4f} | ACC: {val_m['acc']:.4f} | "
                f"F1: {val_m['f1']:.4f} | Prec: {val_m['prec']:.4f} | Recall: {val_m['recall']:.4f} | "
                f"AUROC: {val_m['auc']:.4f} | AUPRC: {val_m['auprc']:.4f} | MCC: {val_m['mcc']:.4f}"
            )

            if val_m["f1"] > best_fold_f1:
                best_fold_f1 = val_m["f1"]
                best_fold_m = val_m
                torch.save(model.state_dict(), fold_pt)

        fold_metrics.append(best_fold_m)
        if best_fold_f1 > best_overall_f1:
            best_overall_f1 = best_fold_f1
            shutil.copyfile(fold_pt, final_pt)

    summary = " | ".join(
        f"{k.upper()}: {np.mean([m[k] for m in fold_metrics]):.4f}"
        for k in ["acc", "f1", "prec", "recall", "auc", "auprc", "mcc"]
    )
    print(f"--> {display_name} 5-Fold Val Mean | {summary}")
    print(f"--> Saved best {display_name} checkpoint to: {final_pt.name}")
    return final_pt


def plot_gnn_comparison(results_list: list, out_png: Path):
    # Plots side-by-side AUROC (left) and AUPRC (right) across GCN, GAT, and Graph Transformer.
    plt.rcParams.update({"font.size": 11, "font.family": "sans-serif"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.8), dpi=300)

    for display_name, color, lw, (y_true, y_probs, m) in results_list:
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        prec, rec, _ = precision_recall_curve(y_true, y_probs)

        ax1.plot(fpr, tpr, color=color, lw=lw, label=f"{display_name} (AUROC = {m['auc']:.4f})")
        ax2.plot(rec, prec, color=color, lw=lw, label=f"{display_name} (AUPRC = {m['auprc']:.4f})")

    ax1.plot([0, 1], [0, 1], color="navy", lw=1.2, linestyle="--", alpha=0.6)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.02])
    ax1.set_xlabel("False Positive Rate", fontweight="bold")
    ax1.set_ylabel("True Positive Rate", fontweight="bold")
    ax1.set_title("Receiver Operating Characteristic (ROC) Curve", fontweight="bold", pad=10)
    ax1.legend(loc="lower right", frameon=True, fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax2.plot([0, 1], [1, 0], color="navy", lw=1.2, linestyle="--", alpha=0.6)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.02])
    ax2.set_xlabel("Recall", fontweight="bold")
    ax2.set_ylabel("Precision", fontweight="bold")
    ax2.set_title("Precision-Recall (PR) Curve", fontweight="bold", pad=10)
    ax2.legend(loc="lower left", frameon=True, fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Isolated Benchmark of ProtT5 + {GCN, GAT, Graph Transformer}"
    )
    parser.add_argument("--epochs", type=int, default=1, help="Epochs per fold (default: 1)")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds (default: 5)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (default: 5e-5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-train", action="store_true", help="Skip training if .pt files exist")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = Path(__file__).resolve().parent
    data_dir = resolve_data_dir()
    local_cache_dir = base_dir / "cache_prott5"

    extractor = IsolatedProtT5Extractor(device=device, cache_dir=local_cache_dir)
    train_ds = ProteinGraphDataset(data_dir / "all.train.fasta", extractor=extractor)
    test_ds = ProteinGraphDataset(data_dir / "all.test.fasta", extractor=extractor)

    all_labels = np.array([int(rec[2]) for rec in train_ds.records])
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    criterion_cls = LabelSmoothedFocalLoss(alpha=0.5, gamma=1.0, smoothing=0.1)
    test_indices = list(range(len(test_ds)))

    gnn_configs = [
        ("gcn", "ProtT5 + GCN",               "#eab308", 2.2),  # Yellow
        ("gat", "ProtT5 + GAT",               "#1f77b4", 2.2),  # Blue
        ("gt",  "ProtT5 + Graph Transformer", "#d62728", 2.4),  # Red
    ]

    table_rows = []
    plot_items = []

    for gnn_type, display_name, color, lw in gnn_configs:
        pt_path = base_dir / f"model_{gnn_type}.pt"
        if not (args.skip_train and pt_path.exists()):
            train_single_gnn_family(
                gnn_type, display_name, train_ds, extractor, skf, all_labels, args, base_dir, device
            )

        # Ensure all test set embeddings are cached before unloading ProtT5
        for idx in test_indices:
            _ = test_ds[idx]
        extractor.unload_from_gpu()

        model = ProtT5GNNClassifier(gnn_type=gnn_type, prot_dim=1024, hidden_dim=192, dropout=0.2).to(device)
        model.load_state_dict(torch.load(pt_path, map_location=device, weights_only=True), strict=True)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        y_true, y_probs, test_m = evaluate_gnn(model, test_ds, test_indices, criterion_cls, device)
        test_m["params"] = num_params

        # Save per-model test predictions TSV inside PPLM + GNN
        tsv_path = base_dir / f"test_predictions_{gnn_type}.tsv"
        rows_tsv = [
            [test_ds.records[i][0], int(y_true[i]), round(float(y_probs[i]), 6),
             "allergenicity" if y_probs[i] >= 0.5 else "non-allergenicity"]
            for i in range(len(y_probs))
        ]
        pd.DataFrame(
            rows_tsv, columns=["protein", "true_label", "score", "predict_result"]
        ).to_csv(tsv_path, sep="\t", index=False)

        table_rows.append((display_name, test_m))
        plot_items.append((display_name, color, lw, (y_true, y_probs, test_m)))

    # Print & log final test comparison table
    header = (
        f"\n{'=' * 116}\n"
        f"{'PPLM + GNN Architecture':<28} | {'ACC':>7} | {'Prec':>7} | {'Recall':>7} | "
        f"{'F1':>7} | {'AUROC':>7} | {'AUPRC':>7} | {'MCC':>7} | {'Params':>9} | {'ms/protein':>7}\n"
        f"{'-' * 116}"
    )
    lines = [header]
    for name, m in table_rows:
        lines.append(
            f"{name:<28} | {m['acc']:7.4f} | {m['prec']:7.4f} | {m['recall']:7.4f} | "
            f"{m['f1']:7.4f} | {m['auc']:7.4f} | {m['auprc']:7.4f} | {m['mcc']:7.4f} | "
            f"{m['params']:9,d} | {m['ms_per_protein']:7.2f}"
        )
    lines.append("=" * 116)
    output_table = "\n".join(lines)
    print(output_table)

    with open(base_dir / "gnn_benchmark.log", "a") as f:
        f.write(output_table + "\n")

    out_png = base_dir / "gnn_auroc_auprc_comparison.png"
    plot_gnn_comparison(plot_items, out_png)
    print(f"\nSaved side-by-side GNN comparison figure to: {out_png}")


if __name__ == "__main__":
    main()