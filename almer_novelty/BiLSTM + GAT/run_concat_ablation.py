import argparse
import random
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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

from prott5_dataset import resolve_data_dir, ProtT5FeatureExtractor, AllergenProteinGraphDataset
from gat_model import ProtT5GATBranch
from bilstm_model import BiLSTMPhysEvoBranch
from fusion_model import MultimodalAllergenModel, LabelSmoothedFocalLoss
from run_ablations import (
    GATOnlyAblationModel,
    BiLSTMOnlyAblationModel,
    compute_all_metrics,
    predict_gat_only,
    predict_bilstm_only,
    get_full_model_test_predictions,
)


# ==============================================================================
# 1. Simple Concatenation Model (No Cross-Modal Attention, No Fusion Residuals)
# ==============================================================================
class ConcatOnlyAblationModel(nn.Module):
    """
    BiLSTM + GAT Simple Concatenation Ablation (Table 3 Row 3 / Fig. 6c):
    - Extracts F_g [1, 192] from ProtT5GATBranch
    - Extracts F_p [1, 192] from BiLSTMPhysEvoBranch
    - Skips BidirectionalCrossModalAttention & its residual connections
    - Directly concatenates [F_g, F_p] -> [1, 384] into the identical classifier head
    """
    def __init__(self, prot_dim: int = 1024, desc_dim: int = 227, hidden_dim: int = 192, dropout: float = 0.2):
        super().__init__()
        self.gat_branch = ProtT5GATBranch(in_dim=prot_dim, out_dim=hidden_dim)
        self.bilstm_branch = BiLSTMPhysEvoBranch(
            in_features=desc_dim, out_dim=hidden_dim, dropout=dropout
        )

        # Identical 384 -> 192 -> 1 classifier head as MultimodalAllergenModel
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Self-supervised node reconstruction branch (Eq. 26)
        self.reconstructor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, prot_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, desc: torch.Tensor):
        f_g, h2 = self.gat_branch(x, edge_index)  # f_g: [1, 192], h2: [V, 192]
        f_p = self.bilstm_branch(desc)            # f_p: [1, 192]

        # Direct concatenation without cross-modal interaction or fusion residuals
        f_concat = torch.cat([f_g, f_p], dim=-1)  # [1, 384]

        logits = self.classifier(f_concat)        # [1, 1]
        x_recon = self.reconstructor(h2)          # [V, prot_dim]
        return logits, x_recon, f_concat


# ==============================================================================
# 2. Evaluation & 5-Fold Cross-Validation Trainer for Concat-Only
# ==============================================================================
@torch.no_grad()
def predict_concat_only(model, dataset, indices, criterion_cls, device):
    model.eval()
    probs, labels = [], []
    total_loss = 0.0

    for idx in indices:
        x, edge_index, desc, y, _ = dataset[idx]
        x = x.to(device)
        edge_index = edge_index.to(device)
        desc = desc.to(device).unsqueeze(0)
        y_dev = y.to(device).view(1, 1)

        logits, x_recon, _ = model(x, edge_index, desc)
        loss_cls = criterion_cls(logits, y_dev)
        loss_recon = torch.mean(torch.abs(x_recon - x))
        total_loss += (loss_cls + 0.1 * loss_recon).item()

        probs.append(torch.sigmoid(logits).item())
        labels.append(int(y.item()))

    y_true = np.array(labels, dtype=np.int32)
    y_probs = np.array(probs, dtype=np.float32)
    metrics = compute_all_metrics(y_true, y_probs)
    metrics["loss"] = total_loss / max(len(indices), 1)
    return y_true, y_probs, metrics


def train_cv_concat_only(full_train_ds, skf, all_labels, args, prot_dim, base_dir, device):
    print("\n======================================================================")
    print("TRAINING ABLATION 3: SIMPLE CONCATENATION (BiLSTM + GAT Without Interaction)")
    print("======================================================================")
    criterion_cls = LabelSmoothedFocalLoss(alpha=0.5, gamma=1.0, smoothing=0.1)
    use_amp = device.type == "cuda"
    best_overall_f1 = -1.0
    final_path = base_dir / "model_concat_only.pt"
    fold_metrics_list = []

    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(np.zeros(len(all_labels)), all_labels), start=1):
        print(f"\n--- Concat-Only | Fold {fold_idx} / {args.folds} ---")
        model = ConcatOnlyAblationModel(
            prot_dim=prot_dim, desc_dim=227, hidden_dim=192, dropout=0.2
        ).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(args.epochs, 1), T_mult=1, eta_min=1e-6
        )
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        train_order = train_idx.tolist()
        fold_path = base_dir / f"model_concat_only.fold{fold_idx}.pt"
        best_fold_f1 = -1.0
        best_fold_m = None

        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            random.shuffle(train_order)

            for step, idx in enumerate(train_order, start=1):
                x, edge_index, desc, y, _ = full_train_ds[idx]
                x = x.to(device)
                edge_index = edge_index.to(device)
                desc = desc.to(device).unsqueeze(0)
                y_dev = y.to(device).view(1, 1)

                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits, x_recon, _ = model(x, edge_index, desc)
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
                        f"  [Concat-Only | Fold {fold_idx} | {step}/{len(train_order)}] "
                        f"Avg Train Loss: {running_loss / step:.4f}"
                    )

            scheduler.step()
            _, _, val_m = predict_concat_only(model, full_train_ds, valid_idx.tolist(), criterion_cls, device)
            print(
                f"  Fold {fold_idx} Val -> Loss: {val_m['loss']:.4f} | ACC: {val_m['acc']:.4f} | "
                f"F1: {val_m['f1']:.4f} | Prec: {val_m['prec']:.4f} | Recall: {val_m['recall']:.4f} | "
                f"AUC: {val_m['auc']:.4f} | AUPRC: {val_m['auprc']:.4f} | MCC: {val_m['mcc']:.4f}"
            )
            if val_m["f1"] > best_fold_f1:
                best_fold_f1 = val_m["f1"]
                best_fold_m = val_m
                torch.save(model.state_dict(), fold_path)

        fold_metrics_list.append(best_fold_m)
        if best_fold_f1 > best_overall_f1:
            best_overall_f1 = best_fold_f1
            shutil.copyfile(fold_path, final_path)

    print("\n--- 5-Fold CV Summary (Simple Concatenation) ---")
    summary_parts = []
    for key in ["loss", "acc", "f1", "prec", "recall", "auc", "auprc", "mcc"]:
        vals = [m[key] for m in fold_metrics_list]
        summary_parts.append(f"{key.upper()}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
    print(" | ".join(summary_parts))
    print(f"--> Best Concat-Only model saved to: {final_path.name} (Val F1: {best_overall_f1:.4f})")
    return final_path


# ==============================================================================
# 3. Side-by-Side AUROC & AUPRC Plotter (All Ablation Variants)
# ==============================================================================
def plot_ablation_curves(results_list: list, out_png: Path):
    plt.rcParams.update({"font.size": 11, "font.family": "sans-serif"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    for name, color, lw, (y_true, y_probs, m) in results_list:
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        prec, rec, _ = precision_recall_curve(y_true, y_probs)

        ax1.plot(fpr, tpr, color=color, lw=lw, label=f"{name} (AUROC = {m['auc']:.4f})")
        ax2.plot(rec, prec, color=color, lw=lw, label=f"{name} (AUPRC = {m['auprc']:.4f})")

    # Left: AUROC
    ax1.plot([0, 1], [0, 1], color="navy", lw=1.2, linestyle="--", alpha=0.6)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.02])
    ax1.set_xlabel("False Positive Rate", fontweight="bold")
    ax1.set_ylabel("True Positive Rate", fontweight="bold")
    ax1.set_title("Receiver Operating Characteristic (ROC) Curve", fontweight="bold", pad=10)
    ax1.legend(loc="lower right", frameon=True, fontsize=9.5)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Right: AUPRC
    ax2.plot([0, 1], [1, 0], color="navy", lw=1.2, linestyle="--", alpha=0.6)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.02])
    ax2.set_xlabel("Recall", fontweight="bold")
    ax2.set_ylabel("Precision", fontweight="bold")
    ax2.set_title("Precision-Recall (PR) Curve", fontweight="bold", pad=10)
    ax2.legend(loc="lower left", frameon=True, fontsize=9.5)
    ax2.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train & Test Simple Concatenation Ablation (BiLSTM + GAT)")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs per fold (default: 1)")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds (default: 5)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Initial learning rate (default: 5e-5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-train", action="store_true", help="Skip training if model_concat_only.pt exists")
    parser.add_argument("--concat-3-layers", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = Path(__file__).resolve().parent
    data_dir = resolve_data_dir()
    cache_dir = base_dir / "cache_prott5"

    extractor = ProtT5FeatureExtractor(
        device=device,
        concat_last_3_layers=args.concat_3_layers,
        cache_dir=cache_dir,
    )
    prot_dim = 3072 if args.concat_3_layers else 1024

    full_train_ds = AllergenProteinGraphDataset(data_dir / "all.train.fasta", extractor=extractor)
    test_ds = AllergenProteinGraphDataset(data_dir / "all.test.fasta", extractor=extractor)

    all_labels = np.array([int(rec[2]) for rec in full_train_ds.records])
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    concat_pt = base_dir / "model_concat_only.pt"
    if not (args.skip_train and concat_pt.exists()):
        train_cv_concat_only(full_train_ds, skf, all_labels, args, prot_dim, base_dir, device)
    extractor.unload_from_gpu()

    # Evaluate Simple Concatenation Model on all.test.fasta
    criterion_cls = LabelSmoothedFocalLoss(alpha=0.5, gamma=1.0, smoothing=0.1)
    test_indices = list(range(len(test_ds)))

    concat_model = ConcatOnlyAblationModel(prot_dim=prot_dim, desc_dim=227, hidden_dim=192).to(device)
    concat_model.load_state_dict(torch.load(concat_pt, map_location=device, weights_only=True), strict=True)
    y_concat, probs_concat, m_concat = predict_concat_only(
        concat_model, test_ds, test_indices, criterion_cls, device
    )

    # Save per-protein test predictions for Simple Concatenation
    concat_records = []
    for idx, prob in enumerate(probs_concat):
        rec_id, _, label = test_ds.records[idx]
        pred_str = "allergenicity" if prob >= 0.5 else "non-allergenicity"
        concat_records.append([rec_id, int(label), round(float(prob), 6), pred_str])
    pd.DataFrame(
        concat_records, columns=["protein", "true_label", "score", "predict_result"]
    ).to_csv(base_dir / "test_predictions_concat.tsv", sep="\t", index=False)

    # Gather other ablation models if present for a complete Table 3 comparison
    rows = []
    plot_items = []

    bilstm_pt = base_dir / "model_bilstm_only.pt"
    if bilstm_pt.exists():
        bilstm_model = BiLSTMOnlyAblationModel(desc_dim=227, hidden_dim=192).to(device)
        bilstm_model.load_state_dict(torch.load(bilstm_pt, map_location=device, weights_only=True), strict=True)
        y_bi, probs_bi, m_bi = predict_bilstm_only(bilstm_model, test_ds, test_indices, criterion_cls, device)
        rows.append(("BiLSTM-Only (Phys+Evo)", m_bi))
        plot_items.append(("BiLSTM-Only", "#eab308", 2.0, (y_bi, probs_bi, m_bi)))  # Yellow

    gat_pt = base_dir / "model_gat_only.pt"
    if gat_pt.exists():
        gat_model = GATOnlyAblationModel(prot_dim=prot_dim, hidden_dim=192).to(device)
        gat_model.load_state_dict(torch.load(gat_pt, map_location=device, weights_only=True), strict=True)
        y_gat, probs_gat, m_gat = predict_gat_only(gat_model, test_ds, test_indices, criterion_cls, device)
        rows.append(("GAT-Only (ProtT5)", m_gat))
        plot_items.append(("GAT-Only", "#1f77b4", 2.0, (y_gat, probs_gat, m_gat)))  # Blue

    # Add Simple Concatenation (Green)
    rows.append(("BiLSTM + GAT (Simple Concat)", m_concat))
    plot_items.append(("BiLSTM + GAT (Simple Concat)", "#2ca02c", 2.2, (y_concat, probs_concat, m_concat)))

    # Add Full Architecture with Cross-Modal Attention + Residuals (Red)
    if (base_dir / "model.pt").exists() or (base_dir / "test_predictions.tsv").exists():
        y_full, probs_full, m_full = get_full_model_test_predictions(base_dir, test_ds, prot_dim, device)
        rows.append(("Full Model (+ Cross-Modal Attn)", m_full))
        plot_items.append(("Full Model (+ Cross-Modal Attn)", "#d62728", 2.4, (y_full, probs_full, m_full)))

    # Print & Log Complete Benchmark Table
    header = (
        f"\n{'=' * 100}\n"
        f"{'Ablation Stage / Model':<34} | {'ACC':>7} | {'Prec':>7} | {'Recall':>7} | "
        f"{'F1':>7} | {'AUROC':>7} | {'AUPRC':>7} | {'MCC':>7}\n"
        f"{'-' * 100}"
    )
    lines = [header]
    for title, m in rows:
        lines.append(
            f"{title:<34} | {m['acc']:7.4f} | {m['prec']:7.4f} | {m['recall']:7.4f} | "
            f"{m['f1']:7.4f} | {m['auc']:7.4f} | {m['auprc']:7.4f} | {m['mcc']:7.4f}"
        )
    lines.append("=" * 100)
    table_str = "\n".join(lines)
    print(table_str)

    with open(base_dir / "ablation_test.log", "a") as f:
        f.write(table_str + "\n")

    # Save Side-by-Side AUROC & AUPRC Plot (ordered with Full Model on top in legend)
    plot_items_reversed = list(reversed(plot_items))
    out_png = base_dir / "ablation_auroc_auprc_comparison.png"
    plot_ablation_curves(plot_items_reversed, out_png)
    print(f"\nSaved updated side-by-side AUROC & AUPRC plot to: {out_png}")


if __name__ == "__main__":
    main()