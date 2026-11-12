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


# ==============================================================================
# 1. Ablation Model Definitions (Exact Branches, Shape-Fitted Classifier 192->192->1)
# ==============================================================================
class GATOnlyAblationModel(nn.Module):
    """
    First Ablation (GAT-Only):
    Uses ProtT5GATBranch [V, 1024] -> F_g [1, 192] directly into the classifier
    without Bi-LSTM or Bidirectional Cross-Modal Attention Fusion.
    """
    def __init__(self, prot_dim: int = 1024, hidden_dim: int = 192, dropout: float = 0.2):
        super().__init__()
        self.gat_branch = ProtT5GATBranch(in_dim=prot_dim, out_dim=hidden_dim)

        # Shape-fitted from (hidden_dim * 2, hidden_dim) -> (hidden_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.reconstructor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, prot_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        f_g, h2 = self.gat_branch(x, edge_index)  # f_g: [1, 192], h2: [V, 192]
        logits = self.classifier(f_g)             # [1, 1]
        x_recon = self.reconstructor(h2)          # [V, prot_dim]
        return logits, x_recon, f_g


class BiLSTMOnlyAblationModel(nn.Module):
    """
    Second Ablation (BiLSTM-Only):
    Uses BiLSTMPhysEvoBranch [1, 227] -> F_p [1, 192] directly into the classifier
    without ProtT5-GAT or Bidirectional Cross-Modal Attention Fusion.
    """
    def __init__(self, desc_dim: int = 227, hidden_dim: int = 192, dropout: float = 0.2):
        super().__init__()
        self.bilstm_branch = BiLSTMPhysEvoBranch(
            in_features=desc_dim, out_dim=hidden_dim, dropout=dropout
        )

        # Shape-fitted from (hidden_dim * 2, hidden_dim) -> (hidden_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, desc: torch.Tensor):
        f_p = self.bilstm_branch(desc)  # f_p: [1, 192]
        logits = self.classifier(f_p)   # [1, 1]
        return logits, f_p


# ==============================================================================
# 2. Metric Calculation Helper
# ==============================================================================
def compute_all_metrics(y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5) -> dict:
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


# ==============================================================================
# 3. 5-Fold Cross-Validation Trainers for GAT-Only & BiLSTM-Only
# ==============================================================================
@torch.no_grad()
def predict_gat_only(model, dataset, indices, criterion_cls, device):
    model.eval()
    probs, labels = [], []
    total_loss = 0.0
    for idx in indices:
        x, edge_index, _, y, _ = dataset[idx]
        x = x.to(device)
        edge_index = edge_index.to(device)
        y_dev = y.to(device).view(1, 1)

        logits, x_recon, _ = model(x, edge_index)
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


@torch.no_grad()
def predict_bilstm_only(model, dataset, indices, criterion_cls, device):
    model.eval()
    probs, labels = [], []
    total_loss = 0.0
    for idx in indices:
        desc = dataset.desc_Lookup[idx].to(device).unsqueeze(0)
        label_val = dataset.records[idx][2]
        y_dev = torch.tensor([[label_val]], dtype=torch.float32, device=device)

        logits, _ = model(desc)
        loss_cls = criterion_cls(logits, y_dev)
        total_loss += loss_cls.item()

        probs.append(torch.sigmoid(logits).item())
        labels.append(int(label_val))

    y_true = np.array(labels, dtype=np.int32)
    y_probs = np.array(probs, dtype=np.float32)
    metrics = compute_all_metrics(y_true, y_probs)
    metrics["loss"] = total_loss / max(len(indices), 1)
    return y_true, y_probs, metrics


def train_cv_gat_only(full_train_ds, skf, all_labels, args, prot_dim, base_dir, device):
    print("\n======================================================================")
    print("TRAINING ABLATION 1: GAT-ONLY (ProtT5 Node Features -> GAT -> Classifier)")
    print("======================================================================")
    criterion_cls = LabelSmoothedFocalLoss(alpha=0.5, gamma=1.0, smoothing=0.1)
    use_amp = device.type == "cuda"
    best_overall_f1 = -1.0
    final_path = base_dir / "model_gat_only.pt"

    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(np.zeros(len(all_labels)), all_labels), start=1):
        model = GATOnlyAblationModel(prot_dim=prot_dim, hidden_dim=192, dropout=0.2).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(args.epochs, 1), T_mult=1, eta_min=1e-6
        )
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        train_order = train_idx.tolist()
        fold_path = base_dir / f"model_gat_only.fold{fold_idx}.pt"
        best_fold_f1 = -1.0

        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            random.shuffle(train_order)

            for step, idx in enumerate(train_order, start=1):
                x, edge_index, _, y, _ = full_train_ds[idx]
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
                    print(f"  [GAT-Only | Fold {fold_idx} | {step}/{len(train_order)}] Loss: {running_loss / step:.4f}")

            scheduler.step()
            _, _, val_m = predict_gat_only(model, full_train_ds, valid_idx.tolist(), criterion_cls, device)
            print(
                f"  Fold {fold_idx} Val -> ACC: {val_m['acc']:.4f} | F1: {val_m['f1']:.4f} | "
                f"Prec: {val_m['prec']:.4f} | Recall: {val_m['recall']:.4f} | "
                f"AUC: {val_m['auc']:.4f} | MCC: {val_m['mcc']:.4f}"
            )
            if val_m["f1"] > best_fold_f1:
                best_fold_f1 = val_m["f1"]
                torch.save(model.state_dict(), fold_path)

        if best_fold_f1 > best_overall_f1:
            best_overall_f1 = best_fold_f1
            shutil.copyfile(fold_path, final_path)

    print(f"--> Best GAT-Only model saved to {final_path.name} (Val F1: {best_overall_f1:.4f})")
    return final_path


def train_cv_bilstm_only(full_train_ds, skf, all_labels, args, base_dir, device):
    print("\n======================================================================")
    print("TRAINING ABLATION 2: BiLSTM-ONLY (Phys+Evo Features -> BiLSTM -> Classifier)")
    print("======================================================================")
    criterion_cls = LabelSmoothedFocalLoss(alpha=0.5, gamma=1.0, smoothing=0.1)
    use_amp = device.type == "cuda"
    best_overall_f1 = -1.0
    final_path = base_dir / "model_bilstm_only.pt"

    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(np.zeros(len(all_labels)), all_labels), start=1):
        model = BiLSTMOnlyAblationModel(desc_dim=227, hidden_dim=192, dropout=0.2).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(args.epochs, 1), T_mult=1, eta_min=1e-6
        )
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        train_order = train_idx.tolist()
        fold_path = base_dir / f"model_bilstm_only.fold{fold_idx}.pt"
        best_fold_f1 = -1.0

        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            random.shuffle(train_order)

            for step, idx in enumerate(train_order, start=1):
                # Direct RAM lookup (no ProtT5 graph disk I/O needed for BiLSTM-Only)
                desc = full_train_ds.desc_Lookup[idx].to(device).unsqueeze(0)
                label_val = full_train_ds.records[idx][2]
                y_dev = torch.tensor([[label_val]], dtype=torch.float32, device=device)

                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits, _ = model(desc)
                    loss = criterion_cls(logits, y_dev)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()

                if step % 1000 == 0 or step == len(train_order):
                    print(f"  [BiLSTM-Only | Fold {fold_idx} | {step}/{len(train_order)}] Loss: {running_loss / step:.4f}")

            scheduler.step()
            _, _, val_m = predict_bilstm_only(model, full_train_ds, valid_idx.tolist(), criterion_cls, device)
            print(
                f"  Fold {fold_idx} Val -> ACC: {val_m['acc']:.4f} | F1: {val_m['f1']:.4f} | "
                f"Prec: {val_m['prec']:.4f} | Recall: {val_m['recall']:.4f} | "
                f"AUC: {val_m['auc']:.4f} | MCC: {val_m['mcc']:.4f}"
            )
            if val_m["f1"] > best_fold_f1:
                best_fold_f1 = val_m["f1"]
                torch.save(model.state_dict(), fold_path)

        if best_fold_f1 > best_overall_f1:
            best_overall_f1 = best_fold_f1
            shutil.copyfile(fold_path, final_path)

    print(f"--> Best BiLSTM-Only model saved to {final_path.name} (Val F1: {best_overall_f1:.4f})")
    return final_path


# ==============================================================================
# 4. Full Model Evaluation Loader & 3-Color Side-by-Side Plotting
# ==============================================================================
@torch.no_grad()
def get_full_model_test_predictions(base_dir: Path, test_ds, prot_dim: int, device: torch.device):
    tsv_path = base_dir / "test_predictions.tsv"
    full_pt = base_dir / "model.pt"

    if tsv_path.exists():
        print(f"\nLoading Full Architecture (BiLSTM + GAT) test predictions from: {tsv_path.name}")
        df = pd.read_csv(tsv_path, sep="\t")
        y_true = df["true_label"].to_numpy(dtype=np.int32)
        y_probs = df["score"].to_numpy(dtype=np.float32)
        return y_true, y_probs, compute_all_metrics(y_true, y_probs)

    print(f"\nEvaluating Full Architecture (BiLSTM + GAT) from: {full_pt.name}")
    model = MultimodalAllergenModel(prot_dim=prot_dim, desc_dim=227, hidden_dim=192).to(device)
    model.load_state_dict(torch.load(full_pt, map_location=device, weights_only=True), strict=True)
    model.eval()

    probs, labels = [], []
    for idx in range(len(test_ds)):
        x, edge_index, desc, y, _ = test_ds[idx]
        logits, _, _ = model(x.to(device), edge_index.to(device), desc.to(device).unsqueeze(0))
        probs.append(torch.sigmoid(logits).item())
        labels.append(int(y.item()))

    y_true = np.array(labels, dtype=np.int32)
    y_probs = np.array(probs, dtype=np.float32)
    return y_true, y_probs, compute_all_metrics(y_true, y_probs)


def plot_three_curves(results_dict: dict, out_png: Path):
    """
    Plots side-by-side AUROC (left) and AUPRC (right) with 3 colors:
    - Red: Full Architecture (BiLSTM + GAT)
    - Blue: First Ablation (GAT-Only)
    - Yellow/Gold: Second Ablation (BiLSTM-Only)
    """
    plt.rcParams.update({"font.size": 11, "font.family": "sans-serif"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.8), dpi=300)

    styles = [
        ("Full Architecture (BiLSTM + GAT)", "#d62728", 2.4),  # Red
        ("First Ablation (GAT-Only)",        "#1f77b4", 2.2),  # Blue
        ("Second Ablation (BiLSTM-Only)",    "#eab308", 2.2),  # Vivid Yellow/Gold
    ]

    for name, color, lw in styles:
        y_true, y_probs, m = results_dict[name]
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        prec, rec, _ = precision_recall_curve(y_true, y_probs)

        ax1.plot(
            fpr, tpr, color=color, lw=lw,
            label=f"{name} (AUROC = {m['auc']:.4f})"
        )
        ax2.plot(
            rec, prec, color=color, lw=lw,
            label=f"{name} (AUPRC = {m['auprc']:.4f})"
        )

    # Left Subplot Formatting (AUROC)
    ax1.plot([0, 1], [0, 1], color="navy", lw=1.2, linestyle="--", alpha=0.6)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.02])
    ax1.set_xlabel("False Positive Rate", fontweight="bold")
    ax1.set_ylabel("True Positive Rate", fontweight="bold")
    ax1.set_title("Receiver Operating Characteristic (ROC) Curve", fontweight="bold", pad=10)
    ax1.legend(loc="lower right", frameon=True, fontsize=9.5)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Right Subplot Formatting (AUPRC)
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
    parser = argparse.ArgumentParser(description="Run GAT-Only and BiLSTM-Only Ablations + Plot 3-Color Curves")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs per fold (default: 1)")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds (default: 5)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Initial learning rate (default: 5e-5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-train", action="store_true", help="Skip training if ablation .pt files already exist")
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

    gat_pt = base_dir / "model_gat_only.pt"
    bilstm_pt = base_dir / "model_bilstm_only.pt"

    if not (args.skip_train and gat_pt.exists()):
        train_cv_gat_only(full_train_ds, skf, all_labels, args, prot_dim, base_dir, device)
    extractor.unload_from_gpu()

    if not (args.skip_train and bilstm_pt.exists()):
        train_cv_bilstm_only(full_train_ds, skf, all_labels, args, base_dir, device)

    # Evaluate all 3 models on the independent test set (all.test.fasta)
    criterion_cls = LabelSmoothedFocalLoss(alpha=0.5, gamma=1.0, smoothing=0.1)
    test_indices = list(range(len(test_ds)))

    # 1. Full Model (BiLSTM + GAT)
    y_full, probs_full, m_full = get_full_model_test_predictions(base_dir, test_ds, prot_dim, device)

    # 2. First Ablation (GAT-Only)
    gat_model = GATOnlyAblationModel(prot_dim=prot_dim, hidden_dim=192).to(device)
    gat_model.load_state_dict(torch.load(gat_pt, map_location=device, weights_only=True), strict=True)
    y_gat, probs_gat, m_gat = predict_gat_only(gat_model, test_ds, test_indices, criterion_cls, device)

    # 3. Second Ablation (BiLSTM-Only)
    bilstm_model = BiLSTMOnlyAblationModel(desc_dim=227, hidden_dim=192).to(device)
    bilstm_model.load_state_dict(torch.load(bilstm_pt, map_location=device, weights_only=True), strict=True)
    y_bilstm, probs_bilstm, m_bilstm = predict_bilstm_only(bilstm_model, test_ds, test_indices, criterion_cls, device)

    # Print & Log Comprehensive Scoring Table
    header = (
        f"\n{'=' * 98}\n"
        f"{'Model Variant':<34} | {'ACC':>7} | {'Prec':>7} | {'Recall':>7} | {'F1':>7} | {'AUROC':>7} | {'AUPRC':>7} | {'MCC':>7}\n"
        f"{'-' * 98}"
    )
    rows = [
        ("Second Ablation (BiLSTM-Only)",    m_bilstm),
        ("First Ablation (GAT-Only)",        m_gat),
        ("Full Architecture (BiLSTM + GAT)", m_full),
    ]
    lines = [header]
    for title, m in rows:
        lines.append(
            f"{title:<34} | {m['acc']:7.4f} | {m['prec']:7.4f} | {m['recall']:7.4f} | "
            f"{m['f1']:7.4f} | {m['auc']:7.4f} | {m['auprc']:7.4f} | {m['mcc']:7.4f}"
        )
    lines.append("=" * 98)
    table_output = "\n".join(lines)
    print(table_output)

    with open(base_dir / "ablation_test.log", "a") as f:
        f.write(table_output + "\n")

    # Plot 3-Color Side-by-Side AUROC & AUPRC
    results_dict = {
        "Full Architecture (BiLSTM + GAT)": (y_full, probs_full, m_full),
        "First Ablation (GAT-Only)":        (y_gat, probs_gat, m_gat),
        "Second Ablation (BiLSTM-Only)":    (y_bilstm, probs_bilstm, m_bilstm),
    }
    out_png = base_dir / "ablation_auroc_auprc_comparison.png"
    plot_three_curves(results_dict, out_png)
    print(f"\nSaved 3-color side-by-side AUROC & AUPRC figure to: {out_png}")


if __name__ == "__main__":
    main()