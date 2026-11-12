import argparse
import random
import shutil
import numpy as np
import torch
import torch.optim as optim
import torchmetrics
from pathlib import Path
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedKFold

from prott5_dataset import resolve_data_dir, ProtT5FeatureExtractor, AllergenProteinGraphDataset
from fusion_model import MultimodalAllergenModel, LabelSmoothedFocalLoss


def evaluate_model(model, dataset, criterion_cls, device):
    model.eval()
    accuracy = torchmetrics.Accuracy(task="binary").to(device)
    precision = torchmetrics.Precision(task="binary").to(device)
    recall = torchmetrics.Recall(task="binary").to(device)
    f1 = torchmetrics.F1Score(task="binary").to(device)
    auroc = torchmetrics.AUROC(task="binary").to(device)
    mcc = torchmetrics.MatthewsCorrCoef(task="binary").to(device)

    all_probs, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for idx in range(len(dataset)):
            x, edge_index, desc, y, _ = dataset[idx]
            x = x.to(device)
            edge_index = edge_index.to(device)
            desc = desc.to(device).unsqueeze(0)
            y = y.to(device).view(1, 1)

            logits, x_recon, _ = model(x, edge_index, desc)
            loss_cls = criterion_cls(logits, y)
            loss_recon = torch.mean(torch.abs(x_recon - x))
            loss = loss_cls + 0.1 * loss_recon

            total_loss += loss.item()
            all_probs.append(torch.sigmoid(logits))
            all_labels.append(y.long())

    preds = torch.cat(all_probs, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return {
        "loss": total_loss / max(len(dataset), 1),
        "acc": accuracy(preds, labels).item(),
        "f1": f1(preds, labels).item(),
        "prec": precision(preds, labels).item(),
        "recall": recall(preds, labels).item(),
        "auc": auroc(preds, labels).item(),
        "mcc": mcc(preds, labels).item(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="5-Fold Cross-Validation Training: ProtT5-GAT + PhysEvo-BiLSTM"
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs per fold (default: 1)")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds (paper: 5)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Initial learning rate (paper: 5e-5)")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Linear warmup epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible CV splits")
    parser.add_argument("--concat-3-layers", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

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
    all_labels = np.array([int(rec[2]) for rec in full_train_ds.records])

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    criterion_cls = LabelSmoothedFocalLoss(alpha=0.5, gamma=1.0, smoothing=0.1)

    use_amp = device.type == "cuda"
    fold_metrics_list = []
    best_overall_f1 = -1.0
    best_fold_idx = 1
    final_model_path = base_dir / "model.pt"
    valid_log_path = base_dir / "valid.log"

    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(np.zeros(len(all_labels)), all_labels), start=1):
        print(f"\n==================== Fold {fold_idx} / {args.folds} ====================")
        print(f"Train samples: {len(train_idx)} (80%) | Validation samples: {len(valid_idx)} (20%)")

        train_subset = Subset(full_train_ds, train_idx.tolist())
        valid_subset = Subset(full_train_ds, valid_idx.tolist())

        model = MultimodalAllergenModel(
            prot_dim=prot_dim, desc_dim=227, hidden_dim=192, dropout=0.2
        ).to(device)
        optimizer = optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.999)
        )

        # Configure LR scheduler based on total epochs
        if args.epochs > args.warmup_epochs:
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs
            )
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-6
            )
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs]
            )
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=max(args.epochs, 1), T_mult=1, eta_min=1e-6
            )

        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        best_fold_f1 = -1.0
        best_fold_metrics = None
        epochs_no_improve = 0
        fold_save_path = base_dir / f"model.fold{fold_idx}.pt"
        train_order = list(range(len(train_subset)))

        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            random.shuffle(train_order)

            for step, sub_idx in enumerate(train_order, start=1):
                x, edge_index, desc, y, _ = train_subset[sub_idx]
                x = x.to(device)
                edge_index = edge_index.to(device)
                desc = desc.to(device).unsqueeze(0)
                y = y.to(device).view(1, 1)

                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits, x_recon, _ = model(x, edge_index, desc)
                    loss_cls = criterion_cls(logits, y)
                    loss_recon = torch.mean(torch.abs(x_recon - x))
                    loss = loss_cls + 0.1 * loss_recon

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                if step % 500 == 0 or step == len(train_subset):
                    print(
                        f"  [Fold {fold_idx} | Epoch {epoch:02d} | {step}/{len(train_subset)}] "
                        f"Avg Train Loss: {running_loss / step:.4f}"
                    )

            scheduler.step()
            val_metrics = evaluate_model(model, valid_subset, criterion_cls, device)

            # After Fold 1 Epoch 1 has cached all 80% train + 20% val sequences, unload ProtT5 from VRAM
            if fold_idx == 1 and epoch == 1:
                extractor.unload_from_gpu()

            print(
                f"Fold {fold_idx} | Epoch {epoch:02d} | Train Loss: {running_loss / len(train_subset):.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | ACC: {val_metrics['acc']:.4f} | "
                f"F1: {val_metrics['f1']:.4f} | Prec: {val_metrics['prec']:.4f} | "
                f"Recall: {val_metrics['recall']:.4f} | AUC: {val_metrics['auc']:.4f} | "
                f"MCC: {val_metrics['mcc']:.4f}"
            )

            if val_metrics["f1"] > best_fold_f1:
                best_fold_f1 = val_metrics["f1"]
                best_fold_metrics = val_metrics
                epochs_no_improve = 0
                torch.save(model.state_dict(), fold_save_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(f"Early stopping on Fold {fold_idx} at epoch {epoch}.")
                    break

        fold_metrics_list.append(best_fold_metrics)
        fold_log_line = (
            f"[Fold {fold_idx}] val_loss: {best_fold_metrics['loss']:.4f} | "
            f"ACC: {best_fold_metrics['acc']:.6f} | F1: {best_fold_metrics['f1']:.6f} | "
            f"Prec: {best_fold_metrics['prec']:.6f} | Recall: {best_fold_metrics['recall']:.6f} | "
            f"AUC: {best_fold_metrics['auc']:.6f} | MCC: {best_fold_metrics['mcc']:.6f}"
        )
        print(fold_log_line)
        with open(valid_log_path, "a") as f:
            f.write(fold_log_line + "\n")

        if best_fold_f1 > best_overall_f1:
            best_overall_f1 = best_fold_f1
            best_fold_idx = fold_idx
            shutil.copyfile(fold_save_path, final_model_path)
            print(f"  --> Updated {final_model_path.name} with Fold {fold_idx} weights (F1: {best_overall_f1:.4f})")

    # Compute 5-Fold Cross-Validation Mean +/- Std across all metrics
    print("\n================ 5-Fold Cross-Validation Summary ================")
    summary_parts = []
    for key in ["loss", "acc", "f1", "prec", "recall", "auc", "mcc"]:
        vals = [m[key] for m in fold_metrics_list]
        summary_parts.append(f"{key.upper()}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
    summary_line = "5-Fold CV Mean | " + " | ".join(summary_parts)
    print(summary_line)
    with open(valid_log_path, "a") as f:
        f.write(summary_line + "\n")

    print(
        f"\nBest overall model (from Fold {best_fold_idx}, Val F1: {best_overall_f1:.4f}) "
        f"is ready at: {final_model_path}"
    )


if __name__ == "__main__":
    main()