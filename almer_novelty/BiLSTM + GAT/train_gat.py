import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from pathlib import Path

from prott5_dataset import (
    resolve_data_dir,
    ProtT5FeatureExtractor,
    AllergenProteinGraphDataset,
)
from gat_model import StandaloneGATClassifier


def evaluate(model, dataset, criterion, device):
    model.eval()
    accuracy = torchmetrics.Accuracy(task="binary").to(device)
    precision = torchmetrics.Precision(task="binary").to(device)
    recall = torchmetrics.Recall(task="binary").to(device)
    f1 = torchmetrics.F1Score(task="binary").to(device)
    auroc = torchmetrics.AUROC(task="binary").to(device)

    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for idx in range(len(dataset)):
            x, edge_index, y, _ = dataset[idx]
            x = x.to(device)
            edge_index = edge_index.to(device)
            y = y.to(device).view(1, 1)

            probs, _, x_recon = model(x, edge_index)
            cls_loss = criterion(probs, y)
            recon_loss = torch.mean(torch.abs(x_recon - x))  # Eq. (26) L1 reconstruction loss
            loss = cls_loss + 0.1 * recon_loss               # Eq. (27) lambda = 0.1

            total_loss += loss.item()
            all_preds.append(probs)
            all_labels.append(y.long())

    final_preds = torch.cat(all_preds, dim=0)
    final_labels = torch.cat(all_labels, dim=0)

    return {
        "loss": total_loss / max(len(dataset), 1),
        "acc": accuracy(final_preds, final_labels).item(),
        "prec": precision(final_preds, final_labels).item(),
        "recall": recall(final_preds, final_labels).item(),
        "f1": f1(final_preds, final_labels).item(),
        "auc": auroc(final_preds, final_labels).item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Train 2-Layer ProtT5 + GAT Branch")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Initial learning rate (paper: 5e-5)")
    parser.add_argument("--concat-3-layers", action="store_true", help="Concatenate last 3 ProtT5 layers (3072-dim)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = resolve_data_dir()
    train_fasta = data_dir / "all.train.fasta"
    test_fasta = data_dir / "all.test.fasta"

    extractor = ProtT5FeatureExtractor(
        device=device,
        concat_last_3_layers=args.concat_3_layers,
    )
    in_dim = 3072 if args.concat_3_layers else 1024

    train_dataset = AllergenProteinGraphDataset(train_fasta, extractor=extractor)
    test_dataset = AllergenProteinGraphDataset(test_fasta, extractor=extractor)

    model = StandaloneGATClassifier(in_dim=in_dim, hidden_dim=192).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCELoss().to(device)

    best_f1 = 0.0
    save_path = Path(__file__).resolve().parent / "prott5_gat_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for idx in range(len(train_dataset)):
            x, edge_index, y, _ = train_dataset[idx]
            x = x.to(device)
            edge_index = edge_index.to(device)
            y = y.to(device).view(1, 1)

            optimizer.zero_grad()
            probs, _, x_recon = model(x, edge_index)

            cls_loss = criterion(probs, y)
            recon_loss = torch.mean(torch.abs(x_recon - x))
            loss = cls_loss + 0.1 * recon_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Paper Sec 3.2: clip=0.5
            optimizer.step()

            running_loss += loss.item()

        metrics = evaluate(model, test_dataset, criterion, device)
        print(
            f"Epoch {epoch:03d} | Train Loss: {running_loss / len(train_dataset):.4f} | "
            f"Test Loss: {metrics['loss']:.4f} | ACC: {metrics['acc']:.4f} | "
            f"F1: {metrics['f1']:.4f} | Prec: {metrics['prec']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | AUC: {metrics['auc']:.4f}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), save_path)

    print(f"Training complete. Best Test F1: {best_f1:.4f} saved to {save_path}")


if __name__ == "__main__":
    main()