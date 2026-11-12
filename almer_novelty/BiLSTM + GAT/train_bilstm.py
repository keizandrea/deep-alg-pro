import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchmetrics
from pathlib import Path
from Bio import SeqIO

from prott5_dataset import resolve_data_dir
from phys_evo_features import extract_combined_phys_evo_vector
from bilstm_model import StandaloneBiLSTMClassifier


class PhysEvoFastaDataset(Dataset):
    def __init__(self, fasta_path: Path, max_len: int = 1000):
        self.features = []
        self.labels = []
        self.ids = []

        for record in SeqIO.parse(str(fasta_path), "fasta"):
            seq_str = str(record.seq).strip()
            if 0 < len(seq_str) <= max_len:
                feat = extract_combined_phys_evo_vector(seq_str)
                label = 1.0 if record.id.startswith("allergen") else 0.0
                self.features.append(feat)
                self.labels.append(torch.tensor([label], dtype=torch.float32))
                self.ids.append(record.id)

        print(f"Extracted 227-dim Phys+Evo features for {len(self.features)} sequences from {fasta_path.name}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]


def evaluate(model, loader, criterion, device):
    model.eval()
    accuracy = torchmetrics.Accuracy(task="binary").to(device)
    precision = torchmetrics.Precision(task="binary").to(device)
    recall = torchmetrics.Recall(task="binary").to(device)
    f1 = torchmetrics.F1Score(task="binary").to(device)
    auroc = torchmetrics.AUROC(task="binary").to(device)

    all_preds, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for p, y in loader:
            p, y = p.to(device), y.to(device)
            probs, _ = model(p)
            loss = criterion(probs, y)
            total_loss += loss.item()
            all_preds.append(probs)
            all_labels.append(y.long())

    final_preds = torch.cat(all_preds, dim=0)
    final_labels = torch.cat(all_labels, dim=0)

    return {
        "loss": total_loss / max(len(loader), 1),
        "acc": accuracy(final_preds, final_labels).item(),
        "prec": precision(final_preds, final_labels).item(),
        "recall": recall(final_preds, final_labels).item(),
        "f1": f1(final_preds, final_labels).item(),
        "auc": auroc(final_preds, final_labels).item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Bi-LSTM Physicochemical + Evolutionary Branch")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = resolve_data_dir()

    train_ds = PhysEvoFastaDataset(data_dir / "all.train.fasta")
    test_ds = PhysEvoFastaDataset(data_dir / "all.test.fasta")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = StandaloneBiLSTMClassifier(in_features=227, out_dim=192).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCELoss().to(device)

    best_f1 = 0.0
    save_path = Path(__file__).resolve().parent / "bilstm_physevo_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for p, y in train_loader:
            p, y = p.to(device), y.to(device)
            optimizer.zero_grad()
            probs, _ = model(p)
            loss = criterion(probs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            train_loss += loss.item()

        metrics = evaluate(model, test_loader, criterion, device)
        print(
            f"Epoch {epoch:03d} | Train Loss: {train_loss / len(train_loader):.4f} | "
            f"Test ACC: {metrics['acc']:.4f} | F1: {metrics['f1']:.4f} | "
            f"Prec: {metrics['prec']:.4f} | Recall: {metrics['recall']:.4f} | AUC: {metrics['auc']:.4f}"
        )
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), save_path)

    print(f"Best Bi-LSTM Test F1: {best_f1:.4f} saved to {save_path}")


if __name__ == "__main__":
    main()