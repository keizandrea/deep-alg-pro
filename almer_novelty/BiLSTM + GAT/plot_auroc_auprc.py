import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from Bio import SeqIO
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

from prott5_dataset import resolve_data_dir, ProtT5FeatureExtractor, AllergenProteinGraphDataset
from fusion_model import MultimodalAllergenModel


# ==============================================================================
# 1. Baseline DeepAlgPro (CNN + MHSA / convATTnet) Definition (from model.py)
# ==============================================================================
class selfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.key_layer = nn.Linear(input_size, hidden_size)
        self.query_layer = nn.Linear(input_size, hidden_size)
        self.value_layer = nn.Linear(input_size, hidden_size)

    def trans_to_multiple_heads(self, x):
        new_size = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        key = self.key_layer(x)
        query = self.query_layer(x)
        value = self.value_layer(x)

        key_heads = self.trans_to_multiple_heads(key)
        query_heads = self.trans_to_multiple_heads(query)
        value_heads = self.trans_to_multiple_heads(value)

        attention_scores = torch.matmul(query_heads, key_heads.permute(0, 1, 3, 2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = F.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_probs, value_heads)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_size = context.size()[:-2] + (self.all_head_size,)
        context = context.view(*new_size)
        return context


class convATTnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(21, 64, 16, stride=1)
        self.maxpool = nn.MaxPool1d(5, stride=5)
        self.dropout = nn.Dropout(p=0.1)
        self.selfattention = selfAttention(8, 12, 24)
        self.fc1 = nn.Linear(in_features=985 * 24, out_features=1)

    def forward(self, x):
        x = F.one_hot(x, num_classes=21).float()
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = x.permute(0, 2, 1)
        x = self.dropout(self.maxpool(x))
        x = self.selfattention(x)
        x = x.reshape(-1, 985 * 24)
        x = torch.sigmoid(self.fc1(x))
        return x


CODE_DICT = {
    'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
    'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20
}


def resolve_root_dir() -> Path:
    current_dir = Path(__file__).resolve().parent
    for parent in [current_dir.parent, current_dir.parent.parent]:
        if (parent / "model.pt").exists() and (parent / "data").exists():
            return parent
    return Path(r"D:\DeepAlgPro")


def get_cnn_mhsa_predictions(fasta_path: Path, model_pt_path: Path, device: torch.device):
    """
    Loads D:/DeepAlgPro/model.pt (CNN + MHSA) and extracts probabilities on all.test.fasta.
    """
    print(f"Loading CNN + MHSA baseline from: {model_pt_path}")
    sequences, labels = [], []
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        seq_str = str(record.seq).strip().upper()
        length = len(seq_str)
        if 0 < length <= 1000:
            padded = [0] * (1000 - length) + [CODE_DICT.get(a, 0) for a in seq_str]
            sequences.append(padded)
            labels.append(1 if record.id.startswith("allergen") else 0)

    x_tensor = torch.tensor(sequences, dtype=torch.long)
    y_array = np.array(labels, dtype=np.int32)

    loader = DataLoader(TensorDataset(x_tensor), batch_size=72, shuffle=False)
    model = convATTnet().to(device)
    state_dict = torch.load(model_pt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    probs = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            out = model(batch_x).view(-1)
            probs.append(out.cpu().numpy())

    return y_array, np.concatenate(probs, axis=0)


def get_bilstm_gat_predictions(base_dir: Path, fasta_path: Path, device: torch.device):
    """
    Loads predictions for our BiLSTM + GAT model from test_predictions.tsv if available,
    or runs BiLSTM + GAT/model.pt directly on all.test.fasta.
    """
    tsv_path = base_dir / "test_predictions.tsv"
    our_pt_path = base_dir / "model.pt"

    if tsv_path.exists():
        print(f"Reading cached BiLSTM + GAT predictions from: {tsv_path}")
        df = pd.read_csv(tsv_path, sep="\t")
        return df["true_label"].to_numpy(dtype=np.int32), df["score"].to_numpy(dtype=np.float32)

    print(f"Running BiLSTM + GAT inference from: {our_pt_path}")
    extractor = ProtT5FeatureExtractor(device=device, cache_dir=base_dir / "cache_prott5")
    test_ds = AllergenProteinGraphDataset(fasta_path, extractor=extractor)

    model = MultimodalAllergenModel(prot_dim=1024, desc_dim=227, hidden_dim=192).to(device)
    state_dict = torch.load(our_pt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    labels, probs = [], []
    with torch.no_grad():
        for idx in range(len(test_ds)):
            x, edge_index, desc, y, _ = test_ds[idx]
            logits, _, _ = model(
                x.to(device),
                edge_index.to(device),
                desc.to(device).unsqueeze(0),
            )
            probs.append(torch.sigmoid(logits).item())
            labels.append(int(y.item()))

    return np.array(labels, dtype=np.int32), np.array(probs, dtype=np.float32)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = Path(__file__).resolve().parent
    root_dir = resolve_root_dir()
    data_dir = resolve_data_dir()

    test_fasta = data_dir / "all.test.fasta"
    cnn_mhsa_pt = root_dir / "model.pt"

    # 1. Extract ground-truth labels and predicted probabilities for both models
    y_cnn, probs_cnn = get_cnn_mhsa_predictions(test_fasta, cnn_mhsa_pt, device)
    y_ours, probs_ours = get_bilstm_gat_predictions(base_dir, test_fasta, device)

    # 2. Compute ROC Curve and AUROC
    fpr_cnn, tpr_cnn, _ = roc_curve(y_cnn, probs_cnn)
    auroc_cnn = auc(fpr_cnn, tpr_cnn)

    fpr_ours, tpr_ours, _ = roc_curve(y_ours, probs_ours)
    auroc_ours = auc(fpr_ours, tpr_ours)

    # 3. Compute Precision-Recall Curve and AUPRC
    prec_cnn, rec_cnn, _ = precision_recall_curve(y_cnn, probs_cnn)
    auprc_cnn = average_precision_score(y_cnn, probs_cnn)

    prec_ours, rec_ours, _ = precision_recall_curve(y_ours, probs_ours)
    auprc_ours = average_precision_score(y_ours, probs_ours)

    # 4. Plot side-by-side AUROC (left) and AUPRC (right) with Red & Blue lines
    plt.rcParams.update({"font.size": 11, "font.family": "sans-serif"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    # Left Subplot: AUROC
    ax1.plot(
        fpr_ours, tpr_ours,
        color="#d62728", lw=2.2,
        label=f"BiLSTM + GAT (AUROC = {auroc_ours:.4f})"
    )
    ax1.plot(
        fpr_cnn, tpr_cnn,
        color="#1f77b4", lw=2.2,
        label=f"CNN + MHSA (DeepAlgPro) (AUROC = {auroc_cnn:.4f})"
    )
    ax1.plot([0, 1], [0, 1], color="navy", lw=1.2, linestyle="--", alpha=0.7)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.02])
    ax1.set_xlabel("False Positive Rate", fontweight="bold")
    ax1.set_ylabel("True Positive Rate", fontweight="bold")
    ax1.set_title("Receiver Operating Characteristic (ROC) Curve", fontweight="bold", pad=10)
    ax1.legend(loc="lower right", frameon=True, fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Right Subplot: AUPRC
    ax2.plot(
        rec_ours, prec_ours,
        color="#d62728", lw=2.2,
        label=f"BiLSTM + GAT (AUPRC = {auprc_ours:.4f})"
    )
    ax2.plot(
        rec_cnn, prec_cnn,
        color="#1f77b4", lw=2.2,
        label=f"CNN + MHSA (DeepAlgPro) (AUPRC = {auprc_cnn:.4f})"
    )
    ax2.plot([0, 1], [1, 0], color="navy", lw=1.2, linestyle="--", alpha=0.7)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.02])
    ax2.set_xlabel("Recall", fontweight="bold")
    ax2.set_ylabel("Precision", fontweight="bold")
    ax2.set_title("Precision-Recall (PR) Curve", fontweight="bold", pad=10)
    ax2.legend(loc="lower left", frameon=True, fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    out_png = base_dir / "auroc_auprc_comparison.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n=======================================================")
    print(f"CNN + MHSA (DeepAlgPro) | AUROC: {auroc_cnn:.4f} | AUPRC: {auprc_cnn:.4f}")
    print(f"BiLSTM + GAT            | AUROC: {auroc_ours:.4f} | AUPRC: {auprc_ours:.4f}")
    print(f"Saved side-by-side figure to: {out_png}")
    print(f"=======================================================")


if __name__ == "__main__":
    main()