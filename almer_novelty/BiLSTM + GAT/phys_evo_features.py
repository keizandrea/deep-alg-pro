import math
import numpy as np
import torch
from pathlib import Path
from Bio import SeqIO


# 20 Standard Amino Acids
AA_LIST = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: idx for idx, aa in enumerate(AA_LIST)}

# Standard BLOSUM62 Substitution Matrix (20x20 in AA_LIST order) for instant evolutionary profiling
# when external PSI-BLAST .pssm files are not provided.
BLOSUM62_MATRIX = np.array([
    # A   C   D   E   F   G   H   I   K   L   M   N   P   Q   R   S   T   V   W   Y
    [ 4,  0, -2, -1, -2,  0, -2, -1, -1, -1, -1, -2, -1, -1, -1,  1,  0,  0, -3, -2], # A
    [ 0,  9, -3, -4, -2, -3, -3, -1, -3, -1, -1, -3, -3, -3, -3, -1, -1, -1, -2, -2], # C
    [-2, -3,  6,  2, -3, -1, -1, -3, -1, -4, -3,  1, -1,  0, -2,  0, -1, -3, -4, -3], # D
    [-1, -4,  2,  5, -3, -2,  0, -3,  1, -3, -2,  0, -1,  2,  0,  0, -1, -2, -3, -2], # E
    [-2, -2, -3, -3,  6, -3, -1,  0, -3,  0,  0, -3, -4, -3, -3, -2, -2, -1,  1,  3], # F
    [ 0, -3, -1, -2, -3,  6, -2, -4, -2, -4, -3,  0, -2, -2, -2,  0, -2, -3, -2, -3], # G
    [-2, -3, -1,  0, -1, -2,  8, -3, -1, -3, -2,  1, -2,  0,  0, -1, -2, -3, -2,  2], # H
    [-1, -1, -3, -3,  0, -4, -3,  4, -3,  2,  1, -3, -3, -3, -3, -2, -1,  3, -3, -1], # I
    [-1, -3, -1,  1, -3, -2, -1, -3,  5, -2, -1,  0, -1,  1,  2,  0, -1, -2, -3, -2], # K
    [-1, -1, -4, -3,  0, -4, -3,  2, -2,  4,  2, -3, -3, -2, -2, -2, -1,  1, -2, -1], # L
    [-1, -1, -3, -2,  0, -3, -2,  1, -1,  2,  5, -2, -2,  0, -1, -1, -1,  1, -1, -1], # M
    [-2, -3,  1,  0, -3,  0,  1, -3,  0, -3, -2,  6, -2,  0,  0,  1,  0, -3, -4, -2], # N
    [-1, -3, -1, -1, -4, -2, -2, -3, -1, -3, -2, -2,  7, -1, -2, -1, -1, -2, -4, -3], # P
    [-1, -3,  0,  2, -3, -2,  0, -3,  1, -2,  0,  0, -1,  5,  1,  0, -1, -2, -2, -1], # Q
    [-1, -3, -2,  0, -3, -2,  0, -3,  2, -2, -1,  0, -2,  1,  5, -1, -1, -3, -3, -2], # R
    [ 1, -1,  0,  0, -2,  0, -1, -2,  0, -2, -1,  1, -1,  0, -1,  4,  1, -2, -3, -2], # S
    [ 0, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1,  0, -1, -1, -1,  1,  5,  0, -2, -2], # T
    [ 0, -1, -3, -2, -1, -3, -3,  3, -2,  1,  1, -3, -2, -2, -3, -2,  0,  4, -3, -1], # V
    [-3, -2, -4, -3,  1, -2, -2, -3, -3, -2, -1, -4, -4, -2, -3, -3, -2, -3, 11,  2], # W
    [-2, -2, -3, -2,  3, -3,  2, -1, -2, -1, -1, -2, -3, -1, -2, -2, -2, -1,  2,  7], # Y
], dtype=np.float32)

# Standard 7 Physicochemical Property Groupings for CTD (Dubchak / PROFEAT standard used in Yu et al. 2025)
CTD_GROUPS = {
    "hydrophobicity":        ("RKEDQN", "GASTPHY", "CLVIMFW"),
    "vdw_volume":            ("GASTPDC", "NVEQIL", "MHKFRYW"),
    "polarity":              ("LIFWCMVY", "PATGS", "HQRKNED"),
    "polarizability":        ("GASDT", "CPNVEQIL", "KMHFRYW"),
    "charge":                ("KR", "ANCQGHILMFPSTWYV", "DE"),
    "secondary_structure":   ("EALMQKRH", "VIYCWT", "GNPSD"),
    "solvent_accessibility": ("ALFCGIVW", "RKQEND", "MSPTHY"),
}


def compute_aac(seq: str) -> np.ndarray:
    """20-dim Amino Acid Composition (AAC)."""
    length = max(len(seq), 1)
    counts = np.zeros(20, dtype=np.float32)
    for ch in seq:
        if ch in AA_TO_IDX:
            counts[AA_TO_IDX[ch]] += 1.0
    return counts / length


def compute_ctd(seq: str) -> np.ndarray:
    """
    147-dim Physicochemical Composition-Transition-Distribution (CTD) vector:
    - Composition (C): 21 dims (7 properties x 3 groups)
    - Transition (T):  21 dims (7 properties x 3 group-pair transitions)
    - Distribution (D): 105 dims (7 properties x 3 groups x 5 quantiles [0%, 25%, 50%, 75%, 100%])
    """
    n = len(seq)
    if n == 0:
        return np.zeros(147, dtype=np.float32)

    c_feats, t_feats, d_feats = [], [], []

    for _, (g1, g2, g3) in CTD_GROUPS.items():
        # Map sequence to group indices 1, 2, 3
        mapped = []
        for ch in seq:
            if ch in g1:
                mapped.append(1)
            elif ch in g2:
                mapped.append(2)
            elif ch in g3:
                mapped.append(3)
        mapped = np.array(mapped, dtype=np.int32)
        L = len(mapped)
        if L == 0:
            c_feats.extend([0.0, 0.0, 0.0])
            t_feats.extend([0.0, 0.0, 0.0])
            d_feats.extend([0.0] * 15)
            continue

        # 1. Composition (3 dims)
        for g in (1, 2, 3):
            c_feats.append(np.sum(mapped == g) / L)

        # 2. Transition (3 dims: 1<->2, 1<->3, 2<->3)
        if L > 1:
            a, b = mapped[:-1], mapped[1:]
            t12 = np.sum(((a == 1) & (b == 2)) | ((a == 2) & (b == 1))) / (L - 1)
            t13 = np.sum(((a == 1) & (b == 3)) | ((a == 3) & (b == 1))) / (L - 1)
            t23 = np.sum(((a == 2) & (b == 3)) | ((a == 3) & (b == 2))) / (L - 1)
            t_feats.extend([t12, t13, t23])
        else:
            t_feats.extend([0.0, 0.0, 0.0])

        # 3. Distribution (3 groups x 5 cutoffs = 15 dims)
        for g in (1, 2, 3):
            positions = np.where(mapped == g)[0] + 1  # 1-based positions
            count_g = len(positions)
            if count_g == 0:
                d_feats.extend([0.0] * 5)
            else:
                for frac in (0.0, 0.25, 0.50, 0.75, 1.0):
                    idx = max(1, int(math.floor(count_g * frac))) if frac > 0 else 1
                    d_feats.append(positions[idx - 1] / L)

    return np.array(c_feats + t_feats + d_feats, dtype=np.float32)


def get_pssm_matrix(seq: str, pssm_file: Path = None) -> np.ndarray:
    """
    Returns an [L, 20] evolutionary scoring matrix.
    If an external PSI-BLAST .pssm file exists, parses it; otherwise uses
    the standardized BLOSUM62 evolutionary substitution matrix.
    """
    if pssm_file is not None and pssm_file.exists():
        rows = []
        with open(pssm_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 22 and parts[0].isdigit():
                    scores = [float(x) for x in parts[2:22]]
                    rows.append(scores)
        if len(rows) > 0:
            mat = np.array(rows, dtype=np.float32)
            return 1.0 / (1.0 + np.exp(-mat))  # Sigmoid-normalized PSSM

    # Standardized BLOSUM62 evolutionary profile [L, 20]
    indices = [AA_TO_IDX.get(ch, 0) for ch in seq]
    raw_blosum = BLOSUM62_MATRIX[indices]  # [L, 20]
    return 1.0 / (1.0 + np.exp(-raw_blosum))


def compute_evolutionary_features(seq: str, pssm_file: Path = None) -> np.ndarray:
    """
    Computes the most standardized evolutionary descriptors from Yu et al. (2025):
    1. AAC_PSSM (20 dims): Column-wise mean of evolutionary conservation scores.
    2. DP_PSSM Basic Split (40 dims): Mean positive & below-mean deviation scores per AA column.
    Total = 60 evolutionary dimensions.
    """
    pssm = get_pssm_matrix(seq, pssm_file)  # [L, 20] in (0, 1)
    # 1. AAC_PSSM (20-dim)
    aac_pssm = np.mean(pssm, axis=0)

    # 2. Centered positive/negative evolutionary deviation (DP_PSSM basic, 40-dim)
    centered = pssm - 0.5
    pos_pssm = np.mean(np.maximum(centered, 0.0), axis=0)
    neg_pssm = np.mean(np.minimum(centered, 0.0), axis=0)

    return np.concatenate([aac_pssm, pos_pssm, neg_pssm], axis=0).astype(np.float32)


def extract_combined_phys_evo_vector(seq: str, pssm_file: Path = None) -> torch.Tensor:
    """
    Combines:
    - Physicochemical CTD (147 dims) + AAC (20 dims) = 167 dims
    - Evolutionary AAC_PSSM (20 dims) + DP_PSSM_basic (40 dims) = 60 dims
    Total standardized input vector P: 227 dims.
    """
    seq_clean = seq.strip().upper()
    ctd_vec = compute_ctd(seq_clean)                        # 147-dim
    aac_vec = compute_aac(seq_clean)                        # 20-dim
    evo_vec = compute_evolutionary_features(seq_clean, pssm_file)  # 60-dim

    combined = np.concatenate([ctd_vec, aac_vec, evo_vec], axis=0) # 227-dim
    return torch.from_numpy(combined).float()