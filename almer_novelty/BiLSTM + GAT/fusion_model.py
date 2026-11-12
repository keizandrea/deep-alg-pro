import torch
import torch.nn as nn
import torch.nn.functional as F

from gat_model import ProtT5GATBranch
from bilstm_model import BiLSTMPhysEvoBranch


class BidirectionalCrossModalAttention(nn.Module):
    """
    Bidirectional Cross-Modal Attention Fusion Module (Section 2.4, Eqs. 14-23).
    Includes:
    1. Feature Space Alignment Projection (Eqs. 14-15): F_hat_g = F_g W_g + b_g, F_hat_p = F_p W_p + b_p
    2. 4-Head Bidirectional Cross-Modal Attention (Eqs. 16-20) with scaling factor sqrt(48)
    3. Residual Connections & 384-dim Concatenation (Eqs. 21-23)
    """
    def __init__(self, dim: int = 192, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 192 / 4 = 48
        self.scale = self.head_dim ** -0.5  # 1 / sqrt(48)

        # Eqs. (14)-(15): Feature Space Alignment Projection Layers (W_g, b_g and W_p, b_p)
        self.proj_g = nn.Linear(dim, dim, bias=True)
        self.proj_p = nn.Linear(dim, dim, bias=True)

        # Eqs. (16)-(20): g -> p pathway (F_hat_g generates Q; F_hat_p generates K, V)
        self.wq_g = nn.Linear(dim, dim, bias=False)
        self.wk_p = nn.Linear(dim, dim, bias=False)
        self.wv_p = nn.Linear(dim, dim, bias=False)
        self.wo_g = nn.Linear(dim, dim, bias=False)

        # Reverse p -> g pathway (F_hat_p generates Q; F_hat_g generates K, V)
        self.wq_p = nn.Linear(dim, dim, bias=False)
        self.wk_g = nn.Linear(dim, dim, bias=False)
        self.wv_g = nn.Linear(dim, dim, bias=False)
        self.wo_p = nn.Linear(dim, dim, bias=False)

    def forward(self, f_g: torch.Tensor, f_p: torch.Tensor) -> torch.Tensor:
        batch_size = f_g.size(0)

        # Step 1: Eqs. (14)-(15) Feature Projection into unified space
        f_hat_g = self.proj_g(f_g)  # [B, 192]
        f_hat_p = self.proj_p(f_p)  # [B, 192]

        # Step 2: Eqs. (16)-(20) Graph -> Physicochemical interaction (Attn_{g->p})
        q_g = self.wq_g(f_hat_g).view(batch_size, self.num_heads, self.head_dim)
        k_p = self.wk_p(f_hat_p).view(batch_size, self.num_heads, self.head_dim)
        v_p = self.wv_p(f_hat_p).view(batch_size, self.num_heads, self.head_dim)

        # [B, 4, 48] x [B, 48, 4] -> [B, 4, 4] cross-head attention matrix scaled by sqrt(48)
        attn_weight_gp = F.softmax(torch.matmul(q_g, k_p.transpose(-1, -2)) * self.scale, dim=-1)
        attn_gp = torch.matmul(attn_weight_gp, v_p).reshape(batch_size, -1)
        attn_gp = self.wo_g(attn_gp)  # [B, 192]

        # Step 3: Physicochemical -> Graph interaction (Attn_{p->g})
        q_p = self.wq_p(f_hat_p).view(batch_size, self.num_heads, self.head_dim)
        k_g = self.wk_g(f_hat_g).view(batch_size, self.num_heads, self.head_dim)
        v_g = self.wv_g(f_hat_g).view(batch_size, self.num_heads, self.head_dim)

        attn_weight_pg = F.softmax(torch.matmul(q_p, k_g.transpose(-1, -2)) * self.scale, dim=-1)
        attn_pg = torch.matmul(attn_weight_pg, v_g).reshape(batch_size, -1)
        attn_pg = self.wo_p(attn_pg)  # [B, 192]

        # Step 4: Eqs. (21)-(23) Residual Connections & Channel Concatenation
        f_g_prime = f_hat_g + attn_gp  # [B, 192]
        f_p_prime = f_hat_p + attn_pg  # [B, 192]

        f_fusion = torch.cat([f_g_prime, f_p_prime], dim=-1)  # [B, 384]
        return f_fusion


class LabelSmoothedFocalLoss(nn.Module):
    """
    Section 2.5.1, Eqs. (24)-(25): Label-Smoothed Focal Loss (alpha=0.5, gamma=1.0, eps=0.1).
    """
    def __init__(self, alpha: float = 0.5, gamma: float = 1.0, smoothing: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits).clamp(min=1e-7, max=1.0 - 1e-7)
        # Eq. (25): y_hat_i = y_i * (1 - eps) + eps / 2
        y_hat = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing

        # Eq. (24): -alpha * sum_{i=1}^2 y_hat_i * log(p_i) * (1 - p_i)^gamma
        term_pos = y_hat * torch.log(p) * torch.pow(1.0 - p, self.gamma)
        term_neg = (1.0 - y_hat) * torch.log(1.0 - p) * torch.pow(p, self.gamma)
        loss = -self.alpha * (term_pos + term_neg)
        return loss.mean()


class MultimodalAllergenModel(nn.Module):
    def __init__(self, prot_dim: int = 1024, desc_dim: int = 227, hidden_dim: int = 192, dropout: float = 0.2):
        super().__init__()
        self.gat_branch = ProtT5GATBranch(in_dim=prot_dim, out_dim=hidden_dim)
        self.bilstm_branch = BiLSTMPhysEvoBranch(in_features=desc_dim, out_dim=hidden_dim, dropout=dropout)
        self.fusion_module = BidirectionalCrossModalAttention(dim=hidden_dim, num_heads=4)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.reconstructor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, prot_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, desc: torch.Tensor):
        f_g, h2 = self.gat_branch(x, edge_index)
        f_p = self.bilstm_branch(desc)
        f_fusion = self.fusion_module(f_g, f_p)
        logits = self.classifier(f_fusion)
        x_recon = self.reconstructor(h2)
        return logits, x_recon, f_fusion