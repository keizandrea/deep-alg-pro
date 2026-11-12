import torch
import torch.nn as nn


class BiLSTMPhysEvoBranch(nn.Module):
    """
    Bi-LSTM Physicochemical & Evolutionary Feature Mining Module (Section 2.2, Eqs. 1-8).
    
    Architecture:
    1. Input: Combined Physicochemical (CTD+AAC) & Evolutionary (PSSM) vector P of shape [B, in_features]
       (default in_features = 227; also supports 75 if using raw 75-dim descriptors).
    2. Dimension Expansion: Expands P to P' of shape [B, in_features, 1] for temporal step modeling.
    3. 2-Layer Bidirectional LSTM (Eqs. 1-4):
       - hidden_size = 64 per direction -> 128-dim concatenated forward/backward hidden state h_t^2
       - Full temporal feature matrix H_2 has shape [B, in_features, 128]
    4. Pooling, Flattening, LayerNorm & FC (Eqs. 5-7):
       - AdaptiveMaxPool1d + Flatten + LayerNorm + FC -> H_fc of shape [B, 96]
    5. Residual Concatenation (Eq. 8):
       - F_p = H_fc (+) Proj(P) -> [B, 96] (+) [B, 96] = [B, 192]
       Matches the exact 192-dim input required by the Bidirectional Cross-Modal Attention Fusion Module.
    """
    def __init__(
        self,
        in_features: int = 227,
        lstm_hidden: int = 64,
        num_layers: int = 2,
        pooled_steps: int = 16,
        out_dim: int = 192,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_dim = out_dim
        half_out = out_dim // 2  # 96 dims for H_fc and 96 dims for Proj(P) -> 192 dims concatenated

        # Step 1: Feature expansion projection from scalar descriptor step (1 -> 16) for richer LSTM input
        self.input_expand = nn.Linear(1, 16)

        # Step 2: 2-Layer Bidirectional LSTM (Eqs. 1-4)
        # Forward + Backward hidden states concatenate to 2 * 64 = 128 dims (H_2 in R^{B x L x 128})
        self.bilstm = nn.LSTM(
            input_size=16,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Step 3: Eq. (5) AdaptiveMaxPool1d along the feature-sequence dimension
        # Permutes H_2 to [B, 128, L], pools to [B, 128, pooled_steps], and flattens (Eq. 6)
        self.adaptive_pool = nn.AdaptiveMaxPool1d(output_size=pooled_steps)
        flat_dim = (lstm_hidden * 2) * pooled_steps  # 128 * 16 = 2048

        # Step 4: Eq. (7) H_fc = FC(LayerNorm(H_flat))
        self.layer_norm = nn.LayerNorm(flat_dim)
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, half_out),
            nn.ReLU(),
        )

        # Step 5: Eq. (8) Proj(P) for residual concatenation: F_p = H_fc (+) Proj(P) -> [B, 192]
        self.orig_proj = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, half_out),
            nn.ReLU(),
        )

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """
        Args:
            p: Tensor of shape [B, in_features] (or [in_features] for unbatched single sample)
        Returns:
            f_p: Temporally enhanced physicochemical + evolutionary feature [B, 192]
        """
        if p.dim() == 1:
            p = p.unsqueeze(0)  # [1, in_features]

        # Expand to sequence of descriptor steps: [B, in_features, 1] -> [B, in_features, 16]
        p_prime = self.input_expand(p.unsqueeze(-1))

        # Eqs. (1)-(4): 2-Layer Bi-LSTM -> H_2 of shape [B, in_features, 128]
        h2, _ = self.bilstm(p_prime)

        # Eq. (5): AdaptiveMaxPool1d over sequence dimension -> [B, 128, pooled_steps]
        h_pool = self.adaptive_pool(h2.permute(0, 2, 1))

        # Eq. (6): Flatten -> [B, 128 * pooled_steps]
        h_flat = torch.flatten(h_pool, start_dim=1)

        # Eq. (7): H_fc = FC(LayerNorm(H_flat)) -> [B, 96]
        h_fc = self.fc(self.layer_norm(h_flat))

        # Eq. (8): Residual Concatenation F_p = [H_fc (+) Proj(P)] -> [B, 192]
        p_proj = self.orig_proj(p)
        f_p = torch.cat([h_fc, p_proj], dim=-1)  # [B, 192]

        return f_p


class StandaloneBiLSTMClassifier(nn.Module):
    """
    Wraps BiLSTMPhysEvoBranch with a classification head so you can benchmark
    the Bi-LSTM module independently (like Table 3 Ablation Row 1) before fusing with GAT.
    """
    def __init__(self, in_features: int = 227, out_dim: int = 192, dropout: float = 0.2):
        super().__init__()
        self.bilstm_branch = BiLSTMPhysEvoBranch(in_features=in_features, out_dim=out_dim, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(out_dim // 2, 1),
        )

    def forward(self, p: torch.Tensor):
        f_p = self.bilstm_branch(p)           # [B, 192]
        probs = torch.sigmoid(self.classifier(f_p))  # [B, 1]
        return probs, f_p