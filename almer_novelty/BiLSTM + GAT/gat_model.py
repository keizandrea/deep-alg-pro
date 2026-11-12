import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv as PyGGATConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class FallbackGATConv(nn.Module):
    """
    Pure PyTorch Graph Attention Layer (Veličković et al.) used automatically
    if torch_geometric is not installed.
    """
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1, concat: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.bias = nn.Parameter(torch.Tensor(heads * out_channels if concat else out_channels))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        # Add self-loops so every residue attends to itself and its neighbors
        loop_idx = torch.arange(num_nodes, device=x.device).unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, loop_idx], dim=1)
        src, dst = edge_index[0], edge_index[1]

        # Project node features: [V, heads, out_channels]
        h = self.lin(x).view(num_nodes, self.heads, self.out_channels)

        # Compute attention coefficients e_ij
        alpha_src = (h * self.att_src).sum(dim=-1)  # [V, heads]
        alpha_dst = (h * self.att_dst).sum(dim=-1)  # [V, heads]
        e = self.leaky_relu(alpha_src[src] + alpha_dst[dst])  # [E, heads]

        # Numerically stable neighborhood softmax over destination nodes (dst)
        e_max = torch.zeros((num_nodes, self.heads), device=x.device, dtype=h.dtype)
        e_max.scatter_reduce_(0, dst.unsqueeze(-1).expand_as(e), e, reduce="amax", include_self=False)
        exp_e = torch.exp(e - e_max[dst])

        denom = torch.zeros((num_nodes, self.heads), device=x.device, dtype=h.dtype)
        denom.scatter_add_(0, dst.unsqueeze(-1).expand_as(exp_e), exp_e)
        alpha = exp_e / (denom[dst] + 1e-16)  # [E, heads]

        # Weighted message aggregation
        weighted_msg = h[src] * alpha.unsqueeze(-1)  # [E, heads, out_channels]
        out = torch.zeros((num_nodes, self.heads, self.out_channels), device=x.device, dtype=h.dtype)
        out.scatter_add_(
            0,
            dst.view(-1, 1, 1).expand_as(weighted_msg),
            weighted_msg,
        )

        if self.concat:
            out = out.view(num_nodes, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        return out + self.bias


def make_gat_conv(in_channels: int, out_channels: int, heads: int, concat: bool):
    if HAS_PYG:
        return PyGGATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=concat,
            add_self_loops=True,
        )
    return FallbackGATConv(
        in_channels=in_channels,
        out_channels=out_channels,
        heads=heads,
        concat=concat,
    )


class NodeAttentionPooling(nn.Module):
    """
    Eq. (9): Preprocesses and weights initial ProtT5 node features X
    using a two-layer fully connected network with ReLU activation.
    """
    def __init__(self, in_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.attn_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.attn_mlp(x)  # [V, 1]
        return x * scores          # [V, in_dim]


class GlobalAttentionReadout(nn.Module):
    """
    Eq. (13): Aggregates node-level features H_last [V, 192] into a
    graph-level structural vector F_g [1, 192] using ReLU + Softmax weights.
    """
    def __init__(self, node_dim: int = 192):
        super().__init__()
        self.global_pool = nn.Sequential(
            nn.Linear(node_dim, node_dim // 2),
            nn.ReLU(),
            nn.Linear(node_dim // 2, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # alpha_i = Softmax(GlobalPool(H(i))) across all V residues
        attn_logits = self.global_pool(h)               # [V, 1]
        alpha = F.softmax(attn_logits, dim=0)           # [V, 1]
        f_g = torch.sum(alpha * h, dim=0, keepdim=True) # [1, 192]
        return f_g


class ProtT5GATBranch(nn.Module):
    """
    Modified 2-Layer Graph Attention Structural Feature Extraction Module:
    - Input: ProtT5 node features X [V, in_dim] (default 1024) & edge_index E [2, num_edges]
    - GAT Layer 1: 4-Head GATConv + LayerNorm + ELU + Residual Connection
    - GAT Layer 2: 1-Head GATConv + LayerNorm + Residual Connection -> H_2 [V, 192]
    - Readout: Global Attention Pooling -> F_g [1, 192]
    """
    def __init__(
        self,
        in_dim: int = 1024,
        layer1_head_dim: int = 128,
        num_heads: int = 4,
        out_dim: int = 192,
    ):
        super().__init__()
        layer1_total_dim = layer1_head_dim * num_heads  # 128 * 4 = 512

        # Eq. (9): Node Attention Preprocessing (with ReLU)
        self.node_attention = NodeAttentionPooling(in_dim=in_dim)

        # GAT Layer 1: 4-Headed GAT + LayerNorm + ELU + Residual Connection
        self.gat1 = make_gat_conv(
            in_channels=in_dim,
            out_channels=layer1_head_dim,
            heads=num_heads,
            concat=True,
        )
        self.norm1 = nn.LayerNorm(layer1_total_dim)
        self.elu = nn.ELU()
        self.res_proj1 = nn.Linear(in_dim, layer1_total_dim)

        # GAT Layer 2: 1-Headed GAT + LayerNorm + Residual Connection (192-dim output)
        self.gat2 = make_gat_conv(
            in_channels=layer1_total_dim,
            out_channels=out_dim,
            heads=1,
            concat=False,
        )
        self.norm2 = nn.LayerNorm(out_dim)
        self.res_proj2 = nn.Linear(layer1_total_dim, out_dim)

        # Eq. (13): Global Attention Pooling (with ReLU)
        self.global_readout = GlobalAttentionReadout(node_dim=out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        """
        Returns:
            f_g: Graph-level structural feature [1, 192] for cross-modal fusion / classification
            h2:  Node-level structural feature [V, 192] for self-supervised reconstruction loss
        """
        # Step 1: Eq. (9) Attention-weighted node features
        x_hat = self.node_attention(x)  # [V, in_dim]

        # Step 2: GAT Layer 1 (4 heads, LayerNorm, ELU, Residual)
        h1_conv = self.gat1(x_hat, edge_index)                  # [V, 512]
        h1 = self.elu(self.norm1(h1_conv) + self.res_proj1(x_hat))

        # Step 3: GAT Layer 2 (1 head, LayerNorm, Projection Residual -> 192 dims)
        h2_conv = self.gat2(h1, edge_index)                     # [V, 192]
        h2 = self.norm2(h2_conv) + self.res_proj2(h1)           # [V, 192]

        # Step 4: Eq. (13) Global Attention Pooling -> [1, 192]
        f_g = self.global_readout(h2)

        return f_g, h2


class StandaloneGATClassifier(nn.Module):
    """
    Wraps ProtT5GATBranch with a ReLU + Dropout classification layer and an optional
    node reconstruction decoder so you can train/verify the GAT branch independently.
    """
    def __init__(self, in_dim: int = 1024, hidden_dim: int = 192, dropout: float = 0.2):
        super().__init__()
        self.gat_branch = ProtT5GATBranch(in_dim=in_dim, out_dim=hidden_dim)

        # Classification head using ReLU + Dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Self-supervised node reconstruction branch (Eq. 26: H_last [V, 192] -> X_tilde [V, in_dim])
        self.reconstructor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, in_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        f_g, h2 = self.gat_branch(x, edge_index)
        logits = self.classifier(f_g)         # [1, 1]
        probs = torch.sigmoid(logits)         # [1, 1]
        x_recon = self.reconstructor(h2)      # [V, in_dim]
        return probs, f_g, x_recon