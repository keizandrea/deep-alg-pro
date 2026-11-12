import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import (
        GCNConv as PyGGCNConv,
        GATConv as PyGGATConv,
        TransformerConv as PyGTransformerConv,
    )
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


# Self-contained GNN operators (GCN, GAT, Graph Transformer)
class IsolatedGCNConv(nn.Module):
    # Spectral Graph Convolutional Network layer (Kipf & Welling): H' = D^{-1/2} A_hat D^{-1/2} X W + b
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        loop_idx = torch.arange(num_nodes, device=x.device).unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, loop_idx], dim=1)
        src, dst = edge_index[0], edge_index[1]

        # Compute degree D_hat^{-1/2}
        deg = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
        deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
        norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]  # [E]

        h = self.lin(x)  # [V, out_channels]
        msg = h[src] * norm.unsqueeze(-1)

        out = torch.zeros_like(h)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        return out + self.bias


class IsolatedGATConv(nn.Module):
    # Graph Attention Network layer (Veličković et al.): e_ij = LeakyReLU(a_src^T W h_i + a_dst^T W h_j)
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1, concat: bool = True):
        super().__init__()
        self.heads = heads
        self.out_channels = out_channels
        self.concat = concat

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.bias = nn.Parameter(torch.zeros(heads * out_channels if concat else out_channels))
        self.leaky_relu = nn.LeakyReLU(0.2)

        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        loop_idx = torch.arange(num_nodes, device=x.device).unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, loop_idx], dim=1)
        src, dst = edge_index[0], edge_index[1]

        h = self.lin(x).view(num_nodes, self.heads, self.out_channels)
        alpha_src = (h * self.att_src).sum(dim=-1)
        alpha_dst = (h * self.att_dst).sum(dim=-1)
        e = self.leaky_relu(alpha_src[src] + alpha_dst[dst])

        e_max = torch.zeros((num_nodes, self.heads), device=x.device, dtype=h.dtype)
        e_max.scatter_reduce_(0, dst.unsqueeze(-1).expand_as(e), e, reduce="amax", include_self=False)
        exp_e = torch.exp(e - e_max[dst])

        denom = torch.zeros((num_nodes, self.heads), device=x.device, dtype=h.dtype)
        denom.scatter_add_(0, dst.unsqueeze(-1).expand_as(exp_e), exp_e)
        alpha = exp_e / (denom[dst] + 1e-16)

        msg = h[src] * alpha.unsqueeze(-1)
        out = torch.zeros((num_nodes, self.heads, self.out_channels), device=x.device, dtype=h.dtype)
        out.scatter_add_(0, dst.view(-1, 1, 1).expand_as(msg), msg)

        if self.concat:
            out = out.view(num_nodes, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        return out + self.bias


class IsolatedTransformerConv(nn.Module):
    # Graph Transformer layer (Shi et al. / UniMP):
    # Multi-head scaled dot-product attention over graph neighborhoods: alpha_ij = Softmax_j((Q_i * K_j) / sqrt(d_k))
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1, concat: bool = True):
        super().__init__()
        self.heads = heads
        self.out_channels = out_channels
        self.concat = concat
        self.scale = 1.0 / math.sqrt(out_channels)

        self.lin_q = nn.Linear(in_channels, heads * out_channels, bias=True)
        self.lin_k = nn.Linear(in_channels, heads * out_channels, bias=True)
        self.lin_v = nn.Linear(in_channels, heads * out_channels, bias=True)

        nn.init.xavier_uniform_(self.lin_q.weight)
        nn.init.xavier_uniform_(self.lin_k.weight)
        nn.init.xavier_uniform_(self.lin_v.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        loop_idx = torch.arange(num_nodes, device=x.device).unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, loop_idx], dim=1)
        src, dst = edge_index[0], edge_index[1]

        q = self.lin_q(x).view(num_nodes, self.heads, self.out_channels)
        k = self.lin_k(x).view(num_nodes, self.heads, self.out_channels)
        v = self.lin_v(x).view(num_nodes, self.heads, self.out_channels)

        # Scaled dot-product score along edges (src -> dst)
        e = (q[dst] * k[src]).sum(dim=-1) * self.scale  # [E, heads]

        e_max = torch.zeros((num_nodes, self.heads), device=x.device, dtype=q.dtype)
        e_max.scatter_reduce_(0, dst.unsqueeze(-1).expand_as(e), e, reduce="amax", include_self=False)
        exp_e = torch.exp(e - e_max[dst])

        denom = torch.zeros((num_nodes, self.heads), device=x.device, dtype=q.dtype)
        denom.scatter_add_(0, dst.unsqueeze(-1).expand_as(exp_e), exp_e)
        alpha = exp_e / (denom[dst] + 1e-16)

        msg = v[src] * alpha.unsqueeze(-1)
        out = torch.zeros((num_nodes, self.heads, self.out_channels), device=x.device, dtype=q.dtype)
        out.scatter_add_(0, dst.view(-1, 1, 1).expand_as(msg), msg)

        if self.concat:
            return out.view(num_nodes, self.heads * self.out_channels)
        return out.mean(dim=1)


def build_gnn_layer(gnn_type: str, in_dim: int, head_dim: int, heads: int, concat: bool):
    gnn_type = gnn_type.lower()
    total_out = head_dim * heads if concat else head_dim

    if gnn_type == "gcn":
        if HAS_PYG:
            return PyGGCNConv(in_channels=in_dim, out_channels=total_out, add_self_loops=True)
        return IsolatedGCNConv(in_channels=in_dim, out_channels=total_out)

    if gnn_type == "gat":
        if HAS_PYG:
            return PyGGATConv(
                in_channels=in_dim, out_channels=head_dim, heads=heads, concat=concat, add_self_loops=True
            )
        return IsolatedGATConv(in_channels=in_dim, out_channels=head_dim, heads=heads, concat=concat)

    if gnn_type in ("gt", "transformer"):
        if HAS_PYG:
            return PyGTransformerConv(
                in_channels=in_dim, out_channels=head_dim, heads=heads, concat=concat
            )
        return IsolatedTransformerConv(
            in_channels=in_dim, out_channels=head_dim, heads=heads, concat=concat
        )

    raise ValueError(f"Unknown gnn_type: {gnn_type}. Choose from 'gcn', 'gat', or 'gt'.")


# 2. Shared pre-pooling, global readout, unified GNN backbone & loss function
class NodeAttentionPooling(nn.Module):
    # Shared node feature attention weighting with ReLU.
    def __init__(self, in_dim: int = 1024, hidden_dim: int = 256):
        super().__init__()
        self.attn_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.attn_mlp(x)


class GlobalAttentionReadout(nn.Module):
    # Shared global graph readout with ReLU + Softmax.
    def __init__(self, node_dim: int = 192):
        super().__init__()
        self.global_pool = nn.Sequential(
            nn.Linear(node_dim, node_dim // 2),
            nn.ReLU(),
            nn.Linear(node_dim // 2, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        alpha = F.softmax(self.global_pool(h), dim=0)
        return torch.sum(alpha * h, dim=0, keepdim=True)


class ProtT5GNNClassifier(nn.Module):
    """
    Strictly fair 2-Layer GNN classifier for GCN, GAT, and Graph Transformer:
    - Shared NodeAttentionPooling (1024 -> 1024)
    - Layer 1: GNN (1024 -> 512) + LayerNorm(512) + ELU + Residual Proj1
    - Layer 2: GNN (512 -> 192)  + LayerNorm(192) + Residual Proj2
    - Shared GlobalAttentionReadout -> F_g [1, 192]
    - Shared Classifier (192 -> 192 -> ReLU -> Dropout(0.2) -> 1)
    - Shared Reconstructor (192 -> 384 -> ReLU -> 1024)
    """
    def __init__(
        self,
        gnn_type: str = "gat",
        prot_dim: int = 1024,
        layer1_head_dim: int = 128,
        layer1_heads: int = 4,
        hidden_dim: int = 192,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gnn_type = gnn_type.lower()
        layer1_total_dim = layer1_head_dim * layer1_heads  # 128 * 4 = 512

        self.node_attention = NodeAttentionPooling(in_dim=prot_dim)

        # GNN Layer 1 (1024 -> 512)
        self.gnn1 = build_gnn_layer(
            gnn_type=self.gnn_type,
            in_dim=prot_dim,
            head_dim=layer1_head_dim,
            heads=layer1_heads,
            concat=True,
        )
        self.norm1 = nn.LayerNorm(layer1_total_dim)
        self.elu = nn.ELU()
        self.res_proj1 = nn.Linear(prot_dim, layer1_total_dim)

        # GNN Layer 2 (512 -> 192)
        self.gnn2 = build_gnn_layer(
            gnn_type=self.gnn_type,
            in_dim=layer1_total_dim,
            head_dim=hidden_dim,
            heads=1,
            concat=False,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.res_proj2 = nn.Linear(layer1_total_dim, hidden_dim)

        self.global_readout = GlobalAttentionReadout(node_dim=hidden_dim)

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
        x_hat = self.node_attention(x)
        h1 = self.elu(self.norm1(self.gnn1(x_hat, edge_index)) + self.res_proj1(x_hat))
        h2 = self.norm2(self.gnn2(h1, edge_index)) + self.res_proj2(h1)
        f_g = self.global_readout(h2)
        logits = self.classifier(f_g)
        x_recon = self.reconstructor(h2)
        return logits, x_recon, f_g


class LabelSmoothedFocalLoss(nn.Module):
    # Label-Smoothed Focal Loss (alpha=0.5, gamma=1.0, eps=0.1).
    def __init__(self, alpha: float = 0.5, gamma: float = 1.0, smoothing: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits).clamp(min=1e-7, max=1.0 - 1e-7)
        y_hat = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        term_pos = y_hat * torch.log(p) * torch.pow(1.0 - p, self.gamma)
        term_neg = (1.0 - y_hat) * torch.log(1.0 - p) * torch.pow(p, self.gamma)
        return (-self.alpha * (term_pos + term_neg)).mean()