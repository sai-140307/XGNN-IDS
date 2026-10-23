"""Edge-level GNN classifiers implemented in plain PyTorch.

All models share one signature::

    logits = model(x, edge_index, edge_attr, edge_weight=None)   # [E, C]

* ``x``           [N, F_node] node features
* ``edge_index``  [2, E] directed flows (src -> dst)
* ``edge_attr``   [E, F_edge] flow features
* ``edge_weight`` optional [E] soft mask in [0, 1] used by GNNExplainer to
  switch message passing over individual flows on/off.

Models
------
FlowMLP      - no graph: classifies each flow from its own features (ablation).
GCN          - Kipf & Welling GCN on the symmetrised host graph, edge head.
GAT          - GATv2-style attention with edge features in the attention score.
EGraphSAGE   - direction-aware, edge-featured GraphSAGE (E-GraphSAGE family):
               separate aggregation of inbound and outbound flows, so the
               model sees fan-in (DDoS victims) and fan-out (scanners) apart.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def scatter_mean(msg: torch.Tensor, index: torch.Tensor, n: int, weight: torch.Tensor | None = None) -> torch.Tensor:
    """Mean of ``msg`` rows grouped by ``index``; ``weight`` down-weights rows (masking)."""
    if weight is not None:
        msg = msg * weight.view(-1, *([1] * (msg.dim() - 1)))
    out = msg.new_zeros((n,) + tuple(msg.shape[1:]))
    out.index_add_(0, index, msg)
    cnt = msg.new_zeros(n).index_add_(0, index, torch.ones_like(index, dtype=msg.dtype))
    return out / cnt.clamp(min=1).view(-1, *([1] * (msg.dim() - 1)))


def scatter_softmax(score: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """Softmax of ``score`` [E, H] over groups of rows sharing the same ``index``."""
    h = score.size(1)
    with torch.no_grad():
        mx = torch.full((n, h), float("-inf"), dtype=score.dtype, device=score.device)
        mx = mx.scatter_reduce(0, index.view(-1, 1).expand(-1, h), score, reduce="amax", include_self=True)
    ex = torch.exp(score - mx[index])
    den = score.new_zeros((n, h)).index_add_(0, index, ex)
    return ex / den[index].clamp(min=1e-16)


class EdgeHead(nn.Module):
    """Classify a flow from [h_src || h_dst || e_flow]."""

    def __init__(self, d: int, n_cls: int, dropout: float):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, n_cls))

    def forward(self, h, edge_index, e):
        return self.mlp(torch.cat([h[edge_index[0]], h[edge_index[1]], e], dim=1))


class FlowMLP(nn.Module):
    def __init__(self, n_node: int, n_edge: int, n_cls: int, hidden: int = 64, dropout: float = 0.1, **_):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_edge, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_cls),
        )

    def forward(self, x, edge_index, edge_attr, edge_weight=None):
        return self.net(edge_attr)


# ----------------------------------------------------------------- E-GraphSAGE


class EdgeSAGELayer(nn.Module):
    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.msg_in = nn.Linear(2 * d, d)    # message from u for flow u -> v (received by v)
        self.msg_out = nn.Linear(2 * d, d)   # message from w for flow v -> w (received by v)
        self.update = nn.Linear(3 * d, d)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, edge_index, e, w=None):
        src, dst = edge_index
        n = h.size(0)
        a_in = scatter_mean(F.relu(self.msg_in(torch.cat([h[src], e], 1))), dst, n, w)
        a_out = scatter_mean(F.relu(self.msg_out(torch.cat([h[dst], e], 1))), src, n, w)
        upd = F.relu(self.update(torch.cat([h, a_in, a_out], 1)))
        return self.norm(h + self.drop(upd))


class EGraphSAGE(nn.Module):
    def __init__(self, n_node: int, n_edge: int, n_cls: int, hidden: int = 64, layers: int = 2,
                 dropout: float = 0.1, **_):
        super().__init__()
        self.node_in = nn.Linear(n_node, hidden)
        self.edge_in = nn.Linear(n_edge, hidden)
        self.layers = nn.ModuleList([EdgeSAGELayer(hidden, dropout) for _ in range(layers)])
        self.head = EdgeHead(hidden, n_cls, dropout)

    def forward(self, x, edge_index, edge_attr, edge_weight=None):
        h = F.relu(self.node_in(x))
        e = F.relu(self.edge_in(edge_attr))
        for layer in self.layers:
            h = layer(h, edge_index, e, edge_weight)
        return self.head(h, edge_index, e)


# ----------------------------------------------------------------- GCN


def _gcn_norm(edge_index, n, w):
    src, dst = edge_index
    loop = torch.arange(n, device=src.device)
    s = torch.cat([src, dst, loop])
    d = torch.cat([dst, src, loop])
    ones = torch.ones(src.numel(), device=src.device)
    ww = torch.cat([w if w is not None else ones, w if w is not None else ones, torch.ones(n, device=src.device)])
    deg = torch.zeros(n, device=src.device).index_add_(0, d, ww)
    norm = ww * deg[s].rsqrt() * deg[d].rsqrt()
    return s, d, norm


class GCNLayer(nn.Module):
    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.lin = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, s, d, norm):
        hw = self.lin(h)
        out = torch.zeros_like(hw).index_add_(0, d, hw[s] * norm.unsqueeze(1))
        return self.norm(h + self.drop(F.relu(out)))


class GCN(nn.Module):
    """GCN over hosts. Edge features enter through per-node mean of in/out flow features."""

    def __init__(self, n_node: int, n_edge: int, n_cls: int, hidden: int = 64, layers: int = 2,
                 dropout: float = 0.1, **_):
        super().__init__()
        self.node_in = nn.Linear(n_node + 2 * n_edge, hidden)
        self.edge_in = nn.Linear(n_edge, hidden)
        self.layers = nn.ModuleList([GCNLayer(hidden, dropout) for _ in range(layers)])
        self.head = EdgeHead(hidden, n_cls, dropout)

    def forward(self, x, edge_index, edge_attr, edge_weight=None):
        n = x.size(0)
        src, dst = edge_index
        agg_in = scatter_mean(edge_attr, dst, n, edge_weight)
        agg_out = scatter_mean(edge_attr, src, n, edge_weight)
        h = F.relu(self.node_in(torch.cat([x, agg_in, agg_out], 1)))
        s, d, norm = _gcn_norm(edge_index, n, edge_weight)
        for layer in self.layers:
            h = layer(h, s, d, norm)
        return self.head(h, edge_index, F.relu(self.edge_in(edge_attr)))


# ----------------------------------------------------------------- GAT


class EdgeGATLayer(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        assert d % heads == 0, "hidden size must be divisible by heads"
        self.h, self.dh = heads, d // heads
        self.w_src = nn.Linear(d, d)
        self.w_dst = nn.Linear(d, d)
        self.w_edge = nn.Linear(d, d, bias=False)
        self.att = nn.Parameter(torch.empty(heads, self.dh))
        nn.init.xavier_uniform_(self.att)
        self.out = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)
        self.last_alpha: torch.Tensor | None = None  # kept for attention-based inspection

    def forward(self, h, edge_index, e, w=None):
        n, d = h.shape
        src, dst = edge_index
        loop = torch.arange(n, device=h.device)
        s = torch.cat([src, dst, loop])               # both directions + self loops
        t = torch.cat([dst, src, loop])
        ee = torch.cat([e, e, e.new_zeros(n, d)])
        hs = self.w_src(h)[s].view(-1, self.h, self.dh)
        ht = self.w_dst(h)[t].view(-1, self.h, self.dh)
        he = self.w_edge(ee).view(-1, self.h, self.dh)
        score = (F.leaky_relu(hs + ht + he, 0.2) * self.att).sum(-1)
        alpha = scatter_softmax(score, t, n)
        if w is not None:
            alpha = alpha * torch.cat([w, w, torch.ones(n, device=h.device)]).unsqueeze(1)
        self.last_alpha = alpha.detach()
        msg = (hs + he) * self.drop(alpha).unsqueeze(-1)
        agg = h.new_zeros(n, self.h, self.dh).index_add_(0, t, msg).reshape(n, d)
        return self.norm(h + self.drop(F.elu(self.out(agg))))


class GAT(nn.Module):
    def __init__(self, n_node: int, n_edge: int, n_cls: int, hidden: int = 64, layers: int = 2, heads: int = 4,
                 dropout: float = 0.1, **_):
        super().__init__()
        self.node_in = nn.Linear(n_node, hidden)
        self.edge_in = nn.Linear(n_edge, hidden)
        self.layers = nn.ModuleList([EdgeGATLayer(hidden, heads, dropout) for _ in range(layers)])
        self.head = EdgeHead(hidden, n_cls, dropout)

    def forward(self, x, edge_index, edge_attr, edge_weight=None):
        h = F.relu(self.node_in(x))
        e = F.relu(self.edge_in(edge_attr))
        for layer in self.layers:
            h = layer(h, edge_index, e, edge_weight)
        return self.head(h, edge_index, e)


MODEL_REGISTRY = {
    "mlp": FlowMLP,
    "gcn": GCN,
    "gat": GAT,
    "egraphsage": EGraphSAGE,
}

MODEL_DISPLAY = {
    "mlp": "Flow-MLP (no graph)",
    "gcn": "GCN",
    "gat": "GAT (edge-attention)",
    "egraphsage": "E-GraphSAGE (directed)",
    "egraphsage_nostats": "E-GraphSAGE (no node stats)",
}


def build_model(name: str, n_node: int, n_edge: int, n_cls: int, **hparams) -> nn.Module:
    base = name.replace("_nostats", "")
    if base not in MODEL_REGISTRY:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[base](n_node, n_edge, n_cls, **hparams)


def num_hops(name: str, layers: int) -> int:
    """Receptive field (in hops) needed for exact explanation subgraphs."""
    return 0 if name.startswith("mlp") else layers + 1


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
