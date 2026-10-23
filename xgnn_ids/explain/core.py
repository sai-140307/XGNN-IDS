"""Shared machinery for explaining a single flow (edge) prediction.

For an edge-level GNN with L layers, a flow's prediction depends only on the
(L+1)-hop neighbourhood of its two endpoints. Every explainer therefore works
on that *computation subgraph*, which makes explanations both exact and fast.

Feature-level explainers work in an *explanation space* Z that covers the
flow's own features **and the graph context of its two endpoints** (the
per-window host statistics of the source and destination):

    Z = [ flow features (27) | source-host features (11) | destination-host features (11) ]

so an explanation can say "destination host: 251 distinct sources" as well as
"flow: no response packets".
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn.functional as F

from ..data.features import EDGE_FEATURE_LABELS, EDGE_FEATURE_NAMES, NODE_FEATURE_LABELS, NODE_FEATURE_NAMES
from ..graph.builder import WindowGraph
from ..training.data import Preprocessor


@dataclass
class ExplainContext:
    model: torch.nn.Module
    graph: WindowGraph
    prep: Preprocessor
    x: torch.Tensor           # [n, Fn] scaled node features of the subgraph
    edge_index: torch.Tensor  # [2, m] local indices
    edge_attr: torch.Tensor   # [m, Fe] scaled edge features
    target: int               # local index of the explained flow
    edge_id: int              # index of the flow inside the window graph
    sub_edges: np.ndarray     # [m] window-graph edge ids of the subgraph
    sub_nodes: np.ndarray     # [n] window-graph node ids of the subgraph
    cls: int                  # class being explained (default: predicted)
    probs: np.ndarray         # model probabilities for the target flow

    @property
    def m(self) -> int:
        return self.edge_index.shape[1]

    @property
    def src(self) -> int:
        return int(self.edge_index[0, self.target])

    @property
    def dst(self) -> int:
        return int(self.edge_index[1, self.target])

    @property
    def with_context(self) -> bool:
        return bool(self.prep.use_node_stats)

    @property
    def n_edge_feats(self) -> int:
        return self.edge_attr.size(1)

    @property
    def z0(self) -> np.ndarray:
        """Current point in explanation space."""
        parts = [self.edge_attr[self.target].numpy()]
        if self.with_context:
            parts += [self.x[self.src].numpy(), self.x[self.dst].numpy()]
        return np.concatenate(parts).astype(np.float32)


def feature_space(with_context: bool) -> tuple[list[str], list[str]]:
    """(keys, human labels) of the explanation space."""
    keys = list(EDGE_FEATURE_NAMES)
    labels = [f"Flow: {EDGE_FEATURE_LABELS[n].lower()}" for n in EDGE_FEATURE_NAMES]
    if with_context:
        keys += [f"src:{n}" for n in NODE_FEATURE_NAMES] + [f"dst:{n}" for n in NODE_FEATURE_NAMES]
        labels += [f"Source host: {NODE_FEATURE_LABELS[n].lower()}" for n in NODE_FEATURE_NAMES]
        labels += [f"Destination host: {NODE_FEATURE_LABELS[n].lower()}" for n in NODE_FEATURE_NAMES]
    return keys, labels


def computation_subgraph(graph: WindowGraph, edge_id: int, hops: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes within ``hops`` of the flow's endpoints and the edges induced on them."""
    src, dst = graph.edge_index
    node_mask = np.zeros(graph.num_nodes, bool)
    node_mask[[src[edge_id], dst[edge_id]]] = True
    if hops == 0:
        return np.flatnonzero(node_mask), np.array([edge_id])
    for _ in range(hops):
        em = node_mask[src] | node_mask[dst]
        node_mask[src[em]] = True
        node_mask[dst[em]] = True
    edges = np.flatnonzero(node_mask[src] & node_mask[dst])
    return np.flatnonzero(node_mask), edges


@torch.no_grad()
def build_context(model, prep: Preprocessor, graph: WindowGraph, edge_id: int, hops: int,
                  cls: int | None = None) -> ExplainContext:
    nodes, edges = computation_subgraph(graph, edge_id, hops)
    remap = -np.ones(graph.num_nodes, np.int64)
    remap[nodes] = np.arange(len(nodes))
    ei = remap[graph.edge_index[:, edges]]
    x = torch.from_numpy(prep.node_matrix(graph)[nodes])
    ea = torch.from_numpy(prep.edge_matrix(graph)[edges])
    target = int(np.flatnonzero(edges == edge_id)[0])
    model.eval()
    probs = F.softmax(model(x, torch.from_numpy(ei), ea), 1)[target].numpy()
    return ExplainContext(model, graph, prep, x, torch.from_numpy(ei), ea, target, edge_id, edges, nodes,
                          int(probs.argmax()) if cls is None else cls, probs)


def target_proba(ctx: ExplainContext, edge_attr: torch.Tensor | None = None,
                 edge_weight: torch.Tensor | None = None, x: torch.Tensor | None = None) -> np.ndarray:
    with torch.no_grad():
        logits = ctx.model(ctx.x if x is None else x, ctx.edge_index,
                           ctx.edge_attr if edge_attr is None else edge_attr, edge_weight)
        return F.softmax(logits[ctx.target], 0).numpy()


def _replicate(ctx: ExplainContext, b: int):
    n, m = ctx.x.size(0), ctx.m
    x = ctx.x.repeat(b, 1)
    offs = (torch.arange(b) * n).repeat_interleave(m)
    ei = ctx.edge_index.repeat(1, b) + offs.unsqueeze(0)
    ea = ctx.edge_attr.repeat(b, 1)
    tgt = torch.arange(b) * m + ctx.target
    src = torch.arange(b) * n + ctx.src
    dst = torch.arange(b) * n + ctx.dst
    return x, ei, ea, tgt, src, dst


def _place(ctx: ExplainContext, x, ea, tgt, src, dst, z: torch.Tensor):
    """Write explanation-space vectors z [B, D] into replicated inputs (differentiable)."""
    fe = ctx.n_edge_feats
    ea[tgt] = z[:, :fe]
    if z.size(1) > fe:
        fn = ctx.x.size(1)
        x[src] = z[:, fe:fe + fn]
        x[dst] = z[:, fe + fn:fe + 2 * fn]
    return x, ea


def batched_target_proba(ctx: ExplainContext, Z: np.ndarray, max_edges: int = 300_000) -> np.ndarray:
    """Probabilities for the target flow when its explanation-space vector is replaced by each row of Z.

    Evaluates B variants in one forward pass by replicating the computation
    subgraph B times (disjoint union) - this is what makes SHAP affordable.
    Z may have only the flow-feature columns (then host context is left unchanged).
    """
    Z = np.asarray(Z, np.float32)
    out = []
    chunk = max(1, max_edges // max(ctx.m, 1))
    with torch.no_grad():
        for i in range(0, len(Z), chunk):
            zb = torch.from_numpy(Z[i:i + chunk])
            x, ei, ea, tgt, src, dst = _replicate(ctx, len(zb))
            x, ea = _place(ctx, x, ea, tgt, src, dst, zb)
            out.append(F.softmax(ctx.model(x, ei, ea)[tgt], 1).numpy())
    return np.vstack(out)


def context_at(ctx: ExplainContext, z: np.ndarray) -> ExplainContext:
    """Copy of ctx whose target flow / endpoints take the explanation-space values z."""
    x, ea = ctx.x.clone(), ctx.edge_attr.clone()
    zt = torch.as_tensor(z, dtype=torch.float32)
    fe = ctx.n_edge_feats
    ea[ctx.target] = zt[:fe]
    if len(z) > fe:
        fn = x.size(1)
        x[ctx.src] = zt[fe:fe + fn]
        x[ctx.dst] = zt[fe + fn:fe + 2 * fn]
    new = replace(ctx, x=x, edge_attr=ea)
    new.probs = target_proba(new)
    return new


def log_odds(p: float | np.ndarray, eps: float = 1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p) - np.log1p(-p)
