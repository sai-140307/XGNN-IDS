"""Shared machinery for explaining a single flow (edge) prediction.

For an edge-level GNN with L layers, a flow's prediction depends only on the
(L+1)-hop neighbourhood of its two endpoints. Every explainer therefore works
on that *computation subgraph*, which makes explanations both exact and fast.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

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
                 edge_weight: torch.Tensor | None = None) -> np.ndarray:
    with torch.no_grad():
        logits = ctx.model(ctx.x, ctx.edge_index, ctx.edge_attr if edge_attr is None else edge_attr, edge_weight)
        return F.softmax(logits[ctx.target], 0).numpy()


def _replicate(ctx: ExplainContext, b: int):
    n, m = ctx.x.size(0), ctx.m
    x = ctx.x.repeat(b, 1)
    offs = (torch.arange(b) * n).repeat_interleave(m)
    ei = ctx.edge_index.repeat(1, b) + offs.unsqueeze(0)
    ea = ctx.edge_attr.repeat(b, 1)
    tgt = torch.arange(b) * m + ctx.target
    return x, ei, ea, tgt


def batched_target_proba(ctx: ExplainContext, X: np.ndarray, max_edges: int = 300_000) -> np.ndarray:
    """Probabilities for the target flow when its feature vector is replaced by each row of X.

    Evaluates B variants in one forward pass by replicating the computation
    subgraph B times (disjoint union) - this is what makes SHAP affordable.
    """
    X = np.asarray(X, np.float32)
    out = []
    chunk = max(1, max_edges // max(ctx.m, 1))
    with torch.no_grad():
        for i in range(0, len(X), chunk):
            xb = torch.from_numpy(X[i:i + chunk])
            x, ei, ea, tgt = _replicate(ctx, len(xb))
            ea[tgt] = xb
            out.append(F.softmax(ctx.model(x, ei, ea)[tgt], 1).numpy())
    return np.vstack(out)
