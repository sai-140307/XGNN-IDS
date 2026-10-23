"""Feature scaling and conversion of WindowGraphs to (batched) tensors."""
from __future__ import annotations

import numpy as np
import torch

from ..data.features import Standardizer
from ..graph.builder import WindowGraph


class Preprocessor:
    """Holds the train-set scalers; turns WindowGraphs into model inputs."""

    def __init__(self, node_scaler: Standardizer, edge_scaler: Standardizer, use_node_stats: bool = True):
        self.node_scaler = node_scaler
        self.edge_scaler = edge_scaler
        self.use_node_stats = use_node_stats

    @classmethod
    def fit(cls, graphs: list[WindowGraph], use_node_stats: bool = True) -> "Preprocessor":
        xs = np.vstack([g.x for g in graphs])
        es = np.vstack([g.edge_attr for g in graphs])
        return cls(Standardizer().fit(xs), Standardizer().fit(es), use_node_stats)

    @property
    def n_node_features(self) -> int:
        return len(self.node_scaler.mean) if self.use_node_stats else 1

    @property
    def n_edge_features(self) -> int:
        return len(self.edge_scaler.mean)

    def node_matrix(self, g: WindowGraph) -> np.ndarray:
        if not self.use_node_stats:
            return np.ones((g.num_nodes, 1), np.float32)
        return self.node_scaler.transform(g.x)

    def edge_matrix(self, g: WindowGraph) -> np.ndarray:
        return self.edge_scaler.transform(g.edge_attr)

    def tensors(self, graphs: WindowGraph | list[WindowGraph]):
        """Return (x, edge_index, edge_attr, y) for one graph or a disjoint union of graphs."""
        if isinstance(graphs, WindowGraph):
            graphs = [graphs]
        xs, eis, eas, ys = [], [], [], []
        off = 0
        for g in graphs:
            xs.append(self.node_matrix(g))
            eis.append(g.edge_index + off)
            eas.append(self.edge_matrix(g))
            ys.append(g.y)
            off += g.num_nodes
        return (
            torch.from_numpy(np.vstack(xs)),
            torch.from_numpy(np.hstack(eis)).long(),
            torch.from_numpy(np.vstack(eas)),
            torch.from_numpy(np.concatenate(ys)).long(),
        )

    def to_dict(self) -> dict:
        return {"node_scaler": self.node_scaler.to_dict(), "edge_scaler": self.edge_scaler.to_dict(),
                "use_node_stats": self.use_node_stats}

    @classmethod
    def from_dict(cls, d: dict) -> "Preprocessor":
        return cls(Standardizer.from_dict(d["node_scaler"]), Standardizer.from_dict(d["edge_scaler"]),
                   d.get("use_node_stats", True))


def make_batches(graphs: list[WindowGraph], max_edges: int) -> list[list[WindowGraph]]:
    """Greedily pack consecutive windows into batches of at most ``max_edges`` flows."""
    batches, cur, size = [], [], 0
    for g in graphs:
        if cur and size + g.num_edges > max_edges:
            batches.append(cur)
            cur, size = [], 0
        cur.append(g)
        size += g.num_edges
    if cur:
        batches.append(cur)
    return batches
