"""Concept-drift monitoring with the Population Stability Index (PSI).

A deployed IDS silently degrades when traffic changes (new services, new
network, new attack tools). The monitor compares the feature distribution of
every incoming window against the training distribution:

    PSI = sum_b (p_b - q_b) * ln(p_b / q_b)

Rule of thumb: PSI < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 retrain.
"""
from __future__ import annotations

import numpy as np

from ..data.features import EDGE_FEATURE_NAMES

STABLE, WARN = 0.1, 0.25


class DriftMonitor:
    def __init__(self, edges: list[np.ndarray], ref: list[np.ndarray]):
        self.edges = edges
        self.ref = ref

    @classmethod
    def fit(cls, x: np.ndarray, n_bins: int = 10) -> "DriftMonitor":
        edges, ref = [], []
        for j in range(x.shape[1]):
            e = np.unique(np.quantile(x[:, j], np.linspace(0, 1, n_bins + 1)[1:-1]))
            edges.append(e)
            ref.append(cls._hist(x[:, j], e))
        return cls(edges, ref)

    @staticmethod
    def _hist(col: np.ndarray, e: np.ndarray) -> np.ndarray:
        counts = np.bincount(np.searchsorted(e, col, side="right"), minlength=len(e) + 1).astype(float)
        return (counts + 1e-4) / (counts.sum() + 1e-4 * len(counts))

    def psi(self, x: np.ndarray) -> tuple[float, dict[str, float]]:
        per = {}
        for j, name in enumerate(EDGE_FEATURE_NAMES[: x.shape[1]]):
            q = self._hist(x[:, j], self.edges[j])
            p = self.ref[j]
            per[name] = float(np.sum((q - p) * np.log(q / p)))
        return float(np.mean(list(per.values()))), per

    @staticmethod
    def status(score: float) -> str:
        return "stable" if score < STABLE else "moderate drift" if score < WARN else "significant drift - retrain"

    def to_dict(self) -> dict:
        return {"edges": [e.tolist() for e in self.edges], "ref": [r.tolist() for r in self.ref]}

    @classmethod
    def from_dict(cls, d: dict) -> "DriftMonitor":
        return cls([np.asarray(e) for e in d["edges"]], [np.asarray(r) for r in d["ref"]])
