"""Quantitative explanation-quality metrics (fidelity, sparsity, stability).

The literature review's most consistent gap: explainers are applied but
rarely *measured*. These metrics follow the GraphFramEx / Amara et al. (2022)
definitions adapted to a single explained flow:

* fidelity+  (necessity)   p(c | x) - p(c | x without the top-k items)   higher = better
* fidelity-  (sufficiency) p(c | x) - p(c | x with only the top-k items) lower  = better
* sparsity                 1 - share of items needed to cover 80% of the attribution mass
* stability                Jaccard overlap of top-k items under small input noise
"""
from __future__ import annotations

import numpy as np
import torch

from .core import ExplainContext, batched_target_proba, target_proba


def feature_fidelity(ctx: ExplainContext, attr: np.ndarray, k: int = 5) -> dict:
    x0 = ctx.edge_attr[ctx.target].numpy()
    p = float(ctx.probs[ctx.cls])
    top = np.argsort(-np.abs(attr))[:k]
    x_rm = x0.copy()
    x_rm[top] = 0.0                     # replace by the training mean
    x_keep = np.zeros_like(x0)
    x_keep[top] = x0[top]
    p_rm, p_keep = batched_target_proba(ctx, np.stack([x_rm, x_keep]))[:, ctx.cls]
    return {"fidelity_plus": p - float(p_rm), "fidelity_minus": p - float(p_keep)}


def structural_fidelity(ctx: ExplainContext, edge_mask: np.ndarray, k: int = 10) -> dict:
    others = np.array([i for i in np.argsort(-edge_mask) if i != ctx.target])
    p = float(ctx.probs[ctx.cls])
    if len(others) == 0:
        return {"fidelity_plus": 0.0, "fidelity_minus": 0.0}
    top = others[:k]
    w_rm = torch.ones(ctx.m)
    w_rm[torch.from_numpy(top)] = 0.0
    w_keep = torch.zeros(ctx.m)
    w_keep[torch.from_numpy(top)] = 1.0
    w_keep[ctx.target] = 1.0
    return {
        "fidelity_plus": p - float(target_proba(ctx, edge_weight=w_rm)[ctx.cls]),
        "fidelity_minus": p - float(target_proba(ctx, edge_weight=w_keep)[ctx.cls]),
    }


def sparsity(attr: np.ndarray, mass: float = 0.8) -> float:
    a = np.sort(np.abs(np.asarray(attr, float)))[::-1]
    if a.sum() <= 0:
        return 0.0
    needed = int(np.searchsorted(np.cumsum(a) / a.sum(), mass) + 1)
    return 1.0 - needed / len(a)


def topk_jaccard(a: np.ndarray, b: np.ndarray, k: int = 5) -> float:
    sa = set(np.argsort(-np.abs(a))[:k].tolist())
    sb = set(np.argsort(-np.abs(b))[:k].tolist())
    return len(sa & sb) / max(len(sa | sb), 1)


def perturbed_context(ctx: ExplainContext, sigma: float, seed: int) -> ExplainContext:
    """Copy of ctx with Gaussian noise on the explained flow's features (same predicted class kept)."""
    g = torch.Generator().manual_seed(seed)
    ea = ctx.edge_attr.clone()
    ea[ctx.target] = ea[ctx.target] + sigma * torch.randn(ea.size(1), generator=g)
    new = ExplainContext(**{**ctx.__dict__, "edge_attr": ea})
    new.probs = target_proba(new)
    return new
