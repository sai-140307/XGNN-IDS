"""Quantitative explanation-quality metrics (fidelity, sparsity, stability).

The literature review's most consistent gap: explainers are applied but
rarely *measured*. Definitions follow GraphFramEx / Amara et al. (2022),
adapted to a single explained flow. Fidelity is computed on the **log-odds**
of the explained class, because a well-trained IDS is often 99.9% confident and
probability differences then hide everything:

* fidelity+  (necessity)   LO(c | z) - LO(c | z with the top-k items reset)    higher = better
* fidelity-  (sufficiency) LO(c | z) - LO(c | only the top-k items kept)        lower  = better
* sparsity                 1 - share of items needed for 80% of the attribution mass
* stability                Jaccard overlap of the top-k items under small input noise

"Reset" means replacing a feature by the training mean (0 after scaling), or
removing a neighbouring flow from message passing.
"""
from __future__ import annotations

import numpy as np
import torch

from .core import ExplainContext, batched_target_proba, context_at, log_odds, target_proba


def feature_fidelity(ctx: ExplainContext, attr: np.ndarray, k: int = 5) -> dict:
    z0 = ctx.z0
    top = np.argsort(-np.abs(attr))[:k]
    z_rm = z0.copy()
    z_rm[top] = 0.0
    z_keep = np.zeros_like(z0)
    z_keep[top] = z0[top]
    p0 = float(ctx.probs[ctx.cls])
    p_rm, p_keep = batched_target_proba(ctx, np.stack([z_rm, z_keep]))[:, ctx.cls]
    return {
        "fidelity_plus": float(log_odds(p0) - log_odds(p_rm)),
        "fidelity_minus": float(log_odds(p0) - log_odds(p_keep)),
        "fidelity_plus_prob": p0 - float(p_rm),
        "flip_rate": float(int(np.argmax(batched_target_proba(ctx, z_rm[None])[0])) != ctx.cls),
    }


def structural_fidelity(ctx: ExplainContext, edge_mask: np.ndarray, frac: float = 0.2) -> dict:
    """Remove / keep the top ``frac`` of neighbouring flows ranked by ``edge_mask``."""
    others = np.array([i for i in np.argsort(-edge_mask) if i != ctx.target])
    if len(others) == 0:
        return {"fidelity_plus": 0.0, "fidelity_minus": 0.0, "k": 0}
    k = max(1, int(round(frac * len(others))))
    top = torch.from_numpy(others[:k])
    w_rm = torch.ones(ctx.m)
    w_rm[top] = 0.0
    w_keep = torch.zeros(ctx.m)
    w_keep[top] = 1.0
    w_keep[ctx.target] = 1.0
    p0 = float(ctx.probs[ctx.cls])
    p_rm = float(target_proba(ctx, edge_weight=w_rm)[ctx.cls])
    p_keep = float(target_proba(ctx, edge_weight=w_keep)[ctx.cls])
    return {"fidelity_plus": float(log_odds(p0) - log_odds(p_rm)),
            "fidelity_minus": float(log_odds(p0) - log_odds(p_keep)), "k": k}


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
    """Copy of ctx with Gaussian noise (std ``sigma``, scaled units) on the explanation-space vector."""
    rng = np.random.default_rng(seed)
    z = ctx.z0 + sigma * rng.standard_normal(len(ctx.z0)).astype(np.float32)
    return context_at(ctx, z)
