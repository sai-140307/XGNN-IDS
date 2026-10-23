"""Feature-level, structure-level and counterfactual explainers for one flow.

* Integrated Gradients - gradient attribution over the flow's features and the
                          host context of its endpoints (fast, one batched pass).
* SHAP                 - Shapley values over the same explanation space
                          (KernelSHAP via `shap`, or a built-in permutation estimator).
* GNNExplainer         - learns a soft mask over neighbouring flows (+ a feature
                          mask) that preserves the prediction: *which other
                          flows* in the window the message passing relied on.
* Counterfactuals      - (a) the smallest realistic change to the flow's own
                          properties that makes it benign, (b) how many important
                          neighbouring flows must disappear for it to look benign.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from .core import ExplainContext, _place, _replicate, batched_target_proba, target_proba


def integrated_gradients(ctx: ExplainContext, steps: int = 32, baseline: np.ndarray | None = None) -> np.ndarray:
    """IG attribution of p(cls) over the explanation space. Baseline = training mean (0 after scaling)."""
    z0 = torch.from_numpy(ctx.z0)
    base = torch.zeros_like(z0) if baseline is None else torch.as_tensor(baseline, dtype=z0.dtype)
    alphas = torch.linspace(1.0 / steps, 1.0, steps).unsqueeze(1)
    path = (base + alphas * (z0 - base)).detach().requires_grad_(True)
    x, ei, ea, tgt, src, dst = _replicate(ctx, steps)
    x, ea = _place(ctx, x.clone(), ea.clone(), tgt, src, dst, path)
    p = F.softmax(ctx.model(x, ei, ea)[tgt], 1)[:, ctx.cls].sum()
    (grads,) = torch.autograd.grad(p, path)
    return ((z0 - base) * grads.mean(0)).detach().numpy()


def shap_attribution(ctx: ExplainContext, background: np.ndarray, nsamples: int = 200, seed: int = 0) -> np.ndarray:
    """Shapley values over the explanation space (KernelSHAP if `shap` is installed)."""
    z0 = ctx.z0
    bg = np.asarray(background, np.float32)[:, : len(z0)]

    def f(Z):
        return batched_target_proba(ctx, Z)[:, ctx.cls]

    try:
        import shap  # noqa: WPS433

        explainer = shap.KernelExplainer(f, bg)
        vals = explainer.shap_values(z0[None, :], nsamples=nsamples, silent=True)
        return np.asarray(vals).reshape(-1)[: len(z0)]
    except ImportError:
        return permutation_shap(f, z0, bg, n_perm=max(4, nsamples // len(z0)), seed=seed)


def permutation_shap(f, x: np.ndarray, background: np.ndarray, n_perm: int = 8, seed: int = 0) -> np.ndarray:
    """Monte-Carlo Shapley values (Strumbelj & Kononenko), batched per permutation."""
    rng = np.random.default_rng(seed)
    d = len(x)
    phi = np.zeros(d)
    for k in range(n_perm):
        base = background[k % len(background)].copy()
        perm = rng.permutation(d)
        rows = np.repeat(base[None, :], d + 1, 0)
        for i, j in enumerate(perm):
            rows[i + 1:, j] = x[j]
        v = f(rows)
        phi[perm] += np.diff(v)
    return phi / n_perm


def edge_integrated_gradients(ctx: ExplainContext, steps: int = 32, max_edges: int = 400_000) -> np.ndarray:
    """Integrated Gradients over the message-passing weights of the neighbouring flows.

    Path: all neighbouring flows scaled from absent (weight 0) to present (weight 1); the explained
    flow itself stays present. Returns one attribution per flow of the computation subgraph
    (positive = the flow pushes the prediction towards the explained class).
    """
    m = ctx.m
    if m <= 1:
        return np.ones(m)
    keep = torch.zeros(m, dtype=torch.bool)
    keep[ctx.target] = True
    alphas = torch.linspace(1.0 / steps, 1.0, steps)
    chunk = max(1, max_edges // m)
    total = torch.zeros(m)
    for p in ctx.model.parameters():
        p.requires_grad_(False)
    try:
        for i in range(0, steps, chunk):
            a = alphas[i:i + chunk]
            b = len(a)
            w = a.repeat_interleave(m).view(b, m).clone()
            w[:, keep] = 1.0
            w = w.reshape(-1).requires_grad_(True)
            x, ei, ea, tgt, _, _ = _replicate(ctx, b)
            prob = F.softmax(ctx.model(x, ei, ea, w)[tgt], 1)[:, ctx.cls].sum()
            (g,) = torch.autograd.grad(prob, w)
            total += g.view(b, m).sum(0)
    finally:
        for p in ctx.model.parameters():
            p.requires_grad_(True)
    attr = (total / steps).numpy()
    attr[ctx.target] = np.abs(attr).max() if m > 1 else 1.0
    return attr


def _entropy(q: torch.Tensor) -> torch.Tensor:
    return -(q * (q + 1e-12).log() + (1 - q) * (1 - q + 1e-12).log())


def gnn_explainer(ctx: ExplainContext, epochs: int = 100, lr: float = 0.05, seed: int = 0,
                  edge_size: float = 0.5, edge_ent: float = 0.5, feat_size: float = 0.5, feat_ent: float = 0.1):
    """GNNExplainer (Ying et al., 2019) adapted to edge classification.

    Returns (edge_mask [m] in [0,1], feature_mask over the explanation space in [0,1]).
    The explained flow itself is always kept (mask = 1). The node-feature mask is
    shared by all hosts and reported for both endpoints.
    """
    torch.manual_seed(seed)
    m, fe, fn = ctx.m, ctx.n_edge_feats, ctx.x.size(1)
    edge_logit = torch.nn.Parameter(torch.randn(m) * 0.1 + 1.0)
    efeat_logit = torch.nn.Parameter(torch.randn(fe) * 0.1 + 1.0)
    nfeat_logit = torch.nn.Parameter(torch.randn(fn) * 0.1 + 1.0)
    params = [edge_logit, efeat_logit] + ([nfeat_logit] if ctx.with_context else [])
    opt = torch.optim.Adam(params, lr=lr)
    keep = torch.zeros(m, dtype=torch.bool)
    keep[ctx.target] = True
    for p in ctx.model.parameters():
        p.requires_grad_(False)
    try:
        for _ in range(epochs):
            opt.zero_grad()
            em = torch.where(keep, torch.ones(m), torch.sigmoid(edge_logit))
            fm_e = torch.sigmoid(efeat_logit)
            x = ctx.x * torch.sigmoid(nfeat_logit) if ctx.with_context else ctx.x
            logits = ctx.model(x, ctx.edge_index, ctx.edge_attr * fm_e, em)
            loss = -F.log_softmax(logits[ctx.target], 0)[ctx.cls]
            other = em[~keep]
            if other.numel():
                loss = loss + edge_size * other.mean() + edge_ent * _entropy(other).mean()
            fm = torch.cat([fm_e, torch.sigmoid(nfeat_logit)]) if ctx.with_context else fm_e
            loss = loss + feat_size * fm.mean() + feat_ent * _entropy(fm).mean()
            loss.backward()
            opt.step()
    finally:
        for p in ctx.model.parameters():
            p.requires_grad_(True)
    em = torch.sigmoid(edge_logit).detach().numpy()
    em[ctx.target] = 1.0
    fm_e = torch.sigmoid(efeat_logit).detach().numpy()
    if ctx.with_context:
        fm_n = torch.sigmoid(nfeat_logit).detach().numpy()
        return em, np.concatenate([fm_e, fm_n, fm_n])
    return em, fm_e


# ------------------------------------------------------------------ counterfactuals

# Properties of a flow that can realistically differ (timing, volume, sizes, flags).
# Protocol and destination-port category (which service) and binary indicators are fixed.
MUTABLE_FEATURES = (
    "log_duration_ms", "log_fwd_pkts", "log_bwd_pkts", "log_fwd_bytes", "log_bwd_bytes", "log_bytes_per_s",
    "log_pkts_per_s", "log_fwd_pkt_len", "log_bwd_pkt_len", "log_iat_ms", "log_syn", "log_ack", "log_fin",
    "log_rst", "log_psh",
)


def feature_counterfactual(ctx: ExplainContext, feature_names: list[str], target_cls: int = 0,
                           bounds: tuple[np.ndarray, np.ndarray] | None = None, steps: int = 300,
                           lr: float = 0.05, l1: float = 0.05, max_changes: int = 4) -> dict:
    """Smallest realistic change to the flow's own features that makes the model predict ``target_cls``.

    The search stays inside ``bounds`` (per-feature range seen in training, scaled units) and
    is then pruned to the fewest features that still flip the decision.
    """
    mutable = torch.tensor([n in MUTABLE_FEATURES for n in feature_names], dtype=torch.float32)
    x0 = ctx.edge_attr[ctx.target].clone()
    lo = torch.as_tensor(bounds[0], dtype=torch.float32) if bounds is not None else x0 - 3
    hi = torch.as_tensor(bounds[1], dtype=torch.float32) if bounds is not None else x0 + 3
    lo, hi = torch.minimum(lo, x0), torch.maximum(hi, x0)
    delta = torch.zeros_like(x0, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    found = None
    for p in ctx.model.parameters():
        p.requires_grad_(False)

    def predict(vec):
        ea = ctx.edge_attr.clone()
        ea[ctx.target] = vec
        return ctx.model(ctx.x, ctx.edge_index, ea)[ctx.target]

    try:
        for _ in range(steps):
            opt.zero_grad()
            cand = torch.clamp(x0 + delta * mutable, lo, hi)
            logits = predict(cand)
            if int(logits.argmax()) == target_cls:
                found = cand.detach().clone()
                break
            loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([target_cls])) + l1 * (delta * mutable).abs().sum()
            loss.backward()
            opt.step()
    finally:
        for p in ctx.model.parameters():
            p.requires_grad_(True)
    if found is None:
        return {"found": False}
    # prune: keep the fewest (largest) changes that still flip the decision
    d = (found - x0).numpy()
    order = [int(i) for i in np.argsort(-np.abs(d)) if abs(d[i]) > 1e-3]
    with torch.no_grad():
        for k in range(1, min(max_changes, len(order)) + 1):
            cand = x0.clone()
            cand[order[:k]] = found[order[:k]]
            if int(predict(cand).argmax()) == target_cls:
                found, order = cand, order[:k]
                break
        else:
            order = order[:max_changes]
    changes = [(i, float(x0[i]), float(found[i])) for i in order]
    ea = ctx.edge_attr.clone()
    ea[ctx.target] = found
    new_p = target_proba(ctx, ea)
    return {"found": int(new_p.argmax()) == target_cls, "changes": changes, "new_probs": new_p.tolist()}


def structural_counterfactual(ctx: ExplainContext, edge_mask: np.ndarray, target_cls: int = 0) -> dict:
    """Remove the most important neighbouring flows (by GNNExplainer mask) until the prediction flips."""
    order = [int(i) for i in np.argsort(-edge_mask) if i != ctx.target]
    if not order:
        return {"found": False, "reason": "flow has no neighbours", "total_neighbours": 0}
    ks = sorted({k for k in [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987] if k < len(order)} | {len(order)})
    for k in ks:
        w = torch.ones(ctx.m)
        w[order[:k]] = 0.0
        p = target_proba(ctx, edge_weight=w)
        if int(p.argmax()) == target_cls:
            return {"found": True, "removed": k, "new_probs": p.tolist(), "total_neighbours": len(order)}
    return {"found": False, "reason": "prediction does not depend on neighbouring flows alone",
            "total_neighbours": len(order)}


def timed(fn, *args, **kwargs):
    t = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, (time.perf_counter() - t) * 1000.0
