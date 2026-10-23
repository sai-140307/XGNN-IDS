"""Feature-level, structure-level and counterfactual explainers for one flow.

* Integrated Gradients - fast gradient attribution over the flow's features.
* SHAP                 - Shapley values of the flow's features (KernelSHAP via
                          the `shap` package, or a built-in permutation estimator).
* GNNExplainer         - learns a soft mask over neighbouring flows (+ a feature
                          mask) that preserves the prediction: *which other
                          flows* in the window made this one look malicious.
* Counterfactuals      - (a) the smallest feature change that makes the flow
                          benign, (b) the smallest set of neighbouring flows whose
                          removal makes it benign.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from .core import ExplainContext, _replicate, batched_target_proba, target_proba


def integrated_gradients(ctx: ExplainContext, steps: int = 32, baseline: np.ndarray | None = None) -> np.ndarray:
    """IG attribution of p(cls) to the target flow's (scaled) features. Baseline = training mean (0)."""
    x0 = ctx.edge_attr[ctx.target].clone()
    base = torch.zeros_like(x0) if baseline is None else torch.as_tensor(baseline, dtype=x0.dtype)
    alphas = torch.linspace(1.0 / steps, 1.0, steps).unsqueeze(1)
    path = (base + alphas * (x0 - base)).detach().requires_grad_(True)
    x, ei, ea, tgt = _replicate(ctx, steps)
    ea = ea.clone()
    ea[tgt] = path
    p = F.softmax(ctx.model(x, ei, ea)[tgt], 1)[:, ctx.cls].sum()
    (grads,) = torch.autograd.grad(p, path)
    return ((x0 - base) * grads.mean(0)).detach().numpy()


def shap_attribution(ctx: ExplainContext, background: np.ndarray, nsamples: int = 200, seed: int = 0) -> np.ndarray:
    """Shapley values for the target flow's features (KernelSHAP if `shap` is installed)."""
    x0 = ctx.edge_attr[ctx.target].numpy()[None, :]

    def f(X):
        return batched_target_proba(ctx, X)[:, ctx.cls]

    try:
        import shap  # noqa: WPS433

        explainer = shap.KernelExplainer(f, background)
        vals = explainer.shap_values(x0, nsamples=nsamples, silent=True)
        return np.asarray(vals).reshape(-1)
    except ImportError:
        return permutation_shap(f, x0[0], background, n_perm=max(4, nsamples // x0.shape[1]), seed=seed)


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


def gnn_explainer(ctx: ExplainContext, epochs: int = 100, lr: float = 0.05, seed: int = 0,
                  edge_size: float = 1.0, edge_ent: float = 0.5, feat_size: float = 0.5, feat_ent: float = 0.1):
    """GNNExplainer (Ying et al., 2019) adapted to edge classification.

    Returns (edge_mask [m] in [0,1], feature_mask [Fe] in [0,1]). The explained
    flow itself is always kept (mask = 1).
    """
    torch.manual_seed(seed)
    m, fe = ctx.m, ctx.edge_attr.size(1)
    if m <= 1:
        return np.ones(m), np.ones(fe)
    edge_logit = torch.nn.Parameter(torch.randn(m) * 0.1 + 1.0)
    feat_logit = torch.nn.Parameter(torch.randn(fe) * 0.1 + 1.0)
    opt = torch.optim.Adam([edge_logit, feat_logit], lr=lr)
    keep = torch.zeros(m, dtype=torch.bool)
    keep[ctx.target] = True
    for p in ctx.model.parameters():
        p.requires_grad_(False)
    try:
        for _ in range(epochs):
            opt.zero_grad()
            em = torch.where(keep, torch.ones(m), torch.sigmoid(edge_logit))
            fm = torch.sigmoid(feat_logit)
            logits = ctx.model(ctx.x, ctx.edge_index, ctx.edge_attr * fm, em)
            loss = -F.log_softmax(logits[ctx.target], 0)[ctx.cls]
            other = em[~keep]
            ent = lambda q: -(q * (q + 1e-12).log() + (1 - q) * (1 - q + 1e-12).log())  # noqa: E731
            loss = loss + edge_size * other.mean() + edge_ent * ent(other).mean()
            loss = loss + feat_size * fm.mean() + feat_ent * ent(fm).mean()
            loss.backward()
            opt.step()
    finally:
        for p in ctx.model.parameters():
            p.requires_grad_(True)
    em = torch.sigmoid(edge_logit).detach().numpy()
    em[ctx.target] = 1.0
    return em, torch.sigmoid(feat_logit).detach().numpy()


# ------------------------------------------------------------------ counterfactuals

# features an attacker / the traffic itself can plausibly change (timing, volume, sizes, flags);
# protocol and destination-port categories are treated as fixed context.
MUTABLE_PREFIXES = ("log_", "no_response", "bwd_fwd_byte_ratio")


def feature_counterfactual(ctx: ExplainContext, feature_names: list[str], target_cls: int = 0, steps: int = 300,
                           lr: float = 0.05, l1: float = 0.02) -> dict:
    """Smallest (L1) change to the flow's own features that makes the model predict ``target_cls``."""
    mutable = torch.tensor([n.startswith(MUTABLE_PREFIXES) for n in feature_names], dtype=torch.float32)
    x0 = ctx.edge_attr[ctx.target].clone()
    delta = torch.zeros_like(x0, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    found = None
    for p in ctx.model.parameters():
        p.requires_grad_(False)
    try:
        for step in range(steps):
            opt.zero_grad()
            ea = ctx.edge_attr.clone()
            ea[ctx.target] = x0 + delta * mutable
            logits = ctx.model(ctx.x, ctx.edge_index, ea)[ctx.target]
            if int(logits.argmax()) == target_cls:
                found = (delta * mutable).detach().clone()
                break
            loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([target_cls])) + l1 * delta.abs().sum()
            loss.backward()
            opt.step()
            with torch.no_grad():
                delta.clamp_(-6, 6)
    finally:
        for p in ctx.model.parameters():
            p.requires_grad_(True)
    if found is None:
        return {"found": False}
    d = found.numpy()
    order = np.argsort(-np.abs(d))
    changes = [(int(i), float(x0[i]), float(x0[i] + d[i])) for i in order if abs(d[i]) > 0.05][:4]
    ea = ctx.edge_attr.clone()
    ea[ctx.target] = x0 + found
    return {"found": True, "changes": changes, "new_probs": target_proba(ctx, ea).tolist(), "l1": float(np.abs(d).sum())}


def structural_counterfactual(ctx: ExplainContext, edge_mask: np.ndarray, target_cls: int = 0) -> dict:
    """Remove the most important neighbouring flows (by GNNExplainer mask) until the prediction flips."""
    order = [int(i) for i in np.argsort(-edge_mask) if i != ctx.target]
    if not order:
        return {"found": False, "reason": "flow has no neighbours"}
    ks = sorted({k for k in [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987] if k < len(order)} | {len(order)})
    for k in ks:
        w = torch.ones(ctx.m)
        w[order[:k]] = 0.0
        p = target_proba(ctx, edge_weight=w)
        if int(p.argmax()) == target_cls:
            return {"found": True, "removed": k, "removed_edges": order[:k], "new_probs": p.tolist(),
                    "total_neighbours": len(order)}
    return {"found": False, "reason": "prediction does not depend on neighbouring flows alone",
            "total_neighbours": len(order)}


def timed(fn, *args, **kwargs):
    t = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, (time.perf_counter() - t) * 1000.0
