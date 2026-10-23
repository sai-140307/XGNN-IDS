"""Benchmark the explainers: fidelity, sparsity, stability and runtime per method."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CLASSES
from ..graph.builder import WindowGraph
from .core import build_context
from .explainers import gnn_explainer, integrated_gradients, shap_attribution, timed
from .quality import feature_fidelity, perturbed_context, sparsity, structural_fidelity, topk_jaccard


def sample_alerts(model, prep, graphs: list[WindowGraph], per_class: int, seed: int = 0) -> list[tuple[int, int]]:
    """Pick correctly-classified attack flows, ``per_class`` per attack family: (graph idx, edge id)."""
    from ..training.trainer import predict_proba

    rng = np.random.default_rng(seed)
    picks = []
    cands: dict[int, list[tuple[int, int]]] = {}
    for gi, g in enumerate(graphs):
        p, _ = predict_proba(model, prep, [g])
        ok = np.flatnonzero((p.argmax(1) == g.y) & (g.y > 0))
        for e in ok:
            cands.setdefault(int(g.y[e]), []).append((gi, int(e)))
    for c, lst in sorted(cands.items()):
        idx = rng.choice(len(lst), min(per_class, len(lst)), replace=False)
        picks += [lst[i] for i in idx]
    return picks


def evaluate_explainers(model, prep, graphs: list[WindowGraph], hops: int, background: np.ndarray,
                        per_class: int = 8, k: int = 5, shap_nsamples: int = 150, sigma: float = 0.1,
                        log=print) -> dict:
    rng = np.random.default_rng(0)
    feat_rows, struct_rows = [], []
    picks = sample_alerts(model, prep, graphs, per_class)
    log(f"  explaining {len(picks)} alerts")
    for n, (gi, e) in enumerate(picks):
        ctx = build_context(model, prep, graphs[gi], e, hops)
        cls = CLASSES[ctx.cls]
        noisy = perturbed_context(ctx, sigma, seed=n)
        if int(noisy.probs.argmax()) != ctx.cls:
            noisy = None
        methods = {
            "Integrated Gradients": lambda c: integrated_gradients(c),
            "SHAP": lambda c: shap_attribution(c, background, shap_nsamples),
            "GNNExplainer (feature mask)": lambda c: gnn_explainer(c, epochs=100)[1] if c.m > 1 else np.ones(c.edge_attr.size(1)),
            "Random": lambda c: rng.standard_normal(c.edge_attr.size(1)),
        }
        for mname, fn in methods.items():
            attr, ms = timed(fn, ctx)
            fid = feature_fidelity(ctx, attr, k)
            stab = topk_jaccard(attr, fn(noisy), k) if noisy is not None else np.nan
            feat_rows.append({"method": mname, "class": cls, "runtime_ms": ms, "sparsity": sparsity(attr),
                              "stability": stab, **fid})
        if ctx.m > 1:
            (em, _), ms = timed(gnn_explainer, ctx, 100)
            em2, _ = gnn_explainer(ctx, 100, seed=1)
            others = [i for i in range(ctx.m) if i != ctx.target]
            sub = ctx.sub_edges
            dst = graphs[gi].edge_index[1, sub]
            # heuristic baseline: flows sharing the destination are important
            heur = np.where(dst == dst[ctx.target], 1.0, 0.0) + 1e-3 * rng.random(ctx.m)
            rand = rng.random(ctx.m)
            for mname, mask, t in (("GNNExplainer", em, ms), ("Same-destination heuristic", heur, 0.0),
                                   ("Random", rand, 0.0)):
                fid = structural_fidelity(ctx, mask, k=10)
                struct_rows.append({
                    "method": mname, "class": cls, "runtime_ms": t, "neighbours": len(others), **fid,
                    "sparsity": float(1 - (mask[others] > 0.5).mean()) if mname == "GNNExplainer" else np.nan,
                    "stability": topk_jaccard(em[others], em2[others], 10) if mname == "GNNExplainer" else np.nan,
                })
    feat = pd.DataFrame(feat_rows)
    struct = pd.DataFrame(struct_rows)

    def agg(df):
        if df.empty:
            return []
        g = df.drop(columns=["class"]).groupby("method").agg(["mean", "std"])
        out = []
        for m in df["method"].unique():
            r = {"method": m, "n": int((df["method"] == m).sum())}
            for col in ("fidelity_plus", "fidelity_minus", "sparsity", "stability", "runtime_ms"):
                if col in g.columns.get_level_values(0):
                    r[col] = float(g.loc[m, (col, "mean")])
                    r[col + "_std"] = float(g.loc[m, (col, "std")])
            out.append(r)
        return out

    per_class = (feat[feat["method"] != "Random"].groupby(["class", "method"])["fidelity_plus"].mean()
                 .unstack().round(4).reset_index().to_dict("records")) if not feat.empty else []
    return {"feature_explainers": agg(feat), "structural_explainers": agg(struct),
            "feature_fidelity_by_class": per_class, "n_alerts": len(picks), "top_k": k}
