"""Assemble a complete, analyst-readable explanation for one flagged flow."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CLASS_INFO, CLASSES
from ..data.features import EDGE_FEATURE_LABELS, EDGE_FEATURE_NAMES
from ..graph.builder import WindowGraph
from .core import build_context, feature_space
from .evidence import graph_evidence
from .explainers import (feature_counterfactual, gnn_explainer, integrated_gradients, shap_attribution,
                         structural_counterfactual, timed)

PROTO = {6: "TCP", 17: "UDP", 1: "ICMP"}

# log-scaled feature -> unit of the de-logged value (counts are shown as integers)
_UNITS = {
    "log_duration_ms": "ms", "log_iat_ms": "ms", "log_fwd_bytes": "B", "log_bwd_bytes": "B",
    "log_bytes_per_s": "B/s", "log_pkts_per_s": "packets/s", "log_fwd_pkt_len": "B", "log_bwd_pkt_len": "B",
    "log_out_bytes": "B", "log_in_bytes": "B",
}
_COUNTS = {"log_fwd_pkts", "log_bwd_pkts", "log_syn", "log_ack", "log_fin", "log_rst", "log_psh", "log_out_flows",
           "log_in_flows", "log_out_peers", "log_in_peers", "log_out_dports", "log_in_dports"}


def raw_feature_text(name: str, value: float) -> str:
    """Human text for a feature given its *unscaled* value (as produced by the feature builders)."""
    if name in _COUNTS:
        return f"{max(np.expm1(value), 0):,.0f}"
    if name in _UNITS:
        v = max(float(np.expm1(value)), 0.0)
        return f"{v:,.0f} {_UNITS[name]}" if v >= 10 else f"{v:,.2f} {_UNITS[name]}"
    if name == "bwd_fwd_byte_ratio":
        return f"{np.exp(value):.2f}x"
    if name.endswith("_ratio"):
        return f"{value:.0%}"
    return "yes" if value >= 0.5 else "no"


def scaled_to_text(prep, name: str, z: float) -> str:
    i = EDGE_FEATURE_NAMES.index(name)
    sc = prep.edge_scaler
    return raw_feature_text(name, z * sc.std[i] + sc.mean[i])


def _raw_z(ctx) -> np.ndarray:
    """Unscaled values of the explanation-space vector (for display)."""
    g = ctx.graph
    parts = [g.edge_attr[ctx.edge_id]]
    if ctx.with_context:
        parts += [g.x[ctx.sub_nodes[ctx.src]], g.x[ctx.sub_nodes[ctx.dst]]]
    return np.concatenate(parts)


def _top_features(ctx, attr: np.ndarray, k: int = 8) -> list[dict]:
    keys, labels = feature_space(ctx.with_context)
    raw = _raw_z(ctx)
    order = np.argsort(-np.abs(attr))[:k]
    out = []
    for i in order:
        key = keys[i]
        base = key.split(":", 1)[-1]
        out.append({"feature": key, "label": labels[i], "attribution": float(attr[i]),
                    "value": raw_feature_text(base, float(raw[i])),
                    "scope": "flow" if ":" not in key else ("source host" if key.startswith("src") else "destination host")})
    return out


def explain_flow(model, prep, graph: WindowGraph, df: pd.DataFrame, edge_id: int, hops: int,
                 background: np.ndarray | None = None, methods=("ig", "gnnexplainer", "counterfactual"),
                 shap_nsamples: int = 200) -> dict:
    """Run the requested explainers and build the structured explanation + narrative."""
    ctx = build_context(model, prep, graph, edge_id, hops)
    row = df.iloc[graph.flow_rows[edge_id]]
    cls_name = CLASSES[ctx.cls]
    out: dict = {
        "flow": {
            "src_ip": str(row["src_ip"]), "dst_ip": str(row["dst_ip"]), "src_port": int(row["src_port"]),
            "dst_port": int(row["dst_port"]), "protocol": PROTO.get(int(row["protocol"]), str(int(row["protocol"]))),
            "ts": float(row["ts"]), "true_label": str(row["label"]), "window": graph.window_id,
        },
        "prediction": {"label": cls_name, "confidence": float(ctx.probs[ctx.cls]),
                       "probs": {c: float(p) for c, p in zip(CLASSES, ctx.probs)}},
        "subgraph": {"nodes": int(len(ctx.sub_nodes)), "flows": int(ctx.m)},
        "features": {}, "timings_ms": {}, **{k: CLASS_INFO[cls_name][k] for k in ("mitre", "response", "severity")},
    }
    if "ig" in methods:
        attr, ms = timed(integrated_gradients, ctx)
        out["features"]["Integrated Gradients"] = _top_features(ctx, attr)
        out["timings_ms"]["Integrated Gradients"] = ms
    if "shap" in methods and background is not None:
        attr, ms = timed(shap_attribution, ctx, background, shap_nsamples)
        out["features"]["SHAP"] = _top_features(ctx, attr)
        out["timings_ms"]["SHAP"] = ms
    edge_mask = None
    if "gnnexplainer" in methods and ctx.m > 1:
        (edge_mask, feat_mask), ms = timed(gnn_explainer, ctx)
        out["timings_ms"]["GNNExplainer"] = ms
        out["features"]["GNNExplainer (feature mask)"] = _top_features(ctx, feat_mask - feat_mask.min())
        out["structure"] = structural_summary(ctx, edge_mask, df)
    if "counterfactual" in methods and ctx.cls != 0:
        sc = prep.edge_scaler
        bounds = (sc.lo, sc.hi) if sc.lo is not None else None
        cf, ms = timed(feature_counterfactual, ctx, EDGE_FEATURE_NAMES, 0, bounds)
        if cf.get("changes"):
            cf["text"] = "; ".join(
                f"{EDGE_FEATURE_LABELS[EDGE_FEATURE_NAMES[i]].lower()} were "
                f"{scaled_to_text(prep, EDGE_FEATURE_NAMES[i], b)} instead of {scaled_to_text(prep, EDGE_FEATURE_NAMES[i], a)}"
                for i, a, b in cf["changes"])
            cf.pop("changes")
        out["counterfactual_features"] = cf
        if edge_mask is not None:
            out["counterfactual_structure"] = structural_counterfactual(ctx, edge_mask)
        out["timings_ms"]["Counterfactual"] = ms
    out["evidence"] = graph_evidence(df, graph, edge_id)
    out["narrative"] = narrative(out)
    return out


def structural_summary(ctx, edge_mask: np.ndarray, df: pd.DataFrame, k: int = 10) -> dict:
    g = ctx.graph
    sub = ctx.sub_edges
    src, dst = g.edge_index[:, sub]
    t_src, t_dst = src[ctx.target], dst[ctx.target]
    others = np.arange(ctx.m) != ctx.target
    mass = edge_mask[others].sum()
    share_dst = float(edge_mask[others & (dst == t_dst)].sum() / mass) if mass > 0 else 0.0
    share_src = float(edge_mask[others & (src == t_src)].sum() / mass) if mass > 0 else 0.0
    top = [i for i in np.argsort(-edge_mask) if i != ctx.target][:k]
    rows = df.iloc[g.flow_rows[sub[top]]] if top else df.iloc[[]]
    top_edges = [{"src": str(r.src_ip), "dst": str(r.dst_ip), "dst_port": int(r.dst_port),
                  "importance": float(edge_mask[i])} for i, r in zip(top, rows.itertuples())]
    top_share = float(edge_mask[top].sum() / mass) if mass > 0 and top else 0.0
    return {"neighbour_flows": int(others.sum()), "share_same_destination": share_dst,
            "share_same_source": share_src, "important_flows": int((edge_mask[others] > 0.5).sum()),
            "top_share": top_share, "max_importance": float(edge_mask[others].max()) if others.any() else 0.0,
            "top_edges": top_edges, "edge_mask": edge_mask.tolist(), "sub_edges": sub.tolist()}


def narrative(e: dict) -> str:
    f, p = e["flow"], e["prediction"]
    lines = [f"**{p['label']}** detected with {p['confidence']:.0%} confidence on flow "
             f"`{f['src_ip']}:{f['src_port']} -> {f['dst_ip']}:{f['dst_port']}/{f['protocol']}`."]
    if p["label"] == "BENIGN":
        return lines[0] + " The flow looks normal."
    why = [f"- **Network context:** {ev['text']}" for ev in e.get("evidence", [])[:3]]
    feats = next(iter(e["features"].values()), [])
    pos = [x for x in feats if x["attribution"] > 0]
    host = [x for x in pos if x["scope"] != "flow"][:2]
    flow = [x for x in pos if x["scope"] == "flow"][:3]
    if host:
        why.append("- **Graph context the model relied on:** "
                   + "; ".join(f"{x['label'].lower()} = {x['value']}" for x in host) + ".")
    if flow:
        why.append("- **Properties of the flow itself:** "
                   + "; ".join(f"{x['label'].split(': ', 1)[-1]} = {x['value']}" for x in flow) + ".")
    st = e.get("structure")
    if st and st["neighbour_flows"]:
        where = ("flows towards the same destination" if st["share_same_destination"] >= st["share_same_source"]
                 else "other flows from the same source")
        share = max(st["share_same_destination"], st["share_same_source"])
        if st["max_importance"] >= 0.5:
            why.append(f"- **Neighbouring flows (GNNExplainer):** {st['important_flows']} of {st['neighbour_flows']} "
                       f"neighbouring flows are important; {share:.0%} of the weight is on {where}.")
        else:
            why.append(f"- **Neighbouring flows (GNNExplainer):** no single one of the {st['neighbour_flows']} "
                       f"neighbouring flows is decisive - the evidence is spread out ({share:.0%} of the weight is on "
                       f"{where}).")
    if len(why) > 0:
        lines += ["", "**Why:**", *why]
    cf, scf = e.get("counterfactual_features"), e.get("counterfactual_structure")
    what_if = []
    if cf and cf.get("found") and cf.get("text"):
        what_if.append(f"- It would be classified BENIGN if its {cf['text']}.")
    elif cf is not None:
        what_if.append("- No realistic change to this flow's own properties makes it benign - "
                       "the decision rests on its context.")
    if scf and scf.get("found"):
        what_if.append(f"- Removing the {scf['removed']} most important of {scf['total_neighbours']} neighbouring "
                       "flows from message passing would make it look BENIGN.")
    if what_if:
        lines += ["", "**What would change the decision:**", *what_if]
    lines += ["", f"**MITRE ATT&CK:** {e['mitre']}", f"**Recommended response:** {e['response']}"]
    return "\n".join(lines)

