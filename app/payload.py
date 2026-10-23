"""Turn a detection result into the compact JSON the custom console renders.

Kept free of PyTorch so it can be unit-tested (and previewed) on its own: it
only needs objects with the attributes of ``xgnn_ids.inference.Detection``.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from xgnn_ids.config import CLASS_INFO, CLASSES
from xgnn_ids.data.features import is_private_ip

MAX_INTERNAL = 160       # internal hosts are always drawn: they are "your network"
MAX_NODES = 120          # busiest other hosts
MAX_ATTACK_NODES = 230   # extra budget so a DDoS keeps its crowd of sources


def _f(x, nd=4):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) or math.isinf(v) else round(v, nd)


def window_graph(res, gi: int, min_conf: float) -> dict:
    """Nodes + aggregated host-pair links of one window, capped for drawing."""
    g = res.graphs[gi]
    w = res.df.iloc[g.flow_rows]
    src, dst = g.edge_index
    pred = w["pred_label"].to_numpy()
    conf = w["confidence"].to_numpy(float)
    is_alert = (pred != "BENIGN") & (conf >= min_conf)
    deg = np.bincount(src, minlength=g.num_nodes) + np.bincount(dst, minlength=g.num_nodes)
    alert_nodes = np.unique(np.concatenate([src[is_alert], dst[is_alert]])) if is_alert.any() else np.array([], int)

    internal = np.array([is_private_ip(str(ip)) for ip in g.node_ips])
    by_deg = np.argsort(-deg)
    keep = set(by_deg[internal[by_deg]][:MAX_INTERNAL].tolist())
    keep |= set(by_deg[:MAX_NODES].tolist())
    if len(alert_nodes):
        ranked = alert_nodes[np.argsort(-deg[alert_nodes])]
        keep |= set(ranked[:MAX_ATTACK_NODES].tolist())
    keep_list = sorted(keep, key=lambda n: -deg[n])
    local = {n: i for i, n in enumerate(keep_list)}

    cls_id = {c: i for i, c in enumerate(CLASSES)}
    pairs: dict[tuple[int, int], list] = {}
    rows = g.flow_rows
    for k in range(g.num_edges):
        s, d = int(src[k]), int(dst[k])
        if s not in local or d not in local:
            continue
        rec = pairs.get((s, d))
        if rec is None:
            rec = pairs[(s, d)] = {"n": 0, "votes": {}, "row": int(rows[k]), "conf": 0.0, "attack": False}
        rec["n"] += 1
        if is_alert[k]:
            rec["votes"][pred[k]] = rec["votes"].get(pred[k], 0) + 1
            if not rec["attack"] or conf[k] > rec["conf"]:
                rec["row"], rec["conf"], rec["attack"] = int(rows[k]), float(conf[k]), True
    links = []
    for (s, d), rec in pairs.items():
        label = max(rec["votes"], key=rec["votes"].get) if rec["votes"] else "BENIGN"
        links.append([local[s], local[d], cls_id[label], rec["n"], rec["row"], round(rec["conf"], 3)])
    alert_set = set(alert_nodes.tolist())
    nodes = [[str(g.node_ips[n]), int(internal[n]), int(deg[n]), int(n in alert_set)] for n in keep_list]
    return {"nodes": nodes, "links": links, "hidden": int(g.num_nodes - len(keep_list))}


def windows_summary(res) -> list[dict]:
    out = []
    for _, r in res.windows.iterrows():
        out.append({
            "i": int(r["window"]), "t0": _f(r["t0"], 2), "t1": _f(r["t1"], 2), "flows": int(r["flows"]),
            "hosts": int(r["hosts"]), "alerts": int(r["alerts"]), "latency": _f(r["latency_ms"], 2),
            "drift": _f(r.get("drift_psi")), "counts": [int(r[c]) for c in CLASSES[1:]],
        })
    return out


def _window_of_time(windows: list[dict], t: float) -> int:
    for w in windows:
        if w["t0"] <= t < w["t1"]:
            return w["i"]
    return windows[-1]["i"] if windows else 0


def incidents_payload(incidents: list[dict], alerts: pd.DataFrame, windows: list[dict]) -> list[dict]:
    out = []
    for inc in incidents:
        m = (alerts["pred_label"] == inc["class"]) & alerts["ts"].between(inc["start"], inc["end"])
        for k, v in inc["entity"].items():
            m &= alerts[k] == v
        sample = alerts[m].sort_values("confidence", ascending=False).drop_duplicates(["src_ip", "dst_ip"]).head(6)
        out.append({
            "id": inc["id"], "cls": inc["class"], "title": inc["title"], "start": _f(inc["start"], 2),
            "end": _f(inc["end"], 2), "flows": inc["flows"], "sources": inc["sources"], "targets": inc["targets"],
            "ports": inc["top_ports"], "conf": _f(inc["mean_confidence"], 3), "severity": inc["severity"],
            "mitre": inc["mitre"], "response": inc["response"], "rules": inc["firewall"]["iptables"],
            "nft": inc["firewall"]["nftables"], "entity": inc["entity"],
            "w0": _window_of_time(windows, inc["start"]), "w1": _window_of_time(windows, inc["end"]),
            "hosts": list(dict.fromkeys(inc["top_sources"][:12] + inc["top_targets"][:6])),
            "samples": [{"row": int(i), "src": f"{r.src_ip}:{int(r.src_port)}", "dst": f"{r.dst_ip}:{int(r.dst_port)}",
                         "conf": _f(r.confidence, 3)} for i, r in sample.iterrows()],
        })
    return out


def truth_payload(res) -> dict | None:
    from xgnn_ids.training.metrics import classification_metrics

    df = res.df
    if df["label"].isin(CLASSES).mean() < 0.95:
        return None
    idx = {c: i for i, c in enumerate(CLASSES)}
    y = df["label"].map(idx).fillna(-1).astype(int).to_numpy()
    p = df["pred_label"].map(idx).to_numpy()
    m = classification_metrics(y, p)
    rows = []
    for c in CLASSES:
        pc = m["per_class"][c]
        if pc["support"] == 0:
            continue
        truth = df[df["label"] == c]
        hit = truth[truth["pred_label"] == c]
        delay = float(hit["ts"].min() - truth["ts"].min()) if len(hit) and c != "BENIGN" else None
        rows.append({"cls": c, "support": pc["support"], "precision": _f(pc["precision"]), "recall": _f(pc["recall"]),
                     "f1": _f(pc["f1"]), "delay": _f(delay, 1)})
    return {"macro_f1": _f(m["macro_f1"]), "detection_rate": _f(m["detection_rate"]), "fpr": _f(m["fpr"], 5),
            "accuracy": _f(m["accuracy"]), "rows": rows}


def eval_payload(report: dict | None, names: dict[str, str]) -> dict | None:
    """Trim results/metrics.json to what the evaluation page shows."""
    if not report:
        return None

    def model_row(k, v):
        return {"key": k, "name": names.get(k, k), "macro_f1": _f(v["macro_f1"]), "ddos_f1": _f(v.get("ddos_f1")),
                "fpr": _f(v["fpr"], 5), "detection_rate": _f(v["detection_rate"]), "accuracy": _f(v["accuracy"]),
                "latency": _f(v["latency_ms_per_window"], 2), "throughput": _f(v["throughput_flows_per_s"], 0),
                "per_class": {c: _f(v["per_class"][c]["f1"]) for c in CLASSES}}

    out = {"source": report["data"]["source"], "flows": report["data"]["flows"],
           "windows": report["data"]["all"]["windows"], "default": report["default_model"],
           "models": [model_row(k, v) for k, v in report["results"].items()]}
    cd = report.get("cross_domain")
    if cd:
        out["cross"] = {"psi_in": _f(cd["drift_psi_in_domain"]), "psi_out": _f(cd["drift_psi_iot"]),
                        "models": [{"key": k, "name": names.get(k, k), "in": _f(report["results"][k]["macro_f1"]),
                                    "out": _f(v["macro_f1"]), "fpr": _f(v["fpr"], 4)} for k, v in cd["results"].items()]}
        fs = cd.get("few_shot")
        if fs:
            out["few_shot"] = {"adapt_flows": fs["adapt_flows"], "eval_flows": fs["eval_flows"],
                               "minutes": round(fs["adapt_windows"] * 20 / 60),
                               "models": [{"key": k, "name": names.get(k, k),
                                           "f1_before": _f(v["zero_shot"]["macro_f1"]), "f1_after": _f(v["few_shot"]["macro_f1"]),
                                           "fpr_before": _f(v["zero_shot"]["fpr"], 4), "fpr_after": _f(v["few_shot"]["fpr"], 4),
                                           "home_after": _f(v["enterprise_after"]["macro_f1"])}
                                          for k, v in fs["results"].items()]}
    if report.get("zero_day"):
        out["zero_day"] = [{"cls": r["unseen_class"], "flows": r["flows"], "attack": _f(r["flagged_as_attack"]),
                            "uncertain": _f(r["flagged_uncertain"]), "detected": _f(r["detected"]),
                            "label": r["most_common_prediction"], "false_alarm": _f(r["benign_false_alarm"])}
                           for r in report["zero_day"]["results"]]
    if report.get("adversarial"):
        out["adversarial"] = [{"key": k, "name": names.get(k, k),
                               "points": [[_f(p["eps"]), _f(p["detection_rate"])] for p in v]}
                              for k, v in report["adversarial"].items()]
    if report.get("xai"):
        def xrows(rows):
            return [{"method": r["method"], "fid_plus": _f(r.get("fidelity_plus"), 3),
                     "fid_minus": _f(r.get("fidelity_minus"), 3), "sparsity": _f(r.get("sparsity"), 3),
                     "stability": _f(r.get("stability"), 3), "runtime": _f(r.get("runtime_ms"), 1),
                     "flip": _f(r.get("flip_rate"), 3)} for r in rows]

        out["xai"] = {"n": report["xai"]["n_alerts"], "features": xrows(report["xai"]["feature_explainers"]),
                      "structure": xrows(report["xai"]["structural_explainers"])}
    return out


def explanation_payload(exp: dict, row: int, with_shap: bool) -> dict:
    """Strip heavy arrays; keep what the case file shows."""
    e = {k: v for k, v in exp.items() if k not in {"structure", "structure_gnnexplainer"}}
    st = exp.get("structure")
    if st:
        e["structure"] = {k: v for k, v in st.items() if k not in {"edge_mask", "sub_edges"}}
    e["row"] = row
    e["with_shap"] = with_shap
    e["class_info"] = CLASS_INFO.get(exp["prediction"]["label"], {})
    return e


def build_payload(res, incidents: list[dict], min_conf: float, meta: dict, explanation: dict | None = None,
                  evaluation: dict | None = None, nonce=None, error: str | None = None) -> dict:
    alerts = res.df[(res.df["pred_label"] != "BENIGN") & (res.df["confidence"] >= min_conf)]
    windows = windows_summary(res)
    payload = {
        "nonce": nonce,
        "meta": {**meta, "classes": CLASSES, "flows": int(len(res.df)), "alerts": int(len(alerts)),
                 "hosts": int(pd.concat([res.df["src_ip"], res.df["dst_ip"]]).nunique()),
                 "duration": _f(res.df["ts"].max() - res.df["ts"].min(), 1) if len(res.df) else 0,
                 "latency": _f(res.windows["latency_ms"].mean(), 2) if len(res.windows) else None,
                 "drift": _f(res.windows["drift_psi"].median()) if "drift_psi" in res.windows else None},
        "windows": windows,
        "graphs": [window_graph(res, gi, min_conf) for gi in range(len(res.graphs))],
        "incidents": incidents_payload(incidents, alerts, windows),
        "truth": truth_payload(res),
        "explanation": explanation,
        "eval": evaluation,
        "error": error,
    }
    return payload
