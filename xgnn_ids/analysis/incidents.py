"""Alert correlation: collapse thousands of flow alerts into a handful of incidents.

A DDoS produces thousands of malicious flows; an analyst wants *one*
incident: "DDoS against 192.168.10.50:80 from 312 sources, 14:02-14:03".
Alerts are grouped by attack family and the entity that ties the campaign
together (victim for DDoS, scanner for scans, C2 server for botnets, ...),
then split into separate incidents when there is a quiet gap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CLASS_INFO
from .firewall import firewall_rules

# which entity ties a campaign together, per attack family
GROUP_KEY = {
    "DDoS": ("dst_ip",),
    "DoS": ("src_ip", "dst_ip"),
    "PortScan": ("src_ip",),
    "BruteForce": ("src_ip", "dst_ip"),
    "Botnet": ("dst_ip",),
    "WebAttack": ("src_ip", "dst_ip"),
    "Infiltration": ("src_ip",),
}
SEVERITY_RANK = {"critical": 3, "high": 2, "medium": 1, "none": 0}


def correlate_incidents(alerts: pd.DataFrame, gap_s: float = 60.0, min_flows: int = 3) -> list[dict]:
    """``alerts`` needs columns ts, src_ip, dst_ip, dst_port, pred_label, confidence."""
    incidents = []
    if alerts.empty:
        return incidents
    for cls, keys in GROUP_KEY.items():
        sub = alerts[alerts["pred_label"] == cls]
        if sub.empty:
            continue
        for key_vals, grp in sub.groupby(list(keys), sort=False):
            grp = grp.sort_values("ts")
            split = np.flatnonzero(np.diff(grp["ts"].to_numpy()) > gap_s) + 1
            for part in np.split(np.arange(len(grp)), split):
                g = grp.iloc[part]
                if len(g) < min_flows:
                    continue
                incidents.append(_make_incident(cls, keys, key_vals, g))
    incidents.sort(key=lambda i: (-SEVERITY_RANK[i["severity"]], -i["flows"]))
    for n, inc in enumerate(incidents, 1):
        inc["id"] = f"INC-{n:04d}"
        inc["firewall"] = firewall_rules(inc)
    return incidents


def _make_incident(cls: str, keys, key_vals, g: pd.DataFrame) -> dict:
    key_vals = key_vals if isinstance(key_vals, tuple) else (key_vals,)
    entity = dict(zip(keys, key_vals))
    sources = g["src_ip"].value_counts()
    targets = g["dst_ip"].value_counts()
    ports = g["dst_port"].value_counts()
    info = CLASS_INFO[cls]
    if cls == "DDoS":
        title = f"DDoS against {entity['dst_ip']} from {len(sources):,} sources"
    elif cls == "PortScan":
        title = f"Port scan by {entity['src_ip']} ({g['dst_port'].nunique():,} ports on {len(targets)} host(s))"
    elif cls == "Botnet":
        title = f"Botnet C2 beaconing to {entity['dst_ip']} from {len(sources)} internal host(s)"
    elif cls == "Infiltration":
        title = f"Lateral movement / exfiltration from {entity['src_ip']} to {len(targets)} host(s)"
    else:
        title = f"{cls} from {entity['src_ip']} against {entity['dst_ip']}"
    return {
        "class": cls,
        "title": title,
        "entity": entity,
        "start": float(g["ts"].min()),
        "end": float(g["ts"].max()),
        "flows": int(len(g)),
        "sources": int(len(sources)),
        "targets": int(len(targets)),
        "top_sources": sources.head(25).index.tolist(),
        "top_targets": targets.head(10).index.tolist(),
        "top_ports": [int(p) for p in ports.head(5).index],
        "mean_confidence": float(g["confidence"].mean()),
        "severity": info["severity"],
        "mitre": info["mitre"],
        "response": info["response"],
    }
