"""Graph-level evidence: human-readable structural facts around a flagged flow.

Feature attributions say *which properties of the flow* mattered. Analysts
also need *which relationship in the network* is unusual - e.g. "312 distinct
sources hit this host in 20 s" for DDoS. These statistics are computed from
the raw flows of the window, each compared against the window's own baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..graph.builder import WindowGraph


def _ratio(v: float, base: float) -> float:
    return (v + 1.0) / (base + 1.0)


def graph_evidence(df: pd.DataFrame, graph: WindowGraph, edge_id: int, max_items: int = 4,
                   lookback_s: float = 300.0) -> list[dict]:
    w = df.iloc[graph.flow_rows].reset_index(drop=True)
    src, dst = graph.edge_index
    s, d = int(src[edge_id]), int(dst[edge_id])
    s_ip, d_ip = graph.node_ips[s], graph.node_ips[d]
    row = w.iloc[edge_id]
    dur_s = max(graph.t1 - graph.t0, 1e-9)
    items: list[dict] = []

    def add(key, title, value, baseline, text, salience=None):
        r = _ratio(value, baseline)
        items.append({"key": key, "title": title, "value": float(value), "baseline": float(baseline),
                      "ratio": float(r), "text": text, "salience": float(abs(np.log(r)) if salience is None else salience)})

    pairs = pd.DataFrame({"s": src, "d": dst, "port": w["dst_port"].to_numpy()})
    # 1. fan-in to the destination
    in_sources = pairs.groupby("d")["s"].nunique()
    v = int(in_sources.get(d, 0))
    base = float(in_sources.median())
    n_in = int((dst == d).sum())
    add("fan_in", "Many sources converge on the destination", v, base,
        f"{d_ip} received {n_in:,} flows from {v:,} distinct sources in this {dur_s:.0f}s window "
        f"(typical host: {base:.0f} source{'s' if base != 1 else ''}).")
    # 2. fan-out of the source over hosts
    out_hosts = pairs.groupby("s")["d"].nunique()
    v = int(out_hosts.get(s, 0))
    base = float(out_hosts.median())
    add("fan_out_hosts", "Source contacts many hosts", v, base,
        f"{s_ip} contacted {v:,} distinct hosts in this window (typical host: {base:.0f}).")
    # 3. port fan-out towards this destination
    pair_ports = pairs.groupby(["s", "d"])["port"].nunique()
    v = int(pair_ports.get((s, d), 0))
    base = float(pair_ports.median())
    add("fan_out_ports", "Source probes many ports", v, base,
        f"{s_ip} used {v:,} different destination ports towards {d_ip} (typical pair: {base:.0f}).")
    # 4. repetition of the same service connection
    rep = pairs.groupby(["s", "d", "port"]).size()
    v = int(rep.get((s, d, int(row["dst_port"])), 0))
    base = float(rep.median())
    add("repetition", "Repeated connections to the same service", v, base,
        f"{s_ip} opened {v:,} connections to {d_ip}:{int(row['dst_port'])} in this window "
        f"(typical: {base:.0f}).")
    # 5. half-open / unanswered connections towards the destination
    unanswered = (w["bwd_pkts"].to_numpy() == 0) | ((w["syn"].to_numpy() > 0) & (w["ack"].to_numpy() <= 1))
    to_d = dst == d
    share = float(unanswered[to_d].mean()) if to_d.any() else 0.0
    glob = float(unanswered.mean())
    if share > 0.3 and share > 1.5 * glob:
        add("half_open", "Unanswered / half-open connections", share * 100, glob * 100,
            f"{share:.0%} of flows to {d_ip} got no proper response (window average {glob:.0%}).",
            salience=abs(np.log((share + 0.05) / (glob + 0.05))))
    # 6. periodic beaconing src -> dst (looks back `lookback_s` seconds, beacons are slow)
    seg = df["segment"].to_numpy() == graph.segment
    hist = df.loc[seg & (df["src_ip"].to_numpy() == s_ip) & (df["dst_ip"].to_numpy() == d_ip)
                  & (df["ts"].to_numpy() >= graph.t1 - lookback_s) & (df["ts"].to_numpy() <= graph.t1)]
    t = np.sort(hist["ts"].to_numpy())
    if len(t) >= 4:
        gaps = np.diff(t)
        cv = float(gaps.std() / max(gaps.mean(), 1e-9))
        if cv < 0.3:
            n_hosts = int(in_sources.get(d, 0))
            add("beacon", "Periodic beaconing", len(t), 0,
                f"{s_ip} contacted {d_ip} {len(t)} times in the last {lookback_s / 60:.0f} min at a near-constant "
                f"interval of {gaps.mean():.1f}s (variation {cv:.0%}); {n_hosts} host(s) talk to {d_ip} "
                f"in this window.", salience=3.0 * (1 - cv))
    # 7. unusual payload size for the service
    same_port = w["dst_port"].to_numpy() == row["dst_port"]
    fwd_len = w["fwd_bytes"].to_numpy() / np.maximum(w["fwd_pkts"].to_numpy(), 1)
    v = float(fwd_len[edge_id])
    base = float(np.median(fwd_len[same_port]))
    if v > 1.5 * base:
        add("payload", "Unusually large request payload", v, base,
            f"Mean request packet of this flow is {v:.0f} B vs {base:.0f} B for other port-{int(row['dst_port'])} flows.")
    # 8. long-lived slow connection
    durs = w["duration"].to_numpy()
    v, base = float(row["duration"]), float(np.median(durs[same_port]))
    if v > 5 * max(base, 0.05) and v > 5:
        add("slow", "Long-lived low-volume connection", v, base,
            f"This connection stayed open {v:.1f}s while carrying {int(row['fwd_pkts'] + row['bwd_pkts'])} packets "
            f"(typical port-{int(row['dst_port'])} flow: {base:.2f}s).")

    items = [i for i in items if i["ratio"] >= 2.0 or i["key"] in {"beacon", "half_open", "payload", "slow"}]
    items.sort(key=lambda i: -i["salience"])
    return items[:max_items]
