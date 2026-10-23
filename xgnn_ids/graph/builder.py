"""Turn canonical flow tables into directed temporal graph snapshots.

Each snapshot (window) is a directed multigraph:

* nodes  = IP addresses active in the window,
* edges  = individual flows (src -> dst), carrying the flow feature vector,
* node features = per-window host statistics (fan-in/fan-out, distinct peers,
  distinct ports, bytes, share of half-open connections ...).

Edge-level classification on these graphs is the E-GraphSAGE formulation
(Lo et al., 2022) used by Baahmed et al. (2025) and Luša et al. (2025).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import CLASSES, GraphConfig
from ..data.features import NODE_FEATURE_NAMES, edge_feature_matrix, is_private_ip

CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}


@dataclass
class WindowGraph:
    window_id: int
    segment: str
    t0: float
    t1: float
    node_ips: np.ndarray        # [N] object
    x: np.ndarray               # [N, F_node] raw node features
    edge_index: np.ndarray      # [2, E] int64
    edge_attr: np.ndarray       # [E, F_edge] raw edge features
    y: np.ndarray               # [E] int64, -1 = unknown label
    flow_rows: np.ndarray       # [E] row positions in the source DataFrame

    @property
    def num_nodes(self) -> int:
        return len(self.node_ips)

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]


def node_statistics(src: np.ndarray, dst: np.ndarray, n: int, df: pd.DataFrame, ips: np.ndarray) -> np.ndarray:
    """Per-host statistics computed only from the flows inside the window."""
    out_flows = np.bincount(src, minlength=n).astype(float)
    in_flows = np.bincount(dst, minlength=n).astype(float)

    pairs = np.unique(np.stack([src, dst], 1), axis=0)
    out_peers = np.bincount(pairs[:, 0], minlength=n)
    in_peers = np.bincount(pairs[:, 1], minlength=n)

    dport = df["dst_port"].to_numpy(np.int64)
    sp = np.unique(np.stack([src, dport], 1), axis=0)
    out_dports = np.bincount(sp[:, 0], minlength=n)
    dp = np.unique(np.stack([dst, dport], 1), axis=0)
    in_dports = np.bincount(dp[:, 0], minlength=n)

    fb = df["fwd_bytes"].to_numpy(float)
    bb = df["bwd_bytes"].to_numpy(float)
    out_bytes = np.bincount(src, weights=fb, minlength=n) + np.bincount(dst, weights=bb, minlength=n)
    in_bytes = np.bincount(dst, weights=fb, minlength=n) + np.bincount(src, weights=bb, minlength=n)

    syn_only = ((df["syn"].to_numpy() > 0) & (df["bwd_pkts"].to_numpy() <= 1) &
                (df["fwd_pkts"].to_numpy() <= 3) & (df["psh"].to_numpy() == 0)).astype(float)
    syn_only_in = np.bincount(dst, weights=syn_only, minlength=n) / np.maximum(in_flows, 1)
    no_resp = (df["bwd_pkts"].to_numpy() == 0).astype(float)
    no_resp_out = np.bincount(src, weights=no_resp, minlength=n) / np.maximum(out_flows, 1)
    private = np.array([is_private_ip(ip) for ip in ips], dtype=float)

    feats = {
        "log_out_flows": np.log1p(out_flows),
        "log_in_flows": np.log1p(in_flows),
        "log_out_peers": np.log1p(out_peers),
        "log_in_peers": np.log1p(in_peers),
        "log_out_dports": np.log1p(out_dports),
        "log_in_dports": np.log1p(in_dports),
        "log_out_bytes": np.log1p(out_bytes),
        "log_in_bytes": np.log1p(in_bytes),
        "syn_only_in_ratio": syn_only_in,
        "no_resp_out_ratio": no_resp_out,
        "is_private": private,
    }
    return np.stack([feats[k] for k in NODE_FEATURE_NAMES], 1).astype(np.float32)


def build_window(df: pd.DataFrame, rows: np.ndarray, window_id: int, segment: str, t0: float, t1: float,
                 edge_feats: np.ndarray | None = None) -> WindowGraph:
    w = df.iloc[rows]
    codes, ips = pd.factorize(pd.concat([w["src_ip"], w["dst_ip"]], ignore_index=True), sort=False)
    e = len(w)
    src, dst = codes[:e].astype(np.int64), codes[e:].astype(np.int64)
    ips = np.asarray(ips, dtype=object)
    x = node_statistics(src, dst, len(ips), w, ips)
    ea = edge_feats[rows] if edge_feats is not None else edge_feature_matrix(w)
    y = w["label"].map(CLASS_TO_ID).fillna(-1).to_numpy(np.int64)
    return WindowGraph(window_id, segment, float(t0), float(t1), ips, x, np.stack([src, dst]), ea, y,
                       np.asarray(rows, dtype=np.int64))


def build_graphs(df: pd.DataFrame, cfg: GraphConfig | None = None) -> list[WindowGraph]:
    """Slice a canonical flow table into time windows and build one graph per window.

    Windows are computed per segment. Windows with more than ``cfg.max_edges``
    flows are split into consecutive chunks (keeps memory bounded for DDoS peaks).
    """
    cfg = cfg or GraphConfig()
    df = df.reset_index(drop=True)
    edge_feats = edge_feature_matrix(df)
    graphs: list[WindowGraph] = []
    wid = 0
    for seg, idx in df.groupby("segment", sort=False).indices.items():
        idx = np.asarray(idx)
        ts = df["ts"].to_numpy(float)[idx]
        order = np.argsort(ts, kind="stable")
        idx, ts = idx[order], ts[order]
        start = ts.min() if len(ts) else 0.0
        win = np.floor((ts - start) / cfg.window_s).astype(np.int64)
        bounds = np.flatnonzero(np.diff(win)) + 1
        for chunk in np.split(np.arange(len(idx)), bounds):
            if len(chunk) == 0:
                continue
            k = win[chunk[0]]
            t0, t1 = start + k * cfg.window_s, start + (k + 1) * cfg.window_s
            for sub in np.array_split(chunk, int(np.ceil(len(chunk) / cfg.max_edges))):
                graphs.append(build_window(df, idx[sub], wid, str(seg), t0, t1, edge_feats))
                wid += 1
    return graphs


def chronological_split(graphs: list[WindowGraph], fractions=(0.6, 0.2, 0.2)):
    """Split windows by time *within each segment* (train on the past, test on the future).

    Splitting per segment keeps every capture day/file represented in each split
    (important for CIC-IDS2017 where each day contains different attacks) while
    still avoiding temporal leakage inside a segment.
    """
    train, val, test = [], [], []
    by_seg: dict[str, list[WindowGraph]] = {}
    for g in graphs:
        by_seg.setdefault(g.segment, []).append(g)
    c1, c2 = fractions[0], fractions[0] + fractions[1]
    for seg_graphs in by_seg.values():
        t_lo = min(g.t0 for g in seg_graphs)
        t_hi = max(g.t1 for g in seg_graphs)
        span = max(t_hi - t_lo, 1e-9)
        for g in seg_graphs:
            pos = (g.t0 - t_lo) / span
            (train if pos < c1 else val if pos < c2 else test).append(g)
    return train, val, test


def graph_summary(graphs: list[WindowGraph]) -> dict:
    ys = np.concatenate([g.y for g in graphs]) if graphs else np.array([], int)
    counts = {c: int((ys == i).sum()) for i, c in enumerate(CLASSES)}
    return {
        "windows": len(graphs),
        "flows": int(sum(g.num_edges for g in graphs)),
        "mean_nodes": float(np.mean([g.num_nodes for g in graphs])) if graphs else 0.0,
        "mean_edges": float(np.mean([g.num_edges for g in graphs])) if graphs else 0.0,
        "class_counts": counts,
    }
