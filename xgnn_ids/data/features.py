"""Canonical flow schema and feature engineering.

Every data source (synthetic generator, CIC-IDS2017/2018, CICDDoS2019,
NetFlow NF-* datasets, live capture) is normalised to CANONICAL_COLUMNS, and
from there to the same numeric edge-feature vector.
"""
from __future__ import annotations

import ipaddress
from functools import lru_cache

import numpy as np
import pandas as pd

CANONICAL_COLUMNS = [
    "ts",          # flow start, seconds (float, monotonic within a segment)
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",    # IANA number: 6 TCP, 17 UDP, 1 ICMP
    "duration",    # seconds
    "fwd_pkts",
    "bwd_pkts",
    "fwd_bytes",
    "bwd_bytes",
    "syn",         # number of packets carrying the flag (both directions)
    "ack",
    "fin",
    "rst",
    "psh",
    "iat_mean",    # mean packet inter-arrival time, seconds
    "label",       # unified class name (see config.CLASSES) or "UNKNOWN"
    "segment",     # capture/session identifier (file, day, profile ...)
]

NUMERIC_COLUMNS = [
    "ts", "src_port", "dst_port", "protocol", "duration", "fwd_pkts", "bwd_pkts",
    "fwd_bytes", "bwd_bytes", "syn", "ack", "fin", "rst", "psh", "iat_mean",
]

WEB_PORTS = {80, 443, 8080, 8443, 8000}
REMOTE_PORTS = {21, 22, 23, 3389, 5900}
SMB_PORTS = {135, 137, 138, 139, 445}

# (feature name, human readable label used in explanations)
EDGE_FEATURES: list[tuple[str, str]] = [
    ("log_duration_ms", "Flow duration"),
    ("log_fwd_pkts", "Forward packets"),
    ("log_bwd_pkts", "Backward (response) packets"),
    ("log_fwd_bytes", "Forward bytes"),
    ("log_bwd_bytes", "Backward (response) bytes"),
    ("log_bytes_per_s", "Byte rate"),
    ("log_pkts_per_s", "Packet rate"),
    ("log_fwd_pkt_len", "Mean forward packet size"),
    ("log_bwd_pkt_len", "Mean backward packet size"),
    ("log_iat_ms", "Mean inter-arrival time"),
    ("log_syn", "SYN flag count"),
    ("log_ack", "ACK flag count"),
    ("log_fin", "FIN flag count"),
    ("log_rst", "RST flag count"),
    ("log_psh", "PSH flag count"),
    ("no_response", "No response from destination"),
    ("bwd_fwd_byte_ratio", "Response/request byte ratio"),
    ("proto_tcp", "Protocol = TCP"),
    ("proto_udp", "Protocol = UDP"),
    ("proto_other", "Protocol = other (ICMP...)"),
    ("port_web", "Destination port is web (80/443/8080)"),
    ("port_remote", "Destination port is remote-login (21/22/23/3389)"),
    ("port_dns", "Destination port is DNS (53)"),
    ("port_smb", "Destination port is SMB/NetBIOS"),
    ("port_wellknown_other", "Other well-known port (<1024)"),
    ("port_registered", "Registered port (1024-49151)"),
    ("port_dynamic", "Dynamic/ephemeral port (>=49152)"),
]
EDGE_FEATURE_NAMES = [f for f, _ in EDGE_FEATURES]
EDGE_FEATURE_LABELS = dict(EDGE_FEATURES)

NODE_FEATURES: list[tuple[str, str]] = [
    ("log_out_flows", "Flows sent by host"),
    ("log_in_flows", "Flows received by host"),
    ("log_out_peers", "Distinct hosts contacted"),
    ("log_in_peers", "Distinct hosts contacting it"),
    ("log_out_dports", "Distinct destination ports used"),
    ("log_in_dports", "Distinct own ports targeted"),
    ("log_out_bytes", "Bytes sent"),
    ("log_in_bytes", "Bytes received"),
    ("syn_only_in_ratio", "Share of inbound SYN-only flows"),
    ("no_resp_out_ratio", "Share of outbound flows without response"),
    ("is_private", "Private (internal) address"),
]
NODE_FEATURE_NAMES = [f for f, _ in NODE_FEATURES]
NODE_FEATURE_LABELS = dict(NODE_FEATURES)


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Make numeric columns numeric, replace inf/NaN with 0 and clip negatives."""
    df = df.copy()
    for c in NUMERIC_COLUMNS:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for c in NUMERIC_COLUMNS:
        if c != "ts":
            df[c] = df[c].clip(lower=0)
    df["src_ip"] = df["src_ip"].astype(str)
    df["dst_ip"] = df["dst_ip"].astype(str)
    if "label" not in df.columns:
        df["label"] = "UNKNOWN"
    if "segment" not in df.columns:
        df["segment"] = "default"
    df["segment"] = df["segment"].astype(str)
    return df


def edge_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Turn canonical flow records into the raw (unscaled) edge feature matrix."""
    dur = df["duration"].to_numpy(float)
    fp = df["fwd_pkts"].to_numpy(float)
    bp = df["bwd_pkts"].to_numpy(float)
    fb = df["fwd_bytes"].to_numpy(float)
    bb = df["bwd_bytes"].to_numpy(float)
    proto = df["protocol"].to_numpy(int)
    port = df["dst_port"].to_numpy(int)

    dur_c = np.maximum(dur, 1e-3)
    total_pkts = fp + bp
    total_bytes = fb + bb
    feats = {
        "log_duration_ms": np.log1p(dur * 1000.0),
        "log_fwd_pkts": np.log1p(fp),
        "log_bwd_pkts": np.log1p(bp),
        "log_fwd_bytes": np.log1p(fb),
        "log_bwd_bytes": np.log1p(bb),
        "log_bytes_per_s": np.log1p(total_bytes / dur_c),
        "log_pkts_per_s": np.log1p(total_pkts / dur_c),
        "log_fwd_pkt_len": np.log1p(fb / np.maximum(fp, 1)),
        "log_bwd_pkt_len": np.log1p(bb / np.maximum(bp, 1)),
        "log_iat_ms": np.log1p(df["iat_mean"].to_numpy(float) * 1000.0),
        "log_syn": np.log1p(df["syn"].to_numpy(float)),
        "log_ack": np.log1p(df["ack"].to_numpy(float)),
        "log_fin": np.log1p(df["fin"].to_numpy(float)),
        "log_rst": np.log1p(df["rst"].to_numpy(float)),
        "log_psh": np.log1p(df["psh"].to_numpy(float)),
        "no_response": (bp == 0).astype(float),
        "bwd_fwd_byte_ratio": np.log1p(bb) - np.log1p(fb),
        "proto_tcp": (proto == 6).astype(float),
        "proto_udp": (proto == 17).astype(float),
        "proto_other": ((proto != 6) & (proto != 17)).astype(float),
    }
    web = np.isin(port, list(WEB_PORTS))
    remote = np.isin(port, list(REMOTE_PORTS))
    dns = port == 53
    smb = np.isin(port, list(SMB_PORTS))
    special = web | remote | dns | smb
    feats.update(
        {
            "port_web": web.astype(float),
            "port_remote": remote.astype(float),
            "port_dns": dns.astype(float),
            "port_smb": smb.astype(float),
            "port_wellknown_other": ((port < 1024) & ~special).astype(float),
            "port_registered": ((port >= 1024) & (port < 49152) & ~special).astype(float),
            "port_dynamic": ((port >= 49152) & ~special).astype(float),
        }
    )
    return np.stack([feats[n] for n in EDGE_FEATURE_NAMES], axis=1).astype(np.float32)


@lru_cache(maxsize=200_000)
def is_private_ip(ip: str) -> bool:
    try:
        return ipaddress.ip_address(ip).is_private
    except ValueError:
        return False


class Standardizer:
    """Mean/std scaler that can be serialised to JSON with the model."""

    def __init__(self, mean: np.ndarray | None = None, std: np.ndarray | None = None, clip: float = 10.0,
                 lo: np.ndarray | None = None, hi: np.ndarray | None = None):
        self.mean = mean
        self.std = std
        self.clip = clip
        self.lo = lo   # 0.5th / 99.5th percentile of the scaled training data:
        self.hi = hi   # the "realistic range" used to bound counterfactuals

    def fit(self, x: np.ndarray) -> "Standardizer":
        self.mean = x.mean(axis=0)
        std = x.std(axis=0)
        self.std = np.where(std < 1e-6, 1.0, std)
        z = self.transform(x)
        self.lo, self.hi = np.quantile(z, 0.005, axis=0), np.quantile(z, 0.995, axis=0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean) / self.std
        return np.clip(z, -self.clip, self.clip).astype(np.float32)

    def to_dict(self) -> dict:
        d = {"mean": self.mean.tolist(), "std": self.std.tolist(), "clip": self.clip}
        if self.lo is not None:
            d.update(lo=np.asarray(self.lo).tolist(), hi=np.asarray(self.hi).tolist())
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Standardizer":
        arr = (lambda k: np.asarray(d[k], dtype=np.float64) if k in d else None)  # noqa: E731
        return cls(arr("mean"), arr("std"), d.get("clip", 10.0), arr("lo"), arr("hi"))
