"""Loaders that normalise public IDS datasets to the canonical flow schema.

Supported formats (auto-detected from the column names):

* CICFlowMeter CSVs: CIC-IDS2017 ("GeneratedLabelledFlows"/"TrafficLabelling"
  version, which keeps IPs and timestamps), CSE-CIC-IDS2018, CICDDoS2019,
  and any CSV produced by CICFlowMeter on your own pcaps.
* NetFlow CSVs from the UQ "NF-*" family (NF-UNSW-NB15, NF-ToN-IoT,
  NF-BoT-IoT, NF-CSE-CIC-IDS2018, v2/v3).
* Our own canonical CSV (as written by ``xgnn-ids generate``).

Note: the CIC-IDS2017 "MachineLearningCVE" CSVs drop the IP and timestamp
columns, so no graph can be built from them; use the "TrafficLabelling" CSVs.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import CLASSES
from .features import CANONICAL_COLUMNS, coerce_numeric


def _norm(col: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(col).lower())


# normalised column name -> canonical column
CIC_ALIASES = {
    "sourceip": "src_ip", "srcip": "src_ip",
    "destinationip": "dst_ip", "dstip": "dst_ip",
    "sourceport": "src_port", "srcport": "src_port",
    "destinationport": "dst_port", "dstport": "dst_port",
    "protocol": "protocol",
    "timestamp": "timestamp",
    "flowduration": "duration_us",
    "totalfwdpackets": "fwd_pkts", "totfwdpkts": "fwd_pkts", "totalfwdpacket": "fwd_pkts",
    "totalbackwardpackets": "bwd_pkts", "totbwdpkts": "bwd_pkts", "totalbwdpackets": "bwd_pkts",
    "totalbwdpacket": "bwd_pkts",
    "totallengthoffwdpackets": "fwd_bytes", "totlenfwdpkts": "fwd_bytes", "totallengthoffwdpacket": "fwd_bytes",
    "fwdpacketslengthtotal": "fwd_bytes",
    "totallengthofbwdpackets": "bwd_bytes", "totlenbwdpkts": "bwd_bytes", "totallengthofbwdpacket": "bwd_bytes",
    "bwdpacketslengthtotal": "bwd_bytes",
    "synflagcount": "syn", "synflagcnt": "syn",
    "ackflagcount": "ack", "ackflagcnt": "ack",
    "finflagcount": "fin", "finflagcnt": "fin",
    "rstflagcount": "rst", "rstflagcnt": "rst",
    "pshflagcount": "psh", "pshflagcnt": "psh",
    "flowiatmean": "iat_mean_us",
    "label": "raw_label",
}

NETFLOW_ALIASES = {
    "ipv4srcaddr": "src_ip", "ipv4dstaddr": "dst_ip",
    "l4srcport": "src_port", "l4dstport": "dst_port",
    "protocol": "protocol",
    "inbytes": "fwd_bytes", "outbytes": "bwd_bytes",
    "inpkts": "fwd_pkts", "outpkts": "bwd_pkts",
    "tcpflags": "tcp_flags",
    "flowdurationmilliseconds": "duration_ms",
    "flowstartmilliseconds": "start_ms",
    "attack": "raw_label",
    "label": "binary_label",
}

# ordered (keyword, family) rules applied to the lower-cased raw label
LABEL_RULES: list[tuple[str, str]] = [
    ("benign", "BENIGN"), ("normal", "BENIGN"),
    ("ddos", "DDoS"), ("drdos", "DDoS"), ("syn", "DDoS"), ("udp-lag", "DDoS"), ("udplag", "DDoS"),
    ("loic", "DDoS"), ("hoic", "DDoS"), ("portmap", "DDoS"), ("ntp", "DDoS"), ("tftp", "DDoS"),
    ("ssdp", "DDoS"), ("snmp", "DDoS"), ("ldap", "DDoS"), ("mssql", "DDoS"), ("netbios", "DDoS"),
    ("web attack", "WebAttack"), ("sql", "WebAttack"), ("xss", "WebAttack"), ("injection", "WebAttack"),
    ("brute force -web", "WebAttack"), ("brute force -xss", "WebAttack"),
    ("dos", "DoS"), ("heartbleed", "DoS"), ("slowloris", "DoS"), ("hulk", "DoS"),
    ("portscan", "PortScan"), ("scanning", "PortScan"), ("reconnaissance", "PortScan"), ("scan", "PortScan"),
    ("patator", "BruteForce"), ("bruteforce", "BruteForce"), ("brute force", "BruteForce"),
    ("password", "BruteForce"),
    ("bot", "Botnet"), ("backdoor", "Botnet"), ("c&c", "Botnet"),
    ("infilt", "Infiltration"), ("exploit", "Infiltration"), ("shellcode", "Infiltration"),
    ("worm", "Infiltration"), ("ransomware", "Infiltration"), ("mitm", "Infiltration"),
    ("theft", "Infiltration"),
]


def map_label(raw: str) -> str:
    s = str(raw).strip().lower()
    if s in {"", "nan", "none"}:
        return "UNKNOWN"
    if s in {c.lower() for c in CLASSES}:
        return next(c for c in CLASSES if c.lower() == s)
    for key, fam in LABEL_RULES:
        if key in s:
            return fam
    return "UNKNOWN"


def detect_format(columns) -> str:
    norm = {_norm(c) for c in columns}
    if {"src_ip", "dst_ip", "ts"} <= set(columns):
        return "canonical"
    if {"ipv4srcaddr", "ipv4dstaddr"} <= norm:
        return "netflow"
    if {"srcip", "dstip"} <= norm or {"sourceip", "destinationip"} <= norm:
        return "cicflowmeter"
    raise ValueError(
        "Unrecognised flow CSV: need source/destination IP columns "
        "(CICFlowMeter 'Src IP'/'Dst IP', NetFlow 'IPV4_SRC_ADDR', or canonical 'src_ip'). "
        "Note that CIC-IDS2017 'MachineLearningCVE' files drop IPs - use the TrafficLabelling CSVs."
    )


def _rename(df: pd.DataFrame, aliases: dict) -> pd.DataFrame:
    mapping = {}
    for c in df.columns:
        key = _norm(c)
        if key in aliases and aliases[key] not in mapping.values():
            mapping[c] = aliases[key]
    return df.rename(columns=mapping)


def _parse_time(series: pd.Series) -> np.ndarray:
    """Timestamps -> float seconds; falls back to row order when unparsable."""
    t = pd.to_datetime(series, errors="coerce", format="mixed", dayfirst=True)
    if t.notna().mean() < 0.9:
        return np.arange(len(series), dtype=float) * 1e-3
    secs = (t - t.min()).dt.total_seconds().to_numpy()
    return np.nan_to_num(secs, nan=0.0)


def from_cicflowmeter(df: pd.DataFrame, segment: str) -> pd.DataFrame:
    df = _rename(df, CIC_ALIASES)
    out = pd.DataFrame()
    for c in ("src_ip", "dst_ip", "src_port", "dst_port", "protocol", "fwd_pkts", "bwd_pkts",
              "fwd_bytes", "bwd_bytes", "syn", "ack", "fin", "rst", "psh"):
        out[c] = df[c] if c in df.columns else 0
    out["duration"] = pd.to_numeric(df.get("duration_us", 0), errors="coerce") / 1e6
    out["iat_mean"] = pd.to_numeric(df.get("iat_mean_us", 0), errors="coerce") / 1e6
    out["ts"] = _parse_time(df["timestamp"]) if "timestamp" in df.columns else np.arange(len(df)) * 1e-3
    out["label"] = df["raw_label"].map(map_label) if "raw_label" in df.columns else "UNKNOWN"
    if "raw_label" in df.columns:
        out["attack_detail"] = df["raw_label"].astype(str).str.strip()
    out["segment"] = segment
    return out


def from_netflow(df: pd.DataFrame, segment: str) -> pd.DataFrame:
    df = _rename(df, NETFLOW_ALIASES)
    out = pd.DataFrame()
    for c in ("src_ip", "dst_ip", "src_port", "dst_port", "protocol", "fwd_pkts", "bwd_pkts",
              "fwd_bytes", "bwd_bytes"):
        out[c] = df[c] if c in df.columns else 0
    flags = pd.to_numeric(df.get("tcp_flags", 0), errors="coerce").fillna(0).astype(int)
    for name, bit in (("fin", 1), ("syn", 2), ("rst", 4), ("psh", 8), ("ack", 16)):
        out[name] = ((flags & bit) > 0).astype(int)
    out["duration"] = pd.to_numeric(df.get("duration_ms", 0), errors="coerce") / 1000.0
    total = pd.to_numeric(out["fwd_pkts"], errors="coerce") + pd.to_numeric(out["bwd_pkts"], errors="coerce")
    out["iat_mean"] = out["duration"] / np.maximum(total - 1, 1)
    if "start_ms" in df.columns:
        s = pd.to_numeric(df["start_ms"], errors="coerce").fillna(0)
        out["ts"] = (s - s.min()) / 1000.0
    else:  # v1/v2 NetFlow sets are time-ordered but carry no timestamp
        out["ts"] = np.arange(len(df)) * 1e-3
    if "raw_label" in df.columns:
        out["label"] = df["raw_label"].map(map_label)
        out["attack_detail"] = df["raw_label"].astype(str)
    elif "binary_label" in df.columns:
        out["label"] = np.where(pd.to_numeric(df["binary_label"], errors="coerce").fillna(0) > 0, "UNKNOWN", "BENIGN")
    else:
        out["label"] = "UNKNOWN"
    out["segment"] = segment
    return out


def normalise(df: pd.DataFrame, segment: str = "uploaded") -> pd.DataFrame:
    """Convert any supported flow table to the canonical schema (sorted by time)."""
    fmt = detect_format(df.columns)
    if fmt == "netflow":
        out = from_netflow(df, segment)
    elif fmt == "cicflowmeter":
        out = from_cicflowmeter(df, segment)
    else:
        out = df.copy()
        if "segment" not in out.columns:
            out["segment"] = segment
    out = coerce_numeric(out)
    out["label"] = out["label"].astype(str)
    extra = [c for c in ("attack_detail",) if c in out.columns]
    out = out[CANONICAL_COLUMNS + extra]
    return out.sort_values(["segment", "ts"], kind="stable").reset_index(drop=True)


def load_flows(paths, max_rows: int | None = None) -> pd.DataFrame:
    """Load one or more flow CSVs (each file becomes its own time segment)."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    frames = []
    for p in paths:
        p = Path(p)
        raw = pd.read_csv(p, nrows=max_rows, low_memory=False, encoding_errors="replace")
        frames.append(normalise(raw, segment=p.stem))
    return pd.concat(frames, ignore_index=True)
