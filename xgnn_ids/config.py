"""Central configuration: class taxonomy, MITRE ATT&CK mapping and defaults."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

# Unified attack-family taxonomy used across every dataset this project reads.
CLASSES: list[str] = [
    "BENIGN",
    "DDoS",
    "DoS",
    "PortScan",
    "BruteForce",
    "Botnet",
    "WebAttack",
    "Infiltration",
]
ATTACK_CLASSES: list[str] = [c for c in CLASSES if c != "BENIGN"]
BENIGN_ID = 0

# Analyst-facing context for every class: what it is, MITRE ATT&CK technique,
# and a first-response recommendation shown with each alert.
CLASS_INFO: dict[str, dict[str, str]] = {
    "BENIGN": {
        "description": "Normal traffic.",
        "mitre": "-",
        "response": "No action required.",
        "severity": "none",
    },
    "DDoS": {
        "description": "Distributed denial of service: many sources flood one target.",
        "mitre": "T1498 Network Denial of Service / T1499 Endpoint DoS",
        "response": "Enable upstream rate-limiting / scrubbing, apply SYN cookies, "
        "block or geo-filter top source prefixes, scale the targeted service.",
        "severity": "critical",
    },
    "DoS": {
        "description": "Denial of service from a single (or few) source(s), e.g. HTTP flood, Slowloris.",
        "mitre": "T1499 Endpoint Denial of Service",
        "response": "Block the source IP at the edge, tune web-server connection "
        "timeouts and per-client connection limits.",
        "severity": "high",
    },
    "PortScan": {
        "description": "Reconnaissance: one host probes many ports/services.",
        "mitre": "T1046 Network Service Discovery",
        "response": "Block the scanner, verify which services responded as open, "
        "watch the source for follow-up exploitation.",
        "severity": "medium",
    },
    "BruteForce": {
        "description": "Repeated authentication attempts against SSH/FTP/RDP/Telnet.",
        "mitre": "T1110 Brute Force",
        "response": "Lock out / fail2ban the source, enforce key-based auth or MFA, "
        "audit the targeted accounts for successful logins.",
        "severity": "high",
    },
    "Botnet": {
        "description": "Infected hosts beaconing to a command-and-control server.",
        "mitre": "T1071 Application Layer Protocol (C2)",
        "response": "Isolate the internal hosts, block the C2 address/domain, "
        "image the hosts for forensic analysis.",
        "severity": "critical",
    },
    "WebAttack": {
        "description": "Application-layer exploitation (SQL injection, XSS, web brute force).",
        "mitre": "T1190 Exploit Public-Facing Application",
        "response": "Inspect web-server logs for the request payloads, enable WAF "
        "rules, patch / sanitise the vulnerable endpoint.",
        "severity": "high",
    },
    "Infiltration": {
        "description": "Compromised internal host moving laterally or exfiltrating data.",
        "mitre": "T1021 Remote Services / T1570 Lateral Tool Transfer",
        "response": "Quarantine the internal source host, reset its credentials, "
        "review east-west traffic for other compromised hosts.",
        "severity": "critical",
    },
}


@dataclass
class GraphConfig:
    """How flow records are turned into temporal graphs."""

    window_s: float = 20.0        # length of a graph snapshot in seconds
    max_edges: int = 6000         # larger windows are split into chunks
    use_node_stats: bool = True   # attach per-window host statistics as node features


@dataclass
class TrainConfig:
    hidden: int = 64
    layers: int = 2
    heads: int = 4
    dropout: float = 0.1
    lr: float = 3e-3
    weight_decay: float = 1e-4
    epochs: int = 40
    patience: int = 8
    batch_edges: int = 25000
    seed: int = 42


@dataclass
class ExperimentConfig:
    graph: GraphConfig = field(default_factory=GraphConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    split: tuple[float, float, float] = (0.6, 0.2, 0.2)

    def to_dict(self) -> dict:
        return asdict(self)
