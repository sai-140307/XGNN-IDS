"""Tests for data generation, loaders, graph construction and analysis (no PyTorch needed)."""
import numpy as np
import pandas as pd
import pytest

from xgnn_ids.analysis.drift import DriftMonitor
from xgnn_ids.analysis.incidents import correlate_incidents
from xgnn_ids.capture.flowmeter import ACK, FIN, RST, SYN, FlowMeter, Packet
from xgnn_ids.config import CLASSES, GraphConfig
from xgnn_ids.data.features import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES, edge_feature_matrix
from xgnn_ids.data.loaders import detect_format, map_label, normalise
from xgnn_ids.data.synthetic import generate_traffic, inject_attack
from xgnn_ids.graph.builder import build_graphs, chronological_split


@pytest.fixture(scope="module")
def flows():
    return generate_traffic("enterprise", duration_s=600, seed=1)


def test_generator_covers_all_classes(flows):
    assert set(flows["label"]) == set(CLASSES)
    assert flows["ts"].is_monotonic_increasing
    assert (flows[["fwd_pkts", "fwd_bytes", "duration"]] >= 0).all().all()


def test_generator_is_deterministic():
    a = generate_traffic("iot", duration_s=120, seed=3)
    b = generate_traffic("iot", duration_s=120, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_inject_attack_targets_same_network():
    base = generate_traffic("enterprise", 200, seed=4, attacks=[])
    assert set(base["label"]) == {"BENIGN"}
    out = inject_attack(base, "DDoS", 50, 30, seed=4, attack_seed=0)
    ddos = out[out["label"] == "DDoS"]
    assert len(ddos) > 100 and ddos["dst_ip"].nunique() == 1
    assert ddos["dst_ip"].iloc[0] in set(base["dst_ip"])


def test_edge_features_finite(flows):
    x = edge_feature_matrix(flows)
    assert x.shape == (len(flows), len(EDGE_FEATURE_NAMES))
    assert np.isfinite(x).all()


@pytest.mark.parametrize("raw,expected", [
    ("BENIGN", "BENIGN"), ("DDoS", "DDoS"), ("DoS Hulk", "DoS"), ("DoS slowloris", "DoS"),
    ("PortScan", "PortScan"), ("SSH-Patator", "BruteForce"), ("FTP-Patator", "BruteForce"), ("Bot", "Botnet"),
    ("Web Attack \x96 Sql Injection", "WebAttack"), ("Web Attack – XSS", "WebAttack"), ("Infiltration", "Infiltration"),
    ("DrDoS_LDAP", "DDoS"), ("DDOS attack-HOIC", "DDoS"), ("Brute Force -Web", "WebAttack"),
    ("scanning", "PortScan"), ("password", "BruteForce"), ("Infilteration", "Infiltration"), ("Fuzzers", "UNKNOWN"),
])
def test_label_mapping(raw, expected):
    assert map_label(raw) == expected


def test_cicflowmeter_loader():
    raw = pd.DataFrame({
        " Source IP": ["10.0.0.1", "10.0.0.2"], " Source Port": [5000, 5001],
        " Destination IP": ["10.0.0.9", "10.0.0.9"], " Destination Port": [80, 80], " Protocol": [6, 6],
        " Timestamp": ["07/07/2017 03:30:00", "07/07/2017 03:30:05"], " Flow Duration": [1_000_000, 2_000_000],
        " Total Fwd Packets": [3, 4], " Total Backward Packets": [2, 0], "Total Length of Fwd Packets": [300, 400],
        " Total Length of Bwd Packets": [1000, 0], " SYN Flag Count": [1, 1], " ACK Flag Count": [4, 0],
        "FIN Flag Count": [1, 0], " RST Flag Count": [0, 0], " PSH Flag Count": [1, 0],
        " Flow IAT Mean": [250000, 500000], " Label": ["BENIGN", "DDoS"],
    })
    assert detect_format(raw.columns) == "cicflowmeter"
    df = normalise(raw, "day1")
    assert list(df["label"]) == ["BENIGN", "DDoS"]
    assert df["duration"].tolist() == [1.0, 2.0]
    assert df["ts"].tolist() == [0.0, 5.0]


def test_netflow_loader():
    raw = pd.DataFrame({
        "IPV4_SRC_ADDR": ["1.1.1.1"], "L4_SRC_PORT": [1234], "IPV4_DST_ADDR": ["2.2.2.2"], "L4_DST_PORT": [22],
        "PROTOCOL": [6], "IN_BYTES": [500], "OUT_BYTES": [800], "IN_PKTS": [10], "OUT_PKTS": [9],
        "TCP_FLAGS": [SYN | ACK | FIN], "FLOW_DURATION_MILLISECONDS": [1500], "Label": [1], "Attack": ["password"],
    })
    df = normalise(raw)
    r = df.iloc[0]
    assert r["label"] == "BruteForce" and r["syn"] == 1 and r["rst"] == 0 and r["duration"] == 1.5


def test_canonical_csv_roundtrip(flows, tmp_path):
    path = tmp_path / "flows.csv"
    flows.to_csv(path, index=False)
    from xgnn_ids.data.loaders import load_flows

    back = load_flows(path)
    assert detect_format(flows.columns) == "canonical"
    assert len(back) == len(flows)
    assert back["ts"].max() == pytest.approx(flows["ts"].max())
    assert (back["label"] == flows["label"]).all()


def test_machine_learning_csv_without_ips_is_rejected():
    with pytest.raises(ValueError, match="TrafficLabelling"):
        detect_format([" Destination Port", " Flow Duration", " Label"])


def test_graph_builder(flows):
    graphs = build_graphs(flows, GraphConfig(window_s=20, max_edges=2000))
    assert sum(g.num_edges for g in graphs) == len(flows)
    rows = np.concatenate([g.flow_rows for g in graphs])
    assert len(np.unique(rows)) == len(flows)
    for g in graphs:
        assert g.x.shape == (g.num_nodes, len(NODE_FEATURE_NAMES))
        assert g.edge_index.max() < g.num_nodes
        assert g.num_edges <= 2000
        w = flows.iloc[g.flow_rows]
        assert (g.node_ips[g.edge_index[0]] == w["src_ip"].to_numpy()).all()


def test_chronological_split_has_no_time_overlap(flows):
    tr, va, te = chronological_split(build_graphs(flows))
    assert max(g.t0 for g in tr) < min(g.t0 for g in va) <= max(g.t0 for g in va) < min(g.t0 for g in te)


def test_flowmeter_tcp_session():
    m = FlowMeter()
    m.add(Packet(0.00, "a", "b", 1000, 80, 6, 60, SYN))
    m.add(Packet(0.01, "b", "a", 80, 1000, 6, 60, SYN | ACK))
    m.add(Packet(0.02, "a", "b", 1000, 80, 6, 500, ACK))
    m.add(Packet(0.05, "a", "b", 1000, 80, 6, 52, FIN | ACK))
    m.add(Packet(0.06, "b", "a", 80, 1000, 6, 52, FIN | ACK))
    m.add(Packet(0.10, "c", "b", 2000, 22, 6, 60, SYN))
    m.add(Packet(0.11, "b", "c", 22, 2000, 6, 60, RST | ACK))
    out = {r["src_ip"]: r for r in m.sweep(1.0)}
    a = out["a"]
    assert (a["fwd_pkts"], a["bwd_pkts"], a["fwd_bytes"], a["syn"], a["fin"]) == (3, 2, 612, 2, 2)
    assert out["c"]["rst"] == 1 and out["c"]["bwd_pkts"] == 1


def test_incident_correlation_and_firewall(flows):
    alerts = flows[flows["label"] != "BENIGN"].assign(pred_label=lambda d: d["label"], confidence=0.9)
    inc = correlate_incidents(alerts)
    classes = {i["class"] for i in inc}
    assert {"DDoS", "PortScan", "Botnet"} <= classes
    ddos = next(i for i in inc if i["class"] == "DDoS")
    assert ddos["sources"] > 50 and ddos["severity"] == "critical"
    rules = ddos["firewall"]["iptables"]
    assert any("hashlimit" in r for r in rules)
    name = [r for r in rules if "--hashlimit-name" in r][0].split("--hashlimit-name ")[1].split()[0]
    assert len(name) <= 15


def test_drift_monitor_detects_new_network(flows):
    benign = edge_feature_matrix(flows[flows["label"] == "BENIGN"])
    dm = DriftMonitor.fit(benign[: len(benign) // 2])
    same = dm.psi(benign[len(benign) // 2:])[0]
    iot = generate_traffic("iot", 300, seed=9)
    other = dm.psi(edge_feature_matrix(iot[iot["label"] == "BENIGN"]))[0]
    assert same < 0.1 < other
    assert DriftMonitor.from_dict(dm.to_dict()).psi(benign[:500])[0] == pytest.approx(dm.psi(benign[:500])[0])


def test_console_payload_is_compact_and_serialisable(flows):
    """The custom console's data contract, built from ground truth as stand-in predictions."""
    import json
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
    from payload import MAX_ATTACK_NODES, MAX_INTERNAL, MAX_NODES, build_payload

    df = flows.reset_index(drop=True).copy()
    df["pred_label"], df["confidence"] = df["label"], 0.95
    graphs = build_graphs(df)
    rows = []
    for gi, g in enumerate(graphs):
        p = df["pred_label"].to_numpy()[g.flow_rows]
        rec = {"window": gi, "t0": g.t0, "t1": g.t1, "flows": g.num_edges, "hosts": g.num_nodes,
               "alerts": int((p != "BENIGN").sum()), "latency_ms": 1.0, "drift_psi": 0.01}
        rec.update({c: int((p == c).sum()) for c in CLASSES[1:]})
        rows.append(rec)
    res = SimpleNamespace(df=df, graphs=graphs, windows=pd.DataFrame(rows))
    alerts = df[df["pred_label"] != "BENIGN"]
    payload = build_payload(res, correlate_incidents(alerts), 0.5, {"model": "test", "window_s": 20})
    text = json.dumps(payload)
    assert len(text) < 2_000_000
    assert len(payload["graphs"]) == len(graphs) == len(payload["windows"])
    for g in payload["graphs"]:
        assert len(g["nodes"]) <= MAX_INTERNAL + MAX_NODES + MAX_ATTACK_NODES
        n = len(g["nodes"])
        assert all(0 <= s < n and 0 <= t < n for s, t, *_ in g["links"])
    assert payload["incidents"] and all(0 <= i["w0"] <= i["w1"] < len(graphs) for i in payload["incidents"])
    assert payload["truth"]["detection_rate"] == 1.0


def test_home_profile_has_realistic_benign_mix():
    df = generate_traffic("home", 300, seed=2, episodes_per_attack=1)
    kinds = set(df.loc[df["label"] == "BENIGN", "attack_detail"])
    assert {"home-browsing", "home-dns", "home-quic", "home-discovery", "home-telemetry"} <= kinds
    # page loads make one laptop fan out to many destinations within a window
    w = df[(df["ts"] < 20) & (df["attack_detail"] == "home-browsing")]
    assert w.groupby("src_ip")["dst_ip"].nunique().max() >= 5


def test_flowmeter_orients_mid_connection_flows_client_to_server():
    m = FlowMeter()
    m.add(Packet(0.0, "142.250.9.9", "192.168.1.10", 443, 51111, 6, 1400, ACK))   # server reply seen first
    m.add(Packet(0.1, "192.168.1.10", "142.250.9.9", 51111, 443, 6, 60, ACK))
    (row,) = m.flush()
    assert (row["src_ip"], row["dst_port"], row["fwd_pkts"], row["bwd_pkts"]) == ("192.168.1.10", 443, 1, 1)


def test_live_session_assembles_and_slides():
    from xgnn_ids.capture.live_session import LiveSession

    s = LiveSession(window_s=20, keep_windows=3)
    t0 = s.t0
    for k in range(4):
        s.feed(Packet(t0 + 1 + k * 0.01, "192.168.1.10", "142.250.1.1", 50000, 443, 6, 100, ACK))
    df = s.collect(now=t0 + 30)
    assert len(df) == 1 and df["segment"].iloc[0] == "live" and df["label"].iloc[0] == "UNKNOWN"
    s.feed(Packet(t0 + 75, "192.168.1.11", "8.8.8.8", 40000, 53, 17, 70))
    df = s.collect(now=t0 + 100)
    assert s.origin(t0 + 100) == 60.0
    assert len(df) == 1 and df["ts"].min() >= 60.0          # the old flow slid out of the kept windows
    graphs = build_graphs(df, GraphConfig(window_s=20), origin=s.origin(t0 + 100))
    assert graphs[0].t0 == 60.0


def test_low_confidence_predictions_are_not_alerts():
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
    from payload import effective_labels

    df = pd.DataFrame({"pred_label": ["Botnet", "DDoS", "BENIGN"], "confidence": [0.53, 0.99, 0.7]})
    eff = effective_labels(SimpleNamespace(df=df), 0.9)
    assert list(eff) == ["BENIGN", "DDoS", "BENIGN"]


def test_console_payload_for_empty_live_capture():
    """Live capture before the first flow completes: no flows, no labels -> an empty map, not a crash."""
    from types import SimpleNamespace

    from app.payload import build_payload
    from xgnn_ids.analysis.incidents import correlate_incidents
    from xgnn_ids.config import CLASSES
    from xgnn_ids.data.loaders import CANONICAL_COLUMNS

    df = pd.DataFrame(columns=CANONICAL_COLUMNS + ["pred_label", "confidence", "uncertain", "window"])
    windows = pd.DataFrame(columns=["window", "t0", "t1", "segment", "flows", "hosts", "alerts", "latency_ms",
                                    "drift_psi", *CLASSES[1:]])
    res = SimpleNamespace(df=df, graphs=[], windows=windows, alerts=df)
    p = build_payload(res, correlate_incidents(df), 0.9, {"source": "Live capture"})
    assert p["truth"] is None and p["windows"] == [] and p["incidents"] == [] and p["meta"]["flows"] == 0

    live = df.copy()                      # live flows carry label UNKNOWN: no accuracy is shown
    assert build_payload(SimpleNamespace(df=live, graphs=[], windows=windows, alerts=live), [], 0.9, {})["truth"] is None
