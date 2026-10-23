"""Live intrusion detection on a network interface (requires scapy + root/CAP_NET_RAW).

    sudo python -m xgnn_ids live --iface eth0 --window 20

Packets are aggregated into flows by :class:`FlowMeter`; every ``window``
seconds the finished flows are turned into a graph, classified, correlated
into incidents and printed. Alerts are appended to a JSON-lines file that the
dashboard (or a SIEM) can tail.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from ..inference import Detector
from .flowmeter import FlowMeter, scapy_to_packet


def run_live(iface: str | None, models_dir: str = "models", model: str | None = None, window_s: float = 20.0,
             duration_s: float | None = None, bpf: str | None = None, out: str = "live_alerts.jsonl",
             min_confidence: float = 0.6) -> None:
    from scapy.all import AsyncSniffer  # noqa: WPS433

    det = Detector.load(models_dir, model)
    det.graph_cfg.window_s = window_s
    meter = FlowMeter(idle_timeout=min(5.0, window_s / 2), active_timeout=window_s)
    t_start = time.time()

    def on_packet(pkt):
        p = scapy_to_packet(pkt)
        if p is not None:
            meter.add(p)

    sniffer = AsyncSniffer(iface=iface, prn=on_packet, store=False, filter=bpf)
    sniffer.start()
    print(f"[xgnn-ids] sniffing on {iface or 'default interface'} with {det.display_name}; "
          f"window {window_s:.0f}s; alerts -> {out}  (Ctrl+C to stop)")
    out_path = Path(out)
    try:
        while duration_s is None or time.time() - t_start < duration_s:
            time.sleep(window_s)
            rows = meter.sweep(time.time())
            if not rows:
                print(f"[{time.strftime('%H:%M:%S')}] no completed flows")
                continue
            df = pd.DataFrame(rows)
            df["ts"] = df["ts"] - t_start
            df["segment"] = "live"
            res = det.detect(df)
            alerts = res.alerts[res.alerts["confidence"] >= min_confidence]
            incidents = det.incidents(res, min_confidence)
            print(f"[{time.strftime('%H:%M:%S')}] {len(df):4d} flows, {len(alerts):4d} alerts, "
                  f"{len(incidents)} incident(s)")
            for inc in incidents:
                print(f"   !! [{inc['severity'].upper()}] {inc['title']}  ({inc['flows']} flows, "
                      f"conf {inc['mean_confidence']:.2f})  MITRE {inc['mitre']}")
            with out_path.open("a") as fh:
                for inc in incidents:
                    fh.write(json.dumps({"time": time.time(), **{k: v for k, v in inc.items()}}) + "\n")
    except KeyboardInterrupt:
        pass
    finally:
        sniffer.stop()
        print("[xgnn-ids] stopped")
