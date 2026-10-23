"""Command line interface:  python -m xgnn_ids <command> [options]

Commands
  generate   create a synthetic labelled flow dataset (CSV)
  train      run the full experiment (models, baselines, robustness, XAI) -> models/ + results/
  detect     classify flows from a CSV (CICFlowMeter / NetFlow / canonical) or a pcap file
  explain    explain one flow of a CSV / pcap
  live       live detection on a network interface (needs root + scapy)
  serve      start the REST API (FastAPI)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_input(path: str):
    from .data.loaders import load_flows

    if path.lower().endswith((".pcap", ".pcapng", ".cap")):
        from .capture.flowmeter import pcap_to_flows

        return pcap_to_flows(path)
    return load_flows(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="xgnn_ids", description="Explainable GNN network intrusion detection")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate synthetic traffic")
    g.add_argument("--profile", default="enterprise", choices=["enterprise", "iot"])
    g.add_argument("--duration", type=float, default=1800)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", default="data/synthetic_enterprise.csv")

    t = sub.add_parser("train", help="train and evaluate all models")
    t.add_argument("--data", nargs="*", help="flow CSV files (default: synthetic data)")
    t.add_argument("--quick", action="store_true", help="short run for smoke tests")
    t.add_argument("--models", nargs="*", help="subset of: mlp gcn gat egraphsage egraphsage_nostats")
    t.add_argument("--baselines", nargs="*", help="subset of: rf rf_graphstats hgb")
    t.add_argument("--out", default="results")
    t.add_argument("--models-dir", default="models")
    t.add_argument("--max-rows", type=int, help="rows per CSV (for very large datasets)")
    for flag in ("cross-domain", "zero-day", "adversarial", "xai"):
        t.add_argument(f"--no-{flag}", action="store_true")

    d = sub.add_parser("detect", help="detect intrusions in a CSV or pcap")
    d.add_argument("input")
    d.add_argument("--models-dir", default="models")
    d.add_argument("--model")
    d.add_argument("--out", default="alerts.csv")
    d.add_argument("--incidents", default="incidents.json")
    d.add_argument("--min-confidence", type=float, default=0.5)

    e = sub.add_parser("explain", help="explain one flow")
    e.add_argument("input")
    e.add_argument("--row", type=int, help="row of the flow (default: most confident alert)")
    e.add_argument("--models-dir", default="models")
    e.add_argument("--model")
    e.add_argument("--shap", action="store_true", help="also compute SHAP (slower)")

    lv = sub.add_parser("live", help="live capture")
    lv.add_argument("--iface")
    lv.add_argument("--window", type=float, default=20)
    lv.add_argument("--duration", type=float)
    lv.add_argument("--bpf", help="capture filter, e.g. 'tcp or udp'")
    lv.add_argument("--models-dir", default="models")
    lv.add_argument("--model")
    lv.add_argument("--out", default="live_alerts.jsonl")

    s = sub.add_parser("serve", help="REST API")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)

    a = ap.parse_args(argv)

    if a.cmd == "generate":
        from .data.synthetic import generate_traffic

        df = generate_traffic(a.profile, a.duration, a.seed)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(a.out, index=False)
        print(f"wrote {len(df):,} flows to {a.out}\n{df['label'].value_counts().to_string()}")
    elif a.cmd == "train":
        from .training.experiment import run_experiment

        run_experiment(a.data or None, a.out, a.models_dir, a.quick, a.models, a.baselines,
                       not a.no_cross_domain, not a.no_zero_day, not a.no_adversarial, not a.no_xai, a.max_rows)
    elif a.cmd == "detect":
        from .inference import Detector

        det = Detector.load(a.models_dir, a.model)
        res = det.detect(_load_input(a.input))
        res.df.to_csv(a.out, index=False)
        inc = det.incidents(res, a.min_confidence)
        Path(a.incidents).write_text(json.dumps(inc, indent=1))
        print(f"{len(res.df):,} flows, {len(res.alerts):,} alerts, {len(inc)} incidents  ->  {a.out}, {a.incidents}")
        for i in inc[:15]:
            print(f"  [{i['severity'].upper():8}] {i['id']} {i['title']} ({i['flows']} flows)")
    elif a.cmd == "explain":
        from .inference import Detector

        det = Detector.load(a.models_dir, a.model)
        res = det.detect(_load_input(a.input))
        row = a.row
        if row is None:
            alerts = res.alerts
            if alerts.empty:
                print("no alerts in this input")
                return 0
            row = int(alerts["confidence"].idxmax())
        methods = ("ig", "gnnexplainer", "counterfactual") + (("shap",) if a.shap else ())
        exp = det.explain(res, row, methods)
        print(exp["narrative"])
        for name, feats in exp["features"].items():
            print(f"\n{name} (ran in {exp['timings_ms'].get(name, 0):.0f} ms):")
            for f in feats[:6]:
                print(f"  {f['attribution']:+.4f}  {f['label']} = {f['value']}")
    elif a.cmd == "live":
        from .capture.live import run_live

        run_live(a.iface, a.models_dir, a.model, a.window, a.duration, a.bpf, a.out)
    elif a.cmd == "serve":
        import uvicorn

        uvicorn.run("api.main:app", host=a.host, port=a.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
