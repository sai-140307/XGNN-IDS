"""End-to-end experiment: data -> graphs -> models -> evaluation -> XAI -> report.

Reproduces every number in RESULTS.md:

    python -m xgnn_ids train                 # full run on synthetic data
    python -m xgnn_ids train --quick         # ~2 minute smoke run
    python -m xgnn_ids train --data a.csv b.csv   # real CIC/NetFlow CSVs
"""
from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..analysis.drift import DriftMonitor
from ..config import ATTACK_CLASSES, CLASSES, ExperimentConfig
from ..data.loaders import load_flows
from ..data.synthetic import generate_traffic
from ..explain.evaluation import evaluate_explainers
from ..explain.report import explain_flow
from ..graph.builder import build_graphs, chronological_split, graph_summary
from ..inference import ARTIFACTS
from ..models.gnn import MODEL_DISPLAY, count_parameters, num_hops
from . import plots
from .data import Preprocessor
from .metrics import markdown_table
from .robustness import pgd_evasion, zero_day_experiment
from .trainer import (BASELINES, evaluate_baseline, evaluate_model, finetune_gnn, predict_proba, train_baseline,
                      train_gnn)

GNN_MODELS = ["mlp", "gcn", "gat", "egraphsage", "egraphsage_nostats"]
DISPLAY = {**MODEL_DISPLAY, **{k: v[0] for k, v in BASELINES.items()}}


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def run_experiment(data_paths=None, out_dir="results", models_dir="models", quick=False,
                   models=None, baselines=None, cross_domain=True, zero_day=True, adversarial=True,
                   xai=True, max_rows=None, duration_s=None, log=print) -> dict:
    t_all = time.perf_counter()
    cfg = ExperimentConfig()
    if quick:
        cfg.train.epochs, cfg.train.patience = 12, 4
    out, mdir = Path(out_dir), Path(models_dir)
    out.mkdir(parents=True, exist_ok=True)
    mdir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    models = models or GNN_MODELS
    baselines = list(BASELINES) if baselines is None else baselines
    synthetic = not data_paths

    # ---------------------------------------------------------------- data
    log("[1/8] loading data")
    if synthetic:
        df = generate_traffic("enterprise", duration_s=duration_s or (900 if quick else 1800), seed=0)
        source = "synthetic enterprise network (seed 0)"
    else:
        df = load_flows(data_paths, max_rows=max_rows)
        source = ", ".join(Path(p).name for p in data_paths)
    graphs = build_graphs(df, cfg.graph)
    train_g, val_g, test_g = chronological_split(graphs, cfg.split)
    if not (train_g and val_g and test_g):
        raise ValueError(f"only {len(graphs)} graph windows - too little data for a train/val/test split. "
                         f"Provide a longer capture (at least ~5 minutes) or lower GraphConfig.window_s.")
    data_info = {"source": source, "flows": len(df), "all": graph_summary(graphs), "train": graph_summary(train_g),
                 "val": graph_summary(val_g), "test": graph_summary(test_g)}
    log(f"  {len(df):,} flows -> {len(graphs)} windows (train {len(train_g)}, val {len(val_g)}, test {len(test_g)})")

    prep = Preprocessor.fit(train_g, use_node_stats=True)
    prep_ns = Preprocessor(prep.node_scaler, prep.edge_scaler, use_node_stats=False)

    # ---------------------------------------------------------------- models
    log("[2/8] training graph models")
    results, info, trained = {}, {}, {}
    for name in models:
        p = prep_ns if name.endswith("_nostats") else prep
        model, inf = train_gnn(name, train_g, val_g, p, cfg.train, log=log)
        m = evaluate_model(model, p, test_g)
        m["params"] = count_parameters(model)
        m["train_seconds"] = inf["train_seconds"]
        m["val_macro_f1"] = inf["best_val_macro_f1"]
        results[name], info[name], trained[name] = m, inf, (model, p)
        torch.save(model.state_dict(), mdir / f"{name}.pt")
        log(f"  {DISPLAY[name]:<32} test macro-F1 {m['macro_f1']:.4f}  DDoS-F1 {m['ddos_f1']:.4f}  "
            f"FPR {m['fpr']:.4f}  {m['latency_ms_per_window']:.1f} ms/window")

    log("[3/8] training baselines")
    fitted = {}
    for name in baselines:
        clf, inf = train_baseline(name, train_g, prep, cfg.train.seed)
        m = evaluate_baseline(name, clf, prep, test_g)
        m["train_seconds"] = inf["train_seconds"]
        results[name], fitted[name] = m, clf
        log(f"  {DISPLAY[name]:<32} test macro-F1 {m['macro_f1']:.4f}  DDoS-F1 {m['ddos_f1']:.4f}  FPR {m['fpr']:.4f}")

    graph_models = [m for m in models if m != "mlp"]
    default = max(graph_models, key=lambda m: results[m]["val_macro_f1"]) if graph_models else models[0]
    log(f"  default model (best validation macro-F1): {DISPLAY[default]}")

    # uncertainty thresholds (5th percentile of validation confidence) per model
    thresholds = {}
    yv = np.concatenate([g.y for g in val_g])
    for name, (model, p) in trained.items():
        pv, _ = predict_proba(model, p, val_g)
        thresholds[name] = float(np.quantile(pv[yv >= 0].max(1), 0.05)) if len(yv) else 0.5

    # SHAP background in explanation space: [flow features | source-host | destination-host] of random train flows
    rng = np.random.default_rng(0)
    train_z = np.vstack([np.hstack([prep.edge_matrix(g), prep.node_scaler.transform(g.x)[g.edge_index[0]],
                                    prep.node_scaler.transform(g.x)[g.edge_index[1]]]) for g in train_g])
    background = train_z[rng.choice(len(train_z), 8, replace=False)]
    benign_raw = np.vstack([g.edge_attr[g.y == 0] for g in train_g])
    drift = DriftMonitor.fit(benign_raw)

    artifacts = {
        "version": 1, "classes": CLASSES, "default_model": default, "data": data_info,
        "graph_config": asdict(cfg.graph), "train_config": asdict(cfg.train),
        "preprocessor": prep.to_dict(), "background": background.tolist(), "drift": drift.to_dict(),
        "uncertainty_threshold": thresholds,
        "models": {n: {"file": f"{n}.pt", "display": DISPLAY[n], "use_node_stats": not n.endswith("_nostats"),
                       "val_macro_f1": results[n]["val_macro_f1"], "test_macro_f1": results[n]["macro_f1"]}
                   for n in models},
    }
    (mdir / ARTIFACTS).write_text(json.dumps(artifacts, default=_json_default))

    report: dict = {"config": cfg.to_dict(), "data": data_info, "default_model": default, "results": results,
                    "training": {k: {kk: vv for kk, vv in v.items() if kk != "history"} for k, v in info.items()},
                    "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                    "cpu_threads": torch.get_num_threads(), "platform": platform.platform()}}

    # ---------------------------------------------------------------- cross-domain
    if cross_domain and synthetic:
        log("[4/8] cross-domain test: train enterprise -> test IoT network")
        dfi = generate_traffic("iot", duration_s=min(duration_s or 1200, 600 if quick else 1200), seed=7)
        gi = build_graphs(dfi, cfg.graph)
        cross = {}
        for name, (model, p) in trained.items():
            cross[name] = evaluate_model(model, p, gi)
        for name, clf in fitted.items():
            cross[name] = evaluate_baseline(name, clf, prep, gi)
        benign_test = np.vstack([g.edge_attr[g.y == 0] for g in test_g])
        benign_iot = np.vstack([g.edge_attr[g.y == 0] for g in gi])
        report["cross_domain"] = {
            "target": "synthetic IoT network (seed 7)", "results": cross,
            "drift_psi_in_domain": drift.psi(benign_test)[0], "drift_psi_iot": drift.psi(benign_iot)[0],
        }
        for name in cross:
            log(f"  {DISPLAY[name]:<32} IoT macro-F1 {cross[name]['macro_f1']:.4f}")

        # few-shot adaptation: fine-tune on the first 20% of the IoT capture (+ an equal replay sample of the
        # original training windows against forgetting), evaluate on the last 60%
        adapt_g, _, eval_g = chronological_split(gi, (0.2, 0.2, 0.6))
        replay = [train_g[i] for i in rng.choice(len(train_g), min(len(train_g), len(adapt_g)), replace=False)]
        few = {"adapt_windows": len(adapt_g), "adapt_flows": int(sum(g.num_edges for g in adapt_g)),
               "eval_flows": int(sum(g.num_edges for g in eval_g)), "results": {}}
        for name in [n for n in (default, "mlp") if n in trained]:
            model, p = trained[name]
            tuned = finetune_gnn(model, p, adapt_g + replay, cfg.train)
            few["results"][name] = {"zero_shot": evaluate_model(model, p, eval_g),
                                    "few_shot": evaluate_model(tuned, p, eval_g),
                                    "enterprise_after": evaluate_model(tuned, p, test_g)}
        if "rf_graphstats" in fitted:
            clf, _ = train_baseline("rf_graphstats", train_g + adapt_g, prep, cfg.train.seed)
            few["results"]["rf_graphstats"] = {"zero_shot": evaluate_baseline("rf_graphstats", fitted["rf_graphstats"], prep, eval_g),
                                               "few_shot": evaluate_baseline("rf_graphstats", clf, prep, eval_g),
                                               "enterprise_after": evaluate_baseline("rf_graphstats", clf, prep, test_g)}
        report["cross_domain"]["few_shot"] = few
        for name, v in few["results"].items():
            log(f"  few-shot {DISPLAY[name]:<28} IoT macro-F1 {v['zero_shot']['macro_f1']:.4f} -> "
                f"{v['few_shot']['macro_f1']:.4f}, FPR {v['zero_shot']['fpr']:.4f} -> {v['few_shot']['fpr']:.4f}")
    else:
        log("[4/8] cross-domain test skipped")

    # ---------------------------------------------------------------- zero-day
    if zero_day:
        log("[5/8] zero-day (leave-one-attack-out) experiment")
        zcfg = type(cfg.train)(**{**asdict(cfg.train), "epochs": 8 if quick else 20, "patience": 4})
        classes = ["DDoS", "PortScan", "Botnet"] if quick else ATTACK_CLASSES
        report["zero_day"] = {"model": default, "results": zero_day_experiment(
            default, train_g, val_g, test_g, trained[default][1], zcfg, classes, log)}
    else:
        log("[5/8] zero-day skipped")

    # ---------------------------------------------------------------- adversarial
    if adversarial:
        log("[6/8] adversarial evasion (PGD)")
        eps_list = [0.0, 0.25, 0.5, 1.0, 2.0]
        adv = {}
        for name in [n for n in ("mlp", "gcn", "gat", "egraphsage") if n in trained]:
            model, p = trained[name]
            adv[name] = [pgd_evasion(model, p, test_g, e) for e in eps_list]
            log(f"  {DISPLAY[name]:<32} " + "  ".join(f"ε={r['eps']}:{r['detection_rate']:.3f}" for r in adv[name]))
        report["adversarial"] = adv
    else:
        log("[6/8] adversarial skipped")

    # ---------------------------------------------------------------- XAI
    model, p = trained[default]
    hops = num_hops(default, cfg.train.layers)
    if xai:
        log("[7/8] evaluating explainers")
        report["xai"] = evaluate_explainers(model, p, test_g, hops, background, per_class=3 if quick else 8, log=log)
        examples = []
        for cname in ATTACK_CLASSES:
            cid = CLASSES.index(cname)
            for g in test_g:
                pr, _ = predict_proba(model, p, [g])
                ok = np.flatnonzero((g.y == cid) & (pr.argmax(1) == cid))
                if len(ok):
                    e = explain_flow(model, p, g, df, int(ok[len(ok) // 2]), hops, background,
                                     methods=("ig", "gnnexplainer", "counterfactual"))
                    examples.append(e)
                    break
        report["examples"] = [{k: v for k, v in e.items() if k != "structure"} for e in examples]
        (out / "example_explanations.md").write_text(
            "# Example explanations (auto-generated)\n\n"
            + "\n\n---\n\n".join(f"## {e['prediction']['label']}\n\n{e['narrative']}" for e in examples))
    else:
        log("[7/8] XAI evaluation skipped")

    # ---------------------------------------------------------------- outputs
    log("[8/8] writing results")
    report["total_seconds"] = time.perf_counter() - t_all
    (out / "metrics.json").write_text(json.dumps(report, indent=1, default=_json_default))
    labels = DISPLAY
    plots.confusion_figure(results[default]["confusion_matrix"], f"Confusion matrix - {labels[default]}",
                           out / "confusion_matrix.png")
    plots.per_class_f1_figure(results, [m for m in ("mlp", "rf", "rf_graphstats", "gat", "egraphsage")
                                        if m in results], labels, out / "per_class_f1.png")
    plots.training_curves_figure({k: v["history"] for k, v in info.items()}, labels, out / "training_curves.png")
    if "adversarial" in report:
        plots.robustness_figure(report["adversarial"], labels, out / "adversarial_robustness.png")
    (out / "RESULTS.md").write_text(results_markdown(report))
    log(f"done in {report['total_seconds'] / 60:.1f} min -> {out}/RESULTS.md")
    return report


def results_markdown(r: dict) -> str:
    L = DISPLAY
    d = r["data"]
    lines = ["# Results", "",
             f"Data: **{d['source']}** - {d['flows']:,} flows in {d['all']['windows']} graph windows "
             f"({r['config']['graph']['window_s']:.0f}s each). Chronological split: train "
             f"{d['train']['flows']:,} / val {d['val']['flows']:,} / test {d['test']['flows']:,} flows.", "",
             f"Environment: Python {r['environment']['python']}, PyTorch {r['environment']['torch']}, CPU "
             f"({r['environment']['cpu_threads']} threads). Total run time {r['total_seconds'] / 60:.1f} min.", "",
             "## 1. Detection performance (test split)", ""]
    rows = []
    for name, m in r["results"].items():
        rows.append({"model": L.get(name, name) + (" **(default)**" if name == r["default_model"] else ""),
                     "acc": m["accuracy"], "macro": m["macro_f1"], "weighted": m["weighted_f1"],
                     "ddos": m["ddos_f1"], "fpr": m["fpr"], "dr": m["detection_rate"],
                     "lat": m["latency_ms_per_window"], "thr": m["throughput_flows_per_s"]})
    lines.append(markdown_table(rows, [("model", "Model", ""), ("acc", "Accuracy", ".4f"),
                                       ("macro", "Macro-F1", ".4f"), ("weighted", "Weighted-F1", ".4f"),
                                       ("ddos", "DDoS F1", ".4f"), ("fpr", "FPR", ".4f"),
                                       ("dr", "Detection rate", ".4f"), ("lat", "Latency ms/window", ".1f"),
                                       ("thr", "Flows/s", ",.0f")]))
    best = r["results"][r["default_model"]]
    lines += ["", f"### Per-class results - {L[r['default_model']]}", ""]
    lines.append(markdown_table(
        [{"c": c, **v} for c, v in best["per_class"].items()],
        [("c", "Class", ""), ("precision", "Precision", ".4f"), ("recall", "Recall", ".4f"), ("f1", "F1", ".4f"),
         ("support", "Support", "d")]))
    lines += ["", "![per-class F1](per_class_f1.png)", "", "![confusion matrix](confusion_matrix.png)", ""]
    if "cross_domain" in r:
        cd = r["cross_domain"]
        lines += ["## 2. Cross-domain generalisation (enterprise -> IoT, no retraining)", "",
                  f"Benign-traffic drift (PSI vs. training): in-domain test {cd['drift_psi_in_domain']:.3f}, "
                  f"IoT network {cd['drift_psi_iot']:.3f} (>0.25 = significant drift).", ""]
        lines.append(markdown_table(
            [{"model": L.get(k, k), "macro": v["macro_f1"], "ddos": v["ddos_f1"], "fpr": v["fpr"],
              "dr": v["detection_rate"], "in": r["results"][k]["macro_f1"]} for k, v in cd["results"].items()],
            [("model", "Model", ""), ("in", "In-domain macro-F1", ".4f"), ("macro", "IoT macro-F1", ".4f"),
             ("ddos", "IoT DDoS F1", ".4f"), ("fpr", "IoT FPR", ".4f"), ("dr", "IoT detection rate", ".4f")]))
        lines.append("")
        if "few_shot" in cd:
            fs = cd["few_shot"]
            lines += ["### Few-shot adaptation to the new network", "",
                      f"Fine-tuned on the first {fs['adapt_flows']:,} IoT flows ({fs['adapt_windows']} windows, about "
                      f"{fs['adapt_windows'] * r['config']['graph']['window_s'] / 60:.0f} min of labelled traffic) plus a "
                      f"replay sample of the original training windows; evaluated on the last {fs['eval_flows']:,} IoT flows.", ""]
            lines.append(markdown_table(
                [{"model": L.get(k, k), "zm": v["zero_shot"]["macro_f1"], "zf": v["zero_shot"]["fpr"],
                  "fm": v["few_shot"]["macro_f1"], "ff": v["few_shot"]["fpr"],
                  "em": v["enterprise_after"]["macro_f1"]} for k, v in fs["results"].items()],
                [("model", "Model", ""), ("zm", "IoT macro-F1 before", ".4f"), ("zf", "IoT FPR before", ".4f"),
                 ("fm", "IoT macro-F1 after", ".4f"), ("ff", "IoT FPR after", ".4f"),
                 ("em", "Enterprise macro-F1 after", ".4f")]))
            lines.append("")
    if "zero_day" in r:
        lines += ["## 3. Zero-day test (leave-one-attack-out)", "",
                  f"Model: {L[r['zero_day']['model']]}. The hidden family is present in the training traffic but "
                  "never labelled. *Detected* = classified as any attack or flagged as uncertain.", ""]
        lines.append(markdown_table(r["zero_day"]["results"], [
            ("unseen_class", "Unseen family", ""), ("flows", "Test flows", ",d"),
            ("flagged_as_attack", "Flagged as attack", ".3f"), ("flagged_uncertain", "Uncertain", ".3f"),
            ("detected", "Detected (either)", ".3f"), ("most_common_prediction", "Most common label", ""),
            ("benign_false_alarm", "Benign false alarms", ".3f")]))
        lines.append("")
    if "adversarial" in r:
        lines += ["## 4. Adversarial robustness", "",
                  "White-box PGD (10 steps) on attacker-controllable features (timing, volume, packet sizes); "
                  "ε in standard deviations. Values = share of attack flows still detected.", ""]
        eps = [x["eps"] for x in next(iter(r["adversarial"].values()))]
        rows = [{"model": L.get(k, k), **{f"e{e}": x["detection_rate"] for e, x in zip(eps, v)}}
                for k, v in r["adversarial"].items()]
        lines.append(markdown_table(rows, [("model", "Model", "")] + [(f"e{e}", f"ε={e}", ".3f") for e in eps]))
        lines += ["", "![robustness](adversarial_robustness.png)", ""]
    if "xai" in r:
        x = r["xai"]
        lines += ["## 5. Explanation quality", "",
                  f"{x['n_alerts']} correctly detected attack flows. Feature explainers rank the {x['top_k']} most "
                  "important items of the explanation space (27 flow features + 11 statistics of each endpoint host). "
                  "Fidelity is measured in log-odds of the explained class: Fidelity+ = drop when the top items are reset "
                  "to the training mean (higher = they are necessary), Fidelity- = drop when *only* they are kept "
                  "(lower = they are sufficient). Flip rate = share of alerts whose class changes when the top items are "
                  "reset. Sparsity = share of items not needed for 80% of the attribution mass. Stability = top-k "
                  "overlap (Jaccard) under small input noise.", "", "### Feature attribution", ""]
        cols = [("method", "Method", ""), ("fidelity_plus", "Fidelity+ (log-odds)", ".3f"),
                ("fidelity_minus", "Fidelity- (log-odds)", ".3f"), ("flip_rate", "Flip rate", ".2f"),
                ("sparsity", "Sparsity", ".3f"), ("stability", "Stability", ".3f"), ("runtime_ms", "Runtime ms", ".1f")]
        lines.append(markdown_table(x["feature_explainers"], cols))
        lines += ["", "### Structural explanation (top 20% of neighbouring flows removed / kept)", ""]
        scols = [("method", "Method", ""), ("fidelity_plus", "Fidelity+ (log-odds)", ".3f"),
                 ("fidelity_minus", "Fidelity- (log-odds)", ".3f"), ("sparsity", "Sparsity", ".3f"),
                 ("stability", "Stability", ".3f"), ("runtime_ms", "Runtime ms", ".1f")]
        lines.append(markdown_table(x["structural_explainers"], scols))
        lines += ["", "Example narratives: [example_explanations.md](example_explanations.md)", ""]
    lines += ["## Training curves", "", "![training curves](training_curves.png)", ""]
    return "\n".join(lines)
