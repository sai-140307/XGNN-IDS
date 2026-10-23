"""High-level inference API used by the dashboard, the REST API and live capture.

    det = Detector.load("models")              # best model by default
    result = det.detect(flows_df)              # canonical flow table -> predictions
    incidents = det.incidents(result)          # correlated incidents + firewall rules
    explanation = det.explain(result, row)     # full explanation of one flow
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .analysis.drift import DriftMonitor
from .analysis.incidents import correlate_incidents
from .config import CLASSES, GraphConfig
from .data.features import CANONICAL_COLUMNS, coerce_numeric
from .explain.report import explain_flow
from .graph.builder import WindowGraph, build_graphs
from .models.gnn import MODEL_DISPLAY, build_model, num_hops
from .training.data import Preprocessor
from .training.trainer import predict_proba

ARTIFACTS = "artifacts.json"


@dataclass
class Detection:
    df: pd.DataFrame                 # input flows + pred_label, confidence, uncertain, window
    graphs: list[WindowGraph]
    probs: np.ndarray                # [n_flows, n_classes]
    window_of: np.ndarray            # flow row -> graph index
    edge_of: np.ndarray              # flow row -> edge index inside its graph
    windows: pd.DataFrame = field(default_factory=pd.DataFrame)  # per-window summary (alerts, drift, latency)

    @property
    def alerts(self) -> pd.DataFrame:
        return self.df[self.df["pred_label"] != "BENIGN"]


class Detector:
    def __init__(self, model, name: str, prep: Preprocessor, graph_cfg: GraphConfig, layers: int,
                 background: np.ndarray | None = None, drift: DriftMonitor | None = None,
                 uncertainty_threshold: float = 0.5, meta: dict | None = None):
        self.model = model.eval()
        self.name = name
        self.prep = prep
        self.graph_cfg = graph_cfg
        self.layers = layers
        self.hops = num_hops(name, layers)
        self.background = background
        self.drift = drift
        self.threshold = uncertainty_threshold
        self.meta = meta or {}

    @property
    def display_name(self) -> str:
        return MODEL_DISPLAY.get(self.name, self.name)

    # ------------------------------------------------------------------ IO
    @staticmethod
    def read_artifacts(models_dir: str | Path) -> dict:
        return json.loads((Path(models_dir) / ARTIFACTS).read_text())

    @classmethod
    def available_models(cls, models_dir: str | Path = "models") -> list[str]:
        try:
            return list(cls.read_artifacts(models_dir)["models"].keys())
        except FileNotFoundError:
            return []

    @classmethod
    def load(cls, models_dir: str | Path = "models", name: str | None = None) -> "Detector":
        models_dir = Path(models_dir)
        art = cls.read_artifacts(models_dir)
        name = name or art["default_model"]
        entry = art["models"][name]
        prep = Preprocessor.from_dict(art["preprocessor"])
        prep.use_node_stats = entry.get("use_node_stats", True)
        tc = art["train_config"]
        model = build_model(name, prep.n_node_features, prep.n_edge_features, len(art["classes"]),
                            hidden=tc["hidden"], layers=tc["layers"], heads=tc["heads"], dropout=tc["dropout"])
        model.load_state_dict(torch.load(models_dir / entry["file"], map_location="cpu", weights_only=True))
        drift = DriftMonitor.from_dict(art["drift"]) if art.get("drift") else None
        bg = np.asarray(art["background"], np.float32) if art.get("background") else None
        return cls(model, name, prep, GraphConfig(**art["graph_config"]), tc["layers"], bg, drift,
                   art.get("uncertainty_threshold", {}).get(name, 0.5), art)

    # ------------------------------------------------------------------ inference
    def detect(self, df: pd.DataFrame, drift_min_flows: int = 150) -> Detection:
        df = coerce_numeric(df)
        extra = [c for c in df.columns if c not in CANONICAL_COLUMNS]
        df = df[CANONICAL_COLUMNS + extra].reset_index(drop=True)
        graphs = build_graphs(df, self.graph_cfg)
        n = len(df)
        probs = np.zeros((n, len(CLASSES)), np.float32)
        window_of = np.full(n, -1, np.int64)
        edge_of = np.full(n, -1, np.int64)
        rows = []
        for gi, g in enumerate(graphs):
            t = time.perf_counter()
            p, _ = predict_proba(self.model, self.prep, [g])
            ms = (time.perf_counter() - t) * 1000
            probs[g.flow_rows] = p
            window_of[g.flow_rows] = gi
            edge_of[g.flow_rows] = np.arange(g.num_edges)
            pred = p.argmax(1)
            rec = {"window": gi, "t0": g.t0, "t1": g.t1, "segment": g.segment, "flows": g.num_edges,
                   "hosts": g.num_nodes, "alerts": int((pred != 0).sum()), "latency_ms": ms}
            for ci, c in enumerate(CLASSES[1:], start=1):
                rec[c] = int((pred == ci).sum())
            benign = pred == 0
            if self.drift is not None and benign.sum() >= drift_min_flows:
                rec["drift_psi"] = self.drift.psi(g.edge_attr[benign])[0]
            else:
                rec["drift_psi"] = np.nan
            rows.append(rec)
        out = df.copy()
        pred = probs.argmax(1)
        out["pred_label"] = np.asarray(CLASSES, dtype=object)[pred]
        out["confidence"] = probs.max(1)
        out["uncertain"] = out["confidence"] < self.threshold
        out["window"] = window_of
        return Detection(out, graphs, probs, window_of, edge_of, pd.DataFrame(rows))

    def explain(self, det: Detection, row: int, methods=("ig", "edge_ig", "counterfactual"),
                shap_nsamples: int = 150) -> dict:
        gi, e = int(det.window_of[row]), int(det.edge_of[row])
        return explain_flow(self.model, self.prep, det.graphs[gi], det.df, e, self.hops, self.background,
                            methods, shap_nsamples)

    def incidents(self, det: Detection, min_confidence: float = 0.5, gap_s: float = 60.0) -> list[dict]:
        a = det.alerts
        return correlate_incidents(a[a["confidence"] >= min_confidence], gap_s=gap_s)
