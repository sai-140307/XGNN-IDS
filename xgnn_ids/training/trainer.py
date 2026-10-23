"""Training / inference loops for the GNN models and the classical baselines."""
from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from ..config import CLASSES, TrainConfig
from ..graph.builder import WindowGraph
from ..models.gnn import build_model
from .data import Preprocessor, make_batches
from .metrics import classification_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def class_weights(graphs: list[WindowGraph], n_cls: int) -> torch.Tensor:
    y = np.concatenate([g.y for g in graphs])
    y = y[y >= 0]
    counts = np.bincount(y, minlength=n_cls).astype(float)
    w = np.where(counts > 0, 1.0 / np.sqrt(np.maximum(counts, 1)), 0.0)
    w = w / w[w > 0].mean()
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def predict_proba(model, prep: Preprocessor, graphs: list[WindowGraph]) -> tuple[np.ndarray, list[float]]:
    """Return per-flow class probabilities and per-window latency (ms)."""
    model.eval()
    probs, lat = [], []
    for g in graphs:
        x, ei, ea, _ = prep.tensors(g)
        t = time.perf_counter()
        p = F.softmax(model(x, ei, ea), dim=1)
        lat.append((time.perf_counter() - t) * 1000)
        probs.append(p.numpy())
    return (np.vstack(probs) if probs else np.zeros((0, len(CLASSES)))), lat


def train_gnn(name: str, train_g: list[WindowGraph], val_g: list[WindowGraph], prep: Preprocessor,
              cfg: TrainConfig, n_cls: int = len(CLASSES), log=print):
    """Train one GNN with class-weighted cross entropy and early stopping on val macro-F1."""
    set_seed(cfg.seed)
    model = build_model(name, prep.n_node_features, prep.n_edge_features, n_cls,
                        hidden=cfg.hidden, layers=cfg.layers, heads=cfg.heads, dropout=cfg.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1))
    weights = class_weights(train_g, n_cls)
    batches = [prep.tensors(b) for b in make_batches(train_g, cfg.batch_edges)]
    y_val = np.concatenate([g.y for g in val_g]) if val_g else np.array([], int)

    best, best_f1, bad, history = None, -1.0, 0, []
    t_start = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        random.shuffle(batches)
        total = 0.0
        for x, ei, ea, y in batches:
            opt.zero_grad()
            loss = F.cross_entropy(model(x, ei, ea), y, weight=weights, ignore_index=-1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss)
        sched.step()
        if val_g:
            probs, _ = predict_proba(model, prep, val_g)
            val = classification_metrics(y_val, probs.argmax(1))
            f1 = val["macro_f1"]
        else:
            f1 = -total
        history.append({"epoch": epoch, "loss": total / max(len(batches), 1), "val_macro_f1": f1})
        if f1 > best_f1 + 1e-4:
            best_f1, best, bad = f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if epoch == 1 or epoch % 5 == 0:
            log(f"  [{name}] epoch {epoch:3d} loss {total / max(len(batches), 1):.4f} val macro-F1 {f1:.4f}")
        if bad >= cfg.patience:
            log(f"  [{name}] early stop at epoch {epoch} (best val macro-F1 {best_f1:.4f})")
            break
    if best is not None:
        model.load_state_dict(best)
    model.eval()
    info = {"train_seconds": time.perf_counter() - t_start, "epochs_run": len(history),
            "best_val_macro_f1": best_f1, "history": history, "hparams": asdict(cfg)}
    return model, info


def evaluate_model(model, prep: Preprocessor, graphs: list[WindowGraph]) -> dict:
    probs, lat = predict_proba(model, prep, graphs)
    y = np.concatenate([g.y for g in graphs])
    m = classification_metrics(y, probs.argmax(1))
    n_edges = sum(g.num_edges for g in graphs)
    m["latency_ms_per_window"] = float(np.mean(lat)) if lat else 0.0
    m["throughput_flows_per_s"] = float(n_edges / max(sum(lat) / 1000, 1e-9))
    return m


# ------------------------------------------------------------------ baselines


def tabular_features(graphs: list[WindowGraph], prep: Preprocessor, with_graph_stats: bool):
    xs, ys = [], []
    for g in graphs:
        f = prep.edge_matrix(g)
        if with_graph_stats:
            nx = prep.node_scaler.transform(g.x)
            f = np.hstack([f, nx[g.edge_index[0]], nx[g.edge_index[1]]])
        xs.append(f)
        ys.append(g.y)
    return np.vstack(xs), np.concatenate(ys)


BASELINES = {
    "rf": ("Random Forest (flow features)", False),
    "rf_graphstats": ("Random Forest + graph statistics", True),
    "hgb": ("Gradient Boosting (flow features)", False),
}


def train_baseline(name: str, train_g, prep: Preprocessor, seed: int = 42):
    _, with_stats = BASELINES[name]
    X, y = tabular_features(train_g, prep, with_stats)
    keep = y >= 0
    t = time.perf_counter()
    if name.startswith("rf"):
        clf = RandomForestClassifier(n_estimators=150, min_samples_leaf=2, class_weight="balanced_subsample",
                                     n_jobs=-1, random_state=seed)
    else:
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, class_weight="balanced",
                                             random_state=seed)
    clf.fit(X[keep], y[keep])
    return clf, {"train_seconds": time.perf_counter() - t}


def evaluate_baseline(name: str, clf, prep: Preprocessor, graphs) -> dict:
    _, with_stats = BASELINES[name]
    if hasattr(clf, "n_jobs"):
        clf.n_jobs = 1  # per-window latency comparable to single-threaded GNN inference
    preds, lat, ys = [], [], []
    for g in graphs:
        X, y = tabular_features([g], prep, with_stats)
        t = time.perf_counter()
        preds.append(clf.predict(X))
        lat.append((time.perf_counter() - t) * 1000)
        ys.append(y)
    y, p = np.concatenate(ys), np.concatenate(preds)
    m = classification_metrics(y, p)
    m["latency_ms_per_window"] = float(np.mean(lat))
    m["throughput_flows_per_s"] = float(len(y) / max(sum(lat) / 1000, 1e-9))
    return m
