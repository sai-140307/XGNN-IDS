"""Evaluation metrics beyond accuracy (as the literature review recommends)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from ..config import CLASSES


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str] = CLASSES) -> dict:
    """Per-class P/R/F1, macro/weighted F1, FPR, detection rate and DDoS-specific scores.

    * FPR            = benign flows flagged as any attack / all benign flows
    * detection rate = attack flows flagged as *any* attack / all attack flows
                       (binary view: did the alert fire at all?)
    """
    mask = y_true >= 0
    y_true, y_pred = y_true[mask], y_pred[mask]
    labels = list(range(len(classes)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    benign = y_true == 0
    attack = ~benign
    present = s > 0
    out = {
        "n": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        "macro_f1": float(f[present].mean()) if present.any() else 0.0,
        "weighted_f1": float((f * s).sum() / max(s.sum(), 1)),
        "fpr": float((y_pred[benign] != 0).mean()) if benign.any() else 0.0,
        "detection_rate": float((y_pred[attack] != 0).mean()) if attack.any() else 0.0,
        "per_class": {
            c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, c in enumerate(classes)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    if "DDoS" in classes:
        i = classes.index("DDoS")
        out["ddos"] = out["per_class"]["DDoS"]
        out["ddos_f1"] = float(f[i])
    return out


def markdown_table(rows: list[dict], cols: list[tuple[str, str, str]]) -> str:
    """rows: list of dicts; cols: (key, header, fmt)."""
    head = "| " + " | ".join(h for _, h, _ in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines = [head, sep]
    for r in rows:
        cells = []
        for k, _, fmt in cols:
            v = r.get(k, "")
            cells.append(format(v, fmt) if isinstance(v, (int, float)) and fmt else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
