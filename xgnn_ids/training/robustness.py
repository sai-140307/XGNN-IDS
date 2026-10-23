"""Robustness experiments: adversarial evasion and zero-day (unseen attack) detection."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..config import CLASSES, TrainConfig
from ..data.features import EDGE_FEATURE_NAMES
from ..graph.builder import WindowGraph
from .data import Preprocessor
from .trainer import predict_proba, train_gnn

# What an attacker controls about *their own* flows: timing, volume and packet sizes.
# Protocol / destination-port categories (which service is attacked) and TCP
# handshake semantics are kept fixed.
ATTACKER_MUTABLE = [
    "log_duration_ms", "log_fwd_pkts", "log_fwd_bytes", "log_bytes_per_s", "log_pkts_per_s",
    "log_fwd_pkt_len", "log_iat_ms", "log_psh",
]


def pgd_evasion(model, prep: Preprocessor, graphs: list[WindowGraph], eps: float, steps: int = 10) -> dict:
    """White-box targeted PGD: every attack flow's mutable features are perturbed (L-inf <= eps,
    in standard-deviation units) to make the model call them BENIGN.

    Returns the attack detection rate (attack flows still flagged as *any* attack).
    """
    mut = torch.tensor([n in ATTACKER_MUTABLE for n in EDGE_FEATURE_NAMES], dtype=torch.float32)
    detected, total = 0, 0
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    try:
        for g in graphs:
            x, ei, ea, y = prep.tensors(g)
            atk = y > 0
            if not atk.any():
                continue
            rowmask = atk.float().unsqueeze(1) * mut.unsqueeze(0)
            delta = torch.zeros_like(ea)
            if eps > 0:
                alpha = eps / 4
                for _ in range(steps):
                    delta.requires_grad_(True)
                    logits = model(x, ei, ea + delta * rowmask)
                    loss = F.cross_entropy(logits[atk], torch.zeros(int(atk.sum()), dtype=torch.long))
                    (grad,) = torch.autograd.grad(loss, delta)
                    delta = (delta.detach() - alpha * grad.sign()).clamp(-eps, eps)
            with torch.no_grad():
                pred = model(x, ei, ea + delta * rowmask).argmax(1)
            detected += int((pred[atk] != 0).sum())
            total += int(atk.sum())
    finally:
        for p in model.parameters():
            p.requires_grad_(True)
    return {"eps": eps, "detection_rate": detected / max(total, 1), "attack_flows": total}


def hide_class(graphs: list[WindowGraph], cls_id: int) -> list[WindowGraph]:
    """Copies of the graphs where flows of ``cls_id`` are unlabelled (-1): the traffic is still
    present in the network (realistic) but the model never receives its label."""
    out = []
    for g in graphs:
        y = g.y.copy()
        y[y == cls_id] = -1
        out.append(WindowGraph(**{**g.__dict__, "y": y}))
    return out


def zero_day_experiment(name: str, train_g, val_g, test_g, prep: Preprocessor, cfg: TrainConfig,
                        classes: list[str], log=print) -> list[dict]:
    """Leave-one-attack-out: can the model flag an attack family it was never trained on?

    A flow of the unseen family counts as *detected* if it is classified as any attack
    or if the model is uncertain (max probability below the 5th percentile of
    confidences on the validation set).
    """
    rows = []
    for cname in classes:
        cid = CLASSES.index(cname)
        log(f"  zero-day: hiding {cname}")
        model, _ = train_gnn(name, hide_class(train_g, cid), hide_class(val_g, cid), prep, cfg, log=lambda *_: None)
        pv, _ = predict_proba(model, prep, val_g)
        yv = np.concatenate([g.y for g in val_g])
        known = (yv >= 0) & (yv != cid)
        tau = float(np.quantile(pv[known].max(1), 0.05)) if known.any() else 0.5
        pt, _ = predict_proba(model, prep, test_g)
        yt = np.concatenate([g.y for g in test_g])
        unseen = yt == cid
        benign = yt == 0
        if not unseen.any():
            continue
        pred = pt.argmax(1)
        as_attack = pred[unseen] != 0
        uncertain = pt[unseen].max(1) < tau
        fp_unc = (pt[benign].max(1) < tau) | (pred[benign] != 0)
        top_other = np.bincount(pred[unseen], minlength=len(CLASSES))
        rows.append({
            "unseen_class": cname,
            "flows": int(unseen.sum()),
            "flagged_as_attack": float(as_attack.mean()),
            "flagged_uncertain": float(uncertain.mean()),
            "detected": float((as_attack | uncertain).mean()),
            "benign_false_alarm": float(fp_unc.mean()) if benign.any() else 0.0,
            "most_common_prediction": CLASSES[int(top_other.argmax())],
            "threshold": tau,
        })
    return rows
