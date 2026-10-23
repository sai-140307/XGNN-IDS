"""Static result figures for the report / README (matplotlib, headless)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..config import CLASSES  # noqa: E402

INK = "#1f2933"
MUTED = "#7b8794"
ACCENT = "#2f6fde"
PALETTE = ["#2f6fde", "#e8833a", "#3aa675", "#c94f6d", "#8a63d2", "#b5912f", "#4aa3b5", "#7b8794"]


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="y", color="#e4e7eb", linewidth=0.8)
    ax.set_axisbelow(True)


def confusion_figure(cm: list[list[int]], title: str, path: Path) -> None:
    cm = np.asarray(cm, float)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize=9)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if cm[i, j] > 0:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.6 else INK)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=11, color=INK)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def per_class_f1_figure(results: dict, models: list[str], labels: dict, path: Path) -> None:
    attack = CLASSES
    x = np.arange(len(attack))
    w = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for i, m in enumerate(models):
        f1 = [results[m]["per_class"][c]["f1"] for c in attack]
        ax.bar(x + i * w - 0.4 + w / 2, f1, w, label=labels.get(m, m), color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(x, attack, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 score")
    ax.set_title("Per-class F1 on the chronological test split", fontsize=11, color=INK)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="lower left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def training_curves_figure(histories: dict, labels: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, (m, h) in enumerate(histories.items()):
        ax.plot([r["epoch"] for r in h], [r["val_macro_f1"] for r in h], label=labels.get(m, m),
                color=PALETTE[i % len(PALETTE)], linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation macro-F1")
    ax.set_title("Training curves", fontsize=11, color=INK)
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def robustness_figure(adv: dict, labels: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for i, (m, rows) in enumerate(adv.items()):
        ax.plot([r["eps"] for r in rows], [r["detection_rate"] for r in rows], marker="o", label=labels.get(m, m),
                color=PALETTE[i % len(PALETTE)], linewidth=1.8)
    ax.set_xlabel("Perturbation budget ε (std. units, attacker-controlled features)")
    ax.set_ylabel("Attack detection rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Robustness to adversarial evasion (white-box PGD)", fontsize=11, color=INK)
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
