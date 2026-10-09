"""Scores and figures for evaluation runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

HIGH_AROUSAL = {"angry", "fearful", "happy"}


def auc(pos: list[float], neg: list[float]) -> float:
    """Probability a random positive outranks a random negative (ties count half)."""
    if not pos or not neg:
        return float("nan")
    p, n = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float(np.mean(p > n) + 0.5 * np.mean(p == n))


def confusion(true: list[str], pred: list[str], labels: list[str]) -> np.ndarray:
    idx = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(true, pred):
        if p in idx:  # predictions outside the label set only count as errors
            cm[idx[t], idx[p]] += 1
    return cm


def score(records: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    """Accuracy, macro-F1, per-class recall, arousal AUC and confusion for one run."""
    true = [r["clip"]["emotion"] for r in records]
    pred = [r["emotion"] for r in records]
    cm = confusion(true, pred, labels)
    support = np.array([sum(t == lab for t in true) for lab in labels])
    recall = np.divide(np.diag(cm), support, out=np.zeros(len(labels)), where=support > 0)
    col = cm.sum(axis=0)
    precision = np.divide(np.diag(cm), col, out=np.zeros(len(labels)), where=col > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros(len(labels)), where=denom > 0)
    aro = [(r["arousal"], t in HIGH_AROUSAL) for r, t in zip(records, true) if r.get("arousal") is not None]
    return {
        "n": len(records),
        "accuracy": float(np.mean([t == p for t, p in zip(true, pred)])),
        "macro_f1": float(np.mean(f1)),
        "recall": {lab: float(v) for lab, v in zip(labels, recall)},
        "arousal_auc": auc([a for a, h in aro if h], [a for a, h in aro if not h]),
        "latency_p50_s": float(np.median([r["latency_s"] for r in records])),
        "confusion": cm.tolist(),
    }


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_confusion(cm: list[list[int]], labels: list[str], title: str, path: Path) -> None:
    plt = _plt()
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.6, 4.0), dpi=120)
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    vmax = cm.max() or 1
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > vmax / 2 else "black")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_results(rows: list[dict[str, Any]], variants: list[str], path: Path, chance: float) -> None:
    """Grouped bars: accuracy per language for each variant."""
    plt = _plt()
    langs = list(dict.fromkeys(r["language"] for r in rows))
    width = 0.8 / len(variants)
    colors = ["#c4ccd6", "#8fa3bd", "#5b7aa6", "#d1495b"]
    fig, ax = plt.subplots(figsize=(8.5, 3.6), dpi=130)
    for i, variant in enumerate(variants):
        vals = [
            next((r["accuracy"] for r in rows if r["language"] == lang and r["variant"] == variant), np.nan)
            for lang in langs
        ]
        xs = np.arange(len(langs)) + (i - (len(variants) - 1) / 2) * width
        ax.bar(xs, [v * 100 for v in vals], width, label=variant, color=colors[i % len(colors)])
    ax.axhline(chance * 100, color="#555", linestyle="--", linewidth=1)
    ax.text(-0.48, chance * 100 + 1.5, "chance", ha="left", fontsize=8, color="#555")
    ax.set_xticks(range(len(langs)), langs)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=len(variants), loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
