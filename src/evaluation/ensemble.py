"""Soft-voting ensemble of the top-k models (averaged class probabilities).
Provides the upper-bound row of the comparison table (ablation AB-7).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def soft_voting(results_dir, model_names):
    """Average saved softmax probabilities across model_names.
    Each model must have a `<name>_preds.npz` saved by evaluation.metrics.
    """
    results_dir = Path(results_dir)
    probs, y_true = None, None
    for name in model_names:
        data = np.load(results_dir / f"{name}_preds.npz")
        p = data["y_prob"]
        probs = p if probs is None else probs + p
        if y_true is None:
            y_true = data["y_true"]
        elif not np.array_equal(y_true, data["y_true"]):
            raise ValueError("Models were evaluated on different test orderings.")
    probs /= len(model_names)
    y_pred = probs.argmax(1)
    return {
        "model": "ensemble_soft_voting(" + "+".join(model_names) + ")",
        "members": model_names,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }


def pick_topk(results, k=3, metric="f1_macro"):
    ranked = sorted(results, key=lambda r: r[metric], reverse=True)
    return [r["model"] for r in ranked[:k]]
