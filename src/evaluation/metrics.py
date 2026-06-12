"""Inference and the full metric suite reported for every model:
accuracy, macro precision/recall/F1, weighted F1, per-class F1, AUC-ROC (OvR),
Cohen's kappa, specificity, confusion matrix, and inference latency.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)


@torch.no_grad()
def predict(model, loader, device, tta: bool = False):
    """Return (y_true, y_pred, y_prob) over a loader."""
    model.eval()
    probs, trues = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        if tta:  # horizontal-flip TTA — anatomically valid for brain MRI
            logits = logits + model(torch.flip(x, dims=[3]))
        p = torch.softmax(logits, dim=1)
        probs.append(p.cpu().numpy())
        trues.append(y.numpy())
    y_prob = np.concatenate(probs)
    y_true = np.concatenate(trues)
    y_pred = y_prob.argmax(1)
    return y_true, y_pred, y_prob


def specificity_macro(cm: np.ndarray) -> float:
    """Mean one-vs-rest specificity across classes."""
    specs = []
    total = cm.sum()
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn
        specs.append(tn / (tn + fp) if (tn + fp) else 0.0)
    return float(np.mean(specs))


def measure_latency(model, loader, device, n: int = 100) -> float:
    """Average single-image inference time in milliseconds."""
    model.eval()
    x, _ = next(iter(loader))
    x = x.to(device)
    with torch.no_grad():
        for _ in range(5):                 # warm-up
            model(x[:1])
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        for i in range(n):
            model(x[i % x.size(0):i % x.size(0) + 1])
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.time() - start) / n * 1000


def evaluate(model, loader, device, model_name, classes, params_M=None,
             train_time_s=None, tta=False, results_dir=None):
    y_true, y_pred, y_prob = predict(model, loader, device, tta=tta)
    cm = confusion_matrix(y_true, y_pred)

    results = {
        "model": model_name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, average="weighted"), 4),
        "specificity_macro": round(specificity_macro(cm), 4),
        "kappa": round(cohen_kappa_score(y_true, y_pred), 4),
        "per_class_f1": {c: round(v, 4) for c, v in
                         zip(classes, f1_score(y_true, y_pred, average=None))},
        "confusion_matrix": cm.tolist(),
    }
    try:
        results["auc_roc_ovr"] = round(
            roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"), 4)
    except ValueError:
        results["auc_roc_ovr"] = None

    results["inference_ms"] = round(measure_latency(model, loader, device), 3)
    # On-disk model size (MB) at fp32 — params + buffers.
    n_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    n_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    results["size_mb"] = round(n_bytes / 1e6, 2)
    if params_M is not None:
        results["params_M"] = params_M
    if train_time_s is not None:
        results["train_time_s"] = train_time_s

    if results_dir is not None:
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(results_dir) / f"{model_name}.json", "w") as f:
            json.dump(results, f, indent=2)
        # Save raw predictions for McNemar's test and ensembling.
        np.savez(Path(results_dir) / f"{model_name}_preds.npz",
                 y_true=y_true, y_pred=y_pred, y_prob=y_prob)
    return results
