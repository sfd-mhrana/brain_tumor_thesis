"""Publication-quality figures (all saved at 300 DPI)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import roc_curve, auc  # noqa: E402
from sklearn.preprocessing import label_binarize  # noqa: E402

plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.size": 11})


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm, classes, path, title="Confusion Matrix", normalize=True):
    cm = np.asarray(cm, dtype=float)
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1e-9)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else cm.max())
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]:.2f}" if normalize else int(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    _save(fig, path)


def plot_training_curves(history, path):
    epochs = list(range(1, len(history) + 1))
    tr = [h["train_loss"] for h in history]
    vl = [h["val_loss"] for h in history]
    va = [h["val_acc"] for h in history]
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(epochs, tr, label="train loss")
    ax1.plot(epochs, vl, label="val loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(loc="upper right")
    ax2 = ax1.twinx()
    ax2.plot(epochs, va, "g--", label="val acc")
    ax2.set_ylabel("Val accuracy")
    _save(fig, path)


def plot_multiclass_roc(y_true, y_prob, classes, path, title="ROC (one-vs-rest)"):
    y_bin = label_binarize(y_true, classes=list(range(len(classes))))
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, label=f"{c} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title); ax.legend(loc="lower right")
    _save(fig, path)


def plot_model_comparison(results, path, metric="f1_macro"):
    names = [r["model"] for r in results]
    vals = [r[metric] for r in results]
    order = np.argsort(vals)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([names[i] for i in order], [vals[i] for i in order], color="#3498DB")
    ax.set_xlabel(metric); ax.set_title(f"Model comparison — {metric}")
    for i, v in enumerate([vals[i] for i in order]):
        ax.text(v, i, f" {v:.3f}", va="center")
    _save(fig, path)
