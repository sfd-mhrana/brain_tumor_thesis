"""Phase 3 — Exploratory Data Analysis. Produces class-distribution, sample-grid
and image-dimension figures plus a duplicate report for Chapter 3.

    python scripts/01_eda.py
"""
import _bootstrap  # noqa: F401
import hashlib
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.utils import ensure_dir, load_config  # noqa: E402

COLORS = ["#E74C3C", "#3498DB", "#2ECC71", "#9B59B6"]


def main():
    cfg = load_config()
    classes = cfg["data"]["classes"]
    train_dir = Path(cfg["paths"]["raw_dir"]) / "Training"
    fig_dir = ensure_dir(cfg["paths"]["figures_dir"])

    # 1. Class distribution
    counts = {c: len(list((train_dir / c).glob("*"))) for c in classes}
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.keys(), counts.values(), color=COLORS[: len(classes)])
    ax.set_title("Class Distribution (Training)", fontweight="bold")
    ax.set_ylabel("Number of images")
    for i, v in enumerate(counts.values()):
        ax.text(i, v, str(v), ha="center", va="bottom")
    fig.savefig(fig_dir / "class_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Class counts:", counts)
    print("Imbalance ratio (max/min): %.2f" % (max(counts.values()) / min(counts.values())))

    # 2. Sample grid (3 per class)
    fig, axes = plt.subplots(len(classes), 3, figsize=(9, 3 * len(classes)))
    for i, c in enumerate(classes):
        files = sorted((train_dir / c).glob("*"))[:3]
        for j, fp in enumerate(files):
            img = cv2.cvtColor(cv2.imread(str(fp)), cv2.COLOR_BGR2RGB)
            axes[i][j].imshow(img); axes[i][j].axis("off")
            if j == 0:
                axes[i][j].set_ylabel(c)
            axes[i][j].set_title(f"{c}")
    fig.savefig(fig_dir / "sample_images.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 3. Dimension stats + duplicate detection via perceptual-ish md5 hashing
    heights, widths = [], []
    hashes = defaultdict(list)
    for c in classes:
        for fp in (train_dir / c).glob("*"):
            img = cv2.imread(str(fp))
            if img is None:
                print("  [corrupted]", fp)
                continue
            heights.append(img.shape[0]); widths.append(img.shape[1])
            small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (16, 16))
            hashes[hashlib.md5(small.tobytes()).hexdigest()].append(str(fp))
    dups = {h: fs for h, fs in hashes.items() if len(fs) > 1}
    print(f"Image dims: H {min(heights)}-{max(heights)}, W {min(widths)}-{max(widths)}")
    print(f"Potential duplicate groups: {len(dups)}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(heights, bins=30, alpha=0.6, label="height")
    ax.hist(widths, bins=30, alpha=0.6, label="width")
    ax.set_xlabel("pixels"); ax.set_ylabel("count"); ax.legend()
    ax.set_title("Image dimension distribution")
    fig.savefig(fig_dir / "dimension_hist.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figures written to {fig_dir}/")


if __name__ == "__main__":
    main()
