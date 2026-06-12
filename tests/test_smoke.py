"""End-to-end smoke test on synthetic data — verifies the whole pipeline wiring
(splits -> dataset -> model -> train -> evaluate) without needing the real
dataset, a GPU, or internet. It uses the from-scratch custom_cnn so no pretrained
weights are downloaded.

    python tests/test_smoke.py

Expect it to finish in well under a minute on CPU and print 'SMOKE TEST PASSED'.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import cv2

from src.data.dataset import build_split_csvs, make_loaders
from src.evaluation.metrics import evaluate
from src.train import train_model
from src.utils import get_device, set_seed


def make_fake_dataset(root, classes, per_class=12, size=64):
    """Write random images so each class has a distinct mean intensity (so a CNN
    can actually learn something) under root/Training/<class>/."""
    rng = np.random.default_rng(0)
    for k, c in enumerate(classes):
        d = Path(root) / "Training" / c
        d.mkdir(parents=True, exist_ok=True)
        base = int(40 + k * 50)
        for i in range(per_class):
            img = np.clip(rng.normal(base, 15, (size, size, 3)), 0, 255).astype("uint8")
            cv2.imwrite(str(d / f"{c}_{i}.png"), img)


def main():
    set_seed(42)
    classes = ["glioma", "meningioma", "notumor", "pituitary"]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        make_fake_dataset(tmp, classes, per_class=16, size=64)
        cfg = {
            "seed": 42,
            "paths": {"raw_dir": str(tmp), "splits_dir": str(tmp / "splits"),
                      "results_dir": str(tmp / "results"),
                      "weights_dir": str(tmp / "weights"),
                      "figures_dir": str(tmp / "figures")},
            "data": {"classes": classes, "image_size": 64,
                     "clahe": {"enabled": True, "clip_limit": 2.0, "tile_grid": 8},
                     "split": {"train": 0.6, "val": 0.2, "test": 0.2}},
            "augment": {"rotation_deg": 10, "translate": 0.05, "zoom": 0.05,
                        "brightness": 0.1, "horizontal_flip": True,
                        "vertical_flip": False, "random_erasing": 0.0},
            "train": {"batch_size": 8, "num_workers": 0, "optimizer": "adamw",
                      "weight_decay": 1e-4, "label_smoothing": 0.0, "loss": "focal",
                      "focal_gamma": 2.0, "focal_alpha": 0.25, "scheduler": "cosine",
                      "warmup_epochs": 0, "grad_clip": 1.0, "mixed_precision": False,
                      "early_stopping_patience": 5,
                      "epochs": {"head": 1, "finetune": 1},
                      "lr": {"head": 1e-3, "finetune": 1e-4},
                      "finetune_unfreeze_blocks": 10},
            "eval": {"tta": False, "cv_folds": 2},
        }

        build_split_csvs(tmp / "Training", classes, cfg["paths"]["splits_dir"],
                         cfg["data"]["split"], seed=42)
        loaders = make_loaders(cfg)
        out = train_model("custom_cnn", cfg, loaders=loaders)
        res = evaluate(out["model"], loaders[2], get_device(), "custom_cnn",
                       classes, params_M=out["params_M"],
                       results_dir=cfg["paths"]["results_dir"])
        assert 0.0 <= res["f1_macro"] <= 1.0
        assert Path(cfg["paths"]["results_dir"], "custom_cnn.json").exists()
        print(f"accuracy={res['accuracy']}  macroF1={res['f1_macro']}")
        print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
