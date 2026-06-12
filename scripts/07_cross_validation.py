"""Phase 9 — 5-fold stratified cross-validation for the proposed model (and any
others passed). Reports mean +/- std and 95% CI of macro-F1.

    python scripts/07_cross_validation.py --models efficientnet_b3_cbam efficientnet_b3
"""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

from src.data.dataset import BrainTumorDataset, build_transforms
from src.evaluation.metrics import predict
from src.evaluation.stats_tests import mean_ci
from src.train import train_model
from src.utils import get_device, load_config, set_seed
from sklearn.metrics import f1_score


def _combined_dataset(cfg, train_transform):
    """Pool train+val (test is held out) for cross-validation."""
    splits = Path(cfg["paths"]["splits_dir"])
    common = dict(size=cfg["data"]["image_size"],
                  clahe=cfg["data"]["clahe"]["enabled"],
                  clip_limit=cfg["data"]["clahe"]["clip_limit"],
                  tile_grid=cfg["data"]["clahe"]["tile_grid"])
    train = BrainTumorDataset(splits / "train.csv", train_transform, **common)
    val = BrainTumorDataset(splits / "val.csv", train_transform, **common)
    df = pd.concat([train.df, val.df]).reset_index(drop=True)
    ds = BrainTumorDataset(splits / "train.csv", train_transform, **common)
    ds.df = df
    return ds


def main():
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=[cfg["proposed_model"]])
    parser.add_argument("--folds", type=int, default=cfg["eval"]["cv_folds"])
    args = parser.parse_args()

    device = get_device()
    results_dir = Path(cfg["paths"]["results_dir"]); results_dir.mkdir(exist_ok=True, parents=True)
    eval_tf = build_transforms(cfg, train=False)
    ds = _combined_dataset(cfg, build_transforms(cfg, train=True))
    y_all = ds.df["label_idx"].to_numpy()

    out = {}
    for name in args.models:
        fold_f1 = []
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=cfg["seed"])
        for fold, (tr, va) in enumerate(skf.split(np.zeros(len(y_all)), y_all)):
            set_seed(cfg["seed"] + fold)
            train_loader = DataLoader(Subset(ds, tr), batch_size=cfg["train"]["batch_size"],
                                      shuffle=True, num_workers=cfg["train"]["num_workers"])
            val_ds = BrainTumorDataset(Path(cfg["paths"]["splits_dir"]) / "train.csv",
                                       eval_tf, size=cfg["data"]["image_size"],
                                       clahe=cfg["data"]["clahe"]["enabled"])
            val_ds.df = ds.df.iloc[va].reset_index(drop=True)
            val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"],
                                    shuffle=False, num_workers=cfg["train"]["num_workers"])
            res = train_model(name, cfg, loaders=(train_loader, val_loader, val_loader))
            y_true, y_pred, _ = predict(res["model"], val_loader, device)
            f1 = f1_score(y_true, y_pred, average="macro")
            fold_f1.append(f1)
            print(f"  {name} fold {fold+1}/{args.folds}: macroF1={f1:.4f}")
        out[name] = {"fold_f1": fold_f1, **mean_ci(fold_f1)}
        print(f"{name}: {out[name]['mean']:.4f} +/- {out[name]['std']:.4f} "
              f"(95% CI [{out[name]['ci_low']:.4f}, {out[name]['ci_high']:.4f}])")

    with open(results_dir / "cross_validation.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nCV results -> {results_dir}/cross_validation.json")


if __name__ == "__main__":
    main()
