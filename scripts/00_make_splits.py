"""Phase 4 — build the single stratified train/val/test split that every model
shares. Run once before any training.

    python scripts/00_make_splits.py
"""
import _bootstrap  # noqa: F401
from pathlib import Path

from src.data.dataset import build_split_csvs
from src.utils import load_config, set_seed


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    raw_training = Path(cfg["paths"]["raw_dir"]) / "Training"
    train_df, val_df, test_df = build_split_csvs(
        raw_training, cfg["data"]["classes"], cfg["paths"]["splits_dir"],
        cfg["data"]["split"], seed=cfg["seed"],
    )
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
    for split, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dist = df["label"].value_counts().to_dict()
        print(f"  {split:5s} class distribution: {dist}")


if __name__ == "__main__":
    main()
