"""Dataset, transforms and split construction.

All models read the *same* train/val/test CSVs (created once by
scripts/00_make_splits.py), which is what makes the comparison fair.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .preprocessing import apply_clahe

# ImageNet statistics — the source domain of all pretrained backbones.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_split_csvs(raw_training_dir, classes, splits_dir, ratios, seed=42):
    """Enumerate every image under raw_training_dir/<class>/ and write
    stratified train/val/test CSVs. Returns the three dataframes."""
    raw_training_dir = Path(raw_training_dir)
    records = []
    for label_idx, cls in enumerate(classes):
        cls_dir = raw_training_dir / cls
        if not cls_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {cls_dir}")
        for fp in sorted(cls_dir.glob("*")):
            if fp.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                records.append({"path": str(fp), "label": cls, "label_idx": label_idx})

    df = pd.DataFrame(records).sample(frac=1, random_state=seed).reset_index(drop=True)

    val_test = ratios["val"] + ratios["test"]
    train_df, temp_df = train_test_split(
        df, test_size=val_test, stratify=df["label"], random_state=seed
    )
    rel_test = ratios["test"] / val_test
    val_df, test_df = train_test_split(
        temp_df, test_size=rel_test, stratify=temp_df["label"], random_state=seed
    )

    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)
    return train_df, val_df, test_df


class BrainTumorDataset(Dataset):
    """Reads images listed in a split CSV, applies CLAHE + transforms."""

    def __init__(self, csv_path, transform, size=224, clahe=True,
                 clip_limit=2.0, tile_grid=8):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.transform = transform
        self.size = size
        self.clahe = clahe
        self.clip_limit = clip_limit
        self.tile_grid = tile_grid

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row["path"], cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(row["path"])
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        if self.clahe:
            img = apply_clahe(img, self.clip_limit, self.tile_grid)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img)
        tensor = self.transform(pil)
        return tensor, int(row["label_idx"])

    @property
    def labels(self):
        return self.df["label_idx"].to_numpy()


def build_transforms(cfg, train: bool):
    """Torchvision transform pipeline. Augmentation applies to training only."""
    from torchvision import transforms

    size = cfg["data"]["image_size"]
    if train:
        a = cfg["augment"]
        ops = [
            transforms.RandomRotation(a["rotation_deg"]),
            transforms.RandomAffine(0, translate=(a["translate"], a["translate"]),
                                    scale=(1 - a["zoom"], 1 + a["zoom"])),
            transforms.ColorJitter(brightness=a["brightness"]),
        ]
        if a["horizontal_flip"]:
            ops.append(transforms.RandomHorizontalFlip(p=0.5))
        if a["vertical_flip"]:
            ops.append(transforms.RandomVerticalFlip(p=0.5))
        ops += [transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        if a.get("random_erasing", 0) > 0:
            ops.append(transforms.RandomErasing(p=a["random_erasing"]))
        return transforms.Compose(ops)

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def make_loaders(cfg, clahe_override=None, augment=True):
    """Build train/val/test DataLoaders from the shared split CSVs."""
    splits = Path(cfg["paths"]["splits_dir"])
    clahe = cfg["data"]["clahe"]["enabled"] if clahe_override is None else clahe_override
    common = dict(
        size=cfg["data"]["image_size"], clahe=clahe,
        clip_limit=cfg["data"]["clahe"]["clip_limit"],
        tile_grid=cfg["data"]["clahe"]["tile_grid"],
    )
    train_ds = BrainTumorDataset(splits / "train.csv",
                                 build_transforms(cfg, train=augment), **common)
    val_ds = BrainTumorDataset(splits / "val.csv",
                               build_transforms(cfg, train=False), **common)
    test_ds = BrainTumorDataset(splits / "test.csv",
                                build_transforms(cfg, train=False), **common)

    bs = cfg["train"]["batch_size"]
    nw = cfg["train"]["num_workers"]
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True),
        DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True),
        DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True),
    )


def compute_class_weights(cfg):
    """Inverse-frequency class weights for weighted CE / focal alpha."""
    from sklearn.utils.class_weight import compute_class_weight

    y = pd.read_csv(Path(cfg["paths"]["splits_dir"]) / "train.csv")["label_idx"].to_numpy()
    classes = np.unique(y)
    w = compute_class_weight("balanced", classes=classes, y=y)
    return torch.tensor(w, dtype=torch.float32)
