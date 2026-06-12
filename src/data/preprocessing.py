"""Image preprocessing — CLAHE contrast enhancement applied on the luminance
channel only, which sharpens tumour boundaries without distorting colour
statistics. Saving both raw and CLAHE versions enables ablation AB-1.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: int = 8,
) -> np.ndarray:
    """Apply CLAHE to the L channel of a BGR image and return a BGR image."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    l_enhanced = clahe.apply(l)
    merged = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def load_and_preprocess(
    image_path: str | Path,
    size: int = 224,
    clahe: bool = True,
    clip_limit: float = 2.0,
    tile_grid: int = 8,
) -> np.ndarray:
    """Read an image from disk, resize, optionally CLAHE-enhance.

    Returns an RGB uint8 array of shape (size, size, 3). Grayscale MRIs are
    promoted to 3 channels because ImageNet-pretrained backbones expect RGB.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    if clahe:
        img = apply_clahe(img, clip_limit, tile_grid)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
