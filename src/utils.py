"""Shared utilities: reproducible seeding, config loading, device selection."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import yaml


def set_seed(seed: int = 42) -> None:
    """Seed every RNG we rely on so results are reproducible.

    Reviewers can reject a paper whose numbers cannot be reproduced, so this is
    called at the start of every script.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic cuDNN trades a little speed for reproducibility.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def load_config(path: str | os.PathLike = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
