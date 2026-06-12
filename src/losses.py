"""Loss functions. Focal loss down-weights easy, well-classified examples so the
model concentrates on hard cases — useful for the harder meningioma class
(ablation AB-5).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha=None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        # alpha may be a per-class weight tensor or None
        self.register_buffer(
            "alpha", alpha if alpha is not None else None, persistent=False
        )
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        ce = F.cross_entropy(
            logits, target, weight=self.alpha,
            label_smoothing=self.label_smoothing, reduction="none",
        )
        pt = torch.exp(-ce)                      # prob of the true class
        focal = (1 - pt) ** self.gamma * ce
        return focal.mean()


def build_loss(cfg, class_weights=None):
    name = cfg["train"]["loss"]
    ls = cfg["train"]["label_smoothing"]
    if name == "ce":
        return nn.CrossEntropyLoss(label_smoothing=ls)
    if name == "weighted_ce":
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=ls)
    if name == "focal":
        return FocalLoss(gamma=cfg["train"]["focal_gamma"],
                         alpha=class_weights, label_smoothing=ls)
    raise ValueError(f"Unknown loss '{name}'")
