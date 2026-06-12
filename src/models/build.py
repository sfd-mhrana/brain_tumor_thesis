"""Model factory.

Every model is created through `build_model(name, cfg)` so that the classifier
head, attention insertion point and initialisation are identical across
architectures. Backbones come from `timm`, which exposes 800+ ImageNet-pretrained
models behind one uniform API — exactly what a fair benchmark needs.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

from .attention import CBAM, SEBlock
from .custom_cnn import CustomCNN

# Logical model name -> (timm backbone id, attention type)
MODEL_REGISTRY = {
    "custom_cnn":           (None,              None),
    "vgg16":                ("vgg16",           None),
    "resnet50":             ("resnet50",        None),
    "inception_v3":         ("inception_v3",    None),
    "densenet121":          ("densenet121",     None),
    "efficientnet_b0":      ("efficientnet_b0", None),
    "efficientnet_b3":      ("efficientnet_b3", None),
    "resnet50_se":          ("resnet50",        "se"),
    "efficientnet_b3_cbam": ("efficientnet_b3", "cbam"),   # PROPOSED MODEL
}


class AttentionClassifier(nn.Module):
    """Backbone -> optional attention -> global pool -> MLP head.

    The backbone is created with `num_classes=0, global_pool=''` so that
    `forward` yields the final convolutional feature map (B, C, H, W). This is
    where SE / CBAM attention is applied before pooling — the core of the
    proposed contribution.
    """

    def __init__(self, backbone_id, num_classes, attention=None, pretrained=True,
                 dropout=(0.4, 0.3)):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_id, pretrained=pretrained, num_classes=0, global_pool=""
        )
        # Infer the true channel dimension from an actual forward pass rather than
        # trusting `num_features`: with global_pool="" some backbones (e.g. VGG)
        # report a post-FC width that differs from the conv feature-map channels.
        was_training = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            probe = self.backbone(torch.zeros(1, 3, 224, 224))
            if probe.dim() == 2:
                probe = probe[:, :, None, None]
            feat_dim = probe.shape[1]
        self.backbone.train(was_training)

        if attention == "se":
            self.attention = SEBlock(feat_dim)
        elif attention == "cbam":
            self.attention = CBAM(feat_dim)
        else:
            self.attention = nn.Identity()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(feat_dim),
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout[0]),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout[1]),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        if x.dim() == 2:  # safety: some backbones may already pool
            x = x[:, :, None, None]
        x = self.attention(x)
        x = self.pool(x)
        return self.head(x)

    # --- two-stage fine-tuning helpers ---
    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_last(self, n_layers: int):
        """Unfreeze the last n parameter tensors of the backbone (stage 2)."""
        params = list(self.backbone.parameters())
        for p in params[-n_layers:]:
            p.requires_grad = True


def build_model(name: str, cfg: dict, pretrained: bool = True) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Known: {list(MODEL_REGISTRY)}")
    num_classes = len(cfg["data"]["classes"])

    if name == "custom_cnn":
        return CustomCNN(num_classes=num_classes)

    backbone_id, attention = MODEL_REGISTRY[name]
    return AttentionClassifier(
        backbone_id, num_classes, attention=attention, pretrained=pretrained
    )


def count_params(model: nn.Module) -> float:
    """Trainable + frozen parameter count in millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6
