"""A small CNN trained from scratch. It establishes the performance floor that
every transfer-learning model must beat (~1M parameters).
"""
from __future__ import annotations

import torch.nn as nn


class CustomCNN(nn.Module):
    def __init__(self, num_classes: int = 4, in_channels: int = 3):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(in_channels, 32),   # 224 -> 112
            block(32, 64),            # 112 -> 56
            block(64, 128),           # 56 -> 28
            block(128, 128),          # 28 -> 14
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)
