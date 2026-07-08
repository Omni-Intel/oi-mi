"""Minimal S4D-like temporal model used as a pluggable research baseline."""

from __future__ import annotations

import torch
from torch import nn


class SimpleS4D(nn.Module):
    """A small temporal model with depthwise filtering and dropout."""

    def __init__(self, n_chans: int, n_times: int, n_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_chans, n_chans, kernel_size=15, padding=7, groups=n_chans),
            nn.Conv1d(n_chans, 64, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.5),
            nn.Conv1d(64, 64, kernel_size=15, padding=7),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(64, n_classes)
        self.n_times = n_times

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        flattened = features.squeeze(-1)
        return self.classifier(flattened)
