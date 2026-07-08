"""Lightweight convolutional Transformer for MI EEG decoding."""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionEncoding(nn.Module):
    """Deterministic positional encoding to avoid sequence-length-specific parameters."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        seq_len = tokens.shape[1]
        device = tokens.device
        positions = torch.arange(seq_len, device=device, dtype=tokens.dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=tokens.dtype)
            * (-math.log(10000.0) / self.d_model)
        )
        encoding = torch.zeros(seq_len, self.d_model, device=device, dtype=tokens.dtype)
        encoding[:, 0::2] = torch.sin(positions * div_term)
        encoding[:, 1::2] = torch.cos(positions * div_term[: encoding[:, 1::2].shape[1]])
        return tokens + encoding.unsqueeze(0)


class ConformerLite(nn.Module):
    """Small CNN + Transformer encoder baseline inspired by EEG Conformer/TCFormer."""

    def __init__(self, n_chans: int, n_times: int, n_classes: int, sfreq: float) -> None:
        super().__init__()
        del n_times, sfreq
        d_model = 64

        self.input_norm = nn.BatchNorm1d(n_chans)
        self.patch_embed = nn.Sequential(
            nn.Conv1d(n_chans, 32, kernel_size=25, stride=2, padding=12, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, d_model, kernel_size=15, stride=3, padding=7, groups=32, bias=False),
            nn.Conv1d(d_model, d_model, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(p=0.2),
        )
        self.position_encoding = SinusoidalPositionEncoding(d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=128,
            dropout=0.25,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, 64),
            nn.GELU(),
            nn.Dropout(p=0.35),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        tokens = self.patch_embed(x).transpose(1, 2)
        tokens = self.position_encoding(tokens)
        encoded = self.encoder(tokens)
        pooled = torch.cat([encoded.mean(dim=1), encoded.amax(dim=1)], dim=1)
        return self.classifier(pooled)
