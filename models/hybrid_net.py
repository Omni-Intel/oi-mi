"""Hybrid spectral-temporal EEG model for online MI decoding."""

from __future__ import annotations

import torch
from torch import nn


class _BranchBlock(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_chans, out_chans, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_chans),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class HybridSpectralTemporalNet(nn.Module):
    """Multi-scale temporal CNN with a lightweight spectral summary branch."""

    def __init__(self, n_chans: int, n_times: int, n_classes: int, sfreq: float) -> None:
        super().__init__()
        self.n_chans = n_chans
        self.n_times = n_times
        self.sfreq = float(sfreq)
        branch_width = 24

        self.input_norm = nn.BatchNorm1d(n_chans)
        self.branches = nn.ModuleList(
            [
                _BranchBlock(n_chans, branch_width, kernel_size=7),
                _BranchBlock(n_chans, branch_width, kernel_size=15),
                _BranchBlock(n_chans, branch_width, kernel_size=31),
                _BranchBlock(n_chans, branch_width, kernel_size=63),
            ]
        )
        merged_chans = branch_width * len(self.branches)

        self.temporal_mixer = nn.Sequential(
            nn.Conv1d(merged_chans, merged_chans, kernel_size=15, padding=7, groups=merged_chans, bias=False),
            nn.Conv1d(merged_chans, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(p=0.35),
            nn.Conv1d(128, 128, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )

        self.temporal_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(128, 32, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(32, 128, kernel_size=1),
            nn.Sigmoid(),
        )

        self.spectral_projector = nn.Sequential(
            nn.Linear(n_chans * 4, 128),
            nn.GELU(),
            nn.Dropout(p=0.25),
            nn.Linear(128, 64),
            nn.GELU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 2 + 64, 128),
            nn.GELU(),
            nn.Dropout(p=0.35),
            nn.Linear(128, n_classes),
        )

        band_edges = torch.tensor([[8.0, 12.0], [12.0, 20.0], [20.0, 30.0], [8.0, 30.0]], dtype=torch.float32)
        self.register_buffer("band_edges", band_edges, persistent=False)

    def _spectral_features(self, x: torch.Tensor) -> torch.Tensor:
        freqs = torch.fft.rfftfreq(self.n_times, d=1.0 / self.sfreq).to(x.device)
        spectrum = torch.fft.rfft(x, dim=-1)
        power = spectrum.real.square() + spectrum.imag.square()
        band_features: list[torch.Tensor] = []
        for lo, hi in self.band_edges:
            mask = (freqs >= lo) & (freqs < hi)
            if torch.any(mask):
                band_power = power[..., mask].mean(dim=-1)
            else:
                band_power = torch.zeros(x.shape[0], x.shape[1], device=x.device, dtype=x.dtype)
            band_features.append(torch.log1p(band_power))
        return torch.cat(band_features, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.input_norm(x)
        branch_outputs = [branch(normalized) for branch in self.branches]
        merged = torch.cat(branch_outputs, dim=1)

        temporal = self.temporal_mixer(merged)
        temporal = temporal * self.temporal_gate(temporal)
        pooled_mean = temporal.mean(dim=-1)
        pooled_max = temporal.amax(dim=-1)

        spectral = self._spectral_features(normalized)
        spectral_embed = self.spectral_projector(spectral)

        fused = torch.cat([pooled_mean, pooled_max, spectral_embed], dim=1)
        return self.classifier(fused)
