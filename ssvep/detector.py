"""Lightweight FFT-based SSVEP detector for realtime game control."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


@dataclass(frozen=True, slots=True)
class SSVEPTarget:
    """One visual target mapped to a SSVEP stimulation frequency."""

    label: str
    frequency_hz: float
    direction: str


@dataclass(frozen=True, slots=True)
class SSVEPDetection:
    """A single SSVEP decoding result."""

    target: SSVEPTarget | None
    confidence: float
    scores: dict[str, float]
    stable: bool


DEFAULT_TARGETS: tuple[SSVEPTarget, ...] = (
    SSVEPTarget(label="左上", frequency_hz=10.0, direction="top_left"),
    SSVEPTarget(label="右上", frequency_hz=12.0, direction="top_right"),
    SSVEPTarget(label="左下", frequency_hz=15.0, direction="bottom_left"),
    SSVEPTarget(label="右下", frequency_hz=18.0, direction="bottom_right"),
)

DEFAULT_OCCIPITAL_CHANNELS_32: tuple[int, ...] = (14, 15, 16, 17, 18, 19, 20)


class SSVEPDetector:
    """Detect attended SSVEP targets from a recent EEG window.

    The detector intentionally stays simple for first-day demos: common-average
    reference, 6-45 Hz band-pass, optional 50 Hz notch, then FFT scoring at the
    fundamental and second harmonic of each target frequency.
    """

    def __init__(
        self,
        *,
        sfreq: float,
        targets: Sequence[SSVEPTarget] = DEFAULT_TARGETS,
        channel_indices: Sequence[int] | None = None,
        harmonics: Sequence[int] = (1, 2),
        low_hz: float = 6.0,
        high_hz: float = 45.0,
        notch_hz: float = 50.0,
        stability_windows: int = 2,
        min_confidence: float = 0.35,
    ) -> None:
        if sfreq <= 0:
            raise ValueError("sfreq must be positive.")
        if not targets:
            raise ValueError("At least one SSVEP target is required.")
        if stability_windows <= 0:
            raise ValueError("stability_windows must be positive.")

        self.sfreq = float(sfreq)
        self.targets = tuple(targets)
        self.channel_indices = tuple(channel_indices or DEFAULT_OCCIPITAL_CHANNELS_32)
        self.harmonics = tuple(int(h) for h in harmonics)
        self.low_hz = float(low_hz)
        self.high_hz = float(high_hz)
        self.notch_hz = float(notch_hz)
        self.min_confidence = float(min_confidence)
        self._recent: deque[str] = deque(maxlen=int(stability_windows))

    def reset(self) -> None:
        """Clear target stability history."""

        self._recent.clear()

    def predict(self, eeg: np.ndarray) -> SSVEPDetection:
        """Return the most likely attended target for an EEG window."""

        if eeg.ndim != 2:
            raise ValueError(f"Expected EEG shape (channels, samples), got {eeg.shape}.")
        if eeg.shape[1] < int(self.sfreq):
            raise ValueError("SSVEP window should contain at least 1 second of samples.")

        selected = self._select_channels(eeg)
        processed = self._preprocess(selected)
        scores = self._score_targets(processed)
        best_label = max(scores, key=scores.get)
        best_score = float(scores[best_label])
        total_score = float(sum(max(score, 0.0) for score in scores.values()))
        confidence = best_score / total_score if total_score > 0 else 0.0

        target = next(target for target in self.targets if target.label == best_label)
        if confidence >= self.min_confidence:
            self._recent.append(target.direction)
        else:
            self._recent.append("")
        stable = len(self._recent) == self._recent.maxlen and len(set(self._recent)) == 1 and self._recent[0] != ""
        return SSVEPDetection(target=target, confidence=float(confidence), scores=scores, stable=stable)

    def _select_channels(self, eeg: np.ndarray) -> np.ndarray:
        valid_indices = [idx for idx in self.channel_indices if 0 <= idx < eeg.shape[0]]
        if not valid_indices:
            valid_indices = list(range(eeg.shape[0]))
        return np.asarray(eeg[valid_indices, :], dtype=np.float32)

    def _preprocess(self, data: np.ndarray) -> np.ndarray:
        referenced = data - np.mean(data, axis=0, keepdims=True)
        nyquist = self.sfreq / 2.0
        high_hz = min(self.high_hz, nyquist - 1.0)
        if self.low_hz >= high_hz:
            filtered = referenced
        else:
            b_band, a_band = butter(4, [self.low_hz / nyquist, high_hz / nyquist], btype="bandpass")
            filtered = filtfilt(b_band, a_band, referenced, axis=1)
        if 0 < self.notch_hz < nyquist:
            b_notch, a_notch = iirnotch(self.notch_hz / nyquist, Q=30)
            filtered = filtfilt(b_notch, a_notch, filtered, axis=1)
        return np.asarray(filtered, dtype=np.float32)

    def _score_targets(self, data: np.ndarray) -> dict[str, float]:
        window = np.hanning(data.shape[1]).astype(np.float32)
        spectrum = np.abs(np.fft.rfft(data * window[None, :], axis=1)) ** 2
        freqs = np.fft.rfftfreq(data.shape[1], d=1.0 / self.sfreq)

        scores: dict[str, float] = {}
        for target in self.targets:
            score = 0.0
            for harmonic in self.harmonics:
                freq = target.frequency_hz * harmonic
                if freq >= self.sfreq / 2.0:
                    continue
                center_idx = int(np.argmin(np.abs(freqs - freq)))
                target_lo = max(center_idx - 1, 0)
                target_hi = min(center_idx + 2, spectrum.shape[1])
                noise_left = spectrum[:, max(center_idx - 8, 0) : max(center_idx - 3, 0)]
                noise_right = spectrum[:, min(center_idx + 4, spectrum.shape[1]) : min(center_idx + 9, spectrum.shape[1])]
                noise_bins = np.concatenate((noise_left, noise_right), axis=1)
                target_power = float(np.mean(spectrum[:, target_lo:target_hi]))
                noise_power = float(np.mean(noise_bins)) if noise_bins.size else 1e-12
                score += target_power / max(noise_power, 1e-12)
            scores[target.label] = score
        return scores
