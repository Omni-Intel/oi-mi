"""Shared calibration and realtime EEG preprocessing.

The default profile follows the parts of CBraMod's motor-imagery
preprocessing that are compatible with a sliding-window decoder:

* microvolt input
* common-average reference
* 0.3--40 Hz band-pass
* 200 Hz model rate (resampling is performed by the acquirer)

CBraMod filters continuous recordings and can use montage metadata to
interpolate known bad channels.  This project receives short windows without
montage coordinates, so it uses a deterministic robust spatial replacement
for obviously flat/non-finite/noisy channels.  The exact same function is
used by calibration, offline reconstruction, and realtime inference.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Any

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreprocessingConfig:
    """Parameters for the experiment's deterministic EEG transform."""

    low_hz: float = 0.3
    high_hz: float = 40.0
    filter_order: int = 5
    clip_uv: float = 150.0
    reject_peak_uv: float = 300.0
    max_clip_fraction: float = 0.01
    flat_std_uv: float = 0.05
    noisy_scale_ratio: float = 8.0
    noisy_scale_floor_uv: float = 50.0
    max_bad_channel_fraction: float = 0.2

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_PREPROCESSING = PreprocessingConfig()


@dataclass(frozen=True)
class WindowQuality:
    """Quality measurements made before numerical-safety clipping."""

    accepted: bool
    reasons: tuple[str, ...]
    bad_channel_indices: tuple[int, ...]
    bad_channel_fraction: float
    nonfinite_fraction: float
    clip_fraction: float
    peak_abs_uv: float

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        payload["bad_channel_indices"] = list(self.bad_channel_indices)
        return payload


@dataclass(frozen=True)
class PreprocessingResult:
    """Preprocessed model input and the associated quality report."""

    data: np.ndarray
    quality: WindowQuality


def resample_eeg(
    data: np.ndarray,
    *,
    source_sfreq: float,
    target_sfreq: float,
    axis: int = -1,
) -> np.ndarray:
    """Anti-alias and resample EEG while preserving every non-time dimension."""

    source = float(source_sfreq)
    target = float(target_sfreq)
    if source <= 0 or target <= 0:
        raise ValueError("Sampling frequencies must be positive.")
    if np.isclose(source, target):
        return np.asarray(data, dtype=np.float32).copy()

    ratio = Fraction(target / source).limit_denominator(10_000)
    resampled = resample_poly(
        np.asarray(data, dtype=np.float32),
        up=ratio.numerator,
        down=ratio.denominator,
        axis=axis,
    )
    return np.asarray(resampled, dtype=np.float32)


def common_average_reference(data: np.ndarray) -> np.ndarray:
    """Apply common average reference across channels."""

    array = np.asarray(data)
    if array.ndim != 2:
        raise ValueError(f"Expected EEG shaped (channels, time), got {array.shape}.")
    channel_mean = np.mean(array, axis=0, keepdims=True)
    return array - channel_mean


def bandpass_filter(
    data: np.ndarray,
    sfreq: float,
    low_hz: float = DEFAULT_PREPROCESSING.low_hz,
    high_hz: float = DEFAULT_PREPROCESSING.high_hz,
    order: int = DEFAULT_PREPROCESSING.filter_order,
) -> np.ndarray:
    """Apply a numerically stable zero-phase Butterworth band-pass."""

    array = np.asarray(data, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Expected EEG shaped (channels, time), got {array.shape}.")
    sampling_rate = float(sfreq)
    nyquist = sampling_rate / 2.0
    if sampling_rate <= 0:
        raise ValueError("sfreq must be positive.")
    if not 0.0 < float(low_hz) < float(high_hz) < nyquist:
        raise ValueError(
            f"Band-pass must satisfy 0 < low < high < Nyquist; got "
            f"{low_hz}-{high_hz} Hz at {sampling_rate} Hz."
        )
    if int(order) < 1:
        raise ValueError("Filter order must be at least 1.")

    sos = butter(
        int(order),
        [float(low_hz), float(high_hz)],
        btype="bandpass",
        fs=sampling_rate,
        output="sos",
    )
    try:
        filtered = sosfiltfilt(sos, array, axis=-1)
    except ValueError as exc:
        raise ValueError(
            f"EEG window with {array.shape[-1]} samples is too short for the "
            f"{order}th-order {low_hz}-{high_hz} Hz filter."
        ) from exc
    return np.asarray(filtered, dtype=np.float32)


def reject_artifacts(data: np.ndarray, clip_uv: float = 150.0) -> np.ndarray:
    """Winsorize extreme amplitudes for numerical safety.

    This function is retained for compatibility.  Clipping is not treated as
    artifact rejection; :func:`preprocess_eeg_window` reports whether the
    original window is suitable for training.
    """

    return np.clip(data, -float(clip_uv), float(clip_uv))


def _sanitize_and_repair_channels(
    data: np.ndarray,
    *,
    config: PreprocessingConfig,
) -> tuple[np.ndarray, tuple[int, ...], float]:
    """Replace invalid samples and spatially repair clearly bad channels."""

    array = np.asarray(data, dtype=np.float64)
    finite = np.isfinite(array)
    nonfinite_fraction = float(1.0 - np.mean(finite))
    sanitized = array.copy()
    for channel_index in range(sanitized.shape[0]):
        channel_finite = finite[channel_index]
        replacement = (
            float(np.median(sanitized[channel_index, channel_finite]))
            if np.any(channel_finite)
            else 0.0
        )
        sanitized[channel_index, ~channel_finite] = replacement

    centered = sanitized - np.median(sanitized, axis=-1, keepdims=True)
    channel_std = np.std(centered, axis=-1)
    channel_scale = 1.4826 * np.median(np.abs(centered), axis=-1)
    usable_scale = channel_scale[channel_scale >= config.flat_std_uv]
    typical_scale = float(np.median(usable_scale)) if usable_scale.size else 0.0
    noisy_limit = max(
        config.noisy_scale_floor_uv,
        config.noisy_scale_ratio * typical_scale,
    )
    bad_mask = (
        (channel_std < config.flat_std_uv)
        | (channel_scale > noisy_limit)
        | (~np.all(finite, axis=-1))
    )
    good_mask = ~bad_mask

    if np.any(bad_mask) and np.any(good_mask):
        # Montage coordinates are not available from the realtime forwarding
        # API.  The point-wise median of good channels is robust and preserves
        # shape/channel order without inventing a spatial neighbourhood.
        spatial_replacement = np.median(sanitized[good_mask], axis=0)
        sanitized[bad_mask] = spatial_replacement

    bad_indices = tuple(int(index) for index in np.flatnonzero(bad_mask))
    return sanitized, bad_indices, nonfinite_fraction


def preprocess_eeg_window(
    data: np.ndarray,
    sfreq: float,
    *,
    config: PreprocessingConfig = DEFAULT_PREPROCESSING,
) -> PreprocessingResult:
    """Preprocess one EEG window and report whether it is training quality."""

    array = np.asarray(data)
    if array.ndim != 2:
        raise ValueError(f"Expected EEG shaped (channels, time), got {array.shape}.")
    if array.shape[0] < 2:
        raise ValueError("At least two EEG channels are required for CAR.")
    if array.shape[1] < 2:
        raise ValueError("EEG window must contain at least two samples.")

    # Compatibility with legacy 64-channel recordings that appended one
    # trigger channel.  The Neuracle path already selects its configured 59
    # EEG channels before this function.
    if array.shape[0] == 65:
        array = array[:64, :]

    repaired, bad_indices, nonfinite_fraction = _sanitize_and_repair_channels(
        array,
        config=config,
    )
    referenced = common_average_reference(repaired)
    filtered = bandpass_filter(
        referenced,
        sfreq=sfreq,
        low_hz=config.low_hz,
        high_hz=config.high_hz,
        order=config.filter_order,
    )

    absolute = np.abs(filtered)
    peak_abs_uv = float(np.max(absolute))
    clip_fraction = float(np.mean(absolute > config.clip_uv))
    bad_channel_fraction = float(len(bad_indices) / filtered.shape[0])
    reasons: list[str] = []
    if nonfinite_fraction > 0.0:
        reasons.append("nonfinite_samples")
    if bad_channel_fraction > config.max_bad_channel_fraction:
        reasons.append("too_many_bad_channels")
    if peak_abs_uv > config.reject_peak_uv:
        reasons.append("extreme_amplitude")
    if clip_fraction > config.max_clip_fraction:
        reasons.append("excessive_clipping")

    quality = WindowQuality(
        accepted=not reasons,
        reasons=tuple(reasons),
        bad_channel_indices=bad_indices,
        bad_channel_fraction=bad_channel_fraction,
        nonfinite_fraction=nonfinite_fraction,
        clip_fraction=clip_fraction,
        peak_abs_uv=peak_abs_uv,
    )
    cleaned = reject_artifacts(filtered, clip_uv=config.clip_uv).astype(np.float32)
    return PreprocessingResult(data=cleaned, quality=quality)


def filter_and_transform(data: np.ndarray, sfreq: float) -> np.ndarray:
    """Return the shared model input used by training and realtime decoding."""

    return preprocess_eeg_window(data, sfreq=sfreq).data
