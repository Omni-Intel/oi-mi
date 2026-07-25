"""Rebuild calibration windows from raw-rate continuous EEG and trial metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from utils.preprocessing import (
    DEFAULT_PREPROCESSING,
    preprocess_eeg_window,
    resample_eeg,
)


def _window_specs(
    *,
    window_sec: float,
    stride_sec: float,
    control_start_sec: float,
    control_stop_sec: float,
) -> list[float]:
    last_start = control_stop_sec - window_sec
    if last_start < control_start_sec:
        return []
    count = int(np.floor((last_start - control_start_sec) / stride_sec + 1e-9)) + 1
    return [control_start_sec + index * stride_sec for index in range(count)]


def build_windows(
    continuous_eeg: np.ndarray,
    trials: list[dict[str, Any]],
    *,
    source_sfreq: float,
    target_sfreq: float,
    window_sec: float,
    stride_sec: float,
    control_start_sec: float,
    control_stop_sec: float,
    channel_indices: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Create target-rate raw and processed windows without changing source files."""

    if continuous_eeg.ndim != 2:
        raise ValueError(f"Expected continuous EEG shaped (channels, time), got {continuous_eeg.shape}.")

    source_window_samples = int(round(window_sec * source_sfreq))
    target_window_samples = int(round(window_sec * target_sfreq))
    offsets_sec = _window_specs(
        window_sec=window_sec,
        stride_sec=stride_sec,
        control_start_sec=control_start_sec,
        control_stop_sec=control_stop_sec,
    )
    if not offsets_sec:
        raise ValueError("The configured control interval cannot contain a complete window.")

    raw_windows: list[np.ndarray] = []
    processed_windows: list[np.ndarray] = []
    labels: list[int] = []
    trial_ids: list[int] = []
    block_indices: list[int] = []
    trial_indices: list[int] = []
    windows_in_trial: list[int] = []
    starts_source: list[int] = []
    starts_target: list[int] = []
    clip_fractions: list[float] = []
    peak_abs_uv: list[float] = []
    bad_channel_fractions: list[float] = []
    rejected_windows = 0
    selected_channels = (
        np.arange(continuous_eeg.shape[0], dtype=np.int64)
        if channel_indices is None
        else np.asarray(channel_indices, dtype=np.int64)
    )
    if selected_channels.ndim != 1 or selected_channels.size == 0:
        raise ValueError("channel_indices must select at least one channel.")
    if selected_channels.min() < 0 or selected_channels.max() >= continuous_eeg.shape[0]:
        raise ValueError("channel_indices contains a channel outside the continuous EEG array.")

    for trial_id, trial in enumerate(trials):
        control_on = int(trial["control_on_sample"])
        for window_in_trial, offset_sec in enumerate(offsets_sec):
            start_source = control_on + int(round(offset_sec * source_sfreq))
            stop_source = start_source + source_window_samples
            if start_source < 0 or stop_source > continuous_eeg.shape[1]:
                continue

            source_window = np.asarray(
                continuous_eeg[selected_channels, start_source:stop_source],
                dtype=np.float32,
            )
            target_window = resample_eeg(
                source_window,
                source_sfreq=source_sfreq,
                target_sfreq=target_sfreq,
            )
            if target_window.shape[-1] != target_window_samples:
                raise RuntimeError(
                    f"Resampling produced {target_window.shape[-1]} points; "
                    f"expected {target_window_samples}."
                )

            result = preprocess_eeg_window(target_window, sfreq=target_sfreq)
            if not result.quality.accepted:
                rejected_windows += 1
                continue

            raw_windows.append(target_window)
            clip_fractions.append(result.quality.clip_fraction)
            peak_abs_uv.append(result.quality.peak_abs_uv)
            bad_channel_fractions.append(result.quality.bad_channel_fraction)
            processed_windows.append(result.data)
            labels.append(int(trial["label_id"]))
            trial_ids.append(trial_id)
            block_indices.append(int(trial.get("block_index", -1)))
            trial_indices.append(int(trial.get("trial_index", trial_id)))
            windows_in_trial.append(window_in_trial)
            starts_source.append(start_source)
            starts_target.append(int(round(start_source * target_sfreq / source_sfreq)))

    if not raw_windows:
        raise RuntimeError("No windows could be reconstructed from the supplied EEG and trial metadata.")

    return {
        "raw_windows": np.stack(raw_windows).astype(np.float32),
        "processed_windows": np.stack(processed_windows).astype(np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "trial_ids": np.asarray(trial_ids, dtype=np.int64),
        "block_indices": np.asarray(block_indices, dtype=np.int64),
        "trial_indices": np.asarray(trial_indices, dtype=np.int64),
        "window_indices": np.asarray(windows_in_trial, dtype=np.int64),
        "window_start_source": np.asarray(starts_source, dtype=np.int64),
        "window_start_target": np.asarray(starts_target, dtype=np.int64),
        "quality_clip_fraction": np.asarray(clip_fractions, dtype=np.float32),
        "quality_peak_abs_uv": np.asarray(peak_abs_uv, dtype=np.float32),
        "quality_bad_channel_fraction": np.asarray(
            bad_channel_fractions,
            dtype=np.float32,
        ),
        "quality_rejected_windows": np.asarray([rejected_windows], dtype=np.int64),
        "selected_channels": selected_channels,
        "source_sfreq": np.asarray([source_sfreq], dtype=np.float32),
        "sfreq": np.asarray([target_sfreq], dtype=np.float32),
        "window_sec": np.asarray([window_sec], dtype=np.float32),
        "step_sec": np.asarray([stride_sec], dtype=np.float32),
    }


def _save_dataset(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def reprocess_session(
    session_dir: Path,
    output_dir: Path,
    *,
    source_sfreq: float,
    target_sfreq: float,
    eeg_channel_count: int | None = None,
) -> list[Path]:
    metadata_path = session_dir / "metadata.json"
    continuous_path = session_dir / "continuous_eeg.npy"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    continuous = np.load(continuous_path, mmap_mode="r")
    if eeg_channel_count is not None:
        if eeg_channel_count <= 0 or eeg_channel_count > continuous.shape[0]:
            raise ValueError(
                f"eeg_channel_count must be between 1 and {continuous.shape[0]}, "
                f"got {eeg_channel_count}."
            )
        channel_indices = np.arange(eeg_channel_count, dtype=np.int64)
    else:
        channel_indices = None

    control_start_sec, control_stop_sec = (
        float(value) for value in metadata["control_window_range_sec"]
    )
    stride_sec = float(metadata.get("stride_sec", 0.5))
    trials = list(metadata["trials"])

    datasets = [
        ("training_windows_main_corrected.npz", float(metadata["window_sec"]), stride_sec),
    ]
    if (session_dir / "training_windows_aux_1p5s.npz").exists():
        datasets.append(("training_windows_aux_1p5s_corrected.npz", 1.5, stride_sec))

    written: list[Path] = []
    for filename, window_sec, dataset_stride in datasets:
        payload = build_windows(
            continuous,
            trials,
            source_sfreq=source_sfreq,
            target_sfreq=target_sfreq,
            window_sec=window_sec,
            stride_sec=dataset_stride,
            control_start_sec=control_start_sec,
            control_stop_sec=control_stop_sec,
            channel_indices=channel_indices,
        )
        output_path = output_dir / filename
        _save_dataset(output_path, payload)
        written.append(output_path)

    corrected_metadata = dict(metadata)
    corrected_metadata["preprocessing"] = {
        "source_sfreq": source_sfreq,
        "target_sfreq": target_sfreq,
        "resampling": "scipy.signal.resample_poly",
        "bandpass_hz": [
            DEFAULT_PREPROCESSING.low_hz,
            DEFAULT_PREPROCESSING.high_hz,
        ],
        "bandpass_design": (
            f"Butterworth SOS zero-phase order {DEFAULT_PREPROCESSING.filter_order}"
        ),
        "reference": "common_average",
        "bad_channel_repair": "pointwise_median_of_good_channels",
        "artifact_clip_uv": DEFAULT_PREPROCESSING.clip_uv,
        "artifact_reject_peak_uv": DEFAULT_PREPROCESSING.reject_peak_uv,
        "artifact_max_clip_fraction": DEFAULT_PREPROCESSING.max_clip_fraction,
        "selected_channel_indices": (
            list(range(eeg_channel_count)) if eeg_channel_count is not None else "all"
        ),
        "grouping_fields": ["trial_ids", "block_indices", "trial_indices", "window_indices"],
        "quality_fields": [
            "quality_clip_fraction",
            "quality_peak_abs_uv",
            "quality_bad_channel_fraction",
            "quality_rejected_windows",
        ],
    }
    (output_dir / "metadata_corrected.json").write_text(
        json.dumps(corrected_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source-sfreq", type=float, required=True)
    parser.add_argument("--target-sfreq", type=float, default=200.0)
    parser.add_argument(
        "--eeg-channel-count",
        type=int,
        help="Keep only the leading EEG channels; excludes trailing ECG/EOG auxiliaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    written = reprocess_session(
        args.session_dir.resolve(),
        args.output_dir.resolve(),
        source_sfreq=args.source_sfreq,
        target_sfreq=args.target_sfreq,
        eeg_channel_count=args.eeg_channel_count,
    )
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
