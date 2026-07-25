"""Protocol-driven motor imagery calibration."""

from __future__ import annotations

from collections.abc import Callable
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from rich.console import Console

from acquisition.base import AbstractAcquirer
from adaptation.mi_protocol import (
    LABEL_DESCRIPTION,
    LABEL_DISPLAY,
    LABEL_SYMBOL,
    LABEL_TO_ID,
    RECOMMENDED_INSTRUCTIONS,
    ProtocolConfig,
    SessionPlan,
    build_session_plan,
)
from adaptation.neuroonline import NeuroOnlineConfig, NeuroOnlineModelAdapter
from adaptation.session_recorder import SessionRecorder
from models.factory import BaseModelAdapter, TorchModelAdapter
from utils.markers import MarkerBackend, PROTOCOL_EVENT_CODES
from utils.preprocessing import (
    DEFAULT_PREPROCESSING,
    preprocess_eeg_window,
    resample_eeg,
)

LABEL_SEQUENCE: list[tuple[int, str]] = [(LABEL_TO_ID[label], label) for label in ("left", "right", "idle")]


@dataclass(slots=True)
class CalibrationResult:
    """Result metadata for a calibration run."""

    model_path: Path
    metrics: dict[str, float]
    windows_collected: int
    calibration_data_path: Path | None = None
    session_dir: Path | None = None


class Calibrator:
    """Collect continuous MI protocol data and train or adapt a decoder."""

    def __init__(
        self,
        acquirer: AbstractAcquirer,
        model: BaseModelAdapter,
        marker_backend: MarkerBackend,
        console: Console,
        *,
        sfreq: float,
        window_sec: float,
        step_sec: float,
        model_path: Path,
        calibration_records_dir: Path | None = None,
        protocol_config: ProtocolConfig | None = None,
        online_adaptation_config: dict | None = None,
        experiment_config: dict[str, Any] | None = None,
    ) -> None:
        self._acquirer = acquirer
        self._neuroonline_config = NeuroOnlineConfig.from_mapping(online_adaptation_config)
        if self._neuroonline_config.enabled:
            if not isinstance(model, TorchModelAdapter):
                raise ValueError("NeuroOnline calibration requires a PyTorch decoder model.")
            self._model = NeuroOnlineModelAdapter(
                model,
                config=self._neuroonline_config,
                state_path=None,
            )
        else:
            self._model = model
        self._marker_backend = marker_backend
        self._console = console
        self._sfreq = float(sfreq)
        self._source_sfreq = float(getattr(acquirer, "source_sfreq", self._sfreq))
        self._window_sec = float(window_sec)
        self._step_sec = float(step_sec)
        self._model_path = model_path
        self._calibration_records_dir = calibration_records_dir
        self._experiment_config = copy.deepcopy(experiment_config or {})
        self._protocol = protocol_config or ProtocolConfig.from_config(
            {
                "window_sec": float(window_sec),
                "step_sec": float(step_sec),
            }
        )

    def calibrate(
        self,
        *,
        duration_sec: int | None,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        patience: int,
        head_only: bool,
        include_practice: bool = True,
        heartbeat: Callable[[], None] | None = None,
    ) -> CalibrationResult:
        del duration_sec
        if head_only:
            raise ValueError(
                "Head-only calibration was removed; each experiment must train "
                "a fresh full decoder."
            )
        plan = build_session_plan(self._protocol)
        (
            session_dir,
            raw_windows,
            processed_windows,
            labels,
            trial_groups,
            session_metadata,
        ) = self._collect_training_data(
            plan=plan,
            include_practice=include_practice,
            heartbeat=heartbeat,
        )
        self._console.print("[bold cyan]采集完成，正在保存和训练，请等待工作人员[/bold cyan]")
        if self._neuroonline_config.enabled:
            self._console.print(
                "[bold yellow]正在执行 NeuroOnline 离线训练 "
                f"(最多 {self._neuroonline_config.offline_epochs} epochs，"
                f"patience={self._neuroonline_config.offline_patience})。"
                "在出现“校准完成”和模型保存路径前，请勿返回、刷新或关闭页面。[/bold yellow]"
            )
        if heartbeat is not None:
            heartbeat()
        training_progress = getattr(self._console, "set_stage_progress", None)
        if callable(training_progress):
            training_progress(
                stage_name="模型训练",
                elapsed_sec=0.0,
                duration_sec=float(
                    self._neuroonline_config.offline_epochs
                    if self._neuroonline_config.enabled
                    else epochs
                ),
            )

        def report_training_progress(
            current_epoch: int,
            total_epochs: int,
            epoch_metrics: dict[str, float],
        ) -> None:
            del epoch_metrics
            if callable(training_progress):
                training_progress(
                    stage_name=f"模型训练 epoch {current_epoch}/{total_epochs}",
                    elapsed_sec=float(current_epoch),
                    duration_sec=float(total_epochs),
                )
            if heartbeat is not None:
                heartbeat()

        metrics = self._model.fit(
            processed_windows,
            labels,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=patience,
            head_only=False,
            groups=trial_groups,
            progress_callback=report_training_progress,
        )
        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(self._model_path)
        self._save_metadata(metrics=metrics, windows_collected=int(processed_windows.shape[0]), head_only=False)
        self._write_session_summary(session_dir, metrics=metrics, windows_collected=int(processed_windows.shape[0]), session_metadata=session_metadata)
        self._seal_session_bundle(session_dir)
        self._console.print("[bold green]校准完成，请等待工作人员[/bold green]")
        if heartbeat is not None:
            heartbeat()
        return CalibrationResult(
            model_path=self._model_path,
            metrics=metrics,
            windows_collected=int(processed_windows.shape[0]),
            calibration_data_path=(session_dir / "training_windows_main.npz") if session_dir is not None else None,
            session_dir=session_dir,
        )

    def _collect_training_data(
        self,
        *,
        plan: SessionPlan,
        include_practice: bool = True,
        heartbeat: Callable[[], None] | None = None,
    ) -> tuple[
        Path | None,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[str, Any],
    ]:
        self._console.print("[bold cyan]开始按 MI game control protocol 采集[/bold cyan]")
        self._print_instructions(plan)
        self._acquirer.start_stream()
        recorder = SessionRecorder(
            self._acquirer,
            sfreq=self._source_sfreq,
            n_channels=self._acquirer.metadata.n_channels,
        )
        session_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self._calibration_records_dir / session_stamp if self._calibration_records_dir is not None else None
        trials: list[dict[str, Any]] = []
        try:
            self._emit_event(recorder, "session_start", phase="session", subject_mode=plan.subject_mode)
            if include_practice and plan.practice_labels:
                self._run_practice(plan, recorder=recorder, heartbeat=heartbeat)
            self._run_baseline(plan, recorder=recorder, heartbeat=heartbeat)
            self._run_formal_blocks(plan, recorder=recorder, heartbeat=heartbeat, trials=trials)
            self._emit_event(recorder, "session_end", phase="session")
        finally:
            self._flush_recorder(recorder)
            self._acquirer.stop_stream()
            if heartbeat is not None:
                heartbeat()

        session_metadata = self._build_session_metadata(plan, session_stamp=session_stamp, trials=trials)
        if session_dir is not None:
            recorder.export(session_dir, metadata=session_metadata)
        eeg = self._get_continuous_eeg(session_dir=session_dir, recorder=recorder)
        raw_windows, processed_windows, labels, trial_groups = self._build_training_windows(
            eeg=eeg,
            events=recorder.events,
            trials=trials,
            session_dir=session_dir,
        )
        if raw_windows.shape[0] == 0:
            raise RuntimeError("Calibration did not yield any valid training windows.")
        return (
            session_dir,
            raw_windows,
            processed_windows,
            labels,
            trial_groups,
            session_metadata,
        )

    def _run_practice(
        self,
        plan: SessionPlan,
        *,
        recorder: SessionRecorder,
        heartbeat: Callable[[], None] | None,
    ) -> None:
        self._console.print(f"[bold yellow]接下来是 {len(plan.practice_labels)} 个练习 trial，用于熟悉流程[/bold yellow]")
        self._sleep_with_recording(3.0, recorder=recorder, heartbeat=heartbeat, stage_name="练习说明")
        for index, label in enumerate(plan.practice_labels, start=1):
            self._console.print(f"[bold yellow]练习 {index}/{len(plan.practice_labels)}[/bold yellow] {LABEL_DISPLAY[label]} {LABEL_DESCRIPTION[label]}")
            self._run_trial(
                label=label,
                recorder=recorder,
                heartbeat=heartbeat,
                trial_index=index - 1,
                block_index=-1,
                phase="practice",
                collect_trial=False,
            )
        self._console.print("[bold cyan]练习结束，接下来开始正式采集[/bold cyan]")
        self._sleep_with_recording(3.0, recorder=recorder, heartbeat=heartbeat, stage_name="练习结束")

    def _run_baseline(
        self,
        plan: SessionPlan,
        *,
        recorder: SessionRecorder,
        heartbeat: Callable[[], None] | None,
    ) -> None:
        for segment in plan.baseline_segments:
            self._console.print(f"[bold yellow]Baseline[/bold yellow] {segment.instruction} ({segment.duration_sec:.0f}s)")
            self._emit_event(recorder, "baseline_start", phase="baseline", segment_name=segment.name)
            self._sleep_with_recording(
                segment.duration_sec,
                recorder=recorder,
                heartbeat=heartbeat,
                stage_name=f"Baseline: {segment.name}",
            )
            self._emit_event(recorder, "baseline_end", phase="baseline", segment_name=segment.name)

    def _run_formal_blocks(
        self,
        plan: SessionPlan,
        *,
        recorder: SessionRecorder,
        heartbeat: Callable[[], None] | None,
        trials: list[dict[str, Any]],
    ) -> None:
        total_blocks = len(plan.blocks)
        for block_index, sequence in enumerate(plan.blocks):
            self._console.print(f"[bold cyan]Block {block_index + 1}/{total_blocks}[/bold cyan] 共 {len(sequence)} 个 trial")
            self._emit_event(recorder, "block_start", phase="formal", block_index=block_index)
            for trial_index, label in enumerate(sequence):
                trial_info = self._run_trial(
                    label=label,
                    recorder=recorder,
                    heartbeat=heartbeat,
                    trial_index=trial_index,
                    block_index=block_index,
                    phase="formal",
                    collect_trial=True,
                )
                if trial_info is not None:
                    trials.append(trial_info)
            self._emit_event(recorder, "block_end", phase="formal", block_index=block_index)
            if block_index < total_blocks - 1:
                self._console.print(
                    f"[bold yellow]休息 {plan.rest_between_blocks_sec:.0f} 秒，请放松但不要大幅动作[/bold yellow]"
                )
                self._sleep_with_recording(
                    plan.rest_between_blocks_sec,
                    recorder=recorder,
                    heartbeat=heartbeat,
                    stage_name=f"Block {block_index + 1} 休息",
                )

    def _run_trial(
        self,
        *,
        label: str,
        recorder: SessionRecorder,
        heartbeat: Callable[[], None] | None,
        trial_index: int,
        block_index: int,
        phase: str,
        collect_trial: bool,
    ) -> dict[str, Any] | None:
        trial_timing = self._protocol.trial_timing
        self._console.print("[bold yellow]PRACTICE_FIXATION[/bold yellow]" if phase == "practice" else "[bold yellow]FIXATION[/bold yellow]")
        self._emit_event(
            recorder,
            "fixation_on",
            phase=phase,
            block_index=block_index,
            trial_index=trial_index,
            label=label,
        )
        stage_prefix = "练习" if phase == "practice" else f"Block {block_index + 1} / Trial {trial_index + 1}"
        self._sleep_with_recording(
            trial_timing.fixation_sec,
            recorder=recorder,
            heartbeat=heartbeat,
            stage_name=f"{stage_prefix}: fixation",
        )

        cue_event = f"cue_{label}_on"
        cue_message = f"PRACTICE {LABEL_SYMBOL[label]} {LABEL_DISPLAY[label]}" if phase == "practice" else f"{LABEL_SYMBOL[label]} {LABEL_DISPLAY[label]}"
        self._console.print(f"[bold yellow]{cue_message}[/bold yellow]")
        self._emit_event(
            recorder,
            cue_event,
            phase=phase,
            block_index=block_index,
            trial_index=trial_index,
            label=label,
        )
        self._sleep_with_recording(
            trial_timing.cue_sec,
            recorder=recorder,
            heartbeat=heartbeat,
            stage_name=f"{stage_prefix}: cue {label}",
        )

        control_on_event = self._emit_event(
            recorder,
            "control_on",
            phase=phase,
            block_index=block_index,
            trial_index=trial_index,
            label=label,
        )
        control_on_sample = int(control_on_event.sample_index)
        self._sleep_with_recording(
            trial_timing.control_sec,
            recorder=recorder,
            heartbeat=heartbeat,
            stage_name=f"{stage_prefix}: control {label}",
        )
        control_off_event = self._emit_event(
            recorder,
            "control_off",
            phase=phase,
            block_index=block_index,
            trial_index=trial_index,
            label=label,
        )
        control_off_sample = int(control_off_event.sample_index)

        self._emit_event(
            recorder,
            "iti_on",
            phase=phase,
            block_index=block_index,
            trial_index=trial_index,
            label=label,
        )
        self._console.print("[bold yellow]PRACTICE_ITI[/bold yellow]" if phase == "practice" else "[bold yellow]ITI[/bold yellow]")
        self._sleep_with_recording(
            trial_timing.iti_sec,
            recorder=recorder,
            heartbeat=heartbeat,
            stage_name=f"{stage_prefix}: iti",
        )
        if not collect_trial:
            return None
        return {
            "phase": phase,
            "block_index": block_index,
            "trial_index": trial_index,
            "label": label,
            "label_id": LABEL_TO_ID[label],
            "control_on_sample": control_on_sample,
            "control_off_sample": control_off_sample,
        }

    def _build_training_windows(
        self,
        *,
        eeg: np.ndarray,
        events: list[Any],
        trials: list[dict[str, Any]],
        session_dir: Path | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        del events
        source_window_samples = int(round(self._protocol.window_sec * self._source_sfreq))
        target_window_samples = int(round(self._protocol.window_sec * self._sfreq))
        stride_samples = int(round(self._protocol.stride_sec * self._source_sfreq))
        start_offset = int(round(self._protocol.control_start_offset_sec * self._source_sfreq))
        stop_offset = int(round(self._protocol.control_stop_offset_sec * self._source_sfreq))
        raw_windows: list[np.ndarray] = []
        processed_windows: list[np.ndarray] = []
        labels: list[int] = []
        trial_groups: list[int] = []
        quality_peak_abs_uv: list[float] = []
        quality_clip_fraction: list[float] = []
        quality_bad_channel_fraction: list[float] = []
        quality_bad_channel_indices: list[str] = []
        rejection_reason_counts: dict[str, int] = {}
        window_start_samples: list[int] = []
        window_stop_samples: list[int] = []
        window_offsets_sec: list[float] = []
        rejected_windows = 0

        for trial_group, trial in enumerate(trials):
            control_on = int(trial["control_on_sample"])
            max_start = control_on + stop_offset - source_window_samples
            for offset in range(start_offset, max_start - control_on + 1, stride_samples):
                start = control_on + offset
                stop = start + source_window_samples
                if stop > eeg.shape[1]:
                    continue
                source_window = eeg[:, start:stop].astype(np.float32)
                window = resample_eeg(
                    source_window,
                    source_sfreq=self._source_sfreq,
                    target_sfreq=self._sfreq,
                )
                if window.shape[1] != target_window_samples:
                    raise RuntimeError(
                        f"Resampled calibration window has {window.shape[1]} points; "
                        f"expected {target_window_samples}."
                    )
                result = preprocess_eeg_window(window, sfreq=self._sfreq)
                if not result.quality.accepted:
                    rejected_windows += 1
                    for reason in result.quality.reasons:
                        rejection_reason_counts[reason] = (
                            rejection_reason_counts.get(reason, 0) + 1
                        )
                    continue
                raw_windows.append(window)
                processed_windows.append(result.data)
                labels.append(int(trial["label_id"]))
                trial_groups.append(trial_group)
                quality_peak_abs_uv.append(result.quality.peak_abs_uv)
                quality_clip_fraction.append(result.quality.clip_fraction)
                quality_bad_channel_fraction.append(result.quality.bad_channel_fraction)
                quality_bad_channel_indices.append(
                    json.dumps(
                        list(getattr(result.quality, "bad_channel_indices", ())),
                        separators=(",", ":"),
                    )
                )
                window_start_samples.append(start)
                window_stop_samples.append(stop)
                window_offsets_sec.append(offset / self._source_sfreq)

        empty_shape = (0, eeg.shape[0], target_window_samples)
        raw_X = np.stack(raw_windows, axis=0).astype(np.float32) if raw_windows else np.empty(empty_shape, dtype=np.float32)
        X = np.stack(processed_windows, axis=0).astype(np.float32) if processed_windows else np.empty(empty_shape, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int64)
        groups = np.asarray(trial_groups, dtype=np.int64)

        if session_dir is not None:
            self._save_training_windows(
                session_dir / "training_windows_main.npz",
                raw_windows=raw_X,
                processed_windows=X,
                labels=y,
                trial_ids=groups,
                window_sec=self._protocol.window_sec,
                stride_sec=self._protocol.stride_sec,
                quality_peak_abs_uv=np.asarray(quality_peak_abs_uv, dtype=np.float32),
                quality_clip_fraction=np.asarray(quality_clip_fraction, dtype=np.float32),
                quality_bad_channel_fraction=np.asarray(
                    quality_bad_channel_fraction,
                    dtype=np.float32,
                ),
                rejected_windows=rejected_windows,
                window_start_samples=np.asarray(window_start_samples, dtype=np.int64),
                window_stop_samples=np.asarray(window_stop_samples, dtype=np.int64),
                window_offsets_sec=np.asarray(window_offsets_sec, dtype=np.float32),
                quality_bad_channel_indices=np.asarray(
                    quality_bad_channel_indices,
                    dtype=np.str_,
                ),
                rejection_reason_counts=rejection_reason_counts,
            )
            if self._protocol.export_window_sec is not None:
                alt_raw, alt_processed, alt_labels, alt_groups, alt_quality = self._build_aux_windows(
                    eeg=eeg,
                    trials=trials,
                    window_sec=float(self._protocol.export_window_sec),
                    stride_sec=float(self._protocol.export_stride_sec),
                )
                self._save_training_windows(
                    session_dir / "training_windows_aux_1p5s.npz",
                    raw_windows=alt_raw,
                    processed_windows=alt_processed,
                    labels=alt_labels,
                    trial_ids=alt_groups,
                    window_sec=float(self._protocol.export_window_sec),
                    stride_sec=float(self._protocol.export_stride_sec),
                    **alt_quality,
                )
        if rejected_windows:
            self._console.print(
                f"[yellow]预处理质量控制剔除 {rejected_windows} 个伪迹窗；"
                f"保留 {X.shape[0]} 个训练窗。[/yellow]"
            )
        return raw_X, X, y, groups

    def _build_aux_windows(
        self,
        *,
        eeg: np.ndarray,
        trials: list[dict[str, Any]],
        window_sec: float,
        stride_sec: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        source_window_samples = int(round(window_sec * self._source_sfreq))
        target_window_samples = int(round(window_sec * self._sfreq))
        stride_samples = int(round(stride_sec * self._source_sfreq))
        start_offset = int(round(self._protocol.control_start_offset_sec * self._source_sfreq))
        stop_offset = int(round(self._protocol.control_stop_offset_sec * self._source_sfreq))
        raw_windows: list[np.ndarray] = []
        processed_windows: list[np.ndarray] = []
        labels: list[int] = []
        trial_groups: list[int] = []
        quality_peak_abs_uv: list[float] = []
        quality_clip_fraction: list[float] = []
        quality_bad_channel_fraction: list[float] = []
        quality_bad_channel_indices: list[str] = []
        rejection_reason_counts: dict[str, int] = {}
        window_start_samples: list[int] = []
        window_stop_samples: list[int] = []
        window_offsets_sec: list[float] = []
        rejected_windows = 0
        for trial_group, trial in enumerate(trials):
            control_on = int(trial["control_on_sample"])
            max_start = control_on + stop_offset - source_window_samples
            for offset in range(start_offset, max_start - control_on + 1, stride_samples):
                start = control_on + offset
                stop = start + source_window_samples
                if stop > eeg.shape[1]:
                    continue
                source_window = eeg[:, start:stop].astype(np.float32)
                window = resample_eeg(
                    source_window,
                    source_sfreq=self._source_sfreq,
                    target_sfreq=self._sfreq,
                )
                if window.shape[1] != target_window_samples:
                    raise RuntimeError(
                        f"Resampled auxiliary window has {window.shape[1]} points; "
                        f"expected {target_window_samples}."
                    )
                result = preprocess_eeg_window(window, sfreq=self._sfreq)
                if not result.quality.accepted:
                    rejected_windows += 1
                    for reason in result.quality.reasons:
                        rejection_reason_counts[reason] = (
                            rejection_reason_counts.get(reason, 0) + 1
                        )
                    continue
                raw_windows.append(window)
                processed_windows.append(result.data)
                labels.append(int(trial["label_id"]))
                trial_groups.append(trial_group)
                quality_peak_abs_uv.append(result.quality.peak_abs_uv)
                quality_clip_fraction.append(result.quality.clip_fraction)
                quality_bad_channel_fraction.append(result.quality.bad_channel_fraction)
                quality_bad_channel_indices.append(
                    json.dumps(
                        list(result.quality.bad_channel_indices),
                        separators=(",", ":"),
                    )
                )
                window_start_samples.append(start)
                window_stop_samples.append(stop)
                window_offsets_sec.append(offset / self._source_sfreq)
        shape = (0, eeg.shape[0], target_window_samples)
        raw_X = np.stack(raw_windows, axis=0).astype(np.float32) if raw_windows else np.empty(shape, dtype=np.float32)
        X = np.stack(processed_windows, axis=0).astype(np.float32) if processed_windows else np.empty(shape, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int64)
        groups = np.asarray(trial_groups, dtype=np.int64)
        quality = {
            "quality_peak_abs_uv": np.asarray(quality_peak_abs_uv, dtype=np.float32),
            "quality_clip_fraction": np.asarray(quality_clip_fraction, dtype=np.float32),
            "quality_bad_channel_fraction": np.asarray(
                quality_bad_channel_fraction,
                dtype=np.float32,
            ),
            "rejected_windows": rejected_windows,
            "window_start_samples": np.asarray(window_start_samples, dtype=np.int64),
            "window_stop_samples": np.asarray(window_stop_samples, dtype=np.int64),
            "window_offsets_sec": np.asarray(window_offsets_sec, dtype=np.float32),
            "quality_bad_channel_indices": np.asarray(
                quality_bad_channel_indices,
                dtype=np.str_,
            ),
            "rejection_reason_counts": rejection_reason_counts,
        }
        return raw_X, X, y, groups, quality

    def _save_training_windows(
        self,
        output_path: Path,
        *,
        raw_windows: np.ndarray,
        processed_windows: np.ndarray,
        labels: np.ndarray,
        trial_ids: np.ndarray,
        window_sec: float,
        stride_sec: float,
        quality_peak_abs_uv: np.ndarray,
        quality_clip_fraction: np.ndarray,
        quality_bad_channel_fraction: np.ndarray,
        rejected_windows: int,
        window_start_samples: np.ndarray,
        window_stop_samples: np.ndarray,
        window_offsets_sec: np.ndarray,
        quality_bad_channel_indices: np.ndarray,
        rejection_reason_counts: dict[str, int],
    ) -> None:
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                raw_windows=raw_windows,
                processed_windows=processed_windows,
                labels=labels,
                trial_ids=trial_ids,
                source_sfreq=np.asarray([self._source_sfreq], dtype=np.float32),
                sfreq=np.asarray([self._sfreq], dtype=np.float32),
                window_sec=np.asarray([window_sec], dtype=np.float32),
                step_sec=np.asarray([stride_sec], dtype=np.float32),
                quality_peak_abs_uv=quality_peak_abs_uv,
                quality_clip_fraction=quality_clip_fraction,
                quality_bad_channel_fraction=quality_bad_channel_fraction,
                quality_bad_channel_indices=quality_bad_channel_indices,
                window_start_samples=window_start_samples,
                window_stop_samples=window_stop_samples,
                window_offsets_sec=window_offsets_sec,
                quality_rejected_windows=np.asarray(
                    [rejected_windows],
                    dtype=np.int64,
                ),
                quality_rejection_reason_counts=np.asarray(
                    [
                        json.dumps(
                            rejection_reason_counts,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    ],
                    dtype=np.str_,
                ),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)

    def _get_continuous_eeg(self, *, session_dir: Path | None, recorder: SessionRecorder) -> np.ndarray:
        if session_dir is not None:
            return np.load(session_dir / "continuous_eeg.npy").astype(np.float32)
        return recorder.to_array()

    def _build_session_metadata(
        self,
        plan: SessionPlan,
        *,
        session_stamp: str,
        trials: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "session_id": session_stamp,
            "protocol_name": "mi_game_control_recalibration_protocol_v2",
            "subject_mode": plan.subject_mode,
            "sfreq": self._sfreq,
            "source_sfreq": self._source_sfreq,
            "n_channels": self._acquirer.metadata.n_channels,
            "timestamp_domain": getattr(
                self._acquirer.metadata,
                "timestamp_domain",
                "relative",
            ),
            "timing_diagnostics": getattr(
                self._acquirer,
                "timing_diagnostics",
                {},
            ),
            "window_sec": self._protocol.window_sec,
            "stride_sec": self._protocol.stride_sec,
            "control_window_range_sec": [
                self._protocol.control_start_offset_sec,
                self._protocol.control_stop_offset_sec,
            ],
            "planned_collection_duration_sec": (
                sum(segment.duration_sec for segment in plan.baseline_segments)
                + plan.total_formal_trials * plan.trial_timing.total_sec
                + max(len(plan.blocks) - 1, 0) * plan.rest_between_blocks_sec
            ),
            "formal_trial_count": plan.total_formal_trials,
            "validation_grouping": "trial_ids",
            "preprocessing": {
                **DEFAULT_PREPROCESSING.as_dict(),
                "reference": "common_average",
                "filter_design": "Butterworth SOS zero-phase",
                "bad_channel_repair": "pointwise_median_of_good_channels",
                "input_unit": "uV",
            },
            "trial_timing": {
                "fixation_sec": plan.trial_timing.fixation_sec,
                "cue_sec": plan.trial_timing.cue_sec,
                "control_sec": plan.trial_timing.control_sec,
                "iti_sec": plan.trial_timing.iti_sec,
            },
            "label_map": LABEL_TO_ID,
            "trials": trials,
            "baseline_segments": [
                {
                    "name": segment.name,
                    "duration_sec": segment.duration_sec,
                    "instruction": segment.instruction,
                }
                for segment in plan.baseline_segments
            ],
            "bad_trials": [],
            "low_quality_blocks": [],
            "provenance": self._build_provenance(),
        }

    def _write_session_summary(
        self,
        session_dir: Path | None,
        *,
        metrics: dict[str, float],
        windows_collected: int,
        session_metadata: dict[str, Any],
    ) -> None:
        if session_dir is None:
            return
        summary = dict(session_metadata)
        summary["model_path"] = str(self._model_path)
        summary["windows_collected"] = windows_collected
        summary["metrics"] = metrics
        metadata_path = session_dir / "metadata.json"
        temporary = session_dir / ".metadata.json.tmp"
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, metadata_path)

    def _seal_session_bundle(self, session_dir: Path | None) -> None:
        if session_dir is None:
            return
        metadata_path = session_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model_files: list[dict[str, Any]] = []
        for path in (
            self._model_path,
            Path(f"{self._model_path}.neuroonline.pt"),
        ):
            if path.exists():
                model_files.append(
                    {
                        "path": str(path),
                        "sha256": self._sha256(path),
                        "size_bytes": path.stat().st_size,
                    }
                )
        checksums: list[dict[str, Any]] = []
        for path in sorted(session_dir.iterdir()):
            if not path.is_file() or path == metadata_path:
                continue
            checksums.append(
                {
                    "path": path.name,
                    "sha256": self._sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
        eeg_path = session_dir / "continuous_eeg.npy"
        timestamps_path = session_dir / "continuous_sample_timestamps.npy"
        invalid_timestamps = 0
        timestamp_count_matches = False
        if eeg_path.exists() and timestamps_path.exists():
            eeg = np.load(eeg_path, mmap_mode="r")
            timestamps = np.load(timestamps_path, mmap_mode="r")
            timestamp_count_matches = bool(timestamps.size == eeg.shape[-1])
            invalid_timestamps = int(np.sum(~np.isfinite(timestamps)))
        packet_loss_count = int(
            float(
                (metadata.get("timing_diagnostics", {}) or {}).get(
                    "packet_loss_count",
                    0,
                )
            )
        )
        integrity_status = "complete"
        if packet_loss_count > 0:
            integrity_status = "source_packet_loss"
        elif not timestamp_count_matches or invalid_timestamps > 0:
            integrity_status = "invalid_sample_timestamps"
        metadata["model_files"] = model_files
        metadata["integrity"] = {
            "status": integrity_status,
            "packet_loss_count": packet_loss_count,
            "sample_timestamp_count_matches": timestamp_count_matches,
            "invalid_sample_timestamps": invalid_timestamps,
            "checksums": checksums,
        }
        temporary = session_dir / ".metadata.json.tmp"
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, metadata_path)

    def _build_provenance(self) -> dict[str, Any]:
        project_root = Path(__file__).resolve().parents[1]
        commit: str | None = None
        dirty: bool | None = None
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            dirty = bool(
                subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=project_root,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout.strip()
            )
        except (OSError, subprocess.SubprocessError):
            pass
        packages: dict[str, str | None] = {}
        for package in ("numpy", "scipy", "torch", "scikit-learn", "mne"):
            try:
                packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                packages[package] = None
        encoded_config = json.dumps(
            self._experiment_config,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
        return {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git": {"commit": commit, "dirty": dirty},
            "platform": platform.platform(),
            "python": sys.version,
            "packages": packages,
            "experiment_config": self._experiment_config,
            "experiment_config_sha256": hashlib.sha256(encoded_config).hexdigest(),
            "random_seed": int(self._neuroonline_config.random_seed),
            "deterministic_algorithms_requested": True,
        }

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _print_instructions(self, plan: SessionPlan) -> None:
        self._console.print("[bold cyan]实验指导语[/bold cyan]")
        for line in RECOMMENDED_INSTRUCTIONS:
            self._console.print(f"- {line}")
        self._console.print(
            f"[bold cyan]正式 trial[/bold cyan] fixation={plan.trial_timing.fixation_sec:.1f}s "
            f"cue={plan.trial_timing.cue_sec:.1f}s control={plan.trial_timing.control_sec:.1f}s "
            f"iti={plan.trial_timing.iti_sec:.1f}s"
        )
        self._console.print(
            f"[bold cyan]训练切窗[/bold cyan] window={self._protocol.window_sec:.1f}s stride={self._protocol.stride_sec:.1f}s "
            f"from control [{self._protocol.control_start_offset_sec:.1f}, {self._protocol.control_stop_offset_sec:.1f}]s"
        )

    def _emit_event(self, recorder: SessionRecorder, event_name: str, **payload: Any) -> Any:
        event_time = time.monotonic()
        self._marker_backend.send_event(event_name, timestamp=event_time)
        return recorder.add_event(
            event_name,
            timestamp_monotonic=event_time,
            marker_code=PROTOCOL_EVENT_CODES[event_name],
            **payload,
        )

    def _sleep_with_recording(
        self,
        duration_sec: float,
        *,
        recorder: SessionRecorder,
        heartbeat: Callable[[], None] | None,
        stage_name: str = "",
    ) -> None:
        total = max(float(duration_sec), 0.0)
        started_at = time.monotonic()
        deadline = started_at + total
        self._update_stage_progress(stage_name=stage_name, elapsed_sec=0.0, duration_sec=total)
        while time.monotonic() < deadline:
            self._flush_recorder(recorder)
            if heartbeat is not None:
                heartbeat()
            elapsed = min(time.monotonic() - started_at, total)
            self._update_stage_progress(stage_name=stage_name, elapsed_sec=elapsed, duration_sec=total)
            time.sleep(min(0.05, max(deadline - time.monotonic(), 0.0)))
        self._flush_recorder(recorder)
        self._update_stage_progress(stage_name=stage_name, elapsed_sec=total, duration_sec=total)
        if heartbeat is not None:
            heartbeat()

    def _update_stage_progress(self, *, stage_name: str, elapsed_sec: float, duration_sec: float) -> None:
        progress = getattr(self._console, "set_stage_progress", None)
        if callable(progress):
            progress(stage_name=stage_name, elapsed_sec=elapsed_sec, duration_sec=duration_sec)

    def _flush_recorder(self, recorder: SessionRecorder) -> None:
        try:
            samples = recorder.pull()
        except RuntimeError as exc:
            message = str(exc).lower()
            if "stream" in message and "not started" in message:
                raise RuntimeError(
                    "Calibration interrupted: EEG stream stopped unexpectedly during collection. "
                    "Please check BrainCo device power/network stability and rerun."
                ) from exc
            raise
        if samples.size == 0:
            return

    def _save_metadata(
        self,
        *,
        metrics: dict[str, float],
        windows_collected: int,
        head_only: bool,
    ) -> None:
        metadata = {
            "model_path": str(self._model_path),
            "windows_collected": windows_collected,
            "head_only": head_only,
            "sfreq": self._sfreq,
            "window_sec": self._protocol.window_sec,
            "step_sec": self._protocol.stride_sec,
            "metrics": metrics,
        }
        metadata_path = self._model_path.with_suffix(".metrics.yaml")
        with metadata_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(metadata, handle, sort_keys=False, allow_unicode=True)
