"""Protocol-driven motor imagery calibration."""

from __future__ import annotations

from collections.abc import Callable
import json
import time
from dataclasses import dataclass
from datetime import datetime
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
from utils.preprocessing import filter_and_transform

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
    ) -> None:
        self._acquirer = acquirer
        neuroonline_config = NeuroOnlineConfig.from_mapping(online_adaptation_config)
        if neuroonline_config.enabled:
            if not isinstance(model, TorchModelAdapter):
                raise ValueError("NeuroOnline calibration requires a PyTorch decoder model.")
            self._model = NeuroOnlineModelAdapter(
                model,
                config=neuroonline_config,
                state_path=None,
            )
        else:
            self._model = model
        self._marker_backend = marker_backend
        self._console = console
        self._sfreq = float(sfreq)
        self._window_sec = float(window_sec)
        self._step_sec = float(step_sec)
        self._model_path = model_path
        self._calibration_records_dir = calibration_records_dir
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
        plan = build_session_plan(self._protocol, is_new_subject=not head_only)
        session_dir, raw_windows, processed_windows, labels, session_metadata = self._collect_training_data(
            plan=plan,
            include_practice=include_practice,
            heartbeat=heartbeat,
        )
        self._console.print("[bold cyan]采集完成，正在保存和训练，请等待工作人员[/bold cyan]")
        if heartbeat is not None:
            heartbeat()
        metrics = self._model.fit(
            processed_windows,
            labels,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=patience,
            head_only=head_only,
        )
        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(self._model_path)
        self._save_metadata(metrics=metrics, windows_collected=int(processed_windows.shape[0]), head_only=head_only)
        self._write_session_summary(session_dir, metrics=metrics, windows_collected=int(processed_windows.shape[0]), session_metadata=session_metadata)
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

    def load_existing_weights(self) -> None:
        if not self._model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {self._model_path}")
        self._model.load(self._model_path)

    def _collect_training_data(
        self,
        *,
        plan: SessionPlan,
        include_practice: bool = True,
        heartbeat: Callable[[], None] | None = None,
    ) -> tuple[Path | None, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        self._console.print("[bold cyan]开始按 MI game control protocol 采集[/bold cyan]")
        self._print_instructions(plan)
        self._acquirer.start_stream()
        recorder = SessionRecorder(
            self._acquirer,
            sfreq=self._sfreq,
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
        raw_windows, processed_windows, labels = self._build_training_windows(
            eeg=eeg,
            events=recorder.events,
            trials=trials,
            session_dir=session_dir,
        )
        if raw_windows.shape[0] == 0:
            raise RuntimeError("Calibration did not yield any valid training windows.")
        return session_dir, raw_windows, processed_windows, labels, session_metadata

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

        control_on_sample = recorder.sample_count
        self._emit_event(
            recorder,
            "control_on",
            phase=phase,
            block_index=block_index,
            trial_index=trial_index,
            label=label,
        )
        self._sleep_with_recording(
            trial_timing.control_sec,
            recorder=recorder,
            heartbeat=heartbeat,
            stage_name=f"{stage_prefix}: control {label}",
        )
        control_off_sample = recorder.sample_count
        self._emit_event(
            recorder,
            "control_off",
            phase=phase,
            block_index=block_index,
            trial_index=trial_index,
            label=label,
        )

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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        del events
        window_samples = int(round(self._protocol.window_sec * self._sfreq))
        stride_samples = int(round(self._protocol.stride_sec * self._sfreq))
        start_offset = int(round(self._protocol.control_start_offset_sec * self._sfreq))
        stop_offset = int(round(self._protocol.control_stop_offset_sec * self._sfreq))
        raw_windows: list[np.ndarray] = []
        processed_windows: list[np.ndarray] = []
        labels: list[int] = []

        for trial in trials:
            control_on = int(trial["control_on_sample"])
            max_start = control_on + stop_offset - window_samples
            for offset in range(start_offset, max_start - control_on + 1, stride_samples):
                start = control_on + offset
                stop = start + window_samples
                if stop > eeg.shape[1]:
                    continue
                window = eeg[:, start:stop].astype(np.float32)
                raw_windows.append(window)
                processed_windows.append(filter_and_transform(window, sfreq=self._sfreq))
                labels.append(int(trial["label_id"]))

        raw_X = np.stack(raw_windows, axis=0).astype(np.float32) if raw_windows else np.empty((0, eeg.shape[0], window_samples), dtype=np.float32)
        X = np.stack(processed_windows, axis=0).astype(np.float32) if processed_windows else np.empty((0, eeg.shape[0], window_samples), dtype=np.float32)
        y = np.asarray(labels, dtype=np.int64)

        if session_dir is not None:
            self._save_training_windows(
                session_dir / "training_windows_main.npz",
                raw_windows=raw_X,
                processed_windows=X,
                labels=y,
                window_sec=self._protocol.window_sec,
                stride_sec=self._protocol.stride_sec,
            )
            if self._protocol.export_window_sec is not None:
                alt_raw, alt_processed, alt_labels = self._build_aux_windows(
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
                    window_sec=float(self._protocol.export_window_sec),
                    stride_sec=float(self._protocol.export_stride_sec),
                )
        return raw_X, X, y

    def _build_aux_windows(
        self,
        *,
        eeg: np.ndarray,
        trials: list[dict[str, Any]],
        window_sec: float,
        stride_sec: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        window_samples = int(round(window_sec * self._sfreq))
        stride_samples = int(round(stride_sec * self._sfreq))
        start_offset = int(round(self._protocol.control_start_offset_sec * self._sfreq))
        stop_offset = int(round(self._protocol.control_stop_offset_sec * self._sfreq))
        raw_windows: list[np.ndarray] = []
        processed_windows: list[np.ndarray] = []
        labels: list[int] = []
        for trial in trials:
            control_on = int(trial["control_on_sample"])
            max_start = control_on + stop_offset - window_samples
            for offset in range(start_offset, max_start - control_on + 1, stride_samples):
                start = control_on + offset
                stop = start + window_samples
                if stop > eeg.shape[1]:
                    continue
                window = eeg[:, start:stop].astype(np.float32)
                raw_windows.append(window)
                processed_windows.append(filter_and_transform(window, sfreq=self._sfreq))
                labels.append(int(trial["label_id"]))
        shape = (0, eeg.shape[0], window_samples)
        raw_X = np.stack(raw_windows, axis=0).astype(np.float32) if raw_windows else np.empty(shape, dtype=np.float32)
        X = np.stack(processed_windows, axis=0).astype(np.float32) if processed_windows else np.empty(shape, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int64)
        return raw_X, X, y

    def _save_training_windows(
        self,
        output_path: Path,
        *,
        raw_windows: np.ndarray,
        processed_windows: np.ndarray,
        labels: np.ndarray,
        window_sec: float,
        stride_sec: float,
    ) -> None:
        np.savez_compressed(
            output_path,
            raw_windows=raw_windows,
            processed_windows=processed_windows,
            labels=labels,
            sfreq=np.asarray([self._sfreq], dtype=np.float32),
            window_sec=np.asarray([window_sec], dtype=np.float32),
            step_sec=np.asarray([stride_sec], dtype=np.float32),
        )

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
            "protocol_name": "mi_game_control_data_collection_protocol_v1",
            "subject_mode": plan.subject_mode,
            "sfreq": self._sfreq,
            "n_channels": self._acquirer.metadata.n_channels,
            "window_sec": self._protocol.window_sec,
            "stride_sec": self._protocol.stride_sec,
            "control_window_range_sec": [
                self._protocol.control_start_offset_sec,
                self._protocol.control_stop_offset_sec,
            ],
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
        with (session_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)

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

    def _emit_event(self, recorder: SessionRecorder, event_name: str, **payload: Any) -> None:
        self._marker_backend.send_event(event_name)
        recorder.add_event(event_name, marker_code=PROTOCOL_EVENT_CODES[event_name], **payload)

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
