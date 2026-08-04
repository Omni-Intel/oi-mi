"""Protocol-driven motor imagery calibration."""

from __future__ import annotations

from collections.abc import Callable
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from rich.console import Console

from acquisition.base import AbstractAcquirer
from adaptation.calibration_search import (
    CalibrationSearchConfig,
    load_latest_calibration_search,
    run_calibration_search,
)
from adaptation.mi_protocol import (
    LABEL_DESCRIPTION,
    LABEL_DISPLAY,
    LABEL_SYMBOL,
    LABEL_TO_ID,
    RECOMMENDED_INSTRUCTIONS,
    ProtocolConfig,
    SessionPlan,
    build_session_plan,
    generate_block_sequence,
)
from adaptation.neuroonline import NeuroOnlineConfig, NeuroOnlineModelAdapter
from adaptation.session_recorder import SessionRecorder
from models.factory import BaseModelAdapter, TorchModelAdapter
from utils.markers import MarkerBackend, PROTOCOL_EVENT_CODES
from utils.preprocessing import (
    ContinuousPreprocessingResult,
    continuous_preprocessing_metadata,
    finalize_preprocessed_window,
    preprocess_eeg_continuous,
)

LABEL_SEQUENCE: list[tuple[int, str]] = [(LABEL_TO_ID[label], label) for label in ("left", "right", "idle")]


def _offline_parameter_snapshot(config: NeuroOnlineConfig) -> dict[str, Any]:
    """Return the effective offline settings for session provenance/UI output."""

    return {
        "offline_learning_rate": config.offline_learning_rate,
        "offline_batch_size": config.offline_batch_size,
        "mask_ratio": (
            config.mask_ratio
            if config.offline_mask_ratio is None
            else config.offline_mask_ratio
        ),
        "consistency_weight": (
            config.consistency_weight
            if config.offline_consistency_weight is None
            else config.offline_consistency_weight
        ),
        "weight_decay": config.weight_decay,
        "label_smoothing": config.label_smoothing,
        "offline_epochs": config.offline_epochs,
    }


@dataclass(slots=True)
class CalibrationResult:
    """Result metadata for a calibration run."""

    model_path: Path | None
    metrics: dict[str, float]
    windows_collected: int
    calibration_data_path: Path | None = None
    session_dir: Path | None = None
    hyperparameter_search_path: Path | None = None
    selected_hyperparameters: dict[str, Any] | None = None
    training_performed: bool = True


class CalibrationRunControl:
    """Thread-safe operator controls for an open-ended calibration run."""

    def __init__(self, *, minimum_trials: int) -> None:
        self.minimum_trials = max(int(minimum_trials), 3)
        self._lock = threading.RLock()
        self._pause_requested = False
        self._stop_requested = False
        self._paused = False
        self._collection_finished = False
        self._failed = False
        self._completed_trials = 0
        self._class_counts = {label: 0 for label in LABEL_TO_ID}

    def request_pause(self) -> bool:
        with self._lock:
            if self._stop_requested or self._collection_finished or self._failed:
                return False
            self._pause_requested = True
            return True

    def request_resume(self) -> bool:
        with self._lock:
            if self._collection_finished or self._failed:
                return False
            changed = self._pause_requested or self._paused
            self._pause_requested = False
            return changed

    def request_stop(self) -> bool:
        with self._lock:
            if (
                self._completed_trials < self.minimum_trials
                or self._collection_finished
                or self._failed
            ):
                return False
            self._stop_requested = True
            self._pause_requested = False
            return True

    def mark_paused(self, value: bool) -> None:
        with self._lock:
            self._paused = bool(value)

    def mark_trial_completed(self, label: str) -> None:
        with self._lock:
            if label not in self._class_counts:
                raise ValueError(f"Unknown calibration label: {label}")
            self._completed_trials += 1
            self._class_counts[label] += 1

    def should_pause(self) -> bool:
        with self._lock:
            return self._pause_requested and not self._stop_requested

    def should_finish(self) -> bool:
        with self._lock:
            counts = tuple(self._class_counts.values())
            balanced = bool(counts) and len(set(counts)) == 1
            return bool(
                self._stop_requested
                and self._completed_trials >= self.minimum_trials
                and balanced
            )

    def mark_collection_finished(self) -> None:
        with self._lock:
            self._collection_finished = True
            self._paused = False
            self._pause_requested = False

    def mark_failed(self) -> None:
        with self._lock:
            self._failed = True
            self._paused = False
            self._pause_requested = False

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counts = dict(self._class_counts)
            balanced = len(set(counts.values())) == 1
            if self._failed:
                state = "failed"
            elif self._collection_finished:
                state = "training"
            elif self._stop_requested:
                state = "stop_pending"
            elif self._paused:
                state = "paused"
            elif self._pause_requested:
                state = "pause_pending"
            else:
                state = "collecting"
            return {
                "state": state,
                "completed_trials": self._completed_trials,
                "minimum_trials": self.minimum_trials,
                "class_counts": counts,
                "balanced": balanced,
                "can_request_stop": self._completed_trials >= self.minimum_trials,
                "stop_requested": self._stop_requested,
                "pause_requested": self._pause_requested,
            }


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
        self._calibration_search_config = CalibrationSearchConfig.from_mapping(
            online_adaptation_config
        )
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
        head_only: bool,
        include_practice: bool = True,
        heartbeat: Callable[[], None] | None = None,
        run_control: CalibrationRunControl | None = None,
        train_after_collection: bool = True,
    ) -> CalibrationResult:
        del duration_sec
        if head_only:
            raise ValueError(
                "Head-only calibration was removed; each experiment must train "
                "a fresh full decoder."
            )
        if not train_after_collection:
            plan = build_session_plan(self._protocol)
            (
                session_dir,
                _raw_windows,
                processed_windows,
                _labels,
                _trial_groups,
                session_metadata,
            ) = self._collect_training_data(
                plan=plan,
                include_practice=include_practice,
                heartbeat=heartbeat,
                run_control=run_control,
            )
            session_metadata["training"] = {
                "performed": False,
                "reason": "collection_only",
            }
            windows_collected = int(processed_windows.shape[0])
            self._console.print(
                "[bold green]采集完成，数据已保存；本次未执行模型训练[/bold green]"
            )
            self._write_session_summary(
                session_dir,
                metrics={},
                windows_collected=windows_collected,
                session_metadata=session_metadata,
                training_performed=False,
            )
            self._seal_session_bundle(session_dir, include_model_files=False)
            if heartbeat is not None:
                heartbeat()
            return CalibrationResult(
                model_path=None,
                metrics={},
                windows_collected=windows_collected,
                calibration_data_path=(
                    session_dir / "training_windows_main.npz"
                    if session_dir is not None
                    else None
                ),
                session_dir=session_dir,
                training_performed=False,
            )
        reused_search_report: dict[str, Any] | None = None
        reuse_search_error: str | None = None
        selected_hyperparameters: dict[str, Any] | None = (
            _offline_parameter_snapshot(self._neuroonline_config)
            if self._neuroonline_config.enabled
            else None
        )
        hyperparameter_report_path: Path | None = None
        if (
            self._neuroonline_config.enabled
            and self._calibration_search_config.reuse_latest
        ):
            if not isinstance(self._model, NeuroOnlineModelAdapter):
                raise RuntimeError(
                    "Reusing NeuroOnline hyperparameters requires a NeuroOnline "
                    "model adapter."
                )
            base_template = self._model.base
            try:
                (
                    self._neuroonline_config,
                    reused_search_report,
                    hyperparameter_report_path,
                ) = load_latest_calibration_search(
                    calibration_records_dir=self._calibration_records_dir,
                    base_config=self._neuroonline_config,
                    model_name=base_template.model_name,
                )
            except RuntimeError as exc:
                reuse_search_error = str(exc)
                selected_hyperparameters = _offline_parameter_snapshot(
                    self._neuroonline_config
                )
                self._console.print(
                    "[bold yellow]历史离线参数报告不可用；本次校准将使用 "
                    "config.yaml 中冻结的离线参数继续，不会中止："
                    f"{reuse_search_error}[/bold yellow]"
                )
            else:
                selected_hyperparameters = dict(
                    reused_search_report["best_parameters"]
                )
                self._model = NeuroOnlineModelAdapter(
                    copy.deepcopy(base_template),
                    config=self._neuroonline_config,
                    state_path=None,
                )
                self._console.print(
                    "[bold green]已读取最近一次离线参数搜索结果，将直接训练新模型："
                    f"{hyperparameter_report_path}[/bold green]"
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
            run_control=run_control,
        )
        if reused_search_report is not None:
            session_metadata["hyperparameter_search"] = {
                "mode": "reused_latest",
                "source_report_path": str(hyperparameter_report_path),
                "best_parameters": selected_hyperparameters,
                "source_untouched_holdout_metrics": reused_search_report.get(
                    "untouched_holdout_metrics"
                ),
            }
        elif reuse_search_error is not None:
            session_metadata["hyperparameter_search"] = {
                "mode": "configured_fallback",
                "reason": reuse_search_error,
                "best_parameters": selected_hyperparameters,
            }
        elif (
            self._neuroonline_config.enabled
            and not self._calibration_search_config.enabled
        ):
            session_metadata["hyperparameter_search"] = {
                "mode": "configured_fixed",
                "best_parameters": selected_hyperparameters,
            }
        self._console.print("[bold cyan]采集完成，正在保存和训练，请等待工作人员[/bold cyan]")
        search_result = None
        if self._neuroonline_config.enabled and self._calibration_search_config.enabled:
            if not isinstance(self._model, NeuroOnlineModelAdapter):
                raise RuntimeError("NeuroOnline search requires a NeuroOnline model adapter.")
            self._console.print(
                "[bold yellow]开始按 trial 分组搜索离线预训练参数。"
                "搜索集与最终检验集严格按 trial 隔离；采集无需重做。[/bold yellow]"
            )
            base_template = self._model.base

            def report_search_progress(
                candidate_index: int,
                total_candidates: int,
                stage: str,
                candidate_metrics: dict[str, float],
            ) -> None:
                del candidate_metrics
                if callable(training_progress):
                    training_progress(
                        stage_name=(
                            f"参数搜索 {candidate_index}/{total_candidates}: {stage}"
                        ),
                        elapsed_sec=float(candidate_index),
                        duration_sec=float(total_candidates),
                    )
                if heartbeat is not None:
                    heartbeat()

            training_progress = getattr(self._console, "set_stage_progress", None)
            search_result = run_calibration_search(
                base_template=base_template,
                base_config=self._neuroonline_config,
                search_config=self._calibration_search_config,
                X=processed_windows,
                y=labels,
                groups=trial_groups,
                session_dir=session_dir,
                progress_callback=report_search_progress,
            )
            self._neuroonline_config = search_result.best_config
            self._model = NeuroOnlineModelAdapter(
                copy.deepcopy(base_template),
                config=self._neuroonline_config,
                state_path=None,
            )
            session_metadata["hyperparameter_search"] = {
                "mode": "searched_current_session",
                "report_path": (
                    str(search_result.report_path)
                    if search_result.report_path is not None
                    else None
                ),
                "best_parameters": search_result.report["best_parameters"],
                "untouched_holdout_metrics": search_result.report[
                    "untouched_holdout_metrics"
                ],
            }
            hyperparameter_report_path = search_result.report_path
            selected_hyperparameters = dict(
                search_result.report["best_parameters"]
            )
            best = search_result.report["best_parameters"]
            holdout = search_result.report["untouched_holdout_metrics"]
            self._console.print(
                "[bold green]参数搜索完成："
                f"lr={best['offline_learning_rate']:.1e}, "
                f"batch={best['offline_batch_size']}, "
                f"mask={best['mask_ratio']:.2f}, "
                f"lambda={best['consistency_weight']:.2f}; "
                f"独立 trial 检验 Bal.Acc.={holdout['balanced_accuracy']:.3f}。"
                "[/bold green]"
            )
        if self._neuroonline_config.enabled:
            self._console.print(
                "[bold yellow]正在执行 NeuroOnline 离线训练 "
                f"(固定 {self._neuroonline_config.offline_epochs} epochs，"
                "结束后恢复验证表现最佳的 epoch)。"
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
            head_only=False,
            groups=trial_groups,
            progress_callback=report_training_progress,
        )
        if search_result is not None:
            holdout = search_result.report["untouched_holdout_metrics"]
            metrics.update(
                {
                    "search_holdout_balanced_accuracy": float(
                        holdout["balanced_accuracy"]
                    ),
                    "search_holdout_kappa": float(holdout["kappa"]),
                    "search_holdout_macro_f1": float(holdout["macro_f1"]),
                    "search_holdout_worst_class_accuracy": float(
                        holdout["worst_class_accuracy"]
                    ),
                }
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
            hyperparameter_search_path=(
                hyperparameter_report_path
            ),
            selected_hyperparameters=selected_hyperparameters,
        )

    def _collect_training_data(
        self,
        *,
        plan: SessionPlan,
        include_practice: bool = True,
        heartbeat: Callable[[], None] | None = None,
        run_control: CalibrationRunControl | None = None,
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
            self._run_formal_blocks(
                plan,
                recorder=recorder,
                heartbeat=heartbeat,
                trials=trials,
                run_control=run_control,
            )
            self._emit_event(recorder, "session_end", phase="session")
        finally:
            try:
                self._flush_recorder(recorder)
            finally:
                try:
                    self._acquirer.stop_stream()
                finally:
                    if run_control is not None:
                        run_control.mark_collection_finished()
                    if heartbeat is not None:
                        heartbeat()

        session_metadata = self._build_session_metadata(
            plan,
            session_stamp=session_stamp,
            trials=trials,
            run_control=run_control,
        )
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
        run_control: CalibrationRunControl | None = None,
    ) -> None:
        if run_control is not None and self._protocol.continuous_collection:
            self._run_continuous_formal_blocks(
                plan,
                recorder=recorder,
                heartbeat=heartbeat,
                trials=trials,
                run_control=run_control,
            )
            return
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

    def _run_continuous_formal_blocks(
        self,
        plan: SessionPlan,
        *,
        recorder: SessionRecorder,
        heartbeat: Callable[[], None] | None,
        trials: list[dict[str, Any]],
        run_control: CalibrationRunControl,
    ) -> None:
        """Append balanced blocks until the operator ends at a balanced boundary."""

        rng = random.Random(self._protocol.random_seed)
        counts = {
            label: self._protocol.calibration_trials_per_class_per_block
            for label in LABEL_TO_ID
        }
        block_index = 0
        while True:
            if not self._wait_for_operator(
                recorder=recorder,
                heartbeat=heartbeat,
                run_control=run_control,
            ):
                return
            sequence = generate_block_sequence(counts, rng=rng)
            self._console.print(
                f"[bold cyan]Block {block_index + 1}[/bold cyan] "
                f"共 {len(sequence)} 个 trial；完成后可继续追加"
            )
            self._emit_event(
                recorder,
                "block_start",
                phase="formal",
                block_index=block_index,
                open_ended=True,
            )
            block_completed = True
            for trial_index, label in enumerate(sequence):
                if not self._wait_for_operator(
                    recorder=recorder,
                    heartbeat=heartbeat,
                    run_control=run_control,
                ):
                    block_completed = False
                    break
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
                    run_control.mark_trial_completed(label)
            self._emit_event(
                recorder,
                "block_end",
                phase="formal",
                block_index=block_index,
                completed=block_completed,
            )
            if not block_completed or run_control.should_finish():
                return
            block_index += 1
            if plan.rest_between_blocks_sec > 0:
                self._console.print(
                    f"[bold yellow]休息 {plan.rest_between_blocks_sec:.0f} 秒，"
                    "请放松但不要大幅动作[/bold yellow]"
                )
                self._sleep_with_recording(
                    plan.rest_between_blocks_sec,
                    recorder=recorder,
                    heartbeat=heartbeat,
                    stage_name=f"Block {block_index} 休息",
                )

    def _wait_for_operator(
        self,
        *,
        recorder: SessionRecorder,
        heartbeat: Callable[[], None] | None,
        run_control: CalibrationRunControl,
    ) -> bool:
        """Honor pause and completion requests only between complete trials."""

        if run_control.should_finish():
            return False
        if not run_control.should_pause():
            return True
        snapshot = run_control.snapshot()
        run_control.mark_paused(True)
        self._console.print("[bold yellow]PAUSED 请休息，准备好后由工作人员继续[/bold yellow]")
        self._emit_event(
            recorder,
            "operator_pause_start",
            phase="operator_pause",
            completed_trials=snapshot["completed_trials"],
        )
        while run_control.should_pause():
            self._flush_recorder(recorder)
            self._update_stage_progress(
                stage_name="人工暂停",
                elapsed_sec=0.0,
                duration_sec=0.0,
            )
            if heartbeat is not None:
                heartbeat()
            time.sleep(0.05)
        run_control.mark_paused(False)
        self._emit_event(
            recorder,
            "operator_pause_end",
            phase="operator_pause",
            completed_trials=run_control.snapshot()["completed_trials"],
        )
        if run_control.should_finish():
            return False
        self._console.print("[bold cyan]RESUMED 校准继续[/bold cyan]")
        return True

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
        continuous = preprocess_eeg_continuous(
            eeg,
            source_sfreq=self._source_sfreq,
            target_sfreq=self._sfreq,
        )

        for trial_group, trial in enumerate(trials):
            control_on = int(trial["control_on_sample"])
            max_start = control_on + stop_offset - source_window_samples
            for offset in range(start_offset, max_start - control_on + 1, stride_samples):
                start = control_on + offset
                stop = start + source_window_samples
                if stop > eeg.shape[1]:
                    continue
                target_start = int(round(start * self._sfreq / self._source_sfreq))
                target_stop = target_start + target_window_samples
                window = continuous.raw_data[:, target_start:target_stop]
                if window.shape[1] != target_window_samples:
                    raise RuntimeError(
                        f"Continuous calibration window has {window.shape[1]} points; "
                        f"expected {target_window_samples}."
                    )
                filtered_window = continuous.data[:, target_start:target_stop]
                nonfinite_fraction = float(
                    np.mean(continuous.source_nonfinite_mask[:, start:stop])
                )
                result = finalize_preprocessed_window(
                    filtered_window,
                    bad_channel_indices=continuous.bad_channel_indices,
                    nonfinite_fraction=nonfinite_fraction,
                )
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
                    continuous=continuous,
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
        continuous: ContinuousPreprocessingResult,
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
                target_start = int(round(start * self._sfreq / self._source_sfreq))
                target_stop = target_start + target_window_samples
                window = continuous.raw_data[:, target_start:target_stop]
                if window.shape[1] != target_window_samples:
                    raise RuntimeError(
                        f"Continuous auxiliary window has {window.shape[1]} points; "
                        f"expected {target_window_samples}."
                    )
                filtered_window = continuous.data[:, target_start:target_stop]
                nonfinite_fraction = float(
                    np.mean(continuous.source_nonfinite_mask[:, start:stop])
                )
                result = finalize_preprocessed_window(
                    filtered_window,
                    bad_channel_indices=continuous.bad_channel_indices,
                    nonfinite_fraction=nonfinite_fraction,
                )
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
        run_control: CalibrationRunControl | None = None,
    ) -> dict[str, Any]:
        open_ended = bool(
            run_control is not None and self._protocol.continuous_collection
        )
        return {
            "session_id": session_stamp,
            "protocol_name": "mi_game_control_recalibration_protocol_v2",
            "subject_mode": plan.subject_mode,
            "sfreq": self._sfreq,
            "source_sfreq": self._source_sfreq,
            "n_channels": self._acquirer.metadata.n_channels,
            "channel_names": list(
                getattr(self._acquirer.metadata, "channel_names", ())
            ),
            "channel_types": list(
                getattr(self._acquirer.metadata, "channel_types", ())
            ),
            "channel_selection": getattr(
                self._acquirer,
                "channel_diagnostics",
                {},
            ),
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
                None
                if open_ended
                else (
                    sum(segment.duration_sec for segment in plan.baseline_segments)
                    + plan.total_formal_trials * plan.trial_timing.total_sec
                    + max(len(plan.blocks) - 1, 0) * plan.rest_between_blocks_sec
                )
            ),
            "formal_trial_count": len(trials),
            "collection_mode": (
                "operator_terminated_continuous_blocks"
                if open_ended
                else "fixed_session_plan"
            ),
            "minimum_calibration_trials": (
                run_control.minimum_trials
                if run_control is not None
                else plan.total_formal_trials
            ),
            "operator_control_final": (
                run_control.snapshot() if run_control is not None else None
            ),
            "validation_grouping": "trial_ids",
            "preprocessing": {
                **continuous_preprocessing_metadata(),
                "continuous_span": "complete_calibration_session",
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
        training_performed: bool = True,
    ) -> None:
        if session_dir is None:
            return
        summary = dict(session_metadata)
        summary["training_performed"] = bool(training_performed)
        summary["model_path"] = str(self._model_path) if training_performed else None
        summary["windows_collected"] = windows_collected
        summary["metrics"] = metrics
        metadata_path = session_dir / "metadata.json"
        temporary = session_dir / ".metadata.json.tmp"
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, metadata_path)

    def _seal_session_bundle(
        self,
        session_dir: Path | None,
        *,
        include_model_files: bool = True,
    ) -> None:
        if session_dir is None:
            return
        metadata_path = session_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model_files: list[dict[str, Any]] = []
        candidate_model_paths = (
            self._model_path,
            Path(f"{self._model_path}.neuroonline.pt"),
        ) if include_model_files else ()
        for path in candidate_model_paths:
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
