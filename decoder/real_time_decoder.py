"""Background realtime motor imagery decoding loop."""

from __future__ import annotations

from collections.abc import Callable
import copy
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
import logging
import platform
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import numpy as np
from rich.console import Console

from acquisition.base import AbstractAcquirer
from adaptation.neuroonline import (
    NeuroOnlineConfig,
    NeuroOnlineModelAdapter,
    NeuroOnlineStreamAdapter,
)
from adaptation.online_batch_adapter import BatchAdaptationConfig, OnlineBatchAdapter
from models.factory import BaseModelAdapter, TorchModelAdapter
from utils.markers import LSLCommandOutlet, MarkerBackend
from utils.online_labels import CUED_PROTOCOL_VERSION, OnlineLabelSource
from utils.preprocessing import DEFAULT_PREPROCESSING, preprocess_eeg_window
from utils.stream_writer import StreamWriter

LOGGER = logging.getLogger(__name__)

LABEL_NAMES = {0: "左手", 1: "右手", 2: "静息"}
TEST_MODE_PROMPTS = {0: "想象左手", 1: "想象右手", 2: "保持静息"}


@dataclass(slots=True)
class PredictionResult:
    """One realtime decoding output."""

    label: str
    confidence: float
    uncertainty: float
    class_id: int | None


@dataclass(slots=True)
class _PendingCuedWindow:
    """One labeled window waiting for the future transition guard to close."""

    processed: np.ndarray
    probabilities: np.ndarray
    operational_prediction: int | None
    prediction_model_revision: int
    online_label: Any
    window_start: float
    window_end: float
    quality_accepted: bool
    record_payload: dict[str, Any] | None


class GameCommandOutlet(Protocol):
    """Command transport required by the continuous Unity driving protocol."""

    def push(self, command: str) -> None: ...

    def push_with_ack(self, command: str) -> dict[str, Any]: ...

    def close(self) -> None: ...


class RealTimeDecoder:
    """Continuously decode sliding EEG windows on a background thread."""

    def __init__(
        self,
        acquirer: AbstractAcquirer,
        model: BaseModelAdapter,
        console: Console,
        command_outlet: LSLCommandOutlet,
        game_command_outlet: GameCommandOutlet | None,
        *,
        sfreq: float,
        window_sec: float,
        step_sec: float,
        confidence_threshold: float,
        mc_dropout_passes: int,
        online_update_enabled: bool = False,
        online_update_learning_rate: float = 1e-4,
        online_update_every: int = 1,
        model_save_path: Path | None = None,
        online_label_source: OnlineLabelSource | None = None,
        status_callback: Callable[[dict[str, Any]], None] | None = None,
        thread_context: Any | None = None,
        stop_on_game_disconnect: bool = True,
        batch_update_config: dict[str, Any] | None = None,
        n_classes: int = 3,
        experiment_config: dict[str, Any] | None = None,
        model_name: str | None = None,
        model_source_path: Path | None = None,
    ) -> None:
        self._acquirer = acquirer
        self._model = model
        self._model_lock = threading.RLock()
        self._model_revision = 0
        self._console = console
        self._command_outlet = command_outlet
        self._game_command_outlet = game_command_outlet
        self._sfreq = sfreq
        self._window_sec = window_sec
        self._step_sec = step_sec
        self._confidence_threshold = confidence_threshold
        self._mc_dropout_passes = mc_dropout_passes
        self._n_classes = max(int(n_classes), 1)
        self._online_update_enabled = bool(online_update_enabled)
        self._online_update_learning_rate = float(online_update_learning_rate)
        self._online_update_every = max(int(online_update_every), 1)
        self._model_save_path = model_save_path
        self._online_label_source = online_label_source
        self._lane_transition_guard_sec = max(
            float(getattr(online_label_source, "lane_transition_guard_sec", 0.0)),
            0.0,
        )
        self._pending_cued_windows: list[_PendingCuedWindow] = []
        self._status_callback = status_callback
        self._online_update_count = 0
        self._online_seen_labeled_windows = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_game_command: str | None = None
        self._last_game_transport_command: str | None = None
        self._last_game_transport_error: str | None = None
        self._last_game_transport_sent_at = 0.0
        self._last_game_movement_sent_at = 0.0
        self._game_command_keepalive_sec = max(0.2, min(0.5, step_sec * 1.1))
        self._game_session_started = False
        self._game_disconnect_message: str | None = None
        self._scene_sent_scene_index = -1
        self._scene_sent_label_id: int | None = None
        self._scene_sync_error: str | None = None
        self._failed_scene_indices: set[int] = set()
        self._scene_started_at: dict[int, float] = {}
        self._scene_labels: dict[int, int] = {}
        self._scene_start_lanes: dict[int, int] = {}
        self._scene_safe_lanes: dict[int, int] = {}
        self._scene_end_recorded: set[int] = set()
        self._timestamp_fallback_warned = False
        self._stop_on_game_disconnect = bool(stop_on_game_disconnect)
        self._thread_context = thread_context
        self._experiment_config = copy.deepcopy(experiment_config or {})
        self._model_name = str(
            model_name or getattr(model, "model_name", type(model).__name__)
        )
        self._model_source_path = (
            None if model_source_path is None else Path(model_source_path)
        )
        self._run_id = uuid4().hex
        self._model_revision_records: list[dict[str, Any]] = []
        neuroonline_config = NeuroOnlineConfig.from_mapping(batch_update_config)
        batch_config = BatchAdaptationConfig.from_mapping(batch_update_config)
        self._batch_adapter: OnlineBatchAdapter | None = None
        self._neuroonline_adapter: NeuroOnlineStreamAdapter | None = None
        self._last_batch_notice: tuple[Any, ...] | None = None
        self._neuroonline_training_notice = False
        if neuroonline_config.enabled:
            if not isinstance(model, TorchModelAdapter):
                raise ValueError("NeuroOnline requires a PyTorch decoder model.")
            if model_save_path is None:
                raise ValueError("NeuroOnline adaptation requires model_save_path.")
            self._model = NeuroOnlineModelAdapter(
                model,
                config=neuroonline_config,
                state_path=model_save_path,
            )
            neuroonline_config = self._model.config
            self._neuroonline_adapter = NeuroOnlineStreamAdapter(
                config=neuroonline_config,
                update_callback=self._run_neuroonline_update,
                save_callback=self._save_current_model,
                completion_callback=self._on_neuroonline_update_complete,
                n_classes=n_classes,
            )
        elif batch_config.enabled:
            if model_save_path is None:
                raise ValueError("Periodic online adaptation requires model_save_path.")
            self._batch_adapter = OnlineBatchAdapter(
                config=batch_config,
                model_getter=self._clone_current_model,
                model_swapper=self._swap_model,
                model_save_path=model_save_path,
                n_classes=n_classes,
            )

    def start(self) -> None:
        self._acquirer.start_stream()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._decode_loop, daemon=True)
        if self._thread_context is not None:
            try:
                from streamlit.runtime.scriptrunner import add_script_run_ctx
            except Exception:  # noqa: BLE001
                pass
            else:
                add_script_run_ctx(self._thread, self._thread_context)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        self._flush_pending_cued_windows(force=True)
        self._acquirer.stop_stream()
        if self._game_command_outlet is not None:
            try:
                self._game_command_outlet.push("STOP")
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("Failed to send final AR STOP: %s", exc)
            self._game_command_outlet.close()
        if self._batch_adapter is not None:
            self._batch_adapter.close()
        if self._neuroonline_adapter is not None:
            self._neuroonline_adapter.close()
        if self._online_label_source is not None:
            self._online_label_source.close()

    def run_forever(
        self,
        *,
        subject_id: str | None = None,
        save_dir: Path | None = None,
        record: bool = False,
        heartbeat: Callable[[], None] | None = None,
    ) -> None:
        self._record = record
        self._subject_id = subject_id
        self._last_game_command = None
        self._last_game_transport_command = None
        self._last_game_transport_error = None
        self._last_game_transport_sent_at = 0.0
        self._last_game_movement_sent_at = 0.0
        self._game_session_started = False
        self._game_disconnect_message = None
        self._scene_sent_scene_index = -1
        self._scene_sent_label_id = None
        self._scene_sync_error = None
        self._failed_scene_indices.clear()
        self._scene_started_at.clear()
        self._scene_labels.clear()
        self._scene_start_lanes.clear()
        self._scene_safe_lanes.clear()
        self._scene_end_recorded.clear()
        self._model_revision_records.clear()
        self._pending_cued_windows.clear()
        if record and subject_id:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            realtime_root = save_dir or Path("records_storage") / subject_id / "realtime"
            self._save_dir = realtime_root / timestamp
            self._writer = StreamWriter(self._save_dir)
            self._writer.start({
                "run_id": self._run_id,
                "subject_id": subject_id,
                "mode": "realtime",
                "sfreq": self._sfreq,
                "window_sec": self._window_sec,
                "step_sec": self._step_sec,
                "channels": self._acquirer.metadata.n_channels,
                "model_name": self._model_name,
                "model_revision": self._model_revision,
                "preprocessing": DEFAULT_PREPROCESSING.as_dict(),
                "online_adaptation": self._online_adaptation_status(),
                "online_label_source": self._online_label_source_metadata(),
                "provenance": self._build_run_provenance(),
            })
            self._writer.append_event(
                "session_start",
                run_id=self._run_id,
                subject_id=subject_id,
                model_name=self._model_name,
            )
            self._snapshot_model_revision(0, source="session_start")

        self._push_game_session_command("START")
        self.start()
        try:
            while not self._stop_event.is_set():
                self._sleep_with_heartbeat(min(0.1, max(self._step_sec, 0.1)), heartbeat)
                if heartbeat is not None:
                    heartbeat()
            if self._game_disconnect_message:
                raise RuntimeError(self._game_disconnect_message)
        except KeyboardInterrupt:
            self._console.print("\n[bold red]停止实时解码[/bold red]")
        finally:
            self.stop()
            self._record_active_scene_end(outcome="incomplete", reason="session_stop")
            if heartbeat is not None:
                heartbeat()
            if hasattr(self, "_writer"):
                self._writer.append_event(
                    "session_stop",
                    model_revision=self._model_revision,
                )
                self._writer.stop()
                self._writer.finalize_manifest(
                    {
                        "model_revision": self._model_revision,
                        "model_revisions": list(self._model_revision_records),
                        "online_adaptation": self._online_adaptation_status(),
                        "online_label_source": self._online_label_source_metadata(),
                        "timing_diagnostics": getattr(
                            self._acquirer,
                            "timing_diagnostics",
                            {},
                        ),
                    }
                )
                self._console.print(f"[bold green]实时数据已保存[/bold green] {self._save_dir}")

    def run_test_mode(
        self,
        *,
        subject_id: str,
        marker_backend: MarkerBackend,
        duration_sec: int,
        block_sec: float = 10.0,
        initial_rest_sec: float = 0.0,
        save_dir: Path | None = None,
        heartbeat: Callable[[], None] | None = None,
        stage_progress: Callable[[str, float, float], None] | None = None,
    ) -> dict[str, float | int | str]:
        """Run cue-based testing, save captured EEG/labels, and report accuracy."""

        self._console.print("[bold cyan]测试模式启动（有 cue）[/bold cyan]")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root_dir = save_dir or Path("records_storage") / subject_id / "test_mode" / timestamp
        writer = StreamWriter(root_dir)
        writer.start({
            "run_id": self._run_id,
            "subject_id": subject_id,
            "mode": "test_mode",
            "sfreq": self._sfreq,
            "window_sec": self._window_sec,
            "step_sec": self._step_sec,
            "channels": self._acquirer.metadata.n_channels,
            "preprocessing": DEFAULT_PREPROCESSING.as_dict(),
            "provenance": self._build_run_provenance(),
        })
        writer.append_event("session_start", run_id=self._run_id, mode="test_mode")
        
        def update_stage(stage_name: str, elapsed_sec: float, total_sec: float) -> None:
            if stage_progress is not None:
                stage_progress(stage_name, elapsed_sec, total_sec)

        self._acquirer.start_stream()
        if heartbeat is not None:
            heartbeat()
        if initial_rest_sec > 0:
            self._console.print(f"[bold yellow]Baseline 测试静息注视 ({initial_rest_sec:.0f}s)[/bold yellow]")
            update_stage("测试静息注视", 0.0, initial_rest_sec)
            self._sleep_with_stage_progress(
                initial_rest_sec,
                heartbeat=heartbeat,
                stage_name="测试静息注视",
                stage_progress=stage_progress,
            )
        started = time.monotonic()
        cue_index = 0
        labels = [0, 1, 2]
        collected_windows: list[np.ndarray] = []
        true_labels: list[int] = []
        pred_labels: list[int] = []
        confidences: list[float] = []
        quality_accepted: list[bool] = []
        update_losses: list[float] = []
        run_status = "completed"
        run_error: str | None = None
        try:
            while time.monotonic() - started < duration_sec:
                label = labels[cue_index % len(labels)]
                cue_index += 1
                self._console.print(f"[bold yellow][cue][/bold yellow] {TEST_MODE_PROMPTS[label]}")
                if heartbeat is not None:
                    heartbeat()
                marker_backend.send(label)
                writer.append_event(
                    "test_cue_start",
                    cue_index=cue_index - 1,
                    label_id=label,
                    label_name=TEST_MODE_PROMPTS[label],
                )
                
                # IMPORTANT: Delay for window_sec before starting to evaluate this cue.
                # If window is 4s, the immediate chunk returned still mostly contains data PROR to the cue.
                # We need to give the subject time to react and the ring buffer time to fill with the new intent.
                update_stage(f"测试 {cue_index}: cue {TEST_MODE_PROMPTS[label]}", 0.0, self._window_sec)
                self._sleep_with_stage_progress(
                    self._window_sec,
                    heartbeat=heartbeat,
                    stage_name=f"测试 {cue_index}: cue {TEST_MODE_PROMPTS[label]}",
                    stage_progress=stage_progress,
                )
                
                # Now we predict on the new block length
                # Since we already waited window_sec, we subtract this from the block duration to keep blocks same length
                control_sec = max(0.1, block_sec - self._window_sec)
                control_started = time.monotonic()
                block_end = control_started + control_sec
                update_stage(f"测试 {cue_index}: 预测控制", 0.0, control_sec)
                
                while time.monotonic() < block_end and time.monotonic() - started < duration_sec:
                    loop_started = time.perf_counter()
                    try:
                        window, timestamps = self._acquirer.get_chunk(self._window_sec)
                    except RuntimeError:
                        time.sleep(self._step_sec)
                        continue
                    window_start, window_end = self._resolve_window_time_bounds(timestamps)
                    preprocessing = preprocess_eeg_window(window, sfreq=self._sfreq)
                    processed = preprocessing.data
                    probability_batch, model_revision = self._predict_proba_with_revision(
                        processed[None, ...],
                        mc_dropout_passes=self._mc_dropout_passes,
                    )
                    probabilities = probability_batch[0]
                    raw_prediction = int(np.argmax(probabilities))
                    result = self._post_process(probabilities)
                    self._console.print(
                        f"[green][预测][/green] {result.label} "
                        f"(confidence: {result.confidence:.2f}, uncertainty: {result.uncertainty:.2f})"
                    )
                    self._command_outlet.push(result.label)
                    game_command = self._to_game_command(result)
                    self._push_game_command(game_command)
                    self._emit_status(result, game_command)
                    
                    pred_class = -1 if result.class_id is None else int(result.class_id)
                    timing_diagnostics = getattr(
                        self._acquirer,
                        "timing_diagnostics",
                        {},
                    ) or {}
                    writer.put(
                        window=window.astype(np.float32),
                        y_true=label,
                        y_pred=pred_class,
                        confidence=float(result.confidence),
                        raw_pred=raw_prediction,
                        model_revision=model_revision,
                        label_event_id=f"test-cue-{cue_index - 1:06d}",
                        probabilities=probabilities,
                        uncertainty=float(result.uncertainty),
                        window_start_monotonic=window_start,
                        window_end_monotonic=window_end,
                        scene_index=cue_index - 1,
                        scene_label=label,
                        mapped_command=game_command or "STOP",
                        transport_command=self._last_game_transport_command or "",
                        transport_success=(
                            self._last_game_transport_error is None
                            and self._last_game_transport_sent_at > 0.0
                        ),
                        transport_sent_at_monotonic=self._last_game_transport_sent_at,
                        transport_error=self._last_game_transport_error or "",
                        quality_accepted=preprocessing.quality.accepted,
                        quality_peak_abs_uv=preprocessing.quality.peak_abs_uv,
                        quality_clip_fraction=preprocessing.quality.clip_fraction,
                        quality_bad_channel_fraction=(
                            preprocessing.quality.bad_channel_fraction
                        ),
                        quality_reasons=preprocessing.quality.reasons,
                        quality_bad_channel_indices=(
                            preprocessing.quality.bad_channel_indices
                        ),
                        quality_nonfinite_fraction=(
                            preprocessing.quality.nonfinite_fraction
                        ),
                        timing_queueing_jitter_sec=float(
                            timing_diagnostics.get("queueing_jitter_sec", 0.0)
                        ),
                        timing_transport_delay_compensation_sec=float(
                            timing_diagnostics.get(
                                "transport_delay_compensation_sec",
                                0.0,
                            )
                        ),
                        timing_packet_arrival_monotonic=float(
                            timing_diagnostics.get(
                                "packet_arrival_monotonic",
                                float("nan"),
                            )
                        ),
                        timing_received_packets=float(
                            timing_diagnostics.get("received_packets", 0.0)
                        ),
                        timing_packet_loss_count=float(
                            timing_diagnostics.get("packet_loss_count", 0.0)
                        ),
                        timing_total_source_samples=float(
                            timing_diagnostics.get("total_source_samples", 0.0)
                        ),
                    )

                    update_metrics = (
                        self._maybe_update_model(
                            processed=processed,
                            true_label=label,
                        )
                        if preprocessing.quality.accepted
                        else None
                    )
                    if update_metrics:
                        loss = update_metrics.get("loss")
                        if loss is not None:
                            update_losses.append(float(loss))
                    
                    true_labels.append(label)
                    pred_labels.append(pred_class)
                    confidences.append(float(result.confidence))
                    quality_accepted.append(preprocessing.quality.accepted)
                    if heartbeat is not None:
                        heartbeat()
                    update_stage(
                        f"测试 {cue_index}: 预测控制",
                        min(time.monotonic() - control_started, control_sec),
                        control_sec,
                    )
                    elapsed = time.perf_counter() - loop_started
                    self._sleep_with_heartbeat(max(0.0, self._step_sec - elapsed), heartbeat)
        except KeyboardInterrupt:
            run_status = "interrupted"
            self._console.print("\n[bold red]停止测试模式[/bold red]")
        except Exception as exc:
            run_status = "failed"
            run_error = str(exc)
            raise
        finally:
            self.stop()
            writer.append_event("session_stop", status=run_status, error=run_error)
            writer.stop()
            if run_status == "failed":
                writer.finalize_manifest({"status": run_status, "error": run_error})
            if heartbeat is not None:
                heartbeat()

        if not true_labels:
            writer.finalize_manifest(
                {"status": "no_windows", "error": "No EEG windows were collected."}
            )
            raise RuntimeError("Test mode did not collect any EEG windows.")

        y_true = np.asarray(true_labels, dtype=np.int64)
        y_pred = np.asarray(pred_labels, dtype=np.int64)
        pred_valid = y_pred >= 0
        quality_mask = np.asarray(quality_accepted, dtype=np.bool_)
        accuracy = float(np.mean(y_pred == y_true))
        valid_accuracy = float(np.mean(y_pred[pred_valid] == y_true[pred_valid])) if np.any(pred_valid) else 0.0
        quality_prediction_mask = quality_mask & pred_valid
        quality_accuracy = (
            float(
                np.mean(
                    y_pred[quality_prediction_mask]
                    == y_true[quality_prediction_mask]
                )
            )
            if np.any(quality_prediction_mask)
            else 0.0
        )
        
        writer.finalize_manifest({
            "status": run_status,
            "accuracy": accuracy,
            "valid_accuracy": valid_accuracy,
            "quality_accuracy": quality_accuracy,
            "quality_accepted_windows": int(np.sum(quality_mask)),
            "quality_rejected_windows": int(np.sum(~quality_mask)),
            "online_update_enabled": self._online_update_enabled,
            "online_update_count": self._online_update_count,
            "online_update_learning_rate": self._online_update_learning_rate,
            "online_update_every": self._online_update_every,
            "online_update_mean_loss": float(np.mean(update_losses)) if update_losses else None,
        })
        if self._online_update_enabled and self._model_save_path is not None:
            self._model.save(self._model_save_path)
            self._console.print(f"[bold green]在线更新后的模型已保存[/bold green] {self._model_save_path}")
        self._console.print(f"[bold green]测试数据已保存[/bold green] {root_dir}")

        return {
            "windows": len(true_labels),
            "accuracy": accuracy,
            "valid_accuracy": valid_accuracy,
            "quality_accuracy": quality_accuracy,
            "quality_accepted_windows": int(np.sum(quality_mask)),
        }

    def _decode_loop(self) -> None:
        while not self._stop_event.is_set():
            started_at = time.perf_counter()
            try:
                try:
                    window, timestamps = self._acquirer.get_chunk(self._window_sec)
                except RuntimeError as exc:
                    if "Not enough data" in str(exc):
                        self._sleep_with_heartbeat(self._step_sec, None)
                        continue
                    raise
                window_start, window_end = self._resolve_window_time_bounds(timestamps)
                self._sync_game_scene()
                preprocessing = preprocess_eeg_window(window, sfreq=self._sfreq)
                processed = preprocessing.data
                probability_batch, model_revision = self._predict_proba_with_revision(
                    processed[None, ...],
                    mc_dropout_passes=self._mc_dropout_passes,
                )
                probabilities = probability_batch[0]
                raw_prediction = int(np.argmax(probabilities))
                result = self._post_process(probabilities)
                self._console.print(
                    f"[green][预测][/green] {result.label} "
                    f"(confidence: {result.confidence:.2f}, uncertainty: {result.uncertainty:.2f})"
                )
                self._command_outlet.push(result.label)
                game_command = self._to_game_command(result)
                self._push_game_command(game_command)
                self._emit_status(result, game_command)

                online_label = self._get_online_label(
                    window_start=window_start,
                    window_end=window_end,
                )
                record_payload = None
                if hasattr(self, "_record") and self._record and hasattr(self, "_writer"):
                    pred_class = -1 if result.class_id is None else int(result.class_id)
                    timing_diagnostics = getattr(
                        self._acquirer,
                        "timing_diagnostics",
                        {},
                    ) or {}
                    label_payload = (
                        getattr(online_label, "payload", None) or {}
                        if online_label is not None
                        else {}
                    )
                    scene_index = int(
                        label_payload.get(
                            "scene_index",
                            getattr(self, "_scene_sent_scene_index", -1),
                        )
                    )
                    record_payload = {
                        "window": window.astype(np.float32),
                        "y_pred": pred_class,
                        "confidence": float(result.confidence),
                        "raw_pred": raw_prediction,
                        "model_revision": model_revision,
                        "probabilities": probabilities,
                        "uncertainty": float(result.uncertainty),
                        "window_start_monotonic": window_start,
                        "window_end_monotonic": window_end,
                        "scene_index": scene_index,
                        "scene_start_lane": getattr(
                            self,
                            "_scene_start_lanes",
                            {},
                        ).get(scene_index, -9),
                        "scene_safe_lane": getattr(
                            self,
                            "_scene_safe_lanes",
                            {},
                        ).get(scene_index, -9),
                        "scene_failed": scene_index in self._failed_scene_indices,
                        "mapped_command": game_command or "STOP",
                        "transport_command": self._last_game_transport_command or "",
                        "transport_success": (
                            self._last_game_transport_error is None
                            and self._last_game_transport_sent_at > 0.0
                        ),
                        "transport_sent_at_monotonic": self._last_game_transport_sent_at,
                        "transport_error": self._last_game_transport_error or "",
                        "quality_accepted": preprocessing.quality.accepted,
                        "quality_peak_abs_uv": preprocessing.quality.peak_abs_uv,
                        "quality_clip_fraction": preprocessing.quality.clip_fraction,
                        "quality_bad_channel_fraction": (
                            preprocessing.quality.bad_channel_fraction
                        ),
                        "quality_reasons": preprocessing.quality.reasons,
                        "quality_bad_channel_indices": (
                            preprocessing.quality.bad_channel_indices
                        ),
                        "quality_nonfinite_fraction": (
                            preprocessing.quality.nonfinite_fraction
                        ),
                        "timing_queueing_jitter_sec": float(
                            timing_diagnostics.get("queueing_jitter_sec", 0.0)
                        ),
                        "timing_transport_delay_compensation_sec": float(
                            timing_diagnostics.get(
                                "transport_delay_compensation_sec",
                                0.0,
                            )
                        ),
                        "timing_packet_arrival_monotonic": float(
                            timing_diagnostics.get(
                                "packet_arrival_monotonic",
                                float("nan"),
                            )
                        ),
                        "timing_received_packets": float(
                            timing_diagnostics.get("received_packets", 0.0)
                        ),
                        "timing_packet_loss_count": float(
                            timing_diagnostics.get("packet_loss_count", 0.0)
                        ),
                        "timing_total_source_samples": float(
                            timing_diagnostics.get("total_source_samples", 0.0)
                        ),
                    }

                if (
                    online_label is not None
                    and str(getattr(online_label, "source", "")) == "cued-protocol"
                    and self._lane_transition_guard_sec > 0.0
                ):
                    self._pending_cued_windows.append(
                        _PendingCuedWindow(
                            processed=processed.copy(),
                            probabilities=np.asarray(
                                probabilities,
                                dtype=np.float32,
                            ).copy(),
                            operational_prediction=result.class_id,
                            prediction_model_revision=model_revision,
                            online_label=online_label,
                            window_start=window_start,
                            window_end=window_end,
                            quality_accepted=bool(
                                preprocessing.quality.accepted
                            ),
                            record_payload=record_payload,
                        )
                    )
                else:
                    self._finalize_realtime_window(
                        processed=processed,
                        probabilities=probabilities,
                        operational_prediction=result.class_id,
                        prediction_model_revision=model_revision,
                        online_label=online_label,
                        window_end=window_end,
                        quality_accepted=bool(preprocessing.quality.accepted),
                        record_payload=record_payload,
                    )
                self._flush_pending_cued_windows()
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Realtime decoding failed")
                self._console.print(f"[red]解码失败：{exc}[/red]")

            elapsed = time.perf_counter() - started_at
            sleep_time = max(0.0, self._step_sec - elapsed)
            self._sleep_with_heartbeat(sleep_time, None)

    def _flush_pending_cued_windows(
        self,
        *,
        now: float | None = None,
        force: bool = False,
    ) -> None:
        """Finalize delayed labels after future lane-transition events are known."""

        pending = list(getattr(self, "_pending_cued_windows", []))
        if not pending:
            return
        timestamp = time.monotonic() if now is None else float(now)
        guard = max(float(getattr(self, "_lane_transition_guard_sec", 0.0)), 0.0)
        source = getattr(self, "_online_label_source", None)
        is_guarded = getattr(source, "is_window_transition_guarded", None)
        remaining: list[_PendingCuedWindow] = []
        for item in pending:
            matured = timestamp >= item.window_end + guard
            if not matured and not force:
                remaining.append(item)
                continue

            label_payload = getattr(item.online_label, "payload", None) or {}
            scene_index = int(label_payload.get("scene_index", -1))
            transition_guarded = bool(
                callable(is_guarded)
                and is_guarded(
                    scene_index=scene_index,
                    window_start=item.window_start,
                    window_end=item.window_end,
                )
            )
            shutdown_unconfirmed = bool(force and not matured)
            final_label = (
                None
                if transition_guarded or shutdown_unconfirmed
                else item.online_label
            )
            self._finalize_realtime_window(
                processed=item.processed,
                probabilities=item.probabilities,
                operational_prediction=item.operational_prediction,
                prediction_model_revision=item.prediction_model_revision,
                online_label=final_label,
                window_end=item.window_end,
                quality_accepted=item.quality_accepted,
                record_payload=item.record_payload,
            )
            writer = getattr(self, "_writer", None)
            if writer is not None and (transition_guarded or shutdown_unconfirmed):
                writer.append_event(
                    "training_label_rejected",
                    timestamp_monotonic=timestamp,
                    scene_index=scene_index,
                    window_start_monotonic=item.window_start,
                    window_end_monotonic=item.window_end,
                    original_label_id=int(item.online_label.label_id),
                    reason=(
                        "lane_transition_guard"
                        if transition_guarded
                        else "session_stopped_before_guard_confirmation"
                    ),
                    lane_transition_guard_sec=guard,
                )
        self._pending_cued_windows = remaining

    def _finalize_realtime_window(
        self,
        *,
        processed: np.ndarray,
        probabilities: np.ndarray,
        operational_prediction: int | None,
        prediction_model_revision: int,
        online_label: Any | None,
        window_end: float,
        quality_accepted: bool,
        record_payload: dict[str, Any] | None,
    ) -> None:
        """Commit one transition-safe window to adaptation and recording."""

        if online_label is not None and quality_accepted:
            self._handle_online_label(
                processed=processed,
                probabilities=probabilities,
                operational_prediction=operational_prediction,
                prediction_model_revision=prediction_model_revision,
                online_label=online_label,
                window_end=window_end,
            )
        if record_payload is None:
            return
        self._writer.put(
            y_true=-1 if online_label is None else int(online_label.label_id),
            label_event_id=(
                "" if online_label is None else str(online_label.event_id)
            ),
            scene_label=-1 if online_label is None else int(online_label.label_id),
            **record_payload,
        )

    def _resolve_window_time_bounds(self, timestamps: np.ndarray) -> tuple[float, float]:
        """Resolve an EEG window on the same monotonic clock used by Unity."""

        domain = str(
            getattr(self._acquirer.metadata, "timestamp_domain", "relative")
        ).strip().lower()
        values = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if domain == "monotonic" and values.size:
            window_start = float(values[0])
            window_end = float(values[-1]) + (1.0 / float(self._sfreq))
            duration = window_end - window_start
            now = time.monotonic()
            if (
                np.all(np.isfinite(values))
                and window_end >= window_start
                and np.all(np.diff(values) >= 0.0)
                and abs(duration - self._window_sec) <= max(2.0 / self._sfreq, 0.02)
                and window_end <= now + max(2.0 / self._sfreq, 0.02)
            ):
                return window_start, window_end

        if domain == "monotonic" and not self._timestamp_fallback_warned:
            LOGGER.warning(
                "Acquirer supplied invalid monotonic timestamps; falling back to local retrieval time."
            )
            self._timestamp_fallback_warned = True
        window_end = time.monotonic()
        return window_end - self._window_sec, window_end

    def _handle_online_label(
        self,
        *,
        processed: np.ndarray,
        probabilities: np.ndarray,
        operational_prediction: int | None,
        prediction_model_revision: int,
        online_label: Any,
        window_end: float,
    ) -> None:
        """Route one labeled window to the configured adaptation strategy."""

        label_id = int(online_label.label_id)
        if self._neuroonline_adapter is not None:
            self._neuroonline_adapter.add_window(
                processed,
                label_id,
                predicted_label=int(np.argmax(probabilities)),
                operational_predicted_label=operational_prediction,
                probabilities=probabilities,
                event_id=str(getattr(online_label, "event_id", "")),
                model_revision=prediction_model_revision,
            )
            status = self._neuroonline_adapter.status()
            if status.get("training_in_background") and not self._neuroonline_training_notice:
                self._neuroonline_training_notice = True
                self._console.print(
                    "[bold cyan]NeuroOnline 已在后台训练候选模型，实时预测继续使用当前模型[/bold cyan]"
                )
            return

        if self._batch_adapter is not None:
            event_id = str(
                getattr(online_label, "event_id", "")
                or f"label-{float(online_label.timestamp_monotonic):.6f}"
            )
            self._batch_adapter.add_window(
                processed,
                label_id,
                event_id=event_id,
                now=window_end,
            )
            self._report_batch_adaptation_status()
            return

        self._maybe_update_model(processed=processed, true_label=label_id)

    def _maybe_update_model(
        self,
        *,
        processed: np.ndarray,
        true_label: int,
    ) -> dict[str, float] | None:
        if not self._online_update_enabled:
            return None

        self._online_seen_labeled_windows += 1
        if self._online_seen_labeled_windows % self._online_update_every != 0:
            return None

        update = getattr(self._model, "update", None)
        if not callable(update):
            return None

        model_lock = getattr(self, "_model_lock", None)
        with model_lock if model_lock is not None else nullcontext():
            metrics = update(
                processed[None, ...].astype(np.float32),
                np.asarray([int(true_label)], dtype=np.int64),
                learning_rate=self._online_update_learning_rate,
                epochs=1,
            )
        updated = float(metrics.get("updated", 0.0)) if isinstance(metrics, dict) else 0.0
        if updated > 0:
            self._online_update_count += int(updated)
        return metrics if isinstance(metrics, dict) else None

    def _get_online_label(
        self,
        *,
        window_start: float,
        window_end: float,
    ) -> Any | None:
        if self._online_label_source is None:
            return None
        try:
            label = self._online_label_source.get_label(
                window_start=window_start,
                window_end=window_end,
            )
            if label is None or str(getattr(label, "source", "")) != "cued-protocol":
                return label
            payload = getattr(label, "payload", None) or {}
            label_scene_index = int(payload.get("scene_index", -1))
            if label_scene_index != getattr(self, "_scene_sent_scene_index", -1):
                return None
            return label
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to read online label: %s", exc)
            return None

    def _predict_proba(self, X: np.ndarray, *, mc_dropout_passes: int) -> np.ndarray:
        with self._model_lock:
            return self._model.predict_proba(X, mc_dropout_passes=mc_dropout_passes)

    def _predict_proba_with_revision(
        self,
        X: np.ndarray,
        *,
        mc_dropout_passes: int,
    ) -> tuple[np.ndarray, int]:
        with self._model_lock:
            probabilities = self._model.predict_proba(X, mc_dropout_passes=mc_dropout_passes)
            return probabilities, self._model_revision

    def _clone_current_model(self) -> BaseModelAdapter:
        with self._model_lock:
            return copy.deepcopy(self._model)

    def _run_neuroonline_update(
        self,
        original: np.ndarray,
        time_masked: np.ndarray,
        frequency_masked: np.ndarray,
        labels: np.ndarray,
    ) -> dict[str, Any]:
        update_started_monotonic = time.monotonic()
        with self._model_lock:
            if not isinstance(self._model, NeuroOnlineModelAdapter):
                raise RuntimeError("NeuroOnline model adapter is not active")
            base_model_revision = self._model_revision
            candidate = copy.deepcopy(self._model)
        writer = getattr(self, "_writer", None)
        if writer is not None:
            writer.append_event(
                "model_update_start",
                timestamp_monotonic=update_started_monotonic,
                base_model_revision=base_model_revision,
                training_samples=int(labels.shape[0]),
                class_counts=np.bincount(
                    np.asarray(labels, dtype=np.int64),
                    minlength=self._n_classes,
                ).tolist(),
            )
        result = candidate.neuroonline_update(
            original,
            time_masked,
            frequency_masked,
            labels,
        )
        with self._model_lock:
            self._model = candidate
            self._model_revision += 1
            result["model_revision"] = self._model_revision
        result["base_model_revision"] = base_model_revision
        result["swap_timestamp_monotonic"] = time.monotonic()
        if writer is not None:
            writer.append_event(
                "model_swap",
                timestamp_monotonic=float(result["swap_timestamp_monotonic"]),
                base_model_revision=base_model_revision,
                model_revision=self._model_revision,
                training_samples=int(labels.shape[0]),
            )
        return result

    def _on_neuroonline_update_complete(self, result: dict[str, Any]) -> None:
        self._neuroonline_training_notice = False
        if result.get("error"):
            self._console.print(f"[bold red]NeuroOnline 后台更新失败[/bold red] {result['error']}")
        else:
            self._console.print(
                "[bold green]NeuroOnline 候选模型已原子切换[/bold green] "
                f"revision={int(result.get('model_revision', self._model_revision))} "
                f"loss={float(result.get('loss', 0.0)):.4f}"
            )
        writer = getattr(self, "_writer", None)
        if writer is not None:
            writer.append_event(
                "model_update_complete",
                model_revision=int(result.get("model_revision", self._model_revision)),
                **{
                    key: value
                    for key, value in result.items()
                    if key not in {"model_revision"}
                },
            )
        self._persist_online_adaptation_status()

    def _save_current_model(self) -> None:
        if self._model_save_path is None:
            return
        with self._model_lock:
            model_snapshot = copy.deepcopy(self._model)
            revision = getattr(self, "_model_revision", 0)
        self._model_save_path.parent.mkdir(parents=True, exist_ok=True)
        model_snapshot.save(self._model_save_path)
        self._snapshot_model_revision(
            revision,
            source="online_update",
            model_snapshot=model_snapshot,
        )

    def _swap_model(self, candidate: BaseModelAdapter) -> None:
        with self._model_lock:
            self._model = candidate
            self._model_revision += 1

    def _persist_online_adaptation_status(self) -> None:
        writer = getattr(self, "_writer", None)
        if writer is None:
            return
        writer.update_manifest(
            {
                "model_revision": self._model_revision,
                "online_adaptation": self._online_adaptation_status(),
                "online_label_source": self._online_label_source_metadata(),
            }
        )

    def _report_batch_adaptation_status(self) -> None:
        if self._batch_adapter is None:
            return
        status = self._batch_adapter.status()
        result = status.get("last_result")
        if isinstance(result, dict):
            result_key = (
                result.get("cycle"),
                result.get("accepted"),
                result.get("error"),
                result.get("model_version"),
            )
        else:
            result_key = (str(result),)
        notice_key = (status.get("state"),) + result_key
        if notice_key == self._last_batch_notice:
            return
        self._last_batch_notice = notice_key

        if status.get("state") == "training":
            self._console.print("[bold cyan]10分钟数据已就绪，正在后台训练候选分类头[/bold cyan]")
            return
        if isinstance(result, dict) and result.get("accepted"):
            self._console.print(
                "[bold green]周期模型更新已接受[/bold green] "
                f"v{result.get('model_version')} balanced_accuracy_gain="
                f"{float(result.get('balanced_accuracy_gain', 0.0)):+.3f}"
            )
        elif isinstance(result, dict) and result.get("error"):
            self._console.print(f"[bold red]周期模型更新失败[/bold red] {result['error']}")
        elif isinstance(result, dict):
            self._console.print(
                "[bold yellow]候选模型未通过验证，继续使用旧模型[/bold yellow] "
                f"balanced_accuracy_gain={float(result.get('balanced_accuracy_gain', 0.0)):+.3f}"
            )
        elif result:
            self._console.print(f"[bold yellow]周期模型更新暂缓[/bold yellow] {result}")

    @staticmethod
    def _sleep_with_heartbeat(duration_sec: float, heartbeat: Callable[[], None] | None) -> None:
        remaining = max(float(duration_sec), 0.0)
        while remaining > 0:
            chunk = min(0.1, remaining)
            time.sleep(chunk)
            remaining -= chunk
            if heartbeat is not None:
                heartbeat()

    def _sleep_with_stage_progress(
        self,
        duration_sec: float,
        *,
        heartbeat: Callable[[], None] | None,
        stage_name: str,
        stage_progress: Callable[[str, float, float], None] | None,
    ) -> None:
        total = max(float(duration_sec), 0.0)
        started_at = time.monotonic()
        remaining = total
        while remaining > 0:
            chunk = min(0.1, remaining)
            time.sleep(chunk)
            remaining -= chunk
            elapsed = min(time.monotonic() - started_at, total)
            if stage_progress is not None:
                stage_progress(stage_name, elapsed, total)
            if heartbeat is not None:
                heartbeat()

    def _post_process(self, probabilities: np.ndarray) -> PredictionResult:
        best_index = int(np.argmax(probabilities))
        confidence = float(probabilities[best_index])
        uncertainty = float(1.0 - confidence)
        if confidence < self._confidence_threshold:
            return PredictionResult(
                label="静息",
                confidence=confidence,
                uncertainty=uncertainty,
                class_id=None,
            )
        return PredictionResult(
            label=LABEL_NAMES.get(best_index, f"class-{best_index}"),
            confidence=confidence,
            uncertainty=uncertainty,
            class_id=best_index,
        )

    @staticmethod
    def _to_game_command(result: PredictionResult) -> str | None:
        if result.class_id == 0:
            return "LEFT"
        if result.class_id == 1:
            return "RIGHT"
        return None

    def _sync_game_scene(self) -> None:
        """Negotiate a reachable relative-action scene with authoritative Unity truth."""

        self._consume_game_scene_events()
        status = self._online_label_source_status()
        if not status or status.get("source") != "cued-protocol":
            return
        if status.get("protocol_mode") != "continuous-relative-action":
            return
        if status.get("phase") == "preparing":
            return
        scene_index = int(status.get("scene_index", -1))
        label_value = status.get("label_id")
        label_id = -1 if label_value is None else int(label_value)
        if (
            scene_index == getattr(self, "_scene_sent_scene_index", -1)
            and label_id == getattr(self, "_scene_sent_label_id", None)
        ):
            return

        if label_id < 0:
            state_ack = self._push_game_scene_transport_command("SCENE_STATE")
            if state_ack is None:
                self._scene_sync_error = (
                    self._last_game_transport_error
                    or "Unity lane-state query failed"
                )
                return
            try:
                self._validate_unity_protocol_ack(
                    state_ack,
                    expected_ack="SCENE_STATE",
                    expected_scene_number=scene_index + 1,
                )
                start_lane = int(state_ack["current_lane"])
                prepare_scene = getattr(
                    self._online_label_source,
                    "prepare_scene",
                    None,
                )
                if not callable(prepare_scene):
                    raise RuntimeError(
                        "Relative-action label source does not support scene preparation."
                    )
                label_id = int(
                    prepare_scene(
                        scene_index=scene_index,
                        start_lane=start_lane,
                    )
                )
                status = self._online_label_source_status() or {}
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                self._abort_scene_protocol(f"invalid Unity lane-state ACK: {exc}")
                return

        command_by_label = {0: "SCENE_LEFT", 1: "SCENE_RIGHT", 2: "SCENE_IDLE"}
        command = command_by_label.get(label_id)
        if command is None:
            self._scene_sync_error = f"unsupported scene label id: {label_id}"
            return
        previous_scene = getattr(self, "_scene_sent_scene_index", -1)
        if previous_scene >= 0 and previous_scene != scene_index:
            failed_scene_indices = getattr(self, "_failed_scene_indices", set())
            self._record_scene_end(
                previous_scene,
                outcome=(
                    "failed"
                    if previous_scene in failed_scene_indices
                    else "success"
                ),
                reason="fixed_boundary",
            )
        scene_ack = self._push_game_scene_transport_command(command)
        if scene_ack is not None:
            try:
                self._validate_unity_protocol_ack(
                    scene_ack,
                    expected_ack=command,
                    expected_scene_number=scene_index + 1,
                )
                applied_label_name = str(scene_ack["applied_label"]).strip().lower()
                applied_label_id = {"left": 0, "right": 1, "idle": 2}[
                    applied_label_name
                ]
                start_lane = int(scene_ack["start_lane"])
                safe_lane = int(scene_ack["safe_lane"])
                if applied_label_id != label_id:
                    raise ValueError(
                        f"requested label {label_id}, Unity applied {applied_label_id}"
                    )
            except (KeyError, TypeError, ValueError) as exc:
                self._abort_scene_protocol(f"invalid Unity scene ACK: {exc}")
                return
            confirm_scene = getattr(
                self._online_label_source,
                "confirm_scene_applied",
                None,
            )
            if callable(confirm_scene) and not confirm_scene(
                scene_index=scene_index,
                applied_label_id=applied_label_id,
                start_lane=start_lane,
                safe_lane=safe_lane,
                timestamp_monotonic=self._last_game_transport_sent_at,
            ):
                self._abort_scene_protocol(
                    "Unity scene ACK did not match the prepared relative-action truth "
                    f"for scene {scene_index + 1}: label={applied_label_name}, "
                    f"start_lane={start_lane}, safe_lane={safe_lane}"
                )
                return
            self._scene_sent_scene_index = scene_index
            self._scene_sent_label_id = label_id
            if not hasattr(self, "_scene_started_at"):
                self._scene_started_at = {}
            if not hasattr(self, "_scene_labels"):
                self._scene_labels = {}
            if not hasattr(self, "_scene_start_lanes"):
                self._scene_start_lanes = {}
            if not hasattr(self, "_scene_safe_lanes"):
                self._scene_safe_lanes = {}
            self._scene_started_at[scene_index] = self._last_game_transport_sent_at
            self._scene_labels[scene_index] = label_id
            self._scene_start_lanes[scene_index] = start_lane
            self._scene_safe_lanes[scene_index] = safe_lane
            writer = getattr(self, "_writer", None)
            if writer is not None:
                writer.append_event(
                    "scene_start",
                    timestamp_monotonic=self._last_game_transport_sent_at,
                    scene_index=scene_index,
                    scene_number=scene_index + 1,
                    label_id=label_id,
                    label_name=status.get("label_name"),
                    unity_command=command,
                    ack_confirmed=True,
                    protocol_version=CUED_PROTOCOL_VERSION,
                    start_lane=start_lane,
                    safe_lane=safe_lane,
                    applied_label=applied_label_name,
                    planned_duration_sec=float(
                        self._online_label_source_metadata().get(
                            "scene_duration_sec",
                            0.0,
                        )
                        if self._online_label_source_metadata()
                        else 0.0
                    ),
                )
            self._scene_sync_error = None
            return
        self._scene_sync_error = self._last_game_transport_error or "scene command send failed"

    def _consume_game_scene_events(self) -> None:
        if self._game_command_outlet is None or self._online_label_source is None:
            return
        poll_events = getattr(self._game_command_outlet, "poll_events", None)
        mark_scene_failed = getattr(
            self._online_label_source,
            "mark_scene_failed",
            None,
        )
        if not callable(poll_events) or not callable(mark_scene_failed):
            return
        try:
            events = poll_events()
        except Exception as exc:  # noqa: BLE001
            self._last_game_transport_error = str(exc)
            LOGGER.warning("Failed to poll Unity scene events: %s", exc)
            if self._stop_on_game_disconnect:
                self._game_disconnect_message = f"Unity scene event connection lost: {exc}"
                self._stop_event.set()
            return

        for event in events:
            event_name = str(event.get("event", "")).strip().upper()
            if event_name == "LANE_SETTLED":
                self._handle_lane_settled_event(event)
                continue
            if event_name != "SCENE_FAILED":
                continue
            failed_scene_index = int(event.get("scene_number", 0)) - 1
            if failed_scene_index != self._scene_sent_scene_index:
                LOGGER.warning(
                    "Ignored stale Unity SCENE_FAILED event scene_number=%s; current=%s",
                    event.get("scene_number"),
                    self._scene_sent_scene_index + 1,
                )
                continue
            recorded = mark_scene_failed(
                timestamp_monotonic=time.monotonic(),
                expected_scene_index=failed_scene_index,
            )
            if recorded:
                if not hasattr(self, "_failed_scene_indices"):
                    self._failed_scene_indices = set()
                self._failed_scene_indices.add(failed_scene_index)
                writer = getattr(self, "_writer", None)
                if writer is not None:
                    writer.append_event(
                        "scene_failed",
                        timestamp_monotonic=time.monotonic(),
                        scene_index=failed_scene_index,
                        scene_number=failed_scene_index + 1,
                        label_id=self._scene_labels.get(
                            failed_scene_index,
                            self._scene_sent_label_id,
                        ),
                        unity_event=dict(event),
                    )
                self._console.print(
                    f"[bold yellow]Scene {failed_scene_index + 1} 避障失败；"
                    "保持当前安全车道和动态标签规则，到固定 Scene 边界再进入下一 Scene[/bold yellow]"
                )

    def _handle_lane_settled_event(self, event: dict[str, Any]) -> None:
        """Convert Unity's completed lane transition into a new truth segment."""

        if str(event.get("protocol_version", "")).strip() != CUED_PROTOCOL_VERSION:
            self._abort_scene_protocol(
                f"invalid LANE_SETTLED protocol version: {event!r}"
            )
            return
        scene_index = int(event.get("scene_number", 0)) - 1
        if scene_index != self._scene_sent_scene_index:
            LOGGER.warning(
                "Ignored stale Unity LANE_SETTLED event scene_number=%s; current=%s",
                event.get("scene_number"),
                self._scene_sent_scene_index + 1,
            )
            return
        try:
            current_lane = int(event["current_lane"])
            safe_lane = int(event["safe_lane"])
        except (KeyError, TypeError, ValueError):
            self._abort_scene_protocol(f"invalid LANE_SETTLED payload: {event!r}")
            return
        expected_safe_lane = self._scene_safe_lanes.get(scene_index)
        if (
            current_lane not in {-1, 0, 1}
            or safe_lane not in {-1, 0, 1}
            or expected_safe_lane != safe_lane
        ):
            self._abort_scene_protocol(
                "Unity LANE_SETTLED event does not match active safe-lane truth: "
                f"{event!r}"
            )
            return
        update_current_lane = getattr(
            self._online_label_source,
            "update_current_lane",
            None,
        )
        transition_time = time.monotonic()
        if not callable(update_current_lane) or not update_current_lane(
            scene_index=scene_index,
            current_lane=current_lane,
            safe_lane=safe_lane,
            timestamp_monotonic=transition_time,
        ):
            self._abort_scene_protocol(
                f"Rejected Unity LANE_SETTLED event: {event!r}"
            )
            return
        writer = getattr(self, "_writer", None)
        if writer is not None:
            writer.append_event(
                "lane_settled",
                timestamp_monotonic=transition_time,
                scene_index=scene_index,
                scene_number=scene_index + 1,
                current_lane=current_lane,
                safe_lane=safe_lane,
                dynamic_label_id=(
                    0 if current_lane > safe_lane
                    else 1 if current_lane < safe_lane
                    else 2
                ),
                unity_event=dict(event),
            )

    @staticmethod
    def _validate_unity_protocol_ack(
        response: dict[str, Any],
        *,
        expected_ack: str,
        expected_scene_number: int,
    ) -> None:
        if str(response.get("ack", "")).strip().upper() != expected_ack:
            raise ValueError(f"expected ACK {expected_ack}, received {response!r}")
        if (
            str(response.get("protocol_version", "")).strip()
            != CUED_PROTOCOL_VERSION
        ):
            raise ValueError(
                f"Unity runtime does not implement {CUED_PROTOCOL_VERSION}"
            )
        if int(response.get("scene_number", -1)) != int(expected_scene_number):
            raise ValueError(
                f"expected scene {expected_scene_number}, "
                f"received {response.get('scene_number')}"
            )

    def _abort_scene_protocol(self, message: str) -> None:
        self._scene_sync_error = message
        self._last_game_transport_error = message
        LOGGER.error("Unity relative-action scene protocol failed: %s", message)
        if self._stop_on_game_disconnect:
            self._game_disconnect_message = message
            self._stop_event.set()

    def _push_game_scene_transport_command(
        self,
        command: str,
    ) -> dict[str, Any] | None:
        if self._game_command_outlet is None:
            return None
        push_with_ack = getattr(self._game_command_outlet, "push_with_ack", None)
        if not callable(push_with_ack):
            self._last_game_transport_error = "AR transport does not support Unity scene ACK"
            return None
        try:
            response = push_with_ack(command)
            if not isinstance(response, dict):
                raise RuntimeError(
                    f"Unity returned no structured ACK for {command}"
                )
            self._last_game_transport_command = command
            self._last_game_transport_error = None
            self._last_game_transport_sent_at = time.monotonic()
            return response
        except Exception as exc:  # noqa: BLE001
            self._last_game_transport_error = str(exc)
            LOGGER.warning("Failed to synchronize Unity scene '%s': %s", command, exc)
            if self._stop_on_game_disconnect:
                self._game_disconnect_message = f"Unity scene synchronization failed: {exc}"
                self._stop_event.set()
            return None

    def _push_game_command(self, command: str | None) -> None:
        if self._game_command_outlet is None:
            return

        if command is None:
            if self._last_game_command is None:
                self._push_game_keepalive()
                return
            self._push_game_transport_command("STOP", movement=True)
            self._last_game_command = None
            return

        self._push_game_session_command("START")
        now = time.monotonic()
        if (
            command == self._last_game_command
            and now - self._last_game_movement_sent_at < self._game_command_keepalive_sec
        ):
            return

        if self._push_game_transport_command(command, movement=True):
            self._last_game_command = command

    def _push_game_session_command(self, command: str) -> None:
        if self._game_command_outlet is None or self._game_session_started:
            return
        if self._push_game_transport_command(command):
            self._game_session_started = True

    def _push_game_keepalive(self) -> None:
        if not self._game_session_started:
            return
        now = time.monotonic()
        if now - self._last_game_movement_sent_at < self._game_command_keepalive_sec:
            return
        self._push_game_transport_command("STOP", movement=True)

    def _push_game_transport_command(self, command: str, *, movement: bool = False) -> bool:
        if self._game_command_outlet is None:
            return False
        try:
            self._game_command_outlet.push(command)
            self._last_game_transport_command = command
            self._last_game_transport_error = None
            self._last_game_transport_sent_at = time.monotonic()
            if movement:
                self._last_game_movement_sent_at = self._last_game_transport_sent_at
            return True
        except Exception as exc:  # noqa: BLE001
            self._last_game_transport_error = str(exc)
            LOGGER.warning("Failed to push AR game command '%s': %s", command, exc)
            if self._stop_on_game_disconnect:
                self._game_disconnect_message = f"Unity game connection lost: {exc}"
                self._stop_event.set()
            return False

    def _record_scene_end(
        self,
        scene_index: int,
        *,
        outcome: str,
        reason: str,
        timestamp_monotonic: float | None = None,
    ) -> None:
        index = int(scene_index)
        scene_end_recorded = getattr(self, "_scene_end_recorded", set())
        if index < 0 or index in scene_end_recorded:
            return
        ended_at = (
            time.monotonic()
            if timestamp_monotonic is None
            else float(timestamp_monotonic)
        )
        started_at = getattr(self, "_scene_started_at", {}).get(index)
        failed_scene_indices = getattr(self, "_failed_scene_indices", set())
        writer = getattr(self, "_writer", None)
        if writer is not None:
            writer.append_event(
                "scene_end",
                timestamp_monotonic=ended_at,
                scene_index=index,
                scene_number=index + 1,
                label_id=getattr(self, "_scene_labels", {}).get(index, -1),
                start_lane=getattr(self, "_scene_start_lanes", {}).get(index),
                safe_lane=getattr(self, "_scene_safe_lanes", {}).get(index),
                outcome=str(outcome),
                reason=str(reason),
                collision_recorded=index in failed_scene_indices,
                duration_sec=(
                    None if started_at is None else max(ended_at - started_at, 0.0)
                ),
            )
        if not hasattr(self, "_scene_end_recorded"):
            self._scene_end_recorded = set()
        self._scene_end_recorded.add(index)

    def _record_active_scene_end(self, *, outcome: str, reason: str) -> None:
        self._record_scene_end(
            getattr(self, "_scene_sent_scene_index", -1),
            outcome=outcome,
            reason=reason,
        )

    def _snapshot_model_revision(
        self,
        revision: int,
        *,
        source: str,
        model_snapshot: BaseModelAdapter | None = None,
    ) -> None:
        writer = getattr(self, "_writer", None)
        save_dir = getattr(self, "_save_dir", None)
        if writer is None or save_dir is None:
            return
        revision_value = int(revision)
        if any(
            int(record.get("model_revision", -1)) == revision_value
            for record in self._model_revision_records
        ):
            return
        revision_dir = Path(save_dir) / "model_revisions"
        revision_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = revision_dir / f"revision_{revision_value:04d}.pt"
        sidecar = Path(f"{checkpoint}.neuroonline.pt")
        if (
            revision_value == 0
            and self._model_source_path is not None
            and self._model_source_path.exists()
        ):
            shutil.copy2(self._model_source_path, checkpoint)
            source_sidecar = Path(f"{self._model_source_path}.neuroonline.pt")
            if source_sidecar.exists():
                shutil.copy2(source_sidecar, sidecar)
        elif model_snapshot is not None:
            model_snapshot.save(checkpoint)
        else:
            with self._model_lock:
                snapshot = copy.deepcopy(self._model)
            snapshot.save(checkpoint)
        record: dict[str, Any] = {
            "model_revision": revision_value,
            "source": str(source),
            "checkpoint": checkpoint.relative_to(save_dir).as_posix(),
            "checkpoint_sha256": self._sha256_file(checkpoint),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        if sidecar.exists():
            record["crm_sidecar"] = sidecar.relative_to(save_dir).as_posix()
            record["crm_sidecar_sha256"] = self._sha256_file(sidecar)
        self._model_revision_records.append(record)
        writer.append_event("model_checkpoint", **record)

    def _build_run_provenance(self) -> dict[str, Any]:
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

        package_versions: dict[str, str | None] = {}
        for package in ("numpy", "scipy", "torch", "scikit-learn", "mne"):
            try:
                package_versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                package_versions[package] = None
        config_json = json.dumps(
            self._experiment_config,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        storage_config = self._experiment_config.get("storage", {}) or {}
        native_recording_id = str(
            storage_config.get("native_recording_id", "") or ""
        ).strip()
        model_files: list[dict[str, Any]] = []
        if self._model_source_path is not None and self._model_source_path.exists():
            model_files.append(
                {
                    "path": str(self._model_source_path),
                    "sha256": self._sha256_file(self._model_source_path),
                    "size_bytes": self._model_source_path.stat().st_size,
                }
            )
            source_sidecar = Path(f"{self._model_source_path}.neuroonline.pt")
            if source_sidecar.exists():
                model_files.append(
                    {
                        "path": str(source_sidecar),
                        "sha256": self._sha256_file(source_sidecar),
                        "size_bytes": source_sidecar.stat().st_size,
                    }
                )
        return {
            "run_id": self._run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git": {"commit": commit, "dirty": dirty},
            "platform": platform.platform(),
            "python": sys.version,
            "packages": package_versions,
            "experiment_config": self._experiment_config,
            "experiment_config_sha256": hashlib.sha256(config_json).hexdigest(),
            "random_seed": int(
                self._experiment_config.get("online_adaptation", {})
                .get("neuroonline", {})
                .get("random_seed", 42)
            ),
            "model_name": self._model_name,
            "initial_model_files": model_files,
            "native_amplifier_recording": {
                "required": True,
                "recording_id": native_recording_id or None,
                "declared": bool(native_recording_id),
                "note": "Preserve the Neuracle native BDF/NDF file and trigger channel with this run ID.",
            },
        }

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _emit_status(self, result: PredictionResult, game_command: str | None) -> None:
        if self._status_callback is None:
            return

        payload = {
            "prediction": result.label,
            "confidence": result.confidence,
            "uncertainty": result.uncertainty,
            "class_id": result.class_id,
            "mapped_command": game_command or "STOP",
            "last_transport_command": self._last_game_transport_command,
            "last_send_success": self._last_game_transport_error is None and self._last_game_transport_sent_at > 0.0,
            "last_send_error": self._last_game_transport_error,
            "model_revision": getattr(self, "_model_revision", 0),
            "timing_alignment": getattr(
                getattr(self, "_acquirer", None),
                "timing_diagnostics",
                {},
            ),
            "updated_at": time.time(),
        }
        adaptation_status = self._online_adaptation_status()
        if adaptation_status is not None:
            payload["online_adaptation"] = adaptation_status
        label_source_status = self._online_label_source_status()
        if label_source_status is not None:
            payload["online_label_source"] = label_source_status
        try:
            self._status_callback(payload)
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Failed to publish realtime decoder status: %s", exc)

    def _online_adaptation_status(self) -> dict[str, Any] | None:
        neuroonline_adapter = getattr(self, "_neuroonline_adapter", None)
        if neuroonline_adapter is not None:
            return neuroonline_adapter.status()
        batch_adapter = getattr(self, "_batch_adapter", None)
        if batch_adapter is not None:
            return batch_adapter.status()
        return None

    def _online_label_source_status(self) -> dict[str, Any] | None:
        source = getattr(self, "_online_label_source", None)
        status = getattr(source, "status", None)
        if not callable(status):
            return None
        try:
            payload = status()
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Failed to read online label source status: %s", exc)
            return None
        if not isinstance(payload, dict):
            return None
        result = dict(payload)
        if result.get("source") == "cued-protocol":
            label_value = result.get("label_id")
            result["scene_synced"] = (
                int(result.get("scene_index", -1))
                == getattr(self, "_scene_sent_scene_index", -1)
                and label_value is not None
                and int(label_value) == getattr(self, "_scene_sent_label_id", None)
            )
            result["scene_sync_error"] = getattr(self, "_scene_sync_error", None)
        return result

    def _online_label_source_metadata(self) -> dict[str, Any] | None:
        source = getattr(self, "_online_label_source", None)
        metadata = getattr(source, "metadata", None)
        if callable(metadata):
            try:
                payload = metadata()
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("Failed to read online label source metadata: %s", exc)
            else:
                if isinstance(payload, dict):
                    return dict(payload)
        return self._online_label_source_status()
