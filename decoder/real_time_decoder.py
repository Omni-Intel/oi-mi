"""Background realtime motor imagery decoding loop."""

from __future__ import annotations

from collections.abc import Callable
import copy
from contextlib import nullcontext
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

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
from utils.markers import ArTcpCommandSender, LSLCommandOutlet, MarkerBackend
from utils.online_labels import OnlineLabelSource
from utils.preprocessing import filter_and_transform
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


class RealTimeDecoder:
    """Continuously decode sliding EEG windows on a background thread."""

    def __init__(
        self,
        acquirer: AbstractAcquirer,
        model: BaseModelAdapter,
        console: Console,
        command_outlet: LSLCommandOutlet,
        game_command_outlet: ArTcpCommandSender | None,
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
        self._online_update_enabled = bool(online_update_enabled)
        self._online_update_learning_rate = float(online_update_learning_rate)
        self._online_update_every = max(int(online_update_every), 1)
        self._model_save_path = model_save_path
        self._online_label_source = online_label_source
        self._status_callback = status_callback
        self._online_update_count = 0
        self._online_seen_labeled_windows = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_game_command: str | None = None
        self._last_game_transport_command: str | None = None
        self._last_game_transport_error: str | None = None
        self._last_game_transport_sent_at = 0.0
        self._game_command_keepalive_sec = max(0.2, min(0.5, step_sec * 1.1))
        self._game_session_started = False
        self._game_disconnect_message: str | None = None
        self._stop_on_game_disconnect = bool(stop_on_game_disconnect)
        self._thread_context = thread_context
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
        self._game_session_started = False
        self._game_disconnect_message = None
        if record and subject_id:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._save_dir = save_dir or Path("records_storage") / subject_id / "realtime" / timestamp
            self._writer = StreamWriter(self._save_dir)
            self._writer.start({
                "subject_id": subject_id,
                "mode": "realtime",
                "start_time": time.time(),
                "sfreq": self._sfreq,
                "window_sec": self._window_sec,
                "step_sec": self._step_sec,
                "channels": self._acquirer.metadata.n_channels,
                "model_revision": self._model_revision,
                "online_adaptation": self._online_adaptation_status(),
                "online_label_source": self._online_label_source_metadata(),
            })

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
            if heartbeat is not None:
                heartbeat()
            if hasattr(self, "_writer"):
                self._writer.stop()
                self._writer.update_manifest(
                    {
                        "model_revision": self._model_revision,
                        "online_adaptation": self._online_adaptation_status(),
                        "online_label_source": self._online_label_source_metadata(),
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
            "subject_id": subject_id,
            "mode": "test_mode",
            "start_time": time.time(),
            "sfreq": self._sfreq,
            "window_sec": self._window_sec,
            "step_sec": self._step_sec,
            "channels": self._acquirer.metadata.n_channels,
        })
        
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
                        window, _ = self._acquirer.get_chunk(self._window_sec)
                    except RuntimeError:
                        time.sleep(self._step_sec)
                        continue
                    processed = filter_and_transform(window, sfreq=self._sfreq)
                    probabilities = self._predict_proba(
                        processed[None, ...],
                        mc_dropout_passes=self._mc_dropout_passes,
                    )[0]
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
                    writer.put(
                        window=window.astype(np.float32),
                        y_true=label,
                        y_pred=pred_class,
                        confidence=float(result.confidence)
                    )

                    update_metrics = self._maybe_update_model(
                        processed=processed,
                        true_label=label,
                    )
                    if update_metrics:
                        loss = update_metrics.get("loss")
                        if loss is not None:
                            update_losses.append(float(loss))
                    
                    true_labels.append(label)
                    pred_labels.append(pred_class)
                    confidences.append(float(result.confidence))
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
            writer.stop()
            if run_status == "failed":
                writer.update_manifest({"status": run_status, "error": run_error})
            if heartbeat is not None:
                heartbeat()

        if not true_labels:
            writer.update_manifest({"status": "no_windows", "error": "No EEG windows were collected."})
            raise RuntimeError("Test mode did not collect any EEG windows.")

        y_true = np.asarray(true_labels, dtype=np.int64)
        y_pred = np.asarray(pred_labels, dtype=np.int64)
        pred_valid = y_pred >= 0
        accuracy = float(np.mean(y_pred == y_true))
        valid_accuracy = float(np.mean(y_pred[pred_valid] == y_true[pred_valid])) if np.any(pred_valid) else 0.0
        
        writer.update_manifest({
            "status": run_status,
            "accuracy": accuracy,
            "valid_accuracy": valid_accuracy,
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
        }

    def _decode_loop(self) -> None:
        while not self._stop_event.is_set():
            started_at = time.perf_counter()
            try:
                try:
                    window, _ = self._acquirer.get_chunk(self._window_sec)
                except RuntimeError as exc:
                    if "Not enough data" in str(exc):
                        self._sleep_with_heartbeat(self._step_sec, None)
                        continue
                    raise
                window_end = time.monotonic()
                window_start = window_end - self._window_sec
                processed = filter_and_transform(window, sfreq=self._sfreq)
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
                game_command = self._gate_game_command_for_protocol(game_command)
                self._push_game_command(game_command)
                self._emit_status(result, game_command)

                online_label = self._get_online_label(
                    window_start=window_start,
                    window_end=window_end,
                )
                if online_label is not None:
                    self._handle_online_label(
                        processed=processed,
                        probabilities=probabilities,
                        online_label=online_label,
                        window_end=window_end,
                    )
                
                if hasattr(self, "_record") and self._record and hasattr(self, "_writer"):
                    pred_class = -1 if result.class_id is None else int(result.class_id)
                    self._writer.put(
                        window=window.astype(np.float32),
                        y_true=-1 if online_label is None else int(online_label.label_id),
                        y_pred=pred_class,
                        confidence=float(result.confidence),
                        raw_pred=raw_prediction,
                        model_revision=model_revision,
                        label_event_id="" if online_label is None else str(online_label.event_id),
                    )
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Realtime decoding failed")
                self._console.print(f"[red]解码失败：{exc}[/red]")

            elapsed = time.perf_counter() - started_at
            sleep_time = max(0.0, self._step_sec - elapsed)
            self._sleep_with_heartbeat(sleep_time, None)

    def _handle_online_label(
        self,
        *,
        processed: np.ndarray,
        probabilities: np.ndarray,
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
            return self._online_label_source.get_label(
                window_start=window_start,
                window_end=window_end,
            )
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
        with self._model_lock:
            if not isinstance(self._model, NeuroOnlineModelAdapter):
                raise RuntimeError("NeuroOnline model adapter is not active")
            candidate = copy.deepcopy(self._model)
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
        self._persist_online_adaptation_status()

    def _save_current_model(self) -> None:
        if self._model_save_path is None:
            return
        with self._model_lock:
            self._model.save(self._model_save_path)

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

    def _gate_game_command_for_protocol(self, command: str | None) -> str | None:
        """Keep the car stopped outside an automatic cue protocol's control phase."""

        source_status = self._online_label_source_status()
        if not source_status or source_status.get("source") != "cued-protocol":
            return command
        phase = str(source_status.get("phase", "preparing"))
        if phase == "done":
            self._stop_event.set()
        return command if phase == "control" else "STOP"

    def _push_game_command(self, command: str | None) -> None:
        if self._game_command_outlet is None:
            return

        if command is None:
            if self._last_game_command is None:
                self._push_game_keepalive()
                return
            self._push_game_transport_command("STOP")
            self._last_game_command = None
            return

        self._push_game_session_command("START")
        now = time.monotonic()
        if (
            command == self._last_game_command
            and now - self._last_game_transport_sent_at < self._game_command_keepalive_sec
        ):
            return

        self._push_game_transport_command(command)
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
        if now - self._last_game_transport_sent_at < self._game_command_keepalive_sec:
            return
        self._push_game_transport_command("STOP")

    def _push_game_transport_command(self, command: str) -> bool:
        if self._game_command_outlet is None:
            return False
        try:
            self._game_command_outlet.push(command)
            self._last_game_transport_command = command
            self._last_game_transport_error = None
            self._last_game_transport_sent_at = time.monotonic()
            return True
        except Exception as exc:  # noqa: BLE001
            self._last_game_transport_error = str(exc)
            LOGGER.warning("Failed to push AR game command '%s': %s", command, exc)
            if self._stop_on_game_disconnect:
                self._game_disconnect_message = f"Unity game connection lost: {exc}"
                self._stop_event.set()
            return False

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
        return dict(payload) if isinstance(payload, dict) else None

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
