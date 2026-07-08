"""Background realtime motor imagery decoding loop."""

from __future__ import annotations

from collections.abc import Callable
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
from models.factory import BaseModelAdapter
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
        thread_context: Any | None = None,
    ) -> None:
        self._acquirer = acquirer
        self._model = model
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
        self._online_update_count = 0
        self._online_seen_labeled_windows = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_game_command: str | None = None
        self._last_game_transport_command: str | None = None
        self._last_game_transport_sent_at = 0.0
        self._game_command_keepalive_sec = max(0.2, min(0.5, step_sec * 1.1))
        self._game_session_started = False
        self._thread_context = thread_context

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
            self._game_command_outlet.close()

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
        self._last_game_transport_sent_at = 0.0
        self._game_session_started = False
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
            })

        self._push_game_session_command("START")
        self.start()
        try:
            while True:
                self._sleep_with_heartbeat(min(0.1, max(self._step_sec, 0.1)), heartbeat)
                if heartbeat is not None:
                    heartbeat()
        except KeyboardInterrupt:
            self._console.print("\n[bold red]停止实时解码[/bold red]")
        finally:
            self.stop()
            if heartbeat is not None:
                heartbeat()
            if hasattr(self, "_writer"):
                self._writer.stop()
                self._writer.update_manifest({})
                self._console.print(f"[bold green]实时数据已保存[/bold green] {self._save_dir}")

    def run_test_mode(
        self,
        *,
        subject_id: str,
        marker_backend: MarkerBackend,
        duration_sec: int,
        block_sec: float = 10.0,
        save_dir: Path | None = None,
        heartbeat: Callable[[], None] | None = None,
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
        
        self._acquirer.start_stream()
        if heartbeat is not None:
            heartbeat()
        started = time.monotonic()
        cue_index = 0
        labels = [0, 1, 2]
        collected_windows: list[np.ndarray] = []
        true_labels: list[int] = []
        pred_labels: list[int] = []
        confidences: list[float] = []
        update_losses: list[float] = []
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
                self._sleep_with_heartbeat(self._window_sec, heartbeat)
                
                # Now we predict on the new block length
                # Since we already waited window_sec, we subtract this from the block duration to keep blocks same length
                block_end = time.monotonic() + max(0.1, block_sec - self._window_sec)
                
                while time.monotonic() < block_end and time.monotonic() - started < duration_sec:
                    loop_started = time.perf_counter()
                    try:
                        window, _ = self._acquirer.get_chunk(self._window_sec)
                    except RuntimeError:
                        time.sleep(self._step_sec)
                        continue
                    processed = filter_and_transform(window, sfreq=self._sfreq)
                    probabilities = self._model.predict_proba(
                        processed[None, ...],
                        mc_dropout_passes=self._mc_dropout_passes,
                    )[0]
                    result = self._post_process(probabilities)
                    self._console.print(
                        f"[green][预测][/green] {result.label} "
                        f"(confidence: {result.confidence:.2f}, uncertainty: {result.uncertainty:.2f})"
                    )
                    self._command_outlet.push(result.label)
                    self._push_game_command(self._to_game_command(result))
                    
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
                    elapsed = time.perf_counter() - loop_started
                    self._sleep_with_heartbeat(max(0.0, self._step_sec - elapsed), heartbeat)
        except KeyboardInterrupt:
            self._console.print("\n[bold red]停止测试模式[/bold red]")
        finally:
            self._acquirer.stop_stream()
            if heartbeat is not None:
                heartbeat()

        if not true_labels:
            raise RuntimeError("Test mode did not collect any EEG windows.")

        y_true = np.asarray(true_labels, dtype=np.int64)
        y_pred = np.asarray(pred_labels, dtype=np.int64)
        pred_valid = y_pred >= 0
        accuracy = float(np.mean(y_pred == y_true))
        valid_accuracy = float(np.mean(y_pred[pred_valid] == y_true[pred_valid])) if np.any(pred_valid) else 0.0
        
        writer.stop()
        writer.update_manifest({
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
                window, _ = self._acquirer.get_chunk(self._window_sec)
                window_end = time.monotonic()
                window_start = window_end - self._window_sec
                processed = filter_and_transform(window, sfreq=self._sfreq)
                probabilities = self._model.predict_proba(
                    processed[None, ...],
                    mc_dropout_passes=self._mc_dropout_passes,
                )[0]
                result = self._post_process(probabilities)
                self._console.print(
                    f"[green][预测][/green] {result.label} "
                    f"(confidence: {result.confidence:.2f}, uncertainty: {result.uncertainty:.2f})"
                )
                self._command_outlet.push(result.label)
                self._push_game_command(self._to_game_command(result))

                online_label = self._get_online_label(
                    window_start=window_start,
                    window_end=window_end,
                )
                if online_label is not None:
                    self._maybe_update_model(
                        processed=processed,
                        true_label=online_label.label_id,
                    )
                
                if hasattr(self, "_record") and self._record and hasattr(self, "_writer"):
                    pred_class = -1 if result.class_id is None else int(result.class_id)
                    self._writer.put(
                        window=window.astype(np.float32),
                        y_true=-1 if online_label is None else int(online_label.label_id),
                        y_pred=pred_class,
                        confidence=float(result.confidence)
                    )
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Realtime decoding failed")
                self._console.print(f"[red]解码失败：{exc}[/red]")

            elapsed = time.perf_counter() - started_at
            sleep_time = max(0.0, self._step_sec - elapsed)
            self._sleep_with_heartbeat(sleep_time, None)

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

    @staticmethod
    def _sleep_with_heartbeat(duration_sec: float, heartbeat: Callable[[], None] | None) -> None:
        remaining = max(float(duration_sec), 0.0)
        while remaining > 0:
            chunk = min(0.1, remaining)
            time.sleep(chunk)
            remaining -= chunk
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

    def _push_game_command(self, command: str | None) -> None:
        if self._game_command_outlet is None:
            return

        if command is None:
            if self._last_game_command is None:
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

    def _push_game_transport_command(self, command: str) -> bool:
        if self._game_command_outlet is None:
            return False
        try:
            self._game_command_outlet.push(command)
            self._last_game_transport_command = command
            self._last_game_transport_sent_at = time.monotonic()
            return True
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to push AR game command '%s': %s", command, exc)
            return False
