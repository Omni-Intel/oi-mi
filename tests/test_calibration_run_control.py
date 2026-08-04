from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from adaptation.calibrator import CalibrationRunControl, Calibrator
from adaptation.mi_protocol import ProtocolConfig, build_session_plan
from utils.markers import PROTOCOL_EVENT_CODES


def test_protocol_enables_open_ended_collection_with_original_plan_as_minimum() -> None:
    protocol = ProtocolConfig.from_config(
        {
            "protocol": {
                "calibration_blocks": 4,
                "calibration_trials_per_class_per_block": 5,
                "continuous_collection": True,
            }
        }
    )

    assert protocol.continuous_collection is True
    assert protocol.minimum_calibration_trials == 60


def test_pause_and_resume_requests_are_thread_safe_state_transitions() -> None:
    control = CalibrationRunControl(minimum_trials=6)

    assert control.request_pause() is True
    assert control.snapshot()["state"] == "pause_pending"
    control.mark_paused(True)
    assert control.snapshot()["state"] == "paused"
    assert control.request_resume() is True
    control.mark_paused(False)
    assert control.snapshot()["state"] == "collecting"


def test_completion_requires_minimum_trials_and_balanced_class_counts() -> None:
    control = CalibrationRunControl(minimum_trials=6)
    for label in ("left", "right", "idle", "left", "right"):
        control.mark_trial_completed(label)

    assert control.request_stop() is False
    control.mark_trial_completed("idle")
    assert control.request_stop() is True
    assert control.should_finish() is True
    assert control.snapshot()["state"] == "stop_pending"


def test_early_stop_request_waits_until_class_counts_rebalance() -> None:
    control = CalibrationRunControl(minimum_trials=3)
    for label in ("left", "right", "idle", "left"):
        control.mark_trial_completed(label)

    assert control.request_stop() is True
    assert control.should_finish() is False
    control.mark_trial_completed("right")
    assert control.should_finish() is False
    control.mark_trial_completed("idle")
    assert control.should_finish() is True


def test_pause_events_have_distinct_protocol_codes() -> None:
    pause_codes = {
        PROTOCOL_EVENT_CODES["operator_pause_start"],
        PROTOCOL_EVENT_CODES["operator_pause_end"],
    }
    assert len(pause_codes) == 2
    assert not pause_codes.intersection(
        {
            PROTOCOL_EVENT_CODES["control_on"],
            PROTOCOL_EVENT_CODES["control_off"],
        }
    )


def test_continuous_blocks_stop_with_balanced_completed_trials() -> None:
    protocol = ProtocolConfig.from_config(
        {
            "protocol": {
                "calibration_blocks": 1,
                "calibration_trials_per_class_per_block": 2,
                "continuous_collection": True,
                "minimum_calibration_trials": 6,
                "rest_between_blocks_sec": 0.0,
                "random_seed": 9,
            }
        }
    )

    class AutoStopControl(CalibrationRunControl):
        def mark_trial_completed(self, label: str) -> None:
            super().mark_trial_completed(label)
            if self.snapshot()["completed_trials"] >= self.minimum_trials:
                self.request_stop()

    control = AutoStopControl(minimum_trials=6)
    calibrator = Calibrator.__new__(Calibrator)
    calibrator._protocol = protocol
    calibrator._console = SimpleNamespace(print=lambda *args, **kwargs: None)
    events: list[str] = []
    calibrator._emit_event = lambda recorder, name, **payload: events.append(name)
    calibrator._run_trial = lambda **kwargs: {
        "label": kwargs["label"],
        "label_id": {"left": 0, "right": 1, "idle": 2}[kwargs["label"]],
    }
    trials: list[dict] = []

    calibrator._run_continuous_formal_blocks(
        build_session_plan(protocol),
        recorder=object(),
        heartbeat=None,
        trials=trials,
        run_control=control,
    )

    assert len(trials) == 6
    assert control.snapshot()["class_counts"] == {"left": 2, "right": 2, "idle": 2}
    assert events == ["block_start", "block_end"]


def test_operator_pause_drains_recording_and_emits_paired_events() -> None:
    control = CalibrationRunControl(minimum_trials=3)
    control.request_pause()
    calibrator = Calibrator.__new__(Calibrator)
    calibrator._console = SimpleNamespace(print=lambda *args, **kwargs: None)
    events: list[str] = []
    pulls: list[bool] = []
    calibrator._emit_event = lambda recorder, name, **payload: events.append(name)
    calibrator._flush_recorder = lambda recorder: pulls.append(True)
    calibrator._update_stage_progress = lambda **kwargs: None

    resume = threading.Thread(
        target=lambda: (time.sleep(0.08), control.request_resume()),
        daemon=True,
    )
    resume.start()
    should_continue = calibrator._wait_for_operator(
        recorder=object(),
        heartbeat=None,
        run_control=control,
    )
    resume.join(timeout=1.0)

    assert should_continue is True
    assert pulls
    assert events == ["operator_pause_start", "operator_pause_end"]
    assert control.snapshot()["state"] == "collecting"


def test_collection_only_saves_session_without_training_model(tmp_path: Path) -> None:
    protocol = ProtocolConfig.from_config({"protocol": {}})
    calibrator = Calibrator.__new__(Calibrator)
    calibrator._protocol = protocol
    messages: list[str] = []
    calibrator._console = SimpleNamespace(print=lambda message: messages.append(message))
    calibrator._model = SimpleNamespace(
        fit=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("collection-only mode must not train")
        ),
        save=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("collection-only mode must not save a model")
        ),
    )
    session_metadata: dict = {}
    processed = np.empty((12, 3, 400), dtype=np.float32)
    calibrator._collect_training_data = lambda **kwargs: (
        tmp_path,
        processed.copy(),
        processed,
        np.zeros(12, dtype=np.int64),
        np.arange(12, dtype=np.int64),
        session_metadata,
    )
    summaries: list[dict] = []
    calibrator._write_session_summary = lambda session_dir, **kwargs: summaries.append(
        {"session_dir": session_dir, **kwargs}
    )
    sealed: list[tuple[Path, bool]] = []
    calibrator._seal_session_bundle = (
        lambda session_dir, *, include_model_files=True: sealed.append(
            (session_dir, include_model_files)
        )
    )

    result = calibrator.calibrate(
        duration_sec=None,
        epochs=50,
        batch_size=16,
        learning_rate=1e-4,
        head_only=False,
        train_after_collection=False,
    )

    assert result.training_performed is False
    assert result.model_path is None
    assert result.metrics == {}
    assert result.windows_collected == 12
    assert session_metadata["training"] == {
        "performed": False,
        "reason": "collection_only",
    }
    assert summaries[0]["training_performed"] is False
    assert sealed == [(tmp_path, False)]
    assert any("未执行模型训练" in message for message in messages)
