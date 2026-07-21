"""Tests for periodic model adaptation and label-aware dummy signals."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from acquisition.dummy_acquirer import DummyAcquirer
from adaptation.online_batch_adapter import BatchAdaptationConfig, OnlineBatchAdapter
from utils.online_labels import SimulatedOnlineLabelSource


class _FakeAdaptiveModel:
    def __init__(self, *, trained: bool = False) -> None:
        self.trained = trained

    def predict_proba(self, X: np.ndarray, mc_dropout_passes: int = 1) -> np.ndarray:
        del mc_dropout_passes
        probabilities = np.full((X.shape[0], 3), 0.05, dtype=np.float32)
        if self.trained:
            predictions = np.rint(X[:, 0, 0]).astype(np.int64)
        else:
            predictions = np.zeros((X.shape[0],), dtype=np.int64)
        probabilities[np.arange(X.shape[0]), predictions] = 0.90
        return probabilities

    def update(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        learning_rate: float,
        epochs: int = 1,
        batch_size: int = 32,
    ) -> dict[str, float]:
        del y, learning_rate, epochs, batch_size
        self.trained = True
        return {"updated": float(X.shape[0]), "loss": 0.1}

    def save(self, path: Path) -> None:
        path.write_text("trained" if self.trained else "base", encoding="utf-8")


class OnlineBatchAdaptationTests(unittest.TestCase):
    def test_validated_candidate_is_saved_and_swapped(self) -> None:
        now = [0.0]
        current = [_FakeAdaptiveModel()]
        with tempfile.TemporaryDirectory() as tmp_dir:
            adapter = OnlineBatchAdapter(
                config=BatchAdaptationConfig(
                    enabled=True,
                    update_interval_sec=10.0,
                    epochs=1,
                    min_total_windows=12,
                    min_windows_per_class=3,
                    validation_ratio=0.5,
                    min_balanced_accuracy_gain=0.1,
                    max_class_accuracy_drop=0.0,
                    save_update_dataset=True,
                ),
                model_getter=lambda: copy.deepcopy(current[0]),
                model_swapper=lambda model: current.__setitem__(0, model),
                model_save_path=Path(tmp_dir) / "subject" / "shallowconvnet.pt",
                n_classes=3,
                clock=lambda: now[0],
            )
            for label in range(3):
                for event in range(2):
                    for _ in range(3):
                        adapter.add_window(
                            np.asarray([[float(label)]], dtype=np.float32),
                            label,
                            event_id=f"label-{label}-event-{event}",
                            now=now[0],
                        )
            self.assertTrue(adapter.maybe_start_update(now=10.0, force=True))
            adapter.close(timeout_sec=10.0)
            status = adapter.status()

            self.assertTrue(current[0].trained)
            self.assertEqual(status["model_version"], 1)
            self.assertTrue(status["last_result"]["accepted"])
            self.assertGreater(status["last_result"]["balanced_accuracy_gain"], 0.1)
            self.assertTrue((Path(tmp_dir) / "subject" / "shallowconvnet.pt").exists())
            self.assertTrue(
                (Path(tmp_dir) / "subject" / "shallowconvnet_updates" / "cycle_001_dataset.npz").exists()
            )

    def test_candidate_without_required_gain_is_rejected(self) -> None:
        current = [_FakeAdaptiveModel(trained=True)]
        original = current[0]
        with tempfile.TemporaryDirectory() as tmp_dir:
            adapter = OnlineBatchAdapter(
                config=BatchAdaptationConfig(
                    enabled=True,
                    update_interval_sec=10.0,
                    epochs=1,
                    min_total_windows=12,
                    min_windows_per_class=3,
                    validation_ratio=0.5,
                    min_balanced_accuracy_gain=0.1,
                    max_class_accuracy_drop=0.0,
                    save_update_dataset=False,
                ),
                model_getter=lambda: copy.deepcopy(current[0]),
                model_swapper=lambda model: current.__setitem__(0, model),
                model_save_path=Path(tmp_dir) / "subject" / "shallowconvnet.pt",
                n_classes=3,
                clock=lambda: 0.0,
            )
            for label in range(3):
                for event in range(2):
                    for _ in range(3):
                        adapter.add_window(
                            np.asarray([[float(label)]], dtype=np.float32),
                            label,
                            event_id=f"label-{label}-event-{event}",
                            now=0.0,
                        )
            self.assertTrue(adapter.maybe_start_update(now=10.0, force=True))
            adapter.close(timeout_sec=10.0)
            status = adapter.status()

            self.assertIs(current[0], original)
            self.assertEqual(status["model_version"], 0)
            self.assertFalse(status["last_result"]["accepted"])
            self.assertFalse((Path(tmp_dir) / "subject" / "shallowconvnet.pt").exists())

    def test_label_aware_dummy_changes_pattern_with_simulated_trials(self) -> None:
        now = [100.0]
        acquirer = DummyAcquirer(
            sfreq=250.0,
            n_channels=8,
            label_aware=True,
            noise_std_uv=0.0,
            drift_std_uv=0.0,
        )
        source = SimulatedOnlineLabelSource(
            acquirer,
            trial_sec=6.0,
            settle_sec=2.0,
            seed=17,
            clock=lambda: now[0],
        )
        now[0] = 102.1
        first = source.get_label(window_start=100.1, window_end=102.1)
        self.assertIsNotNone(first)
        first_block = acquirer._generate_block(500, start_index=0, dt=1.0 / 250.0)

        now[0] = 108.1
        second = source.get_label(window_start=106.1, window_end=108.1)
        self.assertIsNotNone(second)
        second_block = acquirer._generate_block(500, start_index=0, dt=1.0 / 250.0)

        self.assertNotEqual(first.label_id, second.label_id)
        self.assertNotEqual(first.event_id, second.event_id)
        self.assertFalse(np.allclose(first_block, second_block))


if __name__ == "__main__":
    unittest.main()
