"""Focused tests for the paper-faithful NeuroOnline integration."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import threading
import time

import numpy as np
import torch
from torch import nn
from click.testing import CliRunner

from adaptation.neuroonline import (
    NeuroOnlineConfig,
    NeuroOnlineModelAdapter,
    NeuroOnlineStreamAdapter,
)
from models.factory import TorchModelAdapter
from cli import cli


class _TinyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(4, 3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs).squeeze(-1))


class NeuroOnlineTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        np.random.seed(7)

    def test_config_selects_neuroonline_without_changing_periodic_default(self) -> None:
        selected = NeuroOnlineConfig.from_mapping(
            {
                "enabled": True,
                "strategy": "neuroonline",
                "neuroonline": {"history_threshold": 8, "update_stride": 2},
            }
        )
        periodic = NeuroOnlineConfig.from_mapping({"enabled": True})
        self.assertTrue(selected.enabled)
        self.assertEqual(selected.history_threshold, 8)
        self.assertEqual(selected.update_stride, 2)
        self.assertFalse(periodic.enabled)

    def test_identity_modulation_preserves_predictions_before_online_update(self) -> None:
        base = TorchModelAdapter("tiny", _TinyDecoder())
        inputs = np.random.randn(5, 2, 16).astype(np.float32)
        expected = base.predict_proba(inputs)
        wrapped = NeuroOnlineModelAdapter(base, config=NeuroOnlineConfig(enabled=True))
        actual = wrapped.predict_proba(inputs)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)

    def test_full_neuroonline_objective_updates_backbone_and_persists_crm(self) -> None:
        config = NeuroOnlineConfig(
            enabled=True,
            learning_rate=1e-3,
            update_batch_size=4,
            epochs=1,
            history_threshold=4,
            update_stride=2,
            recent_samples=4,
            prompt_count=4,
        )
        base = TorchModelAdapter("tiny", _TinyDecoder())
        wrapped = NeuroOnlineModelAdapter(base, config=config)
        original = np.random.randn(8, 2, 16).astype(np.float32)
        time_view = original.copy()
        time_view[..., ::2] = 0.0
        frequency_view = original * 0.5
        labels = np.arange(8, dtype=np.int64) % 3
        before = base.model.features[0].weight.detach().clone()
        metrics = wrapped.neuroonline_update(original, time_view, frequency_view, labels)
        after = base.model.features[0].weight.detach().clone()

        self.assertEqual(metrics["updated"], 8.0)
        self.assertGreater(metrics["loss"], 0.0)
        self.assertIn("gate_alpha", metrics)
        self.assertIn("gate_beta", metrics)
        self.assertFalse(torch.equal(before, after))
        self.assertIsNotNone(wrapped._modulator)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "tiny.pt"
            wrapped.save(path)
            self.assertTrue(path.exists())
            self.assertTrue(Path(f"{path}.neuroonline.pt").exists())

    def test_offline_fit_initializes_and_trains_complete_neuroonline_checkpoint(self) -> None:
        config = NeuroOnlineConfig(
            enabled=True,
            prompt_count=4,
            offline_epochs=2,
            offline_batch_size=6,
            offline_learning_rate=1e-3,
        )
        wrapped = NeuroOnlineModelAdapter(TorchModelAdapter("tiny", _TinyDecoder()), config=config)
        inputs = np.random.randn(30, 2, 16).astype(np.float32)
        labels = np.tile(np.arange(3, dtype=np.int64), 10)
        metrics = wrapped.fit(
            inputs,
            labels,
            epochs=1,
            batch_size=2,
            learning_rate=1e-5,
            patience=1,
        )
        self.assertIsNotNone(wrapped._modulator)
        self.assertIn("val_kappa", metrics)
        self.assertGreaterEqual(metrics["val_acc"], 0.0)

    def test_train_from_records_restores_neuroonline_main_and_crm_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            records = root / "records" / "S001" / "calibration" / "20260722_120000"
            records.mkdir(parents=True)
            inputs = np.random.randn(12, 2, 500).astype(np.float32)
            labels = np.tile(np.arange(3, dtype=np.int64), 4)
            np.savez_compressed(
                records / "training_windows_main.npz",
                raw_windows=inputs,
                processed_windows=inputs,
                labels=labels,
            )
            config_path = root / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "subject_id: S001",
                        "model_name: shallowconvnet",
                        "device_type: neuracle",
                        "hardware_dummy_mode: false",
                        "sfreq: 250",
                        "n_classes: 3",
                        "window_sec: 2.0",
                        "step_sec: 0.5",
                        "new_subject_epochs: 1",
                        "old_subject_epochs: 1",
                        "batch_size: 4",
                        "learning_rate: 0.001",
                        "early_stopping_patience: 1",
                        "storage:",
                        f"  models_dir: {str(root / 'models')!r}",
                        f"  records_dir: {str(root / 'records')!r}",
                        "online_adaptation:",
                        "  enabled: true",
                        "  strategy: neuroonline",
                        "  neuroonline:",
                        "    prompt_count: 4",
                        "    offline_epochs: 1",
                        "    offline_batch_size: 4",
                        "    offline_learning_rate: 0.001",
                    ]
                ),
                encoding="utf-8",
            )

            result = CliRunner().invoke(
                cli,
                [
                    "--config",
                    str(config_path),
                    "train-from-records",
                    "--subject",
                    "S001",
                    "--model",
                    "shallowconvnet",
                    "--device",
                    "neuracle",
                    "--session",
                    "20260722_120000",
                ],
            )

            self.assertEqual(result.exit_code, 0, f"{result.output}\n{result.exception!r}")
            model_path = root / "models" / "S001" / "neuracle" / "shallowconvnet.pt"
            self.assertTrue(model_path.exists())
            self.assertTrue(Path(f"{model_path}.neuroonline.pt").exists())
            self.assertIn("CRM 已保存", result.output)

    def test_stream_updates_at_threshold_then_stride(self) -> None:
        calls: list[int] = []

        def update(
            original: np.ndarray,
            time_view: np.ndarray,
            frequency_view: np.ndarray,
            labels: np.ndarray,
        ) -> dict[str, float]:
            self.assertEqual(original.shape, time_view.shape)
            self.assertEqual(original.shape, frequency_view.shape)
            self.assertEqual(original.shape[0], labels.shape[0])
            calls.append(original.shape[0])
            return {"updated": float(original.shape[0]), "loss": 1.0}

        adapter = NeuroOnlineStreamAdapter(
            config=NeuroOnlineConfig(
                enabled=True,
                history_threshold=4,
                update_stride=2,
                recent_samples=4,
                prompt_count=4,
            ),
            update_callback=update,
        )
        for index in range(7):
            label = index % 3
            adapter.add_window(
                np.full((2, 16), index, dtype=np.float32),
                label,
                predicted_label=label,
            )

        self.assertTrue(adapter.wait_for_idle(timeout_sec=2.0))
        self.assertEqual(calls, [4, 4])
        status = adapter.status()
        self.assertEqual(status["seen_labeled_windows"], 7)
        self.assertEqual(status["update_count"], 2)
        self.assertEqual(status["samples_until_update"], 1)
        self.assertEqual(status["prequential"]["evaluated_windows"], 7)
        self.assertEqual(status["prequential"]["balanced_accuracy"], 1.0)
        self.assertEqual(status["prequential"]["confusion_matrix"], [[3, 0, 0], [0, 2, 0], [0, 0, 2]])
        self.assertEqual(len(status["update_history"]), 2)
        self.assertAlmostEqual(status["progress"], 0.5)

    def test_stream_update_runs_in_background(self) -> None:
        update_started = threading.Event()
        allow_update = threading.Event()

        def update(*_arrays: np.ndarray) -> dict[str, float]:
            update_started.set()
            allow_update.wait(timeout=2.0)
            return {"updated": 4.0, "loss": 1.0}

        adapter = NeuroOnlineStreamAdapter(
            config=NeuroOnlineConfig(
                enabled=True,
                history_threshold=4,
                update_stride=2,
                recent_samples=4,
            ),
            update_callback=update,
        )
        started_at = time.perf_counter()
        for index in range(4):
            adapter.add_window(np.zeros((2, 16), dtype=np.float32), index % 3)

        self.assertTrue(update_started.wait(timeout=1.0))
        self.assertLess(time.perf_counter() - started_at, 0.5)
        self.assertTrue(adapter.status()["training_in_background"])
        allow_update.set()
        self.assertTrue(adapter.wait_for_idle(timeout_sec=2.0))
        self.assertEqual(adapter.status()["update_count"], 1)


if __name__ == "__main__":
    unittest.main()
