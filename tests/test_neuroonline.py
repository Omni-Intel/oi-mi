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

from adaptation.calibration_search import (
    CalibrationSearchConfig,
    run_calibration_search,
)
from adaptation.neuroonline import (
    NeuroOnlineConfig,
    NeuroOnlineModelAdapter,
    NeuroOnlineStreamAdapter,
    _frequency_mask,
    _time_mask,
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
    def test_formal_defaults_update_from_latest_64_primary_windows(self) -> None:
        config = NeuroOnlineConfig.from_mapping(
            {"enabled": True, "strategy": "neuroonline"}
        )
        self.assertEqual(config.history_threshold, 64)
        self.assertEqual(config.update_stride, 64)
        self.assertEqual(config.recent_samples, 64)

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

    def test_mask_ratio_uses_one_contiguous_time_and_frequency_region(self) -> None:
        inputs = torch.ones(3, 2, 20)
        time_masked = _time_mask(
            inputs,
            0.25,
            torch.Generator().manual_seed(11),
        )
        for sample in time_masked:
            masked_positions = torch.where(sample[0] == 0)[0]
            self.assertEqual(masked_positions.numel(), 5)
            self.assertTrue(torch.equal(masked_positions, torch.arange(
                masked_positions[0],
                masked_positions[0] + 5,
            )))
            self.assertTrue(torch.equal(sample[0] == 0, sample[1] == 0))

        signal = torch.randn(3, 2, 20)
        frequency_masked = _frequency_mask(
            signal,
            0.25,
            torch.Generator().manual_seed(11),
        )
        masked_spectrum = torch.fft.rfft(frequency_masked, dim=-1)
        original_spectrum = torch.fft.rfft(signal, dim=-1)
        expected_bins = round(original_spectrum.shape[-1] * 0.25)
        for sample_index in range(signal.shape[0]):
            removed = torch.isclose(
                masked_spectrum[sample_index, 0],
                torch.zeros_like(masked_spectrum[sample_index, 0]),
                atol=1e-5,
            )
            self.assertGreaterEqual(int(removed.sum()), expected_bins)
            self.assertTrue(torch.equal(
                removed,
                torch.isclose(
                    masked_spectrum[sample_index, 1],
                    torch.zeros_like(masked_spectrum[sample_index, 1]),
                    atol=1e-5,
                ),
            ))

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
        self.assertIn("val_balanced_accuracy", metrics)
        self.assertGreaterEqual(metrics["val_acc"], 0.0)

    def test_calibration_search_keeps_trial_holdout_and_saves_report(self) -> None:
        inputs = np.random.randn(30, 2, 16).astype(np.float32)
        labels = np.tile(np.arange(3, dtype=np.int64), 10)
        trial_ids = np.arange(labels.size, dtype=np.int64)
        base_config = NeuroOnlineConfig(
            enabled=True,
            prompt_count=4,
            offline_epochs=2,
            offline_patience=1,
            offline_batch_size=6,
            offline_learning_rate=1e-3,
        )
        search_config = CalibrationSearchConfig(
            enabled=True,
            selection_epochs=1,
            selection_patience=1,
            learning_rates=(1e-3,),
            batch_sizes=(6,),
            mask_ratios=(0.3,),
            consistency_weights=(0.1,),
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_calibration_search(
                base_template=TorchModelAdapter("tiny", _TinyDecoder()),
                base_config=base_config,
                search_config=search_config,
                X=inputs,
                y=labels,
                groups=trial_ids,
                session_dir=Path(tmp_dir),
            )

            self.assertTrue((Path(tmp_dir) / "hyperparameter_search.json").exists())
            self.assertEqual(result.best_config.offline_epochs, 2)
            self.assertEqual(len(result.report["candidates"]), 1)
            split = result.report["split"]
            self.assertEqual(
                split["train_trials"]
                + split["selection_validation_trials"]
                + split["untouched_holdout_trials"],
                30,
            )
            self.assertIn(
                "balanced_accuracy",
                result.report["untouched_holdout_metrics"],
            )

    def test_checkpoint_restores_selected_model_coupled_parameters(self) -> None:
        selected = NeuroOnlineConfig(
            enabled=True,
            prompt_count=4,
            mask_ratio=0.5,
            consistency_weight=0.3,
        )
        wrapped = NeuroOnlineModelAdapter(
            TorchModelAdapter("tiny", _TinyDecoder()),
            config=selected,
        )
        inputs = np.random.randn(2, 2, 16).astype(np.float32)
        wrapped.predict_proba(inputs)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "tiny.pt"
            wrapped.save(model_path)
            restored = NeuroOnlineModelAdapter(
                TorchModelAdapter("tiny", _TinyDecoder()),
                config=NeuroOnlineConfig(
                    enabled=True,
                    prompt_count=4,
                    mask_ratio=0.3,
                    consistency_weight=0.1,
                ),
                state_path=model_path,
            )
            self.assertEqual(restored.config.mask_ratio, 0.5)
            self.assertEqual(restored.config.consistency_weight, 0.3)

    def test_train_from_records_restores_neuroonline_main_and_crm_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            records = root / "records" / "S001" / "calibration" / "20260722_120000"
            records.mkdir(parents=True)
            inputs = np.random.randn(12, 2, 400).astype(np.float32)
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
                        "sfreq: 200",
                        "n_classes: 3",
                        "window_sec: 2.0",
                        "step_sec: 0.5",
                        "calibration_epochs: 1",
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
            self.assertTrue(model_path.with_suffix(".metrics.yaml").exists())
            self.assertIn("CRM 已保存", result.output)
            self.assertIn("训练指标已保存", result.output)

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
        first_update = status["update_history"][0]
        self.assertEqual(first_update["trigger_seen_labeled_windows"], 4)
        self.assertEqual(first_update["snapshot_first_window_id"], 1)
        self.assertEqual(first_update["snapshot_last_window_id"], 4)
        self.assertEqual(first_update["snapshot_samples"], 4)
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

    def test_stream_rejects_duplicate_event_and_non_increasing_timestamp(self) -> None:
        adapter = NeuroOnlineStreamAdapter(
            config=NeuroOnlineConfig(
                enabled=True,
                history_threshold=64,
                update_stride=64,
                recent_samples=64,
            ),
            update_callback=lambda *_arrays: {"updated": 1.0},
        )
        window = np.zeros((2, 16), dtype=np.float32)
        self.assertTrue(
            adapter.add_window(
                window,
                0,
                event_id="scene-000001-primary",
                window_end_monotonic=10.0,
            )
        )
        self.assertFalse(
            adapter.add_window(
                window,
                0,
                event_id="scene-000001-primary",
                window_end_monotonic=10.5,
            )
        )
        self.assertFalse(
            adapter.add_window(
                window,
                1,
                event_id="scene-000002-primary",
                window_end_monotonic=9.5,
            )
        )
        status = adapter.status()
        self.assertEqual(status["seen_labeled_windows"], 1)
        self.assertEqual(status["duplicate_windows_rejected"], 1)
        self.assertEqual(status["stale_windows_rejected"], 1)


if __name__ == "__main__":
    unittest.main()
