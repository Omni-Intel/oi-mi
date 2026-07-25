"""Tests for sampling-rate-safe EEG preprocessing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from acquisition.base import AcquirerMetadata
from acquisition.brainco_acquirer import BrainCoAcquirer
from acquisition.neuracle_acquirer import NeuracleAcquirer
from adaptation.session_recorder import SessionRecorder
from decoder.real_time_decoder import RealTimeDecoder
from tools.reprocess_calibration import build_windows, promote_corrected_datasets
from utils.preprocessing import (
    DEFAULT_PREPROCESSING,
    bandpass_filter,
    preprocess_eeg_window,
    resample_eeg,
)


class PreprocessingTests(unittest.TestCase):
    def test_default_profile_matches_cbramod_mi_band(self) -> None:
        self.assertEqual(DEFAULT_PREPROCESSING.low_hz, 0.3)
        self.assertEqual(DEFAULT_PREPROCESSING.high_hz, 40.0)

        sfreq = 200.0
        time = np.arange(2_000, dtype=np.float64) / sfreq
        ten_hz = np.sin(2.0 * np.pi * 10.0 * time)
        sixty_hz = np.sin(2.0 * np.pi * 60.0 * time)
        filtered_ten = bandpass_filter(ten_hz[None, :], sfreq=sfreq)
        filtered_sixty = bandpass_filter(sixty_hz[None, :], sfreq=sfreq)

        ten_rms = float(np.sqrt(np.mean(filtered_ten[:, 200:-200] ** 2)))
        sixty_rms = float(np.sqrt(np.mean(filtered_sixty[:, 200:-200] ** 2)))
        self.assertGreater(ten_rms, 0.6)
        self.assertLess(sixty_rms, 0.05)

    def test_bad_channel_is_repaired_before_car(self) -> None:
        sfreq = 200.0
        time = np.arange(400, dtype=np.float64) / sfreq
        base = np.sin(2.0 * np.pi * 10.0 * time)
        eeg = np.stack(
            [
                base,
                0.8 * np.sin(2.0 * np.pi * 10.0 * time + 0.4),
                1.2 * np.sin(2.0 * np.pi * 10.0 * time + 0.8),
                np.full_like(base, 500.0),
            ],
            axis=0,
        )

        result = preprocess_eeg_window(eeg, sfreq=sfreq)

        self.assertIn(3, result.quality.bad_channel_indices)
        self.assertFalse(np.allclose(result.data[3], 500.0))
        self.assertTrue(np.all(np.isfinite(result.data)))

    def test_large_neuracle_dc_offsets_do_not_create_filter_edge_artifacts(self) -> None:
        sfreq = 200.0
        time = np.arange(400, dtype=np.float64) / sfreq
        offsets = np.asarray(
            [-16_000.0, -13_600.0, -12_400.0, 11_500.0],
            dtype=np.float64,
        )[:, None]
        phases = np.asarray([0.0, 0.4, 0.8, 1.2], dtype=np.float64)[:, None]
        eeg = offsets + 10.0 * np.sin(2.0 * np.pi * 10.0 * time[None, :] + phases)

        result = preprocess_eeg_window(eeg, sfreq=sfreq)

        self.assertTrue(result.quality.accepted)
        self.assertEqual(result.quality.reasons, ())
        self.assertLess(result.quality.peak_abs_uv, 50.0)
        self.assertEqual(result.quality.clip_fraction, 0.0)

    def test_artifact_is_reported_instead_of_silently_called_rejected(self) -> None:
        sfreq = 200.0
        time = np.arange(400, dtype=np.float64) / sfreq
        base = 10.0 * np.sin(2.0 * np.pi * 10.0 * time)
        eeg = np.stack([base, -base, np.roll(base, 5), -np.roll(base, 5)], axis=0)
        eeg[0, 180:220] += 1_000.0

        result = preprocess_eeg_window(eeg, sfreq=sfreq)

        self.assertFalse(result.quality.accepted)
        self.assertTrue(
            {"extreme_amplitude", "excessive_clipping"}
            & set(result.quality.reasons)
        )
        self.assertLessEqual(float(np.max(np.abs(result.data))), 150.0)

    def test_nonfinite_samples_are_repaired_but_window_is_not_training_quality(self) -> None:
        rng = np.random.default_rng(17)
        eeg = rng.standard_normal((8, 400)).astype(np.float32)
        eeg[2, 20] = np.nan

        result = preprocess_eeg_window(eeg, sfreq=200.0)

        self.assertFalse(result.quality.accepted)
        self.assertIn("nonfinite_samples", result.quality.reasons)
        self.assertIn(2, result.quality.bad_channel_indices)
        self.assertTrue(np.all(np.isfinite(result.data)))

    def test_resample_eeg_converts_1000_hz_to_200_hz(self) -> None:
        time = np.arange(2000, dtype=np.float32) / 1000.0
        signal = np.sin(2 * np.pi * 10.0 * time).astype(np.float32)[None, :]

        result = resample_eeg(signal, source_sfreq=1000.0, target_sfreq=200.0)

        self.assertEqual(result.shape, (1, 400))
        target_time = np.arange(400, dtype=np.float32) / 200.0
        expected = np.sin(2 * np.pi * 10.0 * target_time)
        self.assertLess(float(np.mean(np.abs(result[:, 10:-10] - expected[None, 10:-10]))), 0.01)

    def test_resample_eeg_converts_250_hz_to_200_hz(self) -> None:
        time = np.arange(500, dtype=np.float32) / 250.0
        signal = np.sin(2 * np.pi * 10.0 * time).astype(np.float32)[None, :]

        result = resample_eeg(signal, source_sfreq=250.0, target_sfreq=200.0)

        self.assertEqual(result.shape, (1, 400))
        target_time = np.arange(400, dtype=np.float32) / 200.0
        expected = np.sin(2 * np.pi * 10.0 * target_time)
        self.assertLess(
            float(np.mean(np.abs(result[:, 10:-10] - expected[None, 10:-10]))),
            0.01,
        )

    def test_brainco_window_is_resampled_from_250_hz_to_200_hz(self) -> None:
        acquirer = BrainCoAcquirer(
            sfreq=200.0,
            source_sfreq=250.0,
            n_channels=2,
        )
        time = np.arange(500, dtype=np.float32) / 250.0
        signal = np.sin(2 * np.pi * 10.0 * time).astype(np.float32)
        acquirer._append_eeg_samples(
            np.stack([signal, signal], axis=0),
            from_callback=False,
        )
        acquirer._client = object()
        acquirer._sdk = object()

        with mock.patch.object(
            acquirer,
            "_drain_eeg_buffer",
            return_value=np.empty((2, 0), dtype=np.float32),
        ):
            window, timestamps = acquirer.get_chunk(2.0)

        self.assertEqual(window.shape, (2, 400))
        self.assertEqual(timestamps.shape, (400,))
        self.assertEqual(acquirer.metadata.sfreq, 200.0)
        self.assertEqual(acquirer.source_sfreq, 250.0)

    def test_build_windows_preserves_trial_groups(self) -> None:
        rng = np.random.default_rng(17)
        eeg = rng.standard_normal((2, 2_400)).astype(np.float32)
        trials = [
            {"label_id": 0, "block_index": 0, "trial_index": 0, "control_on_sample": 200},
            {"label_id": 1, "block_index": 0, "trial_index": 1, "control_on_sample": 1_200},
        ]

        payload = build_windows(
            eeg,
            trials,
            source_sfreq=200.0,
            target_sfreq=200.0,
            window_sec=2.0,
            stride_sec=0.5,
            control_start_sec=0.5,
            control_stop_sec=4.5,
        )

        self.assertEqual(payload["raw_windows"].shape, (10, 2, 400))
        self.assertEqual(payload["processed_windows"].shape, (10, 2, 400))
        self.assertEqual(payload["quality_clip_fraction"].shape, (10,))
        self.assertEqual(payload["quality_peak_abs_uv"].shape, (10,))
        np.testing.assert_array_equal(payload["trial_ids"], np.repeat([0, 1], 5))
        np.testing.assert_array_equal(payload["labels"], np.repeat([0, 1], 5))
        np.testing.assert_array_equal(payload["window_indices"], np.tile(np.arange(5), 2))

    def test_reprocessed_dataset_is_promoted_atomically_with_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "training_windows_main.npz"
            corrected = root / "training_windows_main_corrected.npz"
            np.savez_compressed(
                original,
                processed_windows=np.empty((0, 2, 400), dtype=np.float32),
                labels=np.empty((0,), dtype=np.int64),
            )
            np.savez_compressed(
                corrected,
                processed_windows=np.ones((3, 2, 400), dtype=np.float32),
                labels=np.asarray([0, 1, 2], dtype=np.int64),
            )

            promoted = promote_corrected_datasets([corrected])

            self.assertEqual(promoted, [original])
            with np.load(original) as payload:
                self.assertEqual(payload["processed_windows"].shape[0], 3)
            backup = root / "training_windows_main.pre_reprocess.npz"
            self.assertTrue(backup.exists())
            with np.load(backup) as payload:
                self.assertEqual(payload["processed_windows"].shape[0], 0)

    def test_neuracle_window_is_resampled_before_decode(self) -> None:
        class FakeServer:
            def GetBufferDataWithTiming(self) -> tuple[np.ndarray, dict[str, float]]:
                time = np.arange(500, dtype=np.float32) / 250.0
                signal = np.sin(2 * np.pi * 10.0 * time).astype(np.float32)
                return np.stack([signal, signal], axis=0), {
                    "device_end_ms": 12_000.0,
                    "arrival_monotonic": 100.0,
                    "total_samples": 500,
                }

        acquirer = NeuracleAcquirer(
            sfreq=200.0,
            source_sfreq=250.0,
            n_channels=2,
        )
        acquirer._server = FakeServer()  # type: ignore[assignment]

        window, timestamps = acquirer.get_chunk(2.0)

        self.assertEqual(window.shape, (2, 400))
        self.assertEqual(timestamps.shape, (400,))
        self.assertEqual(acquirer.source_sfreq, 250.0)
        self.assertEqual(acquirer.metadata.sfreq, 200.0)
        self.assertAlmostEqual(float(timestamps[0]), 98.0)
        self.assertAlmostEqual(float(timestamps[-1]), 99.995)

    def test_neuracle_source_clock_rejects_arrival_jitter_and_compensates_delay(self) -> None:
        class FakeServer:
            def __init__(self) -> None:
                self.calls = 0

            def GetBufferDataWithTiming(self) -> tuple[np.ndarray, dict[str, float]]:
                timing = [
                    {
                        "device_end_ms": 12_000.0,
                        "arrival_monotonic": 100.0,
                        "total_samples": 500,
                    },
                    {
                        "device_end_ms": 12_500.0,
                        "arrival_monotonic": 100.6,
                        "total_samples": 625,
                    },
                ][self.calls]
                self.calls += 1
                return np.zeros((2, 500), dtype=np.float32), timing

        acquirer = NeuracleAcquirer(
            sfreq=200.0,
            source_sfreq=250.0,
            n_channels=2,
            transport_delay_sec=0.05,
        )
        acquirer._server = FakeServer()  # type: ignore[assignment]

        _first_window, first_timestamps = acquirer.get_chunk(2.0)
        _second_window, second_timestamps = acquirer.get_chunk(2.0)

        self.assertAlmostEqual(float(first_timestamps[-1] + 1.0 / 200.0), 99.95)
        self.assertAlmostEqual(float(second_timestamps[-1] + 1.0 / 200.0), 100.45)
        self.assertAlmostEqual(
            acquirer.timing_diagnostics["queueing_jitter_sec"],
            0.1,
        )

    def test_session_event_projects_local_time_to_source_sample_index(self) -> None:
        class FakeAcquirer:
            metadata = AcquirerMetadata(
                name="fake",
                sfreq=200.0,
                n_channels=2,
                timestamp_domain="monotonic",
            )

            def get_new_samples(self) -> tuple[np.ndarray, np.ndarray]:
                timestamps = 9.9 + (np.arange(20, dtype=np.float64) / 200.0)
                return np.zeros((2, 20), dtype=np.float32), timestamps

        with mock.patch(
            "adaptation.session_recorder.time.monotonic",
            side_effect=[9.0, 10.1],
        ):
            recorder = SessionRecorder(FakeAcquirer(), sfreq=200.0, n_channels=2)  # type: ignore[arg-type]
            recorder.pull()
            event = recorder.add_event("control_on")

        self.assertEqual(event.sample_index, 40)
        self.assertEqual(event.payload["alignment_method"], "source-clock-projection")

    def test_decoder_uses_monotonic_sample_timestamps_for_scene_labeling(self) -> None:
        decoder = RealTimeDecoder.__new__(RealTimeDecoder)
        decoder._acquirer = type(
            "_Acquirer",
            (),
            {
                "metadata": AcquirerMetadata(
                    name="fake",
                    sfreq=200.0,
                    n_channels=2,
                    timestamp_domain="monotonic",
                )
            },
        )()
        decoder._sfreq = 200.0
        decoder._window_sec = 2.0
        decoder._timestamp_fallback_warned = False
        timestamps = 98.0 + (np.arange(400, dtype=np.float64) / 200.0)

        with mock.patch(
            "decoder.real_time_decoder.time.monotonic",
            return_value=100.1,
        ):
            start, end = decoder._resolve_window_time_bounds(timestamps)

        self.assertAlmostEqual(start, 98.0)
        self.assertAlmostEqual(end, 100.0)


if __name__ == "__main__":
    unittest.main()
