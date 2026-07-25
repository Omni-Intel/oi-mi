"""Lightweight smoke tests for CLI helper behavior."""

from __future__ import annotations

import asyncio
import importlib
import json
import random
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock
from pathlib import Path

import numpy as np
import yaml

from acquisition.base import AcquirerMetadata, ElectrodeImpedance, classify_impedance_kohm
from acquisition.brainco_acquirer import BrainCoAcquirer
from adaptation.calibrator import Calibrator
from adaptation.mi_protocol import (
    ProtocolConfig,
    build_session_plan,
    generate_block_sequence,
)
from cli import (
    build_acquirer,
    build_game_command_outlet,
    build_model_path,
    default_config,
    effective_device_name,
    iter_test_mode_chunks,
    load_calibration_windows,
    load_config,
    parse_subject_number,
    replay_test_mode,
    resolve_model_path,
    resolve_records_dir,
)
from decoder.real_time_decoder import PredictionResult, RealTimeDecoder
from game_command_router import SharedGameCommandRouter
from models.factory import ModelFactory, split_train_validation_indices
from utils.online_labels import ManualOnlineLabelSource, coerce_label
from tools.verify_experiment_bundle import verify_bundle


class CliHelperTests(unittest.TestCase):
    """Validate config loading and helper utilities."""

    def test_parse_subject_number(self) -> None:
        self.assertEqual(parse_subject_number("S001"), 1)
        self.assertEqual(parse_subject_number("subject-17"), 17)
        self.assertEqual(parse_subject_number("demo"), 1)

    def test_recalibration_protocol_collects_exactly_twelve_minutes(self) -> None:
        protocol = ProtocolConfig.from_config(
            {
                "window_sec": 2.0,
                "step_sec": 0.5,
                "protocol": {
                    "baseline_segments": [
                        {
                            "name": "eyes_open_fixation",
                            "duration_sec": 60.0,
                            "instruction": "test",
                        }
                    ],
                    "trial_timing": {
                        "fixation_sec": 2.0,
                        "cue_sec": 1.0,
                        "control_sec": 5.0,
                        "iti_sec": 2.0,
                    },
                    "calibration_blocks": 4,
                    "calibration_trials_per_class_per_block": 5,
                    "rest_between_blocks_sec": 20.0,
                },
            }
        )

        plan = build_session_plan(protocol)
        collection_seconds = (
            sum(segment.duration_sec for segment in plan.baseline_segments)
            + plan.total_formal_trials * plan.trial_timing.total_sec
            + (len(plan.blocks) - 1) * plan.rest_between_blocks_sec
        )

        self.assertEqual(plan.subject_mode, "recalibration")
        self.assertEqual(plan.total_formal_trials, 60)
        self.assertEqual(collection_seconds, 720.0)
        for block in plan.blocks:
            self.assertEqual(block.count("left"), 5)
            self.assertEqual(block.count("right"), 5)
            self.assertEqual(block.count("idle"), 5)

    def test_grouped_validation_never_splits_one_trial_across_sets(self) -> None:
        groups = np.repeat(np.arange(60, dtype=np.int64), 5)
        group_labels = np.tile(np.repeat(np.arange(3, dtype=np.int64), 20), 1)
        labels = np.repeat(group_labels, 5)

        train_indices, validation_indices = split_train_validation_indices(
            labels,
            groups=groups,
            random_state=17,
        )

        self.assertFalse(
            set(groups[train_indices]).intersection(groups[validation_indices])
        )
        self.assertEqual(set(labels[validation_indices]), {0, 1, 2})

    def test_build_model_path_extension(self) -> None:
        config = {"storage": {"models_dir": "models_storage"}, "device_type": "brainco"}
        self.assertEqual(
            build_model_path(config, "S001", "riemann-mdm"),
            Path("models_storage") / "S001" / "brainco" / "riemann-mdm.pkl",
        )
        self.assertEqual(
            build_model_path(config, "S001", "eegnet"),
            Path("models_storage") / "S001" / "brainco" / "eegnet.pt",
        )
        self.assertEqual(
            build_model_path(config, "S001", "eegnet", device_name="neuracle"),
            Path("models_storage") / "S001" / "neuracle" / "eegnet.pt",
        )

    def test_effective_device_name_uses_dummy_when_hardware_mode_enabled(self) -> None:
        config = {"device_type": "brainco", "hardware_dummy_mode": True}
        self.assertEqual(effective_device_name(config), "dummy")
        self.assertEqual(effective_device_name(config, "neuracle"), "dummy")

    def test_build_model_path_uses_dummy_namespace_in_hardware_mode(self) -> None:
        config = {
            "storage": {"models_dir": "models_storage"},
            "device_type": "brainco",
            "hardware_dummy_mode": True,
        }
        self.assertEqual(
            build_model_path(config, "S002", "eegnet"),
            Path("models_storage") / "S002" / "dummy" / "eegnet.pt",
        )

    def test_resolve_model_path_falls_back_to_bundled_dummy_asset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            asset_dir = Path(tmp_dir) / "dummy_decoders"
            asset_dir.mkdir(parents=True)
            asset_path = asset_dir / "eegnet_64x500.pt"
            asset_path.write_text("placeholder", encoding="utf-8")
            config = {
                "storage": {"models_dir": Path(tmp_dir) / "models"},
                "device_type": "dummy",
                "sfreq": 250,
                "window_sec": 2.0,
            }
            with mock.patch("cli.DUMMY_DECODER_ASSET_DIR", asset_dir):
                resolved = resolve_model_path(
                    config,
                    "S002",
                    "eegnet",
                    n_chans=64,
                    n_times=500,
                )
            self.assertEqual(resolved, asset_path)

    def test_resolve_records_dir_defaults_and_reads_config(self) -> None:
        self.assertEqual(resolve_records_dir({}), Path("records_storage"))
        self.assertEqual(
            resolve_records_dir({"storage": {"records_dir": "/tmp/records"}}),
            Path("/tmp/records"),
        )

    def test_load_config_requires_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text("{}", encoding="utf-8")
            with self.assertRaises(Exception):
                load_config(config_path)

    def test_load_config_success(self) -> None:
        payload = {
            "subject_id": "S001",
            "model_name": "riemann-mdm",
            "device_type": "neuracle",
            "sfreq": 200,
            "n_channels": 64,
            "n_classes": 3,
            "window_sec": 4.0,
            "step_sec": 0.5,
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            config = load_config(config_path)
            self.assertEqual(config["subject_id"], "S001")

    def test_load_config_rejects_wrong_neuracle_source_rate(self) -> None:
        payload = {
            "subject_id": "S001",
            "model_name": "shallowconvnet",
            "device_type": "neuracle",
            "sfreq": 200,
            "n_classes": 3,
            "window_sec": 2.0,
            "step_sec": 0.5,
            "device": {"neuracle_source_sfreq": 200},
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(Exception, "250 Hz"):
                load_config(config_path)

    def test_load_config_creates_default_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "generated" / "config.yaml"
            config = load_config(config_path)
            expected = default_config()
            self.assertTrue(config_path.exists())
            self.assertEqual(config["subject_id"], expected["subject_id"])
            self.assertEqual(config["storage"]["models_dir"], expected["storage"]["models_dir"])

    def test_default_config_matches_project_config_file(self) -> None:
        project_config = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text(encoding="utf-8"))
        self.assertEqual(default_config(), project_config)

    def test_protocol_block_randomizer_respects_constraints(self) -> None:
        sequence = generate_block_sequence({"left": 8, "right": 8, "idle": 8}, rng=random.Random(17))

        self.assertEqual(len(sequence), 24)
        self.assertEqual(sequence.count("left"), 8)
        self.assertEqual(sequence.count("right"), 8)
        self.assertEqual(sequence.count("idle"), 8)
        self.assertGreaterEqual(len(set(sequence[:3])), 2)
        for idx in range(2, len(sequence)):
            self.assertFalse(sequence[idx] == sequence[idx - 1] == sequence[idx - 2])

    def test_model_registry_names(self) -> None:
        self.assertEqual(
            ModelFactory.list_models(),
            [
                "conformer-lite",
                "deepconvnet",
                "eegnet",
                "hybrid-net",
                "riemann-mdm",
                "s4d",
                "shallowconvnet",
            ],
        )

    def test_default_acquirer_does_not_support_impedance_check(self) -> None:
        from acquisition.base import AbstractAcquirer

        class DummyAcquirer(AbstractAcquirer):
            metadata = AcquirerMetadata(name="dummy", sfreq=250.0, n_channels=2)

            def start_stream(self) -> None:
                return

            def stop_stream(self) -> None:
                return

            def get_chunk(self, window_sec: float):
                return np.empty((2, 0), dtype=np.float32), np.empty((0,), dtype=np.float64)

            def get_new_samples(self):
                return np.empty((2, 0), dtype=np.float32), np.empty((0,), dtype=np.float64)

        acquirer = DummyAcquirer()
        self.assertFalse(acquirer.supports_impedance_check())
        with self.assertRaises(NotImplementedError):
            acquirer.check_impedance()

    def test_impedance_threshold_mapping(self) -> None:
        self.assertEqual(classify_impedance_kohm(None), "unknown")
        self.assertEqual(classify_impedance_kohm(4.99), "good")
        self.assertEqual(classify_impedance_kohm(5.0), "ok")
        self.assertEqual(classify_impedance_kohm(10.0), "ok")
        self.assertEqual(classify_impedance_kohm(10.01), "poor")

    def test_build_brainco_acquirer(self) -> None:
        config = {
            "sfreq": 200,
            "buffer_sec": 60,
            "device": {
                "brainco_addr": "",
                "brainco_port": 0,
                "brainco_source_sfreq": 250,
                "brainco_auto_discover": True,
                "brainco_scan_timeout_sec": 6.0,
                "brainco_ready_timeout_sec": 10.0,
                "brainco_gain": 6,
                "brainco_signal_source": "NORMAL",
                "brainco_device_id": "eeg-cap",
            },
        }
        acquirer = build_acquirer(device_name="brainco", config=config)
        self.assertEqual(acquirer.metadata.name, "brainco")
        self.assertEqual(acquirer.metadata.n_channels, 32)
        self.assertEqual(acquirer.metadata.sfreq, 200)
        self.assertEqual(acquirer.source_sfreq, 250)

    def test_build_neuracle_acquirer_uses_250_hz_source(self) -> None:
        config = {
            "sfreq": 200,
            "buffer_sec": 60,
            "device": {
                "neuracle_host": "127.0.0.1",
                "neuracle_port": 8712,
                "neuracle_source_sfreq": 250,
                "neuracle_eeg_channels": 59,
            },
        }

        acquirer = build_acquirer(device_name="neuracle", config=config)

        self.assertEqual(acquirer.metadata.name, "neuracle")
        self.assertEqual(acquirer.metadata.n_channels, 59)
        self.assertEqual(acquirer.metadata.sfreq, 200)
        self.assertEqual(acquirer.source_sfreq, 250)

    def test_build_acquirer_uses_dummy_when_hardware_dummy_mode_enabled(self) -> None:
        config = {
            "sfreq": 250,
            "buffer_sec": 60,
            "hardware_dummy_mode": True,
            "device": {},
        }
        acquirer = build_acquirer(device_name="brainco", config=config)
        self.assertEqual(acquirer.metadata.name, "dummy")
        self.assertEqual(acquirer.metadata.n_channels, 64)

    def test_dummy_acquirer_streams_chunk_after_buffer_fill(self) -> None:
        from acquisition.dummy_acquirer import DummyAcquirer

        acquirer = DummyAcquirer(sfreq=250.0, n_channels=4, buffer_sec=2.0, startup_delay_sec=0.0, chunk_ms=20.0)
        acquirer.start_stream()
        try:
            deadline = time.monotonic() + 2.0
            window = None
            while time.monotonic() < deadline:
                try:
                    window, timestamps = acquirer.get_chunk(0.5)
                    break
                except RuntimeError:
                    time.sleep(0.05)
            self.assertIsNotNone(window)
            assert window is not None
            self.assertEqual(window.shape, (4, 125))
            self.assertEqual(timestamps.shape, (125,))
        finally:
            acquirer.stop_stream()

    def test_brainco_discovery_uses_configured_port_when_sdk_returns_only_ip(self) -> None:
        acquirer = BrainCoAcquirer(brainco_port=9527, auto_discover=True)
        self.assertEqual(acquirer._coerce_discovered_target("192.168.3.9"), ("192.168.3.9", 9527))

    def test_brainco_discovery_parses_host_port_text(self) -> None:
        acquirer = BrainCoAcquirer(auto_discover=True)
        self.assertEqual(acquirer._coerce_discovered_target("192.168.3.9:9527"), ("192.168.3.9", 9527))

    def test_brainco_resolve_addr_port_caches_discovery_target(self) -> None:
        acquirer = BrainCoAcquirer(auto_discover=True)

        with mock.patch.object(acquirer, "_discover_device", return_value=("192.168.3.9", 53129)) as discover:
            self.assertEqual(acquirer._resolve_addr_port(), ("192.168.3.9", 53129))
            self.assertEqual(acquirer._resolve_addr_port(), ("192.168.3.9", 53129))

        self.assertEqual(discover.call_count, 1)

    def test_brainco_missing_port_prefers_zeroconf_before_callback_rescan(self) -> None:
        class FakeSdk:
            async def mdns_start_scan(self):
                return "192.168.3.9"

            def mdns_stop_scan(self):
                return None

        acquirer = BrainCoAcquirer(auto_discover=True)
        acquirer._sdk = FakeSdk()

        async def run_case() -> tuple[str, int]:
            with (
                mock.patch.object(
                    acquirer,
                    "_discover_device_via_zeroconf_async",
                    new=mock.AsyncMock(return_value=("192.168.3.9", 53129)),
                ) as zeroconf,
                mock.patch.object(
                    acquirer,
                    "_discover_device_via_callback_async",
                    new=mock.AsyncMock(return_value=("192.168.3.9", 9527)),
                ) as callback_scan,
            ):
                result = await acquirer._discover_device_async()
                zeroconf.assert_awaited_once()
                callback_scan.assert_not_awaited()
                return result

        self.assertEqual(asyncio.run(run_case()), ("192.168.3.9", 53129))

    def test_brainco_timeout_stops_sdk_scan_before_fallback(self) -> None:
        class FakeSdk:
            def __init__(self) -> None:
                self.stop_calls = 0

            async def mdns_start_scan(self):
                await asyncio.sleep(0.02)
                return None

            def mdns_stop_scan(self):
                self.stop_calls += 1
                return None

        acquirer = BrainCoAcquirer(auto_discover=True, scan_timeout_sec=0.001)
        acquirer._sdk = FakeSdk()

        async def run_case() -> tuple[str, int]:
            with (
                mock.patch.object(
                    acquirer,
                    "_discover_device_via_zeroconf_async",
                    new=mock.AsyncMock(return_value=("192.168.3.9", 53129)),
                ) as zeroconf,
                mock.patch.object(
                    acquirer,
                    "_discover_device_via_callback_async",
                    new=mock.AsyncMock(return_value=None),
                ) as callback_scan,
            ):
                result = await acquirer._discover_device_async()
                zeroconf.assert_awaited_once()
                callback_scan.assert_not_awaited()
                return result

        self.assertEqual(asyncio.run(run_case()), ("192.168.3.9", 53129))
        self.assertEqual(acquirer._sdk.stop_calls, 1)

    def test_brainco_extracts_msg_id_from_callback_payloads(self) -> None:
        acquirer = BrainCoAcquirer(auto_discover=True)
        dummy = type("Resp", (), {"msgId": 17})()

        self.assertEqual(acquirer._extract_message_id(23), 23)
        self.assertEqual(acquirer._extract_message_id({"msgId": 11}), 11)
        self.assertEqual(acquirer._extract_message_id(("ignore", dummy)), 17)
        self.assertEqual(acquirer._extract_message_id('{"msgId": 5, "status": "ok"}'), 5)
        self.assertIsNone(acquirer._extract_message_id("no-id"))

    def test_brainco_wait_for_command_response_can_continue_when_missing_is_allowed(self) -> None:
        acquirer = BrainCoAcquirer(auto_discover=True)
        acquirer._ready_timeout_sec = 0.0
        acquirer._response_event = threading.Event()
        acquirer._msg_resp_lock = threading.Lock()
        acquirer._pending_msg_responses = {}

        acquirer._wait_for_command_response(1, "set_eeg_config", allow_missing=True)

    def test_brainco_buffer_ingest_updates_cache(self) -> None:
        acquirer = BrainCoAcquirer(n_channels=4)

        acquirer._append_eeg_samples(np.arange(8, dtype=np.float32).reshape(4, 2), from_callback=False)

        self.assertTrue(acquirer._first_sample_event.is_set())
        self.assertEqual(acquirer._eeg_cache.shape, (4, 2))
        self.assertEqual(acquirer._callback_sample_count, 0)
        np.testing.assert_array_equal(acquirer._eeg_cache[:, 0], np.asarray([0, 2, 4, 6], dtype=np.float32))
        self.assertEqual(acquirer._total_samples_seen, 2)

    def test_brainco_callback_ingest_marks_callback_samples(self) -> None:
        acquirer = BrainCoAcquirer(n_channels=4)

        acquirer._append_eeg_samples(np.arange(8, dtype=np.float32).reshape(4, 2), from_callback=True)

        self.assertEqual(acquirer._callback_sample_count, 2)

    def test_brainco_get_chunk_waits_for_fresh_samples(self) -> None:
        acquirer = BrainCoAcquirer(sfreq=200, source_sfreq=250, n_channels=2, buffer_sec=10.0)
        acquirer.metadata = AcquirerMetadata(name="brainco", sfreq=2.0, n_channels=2)
        acquirer.source_sfreq = 2.0
        acquirer._client = object()
        acquirer._sdk = object()

        batches = [
            np.asarray([[1.0, 2.0], [10.0, 20.0]], dtype=np.float32),
            np.empty((2, 0), dtype=np.float32),
            np.asarray([[3.0], [30.0]], dtype=np.float32),
        ]

        def fake_drain() -> np.ndarray:
            if not batches:
                return np.empty((2, 0), dtype=np.float32)
            data = batches.pop(0)
            if data.shape[1] > 0:
                acquirer._append_eeg_samples(data, from_callback=False)
            return data

        with mock.patch.object(acquirer, "_drain_eeg_buffer", side_effect=fake_drain):
            first, _ = acquirer.get_chunk(1.0)
            second, _ = acquirer.get_chunk(1.0)

        np.testing.assert_array_equal(first, np.asarray([[1.0, 2.0], [10.0, 20.0]], dtype=np.float32))
        np.testing.assert_array_equal(second, np.asarray([[2.0, 3.0], [20.0, 30.0]], dtype=np.float32))

    def test_brainco_wait_for_command_response_accepts_generic_callback(self) -> None:
        acquirer = BrainCoAcquirer(n_channels=4, ready_timeout_sec=0.01)
        acquirer._generic_msg_responses.append(("ok",))

        acquirer._wait_for_command_response(7, "set_eeg_config")

    def test_brainco_wait_for_command_response_allows_missing_when_requested(self) -> None:
        acquirer = BrainCoAcquirer(n_channels=4, ready_timeout_sec=0.01)

        acquirer._wait_for_command_response(7, "set_eeg_config", allow_missing=True)

    def test_brainco_normalizes_numeric_impedance_series(self) -> None:
        acquirer = BrainCoAcquirer(n_channels=4)

        results = acquirer._normalize_impedance_payloads([([1.2, 5.0, 12.0, 3.4],)])

        self.assertEqual([result.channel for result in results], [1, 2, 3, 4])
        self.assertEqual([result.status for result in results], ["good", "ok", "poor", "good"])
        self.assertEqual(results[2].impedance_kohm, 12.0)

    def test_brainco_normalizes_sdk_quality_mapping(self) -> None:
        acquirer = BrainCoAcquirer(n_channels=2)

        results = acquirer._normalize_impedance_payloads(
            [
                ({"channel": 0, "quality": "good", "message": "stable"},),
                ({"channel": 1, "quality": "lead_off"},),
            ]
        )

        self.assertEqual([result.channel for result in results], [1, 2])
        self.assertEqual(results[0].status, "good")
        self.assertEqual(results[0].message, "stable")
        self.assertEqual(results[1].status, "poor")
        self.assertIsNone(results[1].impedance_kohm)

    def test_gui_impedance_helpers_are_hardware_free(self) -> None:
        sys.modules.pop("gui", None)
        gui = importlib.import_module("gui")

        results = [
            ElectrodeImpedance(channel=1, name="Fp1", impedance_kohm=3.2, status="good", message=None),
            ElectrodeImpedance(channel=2, name="Fp2", impedance_kohm=None, status="unknown", message="sdk quality only"),
            ElectrodeImpedance(channel=3, name=None, impedance_kohm=11.8, status="poor", message="lead off"),
        ]

        self.assertEqual(gui._format_impedance_value(None), "-")
        self.assertEqual(gui._format_impedance_value(3.2), "3.20")
        self.assertEqual(
            gui._summarize_impedance_results(results),
            {"good": 1, "ok": 0, "poor": 1, "unknown": 1},
        )

        table_html = gui._build_impedance_table_html(results)
        self.assertIn("Fp1", table_html)
        self.assertIn("阻抗 (kΩ)", table_html)
        self.assertIn("sdk quality only", table_html)
        self.assertIn("#15803D", table_html)
        self.assertIn("#6B7280", table_html)
        self.assertEqual(
            gui._format_send_state(
                {"last_send_success": True, "updated_at": 100.0},
                now=101.0,
            ),
            "success",
        )
        self.assertEqual(
            gui._format_send_state(
                {"last_send_success": True, "updated_at": 100.0},
                now=104.1,
            ),
            "stale",
        )
        self.assertIn("新被试", gui._missing_model_guidance({"device_type": "neuracle"}))
        self.assertIn("seed-dummy-decoders", gui._missing_model_guidance({"device_type": "dummy"}))

    def test_build_game_command_outlet_disabled_by_default(self) -> None:
        self.assertIsNone(build_game_command_outlet({"output": {}}))

    def test_build_game_command_outlet_enabled(self) -> None:
        class FakeOutlet:
            def __init__(self) -> None:
                self.commands: list[str] = []

            def push(self, command: str) -> None:
                self.commands.append(command)

        fake_outlet = FakeOutlet()

        class FakeRouter:
            def build_proxy(self, *, source: str):
                self.source = source
                return fake_outlet

        fake_router = FakeRouter()
        with mock.patch("cli.get_shared_game_command_router", return_value=fake_router):
            outlet = build_game_command_outlet(
                {
                    "output": {
                        "ar_game": {
                            "enabled": True,
                            "host": "127.0.0.1",
                            "port": 5005,
                            "timeout_sec": 0.5,
                            "startup_command_delay_sec": 0.0,
                            "startup_sequence": [
                                {"command": "OPEN_LAUNCHER", "delay_after_sec": 0.0},
                                {"command": "OPEN_3D_GAME", "delay_after_sec": 0.0},
                                {"command": "LAUNCHER_SELECT", "delay_after_sec": 0.0},
                            ],
                        }
                    }
                }
            )
        self.assertIs(outlet, fake_outlet)
        self.assertEqual(fake_router.source, "decoder")
        self.assertEqual(
            fake_outlet.commands,
            ["OPEN_LAUNCHER", "OPEN_3D_GAME", "LAUNCHER_SELECT"],
        )

    def test_build_game_command_outlet_can_disable_startup_scene(self) -> None:
        fake_outlet = mock.Mock()
        fake_router = mock.Mock()
        fake_router.build_proxy.return_value = fake_outlet

        with mock.patch("cli.get_shared_game_command_router", return_value=fake_router):
            outlet = build_game_command_outlet(
                {
                    "output": {
                        "ar_game": {
                            "enabled": True,
                            "startup_sequence": [],
                        }
                    }
                }
            )

        self.assertIs(outlet, fake_outlet)
        fake_outlet.push.assert_not_called()

    def test_realtime_decoder_maps_predictions_to_game_commands(self) -> None:
        self.assertEqual(
            RealTimeDecoder._to_game_command(PredictionResult("左手", 0.9, 0.1, 0)),
            "LEFT",
        )
        self.assertEqual(
            RealTimeDecoder._to_game_command(PredictionResult("右手", 0.9, 0.1, 1)),
            "RIGHT",
        )
        self.assertEqual(
            RealTimeDecoder._to_game_command(PredictionResult("静息", 0.9, 0.1, 2)),
            None,
        )
        self.assertEqual(
            RealTimeDecoder._to_game_command(PredictionResult("静息", 0.4, 0.6, None)),
            None,
        )

    def test_realtime_decoder_post_process_suppresses_low_confidence_predictions(self) -> None:
        decoder = RealTimeDecoder.__new__(RealTimeDecoder)
        decoder._confidence_threshold = 0.99

        result = decoder._post_process(np.asarray([0.20, 0.35, 0.45], dtype=np.float32))

        self.assertIsNone(result.class_id)
        self.assertEqual(result.label, "静息")
        self.assertAlmostEqual(result.confidence, 0.45, places=6)

        lateral_result = decoder._post_process(np.asarray([0.41, 0.39, 0.20], dtype=np.float32))

        self.assertIsNone(lateral_result.class_id)
        self.assertEqual(lateral_result.label, "静息")
        self.assertAlmostEqual(lateral_result.confidence, 0.41, places=6)

    def test_realtime_decoder_post_process_keeps_high_confidence_idle_as_real_class(self) -> None:
        decoder = RealTimeDecoder.__new__(RealTimeDecoder)
        decoder._confidence_threshold = 0.7

        result = decoder._post_process(np.asarray([0.05, 0.10, 0.85], dtype=np.float32))

        self.assertEqual(result.class_id, 2)
        self.assertEqual(result.label, "静息")
        self.assertAlmostEqual(result.confidence, 0.85, places=6)

    def test_realtime_decoder_repeats_keepalive_and_emits_stop(self) -> None:
        class FakeSender:
            def __init__(self) -> None:
                self.commands: list[str] = []

            def push(self, command: str) -> None:
                self.commands.append(command)

        decoder = RealTimeDecoder.__new__(RealTimeDecoder)
        decoder._game_command_outlet = FakeSender()
        decoder._game_session_started = True
        decoder._last_game_command = None
        decoder._last_game_transport_command = None
        decoder._last_game_transport_sent_at = 0.0
        decoder._game_command_keepalive_sec = 0.0

        decoder._push_game_command("LEFT")
        decoder._push_game_command("LEFT")
        decoder._push_game_command(None)
        decoder._push_game_command("RIGHT")

        self.assertEqual(decoder._game_command_outlet.commands, ["LEFT", "LEFT", "STOP", "RIGHT"])

    def test_realtime_decoder_stop_sends_final_stop_and_closes_resources(self) -> None:
        class FakeAcquirer:
            def __init__(self) -> None:
                self.stopped = False

            def stop_stream(self) -> None:
                self.stopped = True

        class FakeSender:
            def __init__(self) -> None:
                self.commands: list[str] = []
                self.closed = False

            def push(self, command: str) -> None:
                self.commands.append(command)

            def close(self) -> None:
                self.closed = True

        decoder = RealTimeDecoder.__new__(RealTimeDecoder)
        decoder._stop_event = threading.Event()
        decoder._thread = None
        decoder._acquirer = FakeAcquirer()
        decoder._game_command_outlet = FakeSender()
        decoder._batch_adapter = None
        decoder._neuroonline_adapter = None
        decoder._online_label_source = None

        decoder.stop()

        self.assertTrue(decoder._acquirer.stopped)
        self.assertEqual(decoder._game_command_outlet.commands, ["STOP"])
        self.assertTrue(decoder._game_command_outlet.closed)

    def test_test_mode_without_windows_finalizes_writer_and_game(self) -> None:
        class FakeAcquirer:
            metadata = AcquirerMetadata(name="fake", sfreq=250.0, n_channels=4)

            def __init__(self) -> None:
                self.started = False
                self.stopped = False

            def start_stream(self) -> None:
                self.started = True

            def stop_stream(self) -> None:
                self.stopped = True

        class FakeSender:
            def __init__(self) -> None:
                self.commands: list[str] = []
                self.closed = False

            def push(self, command: str) -> None:
                self.commands.append(command)

            def close(self) -> None:
                self.closed = True

        class FakeConsole:
            def print(self, *_args, **_kwargs) -> None:
                return

        class FakeMarker:
            def send(self, _label: int, timestamp: float | None = None) -> None:
                del timestamp

        acquirer = FakeAcquirer()
        sender = FakeSender()
        decoder = RealTimeDecoder(
            acquirer=acquirer,
            model=object(),
            console=FakeConsole(),
            command_outlet=object(),
            game_command_outlet=sender,
            sfreq=250.0,
            window_sec=2.0,
            step_sec=0.5,
            confidence_threshold=0.7,
            mc_dropout_passes=1,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "empty-test"
            with self.assertRaisesRegex(RuntimeError, "did not collect"):
                decoder.run_test_mode(
                    subject_id="test",
                    marker_backend=FakeMarker(),
                    duration_sec=0,
                    initial_rest_sec=0.0,
                    save_dir=output_dir,
                )
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

        self.assertTrue(acquirer.started)
        self.assertTrue(acquirer.stopped)
        self.assertEqual(sender.commands, ["STOP"])
        self.assertTrue(sender.closed)
        self.assertEqual(manifest["status"], "no_windows")
        self.assertIn("end_time", manifest)

    def test_realtime_decoder_status_callback_reports_command_state(self) -> None:
        payloads: list[dict] = []

        decoder = RealTimeDecoder.__new__(RealTimeDecoder)
        decoder._status_callback = payloads.append
        decoder._last_game_transport_command = "LEFT"
        decoder._last_game_transport_error = None
        decoder._last_game_transport_sent_at = 1.0

        decoder._emit_status(PredictionResult("左手", 0.88, 0.12, 0), "LEFT")

        self.assertEqual(payloads[-1]["prediction"], "左手")
        self.assertEqual(payloads[-1]["mapped_command"], "LEFT")
        self.assertEqual(payloads[-1]["last_transport_command"], "LEFT")
        self.assertTrue(payloads[-1]["last_send_success"])

    def test_realtime_decoder_updates_from_labeled_window_when_enabled(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.calls: list[tuple[tuple[int, ...], list[int], float]] = []

            def update(self, X, y, *, learning_rate: float, epochs: int = 1):
                del epochs
                self.calls.append((tuple(X.shape), y.tolist(), learning_rate))
                return {"updated": float(X.shape[0]), "loss": 0.25}

        model = FakeModel()
        decoder = RealTimeDecoder.__new__(RealTimeDecoder)
        decoder._model = model
        decoder._online_update_enabled = True
        decoder._online_update_learning_rate = 0.001
        decoder._online_update_every = 1
        decoder._online_update_count = 0
        decoder._online_seen_labeled_windows = 0

        metrics = decoder._maybe_update_model(
            processed=np.zeros((2, 500), dtype=np.float32),
            true_label=1,
        )

        self.assertEqual(metrics, {"updated": 1.0, "loss": 0.25})
        self.assertEqual(decoder._online_update_count, 1)
        self.assertEqual(model.calls, [((1, 2, 500), [1], 0.001)])

    def test_realtime_decoder_creates_model_directory_before_online_save(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.saved_path: Path | None = None

            def save(self, path: Path) -> None:
                self.assert_parent(path)
                self.saved_path = path

            @staticmethod
            def assert_parent(path: Path) -> None:
                if not path.parent.is_dir():
                    raise AssertionError("model parent directory was not created")

        with tempfile.TemporaryDirectory() as tmp_dir:
            decoder = RealTimeDecoder.__new__(RealTimeDecoder)
            decoder._model = FakeModel()
            decoder._model_lock = threading.RLock()
            decoder._model_save_path = Path(tmp_dir) / "S001" / "dummy" / "shallowconvnet.pt"

            decoder._save_current_model()

            self.assertIsNone(decoder._model.saved_path)
            self.assertTrue(decoder._model_save_path.parent.is_dir())

    def test_manual_online_label_source_overlaps_decode_window(self) -> None:
        source = ManualOnlineLabelSource(default_ttl_sec=1.0)
        event = source.set_label("right", timestamp_monotonic=10.0, payload={"target_lane": 1})

        self.assertEqual(event.label_id, 1)
        self.assertEqual(event.payload, {"target_lane": 1})
        self.assertIsNotNone(source.get_label(window_start=9.5, window_end=10.2))
        self.assertIsNone(source.get_label(window_start=11.2, window_end=11.5))

    def test_online_label_aliases(self) -> None:
        self.assertEqual(coerce_label("left"), (0, "left"))
        self.assertEqual(coerce_label("右"), (1, "right"))
        self.assertEqual(coerce_label("stop"), (2, "idle"))

    def test_calibrator_saves_calibration_windows(self) -> None:
        class FakeAcquirer:
            metadata = AcquirerMetadata(name="fake", sfreq=250.0, n_channels=2)

            def __init__(self) -> None:
                self._calls = 0
                self._sample_cursor = 0

            def start_stream(self) -> None:
                return

            def stop_stream(self) -> None:
                return

            def get_chunk(self, window_sec: float):
                self._calls += 1
                samples = int(window_sec * self.metadata.sfreq)
                window = np.full((self.metadata.n_channels, samples), float(self._calls), dtype=np.float32)
                timestamps = np.arange(samples, dtype=np.float64) / self.metadata.sfreq
                return window, timestamps

            def get_new_samples(self):
                samples = 25
                start = self._sample_cursor
                stop = start + samples
                self._sample_cursor = stop
                window = np.tile(
                    np.arange(start, stop, dtype=np.float32)[None, :],
                    (self.metadata.n_channels, 1),
                )
                timestamps = np.arange(start, stop, dtype=np.float64) / self.metadata.sfreq
                return window, timestamps

        class FakeModel:
            def fit(self, X, y, **kwargs):
                return {"val_acc": float(len(y) > 0), "val_loss": 0.0}

            def save(self, path: Path) -> None:
                path.write_text("fake-model", encoding="utf-8")

            def load(self, path: Path) -> None:
                return

        class FakeMarkerBackend:
            def send(self, label: int, timestamp: float | None = None) -> None:
                return

            def send_event(self, event_name: str, timestamp: float | None = None) -> None:
                return

        class FakeConsole:
            def print(self, *args, **kwargs) -> None:
                return

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "models" / "fake.pkl"
            records_dir = Path(tmp_dir) / "records"
            calibrator = Calibrator(
                acquirer=FakeAcquirer(),
                model=FakeModel(),
                marker_backend=FakeMarkerBackend(),
                console=FakeConsole(),
                sfreq=250.0,
                window_sec=1.0,
                step_sec=100.0,
                model_path=model_path,
                calibration_records_dir=records_dir,
                protocol_config=ProtocolConfig.from_config(
                    {
                        "window_sec": 1.0,
                        "step_sec": 0.5,
                        "protocol": {
                            "control_start_offset_sec": 0.0,
                            "control_stop_offset_sec": 1.0,
                            "export_window_sec": None,
                            "baseline_segments": [
                                {
                                    "name": "idle_baseline",
                                    "duration_sec": 0.0,
                                    "instruction": "test",
                                }
                            ],
                            "practice_labels": [],
                            "calibration_blocks": 1,
                            "calibration_trials_per_class_per_block": 1,
                            "rest_between_blocks_sec": 0.0,
                            "trial_timing": {
                                "fixation_sec": 0.0,
                                "cue_sec": 0.0,
                                "control_sec": 1.0,
                                "iti_sec": 0.0,
                            },
                        },
                    }
                ),
            )

            with (
                mock.patch(
                    "adaptation.calibrator.preprocess_eeg_window",
                    lambda data, sfreq: SimpleNamespace(
                        data=data.astype(np.float32) + 10.0,
                        quality=SimpleNamespace(
                            accepted=True,
                            peak_abs_uv=1.0,
                            clip_fraction=0.0,
                            bad_channel_fraction=0.0,
                        ),
                    ),
                ),
            ):
                result = calibrator.calibrate(
                    duration_sec=30,
                    epochs=1,
                    batch_size=1,
                    learning_rate=0.001,
                    patience=1,
                    head_only=False,
                )

            assert result.calibration_data_path is not None
            self.assertTrue(result.calibration_data_path.exists())
            with np.load(result.calibration_data_path) as payload:
                self.assertEqual(payload["raw_windows"].shape[0], result.windows_collected)
                self.assertEqual(payload["processed_windows"].shape[0], result.windows_collected)
                self.assertEqual(payload["labels"].shape[0], result.windows_collected)
                self.assertEqual(payload["trial_ids"].shape[0], result.windows_collected)
                self.assertEqual(
                    payload["window_start_samples"].shape[0],
                    result.windows_collected,
                )
                self.assertEqual(
                    payload["window_stop_samples"].shape[0],
                    result.windows_collected,
                )
                self.assertTrue(np.all(payload["processed_windows"] == payload["raw_windows"] + 10.0))
            assert result.session_dir is not None
            continuous_eeg = np.load(result.session_dir / "continuous_eeg.npy")
            sample_timestamps = np.load(
                result.session_dir / "continuous_sample_timestamps.npy"
            )
            self.assertEqual(sample_timestamps.size, continuous_eeg.shape[-1])
            metadata = json.loads(
                (result.session_dir / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["integrity"]["status"], "complete")
            self.assertTrue(metadata["integrity"]["sample_timestamp_count_matches"])
            self.assertTrue(verify_bundle(result.session_dir)["ok"])

    def test_load_calibration_windows_concatenates_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            records_dir = Path(tmp_dir) / "records"
            session_a = records_dir / "S001" / "calibration" / "20260413_100000"
            session_b = records_dir / "S001" / "calibration" / "20260413_110000"
            session_a.mkdir(parents=True)
            session_b.mkdir(parents=True)

            np.savez_compressed(
                session_a / "training_windows_main.npz",
                raw_windows=np.ones((2, 3, 4), dtype=np.float32),
                processed_windows=np.full((2, 3, 4), 10.0, dtype=np.float32),
                labels=np.asarray([0, 1], dtype=np.int64),
                trial_ids=np.asarray([0, 0], dtype=np.int64),
            )
            np.savez_compressed(
                session_b / "training_windows_main.npz",
                raw_windows=np.ones((1, 3, 4), dtype=np.float32) * 2,
                processed_windows=np.full((1, 3, 4), 20.0, dtype=np.float32),
                labels=np.asarray([2], dtype=np.int64),
                trial_ids=np.asarray([0], dtype=np.int64),
            )

            X, y, sessions = load_calibration_windows(records_dir, "S001")

            self.assertEqual(X.shape, (3, 3, 4))
            np.testing.assert_array_equal(y, np.asarray([0, 1, 2], dtype=np.int64))
            self.assertEqual([session.name for session in sessions], ["20260413_100000", "20260413_110000"])
            self.assertTrue(np.all(X[:2] == 10.0))
            self.assertTrue(np.all(X[2:] == 20.0))

            _, _, groups, _ = load_calibration_windows(
                records_dir,
                "S001",
                include_groups=True,
            )
            assert groups is not None
            np.testing.assert_array_equal(groups, np.asarray([0, 0, 1], dtype=np.int64))

    def test_replay_test_mode_runs_model_on_saved_chunks(self) -> None:
        class FakeModel:
            def predict_proba(self, X, mc_dropout_passes=1):
                del mc_dropout_passes
                outputs = []
                for idx in range(X.shape[0]):
                    if idx % 2 == 0:
                        outputs.append([0.7, 0.2, 0.1])
                    else:
                        outputs.append([0.1, 0.8, 0.1])
                return np.asarray(outputs, dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp_dir:
            test_mode_dir = Path(tmp_dir) / "test_mode"
            chunks_dir = test_mode_dir / "chunks"
            chunks_dir.mkdir(parents=True)
            np.savez_compressed(
                chunks_dir / "chunk_000000.npz",
                eeg_windows=np.zeros((4, 2, 8), dtype=np.float32),
                labels_true=np.asarray([0, 1, 0, 1], dtype=np.int64),
                labels_pred=np.asarray([-1, -1, -1, -1], dtype=np.int64),
                confidences=np.zeros(4, dtype=np.float32),
            )

            with mock.patch("cli.filter_and_transform", side_effect=lambda window, sfreq: window):
                result = replay_test_mode(
                    model=FakeModel(),
                    test_mode_dir=test_mode_dir,
                    sfreq=250.0,
                    mc_dropout_passes=3,
                )

            self.assertEqual(result["windows"], 4)
            self.assertAlmostEqual(result["accuracy"], 1.0, places=6)
            self.assertAlmostEqual(result["mean_confidence"], 0.75, places=6)
            np.testing.assert_array_equal(result["y_pred"], np.asarray([0, 1, 0, 1], dtype=np.int64))

    def test_iter_test_mode_chunks_requires_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_mode_dir = Path(tmp_dir) / "test_mode"
            (test_mode_dir / "chunks").mkdir(parents=True)
            with self.assertRaises(Exception):
                iter_test_mode_chunks(test_mode_dir)

    def test_manual_web_override_blocks_decoder_temporarily(self) -> None:
        class FakeTransport:
            def __init__(self) -> None:
                self.commands: list[str] = []

            def push(self, command: str) -> None:
                self.commands.append(command)

        config = {
            "output": {
                "ar_game": {"enabled": True},
                "web_control": {
                    "manual_override_hold_sec": 10.0,
                    "manual_override_release_sec": 0.25,
                },
            }
        }

        transport = FakeTransport()
        with mock.patch("game_command_router._build_transport", return_value=transport):
            router = SharedGameCommandRouter(config)

            router.push("LEFT", source="decoder")
            router.push("RIGHT", source="web")
            router.push("LEFT", source="decoder")

            self.assertEqual(transport.commands, ["LEFT", "RIGHT"])

    def test_manual_stop_allows_decoder_to_resume_after_release_window(self) -> None:
        class FakeTransport:
            def __init__(self) -> None:
                self.commands: list[str] = []

            def push(self, command: str) -> None:
                self.commands.append(command)

        config = {
            "output": {
                "ar_game": {"enabled": True},
                "web_control": {
                    "manual_override_hold_sec": 0.8,
                    "manual_override_release_sec": 0.0,
                },
            }
        }

        transport = FakeTransport()
        with mock.patch("game_command_router._build_transport", return_value=transport):
            router = SharedGameCommandRouter(config)

            router.push("RIGHT", source="web")
            router.push("STOP", source="web")
            router.push("LEFT", source="decoder")

            self.assertEqual(transport.commands, ["RIGHT", "STOP", "LEFT"])

    def test_decoder_proxy_forwards_scene_command_with_ack(self) -> None:
        class FakeTransport:
            def __init__(self) -> None:
                self.commands: list[str] = []
                self.acked_commands: list[str] = []

            def push(self, command: str) -> None:
                self.commands.append(command)

            def push_with_ack(self, command: str) -> None:
                self.acked_commands.append(command)

        config = {
            "output": {
                "ar_game": {"enabled": True},
                "web_control": {
                    "manual_override_hold_sec": 10.0,
                    "manual_override_release_sec": 0.25,
                },
            }
        }

        transport = FakeTransport()
        with mock.patch("game_command_router._build_transport", return_value=transport):
            router = SharedGameCommandRouter(config)
            proxy = router.build_proxy(source="decoder")

            router.push("RIGHT", source="web")
            proxy.push_with_ack("SCENE_LEFT")

            self.assertEqual(transport.commands, ["RIGHT"])
            self.assertEqual(transport.acked_commands, ["SCENE_LEFT"])

    def test_scene_ack_fails_closed_for_unsupported_transport(self) -> None:
        class FakeTransport:
            def push(self, command: str) -> None:
                del command

        config = {"output": {"ar_game": {"enabled": True}}}

        with mock.patch("game_command_router._build_transport", return_value=FakeTransport()):
            proxy = SharedGameCommandRouter(config).build_proxy(source="decoder")

            with self.assertRaisesRegex(RuntimeError, "does not support scene ACK"):
                proxy.push_with_ack("SCENE_RIGHT")


if __name__ == "__main__":
    unittest.main()
