from __future__ import annotations

import unittest

from decoder.real_time_decoder import RealTimeDecoder
from utils.online_labels import CuedOnlineLabelSource


class _Clock:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class _GameOutlet:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.events: list[dict[str, object]] = []

    def push(self, command: str) -> None:
        self.commands.append(command)

    def push_with_ack(self, command: str) -> None:
        self.push(command)

    def poll_events(self) -> list[dict[str, object]]:
        events = list(self.events)
        self.events.clear()
        return events


class CuedOnlineLabelTests(unittest.TestCase):
    def test_only_windows_inside_one_scene_receive_labels(self) -> None:
        clock = _Clock(100.0)
        source = CuedOnlineLabelSource(
            ["left", "right", "idle"],
            scene_duration_sec=5.0,
            start_delay_sec=0.0,
            boundary_guard_sec=0.0,
            clock=clock,
        )
        self.assertTrue(source.confirm_scene_applied(scene_index=0, timestamp_monotonic=100.0))

        clock.value = 102.0
        label = source.get_label(window_start=100.0, window_end=102.0)
        self.assertIsNotNone(label)
        assert label is not None
        self.assertEqual(label.label_name, "left")
        self.assertEqual(label.event_id, "scene-000000")

        clock.value = 105.5
        self.assertIsNone(source.get_label(window_start=103.5, window_end=105.5))

    def test_scene_protocol_repeats_balanced_sequence_without_stopping(self) -> None:
        clock = _Clock(0.0)
        source = CuedOnlineLabelSource(
            ["left", "right", "idle"],
            scene_duration_sec=5.0,
            start_delay_sec=0.0,
            boundary_guard_sec=0.0,
            clock=clock,
        )

        clock.value = 30.0
        status = source.status()
        self.assertEqual(status["phase"], "control")
        self.assertEqual(status["protocol_mode"], "continuous-scene")
        self.assertEqual(status["scene_index"], 6)
        self.assertEqual(status["scene_number"], 7)
        self.assertEqual(status["label_name"], "left")
        self.assertEqual(source.metadata()["balance_pool_scenes"], 3)
        self.assertNotIn("total_trials", status)

    def test_start_delay_has_no_label_or_hidden_control_phase(self) -> None:
        clock = _Clock(10.0)
        source = CuedOnlineLabelSource(
            ["right"],
            scene_duration_sec=5.0,
            start_delay_sec=1.0,
            boundary_guard_sec=0.0,
            clock=clock,
        )

        self.assertEqual(source.status()["phase"], "preparing")
        self.assertIsNone(source.get_label(window_start=9.0, window_end=10.0))
        clock.value = 11.0
        self.assertEqual(source.status()["phase"], "control")

    def test_unity_ack_anchors_scene_start_and_boundary_guard(self) -> None:
        clock = _Clock(100.0)
        source = CuedOnlineLabelSource(
            ["left"],
            scene_duration_sec=5.0,
            start_delay_sec=0.0,
            boundary_guard_sec=0.1,
            clock=clock,
        )

        self.assertIsNone(source.get_label(window_start=100.0, window_end=102.0))
        clock.value = 100.25
        self.assertTrue(
            source.confirm_scene_applied(
                scene_index=0,
                timestamp_monotonic=100.25,
            )
        )
        self.assertAlmostEqual(source.status()["valid_from_monotonic"], 100.25)
        self.assertIsNone(source.get_label(window_start=100.25, window_end=102.25))
        self.assertIsNotNone(source.get_label(window_start=100.35, window_end=102.35))

    def test_online_training_interval_matches_calibration_half_second_offsets(self) -> None:
        clock = _Clock(100.0)
        source = CuedOnlineLabelSource(
            ["left"],
            scene_duration_sec=5.0,
            start_delay_sec=0.0,
            boundary_guard_sec=0.5,
            clock=clock,
        )
        self.assertTrue(source.confirm_scene_applied(scene_index=0, timestamp_monotonic=100.0))

        clock.value = 104.5
        self.assertIsNone(source.get_label(window_start=100.49, window_end=102.49))
        self.assertIsNotNone(source.get_label(window_start=100.5, window_end=102.5))
        self.assertIsNotNone(source.get_label(window_start=102.5, window_end=104.5))
        self.assertIsNone(source.get_label(window_start=102.51, window_end=104.51))

    def test_collision_marks_failure_without_ending_fixed_scene(self) -> None:
        clock = _Clock(100.0)
        source = CuedOnlineLabelSource(
            ["left", "right", "idle"],
            scene_duration_sec=10.0,
            start_delay_sec=0.0,
            boundary_guard_sec=0.0,
            clock=clock,
        )
        self.assertTrue(source.confirm_scene_applied(scene_index=0, timestamp_monotonic=100.0))

        clock.value = 104.0
        self.assertTrue(
            source.mark_scene_failed(
                timestamp_monotonic=104.0,
                expected_scene_index=0,
            )
        )
        status = source.status()
        self.assertEqual(status["scene_index"], 0)
        self.assertEqual(status["label_name"], "left")
        self.assertTrue(status["scene_failed"])
        self.assertEqual(source.metadata()["failed_scenes"], 1)
        self.assertFalse(
            source.mark_scene_failed(
                timestamp_monotonic=104.1,
                expected_scene_index=0,
            )
        )

        clock.value = 110.0
        status = source.status()
        self.assertEqual(status["scene_index"], 1)
        self.assertEqual(status["label_name"], "right")
        self.assertFalse(status["scene_failed"])

    def test_buffered_collision_is_recorded_then_scene_times_out_normally(self) -> None:
        clock = _Clock(0.0)
        source = CuedOnlineLabelSource(
            ["left", "right", "idle"],
            scene_duration_sec=7.0,
            start_delay_sec=0.0,
            boundary_guard_sec=0.0,
            clock=clock,
        )

        clock.value = 7.1
        self.assertTrue(
            source.mark_scene_failed(
                timestamp_monotonic=7.1,
                expected_scene_index=0,
            )
        )
        status = source.status()
        self.assertEqual(status["scene_index"], 1)
        self.assertEqual(status["label_name"], "right")
        self.assertEqual(source.metadata()["failed_scenes"], 1)

    def test_scene_truth_is_required_before_label_is_accepted(self) -> None:
        clock = _Clock(100.0)
        source = CuedOnlineLabelSource(
            ["right"],
            scene_duration_sec=5.0,
            start_delay_sec=0.0,
            boundary_guard_sec=0.0,
            clock=clock,
        )
        outlet = _GameOutlet()
        decoder = RealTimeDecoder.__new__(RealTimeDecoder)
        decoder._online_label_source = source
        decoder._game_command_outlet = outlet
        decoder._last_game_transport_command = None
        decoder._last_game_transport_error = None
        decoder._last_game_transport_sent_at = 0.0
        decoder._last_game_movement_sent_at = 0.0
        decoder._stop_on_game_disconnect = False
        decoder._scene_sent_scene_index = -1
        decoder._scene_sent_label_id = None
        decoder._scene_sync_error = None
        decoder._console = type("_Console", (), {"print": lambda self, *args, **kwargs: None})()
        decoder._stop_event = type("_StopEvent", (), {"set": lambda self: None})()

        self.assertIsNone(decoder._get_online_label(window_start=100.0, window_end=102.0))
        from unittest import mock

        with mock.patch("decoder.real_time_decoder.time.monotonic", return_value=100.0):
            decoder._sync_game_scene()
        self.assertEqual(outlet.commands, ["SCENE_RIGHT"])
        label = decoder._get_online_label(window_start=100.0, window_end=102.0)
        self.assertIsNotNone(label)
        assert label is not None
        self.assertEqual(label.label_name, "right")

    def test_unity_failure_waits_for_fixed_boundary_before_next_scene(self) -> None:
        clock = _Clock(100.0)
        source = CuedOnlineLabelSource(
            ["left", "right"],
            scene_duration_sec=10.0,
            start_delay_sec=0.0,
            boundary_guard_sec=0.0,
            clock=clock,
        )
        outlet = _GameOutlet()
        decoder = RealTimeDecoder.__new__(RealTimeDecoder)
        decoder._online_label_source = source
        decoder._game_command_outlet = outlet
        decoder._last_game_transport_command = None
        decoder._last_game_transport_error = None
        decoder._last_game_transport_sent_at = 0.0
        decoder._last_game_movement_sent_at = 0.0
        decoder._stop_on_game_disconnect = False
        decoder._scene_sent_scene_index = -1
        decoder._scene_sent_label_id = None
        decoder._scene_sync_error = None
        decoder._console = type("_Console", (), {"print": lambda self, *args, **kwargs: None})()
        decoder._stop_event = type("_StopEvent", (), {"set": lambda self: None})()

        from unittest import mock

        with mock.patch("decoder.real_time_decoder.time.monotonic", return_value=100.0):
            decoder._sync_game_scene()
        self.assertEqual(outlet.commands, ["SCENE_LEFT"])

        clock.value = 103.0
        outlet.events.append({"event": "SCENE_FAILED", "scene_number": 1})
        with mock.patch("decoder.real_time_decoder.time.monotonic", return_value=103.0):
            decoder._sync_game_scene()

        self.assertEqual(outlet.commands, ["SCENE_LEFT"])
        self.assertEqual(source.status()["scene_index"], 0)
        self.assertTrue(source.status()["scene_failed"])
        self.assertEqual(source.metadata()["failed_scenes"], 1)

        clock.value = 110.0
        with mock.patch("decoder.real_time_decoder.time.monotonic", return_value=110.0):
            decoder._sync_game_scene()

        self.assertEqual(outlet.commands, ["SCENE_LEFT", "SCENE_RIGHT"])
        self.assertEqual(source.status()["scene_index"], 1)
        self.assertFalse(source.status()["scene_failed"])


if __name__ == "__main__":
    unittest.main()
