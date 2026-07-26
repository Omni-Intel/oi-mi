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
    def __init__(self, *, current_lane: int = 0) -> None:
        self.commands: list[str] = []
        self.events: list[dict[str, object]] = []
        self.current_lane = current_lane
        self.next_scene_number = 1

    def push(self, command: str) -> None:
        self.commands.append(command)

    def push_with_ack(self, command: str) -> dict[str, object]:
        self.push(command)
        if command == "SCENE_STATE":
            return {
                "ack": command,
                "protocol_version": "continuous-scene-v4-dynamic-label",
                "scene_number": self.next_scene_number,
                "current_lane": self.current_lane,
            }
        label_by_command = {
            "SCENE_LEFT": ("left", -1),
            "SCENE_RIGHT": ("right", 1),
            "SCENE_IDLE": ("idle", 0),
        }
        label, delta = label_by_command[command]
        safe_lane = self.current_lane + delta
        if safe_lane not in {-1, 0, 1}:
            raise RuntimeError("unreachable relative action")
        response = {
            "ack": command,
            "protocol_version": "continuous-scene-v4-dynamic-label",
            "scene_number": self.next_scene_number,
            "start_lane": self.current_lane,
            "safe_lane": safe_lane,
            "applied_label": label,
        }
        self.next_scene_number += 1
        return response

    def poll_events(self) -> list[dict[str, object]]:
        events = list(self.events)
        self.events.clear()
        return events


def _bare_decoder(
    source: CuedOnlineLabelSource,
    outlet: _GameOutlet,
) -> RealTimeDecoder:
    decoder = RealTimeDecoder.__new__(RealTimeDecoder)
    decoder._online_label_source = source
    decoder._game_command_outlet = outlet
    decoder._last_game_transport_command = None
    decoder._last_game_transport_error = None
    decoder._last_game_transport_sent_at = 0.0
    decoder._last_game_movement_sent_at = 0.0
    decoder._stop_on_game_disconnect = True
    decoder._scene_sent_scene_index = -1
    decoder._scene_sent_label_id = None
    decoder._scene_sync_error = None
    decoder._game_disconnect_message = None
    decoder._console = type(
        "_Console",
        (),
        {"print": lambda self, *args, **kwargs: None},
    )()
    decoder._stop_event = type(
        "_StopEvent",
        (),
        {
            "__init__": lambda self: setattr(self, "was_set", False),
            "set": lambda self: setattr(self, "was_set", True),
        },
    )()
    return decoder


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
        self.assertEqual(source.prepare_scene(scene_index=0, start_lane=0), 0)
        self.assertTrue(source.confirm_scene_applied(
            scene_index=0,
            applied_label_id=0,
            start_lane=0,
            safe_lane=-1,
            timestamp_monotonic=100.0,
        ))

        clock.value = 102.0
        label = source.get_label(window_start=100.0, window_end=102.0)
        self.assertIsNotNone(label)
        assert label is not None
        self.assertEqual(label.label_name, "left")
        self.assertEqual(label.event_id, "scene-000000-segment-000")

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
        self.assertEqual(status["protocol_mode"], "continuous-relative-action")
        self.assertEqual(status["scene_index"], 6)
        self.assertEqual(status["scene_number"], 7)
        self.assertIsNone(status["label_name"])
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
        self.assertEqual(source.prepare_scene(scene_index=0, start_lane=0), 0)
        self.assertTrue(
            source.confirm_scene_applied(
                scene_index=0,
                applied_label_id=0,
                start_lane=0,
                safe_lane=-1,
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
        self.assertEqual(source.prepare_scene(scene_index=0, start_lane=0), 0)
        self.assertTrue(source.confirm_scene_applied(
            scene_index=0,
            applied_label_id=0,
            start_lane=0,
            safe_lane=-1,
            timestamp_monotonic=100.0,
        ))

        clock.value = 104.5
        self.assertIsNone(source.get_label(window_start=100.49, window_end=102.49))
        self.assertIsNotNone(source.get_label(window_start=100.5, window_end=102.5))
        self.assertIsNotNone(source.get_label(window_start=102.5, window_end=104.5))
        self.assertIsNone(source.get_label(window_start=102.51, window_end=104.51))

    def test_lane_settled_splits_truth_and_rejects_crossing_windows(self) -> None:
        clock = _Clock(100.0)
        source = CuedOnlineLabelSource(
            ["left"],
            scene_duration_sec=5.0,
            start_delay_sec=0.0,
            boundary_guard_sec=0.0,
            clock=clock,
        )
        self.assertEqual(source.prepare_scene(scene_index=0, start_lane=0), 0)
        self.assertTrue(source.confirm_scene_applied(
            scene_index=0,
            applied_label_id=0,
            start_lane=0,
            safe_lane=-1,
            timestamp_monotonic=100.0,
        ))
        self.assertTrue(source.update_current_lane(
            scene_index=0,
            current_lane=-1,
            safe_lane=-1,
            timestamp_monotonic=102.0,
        ))

        left = source.get_label(window_start=100.0, window_end=102.0)
        crossing = source.get_label(window_start=101.0, window_end=103.0)
        idle = source.get_label(window_start=102.0, window_end=104.0)
        self.assertIsNotNone(left)
        self.assertEqual(left.label_name, "left")
        self.assertIsNone(crossing)
        self.assertIsNotNone(idle)
        self.assertEqual(idle.label_name, "idle")
        self.assertEqual(idle.payload["current_lane"], -1)
        self.assertEqual(source.metadata()["label_transition_count"], 1)

    def test_decoder_accepts_dynamic_lane_truth_from_unity_event(self) -> None:
        clock = _Clock(100.0)
        source = CuedOnlineLabelSource(
            ["left"],
            scene_duration_sec=5.0,
            start_delay_sec=0.0,
            boundary_guard_sec=0.0,
            clock=clock,
        )
        outlet = _GameOutlet(current_lane=0)
        decoder = _bare_decoder(source, outlet)

        from unittest import mock

        with mock.patch("decoder.real_time_decoder.time.monotonic", return_value=100.0):
            decoder._sync_game_scene()
        outlet.events.append(
            {
                "event": "LANE_SETTLED",
                "protocol_version": "continuous-scene-v4-dynamic-label",
                "scene_number": 1,
                "current_lane": -1,
                "safe_lane": -1,
            }
        )
        clock.value = 102.0
        with mock.patch("decoder.real_time_decoder.time.monotonic", return_value=102.0):
            decoder._consume_game_scene_events()

        self.assertIsNone(
            decoder._get_online_label(window_start=101.0, window_end=103.0)
        )
        idle = decoder._get_online_label(window_start=102.0, window_end=104.0)
        self.assertIsNotNone(idle)
        self.assertEqual(idle.label_name, "idle")

    def test_collision_marks_failure_without_ending_fixed_scene(self) -> None:
        clock = _Clock(100.0)
        source = CuedOnlineLabelSource(
            ["left", "right", "idle"],
            scene_duration_sec=10.0,
            start_delay_sec=0.0,
            boundary_guard_sec=0.0,
            clock=clock,
        )
        self.assertEqual(source.prepare_scene(scene_index=0, start_lane=0), 0)
        self.assertTrue(source.confirm_scene_applied(
            scene_index=0,
            applied_label_id=0,
            start_lane=0,
            safe_lane=-1,
            timestamp_monotonic=100.0,
        ))

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
        self.assertIsNone(status["label_name"])
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
        self.assertIsNone(status["label_name"])
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
        self.assertEqual(outlet.commands, ["SCENE_STATE", "SCENE_RIGHT"])
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
        self.assertEqual(outlet.commands, ["SCENE_STATE", "SCENE_LEFT"])

        clock.value = 103.0
        outlet.events.append({"event": "SCENE_FAILED", "scene_number": 1})
        with mock.patch("decoder.real_time_decoder.time.monotonic", return_value=103.0):
            decoder._sync_game_scene()

        self.assertEqual(outlet.commands, ["SCENE_STATE", "SCENE_LEFT"])
        self.assertEqual(source.status()["scene_index"], 0)
        self.assertTrue(source.status()["scene_failed"])
        self.assertEqual(source.metadata()["failed_scenes"], 1)

        clock.value = 110.0
        outlet.current_lane = -1
        with mock.patch("decoder.real_time_decoder.time.monotonic", return_value=110.0):
            decoder._sync_game_scene()

        self.assertEqual(
            outlet.commands,
            ["SCENE_STATE", "SCENE_LEFT", "SCENE_STATE", "SCENE_RIGHT"],
        )
        self.assertEqual(source.status()["scene_index"], 1)
        self.assertFalse(source.status()["scene_failed"])

    def test_repeated_empty_direction_uses_actual_lane_after_failure(self) -> None:
        clock = _Clock(0.0)
        source = CuedOnlineLabelSource(
            ["right", "right", "idle"],
            scene_duration_sec=5.0,
            start_delay_sec=0.0,
            boundary_guard_sec=0.0,
            clock=clock,
        )
        outlet = _GameOutlet(current_lane=-1)
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

        with mock.patch("decoder.real_time_decoder.time.monotonic", return_value=0.0):
            decoder._sync_game_scene()
        self.assertEqual(decoder._scene_labels[0], 1)
        self.assertEqual(decoder._scene_start_lanes[0], -1)
        self.assertEqual(decoder._scene_safe_lanes[0], 0)

        clock.value = 5.0
        outlet.current_lane = -1  # The car failed and never left its lane.
        with mock.patch("decoder.real_time_decoder.time.monotonic", return_value=5.0):
            decoder._sync_game_scene()
        self.assertEqual(decoder._scene_start_lanes[1], -1)
        self.assertEqual(decoder._scene_labels[1], 1)
        self.assertEqual(decoder._scene_safe_lanes[1], 0)

    def test_96_scene_pool_stays_exactly_balanced_at_lane_boundaries(self) -> None:
        for reaches_safe_lane in (False, True):
            with self.subTest(reaches_safe_lane=reaches_safe_lane):
                clock = _Clock(0.0)
                source = CuedOnlineLabelSource(
                    ["left"] * 32 + ["right"] * 32 + ["idle"] * 32,
                    scene_duration_sec=5.0,
                    start_delay_sec=0.0,
                    boundary_guard_sec=0.0,
                    clock=clock,
                )
                lane = 0
                counts = {0: 0, 1: 0, 2: 0}
                for scene_index in range(96):
                    source.status()
                    label = source.prepare_scene(
                        scene_index=scene_index,
                        start_lane=lane,
                    )
                    safe_lane = lane + {0: -1, 1: 1, 2: 0}[label]
                    self.assertIn(safe_lane, {-1, 0, 1})
                    self.assertTrue(
                        source.confirm_scene_applied(
                            scene_index=scene_index,
                            applied_label_id=label,
                            start_lane=lane,
                            safe_lane=safe_lane,
                            timestamp_monotonic=clock.value,
                        )
                    )
                    counts[label] += 1
                    if reaches_safe_lane:
                        lane = safe_lane
                    clock.value += 5.0

                self.assertEqual(counts, {0: 32, 1: 32, 2: 32})

    def test_wrong_unity_safe_lane_aborts_without_creating_training_label(self) -> None:
        class _WrongSafeLaneOutlet(_GameOutlet):
            def push_with_ack(self, command: str) -> dict[str, object]:
                response = super().push_with_ack(command)
                if command != "SCENE_STATE":
                    response["safe_lane"] = 1
                return response

        clock = _Clock(0.0)
        source = CuedOnlineLabelSource(
            ["left"],
            scene_duration_sec=5.0,
            start_delay_sec=0.0,
            boundary_guard_sec=0.0,
            clock=clock,
        )
        decoder = _bare_decoder(source, _WrongSafeLaneOutlet(current_lane=0))

        from unittest import mock

        with mock.patch("decoder.real_time_decoder.time.monotonic", return_value=0.0):
            decoder._sync_game_scene()

        self.assertTrue(decoder._stop_event.was_set)
        self.assertIn("safe", decoder._game_disconnect_message.lower())
        self.assertIsNone(
            decoder._get_online_label(window_start=0.0, window_end=2.0)
        )

    def test_old_or_wrong_unity_protocol_aborts_before_scene_command(self) -> None:
        class _OldProtocolOutlet(_GameOutlet):
            def push_with_ack(self, command: str) -> dict[str, object]:
                response = super().push_with_ack(command)
                response["protocol_version"] = "continuous-scene-v2"
                return response

        clock = _Clock(0.0)
        source = CuedOnlineLabelSource(
            ["right"],
            scene_duration_sec=5.0,
            start_delay_sec=0.0,
            boundary_guard_sec=0.0,
            clock=clock,
        )
        outlet = _OldProtocolOutlet(current_lane=0)
        decoder = _bare_decoder(source, outlet)

        from unittest import mock

        with mock.patch("decoder.real_time_decoder.time.monotonic", return_value=0.0):
            decoder._sync_game_scene()

        self.assertEqual(outlet.commands, ["SCENE_STATE"])
        self.assertTrue(decoder._stop_event.was_set)
        self.assertEqual(decoder._scene_sent_scene_index, -1)
        self.assertIsNone(
            decoder._get_online_label(window_start=0.0, window_end=2.0)
        )


if __name__ == "__main__":
    unittest.main()
