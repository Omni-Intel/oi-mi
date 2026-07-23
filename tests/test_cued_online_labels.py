from __future__ import annotations

import unittest
import threading

from decoder.real_time_decoder import RealTimeDecoder
from utils.online_labels import CuedOnlineLabelSource


class _Clock:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class CuedOnlineLabelTests(unittest.TestCase):
    def test_only_fully_aligned_control_windows_receive_labels(self) -> None:
        clock = _Clock(100.0)
        source = CuedOnlineLabelSource(
            ["left", "right", "idle"],
            fixation_sec=2.0,
            cue_sec=1.0,
            control_sec=5.0,
            iti_sec=2.0,
            control_start_offset_sec=0.5,
            control_stop_offset_sec=4.5,
            start_delay_sec=0.0,
            clock=clock,
        )

        clock.value = 105.0
        self.assertIsNone(source.get_label(window_start=103.0, window_end=105.0))
        clock.value = 105.5
        label = source.get_label(window_start=103.5, window_end=105.5)
        self.assertIsNotNone(label)
        assert label is not None
        self.assertEqual(label.label_name, "left")
        self.assertEqual(label.event_id, "cue-000000")
        clock.value = 108.0
        self.assertIsNone(source.get_label(window_start=106.0, window_end=108.0))

    def test_status_tracks_trial_phase_and_target(self) -> None:
        clock = _Clock(10.0)
        source = CuedOnlineLabelSource(
            ["right"],
            fixation_sec=2.0,
            cue_sec=1.0,
            control_sec=5.0,
            iti_sec=2.0,
            control_start_offset_sec=0.5,
            control_stop_offset_sec=4.5,
            start_delay_sec=1.0,
            clock=clock,
        )
        self.assertEqual(source.status()["phase"], "preparing")
        clock.value = 13.5
        status = source.status()
        self.assertEqual(status["phase"], "cue")
        self.assertEqual(status["label_name"], "right")
        clock.value = 21.0
        self.assertEqual(source.status()["phase"], "done")

    def test_continuous_protocol_repeats_balanced_sequence_without_done(self) -> None:
        clock = _Clock(0.0)
        source = CuedOnlineLabelSource(
            ["left", "right", "idle"],
            fixation_sec=2.0,
            cue_sec=1.0,
            control_sec=5.0,
            iti_sec=2.0,
            control_start_offset_sec=0.5,
            control_stop_offset_sec=4.5,
            start_delay_sec=0.0,
            continuous=True,
            clock=clock,
        )

        clock.value = 30.0
        status = source.status()
        self.assertEqual(status["phase"], "fixation")
        self.assertEqual(status["trial_index"], 3)
        self.assertEqual(status["trial_number"], 4)
        self.assertNotIn("cycle_number", status)
        self.assertNotIn("total_trials", status)
        self.assertEqual(status["label_name"], "left")
        self.assertTrue(status["continuous"])
        self.assertEqual(source.metadata()["balance_pool_trials"], 3)
        self.assertNotIn("total_trials", source.metadata())

        clock.value = 35.5
        label = source.get_label(window_start=33.5, window_end=35.5)
        self.assertIsNotNone(label)
        assert label is not None
        self.assertEqual(label.label_name, "left")
        self.assertEqual(label.event_id, "cue-000003")

    def test_continuous_protocol_does_not_stop_decoder_at_cycle_boundary(self) -> None:
        clock = _Clock(0.0)
        source = CuedOnlineLabelSource(
            ["left"],
            fixation_sec=2.0,
            cue_sec=1.0,
            control_sec=5.0,
            iti_sec=2.0,
            control_start_offset_sec=0.5,
            control_stop_offset_sec=4.5,
            start_delay_sec=0.0,
            continuous=True,
            clock=clock,
        )
        decoder = RealTimeDecoder.__new__(RealTimeDecoder)
        decoder._online_label_source = source
        decoder._stop_event = threading.Event()

        clock.value = 10.0
        self.assertEqual(decoder._gate_game_command_for_protocol("LEFT"), "STOP")
        self.assertFalse(decoder._stop_event.is_set())
        clock.value = 14.0
        self.assertEqual(decoder._gate_game_command_for_protocol("LEFT"), "LEFT")
        self.assertFalse(decoder._stop_event.is_set())

    def test_car_commands_are_gated_to_control_phase(self) -> None:
        clock = _Clock(10.0)
        source = CuedOnlineLabelSource(
            ["left"],
            fixation_sec=2.0,
            cue_sec=1.0,
            control_sec=5.0,
            iti_sec=2.0,
            control_start_offset_sec=0.5,
            control_stop_offset_sec=4.5,
            start_delay_sec=0.0,
            clock=clock,
        )
        decoder = RealTimeDecoder.__new__(RealTimeDecoder)
        decoder._online_label_source = source
        decoder._stop_event = threading.Event()

        clock.value = 11.0
        self.assertEqual(decoder._gate_game_command_for_protocol("LEFT"), "STOP")
        clock.value = 14.0
        self.assertEqual(decoder._gate_game_command_for_protocol("LEFT"), "LEFT")
        clock.value = 20.0
        self.assertEqual(decoder._gate_game_command_for_protocol("LEFT"), "STOP")
        self.assertTrue(decoder._stop_event.is_set())


if __name__ == "__main__":
    unittest.main()
