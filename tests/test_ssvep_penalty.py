from __future__ import annotations

import numpy as np

from ssvep.detector import DEFAULT_TARGETS, SSVEPDetector
from ssvep.penalty_game import PenaltyGame, team_by_name


def test_ssvep_detector_selects_synthetic_12hz_target() -> None:
    sfreq = 250.0
    duration_sec = 2.0
    t = np.arange(int(sfreq * duration_sec), dtype=np.float32) / sfreq
    signal = np.sin(2 * np.pi * 12.0 * t) + 0.5 * np.sin(2 * np.pi * 24.0 * t)
    gains = np.linspace(0.75, 1.25, 32, dtype=np.float32)
    eeg = (gains[:, None] * signal[None, :]).astype(np.float32)

    detector = SSVEPDetector(
        sfreq=sfreq,
        targets=DEFAULT_TARGETS,
        channel_indices=tuple(range(32)),
        stability_windows=1,
        min_confidence=0.1,
    )
    result = detector.predict(eeg)

    assert result.target is not None
    assert result.target.direction == "top_right"
    assert result.stable


def test_penalty_game_completes_after_configured_rounds() -> None:
    game = PenaltyGame(
        user_team=team_by_name("China"),
        opponent_team=team_by_name("Brazil"),
        rounds=2,
        keeper_save_probability=0.0,
        opponent_goal_probability=0.0,
        seed=7,
    )

    first = game.shoot("top_left")
    second = game.shoot("bottom_right")

    assert first.goal
    assert second.complete
    assert game.user_score == 2
    assert game.opponent_score == 0
