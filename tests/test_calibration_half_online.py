from __future__ import annotations

import numpy as np
import pytest

from tools.evaluate_calibration_half_online import (
    chronological_trial_masks,
    expected_update_triggers,
    primary_window_mask,
)


def test_chronological_split_uses_planned_trial_midpoint_despite_rejection() -> None:
    trial_ids = np.repeat(np.array([0, 1, 3, 4, 5], dtype=np.int64), 2)

    offline, online, boundary = chronological_trial_masks(trial_ids)

    assert boundary == 3
    assert set(trial_ids[offline]) == {0, 1}
    assert set(trial_ids[online]) == {3, 4, 5}


def test_chronological_split_rejects_shuffled_trials() -> None:
    with pytest.raises(ValueError, match="chronological"):
        chronological_trial_masks(np.array([0, 2, 1, 3]))


def test_primary_window_selection_returns_one_per_available_trial() -> None:
    trials = np.repeat(np.arange(3, dtype=np.int64), 3)
    window_indices = np.tile(np.arange(3, dtype=np.int64), 3)

    mask = primary_window_mask(
        trials,
        window_indices,
        trials >= 1,
        primary_window_index=0,
    )

    assert trials[mask].tolist() == [1, 2]
    assert window_indices[mask].tolist() == [0, 0]


def test_30_primary_windows_cannot_trigger_the_64_window_policy() -> None:
    assert expected_update_triggers(
        30,
        history_threshold=64,
        update_stride=64,
    ) == []
    assert expected_update_triggers(
        130,
        history_threshold=64,
        update_stride=64,
    ) == [64, 128]
