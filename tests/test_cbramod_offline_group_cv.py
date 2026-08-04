from __future__ import annotations

import numpy as np

from tools.search_cbramod_offline_group_cv import (
    OFFLINE_SEARCH_CANDIDATES,
    _compatible_identity,
    _grouped_splits,
    _ranking_score,
    _summarize_runs,
)


def test_offline_search_is_one_exhaustive_source_style_grid() -> None:
    assert len(OFFLINE_SEARCH_CANDIDATES) == 320
    assert {candidate["update_policy"] for candidate in OFFLINE_SEARCH_CANDIDATES} == {
        "full"
    }
    assert {
        candidate["head_learning_rate"]
        for candidate in OFFLINE_SEARCH_CANDIDATES
    } == {1e-6, 1e-5, 3e-5, 1e-4, 3e-4}
    assert all(
        candidate["head_learning_rate"] == candidate["backbone_learning_rate"]
        for candidate in OFFLINE_SEARCH_CANDIDATES
    )
    assert {candidate["batch_size"] for candidate in OFFLINE_SEARCH_CANDIDATES} == {
        16,
        32,
        64,
        128,
    }
    assert {candidate["mask_ratio"] for candidate in OFFLINE_SEARCH_CANDIDATES} == {
        0.1,
        0.3,
        0.5,
        0.7,
    }
    assert {
        candidate["consistency_weight"]
        for candidate in OFFLINE_SEARCH_CANDIDATES
    } == {0.1, 0.25, 0.5, 1.0}


def test_grouped_splits_keep_all_windows_from_each_trial_together() -> None:
    groups = np.repeat(np.arange(15, dtype=np.int64), 5)
    labels = np.repeat(np.tile(np.arange(3, dtype=np.int64), 5), 5)
    windows = np.zeros((groups.size, 2, 16), dtype=np.float32)

    splits = _grouped_splits(windows, labels, groups, folds=5, seed=42)

    assert len(splits) == 5
    for train_indices, validation_indices in splits:
        train_trials = set(groups[train_indices].tolist())
        validation_trials = set(groups[validation_indices].tolist())
        assert train_trials.isdisjoint(validation_trials)
        assert train_trials | validation_trials == set(range(15))


def test_ranking_prioritizes_no_collapse_then_worst_class_then_stability() -> None:
    def report(*, collapsed: int, worst: float, mean: float, std: float) -> dict:
        return {
            "summary": {
                "collapsed_runs": collapsed,
                "mean_trial_worst_class_recall": worst,
                "mean_trial_balanced_accuracy": mean,
                "std_trial_balanced_accuracy": std,
            }
        }

    collapsed_high_accuracy = report(collapsed=1, worst=0.9, mean=0.9, std=0.01)
    stable = report(collapsed=0, worst=0.4, mean=0.6, std=0.02)
    unstable = report(collapsed=0, worst=0.4, mean=0.65, std=0.20)
    weak_worst_class = report(collapsed=0, worst=0.3, mean=0.8, std=0.01)

    ranked = sorted(
        [collapsed_high_accuracy, unstable, weak_worst_class, stable],
        key=_ranking_score,
        reverse=True,
    )

    assert ranked == [stable, unstable, weak_worst_class, collapsed_high_accuracy]


def test_summary_includes_collapsed_folds_in_primary_mean() -> None:
    runs = [
        {
            "class_collapse": False,
            "metrics": {
                "val_trial_balanced_accuracy": 0.6,
                "val_trial_worst_class_accuracy": 0.2,
                "best_epoch": 11.0,
            },
        },
        {
            "class_collapse": True,
            "metrics": {
                "val_trial_balanced_accuracy": 0.3,
                "val_trial_worst_class_accuracy": 0.0,
            },
        },
    ]

    summary = _summarize_runs(runs)

    assert summary["collapsed_runs"] == 1
    assert np.isclose(summary["mean_trial_balanced_accuracy"], 0.45)
    assert summary["mean_noncollapsed_trial_balanced_accuracy"] == 0.6
    assert np.isclose(summary["mean_trial_worst_class_recall"], 0.1)
    assert summary["median_best_epoch"] == 11


def test_resume_identity_allows_same_dataset_hash_at_a_new_path() -> None:
    existing = {
        "schema_version": 1,
        "dataset": "/old/server/training_windows_main.npz",
        "dataset_sha256": "abc123",
        "candidate_index": 7,
        "candidate": {"update_policy": "full"},
    }
    current = {
        **existing,
        "dataset": "/data/new/server/training_windows_main.npz",
    }

    assert _compatible_identity(existing, current)
    assert not _compatible_identity(
        existing,
        {**current, "dataset_sha256": "different"},
    )
