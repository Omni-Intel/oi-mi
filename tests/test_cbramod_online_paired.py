from __future__ import annotations

import numpy as np
import pytest

from tools.search_cbramod_online_paired import (
    _compatible_legacy_mechanics,
    _paired_metrics,
    _ranking_score,
    _validate_checkpoint_binding,
)


def test_paired_metrics_reports_candidate_minus_baseline() -> None:
    truth = np.array([0, 1, 2, 0, 1, 2])
    baseline = np.eye(3, dtype=np.float32)[np.array([0, 0, 0, 0, 0, 0])]
    candidate = np.eye(3, dtype=np.float32)[truth]

    metrics = _paired_metrics(truth, candidate, baseline, n_classes=3)

    assert metrics["candidate"]["balanced_accuracy"] == 1.0
    assert metrics["fixed_baseline"]["balanced_accuracy"] == 1 / 3
    assert np.isclose(metrics["delta_balanced_accuracy"], 2 / 3)


def test_online_ranking_rejects_collapse_before_accuracy_gain() -> None:
    def report(collapsed: int, mean: float, minimum: float, std: float) -> dict:
        return {
            "summary": {
                "collapsed_runs": collapsed,
                "mean_delta_balanced_accuracy": mean,
                "minimum_delta_balanced_accuracy": minimum,
                "std_delta_balanced_accuracy": std,
                "mean_worst_class_recall": 0.2,
            }
        }

    stable = report(0, 0.04, 0.02, 0.01)
    unstable = report(0, 0.05, -0.05, 0.10)
    collapsed = report(1, 0.30, 0.20, 0.01)

    ranked = sorted([collapsed, stable, unstable], key=_ranking_score, reverse=True)

    assert ranked == [unstable, stable, collapsed]


def test_online_checkpoint_must_match_manifest_binding() -> None:
    expected = {"model": {"sha256": "abc"}, "neuroonline_sidecar": {"sha256": "def"}}
    _validate_checkpoint_binding({"source_checkpoint": expected}, expected)

    with pytest.raises(ValueError, match="does not match"):
        _validate_checkpoint_binding(
            {"source_checkpoint": expected},
            {"model": {"sha256": "other"}},
        )


def test_only_hash_verified_legacy_manifest_allows_mechanics_v2() -> None:
    assert _compatible_legacy_mechanics(
        {
            "source_checkpoint_provenance": {
                "contract": "legacy_rank1_summary_hash_verified"
            }
        }
    ) == (2,)
    assert _compatible_legacy_mechanics({}) == ()
    assert _compatible_legacy_mechanics(
        {
            "source_checkpoint_provenance": {
                "contract": "fixed_epoch_cv_rank1_hash_verified"
            }
        }
    ) == ()
