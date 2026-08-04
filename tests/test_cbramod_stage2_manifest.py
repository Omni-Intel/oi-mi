from __future__ import annotations

from tools.build_cbramod_stage2_manifest import build_manifest


def _summary() -> dict:
    learning_rates = (1e-4, 2e-4, 3e-4, 4e-4, 5e-4)
    return {
        "ranked": [
            {
                "rank": index + 1,
                "candidate": {
                    "update_policy": "head" if index < 2 else "full",
                    "head_learning_rate": learning_rates[index],
                    "backbone_learning_rate": None if index < 2 else learning_rates[index],
                },
            }
            for index in range(5)
        ]
    }


def test_stage2_expands_top_four_to_128_candidates() -> None:
    candidates = build_manifest(_summary())

    assert len(candidates) == 128
    assert {candidate["batch_size"] for candidate in candidates} == {16, 32}
    assert {candidate["mask_ratio"] for candidate in candidates} == {0.1, 0.3, 0.5, 0.7}
    assert {candidate["consistency_weight"] for candidate in candidates} == {
        0.1,
        0.25,
        0.5,
        1.0,
    }
    assert {candidate["head_learning_rate"] for candidate in candidates} == {
        1e-4,
        2e-4,
        3e-4,
        4e-4,
    }


def test_stage2_requires_enough_ranked_candidates() -> None:
    try:
        build_manifest({"ranked": []})
    except ValueError as exc:
        assert "at least 4" in str(exc)
    else:
        raise AssertionError("Expected a ValueError for an incomplete stage-1 summary.")
