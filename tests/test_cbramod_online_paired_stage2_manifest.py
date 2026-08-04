from __future__ import annotations

from tools.build_cbramod_online_paired_stage2_manifest import build_candidates


def test_online_stage2_expands_top_four_to_64_candidates() -> None:
    summary = {
        "ranked": [
            {
                "candidate": {
                    "update_policy": "head",
                    "head_learning_rate": float(index + 1) * 1e-5,
                    "backbone_learning_rate": None,
                    "history_threshold": 64,
                    "epochs": 1,
                    "batch_size": 16,
                    "mask_ratio": 0.3,
                    "consistency_weight": 0.5,
                    "source_reference": False,
                }
            }
            for index in range(4)
        ]
    }

    candidates = build_candidates(summary)

    assert len(candidates) == 64
    assert {candidate["mask_ratio"] for candidate in candidates} == {0.1, 0.3, 0.5, 0.7}
    assert {candidate["consistency_weight"] for candidate in candidates} == {
        0.1,
        0.25,
        0.5,
        1.0,
    }
