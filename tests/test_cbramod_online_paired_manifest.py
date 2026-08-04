from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.build_cbramod_online_paired_manifest import (
    build_candidates,
    validate_final_training_report,
)


def test_online_search_has_320_source_aligned_full_model_candidates() -> None:
    candidates = build_candidates()

    assert len(candidates) == 320
    assert {candidate["update_policy"] for candidate in candidates} == {"full"}
    assert {candidate["learning_rate"] for candidate in candidates} == {
        1e-6,
        1e-5,
        3e-5,
        1e-4,
        3e-4,
    }
    assert all("head_learning_rate" not in candidate for candidate in candidates)
    assert all("backbone_learning_rate" not in candidate for candidate in candidates)
    assert {candidate["history_threshold"] for candidate in candidates} == {64}
    assert {candidate["epochs"] for candidate in candidates} == {3}
    assert {candidate["batch_size"] for candidate in candidates} == {
        16,
        32,
        64,
        128,
    }
    assert {candidate["mask_ratio"] for candidate in candidates} == {
        0.1,
        0.3,
        0.5,
        0.7,
    }
    assert {candidate["consistency_weight"] for candidate in candidates} == {
        0.1,
        0.25,
        0.5,
        1.0,
    }


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_final_checkpoint_must_be_bound_to_offline_rank_one(tmp_path: Path) -> None:
    candidate = {"update_policy": "full", "mask_ratio": 0.3}
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "ranked": [
                    {"candidate_index": 7, "candidate": candidate},
                ]
            }
        ),
        encoding="utf-8",
    )
    model_path = tmp_path / "model.pt"
    sidecar_path = tmp_path / "model.pt.neuroonline.pt"
    model_path.write_bytes(b"model")
    sidecar_path.write_bytes(b"sidecar")
    report_path = tmp_path / "training_report.json"
    report_path.write_text(
        json.dumps(
            {
                "identity": {
                    "selected_candidate": candidate,
                    "selected_candidate_index": 7,
                    "offline_summary_sha256": hashlib.sha256(
                        summary_path.read_bytes()
                    ).hexdigest(),
                    "fixed_epoch_cv": True,
                },
                "artifacts": {
                    "model": _artifact(model_path),
                    "neuroonline_sidecar": _artifact(sidecar_path),
                },
            }
        ),
        encoding="utf-8",
    )

    artifacts = validate_final_training_report(
        offline_summary=summary_path,
        final_training_report=report_path,
    )
    assert artifacts["model"] == _artifact(model_path)

    model_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_final_training_report(
            offline_summary=summary_path,
            final_training_report=report_path,
        )


def test_legacy_rank_one_report_without_fixed_epoch_marker_is_accepted(
    tmp_path: Path,
) -> None:
    candidate = {"update_policy": "full", "learning_rate": 1e-4}
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps({"ranked": [{"candidate_index": 25, "candidate": candidate}]}),
        encoding="utf-8",
    )
    model_path = tmp_path / "model.pt"
    sidecar_path = tmp_path / "model.pt.neuroonline.pt"
    model_path.write_bytes(b"model")
    sidecar_path.write_bytes(b"sidecar")
    report_path = tmp_path / "training_report.json"
    report_path.write_text(
        json.dumps(
            {
                "identity": {
                    "selected_candidate": candidate,
                    "offline_summary_sha256": hashlib.sha256(
                        summary_path.read_bytes()
                    ).hexdigest(),
                },
                "artifacts": {
                    "model": _artifact(model_path),
                    "neuroonline_sidecar": _artifact(sidecar_path),
                },
            }
        ),
        encoding="utf-8",
    )

    artifacts = validate_final_training_report(
        offline_summary=summary_path,
        final_training_report=report_path,
    )
    assert artifacts["neuroonline_sidecar"] == _artifact(sidecar_path)
