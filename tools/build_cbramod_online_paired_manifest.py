"""Build the source-aligned full-model online-update search manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_artifact(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Final training report is missing a checkpoint artifact.")
    path = Path(str(value.get("path", ""))).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Final checkpoint artifact does not exist: {path}")
    actual = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if actual != value:
        raise ValueError(f"Final checkpoint artifact hash mismatch: {path}")
    return actual


def validate_final_training_report(
    *,
    offline_summary: Path,
    final_training_report: Path,
) -> dict[str, Any]:
    summary = json.loads(offline_summary.read_text(encoding="utf-8"))
    report = json.loads(final_training_report.read_text(encoding="utf-8"))
    ranked = summary.get("ranked")
    if not isinstance(ranked, list) or not ranked:
        raise ValueError("Offline summary has no ranked candidates.")
    identity = report.get("identity", {})
    if identity.get("selected_candidate") != ranked[0].get("candidate"):
        raise ValueError("Final checkpoint was not trained from the best offline candidate.")
    reported_index = identity.get("selected_candidate_index")
    if reported_index is not None and int(reported_index) != int(
        ranked[0].get("candidate_index", -1)
    ):
        raise ValueError("Final checkpoint candidate index does not match offline rank 1.")
    if identity.get("offline_summary_sha256") != _sha256(offline_summary):
        raise ValueError("Final checkpoint is bound to a different offline summary.")
    artifacts = report.get("artifacts", {})
    return {
        "model": _verified_artifact(artifacts.get("model")),
        "neuroonline_sidecar": _verified_artifact(
            artifacts.get("neuroonline_sidecar")
        ),
    }


def build_candidates() -> list[dict[str, Any]]:
    return [
        {
            "update_policy": "full",
            "learning_rate": float(learning_rate),
            "history_threshold": 64,
            "epochs": 3,
            "batch_size": int(batch_size),
            "mask_ratio": float(mask_ratio),
            "consistency_weight": float(consistency_weight),
        }
        for learning_rate, batch_size, mask_ratio, consistency_weight in product(
            (1e-6, 1e-5, 3e-5, 1e-4, 3e-4),
            (16, 32, 64, 128),
            (0.1, 0.3, 0.5, 0.7),
            (0.1, 0.25, 0.5, 1.0),
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-summary", type=Path, required=True)
    parser.add_argument("--final-training-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.offline_summary.resolve()
    summary = json.loads(source.read_text(encoding="utf-8"))
    ranked = summary.get("ranked")
    if not isinstance(ranked, list) or not ranked:
        raise ValueError("Offline summary has no ranked candidates.")
    candidates = build_candidates()
    final_report = args.final_training_report.resolve()
    source_checkpoint = validate_final_training_report(
        offline_summary=source,
        final_training_report=final_report,
    )
    final_identity = json.loads(final_report.read_text(encoding="utf-8")).get(
        "identity", {}
    )
    checkpoint_contract = (
        "fixed_epoch_cv_rank1_hash_verified"
        if final_identity.get("fixed_epoch_cv") is True
        else "legacy_rank1_summary_hash_verified"
    )
    payload = {
        "schema_version": 1,
        "source_offline_summary": str(source),
        "source_offline_summary_sha256": _sha256(source),
        "source_final_training_report": str(final_report),
        "source_final_training_report_sha256": _sha256(final_report),
        "source_checkpoint": source_checkpoint,
        "source_checkpoint_provenance": {
            "contract": checkpoint_contract,
            "note": (
                "Rank-1 candidate, summary SHA256, and artifact hashes were verified; "
                "legacy reports may predate the fixed_epoch_cv marker."
            ),
        },
        "fixed_online": {
            "update_policy": "full",
            "optimizer": "AdamW",
            "weight_decay": 0.05,
            "history_threshold": 64,
            "update_stride": 64,
            "recent_samples": 320,
            "epochs": 3,
            "seeds": [17, 42, 2026],
            "incomplete_tail_update": False,
            "evaluation_protocol": "final_model_full_stream_resubstitution",
        },
        "grid": {
            "learning_rates": [1e-6, 1e-5, 3e-5, 1e-4, 3e-4],
            "batch_sizes": [16, 32, 64, 128],
            "mask_ratios": [0.1, 0.3, 0.5, 0.7],
            "consistency_weights": [0.1, 0.25, 0.5, 1.0],
        },
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(f"candidates={len(candidates)} output={args.output.resolve()}")


if __name__ == "__main__":
    main()
