"""Merge disjoint NeuroOnline online-search shard reports for final replay."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.search_neuroonline_online_hyperparams import _resume_identity, _score


def candidate_key(candidate: dict[str, Any]) -> tuple[float, int, int, float, float]:
    config = candidate["config"]
    return (
        float(config["learning_rate"]),
        int(config["epochs"]),
        int(config["update_batch_size"]),
        float(config["mask_ratio"]),
        float(config["consistency_weight"]),
    )


def merge_reports(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("No shard reports were provided.")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    identity = _resume_identity(reports[0])
    for path, report in zip(paths[1:], reports[1:], strict=True):
        if _resume_identity(report) != identity:
            raise ValueError(f"Shard report has incompatible inputs or grid: {path}")

    candidates: dict[tuple[float, int, int, float, float], dict[str, Any]] = {}
    for path, report in zip(paths, reports, strict=True):
        for candidate in report.get("candidates", []):
            key = candidate_key(candidate)
            if key in candidates and candidates[key] != candidate:
                raise ValueError(f"Conflicting duplicate candidate {key} in {path}")
            candidates[key] = candidate

    merged = copy.deepcopy(reports[0])
    merged_candidates = sorted(candidates.values(), key=lambda item: int(item["candidate_index"]))
    expected = int(merged["search_space"]["total_candidates"])
    merged["requested_candidate_range"] = [0, expected]
    merged["candidates"] = merged_candidates
    merged["completed_candidates"] = len(merged_candidates)
    merged["search_complete"] = len(merged_candidates) == expected
    successful = [
        candidate
        for candidate in merged_candidates
        if not candidate.get("failed") and "validation" in candidate
    ]
    if not successful:
        raise RuntimeError("Every merged candidate failed.")
    selected = max(successful, key=_score)
    merged["selected"] = {
        **selected["config"],
        "validation": selected["validation"],
        "search_prefix": selected["search_prefix"],
    }
    merged.pop("selected_final_replay", None)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.input_dir.glob("shard_*.json"))
    merged = merge_reports(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"merged={merged['completed_candidates']}/"
        f"{merged['search_space']['total_candidates']} "
        f"complete={merged['search_complete']}"
    )


if __name__ == "__main__":
    main()
