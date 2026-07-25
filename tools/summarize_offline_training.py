"""Aggregate the fixed three-seed offline experiment into one JSON report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _aggregate(
    payload: dict[str, Any],
    *,
    method_key: str | None,
    split: str,
) -> dict[str, Any]:
    runs = payload["runs"]
    if method_key is not None:
        runs = runs[method_key]
    seeds = sorted(runs, key=int)
    output: dict[str, Any] = {"seeds": [int(seed) for seed in seeds]}
    for level in ("window", "trial"):
        output[level] = {}
        for metric in ("accuracy", "balanced_accuracy", "macro_f1", "kappa"):
            values = np.asarray(
                [runs[seed][split][level][metric] for seed in seeds],
                dtype=float,
            )
            output[level][metric] = {
                "values": values.tolist(),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
            }
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    formal: dict[str, dict[str, Any]] = {}
    day2: dict[str, dict[str, Any]] = {}
    for name in ("baseline", "neuroonline", "neuroonline_lambda0"):
        formal_path = args.root / "03_formal" / name / "summary.json"
        diagnostic_path = args.root / "04_day2_diagnostic" / name / "summary.json"
        if formal_path.exists():
            formal[name] = _load(formal_path)
        if diagnostic_path.exists():
            day2[name] = _load(diagnostic_path)
    output: dict[str, Any] = {
        "standard_deviation": "population standard deviation across seeds",
        "d1_same_day_test_block5": {},
        "d1_to_d2_static_cross_day": {},
        "d2_same_day_chronological_test": {},
    }
    for name, payload in formal.items():
        method_key = "baseline" if name == "baseline" else "neuroonline"
        output["d1_same_day_test_block5"][name] = _aggregate(
            payload,
            method_key=method_key,
            split="d1_test_block5",
        )
        output["d1_to_d2_static_cross_day"][name] = _aggregate(
            payload,
            method_key=method_key,
            split="d2_static_cross_day",
        )
    for name, payload in day2.items():
        output["d2_same_day_chronological_test"][name] = _aggregate(
            payload,
            method_key=None,
            split="d2_test_chronological",
        )
    target = args.root / "aggregate_summary.json"
    target.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(_parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
