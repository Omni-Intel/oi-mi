"""Expand the top paired online configurations across mask and lambda."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any


def build_candidates(
    summary: dict[str, Any],
    *,
    top_k: int = 4,
    mask_ratios: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7),
    consistency_weights: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0),
) -> list[dict[str, Any]]:
    ranked = summary.get("ranked")
    if not isinstance(ranked, list) or len(ranked) < top_k:
        raise ValueError(f"Online stage-1 summary must contain at least {top_k} candidates.")
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for entry in ranked[:top_k]:
        base = dict(entry["candidate"])
        base.pop("source_reference", None)
        for mask_ratio, consistency_weight in product(
            mask_ratios,
            consistency_weights,
        ):
            candidate = {
                **base,
                "mask_ratio": float(mask_ratio),
                "consistency_weight": float(consistency_weight),
                "source_reference": False,
            }
            key = tuple(candidate.items())
            if key not in seen:
                seen.add(key)
                candidates.append(candidate)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()
    source = args.stage1_summary.resolve()
    summary = json.loads(source.read_text(encoding="utf-8"))
    source_checkpoint = summary.get("source_checkpoint")
    if not isinstance(source_checkpoint, dict):
        raise ValueError("Online stage-1 summary is missing its source checkpoint.")
    candidates = build_candidates(summary, top_k=args.top_k)
    payload = {
        "schema_version": 1,
        "source_online_stage1_summary": str(source),
        "source_checkpoint": source_checkpoint,
        "top_k": args.top_k,
        "grid": {
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
