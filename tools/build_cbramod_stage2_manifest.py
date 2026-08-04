"""Expand the top offline policy/LR candidates into the stage-2 grid."""

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


def build_manifest(
    summary: dict[str, Any],
    *,
    top_k: int = 4,
    batch_sizes: tuple[int, ...] = (16, 32),
    mask_ratios: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7),
    consistency_weights: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0),
) -> list[dict[str, Any]]:
    ranked = summary.get("ranked")
    if not isinstance(ranked, list) or len(ranked) < top_k:
        raise ValueError(f"Stage-1 summary must contain at least {top_k} ranked candidates.")
    if top_k < 1:
        raise ValueError("top_k must be positive.")

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for entry in ranked[:top_k]:
        base = entry["candidate"]
        for batch_size, mask_ratio, consistency_weight in product(
            batch_sizes,
            mask_ratios,
            consistency_weights,
        ):
            candidate = {
                "update_policy": str(base["update_policy"]),
                "head_learning_rate": float(base["head_learning_rate"]),
                "backbone_learning_rate": (
                    None
                    if base.get("backbone_learning_rate") is None
                    else float(base["backbone_learning_rate"])
                ),
                "batch_size": int(batch_size),
                "mask_ratio": float(mask_ratio),
                "consistency_weight": float(consistency_weight),
            }
            key = tuple(candidate.values())
            if key not in seen:
                seen.add(key)
                candidates.append(candidate)
    return candidates


def _comma_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _comma_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--batch-sizes", type=_comma_ints, default=(16, 32))
    parser.add_argument("--mask-ratios", type=_comma_floats, default=(0.1, 0.3, 0.5, 0.7))
    parser.add_argument(
        "--lambda-grid",
        type=_comma_floats,
        default=(0.1, 0.25, 0.5, 1.0),
    )
    args = parser.parse_args()

    source = args.stage1_summary.resolve()
    summary = json.loads(source.read_text(encoding="utf-8"))
    candidates = build_manifest(
        summary,
        top_k=args.top_k,
        batch_sizes=args.batch_sizes,
        mask_ratios=args.mask_ratios,
        consistency_weights=args.lambda_grid,
    )
    payload = {
        "schema_version": 1,
        "source_stage1_summary": str(source),
        "source_stage1_sha256": _sha256(source),
        "top_k": args.top_k,
        "grid": {
            "batch_sizes": list(args.batch_sizes),
            "mask_ratios": list(args.mask_ratios),
            "consistency_weights": list(args.lambda_grid),
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
