"""Run the agreed staged offline hyperparameter search without test-set leakage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


BASELINE_BATCH_SIZES = (8, 16, 32)
BASELINE_LEARNING_RATES = (3e-5, 1e-4, 3e-4)
NEUROONLINE_MASK_RATIOS = (0.15, 0.30, 0.50)
NEUROONLINE_LAMBDAS = (0.0, 0.03, 0.10, 0.30)


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _completed(output: Path, method: str, seed: int) -> bool:
    summary_path = output / "summary.json"
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = summary["runs"][method][str(seed)]["d1_validation_block4"]["trial"]
        return "kappa" in metrics and "macro_f1" in metrics
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _run_configuration(
    *,
    day1: Path,
    output: Path,
    method: str,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    mask_ratio: float = 0.3,
    consistency_weight: float = 0.1,
) -> None:
    if _completed(output, method, seed):
        print(f"SKIP completed: {output.name}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "tools.train_offline_comparison",
        "--day1",
        str(day1),
        "--output",
        str(output),
        "--evaluation-scope",
        "selection",
        "--methods",
        method,
        "--seeds",
        str(seed),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--learning-rate",
        str(learning_rate),
        "--mask-ratio",
        str(mask_ratio),
        "--consistency-weight",
        str(consistency_weight),
    ]
    print(f"START {output.name}", flush=True)
    with (output / "console.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Training failed for {output.name}; see {output / 'console.log'}."
        )
    print(f"DONE  {output.name}", flush=True)


def _validation_row(output: Path, method: str, seed: int) -> dict[str, Any]:
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    trial = summary["runs"][method][str(seed)]["d1_validation_block4"]["trial"]
    history = json.loads(
        (output / method / f"seed_{seed}" / "history.json").read_text(encoding="utf-8")
    )
    return {
        "output": str(output.resolve()),
        "kappa": float(trial["kappa"]),
        "macro_f1": float(trial["macro_f1"]),
        "balanced_accuracy": float(trial["balanced_accuracy"]),
        "accuracy": float(trial["accuracy"]),
        "best_epoch": int(
            max(
                history,
                key=lambda item: (
                    item["validation_trial"]["kappa"],
                    item["validation_trial"]["macro_f1"],
                ),
            )["epoch"]
        ),
        "epochs_ran": len(history),
    }


def _rank(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (row["kappa"], row["macro_f1"]),
        reverse=True,
    )


def run_baseline_selection(args: argparse.Namespace) -> list[dict[str, Any]]:
    root = args.output / "01_baseline_selection"
    rows: list[dict[str, Any]] = []
    for batch_size in BASELINE_BATCH_SIZES:
        for learning_rate in BASELINE_LEARNING_RATES:
            name = f"bs{batch_size}_lr{learning_rate:.0e}"
            output = root / name
            _run_configuration(
                day1=args.day1,
                output=output,
                method="baseline",
                seed=args.selection_seed,
                epochs=args.epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
            )
            row = _validation_row(output, "baseline", args.selection_seed)
            row.update(
                {
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "seed": args.selection_seed,
                }
            )
            rows.append(row)
            _save_json(root / "ranking.json", _rank(rows))
    ranked = _rank(rows)
    print(json.dumps({"best_baseline": ranked[0]}, ensure_ascii=False, indent=2))
    return ranked


def run_neuroonline_selection(
    args: argparse.Namespace,
    baseline_ranking: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    best_baseline = baseline_ranking[0]
    batch_size = int(best_baseline["batch_size"])
    learning_rate = float(best_baseline["learning_rate"])
    root = args.output / "02_neuroonline_selection"
    rows: list[dict[str, Any]] = []
    for mask_ratio in NEUROONLINE_MASK_RATIOS:
        for consistency_weight in NEUROONLINE_LAMBDAS:
            name = f"mask{mask_ratio:.2f}_lambda{consistency_weight:.2f}"
            output = root / name
            _run_configuration(
                day1=args.day1,
                output=output,
                method="neuroonline",
                seed=args.selection_seed,
                epochs=args.epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                mask_ratio=mask_ratio,
                consistency_weight=consistency_weight,
            )
            row = _validation_row(output, "neuroonline", args.selection_seed)
            row.update(
                {
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "mask_ratio": mask_ratio,
                    "consistency_weight": consistency_weight,
                    "seed": args.selection_seed,
                }
            )
            rows.append(row)
            _save_json(root / "ranking.json", _rank(rows))
    ranked = _rank(rows)
    nonzero = next(row for row in ranked if row["consistency_weight"] > 0)
    zero = next(row for row in ranked if row["consistency_weight"] == 0)
    print(
        json.dumps(
            {"best_neuroonline": nonzero, "best_lambda_zero_ablation": zero},
            ensure_ascii=False,
            indent=2,
        )
    )
    return ranked


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=["baseline", "neuroonline", "selection"],
        default="selection",
    )
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    baseline_path = args.output / "01_baseline_selection" / "ranking.json"
    if args.stage in {"baseline", "selection"}:
        baseline_ranking = run_baseline_selection(args)
    else:
        if not baseline_path.exists():
            raise FileNotFoundError(
                "Baseline ranking is missing; run --stage baseline first."
            )
        baseline_ranking = json.loads(baseline_path.read_text(encoding="utf-8"))
    if args.stage in {"neuroonline", "selection"}:
        run_neuroonline_selection(args, baseline_ranking)


if __name__ == "__main__":
    main()
