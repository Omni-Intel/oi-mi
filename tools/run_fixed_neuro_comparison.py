"""Compare ordinary offline, frozen NeuroOffline, and updating NeuroOnline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.simulate_online_comparison import (
    Method,
    aggregate_final,
    load_online_data,
    save_json,
    simulate,
)


METHODS: list[Method] = [
    "static",
    "neurooffline_static",
    "neuroonline",
]


def run(args: argparse.Namespace) -> dict[str, Any]:
    data = load_online_data(
        args.offline_data,
        args.online_data,
        scaler_trials=args.scaler_trials,
        online_start_trial=args.online_start_trial,
    )
    if not 0 < args.development_trials < len(data.trial_order):
        raise ValueError("Development trials must leave a non-empty final segment.")

    args.output.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for seed in args.seeds:
        ordinary_checkpoint = (
            args.ordinary_checkpoint_root
            / f"seed_{seed}"
            / "shallowconvnet.pt"
        )
        neuro_checkpoint = (
            args.neuro_checkpoint_root
            / f"seed_{seed}"
            / "shallowconvnet.pt"
        )
        if not ordinary_checkpoint.exists():
            raise FileNotFoundError(ordinary_checkpoint)
        if not neuro_checkpoint.exists():
            raise FileNotFoundError(neuro_checkpoint)
        if not Path(f"{neuro_checkpoint}.neuroonline.pt").exists():
            raise FileNotFoundError(f"{neuro_checkpoint}.neuroonline.pt")

        for method in METHODS:
            checkpoint = (
                ordinary_checkpoint
                if method == "static"
                else neuro_checkpoint
            )
            runs.append(
                simulate(
                    data=data,
                    checkpoint=checkpoint,
                    method=method,
                    seed=seed,
                    update_windows=args.update_windows,
                    mask_ratio=args.mask_ratio,
                    consistency_weight=args.consistency_weight,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    label_smoothing=args.label_smoothing,
                    epochs_per_update=args.epochs_per_update,
                    end_trial=None,
                    development_trials=args.development_trials,
                    output_prefix=(
                        args.output
                        / "final"
                        / method
                        / f"seed_{seed}"
                    ),
                )
            )

    config = {
        "scenario": args.scenario,
        "offline_data": str(args.offline_data.resolve()),
        "online_data": str(args.online_data.resolve()),
        "ordinary_checkpoint_root": str(args.ordinary_checkpoint_root.resolve()),
        "neuro_checkpoint_root": str(args.neuro_checkpoint_root.resolve()),
        "scaler_trials": args.scaler_trials,
        "online_start_trial": args.online_start_trial,
        "stream_trials": int(len(data.trial_order)),
        "stream_windows": int(len(data.windows)),
        "development_trials": args.development_trials,
        "final_trials": int(len(data.trial_order) - args.development_trials),
        "methods": METHODS,
        "seeds": args.seeds,
        "update_windows": args.update_windows,
        "mask_ratio": args.mask_ratio,
        "lambda": args.consistency_weight,
        "epochs_per_update": args.epochs_per_update,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "causal_order": "predict trial -> reveal label -> accumulate windows -> update at N",
    }
    summary = {
        "config": config,
        "final": aggregate_final(runs, METHODS),
    }
    save_json(args.output / "final_summary.json", summary)
    return summary


def comma_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--offline-data", type=Path, required=True)
    parser.add_argument("--online-data", type=Path, required=True)
    parser.add_argument("--ordinary-checkpoint-root", type=Path, required=True)
    parser.add_argument("--neuro-checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scaler-trials", type=int)
    parser.add_argument("--online-start-trial", type=int, default=0)
    parser.add_argument("--development-trials", type=int, required=True)
    parser.add_argument("--seeds", type=comma_ints, default=[17, 42, 2026])
    parser.add_argument("--update-windows", type=int, required=True)
    parser.add_argument("--mask-ratio", type=float, required=True)
    parser.add_argument("--consistency-weight", type=float, required=True)
    parser.add_argument("--epochs-per-update", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
