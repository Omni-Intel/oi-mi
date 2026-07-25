"""Train fixed offline configurations on chronological Day-2 trial splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.train_offline_comparison import (
    evaluate_splits,
    predict_torch,
    seed_everything,
    train_baseline,
    train_neuroonline_offline,
)


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(labels, return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    with np.load(args.day2, allow_pickle=False) as payload:
        inputs = payload["processed_windows"].astype(np.float32)
        labels = payload["labels"].astype(np.int64)
        trial_ids = payload["trial_ids"].astype(np.int64)
        sfreq = float(payload["sfreq"][0])

    ordered_trials = np.asarray(list(dict.fromkeys(trial_ids.tolist())), dtype=np.int64)
    if len(ordered_trials) < 10:
        raise ValueError("Day-2 diagnostic requires at least ten trials.")
    train_end = int(np.floor(len(ordered_trials) * 0.60))
    validation_end = int(np.floor(len(ordered_trials) * 0.80))
    train_trials = ordered_trials[:train_end]
    validation_trials = ordered_trials[train_end:validation_end]
    test_trials = ordered_trials[validation_end:]
    train_mask = np.isin(trial_ids, train_trials)
    validation_mask = np.isin(trial_ids, validation_trials)
    test_mask = np.isin(trial_ids, test_trials)

    train_mean = inputs[train_mask].mean(axis=(0, 2), keepdims=True)
    train_std = np.maximum(
        inputs[train_mask].std(axis=(0, 2), keepdims=True),
        1e-6,
    )

    def scale(values: np.ndarray) -> np.ndarray:
        return ((values - train_mean) / train_std).astype(np.float32)

    train_x = scale(inputs[train_mask])
    train_y = labels[train_mask]
    validation = (
        scale(inputs[validation_mask]),
        labels[validation_mask],
        trial_ids[validation_mask],
    )
    evaluation_splits = {
        "d2_validation_chronological": validation,
        "d2_test_chronological": (
            scale(inputs[test_mask]),
            labels[test_mask],
            trial_ids[test_mask],
        ),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output / "scaler.npz",
        channel_mean=train_mean.astype(np.float32),
        channel_std=train_std.astype(np.float32),
    )
    config = {
        "day2": str(args.day2.resolve()),
        "split": "chronological_trials_60_20_20",
        "trial_counts": {
            "train": int(len(train_trials)),
            "validation": int(len(validation_trials)),
            "test": int(len(test_trials)),
        },
        "window_counts": {
            "train": int(train_mask.sum()),
            "validation": int(validation_mask.sum()),
            "test": int(test_mask.sum()),
        },
        "class_counts_by_trial": {
            "train": _class_counts(
                np.asarray(
                    [labels[np.flatnonzero(trial_ids == trial)[0]] for trial in train_trials]
                )
            ),
            "validation": _class_counts(
                np.asarray(
                    [
                        labels[np.flatnonzero(trial_ids == trial)[0]]
                        for trial in validation_trials
                    ]
                )
            ),
            "test": _class_counts(
                np.asarray(
                    [labels[np.flatnonzero(trial_ids == trial)[0]] for trial in test_trials]
                )
            ),
        },
        "channels": int(inputs.shape[1]),
        "samples": int(inputs.shape[2]),
        "sfreq": sfreq,
        "method": args.method,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "mask_ratio": args.mask_ratio,
        "consistency_weight": args.consistency_weight,
    }
    _save_json(args.output / "run_config.json", config)
    summary: dict[str, Any] = {"config": config, "runs": {}}

    for seed in args.seeds:
        seed_everything(seed)
        output_dir = args.output / args.method / f"seed_{seed}"
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.method == "baseline":
            model, history = train_baseline(
                train_x,
                train_y,
                validation,
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                label_smoothing=args.label_smoothing,
                patience=args.patience,
                sfreq=sfreq,
            )
            predict = lambda values, trained=model: predict_torch(
                trained.model,
                values,
                device=trained._device,
                batch_size=args.batch_size,
            )
            model.save(output_dir / "shallowconvnet.pt")
        else:
            model, history = train_neuroonline_offline(
                train_x,
                train_y,
                validation,
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                label_smoothing=args.label_smoothing,
                patience=args.patience,
                sfreq=sfreq,
                mask_ratio=args.mask_ratio,
                consistency_weight=args.consistency_weight,
            )
            predict = model.predict_proba
            model.save(output_dir / "shallowconvnet.pt")
        metrics = evaluate_splits(predict, evaluation_splits)
        _save_json(output_dir / "history.json", history)
        _save_json(output_dir / "metrics.json", metrics)
        summary["runs"][str(seed)] = metrics
        _save_json(args.output / "summary.json", summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=["baseline", "neuroonline"], required=True)
    parser.add_argument(
        "--seeds",
        type=lambda value: [int(item) for item in value.split(",")],
        default=[17, 42, 2026],
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mask-ratio", type=float, default=0.3)
    parser.add_argument("--consistency-weight", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    summary = run(_parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
