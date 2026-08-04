"""Repeated trial-grouped CV for staged CBraMod NeuroOnline offline search."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
from itertools import product
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adaptation.neuroonline import (  # noqa: E402
    ClassCollapseError,
    NEUROONLINE_TRAINING_MECHANICS_VERSION,
    NeuroOnlineConfig,
    NeuroOnlineModelAdapter,
)
from models.factory import ModelFactory, TorchModelAdapter  # noqa: E402


OFFLINE_SEARCH_CANDIDATES = [
    {
        "update_policy": "full",
        "head_learning_rate": learning_rate,
        "backbone_learning_rate": learning_rate,
        "batch_size": batch_size,
        "mask_ratio": mask_ratio,
        "consistency_weight": consistency_weight,
    }
    for learning_rate, batch_size, mask_ratio, consistency_weight in product(
        (1e-6, 1e-5, 3e-5, 1e-4, 3e-4),
        (16, 32, 64, 128),
        (0.1, 0.3, 0.5, 0.7),
        (0.1, 0.25, 0.5, 1.0),
    )
]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_candidates(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return [dict(candidate) for candidate in OFFLINE_SEARCH_CANDIDATES]
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates") if isinstance(payload, dict) else payload
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Candidate manifest must contain a non-empty candidates list.")
    return [dict(candidate) for candidate in candidates]


def _load_dataset(path: Path, feature_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    with np.load(path, allow_pickle=False) as payload:
        required = {feature_key, "labels", "trial_ids", "sfreq"}
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"Calibration dataset is missing: {', '.join(missing)}")
        windows = payload[feature_key].astype(np.float32)
        labels = payload["labels"].astype(np.int64)
        groups = payload["trial_ids"].astype(np.int64)
        sfreq = float(payload["sfreq"].reshape(-1)[0])
    if windows.ndim != 3 or labels.shape != groups.shape or windows.shape[0] != labels.size:
        raise ValueError("Expected windows [N,C,T] and matching labels/trial_ids.")
    for group in np.unique(groups):
        if np.unique(labels[groups == group]).size != 1:
            raise ValueError(f"Trial {int(group)} contains multiple labels.")
    return windows, labels, groups, sfreq


def _grouped_splits(
    windows: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )
    splits = list(splitter.split(windows, labels, groups=groups))
    for train_indices, validation_indices in splits:
        train_groups = set(groups[train_indices].tolist())
        validation_groups = set(groups[validation_indices].tolist())
        if train_groups & validation_groups:
            raise RuntimeError("Trial-group leakage detected in grouped CV split.")
    return splits


def _ranking_score(report: dict[str, Any]) -> tuple[float, ...]:
    summary = report["summary"]
    return (
        -float(summary["collapsed_runs"]),
        float(summary["mean_trial_worst_class_recall"]),
        float(summary["mean_trial_balanced_accuracy"])
        - 0.5 * float(summary["std_trial_balanced_accuracy"]),
        float(summary["mean_trial_balanced_accuracy"]),
        -float(report.get("identity", {}).get("candidate_index", 0)),
    )


def _compatible_identity(existing: dict[str, Any], current: dict[str, Any]) -> bool:
    """Allow content-identical datasets to resume after a server migration."""

    existing_compare = dict(existing)
    current_compare = dict(current)
    existing_compare.pop("dataset", None)
    current_compare.pop("dataset", None)
    return existing_compare == current_compare


def _summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [run for run in runs if not run["class_collapse"]]
    all_bacc = np.asarray(
        [run["metrics"]["val_trial_balanced_accuracy"] for run in runs],
        dtype=np.float64,
    )
    all_worst = np.asarray(
        [run["metrics"]["val_trial_worst_class_accuracy"] for run in runs],
        dtype=np.float64,
    )
    valid_bacc = np.asarray(
        [run["metrics"]["val_trial_balanced_accuracy"] for run in valid],
        dtype=np.float64,
    )
    valid_worst = np.asarray(
        [run["metrics"]["val_trial_worst_class_accuracy"] for run in valid],
        dtype=np.float64,
    )
    best_epochs = [
        float(run["metrics"]["best_epoch"])
        for run in valid
        if run["metrics"].get("best_epoch") is not None
    ]
    return {
        "runs": len(runs),
        "valid_runs": len(valid),
        "collapsed_runs": len(runs) - len(valid),
        "mean_trial_balanced_accuracy": float(np.mean(all_bacc)),
        "std_trial_balanced_accuracy": float(np.std(all_bacc)),
        "mean_trial_worst_class_recall": float(np.mean(all_worst)),
        "minimum_trial_worst_class_recall": float(np.min(all_worst)),
        "mean_noncollapsed_trial_balanced_accuracy": (
            float(np.mean(valid_bacc)) if valid else None
        ),
        "std_noncollapsed_trial_balanced_accuracy": (
            float(np.std(valid_bacc)) if valid else None
        ),
        "mean_noncollapsed_trial_worst_class_recall": (
            float(np.mean(valid_worst)) if valid else None
        ),
        "median_best_epoch": (
            int(np.floor(float(np.median(best_epochs)) + 0.5))
            if best_epochs
            else None
        ),
    }


def _build_adapter(
    *,
    config: NeuroOnlineConfig,
    n_chans: int,
    n_times: int,
    n_classes: int,
    sfreq: float,
) -> NeuroOnlineModelAdapter:
    base = ModelFactory.get(
        "cbramod",
        n_chans=n_chans,
        n_times=n_times,
        n_classes=n_classes,
        sfreq=sfreq,
    )
    if not isinstance(base, TorchModelAdapter):
        raise TypeError("CBraMod grouped CV requires a TorchModelAdapter.")
    return NeuroOnlineModelAdapter(base, config=config)


def _candidate_config(
    base: NeuroOnlineConfig,
    candidate: dict[str, Any],
    *,
    seed: int,
    epochs: int,
) -> NeuroOnlineConfig:
    return replace(
        base,
        enabled=True,
        offline_random_seed=seed,
        offline_epochs=epochs,
        offline_update_policy=str(candidate["update_policy"]),
        offline_learning_rate=float(candidate["head_learning_rate"]),
        offline_backbone_learning_rate=(
            None
            if candidate.get("backbone_learning_rate") is None
            else float(candidate["backbone_learning_rate"])
        ),
        offline_batch_size=int(candidate["batch_size"]),
        offline_mask_ratio=float(candidate["mask_ratio"]),
        offline_consistency_weight=float(candidate["consistency_weight"]),
    )


def evaluate_candidate(args: argparse.Namespace) -> dict[str, Any]:
    dataset = args.dataset.resolve()
    candidates = _load_candidates(args.candidate_manifest)
    if not 0 <= args.candidate_index < len(candidates):
        raise ValueError(f"candidate-index must be in [0, {len(candidates)}).")
    candidate = candidates[args.candidate_index]
    output = args.output.resolve() / f"candidate_{args.candidate_index:03d}.json"
    identity = {
        "schema_version": 1,
        "dataset": str(dataset),
        "dataset_sha256": _sha256(dataset),
        "feature_key": args.feature_key,
        "candidate_index": args.candidate_index,
        "candidate": candidate,
        "folds": args.folds,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "fixed_epoch_training": True,
    }
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if not _compatible_identity(existing.get("identity", {}), identity):
            raise ValueError(f"Existing candidate report has incompatible inputs: {output}")
        print(f"REUSED {output}", flush=True)
        return existing

    windows, labels, groups, sfreq = _load_dataset(dataset, args.feature_key)
    config_mapping = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    base_config = NeuroOnlineConfig.from_mapping(
        config_mapping.get("online_adaptation", {}) or {}
    )
    n_classes = int(config_mapping.get("n_classes", int(labels.max()) + 1))
    runs: list[dict[str, Any]] = []
    started = time.perf_counter()
    for split_seed in args.seeds:
        for fold_index, (train_indices, validation_indices) in enumerate(
            _grouped_splits(
                windows,
                labels,
                groups,
                folds=args.folds,
                seed=split_seed,
            )
        ):
            run_seed = int(split_seed * 100 + fold_index)
            _seed_everything(run_seed)
            config = _candidate_config(
                base_config,
                candidate,
                seed=run_seed,
                epochs=args.epochs,
            )
            run_started = time.perf_counter()
            adapter = _build_adapter(
                config=config,
                n_chans=windows.shape[1],
                n_times=windows.shape[2],
                n_classes=n_classes,
                sfreq=sfreq,
            )
            record: dict[str, Any] = {
                "split_seed": split_seed,
                "fold": fold_index,
                "run_seed": run_seed,
                "train_trials": int(np.unique(groups[train_indices]).size),
                "validation_trials": int(np.unique(groups[validation_indices]).size),
            }
            try:
                metrics = adapter.fit_with_split(
                    windows,
                    labels,
                    train_indices=train_indices,
                    validation_indices=validation_indices,
                    groups=groups,
                )
                record["metrics"] = metrics
                record["class_collapse"] = bool(
                    metrics["val_trial_worst_class_accuracy"] <= 0.0
                )
            except ClassCollapseError as exc:
                record["metrics"] = exc.metrics
                record["class_collapse"] = True
                record["error"] = str(exc)
            record["duration_sec"] = time.perf_counter() - run_started
            runs.append(record)
            del adapter
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            metric = record["metrics"]
            print(
                f"candidate={args.candidate_index + 1}/{len(candidates)} "
                f"seed={split_seed} fold={fold_index + 1}/{args.folds} "
                f"trial_bacc={metric.get('val_trial_balanced_accuracy', -1):.4f} "
                f"trial_worst={metric.get('val_trial_worst_class_accuracy', -1):.4f} "
                f"collapse={record['class_collapse']}",
                flush=True,
            )

    summary = _summarize_runs(runs)
    report = {
        "identity": identity,
        "dataset": {
            "windows": int(windows.shape[0]),
            "trials": int(np.unique(groups).size),
            "shape": list(windows.shape),
        },
        "summary": summary,
        "runs": runs,
        "duration_sec": time.perf_counter() - started,
    }
    _write_json(output, report)
    return report


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    candidates = _load_candidates(args.candidate_manifest)
    reports = []
    for index in range(len(candidates)):
        path = args.output.resolve() / f"candidate_{index:03d}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing candidate report: {path}")
        reports.append(json.loads(path.read_text(encoding="utf-8")))

    for report in reports:
        identity = report.get("identity", {})
        if identity.get("fixed_epoch_training") is not True:
            raise ValueError(
                "Offline candidate report was not produced by fixed-epoch training: "
                f"candidate_{int(identity.get('candidate_index', -1)):03d}.json"
            )
        expected_epochs = int(identity["epochs"])
        incomplete_runs = [
            run
            for run in report.get("runs", [])
            if int(run.get("metrics", {}).get("epochs_completed", -1))
            != expected_epochs
        ]
        if incomplete_runs:
            raise ValueError(
                "Offline candidate contains runs that did not complete all fixed epochs: "
                f"candidate_{int(identity['candidate_index']):03d}.json"
            )
        report["summary"] = _summarize_runs(report["runs"])
    ranked = sorted(reports, key=_ranking_score, reverse=True)
    payload = {
        "schema_version": 1,
        "training_mechanics_version": NEUROONLINE_TRAINING_MECHANICS_VERSION,
        "fixed_epoch_training": True,
        "expected_candidates": len(candidates),
        "completed_candidates": len(reports),
        "ranking_rule": (
            "minimize collapsed folds; maximize mean trial worst-class recall; "
            "maximize mean trial bACC minus 0.5*std; maximize mean trial bACC; "
            "break exact ties by the lowest predeclared candidate index"
        ),
        "ranked": [
            {
                "rank": rank,
                "candidate_index": report["identity"]["candidate_index"],
                "candidate": report["identity"]["candidate"],
                "summary": report["summary"],
            }
            for rank, report in enumerate(ranked, 1)
        ],
        "selected_candidate_index": int(ranked[0]["identity"]["candidate_index"]),
    }
    _write_json(args.output.resolve() / "summary.json", payload)
    return payload


def _comma_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--feature-key", default="processed_windows")
    parser.add_argument("--candidate-manifest", type=Path, default=None)
    parser.add_argument("--candidate-index", type=int, default=None)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=_comma_ints, default=[17, 42, 2026])
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    if args.summarize == (args.candidate_index is not None):
        parser.error("Choose exactly one of --summarize or --candidate-index.")
    return args


def main() -> None:
    args = parse_args()
    payload = summarize(args) if args.summarize else evaluate_candidate(args)
    print(json.dumps(payload.get("summary", payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
