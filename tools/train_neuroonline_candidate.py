"""Search NeuroOnline hyperparameters and train isolated deployment candidates."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adaptation.calibration_search import (  # noqa: E402
    CalibrationSearchResult,
    CalibrationSearchConfig,
    load_latest_calibration_search,
    run_calibration_search,
)
from adaptation.neuroonline import (  # noqa: E402
    ClassCollapseError,
    NEUROONLINE_TRAINING_MECHANICS_VERSION,
    NeuroOnlineConfig,
    NeuroOnlineModelAdapter,
)
from models.factory import (  # noqa: E402
    ModelFactory,
    TorchModelAdapter,
    split_train_validation_indices,
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _aggregate_trials(
    labels: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    trial_labels: list[int] = []
    trial_probabilities: list[np.ndarray] = []
    for group in np.unique(groups):
        mask = groups == group
        unique_labels = np.unique(labels[mask])
        if unique_labels.size != 1:
            raise ValueError(f"Trial {int(group)} contains multiple labels.")
        trial_labels.append(int(unique_labels[0]))
        trial_probabilities.append(np.mean(probabilities[mask], axis=0))
    return (
        np.asarray(trial_labels, dtype=np.int64),
        np.stack(trial_probabilities).astype(np.float32),
    )


def _metrics(
    truth: np.ndarray,
    predictions: np.ndarray,
    *,
    n_classes: int,
) -> dict[str, Any]:
    labels = np.arange(n_classes, dtype=np.int64)
    matrix = confusion_matrix(truth, predictions, labels=labels)
    totals = matrix.sum(axis=1)
    recalls = np.divide(
        np.diag(matrix),
        totals,
        out=np.zeros(n_classes, dtype=np.float64),
        where=totals > 0,
    )
    kappa = float(cohen_kappa_score(truth, predictions))
    if not np.isfinite(kappa):
        kappa = -1.0
    return {
        "accuracy": float(np.mean(truth == predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "kappa": kappa,
        "macro_f1": float(
            f1_score(truth, predictions, average="macro", zero_division=0)
        ),
        "worst_class_recall": float(np.min(recalls)),
        "per_class_recall": {
            str(label): float(recalls[label]) for label in labels
        },
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def _build_model(
    *,
    model_name: str,
    n_chans: int,
    n_times: int,
    n_classes: int,
    sfreq: float,
) -> TorchModelAdapter:
    model = ModelFactory.get(
        model_name,
        n_chans=n_chans,
        n_times=n_times,
        n_classes=n_classes,
        sfreq=sfreq,
    )
    if not isinstance(model, TorchModelAdapter):
        raise TypeError("NeuroOnline candidate training requires a PyTorch model.")
    return model


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    with np.load(args.dataset) as payload:
        required = {args.feature_key, "labels", "trial_ids"}
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"Calibration dataset is missing: {', '.join(missing)}")
        windows = payload[args.feature_key].astype(np.float32)
        labels = payload["labels"].astype(np.int64)
        groups = payload["trial_ids"].astype(np.int64)
        dataset_sfreq = (
            float(payload["sfreq"].reshape(-1)[0])
            if "sfreq" in payload
            else float(config["sfreq"])
        )
    if windows.ndim != 3 or labels.shape != groups.shape:
        raise ValueError("Expected windows [N,C,T] with matching labels and trial_ids.")
    if windows.shape[0] != labels.size:
        raise ValueError("Window and label counts differ.")

    model_name = args.model or str(config.get("model_name", "shallowconvnet"))
    n_classes = int(config.get("n_classes", int(np.max(labels)) + 1))
    if set(np.unique(labels).tolist()) != set(range(n_classes)):
        raise ValueError("Calibration labels do not cover every configured class.")

    args.output.mkdir(parents=True, exist_ok=True)
    neuro_root = config.get("online_adaptation", {}) or {}
    base_config = replace(
        NeuroOnlineConfig.from_mapping(neuro_root),
        enabled=True,
        offline_random_seed=args.search_seed,
    )
    configured_search = CalibrationSearchConfig.from_mapping(neuro_root)
    search_config = replace(
        configured_search,
        enabled=True,
        reuse_latest=False,
        mode=args.search_mode,
    )

    _seed_everything(args.search_seed)
    base_template = _build_model(
        model_name=model_name,
        n_chans=int(windows.shape[1]),
        n_times=int(windows.shape[2]),
        n_classes=n_classes,
        sfreq=dataset_sfreq,
    )
    last_progress: tuple[int, str, int] | None = None

    def progress(
        candidate: int,
        total: int,
        stage: str,
        metrics: dict[str, float],
    ) -> None:
        nonlocal last_progress
        epoch_text = stage.rsplit(" ", 1)[-1]
        current_epoch = int(epoch_text.split("/", 1)[0])
        stage_name = stage.split(" epoch ", 1)[0]
        marker = (candidate, stage_name, current_epoch)
        if (
            last_progress is None
            or candidate != last_progress[0]
            or current_epoch % 10 == 0
            or current_epoch == search_config.selection_epochs
        ):
            print(
                f"SEARCH {candidate}/{total} {stage} "
                f"trial_worst={metrics.get('val_trial_worst_class_accuracy', 0.0):.3f} "
                f"trial_bacc={metrics.get('val_trial_balanced_accuracy', 0.0):.3f}",
                flush=True,
            )
        last_progress = marker

    search_dir = args.output / f"search_seed_{args.search_seed}"
    if args.reuse_search_report:
        best_config, report, report_path = load_latest_calibration_search(
            calibration_records_dir=args.output,
            base_config=base_config,
        )
        if not bool(report.get("deployment_eligible", False)):
            raise RuntimeError(
                "The reusable search report is not deployment eligible."
            )
        if int(report.get("random_seed", -1)) != args.search_seed:
            raise RuntimeError(
                "The reusable search report was produced with a different search seed."
            )
        search_result = CalibrationSearchResult(
            best_config=best_config,
            report=report,
            report_path=report_path,
        )
        print(f"REUSED_SEARCH {report_path.resolve()}", flush=True)
    else:
        search_result = run_calibration_search(
            base_template=base_template,
            base_config=base_config,
            search_config=search_config,
            X=windows,
            y=labels,
            groups=groups,
            session_dir=search_dir,
            progress_callback=progress,
        )
    print(
        "SELECTED "
        + json.dumps(
            search_result.report["best_parameters"],
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )

    development_indices, validation_indices = split_train_validation_indices(
        labels,
        groups=groups,
        random_state=args.search_seed,
    )
    runs: dict[str, Any] = {}
    for seed in args.confirmation_seeds:
        print(f"CONFIRM seed={seed} start", flush=True)
        _seed_everything(seed)
        adapter = NeuroOnlineModelAdapter(
            _build_model(
                model_name=model_name,
                n_chans=int(windows.shape[1]),
                n_times=int(windows.shape[2]),
                n_classes=n_classes,
                sfreq=dataset_sfreq,
            ),
            config=replace(search_result.best_config, offline_random_seed=seed),
        )
        started = time.perf_counter()
        try:
            fit_metrics = adapter.fit_with_split(
                windows,
                labels,
                train_indices=development_indices,
                validation_indices=validation_indices,
                groups=groups,
            )
        except ClassCollapseError as exc:
            runs[str(seed)] = {
                "duration_sec": time.perf_counter() - started,
                "rejected": "training_class_collapse",
                "fit_metrics": exc.metrics,
            }
            print(f"CONFIRM seed={seed} rejected during training", flush=True)
            continue
        probabilities = adapter.predict_proba(windows[validation_indices])
        truth = labels[validation_indices]
        window_metrics = _metrics(
            truth,
            probabilities.argmax(axis=1),
            n_classes=n_classes,
        )
        trial_truth, trial_probabilities = _aggregate_trials(
            truth,
            probabilities,
            groups[validation_indices],
        )
        trial_metrics = _metrics(
            trial_truth,
            trial_probabilities.argmax(axis=1),
            n_classes=n_classes,
        )
        if (
            window_metrics["worst_class_recall"] <= 0.0
            or trial_metrics["worst_class_recall"] <= 0.0
        ):
            runs[str(seed)] = {
                "duration_sec": time.perf_counter() - started,
                "rejected": "confirmation_class_collapse",
                "fit_metrics": fit_metrics,
                "window_metrics": window_metrics,
                "trial_metrics": trial_metrics,
            }
            print(
                f"CONFIRM seed={seed} rejected "
                f"trial_recall={trial_metrics['per_class_recall']}",
                flush=True,
            )
            continue

        seed_dir = args.output / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        model_path = seed_dir / f"{model_name}_candidate.pt"
        adapter.save(model_path)
        sidecar_path = Path(f"{model_path}.neuroonline.pt")
        runs[str(seed)] = {
            "duration_sec": time.perf_counter() - started,
            "fit_metrics": fit_metrics,
            "window_metrics": window_metrics,
            "trial_metrics": trial_metrics,
            "artifacts": {
                "model": _artifact(model_path),
                "neuroonline_sidecar": _artifact(sidecar_path),
            },
        }
        print(
            f"CONFIRM seed={seed} "
            f"trial_recall={trial_metrics['per_class_recall']} "
            f"trial_bacc={trial_metrics['balanced_accuracy']:.3f}",
            flush=True,
        )

    eligible_seeds = [
        seed for seed in args.confirmation_seeds if "artifacts" in runs[str(seed)]
    ]
    if not eligible_seeds:
        raise RuntimeError(
            "Every confirmation seed collapsed at least one class; "
            "no deployment candidate was saved."
        )
    deployment_seed = max(
        eligible_seeds,
        key=lambda seed: (
            runs[str(seed)]["trial_metrics"]["worst_class_recall"],
            runs[str(seed)]["window_metrics"]["worst_class_recall"],
            runs[str(seed)]["trial_metrics"]["balanced_accuracy"],
        ),
    )
    summary = {
        "schema_version": 1,
        "training_mechanics_version": NEUROONLINE_TRAINING_MECHANICS_VERSION,
        "source_dataset": str(args.dataset.resolve()),
        "feature_key": args.feature_key,
        "model_name": model_name,
        "model_unchanged": model_name == "shallowconvnet",
        "dataset": {
            "windows": int(windows.shape[0]),
            "trials": int(np.unique(groups).size),
            "shape": [int(value) for value in windows.shape],
            "class_counts": {
                str(label): int(np.sum(labels == label))
                for label in range(n_classes)
            },
        },
        "objective": (
            "CE(original)+CE(time_masked)+CE(freq_masked)"
            "+lambda*mean(MSE(time,original),MSE(freq,original))"
        ),
        "search_seed": args.search_seed,
        "search_mode": args.search_mode,
        "search_report_reused": bool(args.reuse_search_report),
        "search_report": str(search_result.report_path.resolve()),
        "search_config": asdict(search_config),
        "selected_parameters": search_result.report["best_parameters"],
        "untouched_search_holdout": {
            "trial": search_result.report["untouched_holdout_trial_metrics"],
            "window": search_result.report["untouched_holdout_window_metrics"],
        },
        "confirmation_split": {
            "train_windows": int(development_indices.size),
            "validation_windows": int(validation_indices.size),
            "train_trials": int(np.unique(groups[development_indices]).size),
            "validation_trials": int(np.unique(groups[validation_indices]).size),
        },
        "confirmation_runs": runs,
        "eligible_confirmation_seeds": eligible_seeds,
        "deployment_candidate_seed": deployment_seed,
        "deployment_candidate": runs[str(deployment_seed)]["artifacts"],
        "live_model_overwritten": False,
    }
    _write_json(args.output / "training_summary.json", summary)
    return summary


def _comma_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one seed is required.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--feature-key",
        choices=("processed_windows", "raw_windows"),
        default="processed_windows",
    )
    parser.add_argument("--search-mode", choices=("staged", "full_grid"), default="staged")
    parser.add_argument("--search-seed", type=int, default=42)
    parser.add_argument(
        "--reuse-search-report",
        action="store_true",
        help="Reuse a compatible deployment-eligible report under --output.",
    )
    parser.add_argument(
        "--confirmation-seeds",
        type=_comma_ints,
        default=(17, 42, 2026),
    )
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(
        json.dumps(
            {
                "selected_parameters": summary["selected_parameters"],
                "deployment_candidate_seed": summary["deployment_candidate_seed"],
                "deployment_candidate": summary["deployment_candidate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
