"""Train on the first calibration half and causally replay the second half.

The chronological split is made at a planned trial boundary. Offline epoch
selection only sees trials in the first half. The online half contributes one
primary window per trial, matching the standalone car's adaptation contract.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adaptation.neuroonline import (  # noqa: E402
    ClassCollapseError,
    NeuroOnlineConfig,
    NeuroOnlineModelAdapter,
)
from models.factory import ModelFactory, TorchModelAdapter  # noqa: E402
from tools.simulate_neuroonline_realtime import (  # noqa: E402
    artifact,
    build_adapter,
    causal_replay,
    metric_bundle,
    seed_everything,
    sha256,
)


def chronological_trial_masks(
    trial_ids: np.ndarray,
    *,
    split_trial: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return masks for planned trials before/after a chronological midpoint."""

    groups = np.asarray(trial_ids, dtype=np.int64)
    if groups.ndim != 1 or groups.size == 0:
        raise ValueError("trial_ids must be a non-empty one-dimensional array.")
    if np.any(np.diff(groups) < 0):
        raise ValueError("Calibration trial_ids are not in chronological order.")
    if split_trial is None:
        planned_start = int(groups.min())
        planned_stop = int(groups.max()) + 1
        split_trial = planned_start + (planned_stop - planned_start) // 2
    boundary = int(split_trial)
    offline = groups < boundary
    online = groups >= boundary
    if not np.any(offline) or not np.any(online):
        raise ValueError(
            f"split_trial={boundary} does not leave data in both chronological halves."
        )
    if np.intersect1d(np.unique(groups[offline]), np.unique(groups[online])).size:
        raise RuntimeError("A calibration trial crossed the chronological split.")
    return offline, online, boundary


def primary_window_mask(
    trial_ids: np.ndarray,
    window_indices: np.ndarray,
    online_mask: np.ndarray,
    *,
    primary_window_index: int = 0,
) -> np.ndarray:
    """Select at most one operationally eligible window from each online trial."""

    groups = np.asarray(trial_ids, dtype=np.int64)
    windows = np.asarray(window_indices, dtype=np.int64)
    mask = np.asarray(online_mask, dtype=bool) & (windows == primary_window_index)
    selected_groups = groups[mask]
    if np.unique(selected_groups).size != selected_groups.size:
        raise ValueError("The dataset contains multiple primary windows for one trial.")
    return mask


def expected_update_triggers(
    samples: int,
    *,
    history_threshold: int,
    update_stride: int,
) -> list[int]:
    """List the causal labeled-sample counts that can trigger updates."""

    return [
        seen
        for seen in range(1, int(samples) + 1)
        if seen >= history_threshold and seen % update_stride == 0
    ]


def _load_dataset(path: Path, feature_key: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            feature_key,
            "labels",
            "trial_ids",
            "window_indices",
            "sfreq",
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"Calibration dataset is missing: {', '.join(missing)}")
        result = {
            "windows": payload[feature_key].astype(np.float32),
            "labels": payload["labels"].astype(np.int64),
            "trial_ids": payload["trial_ids"].astype(np.int64),
            "window_indices": payload["window_indices"].astype(np.int64),
            "sfreq": float(payload["sfreq"].reshape(-1)[0]),
            "quality_rejected_windows": (
                int(payload["quality_rejected_windows"].reshape(-1)[0])
                if "quality_rejected_windows" in payload.files
                else None
            ),
        }
    windows = result["windows"]
    labels = result["labels"]
    trial_ids = result["trial_ids"]
    if windows.ndim != 3 or labels.shape != trial_ids.shape:
        raise ValueError("Expected windows [N,C,T] and matching labels/trial_ids.")
    if windows.shape[0] != labels.size or result["window_indices"].shape != labels.shape:
        raise ValueError("Calibration feature and grouping arrays have different lengths.")
    if not np.isfinite(windows).all():
        raise ValueError("Calibration features contain NaN or infinity.")
    for group in np.unique(trial_ids):
        if np.unique(labels[trial_ids == group]).size != 1:
            raise ValueError(f"Trial {int(group)} contains multiple labels.")
    return result


def _build_fresh_adapter(
    config: NeuroOnlineConfig,
    *,
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
        raise TypeError("The chronological calibration diagnostic requires CBraMod.")
    return NeuroOnlineModelAdapter(base, config=config)


def _class_counts(labels: np.ndarray, n_classes: int) -> list[int]:
    return np.bincount(np.asarray(labels, dtype=np.int64), minlength=n_classes).astype(int).tolist()


def _trial_class_counts(
    labels: np.ndarray,
    trial_ids: np.ndarray,
    n_classes: int,
) -> list[int]:
    trial_labels = np.asarray(
        [labels[np.flatnonzero(trial_ids == group)[0]] for group in np.unique(trial_ids)],
        dtype=np.int64,
    )
    return _class_counts(trial_labels, n_classes)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_path = args.dataset.resolve()
    config_path = args.config.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    checkpoint_path = output_dir / "cbramod_first_half.pt"
    predictions_path = output_dir / "online_primary_predictions.npz"
    if any(path.exists() for path in (report_path, checkpoint_path, predictions_path)):
        raise FileExistsError(f"Output directory already contains diagnostic artifacts: {output_dir}")

    data = _load_dataset(dataset_path, args.feature_key)
    windows = data["windows"]
    labels = data["labels"]
    trial_ids = data["trial_ids"]
    offline_mask, online_mask, split_trial = chronological_trial_masks(
        trial_ids,
        split_trial=args.split_trial,
    )
    primary_mask = primary_window_mask(
        trial_ids,
        data["window_indices"],
        online_mask,
        primary_window_index=args.primary_window_index,
    )

    config_mapping = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config = NeuroOnlineConfig.from_mapping(config_mapping.get("online_adaptation", {}) or {})
    if not config.enabled:
        raise ValueError("NeuroOnline must be enabled in the supplied configuration.")
    config = replace(
        config,
        history_threshold=(
            config.history_threshold
            if args.history_threshold is None
            else int(args.history_threshold)
        ),
        update_stride=(
            config.update_stride
            if args.update_stride is None
            else int(args.update_stride)
        ),
    )
    if config.history_threshold < 1 or config.update_stride < 1:
        raise ValueError("history-threshold and update-stride must both be positive.")
    n_classes = int(config_mapping.get("n_classes", int(labels.max()) + 1))
    if set(np.unique(labels[offline_mask]).tolist()) != set(range(n_classes)):
        raise ValueError("The offline half does not contain every configured class.")
    if set(np.unique(labels[primary_mask]).tolist()) != set(range(n_classes)):
        raise ValueError("The online primary-window half does not contain every configured class.")

    offline_windows = windows[offline_mask]
    offline_labels = labels[offline_mask]
    offline_groups = trial_ids[offline_mask]
    started = time.perf_counter()
    selected_epoch = args.fixed_offline_epochs
    if selected_epoch is not None:
        selected_epoch = int(selected_epoch)
        if not 1 <= selected_epoch <= config.offline_epochs:
            raise ValueError(
                "fixed-offline-epochs must be between 1 and configured offline_epochs."
            )
        selection_metrics: dict[str, Any] = {
            "mode": "predeclared_fixed_epoch_exploratory",
            "best_epoch": float(selected_epoch),
            "validation_used": False,
        }
    else:
        seed_everything(config.effective_offline_random_seed)
        selector = _build_fresh_adapter(
            config,
            n_chans=windows.shape[1],
            n_times=windows.shape[2],
            n_classes=n_classes,
            sfreq=data["sfreq"],
        )
        try:
            selection_metrics = selector.fit(
                offline_windows,
                offline_labels,
                epochs=config.offline_epochs,
                batch_size=config.offline_batch_size,
                learning_rate=config.offline_learning_rate,
                groups=offline_groups,
            )
        except ClassCollapseError as exc:
            failure_report = {
                "schema_version": 1,
                "status": "calibration_failed_class_collapse",
                "method": (
                    "chronological planned-trial split; offline epoch selection "
                    "restricted to first-half trials"
                ),
                "identity": {
                    "dataset": str(dataset_path),
                    "dataset_sha256": sha256(dataset_path),
                    "config": str(config_path),
                    "config_sha256": sha256(config_path),
                    "feature_key": args.feature_key,
                    "split_trial": split_trial,
                    "primary_window_index": args.primary_window_index,
                    "history_threshold_override": args.history_threshold,
                    "update_stride_override": args.update_stride,
                    "fixed_offline_epochs": args.fixed_offline_epochs,
                },
                "split": {
                    "offline_planned_trials": f"{int(trial_ids.min())}-{split_trial - 1}",
                    "online_planned_trials": f"{split_trial}-{int(trial_ids.max())}",
                    "offline_valid_trials": int(np.unique(offline_groups).size),
                    "offline_windows": int(offline_labels.size),
                    "offline_trial_class_counts": _trial_class_counts(
                        offline_labels, offline_groups, n_classes
                    ),
                    "offline_window_class_counts": _class_counts(offline_labels, n_classes),
                    "online_valid_trials": int(np.unique(trial_ids[online_mask]).size),
                    "online_all_valid_windows": int(np.sum(online_mask)),
                    "online_primary_windows": int(np.sum(primary_mask)),
                    "online_primary_class_counts": _class_counts(
                        labels[primary_mask], n_classes
                    ),
                    "trial_overlap": [],
                },
                "configuration": asdict(config),
                "failure": {
                    "reason": str(exc),
                    "last_epoch_validation_metrics": exc.metrics,
                    "online_replay_started": False,
                },
                "online_feasibility": {
                    "expected_update_triggers": expected_update_triggers(
                        int(np.sum(primary_mask)),
                        history_threshold=config.history_threshold,
                        update_stride=config.update_stride,
                    ),
                    "additional_primary_windows_to_first_update": max(
                        config.history_threshold - int(np.sum(primary_mask)), 0
                    ),
                },
                "duration_sec": float(time.perf_counter() - started),
            }
            _save_json(report_path, failure_report)
            return failure_report

        del selector
        selected_epoch = int(selection_metrics["best_epoch"])
    seed_everything(config.effective_offline_random_seed)
    calibrated = _build_fresh_adapter(
        config,
        n_chans=windows.shape[1],
        n_times=windows.shape[2],
        n_classes=n_classes,
        sfreq=data["sfreq"],
    )
    full_fit_metrics = calibrated.fit_full(
        offline_windows,
        offline_labels,
        epochs=selected_epoch,
    )
    calibrated.save(checkpoint_path)

    online_windows = windows[primary_mask]
    online_labels = labels[primary_mask]
    online_trials = trial_ids[primary_mask]
    baseline = build_adapter(
        checkpoint_path,
        model_name="cbramod",
        n_chans=windows.shape[1],
        n_times=windows.shape[2],
        n_classes=n_classes,
        sfreq=data["sfreq"],
        config=config,
    )
    baseline_probabilities = baseline.predict_proba(online_windows)
    del baseline
    adapted = build_adapter(
        checkpoint_path,
        model_name="cbramod",
        n_chans=windows.shape[1],
        n_times=windows.shape[2],
        n_classes=n_classes,
        sfreq=data["sfreq"],
        config=config,
    )
    adapted_probabilities, model_revisions, updates = causal_replay(
        adapted,
        online_windows,
        online_labels,
        online_trials,
        config=config,
        n_classes=n_classes,
    )
    triggers = expected_update_triggers(
        len(online_labels),
        history_threshold=config.history_threshold,
        update_stride=config.update_stride,
    )
    if len(updates) != len(triggers):
        raise RuntimeError(
            f"Replay produced {len(updates)} updates but {len(triggers)} were expected."
        )
    if np.any(model_revisions < 0) or np.any(np.diff(model_revisions) < 0):
        raise RuntimeError("Online replay model revisions are not monotonic.")

    np.savez_compressed(
        predictions_path,
        labels=online_labels,
        trial_ids=online_trials,
        baseline_probabilities=baseline_probabilities,
        adapted_probabilities=adapted_probabilities,
        model_revisions=model_revisions,
    )
    baseline_metrics = metric_bundle(
        online_labels,
        baseline_probabilities,
        online_trials,
        n_classes=n_classes,
    )
    adapted_metrics = metric_bundle(
        online_labels,
        adapted_probabilities,
        online_trials,
        n_classes=n_classes,
    )
    post_update_mask = model_revisions > 0
    post_update_comparison: dict[str, Any] | None = None
    if np.any(post_update_mask):
        post_baseline = metric_bundle(
            online_labels[post_update_mask],
            baseline_probabilities[post_update_mask],
            online_trials[post_update_mask],
            n_classes=n_classes,
        )
        post_adapted = metric_bundle(
            online_labels[post_update_mask],
            adapted_probabilities[post_update_mask],
            online_trials[post_update_mask],
            n_classes=n_classes,
        )
        post_update_comparison = {
            "first_window_id": int(np.flatnonzero(post_update_mask)[0] + 1),
            "last_window_id": int(np.flatnonzero(post_update_mask)[-1] + 1),
            "samples": int(np.sum(post_update_mask)),
            "baseline": post_baseline,
            "adapted": post_adapted,
            "delta_accuracy": float(
                post_adapted["window"]["accuracy"]
                - post_baseline["window"]["accuracy"]
            ),
            "delta_balanced_accuracy": float(
                post_adapted["window"]["balanced_accuracy"]
                - post_baseline["window"]["balanced_accuracy"]
            ),
        }
    report = {
        "schema_version": 1,
        "status": "completed_exploratory" if args.fixed_offline_epochs else "completed",
        "method": (
            (
                "chronological planned-trial split; use a predeclared exploratory "
                "offline epoch count; train every valid first-half window; "
            )
            if args.fixed_offline_epochs
            else (
                "chronological planned-trial split; select best epoch within first half; "
                "retrain on every valid first-half window; "
            )
        )
        + (
            "predict each second-half primary window before revealing its label"
        ),
        "identity": {
            "dataset": str(dataset_path),
            "dataset_sha256": sha256(dataset_path),
            "config": str(config_path),
            "config_sha256": sha256(config_path),
            "feature_key": args.feature_key,
            "split_trial": split_trial,
            "primary_window_index": args.primary_window_index,
            "history_threshold_override": args.history_threshold,
            "update_stride_override": args.update_stride,
            "fixed_offline_epochs": args.fixed_offline_epochs,
        },
        "preprocessing": {
            "shape": list(windows.shape),
            "sfreq": data["sfreq"],
            "quality_rejected_windows": data["quality_rejected_windows"],
        },
        "split": {
            "offline_planned_trials": f"{int(trial_ids.min())}-{split_trial - 1}",
            "online_planned_trials": f"{split_trial}-{int(trial_ids.max())}",
            "offline_valid_trials": int(np.unique(offline_groups).size),
            "offline_windows": int(offline_labels.size),
            "offline_trial_class_counts": _trial_class_counts(
                offline_labels, offline_groups, n_classes
            ),
            "offline_window_class_counts": _class_counts(offline_labels, n_classes),
            "online_valid_trials": int(np.unique(trial_ids[online_mask]).size),
            "online_all_valid_windows": int(np.sum(online_mask)),
            "online_primary_windows": int(online_labels.size),
            "online_primary_class_counts": _class_counts(online_labels, n_classes),
            "trial_overlap": [],
        },
        "configuration": asdict(config),
        "offline_calibration": {
            "epoch_selection_metrics": selection_metrics,
            "selected_epoch": selected_epoch,
            "full_first_half_fit_metrics": full_fit_metrics,
        },
        "online_replay": {
            "contract": "one primary window per trial; causal predict-then-label-then-update",
            "baseline": baseline_metrics,
            "adapted": adapted_metrics,
            "updates": updates,
            "update_count": len(updates),
            "post_update_comparison": post_update_comparison,
            "expected_update_triggers": triggers,
            "additional_primary_windows_to_first_update": max(
                config.history_threshold - len(online_labels), 0
            ),
            "metrics_identical_without_update": bool(
                np.array_equal(baseline_probabilities, adapted_probabilities)
            ),
        },
        "artifacts": {
            "checkpoint": artifact(checkpoint_path),
            "checkpoint_sidecar": artifact(Path(f"{checkpoint_path}.neuroonline.pt")),
            "predictions": artifact(predictions_path),
        },
        "duration_sec": float(time.perf_counter() - started),
    }
    _save_json(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--feature-key", default="processed_windows")
    parser.add_argument("--split-trial", type=int, default=None)
    parser.add_argument("--primary-window-index", type=int, default=0)
    parser.add_argument("--history-threshold", type=int, default=None)
    parser.add_argument("--update-stride", type=int, default=None)
    parser.add_argument(
        "--fixed-offline-epochs",
        type=int,
        default=None,
        help=(
            "Exploratory mode: skip first-half validation and train all first-half "
            "windows for this predeclared epoch count."
        ),
    )
    return parser.parse_args()


def main() -> None:
    report = run(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
