"""Trial-grouped hyperparameter selection for NeuroOnline calibration."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
import gc
import json
import os
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

from adaptation.neuroonline import (
    ClassCollapseError,
    NEUROONLINE_TRAINING_MECHANICS_VERSION,
    NeuroOnlineConfig,
    NeuroOnlineModelAdapter,
)
from models.factory import TorchModelAdapter, split_train_validation_indices


@dataclass(frozen=True, slots=True)
class CalibrationSearchConfig:
    """Search or reuse policy for NeuroOnline calibration hyperparameters."""

    enabled: bool = False
    reuse_latest: bool = False
    mode: str = "full_grid"
    selection_epochs: int = 50
    selection_patience: int = 50
    learning_rates: tuple[float, ...] = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3)
    batch_sizes: tuple[int, ...] = (16, 32, 128, 256)
    mask_ratios: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7)
    consistency_weights: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 1.0, 1.5)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "CalibrationSearchConfig":
        root = payload or {}
        neuroonline = root.get("neuroonline", {}) or {}
        data = neuroonline.get("calibration_search", {}) or {}
        defaults = cls()
        mode = str(data.get("mode", "full_grid")).strip().lower()
        if mode not in {"full_grid", "staged"}:
            raise ValueError("Calibration-search mode must be 'full_grid' or 'staged'.")
        enabled = bool(data.get("enabled", False))
        reuse_latest = bool(data.get("reuse_latest", False))
        if enabled and reuse_latest:
            raise ValueError(
                "Calibration search cannot be enabled while reuse_latest is active."
            )
        return cls(
            enabled=enabled,
            reuse_latest=reuse_latest,
            mode=mode,
            selection_epochs=max(int(data.get("selection_epochs", 50)), 1),
            selection_patience=max(int(data.get("selection_patience", 50)), 1),
            learning_rates=_positive_float_tuple(
                data.get("learning_rates"),
                defaults.learning_rates,
            ),
            batch_sizes=_positive_int_tuple(
                data.get("batch_sizes"),
                defaults.batch_sizes,
            ),
            mask_ratios=_bounded_float_tuple(
                data.get("mask_ratios"),
                defaults.mask_ratios,
                lower=0.0,
                upper=1.0,
            ),
            consistency_weights=_bounded_float_tuple(
                data.get("consistency_weights"),
                defaults.consistency_weights,
                lower=0.0,
                upper=None,
            ),
        )


@dataclass(slots=True)
class CalibrationSearchResult:
    best_config: NeuroOnlineConfig
    report: dict[str, Any]
    report_path: Path | None


def load_latest_calibration_search(
    *,
    calibration_records_dir: Path | None,
    base_config: NeuroOnlineConfig,
) -> tuple[NeuroOnlineConfig, dict[str, Any], Path]:
    """Load the newest completed search report without reusing model weights."""

    if calibration_records_dir is None:
        raise RuntimeError(
            "Cannot reuse calibration hyperparameters because the calibration "
            "records directory is not configured."
        )
    reports = sorted(
        calibration_records_dir.glob("*/hyperparameter_search.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if not reports:
        raise RuntimeError(
            "No previous hyperparameter_search.json was found under "
            f"{calibration_records_dir}. Copy the completed search session here "
            "or enable calibration_search.enabled for one new search."
        )

    validation_errors: list[str] = []
    for report_path in reports:
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if not isinstance(report, dict):
                raise ValueError("report root is not an object")
            report_version = int(report.get("training_mechanics_version", 0))
            if report_version != NEUROONLINE_TRAINING_MECHANICS_VERSION:
                raise ValueError(
                    "training mechanics version "
                    f"{report_version} is incompatible with current version "
                    f"{NEUROONLINE_TRAINING_MECHANICS_VERSION}"
                )
            parameters = _validated_selected_parameters(report.get("best_parameters"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            validation_errors.append(f"{report_path}: {exc}")
            continue
        return _config_from_record(base_config, parameters), report, report_path

    details = "; ".join(validation_errors[:3])
    raise RuntimeError(
        "Previous calibration search reports were found, but none contains a "
        f"complete valid best_parameters set. {details}"
    )


def run_calibration_search(
    *,
    base_template: TorchModelAdapter,
    base_config: NeuroOnlineConfig,
    search_config: CalibrationSearchConfig,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    session_dir: Path | None,
    progress_callback: Callable[[int, int, str, dict[str, float]], None] | None = None,
) -> CalibrationSearchResult:
    """Select offline parameters without leaking windows across trial groups."""

    labels = np.asarray(y, dtype=np.int64)
    trial_groups = np.asarray(groups, dtype=np.int64)
    if labels.shape != trial_groups.shape:
        raise ValueError("Calibration search groups must match the labels shape.")
    _validate_trial_groups(labels, trial_groups)

    all_indices = np.arange(labels.size, dtype=np.int64)
    development_indices, holdout_indices = split_train_validation_indices(
        labels,
        groups=trial_groups,
        random_state=base_config.random_seed,
    )
    inner_train_relative, inner_validation_relative = split_train_validation_indices(
        labels[development_indices],
        groups=trial_groups[development_indices],
        random_state=base_config.random_seed + 1,
    )
    train_indices = development_indices[inner_train_relative]
    validation_indices = development_indices[inner_validation_relative]
    _validate_disjoint_groups(
        trial_groups,
        train_indices=train_indices,
        validation_indices=validation_indices,
        holdout_indices=holdout_indices,
    )

    candidates: list[dict[str, Any]] = []
    best_adapter: NeuroOnlineModelAdapter | None = None
    best_config = base_config
    best_rank: tuple[float, ...] | None = None
    started_at = time.perf_counter()

    optimizer_candidates = _unique_candidates(
        replace(
            base_config,
            offline_learning_rate=learning_rate,
            offline_batch_size=batch_size,
            offline_epochs=search_config.selection_epochs,
            offline_patience=search_config.selection_patience,
        )
        for learning_rate in search_config.learning_rates
        for batch_size in search_config.batch_sizes
    )
    if search_config.mode == "full_grid":
        search_candidates = _unique_candidates(
            replace(
                base_config,
                offline_learning_rate=learning_rate,
                offline_batch_size=batch_size,
                offline_mask_ratio=mask_ratio,
                offline_consistency_weight=consistency_weight,
                offline_epochs=search_config.selection_epochs,
                offline_patience=search_config.selection_patience,
            )
            for learning_rate in search_config.learning_rates
            for batch_size in search_config.batch_sizes
            for mask_ratio in search_config.mask_ratios
            for consistency_weight in search_config.consistency_weights
        )
        max_candidates = len(search_candidates)
    else:
        base_offline_mask_ratio = (
            base_config.mask_ratio
            if base_config.offline_mask_ratio is None
            else base_config.offline_mask_ratio
        )
        base_offline_consistency_weight = (
            base_config.consistency_weight
            if base_config.offline_consistency_weight is None
            else base_config.offline_consistency_weight
        )
        base_pair_repeated = (
            base_offline_mask_ratio in search_config.mask_ratios
            and base_offline_consistency_weight in search_config.consistency_weights
        )
        max_candidates = (
            len(optimizer_candidates)
            + len(search_config.mask_ratios) * len(search_config.consistency_weights)
            - int(base_pair_repeated)
        )

    def evaluate(candidate_config: NeuroOnlineConfig, stage: str) -> None:
        nonlocal best_adapter, best_config, best_rank
        adapter = NeuroOnlineModelAdapter(
            copy.deepcopy(base_template),
            config=candidate_config,
            state_path=None,
        )
        candidate_number = len(candidates) + 1

        def epoch_progress(
            current_epoch: int,
            total_epochs: int,
            metrics: dict[str, float],
        ) -> None:
            if progress_callback is not None:
                progress_callback(
                    candidate_number,
                    max_candidates,
                    f"{stage} epoch {current_epoch}/{total_epochs}",
                    metrics,
                )

        candidate_started = time.perf_counter()
        try:
            metrics = adapter.fit_with_split(
                X,
                labels,
                train_indices=train_indices,
                validation_indices=validation_indices,
                groups=trial_groups,
                progress_callback=epoch_progress,
            )
        except ClassCollapseError as exc:
            candidates.append(
                {
                    "candidate_index": candidate_number,
                    "stage": stage,
                    "parameters": _selected_parameters(candidate_config),
                    "metrics": _finite_mapping(exc.metrics),
                    "duration_sec": time.perf_counter() - candidate_started,
                    "rejected": "class_collapse",
                }
            )
            del adapter
            _release_unused_memory()
            return
        record = {
            "candidate_index": candidate_number,
            "stage": stage,
            "parameters": _selected_parameters(candidate_config),
            "metrics": _finite_mapping(metrics),
            "duration_sec": time.perf_counter() - candidate_started,
        }
        candidates.append(record)
        rank = _metric_rank(metrics)
        if best_rank is None or rank > best_rank:
            if best_adapter is not None:
                del best_adapter
            best_adapter = adapter
            best_config = candidate_config
            best_rank = rank
        else:
            del adapter
        _release_unused_memory()

    if search_config.mode == "full_grid":
        for candidate in search_candidates:
            evaluate(candidate, "full_grid")
    else:
        for candidate in optimizer_candidates:
            evaluate(candidate, "optimizer")

        representation_candidates = _unique_candidates(
            replace(
                best_config,
                offline_mask_ratio=mask_ratio,
                offline_consistency_weight=consistency_weight,
            )
            for mask_ratio in search_config.mask_ratios
            for consistency_weight in search_config.consistency_weights
        )
        previously_evaluated = {
            _candidate_key(_config_from_record(base_config, record["parameters"]))
            for record in candidates
        }
        for candidate in representation_candidates:
            if _candidate_key(candidate) not in previously_evaluated:
                evaluate(candidate, "representation")

    if best_adapter is None:
        raise RuntimeError("Calibration hyperparameter search produced no candidate.")
    holdout_probabilities = best_adapter.predict_proba(X[holdout_indices])
    holdout_window_metrics = _classification_metrics(
        labels[holdout_indices],
        holdout_probabilities.argmax(axis=1),
    )
    holdout_trial_truth, holdout_trial_probabilities = _aggregate_trial_probabilities(
        labels[holdout_indices],
        holdout_probabilities,
        trial_groups[holdout_indices],
    )
    holdout_trial_metrics = _classification_metrics(
        holdout_trial_truth,
        holdout_trial_probabilities.argmax(axis=1),
    )
    holdout_collapsed = (
        float(holdout_trial_metrics["worst_class_accuracy"]) <= 0.0
    )
    final_config = replace(
        best_config,
        offline_epochs=base_config.offline_epochs,
        offline_patience=base_config.offline_patience,
    )
    report = {
        "schema_version": 2,
        "training_mechanics_version": NEUROONLINE_TRAINING_MECHANICS_VERSION,
        "selection_method": f"{search_config.mode}_trial_grouped_search",
        "ranking": [
            "validation_trial_worst_class_accuracy",
            "validation_window_worst_class_accuracy",
            "validation_trial_balanced_accuracy",
            "validation_trial_kappa",
            "validation_trial_macro_f1",
            "validation_window_balanced_accuracy",
            "negative_validation_loss",
        ],
        "random_seed": base_config.random_seed,
        "split": {
            "all_windows": int(all_indices.size),
            "all_trials": int(np.unique(trial_groups).size),
            "train_windows": int(train_indices.size),
            "train_trials": int(np.unique(trial_groups[train_indices]).size),
            "selection_validation_windows": int(validation_indices.size),
            "selection_validation_trials": int(
                np.unique(trial_groups[validation_indices]).size
            ),
            "untouched_holdout_windows": int(holdout_indices.size),
            "untouched_holdout_trials": int(
                np.unique(trial_groups[holdout_indices]).size
            ),
        },
        "search_config": asdict(search_config),
        "candidates": candidates,
        "best_parameters": _selected_parameters(final_config),
        "untouched_holdout_metrics": holdout_trial_metrics,
        "untouched_holdout_trial_metrics": holdout_trial_metrics,
        "untouched_holdout_window_metrics": holdout_window_metrics,
        "deployment_eligible": not holdout_collapsed,
        "duration_sec": time.perf_counter() - started_at,
    }
    if holdout_collapsed:
        report["rejected"] = "untouched_holdout_class_collapse"
    report_path = None
    if session_dir is not None:
        report_path = session_dir / "hyperparameter_search.json"
        _write_json_atomic(report_path, report)
    if holdout_collapsed:
        del best_adapter
        _release_unused_memory()
        raise ClassCollapseError(
            "The selected NeuroOnline hyperparameters collapse at least one "
            "class on the untouched trial holdout.",
            {
                f"holdout_{name}": float(value)
                for name, value in holdout_trial_metrics.items()
                if isinstance(value, (int, float))
            },
        )
    del best_adapter
    _release_unused_memory()
    return CalibrationSearchResult(
        best_config=final_config,
        report=report,
        report_path=report_path,
    )


def _classification_metrics(truth: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    labels = np.unique(np.concatenate((truth, predictions)))
    matrix = confusion_matrix(truth, predictions, labels=labels)
    totals = matrix.sum(axis=1)
    per_class = np.divide(
        np.diag(matrix),
        totals,
        out=np.zeros_like(totals, dtype=np.float64),
        where=totals > 0,
    )
    kappa = float(cohen_kappa_score(truth, predictions))
    if not np.isfinite(kappa):
        kappa = -1.0
    return {
        "accuracy": float(np.mean(truth == predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "kappa": kappa,
        "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "worst_class_accuracy": float(np.min(per_class)),
        "per_class_recall": {
            str(int(label)): float(per_class[index])
            for index, label in enumerate(labels)
        },
        "confusion_matrix": matrix.astype(int).tolist(),
        "labels": labels.astype(int).tolist(),
    }


def _metric_rank(metrics: dict[str, float]) -> tuple[float, ...]:
    return (
        float(metrics["val_trial_worst_class_accuracy"]),
        float(metrics["val_worst_class_accuracy"]),
        float(metrics["val_trial_balanced_accuracy"]),
        float(metrics["val_trial_kappa"]),
        float(metrics["val_trial_macro_f1"]),
        float(metrics["val_balanced_accuracy"]),
        -float(metrics["val_loss"]),
    )


def _selected_parameters(config: NeuroOnlineConfig) -> dict[str, Any]:
    return {
        "offline_learning_rate": config.offline_learning_rate,
        "offline_batch_size": config.offline_batch_size,
        "mask_ratio": (
            config.mask_ratio
            if config.offline_mask_ratio is None
            else config.offline_mask_ratio
        ),
        "consistency_weight": (
            config.consistency_weight
            if config.offline_consistency_weight is None
            else config.offline_consistency_weight
        ),
        "weight_decay": config.weight_decay,
        "label_smoothing": config.label_smoothing,
        "offline_epochs": config.offline_epochs,
        "offline_patience": config.offline_patience,
    }


def _validated_selected_parameters(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("best_parameters is missing or is not an object")
    required = {
        "offline_learning_rate",
        "offline_batch_size",
        "mask_ratio",
        "consistency_weight",
        "weight_decay",
        "label_smoothing",
        "offline_epochs",
        "offline_patience",
    }
    missing = sorted(required - value.keys())
    if missing:
        raise ValueError(f"best_parameters is missing: {', '.join(missing)}")
    parameters = {
        "offline_learning_rate": float(value["offline_learning_rate"]),
        "offline_batch_size": int(value["offline_batch_size"]),
        "mask_ratio": float(value["mask_ratio"]),
        "consistency_weight": float(value["consistency_weight"]),
        "weight_decay": float(value["weight_decay"]),
        "label_smoothing": float(value["label_smoothing"]),
        "offline_epochs": int(value["offline_epochs"]),
        "offline_patience": int(value["offline_patience"]),
    }
    if parameters["offline_learning_rate"] <= 0:
        raise ValueError("offline_learning_rate must be positive")
    if parameters["offline_batch_size"] <= 0:
        raise ValueError("offline_batch_size must be positive")
    if not 0.0 <= parameters["mask_ratio"] <= 1.0:
        raise ValueError("mask_ratio must be between 0 and 1")
    if parameters["consistency_weight"] < 0:
        raise ValueError("consistency_weight must be non-negative")
    if parameters["weight_decay"] < 0:
        raise ValueError("weight_decay must be non-negative")
    if not 0.0 <= parameters["label_smoothing"] <= 1.0:
        raise ValueError("label_smoothing must be between 0 and 1")
    if parameters["offline_epochs"] <= 0 or parameters["offline_patience"] <= 0:
        raise ValueError("offline epochs and patience must be positive")
    return parameters


def _config_from_record(
    base: NeuroOnlineConfig,
    parameters: dict[str, Any],
) -> NeuroOnlineConfig:
    record = dict(parameters)
    if "mask_ratio" in record:
        record["offline_mask_ratio"] = record.pop("mask_ratio")
    if "consistency_weight" in record:
        record["offline_consistency_weight"] = record.pop("consistency_weight")
    return replace(base, **record)


def _candidate_key(config: NeuroOnlineConfig) -> tuple[float, int, float, float]:
    return (
        config.offline_learning_rate,
        config.offline_batch_size,
        (
            config.mask_ratio
            if config.offline_mask_ratio is None
            else config.offline_mask_ratio
        ),
        (
            config.consistency_weight
            if config.offline_consistency_weight is None
            else config.offline_consistency_weight
        ),
    )


def _unique_candidates(
    candidates: Any,
) -> list[NeuroOnlineConfig]:
    unique: list[NeuroOnlineConfig] = []
    seen: set[tuple[float, int, float, float]] = set()
    for candidate in candidates:
        key = _candidate_key(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _validate_trial_groups(labels: np.ndarray, groups: np.ndarray) -> None:
    for group in np.unique(groups):
        group_labels = np.unique(labels[groups == group])
        if group_labels.size != 1:
            raise ValueError(
                f"Calibration trial {int(group)} contains multiple labels: "
                f"{group_labels.tolist()}."
            )


def _aggregate_trial_probabilities(
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
            raise ValueError(
                f"Calibration trial {int(group)} contains multiple labels: "
                f"{unique_labels.tolist()}."
            )
        trial_labels.append(int(unique_labels[0]))
        trial_probabilities.append(np.mean(probabilities[mask], axis=0))
    return (
        np.asarray(trial_labels, dtype=np.int64),
        np.stack(trial_probabilities, axis=0),
    )


def _validate_disjoint_groups(
    groups: np.ndarray,
    *,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    holdout_indices: np.ndarray,
) -> None:
    partitions = [
        set(groups[train_indices].tolist()),
        set(groups[validation_indices].tolist()),
        set(groups[holdout_indices].tolist()),
    ]
    if any(partitions[left] & partitions[right] for left in range(3) for right in range(left + 1, 3)):
        raise RuntimeError("Trial leakage detected in calibration hyperparameter search.")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _finite_mapping(metrics: dict[str, float]) -> dict[str, float]:
    return {
        key: float(value) if np.isfinite(float(value)) else -1.0
        for key, value in metrics.items()
    }


def _release_unused_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _positive_float_tuple(value: Any, default: tuple[float, ...]) -> tuple[float, ...]:
    values = default if value is None else tuple(float(item) for item in value)
    filtered = tuple(item for item in values if item > 0)
    if not filtered:
        raise ValueError("Calibration-search learning rates must contain a positive value.")
    return filtered


def _positive_int_tuple(value: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    values = default if value is None else tuple(int(item) for item in value)
    filtered = tuple(item for item in values if item > 0)
    if not filtered:
        raise ValueError("Calibration-search batch sizes must contain a positive value.")
    return filtered


def _bounded_float_tuple(
    value: Any,
    default: tuple[float, ...],
    *,
    lower: float,
    upper: float | None,
) -> tuple[float, ...]:
    values = default if value is None else tuple(float(item) for item in value)
    if not values or any(item < lower or (upper is not None and item > upper) for item in values):
        raise ValueError("Calibration-search values are outside the supported range.")
    return values
