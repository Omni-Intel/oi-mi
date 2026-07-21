"""Periodic supervised adaptation for realtime EEG decoders.

The live decoder keeps predicting with the current model while this component
trains a cloned candidate on a detached batch.  A candidate is only swapped in
after it passes a group-aware holdout check.
"""

from __future__ import annotations

import copy
import json
import logging
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.metrics import balanced_accuracy_score

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BatchAdaptationConfig:
    enabled: bool = False
    update_interval_sec: float = 600.0
    learning_rate: float = 1e-4
    epochs: int = 3
    batch_size: int = 32
    train_scope: str = "head"
    min_total_windows: int = 180
    min_windows_per_class: int = 30
    validation_ratio: float = 0.2
    min_balanced_accuracy_gain: float = 0.02
    max_class_accuracy_drop: float = 0.05
    max_buffer_windows: int = 1800
    keep_previous_versions: int = 5
    random_seed: int = 17
    save_update_dataset: bool = True

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "BatchAdaptationConfig":
        data = payload or {}
        train_scope = str(data.get("train_scope", "head")).strip().lower()
        if train_scope != "head":
            raise ValueError("Periodic online adaptation currently supports train_scope: head only.")
        return cls(
            enabled=bool(data.get("enabled", False)),
            update_interval_sec=max(float(data.get("update_interval_sec", 600.0)), 1.0),
            learning_rate=max(float(data.get("learning_rate", 1e-4)), 1e-8),
            epochs=max(int(data.get("epochs", 3)), 1),
            batch_size=max(int(data.get("batch_size", 32)), 1),
            train_scope=train_scope,
            min_total_windows=max(int(data.get("min_total_windows", 180)), 3),
            min_windows_per_class=max(int(data.get("min_windows_per_class", 30)), 1),
            validation_ratio=min(max(float(data.get("validation_ratio", 0.2)), 0.05), 0.5),
            min_balanced_accuracy_gain=float(data.get("min_balanced_accuracy_gain", 0.02)),
            max_class_accuracy_drop=max(float(data.get("max_class_accuracy_drop", 0.05)), 0.0),
            max_buffer_windows=max(int(data.get("max_buffer_windows", 1800)), 3),
            keep_previous_versions=max(int(data.get("keep_previous_versions", 5)), 1),
            random_seed=int(data.get("random_seed", 17)),
            save_update_dataset=bool(data.get("save_update_dataset", True)),
        )


@dataclass(slots=True)
class _Sample:
    window: np.ndarray
    label: int
    event_id: str


class OnlineBatchAdapter:
    """Collect labeled windows and train a validated candidate periodically."""

    def __init__(
        self,
        *,
        config: BatchAdaptationConfig,
        model_getter: Callable[[], Any],
        model_swapper: Callable[[Any], None],
        model_save_path: Path,
        n_classes: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._model_getter = model_getter
        self._model_swapper = model_swapper
        self._model_save_path = Path(model_save_path)
        self._versions_dir = self._model_save_path.parent / f"{self._model_save_path.stem}_updates"
        self._n_classes = int(n_classes)
        self._clock = clock
        self._lock = threading.Lock()
        self._samples: list[_Sample] = []
        self._worker: threading.Thread | None = None
        self._closed = False
        self._cycle = 0
        self._model_version = self._discover_latest_version()
        self._next_update_at = self._clock() + self.config.update_interval_sec
        self._status: dict[str, Any] = {
            "enabled": self.config.enabled,
            "state": "collecting" if self.config.enabled else "disabled",
            "model_version": self._model_version,
            "buffered_windows": 0,
            "class_counts": {str(index): 0 for index in range(self._n_classes)},
            "seconds_until_update": self.config.update_interval_sec,
            "last_result": None,
        }

    def add_window(
        self,
        window: np.ndarray,
        label: int,
        *,
        event_id: str,
        now: float | None = None,
    ) -> None:
        if not self.config.enabled or self._closed:
            return
        label = int(label)
        if label < 0 or label >= self._n_classes:
            return
        timestamp = self._clock() if now is None else float(now)
        with self._lock:
            self._samples.append(
                _Sample(
                    window=np.asarray(window, dtype=np.float32).copy(),
                    label=label,
                    event_id=str(event_id),
                )
            )
            overflow = len(self._samples) - self.config.max_buffer_windows
            if overflow > 0:
                del self._samples[:overflow]
            self._refresh_status_locked(timestamp)
        self.maybe_start_update(now=timestamp)

    def maybe_start_update(self, *, now: float | None = None, force: bool = False) -> bool:
        if not self.config.enabled or self._closed:
            return False
        timestamp = self._clock() if now is None else float(now)
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                self._refresh_status_locked(timestamp)
                return False
            if not force and timestamp < self._next_update_at:
                self._refresh_status_locked(timestamp)
                return False

            self._next_update_at = timestamp + self.config.update_interval_sec
            readiness_error = self._readiness_error_locked()
            if readiness_error is not None:
                self._status["state"] = "waiting_for_data"
                self._status["last_result"] = readiness_error
                self._refresh_status_locked(timestamp)
                return False

            samples = self._samples
            self._samples = []
            self._cycle += 1
            cycle = self._cycle
            self._status["state"] = "training"
            LOGGER.info("Starting periodic adaptation cycle %s with %s windows", cycle, len(samples))
            self._refresh_status_locked(timestamp)
            self._worker = threading.Thread(
                target=self._train_candidate,
                args=(samples, cycle),
                name=f"online-batch-update-{cycle}",
                daemon=True,
            )
            self._worker.start()
            return True

    def status(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_status_locked(self._clock())
            return copy.deepcopy(self._status)

    def close(self, *, timeout_sec: float = 30.0) -> None:
        self._closed = True
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=max(float(timeout_sec), 0.0))

    def _readiness_error_locked(self) -> str | None:
        counts = Counter(sample.label for sample in self._samples)
        if len(self._samples) < self.config.min_total_windows:
            return f"有效窗口不足: {len(self._samples)}/{self.config.min_total_windows}"
        missing = [
            f"class-{label}={counts.get(label, 0)}"
            for label in range(self._n_classes)
            if counts.get(label, 0) < self.config.min_windows_per_class
        ]
        if missing:
            return "类别窗口不足: " + ", ".join(missing)
        groups_by_class: dict[int, set[str]] = defaultdict(set)
        for sample in self._samples:
            groups_by_class[sample.label].add(sample.event_id)
        if any(len(groups_by_class[label]) < 2 for label in range(self._n_classes)):
            return "每个类别至少需要2个独立标签事件，才能按事件划分训练/验证集"
        return None

    def _train_candidate(self, samples: list[_Sample], cycle: int) -> None:
        result: dict[str, Any]
        try:
            dataset_path = self._save_cycle_dataset(samples, cycle) if self.config.save_update_dataset else None
            train_indices, validation_indices = self._split_by_event(samples, cycle)
            X_train = np.stack([samples[index].window for index in train_indices]).astype(np.float32)
            y_train = np.asarray([samples[index].label for index in train_indices], dtype=np.int64)
            X_validation = np.stack([samples[index].window for index in validation_indices]).astype(np.float32)
            y_validation = np.asarray([samples[index].label for index in validation_indices], dtype=np.int64)

            # The getter returns an isolated snapshot so training never mutates
            # the model currently serving realtime predictions.
            candidate = self._model_getter()
            before = self._evaluate(candidate, X_validation, y_validation)
            update_metrics = candidate.update(
                X_train,
                y_train,
                learning_rate=self.config.learning_rate,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
            )
            after = self._evaluate(candidate, X_validation, y_validation)
            gain = after["balanced_accuracy"] - before["balanced_accuracy"]
            class_drops = [
                before["per_class_accuracy"][str(label)] - after["per_class_accuracy"][str(label)]
                for label in range(self._n_classes)
            ]
            accepted = (
                gain >= self.config.min_balanced_accuracy_gain
                and max(class_drops, default=0.0) <= self.config.max_class_accuracy_drop
            )
            result = {
                "cycle": cycle,
                "accepted": accepted,
                "training_windows": int(len(train_indices)),
                "validation_windows": int(len(validation_indices)),
                "class_counts": dict(Counter(sample.label for sample in samples)),
                "before": before,
                "after": after,
                "balanced_accuracy_gain": gain,
                "update_metrics": update_metrics,
                "completed_at": datetime.now().isoformat(timespec="seconds"),
            }
            if dataset_path is not None:
                result["dataset_path"] = str(dataset_path)
            if accepted:
                self._model_version += 1
                result["model_version"] = self._model_version
                self._save_accepted_candidate(candidate, result)
                self._model_swapper(candidate)
                LOGGER.info(
                    "Accepted adaptation cycle %s as model v%s (balanced accuracy gain %.4f)",
                    cycle,
                    self._model_version,
                    gain,
                )
            else:
                result["model_version"] = self._model_version
                self._write_metadata(result, accepted=False)
                LOGGER.info("Rejected adaptation cycle %s (balanced accuracy gain %.4f)", cycle, gain)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Periodic online adaptation failed")
            result = {
                "cycle": cycle,
                "accepted": False,
                "error": str(exc),
                "completed_at": datetime.now().isoformat(timespec="seconds"),
                "model_version": self._model_version,
            }

        with self._lock:
            self._status["state"] = "collecting"
            self._status["model_version"] = self._model_version
            self._status["last_result"] = result
            self._refresh_status_locked(self._clock())

    def _split_by_event(self, samples: list[_Sample], cycle: int) -> tuple[list[int], list[int]]:
        event_indices: dict[str, list[int]] = defaultdict(list)
        event_labels: dict[str, int] = {}
        for index, sample in enumerate(samples):
            event_indices[sample.event_id].append(index)
            previous = event_labels.setdefault(sample.event_id, sample.label)
            if previous != sample.label:
                raise ValueError(f"标签事件 {sample.event_id!r} 包含多个类别")

        groups_by_class: dict[int, list[str]] = defaultdict(list)
        for event_id, label in event_labels.items():
            groups_by_class[label].append(event_id)

        rng = np.random.default_rng(self.config.random_seed + cycle)
        validation_groups: set[str] = set()
        for label in range(self._n_classes):
            groups = sorted(groups_by_class[label])
            rng.shuffle(groups)
            validation_count = max(1, int(round(len(groups) * self.config.validation_ratio)))
            validation_count = min(validation_count, len(groups) - 1)
            validation_groups.update(groups[:validation_count])

        train_indices: list[int] = []
        validation_indices: list[int] = []
        for event_id, indices in event_indices.items():
            target = validation_indices if event_id in validation_groups else train_indices
            target.extend(indices)
        return train_indices, validation_indices

    def _evaluate(self, model: Any, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        probabilities = model.predict_proba(X, mc_dropout_passes=1)
        predictions = np.argmax(probabilities, axis=1).astype(np.int64)
        per_class: dict[str, float] = {}
        for label in range(self._n_classes):
            mask = y == label
            per_class[str(label)] = float(np.mean(predictions[mask] == y[mask])) if np.any(mask) else 0.0
        return {
            "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
            "accuracy": float(np.mean(predictions == y)),
            "per_class_accuracy": per_class,
        }

    def _save_accepted_candidate(self, candidate: Any, result: dict[str, Any]) -> None:
        self._versions_dir.mkdir(parents=True, exist_ok=True)
        suffix = self._model_save_path.suffix or ".pt"
        version_path = self._versions_dir / f"update_{self._model_version:03d}{suffix}"
        candidate.save(version_path)
        self._model_save_path.parent.mkdir(parents=True, exist_ok=True)
        candidate.save(self._model_save_path)
        result["model_path"] = str(version_path)
        self._write_metadata(result, accepted=True)
        self._prune_old_versions(suffix)

    def _save_cycle_dataset(self, samples: list[_Sample], cycle: int) -> Path:
        self._versions_dir.mkdir(parents=True, exist_ok=True)
        path = self._versions_dir / f"cycle_{cycle:03d}_dataset.npz"
        np.savez_compressed(
            path,
            processed_windows=np.stack([sample.window for sample in samples]).astype(np.float32),
            labels=np.asarray([sample.label for sample in samples], dtype=np.int64),
            event_ids=np.asarray([sample.event_id for sample in samples], dtype=np.str_),
        )
        return path

    def _write_metadata(self, result: dict[str, Any], *, accepted: bool) -> None:
        self._versions_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"update_{self._model_version:03d}" if accepted else f"cycle_{int(result['cycle']):03d}_rejected"
        with (self._versions_dir / f"{prefix}.json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)

    def _prune_old_versions(self, suffix: str) -> None:
        versions = sorted(self._versions_dir.glob(f"update_*{suffix}"))
        for path in versions[: -self.config.keep_previous_versions]:
            path.unlink(missing_ok=True)

    def _discover_latest_version(self) -> int:
        if not self._versions_dir.exists():
            return 0
        versions: list[int] = []
        for path in self._versions_dir.glob("update_*.json"):
            try:
                versions.append(int(path.stem.split("_")[1]))
            except (IndexError, ValueError):
                continue
        return max(versions, default=0)

    def _refresh_status_locked(self, now: float) -> None:
        counts = Counter(sample.label for sample in self._samples)
        self._status["buffered_windows"] = len(self._samples)
        self._status["class_counts"] = {
            str(label): int(counts.get(label, 0)) for label in range(self._n_classes)
        }
        self._status["seconds_until_update"] = max(self._next_update_at - now, 0.0)
