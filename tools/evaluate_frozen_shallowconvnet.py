"""Train a plain ShallowConvNet on calibration data and evaluate it frozen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.factory import ModelFactory, TorchModelAdapter, split_train_validation_indices
from utils.preprocessing import preprocess_eeg_window


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def save_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_committed_realtime(
    recording_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    chunk_paths = sorted((recording_dir / "chunks").glob("chunk_*.npz"))
    if not chunk_paths:
        raise FileNotFoundError(f"No realtime chunks found under {recording_dir}.")

    required = {
        "eeg_windows",
        "labels_true",
        "scene_indices",
        "label_event_ids",
        "window_end_monotonic",
        "training_roles",
        "adaptation_committed",
    }
    windows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    scenes: list[np.ndarray] = []
    event_ids: list[np.ndarray] = []
    timestamps: list[np.ndarray] = []
    sources: list[dict[str, Any]] = []
    for chunk_path in chunk_paths:
        with np.load(chunk_path, allow_pickle=False) as payload:
            missing = sorted(required - set(payload.files))
            if missing:
                raise ValueError(f"{chunk_path.name} is missing: {', '.join(missing)}")
            committed = payload["adaptation_committed"].astype(bool)
            roles = payload["training_roles"][committed].astype(str)
            if np.any(roles != "primary_decision"):
                raise ValueError(f"{chunk_path.name} contains committed non-primary windows.")
            chunk_labels = payload["labels_true"][committed].astype(np.int64)
            if np.any(chunk_labels < 0):
                raise ValueError(f"{chunk_path.name} contains committed windows without labels.")
            windows.append(payload["eeg_windows"][committed].astype(np.float32))
            labels.append(chunk_labels)
            scenes.append(payload["scene_indices"][committed].astype(np.int64))
            event_ids.append(payload["label_event_ids"][committed].astype(str))
            timestamps.append(payload["window_end_monotonic"][committed].astype(np.float64))
            sources.append(
                {
                    "path": str(chunk_path.resolve()),
                    "sha256": sha256_file(chunk_path),
                    "committed_windows": int(committed.sum()),
                }
            )

    result = (
        np.concatenate(windows),
        np.concatenate(labels),
        np.concatenate(scenes),
        np.concatenate(event_ids),
        np.concatenate(timestamps),
        sources,
    )
    if not np.all(np.diff(result[4]) > 0.0):
        raise ValueError("Committed realtime windows are not strictly chronological.")
    return result


def classification_metrics(
    truth: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_classes: int,
) -> dict[str, Any]:
    truth = np.asarray(truth, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if truth.size == 0:
        return {"samples": 0}
    predictions = probabilities.argmax(axis=1)
    labels = np.arange(n_classes, dtype=np.int64)
    matrix = confusion_matrix(truth, predictions, labels=labels)
    support = matrix.sum(axis=1)
    recalls = np.divide(
        np.diag(matrix),
        support,
        out=np.zeros(n_classes, dtype=np.float64),
        where=support > 0,
    )
    observed = support > 0
    true_probabilities = probabilities[np.arange(truth.size), truth]
    kappa = float(cohen_kappa_score(truth, predictions))
    if not np.isfinite(kappa):
        kappa = -1.0
    return {
        "samples": int(truth.size),
        "correct": int(np.sum(predictions == truth)),
        "accuracy": float(np.mean(predictions == truth)),
        "balanced_accuracy": float(np.mean(recalls[observed])),
        "macro_f1": float(
            f1_score(truth, predictions, average="macro", zero_division=0)
        ),
        "kappa": kappa,
        "cross_entropy": float(
            -np.log(np.clip(true_probabilities, 1e-12, 1.0)).mean()
        ),
        "true_class_counts": support.astype(int).tolist(),
        "predicted_class_counts": np.bincount(
            predictions, minlength=n_classes
        ).astype(int).tolist(),
        "per_class_recall": recalls.tolist(),
        "confusion_matrix": matrix.astype(int).tolist(),
        "all_classes_predicted": bool(
            set(np.unique(predictions).tolist()) == set(labels.tolist())
        ),
    }


def aggregate_trial_probabilities(
    truth: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    trial_truth: list[int] = []
    trial_probabilities: list[np.ndarray] = []
    for group in dict.fromkeys(np.asarray(groups, dtype=np.int64).tolist()):
        mask = groups == group
        labels = np.unique(truth[mask])
        if labels.size != 1:
            raise ValueError(f"Trial group {group} contains multiple labels.")
        trial_truth.append(int(labels[0]))
        trial_probabilities.append(probabilities[mask].mean(axis=0))
    return np.asarray(trial_truth, dtype=np.int64), np.stack(trial_probabilities)


def train_plain_shallowconvnet(
    adapter: TorchModelAdapter,
    inputs: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    label_smoothing: float,
    progress_callback: Any,
) -> dict[str, float]:
    device = adapter._device
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(inputs[train_indices], dtype=torch.float32),
            torch.as_tensor(labels[train_indices], dtype=torch.long),
        ),
        batch_size=max(int(batch_size), 1),
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(
        adapter.model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(epochs) * len(loader), 1),
        eta_min=1e-6,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] | None = None
    best_score: tuple[float, ...] | None = None

    for epoch_index in range(max(int(epochs), 1)):
        adapter.model.train()
        losses: list[float] = []
        for batch_inputs, batch_labels in loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            loss = criterion(adapter.model(batch_inputs), batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.item()))

        probabilities = adapter.predict_proba(inputs[validation_indices])
        truth = labels[validation_indices]
        window = classification_metrics(truth, probabilities, n_classes=3)
        trial_truth, trial_probabilities = aggregate_trial_probabilities(
            truth,
            probabilities,
            groups[validation_indices],
        )
        trial = classification_metrics(trial_truth, trial_probabilities, n_classes=3)
        window_recalls = np.asarray(window["per_class_recall"], dtype=np.float64)
        trial_recalls = np.asarray(trial["per_class_recall"], dtype=np.float64)
        window_worst = float(np.min(window_recalls))
        trial_worst = float(np.min(trial_recalls))
        epoch_metrics = {
            "train_loss": float(np.mean(losses)),
            "val_loss": float(window["cross_entropy"]),
            "val_acc": float(window["accuracy"]),
            "val_balanced_accuracy": float(window["balanced_accuracy"]),
            "val_kappa": float(window["kappa"]),
            "val_macro_f1": float(window["macro_f1"]),
            "val_worst_class_accuracy": window_worst,
            "val_trial_acc": float(trial["accuracy"]),
            "val_trial_balanced_accuracy": float(trial["balanced_accuracy"]),
            "val_trial_kappa": float(trial["kappa"]),
            "val_trial_macro_f1": float(trial["macro_f1"]),
            "val_trial_worst_class_accuracy": trial_worst,
            "epochs_completed": float(epoch_index + 1),
        }
        progress_callback(epoch_index + 1, epochs, epoch_metrics)
        noncollapsed = window_worst > 0.0 and trial_worst > 0.0
        score = (
            trial_worst,
            window_worst,
            float(trial["balanced_accuracy"]),
            float(trial["kappa"]),
            float(trial["macro_f1"]),
            float(window["balanced_accuracy"]),
            -float(window["cross_entropy"]),
        )
        if noncollapsed and (best_score is None or score > best_score):
            best_score = score
            best_metrics = dict(epoch_metrics)
            best_metrics["best_epoch"] = float(epoch_index + 1)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in adapter.model.state_dict().items()
            }

    if best_state is None or best_metrics is None:
        raise RuntimeError(
            "Plain ShallowConvNet produced no validation checkpoint with all classes recalled."
        )
    adapter.model.load_state_dict(best_state)
    best_metrics["epochs_completed"] = float(max(int(epochs), 1))
    return best_metrics


def run(args: argparse.Namespace) -> dict[str, Any]:
    calibration_path = args.calibration_data.resolve()
    recording_dir = args.realtime_recording.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(calibration_path, allow_pickle=False) as payload:
        required = {"processed_windows", "labels", "trial_ids", "sfreq"}
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"Calibration data is missing: {', '.join(missing)}")
        calibration_x = payload["processed_windows"].astype(np.float32)
        calibration_y = payload["labels"].astype(np.int64)
        calibration_groups = payload["trial_ids"].astype(np.int64)
        sfreq = float(payload["sfreq"][0])

    train_indices, validation_indices = split_train_validation_indices(
        calibration_y,
        groups=calibration_groups,
        random_state=args.split_seed,
    )
    seed_everything(args.training_seed)
    adapter = ModelFactory.get(
        "shallowconvnet",
        n_chans=calibration_x.shape[1],
        n_times=calibration_x.shape[2],
        sfreq=sfreq,
        n_classes=args.n_classes,
    )
    if not isinstance(adapter, TorchModelAdapter):
        raise TypeError("Expected the PyTorch ShallowConvNet adapter.")

    history: list[dict[str, Any]] = []

    def record_progress(
        epoch: int,
        total_epochs: int,
        metrics: dict[str, float],
    ) -> None:
        history.append(
            {
                "epoch": int(epoch),
                "total_epochs": int(total_epochs),
                **{key: float(value) for key, value in metrics.items()},
            }
        )
        print(
            f"epoch={epoch}/{total_epochs} "
            f"train_loss={metrics['train_loss']:.6f} "
            f"val_loss={metrics['val_loss']:.6f} "
            f"val_acc={metrics['val_acc']:.6f}",
            flush=True,
        )

    training_started = time.perf_counter()
    calibration_metrics = train_plain_shallowconvnet(
        adapter,
        calibration_x,
        calibration_y,
        calibration_groups,
        train_indices,
        validation_indices,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        progress_callback=record_progress,
    )
    training_duration = time.perf_counter() - training_started

    for parameter in adapter.model.parameters():
        parameter.requires_grad_(False)
    adapter.model.eval()
    trainable_parameters = sum(
        int(parameter.numel())
        for parameter in adapter.model.parameters()
        if parameter.requires_grad
    )
    checkpoint_path = output_dir / "shallowconvnet_frozen.pt"
    adapter.save(checkpoint_path)
    frozen_state_before = state_sha256(adapter.model)

    raw_windows, realtime_y, scenes, event_ids, timestamps, chunk_sources = (
        load_committed_realtime(recording_dir)
    )
    processed_windows = np.stack(
        [preprocess_eeg_window(window, sfreq=sfreq).data for window in raw_windows]
    ).astype(np.float32)

    seed_everything(args.inference_seed)
    probabilities = np.stack(
        [
            adapter.predict_proba(
                window[None],
                mc_dropout_passes=args.mc_dropout_passes,
            )[0]
            for window in processed_windows
        ]
    ).astype(np.float32)
    frozen_state_after = state_sha256(adapter.model)
    if frozen_state_after != frozen_state_before:
        raise RuntimeError("Frozen ShallowConvNet weights changed during evaluation.")

    total = len(realtime_y)
    scopes: dict[str, np.ndarray] = {
        "all": np.arange(total),
        "first_384": np.arange(min(total, 384)),
    }
    if total > 64:
        scopes["after_initial_64_through_384"] = np.arange(64, min(total, 384))
    metrics = {
        name: classification_metrics(
            realtime_y[indices],
            probabilities[indices],
            n_classes=args.n_classes,
        )
        for name, indices in scopes.items()
        if indices.size > 0
    }
    segments = []
    for start in range(0, total, 64):
        stop = min(start + 64, total)
        segments.append(
            {
                "start_index": start,
                "stop_index_exclusive": stop,
                **classification_metrics(
                    realtime_y[start:stop],
                    probabilities[start:stop],
                    n_classes=args.n_classes,
                ),
            }
        )

    probabilities_path = output_dir / "frozen_predictions.npz"
    np.savez_compressed(
        probabilities_path,
        probabilities=probabilities,
        predictions=probabilities.argmax(axis=1).astype(np.int64),
        labels=realtime_y,
        scene_indices=scenes,
        event_ids=event_ids,
        window_end_monotonic=timestamps,
    )
    report = {
        "experiment": "frozen_plain_shallowconvnet",
        "model": {
            "name": "shallowconvnet",
            "adapter": type(adapter).__name__,
            "trainable_parameters_after_freeze": trainable_parameters,
            "state_sha256_before_evaluation": frozen_state_before,
            "state_sha256_after_evaluation": frozen_state_after,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
        },
        "configuration": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "fixed_epoch_training": True,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "training_seed": args.training_seed,
            "split_seed": args.split_seed,
            "inference_seed": args.inference_seed,
            "mc_dropout_passes": args.mc_dropout_passes,
            "sfreq": sfreq,
            "n_classes": args.n_classes,
        },
        "calibration": {
            "path": str(calibration_path),
            "sha256": sha256_file(calibration_path),
            "windows": int(len(calibration_y)),
            "trials": int(np.unique(calibration_groups).size),
            "class_counts": np.bincount(
                calibration_y, minlength=args.n_classes
            ).astype(int).tolist(),
            "train_windows": int(train_indices.size),
            "validation_windows": int(validation_indices.size),
            "train_trials": int(np.unique(calibration_groups[train_indices]).size),
            "validation_trials": int(
                np.unique(calibration_groups[validation_indices]).size
            ),
            "best_validation": {
                key: float(value) for key, value in calibration_metrics.items()
            },
            "epochs_ran": len(history),
            "training_duration_sec": float(training_duration),
            "history": history,
        },
        "realtime": {
            "recording_dir": str(recording_dir),
            "windows": int(total),
            "class_counts": np.bincount(
                realtime_y, minlength=args.n_classes
            ).astype(int).tolist(),
            "chunk_sources": chunk_sources,
            "metrics": metrics,
            "chronological_64_window_segments": segments,
        },
        "artifacts": {
            "predictions": str(probabilities_path),
            "predictions_sha256": sha256_file(probabilities_path),
        },
    }
    report_path = output_dir / "report.json"
    save_json(report_path, report)
    print(json.dumps(report["realtime"]["metrics"], ensure_ascii=False, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-data", type=Path, required=True)
    parser.add_argument("--realtime-recording", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--inference-seed", type=int, default=2026)
    parser.add_argument("--mc-dropout-passes", type=int, default=8)
    parser.add_argument("--n-classes", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
