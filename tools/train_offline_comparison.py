"""Train matched baseline and NeuroOnline-offline models with block-safe splits."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
from typing import Any, Callable

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from adaptation.neuroonline import (
    NeuroOnlineConfig,
    NeuroOnlineModelAdapter,
    _frequency_mask,
    _time_mask,
)
from models.factory import ModelFactory, TorchModelAdapter


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def classification_metrics(
    truth: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(truth, predictions)),
        "confusion_matrix": confusion_matrix(truth, predictions, labels=[0, 1, 2]).tolist(),
    }


def trial_metrics(
    truth: np.ndarray,
    probabilities: np.ndarray,
    trial_ids: np.ndarray,
) -> dict[str, Any]:
    trial_truth: list[int] = []
    trial_probabilities: list[np.ndarray] = []
    for trial_id in dict.fromkeys(trial_ids.tolist()):
        mask = trial_ids == trial_id
        labels = truth[mask]
        if not np.all(labels == labels[0]):
            raise ValueError(f"Trial {trial_id} contains multiple labels.")
        trial_truth.append(int(labels[0]))
        trial_probabilities.append(probabilities[mask].mean(axis=0))
    metrics = classification_metrics(
        np.asarray(trial_truth, dtype=np.int64),
        np.stack(trial_probabilities),
    )
    metrics["trials"] = len(trial_truth)
    return metrics


def predict_torch(
    model: nn.Module,
    inputs: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    loader = DataLoader(
        TensorDataset(torch.as_tensor(inputs, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )
    with torch.no_grad():
        for (batch,) in loader:
            logits = model(batch.to(device))
            outputs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def evaluate_splits(
    predict: Callable[[np.ndarray], np.ndarray],
    splits: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, (inputs, labels, trial_ids) in splits.items():
        probabilities = predict(inputs)
        output[name] = {
            "windows": int(len(labels)),
            "window": classification_metrics(labels, probabilities),
            "trial": trial_metrics(labels, probabilities, trial_ids),
        }
    return output


def _is_better(metrics: dict[str, Any], best: dict[str, Any] | None) -> bool:
    if best is None:
        return True
    current_kappa = float(metrics["kappa"])
    best_kappa = float(best["kappa"])
    if current_kappa != best_kappa:
        return current_kappa > best_kappa
    return float(metrics["macro_f1"]) > float(best["macro_f1"])


def train_baseline(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    label_smoothing: float,
    sfreq: float,
) -> tuple[TorchModelAdapter, list[dict[str, Any]]]:
    adapter = ModelFactory.get(
        "shallowconvnet",
        n_chans=train_x.shape[1],
        n_times=train_x.shape[2],
        sfreq=sfreq,
        n_classes=3,
    )
    if not isinstance(adapter, TorchModelAdapter):
        raise TypeError("Expected a PyTorch ShallowConvNet adapter.")
    device = adapter._device
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(train_x, dtype=torch.float32),
            torch.as_tensor(train_y, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        adapter.model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs * len(loader), 1),
        eta_min=1e-6,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)
    best_state: dict[str, torch.Tensor] | None = None
    best_validation: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    val_x, val_y, val_trials = validation

    for epoch in range(epochs):
        adapter.model.train()
        losses: list[float] = []
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            loss = criterion(adapter.model(inputs), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.item()))

        probabilities = predict_torch(
            adapter.model,
            val_x,
            device=device,
            batch_size=batch_size,
        )
        validation_metrics = trial_metrics(val_y, probabilities, val_trials)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "validation_trial": validation_metrics,
            }
        )
        if _is_better(validation_metrics, best_validation):
            best_validation = validation_metrics
            best_state = copy.deepcopy(adapter.model.state_dict())

    if best_state is None:
        raise RuntimeError("Baseline training did not produce a checkpoint.")
    adapter.model.load_state_dict(best_state)
    return adapter, history


def train_neuroonline_offline(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    label_smoothing: float,
    sfreq: float,
    mask_ratio: float,
    consistency_weight: float,
) -> tuple[NeuroOnlineModelAdapter, list[dict[str, Any]]]:
    base = ModelFactory.get(
        "shallowconvnet",
        n_chans=train_x.shape[1],
        n_times=train_x.shape[2],
        sfreq=sfreq,
        n_classes=3,
    )
    if not isinstance(base, TorchModelAdapter):
        raise TypeError("Expected a PyTorch ShallowConvNet adapter.")
    config = NeuroOnlineConfig(
        enabled=True,
        offline_epochs=epochs,
        offline_batch_size=batch_size,
        offline_learning_rate=learning_rate,
        weight_decay=weight_decay,
        mask_ratio=mask_ratio,
        consistency_weight=consistency_weight,
        label_smoothing=label_smoothing,
        random_seed=seed,
    )
    adapter = NeuroOnlineModelAdapter(base, config=config)
    device = adapter._device
    train_tensor = torch.as_tensor(train_x, dtype=torch.float32)
    mask_generator = torch.Generator().manual_seed(seed)
    time_views = _time_mask(train_tensor, mask_ratio, mask_generator)
    frequency_views = _frequency_mask(train_tensor, mask_ratio, mask_generator)
    modulator = adapter._prepare_training(train_tensor[:1])
    loader = adapter._view_loader(
        train_tensor,
        time_views,
        frequency_views,
        train_y,
        batch_size=batch_size,
    )
    optimizer = torch.optim.AdamW(
        list(base.model.parameters()) + list(modulator.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs * len(loader), 1),
        eta_min=1e-6,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)
    best_base: dict[str, torch.Tensor] | None = None
    best_modulator: dict[str, torch.Tensor] | None = None
    best_validation: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    val_x, val_y, val_trials = validation

    for epoch in range(epochs):
        train_metrics = adapter._train_epoch(
            loader,
            optimizer,
            criterion,
            scheduler=scheduler,
            clip_classifier_gradients=True,
        )
        probabilities = adapter.predict_proba(val_x)
        validation_metrics = trial_metrics(val_y, probabilities, val_trials)
        history.append(
            {
                "epoch": epoch + 1,
                **train_metrics,
                "validation_trial": validation_metrics,
            }
        )
        if _is_better(validation_metrics, best_validation):
            best_validation = validation_metrics
            best_base = copy.deepcopy(base.model.state_dict())
            best_modulator = copy.deepcopy(modulator.state_dict())

    if best_base is None or best_modulator is None:
        raise RuntimeError("NeuroOnline offline training did not produce a checkpoint.")
    base.model.load_state_dict(best_base)
    modulator.load_state_dict(best_modulator)
    return adapter, history


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    with np.load(args.day1, allow_pickle=False) as payload:
        day1_x = payload["processed_windows"].astype(np.float32)
        day1_y = payload["labels"].astype(np.int64)
        day1_trials = payload["trial_ids"].astype(np.int64)
        day1_blocks = payload["block_indices"].astype(np.int64)
        day1_quality = payload["quality_clip_fraction"].astype(np.float32)
        sfreq = float(payload["sfreq"][0])
    day2_x: np.ndarray | None = None
    day2_y: np.ndarray | None = None
    day2_trials: np.ndarray | None = None
    day2_quality: np.ndarray | None = None
    if args.evaluation_scope == "all":
        if args.day2 is None:
            raise ValueError("--day2 is required when --evaluation-scope=all.")
        with np.load(args.day2, allow_pickle=False) as payload:
            day2_x = payload["processed_windows"].astype(np.float32)
            day2_y = payload["labels"].astype(np.int64)
            day2_trials = payload["trial_ids"].astype(np.int64)
            day2_quality = payload["quality_clip_fraction"].astype(np.float32)

    train_mask = day1_blocks <= 3
    val_mask = day1_blocks == 4
    test_mask = day1_blocks == 5
    train_mean = day1_x[train_mask].mean(axis=(0, 2), keepdims=True)
    train_std = day1_x[train_mask].std(axis=(0, 2), keepdims=True)
    train_std = np.maximum(train_std, 1e-6)

    def scale(values: np.ndarray) -> np.ndarray:
        return ((values - train_mean) / train_std).astype(np.float32)

    train_x = scale(day1_x[train_mask])
    train_y = day1_y[train_mask]
    validation = (
        scale(day1_x[val_mask]),
        day1_y[val_mask],
        day1_trials[val_mask],
    )
    evaluation_splits = {"d1_validation_block4": validation}
    if args.evaluation_scope == "all":
        assert day2_x is not None
        assert day2_y is not None
        assert day2_trials is not None
        evaluation_splits.update(
            {
                "d1_test_block5": (
                    scale(day1_x[test_mask]),
                    day1_y[test_mask],
                    day1_trials[test_mask],
                ),
                "d2_static_cross_day": (
                    scale(day2_x),
                    day2_y,
                    day2_trials,
                ),
            }
        )
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output / "scaler.npz",
        channel_mean=train_mean.astype(np.float32),
        channel_std=train_std.astype(np.float32),
    )
    run_config = {
        "day1": str(args.day1.resolve()),
        "day2": str(args.day2.resolve()) if args.day2 is not None else None,
        "evaluation_scope": args.evaluation_scope,
        "train_blocks": [0, 1, 2, 3],
        "validation_blocks": [4],
        "test_blocks": [5],
        "train_windows": int(train_mask.sum()),
        "validation_windows": int(val_mask.sum()),
        "test_windows": int(test_mask.sum()),
        "day2_windows": int(len(day2_y)) if day2_y is not None else None,
        "channels": int(day1_x.shape[1]),
        "samples": int(day1_x.shape[2]),
        "sfreq": sfreq,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "fixed_epoch_training": True,
        "mask_ratio": args.mask_ratio,
        "consistency_weight": args.consistency_weight,
        "seeds": args.seeds,
        "quality_clip_fraction": {
            "d1_mean": float(day1_quality.mean()),
            "d2_mean": float(day2_quality.mean()) if day2_quality is not None else None,
        },
    }
    _save_json(args.output / "run_config.json", run_config)

    summary: dict[str, Any] = {"config": run_config, "runs": {}}
    for method in args.methods:
        summary["runs"][method] = {}
        for seed in args.seeds:
            seed_everything(seed)
            output_dir = args.output / method / f"seed_{seed}"
            if method == "baseline":
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
                    sfreq=sfreq,
                )
                predict = lambda values, trained=model: predict_torch(
                    trained.model,
                    values,
                    device=trained._device,
                    batch_size=args.batch_size,
                )
                output_dir.mkdir(parents=True, exist_ok=True)
                model.save(output_dir / "shallowconvnet.pt")
            elif method == "neuroonline":
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
                    sfreq=sfreq,
                    mask_ratio=args.mask_ratio,
                    consistency_weight=args.consistency_weight,
                )
                predict = model.predict_proba
                output_dir.mkdir(parents=True, exist_ok=True)
                model.save(output_dir / "shallowconvnet.pt")
            else:
                raise ValueError(f"Unknown method: {method}")

            metrics = evaluate_splits(predict, evaluation_splits)
            _save_json(output_dir / "history.json", history)
            _save_json(output_dir / "metrics.json", metrics)
            summary["runs"][method][str(seed)] = metrics
            _save_json(args.output / "summary.json", summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day1", type=Path, required=True)
    parser.add_argument("--day2", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--evaluation-scope",
        choices=["selection", "all"],
        default="all",
        help="Selection computes validation only; all additionally evaluates held-out D1 and D2.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["baseline", "neuroonline"],
        default=["baseline", "neuroonline"],
    )
    parser.add_argument(
        "--seeds",
        type=lambda value: [int(item) for item in value.split(",")],
        default=[17, 42, 2026],
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mask-ratio", type=float, default=0.3)
    parser.add_argument("--consistency-weight", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
