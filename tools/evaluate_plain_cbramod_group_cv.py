"""Evaluate plain supervised CBraMod with repeated trial-grouped CV."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

# Required by deterministic CUDA matrix multiplication before CUDA initializes.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.factory import ModelFactory, TorchModelAdapter  # noqa: E402
from tools.evaluate_frozen_shallowconvnet import (  # noqa: E402
    aggregate_trial_probabilities,
    classification_metrics,
)
from tools.search_cbramod_offline_group_cv import (  # noqa: E402
    _grouped_splits,
    _load_dataset,
    _seed_everything,
    _sha256,
    _summarize_runs,
    _write_json,
)


def _enable_deterministic_training() -> None:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def _fit_plain_cbramod(
    adapter: TorchModelAdapter,
    windows: np.ndarray,
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
    n_classes: int,
) -> tuple[dict[str, float | None], bool]:
    for parameter in adapter.model.parameters():
        parameter.requires_grad_(True)
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(windows[train_indices], dtype=torch.float32),
            torch.as_tensor(labels[train_indices], dtype=torch.long),
        ),
        batch_size=max(int(batch_size), 1),
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(
        adapter.model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(epochs) * len(loader), 1),
        eta_min=1e-6,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing)).to(
        adapter._device
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float | None] | None = None
    best_score: tuple[float, ...] | None = None
    latest_metrics: dict[str, float | None] = {}
    epochs_completed = 0

    for epoch_index in range(max(int(epochs), 1)):
        epochs_completed = epoch_index + 1
        adapter.model.train()
        train_losses: list[float] = []
        for batch_inputs, batch_labels in loader:
            batch_inputs = batch_inputs.to(adapter._device)
            batch_labels = batch_labels.to(adapter._device)
            optimizer.zero_grad()
            loss = criterion(adapter.model(batch_inputs), batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_losses.append(float(loss.item()))

        probabilities = adapter.predict_proba(windows[validation_indices])
        truth = labels[validation_indices]
        window = classification_metrics(truth, probabilities, n_classes=n_classes)
        trial_truth, trial_probabilities = aggregate_trial_probabilities(
            truth,
            probabilities,
            groups[validation_indices],
        )
        trial = classification_metrics(
            trial_truth,
            trial_probabilities,
            n_classes=n_classes,
        )
        window_worst = float(np.min(window["per_class_recall"]))
        trial_worst = float(np.min(trial["per_class_recall"]))
        latest_metrics = {
            "train_loss": float(np.mean(train_losses)),
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
            "epochs_completed": float(epochs_completed),
            "best_epoch": None,
        }
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
            best_metrics = dict(latest_metrics)
            best_metrics["best_epoch"] = float(epoch_index + 1)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in adapter.model.state_dict().items()
            }

    if best_state is None or best_metrics is None:
        return latest_metrics, True
    adapter.model.load_state_dict(best_state)
    best_metrics["epochs_completed"] = float(epochs_completed)
    return best_metrics, False


def run(args: argparse.Namespace) -> dict[str, Any]:
    _enable_deterministic_training()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    identity = {
        "schema_version": 1,
        "mechanics_version": 3,
        "experiment": "plain_official_cbramod_grouped_calibration_cv",
        "dataset": str(dataset),
        "dataset_sha256": _sha256(dataset),
        "feature_key": args.feature_key,
        "folds": args.folds,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "fixed_epoch_training": True,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "deterministic_algorithms": True,
        "allow_tf32": False,
        "flash_sdp": False,
        "mem_efficient_sdp": False,
        "removed_components": [
            "context_representation_modulator",
            "time_masked_view",
            "frequency_masked_view",
            "representation_consistency_loss",
        ],
    }
    report_path = output / "report.json"
    if report_path.is_file():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("identity") != identity:
            raise ValueError(f"Existing report has incompatible inputs: {report_path}")
        print(f"REUSED {report_path}", flush=True)
        return existing

    windows, labels, groups, sfreq = _load_dataset(dataset, args.feature_key)
    n_classes = int(labels.max()) + 1
    runs: list[dict[str, Any]] = []
    started_at = time.perf_counter()
    for split_seed in args.seeds:
        splits = _grouped_splits(
            windows,
            labels,
            groups,
            folds=args.folds,
            seed=split_seed,
        )
        for fold_index, (train_indices, validation_indices) in enumerate(splits):
            run_seed = int(split_seed * 100 + fold_index)
            _seed_everything(run_seed)
            adapter = ModelFactory.get(
                "cbramod",
                n_chans=windows.shape[1],
                n_times=windows.shape[2],
                n_classes=n_classes,
                sfreq=sfreq,
            )
            if not isinstance(adapter, TorchModelAdapter):
                raise TypeError("Plain CBraMod CV requires a TorchModelAdapter.")
            run_started = time.perf_counter()
            metrics, class_collapse = _fit_plain_cbramod(
                adapter,
                windows,
                labels,
                groups,
                train_indices,
                validation_indices,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                label_smoothing=args.label_smoothing,
                n_classes=n_classes,
            )
            record = {
                "split_seed": split_seed,
                "fold": fold_index,
                "run_seed": run_seed,
                "train_trials": int(np.unique(groups[train_indices]).size),
                "validation_trials": int(np.unique(groups[validation_indices]).size),
                "metrics": metrics,
                "class_collapse": class_collapse,
                "duration_sec": time.perf_counter() - run_started,
            }
            runs.append(record)
            print(
                f"seed={split_seed} fold={fold_index + 1}/{args.folds} "
                f"trial_bacc={float(metrics['val_trial_balanced_accuracy']):.4f} "
                f"trial_worst={float(metrics['val_trial_worst_class_accuracy']):.4f} "
                f"collapse={class_collapse}",
                flush=True,
            )
            del adapter
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    report = {
        "identity": identity,
        "dataset": {
            "windows": int(windows.shape[0]),
            "trials": int(np.unique(groups).size),
            "shape": list(windows.shape),
            "sfreq": sfreq,
        },
        "summary": _summarize_runs(runs),
        "runs": runs,
        "duration_sec": time.perf_counter() - started_at,
    }
    _write_json(report_path, report)
    return report


def _comma_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-key", default="processed_windows")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=_comma_ints, default=[17, 42, 2026])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    report = run(parse_args())
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
