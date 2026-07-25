"""Train fixed-epoch same-day checkpoints from the first chronological EEG trials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
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


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    with np.load(args.data, allow_pickle=False) as payload:
        all_windows = payload["processed_windows"].astype(np.float32)
        all_labels = payload["labels"].astype(np.int64)
        all_trial_ids = payload["trial_ids"].astype(np.int64)
        sfreq = float(payload["sfreq"][0])
    trial_order = np.asarray(
        list(dict.fromkeys(all_trial_ids.tolist())),
        dtype=np.int64,
    )
    if not 0 < args.pretrain_trials < len(trial_order):
        raise ValueError(
            f"--pretrain-trials must be within 1..{len(trial_order) - 1}."
        )
    train_mask = np.isin(
        all_trial_ids,
        trial_order[: args.pretrain_trials],
    )
    raw_train = all_windows[train_mask]
    train_y = all_labels[train_mask]
    channel_mean = raw_train.mean(axis=(0, 2), keepdims=True)
    channel_std = np.maximum(
        raw_train.std(axis=(0, 2), keepdims=True),
        1e-6,
    )
    train_x = ((raw_train - channel_mean) / channel_std).astype(np.float32)

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output / "scaler.npz",
        channel_mean=channel_mean.astype(np.float32),
        channel_std=channel_std.astype(np.float32),
    )
    config = {
        "data": str(args.data.resolve()),
        "pretrain_trials": args.pretrain_trials,
        "pretrain_windows": int(len(train_y)),
        "trial_ids": trial_order[: args.pretrain_trials].tolist(),
        "label_counts": np.bincount(train_y, minlength=3).tolist(),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "method": args.method,
        "mask_ratio": args.mask_ratio if args.method == "neuroonline" else None,
        "consistency_weight": (
            args.consistency_weight if args.method == "neuroonline" else None
        ),
        "seeds": args.seeds,
    }
    save_json(args.output / "pretrain_config.json", config)

    histories: dict[str, list[dict[str, float]]] = {}
    for seed in args.seeds:
        seed_everything(seed)
        adapter = ModelFactory.get(
            "shallowconvnet",
            n_chans=train_x.shape[1],
            n_times=train_x.shape[2],
            sfreq=sfreq,
            n_classes=3,
        )
        if not isinstance(adapter, TorchModelAdapter):
            raise TypeError("Expected a PyTorch ShallowConvNet adapter.")
        generator = torch.Generator().manual_seed(seed)
        neuro: NeuroOnlineModelAdapter | None = None
        if args.method == "neuroonline":
            neuro = NeuroOnlineModelAdapter(
                adapter,
                config=NeuroOnlineConfig(
                    enabled=True,
                    offline_epochs=args.epochs,
                    offline_batch_size=args.batch_size,
                    offline_learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    mask_ratio=args.mask_ratio,
                    consistency_weight=args.consistency_weight,
                    label_smoothing=args.label_smoothing,
                    random_seed=seed,
                ),
            )
            train_tensor = torch.as_tensor(train_x, dtype=torch.float32)
            time_views = _time_mask(train_tensor, args.mask_ratio, generator)
            frequency_views = _frequency_mask(
                train_tensor,
                args.mask_ratio,
                generator,
            )
            modulator = neuro._prepare_training(train_tensor[:1])
            loader = neuro._view_loader(
                train_tensor,
                time_views,
                frequency_views,
                train_y,
                batch_size=args.batch_size,
            )
            optimizer = torch.optim.AdamW(
                list(adapter.model.parameters()) + list(modulator.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
        else:
            loader = DataLoader(
                TensorDataset(
                    torch.as_tensor(train_x, dtype=torch.float32),
                    torch.as_tensor(train_y, dtype=torch.long),
                ),
                batch_size=args.batch_size,
                shuffle=True,
                generator=generator,
            )
            optimizer = torch.optim.AdamW(
                adapter.model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(args.epochs * len(loader), 1),
            eta_min=1e-6,
        )
        criterion = nn.CrossEntropyLoss(
            label_smoothing=args.label_smoothing,
        ).to(adapter._device)
        history: list[dict[str, float]] = []
        for epoch in range(args.epochs):
            if neuro is not None:
                train_metrics = neuro._train_epoch(
                    loader,
                    optimizer,
                    criterion,
                    scheduler=scheduler,
                    clip_classifier_gradients=True,
                )
                history.append(
                    {
                        "epoch": float(epoch + 1),
                        **train_metrics,
                    }
                )
                continue
            adapter.model.train()
            losses: list[float] = []
            correct = 0
            examples = 0
            for inputs, labels in loader:
                inputs = inputs.to(adapter._device)
                labels = labels.to(adapter._device)
                optimizer.zero_grad()
                logits = adapter.model(inputs)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                losses.append(float(loss.item()))
                correct += int((logits.argmax(dim=1) == labels).sum().item())
                examples += int(len(labels))
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "loss": float(np.mean(losses)),
                    "accuracy": correct / max(examples, 1),
                }
            )
        seed_dir = args.output / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        if neuro is not None:
            neuro.save(seed_dir / "shallowconvnet.pt")
        else:
            adapter.save(seed_dir / "shallowconvnet.pt")
        save_json(seed_dir / "history.json", history)
        histories[str(seed)] = history
    summary = {"config": config, "final_train_epoch": {
        seed: history[-1] for seed, history in histories.items()
    }}
    save_json(args.output / "pretrain_summary.json", summary)
    return summary


def comma_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pretrain-trials", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=comma_ints, default=[17, 42, 2026])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument(
        "--method",
        choices=["baseline", "neuroonline"],
        default="baseline",
    )
    parser.add_argument("--mask-ratio", type=float, default=0.3)
    parser.add_argument("--consistency-weight", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
