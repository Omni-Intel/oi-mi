"""Tune and compare causal online adaptation on chronologically ordered offline EEG."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Literal

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

from adaptation.neuroonline import (
    NeuroOnlineConfig,
    NeuroOnlineModelAdapter,
    _frequency_mask,
    _time_mask,
)
from models.factory import ModelFactory, TorchModelAdapter


Method = Literal[
    "static",
    "standard_online",
    "neurooffline_static",
    "neuroonline",
]


@dataclass(slots=True)
class OnlineData:
    windows: np.ndarray
    labels: np.ndarray
    trial_ids: np.ndarray
    trial_order: np.ndarray
    trial_ordinal: np.ndarray
    sfreq: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
        "confusion_matrix": confusion_matrix(
            truth,
            predictions,
            labels=[0, 1, 2],
        ).tolist(),
    }


def load_online_data(
    day1_path: Path,
    day2_path: Path,
    *,
    online_blocks: list[int] | None = None,
    scaler_trials: int | None = None,
    online_start_trial: int = 0,
) -> OnlineData:
    with np.load(day1_path, allow_pickle=False) as day1:
        day1_windows = day1["processed_windows"].astype(np.float32)
        day1_blocks = day1["block_indices"].astype(np.int64)
        day1_trial_ids = day1["trial_ids"].astype(np.int64)
        sfreq = float(day1["sfreq"][0])
    if scaler_trials is None:
        scaler_mask = day1_blocks <= 3
    else:
        scaler_trial_order = np.asarray(
            list(dict.fromkeys(day1_trial_ids.tolist())),
            dtype=np.int64,
        )
        if not 0 < scaler_trials <= len(scaler_trial_order):
            raise ValueError(
                f"--scaler-trials must be within 1..{len(scaler_trial_order)}."
            )
        scaler_mask = np.isin(
            day1_trial_ids,
            scaler_trial_order[:scaler_trials],
        )
    train_windows = day1_windows[scaler_mask]
    channel_mean = train_windows.mean(axis=(0, 2), keepdims=True)
    channel_std = np.maximum(
        train_windows.std(axis=(0, 2), keepdims=True),
        1e-6,
    )

    with np.load(day2_path, allow_pickle=False) as day2:
        windows = day2["processed_windows"].astype(np.float32)
        labels = day2["labels"].astype(np.int64)
        trial_ids = day2["trial_ids"].astype(np.int64)
        day2_sfreq = float(day2["sfreq"][0])
        block_indices = (
            day2["block_indices"].astype(np.int64)
            if "block_indices" in day2.files
            else None
        )
    if online_blocks is not None:
        if block_indices is None:
            raise ValueError("--online-blocks requires block_indices in the online dataset.")
        online_mask = np.isin(block_indices, np.asarray(online_blocks, dtype=np.int64))
        windows = windows[online_mask]
        labels = labels[online_mask]
        trial_ids = trial_ids[online_mask]
    if online_start_trial:
        full_trial_order = np.asarray(
            list(dict.fromkeys(trial_ids.tolist())),
            dtype=np.int64,
        )
        if not 0 <= online_start_trial < len(full_trial_order):
            raise ValueError(
                f"--online-start-trial must be within 0..{len(full_trial_order) - 1}."
            )
        online_mask = np.isin(
            trial_ids,
            full_trial_order[online_start_trial:],
        )
        windows = windows[online_mask]
        labels = labels[online_mask]
        trial_ids = trial_ids[online_mask]
    if not math.isclose(day2_sfreq, sfreq):
        raise ValueError(f"D1 sfreq={sfreq} differs from D2 sfreq={day2_sfreq}.")
    if windows.shape[1:] != day1_windows.shape[1:]:
        raise ValueError(
            f"D1 shape {day1_windows.shape[1:]} differs from D2 shape {windows.shape[1:]}."
        )

    trial_order = np.asarray(list(dict.fromkeys(trial_ids.tolist())), dtype=np.int64)
    ordinal_lookup = {int(trial_id): index for index, trial_id in enumerate(trial_order)}
    trial_ordinal = np.asarray(
        [ordinal_lookup[int(trial_id)] for trial_id in trial_ids],
        dtype=np.int64,
    )
    for trial_id in trial_order:
        trial_labels = labels[trial_ids == trial_id]
        if not np.all(trial_labels == trial_labels[0]):
            raise ValueError(f"Trial {trial_id} contains multiple labels.")

    return OnlineData(
        windows=((windows - channel_mean) / channel_std).astype(np.float32),
        labels=labels,
        trial_ids=trial_ids,
        trial_order=trial_order,
        trial_ordinal=trial_ordinal,
        sfreq=sfreq,
    )


def make_base_model(data: OnlineData, checkpoint: Path) -> TorchModelAdapter:
    adapter = ModelFactory.get(
        "shallowconvnet",
        n_chans=data.windows.shape[1],
        n_times=data.windows.shape[2],
        sfreq=data.sfreq,
        n_classes=3,
    )
    if not isinstance(adapter, TorchModelAdapter):
        raise TypeError("Expected a PyTorch ShallowConvNet adapter.")
    adapter.load(checkpoint)
    return adapter


class StandardOnlineTrainer:
    """Persistent full-model AdamW updates on newly labeled windows."""

    def __init__(
        self,
        adapter: TorchModelAdapter,
        *,
        learning_rate: float,
        weight_decay: float,
        label_smoothing: float,
    ) -> None:
        self.adapter = adapter
        for parameter in adapter.model.parameters():
            parameter.requires_grad = True
        self.optimizer = torch.optim.AdamW(
            adapter.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
        ).to(adapter._device)

    def update(self, windows: np.ndarray, labels: np.ndarray, *, epochs: int) -> dict[str, float]:
        inputs = torch.as_tensor(
            windows,
            dtype=torch.float32,
            device=self.adapter._device,
        )
        targets = torch.as_tensor(
            labels,
            dtype=torch.long,
            device=self.adapter._device,
        )
        last_loss = 0.0
        self.adapter.model.train()
        for _ in range(max(int(epochs), 1)):
            self.optimizer.zero_grad()
            loss = self.criterion(self.adapter.model(inputs), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.adapter.model.parameters(), 1.0)
            self.optimizer.step()
            last_loss = float(loss.item())
        return {"updated": float(len(labels)), "loss": last_loss}


def metric_bundle(
    window_truth: np.ndarray,
    window_probabilities: np.ndarray,
    window_trial_ordinal: np.ndarray,
) -> dict[str, Any]:
    window = classification_metrics(window_truth, window_probabilities)
    trial_truth: list[int] = []
    trial_probabilities: list[np.ndarray] = []
    for ordinal in dict.fromkeys(window_trial_ordinal.tolist()):
        mask = window_trial_ordinal == ordinal
        labels = window_truth[mask]
        trial_truth.append(int(labels[0]))
        trial_probabilities.append(window_probabilities[mask].mean(axis=0))
    trial = classification_metrics(
        np.asarray(trial_truth, dtype=np.int64),
        np.stack(trial_probabilities),
    )
    trial["trials"] = len(trial_truth)
    return {
        "windows": int(len(window_truth)),
        "window": window,
        "trial": trial,
    }


def simulate(
    *,
    data: OnlineData,
    checkpoint: Path,
    method: Method,
    seed: int,
    update_windows: int,
    mask_ratio: float,
    consistency_weight: float,
    learning_rate: float,
    weight_decay: float,
    label_smoothing: float,
    epochs_per_update: int,
    end_trial: int | None,
    development_trials: int,
    output_prefix: Path | None = None,
) -> dict[str, Any]:
    seed_everything(seed)
    base = make_base_model(data, checkpoint)
    standard: StandardOnlineTrainer | None = None
    neuro: NeuroOnlineModelAdapter | None = None
    mask_generator = torch.Generator().manual_seed(seed)
    if method == "standard_online":
        standard = StandardOnlineTrainer(
            base,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            label_smoothing=label_smoothing,
        )
        predict = base.predict_proba
    elif method in {"neurooffline_static", "neuroonline"}:
        neuro = NeuroOnlineModelAdapter(
            base,
            config=NeuroOnlineConfig(
                enabled=True,
                learning_rate=learning_rate,
                update_batch_size=update_windows,
                epochs=epochs_per_update,
                weight_decay=weight_decay,
                mask_ratio=mask_ratio,
                consistency_weight=consistency_weight,
                label_smoothing=label_smoothing,
                random_seed=seed,
            ),
            state_path=checkpoint,
        )
        predict = neuro.predict_proba
    elif method == "static":
        predict = base.predict_proba
    else:
        raise ValueError(f"Unknown method: {method}")

    final_trial = len(data.trial_order) if end_trial is None else min(
        int(end_trial),
        len(data.trial_order),
    )
    pending_indices: list[int] = []
    all_truth: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    all_ordinals: list[np.ndarray] = []
    update_history: list[dict[str, Any]] = []
    initial_identity_error: float | None = None

    for ordinal in range(final_trial):
        indices = np.flatnonzero(data.trial_ordinal == ordinal)
        windows = data.windows[indices]
        if method == "neuroonline" and ordinal == 0:
            base_probabilities = base.predict_proba(windows)
            probabilities = predict(windows)
            initial_identity_error = float(
                np.max(np.abs(base_probabilities - probabilities))
            )
        else:
            probabilities = predict(windows)
        all_truth.append(data.labels[indices])
        all_probabilities.append(probabilities)
        all_ordinals.append(np.full(len(indices), ordinal, dtype=np.int64))

        if method in {"static", "neurooffline_static"}:
            continue
        pending_indices.extend(indices.tolist())
        while len(pending_indices) >= update_windows:
            update_indices = np.asarray(pending_indices[:update_windows], dtype=np.int64)
            del pending_indices[:update_windows]
            update_x = data.windows[update_indices]
            update_y = data.labels[update_indices]
            if standard is not None:
                update_metrics = standard.update(
                    update_x,
                    update_y,
                    epochs=epochs_per_update,
                )
            else:
                assert neuro is not None
                update_tensor = torch.as_tensor(update_x, dtype=torch.float32)
                time_view = _time_mask(
                    update_tensor,
                    mask_ratio,
                    mask_generator,
                ).numpy()
                frequency_view = _frequency_mask(
                    update_tensor,
                    mask_ratio,
                    mask_generator,
                ).numpy()
                update_metrics = neuro.neuroonline_update(
                    update_x,
                    time_view,
                    frequency_view,
                    update_y,
                    learning_rate=learning_rate,
                    epochs=epochs_per_update,
                    batch_size=update_windows,
                )
            update_history.append(
                {
                    "after_trial_ordinal": ordinal,
                    "after_trial_id": int(data.trial_order[ordinal]),
                    "windows_seen": int(sum(len(item) for item in all_truth)),
                    **update_metrics,
                }
            )

    truth = np.concatenate(all_truth)
    probabilities = np.concatenate(all_probabilities)
    ordinals = np.concatenate(all_ordinals)
    phases: dict[str, Any] = {}
    development_mask = ordinals < min(development_trials, final_trial)
    if development_mask.any():
        phases["development"] = metric_bundle(
            truth[development_mask],
            probabilities[development_mask],
            ordinals[development_mask],
        )
    final_mask = ordinals >= development_trials
    if final_mask.any():
        phases["final"] = metric_bundle(
            truth[final_mask],
            probabilities[final_mask],
            ordinals[final_mask],
        )
    result = {
        "method": method,
        "seed": seed,
        "update_windows": update_windows,
        "mask_ratio": (
            mask_ratio
            if method in {"neurooffline_static", "neuroonline"}
            else None
        ),
        "consistency_weight": (
            consistency_weight
            if method in {"neurooffline_static", "neuroonline"}
            else None
        ),
        "epochs_per_update": epochs_per_update,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "label_smoothing": label_smoothing,
        "streamed_trials": final_trial,
        "streamed_windows": int(len(truth)),
        "pending_untrained_windows": len(pending_indices),
        "update_count": len(update_history),
        "initial_neuro_identity_max_probability_error": initial_identity_error,
        "phases": phases,
        "update_history": update_history,
    }
    if output_prefix is not None:
        save_json(output_prefix.with_suffix(".json"), result)
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_prefix.with_suffix(".npz"),
            truth=truth,
            probabilities=probabilities,
            trial_ordinal=ordinals,
        )
    return result


def score(result: dict[str, Any]) -> tuple[float, float, float]:
    metrics = result["phases"]["development"]["trial"]
    values = (
        float(metrics["kappa"]),
        float(metrics["macro_f1"]),
        float(metrics["balanced_accuracy"]),
    )
    return tuple(value if np.isfinite(value) else float("-inf") for value in values)


def aggregate_candidate(
    runs: list[dict[str, Any]],
    **parameters: Any,
) -> dict[str, Any]:
    trial_mean: dict[str, float] = {}
    trial_std: dict[str, float] = {}
    for metric in ("accuracy", "balanced_accuracy", "macro_f1", "kappa"):
        values = np.asarray(
            [
                run["phases"]["development"]["trial"][metric]
                for run in runs
            ],
            dtype=np.float64,
        )
        trial_mean[metric] = float(values.mean())
        trial_std[metric] = float(values.std(ddof=0))
    return {
        **parameters,
        "trial_mean": trial_mean,
        "trial_std": trial_std,
        "seeds": [run["seed"] for run in runs],
        "update_counts": [run["update_count"] for run in runs],
    }


def candidate_score(candidate: dict[str, Any]) -> tuple[float, float, float]:
    if not candidate["update_counts"] or min(candidate["update_counts"]) < 1:
        return (float("-inf"), float("-inf"), float("-inf"))
    metrics = candidate["trial_mean"]
    values = (
        float(metrics["kappa"]),
        float(metrics["macro_f1"]),
        float(metrics["balanced_accuracy"]),
    )
    return tuple(value if np.isfinite(value) else float("-inf") for value in values)


def aggregate_final(
    runs: list[dict[str, Any]],
    methods: list[Method],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for method in methods:
        method_runs = [run for run in runs if run["method"] == method]
        output[method] = {"seeds": [run["seed"] for run in method_runs]}
        for level in ("window", "trial"):
            output[method][level] = {}
            for metric in ("accuracy", "balanced_accuracy", "macro_f1", "kappa"):
                values = np.asarray(
                    [run["phases"]["final"][level][metric] for run in method_runs],
                    dtype=np.float64,
                )
                output[method][level][metric] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                    "values": values.tolist(),
                }
        output[method]["update_counts"] = [run["update_count"] for run in method_runs]
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    data = load_online_data(
        args.day1,
        args.day2,
        online_blocks=args.online_blocks,
        scaler_trials=args.scaler_trials,
        online_start_trial=args.online_start_trial,
    )
    development_trials = (
        int(args.development_trials)
        if args.development_trials is not None
        else int(math.floor(len(data.trial_order) * args.development_fraction))
    )
    if not 0 < development_trials < len(data.trial_order):
        raise ValueError("Development split must leave at least one trial in each phase.")
    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        "day1": str(args.day1.resolve()),
        "day2": str(args.day2.resolve()),
        "checkpoint_root": str(args.checkpoint_root.resolve()),
        "trials": int(len(data.trial_order)),
        "windows": int(len(data.windows)),
        "development_trials": development_trials,
        "final_trials": int(len(data.trial_order) - development_trials),
        "online_blocks": args.online_blocks,
        "scaler_trials": args.scaler_trials,
        "online_start_trial": args.online_start_trial,
        "causal_order": "predict trial -> reveal label -> accumulate windows -> update at N",
        "update_windows_grid": args.update_windows_grid,
        "mask_ratio_grid": args.mask_ratio_grid,
        "lambda_grid": args.lambda_grid,
        "selection_seeds": args.selection_seeds,
        "final_seeds": args.final_seeds,
        "final_methods": args.final_methods,
        "epochs_per_update": args.epochs_per_update,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
    }
    save_json(args.output / "experiment_config.json", config)

    checkpoint_for = lambda seed: (
        args.checkpoint_root / f"seed_{seed}" / "shallowconvnet.pt"
    )
    selection_common = dict(
        data=data,
        method="neuroonline",
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        epochs_per_update=args.epochs_per_update,
        end_trial=development_trials,
        development_trials=development_trials,
    )

    preliminary: list[dict[str, Any]] = []
    for update_windows in args.update_windows_grid:
        candidate_runs: list[dict[str, Any]] = []
        for seed in args.selection_seeds:
            prefix = (
                args.output
                / "selection"
                / "01_batch_preliminary"
                / f"batch_{update_windows}"
                / f"seed_{seed}"
            )
            candidate_runs.append(
                simulate(
                    **selection_common,
                    checkpoint=checkpoint_for(seed),
                    seed=seed,
                    update_windows=update_windows,
                    mask_ratio=args.reference_mask_ratio,
                    consistency_weight=args.reference_lambda,
                    output_prefix=prefix,
                )
            )
        preliminary.append(
            aggregate_candidate(
                candidate_runs,
                update_windows=update_windows,
                mask_ratio=args.reference_mask_ratio,
                consistency_weight=args.reference_lambda,
            )
        )
    preliminary_best = max(preliminary, key=candidate_score)
    preliminary_batch = int(preliminary_best["update_windows"])

    joint: list[dict[str, Any]] = []
    for mask_ratio in args.mask_ratio_grid:
        for consistency_weight in args.lambda_grid:
            token = f"mask_{mask_ratio:g}_lambda_{consistency_weight:g}".replace(".", "p")
            candidate_runs = []
            for seed in args.selection_seeds:
                prefix = (
                    args.output
                    / "selection"
                    / "02_mask_lambda"
                    / token
                    / f"seed_{seed}"
                )
                candidate_runs.append(
                    simulate(
                        **selection_common,
                        checkpoint=checkpoint_for(seed),
                        seed=seed,
                        update_windows=preliminary_batch,
                        mask_ratio=mask_ratio,
                        consistency_weight=consistency_weight,
                        output_prefix=prefix,
                    )
                )
            joint.append(
                aggregate_candidate(
                    candidate_runs,
                    update_windows=preliminary_batch,
                    mask_ratio=mask_ratio,
                    consistency_weight=consistency_weight,
                )
            )
    joint_best = max(joint, key=candidate_score)
    selected_mask = float(joint_best["mask_ratio"])
    selected_lambda = float(joint_best["consistency_weight"])

    confirmation: list[dict[str, Any]] = []
    for update_windows in args.update_windows_grid:
        candidate_runs = []
        for seed in args.selection_seeds:
            prefix = (
                args.output
                / "selection"
                / "03_batch_confirmation"
                / f"batch_{update_windows}"
                / f"seed_{seed}"
            )
            candidate_runs.append(
                simulate(
                    **selection_common,
                    checkpoint=checkpoint_for(seed),
                    seed=seed,
                    update_windows=update_windows,
                    mask_ratio=selected_mask,
                    consistency_weight=selected_lambda,
                    output_prefix=prefix,
                )
            )
        confirmation.append(
            aggregate_candidate(
                candidate_runs,
                update_windows=update_windows,
                mask_ratio=selected_mask,
                consistency_weight=selected_lambda,
            )
        )
    confirmation_best = max(confirmation, key=candidate_score)
    selected_batch = int(confirmation_best["update_windows"])
    selection_summary = {
        "ranking_rule": "development trial kappa, then macro-F1, then balanced accuracy",
        "eligibility_rule": "each selection seed must trigger at least one real online update",
        "selection_seeds": args.selection_seeds,
        "preliminary": sorted(preliminary, key=candidate_score, reverse=True),
        "preliminary_selected_batch": preliminary_batch,
        "joint_mask_lambda": sorted(joint, key=candidate_score, reverse=True),
        "selected_mask_ratio": selected_mask,
        "selected_lambda": selected_lambda,
        "batch_confirmation": sorted(
            confirmation,
            key=candidate_score,
            reverse=True,
        ),
        "selected_update_windows": selected_batch,
    }
    save_json(args.output / "selection_summary.json", selection_summary)

    final_runs: list[dict[str, Any]] = []
    for seed in args.final_seeds:
        for method in args.final_methods:
            prefix = args.output / "final" / method / f"seed_{seed}"
            final_runs.append(
                simulate(
                    data=data,
                    checkpoint=checkpoint_for(seed),
                    method=method,
                    seed=seed,
                    update_windows=selected_batch,
                    mask_ratio=selected_mask,
                    consistency_weight=selected_lambda,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    label_smoothing=args.label_smoothing,
                    epochs_per_update=args.epochs_per_update,
                    end_trial=None,
                    development_trials=development_trials,
                    output_prefix=prefix,
                )
            )
    summary = {
        "config": config,
        "selected": {
            "update_windows": selected_batch,
            "mask_ratio": selected_mask,
            "lambda": selected_lambda,
        },
        "final": aggregate_final(final_runs, args.final_methods),
    }
    save_json(args.output / "final_summary.json", summary)
    return summary


def comma_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def comma_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day1", type=Path, required=True)
    parser.add_argument("--day2", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development-fraction", type=float, default=0.6)
    parser.add_argument("--development-trials", type=int)
    parser.add_argument("--online-blocks", type=comma_ints)
    parser.add_argument("--scaler-trials", type=int)
    parser.add_argument("--online-start-trial", type=int, default=0)
    parser.add_argument("--update-windows-grid", type=comma_ints, default=[32, 64, 128, 256])
    parser.add_argument("--mask-ratio-grid", type=comma_floats, default=[0.1, 0.3, 0.5, 0.7])
    parser.add_argument("--lambda-grid", type=comma_floats, default=[0.1, 0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--reference-mask-ratio", type=float, default=0.3)
    parser.add_argument("--reference-lambda", type=float, default=0.5)
    parser.add_argument("--selection-seeds", type=comma_ints, default=[17, 42, 2026])
    parser.add_argument("--final-seeds", type=comma_ints, default=[17, 42, 2026])
    parser.add_argument(
        "--final-methods",
        nargs="+",
        choices=[
            "static",
            "standard_online",
            "neurooffline_static",
            "neuroonline",
        ],
        default=["static", "standard_online", "neuroonline"],
    )
    parser.add_argument("--epochs-per-update", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
