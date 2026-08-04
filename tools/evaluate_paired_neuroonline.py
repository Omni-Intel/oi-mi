"""Evaluate causal NeuroOnline updates against a paired frozen baseline.

Inference randomness is reset per window and shared by both arms. Training
randomness is maintained separately so model updates cannot change the
dropout masks used by the counterfactual comparison.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from scipy.stats import binomtest
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adaptation.neuroonline import _frequency_mask, _time_mask  # noqa: E402
from tools.simulate_neuroonline_realtime import (  # noqa: E402
    artifact,
    build_adapter,
    load_checkpoint_config,
    load_committed_data,
    preprocess_windows,
    seed_everything,
)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = probabilities.argmax(axis=1)
    matrix = confusion_matrix(labels, predictions, labels=np.arange(3))
    support = matrix.sum(axis=1)
    recalls = np.divide(
        np.diag(matrix),
        support,
        out=np.zeros(3, dtype=np.float64),
        where=support > 0,
    )
    return {
        "samples": int(labels.size),
        "correct": int(np.sum(predictions == labels)),
        "accuracy": float(np.mean(predictions == labels)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "cross_entropy": float(
            -np.log(
                np.clip(
                    probabilities[np.arange(labels.size), labels],
                    1e-12,
                    1.0,
                )
            ).mean()
        ),
        "per_class_recall": recalls.tolist(),
        "confusion_matrix": matrix.astype(int).tolist(),
        "predictions": predictions,
    }


def _paired_summary(
    labels: np.ndarray,
    baseline_probabilities: np.ndarray,
    adapted_probabilities: np.ndarray,
) -> dict[str, Any]:
    baseline = _metrics(labels, baseline_probabilities)
    adapted = _metrics(labels, adapted_probabilities)
    baseline_predictions = baseline.pop("predictions")
    adapted_predictions = adapted.pop("predictions")
    wins = int(
        np.sum((baseline_predictions != labels) & (adapted_predictions == labels))
    )
    losses = int(
        np.sum((baseline_predictions == labels) & (adapted_predictions != labels))
    )
    discordant = wins + losses
    return {
        "baseline": baseline,
        "adapted": adapted,
        "delta": {
            "accuracy": adapted["accuracy"] - baseline["accuracy"],
            "balanced_accuracy": (
                adapted["balanced_accuracy"] - baseline["balanced_accuracy"]
            ),
            "macro_f1": adapted["macro_f1"] - baseline["macro_f1"],
            "cross_entropy": (
                adapted["cross_entropy"] - baseline["cross_entropy"]
            ),
        },
        "paired_correctness": {
            "adapted_wins": wins,
            "adapted_losses": losses,
            "discordant": discordant,
            "mcnemar_exact_two_sided_p": (
                float(binomtest(wins, discordant, 0.5).pvalue)
                if discordant
                else 1.0
            ),
        },
    }


def _capture_rng() -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    return (
        torch.get_rng_state().clone(),
        (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
    )


def _restore_rng(state: tuple[torch.Tensor, list[torch.Tensor] | None]) -> None:
    cpu, cuda = state
    torch.set_rng_state(cpu)
    if cuda is not None:
        torch.cuda.set_rng_state_all(cuda)


def _predict_with_window_seed(
    adapter: Any,
    window: np.ndarray,
    *,
    inference_seed: int,
    mc_dropout_passes: int,
) -> np.ndarray:
    seed_everything(inference_seed)
    return adapter.predict_proba(
        window[None, ...],
        mc_dropout_passes=mc_dropout_passes,
    )[0]


def _replay_seed(
    *,
    checkpoint: Path,
    windows: np.ndarray,
    labels: np.ndarray,
    config: Any,
    training_seed: int,
    inference_seed_base: int,
    mc_dropout_passes: int,
    max_updates: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    seeded_config = replace(config, random_seed=training_seed)
    adapter = build_adapter(
        checkpoint,
        model_name="shallowconvnet",
        n_chans=windows.shape[1],
        n_times=windows.shape[2],
        n_classes=3,
        sfreq=200.0,
        config=seeded_config,
    )
    seed_everything(training_seed)
    training_rng = _capture_rng()
    mask_generator = torch.Generator().manual_seed(training_seed)
    original: deque[np.ndarray] = deque(maxlen=seeded_config.recent_samples)
    time_views: deque[np.ndarray] = deque(maxlen=seeded_config.recent_samples)
    frequency_views: deque[np.ndarray] = deque(maxlen=seeded_config.recent_samples)
    replay_labels: deque[int] = deque(maxlen=seeded_config.recent_samples)
    probabilities: list[np.ndarray] = []
    revisions: list[int] = []
    updates: list[dict[str, Any]] = []
    revision = 0

    for index, (window, label) in enumerate(zip(windows, labels, strict=True)):
        probabilities.append(
            _predict_with_window_seed(
                adapter,
                window,
                inference_seed=inference_seed_base + index,
                mc_dropout_passes=mc_dropout_passes,
            )
        )
        revisions.append(revision)
        tensor = torch.as_tensor(window, dtype=torch.float32).unsqueeze(0)
        original.append(window.copy())
        time_views.append(
            _time_mask(tensor, seeded_config.mask_ratio, mask_generator)[0].numpy()
        )
        frequency_views.append(
            _frequency_mask(tensor, seeded_config.mask_ratio, mask_generator)[0].numpy()
        )
        replay_labels.append(int(label))

        seen = index + 1
        if (
            revision >= max_updates
            or seen < seeded_config.history_threshold
            or seen % seeded_config.update_stride != 0
        ):
            continue
        _restore_rng(training_rng)
        result = adapter.neuroonline_update(
            np.stack(original),
            np.stack(time_views),
            np.stack(frequency_views),
            np.asarray(replay_labels, dtype=np.int64),
            learning_rate=seeded_config.learning_rate,
            epochs=seeded_config.epochs,
            batch_size=seeded_config.update_batch_size,
        )
        training_rng = _capture_rng()
        revision += 1
        updates.append(
            {
                "update": revision,
                "after_window": seen,
                "training_samples": len(replay_labels),
                **{key: float(value) for key, value in result.items()},
            }
        )

    return (
        np.asarray(probabilities, dtype=np.float32),
        np.asarray(revisions, dtype=np.int64),
        updates,
    )


def _frozen_baseline(
    *,
    checkpoint: Path,
    windows: np.ndarray,
    config: Any,
    inference_seed_base: int,
    mc_dropout_passes: int,
) -> np.ndarray:
    adapter = build_adapter(
        checkpoint,
        model_name="shallowconvnet",
        n_chans=windows.shape[1],
        n_times=windows.shape[2],
        n_classes=3,
        sfreq=200.0,
        config=config,
    )
    return np.asarray(
        [
            _predict_with_window_seed(
                adapter,
                window,
                inference_seed=inference_seed_base + index,
                mc_dropout_passes=mc_dropout_passes,
            )
            for index, window in enumerate(windows)
        ],
        dtype=np.float32,
    )


def _block_bootstrap(
    labels: np.ndarray,
    baseline: np.ndarray,
    adapted: np.ndarray,
    *,
    block_length: int,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    size = labels.size
    blocks = int(np.ceil(size / block_length))
    deltas = np.empty((resamples, 3), dtype=np.float64)
    for sample_index in range(resamples):
        starts = rng.integers(0, size, size=blocks)
        indices = np.concatenate(
            [
                (start + np.arange(block_length, dtype=np.int64)) % size
                for start in starts
            ]
        )[:size]
        paired = _paired_summary(labels[indices], baseline[indices], adapted[indices])
        deltas[sample_index] = (
            paired["delta"]["accuracy"],
            paired["delta"]["balanced_accuracy"],
            paired["delta"]["cross_entropy"],
        )
    names = ("accuracy", "balanced_accuracy", "cross_entropy")
    return {
        "method": "circular_moving_block_bootstrap",
        "block_length": block_length,
        "resamples": resamples,
        "seed": seed,
        "delta_95_percentile_interval": {
            name: np.percentile(deltas[:, index], [2.5, 97.5]).tolist()
            for index, name in enumerate(names)
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve()
    recording = args.recording.resolve()
    data = load_committed_data(recording)
    windows, quality = preprocess_windows(data, sfreq=200.0)
    if quality["rejected_windows"]:
        raise ValueError("Committed replay data unexpectedly failed preprocessing.")
    base_config, _ = load_checkpoint_config(checkpoint)
    config = replace(
        base_config,
        enabled=True,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        update_batch_size=args.batch_size,
        history_threshold=64,
        update_stride=64,
        recent_samples=320,
        mask_ratio=args.mask_ratio,
        consistency_weight=args.consistency_weight,
    )
    baseline = _frozen_baseline(
        checkpoint=checkpoint,
        windows=windows,
        config=config,
        inference_seed_base=args.inference_seed_base,
        mc_dropout_passes=args.mc_dropout_passes,
    )
    post_update = np.arange(data.labels.size) >= config.history_threshold
    runs: list[dict[str, Any]] = []
    primary_probabilities: np.ndarray | None = None
    primary_revisions: np.ndarray | None = None

    for training_seed in args.training_seeds:
        adapted, revisions, updates = _replay_seed(
            checkpoint=checkpoint,
            windows=windows,
            labels=data.labels,
            config=config,
            training_seed=training_seed,
            inference_seed_base=args.inference_seed_base,
            mc_dropout_passes=args.mc_dropout_passes,
            max_updates=args.max_updates,
        )
        if training_seed == args.primary_seed:
            primary_probabilities = adapted
            primary_revisions = revisions
        runs.append(
            {
                "training_seed": training_seed,
                "post_update": _paired_summary(
                    data.labels[post_update],
                    baseline[post_update],
                    adapted[post_update],
                ),
                "full_stream": _paired_summary(data.labels, baseline, adapted),
                "updates": updates,
            }
        )
        print(
            f"SEED {training_seed} "
            f"delta_acc={runs[-1]['post_update']['delta']['accuracy']:.6f} "
            f"delta_bacc={runs[-1]['post_update']['delta']['balanced_accuracy']:.6f}",
            flush=True,
        )

    if primary_probabilities is None or primary_revisions is None:
        raise ValueError("The primary seed must be included in --training-seeds.")
    primary = next(item for item in runs if item["training_seed"] == args.primary_seed)
    primary["phases"] = [
        {
            "revision": int(revision),
            **_paired_summary(
                data.labels[primary_revisions == revision],
                baseline[primary_revisions == revision],
                primary_probabilities[primary_revisions == revision],
            ),
        }
        for revision in np.unique(primary_revisions)
    ]
    primary["post_update"]["block_bootstrap"] = _block_bootstrap(
        data.labels[post_update],
        baseline[post_update],
        primary_probabilities[post_update],
        block_length=args.bootstrap_block_length,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    delta_keys = ("accuracy", "balanced_accuracy", "macro_f1", "cross_entropy")
    delta_matrix = np.asarray(
        [
            [item["post_update"]["delta"][key] for key in delta_keys]
            for item in runs
        ],
        dtype=np.float64,
    )
    report = {
        "schema_version": 1,
        "method": "paired_common_random_numbers_causal_neuroonline_replay",
        "claim_scope": "single_subject_single_session_within_session_counterfactual",
        "source_recording": str(recording),
        "source_checkpoint": artifact(checkpoint),
        "source_sidecar": artifact(Path(f"{checkpoint}.neuroonline.pt")),
        "stream": {
            "samples": int(data.labels.size),
            "class_counts": np.bincount(data.labels, minlength=3).astype(int).tolist(),
            "selection": "recorded_adaptation_committed_primary_decision_windows",
            "causal_order": "predict_then_reveal_label_then_update",
        },
        "config": {
            **asdict(config),
            "mc_dropout_passes": args.mc_dropout_passes,
            "max_updates": args.max_updates,
            "inference_seed_base": args.inference_seed_base,
            "training_seeds": args.training_seeds,
            "primary_seed": args.primary_seed,
            "inference_rng": "per_window_common_random_numbers",
            "training_rng": "separate_persistent_state_per_training_seed",
        },
        "frozen_baseline_full_stream": {
            key: value
            for key, value in _metrics(data.labels, baseline).items()
            if key != "predictions"
        },
        "primary_run": primary,
        "seed_sensitivity": {
            "seeds": len(runs),
            "delta_keys": list(delta_keys),
            "mean": delta_matrix.mean(axis=0).tolist(),
            "sample_std": delta_matrix.std(axis=0, ddof=1).tolist(),
            "minimum": delta_matrix.min(axis=0).tolist(),
            "maximum": delta_matrix.max(axis=0).tolist(),
            "positive_accuracy_seeds": int(np.sum(delta_matrix[:, 0] > 0.0)),
            "runs": runs,
        },
    }
    np.savez_compressed(
        args.output / "primary_paired_predictions.npz",
        labels=data.labels,
        scene_indices=data.scene_indices,
        baseline_probabilities=baseline,
        adapted_probabilities=primary_probabilities,
        model_revisions=primary_revisions,
    )
    _save_json(args.output / "paired_evaluation_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mask-ratio", type=float, default=0.1)
    parser.add_argument("--consistency-weight", type=float, default=1.5)
    parser.add_argument("--mc-dropout-passes", type=int, default=8)
    parser.add_argument("--max-updates", type=int, default=7)
    parser.add_argument("--training-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--primary-seed", type=int, default=2026)
    parser.add_argument("--inference-seed-base", type=int, default=100000)
    parser.add_argument("--bootstrap-block-length", type=int, default=16)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260728)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
