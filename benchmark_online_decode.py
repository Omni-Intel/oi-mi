"""Offline benchmark for calibration-to-test_mode decoding."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import click
import numpy as np

from cli import get_model_factory
from utils.preprocessing import filter_and_transform


def load_processed_test_windows(test_chunk_path: Path, *, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    with np.load(test_chunk_path) as payload:
        X_raw = payload["eeg_windows"].astype(np.float32)
        y_true = payload["labels_true"].astype(np.int64)
    X = np.stack([filter_and_transform(window, sfreq=sfreq) for window in X_raw], axis=0).astype(np.float32)
    return X, y_true


def smooth_probabilities(probabilities: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return probabilities
    half = kernel_size // 2
    smoothed = np.zeros_like(probabilities)
    for index in range(probabilities.shape[0]):
        start = max(0, index - half)
        stop = min(probabilities.shape[0], index + half + 1)
        smoothed[index] = probabilities[start:stop].mean(axis=0)
    return smoothed


def viterbi_decode(probabilities: np.ndarray, *, stay_probability: float, prior: np.ndarray | None = None) -> np.ndarray:
    n_classes = probabilities.shape[1]
    transition = np.full((n_classes, n_classes), (1.0 - stay_probability) / max(n_classes - 1, 1), dtype=np.float64)
    np.fill_diagonal(transition, stay_probability)
    log_transition = np.log(np.clip(transition, 1e-8, 1.0))
    if prior is None:
        prior = np.full(n_classes, 1.0 / n_classes, dtype=np.float64)
    log_prior = np.log(np.clip(prior, 1e-8, 1.0))
    log_prob = np.log(np.clip(probabilities.astype(np.float64), 1e-8, 1.0))

    n_steps = probabilities.shape[0]
    dp = np.empty((n_steps, n_classes), dtype=np.float64)
    back = np.empty((n_steps, n_classes), dtype=np.int64)
    dp[0] = log_prior + log_prob[0]
    back[0] = -1
    for step in range(1, n_steps):
        scores = dp[step - 1][:, None] + log_transition
        back[step] = scores.argmax(axis=0)
        dp[step] = scores.max(axis=0) + log_prob[step]
    path = np.empty(n_steps, dtype=np.int64)
    path[-1] = int(dp[-1].argmax())
    for step in range(n_steps - 2, -1, -1):
        path[step] = back[step + 1, path[step + 1]]
    return path


def evaluate_predictions(y_true: np.ndarray, probabilities: np.ndarray, *, threshold: float) -> dict[str, Any]:
    argmax_pred = probabilities.argmax(axis=1).astype(np.int64)
    confidence = probabilities.max(axis=1)
    gated_pred = np.where(confidence >= threshold, argmax_pred, -1)
    valid_mask = gated_pred >= 0
    return {
        "argmax_acc": float(np.mean(argmax_pred == y_true)),
        "coverage": float(np.mean(valid_mask)),
        "valid_acc": float(np.mean(gated_pred[valid_mask] == y_true[valid_mask])) if np.any(valid_mask) else None,
        "mean_confidence": float(np.mean(confidence)),
        "argmax_counts": dict(Counter(int(value) for value in argmax_pred.tolist())),
        "gated_counts": dict(Counter(int(value) for value in gated_pred.tolist())),
    }


@click.command()
@click.option("--calibration", "calibration_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--test-chunk", "test_chunk_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--model", "model_name", type=str, default="eegnet", show_default=True)
@click.option("--sfreq", type=float, default=200.0, show_default=True)
@click.option("--threshold", type=float, default=0.5, show_default=True)
@click.option("--epochs", type=int, default=35, show_default=True)
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option("--learning-rate", type=float, default=1e-3, show_default=True)
@click.option("--patience", type=int, default=12, show_default=True)
@click.option("--mc-dropout-passes", type=int, default=8, show_default=True)
@click.option("--smooth", "smooth_sizes", type=int, multiple=True, default=(1, 3, 5))
@click.option("--hmm-stay", type=float, default=None, help="Optional Viterbi self-transition probability.")
@click.option("--output-json", type=click.Path(dir_okay=False, path_type=Path), default=None)
def main(
    calibration_path: Path,
    test_chunk_path: Path,
    model_name: str,
    sfreq: float,
    threshold: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    mc_dropout_passes: int,
    smooth_sizes: tuple[int, ...],
    hmm_stay: float | None,
    output_json: Path | None,
) -> None:
    with np.load(calibration_path) as payload:
        X_train = payload["processed_windows"].astype(np.float32)
        y_train = payload["labels"].astype(np.int64)
    X_test, y_test = load_processed_test_windows(test_chunk_path, sfreq=sfreq)

    factory = get_model_factory()
    model = factory.get(
        model_name,
        n_chans=int(X_train.shape[1]),
        sfreq=sfreq,
        n_classes=int(len(np.unique(y_train))),
        n_times=int(X_train.shape[2]),
    )
    fit_metrics = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        patience=patience,
        head_only=False,
    )
    probabilities = model.predict_proba(X_test, mc_dropout_passes=mc_dropout_passes)

    results: dict[str, Any] = {
        "model_name": model_name,
        "fit_metrics": fit_metrics,
        "threshold": threshold,
        "smooth_results": {},
    }
    for kernel_size in smooth_sizes:
        smoothed = smooth_probabilities(probabilities, kernel_size)
        results["smooth_results"][str(kernel_size)] = evaluate_predictions(y_test, smoothed, threshold=threshold)

    if hmm_stay is not None:
        path = viterbi_decode(probabilities, stay_probability=float(hmm_stay))
        hmm_probs = np.eye(probabilities.shape[1], dtype=np.float32)[path]
        results["hmm_result"] = {
            "stay_probability": hmm_stay,
            **evaluate_predictions(y_test, hmm_probs, threshold=0.0),
        }

    rendered = json.dumps(results, ensure_ascii=False, indent=2)
    print(rendered)
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
