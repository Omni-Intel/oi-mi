"""Pretrain on source subjects, personalize on a target subject, and replay test_mode."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import click
import numpy as np
import torch

from benchmark_online_decode import evaluate_predictions, load_processed_test_windows, smooth_probabilities
from cli import get_model_factory


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _subject_dir(records_dir: Path, subject_id: str) -> Path:
    return records_dir / subject_id / "calibration"


def find_calibration_sessions(
    records_dir: Path,
    subject_id: str,
    *,
    min_windows: int,
    latest_only: bool,
) -> list[Path]:
    root = _subject_dir(records_dir, subject_id)
    if not root.exists():
        raise click.ClickException(f"Calibration directory not found: {root}")
    sessions: list[Path] = []
    for session_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        dataset_path = session_dir / "training_windows_main.npz"
        if not dataset_path.exists():
            continue
        with np.load(dataset_path) as payload:
            n_windows = int(payload["labels"].shape[0])
        if n_windows >= min_windows:
            sessions.append(session_dir)
    if not sessions:
        raise click.ClickException(f"No usable calibration sessions for {subject_id} in {root}")
    return sessions[-1:] if latest_only else sessions


def load_subject_windows(
    records_dir: Path,
    subject_id: str,
    *,
    min_windows: int,
    latest_only: bool,
    array_key: str = "processed_windows",
) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    sessions = find_calibration_sessions(
        records_dir,
        subject_id,
        min_windows=min_windows,
        latest_only=latest_only,
    )
    windows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    expected_shape: tuple[int, int] | None = None
    for session_dir in sessions:
        with np.load(session_dir / "training_windows_main.npz") as payload:
            X = payload[array_key].astype(np.float32)
            y = payload["labels"].astype(np.int64)
        current_shape = (int(X.shape[1]), int(X.shape[2]))
        if expected_shape is None:
            expected_shape = current_shape
        elif current_shape != expected_shape:
            raise click.ClickException(
                f"Inconsistent window shape for {subject_id}: expected {expected_shape}, got {current_shape}"
            )
        windows.append(X)
        labels.append(y)
    return np.concatenate(windows, axis=0), np.concatenate(labels, axis=0), sessions


def evaluate_probability_set(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    *,
    threshold: float,
    smooth_sizes: tuple[int, ...],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for kernel_size in smooth_sizes:
        smoothed = smooth_probabilities(probabilities, kernel_size)
        results[str(kernel_size)] = evaluate_predictions(y_true, smoothed, threshold=threshold)
    return results


def train_and_predict(
    *,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    sfreq: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    head_only: bool = False,
    pretrained_state_path: Path | None = None,
    mc_dropout_passes: int = 8,
) -> tuple[dict[str, float], np.ndarray]:
    model = get_model_factory().get(
        model_name,
        n_chans=int(X_train.shape[1]),
        sfreq=sfreq,
        n_classes=int(len(np.unique(y_train))),
        n_times=int(X_train.shape[2]),
    )
    if pretrained_state_path is not None:
        model.load(pretrained_state_path)
    metrics = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        patience=patience,
        head_only=head_only,
    )
    probabilities = model.predict_proba(X_test, mc_dropout_passes=mc_dropout_passes)
    return metrics, probabilities


@click.command()
@click.option("--records-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--source-subject", "source_subjects", multiple=True, required=True)
@click.option("--target-subject", required=True)
@click.option("--test-chunk", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--model", "model_name", type=str, default="eegnet", show_default=True)
@click.option("--sfreq", type=float, default=250.0, show_default=True)
@click.option("--threshold", type=float, default=0.5, show_default=True)
@click.option("--smooth", "smooth_sizes", type=int, multiple=True, default=(1, 3, 5))
@click.option("--min-windows", type=int, default=60, show_default=True)
@click.option("--latest-source/--all-source", default=True, show_default=True)
@click.option("--latest-target/--all-target", default=True, show_default=True)
@click.option("--pretrain-epochs", type=int, default=35, show_default=True)
@click.option("--finetune-epochs", type=int, default=20, show_default=True)
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option("--learning-rate", type=float, default=1e-3, show_default=True)
@click.option("--finetune-learning-rate", type=float, default=3e-4, show_default=True)
@click.option("--patience", type=int, default=10, show_default=True)
@click.option("--mc-dropout-passes", type=int, default=8, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--output-json", type=click.Path(dir_okay=False, path_type=Path), default=None)
def main(
    records_dir: Path,
    source_subjects: tuple[str, ...],
    target_subject: str,
    test_chunk: Path,
    model_name: str,
    sfreq: float,
    threshold: float,
    smooth_sizes: tuple[int, ...],
    min_windows: int,
    latest_source: bool,
    latest_target: bool,
    pretrain_epochs: int,
    finetune_epochs: int,
    batch_size: int,
    learning_rate: float,
    finetune_learning_rate: float,
    patience: int,
    mc_dropout_passes: int,
    seed: int,
    output_json: Path | None,
) -> None:
    """Run scratch, transfer-full, and transfer-head personalization benchmarks."""

    set_seed(seed)
    source_X_parts: list[np.ndarray] = []
    source_y_parts: list[np.ndarray] = []
    source_sessions: dict[str, list[str]] = {}
    for subject_id in source_subjects:
        X, y, sessions = load_subject_windows(
            records_dir,
            subject_id,
            min_windows=min_windows,
            latest_only=latest_source,
        )
        source_X_parts.append(X)
        source_y_parts.append(y)
        source_sessions[subject_id] = [session.name for session in sessions]
    X_source = np.concatenate(source_X_parts, axis=0)
    y_source = np.concatenate(source_y_parts, axis=0)

    X_target, y_target, target_sessions = load_subject_windows(
        records_dir,
        target_subject,
        min_windows=min_windows,
        latest_only=latest_target,
    )
    X_test, y_test = load_processed_test_windows(test_chunk, sfreq=sfreq)

    tmp_dir = Path("/tmp") / "oi_mi_transfer"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pretrain_path = tmp_dir / f"{model_name}_{'_'.join(source_subjects)}_pretrain.pt"

    set_seed(seed)
    source_model = get_model_factory().get(
        model_name,
        n_chans=int(X_source.shape[1]),
        sfreq=sfreq,
        n_classes=int(len(np.unique(y_source))),
        n_times=int(X_source.shape[2]),
    )
    pretrain_metrics = source_model.fit(
        X_source,
        y_source,
        epochs=pretrain_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        patience=patience,
        head_only=False,
    )
    source_model.save(pretrain_path)

    set_seed(seed)
    scratch_metrics, scratch_probs = train_and_predict(
        model_name=model_name,
        X_train=X_target,
        y_train=y_target,
        X_test=X_test,
        sfreq=sfreq,
        epochs=pretrain_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        patience=patience,
        mc_dropout_passes=mc_dropout_passes,
    )

    set_seed(seed)
    transfer_full_metrics, transfer_full_probs = train_and_predict(
        model_name=model_name,
        X_train=X_target,
        y_train=y_target,
        X_test=X_test,
        sfreq=sfreq,
        epochs=finetune_epochs,
        batch_size=batch_size,
        learning_rate=finetune_learning_rate,
        patience=patience,
        pretrained_state_path=pretrain_path,
        head_only=False,
        mc_dropout_passes=mc_dropout_passes,
    )

    set_seed(seed)
    transfer_head_metrics, transfer_head_probs = train_and_predict(
        model_name=model_name,
        X_train=X_target,
        y_train=y_target,
        X_test=X_test,
        sfreq=sfreq,
        epochs=finetune_epochs,
        batch_size=batch_size,
        learning_rate=finetune_learning_rate,
        patience=patience,
        pretrained_state_path=pretrain_path,
        head_only=True,
        mc_dropout_passes=mc_dropout_passes,
    )

    results: dict[str, Any] = {
        "model_name": model_name,
        "source_subjects": list(source_subjects),
        "target_subject": target_subject,
        "source_sessions": source_sessions,
        "target_sessions": [session.name for session in target_sessions],
        "source_shape": list(X_source.shape),
        "target_shape": list(X_target.shape),
        "test_shape": list(X_test.shape),
        "source_label_counts": dict(Counter(int(value) for value in y_source.tolist())),
        "target_label_counts": dict(Counter(int(value) for value in y_target.tolist())),
        "test_label_counts": dict(Counter(int(value) for value in y_test.tolist())),
        "pretrain_metrics": pretrain_metrics,
        "scratch": {
            "metrics": scratch_metrics,
            "smooth_results": evaluate_probability_set(scratch_probs, y_test, threshold=threshold, smooth_sizes=smooth_sizes),
        },
        "transfer_full": {
            "metrics": transfer_full_metrics,
            "smooth_results": evaluate_probability_set(
                transfer_full_probs,
                y_test,
                threshold=threshold,
                smooth_sizes=smooth_sizes,
            ),
        },
        "transfer_head": {
            "metrics": transfer_head_metrics,
            "smooth_results": evaluate_probability_set(
                transfer_head_probs,
                y_test,
                threshold=threshold,
                smooth_sizes=smooth_sizes,
            ),
        },
    }
    rendered = json.dumps(results, ensure_ascii=False, indent=2)
    print(rendered)
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
