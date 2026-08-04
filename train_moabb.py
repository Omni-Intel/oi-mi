"""Offline MOABB training entry point for EEG motor imagery models."""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import torch
import yaml

from download_datasets import canonicalize_dataset_name, resolve_dataset
from models.factory import ModelFactory
from utils.preprocessing import filter_and_transform

LOGGER = logging.getLogger("oi_mi.train_moabb")
DEFAULT_MODELS = ("cbramod",)


def setup_logging(verbose: bool) -> None:
    """Configure console logging."""

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def choose_subjects(
    dataset: object,
    subjects: tuple[int, ...],
    exclude_subjects: tuple[int, ...],
    all_subjects: bool,
) -> list[int]:
    """Resolve the final subject list used for offline training."""

    subject_list = getattr(dataset, "subject_list", None)
    if not subject_list:
        raise click.ClickException("Dataset does not expose a subject_list.")

    if all_subjects or not subjects:
        selected = list(subject_list)
    else:
        selected = list(dict.fromkeys(subjects))

    excluded = set(exclude_subjects)
    selected = [subject for subject in selected if subject not in excluded]
    if not selected:
        raise click.ClickException("No subjects remain after applying exclusions.")
    return selected


def choose_n_classes(dataset_name: str, requested_n_classes: int | None) -> int:
    """Pick a sensible default class count for the selected dataset."""

    if requested_n_classes is not None:
        return requested_n_classes
    if canonicalize_dataset_name(dataset_name) == "BNCI2014_001":
        return 4
    return 3


def choose_output_path(
    *,
    dataset_name: str,
    model_name: str,
    n_classes: int,
    subjects: list[int],
    output_path: Path | None,
) -> Path:
    """Build a default output path when the user does not provide one."""

    if output_path is not None:
        return output_path

    dataset_slug = canonicalize_dataset_name(dataset_name).lower()
    extension = ".pkl" if model_name == "riemann-mdm" else ".pt"
    if len(subjects) <= 5:
        subject_suffix = "-".join(f"s{subject:03d}" for subject in subjects)
    else:
        subject_suffix = f"{len(subjects)}subjects"
    filename = f"{model_name}-{dataset_slug}-{n_classes}class-{subject_suffix}{extension}"
    return Path("models_storage") / "offline" / dataset_slug / filename


def set_random_seed(seed: int) -> None:
    """Seed numpy and torch for more reproducible runs."""

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_windows(
    *,
    dataset_name: str,
    subjects: list[int],
    n_classes: int,
    data_dir: Path | None,
) -> tuple[np.ndarray, np.ndarray, object, dict[str, int]]:
    """Load MI windows from MOABB and encode labels to integer classes."""

    import moabb
    from moabb.paradigms import MotorImagery

    dataset_name = canonicalize_dataset_name(dataset_name)
    dataset = resolve_dataset(dataset_name)
    if data_dir is not None:
        data_dir.mkdir(parents=True, exist_ok=True)
        moabb.set_download_dir(str(data_dir))
        LOGGER.info("MOABB download dir set to %s", data_dir)

    paradigm = MotorImagery(n_classes=n_classes)
    X, labels, metadata = paradigm.get_data(dataset=dataset, subjects=subjects)
    X, labels, metadata = _enforce_label_subset(
        dataset_name=dataset_name,
        n_classes=n_classes,
        X=X,
        labels=labels,
        metadata=metadata,
    )
    unique_labels = [str(label) for label in np.unique(labels)]
    label_to_id = {label: index for index, label in enumerate(unique_labels)}
    encoded = np.asarray([label_to_id[str(label)] for label in labels], dtype=np.int64)
    return X.astype(np.float32), encoded, metadata, label_to_id


def _enforce_label_subset(
    *,
    dataset_name: str,
    n_classes: int,
    X: np.ndarray,
    labels: np.ndarray,
    metadata: object,
) -> tuple[np.ndarray, np.ndarray, object]:
    """Force BNCI2014_001 3-class mode to left/right/feet."""

    canonical_name = canonicalize_dataset_name(dataset_name)
    if canonical_name != "BNCI2014_001" or n_classes != 3:
        return X, labels, metadata

    preferred = ("left_hand", "right_hand", "feet")
    labels_str = np.asarray([str(label) for label in labels], dtype=object)
    available = set(labels_str.tolist())
    if not set(preferred).issubset(available):
        raise click.ClickException(
            "BNCI2014_001 3-class mode requires labels left_hand/right_hand/feet, "
            f"but got {sorted(available)}"
        )

    mask = np.isin(labels_str, preferred)
    filtered_X = X[mask]
    filtered_labels = labels_str[mask]

    filtered_metadata = metadata
    try:
        filtered_metadata = metadata.loc[mask].reset_index(drop=True)
    except Exception:  # noqa: BLE001
        filtered_metadata = metadata

    if filtered_X.shape[0] == 0:
        raise click.ClickException("No windows left after filtering to left/right/feet.")

    LOGGER.info(
        "Filtered BNCI2014_001 to left/right/feet: kept=%s dropped=%s",
        int(filtered_X.shape[0]),
        int(X.shape[0] - filtered_X.shape[0]),
    )
    return filtered_X, filtered_labels, filtered_metadata


def preprocess_windows(X: np.ndarray, sfreq: float) -> np.ndarray:
    """Apply the project's realtime preprocessing stack to each trial."""

    LOGGER.info("Preprocessing %s windows with sfreq=%s", X.shape[0], sfreq)
    processed = [filter_and_transform(window, sfreq=sfreq) for window in X]
    return np.stack(processed, axis=0).astype(np.float32)


def save_metadata(
    *,
    output_path: Path,
    dataset_name: str,
    subjects: list[int],
    n_classes: int,
    sfreq: float,
    model_name: str,
    label_to_id: dict[str, int],
    metrics: dict[str, float],
    windows_shape: tuple[int, ...],
    preprocess: bool,
) -> None:
    """Persist training metadata alongside the trained weights."""

    metadata = {
        "dataset_name": canonicalize_dataset_name(dataset_name),
        "subjects": subjects,
        "n_classes": n_classes,
        "sfreq": sfreq,
        "model_name": model_name,
        "label_to_id": label_to_id,
        "preprocess": preprocess,
        "windows_shape": list(windows_shape),
        "metrics": metrics,
        "model_path": str(output_path),
    }
    metadata_path = output_path.with_suffix(".metrics.yaml")
    with metadata_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False, allow_unicode=True)


@click.command()
@click.option(
    "--dataset",
    default="BNCI2014_001",
    show_default=True,
    help="MOABB dataset name or known alias.",
)
@click.option(
    "--subject",
    "subjects",
    multiple=True,
    type=int,
    help="Subject index to include. Can be provided multiple times.",
)
@click.option(
    "--exclude-subject",
    "exclude_subjects",
    multiple=True,
    type=int,
    help="Subject index to exclude. Useful when training a cross-subject base model.",
)
@click.option(
    "--all-subjects/--selected-subjects",
    default=True,
    show_default=True,
    help="Use all dataset subjects unless explicit selection is desired.",
)
@click.option(
    "--data-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Custom MOABB download/cache directory.",
)
@click.option(
    "--model",
    "model_name",
    type=click.Choice(DEFAULT_MODELS, case_sensitive=False),
    default="cbramod",
    show_default=True,
    help="Model registry name.",
)
@click.option(
    "--n-classes",
    type=click.IntRange(2, 4),
    default=None,
    help="Motor imagery class count. Defaults to 4 for BNCI2014_001, otherwise 3.",
)
@click.option("--epochs", type=int, default=50, show_default=True, help="Training epochs.")
@click.option("--batch-size", type=int, default=32, show_default=True, help="Batch size.")
@click.option(
    "--learning-rate",
    type=float,
    default=1e-3,
    show_default=True,
    help="Adam learning rate.",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Random seed.",
)
@click.option(
    "--preprocess/--no-preprocess",
    default=True,
    show_default=True,
    help="Apply the project's preprocessing stack before training.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Weight output path. Defaults to models_storage/offline/...",
)
@click.option("--verbose", is_flag=True, help="Enable debug logging.")
def main(
    dataset: str,
    subjects: tuple[int, ...],
    exclude_subjects: tuple[int, ...],
    all_subjects: bool,
    data_dir: Path | None,
    model_name: str,
    n_classes: int | None,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    preprocess: bool,
    output_path: Path | None,
    verbose: bool,
) -> None:
    """Train an offline model directly from MOABB motor imagery data."""

    setup_logging(verbose)
    set_random_seed(seed)

    dataset_name = canonicalize_dataset_name(dataset)
    dataset_obj = resolve_dataset(dataset_name)
    selected_subjects = choose_subjects(
        dataset_obj,
        subjects=subjects,
        exclude_subjects=exclude_subjects,
        all_subjects=all_subjects,
    )
    selected_n_classes = choose_n_classes(dataset_name, n_classes)
    output = choose_output_path(
        dataset_name=dataset_name,
        model_name=model_name.lower(),
        n_classes=selected_n_classes,
        subjects=selected_subjects,
        output_path=output_path,
    )

    LOGGER.info(
        "Loading dataset=%s subjects=%s model=%s n_classes=%s",
        dataset_name,
        selected_subjects,
        model_name,
        selected_n_classes,
    )
    X, y, metadata, label_to_id = load_windows(
        dataset_name=dataset_name,
        subjects=selected_subjects,
        n_classes=selected_n_classes,
        data_dir=data_dir,
    )
    sfreq = float(getattr(dataset_obj, "sfreq", 250.0))
    if preprocess:
        X = preprocess_windows(X, sfreq=sfreq)

    model = ModelFactory.get(
        model_name.lower(),
        n_chans=int(X.shape[1]),
        sfreq=sfreq,
        n_classes=int(len(label_to_id)),
        n_times=int(X.shape[2]),
    )
    metrics = model.fit(
        X,
        y,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        head_only=False,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    save_metadata(
        output_path=output,
        dataset_name=dataset_name,
        subjects=selected_subjects,
        n_classes=selected_n_classes,
        sfreq=sfreq,
        model_name=model_name.lower(),
        label_to_id=label_to_id,
        metrics=metrics,
        windows_shape=tuple(int(dimension) for dimension in X.shape),
        preprocess=preprocess,
    )

    LOGGER.info(
        "Offline training complete. windows=%s labels=%s val_acc=%.4f saved=%s",
        X.shape,
        len(label_to_id),
        metrics.get("val_acc", 0.0),
        output,
    )
    if metadata is not None:
        LOGGER.info("Metadata rows loaded: %s", len(metadata))


if __name__ == "__main__":
    main()
