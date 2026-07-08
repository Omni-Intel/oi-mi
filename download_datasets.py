"""Download and warm up MOABB motor imagery datasets for oi-mi.

Use this script on a server to pre-download datasets before running:
- cross-subject pretraining
- no-hardware smoke tests

For pure real-device Neuracle testing, this script is not required.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click

LOGGER = logging.getLogger("oi_mi.download_datasets")
DATASET_ALIASES = {
    "BCI Competition IV 2a": "BNCI2014_001",
    "BCICIV2A": "BNCI2014_001",
    "BCIC_IV_2A": "BNCI2014_001",
}


def setup_logging(verbose: bool) -> None:
    """Configure console logging."""

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def canonicalize_dataset_name(dataset_name: str) -> str:
    """Resolve common aliases to the MOABB dataset identifier."""

    normalized = dataset_name.strip()
    return DATASET_ALIASES.get(normalized, normalized)


def resolve_dataset(dataset_name: str) -> object:
    """Map a user-facing dataset name to a MOABB dataset instance."""

    from moabb.datasets import BNCI2014_001, Cho2017, PhysionetMI

    dataset_name = canonicalize_dataset_name(dataset_name)
    mapping: dict[str, type[object]] = {
        "BNCI2014_001": BNCI2014_001,
        "PhysioNet": PhysionetMI,
        "LeftRightImagery": Cho2017,
    }
    if dataset_name not in mapping:
        available = ", ".join(sorted(mapping))
        raise click.ClickException(
            f"Unsupported dataset '{dataset_name}'. Available datasets: {available}"
        )
    return mapping[dataset_name]()


def parse_subjects(
    dataset: object,
    subjects: tuple[int, ...],
    all_subjects: bool,
    full_dataset: bool,
) -> list[int]:
    """Choose which subjects to download."""

    subject_list = getattr(dataset, "subject_list", None)
    if all_subjects or full_dataset:
        if not subject_list:
            raise click.ClickException("Dataset does not expose a subject_list.")
        return list(subject_list)
    if subjects:
        return list(subjects)
    return [1]


def resolve_cache_classes(
    dataset_name: str,
    requested_n_classes: int | None,
    *,
    full_dataset: bool,
) -> int:
    """Pick the MotorImagery class count used for cache warming."""

    if requested_n_classes is not None:
        return requested_n_classes

    canonical_name = canonicalize_dataset_name(dataset_name)
    if full_dataset and canonical_name == "BNCI2014_001":
        # BCI Competition IV 2a is a four-class MI benchmark.
        return 4
    return 3


def download_dataset(
    *,
    dataset_name: str,
    subjects: list[int],
    data_dir: Path | None,
    warm_paradigm_cache: bool,
    n_classes: int,
) -> None:
    """Download a dataset and optionally warm the MotorImagery cache."""

    import moabb

    dataset = resolve_dataset(dataset_name)

    if data_dir is not None:
        data_dir.mkdir(parents=True, exist_ok=True)
        moabb.set_download_dir(str(data_dir))
        LOGGER.info("MOABB download dir set to %s", data_dir)

    LOGGER.info("Downloading dataset %s for subjects=%s", dataset_name, subjects)
    dataset.get_data(subjects=subjects)
    LOGGER.info("Raw dataset download complete for %s", dataset_name)

    if warm_paradigm_cache:
        from moabb.paradigms import MotorImagery

        LOGGER.info(
            "Warming MotorImagery cache for %s with n_classes=%s",
            dataset_name,
            n_classes,
        )
        paradigm = MotorImagery(n_classes=n_classes)
        X, y, metadata = paradigm.get_data(dataset=dataset, subjects=subjects)
        LOGGER.info(
            "Paradigm cache ready for %s: windows=%s labels=%s metadata_rows=%s",
            dataset_name,
            getattr(X, "shape", None),
            len(y),
            len(metadata),
        )


@click.command()
@click.option(
    "--dataset",
    "datasets",
    multiple=True,
    default=("BNCI2014_001",),
    show_default=True,
    help="Dataset name. Can be provided multiple times.",
)
@click.option(
    "--subject",
    "subjects",
    multiple=True,
    type=int,
    help="Subject index to download. Can be provided multiple times.",
)
@click.option(
    "--all-subjects",
    is_flag=True,
    help="Download all subjects exposed by the dataset.",
)
@click.option(
    "--full-dataset",
    is_flag=True,
    help="Download the complete dataset. For BNCI2014_001 this means all subjects and a 4-class cache.",
)
@click.option(
    "--data-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Custom MOABB download/cache directory.",
)
@click.option(
    "--warm-paradigm-cache/--raw-only",
    default=True,
    show_default=True,
    help="Also run MotorImagery.get_data to warm derived cache.",
)
@click.option(
    "--n-classes",
    type=click.IntRange(2, 4),
    default=None,
    help="MotorImagery class count used when warming paradigm cache. Defaults to 4 for BNCI2014_001 with --full-dataset, otherwise 3.",
)
@click.option("--verbose", is_flag=True, help="Enable debug logging.")
def main(
    datasets: tuple[str, ...],
    subjects: tuple[int, ...],
    all_subjects: bool,
    full_dataset: bool,
    data_dir: Path | None,
    warm_paradigm_cache: bool,
    n_classes: int | None,
    verbose: bool,
) -> None:
    """Download datasets needed by offline training and pretraining."""

    setup_logging(verbose)
    unique_datasets = [canonicalize_dataset_name(name) for name in dict.fromkeys(datasets)]
    LOGGER.info("Requested datasets: %s", unique_datasets)

    for dataset_name in unique_datasets:
        dataset = resolve_dataset(dataset_name)
        selected_subjects = parse_subjects(
            dataset,
            subjects,
            all_subjects=all_subjects,
            full_dataset=full_dataset,
        )
        selected_n_classes = resolve_cache_classes(
            dataset_name,
            n_classes,
            full_dataset=full_dataset,
        )
        try:
            download_dataset(
                dataset_name=dataset_name,
                subjects=selected_subjects,
                data_dir=data_dir,
                warm_paradigm_cache=warm_paradigm_cache,
                n_classes=selected_n_classes,
            )
        except Exception as exc:  # noqa: BLE001
            raise click.ClickException(
                f"Failed to download dataset '{dataset_name}': {exc}"
            ) from exc

    LOGGER.info("All requested dataset downloads completed successfully.")


if __name__ == "__main__":
    main()
