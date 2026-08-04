"""Train one predeclared-seed CBraMod checkpoint from the selected offline CV config."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adaptation.neuroonline import (  # noqa: E402
    NEUROONLINE_TRAINING_MECHANICS_VERSION,
    NeuroOnlineConfig,
    _aggregate_trial_probabilities,
)
from tools.search_cbramod_offline_group_cv import (  # noqa: E402
    _build_adapter,
    _load_dataset,
    _sha256,
    _write_json,
)
from tools.simulate_neuroonline_realtime import artifact, classification_metrics  # noqa: E402


def run(args: argparse.Namespace) -> dict:
    dataset = args.dataset.resolve()
    summary_path = args.offline_summary.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    ranked = summary.get("ranked")
    if not isinstance(ranked, list) or not ranked:
        raise ValueError("Offline CV summary has no selected candidate.")
    if summary.get("fixed_epoch_training") is not True:
        raise ValueError("Offline CV summary was not produced by fixed-epoch training.")
    if int(summary.get("training_mechanics_version", 0)) != (
        NEUROONLINE_TRAINING_MECHANICS_VERSION
    ):
        raise ValueError(
            "Offline CV summary training mechanics do not match the current code."
        )
    if int(summary.get("completed_candidates", -1)) != int(
        summary.get("expected_candidates", -2)
    ):
        raise ValueError("Offline CV summary is incomplete; refusing final training.")
    candidate = ranked[0]["candidate"]
    selected_epochs = args.epochs
    if selected_epochs is None:
        selected_epochs = ranked[0].get("summary", {}).get("median_best_epoch")
    if selected_epochs is None:
        raise ValueError(
            "Selected offline summary has no median_best_epoch; pass --epochs explicitly."
        )
    selected_epochs = int(selected_epochs)
    if selected_epochs < 1:
        raise ValueError("Final offline epochs must be positive.")
    identity = {
        "schema_version": 1,
        "training_mechanics_version": NEUROONLINE_TRAINING_MECHANICS_VERSION,
        "fixed_epoch_cv": True,
        "final_training_strategy": "full_calibration_at_median_cv_best_epoch",
        "dataset": str(dataset),
        "dataset_sha256": _sha256(dataset),
        "offline_summary": str(summary_path),
        "offline_summary_sha256": _sha256(summary_path),
        "selected_candidate": candidate,
        "selected_candidate_index": int(ranked[0]["candidate_index"]),
        "seed": args.seed,
        "epochs": selected_epochs,
        "scheduler_horizon_epochs": args.max_epochs,
        "feature_key": args.feature_key,
    }
    report_path = output_dir / "training_report.json"
    if report_path.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("identity") != identity:
            raise ValueError(f"Existing final-training report has incompatible inputs: {report_path}")
        print(f"REUSED {report_path}", flush=True)
        return existing

    windows, labels, groups, sfreq = _load_dataset(dataset, args.feature_key)
    config_mapping = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    base = NeuroOnlineConfig.from_mapping(
        config_mapping.get("online_adaptation", {}) or {}
    )
    config = replace(
        base,
        enabled=True,
        offline_random_seed=args.seed,
        offline_epochs=args.max_epochs,
        offline_update_policy=str(candidate["update_policy"]),
        offline_learning_rate=float(candidate["head_learning_rate"]),
        offline_backbone_learning_rate=(
            None
            if candidate.get("backbone_learning_rate") is None
            else float(candidate["backbone_learning_rate"])
        ),
        offline_batch_size=int(candidate["batch_size"]),
        offline_mask_ratio=float(candidate["mask_ratio"]),
        offline_consistency_weight=float(candidate["consistency_weight"]),
    )
    n_classes = int(config_mapping.get("n_classes", int(labels.max()) + 1))
    adapter = _build_adapter(
        config=config,
        n_chans=windows.shape[1],
        n_times=windows.shape[2],
        n_classes=n_classes,
        sfreq=sfreq,
    )
    started = time.perf_counter()
    fit_metrics = adapter.fit_full(windows, labels, epochs=selected_epochs)
    probabilities = adapter.predict_proba(windows)
    trial_labels, trial_probabilities = _aggregate_trial_probabilities(
        labels,
        probabilities,
        groups,
    )
    model_path = output_dir / f"cbramod_final_seed{args.seed}.pt"
    adapter.save(model_path)
    report = {
        "identity": identity,
        "method": (
            "selected configuration retrained once on all 2026-07-28 calibration "
            "trials with a predeclared seed; metrics are resubstitution diagnostics"
        ),
        "dataset": {
            "windows": int(windows.shape[0]),
            "trials": int(np.unique(groups).size),
            "shape": list(windows.shape),
            "sfreq": sfreq,
        },
        "fit_metrics": fit_metrics,
        "resubstitution_window_metrics": classification_metrics(
            labels,
            probabilities,
            n_classes=n_classes,
        ),
        "resubstitution_trial_metrics": classification_metrics(
            trial_labels,
            trial_probabilities,
            n_classes=n_classes,
        ),
        "artifacts": {
            "model": artifact(model_path),
            "neuroonline_sidecar": artifact(Path(f"{model_path}.neuroonline.pt")),
        },
        "duration_sec": float(time.perf_counter() - started),
    }
    _write_json(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--offline-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--feature-key", default="processed_windows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the selected candidate's median CV best epoch.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=50,
        help="Keep the cosine scheduler horizon equal to the CV training horizon.",
    )
    return parser.parse_args()


def main() -> None:
    report = run(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
