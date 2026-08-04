"""Compare recorded online predictions with frozen revision-0 replays.

The online arm is never retrained or reconstructed: probabilities, active
model revisions, losses, and update timing come from the formal-session log.
Only the initial model is replayed, with several MC-dropout seed schedules to
quantify randomness in the counterfactual frozen baseline.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.evaluate_paired_neuroonline import (  # noqa: E402
    _block_bootstrap,
    _frozen_baseline,
    _metrics,
    _paired_summary,
    _save_json,
)
from tools.simulate_neuroonline_realtime import (  # noqa: E402
    artifact,
    load_checkpoint_config,
    load_committed_data,
    preprocess_windows,
)


def _load_recorded_outputs(recording: Path, data: Any) -> dict[str, np.ndarray]:
    keys = (
        "labels_true",
        "probabilities",
        "predictions_raw",
        "labels_pred",
        "model_revisions",
        "scene_indices",
        "confidences",
        "uncertainties",
        "quality_accepted",
    )
    values = {
        key: np.empty(
            (data.labels.size, 3) if key == "probabilities" else data.labels.size,
            dtype=(np.float32 if key in {"probabilities", "confidences", "uncertainties"} else np.int64),
        )
        for key in keys
    }
    values["quality_accepted"] = np.empty(data.labels.size, dtype=bool)
    for chunk_name in np.unique(data.source_chunks):
        indices = np.flatnonzero(data.source_chunks == chunk_name)
        rows = data.source_rows[indices]
        with np.load(recording / "chunks" / str(chunk_name), allow_pickle=False) as payload:
            missing = sorted(set(keys) - set(payload.files))
            if missing:
                raise ValueError(f"{chunk_name} is missing: {', '.join(missing)}")
            for key in keys:
                values[key][indices] = payload[key][rows]
    if not np.array_equal(values["labels_true"], data.labels):
        raise ValueError("Recorded labels do not match the committed replay stream.")
    if not np.array_equal(values["scene_indices"], data.scene_indices):
        raise ValueError("Recorded scenes do not match the committed replay stream.")
    if not np.array_equal(values["probabilities"].argmax(axis=1), values["predictions_raw"]):
        raise ValueError("Recorded raw predictions do not match argmax(probabilities).")
    if not np.all(np.isfinite(values["probabilities"])):
        raise ValueError("Recorded probabilities contain non-finite values.")
    return values


def _without_predictions(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "predictions"}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output}")
    if args.primary_inference_seed_base not in args.inference_seed_bases:
        raise ValueError("Primary inference seed base must be in --inference-seed-bases.")
    args.output.mkdir(parents=True, exist_ok=True)
    recording = args.recording.resolve()
    checkpoint = args.checkpoint.resolve()
    data = load_committed_data(recording)
    windows, quality = preprocess_windows(data, sfreq=args.sfreq)
    if quality["rejected_windows"]:
        raise ValueError("Committed replay data unexpectedly failed preprocessing.")
    recorded = _load_recorded_outputs(recording, data)
    config, _ = load_checkpoint_config(checkpoint)

    baselines: list[np.ndarray] = []
    for seed_base in args.inference_seed_bases:
        probabilities = _frozen_baseline(
            checkpoint=checkpoint,
            windows=windows,
            config=config,
            inference_seed_base=seed_base,
            mc_dropout_passes=args.mc_dropout_passes,
        )
        baselines.append(probabilities)
        metrics = _metrics(data.labels, probabilities)
        print(
            f"BASELINE {seed_base} accuracy={metrics['accuracy']:.6f} "
            f"bacc={metrics['balanced_accuracy']:.6f} ce={metrics['cross_entropy']:.6f}",
            flush=True,
        )
    baseline_stack = np.stack(baselines)
    primary_index = args.inference_seed_bases.index(args.primary_inference_seed_base)
    primary_baseline = baseline_stack[primary_index]
    recorded_probabilities = recorded["probabilities"]
    post_update = recorded["model_revisions"] > 0

    runs = []
    for seed_base, baseline in zip(args.inference_seed_bases, baselines, strict=True):
        runs.append(
            {
                "inference_seed_base": seed_base,
                "full_stream": _paired_summary(
                    data.labels,
                    baseline,
                    recorded_probabilities,
                ),
                "post_update": _paired_summary(
                    data.labels[post_update],
                    baseline[post_update],
                    recorded_probabilities[post_update],
                ),
            }
        )
    primary = runs[primary_index]
    primary["phases"] = [
        {
            "recorded_model_revision": int(revision),
            **_paired_summary(
                data.labels[recorded["model_revisions"] == revision],
                primary_baseline[recorded["model_revisions"] == revision],
                recorded_probabilities[recorded["model_revisions"] == revision],
            ),
        }
        for revision in np.unique(recorded["model_revisions"])
    ]
    primary["post_update"]["block_bootstrap"] = _block_bootstrap(
        data.labels[post_update],
        primary_baseline[post_update],
        recorded_probabilities[post_update],
        block_length=args.bootstrap_block_length,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )

    delta_keys = ("accuracy", "balanced_accuracy", "macro_f1", "cross_entropy")
    delta_matrix = np.asarray(
        [[run["post_update"]["delta"][key] for key in delta_keys] for run in runs],
        dtype=np.float64,
    )
    manifest = json.loads((recording / "manifest.json").read_text(encoding="utf-8"))
    actual_config = manifest["online_adaptation"]["config"]
    report = {
        "schema_version": 1,
        "method": "recorded_online_log_vs_frozen_revision0_counterfactual_replay",
        "claim_scope": "single_subject_single_session_within_session_counterfactual",
        "important_limit": (
            "The recorded online MC-dropout RNG state is unavailable. Frozen-baseline "
            "replays therefore cannot share dropout masks with the observed online arm."
        ),
        "source_recording": str(recording),
        "source_checkpoint": artifact(checkpoint),
        "source_sidecar": artifact(Path(f"{checkpoint}.neuroonline.pt")),
        "stream": {
            "samples": int(data.labels.size),
            "class_counts": np.bincount(data.labels, minlength=3).astype(int).tolist(),
            "recorded_revision_counts": {
                str(int(revision)): int(np.sum(recorded["model_revisions"] == revision))
                for revision in np.unique(recorded["model_revisions"])
            },
            "causal_order": "recorded_predict_then_reveal_label_then_background_update",
        },
        "recorded_online": {
            "actual_effective_config": actual_config,
            "raw_full_stream": _without_predictions(_metrics(data.labels, recorded_probabilities)),
            "raw_post_update": _without_predictions(
                _metrics(data.labels[post_update], recorded_probabilities[post_update])
            ),
            "manifest_online_adaptation": manifest["online_adaptation"],
            "manifest_scientific_metrics": manifest["scientific_metrics"],
        },
        "frozen_baseline_replay": {
            "config": {
                **asdict(config),
                "mc_dropout_passes": args.mc_dropout_passes,
                "inference_seed_bases": args.inference_seed_bases,
                "primary_inference_seed_base": args.primary_inference_seed_base,
            },
            "primary_run": primary,
            "inference_seed_sensitivity": {
                "seed_schedules": len(runs),
                "delta_definition": "recorded_online_minus_frozen_baseline",
                "delta_keys": list(delta_keys),
                "mean": delta_matrix.mean(axis=0).tolist(),
                "sample_std": delta_matrix.std(axis=0, ddof=1).tolist(),
                "minimum": delta_matrix.min(axis=0).tolist(),
                "maximum": delta_matrix.max(axis=0).tolist(),
                "online_accuracy_better_schedules": int(np.sum(delta_matrix[:, 0] > 0.0)),
                "runs": runs,
            },
        },
    }
    np.savez_compressed(
        args.output / "recorded_vs_frozen_probabilities.npz",
        labels=data.labels,
        scene_indices=data.scene_indices,
        recorded_probabilities=recorded_probabilities,
        recorded_model_revisions=recorded["model_revisions"],
        frozen_probabilities=baseline_stack,
        inference_seed_bases=np.asarray(args.inference_seed_bases, dtype=np.int64),
    )
    _save_json(args.output / "recorded_vs_frozen_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sfreq", type=float, default=200.0)
    parser.add_argument("--mc-dropout-passes", type=int, default=8)
    parser.add_argument("--inference-seed-bases", type=int, nargs="+", required=True)
    parser.add_argument("--primary-inference-seed-base", type=int, default=100000)
    parser.add_argument("--bootstrap-block-length", type=int, default=16)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260728)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
