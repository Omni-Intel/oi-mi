"""Search CBraMod updates by evaluating the final model on the full stream."""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
import gc
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.simulate_neuroonline_realtime import (  # noqa: E402
    RealtimeData,
    artifact,
    build_adapter,
    causal_replay,
    classification_metrics,
    load_checkpoint_config,
    load_committed_data,
    preprocess_windows,
    seed_everything,
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _one_window_per_scene(data: RealtimeData) -> RealtimeData:
    _, first_indices = np.unique(data.scene_indices, return_index=True)
    selected = np.sort(first_indices)
    values: dict[str, Any] = {}
    for field in fields(RealtimeData):
        value = getattr(data, field.name)
        values[field.name] = value if field.name == "chunk_artifacts" else value[selected]
    return RealtimeData(**values)


def _load_manifest(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates") if isinstance(payload, dict) else payload
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Online candidate manifest must contain a non-empty candidates list.")
    metadata = payload if isinstance(payload, dict) else {}
    return [dict(candidate) for candidate in candidates], metadata


def _load_candidates(path: Path) -> list[dict[str, Any]]:
    candidates, _ = _load_manifest(path)
    return candidates


def _validate_checkpoint_binding(
    manifest: dict[str, Any],
    checkpoint_artifacts: dict[str, Any],
) -> None:
    expected_checkpoint = manifest.get("source_checkpoint")
    if expected_checkpoint is not None and expected_checkpoint != checkpoint_artifacts:
        raise ValueError(
            "The supplied online checkpoint does not match the final offline "
            "checkpoint bound to the candidate manifest."
        )


def _compatible_legacy_mechanics(manifest: dict[str, Any]) -> tuple[int, ...]:
    provenance = manifest.get("source_checkpoint_provenance", {})
    if provenance.get("contract") == "legacy_rank1_summary_hash_verified":
        return (2,)
    return ()


def _paired_metrics(
    truth: np.ndarray,
    candidate_probabilities: np.ndarray,
    baseline_probabilities: np.ndarray,
    *,
    n_classes: int,
) -> dict[str, Any]:
    candidate = classification_metrics(
        truth,
        candidate_probabilities,
        n_classes=n_classes,
    )
    baseline = classification_metrics(
        truth,
        baseline_probabilities,
        n_classes=n_classes,
    )
    return {
        "candidate": candidate,
        "fixed_baseline": baseline,
        "delta_accuracy": float(candidate["accuracy"] - baseline["accuracy"]),
        "delta_balanced_accuracy": float(
            candidate["balanced_accuracy"] - baseline["balanced_accuracy"]
        ),
    }


def _ranking_score(report: dict[str, Any]) -> tuple[float, ...]:
    summary = report["summary"]
    return (
        -float(summary["collapsed_runs"]),
        float(summary["mean_delta_balanced_accuracy"]),
        float(summary["minimum_delta_balanced_accuracy"]),
        -float(summary["std_delta_balanced_accuracy"]),
        float(summary["mean_worst_class_recall"]),
        -float(report.get("identity", {}).get("candidate_index", 0)),
    )


def evaluate_candidate(args: argparse.Namespace) -> dict[str, Any]:
    candidates, manifest = _load_manifest(args.candidate_manifest.resolve())
    if not 0 <= args.candidate_index < len(candidates):
        raise ValueError(f"candidate-index must be in [0, {len(candidates)}).")
    candidate = candidates[args.candidate_index]
    checkpoint = args.checkpoint.resolve()
    recording = args.recording.resolve()
    output = args.output.resolve() / f"candidate_{args.candidate_index:03d}.json"

    data = _one_window_per_scene(load_committed_data(recording))
    processed, quality = preprocess_windows(data, sfreq=args.sfreq)
    if int(quality["rejected_windows"]) != 0:
        raise ValueError("The one-window-per-scene stream contains rejected windows.")
    checkpoint_artifacts = {
        "model": artifact(checkpoint),
        "neuroonline_sidecar": artifact(Path(f"{checkpoint}.neuroonline.pt")),
    }
    _validate_checkpoint_binding(manifest, checkpoint_artifacts)
    identity = {
        "schema_version": 2,
        "recording": str(recording),
        "source_chunks": data.chunk_artifacts,
        "checkpoint": checkpoint_artifacts,
        "candidate_index": args.candidate_index,
        "candidate": candidate,
        "seeds": args.seeds,
        "evaluation_protocol": "final_model_full_stream_resubstitution",
        "report_block_size": args.report_block_size,
        "sfreq": args.sfreq,
        "n_classes": args.n_classes,
    }
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("identity") != identity:
            raise ValueError(f"Existing candidate report has incompatible inputs: {output}")
        print(f"REUSED {output}", flush=True)
        return existing

    base_config, _ = load_checkpoint_config(
        checkpoint,
        compatible_legacy_versions=_compatible_legacy_mechanics(manifest),
    )
    base_config = replace(
        base_config,
        enabled=True,
        update_stride=64,
        recent_samples=320,
    )
    baseline_adapter = build_adapter(
        checkpoint,
        model_name=args.model_name,
        n_chans=processed.shape[1],
        n_times=processed.shape[2],
        n_classes=args.n_classes,
        sfreq=args.sfreq,
        config=base_config,
    )
    baseline_probabilities = baseline_adapter.predict_proba(processed)
    del baseline_adapter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    started = time.perf_counter()
    runs: list[dict[str, Any]] = []
    for seed in args.seeds:
        config = replace(
            base_config,
            update_policy="full",
            learning_rate=float(candidate["learning_rate"]),
            backbone_learning_rate=None,
            history_threshold=int(candidate["history_threshold"]),
            epochs=int(candidate["epochs"]),
            update_batch_size=int(candidate["batch_size"]),
            mask_ratio=float(candidate["mask_ratio"]),
            consistency_weight=float(candidate["consistency_weight"]),
            random_seed=int(seed),
        )
        seed_everything(seed)
        adapter = build_adapter(
            checkpoint,
            model_name=args.model_name,
            n_chans=processed.shape[1],
            n_times=processed.shape[2],
            n_classes=args.n_classes,
            sfreq=args.sfreq,
            config=config,
        )
        run_started = time.perf_counter()
        prequential_probabilities, revisions, updates = causal_replay(
            adapter,
            processed,
            data.labels,
            data.scene_indices,
            config=config,
            n_classes=args.n_classes,
        )
        final_probabilities = adapter.predict_proba(processed)
        paired = _paired_metrics(
            data.labels,
            final_probabilities,
            baseline_probabilities,
            n_classes=args.n_classes,
        )
        blocks: list[dict[str, Any]] = []
        for block_start in range(0, len(data.labels), args.report_block_size):
            block_stop = min(block_start + args.report_block_size, len(data.labels))
            blocks.append(
                {
                    "start_index": block_start,
                    "stop_index_exclusive": block_stop,
                    "metrics": _paired_metrics(
                        data.labels[block_start:block_stop],
                        final_probabilities[block_start:block_stop],
                        baseline_probabilities[block_start:block_stop],
                        n_classes=args.n_classes,
                    ),
                }
            )
        candidate_metrics = paired["candidate"]
        collapsed = bool(
            not candidate_metrics["all_classes_predicted"]
            or candidate_metrics["worst_observed_class_recall"] <= 0.0
        )
        run = {
            "seed": int(seed),
            "class_collapse": collapsed,
            "updates": len(updates),
            "update_triggers": [
                int(entry["trigger_seen_labeled_windows"]) for entry in updates
            ],
            "update_losses": [float(entry["loss"]) for entry in updates],
            "mean_update_duration_sec": float(
                np.mean([entry["duration_sec"] for entry in updates])
            ),
            "paired_post_update": paired,
            "post_adaptation_blocks": blocks,
            "prequential_diagnostic": classification_metrics(
                data.labels,
                prequential_probabilities,
                n_classes=args.n_classes,
            ),
            "prequential_model_revisions": revisions.astype(int).tolist(),
            "duration_sec": float(time.perf_counter() - run_started),
        }
        runs.append(run)
        print(
            f"candidate={args.candidate_index + 1}/{len(candidates)} seed={seed} "
            f"bacc={candidate_metrics['balanced_accuracy']:.4f} "
            f"delta={paired['delta_balanced_accuracy']:+.4f} "
            f"collapse={collapsed}",
            flush=True,
        )
        del adapter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    deltas = np.asarray(
        [run["paired_post_update"]["delta_balanced_accuracy"] for run in runs],
        dtype=np.float64,
    )
    baccs = np.asarray(
        [
            run["paired_post_update"]["candidate"]["balanced_accuracy"]
            for run in runs
        ],
        dtype=np.float64,
    )
    worst_recalls = np.asarray(
        [
            run["paired_post_update"]["candidate"]["worst_observed_class_recall"]
            for run in runs
        ],
        dtype=np.float64,
    )
    report = {
        "identity": identity,
        "stream": {
            "samples": int(len(data.labels)),
            "scenes": int(np.unique(data.scene_indices).size),
            "shape": list(processed.shape),
            "class_counts": np.bincount(
                data.labels,
                minlength=args.n_classes,
            ).astype(int).tolist(),
            "training_samples_through_last_update": int(
                max(
                    trigger
                    for run in runs
                    for trigger in run["update_triggers"]
                )
                if any(run["update_triggers"] for run in runs)
                else 0
            ),
            "unupdated_tail_samples": int(len(data.labels) % 64),
            "evaluation_samples": int(len(data.labels)),
            "evaluation_protocol": "final_model_full_stream_resubstitution",
        },
        "fixed_baseline": classification_metrics(
            data.labels,
            baseline_probabilities,
            n_classes=args.n_classes,
        ),
        "summary": {
            "runs": len(runs),
            "collapsed_runs": sum(run["class_collapse"] for run in runs),
            "mean_balanced_accuracy": float(np.mean(baccs)),
            "mean_delta_balanced_accuracy": float(np.mean(deltas)),
            "std_delta_balanced_accuracy": float(np.std(deltas)),
            "minimum_delta_balanced_accuracy": float(np.min(deltas)),
            "mean_worst_class_recall": float(np.mean(worst_recalls)),
        },
        "runs": runs,
        "input_checkpoint_unchanged": bool(
            checkpoint_artifacts["model"] == artifact(checkpoint)
            and checkpoint_artifacts["neuroonline_sidecar"]
            == artifact(Path(f"{checkpoint}.neuroonline.pt"))
        ),
        "duration_sec": float(time.perf_counter() - started),
    }
    _write_json(output, report)
    return report


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    candidates = _load_candidates(args.candidate_manifest.resolve())
    reports = []
    for index in range(len(candidates)):
        path = args.output.resolve() / f"candidate_{index:03d}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing candidate report: {path}")
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    source_checkpoints = [
        report.get("identity", {}).get("checkpoint") for report in reports
    ]
    if not source_checkpoints or any(
        checkpoint != source_checkpoints[0] for checkpoint in source_checkpoints
    ):
        raise ValueError("Online candidate reports do not share one immutable checkpoint.")
    ranked = sorted(reports, key=_ranking_score, reverse=True)
    payload = {
        "schema_version": 1,
        "method": "multi-seed final-model full-stream resubstitution after causal updates",
        "source_checkpoint": source_checkpoints[0],
        "expected_candidates": len(candidates),
        "completed_candidates": len(reports),
        "ranking_rule": (
            "minimize collapsed seeds; maximize mean paired bACC improvement; "
            "maximize worst-seed improvement; minimize improvement std; "
            "maximize mean worst-class recall; break exact ties by the lowest "
            "predeclared candidate index"
        ),
        "ranked": [
            {
                "rank": rank,
                "candidate_index": report["identity"]["candidate_index"],
                "candidate": report["identity"]["candidate"],
                "summary": report["summary"],
            }
            for rank, report in enumerate(ranked, 1)
        ],
    }
    _write_json(args.output.resolve() / "summary.json", payload)
    return payload


def _comma_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-index", type=int, default=None)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--model-name", default="cbramod")
    parser.add_argument("--sfreq", type=float, default=200.0)
    parser.add_argument("--n-classes", type=int, default=3)
    parser.add_argument("--seeds", type=_comma_ints, default=[17, 42, 2026])
    parser.add_argument("--report-block-size", type=int, default=64)
    args = parser.parse_args()
    if args.summarize == (args.candidate_index is not None):
        parser.error("Choose exactly one of --summarize or --candidate-index.")
    return args


def main() -> None:
    args = parse_args()
    payload = summarize(args) if args.summarize else evaluate_candidate(args)
    print(json.dumps(payload.get("summary", payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
