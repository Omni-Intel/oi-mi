"""Search online NeuroOnline optimizer settings with causal recorded replay."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields, replace
import gc
from itertools import product
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


def _one_window_per_scene(data: RealtimeData) -> RealtimeData:
    _, first_indices = np.unique(data.scene_indices, return_index=True)
    selected = np.sort(first_indices)
    values: dict[str, Any] = {}
    for field in fields(RealtimeData):
        value = getattr(data, field.name)
        values[field.name] = value if field.name == "chunk_artifacts" else value[selected]
    return RealtimeData(**values)


def _save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _score(result: dict[str, Any]) -> tuple[float, float, float, int]:
    if result.get("failed") or result["validation"]["class_collapse"]:
        return (-1.0, -1.0, -1.0, -10_000)
    metrics = result["validation"]["metrics"]
    return (
        float(metrics["balanced_accuracy"]),
        float(metrics["accuracy"]),
        float(metrics["worst_observed_class_recall"]),
        -int(result["config"]["epochs"]),
    )


def _resume_identity(report: dict[str, Any]) -> dict[str, Any]:
    """Return the immutable inputs that make cached candidate results valid."""

    return {
        "schema_version": report.get("schema_version"),
        "method": report.get("method"),
        "source_recording": report.get("source_recording"),
        "source_checkpoint": report.get("source_checkpoint"),
        "stream": report.get("stream"),
        "fixed": report.get("fixed"),
        "search_space": report.get("search_space"),
    }


def _validate_resume_report(
    previous: dict[str, Any],
    current: dict[str, Any],
    *,
    output: Path,
) -> None:
    if _resume_identity(previous) != _resume_identity(current):
        raise ValueError(
            "Existing online-search output was produced from different inputs or "
            f"fixed parameters: {output}. Use a new output path."
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    checkpoint_before = artifact(checkpoint)
    sidecar_before = artifact(Path(f"{checkpoint}.neuroonline.pt"))
    base_config, _ = load_checkpoint_config(checkpoint)
    base_config = replace(
        base_config,
        enabled=True,
        history_threshold=64,
        update_stride=64,
        recent_samples=320,
    )

    if not args.learning_rates:
        raise ValueError("At least one learning rate is required.")
    if not args.epochs_grid or any(value < 1 for value in args.epochs_grid):
        raise ValueError("Epoch counts must be positive integers.")
    if not args.batch_sizes or any(value < 1 for value in args.batch_sizes):
        raise ValueError("Batch sizes must be positive integers.")
    if not args.mask_ratios or any(not 0.0 <= value <= 1.0 for value in args.mask_ratios):
        raise ValueError("Mask ratios must be in [0, 1].")
    if not args.lambda_grid or any(value < 0.0 for value in args.lambda_grid):
        raise ValueError("Consistency weights (lambda) must be non-negative.")

    search_candidates = list(
        product(
            args.learning_rates,
            args.epochs_grid,
            args.batch_sizes,
            args.mask_ratios,
            args.lambda_grid,
        )
    )
    start_index = max(int(args.start_index), 0)
    stop_index = (
        len(search_candidates)
        if args.stop_index is None
        else min(max(int(args.stop_index), start_index), len(search_candidates))
    )

    data = _one_window_per_scene(load_committed_data(args.recording.resolve()))
    processed, quality = preprocess_windows(data, sfreq=args.sfreq)
    if int(quality["rejected_windows"]) != 0:
        raise ValueError("The one-window-per-scene stream contains rejected windows.")
    selection_end = min(int(args.selection_end), len(data.labels))
    selection_start = int(args.selection_start)
    if not base_config.history_threshold <= selection_start < selection_end:
        raise ValueError(
            "Selection indices must satisfy history_threshold <= start < end."
        )
    if selection_end >= len(data.labels):
        raise ValueError("At least one final temporal holdout sample is required.")

    report: dict[str, Any] = {
        "schema_version": 4,
        "method": (
            "causal_predict_then_update_temporal_prefix_search_with_sealed_final_holdout"
        ),
        "source_recording": str(args.recording.resolve()),
        "source_checkpoint": {
            "model": checkpoint_before,
            "neuroonline_sidecar": sidecar_before,
        },
        "stream": {
            "selection": "first_committed_window_per_scene",
            "samples": int(len(data.labels)),
            "scenes": int(np.unique(data.scene_indices).size),
            "class_counts": np.bincount(data.labels, minlength=args.n_classes).astype(int).tolist(),
            "selection_start_index": selection_start,
            "selection_end_index_exclusive": selection_end,
            "selection_samples": selection_end - selection_start,
            "final_holdout_start_index": selection_end,
            "final_holdout_samples": int(len(data.labels) - selection_end),
            "source_chunks": data.chunk_artifacts,
        },
        "fixed": {
            "history_threshold": 64,
            "update_stride": 64,
            "recent_samples": 320,
            "weight_decay": base_config.weight_decay,
            "label_smoothing": base_config.label_smoothing,
            "random_seed": base_config.random_seed,
        },
        "search_space": {
            "learning_rates": args.learning_rates,
            "epochs": args.epochs_grid,
            "batch_sizes": args.batch_sizes,
            "mask_ratios": args.mask_ratios,
            "lambda_grid": args.lambda_grid,
            "masking": (
                "NeuroOnline source-compatible elementwise Bernoulli time-sample and "
                "rFFT-bin masks, generated once when each stream sample is accepted "
                "and reused by later updates"
            ),
            "total_candidates": len(search_candidates),
        },
        "requested_candidate_range": [start_index, stop_index],
        "selection_rule": (
            "reject class collapse; maximize sealed-prefix prequential balanced accuracy, "
            "then accuracy, worst-class recall, and prefer fewer epochs"
        ),
        "candidates": [],
    }
    seed_everything(base_config.random_seed)
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
    report["no_online_update_baseline"] = {
        "selection": classification_metrics(
            data.labels[selection_start:selection_end],
            baseline_probabilities[selection_start:selection_end],
            n_classes=args.n_classes,
        ),
        "final_holdout": classification_metrics(
            data.labels[selection_end:],
            baseline_probabilities[selection_end:],
            n_classes=args.n_classes,
        ),
        "full_stream": classification_metrics(
            data.labels,
            baseline_probabilities,
            n_classes=args.n_classes,
        ),
    }
    del baseline_adapter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    cache: dict[tuple[float, int, int, float, float], dict[str, Any]] = {}
    if output.exists():
        previous = json.loads(output.read_text(encoding="utf-8"))
        _validate_resume_report(previous, report, output=output)
        for candidate in previous.get("candidates", []):
            candidate_config = candidate["config"]
            key = (
                float(candidate_config["learning_rate"]),
                int(candidate_config["epochs"]),
                int(candidate_config["update_batch_size"]),
                float(candidate_config["mask_ratio"]),
                float(candidate_config["consistency_weight"]),
            )
            cache[key] = candidate
            report["candidates"].append(candidate)
        report["resumed_candidates"] = len(cache)

    def evaluate(
        learning_rate: float,
        epochs: int,
        batch_size: int,
        mask_ratio: float,
        consistency_weight: float,
        candidate_index: int,
    ) -> dict[str, Any]:
        key = (
            float(learning_rate),
            int(epochs),
            int(batch_size),
            float(mask_ratio),
            float(consistency_weight),
        )
        if key in cache:
            return cache[key]
        config = replace(
            base_config,
            learning_rate=key[0],
            epochs=key[1],
            update_batch_size=key[2],
            mask_ratio=key[3],
            consistency_weight=key[4],
        )
        candidate_started = time.perf_counter()
        result: dict[str, Any] = {
            "candidate_index": int(candidate_index),
            "config": {
                "learning_rate": key[0],
                "epochs": key[1],
                "update_batch_size": key[2],
                "mask_ratio": key[3],
                "consistency_weight": key[4],
            },
        }
        try:
            seed_everything(config.random_seed)
            adapter = build_adapter(
                checkpoint,
                model_name=args.model_name,
                n_chans=processed.shape[1],
                n_times=processed.shape[2],
                n_classes=args.n_classes,
                sfreq=args.sfreq,
                config=config,
            )
            probabilities, _, updates = causal_replay(
                adapter,
                processed[:selection_end],
                data.labels[:selection_end],
                data.scene_indices[:selection_end],
                config=config,
                n_classes=args.n_classes,
            )
            validation_metrics = classification_metrics(
                data.labels[selection_start:selection_end],
                probabilities[selection_start:selection_end],
                n_classes=args.n_classes,
            )
            prefix_metrics = classification_metrics(
                data.labels[:selection_end],
                probabilities,
                n_classes=args.n_classes,
            )
            result.update(
                {
                    "updates": len(updates),
                    "update_triggers": [
                        int(entry["trigger_seen_labeled_windows"]) for entry in updates
                    ],
                    "mean_update_duration_sec": float(
                        np.mean([entry["duration_sec"] for entry in updates])
                    ),
                    "final_loss": float(updates[-1]["loss"]),
                    "validation": {
                        "metrics": validation_metrics,
                        "class_collapse": bool(
                            not validation_metrics["all_classes_predicted"]
                            or validation_metrics["worst_observed_class_recall"] <= 0.0
                        ),
                    },
                    "search_prefix": {"metrics": prefix_metrics},
                }
            )
        except Exception as exc:
            result.update({"failed": True, "error": f"{type(exc).__name__}: {exc}"})
        finally:
            if "adapter" in locals():
                del adapter
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        result["duration_sec"] = float(time.perf_counter() - candidate_started)
        cache[key] = result
        report["candidates"].append(result)
        _save(output, report)
        metrics = result.get("validation", {}).get("metrics", {})
        print(
            f"candidate={candidate_index + 1}/{len(search_candidates)} "
            f"lr={key[0]:g} epochs={key[1]} batch={key[2]} "
            f"mask={key[3]:g} lambda={key[4]:g} "
            f"acc={metrics.get('accuracy', -1):.4f} "
            f"bacc={metrics.get('balanced_accuracy', -1):.4f} "
            f"collapse={result.get('validation', {}).get('class_collapse', True)}",
            flush=True,
        )
        return result

    for candidate_index in range(start_index, stop_index):
        evaluate(*search_candidates[candidate_index], candidate_index)

    if not report["candidates"]:
        raise ValueError("The requested candidate range is empty and no cached results exist.")
    successful_candidates = [
        candidate
        for candidate in report["candidates"]
        if not candidate.get("failed") and "validation" in candidate
    ]
    if not successful_candidates:
        raise RuntimeError("Every evaluated online-search candidate failed.")
    selected = max(successful_candidates, key=_score)
    report["selected"] = {
        **selected["config"],
        "validation": selected["validation"],
        "search_prefix": selected["search_prefix"],
    }
    report["completed_candidates"] = len(cache)
    report["search_complete"] = len(cache) == len(search_candidates)
    if report["search_complete"]:
        selected_config = replace(
            base_config,
            learning_rate=float(selected["config"]["learning_rate"]),
            epochs=int(selected["config"]["epochs"]),
            update_batch_size=int(selected["config"]["update_batch_size"]),
            mask_ratio=float(selected["config"]["mask_ratio"]),
            consistency_weight=float(selected["config"]["consistency_weight"]),
        )
        seed_everything(selected_config.random_seed)
        final_adapter = build_adapter(
            checkpoint,
            model_name=args.model_name,
            n_chans=processed.shape[1],
            n_times=processed.shape[2],
            n_classes=args.n_classes,
            sfreq=args.sfreq,
            config=selected_config,
        )
        final_probabilities, _, final_updates = causal_replay(
            final_adapter,
            processed,
            data.labels,
            data.scene_indices,
            config=selected_config,
            n_classes=args.n_classes,
        )
        report["selected_final_replay"] = {
            "config": selected["config"],
            "updates": len(final_updates),
            "final_holdout": classification_metrics(
                data.labels[selection_end:],
                final_probabilities[selection_end:],
                n_classes=args.n_classes,
            ),
            "full_stream": classification_metrics(
                data.labels,
                final_probabilities,
                n_classes=args.n_classes,
            ),
        }
        del final_adapter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    report["input_checkpoint_unchanged"] = bool(
        checkpoint_before == artifact(checkpoint)
        and sidecar_before == artifact(Path(f"{checkpoint}.neuroonline.pt"))
    )
    report["duration_sec"] = float(time.perf_counter() - started)
    _save(output, report)
    return report


def _float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",")]


def _int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="cbramod")
    parser.add_argument("--sfreq", type=float, default=200.0)
    parser.add_argument("--n-classes", type=int, default=3)
    parser.add_argument(
        "--learning-rates",
        type=_float_list,
        default=[3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
    )
    parser.add_argument("--epochs-grid", type=_int_list, default=[1, 3, 5])
    parser.add_argument("--batch-sizes", type=_int_list, default=[8, 16, 32, 64])
    parser.add_argument(
        "--mask-ratios",
        type=_float_list,
        default=[0.1, 0.3, 0.5, 0.7, 0.9],
    )
    parser.add_argument(
        "--lambda-grid",
        type=_float_list,
        default=[0.1, 0.25, 0.5, 1.0, 2.0],
        help="Paper lambda grid for the mean two-view consistency loss.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stop-index", type=int, default=None)
    parser.add_argument("--selection-start", type=int, default=160)
    parser.add_argument("--selection-end", type=int, default=320)
    return parser


def main() -> None:
    report = run(build_parser().parse_args())
    print(json.dumps(report["selected"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
