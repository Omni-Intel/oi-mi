"""Search online NeuroOnline optimizer settings with causal recorded replay."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields, replace
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

    data = _one_window_per_scene(load_committed_data(args.recording.resolve()))
    processed, quality = preprocess_windows(data, sfreq=args.sfreq)
    if int(quality["rejected_windows"]) != 0:
        raise ValueError("The one-window-per-scene stream contains rejected windows.")
    validation_start = len(data.labels) // 2
    if validation_start < base_config.history_threshold:
        raise ValueError("The chronological validation split is too short.")

    report: dict[str, Any] = {
        "schema_version": 2,
        "method": "causal_predict_then_update_chronological_online_hyperparameter_search",
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
            "validation_start_index": int(validation_start),
            "validation_samples": int(len(data.labels) - validation_start),
            "source_chunks": data.chunk_artifacts,
        },
        "fixed": {
            "history_threshold": 64,
            "update_stride": 64,
            "recent_samples": 320,
            "mask_ratio": base_config.mask_ratio,
            "consistency_weight": base_config.consistency_weight,
            "weight_decay": base_config.weight_decay,
            "label_smoothing": base_config.label_smoothing,
            "random_seed": base_config.random_seed,
        },
        "selection_rule": (
            "reject class collapse; maximize second-half prequential balanced accuracy, "
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
        "validation": classification_metrics(
            data.labels[validation_start:],
            baseline_probabilities[validation_start:],
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
    cache: dict[tuple[float, int, int], dict[str, Any]] = {}
    if output.exists():
        previous = json.loads(output.read_text(encoding="utf-8"))
        _validate_resume_report(previous, report, output=output)
        for candidate in previous.get("candidates", []):
            candidate_config = candidate["config"]
            key = (
                float(candidate_config["learning_rate"]),
                int(candidate_config["epochs"]),
                int(candidate_config["update_batch_size"]),
            )
            cache[key] = candidate
            report["candidates"].append(candidate)
        report["resumed_candidates"] = len(cache)

    def evaluate(learning_rate: float, epochs: int, batch_size: int, stage: str) -> dict[str, Any]:
        key = (float(learning_rate), int(epochs), int(batch_size))
        if key in cache:
            existing = cache[key]
            existing.setdefault("stages", []).append(stage)
            return existing
        config = replace(
            base_config,
            learning_rate=key[0],
            epochs=key[1],
            update_batch_size=key[2],
        )
        candidate_started = time.perf_counter()
        result: dict[str, Any] = {
            "stages": [stage],
            "config": {
                "learning_rate": key[0],
                "epochs": key[1],
                "update_batch_size": key[2],
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
                processed,
                data.labels,
                data.scene_indices,
                config=config,
                n_classes=args.n_classes,
            )
            validation_metrics = classification_metrics(
                data.labels[validation_start:],
                probabilities[validation_start:],
                n_classes=args.n_classes,
            )
            full_metrics = classification_metrics(
                data.labels,
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
                    "full_stream": {"metrics": full_metrics},
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
            f"{stage} lr={key[0]:g} epochs={key[1]} batch={key[2]} "
            f"acc={metrics.get('accuracy', -1):.4f} "
            f"bacc={metrics.get('balanced_accuracy', -1):.4f} "
            f"collapse={result.get('validation', {}).get('class_collapse', True)}",
            flush=True,
        )
        return result

    stage1 = [
        evaluate(lr, epochs, 16, "learning_rate_epochs")
        for lr in args.learning_rates
        for epochs in args.epochs_grid
    ]
    stage1_best = max(stage1, key=_score)
    best_lr = float(stage1_best["config"]["learning_rate"])
    best_epochs = int(stage1_best["config"]["epochs"])
    stage2 = [
        evaluate(best_lr, best_epochs, batch_size, "batch_size")
        for batch_size in args.batch_sizes
    ]
    selected = max(stage2, key=_score)
    report["stage1_best"] = stage1_best["config"]
    report["selected"] = {
        **selected["config"],
        "validation": selected["validation"],
        "full_stream": selected["full_stream"],
    }
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
    parser.add_argument("--model-name", default="shallowconvnet")
    parser.add_argument("--sfreq", type=float, default=200.0)
    parser.add_argument("--n-classes", type=int, default=3)
    parser.add_argument(
        "--learning-rates",
        type=_float_list,
        default=[3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
    )
    parser.add_argument("--epochs-grid", type=_int_list, default=[1, 3])
    parser.add_argument("--batch-sizes", type=_int_list, default=[8, 16, 32, 64])
    return parser


def main() -> None:
    report = run(build_parser().parse_args())
    print(json.dumps(report["selected"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
