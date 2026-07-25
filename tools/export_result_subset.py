"""Export a clean experiment result containing only selected final methods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    summary = load_json(args.source / "final_summary.json")
    missing = [method for method in args.methods if method not in summary["final"]]
    if missing:
        raise KeyError(f"Methods not found in source summary: {missing}")
    summary["config"]["final_methods"] = args.methods
    summary["config"]["exported_from"] = str(args.source.resolve())
    summary["final"] = {
        method: summary["final"][method]
        for method in args.methods
    }
    save_json(args.output / "final_summary.json", summary)

    selection = load_json(args.source / "selection_summary.json")
    save_json(args.output / "selection_summary.json", selection)

    experiment_config = load_json(args.source / "experiment_config.json")
    experiment_config["final_methods"] = args.methods
    experiment_config["exported_from"] = str(args.source.resolve())
    save_json(args.output / "experiment_config.json", experiment_config)

    for method in args.methods:
        source_dir = args.source / "final" / method
        if source_dir.exists():
            shutil.copytree(source_dir, args.output / "final" / method)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["static", "standard_online", "neuroonline"],
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
