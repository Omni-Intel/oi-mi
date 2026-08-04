from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tools.merge_neuroonline_online_search_shards import merge_reports


def _candidate(index: int, balanced_accuracy: float) -> dict[str, object]:
    return {
        "candidate_index": index,
        "config": {
            "learning_rate": 1e-4 * (index + 1),
            "epochs": 1,
            "update_batch_size": 16,
            "mask_ratio": 0.7,
            "consistency_weight": 1.0,
        },
        "validation": {
            "class_collapse": False,
            "metrics": {
                "balanced_accuracy": balanced_accuracy,
                "accuracy": balanced_accuracy,
                "worst_observed_class_recall": balanced_accuracy,
            },
        },
        "search_prefix": {"metrics": {"balanced_accuracy": balanced_accuracy}},
    }


def _report(candidate: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": 4,
        "method": "causal",
        "source_recording": "recording",
        "source_checkpoint": {"model": {"sha256": "model"}},
        "stream": {"samples": 470},
        "fixed": {"random_seed": 17},
        "search_space": {"total_candidates": 2},
        "candidates": [candidate],
    }


class OnlineSearchShardMergeTests(unittest.TestCase):
    def test_merges_disjoint_candidates_and_selects_best(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [root / "shard_000.json", root / "shard_001.json"]
            for path, candidate in zip(
                paths,
                [_candidate(0, 0.4), _candidate(1, 0.6)],
                strict=True,
            ):
                path.write_text(json.dumps(_report(candidate)), encoding="utf-8")

            merged = merge_reports(paths)

        self.assertTrue(merged["search_complete"])
        self.assertEqual(merged["completed_candidates"], 2)
        self.assertEqual(merged["selected"]["learning_rate"], 2e-4)


if __name__ == "__main__":
    unittest.main()
