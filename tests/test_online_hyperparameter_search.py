from __future__ import annotations

import copy
import unittest
from pathlib import Path

from tools.search_neuroonline_online_hyperparams import _validate_resume_report


def _report() -> dict[str, object]:
    return {
        "schema_version": 2,
        "method": "causal",
        "source_recording": "recording-a",
        "source_checkpoint": {
            "model": {"sha256": "model-a"},
            "neuroonline_sidecar": {"sha256": "sidecar-a"},
        },
        "stream": {
            "samples": 128,
            "source_chunks": [{"sha256": "chunk-a"}],
        },
        "fixed": {"random_seed": 2026, "history_threshold": 64},
    }


class OnlineHyperparameterSearchTests(unittest.TestCase):
    def test_resume_accepts_identical_inputs(self) -> None:
        report = _report()

        _validate_resume_report(report, copy.deepcopy(report), output=Path("report.json"))

    def test_resume_rejects_changed_recording_content(self) -> None:
        previous = _report()
        current = copy.deepcopy(previous)
        current["stream"]["source_chunks"][0]["sha256"] = "chunk-b"

        with self.assertRaisesRegex(ValueError, "different inputs"):
            _validate_resume_report(previous, current, output=Path("report.json"))

    def test_resume_rejects_changed_fixed_seed(self) -> None:
        previous = _report()
        current = copy.deepcopy(previous)
        current["fixed"]["random_seed"] = 42

        with self.assertRaisesRegex(ValueError, "different inputs"):
            _validate_resume_report(previous, current, output=Path("report.json"))


if __name__ == "__main__":
    unittest.main()
