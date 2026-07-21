from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.stream_writer import StreamWriter


class StreamWriterTelemetryTests(unittest.TestCase):
    def test_persists_raw_predictions_revisions_and_adaptation_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            writer = StreamWriter(root, chunk_size=2)
            writer.start({"mode": "realtime"})
            writer.put(
                np.zeros((2, 8), dtype=np.float32),
                y_true=1,
                y_pred=-1,
                confidence=0.4,
                raw_pred=1,
                model_revision=2,
                label_event_id="cue-000001",
            )
            writer.stop()
            writer.update_manifest({"online_adaptation": {"update_count": 2}})

            with np.load(root / "chunks" / "chunk_000000.npz") as payload:
                self.assertEqual(payload["predictions_raw"].tolist(), [1])
                self.assertEqual(payload["model_revisions"].tolist(), [2])
                self.assertEqual(payload["label_event_ids"].tolist(), ["cue-000001"])
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["online_adaptation"]["update_count"], 2)


if __name__ == "__main__":
    unittest.main()
