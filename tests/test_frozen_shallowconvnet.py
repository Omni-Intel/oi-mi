from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.evaluate_frozen_shallowconvnet import (
    classification_metrics,
    load_committed_realtime,
)


class FrozenShallowConvNetExperimentTests(unittest.TestCase):
    def test_classification_metrics_reports_cross_entropy_and_class_counts(self) -> None:
        truth = np.asarray([0, 1, 2], dtype=np.int64)
        probabilities = np.asarray(
            [[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7]],
            dtype=np.float32,
        )

        metrics = classification_metrics(truth, probabilities, n_classes=3)

        self.assertEqual(metrics["correct"], 3)
        self.assertEqual(metrics["predicted_class_counts"], [1, 1, 1])
        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertGreater(metrics["cross_entropy"], 0.0)

    def test_realtime_loader_keeps_only_committed_primary_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            recording = Path(tmp_dir)
            chunks = recording / "chunks"
            chunks.mkdir()
            np.savez(
                chunks / "chunk_000000.npz",
                eeg_windows=np.arange(24, dtype=np.float32).reshape(3, 2, 4),
                labels_true=np.asarray([0, 1, 2]),
                scene_indices=np.asarray([4, 5, 6]),
                label_event_ids=np.asarray(["a", "b", "c"]),
                window_end_monotonic=np.asarray([1.0, 2.0, 3.0]),
                training_roles=np.asarray(
                    ["primary_decision", "auxiliary", "primary_decision"]
                ),
                adaptation_committed=np.asarray([True, False, True]),
            )

            windows, labels, scenes, event_ids, timestamps, sources = (
                load_committed_realtime(recording)
            )

            self.assertEqual(windows.shape, (2, 2, 4))
            np.testing.assert_array_equal(labels, [0, 2])
            np.testing.assert_array_equal(scenes, [4, 6])
            np.testing.assert_array_equal(event_ids, ["a", "c"])
            np.testing.assert_array_equal(timestamps, [1.0, 3.0])
            self.assertEqual(sources[0]["committed_windows"], 2)


if __name__ == "__main__":
    unittest.main()
