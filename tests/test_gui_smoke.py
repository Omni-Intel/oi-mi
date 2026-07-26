"""Streamlit smoke coverage for the operator-facing car workflow."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from streamlit.testing.v1 import AppTest


class GuiSmokeTests(unittest.TestCase):
    def test_gui_opens_and_realtime_page_exposes_car_recovery(self) -> None:
        gui_path = Path(__file__).resolve().parents[1] / "gui.py"
        app = AppTest.from_file(str(gui_path), default_timeout=30).run()

        self.assertEqual(list(app.exception), [])
        app.button(key="nav_btn_实时解码").click().run()

        self.assertEqual(list(app.exception), [])
        buttons = {button.key: button.label for button in app.button}
        self.assertEqual(buttons.get("ar_test_open_car"), "启动/重置并进入小车")
        self.assertIn("开始实时解码", buttons.values())
        metrics = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(metrics.get("状态"), "等待启动")
        self.assertEqual(metrics.get("更新次数"), "0")
        self.assertEqual(metrics.get("主决策窗口"), "0")

        app.button(key="nav_btn_校准").click().run()
        self.assertEqual(list(app.exception), [])
        self.assertEqual(list(app.radio), [])
        self.assertTrue(
            any("预计正式采集 12.0 分钟" in info.value for info in app.info)
        )
        self.assertIn("正式实验", {button.label for button in app.button})

    def test_calibration_run_disables_return_button(self) -> None:
        gui_path = Path(__file__).resolve().parents[1] / "gui.py"
        source = gui_path.read_text(encoding="utf-8")

        self.assertIn('render_experiment_return_button(disabled=is_running)', source)
        self.assertIn('is_running = calibration_view == "run"', source)
        self.assertIn("st.session_state.calibration_last_outcome = outcome", source)
        self.assertIn('st.session_state.gui_nav_mode = "校准"', source)
        self.assertIn("st.rerun()", source)

    def test_completed_calibration_is_recovered_after_browser_disconnect(self) -> None:
        import gui

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {
                "subject_id": "S001",
                "storage": {
                    "runtime_dir": str(root / "runtime"),
                    "records_dir": str(root / "records"),
                },
                "online_adaptation": {
                    "enabled": True,
                    "strategy": "neuroonline",
                },
            }
            model_path = root / "models" / "shallowconvnet.pt"
            model_path.parent.mkdir(parents=True)
            model_path.write_bytes(b"model")
            model_path.with_suffix(".metrics.yaml").write_text(
                "metrics: {}\n",
                encoding="utf-8",
            )
            Path(f"{model_path}.neuroonline.pt").write_bytes(b"crm")

            session_dir = (
                root
                / "records"
                / "S001"
                / "calibration"
                / "20260726_120000"
            )
            session_dir.mkdir(parents=True)
            (session_dir / "training_windows_main.npz").write_bytes(b"windows")
            metadata = {
                "windows_collected": 284,
                "model_path": str(model_path),
                "metrics": {"val_acc": 0.9},
            }

            gui._write_calibration_status(
                config,
                {"state": "running", "started_at_unix": 1.0},
            )
            (session_dir / "metadata.json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )

            recovered = gui._recover_completed_calibration(config)

            self.assertIsNotNone(recovered)
            assert recovered is not None
            self.assertTrue(recovered["ok"])
            self.assertTrue(recovered["recovered_after_reconnect"])
            self.assertEqual(recovered["windows_collected"], 284)
            self.assertEqual(
                gui._read_calibration_status(config)["state"],
                "completed",
            )

    def test_calibration_success_requires_crm_sidecar(self) -> None:
        import gui

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_path = root / "shallowconvnet.pt"
            model_path.write_bytes(b"model")
            model_path.with_suffix(".metrics.yaml").write_text(
                "metrics: {}\n",
                encoding="utf-8",
            )
            session_dir = root / "session"
            session_dir.mkdir()
            (session_dir / "metadata.json").write_text("{}", encoding="utf-8")
            windows_path = session_dir / "training_windows_main.npz"
            windows_path.write_bytes(b"windows")

            with self.assertRaisesRegex(RuntimeError, "CRM"):
                gui._validate_calibration_outcome(
                    {
                        "model_path": str(model_path),
                        "calibration_data_path": str(windows_path),
                        "session_dir": str(session_dir),
                    },
                    require_neuroonline_sidecar=True,
                )


if __name__ == "__main__":
    unittest.main()
