"""Streamlit smoke coverage for the operator-facing car workflow."""

from __future__ import annotations

import unittest
from pathlib import Path

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
        self.assertEqual(metrics.get("缓冲窗口"), "0")

        app.button(key="nav_btn_校准").click().run()
        self.assertEqual(list(app.exception), [])
        self.assertEqual(app.radio[0].value, "新被试 (重新训练)")
        self.assertIn("正式实验", {button.label for button in app.button})

    def test_calibration_run_disables_return_button(self) -> None:
        gui_path = Path(__file__).resolve().parents[1] / "gui.py"
        source = gui_path.read_text(encoding="utf-8")

        self.assertIn('render_experiment_return_button(disabled=is_running)', source)
        self.assertIn('is_running = calibration_view == "run"', source)


if __name__ == "__main__":
    unittest.main()
