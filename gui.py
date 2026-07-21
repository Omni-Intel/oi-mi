"""Streamlit web interface for oi-mi."""

from __future__ import annotations

import argparse
import base64
import html
import re
import sys
import threading
import time
from pathlib import Path

import streamlit as st

from acquisition.base import AbstractAcquirer, ElectrodeImpedance
from acquisition.factory import AcquirerFactory, register_default_acquirers
from adaptation.calibrator import Calibrator
from adaptation.mi_protocol import LABEL_DESCRIPTION, LABEL_DISPLAY, LABEL_SYMBOL, ProtocolConfig
from cli import (
    build_acquirer,
    build_game_command_outlet,
    build_marker_backend,
    build_model_path,
    resolve_model_path,
    load_config as load_app_config,
    resolve_config_path,
    write_config,
)
from decoder.real_time_decoder import RealTimeDecoder, TEST_MODE_PROMPTS
from game_command_router import get_shared_game_command_router
from models.factory import ModelFactory
from utils.markers import LSLCommandOutlet
from utils.online_adaptation_dashboard import render_online_adaptation_panel, render_online_cue_panel
from utils.online_labels import (
    CuedOnlineLabelSource,
    ManualLabelHttpServer,
    ManualOnlineLabelSource,
    OnlineLabelSource,
    SimulatedOnlineLabelSource,
    build_cued_online_label_source,
)

_GUI_ROOT = Path(__file__).resolve().parent
_PAGE_ICON_FILENAME = "OMNI_ICON.svg"
_LOGO_FILENAME = "OMNI_LOGO_ENG_double_line.svg"


def _resolve_asset_path(filename: str) -> Path | None:
    """Resolve asset path across source and installed-package launch modes."""

    candidates = (
        _GUI_ROOT / "assets" / filename,
        Path.cwd() / "assets" / filename,
        Path.cwd() / "oi-mi" / "assets" / filename,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


_PAGE_ICON_PATH = _resolve_asset_path(_PAGE_ICON_FILENAME)
st.set_page_config(
    page_title="oi-mi Control Panel",
    page_icon=str(_PAGE_ICON_PATH) if _PAGE_ICON_PATH is not None else None,
    layout="wide",
)


def parse_config_path(argv: list[str] | None = None) -> Path:
    """Parse the optional config path passed after `streamlit run ... --`."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", dest="config_path", type=Path, default=None)
    args, _ = parser.parse_known_args(argv)
    return resolve_config_path(args.config_path)


CONFIG_PATH = parse_config_path(sys.argv[1:])
PROMPT_TEXTS = tuple(LABEL_DISPLAY.values()) + tuple(TEST_MODE_PROMPTS.values())

_DISPLAY_SYMBOLS = {
    "LEFT": "←",
    "RIGHT": "→",
    "REST": "○",
    "FIXATION": "+",
    "START": "◎",
    "PAUSE": "◌",
    "TRANSITION": "·",
    "BLANK": "",
    "DONE": "✓",
    "ERROR": "✕",
}

_AR_TEST_COMMANDS = ("START", "LEFT", "RIGHT", "STOP")


def _ar_game_mode(ar_game_cfg: dict) -> str:
    return "reverse relay" if bool(ar_game_cfg.get("reverse_enabled", False)) else "direct TCP"


def _ar_game_target(ar_game_cfg: dict) -> str:
    if bool(ar_game_cfg.get("reverse_enabled", False)):
        listen_ip = str(ar_game_cfg.get("reverse_listen_ip", "0.0.0.0"))
        listen_port = int(ar_game_cfg.get("reverse_listen_port", 5006))
        return f"listen {listen_ip}:{listen_port}"
    host = str(ar_game_cfg.get("host", "127.0.0.1"))
    port = int(ar_game_cfg.get("port", 5005))
    return f"{host}:{port}"


def _get_ar_forward_status() -> dict:
    return dict(st.session_state.get("ar_forward_status", {}))


def _set_ar_forward_status(**updates: object) -> None:
    status = _get_ar_forward_status()
    status.update(updates)
    status["updated_at"] = time.time()
    st.session_state.ar_forward_status = status


def _update_ar_decoder_status(payload: dict) -> None:
    _set_ar_forward_status(
        last_prediction=payload.get("prediction", "-"),
        confidence=payload.get("confidence"),
        mapped_command=payload.get("mapped_command", "-"),
        last_transport_command=payload.get("last_transport_command"),
        last_send_success=payload.get("last_send_success"),
        last_send_error=payload.get("last_send_error"),
        online_adaptation=payload.get("online_adaptation"),
    )


def _format_send_state(
    status: dict,
    *,
    now: float | None = None,
    stale_after_sec: float = 3.0,
) -> str:
    updated_at = status.get("updated_at")
    if updated_at is not None:
        current_time = time.time() if now is None else float(now)
        if current_time - float(updated_at) > max(float(stale_after_sec), 0.0):
            return "stale"
    success = status.get("last_send_success")
    if success is True:
        return "success"
    if success is False:
        return "failed"
    return "-"


def _current_streamlit_context() -> object | None:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:  # noqa: BLE001
        return None
    return get_script_run_ctx()


def _missing_model_guidance(config: dict) -> str:
    if bool(config.get("hardware_dummy_mode", False)) or str(config.get("device_type", "")) == "dummy":
        return "请先执行校准，或运行 `oi-mi seed-dummy-decoders` 生成 dummy 测试权重。"
    return "当前是真实设备模式，请先在“校准”页选择“新被试 (重新训练)”并完成正式校准。"


def render_ar_forwarding_panel(config: dict, *, render_adaptation: bool = True) -> None:
    output_cfg = config.get("output", {})
    ar_game_cfg = output_cfg.get("ar_game", {})
    enabled = bool(ar_game_cfg.get("enabled", False))
    status = _get_ar_forward_status()

    st.markdown("### AR 转发状态")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AR output", "enabled" if enabled else "disabled")
    col2.metric("Mode", _ar_game_mode(ar_game_cfg))
    col3.metric("Target", _ar_game_target(ar_game_cfg))
    col4.metric("Last send", _format_send_state(status))
    st.caption("Last send 仅表示最近一次 TCP 写入结果；stale 表示当前没有持续心跳，不能视为 Unity 仍在线。")

    pred_col, command_col, transport_col = st.columns(3)
    confidence = status.get("confidence")
    confidence_text = "-" if confidence is None else f"{float(confidence):.2f}"
    pred_col.metric("Last prediction", str(status.get("last_prediction", "-")), confidence_text)
    command_col.metric("Mapped command", str(status.get("mapped_command", "-")))
    transport_col.metric("Transport command", str(status.get("last_transport_command", "-")))

    error = status.get("last_send_error")
    if error:
        st.warning(f"最近一次 AR 转发失败: {error}")

    if render_adaptation:
        _render_online_adaptation_panel(status.get("online_adaptation"))

    st.markdown("### 小车连接测试")
    st.caption("这些按钮只测试 AR/Unity 转发链路，不依赖 EEG、模型或实时解码。")
    if st.button("启动/重置并进入小车", key="ar_test_open_car"):
        if not enabled:
            _set_ar_forward_status(
                mapped_command="OPEN_3D_GAME + LAUNCHER_SELECT",
                last_transport_command=None,
                last_send_success=False,
                last_send_error="output.ar_game.enabled is false.",
            )
            st.error("AR 游戏 TCP 控制未启用。请先在配置页启用后保存。")
        else:
            try:
                build_game_command_outlet(config)
            except Exception as exc:  # noqa: BLE001
                _set_ar_forward_status(
                    mapped_command="OPEN_3D_GAME + LAUNCHER_SELECT",
                    last_transport_command="LAUNCHER_SELECT",
                    last_send_success=False,
                    last_send_error=str(exc),
                )
                st.error(f"小车启动失败: {exc}")
            else:
                _set_ar_forward_status(
                    mapped_command="OPEN_3D_GAME + LAUNCHER_SELECT",
                    last_transport_command="LAUNCHER_SELECT",
                    last_send_success=True,
                    last_send_error=None,
                )
                st.success("Unity 已启动并进入 Fixed Speed 小车模式。")
    cols = st.columns(len(_AR_TEST_COMMANDS))
    for column, command in zip(cols, _AR_TEST_COMMANDS, strict=True):
        if column.button(f"Send {command}", key=f"ar_test_{command}"):
            if not enabled:
                _set_ar_forward_status(
                    mapped_command=command,
                    last_transport_command=None,
                    last_send_success=False,
                    last_send_error="output.ar_game.enabled is false.",
                )
                st.error("AR 游戏 TCP 控制未启用。请先在配置页启用后保存。")
                continue
            try:
                get_shared_game_command_router(config).push(command, source="web")
            except Exception as exc:  # noqa: BLE001
                _set_ar_forward_status(
                    mapped_command=command,
                    last_transport_command=command,
                    last_send_success=False,
                    last_send_error=str(exc),
                )
                st.error(f"{command} 发送失败: {exc}")
            else:
                _set_ar_forward_status(
                    mapped_command=command,
                    last_transport_command=command,
                    last_send_success=True,
                    last_send_error=None,
                )
                st.success(f"{command} 已发送。")

_CUE_COLORS = {
    "action": "#15803D",
    "rest": "#2563EB",
    "default": "#C2410C",
}

_CALIBRATION_GUIDANCE_STEPS = (
    ("", "保持稳定", "请坐稳，双手自然放松，眼睛看向屏幕中央。"),
    ("←", "左手想象", "看到这个图案时，只在脑海中想象左手重复握拳、松开，不要真的动。"),
    ("→", "右手想象", "看到这个图案时，只在脑海中想象右手重复握拳、松开，不要真的动。"),
    ("○", "静息放松", "看到这个图案时，请放松注视屏幕，不想象左右手动作。"),
    ("+", "准备提示", "看到这个图案时，请注视中央，准备下一次提示。"),
    ("·", "试次间隔", "看到这个图案时，只需等待下一次提示；它不是任务类别。"),
    ("", "重新集中", "如果走神，请在下一次提示出现时重新集中即可。"),
)


def _resolve_display_color(symbol: str, message: str) -> str:
    upper_message = message.upper()
    if symbol in {"←", "→"} or "LEFT" in upper_message or "RIGHT" in upper_message or "左手" in message or "右手" in message:
        return _CUE_COLORS["action"]
    if symbol in {"○", "◌"} or "REST" in upper_message or "IDLE" in upper_message or "休息" in message or "静息" in message:
        return _CUE_COLORS["rest"]
    return _CUE_COLORS["default"]


def _resolve_cue_symbol(message: str, *, event_type: str) -> tuple[str, bool] | None:
    upper_message = message.upper()
    if "接下来是" in message and "练习" in message:
        return _DISPLAY_SYMBOLS["BLANK"], False
    if "练习结束" in message or "开始正式采集" in message:
        return _DISPLAY_SYMBOLS["BLANK"], False
    if "FIXATION" in upper_message:
        return _DISPLAY_SYMBOLS["FIXATION"], False
    if "LEFT" in upper_message or "左手" in message:
        return _DISPLAY_SYMBOLS["LEFT"], event_type == "prediction"
    if "RIGHT" in upper_message or "右手" in message:
        return _DISPLAY_SYMBOLS["RIGHT"], event_type == "prediction"
    if "REST" in upper_message or "IDLE" in upper_message or "静息" in message:
        return _DISPLAY_SYMBOLS["REST"], event_type == "prediction"
    if "校准完成" in message or "测试结束" in message:
        return _DISPLAY_SYMBOLS["DONE"], False
    if "采集完成" in message:
        return _DISPLAY_SYMBOLS["TRANSITION"], False
    if "执行失败" in message:
        return _DISPLAY_SYMBOLS["ERROR"], False
    if "开始按 MI game control protocol 采集" in message:
        return _DISPLAY_SYMBOLS["START"], False
    if "Baseline" in message or "休息" in message:
        return _DISPLAY_SYMBOLS["PAUSE"], False
    if "ITI" in upper_message:
        return _DISPLAY_SYMBOLS["TRANSITION"], False
    if "Block " in message or "练习阶段" in message or "实验指导语" in message:
        return _DISPLAY_SYMBOLS["TRANSITION"], False
    return None


def _subject_facing_message(message: str, *, prediction: bool) -> str:
    """Return concise text for the subject-facing fullscreen view."""

    upper_message = message.upper()
    if prediction:
        return ""
    if "执行失败" in message:
        return "执行失败"
    if "校准完成" in message or "测试结束" in message:
        return "实验结束，请等待工作人员"
    if "采集完成" in message:
        return "采集完成，正在保存和训练，请等待工作人员"
    if "BASELINE" in upper_message:
        return "请放松注视中央"
    if "休息" in message:
        return "请休息，稍后继续"
    if "开始按 MI GAME CONTROL PROTOCOL 采集".upper() in upper_message:
        return "准备开始"
    if "测试模式启动" in message:
        return "准备开始"
    if "接下来是" in message and "练习" in message:
        return "接下来是 6 个练习 trial"
    if "练习结束" in message:
        return "练习结束"
    if "开始正式采集" in message:
        return "接下来开始正式采集"
    if "练习阶段" in message:
        return "接下来是 6 个练习 trial"
    if "练习" in message:
        if "LEFT" in upper_message:
            return "想象左手重复握拳、松开"
        if "RIGHT" in upper_message:
            return "想象右手重复握拳、松开"
        if "REST" in upper_message or "IDLE" in upper_message:
            return "放松注视，不想象动作"
        return "练习"
    if "PRACTICE_FIXATION" in upper_message:
        return "注视中央，准备下一次提示"
    if "PRACTICE_ITI" in upper_message:
        return "短暂休息"
    if "PRACTICE" in upper_message:
        if "LEFT" in upper_message:
            return "想象左手重复握拳、松开"
        if "RIGHT" in upper_message:
            return "想象右手重复握拳、松开"
        if "REST" in upper_message or "IDLE" in upper_message:
            return "放松注视，不想象动作"
        return "练习"
    if "FIXATION" in upper_message:
        return ""
    if (
        "LEFT" in upper_message
        or "RIGHT" in upper_message
        or "REST" in upper_message
        or "IDLE" in upper_message
        or "左手" in message
        or "右手" in message
        or "静息" in message
    ):
        return ""
    if "ITI" in upper_message or "BLOCK " in upper_message:
        return ""
    if message.startswith("- "):
        return ""
    return ""

SIDEBAR_NAV_PAGES = ("首页", "设置", "连通检测", "阻抗检查", "校准", "测试模式", "实时解码")
_IMPEDANCE_STATUS_COLORS = {
    "good": "#15803D",
    "ok": "#CA8A04",
    "poor": "#DC2626",
    "unknown": "#6B7280",
}


def _resolve_logo_svg_path() -> Path | None:
    """Resolve sidebar logo path."""
    return _resolve_asset_path(_LOGO_FILENAME)


def render_sidebar_logo(path: Path) -> None:
    """Render logo without Streamlit's image fullscreen control."""

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    st.markdown(
        (
            "<img "
            f"src='data:image/svg+xml;base64,{encoded}' "
            "alt='Omni-Intelligence' "
            "style='width: 280px; max-width: 100%; height: auto; display: block;'"
            ">"
        ),
        unsafe_allow_html=True,
    )


def load_config() -> dict:
    try:
        return load_app_config(CONFIG_PATH)
    except Exception as exc:  # noqa: BLE001
        st.error(f"加载配置文件失败: {exc}")
        return {}


def save_config(cfg: dict) -> None:
    try:
        write_config(CONFIG_PATH, cfg)
    except Exception as exc:  # noqa: BLE001
        st.error(f"保存配置文件失败: {exc}")


class StreamlitConsole:
    """Minimal Rich Console substitute that writes into Streamlit placeholders."""

    def __init__(self, cue_placeholder, log_placeholder, *, fullscreen: bool = False) -> None:
        self.cue_placeholder = cue_placeholder
        self.log_placeholder = log_placeholder
        self.fullscreen = fullscreen
        self.logs: list[str] = []
        self._lock = threading.Lock()
        self._pending_events: list[tuple[str, str]] = []
        self._ui_thread_id = threading.get_ident()
        self._last_stage_label = ""
        self._fullscreen_symbol_html = ""
        self._fullscreen_message_html = ""
        self._progress_label = "等待阶段"
        self._progress_elapsed = 0.0
        self._progress_duration = 0.0
        self._progress_started_at = time.monotonic()
        self._last_progress_render_at = 0.0

    def print(self, message, *args, **kwargs) -> None:
        raw_message = str(message)
        msg = re.sub(r"\[.*?\]", "", raw_message).strip()
        if not msg:
            return

        event_type = "log"
        if any(prompt in msg for prompt in PROMPT_TEXTS):
            event_type = "cue"
        elif "confidence:" in msg:
            event_type = "prediction"
        elif _resolve_cue_symbol(msg, event_type="log") is not None:
            event_type = "cue"

        with self._lock:
            self._pending_events.append((event_type, msg))

        if threading.get_ident() == self._ui_thread_id:
            self.render_pending()

    def render_pending(self) -> None:
        with self._lock:
            if not self._pending_events:
                return
            pending = list(self._pending_events)
            self._pending_events.clear()

        log_updated = False
        for event_type, msg in pending:
            if self.fullscreen and event_type == "prediction":
                self._append_log(msg)
                log_updated = True
                continue
            if event_type in {"cue", "prediction"}:
                self._render_cue(msg, prediction=(event_type == "prediction"))
                self._append_log(msg)
                log_updated = True
            else:
                self._append_log(msg)
                log_updated = True

        if log_updated and not self.fullscreen:
            self.log_placeholder.code("\n".join(self.logs))

    def _append_log(self, msg: str) -> None:
        self.logs.append(msg)
        if len(self.logs) > 18:
            self.logs.pop(0)

    def set_stage_progress(self, *, stage_name: str, elapsed_sec: float, duration_sec: float) -> None:
        if not self.fullscreen:
            return
        total = max(float(duration_sec), 0.0)
        elapsed = min(max(float(elapsed_sec), 0.0), total) if total > 0 else 0.0
        label = stage_name.strip() or self._last_stage_label or "阶段"
        self._last_stage_label = label
        self._progress_label = label
        self._progress_elapsed = elapsed
        self._progress_duration = total
        if elapsed <= 0.0:
            self._progress_started_at = time.monotonic()
        now = time.monotonic()
        if elapsed >= total or now - self._last_progress_render_at >= 0.25:
            self._render_fullscreen_surface()
            self._last_progress_render_at = now

    def _render_cue(self, msg: str, *, prediction: bool) -> None:
        resolved = _resolve_cue_symbol(msg, event_type="prediction" if prediction else "cue")
        symbol = resolved[0] if resolved is not None else "·"
        is_prediction = resolved[1] if resolved is not None else prediction
        bg = "#F0FFF4" if is_prediction else "#F8FAFC"
        color = _resolve_display_color(symbol, msg)
        if self.fullscreen:
            subject_message = _subject_facing_message(msg, prediction=is_prediction)
            self._fullscreen_symbol_html = ""
            if symbol:
                self._fullscreen_symbol_html = f"<div class='oi-experiment-symbol' style='color: {color};'>{symbol}</div>"
            self._fullscreen_message_html = ""
            if subject_message:
                safe_msg = html.escape(subject_message)
                message_class = "oi-experiment-message" if symbol else "oi-experiment-center-message"
                self._fullscreen_message_html = f"<div class='{message_class}'>{safe_msg}</div>"
            self._render_fullscreen_surface()
            return
        self.cue_placeholder.markdown(
            (
                "<div style='padding: 1.25rem; min-height: 8rem; border-radius: 12px; "
                "display: flex; align-items: center; justify-content: center; "
                f"background-color: {bg}; border: 1px solid #E2E8F0;'>"
                f"<div style='font-size: 4.5rem; line-height: 1; font-weight: 700; color: {color};'>{symbol}</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

    def _render_fullscreen_surface(self) -> None:
        total = max(float(self._progress_duration), 0.0)
        if total > 0:
            elapsed = min(max(time.monotonic() - self._progress_started_at, float(self._progress_elapsed)), total)
            self._progress_elapsed = elapsed
        else:
            elapsed = max(float(self._progress_elapsed), 0.0)
        ratio = 1.0 if total == 0 else elapsed / total
        progress_html = (
            "<div class='oi-debug-progress-card'>"
            "<div class='oi-debug-progress-title'>调试进度</div>"
            "<div class='oi-debug-progress-row'>"
            f"<span>{html.escape(self._progress_label)}</span>"
            f"<span>{elapsed:.1f}s / {total:.1f}s</span>"
            "</div>"
            "<div class='oi-debug-progress-track'>"
            f"<div class='oi-debug-progress-fill' style='width: {ratio * 100:.1f}%;'></div>"
            "</div>"
            "</div>"
        )
        self.cue_placeholder.markdown(
            (
                "<div class='oi-experiment-scroll-shell'>"
                "<section class='oi-experiment-stage'>"
                f"{self._fullscreen_symbol_html}"
                f"{self._fullscreen_message_html}"
                "</section>"
                "<section class='oi-debug-progress-section'>"
                f"{progress_html}"
                "</section>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


def enter_experiment_view() -> None:
    """Switch Streamlit chrome into a subject-facing experiment view."""

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"],
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"] {
          display: none !important;
        }
        html,
        body,
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"],
        .block-container {
          height: 100dvh !important;
          max-height: 100dvh !important;
          overflow: hidden !important;
        }
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        .stApp {
          background: #ffffff !important;
        }
        [data-testid="stMainBlockContainer"],
        .block-container {
          max-width: none !important;
          padding: 0 !important;
        }
        .oi-experiment-scroll-shell {
          width: 100vw;
          height: 100dvh;
          position: fixed;
          inset: 0;
          z-index: 9990;
          background: #f8fafc;
          overflow-x: hidden;
          overflow-y: scroll;
          overscroll-behavior: contain;
          scroll-behavior: auto;
          pointer-events: auto;
          touch-action: pan-y;
          scrollbar-width: thin;
        }
        .oi-experiment-stage {
          width: 100vw;
          height: 100dvh;
          position: relative;
          background: #f8fafc;
          border: none;
          box-sizing: border-box;
          overflow: hidden;
        }
        .oi-experiment-symbol {
          position: absolute;
          top: 45%;
          left: 50%;
          transform: translate(-50%, -50%);
          font-size: clamp(8rem, 24vw, 22rem);
          line-height: 1;
          font-weight: 800;
          text-align: center;
        }
        .oi-experiment-message {
          position: absolute;
          bottom: 16vh;
          left: 8vw;
          right: 8vw;
          text-align: center;
          font-size: clamp(1.3rem, 2.2vw, 2.6rem);
          line-height: 1.35;
          font-weight: 700;
          color: #0f172a;
        }
        .oi-experiment-center-message {
          position: absolute;
          top: 50%;
          left: 10vw;
          right: 10vw;
          transform: translateY(-50%);
          text-align: center;
          font-size: clamp(2.2rem, 5vw, 5rem);
          line-height: 1.18;
          font-weight: 800;
          color: #0f172a;
        }
        .oi-debug-progress-section {
          width: 100vw;
          min-height: 28dvh;
          box-sizing: border-box;
          display: flex;
          align-items: flex-start;
          justify-content: center;
          padding: 2rem 2rem 4rem;
          background: #f8fafc;
        }
        .oi-debug-progress-card {
          width: min(760px, calc(100vw - 4rem));
          margin: 0 auto;
          padding: 0.55rem 0.7rem;
          border: 1px solid #cbd5e1;
          border-radius: 8px;
          background: rgba(255, 255, 255, 0.96);
          color: #0f172a;
          box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
        }
        .oi-debug-progress-title {
          margin-bottom: 0.3rem;
          font-size: 0.72rem;
          line-height: 1.2;
          font-weight: 800;
          color: #334155;
        }
        .oi-debug-progress-row {
          display: flex;
          justify-content: space-between;
          gap: 1rem;
          margin-bottom: 0.3rem;
          font-size: 0.72rem;
          line-height: 1.2;
          font-weight: 600;
        }
        .oi-debug-progress-track {
          height: 0.32rem;
          overflow: hidden;
          border-radius: 999px;
          background: #e2e8f0;
        }
        .oi-debug-progress-fill {
          height: 100%;
          border-radius: inherit;
          background: #2563eb;
          transition: width 120ms linear;
        }
        .st-key-calibration_return_from_experiment {
          position: fixed;
          top: 1.15rem;
          left: 1.35rem;
          z-index: 10000;
          width: auto !important;
        }
        .st-key-calibration_return_from_experiment > button,
        .st-key-calibration_return_from_experiment .stButton > button {
          width: auto !important;
          min-width: 0 !important;
          min-height: 0 !important;
          padding: 0 !important;
          border: none !important;
          background: transparent !important;
          color: #0f172a !important;
          font-size: 1.85rem;
          line-height: 1;
          font-weight: 800;
          box-shadow: none !important;
          opacity: 1;
        }
        .st-key-calibration_return_from_experiment > button:hover,
        .st-key-calibration_return_from_experiment .stButton > button:hover {
          background: transparent !important;
          color: #0f172a !important;
          opacity: 1;
        }
        .oi-guidance-panel {
          position: fixed;
          inset: 0;
          z-index: 9990;
          display: flex;
          align-items: center;
          justify-content: center;
          background: #f8fafc;
          box-sizing: border-box;
          padding: 8vh 10vw;
        }
        .oi-guidance-content {
          width: min(1100px, 100%);
          text-align: center;
          display: flex;
          flex-direction: column;
          align-items: center;
        }
        .oi-guidance-kicker {
          margin-bottom: 1.2rem;
          font-size: clamp(1.4rem, 2vw, 2.2rem);
          font-weight: 700;
          color: #64748b;
          text-align: center;
        }
        .oi-guidance-symbol {
          margin: 0 auto 1.25rem;
          font-size: clamp(9rem, 20vw, 18rem);
          line-height: 1;
          font-weight: 800;
          color: #C2410C;
          text-align: center;
        }
        .oi-guidance-title {
          margin: 0 0 1.7rem;
          font-size: clamp(4rem, 7vw, 7.5rem);
          line-height: 1.12;
          font-weight: 800;
          color: #0f172a;
          text-align: center;
        }
        .oi-guidance-body {
          display: block;
          width: min(980px, 78vw);
          margin: 0 auto;
          max-width: none;
          font-size: clamp(2.3rem, 3.4vw, 3.9rem);
          line-height: 1.35;
          font-weight: 400;
          color: #1e293b;
          text-align: center !important;
          text-wrap: balance;
        }
        .st-key-calibration_guidance_next {
          position: fixed;
          right: 4vw;
          bottom: 4vh;
          z-index: 10000;
          width: min(16rem, 44vw) !important;
        }
        .st-key-calibration_guidance_next > button,
        .st-key-calibration_guidance_next .stButton > button {
          width: 100% !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_experiment_return_button() -> None:
    """Render the operator return button in experiment view."""

    if st.button("≪", key="calibration_return_from_experiment"):
        st.session_state.pop("calibration_experiment_view", None)
        st.session_state.pop("calibration_is_new", None)
        st.session_state.pop("calibration_after_guidance", None)
        st.session_state.pop("calibration_guidance_step", None)
        st.session_state.pop("test_mode_experiment_view", None)
        st.session_state.pop("test_mode_duration", None)
        st.session_state.gui_nav_mode = "测试模式" if st.session_state.get("gui_nav_mode") == "测试模式" else "校准"
        st.rerun()


def render_calibration_guidance() -> None:
    """Render pre-collection subject instructions."""

    step_index = int(st.session_state.get("calibration_guidance_step", 0))
    step_index = max(0, min(step_index, len(_CALIBRATION_GUIDANCE_STEPS) - 1))
    symbol, title, body = _CALIBRATION_GUIDANCE_STEPS[step_index]
    symbol_html = ""
    if symbol:
        symbol_color = _resolve_display_color(symbol, body)
        symbol_html = f"<div class='oi-guidance-symbol' style='color: {symbol_color};'>{html.escape(symbol)}</div>"
    st.markdown(
        (
            "<div class='oi-guidance-panel'>"
            "<div class='oi-guidance-content'>"
            f"<div class='oi-guidance-kicker'>步骤 {step_index + 1} / {len(_CALIBRATION_GUIDANCE_STEPS)}</div>"
            f"{symbol_html}"
            f"<h1 class='oi-guidance-title'>{html.escape(title)}</h1>"
            f"<div class='oi-guidance-body'>{html.escape(body)}</div>"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    is_last_step = step_index >= len(_CALIBRATION_GUIDANCE_STEPS) - 1
    next_label = "开始" if is_last_step else "下一步"
    if st.button(next_label, key="calibration_guidance_next", type="primary"):
        if is_last_step:
            next_view = str(st.session_state.get("calibration_after_guidance", "return"))
            st.session_state.pop("calibration_guidance_step", None)
            st.session_state.pop("calibration_after_guidance", None)
            if next_view == "return":
                st.session_state.pop("calibration_experiment_view", None)
                st.session_state.gui_nav_mode = "校准"
            else:
                st.session_state.calibration_experiment_view = next_view
        else:
            st.session_state.calibration_guidance_step = step_index + 1
        st.rerun()


def init_live_view(*, fullscreen: bool = False) -> tuple[StreamlitConsole, callable]:
    """Create cue/log placeholders for a running EEG page."""

    cue_box = st.empty()
    log_box = st.empty()
    console = StreamlitConsole(cue_box, log_box, fullscreen=fullscreen)

    def refresh() -> None:
        console.render_pending()
        return

    refresh()
    return console, refresh


def run_calibration_practice_preview(protocol: ProtocolConfig) -> None:
    """Run repeatable practice trials without touching hardware."""

    console, refresh = init_live_view(fullscreen=True)
    practice_labels = list(protocol.practice_labels) * max(protocol.practice_repetitions, 0)
    total_trials = len(practice_labels)
    timing = protocol.trial_timing
    practice_baseline_sec = 10.0
    if protocol.baseline_segments:
        practice_baseline_sec = min(practice_baseline_sec, max(float(protocol.baseline_segments[0].duration_sec), 0.0))

    _run_preview_event(
        console,
        refresh,
        message=f"接下来是 {total_trials} 个练习 trial，用于熟悉流程",
        stage_name="练习说明",
        duration_sec=3.0,
    )
    _run_preview_event(
        console,
        refresh,
        message=f"Baseline 练习静息注视 ({practice_baseline_sec:.0f}s)",
        stage_name="练习静息注视",
        duration_sec=practice_baseline_sec,
    )
    for index, label in enumerate(practice_labels, start=1):
        _run_preview_event(
            console,
            refresh,
            message=f"练习 {index}/{total_trials} {LABEL_DISPLAY[label]} {LABEL_DESCRIPTION[label]}",
            stage_name=f"练习 {index}/{total_trials}: 准备",
            duration_sec=0.5,
        )
        _run_preview_event(
            console,
            refresh,
            message="PRACTICE_FIXATION",
            stage_name=f"练习 {index}/{total_trials}: fixation",
            duration_sec=timing.fixation_sec,
        )
        cue_message = f"PRACTICE {LABEL_SYMBOL[label]} {LABEL_DISPLAY[label]}"
        _run_preview_event(
            console,
            refresh,
            message=cue_message,
            stage_name=f"练习 {index}/{total_trials}: cue {label}",
            duration_sec=timing.cue_sec,
        )
        _run_preview_event(
            console,
            refresh,
            message=cue_message,
            stage_name=f"练习 {index}/{total_trials}: control {label}",
            duration_sec=timing.control_sec,
            redraw_cue=False,
        )
        _run_preview_event(
            console,
            refresh,
            message="PRACTICE_ITI",
            stage_name=f"练习 {index}/{total_trials}: iti",
            duration_sec=timing.iti_sec,
        )
    _run_preview_event(
        console,
        refresh,
        message="练习结束",
        stage_name="练习结束",
        duration_sec=3.0,
    )


def _run_preview_event(
    console: StreamlitConsole,
    refresh: callable,
    *,
    message: str,
    stage_name: str,
    duration_sec: float,
    redraw_cue: bool = True,
) -> None:
    _start_preview_stage(console, stage_name=stage_name, duration_sec=duration_sec)
    if redraw_cue:
        console.print(message)
    refresh()
    _sleep_preview_stage(console, refresh, duration_sec=duration_sec)


def _preview_stage_name(message: str) -> str:
    upper_message = message.upper()
    if "PRACTICE_FIXATION" in upper_message:
        return "练习: fixation"
    if "PRACTICE_ITI" in upper_message:
        return "练习: iti"
    if "LEFT" in upper_message:
        return "练习: cue left"
    if "RIGHT" in upper_message:
        return "练习: cue right"
    if "REST" in upper_message or "IDLE" in upper_message:
        return "练习: cue rest"
    if "接下来是" in message:
        return "练习说明"
    if "练习结束" in message:
        return "练习结束"
    return "练习预览"


def _start_preview_stage(console: StreamlitConsole, *, stage_name: str, duration_sec: float) -> None:
    console.set_stage_progress(stage_name=stage_name, elapsed_sec=0.0, duration_sec=duration_sec)


def _sleep_preview_stage(
    console: StreamlitConsole,
    refresh: callable,
    *,
    duration_sec: float,
) -> None:
    total = max(float(duration_sec), 0.0)
    started_at = time.monotonic()
    deadline = started_at + total
    while time.monotonic() < deadline:
        refresh()
        elapsed = min(time.monotonic() - started_at, total)
        console.set_stage_progress(stage_name="", elapsed_sec=elapsed, duration_sec=total)
        time.sleep(min(0.05, max(deadline - time.monotonic(), 0.0)))
    console.set_stage_progress(stage_name="", elapsed_sec=total, duration_sec=total)


def run_calibration_session(config: dict, protocol: ProtocolConfig, *, is_new_flag: bool) -> None:
    """Run real calibration in the subject-facing experiment view."""

    try:
        subject_id = str(config["subject_id"])
        model_name = str(config["model_name"])
        acquirer = build_acquirer(
            device_name=str(config["device_type"]),
            config=config,
        )
        effective_n_channels = int(acquirer.metadata.n_channels)
        console, refresh = init_live_view(fullscreen=True)
        model = ModelFactory.get(
            model_name,
            n_chans=effective_n_channels,
            sfreq=float(config["sfreq"]),
            n_classes=int(config["n_classes"]),
            n_times=int(float(config["sfreq"]) * float(config["window_sec"])),
        )
        model_path = build_model_path(
            config,
            subject_id,
            model_name,
            device_name=str(config["device_type"]),
        )
        calibrator = Calibrator(
            acquirer=acquirer,
            model=model,
            marker_backend=build_marker_backend(config),
            console=console,
            sfreq=float(config["sfreq"]),
            window_sec=float(config["window_sec"]),
            step_sec=float(config["step_sec"]),
            model_path=model_path,
            calibration_records_dir=Path(str(config.get("storage", {}).get("records_dir", "records_storage")))
            / subject_id
            / "calibration",
            protocol_config=protocol,
            online_adaptation_config=config.get("online_adaptation", {}),
        )

        if not is_new_flag:
            load_path = resolve_model_path(
                config,
                subject_id,
                model_name,
                device_name=str(config["device_type"]),
                n_chans=effective_n_channels,
                n_times=int(float(config["sfreq"]) * float(config["window_sec"])),
            )
            if not load_path.exists():
                st.error(
                    f"未找到模型权重文件: "
                    f"{build_model_path(config, subject_id, model_name, device_name=str(config['device_type']))}。"
                    "请先执行校准。"
                )
                return
            if load_path.parent.name == "dummy_decoders":
                st.info(f"使用内置 dummy 测试权重: `{load_path}`")
            model.load(load_path)

        console.set_stage_progress(stage_name="启动 EEG 采集", elapsed_sec=0.0, duration_sec=10.0)
        refresh()
        with st.spinner("校准进行中..."):
            result = calibrator.calibrate(
                duration_sec=None,
                epochs=int(config["new_subject_epochs"] if is_new_flag else config["old_subject_epochs"]),
                batch_size=int(config["batch_size"]),
                learning_rate=float(config["learning_rate"]),
                patience=int(config["early_stopping_patience"]),
                head_only=not is_new_flag,
                include_practice=False,
                heartbeat=refresh,
            )

        refresh()
        st.success("校准完成。")
        st.write(f"- 采集窗口数: **{result.windows_collected}**")
        st.write(f"- 模型保存位置: `{result.model_path}`")
        if result.calibration_data_path is not None:
            st.write(f"- 校准数据保存位置: `{result.calibration_data_path}`")
        if result.session_dir is not None:
            st.write(f"- session 保存位置: `{result.session_dir}`")
    except Exception as exc:  # noqa: BLE001
        st.error(f"执行失败: {exc}")


def _format_impedance_value(impedance_kohm: float | None) -> str:
    if impedance_kohm is None:
        return "-"
    return f"{impedance_kohm:.2f}"


def _summarize_impedance_results(results: list[ElectrodeImpedance]) -> dict[str, int]:
    summary = {"good": 0, "ok": 0, "poor": 0, "unknown": 0}
    for result in results:
        summary[result.status if result.status in summary else "unknown"] += 1
    return summary


def _build_impedance_table_html(results: list[ElectrodeImpedance]) -> str:
    rows: list[str] = []
    for result in results:
        status = result.status if result.status in _IMPEDANCE_STATUS_COLORS else "unknown"
        color = _IMPEDANCE_STATUS_COLORS[status]
        name = html.escape(result.name or "-")
        message = html.escape(result.message or "-")
        rows.append(
            "<tr>"
            f"<td>{result.channel}</td>"
            f"<td>{name}</td>"
            f"<td>{_format_impedance_value(result.impedance_kohm)}</td>"
            f"<td><span style='display:inline-block;padding:0.15rem 0.45rem;border-radius:999px;"
            f"background:{color};color:#ffffff;font-weight:600'>{html.escape(status)}</span></td>"
            f"<td>{message}</td>"
            "</tr>"
        )
    body = "".join(rows) or "<tr><td colspan='5'>暂无结果</td></tr>"
    return (
        "<table style='width:100%;border-collapse:collapse;'>"
        "<thead><tr>"
        "<th style='text-align:left;padding:0.5rem;border-bottom:1px solid #E5E7EB;'>通道</th>"
        "<th style='text-align:left;padding:0.5rem;border-bottom:1px solid #E5E7EB;'>电极名</th>"
        "<th style='text-align:left;padding:0.5rem;border-bottom:1px solid #E5E7EB;'>阻抗 (kΩ)</th>"
        "<th style='text-align:left;padding:0.5rem;border-bottom:1px solid #E5E7EB;'>状态</th>"
        "<th style='text-align:left;padding:0.5rem;border-bottom:1px solid #E5E7EB;'>说明</th>"
        "</tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table>"
    )


def render_home() -> None:
    st.title("Omni-Intelligence® 脑机接口系统")
    st.markdown(
        """
        欢迎你，受试者！

        在接下来的任务中，你需要根据屏幕提示，在脑海中想象左手或右手的动作，或者保持放松状态。

        当出现“左”提示时，请在脑海中想象你的左手正在持续做动作（例如握拳、松开），但请不要实际移动手部。

        当出现“右”提示时，请想象你的右手正在做相同的动作，同样不要产生真实动作。

        当出现“休息”提示时，请保持放松，不要刻意想象任何动作。

        **请注意：**

        - 想象的是“自己在动”，而不是“看见手在动”
        - 保持身体静止，避免手指、肩膀或面部肌肉的实际运动
        - 尽量减少眨眼和其他多余动作

        如果中途注意力分散，请在下一次提示开始时重新集中即可。
        
        本次实验由 NCCLab 提供。
        """
    )



def render_settings(config: dict) -> None:
    st.title("核心参数配置")
    register_default_acquirers()

    protocol_cfg = config.setdefault("protocol", {})
    trial_timing_cfg = protocol_cfg.setdefault("trial_timing", {})
    output_cfg = config.setdefault("output", {})
    ar_game_cfg = output_cfg.setdefault("ar_game", {})

    subject_id = st.text_input("被试 ID (subject_id)", value=str(config.get("subject_id", "S001")))
    models = ModelFactory.list_models()
    model_name = st.selectbox(
        "默认模型 (model_name)",
        models,
        index=models.index(str(config.get("model_name", "riemann-mdm"))),
    )
    devices = AcquirerFactory.list_devices()
    current_device = str(config.get("device_type", devices[0]))
    device_type = st.selectbox(
        "采集设备 (device_type)",
        devices,
        index=devices.index(current_device) if current_device in devices else 0,
    )

    base_col1, base_col2 = st.columns(2)
    window_sec = float(
        base_col1.number_input(
            "特征窗长 (window_sec / 秒)",
            min_value=0.5,
            value=float(config.get("window_sec", 2.0)),
            step=0.25,
        )
    )
    step_sec = float(
        base_col2.number_input(
            "步长/刷新时间 (step_sec / 秒)",
            min_value=0.05,
            value=float(config.get("step_sec", 0.5)),
            step=0.05,
        )
    )

    st.markdown("### MI Game Control Protocol")
    protocol_col1, protocol_col2, protocol_col3 = st.columns(3)
    control_start_offset_sec = float(
        protocol_col1.number_input(
            "control 有效起点 (秒)",
            min_value=0.0,
            value=float(protocol_cfg.get("control_start_offset_sec", 0.5)),
            step=0.1,
        )
    )
    fixation_sec = float(
        protocol_col2.number_input(
            "fixation 时长 (秒)",
            min_value=0.5,
            value=float(trial_timing_cfg.get("fixation_sec", 2.0)),
            step=0.5,
        )
    )
    cue_sec = float(
        protocol_col3.number_input(
            "cue 时长 (秒)",
            min_value=0.5,
            value=float(trial_timing_cfg.get("cue_sec", 1.0)),
            step=0.5,
        )
    )
    protocol_col4, protocol_col5, protocol_col6 = st.columns(3)
    control_sec = float(
        protocol_col4.number_input(
            "control 时长 (秒)",
            min_value=1.0,
            value=float(trial_timing_cfg.get("control_sec", 5.0)),
            step=0.5,
        )
    )
    iti_sec = float(
        protocol_col5.number_input(
            "iti 时长 (秒)",
            min_value=0.5,
            value=float(trial_timing_cfg.get("iti_sec", 2.0)),
            step=0.5,
        )
    )
    rest_between_blocks_sec = float(
        protocol_col6.number_input(
            "block 间休息 (秒)",
            min_value=0.0,
            value=float(protocol_cfg.get("rest_between_blocks_sec", 35.0)),
            step=5.0,
        )
    )
    subject_col1, subject_col2, subject_col3 = st.columns(3)
    new_subject_blocks = int(
        subject_col1.number_input(
            "新被试 block 数",
            min_value=1,
            value=int(protocol_cfg.get("new_subject_blocks", 6)),
            step=1,
        )
    )
    new_subject_trials_per_class_per_block = int(
        subject_col2.number_input(
            "新被试每类每 block trial 数",
            min_value=1,
            value=int(protocol_cfg.get("new_subject_trials_per_class_per_block", 8)),
            step=1,
        )
    )
    old_subject_trials_per_class = int(
        subject_col3.number_input(
            "老被试每类 trial 数",
            min_value=1,
            value=int(protocol_cfg.get("old_subject_trials_per_class", 8)),
            step=1,
        )
    )
    old_subject_baseline_sec = float(
        st.number_input(
            "老被试 baseline idle 时长 (秒)",
            min_value=1.0,
            value=float(protocol_cfg.get("old_subject_baseline_sec", 60.0)),
            step=5.0,
        )
    )

    st.markdown("### AR 游戏控制")
    ar_col1, ar_col2, ar_col3 = st.columns(3)
    ar_game_enabled = ar_col1.checkbox("启用 AR 游戏 TCP 控制", value=bool(ar_game_cfg.get("enabled", False)))
    ar_game_host = ar_col2.text_input("AR 游戏主机", value=str(ar_game_cfg.get("host", "127.0.0.1")))
    ar_game_port = int(
        ar_col3.number_input(
            "AR 游戏端口",
            min_value=1,
            max_value=65535,
            value=int(ar_game_cfg.get("port", 5005)),
            step=1,
        )
    )
    ar_game_timeout_sec = float(
        st.number_input(
            "AR 游戏 TCP 超时 (秒)",
            min_value=0.1,
            value=float(ar_game_cfg.get("timeout_sec", 1.0)),
            step=0.1,
        )
    )

    if st.button("保存配置", type="primary"):
        config.update(
            {
                "subject_id": subject_id,
                "model_name": model_name,
                "device_type": device_type,
                "window_sec": window_sec,
                "step_sec": step_sec,
            }
        )
        protocol_cfg.update(
            {
                "control_start_offset_sec": control_start_offset_sec,
                "trial_timing": {
                    "fixation_sec": fixation_sec,
                    "cue_sec": cue_sec,
                    "control_sec": control_sec,
                    "iti_sec": iti_sec,
                },
                "new_subject_blocks": new_subject_blocks,
                "new_subject_trials_per_class_per_block": new_subject_trials_per_class_per_block,
                "old_subject_baseline_sec": old_subject_baseline_sec,
                "old_subject_trials_per_class": old_subject_trials_per_class,
                "rest_between_blocks_sec": rest_between_blocks_sec,
            }
        )
        next_ar_game_cfg = dict(ar_game_cfg)
        next_ar_game_cfg.update(
            {
                "enabled": ar_game_enabled,
                "host": ar_game_host,
                "port": ar_game_port,
                "timeout_sec": ar_game_timeout_sec,
            }
        )
        output_cfg["ar_game"] = next_ar_game_cfg
        save_config(config)
        st.success("配置已保存。")


def render_probe(config: dict) -> None:
    st.title("连通检测")
    st.markdown("在正式开始前，先确认采集设备网络可达并能返回 EEG 数据。")
    dur = st.number_input("探测时长 (秒)", min_value=0.1, value=3.0, step=0.5)

    if st.button("开始探测", type="primary"):
        selected_device = str(config.get("device_type", "neuracle"))

        with st.spinner(f"正在尝试连接 {selected_device} ..."):
            try:
                acquirer = build_acquirer(device_name=selected_device, config=config)
                st.info(f"设备对象已创建。尝试读取 {dur:.1f} 秒数据...")
                acquirer.start_stream()
                time.sleep(max(dur, 0.1))
                window, _ = acquirer.get_chunk(float(config.get("window_sec", 2.0)))
                acquirer.stop_stream()

                st.success("设备连通正常。")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Shape", str(window.shape))
                col2.metric("Mean (uV)", f"{window.mean():.3f}")
                col3.metric("Std (uV)", f"{window.std():.3f}")
                col4.metric("Max Abs (uV)", f"{abs(window).max():.3f}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"连通失败: {exc}")


def render_impedance(config: dict) -> None:
    st.title("阻抗检查")
    st.markdown("读取设备 SDK 暴露的真实 lead-off / impedance / contact quality 结果，不使用 EEG 波形方差估算。")
    st.caption("如果大面积通道异常，优先检查 reference、ground、bias/DRL，而不是逐个普通电极处理。")
    timeout_sec = float(st.number_input("检查超时 (秒)", min_value=1.0, value=10.0, step=1.0))

    if st.button("开始阻抗检查", type="primary"):
        selected_device = str(config.get("device_type", "neuracle"))
        st.session_state.impedance_results = []
        st.session_state.impedance_error = None
        st.session_state.impedance_device = selected_device

        with st.spinner(f"正在检查 {selected_device} 的电极阻抗..."):
            try:
                acquirer = build_acquirer(device_name=selected_device, config=config)
                if not acquirer.supports_impedance_check():
                    st.session_state.impedance_error = "当前设备/SDK 未暴露阻抗检查接口。"
                else:
                    st.session_state.impedance_results = acquirer.check_impedance(timeout_sec=timeout_sec)
            except Exception as exc:  # noqa: BLE001
                st.session_state.impedance_error = str(exc)

    results = list(st.session_state.get("impedance_results", []))
    error_message = st.session_state.get("impedance_error")
    if error_message:
        st.warning(error_message)
    if not results:
        return

    summary = _summarize_impedance_results(results)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Good", summary["good"])
    col2.metric("OK", summary["ok"])
    col3.metric("Poor", summary["poor"])
    col4.metric("Unknown", summary["unknown"])
    st.markdown(_build_impedance_table_html(results), unsafe_allow_html=True)


def render_calibration(config: dict) -> None:
    protocol = ProtocolConfig.from_config(config)

    calibration_view = st.session_state.get("calibration_experiment_view")
    if calibration_view is not None:
        enter_experiment_view()
        render_experiment_return_button()
        if calibration_view == "guidance":
            render_calibration_guidance()
        elif calibration_view == "practice":
            run_calibration_practice_preview(protocol)
            st.session_state.pop("calibration_experiment_view", None)
            st.session_state.gui_nav_mode = "校准"
            st.rerun()
        elif calibration_view == "run":
            run_calibration_session(
                config,
                protocol,
                is_new_flag=bool(st.session_state.pop("calibration_is_new", True)),
            )
            st.session_state.pop("calibration_experiment_view", None)
        return

    st.title("被试校准")
    st.markdown("开始采集后，页面会显示提示与日志。")
    st.caption(
        f"主训练窗 {protocol.window_sec:.1f}s / 刷新 {protocol.stride_sec:.1f}s。"
        f" 正式 trial 结构为 {protocol.trial_timing.fixation_sec:.1f}s fixation + "
        f"{protocol.trial_timing.cue_sec:.1f}s cue + {protocol.trial_timing.control_sec:.1f}s control + "
        f"{protocol.trial_timing.iti_sec:.1f}s iti。"
    )

    is_new = st.radio("被试类型", ["新被试 (重新训练)", "老被试 (已有模型微调)"])

    tutorial_col, practice_col, run_col = st.columns([1, 1, 1])
    tutorial_requested = tutorial_col.button("教程", type="secondary", use_container_width=True)
    practice_requested = practice_col.button("练习", type="secondary", use_container_width=True)
    run_requested = run_col.button("正式实验", type="primary", use_container_width=True)

    if tutorial_requested:
        st.session_state.calibration_experiment_view = "guidance"
        st.session_state.calibration_after_guidance = "return"
        st.session_state.calibration_guidance_step = 0
        st.rerun()

    if practice_requested:
        st.session_state.calibration_experiment_view = "practice"
        st.rerun()

    if run_requested:
        st.session_state.calibration_is_new = is_new.startswith("新")
        st.session_state.calibration_experiment_view = "run"
        st.rerun()


def render_test_mode(config: dict) -> None:
    test_view = st.session_state.get("test_mode_experiment_view")
    if test_view == "run":
        enter_experiment_view()
        render_experiment_return_button()
        run_test_mode_session(config, duration=int(st.session_state.pop("test_mode_duration", 120)))
        st.session_state.pop("test_mode_experiment_view", None)
        return

    st.title("Cue 测试模式")
    st.markdown("运行过程中会展示 cue 和模型输出日志。")
    duration = st.number_input("测试总时长 (秒)", min_value=30, value=120, step=30)

    if st.button("开始测试", type="primary"):
        st.session_state.test_mode_duration = int(duration)
        st.session_state.test_mode_experiment_view = "run"
        st.rerun()


def run_test_mode_session(config: dict, *, duration: int) -> None:
    try:
            subject_id = str(config["subject_id"])
            model_name = str(config["model_name"])
            acquirer = build_acquirer(
                device_name=str(config["device_type"]),
                config=config,
            )
            effective_n_channels = int(acquirer.metadata.n_channels)
            console, refresh = init_live_view(fullscreen=True)
            model = ModelFactory.get(
                model_name,
                n_chans=effective_n_channels,
                sfreq=float(config["sfreq"]),
                n_classes=int(config["n_classes"]),
                n_times=int(float(config["sfreq"]) * float(config["window_sec"])),
            )
            model_path = resolve_model_path(
                config,
                subject_id,
                model_name,
                device_name=str(config["device_type"]),
                n_chans=effective_n_channels,
                n_times=int(float(config["sfreq"]) * float(config["window_sec"])),
            )
            if not model_path.exists():
                st.error(
                    f"未找到模型权重文件: "
                    f"{build_model_path(config, subject_id, model_name, device_name=str(config['device_type']))}。"
                    f"{_missing_model_guidance(config)}"
                )
                return
            if model_path.parent.name == "dummy_decoders":
                st.info(f"使用内置 dummy 测试权重: `{model_path}`")
            model.load(model_path)

            command_outlet = LSLCommandOutlet(
                stream_name=str(config["output"]["command_stream_name"]),
                stream_type=str(config["output"]["command_stream_type"]),
            )
            decoder = RealTimeDecoder(
                acquirer=acquirer,
                model=model,
                console=console,
                command_outlet=command_outlet,
                game_command_outlet=build_game_command_outlet(config),
                sfreq=float(config["sfreq"]),
                window_sec=float(config["window_sec"]),
                step_sec=float(config["step_sec"]),
                confidence_threshold=float(config["confidence_threshold"]),
                mc_dropout_passes=int(config["mc_dropout_passes"]),
                status_callback=_update_ar_decoder_status,
            )

            block_sec = float(config.get("test_mode", {}).get("block_sec", config.get("collect_block_sec", 10.0)))
            protocol = ProtocolConfig.from_config(config)
            test_mode_cfg = config.get("test_mode", {})
            if "initial_rest_sec" in test_mode_cfg:
                initial_rest_sec = max(float(test_mode_cfg.get("initial_rest_sec", 0.0)), 0.0)
            elif protocol.baseline_segments:
                initial_rest_sec = min(max(float(protocol.baseline_segments[0].duration_sec), 0.0), 10.0)
            else:
                initial_rest_sec = 10.0

            def update_test_progress(stage_name: str, elapsed_sec: float, total_sec: float) -> None:
                console.set_stage_progress(stage_name=stage_name, elapsed_sec=elapsed_sec, duration_sec=total_sec)

            console.set_stage_progress(stage_name="启动测试模式", elapsed_sec=0.0, duration_sec=10.0)
            refresh()
            with st.spinner("测试模式采集中..."):
                result = decoder.run_test_mode(
                    subject_id=subject_id,
                    marker_backend=build_marker_backend(config),
                    duration_sec=int(duration),
                    block_sec=block_sec,
                    initial_rest_sec=initial_rest_sec,
                    save_dir=Path(str(config.get("storage", {}).get("records_dir", "records_storage")))
                    / subject_id
                    / "test_mode",
                    heartbeat=refresh,
                    stage_progress=update_test_progress,
                )

            refresh()
            console.print("[bold green]测试结束[/bold green]")
            refresh()
            st.success("测试结束。")
            st.write(f"- 记录的窗口数: **{result['windows']}**")
            st.write(f"- 准确率: **{result['accuracy']:.3f}**")
            st.write(f"- 有效准确率: **{result['valid_accuracy']:.3f}**")
    except Exception as exc:  # noqa: BLE001
        st.error(f"执行失败: {exc}")


def _render_online_adaptation_notice(adaptation_cfg: dict) -> None:
    if not bool(adaptation_cfg.get("enabled", False)):
        return
    simulation_cfg = adaptation_cfg.get("simulation", {})
    cued_cfg = adaptation_cfg.get("cued_labels", {})
    if bool(simulation_cfg.get("enabled", False)):
        source_text = "标签驱动 Dummy"
    elif bool(cued_cfg.get("enabled", True)):
        source_text = "自动 Cue 实验协议"
    else:
        source_text = "HTTP 真值标签"
    if str(adaptation_cfg.get("strategy", "periodic_head")).lower() == "neuroonline":
        neuro_cfg = adaptation_cfg.get("neuroonline", {})
        st.info(
            "NeuroOnline 已开启："
            f"累计 {int(neuro_cfg.get('history_threshold', 320))} 个标签窗口后，"
            f"每 {int(neuro_cfg.get('update_stride', 64))} 个样本全参数更新一次；"
            f"当前标签源为 {source_text}。"
        )
        return
    st.info(
        f"周期模型更新已开启：每 {float(adaptation_cfg.get('update_interval_sec', 600)) / 60.0:.1f} 分钟检查一次，"
        f"仅微调分类头；当前标签源为 {source_text}。"
    )


def _build_online_label_source(
    config: dict,
    adaptation_cfg: dict,
    acquirer: AbstractAcquirer,
) -> tuple[OnlineLabelSource | None, ManualLabelHttpServer | None]:
    if not bool(adaptation_cfg.get("enabled", False)):
        return None, None

    simulation_cfg = adaptation_cfg.get("simulation", {})
    if bool(simulation_cfg.get("enabled", False)) and str(acquirer.metadata.name) == "dummy":
        st.info("在线适配使用标签驱动 Dummy 模拟被试。")
        return (
            SimulatedOnlineLabelSource(
                acquirer,
                trial_sec=float(simulation_cfg.get("trial_sec", 6.0)),
                settle_sec=float(simulation_cfg.get("settle_sec", config["window_sec"])),
                seed=int(adaptation_cfg.get("random_seed", 17)),
            ),
            None,
        )

    cued_cfg = adaptation_cfg.get("cued_labels", {})
    if bool(cued_cfg.get("enabled", True)):
        st.info("在线适配使用自动平衡 Cue，并仅接收完整落在 control 有效区间内的窗口。")
        return build_cued_online_label_source(config), None

    source = ManualOnlineLabelSource(default_ttl_sec=2.0)
    server = ManualLabelHttpServer(source, host="127.0.0.1", port=8776)
    server.start()
    st.info("在线标签接口已启动: `http://127.0.0.1:8776/api/label`")
    return source, server


def render_realtime(config: dict) -> None:
    st.title("实时解码")
    st.markdown("开始后会持续显示模型输出。")
    render_ar_forwarding_panel(config, render_adaptation=False)
    cue_panel = st.empty()
    adaptation_panel = st.empty()
    online_label_source: OnlineLabelSource | None = None

    def redraw_cue_panel() -> None:
        cue_panel.empty()
        source_status = None
        if isinstance(online_label_source, CuedOnlineLabelSource):
            source_status = online_label_source.status()
        with cue_panel.container():
            render_online_cue_panel(source_status, ui=st)

    def redraw_adaptation_panel() -> None:
        adaptation_panel.empty()
        with adaptation_panel.container():
            render_online_adaptation_panel(
                _get_ar_forward_status().get("online_adaptation"),
                ui=st,
            )

    redraw_cue_panel()
    redraw_adaptation_panel()
    record = st.checkbox(
        "保存实时脑波数据至本地记录",
        value=bool(config.get("storage", {}).get("record_realtime_default", False)),
    )
    adaptation_cfg = config.get("online_adaptation", {})
    _render_online_adaptation_notice(adaptation_cfg)

    if st.button("开始实时解码", type="primary"):
        try:
            subject_id = str(config["subject_id"])
            model_name = str(config["model_name"])
            acquirer = build_acquirer(
                device_name=str(config["device_type"]),
                config=config,
            )
            effective_n_channels = int(acquirer.metadata.n_channels)
            console, refresh_console = init_live_view()
            last_dashboard_refresh = 0.0

            def refresh() -> None:
                nonlocal last_dashboard_refresh
                refresh_console()
                now = time.monotonic()
                if now - last_dashboard_refresh >= 0.5:
                    redraw_cue_panel()
                    redraw_adaptation_panel()
                    last_dashboard_refresh = now
            model = ModelFactory.get(
                model_name,
                n_chans=effective_n_channels,
                sfreq=float(config["sfreq"]),
                n_classes=int(config["n_classes"]),
                n_times=int(float(config["sfreq"]) * float(config["window_sec"])),
            )
            model_path = resolve_model_path(
                config,
                subject_id,
                model_name,
                device_name=str(config["device_type"]),
                n_chans=effective_n_channels,
                n_times=int(float(config["sfreq"]) * float(config["window_sec"])),
            )
            if not model_path.exists():
                st.error(
                    f"未找到模型权重文件: "
                    f"{build_model_path(config, subject_id, model_name, device_name=str(config['device_type']))}。"
                    f"{_missing_model_guidance(config)}"
                )
                return
            if model_path.parent.name == "dummy_decoders":
                st.info(f"使用内置 dummy 测试权重: `{model_path}`")
            model.load(model_path)

            online_label_source, online_label_server = _build_online_label_source(
                config,
                adaptation_cfg,
                acquirer,
            )

            primary_model_path = build_model_path(
                config,
                subject_id,
                model_name,
                device_name=str(config["device_type"]),
            )
            decoder = RealTimeDecoder(
                acquirer=acquirer,
                model=model,
                console=console,
                command_outlet=LSLCommandOutlet(
                    stream_name=str(config["output"]["command_stream_name"]),
                    stream_type=str(config["output"]["command_stream_type"]),
                ),
                game_command_outlet=build_game_command_outlet(config),
                sfreq=float(config["sfreq"]),
                window_sec=float(config["window_sec"]),
                step_sec=float(config["step_sec"]),
                confidence_threshold=float(config["confidence_threshold"]),
                mc_dropout_passes=int(config["mc_dropout_passes"]),
                status_callback=_update_ar_decoder_status,
                thread_context=_current_streamlit_context(),
                online_label_source=online_label_source,
                model_save_path=primary_model_path,
                batch_update_config=adaptation_cfg,
                n_classes=int(config["n_classes"]),
            )

            try:
                with st.spinner("实时解码运行中..."):
                    decoder.run_forever(
                        subject_id=subject_id,
                        record=record,
                        save_dir=Path(str(config.get("storage", {}).get("records_dir", "records_storage")))
                        / subject_id
                        / "realtime",
                        heartbeat=refresh,
                    )
            finally:
                if online_label_server is not None:
                    online_label_server.close()
        except Exception as exc:  # noqa: BLE001
            st.warning(f"解码已停止: {exc}")


def _set_gui_nav_mode(page: str) -> None:
    st.session_state.gui_nav_mode = page


def _inject_gui_nav_styles() -> None:
    st.markdown(
        """
        <style>
        /* Force a light palette so dark-text logo remains readable. */
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMainBlockContainer"] {
          background-color: #ffffff;
          color: #0f172a;
        }
        [data-testid="stHeader"] {
          background-color: #ffffff;
        }
        [data-testid="stToolbar"] {
          color: #334155;
        }
        section[data-testid="stSidebar"] {
          background: linear-gradient(180deg, #fff7ed 0%, #ffffff 70%);
          border-right: 1px solid rgba(15, 23, 42, 0.08);
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
          min-height: 100vh;
          display: flex;
          flex-direction: column;
          padding-bottom: 1rem;
        }
        section[data-testid="stSidebar"] * {
          color: #1e293b;
        }
        .stButton > button {
          color: #0f172a !important;
        }
        .stButton > button * {
          color: inherit !important;
        }
        section[data-testid="stSidebar"] .stButton > button {
          width: 100%;
          border-radius: 10px;
          padding-top: 0.72rem;
          padding-bottom: 0.72rem;
          font-weight: 600;
          font-size: 0.95rem;
          margin-bottom: 0.35rem;
          outline: none;
          transition: background-color 0.12s ease, border-color 0.12s ease, color 0.12s ease;
        }
        section[data-testid="stSidebar"] .stButton > button:focus-visible {
          box-shadow: 0 0 0 2px rgba(255, 90, 1, 0.4);
        }
        section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
          background-color: rgba(248, 250, 252, 0.95);
          border: 1px solid rgba(15, 23, 42, 0.12);
          color: rgb(30, 41, 59);
        }
        section[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
          border-color: rgba(255, 90, 1, 0.4);
          background-color: rgba(255, 90, 1, 0.07);
          color: rgb(15, 23, 42);
        }
        section[data-testid="stSidebar"] .stButton > button[kind="primary"],
        section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
          background-color: #ff4b4b !important;
          border-color: #ff4b4b !important;
          color: #ffffff !important;
        }
        section[data-testid="stSidebar"] .stButton > button[kind="primary"] *,
        section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover * {
          color: #ffffff !important;
        }
        section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"],
        section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"]:hover {
          background-color: #ff4b4b !important;
          border-color: #ff4b4b !important;
          color: #ffffff !important;
        }
        section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] *,
        section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"]:hover * {
          color: #ffffff !important;
        }
        [data-testid="stMain"] .stButton > button[kind="primary"],
        [data-testid="stMain"] .stButton > button[data-testid="stBaseButton-primary"] {
          background-color: #ff4b4b !important;
          border-color: #ff4b4b !important;
          color: #ffffff !important;
        }
        [data-testid="stMain"] .stButton > button[kind="primary"] *,
        [data-testid="stMain"] .stButton > button[data-testid="stBaseButton-primary"] * {
          color: #ffffff !important;
        }
        [data-testid="stMain"] .stButton > button[kind="primary"]:hover,
        [data-testid="stMain"] .stButton > button[data-testid="stBaseButton-primary"]:hover {
          background-color: #e53e3e !important;
          border-color: #e53e3e !important;
          color: #ffffff !important;
        }
        [data-testid="stMain"] .stButton > button[kind="secondary"],
        [data-testid="stMain"] .stButton > button[data-testid="stBaseButton-secondary"] {
          background-color: #ffffff !important;
          border-color: rgba(15, 23, 42, 0.18) !important;
          color: #0f172a !important;
        }
        [data-testid="stMain"] .stButton > button[kind="secondary"] *,
        [data-testid="stMain"] .stButton > button[data-testid="stBaseButton-secondary"] * {
          color: #0f172a !important;
        }
        [data-testid="stMain"] .stButton > button[kind="secondary"]:hover,
        [data-testid="stMain"] .stButton > button[data-testid="stBaseButton-secondary"]:hover {
          border-color: rgba(255, 90, 1, 0.45) !important;
          background-color: rgba(255, 90, 1, 0.06) !important;
          color: #0f172a !important;
        }
        .oi-sidebar-spacer {
          flex: 1 1 auto;
          min-height: 1.5rem;
        }
        .oi-sidebar-copyright {
          margin-top: 1rem;
          padding: 0.25rem 0.1rem 0;
        }
        .oi-sidebar-copyright .oi-company {
          font-size: 0.72rem;
          line-height: 1.45;
          font-weight: 600;
          color: #334155;
        }
        .oi-sidebar-copyright .oi-rights {
          margin-top: 0.35rem;
          font-size: 0.68rem;
          line-height: 1.45;
          color: #64748b;
        }
        .stMarkdown, .stText, p, label, h1, h2, h3, h4, h5, h6 {
          color: #0f172a;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    config = load_config()
    if not config:
        return

    _inject_gui_nav_styles()
    st.session_state.setdefault("gui_nav_mode", SIDEBAR_NAV_PAGES[0])

    with st.sidebar:
        logo_path = _resolve_logo_svg_path()
        if logo_path is not None:
            render_sidebar_logo(logo_path)
        st.title("oi-mi 工作台")
        for page in SIDEBAR_NAV_PAGES:
            is_active = st.session_state.gui_nav_mode == page
            st.button(
                page,
                key=f"nav_btn_{page}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
                on_click=_set_gui_nav_mode,
                args=(page,),
            )
        st.markdown("<div class='oi-sidebar-spacer'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="oi-sidebar-copyright">
              <div class="oi-rights">© 2026 Omni-Intelligence. All rights reserved.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        mode = st.session_state.gui_nav_mode

    if mode == "首页":
        render_home()
    elif mode == "设置":
        render_settings(config)
    elif mode == "连通检测":
        render_probe(config)
    elif mode == "阻抗检查":
        render_impedance(config)
    elif mode == "校准":
        render_calibration(config)
    elif mode == "测试模式":
        render_test_mode(config)
    elif mode == "实时解码":
        render_realtime(config)


if __name__ == "__main__":
    main()
