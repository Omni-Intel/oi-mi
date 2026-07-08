"""Streamlit web interface for oi-mi."""

from __future__ import annotations

import argparse
import html
import re
import sys
import threading
import time
from pathlib import Path

import streamlit as st

from acquisition.base import ElectrodeImpedance
from acquisition.factory import AcquirerFactory, register_default_acquirers
from adaptation.calibrator import Calibrator
from adaptation.mi_protocol import LABEL_DISPLAY, ProtocolConfig
from cli import (
    build_acquirer,
    build_game_command_outlet,
    build_marker_backend,
    build_model_path,
    load_config as load_app_config,
    resolve_config_path,
    write_config,
)
from decoder.real_time_decoder import RealTimeDecoder, TEST_MODE_PROMPTS
from game_command_router import get_shared_game_command_router
from models.factory import ModelFactory
from utils.markers import LSLCommandOutlet

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
    "START": "◎",
    "PAUSE": "◌",
    "TRANSITION": "·",
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
    )


def _format_send_state(status: dict) -> str:
    success = status.get("last_send_success")
    if success is True:
        return "success"
    if success is False:
        return "failed"
    return "-"


def render_ar_forwarding_panel(config: dict) -> None:
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

    pred_col, command_col, transport_col = st.columns(3)
    confidence = status.get("confidence")
    confidence_text = "-" if confidence is None else f"{float(confidence):.2f}"
    pred_col.metric("Last prediction", str(status.get("last_prediction", "-")), confidence_text)
    command_col.metric("Mapped command", str(status.get("mapped_command", "-")))
    transport_col.metric("Transport command", str(status.get("last_transport_command", "-")))

    error = status.get("last_send_error")
    if error:
        st.warning(f"最近一次 AR 转发失败: {error}")

    st.markdown("### 小车连接测试")
    st.caption("这些按钮只测试 AR/Unity 转发链路，不依赖 EEG、模型或实时解码。")
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


def _resolve_cue_symbol(message: str, *, event_type: str) -> tuple[str, bool] | None:
    upper_message = message.upper()
    if "LEFT" in upper_message:
        return _DISPLAY_SYMBOLS["LEFT"], event_type == "prediction"
    if "RIGHT" in upper_message:
        return _DISPLAY_SYMBOLS["RIGHT"], event_type == "prediction"
    if "REST" in upper_message or "IDLE" in upper_message:
        return _DISPLAY_SYMBOLS["REST"], event_type == "prediction"
    if "校准完成" in message:
        return _DISPLAY_SYMBOLS["DONE"], False
    if "执行失败" in message:
        return _DISPLAY_SYMBOLS["ERROR"], False
    if "开始按 MI game control protocol 采集" in message:
        return _DISPLAY_SYMBOLS["START"], False
    if "Baseline" in message or "休息" in message:
        return _DISPLAY_SYMBOLS["PAUSE"], False
    if "Block " in message or "练习阶段" in message or "实验指导语" in message:
        return _DISPLAY_SYMBOLS["TRANSITION"], False
    return None

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

    def __init__(self, cue_placeholder, log_placeholder) -> None:
        self.cue_placeholder = cue_placeholder
        self.log_placeholder = log_placeholder
        self.logs: list[str] = []
        self._lock = threading.Lock()
        self._pending_events: list[tuple[str, str]] = []
        self._ui_thread_id = threading.get_ident()

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
            if event_type in {"cue", "prediction"}:
                self._render_cue(msg, prediction=(event_type == "prediction"))
                self._append_log(msg)
                log_updated = True
            else:
                self._append_log(msg)
                log_updated = True

        if log_updated:
            self.log_placeholder.code("\n".join(self.logs))

    def _append_log(self, msg: str) -> None:
        self.logs.append(msg)
        if len(self.logs) > 18:
            self.logs.pop(0)

    def _render_cue(self, msg: str, *, prediction: bool) -> None:
        resolved = _resolve_cue_symbol(msg, event_type="prediction" if prediction else "cue")
        symbol = resolved[0] if resolved is not None else "·"
        is_prediction = resolved[1] if resolved is not None else prediction
        bg = "#F0FFF4" if is_prediction else "#F8FAFC"
        color = "#0F766E" if is_prediction else "#C2410C"
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


def init_live_view() -> tuple[StreamlitConsole, callable]:
    """Create cue/log placeholders for a running EEG page."""

    cue_box = st.empty()
    log_box = st.empty()
    console = StreamlitConsole(cue_box, log_box)

    def refresh() -> None:
        console.render_pending()
        return

    refresh()
    return console, refresh


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
    st.title("被试校准")
    protocol = ProtocolConfig.from_config(config)
    st.markdown("开始采集后，页面会显示提示与日志。")
    st.caption(
        f"主训练窗 {protocol.window_sec:.1f}s / 刷新 {protocol.stride_sec:.1f}s。"
        f" 正式 trial 结构为 {protocol.trial_timing.fixation_sec:.1f}s fixation + "
        f"{protocol.trial_timing.cue_sec:.1f}s cue + {protocol.trial_timing.control_sec:.1f}s control + "
        f"{protocol.trial_timing.iti_sec:.1f}s iti。"
    )

    is_new = st.radio("被试类型", ["新被试 (重新训练)", "老被试 (已有模型微调)"])

    if st.button("开始校准", type="primary"):
        try:
            is_new_flag = is_new.startswith("新")
            subject_id = str(config["subject_id"])
            model_name = str(config["model_name"])
            acquirer = build_acquirer(
                device_name=str(config["device_type"]),
                config=config,
            )
            effective_n_channels = int(acquirer.metadata.n_channels)
            console, refresh = init_live_view()
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
            )

            if not is_new_flag:
                calibrator.load_existing_weights()

            with st.spinner("校准进行中..."):
                result = calibrator.calibrate(
                    duration_sec=None,
                    epochs=int(config["new_subject_epochs"] if is_new_flag else config["old_subject_epochs"]),
                    batch_size=int(config["batch_size"]),
                    learning_rate=float(config["learning_rate"]),
                    patience=int(config["early_stopping_patience"]),
                    head_only=not is_new_flag,
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


def render_test_mode(config: dict) -> None:
    st.title("Cue 测试模式")
    st.markdown("运行过程中会展示 cue 和模型输出日志。")
    duration = st.number_input("测试总时长 (秒)", min_value=30, value=120, step=30)

    if st.button("开始测试", type="primary"):
        try:
            subject_id = str(config["subject_id"])
            model_name = str(config["model_name"])
            acquirer = build_acquirer(
                device_name=str(config["device_type"]),
                config=config,
            )
            effective_n_channels = int(acquirer.metadata.n_channels)
            console, refresh = init_live_view()
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
            if not model_path.exists():
                st.error(f"未找到模型权重文件: {model_path}。请先执行校准。")
                return
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

            with st.spinner("测试模式采集中..."):
                result = decoder.run_test_mode(
                    subject_id=subject_id,
                    marker_backend=build_marker_backend(config),
                    duration_sec=int(duration),
                    block_sec=float(config.get("collect_block_sec", 10.0)),
                    save_dir=Path(str(config.get("storage", {}).get("records_dir", "records_storage")))
                    / subject_id
                    / "test_mode",
                    heartbeat=refresh,
                )

            refresh()
            st.success("测试结束。")
            st.write(f"- 记录的窗口数: **{result['windows']}**")
            st.write(f"- 准确率: **{result['accuracy']:.3f}**")
            st.write(f"- 有效准确率: **{result['valid_accuracy']:.3f}**")
        except Exception as exc:  # noqa: BLE001
            st.error(f"执行失败: {exc}")


def render_realtime(config: dict) -> None:
    st.title("实时解码")
    st.markdown("开始后会持续显示模型输出。")
    render_ar_forwarding_panel(config)
    record = st.checkbox("保存实时脑波数据至本地记录")

    if st.button("开始实时解码", type="primary"):
        try:
            subject_id = str(config["subject_id"])
            model_name = str(config["model_name"])
            acquirer = build_acquirer(
                device_name=str(config["device_type"]),
                config=config,
            )
            effective_n_channels = int(acquirer.metadata.n_channels)
            console, refresh = init_live_view()
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
            if not model_path.exists():
                st.error(f"未找到模型权重文件: {model_path}。请先执行校准。")
                return
            model.load(model_path)

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
            )

            with st.spinner("实时解码运行中..."):
                decoder.run_forever(
                    subject_id=subject_id,
                    record=record,
                    save_dir=Path(str(config.get("storage", {}).get("records_dir", "records_storage")))
                    / subject_id
                    / "realtime",
                    heartbeat=refresh,
                )
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
        section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover,
        section[data-testid="stSidebar"] .stButton > button[kind="primary"] *,
        section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover * {
          color: #ffffff !important;
        }
        section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"],
        section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"]:hover,
        section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] *,
        section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"]:hover * {
          color: #ffffff !important;
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
            st.image(str(logo_path), width=280)
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
