"""Command-line entry point for oi-mi."""

from __future__ import annotations

import importlib.util
import logging
import re
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import numpy as np
import yaml
from rich.console import Console
from rich.table import Table

from acquisition.factory import AcquirerFactory, register_default_acquirers
from adaptation.mi_protocol import ProtocolConfig
from game_command_router import get_shared_game_command_router
from utils.markers import (
    ArTcpCommandRelay,
    ArTcpCommandSender,
    LSLCommandOutlet,
    NoOpMarkerBackend,
    TriggerBoxMarkerBackend,
)
from utils.online_labels import ManualLabelHttpServer, ManualOnlineLabelSource
from utils.preprocessing import filter_and_transform
from web_command_server import start_web_command_server

LOGGER = logging.getLogger(__name__)
CONSOLE = Console()
DEFAULT_CONFIG_FILENAME = "config.yaml"
_PROJECT_DEFAULT_CONFIG_PATH = Path(__file__).with_name(DEFAULT_CONFIG_FILENAME)
_DEFAULT_CONFIG_TEMPLATE: dict[str, Any] = {
    "subject_id": "S001",
    "model_name": "riemann-mdm",
    "device_type": "neuracle",
    "sfreq": 250,
    "n_classes": 3,
    "window_sec": 2.0,
    "step_sec": 0.5,
    "confidence_threshold": 0.7,
    "mc_dropout_passes": 8,
    "new_subject_duration_sec": 1800,
    "old_subject_duration_sec": 300,
    "new_subject_epochs": 50,
    "old_subject_epochs": 5,
    "batch_size": 32,
    "learning_rate": 0.001,
    "early_stopping_patience": 40,
    "collect_block_sec": 10,
    "buffer_sec": 60,
    "protocol": {
        "control_start_offset_sec": 0.5,
        "control_stop_offset_sec": 4.5,
        "export_window_sec": 1.5,
        "export_stride_sec": 0.5,
        "practice_labels": ["left", "right", "idle", "left", "right"],
        "practice_repetitions": 1,
        "baseline_segments": [
            {
                "name": "eyes_open_fixation",
                "duration_sec": 60.0,
                "instruction": "睁眼注视中央十字，保持放松。",
            },
        ],
        "trial_timing": {
            "fixation_sec": 2.0,
            "cue_sec": 1.0,
            "control_sec": 5.0,
            "iti_sec": 2.0,
        },
        "new_subject_blocks": 6,
        "new_subject_trials_per_class_per_block": 8,
        "old_subject_baseline_sec": 60.0,
        "old_subject_trials_per_class": 8,
        "rest_between_blocks_sec": 35.0,
        "extra_rest_sec": 60.0,
        "dynamic_total_minutes_hint": 30.0,
        "random_seed": 17,
    },
    "device": {
        "neuracle_host": "127.0.0.1",
        "neuracle_port": 8712,
        "brainco_addr": "",
        "brainco_port": 0,
        "brainco_auto_discover": True,
        "brainco_scan_timeout_sec": 6.0,
        "brainco_ready_timeout_sec": 20.0,
        "brainco_start_retries": 2,
        "brainco_gain": 6,
        "brainco_signal_source": "NORMAL",
        "brainco_device_id": "eeg-cap",
        "trigger_serial_port": "",
    },
    "output": {
        "command_stream_name": "oi_mi_commands",
        "command_stream_type": "Markers",
        "ar_game": {
            "enabled": True,
            "host": "127.0.0.1",
            "port": 5005,
            "timeout_sec": 1.0,
            "reverse_enabled": True,
            "reverse_listen_ip": "0.0.0.0",
            "reverse_listen_port": 5006,
        },
        "web_control": {
            "enabled": True,
            "host": "0.0.0.0",
            "port": 8765,
            "manual_override_hold_sec": 0.8,
            "manual_override_release_sec": 0.25,
        },
    },
    "storage": {
        "models_dir": "models_storage",
        "records_dir": "records_storage",
    },
    "ssvep_game": {
        "rounds": 5,
        "keeper_save_probability": 0.35,
        "opponent_goal_probability": 0.70,
        "stability_windows": 2,
        "min_confidence": 0.35,
    },
}


@dataclass(slots=True)
class AppContext:
    """Shared CLI state."""

    config: dict[str, Any]
    config_path: Path
    console: Console


def default_config() -> dict[str, Any]:
    """Return a writable copy of the bundled default config payload."""

    if _PROJECT_DEFAULT_CONFIG_PATH.exists():
        with _PROJECT_DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as handle:
            project_template = yaml.safe_load(handle) or {}
        if isinstance(project_template, dict):
            return deepcopy(project_template)
        LOGGER.warning(
            "Bundled config template at %s is not a mapping; falling back to static defaults.",
            _PROJECT_DEFAULT_CONFIG_PATH,
        )
    return deepcopy(_DEFAULT_CONFIG_TEMPLATE)


def resolve_config_path(config_path: Path | None = None) -> Path:
    """Resolve the config path, defaulting to the bundled oi-mi config.yaml."""

    if config_path is not None:
        return Path(config_path).expanduser().resolve()

    cwd_config = Path.cwd() / DEFAULT_CONFIG_FILENAME
    if cwd_config.exists():
        return cwd_config.resolve()

    return _PROJECT_DEFAULT_CONFIG_PATH.resolve()


def write_config(path: Path, config: dict[str, Any]) -> None:
    """Persist config as UTF-8 YAML."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)


def ensure_config_exists(path: Path) -> Path:
    """Create a default config file when none exists."""

    if not path.exists():
        write_config(path, default_config())
        LOGGER.info("Created default config at %s", path)
    return path


def get_model_factory() -> Any:
    from models.factory import ModelFactory

    return ModelFactory


def get_calibrator_class() -> Any:
    from adaptation.calibrator import Calibrator

    return Calibrator


def get_realtime_decoder_class() -> Any:
    from decoder.real_time_decoder import RealTimeDecoder

    return RealTimeDecoder


def setup_logging() -> None:
    """Configure app-wide logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    # Keep CLI output focused; hide noisy third-party debug/info logs.
    for noisy_name in ("filelock", "torio", "matplotlib", "mne"):
        logging.getLogger(noisy_name).setLevel(logging.WARNING)


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate YAML config."""

    path = ensure_config_exists(resolve_config_path(path))
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    required_keys = {
        "subject_id",
        "model_name",
        "device_type",
        "sfreq",
        "n_classes",
        "window_sec",
        "step_sec",
    }
    missing = sorted(required_keys - set(config))
    if missing:
        raise click.ClickException(f"Missing required config keys: {', '.join(missing)}")
    if config["window_sec"] <= 0 or config["step_sec"] <= 0:
        raise click.ClickException("window_sec and step_sec must be positive.")
    if config["n_classes"] != 3:
        raise click.ClickException("This minimal build currently requires n_classes=3.")
    return config


def parse_subject_number(subject_id: str) -> int:
    """Extract a numeric subject index, defaulting to 1 when absent."""

    match = re.search(r"(\d+)", subject_id)
    if match is None:
        return 1
    return max(int(match.group(1)), 1)


def default_device_channels(device_name: str) -> int:
    """Return the default channel count for each supported device."""

    return 32 if device_name == "brainco" else 64


def _interactive_menu(ctx: click.Context, app: AppContext) -> None:
    """Interactive parameter input when user runs `oi-mi` only."""

    while True:
        CONSOLE.print("\n[bold cyan]oi-mi 交互菜单[/bold cyan]")
        CONSOLE.print("1) 列出可用模型")
        CONSOLE.print("2) 列出可用采集设备")
        CONSOLE.print("3) 校准（新/老被试，cue）")
        CONSOLE.print("4) 实时解码（无 cue 自动输出）")
        CONSOLE.print("5) 测试模式（有 cue + 保存流式 npy + 计算准确率）")
        CONSOLE.print("6) 设备连通性探测（probe-device）")
        CONSOLE.print("7) 配置参数 (窗长, 步长等)")
        CONSOLE.print("8) 启动 GUI (Streamlit)")
        CONSOLE.print("0) 退出")

        CONSOLE.print("9) 眼镜调试模式 (LEFT/RIGHT 循环)")

        choice = click.prompt(
            "选择功能",
            type=click.Choice(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]),
            default="3",
        )
        if choice == "0":
            return
        if choice == "1":
            ctx.invoke(ctx.command.get_command(ctx, "list-models"))
            continue
        if choice == "2":
            ctx.invoke(ctx.command.get_command(ctx, "list-devices"))
            continue

        if choice == "3":
            model_factory = get_model_factory()
            protocol = ProtocolConfig.from_config(app.config)
            subject_id = click.prompt(
                "被试ID",
                default=str(app.config.get("subject_id", "S001")),
                type=str,
            )
            mode = click.prompt("选择新/老被试", type=click.Choice(["new", "old"]), default="new")
            formal_trials = (
                protocol.new_subject_blocks * protocol.new_subject_trials_per_class_per_block * 3
                if mode == "new"
                else protocol.old_subject_trials_per_class * 3
            )
            CONSOLE.print(
                f"[bold cyan]当前 protocol[/bold cyan] formal_trials={formal_trials} "
                f"trial_total={protocol.trial_timing.total_sec:.1f}s "
                f"window={protocol.window_sec:.1f}s stride={protocol.stride_sec:.1f}s"
            )
            model_name = click.prompt(
                "模型(model registry)",
                type=click.Choice(model_factory.list_models()),
                default=str(app.config.get("model_name", "riemann-mdm")),
            )
            calibrate_cmd = ctx.command.get_command(ctx, "calibrate")
            ctx.invoke(
                calibrate_cmd,
                subject_id=subject_id,
                is_new=(mode == "new"),
                is_old=(mode == "old"),
                duration=None,
                model_name=model_name,
            )
            continue

        if choice == "4":
            model_factory = get_model_factory()
            subject_id = click.prompt(
                "被试ID",
                default=str(app.config.get("subject_id", "S001")),
                type=str,
            )
            model_name = click.prompt(
                "模型(model registry)",
                type=click.Choice(model_factory.list_models()),
                default=str(app.config.get("model_name", "riemann-mdm")),
            )
            device_name = click.prompt(
                "设备类型(device_type)",
                type=str,
                default=str(app.config.get("device_type", "neuracle")),
                show_default=True,
            )
            record = click.confirm("是否保存实时解码数据？", default=False)
            run_cmd = ctx.command.get_command(ctx, "run")
            ctx.invoke(
                run_cmd,
                subject_id=subject_id,
                model_name=model_name,
                device_name=device_name,
                test_mode=False,
                test_duration=600,
                record=record,
            )
            continue

        if choice == "5":
            model_factory = get_model_factory()
            subject_id = click.prompt(
                "被试ID",
                default=str(app.config.get("subject_id", "S001")),
                type=str,
            )
            model_name = click.prompt(
                "模型(model registry)",
                type=click.Choice(model_factory.list_models()),
                default=str(app.config.get("model_name", "riemann-mdm")),
            )
            device_name = click.prompt(
                "设备类型(device_type)",
                type=str,
                default=str(app.config.get("device_type", "neuracle")),
                show_default=True,
            )
            test_duration = click.prompt("测试时长(秒)", type=int, default=600)
            run_cmd = ctx.command.get_command(ctx, "run")
            ctx.invoke(
                run_cmd,
                subject_id=subject_id,
                model_name=model_name,
                device_name=device_name,
                test_mode=True,
                test_duration=test_duration,
            )
            continue

        if choice == "6":
            device_name = click.prompt(
                "设备类型(device_type)",
                type=str,
                default=str(app.config.get("device_type", "neuracle")),
            )
            duration = click.prompt("等待时长(秒)", type=float, default=5.0)
            probe_cmd = ctx.command.get_command(ctx, "probe-device")
            ctx.invoke(probe_cmd, device_name=device_name, duration=duration, save_buffer=False)
            continue

        if choice == "7":
            protocol_cfg = app.config.setdefault("protocol", {})
            trial_timing_cfg = protocol_cfg.setdefault("trial_timing", {})
            output_cfg = app.config.setdefault("output", {})
            ar_game_cfg = output_cfg.setdefault("ar_game", {})
            while True:
                CONSOLE.print("\n[bold magenta]-- 配置参数设置 --[/bold magenta]")
                CONSOLE.print(f"1) 主窗长 (window_sec): [green]{app.config.get('window_sec')}[/green]")
                CONSOLE.print(f"2) 刷新步长 (step_sec): [green]{app.config.get('step_sec')}[/green]")
                CONSOLE.print(f"3) 当前被试 (subject_id): [green]{app.config.get('subject_id')}[/green]")
                CONSOLE.print(f"4) 默认模型 (model_name): [green]{app.config.get('model_name')}[/green]")
                CONSOLE.print(f"5) control 起始偏移: [green]{protocol_cfg.get('control_start_offset_sec', 0.5)}[/green]")
                CONSOLE.print(f"6) fixation 时长: [green]{trial_timing_cfg.get('fixation_sec', 2.0)}[/green]")
                CONSOLE.print(f"7) cue 时长: [green]{trial_timing_cfg.get('cue_sec', 1.0)}[/green]")
                CONSOLE.print(f"8) control 时长: [green]{trial_timing_cfg.get('control_sec', 5.0)}[/green]")
                CONSOLE.print(f"9) iti 时长: [green]{trial_timing_cfg.get('iti_sec', 2.0)}[/green]")
                CONSOLE.print(f"10) 新被试 block 数: [green]{protocol_cfg.get('new_subject_blocks', 6)}[/green]")
                CONSOLE.print(f"11) 新被试每类每 block trial 数: [green]{protocol_cfg.get('new_subject_trials_per_class_per_block', 8)}[/green]")
                CONSOLE.print(f"12) 老被试 baseline 时长: [green]{protocol_cfg.get('old_subject_baseline_sec', 60.0)}[/green]")
                CONSOLE.print(f"13) 老被试每类 trial 数: [green]{protocol_cfg.get('old_subject_trials_per_class', 8)}[/green]")
                CONSOLE.print(f"14) block 间休息: [green]{protocol_cfg.get('rest_between_blocks_sec', 35.0)}[/green]")
                CONSOLE.print(f"15) AR游戏控制启用: [green]{ar_game_cfg.get('enabled', False)}[/green]")
                CONSOLE.print(f"16) AR游戏主机: [green]{ar_game_cfg.get('host', '127.0.0.1')}[/green]")
                CONSOLE.print(f"17) AR游戏端口: [green]{ar_game_cfg.get('port', 5005)}[/green]")
                CONSOLE.print(f"18) AR游戏超时: [green]{ar_game_cfg.get('timeout_sec', 1.0)}[/green]")
                CONSOLE.print("0) 返回上级菜单")

                sub_choice = click.prompt("选择要修改的项", type=click.Choice([str(i) for i in range(19)]), default="0")
                if sub_choice == "0":
                    break
                elif sub_choice == "1":
                    val = click.prompt("输入新的窗长 (window_sec)", type=float, default=float(app.config.get("window_sec", 2.0)))
                    app.config["window_sec"] = val
                elif sub_choice == "2":
                    val = click.prompt("输入新的步长 (step_sec)", type=float, default=float(app.config.get("step_sec", 0.5)))
                    app.config["step_sec"] = val
                elif sub_choice == "3":
                    val = click.prompt("输入新的被试ID (subject_id)", type=str, default=str(app.config.get("subject_id", "S001")))
                    app.config["subject_id"] = val
                elif sub_choice == "4":
                    val = click.prompt("输入新的默认模型 (model_name)", type=str, default=str(app.config.get("model_name", "riemann-mdm")))
                    app.config["model_name"] = val
                elif sub_choice == "5":
                    val = click.prompt("control 有效切窗起始偏移 (秒)", type=float, default=float(protocol_cfg.get("control_start_offset_sec", 0.5)))
                    protocol_cfg["control_start_offset_sec"] = val
                elif sub_choice == "6":
                    val = click.prompt("fixation 时长 (秒)", type=float, default=float(trial_timing_cfg.get("fixation_sec", 2.0)))
                    trial_timing_cfg["fixation_sec"] = val
                elif sub_choice == "7":
                    val = click.prompt("cue 时长 (秒)", type=float, default=float(trial_timing_cfg.get("cue_sec", 1.0)))
                    trial_timing_cfg["cue_sec"] = val
                elif sub_choice == "8":
                    val = click.prompt("control 时长 (秒)", type=float, default=float(trial_timing_cfg.get("control_sec", 5.0)))
                    trial_timing_cfg["control_sec"] = val
                elif sub_choice == "9":
                    val = click.prompt("iti 时长 (秒)", type=float, default=float(trial_timing_cfg.get("iti_sec", 2.0)))
                    trial_timing_cfg["iti_sec"] = val
                elif sub_choice == "10":
                    val = click.prompt("新被试 block 数", type=int, default=int(protocol_cfg.get("new_subject_blocks", 6)))
                    protocol_cfg["new_subject_blocks"] = val
                elif sub_choice == "11":
                    val = click.prompt("新被试每类每 block trial 数", type=int, default=int(protocol_cfg.get("new_subject_trials_per_class_per_block", 8)))
                    protocol_cfg["new_subject_trials_per_class_per_block"] = val
                elif sub_choice == "12":
                    val = click.prompt("老被试 baseline 时长 (秒)", type=float, default=float(protocol_cfg.get("old_subject_baseline_sec", 60.0)))
                    protocol_cfg["old_subject_baseline_sec"] = val
                elif sub_choice == "13":
                    val = click.prompt("老被试每类 trial 数", type=int, default=int(protocol_cfg.get("old_subject_trials_per_class", 8)))
                    protocol_cfg["old_subject_trials_per_class"] = val
                elif sub_choice == "14":
                    val = click.prompt("block 间休息时长 (秒)", type=float, default=float(protocol_cfg.get("rest_between_blocks_sec", 35.0)))
                    protocol_cfg["rest_between_blocks_sec"] = val
                elif sub_choice == "15":
                    val = click.confirm("是否启用 AR 游戏 TCP 控制", default=bool(ar_game_cfg.get("enabled", False)))
                    ar_game_cfg["enabled"] = val
                elif sub_choice == "16":
                    val = click.prompt("输入 AR 游戏主机地址", type=str, default=str(ar_game_cfg.get("host", "127.0.0.1")))
                    ar_game_cfg["host"] = val
                elif sub_choice == "17":
                    val = click.prompt("输入 AR 游戏端口", type=int, default=int(ar_game_cfg.get("port", 5005)))
                    ar_game_cfg["port"] = val
                elif sub_choice == "18":
                    val = click.prompt("输入 AR 游戏 TCP 超时(秒)", type=float, default=float(ar_game_cfg.get("timeout_sec", 1.0)))
                    ar_game_cfg["timeout_sec"] = val
                    
                with app.config_path.open("w", encoding="utf-8") as f:
                    yaml.safe_dump(app.config, f, allow_unicode=True, sort_keys=False)
                CONSOLE.print("[bold green]配置已更新！[/bold green]")
            continue

        if choice == "8":
            gui_cmd = ctx.command.get_command(ctx, "gui")
            ctx.invoke(gui_cmd)
            continue

        if choice == "9":
            debug_cmd = ctx.command.get_command(ctx, "debug-glasses")
            ctx.invoke(debug_cmd, interval_sec=1.0, pulse_sec=0.2)
            continue


def build_model_path(
    config: dict[str, Any],
    subject_id: str,
    model_name: str,
    *,
    device_name: str | None = None,
) -> Path:
    """Return the persisted weight path for a subject/device/model tuple."""

    extension = ".pkl" if model_name == "riemann-mdm" else ".pt"
    models_dir = Path(config["storage"]["models_dir"])
    resolved_device = str(device_name or config.get("device_type", "unknown"))
    return models_dir / subject_id / resolved_device / f"{model_name}{extension}"


def resolve_records_dir(config: dict[str, Any]) -> Path:
    """Return the root directory used for recorded sessions."""

    return Path(str(config.get("storage", {}).get("records_dir", "records_storage")))


def load_calibration_windows(
    records_dir: Path,
    subject_id: str,
    *,
    session_ids: tuple[str, ...] = (),
    use_processed: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    """Load and concatenate calibration windows saved on disk."""

    calibration_root = records_dir / subject_id / "calibration"
    if not calibration_root.exists():
        raise click.ClickException(f"Calibration directory not found: {calibration_root}")

    if session_ids:
        session_dirs = [calibration_root / session_id for session_id in session_ids]
    else:
        session_dirs = sorted(path for path in calibration_root.iterdir() if path.is_dir())
    if not session_dirs:
        raise click.ClickException(f"No calibration sessions found in {calibration_root}")

    feature_key = "processed_windows" if use_processed else "raw_windows"
    windows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    used_sessions: list[Path] = []
    reference_shape: tuple[int, int] | None = None

    for session_dir in session_dirs:
        dataset_path = session_dir / "training_windows_main.npz"
        if not dataset_path.exists():
            raise click.ClickException(f"Missing calibration dataset: {dataset_path}")
        with np.load(dataset_path) as payload:
            if feature_key not in payload or "labels" not in payload:
                raise click.ClickException(f"Calibration dataset missing required arrays: {dataset_path}")
            X = payload[feature_key].astype(np.float32)
            y = payload["labels"].astype(np.int64)
        if X.shape[0] != y.shape[0]:
            raise click.ClickException(f"Mismatched window and label counts in {dataset_path}")
        if X.shape[0] == 0:
            continue
        current_shape = (int(X.shape[1]), int(X.shape[2]))
        if reference_shape is None:
            reference_shape = current_shape
        elif current_shape != reference_shape:
            raise click.ClickException(
                f"Inconsistent calibration window shape: expected {reference_shape}, got {current_shape} in {dataset_path}"
            )
        windows.append(X)
        labels.append(y)
        used_sessions.append(session_dir)

    if not windows:
        raise click.ClickException(f"No usable calibration windows found in {calibration_root}")
    return np.concatenate(windows, axis=0), np.concatenate(labels, axis=0), used_sessions


def iter_test_mode_chunks(test_mode_dir: Path) -> list[Path]:
    """Return sorted test-mode chunk files."""

    chunks_dir = test_mode_dir / "chunks"
    chunk_paths = sorted(chunks_dir.glob("chunk_*.npz"))
    if not chunk_paths:
        raise click.ClickException(f"No test-mode chunks found in {chunks_dir}")
    return chunk_paths


def replay_test_mode(
    *,
    model: Any,
    test_mode_dir: Path,
    sfreq: float,
    mc_dropout_passes: int,
) -> dict[str, Any]:
    """Replay saved test-mode windows through the current decoder model."""

    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    confidence_all: list[np.ndarray] = []
    total_windows = 0

    for chunk_path in iter_test_mode_chunks(test_mode_dir):
        with np.load(chunk_path) as payload:
            if "eeg_windows" not in payload or "labels_true" not in payload:
                raise click.ClickException(f"Invalid test-mode chunk format: {chunk_path}")
            windows = payload["eeg_windows"].astype(np.float32)
            y_true = payload["labels_true"].astype(np.int64)
        processed = np.stack(
            [filter_and_transform(window, sfreq=sfreq) for window in windows],
            axis=0,
        ).astype(np.float32)
        probabilities = model.predict_proba(
            processed,
            mc_dropout_passes=mc_dropout_passes,
        )
        y_pred = np.argmax(probabilities, axis=1).astype(np.int64)
        confidences = np.max(probabilities, axis=1).astype(np.float32)
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
        confidence_all.append(confidences)
        total_windows += int(windows.shape[0])

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)
    confidences = np.concatenate(confidence_all, axis=0)
    accuracy = float(np.mean(y_pred == y_true)) if y_true.size else 0.0
    return {
        "windows": total_windows,
        "accuracy": accuracy,
        "mean_confidence": float(np.mean(confidences)) if confidences.size else 0.0,
        "y_true": y_true,
        "y_pred": y_pred,
        "confidences": confidences,
    }


def build_acquirer(
    *,
    device_name: str,
    config: dict[str, Any],
) -> Any:
    """Instantiate the selected acquisition backend."""

    register_default_acquirers()
    device_cfg = config.get("device", {})
    resolved_channels = default_device_channels(device_name)
    kwargs: dict[str, Any] = {
        "sfreq": float(config["sfreq"]),
        "n_channels": resolved_channels,
        "buffer_sec": float(config["buffer_sec"]),
    }
    if device_name == "neuracle":
        kwargs["neuracle_host"] = str(device_cfg.get("neuracle_host", "127.0.0.1"))
        kwargs["neuracle_port"] = int(device_cfg.get("neuracle_port", 8712))
    if device_name == "brainco":
        kwargs["brainco_addr"] = str(device_cfg.get("brainco_addr", ""))
        kwargs["brainco_port"] = int(device_cfg.get("brainco_port", 0))
        kwargs["auto_discover"] = bool(device_cfg.get("brainco_auto_discover", True))
        kwargs["scan_timeout_sec"] = float(device_cfg.get("brainco_scan_timeout_sec", 6.0))
        kwargs["ready_timeout_sec"] = float(device_cfg.get("brainco_ready_timeout_sec", 20.0))
        kwargs["start_retries"] = int(device_cfg.get("brainco_start_retries", 2))
        kwargs["eeg_gain"] = int(device_cfg.get("brainco_gain", 6))
        kwargs["signal_source"] = str(device_cfg.get("brainco_signal_source", "NORMAL"))
        kwargs["device_id"] = str(device_cfg.get("brainco_device_id", "eeg-cap"))
    return AcquirerFactory.create(device_name, **kwargs)


def build_marker_backend(config: dict[str, Any]) -> Any:
    """Select a marker backend based on config."""

    serial_port = str(config.get("device", {}).get("trigger_serial_port", "")).strip()
    if serial_port:
        return TriggerBoxMarkerBackend(serial_port)
    return NoOpMarkerBackend()


def build_game_command_outlet(config: dict[str, Any]) -> Any:
    """Build the shared command outlet used to control the AR game."""

    game_output_cfg = config.get("output", {}).get("ar_game", {})
    if not bool(game_output_cfg.get("enabled", False)):
        return None
    return get_shared_game_command_router(config).build_proxy(source="decoder")


def build_raw_game_transport(config: dict[str, Any]) -> Any:
    """Return the underlying shared transport for infrastructure commands."""

    game_output_cfg = config.get("output", {}).get("ar_game", {})
    if not bool(game_output_cfg.get("enabled", False)):
        return None
    return get_shared_game_command_router(config).raw_transport()


def run_glasses_debug_loop(outlet: Any, console: Console, *, interval_sec: float = 1.0) -> None:
    """Continuously alternate LEFT/RIGHT with STOP gaps for glasses debugging."""

    commands = ("LEFT", "RIGHT")
    console.print(
        "[bold cyan]Glasses debug mode running[/bold cyan] "
        "cycle=LEFT -> STOP -> RIGHT -> STOP, interval=1.0s. Press Ctrl+C to stop."
    )
    outlet.push("START")
    index = 0
    try:
        while True:
            command = commands[index % len(commands)]
            console.print(f"[bold cyan]AR debug -> {command}[/bold cyan]")
            outlet.push(command)
            time.sleep(interval_sec)
            console.print("[bold cyan]AR debug -> STOP[/bold cyan]")
            outlet.push("STOP")
            time.sleep(interval_sec)
            index += 1
    finally:
        try:
            outlet.push("STOP")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to send final STOP in glasses debug mode: %s", exc)
        close = getattr(outlet, "close", None)
        if callable(close):
            close()


def run_debug_glasses_loop(
    app: AppContext,
    *,
    interval_sec: float = 1.0,
    pulse_sec: float = 0.2,
) -> None:
    """Continuously alternate LEFT/RIGHT pulses with STOP in between."""

    if interval_sec <= 0:
        raise click.ClickException("interval_sec must be positive.")
    if pulse_sec <= 0:
        raise click.ClickException("pulse_sec must be positive.")
    if pulse_sec >= interval_sec:
        raise click.ClickException("pulse_sec must be smaller than interval_sec.")

    outlet = build_raw_game_transport(app.config)
    if outlet is None:
        raise click.ClickException("output.ar_game.enabled must be true.")

    commands = ("LEFT", "RIGHT")
    command_index = 0
    app.console.print(
        "[bold cyan]Debug glasses mode running[/bold cyan] "
        f"interval={interval_sec:.2f}s pulse={pulse_sec:.2f}s "
        "commands=LEFT/RIGHT, idle=STOP"
    )
    app.console.print("[bold cyan]Press Ctrl+C to stop.[/bold cyan]")

    try:
        outlet.push("START")
        outlet.push("STOP")
        while True:
            command = commands[command_index]
            outlet.push(command)
            app.console.print(f"[bold green]AR debug[/bold green] {command}")
            time.sleep(pulse_sec)
            outlet.push("STOP")
            time.sleep(interval_sec - pulse_sec)
            command_index = (command_index + 1) % len(commands)
    except KeyboardInterrupt:
        app.console.print("\n[bold red]Stopping debug glasses mode[/bold red]")
    except Exception as exc:
        raise click.ClickException(f"Debug glasses mode failed: {exc}") from exc
    finally:
        try:
            outlet.push("STOP")
        except Exception:
            pass
        if hasattr(outlet, "close"):
            outlet.close()


@click.group(invoke_without_command=True)
@click.option(
    "--config",
    "config_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Path to YAML config. Defaults to ./config.yaml and creates it if missing.",
)
@click.pass_context
def cli(ctx: click.Context, config_path: Path | None) -> None:
    """oi-mi command group."""

    setup_logging()
    resolved_config_path = resolve_config_path(config_path)
    config = load_config(resolved_config_path)
    start_web_command_server(config)
    ctx.obj = AppContext(config=config, config_path=resolved_config_path, console=CONSOLE)
    if ctx.invoked_subcommand is None:
        _interactive_menu(ctx, app=ctx.obj)


@cli.command()
@click.pass_obj
def gui(app: AppContext) -> None:
    """Launch the Streamlit graphical user interface."""
    import sys
    import subprocess
    # Always prefer the gui.py next to this cli.py to avoid
    # accidentally importing an unrelated top-level "gui" module.
    gui_script = Path(__file__).with_name("gui.py").resolve()
    if not gui_script.exists():
        app.console.print("[bold red]未找到 gui.py 文件！[/bold red]")
        return
    app.console.print(f"[bold cyan]正在启动 GUI: streamlit run {gui_script}[/bold cyan]")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(gui_script),
            "--",
            "--config",
            str(app.config_path),
        ]
    )

@cli.command("list-models")
def list_models() -> None:
    """List all registered decoder names."""

    model_factory = get_model_factory()
    table = Table(title="oi-mi Models")
    table.add_column("Model")
    for model_name in model_factory.list_models():
        table.add_row(model_name)
    CONSOLE.print(table)


@cli.command("list-devices")
def list_devices() -> None:
    """List all registered acquisition backends."""

    register_default_acquirers()
    table = Table(title="oi-mi Devices")
    table.add_column("Device")
    for device_name in AcquirerFactory.list_devices():
        table.add_row(device_name)
    CONSOLE.print(table)


@cli.command("relay-game")
@click.pass_obj
def relay_game(app: AppContext) -> None:
    """Run a standalone reverse relay for the AR game control channel."""

    game_output_cfg = app.config.get("output", {}).get("ar_game", {})
    if not bool(game_output_cfg.get("enabled", False)):
        raise click.ClickException("output.ar_game.enabled must be true.")
    if not bool(game_output_cfg.get("reverse_enabled", False)):
        raise click.ClickException("output.ar_game.reverse_enabled must be true.")

    outlet = build_game_command_outlet(app.config)
    if not isinstance(outlet, ArTcpCommandRelay):
        raise click.ClickException("Failed to start reverse AR relay.")

    app.console.print(
        "[bold cyan]AR relay running[/bold cyan] "
        f"local={game_output_cfg.get('host', '127.0.0.1')}:{game_output_cfg.get('port', 5005)} "
        f"downstream={game_output_cfg.get('reverse_listen_ip', '0.0.0.0')}:{game_output_cfg.get('reverse_listen_port', 5006)}"
    )
    app.console.print("[bold cyan]Local tools/oi-mi send to localhost relay; glasses Unity should connect back here.[/bold cyan]")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        app.console.print("\n[bold red]Stopping AR relay[/bold red]")
    finally:
        outlet.close()


@cli.command("debug-glasses")
@click.option(
    "--interval-sec",
    type=float,
    default=1.0,
    show_default=True,
    help="Seconds between alternating LEFT and RIGHT pulses.",
)
@click.option(
    "--pulse-sec",
    type=float,
    default=0.2,
    show_default=True,
    help="How long each LEFT/RIGHT pulse lasts before returning to STOP.",
)
@click.pass_obj
def debug_glasses(app: AppContext, interval_sec: float, pulse_sec: float) -> None:
    """Send alternating LEFT/RIGHT commands to the glasses with STOP in between."""

    run_debug_glasses_loop(app, interval_sec=interval_sec, pulse_sec=pulse_sec)


@cli.command("probe-device")
@click.option("--device", "device_name", type=str, default=None, help="Acquirer registry name.")
@click.option(
    "--duration",
    type=float,
    default=5.0,
    show_default=True,
    help="How many seconds to wait before grabbing one window.",
)
@click.option(
    "--save-buffer",
    is_flag=True,
    help="Save full forwarded buffer as .npy for diagnostics.",
)
@click.pass_obj
def probe_device(
    app: AppContext,
    device_name: str | None,
    duration: float,
    save_buffer: bool,
) -> None:
    """Quickly verify local device forwarding and data reception."""

    config = app.config
    selected_device = device_name or str(config["device_type"])
    acquirer = build_acquirer(device_name=selected_device, config=config)
    resolved_channels = int(acquirer.metadata.n_channels)

    app.console.print(
        f"[bold cyan]连接设备中...[/bold cyan] device={selected_device} channels={resolved_channels}"
    )
    try:
        acquirer.start_stream()
        time.sleep(max(duration, 0.1))
        window, _ = acquirer.get_chunk(float(config["window_sec"]))
        stats = {
            "shape": tuple(int(d) for d in window.shape),
            "mean_uV": float(np.mean(window)),
            "std_uV": float(np.std(window)),
            "max_abs_uV": float(np.max(np.abs(window))),
        }
        app.console.print(
            "[bold green]设备转发正常[/bold green] "
            f"shape={stats['shape']} mean={stats['mean_uV']:.3f} "
            f"std={stats['std_uV']:.3f} max_abs={stats['max_abs_uV']:.3f}"
        )
        if save_buffer and hasattr(acquirer, "save_full_buffer_npy"):
            stamp = time.strftime("%Y%m%d_%H%M%S")
            out = Path("records_storage") / "device_probe" / f"{selected_device}_{stamp}.npy"
            saved_path = acquirer.save_full_buffer_npy(out)  # type: ignore[attr-defined]
            app.console.print(f"[bold green]已保存完整缓冲区[/bold green] {saved_path}")
    except Exception as exc:
        raise click.ClickException(f"Probe failed for device={selected_device}: {exc}") from exc
    finally:
        acquirer.stop_stream()


@cli.command()
@click.option("--subject", "subject_id", required=True, type=str)
@click.option("--new", "is_new", is_flag=True, help="Train a new subject from scratch/base weights.")
@click.option("--old", "is_old", is_flag=True, help="Fast adaptation for an existing subject.")
@click.option("--duration", type=int, default=None, help="Calibration duration in seconds.")
@click.option("--model", "model_name", type=str, default=None, help="Model registry name.")
@click.pass_obj
def calibrate(
    app: AppContext,
    subject_id: str,
    is_new: bool,
    is_old: bool,
    duration: int | None,
    model_name: str | None,
) -> None:
    """Collect calibration data and train or adapt a decoder."""

    if is_new == is_old:
        raise click.ClickException("Choose exactly one of --new or --old.")

    config = app.config
    selected_model = model_name or str(config["model_name"])
    duration_sec = duration or int(
        config["new_subject_duration_sec"] if is_new else config["old_subject_duration_sec"]
    )
    if duration is not None:
        app.console.print(
            "[bold yellow]提示[/bold yellow] 当前校准时长由 protocol 配置驱动，`--duration` 仅保留兼容性，不改变正式 trial 结构。"
        )
    epochs = int(config["new_subject_epochs"] if is_new else config["old_subject_epochs"])
    acquirer = build_acquirer(
        device_name=str(config["device_type"]),
        config=config,
    )
    effective_n_channels = int(acquirer.metadata.n_channels)
    model_path = build_model_path(
        config,
        subject_id,
        selected_model,
        device_name=str(config["device_type"]),
    )
    model_factory = get_model_factory()
    model = model_factory.get(
        selected_model,
        n_chans=effective_n_channels,
        sfreq=float(config["sfreq"]),
        n_classes=int(config["n_classes"]),
        n_times=int(float(config["sfreq"]) * float(config["window_sec"])),
    )
    calibrator_class = get_calibrator_class()
    calibrator = calibrator_class(
        acquirer=acquirer,
        model=model,
        marker_backend=build_marker_backend(config),
        console=app.console,
        sfreq=float(config["sfreq"]),
        window_sec=float(config["window_sec"]),
        step_sec=float(config["step_sec"]),
        model_path=model_path,
        calibration_records_dir=Path(str(config.get("storage", {}).get("records_dir", "records_storage")))
        / subject_id
        / "calibration",
        protocol_config=ProtocolConfig.from_config(config),
    )
    if is_old:
        calibrator.load_existing_weights()
    result = calibrator.calibrate(
        duration_sec=duration_sec,
        epochs=epochs,
        batch_size=int(config["batch_size"]),
        learning_rate=float(config["learning_rate"]),
        patience=int(config["early_stopping_patience"]),
        head_only=is_old,
    )
    app.console.print(
        f"[bold green]校准完成[/bold green] "
        f"windows={result.windows_collected} "
        f"val_acc={result.metrics.get('val_acc', 0.0):.3f} "
        f"saved={result.model_path}"
    )
    if result.calibration_data_path is not None:
        app.console.print(f"[bold green]校准数据已保存[/bold green] {result.calibration_data_path}")


@cli.command()
@click.option("--subject", "subject_id", required=True, type=str)
@click.option("--model", "model_name", type=str, default=None, help="Model registry name.")
@click.option("--device", "device_name", type=str, default=None, help="Acquirer registry name.")
@click.option("--test-mode", is_flag=True, help="Enable cue-based test mode and save EEG/labels.")
@click.option(
    "--test-duration",
    type=int,
    default=600,
    show_default=True,
    help="Test mode duration in seconds.",
)
@click.option("--record", is_flag=True, help="Record realtime decoding data.")
@click.option(
    "--online-update",
    is_flag=True,
    help="In test mode, update the decoder online using each cue-aligned true label.",
)
@click.option(
    "--online-update-lr",
    type=float,
    default=1e-4,
    show_default=True,
    help="Learning rate for supervised online decoder updates.",
)
@click.option(
    "--online-update-every",
    type=int,
    default=1,
    show_default=True,
    help="Update once every N labeled test-mode windows.",
)
@click.option(
    "--label-source",
    type=click.Choice(["none", "manual-http"]),
    default="none",
    show_default=True,
    help="Realtime true-label source for online updates during normal run.",
)
@click.option(
    "--label-host",
    type=str,
    default="127.0.0.1",
    show_default=True,
    help="Host for the manual realtime-label HTTP server.",
)
@click.option(
    "--label-port",
    type=int,
    default=8776,
    show_default=True,
    help="Port for the manual realtime-label HTTP server.",
)
@click.option(
    "--label-ttl-sec",
    type=float,
    default=2.0,
    show_default=True,
    help="How long each manual label remains active for window alignment.",
)
@click.pass_obj
def run(
    app: AppContext,
    subject_id: str,
    model_name: str | None,
    device_name: str | None,
    test_mode: bool,
    test_duration: int,
    record: bool = False,
    online_update: bool = False,
    online_update_lr: float = 1e-4,
    online_update_every: int = 1,
    label_source: str = "none",
    label_host: str = "127.0.0.1",
    label_port: int = 8776,
    label_ttl_sec: float = 2.0,
) -> None:
    """Run the realtime decoder."""

    config = app.config
    selected_model = model_name or str(config["model_name"])
    selected_device = device_name or str(config["device_type"])
    acquirer = build_acquirer(device_name=selected_device, config=config)
    effective_n_channels = int(acquirer.metadata.n_channels)
    model_path = build_model_path(
        config,
        subject_id,
        selected_model,
        device_name=selected_device,
    )
    if not model_path.exists():
        raise click.ClickException(f"Model not found: {model_path}. Run calibrate first.")
    model_factory = get_model_factory()
    model = model_factory.get(
        selected_model,
        n_chans=effective_n_channels,
        sfreq=float(config["sfreq"]),
        n_classes=int(config["n_classes"]),
        n_times=int(float(config["sfreq"]) * float(config["window_sec"])),
    )
    model.load(model_path)
    online_label_source = None
    online_label_server = None
    if label_source == "manual-http" and not test_mode:
        online_label_source = ManualOnlineLabelSource(default_ttl_sec=label_ttl_sec)
        online_label_server = ManualLabelHttpServer(
            online_label_source,
            host=label_host,
            port=label_port,
        )
        online_label_server.start()
        app.console.print(
            f"[bold cyan]实时标签服务器已启动[/bold cyan] "
            f"http://{label_host}:{label_port}/api/label"
        )

    if online_update and not test_mode and online_label_source is None:
        app.console.print(
            "[bold yellow]提示[/bold yellow] 普通实时 run 需要 --label-source manual-http 才有真标签更新。"
        )
        online_update = False
    command_outlet = LSLCommandOutlet(
        stream_name=str(config["output"]["command_stream_name"]),
        stream_type=str(config["output"]["command_stream_type"]),
    )
    game_command_outlet = build_game_command_outlet(config)
    realtime_decoder_class = get_realtime_decoder_class()
    decoder = realtime_decoder_class(
        acquirer=acquirer,
        model=model,
        console=app.console,
        command_outlet=command_outlet,
        game_command_outlet=game_command_outlet,
        sfreq=float(config["sfreq"]),
        window_sec=float(config["window_sec"]),
        step_sec=float(config["step_sec"]),
        confidence_threshold=float(config["confidence_threshold"]),
        mc_dropout_passes=int(config["mc_dropout_passes"]),
        online_update_enabled=online_update,
        online_update_learning_rate=float(online_update_lr),
        online_update_every=int(online_update_every),
        model_save_path=model_path,
        online_label_source=online_label_source,
    )
    if test_mode:
        marker_backend = build_marker_backend(config)
        records_dir = Path(str(config.get("storage", {}).get("records_dir", "records_storage")))
        result = decoder.run_test_mode(
            subject_id=subject_id,
            marker_backend=marker_backend,
            duration_sec=test_duration,
            block_sec=float(config.get("collect_block_sec", 10)),
            save_dir=records_dir / subject_id / "test_mode",
        )
        app.console.print(
            f"[bold green]测试完成[/bold green] windows={result['windows']} "
            f"accuracy={result['accuracy']:.3f} valid_accuracy={result['valid_accuracy']:.3f}"
        )
        return

    app.console.print("[bold cyan]开始实时解码（无 cue），按 Ctrl+C 停止[/bold cyan]")
    records_dir = Path(str(config.get("storage", {}).get("records_dir", "records_storage")))
    try:
        decoder.run_forever(
            subject_id=subject_id,
            record=record,
            save_dir=records_dir / subject_id / "realtime"
        )
    finally:
        if online_update and not test_mode:
            model.save(model_path)
            app.console.print(f"[bold green]在线更新后的模型已保存[/bold green] {model_path}")
        if online_label_server is not None:
            online_label_server.close()


@cli.command("train-from-records")
@click.option("--subject", "subject_id", required=True, type=str)
@click.option("--model", "model_name", type=str, default=None, help="Model registry name.")
@click.option("--device", "device_name", type=str, default=None, help="Acquirer registry name.")
@click.option(
    "--session",
    "session_ids",
    multiple=True,
    type=str,
    help="Calibration session timestamp to include. Defaults to all sessions.",
)
@click.option(
    "--use-processed/--use-raw",
    default=True,
    show_default=True,
    help="Train on processed_windows or raw_windows from calibration records.",
)
@click.option(
    "--head-only",
    is_flag=True,
    help="Load existing weights first and only fine-tune the classifier head.",
)
@click.pass_obj
def train_from_records(
    app: AppContext,
    subject_id: str,
    model_name: str | None,
    device_name: str | None,
    session_ids: tuple[str, ...],
    use_processed: bool,
    head_only: bool,
) -> None:
    """Train a subject model from saved calibration sessions."""

    config = app.config
    selected_model = model_name or str(config["model_name"])
    selected_device = device_name or str(config["device_type"])
    records_dir = resolve_records_dir(config)
    X, y, used_sessions = load_calibration_windows(
        records_dir,
        subject_id,
        session_ids=session_ids,
        use_processed=use_processed,
    )
    app.console.print(
        f"[bold cyan]加载校准数据[/bold cyan] sessions={len(used_sessions)} "
        f"windows={int(X.shape[0])} shape={tuple(int(dim) for dim in X.shape[1:])}"
    )

    model_path = build_model_path(
        config,
        subject_id,
        selected_model,
        device_name=selected_device,
    )
    model = get_model_factory().get(
        selected_model,
        n_chans=int(X.shape[1]),
        sfreq=float(config["sfreq"]),
        n_classes=int(config["n_classes"]),
        n_times=int(X.shape[2]),
    )
    if head_only:
        if not model_path.exists():
            raise click.ClickException(f"Model not found for head-only adaptation: {model_path}")
        model.load(model_path)

    metrics = model.fit(
        X,
        y,
        epochs=int(config["old_subject_epochs"] if head_only else config["new_subject_epochs"]),
        batch_size=int(config["batch_size"]),
        learning_rate=float(config["learning_rate"]),
        patience=int(config["early_stopping_patience"]),
        head_only=head_only,
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    label_ids, label_counts = np.unique(y, return_counts=True)
    distribution = ", ".join(f"{int(label)}:{int(count)}" for label, count in zip(label_ids, label_counts, strict=False))
    app.console.print(
        f"[bold green]训练完成[/bold green] val_acc={metrics.get('val_acc', 0.0):.3f} "
        f"saved={model_path} labels=[{distribution}]"
    )


@cli.command("replay-test-mode")
@click.option("--subject", "subject_id", required=True, type=str)
@click.option("--model", "model_name", type=str, default=None, help="Model registry name.")
@click.option("--device", "device_name", type=str, default=None, help="Acquirer registry name.")
@click.option(
    "--test-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Override the default records_storage/<subject>/test_mode directory.",
)
@click.pass_obj
def replay_test_mode_command(
    app: AppContext,
    subject_id: str,
    model_name: str | None,
    device_name: str | None,
    test_dir: Path | None,
) -> None:
    """Replay saved test-mode windows through the current decoder."""

    config = app.config
    selected_model = model_name or str(config["model_name"])
    selected_device = device_name or str(config["device_type"])
    resolved_test_dir = test_dir or (resolve_records_dir(config) / subject_id / "test_mode")
    chunk_paths = iter_test_mode_chunks(resolved_test_dir)

    first_chunk = chunk_paths[0]
    with np.load(first_chunk) as payload:
        if "eeg_windows" not in payload:
            raise click.ClickException(f"Invalid test-mode chunk format: {first_chunk}")
        sample_windows = payload["eeg_windows"]
        n_chans = int(sample_windows.shape[1])
        n_times = int(sample_windows.shape[2])

    model_path = build_model_path(
        config,
        subject_id,
        selected_model,
        device_name=selected_device,
    )
    if not model_path.exists():
        raise click.ClickException(f"Model not found: {model_path}. Train or calibrate first.")

    model = get_model_factory().get(
        selected_model,
        n_chans=n_chans,
        sfreq=float(config["sfreq"]),
        n_classes=int(config["n_classes"]),
        n_times=n_times,
    )
    model.load(model_path)
    result = replay_test_mode(
        model=model,
        test_mode_dir=resolved_test_dir,
        sfreq=float(config["sfreq"]),
        mc_dropout_passes=int(config["mc_dropout_passes"]),
    )

    y_true = result["y_true"]
    y_pred = result["y_pred"]
    class_ids = sorted(set(np.unique(y_true).tolist()) | set(np.unique(y_pred).tolist()))
    class_summary = []
    for class_id in class_ids:
        mask = y_true == class_id
        class_acc = float(np.mean(y_pred[mask] == y_true[mask])) if np.any(mask) else 0.0
        class_summary.append(f"{int(class_id)}:{class_acc:.3f}")
    app.console.print(
        f"[bold green]回放完成[/bold green] windows={result['windows']} "
        f"accuracy={result['accuracy']:.3f} mean_confidence={result['mean_confidence']:.3f} "
        f"class_acc=[{', '.join(class_summary)}]"
    )


if __name__ == "__main__":
    cli()
