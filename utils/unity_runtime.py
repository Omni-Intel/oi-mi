"""Helpers for locating and launching the bundled Unity driving task."""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILD_DIR = Path(".runtime") / "unity" / "ARPrototype3D-windows-x64"
DEFAULT_EXECUTABLE = DEFAULT_BUILD_DIR / "ARPrototype3D.exe"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


def resolve_project_path(path_value: str | os.PathLike[str], *, project_root: Path | None = None) -> Path:
    """Resolve an absolute or project-relative path."""

    root = (project_root or PROJECT_ROOT).resolve()
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def configured_unity_executable(config: dict[str, Any], *, project_root: Path | None = None) -> Path:
    """Return the configured Unity executable path."""

    ar_game_cfg = config.get("output", {}).get("ar_game", {})
    executable_path = str(ar_game_cfg.get("executable_path") or DEFAULT_EXECUTABLE)
    return resolve_project_path(executable_path, project_root=project_root)


def ensure_unity_game_running(
    config: dict[str, Any],
    *,
    console: Any | None = None,
    project_root: Path | None = None,
) -> subprocess.Popen[Any] | None:
    """Launch the Unity game executable when local auto-launch is enabled."""

    ar_game_cfg = config.get("output", {}).get("ar_game", {})
    if not bool(ar_game_cfg.get("enabled", False)):
        return None
    if not bool(ar_game_cfg.get("auto_launch", False)):
        return None

    host = str(ar_game_cfg.get("host", "127.0.0.1")).strip() or "127.0.0.1"
    port = int(ar_game_cfg.get("port", 5005))
    timeout_sec = float(ar_game_cfg.get("startup_timeout_sec", 15.0))

    if not _is_local_host(host):
        _notify(
            console,
            f"Unity auto-launch skipped because AR game host is not local: {host}:{port}",
        )
        return None

    if _is_tcp_open(host, port, timeout_sec=0.25):
        return None

    executable = configured_unity_executable(config, project_root=project_root)
    if not executable.exists():
        raise RuntimeError(
            "Unity game executable was not found: "
            f"{executable}. Run `python setup_local.py` or "
            "`python tools/download_unity_build.py` before realtime decoding."
        )

    _notify(console, f"Launching Unity game: {executable}")
    process = _launch_process(executable)
    if timeout_sec <= 0:
        return process

    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Unity game exited before opening TCP {host}:{port} "
                f"(exit code {process.returncode})."
            )
        if _is_tcp_open(host, port, timeout_sec=0.25):
            return process
        time.sleep(0.25)

    raise RuntimeError(
        f"Unity game did not open TCP {host}:{port} within {timeout_sec:.1f}s. "
        "Start the executable manually once and check Unity logs if this repeats."
    )


def _launch_process(executable: Path) -> subprocess.Popen[Any]:
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

    return subprocess.Popen(
        [str(executable)],
        cwd=str(executable.parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )


def _is_local_host(host: str) -> bool:
    normalized = host.strip().lower()
    if normalized in LOCAL_HOSTS:
        return True
    try:
        return socket.gethostbyname(normalized).startswith("127.")
    except OSError:
        return False


def _is_tcp_open(host: str, port: int, *, timeout_sec: float) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout_sec):
            return True
    except OSError:
        return False


def _notify(console: Any | None, message: str) -> None:
    LOGGER.info(message)
    if console is not None and hasattr(console, "print"):
        console.print(f"[bold cyan]{message}[/bold cyan]")

