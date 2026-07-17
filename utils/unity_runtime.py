"""Helpers for locating and launching the bundled Unity driving task."""

from __future__ import annotations

import ctypes
import logging
import os
import socket
import subprocess
import time
from ctypes import wintypes
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILD_DIR = Path("unity相关") / "ARPrototype3D-windows-x64"
DEFAULT_EXECUTABLE = DEFAULT_BUILD_DIR / "ARPrototype3D.exe"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}

# Win32 window-style flags used to compensate for Unity builds made with
# PlayerSettings.resizableWindow disabled.  Importing ctypes is portable; the
# WinDLL calls below are guarded by os.name == "nt".
_GWL_STYLE = -16
_WS_CAPTION = 0x00C00000
_WS_THICKFRAME = 0x00040000
_WS_MAXIMIZEBOX = 0x00010000
_SWP_NOSIZE = 0x0001
_SWP_NOMOVE = 0x0002
_SWP_NOZORDER = 0x0004
_SWP_FRAMECHANGED = 0x0020


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
        _enable_existing_window_resize(config, project_root=project_root, console=console)
        return None

    executable = configured_unity_executable(config, project_root=project_root)
    if not executable.exists():
        raise RuntimeError(
            "Unity game executable was not found: "
            f"{executable}. Run `python setup_local.py` or "
            "`python tools/download_unity_build.py` before realtime decoding."
        )

    _notify(console, f"Launching Unity game: {executable}")
    process = _launch_process(executable, ar_game_cfg)
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
            _enable_launched_window_resize(process, ar_game_cfg, console=console)
            return process
        time.sleep(0.25)

    raise RuntimeError(
        f"Unity game did not open TCP {host}:{port} within {timeout_sec:.1f}s. "
        "Start the executable manually once and check Unity logs if this repeats."
    )


def _launch_process(executable: Path, ar_game_cfg: dict[str, Any]) -> subprocess.Popen[Any]:
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

    args = [str(executable)]
    if bool(ar_game_cfg.get("windowed", True)):
        width = int(ar_game_cfg.get("window_width", 1280))
        height = int(ar_game_cfg.get("window_height", 720))
        args.extend([
            "-screen-fullscreen",
            "0",
            "-screen-width",
            str(max(width, 320)),
            "-screen-height",
            str(max(height, 240)),
        ])

    return subprocess.Popen(
        args,
        cwd=str(executable.parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )


def _enable_existing_window_resize(
    config: dict[str, Any],
    *,
    project_root: Path | None,
    console: Any | None,
) -> None:
    """Make an already-running local Unity window resizable when configured."""

    ar_game_cfg = config.get("output", {}).get("ar_game", {})
    if os.name != "nt" or not bool(ar_game_cfg.get("resizable_window", True)):
        return

    executable = configured_unity_executable(config, project_root=project_root)
    title = str(ar_game_cfg.get("window_title") or executable.stem)
    if _make_windows_resizable(window_title=title):
        _notify(console, f"Enabled resizing for Unity window: {title}")


def _enable_launched_window_resize(
    process: subprocess.Popen[Any],
    ar_game_cfg: dict[str, Any],
    *,
    console: Any | None,
) -> None:
    """Make the newly launched Unity window resizable on Windows."""

    if os.name != "nt" or not bool(ar_game_cfg.get("resizable_window", True)):
        return

    if _make_windows_resizable(process_id=process.pid):
        _notify(console, "Enabled resizing for the Unity game window.")
    else:
        LOGGER.warning(
            "Unity opened TCP successfully, but its top-level window was not found; "
            "the window resize workaround was not applied."
        )


def _make_windows_resizable(
    *,
    process_id: int | None = None,
    window_title: str | None = None,
) -> bool:
    """Add resize/maximize styles to matching top-level Windows windows.

    This is a runtime fallback for a bundled Unity player whose source project
    is unavailable.  A future Unity rebuild should instead enable
    ``PlayerSettings.resizableWindow`` and can then disable this workaround in
    config.
    """

    if os.name != "nt" or (process_id is None and not window_title):
        return False

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    enum_windows_callback = ctypes.WINFUNCTYPE(
        wintypes.BOOL, wintypes.HWND, wintypes.LPARAM
    )

    user32.EnumWindows.argtypes = [enum_windows_callback, wintypes.LPARAM]
    user32.EnumWindows.restype = wintypes.BOOL
    user32.GetWindowThreadProcessId.argtypes = [
        wintypes.HWND,
        ctypes.POINTER(wintypes.DWORD),
    ]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.GetWindowLongW.restype = ctypes.c_long
    user32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
    user32.SetWindowLongW.restype = ctypes.c_long
    user32.SetWindowPos.argtypes = [
        wintypes.HWND,
        wintypes.HWND,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.UINT,
    ]
    user32.SetWindowPos.restype = wintypes.BOOL
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int

    expected_title = window_title.casefold() if window_title else None
    changed = False

    @enum_windows_callback
    def visit_window(hwnd: int, _lparam: int) -> bool:
        nonlocal changed
        if not user32.IsWindowVisible(hwnd):
            return True

        if process_id is not None:
            owner_pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner_pid))
            if owner_pid.value != int(process_id):
                return True

        if expected_title is not None:
            length = user32.GetWindowTextLengthW(hwnd)
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, len(buffer))
            if buffer.value.strip().casefold() != expected_title:
                return True

        style = user32.GetWindowLongW(hwnd, _GWL_STYLE)
        if not style & _WS_CAPTION:
            return True
        new_style = style | _WS_THICKFRAME | _WS_MAXIMIZEBOX
        if new_style != style:
            previous = user32.SetWindowLongW(hwnd, _GWL_STYLE, new_style)
            if previous == 0 and ctypes.get_last_error() != 0:
                LOGGER.warning(
                    "Could not update Unity window style: WinError %s",
                    ctypes.get_last_error(),
                )
                return True

        if not user32.SetWindowPos(
            hwnd,
            0,
            0,
            0,
            0,
            0,
            _SWP_NOMOVE | _SWP_NOSIZE | _SWP_NOZORDER | _SWP_FRAMECHANGED,
        ):
            LOGGER.warning(
                "Could not refresh Unity window frame: WinError %s",
                ctypes.get_last_error(),
            )
            return True
        changed = True
        return True

    user32.EnumWindows(visit_window, 0)
    return changed


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
