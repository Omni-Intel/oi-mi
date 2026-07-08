"""Create the local Python environment and install required runtime assets."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"


def main() -> int:
    _require_python_312()
    _ensure_venv()
    venv_python = _venv_python()

    _run([venv_python, "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"])
    _run([venv_python, "-m", "pip", "install", "-e", "."])

    unity_command = [venv_python, str(PROJECT_ROOT / "tools" / "download_unity_build.py")]
    local_unity_zip = os.environ.get("OI_MI_UNITY_BUILD_ZIP")
    if local_unity_zip:
        unity_command.extend(["--from-local-zip", local_unity_zip])
    _run(unity_command)

    _run([venv_python, str(PROJECT_ROOT / "tools" / "check_environment.py")])
    _print_next_steps()
    return 0


def _require_python_312() -> None:
    if sys.version_info[:2] != (3, 12):
        raise SystemExit(
            "setup_local.py must be run with Python 3.12.x. "
            "On Windows Git Bash use: py -3.12 setup_local.py"
        )


def _ensure_venv() -> None:
    if _venv_python().exists():
        print(f"Using existing venv: {VENV_DIR}")
        return
    print(f"Creating venv: {VENV_DIR}")
    _run([Path(sys.executable), "-m", "venv", str(VENV_DIR)])


def _venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _run(command: list[Path | str]) -> None:
    printable = " ".join(str(part) for part in command)
    print(f"+ {printable}")
    subprocess.check_call([str(part) for part in command], cwd=PROJECT_ROOT)


def _print_next_steps() -> None:
    if os.name == "nt":
        activate = "source .venv/Scripts/activate"
    else:
        activate = "source .venv/bin/activate"

    print("")
    print("Setup complete.")
    print("Next commands:")
    print(f"  {activate}")
    print("  streamlit run gui.py")
    print("")
    print("Alternative CLI entrypoint:")
    print("  oi-mi gui")


if __name__ == "__main__":
    raise SystemExit(main())

