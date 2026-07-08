"""Download or install the Unity Windows build used by oi-mi."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = "https://github.com/Omni-Intel/oi-mi/releases/latest/download/ARPrototype3D-windows-x64.zip"
DEFAULT_BUILD_NAME = "ARPrototype3D-windows-x64"
DEFAULT_DEST = PROJECT_ROOT / ".runtime" / "unity" / DEFAULT_BUILD_NAME
DEFAULT_CACHE = PROJECT_ROOT / ".runtime" / "downloads" / f"{DEFAULT_BUILD_NAME}.zip"
EXPECTED_EXE = "ARPrototype3D.exe"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=os.environ.get("OI_MI_UNITY_BUILD_URL", DEFAULT_URL),
        help="Unity build zip URL. Defaults to the latest GitHub Release asset.",
    )
    parser.add_argument(
        "--from-local-zip",
        type=Path,
        default=os.environ.get("OI_MI_UNITY_BUILD_ZIP"),
        help="Install from an existing local zip instead of downloading.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help="Destination directory for the extracted Unity build.",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing extracted build.")
    args = parser.parse_args(argv)

    dest = _resolve_under_project(args.dest)
    expected_exe = dest / EXPECTED_EXE
    if expected_exe.exists() and not args.force:
        print(f"Unity build already installed: {expected_exe}")
        return 0

    if dest.exists() and args.force:
        _remove_directory(dest)

    zip_path = _prepare_zip(args)
    _extract_zip(zip_path, dest)
    exe_path = _find_executable(dest)
    print(f"Unity build installed: {exe_path}")
    return 0


def _prepare_zip(args: argparse.Namespace) -> Path:
    DEFAULT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    local_zip = Path(args.from_local_zip).expanduser() if args.from_local_zip else None
    if local_zip is not None:
        local_zip = local_zip.resolve()
        if not local_zip.exists():
            raise SystemExit(f"Local Unity build zip does not exist: {local_zip}")
        if local_zip != DEFAULT_CACHE.resolve():
            shutil.copy2(local_zip, DEFAULT_CACHE)
        print(f"Using local Unity build zip: {DEFAULT_CACHE}")
        return DEFAULT_CACHE

    if DEFAULT_CACHE.exists() and not args.force:
        print(f"Using cached Unity build zip: {DEFAULT_CACHE}")
        return DEFAULT_CACHE

    url = str(args.url).strip()
    if not url:
        raise SystemExit("Unity build URL is empty.")

    print(f"Downloading Unity build: {url}")
    try:
        urllib.request.urlretrieve(url, DEFAULT_CACHE)
    except (urllib.error.URLError, OSError) as exc:
        raise SystemExit(
            "Failed to download Unity build. Publish the zip as a GitHub Release asset "
            f"named {DEFAULT_CACHE.name}, or pass --from-local-zip. Details: {exc}"
        ) from exc
    return DEFAULT_CACHE


def _extract_zip(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(dest)
    except zipfile.BadZipFile as exc:
        raise SystemExit(f"Invalid Unity build zip: {zip_path}") from exc


def _find_executable(dest: Path) -> Path:
    expected = dest / EXPECTED_EXE
    if expected.exists():
        return expected

    candidates = sorted(
        path for path in dest.glob("*.exe")
        if path.name.lower() != "unitycrashhandler64.exe"
    )
    if candidates:
        return candidates[0]

    raise SystemExit(
        f"No Unity executable found in {dest}. Expected {EXPECTED_EXE}; "
        "make sure the zip contains the full Windows build folder."
    )


def _resolve_under_project(path: Path) -> Path:
    resolved = (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise SystemExit(f"Destination must be inside the project: {resolved}") from exc
    return resolved


def _remove_directory(path: Path) -> None:
    resolved = _resolve_under_project(path)
    runtime_root = (PROJECT_ROOT / ".runtime").resolve()
    try:
        resolved.relative_to(runtime_root)
    except ValueError as exc:
        raise SystemExit(f"Refusing to remove a non-runtime directory: {resolved}") from exc
    shutil.rmtree(resolved)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

