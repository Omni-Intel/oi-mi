from __future__ import annotations

import json
from types import SimpleNamespace

import utils.unity_runtime as unity_runtime


def test_launched_unity_window_is_made_resizable_by_default(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(unity_runtime.os, "name", "nt")
    monkeypatch.setattr(
        unity_runtime,
        "_make_windows_resizable",
        lambda **kwargs: calls.append(kwargs) or True,
    )

    unity_runtime._enable_launched_window_resize(
        SimpleNamespace(pid=1234),
        {},
        console=None,
    )

    assert calls == [{"process_id": 1234}]


def test_launched_unity_window_resize_workaround_can_be_disabled(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(unity_runtime.os, "name", "nt")
    monkeypatch.setattr(
        unity_runtime,
        "_make_windows_resizable",
        lambda **kwargs: calls.append(kwargs) or True,
    )

    unity_runtime._enable_launched_window_resize(
        SimpleNamespace(pid=1234),
        {"resizable_window": False},
        console=None,
    )

    assert calls == []


def test_existing_unity_window_matches_configured_executable_title(
    monkeypatch, tmp_path
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(unity_runtime.os, "name", "nt")
    monkeypatch.setattr(
        unity_runtime,
        "_make_windows_resizable",
        lambda **kwargs: calls.append(kwargs) or True,
    )
    config = {
        "output": {
            "ar_game": {
                "executable_path": "build/MyDrivingGame.exe",
                "resizable_window": True,
            }
        }
    }

    unity_runtime._enable_existing_window_resize(
        config,
        project_root=tmp_path,
        console=None,
    )

    assert calls == [{"window_title": "MyDrivingGame"}]


def test_runtime_manifest_round_trip_and_hash_validation(tmp_path) -> None:
    build_dir = tmp_path / "ARPrototype3D-windows-x64"
    managed_dir = build_dir / "ARPrototype3D_Data" / "Managed"
    managed_dir.mkdir(parents=True)
    executable = build_dir / "ARPrototype3D.exe"
    executable.write_bytes(b"player")
    (build_dir / "UnityPlayer.dll").write_bytes(b"unity")
    managed_dll = managed_dir / "Assembly-CSharp.dll"
    managed_dll.write_bytes(b"managed")
    runtime_dll = managed_dir / "ARPong.Runtime.dll"
    runtime_dll.write_bytes(b"runtime")

    manifest_path = unity_runtime.write_unity_runtime_manifest(
        executable,
        build_id="test-build",
    )
    manifest = unity_runtime.validate_unity_runtime(executable)

    assert manifest_path.name == unity_runtime.RUNTIME_MANIFEST_FILENAME
    assert manifest["build_id"] == "test-build"
    assert manifest["protocol_version"] == unity_runtime.REQUIRED_RUNTIME_PROTOCOL
    assert "ARPrototype3D_Data/Managed/ARPong.Runtime.dll" in manifest["files"]

    runtime_dll.write_bytes(b"tampered")
    try:
        unity_runtime.validate_unity_runtime(executable)
    except RuntimeError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("Tampered Unity runtime unexpectedly validated.")


def test_runtime_manifest_rejects_old_unversioned_build(tmp_path) -> None:
    build_dir = tmp_path / "ARPrototype3D-windows-x64"
    managed_dir = build_dir / "ARPrototype3D_Data" / "Managed"
    managed_dir.mkdir(parents=True)
    executable = build_dir / "ARPrototype3D.exe"
    executable.write_bytes(b"player")
    (build_dir / "UnityPlayer.dll").write_bytes(b"unity")
    (managed_dir / "Assembly-CSharp.dll").write_bytes(b"managed")
    (managed_dir / "ARPong.Runtime.dll").write_bytes(b"runtime")

    try:
        unity_runtime.validate_unity_runtime(executable)
    except RuntimeError as exc:
        assert "manifest was not found" in str(exc)
    else:
        raise AssertionError("Unversioned Unity runtime unexpectedly validated.")


def test_runtime_manifest_requires_runtime_code_hash(tmp_path) -> None:
    build_dir = tmp_path / "ARPrototype3D-windows-x64"
    managed_dir = build_dir / "ARPrototype3D_Data" / "Managed"
    managed_dir.mkdir(parents=True)
    executable = build_dir / "ARPrototype3D.exe"
    executable.write_bytes(b"player")
    (build_dir / "UnityPlayer.dll").write_bytes(b"unity")
    (managed_dir / "Assembly-CSharp.dll").write_bytes(b"managed")
    (managed_dir / "ARPong.Runtime.dll").write_bytes(b"runtime")

    manifest_path = unity_runtime.write_unity_runtime_manifest(
        executable,
        build_id="test-build",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["files"]["ARPrototype3D_Data/Managed/ARPong.Runtime.dll"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    try:
        unity_runtime.validate_unity_runtime(executable)
    except RuntimeError as exc:
        assert "missing critical file hashes" in str(exc)
    else:
        raise AssertionError("Runtime manifest without code hash unexpectedly validated.")
