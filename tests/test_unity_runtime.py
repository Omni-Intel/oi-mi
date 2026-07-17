from __future__ import annotations

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
