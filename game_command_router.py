"""Shared AR game command router with manual web override support."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from utils.markers import ArTcpCommandRelay, ArTcpCommandSender

LOGGER = logging.getLogger(__name__)

_ROUTER_LOCK = threading.Lock()
_ROUTER_INSTANCE: "SharedGameCommandRouter | None" = None


def _build_transport(config: dict[str, Any]) -> Any:
    game_output_cfg = config.get("output", {}).get("ar_game", {})
    enabled = bool(game_output_cfg.get("enabled", False))
    if not enabled:
        return None

    host = str(game_output_cfg.get("host", "127.0.0.1")).strip() or "127.0.0.1"
    port = int(game_output_cfg.get("port", 5005))
    timeout_sec = float(game_output_cfg.get("timeout_sec", 1.0))
    reverse_enabled = bool(game_output_cfg.get("reverse_enabled", False))
    if reverse_enabled:
        downstream_bind_host = str(game_output_cfg.get("reverse_listen_ip", "0.0.0.0")).strip() or "0.0.0.0"
        downstream_bind_port = int(game_output_cfg.get("reverse_listen_port", 5006))
        return ArTcpCommandRelay(
            local_host=host,
            local_port=port,
            downstream_bind_host=downstream_bind_host,
            downstream_bind_port=downstream_bind_port,
            timeout_sec=timeout_sec,
        )
    return ArTcpCommandSender(host=host, port=port, timeout_sec=timeout_sec)


class _GameCommandProxy:
    def __init__(self, router: "SharedGameCommandRouter", *, source: str) -> None:
        self._router = router
        self._source = source

    def push(self, command: str) -> None:
        self._router.push(command, source=self._source)

    def close(self) -> None:
        # Shared transport lives for the process lifetime.
        return


class SharedGameCommandRouter:
    """Arbitrate commands from realtime decoding and manual web control."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._transport = _build_transport(config)
        web_cfg = config.get("output", {}).get("web_control", {})
        self._manual_hold_sec = float(web_cfg.get("manual_override_hold_sec", 0.8))
        self._manual_release_sec = float(web_cfg.get("manual_override_release_sec", 0.25))
        self._manual_override_until = 0.0
        self._last_manual_command = ""
        self._lock = threading.Lock()

    def push(self, command: str, *, source: str) -> None:
        if self._transport is None:
            raise RuntimeError("AR game output is disabled in config.")

        now = time.monotonic()
        with self._lock:
            if source == "web":
                hold_sec = self._manual_release_sec if command == "STOP" else self._manual_hold_sec
                self._manual_override_until = now + max(hold_sec, 0.0)
                self._last_manual_command = command
            elif source == "decoder" and now < self._manual_override_until:
                LOGGER.debug(
                    "Dropped decoder command '%s' because manual override is active for %.3fs",
                    command,
                    self._manual_override_until - now,
                )
                return

            self._transport.push(command)

    def build_proxy(self, *, source: str) -> _GameCommandProxy:
        return _GameCommandProxy(self, source=source)

    def raw_transport(self) -> Any:
        return self._transport


def get_shared_game_command_router(config: dict[str, Any]) -> SharedGameCommandRouter:
    global _ROUTER_INSTANCE
    with _ROUTER_LOCK:
        if _ROUTER_INSTANCE is None:
            _ROUTER_INSTANCE = SharedGameCommandRouter(config)
        return _ROUTER_INSTANCE
