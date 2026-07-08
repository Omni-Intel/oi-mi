"""Lightweight HTTP bridge from web UI to AR game TCP commands."""

from __future__ import annotations

import json
import logging
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from utils.markers import ArTcpCommandSender

LOGGER = logging.getLogger(__name__)

_ALLOWED_COMMANDS = {"LEFT", "RIGHT", "STOP", "START", "RESTART"}
_SERVER_LOCK = threading.Lock()
_SERVER_INSTANCE: "_ServerRuntime | None" = None


class _ServerRuntime:
    def __init__(
        self,
        *,
        listen_host: str,
        listen_port: int,
        target_host: str,
        target_port: int,
        timeout_sec: float,
    ) -> None:
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.target_host = target_host
        self.target_port = target_port
        self.timeout_sec = timeout_sec
        self.sender = ArTcpCommandSender(target_host, target_port, timeout_sec=timeout_sec)
        self.httpd = ThreadingHTTPServer((listen_host, listen_port), _make_handler(self))
        self.thread = threading.Thread(target=self.httpd.serve_forever, name="oi-mi-web-control", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.sender.close()
        self.thread.join(timeout=1.0)


def _make_handler(runtime: _ServerRuntime) -> type[BaseHTTPRequestHandler]:
    class WebControlHandler(BaseHTTPRequestHandler):
        def do_OPTIONS(self) -> None:  # noqa: N802
            self._write_json(HTTPStatus.NO_CONTENT, {})

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self._write_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})
                return
            self._write_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "target_host": runtime.target_host,
                    "target_port": runtime.target_port,
                    "commands": sorted(_ALLOWED_COMMANDS),
                },
            )

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/command":
                self._write_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})
                return

            try:
                body = self._read_json()
            except ValueError as exc:
                self._write_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
                return

            command = str(body.get("command", "")).strip().upper()
            if command not in _ALLOWED_COMMANDS:
                self._write_json(
                    HTTPStatus.BAD_REQUEST,
                    {"ok": False, "error": f"Unsupported command '{command}'."},
                )
                return

            try:
                runtime.sender.push(command)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Failed to forward web control command '%s': %s", command, exc)
                self._write_json(
                    HTTPStatus.BAD_GATEWAY,
                    {"ok": False, "error": str(exc), "command": command},
                )
                return

            self._write_json(HTTPStatus.OK, {"ok": True, "command": command})

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            LOGGER.debug("web_control %s - %s", self.address_string(), format % args)

        def _read_json(self) -> dict[str, Any]:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length)
            if not raw:
                raise ValueError("Missing request body.")
            try:
                payload = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError("Request body must be valid JSON.") from exc
            if not isinstance(payload, dict):
                raise ValueError("Request body must be a JSON object.")
            return payload

        def _write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
            self.end_headers()
            if self.command != "OPTIONS":
                self.wfile.write(raw)

    return WebControlHandler


def start_web_control_server(config: dict[str, Any]) -> None:
    """Start the web control server once for the current process."""

    web_cfg = config.get("web_control", {})
    if not bool(web_cfg.get("enabled", True)):
        return

    listen_host = str(web_cfg.get("host", "127.0.0.1")).strip() or "127.0.0.1"
    listen_port = int(web_cfg.get("port", 8787))

    game_cfg = config.get("output", {}).get("ar_game", {})
    target_host = str(game_cfg.get("host", "127.0.0.1")).strip() or "127.0.0.1"
    target_port = int(game_cfg.get("port", 5005))
    timeout_sec = float(game_cfg.get("timeout_sec", 1.0))

    global _SERVER_INSTANCE
    with _SERVER_LOCK:
        if _SERVER_INSTANCE is not None:
            return
        runtime = _ServerRuntime(
            listen_host=listen_host,
            listen_port=listen_port,
            target_host=target_host,
            target_port=target_port,
            timeout_sec=timeout_sec,
        )
        runtime.start()
        _SERVER_INSTANCE = runtime

    LOGGER.info(
        "Web control server listening on http://%s:%s -> tcp://%s:%s",
        listen_host,
        listen_port,
        target_host,
        target_port,
    )
