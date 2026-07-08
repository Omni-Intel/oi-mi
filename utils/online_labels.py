"""Realtime intent-label sources for online decoder updates."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

LOGGER = logging.getLogger(__name__)

LABEL_NAME_TO_ID = {"left": 0, "right": 1, "idle": 2}
LABEL_ID_TO_NAME = {value: key for key, value in LABEL_NAME_TO_ID.items()}


@dataclass(frozen=True, slots=True)
class OnlineLabel:
    """One realtime label event aligned by local monotonic time."""

    label_id: int
    label_name: str
    timestamp_monotonic: float
    expires_at_monotonic: float
    source: str = "manual"
    payload: dict[str, Any] | None = None

    def is_active_for(self, *, window_start: float, window_end: float) -> bool:
        """Return whether this label overlaps a decoding window."""

        return self.timestamp_monotonic <= window_end and self.expires_at_monotonic >= window_start


class OnlineLabelSource:
    """Base class for optional realtime labels."""

    def get_label(self, *, window_start: float, window_end: float) -> OnlineLabel | None:
        del window_start, window_end
        return None

    def close(self) -> None:
        return


class ManualOnlineLabelSource(OnlineLabelSource):
    """Thread-safe label source updated by an operator or another process."""

    def __init__(self, *, default_ttl_sec: float = 2.0) -> None:
        self._default_ttl_sec = max(float(default_ttl_sec), 0.05)
        self._lock = threading.Lock()
        self._latest: OnlineLabel | None = None

    def set_label(
        self,
        label: str | int,
        *,
        ttl_sec: float | None = None,
        source: str = "manual",
        timestamp_monotonic: float | None = None,
        payload: dict[str, Any] | None = None,
    ) -> OnlineLabel:
        label_id, label_name = coerce_label(label)
        ts = time.monotonic() if timestamp_monotonic is None else float(timestamp_monotonic)
        ttl = self._default_ttl_sec if ttl_sec is None else max(float(ttl_sec), 0.05)
        event = OnlineLabel(
            label_id=label_id,
            label_name=label_name,
            timestamp_monotonic=ts,
            expires_at_monotonic=ts + ttl,
            source=source,
            payload=dict(payload) if payload else None,
        )
        with self._lock:
            self._latest = event
        return event

    def clear(self) -> None:
        with self._lock:
            self._latest = None

    def get_label(self, *, window_start: float, window_end: float) -> OnlineLabel | None:
        with self._lock:
            event = self._latest
        if event is None:
            return None
        if not event.is_active_for(window_start=window_start, window_end=window_end):
            return None
        return event


class ManualLabelHttpServer:
    """Small HTTP server that lets external tools post realtime labels."""

    def __init__(
        self,
        label_source: ManualOnlineLabelSource,
        *,
        host: str = "127.0.0.1",
        port: int = 8776,
    ) -> None:
        self.label_source = label_source
        self.host = host
        self.port = int(port)
        self._httpd = ThreadingHTTPServer((host, self.port), _make_handler(self))
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="oi-mi-label-http-server",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()
        LOGGER.info("Manual label server listening on http://%s:%s", self.host, self.port)

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=1.0)


def coerce_label(label: str | int) -> tuple[int, str]:
    if isinstance(label, int):
        if label not in LABEL_ID_TO_NAME:
            raise ValueError(f"Unsupported label id: {label}")
        return label, LABEL_ID_TO_NAME[label]

    normalized = str(label).strip().lower()
    aliases = {
        "0": "left",
        "1": "right",
        "2": "idle",
        "rest": "idle",
        "stop": "idle",
        "静息": "idle",
        "左": "left",
        "右": "right",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in LABEL_NAME_TO_ID:
        raise ValueError(f"Unsupported label: {label}")
    return LABEL_NAME_TO_ID[normalized], normalized


def _make_handler(runtime: ManualLabelHttpServer) -> type[BaseHTTPRequestHandler]:
    class LabelHandler(BaseHTTPRequestHandler):
        def do_OPTIONS(self) -> None:  # noqa: N802
            self._write_json(HTTPStatus.NO_CONTENT, {})

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/api/label":
                self._write_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})
                return
            now = time.monotonic()
            label = runtime.label_source.get_label(window_start=now, window_end=now)
            self._write_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "label": None if label is None else asdict(label),
                },
            )

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/label":
                self._write_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})
                return
            try:
                payload = self._read_json()
                event = runtime.label_source.set_label(
                    payload.get("label", ""),
                    ttl_sec=payload.get("ttl_sec"),
                    source=str(payload.get("source", "manual")),
                    payload={key: value for key, value in payload.items() if key not in {"label", "ttl_sec", "source"}},
                )
            except Exception as exc:  # noqa: BLE001
                self._write_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
                return
            self._write_json(HTTPStatus.OK, {"ok": True, "label": asdict(event)})

        def do_DELETE(self) -> None:  # noqa: N802
            if self.path != "/api/label":
                self._write_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})
                return
            runtime.label_source.clear()
            self._write_json(HTTPStatus.OK, {"ok": True})

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            LOGGER.debug("label-server " + format, *args)

        def _read_json(self) -> dict[str, Any]:
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise ValueError("Invalid Content-Length") from exc
            raw = self.rfile.read(max(content_length, 0))
            try:
                payload = json.loads(raw.decode("utf-8")) if raw else {}
            except json.JSONDecodeError as exc:
                raise ValueError("Request body must be JSON") from exc
            if not isinstance(payload, dict):
                raise ValueError("Request body must be a JSON object")
            return payload

        def _write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET,POST,DELETE,OPTIONS")
            self.end_headers()
            if self.command != "OPTIONS":
                self.wfile.write(raw)

    return LabelHandler
