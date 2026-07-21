"""Realtime intent-label sources for online decoder updates."""

from __future__ import annotations

import json
import logging
import random
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
    event_id: str = ""
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
        self._event_counter = 0

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
        with self._lock:
            self._event_counter += 1
            event = OnlineLabel(
                label_id=label_id,
                label_name=label_name,
                timestamp_monotonic=ts,
                expires_at_monotonic=ts + ttl,
                event_id=f"manual-{self._event_counter}",
                source=source,
                payload=dict(payload) if payload else None,
            )
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


class SimulatedOnlineLabelSource(OnlineLabelSource):
    """Drive a label-aware dummy acquirer through balanced synthetic trials."""

    def __init__(
        self,
        acquirer: Any,
        *,
        trial_sec: float = 6.0,
        settle_sec: float = 2.0,
        seed: int = 17,
        clock: Any = time.monotonic,
    ) -> None:
        if not hasattr(acquirer, "set_intent"):
            raise TypeError("Simulated labels require an acquirer with set_intent(label).")
        self._acquirer = acquirer
        self._trial_sec = max(float(trial_sec), 1.0)
        self._settle_sec = min(max(float(settle_sec), 0.0), self._trial_sec * 0.8)
        self._clock = clock
        labels = [0, 1, 2]
        random.Random(seed).shuffle(labels)
        self._sequence = tuple(labels)
        self._started_at = float(self._clock())
        self._active_cycle = -1
        self._set_cycle(0)

    def get_label(self, *, window_start: float, window_end: float) -> OnlineLabel | None:
        del window_start
        now = float(self._clock())
        elapsed = max(now - self._started_at, 0.0)
        cycle = int(elapsed // self._trial_sec)
        self._set_cycle(cycle)
        cycle_started = self._started_at + (cycle * self._trial_sec)
        if now - cycle_started < self._settle_sec:
            return None
        label_id = self._sequence[cycle % len(self._sequence)]
        return OnlineLabel(
            label_id=label_id,
            label_name=LABEL_ID_TO_NAME[label_id],
            timestamp_monotonic=cycle_started + self._settle_sec,
            expires_at_monotonic=cycle_started + self._trial_sec,
            event_id=f"sim-{cycle:06d}",
            source="label-aware-dummy",
            payload={"cycle": cycle},
        )

    def _set_cycle(self, cycle: int) -> None:
        if cycle == self._active_cycle:
            return
        label_id = self._sequence[cycle % len(self._sequence)]
        self._acquirer.set_intent(label_id)
        self._active_cycle = cycle


class CuedOnlineLabelSource(OnlineLabelSource):
    """Generate a balanced realtime cue protocol with strict control-window labels."""

    def __init__(
        self,
        sequence: list[str | int],
        *,
        fixation_sec: float,
        cue_sec: float,
        control_sec: float,
        iti_sec: float,
        control_start_offset_sec: float,
        control_stop_offset_sec: float,
        start_delay_sec: float = 5.0,
        clock: Any = time.monotonic,
    ) -> None:
        if not sequence:
            raise ValueError("Cued online protocol requires at least one trial.")
        self._sequence = tuple(coerce_label(label)[0] for label in sequence)
        self._fixation_sec = max(float(fixation_sec), 0.0)
        self._cue_sec = max(float(cue_sec), 0.0)
        self._control_sec = max(float(control_sec), 0.1)
        self._iti_sec = max(float(iti_sec), 0.0)
        self._trial_sec = self._fixation_sec + self._cue_sec + self._control_sec + self._iti_sec
        self._valid_start_offset = min(max(float(control_start_offset_sec), 0.0), self._control_sec)
        self._valid_stop_offset = min(
            max(float(control_stop_offset_sec), self._valid_start_offset),
            self._control_sec,
        )
        self._clock = clock
        self._started_at = float(clock()) + max(float(start_delay_sec), 0.0)

    def get_label(self, *, window_start: float, window_end: float) -> OnlineLabel | None:
        state = self.status(now=window_end)
        if state["phase"] != "control":
            return None
        valid_from = float(state["valid_from_monotonic"])
        valid_until = float(state["valid_until_monotonic"])
        if float(window_start) < valid_from or float(window_end) > valid_until:
            return None
        label_id = int(state["label_id"])
        trial_index = int(state["trial_index"])
        return OnlineLabel(
            label_id=label_id,
            label_name=LABEL_ID_TO_NAME[label_id],
            timestamp_monotonic=valid_from,
            expires_at_monotonic=valid_until,
            event_id=f"cue-{trial_index:06d}",
            source="cued-protocol",
            payload={"trial_index": trial_index, "phase": "control"},
        )

    def status(self, *, now: float | None = None) -> dict[str, Any]:
        timestamp = float(self._clock()) if now is None else float(now)
        elapsed = timestamp - self._started_at
        if elapsed < 0:
            return self._status_payload(
                phase="preparing",
                trial_index=0,
                phase_remaining_sec=-elapsed,
            )
        trial_index = int(elapsed // self._trial_sec)
        if trial_index >= len(self._sequence):
            return self._status_payload(
                phase="done",
                trial_index=len(self._sequence),
                phase_remaining_sec=0.0,
            )

        trial_started = self._started_at + trial_index * self._trial_sec
        within_trial = timestamp - trial_started
        fixation_end = self._fixation_sec
        cue_end = fixation_end + self._cue_sec
        control_end = cue_end + self._control_sec
        if within_trial < fixation_end:
            phase = "fixation"
            phase_end = fixation_end
        elif within_trial < cue_end:
            phase = "cue"
            phase_end = cue_end
        elif within_trial < control_end:
            phase = "control"
            phase_end = control_end
        else:
            phase = "iti"
            phase_end = self._trial_sec
        control_started = trial_started + cue_end
        return self._status_payload(
            phase=phase,
            trial_index=trial_index,
            phase_remaining_sec=max(phase_end - within_trial, 0.0),
            valid_from_monotonic=control_started + self._valid_start_offset,
            valid_until_monotonic=control_started + self._valid_stop_offset,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "source": "cued-protocol",
            "total_trials": len(self._sequence),
            "sequence": [LABEL_ID_TO_NAME[label] for label in self._sequence],
            "timing_sec": {
                "fixation": self._fixation_sec,
                "cue": self._cue_sec,
                "control": self._control_sec,
                "iti": self._iti_sec,
            },
            "valid_control_range_sec": [self._valid_start_offset, self._valid_stop_offset],
        }

    def _status_payload(
        self,
        *,
        phase: str,
        trial_index: int,
        phase_remaining_sec: float,
        valid_from_monotonic: float | None = None,
        valid_until_monotonic: float | None = None,
    ) -> dict[str, Any]:
        active_index = min(trial_index, len(self._sequence) - 1)
        label_id = self._sequence[active_index]
        return {
            "source": "cued-protocol",
            "phase": phase,
            "trial_index": trial_index,
            "trial_number": min(trial_index + 1, len(self._sequence)),
            "total_trials": len(self._sequence),
            "label_id": label_id,
            "label_name": LABEL_ID_TO_NAME[label_id],
            "phase_remaining_sec": float(phase_remaining_sec),
            "valid_from_monotonic": valid_from_monotonic,
            "valid_until_monotonic": valid_until_monotonic,
        }


def build_cued_online_label_source(
    config: dict[str, Any],
    *,
    clock: Any = time.monotonic,
) -> CuedOnlineLabelSource:
    """Build the car experiment's balanced cue source from the project config."""

    from adaptation.mi_protocol import ProtocolConfig, generate_block_sequence

    protocol = ProtocolConfig.from_config(config)
    adaptation = config.get("online_adaptation", {}) or {}
    cue_config = adaptation.get("cued_labels", {}) or {}
    trials_per_class = max(int(cue_config.get("trials_per_class", 32)), 1)
    sequence = generate_block_sequence(
        {label: trials_per_class for label in LABEL_NAME_TO_ID},
        rng=random.Random(int(cue_config.get("random_seed", adaptation.get("random_seed", 17)))),
    )
    timing = protocol.trial_timing
    return CuedOnlineLabelSource(
        sequence,
        fixation_sec=timing.fixation_sec,
        cue_sec=timing.cue_sec,
        control_sec=timing.control_sec,
        iti_sec=timing.iti_sec,
        control_start_offset_sec=protocol.control_start_offset_sec,
        control_stop_offset_sec=protocol.control_stop_offset_sec,
        start_delay_sec=float(cue_config.get("start_delay_sec", 5.0)),
        clock=clock,
    )


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
