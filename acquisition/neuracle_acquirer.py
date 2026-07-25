"""Neuracle/JellyFish acquisition backend based on the legacy collect code."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from acquisition.base import AbstractAcquirer, AcquirerMetadata, EEGChunk
from utils.preprocessing import resample_eeg

LOGGER = logging.getLogger(__name__)


class NeuracleAcquirer(AbstractAcquirer):
    """Wrap `collect.neuracle_api.DataServerThread` behind the unified acquirer API."""

    def __init__(
        self,
        sfreq: float = 200.0,
        n_channels: int = 59,
        buffer_sec: float = 60.0,
        neuracle_host: str = "127.0.0.1",
        neuracle_port: int = 8712,
        ready_timeout_sec: float = 15.0,
        source_sfreq: float = 250.0,
        transport_delay_sec: float = 0.0,
    ) -> None:
        from collect.neuracle_api import DataServerThread

        self.metadata = AcquirerMetadata(
            name="neuracle",
            sfreq=sfreq,
            n_channels=n_channels,
            timestamp_domain="monotonic",
        )
        self._host = neuracle_host
        self._port = neuracle_port
        self._ready_timeout_sec = ready_timeout_sec
        self.source_sfreq = float(source_sfreq)
        self._sample_rate = int(round(self.source_sfreq))
        self._buffer_sec = buffer_sec
        self._transport_delay_sec = max(float(transport_delay_sec), 0.0)
        self._server: DataServerThread | None = None
        self._device_clock_offset_sec: float | None = None
        self._last_device_end_ms: float | None = None
        self._device_timestamp_wraps = 0
        self._last_timing_diagnostics: dict[str, float] = {}

    def start_stream(self) -> None:
        from collect.neuracle_api import DataServerThread

        if self._server is not None:
            # Defensive cleanup to avoid leaking a previous connection state.
            self.stop_stream()

        self._server = DataServerThread(sample_rate=self._sample_rate, t_buffer=self._buffer_sec)
        self._device_clock_offset_sec = None
        self._last_device_end_ms = None
        self._device_timestamp_wraps = 0
        self._last_timing_diagnostics = {}
        not_connected = self._server.connect(hostname=self._host, port=self._port)
        if not_connected:
            self._server = None
            raise RuntimeError("Could not connect to JellyFish/Neuracle forwarder")
        started = time.monotonic()
        while not self._server.isReady():
            if time.monotonic() - started > self._ready_timeout_sec:
                self.stop_stream()
                raise RuntimeError(
                    "Timed out waiting for Neuracle stream metadata. "
                    "Check JellyFish forwarding status and sample-rate settings."
                )
            time.sleep(0.1)
        self._server.start()
        detected_channels = int(getattr(self._server, "n_chan", 0))
        module_name = str(getattr(self._server, "moduleName", "unknown"))
        detected_rates = np.asarray(getattr(self._server, "srates", []), dtype=np.float64).reshape(-1)
        selected_rates = detected_rates[: self.metadata.n_channels]
        detected_sfreq = (
            float(selected_rates[0])
            if selected_rates.size
            else float(getattr(self._server, "sample_rate", self.source_sfreq))
        )
        LOGGER.info(
            "Neuracle metadata ready: module=%s channels=%s sfreq=%.1fHz",
            module_name,
            detected_channels if detected_channels else self.metadata.n_channels,
            detected_sfreq,
        )
        if detected_channels and self.metadata.n_channels > detected_channels:
            raise RuntimeError(
                f"Configured channels={self.metadata.n_channels} exceeds forwarded channels={detected_channels}"
            )
        if selected_rates.size and (
            selected_rates.size < self.metadata.n_channels
            or not np.allclose(selected_rates, self.source_sfreq)
        ):
            self.stop_stream()
            unique_rates = sorted({float(value) for value in selected_rates})
            raise RuntimeError(
                "Neuracle source sampling rate does not match configuration: "
                f"detected={unique_rates}, configured={self.source_sfreq:.1f}Hz. "
                "Set device.neuracle_source_sfreq to the hardware forwarding rate."
            )
        LOGGER.info("Neuracle acquisition started at %s:%s", self._host, self._port)

    def stop_stream(self) -> None:
        if self._server is None:
            return

        server = self._server
        self._last_timing_diagnostics.update(
            {
                "received_packets": float(getattr(server, "packet_count", 0)),
                "packet_loss_count": float(
                    getattr(server, "packet_loss_count", 0)
                ),
                "total_source_samples": float(
                    getattr(server, "totalSamplesReceived", 0)
                ),
            }
        )
        self._server = None
        try:
            server.stop()
        finally:
            # Give the underlying socket thread a short window to exit fully
            # before the next reconnect attempt.
            time.sleep(0.1)
        LOGGER.info("Neuracle acquisition stopped")

    def get_chunk(self, window_sec: float) -> EEGChunk:
        if self._server is None:
            raise RuntimeError("Neuracle stream is not started")
        get_timed_buffer = getattr(self._server, "GetBufferDataWithTiming", None)
        if callable(get_timed_buffer):
            data, timing = get_timed_buffer()
        else:
            data = self._server.GetBufferData()
            timing = None
        if data.ndim != 2:
            raise RuntimeError(f"Unexpected Neuracle buffer shape: {data.shape}")
        if data.shape[0] < self.metadata.n_channels:
            raise RuntimeError(
                f"Forwarded channel count {data.shape[0]} is lower than configured {self.metadata.n_channels}"
            )
        required_source = int(round(window_sec * self.source_sfreq))
        available_source = (
            int(timing.get("total_samples", 0))
            if isinstance(timing, dict)
            else int(data.shape[1])
        )
        if available_source < required_source:
            raise RuntimeError(
                f"Not enough source-rate data in ring buffer: {available_source} < {required_source}"
            )
        raw_eeg = np.asarray(
            data[: self.metadata.n_channels, -required_source:],
            dtype=np.float32,
        )
        eeg = resample_eeg(
            raw_eeg,
            source_sfreq=self.source_sfreq,
            target_sfreq=self.metadata.sfreq,
        )
        required_target = int(round(window_sec * self.metadata.sfreq))
        if eeg.shape[1] != required_target:
            raise RuntimeError(
                f"Resampled Neuracle window has {eeg.shape[1]} points; expected {required_target}."
            )
        window_end = self._resolve_window_end_monotonic(timing)
        timestamps = window_end - (
            np.arange(required_target, 0, -1, dtype=np.float64) / self.metadata.sfreq
        )
        return eeg, timestamps

    def get_new_samples(self) -> EEGChunk:
        if self._server is None:
            raise RuntimeError("Neuracle stream is not started")
        get_timed_update = getattr(self._server, "GetBufferUpdateWithTiming", None)
        if callable(get_timed_update):
            data, timing = get_timed_update()
        else:
            data = self._server.buffer.getUpdate()
            timing = None
        if data.ndim != 2:
            raise RuntimeError(f"Unexpected Neuracle update shape: {data.shape}")
        if data.size == 0:
            return (
                np.empty((self.metadata.n_channels, 0), dtype=np.float32),
                np.empty((0,), dtype=np.float64),
            )
        eeg = np.asarray(data[: self.metadata.n_channels], dtype=np.float32)
        # Incremental reads stay at the hardware rate so calibration events remain
        # aligned to the unmodified continuous recording. Calibrator rescales them
        # when constructing target-rate model windows.
        window_end = self._resolve_window_end_monotonic(timing)
        timestamps = window_end - (
            np.arange(eeg.shape[1], 0, -1, dtype=np.float64) / self.source_sfreq
        )
        return eeg, timestamps

    @property
    def timing_diagnostics(self) -> dict[str, float]:
        """Return the latest source-clock alignment diagnostics."""

        payload = dict(self._last_timing_diagnostics)
        server = self._server
        if server is not None:
            payload["received_packets"] = float(
                getattr(server, "packet_count", 0)
            )
            payload["packet_loss_count"] = float(
                getattr(server, "packet_loss_count", 0)
            )
            payload["total_source_samples"] = float(
                getattr(server, "totalSamplesReceived", 0)
            )
        return payload

    def _resolve_window_end_monotonic(self, timing: object) -> float:
        if not isinstance(timing, dict):
            return time.monotonic() - self._transport_delay_sec

        try:
            device_end_ms = float(timing["device_end_ms"])
            arrival_monotonic = float(timing["arrival_monotonic"])
        except (KeyError, TypeError, ValueError):
            return time.monotonic() - self._transport_delay_sec

        unwrapped_end_ms = self._unwrap_device_timestamp_ms(device_end_ms)
        observed_offset = arrival_monotonic - (unwrapped_end_ms / 1000.0)
        if (
            self._device_clock_offset_sec is None
            or observed_offset < self._device_clock_offset_sec
        ):
            # The lower envelope rejects queueing and scheduling jitter.  A
            # measured fixed transport delay can be supplied separately.
            self._device_clock_offset_sec = observed_offset

        mapped_end = (
            (unwrapped_end_ms / 1000.0)
            + self._device_clock_offset_sec
            - self._transport_delay_sec
        )
        self._last_timing_diagnostics = {
            "packet_arrival_monotonic": arrival_monotonic,
            "window_end_monotonic": mapped_end,
            "queueing_jitter_sec": max(observed_offset - self._device_clock_offset_sec, 0.0),
            "transport_delay_compensation_sec": self._transport_delay_sec,
        }
        return mapped_end

    def _unwrap_device_timestamp_ms(self, timestamp_ms: float) -> float:
        raw = float(timestamp_ms)
        modulus = float(2**32)
        normalized = raw % modulus
        if (
            self._last_device_end_ms is not None
            and normalized < self._last_device_end_ms - (modulus / 2.0)
        ):
            self._device_timestamp_wraps += 1
        self._last_device_end_ms = normalized
        return normalized + (self._device_timestamp_wraps * modulus)

    def save_full_buffer_npy(self, path: Path) -> Path:
        """Persist the current full forwarded buffer for diagnostics."""

        if self._server is None:
            raise RuntimeError("Neuracle stream is not started")

        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self._server.GetBufferData().astype(np.float32))
        return path
