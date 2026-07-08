"""Run SSVEP decoding and send penalty-shot commands to the Unity game."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from cli import build_acquirer, load_config, resolve_config_path
from ssvep.detector import DEFAULT_TARGETS, SSVEPDetector
from utils.markers import ArTcpCommandSender


UNITY_COMMANDS = {
    "top_left": "TOP_LEFT",
    "top_right": "TOP_RIGHT",
    "bottom_left": "BOTTOM_LEFT",
    "bottom_right": "BOTTOM_RIGHT",
}

NEURACLE_OCCIPITAL_CHANNELS = "O1,Oz,O2,PO3,PO4,PO7,PO8,Pz"

CHANNEL_NAME_TO_INDEX_64 = {
    name.upper(): index
    for index, name in enumerate(
        (
            "Fp1",
            "Fpz",
            "Fp2",
            "AF3",
            "AF4",
            "F7",
            "F5",
            "F3",
            "F1",
            "Fz",
            "F2",
            "F4",
            "F6",
            "F8",
            "FT7",
            "FC5",
            "FC3",
            "FC1",
            "FCz",
            "FC2",
            "FC4",
            "FC6",
            "FT8",
            "T7",
            "C5",
            "C3",
            "C1",
            "Cz",
            "C2",
            "C4",
            "C6",
            "T8",
            "TP7",
            "CP5",
            "CP3",
            "CP1",
            "CPz",
            "CP2",
            "CP4",
            "CP6",
            "TP8",
            "P7",
            "P5",
            "P3",
            "P1",
            "Pz",
            "P2",
            "P4",
            "P6",
            "P8",
            "PO7",
            "PO5",
            "PO3",
            "POz",
            "PO4",
            "PO6",
            "PO8",
            "CB1",
            "O1",
            "Oz",
            "O2",
            "CB2",
        )
    )
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SSVEP penalty controller for the Unity shootout game.")
    parser.add_argument("--config", type=Path, default=None, help="Path to oi-mi config.yaml.")
    parser.add_argument("--device", default=None, help="Override config device_type, e.g. brainco or neuracle.")
    parser.add_argument("--host", default="127.0.0.1", help="Unity TCP host.")
    parser.add_argument("--port", type=int, default=5005, help="Unity TCP port.")
    parser.add_argument("--window-sec", type=float, default=None, help="SSVEP window length.")
    parser.add_argument("--step-sec", type=float, default=None, help="Loop sleep between predictions.")
    parser.add_argument("--min-confidence", type=float, default=None, help="Minimum SSVEP confidence.")
    parser.add_argument("--stability-windows", type=int, default=None, help="Consecutive identical predictions required.")
    parser.add_argument(
        "--channels",
        default=None,
        help="Comma-separated channel names or channel numbers, e.g. O1,Oz,O2,PO3,PO4,PO7,PO8,Pz.",
    )
    parser.add_argument("--debug-scores", action="store_true", help="Print per-target FFT scores each loop.")
    parser.add_argument("--open-game", action="store_true", help="Send OPEN_PENALTY before decoding.")
    return parser.parse_args()


def default_runtime_params(device_name: str, config: dict) -> tuple[float, float, int, float, str | None]:
    if device_name == "neuracle":
        return 3.0, 0.25, 1, 0.25, NEURACLE_OCCIPITAL_CHANNELS

    ssvep_cfg = config.get("ssvep_game", {})
    return (
        float(config.get("window_sec", 2.0)),
        float(config.get("step_sec", 0.5)),
        int(ssvep_cfg.get("stability_windows", 2)),
        float(ssvep_cfg.get("min_confidence", 0.35)),
        None,
    )


def parse_channel_indices(value: str | None) -> tuple[int, ...] | None:
    if not value:
        return None

    indices: list[int] = []
    unknown: list[str] = []
    for raw_token in value.split(","):
        token = raw_token.strip()
        if not token:
            continue
        if token.lstrip("+-").isdigit():
            channel_number = int(token)
            if channel_number == 0:
                indices.append(0)
            elif channel_number > 0:
                indices.append(channel_number - 1)
            else:
                unknown.append(token)
            continue

        index = CHANNEL_NAME_TO_INDEX_64.get(token.upper())
        if index is None:
            unknown.append(token)
        else:
            indices.append(index)

    if unknown:
        known = ", ".join(sorted(CHANNEL_NAME_TO_INDEX_64))
        raise ValueError(f"Unknown SSVEP channel(s): {', '.join(unknown)}. Known names: {known}")
    return tuple(dict.fromkeys(indices)) or None


def format_scores(scores: dict[str, float]) -> str:
    return " ".join(f"{label}={score:.3g}" for label, score in scores.items())


def main() -> None:
    args = parse_args()
    config = load_config(resolve_config_path(args.config))
    device_name = args.device or str(config.get("device_type", "brainco"))
    default_window_sec, default_step_sec, default_stability_windows, default_min_confidence, default_channels = (
        default_runtime_params(device_name, config)
    )
    window_sec = float(args.window_sec or default_window_sec)
    step_sec = float(args.step_sec or default_step_sec)
    stability_windows = int(args.stability_windows or default_stability_windows)
    min_confidence = float(args.min_confidence if args.min_confidence is not None else default_min_confidence)
    channel_indices = parse_channel_indices(args.channels or default_channels)

    acquirer = build_acquirer(device_name=device_name, config=config)
    detector = SSVEPDetector(
        sfreq=float(config["sfreq"]),
        targets=DEFAULT_TARGETS,
        channel_indices=channel_indices,
        stability_windows=stability_windows,
        min_confidence=min_confidence,
    )
    unity = ArTcpCommandSender(args.host, int(args.port), timeout_sec=1.0)

    print(f"Connecting Unity TCP {args.host}:{args.port}")
    if args.open_game:
        unity.push("OPEN_PENALTY")
        time.sleep(0.2)
        unity.push("START")

    print(
        f"Starting {device_name} EEG stream for SSVEP penalty control "
        f"window={window_sec:.2f}s step={step_sec:.2f}s "
        f"stability={stability_windows} min_confidence={min_confidence:.2f} "
        f"channels={channel_indices or 'default'}"
    )
    acquirer.start_stream()
    try:
        while True:
            loop_started = time.perf_counter()
            try:
                window, _ = acquirer.get_chunk(window_sec)
                result = detector.predict(window)
            except Exception as exc:  # noqa: BLE001
                print(f"decode skipped: {exc}")
                time.sleep(step_sec)
                continue

            if result.target is None:
                print(f"no target confidence={result.confidence:.2f}")
            else:
                print(
                    f"target={result.target.direction} "
                    f"freq={result.target.frequency_hz:.0f}Hz "
                    f"confidence={result.confidence:.2f} stable={result.stable}"
                )
                if args.debug_scores:
                    print(f"scores {format_scores(result.scores)}")
                if result.stable:
                    command = UNITY_COMMANDS[result.target.direction]
                    unity.push(command)
                    detector.reset()

            elapsed = time.perf_counter() - loop_started
            time.sleep(max(0.0, step_sec - elapsed))
    except KeyboardInterrupt:
        print("Stopping SSVEP penalty controller")
    finally:
        acquirer.stop_stream()
        unity.close()


if __name__ == "__main__":
    main()
