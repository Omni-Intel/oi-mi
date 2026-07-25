"""Motor imagery game-control collection protocol helpers."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

LABEL_TO_ID = {"left": 0, "right": 1, "idle": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
LABEL_DISPLAY = {"left": "LEFT", "right": "RIGHT", "idle": "REST"}
LABEL_SYMBOL = {"left": "←", "right": "→", "idle": "○"}
LABEL_DESCRIPTION = {
    "left": "左手持续握拳/松拳想象",
    "right": "右手持续握拳/松拳想象",
    "idle": "放松注视，不发出控制指令",
}

RECOMMENDED_INSTRUCTIONS = [
    "接下来请你根据屏幕提示进行左手想象、右手想象或保持空闲。",
    "左手想象时，请在脑海中感受自己的左手正在持续做重复握拳/松拳，但实际不要动。",
    "右手想象时，请在脑海中感受自己的右手正在持续做重复握拳/松拳，但实际不要动。",
    "idle 时，请保持放松、睁眼看屏幕、不要刻意思考左手或右手动作，也不要真的动身体。",
    "请尽量使用第一人称的感觉自己在动，不要像看电影一样旁观一只手在动。",
    "节奏尽量稳定，像每秒约 1 次握拳-放松即可，不需要刻意数拍子。",
    "全程请尽量减少眨眼、咬牙、耸肩、吞咽、皱眉和手指微动。",
    "如果中间走神，下一次提示开始时重新进入状态即可。",
]


@dataclass(slots=True)
class TrialTiming:
    fixation_sec: float
    cue_sec: float
    control_sec: float
    iti_sec: float

    @property
    def total_sec(self) -> float:
        return self.fixation_sec + self.cue_sec + self.control_sec + self.iti_sec


@dataclass(slots=True)
class BaselineSegment:
    name: str
    duration_sec: float
    instruction: str


@dataclass(slots=True)
class ProtocolConfig:
    window_sec: float
    stride_sec: float
    export_window_sec: float | None
    export_stride_sec: float
    control_start_offset_sec: float
    control_stop_offset_sec: float
    trial_timing: TrialTiming
    practice_labels: list[str]
    practice_repetitions: int
    baseline_segments: list[BaselineSegment]
    calibration_blocks: int
    calibration_trials_per_class_per_block: int
    rest_between_blocks_sec: float
    random_seed: int

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ProtocolConfig:
        protocol = dict(config.get("protocol", {}))
        timing = dict(protocol.get("trial_timing", {}))
        baseline_cfg = list(protocol.get("baseline_segments", []))
        baseline_segments = [
            BaselineSegment(
                name=str(item.get("name", "idle_baseline")),
                duration_sec=float(item.get("duration_sec", 60.0)),
                instruction=str(item.get("instruction", "保持放松并注视屏幕中央。")),
            )
            for item in baseline_cfg
        ]
        if not baseline_segments:
            baseline_segments = [
                BaselineSegment("eyes_open_fixation", 60.0, "睁眼注视中央十字，保持放松。"),
            ]
        return cls(
            window_sec=float(config.get("window_sec", protocol.get("window_sec", 2.0))),
            stride_sec=float(config.get("step_sec", protocol.get("stride_sec", 0.5))),
            export_window_sec=(
                None
                if "export_window_sec" in protocol and protocol.get("export_window_sec") is None
                else float(protocol.get("export_window_sec", 1.5))
            ),
            export_stride_sec=float(protocol.get("export_stride_sec", 0.5)),
            control_start_offset_sec=float(protocol.get("control_start_offset_sec", 0.5)),
            control_stop_offset_sec=float(protocol.get("control_stop_offset_sec", 4.5)),
            trial_timing=TrialTiming(
                fixation_sec=float(timing.get("fixation_sec", 2.0)),
                cue_sec=float(timing.get("cue_sec", 1.0)),
                control_sec=float(timing.get("control_sec", 5.0)),
                iti_sec=float(timing.get("iti_sec", 2.0)),
            ),
            practice_labels=[str(label) for label in protocol.get("practice_labels", ["left", "right", "idle", "left", "right", "idle"])],
            practice_repetitions=int(protocol.get("practice_repetitions", 1)),
            baseline_segments=baseline_segments,
            calibration_blocks=int(protocol.get("calibration_blocks", 4)),
            calibration_trials_per_class_per_block=int(
                protocol.get("calibration_trials_per_class_per_block", 5)
            ),
            rest_between_blocks_sec=float(protocol.get("rest_between_blocks_sec", 20.0)),
            random_seed=int(protocol.get("random_seed", 17)),
        )


@dataclass(slots=True)
class SessionPlan:
    subject_mode: str
    practice_labels: list[str]
    baseline_segments: list[BaselineSegment]
    blocks: list[list[str]]
    rest_between_blocks_sec: float
    trial_timing: TrialTiming

    @property
    def total_formal_trials(self) -> int:
        return sum(len(block) for block in self.blocks)

    @property
    def total_formal_minutes(self) -> float:
        return self.total_formal_trials * self.trial_timing.total_sec / 60.0


def build_session_plan(protocol: ProtocolConfig) -> SessionPlan:
    """Build the single-session, from-scratch calibration plan."""
    rng = random.Random(protocol.random_seed)
    practice_labels = list(protocol.practice_labels) * max(protocol.practice_repetitions, 0)
    blocks = [
        generate_block_sequence(
            {
                "left": protocol.calibration_trials_per_class_per_block,
                "right": protocol.calibration_trials_per_class_per_block,
                "idle": protocol.calibration_trials_per_class_per_block,
            },
            rng=rng,
        )
        for _ in range(protocol.calibration_blocks)
    ]
    return SessionPlan(
        subject_mode="recalibration",
        practice_labels=practice_labels,
        baseline_segments=list(protocol.baseline_segments),
        blocks=blocks,
        rest_between_blocks_sec=protocol.rest_between_blocks_sec,
        trial_timing=protocol.trial_timing,
    )


def generate_block_sequence(class_counts: dict[str, int], *, rng: random.Random, max_attempts: int = 1000) -> list[str]:
    labels = sorted(class_counts)
    total = sum(int(count) for count in class_counts.values())
    for _ in range(max_attempts):
        remaining = {label: int(count) for label, count in class_counts.items()}
        sequence: list[str] = []
        while len(sequence) < total:
            candidates = []
            for label in labels:
                if remaining[label] <= 0:
                    continue
                if len(sequence) >= 2 and sequence[-1] == label and sequence[-2] == label:
                    continue
                candidates.append(label)
            if not candidates:
                break
            candidates.sort(
                key=lambda label: (
                    _first_half_deficit(sequence, class_counts, label),
                    remaining[label],
                    rng.random(),
                ),
                reverse=True,
            )
            chosen = candidates[0]
            sequence.append(chosen)
            remaining[chosen] -= 1
        if len(sequence) != total:
            continue
        if len(set(sequence[:3])) < 2:
            continue
        if not _is_half_balanced(sequence, class_counts):
            continue
        return sequence
    raise RuntimeError(f"Failed to generate a valid block sequence for counts={class_counts}")


def _first_half_deficit(sequence: list[str], class_counts: dict[str, int], candidate: str) -> float:
    midpoint = sum(class_counts.values()) // 2
    if len(sequence) >= midpoint:
        return 0.0
    target = class_counts[candidate] / 2.0
    return target - sequence[:midpoint].count(candidate)


def _is_half_balanced(sequence: list[str], class_counts: dict[str, int]) -> bool:
    midpoint = len(sequence) // 2
    first = sequence[:midpoint]
    second = sequence[midpoint:]
    for label, count in class_counts.items():
        target = count / 2.0
        if abs(first.count(label) - target) > 1:
            return False
        if abs(second.count(label) - target) > 1:
            return False
    return True
