"""Event detection from positions (Milestone 3A): passes, crosses, shots, goals, key passes.

    from football_fatigue.event_detection import detect
    from football_fatigue.event_detection.inputs import load_positions
    pos = load_positions("data/raw/FIFA World Cup 2022", 3812, cache_dir=None)
    result = detect(pos)                      # result.events: DetectedEvent list, result.touches, result.ball_stats

The detector reads positions and time only (raw ball, raw players); PFF event labels are used for evaluation only.
"""

from __future__ import annotations

from dataclasses import dataclass

from .classify import DetectedEvent, classify
from .config import DetectorConfig
from .inputs import MatchPositions
from .kinematics import ball_track
from .touches import Touch, detect_touches


@dataclass
class DetectionResult:
    events: list[DetectedEvent]
    touches: list[Touch]
    ball_stats: dict[str, int]


def detect(pos: MatchPositions, cfg: DetectorConfig | None = None) -> DetectionResult:
    cfg = cfg or DetectorConfig()
    ball = ball_track(pos, cfg)
    touches = detect_touches(pos, ball, cfg)
    return DetectionResult(classify(pos, ball, touches, cfg), touches, ball.stats)


__all__ = ["DetectedEvent", "DetectionResult", "DetectorConfig", "MatchPositions", "Touch", "detect"]
