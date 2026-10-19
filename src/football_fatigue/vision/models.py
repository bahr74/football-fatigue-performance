"""Data models for Computer Vision Tracking and Detection."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DetectedClass(str, Enum):
    PERSON = "person"  # person detected; football role is not known from the model
    PLAYER = "player"
    GOALKEEPER = "goalkeeper"
    REFEREE = "referee"
    BALL = "ball"


class TeamSide(str, Enum):
    HOME = "home"
    AWAY = "away"
    REFEREE = "referee"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class BoundingBox:
    """Bounding box in normalized or pixel coordinates [x1, y1, x2, y2]."""
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        values = (self.x1, self.y1, self.x2, self.y2)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Bounding box coordinates must be finite.")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("Bounding boxes must have positive width and height.")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Ground contact point for player position (midpoint of bottom edge)."""
        return ((self.x1 + self.x2) / 2.0, self.y2)

    @property
    def center(self) -> tuple[float, float]:
        """Center of bounding box (ideal for ball)."""
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


@dataclass(frozen=True)
class Detection:
    """A raw object detection from a vision model."""
    class_name: DetectedClass
    confidence: float
    bbox: BoundingBox

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.confidence)) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Detection confidence must be finite and within 0-1.")


@dataclass
class TrackedPlayer:
    """A tracked player with persistent ID, team attribution, and pitch coordinates."""
    track_id: int
    team_side: TeamSide
    detected_class: DetectedClass = DetectedClass.PLAYER
    jersey_number: int | None = None
    bbox: BoundingBox | None = None
    confidence: float = 0.0
    pixel_x: float | None = None
    pixel_y: float | None = None
    pitch_x: float | None = None       # in meters, pitch-centre
    pitch_y: float | None = None
    speed_m_s: float | None = None
    is_estimated: bool = False
    is_usable: bool = False
    kit_color: str | None = None        # 'blue' | 'white' | 'other' | 'unknown'; a colour label, not home/away
    kit_confidence: float | None = None
    kit_vote: str | None = None          # raw single-frame colour vote (None = undecided)


@dataclass
class TrackedBall:
    """A tracked ball with pitch coordinates in meters."""
    pixel_x: float | None = None
    pixel_y: float | None = None
    pitch_x: float | None = None
    pitch_y: float | None = None
    pitch_z: float | None = None
    confidence: float = 1.0
    is_estimated: bool = True
    is_usable: bool = False


@dataclass
class VisionFrame:
    """A video tracking frame with optional pitch calibration."""
    frame_num: int
    timestamp_s: float
    players: list[TrackedPlayer] = field(default_factory=list)
    ball: TrackedBall | None = None
    fps: float = 25.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serialisable vision frame record."""
        return {
            "frame_num": self.frame_num,
            "timestamp_s": self.timestamp_s,
            "fps": self.fps,
            "players": [
                {
                    "track_id": p.track_id,
                    "side": p.team_side.value,
                    "jersey": p.jersey_number,
                    "detected_class": p.detected_class.value,
                    "confidence": round(p.confidence, 4),
                    "bbox_xyxy_px": (
                        [p.bbox.x1, p.bbox.y1, p.bbox.x2, p.bbox.y2]
                        if p.bbox is not None else None
                    ),
                    "pixel_x": p.pixel_x,
                    "pixel_y": p.pixel_y,
                    "x": p.pitch_x,
                    "y": p.pitch_y,
                    "speed_m_s": round(p.speed_m_s, 2) if p.speed_m_s is not None else None,
                    "estimated": p.is_estimated,
                    "position_usable": (
                        p.is_usable and p.pitch_x is not None and p.pitch_y is not None
                    ),
                    "kit_color": p.kit_color,
                    "kit_confidence": round(p.kit_confidence, 3) if p.kit_confidence is not None else None,
                    "kit_vote": p.kit_vote,
                    "identity_usable": p.jersey_number is not None,
                    "coord_source": "cv_tracking",
                }
                for p in self.players
            ],
            "ball": (
                {
                    "x": self.ball.pitch_x,
                    "y": self.ball.pitch_y,
                    "z": self.ball.pitch_z,
                    "pixel_x": self.ball.pixel_x,
                    "pixel_y": self.ball.pixel_y,
                    "estimated": self.ball.is_estimated,
                    "position_usable": (
                        self.ball.is_usable
                        and self.ball.pitch_x is not None
                        and self.ball.pitch_y is not None
                    ),
                    "coord_source": "cv_tracking",
                }
                if self.ball
                else None
            ),
        }
