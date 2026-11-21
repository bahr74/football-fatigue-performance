"""Versioned record types for occlusion events.

``football_fatigue.video.occlusion_event`` 1.x rules:

* additive fields bump the minor version; readers ignore unknown fields;
* renaming or retyping a field bumps the major version, and :func:`event_from_json` refuses a
  different major version;
* human review labels are never stored in an event (they belong in a separate, append-only file).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any

SCHEMA_NAME = "football_fatigue.video.occlusion_event"
SCHEMA_VERSION = "1.1.0"   # 1.1.0 adds group_id, flags, suppressed_reason (readers of 1.0 ignore them)

EVENT_CLASSES = ("box_overlap", "probable_occlusion", "detection_loss")
SUBTYPES = {
    "box_overlap": ("overlap",),
    "probable_occlusion": ("partial", "full", "merged"),
    # detector_miss: no cached person box at the predicted spot; tracker_gap: a cached box was there but the
    # tracker did not output it; near_neighbour_uncertain: partly covered, between the clear and hidden thresholds.
    "detection_loss": ("detector_miss", "tracker_gap", "left_frame", "near_neighbour_uncertain"),
}
EVIDENCE_LEVELS = ("low", "medium", "high")


class SchemaError(ValueError):
    """A record does not follow the occlusion-event schema."""


@dataclass(frozen=True)
class SourceInfo:
    source_id: str
    match_id: str
    camera_id: str
    video_sha256: str
    fps: float
    pts_offset_s: float  # container presentation time of decoded frame 0
    width: int
    height: int

    def pts(self, frame: int) -> float:
        return round(self.pts_offset_s + frame / self.fps, 6)


@dataclass
class Member:
    track_id: int
    role: str                       # "near" | "far" | "involved" | "lost"
    first_frame: int
    last_frame: int
    hidden: list[list[int]] = field(default_factory=list)   # [first, last] frames hidden (not detected)
    outcome: str | None = None      # for hidden members: reappeared_same_track | new_track_at_prediction | not_reappeared | truncated


@dataclass
class OcclusionEvent:
    event_id: str
    run_id: str
    source: SourceInfo
    event_class: str
    subtype: str
    start_frame: int
    end_frame: int
    start_pts_s: float
    end_pts_s: float
    peak_frame: int
    members: list[Member]
    parent_event_id: str | None
    measurements: dict[str, Any]
    evidence: list[str]
    evidence_level: str
    uncertain: bool
    uncertainty_reasons: list[str]
    pixel_xy: list[float]           # bottom centre of the members' union box at the peak frame
    pitch: dict[str, Any]
    truncated: str | None           # None | "processing_start" | "processing_end"
    provenance: dict[str, Any]
    schema: str = SCHEMA_NAME
    schema_version: str = SCHEMA_VERSION
    # 1.1.0 (additive): grouping and scope
    group_id: str | None = None              # links concurrent pair events sharing a track (engine profile v2)
    flags: list[str] = field(default_factory=list)   # e.g. involves_referee_label, off_pitch_member
    suppressed_reason: str | None = None     # set only on events written to suppressed.jsonl

    def validate(self) -> None:
        if self.event_class not in EVENT_CLASSES:
            raise SchemaError(f"unknown event_class {self.event_class!r}")
        if self.subtype not in SUBTYPES[self.event_class]:
            raise SchemaError(f"subtype {self.subtype!r} is not valid for {self.event_class}")
        if self.evidence_level not in EVIDENCE_LEVELS:
            raise SchemaError(f"unknown evidence_level {self.evidence_level!r}")
        if not self.start_frame <= self.peak_frame <= self.end_frame:
            raise SchemaError("frames must satisfy start <= peak <= end")
        if not self.members:
            raise SchemaError("an event needs at least one member")

    def to_json(self) -> str:
        self.validate()
        record = asdict(self)
        ordered = {"schema": record.pop("schema"), "schema_version": record.pop("schema_version"), **record}
        return json.dumps(ordered, sort_keys=False, separators=(",", ":"))


def _major(version: str) -> str:
    return str(version).split(".", 1)[0]


def event_from_json(text: str) -> OcclusionEvent:
    """Parse one JSONL line; unknown fields are ignored, a different major version is refused."""
    record = json.loads(text)
    if record.get("schema") != SCHEMA_NAME:
        raise SchemaError(f"not an occlusion event (schema {record.get('schema')!r})")
    if _major(record.get("schema_version", "")) != _major(SCHEMA_VERSION):
        raise SchemaError(f"unsupported schema_version {record.get('schema_version')!r}; reader is {SCHEMA_VERSION}")
    known = {f.name for f in fields(OcclusionEvent)}
    data = {k: v for k, v in record.items() if k in known}
    src_known = {f.name for f in fields(SourceInfo)}
    mem_known = {f.name for f in fields(Member)}
    data["source"] = SourceInfo(**{k: v for k, v in data["source"].items() if k in src_known})
    data["members"] = [Member(**{k: v for k, v in m.items() if k in mem_known}) for m in data["members"]]
    event = OcclusionEvent(**data)
    event.validate()
    return event
