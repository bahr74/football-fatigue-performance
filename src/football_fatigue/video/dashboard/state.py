"""UI-independent logic for the occlusion review dashboard (frame/time sync, navigation, filters, explanations,
review labels). Kept free of Streamlit so it can be unit-tested."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from football_fatigue.video import overlay as ov

# ---------------------------------------------------------------------------------------------- time <-> frame
# Frame indices are decoded-frame indices (0 = first decoded frame), the index used by the detection cache, the
# tracker replay, the events and SoccerTrack's ground truth. "Clock" time is frame / fps from the first decoded frame;
# the container presentation time adds the file's start offset (SourceInfo.pts_offset_s).


def frame_to_clock(frame: int, fps: float) -> float:
    return frame / fps


def frame_to_pts(frame: int, fps: float, pts_offset_s: float) -> float:
    return round(pts_offset_s + frame / fps, 6)


def clock_to_frame(seconds: float, fps: float, lo: int, hi: int) -> int:
    return clamp(int(round(seconds * fps)), lo, hi)


def format_clock(seconds: float) -> str:
    m, s = divmod(max(0.0, seconds), 60.0)
    return f"{int(m):02d}:{s:05.2f}"


def clamp(frame: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(frame)))


def step(frame: int, delta: int, lo: int, hi: int) -> int:
    return clamp(frame + delta, lo, hi)


def playback_next(frame: int, lo: int, hi: int, stride: int = 1, loop: bool = False) -> tuple[int, bool]:
    """Next frame during playback and whether playback continues (stops at the end unless looping)."""
    nxt = frame + max(1, stride)
    if nxt > hi:
        return (lo, True) if loop else (hi, False)
    return nxt, True


# ---------------------------------------------------------------------------------------------- events
@dataclass(frozen=True)
class Filters:
    categories: tuple[str, ...] = ov.CATEGORIES
    min_frames: int = 1
    max_frames: int | None = None
    evidence_levels: tuple[str, ...] = ("low", "medium", "high")
    uncertainty: str = "all"            # all | only | exclude
    track: int | None = None            # only events with this temporary track ID
    uncertain_row: bool = True          # show uncertain events in the "uncertain" category


def duration_frames(event: Mapping[str, Any]) -> int:
    return int(event["end_frame"]) - int(event["start_frame"]) + 1


def filter_events(events: Iterable[Mapping[str, Any]], f: Filters) -> list[Mapping[str, Any]]:
    out = []
    for e in events:
        if ov.category_of(e, f.uncertain_row) not in f.categories:
            continue
        d = duration_frames(e)
        if d < f.min_frames or (f.max_frames is not None and d > f.max_frames):
            continue
        if e.get("evidence_level") not in f.evidence_levels:
            continue
        if f.uncertainty == "only" and not e.get("uncertain"):
            continue
        if f.uncertainty == "exclude" and e.get("uncertain"):
            continue
        if f.track is not None and f.track not in {int(m["track_id"]) for m in e["members"]}:
            continue
        out.append(e)
    return out


def labels_for(events: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """E1, E2, … in saved event order (the same labels as the rendered video and strips)."""
    return {e["event_id"]: f"E{i}" for i, e in enumerate(events, start=1)}


def focus_frame(event: Mapping[str, Any], where: str = "peak") -> int:
    return int({"start": event["start_frame"], "end": event["end_frame"]}.get(where, event["peak_frame"]))


def snapshot_frames(event: Mapping[str, Any], lo: int, hi: int, margin: int = 12) -> dict[str, int]:
    """Before / start / peak / end / after frames, clamped to the run window."""
    return {"before": clamp(event["start_frame"] - margin, lo, hi), "start": int(event["start_frame"]),
            "peak": int(event["peak_frame"]), "end": int(event["end_frame"]),
            "after": clamp(event["end_frame"] + margin, lo, hi)}


def adjacent_event(events: Sequence[Mapping[str, Any]], frame: int, direction: int) -> Mapping[str, Any] | None:
    """The next event starting after ``frame`` (direction +1) or the last one starting before it (-1)."""
    ordered = sorted(events, key=lambda e: (e["start_frame"], e["event_id"]))
    if direction > 0:
        return next((e for e in ordered if e["start_frame"] > frame), None)
    prev = [e for e in ordered if e["start_frame"] < frame]
    return prev[-1] if prev else None


def active_at(rows_by_frame: Mapping[int, Sequence[Mapping[str, Any]]], frame: int, visible_ids: set[str] | None = None) -> list[Mapping[str, Any]]:
    rows = list(rows_by_frame.get(frame, ()))
    return rows if visible_ids is None else [r for r in rows if r["event_id"] in visible_ids]


def baseline_matches(event: Mapping[str, Any], baseline: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Baseline (Step 1) events that share a track with ``event`` and overlap it in time."""
    tracks = {int(m["track_id"]) for m in event["members"]}
    return [b for b in baseline if b["end_frame"] >= event["start_frame"] and b["start_frame"] <= event["end_frame"]
            and tracks & {int(m["track_id"]) for m in b["members"]}]


def timeline_items(events: Sequence[Mapping[str, Any]], fps: float, labels: Mapping[str, str], uncertain_row: bool = True) -> list[dict[str, Any]]:
    """One bar per event for the timeline (seconds on the match clock; zero-length events get one frame)."""
    items = []
    for e in events:
        cat = ov.category_of(e, uncertain_row)
        items.append({"event_id": e["event_id"], "label": labels.get(e["event_id"], ""), "category": cat,
                      "row": ov.CATEGORY_LABELS[cat], "colour": ov.CATEGORY_HEX[cat],
                      "start_s": e["start_frame"] / fps, "dur_s": duration_frames(e) / fps,
                      "start_frame": e["start_frame"], "end_frame": e["end_frame"],
                      "text": f"{labels.get(e['event_id'], '')} {e['event_class']}/{e['subtype']}"})
    return items


# ---------------------------------------------------------------------------------------------- explanations
EVIDENCE_TEXT = {
    "ios>=0.2": "the two players' boxes (widened 10% for limbs) overlap by at least 20% of the smaller box",
    "ios>=0.5": "at some frame the overlap reached at least 50% of the smaller box",
    "far_degraded_conf": "while strongly overlapped, the far player's detection confidence dropped (≥ 0.15 below its recent median)",
    "far_degraded_height": "while strongly overlapped, the far player's box shrank (< 85% of its recent height)",
    "hidden_behind_neighbour": "a player stopped being detected exactly where a nearer player's box covers ≥ 50% of the player's predicted box",
    "occluder_box_grew": "the covering player's box grew sharply when the other disappeared (two players merged into one box)",
    "reappeared_same_track": "the hidden player came back with the same temporary track ID",
    "new_track_at_prediction": "a new temporary track appeared at the predicted position (location match only, not an identity check)",
    "not_reappeared": "the lost track did not come back within 25 frames (1 s)",
    "truncated": "the gap reaches the edge of the processed window",
    "untracked_detection_in_gap": "the detection cache had a person box at the predicted spot during the gap (the tracker did not use it)",
    "group_size>=3": "three or more players overlapped at once",
    "detector_miss": "no person box at the predicted position in the detection cache: the detector missed the player",
    "tracker_gap": "a person box existed at the predicted position in ≥ half of the gap frames but the tracker did not output it",
    "near_neighbour_uncertain": "the predicted box was only partly covered by a neighbour (20–50%): too little for occlusion, too much to rule it out",
    "left_frame": "the predicted box moved out of the image",
}

LEVEL_TEXT = {"high": "high: strong, consistent cues (e.g. hidden behind a neighbour and came back)",
              "medium": "medium: consistent cues without confirmation",
              "low": "low: at least one reason for doubt (listed below)"}

GT_OUTCOME_TEXT = {
    "tp": "matches a ground-truth episode of the same kind for the same two players",
    "fp_gt_overlap_only": "ground truth shows these players overlapping (20–50%) but never ≥ 50% occluded",
    "fp_no_gt_overlap": "ground truth shows no overlap for these two players at this time",
    "fp_duplicate_track": "both tracks belong to the same ground-truth player (duplicate box/track)",
    "non_gt_person": "involves someone who is not one of the 22 ground-truth players (referee, staff, spectator or a false detection): not scored",
    "true_loss": "the player is in the ground truth and not covered by another player: a genuine detection/tracking loss",
    "actually_occluded_in_gt": "ground truth shows this player ≥ 50% covered during the gap: really an occlusion",
    "gt_player_absent": "the ground-truth player is absent during the gap",
}


def explain(event: Mapping[str, Any]) -> list[str]:
    """Why the event fired, in plain language (one line per evidence code)."""
    lines = []
    for code in event.get("evidence", []):
        lines.append(EVIDENCE_TEXT.get(code, code))
    if event["event_class"] == "box_overlap":
        lines.append("classified as box overlap because no player was lost or degraded: both stayed detected")
    elif event["event_class"] == "probable_occlusion" and event["subtype"] == "partial":
        lines.append("classified as partial occlusion from detector-confidence/box-size cues only (no player disappeared)")
    return lines


# ---------------------------------------------------------------------------------------------- reviews
REVIEW_LABELS = ("correct", "incorrect", "uncertain")


class ReviewStore:
    """Human review labels: an append-only JSONL file, separate from the engine output (never modifies a run)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def append(self, run_id: str, event_id: str, label: str, note: str = "", reviewer: str = "local") -> dict[str, Any]:
        if label not in REVIEW_LABELS:
            raise ValueError(f"label must be one of {REVIEW_LABELS}")
        rec = {"review_id": uuid.uuid4().hex, "run_id": run_id, "event_id": event_id, "label": label,
               "note": note.strip(), "reviewer": reviewer, "created_at_utc": datetime.now(timezone.utc).isoformat()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue   # a torn last line never blocks the dashboard
        return out

    def latest(self, run_id: str | None = None) -> dict[str, dict[str, Any]]:
        """Latest label per event (later records win)."""
        out: dict[str, dict[str, Any]] = {}
        for r in self.load():
            if run_id is None or r.get("run_id") == run_id:
                out[r["event_id"]] = r
        return out
