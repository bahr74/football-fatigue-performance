"""Run discovery and loading for the dashboard (no Streamlit here).

A *run* is any completed folder ``<out_root>/<source_id>/occlusion/<engine version>/<run_id>/`` holding
``manifest.json`` + ``events.jsonl`` + ``frames.jsonl``. The dashboard only reads saved runs; it never runs a
detector. Future uploaded-video runs will use the same folder contract (see docs/video_occlusion_step2.md), with the
video path recorded in the manifest instead of a known development source.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from football_fatigue.video.sources import RESERVED_MATCH_IDS, SOURCES


class RunError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunInfo:
    run_dir: Path
    run_id: str
    source_id: str
    profile: str
    engine_version: str
    window: tuple[int, int]
    rendered: bool
    created_at: str
    tracker_input: str = "B0"

    @property
    def title(self) -> str:
        return f"{self.source_id} · {self.profile} · frames {self.window[0]}–{self.window[1]} · engine {self.engine_version}"


def discover_runs(out_root: Path) -> list[RunInfo]:
    runs = []
    for manifest_path in sorted(Path(out_root).glob("*/occlusion/*/*/manifest.json")):
        run_dir = manifest_path.parent
        if run_dir.name.startswith("INVALID") or run_dir.name.endswith(".partial"):
            continue
        try:
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if m.get("status") != "complete" or m.get("run_id") != run_dir.name:
            continue
        sid = m["source"]["source_id"]
        if any(r in sid for r in RESERVED_MATCH_IDS):
            continue
        runs.append(RunInfo(run_dir, m["run_id"], sid, m.get("profile", "step1"), run_dir.parent.name,
                            (int(m["window"]["start"]), int(m["window"]["end"])),
                            (run_dir / "annotated.mp4").exists(), m.get("created_at_utc", ""),
                            m.get("tracker_input", "B0")))
    runs.sort(key=lambda r: (r.source_id, r.window[1] - r.window[0] < 1000, r.profile != "v2", r.tracker_input != "B0",
                             r.engine_version, r.run_id),
              reverse=False)
    return runs


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@dataclass
class RunData:
    info: RunInfo
    manifest: dict[str, Any]
    events: list[dict[str, Any]]
    rows_by_frame: dict[int, list[dict[str, Any]]]
    rows_by_event: dict[str, list[dict[str, Any]]]
    suppressed: list[dict[str, Any]]
    evaluation: dict[str, Any] | None
    eval_by_event: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def fps(self) -> float:
        return float(self.manifest["source"]["fps"])

    @property
    def pts_offset(self) -> float:
        return float(self.manifest["source"]["pts_offset_s"])

    @property
    def size(self) -> tuple[int, int]:
        return int(self.manifest["source"]["width"]), int(self.manifest["source"]["height"])


def load_run(info: RunInfo) -> RunData:
    m = json.loads((info.run_dir / "manifest.json").read_text(encoding="utf-8"))
    events = _jsonl(info.run_dir / "events.jsonl")
    rows_by_frame: dict[int, list[dict[str, Any]]] = {}
    rows_by_event: dict[str, list[dict[str, Any]]] = {}
    for r in _jsonl(info.run_dir / "frames.jsonl"):
        rows_by_frame.setdefault(int(r["frame"]), []).append(r)
        rows_by_event.setdefault(r["event_id"], []).append(r)
    known = {e["event_id"] for e in events}
    for fr in list(rows_by_frame):                       # older runs may hold rows of suppressed events
        rows_by_frame[fr] = [r for r in rows_by_frame[fr] if r["event_id"] in known]
    evaluation = None
    eval_path = info.run_dir.parent.parent / "evaluation" / f"{info.run_id}.eval-v1.json"
    if eval_path.exists():
        evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
    data = RunData(info, m, events, rows_by_frame, rows_by_event, _jsonl(info.run_dir / "suppressed.jsonl"), evaluation)
    if evaluation:
        data.eval_by_event = {e["event_id"]: e for e in evaluation["events"]}
    return data


def video_path(data: RunData, data_root: Path) -> Path:
    """Development sources: data root + the manifest's relative path. (Uploaded runs will store an absolute path.)"""
    m = data.manifest
    if m.get("video_path_absolute"):
        return Path(m["video_path_absolute"])
    return Path(data_root) / m["video_path_relative"]


def replay_dir(data: RunData, data_root: Path) -> Path:
    """Tracks the run analysed: Session A's production replay (B0) or a Session B tracker-input profile (T0)."""
    m = data.manifest
    if m.get("tracks_dir"):
        return data.info.run_dir.parents[3] / m["tracks_dir"]
    run = m.get("replay_run") or SOURCES.get(data.info.source_id, {}).get("replay_run")
    if not run:
        raise RunError("this run does not name its tracker replay")
    return Path(data_root) / "processed" / "vision" / run


def load_tracks(path: Path, lo: int, hi: int) -> dict[int, dict[int, tuple[tuple[float, ...], float, str]]]:
    """Tracker output per frame: {frame: {track id: (xyxy, confidence, class)}} for frames lo..hi."""
    out: dict[int, dict[int, tuple[tuple[float, ...], float, str]]] = {}
    with (Path(path) / "frames.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            fr = int(rec["frame_num"])
            if fr < lo:
                continue
            if fr > hi:
                break
            out[fr] = {int(p["track_id"]): (tuple(float(v) for v in p["bbox_xyxy_px"]), float(p["confidence"]),
                                            str(p.get("observed_class") or p.get("detected_class")))
                       for p in rec["players"]}
    return out


def load_gt(data_root: Path, source_id: str) -> dict[int, list[tuple[int, tuple[float, ...]]]] | None:
    """SoccerTrack MOT ground truth (development sources only; None when there is none)."""
    spec = SOURCES.get(source_id)
    if not spec or not spec.get("gt"):
        return None
    path = Path(data_root) / spec["gt"]
    if not path.exists():
        return None
    from football_fatigue.vision.mot_eval import load_reference

    gt, _stats = load_reference(path, 0)
    return gt


class LockedReader:
    """ExactFrameReader behind a lock (Streamlit may call from more than one thread)."""

    def __init__(self, path: Path, cache_frames: int = 96) -> None:
        from football_fatigue.video.sources import ExactFrameReader

        self._r = ExactFrameReader(path, cache_frames=cache_frames)
        self._lock = threading.Lock()

    def frame(self, index: int):
        with self._lock:
            return self._r.frame(index)
