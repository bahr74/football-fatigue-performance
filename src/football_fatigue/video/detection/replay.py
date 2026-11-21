"""CPU tracker replay of tracker-input filter variants on identical cached detections.

The production replay feeds ByteTrack the cached person detections that passed the turf filter with confidence > 0.4.
Each variant here changes only that input filter; the detections, the ByteTrack implementation and its parameters
(Session A's ``MultiObjectTracker``, activation 0.5, buffer 75, match 0.95) are unchanged. Variant ``B0`` must reproduce
the production replay exactly (checked by ``baseline_check``).
"""

from __future__ import annotations

import gzip
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from football_fatigue.video.detection.diagnose import iter_cache, iter_replay, replay_path
from football_fatigue.video.detection.stages import Det, outside_distance
from football_fatigue.video.sources import SOURCES, guard_not_reserved, play_area_polygon

REPLAY_VERSION = "replay-v1"


@dataclass(frozen=True)
class InputFilter:
    name: str
    threshold: float = 0.4                  # tracker input: confidence > threshold
    turf: str = "cached"                    # cached = honour the cached turf decision; ignore = keep turf-rejected boxes
    turf_gate: float | None = None          # with turf=cached: the turf decision applies only to conf <= turf_gate
    play_area_heights: float | None = None  # drop boxes whose foot is more than k box heights outside the play area
    note: str = ""


VARIANTS: dict[str, InputFilter] = {v.name: v for v in (
    InputFilter("B0", note="production: turf filter, conf > 0.4"),
    InputFilter("T0", turf="ignore", note="turf filter off, conf > 0.4"),
    InputFilter("TG6", turf_gate=0.6, note="turf filter only for conf <= 0.6, conf > 0.4"),
    InputFilter("C3", threshold=0.3, note="turf filter, conf > 0.3"),
    InputFilter("T0C3", threshold=0.3, turf="ignore", note="turf filter off, conf > 0.3"),
    InputFilter("T0P", turf="ignore", play_area_heights=1.0,
                note="turf filter off, conf > 0.4, drop feet > 1 box height outside the annotated play area"),
    InputFilter("T0C3P", threshold=0.3, turf="ignore", play_area_heights=1.0,
                note="turf filter off, conf > 0.3, play-area filter"),
    InputFilter("TG6C3", threshold=0.3, turf_gate=0.6, note="turf filter only for conf <= 0.6, conf > 0.3"),
    InputFilter("T0L", threshold=0.1, turf="ignore",
                note="turf filter off, every cached box (conf > 0.1) to ByteTrack; boxes <= 0.5 only reach its "
                     "second (low-score) association stage and cannot start tracks"),
)}


def select(dets: list[Det], f: InputFilter, poly: np.ndarray | None) -> list[Det]:
    out = []
    for d in dets:
        if not d.person or not d.conf > f.threshold:
            continue
        if f.turf == "cached" and d.turf is False and (f.turf_gate is None or d.conf <= f.turf_gate):
            continue
        if f.play_area_heights is not None:
            dist = outside_distance(d.box, poly)
            if dist is not None and dist < -f.play_area_heights * (d.box[3] - d.box[1]):
                continue
        out.append(d)
    return out


def run_tracker(data_root: Path, source_id: str, f: InputFilter, max_frames: int | None = None
                ) -> dict[int, list[tuple[int, tuple[float, ...], float, str]]]:
    from football_fatigue.vision.models import BoundingBox, DetectedClass, Detection
    from football_fatigue.vision.tracker import MultiObjectTracker

    spec = SOURCES[source_id]
    rep = json.loads((replay_path(data_root, source_id).parent / "manifest.json").read_text(encoding="utf-8"))
    tp = rep["configuration"]["tracker_params"]
    tracker = MultiObjectTracker(use_supervision=True, track_activation_threshold=tp["activation"],
                                 track_buffer=tp["buffer_frames"], matching_threshold=tp["matching_threshold"],
                                 role_policy=rep["configuration"]["class_safety"]["role_policy"])
    if tracker.backend != "bytetrack":
        raise RuntimeError("ByteTrack (supervision) is not available")
    tracker.set_frame_rate(float(rep["configuration"]["video_fps"]))
    poly = np.asarray(play_area_polygon(Path(data_root) / spec["keypoints"]), dtype=np.float32).reshape(-1, 1, 2)
    out: dict[int, list] = {}
    for n, (fr, dets) in enumerate(iter_cache(data_root, source_id)):
        if max_frames is not None and n >= max_frames:
            break
        chosen = select(dets, f, poly)
        tracks = tracker.update([Detection(DetectedClass(d.mapped), d.conf, BoundingBox(*d.box)) for d in chosen], fr)
        out[fr] = [(t.track_id, (t.bbox.x1, t.bbox.y1, t.bbox.x2, t.bbox.y2), float(t.confidence), t.class_name.value)
                   for t in tracks]
    return out


def baseline_check(data_root: Path, source_id: str, tracks: dict[int, list]) -> dict[str, Any]:
    """Whether a replay equals the production replay (same ids and boxes, tolerance 1e-3 px)."""
    diff_frames = 0
    checked = 0
    for fr, prod in iter_replay(replay_path(data_root, source_id)):
        if fr not in tracks:
            continue
        checked += 1
        mine = {tid: box for tid, box, _c, _k in tracks[fr]}
        if set(mine) != set(prod) or any(max(abs(a - b) for a, b in zip(mine[t], prod[t])) > 1e-3 for t in prod):
            diff_frames += 1
    return {"frames_checked": checked, "frames_differing": diff_frames, "identical": diff_frames == 0 and checked > 0}


def score(data_root: Path, source_id: str, tracks: dict[int, list], on_pitch_only: bool) -> dict[str, Any]:
    from football_fatigue.vision.mot_eval import build_sequence_data, load_reference, standard_metrics

    spec = SOURCES[source_id]
    gt, _ = load_reference(Path(data_root) / spec["gt"], 0)
    poly = np.asarray(play_area_polygon(Path(data_root) / spec["keypoints"]), dtype=np.float32).reshape(-1, 1, 2)
    pred: dict[int, list[tuple[int, tuple[float, ...], float]]] = {}
    for fr, ts in tracks.items():
        keep = []
        for tid, box, conf, _k in ts:
            if on_pitch_only:
                dist = outside_distance(box, poly)
                if dist is not None and dist < -1.0 * (box[3] - box[1]):
                    continue
            keep.append((tid, box, conf))
        pred[fr] = keep
    end = max(tracks) + 1
    return standard_metrics(build_sequence_data(gt, pred, min(tracks), end))


def run_variants(data_root: Path, out_root: Path, source_id: str, names: list[str], max_frames: int | None = None,
                 overwrite: bool = False) -> Path:
    spec = SOURCES[source_id]
    guard_not_reserved(source_id, spec["match_id"], spec["cache_run"], spec["replay_run"])
    base = Path(out_root) / source_id / "detection" / REPLAY_VERSION
    base.mkdir(parents=True, exist_ok=True)
    for name in names:
        f = VARIANTS[name]
        out = base / name
        if out.exists():
            if not overwrite:
                continue
            shutil.rmtree(out)
        tmp = base / (name + ".partial")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir()
        t0 = time.perf_counter()
        tracks = run_tracker(data_root, source_id, f, max_frames)
        t_track = time.perf_counter() - t0
        with gzip.open(tmp / "tracks.jsonl.gz", "wt", encoding="utf-8") as fh:
            for fr in sorted(tracks):
                fh.write(json.dumps({"frame": fr, "tracks": [[tid, *[round(v, 2) for v in box], round(c, 4), k]
                                                             for tid, box, c, k in tracks[fr]]}) + "\n")
        result = {
            "replay_version": REPLAY_VERSION, "source_id": source_id, "role": spec["role"], "variant": asdict(f),
            "frames": len(tracks), "cache_run": spec["cache_run"], "tracker_seconds": round(t_track, 1),
            "baseline_check": baseline_check(data_root, source_id, tracks) if name == "B0" else None,
            "metrics_all_tracks": score(data_root, source_id, tracks, on_pitch_only=False),
            "metrics_on_pitch_tracks": score(data_root, source_id, tracks, on_pitch_only=True),
            "scoring_note": "SoccerTrack GT has the 22 players only; referees/staff count as FP in every variant alike. "
                            "on_pitch_tracks drops output boxes whose foot is > 1 box height outside the annotated pitch.",
        }
        (tmp / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        tmp.rename(out)
    return base


def load_tracks(out_root: Path, source_id: str, name: str, lo: int, hi: int) -> dict[int, dict[int, tuple]]:
    """Variant tracker output for frames lo..hi as {frame: {track id: box}} (for the dashboard)."""
    path = Path(out_root) / source_id / "detection" / REPLAY_VERSION / name / "tracks.jsonl.gz"
    out: dict[int, dict[int, tuple]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["frame"] < lo:
                continue
            if r["frame"] > hi:
                break
            out[r["frame"]] = {int(t[0]): tuple(t[1:5]) for t in r["tracks"]}
    return out
