"""Tracker comparison on identical cached detections (CPU; no detector runs here).

Detector: the production YOLO26-L detection cache (unchanged). Tracker inputs (the only detection-side choice):

* ``T0``  - person boxes, conf > 0.4, turf filter ignored (the approved T0 profile; production threshold)
* ``T0L`` - person boxes, conf > 0.1 (every cached person box), turf filter ignored. Two-stage trackers (ByteTrack
            family) use boxes below their high threshold only in their second association; with ``T0`` that stage is
            empty. ``T0L`` is a detector-input change and is compared only within itself.

Trackers (all run with their published defaults; the only overrides are named in the result):

* ``sv-bytetrack`` - supervision ByteTrack through Session A's ``MultiObjectTracker`` = production (activation 0.5,
                     buffer 75, match 0.95, role policy)
* ``ul-bytetrack``, ``ul-botsort``, ``ul-ocsort``, ``ul-deepocsort``, ``ul-fasttrack``, ``ul-tracktrack`` - the
  implementations bundled with the installed Ultralytics 8.4.182 (AGPL-3.0), default YAML, overrides:
  ``track_buffer`` (default 75 frames = production's 3 s), ``gmc_method: none`` (fixed panorama camera; no frames are
  decoded) and ``with_reid: False`` (no appearance model; Deep OC-SORT and TrackTrack therefore run without appearance).

Ultralytics trackers report Kalman-filtered boxes; the output here is the *measured* detection each track was matched
to in that frame (its detection index), as in the production replay. Rows whose Kalman box no longer overlaps that
detection (IoU < 0.3) are dropped and counted. Scoring: the same HOTA/CLEAR/Identity evaluator as ``detection.replay``.
"""

from __future__ import annotations

import gzip
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from football_fatigue.video.detection.diagnose import iter_cache, replay_path
from football_fatigue.video.detection.replay import score
from football_fatigue.video.detection.stages import iou_matrix
from football_fatigue.video.sources import SOURCES, guard_not_reserved

TRACKER_RUNS_VERSION = "tracker-v1"
INPUTS = {"T0": 0.4, "T0L": 0.1}
UL_TRACKERS = {"ul-bytetrack": "bytetrack", "ul-botsort": "botsort", "ul-ocsort": "ocsort",
               "ul-deepocsort": "deepocsort", "ul-fasttrack": "fasttrack", "ul-tracktrack": "tracktrack"}
TRACKERS = ("sv-bytetrack",) + tuple(UL_TRACKERS)
CLS = {"player": 0, "goalkeeper": 1, "referee": 2, "person": 0}
CLS_NAME = {0: "player", 1: "goalkeeper", 2: "referee"}


def run_dir(out_root: Path, source_id: str, inp: str, tracker: str, buffer: int) -> Path:
    return Path(out_root) / "_bench" / TRACKER_RUNS_VERSION / source_id / f"{inp}-{tracker}-b{buffer}"


def _inputs(data_root: Path, source_id: str, inp: str) -> Iterator[tuple[int, list]]:
    thr = INPUTS[inp]
    for fr, dets in iter_cache(data_root, source_id):
        yield fr, [d for d in dets if d.person and d.conf > thr]          # turf decision ignored in both inputs


def _ul_tracker(name: str, buffer: int, fps: float):
    from ultralytics.cfg import get_cfg  # noqa: F401  (import check)
    from ultralytics.trackers.track import TRACKER_MAP
    from ultralytics.utils import IterableSimpleNamespace, YAML
    from ultralytics.utils.checks import check_yaml

    cfg = YAML.load(check_yaml(f"{UL_TRACKERS[name]}.yaml"))
    overrides = {"track_buffer": buffer}
    if "gmc_method" in cfg:
        overrides["gmc_method"] = "none"
    if "with_reid" in cfg:
        overrides["with_reid"] = False
    cfg |= overrides
    args = IterableSimpleNamespace(**cfg)
    tr = TRACKER_MAP[cfg["tracker_type"]](args=args)
    if hasattr(tr, "max_frames_lost"):
        tr.max_frames_lost = buffer                       # Ultralytics scales buffer by frame_rate/30 when given fps
    return tr, {k: v for k, v in cfg.items()}, overrides


def track_source(data_root: Path, source_id: str, inp: str, tracker: str, buffer: int = 75
                 ) -> tuple[dict[int, list], dict[str, Any]]:
    import psutil

    spec = SOURCES[source_id]
    guard_not_reserved(source_id, spec["match_id"], spec["cache_run"])
    prod = json.loads((replay_path(data_root, source_id).parent / "manifest.json").read_text(encoding="utf-8"))
    conf = prod["configuration"]
    fps = float(conf["video_fps"])
    size = (int(conf["video_height_px"]), int(conf["video_width_px"]))
    info: dict[str, Any] = {"tracker": tracker, "input": inp, "input_conf_gt": INPUTS[inp], "buffer_frames": buffer}
    proc = psutil.Process()
    rss0 = peak = proc.memory_info().rss
    out: dict[int, list] = {}
    dropped = 0
    t_track = 0.0
    if tracker == "sv-bytetrack":
        from football_fatigue.vision.models import BoundingBox, DetectedClass, Detection
        from football_fatigue.vision.tracker import MultiObjectTracker

        tp = conf["tracker_params"]
        tr = MultiObjectTracker(use_supervision=True, track_activation_threshold=tp["activation"], track_buffer=buffer,
                                matching_threshold=tp["matching_threshold"], role_policy=conf["class_safety"]["role_policy"])
        tr.set_frame_rate(fps)
        info["config"] = {"implementation": "supervision ByteTrack via Session A MultiObjectTracker", **tp,
                          "buffer_frames": buffer, "role_policy": conf["class_safety"]["role_policy"]}
        for n, (fr, dets) in enumerate(_inputs(data_root, source_id, inp)):
            t0 = time.perf_counter()
            tracks = tr.update([Detection(DetectedClass(d.mapped), d.conf, BoundingBox(*d.box)) for d in dets], fr)
            t_track += time.perf_counter() - t0
            out[fr] = [(int(t.track_id), (t.bbox.x1, t.bbox.y1, t.bbox.x2, t.bbox.y2), float(t.confidence), t.class_name.value)
                       for t in tracks]
            if n % 200 == 0:
                peak = max(peak, proc.memory_info().rss)
    else:
        from ultralytics.engine.results import Boxes

        tr, cfg, overrides = _ul_tracker(tracker, buffer, fps)
        info["config"] = {"implementation": f"ultralytics.trackers ({cfg['tracker_type']})", "yaml_defaults_with_overrides": cfg,
                          "overrides": overrides}
        for n, (fr, dets) in enumerate(_inputs(data_root, source_id, inp)):
            arr = np.array([[*d.box, d.conf, CLS.get(d.mapped, 0)] for d in dets], dtype=np.float32).reshape(-1, 6)
            t0 = time.perf_counter()
            res = tr.update(Boxes(arr, size), None)
            t_track += time.perf_counter() - t0
            rows = []
            if len(res):
                kal = [tuple(r[:4]) for r in res]
                for r, kb in zip(res, kal):
                    idx = int(r[-1])
                    if not 0 <= idx < len(dets):
                        dropped += 1
                        continue
                    d = dets[idx]
                    if iou_matrix([kb], [d.box])[0, 0] < 0.3:
                        dropped += 1
                        continue
                    rows.append((int(r[4]), tuple(d.box), float(d.conf), d.mapped))
            out[fr] = rows
            if n % 200 == 0:
                peak = max(peak, proc.memory_info().rss)
    info |= {"frames": len(out), "tracker_seconds": round(t_track, 2), "ms_per_frame": round(1000 * t_track / max(len(out), 1), 3),
             "rss_increase_mb": round((peak - rss0) / 2**20, 1), "unmeasured_rows_dropped": dropped,
             "output_boxes": sum(len(v) for v in out.values()), "track_ids": len({t[0] for v in out.values() for t in v})}
    return out, info


def recoveries(tracks: dict[int, list], buffer: int) -> dict[str, int]:
    """Track continuations after a gap (the same id output again after missing frames): a tracker-side recovery count."""
    last: dict[int, int] = {}
    rec = {"gaps_recovered_1_5": 0, "gaps_recovered_6_25": 0, "gaps_recovered_26_plus": 0}
    for fr in sorted(tracks):
        for tid, *_ in tracks[fr]:
            if tid in last and fr - last[tid] > 1:
                g = fr - last[tid] - 1
                rec["gaps_recovered_1_5" if g <= 5 else "gaps_recovered_6_25" if g <= 25 else "gaps_recovered_26_plus"] += 1
            last[tid] = fr
    return rec


def run_one(data_root: Path, out_root: Path, source_id: str, inp: str, tracker: str, buffer: int = 75) -> dict[str, Any]:
    dest = run_dir(out_root, source_id, inp, tracker, buffer)
    if (dest / "result.json").exists():
        return json.loads((dest / "result.json").read_text(encoding="utf-8"))
    tmp = dest.with_name(dest.name + ".partial")
    tmp.mkdir(parents=True, exist_ok=True)
    tracks, info = track_source(data_root, source_id, inp, tracker, buffer)
    with gzip.open(tmp / "tracks.jsonl.gz", "wt", encoding="utf-8") as fh:
        for fr in sorted(tracks):
            fh.write(json.dumps({"frame": fr, "tracks": [[tid, *[round(v, 3) for v in box], round(c, 4), k]
                                                         for tid, box, c, k in tracks[fr]]}) + "\n")
    t0 = time.perf_counter()
    result = {"version": TRACKER_RUNS_VERSION, "source_id": source_id, "role": SOURCES[source_id]["role"],
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "detector": "yolo26l-football production cache",
              "cache_run": SOURCES[source_id]["cache_run"], **info, "recoveries": recoveries(tracks, buffer),
              "metrics_all_tracks": score(data_root, source_id, tracks, on_pitch_only=False),
              "metrics_on_pitch_tracks": score(data_root, source_id, tracks, on_pitch_only=True),
              "scoring_note": "same evaluator as detection.replay; SoccerTrack GT has the 22 players only (referees count as FP)"}
    result["scoring_seconds"] = round(time.perf_counter() - t0, 1)
    (tmp / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    if dest.exists():
        raise SystemExit(f"refusing to overwrite {dest}")
    tmp.rename(dest)
    return result


def run_matrix(data_root: Path, out_root: Path, sources: list[str], trackers: list[str], inputs: list[str], buffer: int = 75
               ) -> Iterator[dict[str, Any]]:
    for sid in sources:
        for inp in inputs:
            for t in trackers:
                if t not in TRACKERS:
                    raise SystemExit(f"unknown tracker {t}; choose from {TRACKERS}")
                r = run_one(data_root, out_root, sid, inp, t, buffer)
                h = r["metrics_all_tracks"]
                yield {"source": sid, "input": inp, "tracker": t, "HOTA": h["hota"]["HOTA"], "AssA": h["hota"]["AssA"],
                       "IDF1": h["identity"]["IDF1"], "IDSW": h["clear"]["IDSW"], "Frag": h["clear"]["Frag"],
                       "ms_per_frame": r["ms_per_frame"], "dropped": r["unmeasured_rows_dropped"]}


def load_results(out_root: Path) -> list[dict[str, Any]]:
    root = Path(out_root) / "_bench" / TRACKER_RUNS_VERSION
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.glob("*/*/result.json"))] if root.exists() else []


def load_tracks(out_root: Path, source_id: str, run_name: str, lo: int, hi: int) -> dict[int, list]:
    path = Path(out_root) / "_bench" / TRACKER_RUNS_VERSION / source_id / run_name / "tracks.jsonl.gz"
    out: dict[int, list] = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["frame"] < lo:
                continue
            if r["frame"] > hi:
                break
            out[r["frame"]] = r["tracks"]
    return out
