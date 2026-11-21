"""Tracker-input profiles: which cached detections are given to the (unchanged) production ByteTrack.

* ``B0`` - production and historical baseline: person boxes with conf > 0.4 that passed Session A's turf
  (grass-colour) filter. Its tracks are Session A's own replay, ``<data_root>/processed/vision/<replay_run>/``,
  read-only and never rewritten.
* ``T0`` - recommended (``docs/research/VIDEO_DETECTION_DIAGNOSIS_v1.md``): the same boxes and the same threshold, but
  the turf filter's rejection is ignored. Everything else is identical: the same detection cache (YOLO26-L weights,
  tiles, merge NMS), the same ByteTrack implementation and parameters (activation 0.5, buffer 75, match 0.95).

``build`` writes T0 tracks in the production replay format (``frames.jsonl`` + ``manifest.json``) to
``<out_root>/<source>/tracking/T0-tracker-input-v1/`` (write-once), so the occlusion engine, its evaluation and the
dashboards read either profile through the same loader. Nothing here runs a detector.
"""

from __future__ import annotations

import gzip
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from football_fatigue.video.detection.diagnose import iter_cache, replay_path
from football_fatigue.video.detection.replay import REPLAY_VERSION, VARIANTS, select
from football_fatigue.video.sources import PERSON_CLASSES, SOURCES, SourceError, guard_not_reserved

TRACKER_INPUT_VERSION = "tracker-input-v1"
# Tracking profiles = tracker input (which cached detections) x tracker. B0 and T0 use the production ByteTrack; the
# "T0-<tracker>" profiles (Session B benchmark, ``video.bench.trackers``) keep the T0 input and change only the tracker.
TRACKER_INPUTS = {"B0": VARIANTS["B0"], "T0": VARIANTS["T0"], "T0L": VARIANTS["T0L"],
                  "T0-A03": VARIANTS["T0"], "T0L-A03": VARIANTS["T0L"],
                  "T0-UBT": VARIANTS["T0"], "T0-BOT": VARIANTS["T0"], "T0-OCS": VARIANTS["T0"],
                  "T0-DOCS": VARIANTS["T0"], "T0-FT": VARIANTS["T0"], "T0-TT": VARIANTS["T0"]}
TRACKER_BACKEND = {"B0": "sv-bytetrack", "T0": "sv-bytetrack", "T0L": "sv-bytetrack", "T0-A03": "sv-bytetrack",
                   "T0L-A03": "sv-bytetrack", "T0-UBT": "ul-bytetrack", "T0-BOT": "ul-botsort",
                   "T0-OCS": "ul-ocsort", "T0-DOCS": "ul-deepocsort", "T0-FT": "ul-fasttrack", "T0-TT": "ul-tracktrack"}
# M13 activation ablation (research only): same input as T0 / T0L, ByteTrack track_activation_threshold changed from the
# production 0.5. supervision 0.30.9 semantics: boxes >= activation enter the first association, 0.1 < conf < activation
# the second, and a NEW track needs conf >= activation + 0.1 (0.6 in production, 0.4 at activation 0.3).
ACTIVATION = {"T0-A03": 0.3, "T0L-A03": 0.3}
DEFAULT_TRACKER_INPUT = "B0"        # historical reproduction
RECOMMENDED_TRACKER_INPUT = "T0"
DESCRIPTIONS = {
    "B0": "production baseline: person, conf > 0.4, turf (grass-colour) filter applied",
    "T0": "recommended: person, conf > 0.4, turf filter rejection ignored (same detections, same ByteTrack)",
    "T0L": "experimental: turf filter off, tracker input conf > 0.1 (same detections, same ByteTrack)",
    "T0-A03": "research (M13): T0 input, ByteTrack activation 0.3 instead of 0.5 (all other tracker parameters unchanged)",
    "T0L-A03": "research (M13): T0L input, ByteTrack activation 0.3 instead of 0.5 (all other tracker parameters unchanged)",
    "T0-UBT": "T0 input, Ultralytics ByteTrack (research benchmark)",
    "T0-BOT": "T0 input, Ultralytics BoT-SORT without ReID or camera-motion compensation (research benchmark)",
    "T0-OCS": "T0 input, Ultralytics OC-SORT (research benchmark)",
    "T0-DOCS": "T0 input, Ultralytics Deep OC-SORT without appearance features (research benchmark)",
    "T0-FT": "T0 input, Ultralytics FastTracker (occlusion-aware; research benchmark)",
    "T0-TT": "T0 input, Ultralytics TrackTrack without ReID (research benchmark)",
}


def check_name(name: str) -> str:
    if name not in TRACKER_INPUTS:
        raise SourceError(f"unknown tracker input {name!r}; choose one of {sorted(TRACKER_INPUTS)}")
    return name


def tracks_dir(data_root: Path, out_root: Path, source_id: str, name: str) -> Path:
    """Folder holding ``frames.jsonl`` + ``manifest.json`` of this tracker-input profile."""
    check_name(name)
    if name == "B0":
        return replay_path(data_root, source_id).parent
    return Path(out_root) / source_id / "tracking" / f"{name}-{TRACKER_INPUT_VERSION}"


def accepts(observation, name: str) -> bool:
    """Whether a cached observation (``RawObservation``) may be a tracker input under this profile (any confidence)."""
    check_name(name)
    if observation.mapped_class is None or observation.mapped_class.value not in PERSON_CLASSES:
        return False
    if name == "B0":
        return bool(observation.kept)
    return observation.inside_pitch_mask is not False          # T0: only the turf decision is ignored


def _round_trip_check(tracks: dict[int, list], scored: Path) -> dict[str, Any]:
    """Compare with the diagnostic replay that was scored (boxes rounded to 0.01 px there)."""
    if not scored.exists():
        return {"checked": False, "reason": f"{scored} not found"}
    differing = checked = 0
    with gzip.open(scored, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            fr = int(r["frame"])
            if fr not in tracks:
                continue
            checked += 1
            mine = {t[0]: t[1] for t in tracks[fr]}
            theirs = {int(t[0]): t[1:5] for t in r["tracks"]}
            if set(mine) != set(theirs) or any(max(abs(a - b) for a, b in zip(mine[k], theirs[k])) > 0.006 for k in mine):
                differing += 1
    return {"checked": True, "file": str(scored), "frames_checked": checked, "frames_differing": differing,
            "identical": checked > 0 and differing == 0}


def production_tracker_config(data_root: Path, source_id: str) -> dict[str, Any]:
    return json.loads((replay_path(data_root, source_id).parent / "manifest.json").read_text(encoding="utf-8"))["configuration"]


def run_sv_bytetrack(data_root: Path, source_id: str, f, activation: float | None = None):
    """Yield (frame, chosen detections, tracks) of the production supervision ByteTrack on the cached detections.

    ``f`` is the tracker-input filter; ``activation`` overrides only track_activation_threshold (None = production).
    """
    from football_fatigue.vision.models import BoundingBox, DetectedClass, Detection
    from football_fatigue.vision.tracker import MultiObjectTracker

    conf = production_tracker_config(data_root, source_id)
    tp = conf["tracker_params"]
    tracker = MultiObjectTracker(use_supervision=True,
                                 track_activation_threshold=tp["activation"] if activation is None else activation,
                                 track_buffer=tp["buffer_frames"], matching_threshold=tp["matching_threshold"],
                                 role_policy=conf["class_safety"]["role_policy"])
    if tracker.backend != "bytetrack":
        raise RuntimeError("ByteTrack (supervision) is not available")
    tracker.set_frame_rate(float(conf["video_fps"]))
    for fr, dets in iter_cache(data_root, source_id):
        chosen = select(dets, f, None)
        yield fr, chosen, tracker.update([Detection(DetectedClass(d.mapped), d.conf, BoundingBox(*d.box)) for d in chosen], fr)


def build(data_root: Path, out_root: Path, source_id: str, name: str = "T0", overwrite: bool = False) -> Path:
    """Re-run the production ByteTrack on the cached detections with this tracker input (CPU, about 1 min)."""
    check_name(name)
    if name == "B0":
        raise SystemExit("B0 is Session A's production replay; it is read, never rebuilt")
    if TRACKER_BACKEND[name] != "sv-bytetrack":
        return _build_benchmark_profile(data_root, out_root, source_id, name, overwrite)
    spec = SOURCES[source_id]
    guard_not_reserved(source_id, spec["match_id"], spec["cache_run"], spec["replay_run"])
    out = tracks_dir(data_root, out_root, source_id, name)
    if out.exists():
        if not overwrite:
            raise SystemExit(f"output already exists (write-once): {out}")
        shutil.rmtree(out)
    prod_manifest = json.loads((replay_path(data_root, source_id).parent / "manifest.json").read_text(encoding="utf-8"))
    conf = prod_manifest["configuration"]
    tp = conf["tracker_params"]
    f = TRACKER_INPUTS[name]
    activation = ACTIVATION.get(name, tp["activation"])
    fps = float(conf["video_fps"])

    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    t0 = time.perf_counter()
    kept: dict[int, list] = {}
    n_frames = n_boxes = 0
    with (tmp / "frames.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for fr, chosen, tracks in run_sv_bytetrack(data_root, source_id, f, activation):
            players = []
            rows = []
            for t in tracks:
                box = (t.bbox.x1, t.bbox.y1, t.bbox.x2, t.bbox.y2)
                src = next((d for d in chosen if max(abs(a - b) for a, b in zip(d.box, box)) <= 1e-3), None)
                players.append({"track_id": int(t.track_id), "detected_class": t.class_name.value,
                                "observed_class": src.mapped if src is not None else t.class_name.value,
                                "confidence": round(float(t.confidence), 4), "bbox_xyxy_px": [float(v) for v in box],
                                "measured": src is not None})
                rows.append((int(t.track_id), box))
            kept[fr] = rows
            fh.write(json.dumps({"frame_num": fr, "timestamp_s": round(fr / fps, 6), "fps": fps, "players": players},
                                separators=(",", ":")) + "\n")
            n_frames += 1
            n_boxes += len(players)
    seconds = time.perf_counter() - t0
    check = _round_trip_check(kept, Path(out_root) / source_id / "detection" / REPLAY_VERSION / name / "tracks.jsonl.gz")
    manifest = {
        "status": "complete", "kind": "tracker_input_profile", "version": TRACKER_INPUT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_id": source_id,
        "tracker_input": {"name": name, "description": DESCRIPTIONS[name], "filter": f.__dict__},
        "inputs": dict(prod_manifest["inputs"]),
        "configuration": {**{k: conf[k] for k in ("confidence_threshold", "cache_confidence_floor", "image_size",
                                                  "tile_width", "tile_height", "tile_merge", "tracker_params",
                                                  "class_safety", "requested_tracker", "selected_tracker",
                                                  "video_fps", "video_width_px", "video_height_px")},
                          "tracker_params": {**tp, "activation": activation},
                          "production_tracker_params": tp,
                          # the threshold this profile actually fed to ByteTrack (T0: unchanged 0.4); the occlusion
                          # engine verifies every tracked box against the cache with it
                          "confidence_threshold": f.threshold,
                          "production_confidence_threshold": conf["confidence_threshold"],
                          "turf_filter": "ignored" if f.turf == "ignore" else "applied"},
        "derived_from": {"detection_cache_run": spec["cache_run"], "production_replay_run": spec["replay_run"],
                         "unchanged": ["detector weights", "tile inference and merge NMS", "ByteTrack implementation"]
                                      + (["ByteTrack parameters"] if activation == tp["activation"] else
                                         ["ByteTrack parameters other than track_activation_threshold"])
                                      + (["confidence threshold 0.4"] if f.threshold == 0.4 else []),
                         "changed": [f"tracker input: {f.note}"]
                                    + ([] if activation == tp["activation"] else
                                       [f"ByteTrack track_activation_threshold {tp['activation']} -> {activation}"])},
        "summary": {"frames_written": n_frames, "tracked_boxes": n_boxes, "tracker_seconds": round(seconds, 1)},
        "checks": {"diagnostic_replay_equal": check},
        "notes": ["track IDs are temporary ByteTrack IDs, not player identities",
                  "every output box is a measured (cached) detection; ByteTrack's own predictions for lost tracks are "
                  "not exported (same as the production replay)"],
    }
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    tmp.rename(out)
    return out


def _build_benchmark_profile(data_root: Path, out_root: Path, source_id: str, name: str, overwrite: bool) -> Path:
    """Export a benchmark tracker's T0 tracks in the production replay format (measured boxes only)."""
    from football_fatigue.video.bench.trackers import track_source

    spec = SOURCES[source_id]
    guard_not_reserved(source_id, spec["match_id"], spec["cache_run"], spec["replay_run"])
    out = tracks_dir(data_root, out_root, source_id, name)
    if out.exists():
        if not overwrite:
            raise SystemExit(f"output already exists (write-once): {out}")
        shutil.rmtree(out)
    prod_manifest = json.loads((replay_path(data_root, source_id).parent / "manifest.json").read_text(encoding="utf-8"))
    conf = prod_manifest["configuration"]
    fps = float(conf["video_fps"])
    tracks, info = track_source(data_root, source_id, "T0", TRACKER_BACKEND[name], 75)
    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    n_boxes = 0
    with (tmp / "frames.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for fr in sorted(tracks):
            players = [{"track_id": tid, "detected_class": k, "observed_class": k, "confidence": round(c, 4),
                        "bbox_xyxy_px": [float(v) for v in box], "measured": True} for tid, box, c, k in tracks[fr]]
            n_boxes += len(players)
            fh.write(json.dumps({"frame_num": fr, "timestamp_s": round(fr / fps, 6), "fps": fps, "players": players},
                                separators=(",", ":")) + "\n")
    manifest = {
        "status": "complete", "kind": "tracker_profile", "version": TRACKER_INPUT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_id": source_id,
        "tracker_input": {"name": name, "description": DESCRIPTIONS[name], "filter": TRACKER_INPUTS[name].__dict__},
        "tracker": info, "inputs": dict(prod_manifest["inputs"]),
        "configuration": {k: conf[k] for k in ("confidence_threshold", "cache_confidence_floor", "image_size", "tile_width",
                                               "tile_height", "tile_merge", "video_fps", "video_width_px", "video_height_px")}
                         | {"turf_filter": "ignored", "tracker_backend": TRACKER_BACKEND[name]},
        "derived_from": {"detection_cache_run": spec["cache_run"], "production_replay_run": spec["replay_run"],
                         "unchanged": ["detector weights", "tile inference and merge NMS", "confidence threshold 0.4",
                                       "turf filter ignored (T0)"], "changed": ["tracker implementation"]},
        "summary": {"frames_written": len(tracks), "tracked_boxes": n_boxes, "tracker_seconds": info["tracker_seconds"]},
        "checks": {}, "notes": ["track IDs are temporary tracker IDs, not player identities",
                                "every output box is the measured detection the track was matched to in that frame"],
    }
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    tmp.rename(out)
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m football_fatigue.video.tracking",
                                 description="Build tracker-input profile tracks (CPU replay on cached detections)")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-root", default="data/processed/video")
    ap.add_argument("--source", action="append", required=True)
    ap.add_argument("--profile", default="T0", choices=[n for n in TRACKER_INPUTS if n != "B0"])
    a = ap.parse_args(argv)
    for s in a.source:
        out = build(Path(a.data_root), Path(a.out_root), s, a.profile)
        m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        print(json.dumps({"out": str(out), "summary": m["summary"], "check": m.get("checks", {}).get("diagnostic_replay_equal", "n/a (benchmark tracker profile)")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
