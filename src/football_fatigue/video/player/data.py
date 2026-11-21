"""Read-only data for the native video player: catalogue of videos/runs and per-frame overlay chunks.

Layers are kept strictly separate (and labelled as such in the page):

``d``  detector output: every cached post-merge box (conf > 0.1) with its confidence, class and whether the selected
       tracker-input profile passes it to ByteTrack. **Measured.**
``t``  tracker output of the selected profile: the boxes ByteTrack returned, with temporary IDs. Every one of them is a
       measured detection (``measured`` flag checked against the cache when the tracks were built). ByteTrack's own
       predictions for lost tracks are not exported by the production tracker, so there is no "tracker predicted" box.
``e``  occlusion engine (selected run): **predicted** positions of hidden / lost tracks (state hidden_predicted /
       lost_predicted), with the event label.
``g``  SoccerTrack ground truth (development matches only).

Raw per-tile detector boxes (before the tile merge) were never stored, so they cannot be shown.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from football_fatigue.video import overlay as ov
from football_fatigue.video.dashboard import data as D
from football_fatigue.video.dashboard import state as S
from football_fatigue.video.detection.diagnose import cache_dir
from football_fatigue.video.detection.framestore import LineIndex
from football_fatigue.video.sources import RESERVED_MATCH_IDS, SOURCES, guard_not_reserved
from football_fatigue.video.tracking import DESCRIPTIONS, RECOMMENDED_TRACKER_INPUT, TRACKER_INPUTS, tracks_dir

CHUNK = 250
WADI_RAW = Path("external/wadi-degla/raw")
WADI_DETECTIONS = Path("processed/vision_eval/m108-unified-multicamera-analytics-v1/results/detections")
WADI_VIDEOS = {"wadi_degla_moving": "wadi_degla_moving.mp4", "wadi_degla_fixed_same_match": "wadi_degla_fixed_same_match.mp4",
               "wadi_degla_fixed": "wadi_degla_fixed.mp4"}
CLASS_CODE = {"player": "p", "goalkeeper": "g", "referee": "r", "ball": "b", "person": "p"}


class PlayerError(ValueError):
    pass


def _probe(path: Path) -> dict[str, Any]:
    from football_fatigue.multicam_lab.media import probe_video

    return probe_video(path)


@dataclass
class Catalog:
    data_root: Path
    out_root: Path

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._probe: dict[str, dict[str, Any]] = {}
        self._dev: dict[str, "DevSource"] = {}
        self._wadi: dict[str, dict[str, Any]] = {}
        self._readers: dict[str, Any] = {}
        from football_fatigue.video.player.debug import Debug

        self.debug = Debug(self.data_root, self.out_root)

    # ------------------------------------------------------------------ videos
    def video_path(self, key: str) -> Path:
        if key in SOURCES:
            guard_not_reserved(key, SOURCES[key]["video"])
            return Path(self.data_root) / SOURCES[key]["video"]
        if key in WADI_VIDEOS:
            return Path(self.data_root) / WADI_RAW / WADI_VIDEOS[key]
        raise PlayerError(f"unknown video {key!r}")

    def probe(self, key: str) -> dict[str, Any]:
        with self._lock:
            if key not in self._probe:
                self._probe[key] = _probe(self.video_path(key))
            return self._probe[key]

    def runs(self) -> list[D.RunInfo]:
        return [r for r in D.discover_runs(self.out_root) if not any(m in r.source_id for m in RESERVED_MATCH_IDS)]

    def sources(self) -> list[dict[str, Any]]:
        out = []
        runs = self.runs()
        for sid, spec in SOURCES.items():
            path = self.video_path(sid)
            if not path.is_file():
                continue
            tis = [n for n in TRACKER_INPUTS if (tracks_dir(self.data_root, self.out_root, sid, n) / "frames.jsonl").is_file()]
            out.append({"key": sid, "kind": "soccertrack", "role": spec["role"], "title": f"{sid} (SoccerTrack, {spec['role']})",
                        "has_gt": True, "tracker_inputs": tis, "recommended_tracker_input": RECOMMENDED_TRACKER_INPUT,
                        "tracker_input_text": {n: DESCRIPTIONS[n] for n in tis},
                        "runs": [{"run_id": r.run_id, "profile": r.profile, "tracker_input": r.tracker_input,
                                  "window": list(r.window), "full": r.window[1] - r.window[0] > 1000}
                                 for r in runs if r.source_id == sid]})
        for key, name in WADI_VIDEOS.items():
            if not self.video_path(key).is_file():
                continue
            wins = self.wadi_windows(key)
            out.append({"key": key, "kind": "wadi_degla", "title": f"{name} (Wadi Degla, no ground truth)", "has_gt": False,
                        "tracker_inputs": [], "runs": [], "detection_windows": [
                            {k: w[k] for k in ("id", "start_frame", "end_frame", "t0_s", "dur_s", "step_s", "imgsz",
                                               "upscale", "conf", "n_samples")} for w in wins]})
        return out

    def source_meta(self, key: str) -> dict[str, Any]:
        p = self.probe(key)
        return {"key": key, "fps": p["fps"], "width": p["width"], "height": p["height"], "frames": p["frame_count"],
                "first_pts_s": p["first_pts_s"], "duration_s": p["duration_s"], "codec": p["codec"],
                "video_url": f"/media/{key}"}

    def thumb(self, key: str, frame: int, w: int = 64, h: int = 18, crop: tuple[int, int, int, int] | None = None
              ) -> dict[str, Any]:
        """Grey thumbnail of the exactly decoded frame (decoded index, ``ExactFrameReader``) for the browser's
        frame-accuracy self-test: the page compares it with what the <video> element presents for that frame."""
        import cv2

        from football_fatigue.video.sources import ExactFrameReader

        w, h = max(8, min(w, 256)), max(4, min(h, 128))
        with self._lock:
            if key not in self._readers:
                self._readers[key] = ExactFrameReader(self.video_path(key), cache_frames=8)
            img = self._readers[key].frame(int(frame))
        if img is None:
            raise PlayerError(f"frame {frame} cannot be decoded")
        if crop is not None:
            x, y, cw, ch = (int(v) for v in crop)
            img = img[max(0, y):y + ch, max(0, x):x + cw]
        g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (w, h), interpolation=cv2.INTER_AREA)
        return {"frame": frame, "w": w, "h": h, "grey": g.flatten().tolist()}

    # ------------------------------------------------------------------ development matches
    def dev(self, sid: str) -> "DevSource":
        if sid not in SOURCES:
            raise PlayerError(f"unknown development source {sid!r}")
        with self._lock:
            if sid not in self._dev:
                self._dev[sid] = DevSource(self, sid)
            return self._dev[sid]

    # ------------------------------------------------------------------ Wadi Degla (existing M10.8 detector windows)
    def wadi_windows(self, key: str) -> list[dict[str, Any]]:
        if key in self._wadi:
            return self._wadi[key]["windows"]
        wins = []
        base = Path(self.data_root) / WADI_DETECTIONS
        for f in sorted(base.glob("*.json")) if base.is_dir() else []:
            d = json.loads(f.read_text(encoding="utf-8"))
            m = d["meta"]
            if m.get("video") != WADI_VIDEOS[key]:
                continue
            fps = self.probe(key)["fps"]
            samples = {}
            for fr in d["frames"]:
                idx = int(round(fr["t"] * fps))
                samples[idx] = [[round(b[0], 1), round(b[1], 1), round(b[2], 1), round(b[3], 1), round(float(b[5]), 3),
                                 CLASS_CODE.get(b[4], "p"), 4 | 8] for b in fr["boxes"]]
            wins.append({"id": f.stem, "file": str(f), "start_frame": min(samples), "end_frame": max(samples),
                         "t0_s": m["t0_s"], "dur_s": m["dur_s"], "step_s": m["step_s"], "imgsz": m.get("imgsz"),
                         "upscale": m.get("upscale"), "conf": m.get("conf"), "n_samples": len(samples),
                         "samples": samples, "provenance": {k: m[k] for k in m if k != "class_names"}})
        self._wadi[key] = {"windows": wins}
        return wins

    def chunk(self, key: str, start: int, n: int, tracker_input: str | None, run_id: str | None) -> dict[str, Any]:
        n = max(1, min(int(n), 1000))
        if key in SOURCES:
            return self.dev(key).chunk(start, n, tracker_input or "B0", run_id)
        if key in WADI_VIDEOS:
            frames: dict[int, Any] = {}
            for w in self.wadi_windows(key):
                for f, boxes in w["samples"].items():
                    if start <= f < start + n:
                        frames[f] = {"d": boxes}
            return {"start": start, "n": n, "frames": frames,
                    "note": "Wadi Degla: detector output only (M10.8 CPU windows, YOLO26-L player_detector.pt); "
                            "no tracks, no ground truth"}
        raise PlayerError(f"unknown source {key!r}")


class DevSource:
    """One SoccerTrack development match: cache, tracks per tracker input, GT, engine runs."""

    def __init__(self, cat: Catalog, sid: str) -> None:
        from football_fatigue.vision.mot_eval import load_reference

        spec = SOURCES[sid]
        guard_not_reserved(sid, spec["match_id"], spec["cache_run"], spec["replay_run"], spec["gt"])
        self.cat, self.sid = cat, sid
        cdir = cache_dir(cat.data_root, sid)
        self.columns = json.loads((cdir / "cache_manifest.json").read_text(encoding="utf-8"))["columns"]
        self.cache = LineIndex(cdir / "detections.jsonl", "frame_num")
        self.gt, _ = load_reference(Path(cat.data_root) / spec["gt"], 0)
        self._tracks: dict[str, LineIndex] = {}
        self._runs: dict[str, Any] = {}
        self._lock = threading.Lock()

    def tracks_index(self, name: str) -> LineIndex:
        with self._lock:
            if name not in self._tracks:
                path = tracks_dir(self.cat.data_root, self.cat.out_root, self.sid, name) / "frames.jsonl"
                if not path.is_file():
                    raise PlayerError(f"no tracks for tracker input {name} of {self.sid}")
                self._tracks[name] = LineIndex(path, "frame_num")
            return self._tracks[name]

    def run(self, run_id: str):
        with self._lock:
            if run_id not in self._runs:
                info = next((r for r in self.cat.runs() if r.run_id == run_id and r.source_id == self.sid), None)
                if info is None:
                    raise PlayerError(f"unknown run {run_id!r} for {self.sid}")
                self._runs[run_id] = D.load_run(info)
            return self._runs[run_id]

    def chunk(self, start: int, n: int, tracker_input: str, run_id: str | None) -> dict[str, Any]:
        f = TRACKER_INPUTS.get(tracker_input)
        if f is None:
            raise PlayerError(f"unknown tracker input {tracker_input!r}")
        tix = self.tracks_index(tracker_input)
        run = self.run(run_id) if run_id else None
        if run is not None and run.info.tracker_input != tracker_input:
            raise PlayerError(f"run {run_id} analysed tracker input {run.info.tracker_input}, not {tracker_input}")
        frames: dict[int, Any] = {}
        for fr in range(max(0, start), start + n):
            rec = self.cache.get(fr)
            if rec is None:
                continue
            dets = []
            for row in rec["observations"]:
                o = dict(zip(self.columns, row)) if isinstance(row, list) else row
                mapped = o["mapped_class"]
                person = mapped in ("player", "goalkeeper", "referee", "person")
                turf_rejected = o["passes_turf_filter"] in (0, False)
                passes = person and float(o["confidence"]) > f.threshold and not (f.turf == "cached" and turf_rejected)
                dets.append([round(float(o["x1"]), 1), round(float(o["y1"]), 1), round(float(o["x2"]), 1),
                             round(float(o["y2"]), 1), round(float(o["confidence"]), 3),
                             CLASS_CODE.get(mapped or "", "u"), (1 if passes else 0) | (2 if turf_rejected else 0)
                             | (4 if person else 0)])
            tr = tix.get(fr) or {"players": []}
            tracks = [[int(p["track_id"]), *[round(float(v), 1) for v in p["bbox_xyxy_px"]],
                       round(float(p["confidence"]), 3), CLASS_CODE.get(p.get("observed_class") or "", "p")]
                      for p in tr["players"]]
            item: dict[str, Any] = {"d": dets, "t": tracks,
                                    "g": [[int(gid), *[round(float(v), 1) for v in box]] for gid, box in self.gt.get(fr, [])]}
            oc = self.cat.debug.outcomes(self.sid, tracker_input, fr)     # pipeline outcome per GT player (debugger)
            if oc is not None:
                item["o"] = oc
            if run is not None:
                eng = []
                for r in run.rows_by_frame.get(fr, []):
                    for tid, b in r["boxes"].items():
                        if b["state"] != "visible":
                            eng.append([int(tid), b["state"], *[round(float(v), 1) for v in b["box"]], r["event_id"]])
                item["e"] = eng
            frames[fr] = item
        return {"start": start, "n": n, "frames": frames}

    def events(self, run_id: str) -> dict[str, Any]:
        run = self.run(run_id)
        labels = S.labels_for(run.events)
        out = []
        for e in run.events:
            ge = run.eval_by_event.get(e["event_id"], {})
            out.append({"id": e["event_id"], "label": labels[e["event_id"]], "cat": ov.category_of(e, True),
                        "cls": e["event_class"], "sub": e["subtype"], "start": e["start_frame"], "end": e["end_frame"],
                        "peak": e["peak_frame"], "level": e["evidence_level"], "uncertain": e["uncertain"],
                        "tracks": [int(m["track_id"]) for m in e["members"]], "text": ov.describe(e),
                        "gt": ge.get("outcome"), "flags": e.get("flags", [])})
        ev = run.evaluation["summary"] if run.evaluation else None
        return {"run_id": run_id, "tracker_input": run.info.tracker_input, "profile": run.info.profile,
                "window": list(run.info.window), "events": out, "evaluation_summary": ev,
                "categories": {c: {"label": ov.CATEGORY_LABELS[c], "colour": ov.CATEGORY_HEX[c]} for c in ov.CATEGORIES},
                "pts_offset_s": run.pts_offset, "fps": run.fps}


@lru_cache(maxsize=8)
def _compare_cached(out_root: str, sid: str, stamp: float) -> dict[str, Any]:
    from football_fatigue.video.player.compare import compare_source

    return compare_source(Path(out_root), sid)


def compare(cat: Catalog, sid: str) -> dict[str, Any]:
    base = Path(cat.out_root) / sid
    stamp = max((p.stat().st_mtime for p in base.rglob("*.json")), default=0.0) if base.exists() else 0.0
    return _compare_cached(str(cat.out_root), sid, stamp)
