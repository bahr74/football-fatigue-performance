"""Video sources and the read-only inputs the occlusion engine consumes.

Inputs are read from a caller-supplied data root and never written. Two existing artefacts are used:

* a complete detection cache (``vision.detection_cache``), validated by its own reader;
* the production tracker replay of that cache (``frames.jsonl`` written by ``python -m football_fatigue.vision``).

Every tracked box in the requested frames is checked against the cache (same box among the cached
person observations), so the track history provably comes from that cache.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from football_fatigue.video.occlusion.engine import FrameInput, TrackObs
from football_fatigue.video.occlusion import geometry as g
from football_fatigue.video.schema import SourceInfo

# Never opened by Session B (spent validation, reserved validation, Tier-2 test).
RESERVED_MATCH_IDS = frozenset({"118577", "118578", "128058", "128057", "132831", "132877"})

# Known development sources; paths are relative to the data root (read only). ``role`` follows the Step 2 split:
# tune = thresholds may be changed after looking; check = scored once with frozen settings (development data, not
# independent validation). ``keypoints`` are SoccerTrack's annotated pitch landmarks (image px <-> pitch m), used only
# to build the play-area polygon; ground-truth player boxes are never an engine input.
_ST = "external/soccertrack-v2"
SOURCES: dict[str, dict[str, str]] = {
    "soccertrack-117093": {"match_id": "117093", "camera_id": "st-pano-0", "role": "tune",
                           "video": f"{_ST}/raw/117093/117093.mp4", "gt": f"{_ST}/mot/117093.txt",
                           "keypoints": f"{_ST}/raw/117093/117093_keypoints.json",
                           "cache_run": "soccertrack-117093-m6-d1fixed-full-merged-20261025",
                           "replay_run": "soccertrack-117093-m6-d1fixed-full-merged-20261025-full-stable-replay"},
    "soccertrack-118575": {"match_id": "118575", "camera_id": "st-pano-0", "role": "tune",
                           "video": f"{_ST}/mot/clips/118575.mp4", "gt": f"{_ST}/mot/118575.txt",
                           "keypoints": f"{_ST}/raw/118575/118575_keypoints.json",
                           "cache_run": "soccertrack-118575-m62-d1fixed-full-merged-20261026",
                           "replay_run": "soccertrack-118575-m62-d1fixed-full-merged-20261026-full-stable-replay"},
    "soccertrack-117092": {"match_id": "117092", "camera_id": "st-pano-117092", "role": "check",
                           "video": "raw/soccertrack_v2/mot/117092.mp4", "gt": "raw/soccertrack_v2/mot/117092.txt",
                           "keypoints": f"{_ST}/raw/117092/117092_keypoints.json",
                           "cache_run": "soccertrack-117092-m6-d1ea-full-merged-20261025",
                           "replay_run": "soccertrack-117092-m6-d1ea-full-merged-20261025-full-stable-replay"},
    "soccertrack-118576": {"match_id": "118576", "camera_id": "st-pano-0", "role": "check",
                           "video": f"{_ST}/mot/clips/118576.mp4", "gt": f"{_ST}/mot/118576.txt",
                           "keypoints": f"{_ST}/raw/118576/118576_keypoints.json",
                           "cache_run": "soccertrack-118576-m62-d1fixed-full-merged-20261026",
                           "replay_run": "soccertrack-118576-m62-d1fixed-full-merged-20261026-full-stable-replay"},
}


def play_area_polygon(keypoints_path: Path) -> list[list[float]]:
    """Image polygon of the pitch boundary from annotated landmarks: far touchline (y=0) by x, right goal line
    (x=105) by y, near touchline (y=68) by x descending, left goal line (x=0) by y descending."""
    raw = json.loads(Path(keypoints_path).read_text(encoding="utf-8"))
    pts = {tuple(float(v) for v in k.strip("()").split(",")): (float(p[0]), float(p[1])) for k, p in raw.items()}
    ordered: list[tuple[float, float]] = []
    sides = (
        sorted((p for p in pts if p[1] == 0.0), key=lambda p: p[0]),
        sorted((p for p in pts if p[0] == 105.0), key=lambda p: p[1]),
        sorted((p for p in pts if p[1] == 68.0), key=lambda p: -p[0]),
        sorted((p for p in pts if p[0] == 0.0), key=lambda p: -p[1]),
    )
    for side in sides:
        for p in side:
            if pts[p] not in ordered:
                ordered.append(pts[p])
    if len(ordered) < 4:
        raise SourceError(f"too few boundary keypoints for a play area: {keypoints_path}")
    return [[round(x, 2), round(y, 2)] for x, y in ordered]

PERSON_CLASSES = ("player", "goalkeeper", "referee", "person")


class SourceError(ValueError):
    """An input is missing, reserved, or inconsistent."""


def guard_not_reserved(*texts: str) -> None:
    for text in texts:
        for mid in RESERVED_MATCH_IDS:
            if mid in str(text):
                raise SourceError(f"refusing to touch reserved sequence {mid} ({text})")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source(data_root: Path, source_id: str) -> tuple[SourceInfo, Path]:
    if source_id not in SOURCES:
        raise SourceError(f"unknown source {source_id!r}; known: {sorted(SOURCES)}")
    spec = SOURCES[source_id]
    guard_not_reserved(source_id, spec["match_id"], *(spec.get(k, "") for k in ("video", "gt", "cache_run", "replay_run")))
    video = Path(data_root) / spec["video"]
    if not video.is_file():
        raise SourceError(f"video not found: {video}")
    from football_fatigue.multicam_lab.media import probe_video

    meta = probe_video(video)
    info = SourceInfo(source_id=source_id, match_id=spec["match_id"], camera_id=spec["camera_id"],
                      video_sha256=file_sha256(video), fps=float(meta["fps"]),
                      pts_offset_s=round(float(meta["first_pts_s"]), 6),
                      width=int(meta["width"]), height=int(meta["height"]))
    return info, video


class ExactFrameReader:
    """Frame access by **decoded index** (0 = first decoded frame), the index the detection cache uses.

    ``multicam_lab.media.FrameReader`` derives the index from the absolute PTS, which is off by the container start
    time when the first PTS is not 0 (117093: 0.979 s = 24.5 frames). Here the index is ``round((pts - first_pts) * fps)``
    for constant-frame-rate files; :func:`verify_reader` checks it against plain sequential decoding.
    """

    def __init__(self, path: Path, cache_frames: int = 64) -> None:
        import av

        self._c = av.open(str(path))
        self._vs = self._c.streams.video[0]
        self._tb = float(self._vs.time_base)
        self.fps = float(self._vs.average_rate)
        first = next(self._c.decode(self._vs))
        self.first_pts = first.pts
        self._dec = None
        self._last = -10
        self._cache: dict[int, Any] = {}
        self._order: list[int] = []
        self._max = cache_frames

    def _index_of(self, frame) -> int:
        return int(round((frame.pts - self.first_pts) * self._tb * self.fps))

    def frame(self, index: int):
        if index in self._cache:
            return self._cache[index]
        if self._dec is None or not (self._last < index <= self._last + 90):
            target = self.first_pts + max(0.0, index / self.fps - 1.0) / self._tb
            self._c.seek(int(target), stream=self._vs, backward=True)
            self._dec = self._c.decode(self._vs)
            self._last = -10
        for fr in self._dec:
            i = self._index_of(fr)
            self._last = i
            if i < index - 1:
                continue
            self._cache[i] = fr.to_ndarray(format="bgr24")
            self._order.append(i)
            if len(self._order) > self._max:
                self._cache.pop(self._order.pop(0), None)
            if i >= index:
                break
        return self._cache.get(index)

    def close(self) -> None:
        self._c.close()


def verify_reader(path: Path, indices: list[int]) -> dict[str, Any]:
    """Compare ExactFrameReader frames with sequentially decoded frames (decoded index = count from 0)."""
    import av
    import numpy as np

    wanted = sorted(set(indices))
    sequential: dict[int, Any] = {}
    with av.open(str(path)) as c:
        for i, fr in enumerate(c.decode(c.streams.video[0])):
            if i in wanted:
                sequential[i] = fr.to_ndarray(format="bgr24")
            if i >= wanted[-1]:
                break
    reader = ExactFrameReader(path)
    try:
        diffs = {i: float(np.abs(reader.frame(i).astype(int) - sequential[i].astype(int)).mean()) for i in wanted}
    finally:
        reader.close()
    return {"frames": wanted, "mean_abs_diff": diffs, "exact": all(d == 0.0 for d in diffs.values())}


@dataclass
class LoadedInputs:
    frames: list[FrameInput]
    provenance: dict[str, Any]


def _same_box(a: g.Box, b: g.Box, tol: float = 1e-3) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def load_inputs(cache_dir: Path, replay_dir: Path, source: SourceInfo, first: int, last: int,
                tracker_input: str = "B0") -> LoadedInputs:
    """Tracks from the replay and untracked cached person boxes, for frames ``first..last``.

    ``tracker_input`` names the profile whose tracks ``replay_dir`` holds (``video.tracking``): it decides which cached
    person boxes count as possible tracker inputs (B0: turf filter applied; T0: turf rejection ignored)."""
    from football_fatigue.vision.detection_cache import DetectionCacheReader
    from football_fatigue.video.tracking import accepts

    guard_not_reserved(str(cache_dir), str(replay_dir))
    reader = DetectionCacheReader(cache_dir)
    cache_manifest = reader.manifest
    if cache_manifest["video"]["sha256"] != source.video_sha256:
        raise SourceError("detection cache was made from a different video (SHA-256 mismatch)")
    replay_manifest = json.loads((Path(replay_dir) / "manifest.json").read_text(encoding="utf-8"))
    if replay_manifest["inputs"]["video_sha256"] != source.video_sha256:
        raise SourceError("tracker replay was made from a different video (SHA-256 mismatch)")
    if replay_manifest["inputs"]["weights_sha256"] != cache_manifest["detector"]["weights_sha256"]:
        raise SourceError("tracker replay and cache come from different detector weights")
    threshold = float(replay_manifest["configuration"]["confidence_threshold"])

    tracks: dict[int, dict[int, TrackObs]] = {}
    frames_path = Path(replay_dir) / "frames.jsonl"
    with frames_path.open(encoding="utf-8") as stream:
        for line in stream:
            rec = json.loads(line)
            fr = int(rec["frame_num"])
            if fr < first:
                continue
            if fr > last:
                break
            tracks[fr] = {int(p["track_id"]): TrackObs(tuple(float(v) for v in p["bbox_xyxy_px"]), float(p["confidence"]),
                                                       str(p.get("observed_class") or p.get("detected_class")))
                          for p in rec["players"]}
    missing = [f for f in range(first, last + 1) if f not in tracks]
    if missing:
        raise SourceError(f"tracker replay lacks frames {missing[:5]}…")

    frames: list[FrameInput] = []
    unverified = 0
    for fr, _ts, observations in reader.iter_records(start_frame=first):
        if fr > last:
            break
        people = [o for o in observations if accepts(o, tracker_input)]
        tracked = list(tracks[fr].values())
        for t in tracked:
            if not any(_same_box(t.box, o.xyxy) and o.confidence > threshold - 1e-4 for o in people):
                unverified += 1
        untracked = tuple(o.xyxy for o in people if not any(g.iou(o.xyxy, t.box) > 0.9 for t in tracked))
        frames.append(FrameInput(fr, tracks[fr], untracked))
    if unverified:
        raise SourceError(f"{unverified} tracked boxes are not in the detection cache: replay and cache do not match")
    provenance = {
        "detection_cache": {"run_id": cache_manifest.get("run_id"), "records_sha256": cache_manifest.get("records_sha256"),
                            "confidence_floor": cache_manifest["detector"]["confidence_floor"],
                            "tile_merge": cache_manifest["detector"].get("tile_merge", {}).get("policy")},
        "tracker_replay": {"run_dir": Path(replay_dir).name, "frames_sha256": file_sha256(frames_path),
                           "tracker": replay_manifest["configuration"].get("selected_tracker"),
                           "params": replay_manifest["configuration"].get("tracker_params"),
                           "tracking_confidence": threshold},
        "tracked_boxes_verified_against_cache": True,
    }
    if tracker_input != "B0":       # B0 provenance stays byte-identical to the historical runs
        provenance["tracker_input"] = tracker_input
    return LoadedInputs(frames, provenance)
