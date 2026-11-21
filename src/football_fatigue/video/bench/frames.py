"""Deterministic, stratified benchmark frame manifest (ground truth and annotated pitch landmarks only).

No detector or tracker output is read here, so the frame choice cannot favour or penalise any model. Per match, every
frame gets GT-derived features; each stratum then contributes ``k`` frames drawn with a fixed seed from the frames that
satisfy it (with a minimum gap between chosen frames), plus ``k_normal`` frames drawn from all frames.

Strata (all from the SoccerTrack annotation; heuristics are named as such):

* ``normal``        - uniform draw over all frames
* ``crowded``       - most GT players within 2 median box heights of one player (top decile of the match)
* ``overlapping``   - most overlapping GT box pairs (IoU > 0; top decile)
* ``partial``       - at least one GT player 10-50 % covered by nearer GT boxes (top decile by count)
* ``heavy``         - at least one GT player >= 50 % covered (heavy or hidden; top decile by count)
* ``goal_mouth``    - most GT feet within 16.5 m of the same goal line (top decile of the match, at least 6) (pitch x interpolated along the annotated
                      touchlines; heuristic. Drafts with a nearest-landmark estimate matched 36-100 % of frames
                      (biased towards the goal lines) and were discarded before any detector run)
* ``small``         - most GT players shorter than 35 px (top decile; only if any exist)
* ``goalkeeper``    - a heuristic goalkeeper (GT id with the most extreme median pitch x) within 1.5 heights of another player
* ``fast_motion``   - largest GT box displacement per frame relative to its height (motion-blur proxy; top decile)
* ``edge``          - a GT player in the outer 5 % of the image width or 3 % of its height (panorama ends, strongest
                      distortion; a draft with a 30-px border matched no frame and was discarded)
* ``low_light``     - every frame of 117092 (match-level property; luminance evidence recorded in the manifest)
* camera movement / zoom - not present: the SoccerTrack development videos are a fixed panorama (recorded as N/A)
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from football_fatigue.video.detection.stages import Params, gt_cover, iou_matrix, outside_distance, visibility
from football_fatigue.video.sources import SOURCES, ExactFrameReader, guard_not_reserved, play_area_polygon

MANIFEST_VERSION = "frames-v1"
DEV_SOURCES = ("soccertrack-117093", "soccertrack-118575", "soccertrack-117092", "soccertrack-118576")
STRATA = ("normal", "crowded", "overlapping", "partial", "heavy", "goal_mouth", "small", "goalkeeper", "fast_motion",
          "edge", "low_light")
SEED = 20261120
LOW_LIGHT = {"soccertrack-117092"}


def _frame_size(data_root: Path, source_id: str) -> tuple[int, int]:
    from football_fatigue.video.detection.diagnose import cache_dir

    v = json.loads((cache_dir(data_root, source_id) / "cache_manifest.json").read_text(encoding="utf-8"))["video"]
    return int(v["width"]), int(v["height"])


def _pitch_x_estimator(keypoints_path: Path):
    """Pitch x (metres along the touchline) of image points: interpolate along the annotated near (y=68) and far (y=0)
    touchlines and blend by the point's vertical position between them (the panorama is curved, so no single
    homography; heuristic, used only to define the goal-mouth stratum and heuristic goalkeepers)."""
    raw = json.loads(Path(keypoints_path).read_text(encoding="utf-8"))
    pts = [(tuple(float(v) for v in k.strip("()").split(",")), (float(p[0]), float(p[1]))) for k, p in raw.items()]

    def line(y):
        row = sorted(((img[0], img[1], pit[0]) for pit, img in pts if pit[1] == y), key=lambda r: r[0])
        return np.array(row, dtype=float)
    near, far = line(68.0), line(0.0)

    def est(points: np.ndarray) -> np.ndarray:
        u, v = points[:, 0], points[:, 1]
        xn, yn = np.interp(u, near[:, 0], near[:, 2]), np.interp(u, near[:, 0], near[:, 1])
        xf, yf = np.interp(u, far[:, 0], far[:, 2]), np.interp(u, far[:, 0], far[:, 1])
        t = np.clip((v - yf) / np.where(np.abs(yn - yf) < 1, 1, yn - yf), 0, 1)
        return xf + t * (xn - xf)
    return est


def frame_features(data_root: Path, source_id: str) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    from football_fatigue.vision.mot_eval import load_reference

    spec = SOURCES[source_id]
    gt, _ = load_reference(Path(data_root) / spec["gt"], 0)
    size = _frame_size(data_root, source_id)
    poly = np.asarray(play_area_polygon(Path(data_root) / spec["keypoints"]), dtype=np.float32).reshape(-1, 1, 2)
    est_x = _pitch_x_estimator(Path(data_root) / spec["keypoints"])
    p = Params()
    # heuristic goalkeepers: the GT ids with the smallest / largest median estimated pitch x
    xs: dict[int, list[float]] = {}
    for fr in range(0, max(gt) + 1, 25):
        rows = gt.get(fr, [])
        if rows:
            px = est_x(np.array([[(b[0] + b[2]) / 2, b[3]] for _g, b in rows]))
            for (gid, _b), x in zip(rows, px):
                xs.setdefault(gid, []).append(float(x))
    med = {g: float(np.median(v)) for g, v in xs.items() if len(v) >= 20}
    keepers = sorted({min(med, key=med.get), max(med, key=med.get)}) if med else []
    feats: dict[int, dict[str, Any]] = {}
    prev: dict[int, tuple] = {}
    for fr in sorted(gt):
        rows = gt[fr]
        boxes = [b for _g, b in rows]
        if not boxes:
            continue
        hs = np.array([b[3] - b[1] for b in boxes])
        cover = gt_cover(boxes, p.cover_grid)
        vis = [visibility(b, float(c), outside_distance(b, poly), size, p) for b, c in zip(boxes, cover)]
        iou = iou_matrix(boxes, boxes)
        np.fill_diagonal(iou, 0)
        feet = np.array([[(b[0] + b[2]) / 2, b[3]] for b in boxes])
        dist = np.linalg.norm(feet[:, None] - feet[None], axis=2)
        r = 2 * float(np.median(hs))
        px = est_x(feet)
        speed = 0.0
        for (gid, b), h in zip(rows, hs):
            if gid in prev:
                q = prev[gid]
                speed = max(speed, float(np.hypot((b[0] + b[2] - q[0] - q[2]) / 2, (b[1] + b[3] - q[1] - q[3]) / 2) / max(h, 1)))
        prev = {gid: b for gid, b in rows}
        gk_near = 0
        for i, (gid, b) in enumerate(rows):
            if gid in keepers:
                others = np.delete(dist[i], i)
                if others.size and others.min() < 1.5 * hs[i]:
                    gk_near = 1
        feats[fr] = {
            "n_gt": len(boxes),
            "crowded": int(((dist < r).sum(axis=1) - 1).max()),
            "overlapping": int((np.triu(iou) > 0).sum()),
            "partial": int(sum(v == "partial" for v in vis)),
            "heavy": int(sum(v in ("heavy", "hidden") for v in vis)),
            "goal_mouth": int(max((px < 16.5).sum(), (px > 88.5).sum())),
            "small": int((hs < 35).sum()),
            "goalkeeper": gk_near,
            "fast_motion": round(speed, 4),
            "edge": int(sum(b[0] < 0.05 * size[0] or b[2] > 0.95 * size[0] or b[1] < 0.03 * size[1]
                            or b[3] > 0.97 * size[1] for b in boxes)),
        }
    info = {"frame_size": list(size), "heuristic_goalkeeper_gt_ids": keepers, "frames_with_gt": len(feats)}
    return feats, info


def _eligible(feats: dict[int, dict], source_id: str, stratum: str) -> list[int]:
    frames = sorted(feats)
    if stratum == "normal":
        return frames
    if stratum == "low_light":
        return frames if source_id in LOW_LIGHT else []
    thresholds = {"goalkeeper": 1, "edge": 1}
    if stratum in thresholds:
        return [f for f in frames if feats[f][stratum] >= thresholds[stratum]]
    vals = np.array([feats[f][stratum] for f in frames], dtype=float)
    if not (vals > 0).any():
        return []
    q = max(float(np.quantile(vals, 0.9)), float(vals[vals > 0].min()), 6.0 if stratum == "goal_mouth" else 0.0)
    return [f for f, v in zip(frames, vals) if v >= q and v > 0]


def _luminance(data_root: Path, source_id: str, frames: list[int]) -> float:
    reader = ExactFrameReader(Path(data_root) / SOURCES[source_id]["video"])
    vals = []
    for f in frames:
        bgr = reader.frame(f)
        vals.append(float((0.114 * bgr[..., 0] + 0.587 * bgr[..., 1] + 0.299 * bgr[..., 2]).mean()))
    return round(float(np.mean(vals)), 1)


def build(data_root: Path, out_root: Path, k: int = 2, k_normal: int = 6, min_gap: int = 50,
          sources: tuple[str, ...] = DEV_SOURCES, version: str = MANIFEST_VERSION, seed: int = SEED) -> Path:
    """``version``/``seed`` default to frames-v1 (unchanged); a larger set is a new version with its own seed."""
    dest = Path(out_root) / "_bench" / "frames" / f"{version}.json"
    if dest.exists():
        raise SystemExit(f"manifest already exists (write-once): {dest}")
    rng = np.random.default_rng(seed)
    entries: list[dict[str, Any]] = []
    per_source: dict[str, Any] = {}
    for sid in sources:
        spec = SOURCES[sid]
        guard_not_reserved(sid, spec["match_id"], spec["gt"], spec["video"])
        feats, info = frame_features(data_root, sid)
        chosen: dict[int, str] = {}
        counts = {}
        elig = {s: set(_eligible(feats, sid, s)) for s in STRATA}
        for stratum in STRATA:
            pool = sorted(elig[stratum])
            counts[stratum] = len(pool)
            want = k_normal if stratum == "normal" else k
            picked = 0
            for f in rng.permutation(pool).tolist() if pool else []:
                if picked >= want:
                    break
                if all(abs(f - g) >= min_gap for g in chosen):
                    chosen[int(f)] = stratum
                    picked += 1
        for f in sorted(chosen):
            entries.append({"source_id": sid, "role": spec["role"], "frame": f, "selected_for": chosen[f],
                            "strata": [s for s in STRATA if s != "normal" and f in elig[s]] or ["normal"],
                            "features": feats[f]})
        lum_frames = sorted(rng.choice(sorted(feats), size=3, replace=False).tolist())
        per_source[sid] = {**info, "role": spec["role"], "eligible_frames_per_stratum": counts,
                           "selected": len(chosen), "mean_luma_3_random_frames": _luminance(data_root, sid, lum_frames),
                           "luma_frames": lum_frames}
    body = {"version": version, "seed": seed, "k_per_stratum": k, "k_normal": k_normal, "min_gap_frames": min_gap,
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "strata_doc": __doc__,
            "camera_movement_zoom": "N/A: SoccerTrack development videos are a fixed panorama; Wadi Degla (moving camera) "
                                    "has no ground truth and is excluded from scored metrics",
            "inputs": "SoccerTrack MOT ground truth and annotated pitch landmarks only (no detector or tracker output)",
            "sources": per_source, "frames": entries}
    body["frames_sha256"] = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".partial")
    tmp.write_text(json.dumps(body, indent=1), encoding="utf-8")
    tmp.rename(dest)
    return dest


def load(out_root: Path, version: str = MANIFEST_VERSION) -> dict[str, Any]:
    m = json.loads((Path(out_root) / "_bench" / "frames" / f"{version}.json").read_text(encoding="utf-8"))
    if hashlib.sha256(json.dumps(m["frames"], sort_keys=True).encode()).hexdigest() != m["frames_sha256"]:
        raise RuntimeError("frame manifest was modified after it was written")
    return m
