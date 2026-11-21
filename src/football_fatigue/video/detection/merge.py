"""Tile-merge policies for the post-processing A/B on identical raw per-tile detector output.

Raw records come from ``capture`` (every per-tile box before any merging, with its tile). Policies:

``A0``  production ``edge_aware`` merge, computed by Session A's own ``YOLOPlayerDetector._merge_person_agnostic``
        (same code, same constants) so the baseline is the cached pipeline by construction;
``A1``  cross-tile duplicate suppression only: two boxes from the *same* tile never suppress each other (YOLO26's
        one-to-one head already de-duplicates inside a tile), so overlapping players found in one tile survive;
``A2``  overlap-region-aware: as A1, and a pair is suppressed only if both boxes lie mostly inside the overlap region of
        their two tiles (the only place a cross-tile duplicate can come from);
``A3``  production merge with a higher global NMS IoU (0.7 instead of 0.5).

The edge-truncation rule (a box cut by an interior tile edge and completed by another tile's box) is applied in every
policy exactly as in production.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Sequence

Item = tuple[int, float, list[float]]          # (class id, confidence, xyxy) as in vision.detector
Rect = tuple[int, int, int, int]               # tile (x, y, w, h)
POLICIES = ("A0", "A1", "A2", "A3")


def _detector(names: dict[int, str], merge_iou: float, cross_role_iou: float):
    from football_fatigue.vision.detector import YOLOPlayerDetector

    det = YOLOPlayerDetector(model_path="(not loaded)", tile_width=1280, tile_height=1280, tile_merge="edge_aware",
                             merge_iou=merge_iou, cross_role_iou=cross_role_iou)

    class _Stub:
        pass

    stub = _Stub()
    stub.names = dict(names)
    det._model = stub
    return det


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _inside_share(box: Sequence[float], rect: tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(box[2], rect[2]) - max(box[0], rect[0]))
    iy = max(0.0, min(box[3], rect[3]) - max(box[1], rect[1]))
    area = (box[2] - box[0]) * (box[3] - box[1])
    return ix * iy / area if area > 0 else 0.0


def _overlap_rect(a: Rect, b: Rect) -> tuple[float, float, float, float] | None:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _cross_tile_nms(items: list[Item], tiles: list[Rect], idx: list[int], iou_thr: float, region: bool,
                    same_class: bool) -> list[int]:
    order = sorted(idx, key=lambda k: -items[k][1])
    kept: list[int] = []
    for k in order:
        dup = False
        for j in kept:
            if tiles[j] == tiles[k] or (same_class and items[j][0] != items[k][0]):
                continue
            if _iou(items[j][2], items[k][2]) <= iou_thr:
                continue
            if region:
                ov = _overlap_rect(tiles[j], tiles[k])
                if ov is None or _inside_share(items[j][2], ov) < 0.5 or _inside_share(items[k][2], ov) < 0.5:
                    continue
            dup = True
            break
        if not dup:
            kept.append(k)
    return kept


def merge(policy: str, items: list[Item], tiles: list[Rect], frame_size: tuple[int, int], names: dict[int, str],
          merge_iou: float = 0.5, cross_role_iou: float = 0.7) -> list[Item]:
    """Merged detections of one frame under ``policy`` (items and their tiles in raw capture order)."""
    if policy == "A0":
        return _detector(names, merge_iou, cross_role_iou)._merge_person_agnostic(items, tiles, frame_size)
    if policy == "A3":
        return _detector(names, 0.7, max(0.7, cross_role_iou))._merge_person_agnostic(items, tiles, frame_size)
    if policy not in ("A1", "A2"):
        raise ValueError(f"unknown policy {policy}")
    det = _detector(names, merge_iou, cross_role_iou)
    from football_fatigue.vision.detector import _PERSON_CLASSES

    person = [k for k, it in enumerate(items) if det._map_class(it[0], names) in _PERSON_CLASSES]
    other = [k for k in range(len(items)) if k not in set(person)]
    region = policy == "A2"
    kept = _cross_tile_nms(items, tiles, other, merge_iou, region=False, same_class=True)
    if person:
        survivors = det._resolve_edge_truncation(items, tiles, frame_size, person, names)
        same_role = _cross_tile_nms(items, tiles, survivors, merge_iou, region, same_class=True)
        cross = _cross_tile_nms(items, tiles, same_role, cross_role_iou, region, same_class=False)
        truncated = {a for a in cross if any(b != a and det._truncated_by(items[a], tiles[a], items[b], tiles[b],
                                                                         frame_size) for b in cross)}
        kept += [k for k in cross if k not in truncated]
    return [items[k] for k in sorted(kept)]


def read_capture(path: Path):
    """Yield capture records: {frame, size, tiles, items: [[cls, conf, x1, y1, x2, y2, tile]], turf: [...]}."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def run_ab(data_root: Path, out_root: Path, capture_dir: Path) -> dict[str, Any]:
    """Score every policy on the frames of a capture (detection metrics vs GT; CPU)."""
    from football_fatigue.vision.mot_eval import load_reference

    from football_fatigue.video.detection.stages import Det, Params, attribute_frame, threshold_counts
    from football_fatigue.video.sources import SOURCES, play_area_polygon
    import numpy as np

    manifest = json.loads((Path(capture_dir) / "capture_manifest.json").read_text(encoding="utf-8"))
    source_id = manifest["source_id"]
    spec = SOURCES[source_id]
    names = {int(k): v for k, v in manifest["class_names"].items()}
    floor = float(manifest["cache_floor"])
    gt, _ = load_reference(Path(data_root) / spec["gt"], 0)
    poly = np.asarray(play_area_polygon(Path(data_root) / spec["keypoints"]), dtype=np.float32).reshape(-1, 1, 2)
    from collections import Counter

    res: dict[str, Counter] = {p: Counter() for p in POLICIES}
    vis_recall: dict[str, Counter] = {p: Counter() for p in POLICIES}
    baseline_equal = 0
    frames = 0
    for rec in read_capture(Path(capture_dir) / "raw_tiles.jsonl.gz"):
        frames += 1
        tiles = [tuple(t) for t in rec["tiles"]]
        raw = [(int(r[0]), float(r[1]), [float(v) for v in r[2:6]]) for r in rec["items"] if float(r[1]) > floor]
        tile_of = [tiles[int(r[6])] for r in rec["items"] if float(r[1]) > floor]
        turf = {tuple(round(v, 2) for v in r[2:6]): t for r, t in zip(rec["items"], rec["turf"])}
        size = tuple(rec["size"])
        g = gt.get(int(rec["frame"]), [])
        for p in POLICIES:
            merged = merge(p, raw, tile_of, size, names)
            dets = [Det(tuple(b), c, names.get(k, str(k)), _mapped(names.get(k, "")), turf.get(tuple(round(v, 2) for v in b)))
                    for k, c, b in merged]
            if p == "A0" and rec.get("cached_boxes") is not None:
                cached = sorted(tuple(round(v, 1) for v in b) for b in rec["cached_boxes"])
                baseline_equal += cached == sorted(tuple(round(v, 1) for v in b) for _k, _c, b in merged)
            res[p].update(threshold_counts(g, dets, poly, (0.4,))[0.4])
            for r in attribute_frame(g, dets, {}, poly, size)["gt"]:
                vis_recall[p][(r["visibility"], r["stage"] == "detected_untracked")] += 1
    out = {"source_id": source_id, "frames": frames, "baseline_frames_equal_to_cache": baseline_equal,
           "policies": {}}
    for p in POLICIES:
        c = res[p]
        fp = c["duplicate"] + c["fp_on_pitch"] + c["referee_label"] + c["off_pitch"] + c["poor_overlap"]
        out["policies"][p] = {
            "counts": dict(c), "recall": round(c["tp"] / max(1, c["gt"]), 4),
            "precision_all": round(c["tp"] / max(1, c["tp"] + fp), 4),
            "duplicates_per_100_tp": round(100 * c["duplicate"] / max(1, c["tp"]), 2),
            "recall_by_visibility": {v: round(vis_recall[p][(v, True)] / max(1, vis_recall[p][(v, True)]
                                                                            + vis_recall[p][(v, False)]), 4)
                                     for v in ("visible", "partial", "heavy", "hidden")},
        }
    dest = Path(out_root) / source_id / "detection" / "merge-ab-v1"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{Path(capture_dir).name}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def _mapped(label: str) -> str | None:
    label = label.strip().lower()
    return {"player": "player", "goalkeeper": "goalkeeper", "referee": "referee", "ball": "ball"}.get(label)
