"""Per-frame stage attribution: where is each ground-truth player kept or lost?

Stages, in pipeline order (a ground-truth player is assigned to the first stage that holds a matching box; matching
is one-to-one per stage by the Hungarian method on IoU):

``tracked``              a tracker output box (production ByteTrack replay) matches the player;
``detected_untracked``   a tracker-input detection (person class, turf filter passed, confidence > tracker threshold)
                         matches, but the tracker output no box for it (unconfirmed or rejected by association);
``below_threshold``      only a cached detection with floor < confidence <= tracker threshold matches;
``turf_filtered``        only a cached detection rejected by the turf filter matches;
``poor_overlap``         an unclaimed cached detection overlaps the player with IoU in [0.2, match IoU): detected, but
                         the box disagrees with the annotation (partial box, two-player box, or annotation offset);
``merge_suspect``        nothing matches, but a surviving detection overlaps the player strongly (IoU >= 0.3, or covers
                         >= 70 % of it). Greedy NMS can only remove a box that overlaps a surviving box by IoU > 0.5
                         (or is contained in it, edge rule), so these are the only misses a merge change could recover;
``no_detection``         no cached box near the player: the merge cannot explain it (no surviving neighbour), so the
                         post-merge output simply has nothing there. Without a raw per-tile capture this is the best
                         available evidence of a detector (model) miss.

Visibility classes come from ground truth only (SoccerTrack has the 22 players; no referees): the share of the box
covered by nearer players (larger bottom y), the play-area polygon, and the image border.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

Box = tuple[float, float, float, float]
PERSON = frozenset({"player", "goalkeeper", "referee", "person"})
STAGES = ("tracked", "detected_untracked", "below_threshold", "turf_filtered", "poor_overlap", "merge_suspect",
          "no_detection")
LOST_STAGES = STAGES[1:]
VISIBILITY = ("visible", "partial", "heavy", "hidden", "off_pitch", "uncertain")
SIZE_BINS = ((0, 35, "<35 px"), (35, 50, "35-50 px"), (50, 80, "50-80 px"), (80, 1e9, ">=80 px"))
DET_OUTCOMES = ("tp", "duplicate", "poor_overlap", "off_pitch", "referee_label", "fp_on_pitch")

STAGE_TEXT = {
    "tracked": "tracked: a tracker output box matches",
    "detected_untracked": "detected (conf > tracker threshold) but the tracker output no box",
    "below_threshold": "detected only below the tracker confidence threshold",
    "turf_filtered": "detected but removed by the turf (grass) filter",
    "poor_overlap": "a box exists but disagrees with the annotation (IoU 0.2-0.5)",
    "merge_suspect": "no box; a surviving neighbour overlaps it enough that tile-merge NMS may have removed it",
    "no_detection": "no box near the player after merging; NMS cannot explain it (likely a model miss)",
}


@dataclass(frozen=True)
class Det:
    box: Box
    conf: float
    label: str               # the model's own label
    mapped: str | None       # pipeline class (player / goalkeeper / referee / ball) or None
    turf: bool | None        # turf filter decision (None = not evaluated)

    @property
    def person(self) -> bool:
        return self.mapped in PERSON


@dataclass(frozen=True)
class Params:
    match_iou: float = 0.5
    track_threshold: float = 0.4
    poor_iou: float = 0.2
    suspect_iou: float = 0.3
    suspect_cover: float = 0.7
    border_px: float = 2.0
    off_pitch_heights: float = 0.5
    cover_grid: tuple[int, int] = (6, 12)


def iou_matrix(a: Sequence[Box], b: Sequence[Box]) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    A = np.asarray(a, dtype=float)[:, None, :]
    B = np.asarray(b, dtype=float)[None, :, :]
    iw = np.clip(np.minimum(A[..., 2], B[..., 2]) - np.maximum(A[..., 0], B[..., 0]), 0, None)
    ih = np.clip(np.minimum(A[..., 3], B[..., 3]) - np.maximum(A[..., 1], B[..., 1]), 0, None)
    inter = iw * ih
    aa = (A[..., 2] - A[..., 0]) * (A[..., 3] - A[..., 1])
    bb = (B[..., 2] - B[..., 0]) * (B[..., 3] - B[..., 1])
    union = aa + bb - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def cover_matrix(a: Sequence[Box], b: Sequence[Box]) -> np.ndarray:
    """Share of each box in ``a`` covered by each box in ``b``."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    A = np.asarray(a, dtype=float)[:, None, :]
    B = np.asarray(b, dtype=float)[None, :, :]
    iw = np.clip(np.minimum(A[..., 2], B[..., 2]) - np.maximum(A[..., 0], B[..., 0]), 0, None)
    ih = np.clip(np.minimum(A[..., 3], B[..., 3]) - np.maximum(A[..., 1], B[..., 1]), 0, None)
    aa = (A[..., 2] - A[..., 0]) * (A[..., 3] - A[..., 1])
    return np.where(aa > 0, iw * ih / np.where(aa > 0, aa, 1), 0.0)


def match(iou: np.ndarray, rows: Sequence[int], cols: Sequence[int], threshold: float) -> list[tuple[int, int]]:
    """One-to-one matches (row, col) with IoU >= threshold, maximising total IoU (deterministic)."""
    if not len(rows) or not len(cols):
        return []
    from scipy.optimize import linear_sum_assignment

    sub = iou[np.ix_(list(rows), list(cols))]
    cost = np.where(sub >= threshold, -sub, 1.0)
    r, c = linear_sum_assignment(cost)
    return sorted((rows[i], cols[j]) for i, j in zip(r, c) if sub[i, j] >= threshold)


def gt_cover(boxes: Sequence[Box], grid: tuple[int, int] = (6, 12)) -> np.ndarray:
    """Share of each ground-truth box hidden by nearer ground-truth boxes (larger bottom y), sampled on a grid."""
    n = len(boxes)
    out = np.zeros(n)
    if n < 2:
        return out
    B = np.asarray(boxes, dtype=float)
    gx = (np.arange(grid[0]) + 0.5) / grid[0]
    gy = (np.arange(grid[1]) + 0.5) / grid[1]
    for i in range(n):
        nearer = B[(B[:, 3] > B[i, 3]) & (np.arange(n) != i)]
        if not len(nearer):
            continue
        xs = B[i, 0] + gx * (B[i, 2] - B[i, 0])
        ys = B[i, 1] + gy * (B[i, 3] - B[i, 1])
        X, Y = np.meshgrid(xs, ys)
        X, Y = X.ravel()[:, None], Y.ravel()[:, None]
        inside = (X >= nearer[:, 0]) & (X <= nearer[:, 2]) & (Y >= nearer[:, 1]) & (Y <= nearer[:, 3])
        out[i] = float(inside.any(axis=1).mean())
    return out


def size_bin(height: float) -> str:
    return next(name for lo, hi, name in SIZE_BINS if lo <= height < hi)


def outside_distance(box: Box, poly: np.ndarray | None) -> float | None:
    """Signed distance (px) of the foot point to the play-area polygon; negative = outside."""
    if poly is None:
        return None
    import cv2

    return float(cv2.pointPolygonTest(poly, ((box[0] + box[2]) / 2.0, box[3]), True))


def visibility(box: Box, cover: float, dist: float | None, size: tuple[int, int], p: Params) -> str:
    h = box[3] - box[1]
    if dist is not None and dist < -p.off_pitch_heights * h:
        return "off_pitch"
    w, hh = size
    if box[0] < p.border_px or box[1] < p.border_px or box[2] > w - p.border_px or box[3] > hh - p.border_px:
        return "uncertain"
    if cover < 0.1:
        return "visible"
    if cover < 0.5:
        return "partial"
    if cover < 0.9:
        return "heavy"
    return "hidden"


def attribute_frame(gt: Sequence[tuple[int, Box]], dets: Sequence[Det], tracks: dict[int, Box],
                    poly: np.ndarray | None, size: tuple[int, int], p: Params = Params()) -> dict[str, Any]:
    """Stage attribution of one frame. Returns ``gt`` rows (one per GT player) and ``dets`` rows (one per tracker-input
    detection) plus the indices used, so a viewer can draw exactly what was scored."""
    gboxes = [b for _g, b in gt]
    cover = gt_cover(gboxes, p.cover_grid)
    tids = sorted(tracks)
    tboxes = [tracks[t] for t in tids]
    pidx = [i for i, d in enumerate(dets) if d.person]
    dboxes = [d.box for d in dets]
    g_t = iou_matrix(gboxes, tboxes)
    g_d = iou_matrix(gboxes, dboxes)
    gc_d = cover_matrix(gboxes, dboxes)

    stage: dict[int, str] = {}
    det_of: dict[int, int] = {}
    track_of: dict[int, int] = {}
    claimed: set[int] = set()
    all_g = list(range(len(gt)))
    for gi, ti in match(g_t, all_g, list(range(len(tids))), p.match_iou):
        stage[gi], track_of[gi] = "tracked", tids[ti]
    # detections that are exactly a tracker output box (the replay outputs the detection box itself)
    tracked_dets: set[int] = set()
    if tboxes and dboxes:
        same = iou_matrix(dboxes, tboxes) > 0.99
        tracked_dets = {di for di in pidx if same[di].any()}
        for di in pidx:
            if same[di].any():
                ti = int(np.argmax(same[di]))
                gi = next((g for g, t in track_of.items() if t == tids[ti]), None)
                if gi is not None and gi not in det_of:
                    det_of[gi] = di
                    claimed.add(di)
    def kept(d: Det) -> bool:
        return d.person and d.turf is not False

    tiers = (
        ("detected_untracked", [i for i in pidx if kept(dets[i]) and dets[i].conf > p.track_threshold
                                and i not in tracked_dets]),
        ("below_threshold", [i for i in pidx if kept(dets[i]) and dets[i].conf <= p.track_threshold]),
        ("turf_filtered", [i for i in pidx if dets[i].turf is False]),
    )
    for name, cols in tiers:
        rows = [g for g in all_g if g not in stage]
        for gi, di in match(g_d, rows, [c for c in cols if c not in claimed], p.match_iou):
            stage[gi], det_of[gi] = name, di
            claimed.add(di)
    for gi in all_g:
        if gi in stage:
            continue
        free = [d for d in pidx if d not in claimed and p.poor_iou <= g_d[gi, d] < p.match_iou]
        if free:
            stage[gi], det_of[gi] = "poor_overlap", max(free, key=lambda d: g_d[gi, d])
            continue
        near = [d for d in pidx if dets[d].turf is not False
                and (g_d[gi, d] >= p.suspect_iou or gc_d[gi, d] >= p.suspect_cover)]
        if near:
            stage[gi], det_of[gi] = "merge_suspect", max(near, key=lambda d: (gc_d[gi, d], g_d[gi, d]))
        else:
            stage[gi] = "no_detection"

    grows = []
    for gi, (gid, box) in enumerate(gt):
        dist = outside_distance(box, poly)
        d = det_of.get(gi)
        grows.append({
            "gid": gid, "box": box, "cover": round(float(cover[gi]), 3), "height": round(box[3] - box[1], 1),
            "size": size_bin(box[3] - box[1]), "visibility": visibility(box, float(cover[gi]), dist, size, p),
            "stage": stage[gi], "track_id": track_of.get(gi), "det": d,
            "det_conf": round(dets[d].conf, 3) if d is not None else None,
            "det_iou": round(float(g_d[gi, d]), 3) if d is not None else None,
        })

    # precision side: every tracker-input detection
    matched_gt = {gi for gi, s in stage.items() if s in ("tracked", "detected_untracked")}
    tp_dets = {det_of[gi] for gi in matched_gt if gi in det_of}
    drows = []
    for di in pidx:
        d = dets[di]
        if not (kept(d) and d.conf > p.track_threshold):
            continue
        if di in tp_dets:
            outcome = "tp"
        else:
            best = int(np.argmax(g_d[:, di])) if len(gt) else -1
            biou = float(g_d[best, di]) if best >= 0 else 0.0
            dist = outside_distance(d.box, poly)
            if biou >= 0.3 and best in matched_gt:
                outcome = "duplicate"
            elif biou >= p.poor_iou:
                outcome = "poor_overlap"
            elif dist is not None and dist < -p.off_pitch_heights * (d.box[3] - d.box[1]):
                outcome = "off_pitch"
            elif d.mapped == "referee":
                outcome = "referee_label"
            else:
                outcome = "fp_on_pitch"
        drows.append({"det": di, "outcome": outcome, "conf": round(d.conf, 3), "tracked": di in tracked_dets})
    return {"gt": grows, "dets": drows}


def threshold_counts(gt: Sequence[tuple[int, Box]], dets: Sequence[Det], poly: np.ndarray | None,
                     thresholds: Sequence[float], p: Params = Params()) -> dict[float, dict[str, int]]:
    """Frame-level detection counts of the tracker input at alternative confidence thresholds (CPU replay of the
    tracker-input filter on identical cached detections)."""
    gboxes = [b for _g, b in gt]
    out: dict[float, dict[str, int]] = {}
    for t in thresholds:
        idx = [i for i, d in enumerate(dets) if d.person and d.turf is not False and d.conf > t]
        boxes = [dets[i].box for i in idx]
        m = iou_matrix(gboxes, boxes)
        pairs = match(m, list(range(len(gboxes))), list(range(len(boxes))), p.match_iou)
        used = {c for _r, c in pairs}
        matched = {r for r, _c in pairs}
        c = {"gt": len(gboxes), "tp": len(pairs), "duplicate": 0, "off_pitch": 0, "referee_label": 0, "fp_on_pitch": 0,
             "poor_overlap": 0}
        for j, i in enumerate(idx):
            if j in used:
                continue
            col = m[:, j] if len(gboxes) else np.zeros(0)
            best = int(np.argmax(col)) if len(col) else -1
            biou = float(col[best]) if best >= 0 else 0.0
            dist = outside_distance(dets[i].box, poly)
            h = dets[i].box[3] - dets[i].box[1]
            if biou >= 0.3 and best in matched:
                c["duplicate"] += 1
            elif biou >= p.poor_iou:
                c["poor_overlap"] += 1
            elif dist is not None and dist < -p.off_pitch_heights * h:
                c["off_pitch"] += 1
            elif dets[i].mapped == "referee":
                c["referee_label"] += 1
            else:
                c["fp_on_pitch"] += 1
        out[t] = c
    return out


def average_precision(scored: Sequence[tuple[float, bool]], n_gt: int) -> float:
    """All-point interpolated AP from (confidence, is_tp) pairs over ``n_gt`` ground-truth boxes."""
    if not n_gt or not scored:
        return 0.0
    s = sorted(scored, key=lambda t: -t[0])
    tp = np.cumsum([1.0 if x else 0.0 for _c, x in s])
    fp = np.cumsum([0.0 if x else 1.0 for _c, x in s])
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate([[0.0], rec, [rec[-1]]])
    mpre = np.concatenate([[1.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def greedy_scored(gt: Sequence[tuple[int, Box]], dets: Sequence[Det], iou_thr: float = 0.5) -> list[tuple[float, bool]]:
    """COCO-style greedy matching by descending confidence (for AP)."""
    gboxes = [b for _g, b in gt]
    order = sorted(range(len(dets)), key=lambda i: -dets[i].conf)
    m = iou_matrix(gboxes, [dets[i].box for i in order])
    taken: set[int] = set()
    out = []
    for j, i in enumerate(order):
        best, bi = -1, iou_thr
        for g in range(len(gboxes)):
            if g not in taken and m[g, j] >= bi:
                best, bi = g, m[g, j]
        if best >= 0:
            taken.add(best)
        out.append((dets[i].conf, best >= 0))
    return out
