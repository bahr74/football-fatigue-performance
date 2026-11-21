"""Drawing and per-frame analysis shared by the benchmark dashboard section and the failure gallery.

Colours (BGR): matched prediction = green, false positive = red, duplicate = magenta, missed GT = orange (dashed),
GT = thin white. Labels carry confidences; nothing is drawn for a detector that has no completed run.
"""

from __future__ import annotations

from typing import Any, Sequence

import cv2
import numpy as np

from football_fatigue.video.bench.evaluate import match_frame
from football_fatigue.video.detection.stages import iou_matrix
from football_fatigue.vision.mot_eval import load_reference  # noqa: F401  (re-export for callers)

GREEN, RED, MAGENTA, ORANGE, WHITE, CYAN, YELLOW = (60, 200, 60), (40, 40, 230), (200, 60, 200), (0, 150, 255), (235, 235, 235), (230, 200, 40), (40, 220, 240)


def classify(gt: Sequence[dict], preds: Sequence[Sequence], conf: float, thr: float = 0.5) -> dict[str, list]:
    """Split one frame into matched predictions, false positives, duplicates and missed GT."""
    ps = [p for p in preds if p[4] > conf]
    p2g, g2p = match_frame([g["box"] for g in gt], ps, thr)
    m = iou_matrix([tuple(g["box"]) for g in gt], [tuple(p[:4]) for p in ps]) if gt and ps else None
    out = {"tp": [], "fp": [], "dup": [], "missed": [g for g, x in zip(gt, g2p) if x < 0], "gt": list(gt)}
    for i, (p, x) in enumerate(zip(ps, p2g)):
        if x >= 0:
            out["tp"].append(p)
        elif m is not None and m[:, i].max() >= thr:
            out["dup"].append(p)
        else:
            out["fp"].append(p)
    return out


def _dashed(img, box, color, t=2, dash=8):
    x1, y1, x2, y2 = (int(round(v)) for v in box[:4])
    for a, b in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        n = max(1, int(np.hypot(b[0] - a[0], b[1] - a[1]) // dash))
        for k in range(0, n, 2):
            p = (int(a[0] + (b[0] - a[0]) * k / n), int(a[1] + (b[1] - a[1]) * k / n))
            q = (int(a[0] + (b[0] - a[0]) * min(k + 1, n) / n), int(a[1] + (b[1] - a[1]) * min(k + 1, n) / n))
            cv2.line(img, p, q, color, t)


def draw_detections(img: np.ndarray, split: dict[str, list], scale: float = 1.0, labels: bool = True) -> np.ndarray:
    out = img.copy()
    t = max(1, int(round(2 * scale)))
    fs = 0.45 * scale
    for g in split["gt"]:
        x1, y1, x2, y2 = (int(round(v)) for v in g["box"])
        cv2.rectangle(out, (x1, y1), (x2, y2), WHITE, 1)
    for g in split["missed"]:
        _dashed(out, g["box"], ORANGE, t + 1)
    for key, col in (("tp", GREEN), ("fp", RED), ("dup", MAGENTA)):
        for p in split[key]:
            x1, y1, x2, y2 = (int(round(v)) for v in p[:4])
            cv2.rectangle(out, (x1, y1), (x2, y2), col, t)
            if labels:
                lab = f"{p[4]:.2f}" + ("" if p[6] in ("person", "player") else f" {p[6][:3]}")
                cv2.putText(out, lab, (x1, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, fs, col, max(1, t - 1), cv2.LINE_AA)
    return out


def crop(img: np.ndarray, box: Sequence[float], heights: float = 4.0, out_h: int = 220) -> np.ndarray:
    h = max(box[3] - box[1], 8)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    half = heights * h / 2
    H, W = img.shape[:2]
    x1, y1 = int(max(0, cx - half)), int(max(0, cy - half))
    x2, y2 = int(min(W, cx + half)), int(min(H, cy + half))
    c = img[y1:y2, x1:x2]
    if c.size == 0:
        return np.zeros((out_h, out_h, 3), np.uint8)
    return cv2.resize(c, (int(out_h * c.shape[1] / c.shape[0]), out_h), interpolation=cv2.INTER_CUBIC)


def legend_strip(width: int, text_left: str = "") -> np.ndarray:
    strip = np.full((26, width, 3), 24, np.uint8)
    x = 8
    if text_left:
        cv2.putText(strip, text_left, (x, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
        x += 12 + 9 * len(text_left)
    for lab, col in (("matched", GREEN), ("false pos.", RED), ("duplicate", MAGENTA), ("missed GT", ORANGE), ("GT", WHITE)):
        cv2.rectangle(strip, (x, 7), (x + 14, 19), col, 2)
        cv2.putText(strip, lab, (x + 19, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
        x += 30 + 8 * len(lab)
    return strip


def id_events(gt: dict[int, list], tracks: dict[int, list], lo: int, hi: int, thr: float = 0.5) -> list[dict[str, Any]]:
    """Per GT player, frame-level ID switches (matched track id changes) and fragmentations (matched -> unmatched ->
    matched) inside lo..hi, with a Hungarian IoU match per frame (the same idea as CLEAR MOT; counts can differ
    slightly from TrackEval, which also enforces match continuity)."""
    from scipy.optimize import linear_sum_assignment

    last_tid: dict[int, int] = {}
    last_seen: dict[int, int] = {}
    unmatched_since: dict[int, int] = {}
    events: list[dict[str, Any]] = []
    for fr in range(lo, hi + 1):
        g = gt.get(fr, [])
        t = tracks.get(fr, [])
        if not g:
            continue
        gb = [b for _i, b in g]
        tb = [tuple(x[1:5]) for x in t]
        match: dict[int, int] = {}
        if tb:
            m = iou_matrix(gb, tb)
            r, c = linear_sum_assignment(-m)
            match = {g[i][0]: int(t[j][0]) for i, j in zip(r, c) if m[i, j] >= thr}
        for gid, _b in g:
            if gid in match:
                tid = match[gid]
                if gid in last_tid and last_tid[gid] != tid:
                    events.append({"frame": fr, "gt_id": gid, "kind": "id_switch", "from": last_tid[gid], "to": tid})
                if gid in unmatched_since and gid in last_seen:
                    events.append({"frame": fr, "gt_id": gid, "kind": "fragment_recovered", "gap": fr - unmatched_since[gid],
                                   "track": tid, "same_id": last_tid.get(gid) == tid})
                unmatched_since.pop(gid, None)
                last_tid[gid], last_seen[gid] = tid, fr
            elif gid in last_seen and gid not in unmatched_since:
                unmatched_since[gid] = fr
    return events
