"""M13 detector recovery without retraining (CPU): what the existing post-merge cache can and cannot give back.

1. **Confidence x size table** (cache vs GT, per role). GT boxes are matched one-to-one (Hungarian, IoU >= 0.5) to
   cached person boxes (player/referee/goalkeeper labels, conf > 0.1, turf decision ignored). For each confidence bin
   and GT-height bin: matched GT observations; for each confidence bin and box-height bin: unmatched cached person
   boxes ("unmatched" = false positive *or* an unlabelled referee/staff member - GT annotates the 22 players only).
   This bounds what a (size-dependent) threshold can recover: only GT boxes that already have a cached box.
2. **Gap table** (saved pipeline-v1 T0 episodes): "no box" episodes by length and by whether the same temporary ID
   continued afterwards. Short gaps are what track interpolation could bridge - as a *prediction*, never a detection.

Output (write-once): ``<out_root>/_diag/recovery-m13-v1/summary.json``.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from football_fatigue.video.detection.diagnose import iter_cache
from football_fatigue.video.failures.pipeline import load_episodes
from football_fatigue.video.sources import SOURCES, guard_not_reserved

VERSION = "recovery-m13-v1"
DEV = ("soccertrack-117093", "soccertrack-118575", "soccertrack-117092", "soccertrack-118576")
CONF_BINS = ((0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 1.01))
H_BINS = ((0, 35), (35, 50), (50, 80), (80, 10_000))
PERSON = ("player", "referee", "goalkeeper")


CONF_LABELS = ("0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", ">0.6")
H_LABELS = ("<35", "35-50", "50-80", ">=80")


def _bin(v: float, bins) -> str:
    """Confidence bins are (lo, hi]; height bins are [lo, hi)."""
    labels = CONF_LABELS if bins is CONF_BINS else H_LABELS
    for (lo, hi), name in zip(bins, labels):
        if (lo < v <= hi) if bins is CONF_BINS else (lo <= v < hi):
            return name
    return "out"


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a[:, None, :], b[None, :, :]
    ix = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    iy = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = ix * iy
    u = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1]) + (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1]) - inter
    return np.where(u > 0, inter / np.where(u > 0, u, 1), 0)


def table(data_root: Path, sid: str) -> dict[str, Any]:
    from scipy.optimize import linear_sum_assignment

    from football_fatigue.vision.mot_eval import load_reference

    spec = SOURCES[sid]
    guard_not_reserved(sid, spec["match_id"], spec["cache_run"], spec["gt"])
    gt, _ = load_reference(Path(data_root) / spec["gt"], 0)
    matched = defaultdict(Counter)        # gt height bin -> conf bin counts
    unmatched = defaultdict(Counter)      # box height bin -> conf bin counts
    nobox = Counter()
    gt_total = Counter()
    for fr, dets in iter_cache(data_root, sid):
        g = gt.get(fr, [])
        if not g:
            continue
        gb = np.array([b for _i, b in g], float)
        ds = [d for d in dets if d.label in PERSON]
        db = np.array([d.box for d in ds], float) if ds else np.zeros((0, 4))
        hits: dict[int, int] = {}
        if len(ds):
            iou = _iou(gb, db)
            r, c = linear_sum_assignment(-iou)
            hits = {int(i): int(j) for i, j in zip(r, c) if iou[i, j] >= 0.5}
        for i, b in enumerate(gb):
            hb = _bin(b[3] - b[1], H_BINS)
            gt_total[hb] += 1
            if i in hits:
                matched[hb][_bin(ds[hits[i]].conf, CONF_BINS)] += 1
            else:
                nobox[hb] += 1
        used = set(hits.values())
        for j, d in enumerate(ds):
            if j not in used:
                unmatched[_bin(d.box[3] - d.box[1], H_BINS)][_bin(d.conf, CONF_BINS)] += 1
    return {"gt_observations": dict(gt_total), "no_box": dict(nobox),
            "matched_by_gt_height": {k: dict(v) for k, v in matched.items()},
            "unmatched_by_box_height": {k: dict(v) for k, v in unmatched.items()}}


def gaps(out_root: Path, sid: str) -> dict[str, Any]:
    eps = (load_episodes(out_root, sid, "T0") or {}).get("episodes", [])
    out: dict[str, Counter] = defaultdict(Counter)
    for e in eps:
        if e["cause"] != "miss":
            continue
        n = e["frames"]
        lb = "1" if n == 1 else "2-5" if n <= 5 else "6-25" if n <= 25 else ">25"
        out[lb][e["id_after_gap"] or "not_recovered"] += 1
        out[lb]["gt_frames"] += n
    return {k: dict(v) for k, v in out.items()}


def run(data_root: Path, out_root: Path) -> Path:
    dest = Path(out_root) / "_diag" / VERSION
    if (dest / "summary.json").exists():
        raise SystemExit(f"output already exists (write-once): {dest}")
    res = {}
    for sid in DEV:
        res[sid] = {"role": SOURCES[sid]["role"], "conf_size": table(data_root, sid), "no_box_gaps_T0": gaps(out_root, sid)}
        print(sid, flush=True)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "summary.json").write_text(json.dumps({
        "version": VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(), "matches": res,
        "rules": {"matching": "Hungarian, GT vs cached person boxes (conf > 0.1, turf ignored), IoU >= 0.5",
                  "unmatched": "cached person box matched to no GT player: false positive OR unlabelled referee/staff",
                  "gaps": "T0 'no box' episodes; id_after_gap from pipeline-v1 (same / changed / not recovered)"}},
        indent=1), encoding="utf-8")
    return dest
