"""Heavy overlap, merge NMS, size and low light: what the post-merge cache can and cannot show (``_diag/heavy-v1.json``).

Measured from cached data:
* GT player pairs with IoU >= 0.5 vs cached person-box pairs with IoU >= 0.5 (the tile merge suppresses overlapping
  boxes at IoU 0.5, so the cache cannot hold two strongly overlapping players: a ceiling on heavy-overlap recall).
* T0 outcome per GT-derived visibility class and the 'miss' geometry hints (merge_suspect / poor_localisation / none).
* Confidence of the matched cached box per visibility class (does the 0.4 threshold matter for overlapped players?).
Not measurable without the raw per-tile capture: whether a missed overlapped player was ever proposed by the model.
The capture plan proposes the 125-frame windows with the most 'merge_suspect' misses (T0) per match.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from football_fatigue.video.detection.diagnose import iter_cache
from football_fatigue.video.detection.stages import iou_matrix
from football_fatigue.video.failures.pipeline import CODES, load_episodes, load_summary, out_dir
from football_fatigue.video.sources import SOURCES

DEV = ("soccertrack-117093", "soccertrack-118575", "soccertrack-117092", "soccertrack-118576")
WINDOW = 125


def _pairs(boxes: list) -> int:
    if len(boxes) < 2:
        return 0
    m = iou_matrix(boxes, boxes)
    return int(((m >= 0.5).sum() - len(boxes)) // 2)


def analyse(data_root: Path, out_root: Path) -> Path:
    from football_fatigue.vision.mot_eval import load_reference

    res: dict[str, Any] = {"version": "heavy-v1", "matches": {}}
    for sid in DEV:
        gt, _ = load_reference(Path(data_root) / SOURCES[sid]["gt"], 0)
        gt_pairs = det_pairs = 0
        codes_conf: dict[str, list] = defaultdict(list)
        for r in (json.loads(x) for x in (out_dir(out_root, sid) / "T0" / "frames.jsonl").open(encoding="utf-8")):
            for gid, code, _tid, conf in r["o"]:
                if conf is not None:
                    codes_conf[CODES[code]].append(conf)
        for fr, dets in iter_cache(data_root, sid):
            gt_pairs += _pairs([tuple(b) for _g, b in gt.get(fr, [])])
            det_pairs += _pairs([d.box for d in dets if d.person])
        s = load_summary(out_root, sid, "T0")
        e = load_episodes(out_root, sid, "T0")
        dens = Counter()
        for ep in e["episodes"]:
            n = ep["miss_hints"].get("merge_suspect", 0)
            if n:
                for f in range(ep["start"], ep["end"] + 1):
                    dens[f] += n / ep["frames"]
        frames = sorted(dens)
        best, best_n = None, 0.0
        if frames:
            arr = np.zeros(max(frames) + WINDOW + 1)
            for f, v in dens.items():
                arr[f] = v
            cs = np.concatenate([[0], np.cumsum(arr)])
            sums = cs[WINDOW:] - cs[:-WINDOW]
            best = int(np.argmax(sums))
            best_n = float(sums[best])
        res["matches"][sid] = {
            "role": SOURCES[sid]["role"], "frames": s["frames"],
            "gt_pairs_iou_ge_0_5": gt_pairs, "cached_person_pairs_iou_ge_0_5": det_pairs,
            "t0_by_visibility": s["by_visibility"], "t0_miss_hints_by_visibility": s["miss_hints_by_visibility"],
            "t0_by_size": s["by_size"],
            "matched_box_conf_median_by_code": {k: round(float(np.median(v)), 3) for k, v in codes_conf.items() if v},
            "capture_window": None if best is None else {"start": best, "frames": WINDOW,
                                                         "merge_suspect_gt_frames_in_window": round(best_n, 1)},
        }
    res["capture_plan"] = {
        "tool": "python -m football_fatigue.video.detection capture --approved (video.detection.capture; NOT run)",
        "what": "production detector settings from each cache manifest (YOLO26-L, 1280x1280 tiles, 20 % overlap, imgsz "
                "1920, batch 2) on the windows below; stores every per-tile box BEFORE the merge (conf >= 0.05) with its "
                "tile and turf decision, plus the cached post-merge boxes for the same frames",
        "windows": [{"source_id": sid, **v["capture_window"]} for sid, v in res["matches"].items() if v["capture_window"]],
        "cost": "about 4 min per 125 frames on the GTX 1650 Ti (measured 1.77 s/frame production speed), < 3 GB VRAM, "
                "~16 min for 4 windows; output <out_root>/<source>/detection/captures/<source>-<start>-125/",
        "answers": ["share of 'miss' observations with a raw per-tile box (IoU >= 0.5) that the merge removed (= NMS loss)",
                    "share with no raw box at any confidence >= 0.05 (= model miss)",
                    "effect of merge IoU 0.5 vs 0.6/0.7 replayed offline on the captured raw boxes (no new inference)"],
    }
    dest = Path(out_root) / "_diag" / "heavy-v1.json"
    dest.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return dest
