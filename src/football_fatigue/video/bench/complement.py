"""M13: do other *local* detectors find the GT players the production cache misses? (existing runs only, CPU)

On the frozen 92-frame benchmark set (frames-v1), every GT player is matched greedily (IoU >= 0.5, conf > floor,
person-mapped boxes) to the production cache and to each other completed detector run. For the GT boxes the production
cache misses, the table counts how many the other run finds, by GT-derived visibility, GT height and light. The reverse
direction (production finds, other misses) is reported too, so complementarity is not one-sided.

Caveats: the 92 frames are stratified towards hard cases (not representative of a match); runs differ in inference
size (production cache imgsz 1920; COCO runs 1280 or 640) - recorded per run. Output (write-once):
``<out_root>/_bench/complement-m13-v1.json``.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from football_fatigue.video.bench.detectors import load_dets, runs_root
from football_fatigue.video.bench.evaluate import PERSON, load_gt, match_frame

VERSION = "complement-m13-v1"
BASE = "yolo26l-football__production-cache"
LOW_LIGHT = "soccertrack-117092"


def _found(gt: list[dict], boxes: list[list], floor: float) -> list[bool]:
    preds = [b for b in boxes if b[6] in PERSON and b[4] > floor]
    _p2g, g2p = match_frame([g["box"] for g in gt], preds, 0.5)
    return [i >= 0 for i in g2p]


def run(data_root: Path, out_root: Path, floor: float = 0.1) -> Path:
    out_root = Path(out_root)
    dest = out_root / "_bench" / f"{VERSION}.json"
    if dest.exists():
        raise SystemExit(f"output already exists (write-once): {dest}")
    gt = load_gt(data_root, out_root)
    base = load_dets(out_root, BASE)
    keys = sorted(p.name for p in runs_root(out_root).iterdir() if (p / "dets.jsonl").exists() and p.name != BASE)
    res: dict[str, Any] = {}
    for key in keys:
        alt = load_dets(out_root, key)
        man = json.loads((runs_root(out_root) / key / "run_manifest.json").read_text(encoding="utf-8"))
        frames = [k for k in gt if k in base and k in alt]           # only frames both runs processed
        c: dict[str, Counter] = defaultdict(Counter)
        for k in frames:
            g = gt[k]
            fb, fa = _found(g, base[k]["boxes"], floor), _found(g, alt[k]["boxes"], floor)
            for gi, x in enumerate(g):
                keys_ = ["all", f"vis:{x['vis']}", f"size:{x['size']}", "light:low" if k[0] == LOW_LIGHT else "light:normal"]
                for s in keys_:
                    c[s]["gt"] += 1
                    c[s]["base_missed"] += not fb[gi]
                    c[s]["base_missed_alt_found"] += (not fb[gi]) and fa[gi]
                    c[s]["alt_missed"] += not fa[gi]
                    c[s]["alt_missed_base_found"] += (not fa[gi]) and fb[gi]
        res[key] = {"frames": len(frames), "image_size": (man.get("protocol_spec") or {}).get("image_size"),
                    "device": man.get("device"), "subset": man.get("subset"), "counts": {k: dict(v) for k, v in c.items()}}
    dest.write_text(json.dumps({"version": VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                                "baseline": BASE, "floor": floor, "matching": "greedy by confidence, IoU >= 0.5, person-mapped",
                                "runs": res}, indent=1), encoding="utf-8")
    return dest
