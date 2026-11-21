"""Raw per-tile detector capture (GPU). NOT run without explicit user approval (CLI flag ``--approved``).

Runs the production detector settings recorded in the source's detection cache (same weights, 1280x1280 tiles, 20 %
overlap, imgsz 1920, batch 2) on a short window and stores every per-tile box *before* merging, with its tile and its
turf-filter decision, plus the cached post-merge boxes for the same frames (so policy A0 can be checked against the
cache). Output: ``<out_root>/<source>/detection/captures/<source>-<start>-<n>/{raw_tiles.jsonl.gz,capture_manifest.json}``.

Estimated cost (measured production speed 1.77 s/frame on the GTX 1650 Ti): about 4 min per 125 frames, < 3 GB VRAM.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import time
from pathlib import Path

from football_fatigue.video.detection.diagnose import cache_dir, iter_cache
from football_fatigue.video.sources import SOURCES, ExactFrameReader, guard_not_reserved

CAPTURE_FLOOR = 0.05


def capture(data_root: Path, out_root: Path, source_id: str, start: int, frames: int) -> Path:
    import numpy as np
    from football_fatigue.vision.detector import YOLOPlayerDetector
    from football_fatigue.vision.models import BoundingBox

    spec = SOURCES[source_id]
    guard_not_reserved(source_id, spec["match_id"], spec["video"], spec["cache_run"])
    cm = json.loads((cache_dir(data_root, source_id) / "cache_manifest.json").read_text(encoding="utf-8"))
    dc = cm["detector"]
    weights = Path(data_root).parent / dc["model_path"]
    if hashlib.sha256(weights.read_bytes()).hexdigest() != dc["weights_sha256"]:
        raise RuntimeError("detector weights differ from the cache's weights")
    det = YOLOPlayerDetector(model_path=str(weights), confidence_threshold=CAPTURE_FLOOR, device="cuda:0",
                             image_size=dc["image_size"], tile_width=dc["tile_width"], tile_height=dc["tile_height"],
                             tile_overlap=dc["tile_overlap"], half=dc["half"], merge_iou=dc["merge_iou"],
                             tile_batch=dc["tile_batch"], tile_merge=dc["tile_merge"]["policy"],
                             cross_role_iou=dc["tile_merge"]["cross_role_iou"])
    det.load_model()
    dest = Path(out_root) / source_id / "detection" / "captures" / f"{source_id}-{start}-{frames}"
    if dest.exists():
        raise FileExistsError(dest)
    dest.mkdir(parents=True)
    cached: dict[int, list] = {}
    for fr, ds in iter_cache(data_root, source_id, start):
        if fr >= start + frames:
            break
        cached[fr] = [list(d.box) for d in ds if d.conf > dc["confidence_floor"]]
    reader = ExactFrameReader(Path(data_root) / spec["video"])
    t0 = time.perf_counter()
    try:
        with gzip.open(dest / "raw_tiles.jsonl.gz", "wt", encoding="utf-8") as fh:
            for fr in range(start, start + frames):
                bgr = reader.frame(fr)
                rgb = np.ascontiguousarray(bgr[:, :, ::-1])
                h, w = rgb.shape[:2]
                tiles = det._tile_grid(w, h)
                crops = [np.ascontiguousarray(bgr[y:y + th, x:x + tw]) for x, y, tw, th in tiles]
                results = det._run_tiles(crops, dict(conf=CAPTURE_FLOOR, device=det.device, imgsz=det.image_size,
                                                     half=det.half, verbose=False))
                items = []
                for ti, (r, (xo, yo, _w, _h)) in enumerate(zip(results, tiles)):
                    if r.boxes is None:
                        continue
                    for c, s, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
                        items.append([int(c), round(float(s), 5), b[0] + xo, b[1] + yo, b[2] + xo, b[3] + yo, ti])
                turf = [det._looks_like_player(rgb, BoundingBox(*it[2:6])) for it in items]
                fh.write(json.dumps({"frame": fr, "size": [w, h], "tiles": tiles, "items": items, "turf": turf,
                                     "cached_boxes": cached.get(fr)}) + "\n")
    finally:
        reader.close()
    (dest / "capture_manifest.json").write_text(json.dumps({
        "source_id": source_id, "start": start, "frames": frames, "capture_floor": CAPTURE_FLOOR,
        "cache_floor": dc["confidence_floor"], "detector": dc, "class_names": dc["class_names"],
        "seconds": round(time.perf_counter() - t0, 1),
    }, indent=1), encoding="utf-8")
    return dest
