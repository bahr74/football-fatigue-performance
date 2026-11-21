"""Detector benchmark against SoccerTrack MOT annotations on a short segment.

Protocol (documented in the report): MOT rows are ``frame,id,x,y,w,h`` with
0-based frame numbers matching the decoded video frame index (verified by the
caller via frame-count equality). Predictions are matched greedily in
descending confidence to unmatched reference boxes at IoU >= ``--iou``.
MOT contains players only, so referees/staff/spectators detected as ``person``
count as false positives here. Reference boxes are never used as predictions.

Config syntax: ``IMGSZ[:TILE_WIDTH][!]`` e.g. ``640``, ``1280``, ``1280:1280``;
a trailing ``!`` feeds channel-swapped frames (the legacy RGB-to-YOLO path).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path


def load_mot(path: Path) -> dict[int, list[tuple[float, float, float, float]]]:
    frames: dict[int, list[tuple[float, float, float, float]]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            x, y, w, h = (float(v) for v in row[2:6])
            frames.setdefault(int(float(row[0])), []).append((x, y, x + w, y + h))
    return frames


def load_mot_ids(path: Path) -> dict[int, list[tuple[int, tuple[float, float, float, float]]]]:
    frames: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            x, y, w, h = (float(v) for v in row[2:6])
            frames.setdefault(int(float(row[0])), []).append((int(float(row[1])), (x, y, x + w, y + h)))
    return frames


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(preds, refs, threshold: float) -> tuple[int, int, int]:
    """Return (tp, fp, fn); ``preds`` is a list of (conf, box)."""
    used: set[int] = set()
    tp = 0
    for _, box in sorted(preds, key=lambda p: -p[0]):
        best, best_i = threshold, -1
        for i, ref in enumerate(refs):
            if i in used:
                continue
            value = iou(box, ref)
            if value >= best:
                best, best_i = value, i
        if best_i >= 0:
            used.add(best_i)
            tp += 1
    return tp, len(preds) - tp, len(refs) - tp


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("mot", type=Path)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--configs", nargs="+", default=["640", "960", "1280", "1280:1280"])
    ap.add_argument("--start-frame", type=int, default=3000)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--step", type=int, default=5, help="evaluate every Nth frame")
    ap.add_argument("--confidence", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--half", action="store_true", help="FP16; produces NaN boxes with tiling on GTX 16xx, avoid")
    ap.add_argument("--keypoints", type=Path, help="Keypoints JSON; detections whose feet fall outside the dilated pitch outline are dropped")
    ap.add_argument("--mask-margin", type=int, default=60)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    import cv2
    import torch

    from football_fatigue.vision.detector import YOLOPlayerDetector

    refs = load_mot(args.mot)
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    alignment = {
        "video_frames": total, "mot_frames": len(refs),
        "mot_min": min(refs), "mot_max": max(refs),
        "aligned_by_count": total == len(refs) and min(refs) == 0 and max(refs) == total - 1,
    }
    end = args.start_frame + int(args.seconds * fps)
    sample = []
    t0 = time.perf_counter()
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    for n in range(args.start_frame, end):
        ok, bgr = cap.read()
        if not ok:
            break
        if (n - args.start_frame) % args.step == 0:
            sample.append((n, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    decode_ms = (time.perf_counter() - t0) * 1000 / max(1, end - args.start_frame)
    cap.release()

    polygon = None
    if args.keypoints:
        from football_fatigue.vision.calibration import pitch_polygon_from_keypoints
        polygon = pitch_polygon_from_keypoints(str(args.keypoints), args.mask_margin)
    report = {"pitch_mask": bool(polygon), "alignment": alignment, "frames_evaluated": len(sample), "iou": args.iou,
              "confidence": args.confidence, "decode_ms_per_frame_incl_skipped": round(decode_ms, 2),
              "configs": {}}
    for spec in args.configs:
        legacy = spec.endswith("!")
        size, _, tile = spec.rstrip("!").partition(":")
        det = YOLOPlayerDetector(
            str(args.weights), args.confidence, args.device, image_size=int(size),
            tile_width=int(tile) if tile else None, half=args.half,
        )
        det.load_model()
        det.detect(sample[0][1])  # warm-up
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        tp = fp = fn = 0
        elapsed = 0.0
        for n, rgb in sample:
            frame = rgb[:, :, ::-1].copy() if legacy else rgb
            t = time.perf_counter()
            dets = det.detect(frame, pitch_mask_polygon=polygon)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - t
            preds = [(d.confidence, (d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2)) for d in dets
                     if d.class_name.value in ("person", "player", "goalkeeper", "referee")]
            a, b, c = match(preds, refs.get(n, []), args.iou)
            tp, fp, fn = tp + a, fp + b, fn + c
        report["configs"][spec] = {
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None,
            "tp": tp, "fp": fp, "fn": fn,
            "infer_ms_per_frame": round(elapsed * 1000 / len(sample), 1),
            "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None,
        }
        print(spec, report["configs"][spec], flush=True)
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
