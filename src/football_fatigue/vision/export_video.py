"""Burn model-predicted tracking overlays into an H.264 MP4.

Draws only records from a saved run's ``frames.jsonl`` (model output); MOT
reference annotations are never read. Frames are drawn by their source frame
index, so overlays are synchronised by construction. Output keeps the source
resolution, frame order and fps; its timeline starts at 0 (the source
container starts at its first-frame timestamp, ~0.979 s for sample 117093).
"""

from __future__ import annotations

import argparse
import json
import time
from fractions import Fraction
from pathlib import Path

# BGR
KIT_COLORS = {
    "blue": (255, 140, 30),
    "white": (255, 255, 255),
    "other": (0, 215, 255),
    "unknown": (160, 160, 160),
}
KIT_TAGS = {"blue": "B", "white": "W", "other": "O", "unknown": "?"}
LEGEND = (
    ("blue", "Blue kit"),
    ("white", "White kit"),
    ("other", "Other (goalkeeper / referee / staff)"),
    ("unknown", "Unknown"),
)


def _text(img, text, org, scale, color, cv2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_frame(img, record: dict | None, frame_index: int, fps: float, cv2) -> None:
    """Draw boxes, IDs/kit tags, legend and frame counter in place."""
    for p in (record or {}).get("players", []):
        box = p.get("bbox_xyxy_px")
        if not box:
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        kit = p.get("kit_color") or "unknown"
        color = KIT_COLORS.get(kit, KIT_COLORS["unknown"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 4)  # contrasting outline
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        _text(img, f"{p['track_id']}{KIT_TAGS.get(kit, '?')}", (x1, max(14, y1 - 5)), 0.55, color, cv2)
    # legend
    x0, y0, w, h = 20, 20, 560, 168
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    _text(img, "Model output: YOLOv8n + ByteTrack (colour labels only)", (x0 + 12, y0 + 28), 0.62, (255, 255, 255), cv2)
    for i, (kit, label) in enumerate(LEGEND):
        y = y0 + 58 + i * 26
        cv2.rectangle(img, (x0 + 12, y - 14), (x0 + 36, y + 6), (0, 0, 0), 3)
        cv2.rectangle(img, (x0 + 12, y - 14), (x0 + 36, y + 6), KIT_COLORS[kit], -1)
        _text(img, f"{KIT_TAGS[kit]} = {label}", (x0 + 48, y + 3), 0.6, (255, 255, 255), cv2)
    _text(img, "IDs are tracker labels, not verified player identities", (x0 + 12, y0 + h - 8), 0.5, (200, 200, 200), cv2)
    t = frame_index / fps
    _text(img, f"frame {frame_index}  t={int(t // 60):02d}:{t % 60:04.1f}", (20, img.shape[0] - 20), 0.7, (255, 255, 255), cv2)


def export(
    video: Path,
    frames_jsonl: Path,
    out: Path,
    *,
    start_frame: int = 0,
    max_frames: int | None = None,
    crf: int = 20,
    preset: str = "fast",
    scale_width: int | None = None,
) -> dict:
    import av
    import cv2

    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    records: dict[int, dict] = {}
    with frames_jsonl.open(encoding="utf-8") as stream:
        for line in stream:
            rec = json.loads(line)
            records[rec["frame_num"]] = rec

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_w, out_h = (scale_width, int(round(height * scale_width / width / 2)) * 2) if scale_width else (width, height)
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    last = total if max_frames is None else min(total, start_frame + max_frames)

    out.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(out), "w", options={"movflags": "+faststart"})
    stream = container.add_stream("libx264", rate=Fraction(int(round(fps * 1000)), 1000))
    stream.width, stream.height, stream.pix_fmt = out_w, out_h, "yuv420p"
    stream.options = {"crf": str(crf), "preset": preset}
    missing = 0
    t0 = time.perf_counter()
    for n in range(start_frame, last):
        ok, bgr = cap.read()
        if not ok:
            break
        rec = records.get(n)
        missing += rec is None
        draw_frame(bgr, rec, n, fps, cv2)
        if scale_width:
            bgr = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
        frame = av.VideoFrame.from_ndarray(bgr, format="bgr24")
        frame.pts = n - start_frame
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    cap.release()
    return {
        "output": str(out), "frames_written": last - start_frame, "frames_without_model_record": missing,
        "source_fps": fps, "source_resolution": [width, height], "output_resolution": [out_w, out_h],
        "start_frame": start_frame, "encode_seconds": round(time.perf_counter() - t0, 1),
        "codec": "libx264", "crf": crf, "preset": preset, "size_bytes": out.stat().st_size,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("frames", type=Path, help="frames.jsonl with model output")
    ap.add_argument("out", type=Path, help="new .mp4 path (never overwritten)")
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--preset", default="fast")
    ap.add_argument("--scale-width", type=int, help="optional downscale (preview), e.g. 2048")
    args = ap.parse_args(argv)
    info = export(args.video, args.frames, args.out, start_frame=args.start_frame, max_frames=args.max_frames,
                  crf=args.crf, preset=args.preset, scale_width=args.scale_width)
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
