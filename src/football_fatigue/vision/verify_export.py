"""Programmatic checks of an annotated MP4 against its source video and model records.

1. Stream: codec, resolution, fps, frame count.
2. Alignment: for sampled frames, the output frame must equal the *same-index*
   source frame outside the drawn overlays (and differ more from neighbours).
3. Overlay: sampled boxes must have the kit colour on their drawn rectangle edge.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from football_fatigue.vision.export_video import KIT_COLORS


def verify(video: Path, annotated: Path, frames_jsonl: Path, start_frame: int = 0, samples: int = 12) -> dict:
    import av
    import cv2
    import numpy as np

    records = {}
    for line in frames_jsonl.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        records[r["frame_num"]] = r
    with av.open(str(annotated)) as c:
        s = c.streams.video[0]
        info = {"codec": s.codec_context.name, "width": s.width, "height": s.height,
                "fps": float(s.average_rate), "frames": s.frames or sum(1 for _ in c.decode(video=0))}
    out_cap, src_cap = cv2.VideoCapture(str(annotated)), cv2.VideoCapture(str(video))
    n_out = int(out_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = sorted({int(round(i)) for i in np.linspace(0, n_out - 1, samples)})
    exact, neighbour, edge_ok, edge_total = [], [], 0, 0
    for k in idx:
        out_cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, out = out_cap.read()
        diffs = {}
        for d in (-1, 0, 1):
            src_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + k + d)
            ok2, src = src_cap.read()
            if not (ok and ok2):
                continue
            mask = np.ones(out.shape[:2], bool)
            mask[:200, :600] = False  # legend
            mask[-50:, :400] = False  # frame counter
            for p in records.get(start_frame + k, {}).get("players", []):
                x1, y1, x2, y2 = (int(v) for v in p["bbox_xyxy_px"])
                mask[max(0, y1 - 22):y2 + 6, max(0, x1 - 6):x2 + 6] = False
            diffs[d] = float(np.abs(out.astype(int) - src.astype(int))[mask].mean())
        exact.append(diffs.get(0, np.nan))
        neighbour.append(min(diffs.get(-1, 9e9), diffs.get(1, 9e9)))
        for p in records.get(start_frame + k, {}).get("players", [])[:8]:
            x1, y1, x2, y2 = (int(round(v)) for v in p["bbox_xyxy_px"])
            color = np.array(KIT_COLORS[p.get("kit_color") or "unknown"])
            xm = (x1 + x2) // 2
            if 2 <= y1 < out.shape[0] - 2:
                edge_total += 1
                edge_ok += bool(np.abs(out[y1 - 1:y1 + 1, xm].astype(int) - color).sum(axis=1).min() < 60)
    return {
        **info,
        "sampled_frames": idx,
        "mean_abs_diff_vs_same_index_source": round(float(np.nanmean(exact)), 3),
        "mean_abs_diff_vs_nearest_neighbour_source": round(float(np.mean(neighbour)), 3),
        "same_index_is_closest": bool(np.nanmean(exact) <= np.mean(neighbour)),
        "box_edges_with_expected_kit_colour": f"{edge_ok}/{edge_total}",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", type=Path)
    ap.add_argument("annotated", type=Path)
    ap.add_argument("frames", type=Path)
    ap.add_argument("--start-frame", type=int, default=0)
    args = ap.parse_args(argv)
    print(json.dumps(verify(args.video, args.annotated, args.frames, args.start_frame), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
