"""Annotated video and per-event image strips for occlusion events (display only; nothing here is evidence)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from football_fatigue.video.occlusion.engine import FrameInput
from football_fatigue.video.schema import OcclusionEvent

# BGR colours
COLOURS = {
    "box_overlap": (0, 215, 255),          # yellow
    "probable_occlusion": (0, 120, 255),   # orange
    "detection_loss": (60, 60, 235),       # red
}
GREY = (175, 175, 175)
WHITE = (255, 255, 255)
BANNER_FRAMES = 12


def event_labels(events: Sequence[OcclusionEvent]) -> dict[str, str]:
    """Short display labels E1, E2, … in event order (deterministic)."""
    return {e.event_id: f"E{i}" for i, e in enumerate(events, start=1)}


def rows_by_frame(rows: Sequence[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(int(r["frame"]), []).append(r)
    return out


def _dashed_rect(img, p1, p2, colour, thickness=2, dash=6):
    x1, y1 = p1
    x2, y2 = p2
    for (a, b) in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        length = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])))
        for s in range(0, max(length, 1), dash * 2):
            t0, t1 = s / max(length, 1), min(s + dash, length) / max(length, 1)
            q0 = (int(a[0] + (b[0] - a[0]) * t0), int(a[1] + (b[1] - a[1]) * t0))
            q1 = (int(a[0] + (b[0] - a[0]) * t1), int(a[1] + (b[1] - a[1]) * t1))
            cv2.line(img, q0, q1, colour, thickness, cv2.LINE_AA)


def draw_frame(img: np.ndarray, frame: FrameInput, frame_rows: Sequence[dict[str, Any]],
               events_by_id: dict[str, OcclusionEvent], labels: dict[str, str],
               offset=(0, 0), scale=1.0, font=0.5, hud: str | None = None) -> np.ndarray:
    """Draw all tracks (grey) and the event overlays of one frame onto ``img`` (crop offset/scale aware)."""
    ox, oy = offset

    def P(x, y):
        return int(round((x - ox) * scale)), int(round((y - oy) * scale))

    overlay = img.copy()
    member_tracks = set()
    for r in frame_rows:
        ev = events_by_id[r["event_id"]]
        colour = COLOURS[ev.event_class]
        boxes = r["boxes"]
        for pair in r["pairs"]:
            a, b = boxes.get(str(pair["a"])), boxes.get(str(pair["b"]))
            if a and b:
                ix1, iy1 = max(a["box"][0], b["box"][0]), max(a["box"][1], b["box"][1])
                ix2, iy2 = min(a["box"][2], b["box"][2]), min(a["box"][3], b["box"][3])
                if ix2 > ix1 and iy2 > iy1:
                    cv2.rectangle(overlay, P(ix1, iy1), P(ix2, iy2), colour, -1)
        for tid, info in boxes.items():
            member_tracks.add(int(tid))
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
    for tid, obs in frame.tracks.items():
        if tid in member_tracks:
            continue
        x1, y1, x2, y2 = obs.box
        cv2.rectangle(img, P(x1, y1), P(x2, y2), GREY, 1, cv2.LINE_AA)
        cv2.putText(img, str(tid), P(x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, font * 0.8, GREY, 1, cv2.LINE_AA)
    for r in frame_rows:
        ev = events_by_id[r["event_id"]]
        colour = COLOURS[ev.event_class]
        label = labels[ev.event_id]
        tops = []
        for tid, info in r["boxes"].items():
            x1, y1, x2, y2 = info["box"]
            if info["state"] == "visible":
                cv2.rectangle(img, P(x1, y1), P(x2, y2), colour, 2, cv2.LINE_AA)
            else:
                _dashed_rect(img, P(x1, y1), P(x2, y2), colour, 2)
            cv2.putText(img, tid + ("?" if info["state"] != "visible" else ""), P(x1, y2 + 12 / scale),
                        cv2.FONT_HERSHEY_SIMPLEX, font, colour, 1, cv2.LINE_AA)
            tops.append((x1, y1))
        if tops:
            x, y = min(t[0] for t in tops), min(t[1] for t in tops)
            cv2.putText(img, label, P(x, y - 6 / scale), cv2.FONT_HERSHEY_SIMPLEX, font * 1.2, colour, 2, cv2.LINE_AA)
    if hud:
        cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(img, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 1, cv2.LINE_AA)
    return img


def render_video(out_path: Path, read_frame: Callable[[int], np.ndarray | None], frames: Sequence[FrameInput],
                 events: Sequence[OcclusionEvent], rows: Sequence[dict[str, Any]], window: tuple[int, int],
                 fps: float, pts: Callable[[int], float]) -> dict[str, Any]:
    import av

    labels = event_labels(events)
    by_id = {e.event_id: e for e in events}
    by_frame = rows_by_frame(rows)
    inputs = {f.frame: f for f in frames}
    starts: dict[int, list[OcclusionEvent]] = {}
    ends: dict[int, list[OcclusionEvent]] = {}
    for e in events:
        starts.setdefault(e.start_frame, []).append(e)
        ends.setdefault(e.end_frame, []).append(e)
    container = av.open(str(out_path), mode="w")
    stream = None
    written = 0
    try:
        for fr in range(window[0], window[1] + 1):
            img = read_frame(fr)
            if img is None:
                raise RuntimeError(f"frame {fr} could not be decoded")
            img = img.copy()
            frame_rows = by_frame.get(fr, [])
            active = {by_id[r["event_id"]].event_class for r in frame_rows}
            counts = " ".join(f"{k}:{sum(1 for r in frame_rows if by_id[r['event_id']].event_class == k)}" for k in COLOURS)
            hud = (f"SoccerTrack {fr:05d}  PTS {pts(fr):.3f}s  active events  {counts}   "
                   f"[yellow box_overlap | orange probable_occlusion | red detection_loss | dashed = predicted, not detected]")
            draw_frame(img, inputs[fr], frame_rows, by_id, labels, hud=hud)
            banner = []
            for k in range(BANNER_FRAMES):
                banner += [f"START {labels[e.event_id]} {e.event_class}/{e.subtype}" for e in starts.get(fr - k, [])]
                banner += [f"END {labels[e.event_id]}" for e in ends.get(fr - k, [])]
            for i, text in enumerate(banner[:8]):
                cv2.putText(img, text, (10, 64 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
            if stream is None:
                stream = container.add_stream("libx264", rate=int(round(fps)))
                stream.width, stream.height, stream.pix_fmt = img.shape[1], img.shape[0], "yuv420p"
                stream.options = {"crf": "20", "preset": "veryfast"}
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="bgr24")):
                container.mux(packet)
            written += 1
        if stream is not None:
            for packet in stream.encode():
                container.mux(packet)
    finally:
        container.close()
    return {"path": out_path.name, "frames": written, "first_frame": window[0], "codec": "h264", "fps": fps}


def select_strip_events(events: Sequence[OcclusionEvent], window: tuple[int, int], n: int = 10) -> list[OcclusionEvent]:
    """Deterministic, class-balanced choice of events to illustrate (not a quality ranking)."""
    inside = [e for e in events if window[0] <= e.start_frame <= window[1]]
    def pick(pred, key, k):
        return sorted((e for e in inside if pred(e)), key=key)[:k]
    chosen = (
        pick(lambda e: e.event_class == "probable_occlusion" and e.subtype in ("full", "merged"),
             lambda e: (-e.measurements["max_onset_cover"], e.start_frame, e.event_id), 4)
        + pick(lambda e: e.subtype == "partial", lambda e: (-e.measurements["max_ios_widened"], e.start_frame, e.event_id), 2)
        + pick(lambda e: e.event_class == "box_overlap",
               lambda e: (-e.measurements["max_ios_widened"], -e.measurements["evidence_frames"], e.event_id), 2)
        + pick(lambda e: e.subtype == "detector_miss", lambda e: (-e.measurements["duration_frames"], e.start_frame, e.event_id), 1)
        + pick(lambda e: e.subtype == "tracker_gap", lambda e: (-e.measurements["duration_frames"], e.start_frame, e.event_id), 1)
    )
    ids = {e.event_id for e in chosen}
    rest = [e for e in inside if e.event_id not in ids]
    chosen += rest[: max(0, n - len(chosen))]
    order = {e.event_id: i for i, e in enumerate(events)}
    return sorted(chosen, key=lambda e: order[e.event_id])


def render_strip(out_path: Path, event: OcclusionEvent, label: str, read_frame: Callable[[int], np.ndarray | None],
                 frames: Sequence[FrameInput], rows: Sequence[dict[str, Any]], events_by_id: dict[str, OcclusionEvent],
                 labels: dict[str, str], target_height: int = 300) -> dict[str, Any]:
    inputs = {f.frame: f for f in frames}
    lo, hi = frames[0].frame, frames[-1].frame
    picks = [("before", max(lo, event.start_frame - 5)), ("start", event.start_frame), ("peak", event.peak_frame),
             ("end", event.end_frame), ("after", min(hi, event.end_frame + 5))]
    ev_rows = [r for r in rows if r["event_id"] == event.event_id]
    boxes = [info["box"] for r in ev_rows for info in r["boxes"].values()]
    x1, y1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x2, y2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    h = max(y2 - y1, 40.0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_h = max(h * 1.2, 60.0)
    half_w = max((x2 - x1) / 2 + h * 0.8, half_h * 0.8)
    any_frame = read_frame(event.start_frame)
    H, W = any_frame.shape[:2]
    cx1, cy1 = int(max(0, cx - half_w)), int(max(0, cy - half_h))
    cx2, cy2 = int(min(W, cx + half_w)), int(min(H, cy + half_h))
    scale = target_height / max(cy2 - cy1, 1)
    by_frame = rows_by_frame(rows)
    panels = []
    for name, fr in picks:
        img = read_frame(fr)
        crop = cv2.resize(img[cy1:cy2, cx1:cx2], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        draw_frame(crop, inputs[fr], by_frame.get(fr, []), events_by_id, labels, offset=(cx1, cy1), scale=scale, font=0.45)
        cv2.rectangle(crop, (0, 0), (crop.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(crop, f"{name} f{fr}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
        panels.append(crop)
        panels.append(np.full((crop.shape[0], 4, 3), 40, np.uint8))
    strip = np.hstack(panels[:-1])
    title = np.zeros((30, strip.shape[1], 3), np.uint8)
    flag = "  UNCERTAIN" if event.uncertain else ""
    text = (f"{label} {event.event_class}/{event.subtype} frames {event.start_frame}-{event.end_frame} "
            f"tracks {','.join(str(m.track_id) for m in event.members)} evidence={event.evidence_level}{flag}")
    cv2.putText(title, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOURS[event.event_class], 1, cv2.LINE_AA)
    out = np.vstack([title, strip])
    cv2.imwrite(str(out_path), out)
    return {"event_id": event.event_id, "label": label, "file": out_path.name, "frames": [f for _n, f in picks]}
