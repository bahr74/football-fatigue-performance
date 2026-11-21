"""Focus-layout review video: an overview of the whole panorama with event markers, plus two enlarged, fixed crops
of the most relevant active events (readable player boxes, temporary track IDs, observed vs predicted boxes,
short motion trails), a compact legend and no event list over the pitch. Display only; nothing here is evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from football_fatigue.video import overlay as ov

CANVAS_W, CANVAS_H = 1920, 1120
TOP_MAX_H = 470
HEADER_H, LEGEND_H, CAPTION_H = 34, 34, 30
PANEL_W = 952
TRAIL_FRAMES = 12
RANK = {"probable_occlusion": 0, "box_overlap": 1, "tracker_gap": 2, "detector_miss": 2, "uncertain": 3}


def _put(img, text, org, scale=0.6, colour=ov.WHITE, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def assign_slots(slots: list[str | None], active: Sequence[dict[str, Any]]) -> list[str | None]:
    """Keep events in their panel while active; fill free panels by rank (probable > overlap > loss), then start."""
    ids = {e["event_id"] for e in active}
    slots = [s if s in ids else None for s in slots]
    for e in sorted(active, key=lambda e: (RANK[ov.category_of(e)], e["start_frame"], e["event_id"])):
        if e["event_id"] in slots:
            continue
        if None in slots:
            slots[slots.index(None)] = e["event_id"]
    return slots


def caption(event: dict[str, Any], label: str, frame: int) -> str:
    phase = "START" if frame == event["start_frame"] else ("END" if frame == event["end_frame"] else
                                                           f"{frame - event['start_frame'] + 1}/{event['end_frame'] - event['start_frame'] + 1}")
    unc = "  [uncertain]" if event.get("uncertain") else ""
    return f"{label}  {ov.CATEGORY_LABELS[ov.category_of(event, uncertain_row=False)]} ({event['subtype']}){unc}  {phase}  -  {ov.describe(event)}"


def render_focus_video(out_path: Path, read_frame: Callable[[int], np.ndarray | None], frames: Sequence[Any],
                       events: Sequence[Any], rows: Sequence[dict[str, Any]], window: tuple[int, int],
                       fps: float, pts: Callable[[int], float]) -> dict[str, Any]:
    import av

    evs = [json.loads(e.to_json()) if hasattr(e, "to_json") else e for e in events]
    labels = {e["event_id"]: f"E{i}" for i, e in enumerate(evs, start=1)}
    by_id = {e["event_id"]: e for e in evs}
    rows_by_frame: dict[int, list[dict[str, Any]]] = {}
    rows_by_event: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        rows_by_frame.setdefault(int(r["frame"]), []).append(r)
        rows_by_event.setdefault(r["event_id"], []).append(r)
    inputs = {f.frame: f for f in frames}
    zoom_cache: dict[str, tuple[int, int, int, int]] = {}
    slots: list[str | None] = [None, None]
    container = av.open(str(out_path), mode="w")
    stream = container.add_stream("libx264", rate=int(round(fps)))
    stream.width, stream.height, stream.pix_fmt = CANVAS_W, CANVAS_H, "yuv420p"
    stream.options = {"crf": "21", "preset": "veryfast"}
    leg = ov.legend(CANVAS_W, LEGEND_H)
    written = 0
    try:
        for fr in range(window[0], window[1] + 1):
            img = read_frame(fr)
            if img is None:
                raise RuntimeError(f"frame {fr} could not be decoded")
            h, w = img.shape[:2]
            tracks = {tid: o.box for tid, o in inputs[fr].tracks.items()}
            frame_rows = rows_by_frame.get(fr, [])
            canvas = np.full((CANVAS_H, CANVAS_W, 3), 18, np.uint8)
            # header
            active = [by_id[r["event_id"]] for r in frame_rows if r["event_id"] in by_id]
            counts = {k: sum(ov.category_of(e) == k for e in active) for k in ov.CATEGORIES}
            _put(canvas, f"frame {fr}   t = {pts(fr):.3f} s   active: " +
                 "  ".join(f"{ov.CATEGORY_LABELS[k]} {v}" for k, v in counts.items() if v), (12, 24), 0.62)
            # overview
            top_w = min(CANVAS_W, int(TOP_MAX_H * w / h))
            top = ov.render_view(img, (0, 0, w, h), top_w, tracks, frame_rows, by_id, labels,
                                 opts=ov.ViewOptions(markers_only=True, show_track_ids=False, font_scale=0.5))
            x_off = (CANVAS_W - top_w) // 2
            canvas[HEADER_H:HEADER_H + top.shape[0], x_off:x_off + top_w] = top
            y = HEADER_H + top.shape[0]
            canvas[y:y + LEGEND_H] = leg
            y += LEGEND_H + 6
            # enlarged panels
            slots = assign_slots(slots, active)
            panel_h = int(PANEL_W * 9 / 16)
            for i, eid in enumerate(slots):
                px = 8 + i * (PANEL_W + 8)
                if eid is None:
                    cv2.rectangle(canvas, (px, y), (px + PANEL_W, y + panel_h), (40, 40, 40), 1)
                    _put(canvas, "no further active event", (px + 20, y + 40), 0.6, (120, 120, 120))
                    continue
                ev = by_id[eid]
                if eid not in zoom_cache:
                    zoom_cache[eid] = ov.event_zoom_rect(ev, rows_by_event.get(eid, []), w, h)
                trails = {}
                for t in {int(k) for r in frame_rows if r["event_id"] == eid for k in r["boxes"]}:
                    pts_ = [((inputs[f].tracks[t].box[0] + inputs[f].tracks[t].box[2]) / 2, inputs[f].tracks[t].box[3])
                            for f in range(max(window[0], fr - TRAIL_FRAMES), fr + 1) if f in inputs and t in inputs[f].tracks]
                    trails[t] = pts_
                panel = ov.render_view(img, zoom_cache[eid], PANEL_W, tracks, frame_rows, by_id, labels, selected=eid,
                                       opts=ov.ViewOptions(trails=trails))
                panel = panel[:panel_h]
                canvas[y:y + panel.shape[0], px:px + PANEL_W] = panel
                colour = ov.CATEGORY_BGR[ov.category_of(ev)]
                cv2.rectangle(canvas, (px, y), (px + PANEL_W, y + panel.shape[0]), colour, 2)
                text = caption(ev, labels[eid], fr)
                _put(canvas, text[:118], (px, y + panel_h + 21), 0.52, colour)
            for packet in stream.encode(av.VideoFrame.from_ndarray(canvas, format="bgr24")):
                container.mux(packet)
            written += 1
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
    return {"path": Path(out_path).name, "frames": written, "first_frame": window[0], "codec": "h264", "fps": fps,
            "layout": "focus", "size": [CANVAS_W, CANVAS_H]}


def render_event_strip(out_path: Path, event: Any, label: str, read_frame: Callable[[int], np.ndarray | None],
                       frames: Sequence[Any], rows: Sequence[dict[str, Any]], events: Sequence[Any], window: tuple[int, int],
                       margin: int = 12, tile_w: int = 520) -> str:
    """Before / start / peak / end / after tiles of one event with a fixed enlarged crop and a caption bar."""
    evs = [json.loads(e.to_json()) if hasattr(e, "to_json") else e for e in events]
    ev = json.loads(event.to_json()) if hasattr(event, "to_json") else event
    by_id = {e["event_id"]: e for e in evs}
    labels = {e["event_id"]: f"E{i}" for i, e in enumerate(evs, start=1)}
    inputs = {f.frame: f for f in frames}
    ev_rows = [r for r in rows if r["event_id"] == ev["event_id"]]
    rows_at: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        rows_at.setdefault(int(r["frame"]), []).append(r)
    lo, hi = min(inputs), max(inputs)
    picks = {"before": max(lo, ev["start_frame"] - margin), "start": ev["start_frame"], "peak": ev["peak_frame"],
             "end": ev["end_frame"], "after": min(hi, ev["end_frame"] + margin)}
    tiles = []
    for name, fr in picks.items():
        img = read_frame(fr)
        if img is None:
            raise RuntimeError(f"frame {fr} could not be decoded")
        h, w = img.shape[:2]
        rect = ov.event_zoom_rect(ev, ev_rows, w, h)
        tracks = {tid: o.box for tid, o in inputs[fr].tracks.items()} if fr in inputs else {}
        tile = ov.render_view(img, rect, tile_w, tracks, rows_at.get(fr, []), by_id, labels, selected=ev["event_id"])
        bar = np.full((28, tile_w, 3), 24, np.uint8)
        _put(bar, f"{name}  f{fr}", (8, 20), 0.6)
        tiles.append(np.vstack([bar, tile]))
    hmax = max(t.shape[0] for t in tiles)
    tiles = [np.vstack([t, np.zeros((hmax - t.shape[0], tile_w, 3), np.uint8)]) for t in tiles]
    body = np.hstack(tiles)
    head = np.full((36, body.shape[1], 3), 18, np.uint8)
    colour = ov.CATEGORY_BGR[ov.category_of(ev)]
    _put(head, caption(ev, label, ev["peak_frame"]).replace(f"  {ev['peak_frame'] - ev['start_frame'] + 1}/", "  peak ")[:160],
         (10, 25), 0.62, colour)
    out = np.vstack([head, body, ov.legend(body.shape[1])])
    cv2.imwrite(str(out_path), out)
    return Path(out_path).name
