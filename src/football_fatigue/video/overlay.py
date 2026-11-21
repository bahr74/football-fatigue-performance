"""Readable overlays for occlusion events, shared by the focus MP4 renderer and the dashboard.

Everything here works on plain dicts (events as written to ``events.jsonl``, rows as written to ``frames.jsonl``) and
pixel boxes in the source image, so the same drawing is used for saved runs and, later, uploaded-video runs.
Labels are drawn *after* scaling a region up, so text has a fixed on-screen size whatever the zoom.

Track labels are ``T<n>``: temporary tracker IDs, which are **not** persistent player identities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

Box = Sequence[float]

# Display categories (timeline rows, colours). An uncertain event is shown in the "uncertain" category; its class and
# subtype stay visible in the inspector.
CATEGORIES = ("box_overlap", "probable_occlusion", "detector_miss", "tracker_gap", "uncertain")
CATEGORY_LABELS = {"box_overlap": "Box overlap", "probable_occlusion": "Probable occlusion", "detector_miss": "Detector miss",
                   "tracker_gap": "Tracker gap", "uncertain": "Uncertain"}
CATEGORY_HEX = {"box_overlap": "#E6B422", "probable_occlusion": "#F06A1D", "detector_miss": "#D63A3A",
                "tracker_gap": "#8B5CF6", "uncertain": "#8C939D"}
GREY = (150, 150, 150)
DIM = (95, 95, 95)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GT_COLOUR = (235, 235, 235)
TRACK_ID_NOTE = "T## = temporary tracker ID, not a player identity"


def hex_to_bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


CATEGORY_BGR = {k: hex_to_bgr(v) for k, v in CATEGORY_HEX.items()}


def category_of(event: Mapping[str, Any], uncertain_row: bool = True) -> str:
    """Display category. With uncertain_row (default) every uncertain event is shown as "uncertain"; otherwise
    it stays in its class row. Detection losses are split into detector miss / tracker gap; the other loss subtypes
    (partly covered by a neighbour, left the image) are always "uncertain"."""
    if uncertain_row and event.get("uncertain"):
        return "uncertain"
    cls = event["event_class"]
    if cls == "detection_loss":
        return event["subtype"] if event["subtype"] in ("detector_miss", "tracker_gap") else "uncertain"
    return cls


# ---------------------------------------------------------------------------------------------------- geometry
def clamp_rect(x0: float, y0: float, w: float, h: float, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    """Shift (not shrink) a w x h rectangle into the frame; shrink only if it is larger than the frame."""
    w, h = min(w, frame_w), min(h, frame_h)
    x0 = min(max(0.0, x0), frame_w - w)
    y0 = min(max(0.0, y0), frame_h - h)
    return int(round(x0)), int(round(y0)), int(round(x0 + w)), int(round(y0 + h))


def zoom_rect(boxes: Sequence[Box], frame_w: int, frame_h: int, aspect: float = 16 / 9, pad: float = 0.8,
              min_h: float = 180.0) -> tuple[int, int, int, int]:
    """A crop with the given aspect ratio that contains all boxes plus ``pad`` of their extent on every side."""
    if not boxes:
        return 0, 0, frame_w, frame_h
    x1 = min(b[0] for b in boxes); y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes); y2 = max(b[3] for b in boxes)
    bw, bh = x2 - x1, y2 - y1
    ref = max(bh, bw / aspect, 1.0)
    h = max(min_h, bh + 2 * pad * ref, (bw + 2 * pad * ref) / aspect)
    w = h * aspect
    if w > frame_w:
        w = frame_w; h = w / aspect
    if h > frame_h:
        h = frame_h; w = min(frame_w, h * aspect)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return clamp_rect(cx - w / 2.0, cy - h / 2.0, w, h, frame_w, frame_h)


def region_rect(cx: float, cy: float, zoom: float, frame_w: int, frame_h: int, aspect: float = 16 / 9) -> tuple[int, int, int, int]:
    """Manual zoom: a crop centred at (cx, cy) whose height is frame_h / zoom (zoom 1 = full height)."""
    zoom = max(1.0, float(zoom))
    h = frame_h / zoom
    w = min(frame_w, h * aspect)
    return clamp_rect(cx - w / 2.0, cy - h / 2.0, w, h, frame_w, frame_h)


# ---------------------------------------------------------------------------------------------------- drawing
@dataclass
class ViewOptions:
    show_tracks: bool = True          # other tracked players (thin, grey)
    show_track_ids: bool = True       # T## on other tracks (only when zoomed enough to read)
    show_events: bool = True
    dim_others: bool = True           # with a selected event, other events are drawn grey
    show_predicted: bool = True
    trails: Mapping[int, Sequence[tuple[float, float]]] = field(default_factory=dict)   # tid -> foot points
    gt_boxes: Sequence[Box] = ()      # optional ground-truth boxes (development matches only)
    markers_only: bool = False        # overview: one outline + E-label per event, no per-player labels
    font_scale: float = 0.55


def _dashed_rect(img, p1, p2, colour, thickness=2, dash=7):
    (x1, y1), (x2, y2) = p1, p2
    for a, b in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        length = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])))
        for s in range(0, max(length, 1), dash * 2):
            t0, t1 = s / max(length, 1), min(s + dash, length) / max(length, 1)
            cv2.line(img, (int(a[0] + (b[0] - a[0]) * t0), int(a[1] + (b[1] - a[1]) * t0)),
                     (int(a[0] + (b[0] - a[0]) * t1), int(a[1] + (b[1] - a[1]) * t1)), colour, thickness, cv2.LINE_AA)


def _overlaps(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def pill(img, text: str, box: tuple[tuple[int, int], tuple[int, int]], colour, font_scale: float = 0.55,
         placed: list | None = None) -> tuple[int, int, int, int]:
    """A filled label with dark/light text on the category colour, kept inside the image. Candidate positions:
    above the box (left, right aligned), below it, then stacked further up; the first one that does not overlap an
    already placed label (placed) is used."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    pad = 3
    lw, lh = tw + 2 * pad, th + base + 2 * pad
    (bx1, by1), (bx2, by2) = box
    cands = [(bx1, by1 - lh - 2), (bx2 - lw, by1 - lh - 2), (bx1, by2 + 2), (bx2 - lw, by2 + 2)]
    cands += [(bx1, by1 - (k + 2) * (lh + 2)) for k in range(4)]
    H, W = img.shape[:2]
    chosen = None
    for cx, cy in cands:
        x = int(min(max(0, cx), W - lw)); y = int(min(max(0, cy), H - lh))
        r = (x, y, x + lw, y + lh)
        if not placed or not any(_overlaps(r, q) for q in placed):
            chosen = r
            break
    if chosen is None:
        x = int(min(max(0, cands[0][0]), W - lw)); y = int(min(max(0, cands[0][1]), H - lh))
        chosen = (x, y, x + lw, y + lh)
    if placed is not None:
        placed.append(chosen)
    x, y = chosen[0], chosen[1]
    cv2.rectangle(img, (x, y), (x + lw, y + lh), colour, -1)
    luminance = 0.114 * colour[0] + 0.587 * colour[1] + 0.299 * colour[2]
    cv2.putText(img, text, (x + pad, y + th + pad), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                BLACK if luminance > 140 else WHITE, 1, cv2.LINE_AA)
    return chosen


def render_view(img: np.ndarray, rect: tuple[int, int, int, int], out_w: int, tracks: Mapping[int, Box],
                rows: Sequence[Mapping[str, Any]], events_by_id: Mapping[str, Mapping[str, Any]],
                labels: Mapping[str, str], selected: str | None = None, opts: ViewOptions | None = None) -> np.ndarray:
    """Crop ``rect`` from ``img``, scale it to width ``out_w`` and draw overlays at screen resolution."""
    opts = opts or ViewOptions()
    x0, y0, x1, y1 = rect
    crop = img[y0:y1, x0:x1]
    s = out_w / max(1, (x1 - x0))
    out_h = max(1, int(round((y1 - y0) * s)))
    view = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)

    def tr(b: Box) -> tuple[tuple[int, int], tuple[int, int]]:
        return (int((b[0] - x0) * s), int((b[1] - y0) * s)), (int((b[2] - x0) * s), int((b[3] - y0) * s))

    event_tracks: set[int] = set()
    for r in rows:
        event_tracks |= {int(t) for t in r["boxes"]}

    for gb in opts.gt_boxes:
        p1, p2 = tr(gb)
        _dashed_rect(view, p1, p2, GT_COLOUR, 1, 3)

    readable = s >= 1.2
    if opts.show_tracks:
        for tid, b in tracks.items():
            if tid in event_tracks and opts.show_events:
                continue
            p1, p2 = tr(b)
            cv2.rectangle(view, p1, p2, DIM if selected else GREY, 1, cv2.LINE_AA)
            if opts.show_track_ids and readable and not opts.markers_only:
                cv2.putText(view, f"T{tid}", (p1[0], max(10, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                            opts.font_scale * 0.8, GREY, 1, cv2.LINE_AA)

    for tid, pts in opts.trails.items():
        if len(pts) >= 2:
            poly = np.array([[(px - x0) * s, (py - y0) * s] for px, py in pts], dtype=np.int32)
            colour = CATEGORY_BGR["probable_occlusion"] if tid in event_tracks else GREY
            cv2.polylines(view, [poly], False, colour, 2 if tid in event_tracks else 1, cv2.LINE_AA)

    if not opts.show_events:
        return view
    # selected event last, so it is on top
    ordered = sorted(rows, key=lambda r: (r["event_id"] != selected, r["event_id"]))   # selected labels placed first
    placed: list = []
    for r in ordered:
        ev = events_by_id.get(r["event_id"])
        if ev is None:
            continue
        is_sel = r["event_id"] == selected
        colour = CATEGORY_BGR[category_of(ev)]
        if selected and not is_sel and opts.dim_others:
            colour = DIM
        thick = 3 if is_sel or not selected else 1
        boxes = [(int(t), info) for t, info in r["boxes"].items()]
        if opts.markers_only:
            ub = (min(i["box"][0] for _t, i in boxes), min(i["box"][1] for _t, i in boxes),
                  max(i["box"][2] for _t, i in boxes), max(i["box"][3] for _t, i in boxes))
            p1, p2 = tr(ub)
            m = 4
            cv2.rectangle(view, (p1[0] - m, p1[1] - m), (p2[0] + m, p2[1] + m), colour, thick, cv2.LINE_AA)
            if not selected or is_sel:
                pill(view, labels.get(r["event_id"], "E?"), ((p1[0] - m, p1[1] - m), (p2[0] + m, p2[1] + m)), colour,
                     opts.font_scale * 0.8, placed)
            continue
        for tid, info in boxes:
            p1, p2 = tr(info["box"])
            predicted = info.get("state") != "visible"
            if predicted:
                if not opts.show_predicted:
                    continue
                _dashed_rect(view, p1, p2, colour, max(1, thick - 1))
            else:
                cv2.rectangle(view, p1, p2, colour, thick, cv2.LINE_AA)
            if not selected or is_sel:
                text = f"T{tid}" + (" predicted" if predicted else "")
                pill(view, text, (p1, p2), colour, opts.font_scale, placed)
    return view


def legend(width: int, height: int = 34, font_scale: float = 0.5) -> np.ndarray:
    """A compact one-line legend strip."""
    bar = np.full((height, width, 3), 24, np.uint8)
    x = 10
    for k in CATEGORIES:
        cv2.rectangle(bar, (x, 10), (x + 16, height - 10), CATEGORY_BGR[k], -1)
        cv2.putText(bar, CATEGORY_LABELS[k], (x + 22, height - 12), cv2.FONT_HERSHEY_SIMPLEX, font_scale, WHITE, 1, cv2.LINE_AA)
        x += 32 + cv2.getTextSize(CATEGORY_LABELS[k], cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0]
    _dashed_rect(bar, (x, 9), (x + 16, height - 9), WHITE, 1, 3)
    cv2.putText(bar, "predicted (not detected)", (x + 22, height - 12), cv2.FONT_HERSHEY_SIMPLEX, font_scale, WHITE, 1, cv2.LINE_AA)
    x += 40 + cv2.getTextSize("predicted (not detected)", cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0]
    cv2.putText(bar, TRACK_ID_NOTE, (x, height - 12), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), 1, cv2.LINE_AA)
    return bar


def event_zoom_rect(event: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], frame_w: int, frame_h: int,
                    aspect: float = 16 / 9) -> tuple[int, int, int, int]:
    """A fixed crop for the whole event (no camera jitter): all member boxes over its frames."""
    boxes = [info["box"] for r in rows for info in r["boxes"].values()]
    if not boxes:
        x, y = event.get("pixel_xy", [frame_w / 2, frame_h / 2])
        return region_rect(x, y, 4.0, frame_w, frame_h, aspect)
    return zoom_rect(boxes, frame_w, frame_h, aspect)


def describe(event: Mapping[str, Any]) -> str:
    """One-line human description of an event."""
    mem = event["members"]
    ids = ", ".join(f"T{m['track_id']}" for m in mem)
    cls, sub = event["event_class"], event["subtype"]
    if cls == "detection_loss":
        what = {"detector_miss": "no detection at the predicted position", "tracker_gap": "a detection existed but the tracker did not output it",
                "left_frame": "left the image", "near_neighbour_uncertain": "partly covered by a neighbour"}.get(sub, sub)
        return f"T{mem[0]['track_id']} lost for {event['end_frame'] - event['start_frame'] + 1} frames: {what}"
    hidden = [m for m in mem if m.get("hidden")]
    if cls == "probable_occlusion" and hidden:
        return f"{', '.join('T%d' % m['track_id'] for m in hidden)} hidden behind a neighbour ({sub}); members {ids}"
    if cls == "probable_occlusion":
        return f"far player degraded while overlapped ({sub}); members {ids}"
    return f"boxes overlap, both players stay detected; members {ids}"
