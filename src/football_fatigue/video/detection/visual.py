"""Visual stage comparison: the same zoomed region shown at every pipeline stage.

Panels (left to right): ground truth | raw per-tile output (only when a capture exists) | post-merge cache, coloured by
what happened next | tracker output | occlusion-engine predicted boxes. A header names the stage that lost the player.
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from football_fatigue.video.detection import DIAG_VERSION
from football_fatigue.video.detection.diagnose import (
    find_engine_run, iter_cache, iter_replay, load_predicted, replay_path,
)
from football_fatigue.video.detection.stages import LOST_STAGES, STAGE_TEXT, Det
from football_fatigue.video.sources import SOURCES

# BGR
WHITE, YELLOW, GREEN, ORANGE, MAGENTA, CYAN, RED, GREY = ((255, 255, 255), (0, 230, 255), (80, 220, 80),
                                                          (0, 150, 255), (255, 0, 220), (255, 220, 0), (60, 60, 255),
                                                          (150, 150, 150))
DET_LEGEND = (("tracker input (conf > 0.4)", GREEN), ("below threshold (0.1-0.4)", ORANGE),
              ("turf filter rejected", MAGENTA))


def det_colour(d: Det, threshold: float = 0.4) -> tuple[int, int, int]:
    if d.turf is False:
        return MAGENTA
    return GREEN if d.conf > threshold else ORANGE


def crop_rect(box, size: tuple[int, int], side_heights: float = 4.0, min_side: int = 160) -> tuple[int, int, int, int]:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    side = max(min_side, int(side_heights * (box[3] - box[1])))
    w, h = size
    x1 = int(min(max(0, cx - side / 2), max(0, w - side)))
    y1 = int(min(max(0, cy - side / 2), max(0, h - side)))
    return x1, y1, min(w, x1 + side), min(h, y1 + side)


def _dashed(img, p1, p2, colour, thick=1, dash=6):
    import cv2

    x1, y1 = p1
    x2, y2 = p2
    for (a, b) in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        n = max(1, int(np.hypot(b[0] - a[0], b[1] - a[1]) // dash))
        for k in range(0, n, 2):
            s = (int(a[0] + (b[0] - a[0]) * k / n), int(a[1] + (b[1] - a[1]) * k / n))
            e = (int(a[0] + (b[0] - a[0]) * min(n, k + 1) / n), int(a[1] + (b[1] - a[1]) * min(n, k + 1) / n))
            cv2.line(img, s, e, colour, thick, cv2.LINE_AA)


def draw_layer(crop: np.ndarray, rect, scale: float, items: list[dict[str, Any]]) -> np.ndarray:
    """Draw boxes (``box``, ``colour``, optional ``label``, ``dashed``, ``thick``) onto an upscaled crop."""
    import cv2

    out = crop.copy()
    for it in items:
        b = it["box"]
        p1 = (int((b[0] - rect[0]) * scale), int((b[1] - rect[1]) * scale))
        p2 = (int((b[2] - rect[0]) * scale), int((b[3] - rect[1]) * scale))
        if it.get("dashed"):
            _dashed(out, p1, p2, it["colour"], it.get("thick", 2))
        else:
            cv2.rectangle(out, p1, p2, it["colour"], it.get("thick", 2), cv2.LINE_AA)
        if it.get("label"):
            txt = str(it["label"])
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            y = max(th + 3, p1[1] - 3)
            cv2.rectangle(out, (p1[0], y - th - 3), (p1[0] + tw + 4, y + 2), (20, 20, 20), -1)
            cv2.putText(out, txt, (p1[0] + 2, y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, it["colour"], 1, cv2.LINE_AA)
    return out


def stage_layers(gt: list[tuple[int, tuple]], focus_gid: int | None, dets: list[Det], tracks: dict[int, tuple],
                 predicted: list[tuple], raw: list[tuple] | None) -> dict[str, list[dict[str, Any]]]:
    """Box lists per panel; shared by the PNG panels and the dashboard."""
    layers = {
        "Ground truth": [{"box": b, "colour": YELLOW if g == focus_gid else WHITE, "label": f"GT{g}",
                          "thick": 3 if g == focus_gid else 1} for g, b in gt],
        "Post-merge (cache)": [{"box": d.box, "colour": det_colour(d), "label": f"{d.conf:.2f}"}
                               for d in dets if d.person],
        "Tracker output": [{"box": b, "colour": CYAN, "label": f"T{t}"} for t, b in sorted(tracks.items())],
        "Engine predicted": [{"box": b, "colour": RED, "label": f"T{t} {state.split('_')[0]}", "dashed": True}
                             for _e, t, state, b in predicted],
    }
    if raw is not None:
        layers = {"Ground truth": layers["Ground truth"],
                  "Raw per-tile": [{"box": b, "colour": GREY if c <= 0.4 else GREEN, "label": f"{c:.2f} t{tile}"}
                                   for c, b, tile in raw],
                  **{k: v for k, v in layers.items() if k != "Ground truth"}}
    return layers


def compose(img: np.ndarray, rect, layers: dict[str, list[dict]], header: list[str], panel: int = 320,
            raw_missing_note: bool = True) -> np.ndarray:
    import cv2

    crop = img[rect[1]:rect[3], rect[0]:rect[2]]
    scale = panel / max(1, crop.shape[1])
    crop = cv2.resize(crop, (panel, int(crop.shape[0] * scale)), interpolation=cv2.INTER_CUBIC)
    tiles = []
    names = list(layers)
    if raw_missing_note and "Raw per-tile" not in layers:
        names.insert(1, "Raw per-tile")
    for name in names:
        if name == "Raw per-tile" and name not in layers:
            t = np.full_like(crop, 40)
            for k, line in enumerate(("raw per-tile boxes", "were not cached", "(needs GPU capture)")):
                cv2.putText(t, line, (12, 40 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        else:
            t = draw_layer(crop, rect, scale, layers[name])
        bar = np.full((26, t.shape[1], 3), 25, np.uint8)
        cv2.putText(bar, name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, t]))
    row = np.hstack([np.pad(t, ((0, 0), (0, 4), (0, 0)), constant_values=0) for t in tiles])
    head = np.full((22 * len(header) + 10, row.shape[1], 3), 18, np.uint8)
    for k, line in enumerate(header):
        cv2.putText(head, line, (8, 20 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    legend = np.full((24, row.shape[1], 3), 18, np.uint8)
    x = 8
    for text, col in (("GT focus", YELLOW), *DET_LEGEND, ("tracker", CYAN), ("engine prediction (dashed)", RED)):
        cv2.rectangle(legend, (x, 7), (x + 12, 19), col, -1)
        cv2.putText(legend, text, (x + 16, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        x += 22 + cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0] + 12
    return np.vstack([head, row, legend])


def select_examples(out_root: Path, source_id: str, per_stage: int = 3) -> list[dict[str, Any]]:
    """Deterministic examples: per lost stage, the longest runs of consecutive frames in which one visible (for
    merge_suspect: any visibility) ground-truth player stays in that stage; the middle frame of each run."""
    import pandas as pd

    d = pd.read_csv(Path(out_root) / source_id / "detection" / DIAG_VERSION / "gt_rows.csv.gz")
    out = []
    for stage in LOST_STAGES:
        s = d[(d.stage == stage) & ((d.visibility == "visible") | (stage == "merge_suspect"))].sort_values(["gid", "frame"])
        if s.empty:
            continue
        run = (s.frame.diff().ne(1) | s.gid.diff().ne(0)).cumsum()
        g = s.groupby(run).agg(gid=("gid", "first"), start=("frame", "min"), end=("frame", "max"), n=("frame", "size"))
        g = g.sort_values(["n", "start"], ascending=[False, True]).head(per_stage)
        for r in g.itertuples():
            out.append({"source_id": source_id, "stage": stage, "gid": int(r.gid), "start": int(r.start),
                        "end": int(r.end), "frames": int(r.n), "frame": int((r.start + r.end) // 2)})
    return out


def render_examples(data_root: Path, out_root: Path, source_id: str, per_stage: int = 3) -> list[Path]:
    import cv2
    from football_fatigue.vision.mot_eval import load_reference

    from football_fatigue.video.sources import ExactFrameReader

    spec = SOURCES[source_id]
    ex = select_examples(out_root, source_id, per_stage)
    wanted = {e["frame"] for e in ex}
    dets_at: dict[int, list[Det]] = {}
    for fr, dets in iter_cache(data_root, source_id):
        if fr in wanted:
            dets_at[fr] = dets
        if fr > max(wanted, default=-1):
            break
    tracks_at = {fr: t for fr, t in iter_replay(replay_path(data_root, source_id)) if fr in wanted}
    predicted, _events = load_predicted(find_engine_run(out_root, source_id))
    gt, _ = load_reference(Path(data_root) / spec["gt"], 0)
    import pandas as pd

    rows = pd.read_csv(Path(out_root) / source_id / "detection" / DIAG_VERSION / "gt_rows.csv.gz")
    rows = rows[rows.frame.isin(wanted)].set_index(["frame", "gid"])
    dest = Path(out_root) / source_id / "detection" / DIAG_VERSION / "examples"
    dest.mkdir(exist_ok=True)
    reader = ExactFrameReader(Path(data_root) / spec["video"])
    written = []
    index = []
    try:
        for e in sorted(ex, key=lambda e: e["frame"]):
            fr = e["frame"]
            img = reader.frame(fr)
            h, w = img.shape[:2]
            box = dict(gt.get(fr, []))[e["gid"]]
            r = rows.loc[(fr, e["gid"])]
            rect = crop_rect(box, (w, h))
            layers = stage_layers(gt.get(fr, []), e["gid"], dets_at.get(fr, []), tracks_at.get(fr, {}),
                                  predicted.get(fr, []), None)
            conf = "" if pd.isna(r.det_conf) else f", matched box conf {r.det_conf:.2f}"
            header = [f"{source_id}  frame {fr}  GT player {e['gid']}  ({r.visibility}, {int(r.cover * 100)}% covered by "
                      f"nearer GT players, height {r.height:.0f} px)",
                      f"Lost at: {e['stage']} - {STAGE_TEXT[e['stage']]}{conf}",
                      f"This player stays in this stage for {e['frames']} consecutive frames ({e['start']}-{e['end']})"]
            panel = compose(img, rect, layers, header)
            p = dest / f"{e['stage']}-f{fr:05d}-gt{e['gid']}.png"
            cv2.imwrite(str(p), panel)
            written.append(p)
            index.append(e | {"png": p.name, "visibility": r.visibility, "cover": float(r.cover),
                              "det_conf": None if pd.isna(r.det_conf) else float(r.det_conf)})
    finally:
        reader.close()
    (dest / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    return written
