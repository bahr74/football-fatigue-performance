"""M13 ByteTrack activation ablation (CPU): does a lower ``track_activation_threshold`` improve tracking?

Design (pre-declared before any run):

* identical detection cache, identical supervision ByteTrack (buffer 75 -> 62 frames at 25 fps, match 0.95,
  minimum_consecutive_frames 1); only the tracker-input threshold (0.4 = T0 input, 0.1 = T0L input) and
  ``track_activation_threshold`` change;
* **tune grid** (117093, 118575 only): input {0.4, 0.1} x activation {0.5, 0.4, 0.3, 0.2} = 8 cells;
* **selection rule**: highest mean tune HOTA; ties (< 0.002) broken by mean tune IDF1, then by fewer IDSW;
* **check** (117092, 118576): only the declared cells (T0, T0L, T0-A03, T0L-A03) and the tune-selected cell are
  scored. The selected cell's tune score is optimistic (max over 8 cells); only its check score is an estimate.

Tracking metrics come from ``detection.replay.score`` (TrackEval HOTA / CLEAR / Identity, IoU 0.5, all tracks, GT =
22 annotated players, so referees count as false positives for every cell alike). Track initialisations = first
frame of each temporary ID; continuations = tracked boxes whose ID was also output in the previous frame.

Output (write-once): ``<out_root>/_diag/activation-ablation-v1/{tune_grid.json, check.json, summary.json}``.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from football_fatigue.video.detection.replay import VARIANTS, score
from football_fatigue.video.sources import SOURCES, guard_not_reserved
from football_fatigue.video.tracking import run_sv_bytetrack

VERSION = "activation-ablation-v1"
TUNE = ("soccertrack-117093", "soccertrack-118575")
CHECK = ("soccertrack-117092", "soccertrack-118576")
INPUTS = {0.4: "T0", 0.1: "T0L"}
ACTIVATIONS = (0.5, 0.4, 0.3, 0.2)
DECLARED = (("T0", 0.5), ("T0L", 0.5), ("T0", 0.3), ("T0L", 0.3))
TIE = 0.002


def cell_name(inp: str, act: float) -> str:
    return inp if act == 0.5 else f"{inp}-A{int(round(act * 100)):02d}"


def _peak_rss_mb() -> float | None:
    try:
        import psutil

        mi = psutil.Process().memory_info()
        return round(getattr(mi, "peak_wset", mi.rss) / 2**20, 1)
    except Exception:  # pragma: no cover - psutil missing
        return None


def run_cell(data_root: Path, source_id: str, inp: str, act: float) -> dict[str, Any]:
    spec = SOURCES[source_id]
    guard_not_reserved(source_id, spec["match_id"], spec["cache_run"], spec["replay_run"], spec["gt"])
    t0 = time.perf_counter()
    tracks: dict[int, list] = {}
    inits = conts = boxes = inputs = 0
    seen: set[int] = set()
    prev: set[int] = set()
    for fr, chosen, out in run_sv_bytetrack(data_root, source_id, VARIANTS[inp], act):
        inputs += len(chosen)
        cur = set()
        rows = []
        for t in out:
            tid = int(t.track_id)
            rows.append((tid, (t.bbox.x1, t.bbox.y1, t.bbox.x2, t.bbox.y2), float(t.confidence), None))
            cur.add(tid)
            inits += tid not in seen
            conts += tid in prev
            seen.add(tid)
        boxes += len(rows)
        tracks[fr] = rows
        prev = cur
    secs = time.perf_counter() - t0
    m = score(data_root, source_id, tracks, on_pitch_only=False)
    frames = len(tracks)
    c = m["clear"]
    return {"source_id": source_id, "role": spec["role"], "cell": cell_name(inp, act), "input": inp,
            "input_threshold": VARIANTS[inp].threshold, "activation": act, "new_track_min_conf": round(act + 0.1, 2),
            "HOTA": m["hota"]["HOTA"], "DetA": m["hota"]["DetA"], "AssA": m["hota"]["AssA"], "IDF1": m["identity"]["IDF1"],
            "MOTA": c["MOTA"], "IDSW": c["IDSW"], "Frag": c["Frag"], "TP": c["CLR_TP"], "FP": c["CLR_FP"], "FN": c["CLR_FN"],
            "gt_coverage": round(c["CLR_TP"] / max(1, c["CLR_TP"] + c["CLR_FN"]), 5),
            "missing_gt_players_per_frame": round(c["CLR_FN"] / max(1, frames), 3), "MT": c["MT"], "ML": c["ML"],
            "track_initialisations": inits, "track_continuations": conts, "tracked_boxes": boxes,
            "tracker_input_boxes": inputs, "temporary_ids": len(seen), "frames": frames,
            "tracker_seconds": round(secs, 1), "process_peak_rss_mb": _peak_rss_mb()}


def _mean(rows: list[dict], key: str) -> float:
    return round(float(np.mean([r[key] for r in rows])), 4)


def _agg(rows: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {k: _mean(rows, k) for k in ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "gt_coverage",
                                                      "missing_gt_players_per_frame")}
    out.update({k: int(sum(r[k] for r in rows)) for k in ("IDSW", "Frag", "FP", "FN", "track_initialisations",
                                                          "track_continuations", "temporary_ids")})
    return out


def select_cell(tune_rows: list[dict]) -> str:
    """The pre-declared rule: max mean tune HOTA; within TIE, max IDF1, then fewest IDSW."""
    by: dict[str, list] = {}
    for r in tune_rows:
        by.setdefault(r["cell"], []).append(r)
    agg = {k: _agg(v) for k, v in by.items()}
    best = max(a["HOTA"] for a in agg.values())
    near = [k for k, a in agg.items() if best - a["HOTA"] < TIE]
    return sorted(near, key=lambda k: (-agg[k]["IDF1"], agg[k]["IDSW"], k))[0]


def run(data_root: Path, out_root: Path) -> Path:
    dest = Path(out_root) / "_diag" / VERSION
    if (dest / "summary.json").exists():
        raise SystemExit(f"output already exists (write-once): {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    tune = []
    for sid in TUNE:
        for thr, inp in INPUTS.items():
            for act in ACTIVATIONS:
                r = run_cell(data_root, sid, inp, act)
                print(json.dumps({k: r[k] for k in ("source_id", "cell", "HOTA", "IDF1", "IDSW", "Frag", "tracker_seconds")}),
                      flush=True)
                tune.append(r)
    (dest / "tune_grid.json").write_text(json.dumps(tune, indent=1), encoding="utf-8")
    selected = select_cell(tune)
    cells = list(dict.fromkeys([cell_name(i, a) for i, a in DECLARED] + [selected]))
    lookup = {cell_name(i, a): (i, a) for i in INPUTS.values() for a in ACTIVATIONS}
    check = []
    for sid in CHECK:
        for c in cells:
            r = run_cell(data_root, sid, *lookup[c])
            print(json.dumps({k: r[k] for k in ("source_id", "cell", "HOTA", "IDF1", "IDSW", "Frag")}), flush=True)
            check.append(r)
    (dest / "check.json").write_text(json.dumps(check, indent=1), encoding="utf-8")
    by_t: dict[str, list] = {}
    for r in tune:
        by_t.setdefault(r["cell"], []).append(r)
    by_c: dict[str, list] = {}
    for r in check:
        by_c.setdefault(r["cell"], []).append(r)
    summary = {"version": VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "design": {"tune": TUNE, "check": CHECK, "inputs": {str(k): v for k, v in INPUTS.items()},
                          "activations": ACTIVATIONS, "declared_cells": [cell_name(i, a) for i, a in DECLARED],
                          "selection_rule": f"max mean tune HOTA; within {TIE}: max IDF1, then fewest IDSW",
                          "unchanged": "detections, supervision ByteTrack 0.30.9, lost_track_buffer 75 (62 frames at 25 fps), "
                                       "minimum_matching_threshold 0.95, minimum_consecutive_frames 1, frame rate 25"},
               "selected_on_tune": selected, "tune": {k: _agg(v) for k, v in by_t.items()},
               "check": {k: _agg(v) for k, v in by_c.items()},
               "selection_bias_note": "the selected cell's tune score is the max over 8 cells and is optimistic; "
                                      "judge it on check only, against the declared cells"}
    (dest / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return dest
