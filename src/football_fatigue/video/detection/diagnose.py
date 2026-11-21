"""Batch detection diagnosis of one development source from existing artefacts (CPU only, no inference).

Reads (never writes) the detection cache, the production tracker replay and SoccerTrack ground truth, plus the
Session B v2 occlusion run of the same source. Writes ``<out_root>/<source>/detection/diag-v1/``:

* ``summary.json``     stage x visibility x size counts, tracker-input precision split, confidence-threshold sweep,
                       AP@0.5, engine cross-reference, provenance;
* ``gt_rows.csv.gz``   one row per ground-truth player per frame (stage, visibility, size, matched box);
* ``engine_xref.csv``  every predicted box of the occlusion engine matched to ground truth and to its stage.
"""

from __future__ import annotations

import csv
import gzip
import json
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from football_fatigue.video.detection import DIAG_VERSION
from football_fatigue.video.detection.stages import (
    Det, Params, attribute_frame, average_precision, greedy_scored, iou_matrix, outside_distance, threshold_counts,
)
from football_fatigue.video.sources import SOURCES, guard_not_reserved, play_area_polygon

THRESHOLDS = (0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5)
GT_COLUMNS = ("frame", "gid", "stage", "visibility", "size", "height", "cover", "track_id", "det_conf", "det_iou",
              "x1", "y1", "x2", "y2")


def cache_dir(data_root: Path, source_id: str) -> Path:
    return Path(data_root) / "processed" / "vision" / SOURCES[source_id]["cache_run"] / "detection_cache"


def replay_path(data_root: Path, source_id: str) -> Path:
    return Path(data_root) / "processed" / "vision" / SOURCES[source_id]["replay_run"] / "frames.jsonl"


def iter_cache(data_root: Path, source_id: str, start: int = 0) -> Iterator[tuple[int, list[Det]]]:
    from football_fatigue.vision.detection_cache import DetectionCacheReader

    reader = DetectionCacheReader(cache_dir(data_root, source_id))
    for fr, _ts, obs in reader.iter_records(start_frame=start):
        yield fr, [Det(tuple(float(v) for v in o.xyxy), float(o.confidence), str(o.class_label),
                       o.mapped_class.value if o.mapped_class is not None else None, o.passes_turf_filter)
                   for o in obs]


def iter_replay(path: Path) -> Iterator[tuple[int, dict[int, tuple[float, ...]]]]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            yield int(rec["frame_num"]), {int(p["track_id"]): tuple(float(v) for v in p["bbox_xyxy_px"])
                                          for p in rec["players"]}


def find_engine_run(out_root: Path, source_id: str, profile: str = "v2") -> Path | None:
    base = Path(out_root) / source_id / "occlusion" / "0.2.0"
    runs = sorted(p for p in base.glob(f"occl-{source_id}-0-*-{profile}-*") if (p / "manifest.json").exists())
    return runs[-1] if runs else None


def load_predicted(run: Path | None) -> tuple[dict[int, list[tuple[str, int, str, tuple[float, ...]]]], dict[str, dict]]:
    """Engine predicted boxes per frame: (event_id, track_id, state, box), and the events by id."""
    by_frame: dict[int, list] = defaultdict(list)
    events: dict[str, dict] = {}
    if run is None:
        return by_frame, events
    with (run / "events.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            e = json.loads(line)
            events[e["event_id"]] = e
    with (run / "frames.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            for tid, b in r["boxes"].items():
                if b["state"] != "visible":
                    by_frame[int(r["frame"])].append((r["event_id"], int(tid), b["state"], tuple(b["box"])))
    return by_frame, events


def frame_size(data_root: Path, source_id: str) -> tuple[int, int]:
    m = json.loads((cache_dir(data_root, source_id) / "cache_manifest.json").read_text(encoding="utf-8"))
    return int(m["video"]["width"]), int(m["video"]["height"])


def diagnose(data_root: Path, out_root: Path, source_id: str, p: Params = Params(), max_frames: int | None = None,
             overwrite: bool = False, engine_root: Path | None = None) -> Path:
    from football_fatigue.vision.mot_eval import load_reference

    spec = SOURCES[source_id]
    guard_not_reserved(source_id, spec["match_id"], spec["cache_run"], spec["replay_run"], spec["gt"])
    out = Path(out_root) / source_id / "detection" / DIAG_VERSION
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"{out} exists (diagnosis folders are write-once; pass overwrite to replace)")
        shutil.rmtree(out)
    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    t0 = time.perf_counter()
    gt, _ = load_reference(Path(data_root) / spec["gt"], 0)
    poly = np.asarray(play_area_polygon(Path(data_root) / spec["keypoints"]), dtype=np.float32).reshape(-1, 1, 2)
    size = frame_size(data_root, source_id)
    run = find_engine_run(engine_root or out_root, source_id)
    predicted, events = load_predicted(run)

    stage_vis = Counter()          # (stage, visibility)
    stage_vis_size = Counter()     # (stage, visibility, size)
    det_out = Counter()            # outcome of every tracker-input detection
    det_conf_hist = Counter()      # tracker-input TP confidence deciles
    sweep: dict[float, Counter] = {t: Counter() for t in THRESHOLDS}
    scored_all: list[tuple[float, bool]] = []
    scored_onpitch: list[tuple[float, bool]] = []
    n_gt_all = n_gt_onpitch = 0
    xref_rows = []
    frames = 0

    gz = gzip.open(tmp / "gt_rows.csv.gz", "wt", newline="", encoding="utf-8")
    writer = csv.writer(gz)
    writer.writerow(GT_COLUMNS)
    replay = iter_replay(replay_path(data_root, source_id))
    rframe, rtracks = next(replay, (None, {}))
    for fr, dets in iter_cache(data_root, source_id):
        if max_frames is not None and frames >= max_frames:
            break
        while rframe is not None and rframe < fr:
            rframe, rtracks = next(replay, (None, {}))
        tracks = rtracks if rframe == fr else {}
        g = gt.get(fr, [])
        res = attribute_frame(g, dets, tracks, poly, size, p)
        frames += 1
        for r in res["gt"]:
            stage_vis[(r["stage"], r["visibility"])] += 1
            stage_vis_size[(r["stage"], r["visibility"], r["size"])] += 1
            b = r["box"]
            writer.writerow((fr, r["gid"], r["stage"], r["visibility"], r["size"], r["height"], r["cover"],
                             r["track_id"] if r["track_id"] is not None else "", r["det_conf"] if r["det_conf"] is not None else "",
                             r["det_iou"] if r["det_iou"] is not None else "", *(round(v, 1) for v in b)))
        for d in res["dets"]:
            det_out[d["outcome"]] += 1
            if d["outcome"] == "tp":
                det_conf_hist[min(9, int(d["conf"] * 10))] += 1
        for t, c in threshold_counts(g, dets, poly, THRESHOLDS, p).items():
            sweep[t].update(c)
        # AP over every cached person box that passed the turf filter (cache floor 0.1)
        persons = [d for d in dets if d.person and d.turf is not False]
        scored_all += greedy_scored(g, persons, p.match_iou)
        n_gt_all += len(g)
        onp = [d for d in persons if d.mapped != "referee"
               and not ((dd := outside_distance(d.box, poly)) is not None and dd < -p.off_pitch_heights * (d.box[3] - d.box[1]))]
        g_on = [x for x, r in zip(g, res["gt"]) if r["visibility"] != "off_pitch"]
        scored_onpitch += greedy_scored(g_on, onp, p.match_iou)
        n_gt_onpitch += len(g_on)
        # engine cross-reference: what was really at each predicted box?
        if fr in predicted:
            gboxes = [b for _gid, b in g]
            pb = [x[3] for x in predicted[fr]]
            m = iou_matrix(pb, gboxes)
            for k, (eid, tid, state, box) in enumerate(predicted[fr]):
                e = events.get(eid, {})
                row = {"frame": fr, "event_id": eid, "event_class": e.get("event_class"), "subtype": e.get("subtype"),
                       "track_id": tid, "state": state, "gid": "", "gt_iou": 0.0, "stage": "no_gt_player",
                       "visibility": "", "gt_track_id": ""}
                if gboxes:
                    j = int(np.argmax(m[k]))
                    if m[k, j] >= 0.3:
                        r = res["gt"][j]
                        stage = r["stage"]
                        if stage == "tracked" and r["track_id"] != tid:
                            stage = "tracked_other_id"
                        elif stage == "tracked":
                            stage = "tracked_same_id"
                        row.update(gid=r["gid"], gt_iou=round(float(m[k, j]), 3), stage=stage,
                                   visibility=r["visibility"], gt_track_id=r["track_id"] or "")
                xref_rows.append(row)
    gz.close()

    with (tmp / "engine_xref.csv").open("w", newline="", encoding="utf-8") as fh:
        cols = ("frame", "event_id", "event_class", "subtype", "track_id", "state", "gid", "gt_iou", "stage",
                "visibility", "gt_track_id")
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(xref_rows)

    summary = {
        "diag_version": DIAG_VERSION, "source_id": source_id, "role": spec["role"], "frames": frames,
        "params": p.__dict__ | {"cover_grid": list(p.cover_grid)},
        "inputs": {"cache_run": spec["cache_run"], "replay_run": spec["replay_run"], "gt": spec["gt"],
                   "engine_run": run.name if run else None,
                   "cache_manifest_sha": _sha(cache_dir(data_root, source_id) / "cache_manifest.json")},
        "stage_visibility": _nested(stage_vis),
        "stage_visibility_size": {f"{s}|{v}|{z}": n for (s, v, z), n in sorted(stage_vis_size.items())},
        "tracker_input_detections": dict(det_out),
        "tracker_input_tp_conf_decile": {str(k): v for k, v in sorted(det_conf_hist.items())},
        "threshold_sweep": {str(t): dict(c) for t, c in sweep.items()},
        "ap50": {"all_person_boxes": round(average_precision(scored_all, n_gt_all), 4),
                 "on_pitch_non_referee": round(average_precision(scored_onpitch, n_gt_onpitch), 4),
                 "note": "cache floor 0.1: boxes below it were never stored, so AP is truncated at that confidence"},
        "engine_xref": _xref_summary(xref_rows),
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    (tmp / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    tmp.rename(out)
    return out


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _nested(c: Counter) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for (a, b), n in sorted(c.items()):
        out.setdefault(a, {})[b] = n
    return out


def _xref_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per engine event: the dominant ground-truth stage over its predicted frames."""
    by_event: dict[str, Counter] = defaultdict(Counter)
    meta: dict[str, tuple[str, str]] = {}
    for r in rows:
        by_event[r["event_id"]][(r["stage"], r["visibility"])] += 1
        meta[r["event_id"]] = (r["event_class"], r["subtype"])
    per_kind: dict[str, Counter] = defaultdict(Counter)
    for eid, c in by_event.items():
        (stage, vis), _n = c.most_common(1)[0]
        kind = "/".join(x for x in meta[eid] if x)
        per_kind[kind][stage if stage in ("no_gt_player",) else f"{stage}|{vis}"] += 1
    frames = Counter((r["event_class"], r["stage"]) for r in rows)
    return {"events_by_dominant_stage": {k: dict(v.most_common()) for k, v in sorted(per_kind.items())},
            "predicted_frames_by_stage": {f"{a}|{b}": n for (a, b), n in sorted(frames.items())}}
