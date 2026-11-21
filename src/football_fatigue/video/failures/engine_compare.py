"""B0 vs T0 vs T0L: tracking metrics and occlusion-engine v2 accuracy on the same development matches.

All three profiles use the same detection cache, the same production ByteTrack (supervision, activation 0.5, buffer
75, match 0.95) and the same engine profile v2 (config ff8815ed); only the tracker input differs. Tracking metrics are
recomputed here with one function (``detection.replay.score``, TrackEval HOTA/CLEAR/Identity) for all three, from the
``frames.jsonl`` the engine read. Engine numbers come from the saved eval-v1 files. Event classification changes are
found by pairing events of two profiles that involve the same GT players (via eval-v1 ``member_gt``) and overlap in time.

Output (write-once): ``<out_root>/_diag/engine-t0l-v1/{summary.json, <source>.json}``.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from football_fatigue.video.detection.replay import score
from football_fatigue.video.sources import SOURCES
from football_fatigue.video.tracking import tracks_dir

VERSION = "engine-t0l-v1"
PROFILES = ("B0", "T0", "T0L")
DEV = ("soccertrack-117093", "soccertrack-118575", "soccertrack-117092", "soccertrack-118576")


def _tracks(data_root: Path, out_root: Path, sid: str, prof: str) -> dict[int, list]:
    out = {}
    with (tracks_dir(data_root, out_root, sid, prof) / "frames.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            out[int(r["frame_num"])] = [(int(p["track_id"]), tuple(p["bbox_xyxy_px"]), float(p["confidence"]), p.get("observed_class"))
                                        for p in r["players"]]
    return out


def _engine(out_root: Path, sid: str, prof: str) -> tuple[Path, dict] | None:
    base = Path(out_root) / sid / "occlusion"
    runs = sorted((base / "0.2.0").glob(f"occl-{sid}-0-*-v2-{prof}-ff8815ed")) if prof != "B0" else \
        sorted((base / "0.2.0").glob(f"occl-{sid}-0-*-v2-ff8815ed"))
    runs = [r for r in runs if int(r.name.split("-")[3]) == 0 and int(r.name.split("-")[4]) > 1000]
    if not runs:
        return None
    ev = base / "evaluation" / f"{runs[0].name}.eval-v1.json"
    return (runs[0], json.loads(ev.read_text(encoding="utf-8"))) if ev.exists() else None


def _events(run: Path) -> dict[str, dict]:
    with (run / "events.jsonl").open(encoding="utf-8") as fh:
        return {e["event_id"]: e for e in (json.loads(x) for x in fh if x.strip())}


def _pair_changes(a: dict, b: dict) -> dict[str, Any]:
    """How events of profile a map onto events of profile b (same GT players, overlapping frames)."""
    def keyed(ev):
        out = []
        for e in ev["events"]:
            # member_gt maps the event's temporary track IDs to the GT players they matched (None = no GT player)
            gids = frozenset(int(v) for v in (e.get("member_gt") or {}).values() if v is not None)
            out.append((gids, e["start_frame"], e["end_frame"], e["event_class"], e["outcome"]))
        return out
    A, B = keyed(a), keyed(b)
    by_gids: dict[frozenset, list] = {}
    for i, x in enumerate(B):
        by_gids.setdefault(x[0], []).append(i)
    used = set()
    trans = Counter()
    for g, s, e, c, _o in A:
        if not g:
            trans[(c, "unmapped (no GT member)")] += 1
            continue
        cand = [i for i in by_gids.get(g, []) if i not in used and B[i][1] <= e and B[i][2] >= s]
        if cand:
            i = max(cand, key=lambda i: min(e, B[i][2]) - max(s, B[i][1]))
            used.add(i)
            trans[(c, B[i][3])] += 1
        else:
            trans[(c, "none")] += 1
    for i, x in enumerate(B):
        if i not in used:
            trans[("none" if x[0] else "unmapped (no GT member)", x[3])] += 1
    return {f"{k[0]} -> {k[1]}": v for k, v in sorted(trans.items(), key=lambda kv: -kv[1])}


def compare(data_root: Path, out_root: Path, sources: tuple[str, ...] = DEV, profiles: tuple[str, ...] = PROFILES,
            version: str = VERSION) -> Path:
    PROFILES = profiles  # noqa: N806 - the profiles compared in this run
    dest = Path(out_root) / "_diag" / version
    if (dest / "summary.json").exists():
        raise SystemExit(f"output already exists (write-once): {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for sid in sources:
        per = {"source_id": sid, "role": SOURCES[sid]["role"], "profiles": {}}
        evals = {}
        for prof in PROFILES:
            tr = _tracks(data_root, out_root, sid, prof)
            m = score(data_root, sid, tr, on_pitch_only=False)
            e = _engine(out_root, sid, prof)
            item: dict[str, Any] = {"tracking": {"HOTA": m["hota"]["HOTA"], "DetA": m["hota"]["DetA"], "AssA": m["hota"]["AssA"],
                                                 "IDF1": m["identity"]["IDF1"], "IDSW": m["clear"]["IDSW"], "Frag": m["clear"]["Frag"],
                                                 "MOTA": m["clear"].get("MOTA"),
                                                 "track_ids": len({t[0] for v in tr.values() for t in v}),
                                                 "tracked_boxes": sum(len(v) for v in tr.values())}}
            if e is None:
                item["engine"] = None
            else:
                run, ev = e
                evals[prof] = ev
                events = _events(run)
                s = ev["summary"]
                po, bo = s["probable_occlusion"], s["box_overlap"]
                dur = Counter()
                lens: dict[str, list] = {}
                for x in events.values():
                    dur[x["event_class"]] += 1
                    lens.setdefault(x["event_class"], []).append(x["end_frame"] - x["start_frame"] + 1)
                eps = [x for x in ev["episodes"] if x["tier"] == "occlusion"]
                item["engine"] = {
                    "run_id": run.name, "events_total": len(events), "events_by_class": dict(dur),
                    "event_duration_median_frames": {k: float(np.median(v)) for k, v in lens.items()},
                    "po_events": po["events"], "po_outcomes": po["outcomes"],
                    "po_precision": po["vs_occlusion_episodes"]["precision"], "po_recall": po["vs_occlusion_episodes"]["recall"],
                    "po_f1": po["vs_occlusion_episodes"]["f1"],
                    "false_occlusion_alerts": sum(v for k, v in po["outcomes"].items() if k.startswith("fp_")),
                    "po_on_unannotated_person": po["outcomes"].get("non_gt_person", 0),
                    "occlusion_episodes": len(eps), "occlusion_episodes_missed_by_po": len(eps) - po["vs_occlusion_episodes"]["episodes_matched"],
                    "occlusion_episodes_missed_any_class": round((1 - s["occlusion_episode_recall_any_class"]["all"]) * len(eps)),
                    "po_timing_frames": po["timing_frames"],
                    "bo_events": bo["events"], "bo_precision": bo["vs_overlap_episodes"]["precision"],
                    "bo_recall": bo["vs_overlap_episodes"]["recall"],
                    "detection_loss": s["detection_loss"],
                    "loss_not_occlusion_precision": s["detection_loss_not_occlusion_precision"],
                }
            per["profiles"][prof] = item
        if "T0" in evals and "T0L" in evals:
            per["classification_changes_T0_to_T0L"] = _pair_changes(evals["T0"], evals["T0L"])
        for other in [p for p in PROFILES if p not in ("B0", "T0") and p in evals and "T0" in evals]:
            if other != "T0L":
                per[f"classification_changes_T0_to_{other}"] = _pair_changes(evals["T0"], evals[other])
        if "B0" in evals and "T0" in evals:
            per["classification_changes_B0_to_T0"] = _pair_changes(evals["B0"], evals["T0"])
        (dest / f"{sid}.json").write_text(json.dumps(per, indent=1), encoding="utf-8")
        rows.append(per)
    agg: dict[str, Any] = {}
    for role in ("tune", "check"):
        rs = [r for r in rows if r["role"] == role]
        agg[role] = {}
        for prof in PROFILES:
            t = [r["profiles"][prof]["tracking"] for r in rs]
            e = [r["profiles"][prof]["engine"] for r in rs if r["profiles"][prof]["engine"]]
            agg[role][prof] = {
                "matches": [r["source_id"] for r in rs],
                **{f"mean_{k}": round(float(np.mean([x[k] for x in t])), 4) for k in ("HOTA", "DetA", "AssA", "IDF1")},
                **{f"sum_{k}": int(sum(x[k] for x in t)) for k in ("IDSW", "Frag")},
                **({"engine_matches": len(e),
                    "sum_po_events": sum(x["po_events"] for x in e),
                    "pooled_false_occlusion_alerts": sum(x["false_occlusion_alerts"] for x in e),
                    "pooled_occlusion_episodes_missed_by_po": sum(x["occlusion_episodes_missed_by_po"] for x in e),
                    "mean_po_precision": round(float(np.mean([x["po_precision"] for x in e])), 4),
                    "mean_po_recall": round(float(np.mean([x["po_recall"] for x in e])), 4),
                    "mean_po_f1": round(float(np.mean([x["po_f1"] for x in e])), 4),
                    "sum_events": sum(x["events_total"] for x in e)} if e else {}),
            }
    summary = {"version": version, "profiles": list(PROFILES), "created_at_utc": datetime.now(timezone.utc).isoformat(), "by_role": agg,
               "matches": [{"source_id": r["source_id"], "role": r["role"],
                            **{p: {"tracking": r["profiles"][p]["tracking"],
                                   "engine": {k: v for k, v in (r["profiles"][p]["engine"] or {}).items()
                                              if k not in ("detection_loss",)}} for p in PROFILES}} for r in rows],
               "notes": ["same detections, same ByteTrack, same engine v2; only the tracker input differs",
                         "tracking metrics recomputed with detection.replay.score (all tracks, GT = 22 players)",
                         "PO precision/recall against GT-derived occlusion episodes (eval-v1), not human-confirmed occlusion"]}
    (dest / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return dest
