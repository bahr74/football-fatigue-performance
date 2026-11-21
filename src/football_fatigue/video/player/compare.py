"""B0 vs T0 comparison from saved outputs only (tracker replays, occlusion runs, their eval-v1 scores).

``python -m football_fatigue.video.player.compare`` writes ``<out_root>/_comparison/b0-t0-v1/{comparison.json,
comparison.md}`` (write-once unless --overwrite). Both profiles use the identical detection cache, ByteTrack
parameters, engine profile ``v2`` (config ff8815ed) and evaluation (eval-v1); only the tracker input differs.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from football_fatigue.video import overlay as ov
from football_fatigue.video.sources import SOURCES

COMPARISON_VERSION = "b0-t0-v1"
PROFILES = ("B0", "T0")


def _engine_run(out_root: Path, sid: str, ti: str, profile: str = "v2") -> Path | None:
    base = Path(out_root) / sid / "occlusion" / "0.2.0"
    pat = f"occl-{sid}-0-*-{profile}-T0-*" if ti == "T0" else f"occl-{sid}-0-*-{profile}-*"
    runs = [p for p in sorted(base.glob(pat)) if (p / "manifest.json").exists()
            and json.loads((p / "manifest.json").read_text(encoding="utf-8")).get("tracker_input", "B0") == ti]
    return runs[-1] if runs else None


def _tracking(out_root: Path, sid: str, ti: str) -> dict[str, Any] | None:
    p = Path(out_root) / sid / "detection" / "replay-v1" / ti / "result.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text(encoding="utf-8"))
    m = r["metrics_all_tracks"]
    return {"HOTA": m["hota"]["HOTA"], "DetRe": m["hota"]["DetRe"], "DetPr": m["hota"]["DetPr"], "AssA": m["hota"]["AssA"],
            "IDF1": m["identity"]["IDF1"], "IDSW": m["clear"]["IDSW"], "Frag": m["clear"].get("Frag"),
            "MOTA": m["clear"].get("MOTA"), "source": str(p.relative_to(out_root)).replace("\\", "/")}


def _engine(out_root: Path, sid: str, ti: str) -> dict[str, Any] | None:
    run = _engine_run(out_root, sid, ti)
    if run is None:
        return None
    events = [json.loads(line) for line in (run / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    cats = Counter(ov.category_of(e, False) for e in events)
    referee = sum(1 for e in events if "involves_referee_label" in (e.get("flags") or []))
    out: dict[str, Any] = {"run_id": run.name, "events_total": len(events), "by_category": dict(cats),
                           "uncertain": sum(1 for e in events if e["uncertain"]), "referee_label_events": referee,
                           "unique_track_ids": len({int(m["track_id"]) for e in events for m in e["members"]})}
    ev = run.parent.parent / "evaluation" / f"{run.name}.eval-v1.json"
    if ev.exists():
        s = json.loads(ev.read_text(encoding="utf-8"))["summary"]
        po, bo, dl = s["probable_occlusion"], s["box_overlap"], s["detection_loss"]
        pc = s.get("pair_claims", {})
        out["eval"] = {
            "po_events": po["events"], "po_outcomes": po["outcomes"],
            "po_false": sum(v for k, v in po["outcomes"].items() if k.startswith("fp_")),
            "po_non_gt_person": po["outcomes"].get("non_gt_person", 0),
            "po_precision": po["vs_detectable_occlusion_episodes"]["precision"],
            "po_recall": po["vs_detectable_occlusion_episodes"]["recall"],
            "po_f1": po["vs_detectable_occlusion_episodes"]["f1"],
            "po_end_abs_err": po["timing_frames"].get("end_abs_error_median"),
            "claim_occ_precision": pc.get("occlusion", {}).get("precision"),
            "claim_occ_outcomes": pc.get("occlusion", {}).get("outcomes"),
            "bo_events": bo["events"], "bo_precision": bo["vs_detectable_overlap_episodes"]["precision"],
            "bo_recall": bo["vs_detectable_overlap_episodes"]["recall"],
            "occ_recall_any": s["occlusion_episode_recall_any_class"]["detectable"],
            "detector_miss": dl.get("detector_miss", {}), "tracker_gap": dl.get("tracker_gap", {}),
            "loss_not_occlusion_precision": s["detection_loss_not_occlusion_precision"],
            "source": str(ev.relative_to(out_root)).replace("\\", "/")}
    return out


def compare_source(out_root: Path, sid: str) -> dict[str, Any]:
    return {"source_id": sid, "role": SOURCES.get(sid, {}).get("role"),
            "profiles": {ti: {"tracking": _tracking(out_root, sid, ti), "engine": _engine(out_root, sid, ti)}
                         for ti in PROFILES}}


def _f(v, nd=3):
    return "–" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def markdown(rows: list[dict[str, Any]]) -> str:
    L = ["# B0 vs T0 (development matches; identical detections, ByteTrack, engine v2 and eval-v1)", "",
         "## Tracking (TrackEval vs SoccerTrack GT, 22 players, no referees)", "",
         "| match | role | input | HOTA | DetRe | DetPr | AssA | IDF1 | IDSW | Frag |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        for ti in PROFILES:
            t = r["profiles"][ti]["tracking"] or {}
            L.append(f"| {r['source_id'][12:]} | {r['role']} | {ti} | {_f(t.get('HOTA'))} | {_f(t.get('DetRe'))} | "
                     f"{_f(t.get('DetPr'))} | {_f(t.get('AssA'))} | {_f(t.get('IDF1'))} | {_f(t.get('IDSW'))} | {_f(t.get('Frag'))} |")
    L += ["", "## Occlusion engine events (v2, full match) and their GT outcome (eval-v1)", "",
          "| match | input | events | detector miss (GT player / non-GT) | tracker gap (GT player / non-GT) | "
          "probable occl. | PO false | PO non-GT | PO P | PO R | PO F1 | claim P (occl) | box overlap | BO P | BO R | "
          "occl. recall (any) | referee-label events |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        for ti in PROFILES:
            e = r["profiles"][ti]["engine"]
            if not e:
                L.append(f"| {r['source_id'][12:]} | {ti} | not run | | | | | | | | | | | | | | |")
                continue
            v = e.get("eval", {})
            dm, tg = v.get("detector_miss", {}), v.get("tracker_gap", {})
            L.append(f"| {r['source_id'][12:]} | {ti} | {e['events_total']} | "
                     f"{e['by_category'].get('detector_miss', 0)} ({dm.get('true_loss', 0)} / {dm.get('non_gt_person', 0)}) | "
                     f"{e['by_category'].get('tracker_gap', 0)} ({tg.get('true_loss', 0)} / {tg.get('non_gt_person', 0)}) | "
                     f"{v.get('po_events', '–')} | {v.get('po_false', '–')} | {v.get('po_non_gt_person', '–')} | "
                     f"{_f(v.get('po_precision'))} | {_f(v.get('po_recall'))} | {_f(v.get('po_f1'))} | "
                     f"{_f(v.get('claim_occ_precision'))} | {v.get('bo_events', '–')} | {_f(v.get('bo_precision'))} | "
                     f"{_f(v.get('bo_recall'))} | {_f(v.get('occ_recall_any'))} | {e['referee_label_events']} |")
    L += ["", "Event counts use the class rows (uncertain events stay in their class). 'PO false' = fp_gt_overlap_only + "
          "fp_no_gt_overlap + fp_duplicate_track; non-GT person events (referees, staff, false detections) are not "
          "scored. Development data only; 117092 and 118576 are check matches, not independent validation."]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m football_fatigue.video.player.compare")
    ap.add_argument("--out-root", default="data/processed/video")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    out_root = Path(a.out_root)
    rows = [compare_source(out_root, sid) for sid in SOURCES]
    out = out_root / "_comparison" / COMPARISON_VERSION
    if out.exists() and not a.overwrite:
        raise SystemExit(f"output already exists (write-once): {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    (out / "comparison.md").write_text(markdown(rows), encoding="utf-8")
    print(markdown(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
