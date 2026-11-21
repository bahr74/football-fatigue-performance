"""M13 immutable baseline: provenance of the M12 results and every headline number with its profile and denominator.

Reads only saved outputs (detection-cache manifests, the production replay manifest, the T0/T0L engine-v2 run
manifests, pipeline-v1, engine-t0l-v1) and records sha256 hashes of the historical result files, so a later run can
prove they were not changed (``verify``). No inference, no tracker run.

Output (write-once): ``<out_root>/_diag/m13-baseline-v1/manifest.json``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from football_fatigue.video.detection.diagnose import cache_dir
from football_fatigue.video.sources import SOURCES, guard_not_reserved
from football_fatigue.video.tracking import production_tracker_config

VERSION = "m13-baseline-v1"
DEV = ("soccertrack-117093", "soccertrack-118575", "soccertrack-117092", "soccertrack-118576")


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def historical_files(out_root: Path) -> list[Path]:
    d = Path(out_root) / "_diag"
    files = [d / "pipeline-v1" / "aggregate.json", d / "engine-t0l-v1" / "summary.json", d / "heavy-v1.json",
             d / "gallery-v2" / "index.json"]
    for sid in DEV:
        files.append(d / "pipeline-v1" / sid / "manifest.json")
        for p in ("B0", "T0", "T0L"):
            files += [d / "pipeline-v1" / sid / p / "summary.json", d / "pipeline-v1" / sid / p / "episodes.json"]
        for p in ("T0", "T0L"):
            files.append(Path(out_root) / sid / "tracking" / f"{p}-tracker-input-v1" / "frames.jsonl")
    return [f for f in files if f.exists()]


def _run_manifest(out_root: Path, sid: str, prof: str) -> dict | None:
    runs = sorted((Path(out_root) / sid / "occlusion" / "0.2.0").glob(f"occl-{sid}-0-*-v2-{prof}-ff8815ed"))
    runs = [r for r in runs if int(r.name.split("-")[4]) > 1000]
    return json.loads((runs[0] / "manifest.json").read_text(encoding="utf-8")) if runs else None


def build(data_root: Path, out_root: Path) -> Path:
    out_root = Path(out_root)
    dest = out_root / "_diag" / VERSION
    if (dest / "manifest.json").exists():
        raise SystemExit(f"output already exists (write-once): {dest}")
    agg = json.loads((out_root / "_diag" / "pipeline-v1" / "aggregate.json").read_text(encoding="utf-8"))
    eng = json.loads((out_root / "_diag" / "engine-t0l-v1" / "summary.json").read_text(encoding="utf-8"))
    sources, codes = {}, {}
    for sid in DEV:
        spec = SOURCES[sid]
        guard_not_reserved(sid, spec["match_id"], spec["cache_run"], spec["replay_run"], spec["gt"])
        cm = json.loads((cache_dir(data_root, sid) / "cache_manifest.json").read_text(encoding="utf-8"))
        prod = production_tracker_config(data_root, sid)
        run = _run_manifest(out_root, sid, "T0") or {}
        sources[sid] = {
            "role": spec["role"], "match_id": spec["match_id"], "video": spec["video"],
            "video_sha256": (run.get("source") or {}).get("video_sha256"), "fps": prod["video_fps"],
            "size_px": [prod["video_width_px"], prod["video_height_px"]], "gt": spec["gt"],
            "detection_cache_run": spec["cache_run"], "production_replay_run": spec["replay_run"],
            "detector": cm["detector"], "production_tracker": {"confidence_threshold": prod["confidence_threshold"],
                                                               **prod["tracker_params"]},
            "engine_v2_T0": {"run_id": run.get("run_id"), "provenance": run.get("provenance"), "config": run.get("config")},
        }
        for prof, P in agg["profiles"].items():
            v = P["per_match"][sid]
            codes.setdefault(prof, {})[sid] = {"evaluable": v["evaluable"], "codes": v["codes"],
                                               "id_changes": v["id_changes"]}

    def pooled(prof: str, code: str, sids=DEV) -> dict[str, Any]:
        n = sum(codes[prof][s]["codes"].get(code, 0) for s in sids)
        d = sum(codes[prof][s]["evaluable"] for s in sids)
        return {"count": n, "denominator": d, "pct": round(100 * n / d, 2)}

    claims = [
        {"claim": "no post-merge box ~6.37 %", "profile": "detector stage (identical for B0/T0/T0L)",
         "denominator": "evaluable GT player observations, 4 dev matches", **pooled("T0", "miss"),
         "caveat": "includes merge/NMS suppression; NOT a confirmed YOLO model miss"},
        {"claim": "117092 no post-merge box ~14.1 %", "profile": "detector stage", "denominator": "evaluable obs, 117092",
         **pooled("T0", "miss", ("soccertrack-117092",))},
        {"claim": "rejected by confidence ~3.00 %", "profile": "B0 and T0 (same 0.4 threshold; 0 in T0L)",
         "denominator": "evaluable obs, 4 matches", **pooled("T0", "rej_conf"),
         "B0_equal": pooled("B0", "rej_conf")["count"] == pooled("T0", "rej_conf")["count"]},
        {"claim": "reached ByteTrack but not tracked ~0.80 %", "profile": "T0", "denominator": "evaluable obs",
         **pooled("T0", "untracked"), "other_profiles": {"B0": pooled("B0", "untracked"), "T0L": pooled("T0L", "untracked")}},
        {"claim": "turf filter losses", "profile": "B0 only (T0/T0L ignore the turf filter)", "denominator": "evaluable obs",
         **pooled("B0", "rej_turf")},
        {"claim": "~1,890 temporary-ID changes", "profile": "T0",
         "count": sum(codes["T0"][s]["id_changes"]["total"] for s in DEV), "denominator": "events (not observations)",
         "per_match": {s: codes["T0"][s]["id_changes"] for s in DEV}},
        {"claim": "T0L reduces fragmentation ~33-37 %", "profile": "T0 vs T0L",
         "tune_frag": [eng["by_role"]["tune"]["T0"]["sum_Frag"], eng["by_role"]["tune"]["T0L"]["sum_Frag"]],
         "check_frag": [eng["by_role"]["check"]["T0"]["sum_Frag"], eng["by_role"]["check"]["T0L"]["sum_Frag"]],
         "pct_tune": round(100 * (1 - eng["by_role"]["tune"]["T0L"]["sum_Frag"] / eng["by_role"]["tune"]["T0"]["sum_Frag"]), 1),
         "pct_check": round(100 * (1 - eng["by_role"]["check"]["T0L"]["sum_Frag"] / eng["by_role"]["check"]["T0"]["sum_Frag"]), 1)},
        {"claim": "T0L: little HOTA / PO-F1 change", "profile": "T0 vs T0L",
         "HOTA": {r: [eng["by_role"][r]["T0"]["mean_HOTA"], eng["by_role"][r]["T0L"]["mean_HOTA"]] for r in ("tune", "check")},
         "PO_F1": {r: [eng["by_role"][r]["T0"]["mean_po_f1"], eng["by_role"][r]["T0L"]["mean_po_f1"]] for r in ("tune", "check")}},
    ]
    try:
        sv = metadata.version("supervision")
        ul = metadata.version("ultralytics")
    except metadata.PackageNotFoundError:  # pragma: no cover
        sv = ul = None
    tp = sources[DEV[0]]["production_tracker"]
    man = {
        "version": VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(), "sources": sources,
        "software": {"supervision": sv, "ultralytics": ul},
        "tracker_semantics": {
            "implementation": f"supervision {sv} ByteTrack via vision.tracker.MultiObjectTracker",
            "first_association": "conf >= track_activation_threshold (0.5)",
            "second_association": "0.1 < conf < track_activation_threshold",
            "new_track_min_conf": round(tp["activation"] + 0.1, 2),
            "lost_track_buffer_effective_frames": int(25 / 30 * tp["buffer_frames"]),
            "minimum_matching_threshold": tp["matching_threshold"], "minimum_consecutive_frames": 1,
            "correction": "M12 said boxes <= 0.5 cannot start tracks; the code requires conf >= activation + 0.1 = 0.6"},
        "detector_semantics": {"checkpoint_end2end": False,
                               "per_tile_nms": "Ultralytics class-aware NMS, IoU 0.7, max_det 300 (default cfg)",
                               "cross_tile_merge": "edge_aware, merge IoU 0.5, cross-role IoU 0.7",
                               "cache": "post-merge boxes, conf >= 0.1"},
        "gt_matching": {"pipeline-v1": "Hungarian GT-to-cached-box, IoU >= 0.5; track found by IoU > 0.99",
                        "tracking_metrics": "TrackEval HOTA/CLEAR/Identity, IoU 0.5 (CLEAR/Identity), all tracks, "
                                            "GT = 22 annotated players (referees unlabelled -> count as FP)",
                        "engine": "eval-v1 (GT-derived occlusion/overlap episodes, not human-confirmed)"},
        "profiles": {"B0": "conf > 0.4 + turf filter, production ByteTrack (Session A replay)",
                     "T0": "conf > 0.4, turf filter ignored, production ByteTrack (recommended)",
                     "T0L": "conf > 0.1, turf filter ignored, production ByteTrack (experimental, not promoted)"},
        "claims": claims,
        "historical_files_sha256": {str(f.relative_to(out_root)): _sha(f) for f in historical_files(out_root)},
    }
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "manifest.json").write_text(json.dumps(man, indent=1, default=str), encoding="utf-8")
    return dest / "manifest.json"


def verify(out_root: Path) -> dict[str, Any]:
    """Recompute the hashes recorded in the baseline manifest: every historical file must be unchanged."""
    man = json.loads((Path(out_root) / "_diag" / VERSION / "manifest.json").read_text(encoding="utf-8"))
    changed = [k for k, v in man["historical_files_sha256"].items()
               if not (Path(out_root) / k).exists() or _sha(Path(out_root) / k) != v]
    return {"files": len(man["historical_files_sha256"]), "changed": changed, "unchanged": not changed}
