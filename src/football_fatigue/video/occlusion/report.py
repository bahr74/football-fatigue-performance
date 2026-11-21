"""Collect saved evaluations into one before/after table (Step 1 baseline vs v2) per development match."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from football_fatigue.video.sources import SOURCES

ROLE_ORDER = {"tune": 0, "check": 1}


def collect(out_root: Path) -> list[dict[str, Any]]:
    """One row per saved evaluation file under ``<out_root>/<source>/occlusion/evaluation/``."""
    rows = []
    for path in sorted(Path(out_root).glob("*/occlusion/evaluation/*.json")):
        e = json.loads(path.read_text(encoding="utf-8"))
        s = e["summary"]
        po, bo = s["probable_occlusion"], s["box_overlap"]
        pod = po["vs_detectable_occlusion_episodes"]
        bod = bo["vs_detectable_overlap_episodes"]
        anyc = s["occlusion_episode_recall_any_class"]
        claims = s.get("pair_claims", {})
        loss = s["detection_loss"]
        rows.append({
            "source_id": e["source_id"], "match": SOURCES.get(e["source_id"], {}).get("match_id"),
            "role": e.get("split_role"), "profile": e.get("profile"), "run_id": e["run_id"], "eval_file": str(path),
            "frames": e["frame_range"][1] - e["frame_range"][0] + 1, "events": s["events_total"],
            "uncertain": s["uncertain_events"],
            "po_events": po["events"], "po_precision": pod["precision"], "po_recall": pod["recall"], "po_f1": pod["f1"],
            "po_start_abs_err": po["timing_frames"].get("start_abs_error_median"),
            "po_end_abs_err": po["timing_frames"].get("end_abs_error_median"),
            "po_outcomes": po["outcomes"],
            "bo_events": bo["events"], "bo_precision": bod["precision"], "bo_recall": bod["recall"], "bo_f1": bod["f1"],
            "occ_recall_any": anyc["detectable"], "occ_episodes": anyc["detectable_episodes"], "occ_miss_causes": anyc["miss_causes"],
            "claim_occ_precision": claims.get("occlusion", {}).get("precision"),
            "claim_occ_outcomes": claims.get("occlusion", {}).get("outcomes"),
            "claim_ovl_precision": claims.get("overlap", {}).get("precision"),
            "loss_true": sum(v.get("true_loss", 0) for v in loss.values()),
            "loss_occluded_in_gt": sum(v.get("actually_occluded_in_gt", 0) for v in loss.values()),
            "loss_non_gt_person": sum(v.get("non_gt_person", 0) for v in loss.values()),
            "loss_by_subtype": loss,
            "loss_not_occlusion_precision": s["detection_loss_not_occlusion_precision"],
        })
    rows.sort(key=lambda r: (ROLE_ORDER.get(r["role"], 9), r["match"] or "", r["profile"] != "step1"))
    return rows


def markdown(rows: list[dict[str, Any]]) -> str:
    cols = [("match", "match"), ("role", "role"), ("profile", "profile"), ("events", "events"),
            ("po_events", "PO ev"), ("po_precision", "PO P"), ("po_recall", "PO R"), ("po_f1", "PO F1"),
            ("claim_occ_precision", "claim P (occl)"), ("po_start_abs_err", "|start err|"), ("po_end_abs_err", "|end err|"),
            ("bo_events", "BO ev"), ("bo_precision", "BO P"), ("bo_recall", "BO R"), ("claim_ovl_precision", "claim P (ovl)"),
            ("occ_recall_any", "occl recall (any)"), ("loss_not_occlusion_precision", "loss-not-occl P")]
    out = ["| " + " | ".join(h for _k, h in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join("" if r[k] is None else (f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k])) for k, _h in cols) + " |")
    return "\n".join(out)
