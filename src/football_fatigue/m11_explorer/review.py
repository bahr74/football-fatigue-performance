"""Human review queue: the 100 Phase 2 stratified review events and the 25 Phase 2 severe motion contradictions.

Nothing is ever marked reviewed automatically. A decision exists only when a named reviewer saves one; decisions are appended to
``data/processed/m11-human-review-v1/decisions/<reviewer>.jsonl`` (append-only, every save kept) and exports are new timestamped
files. A figure or animation is an aid for the reviewer, not independent ground truth.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

from football_fatigue.m11_phase2 import common as C

REVIEW_DIR = C.PROC / "m11-phase2-review-sample-v1"
SEVERE = C.PROC / "m11-phase2-ambiguity-benchmark-v1" / "severe_failures_top25.parquet"
STAGE2 = C.PROC / "m11-stage2-candidates-v1" / "matches"
OUT = C.PROC / "m11-human-review-v1"
SCHEMA = "m11-human-review-decision-v1"

QUESTIONS = {
    "positions_plausible": ("Are both players' positions physically plausible around the event?", ["unsure", "yes", "no"]),
    "players_actually_close": ("Were the two players actually close (geometric proximity only)?", ["unsure", "yes", "no"]),
    "identity_continuity_plausible": ("Is each provider identity continuous (no swap of labels between the players)?",
                                      ["unsure", "yes", "no - swap suspected", "no - other break"]),
    "provider_artifact_suspected": ("Which provider artefact, if any, is suspected?",
                                    ["none", "smooth ESTIMATED interpolation", "clone-like duplicated trace", "position jump",
                                     "identity swap", "other"]),
    "unexplained_contradiction": ("Is there a physical contradiction the provider flags do not explain?", ["unsure", "no", "yes"]),
    "visual_occlusion_checkable": ("Can visual occlusion be checked from this material?", ["no (tracking data only)", "yes (video available)"]),
}


def _slots_for(match: int, candidate_id: str) -> tuple[int, int]:
    c = pd.read_parquet(STAGE2 / str(match) / "candidates.parquet", columns=["candidate_id", "slot_a", "slot_b"]).set_index("candidate_id")
    r = c.loc[candidate_id]
    return int(r["slot_a"]), int(r["slot_b"])


def queue() -> pd.DataFrame:
    """Unified queue (125 rows): review sample first (R…), then severe failures (S01…S25)."""
    r = pd.read_parquet(REVIEW_DIR / "review_manifest.parquet")
    a = pd.DataFrame(dict(review_id=r["review_id"], source="phase2 stratified review sample", match_id=r["match_id"].astype(int),
                          period=r["period"].astype(int), slot_a=r["slot_a"].astype(int), slot_b=r["slot_b"].astype(int), t_star=r["t_star"],
                          start=r["start"], end=r["end"], radius_m=r["radius_m"], d_min=r["d_min"], stratum=r["stratum"],
                          status=r["status_composition"], cv_tier=r["cv_tier"], label_risk=r["label_risk"].astype(bool),
                          clone_like=r["estimated_comotion"].astype(bool), figure=r["figure"].map(lambda f: str(REVIEW_DIR / f) if f else "")))
    s = pd.read_parquet(SEVERE).reset_index(drop=True)
    C.checked(sorted({str(int(v)) for v in s["match_id"]}))
    sl = [_slots_for(int(m), str(cid)) for m, cid in zip(s["match_id"], s["candidate_id"])]
    b = pd.DataFrame(dict(review_id=[f"S{i + 1:02d}" for i in range(len(s))], source="phase2 severe motion contradiction",
                          match_id=s["match_id"].astype(int), period=s["period"].astype(int), slot_a=[v[0] for v in sl], slot_b=[v[1] for v in sl],
                          t_star=s["t_star"], start=s["t_star"] - 0.4, end=s["t_star"] + 0.4, radius_m=0.5, d_min=s["d_min"],
                          stratum="severe: all physics methods wrong, " + s["cv_bidir_margin_against_truth_m"].round(1).astype(str) + " m against truth",
                          status=s["catalogue_status"].astype(str), cv_tier=s["cv_tier"].astype(str), label_risk=s["label_risk"].astype(bool),
                          clone_like=s["clone_pair"].astype(bool), figure=""))
    q = pd.concat([a, b], ignore_index=True)
    C.checked(sorted({str(v) for v in q["match_id"]}))
    return q


def _safe(name: str) -> str:
    n = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())[:40]
    if not n:
        raise ValueError("a reviewer name is required")
    return n


def save_decision(reviewer: str, review_id: str, answers: dict[str, str], notes: str = "", out: Path = OUT) -> dict[str, Any]:
    """Append one decision (never overwrites earlier saves)."""
    bad = set(answers) - set(QUESTIONS)
    if bad:
        raise ValueError(f"unknown fields {sorted(bad)}")
    for k, v in answers.items():
        if v not in QUESTIONS[k][1]:
            raise ValueError(f"{k}: {v!r} is not an allowed answer")
    rec = dict(schema=SCHEMA, reviewer=_safe(reviewer), review_id=str(review_id), saved_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               answers=answers, notes=str(notes), human_reviewed=True)
    d = out / "decisions"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{rec['reviewer']}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def decisions(out: Path = OUT) -> pd.DataFrame:
    """Latest decision per (reviewer, review_id)."""
    rows = []
    for p in sorted((out / "decisions").glob("*.jsonl")) if (out / "decisions").exists() else []:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows.append(dict(reviewer=r["reviewer"], review_id=r["review_id"], saved_at=r["saved_at"], notes=r.get("notes", ""), **r["answers"]))
    if not rows:
        return pd.DataFrame(columns=["reviewer", "review_id", "saved_at", "notes", *QUESTIONS])
    df = pd.DataFrame(rows)
    return df.groupby(["reviewer", "review_id"], as_index=False).last()


def export(reviewer: str, out: Path = OUT) -> Path:
    """New timestamped CSV: the full queue joined with this reviewer's latest decisions (unreviewed rows stay 'not reviewed')."""
    rv = _safe(reviewer)
    q = queue()
    d = decisions(out)
    d = d[d["reviewer"] == rv]
    m = q.merge(d, on="review_id", how="left")
    m["review_state"] = m["saved_at"].notna().map({True: "human reviewed", False: "not reviewed"})
    p = out / "exports" / f"review_export_{rv}_{time.strftime('%Y%m%dT%H%M%S')}.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        raise SystemExit(f"{p} exists; not overwritten")
    m.to_csv(p, index=False)
    return p
