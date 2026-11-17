"""Stage S1.1: temporal-intersection quality audit v3 across development matches (M11 advisor-directed pipeline).

Differences from the v2 audit (:mod:`.audit`, unchanged and still the code of the v2 outputs):

1. **Speed-spike correction.** Every update step of each player inside the +-2.5 s window whose speed exceeds
   ``spike_speed_m_s`` (12 m/s, the Step 1 physiological bound) is counted, whatever its length (Step 1 calls the sub-metre ones
   ``OVERSPEED_STEP`` and does not treat them as jumps). A spike landing within ``spike_near_s`` of that player's own crossing
   time adds the reason ``speed_spike_near_crossing``; a clean candidate with it becomes uncertain. Spikes elsewhere in the
   window are recorded only (S1 finding on 3812: most spikes occur away from the crossing with the players metres apart, so they
   are position noise, not identity exchanges).
2. **Detection in memory** with the unchanged v1 detector (:func:`.detect.detect`), so matches without v1 outputs can be
   audited. For a match with v1 outputs the events must equal the v1 events exactly (checked; recorded in the manifest).
3. Output ``data/processed/m11-intersections-v3/<match>/``: ``root.events.jsonl`` (geometric catalogue), ``root.eligibility.jsonl``,
   ``root.trajectories.npz``, ``root.references.jsonl``, ``root.summary.json``, ``root.config.json``, ``root.manifest.json``.
   Cross-match report: ``m11-intersections-v3/cross-match/``.

Development matches only; validation and test matches are refused here. Run:
``python -m football_fatigue.intersections.audit_v3 --matches 3812 3814 ...`` (``--dry-run`` writes nothing).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from football_fatigue.ingestion import EventVersion, IngestionError, approved_output_root, safe_output_path
from football_fatigue.matching import ml, validate
from football_fatigue.matching.benchmark import SOURCE_DIR, jsonl, load_source

from . import audit as AU
from .__main__ import peak_memory_mb, write_dir
from .detect import DetectConfig, detect

SCHEMA = "m11-intersections-v3"
SPIKE_REASON = "speed_spike_near_crossing"


@dataclass(frozen=True)
class AuditV3Config(AU.AuditConfig):
    spike_speed_m_s: float = 12.0
    spike_near_s: float = 0.5
    seed: str = "m11-intersections-v3"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def speed_spikes(cx: AU.Context, slot: int, rows: np.ndarray, t_cross: float, cfg: AuditV3Config) -> dict[str, Any]:
    """Update steps faster than ``spike_speed_m_s`` within the window rows (steps across gaps > max_gap_s are not steps)."""
    c = cx.c
    r = rows[cx.upd[rows, slot]]
    if len(r) < 2:
        return dict(spikes=0, spikes_near_crossing=0, max_step_speed_m_s=None, max_speed_near_crossing_m_s=None)
    dt = np.diff(c.video_time_s[r])
    ok = (dt > 0) & (dt <= cfg.max_gap_s)
    sp = np.hypot(np.diff(c.x[r, slot]), np.diff(c.y[r, slot])) / np.where(dt > 0, dt, np.inf)
    sp = np.where(ok, sp, 0.0)
    land = c.video_time_s[r[1:]]
    near = np.abs(land - t_cross) <= cfg.spike_near_s
    spike = sp > cfg.spike_speed_m_s
    return dict(spikes=int(spike.sum()), spikes_near_crossing=int((spike & near).sum()), max_step_speed_m_s=AU._f(float(sp.max()), 3),
                max_speed_near_crossing_m_s=AU._f(float(sp[near].max()), 3) if near.any() else None)


def audit_events(cx: AU.Context, events: list[dict]) -> list[dict]:
    cfg: AuditV3Config = cx.cfg
    recs = []
    for ev in events:
        m = AU.measure(cx, ev)
        ck = AU.checks(m, cfg)
        m.pop("_sin")
        rows = cx.tl.between(ev["period"], *m["window_s"])
        m["spikes_a"] = speed_spikes(cx, ev["slot_a"], rows, ev["time_a_s"], cfg)
        m["spikes_b"] = speed_spikes(cx, ev["slot_b"], rows, ev["time_b_s"], cfg)
        ck[SPIKE_REASON] = bool(m["spikes_a"]["spikes_near_crossing"] or m["spikes_b"]["spikes_near_crossing"])
        status = {}
        for tau in cfg.thresholds_s:
            s = AU.classify(m, ck, tau, cfg)
            if ck[SPIKE_REASON]:
                s["reasons"].append(SPIKE_REASON)
                if s["category"] == "clean_candidate":
                    s["category"] = "uncertain_candidate"
            status[AU.tkey(tau)] = s
        lo, hi = sorted((ev["slot_a"], ev["slot_b"]))
        recs.append(dict(event_id=ev["event_id"], event_key=ev["event_key"], match_id=ev["match_id"], period=ev["period"],
                         pair_id=f"{ev['match_id']}:{lo}-{hi}", v1_type=ev["type"], v1_cluster_id=ev["cluster_id"],
                         v1_cluster_size=ev["cluster_size"], v1_n_windows=ev["n_windows"], measurements=m, checks=ck, status=status,
                         primary_category=status[AU.tkey(cfg.primary_s)]["category"], primary_reasons=status[AU.tkey(cfg.primary_s)]["reasons"]))
    for tau in cfg.thresholds_s:
        AU.candidate_clusters(recs, AU.tkey(tau), cfg.cluster_gap_s)
    return recs


def compare_v2(raw_root, match_id, recs: list[dict]) -> dict[str, Any] | None:
    """Differences from the v2 audit of the same match: every change must be explained by the spike correction."""
    d = approved_output_root(raw_root) / AU.SCHEMA / str(match_id) / "root.eligibility.jsonl"
    if not d.exists():
        return None
    v2 = {json.loads(line)["event_id"]: json.loads(line) for line in d.read_text(encoding="utf-8").splitlines()}
    changed, unexplained = Counter(), []
    for r in recs:
        o = v2[r["event_id"]]
        for k, s in r["status"].items():
            a, b = o["status"][k], s
            if (a["category"], a["reasons"]) == (b["category"], b["reasons"]):
                continue
            ok = (b["reasons"] == a["reasons"] + [SPIKE_REASON]
                  and (b["category"] == a["category"] or (a["category"], b["category"]) == ("clean_candidate", "uncertain_candidate")))
            if ok:
                changed[f"{k}:{a['category']}->{b['category']}"] += 1
            else:
                unexplained.append((r["event_id"], k))
    return dict(events_v2=len(v2), same_event_ids=sorted(v2) == sorted(r["event_id"] for r in recs), changes=dict(changed),
                unexplained=len(unexplained), unexplained_examples=unexplained[:5])


def v1_regression(raw_root, match_id, events: list[dict]) -> dict[str, Any] | None:
    d = approved_output_root(raw_root) / "m11-intersections-v1" / str(match_id) / "root.events.jsonl"
    if not d.exists():
        return None
    v1 = [json.loads(line) for line in d.read_text(encoding="utf-8").splitlines()]
    same = jsonl(v1) == jsonl(events)
    return dict(events_v1=len(v1), events_now=len(events), identical=same,
                same_ids=[e["event_id"] for e in v1] == [e["event_id"] for e in events])


def audit_match(raw_root: str, match_id: str, cfg: AuditV3Config, source_root=None) -> tuple[list[dict], list[dict], Any, dict, dict]:
    c, mask, _enc, src_manifest, prov = load_source(raw_root, match_id, source_root)
    base = Path(source_root) if source_root is not None else approved_output_root(raw_root) / SOURCE_DIR
    qpath = safe_output_path(raw_root, match_id, EventVersion.ROOT, base, suffix=".quality.npz")
    with np.load(qpath) as z:
        steps = {k[5:]: z[k] for k in z.files if k.startswith("step_")}
    events, dstats = detect(c, mask, steps, DetectConfig(), quarantine=tuple(c.quarantine_periods))
    cx = AU.Context(c, mask, cfg)
    recs = audit_events(cx, events)
    return events, recs, cx, dstats, dict(step1=prov, step1_schema=src_manifest["schema"])


def run_match(raw_root: str, match_id: str, dry_run: bool, cfg: AuditV3Config = AuditV3Config()) -> dict[str, Any]:
    t0 = time.perf_counter()
    split = json.loads((approved_output_root(raw_root) / ml.SPLITS_DIR / "splits.json").read_text(encoding="utf-8"))
    validate.check_split(split, [str(match_id)], "development")
    base = approved_output_root(raw_root) / SCHEMA
    out = safe_output_path(raw_root, match_id, EventVersion.ROOT, base, suffix=".manifest.json").parent
    if not dry_run and out.exists():
        raise IngestionError(f"{out} already exists and is never overwritten")
    snap0 = validate.raw_snapshot(raw_root)
    events, recs, cx, dstats, src = audit_match(raw_root, match_id, cfg)
    store, refs = AU.build_store(cx, events, recs)
    ids = {x["event_id"] for x in refs}
    for r in recs:
        r["in_trajectory_store"] = r["event_id"] in ids
    summary = AU.summarize(recs, cfg)
    summary["schema"] = SCHEMA
    summary["detection"] = dstats
    summary["spikes"] = dict(events_with_spike_in_window=sum(bool(r["measurements"]["spikes_a"]["spikes"] or r["measurements"]["spikes_b"]["spikes"]) for r in recs),
                             events_with_spike_near_crossing=sum(r["checks"][SPIKE_REASON] for r in recs))
    summary["trajectory_store"] = dict(events=len(refs), rows=int(store["offsets"][-1]), window_half_s=cfg.window_half_s)
    summary["v1_regression"] = v1_regression(raw_root, match_id, events)
    summary["v2_comparison"] = compare_v2(raw_root, match_id, recs)
    summary["seconds"] = round(time.perf_counter() - t0, 1)
    summary["peak_memory_mb"] = peak_memory_mb()
    if summary["v1_regression"] and not summary["v1_regression"]["identical"]:
        raise IngestionError(f"match {match_id}: in-memory detection does not reproduce the v1 events")
    if summary["v2_comparison"] and summary["v2_comparison"]["unexplained"]:
        raise IngestionError(f"match {match_id}: differences from v2 not explained by the spike correction")
    if dry_run:
        return summary
    blobs = {"root.events.jsonl": jsonl(events), "root.eligibility.jsonl": jsonl(recs), "root.references.jsonl": jsonl(refs),
             "root.trajectories.npz": AU.npz_bytes(store), "root.summary.json": json.dumps(summary, indent=1, sort_keys=True).encode(),
             "root.config.json": json.dumps(cfg.as_dict(), indent=1, sort_keys=True).encode()}
    snap1 = validate.raw_snapshot(raw_root)
    manifest = dict(schema=SCHEMA, match_id=int(match_id), split=split["matches"][str(match_id)], config=cfg.as_dict(),
                    detect_config=DetectConfig().as_dict(),
                    code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))},
                    **src, raw_unchanged=snap0 == snap1, coordinate_policy="raw", provider_smoothed_coordinates_used=False,
                    outputs_sha256={n: hashlib.sha256(b).hexdigest() for n, b in blobs.items()}, python=platform.python_version(),
                    numpy=np.__version__, file_roles=dict(audit_only=["root.events.jsonl", "root.eligibility.jsonl"],
                                                          identity_references=["root.references.jsonl"],
                                                          trajectory_store_without_identities=["root.trajectories.npz"]))
    blobs["root.manifest.json"] = json.dumps(manifest, indent=1, sort_keys=True).encode()
    write_dir(out, blobs)
    summary["written_to"] = str(out)
    return summary


def cross_match(raw_root: str, summaries: dict[str, dict], cfg: AuditV3Config, dry_run: bool) -> dict[str, Any]:
    rows = {}
    for mid, s in summaries.items():
        b = s["by_threshold"]
        rows[mid] = dict(events=s["events"], spikes_near=s["spikes"]["events_with_spike_near_crossing"],
                         duplicate_position=s["checks_any_threshold"].get("duplicate_position", 0),
                         **{f"{cat}@{k}": v["categories"][cat] for k, v in b.items() for cat in AU.CATEGORIES},
                         **{f"clean_pairs@{k}": v["candidates"]["clean_distinct_pairs"] for k, v in b.items()},
                         **{f"clean_clusters@{k}": v["candidates"]["clean_candidate_clusters"] for k, v in b.items()},
                         invalid_share=round(b[AU.tkey(cfg.primary_s)]["categories"]["invalid_tracking_artifact"] / max(1, s["events"]), 4),
                         seconds=s["seconds"])
    tot = {k: sum(r[k] for r in rows.values()) for k in next(iter(rows.values())) if k not in ("invalid_share", "seconds")}
    rep = dict(schema=SCHEMA, matches=sorted(rows), per_match=rows, totals=tot, config=cfg.as_dict())
    if not dry_run:
        out = approved_output_root(raw_root) / SCHEMA / "cross-match"
        write_dir(out, {"summary.json": json.dumps(rep, indent=1, sort_keys=True).encode()})
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m football_fatigue.intersections.audit_v3")
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--matches", nargs="+", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    root = str(Path(a.root).resolve())
    summaries = {}
    try:
        for mid in a.matches:
            summaries[mid] = run_match(root, mid, a.dry_run)
            s = summaries[mid]
            print(mid, s["events"], {k: v["categories"] for k, v in s["by_threshold"].items()}, s["seconds"], "s", flush=True)
        rep = cross_match(root, summaries, AuditV3Config(), a.dry_run)
    except IngestionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(json.dumps(rep["totals"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
