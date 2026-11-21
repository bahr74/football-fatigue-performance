"""Identity-free player contact catalogue (S1.2, ``m11-contacts-v1``) for unbiased identity-recovery samples.

**Why.** A geometric path intersection is computed from the provider's identity-labelled trajectories: selecting only
intersections guarantees that the true continuation lies on the far side of the other player's path. On development match
3812 the rule "choose the assignment whose continuation is *farther* away" therefore agrees with the label ~75 % of the time
without any motion model. A tracker at inference does not know whether two meeting players crossed or turned back; the
selection must not know either.

**Definition.** For every pair of slots and every period, the same-time distance d(t) between the two RAW positions (an
unordered pair of points per frame, so no identity is used) defines *contact episodes*: maximal runs of frames with
d <= ``r_exit_m`` that contain at least one frame with d <= ``r_enter_m`` (a frame where either player is unobserved ends a run).
The contact time ``t*`` is the first frame of minimum distance. Each episode is one event, compatible with the v3 audit
(both "crossing times" = t*, so the time difference is 0 and every event is time-aligned): the same quality checks, spike rule,
complete-trajectory store and references are produced by :mod:`.audit_v3` code unchanged.

**Geometry (stratum, never a selection criterion).** ``geometric_crossing``: a v1-detector path intersection of the same pair
whose two traversal times both lie inside the episode (+-``link_pad_s``); ``aligned_crossing``: such an intersection with
|dt| <= 0.20 s (the advisor-directed candidate definition).

Run: ``python -m football_fatigue.intersections.contacts --matches <development ids> [--dry-run]``.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
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
from . import audit_v3 as A3
from .__main__ import peak_memory_mb, write_dir
from .detect import DetectConfig, assign_clusters, detect

SCHEMA = "m11-contacts-v1"


@dataclass(frozen=True)
class ContactConfig:
    r_enter_m: float = 1.0           # contact: the players come within one metre (about two body widths)
    r_exit_m: float = 1.5            # hysteresis: the episode lasts while they stay within 1.5 m
    link_pad_s: float = 0.5
    aligned_tau_s: float = 0.20
    cluster_gap_s: float = 5.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def contact_events(c, cfg: ContactConfig) -> list[dict]:
    events = []
    t = c.video_time_s
    for p in [int(p) for p in np.unique(c.canonical_period) if p > 0 and int(p) not in c.quarantine_periods]:
        rows = np.flatnonzero(c.canonical_period == p)
        X, Y = c.x[rows], c.y[rows]
        for a, b in itertools.combinations(range(c.n_slots), 2):
            d = np.hypot(X[:, a] - X[:, b], Y[:, a] - Y[:, b])
            inside = d <= cfg.r_exit_m                                   # NaN compares False: unobserved frames end a run
            if not (d <= cfg.r_enter_m).any():
                continue
            edges = np.flatnonzero(np.diff(np.r_[0, inside.astype(np.int8), 0]))
            for s, e in zip(edges[::2], edges[1::2]):
                seg = d[s:e]
                if not (seg <= cfg.r_enter_m).any():
                    continue
                k = s + int(np.nanargmin(seg))
                r = rows[k]
                ts = float(t[r])
                key = f"{c.match_id}|{p}|{a}|{b}|{int(c.record_first[r])}"
                events.append(dict(
                    event_id=hashlib.sha256(key.encode()).hexdigest()[:16], event_key=key, match_id=int(c.match_id), period=p,
                    slot_a=a, slot_b=b, side_a=str(c.slot_side[a]), side_b=str(c.slot_side[b]), jersey_a=int(c.slot_jersey[a]),
                    jersey_b=int(c.slot_jersey[b]), team_relation="same_team" if c.slot_side[a] == c.slot_side[b] else "opposite_teams",
                    type="contact", x=float((X[k, a] + X[k, b]) / 2), y=float((Y[k, a] + Y[k, b]) / 2), time_a_s=ts, time_b_s=ts,
                    time_diff_s=0.0, abs_time_diff_s=0.0, min_synchronous_distance_m=float(seg.min()),
                    episode_s=[float(t[rows[s]]), float(t[rows[e - 1]])], episode_frames=[int(c.frame_num[rows[s]]), int(c.frame_num[rows[e - 1]])],
                    n_windows=0, quality={}))
    events.sort(key=lambda e: (e["period"], e["time_a_s"], e["slot_a"], e["slot_b"]))
    assign_clusters(events, cfg.cluster_gap_s)
    return events


def link_geometry(events: list[dict], geo: list[dict], cfg: ContactConfig) -> None:
    by: dict[tuple, list[dict]] = {}
    for g in geo:
        by.setdefault((g["period"], *sorted((g["slot_a"], g["slot_b"]))), []).append(g)
    for e in events:
        lo, hi = e["episode_s"][0] - cfg.link_pad_s, e["episode_s"][1] + cfg.link_pad_s
        hits = [g for g in by.get((e["period"], e["slot_a"], e["slot_b"]), []) if lo <= g["time_a_s"] <= hi and lo <= g["time_b_s"] <= hi]
        e["geometry"] = dict(geometric_crossing=bool(hits), aligned_crossing=any(g["abs_time_diff_s"] <= cfg.aligned_tau_s for g in hits),
                             crossing_event_ids=[g["event_id"] for g in hits],
                             min_crossing_abs_time_diff_s=min((g["abs_time_diff_s"] for g in hits), default=None))


def run_match(raw_root: str, match_id: str, dry_run: bool, split_expected: str = "development", source_root=None,
              out_base: Path | None = None, cfg: ContactConfig = ContactConfig(), acfg: A3.AuditV3Config = A3.AuditV3Config()) -> dict[str, Any]:
    t0 = time.perf_counter()
    split = json.loads((approved_output_root(raw_root) / ml.SPLITS_DIR / "splits.json").read_text(encoding="utf-8"))
    validate.check_split(split, [str(match_id)], split_expected)
    base = out_base or approved_output_root(raw_root) / SCHEMA
    out = safe_output_path(raw_root, match_id, EventVersion.ROOT, base, suffix=".manifest.json").parent
    if not dry_run and out.exists():
        raise IngestionError(f"{out} already exists and is never overwritten")
    snap0 = validate.raw_snapshot(raw_root)
    c, mask, _enc, src_manifest, prov = load_source(raw_root, match_id, source_root)
    sbase = Path(source_root) if source_root is not None else approved_output_root(raw_root) / SOURCE_DIR
    with np.load(safe_output_path(raw_root, match_id, EventVersion.ROOT, sbase, suffix=".quality.npz")) as z:
        steps = {k[5:]: z[k] for k in z.files if k.startswith("step_")}
    geo, _ = detect(c, mask, steps, DetectConfig(), quarantine=tuple(c.quarantine_periods))
    events = contact_events(c, cfg)
    link_geometry(events, geo, cfg)
    cx = AU.Context(c, mask, acfg)
    recs = A3.audit_events(cx, events)
    for r, e in zip(recs, events):
        r["geometry"] = e["geometry"]
        r["min_same_time_distance_m"] = e["min_synchronous_distance_m"]
    store, refs = AU.build_store(cx, events, recs)
    ids = {x["event_id"] for x in refs}
    for r in recs:
        r["in_trajectory_store"] = r["event_id"] in ids
    summary = AU.summarize(recs, acfg)
    k = AU.tkey(acfg.primary_s)
    cand = [r for r in recs if r["status"][k]["category"] in ("clean_candidate", "uncertain_candidate")]
    summary |= dict(schema=SCHEMA, contacts=len(events), geometric_intersections=len(geo),
                    geometry=dict(all=dict(Counter((r["geometry"]["geometric_crossing"], r["geometry"]["aligned_crossing"]) for r in recs).items()),
                                  candidates=dict(Counter(f"crossing={r['geometry']['geometric_crossing']}|aligned={r['geometry']['aligned_crossing']}"
                                                          for r in cand))),
                    seconds=round(time.perf_counter() - t0, 1), peak_memory_mb=peak_memory_mb())
    summary["geometry"]["all"] = {f"crossing={a}|aligned={b}": n for (a, b), n in summary["geometry"]["all"].items()}
    if dry_run:
        return summary
    blobs = {"root.events.jsonl": jsonl(events), "root.eligibility.jsonl": jsonl(recs), "root.references.jsonl": jsonl(refs),
             "root.trajectories.npz": AU.npz_bytes(store), "root.summary.json": json.dumps(summary, indent=1, sort_keys=True, default=str).encode(),
             "root.config.json": json.dumps(dict(contacts=cfg.as_dict(), audit=acfg.as_dict()), indent=1, sort_keys=True).encode()}
    manifest = dict(schema=SCHEMA, match_id=int(match_id), split=split["matches"][str(match_id)], config=dict(contacts=cfg.as_dict(), audit=acfg.as_dict()),
                    code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))},
                    step1=prov, step1_schema=src_manifest["schema"], raw_unchanged=snap0 == validate.raw_snapshot(raw_root),
                    coordinate_policy="raw", outputs_sha256={n: hashlib.sha256(b).hexdigest() for n, b in blobs.items()},
                    python=platform.python_version(), numpy=np.__version__)
    blobs["root.manifest.json"] = json.dumps(manifest, indent=1, sort_keys=True).encode()
    write_dir(out, blobs)
    summary["written_to"] = str(out)
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m football_fatigue.intersections.contacts")
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--matches", nargs="+", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    root = str(Path(a.root).resolve())
    try:
        for mid in a.matches:
            s = run_match(root, mid, a.dry_run)
            b = s["by_threshold"][AU.tkey(A3.AuditV3Config().primary_s)]["categories"]
            print(mid, "contacts", s["contacts"], b, s["geometry"]["candidates"], s["seconds"], "s", flush=True)
    except IngestionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
