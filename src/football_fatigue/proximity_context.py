"""Event-centred 5 s trajectory contexts of the ``m11-sync-proximity-v2`` encounters: ``m11-sync-proximity-v3``.

Detection is **not** rerun or changed: every 0.25 m episode of v2 (its event ID, pair, t*, d_min, boundaries, threshold and
labels) is reused as it is, and v3 stores only the new contexts plus the line-for-line reference to the v2 record. Event IDs
are unchanged (v3 line k = v2 line k = the same ``event_id``).

**What changed.** v2 kept the context of the episode's *canonical window*: the advisor's window grid centres windows on
integer game seconds (period-relative), so the context was [c - 2.5, c + 2.5] with c the integer second nearest to t*, i.e.
t* -2.0 ... -3.0 s before and +2.0 ... +3.0 s after. It also exported only the segments lying entirely inside that window,
so up to one segment (<= 0.5 s + 1 frame) could be missing at each edge. v3 uses

    nominal window = [t* - 2.5 s, t* + 2.5 s]   (continuous t*, never rounded)

clipped only to the match period (``window_clipped_by_period``). Each player's context is the Step 1 valid run containing the
player's segment at t* (the same run as v2: no gap > 0.5 s, implausible jump, unattributed / ambiguous sample or period change
inside it), restricted to the window. Where a joined segment straddles a window edge, the position at the edge is the linear
interpolation *on that valid segment* (``sample_is_boundary``; source row -1; ESTIMATED when either end is ESTIMATED, flags of
both ends). Nothing is extrapolated, nothing is joined across a cut and no other run of the player is added. The parts of the
nominal window without valid support are listed per player (``unavailable``) with the Step 1 reason the run stops.

Run: ``python -m football_fatigue.proximity_context --root data/raw [--matches ...] [--jobs 3]`` (development matches only;
existing outputs are verified and never overwritten).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from football_fatigue.encounters import quality as Q
from football_fatigue.encounters.trajectories import VISIBILITY
from football_fatigue.ingestion import IngestionError, approved_output_root
from football_fatigue.intersections import exact as X
from football_fatigue.intersections import proximity as P
from football_fatigue.intersections.catalogue import load_steps
from football_fatigue.matching import ml, validate
from football_fatigue.matching.benchmark import jsonl, load_source

SOURCE = P.SCHEMA                    # m11-sync-proximity-v2 (events reused, never modified)
SCHEMA = "m11-sync-proximity-v3"
PRIMARY = "0.25m"
HALF_S = 2.5
TOL_S = 1e-9
CONFIG = dict(context="event_centred", half_width_s=HALF_S, clipped_to="match period", threshold=PRIMARY,
              trajectory="Step 1 valid run containing the segment at t* (intersections.exact.trajectories, unchanged)",
              window_edge="linear interpolation on the straddling valid segment; never extrapolated")


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def source_dir(raw_root: str) -> Path:
    return approved_output_root(raw_root) / SOURCE


def out_dir(raw_root: str) -> Path:
    return approved_output_root(raw_root) / SCHEMA


def development(raw_root: str) -> list[str]:
    split = json.loads((approved_output_root(raw_root) / ml.SPLITS_DIR / "splits.json").read_text(encoding="utf-8"))
    return [str(m) for m in split["splits"]["development"]]


# ---------------------------------------------------------------- runs and windows

def run_limits(seg: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """For every segment, the first and last segment index of its valid run (same slot, joined end to start)."""
    n = len(seg["slot"])
    joined = np.r_[False, (seg["slot"][1:] == seg["slot"][:-1]) & (seg["row0"][1:] == seg["row1"][:-1])]
    start = np.maximum.accumulate(np.where(~joined, np.arange(n), 0))
    nxt = np.r_[joined[1:], False]
    end_flags = ~nxt
    end = np.minimum.accumulate(np.where(end_flags, np.arange(n), n)[::-1])[::-1]
    return start, end


def clip_run(seg, c, mask, rs: int, re_: int, lo: float, hi: float) -> dict[str, np.ndarray]:
    """Vertices of run [rs, re_] inside [lo, hi] plus interpolated vertices at an edge that a segment straddles."""
    slot = int(seg["slot"][rs])
    t = np.r_[seg["t0"][rs:re_ + 1], seg["t1"][re_]]
    x = np.r_[seg["x0"][rs:re_ + 1], seg["x1"][re_]]
    y = np.r_[seg["y0"][rs:re_ + 1], seg["y1"][re_]]
    rows = np.r_[seg["row0"][rs:re_ + 1], seg["row1"][re_]]
    vis = c.visibility[rows, slot].astype(np.int8)
    bits = mask[rows, slot].astype(np.int64)
    k0, k1 = int(np.searchsorted(t, lo, "left")), int(np.searchsorted(t, hi, "right"))
    parts = [dict(t=t[k0:k1], x=x[k0:k1], y=y[k0:k1], row=rows[k0:k1], frame=c.frame_num[rows[k0:k1]], vis=vis[k0:k1], bits=bits[k0:k1],
                  edge=np.zeros(k1 - k0, bool))]

    def edge(te: float, i: int) -> dict[str, np.ndarray]:            # te strictly inside (t[i], t[i + 1])
        f = (te - t[i]) / (t[i + 1] - t[i])
        return dict(t=np.array([te]), x=np.array([x[i] + f * (x[i + 1] - x[i])]), y=np.array([y[i] + f * (y[i + 1] - y[i])]),
                    row=np.array([-1]), frame=np.array([-1]), vis=np.array([max(vis[i], vis[i + 1])], np.int8),
                    bits=np.array([bits[i] | bits[i + 1]]), edge=np.array([True]))

    if k0 > 0 and (k0 == len(t) or t[k0] > lo):                       # a segment straddles the start
        parts.insert(0, edge(lo, k0 - 1))
    if k1 < len(t) and (k1 == 0 or t[k1 - 1] < hi):                   # ... the end
        parts.append(edge(hi, k1 - 1))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def player_context(seg, c, mask, si, g: int, rs: int, re_: int, ts: float, ns: float, ne: float, lo: float, hi: float) -> tuple[dict, dict]:
    v = clip_run(seg, c, mask, rs, re_, lo, hi)
    slot = int(seg["slot"][g])
    run_t0, run_t1 = float(seg["t0"][rs]), float(seg["t1"][re_])
    cov0, cov1 = max(run_t0, lo), min(run_t1, hi)
    before, after = run_t0 <= ns + TOL_S, run_t1 >= ne - TOL_S

    def why(at_start: bool) -> str:
        if (run_t0 <= lo + TOL_S) if at_start else (run_t1 >= hi - TOL_S):
            return "period_boundary"                                     # the run reaches the period edge, which clips the window
        if at_start:
            return P._cut_reason(si, si["to"].get((slot, int(seg["row0"][rs]))))
        last = int(seg["row0"][re_]) if seg["stationary"][re_] else int(seg["row1"][re_])
        return P._cut_reason(si, si["frm"].get((slot, last)))

    una = []                                                             # [start, end, reason] inside the nominal window
    if lo > ns + TOL_S:
        una.append([ns, lo, "outside_period"])
    if cov0 > lo + TOL_S:
        una.append([lo, cov0, why(True)])
    if cov1 < hi - TOL_S:
        una.append([cov1, hi, why(False)])
    if hi < ne - TOL_S:
        una.append([hi, ne, "outside_period"])
    orig = ~v["edge"]
    bits = int(np.bitwise_or.reduce(v["bits"])) if len(v["bits"]) else 0
    info = dict(complete_before=bool(before), complete_after=bool(after), cut_before=None if before else why(True),
                cut_after=None if after else why(False), run_start_s=run_t0, run_end_s=run_t1, covered_start_s=cov0, covered_end_s=cov1,
                incoming_s=max(ts - cov0, 0.0), outgoing_s=max(cov1 - ts, 0.0), unavailable=una, n_samples=int(orig.sum()),
                n_window_edge_points=int(v["edge"].sum()),
                observed_share=round(float((v["vis"][orig] == VISIBILITY["VISIBLE"]).mean()), 6) if orig.any() else 0.0,
                context_flags=[n for n, b in Q.MASK_BITS.items() if bits & b],
                critical_flags=[n for n, b in Q.MASK_BITS.items() if bits & b & P.CRITICAL_BITS])
    return v, info


def event_context(seg, c, mask, si, starts, ends, key: dict, rec: dict) -> tuple[dict, dict]:
    ts = rec["t_star_s"]
    ns, ne = ts - HALF_S, ts + HALF_S
    lo, hi = max(ns, rec["period_start_s"]), min(ne, rec["period_end_s"])
    out, info = {}, {}
    for side in ("a", "b"):
        pl = rec[f"player_{side}"]
        s = pl["segment_at_t_star"]
        g = key.get((pl["slot"], s["row0"], s["row1"]))
        if g is None or not seg["t0"][g] - TOL_S <= ts <= seg["t1"][g] + TOL_S:
            raise IngestionError(f"{rec['event_id']}: the segment at t* of player {side} is not a valid segment")
        out[side], info[side] = player_context(seg, c, mask, si, g, int(starts[g]), int(ends[g]), ts, ns, ne, lo, hi)
        p = (np.interp(ts, out[side]["t"], out[side]["x"]), np.interp(ts, out[side]["t"], out[side]["y"]))
        if max(abs(p[0] - pl["position_at_t_star"][0]), abs(p[1] - pl["position_at_t_star"][1])) > 1e-6:
            raise IngestionError(f"{rec['event_id']}: the context does not reproduce {side}(t*)")
    a, b = out["a"], out["b"]
    s0, s1 = max(a["t"][0], b["t"][0]), min(a["t"][-1], b["t"][-1])
    tt = np.unique(np.r_[a["t"], b["t"], [rec["t_start_s"], ts, rec["t_end_s"]]])
    tt = tt[(tt >= s0) & (tt <= s1)]
    xa, ya = np.interp(tt, a["t"], a["x"]), np.interp(tt, a["t"], a["y"])
    xb, yb = np.interp(tt, b["t"], b["x"]), np.interp(tt, b["t"], b["y"])
    sync = np.c_[tt, xa, ya, xb, yb, np.hypot(xa - xb, ya - yb), np.isin(tt, a["t"][~a["edge"]]), np.isin(tt, b["t"][~b["edge"]])]
    ia, ib = info["a"], info["b"]
    cx = dict(definition="event_centred", half_width_s=HALF_S, nominal_start_s=ns, nominal_end_s=ne, window_start_s=lo, window_end_s=hi,
              window_clipped_by_period=bool(lo > ns or hi < ne), shared_start_s=float(s0), shared_end_s=float(s1),
              complete=all(d[k] for d in (ia, ib) for k in ("complete_before", "complete_after")),
              critical_flags=sorted(set(ia["critical_flags"]) | set(ib["critical_flags"])), player_a=ia, player_b=ib)
    return dict(a=a, b=b, sync=sync), cx


def store(trajs: list[dict], t_star: list[float]) -> dict[str, np.ndarray]:
    """v2 layout (``proximity.store``) plus ``sample_is_boundary`` (an interpolated window-edge point, not a provider sample)."""
    sm, sy = [], []
    for k, (tr, ts) in enumerate(zip(trajs, t_star)):
        for side in (0, 1):
            v = tr["ab"[side]]
            sm.append(np.c_[np.full(len(v["t"]), k), np.full(len(v["t"]), side), v["t"], v["x"], v["y"], v["row"], v["frame"], v["vis"],
                            v["bits"], v["t"] < ts, v["edge"]])
        sy.append(np.c_[np.full(len(tr["sync"]), k), tr["sync"]])
    S = np.concatenate(sm) if sm else np.zeros((0, 11))
    Y = np.concatenate(sy) if sy else np.zeros((0, 9))
    return dict(sample_event_index=S[:, 0].astype(np.int64), sample_side=S[:, 1].astype(np.int8), sample_t=S[:, 2], sample_x=S[:, 3],
                sample_y=S[:, 4], sample_source_row=S[:, 5].astype(np.int64), sample_frame=S[:, 6].astype(np.int64),
                sample_visibility=S[:, 7].astype(np.int8), sample_quality_bits=S[:, 8].astype(np.int64), sample_before_t_star=S[:, 9].astype(bool),
                sample_is_boundary=S[:, 10].astype(bool),
                sync_event_index=Y[:, 0].astype(np.int64), sync_t=Y[:, 1], sync_xa=Y[:, 2], sync_ya=Y[:, 3], sync_xb=Y[:, 4], sync_yb=Y[:, 5],
                sync_d=Y[:, 6], sync_a_is_sample=Y[:, 7].astype(bool), sync_b_is_sample=Y[:, 8].astype(bool))


# ---------------------------------------------------------------- per match

def source_files(raw_root: str, mid: str) -> dict[str, Path]:
    d = source_dir(raw_root) / mid
    return {"events": d / f"root.events-{PRIMARY}.jsonl", "manifest": d / "root.manifest.json"}


def render(raw_root: str, mid: str) -> tuple[dict[str, bytes], dict[str, Any]]:
    src = source_files(raw_root, mid)
    man = json.loads(src["manifest"].read_text(encoding="utf-8"))
    if man["outputs_sha256"][src["events"].name] != _sha(src["events"]):
        raise IngestionError(f"{src['events']} does not match its v2 manifest")
    recs = [json.loads(x) for x in src["events"].read_text(encoding="utf-8").splitlines()]
    c, mask, _enc, _m, prov = load_source(raw_root, mid)
    seg, _ = X.trajectories(c, mask, load_steps(raw_root, mid))
    si = P.step_index(load_steps(raw_root, mid))
    starts, ends = run_limits(seg)
    key = {(int(s), int(a), int(b)): g for g, (s, a, b) in enumerate(zip(seg["slot"], seg["row0"], seg["row1"]))}
    trajs, lines = [], []
    for k, rec in enumerate(recs):
        tr, cx = event_context(seg, c, mask, si, starts, ends, key, rec)
        trajs.append(tr)
        old = rec["context"]
        lines.append(dict(event_id=rec["event_id"], match_id=rec["match_id"], v2_line=k, t_star_s=rec["t_star_s"], context=cx,
                          v2_context=dict(window_start_s=old["window_start_s"], window_end_s=old["window_end_s"], complete=old["complete"])))
    blobs = {"root.contexts.jsonl": jsonl(lines), "root.trajectories.npz": X.AU.npz_bytes(store(trajs, [r["t_star_s"] for r in recs]))}
    summ = dict(schema=SCHEMA, match_id=int(mid), events=len(lines), complete=sum(x["context"]["complete"] for x in lines),
                v2_complete=sum(x["v2_context"]["complete"] for x in lines),
                clipped_by_period=sum(x["context"]["window_clipped_by_period"] for x in lines),
                window_edge_points=int(sum(x["context"][p]["n_window_edge_points"] for x in lines for p in ("player_a", "player_b"))))
    blobs["root.summary.json"] = json.dumps(summ, indent=1, sort_keys=True).encode()
    return blobs, dict(summary=summ, step1=prov, v2_manifest_sha256=_sha(src["manifest"]), v2_events_sha256=_sha(src["events"]))


def _code_sha256() -> dict[str, str]:
    files = [Path(__file__)] + sorted((Path(X.__file__).parent).glob("*.py"))
    return {p.name if p != Path(__file__) else "proximity_context.py": _sha(p) for p in files}


def verify_existing(out: Path) -> str | None:
    mp = out / "root.manifest.json"
    if not mp.exists():
        return "no manifest"
    m = json.loads(mp.read_text(encoding="utf-8"))
    if m.get("schema") != SCHEMA or m.get("config") != CONFIG:
        return "schema or configuration differs"
    for n, h in m.get("outputs_sha256", {}).items():
        if not (out / n).exists() or _sha(out / n) != h:
            return f"{n} missing or modified"
    return None


def run_match(raw_root: str, mid: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    split = json.loads((approved_output_root(raw_root) / ml.SPLITS_DIR / "splits.json").read_text(encoding="utf-8"))
    validate.check_split(split, [mid], "development")
    out = out_dir(raw_root) / mid
    if out.exists():
        why = verify_existing(out)
        if why is None:
            return dict(match_id=mid, status="skipped_complete")
        raise IngestionError(f"{out} exists but is not a verified complete output ({why}); it is never overwritten")
    snap = validate.raw_snapshot(raw_root)
    blobs, meta = render(raw_root, mid)
    manifest = dict(schema=SCHEMA, match_id=int(mid), split=split["matches"][mid], config=CONFIG, code_sha256=_code_sha256(),
                    source=dict(dataset=SOURCE, events_file=f"root.events-{PRIMARY}.jsonl", events_sha256=meta["v2_events_sha256"],
                                manifest_sha256=meta["v2_manifest_sha256"], step1=meta["step1"]),
                    event_reference="line k of root.contexts.jsonl / event index k of root.trajectories.npz = line k of the v2 "
                                    f"root.events-{PRIMARY}.jsonl; event_id unchanged",
                    raw_unchanged=snap == validate.raw_snapshot(raw_root), python=platform.python_version(), numpy=np.__version__,
                    outputs_sha256={n: hashlib.sha256(b).hexdigest() for n, b in blobs.items()},
                    note="Context re-extraction only; detection, thresholds and event records are those of v2. Proximity encounters "
                         "are tracking-derived identity-ambiguity proxies, not verified occlusions; slots are provider labels.")
    blobs["root.manifest.json"] = json.dumps(manifest, indent=1, sort_keys=True).encode()
    P.write_retry(out, blobs)
    return dict(match_id=mid, status="written", seconds=round(time.perf_counter() - t0, 1),
                **{k: v for k, v in meta["summary"].items() if k not in ("match_id", "schema")})


def _one(a: tuple[str, str]) -> dict[str, Any]:
    try:
        return run_match(*a)
    except Exception as e:                                               # noqa: BLE001 - reported per match
        return dict(match_id=a[1], status="failed", error=f"{type(e).__name__}: {e}")


def write_index(raw_root: str, matches: list[str]) -> dict[str, Any]:
    """Development index: the v2 -> v3 event mapping (identity of IDs, line numbers, old / new windows) and a summary."""
    rows, per = [], {}
    for mid in matches:
        d = out_dir(raw_root) / mid
        lines = [json.loads(x) for x in (d / "root.contexts.jsonl").read_text(encoding="utf-8").splitlines()]
        for k, x in enumerate(lines):
            cx, old = x["context"], x["v2_context"]
            rows.append(dict(match_id=mid, event_id=x["event_id"], v2_line=x["v2_line"], v3_line=k, t_star_s=x["t_star_s"],
                             v2_window_start_rel_s=round(old["window_start_s"] - x["t_star_s"], 6),
                             v2_window_end_rel_s=round(old["window_end_s"] - x["t_star_s"], 6), v2_complete=old["complete"],
                             v3_window_start_rel_s=round(cx["window_start_s"] - x["t_star_s"], 6),
                             v3_window_end_rel_s=round(cx["window_end_s"] - x["t_star_s"], 6), v3_complete=cx["complete"],
                             v3_clipped_by_period=cx["window_clipped_by_period"]))
        per[mid] = json.loads((d / "root.summary.json").read_text(encoding="utf-8"))
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    n = len(rows)
    both = lambda a, b: sum(r["v2_complete"] == a and r["v3_complete"] == b for r in rows)  # noqa: E731
    summ = dict(schema=SCHEMA, matches=len(matches), events=n, v2_complete=sum(r["v2_complete"] for r in rows),
                v3_complete=sum(r["v3_complete"] for r in rows), complete_in_both=both(True, True), complete_v2_only=both(True, False),
                complete_v3_only=both(False, True), incomplete_in_both=both(False, False),
                clipped_by_period=sum(r["v3_clipped_by_period"] for r in rows),
                v2_window_rel_s=dict(start=[min(r["v2_window_start_rel_s"] for r in rows), max(r["v2_window_start_rel_s"] for r in rows)],
                                     end=[min(r["v2_window_end_rel_s"] for r in rows), max(r["v2_window_end_rel_s"] for r in rows)]),
                v3_window_rel_s=dict(start=[min(r["v3_window_start_rel_s"] for r in rows), max(r["v3_window_start_rel_s"] for r in rows)],
                                     end=[min(r["v3_window_end_rel_s"] for r in rows), max(r["v3_window_end_rel_s"] for r in rows)]),
                per_match={m: dict(events=s["events"], v3_complete=s["complete"], v2_complete=s["v2_complete"]) for m, s in per.items()})
    files = {"event_mapping.csv": buf.getvalue().encode(), "summary.json": json.dumps(summ, indent=1, sort_keys=True).encode()}
    idx = out_dir(raw_root) / "development-index"
    idx.mkdir(parents=True, exist_ok=True)
    for name, b in files.items():
        (idx / name).write_bytes(b)
    (idx / "manifest.json").write_text(json.dumps(dict(
        schema=SCHEMA, index="development-index", matches=matches, config=CONFIG,
        match_manifests_sha256={m: _sha(out_dir(raw_root) / m / "root.manifest.json") for m in matches},
        source_index_manifest_sha256=_sha(source_dir(raw_root) / "development-index" / "manifest.json"),
        outputs_sha256={k: hashlib.sha256(b).hexdigest() for k, b in files.items()}), indent=1), encoding="utf-8")
    return summ


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--matches", nargs="*")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args(argv)
    root = str(Path(a.root).resolve())
    ms = a.matches or development(root)
    failed = 0
    with ProcessPoolExecutor(max(1, a.jobs)) as ex:
        for r in ex.map(_one, [(root, m) for m in ms]):
            failed += r["status"] == "failed"
            print(json.dumps(r), flush=True)
    if not failed and not a.matches:
        print(json.dumps({k: v for k, v in write_index(root, ms).items() if k != "per_match"}, indent=1))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
