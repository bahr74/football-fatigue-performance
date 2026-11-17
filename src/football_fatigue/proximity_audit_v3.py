"""Independent audit of the event-centred contexts ``m11-sync-proximity-v3``: ``m11-sync-proximity-v3-audit``.

Read-only (development matches only). The v2 event records are unchanged and reused; every 0.25 m episode is checked again on
its new context with the geometry of :mod:`football_fatigue.proximity_audit` (no code shared with the builder), plus:

* **window**: nominal [t* - 2.5, t* + 2.5] exactly, clipped only to the period (period limits recomputed from the RAW cache);
  every exported vertex inside the window; interpolated points only at the window edges;
* **edge points**: re-derived from the RAW cache with the documented representation (update instants; a held run of >= 3
  frames adds a vertex one frame before its end) and checked to lie on a valid joined step (<= 0.5 s unobserved, no jump);
* **cuts**: where a track stops inside the window, the step beyond it is re-derived from the cache and must be unjoinable
  (gap / period, implausible jump, unattributed or ambiguous sample, or no further sample) - otherwise valid support was lost;
* **coverage**: completeness flags, unavailable intervals and incoming / outgoing seconds against the vertices.

Presentation readiness (configurable; recorded in the manifest): mathematically valid, tracking support valid, both players'
incoming and outgoing tracks cover the nominal window, no suspected provider artifact, both observed at t*, at least
``--min-observed-share`` of each player's original samples observed, each player moving at least ``--min-path-m``.

Run: ``python -m football_fatigue.proximity_audit_v3 --root data/raw [--matches ...] [--jobs 3]``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from football_fatigue import proximity_audit as A
from football_fatigue.encounters.catalogue import OUTPUT_DIR as STEP1_DIR
from football_fatigue.encounters.trajectories import STATE_AVAILABLE
from football_fatigue.ingestion import approved_output_root

SOURCE, CONTEXTS, SCHEMA = A.SOURCE, "m11-sync-proximity-v3", "m11-sync-proximity-v3-audit"
OLD_AUDIT = A.SCHEMA
HALF_S = 2.5
TOL_W = 1e-9                     # window arithmetic (float seconds)
UNUSABLE_BITS = 256 | 4096 | 1024  # Step 1 ATTRIBUTION_UNRESOLVED | AMBIGUOUS_RAW | SIDE_OVERCOUNT (never joined)
RULES = dict(min_observed_share=A.MIN_OBSERVED_SHARE, min_path_m=A.MIN_PATH_M)

REASONS = dict(A.REASONS)
REASONS.update({
    "window_not_centred": "the nominal window is not [t* - 2.5 s, t* + 2.5 s] clipped to the period",
    "vertex_outside_window": "an exported vertex lies outside the context window",
    "edge_point_invalid": "a window-edge point is not the interpolation on a valid joined step of the RAW samples",
    "cut_not_justified": "a track stops inside the window although the next step is joinable (valid support lost)",
    "cut_reason_mismatch": "the recorded reason a track stops differs from the re-derived reason",
    "coverage_mismatch": "completeness flags, unavailable intervals or covered seconds disagree with the vertices",
    "incoming_incomplete_a": "Player A's valid track does not cover t* - 2.5 s ... t*",
    "incoming_incomplete_b": "Player B's valid track does not cover t* - 2.5 s ... t*",
    "outgoing_incomplete_a": "Player A's valid track does not cover t* ... t* + 2.5 s",
    "outgoing_incomplete_b": "Player B's valid track does not cover t* ... t* + 2.5 s",
    "period_boundary": "the 5 s window extends beyond the match period",
})


# ---------------------------------------------------------------- RAW re-derivation around a window edge

def local_vertices(cache: dict, slot: int, period: int, t_lo: float, t_hi: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Piecewise-linear vertices of one slot from the RAW cache on [t_lo, t_hi] (update instants; a held run of >= 3 frames adds
    a vertex one frame before its end): times, x, y, cache rows."""
    t = cache["video_time_s"]
    r = np.flatnonzero((t >= t_lo) & (t <= t_hi) & (cache["canonical_period"] == period) & (cache["state"][:, slot] == STATE_AVAILABLE))
    if not len(r):
        return (np.zeros(0),) * 3 + (np.zeros(0, np.int64),)
    x, y, f = cache["x"][r, slot], cache["y"][r, slot], cache["frame_num"][r]
    rep = np.r_[False, (np.diff(f) == 1) & (x[1:] == x[:-1]) & (y[1:] == y[:-1])]
    starts = np.flatnonzero(~rep)
    ends = np.r_[starts[1:] - 1, len(r) - 1]
    vi = []
    for s, e in zip(starts, ends):
        vi.append(s)
        if e - s >= 2:
            vi.append(e - 1)
    vi = np.array(vi)
    return t[r[vi]], x[vi], y[vi], r[vi]


def step_ok(cache: dict, mask: np.ndarray, slot: int, r0: int, r1: int) -> str | None:
    """None when the step between vertex rows r0 < r1 is joinable under the Step 1 rules; otherwise why not."""
    t = cache["video_time_s"]
    if cache["canonical_period"][r0] != cache["canonical_period"][r1]:
        return "gap_period_or_time"
    av = np.flatnonzero(cache["state"][r0:r1 + 1, slot] == STATE_AVAILABLE) + r0
    if np.diff(t[av]).max(initial=0) > A.MAX_GAP_S + 1e-9:
        return "gap_period_or_time"
    # the jump rule on the update-instant step (the run's first sample to the next update): shortest change interval
    run0 = r0
    while run0 > 0 and cache["state"][run0 - 1, slot] == STATE_AVAILABLE and cache["frame_num"][run0] - cache["frame_num"][run0 - 1] == 1 \
            and cache["x"][run0 - 1, slot] == cache["x"][r0, slot] and cache["y"][run0 - 1, slot] == cache["y"][r0, slot]:
        run0 -= 1
    d = float(np.hypot(cache["x"][r1, slot] - cache["x"][r0, slot], cache["y"][r1, slot] - cache["y"][r0, slot]))
    change = min(t[r1] - t[run0], t[r1] - t[av[-2]] + A.FRAME_S) if len(av) > 1 else t[r1] - t[run0]
    if d > A.JUMP_MIN_M and d / change > A.JUMP_MPS:
        return "implausible_jump"
    if (int(mask[run0, slot]) | int(mask[r1, slot])) & UNUSABLE_BITS:
        return "unattributed_or_ambiguous"
    return None


def edge_and_cut_checks(s: dict, info: dict, slot: int, period: int, cache: dict, mask: np.ndarray, w0: float, w1: float,
                        p_lo: float, p_hi: float) -> list[str]:
    out = []
    t, e = s["t"], s["is_boundary"]
    if e.any():
        for i in np.flatnonzero(e):
            if not (abs(t[i] - w0) <= TOL_W or abs(t[i] - w1) <= TOL_W) or (i not in (0, len(t) - 1)):
                out.append("edge_point_invalid")
                continue
            vt, vx, vy, vr = local_vertices(cache, slot, period, t[i] - 1.5, t[i] + 1.5)
            k = int(np.searchsorted(vt, t[i]))
            if not 0 < k < len(vt) or abs(np.interp(t[i], vt, vx) - s["x"][i]) > 1e-6 or abs(np.interp(t[i], vt, vy) - s["y"][i]) > 1e-6:
                out.append("edge_point_invalid")
            elif step_ok(cache, mask, slot, int(vr[k - 1]), int(vr[k])) is not None:
                out.append("edge_point_invalid")
    for end, key, inside in ((0, "cut_before", t[0] > w0 + TOL_W), (-1, "cut_after", t[-1] < w1 - TOL_W)):
        if not inside:
            continue
        if e[end]:
            out.append("edge_point_invalid")
            continue
        r = int(s["source_row"][end])
        vt, vx, vy, vr = local_vertices(cache, slot, period, t[end] - 3.0, t[end] + 3.0)
        k = int(np.flatnonzero(vr == r)[0]) if (vr == r).any() else -1
        if k < 0:
            out.append("cut_not_justified")
            continue
        nb = k - 1 if end == 0 else k + 1
        if not 0 <= nb < len(vt):
            # no further sample in the period: the player is not tracked (or the period ends) there
            why = "no_adjacent_sample"
            if (end == 0 and abs(t[0] - p_lo) <= 1e-6) or (end == -1 and abs(t[-1] - p_hi) <= 1e-6):
                why = "period_edge"
        else:
            r0, r1 = (int(vr[nb]), r) if end == 0 else (r, int(vr[nb]))
            why = step_ok(cache, mask, slot, r0, r1)
            if why is None:
                out.append("cut_not_justified")
                continue
        rec = info.get(key)
        if why == "period_edge":
            if rec not in ("gap_period_or_time", "no_adjacent_sample", "period_boundary"):
                out.append("cut_reason_mismatch")
        elif rec != why and not (why == "no_adjacent_sample" and rec in ("gap_period_or_time", "no_adjacent_sample")):
            out.append("cut_reason_mismatch")
    return out


# ---------------------------------------------------------------- per event

def sides(npz: dict[str, np.ndarray], k: int, side: int) -> dict[str, np.ndarray]:
    w = np.flatnonzero((npz["sample_event_index"] == k) & (npz["sample_side"] == side))
    w = w[np.argsort(npz["sample_t"][w], kind="stable")]
    return {f: npz[f"sample_{f}"][w] for f in ("t", "x", "y", "source_row", "frame", "visibility", "before_t_star", "is_boundary")}


def audit_event(rec: dict, cx: dict, a: dict, b: dict, sync: dict, cache: dict | None, mask: np.ndarray | None,
                period_limits: tuple[float, float] | None, rules: dict = RULES) -> dict[str, Any]:
    R = rec["threshold_m"]
    ts, t0, t1 = rec["t_star_s"], rec["t_start_s"], rec["t_end_s"]
    tracks = (a["t"], a["x"], a["y"], b["t"], b["x"], b["y"])
    g = A.grid(*tracks, extra=(t0, t1))
    lo, hi = A.support(a["t"], b["t"])
    math_r, track_r, cov_r = [], [], []

    # detection (as in the v2 audit, on the new context)
    d_min, t_ind = A.closest(g, t0, t1)
    if abs(d_min - rec["d_min_m"]) > A.TOL_POS_M:
        math_r.append("d_min_mismatch")
    if not lo - A.TOL_T_S <= ts <= hi + A.TOL_T_S or not t0 - A.TOL_T_S <= ts <= t1 + A.TOL_T_S:
        math_r.append("t_star_outside_support")
    elif A.d_at(*tracks, ts) > d_min + A.TOL_POS_M or (t_ind < ts - 1e-5 and A.d_at(*tracks, t_ind) < rec["d_min_m"] - A.TOL_POS_M):
        math_r.append("t_star_mismatch")
    pa = (float(np.interp(ts, a["t"], a["x"])), float(np.interp(ts, a["t"], a["y"])))
    pb = (float(np.interp(ts, b["t"], b["x"])), float(np.interp(ts, b["t"], b["y"])))
    ea, eb = rec["player_a"]["position_at_t_star"], rec["player_b"]["position_at_t_star"]
    if max(abs(pa[0] - ea[0]), abs(pa[1] - ea[1]), abs(pb[0] - eb[0]), abs(pb[1] - eb[1])) > A.TOL_POS_M:
        math_r.append("position_mismatch")
    if max(abs(rec["midpoint"][0] - (ea[0] + eb[0]) / 2), abs(rec["midpoint"][1] - (ea[1] + eb[1]) / 2)) > 1e-9:
        math_r.append("midpoint_mismatch")
    if rec["d_min_m"] > R + 1e-12:
        math_r.append("threshold_violated")
    il, ih = max(t0, lo), min(t1, hi)
    if il <= ih and A.max_d_on(g, il, ih, tracks) > R + A.TOL_POS_M:
        math_r.append("episode_exceeds_r")

    def boundary(t: float, start: bool) -> str:
        edge = lo if start else hi
        if (t > edge + A.TOL_T_S) if start else (t < edge - A.TOL_T_S):
            return "verified_d_eq_R" if abs(A.d_at(*tracks, t) - R) <= A.TOL_BOUND_M else "mismatch"
        end = 0 if start else -1
        lim = [(key, s) for key, s in (("player_a", a), ("player_b", b)) if abs(s["t"][end] - edge) <= A.TOL_T_S]
        at_window = all(abs(s["t"][end] - (cx["window_start_s"] if start else cx["window_end_s"])) <= TOL_W for _, s in lim)
        if at_window:
            return "beyond_window"                                       # the episode continues outside the 5 s context
        if rec["support_cut_at_start" if start else "support_cut_at_end"]:
            return "support_cut"
        return "mismatch"

    bs, be = boundary(t0, True), boundary(t1, False)
    if bs == "mismatch":
        math_r.append("start_boundary_mismatch")
    if be == "mismatch":
        math_r.append("end_boundary_mismatch")
    if abs(rec["duration_s"] - (t1 - t0)) > 1e-9 or (rec["contact_class"] == "point") != (t1 == t0):
        math_r.append("duration_mismatch")
    ident_ep, ident_ctx = A.identical_seconds(g, t0, t1), A.identical_seconds(g)
    zero_bad = (rec["d_min_m"] == 0) != rec["exact_coincidence"] or (rec["exact_coincidence"] and d_min > A.TOL_POS_M)
    covered = t0 >= lo - A.TOL_T_S and t1 <= hi + A.TOL_T_S
    if rec["coincident_interval"] and ident_ep == 0 and covered or not rec["coincident_interval"] and ident_ep > 0:
        zero_bad = True
    if zero_bad:
        math_r.append("zero_label_mismatch")

    # window
    ns, ne = ts - HALF_S, ts + HALF_S
    if period_limits is not None:
        p_lo, p_hi = period_limits
    else:
        p_lo, p_hi = rec["period_start_s"], rec["period_end_s"]
    w0, w1 = max(ns, p_lo), min(ne, p_hi)
    if (abs(cx["nominal_start_s"] - ns) > TOL_W or abs(cx["nominal_end_s"] - ne) > TOL_W or abs(cx["window_start_s"] - w0) > 1e-6
            or abs(cx["window_end_s"] - w1) > 1e-6 or cx["window_clipped_by_period"] != (w0 > ns + TOL_W or w1 < ne - TOL_W)):
        track_r.append("window_not_centred")
    for s in (a, b):
        if s["t"][0] < cx["window_start_s"] - TOL_W or s["t"][-1] > cx["window_end_s"] + TOL_W:
            track_r.append("vertex_outside_window")

    # tracking support: original vertices against the cache; edge points and cuts re-derived
    for key, s in (("player_a", a), ("player_b", b)):
        pl = rec[key]
        if len(s["t"]) > 1 and not (np.diff(s["t"]) > 0).all():
            track_r.append("time_order")
        if not np.array_equal(s["before_t_star"].astype(bool), s["t"] < ts):
            track_r.append("before_flag_mismatch")
        if cache is None:
            continue
        sl = pl["slot"]
        if (str(cache["slot_side"][sl]) != pl["side"] or int(cache["slot_jersey"][sl]) != pl["jersey"]
                or int(cache["slot_player_id"][sl]) != pl["player_id"] or int(cache["slot_team_id"][sl]) != pl["team_id"]):
            track_r.append("slot_identity_mismatch")
        o = ~s["is_boundary"]
        orig = {f: v[o] for f, v in s.items()}
        track_r += A.source_problems(orig, sl, rec["period"], cache)
        track_r += edge_and_cut_checks(s, cx[key], sl, rec["period"], cache, mask, cx["window_start_s"], cx["window_end_s"], p_lo, p_hi)
    gx = A.grid(*tracks, extra=[t for t in (t0, ts, t1) if lo <= t <= hi])
    if (len(gx["t"]) != len(sync["t"]) or not np.allclose(gx["t"], sync["t"], atol=1e-9, rtol=0)
            or not np.allclose(gx["d"], sync["d"], atol=1e-9, rtol=0)
            or not np.array_equal(sync["a_is_sample"].astype(bool), np.isin(sync["t"], a["t"][~a["is_boundary"]]))
            or not np.array_equal(sync["b_is_sample"].astype(bool), np.isin(sync["t"], b["t"][~b["is_boundary"]]))):
        track_r.append("sync_grid_mismatch")

    # coverage against the nominal window
    obs, plen, cov = {}, {}, {}
    for p, key, s in (("a", "player_a", a), ("b", "player_b", b)):
        c = cx[key]
        before, after = s["t"][0] <= ns + TOL_W, s["t"][-1] >= ne - TOL_W
        una = [(ns, cx["window_start_s"])] if cx["window_start_s"] > ns + TOL_W else []
        una += [(cx["window_start_s"], s["t"][0])] if s["t"][0] > cx["window_start_s"] + TOL_W else []
        una += [(s["t"][-1], cx["window_end_s"])] if s["t"][-1] < cx["window_end_s"] - TOL_W else []
        una += [(cx["window_end_s"], ne)] if cx["window_end_s"] < ne - TOL_W else []
        rec_una = [(u[0], u[1]) for u in c["unavailable"]]
        if (c["complete_before"] != before or c["complete_after"] != after or len(rec_una) != len(una)
                or any(abs(x0 - y0) > 1e-6 or abs(x1 - y1) > 1e-6 for (x0, x1), (y0, y1) in zip(rec_una, una))
                or abs(c["incoming_s"] - max(ts - s["t"][0], 0)) > 1e-6 or abs(c["outgoing_s"] - max(s["t"][-1] - ts, 0)) > 1e-6):
            track_r.append("coverage_mismatch")
        if not before:
            cov_r.append(f"incoming_incomplete_{p}")
        if not after:
            cov_r.append(f"outgoing_incomplete_{p}")
        o = ~s["is_boundary"]
        obs[p] = float((s["visibility"][o] == 1).mean()) if o.any() else 0.0
        plen[p] = A.path_length(s["x"], s["y"])
        cov[f"{p}_pre_s"], cov[f"{p}_post_s"] = ts - s["t"][0], s["t"][-1] - ts
    if cx["window_clipped_by_period"]:
        cov_r.append("period_boundary")
    if cx["complete"] != (not any(r.startswith(("incoming", "outgoing")) for r in cov_r)):
        track_r.append("coverage_mismatch")
    track_r = sorted(set(track_r))

    art_r = []
    if ident_ctx > 0 or rec["coincident_interval"]:
        art_r.append("identical_positions")
    if rec["player_a"]["duplicate_provider_position"] or rec["player_b"]["duplicate_provider_position"]:
        art_r.append("duplicate_provider_position")
    art_r += [f"flag_{f.lower()}" for f in A.ARTIFACT_FLAGS if f in cx["critical_flags"]]
    ev_r = []
    if rec["pair_quality"] != "observed/observed":
        ev_r.append("estimated_at_t_star")
    if min(obs.values()) < rules["min_observed_share"]:
        ev_r.append("low_observed_share")
    if min(plen.values()) < rules["min_path_m"]:
        ev_r.append("little_motion")
    cross = A.spatial_crossings(*tracks)
    st = dict(mathematically_valid=not math_r, tracking_support_valid=not track_r,
              incoming_complete=not any(r.startswith("incoming") for r in cov_r), outgoing_complete=not any(r.startswith("outgoing") for r in cov_r),
              context_complete=bool(cx["complete"]), provider_artifact_suspected=bool(art_r))
    st["presentation_ready"] = bool(st["mathematically_valid"] and st["tracking_support_valid"] and st["incoming_complete"]
                                    and st["outgoing_complete"] and not art_r and not ev_r)
    return dict(
        match_id=str(rec["match_id"]), event_id=rec["event_id"], period=rec["period"], t_star_s=ts, pair_quality=rec["pair_quality"],
        team_relation=rec["team_relation"], d_min_m=rec["d_min_m"], d_min_recomputed_m=d_min, t_star_recomputed_s=t_ind,
        exact_coincidence=rec["exact_coincidence"], coincident_interval=rec["coincident_interval"],
        encounter_type=("potential_tracking_ambiguity" if "identical_positions" in art_r else
                        "exact_synchronized_coincidence" if rec["d_min_m"] == 0 else "closest_synchronized_approach"),
        **st, window_start_rel_s=round(cx["window_start_s"] - ts, 6), window_end_rel_s=round(cx["window_end_s"] - ts, 6),
        window_clipped_by_period=cx["window_clipped_by_period"], requested_pre_s=round(ts - cx["window_start_s"], 6),
        requested_post_s=round(cx["window_end_s"] - ts, 6), **{k: round(float(v), 6) for k, v in cov.items()}, shared_pre_s=round(ts - lo, 6), shared_post_s=round(hi - ts, 6),
        identical_seconds_episode=round(ident_ep, 6), identical_seconds_context=round(ident_ctx, 6), start_boundary=bs, end_boundary=be,
        a_observed_share=round(obs["a"], 4), b_observed_share=round(obs["b"], 4), a_path_m=round(plen["a"], 3), b_path_m=round(plen["b"], 3),
        spatial_path_crossings=len(cross), simultaneous_crossings=sum(abs(c[2] - c[3]) <= 1e-6 for c in cross),
        reasons=";".join(math_r + track_r + cov_r + art_r + ev_r))


# ---------------------------------------------------------------- per match / dataset

def audit_match(raw_root: str, mid: str, rules: dict = RULES) -> dict[str, Any]:
    if mid not in A.development(raw_root):
        raise RuntimeError(f"{mid} is not a development match")
    v2, v3 = A.base(raw_root) / mid, approved_output_root(raw_root) / CONTEXTS / mid
    ev_path = v2 / f"root.events-{A.PRIMARY}.jsonl"
    recs = [json.loads(x) for x in ev_path.read_text(encoding="utf-8").splitlines()]
    ctx = [json.loads(x) for x in (v3 / "root.contexts.jsonl").read_text(encoding="utf-8").splitlines()]
    man3 = json.loads((v3 / "root.manifest.json").read_text(encoding="utf-8"))
    man2 = json.loads((v2 / "root.manifest.json").read_text(encoding="utf-8"))
    checks = dict(v3_manifest_hashes=all(A._sha(v3 / n) == h for n, h in man3["outputs_sha256"].items()),
                  v2_events_unchanged=A._sha(ev_path) == man2["outputs_sha256"][ev_path.name] == man3["source"]["events_sha256"],
                  v2_manifest_unchanged=A._sha(v2 / "root.manifest.json") == man3["source"]["manifest_sha256"],
                  same_events_same_order=[x["event_id"] for x in ctx] == [r["event_id"] for r in recs]
                  and all(x["v2_line"] == k for k, x in enumerate(ctx)),
                  t_star_unchanged=all(x["t_star_s"] == r["t_star_s"] for x, r in zip(ctx, recs)))
    with np.load(v3 / "root.trajectories.npz") as z:
        npz = {k: z[k] for k in z.files}
    checks["npz_rows"] = int(npz["sample_event_index"].max()) + 1 == len(recs) == int(npz["sync_event_index"].max()) + 1
    cache = A.load_source_cache(raw_root, mid)
    d1 = approved_output_root(raw_root) / STEP1_DIR / mid
    m1 = json.loads((d1 / "root.manifest.json").read_text(encoding="utf-8"))
    if m1["outputs_sha256"]["root.quality.npz"] != A._sha(d1 / "root.quality.npz"):
        raise RuntimeError(f"{d1 / 'root.quality.npz'} does not match its Step 1 manifest")
    with np.load(d1 / "root.quality.npz") as z:
        mask = z["mask"]
    limits = {}
    for p in np.unique(cache["canonical_period"]):
        if p > 0:
            tp = cache["video_time_s"][cache["canonical_period"] == p]
            limits[int(p)] = (float(tp.min()), float(tp.max()))
    rows = []
    for k, (rec, x) in enumerate(zip(recs, ctx)):
        a, b = sides(npz, k, 0), sides(npz, k, 1)
        w = np.flatnonzero(npz["sync_event_index"] == k)
        w = w[np.argsort(npz["sync_t"][w], kind="stable")]
        sync = {f: npz[f"sync_{f}"][w] for f in ("t", "d", "a_is_sample", "b_is_sample")}
        rows.append(audit_event(rec, x["context"], a, b, sync, cache, mask, limits[rec["period"]], rules))
    return dict(match_id=mid, rows=rows, checks=checks)


def _one(args) -> dict[str, Any]:
    return audit_match(*args)


def old_rows(raw_root: str) -> dict[tuple[str, str], dict]:
    p = approved_output_root(raw_root) / OLD_AUDIT / "event_validation.csv"
    with open(p, encoding="utf-8") as f:
        return {(r["match_id"], r["event_id"]): r for r in csv.DictReader(f)}


def compare(rows: list[dict], old: dict[tuple[str, str], dict]) -> dict[str, Any]:
    """v2 (integer-second window) vs v3 (event-centred) audit statuses, event by event."""
    out: dict[str, Any] = {}
    for f in A.STATUS_FIELDS:
        tr = Counter((old[(r["match_id"], r["event_id"])][f] == "True", bool(r[f])) for r in rows)
        out[f] = dict(v2=sum(n for (o, _), n in tr.items() if o), v3=sum(n for (_, v), n in tr.items() if v),
                      kept=tr[(True, True)], gained=tr[(False, True)], lost=tr[(True, False)], neither=tr[(False, False)])
    gained = [r for r in rows if r["presentation_ready"] and old[(r["match_id"], r["event_id"])]["presentation_ready"] != "True"]
    lost = [r for r in rows if not r["presentation_ready"] and old[(r["match_id"], r["event_id"])]["presentation_ready"] == "True"]
    out["ready_gained_v2_reasons"] = dict(Counter(x for r in gained for x in old[(r["match_id"], r["event_id"])]["reasons"].split(";") if x))
    out["ready_lost_v3_reasons"] = dict(Counter(x for r in lost for x in r["reasons"].split(";") if x))
    return out


def run(raw_root: str, matches: list[str], jobs: int, rules: dict = RULES) -> dict[str, Any]:
    bad = [m for m in matches if m not in A.development(raw_root)]
    if bad:
        raise SystemExit(f"not development matches: {bad}")
    with ProcessPoolExecutor(max(1, jobs)) as ex:
        res = list(ex.map(_one, [(raw_root, m, rules) for m in matches]))
    rows = [r for m in res for r in m["rows"]]
    ds: dict[str, Any] = dict(per_match={m["match_id"]: m["checks"] for m in res})
    ds["all_match_checks_pass"] = all(v is True for m in res for v in m["checks"].values())
    ds["global_unique_ids"] = len({(r["match_id"], r["event_id"]) for r in rows}) == len(rows)
    idx = approved_output_root(raw_root) / CONTEXTS / "development-index"
    if sorted(matches) == sorted(A.development(raw_root)) and (idx / "event_mapping.csv").exists():
        with open(idx / "event_mapping.csv", encoding="utf-8") as f:
            mp = list(csv.DictReader(f))
        ds["mapping_rows_equal_events"] = len(mp) == len(rows)
        ds["mapping_keys_equal_events"] = [(r["match_id"], r["event_id"]) for r in mp] == [(r["match_id"], r["event_id"]) for r in rows]
        ds["mapping_lines_identity"] = all(r["v2_line"] == r["v3_line"] for r in mp)
        im = json.loads((idx / "manifest.json").read_text(encoding="utf-8"))
        ds["index_hashes"] = all(A._sha(idx / n) == h for n, h in im["outputs_sha256"].items())
        v2sum = json.loads((A.base(raw_root) / "development-index" / "summary.json").read_text(encoding="utf-8"))
        ds["events_equal_v2_summary"] = len(rows) == v2sum["episodes"]
        with open(A.base(raw_root) / "development-index" / "subset_high_quality_candidates.csv", encoding="utf-8") as f:
            hq = {(r["match_id"], r["event_id"]) for r in csv.DictReader(f)}
        ready = {(r["match_id"], r["event_id"]) for r in rows if r["presentation_ready"]}
        with open(A.base(raw_root) / "development-index" / "encounters_index.csv", encoding="utf-8") as f:
            v2_complete = {(r["match_id"], r["event_id"]): r["context_complete"] == "True" for r in csv.DictReader(f)}
        extra = ready - hq                                               # v2's high-quality rule requires its own (v2) context complete
        ds["presentation_ready_not_in_v2_high_quality"] = len(extra)
        ds["presentation_ready_not_in_v2_high_quality_explained_by_v2_context"] = all(not v2_complete[k] for k in extra)
        ds["high_quality_not_ready"] = dict(Counter(x for r in rows if (r["match_id"], r["event_id"]) in hq and not r["presentation_ready"]
                                                    for x in r["reasons"].split(";") if x))
    summary = A.summarize(rows)
    summary["window_clipped_by_period"] = sum(bool(r["window_clipped_by_period"]) for r in rows)
    old = old_rows(raw_root)
    comparison = compare(rows, old) if all((r["match_id"], r["event_id"]) in old for r in rows) else None
    per_match = []
    for m in res:
        s = A.summarize(m["rows"])
        per_match.append(dict(match_id=m["match_id"], **{k: s[k] for k in ("encounters", *A.STATUS_FIELDS)}))
    return dict(rows=rows, summary=summary, per_match=per_match, dataset_checks=ds, comparison=comparison)


def write(raw_root: str, result: dict[str, Any], matches: list[str], rules: dict) -> Path:
    out = approved_output_root(raw_root) / SCHEMA
    out.mkdir(parents=True, exist_ok=True)
    files = {"event_validation.csv": A.write_csv(result["rows"]), "per_match.csv": A.write_csv(result["per_match"]),
             "summary.json": json.dumps(result["summary"], indent=1, sort_keys=True).encode(),
             "dataset_checks.json": json.dumps(result["dataset_checks"], indent=1, sort_keys=True).encode(),
             "comparison_v2_v3.json": json.dumps(result["comparison"], indent=1, sort_keys=True).encode()}
    for n, b in files.items():
        (out / n).write_bytes(b)
    man = dict(schema=SCHEMA, events_source=SOURCE, contexts_source=CONTEXTS, matches=matches, reasons=REASONS, status_fields=A.STATUS_FIELDS,
               tolerances=dict(TOL_POS_M=A.TOL_POS_M, TOL_RAW_M=A.TOL_RAW_M, TOL_T_S=A.TOL_T_S, TOL_BOUND_M=A.TOL_BOUND_M, TOL_W=TOL_W,
                               ZERO_M=A.ZERO_M, MAX_GAP_S=A.MAX_GAP_S, JUMP_MPS=A.JUMP_MPS, JUMP_MIN_M=A.JUMP_MIN_M),
               presentation_rules=dict(rules, require="math valid, tracking valid, both players' incoming and outgoing tracks cover "
                                                      "t* +- 2.5 s, no suspected artifact, both observed at t*"),
               contexts_index_manifest_sha256=A._sha(approved_output_root(raw_root) / CONTEXTS / "development-index" / "manifest.json"),
               code_sha256={p.name: A._sha(p) for p in (Path(__file__), Path(A.__file__))},
               outputs_sha256={n: hashlib.sha256(b).hexdigest() for n, b in files.items()},
               note="Audit view of the unchanged v2 events on the v3 event-centred contexts; statuses are not labels.")
    (out / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--matches", nargs="*")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--min-observed-share", type=float, default=RULES["min_observed_share"])
    ap.add_argument("--min-path-m", type=float, default=RULES["min_path_m"])
    a = ap.parse_args(argv)
    rules = dict(min_observed_share=a.min_observed_share, min_path_m=a.min_path_m)
    ms = a.matches or A.development(a.root)
    res = run(a.root, ms, a.jobs, rules)
    p = write(a.root, res, ms, rules)
    print(json.dumps(dict(out=str(p), **{k: res["summary"][k] for k in ("encounters", *A.STATUS_FIELDS)},
                          window_clipped_by_period=res["summary"]["window_clipped_by_period"]), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
