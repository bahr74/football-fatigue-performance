"""Independent audit of ``m11-sync-proximity-v2`` (development matches only): ``m11-sync-proximity-v2-audit``.

Read-only over the dataset and the Step 1 RAW caches; it writes a separate, versioned audit directory and never changes an
event record. Every 0.25 m episode is re-derived with an implementation that shares no code with :mod:`.proximity`:

* **source**: each exported trajectory vertex is looked up in the checksum-verified Step 1 RAW cache (frame, video time, raw
  x / y, state, visibility, period of the claimed provider slot), and the event's slot identity (side, jersey, player, team)
  is compared with the cache roster;
* **detection**: from the exported vertices, both players piecewise linear, compared at the same instants on the union of their
  sample times; on each interval the relative position is linear, so d^2 is quadratic and its minimum is analytic. The
  recomputed minimum, t*, A(t*), B(t*), P*, the threshold, the episode boundaries (d = R where not cut or beyond the exported
  window), duration and the zero-distance labels are compared with the record;
* **integrity**: time order, period, the 0.5 s unobserved-gap rule measured on the cache states, the 12 m/s + 1 m jump rule on
  every joined step, sample / before-t* / synchronized-grid references;
* **coverage**: the requested canonical window, each player's and the shared valid coverage before and after t*;
* **artifacts**: identical positions over time (provider clones), duplicated provider positions, critical Step 1 flags, observed
  evidence.

Statuses and presentation readiness are derived from these with explicit reason codes (:data:`REASONS`). It lives outside ``intersections`` so the v2 code-provenance hashes stay valid. Spatial crossings of
the two paths at *different* times are reported as a descriptive field only (no extraction is added to the pipeline).

Run: ``python -m football_fatigue.proximity_audit --root data/raw [--matches ...] [--jobs 3]``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from football_fatigue.encounters.catalogue import OUTPUT_DIR as STEP1_DIR
from football_fatigue.encounters.trajectories import STATE_AVAILABLE
from football_fatigue.ingestion import approved_output_root
from football_fatigue.matching import ml

SOURCE = "m11-sync-proximity-v2"
SCHEMA = "m11-sync-proximity-v2-audit"
THRESHOLDS = {"0m": 0.0, "0.1m": 0.10, "0.25m": 0.25, "0.5m": 0.50}
PRIMARY = "0.25m"

# numerical tolerances (documented in the report)
TOL_POS_M = 1e-6          # recomputed positions / distances vs. record (float64 on exact-mm vertices)
TOL_RAW_M = 0.0005 + 1e-9  # exported vertex vs. raw cache coordinate (vertices are the raw values rounded to whole mm)
TOL_T_S = 1e-6            # vertex time vs. cache video time (vertices are whole microseconds)
TOL_BOUND_M = 1e-5        # d at a recorded episode boundary vs. R
ZERO_M = 1e-9             # identical positions
MAX_GAP_S = 0.5           # Step 1: unobserved time that keeps a trajectory continuous
JUMP_MPS, JUMP_MIN_M = 12.0, 1.0
FRAME_S = 1 / 29.97
EDGE_S = MAX_GAP_S + FRAME_S + TOL_T_S   # a complete track is exported from its first vertex inside the window: a joined segment
                                         # straddling the edge (at most 0.5 s + 1 frame) can leave that much undrawn

# presentation rules (a view of the data, not a change of the dataset)
MIN_OBSERVED_SHARE = 0.5   # each player's share of VISIBLE vertices in the context
MIN_PATH_M = 1.0           # each player moves at least this far in the context (two visible tracks)

ARTIFACT_FLAGS = ("IMPLAUSIBLE_JUMP", "LABEL_TRANSFER_CANDIDATE", "RECIPROCAL_EXCHANGE", "ATTRIBUTION_UNRESOLVED", "AMBIGUOUS_RAW",
                  "SIDE_OVERCOUNT", "MISSING_RAW", "FRAME_WIDE_SHIFT", "DISCONTINUITY_CANDIDATE")

REASONS = {
    # mathematical validity
    "d_min_mismatch": "the recomputed minimum synchronized distance differs from the record",
    "t_star_mismatch": "the distance at the recorded t* is not the minimum",
    "position_mismatch": "A(t*) or B(t*) does not match the trajectories",
    "midpoint_mismatch": "P* is not the midpoint of A(t*) and B(t*)",
    "threshold_violated": "the minimum distance exceeds the threshold",
    "episode_exceeds_r": "the distance exceeds R inside the recorded episode",
    "start_boundary_mismatch": "the episode start is neither d = R, a support cut nor beyond the exported window",
    "end_boundary_mismatch": "the episode end is neither d = R, a support cut nor beyond the exported window",
    "t_star_outside_support": "t* lies outside the shared valid support or the episode",
    "duration_mismatch": "duration or contact class is inconsistent with the boundaries",
    "zero_label_mismatch": "exact-coincidence / identical-interval labels disagree with the recomputation",
    # tracking support
    "slot_identity_mismatch": "the provider slot's side / jersey / player / team differs from the RAW roster",
    "source_row_mismatch": "an exported vertex differs from the RAW cache (time, position, frame, state, visibility or period)",
    "time_order": "vertex times are not strictly increasing",
    "unobserved_gap_joined": "a trajectory is joined across more than 0.5 s without an available sample",
    "jump_joined": "a trajectory is joined across an implausible jump (> 12 m/s and > 1 m)",
    "before_flag_mismatch": "the before-t* marks of the samples are inconsistent",
    "sync_grid_mismatch": "the synchronized grid is not both players at the same instants",
    "coverage_flag_mismatch": "a track marked complete does not reach the window edge",
    # coverage
    "incoming_incomplete_a": "Player A's valid track does not cover the window before t*",
    "incoming_incomplete_b": "Player B's valid track does not cover the window before t*",
    "outgoing_incomplete_a": "Player A's valid track does not cover the window after t*",
    "outgoing_incomplete_b": "Player B's valid track does not cover the window after t*",
    # artifacts
    "identical_positions": "both players have identical coordinates over time (provider-cloned positions)",
    "duplicate_provider_position": "a provider position is duplicated for another slot in the same frame",
    **{f"flag_{f.lower()}": f"Step 1 flag in the context: {f.lower().replace('_', ' ')}" for f in ARTIFACT_FLAGS},
    # evidence (presentation)
    "estimated_at_t_star": "at least one player is a provider estimate at t*",
    "low_observed_share": f"less than {MIN_OBSERVED_SHARE:.0%} of a player's context vertices are observed",
    "little_motion": f"a player moves less than {MIN_PATH_M:g} m in the context (no visible track)",
}
STATUS_FIELDS = ("mathematically_valid", "tracking_support_valid", "incoming_complete", "outgoing_complete", "context_complete",
                 "provider_artifact_suspected", "presentation_ready")


# ---------------------------------------------------------------- synchronized geometry (independent of .proximity)

def support(ta: np.ndarray, tb: np.ndarray) -> tuple[float, float]:
    return max(ta[0], tb[0]), min(ta[-1], tb[-1])


def grid(ta, xa, ya, tb, xb, yb, extra=()) -> dict[str, np.ndarray]:
    """Both players at the same instants: the union of their vertex times (plus ``extra``) inside the shared support."""
    lo, hi = support(ta, tb)
    t = np.unique(np.r_[ta, tb, np.asarray(extra, float)])
    t = t[(t >= lo) & (t <= hi)]
    ax, ay, bx, by = np.interp(t, ta, xa), np.interp(t, ta, ya), np.interp(t, tb, xb), np.interp(t, tb, yb)
    return dict(t=t, ax=ax, ay=ay, bx=bx, by=by, d=np.hypot(ax - bx, ay - by))


def closest(g: dict[str, np.ndarray], lo: float = -np.inf, hi: float = np.inf) -> tuple[float, float]:
    """Analytic minimum of d(t) over the grid's intervals inside [lo, hi] (relative motion linear on each; ``lo`` / ``hi`` must be
    grid instants or outside the grid): (d_min, earliest t*)."""
    w = (g["t"] >= lo) & (g["t"] <= hi)
    t, rx, ry = g["t"][w], (g["ax"] - g["bx"])[w], (g["ay"] - g["by"])[w]
    if len(t) == 1:
        return float(np.hypot(rx[0], ry[0])), float(t[0])
    ex, ey = np.diff(rx), np.diff(ry)
    ee = ex * ex + ey * ey
    s = np.where(ee > 0, np.clip(-(rx[:-1] * ex + ry[:-1] * ey) / np.where(ee > 0, ee, 1), 0, 1), 0.0)
    d = np.hypot(rx[:-1] + s * ex, ry[:-1] + s * ey)
    ts = t[:-1] + s * np.diff(t)
    k = np.flatnonzero(d <= d.min() + ZERO_M)
    k = k[np.argmin(ts[k])]
    return float(d[k]), float(ts[k])


def d_at(ta, xa, ya, tb, xb, yb, t: float) -> float:
    return float(np.hypot(np.interp(t, ta, xa) - np.interp(t, tb, xb), np.interp(t, ta, ya) - np.interp(t, tb, yb)))


def max_d_on(g: dict[str, np.ndarray], lo: float, hi: float, tracks) -> float:
    """Maximum of d on [lo, hi] (convex on each interval, so attained at a grid instant or an end)."""
    ts = np.r_[g["t"][(g["t"] >= lo) & (g["t"] <= hi)], lo, hi]
    return max(d_at(*tracks, float(t)) for t in ts)


def identical_seconds(g: dict[str, np.ndarray], lo: float = -np.inf, hi: float = np.inf) -> float:
    """Time on which both players have identical coordinates (d = 0 at both ends of an interval: zero throughout)."""
    t, z = g["t"], g["d"] <= ZERO_M
    w = z[:-1] & z[1:]
    a, b = np.maximum(t[:-1], lo), np.minimum(t[1:], hi)
    return float(np.clip(b - a, 0, None)[w].sum())


def path_length(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.hypot(np.diff(x), np.diff(y)).sum()) if len(x) > 1 else 0.0


def spatial_crossings(ta, xa, ya, tb, xb, yb) -> list[tuple[float, float, float, float]]:
    """Points where the two *paths* cross in space, with the time each player passes there: (x, y, tA, tB). Descriptive only:
    a crossing with tA != tB is not a synchronized encounter."""
    if len(ta) < 2 or len(tb) < 2:
        return []
    p, r = np.c_[xa[:-1], ya[:-1]], np.c_[np.diff(xa), np.diff(ya)]
    q, s = np.c_[xb[:-1], yb[:-1]], np.c_[np.diff(xb), np.diff(yb)]
    rxs = r[:, None, 0] * s[None, :, 1] - r[:, None, 1] * s[None, :, 0]
    qp = q[None, :, :] - p[:, None, :]
    ok = np.abs(rxs) > 1e-12
    den = np.where(ok, rxs, 1)
    u = (qp[..., 0] * s[None, :, 1] - qp[..., 1] * s[None, :, 0]) / den
    v = (qp[..., 0] * r[:, None, 1] - qp[..., 1] * r[:, None, 0]) / den
    i, j = np.nonzero(ok & (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1))
    out = []
    for a, b in zip(i.tolist(), j.tolist()):
        out.append((float(p[a, 0] + u[a, b] * r[a, 0]), float(p[a, 1] + u[a, b] * r[a, 1]),
                    float(ta[a] + u[a, b] * (ta[a + 1] - ta[a])), float(tb[b] + v[a, b] * (tb[b + 1] - tb[b]))))
    return out


# ---------------------------------------------------------------- per-event audit

def _side_arrays(npz: dict[str, np.ndarray], k: int, side: int) -> dict[str, np.ndarray]:
    w = np.flatnonzero((npz["sample_event_index"] == k) & (npz["sample_side"] == side))
    w = w[np.argsort(npz["sample_t"][w], kind="stable")]
    return {f: npz[f"sample_{f}"][w] for f in ("t", "x", "y", "source_row", "frame", "visibility", "before_t_star")}


def source_problems(s: dict[str, np.ndarray], slot: int, period: int, cache: dict[str, np.ndarray]) -> list[str]:
    """Vertices vs. the RAW cache of the claimed slot; unobserved gaps and jumps on joined steps."""
    out = []
    r = s["source_row"].astype(np.int64)
    if (r < 0).any() or (r >= len(cache["frame_num"])).any():
        return ["source_row_mismatch"]
    bad = ((cache["frame_num"][r] != s["frame"]) | (np.abs(cache["video_time_s"][r] - s["t"]) > TOL_T_S)
           | ~(np.abs(cache["x"][r, slot] - s["x"]) <= TOL_RAW_M) | ~(np.abs(cache["y"][r, slot] - s["y"]) <= TOL_RAW_M)
           | (cache["state"][r, slot] != STATE_AVAILABLE) | (cache["visibility"][r, slot] != s["visibility"])
           | (cache["canonical_period"][r] != period))
    if bad.any():
        out.append("source_row_mismatch")
    if len(s["t"]) > 1 and not (np.diff(s["t"]) > 0).all():
        out.append("time_order")
    for k in range(len(r) - 1):
        r0, r1 = r[k], r[k + 1]
        if r1 - r0 > 1:                                                  # rows in between: longest wait for an available sample
            av = np.flatnonzero(cache["state"][r0:r1 + 1, slot] == STATE_AVAILABLE) + r0
            if np.diff(cache["video_time_s"][av]).max() > MAX_GAP_S + FRAME_S + TOL_T_S:
                out.append("unobserved_gap_joined")
                break
    dist = np.hypot(np.diff(s["x"]), np.diff(s["y"]))
    dt = np.diff(s["t"])
    if ((dist > JUMP_MIN_M) & (dist > JUMP_MPS * np.where(dt > 0, dt, 1e-12))).any():
        out.append("jump_joined")
    return out


def audit_event(rec: dict, a: dict, b: dict, sync: dict, cache: dict | None) -> dict[str, Any]:
    """All checks and metrics of one 0.25 m episode; ``cache`` None skips the source checks (unit tests)."""
    R = rec["threshold_m"]
    ts, t0, t1 = rec["t_star_s"], rec["t_start_s"], rec["t_end_s"]
    tracks = (a["t"], a["x"], a["y"], b["t"], b["x"], b["y"])
    g = grid(*tracks, extra=(t0, t1))                                   # the episode ends are grid instants
    lo, hi = support(a["t"], b["t"])
    cx = rec["context"]
    ws, we = cx["window_start_s"], cx["window_end_s"]
    math_r, track_r = [], []

    d_min, t_ind = closest(g, t0, t1)                                    # the episode's own minimum (other approaches may share the window)
    if abs(d_min - rec["d_min_m"]) > TOL_POS_M:
        math_r.append("d_min_mismatch")
    if not lo - TOL_T_S <= ts <= hi + TOL_T_S or not t0 - TOL_T_S <= ts <= t1 + TOL_T_S:
        math_r.append("t_star_outside_support")
    elif d_at(*tracks, ts) > d_min + TOL_POS_M or (t_ind < ts - 1e-5 and d_at(*tracks, t_ind) < rec["d_min_m"] - TOL_POS_M):
        math_r.append("t_star_mismatch")
    pa = (float(np.interp(ts, a["t"], a["x"])), float(np.interp(ts, a["t"], a["y"])))
    pb = (float(np.interp(ts, b["t"], b["x"])), float(np.interp(ts, b["t"], b["y"])))
    ea, eb = rec["player_a"]["position_at_t_star"], rec["player_b"]["position_at_t_star"]
    if max(abs(pa[0] - ea[0]), abs(pa[1] - ea[1]), abs(pb[0] - eb[0]), abs(pb[1] - eb[1])) > TOL_POS_M:
        math_r.append("position_mismatch")
    if max(abs(rec["midpoint"][0] - (ea[0] + eb[0]) / 2), abs(rec["midpoint"][1] - (ea[1] + eb[1]) / 2)) > 1e-9:
        math_r.append("midpoint_mismatch")
    if rec["d_min_m"] > R + 1e-12:
        math_r.append("threshold_violated")
    il, ih = max(t0, lo), min(t1, hi)
    if il <= ih and max_d_on(g, il, ih, tracks) > R + TOL_POS_M:
        math_r.append("episode_exceeds_r")

    def boundary(t: float, start: bool) -> str:
        """d = R inside the shared support; at its edge, the player limiting the support is either cut there (support cut) or
        continues outside the exported window (the boundary is then not checkable from the export)."""
        edge = lo if start else hi
        if (t > edge + TOL_T_S) if start else (t < edge - TOL_T_S):
            return "verified_d_eq_R" if abs(d_at(*tracks, t) - R) <= TOL_BOUND_M else "mismatch"
        end = 0 if start else -1
        lim = [key for key, s in (("player_a", a), ("player_b", b)) if abs(s["t"][end] - edge) <= TOL_T_S]
        full = [cx[key]["complete_before" if start else "complete_after"] for key in lim]
        if rec["support_cut_at_start" if start else "support_cut_at_end"] and not all(full):
            return "support_cut"
        return "beyond_exported_window" if all(full) else "mismatch"

    bs, be = boundary(t0, True), boundary(t1, False)
    if bs == "mismatch":
        math_r.append("start_boundary_mismatch")
    if be == "mismatch":
        math_r.append("end_boundary_mismatch")
    if abs(rec["duration_s"] - (t1 - t0)) > 1e-9 or (rec["contact_class"] == "point") != (t1 == t0):
        math_r.append("duration_mismatch")
    ident_ep = identical_seconds(g, t0, t1)
    ident_ctx = identical_seconds(g)
    zero_bad = (rec["d_min_m"] == 0) != rec["exact_coincidence"] or (rec["exact_coincidence"] and d_min > TOL_POS_M)
    covered = t0 >= lo - TOL_T_S and t1 <= hi + TOL_T_S                 # the whole episode is inside the export
    if rec["coincident_interval"] and ident_ep == 0 and covered:
        zero_bad = True
    if not rec["coincident_interval"] and ident_ep > 0:
        zero_bad = True
    if zero_bad:
        math_r.append("zero_label_mismatch")

    # tracking support
    if cache is not None:
        for key, s in (("player_a", a), ("player_b", b)):
            pl = rec[key]
            sl = pl["slot"]
            if (str(cache["slot_side"][sl]) != pl["side"] or int(cache["slot_jersey"][sl]) != pl["jersey"]
                    or int(cache["slot_player_id"][sl]) != pl["player_id"] or int(cache["slot_team_id"][sl]) != pl["team_id"]):
                track_r.append("slot_identity_mismatch")
            track_r += source_problems(s, sl, rec["period"], cache)
    else:
        for s in (a, b):
            if len(s["t"]) > 1 and not (np.diff(s["t"]) > 0).all():
                track_r.append("time_order")
    for s in (a, b):
        if not np.array_equal(s["before_t_star"].astype(bool), s["t"] < ts):
            track_r.append("before_flag_mismatch")
    gx = grid(*tracks, extra=[t for t in (t0, ts, t1) if lo <= t <= hi])
    if (len(gx["t"]) != len(sync["t"]) or not np.allclose(gx["t"], sync["t"], atol=1e-9, rtol=0)
            or not np.allclose(gx["d"], sync["d"], atol=1e-9, rtol=0)
            or not np.array_equal(sync["a_is_sample"].astype(bool), np.isin(sync["t"], a["t"]))
            or not np.array_equal(sync["b_is_sample"].astype(bool), np.isin(sync["t"], b["t"]))):
        track_r.append("sync_grid_mismatch")
    track_r = sorted(set(track_r))

    # coverage (relative to t*)
    cov = dict(requested_pre_s=ts - ws, requested_post_s=we - ts, a_pre_s=ts - a["t"][0], a_post_s=a["t"][-1] - ts,
               b_pre_s=ts - b["t"][0], b_post_s=b["t"][-1] - ts, shared_pre_s=ts - lo, shared_post_s=hi - ts)
    cov_r = []
    for p, key, s in (("a", "player_a", a), ("b", "player_b", b)):
        c = cx[key]
        if (c["complete_before"] and s["t"][0] - ws > EDGE_S) or (c["complete_after"] and we - s["t"][-1] > EDGE_S):
            track_r.append("coverage_flag_mismatch")
        if not c["complete_before"]:
            cov_r.append(f"incoming_incomplete_{p}")
        if not c["complete_after"]:
            cov_r.append(f"outgoing_incomplete_{p}")

    # artifacts and evidence
    art_r = []
    if ident_ctx > 0 or rec["coincident_interval"]:
        art_r.append("identical_positions")
    if rec["player_a"]["duplicate_provider_position"] or rec["player_b"]["duplicate_provider_position"]:
        art_r.append("duplicate_provider_position")
    art_r += [f"flag_{f.lower()}" for f in ARTIFACT_FLAGS if f in cx["critical_flags"]]
    obs = {p: float((s["visibility"] == 1).mean()) if len(s["t"]) else 0.0 for p, s in (("a", a), ("b", b))}
    plen = {p: path_length(s["x"], s["y"]) for p, s in (("a", a), ("b", b))}
    ev_r = []
    if rec["pair_quality"] != "observed/observed":
        ev_r.append("estimated_at_t_star")
    if min(obs.values()) < MIN_OBSERVED_SHARE:
        ev_r.append("low_observed_share")
    if min(plen.values()) < MIN_PATH_M:
        ev_r.append("little_motion")
    cross = [c for c in spatial_crossings(*tracks)]
    sim = [c for c in cross if abs(c[2] - c[3]) <= 1e-6]

    st = dict(mathematically_valid=not math_r, tracking_support_valid=not track_r,
              incoming_complete=not any(r.startswith("incoming") for r in cov_r), outgoing_complete=not any(r.startswith("outgoing") for r in cov_r),
              context_complete=bool(cx["complete"]), provider_artifact_suspected=bool(art_r))
    st["presentation_ready"] = bool(st["mathematically_valid"] and st["tracking_support_valid"] and st["incoming_complete"]
                                    and st["outgoing_complete"] and not art_r and not ev_r)
    return dict(
        match_id=str(rec["match_id"]), event_id=rec["event_id"], period=rec["period"], t_star_s=ts, pair_quality=rec["pair_quality"],
        team_relation=rec["team_relation"], d_min_m=rec["d_min_m"], d_min_recomputed_m=d_min, t_star_recomputed_s=t_ind,
        exact_coincidence=rec["exact_coincidence"], coincident_interval=rec["coincident_interval"],
        encounter_type=("potential_tracking_ambiguity" if art_r and ("identical_positions" in art_r) else
                        "exact_synchronized_coincidence" if rec["d_min_m"] == 0 else "closest_synchronized_approach"),
        **st, **{k: round(float(v), 6) for k, v in cov.items()},
        identical_seconds_episode=round(ident_ep, 6), identical_seconds_context=round(ident_ctx, 6),
        start_boundary=bs, end_boundary=be, a_observed_share=round(obs["a"], 4), b_observed_share=round(obs["b"], 4),
        a_path_m=round(plen["a"], 3), b_path_m=round(plen["b"], 3), spatial_path_crossings=len(cross),
        crossing_min_time_offset_s=round(min(abs(c[2] - c[3]) for c in cross), 6) if cross else None,
        simultaneous_crossings=len(sim),
        reasons=";".join(math_r + sorted(set(track_r)) + cov_r + art_r + ev_r))


# ---------------------------------------------------------------- dataset level

def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def base(raw_root: str) -> Path:
    return approved_output_root(raw_root) / SOURCE


def out_dir(raw_root: str) -> Path:
    return approved_output_root(raw_root) / SCHEMA


def development(raw_root: str) -> list[str]:
    split = json.loads((approved_output_root(raw_root) / ml.SPLITS_DIR / "splits.json").read_text(encoding="utf-8"))
    return [str(m) for m in split["splits"]["development"]]


CACHE_KEYS = ("frame_num", "video_time_s", "canonical_period", "slot_side", "slot_jersey", "slot_player_id", "slot_team_id", "x", "y",
              "state", "visibility")


def load_source_cache(raw_root: str, mid: str) -> dict[str, np.ndarray]:
    d = approved_output_root(raw_root) / STEP1_DIR / mid
    p = d / "root.cache.npz"
    m = json.loads((d / "root.manifest.json").read_text(encoding="utf-8"))
    if m["outputs_sha256"].get(p.name) != _sha(p):
        raise RuntimeError(f"{p} does not match its Step 1 manifest")
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in CACHE_KEYS}


def audit_match(raw_root: str, mid: str) -> dict[str, Any]:
    """Every 0.25 m episode of one development match, plus match-level consistency checks of all four thresholds."""
    if mid not in development(raw_root):
        raise RuntimeError(f"{mid} is not a development match")
    d = base(raw_root) / mid
    ev = {t: [json.loads(x) for x in (d / f"root.events-{t}.jsonl").read_text(encoding="utf-8").splitlines()] for t in THRESHOLDS}
    man = json.loads((d / "root.manifest.json").read_text(encoding="utf-8"))
    checks: dict[str, Any] = {}
    checks["manifest_hashes"] = all(_sha(d / n) == h for n, h in man["outputs_sha256"].items())
    with np.load(d / "root.trajectories.npz") as z:
        npz = {k: z[k] for k in z.files}
    prim = ev[PRIMARY]
    checks["npz_rows"] = (int(npz["sample_event_index"].max()) + 1 == len(prim) and int(npz["sync_event_index"].max()) + 1 == len(prim))
    ids = {t: [r["event_id"] for r in rs] for t, rs in ev.items()}
    checks["unique_ids"] = all(len(set(v)) == len(v) for v in ids.values())
    checks["ids_reproduce"] = all(hashlib.sha256(r["event_key"].encode()).hexdigest()[:16] == r["event_id"] for rs in ev.values() for r in rs)
    checks["threshold_fields"] = all(r["threshold_m"] == THRESHOLDS[t] and r["d_min_m"] <= THRESHOLDS[t] + 1e-12 and str(r["match_id"]) == mid
                                     for t, rs in ev.items() for r in rs)
    key = lambda r: (r["period"], r["player_a"]["slot"], r["player_b"]["slot"], r["t_start_s"])   # noqa: E731
    checks["deterministic_order"] = {t: [key(r) for r in rs] == sorted(key(r) for r in rs) for t, rs in ev.items()}
    checks["no_duplicate_records"] = all(len({(r["pair_id"], r["t_start_s"]) for r in rs}) == len(rs) for rs in ev.values())
    by = {t: {r["event_id"]: r for r in rs} for t, rs in ev.items()}
    nest_ok = True
    for lo_t, hi_t in (("0m", "0.1m"), ("0.1m", "0.25m"), ("0.25m", "0.5m")):
        for r in ev[lo_t]:
            p = by[hi_t].get(r.get("parent_event_id"))
            nest_ok &= p is not None and p["pair_id"] == r["pair_id"] and p["t_start_s"] <= r["t_start_s"] + 1e-9 and r["t_end_s"] <= p["t_end_s"] + 1e-9
            nest_ok &= p is not None and r["event_id"] in p.get("children", {}).get(lo_t, [])
    checks["threshold_nesting"] = bool(nest_ok)
    checks["counts"] = {t: len(rs) for t, rs in ev.items()}
    cache = load_source_cache(raw_root, mid)
    rows = []
    for k, rec in enumerate(prim):
        a, b = _side_arrays(npz, k, 0), _side_arrays(npz, k, 1)
        w = np.flatnonzero(npz["sync_event_index"] == k)
        w = w[np.argsort(npz["sync_t"][w], kind="stable")]
        sync = {f: npz[f"sync_{f}"][w] for f in ("t", "d", "a_is_sample", "b_is_sample")}
        rows.append(audit_event(rec, a, b, sync, cache))
    return dict(match_id=mid, rows=rows, checks=checks)


def _one(args: tuple[str, str]) -> dict[str, Any]:
    return audit_match(*args)


def write_csv(rows: list[dict]) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


def summarize(rows: list[dict]) -> dict[str, Any]:
    n = len(rows)
    out: dict[str, Any] = dict(encounters=n, **{f: sum(bool(r[f]) for r in rows) for f in STATUS_FIELDS})
    out["incomplete_or_rejected"] = n - out["presentation_ready"]
    out["by_encounter_type"] = dict(Counter(r["encounter_type"] for r in rows))
    out["by_quality"] = {q: dict(encounters=sum(r["pair_quality"] == q for r in rows),
                                 presentation_ready=sum(r["pair_quality"] == q and r["presentation_ready"] for r in rows),
                                 provider_artifact_suspected=sum(r["pair_quality"] == q and r["provider_artifact_suspected"] for r in rows))
                         for q in ("observed/observed", "observed/estimated", "estimated/estimated")}
    out["reasons"] = dict(Counter(x for r in rows for x in r["reasons"].split(";") if x).most_common())
    out["boundaries"] = dict(start=dict(Counter(r["start_boundary"] for r in rows)), end=dict(Counter(r["end_boundary"] for r in rows)))
    out["spatial_path_crossings"] = dict(events_with_crossing=sum(r["spatial_path_crossings"] > 0 for r in rows),
                                         events_with_simultaneous_crossing=sum(r["simultaneous_crossings"] > 0 for r in rows))
    cov = {k: np.array([r[k] for r in rows]) for k in ("requested_pre_s", "requested_post_s", "shared_pre_s", "shared_post_s")}
    out["coverage_quantiles_s"] = {k: [round(float(q), 3) for q in np.quantile(v, [0, 0.1, 0.5, 0.9, 1])] for k, v in cov.items()}
    return out


def run(raw_root: str, matches: list[str], jobs: int) -> dict[str, Any]:
    dev = development(raw_root)
    bad = [m for m in matches if m not in dev]
    if bad:
        raise SystemExit(f"not development matches: {bad}")
    if jobs > 1:
        with ProcessPoolExecutor(jobs) as ex:
            res = list(ex.map(_one, [(raw_root, m) for m in matches]))
    else:
        res = [audit_match(raw_root, m) for m in matches]
    rows = [r for m in res for r in m["rows"]]
    idx = base(raw_root) / "development-index"
    summ = json.loads((idx / "summary.json").read_text(encoding="utf-8"))
    man = json.loads((idx / "manifest.json").read_text(encoding="utf-8"))
    ds: dict[str, Any] = dict(per_match={m["match_id"]: m["checks"] for m in res})
    if sorted(matches) == sorted(dev):
        with open(idx / "encounters_index.csv", encoding="utf-8") as f:
            index = list(csv.DictReader(f))
        ds["index_rows_equal_events"] = len(index) == len(rows) == summ["episodes"]
        ds["index_keys_equal_events"] = {(r["match_id"], r["event_id"]) for r in index} == {(r["match_id"], r["event_id"]) for r in rows}
        ds["index_lines_match"] = all(int(r["events_line"]) == int(r["npz_event_index"]) for r in index)
        ds["global_unique_ids"] = len({(r["match_id"], r["event_id"]) for r in rows}) == len(rows)
        tot = Counter()
        for m in res:
            tot.update(m["checks"]["counts"])
        ds["sensitivity_counts_equal_summary"] = all(tot[t] == summ["sensitivity_episodes"][f"episodes_{t.replace('.', '_')}"]
                                                      for t in THRESHOLDS)
        ds["per_match_counts_equal_summary"] = all(m["checks"]["counts"][PRIMARY] == summ["per_match"][m["match_id"]]["episodes_0_25m"]
                                                   for m in res)
        ds["index_manifest_hashes"] = all(_sha(idx / n) == h for n, h in man["outputs_sha256"].items())
        ds["match_manifest_hashes"] = all(_sha(base(raw_root) / m / "root.manifest.json") == h for m, h in man["match_manifests_sha256"].items())
        with open(idx / "subset_high_quality_candidates.csv", encoding="utf-8") as f:
            hq = {(r["match_id"], r["event_id"]) for r in csv.DictReader(f)}
        ready = {(r["match_id"], r["event_id"]) for r in rows if r["presentation_ready"]}
        ds["presentation_ready_subset_of_high_quality"] = ready <= hq
        ds["high_quality_not_ready"] = dict(Counter(x for r in rows if (r["match_id"], r["event_id"]) in hq and not r["presentation_ready"]
                                                    for x in r["reasons"].split(";") if x))
    summary = summarize(rows)
    per_match = []
    for m in res:
        s = summarize(m["rows"])
        per_match.append(dict(match_id=m["match_id"], **{k: s[k] for k in ("encounters", *STATUS_FIELDS)}))
    return dict(rows=rows, summary=summary, per_match=per_match, dataset_checks=ds)


def write(raw_root: str, result: dict[str, Any], matches: list[str]) -> Path:
    out = out_dir(raw_root)
    out.mkdir(parents=True, exist_ok=True)
    files = {"event_validation.csv": write_csv(result["rows"]), "per_match.csv": write_csv(result["per_match"]),
             "summary.json": json.dumps(result["summary"], indent=1, sort_keys=True).encode(),
             "dataset_checks.json": json.dumps(result["dataset_checks"], indent=1, sort_keys=True).encode()}
    for n, b in files.items():
        (out / n).write_bytes(b)
    man = dict(schema=SCHEMA, source=SOURCE, matches=matches, reasons=REASONS, status_fields=STATUS_FIELDS,
               tolerances=dict(TOL_POS_M=TOL_POS_M, TOL_RAW_M=TOL_RAW_M, TOL_T_S=TOL_T_S, TOL_BOUND_M=TOL_BOUND_M, ZERO_M=ZERO_M,
                               MAX_GAP_S=MAX_GAP_S, JUMP_MPS=JUMP_MPS, JUMP_MIN_M=JUMP_MIN_M),
               presentation_rules=dict(MIN_OBSERVED_SHARE=MIN_OBSERVED_SHARE, MIN_PATH_M=MIN_PATH_M),
               source_index_manifest_sha256=_sha(base(raw_root) / "development-index" / "manifest.json"),
               code_sha256=_sha(Path(__file__)), outputs_sha256={n: hashlib.sha256(b).hexdigest() for n, b in files.items()},
               note="Audit view of the unchanged v2 records; statuses are not labels and not identity ground truth.")
    (out / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--matches", nargs="*")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args(argv)
    ms = a.matches or development(a.root)
    ms = [m for m in ms if (base(a.root) / m / f"root.events-{PRIMARY}.jsonl").exists()]
    res = run(a.root, ms, a.jobs)
    p = write(a.root, res, ms)
    print(json.dumps(dict(out=str(p), **{k: res["summary"][k] for k in ("encounters", *STATUS_FIELDS)}), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
