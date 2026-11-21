"""Synchronized player-proximity encounters (M11): ``m11-sync-proximity-v1``.

Objective: two players near the same position at exactly the same timestamp. An encounter of players A and B is a maximal time
interval on which d(t) = |A(t) - B(t)| <= R, both positions evaluated at the same t (never B(t + dt)). Primary R = 0.25 m;
sensitivity R = 0.00, 0.10, 0.50 m. These are tracking-derived identity-ambiguity proxies, not visually verified occlusions.

**Trajectories** are those of ``intersections.exact`` (unchanged): each provider slot is piecewise linear between its update
instants (held positions stationary, then moving), joined only by Step 1 valid steps (same period, at most 0.5 s unobserved, no
implausible jump, no unattributed/ambiguous sample), in exact integer mm / us. A trajectory is many short segments, so curved or
zigzag motion is followed sample by sample; it is never reduced to one straight line.

**Solver.** Every pair of segments of two players whose time intervals overlap gives a pair-interval [lo, hi] (the union of both
players' sample instants splits time; synchronous groups plus :func:`exact.overlap_pairs` are every such pair). On it the
relative position r(t) = A(t) - B(t) is linear: r(s) = r0 + s e, s in [0, 1], so d^2(s) = |r0|^2 + 2 s r0.e + s^2 |e|^2 is
quadratic. Its minimum is at s* = clip(-r0.e / |e|^2, 0, 1); the within-R part is the solution interval of d^2(s) <= R^2 (a
convex set: one sub-interval or none), from the roots of the quadratic, clipped to [0, 1]. Decisions are made in float64 on mm
values (exact integers on synchronous intervals); any pair-interval whose minimum lies within 1e-3 mm^2 of a threshold's R^2 is
decided again with exact rational arithmetic, so R = 0 reproduces the exact solver. Within-R sub-intervals of one pair that touch
or overlap are merged into one episode; separate approaches of one pair stay separate episodes.

**Windows** (the advisor's framework) are centred on every integer game second of each period, spanning [c - 2.5, c + 2.5]
clipped to the period. An episode is detected by every window it overlaps (window-level detections); global deduplication is the
episode itself (pair + contiguous synchronized time), identical to merging the window detections (brute-force test). Each
episode keeps the 5 s context of its canonical window (centre nearest to t*).

**Classes** (descriptive, never an acceptance rule): ``point`` (zero duration), ``isolated`` (within R at no more than one
sample instant: the contact lives between consecutive samples), ``sustained`` (within R across at least two consecutive sample
instants). ``exact_coincidence`` / ``coincident_interval`` mark d = 0. Quality is kept as flags (observed/estimated at t*,
support cut by a gap/period/substitution, interpolation across unobserved samples, held positions, duplicate provider
positions); nothing is discarded for quality.

Run: ``python -m football_fatigue.intersections.proximity --matches 3812 [...] [--dry-run]``; ``--examples`` writes the advisor
examples of one match; ``--cross-match-only`` aggregates.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from football_fatigue.encounters import quality as Q
from football_fatigue.encounters.trajectories import VISIBILITY, RawCache
from football_fatigue.ingestion import IngestionError, approved_output_root
from football_fatigue.matching import ml, validate
from football_fatigue.matching.benchmark import jsonl, load_source

from . import audit as AU
from . import exact as X
from .__main__ import peak_memory_mb
from .advisor import run_bounds
from .catalogue import load_steps, write_retry
from .detect import duplicate_positions

SCHEMA = "m11-sync-proximity-v2"   # v1 episodes unchanged; v2 adds context completeness / cut reasons and deterministic hashed outputs
MM, US, HALF_US = X.MM, X.US, X.HALF_US
BAND_MM2 = 1e-3                  # float decisions this close to R^2 (mm^2) are re-decided exactly
CHUNK = 4_000_000
# Step 1 sample flags that question the identity or integrity of a trajectory sample (metadata; nothing is dropped for them)
CRITICAL_BITS = (Q.IMPLAUSIBLE_JUMP | Q.LABEL_TRANSFER_CANDIDATE | Q.RECIPROCAL_EXCHANGE | Q.ATTRIBUTION_UNRESOLVED | Q.AMBIGUOUS_RAW
                 | Q.SIDE_OVERCOUNT | Q.MISSING_RAW | Q.FRAME_WIDE_SHIFT | Q.DISCONTINUITY_CANDIDATE)


@dataclass(frozen=True)
class ProximityConfig:
    thresholds_m: tuple[float, ...] = (0.0, 0.10, 0.25, 0.50)
    primary_m: float = 0.25
    context_half_s: float = 2.5
    window_stride_s: float = 1.0
    missing_sample_factor: float = 1.5     # flag only: a moving segment longer than this x the match's median segment spans unobserved samples

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def r_mm(self) -> list[int]:
        return [round(r * MM) for r in self.thresholds_m]


# ---------------------------------------------------------------- pair-intervals

def _pos(seg, g: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Float position (mm) of segments ``g`` at times ``t`` (us); exact integers at segment end instants."""
    f = (t - seg["T0"][g]) / (seg["T1"][g] - seg["T0"][g])
    return seg["X0"][g] + f * (seg["X1"][g] - seg["X0"][g]), seg["Y0"][g] + f * (seg["Y1"][g] - seg["Y0"][g])


def pair_rows(seg, keep_mm: float) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Every pair-interval (two players' segments over their common time interval, both evaluated at the same instants) with its
    closest approach; rows whose minimum is above ``keep_mm`` are counted but not kept."""
    ia, ib = X._pairs_by_group(seg, np.arange(len(seg["slot"])))
    oa, ob = X.overlap_pairs(seg)
    parts, pairs_seen, n_zero = [], set(), 0
    for sync, A, B in ((True, ia, ib), (False, oa, ob)):
        for k in range(0, len(A), CHUNK):
            a, b = A[k:k + CHUNK], B[k:k + CHUNK]
            swap = seg["slot"][a] > seg["slot"][b]
            a, b = np.where(swap, b, a), np.where(swap, a, b)                 # A is the lower slot
            lo, hi = np.maximum(seg["T0"][a], seg["T0"][b]), np.minimum(seg["T1"][a], seg["T1"][b])
            ok = hi > lo
            n_zero += int((~ok).sum())
            a, b, lo, hi = a[ok], b[ok], lo[ok], hi[ok]
            pairs_seen.update(np.unique(seg["period"][a] * 10**6 + seg["slot"][a] * 1000 + seg["slot"][b]).tolist())
            ax0, ay0 = _pos(seg, a, lo)
            bx0, by0 = _pos(seg, b, lo)
            ax1, ay1 = _pos(seg, a, hi)
            bx1, by1 = _pos(seg, b, hi)
            r0x, r0y = ax0 - bx0, ay0 - by0
            ex, ey = (ax1 - bx1) - r0x, (ay1 - by1) - r0y
            ee = ex * ex + ey * ey
            s = np.where(ee > 0, np.clip(-(r0x * ex + r0y * ey) / np.where(ee > 0, ee, 1.0), 0.0, 1.0), 0.0)
            d2 = (r0x + s * ex) ** 2 + (r0y + s * ey) ** 2
            w = d2 <= keep_mm * keep_mm + BAND_MM2
            parts.append(dict(ga=a[w], gb=b[w], lo=lo[w], hi=hi[w], r0x=r0x[w], r0y=r0y[w], ex=ex[w], ey=ey[w], s=s[w], d2=d2[w],
                              sync=np.full(int(w.sum()), sync)))
    rows = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]} if parts else {}
    counts = dict(synchronous_pair_intervals=int(len(ia)), overlap_pair_intervals=int(len(oa)),
                  pair_interval_comparisons=int(len(ia) + len(oa) - n_zero), zero_length_overlaps=n_zero,
                  period_slot_pairs_with_common_support=len(pairs_seen), rows_within_max_threshold=int(len(rows.get("ga", ()))))
    return rows, counts


def exact_row(seg, ga: int, gb: int, lo: int, hi: int) -> tuple[Fraction, Fraction, bool]:
    """Exact minimum of d^2 (mm^2) on one pair-interval, its parameter s*, and whether d = 0 over the whole interval."""
    flo, fhi = Fraction(int(lo)), Fraction(int(hi))
    a0, a1, b0, b1 = X._at(seg, ga, flo), X._at(seg, ga, fhi), X._at(seg, gb, flo), X._at(seg, gb, fhi)
    r0 = (a0[0] - b0[0], a0[1] - b0[1])
    e = (a1[0] - b1[0] - r0[0], a1[1] - b1[1] - r0[1])
    ee = e[0] * e[0] + e[1] * e[1]
    s = Fraction(0) if ee == 0 else min(max(-(r0[0] * e[0] + r0[1] * e[1]) / ee, Fraction(0)), Fraction(1))
    d2 = (r0[0] + s * e[0]) ** 2 + (r0[1] + s * e[1]) ** 2
    return d2, s, ee == 0 and r0 == (0, 0)


def exact_border(seg, rows, r_mm: list[int]) -> None:
    """Exact re-decision of rows whose float minimum is within ``BAND_MM2`` of some R^2 (adds ``exact`` / ``exact_d2`` /
    ``exact_s`` / ``whole``)."""
    n = len(rows["ga"])
    border = np.zeros(n, bool)
    for r in r_mm:
        border |= np.abs(rows["d2"] - r * r) <= BAND_MM2
    rows["exact"] = border
    rows["exact_d2"] = np.full(n, np.nan)
    rows["exact_zero"] = np.zeros(n, bool)
    rows["exact_s"] = rows["s"].copy()
    rows["whole"] = np.zeros(n, bool)
    rows["_frac_d2"] = {}
    for k in np.flatnonzero(border):
        d2, s, whole = exact_row(seg, int(rows["ga"][k]), int(rows["gb"][k]), int(rows["lo"][k]), int(rows["hi"][k]))
        rows["_frac_d2"][int(k)] = d2
        rows["exact_d2"][k], rows["exact_s"][k], rows["whole"][k], rows["exact_zero"][k] = float(d2), float(s), whole, d2 == 0


def qualify(rows, r: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows with min d <= r (mm) and their within-r parameter interval [s1, s2] (exact decision on border rows)."""
    R2 = r * r
    q = rows["d2"] <= R2
    for k, d2 in rows["_frac_d2"].items():
        q[k] = d2 <= R2
    idx = np.flatnonzero(q)
    r0x, r0y, ex, ey, s = (rows[k][idx] for k in ("r0x", "r0y", "ex", "ey", "exact_s"))
    ee = ex * ex + ey * ey
    be = r0x * ex + r0y * ey
    c0 = r0x * r0x + r0y * r0y
    c1 = (r0x + ex) ** 2 + (r0y + ey) ** 2
    sq = np.sqrt(np.maximum(be * be - ee * (c0 - R2), 0.0))
    safe = np.where(ee > 0, ee, 1.0)
    s1 = np.where(ee > 0, (-be - sq) / safe, 0.0)
    s2 = np.where(ee > 0, (-be + sq) / safe, 1.0)
    s1, s2 = np.clip(np.minimum(s1, s), 0, 1), np.clip(np.maximum(s2, s), 0, 1)
    s1, s2 = np.where(c0 <= R2, 0.0, s1), np.where(c1 <= R2, 1.0, s2)
    whole, ex_zero = rows["whole"][idx], rows["exact_zero"][idx]
    if r == 0:                                                           # exact solutions only: the point s*, or the whole interval
        s1, s2 = np.where(whole, 0.0, np.where(ex_zero, s, s1)), np.where(whole, 1.0, np.where(ex_zero, s, s2))
    return idx, s1, s2


# ---------------------------------------------------------------- episodes

def episodes(seg, rows, idx: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> list[dict]:
    """Merge within-R sub-intervals of one pair that touch or overlap; every other approach is its own episode."""
    ga, gb = rows["ga"][idx], rows["gb"][idx]
    lo, hi = rows["lo"][idx].astype(float), rows["hi"][idx].astype(float)
    t_in, t_out = lo + s1 * (hi - lo), lo + s2 * (hi - lo)
    t_in, t_out = np.where(s1 == 0, lo, t_in), np.where(s2 == 1, hi, t_out)     # exact breakpoints, so contiguity is exact
    p, sa, sb = seg["period"][ga], seg["slot"][ga], seg["slot"][gb]
    order = np.lexsort((t_in, sb, sa, p))
    out, cur = [], None
    for k in order.tolist():
        key = (int(p[k]), int(sa[k]), int(sb[k]))
        if cur is not None and cur["pair"] == key and t_in[k] <= cur["t_out"]:
            cur["t_out"] = max(cur["t_out"], float(t_out[k]))
            cur["members"].append(k)
        else:
            cur = dict(pair=key, t_in=float(t_in[k]), t_out=float(t_out[k]), members=[k])
            out.append(cur)
    for e in out:
        m = np.array(e["members"])
        e["rows"] = idx[m]
        e["s1"], e["s2"], e["mt_in"], e["mt_out"] = s1[m], s2[m], t_in[m], t_out[m]
    return out


def _vis_class(c: RawCache, rows_: tuple[int, int], slot: int) -> str:
    v = {int(c.visibility[r, slot]) for r in rows_}
    return "observed" if v == {VISIBILITY["VISIBLE"]} else "estimated"


def describe(c: RawCache, mask: np.ndarray, seg, rows, e: dict, r: int, bounds, dup: np.ndarray, med_seg_us: float,
             cfg: ProximityConfig) -> dict[str, Any]:
    p, sa, sb = e["pair"]
    rr = e["rows"]
    d2 = np.where(rows["exact"][rr], rows["exact_d2"][rr], rows["d2"][rr])
    lo, hi = rows["lo"][rr].astype(float), rows["hi"][rr].astype(float)
    tstar_all = lo + rows["exact_s"][rr] * (hi - lo)
    best = int(np.lexsort((tstar_all, d2))[0])                           # minimum distance, earliest on ties
    k = int(rr[best])
    e["best_row"] = k
    ga, gb = int(rows["ga"][k]), int(rows["gb"][k])
    t_star = float(tstar_all[best])
    ax, ay = (float(v[0]) / MM for v in _pos(seg, np.array([ga]), np.array([t_star])))
    bx, by = (float(v[0]) / MM for v in _pos(seg, np.array([gb]), np.array([t_star])))
    d_min = 0.0 if rows["exact_zero"][k] else math.sqrt(max(float(d2[best]), 0.0)) / MM
    inst = np.unique(np.r_[lo, hi])
    n_inst = int(((inst >= e["t_in"]) & (inst <= e["t_out"])).sum())
    dur_us = e["t_out"] - e["t_in"]
    cls = "point" if dur_us == 0 else ("isolated" if n_inst <= 1 else "sustained")
    whole = rows["whole"][rr]
    start, end = bounds[p]
    wins = X.windows_of(e["t_in"], e["t_out"], start, end)
    canon = min(wins, key=lambda w: (abs(start + w * US - t_star), w))
    segs = {"a": np.unique(rows["ga"][rr]), "b": np.unique(rows["gb"][rr])}

    def cut(g: int, at_start: bool) -> bool:
        """The player's valid trajectory starts (ends) at this segment: no joined neighbour."""
        n = g - 1 if at_start else g + 1
        if not 0 <= n < len(seg["slot"]) or seg["slot"][n] != seg["slot"][g]:
            return True
        return seg["row1"][n] != seg["row0"][g] if at_start else seg["row0"][n] != seg["row1"][g]

    first, last = int(rr[0]), int(rr[-1])
    trunc_start = bool(e["s1"][0] == 0 and (
        (rows["lo"][first] == seg["T0"][rows["ga"][first]] and cut(int(rows["ga"][first]), True))
        or (rows["lo"][first] == seg["T0"][rows["gb"][first]] and cut(int(rows["gb"][first]), True))))
    trunc_end = bool(e["s2"][-1] == 1 and (
        (rows["hi"][last] == seg["T1"][rows["ga"][last]] and cut(int(rows["ga"][last]), False))
        or (rows["hi"][last] == seg["T1"][rows["gb"][last]] and cut(int(rows["gb"][last]), False))))

    def player(side: str, slot: int, g: int) -> dict[str, Any]:
        gs = segs[side]
        smp = np.unique(np.r_[seg["row0"][gs], seg["row1"][gs]])
        moving = gs[~seg["stationary"][gs]]
        bits = int(np.bitwise_or.reduce(mask[smp, slot])) if len(smp) else 0
        r0, r1 = int(seg["row0"][g]), int(seg["row1"][g])
        return dict(slot=int(slot), side=str(c.slot_side[slot]), team_id=int(c.slot_team_id[slot]), jersey=int(c.slot_jersey[slot]),
                    player_id=int(c.slot_player_id[slot]), status_at_t_star=_vis_class(c, (r0, r1), slot),
                    estimated_share=round(float((c.visibility[smp, slot] == VISIBILITY["ESTIMATED"]).mean()), 4),
                    quality_flags=[n for n, b in Q.MASK_BITS.items() if bits & b],
                    spans_missing_samples=bool(len(moving) and ((seg["T1"][moving] - seg["T0"][moving]) > cfg.missing_sample_factor * med_seg_us).any()),
                    held_position=bool(seg["stationary"][gs].any()), duplicate_provider_position=bool(dup[smp, slot].any()),
                    segment_at_t_star=dict(row0=r0, row1=r1, frame0=int(c.frame_num[r0]), frame1=int(c.frame_num[r1]), t0=float(seg["t0"][g]),
                                           t1=float(seg["t1"][g]), x0=float(seg["x0"][g]), y0=float(seg["y0"][g]), x1=float(seg["x1"][g]),
                                           y1=float(seg["y1"][g]), visibility=[int(c.visibility[r0, slot]), int(c.visibility[r1, slot])],
                                           stationary=bool(seg["stationary"][g])),
                    position_at_t_star=[ax, ay] if side == "a" else [bx, by])

    A, B = player("a", sa, ga), player("b", sb, gb)
    key = f"{c.match_id}|{p}|{sa}|{sb}|{r}|{round(e['t_in'])}"
    return dict(
        event_id=hashlib.sha256(key.encode()).hexdigest()[:16], event_key=key, schema=SCHEMA, threshold_m=r / MM, match_id=int(c.match_id),
        period=int(p), pair_id=f"{c.match_id}|{p}|{sa}|{sb}", player_a=A, player_b=B,
        team_relation="same_team" if A["side"] == B["side"] else "opposite_teams",
        t_start_s=e["t_in"] / US, t_end_s=e["t_out"] / US, duration_s=dur_us / US, t_star_s=t_star / US, d_min_m=d_min,
        midpoint=[(ax + bx) / 2, (ay + by) / 2], contact_class=cls, sample_instants_within=n_inst,
        exact_coincidence=bool(rows["exact_zero"][rr].any()), coincident_interval=bool(whole.any()),
        coincident_seconds=float(((hi - lo)[whole]).sum() / US),
        pair_quality="/".join(sorted((A["status_at_t_star"], B["status_at_t_star"]), reverse=True)),
        support_cut_at_start=trunc_start, support_cut_at_end=trunc_end, pair_intervals=int(len(rr)),
        synchronous_only=bool(rows["sync"][rr].all()), windows_game_s=wins, n_windows=len(wins), canonical_window_game_s=canon,
        period_start_s=start / US, period_end_s=end / US)


# ---------------------------------------------------------------- context export

def step_index(steps: dict[str, np.ndarray]) -> dict[str, Any]:
    """Lookup of Step 1 steps by (slot, landing row) and (slot, starting row), to name why a valid trajectory run ends."""
    s, a, b = steps["slot"].tolist(), steps["row_from"].tolist(), steps["row_to"].tolist()
    return dict(to={(si, bi): k for k, (si, bi) in enumerate(zip(s, b))}, frm={(si, ai): k for k, (si, ai) in enumerate(zip(s, a))},
                valid=steps["valid"], jump=steps["jump"])


def _cut_reason(si: dict[str, Any], k: int | None) -> str:
    """Why the neighbouring step was not joined (Step 1 rules)."""
    if k is None:
        return "no_adjacent_sample"                # the player has no sample on that side: entry, exit, substitution, period edge
    if not si["valid"][k]:
        return "gap_period_or_time"                # more than 0.5 s unobserved, a period change, or a non-increasing timestamp
    if si["jump"][k]:
        return "implausible_jump"                  # > 12 m/s and > 1 m
    return "unattributed_or_ambiguous"


def context(c: RawCache, mask: np.ndarray, seg, ga: int, gb: int, lo_s: float, hi_s: float, extra_t: list[float],
            si: dict[str, Any] | None = None) -> dict[str, Any]:
    """Both players' original samples (vertices) in [lo_s, hi_s] on the valid run containing ``ga`` / ``gb``, and the synchronized
    comparison grid: the union of both players' sample instants (plus ``extra_t``) where both have support, each player
    interpolated at exactly the same t, with masks saying which positions are original samples.

    ``info`` says, per player, whether the valid run covers the whole window before and after (it continues beyond the window, or
    its first / last sample reaches the window edge) and, when it does not, why the run ends there (with ``si`` given)."""
    out, info = {}, {}
    for side, g in (("a", ga), ("b", gb)):
        slot = int(seg["slot"][g])
        i, j = run_bounds(seg, g, lo_s, hi_s)
        rows_ = np.r_[seg["row0"][i:j + 1], seg["row1"][j]]
        t = np.r_[seg["t0"][i:j + 1], seg["t1"][j]]
        x = np.r_[seg["x0"][i:j + 1], seg["x1"][j]]
        y = np.r_[seg["y0"][i:j + 1], seg["y1"][j]]
        out[side] = np.c_[t, x, y, rows_, c.frame_num[rows_], c.visibility[rows_, slot], mask[rows_, slot]]
        joined_before = i > 0 and seg["slot"][i - 1] == slot and seg["row1"][i - 1] == seg["row0"][i]
        joined_after = j + 1 < len(seg["slot"]) and seg["slot"][j + 1] == slot and seg["row0"][j + 1] == seg["row1"][j]
        before = bool(joined_before or t[0] <= lo_s + 1e-9)
        after = bool(joined_after or t[-1] >= hi_s - 1e-9)
        d = dict(complete_before=before, complete_after=after, first_sample_s=float(t[0]), last_sample_s=float(t[-1]), n_samples=int(len(t)))
        if si is not None:
            d["cut_before"] = None if before else _cut_reason(si, si["to"].get((slot, int(seg["row0"][i]))))
            last = int(seg["row0"][j]) if seg["stationary"][j] else int(seg["row1"][j])     # a held tail ends one frame before its run
            d["cut_after"] = None if after else _cut_reason(si, si["frm"].get((slot, last)))
        bits = int(np.bitwise_or.reduce(mask[rows_, slot]))
        d["context_flags"] = [n for n, b in Q.MASK_BITS.items() if bits & b]
        d["critical_flags"] = [n for n, b in Q.MASK_BITS.items() if bits & b & CRITICAL_BITS]
        info[side] = d
    a, b = out["a"], out["b"]
    lo, hi = max(a[0, 0], b[0, 0]), min(a[-1, 0], b[-1, 0])
    tt = np.unique(np.r_[a[:, 0], b[:, 0], extra_t])
    tt = tt[(tt >= lo) & (tt <= hi)]
    xa, ya = np.interp(tt, a[:, 0], a[:, 1]), np.interp(tt, a[:, 0], a[:, 2])
    xb, yb = np.interp(tt, b[:, 0], b[:, 1]), np.interp(tt, b[:, 0], b[:, 2])
    sync = np.c_[tt, xa, ya, xb, yb, np.hypot(xa - xb, ya - yb), np.isin(tt, a[:, 0]), np.isin(tt, b[:, 0])]
    return dict(a=a, b=b, sync=sync, info=info)


def traj_for(c, mask, seg, rows, e: dict, rec: dict, si: dict[str, Any] | None = None) -> dict[str, Any]:
    """Canonical 5 s context of an episode; also stores its completeness in ``rec["context"]``."""
    k = e["best_row"]                                                    # the pair-interval of t* (set by :func:`describe`)
    cw = rec["period_start_s"] + rec["canonical_window_game_s"]
    lo, hi = max(cw - 2.5, rec["period_start_s"]), min(cw + 2.5, rec["period_end_s"])
    tr = context(c, mask, seg, int(rows["ga"][k]), int(rows["gb"][k]), lo, hi, [rec["t_start_s"], rec["t_star_s"], rec["t_end_s"]], si)
    ia, ib = tr["info"]["a"], tr["info"]["b"]
    rec["context"] = dict(window_centre_s=cw, window_start_s=lo, window_end_s=hi, window_clipped_by_period=bool(lo > cw - 2.5 or hi < cw + 2.5),
                          complete=all(d[k2] for d in (ia, ib) for k2 in ("complete_before", "complete_after")),
                          critical_flags=sorted(set(ia["critical_flags"]) | set(ib["critical_flags"])), player_a=ia, player_b=ib)
    return tr


def store(trajs: list[dict[str, np.ndarray]], t_star: list[float]) -> dict[str, np.ndarray]:
    """Original samples of both players (side 0 = A, 1 = B; ``before`` marks t < t*) and the synchronized grid, per event."""
    sm, sy = [], []
    for k, (tr, ts) in enumerate(zip(trajs, t_star)):
        for side in (0, 1):
            v = tr["ab"[side]]
            sm.append(np.c_[np.full(len(v), k), np.full(len(v), side), v, v[:, 0] < ts])
        sy.append(np.c_[np.full(len(tr["sync"]), k), tr["sync"]])
    S = np.concatenate(sm) if sm else np.zeros((0, 10))
    Y = np.concatenate(sy) if sy else np.zeros((0, 9))
    return dict(sample_event_index=S[:, 0].astype(np.int64), sample_side=S[:, 1].astype(np.int8), sample_t=S[:, 2], sample_x=S[:, 3],
                sample_y=S[:, 4], sample_source_row=S[:, 5].astype(np.int64), sample_frame=S[:, 6].astype(np.int64),
                sample_visibility=S[:, 7].astype(np.int8), sample_quality_bits=S[:, 8].astype(np.int64), sample_before_t_star=S[:, 9].astype(bool),
                sync_event_index=Y[:, 0].astype(np.int64), sync_t=Y[:, 1], sync_xa=Y[:, 2], sync_ya=Y[:, 3], sync_xb=Y[:, 4], sync_yb=Y[:, 5],
                sync_d=Y[:, 6], sync_a_is_sample=Y[:, 7].astype(bool), sync_b_is_sample=Y[:, 8].astype(bool))


# ---------------------------------------------------------------- statistics

def _q(v) -> dict[str, float] | None:
    v = np.asarray(v, float)
    if not len(v):
        return None
    return {k: round(float(x), 6) for k, x in zip(("min", "p10", "p25", "median", "p75", "p90", "max"), np.quantile(v, [0, .1, .25, .5, .75, .9, 1]))}


def stats(recs: list[dict], n_rows: int) -> dict[str, Any]:
    wl = sum(r["n_windows"] for r in recs)
    players = {(r["player_a"]["slot"]) for r in recs} | {(r["player_b"]["slot"]) for r in recs}
    cnt = lambda f: sum(1 for r in recs if f(r))  # noqa: E731
    return dict(
        segment_level_detections=n_rows, window_level_detections=wl, repeated_window_detections=wl - len(recs), unique_episodes=len(recs),
        point=cnt(lambda r: r["contact_class"] == "point"), isolated=cnt(lambda r: r["contact_class"] == "isolated"),
        sustained=cnt(lambda r: r["contact_class"] == "sustained"), exact_coincidence=cnt(lambda r: r["exact_coincidence"]),
        coincident_interval=cnt(lambda r: r["coincident_interval"]),
        distinct_pairs=len({r["pair_id"] for r in recs}), distinct_players=len(players),
        same_team=cnt(lambda r: r["team_relation"] == "same_team"), opposite_teams=cnt(lambda r: r["team_relation"] == "opposite_teams"),
        observed_observed=cnt(lambda r: r["pair_quality"] == "observed/observed"),
        observed_estimated=cnt(lambda r: r["pair_quality"] == "observed/estimated"),
        estimated_estimated=cnt(lambda r: r["pair_quality"] == "estimated/estimated"),
        support_cut=cnt(lambda r: r["support_cut_at_start"] or r["support_cut_at_end"]),
        spans_missing_samples=cnt(lambda r: r["player_a"]["spans_missing_samples"] or r["player_b"]["spans_missing_samples"]),
        held_position=cnt(lambda r: r["player_a"]["held_position"] or r["player_b"]["held_position"]),
        duplicate_provider_position=cnt(lambda r: r["player_a"]["duplicate_provider_position"] or r["player_b"]["duplicate_provider_position"]),
        non_synchronous=cnt(lambda r: not r["synchronous_only"]),
        d_min_m=_q([r["d_min_m"] for r in recs]), duration_s=_q([r["duration_s"] for r in recs]),
        d_min_hist_m={f"<={b:g}": cnt(lambda r, b=b: r["d_min_m"] <= b) for b in (0.0, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)},
        duration_hist_s={f"<={b:g}": cnt(lambda r, b=b: r["duration_s"] <= b) for b in (0.0, 0.0667, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)})


def nest(recs_by_r: dict[int, list[dict]]) -> None:
    """Link every episode to the episode containing it at the next larger threshold (thresholds nest: d <= r implies d <= r')."""
    rs = sorted(recs_by_r)
    for r, r2 in zip(rs[:-1], rs[1:]):
        by_pair: dict[str, list[dict]] = {}
        for p in recs_by_r[r2]:
            by_pair.setdefault(p["pair_id"], []).append(p)
        for e in recs_by_r[r]:
            par = [p for p in by_pair.get(e["pair_id"], []) if p["t_start_s"] <= e["t_start_s"] + 1e-9 and p["t_end_s"] >= e["t_end_s"] - 1e-9]
            if len(par) != 1:
                raise AssertionError(f"episode {e['event_id']} has {len(par)} parents at {r2} mm")
            e["parent_event_id"] = par[0]["event_id"]
            par[0].setdefault("children", {}).setdefault(f"{r / MM:g}m", []).append(e["event_id"])


def replica_near(seg, rows, r_mm: int = 500) -> int:
    """The historical ``near_0.5m`` count of ``exact.near_episodes``: runs of consecutive *synchronous* pair-intervals of one pair whose
    float minimum, rounded to mm, is <= r; partial overlaps (held positions, irregular cadence) are not part of it."""
    dmin = np.sqrt(rows["d2"]) / MM                                      # as there: float minimum <= r, then rounded to mm
    w = rows["sync"] & (dmin <= r_mm / MM) & (np.round(dmin * MM) <= r_mm)
    a, b = rows["ga"][w], rows["gb"][w]
    key = np.c_[seg["period"][a], seg["slot"][a], seg["slot"][b], seg["T0"][a], seg["T1"][a]]
    key = key[np.lexsort(key.T[::-1])]
    new = np.r_[True, np.any(key[1:, :3] != key[:-1, :3], axis=1) | (key[1:, 3] != key[:-1, 4])]
    return int(new.sum()) if len(key) else 0


def exact_mapping(raw_root: str, match_id: str, recs_by_r: dict[int, list[dict]]) -> list[dict] | None:
    """Each event of ``m11-exact-spatiotemporal-v1`` (when written) with the proximity episodes containing it."""
    p = approved_output_root(raw_root) / X.SCHEMA / str(match_id) / "root.events.jsonl"
    if not p.exists():
        return None
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        x = json.loads(line)
        lo, hi = (x["t_s"], x["t_s"]) if x["kind"] == "point" else (x["t_start_s"], x["t_end_s"])
        pid = f"{match_id}|{x['period']}|{x['slot_a']}|{x['slot_b']}"
        row = dict(exact_event_id=x["event_id"], kind=x["kind"], t_start_s=lo, t_end_s=hi, period=x["period"],
                   player_a=dict(slot=x["slot_a"], side=x["side_a"], jersey=x["jersey_a"], player_id=x["player_id_a"]),
                   player_b=dict(slot=x["slot_b"], side=x["side_b"], jersey=x["jersey_b"], player_id=x["player_id_b"]),
                   team_relation=x["team_relation"], segment_a=x["segment_a"], segment_b=x["segment_b"],
                   v1_duplicate_position_rule_would_exclude=x["v1_duplicate_position_rule_would_exclude"])
        for r, recs in sorted(recs_by_r.items()):
            hit = [e for e in recs if e["pair_id"] == pid and e["t_start_s"] <= lo + 1e-9 and e["t_end_s"] >= hi - 1e-9]
            row[f"episode_{r / MM:g}m"] = (dict(event_id=hit[0]["event_id"], t_start_s=hit[0]["t_start_s"], t_end_s=hit[0]["t_end_s"],
                                                 duration_s=hit[0]["duration_s"], contact_class=hit[0]["contact_class"],
                                                 pair_quality=hit[0]["pair_quality"]) if hit else None)
        out.append(row)
    return out


# ---------------------------------------------------------------- per match

def detect_all(c: RawCache, mask: np.ndarray, seg, bounds, dup, cfg: ProximityConfig) -> tuple[dict, dict, dict, dict]:
    rows, counts = pair_rows(seg, max(cfg.r_mm) + 1.0)
    exact_border(seg, rows, cfg.r_mm)
    moving = ~seg["stationary"]
    med = float(np.median(seg["T1"][moving] - seg["T0"][moving])) if moving.any() else 0.0
    recs_by_r, eps_by_r, n_rows = {}, {}, {}
    for r in cfg.r_mm:
        idx, s1, s2 = qualify(rows, r)
        eps = episodes(seg, rows, idx, s1, s2)
        recs_by_r[r] = [describe(c, mask, seg, rows, e, r, bounds, dup, med, cfg) for e in eps]
        eps_by_r[r], n_rows[r] = eps, len(idx)
    nest(recs_by_r)
    counts["median_moving_segment_s"] = med / US
    return rows, counts, dict(recs=recs_by_r, eps=eps_by_r, n_rows=n_rows), dict(replica_exact_near_0_5m=replica_near(seg, rows))


def build(raw_root: str, match_id: str, cfg: ProximityConfig = ProximityConfig()) -> tuple[dict, dict, dict]:
    c, mask, _enc, src_manifest, prov = load_source(raw_root, match_id)
    steps = load_steps(raw_root, match_id)
    seg, seg_counts = X.trajectories(c, mask, steps)
    bounds = X.period_bounds(seg, c)
    dup = duplicate_positions(c)
    rows, counts, det, extra = detect_all(c, mask, seg, bounds, dup, cfg)
    win = X.window_accounting(seg, bounds)
    summary = dict(schema=SCHEMA, match_id=int(match_id), config=cfg.as_dict(), trajectories=seg_counts,
                   windows=win, comparisons=counts,
                   thresholds={f"{r / MM:g}m": stats(det["recs"][r], det["n_rows"][r]) for r in cfg.r_mm},
                   historical=dict(extra, exact_v1_events=exact_mapping(raw_root, match_id, det["recs"]),
                                   spatial_crossings_v1=X.spatial_stage(raw_root, match_id)),
                   source=dict(step1=prov, step1_schema=src_manifest["schema"]))
    return summary, dict(c=c, mask=mask, seg=seg, rows=rows, bounds=bounds, si=step_index(steps), **det), {}


def _code_sha256() -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))}


def verify_existing(out: Path, cfg: ProximityConfig) -> str | None:
    """None when ``out`` is a complete, unmodified output of this schema, configuration and code; otherwise the reason."""
    mp = out / "root.manifest.json"
    if not mp.exists():
        return "no manifest"
    m = json.loads(mp.read_text(encoding="utf-8"))
    if m.get("schema") != SCHEMA or m.get("config") != json.loads(json.dumps(cfg.as_dict())):
        return "schema or configuration differs"
    if m.get("code_sha256") != _code_sha256():
        return "written by different code"
    for name, h in m.get("outputs_sha256", {}).items():
        f = out / name
        if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest() != h:
            return f"{name} missing or modified"
    need = {f"root.events-{r:g}m.jsonl" for r in cfg.thresholds_m} | {"root.trajectories.npz", "root.summary.json"}
    if not need <= set(m.get("outputs_sha256", {})):
        return "incomplete output set"
    return None


def render(raw_root: str, match_id: str, cfg: ProximityConfig = ProximityConfig()) -> tuple[dict[str, bytes], dict[str, Any]]:
    """All hashed outputs of one match (deterministic bytes) and its summary."""
    summary, st, _ = build(raw_root, match_id, cfg)
    primary = round(cfg.primary_m * MM)
    recs = st["recs"][primary]
    trajs = [traj_for(st["c"], st["mask"], st["seg"], st["rows"], e, r, st["si"]) for e, r in zip(st["eps"][primary], recs)]
    summary["context"] = dict(complete=sum(r["context"]["complete"] for r in recs), incomplete=sum(not r["context"]["complete"] for r in recs),
                              with_critical_flags=sum(bool(r["context"]["critical_flags"]) for r in recs))
    blobs = {f"root.events-{r / MM:g}m.jsonl": jsonl(v) for r, v in st["recs"].items()}
    blobs["root.trajectories.npz"] = AU.npz_bytes(store(trajs, [r["t_star_s"] for r in recs]))
    blobs["root.summary.json"] = json.dumps(summary, indent=1, sort_keys=True).encode()
    return blobs, summary


def run_match(raw_root: str, match_id: str, dry_run: bool, cfg: ProximityConfig = ProximityConfig()) -> dict[str, Any]:
    t0 = time.perf_counter()
    split = json.loads((approved_output_root(raw_root) / ml.SPLITS_DIR / "splits.json").read_text(encoding="utf-8"))
    validate.check_split(split, [str(match_id)], "development")
    out = approved_output_root(raw_root) / SCHEMA / str(match_id)
    if not dry_run and out.exists():
        why = verify_existing(out, cfg)
        if why is None:
            return dict(match_id=int(match_id), status="skipped_complete", written_to=str(out))
        raise IngestionError(f"{out} exists but is not a verified complete output ({why}); it is never overwritten")
    snap0 = validate.raw_snapshot(raw_root)
    blobs, summary = render(raw_root, match_id, cfg)
    runtime = dict(seconds=round(time.perf_counter() - t0, 1), peak_memory_mb=peak_memory_mb())
    summary.update(runtime)
    if dry_run:
        return summary
    manifest = dict(schema=SCHEMA, match_id=int(match_id), split=split["matches"][str(match_id)], config=cfg.as_dict(),
                    code_sha256=_code_sha256(), raw_unchanged=snap0 == validate.raw_snapshot(raw_root), source=summary["source"],
                    outputs_sha256={n: hashlib.sha256(b).hexdigest() for n, b in blobs.items()}, python=platform.python_version(),
                    numpy=np.__version__, primary_threshold_m=cfg.primary_m, trajectories_exported_for=f"{cfg.primary_m:g}m",
                    representation="piecewise-linear between update instants (exact mm / us); analytic quadratic minimum per pair-interval",
                    output_precision="JSON floats are full float64 (no rounding); times in s, positions in m (RAW pitch frame)",
                    note="Synchronized proximity encounters of provider trajectories; tracking-derived identity-ambiguity proxies, not "
                         "visually verified occlusions; provider slots, not verified identities; no labels.")
    blobs["root.manifest.json"] = json.dumps(manifest, indent=1, sort_keys=True).encode()
    blobs["root.runtime.json"] = json.dumps(runtime, indent=1).encode()          # not hashed: timing varies between runs
    write_retry(out, blobs)
    summary.update(status="written", written_to=str(out))
    return summary


def _run_one(args: tuple[str, str, bool]) -> dict[str, Any]:
    """Worker: one match, compact result; errors are returned, not raised, so other matches continue."""
    root, mid, dry = args
    try:
        s = run_match(root, mid, dry)
        if s.get("status") == "skipped_complete":
            return s
        return dict(match_id=int(mid), status=s.get("status", "dry_run"), seconds=s["seconds"], peak_memory_mb=s["peak_memory_mb"],
                    episodes={t: v["unique_episodes"] for t, v in s["thresholds"].items()}, context=s["context"])
    except Exception as e:                                               # noqa: BLE001 - reported per match
        return dict(match_id=int(mid), status="failed", error=f"{type(e).__name__}: {e}")


def cross_match(raw_root: str, matches: list[str], dry_run: bool) -> dict[str, Any]:
    keys = ("unique_episodes", "window_level_detections", "segment_level_detections", "point", "isolated", "sustained", "exact_coincidence",
            "coincident_interval", "distinct_pairs", "distinct_players", "same_team", "opposite_teams", "observed_observed", "observed_estimated",
            "estimated_estimated", "support_cut", "spans_missing_samples", "held_position", "duplicate_provider_position", "non_synchronous")
    rows, durs = {}, {}
    for mid in matches:
        d = approved_output_root(raw_root) / SCHEMA / mid
        s = json.loads((d / "root.summary.json").read_text(encoding="utf-8"))
        rows[mid] = dict(windows=s["windows"]["windows"], pair_window_comparisons=s["windows"]["pair_window_comparisons"],
                         pair_interval_comparisons=s["comparisons"]["pair_interval_comparisons"],
                         replica_exact_near_0_5m=s["historical"]["replica_exact_near_0_5m"],
                         **{f"{t}|{k}": v[k] for t, v in s["thresholds"].items() for k in keys})
        for t in s["thresholds"]:
            for line in (d / f"root.events-{t}.jsonl").read_text(encoding="utf-8").splitlines():
                x = json.loads(line)
                durs.setdefault(t, {"d": [], "u": []})
                durs[t]["d"].append(x["d_min_m"])
                durs[t]["u"].append(x["duration_s"])
    tot = {k: sum(r[k] for r in rows.values()) for k in next(iter(rows.values()))}
    rep = dict(schema=SCHEMA, matches=sorted(rows), per_match=rows, totals=tot,
               distributions={t: dict(d_min_m=_q(v["d"]), duration_s=_q(v["u"])) for t, v in durs.items()})
    if not dry_run:
        write_retry(approved_output_root(raw_root) / SCHEMA / "cross-match", {"summary.json": json.dumps(rep, indent=1, sort_keys=True).encode()})
    return rep


# ---------------------------------------------------------------- figures and examples

def figure(rec: dict, tr: dict[str, np.ndarray], title: str, r_m: float) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ad) = plt.subplots(1, 2, figsize=(13, 5.8), gridspec_kw=dict(width_ratios=[1.15, 1]))
    ts = rec["t_star_s"]
    for side, col in (("a", "tab:blue"), ("b", "tab:red")):
        v = tr[side]
        pl = rec[f"player_{side}"]
        ax.plot(v[:, 1], v[:, 2], "-", color=col, lw=1.4, alpha=0.8, label=f"{side.upper()} #{pl['jersey']} ({pl['side']}) interpolated path")
        vis = v[:, 5] == VISIBILITY["VISIBLE"]
        ax.plot(v[vis, 1], v[vis, 2], "o", color=col, ms=3, label=f"{side.upper()} original samples (VISIBLE)")
        ax.plot(v[~vis, 1], v[~vis, 2], "o", color=col, ms=3, mfc="none", label=f"{side.upper()} original samples (ESTIMATED)")
        ax.plot(v[0, 1], v[0, 2], ">", color=col, ms=7)
    sy = tr["sync"]
    for dt, mk in ((-1.0, "s"), (-0.5, "^"), (0.5, "v"), (1.0, "D")):
        if sy[0, 0] <= ts + dt <= sy[-1, 0]:
            q = [np.interp(ts + dt, sy[:, 0], sy[:, k]) for k in (1, 2, 3, 4)]
            ax.plot([q[0], q[2]], [q[1], q[3]], ":", color="grey", lw=1)
            ax.plot([q[0], q[2]], [q[1], q[3]], mk, color="k", ms=6, mfc="none", label=f"both at t*{dt:+.1f} s")
    (xa, ya), (xb, yb) = rec["player_a"]["position_at_t_star"], rec["player_b"]["position_at_t_star"]
    ax.plot([xa], [ya], "o", color="tab:blue", ms=10, mfc="none", mew=2)
    ax.plot([xb], [yb], "o", color="tab:red", ms=10, mfc="none", mew=2)
    ax.plot(*rec["midpoint"], "k*", ms=12, label=f"P* (midpoint) at t* = {ts:.3f} s, d_min = {rec['d_min_m'] * 100:.1f} cm")
    ax.add_patch(plt.Circle(rec["midpoint"], r_m, fill=False, ls="--", color="k", lw=0.8))
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m, RAW)")
    ax.set_ylabel("y (m, RAW)")
    ax.legend(fontsize=6.5, loc="best")
    ad.plot(sy[:, 0] - ts, sy[:, 5], "-", color="k", lw=1.2, label="d(t) = |A(t) - B(t)| (same t)")
    both = sy[:, 6].astype(bool) & sy[:, 7].astype(bool)
    ad.plot(sy[both, 0] - ts, sy[both, 5], ".", color="k", ms=4, label="both players at an original sample")
    ad.axhline(r_m, color="tab:green", ls="--", lw=1, label=f"R = {r_m:g} m")
    if rec.get("t_start_s") is not None and rec["t_end_s"] >= rec["t_start_s"]:
        ad.axvspan(rec["t_start_s"] - ts, rec["t_end_s"] - ts, color="tab:green", alpha=0.15, label="episode (d <= R)")
    ad.axvline(0, color="grey", lw=0.8)
    ad.set_xlabel("t - t* (s)")
    ad.set_ylabel("distance at the same instant (m)")
    ad.set_ylim(0, max(1.0, min(float(sy[:, 5].max()), 10.0)))
    ad.legend(fontsize=7)
    fig.suptitle(title, fontsize=8.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _turn_deg(sy: np.ndarray, ts: float) -> float:
    """Largest heading change of either player between [t*-0.5, t*] and [t*, t*+0.5] (descriptive, for choosing an example)."""
    best = 0.0
    for cx, cy in ((1, 2), (3, 4)):
        p = [np.array([np.interp(ts + d, sy[:, 0], sy[:, cx]), np.interp(ts + d, sy[:, 0], sy[:, cy])]) for d in (-0.5, 0, 0.5)]
        u, v = p[1] - p[0], p[2] - p[1]
        if np.hypot(*u) > 0.1 and np.hypot(*v) > 0.1:
            best = max(best, math.degrees(math.acos(np.clip(u @ v / np.hypot(*u) / np.hypot(*v), -1, 1))))
    return best


def example_record(rec: dict, tr: dict[str, np.ndarray], category: str, why: str) -> dict[str, Any]:
    keep = {k: rec[k] for k in ("event_id", "match_id", "period", "threshold_m", "t_start_s", "t_end_s", "duration_s", "t_star_s", "d_min_m",
                                "midpoint", "contact_class", "pair_quality", "team_relation", "exact_coincidence", "coincident_interval",
                                "support_cut_at_start", "support_cut_at_end") if k in rec}
    samples = {}
    for side in ("a", "b"):
        v = tr[side]
        near = np.argsort(np.abs(v[:, 0] - rec["t_star_s"]))[:4]
        samples[side] = [dict(t_s=float(v[i, 0]), x=float(v[i, 1]), y=float(v[i, 2]), source_row=int(v[i, 3]), frame=int(v[i, 4]),
                              visibility="VISIBLE" if v[i, 5] == 1 else "ESTIMATED") for i in sorted(near)]
    return dict(category=category, selection_rule=why, **keep, player_a={k: rec["player_a"][k] for k in ("slot", "side", "jersey", "player_id",
                "status_at_t_star", "position_at_t_star", "quality_flags")}, player_b={k: rec["player_b"][k] for k in ("slot", "side", "jersey",
                "player_id", "status_at_t_star", "position_at_t_star", "quality_flags")}, nearest_original_samples=samples)


def examples(raw_root: str, match_id: str, catalogue_event: str | None, cfg: ProximityConfig = ProximityConfig()) -> dict[str, bytes]:
    """Advisor examples of one match (selection rules are for illustration only, never acceptance)."""
    summary, st, _ = build(raw_root, match_id, cfg)
    c, mask, seg, rows = st["c"], st["mask"], st["seg"], st["rows"]
    r = round(cfg.primary_m * MM)
    recs, eps = st["recs"][r], st["eps"][r]
    trs = {rec["event_id"]: traj_for(c, mask, seg, rows, e, rec) for e, rec in zip(eps, recs)}
    by_id = {rec["event_id"]: rec for rec in recs}
    mapping = summary["historical"]["exact_v1_events"] or []
    obs = [x for x in recs if x["pair_quality"] == "observed/observed" and not x["exact_coincidence"]]
    picks: list[tuple[str, str, dict | None]] = []

    def from_exact(kind: str, prefer: str | None) -> dict | None:
        cand = [m for m in mapping if m["kind"] == kind and m[f"episode_{cfg.primary_m:g}m"]]
        cand.sort(key=lambda m: (m["exact_event_id"] != prefer, m["exact_event_id"]))
        return by_id[cand[0][f"episode_{cfg.primary_m:g}m"]["event_id"]] if cand else None

    picks.append(("1 previous isolated exact intersection", "the 0.25 m episode containing the exact-v1 interior point", from_exact("point", None)))
    picks.append(("2 previous coincident interval", "the 0.25 m episode containing exact-v1 interval 1803eb7e8a5b7a57 (else the first)",
                  from_exact("interval", "1803eb7e8a5b7a57")))
    opp = [x for x in obs if x["team_relation"] == "opposite_teams" and x["d_min_m"] > 0.05]
    picks.append(("3 0.25 m encounter, nonidentical trajectories", "opposite teams, both observed, d_min > 5 cm, smallest d_min",
                  min(opp, key=lambda x: (x["d_min_m"], x["event_id"])) if opp else None))
    turn = [(x, _turn_deg(trs[x["event_id"]]["sync"], x["t_star_s"])) for x in obs]
    picks.append(("4 changing movement direction", "both observed; largest heading change of either player over t* +- 0.5 s",
                  max(turn, key=lambda z: (z[1], z[0]["event_id"]))[0] if turn else None))
    brief = [x for x in obs if x["contact_class"] == "isolated" and x["duration_s"] > 0]
    picks.append(("5 brief threshold crossing", "isolated class (within R at <= 1 sample instant), both observed, shortest duration",
                  min(brief, key=lambda x: (x["duration_s"], x["event_id"])) if brief else None))
    sus = [x for x in obs if x["contact_class"] == "sustained"]
    picks.append(("6 sustained proximity episode", "sustained class, both observed, not coincident, longest duration",
                  max(sus, key=lambda x: (x["duration_s"], x["event_id"])) if sus else None))
    low = [x for x in recs if x["pair_quality"] == "estimated/estimated" and not x["exact_coincidence"]
           and (x["player_a"]["spans_missing_samples"] or x["player_b"]["spans_missing_samples"] or x["support_cut_at_start"] or x["support_cut_at_end"])]
    picks.append(("8 low-quality / estimated example", "estimated/estimated, not coincident, with a missing-sample or support-cut flag; first by id",
                  min(low, key=lambda x: x["event_id"]) if low else None))
    blobs, meta = {}, []
    for cat, why, rec in picks:
        if rec is None:
            meta.append(dict(category=cat, selection_rule=why, found=False))
            continue
        tr = trs[rec["event_id"]]
        title = (f"{match_id} | {cat} | event {rec['event_id']} | #{rec['player_a']['jersey']} ({rec['player_a']['side']}) x "
                 f"#{rec['player_b']['jersey']} ({rec['player_b']['side']}) | {rec['pair_quality']} | {rec['contact_class']}, "
                 f"{rec['duration_s']:.3f} s | d_min {rec['d_min_m'] * 100:.2f} cm")
        blobs[f"fig-{cat.split()[0]}-{rec['event_id']}.png"] = figure(rec, tr, title, cfg.primary_m)
        meta.append(dict(found=True, **example_record(rec, tr, cat, why)))
    if catalogue_event:                                                  # 7: a 2D crossing at different times, correctly rejected
        cat_rec = None
        with open(approved_output_root(raw_root) / "m11-occlusion-catalogue-v1" / str(match_id) / "root.catalogue.jsonl", encoding="utf-8") as f:
            for line in f:
                x = json.loads(line)
                if x["event_id"] == catalogue_event:
                    cat_rec = x
                    break
        key = {(int(seg["slot"][g]), int(seg["row0"][g])): g for g in range(len(seg["slot"]))}
        ga, gb = key[(cat_rec["slot_a"], cat_rec["segment_a"]["row0"])], key[(cat_rec["slot_b"], cat_rec["segment_b"]["row0"])]
        t_ref = (cat_rec["time_a_s"] + cat_rec["time_b_s"]) / 2
        tr = context(c, mask, seg, ga, gb, t_ref - 2.5, t_ref + 2.5, [cat_rec["time_a_s"], cat_rec["time_b_s"]])
        k = int(np.argmin(tr["sync"][:, 5]))
        sy = tr["sync"][k]
        sides = {s: dict(slot=int(cat_rec[f"slot_{s}"]), side=cat_rec[f"side_{s}"], jersey=int(cat_rec[f"jersey_{s}"]),
                         player_id=int(cat_rec.get(f"player_id_{s}", -1)), status_at_t_star="n/a", quality_flags=[]) for s in ("a", "b")}
        sides["a"]["position_at_t_star"], sides["b"]["position_at_t_star"] = [float(sy[1]), float(sy[2])], [float(sy[3]), float(sy[4])]
        rec = dict(event_id=catalogue_event, match_id=int(match_id), t_star_s=float(sy[0]), d_min_m=float(sy[5]), t_start_s=None, t_end_s=None,
                   midpoint=[float(sy[1] + sy[3]) / 2, float(sy[2] + sy[4]) / 2], player_a=sides["a"], player_b=sides["b"])
        cat = "7 2D crossing at different times (rejected)"
        title = (f"{match_id} | {cat} | catalogue {catalogue_event} | A at P {cat_rec['time_a_s']:.3f} s, B at P {cat_rec['time_b_s']:.3f} s "
                 f"(dt {cat_rec['time_b_s'] - cat_rec['time_a_s']:+.3f} s) | closest same-time distance in +-2.5 s: {sy[5]:.2f} m > {cfg.primary_m:g} m")
        blobs[f"fig-7-{catalogue_event}.png"] = figure(dict(rec, t_end_s=-1.0, t_start_s=0.0), tr, title, cfg.primary_m)
        meta.append(dict(found=True, crossing_point=[cat_rec["x"], cat_rec["y"]], time_a_at_P_s=cat_rec["time_a_s"], time_b_at_P_s=cat_rec["time_b_s"],
                         **example_record(rec, tr, cat, "geometric-catalogue crossing (v1 detector) with |dt| ~ 2 s; closest same-time approach shown")))
    blobs["examples.json"] = json.dumps(dict(schema=SCHEMA, match_id=int(match_id), threshold_m=cfg.primary_m, examples=meta), indent=1).encode()
    return blobs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m football_fatigue.intersections.proximity")
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--matches", nargs="+", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cross-match-only", action="store_true")
    ap.add_argument("--examples", action="store_true", help="advisor examples of the (single) match")
    ap.add_argument("--catalogue-event", default=None, help="with --examples: a geometric-catalogue crossing to show as rejected")
    ap.add_argument("--jobs", type=int, default=1, help="matches processed in parallel (about 1.8 GB each)")
    a = ap.parse_args(argv)
    root = str(Path(a.root).resolve())
    try:
        if a.examples:
            validate.check_split(json.loads((approved_output_root(root) / ml.SPLITS_DIR / "splits.json").read_text(encoding="utf-8")),
                                 [a.matches[0]], "development")
            out = approved_output_root(root) / SCHEMA / f"examples-{a.matches[0]}"
            if out.exists():
                raise IngestionError(f"{out} already exists and is never overwritten")
            blobs = examples(root, a.matches[0], a.catalogue_event)
            if not a.dry_run:
                write_retry(out, blobs)
            print(out)
            return 0
        if a.cross_match_only:
            print(json.dumps(cross_match(root, a.matches, a.dry_run)["totals"], indent=1))
            return 0
        from concurrent.futures import ProcessPoolExecutor
        failed = 0
        with ProcessPoolExecutor(max_workers=max(1, a.jobs)) as ex:     # each match is independent; outputs are written atomically
            for r in ex.map(_run_one, [(root, mid, a.dry_run) for mid in a.matches]):
                failed += r["status"] == "failed"
                print(json.dumps(r), flush=True)
        return 1 if failed else 0
    except IngestionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
