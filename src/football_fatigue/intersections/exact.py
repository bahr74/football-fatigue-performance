"""Exact spatiotemporal intersections of player trajectories (M11): ``m11-exact-spatiotemporal-v1``.

Requirement: a genuine spatiotemporal intersection A(t*) = B(t*) of two timestamped 2D trajectories. There is no temporal or spatial
tolerance, no speed, angle, noise-band or smoothing criterion, and no confidence cutoff. A 2D crossing at different times does not count.

**Trajectory representation** (documented, the only modelling choice):

- each player (provider slot) is the piecewise-linear interpolation of its RAW positions at its *update instants*; the provider
  repeats each position for two 29.97 fps frames, and repeats are not vertices;
- consecutive update instants are joined by the Step 1 *valid* steps only: same period, at most 0.5 s unobserved, not an
  implausible jump (> 12 m/s and > 1 m), not touching an unattributed or ambiguous sample. Nothing is joined across
  substitutions, periods or gaps;
- unlike v1, samples whose position is bit-identical to another player's are **kept**: such samples are exact coincidences, and
  they are reported (and flagged), not removed;
- arithmetic is exact: every RAW coordinate is a 3-decimal number (an integer number of millimetres) and every timestamp a
  6-decimal number (an integer number of microseconds); both are checked.

**Solution.** On a time interval where both players move linearly, D(t) = A(t) - B(t) is linear, so D(t*) = 0 is two equations in
one unknown. With D0, D1 the differences at the interval's ends:

- D0 = D1 = 0: the players coincide over the whole interval (a *coincident interval*, kept as one interval, not infinitely many points);
- D0 = 0 or D1 = 0: they coincide at the interval's end instant (a *sample-instant point*);
- cross(D0, D1) = 0 and D0 . D1 < 0: one interior point, s* = D0 . (D0 - D1) / |D0 - D1|^2 (an *interior point*);
- otherwise there is no solution in that interval.

Update instants are almost always shared by all players (*synchronous* intervals: identical [t0, t1]), where the test runs in
integers. Where one player's segment overlaps another's only partly, the positions at the overlap's ends are interpolated. A
float pre-screen whose bound (1e-9 m^2) exceeds the float error by orders of magnitude selects candidates, and every candidate
is decided again with exact rational arithmetic. The pre-screen can only drop a pair that cannot be a solution; it never
accepts one.

**Windows** are centred on every integer game second of each period (the period's first frame is second 0). Each spans
[c - 2.5, c + 2.5], clipped to the period, so edge windows are partial and every instant is covered. Every pair of players
available in a window is a pair-window comparison. A point solution at t* is detected by every window with |t* - c| <= 2.5,
and an interval by every window it overlaps. Deduplication uses the exact key (pair, t*, P) for points and maximal contiguous
intervals for coincidences. Different events of one pair are never merged. Restricting each pair's trajectories to a window
does not change positions, so this is identical to solving all pairs within every window (checked by a brute-force test).

Run: ``python -m football_fatigue.intersections.exact --matches 3812 [...] [--dry-run]``.
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
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from football_fatigue.encounters import quality as Q
from football_fatigue.encounters.trajectories import STATE_AVAILABLE, RawCache
from football_fatigue.ingestion import IngestionError, approved_output_root
from football_fatigue.matching import ml, validate
from football_fatigue.matching.benchmark import jsonl, load_source

from . import audit as AU
from .__main__ import peak_memory_mb
from .advisor import run_bounds
from .catalogue import load_steps, write_retry
from .detect import UNUSABLE, duplicate_positions

SCHEMA = "m11-exact-spatiotemporal-v1"
MM, US = 1000, 1_000_000
HALF_US = 2_500_000
SCREEN = 1e-9                                  # float pre-screen bound (m^2); exact arithmetic decides
DESCRIPTIVE_M = (0.001, 0.01, 0.1, 0.5)        # closest-approach scales for the descriptive near-meeting counts only


# ---------------------------------------------------------------- trajectories

def as_int(v: np.ndarray, scale: int, what: str, float_noise: float = 0.0) -> np.ndarray:
    """Exact integer units (mm, us). Raises if a value is not the float representation of a whole number of units: coordinates
    are parsed directly and must match exactly; timestamps are ``videoTimeMs / 1000``, whose float division leaves at most ~1e-6
    unit of representation noise, so ``float_noise`` (1e-3 unit) only absorbs that representation, never a physical tolerance."""
    r = np.round(v * scale)
    if not np.all((r / scale == v) | (np.abs(v * scale - r) <= float_noise)):
        raise IngestionError(f"{what}: values are not whole multiples of 1/{scale}")
    return r.astype(np.int64)


def trajectories(c: RawCache, mask: np.ndarray, steps: dict[str, np.ndarray], join_jumps: bool = False) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Valid segments (see the module docstring) with exact integer coordinates (mm) and times (us).

    Held positions: a run of equal samples longer than the provider's two-frame repetition (Step 1 ``HELD_POSITION``, L >= 3
    frames) is a stationary player (or a provider freeze). The player is represented as stationary from the run's first frame to
    its second-to-last frame (its last 15 Hz instant), then moving linearly to the next update instant. Joining the update
    instants directly, as v1 does, would spread the movement over the whole hold and misplace the time. Two-frame runs add no
    vertex. A held run is kept even when no valid step follows it."""
    _rep, upd, _held = Q.cadence(c)
    avail = c.state == STATE_AVAILABLE
    jump = np.zeros_like(steps["jump"]) if join_jumps else steps["jump"]
    usable = ~jump & steps["valid"]
    s_all, a_all, b_all = steps["slot"], steps["row_from"], steps["row_to"]
    unus = ((mask[a_all, s_all] | mask[b_all, s_all]) & UNUSABLE) != 0
    step_of = {(int(s), int(a)): (int(b), bool(u and not n)) for s, a, b, u, n in zip(s_all, a_all, b_all, usable, unus)}
    cols: dict[str, list] = {k: [] for k in ("slot", "row0", "row1", "stationary")}
    counts = dict(steps_total=int(len(s_all)), steps_joined=0, invalid_gap_or_period=int((~steps["valid"]).sum()),
                  implausible_jump=int((steps["valid"] & jump).sum()), unattributed_or_ambiguous=int((usable & unus).sum()),
                  held_runs=0, stationary_segments=0)
    for s in range(c.n_slots):
        rows = np.flatnonzero(avail[:, s])
        if not len(rows):
            continue
        st = np.flatnonzero(upd[rows, s])
        first, last = rows[st], rows[np.r_[st[1:] - 1, len(rows) - 1]]
        for a, la in zip(first.tolist(), last.tolist()):
            start = a
            if (la - a >= 2 and not (mask[a, s] & UNUSABLE) and c.canonical_period[a] > 0
                    and c.canonical_period[a] == c.canonical_period[la - 1]):
                counts["held_runs"] += 1
                counts["stationary_segments"] += 1
                start = la - 1
                cols["slot"].append(s), cols["row0"].append(a), cols["row1"].append(start), cols["stationary"].append(True)
            nxt = step_of.get((s, a))
            if nxt is not None and nxt[1]:
                counts["steps_joined"] += 1
                cols["slot"].append(s), cols["row0"].append(start), cols["row1"].append(nxt[0]), cols["stationary"].append(False)
    sl, r0, r1 = (np.array(cols[k], np.int64) for k in ("slot", "row0", "row1"))
    t = c.video_time_s
    o = np.lexsort((t[r0], sl))
    sl, r0, r1, stat = sl[o], r0[o], r1[o], np.array(cols["stationary"], bool)[o]
    seg = dict(slot=sl, row0=r0, row1=r1, t0=t[r0], t1=t[r1], period=c.canonical_period[r0].astype(int), stationary=stat,
               x0=c.x[r0, sl], y0=c.y[r0, sl], x1=c.x[r1, sl], y1=c.y[r1, sl])
    seg["x1"] = np.where(stat, seg["x0"], seg["x1"])                       # held: the repeated sample equals the first one
    seg["y1"] = np.where(stat, seg["y0"], seg["y1"])
    for k in ("x0", "y0", "x1", "y1"):
        seg[k.upper()] = as_int(seg[k], MM, "coordinates")
    for k in ("t0", "t1"):
        seg[k.upper()] = as_int(seg[k], US, "timestamps", 1e-3)
    counts["segments"] = int(len(sl))
    return seg, counts


# ---------------------------------------------------------------- exact solution of one interval

def classify(d0: tuple, d1: tuple) -> tuple[str, Fraction | None] | None:
    """Exact solution of D(s) = d0 + s (d1 - d0) = 0 on s in [0, 1] (components: ints or Fractions)."""
    z0, z1 = d0 == (0, 0), d1 == (0, 0)
    if z0 and z1:
        return "interval", None
    if z0:
        return "point", Fraction(0)
    if z1:
        return "point", Fraction(1)
    cross = d0[0] * d1[1] - d0[1] * d1[0]
    dot = d0[0] * d1[0] + d0[1] * d1[1]
    if cross == 0 and dot < 0:
        e = (d0[0] - d1[0], d0[1] - d1[1])
        return "point", Fraction(d0[0] * e[0] + d0[1] * e[1]) / Fraction(e[0] * e[0] + e[1] * e[1])
    return None


def _at(seg, g: int, t: Fraction) -> tuple[Fraction, Fraction]:
    """Exact position (mm) of segment ``g`` at time ``t`` (us)."""
    f = (t - seg["T0"][g]) / (int(seg["T1"][g]) - int(seg["T0"][g]))
    return (int(seg["X0"][g]) + f * (int(seg["X1"][g]) - int(seg["X0"][g])),
            int(seg["Y0"][g]) + f * (int(seg["Y1"][g]) - int(seg["Y0"][g])))


def solve_overlap(seg, ga: int, gb: int) -> tuple[str, Fraction, Fraction, tuple[Fraction, Fraction] | None] | None:
    """Exact solution for two segments over their common time interval: (kind, t_lo, t_hi or t*, P) in us / mm."""
    lo, hi = Fraction(max(int(seg["T0"][ga]), int(seg["T0"][gb]))), Fraction(min(int(seg["T1"][ga]), int(seg["T1"][gb])))
    if hi <= lo:
        return None
    a0, a1, b0, b1 = _at(seg, ga, lo), _at(seg, ga, hi), _at(seg, gb, lo), _at(seg, gb, hi)
    r = classify((a0[0] - b0[0], a0[1] - b0[1]), (a1[0] - b1[0], a1[1] - b1[1]))
    if r is None:
        return None
    if r[0] == "interval":
        return "interval", lo, hi, None
    t = lo + r[1] * (hi - lo)
    return "point", t, t, _at(seg, ga, t)


# ---------------------------------------------------------------- vectorised search

def _pairs_by_group(seg, sel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """All within-group index pairs (i < j by slot) for segments ``sel`` grouped by identical (period, T0, T1)."""
    o = sel[np.lexsort((seg["slot"][sel], seg["T1"][sel], seg["T0"][sel], seg["period"][sel]))]
    key = np.c_[seg["period"][o], seg["T0"][o], seg["T1"][o]]
    brk = np.r_[True, np.any(key[1:] != key[:-1], axis=1)]
    starts = np.flatnonzero(brk)
    sizes = np.diff(np.r_[starts, len(o)])
    ia, ib = [], []
    for n in np.unique(sizes[sizes >= 2]):
        st = starts[sizes == n]
        m = o[st[:, None] + np.arange(n)[None]]                       # (groups, n)
        i, j = np.triu_indices(n, 1)
        ia.append(m[:, i].ravel())
        ib.append(m[:, j].ravel())
    return (np.concatenate(ia), np.concatenate(ib)) if ia else (np.zeros(0, int), np.zeros(0, int))


def synchronous(seg, chunk: int = 4_000_000) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    """Exact integer test on every pair of segments with identical time intervals; returns the solving pairs and descriptive
    closest-approach data (pair-intervals within the largest descriptive scale)."""
    ia, ib = _pairs_by_group(seg, np.arange(len(seg["slot"])))
    hits, near = [], []
    for k in range(0, len(ia), chunk):
        a, b = ia[k:k + chunk], ib[k:k + chunk]
        d0x, d0y = seg["X0"][a] - seg["X0"][b], seg["Y0"][a] - seg["Y0"][b]
        d1x, d1y = seg["X1"][a] - seg["X1"][b], seg["Y1"][a] - seg["Y1"][b]
        cross = d0x * d1y - d0y * d1x
        dot = d0x * d1x + d0y * d1y
        z0, z1 = (d0x == 0) & (d0y == 0), (d1x == 0) & (d1y == 0)
        h = z0 | z1 | ((cross == 0) & (dot < 0))
        hits.extend(zip(a[h].tolist(), b[h].tolist()))
        ex, ey = (d1x - d0x).astype(float), (d1y - d0y).astype(float)
        ee = ex * ex + ey * ey
        s = np.where(ee > 0, np.clip(-(d0x * ex + d0y * ey) / np.where(ee > 0, ee, 1.0), 0, 1), 0.0)
        dmin = np.hypot(d0x + s * ex, d0y + s * ey) / MM                 # closest approach on the interval (m), descriptive
        w = dmin <= DESCRIPTIVE_M[-1]
        near.append(np.c_[a[w], b[w], (dmin[w] * MM).round().astype(np.int64)])
    return hits, dict(pairs=int(len(ia)), near=np.concatenate(near) if near else np.zeros((0, 3), np.int64))


def irregular(seg) -> np.ndarray:
    """Segments spanning more than one elementary interval of their period (the union of all players' update instants)."""
    out = np.zeros(len(seg["slot"]), bool)
    for p in np.unique(seg["period"]):
        ix = np.flatnonzero(seg["period"] == p)
        bp = np.unique(np.r_[seg["T0"][ix], seg["T1"][ix]])
        nxt = bp[np.minimum(np.searchsorted(bp, seg["T0"][ix], "right"), len(bp) - 1)]
        out[ix] = seg["T1"][ix] != nxt
    return out


def overlap_pairs(seg) -> tuple[np.ndarray, np.ndarray]:
    """All pairs of segments of different players (same period) whose time intervals overlap but differ (at least one is
    irregular), each pair once. Together with the synchronous groups this is every overlapping pair of segments."""
    irr = irregular(seg)
    per: dict[tuple[int, int], np.ndarray] = {}
    for p in np.unique(seg["period"]):
        for s in np.unique(seg["slot"][seg["period"] == p]):
            per[(int(p), int(s))] = np.flatnonzero((seg["period"] == p) & (seg["slot"] == s))   # time-sorted (segments() order)
    A, B = [], []
    for g in np.flatnonzero(irr):
        p, sa = int(seg["period"][g]), int(seg["slot"][g])
        for (pp, sb), ix in per.items():
            if pp != p or sb == sa:
                continue
            lo = np.searchsorted(seg["T1"][ix], seg["T0"][g], "right")
            hi = np.searchsorted(seg["T0"][ix], seg["T1"][g], "left")
            for h in ix[lo:hi]:
                same = seg["T0"][h] == seg["T0"][g] and seg["T1"][h] == seg["T1"][g]
                if same or (irr[h] and sb < sa):
                    continue                                              # identical: synchronous path; both irregular: once
                A.append(g)
                B.append(int(h))
    return np.array(A, int), np.array(B, int)


def overlapping(seg) -> tuple[list[tuple[int, int]], int]:
    """Pairs of :func:`overlap_pairs` passing a float pre-screen; exact verification follows in :func:`solve_overlap`. Returns
    the screened pairs and the number examined."""
    a, b = overlap_pairs(seg)
    if not len(a):
        return [], 0
    lo = np.maximum(seg["T0"][a], seg["T0"][b]).astype(float)
    hi = np.minimum(seg["T1"][a], seg["T1"][b]).astype(float)

    def pos(g, t):
        f = (t - seg["T0"][g]) / (seg["T1"][g] - seg["T0"][g])
        return seg["X0"][g] + f * (seg["X1"][g] - seg["X0"][g]), seg["Y0"][g] + f * (seg["Y1"][g] - seg["Y0"][g])

    (ax0, ay0), (ax1, ay1), (bx0, by0), (bx1, by1) = pos(a, lo), pos(a, hi), pos(b, lo), pos(b, hi)
    d0x, d0y, d1x, d1y = (ax0 - bx0) / MM, (ay0 - by0) / MM, (ax1 - bx1) / MM, (ay1 - by1) / MM
    cross = d0x * d1y - d0y * d1x
    dot = d0x * d1x + d0y * d1y
    small0, small1 = np.hypot(d0x, d0y) <= SCREEN, np.hypot(d1x, d1y) <= SCREEN
    keep = small0 | small1 | ((np.abs(cross) <= SCREEN) & (dot <= SCREEN))
    return list(zip(a[keep].tolist(), b[keep].tolist())), int(len(a))


# ---------------------------------------------------------------- events

def solutions(seg, pairs: list[tuple[int, int]]) -> list[dict]:
    out = []
    for ga, gb in pairs:
        if seg["slot"][ga] > seg["slot"][gb]:
            ga, gb = gb, ga
        r = solve_overlap(seg, ga, gb)
        if r is not None:
            out.append(dict(kind=r[0], ga=ga, gb=gb, lo=r[1], hi=r[2], p=r[3]))
    return out


def merge_solutions(seg, sols: list[dict]) -> list[dict]:
    """Deduplicate exact solutions into events: identical points are one event; contiguous coincident intervals of a pair are
    one interval; points at an interval's end instants belong to that interval."""
    iv: dict[tuple, list[dict]] = {}
    pts: dict[tuple, dict] = {}
    for s in sols:
        pair = (int(seg["period"][s["ga"]]), int(seg["slot"][s["ga"]]), int(seg["slot"][s["gb"]]))
        if s["kind"] == "interval":
            iv.setdefault(pair, []).append(s)
        else:
            k = pair + (s["lo"], s["p"])
            pts.setdefault(k, dict(s, members=[]))["members"].append((s["ga"], s["gb"]))
    events = []
    covered: dict[tuple, list[tuple[Fraction, Fraction]]] = {}
    for pair, ss in iv.items():
        ss = sorted(ss, key=lambda s: s["lo"])
        cur = None
        for s in ss:
            if cur is not None and s["lo"] <= cur["hi"]:
                cur["hi"] = max(cur["hi"], s["hi"])
                cur["members"].append((s["ga"], s["gb"]))
            else:
                if cur is not None:
                    events.append(cur)
                cur = dict(s, pair=pair, members=[(s["ga"], s["gb"])])
        events.append(cur)
        covered[pair] = [(e["lo"], e["hi"]) for e in events if e.get("pair") == pair]
    for k, s in pts.items():
        pair = k[:3]
        if any(lo <= s["lo"] <= hi for lo, hi in covered.get(pair, [])):
            continue
        events.append(dict(s, pair=pair))
    return sorted(events, key=lambda e: (e["pair"][0], e["lo"], e["pair"][1], e["pair"][2]))


def period_bounds(seg, c: RawCache) -> dict[int, tuple[int, int]]:
    out = {}
    for p in np.unique(seg["period"]):
        t = c.video_time_s[c.canonical_period == p]
        out[int(p)] = (int(as_int(np.array([t.min()]), US, "timestamps", 1e-3)[0]), int(as_int(np.array([t.max()]), US, "timestamps", 1e-3)[0]))
    return out


def windows_of(lo: Fraction, hi: Fraction, start: int, end: int) -> list[int]:
    """Integer game seconds k whose window [c - 2.5 s, c + 2.5 s] (c = start + k s, c <= end) contains / overlaps [lo, hi]."""
    k0 = max(0, math.ceil((lo - start - HALF_US) / US))
    k1 = math.floor((hi - start + HALF_US) / US)
    return [k for k in range(k0, k1 + 1) if start + k * US <= end]


def window_accounting(seg, bounds: dict[int, tuple[int, int]]) -> dict[str, Any]:
    """Windows per period, players available per window (any valid segment overlapping the window) and pair-window comparisons."""
    out, n_av = {}, []
    for p, (start, end) in bounds.items():
        K = np.arange(0, (end - start) // US + 1)
        c = start + K * US
        lo, hi = np.maximum(c - HALF_US, start), np.minimum(c + HALF_US, end)
        n = np.zeros(len(K), np.int64)
        for s in np.unique(seg["slot"][seg["period"] == p]):
            ix = np.flatnonzero((seg["period"] == p) & (seg["slot"] == s))
            first = np.searchsorted(seg["T1"][ix], lo, "right")           # first segment ending after the window starts
            ok = first < len(ix)
            ok[ok] &= seg["T0"][ix][first[ok]] < hi[ok]
            n += ok
        n_av.append(n)
        out[str(p)] = dict(windows=int(len(K)), partial_windows=int(((c - HALF_US) < start).sum() + ((c + HALF_US) > end).sum()),
                           pair_window_comparisons=int((n * (n - 1) // 2).sum()))
    allv = np.concatenate(n_av) if n_av else np.zeros(0, int)
    return dict(periods=out, windows=int(len(allv)), theoretical_pair_windows_22_players=231 * int(len(allv)),
                pair_window_comparisons=int((allv * (allv - 1) // 2).sum()),
                available_players_per_window=dict(min=int(allv.min()), median=float(np.median(allv)), max=int(allv.max())) if len(allv) else None)


def near_episodes(seg, near: np.ndarray) -> dict[str, int]:
    """Descriptive only (never a selection): distinct same-time near-meetings, i.e. runs of consecutive synchronous intervals of
    one pair whose closest approach is at most r, for several r."""
    out = {}
    if not len(near):
        return {f"{r:g}m": 0 for r in DESCRIPTIVE_M}
    a, b, d = near[:, 0], near[:, 1], near[:, 2]
    for r in DESCRIPTIVE_M:
        w = d <= round(r * MM)
        aa, bb = a[w], b[w]
        key = np.c_[seg["period"][aa], seg["slot"][aa], seg["slot"][bb], seg["T0"][aa], seg["T1"][aa]]
        key = key[np.lexsort(key.T[::-1])]
        new = np.r_[True, np.any(key[1:, :3] != key[:-1, :3], axis=1) | (key[1:, 3] != key[:-1, 4])]
        out[f"{r:g}m"] = int(new.sum())
    return out


def _fs(v: Fraction, scale: int) -> str:
    """Exact rational in base units as 'num/den' of (scale) units."""
    return f"{v.numerator}/{v.denominator}" if v.denominator != 1 else str(v.numerator)


def record(c: RawCache, mask: np.ndarray, seg, e: dict, bounds, dup: np.ndarray) -> dict[str, Any]:
    p, sa, sb = e["pair"]
    ga, gb = e["ga"], e["gb"]
    start, end = bounds[p]
    wins = windows_of(e["lo"], e["hi"], start, end)
    mid = (e["lo"] + e["hi"]) / 2
    canon = min(wins, key=lambda k: (abs(start + k * US - mid), k))

    def segd(g, slot):
        r0, r1 = int(seg["row0"][g]), int(seg["row1"][g])
        return dict(row0=r0, row1=r1, frame0=int(c.frame_num[r0]), frame1=int(c.frame_num[r1]), t0=float(seg["t0"][g]), t1=float(seg["t1"][g]),
                    x0=float(seg["x0"][g]), y0=float(seg["y0"][g]), x1=float(seg["x1"][g]), y1=float(seg["y1"][g]),
                    visibility=[int(c.visibility[r0, slot]), int(c.visibility[r1, slot])],
                    quality_bits=[int(mask[r0, slot]), int(mask[r1, slot])], duplicate_position=[bool(dup[r0, slot]), bool(dup[r1, slot])])

    sync = seg["T0"][ga] == seg["T0"][gb] and seg["T1"][ga] == seg["T1"][gb]
    if e["kind"] == "point":
        s_a = (e["lo"] - int(seg["T0"][ga])) / (int(seg["T1"][ga]) - int(seg["T0"][ga]))
        at_sample = e["lo"] in (Fraction(int(seg["T0"][ga])), Fraction(int(seg["T1"][ga])), Fraction(int(seg["T0"][gb])), Fraction(int(seg["T1"][gb])))
        geo = dict(t_s=float(e["lo"] / US), t_us_exact=_fs(e["lo"], US), x=float(e["p"][0] / MM), y=float(e["p"][1] / MM),
                   x_mm_exact=_fs(e["p"][0], MM), y_mm_exact=_fs(e["p"][1], MM), segment_parameter_a=float(s_a), at_sample_instant=bool(at_sample))
    else:
        geo = dict(t_start_s=float(e["lo"] / US), t_end_s=float(e["hi"] / US), duration_s=float((e["hi"] - e["lo"]) / US),
                   intervals=len(e["members"]))
    key = f"{c.match_id}|{p}|{sa}|{sb}|{e['kind']}|{_fs(e['lo'], US)}"
    return dict(event_id=hashlib.sha256(key.encode()).hexdigest()[:16], event_key=key, match_id=int(c.match_id), period=int(p), kind=e["kind"],
                slot_a=int(sa), slot_b=int(sb), side_a=str(c.slot_side[sa]), side_b=str(c.slot_side[sb]), jersey_a=int(c.slot_jersey[sa]),
                jersey_b=int(c.slot_jersey[sb]), player_id_a=int(c.slot_player_id[sa]), player_id_b=int(c.slot_player_id[sb]),
                team_relation="same_team" if c.slot_side[sa] == c.slot_side[sb] else "opposite_teams", **geo,
                synchronous_samples=bool(sync), segment_a=segd(ga, sa), segment_b=segd(gb, sb), detections_merged=len(e["members"]),
                windows_game_s=wins, n_windows=len(wins), canonical_window_game_s=canon, period_start_s=start / US,
                v1_duplicate_position_rule_would_exclude=bool(dup[int(seg["row0"][ga]), sa] or dup[int(seg["row1"][ga]), sa]
                                                              or dup[int(seg["row0"][gb]), sb] or dup[int(seg["row1"][gb]), sb]))


# ---------------------------------------------------------------- trajectory export

def export(c: RawCache, mask: np.ndarray, seg, rec: dict, ga: int, gb: int) -> dict[str, np.ndarray]:
    """Both players' original samples in the event's canonical window (clipped to it, on the contiguous valid run containing the
    event), split into before / during / after, with the exact event point added as the boundary vertex."""
    cw = rec["period_start_s"] + rec["canonical_window_game_s"]
    lo, hi = cw - 2.5, cw + 2.5
    t_a, t_b = (rec["t_s"], rec["t_s"]) if rec["kind"] == "point" else (rec["t_start_s"], rec["t_end_s"])
    out = {}
    for side, g, slot in (("a", ga, rec["slot_a"]), ("b", gb, rec["slot_b"])):
        i, j = run_bounds(seg, g, lo, hi)
        rows = np.r_[seg["row0"][i:j + 1], seg["row1"][j]]
        t = c.video_time_s[rows]
        v = np.c_[c.x[rows, slot], c.y[rows, slot], t, rows, c.visibility[rows, slot], mask[rows, slot]]
        before, during, after = v[t < t_a], v[(t >= t_a) & (t <= t_b)], v[t > t_b]
        if rec["kind"] == "point":
            pt = np.array([[rec["x"], rec["y"], rec["t_s"], -1, -1, -1]])
            before, after = np.r_[before, pt], np.r_[pt, after]
        out[f"{side}_before"], out[f"{side}_during"], out[f"{side}_after"] = before, during, after
    return out


def store(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    names = ("a_before", "a_during", "a_after", "b_before", "b_during", "b_after")
    ev, part, xs = [], [], []
    for k, p in enumerate(parts):
        for n, name in enumerate(names):
            ev.append(k)
            part.append(n)
            xs.append(p[name])
    lens = np.array([len(v) for v in xs], np.int64)
    v = np.concatenate(xs) if xs else np.zeros((0, 6))
    return dict(event_index=np.array(ev, np.int64), part=np.array(part, np.int8), offsets=np.r_[0, np.cumsum(lens)].astype(np.int64),
                x=v[:, 0], y=v[:, 1], t=v[:, 2], source_row=v[:, 3].astype(np.int64), visibility=v[:, 4].astype(np.int8),
                quality_bits=v[:, 5].astype(np.int64), part_names=np.array(names))


# ---------------------------------------------------------------- figures

def figure(c: RawCache, rec: dict, parts: dict[str, np.ndarray], title: str) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    pos = {}
    for side, col in (("a", "tab:blue"), ("b", "tab:red")):
        b, d, a = parts[f"{side}_before"], parts[f"{side}_during"], parts[f"{side}_after"]
        ax.plot(b[:, 0], b[:, 1], "-", color=col, lw=2, label=f"{side.upper()} before (#{rec['jersey_' + side]})")
        if len(d) > 1:
            ax.plot(d[:, 0], d[:, 1], "-", color="k", lw=4, alpha=0.4, label="coincident" if side == "a" else None)
        ax.plot(a[:, 0], a[:, 1], "--", color=col, lw=2, label=f"{side.upper()} after")
        allv = np.r_[b, d, a]
        pos[side] = allv[np.argsort(allv[:, 2])]
    t_ev = rec["t_s"] if rec["kind"] == "point" else rec["t_start_s"]
    for dt, mk in ((-1.0, "s"), (-0.5, "^"), (0.0, "o"), (0.5, "v"), (1.0, "D")):
        q = []
        for side in ("a", "b"):
            v = pos[side]
            if v[0, 2] <= t_ev + dt <= v[-1, 2]:
                q.append((np.interp(t_ev + dt, v[:, 2], v[:, 0]), np.interp(t_ev + dt, v[:, 2], v[:, 1])))
        if len(q) == 2:
            ax.plot(*zip(*q), ":", color="grey", lw=1)
            ax.plot(*zip(*q), mk, color="k", ms=6, mfc="none", label=f"both players at t*{dt:+.1f} s")
    if rec["kind"] == "point":
        ax.plot(rec["x"], rec["y"], "k*", ms=13, label=f"A(t*) = B(t*), t* = {rec['t_s']:.6f} s")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m, RAW)")
    ax.set_ylabel("y (m, RAW)")
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=7)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------- per match

def detect_exact(seg) -> tuple[list[dict], dict[str, Any]]:
    hits, syn = synchronous(seg)
    screened, examined = overlapping(seg)
    sols = solutions(seg, hits + screened)
    return sols, dict(synchronous_pair_intervals=syn["pairs"], synchronous_solving_pairs=len(hits), overlap_pairs_examined=examined,
                      overlap_pairs_screened=len(screened), exact_solutions=len(sols), near=syn["near"])


def perturbation(seg, seed: int = 0) -> dict[str, int]:
    """Evidence test (counts only): every coordinate moved by an independent uniform offset within +-0.5 mm (inside the 1 mm
    recording resolution); solutions of the synchronous intervals are then counted with the float pre-screen (an upper bound)."""
    rng = np.random.default_rng(seed)
    ia, ib = _pairs_by_group(seg, np.arange(len(seg["slot"])))
    jit = {k: seg[k] / MM + rng.uniform(-0.0005, 0.0005, len(seg[k])) for k in ("X0", "Y0", "X1", "Y1")}
    # one offset per sample: the end of a segment is the start of the next one of the same run
    jit["X1"][:-1] = np.where(seg["row1"][:-1] == seg["row0"][1:], jit["X0"][1:], jit["X1"][:-1])
    jit["Y1"][:-1] = np.where(seg["row1"][:-1] == seg["row0"][1:], jit["Y0"][1:], jit["Y1"][:-1])
    n, exact = 0, 0
    for k in range(0, len(ia), 4_000_000):
        a, b = ia[k:k + 4_000_000], ib[k:k + 4_000_000]
        d0x, d0y = jit["X0"][a] - jit["X0"][b], jit["Y0"][a] - jit["Y0"][b]
        d1x, d1y = jit["X1"][a] - jit["X1"][b], jit["Y1"][a] - jit["Y1"][b]
        cross, dot = d0x * d1y - d0y * d1x, d0x * d1x + d0y * d1y
        cand = (np.hypot(d0x, d0y) <= SCREEN) | (np.hypot(d1x, d1y) <= SCREEN) | ((np.abs(cross) <= SCREEN) & (dot <= SCREEN))
        n += int(cand.sum())
        for i, j in zip(a[cand], b[cand]):                                # exact decision on the perturbed float values
            f = {key: (Fraction(float(jit[key][i])), Fraction(float(jit[key][j]))) for key in ("X0", "Y0", "X1", "Y1")}
            d0 = (f["X0"][0] - f["X0"][1], f["Y0"][0] - f["Y0"][1])
            d1 = (f["X1"][0] - f["X1"][1], f["Y1"][0] - f["Y1"][1])
            exact += classify(d0, d1) is not None
    return dict(pair_intervals=int(len(ia)), screened_candidates=n, exact_solutions=exact)


def spatial_stage(raw_root: str, match_id: str) -> dict[str, Any] | None:
    p = approved_output_root(raw_root) / "m11-occlusion-catalogue-v1" / str(match_id) / "root.summary.json"
    if not p.exists():
        return None
    a = json.loads(p.read_text(encoding="utf-8"))["accounting"]
    return dict(source="m11-occlusion-catalogue-v1 (v1 detector: 2D crossings, half-second centres, duplicate-position samples excluded)",
                pair_windows_with_crossing=a["pair_windows_with_intersection"], window_level_crossings=a["window_segment_pair_hits"],
                unique_crossings=a["events"])


def build(raw_root: str, match_id: str, with_evidence: bool = True) -> tuple[list[dict], dict[str, Any], dict[str, bytes]]:
    c, mask, _enc, src_manifest, prov = load_source(raw_root, match_id)
    steps = load_steps(raw_root, match_id)
    seg, seg_counts = trajectories(c, mask, steps)
    dup = duplicate_positions(c)
    bounds = period_bounds(seg, c)
    sols, st = detect_exact(seg)
    events = merge_solutions(seg, sols)
    recs = [record(c, mask, seg, e, bounds, dup) for e in events]
    parts = [export(c, mask, seg, r, e["ga"], e["gb"]) for r, e in zip(recs, events)]
    near = st.pop("near")
    pts = [r for r in recs if r["kind"] == "point"]
    ivs = [r for r in recs if r["kind"] == "interval"]
    summary = dict(
        schema=SCHEMA, match_id=int(match_id), trajectories=dict(segments=seg_counts, exact_units="mm, us (checked)"),
        windows=window_accounting(seg, bounds),
        stages=dict(spatial_crossings_v1=spatial_stage(raw_root, match_id), **st,
                    window_level_detections=sum(r["n_windows"] for r in recs), unique_events=len(recs),
                    points=len(pts), interior_points=sum(not r["at_sample_instant"] for r in pts),
                    sample_instant_points=sum(r["at_sample_instant"] for r in pts), coincident_intervals=len(ivs),
                    coincident_seconds=round(sum(r["duration_s"] for r in ivs), 3),
                    v1_duplicate_rule_would_exclude=sum(r["v1_duplicate_position_rule_would_exclude"] for r in recs),
                    non_synchronous_events=sum(not r["synchronous_samples"] for r in recs)),
        descriptive_same_time_near_meetings=near_episodes(seg, near),
        source=dict(step1=prov, step1_schema=src_manifest["schema"]))
    if with_evidence:
        summary["evidence"] = dict(perturbation_half_mm=perturbation(seg))
        jseg, _ = trajectories(c, mask, steps, join_jumps=True)
        jsols, _ = detect_exact(jseg)
        summary["evidence"]["implausible_jumps_joined_unique_events"] = len(merge_solutions(jseg, jsols))
    blobs = {"root.trajectories.npz": AU.npz_bytes(store(parts))}
    order = sorted(range(len(recs)), key=lambda k: (recs[k]["kind"] != "point", recs[k]["event_id"]))
    for k in order[:8]:
        r = recs[k]
        what = (f"interior point, segment parameter {r['segment_parameter_a']:.4f}" if r["kind"] == "point" and not r["at_sample_instant"]
                else "point at a sample instant" if r["kind"] == "point" else f"coincident interval {r['duration_s']:.3f} s")
        title = (f"{match_id} event {r['event_id']}: #{r['jersey_a']} ({r['side_a']}) x #{r['jersey_b']} ({r['side_b']}), period {r['period']}; "
                 f"{what}; canonical window game second {r['canonical_window_game_s']}")
        blobs[f"root.fig-{r['kind']}-{r['event_id']}.png"] = figure(c, r, parts[k], title)
    return recs, summary, blobs


def run_match(raw_root: str, match_id: str, dry_run: bool) -> dict[str, Any]:
    t0 = time.perf_counter()
    split = json.loads((approved_output_root(raw_root) / ml.SPLITS_DIR / "splits.json").read_text(encoding="utf-8"))
    validate.check_split(split, [str(match_id)], "development")
    out = approved_output_root(raw_root) / SCHEMA / str(match_id)
    if not dry_run and out.exists():
        raise IngestionError(f"{out} already exists and is never overwritten")
    snap0 = validate.raw_snapshot(raw_root)
    recs, summary, blobs = build(raw_root, match_id)
    summary["seconds"] = round(time.perf_counter() - t0, 1)
    summary["peak_memory_mb"] = peak_memory_mb()
    if dry_run:
        return summary
    blobs = {"root.events.jsonl": jsonl(recs), "root.summary.json": json.dumps(summary, indent=1, sort_keys=True).encode(), **blobs}
    manifest = dict(schema=SCHEMA, match_id=int(match_id), split=split["matches"][str(match_id)],
                    code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))},
                    raw_unchanged=snap0 == validate.raw_snapshot(raw_root), source=summary["source"],
                    outputs_sha256={n: hashlib.sha256(b).hexdigest() for n, b in blobs.items()}, python=platform.python_version(),
                    numpy=np.__version__, selection_thresholds="none", representation="piecewise-linear between update instants; exact mm / us arithmetic",
                    note="Exact spatiotemporal coincidences A(t*) = B(t*) of provider trajectories; tracking-derived, not verified camera "
                         "occlusions; provider slots, not verified identities.")
    blobs["root.manifest.json"] = json.dumps(manifest, indent=1, sort_keys=True).encode()
    write_retry(out, blobs)
    summary["written_to"] = str(out)
    return summary


def cross_match(raw_root: str, matches: list[str], dry_run: bool) -> dict[str, Any]:
    rows = {}
    for mid in matches:
        s = json.loads((approved_output_root(raw_root) / SCHEMA / mid / "root.summary.json").read_text(encoding="utf-8"))
        st, w, sp = s["stages"], s["windows"], s["stages"]["spatial_crossings_v1"] or {}
        rows[mid] = dict(windows=w["windows"], theoretical_pair_windows_22_players=w["theoretical_pair_windows_22_players"],
                         pair_window_comparisons=w["pair_window_comparisons"], v1_window_level_crossings=sp.get("window_level_crossings", 0),
                         v1_unique_crossings=sp.get("unique_crossings", 0), synchronous_pair_intervals=st["synchronous_pair_intervals"],
                         exact_solutions=st["exact_solutions"], window_level_detections=st["window_level_detections"],
                         unique_events=st["unique_events"], interior_points=st["interior_points"], sample_instant_points=st["sample_instant_points"],
                         coincident_intervals=st["coincident_intervals"], v1_duplicate_rule_would_exclude=st["v1_duplicate_rule_would_exclude"],
                         perturbed_exact_solutions=s["evidence"]["perturbation_half_mm"]["exact_solutions"],
                         jumps_joined_unique_events=s["evidence"]["implausible_jumps_joined_unique_events"],
                         **{f"near_{k}": v for k, v in s["descriptive_same_time_near_meetings"].items()})
    tot = {k: sum(r[k] for r in rows.values()) for k in next(iter(rows.values()))}
    rep = dict(schema=SCHEMA, matches=sorted(rows), per_match=rows, totals=tot)
    if not dry_run:
        write_retry(approved_output_root(raw_root) / SCHEMA / "cross-match", {"summary.json": json.dumps(rep, indent=1, sort_keys=True).encode()})
    return rep


def contrast(raw_root: str, match_id: str, event_ids: list[str]) -> dict[str, bytes]:
    """Figures of 2D crossings from the geometric catalogue, with both players' positions marked at the same timestamps: the
    paths cross, but the players are apart at every common instant (written to ``<SCHEMA>/contrast-<match>/``)."""
    c, mask, *_ = load_source(raw_root, match_id)
    seg, _ = trajectories(c, mask, load_steps(raw_root, match_id))
    key = {(int(seg["slot"][g]), int(seg["row0"][g])): g for g in range(len(seg["slot"]))}
    cat = {}
    with open(approved_output_root(raw_root) / "m11-occlusion-catalogue-v1" / str(match_id) / "root.catalogue.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["event_id"] in event_ids:
                cat[r["event_id"]] = r
    blobs = {}
    for eid in event_ids:
        r = cat[eid]
        t_ref = (r["time_a_s"] + r["time_b_s"]) / 2
        rec = dict(kind="contrast", jersey_a=r["jersey_a"], jersey_b=r["jersey_b"], t_start_s=t_ref)
        parts = {}
        for side in ("a", "b"):
            g = key[(r[f"slot_{side}"], r[f"segment_{side}"]["row0"])]
            i, j = run_bounds(seg, g, t_ref - 2.5, t_ref + 2.5)
            rows = np.r_[seg["row0"][i:j + 1], seg["row1"][j]]
            v = np.c_[c.x[rows, r[f"slot_{side}"]], c.y[rows, r[f"slot_{side}"]], c.video_time_s[rows]]
            tp = r[f"time_{side}_s"]
            pt = np.array([[r["x"], r["y"], tp]])
            parts[f"{side}_before"], parts[f"{side}_during"], parts[f"{side}_after"] = np.r_[v[v[:, 2] < tp], pt], np.zeros((0, 3)), np.r_[pt, v[v[:, 2] > tp]]
        title = (f"{match_id} 2D crossing {eid} (geometric catalogue): A at P at {r['time_a_s']:.3f} s, B at P at {r['time_b_s']:.3f} s "
                 f"(dt = {r['time_b_s'] - r['time_a_s']:+.3f} s); markers: both players at the same instants around their mean time")
        blobs[f"fig-{eid}.png"] = figure(c, rec, parts, title)
    return blobs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m football_fatigue.intersections.exact")
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--matches", nargs="+", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cross-match-only", action="store_true")
    ap.add_argument("--contrast", nargs="+", metavar="EVENT_ID", help="figures of these catalogue crossings of the (single) match")
    a = ap.parse_args(argv)
    root = str(Path(a.root).resolve())
    try:
        if a.contrast:
            out = approved_output_root(root) / SCHEMA / f"contrast-{a.matches[0]}"
            if not a.dry_run:
                write_retry(out, contrast(root, a.matches[0], a.contrast))
            print(out)
            return 0
        if a.cross_match_only:
            print(json.dumps(cross_match(root, a.matches, a.dry_run), indent=1))
            return 0
        for mid in a.matches:
            s = run_match(root, mid, a.dry_run)
            print(mid, json.dumps(dict(windows=s["windows"]["windows"], comparisons=s["windows"]["pair_window_comparisons"],
                                       stages=s["stages"], near=s["descriptive_same_time_near_meetings"], evidence=s.get("evidence"),
                                       seconds=s["seconds"])), flush=True)
    except IngestionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
