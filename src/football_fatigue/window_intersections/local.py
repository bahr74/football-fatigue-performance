"""Local descriptors of the v1 window-intersection events and the four-method comparison pilot (M11): ``m11-window-local-methods-v1``.

**Reuse.** The deduplicated primary events of ``m11-window-intersections-v1`` are read, never written, and their SHA-256 is checked
against the v1 manifest. Trajectories are rebuilt from the Step 1 caches with the unchanged v1 quality policy
(``core.build_segments``). The sliding-window scan is not repeated: the primary intersection of a pair-window is chosen without
looking at any rule below, so every method is a filter on the same population.

**Geometry.** Positions are piecewise linear in time between *knots*: the native 15 Hz vertices for ``linear``, and the PCHIP
sub-points (``pchip_substeps`` per native segment) for ``pchip``. This is exactly the geometry on which v1 found the intersections,
so both players lie on P at their arrival times (checked per event). Each player is used only on the continuous valid chain that
contains its intersection segment; nothing is bridged across a gap, a quality cut, a substitution or a period boundary.

**Descriptors** (per event, per variant). P is the intersection, tA and tB the arrival times, t- = min(tA, tB), t+ = max(tA, tB),
and d(t) = |A(t) - B(t)| the synchronized distance, available where both chains exist.

- ``d_at_ta`` / ``d_at_tb``: d at each arrival (how far the other player is when one stands on P).
- Neighbourhood ``arr`` (h): [t- - h, t+ + h]. Neighbourhood ``pt`` (h): [tA - h, tA + h] U [tB - h, tB + h]. Both are clipped to the
  span where both chains exist. For each: the exact minimum of d (closed form per piece, both players linear in time), its time t*,
  the distance of the pair midpoint at t* from P (``mid_to_p``), and whether clipping removed part of the neighbourhood.
- Occupancy (eps): for each player, the maximal time interval around its own arrival during which it stays within eps of P (exact
  roots per piece), searched at most ``occupancy_cap_s`` from the arrival. ``gap`` = max(0, later start - earlier end).
- Velocities on the piece containing each arrival, crossing angle and relative speed.

**Methods** (masks on the descriptors; every method requires the geometric intersection, i.e. is a v1 primary event):

- F1(tau): |tA - tB| <= tau (identical to v1 ``a_tau_*``; checked).
- F2(nb, h, R[, guard]): local minimum of d in neighbourhood nb(h) <= R; with the guard, also ``mid_to_p`` <= rho.
- F3: F1(tau) AND F2(nb, h, R[, guard]).
- F4(eps, tau): occupancy gap <= tau. F4 ``tier``: eps per player from its sample state at the intersection (VISIBLE vs ESTIMATED),
  the measurement-uncertainty interpretation; fixed eps is the physical-region interpretation. eps -> 0 gives F1.

Identities are used only to rebuild the two trajectories and for audit columns; nothing here is a model input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import core as W

SCHEMA = "m11-window-local-methods-v1"
V1_SHARED = "m11-window-intersections-v1"
SEED = 20261113


@dataclass(frozen=True)
class LocalConfig:
    hs_s: tuple[float, ...] = (0.0, 0.10, 0.25, 0.50)        # neighbourhood margins
    radii_m: tuple[float, ...] = (0.25, 0.50, 1.00)          # F2/F3 distance tolerance (0.50, 1.00 are sensitivities)
    taus_s: tuple[float, ...] = (0.10, 0.20, 0.30)           # F1/F3/F4 temporal tolerance
    eps_m: tuple[float, ...] = (0.05, 0.10, 0.25)            # F4 region around P
    rho_m: float = 1.0                                       # locality guard: pair midpoint at t* within rho of P
    occupancy_cap_s: float = 2.5                             # occupancy search limit either side of an arrival
    eps_tier_visible_m: float = 0.05                         # F4 tier: VISIBLE jitter ~1 cm (generous 5 sigma)
    eps_tier_estimated_m: float = 0.10                       # F4 tier: ESTIMATED jitter ~8-10 cm (about 1 sigma, a lower bound)
    primary_h_s: float = 0.10                                # Stage 1 decision (pilot-v2 used 0.25)
    primary_r_m: float = 0.25
    primary_tau_s: float = 0.30
    implausible_speed_m_s: float = 12.0                      # chord speed above this flags implausible motion (configurable)
    chord_half_s: float = 0.2                                # chord speed: |p(t + half) - p(t - half)| / (2 half)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def key(v: float) -> str:
    return f"{v:.2f}".replace(".", "p")


# ---------------------------------------------------------------- piecewise-linear tracks

def clip_track(trk: tuple[np.ndarray, np.ndarray, np.ndarray], lo: float, hi: float) -> tuple[np.ndarray, ...] | None:
    """Restrict a knot track to [lo, hi] (inside its range); the end points are interpolated (position is linear between knots)."""
    T, X, Y = trk
    lo, hi = max(lo, float(T[0])), min(hi, float(T[-1]))
    if hi < lo:
        return None
    if hi == lo:
        Tc = np.array([lo])
    else:
        inner = (T > lo) & (T < hi)
        Tc = np.r_[lo, T[inner], hi]
    return Tc, np.interp(Tc, T, X), np.interp(Tc, T, Y)


def chain_knots(seg, cx, cy, first: int, last: int, lo: float, hi: float, K: int) -> tuple[np.ndarray, ...] | None:
    """Knots of one chain (segments ``first..last``) on [lo, hi] clipped to the chain span. K = 1: native vertices (linear);
    K > 1: PCHIP sub-points, the polyline v1 intersected."""
    lo, hi = max(lo, float(seg["t0"][first])), min(hi, float(seg["t1"][last]))
    if hi < lo:
        return None
    t0, t1 = seg["t0"][first:last + 1], seg["t1"][first:last + 1]
    a = first + int(np.searchsorted(t1, lo, "left"))
    b = first + int(np.searchsorted(t0, hi, "right"))
    g = np.arange(min(a, last), max(b, min(a, last) + 1))
    px, py, pt = W.sub_points(seg, cx, cy, g, K)
    T = np.r_[pt[:, :-1].ravel(), pt[-1, -1]]
    X = np.r_[px[:, :-1].ravel(), px[-1, -1]]
    Y = np.r_[py[:, :-1].ravel(), py[-1, -1]]
    return clip_track((T, X, Y), lo, hi)


def at(trk, t: float) -> tuple[float, float]:
    T, X, Y = trk
    return float(np.interp(t, T, X)), float(np.interp(t, T, Y))


def _inside_interval(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per piece, the interval [u1, u2] of u in [0, 1] where a u^2 + b u + c <= 0 (a >= 0, convex). Empty: u1 > u2."""
    u1, u2 = np.ones_like(a), np.zeros_like(a)
    quad = a > 1e-18
    disc = b * b - 4 * a * c
    ok = quad & (disc >= 0)
    sq = np.sqrt(np.where(ok, disc, 0.0))
    den = np.where(quad, 2 * a, 1.0)
    u1 = np.where(ok, (-b - sq) / den, u1)
    u2 = np.where(ok, (-b + sq) / den, u2)
    lin = ~quad & (np.abs(b) > 1e-18)                        # b u + c <= 0
    r = np.where(lin, -c / np.where(lin, b, 1.0), 0.0)
    u1 = np.where(lin & (b > 0), -np.inf, np.where(lin, r, u1))
    u2 = np.where(lin & (b > 0), r, np.where(lin, np.inf, u2))
    const = ~quad & ~lin
    u1 = np.where(const & (c <= 0), -np.inf, np.where(const, 1.0, u1))
    u2 = np.where(const & (c <= 0), np.inf, np.where(const, 0.0, u2))
    return np.maximum(u1, 0.0), np.minimum(u2, 1.0)


def pair_min(A, B, R: float | None = None) -> tuple[float, float, float | None]:
    """Exact minimum of |A(t) - B(t)| for two knot tracks on the same interval, its time, and (with R) the time with d <= R."""
    TA, XA, YA = A
    TB, XB, YB = B
    T = np.union1d(TA, TB)
    rx = np.interp(T, TA, XA) - np.interp(T, TB, XB)
    ry = np.interp(T, TA, YA) - np.interp(T, TB, YB)
    if len(T) == 1:
        d = float(math.hypot(rx[0], ry[0]))
        return d, float(T[0]), (0.0 if R is not None else None)
    wx, wy = np.diff(rx), np.diff(ry)
    ww = wx * wx + wy * wy
    dot = rx[:-1] * wx + ry[:-1] * wy
    u = np.where(ww > 1e-18, np.clip(-dot / np.where(ww > 1e-18, ww, 1.0), 0.0, 1.0), 0.0)
    d = np.hypot(rx[:-1] + u * wx, ry[:-1] + u * wy)
    i = int(np.argmin(d))
    ts = float(T[i] + u[i] * (T[i + 1] - T[i]))
    dur = None
    if R is not None:
        u1, u2 = _inside_interval(ww, 2 * dot, rx[:-1] ** 2 + ry[:-1] ** 2 - R * R)
        dur = float((np.clip(u2 - u1, 0.0, None) * np.diff(T)).sum())
    return float(d[i]), ts, dur


def occupancy(trk, P: tuple[float, float], t0: float, eps: float) -> tuple[float, float, bool]:
    """Maximal interval containing ``t0`` during which |pos(t) - P| <= eps (exact per piece). ``capped``: the interval reaches the
    end of the available track (search limit or chain end), so its true extent is unknown on that side."""
    lo, hi, clo, chi = occupancy_sides(trk, P, t0, eps)
    return lo, hi, bool(clo or chi)


def occupancy_sides(trk, P: tuple[float, float], t0: float, eps: float) -> tuple[float, float, bool, bool]:
    """``occupancy`` with the cap reported per side: (lo, hi, lo_capped, hi_capped)."""
    T, X, Y = trk
    n = len(T)
    if n < 2:
        return t0, t0, True, True
    dx, dy = X - P[0], Y - P[1]
    wx, wy = np.diff(X), np.diff(Y)
    u1, u2 = _inside_interval(wx * wx + wy * wy, 2 * (dx[:-1] * wx + dy[:-1] * wy), dx[:-1] ** 2 + dy[:-1] ** 2 - eps * eps)
    i = int(np.clip(np.searchsorted(T, t0, "right") - 1, 0, n - 2))
    clo = chi = False
    hi = None
    for j in range(i, n - 1):
        if u2[j] < 1 - 1e-12 or u1[j] > u2[j]:
            hi = max(t0, float(T[j] + max(u2[j], 0.0) * (T[j + 1] - T[j]))) if u1[j] <= u2[j] else max(t0, float(T[j]))
            break
    if hi is None:
        hi, chi = float(T[-1]), True
    lo = None
    for j in range(i, -1, -1):
        if u1[j] > 1e-12 or u1[j] > u2[j]:
            lo = min(t0, float(T[j] + min(u1[j], 1.0) * (T[j + 1] - T[j]))) if u1[j] <= u2[j] else min(t0, float(T[j + 1]))
            break
    if lo is None:
        lo, clo = float(T[0]), True
    return lo, hi, clo, chi


def chord_speeds(trk, half: float) -> np.ndarray:
    """Chord speed at every knot: |p(t + half) - p(t - half)| / span with the chord clipped to the track; NaN where less than one
    half-chord is available. Robust to single-sample jitter, unlike the per-segment speed."""
    T, X, Y = trk
    lo, hi = np.maximum(T - half, T[0]), np.minimum(T + half, T[-1])
    span = hi - lo
    v = np.hypot(np.interp(hi, T, X) - np.interp(lo, T, X), np.interp(hi, T, Y) - np.interp(lo, T, Y)) / np.where(span > 0, span, 1.0)
    return np.where(span >= half - 1e-12, v, np.nan)


def chord_speed_at(trk, t: float, half: float) -> float:
    T = trk[0]
    lo, hi = max(t - half, float(T[0])), min(t + half, float(T[-1]))
    if hi - lo < half - 1e-12:
        return float("nan")
    a, b = at(trk, lo), at(trk, hi)
    return math.hypot(b[0] - a[0], b[1] - a[1]) / (hi - lo)


def max_chord_speed(trk, half: float) -> float:
    if trk is None or len(trk[0]) < 2:
        return float("nan")
    v = chord_speeds(trk, half)
    return float(np.nanmax(v)) if np.isfinite(v).any() else float("nan")


def velocity(trk, t: float) -> np.ndarray:
    T, X, Y = trk
    if len(T) < 2:
        return np.zeros(2)
    i = int(np.clip(np.searchsorted(T, t, "right") - 1, 0, len(T) - 2))
    return np.array([X[i + 1] - X[i], Y[i + 1] - Y[i]]) / (T[i + 1] - T[i])


# ---------------------------------------------------------------- per-event descriptors

def segment_lookup(seg) -> dict[int, tuple[int, np.ndarray]]:
    out = {}
    for s in np.unique(seg["slot"]):
        g = W.slot_slice(seg, int(s))
        out[int(s)] = (int(g[0]), seg["row0"][g])
    return out


def find_segment(lk, seg, slot: int, row0: int, t: float) -> int:
    start, rows = lk[slot]
    i = int(np.searchsorted(rows, row0))
    g = start + i
    if i >= len(rows) or rows[i] != row0 or not (seg["t0"][g] - 1e-9 <= t <= seg["t1"][g] + 1e-9):
        raise LookupError(f"segment of slot {slot} row {row0} not found")
    return g


def _trunc_cause(KA, KB, before: bool) -> str:
    """Which chain limits the common coverage on one side: 'a'/'b' + '_chain_start'/'_chain_end' (both when they coincide)."""
    if before:
        ea, eb, suffix = float(KA[0][0]), float(KB[0][0]), "chain_start"
        who = "a" if ea > eb + 1e-9 else "b" if eb > ea + 1e-9 else "ab"
    else:
        ea, eb, suffix = float(KA[0][-1]), float(KB[0][-1]), "chain_end"
        who = "a" if ea < eb - 1e-9 else "b" if eb < ea - 1e-9 else "ab"
    return f"{who}_{suffix}"


def _neighbourhood(KA, KB, pieces: list[tuple[float, float]], lo_av: float, hi_av: float, P, R: float | None) -> dict[str, Any]:
    """Exact minimum of d over the neighbourhood pieces clipped to the common coverage [lo_av, hi_av]. ``trunc_side``: '', 'before',
    'after' or 'both' (part of the neighbourhood is outside the coverage on that side); ``trunc_cause``: the limiting chain end(s),
    or 'no_common_coverage' when nothing of the neighbourhood is observable."""
    best = (np.inf, np.nan, None)
    dur, empty = 0.0, True
    tb_, ta_ = False, False
    for lo, hi in pieces:
        lo2, hi2 = max(lo, lo_av), min(hi, hi_av)
        tb_ |= lo2 > lo + 1e-9
        ta_ |= hi2 < hi - 1e-9
        if hi2 < lo2:
            continue
        empty = False
        d, ts, du = pair_min(clip_track(KA, lo2, hi2), clip_track(KB, lo2, hi2), R)
        dur += du or 0.0
        if d < best[0]:
            best = (d, ts, du)
    side = "both" if tb_ and ta_ else "before" if tb_ else "after" if ta_ else ""
    if empty:
        return dict(d=np.nan, t=np.nan, mid=np.nan, far=np.nan, trunc=True, dur=np.nan, side=side or "both", cause="no_common_coverage")
    cause = ";".join(c for c, f in ((_trunc_cause(KA, KB, True), tb_), (_trunc_cause(KA, KB, False), ta_)) if f)
    d, ts, _ = best
    a, b = at(KA, ts), at(KB, ts)
    mid = math.hypot((a[0] + b[0]) / 2 - P[0], (a[1] + b[1]) / 2 - P[1])
    far = max(math.hypot(a[0] - P[0], a[1] - P[1]), math.hypot(b[0] - P[0], b[1] - P[1]))
    return dict(d=d, t=ts, mid=mid, far=far, trunc=bool(tb_ or ta_), dur=dur if R is not None else np.nan, side=side, cause=cause)


def event_descriptors(seg, ch, cx, cy, K: int, ev: pd.DataFrame, cfg: LocalConfig) -> pd.DataFrame:
    lk = segment_lookup(seg)
    hmax, cap = max(cfg.hs_s), cfg.occupancy_cap_s
    rows = []
    for e in ev.itertuples(index=False):
        ga = find_segment(lk, seg, e.slot_a, e.seg_a_row0, e.t_a_s)
        gb = find_segment(lk, seg, e.slot_b, e.seg_b_row0, e.t_b_s)
        ka, kb = int(seg["chain"][ga]), int(seg["chain"][gb])
        ta, tb, P = float(e.t_a_s), float(e.t_b_s), (float(e.x_m), float(e.y_m))
        tmin, tmax, te = min(ta, tb), max(ta, tb), (ta + tb) / 2
        KA = chain_knots(seg, cx, cy, int(ch["first"][ka]), int(ch["last"][ka]), min(tmin - hmax, ta - cap), max(tmax + hmax, ta + cap), K)
        KB = chain_knots(seg, cx, cy, int(ch["first"][kb]), int(ch["last"][kb]), min(tmin - hmax, tb - cap), max(tmax + hmax, tb + cap), K)
        pa, pb = at(KA, ta), at(KB, tb)
        r: dict[str, Any] = dict(event_id=e.event_id, reproj_a_m=math.hypot(pa[0] - P[0], pa[1] - P[1]), reproj_b_m=math.hypot(pb[0] - P[0], pb[1] - P[1]))
        lo_av, hi_av = max(KA[0][0], KB[0][0]), min(KA[0][-1], KB[0][-1])
        for tag, t in (("ta", ta), ("tb", tb)):
            if lo_av - 1e-9 <= t <= hi_av + 1e-9:
                a_, b_ = at(KA, t), at(KB, t)
                r[f"d_at_{tag}_m"] = math.hypot(a_[0] - b_[0], a_[1] - b_[1])
            else:
                r[f"d_at_{tag}_m"] = np.nan
        va, vb = velocity(KA, ta), velocity(KB, tb)
        sa, sb = float(np.hypot(*va)), float(np.hypot(*vb))
        r.update(speed_a_m_s=sa, speed_b_m_s=sb, speed_min_m_s=min(sa, sb), rel_speed_m_s=float(np.hypot(*(va - vb))),
                 angle_deg=float(np.degrees(np.arccos(np.clip(va @ vb / (sa * sb), -1, 1)))) if sa * sb > 1e-12 else np.nan,
                 both_chains_cover_arrivals=bool(lo_av <= tmin + 1e-9 and hi_av >= tmax - 1e-9))
        for h in cfg.hs_s:
            for nb, pieces in (("arr", [(tmin - h, tmax + h)]), ("pt", [(ta - h, ta + h), (tb - h, tb + h)])):
                R = cfg.primary_r_m if nb == "arr" and abs(h - cfg.primary_h_s) < 1e-12 else None   # pt pieces may overlap
                o = _neighbourhood(KA, KB, pieces, lo_av, hi_av, P, R)
                p = f"{nb}_h{key(h)}"
                r[f"{p}_d_m"], r[f"{p}_t_rel_s"], r[f"{p}_mid_to_p_m"], r[f"{p}_far_to_p_m"], r[f"{p}_truncated"] = \
                    o["d"], o["t"] - te, o["mid"], o["far"], o["trunc"]
                r[f"{p}_trunc_side"], r[f"{p}_trunc_cause"] = o["side"], o["cause"]
                if R is not None:
                    r[f"{p}_time_within_r_s"] = o["dur"]
        epsA = cfg.eps_tier_estimated_m if e.est_at_a else cfg.eps_tier_visible_m
        epsB = cfg.eps_tier_estimated_m if e.est_at_b else cfg.eps_tier_visible_m
        OA, OB = clip_track(KA, ta - cap, ta + cap), clip_track(KB, tb - cap, tb + cap)
        # an occupancy edge that reaches the searched track is a search-limit cap when the track was cut at t_i +- cap, else a chain end
        lim = {("a", 0): OA[0][0] <= ta - cap + 1e-9, ("a", 1): OA[0][-1] >= ta + cap - 1e-9,
               ("b", 0): OB[0][0] <= tb - cap + 1e-9, ("b", 1): OB[0][-1] >= tb + cap - 1e-9}
        for tag, (ea, eb) in [(f"e{key(x)}", (x, x)) for x in cfg.eps_m] + [("tier", (epsA, epsB))]:
            la, ha, cla, cha = occupancy_sides(OA, P, ta, ea)
            lb, hb, clb, chb = occupancy_sides(OB, P, tb, eb)
            r[f"occ_{tag}_gap_s"] = max(0.0, max(la, lb) - min(ha, hb))
            r[f"occ_{tag}_len_a_s"], r[f"occ_{tag}_len_b_s"], r[f"occ_{tag}_capped"] = ha - la, hb - lb, bool(cla or cha or clb or chb)
            # the gap can only shrink if the earlier-ending interval extends later or the later-starting one extends earlier
            end_who = "a" if ha <= hb else "b"
            start_who = "a" if la >= lb else "b"
            facing = []
            if (cha if end_who == "a" else chb):
                facing.append(f"{end_who}_end_{'search_limit' if lim[(end_who, 1)] else 'chain_end'}")
            if (cla if start_who == "a" else clb):
                facing.append(f"{start_who}_start_{'search_limit' if lim[(start_who, 0)] else 'chain_start'}")
            r[f"occ_{tag}_facing_cap"] = bool(facing)
            r[f"occ_{tag}_facing_cap_cause"] = ";".join(facing)
        r["chord_speed_a_m_s"] = chord_speed_at(KA, ta, cfg.chord_half_s)
        r["chord_speed_b_m_s"] = chord_speed_at(KB, tb, cfg.chord_half_s)
        r["max_chord_speed_a_m_s"] = max_chord_speed(clip_track(KA, tmin - 1.0, tmax + 1.0), cfg.chord_half_s)
        r["max_chord_speed_b_m_s"] = max_chord_speed(clip_track(KB, tmin - 1.0, tmax + 1.0), cfg.chord_half_s)
        vm = np.nan_to_num([r["max_chord_speed_a_m_s"], r["max_chord_speed_b_m_s"]], nan=0.0)
        r["implausible_motion"] = bool((vm > cfg.implausible_speed_m_s).any())
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- methods

def method_masks(ev: pd.DataFrame, D: pd.DataFrame, cfg: LocalConfig) -> pd.DataFrame:
    """Boolean pass column per method configuration (row order = ``ev``)."""
    if not (ev["event_id"].values == D["event_id"].values).all():
        raise ValueError("events and descriptors are not aligned")
    M: dict[str, np.ndarray] = {}
    dt = ev["abs_dt_s"].to_numpy(float)
    f1 = {t: dt <= t + 1e-12 for t in cfg.taus_s}
    for t in cfg.taus_s:
        M[f"F1_tau{key(t)}"] = f1[t]
    for nb in ("arr", "pt"):
        for h in cfg.hs_s:
            d = D[f"{nb}_h{key(h)}_d_m"].to_numpy(float)
            g = D[f"{nb}_h{key(h)}_mid_to_p_m"].to_numpy(float) <= cfg.rho_m + 1e-12
            for R in cfg.radii_m:
                base = np.nan_to_num(d, nan=np.inf) <= R + 1e-12
                for gname, m in (("", base), (f"_g{key(cfg.rho_m)}", base & g)):
                    name = f"{nb}_h{key(h)}_R{key(R)}{gname}"
                    M[f"F2_{name}"] = m
                    for t in cfg.taus_s:
                        M[f"F3_tau{key(t)}_{name}"] = f1[t] & m
    for tag in [f"e{key(x)}" for x in cfg.eps_m] + ["tier"]:
        gap = D[f"occ_{tag}_gap_s"].to_numpy(float)
        for t in cfg.taus_s:
            M[f"F4_{tag}_tau{key(t)}"] = gap <= t + 1e-12
    return pd.DataFrame(M)


FAIL, PASS, UNDETERMINED = 0, 1, 2
STATE_NAMES = {FAIL: "FAIL", PASS: "PASS", UNDETERMINED: "UNDETERMINED"}


def method_states(ev: pd.DataFrame, D: pd.DataFrame, cfg: LocalConfig) -> pd.DataFrame:
    """Three-state outcome per method configuration (int8: 0 FAIL, 1 PASS, 2 UNDETERMINED); same columns as ``method_masks``.

    F1 is never undetermined (both arrivals are observed). F2: PASS when the observed minimum is <= R (the unobserved part can only
    lower the minimum), except with the locality guard on a truncated neighbourhood, where the true arg-min may lie in the unobserved
    part (UNDETERMINED); FAIL only when the whole neighbourhood is observed; otherwise UNDETERMINED. F3: FAIL when F1 fails, else the
    F2 state. F4: PASS when the observed occupancy gap is <= tau (the gap can only shrink); FAIL when the two occupancy ends facing
    each other are not capped; otherwise UNDETERMINED."""
    M = method_masks(ev, D, cfg)
    S: dict[str, np.ndarray] = {}
    f1 = {t: M[f"F1_tau{key(t)}"].to_numpy(bool) for t in cfg.taus_s}
    for t in cfg.taus_s:
        S[f"F1_tau{key(t)}"] = f1[t].astype(np.int8)
    for nb in ("arr", "pt"):
        for h in cfg.hs_s:
            tr = D[f"{nb}_h{key(h)}_truncated"].to_numpy(bool) | D[f"{nb}_h{key(h)}_d_m"].isna().to_numpy()
            for R in cfg.radii_m:
                for gname, guarded in (("", False), (f"_g{key(cfg.rho_m)}", True)):
                    name = f"{nb}_h{key(h)}_R{key(R)}{gname}"
                    m = M[f"F2_{name}"].to_numpy(bool)
                    st = np.where(m, PASS, np.where(tr, UNDETERMINED, FAIL)).astype(np.int8)
                    if guarded:
                        st[m & tr] = UNDETERMINED
                    S[f"F2_{name}"] = st
                    for t in cfg.taus_s:
                        S[f"F3_tau{key(t)}_{name}"] = np.where(f1[t], st, FAIL).astype(np.int8)
    for tag in [f"e{key(x)}" for x in cfg.eps_m] + ["tier"]:
        fc = D[f"occ_{tag}_facing_cap"].to_numpy(bool)
        for t in cfg.taus_s:
            m = M[f"F4_{tag}_tau{key(t)}"].to_numpy(bool)
            S[f"F4_{tag}_tau{key(t)}"] = np.where(m, PASS, np.where(fc, UNDETERMINED, FAIL)).astype(np.int8)
    return pd.DataFrame(S)


def check_states(ev: pd.DataFrame, D: pd.DataFrame, M: pd.DataFrame, S: pd.DataFrame, cfg: LocalConfig) -> dict[str, Any]:
    """PASS implies the mask; a mask pass that is not PASS occurs only for the guarded F2/F3 on a truncated neighbourhood;
    UNDETERMINED occurs only with truncation (F2/F3) or a facing cap (F4); F1 is never UNDETERMINED; F3 = FAIL whenever F1 fails."""
    ok: dict[str, bool] = dict(same_columns=list(M.columns) == list(S.columns))
    p_in_mask = und_has_reason = mask_not_pass_explained = f1_two_state = f3_follows_f1 = True
    for c in M.columns:
        m, s = M[c].to_numpy(bool), S[c].to_numpy()
        p_in_mask &= bool((~(s == PASS) | m).all())
        if c.startswith("F1_"):
            f1_two_state &= bool((s != UNDETERMINED).all())
            continue
        if c.startswith("F4_"):
            tag = c[3:c.rindex("_tau")]
            reason = D[f"occ_{tag}_facing_cap"].to_numpy(bool)
            mask_not_pass_explained &= bool((m <= (s == PASS)).all())
        else:
            nbh = c.split("_")[1:3] if c.startswith("F2_") else c.split("_")[2:4]
            reason = D[f"{nbh[0]}_{nbh[1]}_truncated"].to_numpy(bool) | D[f"{nbh[0]}_{nbh[1]}_d_m"].isna().to_numpy()
            mask_not_pass_explained &= bool((~(m & (s != PASS)) | (reason & c.endswith(f"_g{key(cfg.rho_m)}"))).all())
            if c.startswith("F3_"):
                f3_follows_f1 &= bool(((s == FAIL) | M[f"F1_tau{c[6:10]}"].to_numpy(bool)).all())
        und_has_reason &= bool((~(s == UNDETERMINED) | reason).all())
    ok.update(pass_implies_mask=p_in_mask, undetermined_has_truncation_or_cap=und_has_reason,
              mask_without_pass_only_guard_truncated=mask_not_pass_explained, f1_never_undetermined=f1_two_state, f3_fail_when_f1_fails=f3_follows_f1)
    return dict(ok, all_pass=all(ok.values()))


def primary_columns(cfg: LocalConfig) -> dict[str, str]:
    h, R, t, g = key(cfg.primary_h_s), key(cfg.primary_r_m), key(cfg.primary_tau_s), f"_g{key(cfg.rho_m)}"
    return {
        f"F1 tau {cfg.primary_tau_s:.2f}": f"F1_tau{t}",
        "F1 tau 0.20": "F1_tau0p20",
        f"F2 arr h{cfg.primary_h_s:.2f} R{cfg.primary_r_m:.2f}": f"F2_arr_h{h}_R{R}",
        f"F2 arr h{cfg.primary_h_s:.2f} R{cfg.primary_r_m:.2f} +guard": f"F2_arr_h{h}_R{R}{g}",
        f"F2 pt h{cfg.primary_h_s:.2f} R{cfg.primary_r_m:.2f}": f"F2_pt_h{h}_R{R}",
        f"F2 pt h{cfg.primary_h_s:.2f} R{cfg.primary_r_m:.2f} +guard": f"F2_pt_h{h}_R{R}{g}",
        f"F3 tau {cfg.primary_tau_s:.2f} arr": f"F3_tau{t}_arr_h{h}_R{R}",
        f"F3 tau {cfg.primary_tau_s:.2f} arr +guard": f"F3_tau{t}_arr_h{h}_R{R}{g}",
        f"F3 tau {cfg.primary_tau_s:.2f} pt": f"F3_tau{t}_pt_h{h}_R{R}",
        "F3 tau 0.20 arr +guard": f"F3_tau0p20_arr_h{h}_R{R}{g}",
        **{f"F4 eps {x:.2f} tau {cfg.primary_tau_s:.2f}": f"F4_e{key(x)}_tau{t}" for x in cfg.eps_m},
        f"F4 tier tau {cfg.primary_tau_s:.2f}": f"F4_tier_tau{t}",
    }


def _q(v) -> dict[str, float] | None:
    v = pd.to_numeric(pd.Series(v), errors="coerce").dropna()
    return None if v.empty else {f"p{q}": round(float(np.percentile(v, q)), 4) for q in (5, 25, 50, 75, 95)} | {"n": int(len(v))}


def summarize(ev: pd.DataFrame, D: pd.DataFrame, m: np.ndarray, h_s: float = LocalConfig.primary_h_s) -> dict[str, Any]:
    E, X = ev[m], D[m]
    vc = lambda s: {str(k): int(v) for k, v in s.value_counts().items()}  # noqa: E731
    return dict(events=int(m.sum()), per_match=vc(E["match_id"]), status=vc(E["status"]), tier=vc(E["tier"]),
                accepted=int((E["status"] == "accepted").sum()), sample_ready=int(E["sample_ready"].sum()),
                label_reliable=int(E["label_reliable"].sum()), team_relation=vc(E["team_relation"]),
                third_player_within_2m=int((E["crowd_others_within_2m"] > 0).sum()), abs_dt_s=_q(E["abs_dt_s"]),
                speed_min_m_s=_q(X["speed_min_m_s"]), **{f"arr_h{key(h_s)}_d_m": _q(X[f"arr_h{key(h_s)}_d_m"])},
                extreme_speed_gt_10=int(((X["speed_a_m_s"] > 10) | (X["speed_b_m_s"] > 10)).sum()),
                multi_intersection_window=int((E["max_intersections_in_window"] > 1).sum()))


def overlap_table(M: pd.DataFrame, cols: dict[str, str]) -> list[dict[str, Any]]:
    names = list(cols)
    bits = np.stack([M[cols[n]].to_numpy(bool) for n in names], 1)
    out = []
    for combo, cnt in pd.Series([tuple(r) for r in bits]).value_counts().items():
        if any(combo):
            out.append(dict(members=[n for n, b in zip(names, combo) if b], events=int(cnt)))
    return out


def stratified(ev: pd.DataFrame, D: pd.DataFrame, M: pd.DataFrame, cols: dict[str, str]) -> dict[str, Any]:
    """Pass counts of each primary method by tier, status, sample readiness, speed band and |dt| band."""
    strata = dict(tier=ev["tier"], status=ev["status"], sample_ready=ev["sample_ready"].map({True: "ready", False: "not_ready"}),
                  speed_min=pd.cut(D["speed_min_m_s"], [-0.01, 0.5, 1, 2, 4, 7, 10, np.inf], labels=["<0.5", "0.5-1", "1-2", "2-4", "4-7", "7-10", ">10"]).astype(str),
                  abs_dt=pd.cut(ev["abs_dt_s"], [-0.001, 0.1, 0.2, 0.3, 0.5, 1, 2, 5.01], labels=["<=0.1", "0.1-0.2", "0.2-0.3", "0.3-0.5", "0.5-1", "1-2", "2-5"]).astype(str))
    out = {}
    for sname, s in strata.items():
        out[sname] = {lab: {str(k): int(v) for k, v in s[M[c].to_numpy(bool)].value_counts().sort_index().items()} for lab, c in cols.items()}
        out[sname]["_all_primary_events"] = {str(k): int(v) for k, v in s.value_counts().sort_index().items()}
    return out


def redundancy(ev: pd.DataFrame, D: pd.DataFrame, M: pd.DataFrame, cfg: LocalConfig) -> dict[str, Any]:
    """How much of each spatial condition is implied by the temporal one (and vice versa)."""
    out = {}
    vmin = D["speed_min_m_s"].to_numpy(float)
    h, R, g = key(cfg.primary_h_s), key(cfg.primary_r_m), f"_g{key(cfg.rho_m)}"
    for t in cfg.taus_s:
        f1 = M[f"F1_tau{key(t)}"].to_numpy(bool)
        f2 = M[f"F2_arr_h{h}_R{R}{g}"].to_numpy(bool)
        slow = vmin * t <= cfg.primary_r_m
        out[f"tau{key(t)}"] = dict(
            F1=int(f1.sum()), F1_and_F2=int((f1 & f2).sum()), F2_given_F1=round(float((f1 & f2).sum() / max(1, f1.sum())), 4),
            F1_with_vmin_le_R_over_tau=int((f1 & slow).sum()), of_which_pass_F2=int((f1 & slow & f2).sum()),
            F1_fast_vmin_gt_R_over_tau=int((f1 & ~slow).sum()), of_which_pass_F2_fast=int((f1 & ~slow & f2).sum()),
            F2_given_F1_by_speed_min={lab: round(float((f1 & f2 & sel).sum() / max(1, (f1 & sel).sum())), 4)
                                      for lab, sel in (("<1", vmin < 1), ("1-2", (vmin >= 1) & (vmin < 2)), ("2-4", (vmin >= 2) & (vmin < 4)),
                                                       ("4-7", (vmin >= 4) & (vmin < 7)), (">=7", vmin >= 7))},
            F4_tier_minus_F1=int((M[f"F4_tier_tau{key(t)}"].to_numpy(bool) & ~f1).sum()),
            **{f"F4_e{key(x)}_minus_F1": int((M[f"F4_e{key(x)}_tau{key(t)}"].to_numpy(bool) & ~f1).sum()) for x in cfg.eps_m})
    bound = np.minimum(D["speed_a_m_s"], D["speed_b_m_s"]) * ev["abs_dt_s"]
    d0 = D["arr_h0p00_d_m"]
    ok = d0.notna()
    out["straight_line_bound_d_arr_h0_le_vmin_dt_plus_1cm"] = round(float((d0[ok] <= bound[ok] + 0.01).mean()), 4)
    return out


# ---------------------------------------------------------------- checks

def check_descriptors(ev: pd.DataFrame, D: pd.DataFrame, M: pd.DataFrame, cfg: LocalConfig) -> dict[str, Any]:
    tol = 1e-9
    chk: dict[str, Any] = dict(events=int(len(ev)), reproj_max_m=float(max(D["reproj_a_m"].max(), D["reproj_b_m"].max())))
    chk["reproj_ok"] = chk["reproj_max_m"] <= 1e-6
    chk["f1_equals_v1_a_tau"] = all(bool((M[f"F1_tau{key(t)}"].to_numpy(bool) == ev[f"a_tau_{key(t)}"].to_numpy(bool)).all()) for t in cfg.taus_s)
    mono = True
    for nb in ("arr", "pt"):
        for h0, h1 in zip(cfg.hs_s[:-1], cfg.hs_s[1:]):
            a, b = D[f"{nb}_h{key(h0)}_d_m"], D[f"{nb}_h{key(h1)}_d_m"]
            ok = a.notna() & b.notna()
            mono &= bool((b[ok] <= a[ok] + tol).all()) and bool((a.notna() <= b.notna()).all())
    chk["d_nonincreasing_in_h"] = mono
    sub = True
    for h in cfg.hs_s:
        a, p = D[f"arr_h{key(h)}_d_m"], D[f"pt_h{key(h)}_d_m"]
        ok = a.notna() & p.notna()
        sub &= bool((a[ok] <= p[ok] + tol).all())
    chk["arr_le_pt_same_h"] = sub
    both = D["d_at_ta_m"].notna() & D["d_at_tb_m"].notna()
    mn = np.minimum(D["d_at_ta_m"], D["d_at_tb_m"])
    chk["arr_h0_le_arrival_distances"] = bool((D.loc[both, "arr_h0p00_d_m"] <= mn[both] + tol).all())
    chk["pt_h0_equals_min_arrival_distance"] = bool(np.allclose(D.loc[both, "pt_h0p00_d_m"], mn[both], atol=1e-9))
    gaps = [D[f"occ_e{key(x)}_gap_s"] for x in sorted(cfg.eps_m)]
    chk["occ_gap_nonincreasing_in_eps"] = all(bool((b <= a + tol).all()) for a, b in zip(gaps[:-1], gaps[1:]))
    chk["occ_gap_le_abs_dt"] = all(bool((g <= ev["abs_dt_s"].to_numpy() + tol).all()) for g in gaps + [D["occ_tier_gap_s"]])
    chk["f3_equals_f1_and_f2"] = all(bool((M[c].to_numpy(bool) == (M[f"F1_tau{c[6:10]}"].to_numpy(bool) & M["F2_" + c[11:]].to_numpy(bool))).all())
                                     for c in M.columns if c.startswith("F3_"))
    chk[f"neighbourhood_unavailable_arr_h{key(cfg.primary_h_s)}"] = int(D[f"arr_h{key(cfg.primary_h_s)}_d_m"].isna().sum())
    st = check_states(ev, D, M, method_states(ev, D, cfg), cfg)
    chk["states_consistent"] = st.pop("all_pass")
    chk["states_detail"] = st
    chk["all_pass"] = all(v for k, v in chk.items() if isinstance(v, bool))
    return chk


def dense_check(seg, ch, cx, cy, K: int, ev: pd.DataFrame, D: pd.DataFrame, cfg: LocalConfig, n: int, seed: int, c=None) -> dict[str, Any]:
    """Independent check on a random sample: the knot tracks sampled every 1 ms (not the closed form), for the arr primary-h minimum
    and the eps=0.10 occupancy interval of player A. Trajectory checks: the knot track agrees with the variant curve evaluated
    independently by ``core.positions_at`` (linear: at random times; PCHIP: at the knot times), and (with the raw cache ``c``) every
    native vertex of the track equals the raw Step 1 sample of that row."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ev), min(n, len(ev)), replace=False)
    lk = segment_lookup(seg)
    worst_d, worst_occ, n_d, n_occ = 0.0, 0.0, 0, 0
    worst_curve, worst_raw, n_raw = 0.0, 0.0, 0
    for i in idx:
        e = ev.iloc[i]
        ga = find_segment(lk, seg, int(e.slot_a), int(e.seg_a_row0), float(e.t_a_s))
        gb = find_segment(lk, seg, int(e.slot_b), int(e.seg_b_row0), float(e.t_b_s))
        ka, kb = int(seg["chain"][ga]), int(seg["chain"][gb])
        ta, tb = float(e.t_a_s), float(e.t_b_s)
        tmin, tmax, h = min(ta, tb), max(ta, tb), cfg.primary_h_s
        KA = chain_knots(seg, cx, cy, int(ch["first"][ka]), int(ch["last"][ka]), tmin - h - 3, tmax + h + 3, K)
        KB = chain_knots(seg, cx, cy, int(ch["first"][kb]), int(ch["last"][kb]), tmin - h - 3, tmax + h + 3, K)
        for trk, s_, k_ in ((KA, int(e.slot_a), ka), (KB, int(e.slot_b), kb)):
            # PCHIP: interior knots only; the two clip end points are linear interpolations on the polyline, not cubic samples
            tq = rng.uniform(trk[0][0], trk[0][-1], 20) if K == 1 else trk[0][1:-1]
            px, py = W.positions_at(seg, cx, cy, s_, int(e.period), tq)
            worst_curve = max(worst_curve, float(np.nanmax(np.hypot(px - np.interp(tq, trk[0], trk[1]), py - np.interp(tq, trk[0], trk[2])))),
                              float(np.isnan(px).any()))
            if c is not None:
                g = np.arange(int(ch["first"][k_]), int(ch["last"][k_]) + 1)
                g = g[(seg["t0"][g] >= trk[0][0]) & (seg["t0"][g] <= trk[0][-1])]
                rx, ry = c.x[seg["row0"][g], s_], c.y[seg["row0"][g], s_]
                worst_raw = max(worst_raw, float(np.max(np.hypot(np.interp(seg["t0"][g], trk[0], trk[1]) - rx,
                                                                   np.interp(seg["t0"][g], trk[0], trk[2]) - ry), initial=0.0)))
                n_raw += len(g)
        lo, hi = max(tmin - h, KA[0][0], KB[0][0]), min(tmax + h, KA[0][-1], KB[0][-1])
        if hi >= lo:
            tt = np.r_[np.arange(lo, hi, 0.001), hi]
            dd = np.hypot(np.interp(tt, *KA[:2]) - np.interp(tt, *KB[:2]), np.interp(tt, KA[0], KA[2]) - np.interp(tt, KB[0], KB[2]))
            dex = float(D.iloc[i][f"arr_h{key(h)}_d_m"])
            step = float(np.abs(np.diff(dd)).max()) if len(dd) > 1 else 0.0
            worst_d = max(worst_d, dex - dd.min(), dd.min() - dex - step)   # exact <= sampled min <= exact + one sampling step
            n_d += 1
        P, eps = (float(e.x_m), float(e.y_m)), 0.10
        lo_s, hi_s = max(ta - cfg.occupancy_cap_s, KA[0][0]), min(ta + cfg.occupancy_cap_s, KA[0][-1])
        # 1 ms grid plus the knots: distance to P is convex on each piece, so a brief exit can only straddle a knot
        tt = np.union1d(np.arange(lo_s, hi_s, 0.001), KA[0][(KA[0] >= lo_s) & (KA[0] < hi_s)])
        inside = np.hypot(np.interp(tt, KA[0], KA[1]) - P[0], np.interp(tt, KA[0], KA[2]) - P[1]) <= eps
        j = int(np.argmin(np.abs(tt - ta)))
        if len(tt) and inside[j]:
            a_ = j
            while a_ > 0 and inside[a_ - 1]:
                a_ -= 1
            b_ = j
            while b_ < len(tt) - 1 and inside[b_ + 1]:
                b_ += 1
            lo_e, hi_e, _ = occupancy(chain_knots(seg, cx, cy, int(ch["first"][ka]), int(ch["last"][ka]), ta - cfg.occupancy_cap_s,
                                                  ta + cfg.occupancy_cap_s, K), P, ta, eps)
            worst_occ = max(worst_occ, abs(tt[a_] - lo_e), abs(tt[b_] - hi_e))
            n_occ += 1
    return dict(sampled=int(len(idx)), distance_checked=n_d, distance_worst_violation_m=float(worst_d), distance_ok=bool(worst_d <= 1e-6),
                occupancy_checked=n_occ, occupancy_worst_endpoint_error_s=float(worst_occ), occupancy_ok=bool(worst_occ <= 0.0011),
                curve_worst_m=float(worst_curve), curve_ok=bool(worst_curve <= 1e-9), raw_vertices_checked=n_raw,
                raw_worst_m=float(worst_raw) if c is not None else None, raw_ok=bool(worst_raw <= 1e-9) if c is not None else None)


# ---------------------------------------------------------------- linear vs PCHIP

def variant_agreement(evL: pd.DataFrame, ML: pd.DataFrame, evP: pd.DataFrame, MP: pd.DataFrame, cols: dict[str, str]) -> dict[str, Any]:
    pidx = pd.Series(np.arange(len(evP)), index=evP["event_id"].values)
    matched = evL["pchip_match"].notna().to_numpy()
    j = pidx.reindex(evL["pchip_match"].where(evL["pchip_match"].notna(), "__none__")).to_numpy()
    out = {}
    for lab, c in cols.items():
        lp, pp = ML[c].to_numpy(bool), MP[c].to_numpy(bool)
        pj = np.zeros(len(evL), bool)
        ok = matched & ~np.isnan(j)
        pj[ok] = pp[j[ok].astype(int)]
        out[lab] = dict(linear_pass=int(lp.sum()), pchip_pass=int(pp.sum()), matched_both_pass=int((lp & ok & pj).sum()),
                        matched_linear_only=int((lp & ok & ~pj).sum()), matched_pchip_only=int((~lp & ok & pj).sum()),
                        linear_pass_unmatched=int((lp & ~ok).sum()),
                        membership_flip_share_of_linear=round(float((lp & ok & ~pj).sum() + (~lp & ok & pj).sum()) / max(1, lp.sum()), 4))
    return out


# ---------------------------------------------------------------- main

def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------- resumable per-match runner (atomic, hashed)

class SimulatedCrash(RuntimeError):
    """Raised by ``run_matches`` after ``crash_after`` matches (crash / resume testing)."""


def atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def verify_match_dir(d: Path) -> dict[str, Any]:
    """Re-hash every file listed in ``match_done.json``; any mismatch, missing or unlisted file is an error. Returns the summary."""
    done = json.loads((d / "match_done.json").read_text(encoding="utf-8"))
    present = {str(p.relative_to(d)).replace("\\", "/") for p in d.rglob("*") if p.is_file()} - {"match_done.json"}
    if present != set(done["files_sha256"]):
        raise RuntimeError(f"{d}: files differ from match_done.json (extra {sorted(present - set(done['files_sha256']))}, "
                           f"missing {sorted(set(done['files_sha256']) - present)})")
    for rel, h in done["files_sha256"].items():
        if _sha(d / rel) != h:
            raise RuntimeError(f"{d / rel}: SHA-256 does not match match_done.json")
    return done["summary"]


def run_matches(out: Path, matches: list[str], fn, crash_after: int | None = None, log=print) -> dict[str, dict[str, Any]]:
    """Process matches one at a time into ``out/matches/<m>/``. Work happens in ``<m>.partial`` and the directory is renamed only
    after every file and ``match_done.json`` (SHA-256 of each file, written last) are on disk, so a crash leaves either a complete
    verified match or a partial directory that the next resume discards and recomputes. Completed matches are verified, not redone."""
    from .__main__ import dumps
    root = out / "matches"
    root.mkdir(parents=True, exist_ok=True)
    res: dict[str, dict[str, Any]] = {}
    n_new = 0
    for m in matches:
        fin = root / m
        if fin.exists():
            res[m] = verify_match_dir(fin)
            log(f"{m}: complete, hashes verified (resumed)")
            continue
        part = root / f"{m}.partial"
        if part.exists():
            shutil.rmtree(part)                      # incomplete work of an interrupted run (never a finished output)
            log(f"{m}: discarded an incomplete partial directory")
        part.mkdir()
        summ = fn(m, part)
        files = sorted(p for p in part.rglob("*") if p.is_file())
        done = dict(match=m, summary=summ, files_sha256={str(p.relative_to(part)).replace("\\", "/"): _sha(p) for p in files})
        atomic_write_text(part / "match_done.json", dumps(done))
        os.replace(part, fin)
        res[m] = json.loads(dumps(summ))
        n_new += 1
        if crash_after is not None and n_new >= crash_after:
            raise SimulatedCrash(f"simulated crash after {n_new} newly completed match(es)")
    return res


# ---------------------------------------------------------------- main

V2_PILOT = "m11-window-local-methods-pilot-v2"
EV_COLS = ["match_id", "period", "event_id", "slot_a", "slot_b", "team_relation", "x_m", "y_m", "t_a_s", "t_b_s", "t_event_s", "abs_dt_s",
           "seg_a_row0", "seg_b_row0", "est_at_a", "est_at_b", "a_tau_0p10", "a_tau_0p20", "a_tau_0p30", "status", "tier", "sample_ready",
           "label_reliable", "crowd_others_within_2m", "max_intersections_in_window", "pchip_match"]


def aggregation_columns(cfg: LocalConfig) -> list[str]:
    """Descriptor columns the aggregate analysis and the figures read (bounded memory: the per-event string cause columns are read only
    for the primary h)."""
    cols = ["event_id", "speed_a_m_s", "speed_b_m_s", "speed_min_m_s", "angle_deg", "d_at_ta_m", "d_at_tb_m", "implausible_motion",
            f"arr_h{key(cfg.primary_h_s)}_t_rel_s"]
    for nb in ("arr", "pt"):
        cols += [f"{nb}_h{key(cfg.primary_h_s)}_trunc_side", f"{nb}_h{key(cfg.primary_h_s)}_trunc_cause"]
        for h in cfg.hs_s:
            cols += [f"{nb}_h{key(h)}_{x}" for x in ("d_m", "mid_to_p_m", "truncated")]
    for tag in [f"e{key(x)}" for x in cfg.eps_m] + ["tier"]:
        cols += [f"occ_{tag}_{x}" for x in ("gap_s", "capped", "facing_cap", "facing_cap_cause")]
    return cols


def cohort_columns(cfg: LocalConfig) -> dict[str, str]:
    """F1-F4 configurations attached to the identity-free candidates (stratification metadata)."""
    h, R, t, g = key(cfg.primary_h_s), key(cfg.primary_r_m), key(cfg.primary_tau_s), f"_g{key(cfg.rho_m)}"
    return {"F1_tau0p30": f"F1_tau{t}", "F1_tau0p20": "F1_tau0p20", "F2_primary": f"F2_arr_h{h}_R{R}{g}", "F2_R0p50": f"F2_arr_h{h}_R0p50{g}",
            "F3_primary": f"F3_tau{t}_arr_h{h}_R{R}{g}", "F3_tau0p20": f"F3_tau0p20_arr_h{h}_R{R}{g}", "F4_tier": f"F4_tier_tau{t}",
            "F4_e0p25": f"F4_e0p25_tau{t}"}


def regression_v2(v2: Path, v2man: dict, m: str, v: str, ev: pd.DataFrame, D: pd.DataFrame, M: pd.DataFrame) -> dict[str, Any]:
    """Agreement of the recomputed descriptors and masks with pilot-v2 (hash-checked) on every column the two share."""
    out: dict[str, Any] = {}
    for kind, X in (("descriptors", D), ("method_masks", M)):
        rel = f"descriptors/{m}/{kind}_{v}.parquet"
        if _sha(v2 / rel) != v2man["outputs_sha256"][rel]:
            raise RuntimeError(f"pilot-v2 file {rel} does not match its manifest")
        Y = pd.read_parquet(v2 / rel)
        if kind == "method_masks":
            Y = Y.drop(columns=["event_id", "match_id"])
        elif not (Y["event_id"].values == X["event_id"].values).all():
            raise RuntimeError(f"{m} {v}: event order differs from pilot-v2")
        common = [c for c in Y.columns if c in X.columns]
        bad = []
        for c in common:
            a, b = X[c].to_numpy(), Y[c].to_numpy()
            if a.dtype.kind == "f" or b.dtype.kind == "f":
                same = np.allclose(a.astype(float), b.astype(float), rtol=0, atol=1e-12, equal_nan=True)
            else:
                same = bool((a == b).all())
            if not same:
                bad.append(c)
        out[kind] = dict(rows=int(len(Y)), common_columns=len(common), mismatched_columns=bad, only_in_v2=[c for c in Y.columns if c not in X.columns],
                         new_in_v3=int(sum(c not in Y.columns for c in X.columns)))
    out["identical_on_common_columns"] = not out["descriptors"]["mismatched_columns"] and not out["method_masks"]["mismatched_columns"]
    return out


class LazyCaches:
    """Step 1 caches for the figure gallery, loaded on demand and one match at a time (bounded memory)."""

    def __init__(self, loader):
        self._loader, self._m, self._v = loader, None, None

    def __getitem__(self, m: str) -> dict[str, Any]:
        if m != self._m:
            self._v, self._m = None, None
            self._v, self._m = self._loader(m), m
        return self._v


def state_counts(S: pd.DataFrame, cols: dict[str, str]) -> dict[str, dict[str, int]]:
    return {lab: {STATE_NAMES[k]: int((S[c].to_numpy() == k).sum()) for k in (PASS, FAIL, UNDETERMINED)} for lab, c in cols.items()}


def cause_counts(D: pd.DataFrame, cfg: LocalConfig) -> dict[str, Any]:
    vc = lambda s: {str(k): int(v) for k, v in s.value_counts().items()}  # noqa: E731
    h = key(cfg.primary_h_s)
    return dict(**{f"{nb}_h{h}_trunc_side": vc(D[f"{nb}_h{h}_trunc_side"].replace("", "none")) for nb in ("arr", "pt")},
                **{f"{nb}_h{h}_trunc_cause": vc(D[f"{nb}_h{h}_trunc_cause"].replace("", "none")) for nb in ("arr", "pt")},
                **{f"occ_{t}_facing_cap_cause": vc(D[f"occ_{t}_facing_cap_cause"].replace("", "none")) for t in [f"e{key(x)}" for x in cfg.eps_m] + ["tier"]},
                implausible_motion=int(D["implausible_motion"].sum()))


def main(argv: list[str] | None = None) -> int:
    from .__main__ import PROC, RAW, code_sha256, dev_matches, dumps, finish_manifest, write_index
    from football_fatigue.intersections.catalogue import load_steps
    from football_fatigue.matching import validate
    from football_fatigue.matching.benchmark import load_source
    import psutil

    from . import candidates as C

    from .split import load_split

    ap = argparse.ArgumentParser(description="M11 Stage 1/2: three-state four-method descriptors and identity-free candidates "
                                             "(reads v1 events; writes a new versioned directory, resumable)")
    ap.add_argument("--matches", nargs="+", required=True, help="development match ids, or 'all' for every development match")
    ap.add_argument("--out", default="m11-stage2-candidates-v1")
    ap.add_argument("--split", default=None, help="frozen split directory (e.g. m11-stage2-split-v1); stamps train_cv / heldout")
    ap.add_argument("--variants", nargs="+", default=list(W.VARIANTS), choices=list(W.VARIANTS))
    ap.add_argument("--dense", type=int, default=400, help="events per match and variant for the 1 ms dense check")
    ap.add_argument("--gallery", type=int, default=4, help="example figures per category (0: no gallery)")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run (same configuration; finished matches are verified)")
    ap.add_argument("--simulate-crash-after", type=int, default=None, help="testing: stop abruptly after N newly completed matches")
    ap.add_argument("--no-regression", action="store_true", help="skip the comparison with pilot-v2 (other matches)")
    a = ap.parse_args(argv)
    devs = dev_matches()
    if a.matches == ["all"]:
        a.matches = sorted(devs, key=int)
    bad = [m for m in a.matches if m not in devs]
    if bad:
        raise SystemExit(f"not development matches (refused): {bad}")
    if "linear" not in a.variants:
        raise SystemExit("the linear variant is the reference and is required")
    out = PROC / a.out
    cfg, wcfg, ccfg = LocalConfig(), W.WindowConfig(), C.CandidateConfig()
    split = load_split(PROC, a.split) if a.split else None
    split_of = ({m: "heldout" for m in split["heldout"]} | {m: "train_cv" for m in split["train_cv"]}) if split else {}
    if split and any(m not in split_of for m in a.matches):
        raise SystemExit("a requested match is not in the frozen split")
    run_cfg = dumps(dict(schema=SCHEMA, candidates_schema=C.SCHEMA, matches=a.matches, variants=a.variants, dense=a.dense, gallery=a.gallery,
                         split=a.split, split_heldout=split["heldout"] if split else None,
                         regression=not a.no_regression, local=cfg.as_dict(), window=wcfg.as_dict(), candidates=ccfg.as_dict(),
                         code_sha256=code_sha256()))
    if a.resume:
        if not out.exists():
            raise SystemExit(f"{out} does not exist; nothing to resume")
        if (out / "manifest.json").exists():
            raise SystemExit(f"{out} is complete (manifest.json present); versioned outputs are never overwritten")
        if (out / "run_config.json").read_text(encoding="utf-8") != run_cfg:
            raise SystemExit("run configuration (arguments, config or code) differs from the interrupted run; refusing to mix")
    else:
        if out.exists():
            raise SystemExit(f"{out} exists; versioned outputs are never overwritten (use --resume for an interrupted run)")
        out.mkdir(parents=True)
        atomic_write_text(out / "run_config.json", run_cfg)
    v1 = PROC / V1_SHARED
    v1man = json.loads((v1 / "manifest.json").read_text(encoding="utf-8"))
    v2 = PROC / V2_PILOT
    v2man = json.loads((v2 / "manifest.json").read_text(encoding="utf-8")) if not a.no_regression else None
    proc = psutil.Process()
    snap0 = validate.raw_snapshot(RAW)
    T0 = time.perf_counter()
    cohorts = cohort_columns(cfg)

    def load(m: str):
        c, mask, _e, _sm, _p = load_source(str(RAW), m)
        seg, _cnt, dup = W.build_segments(c, mask, load_steps(str(RAW), m))
        return c, mask, seg, W.chain_table(seg), dup

    def read_v1(m: str, v: str) -> pd.DataFrame:
        rel = f"{m}/events_{v}.parquet"
        if _sha(v1 / rel) != v1man["outputs_sha256"][rel]:
            raise RuntimeError(f"v1 input {rel} does not match the v1 manifest")
        import pyarrow.parquet as pq
        names = set(pq.read_schema(v1 / rel).names)              # pchip_match exists only in the linear file
        return pd.read_parquet(v1 / rel, columns=[c for c in EV_COLS if c in names]).reset_index(drop=True)

    def process(m: str, d: Path) -> dict[str, Any]:
        rt: dict[str, float] = {}
        t = time.perf_counter()
        evs = {v: read_v1(m, v) for v in a.variants}
        rt["read_v1_events_s"] = time.perf_counter() - t
        t = time.perf_counter()
        c, mask, seg, ch, dup = load(m)
        rt["load_step1_and_segments_s"] = time.perf_counter() - t
        summ: dict[str, Any] = dict(checks={}, regression_v2={}, states={})
        keep = {}
        for v in a.variants:
            K = wcfg.pchip_substeps if v == "pchip" else 1
            t = time.perf_counter()
            cx, cy, _sup = W.coefficients(seg, v)
            ev = evs[v]
            D = event_descriptors(seg, ch, cx, cy, K, ev, cfg)
            M = method_masks(ev, D, cfg)
            S = method_states(ev, D, cfg)
            rt[f"descriptors_{v}_s"] = time.perf_counter() - t
            t = time.perf_counter()
            summ["checks"][v] = dict(check_descriptors(ev, D, M, cfg), dense=dense_check(seg, ch, cx, cy, K, ev, D, cfg, a.dense, SEED + int(m), c=c))
            if v2man is not None and f"descriptors/{m}/descriptors_{v}.parquet" in v2man["outputs_sha256"]:
                summ["regression_v2"][v] = regression_v2(v2, v2man, m, v, ev, D, M)
            rt[f"checks_{v}_s"] = time.perf_counter() - t
            summ["states"][v] = state_counts(S, primary_columns(cfg))
            atomic_parquet(D, d / f"descriptors_{v}.parquet")
            atomic_parquet(pd.concat([ev[["event_id", "match_id"]], M], axis=1), d / f"method_masks_{v}.parquet")
            atomic_parquet(pd.concat([ev[["event_id", "match_id"]], S], axis=1), d / f"method_states_{v}.parquet")
            if v == "linear":
                keep = dict(ev=ev, D=D, S=S)
        t = time.perf_counter()
        cand = C.match_candidates(c, mask, dup, seg, ch, ccfg, wcfg)
        cand, cov = C.link_events(cand, keep["ev"], keep["D"], keep["S"], cohorts, ccfg, cfg.primary_h_s)
        cand["split"] = split_of.get(m, "unsplit")
        summ["candidates_checks"] = dict(C.check_candidates(cand, ccfg), coverage=cov)
        atomic_parquet(cand, d / "candidates.parquet")
        rt["candidates_s"] = time.perf_counter() - t
        t = time.perf_counter()
        F, frep = C.framewise_audit(c, mask, dup, seg, cand, ccfg, tuple(c.quarantine_periods))
        summ["framewise"] = frep
        atomic_parquet(F, d / "framewise_audit.parquet")
        rt["framewise_audit_s"] = time.perf_counter() - t
        rt["total_s"] = sum(rt.values())
        summ["runtime_s"] = {k: round(x, 1) for k, x in rt.items()}
        summ["rss_mb_after"] = round(proc.memory_info().rss / 2**20, 1)
        summ["candidates"] = int(len(cand))
        print(f"{m}: {summ['runtime_s']} rss {summ['rss_mb_after']} MB, {len(cand)} candidates", flush=True)
        return summ

    try:
        per_match = run_matches(out, a.matches, process, a.simulate_crash_after)
    except SimulatedCrash as ex:
        print(f"SIMULATED CRASH: {ex}", flush=True)
        os._exit(3)                                           # abrupt: no cleanup, no manifest
    runtime: dict[str, Any] = dict(per_match={m: r["runtime_s"] for m, r in per_match.items()})

    # ---- aggregation (reads the per-match files; selected columns only)
    t = time.perf_counter()
    mdir = out / "matches"
    EV = {v: pd.concat([read_v1(m, v) for m in a.matches], ignore_index=True) for v in a.variants}
    dcols = aggregation_columns(cfg)
    DD = {v: pd.concat([pd.read_parquet(mdir / m / f"descriptors_{v}.parquet", columns=dcols) for m in a.matches], ignore_index=True) for v in a.variants}
    MM = {v: pd.concat([pd.read_parquet(mdir / m / f"method_masks_{v}.parquet").drop(columns=["event_id", "match_id"]) for m in a.matches],
                       ignore_index=True) for v in a.variants}
    SS = {v: pd.concat([pd.read_parquet(mdir / m / f"method_states_{v}.parquet").drop(columns=["event_id", "match_id"]) for m in a.matches],
                       ignore_index=True) for v in a.variants}
    CAND = pd.concat([pd.read_parquet(mdir / m / "candidates.parquet") for m in a.matches], ignore_index=True)
    cols = primary_columns(cfg)
    grid = {v: {c: dict(events=int(MM[v][c].sum()), pass_=int((SS[v][c] == PASS).sum()), undetermined=int((SS[v][c] == UNDETERMINED).sum()),
                        sample_ready=int((MM[v][c] & EV[v]["sample_ready"]).sum())) for c in MM[v].columns} for v in a.variants}
    primary = {v: {lab: summarize(EV[v], DD[v], MM[v][c].to_numpy(bool), cfg.primary_h_s) for lab, c in cols.items()} for v in a.variants}
    main_sets = {k: cols[k] for k in (f"F1 tau {cfg.primary_tau_s:.2f}", f"F2 arr h{cfg.primary_h_s:.2f} R{cfg.primary_r_m:.2f} +guard",
                                      f"F2 pt h{cfg.primary_h_s:.2f} R{cfg.primary_r_m:.2f}", f"F4 eps 0.25 tau {cfg.primary_tau_s:.2f}")}
    hk, Rk, gk, tk = key(cfg.primary_h_s), key(cfg.primary_r_m), f"_g{key(cfg.rho_m)}", key(cfg.primary_tau_s)
    analysis = dict(
        states={v: state_counts(SS[v], cols) for v in a.variants},
        undetermined_causes={v: cause_counts(DD[v], cfg) for v in a.variants},
        overlap={v: overlap_table(MM[v], main_sets) for v in a.variants},
        stratified=stratified(EV["linear"], DD["linear"], MM["linear"], cols),
        redundancy={v: redundancy(EV[v], DD[v], MM[v], cfg) for v in a.variants},
        guard_effect={v: {f"{nb}_h{key(h)}": dict(
            pass_no_guard=int(MM[v][f"F2_{nb}_h{key(h)}_R{Rk}"].sum()),
            removed_by_guard=int((MM[v][f"F2_{nb}_h{key(h)}_R{Rk}"] & ~MM[v][f"F2_{nb}_h{key(h)}_R{Rk}{gk}"]).sum()),
            removed_within_F3=int((MM[v][f"F3_tau{tk}_{nb}_h{key(h)}_R{Rk}"] & ~MM[v][f"F3_tau{tk}_{nb}_h{key(h)}_R{Rk}{gk}"]).sum()),
            neighbourhood_truncated_by_chain=int(DD[v][f"{nb}_h{key(h)}_truncated"].sum()))
            for nb in ("arr", "pt") for h in cfg.hs_s} for v in a.variants},
        occupancy_capped={v: {x: dict(any_cap=int(DD[v][f"occ_{x}_capped"].sum()), facing_cap=int(DD[v][f"occ_{x}_facing_cap"].sum()))
                              for x in [f"e{key(e)}" for e in cfg.eps_m] + ["tier"]} for v in a.variants},
        linear_vs_pchip=variant_agreement(EV["linear"], MM["linear"], EV["pchip"], MM["pchip"], cols) if "pchip" in a.variants else None)
    cand_summary = C.summary(CAND, ccfg, list(cohorts))
    cand_summary["cohort_states"] = {lab: {STATE_NAMES.get(int(k), 'no_linked_event'): int(v) for k, v in CAND[f"cohort_{lab}"].value_counts().sort_index().items()} for lab in cohorts}
    train = CAND[CAND["split"] == "train_cv"] if split else CAND
    cand_summary["controls_population"] = "train_cv matches only" if split else "all processed matches (no split given)"
    cand_summary["controls"] = {key(g): C.control_availability(train, {lab: f"cohort_{lab}" for lab in cohorts}, g, ccfg) for g in ccfg.gaps_s}
    cand_summary["coverage"] = {m: r["candidates_checks"]["coverage"] for m, r in per_match.items()}
    fw = {m: r["framewise"] for m, r in per_match.items()}
    agg: dict[str, Any] = {}
    for r in fw.values():
        for k_, v_ in r.items():
            if isinstance(v_, int):
                agg[k_] = agg.get(k_, 0) + v_
            elif isinstance(v_, dict) and k_ == "framewise_unmatched_reasons":
                for kk, vv in v_.items():
                    agg.setdefault(k_, {})[kk] = agg.setdefault(k_, {}).get(kk, 0) + vv
    cand_summary["framewise_audit"] = dict(total=agg, per_match=fw)
    runtime["aggregation_s"] = round(time.perf_counter() - t, 1)
    all_ok = all(r["checks"][v]["all_pass"] and r["checks"][v]["dense"]["distance_ok"] and r["checks"][v]["dense"]["occupancy_ok"]
                 and r["checks"][v]["dense"]["curve_ok"] and r["checks"][v]["dense"]["raw_ok"] for r in per_match.values() for v in a.variants)
    cand_ok = all(r["candidates_checks"]["all_pass"] and r["candidates_checks"]["coverage"]["all_covered"] for r in per_match.values())
    regs = [r["regression_v2"][v]["identical_on_common_columns"] for r in per_match.values() for v in r["regression_v2"]]
    reg_ok = None if a.no_regression or not regs else all(regs)
    atomic_write_text(out / "config.json", dumps(dict(schema=SCHEMA, candidates_schema=C.SCHEMA, local=cfg.as_dict(), window=wcfg.as_dict(),
                                                      candidates=ccfg.as_dict(), primary_columns=cols, cohort_columns=cohorts,
                                                      state_codes=STATE_NAMES, reference_variant="linear", sensitivity_variant="pchip")))
    atomic_write_text(out / "counts_grid.json", dumps(grid))
    atomic_write_text(out / "counts_primary.json", dumps(primary))
    atomic_write_text(out / "analysis.json", dumps(analysis))
    atomic_write_text(out / "candidates_summary.json", dumps(cand_summary))
    atomic_write_text(out / "checks.json", dumps(dict(all_pass=all_ok, candidates_all_pass=cand_ok, regression_v2_identical=reg_ok,
                                                      per_match={m: dict(checks=r["checks"], candidates=r["candidates_checks"],
                                                                         regression_v2=r["regression_v2"]) for m, r in per_match.items()})))
    t = time.perf_counter()
    if a.gallery and "pchip" in a.variants:
        from . import local_figures as LF

        def fig_cache(m: str) -> dict[str, Any]:
            c, mask, seg, ch, _dup = load(m)
            return dict(c=c, mask=mask, seg=seg, ch=ch, coefs={"linear": W.coefficients(seg, "linear")[:2]})
        figs = LF.make_all(out / "figures", EV, DD, MM, cfg, cols, LazyCaches(fig_cache), a.gallery, SEED)
        write_index(out / "visual_index.html", [(title, out / "figures", items) for title, items in figs],
                    f"Four-method descriptors on {len(a.matches)} FIFA development matches (actual data; linear reference unless stated; "
                    f"primary h = {cfg.primary_h_s} s). Gallery examples are a seeded random sample (seed {SEED}), not hand-picked.")
    runtime["figures_s"] = round(time.perf_counter() - t, 1)
    runtime["total_this_invocation_s"] = round(time.perf_counter() - T0, 1)
    mi = proc.memory_info()
    runtime["peak_memory_mb"] = round(getattr(mi, "peak_wset", mi.rss) / 2**20, 1)
    runtime["peak_memory_kind"] = "peak working set (Windows)" if hasattr(mi, "peak_wset") else "rss at end"
    runtime["resumed"] = bool(a.resume)
    atomic_write_text(out / "runtime.json", dumps(runtime))
    snap1 = validate.raw_snapshot(RAW)
    finish_manifest(out, dict(schema=SCHEMA, candidates_schema=C.SCHEMA, matches=a.matches, split="development only",
                              internal_split=a.split, internal_heldout=split["heldout"] if split else None, inputs=dict(
        v1_shared=V1_SHARED, v1_files_sha256={f"{m}/events_{v}.parquet": v1man["outputs_sha256"][f"{m}/events_{v}.parquet"] for m in a.matches for v in a.variants},
        pilot_v2=None if a.no_regression else V2_PILOT), config=cfg.as_dict(), candidate_config=ccfg.as_dict(), code_sha256=code_sha256(),
        raw_unchanged=snap0 == snap1, raw_snapshot=snap1, created_unix=int(time.time()), argv=sys.argv[1:]))
    print(dumps(dict(all_checks_pass=all_ok, candidates_checks_pass=cand_ok, regression_v2_identical=reg_ok, raw_unchanged=snap0 == snap1,
                     runtime=runtime, states_linear=analysis["states"]["linear"], candidates=cand_summary["candidates"],
                     outcomes={g: {o: r["funnel"][o] for o in C.OUTCOMES} for g, r in cand_summary["per_gap"].items()})))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
