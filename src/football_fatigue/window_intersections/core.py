"""Sliding-window geometric-intersection extraction on FIFA tracking (M11, advisor-aligned): ``m11-window-intersections-v1``.

**Windows.** In every continuous playing period, one window is anchored at every whole second ``k`` of period time (the period's
first frame is second 0). The window is [k - 2.5 s, k + 2.5 s] on the actual frame timestamps. A window is *complete* when
it lies entirely inside the period. Incomplete windows (the first and last seconds of a period) are counted and not evaluated.
A player is *active* in a window when at least one of its samples is AVAILABLE in it. Every unordered pair of active players is one
pair-window evaluation: with 22 active players there are 231. (A, B) and (B, A) are never both formed, because a pair is always
stored with ``slot_a < slot_b``.

**Trajectories (quality policy).** These are the Step 1 valid segments of ``intersections.exact.trajectories``. A segment joins
consecutive 15 Hz update instants of one player within one period. It is never formed across an invalid step (gap > 0.5 s or a
period change), an implausible jump, or an unattributed or ambiguous sample. Held positions are represented as stationary
segments. In addition, segments touching a sample bit-identical to another player's sample (confirmed corruption, see
``intersections.detect.duplicate_positions``) are removed, as are segments of period 0 and quarantined periods. A *chain* is a
maximal run of joined segments. Nothing is interpolated across a chain break, a substitution or a period boundary.

**Variants.** Each segment is a cubic in local time s = t - t0, with coefficients for x and y.

- ``linear``: the chord between the two samples.
- ``pchip``: the PCHIP interpolant of the chain's vertices (``scipy.interpolate.PchipInterpolator``, per coordinate, in time).
  PCHIP needs at least 3 vertices. A one-segment chain stays linear and is flagged ``pchip_supported = False``. PCHIP is monotone
  on every interval, so it never leaves the bounding box of the interval's two samples. This is checked.

For geometry, a PCHIP segment is the polyline through ``pchip_substeps + 1`` points equally spaced in time. Within a sub-piece,
time is linear.

**Intersections.** These are spatial: the two players' paths meet, at any times tA and tB. They are computed once per player pair
and period, exactly on the (sub-)segments. A meeting at a shared vertex, found by two neighbouring segments, is one intersection
when both times agree within ``same_hit_s``. Zero-length (held) pieces are not tested. A contact with a held position is found
through the neighbouring moving segment's end point. Collinear overlaps are reported at the overlap midpoint (kind
``collinear``).

**Window membership** is exact: an intersection belongs to window k when both tA and tB lie in [k - 2.5, k + 2.5]. This is the
same as clipping both trajectories to the window and intersecting them; a brute-force check in the pilot compares the two.

**First intersection.** The primary intersection of a pair-window is the first in chronological order. The ordering key is the
earliest arrival of either player, min(tA, tB), then max(tA, tB), then x, then y. The number of further intersections in the
window is recorded, and up to ``max_extra`` of them are kept as diagnostics. Only the primary intersection becomes a candidate.

**Methods** (evaluated on the primary intersection of a pair-window):

- A: |tA - tB| <= tau.
- B: the synchronized distance at the anchor time, |A(k) - B(k)|, is <= R. Positions come from the variant's curve. The rule is
  unavailable when a player has no segment at k. The minimum distance anywhere in the window is a separate diagnostic.
- C: A and B both hold.

**Deduplication.** The same physical intersection is the primary of up to 5 overlapping windows. It becomes one *event*, keyed by
pair and intersection, and keeps the list of windows that found it.

- For method B, an event qualifies when the anchor-distance rule holds in at least one of its windows; those windows are listed.
- *Encounters* group the events of one pair that follow each other within ``encounter_gap_s`` and ``encounter_dist_m``. They are
  an audit grouping and never remove events.

Identities (slot, player id, jersey, side) are metadata for labels, grouping and audits, never model inputs.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator

from football_fatigue.encounters import quality as Q
from football_fatigue.encounters.trajectories import STATE_AVAILABLE, RawCache
from football_fatigue.intersections import exact as X
from football_fatigue.intersections.detect import UNUSABLE, duplicate_positions

SCHEMA = "m11-window-intersections-v1"
VARIANTS = ("linear", "pchip")
PROPER, ENDPOINT, COLLINEAR = "proper", "endpoint", "collinear"
SUSPECT_BITS = (("implausible_jump_in_context", Q.IMPLAUSIBLE_JUMP), ("frame_wide_shift_in_context", Q.FRAME_WIDE_SHIFT),
                ("discontinuity_candidate_in_context", Q.DISCONTINUITY_CANDIDATE), ("substitution_window_in_context", Q.SUBSTITUTION_WINDOW))


@dataclass(frozen=True)
class WindowConfig:
    half_s: float = 2.5                       # window = anchor +- half_s
    taus_s: tuple[float, ...] = (0.10, 0.20, 0.30)          # method A / C temporal tolerances (nested)
    radius_m: float = 0.25                    # method B / C anchor-distance tolerance
    radius_diagnostic_m: tuple[float, ...] = (0.50, 1.00)   # sensitivity of B, reported only (not a dataset)
    pchip_substeps: int = 10                  # PCHIP sub-pieces per native segment (geometry only)
    tol_m: float = 1e-6                       # geometric tolerance (endpoint / collinear / degenerate)
    same_hit_s: float = 1e-6                  # two hits with both times this close are one intersection (shared vertex)
    encounter_gap_s: float = 1.0
    encounter_dist_m: float = 1.0
    crowd_radius_m: float = 2.0               # third-player diagnostic
    shared_player_s: float = 1.0
    max_extra: int = 8                        # further intersections kept per pair-window (diagnostic)
    context_s: float = 2.5                    # before/after availability around the event time
    min_context_s: float = 1.0
    extreme_speed_m_s: float = 10.0           # plausible-extreme flags (kept, never rejected)
    extreme_acc_m_s2: float = 10.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def tau_key(self, tau: float) -> str:
        return f"{tau:.2f}".replace(".", "p")


# ---------------------------------------------------------------- segments and curves

def build_segments(c: RawCache, mask: np.ndarray, steps: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, int], np.ndarray]:
    """Valid native segments (sorted by slot, then time) with chain ids, the exclusion counts and the duplicate-position mask."""
    seg, counts = X.trajectories(c, mask, steps)
    dup = duplicate_positions(c)
    sl = seg["slot"]
    bad_p = (seg["period"] <= 0) | np.isin(seg["period"], np.array(c.quarantine_periods, int))
    bad_d = (dup[seg["row0"], sl] | dup[seg["row1"], sl]) & ~bad_p
    keep = ~bad_p & ~bad_d
    seg = finalize_segments({k: v[keep] for k, v in seg.items()})
    counts = dict(counts, excluded_period0_or_quarantine=int(bad_p.sum()), excluded_duplicate_position=int(bad_d.sum()),
                  segments_used=len(seg["slot"]), chains=int(seg["chain"][-1] + 1) if len(seg["slot"]) else 0)
    return seg, counts, dup


def finalize_segments(seg: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Add chain ids (maximal runs of joined segments: same slot, row1 == next row0), durations and lengths."""
    n = len(seg["slot"])
    brk = np.r_[True, (seg["slot"][1:] != seg["slot"][:-1]) | (seg["row0"][1:] != seg["row1"][:-1])] if n else np.zeros(0, bool)
    seg["chain"] = np.cumsum(brk) - 1
    seg["dt"] = seg["t1"] - seg["t0"]
    if n and not (seg["dt"] > 0).all():
        raise ValueError("non-positive segment duration")
    seg["length"] = np.hypot(seg["x1"] - seg["x0"], seg["y1"] - seg["y0"])
    return seg


def chain_table(seg: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    st = np.flatnonzero(np.r_[True, seg["chain"][1:] != seg["chain"][:-1]])
    en = np.r_[st[1:], len(seg["chain"])] - 1
    return dict(first=st, last=en, slot=seg["slot"][st], period=seg["period"][st], t0=seg["t0"][st], t1=seg["t1"][en],
                n_segments=en - st + 1)


def coefficients(seg: dict[str, np.ndarray], variant: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cubic coefficients (n, 4) for x and y in local time (c0 s^3 + c1 s^2 + c2 s + c3), and ``pchip_supported`` per segment."""
    n = len(seg["t0"])
    cx, cy = np.zeros((n, 4)), np.zeros((n, 4))
    cx[:, 2], cx[:, 3] = (seg["x1"] - seg["x0"]) / seg["dt"], seg["x0"]
    cy[:, 2], cy[:, 3] = (seg["y1"] - seg["y0"]) / seg["dt"], seg["y0"]
    ch = chain_table(seg)
    sup = np.repeat(ch["n_segments"] >= 2, ch["n_segments"])      # PCHIP needs >= 3 vertices; same flag for both variants
    if variant == "linear":
        return cx, cy, sup
    if variant != "pchip":
        raise ValueError(variant)
    for a, b in zip(ch["first"], ch["last"] + 1):
        if b - a < 2:
            continue
        T = np.r_[seg["t0"][a:b], seg["t1"][b - 1]]
        if not np.array_equal(T[1:-1], seg["t1"][a:b - 1]):
            raise ValueError("chain segments are not contiguous in time")
        for arr, v0, v1 in ((cx, seg["x0"], seg["x1"]), (cy, seg["y0"], seg["y1"])):
            arr[a:b] = PchipInterpolator(T, np.r_[v0[a:b], v1[b - 1]]).c.T
    return cx, cy, sup


def evalc(cf: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Evaluate cubic rows ``cf`` (..., 4) at local times ``s`` (broadcast over the leading axes)."""
    return ((cf[..., 0] * s + cf[..., 1]) * s + cf[..., 2]) * s + cf[..., 3]


def slot_slice(seg, s: int, period: int | None = None) -> np.ndarray:
    """Segment indices of one slot (optionally one period); segments are sorted by slot, then time."""
    a, b = np.searchsorted(seg["slot"], s, "left"), np.searchsorted(seg["slot"], s, "right")
    g = np.arange(a, b)
    return g if period is None else g[seg["period"][g] == period]


def sub_points(seg, cx, cy, g: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(m, K+1) x, y, t of segments ``g``. The end points are the exact samples."""
    u = np.arange(K + 1) / K
    s = seg["dt"][g][:, None] * u[None]
    px, py = evalc(cx[g][:, None, :], s), evalc(cy[g][:, None, :], s)
    px[:, 0], px[:, -1], py[:, 0], py[:, -1] = seg["x0"][g], seg["x1"][g], seg["y0"][g], seg["y1"][g]
    return px, py, seg["t0"][g][:, None] + s


# ---------------------------------------------------------------- elementwise segment intersection

def seg_intersect(p: np.ndarray, r: np.ndarray, q: np.ndarray, s: np.ndarray, tol: float) -> tuple[np.ndarray, ...]:
    """Elementwise intersection of segments p -> p + r and q -> q + s ((m, 2) arrays). Same rules as ``intersections.geometry``.

    Returns (ok, ua, ub, x, y, kind). Here ua and ub are the parameters in [0, 1] along each segment; degenerate segments are
    never ok."""
    m = len(p)
    ok, ua, ub = np.zeros(m, bool), np.zeros(m), np.zeros(m)
    kind = np.full(m, "", "<U9")
    lr, ls = np.hypot(r[:, 0], r[:, 1]), np.hypot(s[:, 0], s[:, 1])
    valid = (lr >= tol) & (ls >= tol)
    qp = q - p
    den = r[:, 0] * s[:, 1] - r[:, 1] * s[:, 0]
    par = np.abs(den) <= 1e-10 * lr * ls
    n = np.flatnonzero(valid & ~par)
    if len(n):
        a = (qp[n, 0] * s[n, 1] - qp[n, 1] * s[n, 0]) / den[n]
        b = (qp[n, 0] * r[n, 1] - qp[n, 1] * r[n, 0]) / den[n]
        ea, eb = tol / lr[n], tol / ls[n]
        good = (a >= -ea) & (a <= 1 + ea) & (b >= -eb) & (b <= 1 + eb)
        end = (a <= ea) | (a >= 1 - ea) | (b <= eb) | (b >= 1 - eb)
        k = n[good]
        ok[k], ua[k], ub[k] = True, np.clip(a[good], 0, 1), np.clip(b[good], 0, 1)
        kind[k] = np.where(end[good], ENDPOINT, PROPER)
    k = np.flatnonzero(valid & par)
    if len(k):
        off = np.abs(qp[k, 0] * r[k, 1] - qp[k, 1] * r[k, 0]) / lr[k]
        k = k[off <= tol]
        if len(k):
            rr = (r[k] ** 2).sum(1)
            t0 = (qp[k] * r[k]).sum(1) / rr
            t1 = t0 + (s[k] * r[k]).sum(1) / rr
            lo, hi = np.maximum(np.minimum(t0, t1), 0.0), np.minimum(np.maximum(t0, t1), 1.0)
            span = (hi - lo) * lr[k]
            keep = span >= -tol
            k, lo, hi, span, t0, t1 = k[keep], lo[keep], hi[keep], span[keep], t0[keep], t1[keep]
            mid = (lo + hi) / 2
            ok[k], ua[k], ub[k] = True, mid, np.clip((mid - t0) / (t1 - t0), 0, 1)
            kind[k] = np.where(span > tol, COLLINEAR, ENDPOINT)
    x, y = p[:, 0] + ua * r[:, 0], p[:, 1] + ua * r[:, 1]
    return ok, ua, ub, x, y, kind


def _expand(s0: np.ndarray, n0: np.ndarray, s1: np.ndarray, n1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """All index pairs of the ranges [s0, s0 + n0) x [s1, s1 + n1), concatenated over rows."""
    tot = n0 * n1
    if not tot.sum():
        return np.zeros(0, int), np.zeros(0, int)
    rep = np.repeat(np.arange(len(tot)), tot)
    off = np.arange(tot.sum()) - np.repeat(np.cumsum(tot) - tot, tot)
    return s0[rep] + off // n1[rep], s1[rep] + off % n1[rep]


def candidate_pairs(ga: np.ndarray, gb: np.ndarray, seg, K: int) -> tuple[np.ndarray, ...]:
    """Sub-piece pairs of segment pairs (ga, gb): indices into the pair list and sub-piece numbers."""
    m = len(ga)
    pair = np.repeat(np.arange(m), K * K)
    k = np.tile(np.arange(K * K), m)
    return pair, k // K, k % K


def pair_segment_hits(seg, cx, cy, ga: np.ndarray, gb: np.ndarray, K: int, tol: float) -> dict[str, np.ndarray]:
    """Intersections of the (sub-)segments of candidate segment pairs (ga[i], gb[i])."""
    ax, ay, at = sub_points(seg, cx, cy, ga, K)
    bx, by, bt = sub_points(seg, cx, cy, gb, K)
    pi, ka, kb = candidate_pairs(ga, gb, seg, K)
    p = np.c_[ax[pi, ka], ay[pi, ka]]
    r = np.c_[ax[pi, ka + 1], ay[pi, ka + 1]] - p
    q = np.c_[bx[pi, kb], by[pi, kb]]
    s = np.c_[bx[pi, kb + 1], by[pi, kb + 1]] - q
    ok, ua, ub, x, y, kind = seg_intersect(p, r, q, s, tol)
    i = np.flatnonzero(ok)
    pi, ka, kb = pi[i], ka[i], kb[i]
    ta = at[pi, ka] + ua[i] * (at[pi, ka + 1] - at[pi, ka])
    tb = bt[pi, kb] + ub[i] * (bt[pi, kb + 1] - bt[pi, kb])
    return dict(ga=ga[pi], gb=gb[pi], sub_a=ka, sub_b=kb, x=x[i], y=y[i], ta=ta, tb=tb, kind=kind[i])


def merge_same(h: dict[str, np.ndarray], same_s: float) -> dict[str, np.ndarray]:
    """Sort hits of one pair by (ta, tb) and collapse hits whose two times both agree within ``same_s`` (shared vertices)."""
    if not len(h["ta"]):
        return h
    o = np.lexsort((h["tb"], h["ta"]))
    h = {k: v[o] for k, v in h.items()}
    keep = np.ones(len(o), bool)
    last = 0
    for i in range(1, len(o)):
        if abs(h["ta"][i] - h["ta"][last]) <= same_s and abs(h["tb"][i] - h["tb"][last]) <= same_s:
            keep[i] = False
            if h["kind"][i] != PROPER and h["kind"][last] == PROPER:
                h["kind"][last] = h["kind"][i]
        else:
            last = i
    return {k: v[keep] for k, v in h.items()}


def period_hits(seg, cx, cy, period: int, pstart: float, cfg: WindowConfig, variant: str) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """All intersections of every player pair in one period with |tA - tB| <= 2 * half (anything farther apart can share no window)."""
    K = cfg.pchip_substeps if variant == "pchip" else 1
    span = 2 * cfg.half_s
    idx = np.flatnonzero((seg["period"] == period) & (seg["length"] >= cfg.tol_m))   # degenerate (held) pieces are never tested
    st = dict(segments_tested=int(len(idx)), segment_pairs_checked=0, subpiece_pairs_checked=0, raw_hits=0)
    keys = ("ga", "gb", "sub_a", "sub_b", "x", "y", "ta", "tb", "kind", "sa", "sb")
    if not len(idx):
        return {k: np.zeros(0) for k in keys}, st
    max_dt = float(seg["dt"][idx].max())
    lag = int(math.ceil(span + max_dt)) + 1
    b_all = np.floor(seg["t0"][idx] - pstart).astype(int)
    nb = int(b_all.max()) + 1
    per = {}
    for s in np.unique(seg["slot"][idx]):
        sel = seg["slot"][idx] == s
        g, bs = idx[sel], b_all[sel]
        starts = np.searchsorted(bs, np.arange(nb + 1))
        box = np.full((nb, 4), np.nan)
        ne = np.flatnonzero(starts[1:] > starts[:-1])
        xs0, xs1 = np.minimum(seg["x0"][g], seg["x1"][g]), np.maximum(seg["x0"][g], seg["x1"][g])
        ys0, ys1 = np.minimum(seg["y0"][g], seg["y1"][g]), np.maximum(seg["y0"][g], seg["y1"][g])
        for col, arr, fn in ((0, xs0, np.minimum), (1, ys0, np.minimum), (2, xs1, np.maximum), (3, ys1, np.maximum)):
            box[ne, col] = fn.reduceat(arr, starts[ne])
        per[int(s)] = (g, starts, box)
    out: list[dict[str, np.ndarray]] = []
    tol = cfg.tol_m
    for sa, sb in itertools.combinations(sorted(per), 2):
        (gA, stA, bA), (gB, stB, bB) = per[sa], per[sb]
        ci, cj = [], []
        for L in range(-lag, lag + 1):
            i = np.arange(max(0, -L), min(nb, nb - L))
            j = i + L
            ov = (bA[i, 0] <= bB[j, 2] + tol) & (bB[j, 0] <= bA[i, 2] + tol) & (bA[i, 1] <= bB[j, 3] + tol) & (bB[j, 1] <= bA[i, 3] + tol)
            ci.append(i[ov]), cj.append(j[ov])
        ci, cj = np.concatenate(ci), np.concatenate(cj)
        if not len(ci):
            continue
        ia, ib = _expand(stA[ci], stA[ci + 1] - stA[ci], stB[cj], stB[cj + 1] - stB[cj])
        ga, gb = gA[ia], gB[ib]
        near = np.maximum(seg["t0"][ga], seg["t0"][gb]) - np.minimum(seg["t1"][ga], seg["t1"][gb]) <= span + 1e-9
        ga, gb = ga[near], gb[near]
        bo = ((np.minimum(seg["x0"][ga], seg["x1"][ga]) <= np.maximum(seg["x0"][gb], seg["x1"][gb]) + tol)
              & (np.minimum(seg["x0"][gb], seg["x1"][gb]) <= np.maximum(seg["x0"][ga], seg["x1"][ga]) + tol)
              & (np.minimum(seg["y0"][ga], seg["y1"][ga]) <= np.maximum(seg["y0"][gb], seg["y1"][gb]) + tol)
              & (np.minimum(seg["y0"][gb], seg["y1"][gb]) <= np.maximum(seg["y0"][ga], seg["y1"][ga]) + tol))
        ga, gb = ga[bo], gb[bo]
        st["segment_pairs_checked"] += int(len(ga))
        st["subpiece_pairs_checked"] += int(len(ga)) * K * K
        if not len(ga):
            continue
        h = pair_segment_hits(seg, cx, cy, ga, gb, K, tol)
        st["raw_hits"] += int(len(h["ta"]))
        keep = np.abs(h["ta"] - h["tb"]) <= span + 1e-9
        h = merge_same({k: v[keep] for k, v in h.items()}, cfg.same_hit_s)
        if len(h["ta"]):
            h["sa"], h["sb"] = np.full(len(h["ta"]), sa), np.full(len(h["ta"]), sb)
            out.append(h)
    if not out:
        return {k: np.zeros(0) for k in keys}, st
    return {k: np.concatenate([h[k] for h in out]) for k in keys}, st


# ---------------------------------------------------------------- windows

def period_anchors(pstart: float, pend: float, half: float) -> tuple[np.ndarray, np.ndarray]:
    """Whole period seconds 0..floor(duration) and whether each window [k - half, k + half] lies inside the period."""
    dur = pend - pstart
    k = np.arange(0, int(math.floor(dur + 1e-9)) + 1)
    return k, (k - half >= -1e-9) & (k + half <= dur + 1e-9)


def order_key(ta: np.ndarray, tb: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Deterministic chronological order: earliest arrival of either player, then the later arrival, then x, then y."""
    return np.lexsort((y, x, np.maximum(ta, tb), np.minimum(ta, tb)))


def assign_windows(h: dict[str, np.ndarray], pstart: float, kc: np.ndarray, half: float, max_extra: int) -> list[dict[str, Any]]:
    """Pair-window detections: for every (pair, complete anchor) with at least one intersection, the primary intersection
    (first in ``order_key``), the number of intersections and up to ``max_extra`` further ones (hit indices)."""
    if not len(kc) or not len(h["ta"]):
        return []
    k0, k1 = int(kc.min()), int(kc.max())
    det: dict[tuple, dict] = {}
    o = order_key(h["ta"], h["tb"], h["x"], h["y"])
    tmin, tmax = np.minimum(h["ta"], h["tb"]) - pstart, np.maximum(h["ta"], h["tb"]) - pstart
    for i in o:
        lo = max(k0, int(math.ceil(tmax[i] - half - 1e-9)))
        hi = min(k1, int(math.floor(tmin[i] + half + 1e-9)))
        for k in range(lo, hi + 1):
            key = (int(h["sa"][i]), int(h["sb"][i]), k)
            d = det.get(key)
            if d is None:
                det[key] = dict(sa=key[0], sb=key[1], anchor=k, hit=int(i), n_int=1, extra=[])
            else:
                d["n_int"] += 1
                if len(d["extra"]) < max_extra:
                    d["extra"].append(int(i))
    return sorted(det.values(), key=lambda d: (d["anchor"], d["sa"], d["sb"]))


def positions_at(seg, cx, cy, slot: int, period: int, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Variant position of one player at ``times`` (NaN where no valid segment covers the instant)."""
    g = slot_slice(seg, slot, period)
    px, py = np.full(len(times), np.nan), np.full(len(times), np.nan)
    if not len(g):
        return px, py
    i = np.searchsorted(seg["t0"][g], times, "right") - 1
    ok = (i >= 0) & (seg["t1"][g][np.clip(i, 0, None)] >= times - 1e-9)
    gg = g[np.clip(i, 0, None)][ok]
    s = times[ok] - seg["t0"][gg]
    px[ok], py[ok] = evalc(cx[gg], s), evalc(cy[gg], s)
    return px, py


def window_stats(c: RawCache, seg, ch, period: int, pstart: float, k: np.ndarray, half: float) -> dict[str, Any]:
    """Per complete window: active players, players with a valid trajectory, players with a complete trajectory (one chain covering
    the whole window), and pair-window counts."""
    rows = np.flatnonzero(c.canonical_period == period)
    tp = c.video_time_s[rows]
    lo_t, hi_t = pstart + k - half, pstart + k + half
    a, b = np.searchsorted(tp, lo_t, "left"), np.searchsorted(tp, hi_t, "right")
    avail = c.state[rows] == STATE_AVAILABLE
    cum = np.vstack([np.zeros((1, c.n_slots), int), np.cumsum(avail, 0)])
    active = (cum[b] - cum[a]) > 0                                      # (n_anchor, n_slots)
    valid, complete = np.zeros_like(active), np.zeros_like(active)
    for s in range(c.n_slots):
        g = slot_slice(seg, s, period)
        if len(g):
            i = np.searchsorted(seg["t0"][g], hi_t, "right") - 1
            valid[:, s] = (i >= 0) & (seg["t1"][g][np.clip(i, 0, None)] >= lo_t)
        cs = np.flatnonzero((ch["slot"] == s) & (ch["period"] == period))
        if len(cs):
            j = np.searchsorted(ch["t0"][cs], lo_t + 1e-9, "right") - 1
            complete[:, s] = (j >= 0) & (ch["t1"][cs][np.clip(j, 0, None)] >= hi_t - 1e-9)
    na, nv, nc = active.sum(1), (valid & active).sum(1), (complete & active).sum(1)
    return dict(windows=int(len(k)), player_windows_active=int(na.sum()), player_windows_valid_trajectory=int(nv.sum()),
                player_windows_complete_trajectory=int(nc.sum()), pair_window_evaluations=int((na * (na - 1) // 2).sum()),
                pair_windows_both_valid=int((nv * (nv - 1) // 2).sum()), pair_windows_both_complete=int((nc * (nc - 1) // 2).sum()),
                active_players_per_window=dict(min=int(na.min()) if len(na) else None, median=float(np.median(na)) if len(na) else None,
                                               max=int(na.max()) if len(na) else None),
                windows_with_22_active=int((na == 22).sum()), _complete=complete)


# ---------------------------------------------------------------- events, quality, methods

def _hash(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def _speed_acc(seg) -> tuple[np.ndarray, np.ndarray]:
    """Per-segment speed and the acceleration to the next segment of the same chain (both moving), NaN otherwise."""
    v = np.c_[seg["x1"] - seg["x0"], seg["y1"] - seg["y0"]] / seg["dt"][:, None]
    sp = np.hypot(v[:, 0], v[:, 1])
    acc = np.full(len(sp), np.nan)
    if len(sp) > 1:
        same = (seg["chain"][1:] == seg["chain"][:-1]) & ~seg["stationary"][1:] & ~seg["stationary"][:-1]
        dv = np.hypot(*(v[1:] - v[:-1]).T) / ((seg["dt"][1:] + seg["dt"][:-1]) / 2)
        acc[:-1] = np.where(same, dv, np.nan)
    return sp, acc


def build_events(c: RawCache, mask: np.ndarray, dup: np.ndarray, seg, ch, cx, cy, sup, hits: dict[str, np.ndarray], dets: list[dict],
                 period: int, pstart: float, pend: float, complete: np.ndarray, kc: np.ndarray, cfg: WindowConfig, variant: str
                 ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deduplicated event records and the pair-window detection records of one period."""
    t_all = c.video_time_s
    rows_p = np.flatnonzero(c.canonical_period == period)
    tp = t_all[rows_p]
    est = (mask & Q.ESTIMATED) != 0
    sp, acc = _speed_acc(seg)
    kpos = {int(k): i for i, k in enumerate(kc)}
    # anchor positions of every slot at every complete anchor (variant curve)
    anc_pos = {}
    for s in np.unique(np.r_[[d["sa"] for d in dets], [d["sb"] for d in dets]]).astype(int) if dets else []:
        anc_pos[s] = positions_at(seg, cx, cy, s, period, pstart + kc.astype(float))
    det_rows = []
    by_hit: dict[int, list[dict]] = {}
    for d in dets:
        sa, sb, k = d["sa"], d["sb"], d["anchor"]
        j = kpos[k]
        pa, pb = (anc_pos[sa][0][j], anc_pos[sa][1][j]), (anc_pos[sb][0][j], anc_pos[sb][1][j])
        dist = float(np.hypot(pa[0] - pb[0], pa[1] - pb[1]))
        a, b = np.searchsorted(tp, pstart + k - cfg.half_s, "left"), np.searchsorted(tp, pstart + k + cfg.half_s, "right")
        rr = rows_p[a:b]
        dd = np.hypot(c.x[rr, sa] - c.x[rr, sb], c.y[rr, sa] - c.y[rr, sb])
        dmin = float(np.nanmin(dd)) if np.isfinite(dd).any() else float("nan")
        rec = dict(period=period, anchor_s=k, anchor_video_s=pstart + k, slot_a=sa, slot_b=sb, hit=d["hit"], n_intersections=d["n_int"],
                   abs_dt_s=abs(float(hits["ta"][d["hit"]]) - float(hits["tb"][d["hit"]])),
                   anchor_dist_m=dist, window_min_raw_dist_m=dmin, complete_a=bool(complete[j, sa]), complete_b=bool(complete[j, sb]),
                   extra=[dict(x=float(hits["x"][e]), y=float(hits["y"][e]), t_a_s=float(hits["ta"][e]), t_b_s=float(hits["tb"][e]))
                          for e in d["extra"]])
        det_rows.append(rec)
        by_hit.setdefault(d["hit"], []).append(rec)
    # present-but-not-primary windows per hit
    present: dict[int, int] = {}
    for d in dets:
        for e in d["extra"]:
            present[e] = present.get(e, 0) + 1
    events = []
    hit_event: dict[int, str] = {}
    for hi_, recs in sorted(by_hit.items()):
        sa, sb = int(hits["sa"][hi_]), int(hits["sb"][hi_])
        ga, gb = int(hits["ga"][hi_]), int(hits["gb"][hi_])
        ta, tb = float(hits["ta"][hi_]), float(hits["tb"][hi_])
        te = (ta + tb) / 2
        ad = np.array([r["anchor_dist_m"] for r in recs])
        ev = dict(
            schema=SCHEMA, variant=variant, match_id=int(c.match_id), period=int(period),
            event_id=_hash(c.match_id, variant, period, sa, sb, round(ta * 1e6), round(tb * 1e6)),
            slot_a=sa, slot_b=sb, player_id_a=int(c.slot_player_id[sa]), player_id_b=int(c.slot_player_id[sb]),
            jersey_a=int(c.slot_jersey[sa]), jersey_b=int(c.slot_jersey[sb]), side_a=str(c.slot_side[sa]), side_b=str(c.slot_side[sb]),
            team_relation="same_team" if c.slot_side[sa] == c.slot_side[sb] else "opposite_teams",
            x_m=float(hits["x"][hi_]), y_m=float(hits["y"][hi_]), kind=str(hits["kind"][hi_]),
            t_a_s=ta, t_b_s=tb, t_a_rel_s=ta - pstart, t_b_rel_s=tb - pstart, t_event_s=te, t_event_rel_s=te - pstart, abs_dt_s=abs(ta - tb),
            seg_a_row0=int(seg["row0"][ga]), seg_a_row1=int(seg["row1"][ga]), seg_a_frame0=int(c.frame_num[seg["row0"][ga]]),
            seg_a_frame1=int(c.frame_num[seg["row1"][ga]]), sub_a=int(hits["sub_a"][hi_]),
            seg_b_row0=int(seg["row0"][gb]), seg_b_row1=int(seg["row1"][gb]), seg_b_frame0=int(c.frame_num[seg["row0"][gb]]),
            seg_b_frame1=int(c.frame_num[seg["row1"][gb]]), sub_b=int(hits["sub_b"][hi_]),
            est_at_a=bool(est[seg["row0"][ga], sa] or est[seg["row1"][ga], sa]), est_at_b=bool(est[seg["row0"][gb], sb] or est[seg["row1"][gb], sb]),
            pchip_supported_a=bool(sup[ga]), pchip_supported_b=bool(sup[gb]),
            windows=json.dumps([r["anchor_s"] for r in recs]), n_windows=len(recs), windows_present_not_primary=present.get(hi_, 0),
            max_intersections_in_window=max(r["n_intersections"] for r in recs),
            anchor_dists_m=json.dumps([None if not np.isfinite(v) else round(float(v), 6) for v in ad]),
            anchor_dist_min_m=float(np.nanmin(ad)) if np.isfinite(ad).any() else float("nan"),
            anchor_dist_unavailable_windows=int((~np.isfinite(ad)).sum()),
            window_min_raw_dist_m=float(np.nanmin([r["window_min_raw_dist_m"] for r in recs])),
            both_complete_any_window=any(r["complete_a"] and r["complete_b"] for r in recs),
        )
        for tau in cfg.taus_s:
            ev[f"a_tau_{cfg.tau_key(tau)}"] = ev["abs_dt_s"] <= tau + 1e-12
        ev["b_pass"] = bool(np.nanmin(ad) <= cfg.radius_m) if np.isfinite(ad).any() else False
        ev["b_pass_windows"] = json.dumps([r["anchor_s"] for r in recs if np.isfinite(r["anchor_dist_m"]) and r["anchor_dist_m"] <= cfg.radius_m])
        for r_ in cfg.radius_diagnostic_m:
            ev[f"b_diag_{cfg.tau_key(r_)}"] = bool(np.nanmin(ad) <= r_) if np.isfinite(ad).any() else False
        for tau in cfg.taus_s:
            ev[f"c_tau_{cfg.tau_key(tau)}"] = bool(ev[f"a_tau_{cfg.tau_key(tau)}"] and ev["b_pass"])
        ev.update(quality(c, mask, dup, est, seg, ch, sp, acc, rows_p, tp, ev, ga, gb, pstart, pend, cfg))
        events.append(ev)
        hit_event[hi_] = ev["event_id"]
    for r in det_rows:
        r["event_id"] = hit_event[r["hit"]]
    return events, det_rows


def quality(c, mask, dup, est, seg, ch, sp, acc, rows_p, tp, ev, ga: int, gb: int, pstart: float, pend: float, cfg: WindowConfig) -> dict[str, Any]:
    """Shared validation layer for one event: corruption (reject), suspected artifacts (flag), uncertain and plausible-extreme
    information, and before/after context availability around the event time."""
    te, h = ev["t_event_s"], cfg.context_s
    a, b = np.searchsorted(tp, te - h, "left"), np.searchsorted(tp, te + h, "right")
    rr = rows_p[a:b]
    reject, flag, info = [], [], []
    out: dict[str, Any] = {}
    for tag, s, g in (("a", ev["slot_a"], ga), ("b", ev["slot_b"], gb)):
        m = mask[rr, s]
        av = c.state[rr, s] == STATE_AVAILABLE
        out[f"missing_frac_{tag}"] = float(1 - av.mean()) if len(rr) else 1.0
        out[f"est_frac_{tag}"] = float(est[rr, s][av].mean()) if av.any() else float("nan")
        if dup[rr, s].any():
            reject.append(f"duplicate_position_in_context_{tag}")
        if ((m & UNUSABLE) != 0).any():
            reject.append(f"unattributed_or_ambiguous_in_context_{tag}")
        for name, bit in SUSPECT_BITS:
            if ((m & bit) != 0).any():
                flag.append(f"{name}_{tag}")
        if ((m & Q.OVERSPEED_STEP) != 0).any():
            info.append(f"overspeed_step_in_context_{tag}")
        k = int(seg["chain"][g])
        c0, c1 = float(ch["t0"][k]), float(ch["t1"][k])
        out[f"before_s_{tag}"] = float(np.clip(te - c0, 0, h))
        out[f"after_s_{tag}"] = float(np.clip(c1 - te, 0, h))
        gs = slot_slice(seg, s)
        sel = gs[np.searchsorted(seg["t1"][gs], te - h, "left"):np.searchsorted(seg["t0"][gs], te + h, "right")]
        sel = sel[seg["period"][sel] == ev["period"]]
        vmax = float(np.nanmax(sp[sel])) if len(sel) else float("nan")
        amax = float(np.nanmax(acc[sel])) if len(sel) and np.isfinite(acc[sel]).any() else float("nan")
        out[f"max_speed_m_s_{tag}"], out[f"max_acc_m_s2_{tag}"] = vmax, amax
        if np.isfinite(vmax) and vmax > cfg.extreme_speed_m_s:
            info.append(f"extreme_speed_{tag}")
        if np.isfinite(amax) and amax > cfg.extreme_acc_m_s2:
            info.append(f"extreme_acceleration_{tag}")
    if te - h < pstart or te + h > pend:
        info.append("context_crosses_period_boundary")
    if ev["est_at_a"] or ev["est_at_b"]:
        info.append("estimated_at_intersection")
    if not (ev["pchip_supported_a"] and ev["pchip_supported_b"]):
        info.append("pchip_unsupported_segment")
    tier = "T1_visible_visible" if not (ev["est_at_a"] or ev["est_at_b"]) else "T3_estimated_estimated" if (ev["est_at_a"] and ev["est_at_b"]) else "T2_mixed"
    status = "rejected" if reject else "flagged" if flag else "accepted"
    full = min(out["before_s_a"], out["after_s_a"], out["before_s_b"], out["after_s_b"]) >= h - 1e-6
    mn = min(out["before_s_a"], out["after_s_a"], out["before_s_b"], out["after_s_b"]) >= cfg.min_context_s - 1e-6
    # crowding: other players near the pair midpoint (or the intersection point) at the frame nearest the event time
    r0 = rows_p[int(np.clip(np.searchsorted(tp, te), 0, len(tp) - 1))]
    sa, sb = ev["slot_a"], ev["slot_b"]
    mid = ((c.x[r0, sa] + c.x[r0, sb]) / 2, (c.y[r0, sa] + c.y[r0, sb]) / 2)
    if not np.isfinite(mid[0]):
        mid = (ev["x_m"], ev["y_m"])
    others = [s for s in range(c.n_slots) if s not in (sa, sb)]
    dd = np.hypot(c.x[r0, others] - mid[0], c.y[r0, others] - mid[1])
    crowd = int(np.nansum(dd <= cfg.crowd_radius_m))
    if crowd:
        info.append("third_player_within_2m")
    label_reliable = status == "accepted" or (status == "flagged" and not any("substitution" in f or "discontinuity" in f for f in flag))
    return dict(out, status=status, reject_reasons=json.dumps(sorted(reject)), flag_reasons=json.dumps(sorted(flag)), info_flags=json.dumps(sorted(info)),
                tier=tier, context_full=bool(full), context_min=bool(mn), crowd_others_within_2m=crowd,
                label_reliable=bool(label_reliable), sample_ready=bool(mn and status == "accepted"))


def encounters_and_sharing(events: list[dict[str, Any]], cfg: WindowConfig) -> None:
    """Add ``encounter_id`` (audit grouping per pair) and ``shared_player_events`` (events of other pairs that share a player
    within ``shared_player_s``)."""
    by: dict[tuple, list[dict]] = {}
    for e in events:
        by.setdefault((e["period"], e["slot_a"], e["slot_b"]), []).append(e)
    n = 0
    for key in sorted(by):
        prev = None
        for e in sorted(by[key], key=lambda e: e["t_event_s"]):
            if prev is None or e["t_event_s"] - prev["t_event_s"] > cfg.encounter_gap_s or \
                    math.hypot(e["x_m"] - prev["x_m"], e["y_m"] - prev["y_m"]) > cfg.encounter_dist_m:
                n += 1
            e["encounter_id"] = f"{e['match_id']}-{e['variant']}-enc{n:05d}"
            prev = e
    by_slot: dict[tuple, list[dict]] = {}
    for e in events:
        for s in (e["slot_a"], e["slot_b"]):
            by_slot.setdefault((e["period"], s), []).append(e)
    for e in events:
        cnt = set()
        for s in (e["slot_a"], e["slot_b"]):
            for f in by_slot[(e["period"], s)]:
                if f is not e and (f["slot_a"], f["slot_b"]) != (e["slot_a"], e["slot_b"]) and abs(f["t_event_s"] - e["t_event_s"]) <= cfg.shared_player_s:
                    cnt.add(f["event_id"])
        e["shared_player_events"] = len(cnt)
        if cnt:
            info = json.loads(e["info_flags"])
            e["info_flags"] = json.dumps(sorted(info + ["overlapping_pair_events_shared_player"]))


# ---------------------------------------------------------------- one match

def run_variant(c: RawCache, mask: np.ndarray, seg, ch, dup, variant: str, cfg: WindowConfig) -> dict[str, Any]:
    cx, cy, sup = coefficients(seg, variant)
    events, dets, hit_tables, scan = [], [], {}, dict(periods=[], window=dict(), hits=dict(segments_tested=0, segment_pairs_checked=0,
                                                                                            subpiece_pairs_checked=0, raw_hits=0, intersections=0))
    wsum: dict[str, Any] = {}
    for p in sorted(int(v) for v in np.unique(seg["period"])):
        rows = np.flatnonzero(c.canonical_period == p)
        pstart, pend = float(c.video_time_s[rows[0]]), float(c.video_time_s[rows[-1]])
        k, comp = period_anchors(pstart, pend, cfg.half_s)
        kc = k[comp]
        ws = window_stats(c, seg, ch, p, pstart, kc, cfg.half_s)
        complete = ws.pop("_complete")
        ws["windows_incomplete"] = int((~comp).sum())
        h, st = period_hits(seg, cx, cy, p, pstart, cfg, variant)
        st["intersections"] = int(len(h["ta"]))
        for kk, v in st.items():
            scan["hits"][kk] += v
        d = assign_windows(h, pstart, kc, cfg.half_s, cfg.max_extra)
        ev, dr = build_events(c, mask, dup, seg, ch, cx, cy, sup, h, d, p, pstart, pend, complete, kc, cfg, variant)
        hit_tables[p] = h
        events += ev
        dets += [dict(r, variant=variant, match_id=int(c.match_id)) for r in dr]
        scan["periods"].append(dict(period=p, start_s=pstart, end_s=pend, **{kk: v for kk, v in ws.items()}))
        for kk, v in ws.items():
            if isinstance(v, int):
                wsum[kk] = wsum.get(kk, 0) + v
    encounters_and_sharing(events, cfg)
    scan["window"] = wsum
    scan["pair_window_detections"] = len(dets)
    scan["pair_windows_multi_intersection"] = int(sum(r["n_intersections"] > 1 for r in dets))
    scan["events"] = len(events)
    scan["intersections_never_primary"] = scan["hits"]["intersections"] - len(events)
    return dict(events=events, detections=dets, hits=hit_tables, cx=cx, cy=cy, sup=sup, scan=scan)


def compare_variants(lin: list[dict], pch: list[dict], dt_tol: float = 0.2, d_tol: float = 0.5) -> dict[str, Any]:
    """One-to-one matching of linear and PCHIP events of the same pair (greedy by time difference, within ``dt_tol`` s and
    ``d_tol`` m). Adds ``pchip_match`` columns to the linear events and ``linear_match`` to the PCHIP events."""
    by: dict[tuple, list[dict]] = {}
    for e in pch:
        by.setdefault((e["period"], e["slot_a"], e["slot_b"]), []).append(e)
    used: set[str] = set()
    cand = []
    for e in lin:
        for f in by.get((e["period"], e["slot_a"], e["slot_b"]), []):
            dt = abs(e["t_event_s"] - f["t_event_s"])
            dd = math.hypot(e["x_m"] - f["x_m"], e["y_m"] - f["y_m"])
            if dt <= dt_tol and dd <= d_tol:
                cand.append((dt, dd, e["event_id"], f["event_id"], e, f))
    cand.sort(key=lambda r: (r[0], r[1], r[2], r[3]))
    done: set[str] = set()
    for dt, dd, _i, _j, e, f in cand:
        if e["event_id"] in done or f["event_id"] in used:
            continue
        done.add(e["event_id"]), used.add(f["event_id"])
        e["pchip_match"], f["linear_match"] = f["event_id"], e["event_id"]
        e["pchip_loc_diff_m"] = f["linear_loc_diff_m"] = dd
        e["pchip_abs_dt_change_s"] = f["linear_abs_dt_change_s"] = f["abs_dt_s"] - e["abs_dt_s"]
    for e in lin:
        e.setdefault("pchip_match", None), e.setdefault("pchip_loc_diff_m", float("nan")), e.setdefault("pchip_abs_dt_change_s", float("nan"))
    for f in pch:
        f.setdefault("linear_match", None), f.setdefault("linear_loc_diff_m", float("nan")), f.setdefault("linear_abs_dt_change_s", float("nan"))
    return dict(matched=len(done), linear_only=len(lin) - len(done), pchip_only=len(pch) - len(used))


# ---------------------------------------------------------------- brute-force check (pilot)

def brute_force_window(seg, cx, cy, sa: int, sb: int, period: int, lo: float, hi: float, K: int, cfg: WindowConfig) -> list[tuple[float, float]]:
    """Intersections of two players' trajectories clipped to [lo, hi], computed independently with
    ``intersections.geometry.intersect`` (all piece pairs). Returns the (tA, tB) of every intersection in ``order_key`` order."""
    from football_fatigue.intersections.geometry import intersect

    def pieces(s):
        g = slot_slice(seg, s, period)
        g = g[(seg["t1"][g] >= lo) & (seg["t0"][g] <= hi) & (seg["length"][g] >= cfg.tol_m)]
        if not len(g):
            return None
        px, py, pt = sub_points(seg, cx, cy, g, K)
        p0 = np.c_[px[:, :-1].ravel(), py[:, :-1].ravel()]
        p1 = np.c_[px[:, 1:].ravel(), py[:, 1:].ravel()]
        t0, t1 = pt[:, :-1].ravel(), pt[:, 1:].ravel()
        u0, u1 = np.clip((lo - t0) / (t1 - t0), 0, 1), np.clip((hi - t0) / (t1 - t0), 0, 1)
        keep = u1 > u0
        d = p1 - p0
        q0, q1 = p0 + u0[:, None] * d, p0 + u1[:, None] * d
        return q0[keep], q1[keep], (t0 + u0 * (t1 - t0))[keep], (t0 + u1 * (t1 - t0))[keep]

    A, B = pieces(sa), pieces(sb)
    if A is None or B is None:
        return []
    h = intersect(A[0], A[1], B[0], B[1], cfg.tol_m)
    if not len(h):
        return []
    ta = A[2][h.i] + h.ta * (A[3][h.i] - A[2][h.i])
    tb = B[2][h.j] + h.tb * (B[3][h.j] - B[2][h.j])
    hh = merge_same(dict(ta=ta, tb=tb, x=h.x, y=h.y, kind=h.kind.astype("<U9")), cfg.same_hit_s)
    o = order_key(hh["ta"], hh["tb"], hh["x"], hh["y"])
    return [(float(hh["ta"][i]), float(hh["tb"][i])) for i in o]
