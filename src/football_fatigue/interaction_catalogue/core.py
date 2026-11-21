"""Interaction Catalogue v1 (M11, Session A Phase 1): ``m11-interaction-catalogue-v1``.

A descriptive, identity-free catalogue of player-player interactions in FIFA tracking. It is **not** a training dataset, and no
radius here is a supervised training threshold. Definitions (full text: ``docs/research/m11_interaction_catalogue_v1_methodology.md``):

**Trajectories.** The Step 1 valid segments and chains of ``window_intersections.core.build_segments`` (the same trajectories as the
historical Stage 2 candidates): linear between 15 Hz provider updates, never joined across a gap > 0.5 s, an implausible jump, an
unattributed / ambiguous sample, a duplicate-position (cloned) sample, a period boundary or a quarantined period. Nothing is
interpolated across a chain break. VISIBLE and ESTIMATED samples are treated identically.

**Nodes** are player chains (slot, period, chain index): an identity break is an explicit node boundary.

**Proximity edge (radius R).** For two chains that overlap in time, d(t) = |A(t) - B(t)| is piecewise quadratic between knots, so
{t : d(t) <= R} is found exactly (``window_intersections.candidates.episodes_on``). Every maximal interval is one edge, with start,
end, t* (time of the minimum), d_min and descriptors. An edge is counted once, however many sliding windows overlap it
(deduplication rule: *count edges, never windows*). At R = 0.5 m the edges are exactly the historical Stage 2 episodes.

**Edge chain** (repeated same-pair contact): consecutive edges of one pair with start - (latest end so far) <= ``chain_merge_gap_s``.
At R = 0.5 m and 1.0 s these are exactly the historical Stage 2 merged candidates. The member edges are kept.

**Geometric crossing**: the two players' paths meet in space at times tA and tB with |tA - tB| <= 2 * window_half_s
(``window_intersections.core.period_hits``, linear). **Every** crossing is kept, not only the first one of a window. A crossing is
*synchronous* when |tA - tB| <= ``sync_cross_s``; ``d_sync_m`` is the players' separation when the first of them is at the point.

**Groups** (multi-player interactions, per radius and link tolerance delta): connected components of edges, where two edges are
linked when they share a player and their intervals are at most delta apart. delta = 0: *simultaneous* groups (edges overlap in
time). delta = ``chain_merge_gap_s``: *temporally connected* groups. Each edge belongs to exactly one group per (R, delta). For
simultaneous groups the membership is followed through time: formation, joins, leaves, merges, splits, dissolution.

**Not observable here.** Visual occlusion, bounding-box overlap, tracker loss and identity switches need images. Proximity and
crossings are geometric facts about provider coordinates only. *Candidate identity-ambiguity* is a descriptive constant-velocity
indicator (``cv_margin_m``), not a label and not a sample selection.

**Quality (metadata only; never removes an edge).** Three separate concepts:

* eligibility: every edge built on validated chains is in the catalogue (``eligible`` is always True; excluded data are counted
  per match instead);
* measurement reliability: speed / acceleration anomalies, out-of-pitch samples, over-speed sub-metre steps, long ESTIMATED
  co-motion (``reliability`` = nominal / reduced, with ``reliability_reasons``). Provider status alone never reduces reliability;
* possible label corruption: identity-risk flags, implausible jumps, frame-wide shifts or duplicate positions near the edge, or an
  edge truncated by a chain boundary (``label_risk`` with ``label_risk_reasons``).

Identities (slot, player id, side) are used for pairing, grouping and audit, never as predictions.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from football_fatigue.encounters import quality as Q
from football_fatigue.window_intersections import candidates as C
from football_fatigue.window_intersections import core as W
from football_fatigue.window_intersections import local as L

SCHEMA = "m11-interaction-catalogue-v1"
IDENTITY_RISK = Q.LABEL_TRANSFER_CANDIDATE | Q.RECIPROCAL_EXCHANGE | Q.DISCONTINUITY_CANDIDATE | Q.SUBSTITUTION_WINDOW
CV_TIERS = ("cv_confusable", "marginal", "cv_separable", "unavailable")


@dataclass(frozen=True)
class CatalogueConfig:
    radii_m: tuple[float, ...] = (0.25, 0.5, 1.0, 1.5, 2.0)   # descriptive analysis radii, not training thresholds
    chain_merge_gap_s: float = 1.0           # repeated same-pair contact -> one edge chain (= historical Stage 2 merge gap)
    group_link_s: tuple[float, ...] = (0.0, 1.0)   # 0: simultaneous groups; 1.0: temporally connected groups
    window_half_s: float = 2.5               # 5 s sliding windows at 1 s stride (window_intersections v1 convention)
    sync_cross_s: float = 0.3                # crossing synchronous when |tA - tB| <= this
    edge_cross_pad_s: float = 0.5            # crossings with both times in [start - pad, end + pad] are attached to an edge
    boundary_margin_m: float = 0.05          # edge "boundary uncertain" when d_min > R - margin
    status_pad_s: float = 0.1                # status composition over [start - pad, end + pad]
    quality_pad_s: float = 1.0               # flags over [start - pad, end + pad]
    kin_half_s: float = 0.2                  # central-difference half step for speed / acceleration (smoothed kinematics)
    grid_s: float = 1.0 / 15.0
    max_speed_m_s: float = 12.0              # same physiological bound as encounters.quality
    max_accel_m_s2: float = 25.0             # ~2.5 g on 0.4 s-smoothed differences; the 3816 pilot showed 15 m/s2 flags 14 % of
                                             # edges (genuine cuts / turns), so only gross anomalies are flagged; distribution reported
    oob_margin_m: float = 3.0                # out-of-pitch: beyond the touch / goal line by more than this
    comotion_radius_m: float = 0.25
    comotion_min_s: float = 1.0              # long co-motion: >= this within comotion_radius ...
    comotion_est_frac: float = 0.95          # ... with both players at least this ESTIMATED
    cv_gap_s: float = 0.8                    # ambiguity indicator: hidden interval centred at t*
    cv_history_s: float = 0.4                # velocity from the chord over this before the hidden interval
    cv_separable_m: float = 0.5
    lifecycle_min_players: int = 3           # membership timelines stored for simultaneous groups of at least this size

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def rkey(r: float) -> str:
    return f"r{r:.2f}".replace(".", "p")


def _hash(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


# ---------------------------------------------------------------- frame-level context (status and flags)

@dataclass
class FrameContext:
    """Per-period sample arrays (rows x slots) used for status and quality metadata."""
    t: np.ndarray            # row times (s), strictly increasing
    mask: np.ndarray         # uint32 encounters.quality mask bits
    avail: np.ndarray        # bool, AVAILABLE sample
    dup: np.ndarray          # bool, bit-identical to another slot (cloned position)
    oob: np.ndarray          # bool, outside the pitch by more than the margin

    def rows(self, lo: float, hi: float) -> slice:
        return slice(int(np.searchsorted(self.t, lo, "left")), int(np.searchsorted(self.t, hi, "right")))


def edge_quality(fc: FrameContext, sa: int, sb: int, start: float, end: float, cfg: CatalogueConfig) -> dict[str, Any]:
    core = fc.rows(start - cfg.status_pad_s, end + cfg.status_pad_s)
    pad = fc.rows(start - cfg.quality_pad_s, end + cfg.quality_pad_s)
    out: dict[str, Any] = {}
    for tag, s in (("a", sa), ("b", sb)):
        m = fc.mask[core, s]
        upd = fc.avail[core, s] & ((m & Q.UPDATE_INSTANT) != 0)
        n = int(upd.sum())
        out[f"est_frac_{tag}"] = float(((m[upd] & Q.ESTIMATED) != 0).mean()) if n else float("nan")
        out[f"n_status_updates_{tag}"] = n
    ea, eb = out["est_frac_a"], out["est_frac_b"]
    if not (np.isfinite(ea) and np.isfinite(eb)):
        out["status_composition"] = "unknown"
    elif ea == 0 and eb == 0:
        out["status_composition"] = "both_visible"
    elif ea == 1 and eb == 1:
        out["status_composition"] = "both_estimated"
    else:
        out["status_composition"] = "mixed"
    mp = fc.mask[pad][:, [sa, sb]]
    out["n_jump_flags"] = int(((mp & Q.IMPLAUSIBLE_JUMP) != 0).sum())
    out["n_overspeed_flags"] = int(((mp & Q.OVERSPEED_STEP) != 0).sum())
    out["n_frame_shift_flags"] = int(((mp & Q.FRAME_WIDE_SHIFT) != 0).sum())
    out["n_identity_risk_flags"] = int(((mp & IDENTITY_RISK) != 0).sum())
    out["n_duplicate_samples"] = int(fc.dup[pad][:, [sa, sb]].sum())
    out["n_oob_samples"] = int(fc.oob[pad][:, [sa, sb]].sum())
    return out


# ---------------------------------------------------------------- kinematics on knot tracks

def _p(trk, t):
    T, X, Y = trk
    return np.interp(t, T, X), np.interp(t, T, Y)


def chord_velocity(trk, t: float, h: float) -> np.ndarray:
    """Central chord velocity over [t - h, t + h]; NaN unless the track covers the whole interval."""
    T = trk[0]
    if t - h < T[0] - 1e-9 or t + h > T[-1] + 1e-9:
        return np.full(2, np.nan)
    x0, y0 = _p(trk, t - h)
    x1, y1 = _p(trk, t + h)
    return np.array([x1 - x0, y1 - y0]) / (2 * h)


def max_kinematics(trk, lo: float, hi: float, h: float, step: float) -> tuple[float, float]:
    """Max smoothed speed and acceleration (central differences with half step h) on a grid inside [lo, hi] and the track."""
    T = trk[0]
    a, b = max(lo, float(T[0]) + 2 * h), min(hi, float(T[-1]) - 2 * h)
    if b < a:
        return float("nan"), float("nan")
    g = np.arange(a, b + 1e-9, step) if b > a else np.array([a])
    xm, ym = _p(trk, g - h)
    xp, yp = _p(trk, g + h)
    v = np.hypot(xp - xm, yp - ym) / (2 * h)
    x2m, y2m = _p(trk, g - 2 * h)
    x0, y0 = _p(trk, g)
    x2p, y2p = _p(trk, g + 2 * h)
    acc = np.hypot(x2p - 2 * x0 + x2m, y2p - 2 * y0 + y2m) / (4 * h * h)
    return float(v.max()), float(acc.max())


def cv_margin(KA, KB, ts: float, gap: float, hist: float) -> float:
    """Constant-velocity ambiguity indicator. Hide [ts - gap/2, ts + gap/2]; extrapolate each player with the chord velocity of the
    preceding ``hist`` seconds; margin = (cost of the swapped assignment) - (cost of the provider-chain assignment), summed endpoint
    errors in metres. Negative: constant velocity would exchange the two players. NaN when either chain lacks coverage."""
    g0, g1, s = ts - gap / 2, ts + gap / 2, ts - gap / 2 - hist
    for K in (KA, KB):
        if s < K[0][0] - 1e-9 or g1 > K[0][-1] + 1e-9:
            return float("nan")
    pred, end = [], []
    for K in (KA, KB):
        x0, y0 = _p(K, g0)
        xs, ys = _p(K, s)
        pred.append(np.array([x0 + (x0 - xs) / hist * gap, y0 + (y0 - ys) / hist * gap]))
        end.append(np.array(_p(K, g1)))
    m = np.hypot(*(pred[0] - end[0])) + np.hypot(*(pred[1] - end[1]))
    w = np.hypot(*(pred[0] - end[1])) + np.hypot(*(pred[1] - end[0]))
    return float(w - m)


def cv_tier(m: float, cfg: CatalogueConfig) -> str:
    if not np.isfinite(m):
        return "unavailable"
    return "cv_confusable" if m < 0 else ("marginal" if m < cfg.cv_separable_m else "cv_separable")


# ---------------------------------------------------------------- edges

def chains_by_slot(ch, period: int) -> dict[int, np.ndarray]:
    sel = np.flatnonzero(ch["period"] == period)
    out: dict[int, list[int]] = defaultdict(list)
    for k in sel:
        out[int(ch["slot"][k])].append(int(k))
    return {s: np.array(v, int) for s, v in out.items()}


def _far(KA, KB, R: float) -> bool:
    return bool(KA[1].min() - KB[1].max() > R or KB[1].min() - KA[1].max() > R or KA[2].min() - KB[2].max() > R or KB[2].min() - KA[2].max() > R)


def pair_edges(seg, ch, cx, cy, cbs: dict[int, np.ndarray], sa: int, sb: int, cfg: CatalogueConfig, fc: FrameContext | None) -> list[dict[str, Any]]:
    """Every proximity edge of one pair (all radii), with kinematic, ambiguity and quality descriptors."""
    if sa not in cbs or sb not in cbs:
        return []
    out = []
    rmax = max(cfg.radii_m)
    for a, b, lo, hi in C.overlaps(ch, cbs[sa], cbs[sb]):
        KA = L.chain_knots(seg, cx, cy, int(ch["first"][a]), int(ch["last"][a]), lo, hi, 1)
        KB = L.chain_knots(seg, cx, cy, int(ch["first"][b]), int(ch["last"][b]), lo, hi, 1)
        if KA is None or KB is None or _far(KA, KB, rmax):
            continue
        for R in cfg.radii_m:
            if _far(KA, KB, R):
                continue
            for e in C.episodes_on(KA, KB, R):
                ts = e["t_star"]
                r = dict(radius_m=R, slot_a=sa, slot_b=sb, chain_a=a, chain_b=b, start=e["start"], end=e["end"],
                         duration_s=e["end"] - e["start"], t_star=ts, d_min=e["d_min"], n_local_minima=e["n_local_minima"],
                         trunc_before=e["start"] <= lo + 1e-9, trunc_after=e["end"] >= hi - 1e-9)
                (xa, ya), (xb, yb) = _p(KA, ts), _p(KB, ts)
                r.update(x_a=float(xa), y_a=float(ya), x_b=float(xb), y_b=float(yb))
                va, vb = chord_velocity(KA, ts, cfg.kin_half_s), chord_velocity(KB, ts, cfg.kin_half_s)
                r["speed_a_m_s"], r["speed_b_m_s"] = float(np.hypot(*va)), float(np.hypot(*vb))
                r["rel_speed_m_s"] = float(np.hypot(*(va - vb)))
                na, nb = np.hypot(*va), np.hypot(*vb)
                r["rel_heading_deg"] = (float(np.degrees(np.arccos(np.clip(va @ vb / (na * nb), -1, 1))))
                                        if np.isfinite(na * nb) and na > 0.2 and nb > 0.2 else float("nan"))
                if R >= cfg.comotion_radius_m:
                    A2, B2 = L.clip_track(KA, e["start"], e["end"]), L.clip_track(KB, e["start"], e["end"])
                    r["time_within_comotion_s"] = float(L.pair_min(A2, B2, cfg.comotion_radius_m)[2] or 0.0)
                else:
                    r["time_within_comotion_s"] = r["duration_s"]
                spa, aca = max_kinematics(KA, e["start"] - cfg.quality_pad_s, e["end"] + cfg.quality_pad_s, cfg.kin_half_s, cfg.grid_s)
                spb, acb = max_kinematics(KB, e["start"] - cfg.quality_pad_s, e["end"] + cfg.quality_pad_s, cfg.kin_half_s, cfg.grid_s)
                r["max_speed_m_s"] = float(np.nanmax([spa, spb])) if np.isfinite([spa, spb]).any() else float("nan")
                r["max_accel_m_s2"] = float(np.nanmax([aca, acb])) if np.isfinite([aca, acb]).any() else float("nan")
                r["cv_margin_m"] = cv_margin(KA, KB, ts, cfg.cv_gap_s, cfg.cv_history_s)
                r["cv_tier"] = cv_tier(r["cv_margin_m"], cfg)
                r["boundary_uncertain"] = bool(e["d_min"] > R - cfg.boundary_margin_m)
                if fc is not None:
                    r.update(edge_quality(fc, sa, sb, e["start"], e["end"], cfg))
                out.append(r)
    return out


def classify_quality(df: pd.DataFrame, cfg: CatalogueConfig) -> pd.DataFrame:
    """Eligibility, measurement reliability and possible label corruption: three separate columns (status alone never decides)."""
    df["eligible"] = True
    rel = pd.Series([""] * len(df), index=df.index, dtype=object)

    def add(s: pd.Series, cond: pd.Series, name: str) -> pd.Series:
        return s.where(~cond, s + np.where(s == "", "", ";") + name)

    rel = add(rel, df["max_speed_m_s"] > cfg.max_speed_m_s, "speed_anomaly")
    rel = add(rel, df["max_accel_m_s2"] > cfg.max_accel_m_s2, "accel_anomaly")
    for col, name in (("n_overspeed_flags", "overspeed_step"), ("n_oob_samples", "out_of_pitch")):
        if col in df:
            rel = add(rel, df[col] > 0, name)
    if "est_frac_a" in df:
        df["estimated_comotion"] = ((df["est_frac_a"] >= cfg.comotion_est_frac) & (df["est_frac_b"] >= cfg.comotion_est_frac)
                                    & (df["time_within_comotion_s"] >= cfg.comotion_min_s))
        rel = add(rel, df["estimated_comotion"], "estimated_comotion")
    df["reliability_reasons"] = rel
    df["reliability"] = np.where(rel == "", "nominal", "reduced")
    lab = pd.Series([""] * len(df), index=df.index, dtype=object)
    for col, name in (("n_identity_risk_flags", "identity_risk_flag"), ("n_jump_flags", "implausible_jump"),
                      ("n_frame_shift_flags", "frame_wide_shift"), ("n_duplicate_samples", "duplicate_position")):
        if col in df:
            lab = add(lab, df[col] > 0, name)
    lab = add(lab, df["trunc_before"] | df["trunc_after"], "chain_boundary")
    df["label_risk_reasons"] = lab
    df["label_risk"] = lab != ""
    return df


# ---------------------------------------------------------------- crossings and windows

def crossings_period(seg, cx, cy, period: int, pstart: float, cfg: CatalogueConfig) -> pd.DataFrame:
    """Every geometric crossing (linear paths) with |tA - tB| <= 2 * window_half_s; none is discarded."""
    h, _ = W.period_hits(seg, cx, cy, period, pstart, W.WindowConfig(half_s=cfg.window_half_s), "linear")
    n = len(h["ta"])
    if not n:
        return pd.DataFrame(columns=["period", "slot_a", "slot_b", "ta", "tb", "x", "y", "kind", "abs_dt_s", "synchronous", "d_sync_m"])
    df = pd.DataFrame(dict(period=period, slot_a=h["sa"].astype(int), slot_b=h["sb"].astype(int), ta=h["ta"], tb=h["tb"],
                           x=h["x"], y=h["y"], kind=h["kind"], chain_a=seg["chain"][h["ga"].astype(int)],
                           chain_b=seg["chain"][h["gb"].astype(int)]))
    df["abs_dt_s"] = (df["ta"] - df["tb"]).abs()
    df["synchronous"] = df["abs_dt_s"] <= cfg.sync_cross_s
    d_ta, d_tb = np.full(n, np.nan), np.full(n, np.nan)
    for (sa, sb), g in df.groupby(["slot_a", "slot_b"]).groups.items():
        g = np.asarray(g)
        bx, by = W.positions_at(seg, cx, cy, int(sb), period, df["ta"].to_numpy()[g])     # B when A is at the point
        ax, ay = W.positions_at(seg, cx, cy, int(sa), period, df["tb"].to_numpy()[g])     # A when B is at the point
        d_ta[g] = np.hypot(bx - df["x"].to_numpy()[g], by - df["y"].to_numpy()[g])
        d_tb[g] = np.hypot(ax - df["x"].to_numpy()[g], ay - df["y"].to_numpy()[g])
    first_a = df["ta"].to_numpy() <= df["tb"].to_numpy()
    df["d_sync_m"] = np.where(first_a, d_ta, d_tb)           # separation when the first player is at the crossing point
    df = df.sort_values(["slot_a", "slot_b", "ta", "tb"]).reset_index(drop=True)
    return df


def window_range(t_lo: np.ndarray, t_hi: np.ndarray, pstart: float, k0: int, k1: int, half: float, contain: bool) -> tuple[np.ndarray, np.ndarray]:
    """Complete-window anchors k (period seconds) per item. contain=True: both t_lo and t_hi inside [k - half, k + half]
    (crossing membership, as v1). contain=False: the window overlaps [t_lo, t_hi] (edge membership)."""
    a, b = (t_hi, t_lo) if contain else (t_lo, t_hi)
    lo = np.maximum(k0, np.ceil(a - pstart - half - 1e-9)).astype(int)
    hi = np.minimum(k1, np.floor(b - pstart + half + 1e-9)).astype(int)
    return lo, hi


def pair_windows(cr: pd.DataFrame, pstart: float, k0: int, k1: int, half: float) -> pd.DataFrame:
    """(pair, anchor) -> number of crossings (and synchronous crossings) inside that complete window."""
    if not len(cr):
        return pd.DataFrame(columns=["period", "slot_a", "slot_b", "anchor", "n_intersections", "n_synchronous"])
    lo, hi = window_range(np.minimum(cr["ta"], cr["tb"]).to_numpy(), np.maximum(cr["ta"], cr["tb"]).to_numpy(), pstart, k0, k1, half, True)
    n = np.clip(hi - lo + 1, 0, None)
    idx = np.repeat(np.arange(len(cr)), n)
    k = lo[idx] + (np.arange(len(idx)) - np.repeat(np.cumsum(n) - n, n))
    t = pd.DataFrame(dict(slot_a=cr["slot_a"].to_numpy()[idx], slot_b=cr["slot_b"].to_numpy()[idx], anchor=k,
                          sync=cr["synchronous"].to_numpy()[idx].astype(int)))
    g = t.groupby(["slot_a", "slot_b", "anchor"]).agg(n_intersections=("sync", "size"), n_synchronous=("sync", "sum")).reset_index()
    g.insert(0, "period", int(cr["period"].iloc[0]))
    return g


def union_length(lo: np.ndarray, hi: np.ndarray) -> int:
    """Number of integers covered by the union of integer ranges [lo, hi]."""
    ok = hi >= lo
    lo, hi = lo[ok], hi[ok]
    if not len(lo):
        return 0
    o = np.argsort(lo, kind="stable")
    tot, cs, ce = 0, lo[o[0]], hi[o[0]]
    for i in o[1:]:
        if lo[i] <= ce + 1:
            ce = max(ce, hi[i])
        else:
            tot += ce - cs + 1
            cs, ce = lo[i], hi[i]
    return int(tot + ce - cs + 1)


# ---------------------------------------------------------------- chains, concurrency and groups

def edge_chains(start: np.ndarray, end: np.ndarray, gap: float) -> np.ndarray:
    """Chain index of each edge of ONE pair: sorted by (start, end); a new chain when start - max(end so far) > gap."""
    o = np.lexsort((end, start))
    lab = np.empty(len(start), int)
    c, cur = -1, -np.inf
    for i in o:
        if start[i] - cur > gap + 1e-12:
            c += 1
            cur = end[i]
        else:
            cur = max(cur, end[i])
        lab[i] = c
    return lab


def link_components(start: np.ndarray, end: np.ndarray, pa: np.ndarray, pb: np.ndarray, delta: float) -> np.ndarray:
    """Group label per edge: union of edges sharing a player whose intervals are at most ``delta`` apart (per-player sweep)."""
    n = len(start)
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    inc: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        inc[int(pa[i])].append(i)
        inc[int(pb[i])].append(i)
    for idx in inc.values():
        idx = sorted(idx, key=lambda i: (start[i], end[i]))
        prev, cur = None, -np.inf
        for i in idx:
            if prev is not None and start[i] <= cur + delta + 1e-12:
                ri, rp = find(i), find(prev)
                if ri != rp:
                    parent[max(ri, rp)] = min(ri, rp)
                cur = max(cur, end[i])
            else:
                cur = end[i]
            prev = i
    roots = np.array([find(i) for i in range(n)], int)
    first = {}
    for i in np.lexsort((end, start)):
        first.setdefault(int(roots[i]), len(first))
    return np.array([first[int(r)] for r in roots], int)


def shared_counts(start: np.ndarray, end: np.ndarray, pa: np.ndarray, pb: np.ndarray, gap: float) -> tuple[np.ndarray, np.ndarray]:
    """Per edge: number of edges with a *different partner* that share a player and overlap in time (simultaneous), and that
    share a player without overlapping but within ``gap`` (sequential)."""
    n = len(start)
    conc, seq = np.zeros(n, int), np.zeros(n, int)
    inc: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        inc[int(pa[i])].append(i)
        inc[int(pb[i])].append(i)
    for p, idx in inc.items():
        idx = np.array(idx)
        if len(idx) < 2:
            continue
        s, e = start[idx], end[idx]
        other = np.where(pa[idx] == p, pb[idx], pa[idx])
        ov = (s[:, None] <= e[None, :]) & (e[:, None] >= s[None, :])
        sep = np.maximum(s[:, None] - e[None, :], s[None, :] - e[:, None])
        diff = other[:, None] != other[None, :]
        np.fill_diagonal(diff, False)
        conc[idx] += (ov & diff).sum(1)
        seq[idx] += (~ov & diff & (sep <= gap + 1e-12)).sum(1)
    return conc, seq


def _components(edges: list[tuple[int, int]]) -> list[frozenset]:
    parent: dict[int, int] = {}

    def find(i: int) -> int:
        parent.setdefault(i, i)
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    comp: dict[int, set] = defaultdict(set)
    for v in list(parent):
        comp[find(v)].add(v)
    return [frozenset(c) for c in comp.values()]


def lifecycle(start: np.ndarray, end: np.ndarray, pa: np.ndarray, pb: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Membership of one group through time. States are evaluated at every boundary and between boundaries; transitions count
    joins (a player appears), leaves, merges (one component formed from >= 2) and splits (one component -> >= 2)."""
    b = np.unique(np.r_[start, end])
    pts = np.sort(np.r_[b, (b[1:] + b[:-1]) / 2]) if len(b) > 1 else b
    states, timeline = [], []
    for t in pts:
        act = np.flatnonzero((start <= t + 1e-12) & (end >= t - 1e-12))
        st = frozenset(_components([(int(pa[i]), int(pb[i])) for i in act]))
        if not states or st != states[-1][1]:
            states.append((float(t), st, len(act)))
    stats = dict(n_states=len(states), n_joins=0, n_leaves=0, n_merges=0, n_splits=0,
                 max_simultaneous_players=max((max((len(c) for c in s), default=0) for _, s, _ in states), default=0),
                 max_concurrent_edges=max((k for _, _, k in states), default=0))
    for (_, s0, _), (_, s1, _) in zip(states[:-1], states[1:]):
        p0 = set().union(*s0) if s0 else set()
        p1 = set().union(*s1) if s1 else set()
        stats["n_joins"] += len(p1 - p0)
        stats["n_leaves"] += len(p0 - p1)
        stats["n_merges"] += sum(max(0, sum(1 for c0 in s0 if c0 & c1) - 1) for c1 in s1)
        stats["n_splits"] += sum(max(0, sum(1 for c1 in s1 if c0 & c1) - 1) for c0 in s0)
    for i, (t, s, k) in enumerate(states):
        timeline.append(dict(state=i, t=t, n_active_edges=k, n_players=sum(len(c) for c in s),
                             components="|".join(sorted(";".join(str(v) for v in sorted(c)) for c in s))))
    return stats, timeline


# ---------------------------------------------------------------- one period

def period_catalogue(seg, ch, cx, cy, period: int, pstart: float, pend: float, fc: FrameContext | None, slot_side: np.ndarray,
                     match_id: int, cfg: CatalogueConfig) -> dict[str, pd.DataFrame]:
    cbs = chains_by_slot(ch, period)
    slots = sorted(cbs)
    rows: list[dict[str, Any]] = []
    for i, sa in enumerate(slots):
        for sb in slots[i + 1:]:
            rows += pair_edges(seg, ch, cx, cy, cbs, sa, sb, cfg, fc)
    ed = pd.DataFrame(rows)
    k_all, comp = W.period_anchors(pstart, pend, cfg.window_half_s)
    kc = k_all[comp]
    k0, k1 = (int(kc.min()), int(kc.max())) if len(kc) else (0, -1)
    cr = crossings_period(seg, cx, cy, period, pstart, cfg)
    cr.insert(0, "match_id", match_id)
    cr["crossing_id"] = [_hash(match_id, period, a, b, round(x * 1e6), round(y * 1e6)) for a, b, x, y in zip(cr["slot_a"], cr["slot_b"], cr["ta"], cr["tb"])]
    pw = pair_windows(cr, pstart, k0, k1, cfg.window_half_s)
    pw.insert(0, "match_id", match_id)
    for R in cfg.radii_m:          # proximity class of each crossing: smallest radius containing d_sync
        cr[f"within_{rkey(R)}"] = cr["d_sync_m"] <= R + 1e-12
    out = dict(crossings=cr, pair_windows=pw)
    if not len(ed):
        out.update(edges=ed, edge_chains=pd.DataFrame(), groups=pd.DataFrame(), group_timeline=pd.DataFrame(),
                   window_stats=pd.DataFrame())
        return out
    ed.insert(0, "period", period)
    ed.insert(0, "match_id", match_id)
    ed = classify_quality(ed, cfg)
    ed["team_relation"] = np.where(slot_side[ed["slot_a"]] == slot_side[ed["slot_b"]], "same_team", "opposite_teams")
    ed["t_star_rel_s"] = ed["t_star"] - pstart
    ed["edge_id"] = [_hash(match_id, period, a, b, R, round(s * 1e6), round(e * 1e6))
                     for a, b, R, s, e in zip(ed["slot_a"], ed["slot_b"], ed["radius_m"], ed["start"], ed["end"])]
    ed["node_a"] = [f"{match_id}:{period}:{s}:{c}" for s, c in zip(ed["slot_a"], ed["chain_a"])]
    ed["node_b"] = [f"{match_id}:{period}:{s}:{c}" for s, c in zip(ed["slot_b"], ed["chain_b"])]
    # crossings attached to each edge (same pair, both crossing times inside the padded edge)
    ed["n_crossings_in_edge"], ed["n_sync_crossings_in_edge"] = 0, 0
    crg = {k: g for k, g in cr.groupby(["slot_a", "slot_b"])} if len(cr) else {}
    for (sa, sb), g in ed.groupby(["slot_a", "slot_b"]).groups.items():
        c = crg.get((sa, sb))
        if c is None:
            continue
        tlo, thi = np.minimum(c["ta"], c["tb"]).to_numpy(), np.maximum(c["ta"], c["tb"]).to_numpy()
        syn = c["synchronous"].to_numpy()
        s, e = ed.loc[g, "start"].to_numpy() - cfg.edge_cross_pad_s, ed.loc[g, "end"].to_numpy() + cfg.edge_cross_pad_s
        inside = (tlo[None, :] >= s[:, None]) & (thi[None, :] <= e[:, None])
        ed.loc[g, "n_crossings_in_edge"] = inside.sum(1)
        ed.loc[g, "n_sync_crossings_in_edge"] = (inside & syn[None, :]).sum(1)
    # windows overlapping each edge (what a sliding-window detector would report)
    lo, hi = window_range(ed["start"].to_numpy(), ed["end"].to_numpy(), pstart, k0, k1, cfg.window_half_s, False)
    ed["n_windows"] = np.clip(hi - lo + 1, 0, None)
    ed["_wlo"], ed["_whi"] = lo, hi
    chains, groups, timeline, wst = [], [], [], []
    for R, er in ed.groupby("radius_m"):
        er = er.copy()
        # edge chains (repeated same-pair contact)
        ec = np.empty(len(er), int)
        cid = np.empty(len(er), object)
        for (sa, sb), g in er.groupby(["slot_a", "slot_b"]).groups.items():
            pos = er.index.get_indexer(g)
            lab = edge_chains(er["start"].to_numpy()[pos], er["end"].to_numpy()[pos], cfg.chain_merge_gap_s)
            ec[pos] = lab
            for lb in np.unique(lab):
                pp = pos[lab == lb]
                c0 = float(er["start"].to_numpy()[pp].min())
                cid[pp] = _hash(match_id, period, sa, sb, R, "chain", round(c0 * 1e6))
        er["edge_chain_id"] = cid
        ed.loc[er.index, "edge_chain_id"] = cid
        ch_rows = er.groupby("edge_chain_id")
        best = er.loc[er.sort_values(["d_min", "t_star"]).groupby("edge_chain_id").head(1).index].set_index("edge_chain_id")
        agg = ch_rows.agg(slot_a=("slot_a", "first"), slot_b=("slot_b", "first"), n_edges=("edge_id", "size"), start=("start", "min"),
                          end=("end", "max"), time_within_s=("duration_s", "sum"), n_crossings=("n_crossings_in_edge", "sum"),
                          n_sync_crossings=("n_sync_crossings_in_edge", "sum"))
        agg["t_star"], agg["d_min"] = best["t_star"], best["d_min"]
        agg = agg.reset_index()
        agg.insert(0, "radius_m", R)
        agg.insert(0, "period", period)
        agg.insert(0, "match_id", match_id)
        chains.append(agg)
        ed.loc[er.index, "order_in_chain"] = er.groupby("edge_chain_id")["start"].rank(method="first").astype(int)
        # concurrency with other partners
        st, en = er["start"].to_numpy(), er["end"].to_numpy()
        pa, pb = er["slot_a"].to_numpy(), er["slot_b"].to_numpy()
        conc, seq = shared_counts(st, en, pa, pb, cfg.chain_merge_gap_s)
        ed.loc[er.index, "n_concurrent_other"] = conc
        ed.loc[er.index, "n_sequential_other"] = seq
        # unique pair-windows touched by edges (window inflation)
        uw = sum(union_length(g["_wlo"].to_numpy(), g["_whi"].to_numpy()) for _, g in er.groupby(["slot_a", "slot_b"]))
        wst.append(dict(match_id=match_id, period=period, radius_m=R, edges=len(er), window_detections=int(er["n_windows"].sum()),
                        unique_pair_windows=int(uw)))
        for delta in cfg.group_link_s:
            lab = link_components(st, en, pa, pb, delta)
            dk = f"{delta:.2f}".replace(".", "p")
            gids = {}
            for gl in np.unique(lab):
                ii = np.flatnonzero(lab == gl)
                gids[gl] = _hash(match_id, period, R, dk, round(st[ii].min() * 1e6), *sorted(set(pa[ii]) | set(pb[ii])))
            gcol = np.array([gids[v] for v in lab], object)
            ed.loc[er.index, f"group_d{dk}"] = gcol
            for gl in np.unique(lab):
                ii = np.flatnonzero(lab == gl)
                players = sorted(set(pa[ii].tolist()) | set(pb[ii].tolist()))
                sides = slot_side[players]
                g = dict(match_id=match_id, period=period, radius_m=R, link_s=delta, group_id=gids[gl], n_edges=len(ii),
                         n_players=len(players), n_pairs=len({(int(pa[i]), int(pb[i])) for i in ii}),
                         players=";".join(str(p) for p in players), n_home=int((sides == "home").sum()), n_away=int((sides == "away").sum()),
                         start=float(st[ii].min()), end=float(en[ii].max()))
                g["duration_s"] = g["end"] - g["start"]
                sub = er.iloc[ii]
                g["n_sync_crossings"] = int(sub["n_sync_crossings_in_edge"].sum())
                if "status_composition" in sub:
                    g["frac_edges_both_estimated"] = float((sub["status_composition"] == "both_estimated").mean())
                    g["any_estimated_comotion"] = bool(sub.get("estimated_comotion", pd.Series(False)).any())
                g["any_label_risk"] = bool(sub.get("label_risk", pd.Series(False)).any())
                if delta == 0.0:
                    if len(ii) > 1:
                        stats, tl = lifecycle(st[ii], en[ii], pa[ii], pb[ii])
                    else:
                        stats, tl = dict(n_states=1, n_joins=0, n_leaves=0, n_merges=0, n_splits=0, max_simultaneous_players=2,
                                         max_concurrent_edges=1), []
                    g.update(stats)
                    if len(players) >= cfg.lifecycle_min_players:
                        timeline += [dict(match_id=match_id, period=period, radius_m=R, group_id=gids[gl], **r) for r in tl]
                groups.append(g)
        ed.loc[er.index, "order_in_group_d0p00"] = er.assign(_g=ed.loc[er.index, "group_d0p00"]).groupby("_g")["start"].rank(method="first").astype(int)
    ed = ed.drop(columns=["_wlo", "_whi"])
    out.update(edges=ed, edge_chains=pd.concat(chains, ignore_index=True), groups=pd.DataFrame(groups),
               group_timeline=pd.DataFrame(timeline), window_stats=pd.DataFrame(wst))
    return out


def nodes_table(seg, ch, c_slot_player: np.ndarray, slot_side: np.ndarray, match_id: int) -> pd.DataFrame:
    s = ch["slot"]
    return pd.DataFrame(dict(match_id=match_id, node_id=[f"{match_id}:{p}:{sl}:{k}" for k, (sl, p) in enumerate(zip(s, ch["period"]))],
                             period=ch["period"], slot=s, chain=np.arange(len(s)), player_id=c_slot_player[s], side=slot_side[s],
                             t0=ch["t0"], t1=ch["t1"], duration_s=ch["t1"] - ch["t0"], n_segments=ch["n_segments"]))
