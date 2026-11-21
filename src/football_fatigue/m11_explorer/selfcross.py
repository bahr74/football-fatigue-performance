"""Self-intersections of ONE player's continuous trajectory (loops / returns to the same place).

A self-crossing is descriptive only. It is never a two-player encounter, never an association candidate and never occlusion.

Method (per continuous provider chain; nothing is linked across a chain break):

1. Arc-length resampling: a vertex is kept once the player has moved ``step_m`` from the previous kept vertex. This removes
   stationary jitter (sub-``step_m`` wobble cannot form loops) and bounds the work.
2. Candidate segment pairs share a grid cell neighbourhood; adjacent segments (sharing an endpoint) are never tested.
3. Proper intersection (strict, so touching endpoints and collinear overlaps are excluded).
4. A hit is *meaningful* only if the time between the two visits is >= ``min_sep_s``, the path travelled between them is
   >= ``min_path_m`` and the loop reaches >= ``min_extent_m`` from the crossing point (rules out tiny wiggles).
5. Hits whose two visit times are both within ``merge_s`` of an earlier kept hit are merged (one loop, several tiny crossings).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SelfCrossConfig:
    step_m: float = 0.25
    cell_m: float = 1.0
    min_sep_s: float = 1.0
    min_path_m: float = 2.0
    min_extent_m: float = 0.5
    merge_s: float = 0.5
    est_low_quality_frac: float = 0.5

    def as_dict(self):
        return asdict(self)


COLUMNS = ["slot", "chain", "period", "x", "y", "t_first", "t_second", "dt_s", "path_m", "extent_m", "n_raw_hits", "est_frac_loop",
           "quality"]


def resample(t: np.ndarray, x: np.ndarray, y: np.ndarray, est: np.ndarray | None, step: float):
    """Indices of kept vertices (arc-length thinning; first and last always kept)."""
    keep = [0]
    lx, ly = x[0], y[0]
    for i in range(1, len(x)):
        if (x[i] - lx) ** 2 + (y[i] - ly) ** 2 >= step * step:
            keep.append(i)
            lx, ly = x[i], y[i]
    if keep[-1] != len(x) - 1:
        keep.append(len(x) - 1)
    return np.asarray(keep)


def _intersect(p, r, q, s):
    """Proper intersection of segments p + u r and q + v s (vectorised); returns (hit, u, v)."""
    den = r[:, 0] * s[:, 1] - r[:, 1] * s[:, 0]
    qp = q - p
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (qp[:, 0] * s[:, 1] - qp[:, 1] * s[:, 0]) / den
        v = (qp[:, 0] * r[:, 1] - qp[:, 1] * r[:, 0]) / den
    eps = 1e-9
    hit = (np.abs(den) > 1e-12) & (u > eps) & (u < 1 - eps) & (v > eps) & (v < 1 - eps)
    return hit, u, v


def _candidate_pairs(mid: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """All (i < j) whose midpoints fall in the same or a neighbouring cell (segments are <= cell long after thinning... mostly;
    longer ones are added against everything)."""
    cx = np.floor(mid[:, 0] / cell).astype(np.int64)
    cy = np.floor(mid[:, 1] / cell).astype(np.int64)
    buckets: dict[tuple[int, int], list[int]] = {}
    for i, k in enumerate(zip(cx.tolist(), cy.tolist())):
        buckets.setdefault(k, []).append(i)
    I, J = [], []
    for (a, b), idx in buckets.items():
        idx = np.asarray(idx)
        for da, db in ((0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)):
            other = buckets.get((a + da, b + db))
            if other is None:
                continue
            o = np.asarray(other)
            ii, jj = np.meshgrid(idx, o, indexing="ij")
            ii, jj = ii.ravel(), jj.ravel()
            if da == 0 and db == 0:
                m = ii < jj
                ii, jj = ii[m], jj[m]
            I.append(np.minimum(ii, jj))
            J.append(np.maximum(ii, jj))
    if not I:
        return np.zeros(0, int), np.zeros(0, int)
    return np.concatenate(I), np.concatenate(J)


def chain_self_crossings(t: np.ndarray, x: np.ndarray, y: np.ndarray, est: np.ndarray | None = None, *,
                         cfg: SelfCrossConfig = SelfCrossConfig(), slot: int = -1, chain: int = -1, period: int = 0) -> pd.DataFrame:
    """Meaningful self-crossings of one continuous chain (time-ordered samples, no NaN)."""
    t, x, y = (np.asarray(a, float) for a in (t, x, y))
    ok = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
    t, x, y = t[ok], x[ok], y[ok]
    est = None if est is None else np.asarray(est, bool)[ok]
    if len(t) < 4:
        return pd.DataFrame(columns=COLUMNS)
    k = resample(t, x, y, est, cfg.step_m)
    P = np.c_[x[k], y[k]]
    tk = t[k]
    if len(P) < 4:
        return pd.DataFrame(columns=COLUMNS)
    seg = np.diff(P, axis=0)
    seglen = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.r_[0.0, np.cumsum(seglen)]
    mid = (P[:-1] + P[1:]) / 2
    I, J = _candidate_pairs(mid, cfg.cell_m)
    long = np.flatnonzero(seglen > cfg.cell_m)                  # rare long pieces: test against segments inside their bounding box
    if len(long):
        lo, hi = np.minimum(P[:-1], P[1:]), np.maximum(P[:-1], P[1:])
        LI, LJ = [I], [J]
        for i in long:
            o = np.flatnonzero((hi[:, 0] >= lo[i, 0]) & (lo[:, 0] <= hi[i, 0]) & (hi[:, 1] >= lo[i, 1]) & (lo[:, 1] <= hi[i, 1]))
            LI.append(np.minimum(i, o))
            LJ.append(np.maximum(i, o))
        I, J = np.concatenate(LI), np.concatenate(LJ)
    m = J - I >= 2                                              # never adjacent segments (shared endpoint)
    I, J = I[m], J[m]
    if len(I):
        pr = np.unique(np.c_[I, J], axis=0)
        I, J = pr[:, 0], pr[:, 1]
    hit, u, v = _intersect(P[I], seg[I], P[J], seg[J])
    I, J, u, v = I[hit], J[hit], u[hit], v[hit]
    rows = []
    order = np.lexsort((J, I))
    kept: list[tuple[float, float]] = []
    n_raw: list[int] = []
    for n in order:
        i, j = int(I[n]), int(J[n])
        ta = tk[i] + u[n] * (tk[i + 1] - tk[i])
        tb = tk[j] + v[n] * (tk[j + 1] - tk[j])
        path = (cum[j] + v[n] * seglen[j]) - (cum[i] + u[n] * seglen[i])
        px, py = P[i] + u[n] * seg[i]
        loop = P[i + 1:j + 1]
        extent = float(np.max(np.hypot(loop[:, 0] - px, loop[:, 1] - py))) if len(loop) else 0.0
        if tb - ta < cfg.min_sep_s or path < cfg.min_path_m or extent < cfg.min_extent_m:
            continue
        dup = next((q for q, (a, b) in enumerate(kept) if abs(a - ta) <= cfg.merge_s and abs(b - tb) <= cfg.merge_s), None)
        if dup is not None:
            n_raw[dup] += 1
            continue
        kept.append((ta, tb))
        n_raw.append(1)
        if est is not None:
            sel = (t >= ta) & (t <= tb)
            ef = float(est[sel].mean()) if sel.any() else float("nan")
        else:
            ef = float("nan")
        rows.append(dict(slot=slot, chain=chain, period=period, x=float(px), y=float(py), t_first=float(ta), t_second=float(tb),
                         dt_s=float(tb - ta), path_m=float(path), extent_m=extent, n_raw_hits=1, est_frac_loop=ef,
                         quality="low" if (np.isfinite(ef) and ef >= cfg.est_low_quality_frac) else "ok"))
    for r, nr in zip(rows, n_raw):
        r["n_raw_hits"] = nr
    return pd.DataFrame(rows, columns=COLUMNS)


def segments_self_crossings(seg: dict[str, np.ndarray], est_of_row=None, *, slots=None, period: int | None = None, t0: float | None = None,
                            t1: float | None = None, cfg: SelfCrossConfig = SelfCrossConfig()) -> pd.DataFrame:
    """Self-crossings for every chain in a catalogue segment table (``window_intersections`` ``build_segments`` output), optionally
    restricted to slots / a period / a time window. Chains are processed separately, so a track break never creates a loop."""
    out = []
    chains = np.unique(seg["chain"])
    for ch in chains:
        g = np.flatnonzero(seg["chain"] == ch)
        s, p = int(seg["slot"][g[0]]), int(seg["period"][g[0]])
        if slots is not None and s not in slots:
            continue
        if period is not None and p != period:
            continue
        t = np.r_[seg["t0"][g], seg["t1"][g[-1]]]
        x = np.r_[seg["x0"][g], seg["x1"][g[-1]]]
        y = np.r_[seg["y0"][g], seg["y1"][g[-1]]]
        if t0 is not None:
            w = (t >= t0) & (t <= t1)
            if w.sum() < 4:
                continue
            t, x, y = t[w], x[w], y[w]
        est = None if est_of_row is None else est_of_row(s, t)
        df = chain_self_crossings(t, x, y, est, cfg=cfg, slot=s, chain=int(ch), period=p)
        if len(df):
            out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=COLUMNS)
