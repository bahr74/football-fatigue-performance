"""Ball cleaning and kinematics on raw positions. Every removed sample is counted; nothing is interpolated.

Raw PFF positions update about every second frame (audit §4.2), so a sample identical to the previous one is a repeat, not
a new measurement; using it would produce a zero/double velocity sawtooth. Repeats are therefore ignored for fitting (they
are not deleted from the source arrays). Single-sample spikes (implied speed above ``max_ball_speed_mps`` both into and out
of the sample) are dropped and counted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DetectorConfig
from .inputs import MatchPositions


@dataclass
class BallTrack:
    x: np.ndarray            # (F,) cleaned fresh samples, NaN elsewhere
    y: np.ndarray
    z: np.ndarray
    fresh: np.ndarray        # (F,) bool: a new, accepted measurement
    present: np.ndarray      # (F,) bool: the source has a ball in this frame (fresh or repeated)
    v_before: np.ndarray     # (F, 2) velocity fitted on the half window ending at the frame (m/s), NaN when not fittable
    v_after: np.ndarray      # (F, 2) velocity fitted on the half window starting at the frame
    stats: dict[str, int]

    @property
    def dv(self) -> np.ndarray:
        """|v_after - v_before| per frame (NaN when either side is not fittable)."""
        return np.hypot(*(self.v_after - self.v_before).T)

    @property
    def speed_after(self) -> np.ndarray:
        return np.hypot(*self.v_after.T)


def _window_sum(a: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """s[i] = sum(a[i+lo : i+hi+1]) with out-of-range terms treated as zero."""
    c = np.concatenate([[0.0], np.cumsum(a)])
    n = a.shape[0]
    i = np.arange(n)
    start = np.clip(i + lo, 0, n)
    stop = np.clip(i + hi + 1, 0, n)
    return c[stop] - c[start]


def _fit_slope(t: np.ndarray, v: np.ndarray, w: np.ndarray, lo: int, hi: int, min_n: int) -> np.ndarray:
    """Least-squares slope of v(t) over window offsets [lo, hi] using only samples with weight 1."""
    n = _window_sum(w, lo, hi)
    st = _window_sum(w * t, lo, hi)
    sv = _window_sum(w * v, lo, hi)
    stt = _window_sum(w * t * t, lo, hi)
    stv = _window_sum(w * t * v, lo, hi)
    den = n * stt - st * st
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = (n * stv - st * sv) / den
    slope[(n < min_n) | ~(np.abs(den) > 1e-9)] = np.nan
    return slope


def ball_track(pos: MatchPositions, cfg: DetectorConfig) -> BallTrack:
    bx = pos.ball[:, 0].astype(np.float64)
    by = pos.ball[:, 1].astype(np.float64)
    bz = pos.ball[:, 2].astype(np.float64)
    present = ~np.isnan(bx) & ~np.isnan(by)
    same = np.zeros_like(present)
    same[1:] = present[1:] & present[:-1] & (bx[1:] == bx[:-1]) & (by[1:] == by[:-1]) & (pos.period[1:] == pos.period[:-1])
    # within a run of identical samples every second one is a held repeat (A A B B -> A B); a resting ball keeps half of its
    # samples (A A A A -> A A), so it still yields a zero velocity instead of no velocity at all
    run_start = np.maximum.accumulate(np.where(~same, np.arange(same.size), 0))
    repeat = same & ((np.arange(same.size) - run_start) % 2 == 1)
    fresh = present & ~repeat

    # single-sample spikes between fresh neighbours of the same period
    idx = np.flatnonzero(fresh)
    spikes = 0
    if idx.size >= 3:
        t, x, y, per = pos.t[idx], bx[idx], by[idx], pos.period[idx]
        dt = np.diff(t)
        sp = np.hypot(np.diff(x), np.diff(y)) / np.where(dt > 0, dt, np.nan)
        same = per[1:] == per[:-1]
        bad_in = np.zeros(idx.size, bool)
        bad_out = np.zeros(idx.size, bool)
        bad_in[1:] = same & (sp > cfg.max_ball_speed_mps)
        bad_out[:-1] = same & (sp > cfg.max_ball_speed_mps)
        spike = bad_in & bad_out
        spikes = int(spike.sum())
        fresh[idx[spike]] = False

    x = np.where(fresh, bx, np.nan)
    y = np.where(fresh, by, np.nan)
    z = np.where(fresh, bz, np.nan)

    # velocity fits on [i-W, i] and [i, i+W]; time relative to the period start keeps the sums well conditioned
    W = cfg.velocity_half_window
    t_rel = _period_relative(pos.t, pos.period)
    w = fresh.astype(np.float64)
    xs, ys = np.nan_to_num(x), np.nan_to_num(y)
    vb = np.stack([_fit_slope(t_rel, xs, w, -W, 0, cfg.min_fit_samples), _fit_slope(t_rel, ys, w, -W, 0, cfg.min_fit_samples)], 1)
    va = np.stack([_fit_slope(t_rel, xs, w, 0, W, cfg.min_fit_samples), _fit_slope(t_rel, ys, w, 0, W, cfg.min_fit_samples)], 1)
    # windows may not straddle a period boundary
    per = pos.period.astype(np.int64)
    n = per.size
    i = np.arange(n)
    vb[per[np.clip(i - W, 0, n - 1)] != per] = np.nan
    va[per[np.clip(i + W, 0, n - 1)] != per] = np.nan
    stats = dict(frames=int(n), ball_present=int(present.sum()), repeated_samples=int(repeat.sum()), spikes_removed=spikes,
                 fresh_samples=int(fresh.sum()))
    return BallTrack(x, y, z, fresh, present, vb, va, stats)


def _period_relative(t: np.ndarray, period: np.ndarray) -> np.ndarray:
    out = t.astype(np.float64).copy()
    for p in np.unique(period):
        m = period == p
        out[m] -= out[m][0]
    return out


def player_distances(pos: MatchPositions, i: int, x: float, y: float) -> np.ndarray:
    """Distance of every player column to (x, y) in frame ``i`` (NaN when the player is not observed)."""
    return np.hypot(pos.px[i].astype(np.float64) - x, pos.py[i].astype(np.float64) - y)
