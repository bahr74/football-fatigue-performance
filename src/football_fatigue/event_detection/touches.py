"""Touch detection: frames where the ball's velocity changes sharply while a player is next to it."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace

import numpy as np

from .config import DetectorConfig
from .inputs import MatchPositions
from .kinematics import BallTrack


@dataclass(frozen=True)
class Touch:
    frame: int               # row index in MatchPositions
    t_s: float
    period: int
    player: int              # player column
    side: int
    distance_m: float        # closest player-ball distance found around the touch
    dv_mps: float
    speed_after_mps: float
    x: float                 # ball position at the touch (nearest fresh sample)
    y: float
    alt_player: int | None = None      # nearest player of the other side, when also within the touch radius
    alt_distance_m: float | None = None

    def as_alternative(self) -> "Touch | None":
        """The same touch attributed to the nearest player of the other side (None when there is none in range)."""
        if self.alt_player is None:
            return None
        return replace(self, player=self.alt_player, side=1 - self.side, distance_m=self.alt_distance_m, alt_player=self.player,
                       alt_distance_m=self.distance_m)


def _sliding_max(a: np.ndarray, half: int) -> np.ndarray:
    pad = np.full(half, -np.inf)
    v = np.lib.stride_tricks.sliding_window_view(np.concatenate([pad, a, pad]), 2 * half + 1)
    return v.max(axis=1)


def _nearest_fresh(ball: BallTrack, i: int, reach: int) -> int | None:
    lo, hi = max(i - reach, 0), min(i + reach + 1, ball.fresh.size)
    cand = np.flatnonzero(ball.fresh[lo:hi])
    if cand.size == 0:
        return None
    return int(lo + cand[np.argmin(np.abs(lo + cand - i))])


def detect_touches(pos: MatchPositions, ball: BallTrack, cfg: DetectorConfig) -> list[Touch]:
    dv = np.nan_to_num(ball.dv, nan=-np.inf)
    peaks = np.flatnonzero((dv >= cfg.touch_dv_mps) & (dv == _sliding_max(dv, cfg.touch_peak_window)))
    speed_after = ball.speed_after
    touches: list[Touch] = []
    last = -10**9
    for i in peaks:
        if i - last <= cfg.touch_peak_window:      # plateau of equal maxima: keep the first frame
            continue
        last = i
        j = _nearest_fresh(ball, int(i), cfg.velocity_half_window)
        if j is None:
            continue
        bx, by = ball.x[j], ball.y[j]
        lo, hi = max(i - cfg.touch_search_frames, 0), min(i + cfg.touch_search_frames + 1, pos.n_frames)
        lo_p = lo if pos.period[lo] == pos.period[i] else i
        d = np.hypot(pos.px[lo_p:hi].astype(np.float64) - bx, pos.py[lo_p:hi].astype(np.float64) - by)
        if np.all(np.isnan(d)):
            continue
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)          # players never seen in the window: all-NaN columns
            dmin = np.nanmin(d, axis=0)
        k = int(np.nanargmin(dmin))
        if not dmin[k] <= cfg.touch_radius_m:
            continue
        other = np.where(pos.player_side != pos.player_side[k], dmin, np.nan)
        alt = int(np.nanargmin(other)) if not np.all(np.isnan(other)) else None
        if alt is not None and not other[alt] <= cfg.touch_radius_m:
            alt = None
        touches.append(Touch(frame=int(i), t_s=float(pos.t[i]), period=int(pos.period[i]), player=k, side=int(pos.player_side[k]),
                             distance_m=float(dmin[k]), dv_mps=float(ball.dv[i]), speed_after_mps=float(np.nan_to_num(speed_after[i])),
                             x=float(bx), y=float(by), alt_player=alt, alt_distance_m=None if alt is None else float(other[alt])))
    return touches
