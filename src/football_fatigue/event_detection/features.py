"""Per-touch feature table for the ML classifier (positions and time only; NaN = not measurable).

Coordinates are turned into the actor's attacking frame (attacked goal at +52.5 m, a 180-degree rotation for the team
attacking -x), so the model sees the same geometry for both teams and both halves.
"""

from __future__ import annotations

import numpy as np

from .classify import _ball_gap_s, _flight, _goal_test, _is_carry, _out_of_play
from .config import DetectorConfig
from .inputs import MatchPositions
from .kinematics import BallTrack
from .touches import Touch

FEATURES = (
    "x_att", "y_abs", "goal_dist", "goal_angle", "dv", "speed_before", "speed_after", "player_dist",
    "z_touch", "z_max_flight", "dir_speed", "dir_x_att", "dir_y_toward_centre", "dir_goal_cos", "proj_goal_y_abs",
    "dt_next", "next_same_side", "next_same_player", "carry", "next_x_att", "next_y_abs", "travel", "gap_after",
    "out_of_play", "end_x_att", "end_y_abs", "mouth_dist", "pause_after", "dt_prev", "prev_same_side", "prev_same_player",
    "opp_within_3m", "opp_within_6m", "nearest_opp", "nearest_mate", "ball_present_1s", "other_side_dist", "actor_is_nearest",
)


def _nan(v) -> float:
    return float("nan") if v is None else float(v)


def actor_rows(touches: list[Touch]) -> list[tuple[int, Touch]]:
    """One row per (touch, candidate actor): the nearest player, and the nearest player of the other side when in range."""
    rows = []
    for k, t in enumerate(touches):
        rows.append((k, t))
        alt = t.as_alternative()
        if alt is not None:
            rows.append((k, alt))
    return rows


def touch_features(pos: MatchPositions, ball: BallTrack, touches: list[Touch], cfg: DetectorConfig,
                   rows: list[tuple[int, Touch]] | None = None) -> np.ndarray:
    """Feature matrix, one row per (touch index, actor); ``rows`` defaults to the touches as detected."""
    rows = list(enumerate(touches)) if rows is None else rows
    n = len(touches)
    X = np.full((len(rows), len(FEATURES)), np.nan)
    half = cfg.pitch_length_m / 2
    for r, (k, t) in enumerate(rows):
        f = dict.fromkeys(FEATURES, np.nan)
        f["other_side_dist"] = _nan(t.alt_distance_m)
        f["actor_is_nearest"] = float(t.alt_distance_m is None or t.distance_m <= t.alt_distance_m)
        right = pos.attacks_right(t.side, t.period)
        sgn = np.nan if right is None else (1.0 if right else -1.0)
        nxt = touches[k + 1] if k + 1 < n and touches[k + 1].period == t.period else None
        prv = touches[k - 1] if k > 0 and touches[k - 1].period == t.period else None
        x_att = sgn * t.x
        f.update(x_att=x_att, y_abs=abs(t.y), dv=t.dv_mps, speed_after=t.speed_after_mps, player_dist=t.distance_m,
                 speed_before=float(np.hypot(*ball.v_before[t.frame])), z_touch=float(ball.z[t.frame]) if ball.fresh[t.frame] else np.nan)
        gx, gy = half - x_att, t.y
        f["goal_dist"] = float(np.hypot(gx, gy))
        a1, a2 = np.arctan2(3.66 - gy, gx), np.arctan2(-3.66 - gy, gx)
        f["goal_angle"] = float(abs(a1 - a2))
        stop = min(nxt.frame if nxt is not None else pos.n_frames - 1, t.frame + int(cfg.max_pass_flight_s * pos.fps))
        fl = _flight(pos, ball, t, stop, cfg)
        if not np.isnan(fl.direction).any() and not np.isnan(sgn):
            vx, vy = sgn * fl.direction[0], sgn * fl.direction[1]
            sp = float(np.hypot(vx, vy))
            f.update(dir_speed=sp, dir_x_att=vx / sp if sp > 0 else 0.0)
            yy = sgn * t.y
            f["dir_y_toward_centre"] = (-np.sign(yy) * vy / sp) if sp > 0 else 0.0
            if sp > 0:
                f["dir_goal_cos"] = float((vx * gx + vy * (0 - yy)) / (sp * max(f["goal_dist"], 1e-6)))
            if vx > 0:
                f["proj_goal_y_abs"] = float(min(abs(yy + vy * (half - x_att) / vx), 60.0))
        if fl.idx.size:
            zz = ball.z[fl.idx]
            f["z_max_flight"] = float(np.nanmax(zz)) if not np.isnan(zz).all() else np.nan
            ex, ey = ball.x[fl.idx[-1]], ball.y[fl.idx[-1]]
            f.update(end_x_att=sgn * ex, end_y_abs=abs(ey))
        f["gap_after"] = _ball_gap_s(pos, ball, t.frame, stop)
        f["out_of_play"] = float(_out_of_play(ball, fl, cfg))
        if nxt is not None:
            f.update(dt_next=nxt.t_s - t.t_s, next_same_side=float(nxt.side == t.side), next_same_player=float(nxt.player == t.player),
                     carry=float(nxt.player == t.player and _is_carry(ball, t, nxt, cfg)), next_x_att=sgn * nxt.x, next_y_abs=abs(nxt.y),
                     travel=float(np.hypot(nxt.x - t.x, nxt.y - t.y)))
        if prv is not None:
            f.update(dt_prev=t.t_s - prv.t_s, prev_same_side=float(prv.side == t.side), prev_same_player=float(prv.player == t.player))
        if not np.isnan(sgn):
            _, g = _goal_test(pos, ball, t, half * sgn, touches[k + 1:k + 60], cfg)
            f.update(mouth_dist=_nan(g["ball_goal_mouth_distance_m"]), pause_after=_nan(g["stoppage_after_shot_s"]))
        # pressure around the actor at the touch
        ax, ay = pos.px[t.frame, t.player], pos.py[t.frame, t.player]
        if not np.isnan(ax):
            d = np.hypot(pos.px[t.frame] - ax, pos.py[t.frame] - ay)
            opp = d[pos.player_side != t.side]
            mate = d[(pos.player_side == t.side) & (np.arange(d.size) != t.player)]
            opp, mate = opp[~np.isnan(opp)], mate[~np.isnan(mate)]
            f.update(opp_within_3m=float((opp <= 3).sum()), opp_within_6m=float((opp <= 6).sum()),
                     nearest_opp=float(opp.min()) if opp.size else np.nan, nearest_mate=float(mate.min()) if mate.size else np.nan)
        w = slice(max(t.frame - int(pos.fps), 0), t.frame + int(pos.fps) + 1)
        f["ball_present_1s"] = float(ball.present[w].mean())
        X[r] = [f[name] for name in FEATURES]
    return X
