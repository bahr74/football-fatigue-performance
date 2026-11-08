"""Rule-based classification of touches into passes, crosses, shots, goals, and key passes.

A *release* is a touch after which the ball next reaches a different player (or the same player only after leaving him),
or after which the play ends. Every detected event carries the feature values that decided it (``reason``) so each
decision can be checked by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import DetectorConfig
from .inputs import MatchPositions
from .kinematics import BallTrack
from .touches import Touch

PASS, CROSS, SHOT = "PA", "CR", "SH"


@dataclass
class DetectedEvent:
    type: str                         # PA | CR | SH
    touch: Touch
    team_side: int
    outcome: str | None               # PA/CR: C complete, D defended (opponent next), O out of play; SH: G goal or None
    receiver: int | None = None       # player column
    end_touch: Touch | None = None
    key_pass: bool = False
    confidence: float = 0.5
    reason: dict[str, Any] = field(default_factory=dict)

    @property
    def is_goal(self) -> bool:
        return self.type == SHOT and self.outcome == "G"


@dataclass
class _Flight:
    """The ball between a release and the next touch (or the end of the search window)."""
    start: int
    stop: int
    idx: np.ndarray           # rows with fresh ball samples in [start, stop]
    direction: np.ndarray     # (vx, vy) from displacement over the first ``direction_horizon_s``; NaN when unknown

    def end_xy(self, ball: BallTrack) -> tuple[float, float] | None:
        return (float(ball.x[self.idx[-1]]), float(ball.y[self.idx[-1]])) if self.idx.size else None


def _attacked_goal_x(pos: MatchPositions, side: int, period: int, cfg: DetectorConfig) -> float | None:
    right = pos.attacks_right(side, period)
    if right is None:
        return None
    return cfg.pitch_length_m / 2 if right else -cfg.pitch_length_m / 2


def _flight(pos: MatchPositions, ball: BallTrack, t: Touch, stop: int, cfg: DetectorConfig) -> _Flight:
    idx = t.frame + np.flatnonzero(ball.fresh[t.frame:stop + 1])
    idx = idx[pos.period[idx] == t.period]
    direction = np.array([np.nan, np.nan])
    if idx.size >= 2:
        horizon = pos.t[idx] - pos.t[idx[0]] <= cfg.direction_horizon_s
        j = idx[horizon][-1] if horizon.sum() >= 2 else idx[1]
        dt = pos.t[j] - pos.t[idx[0]]
        if dt > 0:
            direction = np.array([ball.x[j] - ball.x[idx[0]], ball.y[j] - ball.y[idx[0]]]) / dt
    return _Flight(t.frame, stop, idx, direction)


def _ball_gap_s(pos: MatchPositions, ball: BallTrack, a: int, b: int) -> float:
    """Longest run of frames without a ball between rows a and b (seconds)."""
    seg = ~ball.present[a:b + 1]
    if not seg.any():
        return 0.0
    edges = np.flatnonzero(np.diff(np.concatenate([[0], seg.astype(np.int8), [0]])))
    return float((edges[1::2] - edges[::2]).max()) / pos.fps


def _shot_test(pos: MatchPositions, ball: BallTrack, t: Touch, fl: _Flight, goal_x: float, cfg: DetectorConfig) -> tuple[bool, dict]:
    dist = float(np.hypot(goal_x - t.x, t.y))
    if dist > cfg.shot_max_distance_m:
        return False, {}
    mouth = cfg.goal_half_width_m + cfg.shot_post_margin_m
    sign = np.sign(goal_x)
    # (a) direction of the struck ball crosses the goal line near the posts
    projected_ok, y_cross, speed = False, None, float("nan")
    for v in (fl.direction, ball.v_after[t.frame]):
        if np.isnan(v).any() or sign * v[0] <= 0:
            continue
        s = float(np.hypot(*v))
        y = float(t.y + v[1] * (goal_x - t.x) / v[0])
        if s >= cfg.shot_min_speed_mps and abs(y) <= mouth:
            projected_ok, y_cross, speed = True, y, s
            break
    # (b) the ball is seen at the goal mouth during the flight
    reached = False
    if fl.idx.size:
        bx, by = ball.x[fl.idx], ball.y[fl.idx]
        reached = bool(np.any((sign * bx >= abs(goal_x) - cfg.shot_reach_goal_m) & (np.abs(by) <= mouth)))
    ok = projected_ok or (reached and t.speed_after_mps >= cfg.min_release_speed_mps)
    r = dict(goal_distance_m=round(dist, 1), ball_reaches_goal_mouth=reached)
    if projected_ok:
        r.update(shot_speed_mps=round(speed, 2), projected_goal_line_y_m=round(y_cross, 2))
    return ok, r


def _goal_test(pos: MatchPositions, ball: BallTrack, t: Touch, goal_x: float, later: list[Touch], cfg: DetectorConfig) -> tuple[bool, dict]:
    """A goal needs two independent signs: the ball reaches the goal mouth, and play stops for a kickoff-length pause.

    A pause alone (injury, VAR, time-wasting, a lost ball) or a ball near the goal alone (a save, a shot just wide) is not
    enough. Very long pauses (over ``goal_max_stoppage_s``) are treated as other stoppages.
    """
    end = min(t.frame + int(cfg.goal_line_window_s * pos.fps), pos.n_frames - 1)
    s = slice(t.frame, end + 1)
    ok = (pos.period[s] == t.period) & ball.fresh[s]
    x, y, z = ball.x[s][ok], ball.y[s][ok], ball.z[s][ok]
    with np.errstate(invalid="ignore"):
        crossed = bool(np.any((np.sign(goal_x) * x >= abs(goal_x)) & (np.abs(y) <= cfg.goal_half_width_m + 0.5)
                              & (np.isnan(z) | (z <= cfg.goal_height_m + 0.3))))
    # closest approach of the ball to the goal mouth (0 inside the frame of the goal, measured in the pitch plane)
    mouth = float(np.min(np.hypot(abs(goal_x) - np.sign(goal_x) * x, np.clip(np.abs(y) - cfg.goal_half_width_m, 0, None)))) if x.size else None
    at_mouth = crossed or (mouth is not None and mouth <= cfg.goal_mouth_max_m)
    # longest pause between touches that starts within ``goal_stoppage_start_s`` of the shot (a stray touch on the rebound or
    # during the celebration must not hide the restart); None when the period ends first (no restart to observe)
    times = [t.t_s] + [x.t_s for x in later if x.period == t.period]
    stoppage, restart = None, None
    for a, b in zip(times, times[1:]):
        if a > t.t_s + cfg.goal_stoppage_start_s:
            break
        if stoppage is None or b - a > stoppage:
            stoppage, restart = b - a, b
    stopped = stoppage is not None and cfg.goal_min_stoppage_s <= stoppage <= cfg.goal_max_stoppage_s
    return at_mouth and stopped, dict(ball_crossed_goal_line=crossed, ball_goal_mouth_distance_m=None if mouth is None else round(mouth, 2),
                                      stoppage_after_shot_s=None if stoppage is None else round(stoppage, 1),
                                      restart_at_s=round(restart, 2) if stopped else None)


def _cross_test(t: Touch, fl: _Flight, end: tuple[float, float] | None, goal_x: float, cfg: DetectorConfig) -> tuple[bool, dict]:
    from_line = abs(goal_x - t.x)
    wide = abs(t.y) >= cfg.cross_min_abs_y_m
    v = fl.direction
    speed = float(np.hypot(*v)) if not np.isnan(v).any() else t.speed_after_mps
    inward = (not np.isnan(v).any()) and v[1] * np.sign(t.y) < 0
    lands_in_box = end is not None and abs(end[1]) <= 22.0 and abs(goal_x - end[0]) <= 20.0
    ok = wide and from_line <= cfg.cross_max_goal_distance_m and speed >= cfg.cross_min_speed_mps and bool(inward) and lands_in_box
    return ok, dict(start_abs_y_m=round(abs(t.y), 1), distance_from_goal_line_m=round(from_line, 1), cross_speed_mps=round(speed, 2),
                    ends_in_box_area=bool(lands_in_box))


def _out_of_play(ball: BallTrack, fl: _Flight, cfg: DetectorConfig) -> bool:
    if fl.idx.size == 0:
        return False
    last = fl.idx[-1]
    return abs(ball.x[last]) > cfg.pitch_length_m / 2 + 0.5 or abs(ball.y[last]) > cfg.pitch_width_m / 2 + 0.5


def _is_carry(ball: BallTrack, t: Touch, nxt: Touch, cfg: DetectorConfig) -> bool:
    seg = slice(t.frame, nxt.frame + 1)
    f = ball.fresh[seg]
    if not f.any():
        return True
    d = np.hypot(ball.x[seg][f] - t.x, ball.y[seg][f] - t.y)
    return bool(d.max() <= cfg.carry_radius_m + np.hypot(nxt.x - t.x, nxt.y - t.y))


def classify(pos: MatchPositions, ball: BallTrack, touches: list[Touch], cfg: DetectorConfig) -> list[DetectedEvent]:
    events: list[DetectedEvent] = []
    n = len(touches)
    for k, t in enumerate(touches):
        nxt = touches[k + 1] if k + 1 < n else None
        same_period = nxt is not None and nxt.period == t.period
        if same_period and nxt.player == t.player and _is_carry(ball, t, nxt, cfg):
            continue                                               # the same player keeps the ball: a carry, not a release
        if t.speed_after_mps < cfg.min_release_speed_mps:
            continue
        connected = (same_period and nxt.t_s - t.t_s <= cfg.max_pass_flight_s
                     and _ball_gap_s(pos, ball, t.frame, nxt.frame) <= cfg.dead_ball_gap_s)
        stop = min(nxt.frame if same_period else pos.n_frames - 1, t.frame + int(cfg.max_pass_flight_s * pos.fps))
        fl = _flight(pos, ball, t, stop, cfg)
        goal_x = _attacked_goal_x(pos, t.side, t.period, cfg)
        reason: dict[str, Any] = dict(touch_dv_mps=round(t.dv_mps, 2), player_ball_distance_m=round(t.distance_m, 2),
                                      release_speed_mps=round(t.speed_after_mps, 2))
        if goal_x is not None:
            is_shot, r = _shot_test(pos, ball, t, fl, goal_x, cfg)
            if is_shot:
                reason.update(r)
                scored, r2 = _goal_test(pos, ball, t, goal_x, touches[k + 1:k + 60], cfg)
                reason.update(r2)
                events.append(DetectedEvent(SHOT, t, t.side, "G" if scored else None, confidence=0.6, reason=reason))
                continue
        if connected:
            outcome = "C" if nxt.side == t.side else "D"
            end_xy = (nxt.x, nxt.y)
        else:
            outcome = "O" if _out_of_play(ball, fl, cfg) else None
            end_xy = fl.end_xy(ball)
            if outcome is None:
                continue                                          # the ball is lost (missing) without a receiver: no claim
        typ = PASS
        if goal_x is not None:
            is_cross, r = _cross_test(t, fl, end_xy, goal_x, cfg)
            if is_cross:
                typ = CROSS
                reason.update(r)
        reason["next_touch"] = "teammate" if outcome == "C" else "opponent" if outcome == "D" else "out_of_play"
        events.append(DetectedEvent(typ, t, t.side, outcome, receiver=nxt.player if outcome == "C" else None,
                                    end_touch=nxt if connected else None, confidence=0.6 if outcome == "C" else 0.5, reason=reason))
    _one_goal_per_restart(events)
    _mark_key_passes(events, touches, cfg)
    return events


def _one_goal_per_restart(events: list[DetectedEvent]) -> None:
    """Several shots can precede the same long pause; only the last of them can have scored."""
    last_by_restart: dict[float, DetectedEvent] = {}
    for e in events:
        r = e.reason.get("restart_at_s")
        if e.is_goal and r is not None:
            prev = last_by_restart.get(r)
            if prev is not None:
                prev.outcome = None
                prev.reason["goal_withdrawn"] = "a later shot precedes the same restart"
            last_by_restart[r] = e


def _mark_key_passes(events: list[DetectedEvent], touches: list[Touch], cfg: DetectorConfig) -> None:
    """Key pass: the last completed pass or cross of the shooting team before a shot, with no opponent touch in between.

    ``touches`` are the opponent actions that break the chain (the ML detector passes its events' touches, because its
    loose candidate touches include many contacts that are not actions)."""
    for i, sh in enumerate(events):
        if sh.type != SHOT:
            continue
        for ev in reversed(events[:i]):
            if sh.touch.t_s - ev.touch.t_s > cfg.key_pass_window_s or ev.touch.period != sh.touch.period:
                break
            if ev.team_side != sh.team_side:
                break
            if ev.type in (PASS, CROSS) and ev.outcome == "C":
                between = [t for t in touches if ev.touch.t_s < t.t_s < sh.touch.t_s and t.side != sh.team_side]
                if not between:
                    ev.key_pass = True
                    ev.reason["key_pass_for_shot_at_s"] = round(sh.touch.t_s, 2)
                break
