"""M1 baselines C0 (naive) and C1 (Vidal-Codina et al. 2022 as implemented in databallpy). Tag-free, numpy only.

Inputs (per frame, same length n): time t (s), period, our smoothed raw ball (x, y, z, vx, vy and its flag, see
``smoother``), raw player positions px, py (n, P) with NaN for absent players, and per column the player id and team id.
Output per frame: state (0 DEAD, 1 CONTROL, 2 TRAVELLING, 3 LOOSE), player id and team id (-1 when none), and a rule name.

C0 – naive (plan, Approach section): DEAD = ball missing; CONTROL = nearest player within 1.5 m; TRAVELLING = ball speed
> 5 m/s and nobody within 1.5 m; LOOSE = the rest.

C1 – paper. Possession part follows databallpy ``features/player_possession.py`` (read 2026-11-06):
  1. candidate per frame = nearest player with distance < possession-zone radius, else none (missing ball: none);
  2. segments = maximal runs of the same candidate;
  3. a segment is a valid gain if it lasts >= ``min_frames`` and (the angle between the ball's incoming vector
     ball(start) - ball(start-1) and outgoing vector ball(end+1) - ball(end) exceeds the angle threshold, OR the
     frame-to-frame change of ball speed exceeds the speed threshold somewhere in the segment);
  4. a valid gain starts a possession; later segments of the same player extend it; it is lost at the end of that
     player's last segment before another player's valid gain.
Values (Vidal-Codina et al. 2022, given in the plan): zone 1.0 m, duel zone 1.0 m, 10 deg, 5 m/s, spatial threshold 0.1 m.
Choices of ours (databallpy has no defaults; the paper needs a ball-alive flag that PFF does not supply), marked [OURS]:
  - min_frames = 1 [OURS];
  - spatial threshold: a direction vector is taken over as many frames (<= 15) as needed for the ball to move >= 0.1 m [OURS
    reading of the paper's spatial threshold];
  - duel [VC]: an opponent of the nearest player is also within the duel zone -> LOOSE (overrides CONTROL);
  - frames between a possession loss and the next gain -> TRAVELLING with the last holder's team, or LOOSE after a duel [OURS];
  - dead ball: the ball centre beyond a touch- or goal line by more than the ball radius (0.11 m [LAW]) -> DEAD from that
    frame until the restart = the first valid gain that fits the T2 pattern of the line crossed (throw-in: taker within 0.5 m
    of the touchline or beyond; corner: ball within 3 m of a corner; goal kick: ball inside the goal area + 0.5 m;
    kick-off after a ball between the posts: ball within 2 m of the centre spot), else the first valid gain (default: a
    restart matching no pattern is a free kick [VC]); also DEAD before the first gain of each period [OURS, A5];
  - missing ball: the previous state is carried forward [OURS]; there is no whistle detection (the paper used the provider's
    ball-alive flag).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .smoother import FLAG_MISSING

DEAD, CONTROL, TRAVELLING, LOOSE = 0, 1, 2, 3
BALL_RADIUS_M = 0.11
GOAL_HALF_WIDTH_M = 3.66
GOAL_HEIGHT_M = 2.44
GOAL_AREA_DEPTH_M, GOAL_AREA_HALF_WIDTH_M = 5.5, 9.16


@dataclass
class BaselineResult:
    state: np.ndarray
    player: np.ndarray
    team: np.ndarray
    rule: np.ndarray
    rule_names: tuple
    passes: list | None = None        # (release frame, passer id, next holder id) where defined


@dataclass(frozen=True)
class C1Config:
    zone_m: float = 1.0
    duel_m: float = 1.0
    angle_deg: float = 10.0
    speed_change_mps: float = 5.0
    spatial_m: float = 0.1
    min_frames: int = 1
    max_vector_frames: int = 15
    restart_search_s: float = 120.0


def _nearest(bx, by, px, py):
    d = np.hypot(px - bx[:, None], py - by[:, None])
    d = np.where(np.isfinite(d), d, np.inf)
    j = np.argmin(d, axis=1) if d.shape[1] else np.zeros(len(bx), int)
    dmin = d[np.arange(len(bx)), j] if d.shape[1] else np.full(len(bx), np.inf)
    return j, dmin, d


def c0_naive(t, period, ball, px, py, player_id, player_team, radius_m=1.5, speed_mps=5.0) -> BaselineResult:
    present = ball.flag != FLAG_MISSING
    bx, by = np.where(present, ball.x, np.nan), np.where(present, ball.y, np.nan)
    j, dmin, _ = _nearest(bx, by, px, py)
    speed = np.hypot(ball.vx, ball.vy)
    n = len(t)
    state = np.full(n, LOOSE, np.int8)
    rule = np.full(n, 3, np.int8)
    close = present & (dmin <= radius_m)
    fast = present & ~close & (np.nan_to_num(speed) > speed_mps)
    state[close], rule[close] = CONTROL, 1
    state[fast], rule[fast] = TRAVELLING, 2
    state[~present], rule[~present] = DEAD, 0
    player = np.where(close, player_id[j], -1).astype(np.int64)
    team = np.where(close, player_team[j], -1).astype(np.int64)
    return BaselineResult(state, player, team, rule, ("ball_missing", "nearest_within_radius", "fast_nobody_near", "rest"))


def _angle(v1, v2) -> float:
    n1, n2 = np.hypot(*v1), np.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    c = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, c)))))


def _vector(bx, by, i, step, cfg) -> np.ndarray | None:
    """Displacement from frame i over as many frames as needed (<= max_vector_frames) to move >= spatial_m."""
    n = len(bx)
    for k in range(1, cfg.max_vector_frames + 1):
        j = i + step * k
        if not 0 <= j < n or not np.isfinite(bx[j]) or not np.isfinite(bx[i]):
            return None
        v = np.array([bx[j] - bx[i], by[j] - by[i]]) * step
        if np.hypot(*v) >= cfg.spatial_m:
            return v
    return None


def c1_prepare(t, period, ball, px, py, player_id, player_team, pitch_length_m, pitch_width_m, zone_m=1.0, duel_m=1.0,
               spatials=(0.05, 0.1, 0.15, 0.2), max_vector_frames=15) -> dict:
    """Stage 1 of C1 (everything that depends on the zone radius only): candidates, segments and per-segment measures."""
    n = len(t)
    period = np.asarray(period)
    present = ball.flag != FLAG_MISSING
    bx, by, bz = (np.where(present, a, np.nan) for a in (ball.x, ball.y, ball.z))
    hl, hw = pitch_length_m / 2, pitch_width_m / 2
    out = present & ((np.abs(bx) > hl + BALL_RADIUS_M) | (np.abs(by) > hw + BALL_RADIUS_M))
    j, dmin, d = _nearest(bx, by, px, py)
    cand = np.where(present & ~out & (dmin < zone_m), j, -1)
    near_team = player_team[j]
    opp_close = ((player_team[None, :] != near_team[:, None]) & (d <= duel_m)).any(axis=1)
    duel = present & ~out & (dmin <= duel_m) & opp_close
    speed = np.hypot(ball.vx, ball.vy)
    dspeed = np.abs(np.diff(np.nan_to_num(speed), prepend=np.nan_to_num(speed[:1])))
    change = np.flatnonzero((cand[1:] != cand[:-1]) | (period[1:] != period[:-1])) + 1
    starts = np.r_[0, change]
    ends = np.r_[change - 1, n - 1]
    keep = cand[starts] >= 0
    s_, e_ = starts[keep], ends[keep]
    c_ = cand[s_]
    maxds = np.array([dspeed[s:e + 1].max() for s, e in zip(s_, e_)]) if len(s_) else np.zeros(0)
    angles = {}
    for sp in spatials:
        cfg = C1Config(spatial_m=sp, max_vector_frames=max_vector_frames)
        a = np.zeros(len(s_))
        for k, (s, e) in enumerate(zip(s_, e_)):
            vin, vout = _vector(bx, by, s, -1, cfg), _vector(bx, by, e, 1, cfg)
            a[k] = _angle(vin, vout) if vin is not None and vout is not None else 0.0
        angles[sp] = a
    return dict(n=n, t=np.asarray(t), period=period, present=present, bx=bx, by=by, bz=bz, px=px, py=py, out=out, duel=duel,
                seg_s=s_, seg_e=e_, seg_c=c_, seg_maxds=maxds, seg_angle=angles, player_id=player_id, player_team=player_team,
                hl=hl, hw=hw, zone_m=zone_m)


def c1_finish(prep: dict, angle_deg=10.0, speed_change_mps=5.0, spatial_m=0.1, min_frames=1, restart_search_s=120.0,
              alive: np.ndarray | None = None) -> BaselineResult:
    """Stage 2 of C1: valid gains, possessions, states. ``alive`` (bool per frame) replaces C1's own dead-ball logic with
    an external ball-alive flag (the condition of the paper); only for the oracle comparison, never for a real contestant."""
    n, t, period = prep["n"], prep["t"], prep["period"]
    s_, e_, c_ = prep["seg_s"], prep["seg_e"], prep["seg_c"]
    pid, pteam = prep["player_id"], prep["player_team"]
    duel, present = prep["duel"], prep["present"]
    valid = ((e_ - s_ + 1) >= min_frames) & ((prep["seg_maxds"] > speed_change_mps) | (prep["seg_angle"][spatial_m] > angle_deg))
    poss = []
    for s, e, c, v in zip(s_.tolist(), e_.tolist(), c_.tolist(), valid.tolist()):
        if poss and poss[-1][2] == c and period[poss[-1][0]] == period[s]:
            poss[-1] = (poss[-1][0], e, c)
        elif v:
            poss.append((s, e, c))
    names = ("period_start", "out_of_bounds", "possession", "duel", "released", "after_duel", "missing_carry", "no_rule",
             "oracle_dead")
    state = np.full(n, -1, np.int8)
    player = np.full(n, -1, np.int64)
    team = np.full(n, -1, np.int64)
    rule = np.full(n, 7, np.int8)
    passes = []
    for k, (s, e, c) in enumerate(poss):
        state[s:e + 1], rule[s:e + 1], player[s:e + 1], team[s:e + 1] = CONTROL, 2, pid[c], pteam[c]
        same = k + 1 < len(poss) and period[poss[k + 1][0]] == period[s]
        nxt = poss[k + 1][0] if same else None
        if same:
            passes.append((e, int(pid[c]), int(pid[poss[k + 1][2]])))
        g1 = (nxt - 1) if nxt is not None else int(np.flatnonzero(period == period[s])[-1])
        if g1 > e:
            had_duel = bool(duel[max(s, e - 3):e + 4].any())
            state[e + 1:g1 + 1] = LOOSE if had_duel else TRAVELLING
            rule[e + 1:g1 + 1] = 5 if had_duel else 4
            team[e + 1:g1 + 1] = -1 if had_duel else pteam[c]
            player[e + 1:g1 + 1] = -1 if had_duel else pid[c]
    dz = duel & (state == CONTROL)
    state[dz], rule[dz], player[dz], team[dz] = LOOSE, 3, -1, -1
    gains = np.array([p[0] for p in poss], int)
    if alive is None:
        for p in np.unique(period):
            if p <= 0:
                continue
            fr = np.flatnonzero(period == p)
            first = gains[(gains >= fr[0]) & (gains <= fr[-1])]
            end = (first[0] - 1) if len(first) else fr[-1]
            state[fr[0]:end + 1], rule[fr[0]:end + 1], player[fr[0]:end + 1], team[fr[0]:end + 1] = DEAD, 0, -1, -1
        out = prep["out"]
        bx, by, bz, px, py, hl, hw = prep["bx"], prep["by"], prep["bz"], prep["px"], prep["py"], prep["hl"], prep["hw"]
        step = float(np.median(np.diff(t))) if n > 1 else 1.0
        tol_search = int(restart_search_s / max(step, 1e-6))
        starts = np.flatnonzero(out & ~np.r_[False, out[:-1]])
        done_until = -1
        for i in starts.tolist():
            if i < done_until:
                continue
            kind = _line_kind(bx[i], by[i], bz[i], hl, hw)
            later = gains[(gains > i) & (period[np.minimum(gains, n - 1)] == period[i])]
            restart = next((g for g in later.tolist() if g - i <= tol_search and _fits(kind, g, bx, by, px, py, hl, hw)), None)
            if restart is None:
                restart = int(later[0]) if len(later) else int(np.flatnonzero(period == period[i])[-1]) + 1
            state[i:restart], rule[i:restart], player[i:restart], team[i:restart] = DEAD, 1, -1, -1
            done_until = restart
    # missing ball: carry the previous state forward within the period (vectorised forward fill)
    fill = (state < 0) | (~present & ~np.isin(rule, (0, 1)))
    src = np.where(~fill, np.arange(n), -1)
    src = np.maximum.accumulate(src)
    okf = fill & (src >= 0)
    okf[okf] = period[src[okf]] == period[okf]
    state[okf], player[okf], team[okf] = state[src[okf]], player[src[okf]], team[src[okf]]
    rule[okf & ~present] = 6
    bad = fill & ~okf
    state[bad], rule[bad], player[bad], team[bad] = DEAD, 0, -1, -1
    if alive is not None:
        dead = ~np.asarray(alive, bool)
        state[dead], rule[dead], player[dead], team[dead] = DEAD, 8, -1, -1
    return BaselineResult(state, player, team, rule, names, passes)


def c1_paper(t, period, ball, px, py, player_id, player_team, pitch_length_m, pitch_width_m,
             cfg: C1Config | None = None, alive=None) -> BaselineResult:
    """C1 with one parameter set (convenience wrapper around c1_prepare + c1_finish)."""
    cfg = cfg or C1Config()
    prep = c1_prepare(t, period, ball, px, py, player_id, player_team, pitch_length_m, pitch_width_m, zone_m=cfg.zone_m,
                      duel_m=cfg.duel_m, spatials=(cfg.spatial_m,), max_vector_frames=cfg.max_vector_frames)
    return c1_finish(prep, cfg.angle_deg, cfg.speed_change_mps, cfg.spatial_m, cfg.min_frames, cfg.restart_search_s, alive=alive)


def _line_kind(x, y, z, hl, hw) -> str:
    if abs(x) > hl + BALL_RADIUS_M:
        if abs(y) <= GOAL_HALF_WIDTH_M and (not np.isfinite(z) or z <= GOAL_HEIGHT_M):
            return "goal"
        return "goal_line"
    return "touchline"


def _fits(kind, g, bx, by, px, py, hl, hw) -> bool:
    """T2 restart patterns (tolerances from the plan, values from [VC] S4.2 and [LAW])."""
    x, y = bx[g], by[g]
    if not np.isfinite(x):
        return False
    if kind == "touchline":                                   # throw-in: a player at/behind the touchline (0.5 m)
        return bool(np.nanmax(np.abs(py[g])) >= hw - 0.5) and abs(y) >= hw - 3.0
    if kind == "goal_line":                                   # corner (3 m from a corner mark) or goal kick (goal area + 0.5 m)
        corner = min(np.hypot(abs(x) - hl, abs(y) - hw), 99) <= 3.0
        goal_kick = abs(x) >= hl - GOAL_AREA_DEPTH_M - 0.5 and abs(y) <= GOAL_AREA_HALF_WIDTH_M + 0.5
        return bool(corner or goal_kick)
    return bool(np.hypot(x, y) <= 2.0)                        # kick-off after a goal: ball at the centre spot
