"""M1 contestant C2: full rule set (plan, "C2 full rules" A1-A7, B1-B4, C1-C10, D1-D5 and table T2). Tag-free, numpy only.

C2 v2 (``C2Params.v2``): alive / dead decided first (layer 1), the C rules only on alive frames (layer 2); A3 slowdowns need a
T2 restart within ``restart_limit_s`` and are dated back from it; the controller must be ``clear_margin_m`` closer than the
nearest opponent; a new owner is dated back to when control began [CH]; missing-ball gaps keep the team that had the ball on
both sides, and a long gap with the ball back at the same spot is a free kick [KI]. [CH] coderhema, [KI] KoichiInoue7
(Barca hackathon solutions).

Inputs per frame: time, period, our smoothed raw ball (x, y, z, vx, vy, flag) and raw player positions (px, py with NaN when
absent) with player id and team id per column, pitch size. Output per frame: state (0 DEAD, 1 CONTROL, 2 TRAVELLING, 3 LOOSE,
4 UNKNOWN), player, team, confidence (0-1) and the rule that decided it; plus the detected touches and passes.

Thresholds of table T1 live in :class:`C2Params` (start values = the table). Values the plan does not give are marked [GUESS]
in the field comments. How each rule is realised, and where it had to be simplified, is written next to its code.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .smoother import FLAG_MISSING

DEAD, CONTROL, TRAVELLING, LOOSE, UNKNOWN = 0, 1, 2, 3, 4
BALL_R = 0.11
POST_Y, BAR_Z = 3.66, 2.44
BOX_DEPTH, BOX_HALF_W = 16.5, 20.16
GOAL_AREA_DEPTH, GOAL_AREA_HALF_W = 5.5, 9.16
PEN_SPOT = 11.0


@dataclass
class C2Params:
    # --- T1, control / duel group
    control_radius: float = 1.5
    duel_radius: float = 1.0
    grace_s: float = 0.5
    reach_s: float = 1.0
    release_m: float = 5.0
    arrival_m: float = 5.0
    min_failed_pass_m: float = 2.5
    shot_post_m: float = 5.0
    # --- T1, touch group
    touch_foot: float = 2.5
    touch_reception: float = 2.0
    touch_tackle: float = 3.0
    h_foot: float = 1.5
    h_header: float = 1.0
    h_max: float = 3.5
    kick_acc: float = 10.0
    dir_change: float = 12.5
    speed_ratio: float = 1.5
    speed_abs: float = 5.0
    # --- T1, dead-ball group (+ T2 tolerances)
    out_margin: float = 0.3
    out_time_s: float = 0.3
    stop_context_s: float = 4.0
    stop_smooth_s: float = 3.6
    min_stop_s: float = 2.0
    kickoff_tol: float = 4.0            # T2 3-5 m
    throwin_tol: float = 0.5            # T2
    corner_m: float = 3.0               # T2
    goal_area_tol: float = 0.5          # T2
    fk_wall_tol: float = 4.0            # T2 (penalty tolerance, used for the 9.15 m free-kick distance too)
    # --- [GUESS] values the plan does not give
    match_tol_s: float = 0.3            # T1 fixed: +-0.3 s around a peak for the player-ball minimum
    peak_frames: int = 3                # half-width of the velocity windows and of the local-peak window
    walk_speed: float = 1.5             # [GUESS] mean player speed below this = "most players walking" (tuned, 1.0-2.0)
    press_m: float = 4.6                # [PFF] 5 yd pressure distance: an opponent this close = contested
    still_speed: float = 0.5            # [GUESS] ball speed of a placed ball
    still_s: float = 1.0                # [GUESS] how long a placed ball rests before the kick
    short_gap_s: float = 1.0            # [GUESS] A6: keep the previous state this long when the ball is missing
    vanish_line_m: float = 1.0          # [GUESS] A1: ball last seen this close to a line before it vanishes
    control_max_speed: float = 8.0      # [GUESS] C1: ball faster than this is not under control
    run_speed: float = 3.0              # [GUESS] C1 reach-ahead: player speed used to reach the predicted ball
    tie_m: float = 0.5                  # [GUESS] B3: two players within this of each other's distance = a tie
    clearance_len_m: float = 30.0       # [GUESS] C3: a clearance travels at least this far ...
    clearance_z_m: float = 2.5          # [GUESS] ... and this high, from the own third
    deflect_angle: float = 20.0         # [GUESS] C5: a deflection turns the ball less than this ...
    deflect_ratio: float = 1.25         # [GUESS] ... and keeps its speed within this ratio
    min_state_s: float = 0.1            # [GUESS] D3: shorter TRAVELLING / LOOSE runs are absorbed by the state before
    # --- C2 v2 (off by default, so C2-tuned stays reproducible)
    v2: bool = False
    restart_limit_s: float = 60.0       # v2 A3: a slowdown is a stoppage only if a T2 restart follows within this time [user]
    clear_margin_m: float = 1.0         # v2 C1: the controller is this much closer to the ball than the nearest opponent [CH]
    gap_long_s: float = 4.0             # [GUESS] v2 A6: a missing-ball gap at least this long ...
    still_speed_v2: float = 2.0         # [GUESS] v2: median ball speed of a placed ball in the second before the kick
    v2_skip: str = ""                  # diagnostics: v2 fixes to leave out (a whistle, b layers, c margin, d back-dating, e gaps)
    gap_same_spot_m: float = 3.0        # [GUESS] ... with the ball back within this of where it vanished = free kick, DEAD [KI]

    def as_dict(self) -> dict:
        return asdict(self)


GROUPS = {
    "dead": ("out_margin", "out_time_s", "stop_context_s", "stop_smooth_s", "min_stop_s", "walk_speed", "kickoff_tol",
             "throwin_tol", "corner_m", "goal_area_tol", "fk_wall_tol"),
    "touch": ("touch_foot", "touch_reception", "touch_tackle", "h_foot", "h_header", "h_max", "kick_acc", "dir_change",
              "speed_ratio", "speed_abs"),
    "control": ("control_radius", "duel_radius", "grace_s", "reach_s", "release_m", "arrival_m", "min_failed_pass_m",
                "shot_post_m"),
    "v2": ("restart_limit_s", "clear_margin_m", "gap_long_s", "gap_same_spot_m"),
}
RANGES = {  # T1 search ranges (T2 tolerances: kick-off 3-5 m from the table; the others +-50 % around the table value)
    "control_radius": (0.5, 3.0), "touch_foot": (1.0, 3.5), "touch_reception": (1.0, 3.0), "touch_tackle": (1.5, 4.0),
    "duel_radius": (0.5, 2.0), "h_foot": (1.0, 2.0), "h_header": (0.8, 1.5), "h_max": (2.5, 4.0), "kick_acc": (4.0, 20.0),
    "dir_change": (5.0, 25.0), "speed_ratio": (1.1, 1.5), "speed_abs": (1.0, 5.0), "grace_s": (0.2, 2.0), "reach_s": (0.5, 1.5),
    "release_m": (2.0, 8.0), "arrival_m": (2.0, 8.0), "min_failed_pass_m": (0.9, 7.5), "shot_post_m": (3.0, 8.0),
    "out_margin": (0.1, 1.0), "out_time_s": (0.1, 1.0), "stop_context_s": (2.0, 8.0), "stop_smooth_s": (1.0, 5.0),
    "min_stop_s": (1.0, 4.0), "kickoff_tol": (3.0, 5.0), "throwin_tol": (0.25, 0.75), "corner_m": (1.5, 4.5),
    "goal_area_tol": (0.25, 0.75), "fk_wall_tol": (2.0, 6.0), "walk_speed": (1.0, 2.0),
    "restart_limit_s": (20.0, 120.0), "clear_margin_m": (0.0, 2.0), "gap_long_s": (2.0, 10.0), "gap_same_spot_m": (1.0, 6.0),
}
CONFIDENCE = {"control_confirmed": 0.9, "dribble": 0.85, "knock_on": 0.7, "hand_over": 0.6, "keeper_catch": 0.85,
              "interception": 0.8, "one_touch": 0.7, "release": 0.85, "shot": 0.8, "clearance": 0.6, "deflection": 0.6,
              "block": 0.6, "parry": 0.6, "duel": 0.6, "loose_ball": 0.5, "tackle_short_pass": 0.55, "out_of_play": 0.9,
              "goal": 0.9, "whistle": 0.6, "restart_setup": 0.8, "before_kickoff": 0.9, "after_end": 0.8, "missing_carry": 0.4,
              "missing_unknown": 0.1, "no_touch_yet": 0.2, "absorbed": 0.4, "whistle_restart": 0.6, "gap_free_kick": 0.5,
              "gap_same_team": 0.5}


@dataclass
class C2Result:
    state: np.ndarray
    player: np.ndarray
    team: np.ndarray
    confidence: np.ndarray
    rule: np.ndarray
    rule_names: list
    touches: list = field(default_factory=list)   # dicts: frame, col, kind
    passes: list = field(default_factory=list)    # (release frame, passer id, next actor id or -1)
    restarts: list = field(default_factory=list)  # (frame, T2 kind, why the ball was dead)


# ---------------------------------------------------------------- shared measurements

def _kick_signal(t, x, y, k):
    n = len(t)
    va, vb = np.full((n, 2), np.nan), np.full((n, 2), np.nan)
    i = np.arange(k, n - k)
    va[i] = np.stack([x[i + k] - x[i], y[i + k] - y[i]], 1) / (t[i + k] - t[i])[:, None]
    vb[i] = np.stack([x[i] - x[i - k], y[i] - y[i - k]], 1) / (t[i] - t[i - k])[:, None]
    sa, sb = np.hypot(*va.T), np.hypot(*vb.T)
    dv = np.hypot(*(va - vb).T)
    cos = np.sum(va * vb, 1) / np.maximum(sa * sb, 1e-9)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    acc = dv / (k * np.median(np.diff(t)) if n > 1 else 1.0)
    return acc, sa, sb, ang


def _local_max(v, w):
    vv = np.where(np.isfinite(v), v, -np.inf)
    pad = np.pad(vv, w, constant_values=-np.inf)
    return (vv == np.lib.stride_tricks.sliding_window_view(pad, 2 * w + 1).max(axis=1)) & np.isfinite(v)


def _median_filter(v, w):
    """Median filter (window w samples, NaN ignored) on a coarse series."""
    if w <= 1:
        return v
    h = w // 2
    pad = np.pad(v, h, constant_values=np.nan)
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * h + 1)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(win, axis=1)


class _Scene:
    def __init__(self, t, period, ball, px, py, player_id, team_id, pitch, p: C2Params):
        self.t, self.period = np.asarray(t, float), np.asarray(period)
        self.n = len(t)
        self.dt = float(np.median(np.diff(self.t))) if self.n > 1 else 1 / 30
        self.fps = 1 / self.dt
        self.present = ball.flag != FLAG_MISSING
        self.bx, self.by = np.where(self.present, ball.x, np.nan), np.where(self.present, ball.y, np.nan)
        self.bz = np.where(self.present, np.nan_to_num(ball.z), np.nan)
        self.vx, self.vy = ball.vx, ball.vy
        self.speed = np.hypot(np.nan_to_num(ball.vx), np.nan_to_num(ball.vy))
        self.px, self.py = px, py
        self.pid, self.team = np.asarray(player_id), np.asarray(team_id)
        self.hl, self.hw = pitch[0] / 2, pitch[1] / 2
        self.teams = sorted(set(int(x) for x in self.team))
        d = np.hypot(px - self.bx[:, None], py - self.by[:, None]).astype(np.float32)
        self.d = np.where(np.isfinite(d), d, np.inf)
        self.near = np.argmin(self.d, axis=1)
        self.dmin = self.d[np.arange(self.n), self.near]
        self.acc, self.sa, self.sb, self.ang = _kick_signal(self.t, self.bx, self.by, p.peak_frames)
        # player speeds -> team activity (mean speed of present players), computed at 5 Hz and median-filtered
        vxp = np.gradient(np.nan_to_num(px), self.t, axis=0)
        vyp = np.gradient(np.nan_to_num(py), self.t, axis=0)
        sp = np.where(np.isfinite(px), np.hypot(vxp, vyp), np.nan)
        sp = np.where(sp > 12, np.nan, sp)            # jumps of the tracker are not running speed
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                self.mean_speed = np.nanmean(sp, axis=1)
        step = max(1, int(round(self.fps / 5)))
        coarse = self.mean_speed[::step]
        w = max(1, int(round(p.stop_smooth_s * 5)))
        sm = _median_filter(coarse, w)
        self.activity = np.repeat(sm, step)[: self.n]
        # keepers: per team and period, the player whose median x is nearest the team's own goal line
        self.keeper = {}
        self.own_sign = {}
        for per in np.unique(self.period):
            sel = self.period == per
            for tm in self.teams:
                cols = np.flatnonzero(self.team == tm)
                with np.errstate(all="ignore"):
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        med = np.nanmedian(px[sel][:, cols], axis=0)
                if not np.isfinite(med).any():
                    continue
                side = -1.0 if np.nanmean(med) < 0 else 1.0       # the half a team stands in = its own half
                self.own_sign[(int(per), tm)] = side
                self.keeper[(int(per), tm)] = int(cols[np.nanargmax(side * med)])

    def is_keeper_in_box(self, col, f) -> bool:
        tm = int(self.team[col])
        per = int(self.period[f])
        if self.keeper.get((per, tm)) != col:
            return False
        side = self.own_sign.get((per, tm), 1.0)
        x, y = self.px[f, col], self.py[f, col]
        return bool(np.isfinite(x) and side * x >= self.hl - BOX_DEPTH and abs(y) <= BOX_HALF_W)


# ---------------------------------------------------------------- B: touches

def detect_touches(sc: _Scene, p: C2Params) -> list[dict]:
    """B1: local peak of ball acceleration with a speed or direction change, a player within the touch radius at the local
    minimum of player-ball distance within +-match_tol_s, ball height within the limits. B2 follows (no change = no peak).
    B3 actor: nearest at that minimum; a team-mate tie goes to the previous actor; an opponent tie is flagged as a duel.
    B4 kind: 'reception' when the ball slows down, 'kick' when it speeds up, 'turn' otherwise."""
    k = p.peak_frames
    ratio = np.maximum(sc.sa, sc.sb) / np.maximum(np.minimum(sc.sa, sc.sb), 0.5)
    changed = (np.abs(sc.sa - sc.sb) >= p.speed_abs) | (ratio >= p.speed_ratio) | \
              ((sc.ang >= p.dir_change) & (np.minimum(sc.sa, sc.sb) > 1.0))
    cand = np.flatnonzero(_local_max(sc.acc, k) & (sc.acc >= p.kick_acc) & changed & sc.present)
    w = int(round(p.match_tol_s * sc.fps))
    out, last_col = [], -1
    for f in cand.tolist():
        a, b = max(0, f - w), min(sc.n, f + w + 1)
        dw = sc.d[a:b]
        if not np.isfinite(dw).any():
            continue
        mins = dw.min(axis=0)
        order = np.argsort(mins)
        col = int(order[0])
        dmin = float(mins[col])
        kind = "kick" if sc.sa[f] > sc.sb[f] * 1.1 else "reception" if sc.sb[f] > sc.sa[f] * 1.1 else "turn"
        radius = {"kick": p.touch_foot, "reception": p.touch_reception, "turn": p.touch_foot}[kind]
        tackle_ok = dmin <= p.touch_tackle
        if dmin > radius and not tackle_ok:
            continue
        z = sc.bz[f] if np.isfinite(sc.bz[f]) else 0.0
        keeper = sc.is_keeper_in_box(col, f)
        if z > (p.h_max + (1.0 if keeper else 0.0)):
            continue
        if p.h_foot < z < p.h_header:                 # neither a foot nor a header height (only when ranges leave a gap)
            continue
        duel = False
        if len(order) > 1 and mins[order[1]] - dmin <= p.tie_m:
            other = int(order[1])
            if sc.team[other] == sc.team[col] and other == last_col:
                col, dmin = other, float(mins[other])
            elif sc.team[other] != sc.team[col]:
                duel = True
        fmin = a + int(np.argmin(dw[:, col]))
        out.append(dict(frame=f, fmin=fmin, col=col, dist=dmin, kind=kind, duel=duel, keeper=keeper,
                        within_radius=dmin <= radius, z=z))
        last_col = col
    return out


# ---------------------------------------------------------------- C + D: state machine

def _control_confirmed(sc: _Scene, p: C2Params, col, f, until):
    """C1: after the touch the ball stays within the control radius of the player for the grace period (>= 60 % of the
    frames with a ball), or the player can reach the ball's position predicted reach_s ahead; ball slow enough."""
    g = int(round(p.grace_s * sc.fps))
    b = min(until, f + g + 1, sc.n)
    if b <= f:
        return False
    dd = sc.d[f:b, col]
    ok = np.isfinite(dd)
    if ok.sum() == 0:
        return False
    slow = np.nanmedian(sc.speed[f:b]) <= p.control_max_speed
    if p.v2 and "c" not in p.v2_skip and p.clear_margin_m > 0:                     # v2: clearly closer than the nearest opponent [CH]
        dopp = sc.d[f:b][:, sc.team != sc.team[col]].min(axis=1)
        if (dd[ok] + p.clear_margin_m <= dopp[ok]).mean() < 0.6:
            return False
    if (dd[ok] <= p.control_radius).mean() >= 0.6 and slow:
        return True
    r = int(round(p.reach_s * sc.fps))
    fr = min(f + r, sc.n - 1)
    if not (np.isfinite(sc.bx[f]) and np.isfinite(sc.px[fr, col])):
        return False
    predx, predy = sc.bx[f] + np.nan_to_num(sc.vx[f]) * p.reach_s, sc.by[f] + np.nan_to_num(sc.vy[f]) * p.reach_s
    reach = np.hypot(sc.px[f, col] - predx, sc.py[f, col] - predy) if np.isfinite(sc.px[f, col]) else np.inf
    return bool(slow and reach <= p.control_radius + p.run_speed * p.reach_s)


def _released(sc: _Scene, p: C2Params, col, f, until):
    """C3: after a kick the ball moves beyond the release distance from the kicker before the next touch."""
    b = min(until, sc.n)
    if b <= f + 1 or not np.isfinite(sc.px[f, col]):
        return False, None
    dd = sc.d[f:b, col]                                    # distance to the kicker where he is now, not where he kicked
    hit = np.flatnonzero(np.isfinite(dd) & (dd >= p.release_m))
    return (True, f + int(hit[0])) if len(hit) else (False, None)


def _is_shot(sc: _Scene, p: C2Params, col, f, until):
    """A release towards the goal the kicker attacks, whose path crosses that goal line within shot_post_m of a post."""
    tm, per = int(sc.team[col]), int(sc.period[f])
    side = -sc.own_sign.get((per, tm), 1.0)               # attacked goal is opposite the own half
    vx, vy = np.nan_to_num(sc.vx[min(f + 3, sc.n - 1)]), np.nan_to_num(sc.vy[min(f + 3, sc.n - 1)])
    if side * vx <= 0 or not np.isfinite(sc.bx[f]):
        return False
    tt = (side * sc.hl - sc.bx[f]) / vx
    if tt <= 0 or tt > 3:
        return False
    yy = sc.by[f] + vy * tt
    return bool(abs(yy) <= POST_Y + p.shot_post_m and side * sc.bx[f] >= sc.hl - 35)


def _is_clearance(sc: _Scene, p: C2Params, col, f, until):
    """C3: a long, high kick from the own third (team-mate direction is not checked: no reliable pass target)."""
    tm, per = int(sc.team[col]), int(sc.period[f])
    side = sc.own_sign.get((per, tm), 1.0)
    if not np.isfinite(sc.bx[f]) or side * sc.bx[f] < sc.hl / 3:
        return False
    b = min(until, sc.n)
    seg_x, seg_y, seg_z = sc.bx[f:b], sc.by[f:b], sc.bz[f:b]
    if not np.isfinite(seg_x).any():
        return False
    far = np.nanmax(np.hypot(seg_x - sc.bx[f], seg_y - sc.by[f]))
    return bool(far >= p.clearance_len_m and np.nanmax(np.nan_to_num(seg_z)) >= p.clearance_z_m)


def _control_start(sc: _Scene, p: C2Params, col, f, lo):
    """v2 back-dating [CH]: a confirmed new owner's control is dated back to when the ball entered their control radius with
    them clearly closer than any opponent; never before the previous touch (lo)."""
    b = f
    opp = sc.team != sc.team[col]
    while b - 1 > lo:
        dc = sc.d[b - 1, col]
        if not (dc <= p.control_radius and dc + p.clear_margin_m <= sc.d[b - 1][opp].min()):
            break
        b -= 1
    return b


def _runs(mask, period):
    """(start, end) of runs of True that stay inside one period."""
    out, i, n = [], 0, len(mask)
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return out
    br = np.flatnonzero((np.diff(idx) > 1) | (period[idx[1:]] != period[idx[:-1]]))
    starts, ends = np.r_[idx[0], idx[br + 1]], np.r_[idx[br], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def run_c2(t, period, ball, px, py, player_id, team_id, pitch, params: C2Params | None = None) -> C2Result:
    p = params or C2Params()
    sc = _Scene(t, period, ball, px, py, player_id, team_id, pitch, p)
    n = sc.n
    rules: list[str] = []

    def rid(name):
        if name not in rules:
            rules.append(name)
        return rules.index(name)

    state = np.full(n, -1, np.int8)
    player = np.full(n, -1, np.int64)
    team = np.full(n, -1, np.int64)
    rule = np.full(n, -1, np.int16)

    def paint(a, b, st, name, col=None, tm=None):
        if b < a:
            return
        state[a:b + 1] = st
        player[a:b + 1] = -1 if col is None else sc.pid[col]
        team[a:b + 1] = (-1 if tm is None else tm) if col is None else sc.team[col]
        if tm is not None and col is None:
            team[a:b + 1] = tm
        rule[a:b + 1] = rid(name)

    touches = detect_touches(sc, p)
    passes = []
    restarts = []
    tframes = np.array([x["frame"] for x in touches], int)

    # ---------------------------------------------------------------- A rules: dead ball (layer 1 in v2)
    def dead_layer():
        dead = np.zeros(n, bool)
        dead_rule = np.full(n, -1, np.int16)
        starts_dead = []                                        # (frame, kind)
        hl, hw = sc.hl, sc.hw
        beyond_x = np.abs(sc.bx) > hl + BALL_R + p.out_margin
        beyond_y = np.abs(sc.by) > hw + BALL_R + p.out_margin
        beyond = sc.present & (beyond_x | beyond_y)
        min_out = max(1, int(round(p.out_time_s * sc.fps)))
        i = 0
        while i < n:                                            # A1 / A2: out over a line for the T1 time
            if beyond[i]:
                j = i
                while j + 1 < n and beyond[j + 1] and sc.period[j + 1] == sc.period[i]:
                    j += 1
                if j - i + 1 >= min_out:
                    c = i
                    while c > 0 and sc.present[c - 1] and (abs(sc.bx[c - 1]) > hl or abs(sc.by[c - 1]) > hw) and sc.period[c - 1] == sc.period[i]:
                        c -= 1                                  # back to the crossing
                    goal = abs(sc.bx[i]) > hl and abs(sc.by[i]) <= POST_Y and (sc.bz[i] if np.isfinite(sc.bz[i]) else 0) <= BAR_Z
                    starts_dead.append((c, "goal" if goal else ("touchline" if beyond_y[i] and not beyond_x[i] else "goal_line")))
                i = j + 1
            else:
                i += 1
        # A1: the ball disappears at a line
        vanish = np.flatnonzero(sc.present[:-1] & ~sc.present[1:]) if n > 1 else np.zeros(0, int)
        gap_needed = int(round(p.min_stop_s * sc.fps))         # a camera gap is not a ball leaving the pitch
        for v in vanish.tolist():
            near_line = (hl - abs(sc.bx[v]) <= p.vanish_line_m) or (hw - abs(sc.by[v]) <= p.vanish_line_m)
            if near_line and not sc.present[v + 1:v + 1 + gap_needed].any() and v + gap_needed < n:
                starts_dead.append((v + 1, "touchline" if hw - abs(sc.by[v]) <= p.vanish_line_m else "goal_line"))
        starts_dead.sort()

        def first_restart_after(f0):
            """A4: the restart = the first touch with the ball back inside the pitch (the first kick of the placed ball); its
            T2 pattern only names the restart (no match = free kick by default [VC])."""
            per = sc.period[min(f0, n - 1)]
            for k in np.flatnonzero(tframes >= f0).tolist():
                tx = touches[k]
                f = tx["frame"]
                if sc.period[f] != per:
                    break
                if abs(sc.bx[f]) <= hl and abs(sc.by[f]) <= hw:
                    return tx, (restart_kind(tx) or "free_kick")
            return None, None

        covered_until = -1
        for f0, kind in starts_dead:
            if f0 < covered_until:
                continue
            tx, kd = first_restart_after(f0)
            if tx is not None:
                restarts.append((tx["frame"], kd, kind))
            end = (tx["frame"] - 1) if tx is not None else int(np.flatnonzero(sc.period == sc.period[f0])[-1])
            dead[f0:end + 1] = True
            dead_rule[f0:end + 1] = rid("goal" if kind == "goal" else "out_of_play")
            covered_until = end + 1
        ctx = int(round(p.stop_context_s * sc.fps))
        min_stop = int(round(p.min_stop_s * sc.fps))

        def stop_start(f):
            """A3 step 2: search back from the restart for the last competitive moment: players moving or the ball contested."""
            s = f - 1
            per = sc.period[f]
            while s > 0 and sc.period[s - 1] == per and not dead[s - 1]:
                lo = max(0, s - ctx)
                act = np.nanmax(sc.activity[lo:s + 1]) if np.isfinite(sc.activity[lo:s + 1]).any() else 0
                holder = sc.near[s]
                opp = (sc.team != sc.team[holder]) & (sc.d[s] <= sc.dmin[s] + p.press_m)
                contested = sc.present[s] and opp.any() and sc.dmin[s] <= p.control_radius * 2
                if act >= p.walk_speed or contested:
                    break
                s -= 1
            return s

        # A3 whistle: a restart pattern (free kick, penalty, drop ball, kick-off) with no line crossing before it
        for tx in touches:
            f = tx["frame"]
            if dead[max(0, f - 1)]:
                continue
            kd = restart_kind(tx)
            drop = sc.present[f] and f > 15 and np.nanmax(np.nan_to_num(sc.bz[f - 15:f])) >= 1.0 and \
                np.sum(sc.d[f] <= 3.0) >= 2 and len({int(sc.team[c]) for c in np.flatnonzero(sc.d[f] <= 3.0)}) == 2 and \
                np.nanmax(sc.speed[max(0, f - 60):f - 15]) <= 1.0 if f > 60 else False
            if kd not in ("free_kick", "penalty", "kick_off") and not drop:
                continue
            s = stop_start(f)
            if f - s >= min_stop:
                dead[s:f] = True
                dead_rule[s:f] = rid("whistle")
                restarts.append((f, kd or "drop_ball", "whistle"))
        low = np.nan_to_num(sc.activity, nan=0.0) < p.walk_speed
        if p.v2 and "a" not in p.v2_skip:
            # v2 A3: a collective slowdown is a stoppage only if a T2 restart pattern follows within restart_limit_s; the
            # stoppage start is found backwards (step 2) from the slowdown that precedes that restart, since during the set-up
            # the ball looks contested (wall, taker). Touches that match no pattern while players
            # are still slow (fetching the ball) are skipped; a non-pattern touch after play resumed cancels it.
            lim = int(round(p.restart_limit_s * sc.fps))
            for i, j in _runs(low & ~dead, sc.period):
                if j - i + 1 < min_stop:
                    continue
                found = None
                for k in np.flatnonzero((tframes > i) & (tframes <= j + lim)).tolist():
                    tx = touches[k]
                    f = tx["frame"]
                    if sc.period[f] != sc.period[i] or dead[f]:
                        break
                    kd = restart_kind(tx)
                    if kd is not None:
                        found = (f, kd)
                        break
                    if f > j + ctx:
                        break
                if found is None:
                    continue
                f, kd = found
                s = stop_start(i + 1)                       # back from the slowdown: during the set-up players stand close
                if f - s >= min_stop:
                    fill = ~dead[s:f]
                    dead_rule[s:f][fill] = rid("whistle_restart")
                    dead[s:f] = True
                    restarts.append((f, kd, "whistle_slowdown"))
        else:
            # A3 / [MU]: a collective slowdown (team activity below walking speed for at least min_stop_s) is a stoppage; it
            # ends at the first touch after activity recovers (the restart kick), searched within the context window
            i = 0
            while i < n:
                if low[i] and not dead[i]:
                    j = i
                    while j + 1 < n and low[j + 1] and sc.period[j + 1] == sc.period[i]:
                        j += 1
                    if j - i + 1 >= min_stop:
                        after = tframes[(tframes > j) & (tframes <= j + ctx)]
                        after = after[sc.period[after] == sc.period[i]] if len(after) else after
                        end = int(after[0]) - 1 if len(after) else j
                        fill = ~dead[i:end + 1]
                        dead_rule[i:end + 1][fill] = rid("whistle_slowdown")
                        dead[i:end + 1] = True
                    i = j + 1
                else:
                    i += 1
        if p.v2 and "e" not in p.v2_skip:
            # v2 A6 [KI]: a long gap after which the ball reappears near where it vanished = a free kick (DEAD until the kick)
            for i, j in _runs(~sc.present, sc.period):
                if i == 0 or j + 1 >= n or sc.period[i - 1] != sc.period[i] or sc.period[j + 1] != sc.period[i]:
                    continue
                if (j - i + 1) / sc.fps < p.gap_long_s or dead[i:j + 1].all():
                    continue
                if np.hypot(sc.bx[j + 1] - sc.bx[i - 1], sc.by[j + 1] - sc.by[i - 1]) > p.gap_same_spot_m:
                    continue
                k = np.flatnonzero((tframes > j) & (tframes <= j + int(round(p.restart_limit_s * sc.fps))))
                end = int(tframes[k[0]]) - 1 if len(k) and sc.period[tframes[k[0]]] == sc.period[i] else j
                fill = ~dead[i:end + 1]
                dead_rule[i:end + 1][fill] = rid("gap_free_kick")
                dead[i:end + 1] = True
        # A4 restart setup is the stretch up to the first kick (already part of the dead stretch). A5: before kick-off, after end
        for per in np.unique(sc.period):
            fr = np.flatnonzero(sc.period == per)
            tl = [x for x in touches if fr[0] <= x["frame"] <= fr[-1]]
            ko = next((x for x in tl if restart_kind(x) == "kick_off"), tl[0] if tl else None)
            if ko is not None:
                restarts.append((ko["frame"], restart_kind(ko) or "first_touch", "period_start"))
            end0 = (ko["frame"] - 1) if ko else int(fr[-1])
            dead[fr[0]:end0 + 1] = True
            dead_rule[fr[0]:end0 + 1] = rid("before_kickoff")
            if tl:
                lastf = tl[-1]["frame"]
                s = int(fr[-1])
                while s > lastf and (not np.isfinite(sc.activity[s]) or sc.activity[s] < p.walk_speed):
                    s -= 1
                if fr[-1] - s >= min_stop:
                    dead[s + 1:fr[-1] + 1] = True
                    dead_rule[s + 1:fr[-1] + 1] = rid("after_end")
        return dead, dead_rule, min_stop

    def restart_kind(tx) -> str | None:
        """T2 restart patterns at a touch (the first kick of a placed ball)."""
        f, col = tx["frame"], tx["col"]
        hl, hw = sc.hl, sc.hw
        x, y = sc.bx[f], sc.by[f]
        if not np.isfinite(x):
            return None
        tm = int(sc.team[col])
        per = int(sc.period[f])
        pre = slice(max(0, f - int(p.still_s * sc.fps)), f)
        still = np.nanmax(sc.speed[pre]) <= p.still_speed if f > 0 else False
        if p.v2 and "a" not in p.v2_skip and f > 0:               # v2: the smoothed ball speeds up ~0.2 s before the kick
            off = 2 * p.peak_frames
            pre2 = slice(max(0, f - off - int(p.still_s * sc.fps)), max(1, f - off))
            still = bool(np.nanmedian(sc.speed[pre2]) <= p.still_speed_v2)
        if np.hypot(x, y) <= 2.0:                                   # kick-off: everyone in their own half (tolerance)
            ok = True
            for t2 in sc.teams:
                side = sc.own_sign.get((per, t2), 1.0)
                cols = np.flatnonzero(sc.team == t2)
                xs = sc.px[f, cols]
                ok &= bool(np.all(~np.isfinite(xs) | (side * xs >= -p.kickoff_tol)))
            if ok:
                return "kick_off"
        if np.min(np.hypot(abs(x) - hl, abs(y) - hw)) <= p.corner_m and tx["dist"] <= p.corner_m:
            return "corner"
        if abs(sc.py[f, col]) >= hw - p.throwin_tol and abs(y) >= hw - 2.5:
            return "throw_in"
        if abs(x) >= hl - GOAL_AREA_DEPTH - p.goal_area_tol and abs(y) <= GOAL_AREA_HALF_W + p.goal_area_tol and still:
            return "goal_kick"
        if abs(abs(x) - (hl - PEN_SPOT)) <= 2.0 and abs(y) <= 2.0 and still:
            opp = sc.keeper.get((per, [t2 for t2 in sc.teams if t2 != tm][0] if len(sc.teams) > 1 else tm))
            if opp is not None and np.isfinite(sc.px[f, opp]) and hl - abs(sc.px[f, opp]) <= 2.0 and abs(sc.py[f, opp]) <= POST_Y:
                return "penalty"
        if still:
            moved = np.nanmax(np.hypot(sc.px[pre, col] - sc.px[f, col], sc.py[pre, col] - sc.py[f, col])) if f > 0 else 0
            opp_d = sc.d[f][sc.team != tm]
            if moved <= 1.0 + 1e-9 or np.min(opp_d) >= 9.15 - p.fk_wall_tol:
                return "free_kick"
        return None

    layers = p.v2 and "b" not in p.v2_skip
    if layers:                                              # v2 two layers: alive / dead first
        dead, dead_rule, min_stop = dead_layer()
    else:
        dead = np.zeros(n, bool)

    # ---------------------------------------------------------------- C rules: walk the touches of each period (layer 2 in v2)
    for per in np.unique(sc.period):
        fr = np.flatnonzero(sc.period == per)
        f_first, f_last = int(fr[0]), int(fr[-1])
        tl = [x for x in touches if f_first <= x["frame"] <= f_last and not dead[x["frame"]]]
        owner, own_team, travel_team = -1, -1, -1
        last_release = None                         # (frame, team, start x, start y)
        prev_f = f_first
        for i, x in enumerate(tl):
            f, col = x["frame"], x["col"]
            nxt = tl[i + 1]["frame"] if i + 1 < len(tl) else f_last + 1
            if layers and dead[prev_f:f].any():       # play restarted: nothing carries over a dead stretch
                owner, own_team, travel_team, last_release = -1, -1, -1, None
                prev_f = int(np.flatnonzero(dead[prev_f:f])[-1]) + prev_f
            tm = int(sc.team[col])
            ball_team = own_team if owner >= 0 else travel_team
            confirmed = _control_confirmed(sc, p, col, f, nxt)
            released, rel_f = _released(sc, p, col, f, nxt) if x["kind"] != "reception" else (False, None)   # B4
            arrival = np.hypot(sc.bx[f] - sc.px[f, col], sc.by[f] - sc.py[f, col]) if last_release is None else \
                np.hypot(sc.bx[f] - last_release[2], sc.by[f] - last_release[3])
            opp_in_duel = np.sum((sc.team != tm) & (sc.d[f] <= p.duel_radius))
            start_ctrl = _control_start(sc, p, col, f, prev_f) if (p.v2 and "d" not in p.v2_skip and col != owner) else f
            prev_f = f
            # C7 duel: >= 2 opposing players within the duel radius and no clear controller
            if (x["duel"] or opp_in_duel >= 1) and np.sum(sc.d[f] <= p.duel_radius) >= 2 and not (confirmed and col == owner):
                paint(f, nxt - 1, LOOSE, "duel")
                owner, travel_team = -1, -1
                continue
            # C6 keeper
            if x["keeper"] and ((x["z"] >= 0.5 and np.nanmedian(sc.speed[f:min(nxt, f + 15)]) < 2.0) or (confirmed and not released)):
                paint(start_ctrl, nxt - 1, CONTROL, "keeper_catch", col)
                owner, own_team, travel_team, last_release = col, tm, -1, None
                continue
            if x["keeper"] and released and x["kind"] != "kick":
                paint(f, nxt - 1, LOOSE, "parry", tm=tm)
                owner, travel_team = -1, -1
                continue
            # C4 short opponent-won "pass" = tackle / duel: relabel the short travel as LOOSE
            if last_release is not None and ball_team != -1 and tm != ball_team and owner < 0:
                travel_len = np.hypot(sc.bx[f] - last_release[2], sc.by[f] - last_release[3])
                if travel_len < p.min_failed_pass_m:
                    paint(last_release[0], f - 1, LOOSE, "tackle_short_pass")
            if released:
                # C3 release (pass, shot or clearance); C10 one-touch: CONTROL for the touch frame, then the flight
                one_touch = col != owner
                if one_touch:
                    paint(f, f, CONTROL, "one_touch", col)
                    start = f + 1
                else:
                    start = f
                if _is_clearance(sc, p, col, f, nxt):
                    paint(start, nxt - 1, LOOSE, "clearance", tm=tm)
                    travel_team = -1
                else:
                    shot = _is_shot(sc, p, col, f, nxt)
                    paint(start, nxt - 1, TRAVELLING, "shot" if shot else "release", tm=tm)
                    player[start:nxt] = sc.pid[col]
                    travel_team = tm
                    if not shot:
                        nact = int(sc.pid[tl[i + 1]["col"]]) if i + 1 < len(tl) else -1
                        passes.append((f, int(sc.pid[col]), nact))
                last_release = (f, tm, sc.bx[f], sc.by[f])
                owner = -1
                continue
            if confirmed:
                if col == owner:
                    name = "dribble"                                            # C2
                elif ball_team == tm and owner >= 0 and owner != col and x["kind"] != "reception":
                    name = "hand_over"                                          # C9
                elif ball_team not in (-1, tm) and arrival >= p.arrival_m:
                    name = "interception"                                       # C4
                else:
                    name = "control_confirmed"                                  # C1
                paint(start_ctrl if name != "dribble" else f, nxt - 1, CONTROL, name, col)
                owner, own_team, travel_team, last_release = col, tm, -1, None
                continue
            # not controlled, not released
            if col == owner:
                paint(f, nxt - 1, CONTROL, "knock_on", col)                     # C2: same player, ball got away
                continue
            if ball_team not in (-1, tm) and x["dist"] <= p.touch_tackle:
                small_turn = sc.ang[f] < p.deflect_angle and sc.sb[f] > 0 and \
                    1 / p.deflect_ratio <= sc.sa[f] / max(sc.sb[f], 1e-6) <= p.deflect_ratio
                if small_turn and travel_team == ball_team:
                    paint(f, nxt - 1, TRAVELLING, "deflection", tm=ball_team)   # C5 deflection: flight continues
                    continue
                paint(f, nxt - 1, LOOSE, "block", tm=tm)                        # C5 block
                owner, travel_team = -1, -1
                continue
            paint(f, nxt - 1, LOOSE, "loose_ball", tm=tm)                       # C8
            owner, travel_team = -1, -1
        if tl:
            paint(f_first, tl[0]["frame"] - 1, UNKNOWN, "no_touch_yet")

    if not layers:
        dead, dead_rule, min_stop = dead_layer()
    state[dead] = DEAD
    rule[dead] = dead_rule[dead]
    player[dead], team[dead] = -1, -1

    # ---- A6 missing ball (not dead): keep the previous state for short gaps, then UNKNOWN. v2 [KI]: when the same team
    # was on the ball before and after the gap, the gap keeps that team (first half: the state before, second half: after)
    short = int(round(p.short_gap_s * sc.fps))
    miss = ~sc.present & ~dead
    for i, j in _runs(miss, sc.period):
        if p.v2 and "e" not in p.v2_skip and i > 0 and j + 1 < n and sc.period[i - 1] == sc.period[i] == sc.period[j + 1] and \
                state[i - 1] in (CONTROL, TRAVELLING) and state[j + 1] in (CONTROL, TRAVELLING) and team[i - 1] >= 0 and \
                team[i - 1] == team[j + 1]:
            mid = (i + j + 1) // 2
            state[i:mid], player[i:mid], team[i:mid] = state[i - 1], player[i - 1], team[i - 1]
            state[mid:j + 1], player[mid:j + 1], team[mid:j + 1] = state[j + 1], player[j + 1], team[j + 1]
            rule[i:j + 1] = rid("gap_same_team")
            continue
        if i > 0 and sc.period[i - 1] == sc.period[i] and state[i - 1] >= 0:
            k = min(j, i + short - 1)
            state[i:k + 1], player[i:k + 1], team[i:k + 1] = state[i - 1], player[i - 1], team[i - 1]
            rule[i:k + 1] = rid("missing_carry")
            if j > k:
                state[k + 1:j + 1], player[k + 1:j + 1], team[k + 1:j + 1] = UNKNOWN, -1, -1
                rule[k + 1:j + 1] = rid("missing_unknown")
        else:
            state[i:j + 1], rule[i:j + 1] = UNKNOWN, rid("missing_unknown")
    unset = state < 0
    state[unset], rule[unset] = UNKNOWN, rid("no_touch_yet")
    # ---- D3 minimum durations: short TRAVELLING / LOOSE runs are absorbed by the state before them
    mins = max(1, int(round(p.min_state_s * sc.fps)))
    i = 1
    while i < n:
        if state[i] in (TRAVELLING, LOOSE) and state[i] != state[i - 1] and sc.period[i] == sc.period[i - 1]:
            j = i
            while j + 1 < n and state[j + 1] == state[i]:
                j += 1
            if j - i + 1 < mins and state[i - 1] in (CONTROL, TRAVELLING, LOOSE):
                state[i:j + 1], player[i:j + 1], team[i:j + 1] = state[i - 1], player[i - 1], team[i - 1]
                rule[i:j + 1] = rid("absorbed")
            i = j + 1
        else:
            i += 1
    conf = np.array([CONFIDENCE.get(rules[r], 0.5) for r in rule], np.float32)          # D4
    return C2Result(state, player, team, conf, rule, rules, touches, passes, sorted(restarts))             # D5: rule per frame
