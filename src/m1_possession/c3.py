"""M1 contestant C3: gradient-boosting features and the football-logic decoder (plan: "C3 features"). Tag-free, numpy only;
the models are trained in scripts/m1/c3.py.

Rows are sampled every ``STEP`` frames (10 Hz). Three feature blocks per row:
- base: ball (position, speed, kick signal, flags, out-of-bounds, still ball, time since seen), pitch geometry, the players
  around the ball (distances, the margin to the nearest opponent, crowding), team activity with context at -4 ... +4 s [MU],
  ball / distance context, and optionally the gap features;
- c2: C2-v2's own decision (state, rule, confidence, context), its touches (time since / to, kind, release and arrival
  distance, duel) and restarts;
- candidates: the 5 players nearest the ball with per-player measurements (for the learned owner ranking).
Decoder: Viterbi over DEAD / CONTROL-home / CONTROL-away / TRAVELLING / LOOSE with hard football rules: DEAD ends only at a
detected touch (the restart kick), CONTROL is entered only at a touch (the owner changes only at a touch), no direct switch
from one team's CONTROL to the other's; afterwards switches are dated back to the touch and the owner is fixed between touches.
"""
from __future__ import annotations

import warnings

import numpy as np

from .rules import CONFIDENCE, _kick_signal

STEP = 3
OFFS_S = (-4.0, -2.0, -0.8, -0.4, 0.0, 0.4, 0.8, 2.0, 4.0)
CTX_S = (-2.0, -0.8, -0.4, 0.4, 0.8, 2.0)
N_CAND = 5
RULE_VOCAB = sorted(set(CONFIDENCE) | {"whistle_slowdown", "whistle_restart", "gap_free_kick", "gap_same_team"})
RESTART_VOCAB = ("kick_off", "corner", "throw_in", "goal_kick", "penalty", "free_kick", "drop_ball", "first_touch")
WHY_VOCAB = ("touchline", "goal_line", "goal", "whistle", "whistle_slowdown", "period_start")
KIND = {"kick": 0, "reception": 1, "turn": 2}
DEAD, CONTROL, TRAVELLING, LOOSE = 0, 1, 2, 3
S_DEAD, S_CH, S_CA, S_TRAV, S_LOOSE = range(5)


def _nanfix(v):
    return np.where(np.isfinite(v), v, np.nan)


class Prepared:
    """Per-match arrays shared by the feature blocks (full frame rate)."""

    def __init__(self, t, period, bx, by, bz, bvx, bvy, bflag, px, py, team_sign, pitch, real_present):
        warnings.simplefilter("ignore", RuntimeWarning)
        self.t, self.period = np.asarray(t, float), np.asarray(period)
        self.n = n = len(t)
        self.fps = 1 / float(np.median(np.diff(self.t)))
        self.bx, self.by, self.bz = _nanfix(bx), _nanfix(by), _nanfix(bz)
        self.bvx, self.bvy, self.bflag = bvx, bvy, bflag
        self.px, self.py, self.ts = px, py, np.asarray(team_sign, np.float32)
        self.hl, self.hw = pitch[0] / 2, pitch[1] / 2
        self.s = np.ones(n, np.float32)                       # orientation: home defends -x
        home = self.ts > 0
        for per in np.unique(self.period):
            sel = self.period == per
            med = np.nanmedian(px[sel][:, home])
            self.s[sel] = -1.0 if np.isfinite(med) and med > 0 else 1.0
        k = 3
        v = lambda a: np.r_[np.full((k,) + a.shape[1:], np.nan), (a[2 * k:] - a[:-2 * k]) /
                            (self.t[2 * k:] - self.t[:-2 * k]).reshape((-1,) + (1,) * (a.ndim - 1)),
                            np.full((k,) + a.shape[1:], np.nan)].astype(np.float32)
        self.vxp, self.vyp = v(px), v(py)
        sp = np.hypot(self.vxp, self.vyp)
        sp[~(sp <= 12)] = np.nan
        self.sp = sp
        self.accp = np.abs(v(sp))
        top = np.sort(np.nan_to_num(sp, nan=0.0), 1)
        fin = np.isfinite(sp)
        self.act = dict(mean_sp=np.nanmean(sp, 1), top3_sp=top[:, -3:].mean(1), mean_acc=np.nanmean(self.accp, 1),
                        frac_walk=(np.where(fin, sp < 1.5, False).sum(1) / np.maximum(fin.sum(1), 1)).astype(np.float32),
                        spread=np.nanstd(px, 1) + np.nanstd(py, 1))
        d = np.hypot(px - self.bx[:, None], py - self.by[:, None]).astype(np.float32)
        self.d = d
        di = np.where(np.isfinite(d), d, np.float32(np.inf))
        two = np.partition(di, 1, axis=1)[:, :2]
        self.near = np.argmin(di, 1)
        self.dmin = _nanfix(two[:, 0])
        self.d2 = _nanfix(two[:, 1])
        del two
        self.dh = _nanfix(di[:, home].min(1))
        self.da = _nanfix(di[:, ~home].min(1))
        del di
        self.bsp = np.hypot(bvx, bvy)
        self.kacc, self.sa, self.sb, self.ang = _kick_signal(self.t, self.bx, self.by, 3)
        # ball forward-filled (candidate choice when the ball is missing)
        ok = np.isfinite(self.bx)
        li = np.maximum.accumulate(np.where(ok, np.arange(n), 0))
        self.bx_ff, self.by_ff = np.where(ok, self.bx, self.bx[li]), np.where(ok, self.by, self.by[li])
        self.real_present = real_present

    def at(self, v, frames, off_s):
        g = frames + int(round(off_s * self.fps))
        gg = np.clip(g, 0, self.n - 1)
        ok = (g >= 0) & (g < self.n) & (self.period[gg] == self.period[frames])
        return np.where(ok, v[gg], np.nan)

    def since(self, mask, frames, cap, future=False):
        """Seconds since (or until, future=True) the last frame where mask is True, capped."""
        n = self.n
        if future:
            idx = np.where(mask, np.arange(n), n)
            idx = np.minimum.accumulate(idx[::-1])[::-1]
            ok = idx < n
            dt = np.where(ok, self.t[np.minimum(idx, n - 1)] - self.t, cap)
        else:
            idx = np.maximum.accumulate(np.where(mask, np.arange(n), -1))
            ok = idx >= 0
            dt = np.where(ok, self.t - self.t[np.maximum(idx, 0)], cap)
        return np.minimum(dt, cap)[frames]


def base_features(P: Prepared, frames, gapf: dict | None = None):
    cols, names = [], []

    def add(nm, v):
        cols.append(np.asarray(v, np.float32)); names.append(nm)

    f = np.asarray(frames)
    s = P.s[f]
    bx, by = P.bx[f], P.by[f]
    add("ball_x", bx * s); add("ball_y", by); add("ball_z", P.bz[f])
    add("ball_out_x", np.abs(bx) - P.hl); add("ball_out_y", np.abs(by) - P.hw)
    add("ball_speed", P.bsp[f]); add("ball_kick_acc", P.kacc[f]); add("ball_speed_after", P.sa[f]); add("ball_speed_before", P.sb[f])
    add("ball_turn_deg", P.ang[f]); add("ball_flag", P.bflag[f])
    win = int(round(P.fps))
    pad = np.r_[np.full(win, np.nan), P.bsp]
    still = np.lib.stride_tricks.sliding_window_view(pad, win + 1)[f].max(1) if True else None
    add("ball_max_speed_1s", still)
    beyond = (np.abs(P.bx) > P.hl + 0.41) | (np.abs(P.by) > P.hw + 0.41)
    add("ball_beyond_line", beyond[f]); add("since_beyond_s", P.since(beyond, f, 30.0))
    add("since_seen_s", P.since(P.real_present, f, 60.0)); add("until_seen_s", P.since(P.real_present, f, 60.0, future=True))
    cx = P.hl - np.abs(bx); cy = P.hw - np.abs(by)
    add("dist_corner", np.hypot(cx, cy)); add("dist_centre", np.hypot(bx, by))
    add("dist_pen_spot", np.hypot(np.abs(np.abs(bx) - (P.hl - 11.0)), by))
    add("in_goal_area", (np.abs(bx) >= P.hl - 5.5) & (np.abs(by) <= 9.16))
    add("dmin", P.dmin[f]); add("d2", P.d2[f])
    nt = P.ts[P.near[f]]
    add("near_team", np.where(np.isfinite(P.dmin[f]), nt, 0))
    dn = np.where(nt > 0, P.dh[f], P.da[f]); do = np.where(nt > 0, P.da[f], P.dh[f])
    add("margin_opp", do - dn)
    dd = P.d[f]
    same = P.ts[None, :] == nt[:, None]
    add("n_same_2m", np.sum((dd <= 2) & same, 1)); add("n_opp_2m", np.sum((dd <= 2) & ~same, 1))
    add("n_opp_duel1m", np.sum((dd <= 1) & ~same, 1)); add("n_all_5m", np.sum(dd <= 5, 1))
    r = np.arange(len(f))
    nc = P.near[f]
    add("near_speed", P.sp[f, nc])
    ux, uy = (bx - P.px[f, nc]), (by - P.py[f, nc])
    nn = np.maximum(np.hypot(ux, uy), 1e-6)
    add("near_closing", (P.vxp[f, nc] * ux + P.vyp[f, nc] * uy) / nn)
    add("near_touchline", np.abs(P.py[f, nc]) - P.hw)
    for off in OFFS_S:
        for k in ("mean_sp", "top3_sp", "mean_acc", "frac_walk"):
            add(f"{k}@{off:+g}", P.at(P.act[k], f, off))
    for off in (-4.0, 0.0, 4.0):
        add(f"spread@{off:+g}", P.at(P.act["spread"], f, off))
    for off in CTX_S:
        add(f"ball_speed@{off:+g}", P.at(P.bsp, f, off)); add(f"dmin@{off:+g}", P.at(P.dmin, f, off))
    if gapf is not None:
        for k, v in gapf.items():
            add(k, v[f] if not k.endswith("_x") else v[f] * s)
    X = np.stack(cols, 1)
    # candidates: 5 players nearest the (forward-filled) ball
    dff = np.hypot(P.px[f] - P.bx_ff[f, None], P.py[f] - P.by_ff[f, None])
    dffi = np.where(np.isfinite(dff), dff, np.inf)
    cand = np.argsort(dffi, 1)[:, :N_CAND]
    cc, cn = [], []

    def addc(nm, v):
        cc.append(np.asarray(v, np.float32)); cn.append(nm)

    dmin_ff = dffi[r, cand[:, 0]]
    dsel = dffi[r[:, None], cand]
    addc("c_d", np.where(np.isfinite(dsel), dsel, np.nan)); addc("c_rank", np.broadcast_to(np.arange(N_CAND), cand.shape))
    addc("c_d_minus_min", np.where(np.isfinite(dsel), dsel - dmin_ff[:, None], np.nan))
    addc("c_speed", P.sp[f[:, None], cand])
    ux, uy = P.bx_ff[f, None] - P.px[f[:, None], cand], P.by_ff[f, None] - P.py[f[:, None], cand]
    nn = np.maximum(np.hypot(ux, uy), 1e-6)
    addc("c_closing", (P.vxp[f[:, None], cand] * ux + P.vyp[f[:, None], cand] * uy) / nn)
    for off in (-1.0, -0.4, 0.4, 1.0):
        g = np.clip(f + int(round(off * P.fps)), 0, P.n - 1)
        addc(f"c_d@{off:+g}", np.hypot(P.px[g[:, None], cand] - P.bx_ff[g, None], P.py[g[:, None], cand] - P.by_ff[g, None]))
    tsc = P.ts[cand]
    opp_min = np.stack([np.where(P.ts[None, :] != tsc[:, k:k + 1], dffi, np.inf).min(1) for k in range(N_CAND)], 1)
    addc("c_margin_opp", np.where(np.isfinite(opp_min - dsel), opp_min - dsel, np.nan))
    addc("c_team", tsc)
    addc("c_mates_nearer", np.stack([np.sum((P.ts[None, :] == tsc[:, k:k + 1]) & (dffi < dsel[:, k:k + 1]), 1)
                                     for k in range(N_CAND)], 1))
    for nm in ("ball_speed", "ball_z", "dmin"):
        addc(f"c_{nm}", np.broadcast_to(X[:, names.index(nm)][:, None], cand.shape))
    C = np.stack(cc, 2)
    return X, names, cand, C, cn


def c2_features(P: Prepared, frames, r, pid, cand):
    """C2-v2 decision block and the C2-dependent candidate flags. r: C2Result; pid: player id per column."""
    cols, names = [], []

    def add(nm, v):
        cols.append(np.asarray(v, np.float32)); names.append(nm)

    f = np.asarray(frames)
    n = P.n
    add("c2_state", r.state[f]); add("c2_conf", r.confidence[f])
    rv = np.array([RULE_VOCAB.index(x) if x in RULE_VOCAB else -1 for x in r.rule_names])
    add("c2_rule", rv[r.rule[f]])
    for off in (-1.0, 1.0):
        add(f"c2_state@{off:+g}", P.at(r.state.astype(float), f, off))
    w = int(round(2 * P.fps))
    cs = np.r_[0, np.cumsum(r.state == 0)]
    lo, hi = np.clip(f - w, 0, n), np.clip(f + w + 1, 0, n)
    add("c2_dead_frac_4s", (cs[hi] - cs[lo]) / np.maximum(hi - lo, 1))
    col_of = {int(p): c for c, p in enumerate(pid)}
    c2col = np.array([col_of.get(int(p), -1) for p in r.player[f]])
    add("c2_player_is_nearest", (c2col >= 0) & (c2col == P.near[f]))
    add("c2_team", np.where(c2col >= 0, P.ts[np.maximum(c2col, 0)], 0))
    tf = np.array([x["frame"] for x in r.touches], int)
    tcol = np.array([x["col"] for x in r.touches], int)
    if len(tf) == 0:
        tf, tcol = np.array([-10 ** 9]), np.array([0])
        tl = [dict(kind="turn", dist=np.nan, duel=False, keeper=False, z=0.0)]
    else:
        tl = r.touches
    li = np.searchsorted(tf, f, "right") - 1                  # last touch at or before
    ni = np.searchsorted(tf, f, "right")                      # next touch after
    has_l, has_n = li >= 0, ni < len(tf)
    lic, nic = np.maximum(li, 0), np.minimum(ni, len(tf) - 1)
    add("since_touch_s", np.where(has_l, np.minimum(P.t[f] - P.t[np.clip(tf[lic], 0, n - 1)], 10), 10))
    add("until_touch_s", np.where(has_n, np.minimum(P.t[np.clip(tf[nic], 0, n - 1)] - P.t[f], 10), 10))
    add("last_touch_kind", np.where(has_l, [KIND[tl[i]["kind"]] for i in lic], -1))
    add("last_touch_dist", np.where(has_l, [tl[i]["dist"] for i in lic], np.nan))
    add("last_touch_duel", np.where(has_l, [tl[i]["duel"] for i in lic], 0))
    add("last_touch_keeper", np.where(has_l, [tl[i]["keeper"] for i in lic], 0))
    add("last_touch_z", np.where(has_l, [tl[i]["z"] for i in lic], np.nan))
    lc = tcol[lic]
    add("release_dist", np.where(has_l, np.hypot(P.bx[f] - P.px[f, lc], P.by[f] - P.py[f, lc]), np.nan))
    tfc = np.clip(tf[lic], 0, n - 1)
    add("arrival_dist", np.where(has_l, np.hypot(P.bx[f] - P.bx[tfc], P.by[f] - P.by[tfc]), np.nan))
    add("last_actor_team", np.where(has_l, P.ts[lc], 0)); add("next_actor_team", np.where(has_n, P.ts[tcol[nic]], 0))
    add("last_actor_is_nearest", has_l & (lc == P.near[f])); add("next_actor_is_nearest", has_n & (tcol[nic] == P.near[f]))
    w1 = int(round(P.fps))
    add("touches_2s", np.searchsorted(tf, f + w1, "right") - np.searchsorted(tf, f - w1, "left"))
    rs = [x for x in r.restarts]
    rf = np.array([x[0] for x in rs], int) if rs else np.array([-10 ** 9])
    ri = np.searchsorted(rf, f, "right") - 1
    hr = ri >= 0
    ric = np.maximum(ri, 0)
    add("since_restart_s", np.where(hr, np.minimum(P.t[f] - P.t[np.clip(rf[ric], 0, n - 1)], 120), 120))
    add("restart_kind", np.where(hr, [RESTART_VOCAB.index(rs[i][1]) if rs and rs[i][1] in RESTART_VOCAB else -1 for i in ric], -1))
    add("restart_why", np.where(hr, [WHY_VOCAB.index(rs[i][2]) if rs and rs[i][2] in WHY_VOCAB else -1 for i in ric], -1))
    X = np.stack(cols, 1)
    # candidate flags
    cc = []
    cc.append((cand == c2col[:, None]) & (c2col[:, None] >= 0))
    cc.append(has_l[:, None] & (cand == lc[:, None]))
    cc.append(has_n[:, None] & (cand == tcol[nic][:, None]))
    own = np.full(cand.shape, 30.0, np.float32)               # seconds since this candidate's own last touch
    for c in np.unique(cand):
        ft = tf[tcol == c] if len(r.touches) else np.zeros(0, int)
        if len(ft) == 0:
            continue
        k = np.searchsorted(ft, f, "right") - 1
        val = np.where(k >= 0, np.minimum(P.t[f] - P.t[np.clip(ft[np.maximum(k, 0)], 0, n - 1)], 30), 30)
        m = cand == c
        own[m] = np.broadcast_to(val[:, None], cand.shape)[m]
    cc.append(own)
    C = np.stack([np.asarray(x, np.float32) for x in cc], 2)
    return X, names, C, ["c_is_c2_player", "c_is_last_actor", "c_is_next_actor", "c_since_own_touch"]


# ---------------------------------------------------------------- decoder

HMM_DEFAULT = dict(tol_s=0.3, pen_restart=-2.0, pen_dead=-6.0, pen_ctrl=-2.0, pen_leave=-3.0, pen_tl=-3.0, w_emit=1.0)
HMM_RANGES = dict(tol_s=(0.1, 1.0), pen_restart=(-12.0, 0.0), pen_dead=(-12.0, 0.0), pen_ctrl=(-12.0, 0.0),
                  pen_leave=(-12.0, 0.0), pen_tl=(-12.0, 0.0), w_emit=(0.3, 2.0))
NEG = -1e9


def _trans(prm, at_touch: bool):
    A = np.full((5, 5), NEG)
    np.fill_diagonal(A, 0.0)
    for a in (S_CH, S_CA, S_TRAV, S_LOOSE):
        A[a, S_DEAD] = prm["pen_dead"]                            # out of play / whistle: no touch needed
        if at_touch:
            A[S_DEAD, a] = prm["pen_restart"]                     # DEAD ends only at a restart kick (a touch)
    for c in (S_CH, S_CA):
        A[c, S_TRAV] = prm["pen_leave"]; A[c, S_LOOSE] = prm["pen_leave"]
        if at_touch:
            A[S_TRAV, c] = prm["pen_ctrl"]; A[S_LOOSE, c] = prm["pen_ctrl"]   # the owner changes only at a touch
    A[S_TRAV, S_LOOSE] = prm["pen_tl"]; A[S_LOOSE, S_TRAV] = prm["pen_tl"]
    return A                                                       # CONTROL-home <-> CONTROL-away stays forbidden


def emissions(p_dead, p_state, q_home):
    """log emissions (T, 5) from P(dead), P(CONTROL/TRAVELLING/LOOSE | alive) and P(home | CONTROL)."""
    eps = 1e-6
    pa = 1 - p_dead
    E = np.stack([p_dead, pa * p_state[:, 0] * q_home, pa * p_state[:, 0] * (1 - q_home), pa * p_state[:, 1], pa * p_state[:, 2]], 1)
    return np.log(np.clip(E, eps, 1))


def viterbi(logE, touch_ok, period, prm):
    w = prm["w_emit"]
    At, Ao = _trans(prm, True), _trans(prm, False)
    T = len(logE)
    path = np.zeros(T, np.int8)
    starts = np.r_[0, np.flatnonzero(period[1:] != period[:-1]) + 1, T]
    for a, b in zip(starts[:-1], starts[1:]):
        delta = w * logE[a] + np.r_[0.0, np.full(4, NEG)]      # A5: every period starts DEAD (before the kick-off)
        back = np.zeros((b - a, 5), np.int8)
        for i in range(a + 1, b):
            m = delta[:, None] + (At if touch_ok[i] else Ao)
            back[i - a] = m.argmax(0)
            delta = m.max(0) + w * logE[i]
        s = int(np.argmax(delta))
        for i in range(b - 1, a - 1, -1):
            path[i] = s
            s = back[i - a, s] if i > a else s
    return path


def touch_mask(touch_frames, rows, fps, tol_s):
    """Rows (frame indices) within tol_s of a detected touch."""
    tf = np.sort(np.asarray(touch_frames, int))
    if len(tf) == 0:
        return np.zeros(len(rows), bool)
    tol = int(round(tol_s * fps))
    k = np.searchsorted(tf, rows)
    lo = np.abs(rows - tf[np.clip(k - 1, 0, len(tf) - 1)]) <= tol
    hi = np.abs(tf[np.clip(k, 0, len(tf) - 1)] - rows) <= tol
    return lo | hi


def finish(path_rows, rows, n, period, touches, team_ids, team_sign, pid, cand_rows, cand_prob, cand_team_sign, fps, tol_s):
    """Rows -> frames: hold each row's state; back-date CONTROL starts to the touch (by a player of that team) within tol_s
    before; the owner = the candidate of the decoded team with the highest summed score between consecutive touches.
    Returns state, player, team per frame and passes (CONTROL -> TRAVELLING by the owner)."""
    idx = np.clip(np.searchsorted(rows, np.arange(n), "right") - 1, 0, len(rows) - 1)
    hs = path_rows[idx]
    state = np.select([hs == S_DEAD, (hs == S_CH) | (hs == S_CA), hs == S_TRAV], [DEAD, CONTROL, TRAVELLING], LOOSE).astype(np.int8)
    tsign = np.where(hs == S_CH, 1, np.where(hs == S_CA, -1, 0)).astype(np.int8)
    tf = np.array([x["frame"] for x in touches], int)
    tc = np.array([x["col"] for x in touches], int)
    # back-dating of CONTROL starts
    ctrl = state == CONTROL
    starts = np.flatnonzero(ctrl & ~np.r_[False, ctrl[:-1]])
    tol = int(round(tol_s * fps))
    for s0 in starts:
        k = np.searchsorted(tf, s0, "right") - 1
        while k >= 0 and s0 - tf[k] <= tol:
            if team_sign[tc[k]] == tsign[s0] and tf[k] < s0 and period[tf[k]] == period[s0] and state[tf[k]] != DEAD:
                state[tf[k]:s0], tsign[tf[k]:s0] = CONTROL, tsign[s0]
                break
            k -= 1
    # owner per CONTROL piece between touches
    player = np.full(n, -1, np.int64)
    team = np.full(n, -1, np.int64)
    ctrl = state == CONTROL
    cut = np.zeros(n, bool)
    cut[np.clip(tf, 0, n - 1)] = True
    seg_start = ctrl & (~np.r_[False, ctrl[:-1]] | cut | np.r_[False, tsign[1:] != tsign[:-1]])
    st_idx = np.flatnonzero(seg_start)
    ends = np.r_[st_idx[1:], n]
    for s0, e0 in zip(st_idx, ends):
        e1 = s0
        while e1 < e0 and ctrl[e1] and tsign[e1] == tsign[s0]:
            e1 += 1
        ra, rb = np.searchsorted(rows, s0, "left"), max(np.searchsorted(rows, e1, "left"), np.searchsorted(rows, s0, "left") + 1)
        ra = max(0, min(ra, len(rows) - 1) - (0 if rows[min(ra, len(rows) - 1)] <= s0 else 1))
        cr, pr, ts = cand_rows[ra:rb], cand_prob[ra:rb], cand_team_sign[ra:rb]
        score = {}
        for c_, p_, s_ in zip(cr.ravel(), pr.ravel(), ts.ravel()):
            if s_ == tsign[s0]:
                score[int(c_)] = score.get(int(c_), 0.0) + float(p_)
        if score:
            best = max(score, key=score.get)
            player[s0:e1] = pid[best]
            team[s0:e1] = team_ids[best]
    # TRAVELLING keeps the team of the last CONTROL; passes = CONTROL -> TRAVELLING
    passes = []
    last_team, last_player, last_p = -1, -1, None
    trav_start = np.flatnonzero((state == TRAVELLING) & np.r_[True, state[:-1] != TRAVELLING])
    i_ctrl = np.flatnonzero(ctrl)
    for s0 in trav_start:
        k = np.searchsorted(i_ctrl, s0) - 1
        e = s0
        while e < n and state[e] == TRAVELLING:
            e += 1
        if k >= 0 and period[i_ctrl[k]] == period[s0]:
            team[s0:e] = team[i_ctrl[k]]
            if i_ctrl[k] == s0 - 1 and player[s0 - 1] >= 0:
                nxt = player[e] if e < n and state[e] == CONTROL else -1
                passes.append((int(s0 - 1), int(player[s0 - 1]), int(nxt)))
    return state, player, team, passes
