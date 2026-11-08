"""M1 ball-from-players: features that describe where the ball probably is from the players' movement alone, plus the last
seen ball position and the time since it was seen. Tag-free, numpy only (the regressors are trained in scripts/m1/bfp.py).

Coordinates are turned so that the home team defends the negative-x goal in every period (``orient``); the target is the
ball's displacement from where it was last seen, in those coordinates.
"""
from __future__ import annotations

import numpy as np

V_HALF = 3            # frames either side for player velocities
MAX_RUN = 12.0        # m/s; faster = tracker jump


def orient(period, px, team_sign) -> np.ndarray:
    """Per-frame sign s so that s * x puts the home team (team_sign +1) in the negative-x half (from its median x per period)."""
    s = np.ones(len(period), np.float32)
    home = team_sign > 0
    for per in np.unique(period):
        sel = period == per
        with np.errstate(all="ignore"):
            med = np.nanmedian(px[sel][:, home])
        s[sel] = -1.0 if np.isfinite(med) and med > 0 else 1.0
    return s


def last_seen(t, period, present) -> tuple[np.ndarray, np.ndarray]:
    """Index of the last frame with the ball present at or before each frame in the same period (-1 if none) and seconds since."""
    n = len(t)
    idx = np.where(present, np.arange(n), -1)
    idx = np.maximum.accumulate(idx)
    starts = np.r_[0, np.flatnonzero(period[1:] != period[:-1]) + 1]
    for s in starts:                                          # do not look across a period boundary
        bad = (np.arange(n) >= s) & (idx < s)
        idx[bad] = -1
    h = np.where(idx >= 0, t - t[np.maximum(idx, 0)], np.nan)
    return idx, h


NAMES: list[str] = []


def features(t, period, px, py, team_sign, s, frames, last, lx, ly, lvx, lvy) -> np.ndarray:
    """Feature matrix (len(frames), F). frames: frame indices; last: last-seen frame index per row (-1 = never seen);
    lx, ly, lvx, lvy: last seen ball position / velocity per row (raw coordinates)."""
    n = len(t)
    f = np.asarray(frames)
    a, b = np.clip(f - V_HALF, 0, n - 1), np.clip(f + V_HALF, 0, n - 1)
    sg = s[f][:, None]
    X, Y = px[f] * sg, py[f]
    dt = np.maximum(t[b] - t[a], 1e-3)[:, None]
    VX = (px[b] - px[a]) * sg / dt
    VY = (py[b] - py[a]) / dt
    sp = np.hypot(VX, VY)
    bad = ~(sp <= MAX_RUN)
    VX, VY, sp = (np.where(bad, np.nan, v) for v in (VX, VY, sp))
    h = np.where(last >= 0, t[f] - t[np.maximum(last, 0)], 60.0)
    LX, LY = lx * s[f], ly
    LVX, LVY = lvx * s[f], lvy
    ex, ey = LX + np.nan_to_num(LVX) * np.minimum(h, 1.0), LY + np.nan_to_num(LVY) * np.minimum(h, 1.0)
    cols, names = [], []

    def add(name, v):
        cols.append(np.asarray(v, np.float32))
        names.append(name)

    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for nm, v in (("h", h), ("last_x", LX), ("last_y", LY), ("last_vx", LVX), ("last_vy", LVY), ("extrap_x", ex),
                          ("extrap_y", ey), ("last_speed", np.hypot(LVX, LVY))):
                add(nm, v)
            for tm, tn in ((1, "home"), (-1, "away")):
                c = team_sign == tm
                for nm, v in (("cx", np.nanmean(X[:, c], 1)), ("cy", np.nanmean(Y[:, c], 1)), ("cvx", np.nanmean(VX[:, c], 1)),
                              ("cvy", np.nanmean(VY[:, c], 1)), ("sdx", np.nanstd(X[:, c], 1)), ("sdy", np.nanstd(Y[:, c], 1)),
                              ("minx", np.nanmin(X[:, c], 1)), ("maxx", np.nanmax(X[:, c], 1)),
                              ("speed", np.nanmean(sp[:, c], 1))):
                    add(f"{tn}_{nm}", v)
            w = np.nan_to_num(sp)
            add("speed_w_x", np.nansum(np.nan_to_num(X) * w, 1) / np.maximum(w.sum(1), 1e-6))
            add("speed_w_y", np.nansum(np.nan_to_num(Y) * w, 1) / np.maximum(w.sum(1), 1e-6))
            run = sp > 2.0
            add("runto_x", np.nanmean(np.where(run, X + VX, np.nan), 1))
            add("runto_y", np.nanmean(np.where(run, Y + VY, np.nan), 1))
            # convergence point of the running directions (least squares over the lines through each runner)
            ux, uy = np.where(run, VX / np.maximum(sp, 1e-6), 0), np.where(run, VY / np.maximum(sp, 1e-6), 0)
            ww = np.where(run, sp, 0)
            a11, a12, a22 = (ww * (1 - ux * ux)).sum(1), (ww * (-ux * uy)).sum(1), (ww * (1 - uy * uy)).sum(1)
            xx, yy = np.nan_to_num(X), np.nan_to_num(Y)
            b1 = (ww * ((1 - ux * ux) * xx - ux * uy * yy)).sum(1)
            b2 = (ww * (-ux * uy * xx + (1 - uy * uy) * yy)).sum(1)
            det = a11 * a22 - a12 * a12
            ok = det > 1e-3
            add("conv_x", np.where(ok, np.clip((a22 * b1 - a12 * b2) / np.where(ok, det, 1), -60, 60), np.nan))
            add("conv_y", np.where(ok, np.clip((a11 * b2 - a12 * b1) / np.where(ok, det, 1), -40, 40), np.nan))
            # players nearest the extrapolated last-seen point
            dd = np.hypot(X - ex[:, None], Y - ey[:, None])
            dd = np.where(np.isfinite(dd), dd, np.inf)
            order = np.argsort(dd, 1)[:, :4]
            r = np.arange(len(f))[:, None]
            for k in range(4):
                o = order[:, k]
                rr = r[:, 0]
                add(f"near{k}_dx", X[rr, o] - ex)
                add(f"near{k}_dy", Y[rr, o] - ey)
                add(f"near{k}_vx", VX[rr, o])
                add(f"near{k}_vy", VY[rr, o])
                add(f"near{k}_team", team_sign[o])
                add(f"near{k}_d", np.where(np.isfinite(dd[rr, o]), dd[rr, o], np.nan))
            # fastest players
            fo = np.argsort(-np.nan_to_num(sp, nan=-1), 1)[:, :3]
            for k in range(3):
                o = fo[:, k]
                rr = r[:, 0]
                add(f"fast{k}_x", X[rr, o]); add(f"fast{k}_y", Y[rr, o])
                add(f"fast{k}_vx", VX[rr, o]); add(f"fast{k}_vy", VY[rr, o]); add(f"fast{k}_team", team_sign[o])
            # crowding: the player with most others within 6 m
            pd = np.hypot(X[:, :, None] - X[:, None, :], Y[:, :, None] - Y[:, None, :])
            cnt = np.nansum(pd <= 6.0, 2) - 1
            cnt = np.where(np.isfinite(X), cnt, -1)
            ci = np.argmax(cnt, 1)
            add("crowd_x", X[r[:, 0], ci]); add("crowd_y", Y[r[:, 0], ci]); add("crowd_n", cnt[r[:, 0], ci])
            # the player nearest the ball when it was last seen, where they are now
            lf = np.maximum(last, 0)
            dl = np.hypot(px[lf] - lx[:, None], py[lf] - ly[:, None])
            dl = np.where(np.isfinite(dl), dl, np.inf)
            hi = np.argmin(dl, 1)
            okh = (last >= 0) & np.isfinite(dl[r[:, 0], hi])
            add("holder_x", np.where(okh, X[r[:, 0], hi], np.nan)); add("holder_y", np.where(okh, Y[r[:, 0], hi], np.nan))
            add("holder_vx", np.where(okh, VX[r[:, 0], hi], np.nan)); add("holder_vy", np.where(okh, VY[r[:, 0], hi], np.nan))
            add("holder_team", np.where(okh, team_sign[hi], 0))
    NAMES[:] = names
    return np.stack(cols, 1)
