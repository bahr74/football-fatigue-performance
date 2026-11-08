"""Clean the raw tracking ball: drop repeats and impossible jumps, fill short gaps, smooth, estimate velocity.

Input is per-frame arrays of the RAW ball only (time, x, y, z, optional visibility and period). The module reads no
event tags, no ``game_event`` / ``possession_event`` field and no ``*Smoothed`` field; it has no access to them.

Steps (all parameters in :class:`SmootherConfig`):

a. repeats: within a run of identical consecutive positions only every ``repeat_keep_every``-th frame is kept as a
   real update (the raw ball is held for two frames, about 15 Hz). A resting ball therefore keeps its samples.
b. jumps: a kept sample whose implied speed into and out of it is above ``max_speed_mps`` while its neighbours agree
   with each other is a one-sample spike and is dropped (repeated ``spike_passes`` times). A remaining impossible
   jump splits the track; a single sample left between two such jumps is dropped.
c. gaps: consecutive kept samples further apart than ``max_gap_s`` (or in different periods) are not joined; frames
   between them stay missing. Shorter gaps are filled by the method below.
d. smoothing and velocity, per segment: ``linear`` (interpolation and central differences), ``savgol``
   (Savitzky-Golay on the frame grid after interpolation), ``rts`` (constant-velocity Kalman filter with a
   Rauch-Tung-Striebel smoother, as in Link & Hoernig 2017), ``rts_adaptive`` (rts that multiplies the process noise
   for one step when the x/y innovation exceeds a chi-square threshold, and adds velocity variance at kicks found in
   the track itself, so kicks are not spread out; the kick detector uses only the kept raw samples).

Flags per frame: ``observed`` (a kept raw update at that frame), ``estimated`` (a kept raw update whose source visibility
is ESTIMATED; only when ``visibility`` is given), ``interpolated`` (inside a segment without a kept update: a dropped
repeat, spike or short gap), ``missing``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import SmootherConfig

FLAG_MISSING, FLAG_OBSERVED, FLAG_INTERPOLATED, FLAG_ESTIMATED = 0, 1, 2, 3
FLAG_NAMES = {FLAG_MISSING: "missing", FLAG_OBSERVED: "observed", FLAG_INTERPOLATED: "interpolated", FLAG_ESTIMATED: "estimated"}


@dataclass
class SmoothedBall:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    vz: np.ndarray
    speed: np.ndarray
    flag: np.ndarray          # FLAG_MISSING / FLAG_OBSERVED / FLAG_ESTIMATED / FLAG_INTERPOLATED
    counts: dict = field(default_factory=dict)


def smooth_ball(t, x, y, z=None, visibility=None, period=None, cfg: SmootherConfig | None = None) -> SmoothedBall:
    """Smooth one match (or one period) of raw ball samples given per frame; missing samples are NaN."""
    cfg = cfg or SmootherConfig()
    t = np.asarray(t, float)
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(t)
    z = np.full(n, np.nan) if z is None else np.asarray(z, float)
    per = np.zeros(n, int) if period is None else np.asarray(period)
    if not (len(x) == len(y) == len(z) == len(per) == n):
        raise ValueError("t, x, y, z, period must have the same length")
    if n > 1 and np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing")
    counts: dict = dict(frames=n)

    valid = np.isfinite(x) & np.isfinite(y)
    counts["input_samples"] = int(valid.sum())
    est = np.zeros(n, bool)
    if visibility is not None:
        est = np.asarray([v == "ESTIMATED" for v in visibility], bool)
        if len(est) != n:
            raise ValueError("visibility must have the same length as t")
    if cfg.drop_estimated:
        counts["estimated_dropped"] = int((valid & est).sum())
        valid &= ~est

    keep = _drop_repeats(t, x, y, z, per, valid, cfg, counts)
    held = valid & ~keep
    idx = np.flatnonzero(keep)
    idx = _drop_spikes(t, x, y, per, idx, cfg, counts)
    segments = _segments(t, x, y, per, idx, cfg, counts)

    out = {k: np.full(n, np.nan) for k in ("x", "y", "z", "vx", "vy", "vz")}
    flag = np.zeros(n, np.int8)
    filled = 0
    for seg in segments:
        i0, i1 = seg[0], seg[-1]
        while i1 + 1 < n and held[i1 + 1] and per[i1 + 1] == per[i1]:   # held repeats after the last update
            i1 += 1
        fr = np.arange(i0, i1 + 1)
        flag[fr] = FLAG_INTERPOLATED
        flag[seg] = np.where(est[seg], FLAG_ESTIMATED, FLAG_OBSERVED)
        filled += len(fr) - len(seg)
        if len(seg) == 1:
            out["x"][i0], out["y"][i0], out["z"][i0] = x[i0], y[i0], z[i0]
            continue
        res = _smooth_segment(t[fr], seg - i0, x[seg], y[seg], z[seg], cfg)
        for k, v in res.items():
            out[k][fr] = v
    counts["segments"] = len(segments)
    counts["frames_filled"] = int(filled)
    counts["frames_observed"] = int((flag == FLAG_OBSERVED).sum())
    counts["frames_estimated"] = int((flag == FLAG_ESTIMATED).sum())
    counts["frames_missing"] = int((flag == FLAG_MISSING).sum())
    vz = np.where(np.isfinite(out["vz"]), out["vz"], 0.0)
    speed = np.sqrt(out["vx"] ** 2 + out["vy"] ** 2 + vz ** 2)
    return SmoothedBall(out["x"], out["y"], out["z"], out["vx"], out["vy"], out["vz"], speed, flag, counts)


# ---------------------------------------------------------------- a. repeats

def _drop_repeats(t, x, y, z, per, valid, cfg, counts) -> np.ndarray:
    n = len(t)
    dt = np.diff(t)
    step = np.median(dt) if n > 1 else 0.0
    same = np.zeros(n, bool)
    if n > 1:
        zeq = (z[1:] == z[:-1]) | (np.isnan(z[1:]) & np.isnan(z[:-1]))
        same[1:] = (valid[1:] & valid[:-1] & (x[1:] == x[:-1]) & (y[1:] == y[:-1]) & zeq & (per[1:] == per[:-1])
                    & (dt < 1.5 * step))
    run = np.zeros(n, int)
    for i in range(1, n):          # position within a run of identical consecutive samples
        if same[i]:
            run[i] = run[i - 1] + 1
    keep = valid & (run % cfg.repeat_keep_every == 0)
    counts["repeats_dropped"] = int((valid & ~keep).sum())
    return keep


# ---------------------------------------------------------------- b. jumps

def _speed(t, x, y, a, b) -> np.ndarray:
    return np.hypot(x[b] - x[a], y[b] - y[a]) / (t[b] - t[a])


def _drop_spikes(t, x, y, per, idx, cfg, counts) -> np.ndarray:
    spikes = 0
    for _ in range(cfg.spike_passes):
        if len(idx) < 3:
            break
        a, m, b = idx[:-2], idx[1:-1], idx[2:]
        same_p = (per[a] == per[m]) & (per[m] == per[b])
        bad = same_p & (_speed(t, x, y, a, m) > cfg.max_speed_mps) & (_speed(t, x, y, m, b) > cfg.max_speed_mps) \
            & (_speed(t, x, y, a, b) <= cfg.max_speed_mps)
        if not bad.any():
            break
        spikes += int(bad.sum())
        drop = set(m[bad].tolist())
        idx = np.array([i for i in idx if i not in drop], dtype=int)
    counts["spikes_dropped"] = spikes
    return idx


def _segments(t, x, y, per, idx, cfg, counts) -> list[np.ndarray]:
    if len(idx) == 0:
        counts.update(jump_breaks=0, gap_breaks=0, isolated_dropped=0)
        return []
    a, b = idx[:-1], idx[1:]
    jump = (per[a] == per[b]) & (_speed(t, x, y, a, b) > cfg.max_speed_mps) if len(idx) > 1 else np.zeros(0, bool)
    gap = (per[a] != per[b]) | (t[b] - t[a] > cfg.max_gap_s + 1e-9) if len(idx) > 1 else np.zeros(0, bool)
    cut = np.flatnonzero(jump | gap) + 1
    pieces = np.split(idx, cut)
    starts_after_jump = np.zeros(len(pieces), bool)
    ends_before_jump = np.zeros(len(pieces), bool)
    for k, c in enumerate(cut):
        if jump[c - 1]:
            ends_before_jump[k] = True
            starts_after_jump[k + 1] = True
    segs, isolated = [], 0
    for k, p in enumerate(pieces):
        if len(p) == 1 and (starts_after_jump[k] or ends_before_jump[k]):
            isolated += 1          # a lone sample cut off by an impossible jump is not trusted
            continue
        segs.append(p)
    counts.update(jump_breaks=int(jump.sum()), gap_breaks=int(gap.sum()), isolated_dropped=isolated)
    return segs


# ---------------------------------------------------------------- d. smoothing per segment

def _interp(tf, ts, v) -> np.ndarray:
    ok = np.isfinite(v)
    if ok.sum() == 0:
        return np.full(len(tf), np.nan)
    if ok.sum() == 1:
        return np.full(len(tf), v[ok][0])
    return np.interp(tf, ts[ok], v[ok])


def _smooth_segment(tf, pos, mx, my, mz, cfg) -> dict:
    """tf: frame times of the segment; pos: frame offsets of the kept samples; m*: their raw values."""
    ts = tf[pos]
    if cfg.method in ("linear", "savgol"):
        px, py, pz = (_interp(tf, ts, m) for m in (mx, my, mz))
        if cfg.method == "savgol":
            return _savgol(tf, px, py, pz, cfg)
        g = (lambda p: np.gradient(p, tf)) if len(tf) > 1 else (lambda p: np.full(len(tf), np.nan))
        return dict(x=px, y=py, z=pz, vx=g(px), vy=g(py), vz=g(pz) if np.isfinite(pz).all() else np.full(len(tf), np.nan))
    adaptive = cfg.method == "rts_adaptive"
    kicks = _kick_frames(tf, _interp(tf, ts, mx), _interp(tf, ts, my), cfg) if adaptive else None
    return _rts(tf, pos, mx, my, mz, cfg, adaptive=adaptive, kicks=kicks)


def _kick_frames(tf, px, py, cfg) -> np.ndarray:
    """Local maxima (+-kick_frames) of the change between mean velocities after and before a frame, >= kick_dv_mps."""
    n, k = len(tf), cfg.kick_frames
    out = np.zeros(n, bool)
    if n < 2 * k + 1:
        return out
    i = np.arange(k, n - k)
    va = np.stack([px[i + k] - px[i], py[i + k] - py[i]]) / (tf[i + k] - tf[i])
    vb = np.stack([px[i] - px[i - k], py[i] - py[i - k]]) / (tf[i] - tf[i - k])
    dv = np.full(n, -1.0)
    dv[i] = np.hypot(*(va - vb))
    w = np.lib.stride_tricks.sliding_window_view(np.pad(dv, k, constant_values=-1.0), 2 * k + 1).max(axis=1)
    out[:] = (dv >= cfg.kick_dv_mps) & (dv == w)
    return out


def _sg_matrices(w: int, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows: least-squares polynomial (degree p) fitted on w points, evaluated at each of the w offsets.

    Returns (S, D): S[j] are the smoothing weights for output offset j, D[j] the first-derivative weights (per frame).
    """
    k = np.arange(w, dtype=float)
    A = np.vander(k, p + 1, increasing=True)          # columns k^0..k^p
    pinv = np.linalg.pinv(A)                          # coefficients = pinv @ y
    E = np.vander(k, p + 1, increasing=True)
    dE = np.zeros_like(E)
    for j in range(1, p + 1):
        dE[:, j] = j * k ** (j - 1)
    return E @ pinv, dE @ pinv


def _savgol(tf, px, py, pz, cfg) -> dict:
    n = len(tf)
    dt = float(np.median(np.diff(tf))) if n > 1 else 1.0
    w = min(cfg.savgol_window, n if n % 2 else n - 1)
    if w < 3:
        g = lambda p: np.gradient(p, tf) if n > 1 else np.full(n, np.nan)  # noqa: E731
        return dict(x=px, y=py, z=pz, vx=g(px), vy=g(py), vz=g(pz))
    p = min(cfg.savgol_poly, w - 1)
    S, D = _sg_matrices(w, p)
    h = w // 2
    out = {}
    for name, v in (("x", px), ("y", py), ("z", pz)):
        if not np.isfinite(v).all():
            out[name], out["v" + name] = v, np.full(n, np.nan)
            continue
        win = np.lib.stride_tricks.sliding_window_view(v, w)       # (n - w + 1, w)
        s, d = np.empty(n), np.empty(n)
        s[h:n - h], d[h:n - h] = win @ S[h], win @ D[h]
        s[:h], d[:h] = S[:h] @ v[:w], D[:h] @ v[:w]                 # edges: same fit, evaluated off-centre
        s[n - h:], d[n - h:] = S[w - h:] @ v[n - w:], D[w - h:] @ v[n - w:]
        out[name], out["v" + name] = s, d / dt
    return out


def _rts(tf, pos, mx, my, mz, cfg, adaptive: bool, kicks: np.ndarray | None = None) -> dict:
    n = len(tf)
    meas = np.full((n, 3), np.nan)
    meas[pos, 0], meas[pos, 1], meas[pos, 2] = mx, my, mz
    r2 = cfg.rts_r_m ** 2
    q = cfg.rts_q
    # per axis state (p, v) and covariance (a=Ppp, b=Ppv, c=Pvv); arrays (n, 3)
    mp_f, mv_f = np.zeros((n, 3)), np.zeros((n, 3))
    pa_f, pb_f, pc_f = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 3))
    mp_p, mv_p = np.zeros((n, 3)), np.zeros((n, 3))
    pa_p, pb_p, pc_p = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 3))
    first = meas[0]
    mp = np.where(np.isfinite(first), first, 0.0)
    if not np.isfinite(first[2]):                      # z unknown until measured
        zf = mz[np.isfinite(mz)]
        mp[2] = zf[0] if len(zf) else 0.0
    mv = np.zeros(3)
    a, b, c = np.full(3, r2), np.zeros(3), np.full(3, 100.0)
    for i in range(n):
        if i > 0:
            dt = tf[i] - tf[i - 1]
            q_eff = q

            def predict(qq):
                pp = mp + mv * dt
                aa = a + 2 * dt * b + dt * dt * c + qq * dt ** 3 / 3
                bb = b + dt * c + qq * dt ** 2 / 2
                cc = c + qq * dt
                return pp, aa, bb, cc
            pp, aa, bb, cc = predict(q_eff)
            if kicks is not None and kicks[i]:
                cc = cc + cfg.kick_var                 # the velocity may change abruptly at a detected kick
            m = meas[i]
            if adaptive and np.isfinite(m[0]):
                nis = (m[0] - pp[0]) ** 2 / (aa[0] + r2) + (m[1] - pp[1]) ** 2 / (aa[1] + r2)
                if nis > cfg.rts_adapt_nis:
                    pp, aa, bb, cc = predict(q * cfg.rts_adapt_factor)
            mv = mv.copy()
            mp, a, b, c = pp, aa, bb, cc
        mp_p[i], mv_p[i], pa_p[i], pb_p[i], pc_p[i] = mp, mv, a, b, c
        m = meas[i]
        ok = np.isfinite(m)
        if ok.any():
            s = a + r2
            k1, k2 = a / s, b / s
            innov = np.where(ok, m - mp, 0.0)
            mp = mp + np.where(ok, k1 * innov, 0.0)
            mv = mv + np.where(ok, k2 * innov, 0.0)
            na, nb, nc = (1 - k1) * a, (1 - k1) * b, c - k2 * b
            a, b, c = np.where(ok, na, a), np.where(ok, nb, b), np.where(ok, nc, c)
        mp_f[i], mv_f[i], pa_f[i], pb_f[i], pc_f[i] = mp, mv, a, b, c
    sp, sv = mp_f.copy(), mv_f.copy()
    for i in range(n - 2, -1, -1):                    # Rauch-Tung-Striebel backward pass (means only)
        dt = tf[i + 1] - tf[i]
        # C = P_f F^T inv(P_pred[i+1]),  F = [[1, dt], [0, 1]]
        fa = pa_f[i] + dt * pb_f[i]                   # (P_f F^T)[0,0]
        fb = pb_f[i]                                  # (P_f F^T)[0,1]
        fc = pb_f[i] + dt * pc_f[i]                   # (P_f F^T)[1,0]
        fd = pc_f[i]                                  # (P_f F^T)[1,1]
        A, B, Cc = pa_p[i + 1], pb_p[i + 1], pc_p[i + 1]
        det = A * Cc - B * B
        ia, ib, ic = Cc / det, -B / det, A / det      # inverse of [[A, B], [B, Cc]]
        c00, c01 = fa * ia + fb * ib, fa * ib + fb * ic
        c10, c11 = fc * ia + fd * ib, fc * ib + fd * ic
        dp, dv = sp[i + 1] - mp_p[i + 1], sv[i + 1] - mv_p[i + 1]
        sp[i] = mp_f[i] + c00 * dp + c01 * dv
        sv[i] = mv_f[i] + c10 * dp + c11 * dv
    if not np.isfinite(mz).any():
        sp[:, 2], sv[:, 2] = np.nan, np.nan
    return dict(x=sp[:, 0], y=sp[:, 1], z=sp[:, 2], vx=sv[:, 0], vy=sv[:, 1], vz=sv[:, 2])
