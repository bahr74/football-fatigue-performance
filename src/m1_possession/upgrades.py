"""M1 input upgrades (each switchable in the pipeline; tag-free, numpy only).

- ``smooth_players``: Savitzky-Golay smoothing of raw player positions, run by run (no bridging of missing samples). A local
  quadratic fit keeps the positions where they are and removes the frame-to-frame jitter that spoils speeds and accelerations.
- ``fill_medium_gaps``: a physics path through ball gaps of 0.5-2 s between where the ball vanished and where it reappeared:
  a ballistic curve when it was in the air, a decelerating roll on the ground. Frames are flagged FLAG_GUESSED.
- ``gap_features``: for long gaps no path is invented; each frame of a gap gets its length, the vanish / reappear positions,
  their distance and the team nearest the ball at each end.
"""
from __future__ import annotations

import numpy as np

from .smoother import FLAG_MISSING

FLAG_GUESSED, FLAG_INFERRED = 4, 5        # extend FLAG_* of the smoother: physics gap path / ball-from-players estimate
G = 9.81


def _runs(mask, period):
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    br = np.flatnonzero((np.diff(idx) > 1) | (period[idx[1:]] != period[idx[:-1]]))
    return list(zip(np.r_[idx[0], idx[br + 1]].tolist(), np.r_[idx[br], idx[-1]].tolist()))


def _savgol_coeffs(w: int, order: int) -> np.ndarray:
    h = w // 2
    a = np.vander(np.arange(-h, h + 1, dtype=float), order + 1, increasing=True)
    return np.linalg.pinv(a)[0]                       # value of the fitted polynomial at the window centre


def smooth_players(px: np.ndarray, py: np.ndarray, period: np.ndarray, window: int = 15, order: int = 2):
    """Smooth each player's finite runs with a centred Savitzky-Golay filter (window frames, odd); run ends and runs shorter
    than the window keep their raw values."""
    c = _savgol_coeffs(window, order)
    h = window // 2
    out = []
    for arr in (px, py):
        sm = arr.astype(np.float32).copy()
        for col in range(arr.shape[1]):
            v = arr[:, col]
            for a, b in _runs(np.isfinite(v), period):
                if b - a + 1 < window:
                    continue
                seg = v[a:b + 1].astype(float)
                sm[a + h:b + 1 - h, col] = np.convolve(seg, c[::-1], mode="valid")
        out.append(sm)
    return out[0], out[1]


def fill_medium_gaps(t, period, x, y, z, vx, vy, flag, min_s: float = 0.5, max_s: float = 2.0, air_z: float = 0.5):
    """Physics path through missing-ball gaps of min_s-max_s seconds with the ball seen on both sides in the same period.
    Air (ball higher than air_z at either end): straight line in x-y, height on a parabola under gravity through both ends.
    Ground: the ball rolls along the straight line slowing down at a constant rate (its speed at the vanish point, ending
    exactly at the reappear point; a linear path when that is impossible). Returns new arrays and the number of frames."""
    x, y, z, vx, vy, flag = (np.array(a, dtype=float if a is not flag else np.int8) for a in (x, y, z, vx, vy, flag))
    z = np.nan_to_num(z)
    n_fill = 0
    for i, j in _runs(flag == FLAG_MISSING, period):
        a, b = i - 1, j + 1
        if a < 0 or b >= len(t) or period[a] != period[i] or period[b] != period[i]:
            continue
        T = t[b] - t[a]
        if not (min_s <= t[j] - t[i] + (t[1] - t[0]) <= max_s) or not np.isfinite(x[a] + x[b]):
            continue
        tau = t[i:j + 1] - t[a]
        dx, dy = x[b] - x[a], y[b] - y[a]
        D = np.hypot(dx, dy)
        if max(z[a], z[b]) > air_z:
            frac = tau / T
            z[i:j + 1] = z[a] + (z[b] - z[a]) * frac + 0.5 * G * tau * (T - tau)
        else:
            v0 = np.hypot(np.nan_to_num(vx[a]), np.nan_to_num(vy[a]))
            dec = 2 * (v0 * T - D) / T ** 2
            if D > 1e-6 and dec >= 0 and v0 - dec * T >= 0:
                frac = (v0 * tau - 0.5 * dec * tau ** 2) / D
            elif D > 1e-6 and dec >= 0:                       # would stop before the end: slow to rest at the reappear point
                frac = 1 - (1 - tau / T) ** 2
            else:
                frac = tau / T
            z[i:j + 1] = 0.0
        x[i:j + 1] = x[a] + dx * frac
        y[i:j + 1] = y[a] + dy * frac
        vx[i:j + 1] = np.gradient(x[a:b + 1], t[a:b + 1])[1:-1]
        vy[i:j + 1] = np.gradient(y[a:b + 1], t[a:b + 1])[1:-1]
        flag[i:j + 1] = FLAG_GUESSED
        n_fill += j - i + 1
    return x, y, z, vx, vy, flag, n_fill


GAP_FEATURES = ("gap_in", "gap_len_s", "gap_since_s", "gap_to_end_s", "gap_vanish_x", "gap_vanish_y", "gap_reappear_x",
                "gap_reappear_y", "gap_dist_m", "gap_team_vanish", "gap_team_reappear", "gap_same_team")


def gap_features(t, period, missing, x, y, px, py, team_sign) -> dict:
    """Per-frame description of the missing-ball gap a frame belongs to (zeros outside gaps; NaN where an end is unseen).
    team_sign: +1 / -1 per player column; the nearest team at an end is that sign."""
    n = len(t)
    f = {k: np.zeros(n, np.float32) for k in GAP_FEATURES}
    for k in GAP_FEATURES[4:]:
        f[k][:] = np.nan
    for i, j in _runs(missing, period):
        a, b = i - 1, j + 1
        sl = slice(i, j + 1)
        f["gap_in"][sl] = 1
        f["gap_len_s"][sl] = t[j] - t[i]
        f["gap_since_s"][sl] = t[sl] - t[i]
        f["gap_to_end_s"][sl] = t[j] - t[sl]
        ends = {}
        for name, e in (("vanish", a), ("reappear", b)):
            if 0 <= e < n and period[e] == period[i] and np.isfinite(x[e]):
                f[f"gap_{name}_x"][sl], f[f"gap_{name}_y"][sl] = x[e], y[e]
                d = np.hypot(px[e] - x[e], py[e] - y[e])
                if np.isfinite(d).any():
                    ends[name] = team_sign[int(np.nanargmin(d))]
                    f[f"gap_team_{name}"][sl] = ends[name]
        if np.isfinite(f["gap_vanish_x"][i]) and np.isfinite(f["gap_reappear_x"][i]):
            f["gap_dist_m"][sl] = np.hypot(f["gap_reappear_x"][i] - f["gap_vanish_x"][i], f["gap_reappear_y"][i] - f["gap_vanish_y"][i])
        if len(ends) == 2:
            f["gap_same_team"][sl] = float(ends["vanish"] == ends["reappear"])
    return f
