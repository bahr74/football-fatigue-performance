"""Opt-in trajectory stabilisation for observed pitch positions (M7.3).

M7.3 A1 found that the M7.2 artefacts come from (1) runs cut at every noisy raw step, leaving 2-4 sample runs that are never smoothed,
(2) the nearest-grid mapping inverse and (3) a projection that turns one pixel of foot-point noise into up to 0.5 m. The stabilisers here
therefore:

* split a trajectory only where it must be split: an unusable sample, a gap longer than ``max_gap_s``, non-increasing time, or a
  *persistent* position jump (the medians before and after disagree by more than a plausible movement); an isolated noisy sample does not
  split the run, it is down-weighted by the filter;
* never use a sample that is not observed; a gap inside a run (at most ``max_gap_s``) is only bridged by the filter's model, and the
  output exists at observed samples only;
* drop runs shorter than ``min_run_s`` (too short to tell motion from noise); where the stabilised speed still exceeds the plausible
  maximum, cut out that stretch (+-0.2 s) and re-stabilise the pieces (a run still implausible after two cuts is dropped); all counted.

Methods (all deterministic, numpy / scipy only):

* ``savgol``        Savitzky-Golay (quadratic) on a uniform grid inside the run; velocity is the analytic derivative.
* ``hampel_savgol`` Hampel outlier replacement (median +- 3 scaled MAD) followed by ``savgol``.
* ``robust_poly``   local quadratic regression, tricube kernel, two bisquare robustness iterations (LOWESS-style).
* ``kalman_rts``    constant-velocity Kalman filter + Rauch-Tung-Striebel smoother, per-sample measurement variance (confidence weighting,
                    e.g. from the projection Jacobian) and innovation gating (outlier-resistant).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

KMH = 3.6
STABILIZERS = ("savgol", "hampel_savgol", "robust_poly", "kalman_rts")


@dataclass(frozen=True)
class MotionConfig:
    method: str = "kalman_rts"
    window_s: float = 0.6            # savgol / robust_poly / hampel window
    accel_noise: float = 9.0         # kalman: white-noise acceleration spectral density (m^2/s^3)
    meas_sigma_m: float = 0.25       # measurement sigma when no per-sample variance is given (m)
    gate_sigma: float = 3.0          # kalman: innovations beyond this many sigma are down-weighted
    min_run_s: float = 0.8           # shorter runs cannot separate motion from noise: not analysed
    jump_window: int = 5             # samples per side for the persistent-jump test

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------------------------------------------------------- runs
def split_runs(t: np.ndarray, x: np.ndarray, y: np.ndarray, usable: np.ndarray, max_gap_s: float, max_plausible_mps: float,
               jump_window: int = 5) -> list[tuple[int, int]]:
    """Half-open index ranges of runs: consecutive usable samples, gaps <= max_gap_s, increasing time, no persistent jump."""
    n = len(t)
    brk = np.zeros(n, bool)
    if n > 1:
        dt = np.diff(t)
        brk[1:] = (dt > max_gap_s) | (dt <= 0) | ~usable[1:] | ~usable[:-1]
    out = []
    edges = np.r_[0, np.flatnonzero(brk), n]
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 2 or not usable[a:b].all():
            continue
        cuts = _persistent_jumps(t[a:b], x[a:b], y[a:b], max_plausible_mps, jump_window)
        bounds = np.r_[0, cuts, b - a]
        out += [(a + p, a + q) for p, q in zip(bounds[:-1], bounds[1:]) if q - p >= 2]
    return out


def _persistent_jumps(t, x, y, vmax, w):
    """Indices i where the median position of the w samples from i on is further from the median of the w samples before i than vmax
    allows over the time between the two windows' centres (a level shift, e.g. a box jumping to another person; not a single spike)."""
    n = len(t)
    if n < 2 * w:
        return np.array([], int)
    cuts = []
    last = 0
    for i in range(w, n - w + 1):
        if i - last < w:
            continue
        pa = np.array([np.median(x[i - w:i]), np.median(y[i - w:i])])
        pb = np.array([np.median(x[i:i + w]), np.median(y[i:i + w])])
        span = np.median(t[i:i + w]) - np.median(t[i - w:i])
        if span > 0 and np.hypot(*(pb - pa)) / span > vmax:
            cuts.append(i)
            last = i
    return np.array(cuts, int)


# -------------------------------------------------------------------------------------------------------------- filters
def _uniform(t: np.ndarray, *cols: np.ndarray):
    dt = float(np.median(np.diff(t)))
    k = np.rint((t - t[0]) / dt).astype(int)
    grid = np.arange(k[-1] + 1)
    return dt, k, [np.interp(grid, k, c) for c in cols]


def _savgol_xy(t, x, y, window_s):
    from scipy.signal import savgol_filter

    dt, k, (xu, yu) = _uniform(t, x, y)
    m = len(xu)
    w = int(round(window_s / dt)) | 1
    w = min(w, m if m % 2 else m - 1)
    if w < 5:                                    # too short for a quadratic window: straight-line fit over the run
        A = np.c_[np.ones(m), np.arange(m) * dt]
        cx, cy = np.linalg.lstsq(A, xu, rcond=None)[0], np.linalg.lstsq(A, yu, rcond=None)[0]
        return (A @ cx)[k], (A @ cy)[k], np.full(len(k), cx[1]), np.full(len(k), cy[1])
    sx, sy = savgol_filter(xu, w, 2, mode="interp"), savgol_filter(yu, w, 2, mode="interp")
    vx, vy = savgol_filter(xu, w, 2, deriv=1, delta=dt, mode="interp"), savgol_filter(yu, w, 2, deriv=1, delta=dt, mode="interp")
    return sx[k], sy[k], vx[k], vy[k]


def _hampel(v, half):
    out = v.copy()
    for i in range(len(v)):
        seg = v[max(0, i - half): i + half + 1]
        med = np.median(seg)
        mad = 1.4826 * np.median(np.abs(seg - med))
        if mad > 0 and abs(v[i] - med) > 3 * mad:
            out[i] = med
    return out


def _robust_poly_1d(t, v, window_s, iters=2):
    n = len(t)
    h = window_s / 2
    rob = np.ones(n)
    fit, der = v.copy(), np.zeros(n)
    for it in range(iters + 1):
        for i in range(n):
            lo, hi = np.searchsorted(t, t[i] - h), np.searchsorted(t, t[i] + h, side="right")
            tt = t[lo:hi] - t[i]
            w = (1 - np.clip(np.abs(tt) / (h + 1e-9), 0, 1) ** 3) ** 3 * rob[lo:hi]
            deg = 2 if hi - lo >= 5 else 1
            if hi - lo < deg + 1 or (w > 0).sum() < deg + 1:
                continue
            A = np.vander(tt, deg + 1, increasing=True) * np.sqrt(w)[:, None]
            c = np.linalg.lstsq(A, v[lo:hi] * np.sqrt(w), rcond=None)[0]
            fit[i], der[i] = c[0], c[1]
        if it < iters:
            r = v - fit
            s = 6 * np.median(np.abs(r)) + 1e-9
            rob = (1 - np.clip(np.abs(r) / s, 0, 1) ** 2) ** 2
    return fit, der


def _kalman_rts_1d(t, z, var, q, gate):
    """Constant-velocity Kalman + RTS smoother on one axis (pure floats; runs are short)."""
    n = len(t)
    xs = np.zeros((n, 2)); Ps = np.zeros((n, 2, 2)); xp = np.zeros((n, 2)); Pp = np.zeros((n, 2, 2))
    x = np.array([z[0], 0.0]); P = np.array([[var[0], 0.0], [0.0, 25.0]])
    for i in range(n):
        if i:
            dt = t[i] - t[i - 1]
            F = np.array([[1.0, dt], [0.0, 1.0]])
            Q = q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
            x = F @ x; P = F @ P @ F.T + Q
        xp[i], Pp[i] = x, P
        r = var[i]
        S = P[0, 0] + r
        innov = z[i] - x[0]
        if innov * innov > gate * gate * S:                 # outlier-resistant: inflate the variance of an implausible measurement
            r *= innov * innov / (gate * gate * S)
            S = P[0, 0] + r
        K = P[:, 0] / S
        x = x + K * innov
        P = P - np.outer(K, P[0, :])
        xs[i], Ps[i] = x, P
    out = xs.copy()
    for i in range(n - 2, -1, -1):
        dt = t[i + 1] - t[i]
        F = np.array([[1.0, dt], [0.0, 1.0]])
        C = Ps[i] @ F.T @ np.linalg.inv(Pp[i + 1])
        out[i] = xs[i] + C @ (out[i + 1] - xp[i + 1])
    return out[:, 0], out[:, 1]


def stabilize_run(t, x, y, cfg: MotionConfig, var_x=None, var_y=None):
    """Stabilised positions and velocity of ONE run at its observed samples: (x, y, vx, vy)."""
    t, x, y = (np.asarray(v, float) for v in (t, x, y))
    if cfg.method == "savgol":
        return _savgol_xy(t, x, y, cfg.window_s)
    if cfg.method == "hampel_savgol":
        half = max(1, int(round(cfg.window_s / float(np.median(np.diff(t))))) // 2)
        return _savgol_xy(t, _hampel(x, half), _hampel(y, half), cfg.window_s)
    if cfg.method == "robust_poly":
        (sx, vx), (sy, vy) = _robust_poly_1d(t, x, cfg.window_s), _robust_poly_1d(t, y, cfg.window_s)
        return sx, sy, vx, vy
    if cfg.method == "kalman_rts":
        d = cfg.meas_sigma_m ** 2
        vxv = np.full(len(t), d) if var_x is None else np.asarray(var_x, float)
        vyv = np.full(len(t), d) if var_y is None else np.asarray(var_y, float)
        sx, vx = _kalman_rts_1d(t, x, vxv, cfg.accel_noise, cfg.gate_sigma)
        sy, vy = _kalman_rts_1d(t, y, vyv, cfg.accel_noise, cfg.gate_sigma)
        return sx, sy, vx, vy
    raise ValueError(f"unknown stabiliser {cfg.method!r}")


def stabilized_runs(t_s, x_m, y_m, cfg: MotionConfig, *, max_gap_s: float, max_plausible_kmh: float,
                    usable: Sequence[bool] | None = None, var_x=None, var_y=None, stats: dict | None = None) -> list[dict[str, Any]]:
    """Runs in the format of :func:`football_fatigue.analytics.coach.runs` (t, x, y, speed, accel, raw_distance), stabilised."""
    t, x, y = (np.asarray(v, float) for v in (t_s, x_m, y_m))
    ok = np.ones(len(t), bool) if usable is None else np.asarray(usable, bool)
    vmax = max_plausible_kmh / KMH
    st = stats if stats is not None else {}
    for k in ("runs", "short_runs", "implausible_runs", "samples_in_runs", "samples_used"):
        st.setdefault(k, 0)
    st.setdefault("implausible_cuts", 0)
    out = []

    def one(a, b, depth):
        ts = t[a:b]
        if b - a < 2 or ts[-1] - ts[0] < cfg.min_run_s:
            st["short_runs"] += 1
            return
        sx, sy, vx, vy = stabilize_run(ts, x[a:b], y[a:b], cfg, None if var_x is None else np.asarray(var_x)[a:b],
                                       None if var_y is None else np.asarray(var_y)[a:b])
        v = np.hypot(vx, vy)
        if not np.all(v <= vmax):
            if depth >= 2:
                st["implausible_runs"] += 1
                return
            # cut out only the implausible stretch (+-0.2 s) and re-stabilise the pieces on either side
            bad = np.zeros(b - a, bool)
            for i in np.flatnonzero(~(v <= vmax)):
                bad[np.abs(ts - ts[i]) <= 0.2] = True
            st["implausible_cuts"] += 1
            edges = np.flatnonzero(np.diff(np.r_[1, bad.astype(int), 1]))     # alternating starts / ends of good stretches
            for p, q in zip(edges[::2], edges[1::2]):
                one(a + p, a + q, depth + 1)
            return
        st["samples_used"] += b - a
        out.append({"t": ts, "x": sx, "y": sy, "speed": v, "accel": np.gradient(v, ts) if len(ts) > 1 else np.zeros(1),
                    "raw_distance": float(np.hypot(np.diff(x[a:b]), np.diff(y[a:b])).sum())})

    for a, b in split_runs(t, x, y, ok, max_gap_s, vmax, cfg.jump_window):
        st["runs"] += 1
        st["samples_in_runs"] += b - a
        one(a, b, 0)
    return out


def measurement_variance(jac: np.ndarray, sigma_u_px: float, sigma_v_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample pitch variance (x, y) from the pixel->pitch Jacobian (n, 2 metre, 2 pixel) and pixel sigmas (independent u, v)."""
    j = np.asarray(jac, float)
    vx = j[:, 0, 0] ** 2 * sigma_u_px ** 2 + j[:, 0, 1] ** 2 * sigma_v_px ** 2
    vy = j[:, 1, 0] ** 2 * sigma_u_px ** 2 + j[:, 1, 1] ** 2 * sigma_v_px ** 2
    return vx, vy


def config_from_dict(d: dict[str, Any] | None) -> MotionConfig:
    d = dict(d or {})
    cfg = MotionConfig(**d)
    if cfg.method not in STABILIZERS:
        raise ValueError(f"method must be one of {', '.join(STABILIZERS)}")
    for k in ("window_s", "accel_noise", "meas_sigma_m", "gate_sigma", "min_run_s"):
        v = getattr(cfg, k)
        if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
            raise ValueError(f"{k} must be a positive number")
    return cfg
