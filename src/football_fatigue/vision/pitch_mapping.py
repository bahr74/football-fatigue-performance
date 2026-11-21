"""Image <-> pitch mapping candidates for the SoccerTrack panorama, with held-out evaluation (M7 sprint).

The panorama is a stitched, strongly distorted wide view (far touchline compressed into ~100 px of image height, near touchline ~470 px), so a
single global homography is not assumed. Candidates, all fitted only to the supplied pitch keypoints (pitch metres, corner origin
x 0..105 along the touchlines, y 0..68 from the far touchline):

* ``homography``  - one global plane homography (reference; expected to fail on this panorama),
* ``fisheye``     - the existing fisheye + pose prototype (``pitch_prototype``),
* ``tps``         - a smoothed thin-plate-spline warp image -> pitch (piecewise-smooth, no camera model),
* ``hybrid``      - fisheye prototype + thin-plate-spline correction of its residual in pitch space.

Errors are only ever reported on keypoints NOT used to fit (k-fold, leave-one-out, and leave-one-out restricted to the 11 interior points,
which is the honest figure for where players are). Acceptance gates are fixed here before any candidate was evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from football_fatigue.vision import pitch_prototype as P

PITCH_LENGTH_M, PITCH_WIDTH_M = 105.0, 68.0
# Pre-declared gates (held-out metres): the first two are the existing project criteria; the zone gate is for coarse (>= ~17 m) zone occupancy.
GATES = {
    "distance_speed": {"median_m": 0.5, "p90_m": 1.0},
    "team_shape": {"median_m": 1.0, "p90_m": 2.0},
    "zone_occupancy": {"median_m": 2.5, "p90_m": 5.0},
}
_S = 1000.0  # pixel scaling for conditioning


@dataclass
class Mapper:
    kind: str
    _to_pitch: object
    _to_image: object | None = None

    def to_pitch(self, px: np.ndarray) -> np.ndarray:
        return self._to_pitch(np.asarray(px, float).reshape(-1, 2))

    def to_image(self, pitch_xy: np.ndarray) -> np.ndarray:
        if self._to_image is None:
            raise NotImplementedError(self.kind)
        return self._to_image(np.asarray(pitch_xy, float).reshape(-1, 2))


def fit_homography(pitch: np.ndarray, image: np.ndarray) -> Mapper:
    import cv2

    h, _ = cv2.findHomography(image, pitch, 0)
    hi = np.linalg.inv(h)

    def app(m, x):
        y = np.c_[x, np.ones(len(x))] @ m.T
        return y[:, :2] / y[:, 2:3]
    return Mapper("homography", lambda x: app(h, x), lambda x: app(hi, x))


def fit_tps(pitch: np.ndarray, image: np.ndarray, smoothing: float = 1e-3) -> Mapper:
    from scipy.interpolate import RBFInterpolator

    fwd = RBFInterpolator(image / _S, pitch, kernel="thin_plate_spline", smoothing=smoothing)
    inv = RBFInterpolator(pitch / 100.0, image / _S, kernel="thin_plate_spline", smoothing=smoothing)
    return Mapper("tps", lambda x: fwd(x / _S), lambda x: inv(x / 100.0) * _S)


def fit_fisheye(pitch: np.ndarray, image: np.ndarray, intrinsics: Path, step: float = 0.25) -> Mapper:
    par = P.fit(pitch, image, intrinsics)
    pm = P.PitchMap(par, step=step)
    return Mapper("fisheye", pm.to_pitch, lambda x: P.project(par, x))


def refine_inverse(par: np.ndarray, pixels: np.ndarray, seed: np.ndarray, iters: int = 6, h: float = 0.05) -> np.ndarray:
    """Continuous pixel -> pitch inverse of the fisheye model: Newton steps on project(par, X) = pixel from a grid seed (removes the
    nearest-grid-point quantisation of ``PitchMap``). Points whose Jacobian is singular keep their seed."""
    X = np.asarray(seed, float).copy()
    px = np.asarray(pixels, float).reshape(-1, 2)
    for _ in range(iters):
        f0 = P.project(par, X)
        jx = (P.project(par, X + [h, 0]) - f0) / h
        jy = (P.project(par, X + [0, h]) - f0) / h
        J = np.stack([jx, jy], 2)                       # (n, 2 pixel, 2 metre)
        det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        ok = np.abs(det) > 1e-9
        r = px - f0
        dx = np.zeros_like(X)
        dx[ok, 0] = (J[ok, 1, 1] * r[ok, 0] - J[ok, 0, 1] * r[ok, 1]) / det[ok]
        dx[ok, 1] = (-J[ok, 1, 0] * r[ok, 0] + J[ok, 0, 0] * r[ok, 1]) / det[ok]
        X += np.clip(dx, -2.0, 2.0)
    return X


def fit_fisheye_continuous(pitch: np.ndarray, image: np.ndarray, intrinsics: Path, step: float = 0.5) -> Mapper:
    par = P.fit(pitch, image, intrinsics)
    pm = P.PitchMap(par, step=step)
    return Mapper("fisheye_continuous", lambda x: refine_inverse(par, x, pm.to_pitch(x)), lambda x: P.project(par, x))


def fit_hybrid(pitch: np.ndarray, image: np.ndarray, intrinsics: Path, smoothing: float = 1e-2, step: float = 0.25,
               continuous: bool = False) -> Mapper:
    """Fisheye + TPS residual. ``continuous=True`` (M7.1 candidate) uses the Newton-refined inverse instead of the nearest grid point."""
    from scipy.interpolate import RBFInterpolator

    base = fit_fisheye_continuous(pitch, image, intrinsics) if continuous else fit_fisheye(pitch, image, intrinsics, step)
    resid = pitch - base.to_pitch(image)
    corr = RBFInterpolator(image / _S, resid, kernel="thin_plate_spline", smoothing=smoothing)
    # inverse direction (diagnostics: reprojecting pitch markings): fisheye projection + TPS correction of its image-space residual
    iresid = image - base.to_image(pitch)
    icorr = RBFInterpolator(pitch / 100.0, iresid / _S, kernel="thin_plate_spline", smoothing=smoothing)
    return Mapper("hybrid", lambda x: base.to_pitch(x) + corr(x / _S), lambda x: base.to_image(x) + icorr(x / 100.0) * _S)


def fit_kind(kind: str, pitch: np.ndarray, image: np.ndarray, intrinsics: Path) -> Mapper:
    if kind == "homography":
        return fit_homography(pitch, image)
    if kind == "tps":
        return fit_tps(pitch, image)
    if kind == "fisheye":
        return fit_fisheye(pitch, image, intrinsics, step=0.4)
    if kind == "hybrid":
        return fit_hybrid(pitch, image, intrinsics, step=0.4)
    raise ValueError(kind)


def interior_mask(pitch: np.ndarray) -> np.ndarray:
    return (pitch[:, 0] > 0) & (pitch[:, 0] < PITCH_LENGTH_M) & (pitch[:, 1] > 0) & (pitch[:, 1] < PITCH_WIDTH_M)


def held_out_errors(kind: str, pitch: np.ndarray, image: np.ndarray, intrinsics: Path, protocol: str, folds: int = 8, seed: int = 2) -> np.ndarray:
    """Pitch-space error (m) of every held-out keypoint under ``protocol``: 'kfold' (random folds, as the prototype),
    'loo' (leave one out over all points) or 'loo_interior' (leave out each interior point; the rest stay in training)."""
    n = len(pitch)
    if protocol == "kfold":
        idx = np.random.default_rng(seed).permutation(n)
        groups = np.array_split(idx, folds)
    elif protocol == "loo":
        groups = [np.array([i]) for i in range(n)]
    elif protocol == "loo_interior":
        groups = [np.array([i]) for i in np.flatnonzero(interior_mask(pitch))]
    else:
        raise ValueError(protocol)
    errs = []
    for held in groups:
        train = np.setdiff1d(np.arange(n), held)
        m = fit_kind(kind, pitch[train], image[train], intrinsics)
        errs += list(np.linalg.norm(m.to_pitch(image[held]) - pitch[held], axis=1))
    return np.array(errs)


def summarise(e: np.ndarray) -> dict:
    return {"n": int(len(e)), "median_m": round(float(np.median(e)), 2), "p90_m": round(float(np.percentile(e, 90)), 2),
            "max_m": round(float(e.max()), 2), "mean_m": round(float(e.mean()), 2)}


def gates(summary: dict) -> dict:
    return {g: summary["median_m"] <= c["median_m"] and summary["p90_m"] <= c["p90_m"] for g, c in GATES.items()}


def pitch_lines(step: float = 0.5) -> list[np.ndarray]:
    """Polylines (metres, corner origin) of the standard 105 x 68 markings, for reprojection diagnostics."""
    L, W = PITCH_LENGTH_M, PITCH_WIDTH_M
    seg = lambda a, b: np.linspace(a, b, max(2, int(np.hypot(b[0] - a[0], b[1] - a[1]) / step)))
    out = [seg((0, 0), (L, 0)), seg((L, 0), (L, W)), seg((L, W), (0, W)), seg((0, W), (0, 0)), seg((L / 2, 0), (L / 2, W))]
    t = np.linspace(0, 2 * np.pi, 120)
    out.append(np.c_[L / 2 + 9.15 * np.cos(t), W / 2 + 9.15 * np.sin(t)])
    for x0, s in ((0, 1), (L, -1)):
        for depth, half in ((16.5, 20.16), (5.5, 9.16)):
            out.append(np.array([(x0, W / 2 - half), (x0 + s * depth, W / 2 - half), (x0 + s * depth, W / 2 + half), (x0, W / 2 + half)]))
    return out
