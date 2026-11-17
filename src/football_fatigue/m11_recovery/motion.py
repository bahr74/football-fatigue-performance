"""Motion-model baselines and engineered association features for two-player identity recovery.

All functions see only a :class:`.views.Sample` (two before tracks, two anonymous after tracks, gap). For a pair (before i,
after j) a *cost* measures how badly after j continues before i; an assignment's cost is the sum over its pairs, and the
decision compares H0 = {0->0, 1->1} with H1 = {0->1, 1->0} (one-to-one by construction).

- ``nearest``: distance between before i's last and after j's first observed point;
- ``cv_fwd``: before i extrapolated at constant velocity (least squares over its last ``FIT_S``) to after j's first time;
- ``cv_bidir``: forward extrapolation of before i and backward extrapolation of after j, both to the gap centre (t = 0);
- ``ca_bidir``: the same with constant acceleration (quadratic fit over ``FIT_CA_S``);
- ``cv1_*`` / ``ca2_*``: the same over longer spans (1 s constant velocity, 2 s constant acceleration), robust to the sub-second
  position jitter found at player contacts in the provider data.

``features`` builds permutation-level features for the learned model: for every pair quantity f, (f00 + f11) - (f01 + f10)
(antisymmetric: swapping the after order negates them), plus symmetric context (gap, speeds, separation, crossing geometry).
"""

from __future__ import annotations

import numpy as np

from .views import Sample

FIT_S = 0.5
FIT_CA_S = 1.0
FIT_LONG_CV_S = 1.0
FIT_LONG_CA_S = 2.0
BASELINES = ("nearest", "cv_fwd", "cv_bidir", "ca_bidir", "cv1_fwd", "cv1_bidir", "ca2_fwd", "ca2_bidir")


def _fit(t: np.ndarray, p: np.ndarray, end: bool, span: float, deg: int) -> tuple[np.ndarray, float] | None:
    """Polynomial fit (degree ``deg``) of the observed points within ``span`` of the track's end (before) or start (after).
    Returns coefficients (deg+1, 2) in powers of (t - t_ref) and t_ref."""
    ok = np.isfinite(p[:, 0])
    if not ok.any():
        return None
    tt, pp = t[ok], p[ok]
    t_ref = tt[-1] if end else tt[0]
    k = (tt >= t_ref - span - 1e-9) if end else (tt <= t_ref + span + 1e-9)
    tt, pp = tt[k] - t_ref, pp[k]
    d = min(deg, len(tt) - 1)
    A = np.vander(tt, d + 1, increasing=True)
    coef = np.linalg.lstsq(A, pp, rcond=None)[0]
    if d < deg:
        coef = np.r_[coef, np.zeros((deg - d, 2))]
    return coef, t_ref


def _at(fit, t: float) -> np.ndarray:
    coef, t_ref = fit
    dt = t - t_ref
    return sum(coef[i] * dt ** i for i in range(len(coef)))


def _vel(fit) -> np.ndarray:
    return fit[0][1] if len(fit[0]) > 1 else np.zeros(2)


def _acc(fit) -> np.ndarray:
    return 2 * fit[0][2] if len(fit[0]) > 2 else np.zeros(2)


class Kin:
    """Per-sample fits, computed once."""

    def __init__(self, s: Sample):
        self.s = s
        self.b_cv = [_fit(s.t_before, s.before[i], True, FIT_S, 1) for i in (0, 1)]
        self.a_cv = [_fit(s.t_after, s.after[j], False, FIT_S, 1) for j in (0, 1)]
        self.b_ca = [_fit(s.t_before, s.before[i], True, FIT_CA_S, 2) for i in (0, 1)]
        self.a_ca = [_fit(s.t_after, s.after[j], False, FIT_CA_S, 2) for j in (0, 1)]
        self.b_cv1 = [_fit(s.t_before, s.before[i], True, FIT_LONG_CV_S, 1) for i in (0, 1)]
        self.a_cv1 = [_fit(s.t_after, s.after[j], False, FIT_LONG_CV_S, 1) for j in (0, 1)]
        self.b_ca2 = [_fit(s.t_before, s.before[i], True, FIT_LONG_CA_S, 2) for i in (0, 1)]
        self.a_ca2 = [_fit(s.t_after, s.after[j], False, FIT_LONG_CA_S, 2) for j in (0, 1)]
        self.valid = all(f is not None for f in self.b_cv + self.a_cv)

    def last(self, i):
        p = self.s.before[i]
        return p[np.flatnonzero(np.isfinite(p[:, 0]))[-1]]

    def first(self, j):
        p = self.s.after[j]
        return p[np.flatnonzero(np.isfinite(p[:, 0]))[0]]


def pair_costs(k: Kin) -> dict[str, np.ndarray]:
    """(2, 2) cost matrices (before i, after j) for each baseline."""
    out = {n: np.zeros((2, 2)) for n in BASELINES}
    for i in (0, 1):
        for j in (0, 1):
            bc, ac, ba, aa = k.b_cv[i], k.a_cv[j], k.b_ca[i], k.a_ca[j]
            out["nearest"][i, j] = np.hypot(*(k.last(i) - k.first(j)))
            out["cv_fwd"][i, j] = np.hypot(*(_at(bc, ac[1]) - k.first(j)))
            out["cv_bidir"][i, j] = np.hypot(*(_at(bc, 0.0) - _at(ac, 0.0)))
            out["ca_bidir"][i, j] = np.hypot(*(_at(ba, 0.0) - _at(aa, 0.0)))
            for n, bf, af in (("cv1", k.b_cv1[i], k.a_cv1[j]), ("ca2", k.b_ca2[i], k.a_ca2[j])):
                out[f"{n}_fwd"][i, j] = np.hypot(*(_at(bf, af[1]) - _at(af, af[1])))
                out[f"{n}_bidir"][i, j] = np.hypot(*(_at(bf, 0.0) - _at(af, 0.0)))
    return out


def margin(c: np.ndarray) -> float:
    """cost(H1) - cost(H0): positive favours H0."""
    return float((c[0, 1] + c[1, 0]) - (c[0, 0] + c[1, 1]))


def _ang(u, v) -> float:
    nu, nv = np.hypot(*u), np.hypot(*v)
    if nu < 1e-6 or nv < 1e-6:
        return 0.0
    return float(np.arccos(np.clip(np.dot(u, v) / (nu * nv), -1, 1)))


FEATURE_NAMES: list[str] = []


PAIR_PHYS = ("logd_nearest", "logd_cv_bidir", "logd_ca_bidir", "logd_cv_fwd", "logd_cv1_bidir", "logd_ca2_bidir", "logd_cv1_fwd",
             "logd_ca2_fwd", "dv", "dheading", "dspeed", "cv_mid_err_rel")


def pair_matrices(s: Sample, k: Kin | None = None) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Pair quantities (2, 2) and symmetric context of one sample."""
    k = k or Kin(s)
    pc = pair_costs(k)
    pair = {f"d_{n}": c for n, c in pc.items()}
    pair.update({f"logd_{n}": np.log(c + 0.05) for n, c in pc.items()})
    m = {n: np.zeros((2, 2)) for n in ("dv", "dspeed", "dheading", "dacc", "implied_minus_speed", "cv_mid_err_rel", "ca_fwd", "ca_bwd", "cv_bwd")}
    for i in (0, 1):
        for j in (0, 1):
            vb, va = _vel(k.b_cv[i]), _vel(k.a_cv[j])
            m["dv"][i, j] = np.hypot(*(vb - va))
            m["dspeed"][i, j] = abs(np.hypot(*vb) - np.hypot(*va))
            m["dheading"][i, j] = _ang(vb, va)
            m["dacc"][i, j] = np.hypot(*(_acc(k.b_ca[i]) - _acc(k.a_ca[j])))
            dt = k.a_cv[j][1] - k.b_cv[i][1]
            m["implied_minus_speed"][i, j] = abs(np.hypot(*(k.first(j) - k.last(i))) / max(dt, 1e-3) - 0.5 * (np.hypot(*vb) + np.hypot(*va)))
            m["cv_mid_err_rel"][i, j] = pc["cv_bidir"][i, j] / (0.25 + 0.5 * s.gap_s * (np.hypot(*vb) + np.hypot(*va)) / 2)
            m["ca_fwd"][i, j] = np.hypot(*(_at(k.b_ca[i], k.a_ca[j][1]) - k.first(j)))
            m["ca_bwd"][i, j] = np.hypot(*(_at(k.a_ca[j], k.b_ca[i][1]) - k.last(i)))
            m["cv_bwd"][i, j] = np.hypot(*(_at(k.a_cv[j], k.b_cv[i][1]) - k.last(i)))
    pair.update(m)
    vb = [_vel(k.b_cv[i]) for i in (0, 1)]
    va = [_vel(k.a_cv[j]) for j in (0, 1)]
    ctx = dict(gap_s=s.gap_s, sep_before_end=np.hypot(*(k.last(0) - k.last(1))), sep_after_start=np.hypot(*(k.first(0) - k.first(1))),
               speed_before_min=min(np.hypot(*v) for v in vb), speed_before_max=max(np.hypot(*v) for v in vb),
               speed_after_min=min(np.hypot(*v) for v in va), speed_after_max=max(np.hypot(*v) for v in va),
               angle_before=_ang(vb[0], vb[1]), angle_after=_ang(va[0], va[1]),
               cv_bidir_min_pair=float(pc["cv_bidir"].min()), cv_bidir_abs_margin=abs(margin(pc["cv_bidir"])))
    return pair, ctx


def features(s: Sample, pair=None, ctx=None) -> np.ndarray:
    """Permutation-level antisymmetric pair differences, then symmetric context (names in FEATURE_NAMES)."""
    if pair is None:
        pair, ctx = pair_matrices(s)
    names, vals = [], []
    for n, c in pair.items():
        names.append(f"perm_{n}")
        vals.append((c[0, 1] + c[1, 0]) - (c[0, 0] + c[1, 1]))
    names += list(ctx)
    vals += list(ctx.values())
    if not FEATURE_NAMES:
        FEATURE_NAMES.extend(names)
    return np.asarray(vals, float)




def usable(s: Sample, min_points: int = 3) -> bool:
    """Every track has at least ``min_points`` observed points (otherwise no motion fit is defined)."""
    return all(np.isfinite(tr[:, 0]).sum() >= min_points for tr in (*s.before, *s.after))


def swap_after(s: Sample) -> Sample:
    """The same sample with the after order exchanged (label flips); used for antisymmetric augmentation and symmetrised
    prediction."""
    return Sample(s.before, s.after[::-1].copy(), s.t_before, s.t_after, s.gap_s, s.noise_m, 1 - s.y, s.key)
