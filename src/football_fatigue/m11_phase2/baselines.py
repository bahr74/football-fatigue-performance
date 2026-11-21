"""Deterministic motion baselines for two-player identity re-association (CPU only, no learned parameters, no labels).

Input contract = the Stage 3 model-input arrays (``window_intersections.samples.INPUT_FIELDS``): two incoming tracklets
``in_pos`` (N, 2, T, 2) observed before the hidden interval and two anonymous outgoing tracklets ``out_pos`` observed after it, with
masks and times on the fixed 15 Hz grid (``in_dt`` relative to g0, ``out_dt`` relative to g1) and the gap length ``gap_s``. Nothing
from inside the hidden interval and no label is ever read. Times are converted to the gap centre (t = 0), as in
``m11_benchmark.data.to_sample``.

Every method returns a pairwise cost matrix ``cost[n, i, j]`` (incoming i -> outgoing j). The two-player decision is the
minimum-cost one-to-one assignment (``assign``: Hungarian / ``linear_sum_assignment``; for 2 x 2 it compares the two permutations).
``margin = cost(swapped) - cost(matched)``; positive favours MATCHED (Stage 3 y = 0), negative SWAPPED (y = 1). Because costs are
pairwise, exchanging the outgoing order permutes the columns and negates the margin exactly (tested).

Methods:
- ``nearest``: distance between the last incoming and the first outgoing observation;
- ``const_pos``: mean position over the last 0.5 s (incoming) vs the first 0.5 s (outgoing) - a smoothed constant-position model;
- ``cv_fwd`` / ``cv_bidir``: least-squares constant velocity over 0.5 s; forward to the first outgoing observation, or both sides
  to the gap centre;
- ``ca_fwd`` / ``ca_bidir``: least-squares constant acceleration over 1.0 s with stability constraints (>= 5 points spanning
  >= 0.5 s, otherwise constant velocity; |a| clipped to ``A_MAX``; extrapolated speed clipped to ``V_MAX``); ``ca_fwd_raw`` is the
  unconstrained fit (diagnostic only);
- ``kf_fwd``: constant-velocity Kalman filter over the incoming observations, predicted to the first outgoing observation; cost =
  innovation negative log-likelihood (Mahalanobis d^2 + log det S); a link is *gated out* when d^2 > chi2(2, 0.99);
- ``kf_bidir``: forward filter of the incoming and backward filter of the outgoing tracklet, both predicted to the gap centre;
  Mahalanobis on the difference with summed covariance (offline association);
- ``ocm_dir``: OC-SORT-*inspired* direction-consistency cue only (angle between the observed incoming heading and the direction from
  the last incoming to the first outgoing observation). It is not OC-SORT (no boxes, IoU, ORU or detector scores).

Kalman noise levels are selected without labels by the one-step predictive likelihood of the observed tracklets (``kf_select``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

DT = 1.0 / 15.0
FIT_CV_S = 0.5
FIT_CA_S = 1.0
CA_MIN_POINTS = 5
CA_MIN_SPAN_S = 0.5
A_MAX = 7.0              # m/s2: sustained player accelerations rarely exceed ~6-8 m/s2; larger fitted values are jitter
V_MAX = 12.0             # m/s: the project's physiological speed bound (encounters.quality)
OCM_DELTA_S = 0.2
OCM_MIN_SPEED = 0.3
GATE_CHI2 = 9.2103       # chi2 quantile 0.99, 2 degrees of freedom
METHODS = ("nearest", "const_pos", "cv_fwd", "cv_bidir", "ca_fwd", "ca_bidir", "ca_fwd_raw", "kf_fwd", "kf_bidir", "ocm_dir")
CORE = ("cv_fwd", "cv_bidir", "ca_bidir", "kf_fwd", "kf_bidir")


@dataclass(frozen=True)
class Tracks:
    """Tracklets in gap-centred time: pos (N, 2, T, 2) with NaN where unobserved, t (N, T)."""
    pos: np.ndarray
    t: np.ndarray


def tracks(x: dict[str, np.ndarray]) -> tuple[Tracks, Tracks, np.ndarray]:
    g = x["gap_s"].astype(np.float64)
    pin = np.where(x["in_mask"][..., None], x["in_pos"], np.nan).astype(np.float64)
    pout = np.where(x["out_mask"][..., None], x["out_pos"], np.nan).astype(np.float64)
    tin = x["in_dt"].astype(np.float64) - g[:, None] / 2
    tout = x["out_dt"].astype(np.float64) + g[:, None] / 2
    return Tracks(pin, tin), Tracks(pout, tout), g


def _endpoint(tr: Tracks, last: bool) -> tuple[np.ndarray, np.ndarray]:
    """Position and time of the last (incoming) / first (outgoing) observation: (N, 2, 2), (N, 2)."""
    ok = np.isfinite(tr.pos[..., 0])
    T = ok.shape[-1]
    idx = (T - 1 - np.argmax(ok[..., ::-1], axis=-1)) if last else np.argmax(ok, axis=-1)
    p = np.take_along_axis(tr.pos, idx[..., None, None], axis=2)[:, :, 0]
    t = np.take_along_axis(np.broadcast_to(tr.t[:, None, :], ok.shape), idx[..., None], axis=2)[..., 0]
    return p, t


def _window(tr: Tracks, last: bool, span: float) -> tuple[np.ndarray, np.ndarray]:
    """Observation weights within ``span`` of the end (incoming) / start (outgoing): (N, 2, T) bool, reference time (N, 2)."""
    ok = np.isfinite(tr.pos[..., 0])
    _, tref = _endpoint(tr, last)
    tt = np.broadcast_to(tr.t[:, None, :], ok.shape)
    w = ok & ((tt >= tref[..., None] - span - 1e-9) if last else (tt <= tref[..., None] + span + 1e-9))
    return w, tref


def polyfit(tr: Tracks, last: bool, span: float, deg: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Weighted least squares per tracklet in powers of (t - tref): coef (N, 2, deg+1, 2), tref (N, 2), n points, time span."""
    w, tref = _window(tr, last, span)
    tt = np.broadcast_to(tr.t[:, None, :], w.shape) - tref[..., None]
    A = np.stack([tt ** k for k in range(deg + 1)], -1) * w[..., None]            # (N, 2, T, d+1)
    b = np.where(w[..., None], np.nan_to_num(tr.pos), 0.0)                        # (N, 2, T, 2)
    AtA = np.einsum("nitk,nitl->nikl", A, A) + 1e-9 * np.eye(deg + 1)
    Atb = np.einsum("nitk,nitc->nikc", A, b)
    coef = np.linalg.solve(AtA, Atb)
    n = w.sum(-1)
    tmin = np.where(w, tt, np.inf).min(-1)
    tmax = np.where(w, tt, -np.inf).max(-1)
    return coef, tref, n, np.where(n > 0, tmax - tmin, 0.0)


def _eval(coef: np.ndarray, tref: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Position of every fitted tracklet (N, 2) at times t (N, 2, K) -> (N, 2, K, 2)."""
    dt = t - tref[..., None]
    return sum(coef[:, :, k][:, :, None, :] * (dt[..., None] ** k) for k in range(coef.shape[2]))


def _pair_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N, 2[i], 2[j], 2) predicted/observed positions for pair (i, j); b same -> (N, 2, 2)."""
    return np.hypot(*(a - b).transpose(3, 0, 1, 2))


def cost_nearest(I: Tracks, O: Tracks) -> np.ndarray:
    pi, _ = _endpoint(I, True)
    po, _ = _endpoint(O, False)
    return np.hypot(*(pi[:, :, None, :] - po[:, None, :, :]).transpose(3, 0, 1, 2))


def cost_const_pos(I: Tracks, O: Tracks) -> np.ndarray:
    ci, *_ = polyfit(I, True, FIT_CV_S, 0)
    co, *_ = polyfit(O, False, FIT_CV_S, 0)
    a, b = ci[:, :, 0], co[:, :, 0]
    return np.hypot(*(a[:, :, None, :] - b[:, None, :, :]).transpose(3, 0, 1, 2))


def _constrain(coef: np.ndarray, n: np.ndarray, span: np.ndarray, cv_coef: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CA stability: fall back to the CV fit when too few points / too short; clip |a| to A_MAX."""
    c = coef.copy()
    bad = (n < CA_MIN_POINTS) | (span < CA_MIN_SPAN_S)
    c[bad] = np.concatenate([cv_coef[bad], np.zeros_like(cv_coef[bad][:, :1])], 1)
    a = 2 * c[:, :, 2]
    an = np.hypot(a[..., 0], a[..., 1])
    s = np.where(an > A_MAX, A_MAX / np.maximum(an, 1e-12), 1.0)
    c[:, :, 2] *= s[..., None]
    return c, bad


def _ca_pos(coef: np.ndarray, tref: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Constrained CA extrapolation: integrate with the speed clipped at V_MAX (closed form per step on a fine grid)."""
    dt = t - tref[..., None]                                          # (N, 2, K)
    out = np.empty(dt.shape + (2,))
    for k in range(dt.shape[-1]):
        d = dt[..., k]
        steps = 12
        h = d / steps
        p = coef[:, :, 0].copy()
        v = coef[:, :, 1].copy()
        a = 2 * coef[:, :, 2]
        for _ in range(steps):
            v_new = v + a * h[..., None]
            sp = np.hypot(v_new[..., 0], v_new[..., 1])
            v_new *= np.where(sp > V_MAX, V_MAX / np.maximum(sp, 1e-12), 1.0)[..., None]
            p = p + 0.5 * (v + v_new) * h[..., None]
            v = v_new
        out[..., k, :] = p
    return out


def motion_costs(I: Tracks, O: Tracks) -> dict[str, np.ndarray]:
    """nearest, const_pos, cv_*, ca_* cost matrices (N, 2, 2)."""
    out = {"nearest": cost_nearest(I, O), "const_pos": cost_const_pos(I, O)}
    po, to = _endpoint(O, False)
    pi, ti = _endpoint(I, True)
    ci, ri, ni, si = polyfit(I, True, FIT_CV_S, 1)
    co, ro, no, so = polyfit(O, False, FIT_CV_S, 1)
    N = len(po)
    zeros = np.zeros((N, 2))
    # forward: incoming i evaluated at outgoing j's first time
    pred = _eval(ci, ri, np.broadcast_to(to[:, None, :], (N, 2, 2)))       # (N, i, j, 2)
    out["cv_fwd"] = np.hypot(*(pred - po[:, None, :, :]).transpose(3, 0, 1, 2))
    mi = _eval(ci, ri, zeros[..., None])[:, :, 0]
    mo = _eval(co, ro, zeros[..., None])[:, :, 0]
    out["cv_bidir"] = np.hypot(*(mi[:, :, None, :] - mo[:, None, :, :]).transpose(3, 0, 1, 2))
    ai, rai, nai, sai = polyfit(I, True, FIT_CA_S, 2)
    ao, rao, nao, sao = polyfit(O, False, FIT_CA_S, 2)
    pred_raw = _eval(ai, rai, np.broadcast_to(to[:, None, :], (N, 2, 2)))
    out["ca_fwd_raw"] = np.hypot(*(pred_raw - po[:, None, :, :]).transpose(3, 0, 1, 2))
    # constrained: refit CV over the same 1 s span as fallback for short tracklets
    cvi, *_ = polyfit(I, True, FIT_CA_S, 1)
    cvo, *_ = polyfit(O, False, FIT_CA_S, 1)
    ki, bad_i = _constrain(ai, nai, sai, cvi)
    ko, bad_o = _constrain(ao, nao, sao, cvo)
    predc = _ca_pos(ki, rai, np.broadcast_to(to[:, None, :], (N, 2, 2)))
    out["ca_fwd"] = np.hypot(*(predc - po[:, None, :, :]).transpose(3, 0, 1, 2))
    # backward extrapolation of the outgoing tracklet = forward extrapolation in reversed time
    kr = ko.copy()
    kr[:, :, 1] *= -1
    mci = _ca_pos(ki, rai, zeros[..., None])[:, :, 0]
    mco = _ca_pos(kr, -rao, zeros[..., None])[:, :, 0]
    out["ca_bidir"] = np.hypot(*(mci[:, :, None, :] - mco[:, None, :, :]).transpose(3, 0, 1, 2))
    out["_ca_fallback"] = np.stack([bad_i.any(1), bad_o.any(1)], 1).any(1)
    return out


def ocm_costs(I: Tracks, O: Tracks) -> np.ndarray:
    """OC-SORT-inspired direction consistency: angle between the incoming observed heading (chord over the last OCM_DELTA_S) and
    the direction from the last incoming to the first outgoing observation. Zero cost when the incoming speed is too small."""
    pi, ti = _endpoint(I, True)
    ok = np.isfinite(I.pos[..., 0])
    tt = np.broadcast_to(I.t[:, None, :], ok.shape)
    w = ok & (tt <= ti[..., None] - OCM_DELTA_S + 1e-9)
    idx = np.where(w.any(-1), (w.shape[-1] - 1 - np.argmax(w[..., ::-1], -1)), np.argmax(ok, -1))
    p0 = np.take_along_axis(I.pos, idx[..., None, None], 2)[:, :, 0]
    t0 = np.take_along_axis(tt, idx[..., None], 2)[..., 0]
    h = (pi - p0) / np.maximum(ti - t0, 1e-6)[..., None]
    sp = np.hypot(h[..., 0], h[..., 1])
    po, _ = _endpoint(O, False)
    d = po[:, None, :, :] - pi[:, :, None, :]
    dot = (h[:, :, None, :] * d).sum(-1)
    nd = np.hypot(d[..., 0], d[..., 1])
    cosang = np.clip(dot / np.maximum(sp[:, :, None] * nd, 1e-12), -1, 1)
    ang = np.arccos(cosang)
    return np.where((sp[:, :, None] < OCM_MIN_SPEED) | (nd < 1e-6), 0.0, ang)


# ---------------------------------------------------------------- Kalman filter (constant velocity), vectorised over samples

def _FQ(dt: np.ndarray, sa: float) -> tuple[np.ndarray, np.ndarray]:
    """Transition and white-acceleration process noise for per-row time steps dt (M,) -> (M, 4, 4) each."""
    M = len(dt)
    F = np.tile(np.eye(4), (M, 1, 1))
    F[:, 0, 2] = F[:, 1, 3] = dt
    q = sa ** 2
    Q = np.zeros((M, 4, 4))
    a, b, c = dt ** 4 / 4, dt ** 3 / 2, dt ** 2
    for p, v in ((0, 2), (1, 3)):
        Q[:, p, p], Q[:, p, v], Q[:, v, p], Q[:, v, v] = a * q, b * q, b * q, c * q
    return F, Q


def kf_filter(pos: np.ndarray, t: np.ndarray, sa: float, sm: float, v0: float = 5.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Forward CV Kalman filter over tracklets pos (M, T, 2) with NaN gaps, times t (M, T). Returns state (M, 4), covariance
    (M, 4, 4), time of the state (M,) = time of the last grid step, and the summed one-step predictive NLL (M,)."""
    M, T, _ = pos.shape
    x = np.zeros((M, 4))
    P = np.tile(np.diag([sm ** 2, sm ** 2, v0 ** 2, v0 ** 2]), (M, 1, 1))
    init = np.zeros(M, bool)
    nll = np.zeros(M)
    R = np.eye(2) * sm ** 2
    H = np.zeros((2, 4))
    H[0, 0] = H[1, 1] = 1
    for k in range(T):
        z = pos[:, k]
        obs = np.isfinite(z[:, 0])
        if k > 0:
            F, Q = _FQ(t[:, k] - t[:, k - 1], sa)
            x = np.where(init[:, None], np.einsum("mij,mj->mi", F, x), x)
            P = np.where(init[:, None, None], F @ P @ F.transpose(0, 2, 1) + Q, P)
        new = obs & ~init
        x[new, :2] = z[new]
        x[new, 2:] = 0
        upd = obs & init
        if upd.any():
            S = P[upd][:, :2, :2] + R
            Si = np.linalg.inv(S)
            nu = z[upd] - x[upd, :2]
            d2 = np.einsum("mi,mij,mj->m", nu, Si, nu)
            nll[upd] += 0.5 * d2 + 0.5 * np.log(np.linalg.det(S)) + np.log(2 * np.pi)
            Kg = P[upd][:, :, :2] @ Si
            x[upd] = x[upd] + np.einsum("mij,mj->mi", Kg, nu)
            P[upd] = P[upd] - Kg @ np.einsum("ij,mjk->mik", H, P[upd])
        init |= obs
    return x, P, t[:, -1].copy(), nll


def _predict(x: np.ndarray, P: np.ndarray, dt: np.ndarray, sa: float) -> tuple[np.ndarray, np.ndarray]:
    F, Q = _FQ(dt, sa)
    return np.einsum("mij,mj->mi", F, x), F @ P @ F.transpose(0, 2, 1) + Q


def kf_costs(I: Tracks, O: Tracks, sa: float, sm: float) -> dict[str, np.ndarray]:
    N = I.pos.shape[0]
    T = I.pos.shape[2]
    ti = np.repeat(I.t, 2, 0)
    xi, Pi, tsi, nll_i = kf_filter(I.pos.reshape(N * 2, T, 2), ti, sa, sm)
    # backward filter of outgoing tracklets: reverse time order and negate time
    po = O.pos[:, :, ::-1].reshape(N * 2, T, 2)
    to = -np.repeat(O.t[:, ::-1], 2, 0)
    xo, Po, tso, nll_o = kf_filter(po, to, sa, sm)
    pof, tof = _endpoint(O, False)                                      # first outgoing observation (N, 2, 2), (N, 2)
    R = np.eye(2) * sm ** 2
    fwd_nll = np.zeros((N, 2, 2))
    fwd_d2 = np.zeros((N, 2, 2))
    for j in (0, 1):
        dt = np.repeat(tof[:, j], 2) - tsi                              # rows ordered (n, i)
        xp, Pp = _predict(xi, Pi, dt, sa)
        S = Pp[:, :2, :2] + R
        nu = np.repeat(pof[:, j], 2, 0) - xp[:, :2]
        d2 = np.einsum("mi,mij,mj->m", nu, np.linalg.inv(S), nu)
        fwd_d2[:, :, j] = d2.reshape(N, 2)
        fwd_nll[:, :, j] = (d2 + np.log(np.linalg.det(S))).reshape(N, 2)
    # bidirectional: both predicted to the gap centre (t = 0; reversed time 0 for the outgoing filter)
    xpi, Ppi = _predict(xi, Pi, 0.0 - tsi, sa)
    xpo, Ppo = _predict(xo, Po, 0.0 - tso, sa)
    mi, Si_ = xpi[:, :2].reshape(N, 2, 2), Ppi[:, :2, :2].reshape(N, 2, 2, 2)
    mo, So_ = xpo[:, :2].reshape(N, 2, 2), Ppo[:, :2, :2].reshape(N, 2, 2, 2)
    S = Si_[:, :, None] + So_[:, None, :]                               # (N, i, j, 2, 2)
    nu = mi[:, :, None, :] - mo[:, None, :, :]
    d2 = np.einsum("nijk,nijkl,nijl->nij", nu, np.linalg.inv(S), nu)
    bid = d2 + np.log(np.linalg.det(S))
    return dict(kf_fwd=fwd_nll, kf_fwd_d2=fwd_d2, kf_bidir=bid, kf_bidir_d2=d2,
                track_nll=np.r_[nll_i, nll_o], track_obs=np.r_[np.isfinite(I.pos[..., 0]).reshape(N * 2, T).sum(1),
                                                               np.isfinite(O.pos[..., 0]).reshape(N * 2, T).sum(1)])


def kf_select(I: Tracks, O: Tracks, sas=(1.0, 2.0, 4.0, 8.0, 16.0, 32.0), sms=(0.02, 0.05, 0.1, 0.2)) -> dict[str, Any]:
    """Label-free choice of (sigma_a, sigma_m): minimum mean one-step predictive NLL per observation over all observed tracklets
    (incoming forward, outgoing backward). Uses no label and no hidden-interval data."""
    res = {}
    T = I.pos.shape[2]
    N = I.pos.shape[0]
    for sa in sas:
        for sm in sms:
            _, _, _, a = kf_filter(I.pos.reshape(N * 2, T, 2), np.repeat(I.t, 2, 0), sa, sm)
            _, _, _, b = kf_filter(O.pos[:, :, ::-1].reshape(N * 2, T, 2), -np.repeat(O.t[:, ::-1], 2, 0), sa, sm)
            nobs = np.isfinite(I.pos[..., 0]).sum() + np.isfinite(O.pos[..., 0]).sum() - 4 * N
            res[(sa, sm)] = float((a.sum() + b.sum()) / max(nobs, 1))
    best = min(res, key=res.get)
    return dict(sigma_a=best[0], sigma_m=best[1], grid={f"{k[0]}|{k[1]}": v for k, v in res.items()})


# ---------------------------------------------------------------- assignment

def margin(cost: np.ndarray) -> np.ndarray:
    """cost(swapped) - cost(matched); positive favours MATCHED."""
    return (cost[:, 0, 1] + cost[:, 1, 0]) - (cost[:, 0, 0] + cost[:, 1, 1])


def assign(cost: np.ndarray, null_cost: float | None = None) -> list[tuple[int, int]]:
    """Minimum-cost assignment of one cost matrix (rows incoming, columns outgoing). With ``null_cost`` every row / column may
    stay unmatched at that cost (dummy rows / columns); pairs with a cost >= null_cost are never chosen. Returns matched pairs."""
    c = np.asarray(cost, float)
    n, m = c.shape
    if null_cost is None:
        r, k = linear_sum_assignment(c)
        return [(int(a), int(b)) for a, b in zip(r, k)]
    big = 1e12
    A = np.full((n + m, m + n), big)
    A[:n, :m] = np.where(np.isfinite(c) & (c < null_cost), c, big)
    A[:n, m:] = np.where(np.eye(n, dtype=bool), null_cost, big)
    A[n:, :m] = np.where(np.eye(m, dtype=bool), null_cost, big)
    A[n:, m:] = 0.0
    r, k = linear_sum_assignment(A)
    return [(int(a), int(b)) for a, b in zip(r, k) if a < n and b < m and A[a, b] < big]


def gated_decision(d2: np.ndarray, nll: np.ndarray) -> np.ndarray:
    """2 x 2 decision with Mahalanobis gating: 0 MATCHED, 1 SWAPPED, -1 abstain (a full one-to-one assignment is not feasible
    inside the gate). Among feasible permutations the lower NLL wins."""
    ok = d2 <= GATE_CHI2
    f0 = ok[:, 0, 0] & ok[:, 1, 1]
    f1 = ok[:, 0, 1] & ok[:, 1, 0]
    m = margin(nll)
    return np.where(f0 & f1, (m < 0).astype(int), np.where(f0, 0, np.where(f1, 1, -1)))


def all_costs(x: dict[str, np.ndarray], sa: float, sm: float) -> dict[str, np.ndarray]:
    I, O, _ = tracks(x)
    out = motion_costs(I, O)
    out.update(kf_costs(I, O, sa, sm))
    out["ocm_dir"] = ocm_costs(I, O)
    return out
