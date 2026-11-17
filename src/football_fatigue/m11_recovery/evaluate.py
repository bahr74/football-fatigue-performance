"""Metrics for two-player identity recovery: decisions, calibration, abstention, strata and match-level uncertainty.

``p`` is P(H0). The decision is H0 when p > 0.5. Confidence = max(p, 1 - p). With abstention at confidence threshold c, a
*committed* decision is one with confidence >= c; **false-association rate** = wrong committed / committed; **coverage** =
committed / all; **recovery rate** = correct committed / all (never accuracy on committed decisions presented as accuracy).
"""

from __future__ import annotations

from typing import Any

import numpy as np

FA_TARGETS = (0.01, 0.02, 0.05, 0.10)


def _r(v, nd=4):
    return None if v is None or not np.isfinite(v) else round(float(v), nd)


def basic(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    if not len(y):
        return dict(n=0)
    pred = (p > 0.5).astype(int)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    tp, fp, fn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    rec1 = tp / max(1, (y == 1).sum())
    rec0 = int(((pred == 0) & (y == 0)).sum()) / max(1, (y == 0).sum())
    prec = tp / max(1, tp + fp)
    conf = np.maximum(p, 1 - p)
    bins = np.minimum((conf - 0.5) / 0.05, 9).astype(int)
    ece = sum(abs(np.mean(pred[bins == b] == y[bins == b]) - np.mean(conf[bins == b])) * np.mean(bins == b) for b in np.unique(bins))
    return dict(n=int(len(y)), accuracy=_r(np.mean(pred == y)), balanced_accuracy=_r((rec0 + rec1) / 2), precision_h0=_r(prec),
                recall_h0=_r(rec1), f1_h0=_r(2 * prec * rec1 / max(1e-9, prec + rec1)),
                log_loss=_r(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))), brier=_r(np.mean((p - y) ** 2)), ece=_r(ece))


def at_threshold(y, p, c) -> dict[str, Any]:
    conf = np.maximum(p, 1 - p)
    com = conf >= c
    ok = (p > 0.5).astype(int) == y
    n = max(1, len(y))
    return dict(threshold=_r(c), coverage=_r(com.mean() if len(y) else np.nan), committed=int(com.sum()),
                false_association_rate=_r(1 - ok[com].mean() if com.any() else np.nan), recovery_rate=_r((ok & com).sum() / n))


def threshold_for(y, p, fa: float) -> float:
    """Smallest confidence threshold whose committed false-association rate is <= fa (1.0 + eps when none)."""
    conf = np.maximum(p, 1 - p)
    order = np.argsort(-conf)
    wrong = ((p > 0.5).astype(int) != y)[order]
    rate = np.cumsum(wrong) / np.arange(1, len(y) + 1)
    ok = np.flatnonzero(rate <= fa)
    if not len(ok):
        return 1.0 + 1e-9
    k = ok[-1]
    return float(conf[order][k])


def abstention(y, p, thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    """Operating points: at fixed thresholds (e.g. frozen on development) or, when None, fitted on this set (descriptive)."""
    out = {}
    for fa in FA_TARGETS:
        key = f"fa_{int(fa * 100):02d}"
        c = thresholds[key] if thresholds else threshold_for(y, p, fa)
        out[key] = at_threshold(y, p, c)
    return out


def match_bootstrap(t, p, thresholds: dict[str, float] | None, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """95 % intervals by resampling matches (events of a match stay together)."""
    rng = np.random.default_rng(seed)
    ms = np.unique(t["match"])
    idx = {m: np.flatnonzero(t["match"] == m) for m in ms}
    acc, cov = [], []
    for _ in range(n_boot):
        sel = np.concatenate([idx[m] for m in rng.choice(ms, len(ms))])
        y, q = t["y"][sel], p[sel]
        acc.append(np.mean((q > 0.5) == y))
        if thresholds:
            a = at_threshold(y, q, thresholds["fa_05"])
            cov.append((a["coverage"] or 0, a["false_association_rate"] if a["false_association_rate"] is not None else 0))
    out = dict(accuracy_ci=[_r(np.quantile(acc, 0.025)), _r(np.quantile(acc, 0.975))])
    if cov:
        cov = np.array(cov)
        out["fa05_coverage_ci"] = [_r(np.quantile(cov[:, 0], 0.025)), _r(np.quantile(cov[:, 0], 0.975))]
        out["fa05_false_association_ci"] = [_r(np.quantile(cov[:, 1], 0.025)), _r(np.quantile(cov[:, 1], 0.975))]
    return out


def strata(t, p, thresholds=None) -> dict[str, Any]:
    def grp(mask):
        r = basic(t["y"][mask], p[mask])
        if thresholds and mask.any():
            r["fa05"] = at_threshold(t["y"][mask], p[mask], thresholds["fa_05"])
        return r

    dt = t["abs_time_diff_s"]
    est = t["estimated_share"]
    out = dict(
        by_gap={f"{g:g}": grp(t["gap"] == g) for g in np.unique(t["gap"])},
        by_noise={f"{n:g}": grp(t["noise"] == n) for n in np.unique(t["noise"])},
        by_tier={k: grp(t["tier"] == k) for k in np.unique(t["tier"])},
        by_geometry={"no_crossing": grp(~t["crossing"]), "crossing_not_aligned": grp(t["crossing"] & ~t["aligned_crossing"]),
                     "aligned_crossing (advisor candidates)": grp(t["aligned_crossing"])},
        by_abs_time_diff={"<=0.05": grp(dt <= 0.05), "0.05-0.10": grp((dt > 0.05) & (dt <= 0.1)), "0.10-0.20": grp(dt > 0.1)},
        by_estimated_share={"0": grp(est == 0), "0-0.5": grp((est > 0) & (est <= 0.5)), "0.5-1": grp((est > 0.5) & (est < 1)), "1": grp(est >= 1)},
        by_match={m: grp(t["match"] == m) for m in np.unique(t["match"])},
        by_gap_noise={f"{g:g}|{n:g}": grp((t["gap"] == g) & (t["noise"] == n)) for g in np.unique(t["gap"]) for n in np.unique(t["noise"])},
    )
    for r in ("jitter", "implausible_jump", "all_estimated_player", "timing_ill_conditioned", "speed_spike_near_crossing", "stationary_player"):
        m = np.array([r in x.split(",") for x in t["reasons"]])
        out.setdefault("by_reason", {})[r] = grp(m)
    return out


def report(t, p, thresholds=None, boot=True) -> dict[str, Any]:
    rep = dict(overall=basic(t["y"], p), abstention=abstention(t["y"], p, thresholds), strata=strata(t, p, thresholds))
    if boot:
        rep["match_bootstrap"] = match_bootstrap(t, p, thresholds)
    return rep


def _ll(y, p):
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(pc) + (1 - y) * np.log(1 - pc))


def paired_difference(t, p_a, p_b, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Log-loss and accuracy difference (a - b) with 95 % match-bootstrap intervals (both models' predictions resampled together)."""
    rng = np.random.default_rng(seed)
    y = t["y"]
    ms = np.unique(t["match"])
    idx = {m: np.flatnonzero(t["match"] == m) for m in ms}
    la, lb = _ll(y, p_a), _ll(y, p_b)
    ca, cb = ((p_a > 0.5) == y).astype(float), ((p_b > 0.5) == y).astype(float)
    dl, da = [], []
    for _ in range(n_boot):
        sel = np.concatenate([idx[m] for m in rng.choice(ms, len(ms))])
        dl.append(la[sel].mean() - lb[sel].mean())
        da.append(ca[sel].mean() - cb[sel].mean())
    q = lambda v: [_r(np.quantile(v, 0.025)), _r(np.quantile(v, 0.975))]  # noqa: E731
    return dict(log_loss_diff=_r(la.mean() - lb.mean()), log_loss_diff_ci=q(dl), accuracy_diff=_r(ca.mean() - cb.mean()), accuracy_diff_ci=q(da))
