"""Metrics with p = P(SWAPPED) and SWAPPED as the positive class; match-level bootstrap intervals; paired comparisons; strata;
abstention (coverage vs error)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

ECE_BINS = 15


def _r(v, nd=4):
    return None if v is None or not np.isfinite(v) else round(float(v), nd)


def ll_vec(y, p):
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(pc) + (1 - y) * np.log(1 - pc))


def ece(y, p, bins: int = ECE_BINS) -> float:
    """Reliability of P(SWAPPED) in equal-width bins."""
    b = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(y[b == k].mean() - p[b == k].mean()) * np.mean(b == k) for k in np.unique(b)))


def bal_acc(y, p) -> float:
    pred = p > 0.5
    r1 = pred[y == 1].mean() if (y == 1).any() else np.nan
    r0 = (~pred[y == 0]).mean() if (y == 0).any() else np.nan
    return float(np.nanmean([r0, r1]))


def basic(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y).astype(int)
    if not len(y):
        return dict(n=0)
    pred = (p > 0.5).astype(int)
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    fn, tn = int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
    auc = roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan
    return dict(n=int(len(y)), swapped=int(y.sum()), balanced_accuracy=_r(bal_acc(y, p)), log_loss=_r(ll_vec(y, p).mean()),
                accuracy=_r(np.mean(pred == y)), precision_swapped=_r(prec), recall_swapped=_r(rec),
                f1_swapped=_r(2 * prec * rec / max(1e-12, prec + rec)), roc_auc=_r(auc), brier=_r(np.mean((p - y) ** 2)), ece=_r(ece(y, p)),
                confusion=dict(tn=tn, fp=fp, fn=fn, tp=tp))


def bootstrap(y, p, groups, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """95 % intervals by resampling matches (samples of a match stay together)."""
    rng = np.random.default_rng(seed)
    ms = np.unique(groups)
    idx = [np.flatnonzero(groups == m) for m in ms]
    ba, ll, auc = [], [], []
    lv = ll_vec(y, p)
    for _ in range(n_boot):
        sel = np.concatenate([idx[k] for k in rng.integers(0, len(ms), len(ms))])
        ba.append(bal_acc(y[sel], p[sel]))
        ll.append(lv[sel].mean())
        if 0 < y[sel].sum() < len(sel):
            auc.append(roc_auc_score(y[sel], p[sel]))
    q = lambda v: [_r(np.quantile(v, 0.025)), _r(np.quantile(v, 0.975))]  # noqa: E731
    return dict(balanced_accuracy_ci=q(ba), log_loss_ci=q(ll), roc_auc_ci=q(auc) if auc else None, n_boot=n_boot)


def paired(y, pa, pb, groups, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Model a minus model b (balanced accuracy and log loss) with 95 % match-bootstrap intervals (resampled together)."""
    rng = np.random.default_rng(seed)
    ms = np.unique(groups)
    idx = [np.flatnonzero(groups == m) for m in ms]
    la, lb = ll_vec(y, pa), ll_vec(y, pb)
    dba, dll = [], []
    for _ in range(n_boot):
        sel = np.concatenate([idx[k] for k in rng.integers(0, len(ms), len(ms))])
        dba.append(bal_acc(y[sel], pa[sel]) - bal_acc(y[sel], pb[sel]))
        dll.append(la[sel].mean() - lb[sel].mean())
    q = lambda v: [_r(np.quantile(v, 0.025)), _r(np.quantile(v, 0.975))]  # noqa: E731
    d_ba, d_ll = bal_acc(y, pa) - bal_acc(y, pb), la.mean() - lb.mean()
    ci_ba, ci_ll = q(dba), q(dll)
    return dict(balanced_accuracy_diff=_r(d_ba), balanced_accuracy_diff_ci=ci_ba, log_loss_diff=_r(d_ll), log_loss_diff_ci=ci_ll,
                ba_significant=bool(ci_ba[0] is not None and (ci_ba[0] > 0 or ci_ba[1] < 0)),
                ll_significant=bool(ci_ll[0] is not None and (ci_ll[0] > 0 or ci_ll[1] < 0)))


def per_fold(y, p, folds) -> dict[str, Any]:
    v = {int(f): basic(y[folds == f], p[folds == f]) for f in np.unique(folds)}
    ba = np.array([r["balanced_accuracy"] for r in v.values()], float)
    ll = np.array([r["log_loss"] for r in v.values()], float)
    return dict(balanced_accuracy_mean=_r(ba.mean()), balanced_accuracy_sd=_r(ba.std(ddof=1)), log_loss_mean=_r(ll.mean()),
                log_loss_sd=_r(ll.std(ddof=1)), folds={f: dict(balanced_accuracy=r["balanced_accuracy"], log_loss=r["log_loss"], n=r["n"])
                                                       for f, r in v.items()})


def coverage_curve(y, p, levels=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)) -> dict[str, Any]:
    """Error among the most confident fraction of samples (abstaining on the rest), and the coverage reachable at error targets."""
    conf = np.maximum(p, 1 - p)
    order = np.argsort(-conf, kind="stable")
    wrong = ((p > 0.5).astype(int) != y)[order]
    cum = np.cumsum(wrong) / np.arange(1, len(y) + 1)
    curve = {f"{c:.1f}": dict(error=_r(cum[max(0, int(round(c * len(y))) - 1)]), threshold=_r(conf[order][max(0, int(round(c * len(y))) - 1)]))
             for c in levels}
    tgt = {}
    for e in (0.01, 0.02, 0.05, 0.10):
        ok = np.flatnonzero(cum <= e)
        k = int(ok[-1]) + 1 if len(ok) else 0
        tgt[f"error_{int(e * 100):02d}"] = dict(coverage=_r(k / len(y)), threshold=_r(conf[order][k - 1]) if k else None)
    return dict(curve=curve, coverage_at_error=tgt, aurc=_r(cum.mean()))


def summarize(y, p, groups, folds, indep, boot: bool = True) -> dict[str, Any]:
    out = {}
    for name, m in (("independent", indep), ("all", np.ones(len(y), bool))):
        r = dict(overall=basic(y[m], p[m]), per_fold=per_fold(y[m], p[m], folds[m]))
        if boot:
            r["bootstrap"] = bootstrap(y[m], p[m], groups[m])
        out[name] = r
    out["per_match_independent"] = {str(int(g)): basic(y[indep & (groups == g)], p[indep & (groups == g)]) for g in np.unique(groups)}
    out["abstention_independent"] = coverage_curve(y[indep], p[indep])
    return out


def strata(y, p, meta, mask, keys: dict[str, np.ndarray]) -> dict[str, Any]:
    out = {}
    for name, col in keys.items():
        out[name] = {}
        for v in sorted({str(c) for c in col[mask]}):
            m = mask & (col.astype(str) == v)
            r = basic(y[m], p[m])
            out[name][v] = {k: r.get(k) for k in ("n", "balanced_accuracy", "log_loss", "accuracy", "roc_auc", "ece")}
    return out
