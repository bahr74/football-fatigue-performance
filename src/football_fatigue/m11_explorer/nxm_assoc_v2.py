"""N x M association benchmark with separate fit / calibration / test partitions (``m11-nxm-association-v2``; EXPLORATORY).

Runs the same protocol on the M12 prototype (``m11-nxm-exploratory-v1``) and on the corrected one (``m11-nxm-exploratory-v2``).
Truth is provider identity (not human-reviewed); nothing here is a video or validated result.

**Partitions (all match-disjoint; the frozen Stage 4 outer folds, 8 folds of 4 matches).** For outer test fold f:
calibration folds = {f+1, f+2} (mod 8), fit folds = the other 5. Learned scorers are fitted on the fit folds only; every
threshold (null tau, hybrid weight, abstention thresholds), the isotonic link calibrator and the decision / scenario confidence
models are chosen on the calibration folds only; the test fold is touched once, for scoring. Physics parameters (S2) have no fitted
model and are tuned on a bounded subset of the 7 non-test folds (as in M12).

**Model view.** Every cost / feature function receives ``view(d)``: incoming and outgoing tracks as {id, t, x, y} only. Truth,
meta (slots, player ids, provider status, hidden intervals, variant) and incoming labels are removed (tested).

**Systems.** RAND (random one-to-one), SC_order (outgoing id order), SC_timing (rank of disappearance vs rank of reappearance),
SC_hgb (HGB on non-motion features only: time difference, relative times, sample counts, N, M; with null), SC_tex (HGB on provider
texture only, with null); S0 / S1 / S2 physics (as M12); L1 (learned, no null), L2 (learned + null), L2t (+ texture), H (hybrid:
-log p_HGB + beta * bidirectional d2, with null).

**Decisions and confidence.** Each incoming track gets one decision: a link or null ("no outgoing match": the player left).
Abstention is a separate act: refusing to output a decision because its confidence is low. Decision margin = increase of the
optimal total cost when that decision is forbidden. Confidence models (fitted on calibration folds): isotonic link probability;
decision confidence = logistic(log margin, is_null, calibrated p, N, M); scenario confidence = logistic(min / mean decision
confidence, log min margin, N, M).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from football_fatigue.m11_phase2 import common as C

from . import multiplayer as MP
from . import nxm_compare as NC

OUT = "m11-nxm-association-v2"
BIG = 1e12
CAP = 1e3
N_FOLDS = 8
TAUS = NC.TAUS
BETAS = (0.02, 0.05, 0.1, 0.2)
NOISE_M = 0.15
TUNE_SUB = 1500
SC_FEATURES = ["dt", "t_end_rel", "t_first_rel", "n_in", "n_out", "N", "M"]
TEX_FEATURES = ["texture_diff", "texture_in", "texture_out", "N", "M"]
TARGET_ACC = 0.95
TARGET_EXACT = 0.90


# ---------------------------------------------------------------- model view and partitions

def view(d: dict[str, Any]) -> dict[str, Any]:
    """The only input any system sees: anonymous tracks with times and positions (no truth, meta, labels or provider status)."""
    cut = lambda tr: {k: tr[k] for k in ("id", "t", "x", "y")}  # noqa: E731
    return dict(incoming=[cut(t) for t in d["incoming"]], outgoing=[cut(t) for t in d["outgoing"]])


def partitions(fold: np.ndarray, f: int, n_folds: int = N_FOLDS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(fit, calibration, test) scenario indices for outer test fold ``f``; folds are match-disjoint."""
    cal = {(f + 1) % n_folds, (f + 2) % n_folds}
    te = np.flatnonzero(fold == f)
    ca = np.flatnonzero(np.isin(fold, list(cal)))
    fi = np.flatnonzero((fold != f) & ~np.isin(fold, list(cal)))
    return fi, ca, te


# ---------------------------------------------------------------- assignment with per-track null costs, decision margins

def _vec(lam, n: int, forced: float) -> np.ndarray:
    return np.full(n, forced) if lam is None else (np.full(n, float(lam)) if np.isscalar(lam) else np.asarray(lam, float))


def assign_v(cost: np.ndarray, lin=None, lout=None) -> tuple[list[tuple[int, int]], float]:
    """One-to-one assignment with per-row / per-column null costs (scalars or vectors). ``None`` = no explicit null: a track stays
    unmatched only when the shape forces it. Returns (pairs, total cost including null costs)."""
    c = np.where(np.isfinite(cost), cost, BIG)
    n, m = c.shape
    li = _vec(lin, n, BIG / 10 if n > m else BIG)
    lo = _vec(lout, m, BIG / 10 if m > n else BIG)
    A = np.full((n + m, m + n), BIG)
    A[:n, :m] = c
    A[:n, m:][np.eye(n, dtype=bool)] = li
    A[n:, :m][np.eye(m, dtype=bool)] = lo
    A[n:, m:] = 0.0
    from scipy.optimize import linear_sum_assignment
    r, k = linear_sum_assignment(A)
    pairs = [(int(a), int(b)) for a, b in zip(r, k) if a < n and b < m and A[a, b] < BIG / 100]
    return pairs, float(A[r, k].sum())


def decisions(cost: np.ndarray, lin=None, lout=None, margins: bool = True) -> tuple[list[int], list[float]]:
    """Per incoming row: chosen outgoing column (-1 = null) and the decision margin (cost increase when it is forbidden, capped)."""
    n, m = cost.shape
    pairs, base = assign_v(cost, lin, lout)
    ch = [-1] * n
    for a, b in pairs:
        ch[a] = b
    if not margins:
        return ch, [np.nan] * n
    mg = []
    for i in range(n):
        if ch[i] >= 0:
            c2 = cost.copy()
            c2[i, ch[i]] = np.inf
            _, t2 = assign_v(c2, lin, lout)
        else:
            li = _vec(lin, n, BIG / 10 if n > m else BIG).copy()
            li[i] = BIG
            _, t2 = assign_v(cost, li, lout)
        mg.append(float(min(max(t2 - base, 0.0), CAP)))
    return ch, mg


def score(d: dict[str, Any], ch: list[int]) -> dict[str, Any]:
    ins, outs = [t["id"] for t in d["incoming"]], [t["id"] for t in d["outgoing"]]
    truth = d["truth"]
    pred = {ins[i]: (outs[j] if j >= 0 else None) for i, j in enumerate(ch)}
    tp = sum(1 for i, o in pred.items() if o is not None and truth[i] == o)
    used, true_targets = {o for o in pred.values() if o is not None}, {o for o in truth.values() if o is not None}
    new_true = [o for o in outs if o not in true_targets]
    new_pred = [o for o in outs if o not in used]
    return dict(exact=int(all(pred[i] == truth[i] for i in ins)), links_tp=tp, links_pred=sum(o is not None for o in pred.values()),
                links_true=len(true_targets), null_tp=sum(1 for i in ins if pred[i] is None and truth[i] is None),
                null_pred=sum(o is None for o in pred.values()), null_true=sum(v is None for v in truth.values()),
                id_switches=sum(1 for i in ins if pred[i] is not None and truth[i] is not None and pred[i] != truth[i]),
                new_tp=len(set(new_true) & set(new_pred)), new_pred=len(new_pred), new_true=len(new_true),
                correct=[int(pred[i] == truth[i]) for i in ins], is_null=[int(pred[i] is None) for i in ins])


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    s = lambda k: float(sum(r[k] for r in rows))  # noqa: E731
    p, r = s("links_tp") / max(s("links_pred"), 1), s("links_tp") / max(s("links_true"), 1)
    rd = lambda a, b: round(a / b, 4) if b else None  # noqa: E731
    return dict(n=len(rows), exact=round(s("exact") / len(rows), 4), link_precision=round(p, 4), link_recall=round(r, 4),
                link_f1=round(2 * p * r / max(p + r, 1e-12), 4), id_switches_per_scenario=round(s("id_switches") / len(rows), 4),
                false_link_rate=rd(s("links_pred") - s("links_tp"), s("links_pred")), null_recall=rd(s("null_tp"), s("null_true")),
                null_precision=rd(s("null_tp"), s("null_pred")), new_recall=rd(s("new_tp"), s("new_true")),
                new_precision=rd(s("new_tp"), s("new_pred")))


# ---------------------------------------------------------------- features (model view only)

def shortcut_features(v: dict[str, Any]) -> np.ndarray:
    """Non-motion pair features: what a model could exploit without looking at where anybody moves."""
    ins, outs = v["incoming"], v["outgoing"]
    te = np.array([t["t"][-1] for t in ins], float)
    tf = np.array([t["t"][0] for t in outs], float)
    rows = []
    for i in range(len(ins)):
        for j in range(len(outs)):
            rows.append([tf[j] - te[i], te[i] - te.mean(), tf[j] - tf.mean(), len(ins[i]["t"]), len(outs[j]["t"]), len(ins), len(outs)])
    return np.asarray(rows, float)


def texture_features(v: dict[str, Any]) -> np.ndarray:
    ri = [np.log1p(1e3 * NC._rough(t)) for t in v["incoming"]]
    ro = [np.log1p(1e3 * NC._rough(t)) for t in v["outgoing"]]
    return np.asarray([[abs(a - b), a, b, len(ri), len(ro)] for a in ri for b in ro], float)


def featurize(d: dict[str, Any]) -> dict[str, Any]:
    v = view(d)
    X, idx = NC.pair_features(v, texture=True)
    mc = NC.motion_costs(v)               # default sigma_a (pair_features computes the same internally; kept separate, v1 code untouched)
    return dict(X=X, idx=idx, SC=shortcut_features(v), TX=texture_features(v), fwd=mc["fwd"], bi=mc["bi"],
                t_end=mc["t_end"], t_first=mc["t_first"])


def labels_of(d: dict[str, Any], idx) -> np.ndarray:
    ins, outs = [t["id"] for t in d["incoming"]], [t["id"] for t in d["outgoing"]]
    return np.array([int(d["truth"][ins[i]] == outs[j]) for i, j in idx])


def _mat(p: np.ndarray, idx, n: int, m: int) -> np.ndarray:
    P = np.zeros((n, m))
    for (i, j), v in zip(idx, p):
        P[i, j] = v
    return np.clip(P, 1e-6, 1 - 1e-6)


# ---------------------------------------------------------------- calibration metrics

def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    if not len(p):
        return float("nan")
    k = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[k == b].mean() - y[k == b].mean()) * (k == b).mean() for b in range(bins) if (k == b).any()))


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)) if len(p) else float("nan")


def risk_coverage(conf: np.ndarray, ok: np.ndarray, coverages=(1.0, 0.9, 0.8, 0.6, 0.4, 0.2)) -> dict[str, Any]:
    o = np.argsort(-np.asarray(conf), kind="stable")
    okc = np.asarray(ok, float)[o]
    cum = np.cumsum(okc) / np.arange(1, len(okc) + 1)
    out = {str(c): round(float(cum[max(1, int(round(c * len(okc)))) - 1]), 4) for c in coverages}
    out["aurc_error"] = round(float(np.mean(1 - cum)), 4)
    return out


def threshold_for_accuracy(conf: np.ndarray, ok: np.ndarray, target: float) -> float:
    """Lowest confidence threshold whose selective accuracy on the given (calibration) data is >= target (max coverage)."""
    o = np.argsort(-np.asarray(conf), kind="stable")
    c, okc = np.asarray(conf)[o], np.asarray(ok, float)[o]
    acc = np.cumsum(okc) / np.arange(1, len(okc) + 1)
    good = np.flatnonzero(acc >= target)
    return float(c[good[-1]]) if len(good) else float("inf")


def selective(conf, ok, is_link, thr: float) -> dict[str, Any]:
    conf, ok, is_link = map(np.asarray, (conf, ok, is_link))
    keep = conf >= thr
    kl = keep & (is_link == 1)
    return dict(threshold=None if not np.isfinite(thr) else round(float(thr), 4), coverage=round(float(keep.mean()), 4),
                selective_accuracy=round(float(ok[keep].mean()), 4) if keep.any() else None,
                false_link_rate_accepted=round(float(1 - ok[kl].mean()), 4) if kl.any() else None, n=int(len(conf)))


# ---------------------------------------------------------------- experiment

def _hgb():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)


def _lr():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))


def perturb(d: dict[str, Any], sigma: float, seed: int) -> dict[str, Any]:
    """Stress copy: independent Gaussian position noise (sigma metres) on every observed sample (truth unchanged)."""
    rng = np.random.default_rng(seed)
    e = dict(d)
    e["incoming"] = [dict(t, x=(np.asarray(t["x"]) + rng.normal(0, sigma, len(t["x"]))).tolist(),
                          y=(np.asarray(t["y"]) + rng.normal(0, sigma, len(t["y"]))).tolist()) for t in d["incoming"]]
    e["outgoing"] = [dict(t, x=(np.asarray(t["x"]) + rng.normal(0, sigma, len(t["x"]))).tolist(),
                          y=(np.asarray(t["y"]) + rng.normal(0, sigma, len(t["y"]))).tolist()) for t in d["outgoing"]]
    return e


def run(sc: list[dict[str, Any]], fold: np.ndarray, log=print, stress_every: int = 2) -> dict[str, Any]:
    from sklearn.isotonic import IsotonicRegression
    t0 = time.perf_counter()
    F = []
    for k, d in enumerate(sc):
        F.append(featurize(d))
        if k % 2000 == 0:
            log(f"  features {k}/{len(sc)} {time.perf_counter() - t0:.0f} s")
    Y = [labels_of(d, f["idx"]) for d, f in zip(sc, F)]
    _sig: dict[tuple[float, int], np.ndarray] = {}

    def sig_cost(sa: float, k: int) -> np.ndarray:
        if sa == MP.SIGMA_A:
            return F[k]["bi"]
        if (sa, k) not in _sig:
            _sig[(sa, k)] = NC.motion_costs(view(sc[k]), sa)["bi"]
        return _sig[(sa, k)]
    G = np.arange(len(sc))[:: max(1, (len(sc) * 7 // 8) // TUNE_SUB)]       # one fixed physics-tuning subset (test fold removed per fold)
    offs = np.r_[0, np.cumsum([len(f["idx"]) for f in F])]
    XA = {"X": np.concatenate([f["X"] for f in F]), "SC": np.concatenate([f["SC"] for f in F]), "TX": np.concatenate([f["TX"] for f in F])}
    log(f"  features {time.perf_counter() - t0:.0f} s")
    nf = len(NC.FEATURES)
    systems = ["RAND", "SC_order", "SC_timing", "SC_hgb", "SC_tex", "S0", "S1", "S2", "L1_lr", "L1_hgb", "L2_lr", "L2_hgb", "L2t_hgb", "H_hgb"]
    rows: dict[str, list] = {s: [None] * len(sc) for s in systems}
    dec_rows: list[dict[str, Any]] = []          # test-fold decisions with calibrated confidence (calibrated systems)
    scen_rows: list[dict[str, Any]] = []
    chosen: dict[int, dict[str, Any]] = {}
    calib: dict[int, dict[str, Any]] = {}
    stress_rows: list[dict[str, Any]] = []
    for f in range(N_FOLDS):
        fi, ca, te = partitions(fold, f)
        if not len(te):
            continue
        ch: dict[str, Any] = {}
        # ---- S2 tuning on the 7 non-test folds (bounded subset; no fitted model)
        sub7 = G[fold[G] != f]
        best, bp = -1.0, None
        for sa in NC.SIGMAS:
            for li in NC.NULLS:
                for lo in NC.NULLS:
                    ex = np.mean([score(sc[k], decisions(sig_cost(sa, k), li, lo, False)[0])["exact"] for k in sub7])
                    if ex > best:
                        best, bp = ex, (sa, li, lo)
        ch["S2"] = dict(sigma_a=bp[0], null_in=bp[1], null_out=bp[2])
        # ---- learned models on the fit folds only
        def stack(key, ks, cols=None):
            X = np.concatenate([F[k][key] for k in ks])
            return X if cols is None else X[:, cols]
        yfi = np.concatenate([Y[k] for k in fi])
        models = {"lr": _lr().fit(stack("X", fi, slice(0, nf)), yfi), "hgb": _hgb().fit(stack("X", fi, slice(0, nf)), yfi),
                  "hgbt": _hgb().fit(stack("X", fi), yfi), "sc": _hgb().fit(stack("SC", fi), yfi), "tex": _hgb().fit(stack("TX", fi), yfi)}

        cols = {"lr": ("X", slice(0, nf)), "hgb": ("X", slice(0, nf)), "hgbt": ("X", slice(None)), "sc": ("SC", slice(None)), "tex": ("TX", slice(None))}
        need = np.concatenate([ca, te])
        PA: dict[str, dict[int, np.ndarray]] = {}
        for name, (key, cs) in cols.items():                 # one batched prediction per model (calibration + test scenarios)
            rows_ = np.concatenate([np.arange(offs[k], offs[k + 1]) for k in need])
            p = models[name].predict_proba(XA[key][rows_][:, cs])[:, 1]
            PA[name], o = {}, 0
            for k in need:
                L_ = offs[k + 1] - offs[k]
                PA[name][int(k)] = _mat(p[o:o + L_], F[k]["idx"], len(sc[k]["incoming"]), len(sc[k]["outgoing"]))
                o += L_

        def P(k, name, feats=None):
            if feats is None:
                return PA[name][int(k)]
            key, cs = cols[name]
            return _mat(models[name].predict_proba(feats[key][:, cs])[:, 1], feats["idx"], len(sc[k]["incoming"]), len(sc[k]["outgoing"]))
        # ---- thresholds on the calibration folds only
        Pc = {name: {k: P(k, name) for k in ca} for name in models}
        ca_sub = ca[:: max(1, len(ca) // TUNE_SUB)]
        for name in models:
            ch[f"tau_{name}"] = max(TAUS, key=lambda tau: np.mean([score(sc[k], decisions(-np.log(Pc[name][k]), -np.log(tau), -np.log(tau), False)[0])["exact"] for k in ca_sub]))
        hb = max(((b, tau) for b in BETAS for tau in TAUS), key=lambda bt: np.mean([
            score(sc[k], decisions(-np.log(Pc["hgb"][k]) + bt[0] * np.minimum(F[k]["bi"], 1e4), -np.log(bt[1]), -np.log(bt[1]), False)[0])["exact"]
            for k in ca_sub]))
        ch["H"] = dict(beta=hb[0], tau=hb[1])
        # isotonic link calibrator (pairs of calibration folds)
        iso = {}
        for name in ("lr", "hgb"):
            iso[name] = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(
                np.concatenate([Pc[name][k][tuple(np.array(F[k]["idx"]).T)] for k in ca]), np.concatenate([Y[k] for k in ca]))

        def cost_of(sys_, k, feats=None, Pm=None):
            f_ = feats or F[k]
            if sys_ == "S2":
                return sig_cost(bp[0], k), bp[1], bp[2]
            name = {"L2_lr": "lr", "L2_hgb": "hgb"}[sys_]
            tau = ch[f"tau_{name}"]
            return -np.log(Pm if Pm is not None else P(k, name, feats)), -np.log(tau), -np.log(tau)

        def dec_feats(sys_, k, chs, mg, feats=None, Pm=None):
            """Decision-level confidence features (model view only)."""
            n, m = len(sc[k]["incoming"]), len(sc[k]["outgoing"])
            rowsf = []
            Pi = None
            if sys_ != "S2":
                name = {"L2_lr": "lr", "L2_hgb": "hgb"}[sys_]
                Pr = Pm if Pm is not None else P(k, name, feats)
                Pi = iso[name].predict(Pr.ravel()).reshape(Pr.shape)
            for i, (j, g) in enumerate(zip(chs, mg)):
                pc = 0.0 if Pi is None else (float(Pi[i, j]) if j >= 0 else float(1 - Pi[i].max()))
                rowsf.append([np.log1p(g), float(j < 0), pc, n, m])
            return np.asarray(rowsf, float)
        cal_sys = ("S2", "L2_lr", "L2_hgb")
        conf_models, scen_models, thr = {}, {}, {}
        from sklearn.linear_model import LogisticRegression
        for s in cal_sys:
            Xd, yd, Xs, ys = [], [], [], []
            for k in ca:
                c, li, lo = cost_of(s, k, Pm=Pc[{"S2": "hgb", "L2_lr": "lr", "L2_hgb": "hgb"}[s]][k] if s != "S2" else None)
                chs, mg = decisions(c, li, lo, True)
                r = score(sc[k], chs)
                fd = dec_feats(s, k, chs, mg, Pm=Pc[{"L2_lr": "lr", "L2_hgb": "hgb"}[s]][k] if s != "S2" else None)
                Xd.append(fd)
                yd += r["correct"]
                Xs.append(fd)
                ys.append(r["exact"])
            Xd_ = np.concatenate(Xd)
            cm = LogisticRegression(max_iter=2000).fit(Xd_, yd)
            conf_models[s] = cm
            cc = cm.predict_proba(Xd_)[:, 1]
            sfe = np.array([[x[:, 0].min(), x[:, 0].mean(), x[:, 3][0], x[:, 4][0]] for x in Xs])
            dcs = np.split(cc, np.cumsum([len(x) for x in Xs])[:-1])
            sfe = np.c_[sfe, [v.min() for v in dcs], [v.mean() for v in dcs]]
            sm = LogisticRegression(max_iter=2000).fit(sfe, ys)
            scen_models[s] = sm
            sconf = sm.predict_proba(sfe)[:, 1]
            thr[s] = dict(dec_acc95=threshold_for_accuracy(cc, np.array(yd), TARGET_ACC), dec_cov80=float(np.quantile(cc, 0.2)),
                          dec_cov60=float(np.quantile(cc, 0.4)), scen_exact90=threshold_for_accuracy(sconf, np.array(ys), TARGET_EXACT))
        calib[f] = {s: {k: (None if not np.isfinite(v) else round(v, 4)) for k, v in thr[s].items()} for s in cal_sys}
        chosen[f] = ch
        # ---- test fold: score once
        rng = np.random.default_rng(1000 + f)
        for k in te:
            d = sc[k]
            n, m = len(d["incoming"]), len(d["outgoing"])
            v = view(d)
            perm = rng.permutation(max(n, m))
            rows["RAND"][k] = score(d, [int(perm[i]) if perm[i] < m else -1 for i in range(n)])
            order = np.argsort([o["id"] for o in v["outgoing"]])
            rows["SC_order"][k] = score(d, [int(order[i]) if i < m else -1 for i in range(n)])
            rk = np.abs(np.argsort(np.argsort(F[k]["t_end"]))[:, None] - np.argsort(np.argsort(F[k]["t_first"]))[None, :]).astype(float)
            rows["SC_timing"][k] = score(d, decisions(rk + 1e-3 * np.arange(m)[None, :], None, None, False)[0])
            for s_, name in (("SC_hgb", "sc"), ("SC_tex", "tex")):
                tau = ch[f"tau_{name}"]
                rows[s_][k] = score(d, decisions(-np.log(P(k, name)), -np.log(tau), -np.log(tau), False)[0])
            c0 = F[k]["fwd"]
            rows["S0"][k] = score(d, decisions(np.where(c0 <= MP.GATE, c0, np.inf), MP.GATE, MP.GATE, False)[0])
            c1 = F[k]["bi"]
            rows["S1"][k] = score(d, decisions(np.where(c1 <= MP.GATE, c1, np.inf), MP.GATE, MP.GATE, False)[0])
            for name in ("lr", "hgb"):
                rows[f"L1_{name}"][k] = score(d, decisions(-np.log(P(k, name)), None, None, False)[0])
            tau = ch["tau_hgbt"]
            rows["L2t_hgb"][k] = score(d, decisions(-np.log(P(k, "hgbt")), -np.log(tau), -np.log(tau), False)[0])
            rows["H_hgb"][k] = score(d, decisions(-np.log(P(k, "hgb")) + hb[0] * np.minimum(F[k]["bi"], 1e4), -np.log(hb[1]), -np.log(hb[1]), False)[0])
            for s in cal_sys:
                c, li, lo = cost_of(s, k)
                chs, mg = decisions(c, li, lo, True)
                r = score(d, chs)
                rows[s][k] = r
                fd = dec_feats(s, k, chs, mg)
                cc = conf_models[s].predict_proba(fd)[:, 1]
                sconf = float(scen_models[s].predict_proba(np.r_[fd[:, 0].min(), fd[:, 0].mean(), n, m, cc.min(), cc.mean()][None])[:, 1][0])
                for i in range(n):
                    dec_rows.append(dict(system=s, fold=f, k=int(k), conf=float(cc[i]), correct=r["correct"][i], is_null=r["is_null"][i],
                                         p_cal=float(fd[i, 2])))
                scen_rows.append(dict(system=s, fold=f, k=int(k), conf=sconf, exact=r["exact"]))
                if s == "L2_hgb" and stress_every and k % stress_every == 0:
                    e = perturb(d, NOISE_M, 7 + int(k))
                    fe = featurize(e)
                    ce, lie, loe = cost_of(s, k, fe)
                    che, mge = decisions(ce, lie, loe, True)
                    re = score(d, che)
                    ccs = conf_models[s].predict_proba(dec_feats(s, k, che, mge, fe))[:, 1]
                    for i in range(n):
                        stress_rows.append(dict(fold=f, k=int(k), conf=float(ccs[i]), correct=re["correct"][i], is_null=re["is_null"][i],
                                                exact=re["exact"], base_correct=r["correct"][i]))
        log(f"  fold {f}: fit {len(fi)} / cal {len(ca)} / test {len(te)}  S2 {ch['S2']}  tau_hgb {ch['tau_hgb']}  H {ch['H']}  "
            f"({time.perf_counter() - t0:.0f} s)")
    return dict(rows=rows, dec=pd.DataFrame(dec_rows), scen=pd.DataFrame(scen_rows), stress=pd.DataFrame(stress_rows), chosen=chosen,
                calib=calib, runtime_s=round(time.perf_counter() - t0, 1))


# ---------------------------------------------------------------- tables

def strata(sc: list[dict[str, Any]], idx: pd.DataFrame) -> pd.DataFrame:
    """Per-scenario descriptors (for tables only; never model input), normalised across v1 / v2 index schemas."""
    s = pd.DataFrame(dict(match_id=idx["match_id"].to_numpy(), N=idx["N"].to_numpy(), M=idx["M"].to_numpy(), variant=idx["variant"].to_numpy(),
                          n_null=idx["n_null"].to_numpy(), est=idx["est_mean"].to_numpy()))
    s["n_new"] = idx["n_new"].to_numpy() if "n_new" in idx else idx["n_new_out"].to_numpy()
    s["gap"] = idx["gap_mean_s"].to_numpy() if "gap_mean_s" in idx else idx["base_gap_s"].to_numpy()
    s["crossing"] = idx["n_sync_crossings"].to_numpy() > 0
    s["label_risk"] = idx["label_risk"].to_numpy().astype(bool)
    s["encounter"] = idx["encounter_id"].to_numpy() if "encounter_id" in idx else idx["event_id"].to_numpy()
    s["NxM"] = s["N"].astype(str) + "x" + s["M"].astype(str)
    s["N_band"] = np.where(s["N"] >= 4, "N>=4", s["N"].astype(str))
    s["shape"] = np.select([s["M"] < s["N"], s["M"] > s["N"]], ["M<N", "M>N"], "M=N")
    s["nulls"] = s["n_null"].clip(upper=2).astype(str)
    s["news"] = s["n_new"].clip(upper=2).astype(str)
    s["gap_band"] = pd.cut(s["gap"], [0, 0.7, 1.0, 1.3, 9], labels=["<=0.7", "0.7-1.0", "1.0-1.3", ">1.3"]).astype(str)
    s["est_band"] = pd.cut(s["est"], [-1e-9, 0.0, 0.5, 1.0], labels=["visible", "est<=50%", "est>50%"]).astype(str)
    s["difficulty"] = np.select([(s["n_null"] > 0) & (s["n_new"] > 0), (s["n_null"] > 0) | (s["n_new"] > 0), s["crossing"]],
                                ["null+new", "null-or-new", "crossing"], "plain")
    return s


def tables(rows: dict[str, list], st: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"overall": {k: summarize(v) for k, v in rows.items()}}
    for by in ("N", "NxM", "N_band", "shape", "nulls", "news", "variant", "gap_band", "est_band", "crossing", "label_risk", "difficulty"):
        out[f"by_{by}"] = {str(lv): {k: summarize([v[i] for i in np.flatnonzero(st[by].to_numpy() == lv)]) for k, v in rows.items()}
                           for lv in sorted(st[by].unique(), key=str)}
    ex = st[["match_id"]].copy()
    for k, v in rows.items():
        ex[k] = [r["exact"] for r in v]
    out["exact_ci"] = {k: NC.boot_exact(ex, k) for k in rows}
    pairs = [("L2_hgb", "S2"), ("L2_hgb", "S0"), ("S2", "S0"), ("S1", "S0"), ("L1_hgb", "S1"), ("L2t_hgb", "L2_hgb"), ("H_hgb", "L2_hgb"),
             ("L2_hgb", "SC_hgb"), ("S2", "SC_hgb"), ("SC_hgb", "RAND"), ("SC_timing", "RAND"), ("SC_tex", "RAND"), ("L2_hgb", "L2_lr")]
    out["paired"] = {f"{a} - {b}": NC.boot_exact(ex, a, b) for a, b in pairs}
    for lv in ("N>=4",):
        sub = ex[st["N_band"].to_numpy() == lv]
        out[f"paired_{lv}"] = {f"{a} - {b}": NC.boot_exact(sub, a, b) for a, b in (("L2_hgb", "S2"), ("L2_hgb", "SC_hgb"))}
    sub = ex[(st["n_null"] > 0).to_numpy()]
    out["paired_has_null"] = {f"{a} - {b}": NC.boot_exact(sub, a, b) for a, b in (("L2_hgb", "S2"), ("L2_hgb", "SC_hgb"))}
    return out


def calibration_tables(res: dict[str, Any], st: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    dec, scen = res["dec"], res["scen"]
    for s, g in dec.groupby("system"):
        conf, ok = g["conf"].to_numpy(), g["correct"].to_numpy()
        o = dict(decisions=int(len(g)), accuracy_all=round(float(ok.mean()), 4), ece=round(ece(conf, ok), 4), brier=round(brier(conf, ok), 4),
                 risk_coverage=risk_coverage(conf, ok))
        if s != "S2":
            o["p_cal_links"] = dict(ece=round(ece(g.loc[g["is_null"] == 0, "p_cal"], g.loc[g["is_null"] == 0, "correct"]), 4),
                                    brier=round(brier(g.loc[g["is_null"] == 0, "p_cal"], g.loc[g["is_null"] == 0, "correct"]), 4))
        # thresholds selected on calibration folds, applied to each test fold
        for name in ("dec_acc95", "dec_cov80", "dec_cov60"):
            parts = []
            for f, gf in g.groupby("fold"):
                t = res["calib"][f][s][name]
                parts.append(gf.assign(keep=gf["conf"] >= (np.inf if t is None else t)))
            a = pd.concat(parts)
            kl = a["keep"] & (a["is_null"] == 0)
            kn = a["keep"] & (a["is_null"] == 1)
            o[f"test_at_{name}"] = dict(coverage=round(float(a["keep"].mean()), 4),
                                        selective_accuracy=round(float(a.loc[a["keep"], "correct"].mean()), 4) if a["keep"].any() else None,
                                        false_link_rate_accepted=round(float(1 - a.loc[kl, "correct"].mean()), 4) if kl.any() else None,
                                        null_precision_accepted=round(float(a.loc[kn, "correct"].mean()), 4) if kn.any() else None,
                                        abstained_share=round(float(1 - a["keep"].mean()), 4))
        nul = g[g["is_null"] == 1]
        o["null_vs_abstain"] = dict(null_decisions=int(len(nul)), null_decision_accuracy=round(float(nul["correct"].mean()), 4) if len(nul) else None,
                                    link_decisions=int((g["is_null"] == 0).sum()),
                                    link_accuracy=round(float(g.loc[g["is_null"] == 0, "correct"].mean()), 4))
        gs = scen[scen["system"] == s]
        sc_o = dict(ece=round(ece(gs["conf"], gs["exact"]), 4), brier=round(brier(gs["conf"], gs["exact"]), 4),
                    risk_coverage=risk_coverage(gs["conf"].to_numpy(), gs["exact"].to_numpy()))
        parts = []
        for f, gf in gs.groupby("fold"):
            t = res["calib"][f][s]["scen_exact90"]
            parts.append(gf.assign(keep=gf["conf"] >= (np.inf if t is None else t)))
        a = pd.concat(parts)
        sc_o["test_at_scen_exact90"] = dict(coverage=round(float(a["keep"].mean()), 4),
                                            selective_exact=round(float(a.loc[a["keep"], "exact"].mean()), 4) if a["keep"].any() else None)
        o["scenario_level"] = sc_o
        # strata (domain-shift proxies) on decisions
        g2 = g.merge(st.reset_index().rename(columns={"index": "k"})[["k", "est_band", "gap_band", "N_band", "difficulty", "label_risk"]], on="k")
        o["strata"] = {}
        for by in ("est_band", "gap_band", "N_band", "difficulty", "label_risk"):
            o["strata"][by] = {}
            for lv, h in g2.groupby(by):
                parts = []
                for f, gf in h.groupby("fold"):
                    t = res["calib"][f][s]["dec_acc95"]
                    parts.append(gf.assign(keep=gf["conf"] >= (np.inf if t is None else t)))
                a = pd.concat(parts)
                o["strata"][by][str(lv)] = dict(n=int(len(h)), accuracy=round(float(h["correct"].mean()), 4), mean_conf=round(float(h["conf"].mean()), 4),
                                                ece=round(ece(h["conf"], h["correct"]), 4), coverage_at_acc95=round(float(a["keep"].mean()), 4),
                                                selective_accuracy_at_acc95=round(float(a.loc[a["keep"], "correct"].mean()), 4) if a["keep"].any() else None)
        out[s] = o
    sr = res["stress"]
    if len(sr):
        parts = []
        for f, gf in sr.groupby("fold"):
            t = res["calib"][f]["L2_hgb"]["dec_acc95"]
            parts.append(gf.assign(keep=gf["conf"] >= (np.inf if t is None else t)))
        a = pd.concat(parts)
        out["stress_noise"] = dict(sigma_m=NOISE_M, decisions=int(len(sr)), accuracy=round(float(sr["correct"].mean()), 4),
                                   baseline_accuracy_same_decisions=round(float(sr["base_correct"].mean()), 4),
                                   exact=round(float(sr.drop_duplicates("k")["exact"].mean()), 4), ece=round(ece(sr["conf"], sr["correct"]), 4),
                                   brier=round(brier(sr["conf"], sr["correct"]), 4), mean_conf=round(float(sr["conf"].mean()), 4),
                                   coverage_at_acc95=round(float(a["keep"].mean()), 4),
                                   selective_accuracy_at_acc95=round(float(a.loc[a["keep"], "correct"].mean()), 4) if a["keep"].any() else None)
    return out


# ---------------------------------------------------------------- dataset quality, coverage and shortcut audit (descriptive)

def coverage(idx: pd.DataFrame, min_enc: int = 30, min_matches: int = 10) -> dict[str, Any]:
    """Coverage tables with a credibility flag: a cell supports conclusions only with >= ``min_enc`` distinct encounters from
    >= ``min_matches`` matches (variants of one encounter are not independent)."""
    st = strata([None] * len(idx), idx)
    out: dict[str, Any] = {}
    for by in ("NxM", "N", "M", "nulls", "news", "shape", "gap_band", "est_band", "variant", "difficulty"):
        g = st.groupby(by)
        t = pd.DataFrame(dict(scenarios=g.size(), encounters=g["encounter"].nunique(), matches=g["match_id"].nunique()))
        t["credible"] = (t["encounters"] >= min_enc) & (t["matches"] >= min_matches)
        out[by] = {str(k): {c: (bool(v) if c == "credible" else int(v)) for c, v in r.items()} for k, r in t.iterrows()}
    out["N_by_nulls"] = st.groupby(["N_band", "nulls"])["encounter"].nunique().unstack(fill_value=0).to_dict()
    out["N_by_news"] = st.groupby(["N_band", "news"])["encounter"].nunique().unstack(fill_value=0).to_dict()
    out["per_match"] = dict(st.groupby("match_id")["encounter"].nunique().describe().round(2))
    out["rule"] = f"credible = >= {min_enc} distinct encounters and >= {min_matches} matches"
    return out


def audit(sc: list[dict[str, Any]], idx: pd.DataFrame, fold: np.ndarray) -> dict[str, Any]:
    st = strata(sc, idx)
    q: dict[str, Any] = dict(scenarios=len(sc), encounters=int(st["encounter"].nunique()), matches=int(st["match_id"].nunique()))
    q["scenarios_per_encounter"] = C.q(st.groupby("encounter").size())
    # count shortcut: null presence given the shape
    q["count_shortcut"] = {sh: round(float((st.loc[st["shape"] == sh, "n_null"] > 0).mean()), 4) for sh in ("M<N", "M=N", "M>N") if (st["shape"] == sh).any()}
    # timing: distribution of (t_first - t_end) for true vs wrong pairs
    tt, tw, li, lo_t, lo_w = [], [], [], [], []
    for d in sc:
        for i in d["incoming"]:
            for o in d["outgoing"]:
                dt = o["t"][0] - i["t"][-1]
                if d["truth"][i["id"]] == o["id"]:
                    tt.append(dt)
                    lo_t.append(len(o["t"]))
                else:
                    tw.append(dt)
                    lo_w.append(len(o["t"]))
    q["timing_dt_true"], q["timing_dt_wrong"] = C.q(np.array(tt)), C.q(np.array(tw))
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.ones(len(tt)), np.zeros(len(tw))]
    q["timing_auc_true_vs_wrong"] = round(float(roc_auc_score(y, -np.abs(np.r_[tt, tw] - np.median(np.r_[tt, tw])))), 4)
    q["outgoing_length_auc_true_vs_wrong"] = round(float(roc_auc_score(y, np.r_[lo_t, lo_w])), 4)
    q["outgoing_id_is_sorted"] = all([o["id"] for o in d["outgoing"]] == sorted(o["id"] for o in d["outgoing"]) for d in sc)
    # duplicates: encounters / events sharing >= 2 members within overlapping windows (same match and period)
    ev = pd.DataFrame([dict(e=st["encounter"].iloc[k], m=d["meta"]["match"], p=d["meta"]["period"], t=d["meta"]["t_centre"],
                            mem=frozenset(d["meta"]["members"])) for k, d in enumerate(sc)]).drop_duplicates("e")
    w = 2 * (0.9 + 2.0) if "encounter_id" in idx else 6.8
    dup = 0
    for _, g in ev.sort_values("t").groupby(["m", "p"]):
        r = g.to_dict("records")
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                if r[j]["t"] - r[i]["t"] >= w:
                    break
                dup += len(r[i]["mem"] & r[j]["mem"]) >= 2
    q["overlapping_event_pairs_sharing_2_members"] = int(dup)
    q["events"] = int(len(ev))
    # fold overlap: matches never shared; players (provider ids) seen in other folds
    pids = [[h.get("player_id") for h in d["meta"]["hidden"].values()] for d in sc]
    if any(p is not None for ps in pids for p in ps):
        po = {}
        for f in np.unique(fold):
            te = {p for k in np.flatnonzero(fold == f) for p in pids[k]}
            tr = {p for k in np.flatnonzero(fold != f) for p in pids[k]}
            po[int(f)] = round(len(te & tr) / max(len(te), 1), 4)
        q["test_players_seen_in_other_folds"] = po
    mf = pd.DataFrame(dict(m=st["match_id"], f=fold)).drop_duplicates()
    q["matches_in_more_than_one_fold"] = int((mf.groupby("m").size() > 1).sum())
    q["label_uncertainty"] = dict(label_risk=round(float(idx["label_risk"].mean()), 4), estimated_comotion=round(float(idx["estimated_comotion"].mean()), 4))
    q["estimated"] = dict(mean=round(float(idx["est_mean"].mean()), 4), share_max_track_over_50pct=round(float((idx["est_max"] > 0.5).mean()), 4),
                          share_all_visible=round(float((idx["est_max"] == 0).mean()), 4))
    rep = "repeated_encounter" if "repeated_encounter" in idx else "repeated_interaction"
    e1 = idx.drop_duplicates("encounter_id" if "encounter_id" in idx else "event_id")
    q["repeated_flag"] = dict(column=rep, share_of_events=round(float(e1[rep].mean()), 4))
    if "n_catalogue_groups" in idx:
        q["catalogue_rows_per_encounter"] = C.q(e1["n_catalogue_groups"])
        q["prior_gap_s"] = C.q(e1["prior_gap_s"].dropna())
    q["motion_plausibility"] = dict(max_speed_over_12=round(float((idx["max_speed_m_s"] > 12).mean()), 4))
    return q


# ---------------------------------------------------------------- main

def _load(ds: str):
    from . import nxm_dataset as ND
    from . import nxm_dataset_v2 as N2
    mod = {"v1": ND, "v2": N2}[ds]
    sc = mod.load()
    idx = pd.read_parquet(C.PROC / mod.OUT / "index.parquet")
    if list(idx["scenario_id"]) != [d["scenario_id"] for d in sc]:
        raise SystemExit(f"{ds}: index and scenarios are not aligned")
    return sc, idx, mod.OUT


def main(datasets=("v2", "v1")) -> int:
    out = C.Out(OUT)
    t0 = time.perf_counter()
    summary: dict[str, Any] = {}
    for ds in datasets:
        sc, idx, src = _load(ds)
        fold = NC.folds_of(sc)
        print(f"{ds}: {len(sc)} scenarios", flush=True)
        q = audit(sc, idx, fold)
        cov = coverage(idx)
        out.json(f"{ds}/quality.json", q)
        out.json(f"{ds}/coverage.json", cov)
        res = run(sc, fold, log=lambda s: print(s, flush=True))
        st = strata(sc, idx)
        t = tables(res["rows"], st)
        cal = calibration_tables(res, st)
        out.json(f"{ds}/benchmark.json", dict(schema=OUT, dataset=src, label="EXPLORATORY (provider-derived truth, not human-reviewed, not frozen)",
                                              tables=t, chosen_parameters=res["chosen"], runtime_s=res["runtime_s"]))
        out.json(f"{ds}/calibration.json", dict(schema=OUT, dataset=src, thresholds_from_calibration_folds=res["calib"], tables=cal))
        per = pd.DataFrame(dict(scenario_id=[d["scenario_id"] for d in sc], match_id=st["match_id"], fold=fold,
                                **{f"{k}_exact": [r["exact"] for r in v] for k, v in res["rows"].items()}))
        out.parquet(f"{ds}/per_scenario.parquet", per)
        out.parquet(f"{ds}/test_decisions.parquet", res["dec"])
        summary[ds] = dict(overall={k: v.get("exact") for k, v in t["overall"].items()}, runtime_s=res["runtime_s"])
        print(json.dumps(summary[ds], indent=1), flush=True)
    here = Path(__file__).resolve().parent
    out.json("summary.json", dict(schema=OUT, summary=summary, runtime_s=round(time.perf_counter() - t0, 1)))
    out.manifest(inputs={ds: f"data/processed/{_load_name(ds)}" for ds in datasets},
                 parameters=dict(n_folds=N_FOLDS, calibration_folds="(f+1, f+2) mod 8", taus=TAUS, betas=BETAS, sigmas=NC.SIGMAS, nulls=NC.NULLS,
                                 noise_m=NOISE_M, tune_subset=TUNE_SUB, target_decision_accuracy=TARGET_ACC, target_scenario_exact=TARGET_EXACT,
                                 features=NC.FEATURES, shortcut_features=SC_FEATURES, texture_features=TEX_FEATURES),
                 code={p: C.sha256_file(here / p) for p in ("nxm_assoc_v2.py", "nxm_compare.py", "nxm_dataset_v2.py", "nxm_dataset.py", "multiplayer.py")})
    return 0


def _load_name(ds: str) -> str:
    return {"v1": "m11-nxm-exploratory-v1", "v2": "m11-nxm-exploratory-v2"}[ds]
