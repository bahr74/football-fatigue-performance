"""M11 step 5: diagnosis, football-aware features, bounded model comparison and operating points on task B.

    python -m football_fatigue.matching ml2 --matches 3812 3816 10506 3814 3847 3836 3833 3815 3851 3829

Builds on :mod:`.ml` (step 4) without changing it: the same verified ``m11-matching-v3`` development inputs, the same
matcher-visible features, label protocol, one-to-one log-odds assignment and MATCHED / UNMATCHED / UNCERTAIN decisions.

Protocol (every number is agreement with the provider identity reference, not verified identity accuracy):

- outer leave-one-match-out over the development matches given; the held-out match is touched only for scoring;
- inside each outer fold, the training matches are split into ``inner_groups`` match groups. Out-of-fold predictions of
  that inner split choose the hyperparameters (a bounded, a-priori grid, by pooled log-loss), fit the Platt calibrator and
  choose the operating point (threshold x margin) that maximises coverage under each false-association target;
- feature ablations, learning curves and grouped permutation importance use fixed a-priori configurations and are
  match-disjoint;
- uncertainty is reported per match and by a match-level bootstrap, never by treating pairs or encounters as independent.

Validation and test matches are refused. No synthetic swap is injected. FOOTBALLTrace-inspired features (family F) are an
ablation only; they are not the published FOOTBALLTrace correction algorithm.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import time
import uuid
import warnings
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from football_fatigue.encounters.splits import MANIFEST_NAME
from football_fatigue.ingestion import IngestionError, approved_output_root

from . import ml
from .baselines import solve
from .benchmark import TASK_B, _sha256, code_digest
from .evaluation import evaluate, gap_bin

SCHEMA_VERSION = OUTPUT_DIR = "m11-ml-v2"
LABEL_TIER, EXPLORATORY_TIER, UNVERIFIED_TIER = ml.LABEL_TIER, ml.EXPLORATORY_TIER, ml.UNVERIFIED_TIER

FOOTBALL_FEATURES: dict[str, str] = {
    "hermite_peak_speed": "peak speed along the cubic Hermite path joining (p_in, v_in) to (p_out, v_out) (m/s); above max_speed is implausible",
    "ca_bwd_err": "constant-acceleration backward extrapolation error from the outgoing start, outgoing acceleration capped (m)",
    "ca_mid_err": "distance between the forward and backward constant-acceleration half-gap extrapolations (m)",
    "turn_in_deg": "angle between v_in and the junction displacement (deg; 0 when slower than moving_m_s or the displacement is under 0.5 m)",
    "turn_out_deg": "angle between the junction displacement and v_out (deg; same rule)",
    "turn_acc_demand": "(speed_in * turn_in + speed_out * turn_out) / gap, angles in radians: turning demand onto and off the junction "
                       "(m/s^2); a sharp turn at low speed is cheap, so a sharp turn is not treated as an error by itself",
    "speed_consistency": "|dist_per_s - (speed_in + speed_out) / 2|: mean junction speed against the endpoint speeds (m/s)",
    "in_sep_m": "distance from the incoming end to the nearest other incoming end (m; rel_cap if alone)",
    "in_closing_m_s": "closing speed of the incoming player and that nearest other incoming player at their ends (m/s; positive converging)",
    "out_sep_m": "distance from the outgoing start to the nearest other outgoing start (m; rel_cap if alone)",
    "n_reach_row": "reachable outgoing candidates of the incoming tracklet",
    "n_reach_col": "reachable incoming candidates of the outgoing tracklet",
    "assign_regret_cv": "best total capped cv_mid_err of a one-to-one assignment forced to contain this pair minus the unconstrained best (m)",
    "assign_regret_hermite": "as assign_regret_cv with capped hermite_rms_acc costs (m/s^2)",
}
G_FEATURES = tuple(FOOTBALL_FEATURES)
FEATURES2 = ml.ALL_FEATURES + G_FEATURES

FAMILIES: dict[str, tuple[str, ...]] = {
    "A_spatial_reach": ("gap_s", "dist_m", "dist_per_s", "reach_ratio"),
    "B_velocity_extrapolation": ("speed_in", "speed_out", "speed_diff", "vel_diff", "cv_fwd_err", "cv_bwd_err", "cv_mid_err", "ca_fwd_err"),
    "C_heading_acc_turn": ("heading_diff_deg", "both_moving", "req_acc", "hermite_max_acc", "hermite_rms_acc", "acc_in", "acc_out",
                           "turn_rate_in", "turn_rate_out", "straight_in", "straight_out"),
    "D_quality_coverage": ("n_upd_in", "n_upd_out", "est_share_in", "est_share_out", "conf_in", "conf_out"),
    "E_competitive": ("n_incoming", "n_outgoing") + tuple(f"{s}_{k}" for k in ml.REL_KEYS for s in ("row_gap", "col_gap")),
    "F_footballtrace": ml.FT_FEATURES,
    "G_football_aware": G_FEATURES,
}
CORE = tuple(f for fam in ("A_spatial_reach", "B_velocity_extrapolation", "C_heading_acc_turn", "D_quality_coverage", "E_competitive") for f in FAMILIES[fam])
FEATURE_SETS = dict(core=CORE, core_g=CORE + G_FEATURES, core_f=CORE + ml.FT_FEATURES, core_fg=CORE + ml.FT_FEATURES + G_FEATURES,
                    distance=ml.DISTANCE_FEATURES)

# bounded a-priori grids (chosen before any step-5 result); "lr" and "rf" are the unchanged step-4 configurations
GRIDS: dict[str, list[dict]] = {
    "lr": [dict()],
    "rf": [dict()],
    "rf_reg": [dict(min_samples_leaf=a, max_features=b) for a, b in product((5, 20, 50), ("sqrt", 0.33))],
    "hgb": [dict(learning_rate=0.1, max_leaf_nodes=b, min_samples_leaf=c) for b, c in product((7, 15), (20, 50))],
    "mlp": [dict(hidden_layer_sizes=a, alpha=b) for a, b in product(((32,), (32, 16)), (1e-3, 1e-2, 1e-1))],
}
FIXED = dict(lr=dict(), rf=dict(), hgb=dict(learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=20))   # ablations, curves, importance
COMPARISON = (("lr", "core"), ("rf", "core"), ("rf_reg", "core"), ("hgb", "core"), ("mlp", "core"),
              ("lr", "core_g"), ("rf", "core_g"), ("rf_reg", "core_g"), ("hgb", "core_g"), ("mlp", "core_g"))
BASELINE_KEY = "rf/core"     # the step-4 Random Forest
EVIDENCE_KEYS = ml.EVIDENCE_KEYS + ("turn_acc_demand", "assign_regret_cv")


@dataclass(frozen=True)
class ML2Config:
    base: ml.MLConfig = ml.MLConfig()
    inner_groups: int = 3
    targets: tuple = (0.05, 0.10)
    threshold_grid: tuple = tuple(round(0.5 + 0.01 * k, 2) for k in range(50))
    margin_grid: tuple = (0.0, 0.25, 0.5, 1.0, 2.0)
    hgb_iter: int = 200
    n_jobs_folds: int = 4         # outer folds run in parallel processes; results do not depend on it
    n_jobs_model: int = 2         # forest threads per process
    mlp_iter: int = 400
    curve_sizes: tuple = (1, 2, 3, 5, 7)
    curve_reps: int = 3
    perm_repeats: int = 3
    bootstrap: int = 2000
    coverage_levels: tuple = (0.2, 0.3, 0.4, 0.5)
    # a-priori selection rule: the highest pooled held-out coverage at the 5% target operating point among models whose pooled
    # held-out false-association rate is <= selection_fa_cap; "improves" also needs the match-bootstrap 95% interval of the
    # coverage difference to the step-4 RF to exclude zero while the false-association difference interval stays <= fa_tolerance
    selection_fa_cap: float = 0.075
    fa_tolerance: float = 0.02

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------- football-aware features (matcher-visible instance only)

def hermite_peak_speed(pa: np.ndarray, va: np.ndarray, pb: np.ndarray, vb: np.ndarray, gap: float, n: int = 21) -> float:
    s = np.linspace(0, 1, n)[:, None]
    v = ((6 * s ** 2 - 6 * s) * pa + (3 * s ** 2 - 4 * s + 1) * gap * va + (6 * s - 6 * s ** 2) * pb + (3 * s ** 2 - 2 * s) * gap * vb) / gap
    return float(np.hypot(v[:, 0], v[:, 1]).max())


def _angle(u: np.ndarray, w: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(u @ w / (np.hypot(*u) * np.hypot(*w)), -1, 1))))


def assignment_regret(C: np.ndarray, cap: float) -> np.ndarray:
    """Per pair: best total of a one-to-one assignment (abstaining costs ``cap``) forced to contain it, minus the best total."""
    C = np.minimum(np.nan_to_num(C, nan=cap, posinf=cap), cap)
    n, m = C.shape
    abstain = np.full(n, cap)
    best, pick = solve(C, abstain)
    out = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            if pick[i] == j:
                continue
            F = C.copy()
            F[i, :], F[:, j] = np.inf, np.inf
            F[i, j] = C[i, j]
            ab = abstain.copy()
            ab[i] = np.inf
            out[i, j] = solve(F, ab)[0] - best
    return out


def football_features(inst: dict, cfg: ml.MLConfig = ml.MLConfig()) -> np.ndarray:
    """Family G for every pair of one task B instance, in :func:`ml.instance_features` row order (incoming-major)."""
    if inst.get("task") != TASK_B or inst["during"]["t"]:
        raise ValueError("features are computed on task B instances only (no interval observations)")
    I, O = inst["incoming"], inst["outgoing"]
    SA = [ml.tracklet_summary(a, True, cfg) for a in I]
    SB = [ml.tracklet_summary(b, False, cfg) for b in O]
    n, m = len(SA), len(SB)
    if not n or not m:
        return np.zeros((0, len(G_FEATURES)))

    def nearest(S: list[dict], k: int) -> int | None:
        d = [np.hypot(*(S[q]["p"] - S[k]["p"])) if q != k else np.inf for q in range(len(S))]
        return int(np.argmin(d)) if len(S) > 1 else None

    vals = [[{} for _ in range(m)] for _ in range(n)]
    cvm, her, reach = np.zeros((n, m)), np.zeros((n, m)), np.zeros((n, m), bool)
    for i, a in enumerate(SA):
        q = nearest(SA, i)
        if q is None:
            in_sep, closing = cfg.rel_cap, 0.0
        else:
            r = SA[q]["p"] - a["p"]
            in_sep = float(np.hypot(*r))
            closing = float(-(SA[q]["v"] - a["v"]) @ r / in_sep) if in_sep > 1e-6 else 0.0
        for j, b in enumerate(SB):
            base = ml.pair_values(a, b, cfg)
            gap = base["gap_s"]
            D = b["p"] - a["p"]
            dist = base["dist_m"]
            ti = _angle(a["v"], D) if a["speed"] >= cfg.moving_m_s and dist >= 0.5 else 0.0
            to = _angle(D, b["v"]) if b["speed"] >= cfg.moving_m_s and dist >= 0.5 else 0.0
            h = gap / 2
            fwd = a["p"] + a["v"] * h + 0.5 * a["acc"] * h ** 2
            bwd = b["p"] - b["v"] * h + 0.5 * b["acc"] * h ** 2
            qo = nearest(SB, j)
            vals[i][j] = dict(
                hermite_peak_speed=hermite_peak_speed(a["p"], a["v"], b["p"], b["v"], gap),
                ca_bwd_err=float(np.hypot(*(b["p"] - b["v"] * gap + 0.5 * b["acc"] * gap ** 2 - a["p"]))),
                ca_mid_err=float(np.hypot(*(fwd - bwd))),
                turn_in_deg=ti, turn_out_deg=to,
                turn_acc_demand=(a["speed"] * np.radians(ti) + b["speed"] * np.radians(to)) / gap,
                speed_consistency=abs(base["dist_per_s"] - (a["speed"] + b["speed"]) / 2),
                in_sep_m=in_sep, in_closing_m_s=closing,
                out_sep_m=cfg.rel_cap if qo is None else float(np.hypot(*(SB[qo]["p"] - b["p"]))))
            cvm[i, j], her[i, j], reach[i, j] = base["cv_mid_err"], base["hermite_rms_acc"], base["reach_ratio"] <= 1.0
    rc, rh = assignment_regret(cvm, cfg.rel_cap), assignment_regret(her, cfg.rel_cap)
    X = []
    for i in range(n):
        for j in range(m):
            v = vals[i][j]
            v.update(n_reach_row=float(reach[i].sum()), n_reach_col=float(reach[:, j].sum()), assign_regret_cv=float(rc[i, j]),
                     assign_regret_hermite=float(rh[i, j]))
            X.append([v[k] for k in G_FEATURES])
    X = np.nan_to_num(np.asarray(X, float), nan=0.0, posinf=1e3, neginf=-1e3)
    return np.clip(X, -1e3, 1e3)


def match_table2(data: dict, cfg: ml.MLConfig = ml.MLConfig()) -> dict[str, Any]:
    """:func:`ml.match_table` plus family G, as one matrix in ``FEATURES2`` order."""
    t = ml.match_table(data, cfg)
    G = [football_features(inst, cfg) for inst in data["instances"]]
    G = np.vstack(G) if G else np.zeros((0, len(G_FEATURES)))
    if G.shape[0] != t["X"].shape[0]:
        raise ValueError("feature rows misaligned")
    t["X"] = np.hstack([t["X"], G])
    return t


def cols(features: tuple[str, ...]) -> list[int]:
    return [FEATURES2.index(f) for f in features]


# ---------------------------------------------------------------- models

def make_model(kind: str, params: dict, cfg: ML2Config = ML2Config()):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    b = cfg.base
    if kind == "lr":
        return ml.make_model(kind, b)
    if kind == "rf":                 # the step-4 forest; n_jobs does not change a seeded forest's result
        return ml.make_model(kind, b).set_params(clf__n_jobs=cfg.n_jobs_model)
    if kind == "rf_reg":
        return Pipeline([("clf", RandomForestClassifier(n_estimators=b.rf_trees, class_weight="balanced_subsample", random_state=b.seed, n_jobs=cfg.n_jobs_model, **params))])
    if kind == "hgb":
        return Pipeline([("clf", HistGradientBoostingClassifier(max_iter=cfg.hgb_iter, l2_regularization=1.0, early_stopping=False,
                                                                class_weight="balanced", random_state=b.seed, **params))])
    if kind == "mlp":
        return Pipeline([("scale", StandardScaler()),
                         ("clf", MLPClassifier(max_iter=cfg.mlp_iter, early_stopping=False, random_state=b.seed, **params))])
    raise ValueError(f"unknown model {kind}")


def fit_model(kind: str, params: dict, X: np.ndarray, y: np.ndarray, cfg: ML2Config = ML2Config()):
    if len(set(y.tolist())) < 2:
        raise ValueError("training needs positive and negative pairs")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # MLP convergence notices at the fixed iteration budget
        return make_model(kind, params, cfg).fit(X, y)


def proba(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1] if len(X) else np.zeros(0)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


@dataclass
class Platt:
    a: float = 1.0
    b: float = 0.0

    def __call__(self, p: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.a * _logit(p) + self.b)))


def fit_platt(p: np.ndarray, y: np.ndarray) -> Platt:
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p)[:, None], y)
    return Platt(float(lr.coef_[0, 0]), float(lr.intercept_[0]))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def match_groups(match_ids: list[str], k: int) -> list[list[str]]:
    """Deterministic match partition (sorted IDs dealt round-robin): the inner grouped split."""
    ids = sorted(match_ids, key=lambda m: (len(m), m))
    k = min(k, len(ids))
    return [ids[g::k] for g in range(k)]


def _parallel(fn, items: list, cfg: ML2Config) -> list:
    """``[fn(*item) for item in items]``, in order, over ``n_jobs_folds`` processes (deterministic: each call is seeded)."""
    if cfg.n_jobs_folds <= 1 or len(items) <= 1:
        return [fn(*it) for it in items]
    from joblib import Parallel, delayed

    return Parallel(n_jobs=cfg.n_jobs_folds, backend="loky")(delayed(fn)(*it) for it in items)


# ---------------------------------------------------------------- decisions, vectorised over the operating grid

def decision_table(assigned: dict[str, dict], refs: dict[str, dict]) -> dict[str, np.ndarray]:
    """One row per incoming tracklet of the assigned instances, with what any (threshold, margin) decision needs."""
    pick, p, margin, nop, correct, exp_none, inst = [], [], [], [], [], [], []
    for n, (iid, a) in enumerate(assigned.items()):
        exp = refs[iid]["expected"]
        for i, r in enumerate(a["raw"]):
            e = exp[a["incoming"][i]]
            pick.append(r["pick"] >= 0)
            p.append(-1.0 if r["p"] is None else r["p"])
            margin.append(-1.0 if r["margin"] is None else r["margin"])
            nop.append(r["no_plausible"])
            correct.append(r["pick"] >= 0 and e is not None and a["outgoing"][r["pick"]] == e)
            exp_none.append(e is None)
            inst.append(n)
    A = lambda v, t=float: np.asarray(v, t)   # noqa: E731
    return dict(pick=A(pick, bool), p=A(p), margin=A(margin), no_plausible=A(nop, bool), correct=A(correct, bool), exp_none=A(exp_none, bool),
                inst=A(inst, int), n_inst=len(assigned))


def table_rates(T: dict, thr: float, mrg: float) -> dict[str, Any]:
    if not len(T["p"]):
        return dict(incoming=0, matched=0, correct=0, wrong=0, coverage=None, correct_match_rate=None, false_association_rate=None,
                    complete_assignment_accuracy=None, uncertain_rate=None, unmatched_rate=None)
    matched = T["pick"] & (T["p"] >= thr) & (T["margin"] >= mrg) & ~T["no_plausible"]
    cm, wm = matched & T["correct"], matched & ~T["correct"]
    ok = cm | (T["no_plausible"] & T["exp_none"])
    bad = np.bincount(T["inst"], weights=~ok, minlength=T["n_inst"])
    n, k = len(matched), int(matched.sum())
    return dict(incoming=n, matched=k, correct=int(cm.sum()), wrong=int(wm.sum()), coverage=k / n, correct_match_rate=cm.sum() / n,
                false_association_rate=wm.sum() / k if k else None, complete_assignment_accuracy=float(np.mean(bad == 0)),
                uncertain_rate=float((~matched & ~T["no_plausible"]).mean()), unmatched_rate=float(T["no_plausible"].mean()))


def operating_curve(T: dict, cfg: ML2Config = ML2Config()) -> list[dict]:
    return [dict(threshold=t, margin=m, **table_rates(T, t, m)) for t in cfg.threshold_grid for m in cfg.margin_grid]


def select_operating(T: dict, target: float, cfg: ML2Config = ML2Config()) -> dict[str, Any]:
    """Maximum coverage with false association <= target (ties: the more conservative point); else the most conservative point.
    Called on inner out-of-fold training-match decisions only."""
    curve = operating_curve(T, cfg)
    ok = [c for c in curve if c["false_association_rate"] is not None and c["false_association_rate"] <= target]
    if ok:
        best = max(ok, key=lambda c: (c["coverage"], c["threshold"], c["margin"]))
        return dict(threshold=best["threshold"], margin=best["margin"], target=target, target_met=True, inner_oof=best)
    c = curve[-1]
    return dict(threshold=c["threshold"], margin=c["margin"], target=target, target_met=False, inner_oof=c)


def best_at(curve: list[dict], fa_budget: float | None = None, coverage: float | None = None) -> dict | None:
    """Descriptive only (never used to choose): the best curve point under a false-association budget or a coverage level."""
    if fa_budget is not None:
        ok = [c for c in curve if c["false_association_rate"] is not None and c["false_association_rate"] <= fa_budget]
        return max(ok, key=lambda c: c["coverage"]) if ok else None
    ok = [c for c in curve if c["coverage"] >= coverage and c["false_association_rate"] is not None]
    return min(ok, key=lambda c: c["false_association_rate"]) if ok else None


def _hi(t: dict) -> np.ndarray:
    return ml._tier_mask(t, LABEL_TIER)


def assigned_for(t: dict, p: np.ndarray, mask: np.ndarray, cfg: ML2Config) -> dict[str, dict]:
    return ml.instance_assignments([r for r, k in zip(t["rows"], mask) if k], p[mask], cfg.base)


# ---------------------------------------------------------------- nested fit: hyperparameters, calibration, operating points

def _xy(tables: list[dict], features: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    X, y = ml._stack(tables, ml._train_mask)
    return X[:, cols(features)], y


def nested_fit(kind: str, features: tuple[str, ...], train: list[dict], cfg: ML2Config = ML2Config()) -> dict[str, Any]:
    """Inner grouped split over the training matches only: OOF predictions for every grid point, the configuration with the
    lowest pooled OOF log-loss, a Platt calibrator on its OOF predictions, operating points on its calibrated OOF
    assignments, then a refit on all training matches."""
    groups = match_groups([t["match_id"] for t in train], cfg.inner_groups)
    c = cols(features)
    scores, oof_by = [], []
    for params in GRIDS[kind]:
        oof = {}
        for g in groups:
            fit_t = [t for t in train if t["match_id"] not in g]
            model = fit_model(kind, params, *_xy(fit_t, features), cfg)
            for t in train:
                if t["match_id"] in g:
                    oof[t["match_id"]] = proba(model, t["X"][:, c])
        ys = np.concatenate([np.array([lb["label"] for lb, k in zip(t["labels"], ml._train_mask(t)) if k], int) for t in train])
        ps = np.concatenate([oof[t["match_id"]][ml._train_mask(t)] for t in train])
        scores.append(log_loss(ys, ps))
        oof_by.append(oof)
    k = int(np.argmin(scores))
    params, oof = GRIDS[kind][k], oof_by[k]
    ys = np.concatenate([np.array([lb["label"] for lb, kk in zip(t["labels"], ml._train_mask(t)) if kk], int) for t in train])
    ps = np.concatenate([oof[t["match_id"]][ml._train_mask(t)] for t in train])
    cal = fit_platt(ps, ys)
    assigned, refs = {}, {}
    for t in train:
        assigned |= assigned_for(t, cal(oof[t["match_id"]]), _hi(t), cfg)
        refs |= t["refs"]
    T = decision_table(assigned, refs)
    ops = {str(tg): select_operating(T, tg, cfg) for tg in cfg.targets}
    model = fit_model(kind, params, *_xy(train, features), cfg)
    return dict(model=model, params=params, grid_logloss=[round(s, 5) for s in scores], calibrator=cal, operating=ops,
                inner_groups=groups, inner_oof_logloss_raw=round(scores[k], 5), inner_oof_logloss_platt=round(log_loss(ys, cal(ps)), 5))


def _outer_fold(kind: str, fs: str, tables: list[dict], h: int, cfg: ML2Config) -> tuple[np.ndarray, dict]:
    train = [u for m, u in enumerate(tables) if m != h]
    nf = nested_fit(kind, FEATURE_SETS[fs], train, cfg)
    return proba(nf["model"], tables[h]["X"][:, cols(FEATURE_SETS[fs])]), nf


def run_comparison(tables: list[dict], cfg: ML2Config = ML2Config(), comparison=COMPARISON, log=print) -> dict[str, Any]:
    """Outer leave-one-match-out over every (model, feature set). Held-out matches are never seen by fitting or selection."""
    if len(tables) < 4:
        raise ValueError("nested match-level evaluation needs at least four development matches")
    ids = [t["match_id"] for t in tables]
    iids = [i for t in tables for i in t["refs"]]
    if len(set(ids)) != len(ids) or len(set(iids)) != len(iids):
        raise ValueError("instance or match IDs repeat across matches")
    res = {}
    for kind, fs in comparison:
        key = f"{kind}/{fs}"
        r = dict(kind=kind, features=fs, p_raw={}, p={}, folds={}, decisions={})
        t0 = time.perf_counter()
        fits = _parallel(_outer_fold, [(kind, fs, tables, h, cfg) for h in range(len(tables))], cfg)
        for t, (raw, nf) in zip(tables, fits):
            r["p_raw"][t["match_id"]], r["p"][t["match_id"]] = raw, nf["calibrator"](raw)
            r["folds"][t["match_id"]] = nf
        log(f"{key}: {time.perf_counter() - t0:.0f} s")
        res[key] = r
    return res


def final_fit(kind: str, fs: str, tables: list[dict], cfg: ML2Config = ML2Config()) -> dict[str, Any]:
    """The deployable development model: the same nested procedure over all development matches."""
    return nested_fit(kind, FEATURE_SETS[fs], tables, cfg)


# ---------------------------------------------------------------- evaluation of the held-out predictions

def heldout_tables(tables: list[dict], r: dict, cfg: ML2Config, tier: str = LABEL_TIER) -> dict[str, dict]:
    out = {}
    for t in tables:
        m = ml._tier_mask(t, tier)
        out[t["match_id"]] = decision_table(assigned_for(t, r["p"][t["match_id"]], m, cfg), t["refs"])
    return out


def _concat(Ts: list[dict]) -> dict:
    off, inst = 0, []
    for T in Ts:
        inst.append(T["inst"] + off)
        off += T["n_inst"]
    return {k: (np.concatenate([T[k] for T in Ts]) if k not in ("inst", "n_inst") else None) for k in Ts[0]} | dict(inst=np.concatenate(inst), n_inst=off)


def operating_results(tables: list[dict], r: dict, cfg: ML2Config) -> dict[str, Any]:
    """Held-out encounter-level rates at each fold's inner-chosen operating point, pooled and per match."""
    out = {}
    for tier in (LABEL_TIER, EXPLORATORY_TIER):
        Ts = heldout_tables(tables, r, cfg, tier)
        per = {}
        for tg in cfg.targets:
            pm = {mid: table_rates(T, r["folds"][mid]["operating"][str(tg)]["threshold"], r["folds"][mid]["operating"][str(tg)]["margin"]) for mid, T in Ts.items()}
            tot = {k: sum(v[k] for v in pm.values()) for k in ("incoming", "matched", "correct", "wrong")}
            pooled = dict(tot, coverage=tot["matched"] / tot["incoming"], correct_match_rate=tot["correct"] / tot["incoming"],
                          false_association_rate=tot["wrong"] / tot["matched"] if tot["matched"] else None)
            per[str(tg)] = dict(pooled=pooled, by_match=pm)
        out[tier] = per
    return out


def bootstrap(per_match: dict[str, dict], other: dict[str, dict] | None, n: int, seed: int) -> dict[str, Any]:
    """Match-level bootstrap of pooled coverage and false association (and of the difference to ``other``)."""
    ids = sorted(per_match)
    rng = np.random.default_rng(seed)
    a = np.array([[per_match[m][k] for k in ("incoming", "matched", "wrong")] for m in ids], float)
    b = None if other is None else np.array([[other[m][k] for k in ("incoming", "matched", "wrong")] for m in ids], float)
    cov, fa, dcov, dfa = [], [], [], []
    for _ in range(n):
        s = rng.integers(0, len(ids), len(ids))
        A = a[s].sum(0)
        cov.append(A[1] / A[0])
        fa.append(A[2] / A[1] if A[1] else np.nan)
        if b is not None:
            B = b[s].sum(0)
            dcov.append(A[1] / A[0] - B[1] / B[0])
            dfa.append((A[2] / A[1] if A[1] else np.nan) - (B[2] / B[1] if B[1] else np.nan))
    q = lambda v: [ml._r(np.nanpercentile(v, 2.5)), ml._r(np.nanpercentile(v, 97.5))]   # noqa: E731
    out = dict(coverage_ci95=q(cov), false_association_ci95=q(fa))
    if b is not None:
        out |= dict(coverage_diff_ci95=q(dcov), false_association_diff_ci95=q(dfa))
    return out


def pairwise_block(tables: list[dict], r: dict, cfg: ML2Config, key: str = "p") -> dict[str, Any]:
    out = {}
    for tier in (LABEL_TIER, EXPLORATORY_TIER):
        ys, ps, rows = [], [], []
        for t in tables:
            m = ml._tier_mask(t, tier)
            ys.append(np.array([lb["label"] for lb, k in zip(t["labels"], m) if k], int))
            ps.append(r[key][t["match_id"]][m])
            rows += [x for x, k in zip(t["rows"], m) if k]
            if tier == LABEL_TIER:
                pw = ml.pairwise_metrics(ys[-1], ps[-1], [x for x, k in zip(t["rows"], m) if k], cfg.base.reliability_bins)
                out.setdefault("high_by_match", {})[t["match_id"]] = {k2: v for k2, v in pw.items() if k2 != "reliability"} | dict(log_loss=ml._r(log_loss(ys[-1], ps[-1])))
        pw = ml.pairwise_metrics(np.concatenate(ys), np.concatenate(ps), rows, cfg.base.reliability_bins)
        out[tier] = pw | dict(log_loss=ml._r(log_loss(np.concatenate(ys), np.concatenate(ps))))
    return out


def strata_eval(tables: list[dict], r: dict, cfg: ML2Config, target: float, name: str) -> tuple[dict, list[dict]]:
    """Decisions at the inner-chosen operating point, scored with the step-2 evaluator (strata included)."""
    preds = []
    for t in tables:
        op = r["folds"][t["match_id"]]["operating"][str(target)]
        assigned = ml.instance_assignments(t["rows"], r["p"][t["match_id"]], cfg.base)
        ds = ml.decisions(assigned, op["threshold"], replace(cfg.base, margin_logit=op["margin"]))
        for iid, dl in ds.items():                     # evidence for the chosen or candidate pair
            a = assigned[iid]
            for i, d in enumerate(dl):
                kk = a["rows"][i, a["outgoing"].index(d["candidate"])] if d["candidate"] is not None else None
                d["evidence"] = None if kk is None else {f: ml._r(t["X"][kk, FEATURES2.index(f)], 3) for f in EVIDENCE_KEYS}
        preds += [dict(baseline=name, match_id=t["match_id"], instance_id=iid, threshold=op["threshold"], margin=op["margin"], decisions=d) for iid, d in ds.items()]
    refs = [x for t in tables for x in t["refs"].values()]
    out = {}
    for tier in (LABEL_TIER, EXPLORATORY_TIER):
        rs = [x for x in refs if x["tier"] == tier]
        ids = {x["instance_id"] for x in rs}
        out[tier] = evaluate([p for p in preds if p["instance_id"] in ids], rs, methods=(name,))[name]
    return out, preds


# ---------------------------------------------------------------- diagnostics (step-4 LR and RF, fixed configurations)

def _auc(y: np.ndarray, p: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score

    return ml._r(roc_auc_score(y, p)) if len(set(y.tolist())) == 2 else None


def _f1(y: np.ndarray, p: np.ndarray) -> float | None:
    yh = p >= 0.5
    tp, fp, fn = (yh & (y == 1)).sum(), (yh & (y == 0)).sum(), (~yh & (y == 1)).sum()
    return ml._r(2 * tp / (2 * tp + fp + fn)) if tp else None


def _heldout_clean(t: dict, c: list[int]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    m = ml._train_mask(t)
    return t["X"][m][:, c], np.array([lb["label"] for lb, k in zip(t["labels"], m) if k], int), [r for r, k in zip(t["rows"], m) if k]


def _diag_fold(kind: str, features: str, tables: list[dict], h: int, cfg: ML2Config) -> dict[str, Any]:
    from sklearn.model_selection import GroupKFold

    train = [u for m, u in enumerate(tables) if m != h]
    X, y = _xy(train, FEATURE_SETS[features])
    g = np.array([r["instance_id"] for u in train for r, k in zip(u["rows"], ml._train_mask(u)) if k])
    model = fit_model(kind, FIXED[kind], X, y, cfg)
    ptr = proba(model, X)
    cv = np.zeros(len(y))
    for a, b in GroupKFold(5).split(X, y, g):
        cv[b] = proba(fit_model(kind, FIXED[kind], X[a], y[a], cfg), X[b])
    oob = None
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier

        rf = RandomForestClassifier(n_estimators=cfg.base.rf_trees, min_samples_leaf=cfg.base.rf_min_leaf, class_weight="balanced_subsample",
                                    random_state=cfg.base.seed, n_jobs=cfg.n_jobs_model, oob_score=True).fit(X, y)
        oob = _auc(y, rf.oob_decision_function_[:, 1])
    Xh, yh, rows = _heldout_clean(tables[h], cols(FEATURE_SETS[features]))
    ph = proba(model, Xh)
    pm = ml.pairwise_metrics(yh, ph, rows)
    return dict(train_pairs=int(len(y)), train_auc=_auc(y, ptr), train_f1=_f1(y, ptr), train_logloss=ml._r(log_loss(y, ptr)),
                within_training_encounter_cv_auc=_auc(y, cv), within_training_encounter_cv_f1=_f1(y, cv),
                within_training_encounter_cv_logloss=ml._r(log_loss(y, cv)), oob_auc=oob, heldout_pairs=int(len(yh)), heldout_auc=pm["roc_auc"],
                heldout_f1=pm["f1"], heldout_brier=pm["brier"], heldout_ece=pm["ece"], heldout_logloss=ml._r(log_loss(yh, ph)), heldout_top1=pm["top1_incoming"])


def fold_diagnostics(tables: list[dict], cfg: ML2Config = ML2Config(), features: str = "core") -> dict[str, Any]:
    """Per outer fold, fixed step-4 configurations: in-sample training AUC/F1, a within-training grouped-by-encounter 5-fold
    estimate (same matches, unseen encounters), RF out-of-bag AUC, and held-out-match AUC/F1/Brier/log-loss/top-1."""
    out = {}
    for kind in ("lr", "rf", "hgb"):
        res = _parallel(_diag_fold, [(kind, features, tables, h, cfg) for h in range(len(tables))], cfg)
        per = {t["match_id"]: r for t, r in zip(tables, res)}
        keys = [k for k in res[0] if k not in ("train_pairs", "heldout_pairs") and res[0][k] is not None]
        out[kind] = dict(by_fold=per, mean={k: ml._r(np.mean([v[k] for v in per.values() if v[k] is not None])) for k in keys})
    return out


def _curve_point(kind: str, features: str, tables: list[dict], h: int, sub: tuple, cfg: ML2Config) -> dict | None:
    X, y = _xy([tables[m] for m in sub], FEATURE_SETS[features])
    if len(set(y.tolist())) < 2:
        return None
    model = fit_model(kind, FIXED[kind], X, y, cfg)
    Xh, yh, _ = _heldout_clean(tables[h], cols(FEATURE_SETS[features]))
    ph, ptr = proba(model, Xh), proba(model, X)
    return dict(heldout=tables[h]["match_id"], n_matches=len(sub), train_pairs=int(len(y)), train_auc=_auc(y, ptr), heldout_auc=_auc(yh, ph),
                train_logloss=ml._r(log_loss(y, ptr)), heldout_logloss=ml._r(log_loss(yh, ph)))


def learning_curves(tables: list[dict], cfg: ML2Config = ML2Config(), features: str = "core") -> dict[str, Any]:
    """Held-out-match AUC and log-loss against the number of training matches (seeded random match subsets)."""
    rng = np.random.default_rng(cfg.base.seed)
    sizes = [k for k in cfg.curve_sizes if k < len(tables) - 1] + [len(tables) - 1]
    plan = []
    for h in range(len(tables)):
        others = [m for m in range(len(tables)) if m != h]
        for k in sizes:
            for _ in range(1 if k == len(others) else cfg.curve_reps):
                plan.append((h, tuple(sorted(rng.choice(others, k, replace=False).tolist()))))
    out = {}
    for kind in ("lr", "rf", "hgb"):
        pts = [p for p in _parallel(_curve_point, [(kind, features, tables, h, sub, cfg) for h, sub in plan], cfg) if p is not None]
        agg = {}
        for k in sizes:
            s = [p for p in pts if p["n_matches"] == k]
            agg[str(k)] = {f: [ml._r(np.mean([p[f] for p in s])), ml._r(np.std([p[f] for p in s]))]
                           for f in ("train_pairs", "train_auc", "heldout_auc", "train_logloss", "heldout_logloss")} | dict(n=len(s))
        out[kind] = dict(points=pts, by_size=agg)
    return out


def _fixed_fold(kind: str, fs: tuple, tables: list[dict], h: int, cfg: ML2Config) -> np.ndarray:
    train = [u for m, u in enumerate(tables) if m != h]
    return proba(fit_model(kind, FIXED[kind], *_xy(train, fs), cfg), tables[h]["X"][:, cols(fs)])


def ablation_variants() -> dict[str, tuple[str, ...]]:
    full = CORE + ml.FT_FEATURES + G_FEATURES
    variants = {"full": full, "distance_only": ml.DISTANCE_FEATURES}
    for fam, fs in FAMILIES.items():
        variants[f"drop_{fam}"] = tuple(f for f in full if f not in fs)
        variants[f"only_{fam}"] = fs
    return variants


def ablations(tables: list[dict], cfg: ML2Config = ML2Config()) -> dict[str, Any]:
    """Feature-family ablations with fixed configurations under leave-one-match-out: drop one family, only one family."""
    variants = ablation_variants()
    out = {}
    for kind in ("rf", "hgb"):
        ps = iter(_parallel(_fixed_fold, [(kind, fs, tables, h, cfg) for fs in variants.values() for h in range(len(tables))], cfg))
        res = {}
        for name, fs in variants.items():
            p = {t["match_id"]: next(ps) for t in tables}
            per = {}
            for t in tables:
                m = ml._train_mask(t)
                yh = np.array([lb["label"] for lb, k in zip(t["labels"], m) if k], int)
                per[t["match_id"]] = dict(auc=_auc(yh, p[t["match_id"]][m]), logloss=ml._r(log_loss(yh, p[t["match_id"]][m])))
            pw = pairwise_block(tables, dict(p=p), cfg)[LABEL_TIER]
            curve = operating_curve(_concat(list(heldout_tables(tables, dict(p=p), cfg).values())), cfg)
            res[name] = dict(n_features=len(fs), auc=pw["roc_auc"], logloss=pw["log_loss"], top1=pw["top1_incoming"], brier=pw["brier"], by_match=per,
                             heldout_curve_best_fa10=best_at(curve, fa_budget=0.10), heldout_curve_best_fa05=best_at(curve, fa_budget=0.05))
        for name, v in res.items():
            d = [v["by_match"][m]["auc"] - res["full"]["by_match"][m]["auc"] for m in v["by_match"]]
            v["auc_delta_vs_full"] = dict(mean=ml._r(np.mean(d)), sd=ml._r(np.std(d)), matches_better=int(sum(x > 0 for x in d)), matches=len(d))
        out[kind] = res
    return out


def _perm_fold(kind: str, fs: tuple, tables: list[dict], h: int, cfg: ML2Config) -> dict[str, float]:
    rng = np.random.default_rng([cfg.base.seed, h])
    train = [u for m, u in enumerate(tables) if m != h]
    model = fit_model(kind, FIXED[kind], *_xy(train, fs), cfg)
    X, y, _ = _heldout_clean(tables[h], cols(fs))
    base = _auc(y, proba(model, X))
    groups = {f: [fs.index(x) for x in FAMILIES[f] if x in fs] for f in FAMILIES if any(x in fs for x in FAMILIES[f])}
    groups |= {f"#{f}": [fs.index(f)] for f in fs}
    out = {}
    for name, idx in groups.items():
        drops = []
        for _ in range(cfg.perm_repeats):
            Xp = X.copy()
            Xp[:, idx] = X[rng.permutation(len(X))][:, idx]
            drops.append(base - _auc(y, proba(model, Xp)))
        out[name] = float(np.mean(drops))
    return out


def permutation_importance(tables: list[dict], cfg: ML2Config = ML2Config(), features: str = "core_fg") -> dict[str, Any]:
    """Grouped permutation importance: each fold's model never saw its held-out match, and columns are permuted within the
    held-out match's clean high-tier pairs only. Families are permuted jointly; single features one at a time."""
    fs = FEATURE_SETS[features]
    out = {}
    for kind in ("rf", "hgb"):
        folds = _parallel(_perm_fold, [(kind, fs, tables, h, cfg) for h in range(len(tables))], cfg)

        def rank(keys, pre):
            return {k: dict(mean_auc_drop=ml._r(np.mean([f[pre + k] for f in folds])), sd=ml._r(np.std([f[pre + k] for f in folds])),
                            folds_positive=int(sum(f[pre + k] > 0 for f in folds))) for k in keys}

        fr = rank(fs, "#")
        per_fold = np.array([[f["#" + x] for f in folds] for x in fs])                 # features x folds
        ranks = (-per_fold).argsort(0).argsort(0) + 1
        for n, f in enumerate(fs):
            fr[f]["rank_mean"], fr[f]["rank_sd"] = ml._r(ranks[n].mean(), 2), ml._r(ranks[n].std(), 2)
        out[kind] = dict(families=rank([k for k in folds[0] if not k.startswith("#")], ""),
                         features=dict(sorted(fr.items(), key=lambda kv: -kv[1]["mean_auc_drop"])))
    return out


def redundancy(tables: list[dict], threshold: float = 0.95) -> list[list]:
    """Feature pairs whose Spearman correlation on the clean high-tier pairs is at least ``threshold`` in absolute value."""
    from scipy.stats import spearmanr

    X = np.vstack([t["X"][ml._train_mask(t)] for t in tables])
    keep = [k for k in range(X.shape[1]) if np.std(X[:, k]) > 0]
    rho = spearmanr(X[:, keep]).statistic
    out = []
    for a in range(len(keep)):
        for b in range(a + 1, len(keep)):
            if abs(rho[a, b]) >= threshold:
                out.append([FEATURES2[keep[a]], FEATURES2[keep[b]], round(float(rho[a, b]), 3)])
    constant = [FEATURES2[k] for k in range(X.shape[1]) if k not in keep]
    return out + [[f, "constant", None] for f in constant]


def label_noise_checks(tables: list[dict], r: dict, cfg: ML2Config) -> dict[str, Any]:
    """Descriptive checks on the provisional reference (never used for training or selection)."""
    unreachable_true, clean_wrong, total_true = 0, [], 0
    for t in tables:
        idx = {(x["instance_id"], x["incoming"], x["outgoing"]): n for n, x in enumerate(t["rows"])}
        for iid, ref in t["refs"].items():
            if ref["tier"] != LABEL_TIER:
                continue
            for i, o in ref["expected"].items():
                if o is None:
                    continue
                total_true += 1
                k = idx[(iid, i, o)]
                if not t["rows"][k]["reachable"]:
                    unreachable_true += 1
                p_true = r["p"][t["match_id"]][k]
                alt = [(r["p"][t["match_id"]][n], n) for (ii, inc, out), n in idx.items() if ii == iid and inc == i and out != o]
                if not alt:
                    continue
                pa, na = max(alt)
                xa, xt = t["X"][na], t["X"][k]
                cm, hr = FEATURES2.index("cv_mid_err"), FEATURES2.index("hermite_rms_acc")
                if pa >= 0.8 and p_true <= 0.2 and xa[cm] < 0.5 and xa[hr] < 2.0 and xt[hr] > 4.0:
                    clean_wrong.append(dict(match_id=t["match_id"], instance_id=iid, incoming=i, p_alternative=ml._r(pa), p_reference=ml._r(p_true),
                                            alt_cv_mid_err=ml._r(xa[cm], 3), alt_hermite_rms=ml._r(xa[hr], 3), ref_hermite_rms=ml._r(xt[hr], 3)))
    return dict(reference_pairs=total_true, reference_pair_unreachable=unreachable_true,
                smooth_alternative_vs_strained_reference=dict(count=len(clean_wrong), share=ml._r(len(clean_wrong) / total_true) if total_true else None,
                                                              rule="alternative p >= 0.8, reference p <= 0.2, alternative cv_mid_err < 0.5 m and Hermite RMS < 2, reference Hermite RMS > 4 m/s^2",
                                                              examples=clean_wrong[:20]))


def dataset_profile(t: dict) -> dict[str, Any]:
    out = ml.dataset_stats(t)
    hi = [r for r in t["refs"].values() if r["tier"] == LABEL_TIER]
    if hi:
        out["high_profile"] = dict(
            incoming_count={str(k): sum(len(r["incoming"]) == k for r in hi) for k in sorted({len(r["incoming"]) for r in hi})},
            gap_bins={b: sum(gap_bin(r["difficulty"]["ambiguity_s"]) == b for r in hi) for b in sorted({gap_bin(r["difficulty"]["ambiguity_s"]) for r in hi})},
            sharp_turn=sum("sharp_turn" in r["difficulty"]["reasons"] for r in hi), crossing=sum(bool(r["difficulty"]["crossing"]) for r in hi),
            absent_tracklet=sum(bool(r["without_incoming"] or r["without_outgoing"]) for r in hi),
            gap_median_s=ml._r(float(np.median([r["difficulty"]["ambiguity_s"] for r in hi])), 3))
    return out


# ---------------------------------------------------------------- the step-5 run

def _dump(obj: Any) -> bytes:
    def default(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.floating, np.integer, np.bool_)):
            return v.item()
        return str(v)
    return json.dumps(obj, indent=1, sort_keys=True, default=default).encode()


def run_ml2(raw_root: str | Path, match_ids: list[str], *, source_root: str | Path | None = None, out_root: str | Path | None = None,
            cfg: ML2Config = ML2Config(), log=print) -> tuple[Path, dict[str, Any]]:
    """Load and featurise the development matches, run diagnostics, ablations, curves, importance and the nested comparison,
    select a model by the a-priori rule, and write ``<out_root>`` (default ``<processed>/m11-ml-v2``) all at once."""
    import joblib
    import sklearn

    t0 = time.perf_counter()
    out = Path(out_root) if out_root is not None else approved_output_root(raw_root) / OUTPUT_DIR
    if out.exists():
        raise IngestionError(f"{out} already exists and is never overwritten")
    split = ml._splits(raw_root)
    ids = [str(m) for m in match_ids]
    bad = [m for m in ids if split["matches"].get(m, {}).get("split") != "development"]
    if bad:
        raise IngestionError(f"only development matches are used; refused: {bad}")
    data = [ml.load_match(raw_root, m, source_root) for m in ids]
    tables = [match_table2(d, cfg.base) for d in data]
    log(f"features: {time.perf_counter() - t0:.0f} s")
    diag = fold_diagnostics(tables, cfg)
    log(f"diagnostics: {time.perf_counter() - t0:.0f} s")
    curves = learning_curves(tables, cfg)
    log(f"learning curves: {time.perf_counter() - t0:.0f} s")
    abl = ablations(tables, cfg)
    log(f"ablations: {time.perf_counter() - t0:.0f} s")
    imp = permutation_importance(tables, cfg)
    log(f"importance: {time.perf_counter() - t0:.0f} s")
    comp = run_comparison(tables, cfg, log=log)
    comp_ft = run_comparison(tables, cfg, comparison=(("hgb", "core_fg"), ("rf", "core_f"), ("hgb", "core_f"), ("rf", "core_fg")), log=log)
    comp |= comp_ft
    results = {}
    for key, r in comp.items():
        ops = operating_results(tables, r, cfg)
        curve = operating_curve(_concat(list(heldout_tables(tables, r, cfg).values())), cfg)
        results[key] = dict(
            pairwise=pairwise_block(tables, r, cfg), pairwise_raw=pairwise_block(tables, r, cfg, key="p_raw"), operating=ops,
            heldout_curve=dict(best_under_fa={str(b): best_at(curve, fa_budget=b) for b in (0.02, 0.05, 0.10, 0.15)},
                               best_at_coverage={str(c): best_at(curve, coverage=c) for c in cfg.coverage_levels},
                               points=[c for c in curve if c["margin"] in (0.0, 0.5) and round(c["threshold"] * 100) % 5 == 0]),
            folds={mid: {k: v for k, v in f.items() if k not in ("model", "calibrator")} | dict(platt=asdict(f["calibrator"])) for mid, f in r["folds"].items()})
    base_pm = results[BASELINE_KEY]["operating"][LABEL_TIER]
    for key, res in results.items():
        for tg in cfg.targets:
            pm = res["operating"][LABEL_TIER][str(tg)]["by_match"]
            res["operating"][LABEL_TIER][str(tg)]["bootstrap"] = bootstrap(pm, None if key == BASELINE_KEY else base_pm[str(tg)]["by_match"], cfg.bootstrap, cfg.base.seed)
    # a-priori selection rule (ML2Config): coverage at the 5% target among models with held-out FA <= cap
    elig = {k: v["operating"][LABEL_TIER]["0.05"]["pooled"] for k, v in results.items()}
    elig = {k: v for k, v in elig.items() if v["false_association_rate"] is not None and v["false_association_rate"] <= cfg.selection_fa_cap}
    selected = max(elig, key=lambda k: (elig[k]["coverage"], -results[k]["pairwise"][LABEL_TIER]["log_loss"])) if elig else BASELINE_KEY
    b = results[selected]["operating"][LABEL_TIER]["0.05"]["bootstrap"]
    improves = selected != BASELINE_KEY and b["coverage_diff_ci95"][0] > 0 and b["false_association_diff_ci95"][1] <= cfg.fa_tolerance
    chosen = selected if improves else BASELINE_KEY
    log(f"selected by rule: {selected}; improves over step-4 RF: {improves}; retained: {chosen}")
    strata, preds = {}, []
    for key in dict.fromkeys([chosen, selected, BASELINE_KEY, "lr/core"]):
        for tg in cfg.targets:
            s, p = strata_eval(tables, comp[key], cfg, tg, f"{key}@{tg}")
            strata[f"{key}@{tg}"] = s
            preds += p
    refs = {iid: x for t in tables for iid, x in t["refs"].items()}
    base_strata = {}
    bpreds = [p for d in data for p in d["baseline_predictions"]]
    for tier in (LABEL_TIER, EXPLORATORY_TIER):
        rs = [x for x in refs.values() if x["tier"] == tier]
        idset = {x["instance_id"] for x in rs}
        base_strata[tier] = evaluate([p for p in bpreds if p["instance_id"] in idset], rs)
    unverified = {}
    for p in preds:
        if refs[p["instance_id"]]["tier"] == UNVERIFIED_TIER:
            for d in p["decisions"]:
                unverified.setdefault(p["baseline"], {}).setdefault(d["decision"], 0)
                unverified[p["baseline"]][d["decision"]] += 1
    fails = failure_cases(tables, comp[chosen], cfg, chosen)
    noise = label_noise_checks(tables, comp[chosen], cfg)
    kind, fs = comp[chosen]["kind"], comp[chosen]["features"]
    final = final_fit(kind, fs, tables, cfg)
    log(f"final fit: {time.perf_counter() - t0:.0f} s")
    summary = dict(schema=SCHEMA_VERSION, task=TASK_B, matches=ids, dataset={t["match_id"]: dataset_profile(t) for t in tables},
                   diagnostics=diag, learning_curves=curves, ablations=abl, permutation_importance=imp, redundancy=redundancy(tables),
                   results=results, selection=dict(rule_selected=selected, improves_over_step4_rf=bool(improves), retained=chosen,
                                                   eligible_at_5pct=elig),
                   strata=strata, baseline_strata=base_strata, unverified_decisions=unverified, failures_top=fails, label_noise=noise,
                   final=dict(model=chosen, params=final["params"], platt=asdict(final["calibrator"]), operating=final["operating"],
                              inner_groups=final["inner_groups"], grid_logloss=final["grid_logloss"]),
                   synthetic_swaps_injected=False, evaluation_kind="nested leave-one-match-out over development matches (no validation or test match used)",
                   note="all accuracies are agreement with the provider identity reference, not verified identity accuracy")
    config = dict(schema=SCHEMA_VERSION, task=TASK_B, ml2=cfg.as_dict(), features={**ml.FEATURES, **FOOTBALL_FEATURES}, families={k: list(v) for k, v in FAMILIES.items()},
                  feature_sets={k: list(v) for k, v in FEATURE_SETS.items()}, grids=GRIDS, fixed=FIXED, comparison=[f"{k}/{f}" for k, f in COMPARISON],
                  baseline=BASELINE_KEY, selection_rule=("highest pooled held-out coverage at the 5% target operating point among models with pooled held-out "
                                                         f"false association <= {cfg.selection_fa_cap}; improves only if the match-bootstrap 95% CI of the "
                                                         f"coverage difference to {BASELINE_KEY} excludes 0 and its false-association difference CI upper bound <= {cfg.fa_tolerance}"),
                  physical_assumptions=dict(max_speed_m_s=cfg.base.max_speed_m_s, reach_base_m=cfg.base.reach_base_m, moving_m_s=cfg.base.moving_m_s,
                                            acc_cap_m_s2=cfg.base.acc_cap_m_s2, note="fixed assumptions carried over from steps 1-4; none estimated from data"),
                  source=dict(schema=ml.SOURCE_SCHEMA, directory=ml.SOURCE_DIR))
    pair_rows = []
    for t in tables:
        for n, (x, lb) in enumerate(zip(t["rows"], t["labels"])):
            pair_rows.append(dict(x, label=lb["label"], status=lb["status"], tier=lb["tier"], features={f: ml._r(t["X"][n, k], 5) for k, f in enumerate(FEATURES2)},
                                  heldout_probability={key: ml._r(comp[key]["p"][t["match_id"]][n]) for key in dict.fromkeys([chosen, selected, BASELINE_KEY, "lr/core"])}))
    blobs = {"config.json": _dump(config), "summary.json": _dump(summary), "pairs.jsonl": ml._jsonl(pair_rows), "predictions.jsonl": ml._jsonl(preds),
             "failures.jsonl": ml._jsonl(fails)}
    tmp = out.parent / f".{out.name}.partial-{uuid.uuid4().hex[:8]}"
    tmp.mkdir(parents=True)
    try:
        for name, blob in blobs.items():
            with open(tmp / name, "xb") as f:
                f.write(blob)
        for key in dict.fromkeys([chosen, selected, BASELINE_KEY]):
            for mid, f in comp[key]["folds"].items():
                d = tmp / "models" / f"heldout-{mid}"
                d.mkdir(parents=True, exist_ok=True)
                joblib.dump(dict(model=f["model"], calibrator=asdict(f["calibrator"]), features=FEATURE_SETS[comp[key]["features"]], operating=f["operating"]),
                            d / f"{key.replace('/', '__')}.joblib")
        (tmp / "models" / "final").mkdir(parents=True)
        joblib.dump(dict(model=final["model"], calibrator=asdict(final["calibrator"]), features=FEATURE_SETS[fs], operating=final["operating"], params=final["params"]),
                    tmp / "models" / "final" / f"{chosen.replace('/', '__')}.joblib")
        digests = {str(p.relative_to(tmp)).replace("\\", "/"): _sha256(p) for p in sorted(tmp.rglob("*")) if p.is_file()}
        sp = approved_output_root(raw_root) / ml.SPLITS_DIR / MANIFEST_NAME
        manifest = dict(schema=SCHEMA_VERSION, matches=ids, split_manifest=dict(path=str(sp), sha256=_sha256(sp)), sources={d["match_id"]: d["provenance"] for d in data},
                        outputs_sha256=digests, code_sha256=code_digest(), coordinate_policy="raw", provider_smoothed_coordinates_used=False,
                        synthetic_swaps_injected=False, validation_or_test_matches_used=False, processing=dict(seconds=round(time.perf_counter() - t0, 1)),
                        python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__, joblib=joblib.__version__)
        with open(tmp / "manifest.json", "xb") as f:
            f.write(_dump(manifest))
        if out.exists():
            raise IngestionError(f"{out} appeared while writing; not overwritten")
        os.rename(tmp, out)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    return out, summary


def failure_cases(tables: list[dict], r: dict, cfg: ML2Config, name: str, k: int = 15) -> list[dict]:
    """Most confident held-out wrong matches (high tier) at the 10% operating point, with the reference pair's probability."""
    rows = []
    for t in tables:
        op = r["folds"][t["match_id"]]["operating"]["0.1"]
        assigned = ml.instance_assignments(t["rows"], r["p"][t["match_id"]], cfg.base)
        ds = ml.decisions(assigned, op["threshold"], replace(cfg.base, margin_logit=op["margin"]))
        idx = {(x["instance_id"], x["incoming"], x["outgoing"]): n for n, x in enumerate(t["rows"])}
        for iid, dl in ds.items():
            ref = t["refs"][iid]
            if ref["tier"] != LABEL_TIER:
                continue
            for d in dl:
                e = ref["expected"][d["incoming"]]
                if d["decision"] == "MATCHED" and d["outgoing"] != e:
                    kk = idx[(iid, d["incoming"], d["outgoing"])]
                    ke = idx[(iid, d["incoming"], e)] if e else None
                    ev = lambda n: {f: ml._r(t["X"][n, FEATURES2.index(f)], 3) for f in ("gap_s", "dist_m", "cv_mid_err", "hermite_rms_acc", "turn_acc_demand", "heading_diff_deg")}   # noqa: E731
                    rows.append(dict(model=name, match_id=t["match_id"], instance_id=iid, incoming=d["incoming"], chosen=d["outgoing"], expected=e,
                                     probability=d["probability"], margin=d["margin"], reference_pair_probability=None if ke is None else ml._r(r["p"][t["match_id"]][ke]),
                                     chosen_evidence=ev(kk), reference_evidence=None if ke is None else ev(ke), n_incoming=len(ref["incoming"]),
                                     gap_s=ref["difficulty"]["ambiguity_s"], difficulty=ref["difficulty"]["reasons"], crossing=ref["difficulty"]["crossing"]))
    return sorted(rows, key=lambda x: -x["probability"])[:k]
