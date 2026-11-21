"""M11 step 4: first supervised pairwise identity-matching classifiers on task B (interval withheld).

    python -m football_fatigue.matching ml --matches 3812 3816 10506 3814

Inputs are verified ``m11-matching-v3`` outputs of development matches. Features are computed by :func:`instance_features`
from one matcher-visible task B instance and nothing else: no reference, no provider identity, no match ID, no frame, no
absolute time, no tracklet order, no interval observation. Labels come only from :func:`pair_labels`, which reads the
evaluation-only reference. The two never meet before training or scoring.

Label protocol (the provider identity mapping is a provisional reference, not verified identity ground truth):

- positive pair: incoming and outgoing tracklets of the same provider identity within one encounter;
- negative pair: incoming and outgoing tracklets of different provider identities within the same encounter (all candidates
  of an encounter are close together, so the classifier has to separate identities among nearby players, not near from far);
- supervised labels come only from ``high_confidence_reference`` instances, and a pair is withheld from training when either
  endpoint window (``estimated_window_s``) holds an ESTIMATED observation;
- ``lower_confidence_reference`` instances are scored for a separate exploratory evaluation and never trained on;
- ``natural_suspicious_unverified`` instances are unverified: never labelled, never trained on, never scored for accuracy.

Evaluation is leave-one-match-out over the development matches given: a model is fitted on the other matches only, its
acceptance threshold is chosen by an inner leave-one-match-out over those training matches only, and it is scored once on
the held-out match. Model complexity is fixed a priori (:class:`MLConfig`) and not tuned. Validation and test matches are
refused. No synthetic swap is injected. Every accuracy is agreement with the provider reference.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from football_fatigue.encounters.splits import MANIFEST_NAME
from football_fatigue.ingestion import IngestionError, approved_output_root

from .baselines import BASELINES, solve
from .benchmark import OUTPUT_DIR as SOURCE_DIR, SCHEMA_VERSION as SOURCE_SCHEMA, TASK_B, _sha256, code_digest, output_paths, tracklet_velocity
from .evaluation import evaluate, outcome
from .footballtrace import FootballTraceConfig, junction_sharpness, paper_clock_track

SCHEMA_VERSION = OUTPUT_DIR = "m11-ml-v1"
SPLITS_DIR = "m11-encounters-v1"            # the frozen split manifest
LABEL_TIER = "high_confidence_reference"
EXPLORATORY_TIER = "lower_confidence_reference"
UNVERIFIED_TIER = "natural_suspicious_unverified"
MODELS = ("lr", "rf", "lr_ft", "rf_ft", "lr_distance")

FEATURES: dict[str, str] = {
    "gap_s": "time from the last incoming to the first outgoing observation (s)",
    "dist_m": "distance between the incoming end and the outgoing start (m)",
    "dist_per_s": "dist_m / gap_s: mean speed needed to connect the pair (m/s)",
    "reach_ratio": "dist_m / (reach_base_m + max_speed * gap_s); above 1 is physically unreachable",
    "speed_in": "incoming end speed, least squares over update instants of the last span_s (m/s)",
    "speed_out": "outgoing start speed over the first span_s (m/s)",
    "speed_diff": "|speed_out - speed_in| (m/s)",
    "vel_diff": "|v_out - v_in| (m/s)",
    "heading_diff_deg": "angle between v_in and v_out when both move faster than moving_m_s, else 0",
    "both_moving": "1 when both end speeds exceed moving_m_s",
    "cv_fwd_err": "|p_in + v_in * gap - p_out|: constant-velocity forward extrapolation error (m)",
    "cv_bwd_err": "|p_out - v_out * gap - p_in|: constant-velocity backward extrapolation error (m)",
    "cv_mid_err": "distance between the forward and backward half-gap extrapolations (m)",
    "ca_fwd_err": "constant-acceleration forward extrapolation error, incoming acceleration capped (m)",
    "req_acc": "constant acceleration needed to reach p_out from (p_in, v_in) in gap (m/s^2)",
    "hermite_max_acc": "peak acceleration of the cubic Hermite path joining (p_in, v_in) to (p_out, v_out) (m/s^2)",
    "hermite_rms_acc": "RMS acceleration of that cubic Hermite path (m/s^2)",
    "acc_in": "incoming acceleration magnitude, quadratic fit over the last shape_s (m/s^2)",
    "acc_out": "outgoing acceleration magnitude over the first shape_s (m/s^2)",
    "turn_rate_in": "incoming mean absolute heading change over the last shape_s (deg/s)",
    "turn_rate_out": "outgoing mean absolute heading change over the first shape_s (deg/s)",
    "straight_in": "incoming displacement / path length over the last shape_s",
    "straight_out": "outgoing displacement / path length over the first shape_s",
    "n_upd_in": "incoming update instants in the last shape_s (observation coverage)",
    "n_upd_out": "outgoing update instants in the first shape_s",
    "est_share_in": "share of ESTIMATED samples in the incoming endpoint window (quality)",
    "est_share_out": "share of ESTIMATED samples in the outgoing endpoint window",
    "conf_in": "mean provider confidence code in the incoming endpoint window (quality)",
    "conf_out": "mean provider confidence code in the outgoing endpoint window",
    "n_incoming": "incoming tracklets in the encounter",
    "n_outgoing": "outgoing tracklets in the encounter",
}
REL_KEYS = ("dist_m", "cv_fwd_err", "cv_mid_err", "hermite_rms_acc")
for _k in REL_KEYS:
    FEATURES[f"row_gap_{_k}"] = f"{_k} minus the best other outgoing candidate of the same incoming tracklet (capped; rel_cap if none)"
    FEATURES[f"col_gap_{_k}"] = f"{_k} minus the best other incoming candidate of the same outgoing tracklet (capped; rel_cap if none)"
CORE_FEATURES = tuple(FEATURES)
FT_DOC = {
    "ft_sharpness": "FOOTBALLTrace-inspired junction sharpness (1 + cos) / 2 on the 25 Hz SG-smoothed paper clock (E0-Adapted cost; "
                    "not the published method)",
    "row_gap_ft_sharpness": "ft_sharpness minus the best other outgoing candidate of the same incoming tracklet",
}
FEATURES.update(FT_DOC)
FT_FEATURES = tuple(FT_DOC)
ALL_FEATURES = tuple(FEATURES)
DISTANCE_FEATURES = ("gap_s", "dist_m")
MODEL_FEATURES = dict(lr=CORE_FEATURES, rf=CORE_FEATURES, lr_ft=CORE_FEATURES + FT_FEATURES, rf_ft=CORE_FEATURES + FT_FEATURES,
                      lr_distance=DISTANCE_FEATURES)
EVIDENCE_KEYS = ("gap_s", "dist_m", "cv_fwd_err", "cv_mid_err", "hermite_rms_acc", "heading_diff_deg", "reach_ratio")


@dataclass(frozen=True)
class MLConfig:
    span_s: float = 0.5               # endpoint velocity window, as the baselines
    shape_s: float = 1.0              # within-tracklet acceleration, turn and straightness window
    max_speed_m_s: float = 12.0       # physiological bound, as steps 1-3
    reach_base_m: float = 0.5         # reachable: dist <= reach_base + max_speed * gap
    moving_m_s: float = 1.0
    acc_cap_m_s2: float = 8.0         # cap on a fitted within-tracklet acceleration used for extrapolation
    rel_cap: float = 10.0             # relative-gap cap, and its value when there is no competitor
    estimated_window_s: float = 0.5   # label protocol: ESTIMATED inside this endpoint window withholds the pair from training
    lr_c: float = 1.0                 # a priori, not tuned
    rf_trees: int = 300
    rf_min_leaf: int = 5
    seed: int = 0
    margin_logit: float = 0.5         # an alternative assignment within this log-odds: UNCERTAIN (a priori)
    target_false_association: float = 0.05   # threshold chosen on training matches to meet this, with maximum coverage
    threshold_grid: tuple = tuple(round(0.5 + 0.01 * k, 2) for k in range(50))
    reliability_bins: int = 10

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------- features (matcher-visible instance only)

def _window(tr: dict, at_end: bool, span: float) -> np.ndarray:
    t = np.asarray(tr["t"], float)
    return (t >= t[-1] - span) if at_end else (t <= t[0] + span)


def tracklet_summary(tr: dict, at_end: bool, cfg: MLConfig) -> dict[str, Any]:
    """Endpoint state and within-tracklet motion statistics of one tracklet (its end if incoming, its start if outgoing)."""
    t, x, y = (np.asarray(tr[k], float) for k in ("t", "x", "y"))
    v, _ = tracklet_velocity(tr, at_end=at_end, span_s=cfg.span_s, max_speed_m_s=cfg.max_speed_m_s)
    upd = np.r_[True, (np.diff(x) != 0) | (np.diff(y) != 0)] & _window(tr, at_end, cfg.shape_s)
    tu, xu, yu = t[upd], x[upd], y[upd]
    t0 = t[-1] if at_end else t[0]
    acc = np.zeros(2)
    if len(tu) >= 3 and tu[-1] - tu[0] >= 0.2:
        acc = np.array([2 * np.polyfit(tu - t0, xu, 2)[0], 2 * np.polyfit(tu - t0, yu, 2)[0]])
        a = float(np.hypot(*acc))
        if a > cfg.acc_cap_m_s2:
            acc *= cfg.acc_cap_m_s2 / a
    turn, straight = 0.0, 1.0
    if len(tu) >= 2:
        d = np.diff(np.column_stack([xu, yu]), axis=0)
        step = np.hypot(d[:, 0], d[:, 1])
        path = float(step.sum())
        straight = float(np.hypot(xu[-1] - xu[0], yu[-1] - yu[0]) / path) if path > 1e-6 else 1.0
        dtu = np.diff(tu)
        mov = step / np.maximum(dtu, 1e-6) >= cfg.moving_m_s
        h = np.arctan2(d[mov, 1], d[mov, 0])
        if len(h) >= 2 and tu[-1] > tu[0]:
            turn = float(np.degrees(np.abs((np.diff(h) + np.pi) % (2 * np.pi) - np.pi).sum()) / (tu[-1] - tu[0]))
    ew = _window(tr, at_end, cfg.estimated_window_s)
    vis, conf = np.asarray(tr["visibility"])[ew], np.asarray(tr["confidence"], float)[ew]
    return dict(p=np.array([x[-1], y[-1]] if at_end else [x[0], y[0]]), t=float(t0), v=v, speed=float(np.hypot(*v)), acc=acc,
                turn=turn, straight=straight, n_upd=int(len(tu)), est_share=float(np.mean(vis == 2)) if len(vis) else 0.0,
                conf=float(conf.mean()) if len(conf) else 0.0)


def pair_values(a: dict, b: dict, cfg: MLConfig) -> dict[str, float]:
    """Physics features of joining incoming summary ``a`` to outgoing summary ``b`` (no competitor context)."""
    gap = max(b["t"] - a["t"], 1.0 / 30)
    D = b["p"] - a["p"]
    dist = float(np.hypot(*D))
    va, vb = a["v"], b["v"]
    moving = a["speed"] >= cfg.moving_m_s and b["speed"] >= cfg.moving_m_s
    head = float(np.degrees(np.arccos(np.clip(va @ vb / (a["speed"] * b["speed"]), -1, 1)))) if moving else 0.0
    a0 = (6 * D - (4 * va + 2 * vb) * gap) / gap ** 2
    a1 = (-6 * D + (2 * va + 4 * vb) * gap) / gap ** 2
    return dict(
        gap_s=gap, dist_m=dist, dist_per_s=dist / gap, reach_ratio=dist / (cfg.reach_base_m + cfg.max_speed_m_s * gap),
        speed_in=a["speed"], speed_out=b["speed"], speed_diff=abs(b["speed"] - a["speed"]), vel_diff=float(np.hypot(*(vb - va))),
        heading_diff_deg=head, both_moving=float(moving),
        cv_fwd_err=float(np.hypot(*(a["p"] + va * gap - b["p"]))), cv_bwd_err=float(np.hypot(*(b["p"] - vb * gap - a["p"]))),
        cv_mid_err=float(np.hypot(*((a["p"] + va * gap / 2) - (b["p"] - vb * gap / 2)))),
        ca_fwd_err=float(np.hypot(*(a["p"] + va * gap + 0.5 * a["acc"] * gap ** 2 - b["p"]))),
        req_acc=float(np.hypot(*(2 * (D - va * gap) / gap ** 2))),
        hermite_max_acc=float(max(np.hypot(*a0), np.hypot(*a1))), hermite_rms_acc=float(np.sqrt(max(a0 @ a0 + a0 @ a1 + a1 @ a1, 0) / 3)),
        acc_in=float(np.hypot(*a["acc"])), acc_out=float(np.hypot(*b["acc"])), turn_rate_in=a["turn"], turn_rate_out=b["turn"],
        straight_in=a["straight"], straight_out=b["straight"], n_upd_in=float(a["n_upd"]), n_upd_out=float(b["n_upd"]),
        est_share_in=a["est_share"], est_share_out=b["est_share"], conf_in=a["conf"], conf_out=b["conf"])


def _rel_gap(M: np.ndarray, axis: int, cap: float) -> np.ndarray:
    """Each entry minus the minimum of the other entries along ``axis``; ``cap`` when there is no other entry."""
    out = np.full(M.shape, cap)
    n = M.shape[axis]
    for k in range(n):
        others = np.delete(M, k, axis=axis)
        if others.shape[axis]:
            sl = [slice(None)] * 2
            sl[axis] = k
            out[tuple(sl)] = M[tuple(sl)] - others.min(axis=axis)
    return np.clip(out, -cap, cap)


def instance_features(inst: dict, cfg: MLConfig = MLConfig(), ft: FootballTraceConfig = FootballTraceConfig()) -> tuple[list[dict], np.ndarray]:
    """``(pair rows, feature matrix in ALL_FEATURES order)`` for every incoming x outgoing pair of one matcher-visible instance.

    Only the instance is read. Pair features do not depend on the order in which tracklets are listed."""
    if inst.get("task") != TASK_B or inst["during"]["t"]:
        raise ValueError("features are computed on task B instances only (no interval observations)")
    I, O = inst["incoming"], inst["outgoing"]
    SA = [tracklet_summary(a, True, cfg) for a in I]
    SB = [tracklet_summary(b, False, cfg) for b in O]
    vals = [[pair_values(a, b, cfg) for b in SB] for a in SA]
    tin = [paper_clock_track(a, ft, anchor_end=True) for a in I]
    tout = [paper_clock_track(b, ft, anchor_end=False) for b in O]
    ftm = np.array([[junction_sharpness(tin[i], tout[j], ft) for j in range(len(O))] for i in range(len(I))]).reshape(len(I), len(O))
    for i in range(len(I)):
        for j in range(len(O)):
            vals[i][j].update(n_incoming=float(len(I)), n_outgoing=float(len(O)), ft_sharpness=float(ftm[i, j]))
    if len(I) and len(O):
        for k in REL_KEYS:
            M = np.array([[vals[i][j][k] for j in range(len(O))] for i in range(len(I))])
            R, C = _rel_gap(M, 1, cfg.rel_cap), _rel_gap(M, 0, cfg.rel_cap)
            for i in range(len(I)):
                for j in range(len(O)):
                    vals[i][j][f"row_gap_{k}"], vals[i][j][f"col_gap_{k}"] = float(R[i, j]), float(C[i, j])
        R = _rel_gap(ftm, 1, 1.0)
        for i in range(len(I)):
            for j in range(len(O)):
                vals[i][j]["row_gap_ft_sharpness"] = float(R[i, j])
    rows, X = [], []
    for i, a in enumerate(I):
        for j, b in enumerate(O):
            v = vals[i][j]
            rows.append(dict(instance_id=inst["instance_id"], incoming=a["tracklet_id"], outgoing=b["tracklet_id"], reachable=bool(v["reach_ratio"] <= 1.0)))
            X.append([v[k] for k in ALL_FEATURES])
    X = np.nan_to_num(np.asarray(X, float).reshape(len(rows), len(ALL_FEATURES)), nan=0.0, posinf=1e3, neginf=-1e3)
    return rows, np.clip(X, -1e3, 1e3)


def build_features(instances: list[dict], cfg: MLConfig = MLConfig()) -> tuple[list[dict], np.ndarray]:
    rows, Xs = [], []
    for inst in instances:
        r, X = instance_features(inst, cfg)
        rows += r
        Xs.append(X)
    return rows, (np.vstack(Xs) if Xs else np.zeros((0, len(ALL_FEATURES))))


# ---------------------------------------------------------------- labels (evaluation-only reference)

def pair_labels(reference: dict, instance: dict, cfg: MLConfig = MLConfig()) -> list[dict]:
    """Provisional provider-reference label and label status of every pair of one instance.

    ``status``: ``train`` (high tier, clean endpoints), ``excluded_estimated_endpoint`` (high tier, ESTIMATED in an endpoint
    window), ``exploratory`` (lower tier) or ``unverified`` (natural suspicious; ``label`` is None)."""
    if reference["instance_id"] != instance["instance_id"]:
        raise ValueError("reference and instance differ")
    tier = reference["tier"]
    est_in = {a["tracklet_id"]: 2 in np.asarray(a["visibility"])[_window(a, True, cfg.estimated_window_s)] for a in instance["incoming"]}
    est_out = {b["tracklet_id"]: 2 in np.asarray(b["visibility"])[_window(b, False, cfg.estimated_window_s)] for b in instance["outgoing"]}
    rows = []
    for a in instance["incoming"]:
        for b in instance["outgoing"]:
            i, o = a["tracklet_id"], b["tracklet_id"]
            if tier == UNVERIFIED_TIER:
                status, label = "unverified", None
            else:
                label = int(reference["expected"][i] == o)
                status = "exploratory" if tier == EXPLORATORY_TIER else "excluded_estimated_endpoint" if est_in[i] or est_out[o] else "train"
            rows.append(dict(instance_id=instance["instance_id"], incoming=i, outgoing=o, tier=tier, label=label, status=status))
    return rows


# ---------------------------------------------------------------- models

def make_model(name: str, cfg: MLConfig = MLConfig()):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if name.startswith("lr"):
        return Pipeline([("scale", StandardScaler()),
                         ("clf", LogisticRegression(C=cfg.lr_c, class_weight="balanced", max_iter=5000, random_state=cfg.seed))])
    if name.startswith("rf"):
        return Pipeline([("clf", RandomForestClassifier(n_estimators=cfg.rf_trees, min_samples_leaf=cfg.rf_min_leaf, class_weight="balanced_subsample",
                                                        random_state=cfg.seed, n_jobs=1))])
    raise ValueError(f"unknown model {name}")


def columns(name: str) -> list[int]:
    return [ALL_FEATURES.index(f) for f in MODEL_FEATURES[name]]


def fit(name: str, X: np.ndarray, y: np.ndarray, cfg: MLConfig = MLConfig()):
    if len(set(y.tolist())) < 2:
        raise ValueError("training needs positive and negative pairs")
    return make_model(name, cfg).fit(X[:, columns(name)], y)


def predict(model, name: str, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X[:, columns(name)])[:, 1] if len(X) else np.zeros(0)


def lr_contributions(model, name: str, x: np.ndarray, k: int = 3) -> list[list]:
    """Top-k signed log-odds contributions (standardised value x coefficient) of a logistic model for one pair."""
    z = model.named_steps["scale"].transform(x[None, columns(name)])[0] * model.named_steps["clf"].coef_[0]
    return [[MODEL_FEATURES[name][i], round(float(z[i]), 3)] for i in np.argsort(-np.abs(z), kind="stable")[:k]]


# ---------------------------------------------------------------- global one-to-one assignment

def assign(P: np.ndarray, R: np.ndarray, cfg: MLConfig = MLConfig()) -> list[dict]:
    """Threshold-free part of the decision for one instance.

    Maximises the summed log-odds of accepted pairs under one-to-one constraints, over reachable pairs only; leaving a row
    unassigned costs 0, so a pair below p = 0.5 is never taken. Per row: ``pick`` (column or -1), its probability, and
    ``margin``: how much worse (log-odds) the best assignment without this pairing is. A row with no reachable column is
    ``no_plausible``; a low probability alone never makes a row UNMATCHED."""
    p = np.clip(P, 1e-6, 1 - 1e-6)
    # every row pays exactly one term (a pair or abstaining), so adding one constant to both keeps the optimum and the margins
    # while making all costs non-negative, which the pruning in ``solve`` requires
    shift = -np.log(1e-6) + np.log1p(-1e-6) + 1.0
    C = np.where(R, shift - (np.log(p) - np.log1p(-p)), np.inf)
    abstain = np.full(P.shape[0], shift)
    total, pick = solve(C, abstain)
    out = []
    for i in range(P.shape[0]):
        j = pick[i]
        reach = bool(R[i].any())
        cand = j if j >= 0 else (int(np.argmax(np.where(R[i], P[i], -1.0))) if reach else -1)
        margin = solve(C, abstain, forbid=(i, j))[0] - total if j >= 0 else None
        out.append(dict(pick=int(j), candidate=int(cand), p=float(P[i, cand]) if cand >= 0 else None, margin=margin, no_plausible=not reach))
    return out


def decide(raw: dict, threshold: float, cfg: MLConfig = MLConfig()) -> str:
    if raw["no_plausible"]:
        return "UNMATCHED"
    if raw["pick"] >= 0 and raw["p"] >= threshold and raw["margin"] >= cfg.margin_logit:
        return "MATCHED"
    return "UNCERTAIN"


def instance_assignments(rows: list[dict], p: np.ndarray, cfg: MLConfig = MLConfig()) -> dict[str, dict]:
    """Group pair rows by instance and assign. Returns instance_id -> dict(incoming, outgoing, raw decisions)."""
    by: dict[str, list[int]] = {}
    for k, r in enumerate(rows):
        by.setdefault(r["instance_id"], []).append(k)
    out = {}
    for iid, ks in by.items():
        inc = list(dict.fromkeys(rows[k]["incoming"] for k in ks))
        outg = list(dict.fromkeys(rows[k]["outgoing"] for k in ks))
        P, R, K = np.zeros((len(inc), len(outg))), np.zeros((len(inc), len(outg)), bool), np.zeros((len(inc), len(outg)), int)
        for k in ks:
            i, j = inc.index(rows[k]["incoming"]), outg.index(rows[k]["outgoing"])
            P[i, j], R[i, j], K[i, j] = p[k], rows[k]["reachable"], k
        out[iid] = dict(incoming=inc, outgoing=outg, raw=assign(P, R, cfg), rows=K)
    return out


def decisions(assigned: dict[str, dict], threshold: float, cfg: MLConfig = MLConfig()) -> dict[str, list[dict]]:
    res = {}
    for iid, a in assigned.items():
        ds = []
        for i, r in enumerate(a["raw"]):
            dec = decide(r, threshold, cfg)
            cand = a["outgoing"][r["candidate"]] if r["candidate"] >= 0 else None
            ds.append(dict(incoming=a["incoming"][i], decision=dec, outgoing=cand if dec == "MATCHED" else None, candidate=cand,
                           probability=None if r["p"] is None else round(r["p"], 4),
                           margin=None if r["margin"] is None else round(float(r["margin"]), 4)))
        res[iid] = ds
    return res


# ---------------------------------------------------------------- metrics

def _r(v: float | None, n: int = 4) -> float | None:
    return None if v is None or not np.isfinite(v) else round(float(v), n)


def pairwise_metrics(y: np.ndarray, p: np.ndarray, rows: list[dict], bins: int = 10) -> dict[str, Any]:
    """Pair-level classification metrics at p >= 0.5, ranking and calibration. Separate from encounter-level metrics."""
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    y = np.asarray(y, int)
    if not len(y):
        return dict(pairs=0)
    yh = p >= 0.5
    tp, fp, fn = int((yh & (y == 1)).sum()), int((yh & (y == 0)).sum()), int((~yh & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    two = len(set(y.tolist())) == 2
    groups: dict[tuple, list[int]] = {}
    for k, r in enumerate(rows):
        groups.setdefault((r["instance_id"], r["incoming"]), []).append(k)
    top = [y[ks][int(np.argmax(p[ks]))] == 1 for ks in groups.values() if y[ks].sum() == 1]
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rel, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
            rel.append(dict(bin=[round(edges[b], 2), round(edges[b + 1], 2)], n=int(m.sum()), mean_p=_r(p[m].mean()), positive_rate=_r(y[m].mean())))
    return dict(pairs=int(len(y)), positives=int(y.sum()), negatives=int((y == 0).sum()),
                precision=_r(prec), recall=_r(rec), f1=_r(2 * prec * rec / (prec + rec)) if prec and rec else None,
                roc_auc=_r(roc_auc_score(y, p)) if two else None, average_precision=_r(average_precision_score(y, p)) if two else None,
                brier=_r(brier_score_loss(y, p)), ece=_r(ece), top1_incoming=_r(np.mean(top)) if top else None, top1_n=len(top), reliability=rel)


def encounter_summary(preds: dict[str, list[dict]], refs: dict[str, dict]) -> dict[str, float | None]:
    """Coverage, false association and correct rate of one decision set (for threshold selection and trade-off curves)."""
    k = Counter(outcome(d, refs[iid]["expected"][d["incoming"]]) for iid, ds in preds.items() for d in ds)
    n = sum(k.values())
    matched = k["correct_match"] + k["wrong_match"]
    complete = sum(all(outcome(d, refs[iid]["expected"][d["incoming"]]) in ("correct_match", "correct_unmatched") for d in ds) for iid, ds in preds.items())
    return dict(coverage=_r(matched / n) if n else None, false_association_rate=_r(k["wrong_match"] / matched) if matched else None,
                correct_match_rate=_r(k["correct_match"] / n) if n else None, complete_assignment_accuracy=_r(complete / len(preds)) if preds else None)


def select_threshold(assigned: dict[str, dict], refs: dict[str, dict], cfg: MLConfig = MLConfig()) -> dict[str, Any]:
    """Lowest grid threshold whose false-association rate meets the target (maximum coverage), else the highest threshold.
    Called on out-of-fold training-match predictions only."""
    curve = []
    for thr in cfg.threshold_grid:
        s = encounter_summary(decisions(assigned, thr, cfg), refs)
        curve.append(dict(threshold=thr, **s))
    ok = [c for c in curve if c["false_association_rate"] is not None and c["false_association_rate"] <= cfg.target_false_association]
    pick = ok[0] if ok else curve[-1]
    return dict(threshold=pick["threshold"], target_met=bool(ok), training_oof=pick)


# ---------------------------------------------------------------- data

def _splits(raw_root: str | Path) -> dict:
    path = approved_output_root(raw_root) / SPLITS_DIR / MANIFEST_NAME
    if not path.exists():
        raise IngestionError(f"the frozen split manifest {path} is missing")
    return json.loads(path.read_text(encoding="utf-8"))


def load_match(raw_root: str | Path, match_id: object, source_root: str | Path | None = None) -> dict[str, Any]:
    """Verified task B instances (matcher-visible), references and task B baseline predictions (evaluation-only) of one match."""
    paths = output_paths(raw_root, match_id, source_root)
    m = json.loads(paths[".manifest.json"].read_text(encoding="utf-8"))
    if m.get("schema") != SOURCE_SCHEMA or str(m.get("match_id")) != str(match_id) or (m.get("split") or {}).get("split") != "development":
        raise IngestionError(f"{paths['.manifest.json']} is not a {SOURCE_SCHEMA} development output of match {match_id}")
    if m.get("synthetic_swaps_injected") is not False or m.get("provider_smoothed_coordinates_used") is not False:
        raise IngestionError("the source is not a RAW, swap-free matching benchmark")
    digests = {}
    for sfx in (".instances_b.jsonl", ".references.jsonl", ".predictions_b.jsonl"):
        d = _sha256(paths[sfx])
        if m["outputs_sha256"].get(paths[sfx].name) != d:
            raise IngestionError(f"{paths[sfx]} does not match its manifest checksum; not used")
        digests[paths[sfx].name] = d
    read = lambda sfx: [json.loads(x) for x in paths[sfx].read_text(encoding="utf-8").splitlines()]   # noqa: E731
    return dict(match_id=str(match_id), instances=read(".instances_b.jsonl"), references=read(".references.jsonl"),
                baseline_predictions=read(".predictions_b.jsonl"),
                provenance=dict(directory=str(paths[".manifest.json"].parent), sha256=digests, manifest_sha256=_sha256(paths[".manifest.json"])))


def match_table(data: dict, cfg: MLConfig = MLConfig()) -> dict[str, Any]:
    """Features (from instances only), then labels (from references), aligned by pair key."""
    rows, X = build_features(data["instances"], cfg)
    insts = {i["instance_id"]: i for i in data["instances"]}
    lab = {(r["instance_id"], r["incoming"], r["outgoing"]): r for ref in data["references"] for r in pair_labels(ref, insts[ref["instance_id"]], cfg)}
    labels = [lab[(r["instance_id"], r["incoming"], r["outgoing"])] for r in rows]
    for r in rows:
        r["match_id"] = data["match_id"]
    return dict(match_id=data["match_id"], rows=rows, X=X, labels=labels, refs={r["instance_id"]: r for r in data["references"]})


def _train_mask(t: dict) -> np.ndarray:
    return np.array([lb["status"] == "train" for lb in t["labels"]], bool)


def _tier_mask(t: dict, tier: str) -> np.ndarray:
    return np.array([lb["tier"] == tier for lb in t["labels"]], bool)


def _stack(tables: list[dict], mask_fn) -> tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for t in tables:
        m = mask_fn(t)
        Xs.append(t["X"][m])
        ys.append(np.array([lb["label"] for lb, k in zip(t["labels"], m) if k], int))
    return np.vstack(Xs), np.concatenate(ys)


def dataset_stats(t: dict) -> dict[str, Any]:
    out = {}
    for tier in (LABEL_TIER, EXPLORATORY_TIER, UNVERIFIED_TIER):
        lbs = [lb for lb in t["labels"] if lb["tier"] == tier]
        refs = [r for r in t["refs"].values() if r["tier"] == tier]
        out[tier] = dict(instances=len(refs), incoming=sum(len(r["incoming"]) for r in refs), pairs=len(lbs),
                         positives=sum(lb["label"] == 1 for lb in lbs), negatives=sum(lb["label"] == 0 for lb in lbs),
                         status=dict(Counter(lb["status"] for lb in lbs)))
    tr = [lb for lb in t["labels"] if lb["status"] == "train"]
    out["training_pairs"] = dict(pairs=len(tr), positives=sum(lb["label"] for lb in tr), negatives=sum(1 - lb["label"] for lb in tr))
    return out


# ---------------------------------------------------------------- leave-one-match-out experiment

def _oof_threshold(name: str, tables: list[dict], cfg: MLConfig) -> dict[str, Any]:
    """Inner leave-one-match-out over training matches: out-of-fold high-tier assignments, then the threshold."""
    assigned, refs = {}, {}
    for k, t in enumerate(tables):
        X, y = _stack([u for m, u in enumerate(tables) if m != k], _train_mask)
        model = fit(name, X, y, cfg)
        hi = _tier_mask(t, LABEL_TIER)
        rows = [r for r, m in zip(t["rows"], hi) if m]
        assigned |= instance_assignments(rows, predict(model, name, t["X"][hi]), cfg)
        refs |= t["refs"]
    return select_threshold(assigned, refs, cfg)


def run_experiment(tables: list[dict], cfg: MLConfig = MLConfig(), models: tuple[str, ...] = MODELS) -> dict[str, Any]:
    """Outer leave-one-match-out. Returns per-model held-out probabilities, decisions, thresholds, fold models and final models."""
    if len(tables) < 3:
        raise ValueError("match-disjoint evaluation with an inner threshold split needs at least three development matches")
    ids = [t["match_id"] for t in tables]
    iids = [iid for t in tables for iid in t["refs"]]
    if len(set(iids)) != len(iids) or len(set(ids)) != len(ids):
        raise ValueError("instance or match IDs repeat across matches")
    res: dict[str, Any] = dict(folds=[dict(heldout=m, train=[x for x in ids if x != m]) for m in ids], models={})
    for name in models:
        r = dict(p={}, decisions={}, thresholds={}, fold_models={})
        for k, t in enumerate(tables):
            train = [u for m, u in enumerate(tables) if m != k]
            X, y = _stack(train, _train_mask)
            model = fit(name, X, y, cfg)
            thr = _oof_threshold(name, train, cfg)
            p = predict(model, name, t["X"])
            assigned = instance_assignments(t["rows"], p, cfg)
            ds = decisions(assigned, thr["threshold"], cfg)
            for iid, dl in ds.items():                     # evidence for the chosen or candidate pair
                a = assigned[iid]
                for i, d in enumerate(dl):
                    if d["candidate"] is None:
                        d["evidence"] = None
                        continue
                    kk = a["rows"][i, a["outgoing"].index(d["candidate"])]
                    d["evidence"] = {f: _r(t["X"][kk, ALL_FEATURES.index(f)], 3) for f in EVIDENCE_KEYS}
                    if name.startswith("lr"):
                        d["evidence"]["top_log_odds_contributions"] = lr_contributions(model, name, t["X"][kk])
            r["p"][t["match_id"]] = p
            r["decisions"][t["match_id"]] = ds
            r["thresholds"][t["match_id"]] = thr
            r["fold_models"][t["match_id"]] = model
        X, y = _stack(tables, _train_mask)
        r["final_model"] = fit(name, X, y, cfg)
        r["final_threshold"] = _oof_threshold(name, tables, cfg)
        res["models"][name] = r
    return res


def summarize(tables: list[dict], exp: dict[str, Any], baseline_preds: dict[str, list[dict]], cfg: MLConfig = MLConfig()) -> dict[str, Any]:
    refs = {iid: r for t in tables for iid, r in t["refs"].items()}
    tier_refs = {tier: [r for r in refs.values() if r["tier"] == tier] for tier in (LABEL_TIER, EXPLORATORY_TIER)}
    methods = list(BASELINES) + list(exp["models"])
    preds = [p for mid in baseline_preds for p in baseline_preds[mid]]
    for name, r in exp["models"].items():
        preds += [dict(baseline=name, instance_id=iid, decisions=ds) for mid in r["decisions"] for iid, ds in r["decisions"][mid].items()]
    encounter: dict[str, Any] = {}
    for tier, rs in tier_refs.items():
        ids = {r["instance_id"] for r in rs}
        encounter[tier] = evaluate([p for p in preds if p["instance_id"] in ids], rs, methods=tuple(methods))
        encounter[tier + "_by_match"] = {
            t["match_id"]: {m: v["all"] for m, v in evaluate([p for p in preds if p["instance_id"] in ids & set(t["refs"])],
                                                                [r for r in rs if r["instance_id"] in t["refs"]], methods=tuple(methods)).items()}
            for t in tables}
    pairwise, tradeoff, thresholds = {}, {}, {}
    for name, r in exp["models"].items():
        pw = {}
        for tier in (LABEL_TIER, EXPLORATORY_TIER):
            ys, ps, rows = [], [], []
            for t in tables:
                m = _tier_mask(t, tier)
                ys.append(np.array([lb["label"] for lb, k in zip(t["labels"], m) if k], int))
                ps.append(r["p"][t["match_id"]][m])
                rows += [x for x, k in zip(t["rows"], m) if k]
                if tier == LABEL_TIER:
                    pw.setdefault("high_by_match", {})[t["match_id"]] = {k2: v for k2, v in pairwise_metrics(ys[-1], ps[-1], [x for x, k in zip(t["rows"], m) if k], cfg.reliability_bins).items() if k2 != "reliability"}
            pw[tier] = pairwise_metrics(np.concatenate(ys), np.concatenate(ps), rows, cfg.reliability_bins)
        pairwise[name] = pw
        hi = {}                       # held-out trade-off: descriptive only, never used to choose anything
        for t in tables:
            m = _tier_mask(t, LABEL_TIER)
            hi |= instance_assignments([x for x, k in zip(t["rows"], m) if k], r["p"][t["match_id"]][m], cfg)
        hrefs = {iid: refs[iid] for iid in hi}
        tradeoff[name] = [dict(threshold=thr, **encounter_summary(decisions(hi, thr, cfg), hrefs)) for thr in cfg.threshold_grid[::5]]
        thresholds[name] = dict(by_heldout=r["thresholds"], final=r["final_threshold"])
    return dict(encounter=encounter, pairwise=pairwise, tradeoff_heldout=tradeoff, thresholds=thresholds)


def failures(tables: list[dict], exp: dict[str, Any], k: int = 10) -> list[dict]:
    """Most confident wrong matches on held-out high-tier instances, with the true pair's probability."""
    out = []
    for name, r in exp["models"].items():
        rows = []
        for t in tables:
            pidx = {(x["instance_id"], x["incoming"], x["outgoing"]): n for n, x in enumerate(t["rows"])}
            for iid, ds in r["decisions"][t["match_id"]].items():
                ref = t["refs"][iid]
                if ref["tier"] != LABEL_TIER:
                    continue
                for d in ds:
                    exp_o = ref["expected"][d["incoming"]]
                    if outcome(d, exp_o) == "wrong_match":
                        tp = r["p"][t["match_id"]][pidx[(iid, d["incoming"], exp_o)]] if exp_o else None
                        rows.append(dict(model=name, match_id=t["match_id"], instance_id=iid, incoming=d["incoming"], chosen=d["outgoing"],
                                         expected=exp_o, probability=d["probability"], margin=d["margin"], true_pair_probability=_r(tp),
                                         evidence=d["evidence"], n_incoming=len(ref["incoming"]), gap_s=ref["difficulty"]["ambiguity_s"],
                                         difficulty=ref["difficulty"]["reasons"], heading_change_deg=ref["difficulty"]["max_heading_change_deg"],
                                         team_relation=ref["team_relation"]))
        out += sorted(rows, key=lambda x: -x["probability"])[:k]
    return out


# ---------------------------------------------------------------- outputs

def _dump(obj: Any) -> bytes:
    return json.dumps(obj, indent=1, sort_keys=True, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else str(v)).encode()


def _jsonl(rows: list[dict]) -> bytes:
    return "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows).encode()


def run_ml(raw_root: str | Path, match_ids: list[str], *, source_root: str | Path | None = None, out_root: str | Path | None = None,
           cfg: MLConfig = MLConfig()) -> tuple[Path, dict[str, Any]]:
    """Load, featurise, run the leave-one-match-out experiment and write ``<out_root>`` (default ``<processed>/m11-ml-v1``)
    all at once. An existing output directory is never replaced."""
    import joblib
    import sklearn

    t0 = time.perf_counter()
    out = Path(out_root) if out_root is not None else approved_output_root(raw_root) / OUTPUT_DIR
    if out.exists():
        raise IngestionError(f"{out} already exists and is never overwritten")
    split = _splits(raw_root)
    ids = [str(m) for m in match_ids]
    bad = [m for m in ids if split["matches"].get(m, {}).get("split") != "development"]
    if bad:
        raise IngestionError(f"only development matches are used; refused: {bad}")
    data = [load_match(raw_root, m, source_root) for m in ids]
    tables = [match_table(d, cfg) for d in data]
    exp = run_experiment(tables, cfg)
    baseline_preds = {d["match_id"]: d["baseline_predictions"] for d in data}
    summary = dict(schema=SCHEMA_VERSION, task=TASK_B, matches=ids, folds=exp["folds"], dataset={t["match_id"]: dataset_stats(t) for t in tables},
                   **summarize(tables, exp, baseline_preds, cfg), synthetic_swaps_injected=False,
                   evaluation_kind="leave-one-match-out over development matches (no validation or test match used)",
                   note="all accuracies are agreement with the provider identity reference, not verified identity accuracy")
    fails = failures(tables, exp)
    summary["failures_top"] = fails
    for name, r in exp["models"].items():
        m = r["final_model"].named_steps["clf"]
        imp = m.coef_[0] if hasattr(m, "coef_") else m.feature_importances_
        summary.setdefault("final_model_weights", {})[name] = sorted(([f, w] for f, w in zip(MODEL_FEATURES[name], np.round(imp, 4).tolist())),
                                                                     key=lambda kv: -abs(kv[1]))
    config = dict(schema=SCHEMA_VERSION, task=TASK_B, ml=cfg.as_dict(), features=FEATURES, core_features=list(CORE_FEATURES),
                  footballtrace_features=list(FT_FEATURES), model_features={k: list(v) for k, v in MODEL_FEATURES.items()},
                  models=dict(lr="StandardScaler + LogisticRegression(class_weight=balanced)", rf="RandomForest(class_weight=balanced_subsample)",
                              lr_ft="lr + FOOTBALLTrace-inspired features (ablation)", rf_ft="rf + FOOTBALLTrace-inspired features (ablation)",
                              lr_distance="proximity-only control: gap_s and dist_m"),
                  label_protocol=dict(positive="same provider identity within one encounter", negative="different provider identities within the same encounter",
                                      train=f"{LABEL_TIER} only, pairs without ESTIMATED in either endpoint window", exploratory=EXPLORATORY_TIER,
                                      unverified=f"{UNVERIFIED_TIER}: never labelled or scored"),
                  decision=dict(MATCHED="assigned in the one-to-one log-odds assignment, p >= threshold and margin >= margin_logit",
                                UNCERTAIN="a reachable candidate exists but is unassigned, below threshold or within margin",
                                UNMATCHED="no reachable outgoing tracklet (dist > reach_base + max_speed * gap)"),
                  source=dict(schema=SOURCE_SCHEMA, directory=SOURCE_DIR))
    pair_rows, label_rows, pred_rows = [], [], []
    for t in tables:
        for n, (x, lb) in enumerate(zip(t["rows"], t["labels"])):
            pair_rows.append(dict(x, features={f: _r(t["X"][n, k], 5) for k, f in enumerate(ALL_FEATURES)},
                                  heldout_probability={name: _r(r["p"][t["match_id"]][n]) for name, r in exp["models"].items()}))
            label_rows.append(dict(lb, match_id=t["match_id"]))
        for name, r in exp["models"].items():
            for iid, ds in r["decisions"][t["match_id"]].items():
                pred_rows.append(dict(model=name, match_id=t["match_id"], instance_id=iid, threshold=r["thresholds"][t["match_id"]]["threshold"], decisions=ds))
    blobs = {"config.json": _dump(config), "summary.json": _dump(summary), "pairs.jsonl": _jsonl(pair_rows),
             "pair_labels.jsonl": _jsonl(label_rows), "predictions.jsonl": _jsonl(pred_rows), "failures.jsonl": _jsonl(fails)}
    tmp = out.parent / f".{out.name}.partial-{uuid.uuid4().hex[:8]}"
    tmp.mkdir(parents=True)
    try:
        for name, b in blobs.items():
            with open(tmp / name, "xb") as f:
                f.write(b)
        for name, r in exp["models"].items():
            for mid, model in list(r["fold_models"].items()) + [("final", r["final_model"])]:
                d = tmp / "models" / (f"heldout-{mid}" if mid != "final" else "final")
                d.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, d / f"{name}.joblib")
        digests = {str(p.relative_to(tmp)).replace("\\", "/"): _sha256(p) for p in sorted(tmp.rglob("*")) if p.is_file()}
        manifest = dict(schema=SCHEMA_VERSION, matches=ids, split_manifest=dict(path=str(approved_output_root(raw_root) / SPLITS_DIR / MANIFEST_NAME),
                        sha256=_sha256(approved_output_root(raw_root) / SPLITS_DIR / MANIFEST_NAME)),
                        sources={d["match_id"]: d["provenance"] for d in data}, outputs_sha256=digests, code_sha256=code_digest(),
                        coordinate_policy="raw", provider_smoothed_coordinates_used=False, synthetic_swaps_injected=False,
                        validation_or_test_matches_used=False, processing=dict(seconds=round(time.perf_counter() - t0, 1)),
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
