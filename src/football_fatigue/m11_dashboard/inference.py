"""Frozen Step 6 inference on one matcher-visible task B instance, through the unchanged Step 6 code.

The chain is exactly ``validate.run_predict`` restricted to one instance and to the frozen model:
``validate.features_only`` (51 frozen features chosen from ``ml2.FEATURES2``) -> forest probability -> stored Platt calibrator
-> ``validate.decide`` (one-to-one log-odds assignment, then MATCHED / UNCERTAIN / UNMATCHED) at the operating points of the frozen
specification (relaxed 0.76 / 1.0, conservative 0.87 / 1.0). Features depend on one instance only, so the result equals the
batch pipeline's. Nothing is fitted, tuned or written, and no reference is read.

The only addition is :func:`path_contributions`, a read-only decomposition of the forest probability along each tree's decision
path (Saabas / "treeinterpreter"); it does not change any prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from football_fatigue.matching import ml, ml2, validate
from football_fatigue.matching.locked_test import FROZEN_MODEL_SHA256, FROZEN_SPEC_SHA256

from .data import DashboardError, processed_root

N_FEATURES = 51
POINTS = ("relaxed", "conservative")


@dataclass(frozen=True)
class FrozenModel:
    spec: dict
    bundle: dict
    model_sha256: str
    spec_sha256: str
    directory: Path

    @property
    def features(self) -> tuple[str, ...]:
        return tuple(self.bundle["features"])

    @property
    def operating_points(self) -> dict[str, dict]:
        """From the frozen specification (not ``validate.OPERATING``, whose relaxed 0.75 was resolved to 0.76 at freeze time)."""
        return {k: dict(threshold=float(v["threshold"]), margin=float(v["margin"])) for k, v in self.spec["operating_points"].items()}

    @property
    def forest(self):
        return self.bundle["model"].steps[-1][1]


def load_frozen_model(raw_root=None, *, frozen_dir=None, expected=(FROZEN_MODEL_SHA256, FROZEN_SPEC_SHA256)) -> FrozenModel:
    """The checksum-verified frozen model. ``expected`` (model, spec SHA-256) defaults to the values recorded at Step 7; tests
    with a synthetic bundle pass ``None``."""
    from football_fatigue.matching.benchmark import _sha256

    d = Path(frozen_dir) if frozen_dir is not None else processed_root(raw_root) / validate.FROZEN_DIR
    if not (d / "manifest.json").exists() or not (d / "model.joblib").exists():
        raise DashboardError(f"frozen model artifacts are missing in {d} (expected manifest.json, model.joblib, spec.json)")
    try:
        spec, bundle, model_sha = validate.load_frozen(d)
    except Exception as e:
        raise DashboardError(f"the frozen model in {d} failed verification: {e}") from None
    spec_sha = _sha256(d / "spec.json")
    if expected is not None and (model_sha, spec_sha) != tuple(expected):
        raise DashboardError(f"the frozen model or specification in {d} differs from the recorded Step 6/7 checksums")
    fm = FrozenModel(spec, bundle, model_sha, spec_sha, d)
    # the forest was pickled with n_jobs=2: threads add the per-tree probabilities in varying order, so repeated calls differ by
    # ~1e-16. One thread fixes the order (bit-identical repeats); the file is untouched and no prediction changes beyond rounding.
    fm.forest.set_params(n_jobs=1)
    if len(fm.features) != N_FEATURES or fm.bundle["model"].n_features_in_ != N_FEATURES:
        raise DashboardError(f"the frozen model expects {fm.bundle['model'].n_features_in_} features, not {N_FEATURES}")
    if set(fm.operating_points) != set(POINTS):
        raise DashboardError(f"unexpected operating points {sorted(fm.operating_points)}")
    return fm


def infer_instance(inst: dict, fm: FrozenModel, cfg: ml.MLConfig = ml.MLConfig()) -> dict[str, Any]:
    """Pair probabilities and decisions of one task B instance (matcher-visible fields only)."""
    if inst.get("during", {}).get("t"):
        raise DashboardError("the instance contains withheld-interval observations; only task B instances are accepted")
    rows, X = validate.features_only([inst], cfg)
    if not rows:
        raise DashboardError(f"instance {inst['instance_id']} has no candidate pair")
    Xf = X[:, ml2.cols(fm.features)]
    raw = ml2.proba(fm.bundle["model"], Xf)
    p = ml2.Platt(**fm.bundle["calibrator"])(raw)
    decisions = {}
    for pt, op in fm.operating_points.items():
        decisions[pt] = validate.decide(rows, p, X, op["threshold"], op["margin"], cfg)[inst["instance_id"]]
    pairs = [dict(incoming=r["incoming"], outgoing=r["outgoing"], reachable=bool(r["reachable"]), probability=float(p[k]),
                  raw_probability=float(raw[k]), features={f: float(Xf[k, n]) for n, f in enumerate(fm.features)})
             for k, r in enumerate(rows)]
    return dict(instance_id=inst["instance_id"], pairs=pairs, decisions=decisions, operating_points=fm.operating_points, X=X, Xf=Xf, rows=rows)


def check_assignment(result: dict) -> list[str]:
    """Invariants of the frozen decision rule; an empty list when all hold."""
    problems = []
    pairs = {(q["incoming"], q["outgoing"]): q for q in result["pairs"]}
    for pt, ds in result["decisions"].items():
        op = result["operating_points"][pt]
        used = [d["outgoing"] for d in ds if d["decision"] == "MATCHED"]
        if len(used) != len(set(used)):
            problems.append(f"{pt}: an outgoing tracklet is matched twice")
        if len({d["incoming"] for d in ds}) != len(ds):
            problems.append(f"{pt}: an incoming tracklet is decided twice")
        for d in ds:
            if d["decision"] == "MATCHED":
                q = pairs[(d["incoming"], d["outgoing"])]
                if not q["reachable"] or d["probability"] < op["threshold"] - 5e-5 or d["margin"] < op["margin"] - 5e-5:
                    problems.append(f"{pt}: {d['incoming']} MATCHED without reachability, threshold or margin")
            elif d["decision"] == "UNMATCHED" and any(q["reachable"] for (i, _), q in pairs.items() if i == d["incoming"]):
                problems.append(f"{pt}: {d['incoming']} UNMATCHED although a candidate is reachable")
    return problems


# ---------------------------------------------------------------- explanation (read-only)

def path_contributions(fm: FrozenModel, x: np.ndarray) -> dict[str, Any]:
    """Decomposition of the forest probability of one pair: ``bias + sum(contributions) == raw probability``.

    For every tree, the change of the positive-class fraction from a node to the child on the decision path is credited to the
    node's split feature (Saabas); the forest value is the mean over trees. It describes how this forest arrived at its raw
    probability, before Platt calibration; it is not a causal effect and is biased towards features split near the root."""
    rf = fm.forest
    x2 = np.asarray(x, np.float32).reshape(1, -1)
    k1 = list(rf.classes_).index(1)
    contrib = np.zeros(x2.shape[1])
    bias = 0.0
    for est in rf.estimators_:
        tree = est.tree_
        v = tree.value[:, 0, :]
        v1 = v[:, k1] / v.sum(axis=1)
        path = est.decision_path(x2).indices          # node ids increase from the root to the leaf
        bias += v1[path[0]]
        np.add.at(contrib, tree.feature[path[:-1]], v1[path[1:]] - v1[path[:-1]])
    n = len(rf.estimators_)
    bias, contrib = bias / n, contrib / n
    raw = float(fm.bundle["model"].predict_proba(np.asarray(x, float).reshape(1, -1))[0, k1])
    return dict(bias=float(bias), contributions=dict(zip(fm.features, map(float, contrib))), raw_probability=raw,
                additive_error=abs(bias + contrib.sum() - raw))


def global_importance(fm: FrozenModel) -> dict[str, float]:
    """Impurity-based (MDI) importances stored in the frozen forest: training-data, global, not per decision."""
    return dict(zip(fm.features, map(float, fm.forest.feature_importances_)))


FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "Spatial proximity": ("dist_m", "dist_per_s", "reach_ratio", "in_sep_m", "out_sep_m"),
    "Gap duration": ("gap_s",),
    "Velocity continuity": ("speed_in", "speed_out", "speed_diff", "vel_diff", "speed_consistency", "hermite_peak_speed", "acc_in", "acc_out"),
    "Heading consistency": ("heading_diff_deg", "both_moving", "turn_in_deg", "turn_out_deg", "turn_acc_demand", "turn_rate_in", "turn_rate_out",
                            "straight_in", "straight_out"),
    "Extrapolation error": ("cv_fwd_err", "cv_bwd_err", "cv_mid_err", "ca_fwd_err", "ca_bwd_err", "ca_mid_err", "req_acc", "hermite_max_acc",
                            "hermite_rms_acc"),
    "Global assignment ambiguity": ("assign_regret_cv", "assign_regret_hermite", "n_reach_row", "n_reach_col", "n_incoming", "n_outgoing",
                                    "in_closing_m_s") + tuple(f"{s}_{k}" for k in ml.REL_KEYS for s in ("row_gap", "col_gap")),
    "Observation quality": ("n_upd_in", "n_upd_out", "est_share_in", "est_share_out"),
}
DESCRIPTIONS = {**ml.FEATURES, **ml2.FOOTBALL_FEATURES}


def feature_group(name: str) -> str:
    return next((g for g, fs in FEATURE_GROUPS.items() if name in fs), "Other")


# ---------------------------------------------------------------- batch reproduction (CLI ``verify`` and tests)

def batch_reference(instances: list[dict], fm: FrozenModel, match_id: str = "m") -> tuple[list[dict], list[dict]]:
    """The Step 6 batch path (``validate.predict_tables``, frozen model only) on whole instances."""
    rows, X = validate.features_only(instances)
    for r in rows:
        r["match_id"] = match_id
    return validate.predict_tables([dict(match_id=match_id, rows=rows, X=X)], fm.spec, fm.bundle, {})


def compare_with_batch(instances: list[dict], fm: FrozenModel) -> dict[str, Any]:
    """Max probability difference and decision mismatches between :func:`infer_instance` and the batch path."""
    pair_rows, preds = batch_reference(instances, fm)
    batch_p = {(r["instance_id"], r["incoming"], r["outgoing"]): r["probability"]["frozen"] for r in pair_rows}
    batch_d = {(p["method"], p["instance_id"]): p["decisions"] for p in preds}
    max_diff, mismatches, invariant = 0.0, [], []
    for inst in instances:
        res = infer_instance(inst, fm)
        for q in res["pairs"]:
            max_diff = max(max_diff, abs(round(q["probability"], 6) - batch_p[(inst["instance_id"], q["incoming"], q["outgoing"])]))
        for pt, ds in res["decisions"].items():
            if ds != batch_d[(f"frozen@{pt}", inst["instance_id"])]:
                mismatches.append((inst["instance_id"], pt))
        invariant += [f"{inst['instance_id']}: {m}" for m in check_assignment(res)]
    return dict(instances=len(instances), pairs=len(batch_p), max_abs_probability_diff=max_diff, decision_mismatches=mismatches,
                assignment_problems=invariant)
