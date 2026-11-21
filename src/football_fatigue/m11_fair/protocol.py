"""Fair benchmark protocol: partition (re-validated), equal-budget configurations, controls, compute estimate and the approval
package written to ``data/processed/m11-fair-benchmark-plan-v1`` (write-once)."""

from __future__ import annotations

from typing import Any

from football_fatigue.m11_benchmark import folds as FO
from football_fatigue.m11_phase2 import common as C
from football_fatigue.m11_phase2 import readiness as RD

from . import spaces as SP
from .nets import NetConfig, make_pair_net, n_params

OUT = "m11-fair-benchmark-plan-v1"
NEURAL_SEEDS = (0, 1, 2, 3, 4)
TABULAR_SEEDS = (0, 1, 2)
# measured seconds per epoch on the GTX 1650 Ti (Stage 4 records; 1-layer historical sizes) [m11-phase2-model-readiness-v1]
S_PER_EPOCH = {"gru": 1.24, "lstm": 3.15}
CONTROLS = ["independent-event evaluation (greedy non-overlapping subset)", "both-VISIBLE reference stratum",
            "mixed-status and ESTIMATED-dominated strata", "clone-like pair-period sensitivity (exclude and re-evaluate)",
            "texture / smoothing ablations (noise on all inputs; smoothing on all inputs)", "noise and dropout stress cases",
            "centred vs shifted gaps (g0p80 vs g0p80_shift)", "longer gaps (g1p60) and shorter (g0p40)",
            "physics-hard / contested / physics-easy subsets (Phase 2 consensus tiers; evaluation only)",
            "deterministic baselines as reference rows (kf_fwd, kf_bidir, cv_fwd, ca_fwd, nearest)"]


def partition() -> dict[str, Any]:
    return RD.proposed_partition(C.allowed())


def validate_partition(parts: dict[str, Any]) -> dict[str, Any]:
    allowed = set(C.allowed())
    frozen = FO.outer_folds(sorted(allowed, key=int))
    checks = {}
    tests = [m for f in parts.values() for m in f["test"]]
    checks["test folds partition the 32 train/CV matches"] = sorted(tests, key=int) == sorted(allowed, key=int) and len(tests) == len(set(tests))
    checks["test folds equal the frozen Stage 4 outer folds"] = all(frozen[m] == int(f) for f, p in parts.items() for m in p["test"])
    ok_dis, ok_cover, ok_sizes, ok_inner = True, True, True, True
    for f, p in parts.items():
        roles = [set(p[k]) for k in ("fit", "early_stop", "calibration", "test")]
        if any(a & b for i, a in enumerate(roles) for b in roles[i + 1:]):
            ok_dis = False
        if set().union(*roles) != allowed:
            ok_cover = False
        if (len(p["early_stop"]), len(p["calibration"]), len(p["test"]), len(p["fit"])) != (3, 3, 4, 22):
            ok_sizes = False
        if set(p["inner"]) != set(p["fit"]):
            ok_inner = False
    checks["roles match-disjoint in every outer fold"] = ok_dis
    checks["roles cover exactly the 32 train/CV matches"] = ok_cover
    checks["sizes fit 22 / early-stop 3 / calibration 3 / test 4"] = ok_sizes
    checks["inner tuning folds cover exactly the fit matches"] = ok_inner
    checks["no protected match in any role"] = not (set().union(*[set(m) for p in parts.values() for m in p.values() if isinstance(m, list)]) - allowed)
    return dict(passed=all(checks.values()), checks=checks)


def neural_cost(cfg: dict[str, Any], family: str) -> float:
    """Seconds per epoch scaled from the measured 1-layer historical size by parameter count (recurrent cost ~ params) [EST]."""
    base = n_params(make_pair_net(NetConfig(family, 48 if family == "gru" else 40, 48)))
    this = n_params(make_pair_net(NetConfig(family, int(cfg["hidden"]), 48, int(cfg["layers"]))))
    return S_PER_EPOCH[family] * this / base


def compute(configs: dict[str, list[dict[str, Any]]], mean_epoch_frac: float = 0.5, inner_folds: int = 3, budgeted_eta: int = 3) -> dict[str, Any]:
    """GPU-hours [EST]: tuning (configs x inner folds x outer folds) + final (5 seeds x outer folds), mean epochs = frac x cap."""
    out = {}
    for fam in ("gru", "lstm"):
        per_cfg = [neural_cost(c, fam) * c["epochs"] * mean_epoch_frac for c in configs[fam]]
        full = sum(per_cfg) * inner_folds * FO.N_OUTER / 3600
        budget = sum(per_cfg) * FO.N_OUTER / 3600 * (1 / budgeted_eta + 1 / budgeted_eta ** 2 * 2)   # successive halving, 1 inner split
        final = (sum(per_cfg) / len(per_cfg)) * len(NEURAL_SEEDS) * FO.N_OUTER / 3600
        out[fam] = dict(tuning_full_gpu_h=round(full, 1), tuning_budgeted_gpu_h=round(budget, 1), final_5_seeds_gpu_h=round(final, 1))
    return out


def plan() -> dict[str, Any]:
    parts = partition()
    val = validate_partition(parts)
    configs = {f: SP.sample(f) for f in ("gru", "lstm", "svm_nystroem", "svm_rbf_exact", "svm_linear", "hgb")}
    params = {f: [n_params(make_pair_net(NetConfig(f, int(c["hidden"]), int(c["emb"]), int(c["layers"])))) for c in configs[f]] for f in ("gru", "lstm")}
    return dict(schema="m11-fair-benchmark-plan-v1", status="PROPOSED - not run; GPU training requires explicit approval",
                population="Stage 3 m11-stage3-samples-v1 (g0p80 primary; g0p40, g1p60, g0p80_shift evaluation), 32 train/CV matches",
                partition=parts, partition_validation=val, search=dict(n_configs_per_family=SP.N_CONFIGS, spaces=SP.SPACES, fixed=SP.FIXED,
                                                                         configs=configs, neural_param_counts=params),
                seeds=dict(neural=list(NEURAL_SEEDS), tabular=list(TABULAR_SEEDS)),
                selection=("per outer fold: configuration chosen by inner-fold (3 folds over fit matches) early-stop log loss for neural, "
                           "inner-fold AUC for tabular; never by test predictions"),
                calibration="temperature (neural, HGB) / Platt (SVM), symmetric no-intercept, fitted on the 3 calibration matches only",
                abstention="thresholds fitted on calibration matches; risk-coverage reported on test",
                comparisons=["parameter-matched GRU vs LSTM (same hidden budget)", "compute-matched GRU vs LSTM (equal GPU seconds)",
                             "best-of-search per family", "paired match bootstrap differences (2,000 replicates)"],
                input_access=("GRU / LSTM: observed tracklet sequences (x, y, mask, dt); SVM / HGB: 36 motion features computed only from the "
                              "same observed tracklets; provider status, identities and label-derived tiers are never inputs"),
                controls=CONTROLS, compute_estimate_gpu_h=compute(configs),
                commands=dict(validate="python -m football_fatigue.m11_fair plan",
                              cpu_smoke="python -m football_fatigue.m11_fair smoke",
                              tabular_cpu_pilot="python -m football_fatigue.m11_fair cpu-pilot",
                              neural_gpu="python -m football_fatigue.m11_fair train --family lstm --approval <id>   (refused without --approval)"),
                outputs=dict(plan=f"data/processed/{OUT}", pilot="data/processed/m11-fair-cpu-pilot-v1",
                             training="data/processed/m11-fair-benchmark-v1 (future; new checkpoints, never the Stage 4C ones)"))


def main() -> dict[str, Any]:
    p = plan()
    if not p["partition_validation"]["passed"]:
        raise SystemExit(f"partition validation failed: {p['partition_validation']}")
    out = C.Out(OUT)
    out.json("plan.json", p)
    split = C.PROC / "m11-stage2-split-v1" / "split.json"
    out.manifest(inputs={"split": dict(path=str(split.relative_to(C.ROOT)), sha256=C.sha256_file(split))},
                 parameters=dict(n_configs=SP.N_CONFIGS, search_salt="m11-fair-search-v1"), code=code_hashes())
    return p


def code_hashes() -> dict[str, str]:
    from pathlib import Path
    here = Path(__file__).resolve().parent
    return {f"m11_fair/{p.name}": C.sha256_file(p) for p in sorted(here.glob("*.py"))}
