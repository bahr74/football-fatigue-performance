"""Task D: four-model benchmark readiness (read-only audit of saved training records + CPU dry-run validation of a proposed
experiment matrix). Nothing is trained, no checkpoint is opened for writing, no model is run.

D1 reads the JSON training records of Stage 4 (``m11-stage4-benchmarks-v1/preds/g0p80/<model>/seed*.json``) and Stage 4C
(``m11-stage4c-robustness-v1/neural/<config>/seed*/fold*.json``, ``hgb/<config>/fold*.json``) and tabulates budgets, convergence,
seeds, folds and tuning. D3/D4 build a proposed, deterministic partition (outer fold -> tuning inner folds, early-stop matches and a
separate calibration match group) and an experiment matrix, and validate both against the frozen split with explicit rules.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

import numpy as np

from football_fatigue.m11_benchmark import folds as FO

from . import common as C

OUT = "m11-phase2-model-readiness-v1"
S4 = C.STAGE4 / "preds" / "g0p80"
S4C = C.PROC / "m11-stage4c-robustness-v1"
NEURAL_4 = ("gru_gaplast", "lstm_gaplast", "mlp", "gru", "lstm")
TABULAR_4 = ("svm", "hgb", "rf", "lr", "knn")
AT_CAP_TOL = 2                      # best epoch within this many epochs of the cap = budget-bound


def _slope(v: list[float]) -> float:
    if len(v) < 3:
        return float("nan")
    x = np.arange(len(v))
    return float(np.polyfit(x, v, 1)[0])


def neural_stage4() -> dict[str, Any]:
    out = {}
    for mdl in NEURAL_4:
        recs = []
        for f in sorted((S4 / mdl).glob("seed*.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            cap = d["config"]["epochs"]
            for k, fo in d["folds"].items():
                h = fo["history"]
                vl = [e["val_log_loss"] for e in h]
                tl = [e["train_loss"] for e in h]
                recs.append(dict(seed=d["seed"], fold=int(k), cap=cap, best_epoch=fo["best_epoch"], epochs_run=fo["epochs_run"],
                                 at_cap=fo["best_epoch"] >= cap - AT_CAP_TOL, stopped_early=fo["epochs_run"] < cap,
                                 best_val_ll=min(vl), final_train_loss=tl[-1], val_slope_last10=_slope(vl[-10:]),
                                 fit_s=fo["fit_s"], device=fo.get("device"), params=fo.get("params"),
                                 val_ba=fo["val"]["balanced_accuracy"], train_ba=fo["train"]["balanced_accuracy"]))
        if not recs:
            continue
        cfg = json.loads(sorted((S4 / mdl).glob("seed*.json"))[0].read_text(encoding="utf-8"))["config"]
        out[mdl] = dict(config=cfg, n_fits=len(recs), seeds=sorted({r["seed"] for r in recs}), params=recs[0]["params"],
                        fits_at_cap=int(sum(r["at_cap"] for r in recs)), fits_stopped_early=int(sum(r["stopped_early"] for r in recs)),
                        fits_best_within_10_of_cap=int(sum(r["best_epoch"] >= r["cap"] - 10 for r in recs)),
                        best_epoch=C.q([r["best_epoch"] for r in recs], (0.0, 0.5, 1.0)),
                        val_slope_last10_median=float(np.median([r["val_slope_last10"] for r in recs])),
                        fit_s=C.q([r["fit_s"] for r in recs], (0.0, 0.5, 1.0)), devices=sorted({str(r["device"]) for r in recs}),
                        train_minus_val_ba_median=float(np.median([r["train_ba"] - r["val_ba"] for r in recs])),
                        fits=recs)
    return out


def tabular_stage4() -> dict[str, Any]:
    out = {}
    for mdl in TABULAR_4:
        fs = sorted((S4 / mdl).glob("seed*.json"))
        if not fs:
            continue
        sel, fit, gap = [], [], []
        grid = None
        for f in fs:
            d = json.loads(f.read_text(encoding="utf-8"))
            grid = d.get("grid")
            for k, fo in d["folds"].items():
                sel.append(json.dumps(fo.get("selected"), sort_keys=True))
                fit.append(fo["fit_s"])
                gap.append(fo["train"]["balanced_accuracy"] - fo["val"]["balanced_accuracy"])
        out[mdl] = dict(grid_size=len(grid) if grid else None, grid=grid, seeds=len(fs), selected_counts={s: sel.count(s) for s in sorted(set(sel))},
                        fit_s=C.q(fit, (0.0, 0.5, 1.0)), train_minus_val_ba_median=float(np.median(gap)))
    return out


def stage4c() -> dict[str, Any]:
    out = {}
    for cfg_dir in sorted(p for p in (S4C / "neural").iterdir() if p.is_dir()):
        recs = []
        for f in sorted(glob.glob(str(cfg_dir / "seed*" / "fold*.json"))):
            d = json.loads(Path(f).read_text(encoding="utf-8"))
            for cap, v in d.get("caps", {}).items():
                recs.append(dict(seed=Path(f).parent.name, fold=d["fold"], cap=int(cap), best_epoch=v["best_epoch"], epochs_run=v["epochs_run"],
                                 at_cap=v["best_epoch"] >= int(cap) - AT_CAP_TOL, es_loss=v["es_loss"], val_ba=v["val"]["balanced_accuracy"]))
        if not recs:
            continue
        caps = {}
        for cap in sorted({r["cap"] for r in recs}):
            rr = [r for r in recs if r["cap"] == cap]
            caps[str(cap)] = dict(n_fits=len(rr), fits_at_cap=int(sum(r["at_cap"] for r in rr)), val_ba_mean=float(np.mean([r["val_ba"] for r in rr])),
                                  es_loss_mean=float(np.mean([r["es_loss"] for r in rr])))
        out[cfg_dir.name] = dict(model="gru_gaplast (Stage 4C)", caps=caps)
    hgb = {}
    for cfg_dir in sorted(p for p in (S4C / "hgb").iterdir() if p.is_dir()):
        fs = sorted(cfg_dir.glob("fold*.json"))
        hgb[cfg_dir.name] = dict(n_folds=len(fs))
    out["hgb_configs"] = hgb
    return out


# ---------------------------------------------------------------- D3 / D4 proposed partition and matrix (no training)

def proposed_partition(matches: list[str], n_cal: int = 3, n_es: int = 3) -> dict[str, Any]:
    """Per outer fold: test = the frozen outer fold (4 matches); calibration = n_cal training matches; early stopping = n_es of the
    remaining; tuning inner folds = 3 grouped folds over the rest. Deterministic by salted sha256 rank (new salts; the frozen outer
    folds are reused unchanged)."""
    fmap = FO.outer_folds(matches)
    parts = {}
    for f in range(FO.N_OUTER):
        test = sorted((m for m, k in fmap.items() if k == f), key=int)
        train = sorted((m for m, k in fmap.items() if k != f), key=int)
        cal = FO._rank(train, f"m11-phase2-cal-v1|{f}")[:n_cal]
        rest = [m for m in train if m not in cal]
        es = FO._rank(rest, f"m11-phase2-es-v1|{f}")[:n_es]
        fit = [m for m in rest if m not in es]
        inner = FO.inner_folds(fit, f)
        parts[str(f)] = dict(test=test, calibration=sorted(cal, key=int), early_stop=sorted(es, key=int), fit=fit, inner=inner)
    return parts


def experiment_matrix() -> list[dict[str, Any]]:
    """Proposed experiments (not run). Priority 1 = needed for a fair four-family claim; 2 = robustness; 3 = exploratory."""
    fam = dict(gru=dict(kind="neural", seeds=5, epoch_cap=400, patience=25, tuning_configs=24),
               lstm=dict(kind="neural", seeds=5, epoch_cap=400, patience=25, tuning_configs=24),
               svm=dict(kind="tabular", seeds=1, tuning_configs=24), hgb=dict(kind="tabular", seeds=3, tuning_configs=24))
    rows = []
    base = dict(table="g0p80", population="stage3_v1_0p5m", train_dist="centred", eval_sets=["primary", "stress_suite"])
    pid = 0

    def add(priority, purpose, **kw):
        nonlocal pid
        for f, spec in fam.items():
            pid += 1
            rows.append(dict(id=f"E{pid:03d}", priority=priority, purpose=purpose, family=f, **spec, **(base | kw)))

    add(1, "fair re-benchmark, identical population and folds, equal tuning budget")
    add(1, "shift-augmented training offered to all families", train_dist="shift_aug")
    add(2, "variable-gap training 0.2-2.4 s", train_dist="variable_gap", eval_sets=["g0p40", "g0p80", "g1p60", "stress_suite"])
    add(2, "noise / dropout robustness training", train_dist="noise_dropout")
    add(2, "clone-like exclusion sensitivity (train and evaluate without clone-like pair-periods)", population="stage3_v1_excl_clone")
    add(3, "ambiguity-enriched population (requires dataset v2 approval)", population="dataset_v2_candidate", eval_sets=["v2_dev_stratified"])
    for b in ("physics_cv", "physics_ca", "kalman_fwd", "kalman_bidir", "nearest"):
        pid += 1
        rows.append(dict(id=f"E{pid:03d}", priority=1, purpose="deterministic baseline (no training)", family=b, kind="deterministic",
                         seeds=1, tuning_configs=0, **base))
    return rows


RULES = ("all four families present at priority 1", "neural seeds >= 5", "equal tuning budget per family within a purpose",
         "neural epoch cap >= 2x historical best epoch and patience recorded", "calibration matches disjoint from fit, early-stop and test",
         "every partition match is train/CV", "outer folds equal the frozen Stage 4 folds", "test folds partition the 32 matches")


def validate(parts: dict[str, Any], matrix: list[dict[str, Any]], allowed: list[str], frozen_outer: dict[str, int]) -> dict[str, Any]:
    res: dict[str, Any] = {}
    p1 = {r["family"] for r in matrix if r["priority"] == 1}
    res[RULES[0]] = {"gru", "lstm", "svm", "hgb"} <= p1
    res[RULES[1]] = all(r["seeds"] >= 5 for r in matrix if r.get("kind") == "neural")
    by = {}
    for r in matrix:
        if r.get("kind") in ("neural", "tabular"):
            by.setdefault(r["purpose"], set()).add(r["tuning_configs"])
    res[RULES[2]] = all(len(v) == 1 for v in by.values())
    res[RULES[3]] = all(r["epoch_cap"] >= 160 and r.get("patience") for r in matrix if r.get("kind") == "neural")
    ok = True
    for f, p in parts.items():
        s = [set(p[k]) for k in ("test", "calibration", "early_stop", "fit")]
        ok &= all(not (s[i] & s[j]) for i in range(4) for j in range(i + 1, 4))
    res[RULES[4]] = bool(ok)
    allm = {m for p in parts.values() for k in ("test", "calibration", "early_stop", "fit") for m in p[k]}
    res[RULES[5]] = allm <= set(allowed)
    res[RULES[6]] = all(frozen_outer[m] == int(f) for f, p in parts.items() for m in p["test"])
    tests = [m for p in parts.values() for m in p["test"]]
    res[RULES[7]] = sorted(tests, key=int) == sorted(allowed, key=int) and len(tests) == len(set(tests))
    res["all_passed"] = all(v for k, v in res.items() if k in RULES)
    return res


def compute_estimate(neural: dict[str, Any], matrix: list[dict[str, Any]]) -> dict[str, Any]:
    """GPU-hours from historical seconds per epoch (Stage 4 fits on the recorded device), scaled to the proposed caps. [EST]"""
    spe = {}
    for mdl in ("gru_gaplast", "lstm_gaplast"):
        r = neural.get(mdl)
        if r:
            spe[mdl] = float(np.median([f["fit_s"] / f["epochs_run"] for f in r["fits"]]))
    est = []
    for row in matrix:
        if row.get("kind") != "neural":
            continue
        s = spe["gru_gaplast" if row["family"] == "gru" else "lstm_gaplast"]
        mult = 7 if row["train_dist"] in ("shift_aug",) else (4 if row["train_dist"] in ("variable_gap", "noise_dropout") else 1)
        cap = row["epoch_cap"]
        # scenario A, full nested: every config x 3 inner folds x 8 outer folds, mean epochs 0.5 x cap
        tune_a = row["tuning_configs"] * 3 * 0.5 * cap * 8
        # scenario B, budgeted: successive halving (eta 3) on one inner split per outer fold: 24 x cap/9 + 8 x cap/3 + 3 x 0.5 cap
        tune_b = (row["tuning_configs"] * cap / 9 + (row["tuning_configs"] / 3) * cap / 3 + 3 * 0.5 * cap) * 8
        final = row["seeds"] * 8 * 0.5 * cap
        h = lambda ep: round(ep * s * mult / 3600, 1)  # noqa: E731
        est.append(dict(id=row["id"], family=row["family"], purpose=row["purpose"], priority=row["priority"],
                        tuning_full_nested_gpu_h=h(tune_a), tuning_budgeted_gpu_h=h(tune_b), final_gpu_h=h(final)))
    tot = lambda key, pr: round(sum(r[key] + r["final_gpu_h"] for r in est if pr is None or r["priority"] == pr), 1)  # noqa: E731
    return dict(seconds_per_epoch_historical=spe, rows=est,
                assumption="mean epochs = 0.5 x cap under patience; augmentation multiplies epoch cost by the variant factor (shift 7,"
                           " gap/noise 4); seconds per epoch from the Stage 4 records on the recorded device (GTX 1650 Ti). [EST]",
                priority1_full_nested_gpu_h=tot("tuning_full_nested_gpu_h", 1), priority1_budgeted_gpu_h=tot("tuning_budgeted_gpu_h", 1),
                all_full_nested_gpu_h=tot("tuning_full_nested_gpu_h", None), all_budgeted_gpu_h=tot("tuning_budgeted_gpu_h", None))


def main() -> dict[str, Any]:
    allowed = C.checked(None)
    out = C.Out(OUT)
    frozen = json.loads((C.STAGE4 / "folds.json").read_text(encoding="utf-8"))["outer"]
    neural = neural_stage4()
    R = dict(neural_stage4={k: {kk: vv for kk, vv in v.items() if kk != "fits"} for k, v in neural.items()},
             tabular_stage4=tabular_stage4(), stage4c=stage4c())
    parts = proposed_partition(allowed)
    matrix = experiment_matrix()
    R["validation"] = validate(parts, matrix, allowed, frozen)
    R["compute_estimate"] = compute_estimate(neural, matrix)
    out.json("readiness_results.json", R)
    out.json("neural_fit_records.json", {k: v["fits"] for k, v in neural.items()})
    out.json("proposed_partition.json", parts)
    out.json("proposed_experiment_matrix.json", matrix)
    out.manifest(task="Phase 2 Task D readiness (read-only audit + dry-run validation; nothing trained)",
                 inputs=dict(stage4=str(S4.relative_to(C.ROOT)), stage4c=str(S4C.relative_to(C.ROOT)), folds_sha256=C.sha256_file(C.STAGE4 / "folds.json")),
                 code=C.code_sha256(["readiness.py", "common.py"]), command="python -m football_fatigue.m11_phase2 readiness")
    return R


if __name__ == "__main__":
    print(json.dumps(main()["validation"], indent=1))
