"""CPU-only PILOT of the fair tabular protocol (SVM variants and HGB) on the 32 development matches.

PILOT, not the fair benchmark: reduced equal budget (``PILOT_CONFIGS`` per family, seed 0), configuration selection on the early-stop
split, calibration on separate matches, test = frozen outer folds. No model file is saved (no checkpoint for live inference).
Output: ``data/processed/m11-fair-cpu-pilot-v1`` (write-once)."""

from __future__ import annotations

import json
import time
from multiprocessing import Pool
from typing import Any

import numpy as np
import pandas as pd

from football_fatigue.m11_benchmark import data as D
from football_fatigue.m11_benchmark import metrics as MT
from football_fatigue.m11_phase2 import common as C

from . import protocol as PR
from . import runner as RU
from . import spaces as SP
from .train import paired_ba

OUT = "m11-fair-cpu-pilot-v1"
FAMILIES = ("hgb", "svm_linear", "svm_nystroem", "svm_rbf_exact")
PILOT_CONFIGS = 8
PILOT_HGB_MAX_ITER = 1000
_DATA: RU.DevData | None = None


def _init():
    global _DATA
    _DATA = RU.load_dev("g0p80", sequences=False)


def _job(args):
    family, fold, part = args
    cfgs = SP.sample(family, PILOT_CONFIGS, salt="m11-fair-cpu-pilot-v1")
    for c in cfgs:
        if family == "hgb":
            c["max_iter"] = PILOT_HGB_MAX_ITER            # pilot cap (plan: 2000); best iteration chosen on early-stop matches
    return RU.run_fold(family, fold, part, _DATA, cfgs, seeds=(0,), log=lambda s: print(s, flush=True))


def evaluate(data: RU.DevData, preds: dict[str, np.ndarray]) -> dict[str, Any]:
    hist = D.out_dir() / "preds" / "g0p80"
    ps = pd.read_parquet(C.PROC / "m11-phase2-ambiguity-benchmark-v1" / "per_sample_g0p80.parquet")
    if len(ps) != len(data.y) or not np.array_equal(ps["match_id"].astype(str).to_numpy(), data.groups) or not np.array_equal(ps["y"].to_numpy(), data.y):
        raise SystemExit("Phase 2 per-sample table is not row-aligned")
    kf = (ps["margin_kf_fwd"].to_numpy() < 0).astype(float)                   # P(SWAPPED) >= 0.5 iff margin < 0
    ref = {"historical_hgb_seed0": D.load_npz(hist / "hgb" / "seed0.npz")["p"], "historical_svm_seed0": D.load_npz(hist / "svm" / "seed0.npz")["p"],
           "historical_gru_gaplast_seed0": D.load_npz(hist / "gru_gaplast" / "seed0.npz")["p"], "kalman_kf_fwd": kf}
    ind = data.indep
    strata = dict(status=ps["status_in_gap"].astype(str).to_numpy(), physics_tier=ps["physics_tier"].astype(str).to_numpy())
    res = {}
    for k, p in {**preds, **ref}.items():
        ok = np.isfinite(p)
        m = ind & ok
        if k == "kalman_kf_fwd":                                          # decisions only: no probability metrics
            r = dict(independent=dict(balanced_accuracy=round(MT.bal_acc(data.y[m], p[m]), 4), n=int(m.sum())))
        else:
            r = dict(independent=dict(MT.basic(data.y[m], p[m]), n=int(m.sum()), bootstrap=MT.bootstrap(data.y[m], p[m], data.groups[m])))
        r["strata_ba"] = {f"{s}={v}": round(MT.bal_acc(data.y[ind & (a == v)], p[ind & (a == v)]), 4)
                          for s, a in strata.items() for v in np.unique(a[ind])}
        res[k] = r
    paired = {}
    for k, p in preds.items():
        for rname in ref:
            paired[f"{k} - {rname}"] = paired_ba(data.y[ind], p[ind], ref[rname][ind], data.groups[ind])
    return dict(models=res, paired_ba_independent=paired)


def main(workers: int = 6) -> dict[str, Any]:
    t0 = time.perf_counter()
    out = C.Out(OUT)
    parts = PR.partition()
    val = PR.validate_partition(parts)
    if not val["passed"]:
        raise SystemExit(val)
    jobs = [(f, k, parts[k]) for f in FAMILIES for k in sorted(parts, key=int)]
    with Pool(workers, initializer=_init) as pool:
        results = pool.map(_job, jobs, chunksize=1)
    _init()
    data = _DATA
    by = {f: [r for r in results if r["family"] == f] for f in FAMILIES}
    preds = {f"pilot_{f}": RU.assemble(rs, len(data.y))["p"] for f, rs in by.items()}
    for f, rs in by.items():
        out.json(f"folds_{f}.json", rs)
    out.parquet("pilot_predictions.parquet", pd.DataFrame(dict(match_id=data.groups, y=data.y, independent=data.indep, **preds)))
    ev = evaluate(data, preds)
    summary = dict(schema="m11-fair-cpu-pilot-v1", status="PILOT (reduced budget, CPU, seed 0) - not the fair benchmark",
                   budget=dict(configs_per_family=PILOT_CONFIGS, seeds=[0], hgb_max_iter=PILOT_HGB_MAX_ITER, selection="early-stop AUC (single split)",
                               calibration="symmetric temperature / Platt on 3 calibration matches"),
                   partition_validation=val, selected={f: [dict(fold=r["fold"], config=r.get("selected"), edge_flags=r.get("selected_edge_flags"))
                                                         for r in rs] for f, rs in by.items()},
                   fit_seconds={f: float(np.sum([t["s"] for r in rs for t in r.get("trials", [])])) for f, rs in by.items()},
                   failed_trials={f: [dict(fold=r["fold"], config=t["config_id"], error=t["error"]) for r in rs for t in r.get("trials", []) if "error" in t]
                                  for f, rs in by.items()},
                   evaluation=ev, runtime_s=round(time.perf_counter() - t0, 1))
    out.json("summary.json", summary)
    out.manifest(inputs=dict(stage3=str(D.stage3_dir().relative_to(C.ROOT)), features=str((D.out_dir() / "features" / "g0p80.npz").relative_to(C.ROOT))),
                 parameters=dict(families=FAMILIES, pilot_configs=PILOT_CONFIGS, workers=workers), code=PR.code_hashes())
    print(json.dumps({k: v["independent"].get("balanced_accuracy") for k, v in ev["models"].items()}, indent=1))
    return summary
