"""Stage 4 evaluation of out-of-fold predictions (train/CV matches only): per-seed headline metrics (mean / sd), seed-0 details
(match bootstrap, per fold, per match, abstention, strata), train vs validation, paired comparisons. Writes
``metrics/<table>/`` (a new directory per evaluation version, never overwritten)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import data as D
from . import metrics as MT

SUPERSEDED = re.compile(r"^B\d_")           # first physics version (clipped calibration weight), kept on disk, not evaluated
PHYS_PREFIX = "phys_"
TABULAR = ("lr", "knn", "svm", "rf", "hgb")
NEURAL = ("mlp", "gru", "lstm", "gru_gaplast", "lstm_gaplast")
HEADLINE = ("balanced_accuracy", "log_loss", "accuracy", "roc_auc", "brier", "ece", "f1_swapped", "precision_swapped", "recall_swapped")


def angle_map(matches: list[str]) -> dict[str, float]:
    """Crossing angle of the Stage 2 linked v1 event (descriptor ``angle_deg``) per candidate; offline metadata only."""
    out = {}
    root = D.proc_root() / "m11-stage2-candidates-v1" / "matches"
    for m in matches:
        c = pd.read_parquet(root / m / "candidates.parquet", columns=["candidate_id", "linked_event_id"])
        d = pd.read_parquet(root / m / "descriptors_linear.parquet", columns=["event_id", "angle_deg"]).set_index("event_id")["angle_deg"]
        out.update({cid: float(d.get(e, np.nan)) if isinstance(e, str) else np.nan for cid, e in zip(c["candidate_id"], c["linked_event_id"])})
    return out


def strata_keys(meta: pd.DataFrame, angles: dict[str, float]) -> dict[str, np.ndarray]:
    grp = meta.groupby(["match_id", "overlap_group"])["row"].transform("size").to_numpy()
    ang = meta["candidate_id"].map(angles).to_numpy(float)
    ab = np.where(np.isnan(ang), "no_linked_event", pd.cut(np.nan_to_num(ang), [-1, 30, 60, 120, 150, 181],
                                                             labels=["<30", "30-60", "60-120", "120-150", ">=150"]).astype(str))
    est = np.nanmean(meta[["est_frac_used_in0", "est_frac_used_in1"]].to_numpy(float), 1)
    return dict(crossing_class=meta["crossing_class"].to_numpy(), d_band=meta["d_band"].to_numpy(), speed_band=meta["speed_band"].to_numpy(),
                angle_band=ab, team_relation=meta["team_relation"].to_numpy(), near_coincidence=meta["near_coincidence"].to_numpy(),
                jump_transfer_risk=meta["jump_transfer_risk"].to_numpy(), status_in_gap=meta["status_in_gap"].to_numpy(),
                estimated_share_used=np.where(est < 0.5, "visible_heavy(<0.5)", "estimated_heavy(>=0.5)"),
                overlap_group_size=np.where(grp == 1, "1", np.where(grp <= 3, "2-3", ">=4")), frame_mode=meta["frame_mode"].to_numpy(),
                cohort_F2_primary=meta["cohort_F2_primary"].to_numpy(), cohort_F3_primary=meta["cohort_F3_primary"].to_numpy())


def strata_with_ci(y, p, g, mask, keys, n_boot=500):
    out = MT.strata(y, p, None, mask, keys)
    for name, col in keys.items():
        for v in out[name]:
            m = mask & (col.astype(str) == v)
            if m.sum() >= 30 and len(np.unique(g[m])) >= 5:
                out[name][v]["balanced_accuracy_ci"] = MT.bootstrap(y[m], p[m], g[m], n_boot=n_boot)["balanced_accuracy_ci"]
    return out


def discover(table: str) -> dict[str, list[int]]:
    root = D.out_dir() / "preds" / table
    res = {}
    for d in sorted(root.iterdir()) if root.exists() else []:
        if SUPERSEDED.match(d.name):
            continue
        seeds = sorted(int(p.stem[4:]) for p in d.glob("seed*.npz"))
        if seeds:
            res[d.name] = seeds
    return res


def family(name: str) -> str:
    base = name.split("__")[0]
    return "physics" if base.startswith(PHYS_PREFIX) else "tabular" if base in TABULAR else "neural" if base in NEURAL else "other"


def train_val(info: dict[str, Any]) -> dict[str, Any]:
    folds = info.get("folds", {})
    tr = [f["train"]["log_loss"] for f in folds.values() if "train" in f]
    trb = [f["train"]["balanced_accuracy"] for f in folds.values() if "train" in f]
    va = [f["val"]["log_loss"] for f in folds.values() if "val" in f]
    vab = [f["val"]["balanced_accuracy"] for f in folds.values() if "val" in f]
    r = dict(train_log_loss_mean=MT._r(np.mean(tr)) if tr else None, train_balanced_accuracy_mean=MT._r(np.mean(trb)) if trb else None,
             val_log_loss_mean=MT._r(np.mean(va)) if va else None, val_balanced_accuracy_mean=MT._r(np.mean(vab)) if vab else None)
    es = [f["early_stop"]["log_loss"] for f in folds.values() if "early_stop" in f]
    if es:
        r["early_stop_log_loss_mean"] = MT._r(np.mean(es))
        r["best_epoch"] = [f["best_epoch"] for f in folds.values()]
        r["params"] = next(iter(folds.values()))["params"]
    sel = [json.dumps(f.get("selected"), sort_keys=True) for f in folds.values() if "selected" in f]
    if sel:
        r["selected_configs"] = {s: sel.count(s) for s in sorted(set(sel))}
    return r


def evaluate_table(table: str, version: str = "v1", models: list[str] | None = None, boot: bool = True) -> dict[str, Any]:
    out = D.out_dir() / "metrics" / table / version
    if (out / "comparison.json").exists():
        raise SystemExit(f"{out} exists; evaluation versions are never overwritten")
    x, y, g = D.load_inputs(table)
    meta = D.load_meta(table)
    y = y.astype(int)
    indep = meta["independent"].to_numpy(bool)
    found = discover(table)
    if models:
        found = {m: s for m, s in found.items() if m in models}
    angles = angle_map(sorted({str(int(v)) for v in g}, key=int))
    keys = strata_keys(meta, angles)
    P: dict[str, dict[int, np.ndarray]] = {}
    res: dict[str, Any] = {}
    for mdl, seeds in found.items():
        d = D.out_dir() / "preds" / table / mdl
        P[mdl] = {}
        per_seed = {}
        for s in seeds:
            z = D.load_npz(d / f"seed{s}.npz")
            p, fold = z["p"], z["fold"]
            if not np.isfinite(p).all():
                raise RuntimeError(f"{mdl} seed {s}: non-finite or missing predictions")
            P[mdl][s] = p
            per_seed[s] = dict(independent=MT.basic(y[indep], p[indep]), all=MT.basic(y, p))
        s0 = seeds[0]
        info = json.loads((d / f"seed{s0}.json").read_text(encoding="utf-8"))
        agg = {}
        for pop in ("independent", "all"):
            agg[pop] = {k: dict(mean=MT._r(np.mean([per_seed[s][pop][k] for s in seeds])),
                                sd=MT._r(np.std([per_seed[s][pop][k] for s in seeds], ddof=1)) if len(seeds) > 1 else None) for k in HEADLINE}
        det = MT.summarize(y, P[mdl][s0], g, fold, indep, boot=boot)
        res[mdl] = dict(family=family(mdl), seeds=seeds, seed_coverage="complete (3 seeds)" if len(seeds) >= 3 else f"{len(seeds)} seed(s)",
                        aggregate=agg, per_seed={str(s): v for s, v in per_seed.items()}, seed0_detail=det, train_vs_val=train_val(info),
                        strata_independent=strata_with_ci(y, P[mdl][s0], g, indep, keys) if boot else MT.strata(y, P[mdl][s0], meta, indep, keys))
    # paired comparisons (seed 0, independent subset)
    def best(fam):
        c = [m for m in res if res[m]["family"] == fam]
        return min(c, key=lambda m: res[m]["aggregate"]["independent"]["log_loss"]["mean"]) if c else None
    refs = {"best_physics": best("physics"), "best_tabular": best("tabular"), "best_neural": best("neural")}
    for mdl in res:
        res[mdl]["paired"] = {}
        for rn, ref in refs.items():
            if ref and ref != mdl:
                res[mdl]["paired"][f"vs_{rn}:{ref}"] = MT.paired(y[indep], P[mdl][res[mdl]["seeds"][0]][indep], P[ref][res[ref]["seeds"][0]][indep],
                                                                 g[indep], n_boot=2000 if boot else 200)
    out.mkdir(parents=True, exist_ok=True)
    for mdl, r in res.items():
        (out / f"{mdl}.json").write_text(_dumps(r), encoding="utf-8")
    comp = dict(table=table, n=int(len(y)), n_independent=int(indep.sum()), references=refs,
                models={m: dict(family=r["family"], seeds=r["seeds"], **{f"indep_{k}": r["aggregate"]["independent"][k] for k in
                                                                         ("balanced_accuracy", "log_loss", "roc_auc", "ece")},
                                all_balanced_accuracy=r["aggregate"]["all"]["balanced_accuracy"], all_log_loss=r["aggregate"]["all"]["log_loss"],
                                ci_seed0=r["seed0_detail"]["independent"].get("bootstrap"), train_vs_val=r["train_vs_val"],
                                fold_ba_mean_sd=[r["seed0_detail"]["independent"]["per_fold"]["balanced_accuracy_mean"],
                                                 r["seed0_detail"]["independent"]["per_fold"]["balanced_accuracy_sd"]])
                        for m, r in res.items()})
    (out / "comparison.json").write_text(_dumps(comp), encoding="utf-8")
    (out / "comparison.md").write_text(markdown(comp, res), encoding="utf-8")
    return comp


def _dumps(o):
    from .run import dumps
    return dumps(o)


def markdown(comp, res) -> str:
    rows = ["| Model | Family | Seeds | Indep. BA (mean ± sd) | 95% CI (seed 0) | Indep. log loss | AUC | ECE | All BA | Train LL / Val LL |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    order = sorted(res, key=lambda m: res[m]["aggregate"]["independent"]["log_loss"]["mean"])
    for m in order:
        r = comp["models"][m]
        ba, ll = r["indep_balanced_accuracy"], r["indep_log_loss"]
        ci = (r["ci_seed0"] or {}).get("balanced_accuracy_ci")
        sd = lambda v: f" ± {v['sd']:.4f}" if v.get("sd") is not None else ""  # noqa: E731
        tv = r["train_vs_val"]
        rows.append(f"| {m} | {r['family']} | {len(r['seeds'])} | {ba['mean']:.4f}{sd(ba)} | {ci} | {ll['mean']:.4f}{sd(ll)} | "
                    f"{r['indep_roc_auc']['mean']} | {r['indep_ece']['mean']} | {r['all_balanced_accuracy']['mean']} | "
                    f"{tv.get('train_log_loss_mean')} / {tv.get('val_log_loss_mean')} |")
    return "\n".join(rows) + "\n"


def main(table: str, version: str = "v1", models: list[str] | None = None) -> None:
    from .run import log
    comp = evaluate_table(table, version, models)
    log(f"evaluate {table}: {len(comp['models'])} models; references {comp['references']}")
