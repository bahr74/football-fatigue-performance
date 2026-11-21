"""Task B: deterministic motion-ambiguity benchmark on the validated Stage 3 two-player samples (development matches only).

Population: Stage 3 ``m11-stage3-samples-v1`` tables g0p80 (primary), g0p40, g1p60 (gap length) and g0p80_shift (gap centre shifted
off t*), loaded through the Stage 3 adapter, which refuses every held-out match and verifies file hashes. No learned parameter is
fitted: Kalman noise levels are chosen by the label-free one-step likelihood of the observed tracklets. Labels are used only to
score. Evaluation metadata (distance band, status, catalogue context) is joined after prediction and is never an input.

Saved Stage 4 out-of-fold predictions (GRU, LSTM, SVM, HGB; seed 0) are read only, to compare learned models with the physics
consensus tiers. They are historical results; nothing is retrained or re-run.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import numpy as np
import pandas as pd

from football_fatigue.m11_benchmark import metrics as MT
from football_fatigue.window_intersections import samples as S3

from . import baselines as B
from . import common as C

OUT = "m11-phase2-ambiguity-benchmark-v1"
TABLES = ("g0p80", "g0p40", "g1p60", "g0p80_shift")
PRIMARY = "g0p80"
SAVED_MODELS = ("gru_gaplast", "lstm_gaplast", "svm", "hgb")
N_BOOT = 2000
EVAL_METHODS = ("nearest", "const_pos", "cv_fwd", "cv_bidir", "ca_fwd", "ca_bidir", "ca_fwd_raw", "kf_fwd", "kf_bidir", "ocm_dir")


def _rss_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1e6
    except Exception:                                   # pragma: no cover
        return float("nan")


def p_from_margin(m: np.ndarray) -> np.ndarray:
    """Hard decision as p(SWAPPED) in {0, 0.5, 1}: ties (margin exactly 0) get 0.5 and count as half-correct in BA."""
    return np.where(m < 0, 1.0, np.where(m > 0, 0.0, 0.5))


def score(y: np.ndarray, m: np.ndarray, groups: np.ndarray, boot: bool) -> dict[str, Any]:
    p = p_from_margin(m)
    pred = np.where(m < 0, 1, 0)
    ties = int((m == 0).sum() + (~np.isfinite(m)).sum())
    r = dict(n=int(len(y)), swapped=int(y.sum()), balanced_accuracy=MT._r(MT.bal_acc(y, p)), accuracy=MT._r(float(np.mean(pred == y))),
             ties_or_undefined=ties, confusion=dict(tn=int(((pred == 0) & (y == 0)).sum()), fp=int(((pred == 1) & (y == 0)).sum()),
                                                    fn=int(((pred == 0) & (y == 1)).sum()), tp=int(((pred == 1) & (y == 1)).sum())))
    if boot:
        rng = np.random.default_rng(0)
        ms = np.unique(groups)
        idx = [np.flatnonzero(groups == g) for g in ms]
        ba = [MT.bal_acc(y[s], p[s]) for s in (np.concatenate([idx[k] for k in rng.integers(0, len(ms), len(ms))]) for _ in range(N_BOOT))]
        r["balanced_accuracy_ci95"] = [MT._r(np.quantile(ba, 0.025)), MT._r(np.quantile(ba, 0.975))]
        per = np.array([MT.bal_acc(y[groups == g], p[groups == g]) for g in ms])
        r["per_match_ba"] = dict(min=MT._r(per.min()), p25=MT._r(np.quantile(per, 0.25)), median=MT._r(np.median(per)), max=MT._r(per.max()))
    return r


def catalogue_context(meta: pd.DataFrame, ms: list[str]) -> pd.DataFrame:
    """Per sample: the 0.5 m catalogue edge whose t* equals the candidate t* (same match and period), its edge-chain id, status,
    context, group size, quality flags and the cv_margin indicator. Evaluation-only metadata."""
    cols = ["match_id", "period", "t_star", "edge_chain_id", "status_composition", "cv_margin_m", "cv_tier", "n_concurrent_other",
            "n_sequential_other", "group_d0p00", "label_risk", "reliability", "estimated_comotion", "boundary_uncertain", "slot_a", "slot_b"]
    E = C.load_table("edges", ms, cols + ["radius_m"])
    E = E[E.radius_m == 0.5].drop(columns="radius_m")
    G = C.load_table("groups", ms, ["match_id", "period", "radius_m", "link_s", "group_id", "max_simultaneous_players"])
    G = G[(G.radius_m == 0.5) & (G.link_s == 0.0)].drop(columns=["radius_m", "link_s"]).rename(columns={"group_id": "group_d0p00"})
    E = E.merge(G, on=["match_id", "period", "group_d0p00"], how="left")
    E["tkey"] = np.round(E.t_star, 6)
    E = E.drop_duplicates(["match_id", "period", "tkey"])
    clones = pd.read_parquet(C.PROC / "m11-phase2-quality-audit-v1" / "clone_like_pair_periods.parquet")
    ck = set(zip(clones.match_id, clones.period, clones.slot_a, clones.slot_b))
    E["clone_pair"] = [k in ck for k in zip(E.match_id, E.period, E.slot_a, E.slot_b)]
    M = meta[["match_id", "period", "t_star"]].copy()
    M["match_id"] = M.match_id.astype(int)
    M["tkey"] = np.round(M.t_star, 6)
    X = M.merge(E.drop(columns="t_star"), on=["match_id", "period", "tkey"], how="left")
    if len(X) != len(meta):
        raise RuntimeError("context join changed the row count")
    X["context"] = np.where(X.n_concurrent_other > 0, "simultaneous", np.where(X.n_sequential_other > 0, "sequential", "isolated"))
    X.loc[X.edge_chain_id.isna(), "context"] = "unmatched"
    return X


def strata_table(y, decisions: dict[str, np.ndarray], keys: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    out = {}
    for kname, kv in keys.items():
        d = {}
        for lev in sorted(pd.unique(kv[mask]), key=str):
            s = mask & (kv == lev)
            if s.sum() < 30:
                continue
            d[str(lev)] = dict(n=int(s.sum()), **{m: MT._r(MT.bal_acc(y[s], p_from_margin(v[s]))) for m, v in decisions.items()})
        out[kname] = d
    return out


def main(matches: list[str] | None = None) -> dict[str, Any]:
    t_all = time.perf_counter()
    ms = C.checked(matches)                                     # boundary check before any sample is read
    out = C.Out(OUT)
    R: dict[str, Any] = dict(matches=ms, tables={}, runtime={})
    loaded = {}
    for tb in TABLES:
        x, y, g = S3.load_model_inputs(C.STAGE3, tb, ms)          # refuses held-out matches, verifies hashes; inputs only
        loaded[tb] = (x, y.astype(int), g)
    # label-free Kalman noise selection on the primary table's observed tracklets
    I, O, _ = B.tracks(loaded[PRIMARY][0])
    t0 = time.perf_counter()
    sel = B.kf_select(I, O)
    R["kf_selection"] = sel | dict(seconds=round(time.perf_counter() - t0, 1), rule="min mean one-step predictive NLL per observation; no labels")
    sa, sm = sel["sigma_a"], sel["sigma_m"]
    from football_fatigue.m11_benchmark import data as D
    meta = D.load_meta(PRIMARY, ms)                              # evaluation metadata only (refuses held-out)
    if not (meta.y.to_numpy() == loaded[PRIMARY][1]).all():
        raise RuntimeError("metadata not aligned with inputs")
    ctx = catalogue_context(meta, ms)
    margins_all: dict[str, dict[str, np.ndarray]] = {}
    for tb in TABLES:
        x, y, g = loaded[tb]
        I, O, _ = B.tracks(x)
        rt = {}
        r0 = _rss_mb()
        t = time.perf_counter()
        mc = B.motion_costs(I, O)
        rt["motion_s"] = time.perf_counter() - t
        t = time.perf_counter()
        kc = B.kf_costs(I, O, sa, sm)
        rt["kalman_s"] = time.perf_counter() - t
        t = time.perf_counter()
        oc = B.ocm_costs(I, O)
        rt["ocm_s"] = time.perf_counter() - t
        rt["rss_mb_after"] = _rss_mb()
        rt["rss_mb_delta"] = rt["rss_mb_after"] - r0
        rt["us_per_sample_total"] = 1e6 * (rt["motion_s"] + rt["kalman_s"] + rt["ocm_s"]) / len(y)
        costs = {**{k: v for k, v in mc.items() if not k.startswith("_")}, "kf_fwd": kc["kf_fwd"], "kf_bidir": kc["kf_bidir"], "ocm_dir": oc}
        mg = {k: B.margin(v) for k, v in costs.items()}
        margins_all[tb] = mg
        gate = B.gated_decision(kc["kf_fwd_d2"], kc["kf_fwd"])
        meta_t = D.load_meta(tb, ms)
        indep = meta_t.independent.to_numpy(bool)
        res = dict(class_balance=dict(n=int(len(y)), swapped=int(y.sum()), matched=int((y == 0).sum()), independent=int(indep.sum()),
                                      matches=int(len(np.unique(g)))),
                   methods={k: dict(independent=score(y[indep], v[indep], g[indep], boot=(tb == PRIMARY)), all=score(y, v, g, boot=False))
                            for k, v in mg.items() if k in EVAL_METHODS},
                   ca_fallback_samples=int(mc["_ca_fallback"].sum()),
                   kf_gate=dict(full_assignment_feasible=int((gate >= 0).sum()), abstain=int((gate < 0).sum()),
                                links_gated_out=int((kc["kf_fwd_d2"] > B.GATE_CHI2).sum()),
                                accuracy_when_decided=MT._r(float(np.mean(gate[gate >= 0] == y[gate >= 0]))) if (gate >= 0).any() else None,
                                independent_coverage=MT._r(float((gate[indep] >= 0).mean()))),
                   runtime={k: round(v, 3) for k, v in rt.items()})
        R["tables"][tb] = res
        if tb == PRIMARY:
            prim = dict(y=y, g=g, indep=indep, mg=mg, gate=gate, meta=meta_t, kc=kc)
    # ---------------------------------------------------------------- primary-table analyses
    y, g, indep, mg = prim["y"], prim["g"], prim["indep"], prim["mg"]
    meta = prim["meta"]
    correct = {k: (np.where(v < 0, 1, 0) == y) & (v != 0) for k, v in mg.items()}
    core = np.stack([correct[k] for k in B.CORE], 1)
    ncorr = core.sum(1)
    tier = np.where(ncorr == len(B.CORE), "consensus_correct", np.where(ncorr == 0, "consensus_wrong", "contested"))
    R["consensus"] = dict(core_methods=list(B.CORE),
                          all=pd.Series(tier).value_counts().to_dict(), independent=pd.Series(tier[indep]).value_counts().to_dict(),
                          n_core_correct_independent=pd.Series(ncorr[indep]).value_counts().sort_index().to_dict())
    keys = dict(d_band=meta.d_band.to_numpy(), speed_band=meta.speed_band.to_numpy(), status_in_gap=meta.status_in_gap.to_numpy(),
                crossing_class=meta.crossing_class.to_numpy(), team_relation=meta.team_relation.to_numpy(),
                catalogue_status=ctx.status_composition.fillna("unmatched").to_numpy(), context=ctx.context.to_numpy(),
                group_size=ctx.max_simultaneous_players.fillna(0).clip(upper=4).astype(int).astype(str).to_numpy(),
                cv_tier=ctx.cv_tier.fillna("unmatched").to_numpy(), label_risk=ctx.label_risk.fillna(False).astype(bool).to_numpy(),
                reliability=ctx.reliability.fillna("unmatched").to_numpy(), clone_pair=ctx.clone_pair.fillna(False).astype(bool).to_numpy(),
                estimated_comotion=ctx.estimated_comotion.fillna(False).astype(bool).to_numpy(),
                boundary_uncertain=ctx.boundary_uncertain.fillna(False).astype(bool).to_numpy(),
                strict_pass=meta.strict_pass.to_numpy(), near_coincidence=meta.near_coincidence.to_numpy())
    show = {k: mg[k] for k in ("nearest", "const_pos", "cv_fwd", "cv_bidir", "ca_bidir", "kf_fwd", "kf_bidir", "ocm_dir")}
    R["strata_independent"] = strata_table(y, show, keys, indep)
    R["context_join"] = dict(matched=int(ctx.edge_chain_id.notna().sum()), unmatched=int(ctx.edge_chain_id.isna().sum()))
    # tiers by stratum (independent)
    tiers = {}
    for kname in ("d_band", "catalogue_status", "context", "group_size", "cv_tier", "clone_pair", "label_risk", "speed_band"):
        kv = keys[kname]
        tiers[kname] = {str(lev): pd.Series(tier[indep & (kv == lev)]).value_counts().to_dict() | dict(n=int((indep & (kv == lev)).sum()))
                        for lev in sorted(pd.unique(kv[indep]), key=str)}
    R["consensus_by_stratum_independent"] = tiers
    # gap length (same encounters, different hidden interval): accuracy per table for the core methods
    R["by_gap_table_independent_ba"] = {tb: {k: R["tables"][tb]["methods"][k]["independent"]["balanced_accuracy"] for k in EVAL_METHODS} for tb in TABLES}
    # catalogue cv_margin indicator vs Stage 3 baselines (not ground truth)
    cvt = keys["cv_tier"]
    R["cv_indicator_vs_baselines"] = {lev: dict(n=int((indep & (cvt == lev)).sum()),
                                                **{k: MT._r(float(correct[k][indep & (cvt == lev)].mean())) for k in ("cv_fwd", "kf_fwd", "kf_bidir")},
                                                consensus_wrong=int(((tier == "consensus_wrong") & indep & (cvt == lev)).sum()))
                                      for lev in sorted(pd.unique(cvt[indep]))}
    # margin magnitude: near-ties of the bidirectional Kalman cost
    am = np.abs(mg["kf_bidir"])
    R["kf_bidir_margin_quantiles"] = C.q(am[indep], (0.1, 0.25, 0.5, 0.75, 0.9))
    R["kf_bidir_accuracy_by_margin_decile_independent"] = [
        dict(lo=MT._r(float(lo)), hi=MT._r(float(hi)), n=int(s.sum()), accuracy=MT._r(float(correct["kf_bidir"][s].mean())))
        for lo, hi in zip(np.quantile(am[indep], np.linspace(0, 0.9, 10)), np.quantile(am[indep], np.linspace(0.1, 1.0, 10)))
        for s in [indep & (am >= lo) & (am <= hi)]]
    # saved learned-model OOF predictions (read only) within physics tiers
    learned = {}
    for mdl in SAVED_MODELS:
        pth = C.STAGE4 / "preds" / PRIMARY / mdl / "seed0.npz"
        with np.load(pth) as z:
            p = z["p"]
        if len(p) != len(y):
            raise RuntimeError(f"{mdl}: saved predictions not aligned")
        ok = ((p > 0.5).astype(int) == y)
        learned[mdl] = dict(source=str(pth.relative_to(C.ROOT)), sha256=C.sha256_file(pth),
                            independent_ba=MT._r(MT.bal_acc(y[indep], p[indep])),
                            accuracy_by_tier_independent={t: MT._r(float(ok[indep & (tier == t)].mean())) for t in ("consensus_correct", "contested", "consensus_wrong")},
                            accuracy_by_cv_tier_independent={lev: MT._r(float(ok[indep & (cvt == lev)].mean())) for lev in sorted(pd.unique(cvt[indep]))})
    R["saved_oof_models_seed0"] = learned
    # severe failures: consensus-wrong with large bidirectional CV margin against the truth
    sgn = np.where(y == 0, 1.0, -1.0)                       # positive = margin points to the truth
    wrongness = -sgn * mg["cv_bidir"]
    sev = np.flatnonzero(indep & (tier == "consensus_wrong"))
    sev = sev[np.argsort(-wrongness[sev])][:25]
    cols = ["match_id", "period", "candidate_id", "t_star", "d_min", "d_band", "speed_band", "status_in_gap", "crossing_class", "team_relation", "y"]
    SV = meta.iloc[sev][cols].reset_index(drop=True)
    SV["cv_bidir_margin_against_truth_m"] = wrongness[sev]
    SV["kf_bidir_margin"] = mg["kf_bidir"][sev]
    for k in ("catalogue_status", "context", "group_size", "cv_tier", "clone_pair", "label_risk"):
        SV[k] = keys[k][sev]
    R["severe_failures_top25"] = SV.to_dict("records")
    # per-sample table (evaluation artefact: margins, tiers, context; no tracking data)
    PS = meta[["match_id", "period", "candidate_id", "t_star", "d_min", "y", "independent", "d_band", "speed_band", "status_in_gap"]].copy()
    for k in EVAL_METHODS:
        PS[f"margin_{k}"] = mg[k]
    PS["kf_gate_decision"] = prim["gate"]
    PS["physics_tier"] = tier
    PS["n_core_correct"] = ncorr
    for k in ("catalogue_status", "context", "group_size", "cv_tier", "clone_pair", "label_risk"):
        PS[k] = keys[k]
    PS["cv_margin_m"] = ctx.cv_margin_m.to_numpy()
    R["runtime"]["total_s"] = round(time.perf_counter() - t_all, 1)
    out.json("ambiguity_results.json", R)
    out.parquet("per_sample_g0p80.parquet", PS)
    out.parquet("severe_failures_top25.parquet", SV)
    out.manifest(task="Phase 2 Task B deterministic ambiguity benchmark",
                 inputs=dict(stage3=str(C.STAGE3.relative_to(C.ROOT)), stage3_manifest_sha256=C.sha256_file(C.STAGE3 / "manifest.json"),
                             split_sha256=S3.split_sha256(C.PROC), tables=list(TABLES)) | C.catalogue_inputs(ms),
                 parameters=dict(fit_cv_s=B.FIT_CV_S, fit_ca_s=B.FIT_CA_S, a_max=B.A_MAX, v_max=B.V_MAX, gate_chi2=B.GATE_CHI2,
                                 ocm_delta_s=B.OCM_DELTA_S, sigma_a=sa, sigma_m=sm, n_boot=N_BOOT),
                 code=C.code_sha256(["ambiguity.py", "baselines.py", "common.py"]), command="python -m football_fatigue.m11_phase2 ambiguity")
    return R


if __name__ == "__main__":
    r = main()
    print(json.dumps(r["consensus"], indent=1))
