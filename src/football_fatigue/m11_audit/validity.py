"""Scientific-validity audit of the historical Stage 3 / Stage 4 benchmark (``python -m football_fatigue.m11_audit validity``).

Reads only saved development outputs (Stage 3 samples and metadata through the guarded loaders, Stage 4 OOF predictions, Phase 2
per-sample physics, catalogue nodes for player ids). Nothing historical is modified; results go to ``m11-scientific-audit-v1``.

Checks (each returns plain numbers; interpretation is in the report):

* ``fold_integrity``      OOF fold of every sample = frozen outer fold of its match; every sample predicted once; neural early-stop
                          matches never in the evaluated fold; tabular selection is inner-CV (code inspection, recorded).
* ``label_structure``     y = in_order XOR out_order (hash of candidate id); per-match class balance.
* ``strata``              independent BA of saved models by provider status inside the gap, by ESTIMATED share of the *context*
                          (model-visible) samples, clone-like pairs and label risk; paired GRU - HGB with a match bootstrap.
* ``shortcut_probes``     anti-nearest rule; structural probe (valid-point counts, frame mode) with match-grouped CV; label-free
                          texture-similarity rule and a texture-only HGB probe (fine-scale roughness, no positions).
* ``duplicates``          identical model inputs inside the dataset.
* ``player_overlap``      share of evaluated samples whose players also appear in that fold's training matches.
* ``selection``           how many variants were compared on the same OOF predictions; seed spread -> winner's-curse scale.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from football_fatigue.m11_benchmark import data as D
from football_fatigue.m11_phase2 import common as C

OUT = "m11-scientific-audit-v1"
TABLE = "g0p80"
PREDS = D.out_dir() / "preds" / TABLE
MODELS = ("gru_gaplast", "lstm_gaplast", "hgb", "svm")


# ---------------------------------------------------------------- metrics

def ba(y: np.ndarray, p: np.ndarray) -> float:
    y, d = np.asarray(y).astype(int), np.asarray(p) >= 0.5
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        return float("nan")
    return float(0.5 * (d[y == 1].mean() + (~d[y == 0]).mean()))


def match_boot(y, p, g, n_boot: int = 1000, seed: int = 0, q=None) -> dict[str, Any]:
    """BA with a match-level bootstrap CI; with ``q`` the paired difference BA(p) - BA(q)."""
    y, g = np.asarray(y).astype(int), np.asarray(g).astype(str)
    dp = np.asarray(p) >= 0.5
    dq = None if q is None else np.asarray(q) >= 0.5
    ms = np.unique(g)
    # per-match counts: tp, pos, tn, neg (for p and q)
    def counts(d):
        return np.array([[(d[(g == m) & (y == 1)]).sum(), ((g == m) & (y == 1)).sum(), (~d[(g == m) & (y == 0)]).sum(), ((g == m) & (y == 0)).sum()]
                         for m in ms], float)
    cp = counts(dp)
    cq = counts(dq) if dq is not None else None

    def bal(c):
        s = c.sum(0)
        return 0.5 * (s[0] / max(s[1], 1) + s[2] / max(s[3], 1))
    est = bal(cp) - (bal(cq) if cq is not None else 0)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        k = rng.integers(0, len(ms), len(ms))
        bs.append(bal(cp[k]) - (bal(cq[k]) if cq is not None else 0))
    return dict(value=round(float(est), 4), ci95=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
                n=int(len(y)), n_matches=int(len(ms)))


# ---------------------------------------------------------------- loading

def load(table: str = TABLE) -> dict[str, Any]:
    meta = D.load_meta(table)                                   # guarded: refuses held-out
    C.checked(sorted({str(m) for m in meta["match_id"]}))
    preds = {}
    for m in MODELS:
        z = np.load(PREDS / m / "seed0.npz")
        if len(z["p"]) != len(meta):
            raise ValueError(f"{m}: prediction rows {len(z['p'])} != metadata rows {len(meta)}")
        preds[m] = dict(p=z["p"], fold=z["fold"])
    ph = pd.read_parquet(C.PROC / "m11-phase2-ambiguity-benchmark-v1" / f"per_sample_{table}.parquet")
    if not (ph["candidate_id"].to_numpy() == meta["candidate_id"].to_numpy()).all():
        ph = ph.set_index("candidate_id").loc[meta["candidate_id"]].reset_index()
    return dict(meta=meta, preds=preds, phase2=ph)


# ---------------------------------------------------------------- checks

def fold_integrity(L: dict[str, Any]) -> dict[str, Any]:
    folds = json.loads((D.out_dir() / "folds.json").read_text(encoding="utf-8"))["outer"]
    want = L["meta"]["match_id"].astype(str).map(folds).to_numpy()
    out: dict[str, Any] = {"oof_fold_equals_frozen_outer_fold": {}, "early_stop_outside_evaluated_fold": {}}
    for m, v in L["preds"].items():
        out["oof_fold_equals_frozen_outer_fold"][m] = bool((v["fold"] == want).all())
    for d in sorted(p for p in PREDS.iterdir() if p.is_dir()):
        j = d / "seed0.json"
        if not j.exists():
            continue
        info = json.loads(j.read_text(encoding="utf-8")).get("folds", {})
        bad = [f for f, r in info.items() if isinstance(r, dict) and "early_stop_matches" in r
               and any(folds[str(x)] == int(f) for x in r["early_stop_matches"])]
        if any(isinstance(r, dict) and "early_stop_matches" in r for r in info.values()):
            out["early_stop_outside_evaluated_fold"][d.name] = not bad
    out["every_sample_predicted_once"] = bool(np.isfinite(L["preds"]["gru_gaplast"]["p"]).all())
    out["n_matches"] = int(L["meta"]["match_id"].nunique())
    out["outer_folds_partition_32"] = sorted(folds) == sorted(C.allowed()) and len(set(folds.values())) == 8
    out["tabular_selection"] = "inner CV on training folds only (m11_benchmark/run.py:_inner_oof); SVM Platt from inner OOF"
    out["neural_calibration"] = "none (raw sigmoid); early stopping on 3 training matches (folds.early_stop_matches)"
    out["passed"] = (all(out["oof_fold_equals_frozen_outer_fold"].values()) and all(out["early_stop_outside_evaluated_fold"].values())
                     and out["every_sample_predicted_once"] and out["outer_folds_partition_32"])
    return out


def label_structure(L: dict[str, Any]) -> dict[str, Any]:
    m = L["meta"]
    xor = ((m["in_order"] ^ m["out_order"]) == m["y"]).all()
    share = m.groupby("match_id")["y"].mean()
    return dict(y_equals_order_xor=bool(xor), swapped_share=round(float(m["y"].mean()), 4),
                per_match_swapped_share_min_max=[round(float(share.min()), 4), round(float(share.max()), 4)],
                note="labels are a hash of the candidate id; no input carries the order bits")


def _context_est(m: pd.DataFrame) -> pd.Series:
    e = m[["est_frac_used_in0", "est_frac_used_in1"]].mean(1)
    return pd.cut(e, [-1e-9, 0.0, 0.5, 1.0], labels=["context_all_visible", "context_est_le_50pct", "context_est_gt_50pct"]).astype(str)


def strata(L: dict[str, Any]) -> dict[str, Any]:
    m, ph = L["meta"], L["phase2"]
    ind = m["independent"].to_numpy()
    keys = dict(status_in_gap=m["status_in_gap"].astype(str), context_estimated=_context_est(m), clone_pair=ph["clone_pair"].astype(str),
                label_risk=ph["label_risk"].astype(str), crossing_class=m["crossing_class"].astype(str), d_band=m["d_band"].astype(str))
    y, g = m["y"].to_numpy(), m["match_id"].to_numpy()
    kf = 1 / (1 + np.exp(ph["margin_kf_fwd"].to_numpy()))          # positive margin favours MATCHED -> P(SWAPPED) < 0.5
    out = {}
    for name, s in keys.items():
        s = s.to_numpy()
        rows = {}
        for lv in sorted(set(s[ind])):
            k = ind & (s == lv)
            if k.sum() < 200:
                rows[lv] = dict(n=int(k.sum()), note="fewer than 200 independent samples")
                continue
            r = dict(n=int(k.sum()), share=round(float(k.sum() / ind.sum()), 4))
            for mm in MODELS:
                r[mm] = round(ba(y[k], L["preds"][mm]["p"][k]), 4)
            r["kalman_fwd"] = round(ba(y[k], kf[k]), 4)
            r["gru_minus_hgb"] = match_boot(y[k], L["preds"]["gru_gaplast"]["p"][k], g[k], q=L["preds"]["hgb"]["p"][k])
            rows[lv] = r
        out[name] = rows
    out["overall"] = {mm: match_boot(y[ind], L["preds"][mm]["p"][ind], g[ind]) for mm in MODELS}
    out["overall"]["gru_minus_hgb"] = match_boot(y[ind], L["preds"]["gru_gaplast"]["p"][ind], g[ind], q=L["preds"]["hgb"]["p"][ind])
    return out


def roughness(pos: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fine-scale texture of each tracklet (no positions, no direction): median |second difference| (m), share of zero
    displacements (held positions), and std of step length (m). pos (N, 2, T, 2), mask (N, 2, T) -> (N, 2, 3)."""
    n = pos.shape[0]
    out = np.zeros((n, 2, 3))
    for j in range(2):
        p, k = pos[:, j], mask[:, j]
        d1 = np.diff(p, axis=1)
        ok1 = k[:, 1:] & k[:, :-1]
        step = np.hypot(d1[..., 0], d1[..., 1])
        d2 = np.diff(d1, axis=1)
        ok2 = ok1[:, 1:] & ok1[:, :-1]
        a2 = np.hypot(d2[..., 0], d2[..., 1])
        out[:, j, 0] = np.nanmedian(np.where(ok2, a2, np.nan), 1)
        out[:, j, 1] = np.nanmean(np.where(ok1, step < 1e-4, np.nan), 1)
        out[:, j, 2] = np.nanstd(np.where(ok1, step, np.nan), 1)
    return np.nan_to_num(out)


def shortcut_probes(L: dict[str, Any], x: dict[str, np.ndarray]) -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    m, ph = L["meta"], L["phase2"]
    y, g, ind = m["y"].to_numpy(), m["match_id"].to_numpy(), m["independent"].to_numpy()
    fold = L["preds"]["hgb"]["fold"]
    status = m["status_in_gap"].astype(str).to_numpy()
    res: dict[str, Any] = {}
    # 1. anti-nearest: always choose the pairing the nearest-endpoint rule rejects
    anti = (ph["margin_nearest"].to_numpy() > 0).astype(float)          # nearest favours MATCHED -> predict SWAPPED
    res["anti_nearest_rule"] = match_boot(y[ind], anti[ind], g[ind])

    def grouped_cv(F, model):
        p = np.zeros(len(y))
        for f in np.unique(fold):
            tr, te = fold != f, fold == f
            p[te] = model().fit(F[tr], y[tr]).predict_proba(F[te])[:, 1]
        return p
    # 2. structural probe: valid-point counts and frame mode only
    S = m[["n_valid_in0", "n_valid_in1", "n_valid_out0", "n_valid_out1", "frame_mode"]].to_numpy(float)
    ps = grouped_cv(S, lambda: HistGradientBoostingClassifier(max_iter=200, random_state=0))
    res["structural_probe_hgb"] = match_boot(y[ind], ps[ind], g[ind])
    # 3. texture: label-free similarity rule and texture-only HGB probe (antisymmetric pair features, no positions)
    ri, ro = roughness(x["in_pos"], x["in_mask"]), roughness(x["out_pos"], x["out_mask"])
    straight = np.abs(np.log1p(1e3 * ri[:, 0]) - np.log1p(1e3 * ro[:, 0])) + np.abs(np.log1p(1e3 * ri[:, 1]) - np.log1p(1e3 * ro[:, 1]))
    crossed = np.abs(np.log1p(1e3 * ri[:, 0]) - np.log1p(1e3 * ro[:, 1])) + np.abs(np.log1p(1e3 * ri[:, 1]) - np.log1p(1e3 * ro[:, 0]))
    rule = (crossed.sum(1) < straight.sum(1)).astype(float)              # crossed pairing more similar -> SWAPPED
    T = np.c_[straight - crossed, np.abs(ri).reshape(len(y), -1).mean(1, keepdims=True)]
    pt = grouped_cv(T, lambda: HistGradientBoostingClassifier(max_iter=300, random_state=0))
    for nm, p in (("texture_similarity_rule", rule), ("texture_only_hgb_probe", pt)):
        res[nm] = dict(all=match_boot(y[ind], p[ind], g[ind]),
                       **{s: match_boot(y[ind & (status == s)], p[ind & (status == s)], g[ind & (status == s)]) for s in sorted(set(status))})
    res["texture_feature_medians_by_status"] = {s: dict(zip(["d2_median_m", "held_share", "step_std_m"], np.round(np.median(ri[status == s].reshape(-1, 3), 0), 4)))
                                                for s in sorted(set(status))}
    res["logistic_structural_check"] = round(ba(y[ind], grouped_cv(S, lambda: LogisticRegression(max_iter=1000))[ind]), 4)
    return res


def duplicates(x: dict[str, np.ndarray], meta: pd.DataFrame) -> dict[str, Any]:
    h = [hashlib.sha1(np.round(x["in_pos"][i], 3).tobytes() + np.round(x["out_pos"][i], 3).tobytes()).hexdigest() for i in range(len(meta))]
    s = pd.Series(h)
    dup = s.duplicated(keep=False)
    return dict(n=int(len(s)), exact_duplicate_inputs=int(dup.sum()), duplicate_groups=int(s[dup].nunique()),
                cross_match_duplicates=int(meta.loc[dup.to_numpy()].groupby(s[dup].to_numpy())["match_id"].nunique().gt(1).sum()) if dup.any() else 0)


def player_overlap(L: dict[str, Any]) -> dict[str, Any]:
    m = L["meta"]
    cand = []
    nodes = []
    for mid in sorted(m["match_id"].astype(str).unique(), key=int):
        c = pd.read_parquet(C.PROC / "m11-stage2-candidates-v1" / "matches" / mid / "candidates.parquet", columns=["candidate_id", "slot_a", "slot_b"])
        cand.append(c.assign(match_id=int(mid)))
        n = C.load_table("nodes", [mid])[["slot", "player_id"]].drop_duplicates("slot")
        nodes.append(n.assign(match_id=int(mid)))
    cand, nodes = pd.concat(cand), pd.concat(nodes)
    pid = nodes.set_index(["match_id", "slot"])["player_id"]
    j = m[["match_id", "candidate_id"]].merge(cand, on=["match_id", "candidate_id"], how="left")
    pa = pid.reindex(list(zip(j["match_id"], j["slot_a"]))).to_numpy()
    pb = pid.reindex(list(zip(j["match_id"], j["slot_b"]))).to_numpy()
    fold = L["preds"]["hgb"]["fold"]
    seen_any = np.zeros(len(m), bool)
    seen_both = np.zeros(len(m), bool)
    for f in np.unique(fold):
        trp = set(pa[fold != f]) | set(pb[fold != f])
        te = fold == f
        a, b = np.isin(pa[te], list(trp)), np.isin(pb[te], list(trp))
        seen_any[te], seen_both[te] = a | b, a & b
    ind = m["independent"].to_numpy()
    y, g = m["y"].to_numpy(), m["match_id"].to_numpy()
    r = dict(samples_with_player_ids=int(np.isfinite(pa.astype(float)).sum()), share_any_player_seen_in_training=round(float(seen_any.mean()), 4),
             share_both_players_seen_in_training=round(float(seen_both.mean()), 4))
    for nm, k in (("both_seen", ind & seen_both), ("not_both_seen", ind & ~seen_both)):
        r[nm] = {mm: match_boot(y[k], L["preds"][mm]["p"][k], g[k]) for mm in ("gru_gaplast", "hgb")}
    return r


def selection(L: dict[str, Any]) -> dict[str, Any]:
    variants = sorted(p.name for p in PREDS.iterdir() if p.is_dir())
    m = L["meta"]
    ind, y = m["independent"].to_numpy(), m["y"].to_numpy()
    seeds = []
    for s in (0, 1, 2):
        f = PREDS / "gru_gaplast" / f"seed{s}.npz"
        if f.exists():
            seeds.append(ba(y[ind], np.load(f)["p"][ind]))
    sd = float(np.std(seeds, ddof=1)) if len(seeds) > 1 else float("nan")
    k = len([v for v in variants if not v.startswith(("phys_", "B"))])
    from scipy.stats import norm
    emax = float(np.mean(np.sort(norm.rvs(size=(20000, k), random_state=0), 1)[:, -1]))
    return dict(variants_with_saved_oof=len(variants), learned_variants_compared=k, gru_gaplast_seed_ba=[round(v, 4) for v in seeds],
                seed_sd=round(sd, 4), winners_curse_scale_ba=round(sd * emax, 4),
                note="upper scale of selection optimism if the learned variants were exchangeable with the seed spread; the architecture "
                     "itself (gap-last) was designed after earlier OOF results, which this number cannot capture")


def run(table: str = TABLE) -> dict[str, Any]:
    L = load(table)
    x, y, _g = D.load_inputs(table)
    if not (np.asarray(y) == L["meta"]["y"].to_numpy()).all():
        raise ValueError("inputs and metadata are not row-aligned")
    return dict(schema=OUT, table=table, fold_integrity=fold_integrity(L), label_structure=label_structure(L), strata=strata(L),
                shortcut_probes=shortcut_probes(L, x), duplicates=duplicates(x, L["meta"]), player_overlap=player_overlap(L), selection=selection(L))


def main() -> int:
    import time
    t0 = time.perf_counter()
    out = C.Out(OUT)
    r = run()
    r["runtime_s"] = round(time.perf_counter() - t0, 1)
    out.json("audit.json", r)
    here = Path(__file__).resolve().parent
    out.manifest(inputs=dict(stage3=str(D.stage3_dir().relative_to(C.ROOT)), stage4_preds=str(PREDS.relative_to(C.ROOT)),
                             phase2="data/processed/m11-phase2-ambiguity-benchmark-v1/per_sample_g0p80.parquet"),
                 parameters=dict(table=TABLE, models=MODELS, n_boot=1000), code={p.name: C.sha256_file(p) for p in sorted(here.glob("*.py"))})
    print(json.dumps({k: r[k] for k in ("fold_integrity", "label_structure", "duplicates", "selection")}, indent=1, default=str))
    return 0
