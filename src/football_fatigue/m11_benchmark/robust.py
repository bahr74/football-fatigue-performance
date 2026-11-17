"""Stage 4 robustness analyses on train/CV matches only (``python -m football_fatigue.m11_benchmark robust --analyses ...``).

Analyses (``robust/<name>/``, written once; units resumable): ``learning_curve``, ``gap_transfer``, ``shift``, ``texture``,
``dependency``, ``abstention``, ``errors``. Fold models are reused from the main runs where the protocol allows (same outer folds:
a fold model only ever predicts matches of its own outer validation fold)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from . import data as D
from . import folds as FO
from . import metrics as MT
from . import models as MD
from .run import PRIMARY, Ctx, log, pdir, run_neural, run_tabular, write_json_once

RB = lambda: D.out_dir() / "robust"  # noqa: E731
SEL_FILE = "selected_models.json"


def selected() -> dict[str, str]:
    """Best tabular and best sequence model on the primary table by independent-subset log loss (seed-averaged), from the frozen
    v-latest comparison; written once so every analysis uses the same choice."""
    p = RB() / SEL_FILE
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    vers = sorted((D.out_dir() / "metrics" / PRIMARY).iterdir())
    comp = json.loads((vers[-1] / "comparison.json").read_text(encoding="utf-8"))
    ms = comp["models"]
    pick = lambda fam: min((m for m, r in ms.items() if r["family"] == fam and "__" not in m), key=lambda m: ms[m]["indep_log_loss"]["mean"])  # noqa: E731
    sel = dict(tabular=pick("tabular"), neural=pick("neural"), physics=pick("physics"), source=str(vers[-1].name))
    write_json_once(p, sel)
    return sel


# ---------------------------------------------------------------- fold-model reuse

def tabular_fold_model(table: str, model: str, seed: int, f: int):
    return joblib.load(pdir(table, model) / "folds" / f"seed{seed}_fold{f}.joblib")


def neural_fold_model(table: str, model: str, seed: int, f: int):
    import torch
    m = MD.Neural(model, seed=seed)
    dev = m._dev()
    m.net = MD.make_net(model, **MD.NEURAL[model]).to(dev)
    m.net.load_state_dict(torch.load(pdir(table, model) / "folds" / f"seed{seed}_fold{f}.pt", map_location=dev))
    return m


def predict_with_fold_models(sel_model: str, fam: str, ctx: Ctx, F=None, si=None, so=None, train_table: str = PRIMARY, seed: int = 0) -> np.ndarray:
    """Out-of-fold predictions on ``ctx`` (any table) from the fold models trained on ``train_table``."""
    p = np.full(len(ctx.y), np.nan)
    for f in range(FO.N_OUTER):
        va = ctx.fold == f
        if fam == "tabular":
            p[va] = tabular_fold_model(train_table, sel_model, seed, f).predict((ctx.F if F is None else F)[va])
        else:
            m = neural_fold_model(train_table, sel_model, seed, f)
            p[va] = m.predict((ctx.si if si is None else si)[va], (ctx.so if so is None else so)[va], ctx.gap[va].astype(np.float32))
    return p


def score(ctx: Ctx, p: np.ndarray, mask: np.ndarray | None = None, boot: bool = True) -> dict[str, Any]:
    m = ctx.indep if mask is None else mask
    r = MT.basic(ctx.y[m], p[m])
    out = {k: r[k] for k in ("n", "balanced_accuracy", "log_loss", "roc_auc", "ece")}
    if boot:
        out["ci"] = MT.bootstrap(ctx.y[m], p[m], ctx.g[m], n_boot=1000)
    return out


# ---------------------------------------------------------------- A. learning curves

LC_SIZES = (4, 8, 16, 24)


def learning_curve(sel: dict[str, str]) -> dict[str, Any]:
    out_p = RB() / "learning_curve" / "learning_curve.json"
    if out_p.exists():
        return json.loads(out_p.read_text(encoding="utf-8"))
    res: dict[str, Any] = {}
    for fam in ("tabular", "neural"):
        mdl = sel[fam]
        ctx = Ctx(PRIMARY, need_seq=fam == "neural")
        res[mdl] = {}
        for n in LC_SIZES:
            unit = RB() / "learning_curve" / f"{mdl}_n{n}.npz"
            if unit.exists():
                p = D.load_npz(unit)["p"]
                tr_ll = json.loads(unit.with_suffix(".json").read_text(encoding="utf-8"))["train_log_loss_mean"]
            else:
                t0 = time.perf_counter()
                p, tls = np.full(len(ctx.y), np.nan), []
                for f in range(FO.N_OUTER):
                    va = ctx.fold == f
                    sub = FO._rank(ctx.train_matches(f), f"{FO.FOLD_SALT}|lc|{f}")[:n]
                    inm = np.isin(ctx.g.astype(str), sub)
                    if fam == "tabular":
                        cfg = json.loads((pdir(PRIMARY, mdl) / "folds" / f"select_fold{f}.json").read_text(encoding="utf-8"))["selected"]
                        m = MD.Tabular(mdl, cfg, ctx.k, 0).fit(ctx.F[inm], ctx.y[inm])
                        if mdl == "svm":
                            m.platt = MD.fit_platt(m.raw(ctx.F[inm]), ctx.y[inm])
                        p[va] = m.predict(ctx.F[va])
                        tls.append(float(MT.ll_vec(ctx.y[inm], m.predict(ctx.F[inm])).mean()))
                    else:
                        es = sub[:max(1, n // 8)]
                        ges = np.isin(ctx.g.astype(str), es)
                        tr, ev = inm & ~ges, inm & ges
                        G = ctx.gap.astype(np.float32)
                        m = MD.Neural(mdl, seed=0).fit(ctx.si[tr], ctx.so[tr], G[tr], ctx.y[tr], ctx.si[ev], ctx.so[ev], G[ev], ctx.y[ev])
                        p[va] = m.predict(ctx.si[va], ctx.so[va], G[va])
                        tls.append(float(MT.ll_vec(ctx.y[tr], m.predict(ctx.si[tr], ctx.so[tr], G[tr])).mean()))
                tr_ll = float(np.mean(tls))
                D.write_once_npz(unit, dict(p=p))
                write_json_once(unit.with_suffix(".json"), dict(model=mdl, n_train_matches=n, train_log_loss_mean=tr_ll,
                                                                runtime_s=round(time.perf_counter() - t0, 1)))
            res[mdl][str(n)] = dict(n_train_matches=n, train_log_loss=round(tr_ll, 4), **score(ctx, p))
            log(f"learning curve {mdl} n={n}: BA {res[mdl][str(n)]['balanced_accuracy']} LL {res[mdl][str(n)]['log_loss']}")
        z = D.load_npz(pdir(PRIMARY, mdl) / "seed0.npz")["p"]
        info = json.loads((pdir(PRIMARY, mdl) / "seed0.json").read_text(encoding="utf-8"))
        trl = np.mean([v["train"]["log_loss"] for v in info["folds"].values()])
        res[mdl]["28"] = dict(n_train_matches=28, train_log_loss=round(float(trl), 4), source="main run (seed 0)", **score(ctx, z))
    write_json_once(out_p, res)
    return res


# ---------------------------------------------------------------- B. gap-length transfer, C. shift

def gap_transfer(sel: dict[str, str]) -> dict[str, Any]:
    out_p = RB() / "gap_transfer" / "gap_transfer.json"
    if out_p.exists():
        return json.loads(out_p.read_text(encoding="utf-8"))
    res: dict[str, Any] = dict(protocol=(
        "transfer: fold models trained on the 0.8 s table (main runs, seed 0) predict the same outer validation matches of the 0.4 s / "
        "1.6 s tables; retrained: the same model family trained and cross-validated on each gap table with the same folds (seed 0). "
        "Comparisons use candidates present in both tables (paired by candidate id); gap tables share encounters and are never pooled."))
    base = Ctx(PRIMARY, need_seq=True)
    for table in ("g0p40", "g1p60"):
        ctx = Ctx(table, need_seq=True)
        for fam in ("tabular", "neural"):
            mdl = sel[fam]
            pt = predict_with_fold_models(mdl, fam, ctx)
            if fam == "tabular":
                run_tabular(ctx, (mdl,))
            else:
                run_neural(ctx, (mdl,), (0,))
            pr = D.load_npz(pdir(table, mdl) / "seed0.npz")["p"]
            p08 = D.load_npz(pdir(PRIMARY, mdl) / "seed0.npz")["p"]
            ids08 = {c: i for i, c in enumerate(base.meta["candidate_id"])}
            common = np.array([c in ids08 for c in ctx.meta["candidate_id"]]) & ctx.indep
            j = np.array([ids08.get(c, -1) for c in ctx.meta["candidate_id"]])
            res[f"{table}|{mdl}"] = dict(
                transfer_from_0p8=score(ctx, pt), retrained_on_gap=score(ctx, pr),
                paired_common_candidates=dict(n=int(common.sum()), transfer=score(ctx, pt, common, boot=False),
                                              retrained=score(ctx, pr, common, boot=False),
                                              primary_0p8_same_candidates={k: v for k, v in MT.basic(base.y[j[common]], p08[j[common]]).items()
                                                                           if k in ("n", "balanced_accuracy", "log_loss")},
                                              retrained_minus_transfer=MT.paired(ctx.y[common], pr[common], pt[common], ctx.g[common], 1000)))
            log(f"gap transfer {table} {mdl}: transfer BA {res[f'{table}|{mdl}']['transfer_from_0p8']['balanced_accuracy']} "
                f"retrained BA {res[f'{table}|{mdl}']['retrained_on_gap']['balanced_accuracy']}")
    write_json_once(out_p, res)
    return res


def shift(sel: dict[str, str]) -> dict[str, Any]:
    out_p = RB() / "shift" / "shift.json"
    if out_p.exists():
        return json.loads(out_p.read_text(encoding="utf-8"))
    ctx = Ctx("g0p80_shift", need_seq=True)
    base = Ctx(PRIMARY)
    ids = {c: i for i, c in enumerate(base.meta["candidate_id"])}
    j = np.array([ids[c] for c in ctx.meta["candidate_id"]])
    res: dict[str, Any] = dict(protocol="fold models trained on centred 0.8 s gaps predict the shifted-centre table (same outer folds); "
                                        "compared with the centred predictions of the same candidates")
    for fam in ("tabular", "neural", "physics"):
        mdl = sel[fam]
        if fam == "physics":
            continue
        ps = predict_with_fold_models(mdl, fam, ctx)
        pc = D.load_npz(pdir(PRIMARY, mdl) / "seed0.npz")["p"][j]
        yc = base.y[j]
        r = dict(shifted=score(ctx, ps), centred_same_candidates={k: v for k, v in MT.basic(yc[ctx.indep], pc[ctx.indep]).items()
                                                                    if k in ("n", "balanced_accuracy", "log_loss", "roc_auc")}, by_shift={})
        for s in sorted(ctx.meta["shift_s"].unique()):
            m = ctx.indep & (ctx.meta["shift_s"].to_numpy() == s)
            r["by_shift"][f"{s:+.1f}"] = dict(n=int(m.sum()), shifted_ba=MT._r(MT.bal_acc(ctx.y[m], ps[m])), shifted_ll=MT._r(MT.ll_vec(ctx.y[m], ps[m]).mean()),
                                              centred_ba=MT._r(MT.bal_acc(yc[m], pc[m])), centred_ll=MT._r(MT.ll_vec(yc[m], pc[m]).mean()))
        res[mdl] = r
        log(f"shift {mdl}: shifted BA {r['shifted']['balanced_accuracy']} vs centred {r['centred_same_candidates']['balanced_accuracy']}")
    write_json_once(out_p, res)
    return res


# ---------------------------------------------------------------- D. tracking-status texture ablation

def smooth(pos: np.ndarray, mask: np.ndarray, w: int = 5) -> np.ndarray:
    """Mask-aware centred moving average over w points along time (masked points never contribute and stay 0)."""
    k = np.ones(w)
    m = mask.astype(float)
    num = np.stack([np.apply_along_axis(lambda v: np.convolve(v, k, "same"), -1, pos[..., c] * m) for c in (0, 1)], -1)
    den = np.apply_along_axis(lambda v: np.convolve(v, k, "same"), -1, m)[..., None]
    return np.where(mask[..., None], num / np.maximum(den, 1e-9), 0.0).astype(np.float32)


def perturb(pos: np.ndarray, mask: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.where(mask[..., None], pos + rng.normal(0, sigma, pos.shape), 0.0).astype(np.float32)


def texture(sel: dict[str, str]) -> dict[str, Any]:
    out_p = RB() / "texture" / "texture.json"
    if out_p.exists():
        return json.loads(out_p.read_text(encoding="utf-8"))
    ctx = Ctx(PRIMARY, need_seq=True)
    x, _, _ = D.load_inputs(PRIMARY)
    variants = {"original": x,
                "smooth5": dict(x, in_pos=smooth(x["in_pos"], x["in_mask"]), out_pos=smooth(x["out_pos"], x["out_mask"])),
                "noise5cm": dict(x, in_pos=perturb(x["in_pos"], x["in_mask"], 0.05, 1), out_pos=perturb(x["out_pos"], x["out_mask"], 0.05, 2))}
    status = ctx.meta["status_in_gap"].to_numpy()
    res: dict[str, Any] = dict(protocol=("inference-time ablation: the fold models (trained on original inputs) predict transformed inputs "
                                         "of their own validation matches; masks, labels and groups unchanged; transforms applied to all "
                                         "incoming and outgoing tracklets identically; plus the best tabular model retrained on each "
                                         "transform (train and test transformed)."), results={})
    for vn, xv in variants.items():
        unit = RB() / "texture" / f"features_{vn}.npz"
        if vn == "original":
            F = ctx.F
        elif unit.exists():
            F = D.load_npz(unit)["F"]
        else:
            F, _, _ = D.features(xv)
            D.write_once_npz(unit, dict(F=F))
        si, so = D.seq_tensor(xv)
        for fam in ("tabular", "neural"):
            mdl = sel[fam]
            p = predict_with_fold_models(mdl, fam, ctx, F=F, si=si, so=so)
            r = dict(all_status=score(ctx, p), by_status={s: score(ctx, p, ctx.indep & (status == s), boot=False) for s in np.unique(status)})
            res["results"][f"{vn}|{mdl}|inference"] = r
            log(f"texture {vn} {mdl}: BA {r['all_status']['balanced_accuracy']} " + ", ".join(f"{s} {v['balanced_accuracy']}" for s, v in r["by_status"].items()))
        if vn != "original":
            mdl = sel["tabular"]
            unit2 = RB() / "texture" / f"retrained_{vn}_{mdl}.npz"
            if unit2.exists():
                p = D.load_npz(unit2)["p"]
            else:
                p = np.full(len(ctx.y), np.nan)
                for f in range(FO.N_OUTER):
                    tr, va = ctx.fold != f, ctx.fold == f
                    cfg = json.loads((pdir(PRIMARY, mdl) / "folds" / f"select_fold{f}.json").read_text(encoding="utf-8"))["selected"]
                    m = MD.Tabular(mdl, cfg, ctx.k, 0).fit(F[tr], ctx.y[tr])
                    if mdl == "svm":
                        m.platt = MD.fit_platt(m.raw(F[tr]), ctx.y[tr])
                    p[va] = m.predict(F[va])
                D.write_once_npz(unit2, dict(p=p))
            res["results"][f"{vn}|{mdl}|retrained"] = dict(all_status=score(ctx, p), by_status={s: score(ctx, p, ctx.indep & (status == s), boot=False)
                                                                                               for s in np.unique(status)})
    # status-wise spread of the original per-step jitter (second differences), to quantify the texture itself
    acc = np.abs(np.diff(x["in_pos"], 2, axis=2)).sum(-1)
    mk = x["in_mask"][:, :, 2:] & x["in_mask"][:, :, 1:-1] & x["in_mask"][:, :, :-2]
    jit = np.array([np.median(a[m]) if m.any() else np.nan for a, m in zip(acc.reshape(len(acc), -1), mk.reshape(len(mk), -1))])
    res["median_second_difference_m_by_status"] = {s: MT._r(np.nanmedian(jit[status == s]), 5) for s in np.unique(status)}
    write_json_once(out_p, res)
    return res


def texture_nn(sel: dict[str, str]) -> dict[str, Any]:
    """The selected sequence model retrained (seed 0, same folds and protocol) on transformed inputs, train and test transformed;
    separates sensitivity to an input distribution shift from dependence on fine tracking texture."""
    out_p = RB() / "texture" / "texture_nn_retrained.json"
    if out_p.exists():
        return json.loads(out_p.read_text(encoding="utf-8"))
    ctx = Ctx(PRIMARY, need_seq=True)
    x = ctx.x
    status = ctx.meta["status_in_gap"].to_numpy()
    mdl = sel["neural"]
    variants = {"smooth5": dict(x, in_pos=smooth(x["in_pos"], x["in_mask"]), out_pos=smooth(x["out_pos"], x["out_mask"])),
                "noise5cm": dict(x, in_pos=perturb(x["in_pos"], x["in_mask"], 0.05, 1), out_pos=perturb(x["out_pos"], x["out_mask"], 0.05, 2))}
    res: dict[str, Any] = dict(protocol="selected sequence model retrained on each transform (train, early-stop and validation inputs "
                                        "transformed identically; seed 0; same outer folds); compare with texture.json inference rows")
    for vn, xv in variants.items():
        tag = f"__tex_{vn}"
        run_neural(ctx, (mdl,), (0,), tag=tag, transform=D.seq_tensor(xv))
        p = D.load_npz(pdir(PRIMARY, mdl + tag) / "seed0.npz")["p"]
        res[f"{vn}|{mdl}|retrained"] = dict(all_status=score(ctx, p), by_status={s: score(ctx, p, ctx.indep & (status == s), boot=False)
                                                                                 for s in np.unique(status)})
        log(f"texture_nn {vn} {mdl} retrained: BA {res[f'{vn}|{mdl}|retrained']['all_status']['balanced_accuracy']}")
    write_json_once(out_p, res)
    return res


# ---------------------------------------------------------------- E. dependency weighting

def dependency(sel: dict[str, str]) -> dict[str, Any]:
    out_p = RB() / "dependency" / "dependency.json"
    if out_p.exists():
        return json.loads(out_p.read_text(encoding="utf-8"))
    ctx = Ctx(PRIMARY, need_seq=True)
    size = ctx.meta.groupby(["match_id", "overlap_group"])["row"].transform("size").to_numpy(float)
    w = 1.0 / size
    res: dict[str, Any] = dict(protocol="1 / overlap-group size sample weights in training (same folds, seed 0); evaluation on the independent subset")
    tab = sel["tabular"]
    if tab == "knn":
        res["note"] = "best tabular model is KNN (no sample weights); HGB used instead"
        tab = "hgb"
    run_tabular(ctx, (tab,), weights=w, tag="__w_overlap")
    run_neural(ctx, (sel["neural"],), (0,), weights=w, tag="__w_overlap")
    for mdl in (tab, sel["neural"]):
        pu = D.load_npz(pdir(PRIMARY, mdl) / "seed0.npz")["p"]
        pw = D.load_npz(pdir(PRIMARY, mdl + "__w_overlap") / "seed0.npz")["p"]
        res[mdl] = dict(unweighted=score(ctx, pu), weighted=score(ctx, pw),
                        weighted_minus_unweighted=MT.paired(ctx.y[ctx.indep], pw[ctx.indep], pu[ctx.indep], ctx.g[ctx.indep]))
        log(f"dependency {mdl}: weighted BA {res[mdl]['weighted']['balanced_accuracy']} vs {res[mdl]['unweighted']['balanced_accuracy']}")
    write_json_once(out_p, res)
    return res


# ---------------------------------------------------------------- F. abstention with nested thresholds

def abstention(models: list[str]) -> dict[str, Any]:
    out_p = RB() / "abstention" / "abstention.json"
    if out_p.exists():
        return json.loads(out_p.read_text(encoding="utf-8"))
    ctx = Ctx(PRIMARY)
    res: dict[str, Any] = dict(protocol=("for each outer fold, the confidence threshold reaching a target error on the OOF predictions of the "
                                         "OTHER seven folds (independent subset) is applied to the fold; realised error and coverage are "
                                         "pooled. Development OOF only; no held-out data."))
    for mdl in models:
        p = D.load_npz(pdir(PRIMARY, mdl) / "seed0.npz")["p"]
        conf = np.maximum(p, 1 - p)
        wrong = ((p > 0.5).astype(int) != ctx.y)
        r = {}
        for tgt in (0.01, 0.02, 0.05, 0.10):
            com = np.zeros(len(p), bool)
            ths = []
            for f in range(FO.N_OUTER):
                o = ctx.indep & (ctx.fold != f)
                order = np.argsort(-conf[o], kind="stable")
                cum = np.cumsum(wrong[o][order]) / np.arange(1, o.sum() + 1)
                ok = np.flatnonzero(cum <= tgt)
                th = conf[o][order][ok[-1]] if len(ok) else 1.0 + 1e-9
                ths.append(float(th))
                com |= (ctx.fold == f) & (conf >= th)
            m = ctx.indep
            r[f"target_{int(tgt * 100):02d}"] = dict(thresholds=[round(t, 4) for t in ths], coverage=MT._r(com[m].mean()),
                                                     realised_error=MT._r(wrong[m & com].mean()) if (m & com).any() else None,
                                                     recovered_share=MT._r((~wrong & com)[m].mean()))
        r["descriptive_curve"] = MT.coverage_curve(ctx.y[ctx.indep], p[ctx.indep])
        res[mdl] = r
        log(f"abstention {mdl}: 5% target -> coverage {r['target_05']['coverage']} error {r['target_05']['realised_error']}")
    write_json_once(out_p, res)
    return res


# ---------------------------------------------------------------- error analysis

def errors(models: list[str], n_fig: int = 12) -> dict[str, Any]:
    out_p = RB() / "errors" / "errors.json"
    if out_p.exists():
        return json.loads(out_p.read_text(encoding="utf-8"))
    ctx = Ctx(PRIMARY)
    x, _, _ = D.load_inputs(PRIMARY)
    names = ctx.names
    res: dict[str, Any] = {}
    feat = {n: ctx.F[:, names.index(n)] for n in ("speed_before_min", "speed_before_max", "angle_before", "angle_after", "sep_before_end",
                                                  "sep_after_start", "cv_bidir_abs_margin")}
    nvalid = ctx.meta[["n_valid_in0", "n_valid_in1", "n_valid_out0", "n_valid_out1"]].min(1).to_numpy()
    for mdl in models:
        p = D.load_npz(pdir(PRIMARY, mdl) / "seed0.npz")["p"]
        conf = np.maximum(p, 1 - p)
        wrong = (p > 0.5).astype(int) != ctx.y
        hc = ctx.indep & wrong & (conf >= 0.9)
        ref = ctx.indep
        def prof(m):
            mt = ctx.meta[m]
            return dict(n=int(m.sum()), near_coincidence=MT._r(mt["near_coincidence"].mean()), jump_transfer_risk=MT._r(mt["jump_transfer_risk"].mean()),
                        crossing={k: MT._r(v) for k, v in mt["crossing_class"].value_counts(normalize=True).items()},
                        status={k: MT._r(v) for k, v in mt["status_in_gap"].value_counts(normalize=True).items()},
                        d_band={k: MT._r(v) for k, v in mt["d_band"].value_counts(normalize=True).items()},
                        speed_band={k: MT._r(v) for k, v in mt["speed_band"].value_counts(normalize=True).items()},
                        team_same=MT._r((mt["team_relation"] == "same_team").mean()),
                        min_valid_points_lt_30=MT._r((nvalid[m] < 30).mean()),
                        **{f"median_{k}": MT._r(np.median(v[m])) for k, v in feat.items()})
        res[mdl] = dict(high_confidence_wrong=prof(hc), all_independent=prof(ref), correct_high_confidence=prof(ctx.indep & ~wrong & (conf >= 0.9)),
                        n_wrong_independent=int((ctx.indep & wrong).sum()), share_wrong_with_conf_ge_0p9=MT._r(hc.sum() / max(1, (ctx.indep & wrong).sum())),
                        examples=[dict(candidate_id=ctx.meta["candidate_id"].iloc[i], match_id=int(ctx.meta["match_id"].iloc[i]), y=int(ctx.y[i]),
                                       p_swapped=MT._r(p[i]), d_min=MT._r(ctx.meta["d_min"].iloc[i]), crossing=ctx.meta["crossing_class"].iloc[i],
                                       status=ctx.meta["status_in_gap"].iloc[i]) for i in np.flatnonzero(hc)[np.argsort(-conf[hc])][:30]])
        if mdl == models[0]:
            figure(x, ctx, p, np.flatnonzero(hc)[np.argsort(-conf[hc])][:n_fig], RB() / "errors" / f"high_confidence_errors_{mdl}.png", mdl)
    write_json_once(out_p, res)
    return res


def figure(x, ctx, p, idx, path: Path, mdl: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(idx)
    if not n:
        return
    cols = 4
    fig, axes = plt.subplots((n + cols - 1) // cols, cols, figsize=(4 * cols, 3.6 * ((n + cols - 1) // cols)), squeeze=False)
    for ax, i in zip(axes.ravel(), idx):
        y = int(ctx.y[i])
        for j, c in ((0, "#1f77b4"), (1, "#d62728")):
            mi, mo = x["in_mask"][i, j], x["out_mask"][i, j]
            ax.plot(*x["in_pos"][i, j][mi].T, "-", color=c, lw=2, label=f"in{j}")
            jo = j if y == 0 else 1 - j                                     # true continuation of incoming j (reference)
            ax.plot(*x["out_pos"][i, jo][mo if jo == j else x["out_mask"][i, jo]].T, "--", color=c, lw=2, label=f"true out of in{j}")
        ax.set_title(f"{ctx.meta['match_id'].iloc[i]} {ctx.meta['candidate_id'].iloc[i][:6]}  y={'SWAPPED' if y else 'MATCHED'}\n"
                     f"p(SWAPPED)={p[i]:.2f}  d_min={ctx.meta['d_min'].iloc[i]:.2f} m  {ctx.meta['status_in_gap'].iloc[i]}", fontsize=8)
        ax.set_aspect("equal", "datalim")
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=6)
    fig.suptitle(f"Highest-confidence errors of {mdl} (sample frame, metres; solid incoming, dashed reference continuation)", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(analyses: list[str] | None) -> None:
    sel = selected()
    log(f"robust: selected models {sel}")
    todo = analyses or ["abstention", "errors", "texture", "shift", "gap_transfer", "dependency", "learning_curve"]
    for a in todo:
        t = time.perf_counter()
        if a == "abstention":
            abstention([sel["tabular"], sel["neural"], sel["physics"]])
        elif a == "errors":
            errors([sel["tabular"], sel["neural"]])
        elif a == "texture":
            texture(sel)
        elif a == "texture_nn":
            texture_nn(sel)
        elif a == "shift":
            shift(sel)
        elif a == "gap_transfer":
            gap_transfer(sel)
        elif a == "dependency":
            dependency(sel)
        elif a == "learning_curve":
            learning_curve(sel)
        log(f"robust {a} finished in {time.perf_counter() - t:.0f} s")
