"""Stage 4C audit, HGB runs and evaluation (train/CV matches only). See :mod:`.stage4c` for the predeclared design."""

from __future__ import annotations

import json
import re
import time
from typing import Any

import joblib
import numpy as np

from football_fatigue.window_intersections import samples as S

from . import data as D
from . import folds as FO
from . import metrics as MT
from . import models as MD
from . import shiftaug as SA
from .run import PRIMARY, Ctx, pdir, write_json_once
from .stage4c import CAPS, log, out_dir

WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)          # predeclared convex combinations w * GRU + (1 - w) * HGB


# ---------------------------------------------------------------- 1. audit

def _tracklet_slopes(pos, mask, dt, fit_s=S.FIT_S):
    """Least-squares velocity over the last ``fit_s`` before each incoming tracklet's last valid point (frame coordinates), as in
    ``samples._slope``. (audit v1 anchored the window at the grid end, which is wrong for tracklets with masked gap-side points.)"""
    t = np.broadcast_to(dt[:, None, :], mask.shape)
    t_last = np.where(mask, t, -np.inf).max(-1)
    sel = mask & (t >= t_last[..., None] - fit_s - 1e-9)
    w = sel.astype(float)
    n = w.sum(-1)
    tm = (w * t).sum(-1) / np.maximum(n, 1)
    v = []
    for c in (0, 1):
        pm = (w * pos[..., c]).sum(-1) / np.maximum(n, 1)
        cov = (w * (t - tm[..., None]) * (pos[..., c] - pm[..., None])).sum(-1)
        var = (w * (t - tm[..., None]) ** 2).sum(-1)
        v.append(cov / np.maximum(var, 1e-12))
    return np.stack(v, -1)


def audit() -> dict[str, Any]:
    p_out = out_dir() / "audit" / "audit_v2.json"          # audit.json (v1): c4 check implementation error, kept as superseded
    if p_out.exists():
        return json.loads(p_out.read_text(encoding="utf-8"))
    import torch
    res: dict[str, Any] = {}
    ctx = Ctx(PRIMARY, need_seq=True)
    x = ctx.x
    # 1 identical frozen folds
    fj = json.loads((D.out_dir() / "folds.json").read_text(encoding="utf-8"))
    fold_ok, checked = True, 0
    comp = json.loads((D.out_dir() / "metrics" / PRIMARY / "v2" / "comparison.json").read_text(encoding="utf-8"))
    for mdl, r in comp["models"].items():
        for s in r["seeds"]:
            z = D.load_npz(pdir(PRIMARY, mdl) / f"seed{s}.npz")
            fold_ok &= bool(np.array_equal(z["fold"], ctx.fold))
            checked += 1
    outer_json = {str(m): int(k) for m, k in (fj["outer"].items() if isinstance(fj["outer"], dict) else [])}
    res["c1_identical_outer_folds"] = dict(prediction_files_checked=checked, all_fold_vectors_identical=fold_ok,
                                           folds_json_matches_salted_rank=(outer_json == ctx.fmap) if outer_json else "folds.json layout differs; see g2",
                                           n_outer=FO.N_OUTER, matches_per_fold=sorted(np.bincount(list(ctx.fmap.values())).tolist()),
                                           **{"pass": fold_ok})
    # 2 / 10 inputs contain no identity, label, status metadata
    names = ctx.names
    bad_feat = [n for n in names if any(w in n.lower() for w in S.FORBIDDEN_INPUT_WORDS)]
    res["c2_c10_inputs_free_of_metadata"] = dict(adapter_fields=sorted(x), expected=sorted(S.INPUT_FIELDS), feature_count=len(names),
                                                 forbidden_words_in_feature_names=bad_feat,
                                                 seq_channels=["x/10 m", "y/10 m", "valid mask", "dt (s)"],
                                                 status_columns_reach_models=False,
                                                 **{"pass": sorted(x) == sorted(S.INPUT_FIELDS) and not bad_feat})
    # 3 gap-last encoder: reversal, masking, padding, gap-adjacent summary
    si, so = ctx.si, ctx.so
    rev_ok = bool(np.allclose(si[:, :, 0, 3], x["in_dt"][:, -1:].repeat(2, 1) * x["in_mask"][:, :, -1])
                  and np.allclose(si[:, :, :, 0], (x["in_pos"][:, :, ::-1, 0] * x["in_mask"][:, :, ::-1]) / D.POS_SCALE_M, atol=1e-6)
                  and np.allclose(so[:, :, :, 0], x["out_pos"][..., 0] * x["out_mask"] / D.POS_SCALE_M, atol=1e-6))
    masked_zero = bool((si[..., :2][si[..., 2] == 0] == 0).all() and (so[..., :2][so[..., 2] == 0] == 0).all()
                       and (si[..., 3][si[..., 2] == 0] == 0).all())
    torch.manual_seed(0)
    net = MD.make_net("gru_gaplast", **MD.NEURAL["gru_gaplast"]).eval()
    enc = net.enc_in
    rng = np.random.default_rng(0)
    seq = torch.as_tensor(rng.normal(size=(64, 30, 4)).astype(np.float32))
    first_valid = rng.integers(0, 6, 64)                      # masked gap-adjacent steps 0..first_valid-1 (input order)
    last_valid = rng.integers(20, 30, 64)
    m = (np.arange(30)[None] >= first_valid[:, None]) & (np.arange(30)[None] <= last_valid[:, None])
    seq[..., 2] = torch.as_tensor(m.astype(np.float32))
    seq[..., [0, 1, 3]] *= seq[..., 2:3]
    with torch.no_grad():
        e0 = enc(seq)
        s2 = seq.clone()
        for i in range(64):                                    # change values at masked gap-adjacent steps (after the summary step)
            s2[i, :first_valid[i], [0, 1, 3]] = 7.0
        inv_after = float((enc(s2) - e0).abs().max())
        s3 = seq.clone()
        s3[torch.arange(64), torch.as_tensor(first_valid), 0] += 1.0     # nearest valid point
        sens_near = float((enc(s3) - e0).abs().amax(1).min())
    n_in_adj_masked = int((~x["in_mask"][:, :, -1]).sum())
    n_out_adj_masked = int((~x["out_mask"][:, :, 0]).sum())
    res["c3_gaplast_sequence_handling"] = dict(
        reversal_and_scaling_ok=rev_ok, masked_steps_all_zero=masked_zero,
        summary_invariant_to_masked_gap_adjacent_steps_max_abs=inv_after, nearest_valid_point_changes_every_embedding_min=sens_near,
        tracklets_with_masked_gap_adjacent_point=dict(incoming=n_in_adj_masked, outgoing=n_out_adj_masked, of=int(2 * len(ctx.y))),
        note=("Inputs start at the gap (incoming reversed); gap_last flips so the gap-adjacent valid point is the RNN summary step; masked "
              "steps nearer the gap than the first valid point come after the summary and cannot influence it; masked far-end steps are "
              "zero-valued with mask 0 and precede it. Padding is not used (fixed T = 30)."),
        **{"pass": rev_ok and masked_zero and inv_after == 0.0 and sens_near > 0})
    # 4 normalisation uses incoming tracklets only
    last = lambda mk: mk.shape[-1] - 1 - mk[..., ::-1].argmax(-1)  # noqa: E731
    li = last(x["in_mask"])
    n = len(ctx.y)
    p0 = x["in_pos"][np.arange(n), 0, li[:, 0]]
    p1 = x["in_pos"][np.arange(n), 1, li[:, 1]]
    origin_err = float(np.abs(p0 + p1).max())
    v = _tracklet_slopes(x["in_pos"], x["in_mask"], x["in_dt"])
    vs = v.sum(1)
    mode = ctx.meta["frame_mode"].to_numpy()
    m0 = mode == 0
    axis_err = float(np.abs(vs[m0, 1]).max())
    axis_pos = bool((vs[m0, 0] > 0).all())
    res["c4_frame_from_incoming_only"] = dict(
        origin_midpoint_of_last_incoming_points_max_err_m=origin_err, frame_mode0_summed_incoming_velocity_y_component_max=axis_err,
        frame_mode0_axis_points_along_summed_velocity=axis_pos, frame_mode_counts={int(k): int(v) for k, v in zip(*np.unique(mode, return_counts=True))},
        position_scale="fixed 10 m (not fitted)", tabular_scaling="StandardScaler inside the per-fold pipeline (fitted on the training partition only)",
        stage3_hidden_gap_perturbation_tests="see Stage 3 checks (all identical)",
        **{"pass": origin_err < 1e-3 and axis_err < 1e-3 and axis_pos})
    # 5 augmentation inside fold boundaries (augmentation set rows belong to their candidate's match / fold)
    xa, ya, ga, ma = SA.load()
    idx0 = {c: i for i, c in enumerate(ctx.meta["candidate_id"])}
    owner = np.array([idx0[c] for c in ma["candidate_id"]])
    same_fold = bool((FO.fold_index(ga, ctx.fmap) == ctx.fold[owner]).all() and (ga == ctx.g[owner]).all())
    res["c5_augmentation_within_folds"] = dict(
        shift_rows_same_match_and_fold_as_their_encounter=same_fold, n_shift_rows=int(len(ya)),
        neural_batch_augmentation="reflection / order exchanges are applied to batches drawn from training rows only",
        tabular_order_augmentation="antisymmetric copies are created inside fit(X_train) only",
        **{"pass": same_fold})
    # 6 probability complementarity under outgoing-order exchange
    Fs = D.swap_features(ctx.F, ctx.k)
    hgb_err = 0.0
    for f in range(FO.N_OUTER):
        va = ctx.fold == f
        mdl = joblib.load(pdir(PRIMARY, "hgb") / "folds" / f"seed0_fold{f}.joblib")
        hgb_err = max(hgb_err, float(np.abs(mdl.predict(Fs[va]) - (1 - mdl.predict(ctx.F[va]))).max()))
    from .robust import neural_fold_model
    gru_err = 0.0
    for f in range(FO.N_OUTER):
        va = np.flatnonzero(ctx.fold == f)
        mm = neural_fold_model(PRIMARY, "gru_gaplast", 0, f)
        G = ctx.gap.astype(np.float32)
        p = mm.predict(si[va], so[va], G[va])
        q = mm.predict(si[va], so[va][:, ::-1].copy(), G[va])
        gru_err = max(gru_err, float(np.abs(q - (1 - p)).max()))
    tr = ctx.fold != 0
    cm = MD.PhysicsCombo().fit(ctx.margins[tr], ctx.gap[tr], ctx.y[tr])
    phys_err = float(np.abs(cm.predict(-ctx.margins[~tr], ctx.gap[~tr]) - (1 - cm.predict(ctx.margins[~tr], ctx.gap[~tr]))).max())
    res["c6_complementarity"] = dict(hgb_all_folds_max_abs=hgb_err, gru_gaplast_seed0_all_folds_max_abs=gru_err,
                                     phys_B3_combo_fold0_max_abs=phys_err,
                                     note="physics margins are antisymmetric under the outgoing exchange (m -> -m)",
                                     **{"pass": hgb_err < 1e-9 and gru_err < 1e-4 and phys_err < 1e-9})
    # 7 15 Hz grid consistency
    dt_ok = bool(np.allclose(x["in_dt"], S.IN_REL, atol=1e-6) and np.allclose(x["out_dt"], S.OUT_REL, atol=1e-6)
                 and np.allclose(np.diff(S.IN_REL), 1 / 15) and np.allclose(x["gap_s"], 0.8))
    smp = D.to_sample(x, 0)
    t_ok = bool(np.isclose(smp.t_before[-1], -0.4 - 1 / 15, atol=1e-5) and np.isclose(smp.t_after[0], 0.4 + 1 / 15, atol=1e-5)) \
        if hasattr(smp, "t_before") else None
    seq_dt_ok = bool(np.allclose(so[:, :, :, 3], x["out_dt"][:, None, :] * x["out_mask"], atol=1e-6))
    res["c7_15hz_consistency"] = dict(grids_ok=dt_ok, seq_dt_channel_ok=seq_dt_ok, feature_sample_times_relative_to_centre_ok=t_ok,
                                      dt_s=1 / 15, T=30, **{"pass": dt_ok and seq_dt_ok and t_ok is not False})
    # 8 superseded / failed models excluded from rankings
    pdirs = sorted(p.name for p in (D.out_dir() / "preds" / PRIMARY).iterdir())
    ranked = sorted(comp["models"])
    res["c8_superseded_excluded"] = dict(prediction_directories=pdirs, ranked_in_v2=ranked,
                                         superseded_physics_v1_ranked=[m for m in ranked if re.match(r"^B\d_", m)],
                                         robustness_variants_ranked=[m for m in ranked if "__" in m],
                                         failed_gate_files=sorted(p.name for p in (D.out_dir() / "gates").glob("*FAILED*")),
                                         **{"pass": not [m for m in ranked if re.match(r"^B\d_", m) or "__" in m]})
    # 9 reproducibility of reported results from saved predictions
    diffs = {}
    for mdl, r in comp["models"].items():
        vals = []
        for s in r["seeds"]:
            p = D.load_npz(pdir(PRIMARY, mdl) / f"seed{s}.npz")["p"]
            vals.append(MT.basic(ctx.y[ctx.indep], p[ctx.indep]))
        diffs[mdl] = max(abs(MT._r(np.mean([v["balanced_accuracy"] for v in vals])) - r["indep_balanced_accuracy"]["mean"]),
                         abs(MT._r(np.mean([v["log_loss"] for v in vals])) - r["indep_log_loss"]["mean"]))
    res["c9_reproducible_from_saved_predictions"] = dict(max_abs_diff_by_model=diffs, retraining_reproduction="see stage4c config A cap 80 vs Stage 4B",
                                                         **{"pass": max(diffs.values()) < 1e-3})
    # reported best_epoch vs loaded state (Stage 4B logging detail)
    mism, tot = 0, 0
    for s in (0, 1, 2):
        for f in range(FO.N_OUTER):
            j = json.loads((pdir(PRIMARY, "gru_gaplast") / "folds" / f"seed{s}_fold{f}.json").read_text(encoding="utf-8"))
            h = [e["val_log_loss"] for e in j["history"]]
            b, be = np.inf, 0
            for i, v_ in enumerate(h):
                if v_ < b - 1e-4:
                    b, be = v_, i + 1
            mism += int(be != j["best_epoch"])
            tot += 1
    res["n_best_epoch_report_differs_from_loaded_state"] = dict(gru_gaplast_fold_fits=tot, differing=mism,
                                                                note="Stage 4B reports argmin(val loss); the loaded state is the last improvement by > 1e-4")
    res["texture_pairing_probe"] = texture_probe(ctx)
    res["all_pass"] = all(v.get("pass", True) for v in res.values() if isinstance(v, dict))
    write_json_once(p_out, res)
    log(f"audit: all_pass={res['all_pass']}")
    return res


def texture_probe(ctx: Ctx) -> dict[str, Any]:
    """Shortcut probe: can tracking *texture* alone (second / third differences, exact-linearity share; no positions, velocities,
    distances or headings) pair incoming with outgoing tracklets? Antisymmetric pair features, symmetric HGB, same folds."""
    x = ctx.x

    def stats(pos, mask):
        d2 = np.linalg.norm(np.diff(pos, 2, axis=2), axis=-1)
        d3 = np.linalg.norm(np.diff(pos, 3, axis=2), axis=-1)
        m2 = mask[:, :, 2:] & mask[:, :, 1:-1] & mask[:, :, :-2]
        m3 = m2[:, :, 1:] & mask[:, :, :-3]
        def med(a, m):
            a = np.where(m, a, np.nan)
            with np.errstate(all="ignore"):
                return np.nan_to_num(np.nanmedian(a, -1)), np.nan_to_num(np.nanpercentile(a, 90, axis=-1))
        a2, b2 = med(d2, m2)
        a3, _ = med(d3, m3)
        lin = np.where(m2, d2 < 1e-3, False).sum(-1) / np.maximum(m2.sum(-1), 1)
        return np.stack([a2, b2, a3, lin], -1)                 # (N, 2, 4)
    si, so = stats(x["in_pos"], x["in_mask"]), stats(x["out_pos"], x["out_mask"])
    cost = lambda i, j: np.abs(si[:, i] - so[:, j])  # noqa: E731
    X = (cost(0, 0) + cost(1, 1)) - (cost(0, 1) + cost(1, 0))   # positive favours SWAPPED; antisymmetric under outgoing exchange
    X = np.c_[X, si.mean(1), so.mean(1)]                         # symmetric context (texture level)
    k = 4
    status = ctx.meta["status_in_gap"].to_numpy()
    p = np.zeros(len(ctx.y))
    for f in range(FO.N_OUTER):
        tr, va = ctx.fold != f, ctx.fold == f
        m = MD.Tabular("hgb", MD.TABULAR_GRIDS["hgb"][2], k, 0).fit(X[tr], ctx.y[tr])
        p[va] = m.predict(X[va])
    out = dict(features=["median |d2|", "p90 |d2|", "median |d3|", "share |d2| < 1 mm"], all=MT.basic(ctx.y[ctx.indep], p[ctx.indep]),
               ci=MT.bootstrap(ctx.y[ctx.indep], p[ctx.indep], ctx.g[ctx.indep], n_boot=1000)["balanced_accuracy_ci"], by_status={})
    for s in np.unique(status):
        mm = ctx.indep & (status == s)
        out["by_status"][s] = dict(n=int(mm.sum()), balanced_accuracy=MT._r(MT.bal_acc(ctx.y[mm], p[mm])),
                                   ci=MT.bootstrap(ctx.y[mm], p[mm], ctx.g[mm], n_boot=500)["balanced_accuracy_ci"])
    est = np.nanmean(ctx.meta[["est_frac_used_in0", "est_frac_used_in1"]].to_numpy(float), 1)
    bins = [(-0.01, 0.1, "<0.1"), (0.1, 0.5, "0.1-0.5"), (0.5, 0.9, "0.5-0.9"), (0.9, 1.01, ">=0.9")]
    out["by_estimated_share_of_input_points"] = {}
    for lo, hi, nm in bins:
        mm = ctx.indep & (est > lo) & (est <= hi) if lo >= 0 else ctx.indep & (est <= hi)
        if mm.sum() >= 50:
            out["by_estimated_share_of_input_points"][nm] = dict(n=int(mm.sum()), balanced_accuracy=MT._r(MT.bal_acc(ctx.y[mm], p[mm])),
                                                                 ci=_ba_ci(ctx.y[mm], p[mm], ctx.g[mm], 500))
    D.write_once_npz(out_dir() / "audit" / "texture_probe_oof.npz", dict(p=p))
    out["texture_by_status_median_d2_m"] = {s: MT._r(float(np.median(si[status == s][..., 0])), 5) for s in np.unique(status)}
    out["texture_by_status_linear_share"] = {s: MT._r(float(np.mean(si[status == s][..., 3])), 4) for s in np.unique(status)}
    return out


# ---------------------------------------------------------------- 2. HGB under the same training-distribution configs

HGB_CONFIGS = ("A", "B", "C", "Anoise", "Asmooth")


def aug_features() -> np.ndarray:
    p = out_dir() / "features" / "shiftaug.npz"
    if p.exists():
        return D.load_npz(p)["F"]
    xa, _, _, _ = SA.load()
    t = time.perf_counter()
    F, _, _ = D.features(xa)
    D.write_once_npz(p, dict(F=F))
    log(f"features shiftaug {F.shape} in {time.perf_counter() - t:.0f} s")
    return F


class Data:
    """Centred table, augmentation rows and evaluation tables (features and metadata; no sequences)."""

    def __init__(self, need_features: bool = True):
        self.c = Ctx(PRIMARY)
        c = self.c
        idx0 = {cid: i for i, cid in enumerate(c.meta["candidate_id"])}
        _, ya, ga, ma = SA.load()
        self.ya, self.ga, self.ma = ya.astype(int), ga, ma
        self.owner = np.array([idx0[cid] for cid in ma["candidate_id"]])
        self.n_aug = np.bincount(self.owner, minlength=len(c.y))
        self.fold_a = FO.fold_index(ga, c.fmap)
        self.indep_a = c.indep[self.owner]
        self.ev: dict[str, dict[str, Any]] = {}
        for t in ("g0p80_shift", "g0p40", "g1p60"):
            _, y, g = D.load_inputs(t)
            meta = D.load_meta(t)
            self.ev[t] = dict(y=y.astype(int), g=g, fold=FO.fold_index(g, c.fmap), indep=meta["independent"].to_numpy(bool), meta=meta,
                              F=D.feature_table(t)["F"] if need_features else None)
        for vn in ("smooth5", "noise5cm"):
            self.ev[vn] = dict(y=c.y, g=c.g, fold=c.fold, indep=c.indep, meta=c.meta,
                               F=D.load_npz(D.out_dir() / "robust" / "texture" / f"features_{vn}.npz")["F"] if need_features else None)
        self.Fa = aug_features() if need_features else None


def _fit_hgb(cfg, k, X, y, w, seed=0):
    X = np.nan_to_num(X)
    m = MD.Tabular("hgb", cfg, k, seed)
    m.m = MD.make_tabular("hgb", cfg, X.shape[1], seed).fit(np.r_[X, D.swap_features(X, k)], np.r_[y, 1 - y], sample_weight=np.r_[w, w])
    return m


def _train_set(dd: Data, config: str, rows: np.ndarray):
    c = dd.c
    X, y, w = [c.F[rows]], [c.y[rows]], []
    na = dd.n_aug[rows]
    if config == "A":
        w = [np.ones(len(rows))]
    elif config in ("B", "C"):
        ar = np.flatnonzero(np.isin(dd.owner, rows))
        X.append(dd.Fa[ar])
        y.append(dd.ya[ar])
        nao = dd.n_aug[dd.owner[ar]]
        w = [1.0 / (1 + na), 1.0 / (1 + nao)] if config == "B" else [np.where(na > 0, 0.5, 1.0), 0.5 / nao]
    else:
        vn = "noise5cm" if config == "Anoise" else "smooth5"
        X.append(dd.ev[vn]["F"][rows])
        y.append(c.y[rows])
        w = [np.full(len(rows), 0.5), np.full(len(rows), 0.5)]
    return np.concatenate(X), np.concatenate(y), np.concatenate(w)


def run_hgb(configs=HGB_CONFIGS) -> None:
    dd = Data()
    c = dd.c
    for config in configs:
        d = out_dir() / "hgb" / config
        for f in range(FO.N_OUTER):
            unit = d / f"fold{f}.npz"
            if unit.exists():
                continue
            t1 = time.perf_counter()
            cfg = json.loads((pdir(PRIMARY, "hgb") / "folds" / f"select_fold{f}.json").read_text(encoding="utf-8"))["selected"]
            es = FO.early_stop_matches(c.train_matches(f), f)
            ges = np.isin(c.g.astype(str), es)
            tr, va, ev = np.flatnonzero(c.fold != f), c.fold == f, np.flatnonzero((c.fold != f) & ges)
            if config == "A":
                m = joblib.load(pdir(PRIMARY, "hgb") / "folds" / f"seed0_fold{f}.joblib")
            else:
                m = _fit_hgb(cfg, c.k, *_train_set(dd, config, tr))
            m_es = _fit_hgb(cfg, c.k, *_train_set(dd, config, np.flatnonzero((c.fold != f) & ~ges)))
            arr = dict(val=m.predict(c.F[va]), es=m_es.predict(c.F[ev]), aug=m.predict(dd.Fa[dd.fold_a == f]))
            for t, e in dd.ev.items():
                arr[t] = m.predict(e["F"][e["fold"] == f])
            ptr = m.predict(c.F[tr])
            sym = float(np.abs(m.predict(D.swap_features(c.F[va], c.k)) - (1 - arr["val"])).max())
            info = dict(fold=f, selected=cfg, early_stop_matches=sorted(es, key=int), train=MT.basic(c.y[tr], ptr), val=MT.basic(c.y[va], arr["val"]),
                        symmetry_max_err_outgoing=sym, reused_stage4_model=config == "A", fit_s=round(time.perf_counter() - t1, 1))
            d.mkdir(parents=True, exist_ok=True)
            if config != "A":
                joblib.dump(m, d / f"fold{f}.joblib", compress=3)
            D.write_once_npz(unit, arr)
            write_json_once(unit.with_suffix(".json"), info)
            log(f"  hgb {config} f{f}: val BA {info['val']['balanced_accuracy']} LL {info['val']['log_loss']} ({info['fit_s']} s)")
        log(f"hgb {config}: done")


# ---------------------------------------------------------------- 3. evaluation

def _assemble_hgb(dd: Data, config: str) -> dict[str, Any]:
    c = dd.c
    d = out_dir() / "hgb" / config
    out = dict(val=np.full(len(c.y), np.nan), aug=np.full(len(dd.ya), np.nan), es={})
    for t, e in dd.ev.items():
        out[t] = np.full(len(e["y"]), np.nan)
    info = {}
    for f in range(FO.N_OUTER):
        z = D.load_npz(d / f"fold{f}.npz")
        j = json.loads((d / f"fold{f}.json").read_text(encoding="utf-8"))
        out["val"][c.fold == f] = z["val"]
        out["aug"][dd.fold_a == f] = z["aug"]
        out["es"][f] = (np.flatnonzero(np.isin(c.g.astype(str), j["early_stop_matches"]) & (c.fold != f)), z["es"])
        for t, e in dd.ev.items():
            out[t][e["fold"] == f] = z[t]
        info[f] = j
    out["info"] = info
    return out


def _assemble_gru(dd: Data, config: str, seed: int, cap: int) -> dict[str, Any] | None:
    c = dd.c
    d = out_dir() / "neural" / config / f"seed{seed}"
    if not (d / "done.json").exists():
        return None
    out = dict(val=np.full(len(c.y), np.nan), es={})
    for t, e in dd.ev.items():
        out[t] = np.full(len(e["y"]), np.nan)
    info = {}
    for f in range(FO.N_OUTER):
        z = D.load_npz(d / f"fold{f}.npz")
        if (d / f"fold{f}_aug.npz").exists():
            z.update(D.load_npz(d / f"fold{f}_aug.npz"))
        j = json.loads((d / f"fold{f}.json").read_text(encoding="utf-8"))
        out["val"][c.fold == f] = z[f"c{cap}_val"]
        if f"c{cap}_aug" in z:
            out.setdefault("aug", np.full(len(dd.ya), np.nan))[dd.fold_a == f] = z[f"c{cap}_aug"]
        out["es"][f] = (np.flatnonzero(np.isin(c.g.astype(str), j["early_stop_matches"]) & (c.fold != f)), z[f"c{cap}_es"])
        for t, e in dd.ev.items():
            out[t][e["fold"] == f] = z[f"c{cap}_{t}"]
        info[f] = dict(j["caps"][str(cap)], fit_s=j["fit_s"])
    out["info"] = info
    return out


def _ensemble(dd: Data, g: dict[str, Any], h: dict[str, Any], fixed: float | None = None) -> dict[str, Any]:
    """w * GRU + (1 - w) * HGB; w per outer fold from the predeclared grid by log loss on that fold's early-stop matches (training
    partition only: GRU early-stop predictions, HGB refitted without those matches), or fixed."""
    c = dd.c
    out: dict[str, Any] = {k: np.full_like(v, np.nan) for k, v in g.items() if isinstance(v, np.ndarray) and k in h}
    ws = {}
    for f in range(FO.N_OUTER):
        if fixed is None:
            (ig, pg), (ih, ph) = g["es"][f], h["es"][f]
            if not np.array_equal(ig, ih):
                raise RuntimeError("early-stop rows differ between GRU and HGB")
            y = c.y[ig]
            ll = {w: float(MT.ll_vec(y, w * pg + (1 - w) * ph).mean()) for w in WEIGHTS}
            w = min(ll, key=ll.get)
        else:
            w = fixed
        ws[f] = w
        for k in out:
            fold = c.fold if k == "val" else dd.fold_a if k == "aug" else dd.ev[k]["fold"]
            m = fold == f
            out[k][m] = w * g[k][m] + (1 - w) * h[k][m]
    out["weights"] = ws
    return out


def _ba_ci(y, p, g, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    ms = np.unique(g)
    idx = [np.flatnonzero(g == m) for m in ms]
    v = [MT.bal_acc(y[s], p[s]) for s in (np.concatenate([idx[k] for k in rng.integers(0, len(ms), len(ms))]) for _ in range(n_boot))]
    return [MT._r(np.quantile(v, 0.025)), MT._r(np.quantile(v, 0.975))]


def _abstain(dd: Data, p: np.ndarray, pa: np.ndarray | None, target: float = 0.05) -> dict[str, Any]:
    """Nested thresholds (other seven folds, centred independent OOF) applied to each fold, centred and shifted."""
    c = dd.c
    conf, wrong = np.maximum(p, 1 - p), (p > 0.5).astype(int) != c.y
    com = np.zeros(len(p), bool)
    th = {}
    for f in range(FO.N_OUTER):
        o = c.indep & (c.fold != f)
        order = np.argsort(-conf[o], kind="stable")
        cum = np.cumsum(wrong[o][order]) / np.arange(1, o.sum() + 1)
        ok = np.flatnonzero(cum <= target)
        th[f] = float(conf[o][order][ok[-1]]) if len(ok) else 1.0 + 1e-9
        com |= (c.fold == f) & (conf >= th[f])
    m = c.indep
    r = dict(target_error=target, thresholds=[MT._r(th[f]) for f in range(FO.N_OUTER)], centred_coverage=MT._r(com[m].mean()),
             centred_error=MT._r(wrong[m & com].mean()) if (m & com).any() else None)
    if pa is not None:
        ca, wa = np.maximum(pa, 1 - pa), (pa > 0.5).astype(int) != dd.ya
        tha = np.array([th[f] for f in dd.fold_a])
        ka = dd.indep_a & (ca >= tha)
        r.update(shifted_coverage=MT._r(ka.sum() / dd.indep_a.sum()), shifted_error=MT._r(wa[ka].mean()) if ka.any() else None)
    r["aurc_centred"] = MT.coverage_curve(c.y[m], p[m])["aurc"]
    return r


def _short(y, p):
    r = MT.basic(y, p)
    return {k: r[k] for k in ("n", "balanced_accuracy", "log_loss", "roc_auc", "brier", "ece")}


def score_model(dd: Data, P: dict[str, Any], ci: bool = True) -> dict[str, Any]:
    c = dd.c
    m = c.indep
    out: dict[str, Any] = dict(centred=_short(c.y[m], P["val"][m]))
    if ci:
        out["centred"]["ba_ci"] = _ba_ci(c.y[m], P["val"][m], c.g[m])
    st = c.meta["status_in_gap"].to_numpy()
    out["centred_by_status"] = {s: dict(_short(c.y[m & (st == s)], P["val"][m & (st == s)]),
                                        **({"ba_ci": _ba_ci(c.y[m & (st == s)], P["val"][m & (st == s)], c.g[m & (st == s)], 500)} if ci else {}))
                                for s in np.unique(st)}
    close = c.meta["d_band"].to_numpy() == "<0.05"
    out["centred_close_dmin_lt_5cm"] = _short(c.y[m & close], P["val"][m & close])
    if "aug" in P and np.isfinite(P["aug"]).all():
        a, ma = P["aug"], dd.indep_a
        sh = dd.ma["shift_s"].to_numpy()
        out["shifted_all"] = _short(dd.ya[ma], a[ma])
        if ci:
            out["shifted_all"]["ba_ci"] = _ba_ci(dd.ya[ma], a[ma], dd.ga[ma])
        out["by_shift"] = {f"{s:+.1f}": dict(n=int((ma & (sh == s)).sum()), balanced_accuracy=MT._r(MT.bal_acc(dd.ya[ma & (sh == s)], a[ma & (sh == s)])),
                                             log_loss=MT._r(MT.ll_vec(dd.ya[ma & (sh == s)], a[ma & (sh == s)]).mean()))
                           for s in sorted(np.unique(sh))}
        out["by_shift"]["+0.0"] = dict(n=int(m.sum()), balanced_accuracy=out["centred"]["balanced_accuracy"], log_loss=out["centred"]["log_loss"])
        sa = dd.ma["status_in_gap"].to_numpy()
        out["shifted_by_status"] = {s: _short(dd.ya[ma & (sa == s)], a[ma & (sa == s)]) for s in np.unique(sa)}
        cl = dd.ma["d_band"].to_numpy() == "<0.05"
        out["shifted_close_dmin_lt_5cm"] = _short(dd.ya[ma & cl], a[ma & cl])
    for t in ("g0p80_shift", "g0p40", "g1p60", "smooth5", "noise5cm"):
        e = dd.ev[t]
        out[t] = _short(e["y"][e["indep"]], P[t][e["indep"]])
        if t in ("smooth5", "noise5cm"):
            out[t + "_by_status"] = {s: MT._r(MT.bal_acc(c.y[m & (st == s)], P[t][m & (st == s)])) for s in np.unique(st)}
    out["abstention_5pct"] = _abstain(dd, P["val"], P.get("aug"))
    return out


def evaluate(version: str = "v1") -> dict[str, Any]:
    od = out_dir() / "metrics" / version
    if (od / "comparison.json").exists():
        raise SystemExit(f"{od} exists; evaluation versions are never overwritten")
    dd = Data(need_features=False)
    c = dd.c
    res: dict[str, Any] = dict(models={}, paired={}, seeds={}, ensembles={})
    preds: dict[str, dict[str, Any]] = {}
    for cf in HGB_CONFIGS:
        if (out_dir() / "hgb" / cf / f"fold{FO.N_OUTER - 1}.json").exists():
            preds[f"hgb_{cf}"] = _assemble_hgb(dd, cf)
    for cf in ("A", "B", "C", "Anoise", "Asmooth"):
        for s in (0, 1, 2):
            for cap in CAPS:
                P = _assemble_gru(dd, cf, s, cap)
                if P is not None:
                    preds[f"gru_{cf}_c{cap}_s{s}"] = P
    repro = {}
    for s in (0, 1, 2):
        if f"gru_A_c80_s{s}" in preds:
            old = D.load_npz(pdir(PRIMARY, "gru_gaplast") / f"seed{s}.npz")["p"]
            new = preds[f"gru_A_c80_s{s}"]["val"]
            repro[s] = dict(max_abs_diff=float(np.abs(new - old).max()), ba_new=MT._r(MT.bal_acc(c.y[c.indep], new[c.indep])),
                            ba_stage4b=MT._r(MT.bal_acc(c.y[c.indep], old[c.indep])))
    res["stage4b_reproduction"] = repro
    for cf in ("A", "B", "C"):
        for cap in CAPS:
            for s in (0, 1, 2):
                g, h = preds.get(f"gru_{cf}_c{cap}_s{s}"), preds.get(f"hgb_{cf}")
                if g is None or h is None:
                    continue
                e = _ensemble(dd, g, h)
                preds[f"ens_{cf}_c{cap}_s{s}"] = e
                res["ensembles"][f"ens_{cf}_c{cap}_s{s}"] = dict(weights_by_fold=e["weights"])
                preds[f"ens50_{cf}_c{cap}_s{s}"] = _ensemble(dd, g, h, fixed=0.5)
    for name, P in preds.items():
        seed0 = name.startswith("hgb_") or name.endswith("_s0")
        res["models"][name] = score_model(dd, P, ci=seed0)
        if "info" in P:
            inf = P["info"]
            res["models"][name]["training"] = dict(
                best_epoch=[inf[f].get("best_epoch") for f in range(FO.N_OUTER)], epochs_run=[inf[f].get("epochs_run") for f in range(FO.N_OUTER)],
                train_log_loss_mean=MT._r(np.mean([inf[f]["train"]["log_loss"] for f in range(FO.N_OUTER)])),
                val_log_loss_mean=MT._r(np.mean([inf[f]["val"]["log_loss"] for f in range(FO.N_OUTER)])),
                es_loss_mean=MT._r(np.mean([inf[f]["es_loss"] for f in range(FO.N_OUTER)])) if "es_loss" in inf[0] else None,
                symmetry_max_err=max(inf[f].get("symmetry_max_err_outgoing", 0) for f in range(FO.N_OUTER)),
                fit_s_total=MT._r(sum(inf[f].get("fit_s", 0) for f in range(FO.N_OUTER)), 1))
        log(f"eval {name}: centred BA {res['models'][name]['centred']['balanced_accuracy']} "
            f"shifted BA {res['models'][name].get('shifted_all', {}).get('balanced_accuracy')}")
    keys = [("centred", "balanced_accuracy"), ("centred", "log_loss"), ("centred", "ece"), ("shifted_all", "balanced_accuracy"),
            ("shifted_all", "log_loss"), ("g0p80_shift", "balanced_accuracy"), ("g0p40", "balanced_accuracy"), ("g1p60", "balanced_accuracy"),
            ("smooth5", "balanced_accuracy"), ("noise5cm", "balanced_accuracy")]
    groups = sorted({n.rsplit("_s", 1)[0] for n in res["models"] if re.search(r"_s\d$", n)})
    for gname in groups:
        ms = [res["models"][f"{gname}_s{s}"] for s in (0, 1, 2) if f"{gname}_s{s}" in res["models"]]
        agg: dict[str, Any] = {"n_seeds": len(ms)}
        for a, b in keys:
            v = [m[a][b] for m in ms if a in m and m[a].get(b) is not None]
            if v:
                agg[f"{a}.{b}"] = dict(mean=MT._r(np.mean(v)), sd=MT._r(np.std(v, ddof=1)) if len(v) > 1 else None)
        for st in ("both_fully_visible", "both_fully_estimated", "mixed"):
            v = [m["centred_by_status"][st]["balanced_accuracy"] for m in ms if st in m["centred_by_status"]]
            if v:
                agg[f"centred_{st}.balanced_accuracy"] = dict(mean=MT._r(np.mean(v)), sd=MT._r(np.std(v, ddof=1)) if len(v) > 1 else None)
        res["seeds"][gname] = agg

    def pair(a, b):
        if a not in preds or b not in preds:
            return None
        r = dict(centred=MT.paired(c.y[c.indep], preds[a]["val"][c.indep], preds[b]["val"][c.indep], c.g[c.indep], 1000))
        if "aug" in preds[a] and "aug" in preds[b]:
            m = dd.indep_a
            r["shifted"] = MT.paired(dd.ya[m], preds[a]["aug"][m], preds[b]["aug"][m], dd.ga[m], 1000)
        for t in ("g1p60", "noise5cm"):
            e = dd.ev[t]
            r[t] = MT.paired(e["y"][e["indep"]], preds[a][t][e["indep"]], preds[b][t][e["indep"]], e["g"][e["indep"]], 1000)
        return r
    cmp_list = [("gru_A_c120_s0", "gru_A_c80_s0"), ("gru_A_c160_s0", "gru_A_c80_s0"), ("gru_B_c160_s0", "gru_A_c160_s0"),
                ("gru_C_c160_s0", "gru_A_c160_s0"), ("gru_B_c80_s0", "gru_A_c80_s0"), ("gru_C_c80_s0", "gru_A_c80_s0"),
                ("hgb_B", "hgb_A"), ("hgb_C", "hgb_A"), ("gru_A_c160_s0", "hgb_A"), ("gru_B_c160_s0", "hgb_B"), ("gru_C_c160_s0", "hgb_C"),
                ("gru_B_c160_s0", "hgb_A"), ("gru_C_c160_s0", "hgb_A"),
                ("ens_A_c160_s0", "gru_A_c160_s0"), ("ens_A_c160_s0", "hgb_A"), ("ens_B_c160_s0", "gru_B_c160_s0"), ("ens_B_c160_s0", "hgb_B"),
                ("ens_C_c160_s0", "gru_C_c160_s0"), ("ens_C_c160_s0", "hgb_C"), ("ens50_A_c160_s0", "ens_A_c160_s0"),
                ("gru_C_c160_s0", "gru_A_c80_s0"), ("gru_C_c160_s0", "gru_B_c160_s0"), ("gru_A_c160_s0", "hgb_C"),
                ("gru_C_c160_s0", "hgb_C"), ("hgb_C", "hgb_B"), ("ens_C_c160_s0", "hgb_A"),
                ("gru_Anoise_c160_s0", "gru_A_c160_s0"), ("gru_Asmooth_c160_s0", "gru_A_c160_s0"), ("hgb_Anoise", "hgb_A"), ("hgb_Asmooth", "hgb_A")]
    for a, b in cmp_list:
        r = pair(a, b)
        if r is not None:
            res["paired"][f"{a} - {b}"] = r
    od.mkdir(parents=True, exist_ok=True)
    from .run import dumps
    (od / "comparison.json").write_text(dumps(res), encoding="utf-8")
    np.savez_compressed(od / "predictions_centred.npz", **{k: v["val"] for k, v in preds.items()})
    log(f"stage4c evaluate {version}: {len(res['models'])} models")
    return res


def runtime(n: int = 2000) -> dict[str, Any]:
    """Inference cost per sample (fold-0 models, first n samples of fold 0): HGB including motion-feature extraction; GRU on the GPU
    and on the CPU. Wall-clock on this workstation; indicative only."""
    import platform

    import psutil
    import torch
    p_out = out_dir() / "runtime" / "runtime.json"
    if p_out.exists():
        return json.loads(p_out.read_text(encoding="utf-8"))
    ctx = Ctx(PRIMARY, need_seq=True)
    idx = np.flatnonzero(ctx.fold == 0)[:n]
    x = {k: v[idx] for k, v in ctx.x.items()}
    res: dict[str, Any] = dict(n_samples=int(len(idx)), cpu=platform.processor(), threads=psutil.cpu_count(),
                               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    t = time.perf_counter()
    F, _, _ = D.features(x)
    t_feat = time.perf_counter() - t
    for cf in ("A", "C"):
        m = joblib.load(pdir(PRIMARY, "hgb") / "folds" / "seed0_fold0.joblib") if cf == "A" else joblib.load(out_dir() / "hgb" / cf / "fold0.joblib")
        t = time.perf_counter()
        m.predict(F)
        res[f"hgb_{cf}_ms_per_sample"] = dict(features=MT._r(1e3 * t_feat / n, 3), model=MT._r(1e3 * (time.perf_counter() - t) / n, 3))
    for dev in (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]:
        from .stage4c import KIND, predict
        net = MD.make_net(KIND, dropout=0.1, **MD.NEURAL[KIND]).to(dev)
        net.load_state_dict(torch.load(out_dir() / "neural" / "C" / "seed0" / "fold0_c160.pt", map_location=dev))
        si, so, g = ctx.si[idx], ctx.so[idx], ctx.gap[idx].astype(np.float32)
        predict(net, si, so, g, dev)                                  # warm-up
        t = time.perf_counter()
        for _ in range(3):
            predict(net, si, so, g, dev)
        res[f"gru_{dev}_ms_per_sample"] = MT._r(1e3 * (time.perf_counter() - t) / (3 * n), 4)
    res["gru_params"] = MD.n_params(net)
    res["sequence_input_build_note"] = "seq_tensor is a reshaping of the 7 input fields (negligible next to the model)"
    write_json_once(p_out, res)
    return res


if __name__ == "__main__":
    t = time.perf_counter()
    print(json.dumps(audit(), indent=1, default=str)[:6000])
    print(f"{time.perf_counter() - t:.0f} s")
