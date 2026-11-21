"""Stage 4 runner: ``python -m football_fatigue.m11_benchmark <phase> [options]``.

Phases: ``gates``, ``physics``, ``tabular``, ``controls``, ``neural``, ``evaluate`` (and ``robust`` in :mod:`.robust`). Every fitted unit
(model, seed, outer fold) writes its own files once under ``data/processed/m11-stage4-benchmarks-v1``; re-running a phase skips
finished units (resume), so a crash never forces recomputation or mixes configurations. Only the 32 train/CV matches are read.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from football_fatigue.m11_recovery.motion import BASELINES

from . import data as D
from . import folds as FO
from . import metrics as MT
from . import models as MD

PRIMARY = "g0p80"
N_PERM = 20
SEEDS = (0, 1, 2)
TABULAR = ("lr", "knn", "svm", "rf", "hgb")
NEURAL = ("mlp", "gru", "lstm")
# Physics models (second version, "phys_" prefix; the first version in preds/<table>/B*_ clipped the calibration weight at 0 and
# is superseded, kept on disk unchanged). B0-B2: fixed rules; B3: learned sign, best single baseline by inner CV; B3 combo.
PHYS = {"phys_B0_nearest": ["nearest"], "phys_B1_cv_fwd": ["cv_fwd"], "phys_B1_cv_bidir": ["cv_bidir"], "phys_B1_cv1_bidir": ["cv1_bidir"],
        "phys_B2_ca_bidir": ["ca_bidir"], "phys_B2_ca2_bidir": ["ca2_bidir"], "phys_B3_best_single": list(BASELINES),
        "phys_B3_combo": ["combo"]}


def dumps(o: Any) -> str:
    from football_fatigue.window_intersections.__main__ import dumps as _d
    return _d(o)


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    p = D.out_dir() / "PROGRESS.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_json_once(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    txt = dumps(obj)
    if path.exists():
        if path.read_text(encoding="utf-8") != txt:
            raise RuntimeError(f"{path} exists with different content; never overwritten")
        return
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(txt, encoding="utf-8")
    tmp.replace(path)


class Ctx:
    """Features, labels, groups, folds and the independent mask of one table (inputs through the adapter only)."""

    def __init__(self, table: str = PRIMARY, need_seq: bool = False):
        self.table = table
        ft = D.feature_table(table, log)
        self.F, self.margins, self.names = ft["F"], ft["margins"], ft["names"]
        self.k = D.n_perm(self.names)
        x, self.y, self.g = D.load_inputs(table)
        if not np.array_equal(self.g, ft["groups"]):
            raise RuntimeError("feature cache is not aligned with the inputs")
        self.y = self.y.astype(int)
        self.gap = x["gap_s"].astype(np.float64)
        meta = D.load_meta(table)
        if not (meta["y"].to_numpy() == self.y).all():
            raise RuntimeError("metadata not aligned")
        self.meta = meta
        self.indep = meta["independent"].to_numpy(bool)
        self.fmap = FO.outer_folds(sorted({str(int(v)) for v in self.g}, key=int))
        self.fold = FO.fold_index(self.g, self.fmap)
        if need_seq:
            self.si, self.so = D.seq_tensor(x)
        self.x = x if need_seq else None

    def train_matches(self, f: int) -> list[str]:
        return sorted((m for m, k in self.fmap.items() if k != f), key=int)


def pdir(table: str, model: str) -> Path:
    return D.out_dir() / "preds" / table / model


# ---------------------------------------------------------------- physics

def run_physics(ctx: Ctx) -> None:
    for name, cands in PHYS.items():
        d = pdir(ctx.table, name)
        if (d / "seed0.npz").exists():
            log(f"physics {name}: done (skip)")
            continue
        p, info = np.full(len(ctx.y), np.nan), {}
        for f in range(FO.N_OUTER):
            tr, va = ctx.fold != f, ctx.fold == f
            if cands == ["combo"]:
                cm = MD.PhysicsCombo().fit(ctx.margins[tr], ctx.gap[tr], ctx.y[tr])
                p[va] = cm.predict(ctx.margins[va], ctx.gap[va])
                info[f] = dict(selected="combo", coef=cm.lr.coef_[0].tolist(), train=MT.basic(ctx.y[tr], cm.predict(ctx.margins[tr], ctx.gap[tr])))
                continue
            rule = len(cands) == 1
            if rule:
                kb = BASELINES.index(cands[0])
                sel, inner = cands[0], None
            else:
                inner = {}
                imap = FO.inner_folds(ctx.train_matches(f), f)
                ifold = FO.fold_index(ctx.g[tr], imap)
                for b in cands:
                    kk = BASELINES.index(b)
                    xb, yt = MD.physics_x(ctx.margins[tr], ctx.gap[tr], kk), ctx.y[tr]
                    q = np.zeros(tr.sum())
                    for i in range(FO.N_INNER):
                        w = MD.fit_scale(xb[ifold != i], yt[ifold != i], rule=False)
                        q[ifold == i] = MD.physics_predict(xb[ifold == i], w)
                    inner[b] = float(MT.ll_vec(yt, q).mean())
                sel = min(inner, key=inner.get)
                kb = BASELINES.index(sel)
            w = MD.fit_scale(MD.physics_x(ctx.margins[tr], ctx.gap[tr], kb), ctx.y[tr], rule=rule)
            p[va] = MD.physics_predict(MD.physics_x(ctx.margins[va], ctx.gap[va], kb), w)
            ptr = MD.physics_predict(MD.physics_x(ctx.margins[tr], ctx.gap[tr], kb), w)
            info[f] = dict(selected=sel, weight=w, inner_log_loss=inner, train=MT.basic(ctx.y[tr], ptr))
        D.write_once_npz(d / "seed0.npz", dict(p=p, fold=ctx.fold))
        write_json_once(d / "seed0.json", dict(model=name, candidates=cands, folds=info))
        r = MT.basic(ctx.y[ctx.indep], p[ctx.indep])
        log(f"physics {name}: independent BA {r['balanced_accuracy']} LL {r['log_loss']}")


# ---------------------------------------------------------------- tabular (nested grouped CV)

def _inner_oof(name, cfg, ctx, tr, f, seed):
    imap = FO.inner_folds(ctx.train_matches(f), f)
    ifold = FO.fold_index(ctx.g[tr], imap)
    X, y = ctx.F[tr], ctx.y[tr]
    raw = np.zeros(len(y))
    for i in range(FO.N_INNER):
        m = MD.Tabular(name, cfg, ctx.k, seed).fit(X[ifold != i], y[ifold != i])
        raw[ifold == i] = m.raw(X[ifold == i])
    return raw, y


def run_tabular(ctx: Ctx, models=TABULAR, weights: np.ndarray | None = None, tag: str = "") -> None:
    for name in models:
        seeds = SEEDS if name in MD.STOCHASTIC else (0,)
        for seed in seeds:
            mdl = name + tag
            d = pdir(ctx.table, mdl)
            if (d / f"seed{seed}.npz").exists():
                log(f"tabular {mdl} seed {seed}: done (skip)")
                continue
            t0 = time.perf_counter()
            p, info = np.full(len(ctx.y), np.nan), {}
            for f in range(FO.N_OUTER):
                part = d / "folds" / f"seed{seed}_fold{f}.npz"
                tr, va = ctx.fold != f, ctx.fold == f
                if part.exists():
                    z = D.load_npz(part)
                    p[va] = z["p"]
                    info[f] = json.loads(part.with_suffix(".json").read_text(encoding="utf-8"))
                    continue
                t1 = time.perf_counter()
                grid = MD.TABULAR_GRIDS[name]
                inner = {}
                sel_file = d / "folds" / f"select_fold{f}.json"
                if sel_file.exists():
                    sj = json.loads(sel_file.read_text(encoding="utf-8"))
                    cfg, inner = sj["selected"], sj["inner_log_loss"]
                else:
                    if len(grid) == 1:
                        cfg = grid[0]
                    else:
                        for j, c in enumerate(grid):
                            raw, yt = _inner_oof(name, c, ctx, tr, f, 0)
                            q = MD._sigmoid(MD.fit_platt(raw, yt) * raw) if name == "svm" else raw
                            inner[str(j)] = float(MT.ll_vec(yt, q).mean())
                        cfg = grid[int(min(inner, key=inner.get))]
                    write_json_once(sel_file, dict(selected=cfg, inner_log_loss=inner, grid=grid, selection_seed=0))
                m = MD.Tabular(name, cfg, ctx.k, seed)
                if weights is not None:
                    X = np.nan_to_num(ctx.F[tr])
                    m.m = MD.make_tabular(name, cfg, X.shape[1], seed).fit(np.r_[X, D.swap_features(X, ctx.k)], np.r_[ctx.y[tr], 1 - ctx.y[tr]],
                                                                           **_wkw(name, np.r_[weights[tr], weights[tr]]))
                else:
                    m.fit(ctx.F[tr], ctx.y[tr])
                if name == "svm":
                    raw, yt = _inner_oof(name, cfg, ctx, tr, f, seed)
                    m.platt = MD.fit_platt(raw, yt)
                pv, ptr = m.predict(ctx.F[va]), m.predict(ctx.F[tr])
                sym = float(np.abs(m.predict(D.swap_features(ctx.F[va], ctx.k)) - (1 - pv)).max())
                p[va] = pv
                info[f] = dict(selected=cfg, inner_log_loss=inner, train=MT.basic(ctx.y[tr], ptr), val=MT.basic(ctx.y[va], pv),
                               symmetry_max_err=sym, platt=m.platt, fit_s=round(time.perf_counter() - t1, 1))
                joblib.dump(m, (d / "folds" / f"seed{seed}_fold{f}.joblib"), compress=3)
                D.write_once_npz(part, dict(p=pv))
                write_json_once(part.with_suffix(".json"), info[f])
            D.write_once_npz(d / f"seed{seed}.npz", dict(p=p, fold=ctx.fold))
            write_json_once(d / f"seed{seed}.json", dict(model=mdl, seed=seed, grid=MD.TABULAR_GRIDS[name], folds=info,
                                                        weighted=weights is not None, runtime_s=round(time.perf_counter() - t0, 1)))
            r = MT.basic(ctx.y[ctx.indep], p[ctx.indep])
            log(f"tabular {mdl} seed {seed}: independent BA {r['balanced_accuracy']} LL {r['log_loss']} ({time.perf_counter() - t0:.0f} s)")


def _wkw(name, w):
    if name in ("lr", "knn", "svm"):
        key = {"lr": "logisticregression", "svm": "linearsvc"}.get(name)
        if key is None:
            raise ValueError("KNN does not support sample weights")
        return {f"{key}__sample_weight": w}
    return {"sample_weight": w}


# ---------------------------------------------------------------- controls (gates 10 and 11)

def run_controls(ctx: Ctx) -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    out: dict[str, Any] = {}
    rng = np.random.default_rng(20261114)

    def oof(X, y_train, plain: bool, k: int = 0):
        p = np.zeros(len(ctx.y))
        for f in range(FO.N_OUTER):
            tr, va = ctx.fold != f, ctx.fold == f
            if plain:
                m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=40, early_stopping=False,
                                                   random_state=0).fit(X[tr], y_train[tr])
                p[va] = m.predict_proba(X[va])[:, 1]
            else:
                m = MD.Tabular("hgb", MD.TABULAR_GRIDS["hgb"][2], k, 0).fit(X[tr], y_train[tr])
                p[va] = m.predict(X[va])
        return p

    def score(p):
        r = MT.basic(ctx.y[ctx.indep], p[ctx.indep])
        b = MT.bootstrap(ctx.y[ctx.indep], p[ctx.indep], ctx.g[ctx.indep], n_boot=1000)
        return dict(balanced_accuracy=r["balanced_accuracy"], log_loss=r["log_loss"], roc_auc=r["roc_auc"], ci=b["balanced_accuracy_ci"])

    # 10: label-permutation null distribution. Labels are shuffled within each match (N_PERM seeds), the model is trained on the
    # shuffled labels and scored on the true labels. One random-label fit is a single random function applied coherently to all
    # samples, so its score varies by several points between seeds; the match bootstrap of ONE fit does not capture that variance
    # (a first single-permutation version of this gate was therefore ill-posed: 0.524 for one seed, 0.479-0.552 across seeds).
    # Pass: null mean within 3 standard errors of 0.5, and the true-label model far outside the null range.
    null = []
    for s_ in range(N_PERM):
        r_ = np.random.default_rng(1000 + s_)
        yp = ctx.y.copy()
        for m in np.unique(ctx.g):
            i = np.flatnonzero(ctx.g == m)
            yp[i] = ctx.y[r_.permutation(i)]
        pp = oof(ctx.F, yp, plain=False, k=ctx.k)
        null.append(MT.bal_acc(ctx.y[ctx.indep], pp[ctx.indep]))
    null = np.array(null)
    true_ref = score(oof(ctx.F, ctx.y, plain=False, k=ctx.k))
    se = null.std(ddof=1) / np.sqrt(len(null))
    out["label_permutation_null_hgb_motion_features"] = dict(
        n_permutations=int(len(null)), balanced_accuracy_values=[round(float(v), 4) for v in null], mean=round(float(null.mean()), 4),
        sd=round(float(null.std(ddof=1)), 4), se_mean=round(float(se), 4), min=round(float(null.min()), 4), max=round(float(null.max()), 4),
        mean_within_3se_of_chance=bool(abs(null.mean() - 0.5) <= 3 * se), true_label_balanced_accuracy=true_ref["balanced_accuracy"],
        true_minus_null_max=round(float(true_ref["balanced_accuracy"] - null.max()), 4),
        pass_=bool(abs(null.mean() - 0.5) <= 3 * se and true_ref["balanced_accuracy"] - null.max() > 0.1))
    out["true_labels_same_model_reference"] = true_ref
    # 11: shortcut controls on non-motion information
    x, _, _ = D.load_inputs(ctx.table)
    def mstats(mk):
        n = mk.sum(-1)
        first = np.where(mk.any(-1), mk.argmax(-1), -1)
        last = np.where(mk.any(-1), mk.shape[-1] - 1 - mk[..., ::-1].argmax(-1), -1)
        return np.concatenate([n, first, last], 1).astype(float)
    Xm = np.concatenate([mstats(x["in_mask"]), mstats(x["out_mask"])], 1)
    out["mask_pattern_only"] = score(oof(Xm, ctx.y, plain=True))
    mt = ctx.meta
    cat = lambda s: s.astype("category").cat.codes.to_numpy(float)  # noqa: E731
    Xmeta = np.c_[mt["d_min"], mt["chord_speed_min_m_s"], mt["rel_speed_m_s"], cat(mt["crossing_class"]), cat(mt["team_relation"]),
                  mt["est_frac_used_in0"].fillna(-1), mt["est_frac_used_in1"].fillna(-1), mt["est_frac_in_gap_in0"].fillna(-1),
                  mt["est_frac_in_gap_in1"].fillna(-1), mt["frame_mode"], mt["t_star"], mt["period"], mt["n_valid_in0"], mt["n_valid_in1"],
                  mt["n_valid_out0"], mt["n_valid_out1"], *[mt[c] for c in mt.columns if c.startswith("cohort_")]]
    out["metadata_only_excluding_order_bits"] = score(oof(Xmeta, ctx.y, plain=True))
    out["context_features_only_symmetric"] = score(oof(ctx.F[:, ctx.k:], ctx.y, plain=True))
    out["positive_control_order_bits_metadata"] = score(oof(mt[["in_order", "out_order"]].to_numpy(float), ctx.y, plain=True))
    chance = lambda r: r["ci"][0] <= 0.5 <= r["ci"][1] or abs(r["balanced_accuracy"] - 0.5) < 0.02  # noqa: E731
    out["pass"] = out["label_permutation_null_hgb_motion_features"]["pass_"] and all(
        chance(out[k]) for k in ("mask_pattern_only", "metadata_only_excluding_order_bits", "context_features_only_symmetric"))
    out["note"] = ("Controls must stay at chance; the order-bit positive control must be ~1.0 (it shows the metadata path would leak "
                   "if it ever reached a model, which the adapter prevents).")
    return out


# ---------------------------------------------------------------- neural

def run_neural(ctx: Ctx, kinds=NEURAL, seeds=(0,), weights: np.ndarray | None = None, tag: str = "", transform=None) -> None:
    import torch
    torch.backends.cudnn.benchmark = False
    for kind in kinds:
        mdl = kind + tag
        for seed in seeds:
            d = pdir(ctx.table, mdl)
            if (d / f"seed{seed}.npz").exists():
                log(f"neural {mdl} seed {seed}: done (skip)")
                continue
            t0 = time.perf_counter()
            p, info = np.full(len(ctx.y), np.nan), {}
            for f in range(FO.N_OUTER):
                part = d / "folds" / f"seed{seed}_fold{f}.npz"
                va = ctx.fold == f
                if part.exists():
                    p[va] = D.load_npz(part)["p"]
                    info[f] = json.loads(part.with_suffix(".json").read_text(encoding="utf-8"))
                    continue
                t1 = time.perf_counter()
                es = set(FO.early_stop_matches(ctx.train_matches(f), f))
                ges = np.isin(ctx.g.astype(str), list(es))
                tr = (ctx.fold != f) & ~ges
                ev = (ctx.fold != f) & ges
                si, so = (ctx.si, ctx.so) if transform is None else transform
                m = MD.Neural(kind, seed=seed)
                m.fit(si[tr], so[tr], ctx.gap[tr].astype(np.float32), ctx.y[tr], si[ev], so[ev], ctx.gap[ev].astype(np.float32), ctx.y[ev],
                      weights=None if weights is None else weights[tr])
                pv = m.predict(si[va], so[va], ctx.gap[va].astype(np.float32))
                ptr = m.predict(si[tr], so[tr], ctx.gap[tr].astype(np.float32))
                pe = m.predict(si[ev], so[ev], ctx.gap[ev].astype(np.float32))
                sym_o = float(np.abs(m.predict(si[va], so[va][:, ::-1].copy(), ctx.gap[va].astype(np.float32)) - (1 - pv)).max())
                sym_i = float(np.abs(m.predict(si[va][:, ::-1].copy(), so[va], ctx.gap[va].astype(np.float32)) - (1 - pv)).max())
                p[va] = pv
                info[f] = dict(params=MD.n_params(m.net), best_epoch=m.best_epoch, epochs_run=len(m.history), history=m.history,
                               early_stop_matches=sorted(es, key=int), train=MT.basic(ctx.y[tr], ptr), early_stop=MT.basic(ctx.y[ev], pe),
                               val=MT.basic(ctx.y[va], pv), symmetry_max_err_outgoing=sym_o, symmetry_max_err_incoming=sym_i,
                               fit_s=round(time.perf_counter() - t1, 1), device=next(m.net.parameters()).device.type,
                               valid=bool(np.isfinite(pv).all()))
                (d / "folds").mkdir(parents=True, exist_ok=True)
                torch.save(m.net.state_dict(), d / "folds" / f"seed{seed}_fold{f}.pt")
                D.write_once_npz(part, dict(p=pv))
                write_json_once(part.with_suffix(".json"), info[f])
                log(f"  {mdl} s{seed} f{f}: val BA {info[f]['val']['balanced_accuracy']} LL {info[f]['val']['log_loss']} "
                    f"(train LL {info[f]['train']['log_loss']}, epoch {m.best_epoch}/{len(m.history)}, {info[f]['fit_s']} s)")
            D.write_once_npz(d / f"seed{seed}.npz", dict(p=p, fold=ctx.fold))
            write_json_once(d / f"seed{seed}.json", dict(model=mdl, seed=seed, config=dict(MD.NEURAL[kind], epochs=80, patience=10, batch=256,
                                                                                        lr=1e-3, weight_decay=1e-4, clip=1.0, dropout=0.1),
                                                        weighted=weights is not None, folds=info, runtime_s=round(time.perf_counter() - t0, 1)))
            r = MT.basic(ctx.y[ctx.indep], p[ctx.indep])
            log(f"neural {mdl} seed {seed}: independent BA {r['balanced_accuracy']} LL {r['log_loss']} ({time.perf_counter() - t0:.0f} s)")


# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=["gates", "physics", "tabular", "controls", "neural", "evaluate", "robust"])
    ap.add_argument("--table", default=PRIMARY)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--analyses", nargs="+", default=None)
    ap.add_argument("--version", default="v1", help="evaluation version directory (never overwritten)")
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    out = D.out_dir()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(python=platform.python_version(), argv=sys.argv[1:])
    if a.phase == "gates":
        from . import gates as G
        ctx = Ctx(PRIMARY)
        res = {}
        for name, fn in (("g1_manifests", G.g1_manifests), ("g2_split", G.g2_split), ("g3_loader_refuses_heldout", G.g3_loader_refuses_heldout),
                         ("g4_labels_and_order", G.g4_labels_and_order), ("g5_inputs_and_adaptation", lambda: G.g5_inputs_and_adaptation(ctx.names)),
                         ("g6_feature_symmetry", G.g6_feature_symmetry), ("g7_identity_audit", G.g7_identity_audit),
                         ("g9_lower_reconciliation", G.g9_lower_reconciliation)):
            t = time.perf_counter()
            res[name] = fn()
            log(f"gate {name}: {'PASS' if res[name]['pass'] else 'FAIL'} ({time.perf_counter() - t:.0f} s)")
        res["all_pass"] = all(v["pass"] for v in res.values() if isinstance(v, dict))
        write_json_once(out / "gates" / ("gates.json" if res["all_pass"] else f"gates_FAILED_{int(time.time())}.json"), res)
        if not res["all_pass"]:
            return 1
        write_json_once(out / "folds.json", dict(salt=FO.FOLD_SALT, n_outer=FO.N_OUTER, n_inner=FO.N_INNER, outer=res["g2_split"]["folds"]))
        log(f"gates all_pass={res['all_pass']}")
        return 0 if res["all_pass"] else 1
    gates = json.loads((out / "gates" / "gates.json").read_text(encoding="utf-8"))
    if not gates["all_pass"]:
        raise SystemExit("correctness gates have not all passed; refusing to train")
    if a.phase == "controls":
        res = run_controls(Ctx(a.table))
        write_json_once(out / "gates" / ("controls.json" if res["pass"] else f"controls_FAILED_{int(time.time())}.json"), res)
        log(f"controls pass={res['pass']}: " + ", ".join(f"{k} BA {v.get('balanced_accuracy', v.get('mean'))}" for k, v in res.items()
                                                       if isinstance(v, dict)))
        return 0 if res["pass"] else 1
    ctl = out / "gates" / "controls.json"
    if not ctl.exists() or not json.loads(ctl.read_text(encoding="utf-8"))["pass"]:
        raise SystemExit("negative controls have not passed; refusing to train")
    if a.phase == "physics":
        run_physics(Ctx(a.table))
    elif a.phase == "tabular":
        run_tabular(Ctx(a.table), tuple(a.models or TABULAR))
    elif a.phase == "neural":
        run_neural(Ctx(a.table, need_seq=True), tuple(a.models or NEURAL), tuple(a.seeds))
    elif a.phase == "evaluate":
        from . import evaluate as EVL
        EVL.main(a.table, a.version, a.models)
    elif a.phase == "robust":
        from . import robust as RB
        RB.main(a.analyses)
    log(f"phase {a.phase} {a.table} finished in {time.perf_counter() - t0:.0f} s  {env}")
    return 0
