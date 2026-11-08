"""M1 contestant C3 (gradient boosting + football-logic decoder), cross-validated by match on the 30 learning matches.
Never runs on the 16 test matches or the stress set. Inputs tag-free (inputs/*.npz, bfp/*.npz); labels only for training and
scoring (evalset/*.npz).

Layer 1 alive / dead: LightGBM multiclass on {alive, out over a line (outType T), whistle (W), goal (H / A), other dead};
P(dead) = 1 - P(alive). Layer 2 on alive rows: CONTROL / TRAVELLING / LOOSE (LOOSE weighted up). Owner: a LightGBM score
for each of the 5 players nearest the ball; the top one wins. Layer 3: Viterbi decoder (m1_possession.c3).
Each outer fold uses only the other folds' matches (C2-v2 decisions as features come from that fold's C2-v2 parameters).

    python scripts/m1/c3.py features --config none          # feature tables for one input-upgrade configuration
    python scripts/m1/c3.py ablate --config none            # fixed settings, CV score (with/without table)
    python scripts/m1/c3.py final --config <best>           # nested Optuna (LightGBM + decoder) inside the training folds
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import pickle
import shutil
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna

sys.path.insert(0, str(Path(__file__).parent))
from compare import full_score  # noqa: E402
from m1_labels.scoring import score_had  # noqa: E402
from m1_possession import SmoothedBall  # noqa: E402
from m1_possession import c3 as C3  # noqa: E402
from m1_possession.rules import C2Params, run_c2  # noqa: E402
from m1_possession.smoother import FLAG_MISSING  # noqa: E402
from m1_possession.splits import FOLDS, LEARNING_MATCHES  # noqa: E402
from m1_possession.upgrades import FLAG_INFERRED, fill_medium_gaps, gap_features, smooth_players  # noqa: E402

BASE = Path("data/processed/m1_possession")
INP, EVS, BFP = BASE / "inputs", BASE / "evalset", BASE / "bfp"
OUT = BASE / "c3"
CONFIGS = {"none": (0, 0, 0, 0), "smooth": (1, 0, 0, 0), "gapfill": (0, 1, 0, 0), "gapfeat": (0, 0, 1, 0), "bfp": (0, 0, 0, 1),
           "all": (1, 1, 1, 1)}
FIXED = dict(num_leaves=63, learning_rate=0.08, n_estimators=250, min_child_samples=50, colsample_bytree=0.7, subsample=0.7,
             subsample_freq=1, reg_lambda=1.0, loose_weight=3.0)
TRAIN_EVERY = 4            # training rows: every 4th 10 Hz row = 2.5 Hz
REASON = {"out_T": 1, "out_W": 2, "out_H": 3, "out_A": 3}
N_JOBS = 12


def cfg_of(name):
    if name in CONFIGS:
        return CONFIGS[name]
    return tuple(int(x) for x in name.split("-")[1])      # "custom-1011"


def homes():
    return {int(k): v for k, v in json.loads((BASE / "home_team.json").read_text()).items()}


def c2v2_params():
    R = json.loads((BASE / "compare" / "c2v2.json").read_text(encoding="utf-8"))
    return {int(k): v["params"] for k, v in R["choice"].items()}


# ---------------------------------------------------------------- features

def job_features(args):
    mid, config, home, c2p = args
    warnings.simplefilter("ignore")
    smooth, gapfill, gapfeat, bfp = cfg_of(config)
    z = dict(np.load(INP / f"{mid}.npz"))
    e = dict(np.load(EVS / f"{mid}.npz"))
    t, per, fps = z["t"], z["period"], float(z["fps"])
    ts = np.where(z["team_id"] == home, 1.0, -1.0).astype(np.float32)
    px, py = (smooth_players(z["px"], z["py"], per) if smooth else (z["px"], z["py"]))
    bx, by, bz, bvx, bvy, fl = z["bx"], z["by"], z["bz"], z["bvx"], z["bvy"], z["bflag"]
    real_present = fl != FLAG_MISSING
    if gapfill:
        bx, by, bz, bvx, bvy, fl, _ = fill_medium_gaps(t, per, bx, by, bz, bvx, bvy, fl)
    c2ball = SmoothedBall(bx, by, bz, bvx, bvy, np.zeros(len(t)), np.hypot(bvx, bvy), fl)
    fx, fy, fz, fvx, fvy, ffl = (np.array(a) for a in (bx, by, bz, bvx, bvy, fl))
    if bfp:
        inf = dict(np.load(BFP / f"{mid}.npz"))
        m = (fl == FLAG_MISSING) & np.isfinite(inf["x_inf"])
        fx[m], fy[m], fz[m], fvx[m], fvy[m], ffl[m] = inf["x_inf"][m], inf["y_inf"][m], 0.0, np.nan, np.nan, FLAG_INFERRED
    gapf = gap_features(t, per, fl == FLAG_MISSING, bx, by, px, py, ts) if gapfeat else None
    P = C3.Prepared(t, per, fx, fy, fz, fvx, fvy, ffl, px, py, ts, z["pitch"], real_present)
    rows = np.flatnonzero((np.arange(len(t)) % C3.STEP == 0) & (per > 0))
    Xb, nb, cand, Cb, cnb = C3.base_features(P, rows, gapf)
    out = dict(rows=rows, Xb=Xb, cand=cand.astype(np.int8), Cb=Cb)
    key = f"s{smooth}g{gapfill}"
    meta = dict(nb=nb, cnb=cnb)
    for k, prm in c2p.items():
        cp = OUT / "c2cache" / key / f"{mid}_f{k}.pkl"
        if cp.exists():
            r = pickle.loads(cp.read_bytes())
        else:
            r = run_c2(t, per, c2ball, px, py, z["player_id"], z["team_id"], z["pitch"], C2Params(**prm))
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_bytes(pickle.dumps(r))
        Xc, nc, Cc, cnc = C3.c2_features(P, rows, r, z["player_id"], cand)
        out[f"Xc{k}"], out[f"Cc{k}"] = Xc, Cc
        out[f"tf{k}"] = np.array([x["frame"] for x in r.touches], int)
        out[f"tc{k}"] = np.array([x["col"] for x in r.touches], int)
        meta.update(nc=nc, cnc=cnc)
    st, lp = e["state"][rows], e["player_id"][rows]
    rn = e["rule_names"][e["rule"][rows]]
    reason = np.where(st == 0, [REASON.get(x, 4) for x in rn], 0)
    pidc = z["player_id"][cand.astype(int)]
    out.update(y_state=st, y_reason=reason, y_scored=(e["score_mask"][rows] & (st <= 3)), y_cand=(pidc == lp[:, None]))
    d = OUT / "feat" / config
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"{mid}.npz", **out)
    return mid, meta


def run_features(config, workers):
    hm, c2p = homes(), c2v2_params()
    t0 = time.time()
    with multiprocessing.Pool(workers, maxtasksperchild=1) as pool:
        metas = dict(pool.map(job_features, [(m, config, hm[m], c2p) for m in LEARNING_MATCHES]))
    meta = next(iter(metas.values()))
    (OUT / "feat" / config / "meta.json").write_text(json.dumps(meta))
    print(f"features {config}: {len(meta['nb'])} base + {len(meta['nc'])} c2 + candidates {len(meta['cnb'])}+{len(meta['cnc'])} "
          f"({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- data access

def load_feat(config, mid):
    return np.load(OUT / "feat" / config / f"{mid}.npz")


def table(config, mids, k, every, layer):
    """Stack training rows (every n-th 10 Hz row, scored) for outer-fold parameters k."""
    Xs, ys, ws = [], [], []
    for m in mids:
        F = load_feat(config, m)
        sel = np.flatnonzero(F["y_scored"])
        sel = sel[sel % every == 0] if every > 1 else sel
        if layer == 1:
            Xs.append(np.hstack([F["Xb"][sel], F[f"Xc{k}"][sel]])); ys.append(F["y_reason"][sel])
        elif layer == 2:
            sel = sel[F["y_state"][sel] > 0]
            Xs.append(np.hstack([F["Xb"][sel], F[f"Xc{k}"][sel]])); ys.append(F["y_state"][sel] - 1)
        else:
            sel = sel[(F["y_state"][sel] == 1) & F["y_cand"][sel].any(1)]
            C = np.concatenate([F["Cb"][sel], F[f"Cc{k}"][sel]], 2)
            Xs.append(C.reshape(-1, C.shape[2])); ys.append(F["y_cand"][sel].reshape(-1))
    return np.concatenate(Xs), np.concatenate(ys)


def lgb_params(p, n_class=None):
    q = {k: v for k, v in p.items() if k != "loose_weight"}
    q.update(verbose=-1, n_jobs=N_JOBS)
    return q


def train_models(config, mids, k, p, every=TRAIN_EVERY):
    X1, y1 = table(config, mids, k, every, 1)
    m1 = lgb.LGBMClassifier(**lgb_params(p)).fit(X1, y1)
    del X1
    X2, y2 = table(config, mids, k, every, 2)
    w2 = np.where(y2 == 2, p["loose_weight"], 1.0)
    m2 = lgb.LGBMClassifier(**lgb_params(p)).fit(X2, y2, sample_weight=w2)
    del X2
    X3, y3 = table(config, mids, k, every, 3)
    m3 = lgb.LGBMClassifier(**lgb_params(dict(p, n_estimators=min(p["n_estimators"], 200)))).fit(X3, y3)
    return m1, m2, m3


def predict(config, mid, k, models):
    """Row-level probabilities for one match with outer-fold parameters k."""
    m1, m2, m3 = models
    F = load_feat(config, mid)
    X = np.hstack([F["Xb"], F[f"Xc{k}"]])
    pr = m1.predict_proba(X)
    cls = list(m1.classes_)
    p_dead = 1 - pr[:, cls.index(0)]
    p_state = m2.predict_proba(X)
    C = np.concatenate([F["Cb"], F[f"Cc{k}"]], 2)
    pc = m3.predict_proba(C.reshape(-1, C.shape[2]))[:, 1].reshape(C.shape[:2])
    ts = F["Cb"][:, :, list(json.loads((OUT / "feat" / config / "meta.json").read_text())["cnb"]).index("c_team")]
    tot = pc.sum(1)
    q_home = np.where(tot > 0, (pc * (ts > 0)).sum(1) / np.maximum(tot, 1e-9), 0.5)
    return dict(rows=F["rows"], p_dead=p_dead, p_state=p_state, q_home=q_home, pc=pc, cand=F["cand"].astype(int), ts=ts,
                tf=F[f"tf{k}"], tc=F[f"tc{k}"], y_state=F["y_state"], y_scored=F["y_scored"])


def decode(pred, period_rows, fps, hmm):
    logE = C3.emissions(pred["p_dead"], pred["p_state"], pred["q_home"])
    ok = C3.touch_mask(pred["tf"], pred["rows"], fps, hmm["tol_s"])
    return C3.viterbi(logE, ok, period_rows, hmm)


ROW_STATE = np.array([0, 1, 1, 2, 3])


def finish_and_score(mid, pred, path, hmm, home, save_dir=None):
    z = np.load(INP / f"{mid}.npz")
    e = dict(np.load(EVS / f"{mid}.npz"))
    fps = float(z["fps"])
    touches = [dict(frame=int(f), col=int(c)) for f, c in zip(pred["tf"], pred["tc"])]
    state, player, team, passes = C3.finish(path, pred["rows"], len(z["t"]), z["period"], touches, z["team_id"],
                                            np.where(z["team_id"] == home, 1, -1), z["player_id"], pred["cand"], pred["pc"],
                                            pred["ts"], fps, hmm["tol_s"])
    state[z["period"] <= 0] = 0
    s = full_score(e, state, player, team, passes, fps)
    s["had"] = score_had(e, state, team, home)
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_dir / f"{mid}.npz", state=state, player=player, team=team)
    return s


def _finish_job(args):
    warnings.simplefilter("ignore")
    mid, pred, path, hmm, home, save_dir = args
    return mid, finish_and_score(mid, pred, path, hmm, home, save_dir)


def _decode_job(args):
    warnings.simplefilter("ignore")
    mid, pred, hmm, fps = args
    z = np.load(INP / f"{mid}.npz")
    path = decode(pred, z["period"][pred["rows"]], fps, hmm)
    sc = pred["y_scored"]
    return mid, int(np.sum(ROW_STATE[path][sc] == pred["y_state"][sc])), int(sc.sum())


def importance(models, names_x, names_c):
    m1, m2, m3 = models
    g = {}
    for m, nm in ((m1, names_x), (m2, names_x), (m3, names_c)):
        v = m.booster_.feature_importance("gain")
        v = v / max(v.sum(), 1e-9)
        for a, b in zip(nm, v):
            g[a] = g.get(a, 0.0) + float(b)
    return g


# ---------------------------------------------------------------- ablation (fixed settings)

def run_ablate(config, workers):
    hm = homes()
    meta = json.loads((OUT / "feat" / config / "meta.json").read_text())
    names_x, names_c = meta["nb"] + meta["nc"], meta["cnb"] + meta["cnc"]
    mids = list(LEARNING_MATCHES)
    t0 = time.time()
    res, imp = {}, {}
    for k, fold in enumerate(FOLDS):
        train = [m for m in mids if m not in fold]
        models = train_models(config, train, k, FIXED)
        for a, b in importance(models, names_x, names_c).items():
            imp[a] = imp.get(a, 0.0) + b / 5
        preds = {m: predict(config, m, k, models) for m in fold}
        jobs = []
        for m in fold:
            z = np.load(INP / f"{m}.npz")
            path = decode(preds[m], z["period"][preds[m]["rows"]], float(z["fps"]), C3.HMM_DEFAULT)
            jobs.append((m, preds[m], path, C3.HMM_DEFAULT, hm[m], OUT / "pred" / config))
        with multiprocessing.Pool(min(workers, len(jobs))) as pool:
            for m, s in pool.map(_finish_job, jobs):
                res[str(m)] = s
        print(f"  {config} fold {k} done ({time.time() - t0:.0f}s)", flush=True)
    (OUT / f"ablate_{config}.json").write_text(json.dumps(dict(res=res, importance=imp, params=FIXED, hmm=C3.HMM_DEFAULT)))
    from m1_labels.scoring import merge
    mm = merge(list(res.values()))
    print(f"ablate {config}: frame acc {mm['accuracy']:.4f}, player {mm['player_acc']:.3f}, team {mm['team_acc']:.3f}, "
          f"poss-change F1 {mm['possession_change']['f1']:.3f} ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- final (nested Optuna)

def inner_acc(config, train, k, p, every):
    """Inner 4-fold CV (folds of the training matches) frame accuracy of the layer outputs (argmax, no decoder)."""
    inner = [[m for m in f if m in train] for f in FOLDS if any(m in train for m in f)]
    hit = tot = 0
    for j, val in enumerate(inner):
        tr = [m for m in train if m not in val]
        X1, y1 = table(config, tr, k, every, 1)
        m1 = lgb.LGBMClassifier(**lgb_params(p)).fit(X1, y1)
        del X1
        X2, y2 = table(config, tr, k, every, 2)
        m2 = lgb.LGBMClassifier(**lgb_params(p)).fit(X2, y2, sample_weight=np.where(y2 == 2, p["loose_weight"], 1.0))
        del X2
        for m in val:
            F = load_feat(config, m)
            sel = np.flatnonzero(F["y_scored"])
            sel = sel[sel % every == 0]
            X = np.hstack([F["Xb"][sel], F[f"Xc{k}"][sel]])
            pr = m1.predict_proba(X)
            pdd = 1 - pr[:, list(m1.classes_).index(0)]
            ps = m2.predict_proba(X)
            sc = np.c_[pdd, (1 - pdd)[:, None] * ps]
            hit += int(np.sum(sc.argmax(1) == F["y_state"][sel])); tot += len(sel)
    return hit / max(tot, 1)


def run_final(config, workers, lgb_trials, hmm_trials):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    hm = homes()
    meta = json.loads((OUT / "feat" / config / "meta.json").read_text())
    names_x, names_c = meta["nb"] + meta["nc"], meta["cnb"] + meta["cnc"]
    mids = list(LEARNING_MATCHES)
    t0 = time.time()
    res, imp, choice = {}, {}, {}
    for k, fold in enumerate(FOLDS):
        train = [m for m in mids if m not in fold]

        def obj(trial):
            p = dict(num_leaves=trial.suggest_int("num_leaves", 15, 127, log=True),
                     learning_rate=trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
                     n_estimators=trial.suggest_int("n_estimators", 100, 400),
                     min_child_samples=trial.suggest_int("min_child_samples", 20, 200, log=True),
                     colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                     subsample=0.7, subsample_freq=1,
                     reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
                     loose_weight=trial.suggest_float("loose_weight", 1.0, 5.0))
            return inner_acc(config, train, k, p, every=10)
        st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=300 + k))
        st.enqueue_trial({x: FIXED[x] for x in ("num_leaves", "learning_rate", "n_estimators", "min_child_samples",
                                                  "colsample_bytree", "reg_lambda", "loose_weight")})
        st.optimize(obj, n_trials=lgb_trials)
        p = dict(FIXED, **st.best_params)
        print(f"  fold {k}: LightGBM inner acc {st.trials[0].value:.4f} (fixed) -> {st.best_value:.4f} ({time.time() - t0:.0f}s)",
              flush=True)
        # decoder settings on inner out-of-fold predictions
        inner = [[m for m in f if m in train] for f in FOLDS if any(m in train for m in f)]
        oof = {}
        for val in inner:
            mods = train_models(config, [m for m in train if m not in val], k, p)
            for m in val:
                oof[m] = predict(config, m, k, mods)
        fpsd = {m: float(np.load(INP / f"{m}.npz")["fps"]) for m in train}

        def hobj(trial):
            h = {x: trial.suggest_float(x, *C3.HMM_RANGES[x]) for x in C3.HMM_RANGES}
            with multiprocessing.Pool(workers) as pool:
                out = pool.map(_decode_job, [(m, oof[m], h, fpsd[m]) for m in train])
            return sum(o[1] for o in out) / max(sum(o[2] for o in out), 1)
        hs = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=400 + k))
        hs.enqueue_trial(C3.HMM_DEFAULT)
        hs.optimize(hobj, n_trials=hmm_trials)
        hmm = dict(hs.best_params)
        print(f"  fold {k}: decoder inner row acc {hs.trials[0].value:.4f} (default) -> {hs.best_value:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        del oof
        models = train_models(config, train, k, p)
        for a, b in importance(models, names_x, names_c).items():
            imp[a] = imp.get(a, 0.0) + b / 5
        jobs = []
        for m in fold:
            pr = predict(config, m, k, models)
            z = np.load(INP / f"{m}.npz")
            jobs.append((m, pr, decode(pr, z["period"][pr["rows"]], float(z["fps"]), hmm), hmm, hm[m], OUT / "pred" / f"final_{config}"))
        with multiprocessing.Pool(min(workers, len(jobs))) as pool:
            for m, s in pool.map(_finish_job, jobs):
                res[str(m)] = s
        choice[k] = dict(lgb=p, hmm=hmm, lgb_inner_acc=st.best_value, hmm_inner_acc=hs.best_value)
        print(f"  fold {k} done ({time.time() - t0:.0f}s)", flush=True)
    (OUT / f"final_{config}.json").write_text(json.dumps(dict(res=res, importance=imp, choice=choice), default=str))
    from m1_labels.scoring import merge
    mm = merge(list(res.values()))
    print(f"final {config}: frame acc {mm['accuracy']:.4f} ({time.time() - t0:.0f}s)", flush=True)


def main() -> int:
    from m1_possession.config import use_scratch
    use_scratch()                                          # temp files to M1_TMP, default D:/m1_tmp (config.py)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("features", "ablate", "final", "clean"))
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--lgb-trials", type=int, default=8)
    ap.add_argument("--hmm-trials", type=int, default=25)
    a = ap.parse_args()
    if a.stage == "features":
        run_features(a.config, a.workers)
    elif a.stage == "ablate":
        run_ablate(a.config, a.workers)
    elif a.stage == "final":
        run_final(a.config, a.workers, a.lgb_trials, a.hmm_trials)
    else:
        shutil.rmtree(OUT / "feat" / a.config, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
