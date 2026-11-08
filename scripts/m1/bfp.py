"""M1 ball-from-players (input upgrade d): LightGBM regressors that predict the ball's x, y from player movement plus the last
seen ball position and the time since. Cross-validated by match on the 30 learning matches (never test / stress).

1. samples: on frames where the ball IS observed, hide it for a random time h (log-uniform 0.1-30 s): the features use the
   ball seen h seconds earlier; the target is the true position now. Training and test samples use different seeds.
2. per fold: train on the other 4 folds' training samples, report the error on the held-out fold's test samples (median /
   p95 metres, by ball on the ground (z < 1 m) or in the air and by time missing), next to the "last seen position" baseline.
3. fill: every missing frame of a held-out match gets the fold model's estimate (flag "inferred"): bfp/<match>.npz.

    python scripts/m1/bfp.py --workers 3
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np

from m1_possession import ball_infer as BI
from m1_possession.smoother import FLAG_MISSING, FLAG_OBSERVED
from m1_possession.splits import FOLDS, LEARNING_MATCHES

INP = Path("data/processed/m1_possession/inputs")
OUT = Path("data/processed/m1_possession/bfp")
HOME = Path("data/processed/m1_possession/home_team.json")
H_BINS = ((0, 0.5), (0.5, 2), (2, 5), (5, 15), (15, 1e9))
PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=63, min_child_samples=50, colsample_bytree=0.8, subsample=0.8,
              subsample_freq=1, verbose=-1, n_jobs=8)
CHUNK = 8000


def home_team_ids() -> dict:
    if HOME.exists():
        return {int(k): v for k, v in json.loads(HOME.read_text()).items()}
    from football_fatigue.ingestion import Dataset         # metadata only (no tags)
    ds = Dataset("data/raw/FIFA World Cup 2022")
    out = {m: int(ds.metadata(str(m)).home_team_id) for m in LEARNING_MATCHES}
    HOME.write_text(json.dumps(out))
    return out


def prep(mid, home):
    z = dict(np.load(INP / f"{mid}.npz"))
    team_sign = np.where(z["team_id"] == home, 1.0, -1.0).astype(np.float32)
    s = BI.orient(z["period"], z["px"], team_sign)
    return z, team_sign, s


def feats(z, team_sign, s, frames, last):
    lf = np.maximum(last, 0)
    ok = last >= 0
    lx, ly = np.where(ok, z["bx"][lf], np.nan), np.where(ok, z["by"][lf], np.nan)
    lvx, lvy = np.where(ok, z["bvx"][lf], np.nan), np.where(ok, z["bvy"][lf], np.nan)
    out = []
    for c in range(0, len(frames), CHUNK):
        sl = slice(c, c + CHUNK)
        out.append(BI.features(z["t"], z["period"], z["px"], z["py"], team_sign, s, frames[sl], last[sl], lx[sl], ly[sl],
                               lvx[sl], lvy[sl]))
    return np.concatenate(out) if out else np.zeros((0, len(BI.NAMES)), np.float32), lx, ly


def job_samples(args):
    mid, home, seed, n_train, n_test = args
    warnings.simplefilter("ignore")
    z, team_sign, s = prep(mid, home)
    t, per, fl = z["t"], z["period"], z["bflag"]
    fps = float(z["fps"])
    present = fl != FLAG_MISSING
    obs = np.flatnonzero(fl == FLAG_OBSERVED)
    out = {}
    for name, nn, sd in (("train", n_train, seed), ("test", n_test, seed + 7919)):
        rng = np.random.default_rng(sd + mid)
        f = rng.choice(obs, size=min(nn * 2, len(obs)), replace=False)
        h = np.exp(rng.uniform(np.log(0.1), np.log(30.0), len(f)))
        last = f - np.round(h * fps).astype(int)
        ok = (last >= 0)
        last = np.maximum(last, 0)
        ok &= present[last] & (per[last] == per[f])
        f, last = f[ok][:nn], last[ok][:nn]
        X, lx, ly = feats(z, team_sign, s, f, last)
        sg = s[f]
        out[name] = dict(X=X, dx=((z["bx"][f] - lx) * sg).astype(np.float32), dy=(z["by"][f] - ly).astype(np.float32),
                         z=np.nan_to_num(z["bz"][f]).astype(np.float32), h=(t[f] - t[last]).astype(np.float32))
    out["names"] = list(BI.NAMES)
    return mid, out


def job_fill(args):
    mid, home, models = args
    warnings.simplefilter("ignore")
    z, team_sign, s = prep(mid, home)
    present = z["bflag"] != FLAG_MISSING
    last, _ = BI.last_seen(z["t"], z["period"], present)
    f = np.flatnonzero(~present)
    X, lx, ly = feats(z, team_sign, s, f, last[f])
    mx, my = (lgb.Booster(model_str=m) for m in models)
    dx, dy = mx.predict(X), my.predict(X)
    sg = s[f]
    lxo, lyo = np.where(np.isfinite(lx), lx * sg, 0.0), np.where(np.isfinite(ly), ly, 0.0)   # never seen: offset from centre
    x_inf = np.full(len(z["t"]), np.nan, np.float32)
    y_inf = np.full(len(z["t"]), np.nan, np.float32)
    x_inf[f] = (lxo + dx) * sg
    y_inf[f] = lyo + dy
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{mid}.npz", x_inf=x_inf, y_inf=y_inf)
    return mid, len(f)


def err_table(err, base, zz, hh):
    rows = []
    for air in (False, True):
        for lo, hi in H_BINS:
            sel = ((zz >= 1.0) == air) & (hh > lo) & (hh <= hi)
            if sel.sum() < 20:
                continue
            rows.append(dict(ball="air" if air else "ground", h=f"{lo:g}-{hi:g} s" if hi < 1e8 else f"> {lo:g} s", n=int(sel.sum()),
                             med=float(np.median(err[sel])), p95=float(np.percentile(err[sel], 95)),
                             base_med=float(np.median(base[sel])), base_p95=float(np.percentile(base[sel], 95))))
    for air in (False, True):
        sel = (zz >= 1.0) == air
        rows.append(dict(ball="air" if air else "ground", h="all", n=int(sel.sum()), med=float(np.median(err[sel])),
                         p95=float(np.percentile(err[sel], 95)), base_med=float(np.median(base[sel])),
                         base_p95=float(np.percentile(base[sel], 95))))
    rows.append(dict(ball="all", h="all", n=len(err), med=float(np.median(err)), p95=float(np.percentile(err, 95)),
                     base_med=float(np.median(base)), base_p95=float(np.percentile(base, 95))))
    return rows


def main() -> int:
    from m1_possession.config import use_scratch
    use_scratch()                                          # temp files to M1_TMP, default D:/m1_tmp (config.py)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-test", type=int, default=3000)
    a = ap.parse_args()
    t0 = time.time()
    homes = home_team_ids()
    mids = list(LEARNING_MATCHES)
    with multiprocessing.Pool(a.workers, maxtasksperchild=2) as pool:
        S = dict(pool.map(job_samples, [(m, homes[m], 11, a.n_train, a.n_test) for m in mids]))
    names = S[mids[0]]["names"]
    print(f"samples done ({time.time() - t0:.0f}s), {len(names)} features", flush=True)
    all_err, all_base, all_z, all_h, models, imp = [], [], [], [], {}, np.zeros(S[mids[0]]["train"]["X"].shape[1])
    for k, fold in enumerate(FOLDS):
        tr = [m for m in mids if m not in fold]
        X = np.concatenate([S[m]["train"]["X"] for m in tr])
        mods = []
        for tgt in ("dx", "dy"):
            y = np.concatenate([S[m]["train"][tgt] for m in tr])
            reg = lgb.LGBMRegressor(**PARAMS).fit(X, y)
            mods.append(reg.booster_.model_to_string())
            imp += reg.booster_.feature_importance("gain")
        models[k] = mods
        mx, my = (lgb.Booster(model_str=m) for m in mods)
        for m in fold:
            te = S[m]["test"]
            err = np.hypot(mx.predict(te["X"]) - te["dx"], my.predict(te["X"]) - te["dy"])
            all_err.append(err); all_base.append(np.hypot(te["dx"], te["dy"])); all_z.append(te["z"]); all_h.append(te["h"])
        print(f"fold {k} trained ({time.time() - t0:.0f}s)", flush=True)
    err, base, zz, hh = (np.concatenate(v) for v in (all_err, all_base, all_z, all_h))
    table = err_table(err, base, zz, hh)
    for r in table:
        print(r)
    with multiprocessing.Pool(a.workers, maxtasksperchild=2) as pool:
        jobs = [(m, homes[m], models[k]) for k, fold in enumerate(FOLDS) for m in fold]
        filled = dict(pool.map(job_fill, jobs))
    top = sorted(zip(names, imp.tolist()), key=lambda x: -x[1])[:15]
    json.dump(dict(table=table, filled=filled, top_features=top, params=PARAMS), open(OUT / "bfp_eval.json", "w"), indent=1)
    print(f"filled {sum(filled.values())} frames; done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
