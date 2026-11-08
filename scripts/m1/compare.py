"""M1 step 1 follow-up (Parts 3-4): all contestants on the 30 learning matches, cross-validated by match (5 folds).

Contestants: C0; C1-published; C1-tuned (grid on its own parameters, chosen per fold on the other folds); C1 + oracle ball-in-
play (NOT a contestant: uses the answer key's DEAD as the ball-alive flag); C2-start (T1 start values); C2-tuned (group-wise
search per fold on the other folds: dead -> touch -> control). Never runs on the test or stress matches.

Inputs come from data/processed/m1_possession/inputs (tag-free); scoring reads .../evalset (labels + PFF passes).

    python scripts/m1/compare.py --workers 2 [--trials 16]
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from m1_labels.scoring import STATES4, merge, merge_passes, score, score_passes
from m1_possession import SmoothedBall
from m1_possession.baselines import c0_naive, c1_finish, c1_prepare
from m1_possession.rules import GROUPS, RANGES, C2Params, run_c2
from m1_possession.splits import DEV_MATCHES, FOLDS, LEARNING_MATCHES

INP = Path("data/processed/m1_possession/inputs")
EVS = Path("data/processed/m1_possession/evalset")
OUT = Path("data/processed/m1_possession/compare")
C1_GRID = dict(zone=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0), angle=(5.0, 10.0, 15.0, 20.0, 25.0), speed=(1.0, 3.0, 5.0),
               spatial=(0.05, 0.1, 0.2))
C1_PUBLISHED = (1.0, 10.0, 5.0, 0.1)


def load(mid):
    z = dict(np.load(INP / f"{mid}.npz"))
    e = dict(np.load(EVS / f"{mid}.npz"))
    b = SmoothedBall(z["bx"], z["by"], z["bz"], z["bvx"], z["bvy"], np.zeros(len(z["t"])), np.hypot(z["bvx"], z["bvy"]), z["bflag"])
    return z, e, b


def full_score(e, state, player, team, passes, fps, keep_pred=False):
    s = score(e, state, player, team, fps, rule_names=e["rule_names"])
    if passes is not None:
        s["passes"] = score_passes([p[0] for p in passes], [p[1] for p in passes], e["pass_frame"], e["pass_player"], fps)
    if keep_pred:
        s["pred"] = (state, player, team)
    return s


def lite(e, state, player, team, fps):
    s = score(e, state, player, team, fps)
    return dict(confusion=s["confusion"], control_frames=s["control_frames"], player_acc=s["player_acc"], team_acc=s["team_acc"],
                possession_change=s["possession_change"])


def acc_of(scores):
    conf = np.sum([np.asarray(s["confusion"]) for s in scores], axis=0)
    return float(np.trace(conf) / max(conf.sum(), 1))


# ---------------------------------------------------------------- workers

def job_fixed(mid):
    """C0, C1-published, C1-oracle and C2-start for one match (full scoring)."""
    warnings.simplefilter("ignore")
    z, e, b = load(mid)
    fps = float(z["fps"])
    out = {}
    r = c0_naive(z["t"], z["period"], b, z["px"], z["py"], z["player_id"], z["team_id"])
    out["C0"] = full_score(e, r.state, r.player, r.team, None, fps)
    prep = c1_prepare(z["t"], z["period"], b, z["px"], z["py"], z["player_id"], z["team_id"], *z["pitch"], zone_m=1.0, spatials=(0.1,))
    r = c1_finish(prep, 10.0, 5.0, 0.1)
    out["C1-published"] = full_score(e, r.state, r.player, r.team, r.passes, fps)
    alive = e["state"] != 0
    r = c1_finish(prep, 10.0, 5.0, 0.1, alive=alive)
    out["C1+oracle (not a contestant)"] = full_score(e, r.state, r.player, r.team, r.passes, fps)
    r = run_c2(z["t"], z["period"], b, z["px"], z["py"], z["player_id"], z["team_id"], z["pitch"], C2Params())
    out["C2-start"] = full_score(e, r.state, r.player, r.team, r.passes, fps)
    return mid, out


def job_c1_grid(mid):
    warnings.simplefilter("ignore")
    z, e, b = load(mid)
    fps = float(z["fps"])
    res = {}
    for zone in C1_GRID["zone"]:
        prep = c1_prepare(z["t"], z["period"], b, z["px"], z["py"], z["player_id"], z["team_id"], *z["pitch"], zone_m=zone,
                          spatials=C1_GRID["spatial"])
        for ang, sp, spa in itertools.product(C1_GRID["angle"], C1_GRID["speed"], C1_GRID["spatial"]):
            r = c1_finish(prep, ang, sp, spa)
            res[(zone, ang, sp, spa)] = lite(e, r.state, r.player, r.team, fps)
    return mid, res


def job_c1_full(args):
    mid, combo = args
    warnings.simplefilter("ignore")
    z, e, b = load(mid)
    zone, ang, sp, spa = combo
    prep = c1_prepare(z["t"], z["period"], b, z["px"], z["py"], z["player_id"], z["team_id"], *z["pitch"], zone_m=zone, spatials=(spa,))
    r = c1_finish(prep, ang, sp, spa)
    return mid, full_score(e, r.state, r.player, r.team, r.passes, float(z["fps"]))


def job_c2(args):
    mid, params, full = args
    warnings.simplefilter("ignore")
    z, e, b = load(mid)
    r = run_c2(z["t"], z["period"], b, z["px"], z["py"], z["player_id"], z["team_id"], z["pitch"], C2Params(**params))
    fps = float(z["fps"])
    if full:
        s = full_score(e, r.state, r.player, r.team, r.passes, fps)
        s["rules"] = r.rule_names
        np.savez_compressed(OUT / "c2_pred" / f"{mid}.npz", state=r.state, player=r.player, team=r.team, rule=r.rule,
                            rule_names=np.array(r.rule_names))
        return mid, s
    return mid, lite(e, r.state, r.player, r.team, fps)


# ---------------------------------------------------------------- search

def perturb(rng, best: dict, group, scale=0.15):
    p = dict(best)
    if rng.random() < 0.3:                                    # random restart inside the ranges
        for k in GROUPS[group]:
            lo, hi = RANGES[k]
            p[k] = float(rng.uniform(lo, hi))
        return p
    keys = [k for k in GROUPS[group] if rng.random() < 0.5] or [rng.choice(GROUPS[group])]
    for k in keys:
        lo, hi = RANGES[k]
        p[k] = float(np.clip(best[k] + rng.normal(0, scale * (hi - lo)), lo, hi))
    return p


def tune_c2(pool, train, trials, seed, log):
    """Group-wise local search (dead -> touch -> control) on the training matches; objective frame accuracy."""
    rng = np.random.default_rng(seed)
    best = C2Params().as_dict()
    best_score = acc_of([s for _, s in pool.map(job_c2, [(m, best, False) for m in train])])
    for group in ("dead", "touch", "control"):
        for k in range(trials):
            cand = perturb(rng, best, group)
            sc = [s for _, s in pool.map(job_c2, [(m, cand, False) for m in train])]
            a = acc_of(sc)
            log.append(dict(group=group, trial=k, acc=a))
            if a > best_score:
                best, best_score = cand, a
        print(f"    group {group}: best train accuracy {best_score:.4f}", flush=True)
    return best, best_score


def main() -> int:
    from m1_possession.config import use_scratch
    use_scratch()                                          # temp files to M1_TMP, default D:/m1_tmp (config.py)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--trials", type=int, default=16)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "c2_pred").mkdir(exist_ok=True)
    mids = list(LEARNING_MATCHES)
    t0 = time.time()
    res = defaultdict(dict)                         # contestant -> match -> full score
    with multiprocessing.Pool(a.workers, maxtasksperchild=4) as pool:
        for mid, out in pool.imap_unordered(job_fixed, mids):
            for k, v in out.items():
                res[k][mid] = v
        print(f"fixed contestants done ({time.time() - t0:.0f}s)", flush=True)
        grid = {}
        for mid, g in pool.imap_unordered(job_c1_grid, mids):
            grid[mid] = g
        print(f"C1 grid done ({time.time() - t0:.0f}s)", flush=True)
        combos = list(next(iter(grid.values())).keys())
        c1_choice = {}
        for k, fold in enumerate(FOLDS):
            train = [m for m in mids if m not in fold]
            best = max(combos, key=lambda c: acc_of([grid[m][c] for m in train]))
            c1_choice[k] = best
            for mid, s in pool.map(job_c1_full, [(m, best) for m in fold]):
                res["C1-tuned"][mid] = s
        c1_all = max(combos, key=lambda c: acc_of([grid[m][c] for m in mids]))
        print(f"C1-tuned done ({time.time() - t0:.0f}s): per fold {c1_choice}; all learning {c1_all}", flush=True)
        c2_choice, log = {}, []
        for k, fold in enumerate(FOLDS):
            train = [m for m in mids if m not in fold]
            print(f"  C2 fold {k}: tuning on {len(train)} matches", flush=True)
            best, bs = tune_c2(pool, train, a.trials, seed=100 + k, log=log)
            c2_choice[k] = dict(params=best, train_acc=bs)
            for mid, s in pool.map(job_c2, [(m, best, True) for m in fold]):
                res["C2-tuned"][mid] = s
            print(f"  C2 fold {k} done ({time.time() - t0:.0f}s)", flush=True)
        print("  C2 final tuning on all 30 learning matches (for the reported threshold table)", flush=True)
        c2_all, c2_all_acc = tune_c2(pool, mids, a.trials, seed=999, log=log)
    json.dump(dict(res={k: {str(m): {kk: vv for kk, vv in s.items() if kk != "pred"} for m, s in v.items()} for k, v in res.items()},
                   c1_choice={str(k): v for k, v in c1_choice.items()}, c1_all=c1_all,
                   c1_grid_all={str(c): acc_of([grid[m][c] for m in mids]) for c in combos},
                   c2_choice={str(k): v for k, v in c2_choice.items()}, c2_all=c2_all, c2_all_acc=c2_all_acc, log=log),
              open(OUT / "results.json", "w", encoding="utf-8"), default=str)
    print(f"all done in {time.time() - t0:.0f}s -> {OUT / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
