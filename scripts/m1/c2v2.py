"""M1 C2 v2: tune only the new v2 parameters (Optuna TPE) on the training folds, starting from each fold's C2-tuned values;
score the held-out fold (cross-validated by match on the 30 learning matches). Never runs on the test or stress matches.

Writes compare/c2v2.json (per-fold choice, final all-30 choice, full scores) and compare/c2v2_pred/<match>.npz.

    python scripts/m1/c2v2.py --workers 3 --trials 20
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import optuna

sys.path.insert(0, str(Path(__file__).parent))
from compare import OUT, acc_of, full_score, lite, load  # noqa: E402
from m1_possession.rules import RANGES, C2Params, run_c2  # noqa: E402
from m1_possession.splits import FOLDS, LEARNING_MATCHES  # noqa: E402

NEW = ("restart_limit_s", "clear_margin_m", "gap_long_s", "gap_same_spot_m", "still_speed_v2")
RANGES_V2 = {**RANGES, "still_speed_v2": (1.0, 3.0)}


def job(args):
    mid, params, full = args
    warnings.simplefilter("ignore")
    z, e, b = load(mid)
    r = run_c2(z["t"], z["period"], b, z["px"], z["py"], z["player_id"], z["team_id"], z["pitch"], C2Params(**params))
    fps = float(z["fps"])
    if full:
        np.savez_compressed(OUT / "c2v2_pred" / f"{mid}.npz", state=r.state, player=r.player, team=r.team, rule=r.rule,
                            rule_names=np.array(r.rule_names))
        return mid, full_score(e, r.state, r.player, r.team, r.passes, fps)
    return mid, lite(e, r.state, r.player, r.team, fps)


def pc_f1(scores):
    tp = sum(s["possession_change"]["tp"] for s in scores)
    nt = sum(s["possession_change"]["n_true"] for s in scores)
    npred = sum(s["possession_change"]["n_pred"] for s in scores)
    p, r = tp / max(npred, 1), tp / max(nt, 1)
    return 2 * p * r / max(p + r, 1e-9)


def tune(pool, base, train, trials, seed, log, tag):
    def objective(trial):
        prm = dict(base, v2=True, **{k: trial.suggest_float(k, *RANGES_V2[k]) for k in NEW})
        sc = [s for _, s in pool.map(job, [(m, prm, False) for m in train])]
        a = acc_of(sc)
        trial.set_user_attr("pc_f1", pc_f1(sc))
        log.append(dict(tag=tag, trial=trial.number, acc=a, pc_f1=pc_f1(sc), **{k: prm[k] for k in NEW}))
        return a
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    start = C2Params()
    study.enqueue_trial({k: getattr(start, k) for k in NEW})          # trial 0 = the start values
    study.optimize(objective, n_trials=trials)
    best = dict(base, v2=True, **study.best_params)
    return best, study.best_value, study.best_trial.user_attrs["pc_f1"], study.trials[0].value


def main() -> int:
    from m1_possession.config import use_scratch
    use_scratch()                                          # temp files to M1_TMP, default D:/m1_tmp (config.py)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--trials", type=int, default=20)
    a = ap.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    (OUT / "c2v2_pred").mkdir(parents=True, exist_ok=True)
    R = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    mids = list(LEARNING_MATCHES)
    t0 = time.time()
    res, choice, log = {}, {}, []
    with multiprocessing.Pool(a.workers, maxtasksperchild=8) as pool:
        for k, fold in enumerate(FOLDS):
            train = [m for m in mids if m not in fold]
            base = dict(R["c2_choice"][str(k)]["params"])
            best, acc, pcf, acc0 = tune(pool, base, train, a.trials, 200 + k, log, f"fold{k}")
            choice[k] = dict(params=best, train_acc=acc, train_pc_f1=pcf, train_acc_start=acc0)
            for mid, s in pool.map(job, [(m, best, True) for m in fold]):
                res[str(mid)] = s
            print(f"fold {k}: train acc start {acc0:.4f} -> {acc:.4f} (pc F1 {pcf:.3f}); {[round(best[x], 2) for x in NEW]} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        # the reported table: the same search on all 30 learning matches, from the all-30 C2-tuned values
        best_all, acc_all, pcf_all, acc0_all = tune(pool, dict(R["c2_all"]), mids, a.trials, 999, log, "all")
        print(f"all 30: {acc0_all:.4f} -> {acc_all:.4f}; {[round(best_all[x], 2) for x in NEW]}", flush=True)
    json.dump(dict(res=res, choice={str(k): v for k, v in choice.items()}, all=dict(params=best_all, acc=acc_all, pc_f1=pcf_all,
                                                                                     acc_start=acc0_all), log=log),
              open(OUT / "c2v2.json", "w", encoding="utf-8"), default=str)
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
