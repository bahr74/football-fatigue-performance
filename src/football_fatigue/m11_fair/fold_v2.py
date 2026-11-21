"""Fair-benchmark fold runner v2: training distributions, successive halving (or full search), resumable neural training,
write-once checkpoints with integrity sidecars, calibration on separate matches, and predictions on every stress-test table.

Roles per outer fold (match-disjoint): fit -> parameters; early_stop -> epochs, HGB iterations, configuration choice (centred g0p80
rows only); calibration -> temperature (centred g0p80 rows only); test -> reported only. Nothing about the test fold is used for any
decision. ``run_fold_v2`` is resumable: per-configuration training states and finished seed results are reused when present.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from football_fatigue.m11_benchmark import metrics as MT

from . import halving as HV
from . import spaces as SP
from .distributions import Pool, TrainDist
from .evaluation import EvalSets
from .nets import NetConfig
from .train import FairNeural, FairTabular, fit_temperature, save_checkpoint

NEURAL = ("gru", "lstm")


def net_config(family: str, cfg: dict[str, Any]) -> NetConfig:
    return NetConfig(family, int(cfg["hidden"]), int(cfg.get("emb", 48)), int(cfg["layers"]), int(cfg.get("head", 96)), float(cfg["dropout"]),
                     float(cfg.get("rnn_dropout", 0.0)))


def neural(family: str, cfg: dict[str, Any], seed: int, device, epochs: int | None = None) -> FairNeural:
    return FairNeural(net_config(family, cfg), lr=cfg["lr"], weight_decay=cfg["weight_decay"], batch=int(cfg["batch"]),
                      epochs=int(epochs or cfg["epochs"]), patience=int(cfg["patience"]), clip=float(cfg.get("clip", 1.0)), seed=seed, device=device)


def primary_rows(pool: Pool, matches) -> np.ndarray:
    return np.flatnonzero(np.isin(pool.groups[:pool.n0], list(matches)))


def _jsonable(o):
    return o.item() if hasattr(o, "item") else str(o)


def run_fold_v2(family: str, fold: str, part: dict[str, list[str]], pool: Pool, dist: TrainDist, evals: EvalSets,
                configs: list[dict[str, Any]], seeds, out: Path, *, search: str = "halving", rungs=HV.DEFAULT_RUNGS, eta: int = HV.ETA,
                epoch_cap: int | None = None, device=None, log: Callable[[str], None] = print) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    done = out / f"fold{fold}.json"
    if done.exists():
        return json.loads(done.read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    fit = pool.eligible(primary_rows(pool, part["fit"]), dist)
    es, cal = primary_rows(pool, part["early_stop"]), primary_rows(pool, part["calibration"])
    if not (set(part["fit"]).isdisjoint(part["early_stop"]) and set(part["fit"]).isdisjoint(part["calibration"])
            and set(part["test"]).isdisjoint(set(part["fit"]) | set(part["early_stop"]) | set(part["calibration"]))):
        raise RuntimeError("fold roles overlap")
    perturb = (dist.noise_sigma_m, dist.drop_rates) if dist.online else None
    res: dict[str, Any] = dict(family=family, fold=fold, distribution=dist.as_dict(), search=search, n_fit_encounters=int(len(fit)),
                               roles={k: sorted(v, key=int) for k, v in part.items()})
    if family in NEURAL:
        def draw(rng):
            return pool.draw(fit, dist, rng)

        def train(cfg, stop, state, seed=0):
            m = neural(family, cfg, seed, device, epoch_cap)
            return m.fit(pool.si, pool.so, pool.gap, pool.y, pool.si[es], pool.so[es], pool.gap[es], pool.y[es], draw=draw, perturb=perturb,
                         state_path=str(state), stop_at_epoch=stop)
        cap = int(epoch_cap or configs[0]["epochs"])
        rr = tuple(min(r, cap) for r in rungs if r < cap) + (cap,)
        if search == "halving" and len(configs) > 1:
            sh = HV.successive_halving(configs, train, out / f"fold{fold}_search", rr, eta, log)
        else:
            sh = HV.successive_halving(configs, train, out / f"fold{fold}_search", (cap,), 1, log)
        res["search_result"] = sh
        cfg = next(c for c in configs if c["config_id"] == sh["selected"])
    else:
        allowed = np.isin(pool.variant_table, list(dist.variants)) if len(pool.owner) else np.zeros(0, bool)
        var_rows = pool.n0 + np.flatnonzero(allowed & np.isin(pool.owner, fit)) if len(pool.owner) else np.zeros(0, int)
        rows = np.r_[fit, var_rows].astype(int)
        own = np.r_[fit, pool.owner[var_rows - pool.n0]] if len(var_rows) else fit
        w = 1.0 / np.bincount(own, minlength=pool.n0)[own]                      # every encounter weighs 1 in total
        trials = []
        for c in configs:
            t1 = time.perf_counter()
            try:
                m = FairTabular(family, c, pool.k_perm, 0).fit(pool.F[rows], pool.y[rows], pool.F[es], pool.y[es], sample_weight=w)
                from .runner import auc
                trials.append(dict(config_id=c["config_id"], es_score=round(auc(pool.y[es], m.score(pool.F[es])), 6), diagnostics=m.diagnostics,
                                   s=round(time.perf_counter() - t1, 1)))
            except Exception as e:
                trials.append(dict(config_id=c["config_id"], error=repr(e)[:300], s=round(time.perf_counter() - t1, 1)))
            log(f"{family} fold {fold} {c['config_id']}: {trials[-1].get('es_score', trials[-1].get('error'))}")
        ok = [t for t in trials if "es_score" in t]
        if not ok:
            raise RuntimeError("every configuration failed")
        res["trials"] = trials
        cfg = next(c for c in configs if c["config_id"] == max(ok, key=lambda t: t["es_score"])["config_id"])
    res["selected"], res["selected_edge_flags"] = cfg, SP.edge_flags(family, cfg)
    per_seed = {}
    for s in seeds:
        sf = out / f"fold{fold}_seed{s}.json"
        if sf.exists():
            per_seed[str(s)] = json.loads(sf.read_text(encoding="utf-8"))
            continue
        t1 = time.perf_counter()
        preds = {}
        if family in NEURAL:
            reuse = out / f"fold{fold}_search" / f"{cfg['config_id']}.state.pt"     # the search already trained seed 0 to the end
            m = train(cfg, None, reuse if s == 0 and reuse.exists() else out / f"fold{fold}_seed{s}.state.pt", seed=s)
            zc = m.logits(pool.si[cal], pool.so[cal], pool.gap[cal])
            score = lambda t, r: m.logits(evals.sets[t]["si"][r], evals.sets[t]["so"][r], evals.sets[t]["gap"][r])  # noqa: E731
            ck = save_checkpoint(m.net, out / "checkpoints" / f"{family}_fold{fold}_seed{s}.pt",
                                 dict(family=family, fold=fold, seed=s, config=cfg, net=net_config(family, cfg).as_dict(), distribution=dist.name))
            diag = m.diagnostics
            history = m.history
        else:
            m = FairTabular(family, cfg, pool.k_perm, s).fit(pool.F[rows], pool.y[rows], pool.F[es], pool.y[es], sample_weight=w)
            zc = m.score(pool.F[cal])
            score = lambda t, r: m.score(evals.sets[t]["F"][r])  # noqa: E731
            ck, diag, history = None, m.diagnostics, None
        a = fit_temperature(zc, pool.y[cal])
        for t in evals.sets:
            r = evals.rows(t, part["test"])
            p = 1 / (1 + np.exp(-a * score(t, r)))
            preds[t] = dict(candidate_id=evals.sets[t]["candidate_id"][r].tolist(), p=np.round(p, 6).tolist(),
                            basic=MT.basic(evals.sets[t]["y"][r], p))
        per_seed[str(s)] = dict(scale=a, calibration=MT.basic(pool.y[cal], 1 / (1 + np.exp(-a * zc))), diagnostics=diag, history=history,
                                checkpoint=ck, predictions=preds, seconds=round(time.perf_counter() - t1, 1))
        sf.write_text(json.dumps(per_seed[str(s)], default=_jsonable), encoding="utf-8")
        log(f"{family} fold {fold} seed {s}: test BA {preds['g0p80']['basic']['balanced_accuracy']} ({per_seed[str(s)]['seconds']} s)")
    res["per_seed"] = {k: {kk: vv for kk, vv in v.items() if kk != "predictions"} for k, v in per_seed.items()}
    res["runtime_s"] = round(time.perf_counter() - t0, 1)
    done.write_text(json.dumps(res, default=_jsonable), encoding="utf-8")
    return res
