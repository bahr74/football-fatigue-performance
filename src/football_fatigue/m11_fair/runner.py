"""Per-family, per-outer-fold runner shared by the CPU tabular pilot and the (gated) GPU training.

For each outer fold: every configuration is fitted on the *fit* matches and scored on the *early-stop* matches (neural: early-stop
log loss at the best epoch; tabular: early-stop AUC, scale-free); the best configuration is refitted per seed, its scale is fitted
on the *calibration* matches, and only then is the *test* fold predicted. Test predictions never influence any choice.

Simplification vs the full nested plan (documented): configuration selection uses the single early-stop split instead of 3 inner
folds (compute); calibration and test matches are never used for selection."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from football_fatigue.m11_benchmark import data as D
from football_fatigue.m11_benchmark import metrics as MT

from . import spaces as SP
from .nets import NetConfig
from .train import FairNeural, FairTabular, fit_temperature

NEURAL = ("gru", "lstm")


class DevData:
    """Row-aligned development data: labels, match ids, sequences and features (inputs only from observed tracklets)."""

    def __init__(self, y, groups, si=None, so=None, gap=None, F=None, k_perm=None, indep=None):
        self.y, self.groups, self.si, self.so, self.gap, self.F, self.k_perm = y, np.asarray(groups).astype(str), si, so, gap, F, k_perm
        self.indep = np.ones(len(y), bool) if indep is None else np.asarray(indep, bool)

    def rows(self, matches) -> np.ndarray:
        return np.isin(self.groups, list(matches))


def load_dev(table: str = "g0p80", sequences: bool = True, features: bool = True) -> DevData:
    """Through the Stage 3 adapter (refuses held-out matches, verifies hashes). Features: the existing read-only Stage 4 cache when it
    exists; otherwise computed in memory (never written into the historical folder)."""
    x, y, g = D.load_inputs(table)
    meta = D.load_meta(table)
    if not np.array_equal(meta["match_id"].astype(str).to_numpy(), np.asarray(g).astype(str)) or not np.array_equal(meta["y"].to_numpy(), y):
        raise SystemExit("metadata and inputs are not row-aligned")
    si = so = None
    if sequences:
        si, so = D.seq_tensor(x)
    F = k = None
    if features:
        p = D.out_dir() / "features" / f"{table}.npz"
        if p.exists():
            d = D.load_npz(p)
            if not np.array_equal(d["groups"].astype(str), np.asarray(g).astype(str)):
                raise SystemExit("feature cache is not row-aligned with the inputs")
            F, names = d["F"], [str(v) for v in d["names"]]
        else:
            F, _m, names = D.features(x)
        k = D.n_perm(names)
    return DevData(y.astype(int), g, si, so, x["gap_s"].astype(np.float32), F, k, meta["independent"].to_numpy())


def auc(y, s) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def run_fold(family: str, fold: str, part: dict[str, Any], data: DevData, configs: list[dict[str, Any]], seeds, *, device=None,
             ckpt_dir: Path | None = None, log: Callable[[str], None] = print) -> dict[str, Any]:
    fit, es, cal, test = (data.rows(part[k]) for k in ("fit", "early_stop", "calibration", "test"))
    t0 = time.perf_counter()
    trials = []
    for cfg in configs:
        t1 = time.perf_counter()
        try:
            if family in NEURAL:
                m = _neural(family, cfg, 0, device).fit(data.si[fit], data.so[fit], data.gap[fit], data.y[fit], data.si[es], data.so[es],
                                                         data.gap[es], data.y[es])
                pe = m.predict(data.si[es], data.so[es], data.gap[es])
                score, diag = -float(MT.ll_vec(data.y[es], pe).mean()), m.diagnostics
            else:
                m = FairTabular(family, cfg, data.k_perm, 0).fit(data.F[fit], data.y[fit], data.F[es], data.y[es])
                score, diag = auc(data.y[es], m.score(data.F[es])), m.diagnostics
            trials.append(dict(config_id=cfg["config_id"], es_score=round(score, 6), diagnostics=diag, s=round(time.perf_counter() - t1, 1)))
        except Exception as e:                                          # recorded, never hidden
            trials.append(dict(config_id=cfg["config_id"], error=repr(e)[:300], s=round(time.perf_counter() - t1, 1)))
        log(f"{family} fold {fold} {cfg['config_id']}: {trials[-1].get('es_score', trials[-1].get('error'))} ({trials[-1]['s']} s)")
    ok = [t for t in trials if "es_score" in t]
    if not ok:
        return dict(family=family, fold=fold, trials=trials, error="every configuration failed")
    best = max(ok, key=lambda t: t["es_score"])
    cfg = next(c for c in configs if c["config_id"] == best["config_id"])
    per_seed = {}
    for s in seeds:
        if family in NEURAL:
            m = _neural(family, cfg, s, device).fit(data.si[fit], data.so[fit], data.gap[fit], data.y[fit], data.si[es], data.so[es],
                                                     data.gap[es], data.y[es])
            zc, zt = m.logits(data.si[cal], data.so[cal], data.gap[cal]), m.logits(data.si[test], data.so[test], data.gap[test])
            diag = m.diagnostics
            if ckpt_dir is not None:
                import torch
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                p = ckpt_dir / f"{family}_fold{fold}_seed{s}.pt"
                if p.exists():
                    raise SystemExit(f"{p} exists; checkpoints are never overwritten")
                torch.save(m.net.state_dict(), p)
        else:
            m = FairTabular(family, cfg, data.k_perm, s).fit(data.F[fit], data.y[fit], data.F[es], data.y[es])
            zc, zt, diag = m.score(data.F[cal]), m.score(data.F[test]), m.diagnostics
        a = fit_temperature(zc, data.y[cal])
        per_seed[str(s)] = dict(scale=a, p_test=(1 / (1 + np.exp(-a * zt))).tolist(), diagnostics=diag,
                                cal=MT.basic(data.y[cal], 1 / (1 + np.exp(-a * zc))))
    return dict(family=family, fold=fold, selected=cfg, selected_edge_flags=SP.edge_flags(family, cfg), trials=trials,
                per_seed=per_seed, test_rows=np.flatnonzero(test).tolist(), runtime_s=round(time.perf_counter() - t0, 1))


def _neural(family: str, cfg: dict[str, Any], seed: int, device) -> FairNeural:
    net = NetConfig(family, int(cfg["hidden"]), int(cfg.get("emb", 48)), int(cfg["layers"]), int(cfg.get("head", 96)), float(cfg["dropout"]))
    return FairNeural(net, lr=cfg["lr"], weight_decay=cfg["weight_decay"], batch=int(cfg["batch"]), epochs=int(cfg["epochs"]),
                      patience=int(cfg["patience"]), clip=float(cfg.get("clip", 1.0)), seed=seed, device=device)


def assemble(results: list[dict[str, Any]], n: int) -> dict[str, np.ndarray]:
    """Seed-mean test predictions over all outer folds (every row predicted exactly once)."""
    p = np.full(n, np.nan)
    for r in results:
        if "per_seed" not in r:
            continue
        rows = np.asarray(r["test_rows"])
        p[rows] = np.mean([np.asarray(v["p_test"]) for v in r["per_seed"].values()], axis=0)
    return dict(p=p)


def dump(path: Path, obj: Any) -> None:
    if path.exists():
        raise SystemExit(f"{path} exists; not overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
