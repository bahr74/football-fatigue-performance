"""Development experiments: scenario tables per match (cached, written once) and leave-one-match-out cross-validation.

Every model is fitted only on training matches of its fold; the held-out match is predicted once. Out-of-fold predictions
of all development matches are then scored together (and per match). The hybrid network uses one training match (the next in
sorted order) for early stopping. Results go to ``m11-recovery-v1/experiments/<name>/`` (never overwritten).
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from football_fatigue.ingestion import IngestionError, approved_output_root

from . import dataset as DS
from . import evaluate as EV
from .models import GBM, Baseline, Hybrid
from .motion import BASELINES
from .views import load_match

ROOT_NAME = "m11-recovery-v1"
SOURCE = "m11-contacts-v1"                # identity-free contacts (S1.2); the v3 intersections are a biased population
POPULATIONS = {                            # name -> (source directory, view cache directory)
    "contacts": (SOURCE, "views"),
    "intersections": ("m11-intersections-v3", "views-v3"),   # advisor: exact geometric intersections, |t_A - t_B| <= 0.20 s
}
DEV_MATCHES = ("3812", "3814", "3815", "3816", "3829", "3833", "3836", "3847", "3851", "10506")


def out_root(raw_root) -> Path:
    return approved_output_root(raw_root) / ROOT_NAME


def _save(path: Path, arrays: dict[str, np.ndarray], meta: dict) -> None:
    from football_fatigue.intersections.audit import npz_bytes

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as f:
        f.write(npz_bytes(arrays))
    with open(path.with_suffix(".json"), "x", encoding="utf-8") as f:
        json.dump(meta | dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest()), f, indent=1)


def table(raw_root, match_id: str, src_root: Path | None = None, cache_dir: str = "views", shifted: bool = False) -> tuple[dict[str, np.ndarray], dict]:
    """The scenario table of one match, built from the contact catalogue and cached once (checked against its sha256 when reused)."""
    p = out_root(raw_root) / cache_dir / f"{match_id}.npz"
    if p.exists():
        meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
        if hashlib.sha256(p.read_bytes()).hexdigest() != meta["sha256"]:
            raise IngestionError(f"{p} does not match its recorded checksum")
        with np.load(p) as z:
            return {k: z[k] for k in z.files}, meta
    src = (src_root or approved_output_root(raw_root) / SOURCE) / match_id
    its = load_match(src)
    t0 = time.perf_counter()
    t, meta = DS.build(its, shifted=shifted)
    meta |= dict(match=match_id, source=str(src), shifted=shifted, seconds=round(time.perf_counter() - t0, 1))
    _save(p, t, meta)
    return t, json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))     # includes the recorded sha256


MODELS: dict[str, Callable[[list[str]], Any]] = {
    **{b: (lambda names, b=b: Baseline(b)) for b in BASELINES},
    "gbm": lambda names: GBM(names),
    "hybrid": lambda names: Hybrid(),
}


def lomo(t: dict[str, np.ndarray], name: str, names: list[str], matches=None, factory=None) -> tuple[np.ndarray, dict]:
    matches = sorted(set(t["match"])) if matches is None else matches
    p = np.full(len(t["y"]), np.nan)
    info = {}
    for k, m in enumerate(matches):
        te = t["match"] == m
        tr = ~te
        model = (factory or MODELS[name])(names)
        t0 = time.perf_counter()
        if isinstance(model, Hybrid):
            es = matches[(k + 1) % len(matches)]
            fit_idx, es_idx = tr & (t["match"] != es), t["match"] == es
            model.fit(DS.take(t, fit_idx), DS.take(t, es_idx))
            info[m] = dict(early_stop_match=es, epochs=model.epochs_run)
        else:
            model.fit(DS.take(t, tr))
        p[te] = model.predict(DS.take(t, te))
        info.setdefault(m, {})["seconds"] = round(time.perf_counter() - t0, 1)
    return p, info


def population(raw_root, name: str, match_id: str):
    src, cache = POPULATIONS[name]
    return table(raw_root, match_id, src_root=approved_output_root(raw_root) / src, cache_dir=cache)


def run(raw_root, models: list[str], exp: str, rows: np.ndarray | None = None, note: str = "", pop: str = "contacts") -> dict[str, Any]:
    dest = out_root(raw_root) / "experiments" / exp
    if dest.exists():
        raise IngestionError(f"{dest} already exists and is never overwritten")
    tabs, metas = zip(*(population(raw_root, pop, m) for m in DEV_MATCHES))
    t = DS.concat(list(tabs))
    if rows is not None:
        t = DS.take(t, rows(t) if callable(rows) else rows)
    names = metas[0]["feature_names"]
    res, preds = {}, {}
    for name in models:
        t0 = time.perf_counter()
        p, info = lomo(t, name, names)
        th = {f"fa_{int(fa * 100):02d}": EV.threshold_for(t["y"], p, fa) for fa in EV.FA_TARGETS}
        res[name] = dict(report=EV.report(t, p, th), thresholds_fitted_on_this_oof=th, folds=info, seconds=round(time.perf_counter() - t0, 1))
        preds[name] = p
        o = res[name]["report"]["overall"]
        a = res[name]["report"]["abstention"]["fa_05"]
        print(f"{exp} {name:9s} acc {o['accuracy']} ll {o['log_loss']} ece {o['ece']} | FA5%: cov {a['coverage']} rec {a['recovery_rate']}"
              f" | {res[name]['seconds']} s", flush=True)
    summary = dict(experiment=exp, note=note, population=pop, matches=list(DEV_MATCHES), rows=int(len(t["y"])), interactions=int(len(set(t["event"]))),
                   by_tier=dict(zip(*np.unique(t["tier"], return_counts=True))) if len(t["y"]) else {}, results=res,
                   view_tables={m["match"]: dict(sha256=m["sha256"], rows=m["rows"], unusable=m["unusable"]) for m in metas})
    summary["by_tier"] = {str(k): int(v) for k, v in summary["by_tier"].items()}
    from football_fatigue.intersections.__main__ import write_dir
    from football_fatigue.intersections.audit import npz_bytes

    write_dir(dest, {"results.json": json.dumps(summary, indent=1, default=str).encode(),
                     "oof.npz": npz_bytes(dict(event=t["event"], match=t["match"], gap=t["gap"], noise=t["noise"], y=t["y"],
                                               **{f"p_{k}": v for k, v in preds.items()}))})
    return summary


def robustness(raw_root, model: str, exp: str) -> dict[str, Any]:
    """Gap-centre robustness (pre-declared check of the t* choice): each fold fits on the standard views of the training
    matches and predicts *shifted* views (gap centre t* + U{-0.4..0.4} s) of the held-out match."""
    dest = out_root(raw_root) / "experiments" / exp
    if dest.exists():
        raise IngestionError(f"{dest} already exists and is never overwritten")
    tabs, metas = zip(*(table(raw_root, m) for m in DEV_MATCHES))
    stabs, smetas = zip(*(table(raw_root, m, cache_dir="views-shifted", shifted=True) for m in DEV_MATCHES))
    t, ts = DS.concat(list(tabs)), DS.concat(list(stabs))
    names = metas[0]["feature_names"]
    p = np.full(len(ts["y"]), np.nan)
    for m in DEV_MATCHES:
        mdl = MODELS[model](names)
        mdl.fit(DS.take(t, t["match"] != m))
        p[ts["match"] == m] = mdl.predict(DS.take(ts, ts["match"] == m))
    th = {f"fa_{int(fa * 100):02d}": EV.threshold_for(ts["y"], p, fa) for fa in EV.FA_TARGETS}
    rep = EV.report(ts, p, th)
    print(f"{exp} {model} shifted: acc {rep['overall']['accuracy']} ll {rep['overall']['log_loss']}", flush=True)
    from football_fatigue.intersections.__main__ import write_dir
    from football_fatigue.intersections.audit import npz_bytes

    summary = dict(experiment=exp, model=model, rows=int(len(ts["y"])), shifts_s=list(DS.SHIFTS_S), results={model: dict(report=rep, thresholds_fitted_on_this_oof=th)},
                   view_tables={m["match"]: dict(sha256=m["sha256"], rows=m["rows"], unusable=m["unusable"]) for m in smetas})
    write_dir(dest, {"results.json": json.dumps(summary, indent=1, default=str).encode(),
                     "oof.npz": npz_bytes(dict(event=ts["event"], match=ts["match"], gap=ts["gap"], noise=ts["noise"], y=ts["y"], **{f"p_{model}": p}))})
    return summary


def transfer(raw_root, model: str, exp: str, train_pop: str = "contacts", test_pop: str = "intersections") -> dict[str, Any]:
    """Cross-population leave-one-match-out: fit on ``train_pop`` views of the other matches, predict ``test_pop`` views of the
    held-out match (no row of the held-out match is used for fitting)."""
    dest = out_root(raw_root) / "experiments" / exp
    if dest.exists():
        raise IngestionError(f"{dest} already exists and is never overwritten")
    tr_tabs, metas = zip(*(population(raw_root, train_pop, m) for m in DEV_MATCHES))
    te_tabs, te_metas = zip(*(population(raw_root, test_pop, m) for m in DEV_MATCHES))
    t, ts = DS.concat(list(tr_tabs)), DS.concat(list(te_tabs))
    names = metas[0]["feature_names"]
    p = np.full(len(ts["y"]), np.nan)
    for m in DEV_MATCHES:
        mdl = MODELS[model](names)
        if isinstance(mdl, Hybrid):
            raise IngestionError("transfer supports the non-sequence models only")
        mdl.fit(DS.take(t, t["match"] != m))
        p[ts["match"] == m] = mdl.predict(DS.take(ts, ts["match"] == m))
    th = {f"fa_{int(fa * 100):02d}": EV.threshold_for(ts["y"], p, fa) for fa in EV.FA_TARGETS}
    rep = EV.report(ts, p, th)
    print(f"{exp} {model} {train_pop}->{test_pop}: acc {rep['overall']['accuracy']} ll {rep['overall']['log_loss']}", flush=True)
    from football_fatigue.intersections.__main__ import write_dir
    from football_fatigue.intersections.audit import npz_bytes

    summary = dict(experiment=exp, model=model, train_population=train_pop, test_population=test_pop, rows=int(len(ts["y"])),
                   results={model: dict(report=rep, thresholds_fitted_on_this_oof=th)},
                   view_tables={m["match"]: dict(sha256=m["sha256"], rows=m["rows"], unusable=m["unusable"]) for m in te_metas})
    write_dir(dest, {"results.json": json.dumps(summary, indent=1, default=str).encode(),
                     "oof.npz": npz_bytes(dict(event=ts["event"], match=ts["match"], gap=ts["gap"], noise=ts["noise"], y=ts["y"], **{f"p_{model}": p}))})
    return summary


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m football_fatigue.m11_recovery.experiments")
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--models", nargs="+", required=True, choices=sorted(MODELS))
    ap.add_argument("--note", default="")
    ap.add_argument("--shifted", action="store_true", help="gap-centre robustness: test on shifted views")
    ap.add_argument("--population", default="contacts", choices=sorted(POPULATIONS))
    ap.add_argument("--transfer-to", choices=sorted(POPULATIONS), help="fit on --population, test on this population (LOMO)")
    a = ap.parse_args(argv)
    if a.transfer_to:
        for mdl in a.models:
            transfer(str(Path(a.root).resolve()), mdl, a.exp if len(a.models) == 1 else f"{a.exp}-{mdl}", a.population, a.transfer_to)
        return 0
    if a.shifted:
        for mdl in a.models:
            robustness(str(Path(a.root).resolve()), mdl, a.exp if len(a.models) == 1 else f"{a.exp}-{mdl}")
        return 0
    run(str(Path(a.root).resolve()), a.models, a.exp, note=a.note, pop=a.population)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
