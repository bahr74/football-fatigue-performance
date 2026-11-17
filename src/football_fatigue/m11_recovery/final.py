"""Freeze the selected model and run the one official-validation evaluation (M11 identity recovery).

``freeze``: refits the selected model with fixed hyperparameters on all 10 development matches and writes
``m11-recovery-v1/models/<name>/`` (``model.joblib`` [+ ``model.pt``], ``spec.json`` with the operating thresholds frozen from the
development out-of-fold predictions of the selection experiment, ``manifest.json`` with checksums). Never overwritten.

``validate``: builds the identity-free contact catalogue of the 12 official validation matches into
``m11-recovery-v1/validation/contacts`` (the Step 6 validation caches are read, never changed), builds their scenario tables,
applies the frozen models and thresholds with no fitting of any kind, and writes ``m11-recovery-v1/validation/<tag>/`` once.
The locked test partition is never read. Run:
``python -m football_fatigue.m11_recovery.final freeze --exp <experiment> --model gbm`` then
``python -m football_fatigue.m11_recovery.final validate --models <frozen dir names>``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

from football_fatigue.ingestion import IngestionError, approved_output_root

from . import dataset as DS
from . import evaluate as EV
from .experiments import DEV_MATCHES, MODELS, out_root, table
from .models import Hybrid

VALIDATION_SOURCE = "m11-validation-v1/encounters"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def code_digest() -> dict[str, str]:
    return {p.name: _sha(p.read_bytes()) for p in sorted(Path(__file__).parent.glob("*.py"))}


def dev_table(raw_root):
    tabs, metas = zip(*(table(raw_root, m) for m in DEV_MATCHES))
    return DS.concat(list(tabs)), metas


def freeze(raw_root, exp: str, model: str, default_target: str = "fa_05", name: str | None = None) -> Path:
    from football_fatigue.intersections.__main__ import write_dir

    import joblib

    name = name or f"{model}-v1"
    dest = out_root(raw_root) / "models" / name
    if dest.exists():
        raise IngestionError(f"{dest} already exists and is never overwritten")
    edir = out_root(raw_root) / "experiments" / exp
    res_bytes = (edir / "results.json").read_bytes()
    res = json.loads(res_bytes)
    if model not in res["results"]:
        raise IngestionError(f"{model} is not in experiment {exp}")
    r = res["results"][model]
    t, metas = dev_table(raw_root)
    names = metas[0]["feature_names"]
    m = MODELS[model](names)
    t0 = time.perf_counter()
    blobs = {}
    if isinstance(m, Hybrid):
        import torch

        m.epochs = int(np.median([f["epochs"] for f in r["folds"].values()]))         # fixed: median early-stopping epoch
        m.fit(t)
        buf = io.BytesIO()
        torch.save({k: v.cpu() for k, v in m.net.state_dict().items()}, buf)
        blobs["model.pt"] = buf.getvalue()
        net, m.net = m.net, None
        jb = io.BytesIO()
        joblib.dump(m, jb)
        m.net = net
    else:
        m.fit(t)
        jb = io.BytesIO()
        joblib.dump(m, jb)
    blobs["model.joblib"] = jb.getvalue()
    spec = dict(schema="m11-recovery-v1", name=name, kind="hybrid" if isinstance(m, Hybrid) else model, model=model,
                n_phys=int(t["phys"].shape[-1]), feature_names=names, trained_on=list(DEV_MATCHES), rows=int(len(t["y"])),
                operating_points=r["thresholds_fitted_on_this_oof"], default_operating_point=default_target,
                operating_points_source=f"pooled leave-one-match-out predictions of experiment {exp} (development only)",
                development_oof=dict(overall=r["report"]["overall"], abstention=r["report"]["abstention"]),
                fit_seconds=round(time.perf_counter() - t0, 1),
                scope="provider positions with synthetic gaps (0.4-2.0 s) and noise (0-0.3 m); not tested on visual tracking")
    blobs["spec.json"] = json.dumps(spec, indent=1, default=str).encode()
    manifest = dict(outputs_sha256={k: _sha(v) for k, v in blobs.items()}, experiment=exp, experiment_results_sha256=_sha(res_bytes),
                    view_tables={mm["match"]: mm["sha256"] for mm in metas}, code_sha256=code_digest(),
                    validation_or_test_matches_used=False)
    blobs["manifest.json"] = json.dumps(manifest, indent=1).encode()
    write_dir(dest, blobs)
    return dest


def validation_matches(raw_root) -> list[str]:
    from football_fatigue.matching import ml

    split = json.loads((approved_output_root(raw_root) / ml.SPLITS_DIR / "splits.json").read_text(encoding="utf-8"))
    return sorted(split["splits"]["validation"])


def validation_tables(raw_root):
    from football_fatigue.intersections import contacts

    base = out_root(raw_root) / "validation" / "contacts"
    src = approved_output_root(raw_root) / VALIDATION_SOURCE
    tabs, metas = [], []
    for mid in validation_matches(raw_root):
        if not (base / mid).exists():
            s = contacts.run_match(raw_root, mid, False, split_expected="validation", source_root=src, out_base=base)
            print(mid, "validation contacts", s["contacts"], s["seconds"], "s", flush=True)
        t, m = table(raw_root, mid, src_root=base, cache_dir="validation/views")
        tabs.append(t)
        metas.append(m)
    return DS.concat(tabs), metas


def validate(raw_root, models: list[str], tag: str = "final") -> dict:
    from football_fatigue.intersections.__main__ import write_dir
    from football_fatigue.intersections.audit import npz_bytes

    from .infer import RecoveryModel

    dest = out_root(raw_root) / "validation" / tag
    if dest.exists():
        raise IngestionError(f"{dest} exists: the official validation evaluation is run once")
    t, metas = validation_tables(raw_root)
    out, preds = {}, {}
    for name in models:
        rm = RecoveryModel.load(out_root(raw_root) / "models" / name)
        p = rm.model.predict(t)
        th = {k: float(v) for k, v in rm.spec["operating_points"].items()}
        out[name] = dict(report=EV.report(t, p, th), frozen_thresholds=th, default_operating_point=rm.spec["default_operating_point"],
                         descriptive_thresholds_fitted_on_validation=EV.abstention(t["y"], p))
        preds[name] = p
        o, a = out[name]["report"]["overall"], out[name]["report"]["abstention"]["fa_05"]
        print(f"validation {name:12s} acc {o['accuracy']} ll {o['log_loss']} | frozen FA5%: cov {a['coverage']} FA {a['false_association_rate']}"
              f" rec {a['recovery_rate']}", flush=True)
    if len(models) > 1:
        out["paired_vs_first"] = {n: EV.paired_difference(t, preds[n], preds[models[0]]) for n in models[1:]}
    summary = dict(tag=tag, matches=validation_matches(raw_root), rows=int(len(t["y"])), interactions=int(len(set(t["event"]))),
                   by_tier={str(k): int(v) for k, v in zip(*np.unique(t["tier"], return_counts=True))}, results=out,
                   view_tables={m["match"]: dict(sha256=m["sha256"], rows=m["rows"], unusable=m["unusable"]) for m in metas},
                   note="official validation partition, evaluated once with frozen models and thresholds; examined earlier in Step 6 "
                        "(not pristine); locked test not used")
    write_dir(dest, {"results.json": json.dumps(summary, indent=1, default=str).encode(),
                     "predictions.npz": npz_bytes(dict(event=t["event"], match=t["match"], gap=t["gap"], noise=t["noise"], y=t["y"],
                                                       **{f"p_{k}": v for k, v in preds.items()}))})
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m football_fatigue.m11_recovery.final")
    ap.add_argument("--root", default="data/raw")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--exp", required=True)
    f.add_argument("--model", required=True)
    f.add_argument("--name")
    v = sub.add_parser("validate")
    v.add_argument("--models", nargs="+", required=True)
    v.add_argument("--tag", default="final")
    a = ap.parse_args(argv)
    root = str(Path(a.root).resolve())
    try:
        if a.cmd == "freeze":
            print(freeze(root, a.exp, a.model, name=a.name))
        else:
            validate(root, a.models, a.tag)
    except IngestionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
