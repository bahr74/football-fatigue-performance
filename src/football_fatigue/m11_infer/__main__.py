"""CLI: ``python -m football_fatigue.m11_infer <command>``

- ``check-bundles [--device cpu] [--write]``: load all 16 fold models (GRU C seed 0 / 160-epoch budget, HGB C), reproduce their
  stored out-of-fold predictions, test symmetry, recompute the abstention thresholds; ``--write`` records the report and the
  checkpoint hashes once in ``data/processed/m11-infer-v1/bundles_v1.json`` (never overwritten).
- ``check-adapter --match M [--limit N]``: rebuild Stage 3 samples of a train/CV match from raw trajectories through the adapter
  and compare arrays and predictions with the stored ones.
- ``export-fifa --match M --row R --out input.json [--truth truth.json]``: write one development scenario as model-visible
  trajectory JSON (label in a separate file).
- ``predict --json input.json (--fold F | --match M) [--noise-sigma-m S] [--operating-point error_05|error_02]``: routed,
  abstaining prediction; prints the result as JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import adapter as A
from . import bundle as B


def _dump(obj) -> str:
    return json.dumps(obj, indent=1, sort_keys=True, default=B._jsonable)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m football_fatigue.m11_infer")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check-bundles")
    c.add_argument("--device", default="cpu")
    c.add_argument("--write", action="store_true")
    a_ = sub.add_parser("check-adapter")
    a_.add_argument("--match", required=True)
    a_.add_argument("--limit", type=int, default=50)
    a_.add_argument("--device", default="cpu")
    e = sub.add_parser("export-fifa")
    e.add_argument("--match", required=True)
    e.add_argument("--row", type=int, required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--truth")
    p = sub.add_parser("predict")
    p.add_argument("--json", required=True)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--fold", type=int)
    g.add_argument("--match")
    p.add_argument("--noise-sigma-m", type=float)
    p.add_argument("--operating-point", default="error_05", choices=B.OPERATING_POINTS)
    p.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)

    if a.cmd == "check-bundles":
        rep = B.check_bundles(device=a.device, log=lambda s: print(s, file=sys.stderr, flush=True))
        ok = rep["all_reproduced"] and rep["thresholds_match"]
        if a.write:
            man = dict(schema="m11-infer-bundles-v1", source=B.SOURCE, bundles={m: {f: dict(checkpoint=r["checkpoint"],
                                                                                          checkpoint_sha256=r["checkpoint_sha256"])
                                                                                    for f, r in v.items()} for m, v in rep["bundles"].items()},
                       thresholds=B.THRESHOLDS, check=rep)
            if not ok:
                raise SystemExit("bundle check failed; nothing written")
            B.write_json_once(B.manifest_path(), man)
            print(f"wrote {B.manifest_path()}", file=sys.stderr)
        print(_dump(dict(all_reproduced=rep["all_reproduced"], thresholds_match=rep["thresholds_match"],
                         thresholds_recomputed=rep["thresholds_recomputed"],
                         max_abs_diff={m: max(r["max_abs_diff_vs_stored"] for r in v.values()) for m, v in rep["bundles"].items()},
                         symmetry_max_err={m: max(r["symmetry_max_err"] for r in v.values()) for m, v in rep["bundles"].items()},
                         gru_budget_binding_folds=[int(f) for f, r in rep["bundles"]["gru_c"].items() if r["budget_binding"]])))
        return 0 if ok else 1

    if a.cmd == "check-adapter":
        from .fifa import check_adapter
        from .predict import Predictor

        pr = Predictor.for_match(a.match, device=a.device)
        r = check_adapter(a.match, limit=a.limit, predictor=pr, log=lambda s: print(s, file=sys.stderr))
        print(_dump({k: v for k, v in r.items() if k != "rows"}))
        return 0 if r["all_identical"] and r["predictions_reproduced"] else 1

    if a.cmd == "export-fifa":
        from .fifa import MatchSource

        sc = MatchSource(a.match).scenario(a.row)
        Path(a.out).write_text(_dump(sc.input_json()), encoding="utf-8")
        if a.truth:
            Path(a.truth).write_text(_dump(sc.truth_json()), encoding="utf-8")
        print(f"wrote {a.out}" + (f" and {a.truth}" if a.truth else ""), file=sys.stderr)
        return 0

    from .predict import Predictor

    d = json.loads(Path(a.json).read_text(encoding="utf-8"))
    inc, out, g0, g1 = A.tracks_from_json(d)
    noise = a.noise_sigma_m if a.noise_sigma_m is not None else d.get("noise_sigma_m")
    if a.fold is not None:
        pr = Predictor(a.fold, device=a.device)
    else:
        m = a.match or (d.get("source") or {}).get("match_id")
        if m is None:
            raise SystemExit("choose the fold model: --fold F, or --match M (the fold that never saw development match M). "
                             "No model trained on all 32 matches exists yet (Milestone 4).")
        pr = Predictor.for_match(m, device=a.device)
    res = pr.predict(inc, out, g0, g1, noise_sigma_m=noise, operating_point=a.operating_point)
    print(_dump(res.to_dict()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
