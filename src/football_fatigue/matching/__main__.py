"""CLI: ``python -m football_fatigue.matching build --match <id>``, ``ml --matches <id> ...`` and ``ml2 --matches <id> ...``, and the step-6
``freeze`` / ``valbuild`` / ``predict`` / ``score`` stages (see the package docstring)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from football_fatigue.encounters.catalogue import peak_memory_mb
from football_fatigue.encounters.splits import MANIFEST_NAME
from football_fatigue.ingestion import IngestionError, approved_output_root

from .benchmark import OUTPUT_DIR, SOURCE_DIR, run_match

SPLITS_DIR = "m11-encounters-v1"        # the frozen split manifest lives with the first step-1 outputs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m football_fatigue.matching")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--root", default="data/raw", help="raw dataset root (read only)")
    p.add_argument("--match", required=True)
    p.add_argument("--source", default=None, help=f"step-1 catalogue root (default: <processed>/{SOURCE_DIR})")
    p.add_argument("--out", default=None, help=f"output root (default: <processed>/{OUTPUT_DIR})")
    q = sub.add_parser("ml", help="step 4: leave-one-match-out pairwise identity classifiers on task B")
    q.add_argument("--root", default="data/raw", help="raw dataset root (read only)")
    q.add_argument("--matches", nargs="+", required=True, help="development matches with m11-matching-v3 outputs")
    q.add_argument("--source", default=None, help=f"matching benchmark root (default: <processed>/{OUTPUT_DIR})")
    q.add_argument("--out", default=None, help="output directory (default: <processed>/m11-ml-v1)")
    r = sub.add_parser("ml2", help="step 5: nested leave-one-match-out diagnosis, ablations and model comparison on task B")
    r.add_argument("--root", default="data/raw", help="raw dataset root (read only)")
    r.add_argument("--matches", nargs="+", required=True, help="development matches with m11-matching-v3 outputs")
    r.add_argument("--source", default=None, help=f"matching benchmark root (default: <processed>/{OUTPUT_DIR})")
    r.add_argument("--out", default=None, help="output directory (default: <processed>/m11-ml-v2)")
    f = sub.add_parser("freeze", help="step 6: freeze the selected classifier on development matches (writes m11-frozen-v1)")
    f.add_argument("--root", default="data/raw", help="raw dataset root (read only)")
    f.add_argument("--matches", nargs="+", required=True, help="the development matches the model is refitted on")
    v = sub.add_parser("valbuild", help="step 6: Step 2-3 benchmark of one validation match (inside m11-validation-v1)")
    v.add_argument("--root", default="data/raw", help="raw dataset root (read only)")
    v.add_argument("--match", required=True)
    for name in ("predict", "score"):
        s6 = sub.add_parser(name, help=f"step 6: {name} the frozen validation evaluation")
        s6.add_argument("--root", default="data/raw", help="raw dataset root (read only)")
    a = ap.parse_args(argv)
    if a.cmd in ("freeze", "valbuild", "predict", "score"):
        from . import validate
        try:
            t0 = time.perf_counter()
            if a.cmd == "freeze":
                out, res = validate.run_freeze(a.root, a.matches)
                info = dict(output=str(out), model_sha256=res["model_sha256"], operating_points=res["spec"]["operating_points"])
            elif a.cmd == "valbuild":
                info = validate.build_validation_match(a.root, a.match)
            elif a.cmd == "predict":
                out, man = validate.run_predict(a.root)
                info = dict(output=str(out), files=man["outputs_sha256"])
            else:
                out, _ = validate.run_score(a.root)
                info = dict(output=str(out))
            print(json.dumps(dict(info, runtime_s=round(time.perf_counter() - t0, 1)), indent=1, default=str))
            return 0
        except IngestionError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    if a.cmd == "ml2":
        from .ml2 import run_ml2
        try:
            t0 = time.perf_counter()
            out, summary = run_ml2(a.root, a.matches, source_root=a.source, out_root=a.out, log=lambda m: print(m, flush=True))
            print(json.dumps(dict(output=str(out), runtime_s=round(time.perf_counter() - t0, 1), selection=summary["selection"]), indent=1, default=str))
            return 0
        except IngestionError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    if a.cmd == "ml":
        from .ml import run_ml
        try:
            t0 = time.perf_counter()
            out, summary = run_ml(a.root, a.matches, source_root=a.source, out_root=a.out)
            print(json.dumps(dict(output=str(out), runtime_s=round(time.perf_counter() - t0, 1), folds=summary["folds"]), indent=1))
            return 0
        except IngestionError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    try:
        splits_path = approved_output_root(a.root) / SPLITS_DIR / MANIFEST_NAME
        if not splits_path.exists():
            raise IngestionError(f"the frozen split manifest {splits_path} is missing")
        split = json.loads(splits_path.read_text(encoding="utf-8"))["matches"].get(str(a.match))
        if split is None or split["split"] != "development":
            raise IngestionError(f"match {a.match} is not a development match; the matching pilot uses development matches only")
        t0 = time.perf_counter()
        paths, summary = run_match(a.root, a.match, source_root=a.source, out_root=Path(a.out) if a.out else None, split=split)
        au = summary["audit"]
        print(json.dumps(dict(outputs={k: str(v) for k, v in paths.items()}, runtime_s=round(time.perf_counter() - t0, 1), peak_memory_mb=peak_memory_mb(),
                              events=au["events"], accepted_by_tier=au["accepted_by_tier"], rejected=au["rejected"]), indent=1))
        return 0
    except IngestionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
