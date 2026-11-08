"""Command line: detect events for matches, or detect and score them against the PFF labels.

    python -m football_fatigue.event_detection detect 3812 [--overwrite]
    python -m football_fatigue.event_detection evaluate --split test [--workers 4]
    python -m football_fatigue.event_detection viewer 3812          # local HTML: detections vs labels with position replay
    python -m football_fatigue.event_detection train --workers 4    # ML detector: touch table + fold models (all 64 matches)
    python -m football_fatigue.event_detection evaluate --detector ml --split test
    python -m football_fatigue.event_detection report --split test      # mentor report from both evaluations

Exit codes: 0 done, 1 a match failed, 2 invalid arguments.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from . import evaluate as E
from .config import ALL_MATCHES, DETECTOR_VERSION, DEV_MATCHES, DetectorConfig, test_matches
from .ml import ML_VERSION, TouchTable, train_folds
from .report import write_report
from .runner import DEFAULT_MODEL_DIR, ML_NOTES, NOTES, calibrate_threshold, ml_config, run_match, touch_table_rows


def _resolve_matches(args: argparse.Namespace) -> list[int]:
    if args.matches:
        return [int(m) for m in args.matches]
    return list({"dev": DEV_MATCHES, "test": test_matches(), "all": ALL_MATCHES}[args.split])


def _train(args: argparse.Namespace) -> int:
    """Build the touch table for every match (or the given ones) and train the fold models plus the full model."""
    import numpy as np
    matches = [int(m) for m in args.matches] or list(ALL_MATCHES)
    parts = []
    with ProcessPoolExecutor(max(1, args.workers)) as ex:
        for m, X, y, done in ex.map(touch_table_rows, matches, [args.root] * len(matches), [args.cache or None] * len(matches)):
            parts.append((m, X, y, done))
            print(f"{m}: {len(y)} candidate rows", flush=True)
    table = TouchTable(np.concatenate([np.full(len(p[2]), p[0]) for p in parts]), np.vstack([p[1] for p in parts]),
                       np.concatenate([p[2] for p in parts]), np.concatenate([p[3] for p in parts]))
    out = Path(args.model)
    if (out / "manifest.json").exists() and not args.overwrite:
        print(f"a model already exists (use --overwrite): {out}", file=sys.stderr)
        return 2
    table.save(out / "touch_table.npz")
    manifest = train_folds(table, out)
    manifest.update(calibrate_threshold(args.root, args.cache or None, out))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "folds"}, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m football_fatigue.event_detection")
    ap.add_argument("command", choices=["detect", "evaluate", "viewer", "train", "report"])
    ap.add_argument("matches", nargs="*", help="match ids (default: the --split)")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="test")
    ap.add_argument("--root", default="data/raw/FIFA World Cup 2022")
    ap.add_argument("--cache", default="data/processed/event_detection/cache", help="positions cache ('' disables)")
    ap.add_argument("--out", default="data/processed/event_detection/runs", help="detected events output root")
    ap.add_argument("--report", default="reports/event_detection", help="evaluation report root")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--detector", choices=["rules", "ml"], default="rules")
    ap.add_argument("--model", default=str(DEFAULT_MODEL_DIR), help="ML model directory (fold models + full model)")
    args = ap.parse_args(argv)
    if args.command == "train":
        return _train(args)
    if args.command == "report":
        rep = Path(args.report)
        dest = write_report(rep / "mentor_report.html", rep / DETECTOR_VERSION / args.split / "summary.json",
                            rep / ML_VERSION / args.split / "summary.json", rep / "rules-0.1" / args.split / "summary.json",
                            Path(args.model) / "manifest.json")
        print(f"report: {dest}")
        return 0
    try:
        matches = _resolve_matches(args)
    except ValueError:
        ap.print_usage(sys.stderr)
        return 2
    if args.command == "viewer" and not args.matches:
        print("viewer needs explicit match ids", file=sys.stderr)
        return 2
    score = args.command in ("evaluate", "viewer")
    write_out = args.out if args.command == "detect" else None
    viewer_dir = args.report if args.command == "viewer" else None
    jobs = [(m, args.root, args.cache or None, write_out, args.overwrite, score, viewer_dir, args.detector, args.model) for m in matches]
    results, failed = [], []
    with ProcessPoolExecutor(max(1, args.workers)) as ex:
        futs = {ex.submit(run_match, *j): j[0] for j in jobs}
        for f in futs:
            m = futs[f]
            try:
                r = f.result()
            except Exception as e:                      # reported, never hidden
                failed.append((m, repr(e)))
                print(f"{m}: FAILED {e!r}", file=sys.stderr)
                continue
            results.append(r)
            print(f"{m}: {r['events']} events from {r['touches']} touches" + (f" -> {r['written']}" if r.get("written") else ""))
    if args.command == "evaluate" and results:
        per = {r["match_id"]: r["score"] for r in results}
        merged = E.merge_scores(per.values())
        label = "+".join(map(str, args.matches)) if args.matches else args.split
        version = ML_VERSION if args.detector == "ml" else DETECTOR_VERSION
        meta = dict(detector_version=version, split=label, matches=len(per), not_scoreable=sum(s["not_scoreable"] for s in per.values()), notes=NOTES + (ML_NOTES if args.detector == "ml" else []))
        dest = Path(args.report) / version / label
        dest.mkdir(parents=True, exist_ok=True)
        md = E.render_markdown(f"Event detection scorecard: {label}", merged, per, meta)
        (dest / "summary.md").write_text(md, encoding="utf-8")
        (dest / "summary.json").write_text(json.dumps({"meta": meta, "merged": {k: v for k, v in merged.items() if k != "timing_dt_s"},
                                                       "failed": failed,
                                                       "config": (ml_config() if args.detector == "ml" else DetectorConfig()).to_dict()}, indent=1), encoding="utf-8")
        with open(dest / "per_match.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["match_id", "flags", "type", "tolerance_s", "n_truth", "n_pred", "tp", "precision", "recall", "f1"])
            for m, s in sorted(per.items()):
                for key, c in s["counts"].items():
                    typ, tol = key.split("@")
                    w.writerow([m, ";".join(s["flags"]), typ, tol, c["n_truth"], c["n_pred"], c["tp"], *[("" if v is None else round(v, 4)) for v in E.prf(c)]])
        print(md)
        print(f"\nreport: {dest}")
    if failed:
        print(f"{len(failed)} match(es) failed: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
