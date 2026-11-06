"""CLI: ``python -m football_fatigue.encounters {splits,build}`` (see the package docstring)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from football_fatigue.ingestion import Dataset, IngestionError, approved_output_root

from .catalogue import OUTPUT_DIR, peak_memory_mb, run_from_cache, run_match
from .splits import assign, default_quarantine, write_manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m football_fatigue.encounters")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("splits", "build"):
        p = sub.add_parser(name)
        p.add_argument("--root", default="data/raw", help="raw dataset root (read only)")
        p.add_argument("--out", default=None, help=f"output root (default: <processed>/{OUTPUT_DIR})")
        if name == "build":
            p.add_argument("--match", required=True)
            p.add_argument("--max-frames", type=int, default=None, help="bounded development run")
            p.add_argument("--from-cache", default=None, help="recompute from a verified root.cache.npz instead of a raw tracking pass")
            p.add_argument("--allow-validation", action="store_true", help="validation matches are refused unless named explicitly")
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else approved_output_root(a.root) / OUTPUT_DIR
    try:
        ds = Dataset(a.root)
        ids = ds.match_ids()
        manifest = assign(ids) if len(ids) == 64 else None
        if a.cmd == "splits":
            if manifest is None:
                raise IngestionError(f"the split manifest needs the 64 FIFA matches; found {len(ids)}")
            path = write_manifest(a.root, out, manifest)
            print(json.dumps(dict(manifest=str(path), sizes={k: len(v) for k, v in manifest["splits"].items()}), indent=1))
            return 0
        mid = str(a.match)
        split = manifest["matches"].get(mid) if manifest else None
        if split and split["split"] == "test":
            raise IngestionError(f"match {mid} is in the locked test split and is not processed")
        if split and split["split"] == "validation" and not a.allow_validation:
            raise IngestionError(f"match {mid} is a validation match; pass --allow-validation to process it")
        t0 = time.perf_counter()
        if a.from_cache:
            if a.max_frames is not None:
                raise IngestionError("--max-frames does not apply to --from-cache")
            paths, summary = run_from_cache(a.root, a.from_cache, out_root=out, split=split, expected_match_id=mid)
        else:
            paths, summary = run_match(a.root, mid, out_root=out, quarantine_periods=default_quarantine(mid), split=split, max_frames=a.max_frames)
        print(json.dumps(dict(outputs={k: str(v) for k, v in paths.items()}, runtime_s=round(time.perf_counter() - t0, 1), peak_memory_mb=peak_memory_mb(),
                              frames=summary["frames"], encounters={k: v["count"] for k, v in summary["encounters"].items()},
                              jumps=summary["jumps"]["count"]), indent=1))
        return 0
    except IngestionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
