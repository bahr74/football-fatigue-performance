"""Command line: per-match quality report.

    python -m football_fatigue.ingestion quality 3812 --events root [--root data/raw] [--max-frames N] [--write [--overwrite]]

Exit codes: 0 report produced (quality issues, if any, are in the report); 1 fatal structural error in a source;
2 invalid arguments (including an invalid match id or an unsafe output location).
"""

from __future__ import annotations

import argparse
import sys

from .loaders import Dataset
from .models import EventVersion, IngestionError
from .validation import quality_report, write_quality_report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m football_fatigue.ingestion", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("quality", help="per-match quality summary (streams the tracking file once)")
    q.add_argument("match")
    q.add_argument("--events", required=True, choices=[v.value for v in EventVersion], help="event version (no default)")
    q.add_argument("--root", default="data/raw")
    q.add_argument("--max-frames", type=int, default=None, help="bounded partial scan (smoke tests)")
    q.add_argument("--write", action="store_true", help="write the JSON report under <root>/../processed/ingestion_quality")
    q.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    try:
        ds = Dataset(a.root)
        rep = quality_report(ds, a.match, EventVersion(a.events), max_frames=a.max_frames)
        print(rep.summary_text())
        if a.write:
            print(f"wrote {write_quality_report(rep, a.root, overwrite=a.overwrite)}")
    except IngestionError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    return 1 if rep.fatal_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
