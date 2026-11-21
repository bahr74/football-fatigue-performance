"""CLI: python -m football_fatigue.video.failures {decompose,aggregate,engine,gallery} --data-root ..."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_OUT = Path("data/processed/video")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m football_fatigue.video.failures")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("decompose", "aggregate", "engine", "gallery", "heavy", "activation", "cases"):
        p = sub.add_parser(name)
        p.add_argument("--data-root", type=Path, required=True)
        p.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
        if name == "decompose":
            p.add_argument("--source", action="append", required=True)
            p.add_argument("--max-frames", type=int)
            p.add_argument("--profile", action="append", help="tracker-input profiles (default B0, T0, T0L)")
            p.add_argument("--version", default=None, help="output folder under _diag (default pipeline-v1)")
        if name == "engine":
            p.add_argument("--profile", action="append", help="profiles to compare (default B0, T0, T0L)")
            p.add_argument("--version", default=None, help="output folder under _diag (default engine-t0l-v1)")
    a = ap.parse_args(argv)
    if a.cmd == "decompose":
        from football_fatigue.video.failures.pipeline import decompose

        for s in a.source:
            kw = {"profiles": tuple(a.profile)} if a.profile else {}
            if a.version:
                kw["version"] = a.version
            print(decompose(a.data_root, a.out_root, s, max_frames=a.max_frames, **kw), flush=True)
    elif a.cmd == "aggregate":
        from football_fatigue.video.failures.aggregate import write

        print(write(a.out_root))
    elif a.cmd == "engine":
        from football_fatigue.video.failures.engine_compare import compare

        kw = {"profiles": tuple(a.profile)} if a.profile else {}
        if a.version:
            kw["version"] = a.version
        print(compare(a.data_root, a.out_root, **kw))
    elif a.cmd == "gallery":
        from football_fatigue.video.failures.gallery import build

        print(build(a.data_root, a.out_root))
    elif a.cmd == "heavy":
        from football_fatigue.video.failures.heavy import analyse

        print(analyse(a.data_root, a.out_root))
    elif a.cmd == "activation":
        from football_fatigue.video.failures.activation import run

        print(run(a.data_root, a.out_root))
    elif a.cmd == "cases":
        from football_fatigue.video.failures.cases import build as build_cases

        print(build_cases(a.data_root, a.out_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
