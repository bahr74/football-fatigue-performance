"""``python -m football_fatigue.m11_phase2 <command>``: audit | review | ambiguity | readiness (development matches only)."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="football_fatigue.m11_phase2")
    ap.add_argument("command", choices=["audit", "review", "ambiguity", "readiness"])
    ap.add_argument("--matches", nargs="*", default=None, help="subset of train/CV match ids (default: all 32)")
    ap.add_argument("--figures", action="store_true", help="review: also draw trajectory figures (reads tracking of sampled matches)")
    a = ap.parse_args(argv)
    if a.command == "audit":
        from . import audit
        audit.main(a.matches)
    elif a.command == "review":
        from . import review
        review.main(a.matches, figures=a.figures)
    elif a.command == "ambiguity":
        from . import ambiguity
        ambiguity.main(a.matches)
    elif a.command == "readiness":
        from . import readiness
        readiness.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
