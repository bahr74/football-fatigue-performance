"""``python -m football_fatigue.m11_audit validity`` -> data/processed/m11-scientific-audit-v1/ (CPU, development matches only)."""

import sys

from .validity import main

if __name__ == "__main__":
    if sys.argv[1:] != ["validity"]:
        raise SystemExit("usage: python -m football_fatigue.m11_audit validity")
    raise SystemExit(main())
