"""M11 step 1: non-destructive two-player encounter catalogue over RAW FIFA tracking coordinates.

    python -m football_fatigue.encounters splits --root data/raw
    python -m football_fatigue.encounters build --root data/raw --match 3812

Reads through ``MatchData(policy=CoordinatePolicy.RAW)``; never reads provider-smoothed coordinates, never changes, removes,
interpolates or relabels an observation, and writes only below ``data/processed/m11-encounters-v1/``.
"""

from .catalogue import CATEGORIES, find_encounters, run_match, summarize, write_outputs
from .quality import M11Config, assess
from .splits import assign
from .trajectories import RawCache, build_cache

__all__ = ["CATEGORIES", "M11Config", "RawCache", "assess", "assign", "build_cache", "find_encounters", "run_match", "summarize",
           "write_outputs"]
