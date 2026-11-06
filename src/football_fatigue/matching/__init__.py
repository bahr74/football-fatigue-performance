"""M11 steps 2-3: real-first player identity matching benchmark over RAW FIFA encounters (task A, identity-hidden association;
task B, withheld-interval association; FOOTBALLTrace E0-Literal and E0-Adapted).

    python -m football_fatigue.matching build --root data/raw --match 3812

Reads a verified ``m11-encounters-v2`` catalogue. It never reads provider-smoothed coordinates, never injects synthetic swaps,
and never changes, removes, interpolates or relabels an observation. It writes only below ``data/processed/m11-matching-v3/``, which holds task A, task B (interval withheld) and FOOTBALLTrace E0.
``m11-matching-v2`` (task A only) and ``m11-matching-v1`` (a superseded first pilot) are kept unchanged.

Step 4 (``ml``, scikit-learn imported lazily): leave-one-match-out pairwise identity classifiers on task B over development
matches, written to ``data/processed/m11-ml-v1/``:

    python -m football_fatigue.matching ml --matches 3812 3816 10506 3814

Step 5 (``ml2``): nested leave-one-match-out diagnosis, feature-family ablations, football-aware features, a bounded model
comparison and operating points, written to ``data/processed/m11-ml-v2/``.

Step 6 (``freeze``, ``valbuild``, ``predict``, ``score``; see :mod:`.validate`): the frozen classifier and one preregistered
evaluation on the frozen validation matches (``m11-frozen-v1``, ``m11-validation-v1``).
"""

from .baselines import BASELINES, BaselineConfig, run_baselines
from .benchmark import MatchingConfig, build_benchmark, run_match
from .evaluation import evaluate

__all__ = ["BASELINES", "BaselineConfig", "MatchingConfig", "build_benchmark", "evaluate", "run_baselines", "run_match"]
