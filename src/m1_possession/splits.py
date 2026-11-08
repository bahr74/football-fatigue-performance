"""Locked match split for M1 (redrawn 2026-11-06, v2). Pure data; no tags.

Why v2: labels v1 showed that in 18 matches the tracking file puts the tagged ball carrier more than 10 m from the ball in
over 10 % of CONTROL frames (tracker identity switches and estimated player tracks; see the plan's Findings log), and the
ten known coordinate-disagreement matches are among them. Those 18 form the STRESS set: never used for learning, reported
separately at the end. The other 46 are clean.

v2: 16 test matches drawn at random (seed 20261107) from the 38 clean matches outside the dev set; the other 30 clean
matches are the learning set, permuted with the same generator and dealt into 5 folds of 6 (fold k = matches k, k+5, ...).
All 8 dev matches are clean and stay in the learning set. The test matches are run once, at the very end.
"""
from __future__ import annotations

SPLIT_SEED = 20261107
SPLIT_VERSION = "v2"

ALL_MATCHES = (3812, 3813, 3814, 3815, 3816, 3817, 3818, 3819, 3820, 3821, 3822, 3823, 3824, 3825, 3826, 3827, 3828, 3829,
               3830, 3831, 3832, 3833, 3834, 3835, 3836, 3837, 3838, 3839, 3840, 3841, 3842, 3843, 3844, 3845, 3846, 3847,
               3848, 3849, 3850, 3851, 3852, 3853, 3854, 3855, 3856, 3857, 3858, 3859, 10502, 10503, 10504, 10505, 10506,
               10507, 10508, 10509, 10510, 10511, 10512, 10513, 10514, 10515, 10516, 10517)
DEV_MATCHES = (3812, 3813, 3814, 3815, 3818, 3819, 3820, 3821)
STRESS_MATCHES = (3816, 3817, 3826, 3829, 3835, 3837, 3841, 3845, 3846, 3847, 3852, 3853, 3859, 10502, 10504, 10512,
                  10514, 10516)

TEST_MATCHES = (3822, 3825, 3827, 3828, 3834, 3836, 3842, 3844, 3848, 10503, 10505, 10508, 10509, 10510, 10511, 10513)
FOLDS = (
    (3813, 3815, 3818, 3819, 3849, 3858),
    (3820, 3821, 3823, 3831, 10506, 10517),
    (3824, 3832, 3833, 3843, 3851, 3855),
    (3814, 3839, 3854, 3856, 3857, 10515),
    (3812, 3830, 3838, 3840, 3850, 10507),
)
LEARNING_MATCHES = tuple(sorted(m for f in FOLDS for m in f))

# v1 (seed 20261106, 16 test from 56 non-dev, 6 folds of 8) was withdrawn: 5 of its test matches were stress matches.


def make_split(all_matches=ALL_MATCHES, dev=DEV_MATCHES, stress=STRESS_MATCHES, seed=SPLIT_SEED, n_test=16, n_folds=5):
    """Re-draw the split (used by the test to prove the constants above are reproducible)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    clean = [m for m in all_matches if m not in stress]
    pool = sorted(m for m in clean if m not in dev)
    test = tuple(sorted(int(m) for m in rng.choice(pool, n_test, replace=False)))
    learn = [m for m in sorted(clean) if m not in test]
    perm = [int(m) for m in rng.permutation(learn)]
    folds = tuple(tuple(sorted(perm[i::n_folds])) for i in range(n_folds))
    return test, folds


def fold_of(match_id: int) -> int | None:
    """Fold index of a learning match; None for a test or stress match."""
    for k, f in enumerate(FOLDS):
        if int(match_id) in f:
            return k
    return None
