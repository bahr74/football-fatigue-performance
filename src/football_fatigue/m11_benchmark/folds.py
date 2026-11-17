"""Frozen grouped folds over the 32 train/CV matches: 8 outer folds of 4 matches by ``sha256(FOLD_SALT | match)`` rank, and
deterministic inner grouped folds / early-stopping matches inside every outer training partition."""

from __future__ import annotations

import hashlib

import numpy as np

FOLD_SALT = "m11-stage3-folds-v1"
N_OUTER = 8
N_INNER = 3


def _rank(matches, salt: str) -> list[str]:
    return sorted((str(m) for m in matches), key=lambda m: hashlib.sha256(f"{salt}|{m}".encode()).hexdigest())


def outer_folds(matches) -> dict[str, int]:
    r = _rank(matches, FOLD_SALT)
    if len(r) % N_OUTER:
        raise ValueError("match count is not a multiple of the fold count")
    return {m: i % N_OUTER for i, m in enumerate(r)}


def inner_folds(train_matches, outer: int, k: int = N_INNER) -> dict[str, int]:
    r = _rank(train_matches, f"{FOLD_SALT}|inner|{outer}")
    return {m: i % k for i, m in enumerate(r)}


def early_stop_matches(train_matches, outer: int, n: int = 3) -> list[str]:
    """Matches of the outer training partition reserved for early stopping of neural models (never the outer validation fold)."""
    return _rank(train_matches, f"{FOLD_SALT}|early|{outer}")[:n]


def fold_index(groups: np.ndarray, fmap: dict[str, int]) -> np.ndarray:
    return np.array([fmap[str(int(g))] for g in groups])
