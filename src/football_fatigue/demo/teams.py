"""M7.2 opt-in display/analytics team assignment: windowed tracklet votes plus a touchline-person rule.

Evidence (``data/processed/vision_eval/m72-sprint-v1``):

* ``tw_labels`` uses only the existing per-box colour votes, has no fitted parameter, and is applied per tracklet.
  It cut the outfield wrong-team rate on all four development matches, at the cost of more Unknown.
* ``touchline_tracklets`` targets the people outside the player reference who were labelled Blue in 117093.
  They stood a median 0.46 m from a touchline: assistant referees and staff.
  The rule was designed after that diagnosis and can only be checked on 117093, the one calibrated match, so it is not validated.

Neither function caps a team at 11 people; uncertain boxes become Unknown.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from football_fatigue.vision.team_appearance import aggregate_votes

LABELS = ("team_0", "team_1", "other")


def window_votes(frames: Sequence[int], votes: Sequence[str | None], window: int = 25, min_votes: int = 5,
                 min_share: float = 0.7) -> list[tuple[str | None, float]]:
    """Per box of ONE tracklet (sorted by frame), the decided votes within +-window frames: (label or None, share)."""
    f = np.asarray(frames)
    cum = np.zeros((len(votes) + 1, len(LABELS)), int)
    for j, v in enumerate(votes):
        cum[j + 1] = cum[j]
        if v in LABELS:
            cum[j + 1, LABELS.index(v)] += 1
    lo, hi = np.searchsorted(f, f - window, "left"), np.searchsorted(f, f + window, "right")
    out = []
    for a, b in zip(lo, hi):
        c = cum[b] - cum[a]
        n, k = int(c.sum()), int(np.argmax(c))
        out.append((LABELS[k], round(c[k] / n, 3)) if n >= min_votes and c[k] / n >= min_share else (None, 0.0))
    return out


def tw_labels(frames: Sequence[int], votes: Sequence[str | None], **kw) -> list[tuple[str | None, float]]:
    """Window label kept only where the whole-tracklet vote is decided and agrees; otherwise (None, 0)."""
    whole = aggregate_votes(votes, kw.get("min_votes", 5), kw.get("min_share", 0.7)).label
    return [(lab, s) if whole is not None and lab == whole else (None, 0.0) for lab, s in window_votes(frames, votes, **kw)]


def touchline_tracklets(tracklets: dict[int, Sequence[float]], min_boxes: int = 25, median_m: float = 2.0, band_m: float = 4.0,
                        band_share: float = 0.8) -> set[int]:
    """Tracklets that stay on a touchline (``tracklets``: id -> per-box pitch y in metres, 0..68)."""
    out = set()
    for t, ys in tracklets.items():
        y = np.asarray(ys, float)
        d = np.minimum(np.abs(y), np.abs(68.0 - y))
        if len(d) >= min_boxes and np.median(d) <= median_m and (d <= band_m).mean() >= band_share:
            out.add(t)
    return out
