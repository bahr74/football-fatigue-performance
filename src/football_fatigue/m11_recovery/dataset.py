"""Scenario tables: every eligible interaction under every synthetic scenario, with baseline margins, engineered features
and normalised sequence tensors (model inputs only; identities never enter).

Rows: one per (interaction, gap, noise). Scenarios whose sample has a track with fewer than 3 observed points are not usable
(no motion fit is defined) and are counted per scenario, not silently dropped. Sequence tensors are expressed in a frame
computed from the *before* tracks only (origin: midpoint of their last points; x axis: direction of their summed velocity),
so the normalisation cannot carry the label.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from . import motion as MO
from .views import GAPS_S, GRID, NOISES_M, Interaction, make_sample

SCENARIOS = tuple((g, n) for g in GAPS_S for n in NOISES_M)
T_PAD = int(np.sum(GRID < -GAPS_S[0] / 2 - 1e-9))         # longest before / after track
META_KEYS = ("abs_time_diff_s", "timing_sigma_s", "angle_deg", "speed_min", "min_dist_m", "estimated_share")


def frame(s, k: MO.Kin) -> tuple[np.ndarray, np.ndarray]:
    origin = 0.5 * (k.last(0) + k.last(1))
    v = MO._vel(k.b_cv[0]) + MO._vel(k.b_cv[1])
    th = np.arctan2(v[1], v[0]) if np.hypot(*v) > 0.3 else 0.0
    c, si = np.cos(-th), np.sin(-th)
    return origin, np.array([[c, -si], [si, c]])


def tensor(tr: np.ndarray, t: np.ndarray, origin, R, reverse: bool) -> np.ndarray:
    out = np.zeros((T_PAD, 4), np.float32)
    ok = np.isfinite(tr[:, 0])
    p = np.where(ok[:, None], (tr - origin) @ R.T, 0.0)
    rows = np.c_[p, ok.astype(float), t]
    if reverse:                                               # before tracks are reversed so both sequences start at the gap
        rows = rows[::-1]
    rows = rows[:T_PAD]                                        # shifted views only: keep the points nearest the gap
    out[:len(rows)] = rows
    return out


def model_inputs(s) -> dict[str, np.ndarray]:
    """Everything a model sees for one sample (training and inference share this): gap, baseline margins, features, pair
    physics and the two sequence tensors."""
    k = MO.Kin(s)
    pair, ctx = MO.pair_matrices(s, k)
    origin, R = frame(s, k)
    return dict(gap=s.gap_s, margins=[MO.margin(pair[f"d_{b}"]) for b in MO.BASELINES], feats=MO.features(s, pair, ctx),
                phys=np.stack([pair[p] for p in MO.PAIR_PHYS], -1),
                bseq=np.stack([tensor(s.before[i], s.t_before, origin, R, True) for i in (0, 1)]),
                aseq=np.stack([tensor(s.after[j], s.t_after, origin, R, False) for j in (0, 1)]))


SHIFTS_S = tuple(np.round(np.arange(-0.4, 0.41, 0.1), 6))


def shift_for(event_id: str, gap_s: float, noise_m: float) -> float:
    """Seeded gap-centre shift (robustness views only): uniform over SHIFTS_S."""
    from .views import _h

    return float(SHIFTS_S[_h("m11-recovery-v1-shift", event_id, gap_s, noise_m) % len(SHIFTS_S)])


def build(interactions: list[Interaction], seed: str = "m11-recovery-v1", scenarios=SCENARIOS, shifted: bool = False) -> tuple[dict[str, np.ndarray], dict]:
    cols: dict[str, list] = {k: [] for k in ("match", "event", "tier", "gap", "noise", "y", "margins", "feats", "phys", "bseq", "aseq",
                                             "pair_id", "cluster", "reasons", "crossing", "aligned_crossing", *META_KEYS)}
    unusable = Counter()
    for it in interactions:
        for g, n in scenarios:
            s = make_sample(it, g, n, seed, shift_s=shift_for(it.event_id, g, n) if shifted else 0.0)
            if not MO.usable(s):
                unusable[f"{g}|{n}"] += 1
                continue
            for key, v in model_inputs(s).items():
                cols[key].append(v)
            cols["match"].append(it.match_id)
            cols["event"].append(it.event_id)
            cols["tier"].append(it.tier)
            cols["noise"].append(n)
            cols["y"].append(s.y)
            cols["pair_id"].append(it.meta["pair_id"])
            cols["cluster"].append(it.meta["cluster"] or "")
            cols["reasons"].append(",".join(it.reasons))
            cols["crossing"].append(it.meta["crossing"])
            cols["aligned_crossing"].append(it.meta["aligned_crossing"])
            for m in META_KEYS:
                v = it.meta[m]
                cols[m].append(np.nan if v is None else v)
    out = {k: np.asarray(v) for k, v in cols.items()}
    for k in ("feats", "margins", "phys"):
        out[k] = out[k].astype(np.float64)
    return out, dict(rows=len(out["y"]), unusable=dict(unusable), interactions=len(interactions), feature_names=list(MO.FEATURE_NAMES))


def concat(tables: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.concatenate([t[k] for t in tables]) for k in tables[0]}


def take(t: dict[str, np.ndarray], idx) -> dict[str, np.ndarray]:
    return {k: v[idx] for k, v in t.items()}
