"""Model-facing training views for M11 identity recovery at trajectory interactions (built from the immutable v3 store).

Source: ``m11-contacts-v1/<match>/`` (identity-free contacts; primary) or ``m11-intersections-v3/<match>/`` (same layout) — complete RAW trajectories of both players within +-2.5 s of each interaction
(``root.trajectories.npz``), identities (``root.references.jsonl``) and the quality audit (``root.eligibility.jsonl``). Nothing
there is modified; views are derived on the fly and deterministic for a seed.

**Interaction set.** ``eligible(rec, tau)``: time-aligned (|dt| <= tau) clean or uncertain candidates at the primary threshold;
``tier`` = ``clean`` or ``flagged`` (any uncertain reason), kept for stratified evaluation.

**Resampling.** Each player's *update instants* (provider repeats removed) are linearly interpolated onto a common grid of
``GRID_DT`` s relative to the interaction time ``t*``; a grid point is missing (NaN) unless it lies between two update instants at
most ``MAX_BRIDGE_S`` apart. Identical grids for both players remove the 15 Hz phase and held-position shortcuts.

**Scenario** (synthetic, never real occlusion): a dropout gap of ``gap_s`` centred on ``t*`` removes *both* players' positions;
optional i.i.d. Gaussian position noise ``noise_m`` on every observed point. Inputs: two *before* tracks (``t < -gap/2``) and two
anonymous *after* tracks (``t > gap/2``), each (T, 2) in metres with NaN for missing points, plus the shared time grids. The order
of the before tracks and, independently, of the after tracks comes from a seeded hash, so the label is balanced and unrelated to
slot, time or file order. **Label** ``y = 1`` when the assignment {before 0 -> after 0, before 1 -> after 1} is the provider's
continuity (``H0``), else 0 (``H1``: the crossed assignment). Identities never enter a sample.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

GRID_DT = 0.1
HALF_S = 2.5
MAX_BRIDGE_S = 0.5
GAPS_S = (0.4, 0.8, 1.2, 1.6, 2.0)
NOISES_M = (0.0, 0.15, 0.3)
PRIMARY_TAU = "0.20"
GRID = np.round(np.arange(-HALF_S, HALF_S + 1e-9, GRID_DT), 6)


@dataclass
class Interaction:
    """Both players' resampled positions on GRID (track order as stored), plus audit metadata (no identities)."""
    match_id: str
    event_id: str
    pos: np.ndarray                  # (2, len(GRID), 2) metres, NaN = missing
    tier: str                        # clean / flagged
    reasons: list[str]
    meta: dict[str, Any] = field(default_factory=dict)


def _h(seed: str, *parts) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, (seed, *parts))).encode()).digest()[:8], "little")


def eligible(rec: dict, tau: str = PRIMARY_TAU) -> bool:
    s = rec["status"][tau]
    return s["timing"] == "aligned" and s["category"] in ("clean_candidate", "uncertain_candidate")


def resample(t: np.ndarray, x: np.ndarray, y: np.ndarray, upd: np.ndarray, grid: np.ndarray = GRID) -> np.ndarray:
    """Linear interpolation of update instants onto ``grid`` (times relative to t*); NaN where not bridged."""
    k = upd & np.isfinite(x)
    tu, xu, yu = t[k], x[k], y[k]
    out = np.full((len(grid), 2), np.nan)
    if len(tu) < 2:
        return out
    j = np.searchsorted(tu, grid, "right")
    ok = (j > 0) & (j < len(tu))
    jj = np.clip(j, 1, len(tu) - 1)
    t0, t1 = tu[jj - 1], tu[jj]
    ok &= (t1 - t0) <= MAX_BRIDGE_S
    exact = np.isin(grid, tu)
    w = np.where(t1 > t0, (grid - t0) / np.where(t1 > t0, t1 - t0, 1), 0)
    out[:, 0] = np.where(ok, xu[jj - 1] + w * (xu[jj] - xu[jj - 1]), np.nan)
    out[:, 1] = np.where(ok, yu[jj - 1] + w * (yu[jj] - yu[jj - 1]), np.nan)
    if exact.any():                                              # a grid point on the last update instant
        idx = np.searchsorted(tu, grid[exact])
        out[exact] = np.c_[xu[idx], yu[idx]]
    return out


def load_match(v3_dir: Path, tau: str = PRIMARY_TAU, include=eligible) -> list[Interaction]:
    """Eligible interactions of one match from the v3 outputs (checksums verified against the v3 manifest)."""
    m = json.loads((v3_dir / "root.manifest.json").read_text(encoding="utf-8"))
    for name in ("root.trajectories.npz", "root.eligibility.jsonl"):
        if hashlib.sha256((v3_dir / name).read_bytes()).hexdigest() != m["outputs_sha256"][name]:
            raise ValueError(f"{v3_dir / name} does not match its manifest")
    recs = {}
    with open(v3_dir / "root.eligibility.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["in_trajectory_store"]:
                recs[r["event_id"]] = r
    with np.load(v3_dir / "root.trajectories.npz") as z:
        st = {k: z[k] for k in z.files}
    out = []
    for i, eid in enumerate(st["event_id"].tolist()):
        r = recs[eid]
        if not include(r, tau):
            continue
        a, b = st["offsets"][i], st["offsets"][i + 1]
        t = st["video_time_s"][a:b] - st["t_star_s"][i]
        pos = np.stack([resample(t, st["x"][a:b, k], st["y"][a:b, k], st["update_instant"][a:b, k]) for k in (0, 1)])
        s = r["status"][tau]
        mm = r["measurements"]
        est = [mm["window_a"]["estimated_share"], mm["window_b"]["estimated_share"]]
        meta = dict(abs_time_diff_s=mm["abs_time_diff_s"], timing_sigma_s=mm["timing_sigma_s"], angle_deg=mm["crossing_angle_deg"],
                    speed_min=min(mm["motion_a"]["speed_m_s"] or 0, mm["motion_b"]["speed_m_s"] or 0),
                    min_dist_m=mm["min_same_time_distance_m"], estimated_share=float(np.mean([e or 0 for e in est])),
                    category=s["category"], pair_id=r["pair_id"], cluster=r["candidate_cluster"][tau], period=r["period"],
                    crossing=bool(r.get("geometry", {}).get("geometric_crossing", True)),
                    aligned_crossing=bool(r.get("geometry", {}).get("aligned_crossing", True)))
        out.append(Interaction(str(r["match_id"]), eid, pos, "clean" if s["category"] == "clean_candidate" else "flagged", list(s["reasons"]), meta))
    return out


@dataclass
class Sample:
    before: np.ndarray               # (2, Tb, 2)
    after: np.ndarray                # (2, Ta, 2)
    t_before: np.ndarray             # (Tb,) s relative to t*
    t_after: np.ndarray              # (Ta,)
    gap_s: float
    noise_m: float
    y: int                           # 1 = H0 (b0->a0, b1->a1) is the provider continuity
    key: tuple                       # (match, event, gap, noise, draw) — bookkeeping only, never a model input


def make_sample(it: Interaction, gap_s: float, noise_m: float, seed: str = "m11-recovery-v1", draw: int = 0, shift_s: float = 0.0) -> Sample:
    """One scenario view of an interaction (see the module docstring). Deterministic for (seed, event, gap, noise, draw).

    ``shift_s`` (robustness check, a multiple of GRID_DT) centres the gap at t* + shift_s instead of the time of minimum distance;
    sample times are then relative to the gap centre, as at inference, so t* is not revealed."""
    h = _h(seed, it.event_id, gap_s, noise_m, draw)
    bo, ao = (h & 1, (h >> 1) & 1)                               # before order, after order
    bsel, asel = GRID < shift_s - gap_s / 2 - 1e-9, GRID > shift_s + gap_s / 2 + 1e-9
    pos = it.pos.copy()
    if noise_m > 0:
        rng = np.random.default_rng(h >> 2)
        pos = pos + rng.normal(0.0, noise_m, pos.shape)
    b_idx, a_idx = (1, 0) if bo else (0, 1), (1, 0) if ao else (0, 1)
    before, after = pos[list(b_idx)][:, bsel], pos[list(a_idx)][:, asel]
    y = int(b_idx[0] == a_idx[0])
    return Sample(before, after, np.round(GRID[bsel] - shift_s, 6), np.round(GRID[asel] - shift_s, 6), gap_s, noise_m, y,
                  (it.match_id, it.event_id, gap_s, noise_m, draw))
