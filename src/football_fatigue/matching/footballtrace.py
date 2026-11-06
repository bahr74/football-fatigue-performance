"""FOOTBALLTrace trajectory correction (Mostafa et al., icSPORTS 2023, ``data/raw/122268.pdf``): E0-Literal and E0-Adapted.

**What the paper specifies (section 3.3, section 3.4, Algorithm 1, section 3.5, Table 2):**
- **Occlusion candidates:** 50-frame (2 s at 25 fps) sliding windows with 50% overlap. A pair qualifies when its distance is
  ≤ τ for at least 50% of the window. τ = 4 is in pitch top-view (homography image) units; no scale to metres is given. Step 1
  (``encounters.catalogue``) implements this rule in seconds and metres.
- **Manual confirmation:** an operator confirms every candidate before correction. This cannot be reproduced here.
- **Input to correction:** the *ID-labelled* tracker trajectories of the occluded and occluding players in a 5 s window around
  the occlusion.
- **Pre-processing:** a Savitzky-Golay filter (window w, order n), then resampling with step s. Chosen values: w = 7, n = 2,
  s = 7 (majority vote; the grids were w ∈ {5, 7, 9}, n ∈ {2, 3, 4}, s ∈ {5, 7, 9, 10}). All are in samples of a 25 fps track.
- **Breakpoint:** for every interior resampled point P(i), with v1 = P(i) − P(i−1) and v2 = P(i) − P(i+1), Algorithm 1
  stores ``dot(v1, v2) / (|v1| |v2|)`` (Equation 1 calls this θ; it is the *cosine*) and takes the index of the *minimal* value.
  The text says this is the sharp turn, the moment of occlusion. These two statements disagree: the minimal cosine is the
  *straightest* point, while a sharp turn has the minimal geometric angle, which is the *maximal* cosine. Both readings are
  implemented: ``rule="min_cosine"`` (the pseudocode as written) and ``rule="min_angle"`` (the stated intent).
- **Correction:** split both trajectories at their breakpoints and connect T_before of the occluded player to T_after of the
  occluding player, and vice versa. The tails are *always* exchanged. Whether a swap happened at all is left to the operator.

**E0-Literal** (:func:`literal_correct`) applies exactly this to the two provider-labelled trajectories of a 2-player event in a
5 s window centred on the encounter, using the RAW samples at the file's 29.97 fps, 15 Hz repeats included, with w, n and s in
samples. It needs ID-labelled trajectories through the encounter. Task A hides those labels and task B withholds the interval,
so E0-Literal is incompatible with both and is run as a separate evaluation-side protocol ("L") on the reference trajectories.
Because it always exchanges tails, its output disagrees with the provider reference by construction whenever it finds two
breakpoints. Only its breakpoint localisation can be measured.

**E0-Adapted** (:func:`adapted`, task B compatible; on task A it ignores the interval points) turns the paper's premise, that a
wrong connection shows up as a sharp turn, into an association cost:
- each tracklet's update instants are linearly resampled to the paper's 25 Hz clock (a temporary algorithm-stage
  representation; RAW observations are untouched);
- that track is smoothed with SG (w = 7, n = 2) and resampled with step s = 7 (0.28 s, as in the paper), anchored at the
  junction;
- a hypothesis incoming i → outgoing j costs the sharpness ``(1 + cos) / 2`` of the sharpest junction turn (0 straight,
  1 reversal);
- an exact one-to-one assignment with abstention follows, plus a physical-speed feasibility gate.

This is FOOTBALLTrace-*inspired*. It is not the published method.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FootballTraceConfig:
    w: int = 7                 # SG window (samples), paper majority vote
    n: int = 2                 # SG polynomial order
    s: int = 7                 # resampling step (samples)
    paper_fps: float = 25.0    # the paper's frame rate; E0-Adapted resamples to this clock
    window_s: float = 5.0      # E0-Literal: analysis window around the occlusion
    max_speed_m_s: float = 12.0   # E0-Adapted feasibility gate (not in the paper)
    margin: float = 0.1        # E0-Adapted: alternative assignment within this sharpness: UNCERTAIN (a priori, not in the paper)
    neutral_cost: float = 0.5  # E0-Adapted: no turn measurable at the junction (zero-length vector, too few points)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def savgol(y: np.ndarray, w: int, n: int) -> np.ndarray:
    """Savitzky-Golay smoothing with polynomial edge fits (the behaviour of ``scipy.signal.savgol_filter(mode="interp")``).
    Returns a copy; tracks shorter than ``w`` are returned unsmoothed."""
    y = np.asarray(y, float)
    if w % 2 == 0 or n >= w:
        raise ValueError("SG needs an odd window larger than the polynomial order")
    if len(y) < w:
        return y.copy()
    h = w // 2
    A = np.vander(np.arange(-h, h + 1, dtype=float), n + 1, increasing=True)
    coef = np.linalg.pinv(A)[0]                                 # value of the fitted polynomial at the window centre
    out = np.convolve(y, coef[::-1], mode="same")
    for sl, xs in ((slice(0, w), np.arange(h)), (slice(len(y) - w, len(y)), np.arange(h + 1, w))):
        p = np.polyfit(np.arange(w, dtype=float), y[sl], n)
        out[sl.start + xs] = np.polyval(p, xs.astype(float))
    return out


def turn_cosines(P: np.ndarray) -> np.ndarray:
    """Algorithm 1: cos between v1 = P(i) - P(i-1) and v2 = P(i) - P(i+1) for interior points (NaN when a vector is zero).
    The value is -1 for a straight line and +1 for a full reversal."""
    if len(P) < 3:
        return np.zeros(0)
    v1, v2 = P[1:-1] - P[:-2], P[1:-1] - P[2:]
    nn = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(nn > 0, (v1 * v2).sum(axis=1) / nn, np.nan)


def literal_breakpoint(x: np.ndarray, y: np.ndarray, cfg: FootballTraceConfig = FootballTraceConfig(), rule: str = "min_angle") -> int | None:
    """Sample index (into ``x``/``y``) of the breakpoint, or None when fewer than 3 resampled points have a defined turn."""
    P = np.column_stack([savgol(x, cfg.w, cfg.n), savgol(y, cfg.w, cfg.n)])[::cfg.s]
    cos = turn_cosines(P)
    if not np.isfinite(cos).any():
        return None
    if rule == "min_cosine":
        k = int(np.nanargmin(cos))
    elif rule == "min_angle":
        k = int(np.nanargmax(cos))
    else:
        raise ValueError(f"unknown rule {rule}")
    return (k + 1) * cfg.s                                      # interior point k is resampled index k + 1


def literal_correct(ta: dict[str, np.ndarray], tb: dict[str, np.ndarray], cfg: FootballTraceConfig = FootballTraceConfig(),
                    rule: str = "min_angle") -> dict[str, Any]:
    """Algorithm 1 on two labelled trajectories ({"t", "x", "y"}): split each at its breakpoint and exchange the tails.
    Returns the breakpoint times and the corrected trajectories as (source label, sample index range) pieces."""
    ka = literal_breakpoint(ta["x"], ta["y"], cfg, rule)
    kb = literal_breakpoint(tb["x"], tb["y"], cfg, rule)
    if ka is None or kb is None:
        return dict(corrected=False, break_a_s=None, break_b_s=None, a=[("a", 0, len(ta["t"]))], b=[("b", 0, len(tb["t"]))])
    return dict(corrected=True, break_a_s=float(ta["t"][ka]), break_b_s=float(tb["t"][kb]),
                a=[("a", 0, ka), ("b", kb, len(tb["t"]))], b=[("b", 0, kb), ("a", ka, len(ta["t"]))])


def protocol_l(c, references: list[dict], cfg: FootballTraceConfig = FootballTraceConfig()) -> list[dict]:
    """Evaluation-side E0-Literal: Algorithm 1 on the *provider-labelled* RAW trajectories of every 2-player instance.

    This reads identity labels (it is the paper's input), so it is never a task A or task B competitor. A pair is reported as
    agreeing with the reference only when no correction is applied: an applied correction always moves each tail to the other
    label."""
    from football_fatigue.encounters.trajectories import STATE_AVAILABLE

    t = c.video_time_s
    out = []
    for r in references:
        if r["n_participants"] != 2 or len(r["incoming"]) != 2:
            continue
        slots = [v["slot"] for v in r["incoming"].values()]
        (b0, s), (_, e), (_, a1) = r["rows"]["before"], r["rows"]["during"], r["rows"]["after"]
        mid = (t[s] + t[e - 1]) / 2
        rows = np.arange(b0, a1)
        rows = rows[np.abs(t[rows] - mid) <= cfg.window_s / 2]
        tr = []
        for sl in slots:
            rr = rows[c.state[rows, sl] == STATE_AVAILABLE]
            tr.append(dict(t=t[rr] - t[s], x=c.x[rr, sl].copy(), y=c.y[rr, sl].copy()))
        amb = float(t[e - 1] - t[s])
        tol = cfg.s / c.fps                                     # one resampling step
        row = dict(instance_id=r["instance_id"], tier=r["tier"], ambiguity_s=round(amb, 4), samples=[len(x["t"]) for x in tr])
        for rule in ("min_angle", "min_cosine"):
            res = literal_correct(tr[0], tr[1], cfg, rule)
            br = [b for b in (res["break_a_s"], res["break_b_s"]) if b is not None]
            row[rule] = dict(corrected=res["corrected"], break_s=[None if b is None else round(b, 4) for b in (res["break_a_s"], res["break_b_s"])],
                             breaks_inside_interval=sum(-tol <= b <= amb + tol for b in br),
                             reference_agreement=not res["corrected"])
        out.append(row)
    return out


def summarize_protocol_l(rows: list[dict]) -> dict[str, Any]:
    def agg(rs: list[dict]) -> dict[str, Any]:
        res: dict[str, Any] = dict(pairs=len(rs))
        for rule in ("min_angle", "min_cosine"):
            corr = sum(r[rule]["corrected"] for r in rs)
            breaks = 2 * corr
            res[rule] = dict(corrected=corr, reference_agreement_rate=round(sum(r[rule]["reference_agreement"] for r in rs) / len(rs), 4) if rs else None,
                             breakpoints=breaks, breakpoints_inside_interval=sum(r[rule]["breaks_inside_interval"] for r in rs if r[rule]["corrected"]),
                             breakpoint_inside_share=round(sum(r[rule]["breaks_inside_interval"] for r in rs if r[rule]["corrected"]) / breaks, 4) if breaks else None)
        return res
    tiers = sorted({r["tier"] for r in rows})
    return dict(all=agg(rows), by_tier={k: agg([r for r in rows if r["tier"] == k]) for k in tiers},
                note="E0-Literal reads provider-labelled trajectories and always exchanges tails; agreement is 0 whenever it corrects")


# ---------------------------------------------------------------- E0-Adapted (task B compatible)

def paper_clock_track(tr: dict[str, list], cfg: FootballTraceConfig, *, anchor_end: bool) -> np.ndarray:
    """Update instants linearly resampled to the paper's 25 Hz clock, SG-smoothed, then resampled with step s so that the
    junction end (the last sample of an incoming tracklet, the first of an outgoing one) is kept. A temporary representation."""
    t, x, y = (np.asarray(tr[k], float) for k in ("t", "x", "y"))
    upd = np.r_[True, (np.diff(x) != 0) | (np.diff(y) != 0)]
    upd[-1] = True                                              # keep the time extent of a held tail
    tu, xu, yu = t[upd], x[upd], y[upd]
    if len(tu) < 2 or tu[-1] - tu[0] <= 0:
        return np.column_stack([x[-1:] if anchor_end else x[:1], y[-1:] if anchor_end else y[:1]])
    step = 1.0 / cfg.paper_fps
    k = np.arange(int(np.floor((tu[-1] - tu[0]) / step + 1e-9)) + 1) * step
    tg = (tu[-1] - k[::-1]) if anchor_end else (tu[0] + k)
    G = np.column_stack([savgol(np.interp(tg, tu, xu), cfg.w, cfg.n), savgol(np.interp(tg, tu, yu), cfg.w, cfg.n)])
    return G[::-1][::cfg.s][::-1] if anchor_end else G[::cfg.s]


def junction_sharpness(inc: np.ndarray, out: np.ndarray, cfg: FootballTraceConfig) -> float:
    """Sharpness (1 + cos) / 2 of the sharpest turn at the two junction points of incoming ⊕ outgoing."""
    Q = np.vstack([inc[-2:], out[:2]])
    k0 = len(inc[-2:]) - 1                                      # the incoming end within Q
    cos = turn_cosines(Q)
    piv = [i - 1 for i in (k0, k0 + 1) if 1 <= i <= len(Q) - 2]
    vals = [cos[i] for i in piv if np.isfinite(cos[i])]
    return float((1 + max(vals)) / 2) if vals else cfg.neutral_cost


def adapted(inst: dict, cfg: FootballTraceConfig = FootballTraceConfig()) -> list[dict]:
    from .baselines import _decision, solve

    I, O = inst["incoming"], inst["outgoing"]
    tin = [paper_clock_track(a, cfg, anchor_end=True) for a in I]
    tout = [paper_clock_track(b, cfg, anchor_end=False) for b in O]
    C = np.full((len(I), len(O)), np.inf)
    for i, a in enumerate(I):
        for j, b in enumerate(O):
            dt = b["t"][0] - a["t"][-1]
            d = float(np.hypot(b["x"][0] - a["x"][-1], b["y"][0] - a["y"][-1]))
            if dt > 0 and d <= cfg.max_speed_m_s * dt:
                C[i, j] = junction_sharpness(tin[i], tout[j], cfg)
    abstain = np.ones(len(I))                                   # costlier than any feasible connection
    total, pick = solve(C, abstain)
    out = []
    for i, a in enumerate(I):
        j = pick[i]
        if j < 0:
            out.append(_decision(a["tracklet_id"], "UNMATCHED", None, None, None, None))
            continue
        margin = solve(C, abstain, forbid=(i, j))[0] - total
        oid = O[j]["tracklet_id"]
        out.append(_decision(a["tracklet_id"], "UNCERTAIN" if margin < cfg.margin else "MATCHED", oid, oid, C[i, j], margin))
    return out
