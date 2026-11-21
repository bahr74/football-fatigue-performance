"""M11 Milestone 2: reproducible new-scenario generator from real FIFA trajectories (32 train/CV matches only).

**Population.** Encounters are Stage 2 candidates of the train/CV matches that are ACCEPTED at the primary 0.8 s gap and belong
to the independent subset of Stage 3 ``g0p80``. ``N_PER_MATCH`` of them are drawn per match with a seed derived from the
suite name and the match id (no model output is involved in the selection). Every encounter is a close approach of two players
(Stage 2 radius 0.5 m): clearly separated pairs are not part of the population.

**Design.** One factor at a time around the baseline (0.8 s gap, centred on the closest approach, clean positions, full
observations, 2.0 s context):

- ``gap``: 0.2, 0.4, 0.6, 1.0, 1.2, 1.6, 2.0, 2.4 s;
- ``shift``: gap centre moved by -0.5, -0.3, -0.1, +0.1, +0.3, +0.5 s (training used up to +-0.3 s);
- ``noise``: i.i.d. Gaussian noise on every observed x and y, sigma 1, 2, 5, 10, 25, 50 cm;
- ``dropout``: each observation removed with probability 10 / 30 / 50 % (``random10/30/50``), one contiguous 0.5 s run removed
  per tracklet (``burst0p5``), or every other observation removed (``rate7p5``, 7.5 Hz);
- ``context``: only the last / first 1.5, 1.0 or 0.5 s of observations before / after the gap are kept.

**Labels (never fabricated).** The pairing comes from the two players' own continuous provider chains, as in Stage 3. For every
``gap`` and ``shift`` variant the unchanged Stage 2 label-quality procedure (``candidates.label_quality``) is re-run at the new
gap and centre; variants that are not ACCEPTED are recorded as *invalid* with their reasons and never reach a model. Every built
variant is re-verified against the raw provider positions (``samples.verify_label``). ``noise``, ``dropout`` and ``context``
variants perturb or remove observations of the accepted baseline geometry only; they never add or move hidden-gap data.

**Model-visible content.** Only observations strictly before ``g0`` (incoming) or strictly after ``g1`` (outgoing) of the
tracklets are emitted; the adapter then applies the Stage 3 grid (no extrapolation). Order of the two incoming and of the two
outgoing tracklets is drawn from the scenario id (independent of Stage 3's order bits).

**Determinism.** ``scenario_id = sha256(SUITE | candidate | factor | level)[:16]``; every random draw of a scenario uses a
generator seeded from its id, so a scenario can be regenerated exactly from (match, candidate, factor, level).

These are synthetic interruptions of real trajectories in development matches, evaluated with unseen-match fold models. They are
development stress cases, not independent real occlusion events or independent matches.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from football_fatigue.m11_infer import adapter as A
from football_fatigue.m11_infer import bundle as B
from football_fatigue.window_intersections import candidates as C
from football_fatigue.window_intersections import samples as S

SUITE = "m11-scenarios-v1"
N_PER_MATCH = 60
BASE = dict(gap=0.8, shift=0.0, noise=0.0, dropout="none", context=2.0)
FACTORS: dict[str, list[Any]] = {
    "gap": [0.2, 0.4, 0.6, 1.0, 1.2, 1.6, 2.0, 2.4],
    "shift": [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5],
    "noise": [0.01, 0.02, 0.05, 0.10, 0.25, 0.50],
    "dropout": ["random10", "random30", "random50", "burst0p5", "rate7p5"],
    "context": [1.5, 1.0, 0.5],
}
VARIANTS: list[tuple[str, Any]] = [("baseline", "base")] + [(f, v) for f, lv in FACTORS.items() for v in lv]
MARGIN_S = 1.0                      # extra observations beyond the 2 s context so that grid interpolation is bracketed (as Stage 3)
HIDDEN_LEN = 40                     # hidden-path samples stored for display (>= 2.4 s * 15 Hz + 1)
DT = S.DT


def _h(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


def level_key(level: Any) -> str:
    return f"{level:g}" if isinstance(level, float) else str(level)


def scenario_id(candidate_id: str, factor: str, level: Any) -> str:
    return _h(SUITE, candidate_id, factor, level_key(level))[:16]


def parse_level(factor: str, key: str) -> Any:
    """Inverse of ``level_key`` for a factor's declared levels."""
    if factor == "baseline":
        return "base"
    for v in FACTORS[factor]:
        if level_key(v) == str(key):
            return v
    raise ValueError(f"unknown level {key!r} of {factor}")


def params(factor: str, level: Any) -> dict[str, Any]:
    p = dict(BASE)
    if factor != "baseline":
        if factor not in FACTORS or level not in FACTORS[factor]:
            raise ValueError(f"unknown variant {factor}={level}")
        p[factor] = level
    return p


def order_bits(sid: str) -> tuple[int, int]:
    h = int(_h(SUITE, "order", sid)[:8], 16)
    return h & 1, (h >> 1) & 1


def select_encounters(meta: pd.DataFrame, match_id: str, n: int = N_PER_MATCH) -> pd.DataFrame:
    """Seeded selection of independent Stage 3 ``g0p80`` encounters of one match (no model output involved)."""
    m = meta[meta["independent"].astype(bool)].sort_values("candidate_id").reset_index(drop=True)
    rng = np.random.default_rng(int(_h(SUITE, "select", match_id)[:16], 16))
    k = rng.choice(len(m), size=min(n, len(m)), replace=False)
    return m.iloc[np.sort(k)].reset_index(drop=True)


# ---------------------------------------------------------------- perturbations (on observations only)

def cut(V, g0: float, g1: float, side: str, context: float) -> A.Track:
    """Observations of a chain strictly before g0 (in) / after g1 (out) within the context (+ bracketing margin at full context)."""
    t, x, y = V
    keep = context + (MARGIN_S if context >= S.CONTEXT_S - 1e-9 else 0.0)
    k = ((t >= g0 - keep - 1e-9) & (t < g0)) if side == "in" else ((t > g1) & (t <= g1 + keep + 1e-9))
    return A.Track(t[k].astype(np.float64), x[k].astype(np.float64), y[k].astype(np.float64))


def drop(tr: A.Track, mode: str, side: str, rng: np.random.Generator) -> A.Track:
    n = len(tr.t)
    if mode == "none" or n == 0:
        return tr
    if mode.startswith("random"):
        keep = rng.random(n) >= int(mode[len("random"):]) / 100.0
    elif mode == "burst0p5":
        lo, hi = tr.t[0], tr.t[-1]
        a = lo + rng.random() * max(0.0, hi - lo - 0.5)
        keep = ~((tr.t >= a) & (tr.t < a + 0.5))
    elif mode == "rate7p5":
        i = np.arange(n)
        keep = ((n - 1 - i) % 2 == 0) if side == "in" else (i % 2 == 0)      # the observation nearest the gap is kept
    else:
        raise ValueError(mode)
    return A.Track(tr.t[keep], tr.x[keep], tr.y[keep])


def add_noise(tr: A.Track, sigma: float, rng: np.random.Generator) -> A.Track:
    if sigma <= 0 or len(tr.t) == 0:
        return tr
    return A.Track(tr.t, tr.x + rng.normal(0, sigma, len(tr.t)), tr.y + rng.normal(0, sigma, len(tr.t)))


# ---------------------------------------------------------------- descriptive families (evaluation only, never model input)

def _interp(V, t):
    return np.c_[np.interp(t, V[0], V[1]), np.interp(t, V[0], V[2])]


def _vel(V, t0: float, t1: float) -> np.ndarray:
    k = (V[0] >= t0 - 1e-9) & (V[0] <= t1 + 1e-9)
    if k.sum() < 2:
        return np.full(2, np.nan)
    tt = V[0][k] - V[0][k].mean()
    return np.array([np.polyfit(tt, V[1][k], 1)[0], np.polyfit(tt, V[2][k], 1)[0]])


def _ang(u, v) -> float:
    nu, nv = np.hypot(*u), np.hypot(*v)
    if not (nu > 0 and nv > 0) or not (np.isfinite(nu) and np.isfinite(nv)):
        return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / (nu * nv), -1, 1))))


def families(VA, VB, tc: float, g: float = 0.8) -> dict[str, Any]:
    """Movement descriptors of the encounter at the baseline geometry (clean provider chains; descriptive only)."""
    g0, g1 = tc - g / 2, tc + g / 2
    va_in, vb_in = _vel(VA, g0 - 0.5, g0), _vel(VB, g0 - 0.5, g0)
    va_out, vb_out = _vel(VA, g1, g1 + 0.5), _vel(VB, g1, g1 + 0.5)
    sa, sb = np.hypot(*va_in), np.hypot(*vb_in)
    rel = _ang(va_in, vb_in)
    if not (sa >= 0.5 and sb >= 0.5):
        heading = "slow_player"
    else:
        heading = "same_direction" if rel < 45 else ("opposite_direction" if rel > 135 else "crossing_angle")
    d = lambda t: float(np.hypot(*(_interp(VA, [t]) - _interp(VB, [t]))[0]))  # noqa: E731
    d_b, d_0, d_1, d_a = d(g0 - 1.0), d(g0), d(g1), d(g1 + 1.0)
    approach, separate = d_b - d_0, d_a - d_1
    if approach > 0.3 and separate > 0.3:
        trend = "converge_then_diverge"
    elif approach > 0.3:
        trend = "converging_stay_close"
    elif separate > 0.3:
        trend = "close_then_diverge"
    else:
        trend = "parallel_close"
    turn = np.nanmax([_ang(va_in, va_out), _ang(vb_in, vb_out)])
    turn_c = "unknown" if not np.isfinite(turn) else ("straight" if turn < 30 else ("turn_30_90" if turn <= 90 else "sharp_turn_gt90"))
    spd = [(np.hypot(*a), np.hypot(*b)) for a, b in ((va_in, va_out), (vb_in, vb_out))]
    rchg = max(abs(o - i) / max(i, o, 0.5) for i, o in spd)
    return dict(heading_relation=heading, relative_heading_deg=round(rel, 1) if np.isfinite(rel) else None, distance_trend=trend,
                max_heading_change_deg=round(float(turn), 1) if np.isfinite(turn) else None, heading_change=turn_c,
                speed_change="speed_change_gt50pct" if rchg > 0.5 else "steady_speed",
                hard_parallel=bool(heading == "same_direction" and trend == "parallel_close"),
                d_gap_start_m=round(d_0, 3), d_gap_end_m=round(d_1, 3))


# ---------------------------------------------------------------- one match

class MatchData:
    """Raw provider data, Stage 2 candidates and Stage 3 metadata of one train/CV match (read only)."""

    def __init__(self, match_id: str | int):
        from football_fatigue.encounters import quality as Q
        from football_fatigue.intersections.catalogue import load_steps
        from football_fatigue.matching.benchmark import load_source
        from football_fatigue.window_intersections import core as W
        from football_fatigue.window_intersections import local as L
        from football_fatigue.window_intersections.__main__ import RAW

        self.match_id = str(int(match_id))
        self.fold = B.fold_for_match(self.match_id)            # refuses held-out / non-train matches before any data is read
        proc = B.proc_root()
        c, mask, _e, _sm, _p = load_source(str(RAW), self.match_id)
        seg, _cnt, dup = W.build_segments(c, mask, load_steps(str(RAW), self.match_id))
        self.c, self.mask, self.dup, self.seg, self.ch = c, mask, dup, seg, W.chain_table(seg)
        self.cx, self.cy, _ = W.coefficients(seg, "linear")
        self.est = (mask & Q.ESTIMATED) != 0
        cand = pd.read_parquet(proc / S.STAGE2 / "matches" / self.match_id / "candidates.parquet")
        if set(cand["split"]) != {"train_cv"}:
            raise S.HeldOutRefused(f"{self.match_id}: Stage 2 rows are not all train_cv")
        self.cand = cand.set_index("candidate_id", drop=False)
        d3 = proc / "m11-stage3-samples-v1" / "matches" / self.match_id
        L.verify_match_dir(d3)
        self.meta = pd.read_parquet(d3 / "meta_g0p80.parquet")
        self.ccfg = C.CandidateConfig()
        self.periods = {}
        for p in sorted(int(v) for v in np.unique(cand["period"])):
            rows_p = np.flatnonzero(c.canonical_period == p)
            tp = c.video_time_s[rows_p]
            self.periods[p] = (rows_p, tp, float(tp[0]), float(tp[-1]))
        self._v: dict[int, tuple] = {}

    def verts(self, k: int):
        if k not in self._v:
            self._v[k] = S.chain_vertices(self.seg, self.ch, k)
        return self._v[k]


@dataclass
class Scenario:
    scenario_id: str
    match_id: str
    fold: int
    candidate_id: str
    factor: str
    level: str
    params: dict[str, Any]
    valid: bool
    invalid_reason: str = ""
    y: int | None = None
    g0: float | None = None
    g1: float | None = None
    tc: float | None = None
    who_in: tuple[str, str] | None = None
    who_out: tuple[str, str] | None = None
    incoming: list[A.Track] = field(default_factory=list)
    outgoing: list[A.Track] = field(default_factory=list)
    hidden: np.ndarray | None = None          # (2, HIDDEN_LEN, 2) clean provider path of the player behind in0 / in1 inside [g0, g1]
    hidden_mask: np.ndarray | None = None
    label_source: str = ""
    families: dict[str, Any] = field(default_factory=dict)


def build(md: MatchData, enc: pd.Series, factor: str, level: Any, fam: dict[str, Any] | None = None) -> Scenario:
    """One declared Milestone 2 variant (one factor at a time)."""
    cid = str(enc["candidate_id"])
    return _build(md, enc, params(factor, level), scenario_id(cid, factor, level), factor, level_key(level), fam,
                  revalidate=factor in ("gap", "shift"))


DROPOUT_MODES = ("none", "random10", "random30", "random50", "burst0p5", "rate7p5")


def custom_id(candidate_id: str, p: dict[str, Any], seed: int) -> str:
    return _h(SUITE, candidate_id, "custom", *(level_key(float(p[k])) if k != "dropout" else p[k] for k in BASE), seed)[:16]


def build_custom(md: MatchData, enc: pd.Series, p: dict[str, Any], seed: int, fam: dict[str, Any] | None = None) -> Scenario:
    """Any parameter combination (interactive use; not part of the Milestone 2 suite). The label is re-validated with the Stage 2
    procedure whenever gap or centre differ from the accepted 0.8 s / closest-approach geometry; ``seed`` drives order, dropout
    and noise. Same code path as :func:`build`."""
    p = dict(BASE, **p)
    p = dict(gap=float(p["gap"]), shift=float(p["shift"]), noise=float(p["noise"]), dropout=str(p["dropout"]), context=float(p["context"]))
    if not (0.0 < p["gap"] <= 4.0) or abs(p["shift"]) > 1.0 or not (0.0 <= p["noise"] <= 1.0) or not (0.0 < p["context"] <= S.CONTEXT_S)             or p["dropout"] not in DROPOUT_MODES:
        raise ValueError(f"parameters outside the generator's range: {p}")
    cid = str(enc["candidate_id"])
    revalidate = abs(p["gap"] - BASE["gap"]) > 1e-9 or abs(p["shift"]) > 1e-9
    return _build(md, enc, p, custom_id(cid, p, seed), "custom", "custom", fam, revalidate=revalidate, seed=int(seed))


def _build(md: MatchData, enc: pd.Series, p: dict[str, Any], sid: str, factor: str, level: str, fam: dict[str, Any] | None, *,
           revalidate: bool, seed: int | None = None) -> Scenario:
    cid = str(enc["candidate_id"])
    sc = Scenario(sid, md.match_id, md.fold, cid, factor, level, p, valid=False)
    r = md.cand.loc[cid].to_dict()
    t_star = float(r["t_star"])
    if abs(float(enc["gap_centre_s"]) - t_star) > 1e-9:
        raise RuntimeError(f"{cid}: Stage 3 g0p80 centre differs from t*")
    g, tc = float(p["gap"]), t_star + float(p["shift"])
    if revalidate:
        rows_p, tp, pstart, pend = md.periods[int(r["period"])]
        q = C.label_quality(md.c, md.mask, md.dup, md.est, md.seg, md.ch, md.cx, md.cy, dict(r, t_star=tc), rows_p, tp, pstart, pend, g,
                            md.ccfg)
        if q["outcome"] != C.ACCEPTED:
            import json
            why = json.loads(q["reasons"]) or json.loads(q["lower_reasons"])
            sc.invalid_reason = f"label_quality_{q['outcome']}:" + "|".join(why)
            return sc
        sc.label_source = "stage2_label_quality_rerun"
    else:
        sc.label_source = "stage2_accepted_g0p80"
    ka, kb = int(r["chain_a"]), int(r["chain_b"])
    VA, VB = md.verts(ka)[:3], md.verts(kb)[:3]
    io_ = order_bits(sid)
    s = S.build_one(VA, VB, tc, g, *io_)
    if "failure" in s:
        sc.invalid_reason = f"construction:{s['failure']}"
        return sc
    rowa, rowb = md.verts(ka)[3], md.verts(kb)[3]
    raw = {"A": (md.c.video_time_s[rowa], md.c.x[rowa, int(r["slot_a"])], md.c.y[rowa, int(r["slot_a"])]),
           "B": (md.c.video_time_s[rowb], md.c.x[rowb, int(r["slot_b"])], md.c.y[rowb, int(r["slot_b"])])}
    lab = S.verify_label(s, raw)
    if lab != "ok":
        sc.invalid_reason = f"label_verification_{lab}"
        return sc
    rng = np.random.default_rng(int(sid, 16) if seed is None else seed)
    V = {"A": VA, "B": VB}
    g0, g1 = s["g0"], s["g1"]
    inc, out = [], []
    for side, who, dest in (("in", s["who_in"], inc), ("out", s["who_out"], out)):
        for pl in who:
            tr = cut(V[pl], g0, g1, side, float(p["context"]))
            tr = drop(tr, p["dropout"], side, rng)
            tr = add_noise(tr, float(p["noise"]), rng)
            dest.append(tr)
    ht = g0 + np.arange(HIDDEN_LEN) * DT
    hm = ht <= g1 + 1e-9
    hidden = np.zeros((2, HIDDEN_LEN, 2))
    for i, pl in enumerate(s["who_in"]):
        hidden[i][hm] = _interp(V[pl], ht[hm])
    sc.valid, sc.y, sc.g0, sc.g1, sc.tc = True, int(s["y"]), float(g0), float(g1), float(tc)
    sc.who_in, sc.who_out, sc.incoming, sc.outgoing = tuple(s["who_in"]), tuple(s["who_out"]), inc, out
    sc.hidden, sc.hidden_mask = hidden.astype(np.float32), np.stack([hm, hm])
    sc.families = fam if fam is not None else families(VA, VB, t_star)
    return sc


def encounter_families(md: MatchData, enc: pd.Series) -> dict[str, Any]:
    r = md.cand.loc[str(enc["candidate_id"])]
    VA, VB = md.verts(int(r["chain_a"]))[:3], md.verts(int(r["chain_b"]))[:3]
    f = families(VA, VB, float(r["t_star"]))
    f.update({k: (enc[k].item() if hasattr(enc[k], "item") else enc[k]) for k in
              ("crossing_class", "team_relation", "d_min", "d_band", "speed_band", "status_in_gap")})
    return f


def regenerate(match_id: str | int, candidate_id: str, factor: str, level: Any) -> Scenario:
    """Rebuild one scenario exactly (same observations, noise and order) from its defining keys."""
    md = MatchData(match_id)
    if isinstance(level, str) and factor != "baseline" and factor in FACTORS and level not in FACTORS[factor]:
        level = parse_level(factor, level)
    enc = md.meta.set_index("candidate_id", drop=False).loc[candidate_id]
    return build(md, enc, factor, level, encounter_families(md, enc))
