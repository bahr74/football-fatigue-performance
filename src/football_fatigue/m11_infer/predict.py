"""Routed, abstaining prediction for one two-in / two-out ambiguity (routing policy ``m11-routing-v1``).

**What the model answers.** Given two incoming tracklets (in0, in1) and two anonymous outgoing tracklets (out0, out1):
MATCHED = in0 -> out0 and in1 -> out1; SWAPPED = in0 -> out1 and in1 -> out0. p = P(SWAPPED).

**Routing policy m11-routing-v1** (conservative; declared before any new-scenario test, to be revised from Milestone 2 evidence):

1. Malformed input -> ``InputError`` (no prediction).
2. Fewer than 3 valid grid points in any tracklet -> ``unsupported`` (abstain).
3. ``gap_s`` outside [0.4, 1.6] s (the only gap lengths ever evaluated) -> ``unsupported`` (abstain).
4. Declared positional noise above 5 cm (the largest level evaluated) -> ``unsupported`` (abstain).
5. ``gap_s`` within 0.75-0.85 s and noise <= 2 cm or not declared -> ``primary``: GRU C. Development OOF (gap 0.8 s, clean FIFA
   data): centred BA 0.924-0.929, shifted (+-0.3 s centre) 0.909-0.911. Only exactly 0.8 s was evaluated; the +-0.05 s band is
   a declared tolerance, not a measured one.
6. Otherwise (gap in [0.4, 1.6] s outside the GRU band, or noise in (2, 5] cm) -> ``fallback``: HGB C, which transferred better
   than GRU C in development (0.4 s 0.928 vs 0.840, 1.6 s 0.803 vs 0.745, 5 cm noise 0.842 vs 0.757). HGB C was also trained at
   0.8 s only; transfer was measured at 0.4 and 1.6 s and noise at 5 cm only; values in between are untested. Its threshold was
   calibrated at 0.8 s, so the realised error away from 0.8 s is not known.
7. The chosen model abstains when confidence max(p, 1 - p) is below its threshold for the operating point (``error_05``: 5 %
   pooled development OOF error; ``error_02``: 2 %).

For ``unsupported`` inputs both probabilities are still reported when computable, marked diagnostic only; the decision is
ABSTAIN. Undeclared noise is treated as clean (FIFA-like) and noted: video-derived positions must declare ``noise_sigma_m``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np

from . import adapter as A
from . import bundle as B

POLICY = "m11-routing-v1"
GRU_GAP_BAND = (0.75, 0.85)
GAP_RANGE = (0.4, 1.6)
GRU_MAX_NOISE_M = 0.02
MAX_NOISE_M = 0.05
_EPS = 1e-6

MATCHED, SWAPPED, ABSTAIN = "MATCHED", "SWAPPED", "ABSTAIN"
PAIRS = {MATCHED: {"in0": "out0", "in1": "out1"}, SWAPPED: {"in0": "out1", "in1": "out0"}}


@dataclass
class Route:
    support: str                    # primary / fallback / unsupported
    model: str | None               # gru_c / hgb_c / None
    reasons: list[str]


def route(gap_s: float, noise_sigma_m: float | None, enough_points: bool) -> Route:
    reasons: list[str] = []
    if not enough_points:
        return Route("unsupported", None, [f"too_few_valid_points: every tracklet needs >= {A.MIN_POINTS} grid points in its "
                                           f"{A.CONTEXT_S:.1f} s context"])
    if noise_sigma_m is not None and (not np.isfinite(noise_sigma_m) or noise_sigma_m < 0):
        raise A.InputError("noise_sigma_m must be a finite non-negative number (metres) or None")
    if not (GAP_RANGE[0] - _EPS <= gap_s <= GAP_RANGE[1] + _EPS):
        return Route("unsupported", None, [f"gap_outside_evaluated_range: gap_s {gap_s:.3f} s not in "
                                           f"[{GAP_RANGE[0]}, {GAP_RANGE[1]}] s"])
    if noise_sigma_m is not None and noise_sigma_m > MAX_NOISE_M + _EPS:
        return Route("unsupported", None, [f"noise_above_evaluated_level: {noise_sigma_m:.3f} m > {MAX_NOISE_M} m"])
    in_band = GRU_GAP_BAND[0] - _EPS <= gap_s <= GRU_GAP_BAND[1] + _EPS
    clean = noise_sigma_m is None or noise_sigma_m <= GRU_MAX_NOISE_M + _EPS
    if noise_sigma_m is None:
        reasons.append("noise_not_declared: treated as clean FIFA-like positions")
    if in_band and clean:
        reasons.append(f"primary: gap_s {gap_s:.3f} s within the GRU C band {GRU_GAP_BAND} s")
        if abs(gap_s - 0.8) > 1e-3:
            reasons.append("gap_s differs from the evaluated 0.8 s (declared tolerance band)")
        return Route("primary", "gru_c", reasons)
    if not in_band:
        reasons.append(f"fallback: gap_s {gap_s:.3f} s outside the GRU C band {GRU_GAP_BAND} s")
    if not clean:
        reasons.append(f"fallback: declared noise {noise_sigma_m:.3f} m above the GRU C level {GRU_MAX_NOISE_M} m")
    reasons.append("HGB C threshold was calibrated at 0.8 s; realised error at other gaps / noise is not established")
    return Route("fallback", "hgb_c", reasons)


@dataclass
class Prediction:
    decision: str                           # MATCHED / SWAPPED / ABSTAIN
    predicted_pairing: str | None           # MATCHED / SWAPPED from the chosen model (None if no model was used)
    pairs: dict[str, str] | None            # {"in0": "out?", "in1": "out?"} for the decision (None when abstaining)
    p_swapped: float | None                 # chosen model
    confidence: float | None                # max(p, 1 - p) of the chosen model
    threshold: float | None
    abstained: bool
    support: str
    model: str | None
    operating_point: str
    reasons: list[str]
    probabilities: dict[str, float | None]  # both models when computable (diagnostic for the model not chosen / unsupported)
    gap_s: float
    noise_sigma_m: float | None
    policy: str = POLICY
    fold: int | None = None
    checkpoints: dict[str, str] = field(default_factory=dict)
    input_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def decide(p: float, threshold: float) -> tuple[str, str, bool, float]:
    """(predicted pairing, decision, abstained, confidence). Ties at p = 0.5 count as MATCHED (Stage 4 convention p > 0.5)."""
    conf = float(max(p, 1 - p))
    pairing = SWAPPED if p > 0.5 else MATCHED
    abst = conf < threshold
    return pairing, (ABSTAIN if abst else pairing), abst, conf


class Predictor:
    """Both models of one outer fold. ``models`` may be injected (objects with ``predict(arrays) -> p``) for testing."""

    def __init__(self, fold: int, device: str = "cpu", models: dict[str, Any] | None = None):
        self.fold, self.device = fold, device
        if models is None:
            models = {m: B.load_fold_model(m, fold, device=device) for m in B.MODELS}
        self.models = models

    @classmethod
    def for_match(cls, match_id: str | int, device: str = "cpu") -> "Predictor":
        """The fold models that never saw ``match_id`` (development matches only; held-out matches are refused)."""
        return cls(B.fold_for_match(match_id), device=device)

    def checkpoints(self) -> dict[str, str]:
        return {k: getattr(m, "sha256", "injected") for k, m in self.models.items()}

    def predict_arrays(self, x: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {k: np.asarray(m.predict(x), np.float64) for k, m in self.models.items()}

    def predict_input(self, mi: A.ModelInput, noise_sigma_m: float | None = None, operating_point: str = "error_05") -> Prediction:
        if operating_point not in B.OPERATING_POINTS:
            raise A.InputError(f"operating_point must be one of {B.OPERATING_POINTS}")
        r = route(mi.gap_s, noise_sigma_m, mi.enough_points)
        probs: dict[str, float | None] = {k: None for k in self.models}
        if mi.enough_points:
            probs = {k: float(v[0]) for k, v in self.predict_arrays(mi.arrays).items()}
        base = dict(probabilities=probs, gap_s=round(mi.gap_s, 6), noise_sigma_m=noise_sigma_m, fold=self.fold,
                    checkpoints=self.checkpoints(), input_summary=mi.summary(), operating_point=operating_point)
        if r.model is None:
            if mi.enough_points:
                r.reasons.append("probabilities are diagnostic only (unsupported input)")
            return Prediction(decision=ABSTAIN, predicted_pairing=None, pairs=None, p_swapped=None, confidence=None, threshold=None,
                              abstained=True, support=r.support, model=None, reasons=r.reasons, **base)
        p = probs[r.model]
        th = B.THRESHOLDS[r.model][operating_point]
        pairing, dec, abst, conf = decide(p, th)
        reasons = list(r.reasons)
        if abst:
            reasons.append(f"low_confidence: {conf:.4f} < {th} ({r.model}, {operating_point})")
        return Prediction(decision=dec, predicted_pairing=pairing, pairs=None if abst else PAIRS[pairing], p_swapped=p,
                          confidence=conf, threshold=th, abstained=abst, support=r.support, model=r.model, reasons=reasons, **base)

    def predict(self, incoming: Sequence[A.Track], outgoing: Sequence[A.Track], g0: float, g1: float,
                noise_sigma_m: float | None = None, operating_point: str = "error_05") -> Prediction:
        return self.predict_input(A.build_inputs(incoming, outgoing, g0, g1), noise_sigma_m, operating_point)
