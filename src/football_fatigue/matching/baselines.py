"""Transparent, non-trained association baselines (M11 step 2). They are diagnostic references, not the FOOTBALLTrace reproduction.

Each baseline receives one matcher-visible instance (``benchmark.INSTANCE_KEYS``) and nothing else: no reference file, no
provider identity, no match, frame or absolute time. For every incoming tracklet it returns a decision:

- ``MATCHED``: committed to one outgoing tracklet;
- ``UNMATCHED``: no outgoing tracklet is plausible (it may be absent);
- ``UNCERTAIN``: a candidate exists but an alternative is too close to call. The candidate is recorded, but no match is committed.

Every threshold is fixed a priori in :class:`BaselineConfig`. None is tuned, and validation and test matches are never used.

- ``nearest``: the last incoming position against the first outgoing position, judged per incoming tracklet independently (two
  incoming tracklets can claim one outgoing tracklet; this is reported as a conflict).
- ``constant_velocity``: the incoming velocity (least squares over update instants) extrapolated to the first outgoing time, also
  judged independently.
- ``hungarian_cv``: the constant-velocity costs in a global one-to-one assignment with an abstain option, solved exactly.
- ``chain_during``: frame-to-frame one-to-one chaining through the unlabelled ambiguity-interval points, then one assignment to
  the outgoing tracklets. It is the only baseline that uses the encounter-interval observations. The other three see only the
  before and after windows, which makes them equivalent to a gap setting. On task B (interval withheld) it has no interval
  points and reduces to one gated one-to-one step from the incoming ends to the outgoing starts.
- ``e0_adapted``: the FOOTBALLTrace-inspired junction-sharpness association (``footballtrace.adapted``). It is not the published
  method, and it ignores the interval points.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .benchmark import tracklet_velocity

BASELINES = ("nearest", "constant_velocity", "hungarian_cv", "chain_during", "e0_adapted")   # e0_adapted: footballtrace.adapted
DECISIONS = ("MATCHED", "UNMATCHED", "UNCERTAIN")


@dataclass(frozen=True)
class BaselineConfig:
    velocity_span_s: float = 0.5        # incoming velocity from the update instants of its last 0.5 s
    max_speed_m_s: float = 12.0         # same physiological bound as step 1
    gate_base_m: float = 2.0            # an outgoing tracklet is plausible within gate_base + gate_per_s * time gap
    gate_m_per_s: float = 3.0
    margin_m: float = 0.5               # best and next plausible alternative closer than this: UNCERTAIN
    chain_gate_base_m: float = 0.5      # chaining: a point is reachable within chain_gate_base + max_speed * elapsed
    chain_margin_m: float = 0.25        # chaining: any step (or the final link) closer than this to an alternative: UNCERTAIN

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def solve(cost: np.ndarray, abstain: np.ndarray, forbid: tuple[int, int] | None = None) -> tuple[float, tuple[int, ...]]:
    """Exact minimum-cost partial one-to-one assignment by exhaustive search (rows are few: at most the event's participants).

    ``cost[i, j]`` is ``inf`` where row ``i`` may not take column ``j``. ``abstain[i]`` is the cost of leaving row ``i``
    unassigned. ``forbid`` excludes one pairing. Returns ``(total, column per row, or -1)``; ties keep the first solution
    in row and column order."""
    n = cost.shape[0]
    cols = [[int(j) for j in np.flatnonzero(np.isfinite(cost[i]))] for i in range(n)]
    best: list[Any] = [np.inf, tuple([-1] * n)]

    def rec(i: int, used: frozenset, total: float, pick: list[int]) -> None:
        if total >= best[0]:
            return
        if i == n:
            best[0], best[1] = total, tuple(pick)
            return
        for j in cols[i]:
            if j in used or forbid == (i, j):
                continue
            pick.append(j)
            rec(i + 1, used | {j}, total + float(cost[i, j]), pick)
            pick.pop()
        pick.append(-1)
        rec(i + 1, used, total + float(abstain[i]), pick)
        pick.pop()

    rec(0, frozenset(), 0.0, [])
    return float(best[0]), best[1]


def _decision(inc: str, decision: str, out: str | None, cand: str | None, cost: float | None, margin: float | None) -> dict[str, Any]:
    r = lambda v: None if v is None or not np.isfinite(v) else round(float(v), 4)   # noqa: E731
    return dict(incoming=inc, decision=decision, outgoing=out if decision == "MATCHED" else None, candidate=cand, cost=r(cost), margin=r(margin))


def _costs(inst: dict, bc: BaselineConfig, predict: bool) -> tuple[np.ndarray, np.ndarray]:
    I, O = inst["incoming"], inst["outgoing"]
    C = np.full((len(I), len(O)), np.inf)
    G = np.full((len(I), len(O)), np.inf)
    for i, a in enumerate(I):
        pa, ta = np.array([a["x"][-1], a["y"][-1]]), a["t"][-1]
        v = tracklet_velocity(a, at_end=True, span_s=bc.velocity_span_s, max_speed_m_s=bc.max_speed_m_s)[0] if predict else np.zeros(2)
        for j, b in enumerate(O):
            dt = b["t"][0] - ta
            C[i, j] = float(np.hypot(*(np.array([b["x"][0], b["y"][0]]) - (pa + v * dt))))
            G[i, j] = bc.gate_base_m + bc.gate_m_per_s * dt
    return C, G


def independent(inst: dict, bc: BaselineConfig, predict: bool) -> list[dict]:
    C, G = _costs(inst, bc, predict)
    O = [b["tracklet_id"] for b in inst["outgoing"]]
    out = []
    for i, a in enumerate(inst["incoming"]):
        ok = [j for j in np.argsort(C[i], kind="stable") if C[i, j] <= G[i, j]]
        if not ok:
            out.append(_decision(a["tracklet_id"], "UNMATCHED", None, None, None, None))
            continue
        b1 = C[i, ok[0]]
        margin = C[i, ok[1]] - b1 if len(ok) > 1 else np.inf
        out.append(_decision(a["tracklet_id"], "UNCERTAIN" if margin < bc.margin_m else "MATCHED", O[ok[0]], O[ok[0]], b1, margin))
    return out


def hungarian_cv(inst: dict, bc: BaselineConfig) -> list[dict]:
    C, G = _costs(inst, bc, True)
    O = [b["tracklet_id"] for b in inst["outgoing"]]
    Cg = np.where(C <= G, C, np.inf)
    abstain = np.array([G[i].max() if G.shape[1] else 0.0 for i in range(len(C))])
    total, pick = solve(Cg, abstain)
    out = []
    for i, a in enumerate(inst["incoming"]):
        j = pick[i]
        if j < 0:
            out.append(_decision(a["tracklet_id"], "UNMATCHED", None, None, None, None))
            continue
        margin = solve(Cg, abstain, forbid=(i, j))[0] - total
        out.append(_decision(a["tracklet_id"], "UNCERTAIN" if margin < bc.margin_m else "MATCHED", O[j], O[j], C[i, j], margin))
    return out


def chain_during(inst: dict, bc: BaselineConfig) -> list[dict]:
    I, O = inst["incoming"], inst["outgoing"]
    pos = np.array([[a["x"][-1], a["y"][-1]] for a in I], float).reshape(-1, 2)
    last = np.array([a["t"][-1] for a in I], float)
    margin = np.full(len(I), np.inf)

    def step(P: np.ndarray, tP: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
        D = np.hypot(pos[:, None, 0] - P[None, :, 0], pos[:, None, 1] - P[None, :, 1])
        gate = bc.chain_gate_base_m + bc.max_speed_m_s * np.maximum(tP[None, :] - last[:, None], 0.0)
        Cg = np.where(D <= gate, D, np.inf)
        abstain = gate.max(axis=1)
        total, pick = solve(Cg, abstain)
        for i, j in enumerate(pick):
            if j >= 0:
                margin[i] = min(margin[i], solve(Cg, abstain, forbid=(i, j))[0] - total)
        return D, pick

    for tf, pts in zip(inst["during"]["t"], inst["during"]["points"]):
        if not pts or not len(I):
            continue
        P = np.array([[p[0], p[1]] for p in pts], float)
        _, pick = step(P, np.full(len(P), tf))
        for i, j in enumerate(pick):
            if j >= 0:
                pos[i], last[i] = P[j], tf
    if not O or not len(I):
        return [_decision(a["tracklet_id"], "UNMATCHED", None, None, None, None) for a in I]
    P = np.array([[b["x"][0], b["y"][0]] for b in O], float)
    D, pick = step(P, np.array([b["t"][0] for b in O], float))
    out = []
    for i, a in enumerate(I):
        j = pick[i]
        if j < 0:
            out.append(_decision(a["tracklet_id"], "UNMATCHED", None, None, None, None))
        else:
            oid = O[j]["tracklet_id"]
            out.append(_decision(a["tracklet_id"], "UNCERTAIN" if margin[i] < bc.chain_margin_m else "MATCHED", oid, oid, D[i, j], margin[i]))
    return out


def run_baseline(name: str, instance: dict, bc: BaselineConfig) -> list[dict]:
    if name == "nearest":
        return independent(instance, bc, predict=False)
    if name == "constant_velocity":
        return independent(instance, bc, predict=True)
    if name == "hungarian_cv":
        return hungarian_cv(instance, bc)
    if name == "chain_during":
        return chain_during(instance, bc)
    if name == "e0_adapted":
        from .footballtrace import adapted
        return adapted(instance)
    raise ValueError(f"unknown baseline {name}")


def run_baselines(instances: list[dict], bc: BaselineConfig = BaselineConfig()) -> list[dict]:
    """One prediction row per (baseline, instance). Only matcher-visible instances go in."""
    return [dict(baseline=name, instance_id=inst["instance_id"], decisions=run_baseline(name, inst, bc))
            for name in BASELINES for inst in instances]
