"""Cross-camera location association (same rules as the M10.7 experiment): Mahalanobis gate, one-to-one Hungarian assignment,
ambiguity flag when a second candidate is nearly as good, information-weighted fused location.

A link is location evidence for one instant. It is never a player identity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

GATE_CHI2_99 = 9.21
AMBIGUITY_DELTA_D2 = 2.0


@dataclass(frozen=True)
class Uncertainty:
    calibration_sigma_m: float = 0.6
    sync_sigma_s: float = 0.15
    player_speed_mps: float = 4.0          # converts sync uncertainty to position uncertainty

    @property
    def cal(self) -> np.ndarray:
        return self.calibration_sigma_m ** 2 * np.eye(2)

    @property
    def sync(self) -> np.ndarray:
        return (self.player_speed_mps * self.sync_sigma_s) ** 2 * np.eye(2)


def associate(moving: list[dict], fixed: list[dict], unc: Uncertainty = Uncertainty(),
              gate: float = GATE_CHI2_99, ambiguity: float = AMBIGUITY_DELTA_D2) -> list[dict]:
    """moving/fixed: dicts with 'k' (index), 'pitch' (x, y) and 'cov' (2x2). Only pass observations that are on the pitch and,
    for the fixed camera, inside the moving camera's view. Returns links sorted by moving index."""
    nm, nf = len(moving), len(fixed)
    if not nm or not nf:
        return []
    P_m = np.array([o["pitch"] for o in moving], float); P_f = np.array([o["pitch"] for o in fixed], float)
    D2 = np.full((nm, nf), np.inf); DM = np.full((nm, nf), np.inf)
    for a, m in enumerate(moving):
        for b, f in enumerate(fixed):
            d = P_m[a] - P_f[b]; S = np.asarray(m["cov"]) + np.asarray(f["cov"]) + unc.cal + unc.sync
            DM[a, b] = float(np.hypot(*d)); D2[a, b] = float(d @ np.linalg.solve(S, d))
    ok = D2 <= gate
    rows, cols = linear_sum_assignment(np.where(ok, D2, 1e6))
    links = []
    for a, b in zip(rows, cols):
        if not ok[a, b]:
            continue
        alt = [D2[a, k] for k in range(nf) if k != b and ok[a, k]] + [D2[k, b] for k in range(nm) if k != a and ok[k, b]]
        delta = min(alt + [np.inf]) - D2[a, b]
        Sf = np.asarray(fixed[b]["cov"]) + unc.cal / 2; Sm = np.asarray(moving[a]["cov"]) + unc.cal / 2 + unc.sync
        Su = np.linalg.inv(np.linalg.inv(Sf) + np.linalg.inv(Sm)); pu = Su @ (np.linalg.solve(Sf, P_f[b]) + np.linalg.solve(Sm, P_m[a]))
        links.append({"m": moving[a]["k"], "f": fixed[b]["k"], "d_m": round(float(DM[a, b]), 2), "d2": round(float(D2[a, b]), 2),
                      "ambiguous": bool(delta < ambiguity), "second_best_delta_d2": None if not np.isfinite(delta) else round(float(delta), 2),
                      "fused": [round(float(v), 2) for v in pu], "fused_cov": np.round(Su, 3).tolist(),
                      "candidates_m": int(ok[a].sum()), "candidates_f": int(ok[:, b].sum())})
    return sorted(links, key=lambda l: l["m"])


def state_of(on_pitch: bool, link: dict | None, camera: str, other_sees_area: bool) -> str:
    if not on_pitch:
        return "off-pitch"
    if link:
        return "ambiguous" if link["ambiguous"] else "matched"
    return f"{camera}-only" + (" (other camera sees this area)" if other_sees_area else " (outside other camera)")
