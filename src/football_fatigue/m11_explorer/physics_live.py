"""Phase 2 deterministic baselines on one live two-player scenario (the same model-visible arrays the GRU / HGB receive).

No learned parameter; Kalman noise is the label-free Phase 2 selection (sigma_a 8 m/s^2, sigma_m 0.05 m). A positive margin
favours MATCHED (incoming order = outgoing order). The margin is a cost difference in the method's own units, not a probability."""

from __future__ import annotations

from typing import Any

import numpy as np

from football_fatigue.m11_infer import adapter as A
from football_fatigue.m11_phase2 import baselines as B

SIGMA_A, SIGMA_M = 8.0, 0.05
SHOWN = ("kf_fwd", "kf_bidir", "cv_fwd", "cv_bidir", "ca_fwd", "nearest", "ocm_dir")
NAMES = dict(kf_fwd="Kalman forward + Hungarian", kf_bidir="Kalman bidirectional", cv_fwd="Constant velocity (forward)",
             cv_bidir="Constant velocity (both sides)", ca_fwd="Constant acceleration (constrained)", nearest="Nearest endpoint",
             ocm_dir="Direction consistency (OC-SORT-inspired cue, not OC-SORT)")
# independent balanced accuracy on the Stage 3 g0p80 population (Phase 2, development OOF-free: no fitting) [m11-phase2-ambiguity-benchmark-v1]
PHASE2_BA = dict(kf_fwd=0.667, kf_bidir=0.553, cv_fwd=0.611, cv_bidir=0.525, ca_fwd=0.655, nearest=0.354, ocm_dir=0.643)


def run(incoming, outgoing, g0: float, g1: float) -> dict[str, Any]:
    mi = A.build_inputs(incoming, outgoing, g0, g1)
    if not mi.arrays:
        return dict(available=False, reason="; ".join(mi.notes) or "too few valid points")
    c = B.all_costs(mi.arrays, SIGMA_A, SIGMA_M)
    rows = []
    for k in SHOWN:
        m = float(B.margin(c[k])[0])
        rows.append(dict(method=k, name=NAMES[k], margin=m, decision="MATCHED" if m > 0 else ("SWAPPED" if m < 0 else "tie"),
                         phase2_independent_ba=PHASE2_BA[k]))
    gate = int(B.gated_decision(c["kf_fwd_d2"], c["kf_fwd"])[0]) if "kf_fwd_d2" in c else None
    return dict(available=True, rows=rows, kalman_gate={0: "MATCHED", 1: "SWAPPED", -1: "abstain (no full assignment inside the "
                                                                                      "0.99 gate)"}.get(gate, "n/a"),
                kalman_cost_matrix=np.asarray(c["kf_fwd"][0]).tolist())
