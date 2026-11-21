"""Fixed evaluation sets for the fair benchmark (never trained on, never used for selection or calibration).

Per test fold every model predicts: the primary centred table (``g0p80``), the shifted-centre table, 0.4 s and 1.6 s gaps, and two
texture controls of the primary inputs (5-sample smoothing; 5 cm Gaussian noise, same seeds as Stage 4). Strata (status, context
ESTIMATED share, clone-like, physics tier, independence) are attached afterwards from metadata.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from football_fatigue.m11_benchmark import data as D
from football_fatigue.m11_benchmark import robust as RB

STRESS = ("g0p80", "g0p80_shift", "g0p40", "g1p60", "smooth5", "noise5cm")
TEXTURE_FEATURES = D.out_dir() / "robust" / "texture"


class EvalSets:
    def __init__(self, sequences: bool, features: bool, tables=STRESS):
        self.sets: dict[str, dict[str, Any]] = {}
        base_x = None
        for t in tables:
            if t in ("smooth5", "noise5cm"):
                if base_x is None:
                    base_x, by, bg = D.load_inputs("g0p80")
                x = dict(base_x)
                if t == "smooth5":
                    x["in_pos"], x["out_pos"] = RB.smooth(x["in_pos"], x["in_mask"]), RB.smooth(x["out_pos"], x["out_mask"])
                else:
                    x["in_pos"], x["out_pos"] = RB.perturb(x["in_pos"], x["in_mask"], 0.05, 1), RB.perturb(x["out_pos"], x["out_mask"], 0.05, 2)
                y, g, meta = by, bg, D.load_meta("g0p80")
                fpath = TEXTURE_FEATURES / f"features_{t}.npz"
            else:
                x, y, g = D.load_inputs(t)
                meta = D.load_meta(t)
                fpath = D.out_dir() / "features" / f"{t}.npz"
            s: dict[str, Any] = dict(y=np.asarray(y).astype(int), groups=np.asarray(g).astype(str), gap=x["gap_s"].astype(np.float32),
                                     candidate_id=meta["candidate_id"].astype(str).to_numpy())
            if sequences:
                s["si"], s["so"] = D.seq_tensor(x)
            if features:
                s["F"] = D.load_npz(fpath)["F"]
                if len(s["F"]) != len(s["y"]):
                    raise RuntimeError(f"{t}: feature rows do not match inputs")
            self.sets[t] = s

    def rows(self, table: str, matches) -> np.ndarray:
        return np.flatnonzero(np.isin(self.sets[table]["groups"], list(matches)))
