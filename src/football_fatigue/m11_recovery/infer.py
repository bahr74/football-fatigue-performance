"""Inference for two-player identity recovery after a position gap, with abstention (frozen model artifacts only).

Input: two tracks *before* a gap and two anonymous tracks *after* it (metres, pitch coordinates; NaN = missing) with their
times in seconds. Output: P(H0) where H0 = {before 0 -> after 0, before 1 -> after 1}, the decision ``keep`` (H0), ``swap``
(H1) or ``abstain``, and the reason. A decision is committed only when the confidence max(p, 1 - p) reaches the operating
threshold frozen on development out-of-fold predictions for the chosen false-association target, and the input lies inside
the trained domain (gap 0.4-2.0 s, >= 3 observed points per track). Positions are resampled exactly as in training
(:func:`.views.resample` onto the 0.1 s grid centred on the gap) and featurised by :func:`.dataset.model_inputs`.

Scope: trained and validated on provider positions with *synthetic* gaps and noise; it has not been tested on visual tracker
output or real occlusions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import motion as MO
from .dataset import model_inputs
from .views import GAPS_S, GRID, GRID_DT, Sample, resample

GAP_RANGE_S = (GAPS_S[0] - GRID_DT / 2, GAPS_S[-1] + GRID_DT / 2)


def to_sample(before: np.ndarray, t_before: np.ndarray, after: np.ndarray, t_after: np.ndarray) -> Sample:
    """Resample (2, Tb, 2) / (2, Ta, 2) tracks with shared times onto the training grid, centred on the middle of the gap."""
    before, after = np.asarray(before, float), np.asarray(after, float)
    t_before, t_after = np.asarray(t_before, float), np.asarray(t_after, float)
    if before.shape[0] != 2 or after.shape[0] != 2:
        raise ValueError("exactly two before and two after tracks are required")
    t_star = 0.5 * (np.nanmax(t_before) + np.nanmin(t_after))
    tb, ta = np.round(t_before - t_star, 6), np.round(t_after - t_star, 6)       # microsecond rounding: grid end points stay inside
    gb, ga = GRID[(GRID >= tb.min() - 1e-9) & (GRID <= tb.max() + 1e-9)], GRID[(GRID >= ta.min() - 1e-9) & (GRID <= ta.max() + 1e-9)]
    rs = lambda t, tr, g: np.stack([resample(t, tr[i, :, 0], tr[i, :, 1], np.ones(len(t), bool), g) for i in (0, 1)])  # noqa: E731
    b, a = rs(tb, before, gb), rs(ta, after, ga)
    gap = float(round((ga.min() - gb.max()) - 2 * GRID_DT, 6)) if len(ga) and len(gb) else float("nan")
    return Sample(b, a, gb, ga, gap, 0.0, -1, ("inference",))


@dataclass
class Recovery:
    p_h0: float | None
    decision: str                    # keep / swap / abstain
    confidence: float | None
    threshold: float
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class RecoveryModel:
    """A frozen model directory: ``model.joblib`` (or ``model.pt`` + ``model.joblib`` for the hybrid), ``spec.json`` and
    ``manifest.json`` (checksums verified on load)."""

    def __init__(self, model, spec: dict[str, Any]):
        self.model, self.spec = model, spec

    @classmethod
    def load(cls, path: str | Path) -> "RecoveryModel":
        import joblib

        path = Path(path)
        man = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        for name, sha in man["outputs_sha256"].items():
            if hashlib.sha256((path / name).read_bytes()).hexdigest() != sha:
                raise ValueError(f"{path / name} does not match its manifest")
        spec = json.loads((path / "spec.json").read_text(encoding="utf-8"))
        model = joblib.load(path / "model.joblib")
        if spec["kind"] == "hybrid":
            import io

            import torch

            from .models import make_pairnet

            model.net = make_pairnet(spec["n_phys"], model.hidden)
            model.net.load_state_dict(torch.load(io.BytesIO((path / "model.pt").read_bytes()), map_location="cpu"))
        return cls(model, spec)

    def operating_threshold(self, target: str | None = None) -> float:
        return float(self.spec["operating_points"][target or self.spec["default_operating_point"]])

    def recover(self, before, t_before, after, t_after, target: str | None = None) -> Recovery:
        c = self.operating_threshold(target)
        s = to_sample(before, t_before, after, t_after)
        if not (GAP_RANGE_S[0] <= s.gap_s <= GAP_RANGE_S[1]):
            return Recovery(None, "abstain", None, c, f"gap {s.gap_s:.2f} s outside the trained range {GAPS_S[0]}-{GAPS_S[-1]} s")
        if not MO.usable(s):
            return Recovery(None, "abstain", None, c, "fewer than 3 observed points on a track")
        row = {k: np.asarray([v]) for k, v in model_inputs(s).items()}
        p = float(self.model.predict(row)[0])
        conf = max(p, 1 - p)
        if conf < c:
            return Recovery(p, "abstain", conf, c, "confidence below the operating threshold")
        return Recovery(p, "keep" if p > 0.5 else "swap", conf, c, "committed")
