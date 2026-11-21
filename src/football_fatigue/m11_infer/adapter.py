"""Timestamped pitch-coordinate tracklets -> the unchanged Stage 3 model inputs.

**Input convention** (metres, seconds):

- four tracklets: two *incoming* (observed before the hidden interval) and two anonymous *outgoing* (observed after it);
  each is ``Track(t, x, y)``: observation times in seconds (any common origin, strictly increasing) and pitch coordinates in
  metres in one fixed metric pitch frame (the FIFA/PFF frame is: origin at the pitch centre, x along the touchlines, y along
  the end lines). Absolute position and orientation do not enter the model: the per-sample frame below removes them.
- the hidden interval ``[g0, g1]`` in the same time base; ``gap_s = g1 - g0`` is the model's gap input.

**Exact gap semantics** (identical to Stage 3, ``samples.py``): incoming grid ``g0 + k/15`` for k = -30..-1 (its last point is
``g0 - 1/15``), outgoing grid ``g1 + k/15`` for k = 1..30 (first point ``g1 + 1/15``). Positions are linear interpolation of the
observations strictly before ``g0`` (incoming) or strictly after ``g1`` (outgoing); grid points outside the observed span are
masked (no extrapolation). Observations inside ``[g0, g1]`` are ignored and counted. On FIFA data, where observations arrive
every 1/15 s, a Stage 3 gap of ``gap_s = 0.8`` therefore leaves 0.80-0.93 s between the last incoming and the first outgoing
observation. :func:`gap_from_observations` defines ``[g0, g1]`` from those two observation times for real tracks.

**Frame**: ``samples.sample_frame`` on the incoming grid only (origin = midpoint of the last valid incoming points, x axis =
summed incoming velocity, documented fallbacks). The transform is the Stage 3 one, so Stage 3 samples are reproduced exactly.
Sampling is by timestamp; native video rates (25, 30, 50, 60 fps) are interpolated onto the 15 Hz grid, which never creates
grid points beyond the observed span.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from football_fatigue.window_intersections import samples as S

DT = S.DT
T = S.T
CONTEXT_S = S.CONTEXT_S
MIN_POINTS = S.MIN_POINTS


class InputError(ValueError):
    """Malformed trajectory input (not a modelling limitation)."""


@dataclass(frozen=True)
class Track:
    """Observations of one tracklet: times (s) and pitch positions (m)."""

    t: np.ndarray
    x: np.ndarray
    y: np.ndarray

    @classmethod
    def from_lists(cls, t: Sequence[float], x: Sequence[float], y: Sequence[float]) -> "Track":
        return cls(np.asarray(t, np.float64), np.asarray(x, np.float64), np.asarray(y, np.float64))

    def validate(self, label: str) -> None:
        for nm, a in (("t", self.t), ("x", self.x), ("y", self.y)):
            if not isinstance(a, np.ndarray) or a.ndim != 1:
                raise InputError(f"{label}.{nm} must be a 1-D array")
            if not np.isfinite(a).all():
                raise InputError(f"{label}.{nm} contains NaN or infinite values")
        if not (len(self.t) == len(self.x) == len(self.y)):
            raise InputError(f"{label}: t, x and y have different lengths")
        if len(self.t) == 0:
            raise InputError(f"{label} has no observations")
        if len(self.t) > 1 and not (np.diff(self.t) > 0).all():
            raise InputError(f"{label}.t must be strictly increasing")
        if max(np.abs(self.x).max(), np.abs(self.y).max()) > 200.0:
            raise InputError(f"{label}: coordinates beyond 200 m; positions must be metric pitch coordinates")

    def to_dict(self) -> dict[str, list[float]]:
        return dict(t=self.t.tolist(), x=self.x.tolist(), y=self.y.tolist())


def gap_from_observations(t_last_incoming: float, t_first_outgoing: float, eps: float = 1e-6) -> tuple[float, float]:
    """Hidden interval for real tracks: it starts just after the last trusted incoming observation and ends just before the
    first trusted outgoing one, so ``gap_s`` is the unobserved duration (minus ``2 eps``)."""
    if not t_first_outgoing - t_last_incoming > 2 * eps:
        raise InputError("the first outgoing observation must come after the last incoming one")
    return float(t_last_incoming + eps), float(t_first_outgoing - eps)


@dataclass
class ModelInput:
    """Model arrays for one sample (batch of 1, ``samples.INPUT_FIELDS``) plus everything needed to draw and audit it."""

    arrays: dict[str, np.ndarray]
    g0: float
    g1: float
    grid_in: np.ndarray
    grid_out: np.ndarray
    pitch_in: np.ndarray            # (2, T, 2) grid positions in pitch coordinates (0 where masked)
    pitch_out: np.ndarray
    in_mask: np.ndarray             # (2, T)
    out_mask: np.ndarray
    n_valid: tuple[int, int, int, int]
    origin: np.ndarray | None
    R: np.ndarray | None
    frame_mode: int | None
    ignored_in_gap: tuple[int, int, int, int]
    notes: list[str] = field(default_factory=list)

    @property
    def gap_s(self) -> float:
        return float(self.g1 - self.g0)

    @property
    def enough_points(self) -> bool:
        return min(self.n_valid) >= MIN_POINTS

    def summary(self) -> dict[str, Any]:
        return dict(g0=self.g0, g1=self.g1, gap_s=round(self.gap_s, 6), n_valid=list(self.n_valid), frame_mode=self.frame_mode,
                    ignored_observations_in_gap=list(self.ignored_in_gap), notes=list(self.notes))


def build_inputs(incoming: Sequence[Track], outgoing: Sequence[Track], g0: float, g1: float) -> ModelInput:
    """Convert two incoming and two outgoing tracklets and the hidden interval ``[g0, g1]`` into Stage 3 model arrays.

    Raises :class:`InputError` for malformed input. Too few valid grid points is not an error: the result has
    ``enough_points == False`` and no frame, and the predictor abstains."""
    if len(incoming) != 2 or len(outgoing) != 2:
        raise InputError("exactly two incoming and two outgoing tracklets are required (the model is a two-player model)")
    for i, tr in enumerate(incoming):
        tr.validate(f"incoming[{i}]")
    for j, tr in enumerate(outgoing):
        tr.validate(f"outgoing[{j}]")
    g0, g1 = float(g0), float(g1)
    if not (np.isfinite(g0) and np.isfinite(g1)) or not g1 > g0:
        raise InputError("the hidden interval needs finite g0 < g1")
    grid_in, grid_out = g0 + S.IN_REL, g1 + S.OUT_REL
    tin = [S.tracklet(tr.t, tr.x, tr.y, grid_in, "in", g0, g1) for tr in incoming]
    tout = [S.tracklet(tr.t, tr.x, tr.y, grid_out, "out", g0, g1) for tr in outgoing]
    pin, min_ = np.stack([p for p, _ in tin]), np.stack([m for _, m in tin])
    pout, mout = np.stack([p for p, _ in tout]), np.stack([m for _, m in tout])
    n_valid = tuple(int(m.sum()) for m in (*min_, *mout))
    ignored = tuple(int(((tr.t >= g0) & (tr.t <= g1)).sum()) for tr in (*incoming, *outgoing))
    notes = []
    if any(ignored[:2]):
        notes.append("incoming observations at or after g0 were ignored")
    if any(ignored[2:]):
        notes.append("outgoing observations at or before g1 were ignored")
    for nm, tr in (("incoming", incoming), ("outgoing", outgoing)):
        for k, t in enumerate(tr):
            side_ok = (t.t < g0) if nm == "incoming" else (t.t > g1)
            if not side_ok.any():
                notes.append(f"{nm}[{k}] has no observation on its side of the gap")
    res = ModelInput(arrays={}, g0=g0, g1=g1, grid_in=grid_in, grid_out=grid_out, pitch_in=pin, pitch_out=pout, in_mask=min_,
                     out_mask=mout, n_valid=n_valid, origin=None, R=None, frame_mode=None, ignored_in_gap=ignored, notes=notes)
    if min(n_valid) < MIN_POINTS:
        res.notes.append(f"a tracklet has fewer than {MIN_POINTS} valid grid points in its {CONTEXT_S:.1f} s context")
        return res
    origin, R, mode = S.sample_frame(pin, min_, S.IN_REL)
    in_pos = np.where(min_[..., None], (pin - origin) @ R.T, 0.0).astype(np.float32)
    out_pos = np.where(mout[..., None], (pout - origin) @ R.T, 0.0).astype(np.float32)
    res.origin, res.R, res.frame_mode = origin, R, mode
    res.arrays = dict(in_pos=in_pos[None], in_mask=min_.astype(bool)[None], in_dt=S.IN_REL.astype(np.float32)[None],
                      out_pos=out_pos[None], out_mask=mout.astype(bool)[None], out_dt=S.OUT_REL.astype(np.float32)[None],
                      gap_s=np.array([g1 - g0], np.float32))
    return res


def swap_outgoing(incoming: Sequence[Track], outgoing: Sequence[Track]) -> tuple[list[Track], list[Track]]:
    return list(incoming), [outgoing[1], outgoing[0]]


def tracks_from_json(d: dict[str, Any]) -> tuple[list[Track], list[Track], float, float]:
    """``{"incoming": [{t, x, y}, {t, x, y}], "outgoing": [...], "gap": {"start_s": g0, "end_s": g1}}``; alternatively
    ``"gap": {"from_observations": true}`` derives ``[g0, g1]`` from the last incoming / first outgoing observation."""
    try:
        inc = [Track.from_lists(v["t"], v["x"], v["y"]) for v in d["incoming"]]
        out = [Track.from_lists(v["t"], v["x"], v["y"]) for v in d["outgoing"]]
        gap = d["gap"]
    except (KeyError, TypeError) as e:
        raise InputError(f"malformed trajectory JSON: {e}") from e
    if gap.get("from_observations"):
        if any(len(tr.t) == 0 for tr in (*inc, *out)):
            raise InputError("every tracklet needs observations")
        g0, g1 = gap_from_observations(max(tr.t[-1] for tr in inc), min(tr.t[0] for tr in out))
    else:
        try:
            g0, g1 = float(gap["start_s"]), float(gap["end_s"])
        except (KeyError, TypeError, ValueError) as e:
            raise InputError("gap needs start_s and end_s (seconds) or from_observations") from e
    return inc, out, g0, g1
