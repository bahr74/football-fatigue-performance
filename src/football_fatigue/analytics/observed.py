"""Provenance-safe observed analytics (M7 sprint).

Two families, kept apart:

* **Spatial descriptors** (zone occupancy, team centroid / spread, visible-player counts) on pitch metres from an approximate calibration.
  They describe the *visible* subset only and always return the number of contributing observations.
* **Physical metrics** (distance, speed, acceleration, high-speed running, sprints, repeated efforts, recovery). The formulas are
  timestamp-aware and never accumulate across an unobserved gap; they are verified on synthetic trajectories. Real-match values are only
  produced when :func:`physical_metrics_gate` passes (metric coordinates with acceptable calibration error, valid timestamps, enough observed
  coverage, a provenance-safe identity segment). Otherwise the caller must show the metric as unavailable with the returned reasons.

Physical workload proxies are not physiological fatigue.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class PhysicalConfig:
    max_gap_s: float = 0.12            # consecutive samples further apart are not joined (25 fps -> > 2 missing frames breaks the run)
    smooth_window_s: float = 0.5       # centred moving average of position before differentiating (within a run only)
    hsr_mps: float = 5.5               # high-speed running threshold, m/s (19.8 km/h)
    sprint_enter_mps: float = 7.0      # sprint starts when smoothed speed reaches this (25.2 km/h) ...
    sprint_exit_mps: float = 6.5       # ... and ends when it falls below this (hysteresis)
    sprint_min_duration_s: float = 1.0
    sprint_merge_gap_s: float = 0.5    # two sprints separated by less than this (and within one run) are one effort
    rhie_max_recovery_s: float = 21.0  # repeated high-intensity effort: >= rhie_min_efforts efforts with recoveries <= this
    rhie_min_efforts: int = 3
    max_plausible_mps: float = 12.5    # faster samples are treated as tracking/calibration artefacts and break the run

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _runs(t: np.ndarray, cfg: PhysicalConfig) -> list[slice]:
    """Index ranges of samples joined by gaps <= max_gap_s (strictly increasing time required)."""
    if len(t) == 0:
        return []
    cut = np.flatnonzero((np.diff(t) > cfg.max_gap_s) | (np.diff(t) <= 0)) + 1
    edges = np.r_[0, cut, len(t)]
    return [slice(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def _smooth(x: np.ndarray, t: np.ndarray, window_s: float) -> np.ndarray:
    if len(x) < 3 or window_s <= 0:
        return x.copy()
    dt = float(np.median(np.diff(t)))
    k = min(max(1, int(round(window_s / dt)) // 2), len(x) - 1)
    # odd reflection about the end samples: keeps the endpoints and reproduces linear motion exactly (no shrinkage at run edges)
    ext = np.r_[2 * x[0] - x[k:0:-1], x, 2 * x[-1] - x[-2:-k - 2:-1]]
    return np.convolve(ext, np.full(2 * k + 1, 1.0 / (2 * k + 1)), mode="valid")


def kinematics(t_s: Sequence[float], x_m: Sequence[float], y_m: Sequence[float], cfg: PhysicalConfig = PhysicalConfig()) -> list[dict[str, np.ndarray]]:
    """Per observed run: time, smoothed position, speed (m/s) and acceleration (m/s^2). Runs never span a gap or an implausible jump."""
    t, x, y = (np.asarray(v, float) for v in (t_s, x_m, y_m))
    out = []
    for r in _runs(t, cfg):
        tt, xx, yy = t[r], x[r], y[r]
        # break further at physically implausible raw steps
        step = np.hypot(np.diff(xx), np.diff(yy)) / np.diff(tt) if len(tt) > 1 else np.array([])
        cuts = np.r_[0, np.flatnonzero(step > cfg.max_plausible_mps) + 1, len(tt)]
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b - a < 2:
                continue
            ts, xs, ys = tt[a:b], _smooth(xx[a:b], tt[a:b], cfg.smooth_window_s), _smooth(yy[a:b], tt[a:b], cfg.smooth_window_s)
            v = np.hypot(np.gradient(xs, ts), np.gradient(ys, ts))
            out.append({"t": ts, "x": xs, "y": ys, "speed": v, "accel": np.gradient(v, ts),
                        "raw_distance": float(np.hypot(np.diff(xx[a:b]), np.diff(yy[a:b])).sum())})
    return out


def _events(t: np.ndarray, v: np.ndarray, cfg: PhysicalConfig) -> list[tuple[int, int]]:
    """Sprint events (start, end index inclusive) with hysteresis, minimum duration and merge rule, inside one run."""
    ev, on, s = [], False, 0
    for i, vi in enumerate(v):
        if not on and vi >= cfg.sprint_enter_mps:
            on, s = True, i
        elif on and vi < cfg.sprint_exit_mps:
            ev.append((s, i - 1)); on = False
    if on:
        ev.append((s, len(v) - 1))
    merged: list[tuple[int, int]] = []
    for a, b in ev:
        if merged and t[a] - t[merged[-1][1]] < cfg.sprint_merge_gap_s:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return [(a, b) for a, b in merged if t[b] - t[a] >= cfg.sprint_min_duration_s]


def physical_summary(t_s, x_m, y_m, cfg: PhysicalConfig = PhysicalConfig()) -> dict[str, Any]:
    """Distance / HSR / sprint / repeated-effort summary of ONE provenance-safe trajectory in metres and seconds."""
    runs = kinematics(t_s, x_m, y_m, cfg)
    dist = hsr = 0.0
    sprints = []
    observed_s = 0.0
    vmax = 0.0
    for r in runs:
        t, v = r["t"], r["speed"]
        seg = np.hypot(np.diff(r["x"]), np.diff(r["y"]))
        vm = (v[1:] + v[:-1]) / 2
        dist += float(seg.sum())
        hsr += float(seg[vm >= cfg.hsr_mps].sum())
        observed_s += float(t[-1] - t[0])
        vmax = max(vmax, float(v.max()))
        for a, b in _events(t, v, cfg):
            sprints.append({"start_s": float(t[a]), "end_s": float(t[b]), "duration_s": float(t[b] - t[a]),
                            "distance_m": float(seg[a:b].sum()), "peak_speed_mps": float(v[a:b + 1].max())})
    recoveries = [b["start_s"] - a["end_s"] for a, b in zip(sprints, sprints[1:])]
    rhie, cur = 0, 1
    for rec in recoveries:
        if rec <= cfg.rhie_max_recovery_s:
            cur += 1
        else:
            rhie += cur >= cfg.rhie_min_efforts
            cur = 1
    rhie += bool(sprints) and cur >= cfg.rhie_min_efforts
    return {"observed_time_s": observed_s, "runs": len(runs), "distance_m": dist, "hsr_distance_m": hsr, "max_speed_mps": vmax,
            "sprint_count": len(sprints), "sprints": sprints, "sprint_distance_m": float(sum(s["distance_m"] for s in sprints)),
            "recovery_intervals_s": recoveries, "rhie_bouts": int(rhie), "config": cfg.as_dict()}


def window_summaries(t_s, x_m, y_m, window_s: float = 300.0, cfg: PhysicalConfig = PhysicalConfig()) -> list[dict[str, Any]]:
    """Physical summary per time window (samples are assigned by time; runs are cut at window edges, never bridged)."""
    t = np.asarray(t_s, float)
    if not len(t):
        return []
    out = []
    w0 = math.floor(t.min() / window_s) * window_s
    while w0 <= t.max():
        m = (t >= w0) & (t < w0 + window_s)
        s = physical_summary(t[m], np.asarray(x_m, float)[m], np.asarray(y_m, float)[m], cfg)
        out.append({"window_start_s": w0, "window_s": window_s, **{k: s[k] for k in ("observed_time_s", "distance_m", "hsr_distance_m", "sprint_count")}})
        w0 += window_s
    return out


def physical_metrics_gate(*, coordinate_unit: str, calibration_gate_met: bool, timestamps_valid: bool, observed_coverage: float,
                          identity_attributable: bool, min_coverage: float = 0.8) -> dict[str, Any]:
    """Whether real-match physical metrics may be shown; reasons are returned when not."""
    reasons = []
    if coordinate_unit != "metre":
        reasons.append("positions are not in metres")
    if not calibration_gate_met:
        reasons.append("calibration error exceeds the distance/speed gate (median <= 0.5 m and p90 <= 1.0 m held out)")
    if not timestamps_valid:
        reasons.append("timestamps are not valid")
    if observed_coverage < min_coverage:
        reasons.append(f"observed coverage {observed_coverage:.0%} is below {min_coverage:.0%}")
    if not identity_attributable:
        reasons.append("the trajectory is not a provenance-safe attributable identity segment")
    return {"available": not reasons, "reasons": reasons}


# ------------------------------------------------------------------------------------------------ spatial descriptors
def zone_occupancy(x_m, y_m, nx: int = 6, ny: int = 4, length: float = 105.0, width: float = 68.0) -> dict[str, Any]:
    """Counts of observations per pitch zone (nx x ny equal zones; corner origin). Points outside the pitch are not counted."""
    x, y = np.asarray(x_m, float), np.asarray(y_m, float)
    ok = (x >= 0) & (x <= length) & (y >= 0) & (y <= width)
    ix = np.minimum((x[ok] / length * nx).astype(int), nx - 1)
    iy = np.minimum((y[ok] / width * ny).astype(int), ny - 1)
    grid = np.zeros((ny, nx), int)
    np.add.at(grid, (iy, ix), 1)
    return {"grid": grid.tolist(), "n": int(ok.sum()), "excluded_outside": int((~ok).sum()), "zone_m": [length / nx, width / ny]}


def team_shape(x_m, y_m) -> dict[str, Any] | None:
    """Centroid, length (x spread), width (y spread) and mean distance to the centroid of the VISIBLE players of one team in one frame."""
    x, y = np.asarray(x_m, float), np.asarray(y_m, float)
    if len(x) < 2:
        return None
    cx, cy = float(x.mean()), float(y.mean())
    return {"n_visible": int(len(x)), "centroid_m": [cx, cy], "length_m": float(x.max() - x.min()), "width_m": float(y.max() - y.min()),
            "mean_dist_to_centroid_m": float(np.hypot(x - cx, y - cy).mean())}
