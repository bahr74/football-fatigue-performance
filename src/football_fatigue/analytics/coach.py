"""Coach-configurable physical analytics on observed, eligibility-checked trajectory segments (M7.2).

Every threshold is a coach-selected *definition*, not a validated standard: :data:`PRESETS` holds documented examples only, and the
coach can change any value (:class:`CoachSettings`, :func:`settings_from_dict`). The same settings and samples always give the same
result (no randomness, no state).

Safeguards that no setting can switch off:

* only samples marked observed *and* coordinate-valid are used; anything else (interpolated / display-only positions, off-pitch or
  outside the calibrated region) breaks the run, so a metric never crosses a gap or bridges an estimate;
* a run is also broken at a gap longer than ``max_gap_s`` and at a physically implausible step;
* a result belongs to ONE segment (one tracklet, one identity hypothesis). Totals across segments are only produced for segments a
  human reviewer has confirmed as the same person (level L2); nothing is extrapolated to 90 minutes;
* positions must already be pitch metres; pixel coordinates are refused by :func:`segment_eligibility`;
* workload numbers are movement descriptors, never physiological fatigue.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Iterable, Sequence

import numpy as np

from football_fatigue.analytics import motion as _motion
from football_fatigue.analytics.observed import _smooth as _moving_average

SETTINGS_SCHEMA = "m72-coach-analytics-settings-v1"
RESULT_SCHEMA = "m72-coach-analytics-result-v1"
KMH = 3.6
SMOOTHING = ("none", "moving_average", "median") + _motion.STABILIZERS   # the last four: M7.3 opt-in stabilisers (motion.py)
ZONES = ("standing/walking", "jogging", "running", "high-speed running", "sprinting")
LEVELS = {
    "L0": "not eligible: blocked (see blockers)",
    "L1": "observed local segment: one tracklet, observed coordinate-valid samples only; an observed-segment result, not a player total",
    "L2": "human-reviewed aggregation: segments a reviewer confirmed as one person may be summed",
    "L3": "independently validated: no independent validation exists for this pipeline (always blocked)",
}


@dataclass(frozen=True)
class CoachSettings:
    hsr_kmh: float = 19.8                 # high-speed running starts here
    sprint_kmh: float = 25.2              # a sprint starts when smoothed speed reaches this ...
    sprint_exit_kmh: float = 23.4         # ... and ends when it falls below this (hysteresis exit)
    sprint_min_duration_s: float = 1.0
    sprint_min_distance_m: float = 0.0
    min_recovery_s: float = 0.5           # efforts separated by less than this are one effort
    repeated_window_s: float = 21.0       # maximum recovery between efforts of one repeated-sprint bout
    repeated_min_efforts: int = 3
    accel_mps2: float = 2.0               # acceleration event threshold
    decel_mps2: float = 2.0               # deceleration event threshold (magnitude)
    accel_min_duration_s: float = 0.5
    max_gap_s: float = 0.12               # a longer gap between observed samples breaks the run (25 fps: > 2 missing frames)
    smoothing: str = "moving_average"     # none | moving_average | median | M7.3 stabilisers (position, within a run only)
    smooth_window_s: float = 0.5
    min_segment_s: float = 4.0            # shorter segments are not analysed
    min_coverage: float = 0.8             # usable samples / samples expected over the segment span
    max_plausible_kmh: float = 45.0       # faster raw steps are artefacts and break the run
    jog_kmh: float = 7.2                  # zone edges below HSR: standing/walking < jog <= jogging < run <= running < HSR
    run_kmh: float = 14.4
    kalman_accel_noise: float = 27.0      # kalman_rts only: process noise (m^2/s^3); larger follows faster changes, smaller smooths more

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_RANGES = {
    "hsr_kmh": (5.0, 40.0), "sprint_kmh": (5.0, 45.0), "sprint_exit_kmh": (3.0, 45.0), "sprint_min_duration_s": (0.0, 10.0),
    "sprint_min_distance_m": (0.0, 100.0), "min_recovery_s": (0.0, 10.0), "repeated_window_s": (1.0, 300.0),
    "repeated_min_efforts": (2, 20), "accel_mps2": (0.5, 10.0), "decel_mps2": (0.5, 10.0), "accel_min_duration_s": (0.0, 5.0),
    "max_gap_s": (0.04, 1.0), "smooth_window_s": (0.0, 3.0), "min_segment_s": (0.0, 600.0), "min_coverage": (0.0, 1.0),
    "max_plausible_kmh": (20.0, 60.0), "jog_kmh": (1.0, 20.0), "run_kmh": (2.0, 30.0), "kalman_accel_noise": (0.5, 200.0),
}

PRESETS: dict[str, dict[str, Any]] = {
    "example_adult_common": {
        "label": "Example A: commonly reported adult thresholds",
        "description": "HSR 19.8 km/h and sprint 25.2 km/h, values widely reported in football time-motion research. A coach-selected "
                       "definition, not a validated standard for this camera system.",
        "settings": CoachSettings().as_dict(),
    },
    "example_lower_intensity": {
        "label": "Example B: lower thresholds (e.g. youth or lower-intensity squads)",
        "description": "HSR 18.0 km/h and sprint 21.0 km/h. An illustrative coach choice only; set your own squad-specific values.",
        "settings": CoachSettings(hsr_kmh=18.0, sprint_kmh=21.0, sprint_exit_kmh=19.5, jog_kmh=6.0, run_kmh=12.0).as_dict(),
    },
    "example_strict_events": {
        "label": "Example C: strict event definitions",
        "description": "Sprint 25.2 km/h held 2 s and 10 m, acceleration events 3 m/s^2. Fewer, more certain events; still a coach choice.",
        "settings": CoachSettings(sprint_min_duration_s=2.0, sprint_min_distance_m=10.0, accel_mps2=3.0, decel_mps2=3.0).as_dict(),
    },
}


def settings_from_dict(d: dict[str, Any] | None) -> tuple[CoachSettings | None, list[str]]:
    """Validated settings, or (None, errors). Unknown keys, wrong types, out-of-range or inconsistent values are errors (never clipped)."""
    d = dict(d or {})
    errors = []
    names = {f.name: f for f in fields(CoachSettings)}
    for k in d:
        if k not in names:
            errors.append(f"unknown setting {k!r}")
    vals = CoachSettings().as_dict()
    for k, v in d.items():
        if k not in names:
            continue
        if k == "smoothing":
            if v not in SMOOTHING:
                errors.append(f"smoothing must be one of {', '.join(SMOOTHING)}")
            else:
                vals[k] = v
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            errors.append(f"{k} must be a finite number")
            continue
        lo, hi = _RANGES[k]
        if k == "repeated_min_efforts":
            if int(v) != v:
                errors.append(f"{k} must be a whole number")
                continue
            v = int(v)
        if not lo <= v <= hi:
            errors.append(f"{k}={v} is outside [{lo}, {hi}]")
            continue
        vals[k] = v
    if not errors:
        if vals["sprint_exit_kmh"] > vals["sprint_kmh"]:
            errors.append("sprint_exit_kmh must not exceed sprint_kmh (hysteresis exit)")
        if vals["hsr_kmh"] > vals["sprint_kmh"]:
            errors.append("hsr_kmh must not exceed sprint_kmh")
        if not vals["jog_kmh"] < vals["run_kmh"] < vals["hsr_kmh"]:
            errors.append("zone edges must satisfy jog_kmh < run_kmh < hsr_kmh")
        if vals["max_plausible_kmh"] <= vals["sprint_kmh"]:
            errors.append("max_plausible_kmh must exceed sprint_kmh")
    return (None, errors) if errors else (CoachSettings(**vals), [])


# ------------------------------------------------------------------------------------------------------------- kinematics
def _median_filter(x: np.ndarray, k: int) -> np.ndarray:
    ext = np.r_[2 * x[0] - x[k:0:-1], x, 2 * x[-1] - x[-2:-k - 2:-1]]   # odd reflection, as the moving average (no edge shrinkage)
    return np.median(np.lib.stride_tricks.sliding_window_view(ext, 2 * k + 1), axis=1)


def _smooth(x: np.ndarray, t: np.ndarray, s: CoachSettings) -> np.ndarray:
    if s.smoothing == "none" or s.smooth_window_s <= 0 or len(x) < 3:
        return x.copy()
    if s.smoothing == "moving_average":
        return _moving_average(x, t, s.smooth_window_s)
    dt = float(np.median(np.diff(t)))
    k = min(max(1, int(round(s.smooth_window_s / dt)) // 2), len(x) - 1)
    return _median_filter(x, k)


def runs(t_s, x_m, y_m, s: CoachSettings, usable: Sequence[bool] | None = None) -> list[dict[str, np.ndarray]]:
    """Observed runs (time, smoothed position, speed m/s, acceleration m/s^2). Unusable samples are dropped and break the run.

    The M7.3 stabilisers (``smoothing`` in :data:`football_fatigue.analytics.motion.STABILIZERS`) split only at unusable samples, gaps and
    persistent jumps (a noisy single step is down-weighted, not a cut), drop runs shorter than 0.8 s and cut out stretches whose
    stabilised speed exceeds ``max_plausible_kmh``; see :func:`football_fatigue.analytics.motion.stabilized_runs`."""
    t, x, y = (np.asarray(v, float) for v in (t_s, x_m, y_m))
    ok = np.ones(len(t), bool) if usable is None else np.asarray(usable, bool)
    if s.smoothing in _motion.STABILIZERS:
        cfg = _motion.MotionConfig(method=s.smoothing, window_s=max(s.smooth_window_s, 0.08), accel_noise=s.kalman_accel_noise)
        return _motion.stabilized_runs(t, x, y, cfg, max_gap_s=s.max_gap_s, max_plausible_kmh=s.max_plausible_kmh, usable=ok)
    # cut before an unusable sample, at a long gap, at non-increasing time and at an implausible raw step
    brk = np.zeros(len(t), bool)
    if len(t) > 1:
        dt = np.diff(t)
        step = np.hypot(np.diff(x), np.diff(y)) / np.where(dt > 0, dt, np.nan)
        brk[1:] = (dt > s.max_gap_s) | (dt <= 0) | ~(step <= s.max_plausible_kmh / KMH) | ~ok[1:] | ~ok[:-1]
    out = []
    edges = np.r_[0, np.flatnonzero(brk), len(t)]
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 2 or not ok[a:b].all():
            continue
        ts = t[a:b]
        xs, ys = _smooth(x[a:b], ts, s), _smooth(y[a:b], ts, s)
        v = np.hypot(np.gradient(xs, ts), np.gradient(ys, ts))
        out.append({"t": ts, "x": xs, "y": ys, "speed": v, "accel": np.gradient(v, ts),
                    "raw_distance": float(np.hypot(np.diff(x[a:b]), np.diff(y[a:b])).sum())})
    return out


def _above(t: np.ndarray, flag: np.ndarray) -> list[tuple[int, int]]:
    """Index ranges (inclusive) where flag is true."""
    out, s = [], None
    for i, f in enumerate(flag):
        if f and s is None:
            s = i
        elif not f and s is not None:
            out.append((s, i - 1)); s = None
    if s is not None:
        out.append((s, len(flag) - 1))
    return out


def _sprints(t: np.ndarray, v: np.ndarray, s: CoachSettings) -> list[tuple[int, int]]:
    ev, on, a = [], False, 0
    enter, leave = s.sprint_kmh / KMH, s.sprint_exit_kmh / KMH
    for i, vi in enumerate(v):
        if not on and vi >= enter:
            on, a = True, i
        elif on and vi < leave:
            ev.append((a, i - 1)); on = False
    if on:
        ev.append((a, len(v) - 1))
    merged: list[tuple[int, int]] = []
    for a, b in ev:
        if merged and t[a] - t[merged[-1][1]] < s.min_recovery_s:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


def analyse_segment(t_s, x_m, y_m, s: CoachSettings, usable: Sequence[bool] | None = None, series_step: int = 1) -> dict[str, Any]:
    """All metrics of ONE eligible segment (metres, seconds). Values are observed-segment results, never player totals."""
    rs = runs(t_s, x_m, y_m, s, usable)
    edges = np.array([s.jog_kmh, s.run_kmh, s.hsr_kmh, s.sprint_kmh]) / KMH
    zone_d, zone_t = np.zeros(5), np.zeros(5)
    dist = raw = obs_t = vmax = 0.0
    sprints, accels, decels, hsr_efforts, series = [], [], [], [], []
    hist = np.zeros(int(math.ceil(s.max_plausible_kmh)) + 1)
    for r in rs:
        t, v, acc = r["t"], r["speed"], r["accel"]
        seg = np.hypot(np.diff(r["x"]), np.diff(r["y"]))
        vm = (v[1:] + v[:-1]) / 2
        dts = np.diff(t)
        z = np.searchsorted(edges, vm, side="right")
        np.add.at(zone_d, z, seg); np.add.at(zone_t, z, dts)
        dist += float(seg.sum()); raw += r["raw_distance"]; obs_t += float(t[-1] - t[0]); vmax = max(vmax, float(v.max()))
        np.add.at(hist, np.minimum((vm * KMH).astype(int), len(hist) - 1), dts)
        series.append([[round(float(a), 3), round(float(b) * KMH, 2)] for a, b in zip(t[::series_step], v[::series_step])])
        for a, b in _sprints(t, v, s):
            d = float(seg[a:b].sum())
            if t[b] - t[a] >= s.sprint_min_duration_s and d >= s.sprint_min_distance_m:
                sprints.append({"start_s": float(t[a]), "end_s": float(t[b]), "duration_s": round(float(t[b] - t[a]), 3),
                                "distance_m": round(d, 2), "peak_kmh": round(float(v[a:b + 1].max()) * KMH, 2)})
        for a, b in _above(t, v >= s.hsr_kmh / KMH):
            if t[b] - t[a] >= s.sprint_min_duration_s:
                hsr_efforts.append({"start_s": float(t[a]), "end_s": float(t[b]), "distance_m": round(float(seg[a:b].sum()), 2)})
        for out, flag, sign in ((accels, acc >= s.accel_mps2, 1), (decels, acc <= -s.decel_mps2, -1)):
            for a, b in _above(t, flag):
                if t[b] - t[a] >= s.accel_min_duration_s:
                    out.append({"start_s": float(t[a]), "end_s": float(t[b]), "duration_s": round(float(t[b] - t[a]), 3),
                                "peak_mps2": round(float(sign * acc[a:b + 1].max() if sign > 0 else acc[a:b + 1].min()), 2)})
    # repeated efforts: consecutive sprints whose recovery (inside this segment) is <= repeated_window_s
    recoveries = [round(b["start_s"] - a["end_s"], 3) for a, b in zip(sprints, sprints[1:])]
    bouts, cur = [], [0] if sprints else []
    for i, rec in enumerate(recoveries):
        if rec <= s.repeated_window_s:
            cur.append(i + 1)
        else:
            if len(cur) >= s.repeated_min_efforts:
                bouts.append(cur)
            cur = [i + 1]
    if len(cur) >= s.repeated_min_efforts:
        bouts.append(cur)
    work_t = float(zone_t[3:].sum())
    return {
        "observed_time_s": round(obs_t, 3), "runs": len(rs),
        "distance_m": round(dist, 2), "raw_distance_m": round(raw, 2),
        "distance_by_zone_m": {k: round(float(v), 2) for k, v in zip(ZONES, zone_d)},
        "time_by_zone_s": {k: round(float(v), 3) for k, v in zip(ZONES, zone_t)},
        "hsr_distance_m": round(float(zone_d[3:].sum()), 2), "hsr_efforts": hsr_efforts,
        "speed": {"max_kmh": round(vmax * KMH, 2), "average_kmh": round(dist / obs_t * KMH, 2) if obs_t > 0 else None,
                  "distribution_s_per_kmh": [round(float(v), 3) for v in np.trim_zeros(hist, "b")]},
        "sprints": sprints, "sprint_count": len(sprints), "sprint_distance_m": round(sum(x["distance_m"] for x in sprints), 2),
        "recoveries_s": recoveries,
        "repeated_sprint_bouts": [{"efforts": len(b), "start_s": sprints[b[0]]["start_s"], "end_s": sprints[b[-1]]["end_s"]} for b in bouts],
        "accelerations": accels, "decelerations": decels,
        "work_rest": {"work_s": round(work_t, 3), "rest_s": round(float(zone_t[:3].sum()), 3),
                      "ratio": round(work_t / float(zone_t[:3].sum()), 4) if zone_t[:3].sum() > 0 else None},
        "workload": {"distance_per_observed_min_m": round(dist / obs_t * 60, 1) if obs_t > 0 else None,
                     "high_intensity_share": round(work_t / obs_t, 4) if obs_t > 0 else None,
                     "note": "movement descriptor of the observed segment, not physiological load or fatigue"},
        "speed_series_kmh": series,
    }


# ------------------------------------------------------------------------------------------------------------ eligibility
def segment_eligibility(*, coordinate_unit: str, timestamps_valid: bool, identity_safe: bool, role_ok: bool,
                        n_expected: int, n_usable: int, duration_s: float, s: CoachSettings,
                        reviewed_same_person: bool = False) -> dict[str, Any]:
    """Eligibility level of one segment and the blockers that stop a higher level."""
    blockers, codes = [], []
    cov = n_usable / n_expected if n_expected else 0.0
    for bad, code, msg in (
            (coordinate_unit != "metre", "not_metres", "positions are not pitch metres"),
            (not timestamps_valid, "timestamps", "timestamps are not valid"),
            (not identity_safe, "identity", "not one provenance-safe tracklet segment (link, split, jump or gap inside)"),
            (not role_ok, "role", "not an outfield player or goalkeeper (officials, staff and unknown roles are excluded)"),
            (duration_s < s.min_segment_s, "too_short", f"segment {duration_s:.1f} s is shorter than the minimum {s.min_segment_s:g} s"),
            (cov < s.min_coverage, "coverage", f"coordinate-valid coverage {cov:.0%} is below {s.min_coverage:.0%}")):
        if bad:
            blockers.append(msg); codes.append(code)
    level = "L0" if blockers else ("L2" if reviewed_same_person else "L1")
    above = []
    if level in ("L0", "L1"):
        above.append("L2 needs a human review confirming the identity")
    above.append("L3 needs independent validation, which does not exist")
    return {"level": level, "blockers": blockers, "blocker_codes": codes, "next_level_blockers": above, "coverage": round(cov, 4)}


def aggregate(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Sum of several segment results (only for L2 segments reviewed as one person; the caller enforces that)."""
    res = list(results)
    out: dict[str, Any] = {"segments": len(res)}
    for k in ("observed_time_s", "distance_m", "hsr_distance_m", "sprint_count", "sprint_distance_m"):
        out[k] = round(sum(r[k] for r in res), 3)
    out["max_kmh"] = max((r["speed"]["max_kmh"] for r in res), default=None)
    out["accelerations"] = sum(len(r["accelerations"]) for r in res)
    out["decelerations"] = sum(len(r["decelerations"]) for r in res)
    out["note"] = "sum over observed segments only; unobserved time is not estimated and nothing is extrapolated"
    return out
