"""Player-level analytics from observed positions only.

Inputs are per-instant observations of one short-term track (or of every track for a density map). Nothing is interpolated:
trajectories break at gaps and implausible jumps, heatmaps count observations, and physical metrics that need validated
calibration and persistent identity are returned as explicitly unavailable.
"""

from __future__ import annotations

import math

# Required before a physical metric may be shown; all are currently unmet for the Wadi Degla footage.
PHYSICAL_PREREQUISITES = (
    "metric scale verified (pitch length/width measured)",
    "position error below 0.5 m on independent held-out landmarks",
    "persistent identity verified for the player over the interval",
    "synchronised, gap-free observations at >= 5 Hz",
)
CONDITIONAL_METRICS = {
    "distance_covered_m": "sum of |p(t+1) - p(t)| over a smoothed, gap-free trajectory",
    "speed_mps": "|dp/dt| after smoothing (e.g. Savitzky-Golay, 1 s window)",
    "acceleration_mps2": "d(speed)/dt after smoothing",
    "high_speed_running_m": "distance at speed > 5.5 m/s (19.8 km/h)",
    "sprint_count": "episodes with speed > 7 m/s (25.2 km/h) for >= 1 s",
    "movement_intensity": "share of time per speed band",
    "workload_indicators": "accumulated distance/accelerations per minute",
}
NOT_VALIDATED = ("full-match physical performance", "persistent identity across camera transitions", "individual fatigue score")


def trajectory_segments(points: list[dict], max_gap_s: float = 0.45, max_speed_mps: float = 12.0, slack_m: float = 2.0) -> list[list[dict]]:
    """points: dicts with 't' (s), 'p' (x, y) and optional 'break_before' (identity uncertainty, e.g. a changed cross-camera
    partner). A new segment starts at a time gap > max_gap_s, at a jump faster than max_speed_mps (+ slack for position noise),
    or where break_before is set. Single points stay as one-point segments (observed, not connected)."""
    pts = sorted((p for p in points if p.get("p") is not None), key=lambda p: p["t"])
    segs: list[list[dict]] = []
    for p in pts:
        if segs:
            q = segs[-1][-1]; dt = p["t"] - q["t"]; d = math.dist(p["p"], q["p"])
            if dt <= max_gap_s and d <= max_speed_mps * dt + slack_m and not p.get("break_before"):
                segs[-1].append(p); continue
        segs.append([p])
    return segs


def heatmap(points: list[tuple[float, float]], x_range: tuple[float, float], y_range: tuple[float, float], cell_m: float = 3.0) -> dict:
    nx = max(1, int(math.ceil((x_range[1] - x_range[0]) / cell_m))); ny = max(1, int(math.ceil((y_range[1] - y_range[0]) / cell_m)))
    grid = [[0] * nx for _ in range(ny)]; outside = 0
    for x, y in points:
        i = int((x - x_range[0]) // cell_m); j = int((y - y_range[0]) // cell_m)
        if 0 <= i < nx and 0 <= j < ny:
            grid[j][i] += 1
        else:
            outside += 1
    return {"x0": x_range[0], "y0": y_range[0], "cell_m": cell_m, "nx": nx, "ny": ny, "counts": grid,
            "observations": len(points) - outside, "outside": outside}


def player_metrics(obs: list[dict], interval_s: float, step_s: float, cell_m: float = 3.0) -> dict:
    """obs: one short-term track's per-instant observations with 't', 'p', 'conf', 'state' (matched/ambiguous/...-only).
    Returns metrics grouped by validation level; nothing here is a physical-performance measurement."""
    n = len(obs)
    states = [o.get("state", "") for o in obs]
    matched = sum(s == "matched" for s in states); amb = sum(s == "ambiguous" for s in states)
    confs = sorted(o["conf"] for o in obs if o.get("conf") is not None)
    q = lambda f: round(confs[min(len(confs) - 1, int(f * len(confs)))], 3) if confs else None
    segs = trajectory_segments([{"t": o["t"], "p": o["p"]} for o in obs if o.get("p") is not None])
    covered = sum(s[-1]["t"] - s[0]["t"] + step_s for s in segs)
    cells = {(int(o["p"][0] // cell_m), int(o["p"][1] // cell_m)) for o in obs if o.get("p") is not None}
    expected = int(round(interval_s / step_s)) + 1 if interval_s > 0 else 0
    experimental = {
        "observation_count": n,
        "time_observed_s": round(n * step_s, 2),
        "visibility_ratio": round(n / expected, 3) if expected else None,
        "instants_supported_by_both_cameras": matched,
        "instants_ambiguous": amb,
        "instants_single_camera": n - matched - amb,
        "trajectory_segments": len(segs),
        "trajectory_coverage_s": round(covered, 2),
        "detection_confidence": {"p10": q(0.1), "median": q(0.5), "p90": q(0.9)},
        "approx_area_occupied_m2": round(len(cells) * cell_m * cell_m, 1),
        "first_observed_s": obs[0]["t"] if obs else None,
        "last_observed_s": obs[-1]["t"] if obs else None,
    }
    conditional = {k: {"status": "unavailable", "formula": v, "missing": list(PHYSICAL_PREREQUISITES)} for k, v in CONDITIONAL_METRICS.items()}
    return {"experimental": experimental, "conditional": conditional, "not_validated": list(NOT_VALIDATED)}
