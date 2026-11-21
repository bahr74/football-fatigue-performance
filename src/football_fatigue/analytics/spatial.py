"""Quality-aware spatial summaries for positional tracking.

Heatmap cells count usable position observations (not dwell time). Team shape
is descriptive geometry of observed locations and does not imply tactical role.
No coordinates are interpolated, clipped, or silently smoothed.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from football_fatigue.match_data import PreparedFrame
from football_fatigue.analytics.movement import TrajectoryPoint


@dataclass(frozen=True)
class SmoothedTrajectoryPoint:
    """Opt-in median-smoothed coordinate with its original sample retained."""

    original: TrajectoryPoint
    x: float
    y: float
    window_points: int
    flags: tuple[str, ...]

    def as_trajectory_point(self) -> TrajectoryPoint:
        return TrajectoryPoint(
            frame_num=self.original.frame_num,
            video_time_s=self.original.video_time_s,
            canonical_period=self.original.canonical_period,
            x=self.x,
            y=self.y,
            is_estimated=self.original.is_estimated,
            coord_source=f"median_smoothed:{self.original.coord_source}",
        )


def smooth_trajectory_median(
    points: Sequence[TrajectoryPoint],
    *,
    window_points: int = 3,
    max_gap_s: float = 0.5,
) -> list[SmoothedTrajectoryPoint]:
    """Return centered per-axis median coordinates without changing raw points.

    Smoothing is opt-in and is segmented at period changes and gaps larger
    than ``max_gap_s``. Non-increasing timestamps are rejected. Window edges are shortened and
    marked; singleton/short segments are passed through with an explicit flag.
    This operation does not restore missing observations or establish accuracy.
    """
    if window_points < 1 or window_points % 2 == 0:
        raise ValueError("window_points must be a positive odd integer.")
    if not math.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("max_gap_s must be finite and greater than zero.")
    if not points:
        return []
    if any(not all(math.isfinite(v) for v in (p.video_time_s, p.x, p.y)) for p in points):
        raise ValueError("trajectory coordinates and times must be finite.")
    if any(points[i].video_time_s <= points[i - 1].video_time_s for i in range(1, len(points))):
        raise ValueError("trajectory points must have strictly increasing timestamps.")

    segments: list[list[TrajectoryPoint]] = [[points[0]]]
    for prev, current in zip(points, points[1:]):
        dt = current.video_time_s - prev.video_time_s
        if current.canonical_period != prev.canonical_period or dt > max_gap_s:
            segments.append([current])
        else:
            segments[-1].append(current)

    radius = window_points // 2
    results: list[SmoothedTrajectoryPoint] = []
    for segment in segments:
        for index, original in enumerate(segment):
            if len(segment) < window_points:
                results.append(SmoothedTrajectoryPoint(
                    original, original.x, original.y, 1, ("SMOOTHING_WINDOW_SHORT",)
                ))
                continue
            start = max(0, index - radius)
            stop = min(len(segment), index + radius + 1)
            neighborhood = segment[start:stop]
            flags = ("SMOOTHING_EDGE_WINDOW_SHORTENED",) if len(neighborhood) < window_points else ()
            results.append(SmoothedTrajectoryPoint(
                original,
                statistics.median(p.x for p in neighborhood),
                statistics.median(p.y for p in neighborhood),
                len(neighborhood),
                flags,
            ))
    return results


def _usable_positions(frame: PreparedFrame) -> list[tuple[str, int, float, float, bool]]:
    positions = []
    for player in frame.players:
        if (
            player.position_usable
            and player.jersey_number is not None
            and player.source_side in {"home", "away"}
        ):
            x, y = float(player.selected.x), float(player.selected.y)
            if math.isfinite(x) and math.isfinite(y):
                positions.append((player.source_side, int(player.jersey_number), x, y,
                                  "PLAYER_ESTIMATED" in player.caveats))
    return positions


def _convex_hull_area(points: Sequence[tuple[float, float]]) -> float:
    unique = sorted(set(points))
    if len(unique) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    return abs(sum(
        hull[i][0] * hull[(i + 1) % len(hull)][1]
        - hull[(i + 1) % len(hull)][0] * hull[i][1]
        for i in range(len(hull))
    )) / 2.0


def compute_team_shape(
    frames: Iterable[PreparedFrame], *, sample_every_s: float = 1.0,
) -> list[dict[str, Any]]:
    """Describe team footprints at a fixed interval; missing players stay absent."""
    if not math.isfinite(sample_every_s) or sample_every_s <= 0:
        raise ValueError("sample_every_s must be finite and positive.")
    rows = []
    last_sample: dict[int | None, int] = {}
    for frame in frames:
        bucket = int(frame.video_time_s // sample_every_s)
        if last_sample.get(frame.canonical_period) == bucket:
            continue
        last_sample[frame.canonical_period] = bucket
        positions = _usable_positions(frame)
        for side in ("home", "away"):
            team = [(x, y, estimated) for team_side, _, x, y, estimated in positions if team_side == side]
            if not team:
                continue
            xs = [p[0] for p in team]
            ys = [p[1] for p in team]
            rows.append({
                "frame_num": frame.frame_num,
                "video_time_s": frame.video_time_s,
                "canonical_period": frame.canonical_period,
                "side": side,
                "observed_player_count": len(team),
                "estimated_player_count": sum(p[2] for p in team),
                "centroid_x_m": sum(xs) / len(xs),
                "centroid_y_m": sum(ys) / len(ys),
                "length_span_m": max(xs) - min(xs),
                "width_span_m": max(ys) - min(ys),
                "convex_hull_area_m2": _convex_hull_area([(p[0], p[1]) for p in team]),
            })
    return rows


def compute_observation_heatmap(
    frames: Iterable[PreparedFrame],
    *,
    pitch_length_m: float,
    pitch_width_m: float,
    columns: int = 21,
    rows: int = 14,
) -> dict[str, Any]:
    """Count usable player position observations in fixed pitch-coordinate cells."""
    if not math.isfinite(pitch_length_m) or pitch_length_m <= 0:
        raise ValueError("pitch_length_m must be finite and positive.")
    if not math.isfinite(pitch_width_m) or pitch_width_m <= 0:
        raise ValueError("pitch_width_m must be finite and positive.")
    if columns < 1 or rows < 1:
        raise ValueError("heatmap rows and columns must be positive.")
    counts = {side: [[0 for _ in range(columns)] for _ in range(rows)] for side in ("home", "away")}
    estimated = {side: 0 for side in ("home", "away")}
    outside_pitch = {side: 0 for side in ("home", "away")}
    for frame in frames:
        for side, _, x, y, is_estimated in _usable_positions(frame):
            col = int((x + pitch_length_m / 2) / pitch_length_m * columns)
            row = int((pitch_width_m / 2 - y) / pitch_width_m * rows)
            if col < 0 or col >= columns or row < 0 or row >= rows:
                outside_pitch[side] += 1
                continue
            counts[side][row][col] += 1
            estimated[side] += int(is_estimated)
    return {
        "measure": "position_observation_count",
        "unit": "usable player observations per cell",
        "origin": "pitch centre, metres",
        "pitch_length_m": pitch_length_m,
        "pitch_width_m": pitch_width_m,
        "columns": columns,
        "rows": rows,
        "counts": counts,
        "estimated_observations_in_grid": estimated,
        "usable_observations_outside_pitch_grid": outside_pitch,
    }


def compute_positional_changes(
    frames: Iterable[PreparedFrame],
    *,
    window_s: float = 300.0,
    minimum_observations_per_window: int = 2,
) -> list[dict[str, Any]]:
    """Compare per-player mean positions in adjacent explicit-duration windows.

    Windows restart at the first observed frame of each canonical period. Only
    adjacent windows with enough observations are compared. This is descriptive
    displacement of window means, not a tactical-value or fatigue judgement.
    """
    if not math.isfinite(window_s) or window_s <= 0:
        raise ValueError("window_s must be finite and positive.")
    if minimum_observations_per_window < 1:
        raise ValueError("minimum_observations_per_window must be positive.")
    materialized = list(frames)
    period_origins: dict[int, float] = {}
    bins: dict[tuple[int, int, str, int], list[tuple[float, float, bool]]] = {}
    for frame in materialized:
        period = frame.canonical_period
        if period is None:
            continue
        period_origins.setdefault(period, frame.video_time_s)
        bucket = int((frame.video_time_s - period_origins[period]) // window_s)
        for side, jersey, x, y, estimated in _usable_positions(frame):
            bins.setdefault((period, bucket, side, jersey), []).append((x, y, estimated))
    keys = sorted(bins)
    rows = []
    for period, bucket, side, jersey in keys:
        next_key = (period, bucket + 1, side, jersey)
        first = bins[(period, bucket, side, jersey)]
        second = bins.get(next_key, [])
        if len(first) < minimum_observations_per_window or len(second) < minimum_observations_per_window:
            continue
        a = (sum(p[0] for p in first) / len(first), sum(p[1] for p in first) / len(first))
        b = (sum(p[0] for p in second) / len(second), sum(p[1] for p in second) / len(second))
        rows.append({
            "canonical_period": period,
            "from_window": bucket,
            "to_window": bucket + 1,
            "window_s": window_s,
            "side": side,
            "jersey": jersey,
            "from_observations": len(first),
            "to_observations": len(second),
            "from_estimated_observations": sum(p[2] for p in first),
            "to_estimated_observations": sum(p[2] for p in second),
            "mean_x_from_m": a[0],
            "mean_y_from_m": a[1],
            "mean_x_to_m": b[0],
            "mean_y_to_m": b[1],
            "mean_position_change_m": math.hypot(b[0] - a[0], b[1] - a[1]),
        })
    return rows


def compute_spatial_analytics(
    frames: Sequence[PreparedFrame],
    *,
    pitch_length_m: float,
    pitch_width_m: float,
    heatmap_columns: int = 21,
    heatmap_rows: int = 14,
    positional_window_s: float = 300.0,
) -> dict[str, Any]:
    """Build spatial summaries from the exact prepared frames shown on the board."""
    return {
        "heatmap": compute_observation_heatmap(
            frames,
            pitch_length_m=pitch_length_m,
            pitch_width_m=pitch_width_m,
            columns=heatmap_columns,
            rows=heatmap_rows,
        ),
        "team_shape": compute_team_shape(frames),
        "positional_changes": compute_positional_changes(
            frames, window_s=positional_window_s,
        ),
        "scope": {
            "frame_count": len(frames),
            "heatmap_measure": "observation counts; not dwell time",
            "team_shape_source": "usable positions only; observed roster per frame may vary",
            "positional_change_definition": "distance between adjacent-window mean positions per side and jersey",
        },
    }
