"""Tracking Analytics & Movement Workload Module (Z4).

Calculates scientifically defensible movement metrics from validated positional tracking data.

Scientific & Engineering Rules (PROJECT_RULES.md, TASK 3 Section 6):
1. Effective position update cadence: handles repeated or unchanged positions
   without producing alternating velocity sawtooth artifacts.
2. Stationary vs dropout: zero displacement is kept distinct from missing data.
3. Missing, estimated, ambiguous handling: only usable positions are computed;
   estimated positions are tracked and reported in coverage summaries.
4. Valid temporal differences: delta_t must be strictly positive (> 0).
5. Gap exclusion policy: never connects positions across frame gaps (> max_gap_s)
   or across period boundaries (e.g. halftime); trajectories are split into valid segments.
6. Smoothing transparency: records coordinate source and derived filtering windows.
7. Speed spike rejection: rejects biologically impossible speeds (> 12 m/s) and
   accelerations (> 10 m/s^2), tallying spike counts rather than distorting metrics.
8. Observed vs derived separation: all physical metrics are explicitly derived
   and accompanied by coverage, confidence, and provenance metadata.
9. Coverage & confidence criteria: reports sample coverage fraction, estimated fraction,
   and flags low-confidence calculations.
10. Validated on synthetic trajectories and standard football sports science ranges:
   - Walking (< 2.0 m/s)
   - Jogging (2.0 - 4.0 m/s)
   - Running (4.0 - 5.5 m/s)
   - High-Speed Running (5.5 - 7.0 m/s)
   - Sprinting (>= 7.0 m/s)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence

from football_fatigue.match_data import PreparedFrame, PreparedPlayer


# Standard sports science velocity zones (in m/s)
ZONE_WALKING_MAX = 2.0         # 0.0 - 2.0 m/s (0.0 - 7.2 km/h)
ZONE_JOGGING_MAX = 4.0         # 2.0 - 4.0 m/s (7.2 - 14.4 km/h)
ZONE_RUNNING_MAX = 5.5         # 4.0 - 5.5 m/s (14.4 - 19.8 km/h)
ZONE_HSR_MAX = 7.0             # 5.5 - 7.0 m/s (19.8 - 25.2 km/h) (High-speed running)
# Sprinting is >= 7.0 m/s (>= 25.2 km/h)

# Human physiological limits for elite football tracking
PHYSIOLOGICAL_MAX_SPEED_M_S = 12.0     # ~43.2 km/h; speeds above this indicate noise/swaps
PHYSIOLOGICAL_MAX_ACCEL_M_S2 = 10.0    # accelerations above this indicate noise/swaps
ACCEL_BURST_THRESHOLD = 2.5            # threshold for high acceleration count
DECEL_BURST_THRESHOLD = -2.5           # threshold for high deceleration count
DEFAULT_MAX_GAP_S = 0.5                # max elapsed time before treating as a gap


@dataclass(frozen=True)
class TrajectoryPoint:
    """A single spatial observation of a player along a trajectory."""
    frame_num: int
    video_time_s: float
    canonical_period: int
    x: float
    y: float
    is_estimated: bool = False
    coord_source: str = "raw"


@dataclass(frozen=True)
class DerivedKinematics:
    """Instantaneous and windowed kinematic properties derived at a point."""
    frame_num: int
    video_time_s: float
    speed_m_s: float
    acceleration_m_s2: float | None
    is_valid: bool
    is_spike: bool
    zone: str


@dataclass
class MovementCoverage:
    """Data quality and confidence metadata for derived metrics."""
    total_frames_evaluated: int = 0
    usable_points: int = 0
    estimated_points: int = 0
    missing_or_ambiguous_points: int = 0
    gaps_detected: int = 0
    speed_spikes_rejected: int = 0

    @property
    def coverage_fraction(self) -> float:
        """Fraction of evaluated frames with usable tracking positions."""
        if self.total_frames_evaluated == 0:
            return 0.0
        return self.usable_points / self.total_frames_evaluated

    @property
    def estimated_fraction(self) -> float:
        """Fraction of usable tracking points that were estimated by the tracking system."""
        if self.usable_points == 0:
            return 0.0
        return self.estimated_points / self.usable_points

    @property
    def is_reliable(self) -> bool:
        """Heuristic check for scientific reliability: at least 60% coverage."""
        return self.coverage_fraction >= 0.60


@dataclass
class MovementMetrics:
    """Aggregated movement metrics and workload indicators for a player or segment."""
    total_distance_m: float = 0.0
    walking_distance_m: float = 0.0
    jogging_distance_m: float = 0.0
    running_distance_m: float = 0.0
    hsr_distance_m: float = 0.0          # High-speed running (5.5 - 7.0 m/s)
    sprint_distance_m: float = 0.0       # Sprinting (>= 7.0 m/s)
    max_speed_m_s: float = 0.0
    mean_speed_m_s: float = 0.0
    max_acceleration_m_s2: float = 0.0
    max_deceleration_m_s2: float = 0.0
    high_accel_events: int = 0           # Accel > 2.5 m/s^2
    high_decel_events: int = 0           # Decel < -2.5 m/s^2
    coverage: MovementCoverage = field(default_factory=MovementCoverage)

    @property
    def high_intensity_distance_m(self) -> float:
        """Combined High-Speed Running and Sprinting distance (>= 5.5 m/s)."""
        return self.hsr_distance_m + self.sprint_distance_m

    def as_dict(self) -> dict[str, Any]:
        """Convert metrics to a standard serialisable dictionary."""
        return {
            "total_distance_m": round(self.total_distance_m, 2),
            "walking_distance_m": round(self.walking_distance_m, 2),
            "jogging_distance_m": round(self.jogging_distance_m, 2),
            "running_distance_m": round(self.running_distance_m, 2),
            "hsr_distance_m": round(self.hsr_distance_m, 2),
            "sprint_distance_m": round(self.sprint_distance_m, 2),
            "high_intensity_distance_m": round(self.high_intensity_distance_m, 2),
            "max_speed_m_s": round(self.max_speed_m_s, 2),
            "mean_speed_m_s": round(self.mean_speed_m_s, 2),
            "max_acceleration_m_s2": round(self.max_acceleration_m_s2, 2),
            "max_deceleration_m_s2": round(self.max_deceleration_m_s2, 2),
            "high_accel_events": self.high_accel_events,
            "high_decel_events": self.high_decel_events,
            "coverage_fraction": round(self.coverage.coverage_fraction, 4),
            "estimated_fraction": round(self.coverage.estimated_fraction, 4),
            "gaps_detected": self.coverage.gaps_detected,
            "speed_spikes_rejected": self.coverage.speed_spikes_rejected,
            "is_reliable": self.coverage.is_reliable,
        }


def classify_speed_zone(speed: float) -> str:
    """Classify instantaneous or windowed speed into standard football velocity bands."""
    if speed < ZONE_WALKING_MAX:
        return "walking"
    elif speed < ZONE_JOGGING_MAX:
        return "jogging"
    elif speed < ZONE_RUNNING_MAX:
        return "running"
    elif speed < ZONE_HSR_MAX:
        return "hsr"
    else:
        return "sprint"


def compute_player_movement(
    points: Sequence[TrajectoryPoint],
    total_frames_evaluated: int | None = None,
    max_gap_s: float = DEFAULT_MAX_GAP_S,
    window_frames: int = 3,
    max_speed_m_s: float = PHYSIOLOGICAL_MAX_SPEED_M_S,
    max_accel_m_s2: float = PHYSIOLOGICAL_MAX_ACCEL_M_S2,
) -> tuple[MovementMetrics, list[DerivedKinematics]]:
    """Compute validated movement metrics for a chronological sequence of points.

    Parameters
    ----------
    points:
        List of chronological TrajectoryPoints where position was usable.
    total_frames_evaluated:
        The total number of match frames evaluated (used for denominator in coverage).
        If None, len(points) is used.
    max_gap_s:
        Maximum allowed time difference between consecutive points before treating
        as a tracking gap. Never connects positions across gaps.
    window_frames:
        Window size for smoothing velocity against cadence artifacts (e.g. repeated samples).
    max_speed_m_s:
        Physiological speed threshold for rejecting tracking artifacts/spikes.
    max_accel_m_s2:
        Physiological acceleration threshold for rejecting tracking artifacts/spikes.

    Returns
    -------
    (MovementMetrics, list[DerivedKinematics])
    """
    total_eval = total_frames_evaluated if total_frames_evaluated is not None else len(points)
    cov = MovementCoverage(
        total_frames_evaluated=total_eval,
        usable_points=len(points),
        estimated_points=sum(1 for p in points if p.is_estimated),
        missing_or_ambiguous_points=max(0, total_eval - len(points)),
    )

    if len(points) < 2:
        return MovementMetrics(coverage=cov), []

    # 1. Segment trajectory into continuous chunks (split by gaps or period breaks)
    segments: list[list[TrajectoryPoint]] = []
    current_segment: list[TrajectoryPoint] = [points[0]]

    for i in range(1, len(points)):
        prev_p = points[i - 1]
        curr_p = points[i]
        dt = curr_p.video_time_s - prev_p.video_time_s
        period_changed = curr_p.canonical_period != prev_p.canonical_period

        if dt <= 0 or dt > max_gap_s or period_changed:
            cov.gaps_detected += 1
            if current_segment:
                segments.append(current_segment)
            current_segment = [curr_p]
        else:
            current_segment.append(curr_p)

    if current_segment:
        segments.append(current_segment)

    # 2. Compute kinematics for each continuous segment
    all_kinematics: list[DerivedKinematics] = []
    total_dist = 0.0
    zone_dists = {"walking": 0.0, "jogging": 0.0, "running": 0.0, "hsr": 0.0, "sprint": 0.0}
    max_speed = 0.0
    max_accel = 0.0
    max_decel = 0.0
    high_accel_cnt = 0
    high_decel_cnt = 0
    valid_speeds: list[float] = []

    for seg in segments:
        if len(seg) < 2:
            continue

        # Compute point-to-point step displacements and raw speeds
        step_speeds: list[float] = []
        step_dts: list[float] = []
        step_dists: list[float] = []

        for j in range(1, len(seg)):
            p0 = seg[j - 1]
            p1 = seg[j]
            dx = p1.x - p0.x
            dy = p1.y - p0.y
            dist = math.hypot(dx, dy)
            dt = p1.video_time_s - p0.video_time_s

            raw_speed = dist / dt if dt > 0 else 0.0
            step_dists.append(dist)
            step_dts.append(dt)
            step_speeds.append(raw_speed)

        # Apply moving window filter to avoid cadence / repeated position artifacts
        # (e.g. 15 Hz update on 30 Hz recording producing alternating 0 and 2v)
        half_w = max(1, window_frames // 2)
        n_steps = len(step_speeds)

        prev_speed = 0.0
        for j in range(n_steps):
            w_start = max(0, j - half_w)
            w_end = min(n_steps, j + half_w + 1)
            w_dist = sum(step_dists[w_start:w_end])
            w_dt = sum(step_dts[w_start:w_end])

            filtered_speed = w_dist / w_dt if w_dt > 0 else 0.0
            dist = step_dists[j]
            dt = step_dts[j]

            # Spike detection
            is_spike = filtered_speed > max_speed_m_s
            if is_spike:
                cov.speed_spikes_rejected += 1
                all_kinematics.append(DerivedKinematics(
                    frame_num=seg[j + 1].frame_num,
                    video_time_s=seg[j + 1].video_time_s,
                    speed_m_s=filtered_speed,
                    acceleration_m_s2=None,
                    is_valid=False,
                    is_spike=True,
                    zone="spike",
                ))
                continue

            # Acceleration calculation
            accel = (filtered_speed - prev_speed) / dt if dt > 0 and j > 0 else None
            is_accel_spike = accel is not None and abs(accel) > max_accel_m_s2

            if is_accel_spike:
                cov.speed_spikes_rejected += 1
                accel = None

            zone = classify_speed_zone(filtered_speed)
            all_kinematics.append(DerivedKinematics(
                frame_num=seg[j + 1].frame_num,
                video_time_s=seg[j + 1].video_time_s,
                speed_m_s=filtered_speed,
                acceleration_m_s2=accel,
                is_valid=True,
                is_spike=False,
                zone=zone,
            ))

            # Accumulate metrics
            total_dist += dist
            zone_dists[zone] += dist
            valid_speeds.append(filtered_speed)
            if filtered_speed > max_speed:
                max_speed = filtered_speed

            if accel is not None:
                if accel > max_accel:
                    max_accel = accel
                if accel < max_decel:
                    max_decel = accel
                if accel >= ACCEL_BURST_THRESHOLD:
                    high_accel_cnt += 1
                elif accel <= DECEL_BURST_THRESHOLD:
                    high_decel_cnt += 1

            prev_speed = filtered_speed

    mean_spd = sum(valid_speeds) / len(valid_speeds) if valid_speeds else 0.0

    metrics = MovementMetrics(
        total_distance_m=total_dist,
        walking_distance_m=zone_dists["walking"],
        jogging_distance_m=zone_dists["jogging"],
        running_distance_m=zone_dists["running"],
        hsr_distance_m=zone_dists["hsr"],
        sprint_distance_m=zone_dists["sprint"],
        max_speed_m_s=max_speed,
        mean_speed_m_s=mean_spd,
        max_acceleration_m_s2=max_accel,
        max_deceleration_m_s2=max_decel,
        high_accel_events=high_accel_cnt,
        high_decel_events=high_decel_cnt,
        coverage=cov,
    )

    return metrics, all_kinematics


def extract_player_trajectories(
    frames: Iterable[PreparedFrame],
) -> dict[tuple[str, int], list[TrajectoryPoint]]:
    """Extract chronological trajectory points for each (source_side, jersey_number) pair.

    Only usable player coordinates (position_usable=True) are recorded.
    """
    trajectories: dict[tuple[str, int], list[TrajectoryPoint]] = {}

    for pf in frames:
        for p in pf.players:
            if not p.position_usable or p.jersey_number is None:
                continue
            key = (p.source_side, p.jersey_number)
            trajectories.setdefault(key, []).append(TrajectoryPoint(
                frame_num=pf.frame_num,
                video_time_s=pf.video_time_s,
                canonical_period=pf.canonical_period or 1,
                x=p.selected.x,
                y=p.selected.y,
                is_estimated="PLAYER_ESTIMATED" in p.caveats,
                coord_source=p.selected.source or "unknown",
            ))

    return trajectories


def summarize_match_movement(
    frames: Sequence[PreparedFrame],
    window_frames: int = 3,
) -> dict[tuple[str, int], MovementMetrics]:
    """Calculate movement metrics for all players across a sequence of PreparedFrames."""
    total_frames = len(frames)
    trajectories = extract_player_trajectories(frames)
    summaries: dict[tuple[str, int], MovementMetrics] = {}

    for player_key, points in trajectories.items():
        metrics, _ = compute_player_movement(
            points,
            total_frames_evaluated=total_frames,
            window_frames=window_frames,
        )
        summaries[player_key] = metrics

    return summaries
