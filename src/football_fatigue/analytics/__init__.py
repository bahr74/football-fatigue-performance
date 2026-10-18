"""Tracking Analytics and Physical Movement Module (Z4).

Provides scientifically validated player kinematics, workload metrics, and quality-aware
coverage analysis on top of the MatchData layer.
"""

from football_fatigue.analytics.movement import (
    DerivedKinematics,
    MovementCoverage,
    MovementMetrics,
    TrajectoryPoint,
    classify_speed_zone,
    compute_player_movement,
    extract_player_trajectories,
    summarize_match_movement,
)
from football_fatigue.analytics.spatial import (
    SmoothedTrajectoryPoint,
    compute_observation_heatmap,
    compute_positional_changes,
    compute_spatial_analytics,
    compute_team_shape,
    smooth_trajectory_median,
)
from football_fatigue.analytics.cv import (
    build_cv_analytics,
    summarize_vision_movement,
    tactical_metric_readiness,
)

__all__ = [
    "DerivedKinematics",
    "MovementCoverage",
    "MovementMetrics",
    "SmoothedTrajectoryPoint",
    "TrajectoryPoint",
    "classify_speed_zone",
    "compute_player_movement",
    "compute_observation_heatmap",
    "compute_positional_changes",
    "compute_spatial_analytics",
    "compute_team_shape",
    "extract_player_trajectories",
    "summarize_match_movement",
    "smooth_trajectory_median",
    "build_cv_analytics",
    "summarize_vision_movement",
    "tactical_metric_readiness",
]
