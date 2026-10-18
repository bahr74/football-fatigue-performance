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

__all__ = [
    "DerivedKinematics",
    "MovementCoverage",
    "MovementMetrics",
    "TrajectoryPoint",
    "classify_speed_zone",
    "compute_player_movement",
    "extract_player_trajectories",
    "summarize_match_movement",
]
