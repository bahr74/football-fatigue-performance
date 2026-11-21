"""Quality-aware analytics adapters for vision tracking output."""

from __future__ import annotations

import math
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Iterable

from football_fatigue.analytics.movement import TrajectoryPoint, compute_player_movement
from football_fatigue.vision.models import TeamSide, VisionFrame


def summarize_vision_movement(
    frames: Iterable[VisionFrame],
    *,
    canonical_period: int = 1,
    max_gap_s: float = 0.5,
) -> dict[str, Any]:
    """Calculate Stage 4 movement indicators for usable CV player tracks."""
    if canonical_period not in (1, 2, 3, 4):
        raise ValueError("canonical_period must be 1, 2, 3, or 4.")
    materialized = list(frames)
    by_track: dict[int, list[TrajectoryPoint]] = {}
    for frame in materialized:
        for player in frame.players:
            if not player.is_usable or player.pitch_x is None or player.pitch_y is None:
                continue
            if not all(math.isfinite(float(value)) for value in (player.pitch_x, player.pitch_y)):
                continue
            by_track.setdefault(player.track_id, []).append(TrajectoryPoint(
                frame_num=frame.frame_num,
                video_time_s=frame.timestamp_s,
                canonical_period=canonical_period,
                x=player.pitch_x,
                y=player.pitch_y,
                is_estimated=player.is_estimated,
                coord_source="cv_homography",
            ))

    result: dict[str, Any] = {}
    for track_id, points in sorted(by_track.items()):
        metrics, _ = compute_player_movement(
            points, total_frames_evaluated=len(materialized), max_gap_s=max_gap_s,
        )
        result[str(track_id)] = metrics.as_dict()
    return {
        "status": "computed" if result else "unavailable",
        "frame_count": len(materialized),
        "track_count": len(result),
        "tracks": result,
        "limitations": [
            "CV positions are estimates until independently validated against held-out pitch annotations.",
            "Distance, speed, and acceleration are movement indicators, not physiological fatigue measurements.",
        ],
    }


def tactical_metric_readiness(frames: Iterable[VisionFrame]) -> dict[str, Any]:
    """Report Stage 5 input readiness without inventing tactical scores."""
    materialized = list(frames)
    players = [player for frame in materialized for player in frame.players]
    usable = [player for player in players if player.is_usable and player.pitch_x is not None and player.pitch_y is not None]
    teams = {player.team_side for player in usable}
    has_both_teams = TeamSide.HOME in teams and TeamSide.AWAY in teams
    has_usable_ball = any(
        frame.ball is not None and frame.ball.is_usable
        and frame.ball.pitch_x is not None and frame.ball.pitch_y is not None
        for frame in materialized
    )
    missing = [reason for reason, absent in (
        ("stable home/away team attribution", not has_both_teams),
        ("usable grounded ball positions", not has_usable_ball),
    ) if absent]
    try:
        das_dependency = f"accessible-space {version('accessible-space')}"
    except PackageNotFoundError:
        das_dependency = None
    return {
        "pc": {"status": "ready" if not missing else "unavailable", "reasons": missing},
        "epv": {"status": "unavailable", "reasons": [
            "declared possession labels", "labelled training target", "held-out evaluation",
        ]},
        "obso": {"status": "unavailable" if missing else "needs_validation", "reasons": missing or [
            "validated scoring/control formulation and player identities",
        ]},
        "das": {"status": "unavailable" if missing else "needs_validation", "dependency": das_dependency, "reasons": missing or [
            "pass assumptions and held-out validation",
        ]},
        "input_summary": {
            "frames": len(materialized),
            "usable_player_observations": len(usable),
            "team_sides_observed": sorted(side.value for side in teams),
            "usable_ball_observation": has_usable_ball,
        },
    }


def build_cv_analytics(frames: Iterable[VisionFrame], *, canonical_period: int = 1) -> dict[str, Any]:
    """Build the Stage 4 report and Stage 5 input/readiness report."""
    materialized = list(frames)
    return {
        "movement_analytics": summarize_vision_movement(materialized, canonical_period=canonical_period),
        "stage_5_tactical_metrics": tactical_metric_readiness(materialized),
    }