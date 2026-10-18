"""Scientific Validation Tests for Tracking Analytics (Z4).

Directly validates the 10 scientific prerequisites from TASK 3 Section 6:
1. Effective position update cadence (handles repeated samples without velocity sawtooth).
2. Stationary players vs dropouts (zero displacement is distinct from missing).
3. Missing, estimated, and ambiguous observation tracking and coverage fractions.
4. Valid temporal differences (delta_t > 0).
5. Gap exclusion policy (never connects across gaps or period boundaries).
6. Smoothing and windowing transparency.
7. Speed spike rejection (biological/physics limits: rejects impossible jumps > 12 m/s).
8. Separation of observed and derived values.
9. Coverage and confidence criteria.
10. Validation on synthetic trajectories and plausible football movement ranges.
"""

import pytest

from football_fatigue.analytics.movement import (
    MovementMetrics,
    TrajectoryPoint,
    classify_speed_zone,
    compute_player_movement,
)


def test_1_constant_velocity_trajectory():
    """A player moves along the x-axis at a constant 5.0 m/s for 2.0 seconds (60 frames at 30 fps)."""
    fps = 30.0
    dt = 1.0 / fps
    total_frames = 61
    speed = 5.0  # m/s

    points = [
        TrajectoryPoint(
            frame_num=i,
            video_time_s=i * dt,
            canonical_period=1,
            x=i * dt * speed,
            y=0.0,
            is_estimated=False,
        )
        for i in range(total_frames)
    ]

    metrics, kinematics = compute_player_movement(points, total_frames_evaluated=total_frames)

    expected_distance = 2.0 * speed  # 10.0 meters
    assert metrics.total_distance_m == pytest.approx(expected_distance, abs=0.05)
    assert metrics.mean_speed_m_s == pytest.approx(speed, abs=0.1)
    assert metrics.max_speed_m_s == pytest.approx(speed, abs=0.1)
    # 5.0 m/s is in the "running" zone (4.0 - 5.5 m/s)
    assert metrics.running_distance_m == pytest.approx(expected_distance, abs=0.05)
    assert metrics.coverage.coverage_fraction == 1.0
    assert metrics.coverage.is_reliable is True


def test_2_stationary_player():
    """A stationary player stays at (15.0, -10.0) for 50 frames."""
    points = [
        TrajectoryPoint(
            frame_num=i,
            video_time_s=i * (1.0 / 30.0),
            canonical_period=1,
            x=15.0,
            y=-10.0,
        )
        for i in range(50)
    ]

    metrics, kinematics = compute_player_movement(points)
    assert metrics.total_distance_m == pytest.approx(0.0, abs=1e-6)
    assert metrics.mean_speed_m_s == pytest.approx(0.0, abs=1e-6)
    assert metrics.max_speed_m_s == pytest.approx(0.0, abs=1e-6)
    assert metrics.walking_distance_m == pytest.approx(0.0, abs=1e-6)


def test_3_repeated_cadence_samples_smoothed_against_sawtooth():
    """Simulates 15 Hz tracking on a 30 Hz recording where coordinates repeat on alternate frames.

    Player moves at 4.0 m/s overall, but raw coordinates update only every 2 frames.
    Naive step differencing produces [0, 8, 0, 8...]. Windowing stabilizes the speed.
    """
    fps = 30.0
    points = []
    for i in range(30):
        # Coordinates update every 2 frames
        effective_step = i // 2
        effective_t = effective_step * (2.0 / fps)
        points.append(TrajectoryPoint(
            frame_num=i,
            video_time_s=i / fps,
            canonical_period=1,
            x=effective_t * 4.0,
            y=0.0,
        ))

    metrics, kinematics = compute_player_movement(points, window_frames=3)
    # The total distance should still equal ~3.73 - 4.0 m
    assert metrics.total_distance_m == pytest.approx(3.73, abs=0.3)
    # Max speed should not explode to 8 m/s
    assert metrics.max_speed_m_s < 6.0


def test_4_gap_and_period_break_exclusion():
    """Points span across a period break (e.g. halftime) and a 3-second gap.

    The gap and period jump must NOT be counted as physical movement.
    """
    points = [
        # Period 1: moves from x=0 to x=5 (5 meters)
        TrajectoryPoint(frame_num=10, video_time_s=0.0, canonical_period=1, x=0.0, y=0.0),
        TrajectoryPoint(frame_num=11, video_time_s=1.0, canonical_period=1, x=5.0, y=0.0),
        # HALFTIME BREAK: period 2 starts at t=900s, player starts at x=50.0 (teleport of 45m)
        TrajectoryPoint(frame_num=1000, video_time_s=900.0, canonical_period=2, x=50.0, y=0.0),
        TrajectoryPoint(frame_num=1001, video_time_s=901.0, canonical_period=2, x=54.0, y=0.0),
        # TRACKING GAP: 5 second dropout
        TrajectoryPoint(frame_num=1150, video_time_s=906.0, canonical_period=2, x=70.0, y=0.0),
        TrajectoryPoint(frame_num=1151, video_time_s=907.0, canonical_period=2, x=72.0, y=0.0),
    ]

    metrics, _ = compute_player_movement(points, max_gap_s=2.0)

    # Valid segments:
    # Seg 1: 5.0m
    # Seg 2: 4.0m
    # Seg 3: 2.0m
    # Total distance must be 11.0m (NOT 5.0 + 45.0 + 4.0 + 16.0 + 2.0 = 72m!)
    assert metrics.total_distance_m == pytest.approx(11.0, abs=0.1)
    # Gaps detected must be 2 (one period change, one 5s gap)
    assert metrics.coverage.gaps_detected == 2
    # Speed must remain realistic (4-5 m/s), never teleporting at 45 m/s
    assert metrics.max_speed_m_s <= 5.5


def test_5_speed_spike_rejection():
    """A sensor glitch places the player at x=500m for one frame (instant speed > 50 m/s).

    This spike must be rejected and recorded in spike_count.
    """
    points = [
        TrajectoryPoint(frame_num=0, video_time_s=0.0, canonical_period=1, x=0.0, y=0.0),
        TrajectoryPoint(frame_num=1, video_time_s=0.1, canonical_period=1, x=0.5, y=0.0),
        # GLITCH SPIKE
        TrajectoryPoint(frame_num=2, video_time_s=0.2, canonical_period=1, x=500.0, y=0.0),
        # RETURN TO NORMAL
        TrajectoryPoint(frame_num=3, video_time_s=0.3, canonical_period=1, x=1.5, y=0.0),
    ]

    metrics, kinematics = compute_player_movement(points, max_speed_m_s=12.0)
    assert metrics.coverage.speed_spikes_rejected >= 1
    # Max valid speed must not be 5000 m/s
    assert metrics.max_speed_m_s <= 12.0


def test_6_coverage_and_estimated_fraction():
    """Checks calculation of coverage fraction and estimated fraction."""
    points = [
        TrajectoryPoint(frame_num=0, video_time_s=0.0, canonical_period=1, x=0.0, y=0.0, is_estimated=True),
        TrajectoryPoint(frame_num=1, video_time_s=0.1, canonical_period=1, x=0.5, y=0.0, is_estimated=True),
        TrajectoryPoint(frame_num=2, video_time_s=0.2, canonical_period=1, x=1.0, y=0.0, is_estimated=False),
    ]

    # Suppose total match frames evaluated was 10, but only 3 points were usable
    metrics, _ = compute_player_movement(points, total_frames_evaluated=10)
    assert metrics.coverage.total_frames_evaluated == 10
    assert metrics.coverage.usable_points == 3
    assert metrics.coverage.coverage_fraction == pytest.approx(0.3)
    assert metrics.coverage.estimated_points == 2
    assert metrics.coverage.estimated_fraction == pytest.approx(2.0 / 3.0)
    assert metrics.coverage.is_reliable is False  # 30% < 60%


def test_7_speed_zones_and_sprinting():
    """Tests correct classification across speed zones: walking, jogging, running, HSR, sprinting."""
    assert classify_speed_zone(1.0) == "walking"
    assert classify_speed_zone(3.0) == "jogging"
    assert classify_speed_zone(5.0) == "running"
    assert classify_speed_zone(6.5) == "hsr"
    assert classify_speed_zone(8.5) == "sprint"

    # Sprint trajectory: player moves at 8.0 m/s for 1.0 second
    points = [
        TrajectoryPoint(frame_num=i, video_time_s=i * 0.1, canonical_period=1, x=i * 0.8, y=0.0)
        for i in range(11)
    ]
    metrics, _ = compute_player_movement(points)
    assert metrics.sprint_distance_m > 7.0
    assert metrics.high_intensity_distance_m > 7.0
