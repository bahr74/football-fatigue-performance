"""Detector thresholds, match splits, and evaluation exclusions (one place, recorded in every run manifest)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

DETECTOR_VERSION = "rules-0.2"

# Periods without usable tracking attribution (audit §7: suspected side/identity anomaly, or no extra-time tracking).
EXCLUDED_PERIODS: dict[int, tuple[int, ...]] = {10506: (3, 4), 10510: (3, 4), 10511: (3, 4), 10517: (3, 4)}

# Matches whose root/May event coordinates disagree with the supplied tracking (event comparison §3.4). Detection runs on
# them; results are reported separately because event labels and tracking may describe different position streams.
COORDINATE_FLAGGED_MATCHES = frozenset({10502, 10512, 3816, 3817, 3829, 3835, 3841, 3845, 3852, 3859})

# Thresholds are tuned on DEV only; reported numbers come from the remaining (held-out) matches.
DEV_MATCHES = (3812, 3813, 3814, 3815, 3818, 3819, 3820, 3821)

ALL_MATCHES = tuple(range(3812, 3860)) + tuple(range(10502, 10518))


# The ML detector looks at more candidate touches (the classifier removes the false ones); everything else is unchanged.
ML_CANDIDATE_OVERRIDES = dict(touch_dv_mps=2.0, touch_radius_m=4.0, touch_peak_window=5)


def test_matches() -> tuple[int, ...]:
    return tuple(m for m in ALL_MATCHES if m not in DEV_MATCHES)


@dataclass(frozen=True)
class DetectorConfig:
    # ball cleaning
    max_ball_speed_mps: float = 40.0          # a single fresh sample implying more than this on both sides is a spike
    velocity_half_window: int = 6             # frames on each side used for the before/after velocity fits
    min_fit_samples: int = 3                  # fresh ball samples needed in a half window
    # touches (moments where the ball velocity changes next to a player)
    touch_dv_mps: float = 3.0                 # |v_after - v_before| at a touch
    touch_peak_window: int = 8                # frames: a touch is the largest velocity change within +- this
    touch_radius_m: float = 3.0               # nearest player must come this close to the ball around the touch
    touch_search_frames: int = 6              # frames searched on each side for the nearest player
    # releases (a touch that sends the ball away)
    min_release_speed_mps: float = 4.0
    dead_ball_gap_s: float = 1.0              # ball missing longer than this ends the play (no pass across it)
    max_pass_flight_s: float = 6.0            # next touch later than this: no receiver
    carry_radius_m: float = 4.0               # same player touches again and the ball never left this radius: a carry
    direction_horizon_s: float = 0.4          # ball displacement over this time (or until the next touch) gives the release direction
    # shots
    shot_min_speed_mps: float = 6.0
    shot_max_distance_m: float = 25.0
    shot_post_margin_m: float = 4.5           # projected crossing of the goal line within the posts +- this
    goal_half_width_m: float = 3.66
    goal_height_m: float = 2.44
    goal_line_window_s: float = 5.0           # time after the shot searched for the ball at the goal mouth
    shot_reach_goal_m: float = 2.0            # ball observed this close to the goal line, inside the posts + margin: on/near target
    goal_min_stoppage_s: float = 60.0         # no touch for this long after a shot: play restarted from a kickoff (dev data)
    goal_stoppage_start_s: float = 15.0       # the pause must begin within this time after the shot
    goal_max_stoppage_s: float = 300.0        # longer pauses are other stoppages (injury, half-time, long VAR checks)
    goal_mouth_max_m: float = 1.5             # the ball must come this close to the goal mouth (0 = inside the frame) after the shot
    # crosses
    cross_min_abs_y_m: float = 12.0           # wide: well outside the goal area; tuned on dev
    cross_max_goal_distance_m: float = 35.0   # x distance from the attacked goal line
    cross_min_speed_mps: float = 8.0
    # key passes
    key_pass_window_s: float = 10.0
    # pitch
    pitch_length_m: float = 105.0
    pitch_width_m: float = 68.0

    def to_dict(self) -> dict:
        return asdict(self)
