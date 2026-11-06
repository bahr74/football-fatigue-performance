"""Synthetic-trajectory tests for the observed analytics engine (known expected results) and for the M7 pitch mapping helpers."""

import numpy as np
import pytest

from football_fatigue.analytics.observed import (
    PhysicalConfig,
    kinematics,
    physical_metrics_gate,
    physical_summary,
    team_shape,
    window_summaries,
    zone_occupancy,
)
from football_fatigue.demo.overlay import pitch_validity, role_of

FPS = 25.0


def line(speed, seconds, t0=0.0, x0=0.0):
    t = t0 + np.arange(int(seconds * FPS) + 1) / FPS
    return t, x0 + speed * (t - t0), np.zeros_like(t)


def test_constant_speed_distance_and_speed():
    t, x, y = line(4.0, 10.0)
    s = physical_summary(t, x, y)
    assert s["distance_m"] == pytest.approx(40.0, rel=1e-3)
    assert s["max_speed_mps"] == pytest.approx(4.0, rel=1e-3) and s["hsr_distance_m"] == 0 and s["sprint_count"] == 0


def test_gaps_are_never_bridged():
    t1, x1, y1 = line(3.0, 4.0)
    t2, x2, y2 = line(3.0, 4.0, t0=10.0, x0=50.0)               # 6 s unobserved, 38 m apart
    s = physical_summary(np.r_[t1, t2], np.r_[x1, x2], np.r_[y1, y2])
    assert s["runs"] == 2 and s["distance_m"] == pytest.approx(24.0, rel=1e-3)   # 12 + 12, not 12 + 38 + 12
    assert s["observed_time_s"] == pytest.approx(8.0)


def test_implausible_jump_breaks_the_run():
    t, x, y = line(2.0, 4.0)
    x = x.copy(); x[50:] += 30.0                                 # 30 m teleport in one frame (swap or calibration artefact)
    s = physical_summary(t, x, y)
    assert s["runs"] == 2 and s["distance_m"] < 10.0


def test_sprint_with_hysteresis_min_duration_and_hsr():
    # 3 s jog at 4, 2 s sprint at 8, 3 s jog at 4 (smoothing blurs edges by <= 0.25 s)
    v = np.r_[np.full(75, 4.0), np.full(50, 8.0), np.full(76, 4.0)]
    t = np.arange(len(v)) / FPS
    x = np.r_[0, np.cumsum(v[1:] / FPS)]
    s = physical_summary(t, x, np.zeros_like(x))
    assert s["sprint_count"] == 1
    sp = s["sprints"][0]
    assert sp["duration_s"] == pytest.approx(2.0, abs=0.3) and sp["peak_speed_mps"] == pytest.approx(8.0, abs=0.05)
    assert s["hsr_distance_m"] == pytest.approx(16.0, abs=2.5)


def test_short_burst_is_not_a_sprint_and_hysteresis_holds_through_a_dip():
    v = np.r_[np.full(50, 4.0), np.full(15, 8.0), np.full(50, 4.0)]          # 0.6 s at 8 m/s: too short
    t = np.arange(len(v)) / FPS
    assert physical_summary(t, np.r_[0, np.cumsum(v[1:] / FPS)], np.zeros(len(v)))["sprint_count"] == 0
    v = np.r_[np.full(40, 4.0), np.full(30, 8.0), np.full(12, 6.8), np.full(30, 8.0), np.full(40, 4.0)]   # dip stays above exit 6.5
    t = np.arange(len(v)) / FPS
    assert physical_summary(t, np.r_[0, np.cumsum(v[1:] / FPS)], np.zeros(len(v)))["sprint_count"] == 1


def test_repeated_high_intensity_efforts_and_recovery():
    seg = lambda sp, n: np.full(n, sp)
    v = np.concatenate([seg(3.0, 50)] + [np.r_[seg(8.0, 40), seg(3.0, 250)] for _ in range(3)])   # 3 sprints, ~10 s recoveries
    t = np.arange(len(v)) / FPS
    s = physical_summary(t, np.r_[0, np.cumsum(v[1:] / FPS)], np.zeros(len(v)))
    assert s["sprint_count"] == 3 and len(s["recovery_intervals_s"]) == 2
    assert all(9.0 < r < 12.0 for r in s["recovery_intervals_s"]) and s["rhie_bouts"] == 1
    s2 = physical_summary(t, np.r_[0, np.cumsum(v[1:] / FPS)], np.zeros(len(v)), PhysicalConfig(rhie_max_recovery_s=5.0))
    assert s2["rhie_bouts"] == 0


def test_acceleration_sign_and_timestamp_awareness():
    t = np.linspace(0, 4, 101)                                     # 25 Hz
    x = 0.5 * 2.0 * t ** 2                                         # a = 2 m/s^2
    k = kinematics(t, x, np.zeros_like(t), PhysicalConfig(smooth_window_s=0))
    assert np.median(k[0]["accel"][5:-5]) == pytest.approx(2.0, rel=0.02)
    t10 = np.linspace(0, 4, 41)                                    # 10 Hz, same motion and span
    s25, s10 = physical_summary(t, x, 0 * t), physical_summary(t10, 0.5 * 2 * t10 ** 2, 0 * t10)
    assert s25["distance_m"] == pytest.approx(s10["distance_m"], rel=0.02)


def test_window_summaries_cut_at_edges():
    t, x, y = line(5.0, 20.0)
    w = window_summaries(t, x, y, window_s=10.0)
    assert len(w) == 3 and sum(r["distance_m"] for r in w) == pytest.approx(100.0, rel=0.01)


def test_gate_reports_every_failed_condition():
    g = physical_metrics_gate(coordinate_unit="pixel", calibration_gate_met=False, timestamps_valid=True, observed_coverage=0.5, identity_attributable=False)
    assert g["available"] is False and len(g["reasons"]) == 4
    ok = physical_metrics_gate(coordinate_unit="metre", calibration_gate_met=True, timestamps_valid=True, observed_coverage=0.9, identity_attributable=True)
    assert ok == {"available": True, "reasons": []}


def test_zone_occupancy_and_team_shape():
    z = zone_occupancy([1, 104, 52.5, -3], [1, 67, 34, 10], nx=6, ny=4)
    assert z["n"] == 3 and z["excluded_outside"] == 1 and z["grid"][0][0] == 1 and z["grid"][3][5] == 1 and z["grid"][2][3] == 1
    s = team_shape([10, 30, 20], [10, 10, 40])
    assert s["n_visible"] == 3 and s["centroid_m"] == [20.0, 20.0] and s["length_m"] == 20.0 and s["width_m"] == 30.0
    assert team_shape([1], [1]) is None


def test_pitch_validity_and_conservative_roles():
    assert pitch_validity(50, 30) == 0 and pitch_validity(-0.8, 30) == 0 and pitch_validity(-3, 30) == 1 and pitch_validity(50, 80) == 2
    assert role_of("player", "team_0", 0) == 0 and role_of("goalkeeper", "team_1", 0) == 1 and role_of("referee", "team_0", 0) == 2
    assert role_of("player", "other", 0) == 3                       # third colour cluster stays unknown
    assert role_of("player", "team_0", 1) == 3 and role_of("goalkeeper", None, 2) == 3   # off-pitch people are never active players
    assert role_of("player", None, None) == 0                        # without calibration the detector role is kept


def test_hybrid_mapping_recovers_held_out_synthetic_camera():
    """Pitch mapping on a synthetic fisheye camera: the TPS-corrected hybrid must reproduce held-out points within 0.3 m."""
    from pathlib import Path

    from football_fatigue.vision import pitch_mapping as M
    from football_fatigue.vision import pitch_prototype as P

    intr = Path("data/external/soccertrack-v2/raw/117093/117093_camera_intrinsics.npz")
    if not intr.is_file():
        pytest.skip("intrinsics not present")
    par = np.array([1636.0, 1775.4, 2010.3, 495.2, -0.2355, 0.3664, -0.2505, 0.0620, -1.2996, 0.0077, -0.0099, -51.32, -15.41, 83.27])
    gx, gy = np.meshgrid(np.linspace(0, 105, 12), np.linspace(0, 68, 7))
    pitch = np.c_[gx.ravel(), gy.ravel()]
    image = P.project(par, pitch)
    e = M.held_out_errors("hybrid", pitch, image, intr, "kfold", folds=6)
    assert np.median(e) < 0.3
    assert M.gates(M.summarise(e))["team_shape"]
