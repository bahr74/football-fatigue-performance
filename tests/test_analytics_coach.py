"""Coach-configurable physical analytics (M7.2): synthetic trajectories only."""

import numpy as np
import pytest

from football_fatigue.analytics import coach as K

FPS = 25.0


def traj(speeds_mps, fps=FPS):
    """Straight-line trajectory along x from a per-sample speed profile."""
    v = np.asarray(speeds_mps, float)
    t = np.arange(len(v)) / fps
    x = np.r_[0.0, np.cumsum((v[1:] + v[:-1]) / 2 / fps)]
    return t, x, np.zeros_like(x)


def profile(*parts):
    """Concatenate (speed_mps, seconds) pieces; linear ramps of 0.4 s between pieces."""
    out = []
    for i, (v, s) in enumerate(parts):
        if out:
            out += list(np.linspace(out[-1], v, int(0.4 * FPS)))
        out += [v] * int(s * FPS)
    return out


def test_defaults_and_presets_validate():
    s, err = K.settings_from_dict({})
    assert err == [] and s == K.CoachSettings()
    for p in K.PRESETS.values():
        s, err = K.settings_from_dict(p["settings"])
        assert err == [], err
        assert "not a validated standard" in p["description"] or "coach" in p["description"].lower()


@pytest.mark.parametrize("bad, msg", [({"nope": 1}, "unknown setting"), ({"hsr_kmh": 99}, "outside"), ({"hsr_kmh": "fast"}, "finite number"),
                                      ({"sprint_exit_kmh": 30.0}, "hysteresis"), ({"smoothing": "kalman"}, "smoothing"),
                                      ({"repeated_min_efforts": 2.5}, "whole number"), ({"hsr_kmh": True}, "finite number")])
def test_invalid_settings_are_errors_never_clipped(bad, msg):
    s, err = K.settings_from_dict(bad)
    assert s is None and any(msg in e for e in err)


def test_constant_speed_distance_zone_and_no_sprint():
    t, x, y = traj([5.0] * int(10 * FPS))
    r = K.analyse_segment(t, x, y, K.CoachSettings())
    assert r["distance_m"] == pytest.approx(5.0 * t[-1], rel=1e-3)
    assert r["distance_by_zone_m"]["running"] == pytest.approx(r["distance_m"], rel=1e-6)   # 18 km/h is below HSR 19.8
    assert r["sprint_count"] == 0 and r["hsr_distance_m"] == 0
    assert r["speed"]["max_kmh"] == pytest.approx(18.0, abs=0.05)
    assert r["speed"]["average_kmh"] == pytest.approx(18.0, abs=0.05)


def test_sprint_detection_duration_distance_and_min_distance_rule():
    t, x, y = traj(profile((3.0, 4), (8.0, 3), (3.0, 4)))
    r = K.analyse_segment(t, x, y, K.CoachSettings())
    assert r["sprint_count"] == 1
    sp = r["sprints"][0]
    assert 2.9 <= sp["duration_s"] <= 3.6 and sp["peak_kmh"] == pytest.approx(28.8, abs=0.1)
    assert 20 <= sp["distance_m"] <= 30
    s2, _ = K.settings_from_dict({"sprint_min_distance_m": 40.0})
    assert K.analyse_segment(t, x, y, s2)["sprint_count"] == 0
    s3, _ = K.settings_from_dict({"sprint_kmh": 30.0, "sprint_exit_kmh": 29.0, "max_plausible_kmh": 45.0})
    assert K.analyse_segment(t, x, y, s3)["sprint_count"] == 0


def test_thresholds_change_results_deterministically():
    t, x, y = traj(profile((4.0, 3), (5.8, 4), (4.0, 3)))     # 20.9 km/h plateau
    a = K.analyse_segment(t, x, y, K.CoachSettings())
    assert a == K.analyse_segment(t, x, y, K.CoachSettings())
    lo, _ = K.settings_from_dict(K.PRESETS["example_lower_intensity"]["settings"])
    hi, _ = K.settings_from_dict({"hsr_kmh": 21.5, "run_kmh": 14.4})
    assert a["hsr_distance_m"] > 0
    assert K.analyse_segment(t, x, y, lo)["hsr_distance_m"] >= a["hsr_distance_m"]
    assert K.analyse_segment(t, x, y, hi)["hsr_distance_m"] == 0


def test_gap_and_unusable_samples_are_never_bridged():
    t, x, y = traj([4.0] * 250)
    keep = np.r_[0:100, 150:250]                   # a 2 s hole: the person moved 8 m unobserved
    r = K.analyse_segment(t[keep], x[keep], y[keep], K.CoachSettings())
    assert r["runs"] == 2
    assert r["distance_m"] == pytest.approx(4.0 * (t[99] - t[0] + t[249] - t[150]), rel=1e-3)
    usable = np.ones(250, bool); usable[100:150] = False      # e.g. interpolated or off-pitch samples
    r2 = K.analyse_segment(t, x, y, K.CoachSettings(), usable=usable)
    assert r2["runs"] == 2 and r2["distance_m"] == pytest.approx(r["distance_m"], rel=1e-6)


def test_implausible_step_breaks_the_run():
    t, x, y = traj([3.0] * 100)
    x = x.copy(); x[50:] += 5.0                    # 5 m jump in one frame = 450 km/h
    r = K.analyse_segment(t, x, y, K.CoachSettings())
    assert r["runs"] == 2 and r["distance_m"] < 3.0 * t[-1] + 0.5


def test_acceleration_and_deceleration_events():
    v = profile((0.0, 2)) + list(np.linspace(0, 8, int(2 * FPS))) + [8.0] * 50 + list(np.linspace(8, 0, int(1.6 * FPS))) + [0.0] * 50
    t, x, y = traj(v)
    r = K.analyse_segment(t, x, y, K.CoachSettings())        # 4 m/s^2 up, 5 m/s^2 down
    assert len(r["accelerations"]) == 1 and len(r["decelerations"]) == 1
    assert r["decelerations"][0]["peak_mps2"] < -2
    s, _ = K.settings_from_dict({"accel_mps2": 6.0, "decel_mps2": 6.0})
    r2 = K.analyse_segment(t, x, y, s)
    assert r2["accelerations"] == [] and r2["decelerations"] == []


def test_repeated_sprint_bouts_follow_the_window():
    t, x, y = traj(profile((3.0, 3), (8.0, 2), (3.0, 10), (8.0, 2), (3.0, 10), (8.0, 2), (3.0, 3)))
    r = K.analyse_segment(t, x, y, K.CoachSettings())
    assert r["sprint_count"] == 3 and len(r["recoveries_s"]) == 2
    assert len(r["repeated_sprint_bouts"]) == 1 and r["repeated_sprint_bouts"][0]["efforts"] == 3
    s, _ = K.settings_from_dict({"repeated_window_s": 5.0})
    assert K.analyse_segment(t, x, y, s)["repeated_sprint_bouts"] == []
    assert r["work_rest"]["work_s"] > 0 and r["work_rest"]["ratio"] > 0


@pytest.mark.parametrize("method", ["none", "moving_average", "median"])
def test_smoothing_methods_keep_linear_motion(method):
    t, x, y = traj([6.0] * 200)
    s, _ = K.settings_from_dict({"smoothing": method})
    assert K.analyse_segment(t, x, y, s)["distance_m"] == pytest.approx(6.0 * t[-1], rel=1e-3)


def test_eligibility_levels_and_blockers():
    s = K.CoachSettings()
    ok = dict(coordinate_unit="metre", timestamps_valid=True, identity_safe=True, role_ok=True, n_expected=250, n_usable=240, duration_s=10.0, s=s)
    e = K.segment_eligibility(**ok)
    assert e["level"] == "L1" and e["blockers"] == [] and any("L3" in b for b in e["next_level_blockers"])
    assert K.segment_eligibility(**dict(ok, reviewed_same_person=True))["level"] == "L2"
    for change, word in (({"coordinate_unit": "pixel"}, "metres"), ({"identity_safe": False}, "tracklet"), ({"role_ok": False}, "officials"),
                         ({"n_usable": 100}, "coverage"), ({"duration_s": 2.0}, "shorter")):
        e = K.segment_eligibility(**dict(ok, **change))
        assert e["level"] == "L0" and any(word in b for b in e["blockers"])


def test_aggregate_sums_only_and_never_extrapolates():
    t, x, y = traj([5.0] * 250)
    r = K.analyse_segment(t, x, y, K.CoachSettings())
    a = K.aggregate([r, r])
    assert a["distance_m"] == pytest.approx(2 * r["distance_m"]) and a["observed_time_s"] == pytest.approx(2 * r["observed_time_s"])
    assert "extrapolated" in a["note"]
