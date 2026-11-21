"""M7.3 opt-in trajectory stabilisers (football_fatigue.analytics.motion) and their use through the coach engine. Synthetic only."""
import numpy as np
import pytest

from football_fatigue.analytics import coach as K
from football_fatigue.analytics import motion as MO

FPS = 25.0


def track(speed_mps, seconds, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * FPS)) / FPS
    x = 10 + speed_mps * t
    y = np.full_like(t, 30.0)
    return t, x + rng.normal(0, noise, len(t)), y + rng.normal(0, noise, len(t)), x


def runs(t, x, y, method="kalman_rts", **kw):
    cfg = MO.MotionConfig(method=method, **kw)
    return MO.stabilized_runs(t, x, y, cfg, max_gap_s=0.12, max_plausible_kmh=45.0)


@pytest.mark.parametrize("method", MO.STABILIZERS)
def test_stabilisers_recover_speed_far_better_than_raw_differences(method):
    t, x, y, _ = track(5.0, 10)
    rs = runs(t, x, y, method, window_s=1.0)
    assert len(rs) == 1 and len(rs[0]["t"]) == len(t)                  # one run, output at the observed samples only
    raw = np.hypot(np.diff(x), np.diff(y)) * FPS
    err = np.abs(rs[0]["speed"][12:-12] - 5.0)
    assert np.median(err) < 0.5 and np.median(np.abs(raw - 5.0)) > 3 * np.median(err)


def test_noise_does_not_cut_the_run_but_long_gaps_and_unusable_samples_do():
    t, x, y, _ = track(3.0, 6, noise=0.45)       # raw steps often exceed 45 km/h: the M7.2 rule would cut here
    assert len(K.runs(t, x, y, K.CoachSettings())) > 5
    assert len(runs(t, x, y)) == 1
    keep = (t < 2.0) | (t > 2.5)                  # 0.5 s unobserved: never bridged
    assert len(runs(t[keep], x[keep], y[keep])) == 2
    usable = np.ones(len(t), bool); usable[75] = False
    st = {}
    rs = MO.stabilized_runs(t, x, y, MO.MotionConfig(), max_gap_s=0.12, max_plausible_kmh=45, usable=usable, stats=st)
    assert len(rs) == 2 and st["runs"] == 2 and all(75 / FPS not in r["t"] for r in rs)


def test_persistent_jump_splits_but_single_spike_does_not():
    t, x, y, _ = track(2.0, 6, noise=0.05)
    xj = x.copy(); xj[80:] += 12.0                # box jumps to someone 12 m away and stays
    assert len(MO.split_runs(t, xj, y, np.ones(len(t), bool), 0.12, 12.5)) == 2
    xs = x.copy(); xs[80] += 3.0                  # one bad sample
    rs = runs(t, xs, y)
    assert len(rs) == 1 and rs[0]["speed"].max() < 25 / 3.6


def test_short_and_implausible_runs_are_dropped_and_counted():
    t, x, y, _ = track(3.0, 0.5)
    st = {}
    assert MO.stabilized_runs(t, x, y, MO.MotionConfig(), max_gap_s=0.12, max_plausible_kmh=45, stats=st) == []
    assert st["short_runs"] == 1 and st["samples_used"] == 0
    t, x, y, _ = track(15.0, 4, noise=0.01)       # 54 km/h sustained: not a plausible human speed
    st = {}
    assert MO.stabilized_runs(t, x, y, MO.MotionConfig(), max_gap_s=0.12, max_plausible_kmh=45, stats=st) == []
    assert st["implausible_runs"] + st["short_runs"] >= 1 and st["samples_used"] == 0   # cut as jumps or dropped as implausible


def test_confidence_weighting_trusts_precise_samples_more():
    t, x, y, truth = track(4.0, 8, noise=0.0)
    rng = np.random.default_rng(3)
    bad = np.arange(len(t)) % 2 == 0
    xn = x + np.where(bad, rng.normal(0, 1.0, len(t)), rng.normal(0, 0.05, len(t)))
    var = np.where(bad, 1.0, 0.05 ** 2)
    sx_w, _, _, _ = MO.stabilize_run(t, xn, y, MO.MotionConfig(method="kalman_rts"), var, var)
    sx_c, _, _, _ = MO.stabilize_run(t, xn, y, MO.MotionConfig(method="kalman_rts", meas_sigma_m=0.5))
    assert np.sqrt(np.mean((sx_w - truth) ** 2)) < 0.6 * np.sqrt(np.mean((sx_c - truth) ** 2))


def test_coach_engine_accepts_stabilisers_and_keeps_old_options_unchanged():
    s, err = K.settings_from_dict({"smoothing": "kalman_rts", "kalman_accel_noise": 9.0})
    assert not err and s.smoothing == "kalman_rts"
    assert K.settings_from_dict({"kalman_accel_noise": 0.0})[1]                 # out of range: an error, never clipped
    t, x, y, _ = track(6.0, 8, noise=0.0)
    a = K.analyse_segment(t, x, y, K.CoachSettings())
    b = K.analyse_segment(t, x, y, K.CoachSettings(smoothing="moving_average"))
    assert a == b                                                                # default path untouched


def test_sprint_found_with_stabiliser_and_no_false_sprint_when_standing():
    rng = np.random.default_rng(5)
    t = np.arange(int(12 * FPS)) / FPS
    v = np.where((t > 4) & (t < 7), 8.0, 1.0)
    x = 10 + np.cumsum(v) / FPS
    xn, yn = x + rng.normal(0, 0.3, len(t)), 30 + rng.normal(0, 0.3, len(t))
    s = K.CoachSettings(smoothing="kalman_rts")
    r = K.analyse_segment(t, xn, yn, s)
    assert r["sprint_count"] == 1 and 3.5 <= r["sprints"][0]["start_s"] + 0.5 and r["sprints"][0]["end_s"] <= 7.6
    still = K.analyse_segment(t, 50 + rng.normal(0, 0.3, len(t)), 30 + rng.normal(0, 0.3, len(t)), s)
    assert still["sprint_count"] == 0 and still["speed"]["max_kmh"] < 15
    assert K.analyse_segment(t, 50 + rng.normal(0, 0.3, len(t)), 30 + rng.normal(0, 0.3, len(t)), K.CoachSettings())["speed"]["max_kmh"] > still["speed"]["max_kmh"]


def test_config_validation():
    assert MO.config_from_dict({"method": "savgol", "window_s": 0.8}).window_s == 0.8
    with pytest.raises(ValueError):
        MO.config_from_dict({"method": "spline"})
    with pytest.raises(ValueError):
        MO.config_from_dict({"window_s": -1})
