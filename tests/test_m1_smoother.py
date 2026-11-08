"""Unit tests for the M1 ball smoother. All tracks are SYNTHETIC (generated here), not source data."""
import numpy as np
import pytest

from m1_possession import FLAG_ESTIMATED, FLAG_INTERPOLATED, FLAG_MISSING, FLAG_OBSERVED, SmootherConfig, smooth_ball

FPS = 30.0
METHODS = ("linear", "savgol", "rts", "rts_adaptive")


def held(v: np.ndarray) -> np.ndarray:
    """Synthetic source behaviour: every value is held for two frames (about 15 Hz updates)."""
    out = v.copy()
    out[1::2] = v[0::2][: len(out[1::2])]
    return out


def straight(n=90, vx=10.0, vy=-4.0):
    t = np.arange(n) / FPS
    return t, held(vx * t), held(vy * t), np.zeros(n)


@pytest.mark.parametrize("method", METHODS)
def test_constant_velocity_recovered(method):
    t, x, y, z = straight()
    r = smooth_ball(t, x, y, z, cfg=SmootherConfig(method=method))
    mid = slice(15, 75)
    assert np.allclose(r.vx[mid], 10.0, atol=0.6)
    assert np.allclose(r.vy[mid], -4.0, atol=0.6)
    assert np.allclose(r.x[mid], 10.0 * t[mid], atol=0.25)


def test_repeats_dropped_and_resting_ball_kept():
    n = 20
    t = np.arange(n) / FPS
    x = np.full(n, 3.0)                     # ball at rest: identical values every frame
    r = smooth_ball(t, x, np.zeros(n), np.zeros(n), cfg=SmootherConfig(method="linear"))
    assert r.counts["repeats_dropped"] == 10
    assert (r.flag[0::2] == FLAG_OBSERVED).all() and (r.flag[1::2] == FLAG_INTERPOLATED).all()
    assert np.allclose(r.x, 3.0) and np.allclose(r.speed, 0.0)


def test_one_sample_spike_removed():
    t, x, y, z = straight()
    x = x.copy()
    x[40] += 15.0                           # one impossible sample (held frame 41 is a repeat of it)
    x[41] = x[40]
    r = smooth_ball(t, x, y, z, cfg=SmootherConfig(method="linear"))
    assert r.counts["spikes_dropped"] == 1
    assert abs(r.x[40] - 10.0 * t[40]) < 0.4
    assert r.flag[40] == FLAG_INTERPOLATED


def test_short_gap_filled_long_gap_missing():
    t, x, y, z = straight(n=120)
    x, y = x.copy(), y.copy()
    x[30:36] = y[30:36] = np.nan            # 0.2 s: filled
    x[60:90] = y[60:90] = np.nan            # 1.0 s: left missing
    r = smooth_ball(t, x, y, z, cfg=SmootherConfig(method="linear", max_gap_s=0.5))
    assert (r.flag[30:36] == FLAG_INTERPOLATED).all()
    assert (r.flag[61:90] == FLAG_MISSING).all()
    assert np.isnan(r.x[61:90]).all()
    assert r.counts["gap_breaks"] == 1


def test_estimated_flag_from_visibility():
    t, x, y, z = straight(n=40)
    vis = np.array(["VISIBLE"] * 40, dtype=object)
    vis[10:20] = "ESTIMATED"
    r = smooth_ball(t, x, y, z, visibility=vis, cfg=SmootherConfig(method="rts_adaptive"))
    assert (r.flag[10:20:2] == FLAG_ESTIMATED).all()          # kept updates with ESTIMATED visibility
    assert (r.flag[11:20:2] == FLAG_INTERPOLATED).all()       # held repeats stay interpolated
    assert (r.flag[0:10:2] == FLAG_OBSERVED).all() and (r.flag[20::2] == FLAG_OBSERVED).all()
    assert r.counts["frames_estimated"] == 5
    assert np.isfinite(r.x[10:20]).all()                      # estimated samples are still used
    r2 = smooth_ball(t, x, y, z, cfg=SmootherConfig(method="rts_adaptive"))
    assert FLAG_ESTIMATED not in r2.flag                      # no visibility given: no estimated flag


def test_drop_estimated_option():
    t, x, y, z = straight(n=40)
    vis = np.array(["VISIBLE"] * 40, dtype=object)
    vis[10:20] = "ESTIMATED"
    r = smooth_ball(t, x, y, z, visibility=vis, cfg=SmootherConfig(method="linear", drop_estimated=True))
    assert r.counts["estimated_dropped"] == 10
    assert FLAG_ESTIMATED not in r.flag and (r.flag[10:20] == FLAG_INTERPOLATED).all()


def test_period_change_never_joined():
    t, x, y, z = straight(n=60)
    per = np.r_[np.ones(30, int), 2 * np.ones(30, int)]
    r = smooth_ball(t, x, y, z, period=per, cfg=SmootherConfig(method="rts"))
    assert r.counts["segments"] == 2


@pytest.mark.parametrize("method", ("rts_adaptive", "savgol"))
def test_kick_keeps_sharp_speed_change(method):
    n = 120
    t = np.arange(n) / FPS
    k = 60                                  # kick: ball at rest, then 20 m/s
    x = np.where(np.arange(n) < k, 0.0, 20.0 * (t - t[k]))
    r = smooth_ball(t, held(x), np.zeros(n), np.zeros(n), cfg=SmootherConfig(method=method))
    assert r.speed[k - 8] < 2.0
    assert r.speed[k + 8] > 17.0


def test_bad_config_and_inputs_rejected():
    with pytest.raises(ValueError):
        SmootherConfig(method="spline")
    with pytest.raises(ValueError):
        SmootherConfig(savgol_window=8)
    with pytest.raises(ValueError):
        smooth_ball([0.0, 0.0], [1.0, 2.0], [0.0, 0.0])


def test_module_reads_no_tags_or_smoothed_fields():
    import inspect

    import m1_possession.smoother as s
    src = inspect.getsource(s).split('"""', 2)[2]          # code after the module docstring
    for banned in ("game_event", "possession_event", "ballsSmoothed", "PlayersSmoothed", "football_fatigue"):
        assert banned not in src
