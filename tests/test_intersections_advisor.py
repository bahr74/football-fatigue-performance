"""Advisor's definition of the occlusion point (exact curve intersection, then each player's time at that point), on synthetic
RAW caches run through the real Step 1 quality pass and the unchanged v1 detector, cross-checked with the advisor's
project/searchsorted recipe. Cases: equal arrival, different arrival, multiple crossings, no crossing, noisy and
degenerate trajectories; plus the written 3812 cross-check when present."""

import json
from pathlib import Path

import numpy as np
import pytest

from football_fatigue.encounters import quality as Q
from football_fatigue.encounters.trajectories import STATE_ABSENT, STATE_AVAILABLE, RawCache
from football_fatigue.intersections.advisor import advisor_times, project, time_at
from football_fatigue.intersections.detect import DetectConfig, detect

FPS, T0 = 10.0, 100.0
K = np.arange(101)
t = K / FPS - 5.0                     # relative time; video time = 105 + t


def cache(paths) -> RawCache:
    F, S = len(paths[0][0]), len(paths)
    x, y = np.stack([p[0] for p in paths], 1).astype(float), np.stack([p[1] for p in paths], 1).astype(float)
    ok = np.isfinite(x)
    sides = ["home" if s % 2 == 0 else "away" for s in range(S)]
    return RawCache(match_id=999, fps=FPS, tracking_path="synthetic", quarantine_periods=(),
                    frame_num=np.arange(F, dtype=np.int64), video_time_s=T0 + np.arange(F) / FPS, source_period=np.ones(F, np.int16),
                    canonical_period=np.ones(F, np.int8), record_first=np.arange(F, dtype=np.int64), record_count=np.ones(F, np.int16),
                    frame_gap_before=np.full(F, -1, np.int32), frame_qflags=np.zeros(F, np.int64), slot_side=np.array(sides, "<U4"),
                    slot_jersey=np.arange(1, S + 1, dtype=np.int16), slot_player_id=np.arange(S, dtype=np.int64) + 987650,
                    slot_team_id=np.array([1 if s == "home" else 2 for s in sides], np.int64), x=np.where(ok, x, np.nan), y=np.where(ok, y, np.nan),
                    state=np.where(ok, STATE_AVAILABLE, STATE_ABSENT).astype(np.int8), visibility=np.where(ok, 1, 0).astype(np.int8),
                    confidence=np.where(ok, 3, 0).astype(np.int8), attribution=np.where(ok, 1, 0).astype(np.int8), qflags=np.zeros((F, S), np.int64))


def events(paths):
    c = cache(paths)
    qr = Q.assess(c, [])
    ev, _ = detect(c, qr.mask, qr.steps, DetectConfig())
    return ev


def advisor(paths, e):
    """The advisor's recipe on the full polylines (all samples), for the event's point."""
    (ax, ay), (bx, by) = paths
    va, vb = np.c_[ax, ay], np.c_[bx, by]
    return advisor_times(va, T0 + K / FPS, vb, T0 + K / FPS, np.array([e["x"], e["y"]]))


A_LINE = (0.23 + 2.0 * t, np.full(len(t), 0.37))                                   # A along x through (0.23, 0.37) at t = 0


def b_line(offset_s):
    """B along y through the same point at t = offset_s."""
    return (np.full(len(t), 0.2313), 0.37 + 1.5 * (t - offset_s) + 0.0007)


def test_equal_arrival_gives_one_point_and_equal_times():
    paths = [A_LINE, b_line(0.0)]
    ev = events(paths)
    assert len(ev) == 1 and ev[0]["type"] == "proper_crossing"
    e = ev[0]
    assert e["x"] == pytest.approx(0.2313, abs=1e-9) and e["y"] == pytest.approx(0.37, abs=1e-9)
    assert e["time_a_s"] == pytest.approx(105.0 + (0.2313 - 0.23) / 2.0, abs=1e-9)
    assert e["abs_time_diff_s"] < 0.002                                             # same time up to the offsets of the construction
    assert advisor(paths, e) == pytest.approx((e["time_a_s"], e["time_b_s"]), abs=1e-9)


def test_different_arrival_is_preserved_not_forced_equal():
    paths = [A_LINE, b_line(1.5)]
    ev = events(paths)
    assert len(ev) == 1
    e = ev[0]
    assert e["time_b_s"] - e["time_a_s"] == pytest.approx(1.5 - 0.0007 / 1.5 + 0.0, abs=2e-3)
    assert e["abs_time_diff_s"] > 1.4 and not e["simultaneous"]
    assert advisor(paths, e) == pytest.approx((e["time_a_s"], e["time_b_s"]), abs=1e-9)


def test_multiple_crossings_are_separate_events_with_exact_times():
    a = (2.0 * t + 0.011, np.full(len(t), 0.0013))
    b = (0.5 * t, 0.5 * np.sin(np.pi * t))                                          # crosses y = 0.0013 about once per second
    ev = events([a, b])
    yb = b[1] - 0.0013
    expected = int(np.sum(np.sign(yb[:-1]) != np.sign(yb[1:])))
    xs_a = a[0]
    assert len(ev) == expected >= 8
    for e in ev:
        assert e["y"] == pytest.approx(0.0013, abs=1e-9)
        assert e["time_a_s"] == pytest.approx(105.0 + (e["x"] - 0.011) / 2.0, abs=1e-9)          # A's time from its own track
        assert xs_a.min() <= e["x"] <= xs_a.max()
        assert advisor([a, b], e) == pytest.approx((e["time_a_s"], e["time_b_s"]), abs=1e-9)
    assert len({round(e["time_b_s"], 6) for e in ev}) == len(ev)


def test_no_crossing_gives_no_event():
    assert events([A_LINE, (0.23 + 2.0 * t, np.full(len(t), 1.37))]) == []                           # parallel
    assert events([(2.0 * t, 0.4 + 0 * t), (-2.0 * t, -0.4 + 0 * t)]) == []                           # head-on pass, 0.8 m apart


def test_noisy_and_degenerate_trajectories():
    rng = np.random.default_rng(0)
    ax = 0.23 + 2.0 * t + rng.normal(0, 0.05, len(t))
    ay = 0.37 + rng.normal(0, 0.05, len(t))
    ax[70:74], ay[70:74] = ax[70], ay[70]                                           # repeated points (a stationary provider hold)
    a = (ax, ay)
    b = b_line(0.0)
    ev = events([a, b])
    assert len(ev) >= 1
    for e in ev:
        assert advisor([a, b], e) == pytest.approx((e["time_a_s"], e["time_b_s"]), abs=1e-9)
    verts = np.c_[[0, 1, 1, 2], [0, 0, 0, 0]].astype(float)                          # zero-length middle segment
    s = project(verts, np.array([1.5, 0.3]))
    assert s == pytest.approx(1.5) and time_at(verts, np.array([0.0, 1.0, 2.0, 3.0]), s)[0] == pytest.approx(2.5)


@pytest.mark.skipif(not Path("data/processed/m11-intersections-v1/3812/root.events.jsonl").exists(), reason="v1 outputs not generated")
def test_written_3812_crosscheck_report_if_present():
    p = Path("data/processed/m11-advisor-check-v1/3812.json")
    if not p.exists():
        pytest.skip("cross-check report not written")
    r = json.loads(p.read_text(encoding="utf-8"))
    assert r["point_max_err_m"] < 1e-9 and r["time_agree"] + r["self_revisit"] == r["events"] and r["self_revisit"] <= 0.001 * r["events"]
