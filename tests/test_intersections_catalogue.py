"""Advisor occlusion-candidate catalogue (``intersections.catalogue``) on synthetic RAW caches run through the real Step 1 quality
pass and the unchanged v1 detector. Cases: simultaneous crossing, different-time crossing, multiple crossings, no intersection,
repeated encounters of one pair, one crossing seen by overlapping windows, degenerate trajectories, a self-revisiting loop;
Before/After reconstruction; and the written outputs when present."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from football_fatigue.encounters import quality as Q
from football_fatigue.encounters.trajectories import STATE_ABSENT, STATE_AVAILABLE, RawCache
from football_fatigue.intersections.advisor import project, time_at
from football_fatigue.intersections.catalogue import PAIRS_22, describe, paths, segment_index
from football_fatigue.intersections.detect import DetectConfig, detect, duplicate_positions, period_windows, segments

FPS, T0 = 10.0, 100.0
K = np.arange(101)                     # t = 100.0 ... 110.0 s


def cache(paths_xy) -> RawCache:
    F, S = len(paths_xy[0][0]), len(paths_xy)
    x, y = np.stack([p[0] for p in paths_xy], 1).astype(float), np.stack([p[1] for p in paths_xy], 1).astype(float)
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


def catalogue(paths_xy):
    c = cache(paths_xy)
    qr = Q.assess(c, [])
    cfg = DetectConfig()
    ev, stats = detect(c, qr.mask, qr.steps, cfg)
    seg, _ = segments(c, qr.mask, qr.steps, duplicate_positions(c))
    key = segment_index(seg)
    recs = [describe(c, seg, key, e, period_windows(c, 1, cfg), cfg.half) for e in ev]
    return recs, stats, seg, key


def along(verts, n=101, step=0.1, offset=0.03):
    """Positions every ``step`` metres of arc length (from ``offset``) along a polyline, one per frame."""
    v = np.asarray(verts, float)
    cum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(v, axis=0).T))]
    s = np.minimum(offset + step * np.arange(n), cum[-1])
    return np.interp(s, cum, v[:, 0]), np.interp(s, cum, v[:, 1])


def cross(shift_k: float = 0.0):
    """A east along y = 0, B north along x = 0; both reach (0, 0) at frame 49.7 (+ ``shift_k`` frames for B)."""
    a = (0.1 * (K - 50) + 0.03, np.zeros(len(K)))
    b = (np.zeros(len(K)), 0.1 * (K - 50 - shift_k) + 0.03)
    return [a, b]


def test_simultaneous_crossing():
    recs, _, _, _ = catalogue(cross(0.0))
    assert len(recs) == 1
    r = recs[0]
    tm = r["timing"]
    assert (r["x"], r["y"]) == pytest.approx((0.0, 0.0), abs=1e-12)
    assert tm["t_a_s"] == pytest.approx(104.97) and tm["t_b_s"] == pytest.approx(104.97)
    assert tm["numerical_tie"] and tm["first"] == "tie" and tm["concurrent_crossing_segments"]
    assert tm["b_distance_to_p_when_a_at_p_m"] == pytest.approx(0.0, abs=1e-9)
    assert r["advisor_recipe"]["agrees"] and "simultaneous" not in r


def test_different_time_crossing_is_kept_with_both_times():
    recs, _, _, _ = catalogue(cross(10.0))                               # B reaches P 1.0 s after A
    assert len(recs) == 1
    tm = recs[0]["timing"]
    assert tm["t_a_s"] == pytest.approx(104.97) and tm["t_b_s"] == pytest.approx(105.97)
    assert tm["dt_s"] == pytest.approx(1.0) and tm["first"] == "a" and not tm["numerical_tie"]
    assert not tm["concurrent_crossing_segments"] and tm["crossing_segments_overlap_s"] == pytest.approx(-0.9)
    assert tm["b_distance_to_p_when_a_at_p_m"] == pytest.approx(1.0) and tm["a_distance_to_p_when_b_at_p_m"] == pytest.approx(1.0)


def test_multiple_crossings_in_one_window():
    x = 0.08 * (K - 50) + 0.013
    a = (x, 1.0 * np.sin(2 * np.pi * x / 2.0))                          # zero crossings near x = -4 ... 4 every 1 m
    b = (x + 0.031, np.full(len(K), 0.0071))                            # B east along y = 0.0071, alongside A
    recs, _, _, _ = catalogue([a, b])
    assert len(recs) >= 6
    assert len({r["event_id"] for r in recs}) == len(recs)
    for r in recs:                                                       # each point is a separate event with its own exact times
        assert r["y"] == pytest.approx(0.0071, abs=1e-12) and r["advisor_recipe"]["agrees"] and r["window"]["consistent"]
    ts = sorted(r["timing"]["t_a_s"] for r in recs)
    assert np.all(np.diff(ts) > 0.3)


def test_no_intersection():
    a = (0.1 * (K - 50), np.zeros(len(K)))
    b = (0.1 * (K - 50), np.full(len(K), 0.3))                          # parallel, 0.3 m apart: no curve intersection
    recs, stats, _, _ = catalogue([a, b])
    assert recs == [] and stats["pair_windows_with_hit"] == 0 and stats["pair_checks"] == stats["windows"] == 6


def test_repeated_encounters_of_one_pair_stay_separate():
    ax = np.r_[-3 + 0.12 * K[:51], 3 - 0.12 * (K[51:] - 50)]           # out and back across B's path
    recs, _, _, _ = catalogue([(ax, np.full(len(K), 0.37)), (np.full(len(K), 0.23), 3 - 0.06 * K)])
    assert len(recs) == 2 and recs[0]["cluster_id"] == recs[1]["cluster_id"] and recs[0]["cluster_size"] == 2
    assert recs[1]["timing"]["t_a_s"] - recs[0]["timing"]["t_a_s"] == pytest.approx(4.6167, abs=1e-3)


def test_overlapping_windows_give_one_event_with_provenance():
    recs, stats, _, _ = catalogue(cross(0.0))
    assert stats["segment_pair_hits_per_window"] == 5 and stats["unique_events"] == 1         # five windows, one event
    w = recs[0]["window"]
    assert w["centres"] == w["expected_centres"] == [102.5, 103.5, 104.5, 105.5, 106.5] and w["consistent"]
    assert w["canonical_centre_s"] == 104.5 and w["start_s"] == pytest.approx(102.0) and w["end_s"] == pytest.approx(107.0)
    assert recs[0]["dedup"]["segment_pair_hits"] == 1 and recs[0]["dedup"]["windows"] == 5


def test_degenerate_trajectories():
    a = (0.1 * (K - 50) + 0.03, np.zeros(len(K)))
    by = np.r_[np.full(30, -2.0), -2.0 + 0.1 * np.arange(41) + 0.03, np.full(30, 2.03)]   # B stands, walks through, stands
    recs, stats, _, _ = catalogue([a, (np.zeros(len(K)), by)])
    assert len(recs) == 1 and stats["segments"]["zero_length"] == 0     # standing still adds no vertices (repeats are not updates)
    tm = recs[0]["timing"]
    assert tm["t_b_s"] == pytest.approx(T0 + (30 + 19.7) / FPS) and recs[0]["advisor_recipe"]["agrees"]
    c = cache(cross(0.0) + [cross(0.0)[0]])                              # slot 2 bit-identical to slot 0: provider artifact
    assert duplicate_positions(c)[:, [0, 2]].all()
    recs, _, _, _ = catalogue(cross(0.0) + [cross(0.0)[0]])
    assert recs == []                                                    # duplicate samples are never joined into segments


def test_self_revisit_uses_the_crossing_segment_not_project():
    ax, ay = along([(-2, 0), (1, 0), (1, 1), (0, 1), (0, -2)])           # A passes (0, 0) twice: t = 101.97 and 105.97
    b = (0.07 * K - 3.5 + 0.013, 0.07 * K - 3.5 + 0.013)                 # B on the diagonal through (0, 0)
    recs, _, _, _ = catalogue([(ax, ay), b])
    at_origin = [r for r in recs if abs(r["x"]) < 1e-9 and abs(r["y"]) < 1e-9]
    assert sorted(r["timing"]["t_a_s"] for r in at_origin) == pytest.approx([101.97, 105.97])
    v = np.array([(-1, 0), (1, 0), (1, 1), (0, 1), (0, -1)], float)     # exact arithmetic: project returns the first pass
    assert project(v, np.array([0.0, 0.0])) == 1.0
    assert time_at(v, np.arange(5.0), 1.0) == (0.5, 0)                  # ... although the point is also on the last segment


def test_before_after_reconstruction_ends_at_p():
    recs, _, seg, key = catalogue(cross(10.0))
    r = recs[0]
    p = paths(seg, r, key)
    for side, t in (("a", r["timing"]["t_a_s"]), ("b", r["timing"]["t_b_s"])):
        b, a = p[f"{side}_before"], p[f"{side}_after"]
        assert tuple(b[-1]) == pytest.approx((r["x"], r["y"], t)) and tuple(a[0]) == pytest.approx((r["x"], r["y"], t))
        assert np.all(np.diff(np.r_[b[:, 2], a[1:, 2]]) > 0)
        assert b[0, 2] >= r["window"]["start_s"] - 1e-9 and a[-1, 2] <= r["window"]["end_s"] + 1e-9
        tr = r["trajectories"][side]
        assert tr["n_vertices_before"] == len(b) - 1 and tr["n_vertices_after"] == len(a) - 1


def test_pair_count_formula():
    assert PAIRS_22 == 231 == 22 * 21 // 2 and PAIRS_22 * 90 * 60 == 1_247_400


ROOT = Path(__file__).resolve().parents[1] / "data" / "processed" / "m11-occlusion-catalogue-v1"


@pytest.mark.skipif(not (ROOT / "3812" / "root.summary.json").exists(), reason="catalogue outputs not written")
def test_written_3812_catalogue():
    s = json.loads((ROOT / "3812" / "root.summary.json").read_text(encoding="utf-8"))
    a, k = s["accounting"], s["catalogue"]
    assert s["v3_identical"] is True and k["events"] == a["events"] == 18469
    assert a["theoretical_pair_windows_22_players"] == math.comb(22, 2) * a["windows"]
    assert k["windows_consistent"] == k["events"]
    keys, n = set(), 0
    with open(ROOT / "3812" / "root.catalogue.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            n += 1
            keys.add((r["period"], r["slot_a"], r["slot_b"], r["segment_a"]["row0"], r["segment_b"]["row0"], round(r["x"], 6), round(r["y"], 6)))
            assert "simultaneous" not in r and r["timing"]["abs_dt_s"] == pytest.approx(abs(r["time_b_s"] - r["time_a_s"]))
    assert n == len(keys) == 18469                                       # no repeated physical event
