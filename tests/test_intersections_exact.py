"""Exact spatiotemporal intersections A(t*) = B(t*) (``intersections.exact``) on synthetic RAW caches run through the real Step 1
quality pass. Cases: exact simultaneous crossing, spatial crossing at different times, same time at different positions, a
stationary player, a shared sample point, parallel and coincident motion, multiple intersections of one pair, overlapping windows
(with a brute-force all-windows / all-pairs equivalence check), missing data, players sampled at different instants; plus the
written outputs when present."""

import itertools
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from football_fatigue.encounters import quality as Q
from football_fatigue.encounters.trajectories import STATE_ABSENT, STATE_AVAILABLE, RawCache
from football_fatigue.intersections.detect import DetectConfig, detect
from football_fatigue.intersections.exact import (HALF_US, US, classify, detect_exact, export, merge_solutions, period_bounds, record,
                                                  solve_overlap, trajectories, windows_of)
from football_fatigue.intersections.detect import duplicate_positions

FPS, T0 = 10.0, 100.0
K = np.arange(101)                     # t = 100.0 ... 110.0 s


def cache(paths_xy) -> RawCache:
    F, S = len(paths_xy[0][0]), len(paths_xy)
    x = np.round(np.stack([p[0] for p in paths_xy], 1).astype(float), 3)      # RAW coordinates are millimetre decimals
    y = np.round(np.stack([p[1] for p in paths_xy], 1).astype(float), 3)
    ok = np.isfinite(x)
    sides = ["home" if s % 2 == 0 else "away" for s in range(S)]
    return RawCache(match_id=999, fps=FPS, tracking_path="synthetic", quarantine_periods=(),
                    frame_num=np.arange(F, dtype=np.int64), video_time_s=np.round(T0 + np.arange(F) / FPS, 6), source_period=np.ones(F, np.int16),
                    canonical_period=np.ones(F, np.int8), record_first=np.arange(F, dtype=np.int64), record_count=np.ones(F, np.int16),
                    frame_gap_before=np.full(F, -1, np.int32), frame_qflags=np.zeros(F, np.int64), slot_side=np.array(sides, "<U4"),
                    slot_jersey=np.arange(1, S + 1, dtype=np.int16), slot_player_id=np.arange(S, dtype=np.int64) + 987650,
                    slot_team_id=np.array([1 if s == "home" else 2 for s in sides], np.int64), x=np.where(ok, x, np.nan), y=np.where(ok, y, np.nan),
                    state=np.where(ok, STATE_AVAILABLE, STATE_ABSENT).astype(np.int8), visibility=np.where(ok, 1, 0).astype(np.int8),
                    confidence=np.where(ok, 3, 0).astype(np.int8), attribution=np.where(ok, 1, 0).astype(np.int8), qflags=np.zeros((F, S), np.int64))


def run(paths_xy):
    c = cache(paths_xy)
    qr = Q.assess(c, [])
    seg, counts = trajectories(c, qr.mask, qr.steps)
    sols, st = detect_exact(seg)
    events = merge_solutions(seg, sols)
    bounds = period_bounds(seg, c)
    recs = [record(c, qr.mask, seg, e, bounds, duplicate_positions(c)) for e in events]
    return recs, dict(st, events=events, seg=seg, c=c, mask=qr.mask, bounds=bounds, counts=counts)


def brute(seg, bounds) -> set:
    """Every window, every pair of available players, every pair of their segments: exact solutions inside the window."""
    found = set()
    for p, (start, end) in bounds.items():
        for k in range(0, (end - start) // US + 1):
            c = start + k * US
            lo, hi = max(c - HALF_US, start), min(c + HALF_US, end)
            idx = [g for g in range(len(seg["slot"])) if seg["period"][g] == p and seg["T1"][g] > lo and seg["T0"][g] < hi]
            for sa, sb in itertools.combinations(sorted({int(seg["slot"][g]) for g in idx}), 2):
                for ga in (g for g in idx if seg["slot"][g] == sa):
                    for gb in (g for g in idx if seg["slot"][g] == sb):
                        r = solve_overlap(seg, ga, gb)
                        if r is not None and r[0] == "point" and lo <= r[1] <= hi:
                            found.add((p, sa, sb, r[1]))
    return found


def cross(shift_k: float = 0.0):
    """A east along y = 0, B north along x = 0; both reach (0, 0) at frame 49.7 (B ``shift_k`` frames later)."""
    return [(0.1 * (K - 50) + 0.03, np.zeros(len(K))), (np.zeros(len(K)), 0.1 * (K - 50 - shift_k) + 0.03)]


def test_classify():
    assert classify((0, 0), (0, 0)) == ("interval", None)
    assert classify((0, 0), (3, 4)) == ("point", 0) and classify((3, 4), (0, 0)) == ("point", 1)
    assert classify((-70, 70), (30, -30)) == ("point", Fraction(7, 10))
    assert classify((-70, 71), (30, -30)) is None                          # passes 0.4 mm from each other: no solution
    assert classify((10, 0), (20, 0)) is None                              # collinear with the origin but not between


def test_exact_simultaneous_crossing():
    recs, st = run(cross(0.0))
    assert len(recs) == 1
    r = recs[0]
    assert r["kind"] == "point" and not r["at_sample_instant"] and r["synchronous_samples"]
    assert Fraction(r["t_us_exact"]) == 104_970_000 and (r["x"], r["y"]) == (0.0, 0.0)
    assert r["windows_game_s"] == [3, 4, 5, 6, 7] and r["canonical_window_game_s"] == 5
    assert r["n_windows"] == 5 and st["synchronous_solving_pairs"] == 1


def test_spatial_crossing_at_different_times_is_not_an_event():
    recs, st = run(cross(10.0))                                          # same point, B 1.0 s later
    assert recs == []
    c = st["c"]
    qr = Q.assess(c, [])
    assert len(detect(c, qr.mask, qr.steps, DetectConfig())[0]) == 1      # the 2D crossing exists (v1 detector)


def test_same_time_different_positions():
    recs, _ = run([(0.1 * (K - 50), np.zeros(len(K))), (0.1 * (K - 50), np.full(len(K), 1.0))])
    assert recs == []


def test_stationary_player():
    a = cross(0.0)[0]
    b = (np.zeros(len(K)), np.zeros(len(K)))                              # B stands at (0, 0) the whole time
    recs, st = run([a, b])
    assert st["counts"]["stationary_segments"] == 1
    assert len(recs) == 1 and recs[0]["kind"] == "point" and not recs[0]["synchronous_samples"]
    assert Fraction(recs[0]["t_us_exact"]) == 104_970_000


def test_shared_sample_point_counted_once():
    a = (0.1 * (K - 50), np.zeros(len(K)))                                 # both at (0, 0) at frame 50 exactly
    b = (np.zeros(len(K)), 0.1 * (K - 50))
    recs, _ = run([a, b])
    assert len(recs) == 1 and recs[0]["at_sample_instant"] and recs[0]["detections_merged"] == 2
    assert Fraction(recs[0]["t_us_exact"]) == 105_000_000
    assert recs[0]["v1_duplicate_position_rule_would_exclude"]             # v1 removes exactly this kind of coincidence


def test_parallel_and_coincident_motion():
    xa = 0.1 * (K - 50)
    yb = np.where((K >= 40) & (K <= 60), 0.0, 0.5)                         # B on A's path for 2 s, elsewhere 0.5 m away
    recs, _ = run([(xa, np.zeros(len(K))), (xa, yb)])
    iv = [r for r in recs if r["kind"] == "interval"]
    assert len(iv) == 1 and iv[0]["t_start_s"] == pytest.approx(104.0) and iv[0]["t_end_s"] == pytest.approx(106.0)
    assert len(recs) == 1                                                  # its end points belong to the interval
    recs, _ = run([(xa, np.zeros(len(K))), (xa, np.full(len(K), 0.5))])  # parallel, never together
    assert recs == []


def test_multiple_intersections_of_one_pair():
    x = 0.1 * (K - 50)
    ya = np.round(0.4 * np.sin(K / 3.0) + 0.0005, 3)                      # A weaves across B's line, never exactly 0 at a sample
    recs, st = run([(x, ya), (x, np.zeros(len(K)))])                    # B directly beside A in x at every instant
    n_sign = int((np.sign(ya[1:]) != np.sign(ya[:-1])).sum())
    assert len(recs) == n_sign >= 5 and len({r["event_id"] for r in recs}) == len(recs)
    assert all(r["kind"] == "point" and r["y"] == 0.0 for r in recs)
    assert {(r["slot_a"], r["slot_b"]) for r in recs} == {(0, 1)}


def test_overlapping_windows_and_brute_force_equivalence():
    x = 0.1 * (K - 50)
    ya = np.round(0.4 * np.sin(K / 3.0) + 0.0005, 3)
    paths = [(x, ya), (x, np.zeros(len(K))), cross(0.0)[1], (np.zeros(len(K)), np.zeros(len(K)) + 0.3)]
    recs, st = run(paths)
    pts = {(r["period"], r["slot_a"], r["slot_b"], Fraction(r["t_us_exact"])) for r in recs if r["kind"] == "point"}
    assert pts == brute(st["seg"], st["bounds"])
    for r in recs:
        if r["kind"] == "point":
            t = Fraction(r["t_us_exact"])
            assert r["windows_game_s"] == [k for k in range(0, 11) if abs(t - (100 + k) * US) <= HALF_US]
    assert sum(r["n_windows"] for r in recs) > len(recs)                   # repeated across windows, one record each


def test_missing_data():
    a, b = cross(0.0)
    ax, ay = a[0].copy(), a[1].copy()
    ax[44:55] = ay[44:55] = np.nan                                        # A unseen 1.1 s around the crossing: not joined
    assert run([(ax, ay), b])[0] == []
    ax, ay = a[0].copy(), a[1].copy()
    ax[48:51] = ay[48:51] = np.nan                                        # 0.4 s gap: joined (Step 1), A still linear
    recs, _ = run([(ax, ay), b])
    assert len(recs) == 1 and Fraction(recs[0]["t_us_exact"]) == 104_970_000 and not recs[0]["synchronous_samples"]


def test_players_sampled_at_different_instants():
    a, b = cross(0.0)
    ax = a[0].copy()
    ax[1::2] = ax[0::2][: len(ax[1::2])]                                  # A updates on even frames only (held for 2 frames)
    by = b[1].copy()
    by[2::2] = by[1::2][: len(by[2::2])]                                  # B updates on odd frames only
    recs, _ = run([(ax, a[1]), (b[0], by)])
    assert len(recs) == 1 and not recs[0]["synchronous_samples"]
    r = recs[0]
    # A: x linear between even frames = 0.1 (f - 50) + 0.03 at even f, so x = 0 at f = 49.7; B: y = 0 at f = 49.7 as well
    assert Fraction(r["t_us_exact"]) == 104_970_000 and (r["x"], r["y"]) == (0.0, 0.0)


def test_export_before_during_after():
    recs, st = run(cross(0.0))
    e = st["events"][0]
    parts = export(st["c"], st["mask"], st["seg"], recs[0], e["ga"], e["gb"])
    for side in ("a", "b"):
        b, a = parts[f"{side}_before"], parts[f"{side}_after"]
        assert tuple(b[-1, :3]) == tuple(a[0, :3]) == (0.0, 0.0, pytest.approx(104.97))
        assert np.all(b[:-1, 2] < 104.97) and np.all(a[1:, 2] > 104.97)
        assert b[0, 2] >= 102.5 - 1e-9 and a[-1, 2] <= 107.5 + 1e-9
        assert np.all(b[:-1, 3] >= 0)                                      # original source rows


def test_windows_of():
    assert windows_of(Fraction(104_970_000), Fraction(104_970_000), 100_000_000, 110_000_000) == [3, 4, 5, 6, 7]
    assert windows_of(Fraction(100_500_000), Fraction(100_500_000), 100_000_000, 110_000_000) == [0, 1, 2, 3]   # partial edge windows


ROOT = Path(__file__).resolve().parents[1] / "data" / "processed" / "m11-exact-spatiotemporal-v1"


@pytest.mark.skipif(not (ROOT / "3812" / "root.summary.json").exists(), reason="exact outputs not written")
def test_written_3812():
    s = json.loads((ROOT / "3812" / "root.summary.json").read_text(encoding="utf-8"))
    st = s["stages"]
    assert st["unique_events"] == st["points"] + st["coincident_intervals"]
    assert s["evidence"]["perturbation_half_mm"]["exact_solutions"] == 0
    recs = [json.loads(x) for x in (ROOT / "3812" / "root.events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(recs) == st["unique_events"] == len({r["event_id"] for r in recs})
    with np.load(ROOT / "3812" / "root.trajectories.npz") as z:
        assert len(np.unique(z["event_index"])) == len(recs)
