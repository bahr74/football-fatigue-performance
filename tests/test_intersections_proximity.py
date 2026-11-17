"""Synchronized proximity encounters (``intersections.proximity``) on synthetic RAW caches run through the real Step 1 quality pass.

Cases: exact crossing at the same time, geometric crossing at different times, minimum separation in each threshold band and at
equality, stationary players, changing direction, curved trajectories, asynchronous timestamps, missing samples, repeated
timestamps, gaps in valid support, coincident intervals, isolated contacts, repeated encounters of one pair, overlapping-window
deduplication (brute force), period boundaries, jersey / team identity; the analytic solver against a dense numerical reference;
same-timestamp evaluation; R = 0 against the exact solver; plus the written outputs when present."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from football_fatigue.encounters import quality as Q
from football_fatigue.encounters.trajectories import STATE_ABSENT, STATE_AVAILABLE, RawCache
from football_fatigue.intersections.detect import duplicate_positions
from football_fatigue.intersections.exact import HALF_US, US, detect_exact, merge_solutions, period_bounds, trajectories, windows_of
from football_fatigue.intersections.proximity import ProximityConfig, _pos, context, detect_all, episodes, qualify, step_index, traj_for
from football_fatigue.intersections.proximity_dataset import SUBSETS

FPS, T0 = 10.0, 100.0
K = np.arange(101)                     # t = 100.0 ... 110.0 s


def cache(paths_xy, times=None, periods=None, visibility=None) -> RawCache:
    F, S = len(paths_xy[0][0]), len(paths_xy)
    x = np.round(np.stack([p[0] for p in paths_xy], 1).astype(float), 3)
    y = np.round(np.stack([p[1] for p in paths_xy], 1).astype(float), 3)
    ok = np.isfinite(x)
    sides = ["home" if s % 2 == 0 else "away" for s in range(S)]
    t = np.round(T0 + np.arange(F) / FPS, 6) if times is None else np.asarray(times, float)
    per = np.ones(F, np.int8) if periods is None else np.asarray(periods, np.int8)
    vis = np.where(ok, 1, 0).astype(np.int8) if visibility is None else np.where(ok, visibility, 0).astype(np.int8)
    return RawCache(match_id=999, fps=FPS, tracking_path="synthetic", quarantine_periods=(),
                    frame_num=np.arange(F, dtype=np.int64), video_time_s=t, source_period=per.astype(np.int16),
                    canonical_period=per, record_first=np.arange(F, dtype=np.int64), record_count=np.ones(F, np.int16),
                    frame_gap_before=np.full(F, -1, np.int32), frame_qflags=np.zeros(F, np.int64), slot_side=np.array(sides, "<U4"),
                    slot_jersey=np.arange(1, S + 1, dtype=np.int16) * 3, slot_player_id=np.arange(S, dtype=np.int64) + 987650,
                    slot_team_id=np.array([1 if s == "home" else 2 for s in sides], np.int64), x=np.where(ok, x, np.nan), y=np.where(ok, y, np.nan),
                    state=np.where(ok, STATE_AVAILABLE, STATE_ABSENT).astype(np.int8), visibility=vis,
                    confidence=np.where(ok, 3, 0).astype(np.int8), attribution=np.where(ok, 1, 0).astype(np.int8), qflags=np.zeros((F, S), np.int64))


def run(paths_xy, **kw):
    c = cache(paths_xy, **kw)
    qr = Q.assess(c, [])
    seg, counts = trajectories(c, qr.mask, qr.steps)
    bounds = period_bounds(seg, c)
    rows, cnt, det, extra = detect_all(c, qr.mask, seg, bounds, duplicate_positions(c), ProximityConfig())
    return {r: det["recs"][r] for r in det["recs"]}, dict(c=c, mask=qr.mask, seg=seg, rows=rows, bounds=bounds, det=det, counts=counts, cnt=cnt)


def n(recs):
    return {r: len(v) for r, v in recs.items()}


def cross(shift_k: float = 0.0, speed: float = 0.1):
    """A east along y = 0, B north along x = 0; both reach (0, 0) at frame 49.7 (B ``shift_k`` frames later)."""
    return [(speed * (K - 50) + 0.3 * speed, np.zeros(len(K))), (np.zeros(len(K)), speed * (K - 50 - shift_k) + 0.3 * speed)]


def passing(h: float):
    """A east along y = 0, B west along y = h; equal x at frame 49.85 (between samples): d_min = h."""
    return [(0.1 * (K - 50) + 0.03, np.zeros(len(K))), (-0.1 * (K - 50), np.full(len(K), h))]


def vertices(seg, slot):
    g = np.flatnonzero(seg["slot"] == slot)
    return np.r_[seg["t0"][g], seg["t1"][g[-1]]], np.r_[seg["x0"][g], seg["x1"][g[-1]]], np.r_[seg["y0"][g], seg["y1"][g[-1]]]


def dense(seg, sa, sb, step=1e-5):
    """Dense numerical reference: both players interpolated from their own vertices at the same instants of a fine grid."""
    ta, xa, ya = vertices(seg, sa)
    tb, xb, yb = vertices(seg, sb)
    t = np.arange(max(ta[0], tb[0]), min(ta[-1], tb[-1]), step)
    return t, np.hypot(np.interp(t, ta, xa) - np.interp(t, tb, xb), np.interp(t, ta, ya) - np.interp(t, tb, yb))


def check_dense(recs, st, r_mm, sa=0, sb=1):
    t, d = dense(st["seg"], sa, sb)
    R = r_mm / 1000
    for e in recs[r_mm]:
        w = (t >= e["t_start_s"] - 1e-9) & (t <= e["t_end_s"] + 1e-9)
        if w.any():
            assert d[w].max() <= R + 1e-6                                 # inside the episode: always within R
            assert d[w].min() >= e["d_min_m"] - 1e-9 and d[w].min() <= e["d_min_m"] + 1e-3
    inside = d <= R - 1e-6                                               # every dense instant within R lies in an episode
    covered = np.zeros(len(t), bool)
    for e in recs[r_mm]:
        covered |= (t >= e["t_start_s"] - 2e-5) & (t <= e["t_end_s"] + 2e-5)
    assert not (inside & ~covered).any()


def test_exact_crossing_same_time():
    recs, st = run(cross(0.0))
    assert n(recs) == {0: 1, 100: 1, 250: 1, 500: 1}
    e0, e25 = recs[0][0], recs[250][0]
    assert e0["contact_class"] == "point" and e0["d_min_m"] == 0 and e0["exact_coincidence"]
    assert e0["t_star_s"] == pytest.approx(104.97, abs=1e-9)
    assert e25["t_star_s"] == pytest.approx(104.97, abs=1e-9) and e25["d_min_m"] == 0
    half = 0.25 / math.sqrt(2)                                            # d = sqrt(2) |t - t*| at 1 m/s each
    assert e25["t_start_s"] == pytest.approx(104.97 - half, abs=1e-9) and e25["t_end_s"] == pytest.approx(104.97 + half, abs=1e-9)
    assert e25["midpoint"] == pytest.approx([0.0, 0.0], abs=1e-12) and e25["parent_event_id"] == recs[500][0]["event_id"]
    for r in (100, 250, 500):
        check_dense(recs, st, r)


def test_geometric_crossing_at_different_times_is_rejected():
    recs, st = run(cross(10.0))                                          # B reaches (0, 0) 1.0 s after A
    assert n(recs) == {0: 0, 100: 0, 250: 0, 500: 0}
    t, d = dense(st["seg"], 0, 1)
    assert d.min() == pytest.approx(math.sqrt(0.5), abs=1e-4)            # closest same-time approach 0.71 m


@pytest.mark.parametrize("h,expect", [(0.05, {0: 0, 100: 1, 250: 1, 500: 1}), (0.2, {0: 0, 100: 0, 250: 1, 500: 1}),
                                      (0.4, {0: 0, 100: 0, 250: 0, 500: 1}), (0.6, {0: 0, 100: 0, 250: 0, 500: 0})])
def test_minimum_separation_bands(h, expect):
    recs, st = run(passing(h))
    assert n(recs) == expect
    for r, v in recs.items():
        for e in v:
            assert e["d_min_m"] == pytest.approx(h, abs=1e-9) and e["t_star_s"] == pytest.approx(104.985, abs=1e-9)
        check_dense(recs, st, r)


def test_threshold_equality_is_included():
    recs, _ = run(passing(0.25))                                         # minimum exactly 250 mm, between samples
    assert n(recs) == {0: 0, 100: 0, 250: 1, 500: 1}
    e = recs[250][0]
    assert e["contact_class"] == "point" and e["duration_s"] == 0 and e["d_min_m"] == pytest.approx(0.25, abs=1e-12)


def test_stationary_player():
    a = passing(0.0)[0]
    b = (np.zeros(len(K)), np.full(len(K), 0.2))                         # B stands 0.2 m off A's line the whole time
    recs, st = run([a, b])
    assert n(recs) == {0: 0, 100: 0, 250: 1, 500: 1}
    e = recs[250][0]
    assert e["d_min_m"] == pytest.approx(0.2, abs=1e-9) and e["player_b"]["held_position"] and not e["synchronous_only"]
    check_dense(recs, st, 250)


def test_changing_direction():
    xa = np.where(K <= 50, 0.1 * (K - 50), -0.1 * (K - 50))              # A runs east to (0, 0), turns and runs back west
    ya = np.where(K <= 50, 0.0, 0.002 * (K - 50))
    xb, yb = np.full(len(K), 0.05), np.full(len(K), 0.12)                # B stands near the turning point
    recs, st = run([(xa, ya), (xb, yb)])
    assert n(recs)[250] == 1 and n(recs)[0] == 0
    for r in (100, 250, 500):
        check_dense(recs, st, r)


def test_curved_trajectories_against_dense_reference():
    th = 2 * np.pi * K / 100
    a = (5.0 * np.cos(th), 5.0 * np.sin(th))                             # A anticlockwise on r = 5 m
    b = (5.15 * np.cos(-th + 1.0), 5.15 * np.sin(-th + 1.0))             # B clockwise on r = 5.15 m: two passes
    recs, st = run([a, b])
    assert n(recs)[250] == 2 and n(recs)[500] == 2 and n(recs)[100] == 0
    for r in (250, 500):
        check_dense(recs, st, r)


def test_asynchronous_timestamps():
    a, b = cross(0.0)
    ax = a[0].copy()
    ax[1::2] = ax[0::2][: len(ax[1::2])]                                 # A updates on even frames only
    by = b[1].copy()
    by[2::2] = by[1::2][: len(by[2::2])]                                 # B on odd frames only
    recs, st = run([(ax, a[1]), (b[0], by)])
    assert n(recs)[0] == 1 and recs[0][0]["t_star_s"] == pytest.approx(104.97, abs=1e-9)
    assert not recs[250][0]["synchronous_only"]
    for r in (100, 250, 500):
        check_dense(recs, st, r)


def test_same_timestamp_for_both_players():
    """Every pair-interval evaluates both players at its own [lo, hi]: positions equal independent interpolations at that t."""
    a, b = cross(0.0)
    ax = a[0].copy()
    ax[1::2] = ax[0::2][: len(ax[1::2])]
    recs, st = run([(ax, a[1]), b])
    seg, rows = st["seg"], st["rows"]
    for side, slot in (("a", 0), ("b", 1)):
        ta, xa, ya = vertices(seg, slot)
        g = rows["g" + side]
        for t in ("lo", "hi"):
            x, y = _pos(seg, g, rows[t])
            assert np.allclose(x / 1000, np.interp(rows[t] / 1e6, ta, xa), atol=1e-9)
    ax_, ay_ = _pos(seg, rows["ga"], rows["lo"])
    bx_, by_ = _pos(seg, rows["gb"], rows["lo"])
    assert np.allclose(rows["r0x"], ax_ - bx_) and np.allclose(rows["r0y"], ay_ - by_)


def test_missing_samples_and_support_gaps():
    xa = 0.1 * (K - 50)
    base = [(xa, np.zeros(len(K))), (xa, np.full(len(K), 0.1))]          # walking together 0.1 m apart
    recs, _ = run(base)
    assert n(recs)[250] == 1 and recs[250][0]["duration_s"] == pytest.approx(10.0)
    assert recs[250][0]["contact_class"] == "sustained" and recs[250][0]["support_cut_at_start"] and recs[250][0]["support_cut_at_end"]
    ax, ay = xa.copy(), np.zeros(len(K))
    ax[48:51] = ay[48:51] = np.nan                                       # 0.4 s unobserved: joined by Step 1, interpolated
    recs, _ = run([(ax, ay), base[1]])
    assert n(recs)[250] == 1 and recs[250][0]["player_a"]["spans_missing_samples"]
    ax[44:55] = ay[44:55] = np.nan                                       # 1.1 s: not joined -> two episodes, cut at the gap
    recs, _ = run([(ax, ay), base[1]])
    eps = sorted(recs[250], key=lambda e: e["t_start_s"])
    assert len(eps) == 2 and eps[0]["support_cut_at_end"] and eps[1]["support_cut_at_start"]
    assert eps[0]["t_end_s"] == pytest.approx(104.3) and eps[1]["t_start_s"] == pytest.approx(105.5)


def test_repeated_timestamps():
    t = np.round(T0 + K / FPS, 6)
    t[30] = t[29]                                                        # a duplicated timestamp
    recs, st = run(cross(0.0), times=t)
    assert np.all(st["seg"]["T1"] > st["seg"]["T0"]) and n(recs)[0] == 1
    assert all(np.isfinite(e["d_min_m"]) for v in recs.values() for e in v)


def test_coincident_interval():
    xa = 0.1 * (K - 50)
    yb = np.where((K >= 40) & (K <= 60), 0.0, 0.5)                       # B on A's path for 2 s, else 0.5 m away
    recs, st = run([(xa, np.zeros(len(K))), (xa, yb)])
    assert n(recs) == {0: 1, 100: 1, 250: 1, 500: 1}
    e0, e25 = recs[0][0], recs[250][0]
    assert e0["coincident_interval"] and e0["coincident_seconds"] == pytest.approx(2.0) and e0["duration_s"] == pytest.approx(2.0)
    assert e25["t_start_s"] == pytest.approx(103.95) and e25["t_end_s"] == pytest.approx(106.05) and e25["contact_class"] == "sustained"
    assert recs[500][0]["duration_s"] == pytest.approx(10.0)
    check_dense(recs, st, 250)


def test_isolated_contact():
    recs, st = run(cross(0.0, speed=1.0))                                # 10 m/s each: within 0.10 m for 0.014 s, between samples
    e = recs[100][0]
    assert e["contact_class"] == "isolated" and e["sample_instants_within"] == 0
    assert e["duration_s"] == pytest.approx(2 * 0.1 / (10 * math.sqrt(2)), abs=1e-9)
    assert recs[0][0]["contact_class"] == "point"


def test_repeated_encounters_of_one_pair():
    x = 0.1 * (K - 50)
    ya = np.round(0.4 * np.sin(K / 3.0) + 0.0005, 3)                     # A weaves across B's line
    recs, st = run([(x, ya), (x, np.zeros(len(K)))])
    n_sign = int((np.sign(ya[1:]) != np.sign(ya[:-1])).sum())
    assert len(recs[0]) == len(recs[100]) == n_sign >= 5                  # one episode per approach, never one per pair
    assert len({e["event_id"] for e in recs[100]}) == len(recs[100]) and len({e["pair_id"] for e in recs[100]}) == 1
    for r in (100, 250, 500):
        check_dense(recs, st, r)


def test_overlapping_windows_brute_force():
    x = 0.1 * (K - 50)
    ya = np.round(0.4 * np.sin(K / 3.0) + 0.0005, 3)
    paths = [(x, ya), (x, np.zeros(len(K))), cross(0.0)[1], (np.zeros(len(K)), np.full(len(K), 0.3))]
    recs, st = run(paths)
    seg, rows = st["seg"], st["rows"]
    for r in (100, 250, 500):
        idx, s1, s2 = qualify(rows, r)
        lo, hi = rows["lo"][idx].astype(float), rows["hi"][idx].astype(float)
        t_in, t_out = lo + s1 * (hi - lo), lo + s2 * (hi - lo)
        start, end = st["bounds"][1]
        found = {}
        for k in range(0, (end - start) // US + 1):                      # every window: clip, collect per pair
            c = start + k * US
            wlo, whi = max(c - HALF_US, start), min(c + HALF_US, end)
            for j in np.flatnonzero((t_out >= wlo) & (t_in <= whi)):
                pair = (int(seg["slot"][rows["ga"][idx[j]]]), int(seg["slot"][rows["gb"][idx[j]]]))
                found.setdefault(pair, []).append((max(t_in[j], wlo), min(t_out[j], whi)))
        merged = set()
        for pair, iv in found.items():                                   # global deduplication of the window detections
            iv.sort()
            cur = list(iv[0])
            for a, b in iv[1:]:
                if a <= cur[1]:
                    cur[1] = max(cur[1], b)
                else:
                    merged.add((pair, round(cur[0]), round(cur[1])))
                    cur = [a, b]
            merged.add((pair, round(cur[0]), round(cur[1])))
        got = {((e["player_a"]["slot"], e["player_b"]["slot"]), round(e["t_start_s"] * US), round(e["t_end_s"] * US)) for e in recs[r]}
        assert got == merged
        assert sum(e["n_windows"] for e in recs[r]) > len(recs[r])
        for e in recs[r]:
            assert e["windows_game_s"] == windows_of(e["t_start_s"] * US, e["t_end_s"] * US, start, end)


def test_period_boundaries():
    F = 101
    t = np.r_[np.round(T0 + np.arange(51) / FPS, 6), np.round(200.0 + np.arange(50) / FPS, 6)]
    per = np.r_[np.ones(51), np.full(50, 2)]
    xa = np.r_[np.linspace(-5, 0, 51), np.linspace(0, 5, 50)]
    xb = np.r_[np.linspace(5, 0, 51), np.linspace(0, -5, 50)]           # together at the end of period 1 and the start of period 2
    recs, st = run([(xa, np.zeros(F)), (xb, np.zeros(F))], times=t, periods=per)
    eps = sorted(recs[250], key=lambda e: e["t_start_s"])
    assert [e["period"] for e in eps] == [1, 2]
    assert eps[0]["support_cut_at_end"] and eps[1]["support_cut_at_start"]
    assert eps[1]["windows_game_s"][0] == 0 and eps[1]["period_start_s"] == pytest.approx(200.0)


def test_jersey_and_team_identity():
    recs, _ = run(cross(0.0) + [(np.zeros(len(K)) + 0.05, 0.1 * (K - 50) + 0.03)])
    by_pair = {(e["player_a"]["slot"], e["player_b"]["slot"]): e for e in recs[250]}
    e = by_pair[(0, 1)]
    assert (e["player_a"]["jersey"], e["player_a"]["side"], e["player_a"]["player_id"]) == (3, "home", 987650)
    assert (e["player_b"]["jersey"], e["player_b"]["side"], e["player_b"]["team_id"]) == (6, "away", 2)
    assert e["team_relation"] == "opposite_teams" and e["pair_id"] == "999|1|0|1"
    assert by_pair[(0, 2)]["team_relation"] == "same_team"


def test_estimated_positions_are_kept_and_flagged():
    vis = np.ones((len(K), 2), np.int8)
    vis[:, 1] = 2                                                         # B ESTIMATED throughout
    recs, _ = run(cross(0.0), visibility=vis)
    assert n(recs)[250] == 1 and recs[250][0]["pair_quality"] == "observed/estimated"
    assert "ESTIMATED" in recs[250][0]["player_b"]["quality_flags"]


def test_zero_threshold_matches_exact_solver():
    x = 0.1 * (K - 50)
    ya = np.round(0.4 * np.sin(K / 3.0) + 0.0005, 3)
    yb = np.where((K >= 40) & (K <= 60), 0.0, 0.5)
    for paths in ([(x, ya), (x, np.zeros(len(K)))], cross(0.0), [(x, np.zeros(len(K))), (x, yb)],
                  [(0.1 * (K - 50), np.zeros(len(K))), (np.zeros(len(K)), 0.1 * (K - 50))]):
        recs, st = run(paths)
        sols, _ = detect_exact(st["seg"])
        assert len(recs[0]) == len(merge_solutions(st["seg"], sols))


def test_context_export_marks_samples_and_interpolation():
    recs, st = run(cross(0.0))
    e = recs[250][0]
    ep = st["det"]["eps"][250][0]
    k = ep["best_row"]
    tr = context(st["c"], st["mask"], st["seg"], int(st["rows"]["ga"][k]), int(st["rows"]["gb"][k]), 102.5, 107.5,
                 [e["t_start_s"], e["t_star_s"], e["t_end_s"]])
    sy = tr["sync"]
    i = int(np.argmin(np.abs(sy[:, 0] - e["t_star_s"])))
    assert sy[i, 0] == pytest.approx(e["t_star_s"]) and not sy[i, 6] and not sy[i, 7]   # t* is interpolated for both
    assert sy[i, 5] == pytest.approx(0.0, abs=1e-9)
    assert tr["a"][0, 0] >= 102.5 - 1e-9 and tr["a"][-1, 0] <= 107.5 + 1e-9 and np.all(tr["a"][:, 3] >= 0)


ROOT = Path(__file__).resolve().parents[1] / "data" / "processed" / "m11-sync-proximity-v2"


@pytest.mark.skipif(not (ROOT / "3812" / "root.summary.json").exists(), reason="proximity outputs not written")
def test_written_3812():
    s = json.loads((ROOT / "3812" / "root.summary.json").read_text(encoding="utf-8"))
    th = s["thresholds"]
    for t, v in th.items():
        recs = [json.loads(x) for x in (ROOT / "3812" / f"root.events-{t}.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(recs) == v["unique_episodes"] == len({r["event_id"] for r in recs})
        assert v["point"] + v["isolated"] + v["sustained"] == v["unique_episodes"]
    with np.load(ROOT / "3812" / "root.trajectories.npz") as z:
        assert len(np.unique(z["sample_event_index"])) == th["0.25m"]["unique_episodes"]
    v1 = Path(str(ROOT).replace("-v2", "-v1")) / "3812"
    if v1.exists():                                                      # v2 changes context metadata only: identical episodes
        for t in th:
            ids = lambda d: [json.loads(x)["event_id"] for x in (d / f"root.events-{t}.jsonl").read_text(encoding="utf-8").splitlines()]  # noqa: E731
            assert ids(v1) == ids(ROOT / "3812")


def contexts(paths, **kw):
    recs, st = run(paths, **kw)
    si = step_index(Q.assess(st["c"], []).steps)
    out = []
    for e, r in zip(st["det"]["eps"][250], recs[250]):
        traj_for(st["c"], st["mask"], st["seg"], st["rows"], e, r, si)
        out.append(r)
    return out


def test_context_completeness_and_cut_reasons():
    xa = 0.1 * (K - 50)
    base = [(xa, np.zeros(len(K))), (xa, np.full(len(K), 0.1))]
    r = contexts(base)[0]                                                # 10 s together; canonical window inside the data
    assert r["context"]["complete"] and r["context"]["player_a"]["cut_before"] is None
    ax, ay = xa.copy(), np.zeros(len(K))
    ax[44:55] = ay[44:55] = np.nan                                       # 1.1 s unobserved: runs end, not joined
    eps = sorted(contexts([(ax, ay), base[1]]), key=lambda e: e["t_start_s"])
    first, second = eps                                                  # t* = earliest minimum: windows centred at 100 s and 105 s
    assert first["context"]["complete"]                                  # [100, 102.5] lies before the gap
    assert not second["context"]["complete"] and second["context"]["player_a"]["cut_before"] == "gap_period_or_time"
    assert second["context"]["player_b"]["complete_before"]
    jx = xa.copy()
    jx[55:] += 5.0                                                       # a 5 m teleport at frame 55: implausible jump, not joined
    eps = contexts([(jx, np.zeros(len(K))), (xa + 5.0, np.full(len(K), 0.1))])   # B walks beside where A lands
    assert len(eps) == 1 and eps[0]["t_start_s"] == pytest.approx(105.5)
    cx = eps[0]["context"]
    assert not cx["complete"] and cx["player_a"]["cut_before"] == "implausible_jump" and cx["player_b"]["complete_before"]


def test_subset_definitions():
    def rec(q="observed/observed", coinc=False, dup=False, cut=False, complete=True, crit=()):
        p = dict(duplicate_provider_position=dup)
        return dict(pair_quality=q, coincident_interval=coinc, player_a=p, player_b=dict(duplicate_provider_position=False),
                    support_cut_at_start=cut, support_cut_at_end=False, context=dict(complete=complete, critical_flags=list(crit)))
    hq = SUBSETS["high_quality_candidates"]
    assert hq(rec()) and not hq(rec(q="observed/estimated")) and not hq(rec(coinc=True)) and not hq(rec(dup=True))
    assert not hq(rec(cut=True)) and not hq(rec(complete=False)) and not hq(rec(crit=["IMPLAUSIBLE_JUMP"]))
    assert SUBSETS["mixed_quality"](rec(q="observed/estimated")) and SUBSETS["non_coincident"](rec()) and not SUBSETS["non_coincident"](rec(coinc=True))
