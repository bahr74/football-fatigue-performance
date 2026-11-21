"""Spatiotemporal occlusion-proxy decision (``intersections.spatiotemporal``) on synthetic RAW caches run through the real Step 1
quality pass, the unchanged v1 detector, the catalogue and the v3 measurements. Cases: exactly simultaneous, near-synchronous
within the uncertainty, clearly different times, multiple intersections, overlapping windows, a missing sample, ESTIMATED versus
observed positions, a slow player, a noise-made near miss, and the smoothing study; plus the written outputs when present."""

import json
from pathlib import Path

import numpy as np
import pytest

from football_fatigue.encounters import quality as Q
from football_fatigue.encounters.trajectories import STATE_ABSENT, STATE_AVAILABLE, RawCache
from football_fatigue.intersections import audit as AU
from football_fatigue.intersections.audit_v3 import AuditV3Config, audit_events
from football_fatigue.intersections.catalogue import describe, segment_index
from football_fatigue.intersections.detect import DetectConfig, detect, duplicate_positions, merge, period_windows, scan, segments, unique_hits
from football_fatigue.intersections.spatiotemporal import (CoincidenceConfig, decide, evaluate, light_events, match_events,
                                                          smooth_segments)

FPS, T0 = 10.0, 100.0
K = np.arange(101)                     # t = 100.0 ... 110.0 s; synthetic update interval h = 0.1 s


def cache(paths_xy, estimated_slots=()) -> RawCache:
    F, S = len(paths_xy[0][0]), len(paths_xy)
    x, y = np.stack([p[0] for p in paths_xy], 1).astype(float), np.stack([p[1] for p in paths_xy], 1).astype(float)
    ok = np.isfinite(x)
    sides = ["home" if s % 2 == 0 else "away" for s in range(S)]
    vis = np.where(ok, 1, 0).astype(np.int8)
    for s in estimated_slots:
        vis[ok[:, s], s] = 2
    return RawCache(match_id=999, fps=FPS, tracking_path="synthetic", quarantine_periods=(),
                    frame_num=np.arange(F, dtype=np.int64), video_time_s=T0 + np.arange(F) / FPS, source_period=np.ones(F, np.int16),
                    canonical_period=np.ones(F, np.int8), record_first=np.arange(F, dtype=np.int64), record_count=np.ones(F, np.int16),
                    frame_gap_before=np.full(F, -1, np.int32), frame_qflags=np.zeros(F, np.int64), slot_side=np.array(sides, "<U4"),
                    slot_jersey=np.arange(1, S + 1, dtype=np.int16), slot_player_id=np.arange(S, dtype=np.int64) + 987650,
                    slot_team_id=np.array([1 if s == "home" else 2 for s in sides], np.int64), x=np.where(ok, x, np.nan), y=np.where(ok, y, np.nan),
                    state=np.where(ok, STATE_AVAILABLE, STATE_ABSENT).astype(np.int8), visibility=vis,
                    confidence=np.where(ok, 3, 0).astype(np.int8), attribution=np.where(ok, 1, 0).astype(np.int8), qflags=np.zeros((F, S), np.int64))


def dataset(paths_xy, estimated_slots=()):
    c = cache(paths_xy, estimated_slots)
    qr = Q.assess(c, [])
    cfg = DetectConfig()
    ev, _ = detect(c, qr.mask, qr.steps, cfg)
    seg, _ = segments(c, qr.mask, qr.steps, duplicate_positions(c))
    key = segment_index(seg)
    cat = [describe(c, seg, key, e, period_windows(c, 1, cfg), cfg.half) for e in ev]
    el = {e["event_id"]: e for e in audit_events(AU.Context(c, qr.mask, AuditV3Config()), ev)}
    return [evaluate(r, el[r["event_id"]], CoincidenceConfig()) for r in cat], seg, c


def cross(shift_k: float = 0.0, speed: float = 0.1):
    """A east along y = 0, B north along x = 0 (``speed`` m per frame); A at (0, 0) at frame 49.7, B ``shift_k`` frames later."""
    a = (speed * (K - 50) + 0.3 * speed, np.zeros(len(K)))
    b = (np.zeros(len(K)), speed * (K - 50 - shift_k) + 0.3 * speed)
    return [a, b]


def test_decide_rule():
    assert decide(0.0, 0.03, 0.0667, False, [], [], 2, 1) == ("accepted", [])
    assert decide(0.3, 0.03, 0.0667, False, [], [], 2, 1) == ("rejected", ["sequential_pass"])
    assert decide(0.01, 0.05, 0.0667, False, [], [], 2, 1) == ("ambiguous", ["timing_precision_insufficient"])
    assert decide(0.06, 0.02, 0.0667, False, [], [], 2, 1) == ("ambiguous", ["beyond_noise_band_within_resolution"])
    assert decide(0.09, 0.05, 0.0667, False, [], [], 2, 1)[0] == "ambiguous"                 # noise could explain it
    assert decide(0.0, 0.03, 0.0667, True, ["slow_at_crossing"], ["duplicate_position"], 2, 1) == \
        ("ambiguous", ["timing_ill_conditioned", "missing_sample_at_crossing", "duplicate_position"])


def test_exactly_simultaneous_crossing_is_accepted():
    recs, _, _ = dataset(cross(0.0))
    assert len(recs) == 1
    co = recs[0]["coincidence"]
    assert co["abs_dt_s"] == pytest.approx(0.0, abs=1e-9) and co["timing_class"] == "exact_equality"
    assert co["tau_s"] == pytest.approx(0.1) and co["tol_s"] <= co["tau_s"]
    assert co["decision"] == "accepted" and recs[0]["quality"]["observed_both_at_crossing"]


def test_near_synchronous_within_uncertainty_is_accepted():
    recs, _, _ = dataset(cross(0.5))                                     # B 0.05 s after A: below 2 sigma and one update
    co = recs[0]["coincidence"]
    assert co["abs_dt_s"] == pytest.approx(0.05) and co["decision"] == "accepted"


def test_clearly_different_times_are_rejected():
    for shift, dt in ((2.0, 0.2), (10.0, 1.0)):
        co = dataset(cross(shift))[0][0]["coincidence"]
        assert co["abs_dt_s"] == pytest.approx(dt) and co["decision"] == "rejected" and co["reasons"] == ["sequential_pass"]


def test_multiple_intersections_each_decided():
    x = 0.08 * (K - 50) + 0.013
    a = (x, np.sin(2 * np.pi * x / 2.0))                                 # weaves across B's line
    b = (x + 0.031, np.full(len(K), 0.0071))                            # B alongside: reaches each crossing at the same time
    recs, _, _ = dataset([a, b])
    assert len(recs) >= 6 and len({r["event_id"] for r in recs}) == len(recs)
    assert all(r["coincidence"]["abs_dt_s"] < 0.1 for r in recs) and not any(r["coincidence"]["decision"] == "rejected" for r in recs)


def test_overlapping_windows_one_record_with_provenance():
    recs, _, _ = dataset(cross(0.0))
    r = recs[0]
    assert r["n_windows"] == 5 and r["window"]["consistent"] and r["dedup"]["segment_pair_hits"] == 1
    assert r["provenance"]["catalogue"] == "m11-occlusion-catalogue-v1"


def test_missing_sample_at_crossing_makes_it_ambiguous():
    a, b = cross(0.0)
    ax, ay = a[0].copy(), a[1].copy()
    ax[50] = ay[50] = np.nan                                             # A unseen for one sample right at the crossing
    co = dataset([(ax, ay), b])[0][0]
    assert co["quality"]["missing_sample_at_crossing"] and co["coincidence"]["decision"] == "ambiguous"
    assert "missing_sample_at_crossing" in co["coincidence"]["reasons"]


def test_estimated_positions_are_flagged_not_decided():
    obs = dataset(cross(0.0))[0][0]
    est = dataset(cross(0.0), estimated_slots=(1,))[0][0]
    assert est["coincidence"]["decision"] == obs["coincidence"]["decision"] == "accepted"
    assert est["quality"]["estimated_at_crossing_b"] and not est["quality"]["estimated_at_crossing_a"]
    assert not est["quality"]["observed_both_at_crossing"]


def test_slow_player_timing_is_ill_conditioned():
    a, _ = cross(0.0)
    b = (np.zeros(len(K)), 0.03 * (K - 50) + 0.009)                      # B at 0.3 m/s reaches (0, 0) at frame 49.7 too
    r = dataset([a, b])[0][0]
    assert r["coincidence"]["abs_dt_s"] == pytest.approx(0.0, abs=1e-9)
    assert r["coincidence"]["decision"] == "ambiguous" and "timing_ill_conditioned" in r["coincidence"]["reasons"]


def _detect_light(seg, c):
    hits, _ = scan(c, seg, [1], DetectConfig())
    return light_events(seg, merge(unique_hits(hits), seg, DetectConfig()))


def test_noise_made_near_miss_and_smoothing():
    a = (0.1 * (K - 50), np.zeros(len(K)))
    b = (-0.1 * (K - 50), np.full(len(K), 0.02))                         # head-on pass 2 cm apart: no intersection
    recs, _, _ = dataset([a, b])
    assert recs == []
    rng = np.random.default_rng(7)
    a_noisy = (a[0], rng.normal(0, 0.05, len(K)))                        # 5 cm jitter makes A's path weave across B's line
    recs, seg, c = dataset([a_noisy, b])
    dec = [r["coincidence"]["decision"] for r in recs]
    assert "rejected" in dec                                             # noise-made crossings far from the meeting time
    ss = smooth_segments(seg)
    assert len(_detect_light(ss, c)) < len(_detect_light(seg, c))       # smoothing removes jitter crossings (derived copy)


def test_smoothing_preserves_rows_times_runs_and_straight_lines():
    a = (0.1 * (K - 50), 0.02 * (K - 50))
    ax, ay = a[0].copy(), a[1].copy()
    ax[60:64] = ay[60:64] = np.nan                                        # 0.5 s gap: two runs
    c = cache([(ax, ay), cross(0.0)[1]])
    qr = Q.assess(c, [])
    seg, _ = segments(c, qr.mask, qr.steps, duplicate_positions(c))
    ss = smooth_segments(seg)
    for k in ("slot", "row0", "row1", "t0", "t1", "period"):
        assert np.array_equal(ss[k], seg[k])
    for k in ("x0", "y0", "x1", "y1"):
        assert np.allclose(ss[k], seg[k], atol=1e-9)                     # a straight line is a quadratic: unchanged


def test_match_events_one_to_one():
    raw = [dict(period=1, slot_a=0, slot_b=1, time_a_s=10.0, time_b_s=10.0, x=0.0, y=0.0)]
    sm = [dict(period=1, sa=0, sb=1, ta=10.1, tb=10.05, x=0.1, y=0.0), dict(period=1, sa=0, sb=1, ta=10.02, tb=10.0, x=0.0, y=0.0),
          dict(period=1, sa=0, sb=2, ta=10.0, tb=10.0, x=0.0, y=0.0)]
    assert match_events(raw, sm, CoincidenceConfig()) == {0: 1}


ROOT = Path(__file__).resolve().parents[1] / "data" / "processed" / "m11-occlusion-spatiotemporal-v1"


@pytest.mark.skipif(not (ROOT / "3812" / "root.summary.json").exists(), reason="spatiotemporal outputs not written")
def test_written_3812_dataset():
    s = json.loads((ROOT / "3812" / "root.summary.json").read_text(encoding="utf-8"))
    d = s["dataset"]
    assert d["candidates"] == 18469 == sum(d["decisions"].values())
    n = 0
    with open(ROOT / "3812" / "root.events.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            co = r["coincidence"]
            n += 1
            if co["decision"] == "accepted":
                assert co["abs_dt_s"] <= co["tol_s"] <= co["tau_s"] + 1e-12 and not r["quality"]["vetoes"]
            if co["decision"] == "rejected":
                assert co["abs_dt_s"] > max(co["tau_s"], co["tol_s"])
    assert n == 18469
    with np.load(ROOT / "3812" / "root.trajectories.npz") as z:
        assert len(np.unique(z["event_index"])) == d["decisions"]["accepted"] + d["decisions"]["ambiguous"]
