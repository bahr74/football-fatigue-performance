"""Stage S1 temporal-intersection quality audit (v2): thresholds, timing uncertainty, classification, lossless trajectory store,
identity separation, cluster accounting, determinism and v1 integrity. Synthetic RAW caches go through the real Step 1 quality
pass; the last tests read the written 3812 outputs when they exist (skipped otherwise)."""

import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import pytest

from football_fatigue.encounters import quality as Q
from football_fatigue.encounters.trajectories import STATE_ABSENT, STATE_AVAILABLE, RawCache, load_cache
from football_fatigue.intersections import audit as AU
from football_fatigue.intersections.detect import DetectConfig, detect

FPS, T0 = 10.0, 100.0
K = np.arange(101)


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


def audited(c, cfg=AU.AuditConfig()):
    qr = Q.assess(c, [])
    events, _ = detect(c, qr.mask, qr.steps, DetectConfig())
    cx = AU.Context(c, qr.mask, cfg)
    return cx, events, AU.audit_events(cx, events)


def crossing(speed_a=1.0, speed_b=1.0, angle_deg=90.0, offset_s=0.0):
    """A along x through (0.23, 0.37) at t = 105 s; B through the same point at 105 s + offset_s, at the given angle."""
    t = K / FPS - 5.0
    a = (0.23 + speed_a * t, np.full(len(K), 0.37))
    th = math.radians(angle_deg)
    tb = t - offset_s
    b = (0.23 + speed_b * math.cos(th) * tb + 0.0013, 0.37 + speed_b * math.sin(th) * tb)
    return cache([a, b])


# ---------------------------------------------------------------- classification rules

def M(dt, sigma):
    return dict(abs_time_diff_s=dt, timing_sigma_s=sigma)


CK0 = {k: False for k in ("duplicate_position", "attribution_unresolved_or_ambiguous", "period_boundary", "low_coverage", "internal_gap",
                          "velocity_unavailable", "slow_at_crossing", "shallow_angle", "stationary_player", "jitter", "implausible_jump",
                          "discontinuity_candidate", "substitution_window", "all_estimated_player")}
CFG = AU.AuditConfig()


def test_threshold_boundaries():
    assert AU.classify(M(0.20, 0.03), CK0, 0.20, CFG)["category"] == "clean_candidate"           # |dt| == tau is aligned
    s = AU.classify(M(0.2001, 0.03), CK0, 0.20, CFG)
    assert s["category"] == "uncertain_candidate" and s["reasons"] == ["timing_boundary"] and s["band_s"] == pytest.approx(0.26)
    assert AU.classify(M(0.26, 0.03), CK0, 0.20, CFG)["category"] == "uncertain_candidate"         # band edge inclusive
    s = AU.classify(M(0.2601, 0.03), CK0, 0.20, CFG)
    assert s["category"] == "non_time_aligned_crossing" and s["reasons"] == ["abs_time_diff_exceeds_tau_plus_z_sigma"]
    assert AU.classify(M(1.19, 5.0), CK0, 0.20, CFG)["timing"] == "boundary"                       # sigma capped at 0.5 s
    assert AU.classify(M(1.21, 5.0), CK0, 0.20, CFG)["timing"] == "not_aligned"
    assert AU.classify(M(0.1, None), CK0, 0.20, CFG)["reasons"] == ["timing_ill_conditioned"]      # unknown sigma: cap, ill-conditioned


def test_timing_ill_conditioned_depends_on_tau():
    assert AU.classify(M(0.05, 0.06), CK0, 0.10, CFG)["reasons"] == ["timing_ill_conditioned"]      # 0.06 > 0.5 * 0.10
    assert AU.classify(M(0.05, 0.06), CK0, 0.20, CFG)["category"] == "clean_candidate"
    for flag in ("slow_at_crossing", "shallow_angle", "velocity_unavailable"):
        assert AU.classify(M(0.05, 0.01), CK0 | {flag: True}, 0.20, CFG)["reasons"] == ["timing_ill_conditioned"]


def test_invalid_takes_precedence_and_all_reasons_are_kept():
    s = AU.classify(M(3.0, 0.03), CK0 | {"duplicate_position": True, "jitter": True}, 0.20, CFG)
    assert s["category"] == "invalid_tracking_artifact"
    assert s["reasons"] == ["duplicate_position", "abs_time_diff_exceeds_tau_plus_z_sigma", "jitter"]
    s = AU.classify(M(0.1, 0.03), CK0 | {"implausible_jump": True, "all_estimated_player": True}, 0.20, CFG)
    assert s["category"] == "uncertain_candidate" and s["reasons"] == ["implausible_jump", "all_estimated_player"]


def test_noisy_events_are_not_invalid():
    for flag in ("jitter", "implausible_jump", "stationary_player", "discontinuity_candidate", "substitution_window", "all_estimated_player"):
        assert AU.classify(M(0.1, 0.03), CK0 | {flag: True}, 0.20, CFG)["category"] == "uncertain_candidate"


# ---------------------------------------------------------------- measurements

def test_timing_uncertainty_fast_perpendicular_vs_slow_shallow():
    cx, events, recs = audited(crossing(speed_a=4.0, speed_b=4.0, angle_deg=90))
    m = recs[0]["measurements"]
    assert m["crossing_angle_deg"] == pytest.approx(90, abs=0.5)
    assert m["motion_a"]["speed_m_s"] == pytest.approx(4.0, abs=1e-6) and m["motion_b"]["speed_m_s"] == pytest.approx(4.0, abs=1e-6)
    q = cx.update_interval_s / math.sqrt(12)
    assert m["timing_sigma_s"] == pytest.approx(math.sqrt(2) * q, abs=1e-4)          # noise-free lines: quantisation only
    assert recs[0]["status"]["0.20"]["category"] == "clean_candidate"
    _, _, recs = audited(crossing(speed_a=0.3, speed_b=0.3, angle_deg=10))
    r = recs[0]
    assert r["checks"]["slow_at_crossing"] and r["checks"]["shallow_angle"]
    assert "timing_ill_conditioned" in r["status"]["0.20"]["reasons"]


def test_timing_uncertainty_formula_with_noise():
    rng = np.random.default_rng(1)
    c = crossing(speed_a=3.0, speed_b=2.0, angle_deg=60)
    c.x[:, 0] += rng.normal(0, 0.03, len(K))
    c.y[:, 0] += rng.normal(0, 0.03, len(K))                                       # isotropic noise on A only
    cx, events, recs = audited(c)
    m = recs[0]["measurements"]
    sa, sb = m["motion_a"]["local_sigma_m"], m["motion_b"]["local_sigma_m"]
    s = math.sin(math.radians(m["crossing_angle_deg"]))
    want = math.sqrt((sa / (m["motion_a"]["speed_m_s"] * s)) ** 2 + (sb / (m["motion_b"]["speed_m_s"] * s)) ** 2
                     + 2 * (cx.update_interval_s / math.sqrt(12)) ** 2)
    assert 0.015 < sa < 0.06 and sb < 1e-6
    assert m["timing_sigma_s"] == pytest.approx(want, rel=2e-3)


def test_time_difference_and_same_time_distance():
    _, events, recs = audited(crossing(speed_a=2.0, speed_b=2.0, offset_s=0.5))
    m = recs[0]["measurements"]
    assert m["abs_time_diff_s"] == pytest.approx(0.5, abs=1e-3)
    assert recs[0]["status"]["0.20"]["timing"] == "not_aligned" and recs[0]["status"]["0.30"]["timing"] == "not_aligned"
    assert m["min_same_time_distance_m"] == pytest.approx(2.0 * 0.5 / math.sqrt(2), abs=0.05)    # perpendicular, 1 m lag


# ---------------------------------------------------------------- store, references, accounting

def test_store_is_lossless_and_complete():
    c = crossing(speed_a=3.0, speed_b=3.0)
    cx, events, recs = audited(c)
    store, refs = AU.build_store(cx, events, recs)
    lo, hi = recs[0]["measurements"]["window_s"]
    assert hi - lo == pytest.approx(5.0) and recs[0]["measurements"]["t_star_s"] == pytest.approx(105.0, abs=0.01)
    rows = np.flatnonzero((c.video_time_s >= lo - 1e-4) & (c.video_time_s <= hi + 1e-4))
    assert len(refs) == 1 and store["offsets"].tolist() == [0, len(rows)] and len(rows) >= 50   # every frame of the window
    tr = refs[0]["tracks"]
    for k in (0, 1):
        s = tr[f"track_{k}"]["slot"]
        np.testing.assert_array_equal(store["x"][:, k], c.x[rows, s])
        np.testing.assert_array_equal(store["y"][:, k], c.y[rows, s])
        np.testing.assert_array_equal(store["visibility"][:, k], c.visibility[rows, s])
        np.testing.assert_array_equal(store["quality_mask"][:, k], cx.mask[rows, s])
    np.testing.assert_array_equal(store["video_time_s"], c.video_time_s[rows])
    assert np.isfinite(store["x"]).all()                                             # interaction interval not masked


def test_identity_separation():
    cx, events, recs = audited(crossing(speed_a=3.0, speed_b=3.0))
    store, refs = AU.build_store(cx, events, recs)
    assert set(store) == {"event_id", "t_star_s", "period", "offsets", *AU.STORE_ROW_KEYS, *AU.STORE_TRACK_KEYS}
    blob = AU.npz_bytes(store)
    with np.load(io.BytesIO(blob)) as z:
        assert all(z[k].dtype.kind != "U" or k == "event_id" for k in z.files)
    assert {"slot", "side", "jersey", "player_id", "team_id"} <= set(refs[0]["tracks"]["track_0"])
    orders = set()
    for k in range(40):                                                               # seeded track order is not the slot order
        _, _, r2 = audited(crossing(speed_a=3.0, speed_b=3.0), AU.AuditConfig(seed=f"s{k}"))
        _, rf = AU.build_store(AU.Context(cx.c, cx.mask, AU.AuditConfig(seed=f"s{k}")), events, r2)
        orders.add(rf[0]["tracks"]["track_0"]["slot"])
    assert orders == {0, 1}


def test_one_record_per_event_and_cluster_accounting():
    ax = np.r_[-3 + 0.12 * K[:51], 3 - 0.12 * (K[51:] - 50)]
    c = cache([(ax, np.full(len(K), 0.37)), (np.full(len(K), 0.23), 3 - 0.06 * K)])     # two separate crossings of one pair
    _, events, recs = audited(c)
    assert [r["event_id"] for r in recs] == [e["event_id"] for e in events] and len(recs) == 2
    assert {r["pair_id"] for r in recs} == {"999:0-1"}
    fake = [dict(match_id=1, period=1, pair_id="p", measurements=dict(t_star_s=t), status={"0.20": dict(category=cat)})
            for t, cat in ((10, "clean_candidate"), (14, "uncertain_candidate"), (30, "clean_candidate"), (12, "non_time_aligned_crossing"))]
    AU.candidate_clusters(fake, "0.20", 5.0)
    ids = [f["candidate_cluster"]["0.20"] for f in fake]
    assert ids[0] == ids[1] != ids[2] and ids[3] is None


def test_classification_is_monotone_in_tau():
    rng = np.random.default_rng(0)
    for dt in np.r_[0, rng.uniform(0, 1.5, 300)]:
        for sigma in (0.01, 0.04, 0.2, None):
            t = [AU.classify(M(float(dt), sigma), CK0, tau, CFG)["timing"] for tau in CFG.thresholds_s]
            rank = [("not_aligned", "boundary", "aligned").index(x) for x in t]
            assert rank == sorted(rank)


def test_deterministic_outputs():
    c = crossing(speed_a=3.0, speed_b=2.0, angle_deg=60)
    out = []
    for _ in range(2):
        cx, events, recs = audited(c)
        store, refs = AU.build_store(cx, events, recs)
        out.append((AU.jsonl(recs), AU.jsonl(refs), AU.npz_bytes(store)))
    assert out[0] == out[1]


# ---------------------------------------------------------------- real outputs (3812)

DATA = Path(__file__).resolve().parents[1] / "data"
V1 = DATA / "processed" / "m11-intersections-v1" / "3812"
V2 = DATA / "processed" / "m11-intersections-v2" / "3812"


@pytest.mark.skipif(not (V1 / "root.manifest.json").exists(), reason="v1 outputs not present")
def test_v1_outputs_intact():
    events, prov = AU.load_v1(DATA / "raw", "3812")
    assert len(events) == 18469 and prov["files_verified"] == 14


@pytest.mark.skipif(not (V2 / "root.manifest.json").exists(), reason="v2 outputs not written")
def test_written_v2_outputs():
    m = json.loads((V2 / "root.manifest.json").read_text(encoding="utf-8"))
    for name, digest in m["outputs_sha256"].items():
        assert hashlib.sha256((V2 / name).read_bytes()).hexdigest() == digest
    assert m["v1_unchanged"] and m["raw_unchanged"] and m["split"]["split"] == "development"
    recs = [json.loads(line) for line in (V2 / "root.eligibility.jsonl").read_text(encoding="utf-8").splitlines()]
    v1 = [json.loads(line) for line in (V1 / "root.events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["event_id"] for r in recs] == [e["event_id"] for e in v1]
    assert all(r["primary_category"] in AU.CATEGORIES for r in recs)
    s = json.loads((V2 / "root.summary.json").read_text(encoding="utf-8"))
    for k, v in s["by_threshold"].items():
        assert sum(v["categories"].values()) == len(recs)
    refs = [json.loads(line) for line in (V2 / "root.references.jsonl").read_text(encoding="utf-8").splitlines()]
    c = load_cache(DATA / "processed" / "m11-encounters-v2" / "3812" / "root.cache.npz")
    with np.load(V2 / "root.trajectories.npz") as z:
        st = {k: z[k] for k in z.files}
    assert st["event_id"].tolist() == [r["event_id"] for r in refs] and st["offsets"][-1] == len(st["x"])
    for r in refs[:: max(1, len(refs) // 25)]:                                       # lossless spot check against the cache
        a, b = st["offsets"][r["store_index"]], st["offsets"][r["store_index"] + 1]
        rows = np.searchsorted(c.frame_num, st["frame_num"][a:b])
        for k in (0, 1):
            np.testing.assert_array_equal(st["x"][a:b, k], c.x[rows, r["tracks"][f"track_{k}"]["slot"]])
    ws = (V2 / "root.review_worksheet.csv").read_text(encoding="utf-8")
    for token in ("home", "away", "slot", "jersey", "time_diff", "clean", "uncertain", "invalid", "aligned"):
        assert token not in ws
