"""M11 advisor-directed trajectory-intersection pipeline: geometry, detection, deduplication and four-segment samples.

Detection and sample tests use small SYNTHETIC RAW caches passed through the real Step 1 quality pass (``quality.assess``), so
segments come from the same steps and flags as on match data. The last tests read the written 3812 prototype outputs when they
exist (skipped otherwise).
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from football_fatigue.encounters import quality as Q
from football_fatigue.encounters.trajectories import STATE_ABSENT, STATE_AVAILABLE, RawCache
from football_fatigue.ingestion import IngestionError
from football_fatigue.intersections import geometry as G
from football_fatigue.intersections.__main__ import write_dir
from football_fatigue.intersections.detect import DetectConfig, Timeline, detect, duplicate_positions
from football_fatigue.intersections.samples import INSTANCE_KEYS, TRACKLET_KEYS, SampleConfig, _bits, build_sample, build_samples

FPS, T0 = 10.0, 100.0


# ---------------------------------------------------------------- geometry

def seg(*pts):
    return np.array([pts[0]], float), np.array([pts[1]], float)


def hit(a, b, tol=1e-6):
    return G.intersect(*seg(*a), *seg(*b), tol_m=tol)


def test_proper_crossing():
    h = hit(((0, 0), (2, 2)), ((0, 2), (2, 0)))
    assert len(h) == 1 and h.kind[0] == G.PROPER
    assert (h.x[0], h.y[0], h.ta[0], h.tb[0]) == pytest.approx((1, 1, 0.5, 0.5))


def test_non_crossing_and_parallel():
    assert len(hit(((0, 0), (1, 0)), ((2, -1), (2, 1)))) == 0           # line would cross, segments do not
    assert len(hit(((0, 0), (2, 0)), ((0, 1), (2, 1)))) == 0            # parallel, not collinear
    assert len(hit(((0, 0), (1, 0)), ((2, 0), (3, 0)))) == 0            # collinear, disjoint


def test_endpoint_touch_and_t_junction():
    h = hit(((0, 0), (1, 1)), ((1, 1), (2, 0)))
    assert len(h) == 1 and h.kind[0] == G.ENDPOINT
    h = hit(((0, 0), (2, 0)), ((1, 0), (1, 1)))
    assert len(h) == 1 and h.kind[0] == G.ENDPOINT and (h.x[0], h.y[0]) == pytest.approx((1, 0))


def test_collinear_overlap_and_single_point():
    h = hit(((0, 0), (2, 0)), ((1, 0), (3, 0)))
    assert len(h) == 1 and h.kind[0] == G.COLLINEAR and (h.x[0], h.y[0]) == pytest.approx((1.5, 0))
    h = hit(((0, 0), (1, 0)), ((1, 0), (2, 0)))                          # collinear, sharing one end only
    assert len(h) == 1 and h.kind[0] == G.ENDPOINT


def test_degenerate_and_tolerance():
    assert len(hit(((1, 0), (1, 0)), ((0, -1), (2, 1)))) == 0          # repeated coordinate: zero-length segment never tested
    assert len(hit(((0, 0), (1, 0)), ((1 + 5e-7, -1), (1 + 5e-7, 1)))) == 1   # within tol of A's end
    assert len(hit(((0, 0), (1, 0)), ((1 + 1e-3, -1), (1 + 1e-3, 1)))) == 0


def test_vectorised_matches_pairwise():
    rng = np.random.default_rng(0)
    a0, a1, b0, b1 = (rng.uniform(-5, 5, (30, 2)) for _ in range(4))
    h = G.intersect(a0, a1, b0, b1)
    pair = {(i, j) for i in range(30) for j in range(30) if len(G.intersect(a0[i], a1[i], b0[j], b1[j]))}
    assert set(zip(h.i.tolist(), h.j.tolist())) == pair and len(pair) > 0


def test_classify_vertex():
    b0, b1 = np.array([-1.0, 0]), np.array([1.0, 0])
    assert G.classify_vertex(np.array([0, -1.0]), np.array([0, 1.0]), b0, b1) == "crossing"
    assert G.classify_vertex(np.array([-1, 1.0]), np.array([1, 1.0]), b0, b1) == "touch"
    assert G.classify_vertex(np.array([0, 0.0]), np.array([1, 1.0]), b0, b1) == "undetermined"


# ---------------------------------------------------------------- synthetic RAW caches

def cache(paths: list[tuple[np.ndarray, np.ndarray]], sides=None) -> RawCache:
    """Slots moving along the given (x, y) arrays (NaN = no observation), one period, 10 Hz, every sample an update."""
    F, S = len(paths[0][0]), len(paths)
    x, y = np.stack([p[0] for p in paths], 1).astype(float), np.stack([p[1] for p in paths], 1).astype(float)
    ok = np.isfinite(x)
    sides = sides or ["home" if s % 2 == 0 else "away" for s in range(S)]
    return RawCache(match_id=999, fps=FPS, tracking_path="synthetic", quarantine_periods=(),
                    frame_num=np.arange(F, dtype=np.int64), video_time_s=T0 + np.arange(F) / FPS, source_period=np.ones(F, np.int16),
                    canonical_period=np.ones(F, np.int8), record_first=np.arange(F, dtype=np.int64), record_count=np.ones(F, np.int16),
                    frame_gap_before=np.full(F, -1, np.int32), frame_qflags=np.zeros(F, np.int64),
                    slot_side=np.array(sides, "<U4"), slot_jersey=np.arange(1, S + 1, dtype=np.int16), slot_player_id=np.arange(S, dtype=np.int64) + 987650,
                    slot_team_id=np.array([1 if s == "home" else 2 for s in sides], np.int64), x=np.where(ok, x, np.nan), y=np.where(ok, y, np.nan),
                    state=np.where(ok, STATE_AVAILABLE, STATE_ABSENT).astype(np.int8), visibility=np.where(ok, 1, 0).astype(np.int8),
                    confidence=np.where(ok, 3, 0).astype(np.int8), attribution=np.where(ok, 1, 0).astype(np.int8), qflags=np.zeros((F, S), np.int64))


K = np.arange(101)                         # t = 100.0 ... 110.0 s


def run(c: RawCache):
    qr = Q.assess(c, [])
    events, stats = detect(c, qr.mask, qr.steps, DetectConfig())
    return events, stats, qr.mask


def x_crossing(**drop):
    """A along y = 0.37 crosses x = 0.23 at t = 105.23 s; B along x = 0.23 crosses y = 0.37 at t = 104.63 s (no shared sample)."""
    a = (-5 + 0.1 * K, np.full(len(K), 0.37))
    b = (np.full(len(K), 0.23), 5 - 0.1 * K)
    paths = [list(map(np.array, a)), list(map(np.array, b))]
    for slot, (lo, hi) in drop.items():
        s = int(slot[-1])
        gone = (T0 + K / FPS >= lo - 1e-9) & (T0 + K / FPS <= hi + 1e-9)
        paths[s][0][gone] = paths[s][1][gone] = np.nan
    return cache([tuple(p) for p in paths])


def test_detect_single_crossing_deduplicated_across_windows():
    events, stats, _ = run(x_crossing())
    assert stats["windows"] == 6                                          # (10 s - 5 s) / 1 s + 1 full windows
    assert stats["pair_windows_with_hit"] == 4 and stats["segment_pair_hits_per_window"] == 4
    assert len(events) == 1
    e = events[0]
    assert e["type"] == "proper_crossing" and e["n_windows"] == 4 and e["windows"] == [103.5, 104.5, 105.5, 106.5]
    assert (e["x"], e["y"]) == pytest.approx((0.23, 0.37))
    assert e["time_a_s"] == pytest.approx(105.23) and e["time_b_s"] == pytest.approx(104.63)
    assert e["abs_time_diff_s"] == pytest.approx(0.6) and not e["simultaneous"]     # reported, not filtered
    assert e["team_relation"] == "opposite_teams" and e["cluster_size"] == 1


def test_detect_non_crossing():
    a = (-5 + 0.1 * K, np.full(len(K), 0.0))
    b = (-5 + 0.1 * K, np.full(len(K), 1.5))                             # parallel, 1.5 m apart the whole time
    events, stats, _ = run(cache([a, b]))
    assert events == [] and stats["pair_windows_with_hit"] == 0


def test_missing_data_gap_is_not_joined():
    events, stats, _ = run(x_crossing(slot0=(104.8, 105.8)))             # A unseen for 1.1 s over its crossing
    assert events == [] and stats["segments"]["invalid_gap_or_period"] == 1
    events, _, _ = run(x_crossing(slot0=(105.1, 105.4)))                 # 0.5 s gap: within the Step 1 continuity bound
    assert len(events) == 1


def test_separate_crossings_of_one_pair_are_not_merged():
    ax = np.r_[-3 + 0.12 * K[:51], 3 - 0.12 * (K[51:] - 50)]           # out and back along y = 0.37
    a = (ax, np.full(len(K), 0.37))
    b = (np.full(len(K), 0.23), 3 - 0.06 * K)                           # slow, one segment carries both contacts
    events, _, _ = run(cache([a, b]))
    assert len(events) == 2
    assert [e["time_a_s"] for e in events] == pytest.approx([102.69167, 107.30833], abs=1e-4)
    assert {e["segment_b"]["row0"] for e in events} == {43}             # same segment of B, two separate events
    assert len({e["event_id"] for e in events}) == 2
    assert {e["cluster_id"] for e in events} == {events[0]["cluster_id"]} and events[0]["cluster_size"] == 2


def test_vertex_crossing_merges_the_two_endpoint_hits():
    a = (-5 + 0.1 * K, np.zeros(len(K)))                                # A has a vertex exactly at (0, 0)
    b = (np.zeros(len(K)), 3.05 - 0.1 * K)                              # B passes (0, 0) between samples
    events, stats, _ = run(cache([a, b]))
    assert stats["unique_segment_pair_hits"] == 2 and len(events) == 1
    assert events[0]["type"] == "vertex_crossing" and events[0]["member_kinds"] == ["endpoint"]


def test_touch():
    ax = -5 + 0.1 * K
    a = (ax, np.abs(ax))                                                  # V shape touching y = 0 at x = 0
    b = (-3.05 + 0.1 * K, np.zeros(len(K)))
    events, _, _ = run(cache([a, b]))
    assert len(events) == 1 and events[0]["type"] == "touch"


def test_duplicate_positions_are_excluded():
    a = (-5 + 0.1 * K, np.full(len(K), 0.37))
    b = (np.full(len(K), 0.23), 5 - 0.1 * K)
    c = cache([a, a, b])                                                  # slots 0 and 1 bit-identical: provider artifact
    dup = duplicate_positions(c)
    assert dup[:, :2].all() and not dup[:, 2].any()
    events, stats, _ = run(c)
    assert events == [] and stats["duplicate_position_samples"] == 2 * len(K) and stats["segments"]["duplicate_position"] == 200


def test_event_ids_are_deterministic():
    e1, _, _ = run(x_crossing())
    e2, _, _ = run(x_crossing())
    assert e1 == e2


# ---------------------------------------------------------------- four-segment samples

def sample(c, cfg=SampleConfig()):
    events, _, mask = run(c)
    assert len(events) == 1
    return build_sample(c, mask, events[0], Timeline(c), duplicate_positions(c), cfg)


def test_sample_schema_label_and_equal_spans():
    c = x_crossing()
    seen_targets, seen_query = set(), set()
    for k in range(24):
        inst, ref, excl = sample(c, SampleConfig(seed=f"s{k}"))
        assert excl is None
        assert tuple(sorted(inst)) == tuple(sorted(INSTANCE_KEYS))
        assert all(tuple(sorted(tr)) == tuple(sorted(TRACKLET_KEYS)) for tr in inst["incoming"] + inst["outgoing"])
        i0, i1 = inst["incoming"]
        o0, o1 = inst["outgoing"]
        assert i0["t"] == i1["t"] and o0["t"] == o1["t"]                   # identical time stamps: order and length carry no label
        assert max(i0["t"]) < 0 and min(o0["t"]) > inst["ambiguity_interval_s"][1]
        q = ref["incoming"]["in-0"]["slot"]
        tgt = inst["outgoing"][ref["target"]]
        rows = np.searchsorted(c.video_time_s, np.asarray(tgt["t"]) + ref["withheld_video_time_s"][0] - 1e-6)
        assert np.allclose(tgt["x"], c.x[rows, q]) and np.allclose(tgt["y"], c.y[rows, q])
        assert ref["outgoing"][f"out-{ref['target']}"]["slot"] == q
        seen_targets.add(ref["target"])
        seen_query.add(q)
    assert seen_targets == {0, 1} and seen_query == {0, 1}


def test_instance_carries_no_identity_or_event_metadata():
    inst, ref, _ = sample(x_crossing())
    text = json.dumps(inst)
    for token in ("home", "away", "slot", "jersey", "player", "team", "event", "proper", "time_diff", "98765"):
        assert token not in text
    assert inst["instance_id"] == ref["instance_id"] and inst["instance_id"] != ref["event_id"]


def test_label_bits_are_balanced():
    bits = np.array([_bits("m11-intersections-v1", f"key-{k}") for k in range(4000)])
    assert 0.47 < bits[:, 0].mean() < 0.53 and 0.47 < bits[:, 1].mean() < 0.53
    target = bits[:, 0] == bits[:, 1]                                     # derived as in build_sample
    assert 0.47 < target.mean() < 0.53


def test_exclusion_large_time_difference():
    a = (-3 + 0.1 * K, np.full(len(K), 0.37))                            # A crosses at 103.23 s
    b = (np.full(len(K), 0.23), 7.1 - 0.1 * K)                          # B at 106.73 s: |dt| = 3.5 s
    inst, ref, excl = sample(cache([a, b]))
    assert inst is None and excl["reason"] == "segment_span_too_short"


def test_exclusion_period_boundary():
    k = np.arange(61)                                                     # 6 s period: the 5 s span around 104.9 s leaves it
    c = cache([(-5 + 0.1 * k, np.full(61, 0.37)), (np.full(61, 0.23), 5 - 0.1 * k)])
    qr = Q.assess(c, [])
    events, _ = detect(c, qr.mask, qr.steps, DetectConfig())
    _, _, excl = build_sample(c, qr.mask, events[0], Timeline(c), duplicate_positions(c))
    assert excl["reason"] == "period_boundary"


def test_exclusion_outgoing_coverage_and_attribution():
    _, _, excl = sample(x_crossing(slot1=(106.0, 106.5)))
    assert excl["reason"] == "low_coverage" and excl["segment"] == "outgoing"
    c = x_crossing()
    c.attribution[30, 0] = 3                                              # unresolved jersey inside the incoming span
    _, _, excl = sample(c)
    assert excl["reason"] == "attribution_unresolved_or_ambiguous"


def test_build_samples_orders_by_opaque_id():
    c = x_crossing()
    events, _, mask = run(c)
    insts, refs, excl = build_samples(c, mask, events * 1)
    assert [i["instance_id"] for i in insts] == sorted(i["instance_id"] for i in insts) and excl == []


# ---------------------------------------------------------------- outputs

def test_write_dir_never_overwrites(tmp_path):
    out = tmp_path / "m" / "3812"
    write_dir(out, {"root.a.json": b"1"})
    with pytest.raises(IngestionError):
        write_dir(out, {"root.a.json": b"2"})
    assert (out / "root.a.json").read_bytes() == b"1"
    assert [p.name for p in out.parent.iterdir()] == ["3812"]             # no partial directory left behind


ROOT = Path(__file__).resolve().parents[1] / "data"
OUT = ROOT / "processed" / "m11-intersections-v1" / "3812"


def test_cli_refuses_non_development_matches():
    split_path = ROOT / "processed" / "m11-encounters-v1" / "splits.json"
    if not split_path.exists():
        pytest.skip("split manifest not present")
    from football_fatigue.intersections.__main__ import run as cli_run

    matches = json.loads(split_path.read_text(encoding="utf-8"))["matches"]
    for split in ("test", "validation"):
        mid = next(m for m, v in matches.items() if v["split"] == split)
        with pytest.raises(IngestionError):
            cli_run(str(ROOT / "raw"), mid, True, DetectConfig(), SampleConfig())


@pytest.mark.skipif(not (OUT / "root.manifest.json").exists(), reason="3812 prototype outputs not written")
def test_written_prototype_outputs():
    m = json.loads((OUT / "root.manifest.json").read_text(encoding="utf-8"))
    for name, digest in m["outputs_sha256"].items():
        assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == digest
    assert m["split"]["split"] == "development" and m["raw_unchanged"]
    read = lambda n: [json.loads(line) for line in (OUT / n).read_text(encoding="utf-8").splitlines()]  # noqa: E731
    insts, refs = read("root.instances.jsonl"), read("root.references.jsonl")
    assert [i["instance_id"] for i in insts] == [r["instance_id"] for r in refs]
    assert all(set(i) == set(INSTANCE_KEYS) for i in insts)
    assert all(i["incoming"][0]["t"] == i["incoming"][1]["t"] and i["outgoing"][0]["t"] == i["outgoing"][1]["t"] for i in insts)
    y = np.array([r["target"] for r in refs])
    assert 0.45 < y.mean() < 0.55
