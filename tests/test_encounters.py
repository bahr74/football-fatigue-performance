"""Tests for football_fatigue.encounters (M11 step 1). All data is SYNTHETIC (30 fps) and only mimics the shape of the real files."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from football_fatigue.encounters import M11Config, assess, assign, build_cache, find_encounters, run_match
from football_fatigue.encounters import quality as Qm
from football_fatigue.encounters.catalogue import output_paths, run_from_cache, summarize
from football_fatigue.encounters.splits import FIXED_DEVELOPMENT, RELEASE_DISAGREEMENT, write_manifest
from football_fatigue.encounters.trajectories import STATE_MISSING, load_cache
from football_fatigue.ingestion import EventVersion, IngestionError
from football_fatigue.match_data import CoordinatePolicy, MatchData

_spec = importlib.util.spec_from_file_location("ingestion_test_helpers", Path(__file__).with_name("test_ingestion.py"))
H = importlib.util.module_from_spec(_spec)
sys.modules["ingestion_test_helpers"] = H
_spec.loader.exec_module(H)

FPS = 30.0


def roster(home=(1, 2, 3), away=(1, 2, 3)):
    return ([dict(player=dict(id=str(1000 + j), nickname=f"H{j}"), positionGroupType="CM", shirtNumber=str(j), started=j <= 11,
                  team=dict(id="10", name="Home")) for j in home]
            + [dict(player=dict(id=str(2000 + j), nickname=f"A{j}"), positionGroupType="CM", shirtNumber=str(j), started=True,
                    team=dict(id="20", name="Away")) for j in away])


def P(j, x, y, vis="VISIBLE", conf="HIGH"):
    return dict(jerseyNum=str(j), confidence=conf, visibility=vis, x=x, y=y)


def mv(j, x0, y0, vx=1.0, vy=0.0, **kw):
    return lambda i: P(j, x0 + vx * i / FPS, y0 + vy * i / FPS, **kw)


def build(tmp_path, n, home, away, *, period=lambda i: 1, skip=(), events=None, roster_=None, extra=None):
    recs = []
    for i in range(n):
        if i in skip:
            continue
        h = [p for f in home if (p := f(i)) is not None]
        a = [p for f in away if (p := f(i)) is not None]
        recs.append(H.rec(i, period=period(i), home=h, away=a))
        if extra and i in extra:
            recs.append(H.rec(i, period=period(i), home=h, away=a))      # a repeated source record of the same frame
    return H.build(tmp_path, recs, events=events, roster=roster_ or roster())


def analyse(raw, cfg=M11Config()):
    md = MatchData.open(raw, "100", EventVersion.ROOT, policy=CoordinatePolicy.RAW)
    c = build_cache(md)
    qr = assess(c, md.events(), cfg)
    return c, qr, find_encounters(c, qr, cfg)


def slot(c, side, j):
    return int(np.flatnonzero((c.slot_side == side) & (c.slot_jersey == j))[0])


FAR = mv(3, -40.0, 30.0)   # a background player far from everyone


# ---------------------------------------------------------------- cadence, stationary players, raw preservation

def test_stationary_player_is_valid_and_flagged_only_as_held(tmp_path):
    def stopper(i):                       # runs at 2 m/s, stands exactly still for 2 s, runs again
        x = 2 * min(i, 100) / FPS + 2 * max(0, i - 160) / FPS
        return P(1, x, 0.0)
    raw = build(tmp_path, 300, [stopper], [FAR])
    c, qr, _ = analyse(raw)
    s = slot(c, "home", 1)
    held = (qr.mask[:, s] & Qm.HELD_POSITION) != 0
    assert held[101:161].all() and not held[:100].any()
    assert not (qr.mask[:, s] & (Qm.IMPLAUSIBLE_JUMP | Qm.NO_MOTION_CONTEXT)).any()
    cat = Qm.sample_categories(c, qr.mask)[:, s]
    assert (cat == Qm.SAMPLE_CATEGORIES.index("valid_raw")).all()
    assert np.nanmax(qr.speed[:, s]) <= 2.0 + 1e-9


def test_repeated_source_records_are_not_repeated_positions(tmp_path):
    cadence = lambda i: P(1, 0.1 * (i // 2), 0.0)                       # noqa: E731  15 Hz updates in a 30 fps file
    raw = build(tmp_path, 200, [cadence], [FAR], extra={50})
    c, qr, _ = analyse(raw)
    s = slot(c, "home", 1)
    assert c.n_frames == 200 and c.record_count[50] == 2 and (c.record_count == 1).sum() == 199
    rep = (qr.mask[:, s] & Qm.REPEAT_OF_PREVIOUS) != 0
    assert rep.sum() == 100 and rep[1::2].all() and not rep[0::2].any()
    assert not ((qr.mask[:, s] & Qm.HELD_POSITION) != 0).any()
    st = qr.steps
    assert np.median(st["dt_s"][st["valid"] & (st["slot"] == s)]) == pytest.approx(2 / FPS)    # real update interval, not 1/fps
    assert np.nanmedian(qr.speed[:, s]) == pytest.approx(1.5)


def test_raw_values_preserved_missing_and_estimated_flagged_smoothed_ignored(tmp_path):
    def p2(i):
        if 100 <= i < 110:
            return P(2, None, None)
        return P(2, 5.0 + i / FPS, 1.0, vis="ESTIMATED" if i >= 200 else "VISIBLE", conf="LOW" if i >= 200 else "HIGH")
    recs = [H.rec(i, home=[mv(1, 0.0, 0.0)(i), p2(i)], away=[FAR(i)], hs=[P(1, 99.0, 99.0), P(2, 99.0, 99.0)]) for i in range(300)]
    raw = H.build(tmp_path, recs, roster=roster())
    c, qr, _ = analyse(raw)
    s1, s2 = slot(c, "home", 1), slot(c, "home", 2)
    assert np.array_equal(c.x[:, s1], np.array([i / FPS for i in range(300)]))       # raw floats, never the smoothed 99.0
    assert (c.state[100:110, s2] == STATE_MISSING).all() and np.isnan(c.x[100:110, s2]).all()
    assert ((qr.mask[100:110, s2] & Qm.MISSING_RAW) != 0).all()
    est = (qr.mask[:, s2] & Qm.ESTIMATED) != 0
    assert est[200:].all() and not est[:200].any() and ((qr.mask[200:, s2] & Qm.LOW_CONFIDENCE) != 0).all()
    assert c.x[250, s2] == 5.0 + 250 / FPS


# ---------------------------------------------------------------- jumps and identity discontinuities

def test_extreme_jump_is_preserved_flagged_and_kept_out_of_motion_statistics(tmp_path):
    jumper = lambda i: P(2, i / FPS + (30.0 if i >= 150 else 0.0), 0.0)  # noqa: E731
    raw = build(tmp_path, 300, [jumper], [FAR])
    c, qr, _ = analyse(raw)
    s = slot(c, "home", 2)
    assert c.x[150, s] == 150 / FPS + 30.0                                     # the landing value is kept as given
    assert (qr.mask[150, s] & Qm.IMPLAUSIBLE_JUMP) and not (qr.mask[150, s] & Qm.LABEL_TRANSFER_CANDIDATE)
    (j,) = qr.jumps
    assert j["distance_m"] == pytest.approx(30.0 + 1 / FPS) and j["frame_to"] == 150 and not j["label_transfer_candidate"]
    summary = summarize(c, qr, [], M11Config())
    assert summary["motion"]["valid_speed_m_s"]["max"] == pytest.approx(1.0)
    assert summary["motion"]["abs_accel_m_s2"]["max"] == pytest.approx(0.0, abs=1e-6)


def test_label_exchange_is_preserved_as_suspicious_not_corrected(tmp_path):
    def lane(j, y_before, y_after):
        return lambda i: P(j, 2 * i / FPS, y_before if i < 150 else y_after)
    raw = build(tmp_path, 300, [lane(1, 1.9, -1.9), lane(2, -1.9, 1.9)], [FAR])
    c, qr, enc = analyse(raw)
    s1, s2 = slot(c, "home", 1), slot(c, "home", 2)
    assert c.y[149, s1] == 1.9 and c.y[150, s1] == -1.9                         # labels exactly as the provider gave them
    for s in (s1, s2):
        assert qr.mask[150, s] & Qm.RECIPROCAL_EXCHANGE and qr.mask[150, s] & Qm.LABEL_TRANSFER_CANDIDATE
    assert len(qr.jumps) == 2 and all(j["reciprocal_exchange"] for j in qr.jumps)
    e4 = [e for e in enc if e["tau_m"] == 4.0 and {p["slot"] for p in e["pair"]} == {s1, s2}]
    assert e4 and all(e["category"] == "suspicious_identity" and e["possible_identity_discontinuity"] for e in e4)
    assert any(set(e["discontinuity_evidence"]) == {j["jump_id"] for j in qr.jumps} for e in e4)
    assert "confirmed" not in json.dumps(enc)


def test_close_reciprocal_exchange_below_the_transfer_distance_is_a_discontinuity_candidate(tmp_path):
    def lane(j, y_before, y_after):
        return lambda i: P(j, 2 * i / FPS, y_before if i < 150 else y_after)
    c, qr, enc = analyse(build(tmp_path, 300, [lane(1, 0.6, -0.6), lane(2, -0.6, 0.6)], [FAR]))
    assert len(qr.jumps) == 2 and all(j["reciprocal_exchange"] and j["discontinuity_candidate"] and not j["label_transfer_candidate"] for j in qr.jumps)
    e = [e for e in enc if e["tau_m"] == 2.0 and {p["slot"] for p in e["pair"]} == {slot(c, "home", 1), slot(c, "home", 2)}]
    assert e and all(x["category"] == "suspicious_identity" for x in e)


def test_frame_wide_shift_and_sub_metre_overspeed_are_kinematic_flags_not_identity_evidence(tmp_path):
    shift = lambda i: 2.0 if i >= 150 else 0.0                                 # noqa: E731  everyone moves 2 m in one frame
    home = [lambda i, j=j: P(j, -20.0 + 0.5 * j + i / FPS + shift(i), 0.6 * (j % 2)) for j in (1, 2, 3)]   # a close group
    away = [lambda i, j=j: P(j, -20.0 + 5 * j + i / FPS + shift(i), 30.0) for j in (1, 2, 3)]
    wobble = lambda i: P(4, 10.0 + i / FPS + (0.9 if i == 200 else 0.0), -30.0)  # noqa: E731  one 0.9 m spike in one frame
    c, qr, enc = analyse(build(tmp_path, 300, home + [wobble], away, roster_=roster(home=(1, 2, 3, 4))))
    fw = [j for j in qr.jumps if j["frame_wide_shift"]]
    assert len(fw) == 6 and not any(j["discontinuity_candidate"] for j in qr.jumps)
    s4 = slot(c, "home", 4)
    assert qr.mask[200, s4] & Qm.OVERSPEED_STEP and not qr.mask[200, s4] & Qm.IMPLAUSIBLE_JUMP
    assert c.x[200, s4] == 10.0 + 200 / FPS + 0.9                               # the spike value is kept
    summary = summarize(c, qr, enc, M11Config())
    assert summary["motion"]["valid_speed_m_s"]["max"] <= 12.0 and summary["motion"]["overspeed_steps"] >= 1
    assert summary["jumps"]["frame_wide_shift_frames"] == 1
    assert enc and not any(e["category"] in ("suspicious_identity", "trusted_reference_candidate") for e in enc if "FRAME_WIDE_SHIFT" in e["flags"])
    json.dumps(qr.jumps), json.dumps(enc), json.dumps(summary)               # every record is plain JSON (no numpy scalars)


# ---------------------------------------------------------------- encounter extraction

def _pair(c, enc, tau):
    return [e for e in enc if e["tau_m"] == tau and {p["slot"] for p in e["pair"]} == {slot(c, "home", 1), slot(c, "home", 2)}]


def _meet(close_frames):
    """Home #1 runs along y=0; home #2 runs alongside at 0.5 m during ``close_frames`` and drifts away at 6 m/s otherwise."""
    cf = np.array(sorted(close_frames))
    a = mv(1, -10.0, 0.0)
    b = lambda i: P(2, -10.0 + i / FPS, 0.5 + min(20.0, 0.2 * float(np.abs(cf - i).min())))  # noqa: E731
    return a, b


def test_no_encounter_across_period_boundary_or_frame_gap(tmp_path):
    close = set(range(120, 168))
    a, b = _meet(close)
    c, _, enc = analyse(build(tmp_path / "control", 300, [a, b], [FAR]))
    assert len(_pair(c, enc, 1.0)) == 1                                        # 1.6 s contiguous: one encounter
    c, _, enc = analyse(build(tmp_path / "period", 300, [a, b], [FAR], period=lambda i: 1 if i < 144 else 2))
    assert _pair(c, enc, 1.0) == []                                            # 0.8 s on each side of the boundary
    c, _, enc = analyse(build(tmp_path / "gap", 330, [a, _meet(set(range(120, 144)) | set(range(174, 198)))[1]], [FAR], skip=set(range(144, 174))))
    assert c.n_frames == 300 and _pair(c, enc, 1.0) == []                      # 0.8 s, a 1 s frame gap, 0.8 s


def test_trusted_ordinary_and_third_player(tmp_path):
    a, b = _meet(set(range(120, 165)))
    c, _, enc = analyse(build(tmp_path / "trusted", 300, [a, b], [FAR]))
    (e,) = _pair(c, enc, 1.0)
    assert e["category"] == "trusted_reference_candidate" and e["same_team"] and e["third_players"] == []
    assert e["start_frame"] == 118 and e["end_frame"] == 166 and e["min_distance_m"] == 0.5     # within 1 m two frames either side
    assert e["rows"]["before"][1] == e["rows"]["during"][0] and e["provenance"]["source_records"][0] <= 120
    est = lambda i: dict(b(i), visibility="ESTIMATED")                         # noqa: E731
    c, _, enc = analyse(build(tmp_path / "estimated", 300, [a, est], [FAR]))
    (e,) = _pair(c, enc, 1.0)
    assert e["category"] == "ordinary" and e["players"][1]["estimated_share_during"] == 1.0
    third = lambda i: P(1, -10.0 + i / FPS, -0.4 if 120 <= i < 165 else -25.0)  # noqa: E731
    c, _, enc = analyse(build(tmp_path / "third", 300, [a, b], [third, FAR]))
    (e,) = _pair(c, enc, 1.0)
    assert e["category"] == "ordinary" and "THIRD_PLAYER_NEARBY" in e["flags"]
    assert [(t["side"], t["jersey"]) for t in e["third_players"]] == [("away", 1)]


def test_substitution_overlap_is_flagged_and_unresolved(tmp_path):
    home = [mv(j, -30.0 + 5 * j, 20.0, vx=0.3) for j in range(1, 12)]
    home.append(lambda i: P(12, i / FPS + 0.5, -20.0) if 100 <= i <= 130 else None)    # on-player overlaps the off-player
    away = [mv(1, 0.0, -20.0)]
    sub = H.event(9, None, 110 / FPS, gtype="SUB", ptype=None,
                  gameEvents=dict(gameEventType="SUB", period=1, teamId=None, playerId=None, homeTeam=None, playerOffId=1011, playerOnId=1012))
    raw = build(tmp_path, 300, home, away, events=[sub], roster_=roster(home=range(1, 13), away=(1,)))
    c, qr, enc = analyse(raw)
    s11, s12 = slot(c, "home", 11), slot(c, "home", 12)
    over = (qr.mask[:, s11] & Qm.SIDE_OVERCOUNT) != 0
    assert over[100:131].all() and not over[:100].any() and not over[131:].any()
    assert ((qr.mask[100:131, s12] & Qm.SUBSTITUTION_WINDOW) != 0).all()
    assert ((qr.mask[50:276, s11] & Qm.SUBSTITUTION_WINDOW) != 0).sum() == len(range(50, 261))    # 2 s before .. 5 s after
    e = [e for e in enc if s12 in {p["slot"] for p in e["pair"]}]
    assert e and all(x["category"] == "unresolved" and "SUBSTITUTION_WINDOW" in x["flags"] for x in e)


# ---------------------------------------------------------------- outputs, determinism, splits

def test_outputs_are_deterministic_safe_and_never_overwritten(tmp_path):
    a, b = _meet(set(range(120, 165)))
    raw = build(tmp_path, 300, [a, b], [FAR])
    before = H.digest(raw)
    out1, out2 = tmp_path / "processed" / "m11a", tmp_path / "processed" / "m11b"
    p1, s1 = run_match(raw, "100", out_root=out1)
    p2, _ = run_match(raw, "100", out_root=out2)
    for sfx in (".encounters.jsonl", ".jumps.jsonl", ".summary.json"):
        assert p1[sfx].read_bytes() == p2[sfx].read_bytes()
    z1, z2 = np.load(p1[".cache.npz"]), np.load(p2[".cache.npz"])
    assert z1.files == z2.files and all(np.array_equal(z1[k], z2[k], equal_nan=z1[k].dtype.kind == "f") for k in z1.files)
    back = load_cache(p1[".cache.npz"])
    md = MatchData.open(raw, "100", EventVersion.ROOT, policy=CoordinatePolicy.RAW)
    assert find_encounters(back, assess(back, md.events())) == [json.loads(line) for line in p1[".encounters.jsonl"].open()]
    assert p1[".encounters.jsonl"].name == "root.encounters.jsonl" and p1[".encounters.jsonl"].parent == out1 / "100"
    m = json.loads(p1[".manifest.json"].read_text())
    assert m["provider_smoothed_coordinates_used"] is False and m["coordinate_policy"] == "raw"
    assert set(m["outputs_sha256"]) == {p.name for k, p in p1.items() if k != ".manifest.json"}
    assert s1["encounters"]["1m"]["count"] == 1
    with pytest.raises(IngestionError, match="never overwritten"):
        run_match(raw, "100", out_root=out1)
    with pytest.raises(IngestionError, match="approved output root"):
        output_paths(raw, "100", raw / "inside")
    with pytest.raises(IngestionError, match="approved output root"):
        output_paths(raw, "100", tmp_path / "elsewhere")
    assert H.digest(raw) == before
    assert not list((out1 / "100").glob("*.tmp"))


def test_recompute_from_verified_cache_matches_a_raw_run_and_rejects_bad_caches(tmp_path):
    import hashlib
    import shutil

    a, b = _meet(set(range(120, 165)))
    raw = build(tmp_path, 300, [a, b], [FAR])
    proc = tmp_path / "processed"
    p1, _ = run_match(raw, "100", out_root=proc / "v_raw")
    p2, _ = run_from_cache(raw, p1[".cache.npz"], out_root=proc / "v_cache", expected_match_id="100")
    for sfx in (".encounters.jsonl", ".jumps.jsonl", ".summary.json"):
        assert p1[sfx].read_bytes() == p2[sfx].read_bytes()
    m2 = json.loads(p2[".manifest.json"].read_text())
    assert m2["derived_from_cache"]["sha256"] == hashlib.sha256(p1[".cache.npz"].read_bytes()).hexdigest()
    assert m2["processing"]["source"] == "cache"
    with pytest.raises(IngestionError, match="belongs to match"):
        run_from_cache(raw, p1[".cache.npz"], out_root=proc / "v_wrong", expected_match_id="101")
    # an arrays-only cache (the v1 format) is accepted once its manifest checksum matches
    legacy = proc / "legacy" / "100"
    legacy.mkdir(parents=True)
    with np.load(p1[".cache.npz"]) as z:
        np.savez_compressed(legacy / "root.cache.npz", **{k: z[k] for k in z.files if z[k].ndim > 0})
    for sfx in (".summary.json", ".manifest.json"):
        shutil.copy(p1[sfx], legacy / f"root{sfx}")
    m = json.loads((legacy / "root.manifest.json").read_text())
    m["outputs_sha256"]["root.cache.npz"] = hashlib.sha256((legacy / "root.cache.npz").read_bytes()).hexdigest()
    (legacy / "root.manifest.json").write_text(json.dumps(m))
    p3, _ = run_from_cache(raw, legacy / "root.cache.npz", out_root=proc / "v_legacy")
    assert p3[".encounters.jsonl"].read_bytes() == p1[".encounters.jsonl"].read_bytes()
    with open(legacy / "root.cache.npz", "ab") as f:                          # tampered: checksum no longer matches
        f.write(b"\0")
    with pytest.raises(IngestionError, match="checksum"):
        run_from_cache(raw, legacy / "root.cache.npz", out_root=proc / "v_tampered")
    assert not (proc / "v_tampered").exists() and not (proc / "v_wrong").exists()


def _ids():
    named = set(FIXED_DEVELOPMENT) | set(RELEASE_DISAGREEMENT) | {"10510", "10511", "10517"}
    return sorted(named | {str(5000 + i) for i in range(64 - len(named))})


def test_split_manifest_is_deterministic_disjoint_and_frozen(tmp_path):
    ids = _ids()
    m = assign(ids)
    assert m == assign(list(reversed(ids)))
    sp = m["splits"]
    assert [len(sp[k]) for k in ("development", "validation", "test")] == [40, 12, 12]
    assert set(sp["development"]) | set(sp["validation"]) | set(sp["test"]) == set(ids)
    assert len(set(sp["development"]) & set(sp["validation"])) == len(set(sp["development"]) & set(sp["test"])) == 0
    assert set(FIXED_DEVELOPMENT) <= set(sp["development"])
    assert sum(v["release_disagreement"] for v in m["matches"].values()) == 10 and m["matches"]["3816"]["split"] == "development"
    raw = tmp_path / "raw"
    raw.mkdir()
    out = tmp_path / "processed" / "m11"
    p = write_manifest(raw, out, m)
    assert write_manifest(raw, out, m) == p                                     # an identical manifest is accepted
    with pytest.raises(IngestionError, match="never replaced"):
        write_manifest(raw, out, dict(m, seed="other"))
    with pytest.raises(IngestionError):
        assign(ids[:-1])
