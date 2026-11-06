"""Tests for football_fatigue.matching (M11 step 2). All data is SYNTHETIC (30 fps) and only mimics the shape of the real files."""

import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from football_fatigue.encounters import M11Config, assess, build_cache, find_encounters
from football_fatigue.encounters import run_match as run_catalogue
from football_fatigue.ingestion import EventVersion, IngestionError
from football_fatigue.match_data import CoordinatePolicy, MatchData
from football_fatigue.matching import BASELINES, BaselineConfig, MatchingConfig, build_benchmark, evaluate, run_baselines, run_match
from football_fatigue.matching import baselines as B
from football_fatigue.matching import footballtrace as FT
from football_fatigue.matching.benchmark import DURING_KEYS, INSTANCE_KEYS, TRACKLET_KEYS, shuffle_order, withhold_interval
from football_fatigue.matching.footballtrace import protocol_l
from football_fatigue.matching.evaluation import metrics, outcome

_spec = importlib.util.spec_from_file_location("ingestion_test_helpers", Path(__file__).with_name("test_ingestion.py"))
H = importlib.util.module_from_spec(_spec)
sys.modules["ingestion_test_helpers"] = H
_spec.loader.exec_module(H)

FPS = 30.0
BC = BaselineConfig()


def roster(home=(1, 2, 3), away=(1, 2, 3)):
    return ([dict(player=dict(id=str(1000 + j), nickname=f"H{j}"), positionGroupType="CM", shirtNumber=str(j), started=j <= 11,
                  team=dict(id="10", name="Home")) for j in home]
            + [dict(player=dict(id=str(2000 + j), nickname=f"A{j}"), positionGroupType="CM", shirtNumber=str(j), started=True,
                    team=dict(id="20", name="Away")) for j in away])


def P(j, x, y, vis="VISIBLE", conf="HIGH"):
    return dict(jerseyNum=str(j), confidence=conf, visibility=vis, x=x, y=y)


def build(tmp_path, n, home, away, *, period=lambda i: 1, events=None, roster_=None):
    recs = []
    for i in range(n):
        h = [p for f in home if (p := f(i)) is not None]
        a = [p for f in away if (p := f(i)) is not None]
        recs.append(H.rec(i, period=period(i), home=h, away=a))
    return H.build(tmp_path, recs, events=events, roster=roster_ or roster())


def bench(raw, cfg=MatchingConfig()):
    md = MatchData.open(raw, "100", EventVersion.ROOT, policy=CoordinatePolicy.RAW)
    c = build_cache(md)
    qr = assess(c, md.events(), M11Config())
    enc = find_encounters(c, qr, M11Config())
    rows, inst, refs = build_benchmark(c, qr.mask, enc, M11Config(), cfg)
    return c, qr, enc, rows, inst, refs


def slot(c, side, j):
    return int(np.flatnonzero((c.slot_side == side) & (c.slot_jersey == j))[0])


def decisions(name, inst):
    return {d["incoming"]: d for d in B.run_baseline(name, inst, BC)}


def by_player(ref, side, jersey):
    (iid,) = [k for k, v in ref["incoming"].items() if (v["side"], v["jersey"]) == (side, jersey)]
    return iid


FAR = lambda i: P(3, -40.0 + i / FPS, 30.0)     # noqa: E731  a background player far from everyone


def crossing():
    """Home #1 and #2 cross in an X at frame 150 (both at the origin), 2.2 m/s each, 2 m apart 1 s either side."""
    a = lambda i: P(1, 2 * (i - 150) / FPS, 1 * (i - 150) / FPS)     # noqa: E731
    b = lambda i: P(2, 2 * (i - 150) / FPS, -1 * (i - 150) / FPS)    # noqa: E731
    return a, b


# ---------------------------------------------------------------- association tasks

def test_two_player_crossing_with_correct_and_swapped_anonymous_ordering(tmp_path):
    c, _, _, rows, inst, refs = bench(build(tmp_path, 300, list(crossing()), [FAR]))
    (row,) = [r for r in rows if r["accepted"]]
    (i,), (r,) = inst, refs
    assert r["tier"] == "high_confidence_reference" and r["difficulty"]["crossing"] and "crossing" in r["difficulty"]["reasons"]
    assert len(i["incoming"]) == len(i["outgoing"]) == 2 and r["team_relation"] == "same_team"
    a_in = by_player(r, "home", 1)
    a_out = [k for k, v in r["outgoing"].items() if v["jersey"] == 1][0]
    assert r["expected"][a_in] == a_out
    # constant velocity and the global assignment follow each player through the crossing
    for name in ("constant_velocity", "hungarian_cv"):
        d = decisions(name, i)
        assert all(d[k]["decision"] == "MATCHED" and d[k]["outgoing"] == r["expected"][k] for k in d)
    # nearest position prefers the wrong (mirrored) tracklet and is not credited with a correct match
    assert decisions("nearest", i)[a_in]["outgoing"] != a_out
    # both players sit exactly on the origin at frame 150: chaining cannot tell them apart and must abstain, not guess
    assert all(d["decision"] == "UNCERTAIN" for d in decisions("chain_during", i).values())
    # relabel the outgoing tracklets in the opposite order: the reference moves with them and the result is unchanged
    swap = {"out-0": "out-1", "out-1": "out-0"}
    i2 = dict(i, outgoing=[dict(tr, tracklet_id=swap[tr["tracklet_id"]]) for tr in reversed(i["outgoing"])])
    r2 = dict(r, expected={k: swap[v] for k, v in r["expected"].items()})
    res = evaluate(run_baselines([i2], BC), [r2])
    assert res["hungarian_cv"]["all"]["correct_match_rate"] == 1.0 and res["hungarian_cv"]["all"]["complete_assignment_accuracy"] == 1.0
    # a prediction that commits to the swapped assignment is counted as wrong against the reference
    wrong = [dict(baseline=n, instance_id=i["instance_id"], decisions=[dict(incoming=k, decision="MATCHED", outgoing=swap[v], candidate=swap[v])
                                                                       for k, v in r["expected"].items()]) for n in BASELINES]
    assert evaluate(wrong, [r])["nearest"]["all"]["wrong_match_rate"] == 1.0


def test_sharp_turn_without_identity_change_keeps_the_provider_reference(tmp_path):
    def a(i):                                     # comes from the left at 1 m/s, turns 90 degrees up at the meeting point
        d = (i - 150) / FPS
        return P(1, d, 0.5) if i < 150 else P(1, 0.0, 0.5 + d)

    def b(i):                                     # comes from the right, turns down
        d = (i - 150) / FPS
        return P(2, -d, -0.5) if i < 150 else P(2, 0.0, -0.5 - d)
    c, qr, _, rows, inst, refs = bench(build(tmp_path, 300, [a, b], [FAR]))
    (r,) = refs
    assert not qr.jumps and r["tier"] == "high_confidence_reference"
    assert r["difficulty"]["max_heading_change_deg"] == pytest.approx(90.0, abs=1.0) and "sharp_turn" in r["difficulty"]["reasons"]
    assert r["expected"][by_player(r, "home", 1)] == [k for k, v in r["outgoing"].items() if v["jersey"] == 1][0]
    d = decisions("chain_during", inst[0])                              # 1 m apart at the closest: chaining is unambiguous
    assert all(x["decision"] == "MATCHED" and x["outgoing"] == r["expected"][k] for k, x in d.items())


def test_stationary_player_with_held_samples_is_a_valid_participant(tmp_path):
    still = lambda i: P(1, 0.0, 0.0)                  # noqa: E731  never moves: one update instant, then held samples
    runner = lambda i: P(2, -10.0 + 2 * i / FPS, 0.5)  # noqa: E731  passes 0.5 m away at frame 150
    c, qr, _, rows, inst, refs = bench(build(tmp_path, 300, [still, runner], [FAR]))
    (r,) = refs
    assert r["tier"] == "high_confidence_reference" and not r["without_incoming"] and not r["without_outgoing"]
    assert r["difficulty"]["mean_speed_m_s"] is not None
    still_in = [tr for tr in inst[0]["incoming"] if tr["tracklet_id"] == by_player(r, "home", 1)][0]
    assert set(still_in["x"]) == {0.0} and B.tracklet_velocity(still_in, at_end=True, span_s=0.5, max_speed_m_s=12.0)[0].tolist() == [0.0, 0.0]
    for name in ("constant_velocity", "hungarian_cv", "chain_during"):
        assert all(x["decision"] == "MATCHED" and x["outgoing"] == r["expected"][k] for k, x in decisions(name, inst[0]).items())


def test_three_player_encounter_is_one_instance_not_three_pairs(tmp_path):
    def lane(j, y):
        return lambda i: P(j, -10.0 + 2 * i / FPS, y + (0.0 if 120 <= i < 180 else 0.15 * min(abs(i - 120), abs(i - 179)) * np.sign(y)))
    c, _, enc, rows, inst, refs = bench(build(tmp_path, 300, [lane(1, 0.0), lane(2, 0.8), lane(3, -0.8)], [FAR], roster_=roster(home=(1, 2, 3))))
    pairs = {tuple(sorted(p["slot"] for p in e["pair"])) for e in enc if e["tau_m"] == 2.0}
    assert len(pairs) == 3                                              # three pairwise encounters ...
    (r,) = refs                                                         # ... one event, one instance
    assert len([x for x in rows if x["n_participants"] >= 3]) == 1 and r["n_participants"] == 3 and r["team_relation"] == "multi_player"
    assert len(inst[0]["incoming"]) == len(inst[0]["outgoing"]) == 3 and "multi_player" in r["difficulty"]["reasons"]


def test_repeated_encounters_merge_or_never_share_observations(tmp_path):
    a = lambda i: P(1, -10.0 + 2 * i / FPS, 0.0)    # noqa: E731

    def meets(spans):                             # #2 runs alongside #1 at 0.5 m during the spans, else 3 m away (approach 0.1 m/frame)
        def b(i):
            d = min(0 if s <= i < e else min(abs(i - s), abs(i - e + 1)) for s, e in spans)
            return P(2, -10.0 + 2 * i / FPS, min(3.0, 0.5 + 0.1 * d))
        return b
    # (re-contact within about 1 s is already one step-1 encounter; overlapping encounters of several pairs merge: see the
    # three-player test.) Re-contact 2.5 s later: two events, but the second shares observations with the first instance
    _, _, enc, rows, _, refs = bench(build(tmp_path / "far", 400, [a, meets([(100, 140), (215, 255)])], [FAR]))
    assert len([e for e in enc if e["tau_m"] == 2.0]) == 2 and len(rows) == 2 and len(refs) == 1
    assert rows[0]["accepted"] and rows[1]["rejection_reasons"] == ["overlaps_earlier_instance"]


def test_missing_outgoing_tracklet_is_absent_not_forced(tmp_path):
    a = lambda i: P(1, -10.0 + 2 * i / FPS, 0.0)                                      # noqa: E731

    def b(i):                                     # converges on #1 to 0.5 m by frame 110; its raw position is lost from frame 172
        if i >= 172:
            return P(2, None, None)
        return P(2, -10.0 + 2 * i / FPS, 0.5 + 0.1 * max(0, 110 - i))
    c, qr, _, rows, inst, refs = bench(build(tmp_path, 300, [a, b], [FAR]))
    (r,) = refs
    b_in = by_player(r, "home", 2)
    assert r["expected"][b_in] is None and [x["jersey"] for x in r["without_outgoing"]] == [2]
    assert r["tier"] == "lower_confidence_reference" and "missing_incoming_or_outgoing" in r["below_high_reasons"]
    assert len(inst[0]["outgoing"]) == 1
    d = decisions("hungarian_cv", inst[0])
    assert d[b_in]["decision"] in ("UNMATCHED", "UNCERTAIN") and outcome(dict(decision="UNMATCHED", outgoing=None), None) == "correct_unmatched"
    assert d[by_player(r, "home", 1)]["outgoing"] == r["expected"][by_player(r, "home", 1)]


def test_estimated_and_low_confidence_observations_lower_the_tier_and_stay_visible(tmp_path):
    a, b = crossing()
    est = lambda i: dict(b(i), visibility="ESTIMATED", confidence="LOW")   # noqa: E731
    _, _, _, rows, inst, refs = bench(build(tmp_path, 300, [a, est], [FAR]))
    (r,) = refs
    assert r["tier"] == "lower_confidence_reference" and r["below_high_reasons"] == ["estimated_observations"]
    tr = [t for t in inst[0]["incoming"] if t["tracklet_id"] == by_player(r, "home", 2)][0]
    assert set(tr["visibility"]) == {2} and set(tr["confidence"]) == {1}


def test_period_boundary_cuts_the_after_window(tmp_path):
    a, b = crossing()
    _, _, _, rows, inst, refs = bench(build(tmp_path, 300, [a, b], [FAR], period=lambda i: 1 if i < 195 else 2))
    (r,) = refs
    assert "segment_edge" in r["below_high_reasons"] and r["tier"] == "lower_confidence_reference"
    t_end = (194 - r["rows"]["during"][0]) / FPS                        # last frame of period 1, relative to the interval start
    assert all(max(tr["t"]) <= t_end + 1e-3 for tr in inst[0]["outgoing"])


def test_substitution_overlap_is_rejected_with_a_reason(tmp_path):
    home = [lambda i, j=j: P(j, -30.0 + 5 * j + 0.3 * i / FPS, 20.0) for j in range(1, 12)]
    home.append(lambda i: P(12, i / FPS + 0.5, -20.0) if 100 <= i <= 130 else None)
    away = [lambda i: P(1, i / FPS, -20.0)]
    sub = H.event(9, None, 110 / FPS, gtype="SUB", ptype=None,
                  gameEvents=dict(gameEventType="SUB", period=1, teamId=None, playerId=None, homeTeam=None, playerOffId=1011, playerOnId=1012))
    c, _, _, rows, inst, refs = bench(build(tmp_path, 300, home, away, events=[sub], roster_=roster(home=range(1, 13), away=(1,))))
    s12 = slot(c, "home", 12)
    hit = [r for r in rows if s12 in {p["slot"] for p in r["participants"]}]
    assert hit and all(not r["accepted"] and "substitution_window" in r["rejection_reasons"] for r in hit)
    assert all(r["instance_id"] is None for r in hit)


# ---------------------------------------------------------------- leakage

FORBIDDEN = ("jersey", "player", "slot", "side", "team", "frame", "period", "match", "home", "away", "tier", "expected", "encounter")


def _walk_keys(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield path, k
            yield from _walk_keys(v, f"{path}.{k}")
    elif isinstance(o, list):
        for v in o:
            yield from _walk_keys(v, path)


def test_instances_expose_only_the_matcher_visible_schema(tmp_path):
    a, b = crossing()
    _, _, _, _, inst, refs = bench(build(tmp_path, 300, [a, b], [FAR]))
    (i,), (r,) = inst, refs
    assert set(i) == INSTANCE_KEYS and set(i["during"]) == DURING_KEYS
    assert all(set(tr) == TRACKLET_KEYS for tr in i["incoming"] + i["outgoing"])
    keys = {k for path, k in _walk_keys(i) if not path.startswith(".units")}
    assert not any(f in k for k in keys for f in FORBIDDEN)
    assert i["schema"] == "m11-matching-v3" and i["task"] == "A_identity_hidden_association"    # constants
    text = json.dumps({k: v for k, v in i.items() if k not in ("units", "schema")})
    assert not any(f in text for f in FORBIDDEN)
    assert str(r["start_frame"]) not in i["instance_id"] and "100" not in i["instance_id"]
    assert i["ambiguity_interval_s"][0] == 0.0 and max(i["incoming"][0]["t"]) < 0.0 < min(i["outgoing"][0]["t"])   # relative time only
    # tracklet IDs carry no slot order: only the reference links them to players
    assert {tr["tracklet_id"] for tr in i["outgoing"]} == {"out-0", "out-1"} and set(r["outgoing"]) == {"out-0", "out-1"}


def test_orders_are_seeded_random_and_independent_of_slot_order():
    perms = [tuple(shuffle_order("s", 100, f"k{n}", "outgoing", 3)) for n in range(300)]
    share = sum(p == (0, 1, 2) for p in perms) / len(perms)
    assert len(set(perms)) == 6 and 0.08 < share < 0.25                 # about 1/6, never always the slot order
    assert perms == [tuple(shuffle_order("s", 100, f"k{n}", "outgoing", 3)) for n in range(300)]
    assert [tuple(shuffle_order("s", 100, f"k{n}", "incoming", 3)) for n in range(300)] != perms


def test_baselines_take_only_instances_and_ignore_the_references(tmp_path):
    a, b = crossing()
    _, _, _, _, inst, refs = bench(build(tmp_path, 300, [a, b], [FAR]))
    assert list(inspect.signature(run_baselines).parameters) == ["instances", "bc"]
    assert list(inspect.signature(B.run_baseline).parameters) == ["name", "instance", "bc"]
    p1 = run_baselines(inst, BC)
    assert run_baselines(json.loads(json.dumps(inst)), BC) == p1           # same predictions from the serialized instance alone
    shifted = [dict(i, instance_id="x" * 16) for i in inst]
    assert [x["decisions"] for x in run_baselines(shifted, BC)] == [x["decisions"] for x in p1]


# ---------------------------------------------------------------- evaluation, determinism, outputs

def test_metric_definitions_on_a_hand_made_case():
    o = lambda dec, exp, out=None: outcome(dict(decision=dec, outgoing=out), exp)   # noqa: E731
    assert o("MATCHED", "out-0", "out-0") == "correct_match" and o("MATCHED", "out-0", "out-1") == "wrong_match"
    assert o("MATCHED", None, "out-0") == "wrong_match" and o("UNMATCHED", None) == "correct_unmatched"
    assert o("UNMATCHED", "out-0") == "missed_unmatched" and o("UNCERTAIN", "out-0") == "uncertain"
    m = metrics([(["correct_match", "correct_match"], False), (["wrong_match", "uncertain"], True), (["correct_unmatched", "missed_unmatched"], False)])
    assert m["incoming"] == 6 and m["correct_match_rate"] == round(2 / 6, 4) and m["wrong_match_rate"] == round(1 / 6, 4)
    assert m["coverage"] == 0.5 and m["false_association_rate"] == round(1 / 3, 4) and m["abstention_rate"] == 0.5
    assert m["complete_assignment_accuracy"] == round(1 / 3, 4) and m["conflicts"] == 1 and m["instances_with_wrong_match"] == 1


def test_solver_is_exact_and_respects_abstention():
    C = np.array([[1.0, 2.0], [1.5, 10.0]])
    assert B.solve(C, np.array([5.0, 5.0])) == (pytest.approx(3.5), (1, 0))
    assert B.solve(C, np.array([5.0, 0.5]))[1] == (0, -1)
    assert B.solve(np.array([[np.inf]]), np.array([1.0])) == (1.0, (-1,))


def _source(tmp_path):
    raw = build(tmp_path, 300, list(crossing()), [FAR])
    src = tmp_path / "processed" / "enc"
    run_catalogue(raw, "100", out_root=src)
    return raw, src


def test_outputs_are_deterministic_separate_safe_and_non_destructive(tmp_path):
    raw, src = _source(tmp_path)
    raw_digest = H.digest(raw)
    src_digest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (src / "100").iterdir()}
    o1, o2 = tmp_path / "processed" / "m1", tmp_path / "processed" / "m2"
    p1, s1 = run_match(raw, "100", source_root=src, out_root=o1)
    p2, _ = run_match(raw, "100", source_root=src, out_root=o2)
    for sfx in (k for k in p1 if k != ".manifest.json"):
        assert p1[sfx].read_bytes() == p2[sfx].read_bytes()
    m = json.loads(p1[".manifest.json"].read_text())
    assert m["provider_smoothed_coordinates_used"] is False and m["synthetic_swaps_injected"] is False
    assert m["config"]["matcher_visible_files"] == ["root.instances.jsonl", "root.instances_b.jsonl"] and "root.references.jsonl" in m["config"]["evaluation_only_files"]
    assert set(m["outputs_sha256"]) == {p.name for k, p in p1.items() if k != ".manifest.json"}
    assert m["source"]["sha256"]["root.cache.npz"] == src_digest["root.cache.npz"]
    assert s1["audit"]["accepted_by_tier"]["high_confidence_reference"] == 1
    assert "provider identity reference" in s1["note"]
    inst = [json.loads(x) for x in p1[".instances.jsonl"].open()]
    assert all(set(i) == INSTANCE_KEYS for i in inst)
    with pytest.raises(IngestionError, match="never overwritten"):
        run_match(raw, "100", source_root=src, out_root=o1)
    with pytest.raises(IngestionError, match="approved output root"):
        run_match(raw, "100", source_root=src, out_root=tmp_path / "elsewhere")
    assert H.digest(raw) == raw_digest
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (src / "100").iterdir()} == src_digest
    assert not list((o1 / "100").glob("*.tmp"))


def test_tampered_source_catalogue_is_refused(tmp_path):
    raw, src = _source(tmp_path)
    with open(src / "100" / "root.encounters.jsonl", "ab") as f:
        f.write(b"\n")
    with pytest.raises(IngestionError, match="checksum"):
        run_match(raw, "100", source_root=src, out_root=tmp_path / "processed" / "m")
    assert not (tmp_path / "processed" / "m").exists()


def test_build_does_not_modify_its_inputs(tmp_path):
    raw = build(tmp_path, 300, list(crossing()), [FAR])
    md = MatchData.open(raw, "100", EventVersion.ROOT, policy=CoordinatePolicy.RAW)
    c = build_cache(md)
    qr = assess(c, md.events(), M11Config())
    enc = find_encounters(c, qr, M11Config())
    before = ({k: v.copy() for k, v in c.arrays().items()}, qr.mask.copy(), json.dumps(enc, sort_keys=True))
    rows, inst, refs = build_benchmark(c, qr.mask, enc)
    assert all(np.array_equal(before[0][k], v, equal_nan=v.dtype.kind == "f") for k, v in c.arrays().items())
    assert np.array_equal(before[1], qr.mask) and before[2] == json.dumps(enc, sort_keys=True)
    a_before = json.dumps(inst, sort_keys=True)
    [withhold_interval(i) for i in inst]
    protocol_l(c, refs)
    assert json.dumps(inst, sort_keys=True) == a_before
    assert all(np.array_equal(before[0][k], v, equal_nan=v.dtype.kind == "f") for k, v in c.arrays().items())


# ---------------------------------------------------------------- task B (withheld interval)

def taskb(raw):
    c, qr, enc, rows, inst, refs = bench(raw)
    return c, rows, inst, [withhold_interval(i) for i in inst], refs


def test_task_b_withholds_exactly_the_interval_and_keeps_everything_else(tmp_path):
    a, b = crossing()
    est = lambda i: dict(b(i), visibility="ESTIMATED", confidence="LOW")   # noqa: E731
    c, rows, inst, ib, refs = taskb(build(tmp_path, 300, [a, est], [FAR]))
    (i,), (j,), (r,) = inst, ib, refs
    assert j["task"] == "B_withheld_interval_association" and j["during"] == {"t": [], "points": []} and set(j) == INSTANCE_KEYS
    assert {k: v for k, v in j.items() if k not in ("task", "during")} == {k: v for k, v in i.items() if k not in ("task", "during")}
    end = j["ambiguity_interval_s"][1]
    assert all(max(tr["t"]) < 0.0 for tr in j["incoming"]) and all(min(tr["t"]) > end for tr in j["outgoing"])   # boundaries
    assert any(set(tr["visibility"]) == {2} and set(tr["confidence"]) == {1} for tr in j["incoming"])              # quality kept
    text = json.dumps({k: v for k, v in j.items() if k not in ("units", "schema")})
    assert not any(f in text for f in FORBIDDEN)


def test_task_b_crossing_baselines_and_e0_adapted(tmp_path):
    _, _, _, (j,), (r,) = taskb(build(tmp_path, 300, list(crossing()), [FAR]))
    for name in ("constant_velocity", "hungarian_cv", "e0_adapted"):        # straight lines through the crossing
        d = decisions(name, j)
        assert all(x["decision"] == "MATCHED" and x["outgoing"] == r["expected"][k] for k, x in d.items()), name
    d = decisions("chain_during", j)                                        # no interval points: one gated step only
    assert len(d) == 2


def test_task_b_long_and_short_withheld_intervals_and_gap_bins(tmp_path):
    from football_fatigue.matching.evaluation import gap_bin

    def pair(speed):                                                        # X-crossing; slower players stay within 2 m longer
        return (lambda i: P(1, speed * (i - 200) / FPS, 0.5 * speed * (i - 200) / FPS),
                lambda i: P(2, speed * (i - 200) / FPS, -0.5 * speed * (i - 200) / FPS))
    _, _, _, (fast,), (rf,) = taskb(build(tmp_path / "fast", 400, list(pair(2.0)), [FAR]))
    _, _, _, (slow,), (rs,) = taskb(build(tmp_path / "slow", 400, list(pair(0.9)), [FAR]))
    assert slow["ambiguity_interval_s"][1] > 3.0 > fast["ambiguity_interval_s"][1]
    assert gap_bin(rf["difficulty"]["ambiguity_s"]) != gap_bin(rs["difficulty"]["ambiguity_s"])
    assert [gap_bin(x) for x in (0.5, 1.5, 3.0, 4.0, 9.0)] == ["0-1.5s", "1.5-2.5s", "2.5-4s", ">=4s", ">=4s"]
    for inst, ref in ((fast, rf), (slow, rs)):
        d = decisions("e0_adapted", inst)
        assert all(x["decision"] == "MATCHED" and x["outgoing"] == ref["expected"][k] for k, x in d.items())


def test_task_b_irregular_update_cadence(tmp_path):
    a, b = crossing()

    def held(f, period):                                                    # positions update only every `period` frames
        return lambda i: f(i - i % period)
    _, _, _, (j2,), (r2,) = taskb(build(tmp_path / "c2", 300, [held(a, 2), held(b, 2)], [FAR]))
    _, _, _, (j3,), (r3,) = taskb(build(tmp_path / "c3", 300, [held(a, 3), held(b, 5)], [FAR]))
    for j, r in ((j2, r2), (j3, r3)):
        for name in ("constant_velocity", "e0_adapted"):
            d = decisions(name, j)
            assert all(x["decision"] == "MATCHED" and x["outgoing"] == r["expected"][k] and x["cost"] is not None for k, x in d.items()), name


def test_task_b_multi_player_and_missing_outgoing(tmp_path):
    def lane(j, y):
        return lambda i: P(j, -10.0 + 2 * i / FPS, y + (0.0 if 120 <= i < 180 else 0.15 * min(abs(i - 120), abs(i - 179)) * np.sign(y)))
    _, _, _, (j,), (r,) = taskb(build(tmp_path / "three", 300, [lane(1, 0.0), lane(2, 0.8), lane(3, -0.8)], [FAR], roster_=roster(home=(1, 2, 3))))
    for name in BASELINES:
        d = decisions(name, j)
        claimed = [x["outgoing"] for x in d.values() if x["decision"] == "MATCHED"]
        assert len(d) == 3 and (name in ("nearest", "constant_velocity") or len(claimed) == len(set(claimed)))
    a = lambda i: P(1, -10.0 + 2 * i / FPS, 0.0)                         # noqa: E731

    def b(i):
        return P(2, None, None) if i >= 172 else P(2, -10.0 + 2 * i / FPS, 0.5 + 0.1 * max(0, 110 - i))
    _, _, _, (j,), (r,) = taskb(build(tmp_path / "missing", 300, [a, b], [FAR]))
    d = decisions("e0_adapted", j)
    b_in = by_player(r, "home", 2)
    assert r["expected"][b_in] is None and d[b_in]["decision"] != "MATCHED" or d[b_in]["outgoing"] != r["expected"][by_player(r, "home", 1)]
    claimed = [x["outgoing"] for x in d.values() if x["decision"] == "MATCHED"]
    assert len(claimed) == len(set(claimed)) <= 1


def test_task_a_is_unchanged_by_task_b(tmp_path):
    raw, src = _source(tmp_path)
    p, s = run_match(raw, "100", source_root=src, out_root=tmp_path / "processed" / "m")
    inst = [json.loads(x) for x in p[".instances.jsonl"].open()]
    inst_b = [json.loads(x) for x in p[".instances_b.jsonl"].open()]
    md = MatchData.open(raw, "100", EventVersion.ROOT, policy=CoordinatePolicy.RAW)
    c = build_cache(md)
    qr = assess(c, md.events(), M11Config())
    _, ref_inst, _ = build_benchmark(c, qr.mask, find_encounters(c, qr, M11Config()))
    assert inst == ref_inst and inst_b == [withhold_interval(i) for i in ref_inst]
    preds = [json.loads(x) for x in p[".predictions.jsonl"].open()]
    assert preds == run_baselines(ref_inst, BC)                             # task A predictions = the step-2 protocol
    m = json.loads(p[".manifest.json"].read_text())
    assert set(m["config"]["matcher_visible_files"]) == {"root.instances.jsonl", "root.instances_b.jsonl"}
    assert "root.footballtrace_literal.jsonl" in m["config"]["evaluation_only_files"]
    assert set(s["results_b"]) == set(BASELINES) and "footballtrace_literal" in s


# ---------------------------------------------------------------- FOOTBALLTrace E0

def test_savgol_preserves_polynomials_up_to_its_order_including_edges():
    x = np.arange(30, dtype=float)
    y = 0.3 * x ** 2 - 2 * x + 5
    assert np.allclose(FT.savgol(y, 7, 2), y) and np.allclose(FT.savgol(x ** 3, 7, 3), x ** 3)
    assert not np.allclose(FT.savgol(x ** 3, 7, 2), x ** 3)
    noisy = y + np.where(np.arange(30) % 2, 0.5, -0.5)
    assert np.abs(FT.savgol(noisy, 7, 2) - y).max() < 0.5                  # smoothing reduces alternating noise
    with pytest.raises(ValueError):
        FT.savgol(y, 6, 2)


def test_literal_breakpoint_rules_and_tail_exchange():
    t = np.arange(125) / 25.0                                               # 5 s at 25 fps; a V: right, then a sharp turn back
    x = np.where(t < 2.5, t, 5.0 - t) * 2.0
    y = t * 0.5
    k_angle = FT.literal_breakpoint(x, y, rule="min_angle")
    k_cos = FT.literal_breakpoint(x, y, rule="min_cosine")
    assert abs(t[k_angle] - 2.5) <= 7 / 25 and abs(t[k_cos] - 2.5) > 0.5    # the pseudocode as written misses the turn
    other = dict(t=t, x=x[::-1].copy(), y=-y)
    res = FT.literal_correct(dict(t=t, x=x, y=y), other, rule="min_angle")
    assert res["corrected"] and res["a"][0][0] == "a" and res["a"][1][0] == "b" and res["b"][1][0] == "a"   # always exchanges tails
    still = dict(t=t, x=np.zeros_like(t), y=np.zeros_like(t))               # no defined turn anywhere: no correction
    assert not FT.literal_correct(still, still)["corrected"]


def test_protocol_l_reads_labels_and_always_disagrees_when_it_corrects(tmp_path):
    c, _, _, _, inst, refs = bench(build(tmp_path, 300, list(crossing()), [FAR]))
    (row,) = FT.protocol_l(c, refs)
    for rule in ("min_angle", "min_cosine"):
        assert row[rule]["reference_agreement"] is (not row[rule]["corrected"])
    s = FT.summarize_protocol_l([row])
    assert s["all"]["pairs"] == 1 and set(s["all"]) >= {"min_angle", "min_cosine"}


def test_e0_adapted_has_no_reference_access_and_is_deterministic(tmp_path):
    assert list(inspect.signature(FT.adapted).parameters) == ["inst", "cfg"]
    _, _, _, ib, _ = taskb(build(tmp_path, 300, list(crossing()), [FAR]))
    assert FT.adapted(ib[0]) == FT.adapted(json.loads(json.dumps(ib[0])))
    src = Path(FT.__file__).read_text(encoding="utf-8").split("def adapted")[1]
    assert "expected" not in src and "reference" not in src
