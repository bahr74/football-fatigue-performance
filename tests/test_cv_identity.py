"""Tests for the opt-in M6 identity layer. All scenarios here are SYNTHETIC (hand-built boxes and votes)."""

import json
from collections import Counter, defaultdict

import pytest

from football_fatigue.vision.identity import (
    IdentityConfig,
    Observation,
    annotate_records,
    identity_trajectory_points,
    interpolate_gaps,
    main,
    motion_spread,
    observations_from_records,
    solve,
    solve_online,
)
from football_fatigue.vision.identity_eval import evaluate_identity, expected_calibration_error

H = 40.0  # synthetic box height (px)


def box(x, y, w=16.0, h=H):
    return (x - w / 2, y - h, x + w / 2, y)


def walk(x0, y0, vx, vy, frames):
    return {f: (x0 + vx * (f - frames[0]), y0 + vy * (f - frames[0])) for f in frames}


def build(tracks, votes=None, classes=None):
    """tracks: {local_id: {frame: (x, y)}} -> observations with per-frame indices."""
    by_frame = defaultdict(list)
    for tid, path in tracks.items():
        for f, (x, y) in path.items():
            by_frame[f].append((tid, x, y))
    obs = []
    for f in sorted(by_frame):
        for i, (tid, x, y) in enumerate(sorted(by_frame[f])):
            v = votes(tid, f) if votes else None
            c = (classes or {}).get(tid, "player")
            obs.append(Observation(f, i, tid, box(x, y), c, v))
    return obs


def ids_by_track_and_frame(res, obs):
    out = {}
    for o in obs:
        out[(o.local_track_id, o.frame)] = res["boxes"][(o.frame, o.index)]["global_id"]
    return out


def unique_per_frame(res):
    per = defaultdict(list)
    for (f, _), b in res["boxes"].items():
        per[f].append(b["global_id"])
    return all(len(v) == len(set(v)) for v in per.values())


# ---------------------------------------------------------------- linking

def test_fragment_across_short_gap_is_linked():
    a = walk(100, 300, 2.0, 0.0, range(0, 60))
    b = walk(100 + 2.0 * 70, 300, 2.0, 0.0, range(70, 130))
    obs = build({1: a, 2: b}, votes=lambda t, f: "team_0")
    res = solve(obs, IdentityConfig())
    ids = ids_by_track_and_frame(res, obs)
    assert ids[(1, 0)] == ids[(2, 70)]
    assert any(r["decision"] == "linked" for r in res["links"])


def test_link_rejected_on_confident_team_conflict():
    a = walk(100, 300, 2.0, 0.0, range(0, 60))
    b = walk(100 + 2.0 * 70, 300, 2.0, 0.0, range(70, 130))
    obs = build({1: a, 2: b}, votes=lambda t, f: "team_0" if t == 1 else "team_1")
    res = solve(obs, IdentityConfig())
    ids = ids_by_track_and_frame(res, obs)
    assert ids[(1, 0)] != ids[(2, 70)]
    assert any(r.get("reason") == "team_conflict" for r in res["links"])


def test_link_rejected_goalkeeper_vs_player():
    a = walk(100, 300, 2.0, 0.0, range(0, 60))
    b = walk(100 + 2.0 * 70, 300, 2.0, 0.0, range(70, 130))
    obs = build({1: a, 2: b}, classes={1: "goalkeeper"})
    res = solve(obs, IdentityConfig())
    ids = ids_by_track_and_frame(res, obs)
    assert ids[(1, 0)] != ids[(2, 70)]
    assert any(r.get("reason") == "role_conflict" for r in res["links"])


def test_link_rejected_when_motion_implausible():
    a = walk(100, 300, 2.0, 0.0, range(0, 60))
    b = walk(900, 300, 2.0, 0.0, range(65, 130))  # 20 box heights away after 5 frames
    obs = build({1: a, 2: b})
    res = solve(obs, IdentityConfig())
    ids = ids_by_track_and_frame(res, obs)
    assert ids[(1, 0)] != ids[(2, 65)]


def test_abstains_between_equally_plausible_continuations():
    a = walk(100, 300, 0.0, 0.0, range(0, 60))
    b = walk(100, 300 - 6, 0.0, 0.0, range(70, 130))
    c = walk(100, 300 + 6, 0.0, 0.0, range(70, 130))
    obs = build({1: a, 2: b, 3: c})
    res = solve(obs, IdentityConfig())
    ids = ids_by_track_and_frame(res, obs)
    assert ids[(1, 0)] not in (ids[(2, 70)], ids[(3, 70)])
    assert any(r["decision"] == "abstained" for r in res["links"])


def test_no_link_over_long_gap():
    a = walk(100, 300, 0.0, 0.0, range(0, 30))
    b = walk(100, 300, 0.0, 0.0, range(200, 230))  # 170-frame gap > max_link_gap
    obs = build({1: a, 2: b})
    res = solve(obs, IdentityConfig())
    ids = ids_by_track_and_frame(res, obs)
    assert ids[(1, 0)] != ids[(2, 200)]


def test_motion_spread_grows_with_gap():
    cfg = IdentityConfig()
    s = [motion_spread(g, cfg) for g in (1, 5, 25, 75, 150)]
    assert s == sorted(s) and s[0] < 0.2 and s[-1] > s[-2]


# ---------------------------------------------------------------- swap-aware splitting

def crossing_with_swap():
    """Two players cross at frame 100; the tracker exchanges their ids at the crossing."""
    p = walk(100, 300, 2.0, 0.0, range(0, 200))      # team_0 player moving right
    q = walk(500, 300, -2.0, 0.0, range(0, 200))     # team_1 player moving left
    tracks = {1: {f: (p if f < 100 else q)[f] for f in range(200)},
              2: {f: (q if f < 100 else p)[f] for f in range(200)}}
    truth = {(1, f): ("P" if f < 100 else "Q") for f in range(200)}
    truth.update({(2, f): ("Q" if f < 100 else "P") for f in range(200)})
    team = {"P": "team_0", "Q": "team_1"}
    return tracks, truth, lambda t, f: team[truth[(t, f)]]


def test_swap_is_split_and_repaired():
    tracks, truth, votes = crossing_with_swap()
    obs = build(tracks, votes=votes)
    res = solve(obs, IdentityConfig())
    ids = ids_by_track_and_frame(res, obs)
    # player P is track 1 before and track 2 after: one global id outside the ambiguous interval
    assert ids[(1, 20)] == ids[(2, 180)]
    assert ids[(2, 20)] == ids[(1, 180)]
    assert ids[(1, 20)] != ids[(1, 180)]
    splits = [s for s in res["splits"] if s["decision"] == "split"]
    assert splits and all(len(s["cues"]) >= 2 and s.get("repair") == "swap" for s in splits)
    statuses = {res["boxes"][(o.frame, o.index)]["identity_status"] for o in obs if 95 <= o.frame <= 102}
    assert "ambiguous" in statuses
    assert unique_per_frame(res)


def test_team_change_without_second_cue_is_not_split():
    path = walk(100, 300, 2.0, 0.0, range(0, 200))
    obs = build({1: path}, votes=lambda t, f: "team_0" if f < 100 else "team_1")  # isolated, smooth motion
    res = solve(obs, IdentityConfig())
    assert len({b["global_id"] for b in res["boxes"].values()}) == 1
    assert any(s["decision"] == "rejected" and s["reason"] == "single_cue" for s in res["splits"])


def test_split_disabled_keeps_tracker_identity():
    tracks, _, votes = crossing_with_swap()
    obs = build(tracks, votes=votes)
    res = solve(obs, IdentityConfig(split=False, link=False))
    ids = ids_by_track_and_frame(res, obs)
    assert ids[(1, 20)] == ids[(1, 180)] and ids[(2, 20)] == ids[(2, 180)]


def test_unpartnered_split_is_restored_not_fragmented():
    # one track changes team while overlapping a passer-by whose own votes are undecided
    p = walk(100, 300, 2.0, 0.0, range(0, 200))
    r = walk(300, 300, 0.0, 0.0, range(90, 110))
    obs = build({1: p, 5: r}, votes=lambda t, f: None if t == 5 else ("team_0" if f < 100 else "team_1"))
    res = solve(obs, IdentityConfig())
    ids = ids_by_track_and_frame(res, obs)
    split = [s for s in res["splits"] if s["decision"] == "split"]
    assert split and split[0]["repair"] == "restored"
    assert ids[(1, 10)] == ids[(1, 190)]


# ---------------------------------------------------------------- invariants

def random_scene(seed=0):
    import random

    rng = random.Random(seed)
    tracks = {}
    for t in range(1, 15):
        s = rng.randrange(0, 150)
        e = s + rng.randrange(10, 150)
        tracks[t] = walk(rng.uniform(0, 800), rng.uniform(200, 600), rng.uniform(-3, 3), rng.uniform(-1, 1), range(s, e))
    return build(tracks, votes=lambda t, f: ("team_0", "team_1", None)[(t + f // 37) % 3])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_unique_global_id_per_frame_and_every_box_kept(seed):
    obs = random_scene(seed)
    res = solve(obs, IdentityConfig())
    assert len(res["boxes"]) == len(obs)
    assert unique_per_frame(res)
    assert {b["identity_status"] for b in res["boxes"].values()} <= {"confirmed", "provisional", "ambiguous", "unknown"}


def test_deterministic():
    obs = random_scene(3)
    a, b = solve(obs, IdentityConfig()), solve(list(reversed(obs)), IdentityConfig())
    assert a["boxes"] == b["boxes"] and a["links"] == b["links"]


def test_online_is_causal_with_fixed_latency():
    cfg = IdentityConfig(online_window=120, online_step=10, online_latency=20)
    obs = random_scene(4)
    t1 = 150
    early = solve_online([o for o in obs if o.frame <= t1], cfg)["boxes"]
    full = solve_online(obs, cfg)
    stable_until = t1 - cfg.online_latency - cfg.online_step
    for k, b in early.items():
        if k[0] <= stable_until:
            assert full["boxes"][k]["global_id"] == b["global_id"]
    per = defaultdict(list)
    for (f, _), b in full["boxes"].items():
        per[f].append(b["global_id"])
    assert all(len(v) == len(set(v)) for v in per.values())
    assert len(full["boxes"]) == len(obs)


def test_short_tracklet_is_unknown():
    obs = build({1: walk(100, 300, 0, 0, range(0, 3))})
    res = solve(obs, IdentityConfig())
    assert {b["identity_status"] for b in res["boxes"].values()} == {"unknown"}


def test_config_round_trip_ignores_comment_keys():
    cfg = IdentityConfig(link_margin=0.3, calibration=tuple(float(i) for i in range(11)))
    d = json.loads(json.dumps(cfg.to_dict()))
    d["_note"] = "comment"
    assert IdentityConfig.from_dict(d) == cfg


def test_calibrated_confidence_uses_coefficients():
    obs = build({1: walk(100, 300, 1, 0, range(0, 50))})
    hi = solve(obs, IdentityConfig(calibration=(10.0,) + (0.0,) * 10))
    lo = solve(obs, IdentityConfig(calibration=(-10.0,) + (0.0,) * 10))
    assert {b["identity_status"] for b in hi["boxes"].values()} == {"confirmed"}
    assert {b["identity_status"] for b in lo["boxes"].values()} == {"unknown"}


# ---------------------------------------------------------------- records, interpolation, analytics

def records_from(obs, xy=None):
    per = defaultdict(list)
    for o in obs:
        per[o.frame].append(o)
    recs = []
    for f in sorted(per):
        players = []
        for o in sorted(per[f], key=lambda o: o.index):
            p = {"track_id": o.local_track_id, "detected_class": o.detected_class, "bbox_xyxy_px": list(o.box),
                 "confidence": 0.9, "x": None, "y": None, "position_usable": False}
            if xy:
                p.update(x=o.box[0] / 10, y=o.box[3] / 10, position_usable=True)
            players.append(p)
        players.append({"track_id": -1, "detected_class": "ball", "bbox_xyxy_px": [0, 0, 1, 1]})
        recs.append({"frame_num": f, "timestamp_s": f / 25.0, "players": players})
    return recs


def test_annotate_keeps_boxes_and_sets_global_track_id():
    tracks, _, votes = crossing_with_swap()
    obs = build(tracks, votes=votes)
    recs = records_from(obs)
    res = solve(observations_from_records(recs, None), IdentityConfig())
    out = annotate_records(recs, res["boxes"])
    assert [len(r["players"]) for r in out] == [len(r["players"]) for r in recs]
    for r in out:
        for p in r["players"]:
            if p["detected_class"] == "ball":
                assert "global_id" not in p
            else:
                assert p["track_id"] == p["global_id"] and "local_track_id" in p and p["position_status"] == "observed"


def test_interpolation_is_flagged_and_separate():
    a = {f: (100 + 2 * f, 300) for f in list(range(0, 20)) + list(range(26, 40))}
    obs = build({1: a})
    res = solve(obs, IdentityConfig())
    fb = {(o.frame, o.index): o.box for o in obs}
    interp = interpolate_gaps(res["boxes"], fb, max_gap=12)
    assert sorted(interp) == list(range(20, 26))
    assert all(x["position_status"] == "interpolated" for v in interp.values() for x in v)
    recs = {r["frame_num"]: r for r in records_from(obs)}
    recs = [recs.get(f, {"frame_num": f, "timestamp_s": f / 25.0, "players": []}) for f in range(40)]  # every frame has a record
    out = annotate_records(recs, res["boxes"], interp)
    assert all(len([p for p in r["players"] if p["detected_class"] != "ball"]) <= 1 for r in out)
    assert out[20]["identity_interpolated"] and not out[20]["players"]
    assert out[19]["identity_interpolated"] == []


def test_identity_trajectory_points_policy_and_no_double_counting():
    tracks, _, votes = crossing_with_swap()
    obs = build(tracks, votes=votes)
    recs = records_from(obs, xy=True)
    team_votes = defaultdict(list)
    for o in sorted(obs, key=lambda o: (o.frame, o.index)):
        team_votes[o.frame].append(o.team_vote)
    res = solve(observations_from_records(recs, team_votes), IdentityConfig(calibration=(10.0,) + (0.0,) * 10))
    out = annotate_records(recs, res["boxes"])
    gids = {p["global_id"] for r in out for p in r["players"] if "global_id" in p}
    boxes_per_frame = Counter(o.frame for o in obs)
    total, per_frame = 0, Counter()
    for g in gids:
        pts, cov = identity_trajectory_points(out, g, canonical_period=1)
        assert cov["used"] == len(pts) and cov["observed_boxes"] >= len(pts)
        assert len({p.frame_num for p in pts}) == len(pts)
        assert all(p.is_estimated for p in pts)
        total += len(pts)
        per_frame.update(p.frame_num for p in pts)
    assert total < len(obs) and all(per_frame[f] <= boxes_per_frame[f] for f in per_frame)
    # no point within the change guard of an ambiguous or tracklet-change frame
    amb = {(f) for (f, _), b in res["boxes"].items() if b["identity_status"] == "ambiguous"}
    assert amb
    for g in gids:
        pts, _ = identity_trajectory_points(out, g, canonical_period=1, change_guard_frames=12)
        assert all(min(abs(p.frame_num - f) for f in amb) > 12 for p in pts)
    with pytest.raises(ValueError):
        identity_trajectory_points(out, 1, canonical_period=0)


def test_cli_writes_new_run_with_provenance(tmp_path):
    tracks, _, votes = crossing_with_swap()
    run = tmp_path / "run"
    run.mkdir()
    with (run / "frames.jsonl").open("w") as fh:
        for r in records_from(build(tracks)):
            fh.write(json.dumps(r) + "\n")
    out = tmp_path / "run-identity"
    assert main([str(run), "--output", str(out), "--interpolate-max-gap", "5"]) == 0
    m = json.loads((out / "manifest.json").read_text())
    assert m["source_frames_sha256"] and m["mode"] == "offline" and m["team_evidence"]["used"] is False
    assert (out / "identity_graph.json").exists()
    with pytest.raises(FileExistsError):
        main([str(run), "--output", str(out)])


# ---------------------------------------------------------------- evaluation metrics (synthetic reference)

def test_identity_eval_counts_transfers_purity_and_resume():
    ref, pred = {}, {}
    for f in range(100):
        ref[f] = {1: box(100, 300), 2: box(400, 300)}
        if 40 <= f < 50:
            del ref[f][1]  # player 1 unseen: resume after a 10-frame gap
        pred[f] = []
        if 1 in ref[f]:
            pred[f].append({"id": 7 if f < 60 else 8, "box": ref[f][1], "status": "confirmed", "confidence": 0.9})
        pred[f].append({"id": 8 if f < 60 else 7, "box": ref[f][2], "status": "provisional", "confidence": 0.6})
    out = evaluate_identity(pred, ref, range(100), team_ref={1: "white", 2: "blue"})
    assert out["matched_boxes"] == 190
    assert out["within_id_transfers"] == {"transfer_cross_team": 2}
    assert out["resume_after_gap"]["1-12"] == {"correct": 1}
    assert out["gt_id_changes"] == 2
    assert set(out["by_status"]) == {"confirmed", "provisional"}
    assert 0 <= out["ece"] <= 1


def test_ece_perfect_and_worst():
    import numpy as np

    assert expected_calibration_error(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.0
    assert expected_calibration_error(np.array([1.0, 1.0]), np.array([0.0, 0.0])) == 1.0


# ---------------------------------------------------------------- scenario coverage (synthetic)

def test_same_team_swap_is_not_split_without_independent_evidence():
    # a same-team swap is invisible to the team cue; with smooth motion there is no second cue,
    # so the layer keeps the tracker's identities instead of guessing (a documented limitation)
    tracks, _, _ = crossing_with_swap()
    obs = build(tracks, votes=lambda t, f: "team_0")
    res = solve(obs, IdentityConfig())
    ids = ids_by_track_and_frame(res, obs)
    assert ids[(1, 20)] == ids[(1, 180)] and ids[(2, 20)] == ids[(2, 180)]
    assert not [s for s in res["splits"] if s["decision"] == "split"]
    assert unique_per_frame(res)


def test_opposing_team_crossing_without_swap_keeps_identities():
    p = walk(100, 300, 2.0, 0.0, range(0, 200))
    q = walk(500, 300, -2.0, 0.0, range(0, 200))
    obs = build({1: p, 2: q}, votes=lambda t, f: "team_0" if t == 1 else "team_1")
    res = solve(obs, IdentityConfig())
    ids = ids_by_track_and_frame(res, obs)
    assert len({ids[(1, f)] for f in range(200)}) == 1 and len({ids[(2, f)] for f in range(200)}) == 1
    assert ids[(1, 0)] != ids[(2, 0)]
    assert not [s for s in res["splits"] if s["decision"] == "split"]


def test_detector_miss_inside_a_local_track_keeps_one_identity():
    path = {f: (100 + 2 * f, 300) for f in range(120) if not 50 <= f < 54}  # 4 missed frames, same local id
    obs = build({1: path}, votes=lambda t, f: "team_0")
    res = solve(obs, IdentityConfig())
    assert len({b["global_id"] for b in res["boxes"].values()}) == 1
    assert len(res["tracklets"]) == 1


def test_false_positive_and_duplicate_boxes_get_their_own_ids():
    player = walk(100, 300, 2.0, 0.0, range(0, 120))
    fp = {f: (600, 500) for f in range(40, 43)}                          # 3-frame spurious box
    dup = {f: (player[f][0] + 1, 300) for f in range(60, 75)}            # duplicate box on the same player
    obs = build({1: player, 7: fp, 8: dup}, votes=lambda t, f: "team_0")
    res = solve(obs, IdentityConfig())
    ids = ids_by_track_and_frame(res, obs)
    assert len({ids[(1, f)] for f in range(120)}) == 1
    assert ids[(7, 40)] != ids[(1, 40)] and ids[(8, 60)] != ids[(1, 60)]
    assert {res["boxes"][(o.frame, o.index)]["identity_status"] for o in obs if o.local_track_id == 7} == {"unknown"}
    assert unique_per_frame(res) and len(res["boxes"]) == len(obs)


def test_unknown_team_still_links_on_motion_but_cannot_bridge_opposing_teams():
    a = walk(100, 300, 2.0, 0.0, range(0, 60))
    b = walk(100 + 2.0 * 66, 300, 2.0, 0.0, range(66, 120))
    res = solve(build({1: a, 2: b}), IdentityConfig())                    # no team evidence at all
    assert len({x["global_id"] for x in res["boxes"].values()}) == 1
    # team_0 -> unknown -> team_1: a chain may not join two confidently different teams
    c = walk(100 + 2.0 * 126, 300, 2.0, 0.0, range(126, 190))
    votes = lambda t, f: {1: "team_0", 2: None, 3: "team_1"}[t]
    obs = build({1: a, 2: b, 3: c}, votes=votes)
    res = solve(obs, IdentityConfig())
    ids = ids_by_track_and_frame(res, obs)
    assert ids[(1, 0)] != ids[(3, 126)]
    assert any(r.get("reason") in ("chain_team_conflict", "team_conflict") for r in res["links"])


def test_link_rejected_referee_vs_player():
    a = walk(100, 300, 2.0, 0.0, range(0, 60))
    b = walk(100 + 2.0 * 70, 300, 2.0, 0.0, range(70, 130))
    obs = build({1: a, 2: b}, classes={1: "referee"})
    res = solve(obs, IdentityConfig())
    ids = ids_by_track_and_frame(res, obs)
    assert ids[(1, 0)] != ids[(2, 70)]
    assert any(r.get("reason") == "role_conflict" for r in res["links"])
    assert {t["role"] for t in res["tracklets"]} == {"referee", "player"}


def test_exit_and_reentry_elsewhere_and_substitution_get_new_ids():
    out = walk(100, 300, -3.0, 0.0, range(0, 30))                        # leaves at the left edge
    back = walk(1500, 300, -3.0, 0.0, range(40, 90))                     # re-enters far away: implausible
    sub = walk(800, 500, 0.0, 0.0, range(400, 450))                      # same-team substitute, much later
    obs = build({1: out, 2: back, 3: sub}, votes=lambda t, f: "team_0")
    res = solve(obs, IdentityConfig())
    ids = ids_by_track_and_frame(res, obs)
    assert len({ids[(1, 0)], ids[(2, 40)], ids[(3, 400)]}) == 3


def test_online_mode_is_deterministic():
    obs = random_scene(5)
    cfg = IdentityConfig(online_window=120, online_step=10, online_latency=20)
    assert solve_online(obs, cfg)["boxes"] == solve_online(list(reversed(obs)), cfg)["boxes"]


def test_predicted_or_interpolated_positions_never_feed_analytics():
    obs = build({1: walk(100, 300, 1.0, 0.0, range(0, 60))}, votes=lambda t, f: "team_0")
    recs = records_from(obs, xy=True)
    res = solve(observations_from_records(recs, None), IdentityConfig(calibration=(10.0,) + (0.0,) * 10))
    out = annotate_records(recs, res["boxes"])
    gid = out[0]["players"][0]["global_id"]
    for r in out[20:30]:
        r["players"][0]["position_status"] = "predicted"
    pts, cov = identity_trajectory_points(out, gid, canonical_period=1, change_guard_frames=0)
    assert cov["excluded_not_observed"] == 10 and cov["used"] == 50
    assert not {p.frame_num for p in pts} & set(range(20, 30))


def test_team_balance_gate_withdraws_implausible_team_clusters():
    from football_fatigue.vision.team_appearance import gated_team_votes, team_balance_check

    balanced = {f: ["team_0"] * 7 + ["team_1"] * 6 + ["other", None] for f in range(50)}
    gk_vs_rest = {f: ["team_0"] + ["team_1"] * 13 + [None] for f in range(50)}  # GK kit vs both outfield teams
    assert team_balance_check(balanced)["reliable"]
    votes, check = gated_team_votes(gk_vs_rest)
    assert votes is None and check["reason"] == "unbalanced_team_clusters" and check["median_balance"] < 1 / 3
    assert gated_team_votes({0: [None, "other"]})[0] is None


def test_withdrawn_team_evidence_cannot_block_or_force_links():
    # with the gate's output (no votes) the decision rests on motion alone: same result as use_team=False
    a = walk(100, 300, 2.0, 0.0, range(0, 60))
    b = walk(100 + 2.0 * 70, 300, 2.0, 0.0, range(70, 130))
    no_votes = solve(build({1: a, 2: b}), IdentityConfig())
    no_team = solve(build({1: a, 2: b}, votes=lambda t, f: "team_0" if t == 1 else "team_1"), IdentityConfig(use_team=False))
    gid = lambda r: {k: b["global_id"] for k, b in r["boxes"].items()}
    assert gid(no_votes) == gid(no_team) and len(set(gid(no_votes).values())) == 1
    assert not [s for s in no_votes["splits"] if s["decision"] == "split"]


# ---------------------------------------------------------------- M6.1 (synthetic)

def synthetic_kits(seed=0, minority=True):
    """Per-box 6-d torso descriptors: two outfield kits (white, blue) and, optionally, three distinctive
    minority kits (two goalkeepers, a referee) whose chroma dominates a 2-cluster fit (the M6 failure mode)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    kits = {"white": (200, -3, 4, 180, 225, 5), "blue": (80, -3, -6, 65, 115, 7),
            "gk_y": (110, -7, 40, 90, 130, 40), "gk_r": (90, 40, 25, 70, 110, 50), "ref": (60, 0, 40, 50, 80, 40)}
    noise = {"white": (25, 0.6, 0.6, 25, 25, 0.6), "blue": (25, 0.6, 0.6, 25, 25, 0.6)}
    noise.update({k: (15, 4, 4, 12, 12, 4) for k in ("gk_y", "gk_r", "ref")})
    people = [("white", i) for i in range(10)] + [("blue", i) for i in range(10)]
    people += [("gk_y", 0), ("gk_r", 0), ("ref", 0)] if minority else []
    tracks, by_frame = defaultdict(list), {}
    for f in range(60):
        row = []
        for k, i in people:
            x = np.array(kits[k], float) + rng.normal(0, 1, 6) * np.array(noise[k])
            row.append(x)
            tracks[(k, i)].append(x)
        by_frame[f] = row
    return tracks, by_frame, people


def test_balanced_team_model_absorbs_minority_kits_that_break_two_means():
    from football_fatigue.vision.team_appearance import fit_balanced_team_model, fit_team_model, team_balance_check

    tracks, by_frame, people = synthetic_kits()
    m2 = fit_team_model(tracks, method="kmeans2_box")
    assert not team_balance_check({f: [m2.vote(x) for x in fs] for f, fs in by_frame.items()})["reliable"]
    model, votes, info = fit_balanced_team_model(tracks, by_frame)
    assert info["reliable"] and info["selected_k"] > 2 and info["tried"][0]["k"] == 2
    kit_of = {n: k for n, (k, _) in enumerate(people)}
    for kit in ("white", "blue"):  # each outfield kit maps to one team label, and the two labels differ
        labels = Counter(v for fs in votes.values() for n, v in enumerate(fs) if kit_of[n] == kit)
        assert labels.most_common(1)[0][1] / sum(labels.values()) > 0.95
    white = Counter(v for fs in votes.values() for n, v in enumerate(fs) if kit_of[n] == "white").most_common(1)[0][0]
    blue = Counter(v for fs in votes.values() for n, v in enumerate(fs) if kit_of[n] == "blue").most_common(1)[0][0]
    assert white != blue and {white, blue} == {"team_0", "team_1"}
    minority = Counter(v for fs in votes.values() for n, v in enumerate(fs) if kit_of[n] in ("gk_y", "gk_r", "ref"))
    assert minority["other"] / sum(minority.values()) > 0.9


def test_balanced_team_model_keeps_two_means_when_it_is_already_balanced():
    from football_fatigue.vision.team_appearance import fit_balanced_team_model, fit_team_model

    tracks, by_frame, _ = synthetic_kits(minority=False)
    m2 = fit_team_model(tracks, method="kmeans2_box")
    model, votes, info = fit_balanced_team_model(tracks, by_frame)
    assert info["selected_k"] == 2
    assert votes == {f: [m2.vote(x) for x in fs] for f, fs in by_frame.items()}  # identical to M6 kmeans2_box


def test_balanced_team_model_withdraws_when_no_k_is_plausible():
    import numpy as np
    from football_fatigue.vision.team_appearance import fit_balanced_team_model

    rng = np.random.default_rng(1)  # one kit only: no k can produce two balanced teams per frame
    tracks = {i: [rng.normal(100, 1, 6) for _ in range(30)] for i in range(12)}
    by_frame = {f: [tracks[i][f] for i in range(12)] for f in range(30)}
    model, votes, info = fit_balanced_team_model(tracks, by_frame, ks=(2,))
    assert (model is None and votes is None) or info["reliable"]  # never a silent unbalanced model
    if votes is not None:
        from football_fatigue.vision.team_appearance import team_balance_check
        assert team_balance_check(votes)["reliable"]


def test_long_gap_link_requires_known_team_on_both_sides():
    a = walk(100, 300, 2.0, 0.0, range(0, 60))
    b = walk(100 + 2.0 * 100, 300, 2.0, 0.0, range(100, 160))  # 41-frame gap
    obs = build({1: a, 2: b}, votes=lambda t, f: "team_0" if t == 1 else None)
    assert len(set(ids_by_track_and_frame(solve(obs, IdentityConfig()), obs).values())) == 1  # M6: links
    res = solve(obs, IdentityConfig(link_require_team_gap=25))
    assert len(set(ids_by_track_and_frame(res, obs).values())) == 2
    assert any(r.get("reason") == "team_unknown_long_gap" for r in res["links"])
    known = build({1: a, 2: b}, votes=lambda t, f: "team_0")
    assert len(set(ids_by_track_and_frame(solve(known, IdentityConfig(link_require_team_gap=25)), known).values())) == 1
    short = build({1: a, 2: walk(100 + 2.0 * 70, 300, 2.0, 0.0, range(70, 130))}, votes=lambda t, f: "team_0" if t == 1 else None)
    assert len(set(ids_by_track_and_frame(solve(short, IdentityConfig(link_require_team_gap=25)), short).values())) == 1


def test_rule_blocked_alternative_still_makes_the_link_ambiguous():
    a = walk(100, 300, 0.0, 0.0, range(0, 60))
    b = walk(100, 300 - 6, 0.0, 0.0, range(100, 160))   # same team, plausible
    c = walk(100, 300 + 6, 0.0, 0.0, range(100, 160))   # team unknown, equally plausible: blocked by the rule
    obs = build({1: a, 2: b, 3: c}, votes=lambda t, f: None if t == 3 else "team_0")
    res = solve(obs, IdentityConfig(link_require_team_gap=25, team_isolated_only=False))  # b, c stand close
    assert {t["tracklet_id"]: t["team"] for t in res["tracklets"]}[2] == "team_0"
    ids = ids_by_track_and_frame(res, obs)
    assert ids[(1, 0)] not in (ids[(2, 100)], ids[(3, 100)])
    assert any(r["decision"] == "abstained" for r in res["links"])


def test_m61_confidence_features_mark_unverified_interactions():
    from football_fatigue.vision.identity import CONFIDENCE_FEATURES, CONFIDENCE_FEATURES_M61_EXTRA

    p = walk(100, 300, 2.0, 0.0, range(0, 200))
    q = walk(500, 300, -2.0, 0.0, range(0, 200))
    cfg = IdentityConfig(confidence_features="m61", split=False, link=False)
    k = len(CONFIDENCE_FEATURES) + CONFIDENCE_FEATURES_M61_EXTRA.index("log1p_unverified_episodes")
    cross = build({1: p, 2: q}, votes=lambda t, f: "team_0" if t == 1 else "team_1")
    same = build({1: p, 2: q}, votes=lambda t, f: "team_0")
    for obs, unverified in ((cross, False), (same, True)):
        res = solve(obs, cfg)
        feats = [b["features"] for b in res["boxes"].values()]
        assert all(len(x) == len(CONFIDENCE_FEATURES) + len(CONFIDENCE_FEATURES_M61_EXTRA) for x in feats)
        assert any(x[k] > 0 for x in feats) == unverified
    assert all(len(b["features"]) == len(CONFIDENCE_FEATURES) for b in solve(same, IdentityConfig())["boxes"].values())


def test_calibration_length_must_match_feature_set():
    obs = build({1: walk(100, 300, 1.0, 0.0, range(0, 30))}, votes=lambda t, f: "team_0")
    with pytest.raises(ValueError):
        solve(obs, IdentityConfig(confidence_features="m61", calibration=(0.0,) * 11))
    with pytest.raises(ValueError):
        solve(obs, IdentityConfig(confidence_features="nope"))
    res = solve(obs, IdentityConfig(confidence_features="m61", calibration=(0.0,) * 16))
    assert all(b["identity_confidence"] == 0.5 for b in res["boxes"].values())


def test_cli_accepts_m61_config_and_team_method(tmp_path):
    tracks, _, _ = crossing_with_swap()
    run = tmp_path / "run"
    run.mkdir()
    with (run / "frames.jsonl").open("w") as fh:
        for r in records_from(build(tracks)):
            fh.write(json.dumps(r) + "\n")
    cfg = tmp_path / "m61.json"
    cfg.write_text(json.dumps({"confidence_features": "m61", "calibration": [0.0] * 16, "link_require_team_gap": 25,
                               "_note": "synthetic"}))
    out = tmp_path / "out"
    assert main([str(run), "--output", str(out), "--config", str(cfg), "--team-method", "balanced"]) == 0
    m = json.loads((out / "manifest.json").read_text())
    assert m["config"]["confidence_features"] == "m61" and m["config"]["link_require_team_gap"] == 25
