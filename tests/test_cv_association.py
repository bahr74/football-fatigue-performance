"""Tests for the opt-in M6.3-A learned, deferred global association. All scenarios are SYNTHETIC (hand-built boxes and votes).

The link weights used here are hand-written test weights (a simple continuity scorer); the production weights are fitted
on development data and live in the config, not in the tests.
"""

import json
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest

from football_fatigue.vision import association as A
from football_fatigue.vision.identity import IdentityConfig, Observation, annotate_records, main, solve, solve_online, split_tracks

H = 40.0


def box(x, y, h=H):
    w = 0.4 * h
    return (x - w / 2, y - h, x + w / 2, y)


def walk(x0, y0, vx, vy, frames):
    return {f: (x0 + vx * (f - frames[0]), y0 + vy * (f - frames[0])) for f in frames}


def build(tracks, votes=None, classes=None, h=H, app=None):
    by_frame = defaultdict(list)
    for tid, path in tracks.items():
        for f, (x, y) in path.items():
            by_frame[f].append((tid, x, y))
    obs = []
    for f in sorted(by_frame):
        for i, (tid, x, y) in enumerate(sorted(by_frame[f])):
            obs.append(Observation(f, i, tid, box(x, y, h), (classes or {}).get(tid, "player"), votes(tid, f) if votes else None,
                                   app(tid, f) if app else None))
    return obs


def weights(**kw):
    """Test scorer: continuity (small residuals, similar size, short gap) is likely; everything else neutral."""
    w = dict.fromkeys(A.ASSOCIATION_FEATURES, 0.0)
    w.update(bias=3.0, log_gap=-0.4, log1p_disp_h=-1.0, log1p_resid_fwd=-4.0, log1p_resid_bwd=-4.0, abs_log_size_ratio=-4.0)
    w.update(kw)
    return tuple(w[k] for k in A.ASSOCIATION_FEATURES)


def deferred_cfg(**kw):
    base = dict(association_mode="deferred", association_weights=weights(), association_image_width=4096, confirm_threshold=0.95)
    base.update(kw)
    return IdentityConfig(**base)


def gids(res, obs):
    return {(o.local_track_id, o.frame): res["boxes"][(o.frame, o.index)]["global_id"] for o in obs}


def assoc_records(res, decision):
    return [r for r in res["links"] if r.get("source") == "association" and r["decision"] == decision]


def assert_chain_consistency(res):
    per_frame = defaultdict(list)
    for (f, _), b in res["boxes"].items():
        per_frame[f].append(b["global_id"])
    assert all(len(v) == len(set(v)) for v in per_frame.values())            # one global id per box, none twice in a frame
    chains = defaultdict(list)
    for t in res["tracklets"]:
        chains[t["global_id"]].append(t)
    for ts in chains.values():
        ts.sort(key=lambda t: t["start"])
        assert all(a["end"] < b["start"] for a, b in zip(ts, ts[1:]))        # chain members are time-disjoint
        assert len({t["team"] for t in ts if t["team"]}) <= 1                # and carry at most one known team


# ---------------------------------------------------------------- default-off compatibility

def test_default_is_the_m61_linker_and_ignores_association_settings():
    a = walk(100, 300, 0.0, 0.0, range(0, 60))
    b = walk(100, 300, 0.0, 0.0, range(70, 130))
    obs = build({1: a, 2: b}, votes=lambda t, f: "team_0")
    assert IdentityConfig().association_mode == "m61"
    base = solve(obs, IdentityConfig())
    other = solve(obs, IdentityConfig(association_weights=weights(), association_delta=9.0, association_tau=0.99))
    assert base["boxes"] == other["boxes"] and base["links"] == other["links"]
    assert all("association" not in b for b in base["boxes"].values())
    assert all(r.get("source") != "association" for r in base["links"])


def test_config_roundtrip_and_validation():
    cfg = deferred_cfg()
    again = IdentityConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert again == cfg and isinstance(again.association_weights, tuple)
    with pytest.raises(ValueError):
        solve([], IdentityConfig(association_mode="deferred"))                                   # no weights
    with pytest.raises(ValueError):
        solve([], IdentityConfig(association_mode="greedy"))
    with pytest.raises(ValueError):                                                              # discount must keep boxes below 'confirmed'
        solve([], deferred_cfg(association_confidence_discount=0.99, confirm_threshold=0.98))
    with pytest.raises(ValueError):
        solve_online([], deferred_cfg())                                                         # offline only


# ---------------------------------------------------------------- long gaps and never-forced links

def test_long_gap_link_only_in_deferred_mode_and_within_the_gap_limit():
    a = walk(100, 300, 0.0, 0.0, range(0, 60))
    b = walk(100, 300, 0.0, 0.0, range(360, 420))                       # 301-frame gap
    obs = build({1: a, 2: b}, votes=lambda t, f: "team_0")
    assert gids(solve(obs, IdentityConfig()), obs)[(1, 0)] != gids(solve(obs, IdentityConfig()), obs)[(2, 360)]
    res = solve(obs, deferred_cfg())
    g = gids(res, obs)
    assert g[(1, 0)] == g[(2, 360)]
    assert assoc_records(res, "linked") and assoc_records(res, "linked")[0]["gap"] == 301
    short = solve(obs, deferred_cfg(association_max_gap=200))
    assert gids(short, obs)[(1, 0)] != gids(short, obs)[(2, 360)]
    assert_chain_consistency(res)


def test_implausible_only_candidate_is_never_forced():
    a = walk(100, 300, 2.0, 0.0, range(0, 60))
    b = walk(1500, 300, 2.0, 0.0, range(80, 140))                       # 35 box heights away after 20 frames; the only continuation
    obs = build({1: a, 2: b})
    res = solve(obs, deferred_cfg())
    assert gids(res, obs)[(1, 0)] != gids(res, obs)[(2, 80)]
    assert not assoc_records(res, "linked")


# ---------------------------------------------------------------- competing links and abstention

def test_equally_plausible_continuations_are_deferred_not_forced():
    a = walk(100, 300, 0.0, 0.0, range(0, 60))
    b = walk(100, 300 - 6, 0.0, 0.0, range(70, 130))
    c = walk(100, 300 + 6, 0.0, 0.0, range(70, 130))
    obs = build({1: a, 2: b, 3: c})
    res = solve(obs, deferred_cfg())
    g = gids(res, obs)
    assert len({g[(1, 0)], g[(2, 70)], g[(3, 70)]}) == 3                  # nobody is merged
    held = assoc_records(res, "deferred")
    assert held and held[0]["margin"] < 0.5 and held[0]["competitor"] is not None   # kept as a competing hypothesis with its rival
    assert not assoc_records(res, "linked")


def test_clearly_better_continuation_wins_the_deferral_test():
    a = walk(100, 300, 0.0, 0.0, range(0, 60))
    b = walk(100, 300, 0.0, 0.0, range(70, 130))                          # exact continuation
    c = walk(100, 300 + 12, 0.0, 0.0, range(70, 130))                     # 0.3 heights off
    obs = build({1: a, 2: b, 3: c})
    res = solve(obs, deferred_cfg())
    g = gids(res, obs)
    assert g[(1, 0)] == g[(2, 70)] != g[(3, 70)]
    lk = assoc_records(res, "linked")
    assert [(r["from"], r["to"]) for r in lk] and all(r["p"] >= 0.3 for r in lk)
    strict = solve(obs, deferred_cfg(association_delta=50.0))             # an unreachable margin defers everything
    assert not assoc_records(strict, "linked") and assoc_records(strict, "deferred")


def test_defer_counts_competitors_below_tau():
    T = {i: SimpleNamespace(tracklet_id=i) for i in (1, 2, 3)}
    cands = [(T[1], T[2]), (T[1], T[3])]
    p = np.array([0.5, 0.29])                                             # the rival is below tau = 0.3 but still a rival
    matched = A.global_assignment(cands, p, 0.3)
    assert matched == [0]
    kept, held = A.defer(cands, p, matched, 1.0)
    assert not kept and held[0][2] == 1 and held[0][1] == pytest.approx(0.895, abs=1e-3)
    kept, held = A.defer(cands, p, matched, 0.5)
    assert kept == [0] and not held


def test_global_assignment_maximises_total_weight_and_is_one_to_one():
    T = {i: SimpleNamespace(tracklet_id=i) for i in (1, 2, 3, 4)}
    cands = [(T[1], T[2]), (T[1], T[3]), (T[4], T[2])]
    p = np.array([0.90, 0.85, 0.85])
    assert {cands[k][0].tracklet_id * 10 + cands[k][1].tracklet_id for k in A.global_assignment(cands, p, 0.3)} == {13, 42}
    assert A.global_assignment(cands, p, 0.95) == []


# ---------------------------------------------------------------- temporal conflicts, wrong merges, chain consistency

def test_overlapping_tracklets_are_never_candidates_or_merged():
    a = walk(100, 300, 0.0, 0.0, range(0, 60))
    b = walk(100, 420, 0.0, 0.0, range(40, 100))                          # a different player, simultaneous with a
    c = walk(100, 300, 0.0, 0.0, range(80, 140))                          # continues a
    obs = build({1: a, 2: b, 3: c})
    res = solve(obs, deferred_cfg())
    g = gids(res, obs)
    assert g[(1, 0)] == g[(3, 80)] != g[(2, 40)]
    assert_chain_consistency(res)
    cfg = IdentityConfig()
    tracks = defaultdict(list)
    for o in obs:
        tracks[o.local_track_id].append(o)
    tl = split_tracks(tracks, {}, cfg)[0]
    pairs = {(x.local_track_id, y.local_track_id) for x, y in A.candidates(tl, deferred_cfg())}
    assert (1, 2) not in pairs and (2, 1) not in pairs and (2, 3) not in pairs and (1, 3) in pairs   # b ends after c starts


def test_known_different_teams_and_goalkeeper_vs_player_are_hard_exclusions():
    a = walk(100, 300, 2.0, 0.0, range(0, 60))
    b = walk(100 + 2.0 * 70, 300, 2.0, 0.0, range(70, 130))              # kinematically perfect continuation
    obs = build({1: a, 2: b}, votes=lambda t, f: "team_0" if t == 1 else "team_1")
    res = solve(obs, deferred_cfg())
    assert gids(res, obs)[(1, 0)] != gids(res, obs)[(2, 70)] and not assoc_records(res, "linked")
    obs = build({1: a, 2: b}, classes={1: "goalkeeper"})
    res = solve(obs, deferred_cfg())
    assert gids(res, obs)[(1, 0)] != gids(res, obs)[(2, 70)] and not assoc_records(res, "linked")
    obs = build({1: a, 2: b}, classes={1: "goalkeeper", 2: "goalkeeper"})  # a goalkeeper may continue as the same goalkeeper
    res = solve(obs, deferred_cfg())
    assert gids(res, obs)[(1, 0)] == gids(res, obs)[(2, 70)]


def test_chain_never_mixes_two_known_teams_through_an_unknown_middle():
    a = walk(100, 300, 2.0, 0.0, range(0, 41))
    b = walk(100 + 2.0 * 60, 300, 2.0, 0.0, range(60, 101))
    c = walk(100 + 2.0 * 120, 300, 2.0, 0.0, range(120, 161))
    votes = lambda t, f: {1: "team_0", 3: "team_1"}.get(t)                # the middle tracklet has no team evidence
    obs = build({1: a, 2: b, 3: c}, votes=votes)
    res = solve(obs, deferred_cfg())
    assert len(assoc_records(res, "linked")) == 1
    assert [r["reason"] for r in assoc_records(res, "rejected")] == ["chain_team_conflict"]
    assert gids(res, obs)[(1, 0)] != gids(res, obs)[(3, 120)]
    assert_chain_consistency(res)


def test_fragmented_players_are_reassembled_without_cross_player_merges_and_deterministically():
    tracks, truth = {}, {}
    lid = 0
    for k, y in enumerate((200, 320, 440)):                               # three players on parallel lines, three pieces each
        for lo, hi in ((0, 50), (70, 130), (150, 210)):
            lid += 1
            tracks[lid] = walk(100 + 2.0 * lo, y, 2.0, 0.0, range(lo, hi))
            truth[lid] = k
    obs = build(tracks, votes=lambda t, f: "team_0")
    res = solve(obs, deferred_cfg())
    g = {t: gids(res, obs)[(t, min(tracks[t]))] for t in tracks}
    for t in tracks:
        for u in tracks:
            assert (g[t] == g[u]) == (truth[t] == truth[u])
    assert_chain_consistency(res)
    assert solve(obs, deferred_cfg())["boxes"] == res["boxes"]


# ---------------------------------------------------------------- confidence stays separate; newly linked segments are not trusted

def test_linked_segments_are_never_confirmed_and_untouched_ones_keep_their_confidence():
    a = walk(100, 300, 0.0, 0.0, range(0, 60))
    b = walk(100, 300, 0.0, 0.0, range(70, 130))
    far = walk(900, 700, 0.0, 0.0, range(0, 130))                         # one long, untouched track
    obs = build({1: a, 2: b, 3: far})
    kw = dict(confirm_threshold=0.6, provisional_threshold=0.4, association_confidence_discount=0.5)
    base = solve(obs, IdentityConfig(**kw))
    res = solve(obs, deferred_cfg(**kw))
    assert gids(res, obs)[(1, 0)] == gids(res, obs)[(2, 70)]
    touched = [b_ for k, b_ in res["boxes"].items() if b_["association"] == "linked_unvalidated"]
    assert touched and all(b_["identity_status"] != "confirmed" for b_ in touched)
    assert all(b_["identity_confidence"] == pytest.approx(0.5 * b_["identity_confidence_uncapped"], abs=1e-3) for b_ in touched)
    assert any(b_["identity_confidence_uncapped"] >= 0.6 for b_ in touched)   # the cap, not weak evidence, removed 'confirmed'
    far_keys = [(o.frame, o.index) for o in obs if o.local_track_id == 3]
    assert all(res["boxes"][k]["identity_status"] == base["boxes"][k]["identity_status"] == "confirmed" for k in far_keys)
    assert all(res["boxes"][k]["identity_confidence"] == base["boxes"][k]["identity_confidence"] for k in far_keys)
    assert all(res["boxes"][k]["association"] == "none" for k in far_keys)


def test_annotate_records_marks_association_only_in_deferred_mode():
    a = walk(100, 300, 0.0, 0.0, range(0, 60))
    b = walk(100, 300, 0.0, 0.0, range(70, 130))
    obs = build({1: a, 2: b})
    recs = [{"frame_num": f, "players": [{"track_id": o.local_track_id, "bbox_xyxy_px": list(o.box), "detected_class": "player"}
                                         for o in obs if o.frame == f]} for f in sorted({o.frame for o in obs})]
    plain = annotate_records(recs, solve(obs, IdentityConfig())["boxes"])
    deferred = annotate_records(recs, solve(obs, deferred_cfg())["boxes"])
    assert all("association_status" not in p for r in plain for p in r["players"])
    assert {p["association_status"] for r in deferred for p in r["players"]} == {"linked_unvalidated"}


# ---------------------------------------------------------------- features

def tracklets_of(obs):
    tracks = defaultdict(list)
    for o in obs:
        tracks[o.local_track_id].append(o)
    return split_tracks(tracks, {}, IdentityConfig())[0]


def test_features_are_scale_free_for_a_uniformly_scaled_scene():
    def scene(s):
        a = walk(100 * s, 300 * s, 2.0 * s, 0.5 * s, range(0, 60))
        b = walk(100 * s + 2.0 * s * 90, 300 * s + 0.5 * s * 90, 2.0 * s, 0.5 * s, range(90, 150))
        return build({1: a, 2: b}, h=H * s)
    f = []
    for s in (1.0, 2.0):
        obs = scene(s)
        tl = tracklets_of(obs)
        cfg = deferred_cfg(association_image_width=int(4096 * s))
        pairs = A.candidates(tl, cfg)
        f.append(A.candidate_features(pairs, tl, cfg, {}))
    assert f[0].shape == f[1].shape == (1, len(A.ASSOCIATION_FEATURES))
    assert np.allclose(f[0], f[1], atol=1e-6)


def test_colour_descriptor_feature_present_ordered_and_missing_when_unsupported():
    rng = np.random.default_rng(0)
    base = {1: 0.2, 2: 0.2, 3: 0.8, 4: 0.2}

    def app(tid, f):
        if tid == 4 and f % 2:                                             # track 4 has too few described boxes
            return None
        return tuple(base[tid] + 0.01 * rng.standard_normal(6))
    a = walk(100, 300, 0.0, 0.0, range(0, 80))
    b = walk(100, 300, 0.0, 0.0, range(100, 180))
    c = walk(100, 300, 0.0, 0.0, range(100, 180))                          # same place, different shirt colour
    d = walk(100, 300, 0.0, 0.0, range(200, 206))
    obs = build({1: a, 2: b, 3: c, 4: d}, app=lambda t, f: app(t, f))
    tl = tracklets_of(obs)
    cfg = deferred_cfg()
    pairs = A.candidates(tl, cfg)
    X = A.candidate_features(pairs, tl, cfg, {})
    k = {(x.local_track_id, y.local_track_id): i for i, (x, y) in enumerate(pairs)}
    dist, miss = A.ASSOCIATION_FEATURES.index("app_torso_dist"), A.ASSOCIATION_FEATURES.index("app_missing")
    assert X[k[(1, 2)], miss] == 0 and X[k[(1, 3)], miss] == 0
    assert X[k[(1, 2)], dist] < X[k[(1, 3)], dist]
    only3 = [i for (x, y), i in k.items() if y == 4]
    assert only3 and all(X[i, miss] == 1 and X[i, dist] == 0 for i in only3)       # unsupported -> flagged, never imputed


def test_blocked_contact_pieces_are_not_candidates():
    a = walk(100, 300, 0.0, 0.0, range(0, 60))
    b = walk(100, 300, 0.0, 0.0, range(70, 130))
    obs = build({1: a, 2: b})
    tl = tracklets_of(obs)
    assert len(A.candidates(tl, deferred_cfg())) == 1
    assert A.candidates(tl, deferred_cfg(), frozenset({(1, 2)})) == []


# ---------------------------------------------------------------- CLI

def test_cli_runs_deferred_association_from_a_config_file(tmp_path):
    a = walk(100, 300, 0.0, 0.0, range(0, 60))
    b = walk(100, 300, 0.0, 0.0, range(70, 130))
    obs = build({1: a, 2: b})
    run = tmp_path / "run"
    run.mkdir()
    with (run / "frames.jsonl").open("w") as fh:
        for f in sorted({o.frame for o in obs}):
            players = [{"track_id": o.local_track_id, "detected_class": "player", "bbox_xyxy_px": list(o.box), "confidence": 0.9}
                       for o in obs if o.frame == f]
            fh.write(json.dumps({"frame_num": f, "timestamp_s": f / 25.0, "players": players}) + "\n")
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(deferred_cfg().to_dict()))
    out = tmp_path / "out"
    assert main([str(run), "--output", str(out), "--config", str(cfg)]) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["config"]["association_mode"] == "deferred" and manifest["global_ids"] == 1
    rows = [json.loads(line) for line in (out / "frames.jsonl").read_text().splitlines()]
    assert {p["association_status"] for r in rows for p in r["players"]} == {"linked_unvalidated"}
    graph = json.loads((out / "identity_graph.json").read_text())
    assert any(l.get("source") == "association" and l["decision"] == "linked" for l in graph["links"])


# ---------------------------------------------------------------- default configuration validity and A/B/C provenance

def test_deferred_mode_is_valid_with_default_thresholds():
    cfg = IdentityConfig(association_mode="deferred", association_weights=weights(), association_image_width=4096)
    assert cfg.confirm_threshold == 0.9 and 0 < cfg.effective_association_discount < cfg.confirm_threshold
    solve(build({1: walk(100, 300, 0, 0, range(0, 40))}), cfg)                         # must not raise
    assert IdentityConfig(confirm_threshold=0.98).effective_association_discount == 0.9   # the frozen M6.1 behaviour
    with pytest.raises(ValueError):
        solve(build({1: walk(100, 300, 0, 0, range(0, 40))}), IdentityConfig(
            association_mode="deferred", association_weights=weights(), association_confidence_discount=0.95))


def test_boxes_carry_local_confidence_and_link_confidence_separately():
    obs = build({1: walk(100, 300, 0.0, 0.0, range(0, 60)), 2: walk(100, 300, 0.0, 0.0, range(70, 130)),
                 3: walk(900, 700, 0.0, 0.0, range(0, 130))})
    res = solve(obs, deferred_cfg())
    assert all("identity_confidence_uncapped" in b for b in res["boxes"].values())            # A on every box
    linked = [b for b in res["boxes"].values() if b["association"] == "linked_unvalidated"]
    assert linked and all(0.0 < b["association_link_confidence"] <= 1.0 for b in linked)       # B only where a link exists
    assert all("association_link_confidence" not in b for b in res["boxes"].values() if b["association"] == "none")


def test_status_cap_keeps_local_confidence_and_only_withholds_confirmed():
    obs = build({1: walk(100, 300, 0.0, 0.0, range(0, 60)), 2: walk(100, 300, 0.0, 0.0, range(70, 130)),
                 3: walk(900, 700, 0.0, 0.0, range(0, 130))})
    kw = dict(confirm_threshold=0.6, provisional_threshold=0.4)
    res = solve(obs, deferred_cfg(association_cap="status", **kw))
    linked = [b for b in res["boxes"].values() if b["association"] == "linked_unvalidated"]
    assert linked and all(b["identity_status"] != "confirmed" for b in linked)
    assert all(b["identity_confidence"] == b["identity_confidence_uncapped"] for b in linked)
    assert any(b["identity_confidence_uncapped"] >= 0.6 and b["identity_status"] == "provisional" for b in linked)
    with pytest.raises(ValueError):
        solve(obs, deferred_cfg(association_cap="bogus"))
