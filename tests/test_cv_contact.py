"""Tests for the opt-in M6.2 contact-aware keep / swap / ambiguous decisions. All scenarios are SYNTHETIC (hand-built boxes)."""

import json

import numpy as np

from football_fatigue.vision.identity import (
    CONTACT_FEATURES,
    IdentityConfig,
    Observation,
    contact_episodes,
    contact_feature_vector,
    decide_contacts,
    interaction_partners,
    solve,
)

H = 40.0
N = len(CONTACT_FEATURES)


def box(x, y=100.0, w=16.0, h=H):
    return (x - w / 2, y - h, x + w / 2, y)


def crossing(swap: bool, frames=range(0, 61), desc=None):
    """Two players walk towards each other along one line and pass at frame 30 (contact for frames 24..36).

    Track 1 starts on the left moving right, track 2 starts on the right moving left. With ``swap`` the tracker 'bounces':
    after the contact track 1 continues on the left-moving path and track 2 on the right-moving path.
    """
    obs = []
    for f in frames:
        a, b = 20.0 + 2 * f, 140.0 - 2 * f           # true positions of players A (moves right) and B (moves left)
        post = f > 30
        p1, p2 = (b, a) if (swap and post) else (a, b)
        pair = sorted([(1, p1), (2, p2)], key=lambda t: t[1])  # frame-local index by x
        for i, (tid, x) in enumerate(pair):
            obs.append(Observation(f, i, tid, box(x), "player", None, desc(tid, f, swap) if desc else None))
    return obs


def swap_weights(intercept=-3.0, motion=3.0, keep_motion=1.0):
    w = np.zeros(N + 1)
    w[0] = intercept
    w[1 + CONTACT_FEATURES.index("motion_pref_swap")] = motion
    w[1 + CONTACT_FEATURES.index("log_r_keep")] = keep_motion
    return tuple(w.tolist())


def cfg(**kw):
    base = dict(split_on_motion=False, confidence_features="m61", min_tracklet_frames=5)
    base.update(kw)
    return IdentityConfig(**base)


def test_defaults_are_off_and_leave_m61_untouched():
    c = IdentityConfig()
    assert c.contact_mode == "off" and c.contact_weights is None and c.contact_swap_above == 1.0 and c.contact_keep_below == 0.0
    obs = crossing(swap=True)
    res = solve(obs, cfg())
    assert res["contacts"] == []
    res2 = solve(obs, cfg(contact_window=7, contact_min_side=3))      # unused knobs do not change anything
    assert {k: (v["global_id"], v["identity_status"], v["identity_confidence"]) for k, v in res["boxes"].items()} == \
           {k: (v["global_id"], v["identity_status"], v["identity_confidence"]) for k, v in res2["boxes"].items()}


def test_config_roundtrip_keeps_contact_weights_as_tuple():
    c = cfg(contact_mode="evidence", contact_weights=swap_weights(), contact_swap_above=0.7)
    d = c.to_dict()
    c2 = IdentityConfig.from_dict(json.loads(json.dumps(d)))
    assert c2.contact_weights == c.contact_weights and isinstance(c2.contact_weights, tuple) and c2 == c


def test_episodes_have_evidence_and_no_reference_inputs():
    obs = crossing(swap=True)
    by_frame = {}
    tracks = {}
    for o in obs:
        by_frame.setdefault(o.frame, []).append(o)
        tracks.setdefault(o.local_track_id, []).append(o)
    eps = contact_episodes(tracks, interaction_partners(by_frame, IdentityConfig()), IdentityConfig())
    assert {e["local_track_id"] for e in eps} == {1, 2}
    e = next(e for e in eps if e["local_track_id"] == 1)
    assert e["decidable"] and e["start"] >= 24 and e["end"] <= 36 and e["mate"] == 2
    f = contact_feature_vector(e)
    assert f.shape == (N,)
    # the bounce is visible in the motion evidence: the partner continues the pre-contact path, the same track does not
    assert e["r_swap"] is not None and e["r_swap"] < e["r_keep"]
    assert f[CONTACT_FEATURES.index("motion_pref_swap")] > 0


def test_swap_decision_splits_the_track_and_blocks_relinking():
    obs = crossing(swap=True)
    off = solve(obs, cfg())
    ids_off = {(o.local_track_id, o.frame): off["boxes"][(o.frame, o.index)]["global_id"] for o in obs}
    assert ids_off[(1, 10)] == ids_off[(1, 50)]                      # without contact decisions the bounce stays one identity

    on = solve(obs, cfg(contact_mode="evidence", contact_weights=swap_weights(), contact_swap_above=0.6))
    swaps = [c for c in on["contacts"] if c["decision"] == "swap"]
    assert {c["local_track_id"] for c in swaps} == {1, 2} and all(c["p_swap"] >= 0.6 for c in swaps)
    ids_on = {(o.local_track_id, o.frame): on["boxes"][(o.frame, o.index)]["global_id"] for o in obs}
    assert ids_on[(1, 10)] != ids_on[(1, 50)]                        # pieces are different identities ...
    assert ids_on[(2, 10)] != ids_on[(2, 50)]
    reasons = {l.get("reason") for l in on["links"]}
    assert "contact_swap_split" in reasons                           # ... because the relink was explicitly blocked
    # every box is accounted for exactly once and global ids stay unique per frame
    assert len(on["boxes"]) == len(obs)
    per = {}
    for (f, _), b in on["boxes"].items():
        per.setdefault(f, []).append(b["global_id"])
    assert all(len(v) == len(set(v)) for v in per.values())


def test_no_weights_means_ambiguous_and_never_splits():
    obs = crossing(swap=True)
    res = solve(obs, cfg(contact_mode="evidence", contact_weights=None, contact_swap_above=0.1, contact_keep_below=0.9))
    assert res["contacts"] and all(c["decision"] == "ambiguous" and c["p_swap"] is None for c in res["contacts"])
    assert not any(l.get("reason") == "contact_swap_split" for l in res["links"])


def test_keep_decision_verifies_the_episode_for_the_segment_features():
    obs = crossing(swap=False)
    unverified = 15                                                 # index of log1p_unverified_episodes in the m61 feature vector
    base = solve(obs, cfg())
    keep = solve(obs, cfg(contact_mode="evidence", contact_weights=swap_weights(intercept=-8.0, motion=0.0, keep_motion=0.0),
                          contact_keep_below=0.05, contact_swap_above=0.9))
    assert any(c["decision"] == "keep" for c in keep["contacts"])
    k_base = base["boxes"][(5, 0)]["features"][unverified]
    k_keep = keep["boxes"][(5, 0)]["features"][unverified]
    assert k_base > 0 and k_keep < k_base                           # unverified contact episodes were removed by the keep decisions
    assert not any(l.get("reason") == "contact_swap_split" for l in keep["links"])


def test_episodes_without_enough_isolated_boxes_stay_undecidable():
    obs = crossing(swap=True, frames=range(0, 40))                  # clip ends 3 frames after the contact: < 5 boxes after
    res = solve(obs, cfg(contact_mode="evidence", contact_weights=swap_weights(), contact_swap_above=0.1))
    assert res["contacts"] and all(not c["decidable"] and c["decision"] == "ambiguous" for c in res["contacts"])


def test_external_decisions_override_evidence_and_are_labelled():
    obs = crossing(swap=True)
    probe = solve(obs, cfg(contact_mode="evidence"))                # episodes only (no weights -> all ambiguous)
    key = next((c["local_track_id"], c["start"], c["end"]) for c in probe["contacts"] if c["local_track_id"] == 1)
    res = solve(obs, cfg(), {key: "swap"})
    assert [c["source"] for c in res["contacts"] if (c["local_track_id"], c["start"], c["end"]) == key] == ["external"]
    ids = {(o.local_track_id, o.frame): res["boxes"][(o.frame, o.index)]["global_id"] for o in obs}
    assert ids[(1, 10)] != ids[(1, 50)]


def test_appearance_descriptor_enters_the_evidence_but_not_observation_identity():
    desc = lambda tid, f, swap: (50.0, 10.0, 10.0) if (tid == 1) == (not (swap and f > 30)) else (50.0, -10.0, 5.0)
    a = crossing(swap=True, desc=desc)
    b = crossing(swap=True)
    assert a[0] == b[0]                                             # descriptor does not take part in Observation equality
    by_frame, tracks = {}, {}
    for o in a:
        by_frame.setdefault(o.frame, []).append(o)
        tracks.setdefault(o.local_track_id, []).append(o)
    eps = contact_episodes(tracks, interaction_partners(by_frame, IdentityConfig()), IdentityConfig())
    e = next(e for e in eps if e["local_track_id"] == 1)
    assert e["d_keep"] is not None and e["d_swap"] is not None and e["d_swap"] < e["d_keep"]
