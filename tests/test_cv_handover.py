"""M7.3 opt-in short-gap hand-over re-linking (football_fatigue.vision.handover) on synthetic solver outputs."""
from football_fatigue.vision import handover as H
from football_fatigue.vision.identity import IdentityConfig, Observation

CFG = IdentityConfig()


def chain(gid, f0, f1, x0, vx, local, team="team_0", status="provisional", app=(10.0,) * 6, y=400.0, h=60.0):
    obs, boxes = [], {}
    for k, f in enumerate(range(f0, f1 + 1)):
        x = x0 + vx * k
        o = Observation(f, gid, local, (x - 10, y - h, x + 10, y), "player", None, app)
        obs.append(o)
        boxes[(f, gid)] = {"global_id": gid, "tracklet_id": gid, "local_track_id": local, "identity_status": status, "team_cluster": team}
    return obs, boxes


def build(*chains):
    obs, boxes = [], {}
    for o, b in chains:
        obs += o; boxes.update(b)
    return obs, boxes


def test_clean_handover_is_linked_and_capped():
    obs, boxes = build(chain(1, 0, 30, 100, 2.0, 1), chain(2, 34, 60, 100 + 2.0 * 34, 2.0, 2, status="confirmed"))
    cfg = H.HandoverConfig()
    dec = H.decide(H.candidates(boxes, obs, CFG, cfg), cfg)
    assert [d["decision"] for d in dec] == ["linked"]
    new, mapping = H.apply(boxes, dec, cfg)
    assert mapping == {2: 1}
    assert all(v["global_id"] == 1 for v in new.values())
    assert {v["identity_status"] for k, v in new.items() if boxes[k]["global_id"] == 2} == {"provisional"}   # join never confirms
    assert {v["identity_status"] for k, v in new.items() if boxes[k]["global_id"] == 1} == {"provisional"}
    assert boxes[(40, 2)]["global_id"] == 2                                                                   # input not mutated


def test_competing_start_makes_it_abstain():
    obs, boxes = build(chain(1, 0, 30, 100, 2.0, 1), chain(2, 34, 60, 168, 2.0, 2), chain(3, 34, 60, 171, 2.0, 3))
    cfg = H.HandoverConfig()
    dec = H.decide(H.candidates(boxes, obs, CFG, cfg), cfg)
    assert not any(d["decision"] == "linked" for d in dec)
    assert {d["reason"] for d in dec} & {"competing_candidate", "not_mutual_best"}


def test_team_conflict_long_gap_overlap_and_far_jump_are_never_candidates():
    cfg = H.HandoverConfig()
    for other in (chain(2, 34, 60, 168, 2.0, 2, team="team_1"),     # other team
                  chain(2, 70, 90, 220, 2.0, 2),                     # gap 39 > 25 frames
                  chain(2, 20, 60, 140, 2.0, 2),                     # overlaps chain 1 in time
                  chain(2, 34, 60, 900, 2.0, 2)):                    # far away
        obs, boxes = build(chain(1, 0, 30, 100, 2.0, 1), other)
        assert H.candidates(boxes, obs, CFG, cfg) == []


def test_appearance_gate_and_uncapped_variant():
    obs, boxes = build(chain(1, 0, 30, 100, 2.0, 1), chain(2, 34, 60, 168, 2.0, 2, status="confirmed", app=(90.0,) * 6))
    gated = H.HandoverConfig(max_appearance=30.0)
    dec = H.decide(H.candidates(boxes, obs, CFG, gated), gated)
    assert [d["reason"] for d in dec] == ["appearance"]
    loose = H.HandoverConfig(cap_status=False)
    new, _ = H.apply(boxes, H.decide(H.candidates(boxes, obs, CFG, loose), loose), loose)
    assert new[(40, 2)]["identity_status"] == "confirmed" and new[(40, 2)]["global_id"] == 1
