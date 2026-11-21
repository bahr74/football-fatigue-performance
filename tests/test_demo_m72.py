"""M7.2 demo layers: team assignment rules, physical-analytics service, identity review log, server API. SYNTHETIC data only."""

import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from football_fatigue.analytics import coach as K
from football_fatigue.demo import review as RV
from football_fatigue.demo import server as S
from football_fatigue.demo.physical import PhysicalIndex
from football_fatigue.demo.teams import touchline_tracklets, tw_labels, window_votes


# ---------------------------------------------------------------- teams
def test_tw_labels_need_window_and_tracklet_agreement():
    frames = list(range(100))
    votes = ["team_0"] * 60 + ["team_1"] * 40          # tracklet majority team_0 (60 %) is below 0.7 -> undecided
    assert all(lab is None for lab, _ in tw_labels(frames, votes))
    votes = ["team_0"] * 85 + ["team_1"] * 15          # decided team_0; the tail window is team_1 -> Unknown there, never team_1
    out = tw_labels(frames, votes)
    assert out[0][0] == "team_0" and out[-1][0] is None
    assert all(lab in ("team_0", None) for lab, _ in out)
    assert window_votes([0, 1, 2], ["team_0", None, "team_0"])[1] == (None, 0.0)     # fewer than 5 decided votes


def test_touchline_rule_needs_a_long_stay_on_the_line():
    line = [67.5] * 30                                  # 0.5 m from the near touchline for 30 boxes
    visit = [67.5] * 10 + [40.0] * 30                   # a throw-in visit
    short = [67.5] * 10
    assert touchline_tracklets({1: line, 2: visit, 3: short}) == {1}


# ---------------------------------------------------------------- synthetic overlay (v5-like columns)
def overlay(jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    cols = {k: [] for k in ("local", "trk", "gid", "seg", "src", "est", "pxm", "pym", "kpd", "pv", "role", "team", "tl")}
    segs, per_frame = [], {}

    def add_seg(local, frames, x0, vx, *, team=0, role=0, kpd=5.0, sid=None):
        sid = len(segs)
        segs.append({"id": sid, "local": local, "trk": local, "gid": local, "use": "attributable", "f0": frames[0], "f1": frames[-1]})
        for f in frames:
            per_frame.setdefault(f, []).append(dict(local=local, trk=local, gid=local, seg=sid, src=0, est=0,
                                                    pxm=x0 + vx * f / 25 + rng.normal(0, jitter), pym=30.0 + rng.normal(0, jitter),
                                                    kpd=kpd, pv=0, role=role, team=team, tl=0))
    add_seg(1, range(0, 250), 10.0, 5.0)                  # 10 s at 18 km/h: eligible
    add_seg(2, range(0, 50), 50.0, 3.0, team=1)           # 2 s: too short
    add_seg(3, range(0, 250), 20.0, 2.0, role=2)          # official: role blocker
    add_seg(4, range(0, 250), 30.0, 2.0, kpd=20.0)        # outside the coordinate-valid radius: coverage blocker
    add_seg(5, range(260, 510), 60.0, 4.0)                # same tracker id as seg 0? no: local 5, later in time
    per_frame.setdefault(100, []).append(dict(local=1, trk=1, gid=1, seg=-1, src=1, est=1, pxm=99.0, pym=60.0, kpd=5.0, pv=0, role=0, team=0, tl=0))
    off = [0]
    for f in range(510):
        for r in per_frame.get(f, []):
            for k in cols:
                cols[k].append(r[k])
        off.append(len(cols["local"]))
    return {"schema": "m7-demo-overlay-v5", "match_id": "T", "fps": 25.0, "n_frames": 510, "frame_offsets": off, "cols": cols, "segments": segs}


def test_physical_levels_blockers_and_values_on_clean_positions():
    P = PhysicalIndex(overlay())
    r = P.analyse(K.CoachSettings())
    lv = {x["id"]: (x["level"], x["blocker_codes"]) for x in r["segments"]}
    assert lv[0] == ("L1", []) and lv[1][1] == ["too_short"] and "role" in lv[2][1] and lv[3][1] == ["coverage"] and lv[4][0] == "L1"
    assert r["plausibility"]["passed"] and r["values"] == "valid_observed_segment_values"
    seg0 = next(x for x in r["segments"] if x["id"] == 0)
    assert seg0["metrics"]["distance_m"] == pytest.approx(5.0 * 249 / 25, rel=1e-3)      # the estimated row at frame 100 is never used
    assert seg0["metrics"]["speed"]["max_kmh"] == pytest.approx(18.0, abs=0.1)
    assert r["reviewed_identity_totals"] == {} and "speed_series_kmh" not in seg0["metrics"]
    assert any("L3" in b for b in seg0["next_level_blockers"])
    w = P.analyse(K.CoachSettings(), t0=12.0, t1=20.0)                                   # window: only the later segment
    assert [x["id"] for x in w["segments"]] == [4]


def test_noisy_positions_fail_the_plausibility_gate_and_values_are_withheld():
    P = PhysicalIndex(overlay(jitter=0.4))
    r = P.analyse(K.CoachSettings())
    assert not r["plausibility"]["passed"] and r["values"] == "withheld"
    assert all("metrics" not in x for x in r["segments"]) and r["summary"]["distance_per_observed_min_m"] is None
    d = P.analyse(K.CoachSettings(), diagnostic=True)
    assert d["values"] == "diagnostic_values_not_valid" and any("metrics" in x for x in d["segments"])


# ---------------------------------------------------------------- review log
SEGS = {0: {"trk": 1, "local": 1, "f0": 0, "f1": 249}, 1: {"trk": 2, "local": 2, "f0": 0, "f1": 49},
        4: {"trk": 5, "local": 5, "f0": 260, "f1": 509}, 7: {"trk": 1, "local": 1, "f0": 300, "f1": 400}}


def store(tmp_path):
    return RV.ReviewStore(tmp_path / "r.jsonl", "T", "m7-demo-overlay-v5")


def test_review_events_are_validated_and_carry_provenance(tmp_path):
    st = store(tmp_path)
    for bad in ({"action": "confirm", "segments": [0], "identity": "A"}, {"action": "confirm", "reviewer": "c", "segments": [99], "identity": "A"},
                {"action": "merge", "reviewer": "c", "from": "A", "into": "B"}, {"action": "split", "reviewer": "c", "trk": 1, "frame": 9999},
                {"action": "fly", "reviewer": "c"}, {"action": "undo", "reviewer": "c", "target_event": 5}):
        with pytest.raises(RV.ReviewError):
            st.append(bad, SEGS)
    e = st.append({"action": "confirm", "reviewer": "coach", "segments": [0, 4], "identity": "Blue 7"}, SEGS)
    assert e["id"] == 1 and e["reviewer"] == "coach" and e["overlay_schema"] == "m7-demo-overlay-v5" and e["time_utc"]
    s = RV.resolve(st.events(), SEGS)
    assert s["identities"] == {"Blue 7": [0, 4]} and "not model-validation evidence" in s["note"]


def test_contradictory_labels_and_same_reviewer_revision(tmp_path):
    st = store(tmp_path)
    st.append({"action": "confirm", "reviewer": "a", "segments": [0], "identity": "Blue 7"}, SEGS)
    st.append({"action": "confirm", "reviewer": "a", "segments": [0], "identity": "Blue 9"}, SEGS)      # revision by the same reviewer
    assert RV.resolve(st.events(), SEGS)["segments"]["0"]["identity"] == "Blue 9"
    st.append({"action": "confirm", "reviewer": "b", "segments": [0], "identity": "Blue 7"}, SEGS)      # a second reviewer disagrees
    s = RV.resolve(st.events(), SEGS)
    assert "0" not in s["segments"] and s["conflicts"][0]["type"] == "contradictory_labels"


def test_overlapping_identity_split_conflict_and_undo_rollback(tmp_path):
    st = store(tmp_path)
    st.append({"action": "confirm", "reviewer": "a", "segments": [0, 1], "identity": "Blue 7"}, SEGS)  # two people at the same time
    s = RV.resolve(st.events(), SEGS)
    assert any(c["type"] == "overlapping_identity" for c in s["conflicts"]) and s["identities"] == {}
    st.append({"action": "undo", "reviewer": "a", "target_event": 1}, SEGS)
    s = RV.resolve(st.events(), SEGS)
    assert s["conflicts"] == [] and s["segments"] == {} and s["events_undone"] == 1
    with pytest.raises(RV.ReviewError):
        st.append({"action": "undo", "reviewer": "a", "target_event": 1}, SEGS)
    st.append({"action": "confirm", "reviewer": "a", "segments": [0, 7], "identity": "Blue 7"}, SEGS)
    st.append({"action": "split", "reviewer": "a", "trk": 1, "frame": 280}, SEGS)                      # different people across 280
    s = RV.resolve(st.events(), SEGS)
    assert any(c["type"] == "identity_across_split" for c in s["conflicts"]) and s["identities"] == {}
    lines = (tmp_path / "r.jsonl").read_text().splitlines()
    assert len(lines) == 4 and json.loads(lines[0])["action"] == "confirm"                             # append-only: nothing rewritten


def test_merge_needs_explicit_review_and_enables_l2_totals(tmp_path):
    st = store(tmp_path)
    st.append({"action": "confirm", "reviewer": "a", "segments": [0], "identity": "Blue 7"}, SEGS)
    st.append({"action": "confirm", "reviewer": "a", "segments": [4], "identity": "B7"}, SEGS)
    st.append({"action": "merge", "reviewer": "a", "from": "B7", "into": "Blue 7", "explicit": True}, SEGS)
    s = RV.resolve(st.events(), SEGS)
    assert s["identities"] == {"Blue 7": [0, 4]}
    r = PhysicalIndex(overlay()).analyse(K.CoachSettings(), review_state=s)
    assert {x["id"]: x["level"] for x in r["segments"]}[0] == "L2"
    tot = r["reviewed_identity_totals"]["Blue 7"]
    assert tot["segments"] == 2 and "extrapolated" in tot["note"]


# ---------------------------------------------------------------- server API
@pytest.fixture()
def api(tmp_path):
    ov = tmp_path / "overlay.json"
    ov.write_text(json.dumps(overlay()))
    H_ = type("H", (S.DemoHandler,), {"match_id": "T", "video_path": tmp_path / "v.mp4", "overlay_path": ov, "selftest_dir": None,
                                      "_overlay_gz": None, "_physical": None, "review_path": tmp_path / "rev.jsonl", "preset_dir": tmp_path / "presets"})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H_)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1], tmp_path
    srv.shutdown()


def call(port, path, body=None):
    c = HTTPConnection("127.0.0.1", port, timeout=20)
    c.request("POST" if body is not None else "GET", path, body=None if body is None else json.dumps(body).encode())
    r = c.getresponse()
    return r.status, json.loads(r.read() or b"null")


def test_api_analytics_validates_and_recalculates(api):
    port, _ = api
    st, out = call(port, "/api/analytics", {"settings": {"hsr_kmh": 99}})
    assert st == 400 and any("outside" in e for e in out["errors"])
    st, out = call(port, "/api/analytics", {"settings": {}, "local": "1"})
    assert st == 400
    st, a = call(port, "/api/analytics", {"settings": {}, "local": 1})
    assert st == 200 and a["values"] == "valid_observed_segment_values" and "speed_series_kmh" in a["segments"][0]["metrics"]
    st, b = call(port, "/api/analytics", {"settings": {"hsr_kmh": 17.0, "run_kmh": 14.4}, "local": 1})
    assert b["segments"][0]["metrics"]["hsr_distance_m"] > a["segments"][0]["metrics"]["hsr_distance_m"] == 0
    assert a["provenance"]["match_id"] == "T" and a["settings"]["hsr_kmh"] == 19.8


def test_api_presets_and_reviews(api):
    port, tmp = api
    st, p = call(port, "/api/presets")
    assert st == 200 and set(p["builtin"]) == set(K.PRESETS) and p["saved"] == {}
    assert call(port, "/api/presets", {"name": "example_adult_common", "settings": {}})[0] == 400
    assert call(port, "/api/presets", {"name": "../x", "settings": {}})[0] == 400
    st, _ = call(port, "/api/presets", {"name": "U19 squad", "settings": {"hsr_kmh": 18.0}})
    st, _ = call(port, "/api/presets", {"name": "U19 squad", "settings": {"hsr_kmh": 18.5}})
    saved = call(port, "/api/presets")[1]["saved"]["U19 squad"]
    assert saved["settings"]["hsr_kmh"] == 18.5 and saved["history"][0]["settings"]["hsr_kmh"] == 18.0
    assert call(port, "/api/reviews", {"action": "confirm", "segments": [0], "identity": "A"})[0] == 400      # no reviewer
    st, r = call(port, "/api/reviews", {"action": "confirm", "reviewer": "coach", "segments": [0], "identity": "Blue 7"})
    assert st == 200 and r["state"]["identities"] == {"Blue 7": [0]}
    st, a = call(port, "/api/analytics", {"settings": {}, "local": 1})
    assert a["segments"][0]["level"] == "L2" and a["segments"][0]["reviewed_identity"] == "Blue 7"
    assert len(call(port, "/api/reviews")[1]["events"]) == 1
