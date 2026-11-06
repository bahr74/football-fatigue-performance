"""M7.3 demo layers: motion sidecar position source, measurement sigma, hand-over suggestions and decisions. SYNTHETIC data only."""

import importlib.util
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from football_fatigue.analytics import coach as K
from football_fatigue.demo import review as RV
from football_fatigue.demo import server as S
from football_fatigue.demo.physical import PhysicalIndex
_spec = importlib.util.spec_from_file_location("_m72_helpers", Path(__file__).with_name("test_demo_m72.py"))   # pytest uses importlib mode
_m72 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m72)
call, overlay = _m72.call, _m72.overlay


def sidecar(ov, shift=0.0, sig=0.2):
    c = ov["cols"]
    n = len(c["src"])
    return {"schema": "m73-motion-sidecar-v1", "match_id": ov["match_id"], "rows": n, "positions": "test",
            "cols": {"hx": [c["pxm"][i] + shift if c["src"][i] == 0 else None for i in range(n)],
                     "hy": [c["pym"][i] if c["src"][i] == 0 else None for i in range(n)],
                     "sig": [sig if c["src"][i] == 0 else None for i in range(n)]}}


def test_position_source_is_opt_in_and_gated_separately():
    noisy = overlay(jitter=0.35, seed=2)
    clean = overlay()
    side = sidecar(noisy)
    side["cols"]["hx"] = [v if v is None else clean["cols"]["pxm"][i] for i, v in enumerate(side["cols"]["hx"])]
    side["cols"]["hy"] = [v if v is None else clean["cols"]["pym"][i] for i, v in enumerate(side["cols"]["hy"])]
    P = PhysicalIndex(noisy, side)
    s = K.CoachSettings()
    stored, h1 = P.analyse(s), P.analyse(s, positions="h1")
    assert stored["query"]["positions"] == "stored" and not stored["plausibility"]["passed"] and stored["values"] == "withheld"
    assert h1["plausibility"]["passed"] and h1["plausibility"]["positions"] == "h1" and "sidecar" in h1["provenance"]["positions"]
    assert {x.get("position_sigma_m") for x in h1["segments"]} == {0.2}
    with pytest.raises(ValueError):
        P.analyse(s, positions="gps")
    bad = dict(side, rows=3)
    with pytest.raises(ValueError):
        PhysicalIndex(noisy, bad)


def test_stabiliser_settings_change_results_deterministically_on_the_same_positions():
    P = PhysicalIndex(overlay(jitter=0.3, seed=4))
    a = P.analyse(K.CoachSettings(smoothing="kalman_rts"), diagnostic=True)
    b = P.analyse(K.CoachSettings(smoothing="kalman_rts"), diagnostic=True)
    c = P.analyse(K.CoachSettings(smoothing="kalman_rts", kalman_accel_noise=150.0), diagnostic=True)
    assert a == b
    seg = lambda r: next(x for x in r["segments"] if x["id"] == 0)["metrics"]["speed"]["max_kmh"]
    assert seg(c) > seg(a)


def test_handover_decisions_are_versioned_and_conflicts_flagged():
    segs = {0: {"trk": 1, "local": 1, "f0": 0, "f1": 9}, 1: {"trk": 2, "local": 2, "f0": 12, "f1": 30}}
    ok = RV.validate_event({"action": "handover_decision", "reviewer": "a", "suggestion": "H1-2", "decision": "accept", "segments": [0, 1]}, segs)
    assert ok["decision"] == "accept"
    for bad in ({"decision": "maybe"}, {"segments": [0]}, {"segments": [0, 0]}, {"suggestion": ""}):
        e = dict({"action": "handover_decision", "reviewer": "a", "suggestion": "H1-2", "decision": "accept", "segments": [0, 1]}, **bad)
        with pytest.raises(RV.ReviewError):
            RV.validate_event(e, segs)
    ev = [dict(ok, id=1), dict(ok, id=2, reviewer="b", decision="reject")]
    st = RV.resolve(ev, segs)
    assert st["handover_decisions"]["H1-2"]["decision"] == "conflict" and st["conflicts"][0]["type"] == "contradictory_handover"
    assert st["identities"] == {}                                                   # an accepted hand-over never confirms an identity
    st = RV.resolve(ev + [{"id": 3, "action": "undo", "target_event": 2, "reviewer": "b"}], segs)
    assert st["handover_decisions"]["H1-2"]["decision"] == "accept" and not st["conflicts"]


@pytest.fixture()
def api73(tmp_path):
    ov = overlay(jitter=0.35, seed=2)
    (tmp_path / "overlay.json").write_text(json.dumps(ov))
    (tmp_path / "motion.json").write_text(json.dumps(sidecar(ov)))
    (tmp_path / "sug.json").write_text(json.dumps({"match_id": "T", "suggestions": [{"id": "H1-5", "segments": [0, 4]}]}))
    H_ = type("H", (S.DemoHandler,), {"match_id": "T", "video_path": tmp_path / "v.mp4", "overlay_path": tmp_path / "overlay.json", "selftest_dir": None,
                                      "_overlay_gz": None, "_physical": None, "review_path": tmp_path / "rev.jsonl", "preset_dir": tmp_path / "p",
                                      "motion_path": tmp_path / "motion.json", "suggestions_path": tmp_path / "sug.json"})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H_)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1], tmp_path
    srv.shutdown()


def test_api_positions_suggestions_and_handover_decision(api73):
    port, tmp = api73
    st, a = call(port, "/api/analytics", {"settings": {}, "positions": "h1"})
    assert st == 200 and a["query"]["positions"] == "h1" and a["plausibility"]["positions"] == "h1"
    assert call(port, "/api/analytics", {"settings": {}, "positions": "gps"})[0] == 400
    st, sg = call(port, "/api/suggestions")
    assert st == 200 and sg["suggestions"][0]["id"] == "H1-5"
    st, r = call(port, "/api/reviews", {"action": "handover_decision", "reviewer": "coach", "suggestion": "H1-5", "decision": "accept", "segments": [0, 4]})
    assert st == 200 and r["state"]["handover_decisions"]["H1-5"]["decision"] == "accept"
    # a sidecar or suggestion file of another match is ignored
    (tmp / "sug.json").write_text(json.dumps({"match_id": "OTHER", "suggestions": [{"id": "x"}]}))
    assert call(port, "/api/suggestions")[1]["suggestions"] == []


def test_foreign_sidecar_is_not_used(tmp_path):
    ov = overlay()
    (tmp_path / "o.json").write_text(json.dumps(ov))
    side = sidecar(ov); side["match_id"] = "OTHER"
    (tmp_path / "m.json").write_text(json.dumps(side))
    H_ = type("H", (S.DemoHandler,), {"overlay_path": tmp_path / "o.json", "motion_path": tmp_path / "m.json", "_physical": None})
    P = H_._physical_index(H_.__new__(H_))
    assert sorted(P.sources) == ["stored"] and np.isfinite(P.x).any()
