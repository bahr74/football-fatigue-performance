"""Viewer repair, T0 tracker-input profile and native player (Session B).

Unit tests run anywhere. Tests marked ``needs_data`` use FOOTBALL_FATIGUE_DATA_ROOT (development SoccerTrack data) and
the saved Session B outputs under data/processed/video.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / "data" / "processed" / "video"
DATA_ROOT = os.environ.get("FOOTBALL_FATIGUE_DATA_ROOT")
needs_data = pytest.mark.skipif(
    not DATA_ROOT or not (Path(DATA_ROOT or ".") / "external" / "soccertrack-v2" / "raw" / "117093").is_dir()
    or not (OUT_ROOT / "soccertrack-117093" / "tracking" / "T0-tracker-input-v1" / "frames.jsonl").is_file(),
    reason="needs FOOTBALL_FATIGUE_DATA_ROOT and the saved T0 tracks")
APP = REPO / "src" / "football_fatigue" / "video" / "dashboard" / "app.py"


# ------------------------------------------------------------------------------------------- Streamlit root cause
def test_root_cause_writing_a_widget_key_after_the_widget_exists_raises():
    """The old viewer did exactly this (slider key "frame", then ``st.session_state.frame = nxt`` in the play loop)."""
    from streamlit.testing.v1 import AppTest

    old = AppTest.from_string(
        "import streamlit as st\n"
        "st.session_state.setdefault('frame', 0)\n"
        "st.slider('Frame', 0, 100, key='frame')\n"
        "st.session_state.frame = st.session_state.frame + 1\n")
    old.run()
    assert old.exception and "cannot be modified after the widget" in old.exception[0].message


def test_fixed_pattern_internal_key_synced_before_the_widget():
    from streamlit.testing.v1 import AppTest

    new = AppTest.from_string(
        "import streamlit as st\n"
        "st.session_state.setdefault('cur', 0)\n"
        "def step(n): st.session_state.cur = min(100, max(0, st.session_state.cur + n))\n"
        "def changed(): st.session_state.cur = st.session_state.frame_slider\n"
        "st.button('+1', on_click=step, args=(1,), key='p1')\n"
        "st.session_state.frame_slider = st.session_state.cur\n"
        "st.slider('Frame', 0, 100, key='frame_slider', on_change=changed)\n"
        "st.session_state.cur = st.session_state.cur   # later writes go to the internal key only\n")
    new.run()
    new.button(key="p1").click().run()
    assert not new.exception and new.session_state["cur"] == 1 and new.slider(key="frame_slider").value == 1
    new.slider(key="frame_slider").set_value(40).run()
    assert not new.exception and new.session_state["cur"] == 40
    new.button(key="p1").click().run()
    assert new.session_state["cur"] == 41 and new.slider(key="frame_slider").value == 41


def test_app_never_assigns_widget_keys_after_creation():
    src = APP.read_text(encoding="utf-8")
    assert "st.session_state.frame =" not in src and "st.session_state.zoom_mode" not in src
    assert "run_every" not in src, "no rerun-based play loop in the Streamlit viewer"


# ------------------------------------------------------------------------------------------- tracker input profiles
def _obs(mapped="player", turf=True, pitch=None):
    from football_fatigue.vision.models import DetectedClass

    o = SimpleNamespace(mapped_class=DetectedClass(mapped) if mapped else None, passes_turf_filter=turf,
                        inside_pitch_mask=pitch)
    o.kept = o.mapped_class is not None and turf is not False and pitch is not False
    return o


def test_tracker_input_profiles_differ_only_in_the_turf_decision():
    from football_fatigue.video.tracking import TRACKER_INPUTS, accepts, check_name
    from football_fatigue.video.sources import SourceError

    b0, t0 = TRACKER_INPUTS["B0"], TRACKER_INPUTS["T0"]
    assert (b0.threshold, b0.turf, t0.threshold, t0.turf) == (0.4, "cached", 0.4, "ignore")
    assert accepts(_obs(), "B0") and accepts(_obs(), "T0")
    assert not accepts(_obs(turf=False), "B0") and accepts(_obs(turf=False), "T0")
    assert not accepts(_obs(mapped="ball"), "T0") and not accepts(_obs(mapped=None), "T0")
    assert not accepts(_obs(pitch=False), "T0")
    with pytest.raises(SourceError):
        check_name("T9")


def test_engine_cli_refuses_mixing_a_production_replay_with_t0(tmp_path):
    from football_fatigue.video.occlusion.__main__ import main

    with pytest.raises(SystemExit, match="cannot be combined"):
        main(["run", "--data-root", str(tmp_path / "data"), "--out-root", str(tmp_path / "out"), "--source", "soccertrack-117093",
              "--tracker-input", "T0", "--replay-run", "something"])


# ------------------------------------------------------------------------------------------- real data
@needs_data
def test_t0_tracks_manifest_and_equality_with_the_scored_replay():
    for sid in ("soccertrack-117093", "soccertrack-118575", "soccertrack-117092", "soccertrack-118576"):
        m = json.loads((OUT_ROOT / sid / "tracking" / "T0-tracker-input-v1" / "manifest.json").read_text())
        assert m["tracker_input"]["name"] == "T0" and m["configuration"]["turf_filter"] == "ignored"
        assert m["configuration"]["confidence_threshold"] == 0.4
        assert m["configuration"]["tracker_params"] == {"activation": 0.5, "buffer_frames": 75, "matching_threshold": 0.95}
        assert m["checks"]["diagnostic_replay_equal"]["identical"] is True


@needs_data
def test_engine_runs_t0_window_and_b0_window_reproduces_saved_run(tmp_path):
    from football_fatigue.video.occlusion.__main__ import main

    root = tmp_path / "out"
    # T0 tracks live under <out_root>/<source>/tracking: link the saved ones into the temporary out root
    (root / "soccertrack-117093" / "tracking").mkdir(parents=True)
    import shutil

    shutil.copytree(OUT_ROOT / "soccertrack-117093" / "tracking" / "T0-tracker-input-v1",
                    root / "soccertrack-117093" / "tracking" / "T0-tracker-input-v1")
    args = ["run", "--data-root", DATA_ROOT, "--out-root", str(root), "--source", "soccertrack-117093",
            "--start", "3000", "--end", "3499", "--no-render"]
    assert main(args + ["--tracker-input", "T0"]) == 0
    t0 = root / "soccertrack-117093" / "occlusion" / "0.2.0" / "occl-soccertrack-117093-3000-3499-v2-T0-ff8815ed"
    m = json.loads((t0 / "manifest.json").read_text())
    assert m["tracker_input"] == "T0" and m["replay_run"] is None and m["tracks_dir"].endswith("T0-tracker-input-v1")
    assert m["provenance"]["tracker_input"] == "T0"
    assert main(["evaluate", "--data-root", DATA_ROOT, "--run-dir", str(t0)]) == 0
    assert main(args) == 0                                   # B0 (default)
    b0 = root / "soccertrack-117093" / "occlusion" / "0.2.0" / "occl-soccertrack-117093-3000-3499-v2-ff8815ed"
    saved = OUT_ROOT / "soccertrack-117093" / "occlusion" / "0.2.0" / "occl-soccertrack-117093-3000-3499-v2-ff8815ed"
    assert (b0 / "events.jsonl").read_bytes() == (saved / "events.jsonl").read_bytes()
    assert (b0 / "frames.jsonl").read_bytes() == (saved / "frames.jsonl").read_bytes()


@needs_data
def test_comparison_reads_saved_results_consistently():
    from football_fatigue.video.player.compare import compare_source

    c = compare_source(OUT_ROOT, "soccertrack-118576")
    for ti in ("B0", "T0"):
        r = json.loads((OUT_ROOT / "soccertrack-118576" / "detection" / "replay-v1" / ti / "result.json").read_text())
        assert c["profiles"][ti]["tracking"]["HOTA"] == r["metrics_all_tracks"]["hota"]["HOTA"]
        assert c["profiles"][ti]["engine"]["eval"]["po_events"] > 0
    assert c["profiles"]["T0"]["engine"]["run_id"].endswith("-v2-T0-ff8815ed")
    assert c["profiles"]["B0"]["engine"]["run_id"] == "occl-soccertrack-118576-0-5974-v2-ff8815ed"


@pytest.fixture(scope="module")
def player():
    if not DATA_ROOT:
        pytest.skip("needs FOOTBALL_FATIGUE_DATA_ROOT")
    from football_fatigue.video.player.server import start

    srv = start(Path(DATA_ROOT), OUT_ROOT, port=5893)
    yield "http://127.0.0.1:5893"
    srv.shutdown()


def _get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


@needs_data
def test_player_layers_are_separate_and_flag_the_turf_filter(player):
    run = "occl-soccertrack-117093-0-5975-v2-T0-ff8815ed"
    s, _, b = _get(f"{player}/api/frames?source=soccertrack-117093&start=1585&n=1&tracker=T0&run={run}")
    fr = json.loads(b)["frames"]["1585"]
    assert {"d", "t", "g", "e"} <= set(fr) <= {"d", "t", "g", "e", "o"}     # o = debugger pipeline outcome per GT player
    assert all(len(x) == 4 for x in fr.get("o", []))
    turf_rejected = [d for d in fr["d"] if d[6] & 2 and d[4] > 0.4]
    assert turf_rejected and all(d[6] & 1 for d in turf_rejected)          # T0 passes them to the tracker
    s, _, b = _get(f"{player}/api/frames?source=soccertrack-117093&start=1585&n=1&tracker=B0")
    fb = json.loads(b)["frames"]["1585"]
    assert all(not d[6] & 1 for d in fb["d"] if d[6] & 2)                  # B0 does not
    assert "e" not in fb                                                    # no run: no engine predictions
    inputs = {tuple(d[:4]) for d in fr["d"] if d[6] & 1}
    assert all(any(max(abs(a - b) for a, b in zip(t[1:5], i)) <= 0.11 for i in inputs) for t in fr["t"])  # T## = detections
    assert all(p[1] in ("hidden_predicted", "lost_predicted") for p in fr["e"])


@needs_data
def test_player_refuses_bad_inputs_and_serves_ranges(player):
    assert _get(f"{player}/api/meta?source=soccertrack-118577")[0] == 400
    assert _get(f"{player}/media/soccertrack-128058")[0] == 400
    s, _, b = _get(f"{player}/api/frames?source=soccertrack-117093&start=0&n=1&tracker=B0"
                   f"&run=occl-soccertrack-117093-0-5975-v2-T0-ff8815ed")
    assert s == 400 and b"analysed tracker input T0" in b
    s, h, b = _get(f"{player}/media/soccertrack-117093", {"Range": "bytes=100-199"})
    assert s == 206 and h["Content-Range"].startswith("bytes 100-199/") and len(b) == 100
    s, _, b = _get(f"{player}/api/meta?source=soccertrack-117093")
    m = json.loads(b)
    assert (m["width"], m["height"], m["fps"], m["frames"]) == (4096, 1080, 25.0, 5976)
    assert m["first_pts_s"] == pytest.approx(0.979)


@needs_data
def test_dashboard_state_flow_has_no_widget_exceptions(monkeypatch):
    """The reported crash paths: stepping, event navigation, follow event, manual region, slider, run/source switch."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr("sys.argv", ["app.py", "--data-root", DATA_ROOT, "--out-root", str(OUT_ROOT)])
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    assert not at.exception
    lo = at.session_state["cur"]
    at.radio(key="view_radio").set_value("Follow selected event").run()   # no event selected yet
    assert not at.exception
    for label in ("next event ▶", "+1 ▶", "+25 ⏭", "◀ −1", "⏮ −25", "next event ▶", "◀ prev event"):
        next(b for b in at.button if b.label == label).click().run()
        assert not at.exception, label
        assert at.slider(key="frame_slider").value == at.session_state["cur"]
    assert at.session_state["selected"] is not None and at.session_state["view_mode"] == "Follow selected event"
    at.slider(key="frame_slider").set_value(lo + 300).run()
    assert not at.exception and at.session_state["cur"] == lo + 300
    at.radio(key="view_radio").set_value("Manual region").run()
    assert not at.exception
    next(b for b in at.button if b.label == "⏮ −25").click().run()
    assert not at.exception and at.session_state["cur"] == lo + 275
    at.number_input(key="goto_frame").set_value(lo).run()
    assert not at.exception and at.session_state["cur"] == lo
    next(b for b in at.button if b.label == "⏮ −25").click().run()           # clamped at the first frame
    assert at.session_state["cur"] == lo
    run_box = next(s for s in at.sidebar.selectbox if s.label == "Run")
    t0 = next(o for o in run_box.options if o.startswith("T0 · v2"))
    run_box.set_value(t0).run()
    assert not at.exception and at.session_state["selected"] is None
    src_box = next(s for s in at.sidebar.selectbox if s.label == "Match / source")
    src_box.set_value("soccertrack-118576").run()
    assert not at.exception
    next(b for b in at.button if b.label == "next event ▶").click().run()
    assert not at.exception and at.session_state["selected"] is not None
