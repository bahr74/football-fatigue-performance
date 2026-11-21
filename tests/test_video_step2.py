"""Tests for Session B Step 2: engine profile v2, evaluation, overlays and dashboard state logic.

Unit tests use SYNTHETIC tracks and synthetic ground truth. Real-data tests run only when
FOOTBALL_FATIGUE_DATA_ROOT points at a data root holding the 117093 cache and replay.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from football_fatigue.video import overlay as ov
from football_fatigue.video.dashboard import data as D
from football_fatigue.video.dashboard import state as S
from football_fatigue.video.occlusion import evaluate as E
from football_fatigue.video.occlusion import geometry as g
from football_fatigue.video.occlusion.engine import PROFILES, EngineConfig, FrameInput, OcclusionEngine, TrackObs
from football_fatigue.video.schema import SourceInfo, event_from_json
from football_fatigue.video.sources import play_area_polygon

SRC = SourceInfo("synthetic-src", "synthetic", "cam0", "0" * 64, 25.0, 0.5, 1000, 500)
V2 = PROFILES["v2"]


def box(cx, bottom, w=20.0, h=50.0):
    return (cx - w / 2, bottom - h, cx + w / 2, bottom)


def frames_from(n, tracks_at, cls="player"):
    return [FrameInput(f, {t: TrackObs(b, c, cls) for t, (b, c) in tracks_at(f).items()}, ()) for f in range(n)]


def run(frames, config=None, play_area=None):
    return OcclusionEngine(config).run(frames, SRC, "run-test", (frames[0].frame, frames[-1].frame), {"engine": "t"},
                                       play_area=play_area)


def as_dicts(events):
    return [json.loads(e.to_json()) for e in events]


# ------------------------------------------------------------------------------------------- engine v2
def test_profiles_step1_is_the_default_config():
    assert PROFILES["step1"] == EngineConfig()
    assert (V2.grouping, V2.prediction, V2.use_play_area) == ("pair", "damped", True)


def test_pair_grouping_splits_a_chained_group_into_linked_pair_events():
    # 1-2 and 2-3 overlap (1 and 3 do not): step1 gives one 3-member event, v2 two pair events linked by group_id
    def at(f):
        dx = 14 if 10 <= f <= 20 else 60
        return {1: (box(500, 300), 0.9), 2: (box(500 + dx, 301), 0.9), 3: (box(500 + 2 * dx, 302), 0.9)}
    fr = frames_from(40, at)
    step1 = run(fr)
    v2 = run(fr, V2)
    assert len(step1.events) == 1 and len(step1.events[0].members) == 3
    assert sorted(tuple(sorted(m.track_id for m in e.members)) for e in v2.events) == [(1, 2), (2, 3)]
    assert v2.events[0].group_id is not None and v2.events[0].group_id == v2.events[1].group_id
    assert all(e.measurements["group_tracks"] == 3 for e in v2.events)


def test_damped_prediction_is_capped_and_held():
    hist = [(f, (100.0 + 40 * f, 0.0, 120.0 + 40 * f, 50.0)) for f in range(6)]   # 40 px/frame, box height 50
    cv = g.predict_constant_velocity(hist, 5 + 30)
    damped = g.predict_damped(hist, 5 + 30, full_frames=5, max_step_heights=0.25)
    assert cv[0] == pytest.approx(300 + 40 * 30)            # unbounded
    assert damped[0] == pytest.approx(300 + 12.5 * 5)       # 0.25 * 50 px/frame for 5 frames, then held
    assert g.predict_damped(hist, 5 + 100) == damped


def test_play_area_suppresses_off_pitch_events_and_keeps_them_for_review():
    # two people overlapping far outside the polygon (bench), two players overlapping inside
    def at(f):
        dx = 12 if 10 <= f <= 20 else 60
        return {1: (box(500, 300), 0.9), 2: (box(500 + dx, 302), 0.9),
                7: (box(100, 480), 0.9), 8: (box(100 + dx, 482), 0.9)}
    poly = [[200, 150], [900, 150], [900, 400], [200, 400]]
    r = run(frames_from(40, at), V2, play_area=poly)
    assert [sorted(m.track_id for m in e.members) for e in r.events] == [[1, 2]]
    assert [sorted(m.track_id for m in e.members) for e in r.suppressed] == [[7, 8]]
    assert r.suppressed[0].suppressed_reason and "off_pitch_member" in r.suppressed[0].flags
    assert all(row["event_id"] != r.suppressed[0].event_id for row in r.frame_rows)
    assert {row["event_id"] for row in r.suppressed_rows} == {r.suppressed[0].event_id}
    assert r.stats["suppressed_off_pitch_events"] == 1


def test_play_area_missing_is_reported_not_guessed():
    def at(f):
        return {1: (box(500, 300), 0.9)}
    r = run(frames_from(10, at), V2, play_area=None)
    assert r.stats["play_area"] == {"status": "unavailable"}


def test_play_area_polygon_from_keypoints(tmp_path):
    kp = {f"({x},{y})": [10 * x + 5, 5 * y + 7] for x in (0, 52.5, 105) for y in (0, 34, 68)}
    p = tmp_path / "kp.json"
    p.write_text(json.dumps(kp))
    poly = play_area_polygon(p)
    assert poly[0] == [5.0, 7.0]                      # (0, 0) first: far touchline left to right
    assert [1055.0, 7.0] in poly and [5.0, 347.0] in poly
    assert len(poly) == 8                             # boundary points only, the centre (52.5, 34) is excluded


def test_step1_profile_reproduces_step1_events_exactly():
    def at(f):
        out = {1: (box(500, 320), 0.9)}
        if not 10 <= f <= 16:
            out[2] = (box(440 + 4 * f, 300), 0.9)
        return out
    fr = frames_from(40, at)
    a = as_dicts(run(fr).events)
    b = as_dicts(run(fr, EngineConfig()).events)
    assert a == b and a


# ------------------------------------------------------------------------------------------- evaluation
def gt_from(n, at):
    return {f: [(gid, b) for gid, b in at(f).items()] for f in range(n)}


def test_gt_episodes_tiers_and_merging():
    def at(f):
        # player 2 behind player 1: covered 0.6 for frames 5-14 (a 1-frame dip at 9 is merged), then 0.3 for 15-18
        if 5 <= f <= 14:
            dx = 4 if f != 9 else 30
        elif 15 <= f <= 18:
            dx = 14
        else:
            dx = 80
        return {1: box(500, 320), 2: box(500 + dx, 310)}
    eps = E.gt_episodes(gt_from(30, at), 0, 29)
    occ = [e for e in eps if e.tier == "occlusion"]
    ovl = [e for e in eps if e.tier == "overlap"]
    assert [(e.start, e.end) for e in occ] == [(5, 14)]
    assert [(e.start, e.end) for e in ovl] == [(5, 18)]


def _scored(events, rows, tracks, gt, n):
    return E.evaluate_run(as_dicts(events), [], rows, tracks, gt, (0, n - 1))


def test_evaluation_true_positive_and_timing():
    # track 2 walks behind track 1 and is not detected while behind it (frames 13-18)
    def at(f):
        out = {1: (box(500, 320), 0.9)}
        if not 13 <= f <= 18:
            out[2] = (box(440 + 4 * f, 310), 0.9)
        return out
    n = 40
    fr = frames_from(n, at)
    r = run(fr, V2)
    tracks = {f.frame: {t: o.box for t, o in f.tracks.items()} for f in fr}
    gt = gt_from(n, lambda f: {11: box(500, 320), 22: box(440 + 4 * f, 310)})
    res = _scored(r.events, r.frame_rows, tracks, gt, n)
    po = [e for e in res["events"] if e["event_class"] == "probable_occlusion"]
    assert po and po[0]["outcome"] == "tp"
    assert po[0]["member_gt"] == {"1": 11, "2": 22}
    assert res["summary"]["probable_occlusion"]["vs_detectable_occlusion_episodes"]["recall"] == 1.0
    assert res["summary"]["pair_claims"]["occlusion"]["precision"] == 1.0


def test_evaluation_non_gt_person_and_duplicate_track():
    def at(f):
        dx = 12 if 10 <= f <= 20 else 60
        return {1: (box(500, 300), 0.9), 2: (box(500 + dx, 302), 0.9)}
    n = 30
    fr = frames_from(n, at)
    r = run(fr, V2)
    tracks = {f.frame: {t: o.box for t, o in f.tracks.items()} for f in fr}
    only_one = gt_from(n, lambda f: {11: box(500, 300)})                       # track 2 is nobody in GT
    res = _scored(r.events, r.frame_rows, tracks, only_one, n)
    assert [e["outcome"] for e in res["events"]] == ["non_gt_person"]
    both_same = gt_from(n, lambda f: {11: box(500 + (6 if 10 <= f <= 20 else 0), 301, w=26)})
    tracks_dup = {f: {1: box(500, 300), 2: box(506, 301)} for f in range(n)}  # both tracks map to GT 11
    res = _scored(r.events, r.frame_rows, tracks_dup, both_same, n)
    assert [e["outcome"] for e in res["events"]] == ["fp_duplicate_track"]


def test_evaluation_loss_outcomes():
    def at(f):
        return {1: (box(300, 300), 0.9)} if not 10 <= f <= 15 else {}
    n = 30
    fr = frames_from(n, at)
    r = run(fr, V2)
    tracks = {f.frame: {t: o.box for t, o in f.tracks.items()} for f in fr}
    alone = gt_from(n, lambda f: {11: box(300, 300)})
    res = _scored(r.events, r.frame_rows, tracks, alone, n)
    assert [e["outcome"] for e in res["events"]] == ["true_loss"]
    covered = gt_from(n, lambda f: {11: box(300, 300), 12: box(300, 310, w=24, h=55)})
    res = _scored(r.events, r.frame_rows, tracks, covered, n)
    assert [e["outcome"] for e in res["events"]] == ["actually_occluded_in_gt"]


# ------------------------------------------------------------------------------------------- overlays
def test_zoom_rect_contains_boxes_keeps_aspect_and_stays_inside():
    boxes = [(100, 100, 120, 150), (130, 110, 150, 160)]
    r = ov.zoom_rect(boxes, 4096, 1080)
    x0, y0, x1, y1 = r
    assert x0 <= 100 and y0 <= 100 and x1 >= 150 and y1 >= 160
    assert (x1 - x0) / (y1 - y0) == pytest.approx(16 / 9, rel=0.02)
    edge = ov.zoom_rect([(4080, 1060, 4095, 1079)], 4096, 1080)
    assert edge[2] <= 4096 and edge[3] <= 1080 and edge[0] >= 0 and edge[1] >= 0
    assert ov.zoom_rect([], 640, 360) == (0, 0, 640, 360)


def test_region_rect_zoom_levels():
    assert ov.region_rect(2048, 540, 1.0, 4096, 1080) == (1088, 0, 3008, 1080)   # full height, 16:9
    x0, y0, x1, y1 = ov.region_rect(10, 10, 4.0, 4096, 1080)
    assert (x0, y0) == (0, 0) and y1 - y0 == 270


def test_category_of_and_uncertain_row():
    ev = {"event_class": "detection_loss", "subtype": "tracker_gap", "uncertain": False}
    assert ov.category_of(ev) == "tracker_gap"
    assert ov.category_of({**ev, "uncertain": True}) == "uncertain"
    assert ov.category_of({**ev, "uncertain": True}, uncertain_row=False) == "tracker_gap"
    assert ov.category_of({"event_class": "detection_loss", "subtype": "near_neighbour_uncertain", "uncertain": True},
                          uncertain_row=False) == "uncertain"
    assert ov.category_of({"event_class": "probable_occlusion", "subtype": "full", "uncertain": False}) == "probable_occlusion"


def test_render_view_scales_and_draws_selected_event():
    img = np.zeros((200, 400, 3), np.uint8)
    ev = {"event_id": "e1", "event_class": "probable_occlusion", "subtype": "full", "uncertain": False, "members": []}
    rows = [{"event_id": "e1", "frame": 0, "boxes": {"1": {"state": "visible", "box": [100, 50, 120, 100]},
                                                      "2": {"state": "hidden_predicted", "box": [105, 48, 125, 98]}}}]
    out = ov.render_view(img, (50, 0, 250, 112), 800, {1: (100, 50, 120, 100), 9: (300, 50, 320, 100)}, rows, {"e1": ev},
                         {"e1": "E1"}, selected="e1")
    assert out.shape == (448, 800, 3)
    assert out.sum() > 0
    orange = np.array(ov.CATEGORY_BGR["probable_occlusion"])
    assert (np.abs(out.reshape(-1, 3).astype(int) - orange).sum(axis=1) < 30).any()


def test_label_pills_do_not_overlap():
    img = np.zeros((300, 300, 3), np.uint8)
    placed: list = []
    for _ in range(3):
        ov.pill(img, "T123 predicted", ((100, 100), (130, 160)), (0, 120, 255), 0.55, placed)
    for i, a in enumerate(placed):
        for b in placed[i + 1:]:
            assert not ov._overlaps(a, b)


# ------------------------------------------------------------------------------------------- dashboard state
def test_frame_time_sync_round_trip():
    fps, offset = 25.0, 0.979
    for f in (0, 1, 24, 25, 3000, 5975):
        assert S.clock_to_frame(S.frame_to_clock(f, fps), fps, 0, 5975) == f
    assert S.frame_to_pts(0, fps, offset) == pytest.approx(0.979)
    assert S.frame_to_pts(3000, fps, offset) == pytest.approx(120.979)
    assert S.clock_to_frame(9999.0, fps, 0, 5975) == 5975
    assert S.format_clock(121.34) == "02:01.34"


def test_navigation_and_playback():
    assert S.step(10, -25, 0, 100) == 0 and S.step(90, 25, 0, 100) == 100
    assert S.playback_next(98, 0, 100, 1) == (99, True)
    assert S.playback_next(100, 0, 100, 1) == (100, False)
    assert S.playback_next(100, 0, 100, 1, loop=True) == (0, True)
    assert S.playback_next(97, 0, 100, 5) == (100, False)


def _ev(i, start, end, cls="box_overlap", sub="overlap", unc=False, level="medium", tracks=(1, 2)):
    return {"event_id": f"e{i}", "event_class": cls, "subtype": sub, "start_frame": start, "end_frame": end,
            "peak_frame": (start + end) // 2, "uncertain": unc, "evidence_level": level,
            "members": [{"track_id": t} for t in tracks], "evidence": []}


def test_filters_and_event_navigation():
    evs = [_ev(1, 10, 20), _ev(2, 30, 31, "detection_loss", "detector_miss", tracks=(5,)),
           _ev(3, 50, 90, "probable_occlusion", "full", unc=True, level="low", tracks=(2, 3))]
    assert [e["event_id"] for e in S.filter_events(evs, S.Filters())] == ["e1", "e2", "e3"]
    assert [e["event_id"] for e in S.filter_events(evs, S.Filters(categories=("uncertain",)))] == ["e3"]
    assert [e["event_id"] for e in S.filter_events(evs, S.Filters(categories=("probable_occlusion",), uncertain_row=False))] == ["e3"]
    assert [e["event_id"] for e in S.filter_events(evs, S.Filters(min_frames=3))] == ["e1", "e3"]
    assert [e["event_id"] for e in S.filter_events(evs, S.Filters(uncertainty="exclude"))] == ["e1", "e2"]
    assert [e["event_id"] for e in S.filter_events(evs, S.Filters(track=2))] == ["e1", "e3"]
    assert [e["event_id"] for e in S.filter_events(evs, S.Filters(evidence_levels=("low",)))] == ["e3"]
    assert S.adjacent_event(evs, 25, +1)["event_id"] == "e2"
    assert S.adjacent_event(evs, 30, -1)["event_id"] == "e1"
    assert S.adjacent_event(evs, 95, +1) is None
    assert S.focus_frame(evs[2], "start") == 50 and S.focus_frame(evs[2]) == 70
    assert S.snapshot_frames(evs[0], 0, 100, 12) == {"before": 0, "start": 10, "peak": 15, "end": 20, "after": 32}


def test_timeline_items_and_baseline_matches():
    evs = [_ev(1, 25, 49), _ev(2, 0, 0, "detection_loss", "tracker_gap", tracks=(9,))]
    items = S.timeline_items(evs, 25.0, S.labels_for(evs))
    assert items[0]["start_s"] == pytest.approx(1.0) and items[0]["dur_s"] == pytest.approx(1.0)
    assert items[1]["dur_s"] == pytest.approx(0.04) and items[1]["label"] == "E2"
    base = [_ev(10, 0, 30, tracks=(1, 7)), _ev(11, 0, 30, tracks=(8,)), _ev(12, 60, 70, tracks=(1,))]
    assert [b["event_id"] for b in S.baseline_matches(evs[0], base)] == ["e10"]


def test_explanations_cover_every_engine_evidence_code():
    codes = {"ios>=0.2", "ios>=0.5", "far_degraded_conf", "far_degraded_height", "hidden_behind_neighbour", "occluder_box_grew",
             "reappeared_same_track", "new_track_at_prediction", "not_reappeared", "truncated", "untracked_detection_in_gap",
             "group_size>=3", "detector_miss", "tracker_gap", "near_neighbour_uncertain", "left_frame"}
    assert codes <= set(S.EVIDENCE_TEXT)
    lines = S.explain({"event_class": "box_overlap", "subtype": "overlap", "evidence": ["ios>=0.2"]})
    assert len(lines) == 2 and "both stayed detected" in lines[1]


def test_review_store_is_append_only_and_separate(tmp_path):
    store = S.ReviewStore(tmp_path / "reviews" / "src" / "reviews.jsonl")
    store.append("run1", "e1", "incorrect", "two referees")
    store.append("run1", "e1", "correct")
    store.append("run2", "e1", "uncertain")
    with pytest.raises(ValueError):
        store.append("run1", "e2", "maybe")
    assert len(store.load()) == 3
    assert store.latest("run1")["e1"]["label"] == "correct"
    assert store.latest("run2")["e1"]["label"] == "uncertain"
    with store.path.open("a", encoding="utf-8") as fh:
        fh.write('{"torn')                       # an interrupted write never blocks loading
    assert len(store.load()) == 3


def test_discover_runs_skips_invalid_partial_and_incomplete(tmp_path):
    def make(name, run_id=None, status="complete", sid="soccertrack-117093"):
        d = tmp_path / sid / "occlusion" / "0.2.0" / name
        d.mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps({"status": status, "run_id": run_id or name, "profile": "v2",
                                                     "source": {"source_id": sid}, "window": {"start": 0, "end": 9}}))
        (d / "events.jsonl").write_text("")
        (d / "frames.jsonl").write_text("")
    make("occl-a")
    make("INVALID-x", run_id="INVALID-x")
    make("occl-b.partial", run_id="occl-b.partial")
    make("occl-c", status="running")
    make("occl-d", run_id="other")
    make("occl-e", sid="soccertrack-118578")
    runs = D.discover_runs(tmp_path)
    assert [r.run_id for r in runs] == ["occl-a"]
    data = D.load_run(runs[0])
    assert data.events == [] and data.evaluation is None


# ------------------------------------------------------------------------------------------- real data (optional)
DATA_ROOT = os.environ.get("FOOTBALL_FATIGUE_DATA_ROOT")
needs_data = pytest.mark.skipif(
    not DATA_ROOT or not (Path(DATA_ROOT or ".") / "external" / "soccertrack-v2" / "raw" / "117093").is_dir(),
    reason="set FOOTBALL_FATIGUE_DATA_ROOT to a data root holding SoccerTrack 117093")


@needs_data
def test_real_window_v2_run_and_evaluation(tmp_path):
    from football_fatigue.video.occlusion.__main__ import main
    root = tmp_path / "out"
    assert main(["run", "--data-root", DATA_ROOT, "--out-root", str(root), "--source", "soccertrack-117093",
                 "--profile", "v2", "--start", "3000", "--end", "3200", "--no-render"]) == 0
    (run_dir,) = list((root / "soccertrack-117093" / "occlusion" / "0.2.0").glob("occl-*"))
    m = json.loads((run_dir / "manifest.json").read_text())
    assert m["profile"] == "v2" and m["play_area"] and m["stats"]["play_area"]["status"] == "used"
    events = [event_from_json(l) for l in (run_dir / "events.jsonl").read_text().splitlines()]
    known = {e.event_id for e in events}
    assert all(json.loads(l)["event_id"] in known for l in (run_dir / "frames.jsonl").read_text().splitlines())
    assert main(["evaluate", "--data-root", DATA_ROOT, "--run-dir", str(run_dir)]) == 0
    (ev_file,) = list((root / "soccertrack-117093" / "occlusion" / "evaluation").glob("*.json"))
    res = json.loads(ev_file.read_text())
    assert res["split_role"] == "tune" and res["summary"]["events_total"] == len(events)


@needs_data
def test_dashboard_frame_reader_matches_sequential_decode():
    from football_fatigue.video.sources import SOURCES, verify_reader
    video = Path(DATA_ROOT) / SOURCES["soccertrack-117093"]["video"]
    reader = D.LockedReader(video)
    check = verify_reader(video, [3000, 3001, 3125])
    assert check["exact"], check
    a = reader.frame(3125)
    b = reader.frame(3000)          # backwards seek
    c = reader.frame(3125)
    assert a is not None and b is not None and np.array_equal(a, c)


OUT_ROOT = Path(__file__).resolve().parents[1] / "data" / "processed" / "video"


@needs_data
@pytest.mark.skipif(not any(OUT_ROOT.glob("*/occlusion/*/*/manifest.json")), reason="no saved runs in data/processed/video")
def test_dashboard_app_smoke(monkeypatch):
    """Drive the real Streamlit script headless: load, pick an event, step a frame, manual zoom; no exceptions."""
    from streamlit.testing.v1 import AppTest

    app = Path(__file__).resolve().parents[1] / "src" / "football_fatigue" / "video" / "dashboard" / "app.py"
    monkeypatch.setattr("sys.argv", ["app.py", "--data-root", DATA_ROOT, "--out-root", str(OUT_ROOT)])
    at = AppTest.from_file(str(app), default_timeout=240)
    at.run()
    assert not at.exception
    lo = at.session_state["cur"]
    next(b for b in at.button if b.label == "next event ▶").click().run()
    assert not at.exception
    sel = at.session_state["selected"]
    assert sel is not None and at.session_state["cur"] > lo
    assert "Why it fired" in " ".join(m.value for m in at.markdown)
    f = at.session_state["cur"]
    next(b for b in at.button if b.label == "+1 ▶").click().run()
    assert at.session_state["cur"] == f + 1 and not at.exception
    at.radio(key="view_radio").set_value("Manual region").run()
    assert not at.exception
