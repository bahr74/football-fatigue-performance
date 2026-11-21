"""Tests for football_fatigue.video (V1 step 1).

Unit tests use SYNTHETIC tracks (invented boxes, not a real source schema). The integration test runs on
real SoccerTrack 117093 data only when FOOTBALL_FATIGUE_DATA_ROOT points at a data root that holds it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from football_fatigue.video.occlusion import geometry as g
from football_fatigue.video.occlusion.engine import EngineConfig, FrameInput, OcclusionEngine, TrackObs
from football_fatigue.video.schema import (
    SCHEMA_VERSION, Member, OcclusionEvent, SchemaError, SourceInfo, event_from_json,
)
from football_fatigue.video.sources import SourceError, guard_not_reserved

SRC = SourceInfo("synthetic-src", "synthetic", "cam0", "0" * 64, 25.0, 0.5, 1000, 500)
PROV = {"engine": "test"}


def box(cx: float, bottom: float, w: float = 20.0, h: float = 50.0) -> g.Box:
    return (cx - w / 2, bottom - h, cx + w / 2, bottom)


def make_frames(n: int, tracks_at, untracked_at=None) -> list[FrameInput]:
    frames = []
    for f in range(n):
        tracks = {tid: TrackObs(b, conf, "player") for tid, (b, conf) in tracks_at(f).items()}
        frames.append(FrameInput(f, tracks, tuple(untracked_at(f)) if untracked_at else ()))
    return frames


def run(frames, config=None, emit=None):
    emit = emit or (frames[0].frame, frames[-1].frame)
    return OcclusionEngine(config).run(frames, SRC, "run-test", emit, PROV)


def classes(result):
    return sorted((e.event_class, e.subtype) for e in result.events)


# ---------------------------------------------------------------- geometry
def test_geometry_basics():
    a, b = (0, 0, 10, 10), (5, 0, 15, 10)
    assert g.iou(a, b) == pytest.approx(50 / 150)
    assert g.ios(a, b) == pytest.approx(0.5)
    assert g.ios((0, 0, 10, 10), (2, 2, 4, 4)) == pytest.approx(1.0)
    assert g.covered_fraction((0, 0, 10, 10), (0, 0, 5, 10)) == pytest.approx(0.5)
    assert g.widen((0, 0, 10, 10), 0.2) == (-1.0, 0, 11.0, 10)
    assert g.nearer_camera((0, 0, 1, 20), (0, 0, 1, 10)) == 0
    assert g.nearer_camera((0, 0, 1, 10), (0, 0, 1, 20)) == 1
    assert g.outside_fraction((-5, 0, 5, 10), 100, 100) == pytest.approx(0.5)
    assert g.iou(a, (20, 20, 30, 30)) == 0.0


def test_constant_velocity_prediction():
    hist = [(0, (0, 0, 10, 10)), (1, (2, 0, 12, 10)), (2, (4, 0, 14, 10))]
    assert g.predict_constant_velocity(hist, 5) == pytest.approx((10, 0, 20, 10))
    assert g.predict_constant_velocity(hist[:1], 7) == (0, 0, 10, 10)


# ---------------------------------------------------------------- engine scenarios
def test_box_overlap_without_hiding():
    # two players side by side overlap (IoS ~0.4) for frames 10-19 with stable confidence
    def at(f):
        dx = 12 if 10 <= f <= 19 else 60
        return {1: (box(500, 300), 0.9), 2: (box(500 + dx, 302), 0.9)}
    r = run(make_frames(40, at))
    assert classes(r) == [("box_overlap", "overlap")]
    e = r.events[0]
    assert (e.start_frame, e.end_frame) == (10, 19)
    assert e.start_pts_s == pytest.approx(0.5 + 10 / 25)
    assert not any(m.hidden for m in e.members)


def _crossing(hidden_frames, grow=1.0, reappear=True):
    """Track 2 (far, smaller bottom y) walks behind static track 1 (near) and is not detected while behind it."""
    def at(f):
        near_w = 20 * (grow if f in hidden_frames else 1.0)
        out = {1: (box(500, 320, w=near_w), 0.9)}
        cx = 440 + 4 * f
        if f not in hidden_frames and (reappear or f < min(hidden_frames)):
            out[2] = (box(cx, 300), 0.9)
        return out
    return at


def test_full_occlusion_hidden_member_reappears_same_track():
    hidden = set(range(14, 18))  # far player centred on the occluder while missing
    r = run(make_frames(40, _crossing(hidden)))
    occ = [e for e in r.events if e.event_class == "probable_occlusion"]
    assert len(occ) == 1 and occ[0].subtype == "full"
    far = next(m for m in occ[0].members if m.track_id == 2)
    assert far.role == "far" and far.hidden == [[14, 17]] and far.outcome == "reappeared_same_track"
    assert "hidden_behind_neighbour" in occ[0].evidence
    assert not [e for e in r.events if e.event_class == "detection_loss"]  # hidden losses are not detection losses


def test_merged_when_occluder_box_grows():
    hidden = set(range(14, 18))
    r = run(make_frames(40, _crossing(hidden, grow=1.8)))
    occ = [e for e in r.events if e.event_class == "probable_occlusion"]
    assert [e.subtype for e in occ] == ["merged"]
    assert occ[0].measurements["max_occluder_area_ratio"] >= 1.4


def test_hidden_not_reappeared_is_uncertain():
    r = run(make_frames(60, _crossing(set(range(14, 60)), reappear=False)))
    occ = [e for e in r.events if e.event_class == "probable_occlusion"]
    assert len(occ) == 1
    assert occ[0].members[-1].outcome == "not_reappeared" or any(m.outcome == "not_reappeared" for m in occ[0].members)
    assert occ[0].uncertain and occ[0].evidence_level == "low"


def test_detector_miss_and_tracker_gap_are_distinguished():
    def at(f):
        out = {1: (box(200, 300), 0.9)}
        if not 10 <= f <= 14:
            out[2] = (box(700, 300), 0.9)
        return out
    r = run(make_frames(30, at))
    assert classes(r) == [("detection_loss", "detector_miss")]
    assert r.events[0].members[0].hidden == [[10, 14]]

    r2 = run(make_frames(30, at, untracked_at=lambda f: [box(700, 300)] if 10 <= f <= 14 else []))
    assert classes(r2) == [("detection_loss", "tracker_gap")]
    assert "untracked_detection_in_gap" in r2.events[0].evidence


def test_left_frame_and_short_losses():
    def at(f):
        out = {}
        if f < 10:
            out[3] = (box(960 + 8 * f, 300), 0.9)   # leaves the right edge at frame 10
        if f != 20:
            out[4] = (box(300, 300), 0.9)            # one-frame drop: ignored
        return out
    r = run(make_frames(30, at))
    assert classes(r) == [("detection_loss", "left_frame")]
    assert r.stats["short_isolated_losses_ignored"] == 1


def test_three_player_group_is_one_event():
    def at(f):
        dx = 12 if 10 <= f <= 20 else 60
        return {1: (box(500, 300), 0.9), 2: (box(500 + dx, 301), 0.9), 3: (box(500 + 2 * dx, 302), 0.9)}
    r = run(make_frames(40, at))
    assert len(r.events) == 1
    assert sorted(m.track_id for m in r.events[0].members) == [1, 2, 3]
    assert r.events[0].measurements["max_group_size"] == 3
    assert "group_size>=3" in r.events[0].evidence


def test_transient_one_frame_overlap_is_dropped():
    def at(f):
        dx = 12 if f == 10 else 60
        return {1: (box(500, 300), 0.9), 2: (box(500 + dx, 300), 0.9)}
    r = run(make_frames(30, at))
    assert r.events == [] and r.stats["transient_overlaps_dropped"] == 1


def test_partial_occlusion_from_confidence_drop():
    def at(f):
        overlap = 20 <= f <= 30
        # widened IoS 0.55 while overlapping (strong edge), far confidence drops 0.9 -> 0.5
        return {1: (box(500, 320), 0.9), 2: (box(502 if overlap else 560, 300), 0.5 if overlap else 0.9)}
    r = run(make_frames(45, at))
    occ = [e for e in r.events if e.event_class == "probable_occlusion"]
    assert [e.subtype for e in occ] == ["partial"]
    assert "far_degraded_conf" in occ[0].evidence and occ[0].uncertain


def test_lone_hidden_prediction_never_creates_extra_event():
    # the far player moves fast; while missing its prediction leaves the occluder, which must not spawn new events
    def at(f):
        out = {1: (box(500, 320), 0.9)}
        if not 14 <= f <= 22:
            out[2] = (box(430 + 5 * f, 300), 0.9)
        return out
    r = run(make_frames(40, at))
    occ = [e for e in r.events if e.event_class == "probable_occlusion"]
    assert len(occ) == 1


def test_event_ids_and_output_are_deterministic():
    frames = make_frames(40, _crossing(set(range(14, 18))))
    a, b = run(frames), run(frames)
    assert [e.to_json() for e in a.events] == [e.to_json() for e in b.events]
    assert a.frame_rows == b.frame_rows
    assert all(e.event_id.startswith("oe-synthetic-src-") for e in a.events)


def test_emit_range_filters_events():
    def at(f):
        dx = 12 if 10 <= f <= 19 else 60
        return {1: (box(500, 300), 0.9), 2: (box(500 + dx, 302), 0.9)}
    assert run(make_frames(40, at), emit=(25, 39)).events == []


def test_frames_must_be_contiguous():
    frames = make_frames(5, lambda f: {})
    with pytest.raises(ValueError):
        OcclusionEngine().run([frames[0], frames[2]], SRC, "r", (0, 2), PROV)


# ---------------------------------------------------------------- schema
def _event(**kw) -> OcclusionEvent:
    base = dict(event_id="oe-x", run_id="r", source=SRC, event_class="box_overlap", subtype="overlap",
                start_frame=1, end_frame=3, start_pts_s=0.54, end_pts_s=0.62, peak_frame=2,
                members=[Member(1, "near", 1, 3), Member(2, "far", 1, 3)], parent_event_id=None,
                measurements={"max_ios_widened": 0.4}, evidence=["ios>=0.2"], evidence_level="medium",
                uncertain=False, uncertainty_reasons=[], pixel_xy=[1.0, 2.0], pitch={"status": "not_computed"},
                truncated=None, provenance=PROV)
    base.update(kw)
    return OcclusionEvent(**base)


def test_schema_round_trip_and_evolution_rules():
    e = _event()
    text = e.to_json()
    assert json.loads(text)["schema_version"] == SCHEMA_VERSION
    assert event_from_json(text) == e
    extra = json.loads(text)
    extra["new_minor_field"] = 1
    extra["schema_version"] = SCHEMA_VERSION.split(".")[0] + ".9.0"
    assert event_from_json(json.dumps(extra)).event_id == "oe-x"   # unknown fields ignored within a major version
    extra["schema_version"] = "2.0.0"
    with pytest.raises(SchemaError):
        event_from_json(json.dumps(extra))


def test_schema_rejects_invalid_records():
    with pytest.raises(SchemaError):
        _event(subtype="full").to_json()          # full is not a box_overlap subtype
    with pytest.raises(SchemaError):
        _event(peak_frame=9).to_json()
    with pytest.raises(SchemaError):
        _event(members=[]).to_json()


# ---------------------------------------------------------------- guards
def test_reserved_sequences_are_refused():
    with pytest.raises(SourceError):
        guard_not_reserved("soccertrack-118578-anything")
    guard_not_reserved("soccertrack-117093")


def test_cli_refuses_output_inside_data_root(tmp_path):
    from football_fatigue.video.occlusion.__main__ import main
    with pytest.raises(SystemExit, match="inside the read-only data root"):
        main(["run", "--data-root", str(tmp_path), "--out-root", str(tmp_path / "out"), "--source", "soccertrack-117093",
              "--cache-run", "c", "--replay-run", "r", "--start", "0", "--end", "1"])


# ---------------------------------------------------------------- integration (real data, optional)
DATA_ROOT = os.environ.get("FOOTBALL_FATIGUE_DATA_ROOT")
CACHE = "soccertrack-117093-m6-d1fixed-full-merged-20261025"
REPLAY = CACHE + "-full-stable-replay"


needs_data = pytest.mark.skipif(
    not DATA_ROOT or not (Path(DATA_ROOT or ".") / "processed" / "vision" / REPLAY).is_dir(),
    reason="set FOOTBALL_FATIGUE_DATA_ROOT to a data root holding the 117093 cache and replay")


@needs_data
def test_exact_reader_matches_sequential_decoding_despite_pts_offset():
    from football_fatigue.video.sources import SOURCES, verify_reader
    video = Path(DATA_ROOT) / SOURCES["soccertrack-117093"]["video"]
    check = verify_reader(video, [0, 1, 30, 120])
    assert check["exact"], check


@needs_data
def test_integration_117093_short_window_is_valid_and_deterministic(tmp_path):
    from football_fatigue.video.occlusion.__main__ import main
    outs = []
    for k in (1, 2):
        root = tmp_path / f"out{k}"
        assert main(["run", "--data-root", DATA_ROOT, "--out-root", str(root), "--source", "soccertrack-117093",
                     "--cache-run", CACHE, "--replay-run", REPLAY, "--start", "3040", "--end", "3100", "--no-render"]) == 0
        (run_dir,) = list((root / "soccertrack-117093" / "occlusion").glob("*/*"))
        outs.append(run_dir)
    a, b = outs
    assert (a / "events.jsonl").read_bytes() == (b / "events.jsonl").read_bytes()
    assert (a / "frames.jsonl").read_bytes() == (b / "frames.jsonl").read_bytes()
    manifest = json.loads((a / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["provenance"]["tracked_boxes_verified_against_cache"] is True
    pkg = Path(__file__).resolve().parents[1] / "src" / "football_fatigue"
    assert all(Path(p).resolve().is_relative_to(pkg) for p in manifest["import_roots"].values())
    events = [event_from_json(line) for line in (a / "events.jsonl").read_text().splitlines()]
    assert events and all(e.end_frame >= 3040 and e.start_frame <= 3100 for e in events)
    assert {e.event_class for e in events} <= {"box_overlap", "probable_occlusion", "detection_loss"}
