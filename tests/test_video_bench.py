"""Detector / tracker benchmark (Session B research track).

Unit tests run anywhere. ``needs_data`` tests use FOOTBALL_FATIGUE_DATA_ROOT and the saved benchmark outputs under
data/processed/video/_bench.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / "data" / "processed" / "video"
DATA_ROOT = os.environ.get("FOOTBALL_FATIGUE_DATA_ROOT")
MANIFEST = OUT_ROOT / "_bench" / "frames" / "frames-v1.json"
needs_data = pytest.mark.skipif(not DATA_ROOT or not MANIFEST.is_file(), reason="needs FOOTBALL_FATIGUE_DATA_ROOT and the frame manifest")


# ------------------------------------------------------------------------------------------------ metrics
def _g(*boxes):
    return [{"gid": i, "box": list(b), "vis": "visible", "size": ">=80 px"} for i, b in enumerate(boxes)]


def test_match_frame_is_greedy_by_confidence_and_one_to_one():
    from football_fatigue.video.bench.evaluate import match_frame

    gt = [[0, 0, 10, 10], [20, 0, 30, 10]]
    preds = [[0, 0, 10, 10, 0.5], [1, 0, 11, 10, 0.9], [20, 0, 30, 10, 0.7]]
    p2g, g2p = match_frame(gt, preds)
    assert p2g == [-1, 0, 1] and g2p == [1, 2]          # the 0.9 box takes GT 0 first; the 0.5 box is left over


def test_ap_and_scores_on_synthetic_frames():
    from football_fatigue.video.bench.evaluate import _ap, _scores

    gt = _g((0, 0, 10, 10), (20, 0, 30, 10))
    perfect = [(gt, [[0, 0, 10, 10, 0.9, "player", "player"], [20, 0, 30, 10, 0.8, "player", "player"]])]
    assert _ap(perfect, 0.5) == pytest.approx(1.0)
    half = [(gt, [[0, 0, 10, 10, 0.9, "p", "player"], [50, 50, 60, 60, 0.95, "p", "player"]])]
    assert _ap(half, 0.5) == pytest.approx(0.25)           # first ranked is FP: precision 0.5 at recall 0.5
    dup = [(gt, [[0, 0, 10, 10, 0.9, "p", "player"], [0, 0, 10, 9.5, 0.8, "p", "player"]])]
    s = _scores(dup, 0.4)
    assert (s["tp"], s["fp"], s["fn"], s["duplicates"]) == (1, 1, 1, 1)
    assert _scores(dup, 0.85)["fp"] == 0                    # the confidence threshold is strict (>)
    assert s["recall_by_size"][">=80 px"] == {"recall": 0.5, "n": 2}


def test_ap_ignores_hidden_gt_without_rewarding_or_penalising():
    from football_fatigue.video.bench.evaluate import _ap

    gt = _g((0, 0, 10, 10), (20, 0, 30, 10))
    gt[1]["vis"] = "hidden"
    fr = [(gt, [[0, 0, 10, 10, 0.9, "p", "player"], [20, 0, 30, 10, 0.8, "p", "player"]])]
    assert _ap(fr, 0.5, ignore_hidden=True) == pytest.approx(1.0)
    assert _ap([(gt, [[0, 0, 10, 10, 0.9, "p", "player"]])], 0.5, ignore_hidden=False) == pytest.approx(0.5)


def test_views_drop_off_pitch_and_referee_boxes():
    from football_fatigue.video.bench.evaluate import _view

    poly = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32).reshape(-1, 1, 2)
    boxes = [[10, 10, 20, 50, 0.9, "player", "player"], [10, 10, 20, 50, 0.9, "referee", "referee"],
             [10, 150, 20, 190, 0.9, "person", "person"], [10, 10, 20, 50, 0.9, "ball", "ball"]]
    assert len(_view(boxes, "all", poly)) == 3
    assert len(_view(boxes, "on_pitch", poly)) == 2
    assert [b[6] for b in _view(boxes, "no_referee", poly)] == ["player"]


# ------------------------------------------------------------------------------------------------ registry / adapters
def test_registry_status_is_derived_not_assumed(tmp_path):
    from football_fatigue.video.bench.registry import CANDIDATES, status

    assert status("yolo11m-coco", tmp_path) == "REQUIRES WEIGHTS/DOWNLOAD APPROVAL"
    assert status("rfdetr-medium-coco", tmp_path) == "INCOMPATIBLE"
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "yolov8n.pt").write_bytes(b"x")
    assert status("yolov8n-coco", tmp_path) == "AVAILABLE NOT TESTED"
    runs = {"r": {"detector": "yolov8n-coco", "status": "complete", "device": "cpu"}}
    assert status("yolov8n-coco", tmp_path, runs) == "CPU PILOT ONLY"
    runs["r"]["device"] = "cuda:0"
    assert status("yolov8n-coco", tmp_path, runs) == "BENCHMARKED"
    assert {c.track for c in CANDIDATES.values()} == {"B1", "B2"}


def test_checkpoint_identity_is_verified(tmp_path):
    from football_fatigue.video.bench.registry import resolve

    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "yolov8n.pt").write_bytes(b"not the real weights")
    with pytest.raises(RuntimeError, match="SHA-256"):
        resolve("yolov8n-coco", tmp_path)
    assert resolve("yolo11m-coco", tmp_path) is None


def test_unavailable_detectors_and_gpu_need_explicit_approval(tmp_path, monkeypatch):
    from football_fatigue.video.bench.detectors import DetectorUnavailable, make_adapter

    with pytest.raises(DetectorUnavailable, match="REQUIRES WEIGHTS/DOWNLOAD APPROVAL"):
        make_adapter("yolo12m-coco", "tile1280-img1280", tmp_path, tmp_path)
    with pytest.raises(DetectorUnavailable, match="INCOMPATIBLE"):
        make_adapter("rfdetr-medium-coco", "tile1280-img1280", tmp_path, tmp_path)
    with pytest.raises(DetectorUnavailable, match="only the production detector"):
        make_adapter("yolov8n-coco", "production-cache", tmp_path, tmp_path)
    project = Path(DATA_ROOT).parent if DATA_ROOT else None
    if project and (project / "models" / "yolov8n.pt").is_file():
        monkeypatch.delenv("FF_GPU_APPROVED", raising=False)
        with pytest.raises(DetectorUnavailable, match="GPU inference needs explicit approval"):
            make_adapter("yolov8n-coco", "tile1280-img1280", tmp_path, project, device="cuda:0")


def test_tracker_profiles_keep_b0_and_t0_on_the_production_tracker():
    from football_fatigue.video.tracking import TRACKER_BACKEND, TRACKER_INPUTS

    assert TRACKER_BACKEND["B0"] == TRACKER_BACKEND["T0"] == "sv-bytetrack"
    assert all(TRACKER_INPUTS[n] == TRACKER_INPUTS["T0"] for n in TRACKER_INPUTS if n.startswith("T0-"))


def test_ultralytics_trackers_return_measured_boxes_and_stable_ids():
    from types import SimpleNamespace

    import football_fatigue.video.bench.trackers as T

    frames = [(f, [SimpleNamespace(box=(100 + 3 * f, 100, 120 + 3 * f, 150), conf=0.9, mapped="player", person=True),
                   SimpleNamespace(box=(400, 100 + 2 * f, 420, 150 + 2 * f), conf=0.8, mapped="player", person=True)])
              for f in range(40)]
    orig_inputs, orig_score = T._inputs, None
    try:
        T._inputs = lambda d, s, i: iter(frames)
        for name in ("ul-bytetrack", "ul-ocsort", "ul-botsort"):
            tr, _cfg, ov = T._ul_tracker(name, 75, 25.0)
            assert ov["track_buffer"] == 75
            from ultralytics.engine.results import Boxes

            ids = set()
            for f, dets in frames:
                arr = np.array([[*d.box, d.conf, 0] for d in dets], dtype=np.float32)
                res = tr.update(Boxes(arr, (1080, 4096)), None)
                for r in res:
                    assert tuple(dets[int(r[-1])].box) == pytest.approx(tuple(arr[int(r[-1]), :4]))
                    ids.add(int(r[4]))
            assert len(ids) == 2, name
    finally:
        T._inputs = orig_inputs


def test_recoveries_counts_gaps():
    from football_fatigue.video.bench.trackers import recoveries

    tr = {0: [(1,)], 1: [(1,)], 4: [(1,)], 40: [(1,)]}
    assert recoveries(tr, 75) == {"gaps_recovered_1_5": 1, "gaps_recovered_6_25": 0, "gaps_recovered_26_plus": 1}


def test_id_events_detect_switch_and_fragment():
    from football_fatigue.video.bench.visual import id_events

    box = (0.0, 0.0, 10.0, 20.0)
    gt = {f: [(7, box)] for f in range(6)}
    tracks = {0: [[1, *box]], 1: [[1, *box]], 2: [], 3: [[1, *box]], 4: [[2, *box]], 5: [[2, *box]]}
    ev = id_events(gt, tracks, 0, 5)
    assert [e["kind"] for e in ev] == ["fragment_recovered", "id_switch"]
    assert ev[0]["same_id"] is True and ev[1]["from"] == 1 and ev[1]["to"] == 2


def test_player_compare_mode_accepts_any_two_profiles():
    html = (REPO / "src" / "football_fatigue" / "video" / "player" / "player.html").read_text(encoding="utf-8")
    assert 'Q.get("cmp")' in html and "drawPane(f,S.ti,0,h,rect);drawPane(f,S.cmpWith,h,h,rect)" in html


def test_protocol_matches_the_production_cache_settings():
    from football_fatigue.video.bench.detectors import CONF_FLOOR, PROTOCOLS

    p = PROTOCOLS["tile1280-img1920"]
    assert (p.image_size, p.tile, p.overlap, p.merge, CONF_FLOOR) == (1920, 1280, 0.2, "edge_aware", 0.1)


def test_session_a_vision_code_and_models_are_unmodified():
    a = Path(DATA_ROOT).parent if DATA_ROOT else Path("F:/Football-Fatigue-Project")
    if not (a / ".git").exists():
        pytest.skip("Session A repository not available")
    out = subprocess.run(["git", "-C", str(a), "status", "--porcelain", "--", "src/football_fatigue/vision", "models"],
                         capture_output=True, text=True, check=True).stdout
    assert out.strip() == ""


# ------------------------------------------------------------------------------------------------ real data
@needs_data
def test_manifest_is_deterministic_and_protected(tmp_path):
    from football_fatigue.video.bench.frames import load

    m = load(OUT_ROOT)
    assert m["frames_sha256"] == load(OUT_ROOT)["frames_sha256"]
    assert {e["source_id"] for e in m["frames"]} == {"soccertrack-117093", "soccertrack-118575", "soccertrack-117092",
                                                    "soccertrack-118576"}
    assert all(e["role"] in ("tune", "check") for e in m["frames"])
    copy = tmp_path / "_bench" / "frames"
    copy.mkdir(parents=True)
    bad = json.loads(MANIFEST.read_text(encoding="utf-8"))
    bad["frames"][0]["frame"] += 1
    (copy / "frames-v1.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(RuntimeError, match="modified"):
        load(tmp_path)


@needs_data
def test_reserved_matches_are_refused(tmp_path):
    from football_fatigue.video.bench.frames import build
    from football_fatigue.video.sources import SourceError

    with pytest.raises((SourceError, KeyError)):
        build(Path(DATA_ROOT), tmp_path, sources=("soccertrack-118577",))


@needs_data
def test_cache_adapter_boxes_use_the_cache_convention():
    from football_fatigue.video.bench.detectors import CacheAdapter
    from football_fatigue.video.detection.diagnose import iter_cache

    a = CacheAdapter(Path(DATA_ROOT))
    boxes = a.detect("soccertrack-118576", 2790, None)
    cached = next(ds for fr, ds in iter_cache(Path(DATA_ROOT), "soccertrack-118576", 2790))
    assert len(boxes) == sum(d.conf > 0.1 for d in cached)
    assert all(b[2] > b[0] and b[3] > b[1] and 0 <= b[0] < 4096 and 0 <= b[1] < 1080 for b in boxes)


@needs_data
def test_detector_runs_record_provenance():
    from football_fatigue.video.bench.detectors import load_runs

    runs = load_runs(OUT_ROOT)
    if not runs:
        pytest.skip("no detector runs")
    for k, r in runs.items():
        assert r["frame_manifest"]["frames_sha256"] == json.loads(MANIFEST.read_text(encoding="utf-8"))["frames_sha256"]
        assert r["confidence_floor"] == 0.1 and r["box_convention"].startswith("xyxy")
        if r["device"] != "cache":
            assert r["checkpoint_sha256_verified"] == r["candidate"]["sha256"]
            assert r["detector_settings"]["tile_merge"]["policy"] == "edge_aware"


@needs_data
def test_sv_bytetrack_harness_reproduces_the_t0_tracks():
    import itertools

    import football_fatigue.video.bench.trackers as T

    t0 = OUT_ROOT / "soccertrack-118576" / "tracking" / "T0-tracker-input-v1" / "frames.jsonl"
    orig = T._inputs
    try:
        T._inputs = lambda d, s, i: itertools.islice(orig(d, s, i), 300)
        mine, _ = T.track_source(Path(DATA_ROOT), "soccertrack-118576", "T0", "sv-bytetrack", 75)
    finally:
        T._inputs = orig
    with t0.open(encoding="utf-8") as fh:
        for line in itertools.islice(fh, 300):
            r = json.loads(line)
            want = {(p["track_id"], tuple(round(v, 3) for v in p["bbox_xyxy_px"])) for p in r["players"]}
            got = {(tid, tuple(round(v, 3) for v in box)) for tid, box, _c, _k in mine[r["frame_num"]]}
            assert got == want


@needs_data
def test_engine_event_timestamps_follow_frames_and_pts_offset():
    run = next((OUT_ROOT / "soccertrack-117093" / "occlusion" / "0.2.0").glob("occl-soccertrack-117093-0-*-v2-T0-*"))
    for line in list((run / "events.jsonl").read_text(encoding="utf-8").splitlines())[:200]:
        e = json.loads(line)
        assert e["start_pts_s"] == pytest.approx(e["source"]["pts_offset_s"] + e["start_frame"] / e["source"]["fps"], abs=1e-3)
        assert e["end_frame"] >= e["start_frame"]


@needs_data
def test_benchmark_dashboard_section_renders(monkeypatch):
    from streamlit.testing.v1 import AppTest

    if not list((OUT_ROOT / "_bench" / "detector-v1" / "runs").glob("*/dets.jsonl")):
        pytest.skip("no detector runs")
    script = ("import sys\nfrom pathlib import Path\nfrom football_fatigue.video.dashboard import bench_tab\n"
              f"bench_tab.render(Path(r'{DATA_ROOT}'), Path(r'{OUT_ROOT}'), '')\n")
    at = AppTest.from_string(script, default_timeout=300)
    at.run()
    assert not at.exception
    at.toggle(key="bench_on").set_value(True).run()
    assert not at.exception
    assert any("matched" in m.value for m in at.markdown)
    at.selectbox(key="bm_src").set_value("soccertrack-117092").run()
    assert not at.exception
