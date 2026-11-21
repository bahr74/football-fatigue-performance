"""M13: activation ablation profiles, A2-cap v2 (plan / capture records / attribution), GT-free MOT export, baseline.

Unit tests need no data; ``needs_data`` tests read FOOTBALL_FATIGUE_DATA_ROOT and saved outputs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "processed" / "video"
DATA_ROOT = os.environ.get("FOOTBALL_FATIGUE_DATA_ROOT")
needs_data = pytest.mark.skipif(not DATA_ROOT or not (OUT / "_diag" / "pipeline-v1").is_dir(),
                                reason="needs FOOTBALL_FATIGUE_DATA_ROOT and saved _diag outputs")
NAMES = {0: "ball", 1: "player", 2: "referee", 3: "goalkeeper"}


# ------------------------------------------------------------------------------------------- tracker profiles
def test_activation_profiles_change_only_the_activation():
    from football_fatigue.video.tracking import ACTIVATION, TRACKER_BACKEND, TRACKER_INPUTS

    assert TRACKER_INPUTS["T0-A03"] is TRACKER_INPUTS["T0"] and TRACKER_INPUTS["T0L-A03"] is TRACKER_INPUTS["T0L"]
    assert ACTIVATION == {"T0-A03": 0.3, "T0L-A03": 0.3}
    assert TRACKER_BACKEND["T0-A03"] == TRACKER_BACKEND["T0L-A03"] == "sv-bytetrack"
    assert "T0" not in ACTIVATION and "B0" not in ACTIVATION          # production profiles keep the replay's 0.5


def test_supervision_bytetrack_semantics_documented_in_m13():
    sv = pytest.importorskip("supervision")
    bt = sv.ByteTrack(track_activation_threshold=0.5, lost_track_buffer=75, minimum_matching_threshold=0.95, frame_rate=25)
    assert bt.det_thresh == pytest.approx(0.6)                         # new tracks need conf >= activation + 0.1
    assert bt.max_time_lost == 62                                      # 75 frames scaled by 25/30
    assert sv.ByteTrack(track_activation_threshold=0.3, frame_rate=25).det_thresh == pytest.approx(0.4)


def test_activation_cell_names_and_preregistered_selection_rule():
    from football_fatigue.video.failures.activation import cell_name, select_cell

    assert cell_name("T0", 0.5) == "T0" and cell_name("T0L", 0.3) == "T0L-A30"

    def row(cell, hota, idf1, idsw):
        return {"cell": cell, "HOTA": hota, "DetA": 0, "AssA": 0, "IDF1": idf1, "MOTA": 0, "gt_coverage": 0,
                "missing_gt_players_per_frame": 0, "IDSW": idsw, "Frag": 0, "FP": 0, "FN": 0, "track_initialisations": 0,
                "track_continuations": 0, "temporary_ids": 0}
    rows = [row("A", 0.400, 0.40, 10), row("B", 0.4015, 0.39, 10), row("C", 0.390, 0.50, 1)]
    assert select_cell(rows) == "A"                                   # B is within the tie band, A has higher IDF1
    assert select_cell([row("A", 0.40, 0.4, 10), row("B", 0.41, 0.3, 99)]) == "B"


# ------------------------------------------------------------------------------------------- MOT export (GT-free)
def _fixture(tmp_path: Path) -> Path:
    d = tmp_path / "T0-tracker-input-v1"
    d.mkdir()
    rows = [{"frame_num": 0, "players": [{"track_id": 7, "bbox_xyxy_px": [10.0, 20.0, 30.5, 70.25], "confidence": 0.81234},
                                         {"track_id": 3, "bbox_xyxy_px": [100, 50, 120, 95], "confidence": 0.5}]},
            {"frame_num": 1, "players": []},
            {"frame_num": 2, "players": [{"track_id": 7, "bbox_xyxy_px": [12.0, 21.0, 32.0, 71.0], "confidence": 0.7}]}]
    (d / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (d / "manifest.json").write_text(json.dumps({"configuration": {"confidence_threshold": 0.4}}), encoding="utf-8")
    return d


def test_mot_export_format_frame_offset_and_round_trip(tmp_path, monkeypatch):
    import football_fatigue.vision.mot_eval as mot
    from football_fatigue.video.bench.mot_export import export_dir, read_mot

    monkeypatch.setattr(mot, "load_reference", lambda *a, **k: (_ for _ in ()).throw(AssertionError("GT read")))
    out = export_dir(_fixture(tmp_path), "soccertrack-117093", tmp_path / "mot", "T0")
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines == ["1,3,100.000,50.000,20.000,45.000,0.5000,-1,-1,-1",
                     "1,7,10.000,20.000,20.500,50.250,0.8123,-1,-1,-1",
                     "3,7,12.000,21.000,20.000,50.000,0.7000,-1,-1,-1"]
    back = read_mot(out)
    assert sorted(back) == [0, 2] and back[0][1][0] == 7 and back[0][1][1] == pytest.approx((10, 20, 30.5, 70.25))
    side = json.loads((tmp_path / "mot" / "soccertrack-117093.mot.json").read_text(encoding="utf-8"))
    assert side["ground_truth_used"] is False and side["output"]["lines"] == 3 and side["output"]["temporary_ids"] == 2
    assert side["frame_mapping"] == "mot_frame = source_frame + 1"
    with pytest.raises(FileExistsError):                              # write-once
        export_dir(tmp_path / "T0-tracker-input-v1", "soccertrack-117093", tmp_path / "mot", "T0")


def test_mot_export_rejects_degenerate_boxes(tmp_path):
    from football_fatigue.video.bench.mot_export import mot_lines

    p = tmp_path / "frames.jsonl"
    p.write_text(json.dumps({"frame_num": 0, "players": [{"track_id": 1, "bbox_xyxy_px": [5, 5, 5, 9], "confidence": 1}]}) + "\n")
    with pytest.raises(ValueError):
        list(mot_lines(p))


def test_mot_export_refuses_reserved_match(tmp_path):
    from football_fatigue.video.bench.mot_export import export
    from football_fatigue.video.sources import SourceError

    with pytest.raises((SourceError, KeyError, ValueError)):
        export(Path(DATA_ROOT or tmp_path), tmp_path, "soccertrack-118577")


# ------------------------------------------------------------------------------------------- A2-cap v2
def _raw(tile, anchor, cls, conf, box, kept):
    return [tile, anchor, cls, conf, [0, 0, 0, 0], *box, int(kept)]


def test_attribution_separates_every_stage():
    from football_fatigue.video.detection.capture_v2 import attribute_frame

    gt = [(1, [0, 0, 20, 50]), (2, [100, 0, 120, 50]), (3, [200, 0, 220, 50]), (4, [300, 0, 320, 50]),
          (5, [400, 0, 420, 50]), (6, [403, 1, 423, 51]), (7, [500, 0, 520, 50]), (8, [600, 0, 620, 50]),
          (9, [700, 0, 720, 50])]
    raw = [_raw(0, 0, 1, 0.9, [0, 0, 20, 50], 1),          # 0: G1 detected, passes T0
           _raw(0, 1, 1, 0.3, [100, 0, 120, 50], 1),       # 1: G2 final but conf 0.3 -> rejected downstream
           _raw(0, 2, 1, 0.8, [207, 0, 227, 50], 1),       # 2: G3 IoU ~0.48 only -> poor localisation (and a FP)
           _raw(0, 3, 1, 0.9, [400, 0, 420, 50], 1),       # 3: G5 kept
           _raw(0, 4, 1, 0.6, [403, 1, 423, 51], 0),       # 4: G6's proposal (IoU 0.71 with G5's), removed by tile NMS
           _raw(0, 5, 1, 0.7, [500, 0, 520, 50], 1),       # 5: G7 kept by tile NMS, removed by the merge
           _raw(1, 0, 1, 0.75, [501, 0, 521, 50], 1),      # 6: final box near G7 from tile 1
           _raw(0, 6, 1, 0.08, [600, 0, 620, 50], 1),      # 7: G8 kept, but below the 0.1 cache floor
           _raw(0, 7, 0, 0.9, [700, 0, 720, 50], 1)]       # 8: G9 only a 'ball' proposal -> no person proposal
    rec = {"raw": raw, "final": [0, 1, 2, 3, 6]}
    cat = {r["gid"]: r for r in attribute_frame(gt, rec, NAMES)}
    assert cat[1]["category"] == "detected_and_passed"
    assert cat[2]["category"] == "rejected_downstream_conf"
    assert cat[3]["category"] == "poor_localisation"
    assert cat[4]["category"] == "no_raw_proposal"
    assert cat[5]["category"] == "detected_and_passed"
    assert cat[6]["category"] == "removed_by_tile_nms" and cat[6]["suppressor_gid"] == 5
    assert cat[7]["category"] == "detected_and_passed"              # the tile-1 box covers G7 (IoU 0.9) -> found
    assert cat[8]["category"] == "below_cache_floor"
    assert cat[9]["category"] == "no_raw_proposal"                    # class matters: a 'ball' box is no person proposal


def test_attribution_cross_tile_merge_and_shared_box():
    from football_fatigue.video.detection.capture_v2 import attribute_frame

    gt = [(1, [0, 0, 20, 50]), (2, [300, 0, 320, 50]), (3, [304, 0, 324, 50])]
    raw = [_raw(0, 0, 1, 0.7, [0, 0, 20, 50], 1),             # G1: kept by tile NMS, removed by merge (not final)
           _raw(1, 0, 1, 0.9, [40, 0, 60, 50], 1),            # unrelated final box
           _raw(0, 1, 1, 0.9, [302, 0, 322, 50], 1)]          # one final box shared by G2 and G3
    cat = {r["gid"]: r["category"] for r in attribute_frame(gt, {"raw": raw, "final": [1, 2]}, NAMES)}
    assert cat[1] == "removed_by_cross_tile_merge"
    assert sorted([cat[2], cat[3]]) == ["detected_and_passed", "shared_box_given_to_neighbour"]


def test_tile_records_recompute_ultralytics_nms_exactly():
    torch = pytest.importorskip("torch")
    from football_fatigue.video.detection.capture_v2 import _tile_records

    # one tile, letterbox 64x64 == tile 64x64 (no scaling); 3 anchors, 4 classes, xywh
    pred = torch.zeros(1, 8, 3)
    pred[0, :4, 0] = torch.tensor([20.0, 20.0, 10.0, 20.0]); pred[0, 5, 0] = 0.9     # player, kept
    pred[0, :4, 1] = torch.tensor([21.0, 20.0, 10.0, 20.0]); pred[0, 5, 1] = 0.6     # player, IoU>0.7 -> suppressed
    pred[0, :4, 2] = torch.tensor([50.0, 40.0, 8.0, 16.0]); pred[0, 4, 2] = 0.04     # below the 0.05 floor
    call = (pred, (64, 64), [(64, 64)], dict(conf=0.05, iou=0.7, agnostic=False, max_det=300, classes=None, end2end=False))
    results = [SimpleNamespace(boxes=SimpleNamespace(xyxy=torch.tensor([[15.0, 10.0, 25.0, 30.0]])))]
    raws, kept, st = _tile_records(call, [(100, 7, 64, 64)], results, 0)
    assert [(r[1], r[2], r[9]) for r in raws] == [(0, 1, 1), (1, 1, 0)]
    assert raws[0][5:9] == [115.0, 17.0, 125.0, 37.0]                   # tile offset applied
    assert st["nms_equal_results"] and kept == [1]
    with pytest.raises(RuntimeError):
        _tile_records((pred, (64, 64), [(64, 64)], {**call[3], "end2end": True}), [(0, 0, 64, 64)], results, 0)


def test_capture2_cli_refuses_gpu_and_full_capture_without_approval():
    from football_fatigue.video.detection.__main__ import main

    with pytest.raises(SystemExit):
        main(["capture2", "--data-root", "x", "--plan", "p.json"])                     # GPU, no --approved
    with pytest.raises(SystemExit):
        main(["capture2", "--data-root", "x", "--device", "cpu", "--plan", "p.json"])  # full capture, no --approved
    with pytest.raises(SystemExit):
        main(["capture2", "--data-root", "x", "--device", "cuda:0", "--smoke", "1", "--source", "soccertrack-117093",
              "--start", "0"])                                                         # GPU smoke also gated


# ------------------------------------------------------------------------------------------- saved outputs
@needs_data
def test_m12_historical_outputs_unchanged():
    from football_fatigue.video.failures.baseline import verify

    if not (OUT / "_diag" / "m13-baseline-v1" / "manifest.json").exists():
        pytest.skip("baseline manifest not built")
    r = verify(OUT)
    assert r["unchanged"], r["changed"]


@needs_data
def test_baseline_claims_have_profiles_and_denominators():
    p = OUT / "_diag" / "m13-baseline-v1" / "manifest.json"
    if not p.exists():
        pytest.skip("baseline manifest not built")
    m = json.loads(p.read_text(encoding="utf-8"))
    for c in m["claims"]:
        assert c["profile"] and ("denominator" in c or "tune_frag" in c or "HOTA" in c)
    miss = next(c for c in m["claims"] if c["claim"].startswith("no post-merge"))
    assert miss["count"] == 33579 and miss["denominator"] == 526760 and "NOT a confirmed" in miss["caveat"]


@needs_data
def test_a2cap_plan_has_targeted_and_random_windows_without_overlap():
    p = OUT / "_diag" / "a2cap-plan-v1" / "plan.json"
    if not p.exists():
        pytest.skip("plan not built")
    plan = json.loads(p.read_text(encoding="utf-8"))
    for sid in {w["source_id"] for w in plan["windows"]}:
        ws = sorted((w["start"], w["end"], w["kind"]) for w in plan["windows"] if w["source_id"] == sid)
        assert {k for _a, _b, k in ws} == {"overlap", "small_or_clear", "random_a", "random_b"}
        assert all(a[1] < b[0] for a, b in zip(ws, ws[1:]))
    assert not any(w["source_id"] in ("soccertrack-118577", "soccertrack-118578") for w in plan["windows"])


@needs_data
def test_activation_profile_manifests_record_both_thresholds():
    for sid in ("soccertrack-117093", "soccertrack-118576"):
        p = OUT / sid / "tracking" / "T0-A03-tracker-input-v1" / "manifest.json"
        if not p.exists():
            pytest.skip("activation profiles not built")
        m = json.loads(p.read_text(encoding="utf-8"))
        assert m["configuration"]["tracker_params"]["activation"] == 0.3
        assert m["configuration"]["production_tracker_params"]["activation"] == 0.5
        assert m["configuration"]["confidence_threshold"] == 0.4
        assert any("0.5 -> 0.3" in x for x in m["derived_from"]["changed"])


# ------------------------------------------------------------------------------------------- Session A/B contract
def test_ab_tracklets_are_observed_pixels_only_without_calibration(tmp_path):
    from football_fatigue.video.ab_tracklets import candidate, hidden_intervals, tracklets

    p = tmp_path / "frames.jsonl"
    rows = [{"frame_num": 0, "players": [{"track_id": 5, "bbox_xyxy_px": [10, 20, 30, 80], "confidence": 0.9, "measured": True}]},
            {"frame_num": 1, "players": [{"track_id": 5, "bbox_xyxy_px": [11, 20, 31, 80], "confidence": 0.8, "measured": False}]},
            {"frame_num": 4, "players": [{"track_id": 5, "bbox_xyxy_px": [14, 20, 34, 80], "confidence": 0.7, "measured": True}]}]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    t = tracklets(p, fps=25.0, pts_offset_s=0.979)[5]
    assert t["frame"] == [0, 4]                                        # the tracker prediction at f1 is not exported
    assert t["timestamp_s"] == [0.979, pytest.approx(0.979 + 4 / 25)]
    assert t["pixel"][0] == {"x_px": 20.0, "y_px": 80.0, "box_h_px": 60.0, "det_conf": 0.9}
    assert t["pitch"] == [None, None] and set(t["position_status"]) == {"observed"}   # no pixel in the metric field
    assert hidden_intervals(t, 25.0) == [(pytest.approx(0.979 + 1 / 25), pytest.approx(0.979 + 3 / 25))]
    out = candidate(5, t, incoming=False)
    assert out["identity"]["global_id"] is None and out["identity"]["identity_status"] == "anonymous"
    assert candidate(5, t, incoming=True)["identity"]["identity_status"] == "local_only"
    cal = tracklets(p, 25.0, 0.0, calibrate=lambda x, y: (x / 10, y / 10, 0.5))[5]["pitch"][0]
    assert cal == {"x_m": 2.0, "y_m": 8.0, "sigma_m": 0.5}
