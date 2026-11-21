"""Tests for the opt-in M7-0 pixel-space identity-provenance export. All scenarios are SYNTHETIC."""

import copy
import json

import pytest

from football_fatigue.vision.identity import IdentityConfig, annotate_records, observations_from_records, solve
from football_fatigue.vision.identity_export import (
    EXPORT_SCHEMA_VERSION,
    ExportConfig,
    build_export,
    main,
    read_export,
    segment_path_length_px,
    validate_export,
    write_export,
)

H = 40.0


def player(local, x, y, *, tracklet=1, gid=1, status="confirmed", assoc="none", **extra):
    p = {"track_id": gid, "local_track_id": local, "tracklet_id": tracklet, "global_id": gid, "identity_status": status,
         "identity_confidence": 0.99, "identity_confidence_local": 0.99, "position_status": "observed",
         "bbox_xyxy_px": [x - 8, y - H, x + 8, y], "detected_class": "player"}
    if assoc is not None:
        p["association_status"] = assoc
    p.update(extra)
    return p


def rec(frame, players, **extra):
    return dict({"frame_num": frame, "timestamp_s": frame / 25.0, "fps": 25.0, "players": players}, **extra)


def walk(n, start=0, local=7, **kw):
    return [rec(start + i, [player(local, 100 + 2.0 * (start + i), 300, **kw)]) for i in range(n)]


def segs(exp):
    return exp["segments"]


def test_clean_confirmed_track_is_one_attributable_segment_with_metadata():
    exp = build_export(walk(40), source={"match_id": "M", "camera_id": "pano"})
    assert validate_export(exp) == []
    (s,) = segs(exp)
    assert s["n_samples"] == 40 and s["global_id_use"] == "attributable" and s["global_id_use_reasons"] == []
    assert (s["local_track_id"], s["tracklet_id"], s["global_id"]) == (7, 1, 1)
    assert s["segment_id"].startswith("M:") and s["camera_id"] == "pano"
    assert s["x_px"][0] == 100.0 and s["y_px"][0] == 300.0          # foot point = bbox bottom-centre
    cs = exp["coordinate_system"]
    assert cs["unit"] == "pixel" and cs["metric_valid"] is False and "top-left" in cs["origin"] and cs["pitch_calibration"] is None
    assert exp["claims"] == {"metric_distance": False, "global_identity_validated": False, "fatigue_or_load_metric": False}
    assert exp["schema_version"] == EXPORT_SCHEMA_VERSION and exp["known_limitations"]
    assert segment_path_length_px(s) == pytest.approx(2.0 * 39)


def test_short_trusted_segment_is_local_only():
    (s,) = segs(build_export(walk(10)))
    assert s["global_id_use"] == "local_only" and s["global_id_use_reasons"] == ["short_segment"]


def test_unvalidated_link_is_local_only_and_cuts_trusted_run():
    recs = walk(30, 0) + walk(30, 30, assoc="linked_unvalidated", association_link_confidence=0.41)
    exp = build_export(recs)
    a, b = segs(exp)
    assert (a["global_id_use"], b["global_id_use"]) == ("attributable", "local_only")
    assert b["global_id_use_reasons"] == ["linked_unvalidated"] and b["start_break"] == "association_change"
    assert b["association_link_confidence"][0] == 0.41           # link confidence kept apart from local confidence
    assert b["identity_confidence_local"][0] == 0.99


def test_status_not_confirmed_and_missing_provenance_are_not_trusted():
    (p,) = segs(build_export(walk(40, status="provisional")))
    assert p["global_id_use_reasons"] == ["status_not_confirmed"]
    (q,) = segs(build_export(walk(40, assoc=None)))
    assert q["global_id_use"] == "local_only" and q["global_id_use_reasons"] == ["provenance_not_recorded"]
    (r,) = segs(build_export(walk(40, assoc=None), config=ExportConfig(require_provenance=False)))
    assert r["global_id_use"] == "attributable"


def test_tracklet_transition_is_never_bridged():
    recs = [rec(i, [player(7 if i < 30 else 9, 100 + 2.0 * i, 300, tracklet=1 if i < 30 else 2, gid=5)]) for i in range(60)]
    exp = build_export(recs)
    a, b = segs(exp)
    assert a["global_id"] == b["global_id"] == 5 and a["tracklet_id"] != b["tracklet_id"]
    (t,) = exp["transitions"]
    assert t["kind"] == "tracklet_transition" and t["bridged"] is False and t["gap_frames"] == 1
    assert exp["summary"]["transitions_blocked"] == 1 and exp["summary"]["transitions_bridged"] == 0


def test_tracklet_split_of_one_local_track_is_cut():
    recs = [rec(i, [player(7, 100 + 2.0 * i, 300, tracklet=1 if i < 30 else 2, gid=5)]) for i in range(60)]
    exp = build_export(recs)
    assert len(segs(exp)) == 2 and exp["transitions"][0]["kind"] == "tracklet_transition"


def test_identity_swap_with_coordinate_jump_and_global_id_change_are_cut():
    jump = [rec(i, [player(7, 100 + 2.0 * i + (400 if i >= 30 else 0), 300)]) for i in range(60)]       # tracker swap: players 400 px apart
    exp = build_export(jump)
    a, b = segs(exp)
    assert b["start_break"] == "coordinate_discontinuity" and a["frame_end"] == 29 and b["frame_start"] == 30
    gid_change = [rec(i, [player(7, 100 + 2.0 * i, 300, gid=1 if i < 30 else 2)]) for i in range(60)]
    exp2 = build_export(gid_change)
    assert [s["start_break"] for s in segs(exp2)] == ["start_of_track", "global_id_change"]
    assert exp2["transitions"] == []                                  # different global ids: not one trajectory


def test_smooth_interior_swap_is_not_detected_and_is_documented():
    # a swap without any observable discontinuity cannot be cut (M6.3-D); the contract says so
    exp = build_export(walk(60))
    assert len(segs(exp)) == 1
    assert any("swaps inside a tracklet" in s for s in exp["known_limitations"])


def test_missing_frames_cut_unless_explicitly_tolerated():
    recs = [r for r in walk(60) if r["frame_num"] not in (30, 31)]
    a, b = segs(build_export(recs))
    assert b["start_break"] == "missing_observations" and (a["frame_end"], b["frame_start"]) == (29, 32)
    assert len(segs(build_export(recs, config=ExportConfig(max_missing_frames=2)))) == 1
    assert len(segs(build_export(recs, config=ExportConfig(max_missing_frames=1)))) == 2


def test_interpolated_positions_never_enter_segments():
    recs = walk(60)
    for r in recs[30:32]:
        r["players"] = []
    recs[30]["identity_interpolated"] = [{"global_id": 1, "position_status": "interpolated", "bbox_xyxy_px": [150, 260, 166, 300]}]
    recs[31]["players"] = [player(7, 162, 300, position_status="interpolated")]
    exp = build_export(recs)
    assert validate_export(exp) == []
    assert all(f not in (30, 31) for s in segs(exp) for f in s["frames"])
    assert len(segs(exp)) == 2 and segs(exp)[1]["start_break"] == "missing_observations"
    assert exp["summary"]["interpolated_samples"] == 1 and exp["interpolated_samples"][0]["position_status"] == "interpolated"
    assert exp["summary"]["excluded"]["not_observed_position"] == 1


def test_same_global_id_different_local_ids():
    seq = [rec(i, [player(7 if i < 30 else 9, 100 + 2.0 * i, 300, tracklet=1 if i < 30 else 2, gid=5)]) for i in range(60)]
    assert {s["local_track_id"] for s in segs(build_export(seq))} == {7, 9}
    both = [rec(i, [player(7, 100, 300, tracklet=1, gid=5), player(9, 600, 300, tracklet=2, gid=5)]) for i in range(40)]
    exp = build_export(both)
    assert [t["kind"] for t in exp["transitions"]] == ["overlap_conflict"] and exp["summary"]["overlap_conflicts"] == 1
    assert all(t["bridged"] is False for t in exp["transitions"])


def test_camera_change_cuts_the_segment():
    recs = walk(60)
    for r in recs[30:]:
        r["camera_id"] = "cam_B"
    for r in recs[:30]:
        r["camera_id"] = "cam_A"
    a, b = segs(build_export(recs))
    assert (a["camera_id"], b["camera_id"], b["start_break"]) == ("cam_A", "cam_B", "camera_change")


def test_plain_tracker_records_are_local_only():
    plain = [rec(i, [{"track_id": 3, "bbox_xyxy_px": [92, 260, 108, 300], "detected_class": "player"}]) for i in range(40)]
    exp = build_export(plain)
    (s,) = segs(exp)
    assert s["global_id"] is None and s["tracklet_id"] is None and s["local_track_id"] == 3
    assert s["global_id_use"] == "local_only" and s["global_id_use_reasons"] == ["not_identity_annotated"]
    assert exp["transitions"] == []


def test_ball_and_invalid_boxes_are_excluded_and_input_is_not_mutated():
    recs = walk(5)
    recs[0]["players"].append({"detected_class": "ball", "bbox_xyxy_px": [1, 1, 2, 2], "track_id": 99})
    recs[1]["players"].append({"track_id": 4, "bbox_xyxy_px": [float("nan"), 0, 1, 1], "detected_class": "player"})
    before = copy.deepcopy(recs)
    exp = build_export(recs)
    assert recs == before and exp["summary"]["samples"] == 5 and exp["summary"]["excluded"]["no_valid_bbox"] == 1


def test_serialization_round_trip_and_no_overwrite(tmp_path):
    exp = build_export(walk(40) + walk(30, 40, assoc="linked_unvalidated"), source={"match_id": "M"})
    path = tmp_path / "export.json"
    write_export(path, exp)
    assert read_export(path) == json.loads(json.dumps(exp))
    with pytest.raises(FileExistsError):
        write_export(path, exp)


def test_validation_rejects_metric_claims_bridges_and_ragged_arrays():
    exp = build_export(walk(40))
    bad = copy.deepcopy(exp)
    bad["coordinate_system"]["metric_valid"] = True
    assert validate_export(bad)
    bad = copy.deepcopy(exp)
    bad["claims"]["metric_distance"] = True
    assert validate_export(bad)
    bad = copy.deepcopy(exp)
    bad["segments"][0]["frames"][10] += 5                              # silently bridged gap
    assert any("bridges" in e for e in validate_export(bad))
    bad = copy.deepcopy(exp)
    bad["segments"][0]["x_px"].pop()
    assert any("x_px" in e for e in validate_export(bad))
    bad = copy.deepcopy(exp)
    bad["segments"][0]["identity_status"][3] = "provisional"
    assert any("attributable but" in e for e in validate_export(bad))
    with pytest.raises(ValueError):
        write_export("never_written.json", bad)


def test_export_of_solver_output_accounts_for_every_box():
    def b(x, y):
        return [x - 8, y - H, x + 8, y]
    recs = [rec(i, [{"track_id": 1, "bbox_xyxy_px": b(100 + 2.0 * i, 300), "detected_class": "player"},
                    {"track_id": 2, "bbox_xyxy_px": b(900 - 2.0 * i, 500), "detected_class": "player"}]) for i in range(120)]
    res = solve(observations_from_records(recs), IdentityConfig())
    ann = annotate_records(recs, res["boxes"])
    exp = build_export(ann)
    assert validate_export(exp) == [] and exp["summary"]["samples"] == 240
    assert {s["local_track_id"] for s in segs(exp)} == {1, 2}
    # default-config solver output carries no association provenance, so nothing is attributable by default
    assert exp["summary"]["attributable_segments"] == 0
    ann2 = annotate_records(recs, {k: dict(v, association="none", identity_confidence_uncapped=v["identity_confidence"]) for k, v in res["boxes"].items()})
    assert build_export(ann2)["summary"]["attributable_segments"] <= 2


def test_cli_writes_new_file_and_refuses_to_overwrite(tmp_path, capsys):
    frames = tmp_path / "frames.jsonl"
    frames.write_text("\n".join(json.dumps(r) for r in walk(40)), encoding="utf-8")
    out = tmp_path / "out.json"
    assert main([str(frames), "--output", str(out), "--match-id", "M"]) == 0
    exp = read_export(out)
    assert exp["source"]["frames_sha256"] and exp["source"]["match_id"] == "M"
    with pytest.raises(FileExistsError):
        main([str(frames), "--output", str(out)])


def test_default_analytics_do_not_depend_on_the_export():
    import football_fatigue.analytics.cv as cv
    src = open(cv.__file__, encoding="utf-8").read()
    assert "identity_export" not in src
