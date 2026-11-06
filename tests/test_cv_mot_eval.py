"""Synthetic checks of the standard MOT evaluation adapter (TrackEval HOTA/CLEAR/Identity).

Every scenario has hand-computable answers. Boxes are 20x40 px squares of a fixed layout, so
IoU is exactly 1 for a correct prediction and 0 for anything elsewhere.
"""

import json

import pytest

pytest.importorskip("trackeval")
pytest.importorskip("scipy")

from football_fatigue.vision.mot_eval import (  # noqa: E402
    MotEvalError,
    build_sequence_data,
    evaluate_run,
    export_motchallenge,
    load_predictions,
    load_reference,
    standard_metrics,
)


def box(slot):
    x = 100 * slot
    return [x, 50, x + 20, 90]


def write_reference(path, players, frames, base=0):
    """players: {id: slot}; every player is present in every frame."""
    with path.open("w", encoding="utf-8") as out:
        for f in range(frames):
            for ident, slot in players.items():
                x1, y1, x2, y2 = box(slot)
                out.write(f"{f + base},{ident},{x1},{y1},{x2 - x1},{y2 - y1},-1,-1,-1,-1\n")


def write_frames(path, per_frame, frames, skip_records=()):
    """per_frame: f -> list of (track_id, slot, extra dict); a missing f means no tracked players."""
    with path.open("w", encoding="utf-8") as out:
        for f in range(frames):
            if f in skip_records:
                continue
            players = []
            for tid, slot, extra in per_frame.get(f, []):
                p = {"track_id": tid, "bbox_xyxy_px": box(slot), "detected_class": "player", "confidence": 0.9}
                p.update(extra)
                players.append(p)
            out.write(json.dumps({"frame_num": f, "players": players}) + "\n")


def run(tmp_path, players, per_frame, frames=10, **kwargs):
    ref, pred = tmp_path / "gt.txt", tmp_path / "frames.jsonl"
    write_reference(ref, players, frames)
    write_frames(pred, per_frame, frames, kwargs.pop("skip_records", ()))
    return evaluate_run(pred, ref, include_approximate=kwargs.pop("include_approximate", False), **kwargs)


def tracking(frames, id_of, slots, absent=None):
    """Perfect per-frame predictions; id_of(slot, f) names the tracker id; absent: {(slot, f)} to drop."""
    absent = absent or set()
    return {f: [(id_of(s, f), s, {}) for s in slots if (s, f) not in absent] for f in range(frames)}


def metrics(report):
    m = report["standard_metrics"]
    return m["hota"], m["clear"], m["identity"]


def test_perfect_tracking_scores_one(tmp_path):
    report = run(tmp_path, {1: 0, 2: 1, 3: 2}, tracking(10, lambda s, f: 50 + s, [0, 1, 2]))
    hota, clear, identity = metrics(report)
    assert hota["HOTA"] == hota["DetA"] == hota["AssA"] == hota["LocA"] == 1.0
    assert identity["IDF1"] == 1.0 and clear["MOTA"] == 1.0 and clear["IDSW"] == 0 and clear["Frag"] == 0
    assert (clear["CLR_TP"], clear["CLR_FP"], clear["CLR_FN"]) == (30, 0, 0)
    assert report["standard_metrics"]["counts"] == {"gt_ids": 3, "tracker_ids": 3, "gt_dets": 30, "tracker_dets": 30, "timesteps": 10}
    assert report["evaluator"]["trackeval"] and report["schema_version"] == 1


def test_missed_detections(tmp_path):
    absent = {(1, f) for f in range(5)}  # one of two players missed in half the frames
    report = run(tmp_path, {1: 0, 2: 1}, tracking(10, lambda s, f: s + 1, [0, 1], absent))
    hota, clear, identity = metrics(report)
    assert (clear["CLR_TP"], clear["CLR_FP"], clear["CLR_FN"]) == (15, 0, 5)
    assert clear["CLR_Re"] == 0.75 and clear["MOTA"] == 0.75
    # HOTA association counts a missed frame against the identity: player 1 is associated in 5 of its 10
    # frames (0.5), player 2 in all 10 (1.0), averaged over the 15 matched detections = 5/6.
    assert hota["DetA"] == 0.75 and hota["AssA"] == pytest.approx(5 / 6, abs=1e-5)
    assert hota["HOTA"] == pytest.approx((0.75 * 5 / 6) ** 0.5, abs=1e-5)
    assert identity["IDF1"] == pytest.approx(30 / 35, abs=1e-5) and clear["IDSW"] == 0


def test_false_positives(tmp_path):
    per_frame = tracking(10, lambda s, f: s + 1, [0, 1])
    for f in range(4):
        per_frame[f].append((99, 7, {}))  # a person nobody annotated
    hota, clear, identity = metrics(run(tmp_path, {1: 0, 2: 1}, per_frame))
    assert (clear["CLR_TP"], clear["CLR_FP"], clear["CLR_FN"]) == (20, 4, 0)
    assert clear["CLR_Pr"] == pytest.approx(20 / 24, abs=1e-5) and clear["MOTA"] == pytest.approx(1 - 4 / 20, abs=1e-5)
    assert hota["DetA"] == pytest.approx(20 / 24, abs=1e-5)
    assert identity["IDF1"] == pytest.approx(40 / 44, abs=1e-5)


def test_id_switch(tmp_path):
    hota, clear, identity = metrics(run(tmp_path, {1: 0}, tracking(10, lambda s, f: 1 if f < 5 else 2, [0])))
    assert clear["IDSW"] == 1 and clear["Frag"] == 0 and clear["CLR_FN"] == 0
    assert hota["DetA"] == 1.0 and hota["AssA"] == pytest.approx(0.5, abs=1e-5)
    assert hota["HOTA"] == pytest.approx(0.5 ** 0.5, abs=1e-5) and identity["IDF1"] == pytest.approx(0.5, abs=1e-5)


def test_fragmentation_keeps_the_id_across_a_gap(tmp_path):
    # a second player is always tracked so every frame has predictions (see the next test)
    absent = {(0, f) for f in (3, 4, 5)}
    hota, clear, identity = metrics(run(tmp_path, {1: 0, 2: 1}, tracking(10, lambda s, f: s + 1, [0, 1], absent)))
    assert clear["Frag"] == 1 and clear["IDSW"] == 0 and (clear["CLR_TP"], clear["CLR_FN"]) == (17, 3)
    assert identity["IDF1"] == pytest.approx(34 / 37, abs=1e-5) and hota["AssA"] == pytest.approx((7 * 0.7 + 10) / 17, abs=1e-5)


def test_fragmentation_with_a_new_id_after_the_gap(tmp_path):
    absent = {(0, f) for f in (3, 4, 5)}
    ids = lambda s, f: 10 if s == 1 else (1 if f < 3 else 2)  # noqa: E731
    _, clear, _ = metrics(run(tmp_path, {1: 0, 2: 1}, tracking(10, ids, [0, 1], absent)))
    assert clear["IDSW"] == 1 and clear["Frag"] == 1


def test_trackeval_does_not_count_a_gap_in_which_nothing_at_all_is_predicted(tmp_path):
    # Upstream TrackEval behaviour, pinned so a version change that alters it is noticed: a timestep with no
    # prediction at all does not reset the previous-match state, so this gap is a plain miss, not a fragmentation.
    absent = {(0, f) for f in (3, 4, 5)}
    _, clear, _ = metrics(run(tmp_path, {1: 0}, tracking(10, lambda s, f: 1, [0], absent)))
    assert clear["Frag"] == 0 and (clear["CLR_TP"], clear["CLR_FN"]) == (7, 3)


def test_multiple_players_with_a_swap_between_two_of_them(tmp_path):
    # players 1 and 2 cross identities at frame 5; player 3 stays correct
    def ident(slot, f):
        return {0: 1 if f < 5 else 2, 1: 2 if f < 5 else 1, 2: 3}[slot]

    hota, clear, identity = metrics(run(tmp_path, {1: 0, 2: 1, 3: 2}, tracking(10, ident, [0, 1, 2])))
    assert clear["IDSW"] == 2 and (clear["CLR_FP"], clear["CLR_FN"]) == (0, 0) and hota["DetA"] == 1.0
    # best identity assignment keeps half of the two swapped players and all of the third
    assert identity["IDF1"] == pytest.approx(2 * 20 / 60, abs=1e-5) or identity["IDTP"] == 20
    assert 0 < hota["AssA"] < 1


def test_window_offsets_and_gt_frame_base(tmp_path):
    ref, pred = tmp_path / "gt.txt", tmp_path / "frames.jsonl"
    write_reference(ref, {1: 0, 2: 1}, 10, base=1)  # reference counts frames from 1
    # prediction is wrong (wrong slot) before frame 5 and perfect from frame 5 on
    per_frame = {f: [(1, 0, {}), (2, 1, {})] if f >= 5 else [(1, 8, {}), (2, 9, {})] for f in range(10)}
    write_frames(pred, per_frame, 10)
    late = evaluate_run(pred, ref, start=5, end=10, gt_frame_base=1, include_approximate=False)
    early = evaluate_run(pred, ref, start=0, end=5, gt_frame_base=1, include_approximate=False)
    assert metrics(late)[1]["CLR_TP"] == 10 and metrics(late)[0]["HOTA"] == 1.0
    assert late["standard_metrics"]["counts"]["timesteps"] == 5
    assert late["configuration"]["window"] == {"start": 5, "end_exclusive": 10, "frames": 5, "frame_indexing": "0-based decoded frame index"}
    assert metrics(early)[1]["CLR_TP"] == 0 and metrics(early)[1]["CLR_FN"] == 10
    # the base only renumbers the reference: 1-based files start at frame 0 after conversion
    assert min(load_reference(ref, 1)[0]) == 0 and min(load_reference(ref, 0)[0]) == 1
    zero_based = tmp_path / "zero_based.txt"
    zero_based.write_text("0,1,0,0,5,5\n", encoding="utf-8")
    with pytest.raises(MotEvalError, match="below gt_frame_base"):
        load_reference(zero_based, 1)


def test_frames_without_a_prediction_record_count_as_misses(tmp_path):
    report = run(tmp_path, {1: 0}, tracking(10, lambda s, f: 1, [0]), skip_records={4, 5})
    assert report["input_handling"]["predictions"]["window_frames_without_prediction_record"] == 2
    assert metrics(report)[1]["CLR_FN"] == 2 and report["standard_metrics"]["counts"]["timesteps"] == 10


def test_open_end_covers_every_frame_seen_and_invalid_windows_are_rejected(tmp_path):
    report = run(tmp_path, {1: 0}, tracking(10, lambda s, f: 1, [0]))
    assert report["configuration"]["window"]["end_exclusive"] == 10  # default end: last frame seen + 1
    with pytest.raises(MotEvalError, match="empty evaluation window"):
        run(tmp_path, {1: 0}, tracking(10, lambda s, f: 1, [0]), start=6, end=6)


def test_class_view_confidence_and_invalid_input_handling(tmp_path):
    per_frame = {
        0: [
            (1, 0, {}),
            (2, 1, {"detected_class": "goalkeeper"}),
            (3, 2, {"detected_class": "referee"}),
            (4, 3, {"detected_class": "ball"}),
            (5, 4, {"confidence": 0.3}),
            (6, 5, {"bbox_xyxy_px": [10, 10, 10, 40]}),          # zero width
            (7, 6, {"bbox_xyxy_px": [float("nan"), 0, 5, 5]}),
            (8, 7, {"bbox_xyxy_px": None}),
        ]
    }
    pred = tmp_path / "frames.jsonl"
    write_frames(pred, per_frame, 1)
    everything, stats = load_predictions(pred, 0, None, "all_non_ball")
    assert sorted(i for i, _, _ in everything[0]) == [1, 2, 3, 5]
    assert stats["dropped_ball"] == 1 and stats["dropped_invalid_box"] == 2 and stats["dropped_no_box"] == 1
    players, stats = load_predictions(pred, 0, None, "player_goalkeeper")
    assert sorted(i for i, _, _ in players[0]) == [1, 2, 5] and stats["dropped_other_class"] == 1
    confident, stats = load_predictions(pred, 0, None, "all_non_ball", min_confidence=0.3)
    assert 5 not in [i for i, _, _ in confident[0]] and stats["dropped_below_confidence"] == 1  # strict >
    with pytest.raises(MotEvalError, match="unknown class view"):
        load_predictions(pred, 0, None, "everything")


def test_duplicate_ids_in_a_frame_are_errors(tmp_path):
    pred, ref = tmp_path / "frames.jsonl", tmp_path / "gt.txt"
    write_frames(pred, {0: [(1, 0, {}), (1, 1, {})]}, 1)
    with pytest.raises(MotEvalError, match="appears twice in frame 0"):
        load_predictions(pred, 0, None)
    ref.write_text("0,1,0,0,10,10\n0,1,50,0,10,10\n", encoding="utf-8")
    with pytest.raises(MotEvalError, match="appears twice in frame 0"):
        load_reference(ref)
    ref.write_text("0,1,0\n", encoding="utf-8")
    with pytest.raises(MotEvalError, match="at least 6"):
        load_reference(ref)


def test_reference_ignore_rows_and_invalid_boxes(tmp_path):
    ref = tmp_path / "gt.txt"
    ref.write_text("0,1,0,0,10,10,1,1,1\n0,2,0,0,10,10,0,1,1\n0,3,0,0,0,10,-1,-1,-1\n0,4,5,5,4,4\n", encoding="utf-8")
    frames, stats = load_reference(ref)
    assert [i for i, _ in frames[0]] == [1, 4]
    assert stats == {"rows": 4, "ignored_conf_zero": 1, "invalid_boxes": 1}


def test_approximate_project_metrics_stay_separate_from_standard_ones(tmp_path):
    report = run(tmp_path, {1: 0}, tracking(10, lambda s, f: 1 if f < 5 else 2, [0]), include_approximate=True)
    approx = report["approximate_project_metrics"]
    assert "idf1_approximate" in approx and "idf1" not in approx and approx["id_switches"] == 1
    assert "IDF1" in report["standard_metrics"]["identity"] and "idf1_approximate" not in report["standard_metrics"]["identity"]
    assert "not the standard IDF1" in approx["note"]


def test_in_memory_adapter_matches_trackeval_on_the_exported_motchallenge_layout(tmp_path):
    from trackeval import Evaluator, datasets
    from trackeval.metrics import HOTA, CLEAR, Identity

    per_frame = tracking(12, lambda s, f: {0: 1 if f < 6 else 9, 1: 2, 2: 3}[s], [0, 1, 2], {(1, 3), (1, 4), (2, 10)})
    per_frame[2].append((77, 8, {}))
    ref, pred = tmp_path / "gt.txt", tmp_path / "frames.jsonl"
    write_reference(ref, {1: 0, 2: 1, 3: 2}, 12)
    write_frames(pred, per_frame, 12)
    own = evaluate_run(pred, ref, start=2, end=12, include_approximate=False)["standard_metrics"]

    reference, _ = load_reference(ref)
    predictions, _ = load_predictions(pred, 0, None)
    layout = tmp_path / "layout"
    paths = export_motchallenge(reference, predictions, 2, 12, layout, sequence="seq", tracker="trk")
    assert (layout / "gt" / "seq" / "gt" / "gt.txt").exists()
    dataset = datasets.MotChallenge2DBox({
        "GT_FOLDER": str(layout / "gt"), "TRACKERS_FOLDER": str(layout / "trackers"), "OUTPUT_FOLDER": None,
        "TRACKERS_TO_EVAL": ["trk"], "CLASSES_TO_EVAL": ["pedestrian"], "BENCHMARK": "MOT17", "SPLIT_TO_EVAL": "train",
        "INPUT_AS_ZIP": False, "PRINT_CONFIG": False, "DO_PREPROC": True, "TRACKER_SUB_FOLDER": "data",
        "OUTPUT_SUB_FOLDER": "", "SKIP_SPLIT_FOL": True, "SEQMAP_FILE": paths["seqmap"], "SEQ_INFO": {"seq": 10},
        "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt/gt.txt",
    })
    official, _ = Evaluator({
        "USE_PARALLEL": False, "PRINT_RESULTS": False, "PRINT_ONLY_COMBINED": True, "PRINT_CONFIG": False,
        "TIME_PROGRESS": False, "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False, "PLOT_CURVES": False,
        "OUTPUT_EMPTY_CLASSES": False, "BREAK_ON_ERROR": True, "LOG_ON_ERROR": None,
    }).evaluate([dataset], [HOTA(), CLEAR({"PRINT_CONFIG": False}), Identity({"PRINT_CONFIG": False})])
    result = official["MotChallenge2DBox"]["trk"]["COMBINED_SEQ"]["pedestrian"]
    assert own["clear"]["IDSW"] == int(result["CLEAR"]["IDSW"]) == 1
    assert own["clear"]["CLR_FN"] == int(result["CLEAR"]["CLR_FN"]) and own["clear"]["CLR_FP"] == int(result["CLEAR"]["CLR_FP"])
    assert own["identity"]["IDF1"] == pytest.approx(float(result["Identity"]["IDF1"]), abs=1e-5)
    assert own["hota"]["HOTA"] == pytest.approx(float(sum(result["HOTA"]["HOTA"]) / 19), abs=1e-5)
    assert own["hota"]["AssA"] == pytest.approx(float(sum(result["HOTA"]["AssA"]) / 19), abs=1e-5)


def test_standard_metrics_are_deterministic_and_data_dict_is_zero_based(tmp_path):
    reference = {0: [(5, (0, 0, 10, 10))], 2: [(5, (0, 0, 10, 10))]}
    predictions = {0: [(9, (0, 0, 10, 10), 0.9)], 2: [(9, (0, 0, 10, 10), 0.9)]}
    data = build_sequence_data(reference, predictions, 0, 3)
    assert data["num_timesteps"] == 3 and len(data["gt_ids"][1]) == 0 and data["similarity_scores"][1].shape == (0, 0)
    assert standard_metrics(data) == standard_metrics(build_sequence_data(reference, predictions, 0, 3))
    assert standard_metrics(data)["clear"]["CLR_TP"] == 2


def test_cli_writes_a_versioned_report_and_never_overwrites(tmp_path, capsys):
    from football_fatigue.vision.mot_eval import main

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_reference(tmp_path / "gt.txt", {1: 0}, 6)
    write_frames(run_dir / "frames.jsonl", tracking(6, lambda s, f: 1, [0]), 6)
    out = tmp_path / "reports" / "report.json"
    argv = [str(run_dir), str(tmp_path / "gt.txt"), "--start", "1", "--end", "5", "--output", str(out),
            "--export-motchallenge", str(tmp_path / "mot")]
    assert main(argv) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["configuration"]["window"]["frames"] == 4 and report["standard_metrics"]["hota"]["HOTA"] == 1.0
    assert len(report["inputs"]["frames"]["sha256"]) == 64 and (tmp_path / "mot" / "seqmap.txt").exists()
    assert main(argv) == 2  # existing report is never overwritten
    assert not list(run_dir.glob("*.json"))  # nothing is written into the evaluated run
