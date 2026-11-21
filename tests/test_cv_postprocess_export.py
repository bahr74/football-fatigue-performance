"""Refinement (stitching + kit labels), pixel-space movement and MP4 export."""

import json

import numpy as np
import pytest

from football_fatigue.vision import postprocess, run_index


def _rec(f, players):
    return {"frame_num": f, "timestamp_s": f / 25, "fps": 25.0, "players": players, "ball": None}


def _p(tid, x, vote="blue", kit="unknown", y=100.0, h=50.0):
    return {"track_id": tid, "confidence": 0.9, "detected_class": "person",
            "bbox_xyxy_px": [x, y, x + 20, y + h], "pixel_x": x + 10, "pixel_y": y + h,
            "kit_color": kit, "kit_vote": vote}


def _write(path, recs):
    path.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")


def test_stitching_links_a_fragment_and_labels_whole_track(tmp_path):
    recs = []
    for f in range(0, 30):  # track 1 then, after a 10-frame gap, track 2 at the nearby position
        recs.append(_rec(f, [_p(1, 100 + f)]))
    for f in range(40, 70):
        recs.append(_rec(f, [_p(2, 100 + f + 2)]))
    src, dst = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    _write(src, recs)
    stats = postprocess.refine(src, dst)
    assert stats["tracklets_in"] == 2 and stats["links"] == 1 and stats["tracks_out"] == 1
    out = [json.loads(line) for line in dst.read_text(encoding="utf-8").splitlines()]
    assert {p["track_id"] for r in out for p in r["players"]} == {1}
    assert all(p["kit_color"] == "blue" for r in out for p in r["players"])  # no warm-up gap
    assert out[0]["players"][0]["raw_track_id"] == 1 and out[-1]["players"][0]["raw_track_id"] == 2


def test_stitching_refuses_kit_conflict_and_implausible_jump(tmp_path):
    a = [_rec(f, [_p(1, 100, vote="blue")]) for f in range(30)]
    far = [_rec(f, [_p(2, 100, vote="white")]) for f in range(40, 70)]  # same place, other kit
    jump = [_rec(f, [_p(3, 3000, vote="blue")]) for f in range(40, 70)]  # blue but far away
    src, dst = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    merged = {r["frame_num"]: r for r in a}
    for r in far + jump:
        merged.setdefault(r["frame_num"], _rec(r["frame_num"], []))["players"].extend(r["players"])
    _write(src, [merged[k] for k in sorted(merged)])
    stats = postprocess.refine(src, dst)
    assert stats["links"] == 0 and stats["tracks_out"] == 3


def test_short_flicker_tracks_are_dropped_and_unknown_kept(tmp_path):
    recs = [_rec(f, [_p(1, 100, vote=None)] + ([_p(9, 500)] if f < 3 else [])) for f in range(40)]
    src, dst = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    _write(src, recs)
    stats = postprocess.refine(src, dst)
    assert stats["dropped_short_tracks"] == 1 and stats["tracks_out"] == 1
    out = [json.loads(line) for line in dst.read_text(encoding="utf-8").splitlines()]
    assert out[10]["players"][0]["kit_color"] == "unknown"  # no votes: never forced into a team


def test_pixel_space_movement_metrics():
    path = [(f, 100.0 + 4 * f, 200.0) for f in range(26)]  # 4 px/frame = 100 px/s in image space
    m = run_index._movement_px(path, 25.0)
    # the 5-frame moving average shortens a short track's ends, so allow ~10%
    assert m["net_px"] == pytest.approx(100.0, rel=0.1)
    assert m["mean_speed_px_s"] == pytest.approx(100.0, rel=0.1)
    assert run_index._movement_px([(0, 1.0, 1.0)], 25.0)["path_px"] is None


def test_export_and_verify_roundtrip(tmp_path):
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("av")
    from football_fatigue.vision import export_video, verify_export

    video = tmp_path / "src.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25, (640, 360))
    rng = np.random.default_rng(0)
    for i in range(30):
        frame = np.full((360, 640, 3), (40, 140, 40), np.uint8)
        cv2.circle(frame, (50 + 15 * i, 275), 12, (255, 255, 255), -1)  # moving content identifies the index
        frame += rng.integers(0, 3, frame.shape, dtype=np.uint8)
        w.write(frame)
    w.release()
    recs = [_rec(f, [dict(_p(1, 40 + 15 * f, kit="blue", y=250.0, h=50.0))]) for f in range(30)]
    frames = tmp_path / "frames.jsonl"
    _write(frames, recs)
    out = tmp_path / "annotated.mp4"
    info = export_video.export(video, frames, out)
    assert info["frames_written"] == 30 and info["frames_without_model_record"] == 0
    with pytest.raises(FileExistsError):
        export_video.export(video, frames, out)  # never overwrites
    report = verify_export.verify(video, out, frames, samples=6)
    assert report["codec"] == "h264" and report["frames"] == 30
    assert report["same_index_is_closest"] is True
    assert report["box_edges_with_expected_kit_colour"].split("/")[0] != "0"
