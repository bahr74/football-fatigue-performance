"""Tests for bounded CV run access (pixel-space summaries, windows, CSV)."""

import json

from football_fatigue.vision import run_index


def _write_run(path, n=300):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for i in range(n):
            players = [{"track_id": 1, "confidence": 0.9, "detected_class": "person",
                        "bbox_xyxy_px": [i, 10, i + 20, 60], "pixel_x": float(i), "pixel_y": 60.0}]
            if i % 10 != 0:  # track 2 has gaps
                players.append({"track_id": 2, "confidence": 0.5, "detected_class": "person",
                                "bbox_xyxy_px": [5, 5, 25, 55], "pixel_x": 15.0, "pixel_y": 55.0})
            f.write(json.dumps({"frame_num": i, "timestamp_s": i / 25, "fps": 25.0, "players": players, "ball": None}) + "\n")
        f.write('{"frame_num": 999, "pla')  # partial line from an active run


def test_index_summary_and_gaps(tmp_path):
    p = tmp_path / "frames.jsonl"
    _write_run(p)
    idx = run_index.get_index(p, 4096, 1080)
    s = run_index.summary(idx)
    assert s["frames_indexed"] == 300  # partial trailing line ignored
    by_id = {t["track_id"]: t for t in s["tracks"]}
    assert by_id[1]["gaps"] == 0 and by_id[1]["continuity"] == 1.0
    assert by_id[2]["gaps"] > 0 and by_id[2]["continuity"] < 1.0
    assert sum(map(sum, s["heatmap"]["cells"])) == 300 + 270
    assert len(s["timeline"]) <= run_index.TIMELINE_POINTS


def test_frame_window_is_bounded(tmp_path):
    p = tmp_path / "frames.jsonl"
    _write_run(p, 1200)
    idx = run_index.get_index(p)
    w = run_index.frame_window(p, idx, 100, 150)
    assert [r["frame_num"] for r in w] == list(range(100, 150))
    assert len(run_index.frame_window(p, idx, 0, 10**9, limit=10**6)) == run_index.MAX_WINDOW_FRAMES


def test_tracks_csv(tmp_path):
    p = tmp_path / "frames.jsonl"
    _write_run(p, 5)
    idx = run_index.get_index(p)
    text = "".join(run_index.tracks_csv(p, idx)).splitlines()
    assert text[0].startswith("frame_num,timestamp_s,track_id")
    assert len(text) == 1 + 5 + 4


def test_cache_refreshes_when_file_grows(tmp_path):
    p = tmp_path / "frames.jsonl"
    _write_run(p, 10)
    assert len(run_index.get_index(p).offsets) == 10
    _write_run(p, 20)
    assert len(run_index.get_index(p).offsets) == 20


def test_incremental_scan_completes_partial_line(tmp_path):
    p = tmp_path / "frames.jsonl"
    rec = {"frame_num": 0, "fps": 25.0, "players": [{"track_id": 7, "confidence": 0.8, "pixel_x": 1.0, "pixel_y": 2.0}]}
    line = json.dumps(rec) + "\n"
    p.write_text(line + line[:20], encoding="utf-8")
    idx = run_index.get_index(p)
    assert len(idx.offsets) == 1
    rec2 = dict(rec, frame_num=1)
    p.write_text(line + json.dumps(rec2) + "\n", encoding="utf-8")
    idx = run_index.get_index(p)
    assert [f for f, _ in idx.offsets] == [0, 1] and idx.tracks[7].count == 2
