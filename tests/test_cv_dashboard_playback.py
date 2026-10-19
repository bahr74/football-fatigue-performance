"""Regression tests: model overlay data must be available across every frame window."""

import http.server
import json
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

from football_fatigue.playback.server import BoardHandler

HARNESS = Path(__file__).parent / "js" / "cv_playback_harness.js"
N_FRAMES = 400


def _write_run(run: Path) -> None:
    run.mkdir(parents=True)
    with (run / "frames.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for i in range(N_FRAMES):
            players = [{"track_id": 1, "confidence": 0.9, "detected_class": "person",
                        "bbox_xyxy_px": [i, 10, i + 20, 60], "pixel_x": float(i), "pixel_y": 60.0}]
            if i % 10 != 0:
                players.append({"track_id": 2, "confidence": 0.5, "detected_class": "person",
                                "bbox_xyxy_px": [5, 5, 25, 55], "pixel_x": 15.0, "pixel_y": 55.0})
            f.write(json.dumps({"frame_num": i, "timestamp_s": i / 25, "fps": 25.0, "players": players, "ball": None}) + "\n")
    (run / "manifest.json").write_text(json.dumps({
        "status": "processed_unvalidated",
        "configuration": {"video_fps": 25.0, "video_width_px": 4096, "video_height_px": 1080},
        "summary": {"frames_written": N_FRAMES, "total_frames_expected": N_FRAMES},
    }), encoding="utf-8")


@pytest.fixture()
def server(tmp_path):
    run = tmp_path / "vision" / "soccertrack-117093-test"
    _write_run(run)
    (tmp_path / "v.mp4").write_bytes(b"x")
    saved = (BoardHandler.vision_demo_dir, BoardHandler.vision_video_path)
    BoardHandler.vision_demo_dir, BoardHandler.vision_video_path = run, tmp_path / "v.mp4"
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), BoardHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()
    BoardHandler.vision_demo_dir, BoardHandler.vision_video_path = saved


def _get(port, path):
    import urllib.request
    return json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}{path}"))


@pytest.mark.parametrize("start,end", [(0, 100), (90, 110), (100, 200), (250, 350), (300, 500)])
def test_frame_windows_return_exact_records(server, start, end):
    frames = _get(server, f"/api/cv/frames?start={start}&end={end}")["frames"]
    assert [r["frame_num"] for r in frames] == list(range(start, min(end, N_FRAMES)))
    for r in frames:
        assert len(r["players"]) == (1 if r["frame_num"] % 10 == 0 else 2)


def test_demo_payload_is_bounded_and_complete_index(server):
    demo = _get(server, "/api/cv/demo")
    assert len(demo["frames"]) <= 25  # compatibility payload only; not the full run
    assert demo["image_space"]["frames_indexed"] == N_FRAMES


def test_cv_page_is_not_cached(server):
    import urllib.request
    resp = urllib.request.urlopen(f"http://127.0.0.1:{server}/cv")
    assert resp.headers["Cache-Control"] == "no-store"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required for the dashboard script test")
def test_overlay_present_on_every_frame_across_window_boundaries_and_seeks(server):
    html = Path(__file__).parent.parent / "src" / "football_fatigue" / "board" / "cv.html"
    out = subprocess.run(["node", str(HARNESS), str(server), str(html), "340", "304,177,0,399"],
                         capture_output=True, text=True, timeout=120, check=True).stdout
    result = json.loads(out.strip().splitlines()[-1])
    assert result["error"] is None
    assert result["missing"] == [], f"frames without a model overlay: {result['missing'][:10]}"
    for frame, shown in result["seeks"].items():
        assert str(shown) == str(1 if int(frame) % 10 == 0 else 2), (frame, shown)


def test_production_server_is_threaded_so_video_streams_do_not_block_api():
    """Root cause of the stuck 'Loading…' overlay: a single-threaded server queued
    /api/cv/frames behind the browser's long-lived video range request."""
    import inspect

    from football_fatigue.playback import server as server_module

    source = inspect.getsource(server_module.run)
    assert "ThreadingHTTPServer(" in source and "= HTTPServer(" not in source
