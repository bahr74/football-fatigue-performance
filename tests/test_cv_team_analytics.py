"""Team-level image-space analytics, the analytics endpoint and the saved-video downloads."""

import http.server
import json
import threading
import urllib.error
import urllib.request

import pytest

from football_fatigue.playback.server import BoardHandler
from football_fatigue.vision import run_index


def _player(tid, x, y, kit, conf=0.8):
    return {"track_id": tid, "confidence": conf, "detected_class": "person", "kit_color": kit,
            "bbox_xyxy_px": [x - 10, y - 50, x + 10, y], "pixel_x": float(x), "pixel_y": float(y)}


def _frames(n=100):
    """Per frame: 2 blue (left), 1 white (right), 1 unknown on every 5th frame; frames 40-59 are missing."""
    out = []
    for f in range(n):
        if 40 <= f < 60:
            continue
        players = [_player(1, 600, 300, "blue"), _player(2, 700, 300, "blue"), _player(3, 3000, 300, "white", 0.55)]
        if f % 5 == 0:
            players.append(_player(4, 2000, 600, None))
        out.append({"frame_num": f, "timestamp_s": f / 25, "fps": 25.0, "players": players, "ball": None})
    return out


def _write(tmp_path, records):
    path = tmp_path / "frames.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def _analytics(tmp_path, records=None, **kw):
    path = _write(tmp_path, records if records is not None else _frames())
    return run_index.team_analytics(run_index.get_index(path), **kw)


def test_overview_counts_and_missing_label_is_unknown_not_dropped(tmp_path):
    a = _analytics(tmp_path, manifest={"summary": {"total_frames_expected": 100}})
    o = a["overview"]
    assert o["frames_processed"] == 80 and o["frames_expected"] == 100
    assert o["boxes_by_kit"] == {"blue": 160, "white": 80, "other": 0, "unknown": 16}
    assert o["boxes_total"] == 256
    assert o["share_by_kit"]["blue"] == pytest.approx(160 / 256, abs=1e-4)
    assert o["mean_boxes_per_frame"]["blue"] == 2.0
    assert o["team_label_coverage"] == pytest.approx(240 / 256, abs=1e-4)
    assert o["duration_s"] == pytest.approx(80 / 25)


def test_series_bins_hold_means_and_empty_bins_are_null_not_zero(tmp_path):
    s = _analytics(tmp_path)["series"]  # 2 s bins = 50 frames
    assert s["t"] == [0.0, 2.0]
    assert s["blue"] == [2.0, 2.0]
    # frames 40-49 are the only gap inside bin 0; bin 0 averages the 40 frames present
    assert s["unknown"][0] == pytest.approx(8 / 40, abs=1e-2)
    a = _analytics(tmp_path / "gap" if (tmp_path / "gap").mkdir() is None else tmp_path,
                   [r for r in _frames(150) if not 50 <= r["frame_num"] < 100])["series"]
    assert a["blue"][1] is None and a["white"][1] is None  # no frames at all in bin 1


def test_regions_use_thirds_and_halves_of_the_occupied_extent(tmp_path):
    r = _analytics(tmp_path)["regions"]
    blue, white = r["by_kit"]["blue"]["counts"], r["by_kit"]["white"]["counts"]
    assert sum(map(sum, blue)) == 160 and blue[0][0] == 160  # far-left only
    assert white[0][2] == 80  # far-right only
    assert r["by_kit"]["blue"]["share"][0][0] == 1.0
    assert "other" not in r["by_kit"]  # no detections: absent, not a row of zeros
    assert r["definition"]["columns"] == ["left", "centre", "right"]


def test_per_minute_region_share_is_null_when_too_few_boxes(tmp_path):
    recs = _frames(100)
    bm = _analytics(tmp_path, recs)["regions"]["by_minute"]
    assert bm["blue"][0]["boxes"] == 160 and bm["blue"][0]["share"] is not None
    sparse = [dict(r, players=r["players"][:1] if r["frame_num"] > 5 else r["players"]) for r in recs]
    (tmp_path / "s").mkdir()
    bm = _analytics(tmp_path / "s", [dict(r, players=[p for p in r["players"] if p["track_id"] != 3 or r["frame_num"] < 5]) for r in sparse])["regions"]["by_minute"]
    assert bm["white"][0]["boxes"] < run_index.MIN_REGION_BOXES and bm["white"][0]["share"] is None


def test_diagnostics_and_confidence_histogram(tmp_path):
    d = _analytics(tmp_path)["diagnostics"]
    assert d["track_ids"] == 4
    assert d["fragmentation"]["tracks_with_gaps"] == 4  # every track skips frames 40-59
    # tracks 1-3: one gap each; track 4 (every 5th frame): 15 gaps between its 16 appearances
    assert d["fragmentation"]["total_gaps"] == 3 + 15
    hist = d["confidence_histogram"]
    assert sum(b["boxes"] for b in hist) == 256
    assert next(b for b in hist if b["lo"] == 0.55)["boxes"] == 80
    assert d["reference_evaluation"] is None  # not supplied -> reported as absent


def test_movement_only_for_stable_tracks_and_none_is_not_zero(tmp_path):
    recs = []
    for f in range(150):
        recs.append({"frame_num": f, "timestamp_s": f / 25, "fps": 25.0, "ball": None, "players": [
            _player(1, 100 + 4 * f, 300, "blue"),                       # long, continuous, moving
            *([_player(2, 900, 300, "white")] if f < 30 else []),        # too short to be stable
        ]})
    m = _analytics(tmp_path, recs)["movement"]
    assert m["stable_tracks"] == 1 and m["of_tracks"] == 2
    assert m["by_kit"]["blue"]["tracks"] == 1 and m["by_kit"]["blue"]["median_px_s"] == pytest.approx(100, rel=0.05)
    assert m["by_kit"]["white"] == {"tracks": 0, "median_px_s": None, "median_path_px": None}


def test_warnings_flag_low_recall_unknowns_and_identity_caveat(tmp_path):
    texts = " ".join(w["text"] for w in _analytics(tmp_path, reference_eval={"idf1": 0.27})["warnings"])
    assert "not verified player identities" in texts and "IDF1 is 0.27" in texts
    assert "boxes per frame" in texts  # 3.2 per frame is far below 22


def test_empty_run_has_null_overview_values(tmp_path):
    (tmp_path / "frames.jsonl").write_text("", encoding="utf-8")
    a = run_index.team_analytics(run_index.get_index(tmp_path / "frames.jsonl"))
    assert a["overview"]["duration_s"] is None and a["overview"]["team_label_coverage"] is None
    assert a["overview"]["mean_boxes_per_frame"]["blue"] is None and a["regions"]["by_kit"] == {}


@pytest.fixture()
def server(tmp_path):
    run = tmp_path / "vision" / "soccertrack-117093-test"
    run.mkdir(parents=True)
    (run / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in _frames()), encoding="utf-8")
    (run / "manifest.json").write_text(json.dumps({"configuration": {"video_width_px": 4096, "video_height_px": 1080},
                                                   "summary": {"total_frames_expected": 100}}), encoding="utf-8")
    (run / "comparison.json").write_text(json.dumps({"rows": [{"label": "x", "idf1": 0.3}]}), encoding="utf-8")
    exports = tmp_path / "exports"
    exports.mkdir()
    (exports / "annotated_full_117093.mp4").write_bytes(bytes(range(200)))
    (tmp_path / "secret.mp4").write_bytes(b"secret")
    saved = (BoardHandler.vision_demo_dir, BoardHandler.vision_video_path, BoardHandler.vision_export_dir)
    BoardHandler.vision_demo_dir, BoardHandler.vision_video_path, BoardHandler.vision_export_dir = run, tmp_path / "v.mp4", exports
    BoardHandler._cv_analytics_cache = None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), BoardHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()
    BoardHandler.vision_demo_dir, BoardHandler.vision_video_path, BoardHandler.vision_export_dir = saved
    BoardHandler._cv_analytics_cache = None


def _open(port, path, headers=None):
    return urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}{path}", headers=headers or {}))


def test_analytics_endpoint_reports_run_comparison_downloads_and_unavailable_metrics(server):
    d = json.load(_open(server, "/api/cv/analytics"))
    assert d["available"] and d["overview"]["frames_processed"] == 80
    assert d["comparison"]["rows"][0]["idf1"] == 0.3
    by_url = {x["url"]: x for x in d["downloads"]}
    assert by_url["/media/soccertrack/annotated.mp4"]["available"] and by_url["/media/soccertrack/annotated.mp4"]["size_bytes"] == 200
    assert by_url["/media/soccertrack/preview.mp4"]["available"] is False and by_url["/media/soccertrack/preview.mp4"]["size_bytes"] is None
    assert d["advanced_metrics"]["status"] == "unavailable"


def test_download_endpoints_serve_only_whitelisted_files_with_range_and_attachment(server):
    r = _open(server, "/media/soccertrack/annotated.mp4", {"Range": "bytes=10-19"})
    assert r.status == 206 and r.read() == bytes(range(10, 20)) and "Content-Disposition" not in r.headers
    r = _open(server, "/media/soccertrack/annotated.mp4?download=1")
    assert r.status == 200 and len(r.read()) == 200
    assert 'attachment; filename="annotated_full_117093.mp4"' == r.headers["Content-Disposition"]
    for bad in ("/media/soccertrack/preview.mp4", "/media/soccertrack/secret.mp4", "/media/soccertrack/..%2fsecret.mp4"):
        with pytest.raises(urllib.error.HTTPError) as exc:
            _open(server, bad)
        assert exc.value.code == 404


def test_browser_script_renders_missing_values_as_dash_not_zero(tmp_path):
    import shutil
    import subprocess
    from pathlib import Path

    if shutil.which("node") is None:
        pytest.skip("node not available")
    recs = [r for r in _frames(100) if r["frame_num"] < 30]  # one minute with few boxes; no 'other' label
    payload = run_index.team_analytics(run_index.get_index(_write(tmp_path, recs)))
    payload.update(available=True, run="t", comparison=None, advanced_metrics={"status": "unavailable"},
                   downloads=[{"url": "/media/soccertrack/annotated.mp4", "label": "Annotated", "available": False, "size_bytes": None},
                              {"url": "/media/soccertrack/preview.mp4", "label": "Preview", "available": False, "size_bytes": None}])
    (tmp_path / "p.json").write_text(json.dumps(payload), encoding="utf-8")
    root = Path(__file__).parent
    out = json.loads(subprocess.run(
        ["node", str(root / "js" / "cv_team_harness.js"), str(root.parent / "src" / "football_fatigue" / "board" / "cv.html"),
         str(tmp_path / "p.json")], capture_output=True, text=True, encoding="utf-8", check=True, timeout=60).stdout)
    assert "NaN" not in json.dumps(out) and "undefined" not in json.dumps(out)
    other_row = next(r for r in out["region-rows"].split("</tr>") if "Other" in r)
    assert "%" not in other_row and "—" in other_row  # no 'other' detections: dashes, not 0%
    assert "Reference evaluation not available" in out["eval-rows"] and "0.000" not in out["eval-rows"]
    assert out["mv-kit-rows"].count("—") >= 6  # no stable tracks: medians shown as dashes
    assert "file not found" in out["downloads"] and "[disabled]" in out["burned-mode"]
