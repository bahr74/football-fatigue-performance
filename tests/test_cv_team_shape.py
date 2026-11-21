"""Team-shape analytics: robust metrics, events, validation against reference tracks, acceptance criteria, coach view."""

import http.server
import json
import shutil
import subprocess
import threading
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from football_fatigue.playback.server import BoardHandler
from football_fatigue.vision import team_shape as ts

FPS = 25
SECONDS = 70
SPREAD_WINDOW = (40, 50)  # seconds in which both teams double their spread


def _frame(f, drop_blue=0):
    t = f / FPS
    k = 2.5 if SPREAD_WINDOW[0] <= t < SPREAD_WINDOW[1] else 1.0
    players = []
    for team, base, kit in (("blue", 1000, "blue"), ("white", 2200, "white")):
        n = 10 - (drop_blue if team == "blue" else 0)
        for i in range(n):
            x = base + (i - 4.5) * 40 * k
            y = 380 + ((i * 37) % 9 - 4) * 12 * k
            players.append({"track_id": (1 if team == "blue" else 100) + i, "confidence": 0.8, "detected_class": "person",
                            "kit_color": kit, "bbox_xyxy_px": [x - 10, y - 60, x + 10, y], "pixel_x": x, "pixel_y": y})
    return {"frame_num": f, "timestamp_s": t, "fps": float(FPS), "players": players, "ball": None}


def _write(tmp_path, frames, name="frames.jsonl"):
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in frames), encoding="utf-8")
    return path


def test_group_metrics_need_enough_players_and_use_box_height_units():
    x = np.arange(10) * 20.0
    y = np.full(10, 300.0)
    h = np.full(10, 50.0)
    assert ts._group_metrics(x[:5], y[:5], h[:5]) is None  # below MIN_PLAYERS: skipped, not zero
    m = ts._group_metrics(x, y, h)
    assert m["length_px"] == pytest.approx(np.percentile(x, 90) - np.percentile(x, 10))
    assert m["length_bh"] == pytest.approx(m["length_px"] / 50.0)
    # the same shape seen twice as large (twice as close) gives the same player-height spread
    big = ts._group_metrics(x * 2, y * 2, h * 2)
    assert big["spread_bh"] == pytest.approx(m["spread_bh"])


def test_spread_ignores_one_stray_detection_when_eight_or_more():
    x = np.array([0, 10, 20, 30, 40, 50, 60, 70.0])
    y = np.zeros(8)
    base = ts.robust_spread(x, y)
    with_stray = ts.robust_spread(np.append(x, 5000.0), np.append(y, 0.0))
    assert with_stray == pytest.approx(ts.robust_spread(x, y), rel=0.15) and with_stray < 2 * base
    assert ts.robust_spread(np.array([0, 100.0]), np.zeros(2)) > 0  # small groups are not trimmed


def test_series_uses_one_second_medians_and_null_when_half_the_frames_are_invalid(tmp_path):
    frames = [_frame(f, drop_blue=6 if 25 <= f < 50 else 0) for f in range(75)]  # 4 blue players in second 1
    data = ts.read_frames(_write(tmp_path, frames))
    s = ts.bin_series(data, ts.per_frame(data), FPS)
    assert s["blue"]["spread_bh"][0] is not None and s["blue"]["spread_bh"][1] is None  # <6 players: null
    assert s["white"]["spread_bh"][1] is not None


def test_spread_event_is_found_at_the_right_time_and_constant_shape_gives_none(tmp_path):
    data = ts.read_frames(_write(tmp_path, [_frame(f) for f in range(SECONDS * FPS)]))
    series = ts.bin_series(data, ts.per_frame(data), FPS, keys=(*ts.METRICS, "spread_px"))
    events = ts.find_events(series, (0.0, 4096.0))
    spread = [e for e in events if e["type"] in ("spread", "stretch") and e["team"] == "blue"]
    assert spread and all(e["mode"] == "image-space" for e in events)
    assert any(SPREAD_WINDOW[0] - 2 <= e["start_s"] and e["end_s"] <= SPREAD_WINDOW[1] + 3 for e in spread)
    flat = ts.read_frames(_write(tmp_path, [dict(_frame(f), frame_num=f) for f in range(SECONDS * FPS)], "flat.jsonl"))
    for fr in flat:  # constant shape: the same frame repeated
        fr["groups"] = data[0]["groups"]
    s2 = ts.bin_series(flat, ts.per_frame(flat), FPS, keys=(*ts.METRICS, "spread_px"))
    assert [e for e in ts.find_events(s2, (0.0, 4096.0)) if e["type"] in ("spread", "compact", "stretch", "squeeze", "gap")] == []


def test_frame_flags_mark_missing_players_label_doubt_and_low_confidence(tmp_path):
    frames = [_frame(f, drop_blue=7 if f < 60 else 0) for f in range(120)]
    for r in frames[80:]:
        r["players"].extend([{"track_id": 900 + i, "confidence": 0.3, "kit_color": None, "bbox_xyxy_px": [0, 0, 5, 20],
                              "pixel_x": 1.0, "pixel_y": 20.0} for i in range(40)])
    q = ts.frame_flags(ts.read_frames(_write(tmp_path, frames)), FPS)
    assert q["missing_players"]["segments"][0] == {"start_s": 0.0, "end_s": 2.4}
    assert q["label_doubt"]["frames"] == 40 and q["low_confidence"]["frames"] == 40


def test_validation_against_reference_tracks(tmp_path):
    frames = [_frame(f) for f in range(SECONDS * FPS)]
    run = _write(tmp_path, frames)
    lines, kit = [], {"blue": [], "white": [], "other": []}
    for r in frames:
        for p in r["players"]:
            b = p["bbox_xyxy_px"]
            lines.append(f"{r['frame_num']},{p['track_id']},{b[0]},{b[1]},{b[2] - b[0]},{b[3] - b[1]}")
    for p in frames[0]["players"]:
        kit[p["kit_color"]].append(p["track_id"])
    mot = tmp_path / "mot.txt"
    mot.write_text("\n".join(lines) + "\n", encoding="utf-8")
    kit_file = tmp_path / "kit.json"
    kit_file.write_text(json.dumps(kit), encoding="utf-8")
    v = ts.validate(run, mot, kit_file)
    for team in ts.TEAMS:
        assert v["reference"]["teams"][team]["spread_bh"]["pearson_r"] == pytest.approx(1.0, abs=1e-3)
        assert v["reference"]["teams"][team]["cx"]["median_relative_difference"] == pytest.approx(0.0, abs=1e-6)
    assert v["event_agreement_with_reference"]["agree"] == v["event_agreement_with_reference"]["events_checked"] > 0


def test_advanced_metrics_stay_unavailable_unless_every_criterion_is_met():
    weak_cal = {"held_out_cross_validation": {"median_m": 2.04, "p90_m": 4.02}}
    ev = {"idf1": 0.27, "id_switches": 657, "reference_identities": 22, "frames": 5976}
    out = {m["key"]: m for m in ts.advanced_metrics(weak_cal, ev, {"share": 0.22, "frames_sampled": 40})}
    assert set(out) == {"formation", "defensive_line", "possession", "pressing", "distance", "sprints", "pitch_control", "epv_obso_das"}
    assert all(m["status"] == "unavailable" for m in out.values())
    assert "IDF1 0.27" in out["distance"]["requirements"][1]["measured"]
    perfect = ts.advanced_metrics({"held_out_cross_validation": {"median_m": 0.3, "p90_m": 0.8}},
                                  {"idf1": 0.95, "id_switches": 3, "reference_identities": 22, "frames": 5976}, None)
    assert next(m for m in perfect if m["key"] == "distance")["status"] == "available"
    assert next(m for m in perfect if m["key"] == "possession")["status"] == "unavailable"  # no ball evidence at all


def test_ball_probe_ignores_static_spare_balls():
    spare = [0.9, [100, 100, 112, 112]]
    probe = [{"top": [spare]} for _ in range(10)] + [{"top": [spare, [0.6, [900, 300, 912, 312]]]} for _ in range(2)]
    s = ts.ball_probe_summary(probe)
    assert s["static_candidate_positions"] == 1 and s["frames_with_non_static_candidate"] == 2


@pytest.fixture()
def coach_server(tmp_path):
    run = tmp_path / "vision" / "soccertrack-117093-test"
    run.mkdir(parents=True)
    _write(run, [_frame(f) for f in range(SECONDS * FPS)])
    (run / "manifest.json").write_text(json.dumps({"configuration": {"video_width_px": 4096, "video_height_px": 1080}}), encoding="utf-8")
    saved = (BoardHandler.vision_demo_dir, BoardHandler.vision_video_path, BoardHandler.vision_export_dir)
    BoardHandler.vision_demo_dir, BoardHandler.vision_video_path, BoardHandler.vision_export_dir = run, tmp_path / "v.mp4", tmp_path
    BoardHandler._cv_coach_cache = None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), BoardHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1], run
    srv.shutdown()
    BoardHandler.vision_demo_dir, BoardHandler.vision_video_path, BoardHandler.vision_export_dir = saved
    BoardHandler._cv_coach_cache = None


def _get(port, path):
    return json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}{path}"))


def test_coach_endpoint_computes_on_demand_marks_unvalidated_then_prefers_saved_file(coach_server):
    port, run = coach_server
    d = _get(port, "/api/cv/coach")
    assert d["available"] and d["validated"] is False and d["mode"] == "image-space" and d["events"]
    (run / "coach_insights.json").write_text(json.dumps({"mode": "image-space", "events": [], "series": {"t": []}, "validation": {}}), encoding="utf-8")
    d = _get(port, "/api/cv/coach")
    assert d["validated"] is True and d["events"] == []


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_coach_script_filters_low_reliability_moments_and_shows_unavailable_metrics(tmp_path):
    base = {"mode": "image-space", "fps": 25.0, "frames": 1500, "extent_x_px": [0, 4096], "validated": True, "available": True,
            "series": {"t": [0, 1, 2], **{t: {k: [1.0, None, 1.2] for k in (*ts.METRICS, "n")} for t in ts.TEAMS}},
            "teams": {t: {"spread_bh": {"median": 6.0, "p25": 5.0, "p75": 7.0}, "length_bh": None, "width_bh": None,
                          "block_position_share": {"left": 0.2, "centre": 0.3, "right": 0.5}, "valid_seconds": 2} for t in ts.TEAMS},
            "comparison": {"seconds_compared": 2, "blue_more_compact_share": 0.4, "median_spread_ratio_blue_over_white": 1.1},
            "events": [
                {"type": "compact", "metric": "spread_bh", "team": "blue", "start_s": 5, "end_s": 9, "description": "SHOWN moment",
                 "mode": "image-space", "default_visible": True, "reliability": "good", "reference_agrees": True, "near_image_edge": False, "players_per_group": 8},
                {"type": "spread", "metric": "spread_bh", "team": "white", "start_s": 20, "end_s": 24, "description": "HIDDEN moment",
                 "mode": "image-space", "default_visible": False, "reliability": "poor", "reference_agrees": False, "near_image_edge": True, "players_per_group": None}],
            "quality": {"missing_players": {"share": 0.2, "segments": [], "segments_total": 0, "frames": 5}},
            "validation": {"reference": {"teams": {t: {k: {"pearson_r": None} for k in ts.METRICS} for t in ts.TEAMS}},
                           "event_agreement_with_reference": {"agree": 1, "events_checked": 2, "shown_by_default": 1, "events_total": 2},
                           "pitch_prototype_crosscheck": None},
            "advanced_metrics": ts.advanced_metrics({"held_out_cross_validation": {"median_m": 2.0, "p90_m": 4.0}}, None, None)}
    analytics = {"available": True, "run": "t", "overview": {"frames_processed": 1, "boxes_by_kit": {}, "share_by_kit": {}, "mean_boxes_per_frame": {}},
                 "warnings": [], "series": {"bin_s": 2, "t": [], "blue": [], "white": [], "other": [], "unknown": []},
                 "heatmaps": {"by_kit": {}, "cols": 64, "rows": 17, "width_px": 4096, "height_px": 1080}, "regions": {"definition": None, "by_kit": {}, "by_minute": {"blue": [], "white": []}},
                 "diagnostics": {"duration_histogram": [], "confidence_histogram": None, "fragmentation": {}, "reference_evaluation": None}, "comparison": None,
                 "movement": {"criteria": "", "stable_tracks": 0, "of_tracks": 0, "by_kit": {}, "rows": []}, "downloads": []}
    (tmp_path / "p.json").write_text(json.dumps({"coach": base, "analytics": analytics}), encoding="utf-8")
    root = Path(__file__).parent
    out = json.loads(subprocess.run(
        ["node", str(root / "js" / "cv_team_harness.js"), str(root.parent / "src" / "football_fatigue" / "board" / "cv.html"),
         str(tmp_path / "p.json"), "coach"], capture_output=True, text=True, encoding="utf-8", check=True, timeout=60).stdout)
    assert "SHOWN moment" in out["event-list"] and "HIDDEN moment" not in out["event-list"]
    assert "include 1 low-reliability moment" in out["event-filters"]
    assert "NaN" not in json.dumps(out) and "undefined" not in json.dumps(out)
    assert "unavailable" in out["adv-rows"] and "Possession" in out["adv-rows"]
    assert "—" in out["vs-cards"]  # missing length/width medians are dashes, not zero
