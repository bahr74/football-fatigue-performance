"""Tests for the display-only Football Vision Demo (overlay builder + local server). SYNTHETIC data only."""

import gzip
import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from football_fatigue.demo import overlay as O
from football_fatigue.demo import server as S

H = 40.0


def player(local, x, y, *, trk=1, gid=1, status="confirmed", assoc="none", team="team_0", **kw):
    p = {"track_id": gid, "local_track_id": local, "tracklet_id": trk, "global_id": gid, "identity_status": status,
         "identity_confidence": 0.9, "identity_confidence_local": 0.95, "association_status": assoc, "team_cluster": team,
         "team_confidence": 0.9, "position_status": "observed", "detected_class": "player", "bbox_xyxy_px": [x - 8, y - H, x + 8, y]}
    p.update(kw)
    return p


def rec(f, players):
    return {"frame_num": f, "timestamp_s": f / 25.0, "fps": 25.0, "players": players}


def rows(frames, **kw):
    """rows_by_frame structure used by display_interpolation for one track."""
    out = {}
    for f in frames:
        out[f] = [dict(zip(("local", "gid", "trk", "bbox", "st", "conf", "conf_local", "link", "as", "team", "tconf", "cls", "est"),
                           (7, kw.get("gid", 1), kw.get("trk", 1), [100.0 + 2 * f, 260, 116 + 2 * f, 300], kw.get("st", 0), 0.9, 0.9, None,
                            kw.get("as", 0), 0, 0.9, "player", 0)))]
    return out


def build(recs, **kw):
    return O.build_overlay(recs, match_id="T", fps=25.0, video={"url": "/media/T.mp4", "start_time_s": 0.979}, **kw)


# ---------------------------------------------------------------- display interpolation safeguards
def test_gap_inside_one_tracklet_is_filled_and_flagged():
    r = rows([0, 1, 5, 6])
    est = O.display_interpolation(r, 5)
    assert sorted(est) == [2, 3, 4]
    b = est[3][0]
    assert b["est"] == 1 and b["conf"] is None and b["conf_local"] is None and b["link"] is None
    assert b["bbox"][0] == pytest.approx((1 - 0.5) * 102 + 0.5 * 110)                   # linear between the two observed ends


@pytest.mark.parametrize("changed", [{"trk": 2}, {"gid": 2}, {"st": 1}, {"as": 1}])
def test_interpolation_never_crosses_an_identity_status_or_provenance_boundary(changed):
    r = rows([0, 1])
    other = rows([5, 6], **changed)
    merged = {f: r.get(f, []) + other.get(f, []) for f in (0, 1, 5, 6)}
    if "trk" in changed:                      # different tracklet of the same local id
        pass
    assert O.display_interpolation(merged, 5) == {}


def test_long_gaps_and_disabled_mode_are_not_filled():
    assert O.display_interpolation(rows([0, 1, 9, 10]), 5) == {}                           # gap of 7 > 5
    assert O.display_interpolation(rows([0, 1, 5]), 0) == {}
    assert O.display_interpolation(rows([0, 1, 2, 3]), 5) == {}                            # no gap


def test_gap_with_no_second_end_is_not_extrapolated():
    assert O.display_interpolation(rows([0, 1, 2]), 5) == {}


# ---------------------------------------------------------------- overlay columns and honesty
def two_tracks_with_gap():
    recs = []
    for f in range(40):
        ps = [player(7, 100 + 2 * f, 300)] if f not in (10, 11) else []
        ps.append(player(9, 600, 400, trk=2, gid=2, status="unknown", assoc="linked_unvalidated"))
        recs.append(rec(f, ps))
    return recs


def test_columns_offsets_and_counts_are_consistent_and_estimates_are_separate():
    ov = build(two_tracks_with_gap())
    cols, off = ov["cols"], ov["frame_offsets"]
    n = len(cols["local"])
    assert all(len(v) == n for v in cols.values()) and off[0] == 0 and off[-1] == n and off == sorted(off)
    assert ov["n_frames"] == 40 and len(off) == 41
    assert sum(ov["series"]["det"]) == 40 + 38 == ov["summary"]["boxes_observed"]          # estimates never counted
    assert ov["summary"]["boxes_estimated_display_only"] == 2 == sum(cols["est"])
    assert ov["series"]["det"][10] == 1                                                    # frame 10: only the other track is observed
    est_idx = [i for i, e in enumerate(cols["est"]) if e]
    assert all(cols["conf"][i] is None and cols["seg"][i] == -1 for i in est_idx)          # not in any M7-0 safe segment
    f10 = range(off[10], off[11])
    assert sorted(cols["est"][i] for i in f10) == [0, 1]


def test_estimates_do_not_change_observed_arrays_or_histograms():
    a, b = build(two_tracks_with_gap(), interp_max_gap=0), build(two_tracks_with_gap(), interp_max_gap=5)
    obs = lambda ov: [(f, c) for f, c in zip(range(10 ** 9), ov["series"]["det"])]
    assert obs(a) == obs(b) and a["conf_hist"] == b["conf_hist"] and a["summary"]["status_boxes"] == b["summary"]["status_boxes"]
    keep = lambda ov: [(l, ov["cols"]["x1"][i]) for i, l in enumerate(ov["cols"]["local"]) if not ov["cols"]["est"][i]]
    assert keep(a) == keep(b) and a["summary"]["boxes_estimated_display_only"] == 0


def test_identity_fields_stay_separate_and_unknown_is_never_confirmed():
    ov = build(two_tracks_with_gap())
    c = ov["cols"]
    i9 = [i for i, l in enumerate(c["local"]) if l == 9][0]
    assert ov["statuses"][c["st"][i9]] == "unknown" and ov["assoc"][c["as"][i9]] == "linked_unvalidated"
    assert c["conf"][i9] == 0.9 and c["conf_local"][i9] == 0.95 and c["gid"][i9] == 2 and c["trk"][i9] == 2 and c["local"][i9] == 9
    assert ov["summary"]["status_boxes"]["confirmed"] == 38 and ov["summary"]["status_boxes"]["unknown"] == 40


def test_coordinate_and_board_labels_are_non_metric_and_no_workload_fields():
    ov = build(two_tracks_with_gap())
    assert ov["coordinate_system"]["unit"] == "pixel" and ov["coordinate_system"]["metric_valid"] is False
    assert ov["board"]["mode"] == "image_space_non_metric" and "Not registered" in ov["board"]["note"]
    keys = set(ov) | set(ov["cols"]) | set(ov["summary"]) | set(ov["series"])
    for banned in ("speed_m_s", "speed", "distance_m", "distance", "sprints", "fatigue_score", "workload"):
        assert banned not in keys
    assert any("NON-METRIC" in s for s in ov["limitations"])


def test_missing_frames_and_non_observed_positions_are_not_overlaid():
    recs = [rec(0, [player(1, 100, 300)]), rec(1, []), rec(2, [player(1, 104, 300, position_status="interpolated")]), rec(3, [player(1, 108, 300)])]
    recs[0]["players"].append({"detected_class": "ball", "bbox_xyxy_px": [1, 1, 2, 2]})
    ov = build(recs, interp_max_gap=0)
    assert ov["series"]["det"] == [1, 0, 0, 1] and ov["frame_offsets"] == [0, 1, 1, 1, 2]


def test_tracks_report_interruptions_and_segments_keep_use_flags():
    ov = build(two_tracks_with_gap())
    t7 = ov["tracks"]["7"]
    assert t7["interruptions"] == 1 and t7["n"] == 38 and t7["tracklets"] == [1] and t7["gids"] == [1]
    uses = {s["local"]: s["use"] for s in ov["segments"]}
    assert uses[9] == "local_only"                                                         # unknown + unvalidated link: never attributable
    assert sum(s["n"] for s in ov["segments"]) == ov["summary"]["boxes_observed"]


# ---------------------------------------------------------------- server: whitelist, gzip overlay, range video
@pytest.fixture()
def live(tmp_path):
    ov = tmp_path / "overlay.json"
    ov.write_text(json.dumps(build(two_tracks_with_gap())))
    vid = tmp_path / "v.mp4"
    vid.write_bytes(bytes(range(256)) * 8)
    H_ = type("H", (S.DemoHandler,), {"match_id": "T", "video_path": vid, "overlay_path": ov, "selftest_dir": None, "_overlay_gz": None})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H_)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1], tmp_path
    srv.shutdown()


def get(port, path, headers=None, method="GET", body=None):
    c = HTTPConnection("127.0.0.1", port, timeout=10)
    c.request(method, path, body=body, headers=headers or {})
    r = c.getresponse()
    return r.status, dict(r.getheaders()), r.read()


def test_server_serves_page_health_overlay_and_range_video(live):
    port, _ = live
    st, _, body = get(port, "/")
    assert st == 200 and b"Football Vision Demo" in body and b"NON-METRIC" in body
    assert json.loads(get(port, "/api/health")[2]) == {"ok": True, "match": "T", "video": True, "overlay": True}
    st, h, body = get(port, "/api/overlay")
    assert st == 200 and h["Content-Encoding"] == "gzip" and json.loads(gzip.decompress(body))["schema"] == O.OVERLAY_SCHEMA
    st, h, body = get(port, "/media/T.mp4", {"Range": "bytes=10-19"})
    assert st == 206 and body == bytes(range(10, 20)) and h["Content-Range"].startswith("bytes 10-19/")


def test_server_exposes_only_whitelisted_routes(live):
    port, tmp = live
    for path in ("/api/cv/frames", "/tactical", "/media/../overlay.json", "/media/other.mp4", "/%2e%2e/overlay.json", "/api/matches"):
        assert get(port, path)[0] == 404
    assert get(port, "/api/selftest", method="POST", body=b"{}")[0] == 404           # disabled unless --selftest-dir is given
    assert not list(tmp.glob("selftest_*"))


def test_page_contains_required_controls_and_labels():
    html = Path(S.PAGE).read_text(encoding="utf-8")
    for needle in ('id="vid"', 'id="board"', 'id="play"', 'id="speed"', 'id="idLocal"', 'id="idGlobal"', 'id="oEst"', 'id="oConf"',
                   "requestVideoFrameCallback", "APPROXIMATE", "NON-METRIC", "Origin: top-left", "units: pixels", "projection: none"):
        assert needle in html, needle
    import re
    tag = re.search(r'<input[^>]*id="oEst"[^>]*>', html).group(0)
    # M7.1: display-only gap fill is on by default (flagged, fading, never counted); strict observed-only stays one click away
    assert "checked" in tag and "display only" in html.lower()
    # M7.2: physical values come only from the gated analytics service; the page itself holds no number and states the gate
    for needle in ('id="physPanel"', 'id="pRun"', "VALUES WITHHELD", "DIAGNOSTIC VALUES — NOT VALID", "not a validated standard",
                   'id="revPanel"', "not validation evidence", "/api/analytics"):
        assert needle in html, needle


# ---------------------------------------------------------------- M10.1 opt-in presentation mode (page contract; browser behaviour is checked by its self test)
def page():
    return Path(S.PAGE).read_text(encoding="utf-8")


def test_presentation_mode_is_opt_in_and_off_by_default():
    html = page()
    assert 'id="tabPres"' in html and 'QS.get("mode")==="presentation"' in html and "PRES={on:false," in html
    assert '<body>' in html and 'class="presentation"' not in html                         # body starts in the default view
    assert '<button id="tabCoach" class="on">' in html                                    # default tab unchanged
    # every presentation branch is gated by the flag; the default label rules are still the first non-presentation branch
    assert 'if(PRES.on)return "T"+C.local[i];\n  if(S.idMode==="display")return String(C.local[i]);' in html
    assert "if(PRES.on)f=clipF(f);" in html and "PRES.t0=a;PRES.t1=b" in html


def test_presentation_clip_defaults_to_117093_2m20_to_3m00():
    html = page()
    assert 't0:+(QS.get("t0")??140),t1:+(QS.get("t1")??180)' in html
    assert "Math.round(a*fps)" in html and "Math.round(b*fps)-1" in html                  # 140 s -> frame 3500, 180 s -> last frame 4499
    assert "win:{f0:3500,f1:4499" in html                                                 # windowed audit numbers only for that exact clip


def test_presentation_mode_hides_withheld_and_identity_tools():
    import re
    html = page()
    assert "body.presentation .no-pres{display:none!important}" in html
    for sec in ("physPanel", "revPanel", "advPanel"):
        assert re.search(rf'<section class="[^"]*no-pres[^"]*" id="{sec}">', html), sec
    assert re.search(r'<div class="seg no-pres"><button id="idDisplay"', html)            # Number / Tracker ID / Identity / Reviewed
    assert re.search(r'<label class="chk no-pres"[^>]*><input type="checkbox" id="oConf">', html)
    assert '<button id="tabAdv" class="no-pres">' in html
    assert "if(!PRES.on&&S.revSel!==S.sel)" in html                                       # no analytics request from presentation mode
    strip = html[html.index('id="presStrip"'):html.index('<div id="top">')]
    for banned in ("km/h", "m/s", "possession", "fatigue", "sprint", "workload"):
        assert banned not in strip.lower(), banned


def test_presentation_mode_wording_and_disclosure():
    html = page()
    for needle in ("Track numbers are local to the tracker and can restart; they are not player identities",
                   "Clip selected for visual clarity. Whole-clip local tracking: HOTA ${PRES.hotaFull}; team-colour errors are larger in other parts of the match.",
                   "for speed or distance", "hotaFull:0.37", "team colours are wrong in this frame", "c[t]>11"):
        assert needle in html, needle


def test_presentation_trails_counts_and_analytics_follow_the_clip():
    html = page()
    # trails: same tracklet / consecutive-frame rule as trail(), and never before the clip start
    assert "C.trk[i]!==C.trk[prev]||frameOfBox[prev]-frameOfBox[i]!==1" in html.split("function trailPts")[1].split("function fadeTrail")[0]
    assert "if(PRES.on&&frameOfBox[i]<PRES.f0)break;" in html
    # counters: observed rows only, off-pitch and non-player roles excluded, team by the displayed colour rule
    assert "if(src(i)!==0||(C.pv&&C.pv[i]===2)||(C.role&&C.role[i]>1))continue;const t=teamOf(i)" in html
    # analytics window, shape chart, direction and segment insight are bounded by the clip in presentation mode
    assert 'if(PRES.on)return w==="all"?[PRES.f0,PRES.f1]' in html
    assert "base=P?PRES.f0:0,last=P?PRES.f1:S.nFrames-1" in html and "inferDirection(PRES.f0,PRES.f1)" in html
    assert "g0=PRES.on?Math.max(g.f0,PRES.f0):g.f0" in html
    assert "presSelftest" in html and 'R.mode=PRES.on?"presentation":"default"' in html


def test_presentation_url_is_served_by_the_whitelisted_page_route(live):
    port, _ = live
    st, _, body = get(port, "/?mode=presentation&t0=140&t1=180")
    assert st == 200 and b'id="tabPres"' in body and body == get(port, "/")[2]           # same page; the mode is client-side only


def test_missing_overlay_is_a_404_not_a_crash(tmp_path):
    H_ = type("H", (S.DemoHandler,), {"match_id": "T", "video_path": tmp_path / "none.mp4", "overlay_path": tmp_path / "none.json",
                                      "selftest_dir": None, "_overlay_gz": None})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H_)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        port = srv.server_address[1]
        assert get(port, "/api/overlay")[0] == 404 and get(port, "/media/T.mp4")[0] == 404
        assert json.loads(get(port, "/api/health")[2])["video"] is False
    finally:
        srv.shutdown()


# ---------------------------------------------------------------- time base (sync-fix regression: a wrong stored start shifted every overlay by 24 frames)
def tiny_video(path, n=5):
    import cv2
    import numpy as np

    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (64, 48))
    for i in range(n):
        w.write(np.full((48, 64, 3), 10 * i, np.uint8))
    w.release()


def overlay_file(tmp_path, start):
    ov = build(two_tracks_with_gap())
    ov["video"]["start_time_s"] = start
    p = tmp_path / "ov.json"
    p.write_text(json.dumps(ov))
    return p


def test_wrong_stored_start_is_corrected_from_the_container(tmp_path):
    vid = tmp_path / "v.mp4"
    tiny_video(vid)
    actual = S.first_frame_pts_s(vid)
    assert actual is not None and actual < 0.5
    bad = json.loads(S.verified_overlay_bytes(overlay_file(tmp_path, 0.979), vid))
    assert bad["video"]["start_time_s"] == actual and "corrected" in bad["video"]["start_time_source"]
    good = json.loads(S.verified_overlay_bytes(overlay_file(tmp_path, actual), vid))
    assert good["video"]["start_time_source"] == "container_verified"


def test_unreadable_or_missing_video_never_silently_becomes_zero(tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video" * 50)
    assert S.first_frame_pts_s(junk) is None and S.first_frame_pts_s(tmp_path / "missing.mp4") is None
    for v in (junk, tmp_path / "missing.mp4"):
        out = json.loads(S.verified_overlay_bytes(overlay_file(tmp_path, 0.979), v))
        assert out["video"]["start_time_s"] == 0.979 and out["video"]["start_time_source"] == "overlay_unverified"


def test_shipped_overlay_start_matches_the_real_video_pts():
    ov_path, vid = S.MATCHES["117093"]["overlay"], S.MATCHES["117093"]["video"]
    if not (ov_path.is_file() and vid.is_file()):
        pytest.skip("demo assets not present")
    stored = json.loads(ov_path.read_text())["video"]["start_time_s"]
    assert stored == pytest.approx(S.first_frame_pts_s(vid), abs=1e-3) and stored > 0.9       # v1 stored 0.0: 24 frames of overlay error


def test_page_measures_the_first_frame_pts_and_maps_time_with_media_time():
    html = Path(S.PAGE).read_text(encoding="utf-8")
    for needle in ("calibrateStart", "meta.mediaTime", "Math.floor((t-S.ov.video.start_time_s)*S.ov.fps", "start offset corrected", "stale frame presented while a seek"):
        assert needle in html, needle
