"""Pipeline failure decomposition, T0L profile and the player's debugger API (Session B).

Unit tests run anywhere; ``needs_data`` tests read FOOTBALL_FATIGUE_DATA_ROOT and the saved _diag outputs.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "processed" / "video"
DATA_ROOT = os.environ.get("FOOTBALL_FATIGUE_DATA_ROOT")
DIAG = OUT / "_diag" / "pipeline-v1"
needs_data = pytest.mark.skipif(not DATA_ROOT or not (DIAG / "soccertrack-117093" / "manifest.json").is_file(),
                                reason="needs FOOTBALL_FATIGUE_DATA_ROOT and the pipeline-v1 outputs")
DEV = ("soccertrack-117093", "soccertrack-118575", "soccertrack-117092", "soccertrack-118576")


def _det(conf, person=True, turf=True):
    return SimpleNamespace(conf=conf, person=person, turf=turf)


# ------------------------------------------------------------------------------------------- unit
def test_rejection_follows_pipeline_order_per_profile():
    from football_fatigue.video.failures.pipeline import _passes

    assert _passes(_det(0.9, person=False), "T0L") == "rej_class"
    assert _passes(_det(0.3), "T0") == "rej_conf" and _passes(_det(0.3), "T0L") is None
    assert _passes(_det(0.4), "T0") == "rej_conf"                       # strict: conf > 0.4
    assert _passes(_det(0.9, turf=False), "B0") == "rej_turf" and _passes(_det(0.9, turf=False), "T0") is None
    assert _passes(_det(0.3, turf=False), "B0") == "rej_conf"           # threshold is checked before the turf filter
    assert _passes(_det(0.9), "B0") is None


def test_episodes_ids_and_exits_from_synthetic_observations():
    from football_fatigue.video.failures.pipeline import _summarise

    attrs = {7: {f: (0.0, 40.0, "visible", "none" if f in (3, 4) else None, None, 0.0) for f in range(12)}}
    attrs[7][8] = (0.0, 40.0, "off_pitch", None, None, 0.0)
    rows = [(0, "ok", 1, 0.9), (1, "ok", 1, 0.9), (2, "ok", 1, 0.9), (3, "miss", None, None), (4, "miss", None, None),
            (5, "rej_conf", None, 0.3), (6, "id_changed", 2, 0.8), (7, "ok", 2, 0.8), (8, "not_evaluable", 2, 0.8),
            (9, "untracked", None, 0.7), (11, "ok", 2, 0.9)]                       # frame 10: GT absent
    idsw = [{"frame": 6, "gid": 7, "from": 1, "to": 2, "frames_since_last_tracked": 4, "previous_id_still_output": False}]
    attrs[7][11] = (0.0, 40.0, "visible", None, None, 0.0)
    eps, s = _summarise("soccertrack-117093", "T0", {7: rows}, attrs, idsw, {1: Counter({7: 3}), 2: Counter({7: 3})}, 12)
    e = eps["episodes"]
    assert [(x["start"], x["end"], x["cause"]) for x in e] == [(3, 5, "miss"), (9, 9, "untracked")]
    assert e[0]["before"] == {"frame": 2, "tid": 1, "conf": 0.9} and e[0]["after"]["tid"] == 2
    assert e[0]["id_after_gap"] == "changed" and e[0]["codes"] == {"miss": 2, "rej_conf": 1}
    assert e[1]["after"] is None                                   # GT absent at frame 10: nothing continues across it
    assert eps["exits"] == [{"gid": 7, "start": 8, "end": 8, "frames": 1, "reason": "off_pitch"}]
    assert s["observations"]["evaluable"] == 10 and s["id_changes"]["after_gap"] == 1
    assert s["episodes"]["by_group"] == {"E": 1, "C": 1}


def test_t0l_profile_changes_only_the_tracker_input():
    from football_fatigue.video.tracking import TRACKER_BACKEND, TRACKER_INPUTS

    t0, t0l, b0 = TRACKER_INPUTS["T0"], TRACKER_INPUTS["T0L"], TRACKER_INPUTS["B0"]
    assert (t0l.threshold, t0l.turf) == (0.1, "ignore") and (t0.threshold, t0.turf) == (0.4, "ignore")
    assert (b0.threshold, b0.turf) == (0.4, "cached")
    assert TRACKER_BACKEND["T0L"] == TRACKER_BACKEND["T0"] == TRACKER_BACKEND["B0"] == "sv-bytetrack"


def test_player_page_keeps_three_kinds_of_id_apart():
    html = (REPO / "src" / "football_fatigue" / "video" / "player" / "player.html").read_text(encoding="utf-8")
    assert "<b>G##</b> = SoccerTrack ground-truth player" in html and "<b>T##</b> = temporary ByteTrack ID" in html
    assert "not</i> a player identity" in html
    assert "geometry hint, not measured: pre-merge boxes were not stored" in html


# ------------------------------------------------------------------------------------------- real outputs
@needs_data
@pytest.mark.parametrize("sid", DEV)
def test_detector_stage_is_identical_across_profiles(sid):
    s = {p: json.loads((DIAG / sid / p / "summary.json").read_text(encoding="utf-8")) for p in ("B0", "T0", "T0L")}
    c = {p: s[p]["observations"]["codes"] for p in s}
    assert c["B0"]["miss"] == c["T0"]["miss"] == c["T0L"]["miss"]          # the cache does not depend on the profile
    assert c["T0"].get("rej_turf", 0) == 0 and c["T0L"].get("rej_turf", 0) == 0 and c["T0L"].get("rej_conf", 0) == 0
    assert c["B0"].get("rej_conf", 0) == c["T0"].get("rej_conf", 0)
    assert len({s[p]["observations"]["evaluable"] for p in s}) == 1


@needs_data
def test_codes_agree_with_tracks_and_cache():
    import itertools

    from football_fatigue.video.detection.diagnose import iter_cache
    from football_fatigue.video.detection.stages import iou_matrix, match
    from football_fatigue.video.failures.pipeline import CODES
    from football_fatigue.video.sources import SOURCES
    from football_fatigue.video.tracking import tracks_dir
    from football_fatigue.vision.mot_eval import load_reference

    sid, prof = "soccertrack-118576", "T0"
    gt, _ = load_reference(Path(DATA_ROOT) / SOURCES[sid]["gt"], 0)
    tr = {}
    with (tracks_dir(Path(DATA_ROOT), OUT, sid, prof) / "frames.jsonl").open(encoding="utf-8") as fh:
        for line in itertools.islice(fh, 400):
            r = json.loads(line)
            tr[r["frame_num"]] = {p["track_id"] for p in r["players"]}
    cache = dict(itertools.islice(iter_cache(Path(DATA_ROOT), sid), 400))
    with (DIAG / sid / prof / "frames.jsonl").open(encoding="utf-8") as fh:
        for line in itertools.islice(fh, 400):
            r = json.loads(line)
            f = r["frame"]
            gtb = dict(gt.get(f, []))
            gids = [g for g, _b in gt.get(f, [])]
            boxes = [d.box for d in cache[f]]
            m = dict(match(iou_matrix([gtb[g] for g in gids], boxes), list(range(len(gids))), list(range(len(boxes))), 0.5))                 if gids and boxes else {}
            matched = {gids[i] for i in m}
            for gid, ci, tid, conf in r["o"]:
                code = CODES[ci]
                assert gid in gtb                                        # rows are GT players (G ids)
                if code in ("ok", "id_changed"):
                    assert tid in tr[f]                                   # T ids come from the profile's tracks
                if code == "miss":
                    assert conf is None and tid is None and gid not in matched    # no cached box assigned to it
                if code == "rej_conf":
                    assert conf is not None and conf <= 0.4
                if code == "untracked":
                    assert conf is not None and conf > 0.4 and tid is None


@needs_data
def test_no_unsupported_pre_nms_claims():
    m = json.loads((DIAG / "soccertrack-117093" / "manifest.json").read_text(encoding="utf-8"))
    assert "nms" not in " ".join(m["codes"]).lower()
    assert any("post-merge boxes only" in x for x in m["limitations"])
    assert "not separable" in m["code_text"]["miss"]
    e = json.loads((DIAG / "soccertrack-117093" / "T0" / "episodes.json").read_text(encoding="utf-8"))
    assert all(x["cause"] in m["codes"] for x in e["episodes"])          # hints never become causes


@needs_data
def test_multiple_episodes_per_player_are_kept_separate():
    e = json.loads((DIAG / "soccertrack-117093" / "T0" / "episodes.json").read_text(encoding="utf-8"))["episodes"]
    per = Counter(x["gid"] for x in e)
    gid, n = per.most_common(1)[0]
    mine = sorted((x["start"], x["end"]) for x in e if x["gid"] == gid)
    assert n >= 2 and all(a[1] < b[0] for a, b in zip(mine, mine[1:]))   # disjoint, ordered


@needs_data
def test_t0l_tracks_match_the_benchmark_harness_and_engine_threshold():
    import gzip
    import itertools

    sid = "soccertrack-118576"
    man = json.loads((OUT / sid / "tracking" / "T0L-tracker-input-v1" / "manifest.json").read_text(encoding="utf-8"))
    assert man["configuration"]["confidence_threshold"] == 0.1 and man["configuration"]["production_confidence_threshold"] == 0.4
    t0 = json.loads((OUT / sid / "tracking" / "T0-tracker-input-v1" / "manifest.json").read_text(encoding="utf-8"))
    assert t0["configuration"]["confidence_threshold"] == 0.4                     # T0 unchanged
    bench = OUT / "_bench" / "tracker-v1" / sid / "T0L-sv-bytetrack-b75" / "tracks.jsonl.gz"
    if not bench.exists():
        pytest.skip("benchmark T0L run not present")
    with gzip.open(bench, "rt", encoding="utf-8") as b, (OUT / sid / "tracking" / "T0L-tracker-input-v1" / "frames.jsonl").open(encoding="utf-8") as t:
        rows_b = {json.loads(x)["frame"]: json.loads(x)["tracks"] for x in itertools.islice(b, 300)}
        for line in itertools.islice(t, 300):
            r = json.loads(line)
            mine = {p["track_id"]: p["bbox_xyxy_px"] for p in r["players"]}
            theirs = {x[0]: x[1:5] for x in rows_b[r["frame_num"]]}       # benchmark stores boxes rounded to 0.01 px
            assert set(mine) == set(theirs)
            assert all(max(abs(a - b) for a, b in zip(mine[k], theirs[k])) <= 0.011 for k in mine)


@needs_data
def test_t0l_engine_event_timestamps():
    run = next((OUT / "soccertrack-118576" / "occlusion" / "0.2.0").glob("occl-soccertrack-118576-0-*-v2-T0L-ff8815ed"))  # exact: T0L-A03 also exists
    for line in (run / "events.jsonl").read_text(encoding="utf-8").splitlines()[:300]:
        e = json.loads(line)
        assert e["start_pts_s"] == pytest.approx(e["source"]["pts_offset_s"] + e["start_frame"] / e["source"]["fps"], abs=1e-3)
        assert e["end_pts_s"] == pytest.approx(e["source"]["pts_offset_s"] + e["end_frame"] / e["source"]["fps"], abs=1e-3)
        assert e["run_id"].endswith("-v2-T0L-ff8815ed")


@needs_data
def test_b0_inputs_are_the_untouched_production_replay():
    from football_fatigue.video.detection.diagnose import replay_path
    from football_fatigue.video.tracking import tracks_dir

    assert tracks_dir(Path(DATA_ROOT), OUT, "soccertrack-117093", "B0") == replay_path(Path(DATA_ROOT), "soccertrack-117093").parent
    t0 = json.loads((OUT / "soccertrack-117093" / "tracking" / "T0-tracker-input-v1" / "manifest.json").read_text(encoding="utf-8"))
    assert t0["checks"]["diagnostic_replay_equal"]["identical"] is True


@needs_data
def test_reserved_match_is_refused(tmp_path):
    from football_fatigue.video.failures.pipeline import decompose
    from football_fatigue.video.player.debug import Debug
    from football_fatigue.video.sources import SourceError

    with pytest.raises((SourceError, KeyError, ValueError)):
        decompose(Path(DATA_ROOT), tmp_path, "soccertrack-118577")
    with pytest.raises((SourceError, KeyError, ValueError)):
        Debug(Path(DATA_ROOT), OUT).diag("soccertrack-118577", "T0")


@needs_data
def test_debugger_api_serves_episodes_strip_and_outcomes():
    import urllib.request

    from football_fatigue.video.player.server import start

    srv = start(Path(DATA_ROOT), OUT, port=5899)
    try:
        base = "http://127.0.0.1:5899"
        d = json.loads(urllib.request.urlopen(f"{base}/api/diag?source=soccertrack-117093&profile=T0").read())
        assert d["available"] and d["episodes"] and d["id_changes"]
        e = next(x for x in d["episodes"] if x["before"] and x["after"])
        png = urllib.request.urlopen(f"{base}/api/strip?source=soccertrack-117093&gid={e['gid']}&profile=T0&frames="
                                     f"{e['before']['frame']},{e['start']},{e['after']['frame']}").read()
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        ch = json.loads(urllib.request.urlopen(f"{base}/api/frames?source=soccertrack-117093&start={e['start']}&n=1&tracker=T0").read())
        o = ch["frames"][str(e["start"])]["o"]
        assert any(r[0] == e["gid"] and d["codes"][r[1]] == e["first_code"] for r in o)
        try:
            urllib.request.urlopen(f"{base}/api/diag?source=soccertrack-118577&profile=T0")
            raise AssertionError("reserved match served")
        except urllib.error.HTTPError as err:
            assert err.code == 400
    finally:
        srv.shutdown()


def test_session_a_vision_code_and_models_are_unmodified():
    a = Path(DATA_ROOT).parent if DATA_ROOT else Path("F:/Football-Fatigue-Project")
    if not (a / ".git").exists():
        pytest.skip("Session A repository not available")
    out = subprocess.run(["git", "-C", str(a), "status", "--porcelain", "--", "src/football_fatigue/vision", "models"],
                         capture_output=True, text=True, check=True).stdout
    assert out.strip() == ""
