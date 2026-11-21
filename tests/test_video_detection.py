"""Session B detection diagnostics: stage attribution, metrics, merge policies, replay filters, dashboard tab.

Real-data tests need FOOTBALL_FATIGUE_DATA_ROOT (a data root holding the SoccerTrack 117093 cache and replay).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from football_fatigue.video.detection.stages import (
    Det, Params, attribute_frame, average_precision, gt_cover, greedy_scored, threshold_counts, visibility,
)

SIZE = (2000, 1000)
POLY = np.asarray([[0, 0], [2000, 0], [2000, 900], [0, 900]], dtype=np.float32).reshape(-1, 1, 2)


def det(box, conf=0.9, mapped="player", turf=True):
    return Det(tuple(float(v) for v in box), conf, mapped, mapped, turf)


def stages_of(res):
    return {r["gid"]: r["stage"] for r in res["gt"]}


def test_stage_cascade_assigns_first_holding_stage():
    gt = [(1, (100, 100, 120, 150)), (2, (300, 100, 320, 150)), (3, (500, 100, 520, 150)),
          (4, (700, 100, 720, 150)), (5, (900, 100, 920, 150)), (6, (1100, 100, 1120, 150))]
    dets = [det((100, 100, 120, 150)),                         # 1: tracked
            det((300, 101, 320, 151)),                         # 2: detected, no track
            det((500, 100, 520, 150), conf=0.3),               # 3: below threshold
            det((700, 100, 720, 150), conf=0.87, turf=False),  # 4: turf filter
            det((896, 100, 916, 150)),                         # 5: IoU ~0.67 -> matched (detected_untracked)
            ]
    tracks = {7: (100, 100, 120, 150)}
    res = attribute_frame(gt, dets, tracks, POLY, SIZE)
    s = stages_of(res)
    assert s[1] == "tracked" and s[2] == "detected_untracked" and s[3] == "below_threshold"
    assert s[4] == "turf_filtered" and s[6] == "no_detection"
    row4 = next(r for r in res["gt"] if r["gid"] == 4)
    assert row4["det_conf"] == 0.87
    # precision side: the tracked and the untracked detection are TPs; below-threshold/turf boxes are not tracker input
    assert sorted(d["outcome"] for d in res["dets"]) == ["tp", "tp", "tp"]


def test_merge_suspect_and_poor_overlap():
    far = (100, 100, 120, 150)
    near = (102, 105, 124, 160)          # nearer player (larger bottom y) covering the far one
    gt = [(1, far), (2, near), (3, (500, 100, 520, 150))]
    dets = [det(near), det((505, 110, 527, 162))]   # 3 is detected with IoU ~0.4 -> poor overlap
    s = stages_of(attribute_frame(gt, dets, {}, POLY, SIZE))
    assert s[2] == "detected_untracked" and s[1] == "merge_suspect" and s[3] == "poor_overlap"


def test_gt_cover_and_visibility_classes():
    boxes = [(0, 0, 10, 20), (5, 0, 15, 30)]    # second is nearer (bottom 30) and covers the right half of the first
    cov = gt_cover(boxes)
    assert cov[0] == pytest.approx(0.5) and cov[1] == 0.0
    p = Params()
    assert visibility((100, 100, 110, 120), 0.0, 50.0, SIZE, p) == "visible"
    assert visibility((100, 100, 110, 120), 0.3, 50.0, SIZE, p) == "partial"
    assert visibility((100, 100, 110, 120), 0.7, 50.0, SIZE, p) == "heavy"
    assert visibility((100, 100, 110, 120), 0.95, 50.0, SIZE, p) == "hidden"
    assert visibility((100, 100, 110, 120), 0.0, -30.0, SIZE, p) == "off_pitch"   # > 0.5 box heights outside
    assert visibility((0, 100, 10, 120), 0.0, 50.0, SIZE, p) == "uncertain"       # touches the image border


def test_threshold_sweep_and_precision_split():
    gt = [(1, (100, 100, 120, 150))]
    dets = [det((100, 100, 120, 150), conf=0.35), det((101, 100, 121, 150), conf=0.9),   # duplicate pair
            det((100, 950, 120, 990), conf=0.8),                                         # off pitch (foot y 990)
            det((800, 100, 820, 150), conf=0.8, mapped="referee"),
            det((1500, 100, 1520, 150), conf=0.8)]
    c = threshold_counts(gt, dets, POLY, (0.2, 0.4))
    assert c[0.4]["tp"] == 1 and c[0.4]["duplicate"] == 0 and c[0.4]["off_pitch"] == 1
    assert c[0.4]["referee_label"] == 1 and c[0.4]["fp_on_pitch"] == 1
    assert c[0.2]["duplicate"] == 1


def test_average_precision():
    assert average_precision([(0.9, True), (0.8, True)], 2) == pytest.approx(1.0)
    assert average_precision([(0.9, False), (0.8, True)], 1) == pytest.approx(0.5)
    gt = [(1, (0, 0, 10, 20))]
    scored = greedy_scored(gt, [det((0, 0, 10, 20), 0.5), det((0, 0, 10, 20), 0.9)])
    assert sorted(scored) == [(0.5, False), (0.9, True)]


def test_replay_input_filters():
    from football_fatigue.video.detection.replay import VARIANTS, select

    dets = [det((100, 100, 120, 150), 0.9, turf=False), det((200, 100, 220, 150), 0.5, turf=False),
            det((300, 100, 320, 150), 0.35), det((400, 950, 420, 990), 0.9), det((500, 1, 505, 3), 0.9, mapped="ball")]
    n = lambda v: len(select(dets, VARIANTS[v], POLY))  # noqa: E731
    assert n("B0") == 1                  # only the off-pitch box survives turf + 0.4
    assert n("T0") == 3                  # turf ignored
    assert n("TG6") == 2                 # the 0.9 turf-rejected box is kept, the 0.5 one is not
    assert n("C3") == 2 and n("T0C3") == 4
    assert n("T0P") == 2                 # play-area filter drops the off-pitch box
    assert all(d.mapped != "ball" for d in select(dets, VARIANTS["T0"], POLY))


# ------------------------------------------------------------------------------------------------ merge policies
NAMES = {0: "ball", 1: "player", 2: "referee", 3: "goalkeeper"}
TILES = [(0, 0, 1280, 1280), (1024, 0, 1280, 1280)]   # overlap x 1024..1280
FRAME = (2304, 1080)


def test_merge_policies_same_tile_overlap():
    from football_fatigue.video.detection.merge import merge

    a = (1, 0.9, [100.0, 100.0, 120.0, 150.0])
    b = (1, 0.8, [103.0, 100.0, 123.0, 150.0])          # IoU ~0.74, same tile: two overlapping players
    items, tiles = [a, b], [TILES[0], TILES[0]]
    assert len(merge("A0", items, tiles, FRAME, NAMES)) == 1      # production NMS removes one
    assert len(merge("A1", items, tiles, FRAME, NAMES)) == 2      # cross-tile only keeps both
    assert len(merge("A2", items, tiles, FRAME, NAMES)) == 2
    assert len(merge("A3", items, tiles, FRAME, NAMES)) == 1      # IoU 0.74 > 0.7


def test_merge_policies_cross_tile_duplicate():
    from football_fatigue.video.detection.merge import merge

    a = (1, 0.9, [1100.0, 100.0, 1120.0, 150.0])
    b = (1, 0.85, [1101.0, 100.0, 1121.0, 150.0])        # same player seen by both tiles, inside the overlap
    items, tiles = [a, b], [TILES[0], TILES[1]]
    for p in ("A0", "A1", "A2", "A3"):
        assert len(merge(p, items, tiles, FRAME, NAMES)) == 1, p
    c = (1, 0.9, [600.0, 100.0, 620.0, 150.0])           # outside the overlap region of the two tiles
    d = (1, 0.85, [603.0, 100.0, 623.0, 150.0])
    assert len(merge("A1", [c, d], tiles, FRAME, NAMES)) == 1
    assert len(merge("A2", [c, d], tiles, FRAME, NAMES)) == 2     # cannot be a cross-tile duplicate there


def test_merge_a0_is_production_code():
    from football_fatigue.video.detection.merge import _detector, merge

    rng = np.random.default_rng(0)
    items, tiles = [], []
    for _ in range(40):
        x, y = rng.uniform(0, 2200), rng.uniform(0, 1000)
        items.append((int(rng.integers(1, 4)), float(rng.uniform(0.1, 1)), [x, y, x + 20, y + 45]))
        tiles.append(TILES[int(rng.integers(0, 2))])
    ref = _detector(NAMES, 0.5, 0.7)._merge_person_agnostic(items, tiles, FRAME)
    assert merge("A0", items, tiles, FRAME, NAMES) == ref


def test_capture_requires_approval(capsys):
    from football_fatigue.video.detection.__main__ import main

    with pytest.raises(SystemExit):
        main(["capture", "--data-root", ".", "--source", "soccertrack-117093", "--start", "0", "--frames", "1"])
    assert "approval" in capsys.readouterr().err


def test_line_index(tmp_path):
    from football_fatigue.video.detection.framestore import LineIndex

    p = tmp_path / "f.jsonl"
    p.write_text("".join(json.dumps({"frame_num": i, "v": i * i}) + "\n" for i in (0, 1, 2, 10)), encoding="utf-8")
    idx = LineIndex(p, "frame_num")
    assert idx.get(10)["v"] == 100 and idx.get(2)["v"] == 4 and idx.get(5) is None


def test_stage_table_and_report_handle_partial_inputs():
    from football_fatigue.video.dashboard.detection_tab import stage_table
    from football_fatigue.video.detection.report import loss_rows

    summary = {"role": "tune", "stage_visibility": {"tracked": {"visible": 90}, "turf_filtered": {"visible": 10}}}
    t = stage_table(summary)
    assert t.loc["tracked", "visible"] == "90.0%" and t.loc["turf_filtered", "visible"] == "10.0%"
    rows = loss_rows({"soccertrack-117093": summary})
    assert "10 (10.0%)" in rows[-1] and "10 (100%)" in rows[-1]


# ------------------------------------------------------------------------------------------------ real data
DATA_ROOT = os.environ.get("FOOTBALL_FATIGUE_DATA_ROOT")
OUT_ROOT = Path(__file__).resolve().parents[1] / "data" / "processed" / "video"
needs_data = pytest.mark.skipif(
    not DATA_ROOT or not (Path(DATA_ROOT or ".") / "external" / "soccertrack-v2" / "raw" / "117093").is_dir(),
    reason="set FOOTBALL_FATIGUE_DATA_ROOT to a data root holding SoccerTrack 117093")


@needs_data
def test_baseline_replay_reproduces_production():
    from football_fatigue.video.detection.replay import VARIANTS, baseline_check, run_tracker

    tracks = run_tracker(Path(DATA_ROOT), "soccertrack-117093", VARIANTS["B0"], max_frames=150)
    assert baseline_check(Path(DATA_ROOT), "soccertrack-117093", tracks)["identical"]


@needs_data
def test_diagnose_small_window_and_framestore_agree(tmp_path):
    import csv
    import gzip

    from football_fatigue.video.detection.diagnose import diagnose
    from football_fatigue.video.detection.framestore import FrameStore

    out = diagnose(Path(DATA_ROOT), tmp_path, "soccertrack-117093", max_frames=60, engine_root=OUT_ROOT)
    s = json.loads((out / "summary.json").read_text())
    assert s["frames"] == 60 and s["role"] == "tune"
    with pytest.raises(FileExistsError):
        diagnose(Path(DATA_ROOT), tmp_path, "soccertrack-117093", max_frames=60)
    rows = {(int(r["frame"]), int(r["gid"])): r["stage"]
            for r in csv.DictReader(gzip.open(out / "gt_rows.csv.gz", "rt", encoding="utf-8"))}
    fs = FrameStore(Path(DATA_ROOT), tmp_path, "soccertrack-117093")
    for f in (0, 30, 59):
        assert {(f, r["gid"]): r["stage"] for r in fs.attribution(f)["gt"]} == \
               {k: v for k, v in rows.items() if k[0] == f}


@needs_data
@pytest.mark.skipif(not any(OUT_ROOT.glob("*/detection/diag-v1/summary.json")), reason="no diagnosis outputs")
def test_dashboard_detection_tab_smoke(monkeypatch):
    from streamlit.testing.v1 import AppTest

    app = Path(__file__).resolve().parents[1] / "src" / "football_fatigue" / "video" / "dashboard" / "app.py"
    monkeypatch.setattr("sys.argv", ["app.py", "--data-root", DATA_ROOT, "--out-root", str(OUT_ROOT)])
    at = AppTest.from_file(str(app), default_timeout=300)
    at.run()
    assert not at.exception
    f0 = at.session_state["dx_frame"]
    next(b for b in at.button if b.key == "dx_p1").click().run()
    assert not at.exception and at.session_state["dx_frame"] == f0 + 1
    next(b for b in at.button if b.key == "dx_next").click().run()
    assert not at.exception and at.session_state["dx_frame"] > f0 + 1
