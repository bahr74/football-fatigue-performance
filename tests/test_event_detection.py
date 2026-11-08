"""Event detection tests on SYNTHETIC positions (hand-built trajectories, not PFF data or schemas)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from football_fatigue.event_detection import DetectorConfig, detect
from football_fatigue.event_detection import evaluate as E
from football_fatigue.event_detection.export import to_pff_like, write_run
from football_fatigue.event_detection.inputs import HOME, AWAY, MatchPositions, attack_directions, from_match_data
from football_fatigue.match_data import CoordinatePolicy, CoordState

FPS = 29.97


def synthetic_match(players: dict[str, tuple[int, tuple[float, float]]], ball_path: list[tuple[float, float, float, float]],
                    duration_s: float, *, missing: list[tuple[float, float]] = (), cadence: int = 2, carrier: dict | None = None) -> MatchPositions:
    """SYNTHETIC match. ``ball_path``: (t, x, y, z) keyframes, linearly interpolated; ``missing``: (t0, t1) ball gaps;
    ``cadence``: a new ball sample every n frames (repeats in between, like the raw source). Home attacks +x in period 1."""
    n = int(duration_s * FPS)
    t = 100.0 + np.arange(n) / FPS
    kt = np.array([k[0] for k in ball_path]) + 100.0
    ball = np.stack([np.interp(t, kt, [k[i] for k in ball_path]) for i in (1, 2, 3)], 1)
    for i in range(n):
        if i % cadence:
            ball[i] = ball[i - i % cadence]
    for t0, t1 in missing:
        ball[(t >= 100 + t0) & (t < 100 + t1)] = np.nan
    keys = list(players)
    px = np.array([[players[k][1][0] for k in keys]] * n, dtype=np.float32)
    py = np.array([[players[k][1][1] for k in keys]] * n, dtype=np.float32)
    return MatchPositions(match_id=999, source="synthetic", fps=FPS, home_team_id=1, away_team_id=2, home_attacks_right={1: True, 2: False},
                          t=t, period=np.ones(n, dtype=np.int8), frame_num=np.arange(n), ball=ball.astype(np.float32), px=px, py=py,
                          player_side=np.array([players[k][0] for k in keys], dtype=np.int8), player_key=keys,
                          player_id=np.arange(10, 10 + len(keys)))


TEAM = {"home:4": (HOME, (0.0, 0.0)), "home:8": (HOME, (15.0, 0.0)), "away:5": (AWAY, (-20.0, 20.0)), "away:9": (AWAY, (-25.0, -20.0))}


def test_completed_pass_between_teammates():
    # ball at home:4 until t=1, kicked at 12 m/s to home:8 (15 m), stopped there
    pos = synthetic_match(TEAM, [(0, 0, 0, 0), (1.0, 0, 0, 0), (2.25, 15, 0, 0), (5, 15, 0, 0)], 5)
    events = detect(pos).events
    assert [e.type for e in events] == ["PA"]
    e = events[0]
    assert e.outcome == "C" and pos.player_key[e.touch.player] == "home:4" and pos.player_key[e.receiver] == "home:8"
    assert abs(e.touch.t_s - 101.0) <= 0.25


def test_pass_to_opponent_is_defended():
    players = dict(TEAM, **{"away:5": (AWAY, (15.0, 0.0))}, **{"home:8": (HOME, (-30.0, 10.0))})
    pos = synthetic_match(players, [(0, 0, 0, 0), (1.0, 0, 0, 0), (2.25, 15, 0, 0), (5, 15, 0, 0)], 5)
    (e,) = detect(pos).events
    assert e.type == "PA" and e.outcome == "D" and e.receiver is None


def test_ball_lost_mid_flight_makes_no_pass_claim():
    pos = synthetic_match(TEAM, [(0, 0, 0, 0), (1.0, 0, 0, 0), (2.25, 15, 0, 0), (5, 15, 0, 0)], 5, missing=[(1.4, 4.0)])
    assert [e for e in detect(pos).events if e.outcome in ("C", "D")] == []


def test_shot_into_goal_followed_by_stoppage_is_a_goal():
    players = {"home:9": (HOME, (35.0, 0.0)), "away:1": (AWAY, (52.0, 8.0)), "home:4": (HOME, (0.0, 0.0)), "away:5": (AWAY, (-5.0, 3.0))}
    # struck at 25 m/s toward the goal mouth (x = +52.5), crosses the line at y = 1, then no play for 70 s, kickoff at the centre
    path = [(0, 35, 0, 0), (1.0, 35, 0, 0), (1.72, 53, 1, 0.5), (2.0, 53.5, 1, 0.3), (72, 0, 0, 0), (73, 0, 0, 0), (73.5, 6, 0, 0), (75, 6, 0, 0)]
    pos = synthetic_match(players, path, 75, missing=[(2.2, 71.9)])
    shots = [e for e in detect(pos).events if e.type == "SH"]
    assert len(shots) == 1 and shots[0].is_goal and shots[0].reason["ball_crossed_goal_line"]


def test_cross_from_the_wing_into_the_box():
    players = {"home:7": (HOME, (40.0, 28.0)), "home:9": (HOME, (47.0, 2.0)), "away:4": (AWAY, (10.0, 0.0)), "away:5": (AWAY, (0.0, -20.0))}
    path = [(0, 40, 28, 0), (1.0, 40, 28, 0), (2.5, 47, 2, 1.0), (5, 47, 2, 0)]
    (e,) = detect(pos := synthetic_match(players, path, 5)).events
    assert e.type == "CR" and e.outcome == "C" and pos.player_key[e.receiver] == "home:9"


def test_dribble_is_not_a_pass():
    players = {"home:4": (HOME, (0.0, 0.0)), "away:5": (AWAY, (-20.0, 20.0))}
    # the ball is pushed 1.5 m ahead and stopped by the same player, twice
    pos = synthetic_match(players, [(0, 0, 0, 0), (1, 0, 0, 0), (1.2, 1.5, 0, 0), (2, 1.5, 0, 0), (2.2, 0, 0, 0), (4, 0, 0, 0)], 4)
    assert detect(pos).events == []


def test_attack_directions_flip_by_period():
    meta = SimpleNamespace(home_team_start_left=True, home_team_start_left_extra_time=None)
    assert attack_directions(meta) == {1: True, 2: False}


def _fake_frame(ball_raw, ball_smoothed):
    player = SimpleNamespace(source_side="home", jersey_number=4, identity_usable=True, player_id=77,
                             raw=SimpleNamespace(state=CoordState.AVAILABLE, x=1.0, y=2.0))
    ball = SimpleNamespace(raw=ball_raw, smoothed=ball_smoothed)
    return SimpleNamespace(canonical_period=1, video_time_s=10.0, frame_num=300, ball=ball, players=(player,),
                           frame=SimpleNamespace(possession_event_id=123, game_event_id=456))


def test_adapter_reads_raw_ball_only_never_the_event_smoothed_ball():
    missing = SimpleNamespace(state=CoordState.MISSING, x=None, y=None, z=None)
    smoothed = SimpleNamespace(state=CoordState.AVAILABLE, x=5.0, y=5.0, z=0.0)
    md = SimpleNamespace(policy=CoordinatePolicy.RAW, match_id="999",
                         metadata=SimpleNamespace(fps=FPS, home_team_id=1, away_team_id=2, home_team_start_left=True,
                                                  home_team_start_left_extra_time=None),
                         frames=lambda max_frames=None: iter([_fake_frame(missing, smoothed)]))
    pos = from_match_data(md)
    assert np.isnan(pos.ball[0]).all()
    assert pos.player_key == ["home:4"] and int(pos.player_id[0]) == 77 and pos.px[0, 0] == 1.0


def test_adapter_rejects_non_raw_policy():
    with pytest.raises(ValueError):
        from_match_data(SimpleNamespace(policy=CoordinatePolicy.SMOOTHED))


def _lab(typ, t, team=1, outcome="C", key=False, period=1):
    return E.LabelledEvent(typ, t, period, team, None, outcome, key, f"{typ}{t}")


def test_matching_is_one_to_one_within_tolerance_and_period():
    truth = [_lab("PA", 10.0), _lab("PA", 10.4), _lab("PA", 20.0, period=2)]
    pred = [_lab("PA", 10.1), _lab("PA", 20.0)]
    m = E.match_events(pred, truth, 1.0)
    assert [(i, j) for i, j, _ in m] == [(0, 0)]


def test_scores_count_tp_fp_fn():
    s = E.score([_lab("PA", 10.0), _lab("SH", 30.0, outcome="G")], [_lab("PA", 10.3), _lab("PA", 15.0), _lab("SH", 30.5, outcome="S")])
    assert s["counts"]["PA@1.0"] == dict(tp=1, fp=0, fn=1, n_pred=1, n_truth=2)
    assert s["counts"]["GOAL@1.0"]["fp"] == 1 and s["counts"]["SH@1.0"]["tp"] == 1
    assert s["confusion"]["PA->missed"] == 1


def _row(i, typ, t, team, pe):
    return SimpleNamespace(event_time_s=t, canonical_period=1, possession_event_type=typ, attributes={"possessionEvents": pe},
                           source_team_id=team, source_player_id=None, event_row_id=f"r{i}")


def test_truth_key_pass_is_last_completed_pass_before_a_shot_without_opponent_action():
    rows = [_row(0, "PA", 1.0, 1, {"passOutcomeType": "C"}), _row(1, "PA", 3.0, 1, {"passOutcomeType": "C"}), _row(2, "SH", 5.0, 1, {}),
            _row(3, "PA", 20.0, 1, {"passOutcomeType": "C"}), _row(4, "CH", 21.0, 2, {}), _row(5, "SH", 22.0, 1, {})]
    keys = [e.t_s for e in E.truth_events(rows, DetectorConfig()) if e.key_pass]
    assert keys == [3.0]


def test_export_is_pff_shaped_and_never_writes_into_raw(tmp_path):
    pos = synthetic_match(TEAM, [(0, 0, 0, 0), (1.0, 0, 0, 0), (2.25, 15, 0, 0), (5, 15, 0, 0)], 5)
    rec = to_pff_like(pos, detect(pos).events)
    assert rec[0]["possessionEvents"]["possessionEventType"] == "PA" and rec[0]["possessionEvents"]["passOutcomeType"] == "C"
    assert rec[0]["gameEvents"]["teamId"] == 1 and rec[0]["detector"]["reason"]
    raw = tmp_path / "raw"
    raw.mkdir()
    with pytest.raises(ValueError):
        write_run(raw / "out", raw, 999, rec, {})
    dest = write_run(tmp_path / "runs", raw, 999, rec, {"m": 1})
    assert json.loads((dest / "events.json").read_text(encoding="utf-8"))[0]["gameId"] == 999
    with pytest.raises(FileExistsError):
        write_run(tmp_path / "runs", raw, 999, rec, {})


def test_long_pause_without_ball_at_goal_mouth_is_not_a_goal():
    players = {"home:9": (HOME, (35.0, 0.0)), "away:1": (AWAY, (52.0, 8.0)), "home:4": (HOME, (0.0, 0.0)), "away:5": (AWAY, (-5.0, 3.0))}
    # a shot that is stopped 8 m out (e.g. an injury stoppage follows): long pause, but the ball never reaches the goal mouth
    path = [(0, 35, 0, 0), (1.0, 35, 0, 0), (1.4, 44.5, 0.5, 0.3), (1.6, 44.5, 0.5, 0.3), (72, 0, 0, 0), (73, 0, 0, 0), (73.5, 6, 0, 0), (75, 6, 0, 0)]
    pos = synthetic_match(players, path, 75, missing=[(2.2, 71.9)])
    shots = [e for e in detect(pos).events if e.type == "SH"]
    assert shots and not any(s.is_goal for s in shots)
    assert shots[0].reason["stoppage_after_shot_s"] >= 60 and shots[0].reason["ball_goal_mouth_distance_m"] > 1.5


def test_ml_folds_never_score_a_match_with_a_model_that_saw_it(tmp_path):
    pytest.importorskip("sklearn")
    from football_fatigue.event_detection.features import FEATURES
    from football_fatigue.event_detection.ml import CLASSES, N_FOLDS, TouchClassifier, TouchTable, fold_of, train_folds
    rng = np.random.default_rng(0)
    matches = np.repeat(np.arange(100, 100 + 2 * N_FOLDS), 40)                    # SYNTHETIC feature rows, not real data
    X = rng.normal(size=(matches.size, len(FEATURES)))
    y = (X[:, 0] > 1.0).astype(int) + (X[:, 1] > 1.5).astype(int) * 3            # classes 0, 1, 3, 4
    y[y > 4] = 4
    done = np.where(y == 1, (X[:, 2] > 0).astype(int), -1)
    man = train_folds(TouchTable(matches, X, y, done), tmp_path, max_iter=10)
    folds = fold_of(matches)
    assert man["folds"] == {str(m): f for m, f in folds.items()} and set(folds.values()) == set(range(N_FOLDS))
    clf = TouchClassifier(tmp_path)
    name, proba = clf.predict_proba(101, X[:5])
    assert name == f"fold_{folds[101]}" and proba.shape == (5, len(CLASSES))
    assert clf.model_for(99999)[0] == "full" and clf.predict_completion(101, X[:5]).shape == (5,)


def test_ml_rows_pick_the_most_confident_actor_and_use_the_completion_model():
    from football_fatigue.event_detection.features import actor_rows
    from football_fatigue.event_detection.kinematics import ball_track
    from football_fatigue.event_detection.ml import CLASSES, events_from_rows
    from football_fatigue.event_detection.touches import detect_touches
    players = dict(TEAM, **{"away:5": (AWAY, (1.5, 1.5))})                       # an opponent close to the passer
    pos = synthetic_match(players, [(0, 0, 0, 0), (1.0, 0, 0, 0), (2.25, 15, 0, 0), (5, 15, 0, 0)], 5)
    cfg = DetectorConfig(touch_radius_m=4.0)
    ball = ball_track(pos, cfg)
    touches = detect_touches(pos, ball, cfg)
    rows = actor_rows(touches)
    assert rows[1][0] == 0 and rows[1][1].side != rows[0][1].side                # both candidate actors of the first touch
    proba = np.zeros((len(rows), len(CLASSES)))
    proba[:, 0] = 1.0
    proba[0] = [0.4, 0.6, 0, 0, 0]                                               # nearest actor: a pass, 0.6
    proba[1] = [0.1, 0.9, 0, 0, 0]                                               # other-side actor: a pass, 0.9 -> wins
    (e,) = events_from_rows(pos, ball, touches, rows, proba, np.full(len(rows), 0.2), cfg)
    assert e.type == "PA" and e.confidence == 0.9 and e.team_side == rows[1][1].side and e.outcome == "D"


def test_viewer_page_compares_detections_with_labels(tmp_path):
    from football_fatigue.event_detection.viewer import write_viewer
    pos = synthetic_match(TEAM, [(0, 0, 0, 0), (1.0, 0, 0, 0), (2.25, 15, 0, 0), (5, 15, 0, 0)], 5)
    res = detect(pos)
    truth = [E.LabelledEvent("PA", 101.1, 1, 1, 10, "C", False, "r1", 11, 0.5, 0.0),     # SYNTHETIC label, matched
             E.LabelledEvent("SH", 103.0, 1, 1, 11, "S", False, "r2")]                   # SYNTHETIC label, missed
    score = E.score(E.detected_as_labelled(pos, res.events), truth)
    names = {"by_key": {"home:4": "Player Four (#4)"}, "by_id": {10: "Player Four (#4)"}}
    out = write_viewer(tmp_path / "v.html", pos, res.events, truth, [], res.touches, score, names,
                       {"home": "Home FC", "away": "Away FC", "date": "2022-01-01", "stage": "Test"}, DetectorConfig(), "rules-test")
    page = out.read_text(encoding="utf-8")
    data = json.loads(page.split("const D=", 1)[1].split(";\n", 1)[0])
    status = sorted(it["status"] for it in data["items"])
    assert status == ["hit", "missed"] and data["items"][0]["checks"]["team"] is True
    assert "How events are detected" in page and "Gaps &amp; fixes" in page and "Player Four (#4)" in page
