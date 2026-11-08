"""Tests for the M1 baselines (tag-free) and the scoring. All tracks are SYNTHETIC (made up here), not source data."""
import numpy as np

from m1_labels.scoring import change_events, match_events, score
from m1_possession import FLAG_MISSING, FLAG_OBSERVED, SmoothedBall
from m1_possession.baselines import CONTROL, DEAD, LOOSE, TRAVELLING, c0_naive, c1_paper

FPS = 30.0


def ball_from(x, y, flag=None):
    n = len(x)
    t = np.arange(n) / FPS
    vx, vy = np.gradient(x, t), np.gradient(y, t)
    fl = np.full(n, FLAG_OBSERVED, np.int8) if flag is None else flag
    return t, SmoothedBall(x, y, np.zeros(n), vx, vy, np.zeros(n), np.hypot(vx, vy), fl)


def pass_scene():
    """Player A (home, id 1) at (0,0) holds the ball, passes at 10 m/s to B (home, id 2) at (20,0), who stops it."""
    n = 150
    x = np.zeros(n)
    x[30:60] = np.linspace(0, 20, 31)[1:]          # 30 frames = 1 s of flight
    x[60:] = 20.0
    y = np.zeros(n)
    y[60:] += np.linspace(0, 3, 90)                 # B dribbles sideways: direction change at the reception
    px = np.array([[0.3, 20.3, 40.0]] * n, dtype=float)
    py = np.zeros((n, 3))
    py[60:, 1] = y[60:]
    return x, y, px, py, np.array([1, 2, 3]), np.array([10, 10, 20])


def test_c0_states():
    x, y, px, py, pid, team = pass_scene()
    flag = np.full(len(x), FLAG_OBSERVED, np.int8)
    flag[140:] = FLAG_MISSING
    t, b = ball_from(x, y, flag)
    r = c0_naive(t, np.ones(len(t), int), b, px, py, pid, team)
    assert r.state[10] == CONTROL and r.player[10] == 1
    assert r.state[45] == TRAVELLING and r.player[45] == -1
    assert r.state[100] == CONTROL and r.player[100] == 2
    assert (r.state[140:] == DEAD).all()


def test_c1_pass_between_teammates():
    x, y, px, py, pid, team = pass_scene()
    t, b = ball_from(x, y)
    # the ball must enter A's zone with a direction change: A receives it moving in +y, then kicks along +x
    b.y[:10] = np.linspace(-1.5, 0, 10)
    r = c1_paper(t, np.ones(len(t), int), b, px, py, pid, team, 105.0, 68.0)
    assert r.state[45] == TRAVELLING and r.team[45] == 10
    assert r.state[120] == CONTROL and r.player[120] == 2


def test_c1_duel_is_loose():
    n = 60
    x = np.r_[np.linspace(-3, 0, 30), np.zeros(30)]             # ball arrives along x, leaves along y: a valid gain
    y = np.r_[np.zeros(30), np.linspace(0, 2, 30)]
    px = np.array([[0.4, -0.4]] * n)
    py = np.zeros((n, 2))
    t, b = ball_from(x, y)
    r = c1_paper(t, np.ones(n, int), b, px, py, np.array([1, 2]), np.array([10, 20]), 105.0, 68.0)
    assert LOOSE in r.state[25:35]


def test_c1_out_of_bounds_dead_until_restart():
    n = 200
    x = np.r_[np.linspace(40, 54, 40), np.full(100, 53.5), np.linspace(52.5, 45, 60)]    # over the goal line, back for a goal kick
    y = np.r_[np.full(140, 20.0), np.full(60, 3.0)]
    px = np.array([[40.0, 52.0]] * n)
    py = np.array([[20.0, 3.0]] * n)
    px[:40, 0] = x[:40]
    t, b = ball_from(x, y)
    r = c1_paper(t, np.ones(n, int), b, px, py, np.array([1, 2]), np.array([10, 20]), 105.0, 68.0)
    assert (r.state[60:130] == DEAD).all()


def test_scoring_perfect_prediction():
    st = np.array([0] * 10 + [1] * 10 + [2] * 10 + [1] * 10 + [0] * 10)
    tm = np.array([-1] * 10 + [10] * 10 + [10] * 10 + [20] * 10 + [-1] * 10)
    lab = dict(state=st, player_id=tm, team_id=tm, score_mask=np.ones(50, bool), period=np.ones(50, int), rule=np.zeros(50, int))
    s = score(lab, st, tm, tm, FPS)
    assert s["accuracy"] == 1.0 and s["player_acc"] == 1.0
    assert s["possession_change"]["f1"] == 1.0 and s["possession_change"]["n_true"] == 1
    assert s["dead_start_0.3s"]["f1"] == 1.0 and s["dead_end_0.3s"]["f1"] == 1.0
    assert list(change_events(st, tm, np.ones(50, int))) == [30]


def test_match_events_one_to_one():
    assert match_events(np.array([10, 12]), np.array([11]), 3) == 1
    assert match_events(np.array([10, 40]), np.array([12, 38]), 3) == 2
    assert match_events(np.array([10]), np.array([20]), 3) == 0
