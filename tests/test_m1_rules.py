"""Tests for C2 (full rules). Every scene is SYNTHETIC (generated here), not source data.

A scene has two teams (home id 10 attacks +x, defends the -x half; away id 20 the opposite), a keeper each, background
players jogging so that team activity looks like live play, and a ball whose path is written piece by piece.
"""
import numpy as np

from m1_possession import FLAG_MISSING, FLAG_OBSERVED, SmoothedBall
from m1_possession.rules import CONTROL, DEAD, LOOSE, TRAVELLING, C2Params, run_c2

FPS = 30.0
PITCH = np.array([105.0, 68.0])


class Scene:
    def __init__(self, seconds, slow_from=None, slow_to=None):
        self.n = int(seconds * FPS)
        self.t = np.arange(self.n) / FPS
        self.cols = []                      # (id, team, x array, y array)
        self.bx = np.full(self.n, np.nan)
        self.by = np.full(self.n, np.nan)
        self.bz = np.zeros(self.n)
        rng = np.random.default_rng(0)
        for team, side in ((10, -1), (20, 1)):
            self.add(team * 100 + 1, team, side * 50.0, 0.0)                     # keeper on the goal line
            for k in range(7):                                                  # joggers in the own half
                cx, cy = side * (15 + 4 * k), -24 + 8 * k
                ph = rng.uniform(0, 6.28)
                w = 3.0 / 4.0                                                   # 3 m/s on a 4 m circle
                x = cx + 4 * np.cos(w * self.t + ph)
                y = cy + 4 * np.sin(w * self.t + ph)
                if slow_from is not None:
                    a, b = int(slow_from * FPS), int(slow_to * FPS)
                    x[a:b], y[a:b] = x[a], y[a] + 0.01 * np.arange(b - a)        # walking (0.3 m/s)
                    x[b:], y[b:] = x[b:] - x[b] + x[a], y[b:] - y[b] + y[b - 1]
                self.cols.append((team * 100 + 10 + k, team, x, y))

    def add(self, pid, team, x, y):
        self.cols.append((pid, team, np.full(self.n, float(x)), np.full(self.n, float(y))))
        return len(self.cols) - 1

    def move(self, col, f0, f1, x1, y1):
        """Player moves linearly from his position at f0 to (x1, y1) at f1 and stays there."""
        _, _, x, y = self.cols[col]
        x0, y0 = x[f0], y[f0]
        k = np.linspace(0, 1, f1 - f0 + 1)
        x[f0:f1 + 1], y[f0:f1 + 1] = x0 + (x1 - x0) * k, y0 + (y1 - y0) * k
        x[f1:], y[f1:] = x1, y1

    def ball_line(self, f0, f1, p0, p1, z=0.0):
        k = np.linspace(0, 1, f1 - f0 + 1)
        self.bx[f0:f1 + 1] = p0[0] + (p1[0] - p0[0]) * k
        self.by[f0:f1 + 1] = p0[1] + (p1[1] - p0[1]) * k
        self.bz[f0:f1 + 1] = z

    def ball_with(self, col, f0, f1, dx=0.4):
        _, _, x, y = self.cols[col]
        self.bx[f0:f1 + 1], self.by[f0:f1 + 1] = x[f0:f1 + 1] + dx, y[f0:f1 + 1]

    def run(self, **kw):
        flag = np.where(np.isfinite(self.bx), FLAG_OBSERVED, FLAG_MISSING).astype(np.int8)
        vx, vy = np.gradient(np.nan_to_num(self.bx), self.t), np.gradient(np.nan_to_num(self.by), self.t)
        ball = SmoothedBall(self.bx, self.by, self.bz, vx, vy, np.zeros(self.n), np.hypot(vx, vy), flag)
        px = np.stack([c[2] for c in self.cols], 1)
        py = np.stack([c[3] for c in self.cols], 1)
        pid = np.array([c[0] for c in self.cols])
        team = np.array([c[1] for c in self.cols])
        return run_c2(self.t, np.ones(self.n, int), ball, px, py, pid, team, PITCH, C2Params(**kw))


def f(s):
    return int(s * FPS)


def rule_at(r, s):
    return r.rule_names[r.rule[f(s)]]


def dribble(sc, col, f0, f1, vx=2.0, vy=0.0):
    _, _, x, y = sc.cols[col]
    sc.move(col, f0, f1, x[f0] + vx * (f1 - f0) / FPS, y[f0] + vy * (f1 - f0) / FPS)
    sc.ball_with(col, f0, f1)


def kickoff(sc, col):
    """Ball at the centre spot, taker next to it, everybody in the own half; first kick at 1.0 s."""
    sc.ball_line(0, f(1.0), (0.0, 0.0), (0.0, 0.0))


def test_control_and_pass():
    sc = Scene(8)
    b = sc.add(1, 10, -10.0, 0.0)
    a = sc.add(2, 10, 5.0, 0.0)
    sc.move(b, 0, f(1.0), -10.0, 0.0)
    sc.ball_line(0, f(1.0), (-9.6, 0.0), (-9.6, 0.0))                 # placed ball at B
    sc.ball_line(f(1.0), f(2.0), (-9.6, 0.0), (4.6, 0.0))             # pass, about 14 m/s
    dribble(sc, a, f(2.0), f(8.0) - 1)
    r = sc.run()
    assert r.state[f(1.5)] == TRAVELLING and r.team[f(1.5)] == 10
    assert r.state[f(5.0)] == CONTROL and r.player[f(5.0)] == 2
    assert len(r.passes) >= 1 and r.passes[0][1] == 1


def test_one_touch_pass():
    sc = Scene(8)
    b, a, c = sc.add(1, 10, -10.0, 0.0), sc.add(2, 10, 5.0, 0.0), sc.add(3, 10, 5.0, 15.0)
    sc.ball_line(0, f(1.0), (-9.6, 0.0), (-9.6, 0.0))
    sc.ball_line(f(1.0), f(2.0), (-9.6, 0.0), (4.6, 0.0))
    sc.ball_line(f(2.0), f(3.0), (4.6, 0.0), (4.8, 14.6))             # redirected at once
    dribble(sc, c, f(3.0), f(8.0) - 1)
    r = sc.run()
    names = {r.rule_names[k] for k in r.rule[f(1.9):f(2.3)]}
    assert "one_touch" in names
    assert r.state[f(2.5)] == TRAVELLING and r.player[f(2.5)] == 2


def test_interception():
    sc = Scene(8)
    b, a = sc.add(1, 10, -10.0, 0.0), sc.add(2, 10, 10.0, 0.0)
    o = sc.add(3, 20, 0.0, 0.5)
    sc.ball_line(0, f(1.0), (-9.6, 0.0), (-9.6, 0.0))
    sc.ball_line(f(1.0), f(1.7), (-9.6, 0.0), (-0.4, 0.4))
    dribble(sc, o, f(1.7), f(8.0) - 1, vx=-2.0)
    r = sc.run()
    assert r.state[f(5.0)] == CONTROL and r.player[f(5.0)] == 3 and r.team[f(5.0)] == 20
    assert "interception" in {r.rule_names[k] for k in r.rule[f(1.6):f(5.0)]}


def test_block_is_loose():
    sc = Scene(8)
    b, o = sc.add(1, 10, -10.0, 0.0), sc.add(3, 20, -6.5, 0.3)
    sc.ball_line(0, f(1.0), (-9.6, 0.0), (-9.6, 0.0))
    sc.ball_line(f(1.0), f(1.2), (-9.6, 0.0), (-6.8, 0.0))           # 14 m/s at the blocker
    sc.ball_line(f(1.2), f(3.0), (-6.8, 0.0), (-6.8 - 12, 12.0))      # bounces off at about 9 m/s, nobody near
    sc.ball_line(f(3.0), f(8.0) - 1, (-18.8, 12.0), (-18.8, 12.0))
    r = sc.run()
    assert r.state[f(2.0)] == LOOSE


def test_duel_is_loose():
    sc = Scene(8)
    a, o = sc.add(2, 10, 0.0, 0.0), sc.add(3, 20, 0.6, 0.6)
    k = np.arange(f(2.0), f(6.0))
    sc.ball_line(0, f(2.0), (-10.0, 0.0), (0.3, 0.3))                  # arrives fast
    sc.bx[k] = 0.3 + 0.3 * np.sign(np.sin(k / 4.0))                     # poked back and forth between the two
    sc.by[k] = 0.3
    sc.ball_line(f(6.0), f(8.0) - 1, (0.3, 0.3), (0.3, 0.3))
    r = sc.run()
    assert LOOSE in r.state[f(2.0):f(6.0)]
    assert "duel" in {r.rule_names[x] for x in r.rule[f(2.0):f(6.0)]}


def test_keeper_catch():
    sc = Scene(8)
    s = sc.add(1, 10, 34.0, 0.0)
    gk = next(i for i, c in enumerate(sc.cols) if c[0] == 2001)
    sc.ball_line(0, f(1.0), (34.4, 0.0), (34.4, 0.0))
    sc.ball_line(f(1.0), f(1.8), (34.4, 0.0), (49.6, 0.0), z=1.2)      # shot at 19 m/s, chest height
    sc.ball_line(f(1.8), f(8.0) - 1, (49.6, 0.0), (49.6, 0.0), z=1.0)  # held by the keeper
    r = sc.run()
    assert r.state[f(4.0)] == CONTROL and r.player[f(4.0)] == 2001
    assert "keeper_catch" in {r.rule_names[x] for x in r.rule[f(1.7):f(4.0)]}


def test_ball_out_and_throw_in():
    sc = Scene(12)
    a = sc.add(2, 10, 0.0, 28.0)
    taker = sc.add(3, 20, 5.0, 34.2)
    mate = sc.add(4, 20, 5.0, 24.0)
    dribble(sc, a, 0, f(1.0), vx=0.0, vy=2.0)
    sc.ball_line(f(1.0), f(1.5), (0.4, 30.0), (0.4, 37.0))             # kicked over the touchline
    sc.ball_line(f(1.5), f(4.0), (0.4, 37.0), (5.0, 35.0))             # fetched
    sc.ball_line(f(4.0), f(7.0), (5.0, 34.0), (5.0, 34.0), z=2.0)     # held over the head on the line
    sc.ball_line(f(7.0), f(8.0), (5.0, 34.0), (5.0, 24.4))             # thrown in
    dribble(sc, mate, f(8.0), f(12.0) - 1)
    r = sc.run()
    assert r.state[f(3.0)] == DEAD and rule_at(r, 3.0) == "out_of_play"
    assert r.state[f(10.0)] == CONTROL
    assert any(k == "throw_in" for _, k, why in r.restarts)


def test_goal():
    sc = Scene(8)
    s = sc.add(1, 10, 30.0, 1.0)
    sc.ball_line(0, f(1.0), (30.4, 1.0), (30.4, 1.0))
    sc.ball_line(f(1.0), f(2.2), (30.4, 1.0), (54.5, 1.0), z=0.5)     # into the net, between the posts
    sc.ball_line(f(2.2), f(8.0) - 1, (54.5, 1.0), (54.5, 1.0))
    r = sc.run()
    assert r.state[f(4.0)] == DEAD and rule_at(r, 4.0) == "goal"


def test_whistle_and_free_kick():
    sc = Scene(20, slow_from=6.0, slow_to=14.0)
    a, o = sc.add(2, 10, 0.0, 0.0), sc.add(3, 20, 3.0, 0.0)
    taker = sc.add(4, 10, 9.6, 5.0)
    mate = sc.add(5, 10, 25.0, 5.0)
    dribble(sc, a, 0, f(6.0))
    sc.move(a, f(6.0), f(6.5), 12.0, 0.0)
    sc.ball_line(f(6.0), f(9.0), (12.4, 0.0), (10.0, 5.0))             # ball fetched and placed
    sc.ball_line(f(9.0), f(14.0), (10.0, 5.0), (10.0, 5.0))            # still
    sc.ball_line(f(14.0), f(15.0), (10.0, 5.0), (24.6, 5.0))           # free kick
    dribble(sc, mate, f(15.0), f(20.0) - 1)
    for k, c in enumerate(sc.cols):                                    # opponents keep 9.15 m from the ball
        if c[1] == 20 and c[0] != 2001:
            near = np.hypot(c[2] - 10.0, c[3] - 5.0) < 10
            c[2][near] = 22.0
    r = sc.run()
    assert (r.state[f(10.0):f(13.5)] == DEAD).all()
    assert {rule_at(r, 11.0)} <= {"whistle", "whistle_slowdown"}
    assert r.state[f(17.0)] == CONTROL


def test_corner():
    sc = Scene(12)
    a = sc.add(2, 10, 40.0, 20.0)
    taker = sc.add(3, 10, 52.6, 34.2)
    mate = sc.add(4, 10, 45.0, 5.0)
    dribble(sc, a, 0, f(1.0), vx=2.0)
    sc.ball_line(f(1.0), f(1.6), (42.4, 20.0), (55.0, 22.0))           # over the goal line, wide of the post
    sc.ball_line(f(1.6), f(5.0), (55.0, 22.0), (52.3, 33.8))           # carried to the corner
    sc.ball_line(f(5.0), f(8.0), (52.3, 33.8), (52.3, 33.8))           # placed
    sc.ball_line(f(8.0), f(9.0), (52.3, 33.8), (45.4, 5.0), z=1.0)    # corner kick
    dribble(sc, mate, f(9.0), f(12.0) - 1, vx=-2.0)
    r = sc.run()
    assert r.state[f(3.0)] == DEAD
    assert any(k == "corner" for _, k, why in r.restarts)


def test_kick_off():
    sc = Scene(8)
    taker = sc.add(1, 10, -0.6, 0.0)
    mate = sc.add(2, 10, -12.0, 0.0)
    sc.ball_line(0, f(2.0), (0.0, 0.0), (0.0, 0.0))
    sc.ball_line(f(2.0), f(3.0), (0.0, 0.0), (-11.6, 0.0))
    dribble(sc, mate, f(3.0), f(8.0) - 1, vx=-2.0)
    r = sc.run()
    assert r.state[f(1.0)] == DEAD and rule_at(r, 1.0) == "before_kickoff"
    assert any(k == "kick_off" for _, k, why in r.restarts)
    assert r.state[f(5.0)] == CONTROL


def test_every_frame_has_rule_and_confidence():
    sc = Scene(6)
    a = sc.add(2, 10, 0.0, 0.0)
    sc.ball_line(0, f(1.0), (5.0, 0.0), (0.4, 0.0))
    dribble(sc, a, f(1.0), f(6.0) - 1)
    r = sc.run()
    assert (r.rule >= 0).all() and len(r.rule_names) == len(set(r.rule_names))
    assert ((r.confidence >= 0) & (r.confidence <= 1)).all()


# ---------------------------------------------------------------- C2 v2

def _whistle_scene():
    sc = Scene(20, slow_from=6.0, slow_to=14.0)
    a, o = sc.add(2, 10, 0.0, 0.0), sc.add(3, 20, 3.0, 0.0)
    sc.add(4, 10, 9.6, 5.0)
    mate = sc.add(5, 10, 25.0, 5.0)
    dribble(sc, a, 0, f(6.0))
    sc.move(a, f(6.0), f(6.5), 12.0, 0.0)
    sc.ball_line(f(6.0), f(9.0), (12.4, 0.0), (10.0, 5.0))
    sc.ball_line(f(9.0), f(14.0), (10.0, 5.0), (10.0, 5.0))
    return sc, mate


def test_v2_whistle_needs_a_restart():
    sc, mate = _whistle_scene()
    sc.ball_line(f(14.0), f(15.0), (10.0, 5.0), (24.6, 5.0))           # free kick
    dribble(sc, mate, f(15.0), f(20.0) - 1)
    for c in sc.cols:
        if c[1] == 20 and c[0] != 2001:
            near = np.hypot(c[2] - 10.0, c[3] - 5.0) < 10
            c[2][near] = 22.0
    r = sc.run(v2=True)
    assert (r.state[f(10.0):f(13.5)] == DEAD).all()
    assert r.state[f(17.0)] == CONTROL


def test_v2_slowdown_without_restart_stays_alive():
    sc = Scene(20, slow_from=6.0, slow_to=14.0)
    b, a = sc.add(1, 10, -10.0, 0.0), sc.add(2, 10, 5.0, 0.0)
    sc.ball_line(0, f(1.0), (-9.6, 0.0), (-9.6, 0.0))
    sc.ball_line(f(1.0), f(2.0), (-9.6, 0.0), (4.6, 0.0))
    dribble(sc, a, f(2.0), f(6.0))
    sc.ball_with(a, f(6.0), f(20.0) - 1)                               # A shields the ball while everybody slows down
    r1, r2 = sc.run(), sc.run(v2=True)                                  # no touch, so no T2 restart follows
    assert (r1.state[f(10.0):f(13.0)] == DEAD).any()                     # v1: the slowdown alone is a stoppage
    assert "whistle_slowdown" not in r2.rule_names and "whistle_restart" not in r2.rule_names


def test_v2_gap_keeps_the_team():
    sc = Scene(10)
    b = sc.add(1, 10, -10.0, 0.0)
    a = sc.add(2, 10, 5.0, 0.0)
    sc.ball_line(0, f(1.0), (-9.6, 0.0), (-9.6, 0.0))                  # placed ball, then a pass to A
    sc.ball_line(f(1.0), f(2.0), (-9.6, 0.0), (4.6, 0.0))
    dribble(sc, a, f(2.0), f(10.0) - 1)
    sc.bx[f(4.0):f(6.0)] = np.nan                                       # camera loses the ball for 2 s
    r = sc.run(v2=True)
    assert (r.state[f(4.2):f(5.8)] == CONTROL).all() and (r.team[f(4.2):f(5.8)] == 10).all()
    assert "gap_same_team" in {r.rule_names[k] for k in r.rule[f(4.2):f(5.8)]}
