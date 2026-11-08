"""Synthetic tests for the M1 input upgrades, the C3 decoder and the H / A / D scoring (labelled synthetic data only)."""
import numpy as np

from m1_labels.scoring import had, merge_had, score_had
from m1_possession import c3 as C3
from m1_possession.smoother import FLAG_MISSING, FLAG_OBSERVED
from m1_possession.upgrades import FLAG_GUESSED, fill_medium_gaps, gap_features, smooth_players

FPS = 30.0


def test_smooth_players_keeps_a_straight_run_and_removes_jitter():
    n = 300
    t = np.arange(n) / FPS
    true = 3.0 * t
    rng = np.random.default_rng(0)
    px = np.c_[true + rng.normal(0, 0.1, n), np.full(n, np.nan)].astype(np.float32)
    py = np.zeros((n, 2), np.float32)
    sx, sy = smooth_players(px, py, np.ones(n, int))
    inner = slice(20, n - 20)
    assert abs(np.mean(sx[inner, 0] - true[inner])) < 0.02                 # positions stay where they are
    assert np.std(np.diff(sx[inner, 0])) < np.std(np.diff(px[inner, 0])) / 3   # the jitter goes
    assert np.isnan(sx[:, 1]).all()


def test_fill_medium_gap_ground_roll_and_air_curve():
    n = 120
    t = np.arange(n) / FPS
    x = np.where(t < 1.01, 10 * t, np.nan)
    x[60:] = 10 + 6 * (t[60:] - t[60]) + 6 * (t[60] - 1)
    y = np.zeros(n); z = np.zeros(n)
    vx = np.full(n, 10.0); vy = np.zeros(n)
    fl = np.full(n, FLAG_OBSERVED, np.int8)
    fl[31:60] = FLAG_MISSING
    x[31:60] = np.nan
    fx, fy, fz, fvx, fvy, ffl, k = fill_medium_gaps(t, np.ones(n, int), x, y, z, vx, vy, fl)
    assert k == 29 and (ffl[31:60] == FLAG_GUESSED).all()
    assert np.all(np.diff(fx[30:61]) > 0)                                 # moves forward ...
    assert fx[45] - fx[44] < fx[32] - fx[31]                              # ... slowing down
    z2 = z.copy(); z2[30] = 2.0; z2[60] = 2.0
    _, _, gz, _, _, _, _ = fill_medium_gaps(t, np.ones(n, int), x, y, z2, vx, vy, fl)
    assert gz[45] > 2.0                                                   # a ballistic curve over both ends


def test_gap_features_describe_a_long_gap():
    n = 400
    t = np.arange(n) / FPS
    miss = np.zeros(n, bool); miss[100:300] = True
    x = np.zeros(n); y = np.zeros(n); x[99], x[300] = 5.0, 5.5
    px = np.array([[5.0, -20.0]] * n, np.float32); py = np.zeros((n, 2), np.float32)
    f = gap_features(t, np.ones(n, int), miss, x, y, px, py, np.array([1.0, -1.0]))
    assert f["gap_in"][150] == 1 and f["gap_in"][50] == 0
    assert abs(f["gap_len_s"][150] - 199 / FPS) < 1e-4 and abs(f["gap_dist_m"][150] - 0.5) < 1e-5
    assert f["gap_team_vanish"][150] == 1 and f["gap_same_team"][150] == 1


def _const_emissions(T, state):
    E = np.full((T, 5), np.log(0.05))
    E[np.arange(T), state] = np.log(0.8)
    return E


def test_viterbi_dead_ends_only_at_a_touch():
    T = 40
    want = np.r_[np.zeros(10, int), np.full(30, C3.S_CH)]
    path = C3.viterbi(_const_emissions(T, want), np.zeros(T, bool), np.ones(T, int), C3.HMM_DEFAULT)
    assert (path == C3.S_DEAD).all()                                        # no touch: DEAD cannot end
    ok = np.zeros(T, bool); ok[20] = True
    path = C3.viterbi(_const_emissions(T, want), ok, np.ones(T, int), C3.HMM_DEFAULT)
    assert (path[:20] == C3.S_DEAD).all() and (path[20:] == C3.S_CH).all()  # it ends at the touch


def test_viterbi_no_direct_team_switch():
    T = 40
    want = np.r_[np.full(20, C3.S_CH), np.full(20, C3.S_CA)]
    path = C3.viterbi(_const_emissions(T, want), np.ones(T, bool), np.ones(T, int), C3.HMM_DEFAULT)
    sw = np.flatnonzero(path[1:] != path[:-1])
    assert all(not ({path[i], path[i + 1]} == {C3.S_CH, C3.S_CA}) for i in sw)


def test_had_mapping_and_spell_f1():
    st = np.array([1, 1, 2, 3, 0, 0, 1, 1, 3, 3])
    tm = np.array([7, 7, 7, -1, -1, -1, 8, 8, -1, -1])
    per = np.ones(10, int)
    assert had(st, tm, per, 7).tolist() == [0, 0, 0, 0, 2, 2, 1, 1, 1, 1]
    lab = dict(state=st, team_id=tm, score_mask=np.ones(10, bool), period=per)
    s = score_had(lab, st, tm, 7)
    m = merge_had([s])
    assert m["macro_f1"] == 1.0 and m["spell_f1"] == 1.0
    pred_t = tm.copy(); pred_t[6:8] = 7                                      # the second spell given to the wrong team
    m2 = merge_had([score_had(lab, st, pred_t, 7)])
    assert m2["macro_f1"] < 1.0 and m2["spell_f1"] < 1.0
