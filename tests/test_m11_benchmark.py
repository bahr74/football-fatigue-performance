"""M11 Stage 4 benchmark tests (synthetic data only): folds, metrics, input conversion, feature and model symmetry, physics sign."""

import numpy as np
import pytest

from football_fatigue.m11_benchmark import data as D
from football_fatigue.m11_benchmark import folds as FO
from football_fatigue.m11_benchmark import gates as G
from football_fatigue.m11_benchmark import metrics as MT
from football_fatigue.m11_benchmark import models as MD
from football_fatigue.window_intersections import samples as S


def synth(n=60, seed=0, gap=0.8):
    """Two players with random constant velocities; the label follows the Stage 3 builder (random orders)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        va, vb = rng.normal(0, 3, 2), rng.normal(0, 3, 2)
        pa, pb = rng.normal(0, 1, 2), rng.normal(0, 1, 2)
        V = lambda p, v: (np.arange(0, 20, 1 / 15), p[0] + v[0] * (np.arange(0, 20, 1 / 15) - 10), p[1] + v[1] * (np.arange(0, 20, 1 / 15) - 10))  # noqa: E731
        s = S.build_one(V(pa, va), V(pb, vb), 10.0, gap, int(rng.integers(2)), int(rng.integers(2)))
        rows.append(S.model_arrays(s, gap))
    A = S.stack(rows)
    return {k: A[k] for k in S.INPUT_FIELDS}, A["y"].astype(int)


def test_folds_are_deterministic_disjoint_and_balanced():
    ms = [str(m) for m in range(100, 132)]
    f = FO.outer_folds(ms)
    assert f == FO.outer_folds(list(reversed(ms))) and np.bincount(list(f.values())).tolist() == [4] * 8
    for k in range(8):
        tr = [m for m, v in f.items() if v != k]
        inner = FO.inner_folds(tr, k)
        assert set(inner) == set(tr) and sorted(np.bincount(list(inner.values()))) == [9, 9, 10]
        es = FO.early_stop_matches(tr, k)
        assert len(es) == 3 and set(es) <= set(tr) and not set(es) & {m for m, v in f.items() if v == k}


def test_metrics_positive_class_is_swapped():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.6, 0.4, 0.2])
    r = MT.basic(y, p)
    assert r["balanced_accuracy"] == 1.0 and r["confusion"] == dict(tn=2, fp=0, fn=0, tp=2) and r["roc_auc"] == 1.0
    r2 = MT.basic(y, 1 - p)
    assert r2["balanced_accuracy"] == 0.0 and r2["recall_swapped"] == 0.0
    assert MT.ece(np.array([1, 0] * 50), np.full(100, 0.5)) == pytest.approx(0.0)
    g = np.repeat(np.arange(10), 10)
    yy = np.tile([0, 1], 50)
    pp = np.where(yy == 1, 0.7, 0.3)
    b = MT.bootstrap(yy, pp, g, n_boot=200)
    assert b["balanced_accuracy_ci"] == [1.0, 1.0]
    pr = MT.paired(yy, pp, np.full(100, 0.5), g, n_boot=200)
    assert pr["balanced_accuracy_diff"] == 0.5 and pr["ba_significant"] and pr["log_loss_diff"] < 0


def test_coverage_curve_prefers_confident_correct():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.99, 0.01, 0.4, 0.6])                    # two confident correct, two wrong at low confidence
    c = MT.coverage_curve(y, p, levels=(1.0, 0.5))
    assert c["curve"]["0.5"]["error"] == 0.0 and c["curve"]["1.0"]["error"] == 0.5 and c["coverage_at_error"]["error_01"]["coverage"] == 0.5


def test_to_sample_masks_times_and_no_label():
    x, y = synth(4)
    x["in_mask"][0, 0, :5] = False
    s = D.to_sample(x, 0)
    assert np.isnan(s.before[0, :5]).all() and np.isfinite(s.before[0, 5:]).all()
    assert np.allclose(s.t_before, -0.4 + S.IN_REL) and np.allclose(s.t_after, 0.4 + S.OUT_REL) and s.y == 0


def test_feature_swap_equals_recomputation():
    x, y = synth(30, 1)
    F, M, names = D.features(x)
    k = D.n_perm(names)
    xo = dict(x, out_pos=x["out_pos"][:, ::-1].copy(), out_mask=x["out_mask"][:, ::-1].copy())
    Fo, Mo, _ = D.features(xo)
    assert np.allclose(Fo, D.swap_features(F, k), atol=1e-9) and np.allclose(Mo, -M, atol=1e-9)
    assert not any(w in n for n in names for w in S.FORBIDDEN_INPUT_WORDS)


def test_physics_baselines_recover_straight_motion_and_are_symmetric():
    x, y = synth(80, 2)
    _, M, _ = D.features(x)
    from football_fatigue.m11_recovery.motion import BASELINES
    k = BASELINES.index("cv_bidir")
    xb = MD.physics_x(M, x["gap_s"].astype(float), k)
    w = MD.fit_scale(xb, y)
    p = MD.physics_predict(xb, w)
    assert w > 0 and MT.basic(y, p)["accuracy"] == 1.0                  # constant velocity is exact for straight motion
    assert np.allclose(MD.physics_predict(-xb, w), 1 - p)


@pytest.mark.parametrize("name", ["lr", "knn", "hgb", "rf", "svm"])
def test_tabular_models_are_exactly_symmetric(name):
    x, y = synth(120, 3)
    F, _, names = D.features(x)
    k = D.n_perm(names)
    cfg = MD.TABULAR_GRIDS[name][0]
    m = MD.Tabular(name, cfg, k, 0).fit(F[:80], y[:80])
    if name == "svm":
        m.platt = MD.fit_platt(m.raw(F[:80]), y[:80])
    p = m.predict(F[80:])
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    assert np.allclose(m.predict(D.swap_features(F[80:], k)), 1 - p, atol=1e-12)


@pytest.mark.parametrize("kind", ["mlp", "gru", "lstm", "gru_gaplast", "lstm_gaplast"])
def test_neural_pair_logit_is_antisymmetric_and_small(kind):
    torch = pytest.importorskip("torch")
    x, y = synth(16, 4)
    si, so = D.seq_tensor(x)
    torch.manual_seed(0)
    net = make = MD.make_net(kind, **MD.NEURAL[kind])
    net.eval()
    f = lambda a: torch.as_tensor(np.ascontiguousarray(a))  # noqa: E731
    g = f(x["gap_s"])
    z = net(f(si), f(so), g)
    assert torch.allclose(net(f(si), f(so[:, ::-1]), g), -z, atol=1e-5)
    assert torch.allclose(net(f(si[:, ::-1]), f(so), g), -z, atol=1e-5)
    assert 15_000 <= MD.n_params(make) <= 60_000


def test_seq_tensor_masks_and_reversal():
    x, y = synth(3, 5)
    x["in_mask"][0, 1, :4] = False
    si, so = D.seq_tensor(x)
    assert si.shape == (3, 2, 30, 4) and so.shape == (3, 2, 30, 4)
    assert (si[0, 1, -4:] == 0).all()                                    # reversed: far (masked) points at the end, all zero
    assert np.allclose(si[0, 0, 0, :2], x["in_pos"][0, 0, -1] / D.POS_SCALE_M) and si[0, 0, 0, 3] == pytest.approx(-1 / 15)


def test_neural_training_learns_straight_motion():
    pytest.importorskip("torch")
    x, y = synth(400, 6)
    si, so = D.seq_tensor(x)
    g = x["gap_s"]
    m = MD.Neural("mlp", seed=0, epochs=40, patience=40, batch=64)
    m.fit(si[:300], so[:300], g[:300], y[:300], si[300:], so[300:], g[300:], y[300:])
    p = m.predict(si[300:], so[300:], g[300:])
    assert MT.basic(y[300:], p)["accuracy"] > 0.8


def test_kabsch_recovers_rigid_motion():
    rng = np.random.default_rng(0)
    P = rng.normal(0, 3, (40, 2))
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    Q = P @ R.T + np.array([5.0, -2.0])
    R2, t, rms = G._kabsch(P, Q)
    assert rms < 1e-9 and np.allclose(R2, R)


def test_texture_transforms_keep_masks_and_are_deterministic():
    from football_fatigue.m11_benchmark import robust as RB
    x, y = synth(5, 7)
    x["in_mask"][0, 0, :6] = False
    x["in_pos"][0, 0, :6] = 0
    sm = RB.smooth(x["in_pos"], x["in_mask"])
    assert (sm[~x["in_mask"]] == 0).all() and np.isfinite(sm).all()
    # a straight constant-velocity track is unchanged by the centred average away from the ends
    assert np.allclose(sm[1, 0, 2:-2], x["in_pos"][1, 0, 2:-2], atol=1e-4)
    n1, n2 = RB.perturb(x["in_pos"], x["in_mask"], 0.05, 1), RB.perturb(x["in_pos"], x["in_mask"], 0.05, 1)
    assert np.array_equal(n1, n2) and (n1[~x["in_mask"]] == 0).all()
    d = (n1 - x["in_pos"])[x["in_mask"]]
    assert 0.02 < d.std() < 0.08
