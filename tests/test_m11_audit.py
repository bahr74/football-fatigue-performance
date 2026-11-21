"""Automated leakage and validity tests for the M11 benchmark (synthetic unless marked ``data``)."""

from __future__ import annotations

import numpy as np
import pytest

from football_fatigue.m11_audit import validity as V
from football_fatigue.m11_phase2 import common as C
from football_fatigue.window_intersections import samples as S

data = pytest.mark.skipif(not (C.PROC / "m11-stage3-samples-v1" / "matches").exists(), reason="development samples not present")


def _verts(x0, vx, y0=0.0, vy=0.0, t0=0.0, t1=6.0):
    t = np.arange(t0, t1, 1 / 15)
    return t, x0 + vx * (t - t0), y0 + vy * (t - t0)


def test_hidden_gap_vertices_never_change_model_inputs():
    VA, VB = _verts(-3, 1.5), _verts(3, -1.5, 0.4)
    a = S.build_one(VA, VB, 3.0, 0.8, 0, 1)
    g0, g1 = 3.0 - 0.4, 3.0 + 0.4
    pert = []
    for t, x, y in (VA, VB):
        k = (t >= g0) & (t <= g1)
        x2, y2 = x.copy(), y.copy()
        x2[k] += 7.0
        y2[k] -= 4.0
        pert.append((t, x2, y2))
    b = S.build_one(*pert, 3.0, 0.8, 0, 1)
    for key in ("in_pos", "out_pos", "in_mask", "out_mask"):
        assert np.array_equal(a[key], b[key]), key


def test_label_is_only_the_order_bits_and_frame_ignores_outgoing():
    VA, VB = _verts(-3, 1.5), _verts(3, -1.5, 0.4)
    for io in ((0, 0), (0, 1), (1, 0), (1, 1)):
        s = S.build_one(VA, VB, 3.0, 0.8, *io)
        assert s["y"] == (io[0] ^ io[1])
    s0 = S.build_one(VA, VB, 3.0, 0.8, 0, 0)
    VB2 = (VB[0], np.where(VB[0] > 3.4, VB[1] + 5, VB[1]), VB[2])          # change only the outgoing part of player B
    s1 = S.build_one(VA, VB2, 3.0, 0.8, 0, 0)
    assert np.allclose(s0["origin"], s1["origin"]) and np.allclose(s0["R"], s1["R"])


def test_model_features_do_not_depend_on_the_label():
    from football_fatigue.m11_benchmark import data as D
    VA, VB = _verts(-3, 1.5), _verts(3, -1.5, 0.4)
    rows = [S.model_arrays(S.build_one(VA, VB, 3.0, 0.8, 0, k), 0.8) for k in (0, 1)]
    x = S.stack(rows)
    F1, _, _ = D.features(x)
    x2 = dict(x, y=1 - x["y"])
    F2, _, _ = D.features(x2)
    assert np.array_equal(F1, F2)
    a1, b1 = D.seq_tensor(x)
    a2, b2 = D.seq_tensor(x2)
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2)


def test_roughness_measures_texture_not_motion():
    t = np.arange(30) / 15
    pos = np.zeros((2, 2, 30, 2))
    pos[0, :, :, 0] = 3 * t                                                   # straight constant speed: no texture
    pos[1, :, :, 0] = 3 * t + np.random.default_rng(0).normal(0, 0.05, (2, 30))
    r = V.roughness(pos, np.ones((2, 2, 30), bool))
    assert r[0, :, 0].max() < 1e-9 and r[1, :, 0].min() > 0.02


def test_match_bootstrap_paired_difference_of_identical_predictions_is_zero():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = rng.random(500)
    g = np.repeat(np.arange(10), 50)
    r = V.match_boot(y, p, g, n_boot=100, q=p)
    assert r["value"] == 0 and r["ci95"] == [0, 0]
    assert abs(V.match_boot(y, (y == 1).astype(float), g, n_boot=50)["value"] - 1.0) < 1e-12


@data
def test_saved_benchmark_folds_and_labels_are_valid():
    L = V.load()
    fi = V.fold_integrity(L)
    assert fi["passed"], fi
    ls = V.label_structure(L)
    assert ls["y_equals_order_xor"] and 0.45 < ls["swapped_share"] < 0.55


@data
def test_structural_inputs_carry_no_label_signal():
    from football_fatigue.m11_benchmark import data as D
    L = V.load()
    x, _y, _g = D.load_inputs(V.TABLE)
    r = V.shortcut_probes(L, x)
    assert r["structural_probe_hgb"]["ci95"][1] < 0.53                     # valid-point counts / frame mode: chance only
    assert r["texture_similarity_rule"]["both_fully_estimated"]["value"] > r["texture_similarity_rule"]["both_fully_visible"]["value"]


@data
def test_heldout_metadata_is_refused():
    from football_fatigue.m11_benchmark import data as D
    from football_fatigue.window_intersections.split import load_split
    from football_fatigue.window_intersections import samples as SM
    sp = load_split(D.proc_root(), SM.SPLIT)
    assert sp["heldout"] and not set(sp["heldout"]) & set(C.allowed())
    with pytest.raises((Exception, SystemExit)):
        D.load_meta(V.TABLE, [sp["heldout"][0]])
    with pytest.raises(SystemExit, match="refused"):
        C.checked([sp["heldout"][0]])
