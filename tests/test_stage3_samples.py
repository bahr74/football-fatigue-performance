"""M11 Stage 3 tests: supervised MATCHED/SWAPPED sample construction (temporal grid, gap boundaries, frame, labels, ordering,
leakage protection, held-out refusal, determinism). Synthetic data only; no data files are read."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from football_fatigue.encounters import quality as Q
from football_fatigue.encounters.trajectories import STATE_AVAILABLE
from football_fatigue.window_intersections import candidates as C
from football_fatigue.window_intersections import core as W
from football_fatigue.window_intersections import local as L
from football_fatigue.window_intersections import samples as S

FPS = 15
N = 301
TT = np.arange(N) / FPS
CFG, WCFG = C.CandidateConfig(), W.WindowConfig()


def world(tracks, drop=None, est=None):
    """Synthetic match on a 15 Hz grid (same construction as the Stage 2 tests)."""
    ns = max(tracks) + 1
    cols = {k: [] for k in ("slot", "row0", "row1", "t0", "t1", "x0", "y0", "x1", "y1")}
    x, y = np.full((N, ns), np.nan), np.full((N, ns), np.nan)
    for s in sorted(tracks):
        xs, ys = tracks[s]
        x[:, s], y[:, s] = xs, ys
        for i in range(N - 1):
            if drop is not None and drop(s, i):
                continue
            for k, v in zip(cols, (s, i, i + 1, TT[i], TT[i + 1], xs[i], ys[i], xs[i + 1], ys[i + 1])):
                cols[k].append(v)
    seg = {k: np.array(v) for k, v in cols.items()}
    seg["slot"], seg["row0"], seg["row1"] = (seg[k].astype(int) for k in ("slot", "row0", "row1"))
    seg["period"] = np.ones(len(seg["slot"]), int)
    seg["stationary"] = (seg["x0"] == seg["x1"]) & (seg["y0"] == seg["y1"])
    seg = W.finalize_segments(seg)
    mask = np.zeros((N, ns), np.int64)
    if est is not None:
        for s in range(ns):
            mask[est(s), s] |= Q.ESTIMATED
    c = SimpleNamespace(match_id=1, fps=float(FPS), canonical_period=np.ones(N, np.int8), video_time_s=TT.copy(),
                        state=np.full((N, ns), STATE_AVAILABLE, np.int8), slot_side=np.array(["home"] * ns), x=x, y=y)
    return c, mask, np.zeros((N, ns), bool), seg, W.chain_table(seg)


def crossing(dt=0.1, va=2.0, vb=2.0, tc=10.0):
    return {0: (va * (TT - tc), 0 * TT), 1: (0 * TT, vb * (TT - tc - dt))}


def build(w, cand=None):
    c, mask, dup, seg, ch = w
    if cand is None:
        cand = C.match_candidates(c, mask, dup, seg, ch, CFG, WCFG)
    cand = cand.assign(split="train_cv")
    return cand, S.build_match(c, mask, dup, seg, ch, cand, CFG, perturb_seed=1)


def lin(t0, t1, v, p0=(0.0, 0.0), n=None, times=None):
    """Vertices of a straight constant-velocity track."""
    t = np.asarray(times if times is not None else np.arange(t0, t1 + 1e-9, 1 / FPS))
    return t, p0[0] + v[0] * (t - t0), p0[1] + v[1] * (t - t0)


# ---------------------------------------------------------------- temporal grid and gap boundaries

def test_grid_convention_exact_timestamps():
    assert len(S.IN_REL) == len(S.OUT_REL) == S.T == 30
    assert S.IN_REL[0] == pytest.approx(-2.0, abs=1e-12) and S.IN_REL[-1] == pytest.approx(-1 / 15, abs=1e-12)
    assert S.OUT_REL[0] == pytest.approx(1 / 15, abs=1e-12) and S.OUT_REL[-1] == pytest.approx(2.0, abs=1e-12)
    assert np.allclose(np.diff(S.IN_REL), 1 / 15) and np.allclose(np.diff(S.OUT_REL), 1 / 15)
    assert (S.IN_REL < 0).all() and (S.OUT_REL > 0).all()
    s = S.build_one(lin(0, 20, (1, 0)), lin(0, 20, (0, 1), (5, 0)), 10.0, 0.8, 0, 0)
    assert (s["grid_in"] < s["g0"]).all() and (s["grid_out"] > s["g1"]).all()
    assert s["grid_in"][-1] == pytest.approx(9.6 - 1 / 15) and s["grid_out"][0] == pytest.approx(10.4 + 1 / 15)


def test_off_by_one_on_aligned_vertices():
    # vertices exactly on the 15 Hz grid, g0 itself a vertex: the last incoming point is the vertex one step before g0
    V = lin(0, 20, (3.0, 0.0))
    s = S.build_one(V, lin(0, 20, (0, 1), (5, 0)), 10.0, 0.8, 0, 0)
    assert s["in_mask"].all() and s["out_mask"].all()
    assert s["pitch_in"][0, -1, 0] == pytest.approx(3.0 * (9.6 - 1 / 15), abs=1e-9)
    assert s["pitch_in"][0, 0, 0] == pytest.approx(3.0 * 7.6, abs=1e-9)
    assert s["pitch_out"][0, 0, 0] == pytest.approx(3.0 * (10.4 + 1 / 15), abs=1e-9)
    assert s["pitch_out"][0, -1, 0] == pytest.approx(3.0 * 12.4, abs=1e-9)


def test_vertices_on_the_gap_boundary_are_excluded():
    # a vertex exactly at g0 and g1 belongs to the hidden gap; moving it must not change any input
    t = np.array([9.0, 9.3, 9.6, 10.0, 10.4, 10.7, 11.0, 13.0, 7.0, 8.0])
    t.sort()
    x = t.copy()
    pos, ok = S.tracklet(t, x, 0 * x, 9.6 + S.IN_REL, "in", 9.6, 10.4)
    x2 = np.where(np.isclose(t, 9.6) | np.isclose(t, 10.4), 1e3, x)
    pos2, ok2 = S.tracklet(t, x2, 0 * x, 9.6 + S.IN_REL, "in", 9.6, 10.4)
    assert np.array_equal(pos, pos2) and np.array_equal(ok, ok2)
    # the last grid point (9.533) lies after the last strictly-pre-gap vertex (9.3)? no: 9.3 < 9.533 -> masked (no extrapolation)
    assert not ok[-1] and ok[S.IN_REL + 9.6 <= 9.3 + 1e-12].all()
    po, oo = S.tracklet(t, x2, 0 * x, 10.4 + S.OUT_REL, "out", 9.6, 10.4)
    assert not oo[0] and oo[(10.4 + S.OUT_REL >= 10.7 - 1e-12) & (10.4 + S.OUT_REL <= 13.0)].all()


def test_nonuniform_native_timestamps_and_mask_alignment():
    rng = np.random.default_rng(3)
    t = np.cumsum(rng.uniform(0.03, 0.11, 400))
    x, y = np.sin(t), np.cos(0.5 * t)
    g0, g1 = 12.0, 12.8
    gin = g0 + S.IN_REL
    pos, ok = S.tracklet(t, x, y, gin, "in", g0, g1)
    k = t < g0
    exp_ok = (gin >= t[k][0]) & (gin <= t[k][-1])
    assert np.array_equal(ok, exp_ok)
    assert np.allclose(pos[ok, 0], np.interp(gin[ok], t[k], x[k])) and np.allclose(pos[ok, 1], np.interp(gin[ok], t[k], y[k]))
    assert (pos[~ok] == 0).all()


def test_hidden_gap_positions_never_enter_the_input():
    rng = np.random.default_rng(5)
    VA, VB = lin(0, 20, (2, 0), (-20, 0)), lin(0, 20, (0, 2), (0, -20))
    base = S.build_one(VA, VB, 10.0, 0.8, 1, 0)
    for _ in range(5):
        def pert(V):
            t, x, y = V
            inside = (t >= 9.6) & (t <= 10.4)
            return t, np.where(inside, rng.normal(0, 50, len(t)), x), np.where(inside, rng.normal(0, 50, len(t)), y)
        s = S.build_one(pert(VA), pert(VB), 10.0, 0.8, 1, 0)
        for k in ("in_pos", "out_pos", "in_mask", "out_mask"):
            assert np.array_equal(base[k], s[k])
        assert S.gap_perturbation_ok(VA, VB, base, 10.0, 0.8, (1, 0))


def test_missing_chain_coverage_is_masked_never_bridged():
    # chain covers only [8.5, 11.0]: incoming points before 8.5 and outgoing points after 11.0 are masked
    V = lin(8.5, 11.0, (1, 0))
    s = S.build_one(V, lin(0, 20, (0, 1), (5, 0)), 10.0, 0.8, 0, 0)
    assert (s["in_mask"][0] == (s["grid_in"] >= V[0][0])).all()
    assert (s["out_mask"][0] == (s["grid_out"] <= V[0][-1])).all() and not s["out_mask"][0][-1]
    assert (s["in_pos"][0][~s["in_mask"][0]] == 0).all() and (s["out_pos"][0][~s["out_mask"][0]] == 0).all()
    # fewer than MIN_POINTS observed points -> construction failure, never a sample
    short = S.build_one(lin(9.45, 11, (1, 0)), lin(0, 20, (0, 1), (5, 0)), 10.0, 0.8, 0, 0)
    assert short["failure"] == "too_few_valid_points"


def test_no_interpolation_across_a_chain_break():
    # player 0's chain breaks at frame 170 (t = 11.33 s); a second chain of the same slot starts after the break
    w = world(crossing(0.1), drop=lambda s, i: s == 0 and i == 170)
    c, mask, dup, seg, ch = w
    k0 = int(np.flatnonzero((ch["slot"] == 0) & (ch["t0"] <= 10.0) & (ch["t1"] >= 10.0))[0])
    t, x, y, rows = S.chain_vertices(seg, ch, k0)
    assert t[-1] == pytest.approx(170 / FPS) and np.array_equal(rows, np.arange(rows[0], rows[0] + len(rows)))
    pos, ok = S.tracklet(t, x, y, 10.4 + S.OUT_REL, "out", 9.6, 10.4)
    assert not ok[10.4 + S.OUT_REL > 170 / FPS + 1e-12].any()


# ---------------------------------------------------------------- coordinate frame

def _frame_of(VA, VB):
    s = S.build_one(VA, VB, 10.0, 0.8, 0, 0)
    return s, S.sample_frame(s["pitch_in"], s["in_mask"], S.IN_REL)


def test_frame_modes_and_order_invariance():
    cases = {0: (lin(0, 20, (2, 1)), lin(0, 20, (1, 2), (3, 0))),          # common motion
             1: (lin(0, 20, (2, 0)), lin(0, 20, (-2, 0), (0, 1))),         # opposing velocities: sum = 0
             2: (lin(0, 20, (0, 0), (4, 4)), lin(0, 20, (0.05, 0), (5, 4)))}  # nearly stationary
    for mode, (VA, VB) in cases.items():
        s, (o, R, m) = _frame_of(VA, VB)
        assert m == mode
        assert np.allclose(R @ R.T, np.eye(2)) and np.isclose(np.linalg.det(R), 1.0)
        s2 = S.build_one(VA, VB, 10.0, 0.8, 1, 0)                          # incoming order swapped
        o2, R2, m2 = S.sample_frame(s2["pitch_in"], s2["in_mask"], S.IN_REL)
        assert m2 == m and np.allclose(o, o2) and np.allclose(R, R2)
    # mode 0: the summed velocity maps onto +x
    s, (o, R, _) = _frame_of(*cases[0])
    v = R @ np.array([3.0, 3.0])
    assert v[0] > 0 and abs(v[1]) < 1e-9
    # mode 1: the axis points to +x of the pitch (sign fixed), whatever the order
    _, (o, R, _) = _frame_of(*cases[1])
    assert (R @ np.array([1.0, 0.0]))[0] > 0
    # mode 2: pitch axes
    _, (o, R, _) = _frame_of(*cases[2])
    assert np.allclose(R, np.eye(2))


def test_frame_degenerate_inputs_are_finite():
    # identical incoming positions, zero velocity, tiny velocities, far-from-origin coordinates
    for VA, VB in ((lin(0, 20, (0, 0), (1, 1)), lin(0, 20, (0, 0), (1, 1))),
                   (lin(0, 20, (1e-9, 0)), lin(0, 20, (-1e-9, 1e-9))),
                   (lin(0, 20, (0.149, 0), (1e4, -1e4)), lin(0, 20, (0.149, 0.0), (1e4, -1e4 + 0.2)))):
        s = S.build_one(VA, VB, 10.0, 0.8, 0, 1)
        assert np.isfinite(s["in_pos"]).all() and np.isfinite(s["out_pos"]).all() and np.isfinite(s["R"]).all()
        back = s["in_pos"][0][s["in_mask"][0]].astype(np.float64) @ s["R"] + s["origin"]
        assert np.abs(back - s["pitch_in"][0][s["in_mask"][0]]).max() < 1e-3
        assert np.abs(s["in_pos"]).max() < 100                              # frame removes the absolute pitch position


def test_frame_uses_incoming_only():
    VA, VB = lin(0, 20, (2, 1)), lin(0, 20, (1, 2), (3, 0))
    s = S.build_one(VA, VB, 10.0, 0.8, 0, 0)
    t, x, y = VA
    after = t > 10.4
    s2 = S.build_one((t, np.where(after, x + 7 * np.sin(t), x), np.where(after, y - 3, y)), VB, 10.0, 0.8, 0, 0)
    assert np.allclose(s["origin"], s2["origin"]) and np.allclose(s["R"], s2["R"]) and np.array_equal(s["in_pos"], s2["in_pos"])


# ---------------------------------------------------------------- labels and ordering

def test_label_semantics_with_known_identities():
    VA, VB = lin(0, 20, (2, 0), (-20, 0)), lin(0, 20, (0, 2), (0, -20))   # A along +x, B along +y
    for io, oo in ((0, 0), (0, 1), (1, 0), (1, 1)):
        s = S.build_one(VA, VB, 10.0, 0.8, io, oo)
        a_in, a_out = (1 if io else 0), (1 if oo else 0)                      # index of A among incoming / outgoing
        assert np.allclose(s["pitch_in"][a_in][:, 1], 0) and np.allclose(s["pitch_out"][a_out][:, 1], 0)  # A has y = 0
        assert s["y"] == (S.MATCHED if a_in == a_out else S.SWAPPED)
        raw = {"A": VA, "B": VB}
        assert S.verify_label(s, raw) == "ok"
        bad = dict(s, y=1 - s["y"])
        assert S.verify_label(bad, raw) == "mismatch"


def test_swap_helpers_flip_and_preserve_labels():
    VA, VB = lin(0, 20, (2, 0.5), (-20, 0)), lin(0, 20, (0.3, 2), (0, -20))
    s = S.build_one(VA, VB, 10.0, 0.8, 0, 0)
    x = S.model_arrays(s, 0.8)
    y = int(x["y"])
    xo, yo = S.swap_outgoing(x, y)
    assert yo == 1 - y and np.array_equal(xo["out_pos"][0], x["out_pos"][1])
    xb, yb = S.swap_incoming(*S.swap_outgoing(x, y))
    assert yb == y
    # swapping the incoming order equals rebuilding with the incoming order bit flipped (frame is order-invariant)
    xi, yi = S.swap_incoming(x, y)
    r = S.model_arrays(S.build_one(VA, VB, 10.0, 0.8, 1, 0), 0.8)
    assert yi == int(r["y"]) and np.allclose(xi["in_pos"], r["in_pos"], atol=1e-5) and np.allclose(xi["out_pos"], r["out_pos"], atol=1e-5)
    # batched arrays too
    bx = {k: np.stack([x[k], x[k]]) for k in x}
    bo, _ = S.swap_outgoing(bx, np.array([y, y]))
    assert np.array_equal(bo["out_pos"][:, 0], bx["out_pos"][:, 1])


def test_order_bits_deterministic_balanced_and_independent():
    ids = [hashlib.sha256(str(i).encode()).hexdigest()[:16] for i in range(4000)]
    bits = np.array([S.order_bits(i, "g0p80") for i in ids])
    assert all(S.order_bits(i, "g0p80") == tuple(b) for i, b in zip(ids[:50], bits[:50]))
    y = bits[:, 0] ^ bits[:, 1]
    for v in (bits[:, 0], bits[:, 1], y):
        assert abs(v.mean() - 0.5) < 3 * 0.5 / np.sqrt(len(ids))
    assert abs(np.corrcoef(bits[:, 0], bits[:, 1])[0, 1]) < 0.05
    # depends on the table, so the three gap tables are not the same draw
    assert np.mean([S.order_bits(i, "g0p80") != S.order_bits(i, "g0p40") for i in ids]) > 0.6
    assert {S.shift_for(i) for i in ids} == set(S.SHIFTS_S)


def test_recovery_label_conversion():
    from football_fatigue.m11_recovery import views as V
    pos = np.stack([np.c_[V.GRID, 0 * V.GRID], np.c_[0 * V.GRID, V.GRID]])
    it = V.Interaction("1", "e", pos, "clean", [], {})
    for draw in range(8):
        s = V.make_sample(it, 0.8, 0.0, draw=draw)
        matched = np.allclose(s.before[0][:, 1], 0) == np.allclose(s.after[0][:, 1], 0)
        assert s.y == int(matched)                                  # recovery: 1 = matched
        assert S.recovery_y_to_stage3(s.y) == (S.MATCHED if matched else S.SWAPPED)
        assert S.stage3_y_to_recovery(S.recovery_y_to_stage3(s.y)) == s.y


# ---------------------------------------------------------------- end to end on a synthetic match

def _scene():
    rng = np.random.default_rng(11)
    tr = {**crossing(0.1), 2: (0 * TT + 5, 1.5 * (TT - 14)), 3: (0.7 * (TT - 14) + 5, 0 * TT + 0.2)}
    tr = {k: (x + rng.normal(0, 0.01, N), y + rng.normal(0, 0.01, N)) for k, (x, y) in tr.items()}
    return tr


def test_build_match_end_to_end_labels_checks_and_metadata():
    cand, (A, M, F, lower, summ) = build(world(_scene()))
    assert (cand["g0p80_outcome"] == C.ACCEPTED).sum() >= 2
    for t in S.TABLES:
        assert summ[t]["arrays"]["all_pass"], summ[t]["arrays"]
        assert summ[t]["built"] + summ[t]["failed"] == summ[t]["accepted_candidates"]
        assert summ[t]["checks"].get("label_ok", 0) == summ[t]["built"]
        assert set(summ[t]["checks"]) <= {"label_ok", "gap_boundary_ok", "inverse_transform_ok", "perturbation_tested", "perturbation_identical"}
        assert len(M[t]) == len(A[t]["y"]) and (M[t]["y"].to_numpy() == A[t]["y"]).all()
    m = M["g0p80"]
    assert not {"slot_a", "slot_b", "chain_a", "chain_b"} & set(m.columns)          # no slot identities even in metadata
    assert m.groupby("overlap_group")["match_id"].nunique().max() == 1
    assert set(A) == set(S.TABLES) and all(set(A[t]) == set(S.ARRAY_FIELDS) for t in A)


def test_input_schema_has_only_approved_fields():
    assert S.INPUT_FIELDS == ("in_pos", "in_mask", "in_dt", "out_pos", "out_mask", "out_dt", "gap_s")
    for f in S.INPUT_FIELDS:
        assert not any(w in f for w in S.FORBIDDEN_INPUT_WORDS), f
    s = S.build_one(lin(0, 20, (2, 0)), lin(0, 20, (0, 2), (1, -1)), 10.0, 0.8, 0, 0)
    assert set(S.model_arrays(s, 0.8)) == set(S.INPUT_FIELDS) | {"y"}


def test_tracking_status_never_changes_the_inputs():
    tr = _scene()
    _, (A0, *_rest) = build(world(tr))
    _, (A1, *_rest) = build(world(tr, est=lambda s: np.ones(N, bool)))
    rng = np.random.default_rng(2)
    _, (A2, *_rest) = build(world(tr, est=lambda s: rng.random(N) < 0.5))
    for t in S.TABLES:
        for k in S.ARRAY_FIELDS:
            assert np.array_equal(A0[t][k], A1[t][k]) and np.array_equal(A0[t][k], A2[t][k])


def test_samples_are_independent_of_other_samples_and_deterministic():
    w = world(_scene())
    cand, (A, M, *_r) = build(w)
    cand2, (A2, M2, *_r2) = build(w)
    for k in S.ARRAY_FIELDS:
        assert np.array_equal(A["g0p80"][k], A2["g0p80"][k])
    one = cand[cand["g0p80_outcome"] == C.ACCEPTED].iloc[[-1]]
    _, (A1, M1, *_r1) = build(w, one.drop(columns=["split"]))
    i = int(M["g0p80"].index[M["g0p80"]["candidate_id"] == one.iloc[0]["candidate_id"]][0])
    for k in S.ARRAY_FIELDS:
        assert np.array_equal(A1["g0p80"][k][0], A["g0p80"][k][i])           # no cross-sample normalisation


def test_shifted_gap_is_revalidated_and_failures_reported(monkeypatch):
    # player 0's chain breaks 0.7 s after t* + 0.4: fine for the centred gap (margin 0.5), broken for a +0.3 s shift
    w = world(crossing(0.1), drop=lambda s, i: s == 0 and i == int(round((10.05 + 0.4 + 0.7) * FPS)))
    monkeypatch.setattr(S, "shift_for", lambda cid: 0.3)
    cand, (A, M, F, lower, summ) = build(w)
    assert summ["g0p80"]["built"] == 1
    assert summ["g0p80_shift"]["built"] == 0 and summ["g0p80_shift"]["failed"] == 1
    assert F.iloc[0]["stage"] == "shift_revalidation" and "continuous_reference" in F.iloc[0]["reason"]
    monkeypatch.setattr(S, "shift_for", lambda cid: -0.3)
    _, (A, M, F, lower, summ) = build(w)
    assert summ["g0p80_shift"]["built"] == 1 and M["g0p80_shift"].iloc[0]["shift_s"] == -0.3
    assert M["g0p80_shift"].iloc[0]["gap_centre_s"] == pytest.approx(M["g0p80"].iloc[0]["t_star"] - 0.3)


def test_npz_is_byte_deterministic(tmp_path):
    A = S.empty_arrays(3)
    A["in_pos"][:] = 1.5
    S.save_npz(tmp_path / "a.npz", A)
    S.save_npz(tmp_path / "b.npz", A)
    assert (tmp_path / "a.npz").read_bytes() == (tmp_path / "b.npz").read_bytes()
    with np.load(tmp_path / "a.npz", allow_pickle=False) as z:
        assert set(z.files) == set(S.ARRAY_FIELDS) and np.array_equal(z["in_pos"], A["in_pos"])


# ---------------------------------------------------------------- model-input adapter: split protection

def _fake_out(tmp_path, matches, heldout):
    proc = tmp_path / "processed"
    sd = proc / S.SPLIT
    sd.mkdir(parents=True)
    txt = json.dumps(dict(heldout=heldout, train_cv=[m for m in matches if m not in heldout]))
    (sd / "split.json").write_text(txt, encoding="utf-8")
    sha = hashlib.sha256(txt.encode()).hexdigest()
    (sd / "manifest.json").write_text(json.dumps(dict(split_sha256=sha)), encoding="utf-8")
    out = proc / "m11-stage3-samples-test"
    for m in matches:
        def fn(mm, d):
            A = S.empty_arrays(2)
            A["y"][:] = (0, 1)
            for t in S.TABLES:
                S.save_npz(d / f"samples_{t}.npz", A)
            return {}
        L.run_matches(out, [m], fn, log=lambda *_: None)
    return proc, out, sha


def test_adapter_exposes_only_inputs_and_refuses_heldout(tmp_path):
    proc, out, sha = _fake_out(tmp_path, ["1", "2"], ["9"])
    X, y, g = S.load_model_inputs(out, "g0p80", proc=proc, expected_split_sha256=sha)
    assert tuple(X) == S.INPUT_FIELDS and len(y) == 4 and list(g) == [1, 1, 2, 2]
    with pytest.raises(S.HeldOutRefused):
        S.load_model_inputs(out, "g0p80", ["9"], proc=proc, expected_split_sha256=sha)
    with pytest.raises(S.HeldOutRefused):                                     # an unexpected split is refused
        S.load_model_inputs(out, "g0p80", proc=proc)


def test_adapter_refuses_when_a_heldout_shard_exists(tmp_path):
    proc, out, sha = _fake_out(tmp_path, ["1", "9"], ["9"])
    with pytest.raises(S.HeldOutRefused):
        S.load_model_inputs(out, "g0p80", ["1"], proc=proc, expected_split_sha256=sha)


def test_adapter_detects_tampered_shards(tmp_path):
    proc, out, sha = _fake_out(tmp_path, ["1"], ["9"])
    p = out / "matches" / "1" / "samples_g0p80.npz"
    A = S.empty_arrays(2)
    A["y"][:] = (1, 1)
    S.save_npz(p, A)
    with pytest.raises(RuntimeError):
        S.load_model_inputs(out, "g0p80", proc=proc, expected_split_sha256=sha)
