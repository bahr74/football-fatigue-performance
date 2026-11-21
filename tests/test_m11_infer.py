"""M11 Milestone 1 (``m11_infer``): adapter, routing, abstention, CLI and, when the project data are present, checkpoint loading,
reproduction of stored out-of-fold predictions and held-out refusal.

Unit tests use synthetic tracks only. Integration tests read train/CV development data (never held-out or official data) and are
skipped when those files are absent."""

import json

import numpy as np
import pytest

from football_fatigue.m11_infer import adapter as A
from football_fatigue.m11_infer import bundle as B
from football_fatigue.m11_infer import predict as P
from football_fatigue.window_intersections import samples as S

DT = 1.0 / 15.0


# ---------------------------------------------------------------- synthetic helpers

def _line(t0, t1, p0, v, dt=DT):
    t = np.arange(t0, t1 + 1e-9, dt)
    return A.Track(t, p0[0] + v[0] * (t - t0), p0[1] + v[1] * (t - t0))


def _crossing(gap=0.8, tc=10.0, ctx=3.0, dt=DT):
    """Two players crossing at tc: A moves +x, B moves +y; returns full chains (t, x, y) for each."""
    t = np.arange(tc - gap / 2 - ctx, tc + gap / 2 + ctx + 1e-9, dt)
    VA = (t, 2.0 * (t - tc), 0.3 * (t - tc))
    VB = (t, -0.2 * (t - tc), 2.0 * (t - tc) + 0.1)
    return VA, VB


def _cut(V, g0, g1, side):
    t, x, y = V
    k = t < g0 if side == "in" else t > g1
    return A.Track(t[k], x[k], y[k])


class _Const:
    """Stub model: constant P(SWAPPED); antisymmetry is not modelled (routing/abstention tests only)."""

    def __init__(self, p):
        self.p = p

    def predict(self, x):
        return np.full(len(x["gap_s"]), self.p)


# ---------------------------------------------------------------- adapter

def test_adapter_reproduces_build_one_exactly():
    VA, VB = _crossing()
    tc, g = 10.0, 0.8
    for io_ in ((0, 0), (0, 1), (1, 0), (1, 1)):
        s = S.build_one(VA, VB, tc, g, *io_)
        ref = S.model_arrays(s, g)
        V = {"A": VA, "B": VB}
        who_in = ("B", "A") if io_[0] else ("A", "B")
        who_out = ("B", "A") if io_[1] else ("A", "B")
        g0, g1 = tc - g / 2, tc + g / 2
        mi = A.build_inputs([_cut(V[p], g0, g1, "in") for p in who_in], [_cut(V[p], g0, g1, "out") for p in who_out], g0, g1)
        for k in S.INPUT_FIELDS:
            assert np.array_equal(mi.arrays[k][0], ref[k]), k
        assert mi.frame_mode == s["frame_mode"]


def test_adapter_ignores_observations_inside_gap_and_does_not_extrapolate():
    VA, VB = _crossing()
    g0, g1 = 9.6, 10.4
    full = [A.Track(*VA), A.Track(*VB)]                       # observations inside the gap are present
    mi = A.build_inputs(full, full[::-1], g0, g1)
    cut = A.build_inputs([_cut(VA, g0, g1, "in"), _cut(VB, g0, g1, "in")], [_cut(VB, g0, g1, "out"), _cut(VA, g0, g1, "out")], g0, g1)
    for k in S.INPUT_FIELDS:
        assert np.array_equal(mi.arrays[k], cut.arrays[k])
    assert sum(mi.ignored_in_gap) > 0 and any("ignored" in n for n in mi.notes)
    a0 = _line(9.0, 9.55, (0, 0), (1, 0))
    short = A.build_inputs([a0, _line(8.0, 9.55, (0, 1), (1, 0))],
                           [_line(10.45, 12.5, (1, 0), (1, 0)), _line(10.45, 12.5, (1, 1), (1, 0))], g0, g1)
    grid = g0 + S.IN_REL
    assert short.n_valid[0] == int(((grid >= a0.t[0]) & (grid <= a0.t[-1])).sum())        # only inside the observed span
    assert (grid[short.in_mask[0]] >= a0.t[0]).all()                                       # no extrapolation before 9.0


def test_adapter_resamples_by_timestamp_from_any_rate():
    g0, g1 = 9.6, 10.4
    for fps in (25.0, 30.0, 50.0, 60.0):
        dt = 1.0 / fps
        inc = [_line(7.0, 9.59, (0, 0), (2, 0), dt), _line(7.0, 9.59, (0, 2), (2, -1), dt)]
        out = [_line(10.41, 13.0, (7, -1), (2, -1), dt), _line(10.41, 13.0, (7, 0), (2, 0), dt)]
        mi = A.build_inputs(inc, out, g0, g1)
        assert mi.enough_points and mi.arrays["in_pos"].shape == (1, 2, S.T, 2)
        ref_t = g0 + S.IN_REL                                  # grid values are the linear motion at the grid times
        frame = (np.c_[2 * (ref_t - 7.0), np.zeros_like(ref_t)] - mi.origin) @ mi.R.T
        assert np.allclose(mi.arrays["in_pos"][0, 0], frame, atol=1e-4)


@pytest.mark.parametrize("bad", ["nan", "unsorted", "lengths", "empty", "pixels"])
def test_adapter_rejects_malformed_tracks(bad):
    t = np.arange(0, 2, DT)
    tr = {"nan": A.Track(t, np.r_[np.nan, t[1:]], t), "unsorted": A.Track(t[::-1].copy(), t, t),
          "lengths": A.Track(t, t[:-1], t), "empty": A.Track(np.zeros(0), np.zeros(0), np.zeros(0)),
          "pixels": A.Track(t, t * 800, t)}[bad]
    ok = _line(0, 2, (0, 0), (1, 0))
    with pytest.raises(A.InputError):
        A.build_inputs([tr, ok], [ok, ok], 2.1, 2.9)


def test_adapter_rejects_wrong_counts_and_bad_gap():
    ok = _line(0, 2, (0, 0), (1, 0))
    with pytest.raises(A.InputError):
        A.build_inputs([ok], [ok, ok], 2.1, 2.9)
    with pytest.raises(A.InputError):
        A.build_inputs([ok, ok, ok], [ok, ok], 2.1, 2.9)
    with pytest.raises(A.InputError):
        A.build_inputs([ok, ok], [ok, ok], 2.9, 2.1)
    with pytest.raises(A.InputError):
        A.gap_from_observations(3.0, 3.0)


def test_gap_from_observations_and_json_roundtrip():
    inc = [_line(7.6, 9.55, (0, 0), (1, 0)), _line(7.6, 9.55, (0, 1), (1, 0))]
    out = [_line(10.45, 12.4, (3, 0), (1, 0)), _line(10.45, 12.4, (3, 1), (1, 0))]
    t_last, t_first = inc[0].t[-1], out[0].t[0]
    g0, g1 = A.gap_from_observations(t_last, t_first)
    assert t_last < g0 < g1 < t_first and abs((g1 - g0) - (t_first - t_last)) < 1e-5
    d = dict(incoming=[t.to_dict() for t in inc], outgoing=[t.to_dict() for t in out], gap={"from_observations": True})
    i2, o2, a0, a1 = A.tracks_from_json(json.loads(json.dumps(d)))
    assert (a0, a1) == (g0, g1) and np.array_equal(i2[0].t, inc[0].t)
    with pytest.raises(A.InputError):
        A.tracks_from_json(dict(incoming=d["incoming"], outgoing=d["outgoing"], gap={}))


# ---------------------------------------------------------------- routing and abstention

@pytest.mark.parametrize("gap, noise, support, model", [
    (0.8, None, "primary", "gru_c"), (0.8, 0.01, "primary", "gru_c"), (0.75, 0.0, "primary", "gru_c"), (0.85, None, "primary", "gru_c"),
    (0.8, 0.03, "fallback", "hgb_c"), (0.4, None, "fallback", "hgb_c"), (1.2, None, "fallback", "hgb_c"), (1.6, 0.05, "fallback", "hgb_c"),
    (0.3, None, "unsupported", None), (2.0, None, "unsupported", None), (0.8, 0.10, "unsupported", None)])
def test_routing_policy(gap, noise, support, model):
    r = P.route(gap, noise, True)
    assert (r.support, r.model) == (support, model)
    assert r.reasons


def test_routing_too_few_points_and_invalid_noise():
    assert P.route(0.8, None, False).support == "unsupported"
    with pytest.raises(A.InputError):
        P.route(0.8, -0.01, True)


def _inputs(gap=0.8):
    VA, VB = _crossing(gap=gap)
    g0, g1 = 10.0 - gap / 2, 10.0 + gap / 2
    return [_cut(VA, g0, g1, "in"), _cut(VB, g0, g1, "in")], [_cut(VB, g0, g1, "out"), _cut(VA, g0, g1, "out")], g0, g1


@pytest.mark.parametrize("p, op, decision, abstained", [
    (0.95, "error_05", "SWAPPED", False), (0.05, "error_05", "MATCHED", False), (0.62, "error_05", "ABSTAIN", True),
    (0.70, "error_05", "SWAPPED", False), (0.70, "error_02", "ABSTAIN", True), (0.5, "error_05", "ABSTAIN", True)])
def test_abstention_on_primary(p, op, decision, abstained):
    pr = P.Predictor(0, models={"gru_c": _Const(p), "hgb_c": _Const(0.99)})
    res = pr.predict(*_inputs(), operating_point=op)
    assert res.model == "gru_c" and res.support == "primary"
    assert res.decision == decision and res.abstained == abstained
    assert res.threshold == B.THRESHOLDS["gru_c"][op]
    assert res.predicted_pairing == ("SWAPPED" if p > 0.5 else "MATCHED")
    assert (res.pairs is None) == abstained
    if not abstained:
        assert res.pairs == P.PAIRS[decision]
    if abstained:
        assert any(r.startswith("low_confidence") for r in res.reasons)


def test_fallback_uses_hgb_threshold_and_unsupported_abstains():
    pr = P.Predictor(0, models={"gru_c": _Const(0.99), "hgb_c": _Const(0.77)})
    res = pr.predict(*_inputs(1.2))
    assert res.model == "hgb_c" and res.support == "fallback" and res.abstained and res.threshold == B.THRESHOLDS["hgb_c"]["error_05"]
    res = pr.predict(*_inputs(0.8), noise_sigma_m=0.03)
    assert res.model == "hgb_c" and res.abstained
    res = pr.predict(*_inputs(2.0))
    assert res.support == "unsupported" and res.decision == "ABSTAIN" and res.model is None
    assert res.probabilities["gru_c"] == 0.99 and any("diagnostic" in r for r in res.reasons)


def test_too_few_points_abstains_without_probabilities():
    inc = [_line(9.47, 9.55, (0, 0), (1, 0)), _line(7.6, 9.55, (0, 1), (1, 0))]    # 2 observations: < 3 grid points
    out = [_line(10.45, 12.4, (3, 0), (1, 0)), _line(10.45, 12.4, (3, 1), (1, 0))]
    pr = P.Predictor(0, models={"gru_c": _Const(0.9), "hgb_c": _Const(0.9)})
    res = pr.predict(inc, out, 9.6, 10.4)
    assert res.abstained and res.support == "unsupported" and res.probabilities == {"gru_c": None, "hgb_c": None}


def test_invalid_operating_point():
    pr = P.Predictor(0, models={"gru_c": _Const(0.9), "hgb_c": _Const(0.9)})
    with pytest.raises(A.InputError):
        pr.predict(*_inputs(), operating_point="error_10")


# ---------------------------------------------------------------- integration (development data; skipped when absent)

def _have_data():
    try:
        return (B.source_dir() / "neural" / "C" / "seed0" / "fold4_c160.pt").exists() and B.manifest_path().parent.parent.exists()
    except Exception:
        return False


data = pytest.mark.skipif(not _have_data(), reason="M11 development data not present")


@data
def test_heldout_and_unknown_matches_are_refused():
    for m in B.heldout_matches():
        with pytest.raises(S.HeldOutRefused):
            B.fold_for_match(m)
        with pytest.raises(S.HeldOutRefused):
            P.Predictor.for_match(m)
    with pytest.raises(ValueError):
        B.fold_for_match("3813")                   # an official test match: not part of the development split
    assert B.fold_for_match("3812") == 4 and "3812" in B.fold_matches(4)


@data
def test_heldout_refused_by_fifa_source():
    from football_fatigue.m11_infer import fifa

    with pytest.raises(S.HeldOutRefused):
        fifa.MatchSource(B.heldout_matches()[0])


@data
@pytest.mark.parametrize("name", B.MODELS)
def test_fold_checkpoint_loads_and_reproduces_stored_oof(name):
    fm = B.load_fold_model(name, 4)
    assert len(fm.sha256) == 64 and fm.meta["unseen_matches"] == B.fold_matches(4)
    x, y, _g = B.fold_inputs(4)
    k = np.arange(0, len(y), 7)[:300]
    xs = {kk: v[k] for kk, v in x.items()}
    p = fm.predict(xs)
    ref = B.stored_oof(name, 4)[k]
    assert np.abs(p - ref).max() <= B.tolerance(name, "cpu")
    assert ((p > 0.5) == (ref > 0.5)).all()
    sw = dict(xs, out_pos=xs["out_pos"][:, ::-1].copy(), out_mask=xs["out_mask"][:, ::-1].copy())
    assert np.abs(fm.predict(sw) - (1 - p)).max() < 1e-5            # outgoing-order symmetry


@data
def test_checkpoint_hash_mismatch_is_refused():
    with pytest.raises(RuntimeError):
        B.load_fold_model("hgb_c", 0, expected_sha256="0" * 64)


@data
def test_adapter_rebuilds_stage3_from_raw_and_predictions_match():
    from football_fatigue.m11_infer import fifa

    pr = P.Predictor.for_match("3812")
    r = fifa.check_adapter("3812", limit=15, predictor=pr, seed=3, log=lambda s: None)
    assert r["all_identical"] and r["predictions_reproduced"], r


@data
def test_real_scenario_symmetry_and_cli(tmp_path, capsys):
    from football_fatigue.m11_infer import fifa
    from football_fatigue.m11_infer.__main__ import main

    sc = fifa.MatchSource("3812").scenario(0)
    pr = P.Predictor.for_match("3812")
    a = pr.predict(sc.incoming, sc.outgoing, sc.g0, sc.g1)
    b = pr.predict(sc.incoming, sc.outgoing[::-1], sc.g0, sc.g1)
    assert a.model == b.model == "gru_c"
    assert abs(a.p_swapped - (1 - b.p_swapped)) < 1e-5 and a.abstained == b.abstained
    if not a.abstained:
        assert {a.decision, b.decision} == {"MATCHED", "SWAPPED"}
    f = tmp_path / "in.json"
    f.write_text(json.dumps(sc.input_json()), encoding="utf-8")
    assert "y" not in sc.input_json() and "pairing" not in json.dumps(sc.input_json())
    assert main(["predict", "--json", str(f)]) == 0                    # fold chosen from source.match_id (unseen-match fold)
    out = json.loads(capsys.readouterr().out)
    assert out["fold"] == 4 and abs(out["p_swapped"] - a.p_swapped) < 1e-12 and out["policy"] == P.POLICY
