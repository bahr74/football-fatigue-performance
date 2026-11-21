"""M11 Milestone 2 scenario suite: deterministic generation, perturbations, gap handling, label preservation, fold selection,
routing/abstention accounting and protected-match refusal. Unit tests are synthetic; integration tests use train/CV development
data and are skipped when absent."""

import numpy as np
import pandas as pd
import pytest

from football_fatigue.m11_benchmark import scenario_eval as E
from football_fatigue.m11_benchmark import scenarios as SC
from football_fatigue.m11_infer import adapter as A
from football_fatigue.m11_infer import bundle as B
from football_fatigue.m11_infer import predict as P
from football_fatigue.window_intersections import samples as S

DT = 1.0 / 15.0


def _track(t0=0.0, t1=3.0, v=(1.0, 0.5)):
    t = np.arange(t0, t1, DT)
    return A.Track(t, v[0] * t, v[1] * t)


# ---------------------------------------------------------------- ids, levels, variants

def test_variants_and_ids_are_deterministic_and_distinct():
    assert len(SC.VARIANTS) == 1 + sum(len(v) for v in SC.FACTORS.values()) == 29
    ids = [SC.scenario_id("abc", f, lv) for f, lv in SC.VARIANTS]
    assert len(set(ids)) == len(ids) and ids == [SC.scenario_id("abc", f, lv) for f, lv in SC.VARIANTS]
    assert SC.scenario_id("abd", "gap", 1.6) != SC.scenario_id("abc", "gap", 1.6)
    for f, lvls in SC.FACTORS.items():
        for lv in lvls:
            assert SC.parse_level(f, SC.level_key(lv)) == lv
            p = SC.params(f, lv)
            assert p[f] == lv and all(p[k] == SC.BASE[k] for k in SC.BASE if k != f)       # one factor at a time
    with pytest.raises(ValueError):
        SC.params("gap", 3.0)


# ---------------------------------------------------------------- perturbations

def test_cut_respects_gap_and_context():
    t = np.arange(0, 10, DT)
    V = (t, t * 1.0, t * 0.0)
    g0, g1 = 5.0, 5.8
    tin = SC.cut(V, g0, g1, "in", 2.0)
    tout = SC.cut(V, g0, g1, "out", 2.0)
    assert (tin.t < g0).all() and (tout.t > g1).all()
    assert tin.t[0] >= g0 - 3.0 - 1e-9 and tout.t[-1] <= g1 + 3.0 + 1e-9                  # 2 s context + 1 s bracketing margin
    short = SC.cut(V, g0, g1, "in", 1.0)
    assert short.t[0] >= g0 - 1.0 - 1e-9 and (short.t < g0).all()


@pytest.mark.parametrize("mode", ["random10", "random30", "random50", "burst0p5", "rate7p5"])
def test_dropout_is_deterministic_and_only_removes(mode):
    tr = _track()
    a = SC.drop(tr, mode, "in", np.random.default_rng(5))
    b = SC.drop(tr, mode, "in", np.random.default_rng(5))
    assert np.array_equal(a.t, b.t) and len(a.t) < len(tr.t)
    assert np.isin(a.t, tr.t).all()                                                       # no new observation times
    k = np.searchsorted(tr.t, a.t)
    assert np.array_equal(a.x, tr.x[k]) and np.array_equal(a.y, tr.y[k])                  # values untouched
    if mode == "burst0p5":
        gone = np.setdiff1d(tr.t, a.t)
        assert gone.max() - gone.min() < 0.5 + 1e-9
    if mode == "rate7p5":
        assert a.t[-1] == tr.t[-1] and SC.drop(tr, mode, "out", np.random.default_rng(0)).t[0] == tr.t[0]   # nearest-gap point kept
    if mode == "random50":
        assert 0.35 < len(a.t) / len(tr.t) < 0.65


def test_noise_is_seeded_and_keeps_times():
    tr = _track()
    a = SC.add_noise(tr, 0.05, np.random.default_rng(1))
    b = SC.add_noise(tr, 0.05, np.random.default_rng(1))
    assert np.array_equal(a.x, b.x) and np.array_equal(a.t, tr.t)
    assert 0.03 < np.std(a.x - tr.x) < 0.07
    assert SC.add_noise(tr, 0.0, np.random.default_rng(1)) is tr


def test_families_on_synthetic_crossing():
    t = np.arange(0, 10, DT)
    VA = (t, 2.0 * (t - 5), 0.0 * t)
    VB = (t, 0.0 * t, 2.0 * (t - 5) + 0.1)
    f = SC.families(VA, VB, 5.0)
    assert f["heading_relation"] == "crossing_angle" and f["distance_trend"] == "converge_then_diverge"
    assert f["heading_change"] == "straight" and not f["hard_parallel"]
    VC = (t, 2.0 * (t - 5), 0.0 * t + 0.2)
    g = SC.families(VA, VC, 5.0)
    assert g["heading_relation"] == "same_direction" and g["distance_trend"] == "parallel_close" and g["hard_parallel"]


# ---------------------------------------------------------------- routing accounting

class _Const:
    def __init__(self, p):
        self.p = p

    def predict(self, x):
        return np.full(len(x["gap_s"]), self.p)


@pytest.mark.parametrize("gap, noise, pg, ph", [(0.8, 0.0, 0.9, 0.2), (0.8, 0.0, 0.6, 0.2), (1.2, 0.0, 0.9, 0.95), (1.2, 0.0, 0.9, 0.7),
                                                 (0.8, 0.05, 0.9, 0.1), (2.0, 0.0, 0.9, 0.9), (0.8, 0.25, 0.1, 0.1)])
def test_operational_equals_predictor(gap, noise, pg, ph):
    t = np.arange(0, 3, DT)
    inc = [A.Track(t, t, 0 * t), A.Track(t, t, 0 * t + 1)]
    out = [A.Track(t + 3 + gap, t + 3 + gap, 0 * t), A.Track(t + 3 + gap, t + 3 + gap, 0 * t + 1)]
    g0 = 3.0 - 1e-6
    mi = A.build_inputs(inc, out, g0, g0 + gap)
    ref = P.Predictor(0, models={"gru_c": _Const(pg), "hgb_c": _Const(ph)}).predict_input(mi, noise_sigma_m=noise)
    for y in (0, 1):
        o = E.operational(pg, ph, mi.gap_s, noise, mi.enough_points, y)
        assert (o["decision"], o["support"], o["model"], o["threshold"]) == (ref.decision, ref.support, ref.model, ref.threshold)
        if ref.model is None:
            assert o["outcome"] == "abstain_unsupported" and o["p_chosen"] is None
        elif ref.abstained:
            assert o["outcome"] == "abstain_low_confidence"
        else:
            assert o["outcome"] == ("accepted_correct" if (ref.decision == "SWAPPED") == bool(y) else "accepted_wrong")


def test_routed_accounting_sums():
    d = pd.DataFrame(dict(outcome=["accepted_correct"] * 6 + ["accepted_wrong"] * 2 + ["abstain_low_confidence"] * 1 + ["abstain_unsupported"] * 1,
                          support=["primary"] * 9 + ["unsupported"], model=["gru_c"] * 9 + [None],
                          abstain_reason=[""] * 8 + ["low_confidence", "gap_outside_evaluated_range"]))
    r = E._routed(d)
    assert sum(r[k] for k in E.OUTCOMES) == r["n"] == 10
    assert r["coverage"] == 0.8 and r["realized_error"] == 0.25 and r["unsupported_reasons"] == {"gap_outside_evaluated_range": 1}


# ---------------------------------------------------------------- integration (development data)

def _have():
    try:
        return (B.source_dir() / "neural" / "C" / "seed0" / "fold4_c160.pt").exists()
    except Exception:
        return False


data = pytest.mark.skipif(not _have(), reason="M11 development data not present")


@data
def test_protected_matches_refused():
    with pytest.raises(S.HeldOutRefused):
        SC.MatchData(B.heldout_matches()[0])
    with pytest.raises(ValueError):
        SC.MatchData("3813")


@pytest.fixture(scope="module")
def md():
    return SC.MatchData("3812")


@data
def test_selection_deterministic_and_independent(md):
    a, b = SC.select_encounters(md.meta, "3812"), SC.select_encounters(md.meta, "3812")
    assert a["candidate_id"].tolist() == b["candidate_id"].tolist() and len(a) == SC.N_PER_MATCH
    assert a["independent"].all()


@data
def test_scenario_regeneration_gap_semantics_and_truth(md):
    enc = SC.select_encounters(md.meta, "3812").iloc[0]
    fam = SC.encounter_families(md, enc)
    for factor, level in [("baseline", "base"), ("gap", 1.6), ("shift", 0.3), ("noise", 0.05), ("dropout", "random30"), ("context", 1.0)]:
        a, b = SC.build(md, enc, factor, level, fam), SC.build(md, enc, factor, level, fam)
        if not a.valid:
            assert a.invalid_reason and not b.valid
            continue
        assert md.fold == B.fold_for_match("3812") == 4
        for ta, tb in zip(a.incoming + a.outgoing, b.incoming + b.outgoing):
            assert np.array_equal(ta.t, tb.t) and np.array_equal(ta.x, tb.x)
        assert abs((a.g1 - a.g0) - a.params["gap"]) < 1e-9
        assert all((tr.t < a.g0).all() for tr in a.incoming) and all((tr.t > a.g1).all() for tr in a.outgoing)   # nothing from the gap
        assert a.y == (0 if a.who_in[0] == a.who_out[0] else 1)
        assert set(a.who_in) == set(a.who_out) == {"A", "B"}
        assert a.hidden_mask[0].sum() == int(np.floor(a.params["gap"] / DT + 1e-6)) + 1


@data
def test_baseline_prediction_reproduces_stored_oof_under_reordering(md):
    """Baseline = Stage 3 sample with new order bits: by exact antisymmetry p_new = p_stage3 if the label is unchanged, else 1 - p."""
    fm = B.load_fold_model("gru_c", 4)
    offset = 0                                                            # 3812 is the first match of fold 4 (ascending ids)
    assert B.fold_matches(4)[0] == "3812"
    stored = B.stored_oof("gru_c", 4)
    enc = SC.select_encounters(md.meta, "3812").head(8)
    for _, e in enc.iterrows():
        sc = SC.build(md, e, "baseline", "base")
        mi = A.build_inputs(sc.incoming, sc.outgoing, sc.g0, sc.g1)
        p = float(fm.predict(mi.arrays)[0])
        p3 = float(stored[offset + int(e["row"])])
        expect = p3 if sc.y == int(e["y"]) else 1 - p3
        assert abs(p - expect) < 2.5e-4


@data
def test_saved_results_consistent_if_present():
    out = E.out_dir()
    if not (out / "matches").exists() or not any((out / "matches").glob("*/done.json")):
        pytest.skip("scenario suite not run")
    df = E.load_results()
    held = set(B.heldout_matches())
    assert not (set(df["match_id"]) & held)
    for m, f in df.groupby("match_id")["fold"].first().items():
        assert B.fold_for_match(m) == f and m in B.fold_matches(f)
    v = df[df["valid"]]
    assert set(v["outcome"]) <= set(E.OUTCOMES) and (df.loc[~df["valid"], "outcome"] == "invalid").all()
    uns = v[v["outcome"] == "abstain_unsupported"]
    assert uns["model"].isna().all() and (uns["decision"] == "ABSTAIN").all()
    acc = v[v["outcome"].str.startswith("accepted")]
    assert ((acc["p_chosen"] > 0.5).astype(int).eq(acc["y"]) == (acc["outcome"] == "accepted_correct")).all()
    assert (acc["confidence"] >= acc["threshold"]).all()
    m = v["match_id"].iloc[0]
    arr = E.load_arrays(m)
    vm = v[v["match_id"] == m]
    assert len(arr["pitch_in"]) == len(vm) and sorted(vm["arr_row"].astype(int)) == list(range(len(vm)))


def test_empty_tracklet_is_unsupported_not_filled():
    t = np.arange(0, 3, DT)
    sc = SC.Scenario("x", "0", 0, "c", "dropout", "random50", SC.params("dropout", "random50"), True, y=0, g0=3.0, g1=3.8,
                     incoming=[A.Track(t, t, 0 * t), A.Track(np.zeros(0), np.zeros(0), np.zeros(0))],
                     outgoing=[A.Track(t + 4, t, 0 * t), A.Track(t + 4, t, 0 * t + 1)])
    mi = E.model_input(sc)
    assert not mi.enough_points and not mi.in_mask.any() and not mi.out_mask.any()
    o = E.operational(None, None, mi.gap_s, 0.0, mi.enough_points, 0)
    assert o["outcome"] == "abstain_unsupported" and o["abstain_reason"] == "too_few_valid_points"
