"""Tests for football_fatigue.matching.ml2 (M11 step 5). All instances are SYNTHETIC hand-built task B instances; passing these
tests says the pipeline is wired correctly, not that it is accurate on real data."""

import copy
import json
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("sklearn")

from football_fatigue.ingestion import IngestionError  # noqa: E402
from football_fatigue.matching import ml, ml2  # noqa: E402
from football_fatigue.matching.benchmark import TASK, TASK_B  # noqa: E402

# ---------------------------------------------------------------- synthetic instances (as in test_matching_ml)

def track(tid, f, t0, t1, vis=1, step=2):
    """30 fps samples on [t0, t1]; positions update every ``step`` frames (15 Hz like the real data)."""
    k = np.arange(round(t0 * 30), round(t1 * 30) + 1)
    kq = k - (k - k[0]) % step
    P = np.array([f(q / 30) for q in kq])
    return dict(tracklet_id=tid, t=[round(q / 30, 4) for q in k], x=P[:, 0].tolist(), y=P[:, 1].tolist(),
                visibility=[vis] * len(k), confidence=[3] * len(k))


def instance(iid, players, gap=1.5, absent_out=(), vis_out=None):
    """players: list of (in_id, out_id, f). Incoming on [-2, -1/30], outgoing on [gap, gap + 2]."""
    inc = [track(i, f, -2.0, -1 / 30) for i, _, f in players]
    out = [track(o, f, gap, gap + 2.0, vis=(vis_out or {}).get(o, 1)) for _, o, f in players if o not in absent_out]
    return dict(instance_id=iid, schema="test", task=TASK_B, units={}, ambiguity_interval_s=[0.0, gap], incoming=inc, outgoing=out,
                during=dict(t=[], points=[]))


def reference(inst, expected, tier=ml.LABEL_TIER, gap=1.5):
    return dict(instance_id=inst["instance_id"], tier=tier, expected=expected, incoming={k: {} for k in expected},
                outgoing={b["tracklet_id"]: {} for b in inst["outgoing"]}, team_relation="opposite_team", without_incoming=[], without_outgoing=[],
                step1_categories={}, difficulty=dict(label="standard", reasons=[], ambiguity_s=gap, crossing=False, max_heading_change_deg=None))


def line(p, v):
    return lambda t: (p[0] + v[0] * t, p[1] + v[1] * t)


def random_encounter(rng, iid, tier=ml.LABEL_TIER):
    """Two players continuing at constant velocity with a small heading drift; anonymous IDs in random order."""
    gap = float(rng.uniform(1.0, 3.0))
    players, expected = [], {}
    outs = [f"out-{k}" for k in rng.permutation(2)]
    for k in range(2):
        p = rng.uniform(-3, 3, 2)
        v = rng.uniform(-5, 5, 2)
        w = rng.uniform(-0.3, 0.3)
        f = (lambda p, v, w: lambda t: (p[0] + v[0] * t + w * t * t, p[1] + v[1] * t - w * t * t))(p, v, w)
        players.append((f"in-{k}", outs[k], f))
        expected[f"in-{k}"] = outs[k]
    inst = instance(iid, players, gap=gap)
    inst["outgoing"].sort(key=lambda b: b["tracklet_id"])
    return inst, reference(inst, expected, tier, gap)


CFG = ml2.ML2Config(n_jobs_folds=1, n_jobs_model=1, hgb_iter=30, mlp_iter=50, bootstrap=200)


def G(inst, i, o, name):
    rows, _ = ml.instance_features(inst)
    X = ml2.football_features(inst)
    k = next(n for n, r in enumerate(rows) if r["incoming"] == i and r["outgoing"] == o)
    return X[k, ml2.G_FEATURES.index(name)]


def table2(match_id, n, seed, tiers=(ml.LABEL_TIER,)):
    rng = np.random.default_rng(seed)
    insts, refs = [], []
    for k in range(n):
        i, r = random_encounter(rng, f"{match_id}-{k:03d}", tiers[k % len(tiers)])
        insts.append(i)
        refs.append(r)
    return ml2.match_table2(dict(match_id=match_id, instances=insts, references=refs))


@pytest.fixture(scope="module")
def tables():
    return [table2(m, 14, seed=s) for m, s in (("m1", 1), ("m2", 2), ("m3", 3), ("m4", 4))]


# ---------------------------------------------------------------- football-aware features

def test_football_features_straight_runners_and_alignment():
    inst = instance("a", [("in-0", "out-0", line((0, 0), (5, 0))), ("in-1", "out-1", line((0, 4), (0, -3)))], gap=2.0)
    rows, X = ml.instance_features(inst)
    assert ml2.football_features(inst).shape == (len(rows), len(ml2.G_FEATURES))
    assert G(inst, "in-0", "out-0", "turn_in_deg") == pytest.approx(0, abs=1.0)
    assert G(inst, "in-0", "out-0", "hermite_peak_speed") == pytest.approx(5, abs=0.2)     # 15 Hz holds bias the endpoints slightly
    assert G(inst, "in-0", "out-0", "assign_regret_cv") == 0 and G(inst, "in-1", "out-1", "assign_regret_cv") == 0
    assert G(inst, "in-0", "out-1", "assign_regret_cv") > 1.0
    assert G(inst, "in-0", "out-0", "n_reach_row") == 2


def test_football_features_order_invariant_and_task_b_only():
    inst = instance("a", [("in-0", "out-1", line((0, 0), (4, 1))), ("in-1", "out-0", line((0, 3), (-4, 0))), ("in-2", "out-2", line((2, 1), (1, 3)))])
    sh = copy.deepcopy(inst)
    sh["incoming"].reverse()
    sh["outgoing"] = [sh["outgoing"][k] for k in (2, 0, 1)]
    for i in ("in-0", "in-1", "in-2"):
        for o in ("out-0", "out-1", "out-2"):
            for f in ml2.G_FEATURES:
                assert G(inst, i, o, f) == pytest.approx(G(sh, i, o, f), abs=1e-9)
    with pytest.raises(ValueError):
        ml2.football_features(dict(inst, task=TASK))
    with pytest.raises(ValueError):
        ml2.football_features(dict(inst, during=dict(t=[0.5], points=[[]])))


def test_sharp_turn_is_cheap_at_low_speed_and_costly_at_high_speed():
    """A U-turn by a slow player is physically easy; the same geometry at sprint speed is not. Neither is labelled an error."""
    slow = instance("s", [("in-0", "out-0", lambda t: (1.2 * t, 0.0) if t < 0.8 else (1.2 * (1.6 - t), 0.0))], gap=1.0)
    fast = instance("f", [("in-0", "out-0", lambda t: (6.0 * t, 0.0) if t < 0.8 else (6.0 * (1.6 - t), 0.0))], gap=1.0)
    assert G(slow, "in-0", "out-0", "turn_out_deg") > 90 and G(fast, "in-0", "out-0", "turn_out_deg") > 90
    assert G(slow, "in-0", "out-0", "turn_acc_demand") < G(fast, "in-0", "out-0", "turn_acc_demand") / 3


def test_long_gap_reachability_counts():
    inst = instance("a", [("in-0", "out-0", line((0, 0), (1, 0))), ("in-1", "out-1", lambda t: (0.0, 3.0) if t < 0 else (60.0, 3.0))], gap=1.0)
    assert G(inst, "in-0", "out-1", "n_reach_row") == 1          # 60 m in 1 s is out of reach
    assert G(inst, "in-1", "out-1", "n_reach_col") == 0


def test_assignment_regret_multi_player():
    C = np.array([[0.5, 3.0, 4.0], [3.0, 0.4, 2.0], [4.0, 2.5, 0.3]])
    R = ml2.assignment_regret(C, cap=10.0)
    assert np.allclose(np.diag(R), 0)
    assert (R[~np.eye(3, dtype=bool)] > 0).all()
    assert R[0, 1] == pytest.approx((3.0 + 3.0 + 0.3) - (0.5 + 0.4 + 0.3))


def test_no_identity_leakage_into_features():
    rng = np.random.default_rng(5)
    inst, ref = random_encounter(rng, "x")
    tagged = copy.deepcopy(inst)
    for k, tr in enumerate(tagged["incoming"] + tagged["outgoing"]):
        tr.update(player_id=1000 + k, jersey=k, team_id=7)
    assert np.array_equal(ml2.football_features(inst), ml2.football_features(tagged))
    flipped = dict(ref, expected={k: v for k, v in zip(ref["expected"], reversed(list(ref["expected"].values())))})
    a = ml2.match_table2(dict(match_id="m", instances=[inst], references=[ref]))
    b = ml2.match_table2(dict(match_id="m", instances=[inst], references=[flipped]))
    assert np.array_equal(a["X"], b["X"]) and [x["label"] for x in a["labels"]] != [x["label"] for x in b["labels"]]
    tokens = {"id", "player", "jersey", "team", "match", "frame", "label", "slot", "roster"}
    assert not any(tokens & set(f.split("_")) for f in ml2.FEATURES2)


def test_inputs_not_mutated():
    rng = np.random.default_rng(6)
    inst, ref = random_encounter(rng, "x")
    before = copy.deepcopy((inst, ref))
    ml2.match_table2(dict(match_id="m", instances=[inst], references=[ref]))
    assert (inst, ref) == before


# ---------------------------------------------------------------- families and ablations

def test_families_partition_features_and_ablation_variants():
    allf = [f for fs in ml2.FAMILIES.values() for f in fs]
    assert sorted(allf) == sorted(ml2.FEATURES2) and len(allf) == len(set(allf))
    v = ml2.ablation_variants()
    assert set(v["full"]) == set(ml2.FEATURES2)
    for fam, fs in ml2.FAMILIES.items():
        assert set(v[f"drop_{fam}"]) == set(ml2.FEATURES2) - set(fs)
        assert v[f"only_{fam}"] == fs
    assert ml2.FEATURE_SETS["core"] == ml2.CORE and set(ml2.CORE) == set(ml.CORE_FEATURES)


# ---------------------------------------------------------------- grouped isolation, thresholds, reproducibility

def test_nested_fit_never_sees_heldout_rows(tables, monkeypatch):
    seen = []
    real = ml2.fit_model

    def spy(kind, params, X, y, cfg=ml2.ML2Config()):
        seen.append({r.tobytes() for r in X})
        return real(kind, params, X, y, cfg)

    monkeypatch.setattr(ml2, "fit_model", spy)
    held, train = tables[0], tables[1:]
    c = ml2.cols(ml2.CORE)
    held_rows = {r.tobytes() for r in held["X"][:, c]}
    nf = ml2.nested_fit("lr", ml2.CORE, train, CFG)
    assert sorted(m for g in nf["inner_groups"] for m in g) == sorted(t["match_id"] for t in train)
    assert all(not (s & held_rows) for s in seen)
    for g, s in zip(nf["inner_groups"], seen):                       # each inner fit excludes its own inner group
        out_rows = {r.tobytes() for t in train if t["match_id"] in g for r in t["X"][ml._train_mask(t)][:, c]}
        assert not (s & out_rows)


def test_operating_point_ignores_heldout_labels(tables):
    a = ml2.run_comparison(tables, CFG, comparison=(("lr", "core"),), log=lambda m: None)["lr/core"]
    flipped = copy.deepcopy(tables)
    h = flipped[0]
    for lb in h["labels"]:
        lb["label"] = 1 - lb["label"]
    for r in h["refs"].values():
        ks = list(r["expected"])
        r["expected"] = dict(zip(ks, list(r["expected"].values())[::-1]))
    b = ml2.run_comparison(flipped, CFG, comparison=(("lr", "core"),), log=lambda m: None)["lr/core"]
    assert np.array_equal(a["p"]["m1"], b["p"]["m1"])
    assert a["folds"]["m1"]["operating"] == b["folds"]["m1"]["operating"]
    with pytest.raises(ValueError):
        ml2.run_comparison(tables[:3], CFG, comparison=(("lr", "core"),))


def test_reproducible_and_parallel_equals_serial(tables):
    a = ml2.run_comparison(tables, CFG, comparison=(("hgb", "core_g"),), log=lambda m: None)["hgb/core_g"]
    b = ml2.run_comparison(tables, replace(CFG, n_jobs_folds=2), comparison=(("hgb", "core_g"),), log=lambda m: None)["hgb/core_g"]
    for m in a["p"]:
        assert np.array_equal(a["p"][m], b["p"][m])


def test_select_operating_on_inner_table():
    T = dict(pick=np.array([True] * 6), p=np.array([0.95, 0.9, 0.8, 0.7, 0.6, 0.55]), margin=np.full(6, 3.0), no_plausible=np.zeros(6, bool),
             correct=np.array([True, True, True, False, True, False]), exp_none=np.zeros(6, bool), inst=np.arange(6), n_inst=6)
    s = ml2.select_operating(T, 0.05, CFG)
    assert s["target_met"] and s["inner_oof"]["coverage"] == pytest.approx(0.5)
    assert s["threshold"] == pytest.approx(0.80) and s["margin"] == 2.0         # ties go to the most conservative point
    s10 = ml2.select_operating(T, 0.25, CFG)
    assert s10["inner_oof"]["coverage"] == pytest.approx(5 / 6)
    T["correct"][:] = False
    assert not ml2.select_operating(T, 0.05, CFG)["target_met"]


def test_vectorised_rates_match_reference_decisions(tables):
    rng = np.random.default_rng(0)
    t = tables[1]
    p = rng.uniform(0, 1, len(t["rows"]))
    assigned = ml.instance_assignments(t["rows"], p, CFG.base)
    T = ml2.decision_table(assigned, t["refs"])
    for thr, mg in ((0.5, 0.0), (0.7, 0.5), (0.9, 2.0)):
        ref = ml.encounter_summary(ml.decisions(assigned, thr, replace(CFG.base, margin_logit=mg)), t["refs"])
        got = ml2.table_rates(T, thr, mg)
        assert got["coverage"] == pytest.approx(ref["coverage"], abs=1e-4)
        assert got["complete_assignment_accuracy"] == pytest.approx(ref["complete_assignment_accuracy"], abs=1e-4)
        if ref["false_association_rate"] is not None:
            assert got["false_association_rate"] == pytest.approx(ref["false_association_rate"], abs=1e-4)


def test_abstention_and_unmatched():
    T = dict(pick=np.array([True, False, False]), p=np.array([0.97, 0.9, -1.0]), margin=np.array([0.1, -1.0, -1.0]),
             no_plausible=np.array([False, False, True]), correct=np.array([True, False, False]), exp_none=np.array([False, False, True]),
             inst=np.array([0, 0, 1]), n_inst=2)
    r = ml2.table_rates(T, 0.5, 0.5)                                 # margin too small: abstain; unreachable: UNMATCHED
    assert r["matched"] == 0 and r["uncertain_rate"] == pytest.approx(2 / 3) and r["unmatched_rate"] == pytest.approx(1 / 3)
    assert r["complete_assignment_accuracy"] == pytest.approx(0.5)
    assert ml2.table_rates(T, 0.5, 0.0)["matched"] == 1


# ---------------------------------------------------------------- calibration, bootstrap, persistence

def test_platt_calibration_improves_overconfident_scores():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 4000)
    z = (y * 2 - 1) * 0.8 + rng.normal(0, 1, 4000)
    p_over = 1 / (1 + np.exp(-3 * z))                                # overconfident
    cal = ml2.fit_platt(p_over, y)
    assert cal.a < 1
    assert ml2.log_loss(y, cal(p_over)) < ml2.log_loss(y, p_over)
    q = np.linspace(0.01, 0.99, 50)
    assert np.all(np.diff(cal(q)) > 0)


def test_match_bootstrap():
    pm = {m: dict(incoming=100, matched=20, wrong=1) for m in "abcd"}
    b = ml2.bootstrap(pm, pm, 200, 0)
    assert b["coverage_ci95"] == [0.2, 0.2] and b["coverage_diff_ci95"] == [0.0, 0.0]


def test_model_persistence(tables, tmp_path):
    import joblib

    nf = ml2.nested_fit("hgb", ml2.FEATURE_SETS["core_g"], tables[1:], CFG)
    X = tables[0]["X"][:, ml2.cols(ml2.FEATURE_SETS["core_g"])]
    joblib.dump(dict(model=nf["model"], calibrator=vars(nf["calibrator"])), tmp_path / "m.joblib")
    back = joblib.load(tmp_path / "m.joblib")
    assert np.array_equal(ml2.Platt(**back["calibrator"])(ml2.proba(back["model"], X)), nf["calibrator"](ml2.proba(nf["model"], X)))


def test_run_ml2_refuses_non_development_and_existing_output(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    sp = tmp_path / "processed" / ml.SPLITS_DIR
    sp.mkdir(parents=True)
    (sp / "splits.json").write_text(json.dumps(dict(matches={"1": dict(split="development"), "2": dict(split="validation"), "3": dict(split="test")})))
    with pytest.raises(IngestionError, match="development"):
        ml2.run_ml2(raw, ["1", "2"], out_root=tmp_path / "out")
    with pytest.raises(IngestionError, match="development"):
        ml2.run_ml2(raw, ["3"], out_root=tmp_path / "out")
    (tmp_path / "out").mkdir()
    with pytest.raises(IngestionError, match="never overwritten"):
        ml2.run_ml2(raw, ["1"], out_root=tmp_path / "out")
