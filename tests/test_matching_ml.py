"""Tests for football_fatigue.matching.ml (M11 step 4). All instances are SYNTHETIC hand-built task B instances; passing these
tests says the pipeline is wired correctly, not that it is accurate on real data."""

import inspect
import json

import numpy as np
import pytest

pytest.importorskip("sklearn")

from football_fatigue.ingestion import IngestionError
from football_fatigue.matching import ml
from football_fatigue.matching.benchmark import TASK, TASK_B

CFG = ml.MLConfig()


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


def table(match_id, n, seed, tiers=(ml.LABEL_TIER,)):
    rng = np.random.default_rng(seed)
    insts, refs = [], []
    for k in range(n):
        i, r = random_encounter(rng, f"{match_id}-{k:03d}", tiers[k % len(tiers)])
        insts.append(i)
        refs.append(r)
    return ml.match_table(dict(match_id=match_id, instances=insts, references=refs))


# ---------------------------------------------------------------- labels

def test_pair_labels_and_status():
    inst = instance("a", [("in-0", "out-1", line((0, 0), (4, 0))), ("in-1", "out-0", line((0, 3), (-4, 0)))])
    labels = {(r["incoming"], r["outgoing"]): r for r in ml.pair_labels(reference(inst, {"in-0": "out-1", "in-1": "out-0"}), inst)}
    assert {k: v["label"] for k, v in labels.items()} == {("in-0", "out-1"): 1, ("in-0", "out-0"): 0, ("in-1", "out-0"): 1, ("in-1", "out-1"): 0}
    assert {v["status"] for v in labels.values()} == {"train"}
    low = ml.pair_labels(reference(inst, {"in-0": "out-1", "in-1": "out-0"}, ml.EXPLORATORY_TIER), inst)
    assert {r["status"] for r in low} == {"exploratory"}
    sus = ml.pair_labels(reference(inst, {"in-0": "out-1", "in-1": "out-0"}, ml.UNVERIFIED_TIER), inst)
    assert {r["status"] for r in sus} == {"unverified"} and {r["label"] for r in sus} == {None}


def test_estimated_endpoint_withheld_from_training():
    inst = instance("a", [("in-0", "out-1", line((0, 0), (4, 0))), ("in-1", "out-0", line((0, 3), (-4, 0)))], vis_out={"out-0": 2})
    st = {(r["incoming"], r["outgoing"]): r["status"] for r in ml.pair_labels(reference(inst, {"in-0": "out-1", "in-1": "out-0"}), inst)}
    assert st[("in-0", "out-0")] == st[("in-1", "out-0")] == "excluded_estimated_endpoint"
    assert st[("in-0", "out-1")] == st[("in-1", "out-1")] == "train"


def test_absent_outgoing_gives_only_negatives():
    inst = instance("a", [("in-0", "out-0", line((0, 0), (4, 0))), ("in-1", "out-1", line((0, 3), (-4, 0)))], absent_out=("out-1",))
    rows = ml.pair_labels(reference(inst, {"in-0": "out-0", "in-1": None}, ml.EXPLORATORY_TIER), inst)
    assert [r["label"] for r in rows if r["incoming"] == "in-1"] == [0]


# ---------------------------------------------------------------- features

def F(rows, X, i, o, name):
    k = next(n for n, r in enumerate(rows) if r["incoming"] == i and r["outgoing"] == o)
    return X[k, ml.ALL_FEATURES.index(name)]


def test_feature_calculations_straight_runner():
    inst = instance("a", [("in-0", "out-0", line((0, 0), (5, 0))), ("in-1", "out-1", line((0, 4), (0, 0)))], gap=2.0)
    rows, X = ml.instance_features(inst)
    gap = 2.0 + 1 / 30
    assert F(rows, X, "in-0", "out-0", "gap_s") == pytest.approx(gap, abs=1e-3)
    assert F(rows, X, "in-0", "out-0", "dist_m") == pytest.approx(5 * gap, abs=0.2)
    assert F(rows, X, "in-0", "out-0", "speed_in") == pytest.approx(5.0, abs=1e-3)
    assert F(rows, X, "in-0", "out-0", "cv_fwd_err") < 0.2
    assert F(rows, X, "in-0", "out-0", "hermite_rms_acc") < 0.5
    assert F(rows, X, "in-0", "out-0", "heading_diff_deg") == pytest.approx(0, abs=1e-6)
    assert F(rows, X, "in-0", "out-0", "straight_in") == pytest.approx(1, abs=1e-9)
    assert F(rows, X, "in-0", "out-0", "turn_rate_in") == pytest.approx(0, abs=1e-6)
    assert F(rows, X, "in-1", "out-1", "both_moving") == 0          # the standing player
    assert F(rows, X, "in-0", "out-1", "cv_fwd_err") > 5            # running on past a standing player
    assert F(rows, X, "in-0", "out-0", "row_gap_cv_fwd_err") < 0 < F(rows, X, "in-0", "out-1", "row_gap_cv_fwd_err")
    assert F(rows, X, "in-0", "out-0", "reach_ratio") < 1
    assert np.isfinite(X).all() and X.shape == (4, len(ml.ALL_FEATURES))


def test_sharp_turn_and_unreachable():
    turn = lambda t: (5 * t, 0.0) if t < 0 else (0.0, 5 * t)          # noqa: E731
    inst = instance("a", [("in-0", "out-0", turn), ("in-1", "out-1", line((200, 0), (0, 0)))], gap=1.0)
    rows, X = ml.instance_features(inst)
    assert F(rows, X, "in-0", "out-0", "heading_diff_deg") == pytest.approx(90, abs=1)
    assert F(rows, X, "in-0", "out-0", "hermite_rms_acc") > 3
    assert not next(r for r in rows if r["incoming"] == "in-0" and r["outgoing"] == "out-1")["reachable"]


def test_features_ignore_tracklet_order_and_reject_interval_data():
    inst = instance("a", [("in-0", "out-0", line((0, 0), (4, 1))), ("in-1", "out-1", line((0, 3), (-4, 0))),
                          ("in-2", "out-2", line((2, 2), (1, -3)))])
    rows, X = ml.instance_features(inst)
    perm = dict(inst, incoming=inst["incoming"][::-1], outgoing=[inst["outgoing"][k] for k in (1, 2, 0)])
    rows2, X2 = ml.instance_features(perm)
    a = {(r["incoming"], r["outgoing"]): X[k] for k, r in enumerate(rows)}
    b = {(r["incoming"], r["outgoing"]): X2[k] for k, r in enumerate(rows2)}
    assert a.keys() == b.keys() and all(np.allclose(a[k], b[k]) for k in a)
    with pytest.raises(ValueError):
        ml.instance_features(dict(inst, task=TASK))
    with pytest.raises(ValueError):
        ml.instance_features(dict(inst, during=dict(t=[0.1], points=[[[0, 0, 1, 3]]])))


def test_reference_label_separation():
    assert list(inspect.signature(ml.instance_features).parameters) == ["inst", "cfg", "ft"]
    assert list(inspect.signature(ml.build_features).parameters) == ["instances", "cfg"]
    rng = np.random.default_rng(3)
    insts, refs = zip(*[random_encounter(rng, f"m-{k}") for k in range(6)])
    t1 = ml.match_table(dict(match_id="m", instances=list(insts), references=list(refs)))
    flipped = [dict(r, expected={k: ("out-0" if v == "out-1" else "out-1") for k, v in r["expected"].items()}) for r in refs]
    t2 = ml.match_table(dict(match_id="m", instances=list(insts), references=flipped))
    assert np.array_equal(t1["X"], t2["X"])                         # features never see the reference
    assert [lb["label"] for lb in t1["labels"]] == [1 - lb["label"] for lb in t2["labels"]]
    for f in ml.ALL_FEATURES:                                       # no identity, roster, match or frame field is a feature
        assert not set(f.split("_")) & {"jersey", "player", "slot", "team", "match", "frame", "side", "id", "time", "x", "y"}


# ---------------------------------------------------------------- assignment

def test_one_to_one_assignment():
    P = np.array([[0.9, 0.2], [0.8, 0.3]])
    raw = ml.assign(P, np.ones_like(P, bool))
    assert raw[0]["pick"] == 0 and raw[1]["pick"] == -1               # out-0 is claimed once; 0.3 < 0.5 is never taken
    assert ml.decide(raw[0], 0.5) == "MATCHED"
    assert ml.decide(raw[1], 0.5) == "UNCERTAIN"                     # reachable but no acceptable pair: not proof of absence


def test_unmatched_only_without_a_reachable_candidate():
    P = np.array([[0.05, 0.02], [0.9, 0.1]])
    R = np.array([[False, False], [True, True]])
    raw = ml.assign(P, R)
    assert ml.decide(raw[0], 0.5) == "UNMATCHED" and raw[0]["candidate"] == -1
    low = ml.assign(np.array([[0.05]]), np.array([[True]]))
    assert ml.decide(low[0], 0.5) == "UNCERTAIN"


def test_ambiguous_and_threshold_give_uncertain():
    P = np.array([[0.8, 0.79], [0.79, 0.8]])                        # the swapped assignment is almost as good
    raw = ml.assign(P, np.ones_like(P, bool))
    assert [r["pick"] for r in raw] == [0, 1] and all(ml.decide(r, 0.5) == "UNCERTAIN" for r in raw)
    clear = ml.assign(np.array([[0.95, 0.05], [0.05, 0.7]]), np.ones((2, 2), bool))
    assert [ml.decide(r, 0.8) for r in clear] == ["MATCHED", "UNCERTAIN"]
    assert [ml.decide(r, 0.6) for r in clear] == ["MATCHED", "MATCHED"]


# ---------------------------------------------------------------- models

def test_deterministic_training_and_persistence(tmp_path):
    import joblib

    t = table("m", 40, seed=1)
    X, y = ml._stack([t], ml._train_mask)
    for name in ("lr", "rf", "lr_ft"):
        m1, m2 = ml.fit(name, X, y), ml.fit(name, X, y)
        p1 = ml.predict(m1, name, t["X"])
        assert np.array_equal(p1, ml.predict(m2, name, t["X"]))
        joblib.dump(m1, tmp_path / f"{name}.joblib")
        assert np.array_equal(p1, ml.predict(joblib.load(tmp_path / f"{name}.joblib"), name, t["X"]))


def test_synthetic_engineering_check_beats_proximity():
    """Constant-velocity synthetic players: the physics model must rank true continuations above swaps. Synthetic only."""
    tr, te = table("a", 60, seed=2), table("b", 40, seed=5)
    X, y = ml._stack([tr], ml._train_mask)
    yt = np.array([lb["label"] for lb in te["labels"]])
    auc = {n: ml.pairwise_metrics(yt, ml.predict(ml.fit(n, X, y), n, te["X"]), te["rows"])["roc_auc"] for n in ("lr", "lr_distance")}
    assert auc["lr"] > 0.95 and auc["lr"] > auc["lr_distance"]


def test_training_uses_only_clean_high_tier_pairs():
    t = table("m", 9, seed=4, tiers=(ml.LABEL_TIER, ml.EXPLORATORY_TIER, ml.UNVERIFIED_TIER))
    m = ml._train_mask(t)
    assert {lb["tier"] for lb, k in zip(t["labels"], m) if k} == {ml.LABEL_TIER}
    assert sum(m) == sum(lb["tier"] == ml.LABEL_TIER for lb in t["labels"])


# ---------------------------------------------------------------- metrics

def test_pairwise_metrics_definitions():
    rows = [dict(instance_id="a", incoming=i) for i in ("x", "x", "y", "y")]
    m = ml.pairwise_metrics(np.array([1, 0, 1, 0]), np.array([0.9, 0.6, 0.4, 0.2]), rows, bins=2)
    assert (m["precision"], m["recall"], m["f1"]) == (0.5, 0.5, 0.5)
    assert m["roc_auc"] == 0.75 and m["top1_incoming"] == 1.0 and m["top1_n"] == 2
    assert m["ece"] == pytest.approx(0.5 * abs(0.3 - 0.5) + 0.5 * abs(0.75 - 0.5), abs=1e-4)


def test_encounter_summary_and_threshold_selection():
    refs = {"a": dict(expected={"in-0": "out-0", "in-1": "out-1"}), "b": dict(expected={"in-0": "out-0", "in-1": "out-1"})}
    assigned = {"a": dict(incoming=["in-0", "in-1"], outgoing=["out-0", "out-1"], raw=ml.assign(np.array([[0.95, 0.05], [0.05, 0.95]]), np.ones((2, 2), bool))),
                "b": dict(incoming=["in-0", "in-1"], outgoing=["out-0", "out-1"], raw=ml.assign(np.array([[0.1, 0.7], [0.7, 0.1]]), np.ones((2, 2), bool)))}
    s = ml.encounter_summary(ml.decisions(assigned, 0.5), refs)
    assert s == dict(coverage=1.0, false_association_rate=0.5, correct_match_rate=0.5, complete_assignment_accuracy=0.5)
    sel = ml.select_threshold(assigned, refs)
    assert sel["target_met"] and sel["threshold"] == 0.71 and sel["training_oof"]["false_association_rate"] == 0.0


# ---------------------------------------------------------------- split discipline

def test_grouped_split_integrity_and_leakage_guards():
    tables = [table(m, 12, seed=s) for m, s in (("m1", 1), ("m2", 2), ("m3", 3))]
    exp = ml.run_experiment(tables, models=("lr",))
    assert [f["heldout"] for f in exp["folds"]] == ["m1", "m2", "m3"]
    assert all(f["heldout"] not in f["train"] for f in exp["folds"])
    r = exp["models"]["lr"]
    for t in tables:                                                # every held-out instance is decided, by a model blind to it
        assert set(r["decisions"][t["match_id"]]) == set(t["refs"])
        n_train = sum(int(ml._train_mask(u).sum()) for u in tables if u is not t)
        assert r["fold_models"][t["match_id"]].named_steps["scale"].n_samples_seen_ == n_train
    with pytest.raises(ValueError):
        ml.run_experiment(tables[:2], models=("lr",))
    dup = dict(tables[1], match_id="m9", refs=tables[0]["refs"])
    with pytest.raises(ValueError):
        ml.run_experiment([tables[0], dup, tables[2]], models=("lr",))


def test_run_ml_refuses_non_development_and_existing_output(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    sp = tmp_path / "processed" / ml.SPLITS_DIR
    sp.mkdir(parents=True)
    (sp / "splits.json").write_text(json.dumps(dict(matches={"1": dict(split="development"), "2": dict(split="validation"),
                                                             "3": dict(split="test")})))
    with pytest.raises(IngestionError, match="development"):
        ml.run_ml(raw, ["1", "2"], out_root=tmp_path / "out")
    with pytest.raises(IngestionError, match="development"):
        ml.run_ml(raw, ["3"], out_root=tmp_path / "out")
    (tmp_path / "out").mkdir()
    with pytest.raises(IngestionError, match="never overwritten"):
        ml.run_ml(raw, ["1"], out_root=tmp_path / "out")
