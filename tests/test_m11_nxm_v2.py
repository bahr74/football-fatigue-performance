"""M13 regression tests: corrected N x M dataset (repeat flag, canonical encounters, nulls / new players, decoupled timing) and the
association benchmark (model view, null-aware one-to-one assignment, match-disjoint fit / calibration / test partitions)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from football_fatigue.m11_explorer import multiplayer as MP
from football_fatigue.m11_explorer import nxm_assoc_v2 as A
from football_fatigue.m11_explorer import nxm_compare as NC
from football_fatigue.m11_explorer import nxm_dataset_v2 as N2

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- repeated contact and canonical encounters

def _chains(rows):
    return pd.DataFrame(rows, columns=["period", "edge_chain_id", "slot_a", "slot_b", "start", "end"])


def test_v1_repeat_flag_defect_reproduced_order_in_chain_is_one_based():
    # the catalogue rank starts at 1, so v1's ``max(order_in_chain) > 0`` is true for every group
    s = pd.Series([10.0, 12.0, 30.0]).rank(method="first").astype(int)
    assert s.min() == 1 and bool(s.max() > 0)


def test_repeat_requires_a_distinct_earlier_chain_within_the_lookback():
    ch = _chains([(1, "c_now", 3, 7, 100.0, 104.0), (1, "c_old", 3, 7, 70.0, 75.0), (1, "c_far", 3, 7, 10.0, 12.0),
                  (1, "c_other_pair", 3, 9, 80.0, 82.0)])
    row = dict(members=[3, 7], period=1, encounter_start=100.0, encounter_chains=frozenset({"c_now"}))
    r = N2.repeat_info(row, ch)
    assert r["repeated_encounter"] and r["n_prior_encounters"] == 1 and r["prior_gap_s"] == pytest.approx(25.0)
    first = dict(row, encounter_start=70.0, encounter_chains=frozenset({"c_old"}))
    assert not N2.repeat_info(first, ch[ch["edge_chain_id"] != "c_far"])["repeated_encounter"]


def test_fragments_of_the_same_continuous_contact_are_not_repeats():
    # same pair, two catalogue rows within the chain-merge gap: one edge chain, part of the encounter -> not a repeat
    ch = _chains([(1, "c1", 3, 7, 50.0, 55.0)])
    row = dict(members=[3, 7], period=1, encounter_start=50.0, encounter_chains=frozenset({"c1"}))
    assert not N2.repeat_info(row, ch)["repeated_encounter"]
    # a chain ending less than the merge gap before the encounter is never a separate encounter
    ch2 = _chains([(1, "c0", 3, 7, 40.0, 49.5)])
    assert not N2.repeat_info(row, ch2)["repeated_encounter"]


def test_duplicate_catalogue_rows_collapse_into_one_encounter():
    groups = pd.DataFrame(dict(group_id=["g1", "g2", "g3", "g4"], start=[10.0, 11.5, 13.0, 200.0]))
    edges = pd.DataFrame(dict(edge_chain_id=["cA", "cA", "cB", "cC"], group_d0p00=["g1", "g2", "g3", "g4"]))
    cands = pd.DataFrame(dict(group_id=["g1", "g2", "g3", "g4"], period=[1, 1, 1, 1], t_centre=[10.5, 12.0, 13.4, 200.0],
                              members=[[1, 2], [1, 2], [1, 2, 5], [1, 2]], n=[2, 2, 3, 2]))
    c = N2.canonical_encounters(cands, edges, groups)
    e = dict(zip(c["group_id"], c["encounter_id"]))
    assert e["g1"] == e["g2"] == e["g3"] != e["g4"]                    # shared chain (g1, g2); overlapping window + 2 members (g3)
    assert c.set_index("group_id").loc["g1", "n_catalogue_groups"] == 3
    rep = N2.select_representatives(c.assign(_h=[3, 2, 1, 0]), np.ones(4, bool))
    assert len(rep) == 2 and set(rep["group_id"]) == {"g3", "g4"}      # largest N represents the encounter


# ---------------------------------------------------------------- scenario construction on a fake match

class _FakeTracks:
    """Straight-line players at 15 Hz; slot s moves along a distinct heading through the origin region."""

    def __init__(self, n=8):
        self.n = n

    def path(self, slot, period, a, b):
        t = np.arange(np.ceil(a * 25) / 25, b, 1 / 25)
        ang = 0.7 * slot
        return pd.DataFrame(dict(t=t, x=0.3 * slot + 2.0 * np.cos(ang) * t, y=0.2 * slot + 2.0 * np.sin(ang) * t, estimated=(slot % 2 == 0)))


def _spans(n=8):
    return pd.DataFrame(dict(slot=range(n), period=1, chain=0, start=-100.0, end=100.0))


def _positions_at(per, t):
    return pd.DataFrame(dict(slot=range(8), x=[0.3 * s + 2 * np.cos(0.7 * s) * t for s in range(8)],
                             y=[0.2 * s + 2 * np.sin(0.7 * s) * t for s in range(8)]))


def _ev(members):
    return dict(match_id=3812, encounter_id="enc1", group_id="g1", period=1, t_centre=0.0, members=members, n=len(members),
                max_simultaneous=len(members), group_players=len(members), n_sync_crossings=0, frac_edges_both_estimated=0.0,
                estimated_comotion=False, label_risk=False, repeated_encounter=False, n_prior_encounters=0, prior_gap_s=None,
                n_catalogue_groups=1, n_candidates=1)


@pytest.mark.parametrize("variant", N2.VARIANTS)
def test_scenarios_keep_one_to_one_truth_nulls_and_anonymous_new_tracks(variant):
    d = N2.scenario(_FakeTracks(), _spans(), _ev([0, 1, 3, 5]), variant, _positions_at, {s: 1000 + s for s in range(8)})
    assert d is not None and MP.validate(d) == []
    tgt = [v for v in d["truth"].values() if v is not None]
    assert len(tgt) == len(set(tgt))
    nulls = [k for k, v in d["truth"].items() if v is None]
    assert sorted(nulls) == sorted(d["meta"]["null_incoming"])
    new_ids = {n["id"] for n in d["meta"]["new_outgoing"]}
    assert not new_ids & set(tgt)                                       # new players never inherit an incoming identity
    assert len(d["outgoing"]) == len(d["incoming"]) - len(nulls) + len(new_ids)
    for o in d["outgoing"]:
        assert set(o) == {"id", "t", "x", "y"} and o["id"].startswith("out-") and not o["id"][4:].isdigit()
    if variant in ("null2",):
        assert len(nulls) == 2
    if variant == "new2":
        assert len(new_ids) == 2
    assert d["provenance"] == "provider-derived" and d["human_reviewed"] is False


def test_scenarios_are_reproducible_for_identical_seeds():
    a = N2.scenario(_FakeTracks(), _spans(), _ev([0, 1, 3]), "null_new", _positions_at)
    b = N2.scenario(_FakeTracks(), _spans(), _ev([0, 1, 3]), "null_new", _positions_at)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_two_player_null2_is_not_constructible_and_n_gt_m_exists_for_large_groups():
    assert N2.scenario(_FakeTracks(), _spans(), _ev([0, 1]), "null2", _positions_at) is None
    d = N2.scenario(_FakeTracks(), _spans(), _ev([0, 1, 2, 3, 4]), "null2", _positions_at)
    assert len(d["incoming"]) == 5 and len(d["outgoing"]) == 3


def test_disappearance_and_reappearance_times_are_independent_of_identity():
    dt_true, dt_wrong = [], []
    for k in range(60):
        ev = dict(_ev([0, 1, 2, 3]), encounter_id=f"e{k}")
        d = N2.scenario(_FakeTracks(), _spans(), ev, "async", _positions_at)
        for i in d["incoming"]:
            for o in d["outgoing"]:
                (dt_true if d["truth"][i["id"]] == o["id"] else dt_wrong).append(o["t"][0] - i["t"][-1])
    assert abs(np.mean(dt_true) - np.mean(dt_wrong)) < 0.05 and abs(np.std(dt_true) - np.std(dt_wrong)) < 0.05


# ---------------------------------------------------------------- model view, assignment, partitions, calibration

def test_model_view_strips_truth_meta_labels_and_features_ignore_them():
    d = MP.generate(5, 3)
    v = A.view(d)
    assert set(v) == {"incoming", "outgoing"} and all(set(t) == {"id", "t", "x", "y"} for t in v["incoming"] + v["outgoing"])
    d2 = copy.deepcopy(d)
    d2["truth"] = {k: None for k in d2["truth"]}
    d2["meta"] = {"hidden": "x"}
    for t in d2["incoming"]:
        t["label"] = "other"
    f1, f2 = A.featurize(d), A.featurize(d2)
    assert all(np.array_equal(f1[k], f2[k]) for k in ("X", "SC", "TX"))


def test_assign_v_is_one_to_one_and_respects_vector_nulls():
    rng = np.random.default_rng(0)
    for _ in range(20):
        c = rng.random((4, 6)) * 5
        p, _ = A.assign_v(c, rng.random(4) * 3, 2.0)
        assert len({a for a, _ in p}) == len(p) == len({b for _, b in p})
    c = np.array([[1.0, 5.0], [5.0, 1.0]])
    assert A.assign_v(c, np.array([0.1, 9.0]), 9.0)[0] == [(0, 0), (1, 1)]   # null 0.1 + orphaned column 9.0 > link 1.0
    assert A.assign_v(c, np.array([0.1, 9.0]), 0.0)[0] == [(1, 1)]           # cheap row-0 null and free column -> null
    ch, mg = A.decisions(c, None, None)
    assert ch == [0, 1] and mg[0] == pytest.approx(8.0)
    ch, mg = A.decisions(np.array([[1.0], [2.0], [9.0]]), 3.0, 3.0)
    assert ch == [0, -1, -1] and mg[2] == pytest.approx(8.0)   # forcing row 2 to link: 9 + 3 + 3 (row 0 displaced) - (1 + 3 + 3)


def test_score_counts_nulls_and_new_players():
    d = dict(incoming=[{"id": "a"}, {"id": "b"}], outgoing=[{"id": "x"}, {"id": "y"}], truth={"a": "x", "b": None})
    r = A.score(d, [0, -1])
    assert r["exact"] == 1 and r["null_tp"] == 1 and r["new_true"] == 1 and r["new_tp"] == 1
    r = A.score(d, [1, 0])
    assert r["exact"] == 0 and r["id_switches"] == 1 and r["new_pred"] == 0


def test_partitions_are_match_disjoint_and_cover_every_scenario_once_as_test():
    fold = np.repeat(np.arange(8), 5)
    seen = []
    for f in range(8):
        fi, ca, te = A.partitions(fold, f)
        assert not (set(fi) & set(ca) or set(fi) & set(te) or set(ca) & set(te))
        assert set(fold[te]) == {f} and set(fold[ca]) == {(f + 1) % 8, (f + 2) % 8} and len(fi) + len(ca) + len(te) == len(fold)
        seen += list(te)
    assert sorted(seen) == list(range(len(fold)))


def test_calibration_metrics_and_threshold_use_only_given_data():
    p, y = np.array([0.9, 0.9, 0.1, 0.1]), np.array([1, 1, 0, 0])
    assert A.ece(p, y) == pytest.approx(0.1) and A.brier(p, y) == pytest.approx(0.01)
    conf, ok = np.array([0.99, 0.9, 0.8, 0.3]), np.array([1, 1, 0, 1])
    assert A.threshold_for_accuracy(conf, ok, 0.95) == pytest.approx(0.9)
    s = A.selective(conf, ok, np.ones(4), 0.9)
    assert s["coverage"] == 0.5 and s["selective_accuracy"] == 1.0
    rc = A.risk_coverage(conf, ok)
    assert rc["1.0"] == 0.75 and rc["0.4"] == 1.0


# ---------------------------------------------------------------- historical outputs untouched

@pytest.mark.parametrize("folder", ["m11-nxm-exploratory-v1", "m11-nxm-solver-comparison-v1", "m11-review-priority-v1"])
def test_historical_m12_outputs_match_their_manifests(folder):
    from football_fatigue.m11_phase2 import common as C
    root = ROOT / "data" / "processed" / folder
    if not (root / "manifest.json").exists():
        pytest.skip(f"{folder} not present")
    man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for rel, h in man["outputs"].items():
        assert C.sha256_file(root / rel) == h, rel


def test_v1_scenario_flag_is_the_documented_defect_and_v2_flag_is_not_constant():
    p1 = ROOT / "data" / "processed" / "m11-nxm-exploratory-v1" / "index.parquet"
    p2 = ROOT / "data" / "processed" / N2.OUT / "index.parquet"
    if not (p1.exists() and p2.exists()):
        pytest.skip("datasets not built")
    assert pd.read_parquet(p1)["repeated_interaction"].all()
    r = pd.read_parquet(p2).drop_duplicates("encounter_id")["repeated_encounter"]
    assert 0.05 < r.mean() < 0.95


# ---------------------------------------------------------------- review statistics (no review is ever created here)

def test_review_statistics():
    from football_fatigue.m11_explorer import review_v2 as RV
    lo, hi = RV.clopper_pearson(0, 30)
    assert lo == 0.0 and hi == pytest.approx(0.1157, abs=1e-3)
    assert RV.n_required(0.05, 0) == 72 and RV.n_required(0.05, 2) > RV.n_required(0.05, 1) > 72
    assert RV.wilson(0, 30)[1] == pytest.approx(0.1135, abs=1e-3)
    assert RV.cohen_kappa(list("aabb"), list("aabb")) == 1.0 and RV.cohen_kappa(list("abab"), list("aabb")) == pytest.approx(0.0)
