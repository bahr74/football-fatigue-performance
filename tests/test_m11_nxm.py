"""N x M association tools (synthetic only): null-aware assignment, label-free pair features, solver equivalence, and the Session A
side of the A / B contract (observed-only inputs, anonymous outgoing, result invariants)."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from football_fatigue.m11_explorer import ab_contract as AB
from football_fatigue.m11_explorer import multiplayer as MP
from football_fatigue.m11_explorer import nxm_compare as NC


def test_assign2_respects_one_to_one_nulls_and_infeasible_pairs():
    c = np.array([[1.0, 5.0, np.inf], [5.0, 1.0, 9.0]])
    assert sorted(NC.assign2(c, None, None)) == [(0, 0), (1, 1)]
    assert NC.assign2(c, 0.4, 0.4) == []                                 # every link costs more than leaving both tracks unmatched
    assert sorted(NC.assign2(c, 3.0, 3.0)) == [(0, 0), (1, 1)]
    one = NC.assign2(np.array([[np.inf, np.inf]]), 2.0, 2.0)
    assert one == []
    p = NC.assign2(np.random.default_rng(0).random((4, 6)), 0.5, 0.5)
    assert len({a for a, _ in p}) == len(p) == len({b for _, b in p})


def test_link_margin_is_cost_of_best_alternative_minus_optimum():
    c = np.array([[1.0, 4.0], [4.0, 2.0]])
    m = NC.link_margins(c, [(0, 0), (1, 1)], None, None)
    assert m[(0, 0)] == pytest.approx(8.0 - 3.0) and m[(1, 1)] == pytest.approx(5.0)


def test_pair_features_never_read_the_truth():
    d = MP.generate(3, 5)
    X, idx = NC.pair_features(d, texture=True)
    d2 = copy.deepcopy(d)
    d2["truth"] = {k: None for k in d2["truth"]}
    d2["meta"] = {"anything": 1}
    X2, idx2 = NC.pair_features(d2, texture=True)
    assert idx == idx2 and np.array_equal(X, X2) and X.shape[1] == len(NC.FEATURES) + len(NC.TEXTURE)


def test_forward_cost_reproduces_the_original_solver():
    for lv in (1, 3, 5):
        for s in range(15):
            d = MP.generate(lv, s)
            c = NC.motion_costs(d)["fwd"]
            assert NC.decide(d, np.where(c <= MP.GATE, c, np.inf), MP.GATE, MP.GATE, False).links == MP.solve(d).links


def test_bidirectional_cost_is_symmetric_in_time_reversal():
    d = MP.generate(2, 1)
    r = copy.deepcopy(d)
    rev = lambda tr: dict(tr, t=[-v for v in tr["t"]][::-1], x=tr["x"][::-1], y=tr["y"][::-1])  # noqa: E731
    r["incoming"], r["outgoing"] = [rev(t) for t in d["outgoing"]], [rev(t) for t in d["incoming"]]
    a, b = NC.motion_costs(d)["bi"], NC.motion_costs(r)["bi"]
    ok = np.isfinite(a)
    assert np.allclose(a[ok], b.T[ok], rtol=1e-6, atol=1e-6)


def test_contract_round_trip_is_valid_and_ignores_interpolated_samples():
    req, truth = AB.synthetic_request(2, 3)
    res = AB.respond(req)
    assert AB.check_result(res, req) == [] and res["synthetic_example"] is True
    assert res["input_contract"]["domain_check"]["inside_trained_envelope"]
    clean, _ = AB.synthetic_request(2, 3, interpolated_gap=False)
    assert AB.respond(clean)["decision"] == res["decision"]                 # the (1 km away) interpolated samples were ignored
    assert all(truth[a["incoming"]] == a["outgoing"] for a in res["decision"]["assignments"]) or res["decision"]["status"] != "assigned"
    json.dumps(res)


def test_contract_refuses_prefilled_outgoing_identity_and_wrong_schema():
    req, _ = AB.synthetic_request(1, 0)
    bad = copy.deepcopy(req)
    bad["outgoing_candidates"][0]["identity"]["global_id"] = "home-01"
    with pytest.raises(AB.ContractError):
        AB.respond(bad)
    bad = dict(req, schema="other")
    with pytest.raises(AB.ContractError):
        AB.respond(bad)


def test_contract_is_advisory_outside_the_envelope_and_abstains_without_calibration():
    req, _ = AB.synthetic_request(2, 4, sigma_m=0.6)
    res = AB.respond(req)
    assert not res["input_contract"]["domain_check"]["inside_trained_envelope"] and not res["identity_effect"]["propagates_global_id"]
    assert AB.check_result(res, req) == []
    nocal = copy.deepcopy(req)
    nocal["camera"]["calibration"]["available"] = False
    for t in nocal["incoming"] + nocal["outgoing_candidates"]:
        t["observations"]["pitch"] = None
    r2 = AB.respond(nocal)
    assert r2["decision"]["status"] == "abstain" and AB.check_result(r2, nocal) == []


@pytest.mark.parametrize("level,seed", [(1, 0), (3, 2), (5, 7), (7, 1)])
def test_contract_invariants_hold_across_difficulty_levels(level, seed):
    req, _ = AB.synthetic_request(level, seed)
    assert AB.check_result(AB.respond(req), req) == []
