"""Focused synthetic tests for Phase 2 (deterministic baselines, assignment, guard, review sampler). No real data is read."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football_fatigue.m11_phase2 import baselines as B
from football_fatigue.m11_phase2 import common as C

DT = 1.0 / 15.0
T = 30


def make_x(paths, gap=0.8, drop_in=None, noise=0.0, seed=0):
    """Stage 3-shaped inputs from analytic paths p(t) (t relative to the gap centre). paths: [incoming fns], outgoing in the same
    order (MATCHED) unless swapped later."""
    rng = np.random.default_rng(seed)
    in_dt = (np.arange(-T, 0) * DT).astype(np.float32)
    out_dt = (np.arange(1, T + 1) * DT).astype(np.float32)
    tin = in_dt - gap / 2
    tout = out_dt + gap / 2
    pin = np.stack([np.stack([f(t) for t in tin]) for f in paths])[None].astype(np.float32)
    pout = np.stack([np.stack([f(t) for t in tout]) for f in paths])[None].astype(np.float32)
    pin += rng.normal(0, noise, pin.shape).astype(np.float32)
    pout += rng.normal(0, noise, pout.shape).astype(np.float32)
    im = np.ones((1, 2, T), bool)
    om = np.ones((1, 2, T), bool)
    if drop_in is not None:
        im[0, :, drop_in] = False
    return dict(in_pos=pin, in_mask=im, in_dt=in_dt[None], out_pos=pout, out_mask=om, out_dt=out_dt[None], gap_s=np.array([gap], np.float32))


def swap_out(x):
    z = {k: v.copy() for k, v in x.items()}
    z["out_pos"] = z["out_pos"][:, ::-1].copy()
    z["out_mask"] = z["out_mask"][:, ::-1].copy()
    return z


def lin(p0, v):
    p0, v = np.asarray(p0, float), np.asarray(v, float)
    return lambda t: p0 + v * t


CROSS = [lin([-0.0, -0.3], [3.0, 1.0]), lin([0.0, 0.3], [3.0, -1.0])]          # crossing paths, 0.6 m apart at t = 0... crossing


def margins(x, sa=8.0, sm=0.05):
    c = B.all_costs(x, sa, sm)
    return {k: B.margin(c[k])[0] for k in B.METHODS}


def test_cv_and_kalman_recover_crossing_while_nearest_fails():
    x = make_x(CROSS)
    m = margins(x)
    for k in ("cv_fwd", "cv_bidir", "ca_fwd", "ca_bidir", "kf_fwd", "kf_bidir"):
        assert m[k] > 0, k                                   # MATCHED is correct
    # endpoints: players have crossed, so the nearest outgoing endpoint belongs to the other player
    assert m["nearest"] < 0


def test_outgoing_swap_negates_every_margin_exactly():
    x = make_x(CROSS, noise=0.03, seed=1)
    a, b = margins(x), margins(swap_out(x))
    for k in B.METHODS:
        assert a[k] == pytest.approx(-b[k], abs=1e-9), k


def test_parallel_runs_are_separated_by_position():
    # identical velocity, 0.4 m apart, no crossing: every motion model separates them by position
    x = make_x([lin([0, -0.2], [2, 0]), lin([0, 0.2], [2, 0])])
    m = margins(x)
    assert m["cv_fwd"] > 0 and m["kf_bidir"] > 0 and m["nearest"] > 0


def test_masked_incoming_steps_do_not_break_fits():
    x = make_x(CROSS, drop_in=list(range(0, 20)))            # only the last 10 incoming samples observed
    m = margins(x)
    assert np.isfinite(list(m.values())).all()
    assert m["cv_fwd"] > 0 and m["kf_fwd"] > 0


def test_constant_acceleration_is_clipped():
    # a strongly curving incoming track: unconstrained quadratic extrapolation goes further than the clipped one
    f = lambda t: np.array([0.5 * 20.0 * (t + 1.5) ** 2, 0.0]) if t < 0 else np.array([0.5 * 20.0 * 1.5 ** 2 + 30 * t, 0.0])  # noqa: E731
    x = make_x([f, lin([0, 5], [0, 0])])
    I, O, _ = B.tracks(x)
    ai, ri, n, s = B.polyfit(I, True, B.FIT_CA_S, 2)
    cvi, *_ = B.polyfit(I, True, B.FIT_CA_S, 1)
    k, bad = B._constrain(ai, n, s, cvi)
    a = 2 * k[0, 0, 2]
    assert np.hypot(*a) <= B.A_MAX + 1e-9
    assert not bad[0, 0]


def test_hungarian_matches_brute_force_on_2x2_and_null_option():
    rng = np.random.default_rng(3)
    for _ in range(200):
        c = rng.uniform(0, 10, (2, 2))
        pairs = B.assign(c)
        best = 0 if c[0, 0] + c[1, 1] <= c[0, 1] + c[1, 0] else 1
        assert pairs == ([(0, 0), (1, 1)] if best == 0 else [(0, 1), (1, 0)])
    # null option: a link above the null cost is never used
    c = np.array([[1.0, 50.0], [50.0, 60.0]])
    assert B.assign(c, null_cost=20.0) == [(0, 0)]
    # 3 x 2 (extra incoming track) works and is one-to-one
    c = np.array([[1.0, 5.0], [5.0, 1.0], [0.5, 0.5]])
    p = B.assign(c)
    assert len(p) == 2 and len({j for _, j in p}) == 2


def test_gating_abstains_when_no_full_assignment_fits():
    d2 = np.array([[[1.0, 100.0], [100.0, 100.0]]])
    nll = d2.copy()
    assert B.gated_decision(d2, nll)[0] == -1
    d2 = np.array([[[1.0, 100.0], [100.0, 2.0]]])
    assert B.gated_decision(d2, d2)[0] == 0
    d2 = np.array([[[100.0, 1.0], [2.0, 100.0]]])
    assert B.gated_decision(d2, d2)[0] == 1


def test_kalman_selection_is_label_free_and_picks_measurement_noise_scale():
    xs = [make_x([lin([0, -1], [2, 1]), lin([0, 1], [1, -2])], noise=0.1, seed=s) for s in range(8)]
    x = {k: np.concatenate([z[k] for z in xs]) for k in xs[0]}
    I, O, _ = B.tracks(x)
    sel = B.kf_select(I, O, sas=(0.5, 4.0), sms=(0.02, 0.1, 0.5))
    assert sel["sigma_m"] == 0.1                            # recovers the simulated observation noise


def test_baselines_never_read_the_hidden_interval_or_labels():
    x = make_x(CROSS)
    assert set(x) == set(("in_pos", "in_mask", "in_dt", "out_pos", "out_mask", "out_dt", "gap_s"))
    c1 = B.all_costs(x, 8.0, 0.05)
    x2 = dict(x)
    x2["y"] = np.array([1])                                  # an extra label field is ignored
    c2 = B.all_costs(x2, 8.0, 0.05)
    for k in B.METHODS:
        np.testing.assert_array_equal(c1[k], c2[k])


def test_guard_refuses_protected_and_unknown_matches(monkeypatch):
    allowed = C.allowed()
    assert len(allowed) == 32
    for bad in ("3822", "10517", "999999"):
        with pytest.raises(SystemExit):
            C.checked([bad])
    called = []
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: called.append(a) or pd.DataFrame())
    with pytest.raises(SystemExit):
        C.load_table("edges", ["3824"])
    assert not called                                        # nothing was opened before the refusal


def test_out_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "PROC", tmp_path)
    o = C.Out("x-v1")
    o.json("a.json", {"k": 1})
    with pytest.raises(SystemExit):
        o.json("a.json", {"k": 2})
    with pytest.raises(SystemExit):
        C.Out("x-v1")                                         # non-empty versioned folder is refused


def test_review_selection_is_deterministic_and_spread():
    from football_fatigue.m11_phase2 import review as V
    rng = np.random.default_rng(0)
    n = 4000
    E = pd.DataFrame(dict(match_id=rng.integers(1, 33, n), period=1, radius_m=np.where(rng.random(n) < 0.8, 0.5, 2.0),
                          slot_a=rng.integers(0, 10, n), slot_b=rng.integers(10, 20, n), edge_id=[f"e{i}" for i in range(n)],
                          status_composition=rng.choice(["both_visible", "mixed", "both_estimated"], n), label_risk=rng.random(n) < 0.1,
                          reliability=np.where(rng.random(n) < 0.8, "nominal", "reduced"), boundary_uncertain=rng.random(n) < 0.2,
                          estimated_comotion=rng.random(n) < 0.02, n_sync_crossings_in_edge=rng.integers(0, 4, n),
                          duration_s=rng.exponential(0.5, n), cv_tier=rng.choice(["cv_confusable", "marginal", "cv_separable"], n),
                          cv_margin_m=rng.normal(0.3, 1, n), d_min=rng.uniform(0, 2, n), group_d0p00=np.arange(n)))
    G = pd.DataFrame(dict(match_id=E.match_id, period=1, radius_m=0.5, link_s=0.0, group_id=np.arange(n), n_players=2,
                          max_simultaneous_players=rng.choice([2, 3, 4], n, p=[0.9, 0.07, 0.03])))
    a = V.select(E, G, set())
    b = V.select(E.sample(frac=1, random_state=1).reset_index(drop=True), G, set())
    assert list(a.edge_id) == list(b.edge_id)                 # independent of row order
    assert a.edge_id.is_unique
    assert (a.groupby("stratum").match_id.nunique() == a.groupby("stratum").size()).all()   # one per match per stratum


# ---------------------------------------------------------------- Session A / B interface examples (docs, synthetic)

EX_DIR = C.ROOT / "docs" / "research" / "m11_session_ab_interface_v1_examples"


def _load_examples():
    import json
    return {p.name: json.loads(p.read_text(encoding="utf-8")) for p in sorted(EX_DIR.glob("*.json"))}


def test_interface_examples_satisfy_invariants():
    ex = _load_examples()
    assert len(ex) == 3 and all(v["synthetic_example"] is True for v in ex.values())
    req = ex["01_candidate_event_request.json"]
    assert all(c["identity"]["global_id"] is None and c["identity"]["identity_status"] == "anonymous" for c in req["outgoing_candidates"])
    assert req["trigger"]["occlusion_candidate"]["verified_visual_occlusion"] is None
    for name in ("02_reassociation_response.json", "03_abstain_and_delayed.json"):
        r = ex[name]
        d = r["decision"]
        inc, out = r["candidates_scored"]["incoming"], r["candidates_scored"]["outgoing"]
        a = d["assignments"]
        assert len({x["incoming"] for x in a}) == len(a) and len({x["outgoing"] for x in a}) == len(a)
        assert all(x["incoming"] in inc and x["outgoing"] in out for x in a)
        if not d["calibrated"]:
            assert all(x["probability"] is None for x in a)
        if d["status"] == "abstain":
            assert not a and len(d["unmatched_incoming"]) == len(inc) and len(d["unmatched_outgoing"]) == len(out)
        if a:
            cost = {(s["incoming"], s["outgoing"]): s["value"] for s in r["link_scores"]}
            chosen = sum(cost[(x["incoming"], x["outgoing"])] for x in a)
            import itertools
            alts = [sum(cost[(i, o)] for i, o in zip(inc, perm)) for perm in itertools.permutations(out, len(inc))]
            alts = sorted(v for v in alts if abs(v - chosen) > 1e-9)
            assert all(x["margin"] == pytest.approx(alts[0] - chosen, abs=1e-6) for x in a)
