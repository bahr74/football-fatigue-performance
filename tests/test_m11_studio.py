"""M11 Studio back end: window events, self-crossings, N x M association, review queue, leak fixes, fair-benchmark infrastructure,
figure builders and headless app pages. Synthetic unless marked ``data`` (needs the development catalogue / outputs)."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import pytest

from football_fatigue.m11_explorer import events as EV
from football_fatigue.m11_explorer import multiplayer as MP
from football_fatigue.m11_explorer import review as RV
from football_fatigue.m11_explorer import selfcross as SX
from football_fatigue.m11_phase2 import common as C

data = pytest.mark.skipif(not (C.CATALOGUE / "matches").exists(), reason="development catalogue not present")


# ---------------------------------------------------------------- synthetic catalogue tables

def _edge(eid, a, b, s, e, period=1, R=0.5, d0=None, d1=None, conc=0, seq=0, chain=None):
    return dict(edge_id=eid, edge_chain_id=chain or f"c{a}{b}", period=period, radius_m=R, slot_a=a, slot_b=b, player_id_a=a, player_id_b=b,
                start=s, end=e, duration_s=e - s, t_star=(s + e) / 2, d_min=0.3, x_a=0.0, y_a=0.0, x_b=0.3, y_b=0.0, rel_speed_m_s=1.0,
                rel_heading_deg=10.0, team_relation="opposite_teams", status_composition="both_visible", est_frac_a=0.0, est_frac_b=0.0,
                cv_tier="marginal", boundary_uncertain=False, reliability="nominal", reliability_reasons="", label_risk=False,
                label_risk_reasons="", estimated_comotion=False, n_crossings_in_edge=0, n_sync_crossings_in_edge=0, n_concurrent_other=conc,
                n_sequential_other=seq, group_d0p00=d0, group_d1p00=d1, order_in_chain=0)


def _tables():
    E = pd.DataFrame([
        _edge("e1", 1, 2, 10.0, 10.5, d0="g01", d1="h1", seq=1, chain="c12"),        # pair 1-2 meets ...
        _edge("e2", 1, 2, 13.0, 13.4, d0="g02", d1="h1", seq=1, chain="c12b"),       # ... separates and meets again
        _edge("e3", 1, 3, 11.0, 11.6, d0="g03", d1="h1", seq=2),                     # player 1 with a second partner (sequential)
        _edge("e4", 4, 5, 12.0, 12.8, d0="g04", d1="h2", conc=2),                   # simultaneous three-player group 4-5-6
        _edge("e5", 5, 6, 12.1, 12.7, d0="g04", d1="h2", conc=2),
        _edge("e6", 4, 6, 12.2, 12.6, d0="g04", d1="h2", conc=2),
        _edge("e7", 1, 2, 30.0, 30.4, d0="g05", d1="h3"),                           # outside the window
        _edge("e8", 1, 2, 10.0, 10.5, R=1.0, d0="x", d1="y"),                        # another radius
    ])
    X = pd.DataFrame([dict(period=1, slot_a=1, slot_b=2, ta=10.2, tb=10.3, x=0.0, y=0.0, kind="proper", chain_a=1, chain_b=2, abs_dt_s=0.1,
                           synchronous=True, d_sync_m=0.2, crossing_id="x1", within_r0p50=True),
                      dict(period=1, slot_a=1, slot_b=2, ta=13.1, tb=13.8, x=1.0, y=0.0, kind="proper", chain_a=1, chain_b=2, abs_dt_s=0.7,
                           synchronous=False, d_sync_m=0.9, crossing_id="x2", within_r0p50=False),
                      dict(period=1, slot_a=2, slot_b=7, ta=14.0, tb=17.0, x=5.0, y=0.0, kind="proper", chain_a=2, chain_b=7, abs_dt_s=3.0,
                           synchronous=False, d_sync_m=9.0, crossing_id="x3", within_r0p50=False)])
    G = pd.DataFrame([dict(period=1, radius_m=0.5, link_s=0.0, group_id=g, n_players=n, players=np.array(p), start=s, end=e, n_edges=k,
                           max_simultaneous_players=ms, n_merges=0, n_splits=0)
                      for g, n, p, s, e, k, ms in (("g01", 2, [1, 2], 10.0, 10.5, 1, 2), ("g02", 2, [1, 2], 13.0, 13.4, 1, 2),
                                                   ("g03", 2, [1, 3], 11.0, 11.6, 1, 2), ("g04", 3, [4, 5, 6], 12.0, 12.8, 3, 3),
                                                   ("g05", 2, [1, 2], 30.0, 30.4, 1, 2))]
                     + [dict(period=1, radius_m=0.5, link_s=1.0, group_id="h1", n_players=3, players=np.array([1, 2, 3]), start=10.0, end=13.4,
                             n_edges=3, max_simultaneous_players=np.nan, n_merges=1, n_splits=1),
                        dict(period=1, radius_m=0.5, link_s=1.0, group_id="h2", n_players=3, players="4;5;6", start=12.0, end=12.8,
                             n_edges=3, max_simultaneous_players=np.nan, n_merges=0, n_splits=0)])
    return dict(edges=E, crossings=X, groups=G, group_timeline=pd.DataFrame(), nodes=pd.DataFrame())


def test_window_returns_every_event_without_first_only_filtering():
    W = EV.window_events("x", 1, 9.0, 15.0, 0.5, tables=_tables())
    assert sorted(W.edges["edge_id"]) == ["e1", "e2", "e3", "e4", "e5", "e6"]          # e7 outside, e8 other radius
    assert W.edges["edge_id"].is_unique and W.crossings["crossing_id"].is_unique
    pair12 = W.pairs[(W.pairs["slot_a"] == 1) & (W.pairs["slot_b"] == 2)].iloc[0]
    assert pair12["n_edges"] == 2 and pair12["n_chains"] == 2                            # repeated same-pair encounter
    p1 = W.players.set_index("slot").loc[1]
    assert p1["n_partners"] == 2 and p1["n_edges"] == 3                                  # several events of one player
    assert set(W.crossings["event_type"]) == {"crossing_synchronous", "crossing_near_synchronous", "crossing_asynchronous"}
    c = W.counts()
    assert c["repeated_pair"] == 1 and c["sequential_player"] >= 1 and c["simultaneous_group"] == 1


def test_connected_group_size_is_not_simultaneous_size():
    W = EV.window_events("x", 1, 9.0, 15.0, 0.5, tables=_tables())
    g = W.groups.set_index("group_id")
    assert g.loc["h1", "n_players"] == 3 and g.loc["h1", "max_simultaneous_players"] == 2 and bool(g.loc["h1", "inflated"])
    assert g.loc["h2", "max_simultaneous_players"] == 3 and not bool(g.loc["h2", "inflated"])      # genuine three-player simultaneity
    assert W.counts()["connected_group"] == 2 and W.counts()["simultaneous_group"] == 1


def test_group_players_parses_catalogue_strings_and_arrays():
    assert EV.group_players("10;21") == [10, 21] and EV.group_players(np.array([4, 5])) == [4, 5]
    assert len(EV.window_events("x", 1, 9.0, 15.0, 0.5, tables=_tables(), slots={6}).groups) == 2


def test_overlapping_windows_share_stable_ids_and_filters():
    T = _tables()
    a = EV.window_events("x", 1, 9.0, 12.0, 0.5, tables=T)
    b = EV.window_events("x", 1, 11.5, 15.0, 0.5, tables=T)
    both = set(a.edges["edge_id"]) & set(b.edges["edge_id"])
    assert "e3" in both                                                                   # same event, same id in both windows
    assert len(EV.window_events("x", 1, 9.0, 15.0, 0.5, tables=T, slots={4}).edges) == 2
    assert len(EV.window_events("x", 1, 9.0, 15.0, 0.5, tables=T, min_duration_s=0.7).edges) == 1
    assert len(EV.window_events("x", 1, 9.0, 15.0, 0.5, tables=T, crossing_types=("crossing_synchronous",)).crossings) == 1
    with pytest.raises(ValueError):
        EV.window_events("x", 1, 15.0, 9.0, 0.5, tables=T)


def test_window_query_is_fast_on_a_large_table():
    rng = np.random.default_rng(0)
    rows = [_edge(f"e{i}", int(rng.integers(0, 11)), int(rng.integers(11, 22)), s, s + 0.5, d0=f"g{i}", d1=f"h{i}")
            for i, s in enumerate(rng.uniform(0, 3000, 20000))]
    T = _tables()
    T["edges"] = pd.DataFrame(rows)
    t = time.perf_counter()
    W = EV.window_events("x", 1, 1000, 1010, 0.5, tables=T)
    assert time.perf_counter() - t < 2.0 and W.edges["edge_id"].is_unique


# ---------------------------------------------------------------- self-crossings

def _loop():
    th = np.linspace(0, 2 * np.pi, 200)
    x = np.r_[np.linspace(-10, 0, 100), 3 * np.sin(th), np.linspace(0, 10, 100)]
    y = np.r_[np.zeros(100), 3 - 3 * np.cos(th), np.zeros(100)]
    return np.arange(len(x)) / 25, x, y


def test_selfcross_loop_figure_eight_and_spiral():
    d = SX.chain_self_crossings(*_loop())
    assert len(d) == 1 and d.iloc[0]["dt_s"] > 1 and d.iloc[0]["path_m"] > 2 and abs(d.iloc[0]["x"]) < 0.2
    th = np.linspace(0, 2 * np.pi, 400)
    assert len(SX.chain_self_crossings(np.arange(400) / 25, 6 * np.sin(th), 3 * np.sin(2 * th))) == 1
    t = np.arange(0, 20, 1 / 25)
    r = 3 + 0.1 * t
    assert len(SX.chain_self_crossings(t, r * np.cos(0.5 * t), r * np.sin(0.5 * t))) == 0


def test_selfcross_ignores_adjacent_segments_jitter_and_tiny_wiggles():
    zig = np.c_[np.arange(50) * 0.5, np.tile([0.0, 0.3], 25)]                               # shared endpoints only
    assert len(SX.chain_self_crossings(np.arange(50) / 25, zig[:, 0], zig[:, 1])) == 0
    rng = np.random.default_rng(1)
    n = 3000
    assert len(SX.chain_self_crossings(np.arange(n) / 25, rng.normal(0, 0.05, n), rng.normal(0, 0.05, n))) == 0
    th = np.linspace(0, 2 * np.pi, 60)                                                      # 0.2 m loop: below path / extent limits
    x = np.r_[np.linspace(-3, 0, 30), 0.1 * np.sin(th), np.linspace(0, 3, 30)]
    y = np.r_[np.zeros(30), 0.1 - 0.1 * np.cos(th), np.zeros(30)]
    assert len(SX.chain_self_crossings(np.arange(len(x)) / 25, x, y)) == 0


def test_selfcross_never_links_across_chain_breaks_and_drops_missing_samples():
    t, x, y = _loop()
    k = 150                                                                                 # split the loop into two chains
    seg = dict(slot=np.zeros(len(t) - 2, int), period=np.ones(len(t) - 2, int), chain=np.r_[np.zeros(k, int), np.ones(len(t) - 2 - k, int)],
               t0=np.r_[t[:k], t[k + 1:-1]], t1=np.r_[t[1:k + 1], t[k + 2:]], x0=np.r_[x[:k], x[k + 1:-1]], x1=np.r_[x[1:k + 1], x[k + 2:]],
               y0=np.r_[y[:k], y[k + 1:-1]], y1=np.r_[y[1:k + 1], y[k + 2:]])
    assert len(SX.segments_self_crossings(seg)) == 0
    x2 = x.copy()
    x2[50:55] = np.nan
    assert len(SX.chain_self_crossings(t, x2, y)) == 1


# ---------------------------------------------------------------- N x M

def test_nxm_schema_rejects_identity_on_outgoing_and_bad_truth():
    d = MP.generate(4, 3)
    assert MP.validate(d) == []
    bad = json.loads(json.dumps(d))
    bad["outgoing"][0]["label"] = "player-0"
    assert any("anonymous" in p for p in MP.validate(bad))
    bad = json.loads(json.dumps(d))
    ids = [o["id"] for o in bad["outgoing"]]
    bad["truth"] = {k: ids[0] for k in bad["truth"]}
    assert any("one-to-one" in p for p in MP.validate(bad))


@pytest.mark.parametrize("level", sorted(MP.LEVELS))
def test_nxm_solver_is_one_to_one_deterministic_and_never_reads_truth(level):
    d = MP.generate(level, 7)
    assert MP.generate(level, 7) == d
    a = MP.solve(d)
    b = MP.solve(MP.public(d))
    assert a.links == b.links
    used = [o for o in a.links.values() if o is not None]
    assert len(used) == len(set(used))
    assert set(a.unmatched_outgoing) == {o["id"] for o in d["outgoing"]} - set(used)


def test_nxm_null_option_and_time_order():
    t_in = np.arange(-2, 0, 1 / 15)
    t_out = np.arange(0.8, 2.8, 1 / 15)
    inc = [dict(id="in-0", label="p0", t=t_in.tolist(), x=(2 * t_in).tolist(), y=[0.0] * len(t_in)),
           dict(id="in-1", label="p1", t=t_in.tolist(), x=(2 * t_in).tolist(), y=[5.0] * len(t_in))]
    out = [dict(id="out-a", t=t_out.tolist(), x=(2 * t_out).tolist(), y=[0.0] * len(t_out)),
           dict(id="out-b", t=t_out.tolist(), x=(2 * t_out + 30).tolist(), y=[40.0] * len(t_out))]       # implausibly far: must be left unmatched
    d = dict(schema=MP.SCHEMA, incoming=inc, outgoing=out, truth={"in-0": "out-a", "in-1": None})
    dec = MP.solve(d)
    assert dec.links == {"in-0": "out-a", "in-1": None} and dec.unmatched_outgoing == ["out-b"]
    s = MP.score(dec, d["truth"])
    assert s["exact"] and s["null_tp"] == 1
    forced = MP.solve(d, use_null=False)
    assert forced.links["in-1"] == "out-b"                                                  # forcing an assignment links the far track
    early = json.loads(json.dumps(d))
    early["outgoing"][0]["t"] = (np.arange(-1.0, 1.0, 1 / 15)).tolist()[:len(t_out)]           # starts before in-0 ends: impossible link
    early["outgoing"][0]["t"] = sorted(early["outgoing"][0]["t"])
    assert not MP.kalman_cost(early)["feasible"][0, 0]


def test_nxm_metrics():
    dec = MP.Decision({"a": "x", "b": None, "c": "y"}, [], np.zeros((3, 2)), np.ones((3, 2), bool), 0.0)
    s = MP.score(dec, {"a": "x", "b": "y", "c": None})
    assert (s["links_tp"], s["links_pred"], s["links_true"], s["null_tp"], s["null_pred"], s["null_true"], s["exact"]) == (1, 2, 2, 0, 1, 1, False)
    agg = MP.aggregate([s, s])
    assert agg["link_precision"] == 0.5 and agg["exact_assignment"] == 0.0


# ---------------------------------------------------------------- review workflow

def test_review_decisions_are_append_only_and_never_automatic(tmp_path):
    assert len(RV.decisions(tmp_path)) == 0
    ans = {k: v[1] for k, (_q, v) in RV.QUESTIONS.items()}
    RV.save_decision("Reviewer One", "P2R-001", ans, "first", out=tmp_path)
    RV.save_decision("Reviewer One", "P2R-001", dict(ans, positions_plausible="no"), "second", out=tmp_path)
    lines = (tmp_path / "decisions" / "Reviewer_One.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2                                                                  # both saves kept
    d = RV.decisions(tmp_path)
    assert len(d) == 1 and d.iloc[0]["positions_plausible"] == "no" and d.iloc[0]["notes"] == "second"
    with pytest.raises(ValueError):
        RV.save_decision("Reviewer One", "P2R-001", dict(positions_plausible="certainly"), out=tmp_path)
    with pytest.raises(ValueError):
        RV.save_decision("   ", "P2R-001", ans, out=tmp_path)


@pytest.fixture(autouse=True)
def _no_repo_review_writes(monkeypatch, tmp_path_factory):
    """Tests never write review decisions into the repository's data folder."""
    monkeypatch.setattr(RV, "OUT", tmp_path_factory.mktemp("review_out"))
    orig = RV.save_decision

    def guarded(reviewer, review_id, answers, notes="", out=None):
        return orig(reviewer, review_id, answers, notes, out=out if out is not None else RV.OUT)
    monkeypatch.setattr(RV, "save_decision", guarded)


@data
def test_review_queue_and_export_mark_only_saved_rows(tmp_path):
    q = RV.queue()
    assert len(q) == 125 and q["review_id"].is_unique
    assert set(q["source"]) == {"phase2 stratified review sample", "phase2 severe motion contradiction"}
    C.checked(sorted({str(v) for v in q["match_id"]}))
    RV.save_decision("tester", q["review_id"].iloc[3], {k: v[0] for k, (_q, v) in RV.QUESTIONS.items()}, out=tmp_path)
    p = RV.export("tester", out=tmp_path)
    e = pd.read_csv(p)
    assert (e["review_state"] == "human reviewed").sum() == 1 and (e["review_state"] == "not reviewed").sum() == 124


# ---------------------------------------------------------------- lab leak fixes (audit 2026-11-19 D1 / D2)

def test_encounter_label_uses_public_fields_only():
    from football_fatigue.m11_dashboard import lab as L
    r = dict(n=3, clock="12:01", teams="same team", crossing="local crossing", movement="opposite", closest_m=0.11, in_gap="mixed")
    s = L.encounter_label(r)
    assert s == "#3 · 12:01 · same team"
    for f in L.REVEAL_ONLY_FIELDS:
        assert str(r[f]) not in s
    assert not set(L.PRE_REVEAL_LABEL_FIELDS) & set(L.REVEAL_ONLY_FIELDS)


def test_gallery_blind_names_do_not_reveal_outcomes():
    from football_fatigue.m11_dashboard import lab as L
    assert set(L.GALLERY_BLIND) == {k for k, _ in L.GALLERY}
    words = ("correct", "wrong", "abstain", "disagree", "error", "noisy", "gap")
    assert all(not any(w in v.lower() for w in words) for v in L.GALLERY_BLIND.values())


def test_lab_app_dropdown_uses_public_label():
    from pathlib import Path
    src = (Path(C.ROOT) / "src" / "football_fatigue" / "m11_dashboard" / "lab_app.py").read_text(encoding="utf-8")
    assert "r.crossing" not in src and "r.movement" not in src
    studio = (Path(C.ROOT) / "src" / "football_fatigue" / "m11_dashboard" / "studio_app.py").read_text(encoding="utf-8")
    assert "L.encounter_label(r)" in studio and "r.crossing" not in studio


# ---------------------------------------------------------------- fair benchmark infrastructure

def test_configurable_nets_reproduce_historical_nets():
    torch = pytest.importorskip("torch")
    from football_fatigue.m11_benchmark import models as MD
    from football_fatigue.m11_fair import nets as N
    for k in ("gru_gaplast", "lstm_gaplast"):
        torch.manual_seed(0)
        a = MD.make_net(k, **MD.NEURAL[k])
        torch.manual_seed(0)
        b = N.make_pair_net(N.HISTORICAL[k])
        sa, sb = a.state_dict(), b.state_dict()
        assert list(sa) == list(sb) and all(torch.equal(sa[x], sb[x]) for x in sa)
        si, so = torch.randn(4, 2, 30, 4), torch.randn(4, 2, 30, 4)
        si[..., 2] = so[..., 2] = 1
        a.eval()
        b.eval()
        assert torch.allclose(a(si, so, torch.full((4,), 0.8)), b(si, so, torch.full((4,), 0.8)))
    two = N.make_pair_net(N.NetConfig("lstm", 64, 48, 2, rnn_dropout=0.1))
    assert two.enc_in.rnn.num_layers == 2


def test_search_spaces_are_seeded_equal_and_flag_edges():
    from football_fatigue.m11_fair import spaces as SP
    for f in SP.SPACES:
        a, b = SP.sample(f), SP.sample(f)
        assert a == b and len(a) == SP.N_CONFIGS and len({c["config_id"] for c in a}) == SP.N_CONFIGS
    lstm = SP.sample("lstm")
    assert len({c["hidden"] for c in lstm}) > 1 and len({c["layers"] for c in lstm}) == 2
    assert SP.edge_flags("svm_nystroem", dict(C=29.0, gamma=0.5, n_components=2000)) == ["C=29 within 10 % of the range edge"]


def test_training_is_refused_without_approval(capsys):
    from football_fatigue.m11_fair.__main__ import main
    assert main(["train", "--family", "lstm"]) == 2
    assert "REFUSED" in capsys.readouterr().err


def test_temperature_and_paired_ba():
    from football_fatigue.m11_fair.train import fit_temperature, paired_ba
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, 4000)
    y = (rng.random(4000) < 1 / (1 + np.exp(-2.0 * z))).astype(int)
    assert abs(fit_temperature(z, y) - 2.0) < 0.3
    g = np.repeat(np.arange(20), 200)
    r = paired_ba(y, (z > 0).astype(float), (z > 0).astype(float), g, n_boot=50)
    assert r["diff"] == 0 and r["ci95"] == [0, 0]


def test_tabular_fold_runner_uses_roles_disjointly():
    from football_fatigue.m11_fair import runner as RU
    from football_fatigue.m11_fair import spaces as SP
    rng = np.random.default_rng(0)
    n = 600
    y = rng.integers(0, 2, n)
    F = np.c_[np.where(y == 1, -1, 1) * rng.uniform(0.5, 2, n) + rng.normal(0, 1, n), rng.normal(0, 1, n)]
    g = np.repeat([str(k) for k in range(12)], 50)
    data = RU.DevData(y, g, F=F, k_perm=1)
    part = dict(fit=[str(k) for k in range(6)], early_stop=["6", "7"], calibration=["8", "9"], test=["10", "11"])
    cfgs = SP.sample("hgb", 2, salt="t")
    for c in cfgs:
        c["max_iter"] = 40
    r = RU.run_fold("hgb", "0", part, data, cfgs, seeds=(0,))
    assert set(np.asarray(g)[r["test_rows"]]) == {"10", "11"} and len(r["trials"]) == 2
    p = np.asarray(r["per_seed"]["0"]["p_test"])
    assert np.isfinite(p).all() and ((p >= 0.5) == y[r["test_rows"]]).mean() > 0.8


@data
def test_fair_partition_revalidates():
    from football_fatigue.m11_fair import protocol as PR
    v = PR.validate_partition(PR.partition())
    assert v["passed"], v


# ---------------------------------------------------------------- figures and headless app pages

def test_match_figure_animates_all_players_and_marks_self_crossings():
    from football_fatigue.m11_dashboard import studio_figs as FG
    t = np.repeat(np.round(np.arange(0, 2, 0.2), 3), 4)
    fr = pd.DataFrame(dict(t=t, slot=np.tile([0, 1, 2, 3], 10), x=np.random.default_rng(0).normal(0, 5, 40), y=0.0, estimated=False))
    sl = pd.DataFrame(dict(slot=[0, 1, 2, 3], side=["home", "home", "away", "away"], jersey=[1, 2, 3, 4], player_id=[1, 2, 3, 4]))
    sx = pd.DataFrame([dict(slot=0, x=0.0, y=0.0, t_first=0.1, t_second=1.5, path_m=5.0)])
    fig = FG.match_figure(fr, sl, selfx=sx)
    assert len(fig.frames) == 10 and any("self-crossing" in (d.name or "") for d in fig.data)
    assert sum(len(d.x) for d in fig.frames[0].data) == 4


def test_nxm_figure_keeps_outgoing_neutral_before_reveal():
    from football_fatigue.m11_dashboard import studio_figs as FG
    d = MP.generate(4, 1)
    dec = MP.solve(d)
    pre = FG.nxm_figure(d, dec.links, reveal=False)
    outs = [t for t in pre.data if "anonymous" in (t.name or "")]
    assert len(outs) == len(d["outgoing"]) and all(t.line.color == FG.NEUTRAL for t in outs)
    post = FG.nxm_figure(d, dec.links, reveal=True)
    assert any("=" in (t.name or "") for t in post.data)
    FG.assignment_heatmap(d, dec.cost, dec.feasible, dec.links)


@pytest.mark.parametrize("page", ["Multi-player (N×M)", "Evidence & provenance"])
def test_studio_pages_render_headless(page):
    pytest.importorskip("streamlit")
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    app = Path(C.ROOT) / "src" / "football_fatigue" / "m11_dashboard" / "studio_app.py"
    at = AppTest.from_file(str(app), default_timeout=120)
    at.session_state["page"] = page
    at.run()
    assert not at.exception, at.exception


@data
def test_studio_explorer_page_renders_headless():
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    app = Path(C.ROOT) / "src" / "football_fatigue" / "m11_dashboard" / "studio_app.py"
    at = AppTest.from_file(str(app), default_timeout=300)
    at.session_state["page"] = "Match & Interaction Explorer"
    at.run()
    assert not at.exception, at.exception
