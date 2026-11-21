"""Interaction Catalogue v1 tests. All trajectories are SYNTHETIC (hand-made straight lines and curves); no source data is used,
except the protected-boundary tests, which read only the frozen split files (match ids, no tracking data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football_fatigue.encounters import quality as Q
from football_fatigue.interaction_catalogue import core as K
from football_fatigue.window_intersections import candidates as C
from football_fatigue.window_intersections import core as W

DT = 1 / 15
CFG = K.CatalogueConfig()


def path(t0: float, t1: float, f) -> np.ndarray:
    """Samples (t, x, y) of a synthetic player every 1/15 s, offset by 1/30 s so crossings never fall on a vertex."""
    t = np.arange(t0 + DT / 2, t1, DT)
    x, y = f(t)
    return np.c_[t, np.broadcast_to(x, t.shape), np.broadcast_to(y, t.shape)]


def line(p0, v):
    return lambda t: (p0[0] + v[0] * t, p0[1] + v[1] * t)


def build(tracks: dict[int, list[np.ndarray]], period: int = 1):
    """Segments in the window_intersections layout; each array in a slot's list is one chain (a break between arrays)."""
    cols = {k: [] for k in ("slot", "period", "t0", "t1", "x0", "y0", "x1", "y1", "row0", "row1")}
    row = 0
    for s in sorted(tracks):
        for piece in tracks[s]:
            for i in range(len(piece) - 1):
                for k, v in zip(("slot", "period", "t0", "x0", "y0", "t1", "x1", "y1", "row0", "row1"),
                                (s, period, *piece[i], *piece[i + 1], row + i, row + i + 1)):
                    cols[k].append(v)
            row += len(piece) + 1
    seg = {k: np.array(v, dtype=int if k in ("slot", "period", "row0", "row1") else float) for k, v in cols.items()}
    seg = W.finalize_segments(seg)
    cx, cy, _ = W.coefficients(seg, "linear")
    return seg, W.chain_table(seg), cx, cy


def run(tracks, pstart=0.0, pend=10.0, fc=None, sides=None, cfg=CFG):
    seg, ch, cx, cy = build(tracks)
    sides = np.array(sides or ["home", "away"] * 4)
    return K.period_catalogue(seg, ch, cx, cy, 1, pstart, pend, fc, sides, 0, cfg)


def at_radius(df: pd.DataFrame, R: float) -> pd.DataFrame:
    return df[df["radius_m"] == R]


# ---------------------------------------------------------------- events

def test_single_isolated_encounter():
    out = run({0: [path(0, 10, line((-5, 0), (1, 0)))], 1: [path(0, 10, line((0, -5), (0, 1)))]})
    e = out["edges"]
    assert e.groupby("radius_m").size().to_dict() == {r: 1 for r in CFG.radii_m}
    assert np.allclose(e["t_star"], 5.0, atol=0.05) and (e["d_min"] < 0.05).all()
    assert len(out["crossings"]) == 1 and bool(out["crossings"]["synchronous"].iloc[0])
    assert (e["n_sync_crossings_in_edge"] == 1).all()
    g = out["groups"]
    assert (g["n_players"] == 2).all() and len(g) == len(CFG.radii_m) * len(CFG.group_link_s)
    assert e["rel_heading_deg"].between(85, 95).all()            # perpendicular approach


def test_two_intersections_in_one_window_are_all_kept():
    # A weaves across B's straight path: five crossings, 1 s apart, all inside the single complete window [0.5, 5.5]
    a = path(0.5, 5.5, lambda t: (t - 3, 2 * np.sin(np.pi * (t - 3))))
    b = path(0.5, 5.5, line((-3, 0), (1, 0)))
    out = run({0: [a], 1: [b]}, pstart=0.0, pend=6.0)
    assert len(out["crossings"]) == 5
    pw = out["pair_windows"]
    assert len(pw) == 1 and int(pw["n_intersections"].iloc[0]) == 5          # the "first intersection" rule would keep 1
    e05 = at_radius(out["edges"], 0.5)
    assert len(e05) == 5                                                       # five proximity edges ...
    assert e05["edge_chain_id"].nunique() == 1                                 # ... one repeated-contact chain (gaps < 1 s)
    ch = at_radius(out["edge_chains"], 0.5)
    assert int(ch["n_edges"].iloc[0]) == 5


def test_edge_chain_matches_stage2_merge_rule():
    rng = np.random.default_rng(0)
    s = np.sort(rng.uniform(0, 30, 40))
    e = s + rng.uniform(0, 1.5, 40)
    lab = K.edge_chains(s, e, 1.0)
    groups = C.merge_encounters([dict(start=a, end=b, i=i) for i, (a, b) in enumerate(zip(s, e))], 1.0)
    ref = np.empty(40, int)
    for k, g in enumerate(groups):
        for x in g:
            ref[x["i"]] = k
    assert (lab == ref).all()


def test_repeated_same_pair_and_same_player_interactions():
    # A and B meet at t = 2 and again at t = 8 (separate chains); A meets C 0.5 s after the first A-B contact ends
    a = path(0, 10, lambda t: (np.zeros_like(t), np.zeros_like(t)))           # A stands at the origin
    b = path(0, 10, lambda t: (np.where(t < 5, t - 2, 8 - t), np.full_like(t, 0.0)))   # passes A at t = 2 and t = 8
    c = path(0, 10, lambda t: (np.full_like(t, 0.0), t - 3.3))                # passes A at t = 3.3
    out = run({0: [a], 1: [b], 2: [c]})
    e = at_radius(out["edges"], 0.5)
    ab = e[(e["slot_a"] == 0) & (e["slot_b"] == 1)]
    assert len(ab) == 2 and ab["edge_chain_id"].nunique() == 2
    first = ab.sort_values("start").iloc[0]
    assert first["n_sequential_other"] >= 1 and first["n_concurrent_other"] == 0
    g = out["groups"]
    g0 = g[(g["radius_m"] == 0.5) & (g["link_s"] == 0.0)]
    g1 = g[(g["radius_m"] == 0.5) & (g["link_s"] == 1.0)]
    assert sorted(g0["n_players"]) == [2, 2, 2]                               # never simultaneous
    assert sorted(g1["n_players"]) == [2, 3]                                  # A-B then A-C within 1 s: one connected group


def test_three_player_group():
    ang = np.deg2rad([0, 120, 240])
    tr = {i: [path(0, 10, line((-5 * np.cos(a), -5 * np.sin(a)), (np.cos(a), np.sin(a))))] for i, a in enumerate(ang)}
    out = run(tr)
    g = out["groups"]
    g0 = g[(g["radius_m"] == 1.0) & (g["link_s"] == 0.0)]
    assert len(g0) == 1 and int(g0["n_players"].iloc[0]) == 3 and int(g0["max_simultaneous_players"].iloc[0]) == 3
    assert int(g0["n_pairs"].iloc[0]) == 3
    assert len(out["group_timeline"]) > 0


def test_four_player_overlapping_groups_are_transitive():
    # A-B close on [4, 6], B-C on [5.5, 7], C-D on [6.5, 8]: one simultaneous group of 4, never more than 3 at once
    def stay(x, y):
        return lambda t: (np.full_like(t, x), np.full_like(t, y))
    a = path(0, 10, stay(0, 0))
    b = path(0, 10, lambda t: (np.where(t < 4, 0.3 + 2 * (4 - t), np.where(t <= 6, 0.3, 0.3 + 2 * (t - 6))), np.zeros_like(t)))
    c = path(0, 10, lambda t: (np.where(t < 5.5, 0.6 + 2 * (5.5 - t), np.where(t <= 7, 0.6, 0.6 + 2 * (t - 7))), np.full_like(t, 0.2)))
    d = path(0, 10, lambda t: (np.where(t < 6.5, 0.9 + 2 * (6.5 - t), np.where(t <= 8, 0.9, 0.9 + 2 * (t - 8))), np.full_like(t, 0.4)))
    cfg = K.CatalogueConfig(radii_m=(0.5,))
    out = run({0: [a], 1: [b], 2: [c], 3: [d]}, cfg=cfg)
    e = out["edges"]
    pairs = set(zip(e["slot_a"], e["slot_b"]))
    assert {(0, 1), (1, 2), (2, 3)} <= pairs
    g0 = out["groups"][out["groups"]["link_s"] == 0.0]
    big = g0[g0["n_players"] == 4]
    assert len(big) == 1 and 2 <= int(big["max_simultaneous_players"].iloc[0]) <= 4
    # two disjoint simultaneous pairs far apart are two groups
    far = run({0: [path(0, 10, line((-5, 0), (1, 0)))], 1: [path(0, 10, line((0, -5), (0, 1)))],
               2: [path(0, 10, line((15, 0), (1, 0)))], 3: [path(0, 10, line((20, -5), (0, 1)))]}, cfg=cfg)
    gf = far["groups"][far["groups"]["link_s"] == 0.0]
    assert sorted(gf["n_players"]) == [2, 2]


def test_group_formation_merge_split_and_dissolution():
    # formation, a join and a leave: A-B [2, 4], A-C [3, 6]
    st, _ = K.lifecycle(np.array([2.0, 3.0]), np.array([4.0, 6.0]), np.array([0, 0]), np.array([1, 2]))
    assert st["n_joins"] == 1 and st["n_leaves"] == 1 and st["n_merges"] == 0 and st["max_simultaneous_players"] == 3
    # merge: A-B and C-D, then B-C links them
    st, tl = K.lifecycle(np.array([0.0, 0.0, 1.0]), np.array([2.0, 2.0, 3.0]), np.array([0, 2, 1]), np.array([1, 3, 2]))
    assert st["n_merges"] == 1 and st["max_simultaneous_players"] == 4
    assert tl[0]["components"] == "0;1|2;3" and tl[-1]["components"] == "1;2"
    # split: B-C first, then A-B and C-D, then B-C ends
    st, _ = K.lifecycle(np.array([0.0, 1.0, 1.0]), np.array([2.0, 3.0, 3.0]), np.array([1, 0, 2]), np.array([2, 1, 3]))
    assert st["n_splits"] == 1


def test_sliding_windows_never_inflate_edge_counts():
    out = run({0: [path(0, 20, line((-10, 0), (1, 0)))], 1: [path(0, 20, line((0, -10), (0, 1)))]}, pend=20.0)
    ws = out["window_stats"]
    r = ws[ws["radius_m"] == 0.5].iloc[0]
    assert r["edges"] == 1 and r["window_detections"] >= 5 and r["unique_pair_windows"] == r["window_detections"]
    e = at_radius(out["edges"], 0.5)
    assert int(e["n_windows"].iloc[0]) == r["window_detections"]
    # two edges of the same pair in overlapping windows: unique pair-windows < sum of window detections
    b = path(0, 20, lambda t: (np.where(t < 10, t - 8, 12 - t), np.zeros_like(t)))
    out2 = run({0: [path(0, 20, lambda t: (np.zeros_like(t), np.zeros_like(t)))], 1: [b]}, pend=20.0)
    r2 = out2["window_stats"][out2["window_stats"]["radius_m"] == 0.5].iloc[0]
    assert r2["edges"] == 2 and r2["unique_pair_windows"] < r2["window_detections"]


# ---------------------------------------------------------------- quality metadata

def _fc(n_slots: int, t1: float, est: bool = True) -> K.FrameContext:
    t = np.arange(0, t1, 1 / 30)
    mask = np.zeros((len(t), n_slots), np.uint32)
    mask[::2] |= Q.UPDATE_INSTANT
    if est:
        mask |= Q.ESTIMATED
    return K.FrameContext(t=t, mask=mask, avail=np.ones(mask.shape, bool), dup=np.zeros(mask.shape, bool), oob=np.zeros(mask.shape, bool))


def test_status_alone_never_reduces_reliability_or_eligibility():
    tr = {0: [path(0, 10, line((-5, 0), (1, 0)))], 1: [path(0, 10, line((0, -5), (0, 1)))]}
    est = run(tr, fc=_fc(2, 10.0, est=True))["edges"]
    vis = run(tr, fc=_fc(2, 10.0, est=False))["edges"]
    assert (est["status_composition"] == "both_estimated").all() and (vis["status_composition"] == "both_visible").all()
    assert est["eligible"].all() and vis["eligible"].all()
    assert (est["reliability"] == vis["reliability"]).all() and (est["label_risk"] == vis["label_risk"]).all()
    assert len(est) == len(vis)


def test_provider_quality_flags():
    tr = {0: [path(0, 10, line((-5, 0), (1, 0)))], 1: [path(0, 10, line((0, -5), (0, 1)))]}
    fc = _fc(2, 10.0, est=False)
    i = int(np.searchsorted(fc.t, 5.2))
    fc.mask[i, 0] |= Q.IMPLAUSIBLE_JUMP
    fc.oob[i, 1] = True
    e = at_radius(run(tr, fc=fc)["edges"], 1.0).iloc[0]
    assert e["label_risk"] and "implausible_jump" in e["label_risk_reasons"]
    assert e["reliability"] == "reduced" and "out_of_pitch" in e["reliability_reasons"]
    assert e["eligible"]
    # long ESTIMATED co-motion (clone-like): two players moving together 0.1 m apart for 4 s
    co = {0: [path(0, 10, line((-5, 0), (1, 0)))], 1: [path(0, 10, lambda t: (np.where(np.abs(t - 5) < 2, t - 5, np.where(t < 5, -3 - 2 * (3 - t), 3 + 2 * (t - 7))), np.full_like(t, 0.1)))]}
    ec = at_radius(run(co, fc=_fc(2, 10.0, est=True))["edges"], 0.25)
    assert ec["estimated_comotion"].any()
    ev = at_radius(run(co, fc=_fc(2, 10.0, est=False))["edges"], 0.25)
    assert not ev["estimated_comotion"].any()


def test_discontinuous_chain_is_not_bridged():
    # A's chain breaks during [4.6, 5.4] (no samples): no interpolation across the gap; edges touching the break carry label risk
    full = path(0, 10, line((-5, 0), (1, 0)))
    a = [full[full[:, 0] < 4.6], full[full[:, 0] > 5.4]]
    out = run({0: a, 1: [path(0, 10, line((0, -5), (0, 1)))]})
    e = out["edges"]
    assert not ((e["start"] < 5.3) & (e["end"] > 4.7)).any()                    # nothing inside the gap
    assert e["node_a"].nunique() == 2                                            # two chain nodes for A
    assert (e["trunc_before"] | e["trunc_after"]).any()
    assert e.loc[e["trunc_before"] | e["trunc_after"], "label_risk"].all()
    assert "chain_boundary" in ";".join(e["label_risk_reasons"])


def test_radii_are_nested_and_symmetric():
    rng = np.random.default_rng(3)
    tr = {i: [path(0, 20, lambda t, p=rng.uniform(-3, 3, 2), v=rng.uniform(-0.5, 0.5, 2): (p[0] + v[0] * t + 0.3 * np.sin(t + i), p[1] + v[1] * t))]
          for i in range(5)}
    e = run(tr, pend=20.0)["edges"]
    for r0, r1 in zip(CFG.radii_m[:-1], CFG.radii_m[1:]):
        a, b = at_radius(e, r0), at_radius(e, r1)
        for _, x in a.iterrows():
            m = b[(b["slot_a"] == x["slot_a"]) & (b["slot_b"] == x["slot_b"])]
            assert ((m["start"] <= x["start"] + 1e-9) & (m["end"] >= x["end"] - 1e-9)).any()
    assert (e["slot_a"] < e["slot_b"]).all() and e["edge_id"].is_unique


def test_ambiguity_indicator_signs():
    # two players running side by side then separating: constant velocity keeps them apart -> separable
    par = {0: [path(0, 10, line((-5, 0), (1, 0)))], 1: [path(0, 10, lambda t: (t - 5, 0.4 + 0.1 * np.abs(t - 5)))]}
    assert (at_radius(run(par)["edges"], 0.5)["cv_tier"] == "cv_separable").all()
    # two players meet head-on and both turn back at the meeting point: constant velocity predicts a swap
    a = path(0, 10, lambda t: (-np.abs(t - 5), np.zeros_like(t)))
    b = path(0, 10, lambda t: (np.abs(t - 5), np.full_like(t, 0.1)))
    assert (at_radius(run({0: [a], 1: [b]})["edges"], 0.5)["cv_tier"] == "cv_confusable").all()


# ---------------------------------------------------------------- protected boundary

def test_protected_matches_are_refused_before_loading(monkeypatch):
    from football_fatigue.interaction_catalogue import run as M
    allowed = M.allowed_matches()
    prot = M.protected_matches()
    assert len(allowed) == 32 and not set(allowed) & prot
    for m in ("3822", "3824", "3834", "3839", "3844", "10510", "10514", "10517"):
        assert m in prot
    assert len(prot) == 32                                       # 8 internal held-out + 24 official validation / test

    def boom(*a, **k):
        raise AssertionError("data loaded for a protected match")

    monkeypatch.setattr(M, "load_source", boom)
    for m in sorted(prot):
        with pytest.raises(SystemExit):
            M.process_match((m, "unused", CFG.as_dict(), allowed))
    with pytest.raises(SystemExit):
        M.assert_allowed("99999", allowed)
