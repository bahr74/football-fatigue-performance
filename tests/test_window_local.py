"""Unit tests for the M11 local-descriptor / four-method module (synthetic trajectories with closed-form answers; no data files)."""

import math

import numpy as np
import pandas as pd
import pytest

from football_fatigue.window_intersections import core as W
from football_fatigue.window_intersections import local as L

CFG, LCFG = W.WindowConfig(), L.LocalConfig()
HZ = 1 / 15


def make_seg(tracks: dict[int, tuple], drop=None) -> dict[str, np.ndarray]:
    """Segments from {slot: (t, x, y)}; ``drop(slot, t0)`` removes segments (creating a chain break)."""
    cols = {k: [] for k in ("slot", "row0", "row1", "t0", "t1", "x0", "y0", "x1", "y1")}
    row = 0
    for s in sorted(tracks):
        t, x, y = (np.asarray(v, float) for v in tracks[s])
        for i in range(len(t) - 1):
            if drop is not None and drop(s, t[i]):
                continue
            for k, v in zip(cols, (s, row + i, row + i + 1, t[i], t[i + 1], x[i], y[i], x[i + 1], y[i + 1])):
                cols[k].append(v)
        row += len(t) + 10
    seg = {k: np.array(v) for k, v in cols.items()}
    seg["slot"], seg["row0"], seg["row1"] = (seg[k].astype(int) for k in ("slot", "row0", "row1"))
    seg["period"] = np.ones(len(seg["slot"]), int)
    seg["stationary"] = seg["x0"] == seg["x1"]
    return W.finalize_segments(seg)


def events(seg, variant="linear"):
    cx, cy, _ = W.coefficients(seg, variant)
    h, _ = W.period_hits(seg, cx, cy, 1, 0.0, CFG, variant)
    ev = pd.DataFrame(dict(event_id=[f"e{i}" for i in range(len(h["ta"]))], slot_a=h["sa"].astype(int), slot_b=h["sb"].astype(int),
                           seg_a_row0=seg["row0"][h["ga"].astype(int)], seg_b_row0=seg["row0"][h["gb"].astype(int)],
                           t_a_s=h["ta"], t_b_s=h["tb"], x_m=h["x"], y_m=h["y"], est_at_a=False, est_at_b=False))
    ev["abs_dt_s"] = (ev.t_a_s - ev.t_b_s).abs()
    K = CFG.pchip_substeps if variant == "pchip" else 1
    return ev, L.event_descriptors(seg, W.chain_table(seg), cx, cy, K, ev, LCFG)


def crossing(dt, va=2.0, vb=2.0, drop=None):
    """A moves along +x through P=(0,0) at t=5; B along +y through P at t=5+dt (perpendicular)."""
    t = np.arange(0, 10 + 1e-9, HZ)
    return make_seg({0: (t, va * (t - 5), 0 * t), 1: (t, 0 * t, vb * (t - 5 - dt))}, drop)


def test_perpendicular_crossing_closed_form():
    ev, D = events(crossing(0.2))
    assert len(ev) == 1
    r = D.iloc[0]
    assert r.reproj_a_m < 1e-9 and r.reproj_b_m < 1e-9
    assert r.d_at_ta_m == pytest.approx(0.4) and r.d_at_tb_m == pytest.approx(0.4)        # other player is vB * |dt| from P
    dmin = 2 * 2 * 0.2 / math.hypot(2, 2)                                                  # vA vB |dt| sin(theta) / |vA uA - vB uB|
    assert r.arr_h0p00_d_m == pytest.approx(dmin, abs=1e-9)
    assert r.arr_h0p00_t_rel_s == pytest.approx(0.0, abs=1e-9)                             # minimum at the midpoint of the arrivals
    assert r.pt_h0p00_d_m == pytest.approx(0.4)                                            # only the two arrival instants
    assert r.pt_h0p25_d_m == pytest.approx(dmin, abs=1e-9)                                 # [tA +- 0.25] contains the midpoint
    assert r.arr_h0p25_mid_to_p_m == pytest.approx(math.hypot(0.2, 0.2) / 2)
    for eps in (0.05, 0.10, 0.25):                                                         # occupancy [t_i -+ eps / v]
        assert r[f"occ_e{L.key(eps)}_gap_s"] == pytest.approx(max(0.0, 0.2 - eps), abs=1e-9)
        assert r[f"occ_e{L.key(eps)}_len_a_s"] == pytest.approx(eps, abs=1e-9)
    assert r.speed_min_m_s == pytest.approx(2.0) and r.angle_deg == pytest.approx(90.0)


def test_time_within_radius_closed_form():
    _, D = events(crossing(0.1))
    # d(u)^2 = 8 u^2 + 0.02 around the midpoint; <= 0.25^2 for |u| <= sqrt(0.0425 / 8) (inside the arr neighbourhood for h >= 0.03)
    assert D.iloc[0][f"arr_h{L.key(LCFG.primary_h_s)}_time_within_r_s"] == pytest.approx(2 * math.sqrt(0.0425 / 8), abs=1e-9)


def test_slow_players_make_space_redundant_fast_players_do_not():
    _, slow = events(crossing(0.3, va=0.5, vb=0.5))        # vmin * tau = 0.15 m <= R: A(0.30) implies B(0.25)
    _, fast = events(crossing(0.3, va=6.0, vb=6.0))        # same |dt|, 1.27 m apart at best
    assert slow.iloc[0].arr_h0p25_d_m <= 0.25 < fast.iloc[0].arr_h0p25_d_m
    assert fast.iloc[0].arr_h0p25_d_m == pytest.approx(6 * 6 * 0.3 / math.hypot(6, 6), abs=1e-9)


def test_chain_break_truncates_and_never_bridges():
    seg = crossing(0.2, drop=lambda s, t0: s == 1 and 4.5 <= t0 < 5.1)      # B has no track around tA
    ev, D = events(seg)
    r = D.iloc[0]
    assert math.isnan(r.d_at_ta_m) and not math.isnan(r.d_at_tb_m)
    assert bool(r.arr_h0p00_truncated) and bool(r.arr_h0p25_truncated)
    lo = np.arange(0, 10 + 1e-9, HZ)[np.arange(0, 10 + 1e-9, HZ) >= 5.1][0]   # B's chain restarts at the first knot >= 5.1
    assert r.arr_h0p00_d_m == pytest.approx(2 * math.hypot(lo - 5, lo - 5.2), abs=1e-9)   # minimum over what exists, nothing bridged
    assert not r.both_chains_cover_arrivals


def test_pchip_events_lie_on_p_and_invariants_hold():
    rng = np.random.default_rng(5)
    t = np.arange(0, 30, HZ)
    tracks = {s: (t, 5 + np.cumsum(rng.normal(0, 0.2, len(t))), 5 + np.cumsum(rng.normal(0, 0.2, len(t)))) for s in range(3)}
    for variant in ("linear", "pchip"):
        ev, D = events(make_seg(tracks), variant)
        assert len(ev) > 5
        assert max(D.reproj_a_m.max(), D.reproj_b_m.max()) < 1e-6
        ev = ev.assign(**{f"a_tau_{L.key(x)}": ev.abs_dt_s <= x + 1e-12 for x in LCFG.taus_s})
        chk = L.check_descriptors(ev, D, L.method_masks(ev, D, LCFG), LCFG)
        assert chk["all_pass"], chk


def test_occupancy_standing_player_is_capped_and_parallel_pair_min():
    T = np.array([0.0, 1.0, 2.0])
    stand = (T, np.zeros(3), np.zeros(3))
    lo, hi, capped = L.occupancy(stand, (0.0, 0.0), 1.0, 0.05)
    assert (lo, hi, capped) == (0.0, 2.0, True)
    walk = (T, np.array([-1.0, 0.0, 1.0]), np.zeros(3))
    lo, hi, capped = L.occupancy(walk, (0.0, 0.0), 1.0, 0.25)
    assert (lo, hi, capped) == (pytest.approx(0.75), pytest.approx(1.25), False)
    graze = (np.array([-1.0, 0.0, 1.0, 2.0, 3.0]), np.array([-1.0, 0.0, 0.10021, 0.05, 1.0]), np.zeros(5))   # knot 0.2 mm outside eps
    lo, hi, capped = L.occupancy(graze, (0.0, 0.0), 0.0, 0.10)
    assert (lo, hi, capped) == (pytest.approx(-0.1), pytest.approx(0.10 / 0.10021), False)   # ends at the brief exit, not the re-entry
    side = (T, np.array([-1.0, 0.0, 1.0]), np.full(3, 0.3))                 # parallel, 0.3 m apart, same speed
    d, ts, dur = L.pair_min(walk, side, 0.25)
    assert d == pytest.approx(0.3) and dur == 0.0


def test_pair_min_matches_dense_sampling():
    rng = np.random.default_rng(11)
    for _ in range(50):
        T = np.cumsum(rng.uniform(0.02, 0.1, 30))
        A = (T, np.cumsum(rng.normal(0, 0.1, 30)), np.cumsum(rng.normal(0, 0.1, 30)))
        B = (T + rng.uniform(-0.01, 0.01), np.cumsum(rng.normal(0, 0.1, 30)), np.cumsum(rng.normal(0, 0.1, 30)))
        lo, hi = max(A[0][0], B[0][0]), min(A[0][-1], B[0][-1])
        a, b = L.clip_track(A, lo, hi), L.clip_track(B, lo, hi)
        d, ts, _ = L.pair_min(a, b)
        tt = np.linspace(lo, hi, 200001)
        dd = np.hypot(np.interp(tt, a[0], a[1]) - np.interp(tt, b[0], b[1]), np.interp(tt, a[0], a[2]) - np.interp(tt, b[0], b[2]))
        assert d <= dd.min() + 1e-12 and dd.min() - d < 1e-4


def test_method_masks_logic_and_guard():
    ev = pd.DataFrame(dict(event_id=["a", "b", "c", "d"], abs_dt_s=[0.05, 0.25, 2.0, 0.05]))
    D = pd.DataFrame(dict(event_id=ev.event_id))
    for nb in ("arr", "pt"):
        for h in LCFG.hs_s:
            D[f"{nb}_h{L.key(h)}_d_m"] = [0.1, 0.4, 0.2, np.nan]
            D[f"{nb}_h{L.key(h)}_mid_to_p_m"] = [0.1, 0.1, 3.0, np.nan]
    for tag in [f"e{L.key(x)}" for x in LCFG.eps_m] + ["tier"]:
        D[f"occ_{tag}_gap_s"] = [0.0, 0.15, 1.9, 0.0]
    M = L.method_masks(ev, D, LCFG)
    assert M["F1_tau0p30"].tolist() == [True, True, False, True]
    assert M["F2_arr_h0p25_R0p25"].tolist() == [True, False, True, False]                 # NaN distance never passes
    assert M["F2_arr_h0p25_R0p25_g1p00"].tolist() == [True, False, False, False]          # guard removes the far closest approach
    assert M["F3_tau0p30_arr_h0p25_R0p25"].tolist() == [True, False, False, False]
    assert M["F2_arr_h0p25_R0p50"].tolist() == [True, True, True, False]
    assert M["F4_e0p25_tau0p10"].tolist() == [True, False, False, True]
