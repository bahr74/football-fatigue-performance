"""M11 Stage 1 tests: identity-free candidates, label-quality tiers, three-state methods, leakage invariance and crash / resume.

Synthetic trajectories with closed-form answers; no data files are read."""

import json
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from football_fatigue.encounters import quality as Q
from football_fatigue.encounters.trajectories import STATE_AVAILABLE
from football_fatigue.window_intersections import candidates as C
from football_fatigue.window_intersections import core as W
from football_fatigue.window_intersections import local as L

FPS = 15
N = 301                                     # frames 0..300 -> t = 0..20 s
T = np.arange(N) / FPS
CFG, WCFG, LCFG = C.CandidateConfig(), W.WindowConfig(), L.LocalConfig()


def world(tracks: dict[int, tuple], drop=None, est=None, bits=None):
    """Synthetic match on a common 15 Hz frame grid. ``tracks``: {slot: (x(t), y(t))} arrays over T. ``drop(slot, i)`` removes the
    segment from frame i to i + 1 (chain break). ``est(slot)`` / ``bits(slot)``: boolean frame masks for ESTIMATED / other bits."""
    ns = max(tracks) + 1
    cols = {k: [] for k in ("slot", "row0", "row1", "t0", "t1", "x0", "y0", "x1", "y1")}
    x, y = np.full((N, ns), np.nan), np.full((N, ns), np.nan)
    for s in sorted(tracks):
        xs, ys = tracks[s]
        x[:, s], y[:, s] = xs, ys
        for i in range(N - 1):
            if drop is not None and drop(s, i):
                continue
            for k, v in zip(cols, (s, i, i + 1, T[i], T[i + 1], xs[i], ys[i], xs[i + 1], ys[i + 1])):
                cols[k].append(v)
    seg = {k: np.array(v) for k, v in cols.items()}
    seg["slot"], seg["row0"], seg["row1"] = (seg[k].astype(int) for k in ("slot", "row0", "row1"))
    seg["period"] = np.ones(len(seg["slot"]), int)
    seg["stationary"] = (seg["x0"] == seg["x1"]) & (seg["y0"] == seg["y1"])
    seg = W.finalize_segments(seg)
    mask = np.zeros((N, ns), np.int64)
    for s in range(ns):
        if est is not None:
            mask[est(s), s] |= Q.ESTIMATED
        if bits is not None:
            b = bits(s)
            if b is not None:
                mask[b[0], s] |= b[1]
    c = SimpleNamespace(match_id=1, fps=float(FPS), canonical_period=np.ones(N, np.int8), video_time_s=T.copy(),
                        state=np.full((N, ns), STATE_AVAILABLE, np.int8), slot_side=np.array(["home"] * ns), x=x, y=y)
    return c, mask, np.zeros((N, ns), bool), seg, W.chain_table(seg)


def cands(w, cfg=CFG):
    c, mask, dup, seg, ch = w
    return C.match_candidates(c, mask, dup, seg, ch, cfg, WCFG)


def crossing(dt=0.0, va=2.0, vb=2.0, tc=10.0):
    """A along +x through the origin at tc; B along +y through the origin at tc + dt."""
    return {0: (va * (T - tc), 0 * T), 1: (0 * T, vb * (T - tc - dt))}


SELECT = ["period", "t_star", "d_min", "start", "end", "enc_start", "enc_end", "n_episodes", "n_local_minima_total", "strict_pass"]


# ---------------------------------------------------------------- geometry of d(t)

def test_episode_closed_form_and_symmetry():
    tr = crossing(0.2)
    A, B = (T, *tr[0]), (T, *tr[1])
    e = C.episodes_on(A, B, 0.5)
    assert len(e) == 1
    dmin = 2 * 2 * 0.2 / math.hypot(2, 2)                        # vA vB |dt| sin(90) / |vA uA - vB uB|
    assert e[0]["d_min"] == pytest.approx(dmin, abs=1e-9) and e[0]["t_star"] == pytest.approx(10.1, abs=1e-9)
    # d(u)^2 = 8 u^2 + dmin^2 around the minimum: inside for |u| <= sqrt((0.25 - dmin^2) / 8)
    half = math.sqrt((0.25 - dmin ** 2) / 8)
    assert e[0]["end"] - e[0]["start"] == pytest.approx(2 * half, abs=1e-9)
    assert C.episodes_on(B, A, 0.5) == e                           # symmetric in the two players


def test_episode_matches_dense_sampling():
    rng = np.random.default_rng(3)
    for _ in range(30):
        A = (T, np.cumsum(rng.normal(0, 0.05, N)), np.cumsum(rng.normal(0, 0.05, N)))
        B = (T, A[1] + np.cumsum(rng.normal(0, 0.03, N)) + 0.3, A[2] + np.cumsum(rng.normal(0, 0.03, N)))
        eps = C.episodes_on(A, B, 0.5)
        tt = np.linspace(0, T[-1], 400001)
        dd = np.hypot(np.interp(tt, T, A[1]) - np.interp(tt, T, B[1]), np.interp(tt, T, A[2]) - np.interp(tt, T, B[2]))
        inside = dd <= 0.5
        assert sum(e["end"] - e["start"] for e in eps) == pytest.approx(inside.mean() * T[-1], abs=2e-3)
        for e in eps:
            sel = (tt >= e["start"]) & (tt <= e["end"])
            assert e["d_min"] <= dd[sel].min() + 1e-12 and dd[sel].min() - e["d_min"] < 1e-4


def test_merge_and_dedup():
    # B weaves 0.1 <-> 0.9 m from A (period 0.8 s, < 1 s outside R between approaches) -> one encounter of several episodes;
    # then again 3 s later -> second encounter
    bx = np.where(T < 13, 0.5 - 0.4 * np.cos(2 * np.pi * (T - 10) / 0.8), 3.0)
    tr = {0: (0 * T, 0 * T), 1: (np.where(T < 9, 3.0, bx), 0 * T)}
    tr[1] = (np.where((T > 16) & (T < 17), 0.1, tr[1][0]), tr[1][1])
    df = cands(world(tr))
    assert len(df) == 2
    assert df.iloc[0]["n_episodes"] >= 2 and df.iloc[1]["n_episodes"] == 1
    assert C.check_candidates(df, CFG)["all_pass"]


# ---------------------------------------------------------------- leakage evidence

def test_selection_is_invariant_to_swapping_identities():
    """Exchanging the two players' tails at the instant they coincide (d = 0) turns a crossing into a bounce (the SWAPPED
    configuration of the same encounter). Selection depends only on {A(t), B(t)}, so the candidate set is identical; only the
    offline crossing metadata changes."""
    tr = crossing(0.0)
    i0 = int(np.flatnonzero(np.isclose(T, 10.0))[0])
    sw = {0: (np.r_[tr[0][0][:i0], tr[1][0][i0:]], np.r_[tr[0][1][:i0], tr[1][1][i0:]]),
          1: (np.r_[tr[1][0][:i0], tr[0][0][i0:]], np.r_[tr[1][1][:i0], tr[0][1][i0:]])}
    a, b = cands(world(tr)), cands(world(sw))
    assert len(a) == len(b) == 1
    pd.testing.assert_frame_equal(a[SELECT], b[SELECT])
    assert a.iloc[0]["crossing_class"] == "local_crossing"


def test_selection_symmetric_in_slot_labels():
    rng = np.random.default_rng(8)
    tr = {s: (np.cumsum(rng.normal(0, 0.15, N)) + s * 0.4, np.cumsum(rng.normal(0, 0.15, N))) for s in range(3)}
    perm = {0: 2, 1: 0, 2: 1}
    a = cands(world(tr))
    b = cands(world({perm[s]: v for s, v in tr.items()}))
    key = lambda df, mp: sorted((round(r.t_star, 9), round(r.d_min, 12), tuple(sorted((mp[r.slot_a], mp[r.slot_b])))) for r in df.itertuples())  # noqa: E731
    assert len(a) > 0 and key(a, perm) == key(b, {s: s for s in range(3)})


def test_population_contains_non_crossing_encounters():
    # parallel co-running 0.3 m apart: a candidate (d <= 0.5) that never crosses, and not strict (0.3 > 0.25)
    tr = {0: (2 * (T - 10), 0 * T), 1: (2 * (T - 10), 0 * T + 0.3), 2: crossing(0.1)[1]}
    tr[1] = (np.where(np.abs(T - 10) < 1, tr[1][0], 50.0 + T), tr[1][1])
    df = cands(world(tr))
    par = df[(df.slot_a == 0) & (df.slot_b == 1)]
    assert len(par) == 1 and par.iloc[0]["crossing_class"] == "no_crossing" and not par.iloc[0]["strict_pass"]
    assert par.iloc[0]["d_min"] == pytest.approx(0.3, abs=1e-9)
    assert set(df["crossing_class"]) >= {"no_crossing", "local_crossing"}


# ---------------------------------------------------------------- label quality

def _q(df, k, g="0p80"):
    return bool(df.iloc[0][f"g{g}_ok_{k}"])


def _o(df, g="0p80"):
    return df.iloc[0][f"g{g}_outcome"]


def test_label_quality_outcomes_and_each_reason():
    clean = cands(world(crossing(0.1)))
    assert len(clean) == 1 and _o(clean) == C.ACCEPTED and json.loads(clean.iloc[0]["g0p80_reasons"]) == []
    # player 1 ESTIMATED throughout: same outcome as VISIBLE (status is audit metadata only)
    e = cands(world(crossing(0.1), est=lambda s: np.full(N, s == 1)))
    assert _o(e) == C.ACCEPTED and e.iloc[0]["g0p80_visible_before_s_b"] == 0.0 and e.iloc[0]["g0p80_est_frac_used_b"] == 1.0
    # chain break of player 1 0.3 s after the gap ends -> reference not continuous across gap + margin -> REJECTED
    br = cands(world(crossing(0.1), drop=lambda s, i: s == 1 and i == int(round((10.05 + 0.4 + 0.3) * FPS))))
    assert not _q(br, "continuous_reference") and _o(br) == C.REJECTED
    # substitution-window bit on a used sample -> REJECTED
    sub = cands(world(crossing(0.1), bits=lambda s: (np.isclose(T, 8.6), Q.SUBSTITUTION_WINDOW) if s == 0 else None))
    assert not _q(sub, "no_identity_risk") and _o(sub) == C.REJECTED and "substitution_window" in json.loads(sub.iloc[0]["g0p80_flags"])
    # 14 m/s player -> implausible chord speed -> REJECTED
    fast = cands(world(crossing(0.02, va=14.0)))
    assert not _q(fast, "chord_speed_plausible") and _o(fast) == C.REJECTED
    # a jump flag on the first sample of the player's own chain is the step into the chain, not inside it
    jb = int(round(8.0 * FPS))
    own = cands(world(crossing(0.1), drop=lambda s, i: s == 0 and i == jb - 1,
                      bits=lambda s: (np.arange(N) == jb, Q.IMPLAUSIBLE_JUMP) if s == 0 else None))
    assert _q(own, "no_jump_or_shift_flag") and _o(own) == C.ACCEPTED and own.iloc[0]["g0p80_boundary_jumps_in_window"] == 1
    inside = cands(world(crossing(0.1), bits=lambda s: (np.arange(N) == jb, Q.IMPLAUSIBLE_JUMP) if s == 0 else None))
    assert not _q(inside, "no_jump_or_shift_flag") and _o(inside) == C.REJECTED
    # window outside the period
    early = cands(world(crossing(0.1, tc=1.0)))
    assert not _q(early, "window_in_period") and _o(early) == C.REJECTED


def test_lower_confidence_identity_only_for_identity_risk_outside_used_samples():
    # player 0's chain starts at 8.53 s; a label-transfer flag at 8.0 s lies in the window but before the used samples
    cut = int(round(8.47 * FPS))
    w = world(crossing(0.1), drop=lambda s, i: s == 0 and i == cut,
              bits=lambda s: (np.isclose(T, 8.0), Q.LABEL_TRANSFER_CANDIDATE) if s == 0 else None)
    df = cands(w)
    assert _o(df) == C.LOWER and json.loads(df.iloc[0]["g0p80_lower_reasons"]) == ["label_transfer_candidate_near_sample"]
    assert json.loads(df.iloc[0]["g0p80_reasons"]) == []


def test_status_neutrality():
    rng = np.random.default_rng(31)
    tr = {s: (np.cumsum(rng.normal(0, 0.12, N)) + s * 0.3, np.cumsum(rng.normal(0, 0.12, N))) for s in range(4)}
    ends = ("_outcome", "_reasons", "_lower_reasons", "_flags", "_diagnostics", "_independent", "_overlap_group")
    vis = cands(world(tr))
    keep = [c for c in vis.columns if c.endswith(ends)]
    for est in (lambda s: np.ones(N, bool), lambda s: rng.random(N) < 0.5):
        pd.testing.assert_frame_equal(vis[keep], cands(world(tr, est=est))[keep])
    assert len(vis) > 3


def test_diagnostics_never_change_the_outcome():
    nc = cands(world(crossing(0.0)))                                   # paths coincide at the minimum: d_min = 0
    assert json.loads(nc.iloc[0]["g0p80_diagnostics"]) == ["near_coincidence"] and _o(nc) == C.ACCEPTED
    # slow players; player 0's chain starts with a jump landing 1.1 s before the gap, 1.1 m from the partner
    cut = int(round(8.47 * FPS))
    jump = lambda s: (np.arange(N) == cut + 1, Q.IMPLAUSIBLE_JUMP) if s == 0 else None  # noqa: E731
    jr = cands(world(crossing(0.1, va=0.5, vb=0.5), drop=lambda s, i: s == 0 and i == cut, bits=jump))
    assert "jump_transfer_risk" in json.loads(jr.iloc[0]["g0p80_diagnostics"]) and _o(jr) == C.ACCEPTED
    assert json.loads(jr.iloc[0]["g0p80_jump_transfer_risk"]) == ["a_start"]
    far = cands(world(crossing(0.1), drop=lambda s, i: s == 0 and i == cut, bits=jump))     # landing 4.4 m from the partner
    assert "jump_transfer_risk" not in json.loads(far.iloc[0]["g0p80_diagnostics"])


def test_overlap_groups_and_independent_set():
    tr = {0: (0 * T, 0 * T), 1: (2 * (T - 10), 0 * T + 0.1), 2: (0 * T + 0.1, 2 * (T - 11)),
          3: (2 * (T - 16) + 30, 0 * T + 30), 4: (0 * T + 30.1, 2 * (T - 16) + 30)}
    df = cands(world(tr))
    acc = df[df["g0p80_outcome"] == C.ACCEPTED]
    ab, ac, de = (acc[(acc.slot_a == a) & (acc.slot_b == b)].iloc[0] for a, b in ((0, 1), (0, 2), (3, 4)))
    assert ab["g0p80_overlap_group"] == ac["g0p80_overlap_group"] != de["g0p80_overlap_group"]
    assert bool(ab["g0p80_independent"]) and not bool(ac["g0p80_independent"]) and bool(de["g0p80_independent"])
    assert C.check_candidates(df, CFG)["all_pass"]


def test_funnel_and_summary_consistent():
    rng = np.random.default_rng(21)
    tr = {s: (np.cumsum(rng.normal(0, 0.12, N)) + s * 0.3, np.cumsum(rng.normal(0, 0.12, N))) for s in range(4)}
    df = cands(world(tr, est=lambda s: rng.random(N) < 0.3))
    assert len(df) > 3 and C.check_candidates(df, CFG)["all_pass"]
    f = C.funnel(df, 0.8)
    seq = list(f["surviving_after_each_reject_check"].values())
    assert seq == sorted(seq, reverse=True) and seq[-1] == f[C.ACCEPTED] + f[C.LOWER]
    assert f[C.ACCEPTED] + f[C.LOWER] + f[C.REJECTED] == len(df)
    df["cohort_F2_primary"] = 1
    s = C.summary(df, CFG, ["F2_primary"])
    assert set(s["per_gap"]) == {"0p40", "0p80", "1p60"}
    assert s["per_gap"]["0p80"]["dependency_accepted"]["accepted"] == f[C.ACCEPTED]


def test_control_availability_ratio_and_weights():
    rows = ([dict(crossing_class="local_crossing", strict_pass=True, chord_speed_min_m_s=1.5, c=1)] * 6
            + [dict(crossing_class="no_crossing", strict_pass=True, chord_speed_min_m_s=1.5, c=-1)] * 2
            + [dict(crossing_class="local_crossing", strict_pass=False, chord_speed_min_m_s=5.0, c=1)] * 2
            + [dict(crossing_class="no_crossing", strict_pass=False, chord_speed_min_m_s=5.0, c=-1)] * 5)
    df = pd.DataFrame(rows).assign(g0p80_outcome=C.ACCEPTED)
    r = C.control_availability(df, {"X": "c"}, 0.8, CFG)["X"]
    assert r["cases"] == 8 and r["controls_usable_at_most_1to1"] == 4 and r["max_control_weight"] == 3.0
    assert r["achieved_ratio"] == 0.5


def test_framewise_audit_finds_what_chains_cannot():
    tr = {**crossing(0.1), 2: (0 * T + 5, 2 * (T - 14)), 3: (0 * T + 5.3, 2 * (T - 14))}
    tr[3] = (np.where(np.abs(T - 14) < 1.5, 5.3, 50.0), tr[3][1])      # 3 runs beside 2 (0.3 m) near 14 s, far otherwise
    w = world(tr, drop=lambda s, i: s == 3 and 186 <= i < 236)           # 3 has no valid segment while beside 2
    c, mask, dup, seg, ch = w
    df = cands(w)
    F, rep = C.framewise_audit(c, mask, dup, seg, df, CFG)
    m01 = F[(F.slot_a == 0) & (F.slot_b == 1)]
    assert len(m01) == 1 and bool(m01.iloc[0]["matched_to_candidate"])
    u = F[(F.slot_a == 2) & (F.slot_b == 3) & ~F.matched_to_candidate]
    assert len(u) >= 1 and "no_valid_segment_b" in u.iloc[0]["unmatched_reason"]
    assert rep["framewise_unmatched"] >= 1


# ---------------------------------------------------------------- three states and cohort links

def _events(w, variant="linear"):
    c, mask, dup, seg, ch = w
    cx, cy, _ = W.coefficients(seg, variant)
    h, _ = W.period_hits(seg, cx, cy, 1, 0.0, WCFG, variant)
    ev = pd.DataFrame(dict(event_id=[f"e{i}" for i in range(len(h["ta"]))], match_id=1, period=1, slot_a=h["sa"].astype(int),
                           slot_b=h["sb"].astype(int), seg_a_row0=seg["row0"][h["ga"].astype(int)], seg_b_row0=seg["row0"][h["gb"].astype(int)],
                           t_a_s=h["ta"], t_b_s=h["tb"], x_m=h["x"], y_m=h["y"], est_at_a=False, est_at_b=False))
    ev["abs_dt_s"] = (ev.t_a_s - ev.t_b_s).abs()
    ev["t_event_s"] = (ev.t_a_s + ev.t_b_s) / 2
    for x in LCFG.taus_s:
        ev[f"a_tau_{L.key(x)}"] = ev.abs_dt_s <= x + 1e-12
    D = L.event_descriptors(seg, ch, cx, cy, WCFG.pchip_substeps if variant == "pchip" else 1, ev, LCFG)
    return ev, D


def test_three_states_truncation_and_caps():
    # B's chain ends 0.05 s after its own arrival: arr neighbourhood truncated after; F2 observed min > R -> UNDETERMINED
    tc = 10.0
    w = world(crossing(0.6), drop=lambda s, i: s == 1 and i == int(round((tc + 0.65) * FPS)))
    ev, D = _events(w)
    S = L.method_states(ev, D, LCFG)
    r = D.iloc[0]
    h = L.key(LCFG.primary_h_s)
    assert bool(r[f"arr_h{h}_truncated"]) and r[f"arr_h{h}_trunc_side"] == "after" and r[f"arr_h{h}_trunc_cause"] == "b_chain_end"
    assert S.iloc[0][f"F2_arr_h{h}_R0p25"] == L.UNDETERMINED
    assert S.iloc[0]["F1_tau0p30"] == L.FAIL and S.iloc[0][f"F3_tau0p30_arr_h{h}_R0p25"] == L.FAIL    # F1 fails -> F3 FAIL
    M = L.method_masks(ev, D, LCFG)
    assert L.check_states(ev, D, M, S, LCFG)["all_pass"]


@pytest.mark.parametrize("tb, drop_at, state, cause", [
    (10.6, None, L.PASS, ""),                          # B passes while A still stands on P: proven PASS despite the cap
    (12.0, None, L.UNDETERMINED, "a_end_search_limit"),  # A still on P when the 2.5 s search stops; B arrives 3 s later
    (12.0, 11.0, L.UNDETERMINED, "a_end_chain_end"),     # A's chain ends while on P
    (12.0, "leave", L.FAIL, ""),                       # A leaves P at 10 s: the facing end is observed -> FAIL
])
def test_f4_states_with_standing_player(tb, drop_at, state, cause):
    ax = np.where(T < 9.0, -3 + T / 3, 0.0)                                 # walks in, stands on P from 9 s
    if drop_at == "leave":
        ax = np.where(T > 10.0, (T - 10.0) * 2, ax)
    w = world({0: (ax, 0 * T), 1: (0 * T, 2 * (T - tb))},
              drop=(lambda s, i: s == 0 and i == int(round(drop_at * FPS))) if isinstance(drop_at, float) else None)
    ev, D = _events(w)
    S = L.method_states(ev, D, LCFG)
    assert len(ev) >= 1                                    # arriving at and leaving P are two endpoint intersections
    assert (S["F4_e0p05_tau0p30"] == state).all() and (S["F1_tau0p30"] == L.FAIL).all()
    assert (D["occ_e0p05_facing_cap_cause"] == cause).all()
    assert L.check_states(ev, D, L.method_masks(ev, D, LCFG), S, LCFG)["all_pass"]


def test_states_consistent_on_random_trajectories_and_dense_check():
    rng = np.random.default_rng(5)
    tr = {s: (5 + np.cumsum(rng.normal(0, 0.2, N)), 5 + np.cumsum(rng.normal(0, 0.2, N))) for s in range(3)}
    w = world(tr, drop=lambda s, i: s == 2 and i % 97 == 0)
    ev, D = _events(w)
    assert len(ev) > 5
    M, S = L.method_masks(ev, D, LCFG), L.method_states(ev, D, LCFG)
    chk = L.check_descriptors(ev, D, M, LCFG)
    assert chk["all_pass"], chk
    assert (S.values == L.UNDETERMINED).any() or not D[f"arr_h{L.key(LCFG.primary_h_s)}_truncated"].any()
    c, mask, dup, seg, ch = w
    cx, cy, _ = W.coefficients(seg, "linear")
    dc = L.dense_check(seg, ch, cx, cy, 1, ev, D, LCFG, 10, 1, c=c)
    assert dc["distance_ok"] and dc["occupancy_ok"] and dc["curve_ok"] and dc["raw_ok"] and dc["raw_vertices_checked"] > 0
    evp, Dp = _events(w, "pchip")                       # PCHIP: knots on the cubic, clipped track ends excluded from the curve check
    cxp, cyp, _ = W.coefficients(seg, "pchip")
    dp = L.dense_check(seg, ch, cxp, cyp, WCFG.pchip_substeps, evp, Dp, LCFG, 10, 2, c=c)
    assert dp["curve_ok"] and dp["raw_ok"] and dp["distance_ok"] and dp["occupancy_ok"], dp


def test_link_events_and_coverage():
    rng = np.random.default_rng(13)
    tr = {s: (np.cumsum(rng.normal(0, 0.15, N)) + s * 0.2, np.cumsum(rng.normal(0, 0.15, N))) for s in range(3)}
    w = world(tr)
    ev, D = _events(w)
    S = L.method_states(ev, D, LCFG)
    cohorts = L.cohort_columns(LCFG)
    df, cov = C.link_events(cands(w), ev, D, S, cohorts, CFG, LCFG.primary_h_s)
    assert cov["all_covered"], cov
    f2 = S[cohorts["F2_primary"]].to_numpy() == L.PASS
    assert cov["primary_pass_covered"]["F2_primary"]["pass_events"] == int(f2.sum())
    assert set(df["cohort_F2_primary"].unique()) <= {-1, 0, 1, 2}
    assert int(df["n_linked_events"].sum()) == cov["events_linked"]


# ---------------------------------------------------------------- crash and resume

def test_run_matches_crash_resume_and_tamper(tmp_path):
    calls = []

    def fn(m, d):
        calls.append(m)
        L.atomic_parquet(pd.DataFrame(dict(x=[int(m)])), d / "a.parquet")
        L.atomic_write_text(d / "b.json", json.dumps(dict(m=m)))
        return dict(match=m, n=1)

    out = tmp_path / "run"
    with pytest.raises(L.SimulatedCrash):
        L.run_matches(out, ["1", "2", "3"], fn, crash_after=1, log=lambda *_: None)
    assert calls == ["1"] and (out / "matches" / "1" / "match_done.json").exists()
    (out / "matches" / "2.partial").mkdir()                                  # an interrupted match leaves a partial directory
    (out / "matches" / "2.partial" / "a.parquet").write_bytes(b"junk")
    res = L.run_matches(out, ["1", "2", "3"], fn, log=lambda *_: None)
    assert calls == ["1", "2", "3"] and set(res) == {"1", "2", "3"} and res["1"] == dict(match="1", n=1)
    assert not (out / "matches" / "2.partial").exists()
    assert pd.read_parquet(out / "matches" / "2" / "a.parquet")["x"].tolist() == [2]
    (out / "matches" / "3" / "b.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA-256"):
        L.run_matches(out, ["1", "2", "3"], fn, log=lambda *_: None)
    (out / "matches" / "1" / "extra.txt").write_text("x", encoding="utf-8")
    with pytest.raises(RuntimeError, match="files differ"):
        L.verify_match_dir(out / "matches" / "1")


# ---------------------------------------------------------------- frozen held-out split

DEV40 = ("10502 10503 10506 10507 10510 10512 10514 10517 3812 3814 3815 3816 3818 3819 3821 3822 3823 3824 3826 3828 3829 3830 3831 "
         "3833 3834 3835 3836 3837 3838 3839 3840 3843 3844 3847 3849 3851 3852 3855 3858 3859").split()


def test_split_rule_reproduces_the_approved_heldout_set():
    from football_fatigue.window_intersections import split as SP
    sp = SP.make_split(DEV40, set(SP.EXPECTED_EXPOSED))
    assert tuple(sp["heldout"]) == tuple(sorted(SP.APPROVED_HELDOUT, key=int))
    assert len(sp["train_cv"]) == 32 and set(sp["heldout"]).isdisjoint(sp["train_cv"])
    assert set(sp["heldout"]) | set(sp["train_cv"]) == set(DEV40) and set(sp["heldout"]).isdisjoint(SP.EXPECTED_EXPOSED)
    assert SP.make_split(list(reversed(DEV40)), set(SP.EXPECTED_EXPOSED))["heldout"] == sp["heldout"]      # order independent
    with pytest.raises(ValueError):
        SP.make_split(DEV40, {"3829"})                                              # pilots must be excluded
