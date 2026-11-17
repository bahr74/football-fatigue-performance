"""Unit tests for the M11 window-intersection extraction (synthetic trajectories; no data files needed)."""

import numpy as np
import pytest

from football_fatigue.window_intersections import core as W

CFG = W.WindowConfig()


def make_seg(tracks: dict[int, tuple], period: int = 1) -> dict[str, np.ndarray]:
    """Segments from {slot: (t, x, y)} sample arrays; each slot is one unbroken chain (fake row numbers)."""
    cols = {k: [] for k in ("slot", "row0", "row1", "t0", "t1", "x0", "y0", "x1", "y1")}
    row = 0
    for s in sorted(tracks):
        t, x, y = (np.asarray(v, float) for v in tracks[s])
        for i in range(len(t) - 1):
            for k, v in zip(cols, (s, row + i, row + i + 1, t[i], t[i + 1], x[i], y[i], x[i + 1], y[i + 1])):
                cols[k].append(v)
        row += len(t) + 10
    seg = {k: np.array(v) for k, v in cols.items()}
    seg["slot"], seg["row0"], seg["row1"] = (seg[k].astype(int) for k in ("slot", "row0", "row1"))
    seg["period"] = np.full(len(seg["slot"]), period)
    seg["stationary"] = np.zeros(len(seg["slot"]), bool)
    return W.finalize_segments(seg)


def hits_and_windows(seg, variant="linear", pend=40.0):
    cx, cy, sup = W.coefficients(seg, variant)
    h, _ = W.period_hits(seg, cx, cy, 1, 0.0, CFG, variant)
    k, comp = W.period_anchors(0.0, pend, CFG.half_s)
    return h, W.assign_windows(h, 0.0, k[comp], CFG.half_s, CFG.max_extra), (cx, cy)


def test_seg_intersect_cases():
    p = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], float)
    r = np.array([[2, 0], [2, 0], [2, 0], [2, 0], [0, 0]], float)
    q = np.array([[1, -1], [2, -1], [0, 1], [1, 0], [1, -1]], float)
    s = np.array([[0, 2], [0, 2], [2, 0], [2, 0], [0, 2]], float)
    ok, ua, ub, x, y, kind = W.seg_intersect(p, r, q, s, 1e-6)
    assert ok.tolist() == [True, True, False, True, False]          # proper, endpoint, parallel, collinear, degenerate
    assert kind[0] == W.PROPER and kind[1] == W.ENDPOINT and kind[3] == W.COLLINEAR
    assert (x[0], y[0], ua[0], ub[0]) == pytest.approx((1, 0, 0.5, 0.5))


def test_crossing_times_and_window_membership():
    t = np.arange(0, 21, 1.0)
    seg = make_seg({0: (t, t, 0 * t), 1: (t, 5 + 0 * t, t - 7)})   # A at (5,0) at t=5, B at (5,0) at t=7
    h, dets, _ = hits_and_windows(seg)
    assert len(h["ta"]) == 1
    assert (h["ta"][0], h["tb"][0], h["x"][0], h["y"][0]) == pytest.approx((5, 7, 5, 0))
    assert [d["anchor"] for d in dets] == [5, 6, 7]                  # both arrivals inside [k - 2.5, k + 2.5]
    assert all(d["sa"] == 0 and d["sb"] == 1 for d in dets)


def test_pairs_are_unordered_and_never_double_counted():
    t = np.arange(0, 21, 1.0)
    a, b = (t, t, 0 * t), (t, 5 + 0 * t, t - 6)
    _, d1, _ = hits_and_windows(make_seg({0: a, 1: b}))
    _, d2, _ = hits_and_windows(make_seg({0: b, 1: a}))
    assert [(d["anchor"], d["sa"], d["sb"]) for d in d1] == [(d["anchor"], d["sa"], d["sb"]) for d in d2]
    assert len({(d["anchor"], d["sa"], d["sb"]) for d in d1}) == len(d1)


def test_first_intersection_selected_and_extras_counted():
    t = np.arange(0, 21, 0.5)
    yb = np.where(t < 10, t - 8.0, 12.0 - t)                        # B crosses y=0 upwards at t=8 and back down at t=12
    seg = make_seg({0: (t, t, 0 * t), 1: (t, 9 + 0.5 * (t - 10), yb)})
    h, dets, _ = hits_and_windows(seg)
    assert len(h["ta"]) == 2
    by = {d["anchor"]: d for d in dets}
    both = [k for k, d in by.items() if d["n_int"] == 2]
    assert both, "some window must contain both crossings"
    first = int(W.order_key(h["ta"], h["tb"], h["x"], h["y"])[0])
    for k in both:
        assert by[k]["hit"] == first and len(by[k]["extra"]) == 1


def test_vertex_contact_is_one_intersection():
    t = np.arange(0, 11, 1.0)
    seg = make_seg({0: (t, t, 0 * t), 1: (t + 0.5, 4 + 0 * t, t - 4.5)})   # B passes exactly through A's vertex (4, 0)
    h, _, _ = hits_and_windows(seg)
    assert len(h["ta"]) == 1 and h["x"][0] == pytest.approx(4.0)


def test_pchip_passes_through_samples_and_stays_in_interval_box():
    rng = np.random.default_rng(3)
    t = np.arange(0, 6, 0.0667)
    seg = make_seg({0: (t, np.cumsum(rng.normal(0, 0.3, len(t))), np.cumsum(rng.normal(0, 0.3, len(t))))})
    cx, cy, sup = W.coefficients(seg, "pchip")
    assert sup.all()
    g = np.arange(len(seg["t0"]))
    px, py, _ = W.sub_points(seg, cx, cy, g, 50)
    assert np.allclose(W.evalc(cx, seg["dt"]), seg["x1"]) and np.allclose(W.evalc(cy, 0 * seg["dt"]), seg["y0"])
    assert (px.max(1) <= np.maximum(seg["x0"], seg["x1"]) + 1e-9).all() and (px.min(1) >= np.minimum(seg["x0"], seg["x1"]) - 1e-9).all()
    one = make_seg({0: ([0, 1], [0, 1], [0, 1])})
    assert not W.coefficients(one, "pchip")[2].any()                # a one-segment chain is not PCHIP-supported (stays linear)
    assert (W.coefficients(seg, "linear")[2] == sup).all()          # the support flag does not depend on the variant


@pytest.mark.parametrize("variant", ["linear", "pchip"])
def test_fast_windows_match_brute_force(variant):
    rng = np.random.default_rng(7)
    t = np.arange(0, 40, 0.0667)
    tracks = {s: (t, 5 + np.cumsum(rng.normal(0, 0.25, len(t))), 5 + np.cumsum(rng.normal(0, 0.25, len(t)))) for s in range(4)}
    seg = make_seg(tracks)
    h, dets, (cx, cy) = hits_and_windows(seg, variant, pend=float(t[-1]))
    assert len(h["ta"]) > 0
    K = CFG.pchip_substeps if variant == "pchip" else 1
    by = {(d["anchor"], d["sa"], d["sb"]): d for d in dets}
    for k in range(3, int(t[-1] - 3)):
        for sa in range(4):
            for sb in range(sa + 1, 4):
                got = W.brute_force_window(seg, cx, cy, sa, sb, 1, k - 2.5, k + 2.5, K, CFG)
                d = by.get((k, sa, sb))
                if d is None:
                    assert got == []
                else:
                    assert len(got) == d["n_int"]
                    assert got[0] == pytest.approx((h["ta"][d["hit"]], h["tb"][d["hit"]]), abs=1e-6)


def test_compare_variants_and_encounters():
    base = dict(match_id=1, period=1, slot_a=0, slot_b=1, x_m=0.0, y_m=0.0, abs_dt_s=0.1, info_flags="[]")
    lin = [dict(base, event_id="l1", t_event_s=10.0, variant="linear"), dict(base, event_id="l2", t_event_s=10.6, x_m=0.3, variant="linear")]
    pch = [dict(base, event_id="p1", t_event_s=10.05, variant="pchip", abs_dt_s=0.12)]
    r = W.compare_variants(lin, pch)
    assert r == dict(matched=1, linear_only=1, pchip_only=0)
    assert lin[0]["pchip_match"] == "p1" and lin[1]["pchip_match"] is None
    assert lin[0]["pchip_abs_dt_change_s"] == pytest.approx(0.02)
    W.encounters_and_sharing(lin, CFG)
    assert lin[0]["encounter_id"] == lin[1]["encounter_id"]          # 0.6 s and 0.3 m apart: one encounter, two events kept
