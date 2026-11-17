"""Independent audit of ``m11-sync-proximity-v2`` (``football_fatigue.proximity_audit``): synchronized geometry on synthetic tracks,
source / gap / jump checks on a synthetic RAW cache, readiness rules, the reported screenshot cases and the written audit."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from football_fatigue.encounters.trajectories import STATE_AVAILABLE
from football_fatigue import proximity_audit as A

REPO = Path(__file__).resolve().parents[1]
ROOT = str(REPO / "data" / "raw")
AUDIT = REPO / "data" / "processed" / A.SCHEMA
real = pytest.mark.skipif(not (AUDIT / "event_validation.csv").exists(), reason="audit outputs not written")


def tr(t, x, y):
    return np.asarray(t, float), np.asarray(x, float), np.asarray(y, float)


# ---------------------------------------------------------------- synchronized geometry

def test_nonzero_synchronized_proximity_without_intersection():
    a = tr([0, 1, 2], [-1, 0, 1], [0, 0, 0])
    b = tr([0, 1, 2], [-1, 0, 1], [0.2, 0.2, 0.2])                       # parallel, 20 cm apart: valid proximity, paths never cross
    g = A.grid(*a, *b)
    d, t = A.closest(g)
    assert d == pytest.approx(0.2) and t == 0.0                          # flat minimum: the earliest instant
    assert A.spatial_crossings(*a, *b) == [] and A.identical_seconds(g) == 0


def test_exact_synchronized_coincidence_is_a_single_instant():
    a = tr([0, 2], [-1, 1], [0, 0])
    b = tr([0, 2], [0, 0], [-1, 1])                                      # both pass (0, 0) at t = 1
    g = A.grid(*a, *b)
    d, t = A.closest(g)
    assert d == pytest.approx(0, abs=1e-12) and t == pytest.approx(1.0)
    assert A.identical_seconds(g) == 0                                   # a touch, not an identical interval
    (x, y, ta, tb), = A.spatial_crossings(*a, *b)
    assert (x, y, ta, tb) == pytest.approx((0, 0, 1, 1))


def test_geometric_crossing_at_different_times_is_not_an_encounter():
    a = tr([0, 2], [-1, 1], [0, 0])                                      # A at (0, 0) at t = 1
    b = tr([0, 4], [0, 0], [-2, 2])                                      # B at (0, 0) at t = 2
    (x, y, ta, tb), = A.spatial_crossings(*a, *b)
    assert (x, y) == pytest.approx((0, 0)) and abs(ta - tb) == pytest.approx(1.0)
    d, _ = A.closest(A.grid(*a, *b))
    assert d > 0.25                                                      # far apart at every common instant


def test_cloned_identical_positions_measured_as_interval():
    a = tr([0, 1, 2, 3], [0, 1, 2, 3], [0, 0, 0, 0])
    b = tr([0, 1, 2, 3], [0, 1, 2, 5], [1, 0, 0, 1])                     # identical on [1, 2]
    g = A.grid(*a, *b)
    assert A.identical_seconds(g) == pytest.approx(1.0) and A.closest(g)[1] == pytest.approx(1.0)
    assert A.identical_seconds(g, 1.5, 3) == pytest.approx(0.5)


def test_closest_restricted_to_the_episode():
    a = tr([0, 1, 2, 3, 4], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0])
    b = tr([0, 1, 2, 3, 4], [0.05, 1, 0.2, 1, 1], [0, 0, 0, 0, 0])       # a closer approach at t = 0 belongs to another episode
    g = A.grid(*a, *b, extra=(1.8, 2.2))
    assert A.closest(g, 1.8, 2.2) == pytest.approx((0.2, 2.0)) and A.closest(g)[0] == pytest.approx(0.05)


# ---------------------------------------------------------------- source / gap / jump checks on a synthetic RAW cache

def cache(n=40, fps=29.97):
    t = np.arange(n) / fps
    x = np.zeros((n, 1))
    x[:, 0] = np.arange(n) * 0.1
    return dict(frame_num=np.arange(n) + 100, video_time_s=t, canonical_period=np.ones(n, np.int8), x=x, y=np.zeros((n, 1)),
                state=np.full((n, 1), STATE_AVAILABLE, np.int8), visibility=np.ones((n, 1), np.int8),
                slot_side=np.array(["home"]), slot_jersey=np.array([7]), slot_player_id=np.array([70]), slot_team_id=np.array([1]))


def side_of(c, rows):
    r = np.asarray(rows)
    return dict(t=c["video_time_s"][r], x=c["x"][r, 0].copy(), y=c["y"][r, 0].copy(), source_row=r, frame=c["frame_num"][r],
                visibility=c["visibility"][r, 0], before_t_star=np.zeros(len(r), bool))


def test_source_checks_valid_track():
    c = cache()
    assert A.source_problems(side_of(c, [0, 1, 2, 5, 9]), 0, 1, c) == []


def test_source_checks_detect_tampering_gaps_and_jumps():
    c = cache()
    s = side_of(c, [0, 1, 2, 3])
    s["x"][2] += 0.01                                                    # 1 cm off the raw value
    assert "source_row_mismatch" in A.source_problems(s, 0, 1, c)
    s = side_of(c, [0, 1, 2, 3])
    s["visibility"][1] = 2                                               # relabelled
    assert "source_row_mismatch" in A.source_problems(s, 0, 1, c)
    g = cache()
    g["state"][2:20, 0] = 0                                              # 0.6 s without an available sample
    assert "unobserved_gap_joined" in A.source_problems(side_of(g, [0, 1, 20, 21]), 0, 1, g)
    j = cache()
    j["x"][3:, 0] += 5.0                                                 # 5 m in one frame
    assert "jump_joined" in A.source_problems(side_of(j, [0, 1, 2, 3, 4]), 0, 1, j)
    assert "source_row_mismatch" in A.source_problems(side_of(c, [0, 1]), 0, 2, c)   # wrong period


def test_segmentation_across_a_valid_gap_is_kept():
    c = cache()
    c["state"][2:16, 0] = 0                                              # 0.47 s unobserved: still one trajectory (Step 1 rule)
    assert A.source_problems(side_of(c, [0, 1, 16, 17]), 0, 1, c) == []


# ---------------------------------------------------------------- readiness on synthetic records

def record(a, b, *, cut_after_b=None, vis=1, coincident=False, d_min=None, window=(0.0, 4.0)):
    ta, xa, ya = a
    tb, xb, yb = b
    d, t_star = A.closest(A.grid(*a, *b))
    pa = [float(np.interp(t_star, ta, xa)), float(np.interp(t_star, ta, ya))]
    pb = [float(np.interp(t_star, tb, xb)), float(np.interp(t_star, tb, yb))]
    full = dict(complete_before=True, complete_after=True, cut_before=None, cut_after=None)
    ctx_b = dict(full, complete_after=cut_after_b is None, cut_after=cut_after_b)
    pl = lambda p, slot: dict(position_at_t_star=p, slot=slot, side="home", jersey=slot, player_id=slot, team_id=1,  # noqa: E731
                              duplicate_provider_position=False)
    dm = d if d_min is None else d_min
    lo, hi = A.support(ta, tb)

    def edge(t_out: float) -> float:                                     # bisection to d = R between t* and t_out (or the support end)
        if A.d_at(*a, *b, t_out) <= 0.25:
            return t_out
        u, w = t_star, t_out
        for _ in range(200):
            m = (u + w) / 2
            u, w = (m, w) if A.d_at(*a, *b, m) <= 0.25 else (u, m)
        return u

    t0, t1 = edge(lo), edge(hi)
    return dict(match_id=1, event_id="e", period=1, threshold_m=0.25, t_star_s=t_star, t_start_s=t0, t_end_s=t1, duration_s=t1 - t0,
                d_min_m=dm, midpoint=[(pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2], player_a=pl(pa, 0), player_b=pl(pb, 1),
                pair_quality="observed/observed" if vis == 1 else "estimated/estimated", team_relation="opposite_teams",
                exact_coincidence=dm == 0, coincident_interval=coincident, contact_class="sustained", support_cut_at_start=False,
                support_cut_at_end=bool(cut_after_b and t1 >= hi),
                context=dict(window_start_s=window[0], window_end_s=window[1], complete=cut_after_b is None, critical_flags=[],
                             player_a=full, player_b=ctx_b))


def sides(a, b, rec, vis=1):
    out = []
    for t, x, y in (a, b):
        out.append(dict(t=t, x=x, y=y, visibility=np.full(len(t), vis), before_t_star=t < rec["t_star_s"]))
    g = A.grid(*a, *b, extra=[rec["t_start_s"], rec["t_star_s"], rec["t_end_s"]])
    return out[0], out[1], dict(t=g["t"], d=g["d"], a_is_sample=np.isin(g["t"], a[0]), b_is_sample=np.isin(g["t"], b[0]))


T = np.linspace(0, 4, 41)


def test_valid_observed_encounter_is_presentation_ready():
    a = tr(T, T - 2, np.zeros_like(T))
    b = tr(T, np.zeros_like(T), T - 2 + 0.1)                             # crossing paths, 10 cm apart at t* = 2
    rec = record(a, b)
    v = A.audit_event(rec, *sides(a, b, rec), cache=None)
    assert v["mathematically_valid"] and v["presentation_ready"] and v["reasons"] == ""
    assert v["encounter_type"] == "closest_synchronized_approach" and v["start_boundary"] == v["end_boundary"] == "verified_d_eq_R"


def test_missing_outgoing_support_is_not_ready():
    a = tr(T, T - 2, np.zeros_like(T))
    w = T <= 2.0                                                         # B's valid track ends at the encounter
    b = tr(T[w], np.zeros(w.sum()), T[w] - 2 + 0.1)
    rec = record(a, b, cut_after_b="implausible_jump")
    v = A.audit_event(rec, *sides(a, b, rec), cache=None)
    assert v["mathematically_valid"] and not v["outgoing_complete"] and not v["presentation_ready"]
    assert v["shared_post_s"] == pytest.approx(0.05) and v["end_boundary"] == "support_cut" and "outgoing_incomplete_b" in v["reasons"]


def test_cloned_estimated_coordinates_are_artifacts():
    a = tr(T, T - 2, np.zeros_like(T))
    b = tr(T, np.where(T >= 2, T - 2, 0), np.where(T >= 2, 0, T - 2))    # B identical to A from t = 2 on
    rec = record(a, b, vis=2, coincident=True, d_min=0.0)
    v = A.audit_event(rec, *sides(a, b, rec, vis=2), cache=None)
    assert v["provider_artifact_suspected"] and v["encounter_type"] == "potential_tracking_ambiguity"
    assert v["identical_seconds_context"] == pytest.approx(2.0) and not v["presentation_ready"]
    assert "estimated_at_t_star" in v["reasons"] and "identical_positions" in v["reasons"]


def test_mismatches_are_reported_not_hidden():
    a = tr(T, T - 2, np.zeros_like(T))
    b = tr(T, np.zeros_like(T), T - 2 + 0.1)
    rec = record(a, b)
    rec["d_min_m"] = 0.05                                                # wrong minimum
    rec["player_a"]["position_at_t_star"] = [0.3, 0.0]                   # wrong A(t*)
    v = A.audit_event(rec, *sides(a, b, rec), cache=None)
    assert not v["mathematically_valid"] and {"d_min_mismatch", "position_mismatch"} <= set(v["reasons"].split(";"))


# ---------------------------------------------------------------- real audit outputs and the reported cases

@pytest.fixture(scope="module")
def audit_rows():
    with open(AUDIT / "event_validation.csv", encoding="utf-8") as f:
        return {(r["match_id"], r["event_id"]): r for r in csv.DictReader(f)}


@real
def test_screenshot_cases(audit_rows):
    a = audit_rows[("3812", "7d69f374b095e733")]                         # case A / B: P1 47:40.460, outgoing drawn on top of each other
    assert a["provider_artifact_suspected"] == "True" and "identical_positions" in a["reasons"] and "incoming_incomplete_b" in a["reasons"]
    assert float(a["shared_pre_s"]) == pytest.approx(0.467134) and float(a["shared_post_s"]) == pytest.approx(2.002002)
    assert float(a["requested_post_s"]) == pytest.approx(2.039539) and a["end_boundary"] == "beyond_exported_window"
    c = audit_rows[("3812", "548ad35e88e51867")]                         # case C: no incoming track for either player, cloned after t*
    assert float(c["shared_pre_s"]) == 0 and c["incoming_complete"] == "False" and c["start_boundary"] == "support_cut"
    assert float(c["identical_seconds_context"]) == pytest.approx(2.002002)
    d = audit_rows[("3812", "22092c0d35089442")]                         # case D: exact synchronized coincidence, estimated tracks
    assert d["encounter_type"] == "exact_synchronized_coincidence" and d["provider_artifact_suspected"] == "False"
    assert d["mathematically_valid"] == d["context_complete"] == "True" and d["presentation_ready"] == "False"
    assert "estimated_at_t_star" in d["reasons"]
    for r in (a, c, d):
        assert r["mathematically_valid"] == r["tracking_support_valid"] == "True"


@real
def test_written_audit_is_consistent(audit_rows):
    s = json.loads((AUDIT / "summary.json").read_text(encoding="utf-8"))
    rows = list(audit_rows.values())
    assert s["encounters"] == len(rows) == json.loads((REPO / "data" / "processed" / A.SOURCE / "development-index" / "summary.json")
                                                      .read_text(encoding="utf-8"))["episodes"]
    for f in A.STATUS_FIELDS:
        assert s[f] == sum(r[f] == "True" for r in rows)
    for r in rows:
        ready = r["presentation_ready"] == "True"
        assert ready == (r["reasons"] == "")                             # ready exactly when no reason code applies
        if ready:
            assert r["pair_quality"] == "observed/observed" and r["provider_artifact_suspected"] == "False"
            assert min(float(r["a_observed_share"]), float(r["b_observed_share"])) >= A.MIN_OBSERVED_SHARE
            assert min(float(r["a_path_m"]), float(r["b_path_m"])) >= A.MIN_PATH_M
        assert set(x for x in r["reasons"].split(";") if x) <= set(A.REASONS)
    ds = json.loads((AUDIT / "dataset_checks.json").read_text(encoding="utf-8"))
    assert all(v is True for k, v in ds.items() if k not in ("per_match", "high_quality_not_ready"))
    assert all(c["manifest_hashes"] and c["threshold_nesting"] and c["ids_reproduce"] and c["npz_rows"] for c in ds["per_match"].values())


@real
def test_audit_reproduces_for_one_match(audit_rows):
    m = A.audit_match(ROOT, "3812")                                      # recomputed now, compared with the written file
    for r in m["rows"]:
        w = audit_rows[("3812", r["event_id"])]
        for k in ("presentation_ready", "mathematically_valid", "tracking_support_valid", "reasons", "shared_pre_s", "shared_post_s"):
            assert str(r[k]) == w[k]


def test_validation_and_test_matches_refused():
    split = json.loads((REPO / "data" / "processed" / "m11-encounters-v1" / "splits.json").read_text(encoding="utf-8"))
    with pytest.raises(RuntimeError):
        A.audit_match(ROOT, str(split["splits"]["validation"][0]))
    with pytest.raises(RuntimeError):
        A.audit_match(ROOT, str(split["splits"]["test"][0]))
