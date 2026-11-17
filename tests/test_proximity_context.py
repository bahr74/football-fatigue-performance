"""Step 1 segmentation, event-centred contexts (m11-sync-proximity-v3) and their audit.

Synthetic: a small RAW cache is run through the real Step 1 quality pass, the real trajectory segmentation and the v3 context
builder, then audited (including tampered contexts). Real: the reported development cases and the written v3 outputs."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from football_fatigue import proximity_audit as A
from football_fatigue import proximity_audit_v3 as V
from football_fatigue import proximity_context as C
from football_fatigue import step1_verify as S1
from football_fatigue.encounters import quality as Q
from football_fatigue.encounters.trajectories import STATE_AVAILABLE, STATE_MISSING, RawCache
from football_fatigue.ingestion import approved_output_root
from football_fatigue.intersections import exact as X
from football_fatigue.intersections import proximity as P

FPS = 29.97
ROOT = str(Path("data/raw").resolve())


# ---------------------------------------------------------------- synthetic RAW cache

def make_cache(xa, ya, xb, yb, *, period=None, vis_a=None, vis_b=None, t0=100.0) -> RawCache:
    """Two slots (home #1, away #2), one sample per 29.97 fps frame, positions rounded to whole mm (NaN = missing)."""
    n = len(xa)
    t = np.round(t0 + np.arange(n) / FPS, 6)
    x, y = np.c_[xa, xb].astype(float), np.c_[ya, yb].astype(float)
    x, y = np.round(x, 3), np.round(y, 3)
    state = np.where(np.isfinite(x) & np.isfinite(y), STATE_AVAILABLE, STATE_MISSING).astype(np.int8)
    vis = np.ones((n, 2), np.int8)
    if vis_a is not None:
        vis[:, 0] = vis_a
    if vis_b is not None:
        vis[:, 1] = vis_b
    per = np.ones(n, np.int8) if period is None else np.asarray(period, np.int8)
    return RawCache(match_id=1, fps=FPS, tracking_path="synthetic", quarantine_periods=(), frame_num=np.arange(n, dtype=np.int64) + 1000,
                    video_time_s=t, source_period=per.astype(np.int16), canonical_period=per, record_first=np.arange(n, dtype=np.int64),
                    record_count=np.ones(n, np.int16), frame_gap_before=np.full(n, -1, np.int32), frame_qflags=np.zeros(n, np.int64),
                    slot_side=np.array(["home", "away"]), slot_jersey=np.array([1, 2], np.int16), slot_player_id=np.array([11, 22]),
                    slot_team_id=np.array([1, 2]), x=x, y=y, state=state, visibility=vis, confidence=np.full((n, 2), 3, np.int8),
                    attribution=np.ones((n, 2), np.int8), qflags=np.zeros((n, 2), np.int64))


def cadence15(v: np.ndarray) -> np.ndarray:
    """Provider cadence: each value repeated for two frames."""
    v = np.asarray(v, float).copy()
    v[1::2] = v[0::2][:len(v[1::2])]
    return v


def build(c: RawCache):
    q = Q.assess(c, [])
    seg, _ = X.trajectories(c, q.mask, q.steps)
    return q, seg


def record(c: RawCache, seg, t_star: float, R: float | None = None) -> dict:
    """A minimal v2-like 0.25 m record at ``t_star`` (an instant on both players' valid segments)."""
    key = {}
    pl = {}
    for side, slot in (("a", 0), ("b", 1)):
        g = int(np.flatnonzero((seg["slot"] == slot) & (seg["t0"] <= t_star) & (seg["t1"] >= t_star))[0])
        f = (t_star - seg["t0"][g]) / (seg["t1"][g] - seg["t0"][g])
        pos = [float(seg["x0"][g] + f * (seg["x1"][g] - seg["x0"][g])), float(seg["y0"][g] + f * (seg["y1"][g] - seg["y0"][g]))]
        key[side] = g
        pl[side] = dict(slot=slot, side=str(c.slot_side[slot]), jersey=int(c.slot_jersey[slot]), player_id=int(c.slot_player_id[slot]),
                        team_id=int(c.slot_team_id[slot]), status_at_t_star="observed", duplicate_provider_position=False,
                        segment_at_t_star=dict(row0=int(seg["row0"][g]), row1=int(seg["row1"][g])), position_at_t_star=pos)
    d = float(np.hypot(pl["a"]["position_at_t_star"][0] - pl["b"]["position_at_t_star"][0],
                       pl["a"]["position_at_t_star"][1] - pl["b"]["position_at_t_star"][1]))
    p = int(c.canonical_period[seg["row0"][key["a"]]])
    tp = c.video_time_s[c.canonical_period == p]
    return dict(event_id="synthetic", match_id=1, period=p, t_star_s=t_star, t_start_s=t_star, t_end_s=t_star, duration_s=0.0,
                contact_class="point", threshold_m=d if R is None else R, d_min_m=d, exact_coincidence=d == 0, coincident_interval=False,
                midpoint=[(pl["a"]["position_at_t_star"][0] + pl["b"]["position_at_t_star"][0]) / 2,
                          (pl["a"]["position_at_t_star"][1] + pl["b"]["position_at_t_star"][1]) / 2],
                pair_quality="observed/observed", team_relation="opposite_teams", support_cut_at_start=False, support_cut_at_end=False,
                period_start_s=float(tp.min()), period_end_s=float(tp.max()), player_a=pl["a"], player_b=pl["b"])


def contexts(c: RawCache, rec: dict):
    q, seg = build(c)
    starts, ends = C.run_limits(seg)
    key = {(int(s), int(a), int(b)): g for g, (s, a, b) in enumerate(zip(seg["slot"], seg["row0"], seg["row1"]))}
    tr, cx = C.event_context(seg, c, q.mask, P.step_index(q.steps), starts, ends, key, rec)
    return q, seg, tr, cx


def audit(c: RawCache, q, rec: dict, tr: dict, cx: dict) -> dict:
    sides = {}
    for k, side in (("a", tr["a"]), ("b", tr["b"])):
        sides[k] = dict(t=side["t"], x=side["x"], y=side["y"], source_row=side["row"], frame=side["frame"], visibility=side["vis"],
                        before_t_star=side["t"] < rec["t_star_s"], is_boundary=side["edge"])
    sy = tr["sync"]
    sync = dict(t=sy[:, 0], d=sy[:, 5], a_is_sample=sy[:, 6].astype(bool), b_is_sample=sy[:, 7].astype(bool))
    cache = {k: getattr(c, k) for k in A.CACHE_KEYS}
    tp = c.video_time_s[c.canonical_period == rec["period"]]
    return V.audit_event(rec, cx, sides["a"], sides["b"], sync, cache, q.mask, (float(tp.min()), float(tp.max())))


N = 600                                                                  # 20 s at 29.97 fps
TT = np.arange(N) / FPS


def crossing(y_b: float = 0.3):
    """A runs +x at 2 m/s, B runs -x at 2 m/s on a parallel line y_b: closest at t = 7.5 s (frame ~ 224.8)."""
    return cadence15(2 * TT), np.zeros(N), cadence15(30 - 2 * TT), np.full(N, y_b)


# ---------------------------------------------------------------- Step 1 rules

def test_step1_gap_segmentation_and_independent_steps():
    xa, ya, xb, yb = crossing()
    xb[100:116] = np.nan                                                 # 16 missing frames: 17 frames (0.567 s) unobserved -> cut
    xa[300:312] = np.nan                                                 # 12 missing frames: 0.434 s unobserved -> still joined
    c = make_cache(xa, ya, xb, yb)
    q, seg = build(c)
    rep, upd, held = S1.cadence_mine(c)
    mine = S1.steps_mine(c, upd)
    for k in ("slot", "row_from", "row_from_last", "row_to", "valid", "jump"):
        assert np.array_equal(mine[k], q.steps[k]), k                    # independent re-derivation equals Step 1
    gap_b = (q.steps["slot"] == 1) & (q.steps["row_to"] == 116)
    assert gap_b.sum() == 1 and not q.steps["valid"][gap_b][0]
    gap_a = (q.steps["slot"] == 0) & (q.steps["row_to"] == 312)
    assert q.steps["valid"][gap_a][0]
    starts, ends = C.run_limits(seg)
    b = np.flatnonzero(seg["slot"] == 1)
    assert len(set(starts[b].tolist())) == 2                             # B: two runs, split at the gap
    a = np.flatnonzero(seg["slot"] == 0)
    assert len(set(starts[a].tolist())) == 1                             # A: one run across the 0.43 s gap
    assert S1.segmentation_checks(c, q.mask, q.steps)["ok"]


def test_step1_jump_segmentation():
    xa, ya, xb, yb = crossing()
    xa[300:] += 3.0                                                      # 3 m in one update: > 12 m/s and > 1 m
    xb[400:] += 0.95                                                     # 0.82 m net in 2 frames: over-speed noise, not a jump
    c = make_cache(xa, ya, xb, yb)
    q, seg = build(c)
    j = q.steps["jump"]
    assert j.sum() == 1 and q.steps["slot"][j][0] == 0 and q.steps["row_to"][j][0] == 300
    assert q.steps["overspeed"].sum() == 1
    starts, _ = C.run_limits(seg)
    assert len(set(starts[seg["slot"] == 0].tolist())) == 2 and len(set(starts[seg["slot"] == 1].tolist())) == 1
    assert not ((seg["slot"] == 0) & (seg["row0"] < 300) & (seg["row1"] >= 300)).any()   # nothing joined across the jump


# ---------------------------------------------------------------- event-centred contexts

def test_centred_window_non_integer_t_star_and_edge_interpolation():
    c = make_cache(*crossing())
    q, seg = build(c)
    ts = float(c.video_time_s[225]) + 0.0123                             # not a sample instant, not an integer second
    rec = record(c, seg, ts)
    q, seg, tr, cx = contexts(c, rec)
    assert cx["nominal_start_s"] == ts - 2.5 and cx["nominal_end_s"] == ts + 2.5 and not cx["window_clipped_by_period"]
    assert cx["complete"] and cx["player_a"]["unavailable"] == [] and cx["player_a"]["incoming_s"] == pytest.approx(2.5)
    for side in ("a", "b"):
        v = tr[side]
        assert v["t"][0] == ts - 2.5 and v["t"][-1] == ts + 2.5          # exactly the window, both edges interpolated
        assert v["edge"][0] and v["edge"][-1] and not v["edge"][1:-1].any()
        assert (v["row"][v["edge"]] == -1).all()
        slot = 0 if side == "a" else 1
        rows = np.flatnonzero(c.state[:, slot] == STATE_AVAILABLE)
        # the edge point lies on the straight line between the two neighbouring update instants of the RAW cache
        u = rows[np.r_[True, (c.x[rows[1:], slot] != c.x[rows[:-1], slot])]]
        k = np.searchsorted(c.video_time_s[u], ts - 2.5)
        f = (ts - 2.5 - c.video_time_s[u[k - 1]]) / (c.video_time_s[u[k]] - c.video_time_s[u[k - 1]])
        assert v["x"][0] == pytest.approx(c.x[u[k - 1], slot] + f * (c.x[u[k], slot] - c.x[u[k - 1], slot]), abs=1e-9)
    r = audit(c, q, rec, tr, cx)
    assert r["mathematically_valid"] and r["tracking_support_valid"] and r["context_complete"], r["reasons"]
    assert r["window_start_rel_s"] == -2.5 and r["window_end_rel_s"] == 2.5


def test_partial_coverage_gap_and_jump_are_never_bridged():
    xa, ya, xb, yb = crossing()
    xb[170:188] = np.nan                                                 # B unobserved 0.63 s ending ~1.2 s before t*
    xa[270:] += 3.0                                                      # A jumps ~1.5 s after t*
    c = make_cache(xa, ya, xb, yb)
    q, seg = build(c)
    ts = float(c.video_time_s[224])
    rec = record(c, seg, ts)
    q, seg, tr, cx = contexts(c, rec)
    b, a = cx["player_b"], cx["player_a"]
    assert not b["complete_before"] and b["cut_before"] == "gap_period_or_time" and b["complete_after"]
    assert a["complete_before"] and not a["complete_after"] and a["cut_after"] == "implausible_jump"
    assert b["unavailable"] == [[ts - 2.5, b["covered_start_s"], "gap_period_or_time"]]
    assert tr["b"]["t"][0] == pytest.approx(c.video_time_s[188]) and not tr["b"]["edge"][0]     # starts at a real sample, no bridge
    assert tr["a"]["t"][-1] < c.video_time_s[270] and not tr["a"]["edge"][-1]
    assert tr["sync"][0, 0] == tr["b"]["t"][0] and tr["sync"][-1, 0] == tr["a"]["t"][-1]          # shared support only
    assert not cx["complete"]
    r = audit(c, q, rec, tr, cx)
    assert r["tracking_support_valid"], r["reasons"]                    # cuts re-derived from the cache and justified
    assert "incoming_incomplete_b" in r["reasons"] and "outgoing_incomplete_a" in r["reasons"] and not r["presentation_ready"]


def test_period_boundary_clips_the_window():
    xa, ya, xb, yb = crossing()
    per = np.where(np.arange(N) < 260, 1, 2)                             # period 1 ends ~1.2 s after t*
    c = make_cache(xa, ya, xb, yb, period=per)
    q, seg = build(c)
    ts = float(c.video_time_s[224])
    rec = record(c, seg, ts)
    q, seg, tr, cx = contexts(c, rec)
    assert cx["window_clipped_by_period"] and cx["window_end_s"] == pytest.approx(c.video_time_s[259])
    for p in ("player_a", "player_b"):
        u = cx[p]["unavailable"]
        assert u[-1][2] == "outside_period" and u[-1][1] == ts + 2.5
        assert cx[p]["covered_end_s"] <= c.video_time_s[259] + 1e-9
    r = audit(c, q, rec, tr, cx)
    assert r["tracking_support_valid"], r["reasons"]
    assert "period_boundary" in r["reasons"] and not r["outgoing_complete"]


def test_observed_estimated_preserved_and_edge_point_conservative():
    xa, ya, xb, yb = crossing()
    vis_a = np.ones(N, np.int8)
    vis_a[140:160] = 2                                                   # A estimated around t* - 2.5 s
    c = make_cache(xa, ya, xb, yb, vis_a=vis_a)
    q, seg = build(c)
    ts = float(c.video_time_s[224]) + 0.004
    rec = record(c, seg, ts)
    q, seg, tr, cx = contexts(c, rec)
    a = tr["a"]
    o = ~a["edge"]
    assert np.array_equal(a["vis"][o], c.visibility[a["row"][o], 0])    # every sample keeps its provider visibility
    assert a["vis"][0] == 2                                              # the edge point between estimated samples is estimated
    assert ((a["bits"][o] & Q.ESTIMATED) != 0).sum() == (a["vis"][o] == 2).sum()
    assert cx["player_a"]["observed_share"] == pytest.approx(float((a["vis"][o] == 1).mean()))


def test_exact_coincidence_and_nonzero_proximity():
    xa = cadence15(np.round(2 * TT, 3))
    xb = 2 * xa[224] - xa                                                # mirror image: identical position at update frame 224 only
    c = make_cache(xa, np.zeros(N), xb, np.zeros(N))
    q, seg = build(c)
    ts = float(c.video_time_s[224])
    rec = record(c, seg, ts, R=0.0)                                      # R = 0: the exact solver's single instant
    assert rec["d_min_m"] == 0.0 and rec["exact_coincidence"]
    q, seg, tr, cx = contexts(c, rec)
    r = audit(c, q, rec, tr, cx)
    assert r["mathematically_valid"], r["reasons"]
    assert r["encounter_type"] == "exact_synchronized_coincidence" and r["identical_seconds_context"] == 0
    c2 = make_cache(xa, np.zeros(N), xb, np.full(N, 0.2))
    q2, seg2 = build(c2)
    rec2 = record(c2, seg2, float(c2.video_time_s[224]), R=0.2)
    q2, seg2, tr2, cx2 = contexts(c2, rec2)
    r2 = audit(c2, q2, rec2, tr2, cx2)
    assert r2["mathematically_valid"] and r2["encounter_type"] == "closest_synchronized_approach", r2["reasons"]
    assert r2["d_min_recomputed_m"] == pytest.approx(0.2, abs=1e-6) and r2["spatial_path_crossings"] == 0


def test_cloned_tracks_are_an_artifact():
    xa, ya, xb, yb = crossing()
    xb[200:280], yb[200:280] = xa[200:280], ya[200:280]                 # provider gives B A's coordinates
    c = make_cache(xa, ya, xb, yb)
    q, seg = build(c)
    rec = record(c, seg, float(c.video_time_s[240]), R=0.25)
    rec["coincident_interval"], rec["d_min_m"], rec["exact_coincidence"] = True, 0.0, True
    q, seg, tr, cx = contexts(c, rec)
    r = audit(c, q, rec, tr, cx)
    assert r["provider_artifact_suspected"] and "identical_positions" in r["reasons"] and not r["presentation_ready"]
    assert r["encounter_type"] == "potential_tracking_ambiguity"


def test_audit_detects_tampered_contexts():
    xa, ya, xb, yb = crossing()
    xb[170:188] = np.nan
    c = make_cache(xa, ya, xb, yb)
    q, seg = build(c)
    ts = float(c.video_time_s[224]) + 0.01
    rec = record(c, seg, ts)
    q, seg, tr, cx = contexts(c, rec)
    assert audit(c, q, rec, tr, cx)["tracking_support_valid"]
    bad = {k: v.copy() for k, v in tr["a"].items()}
    bad["x"][0] += 0.05                                                  # moved window-edge point
    assert "edge_point_invalid" in audit(c, q, rec, dict(tr, a=bad), cx)["reasons"]
    cut = {k: v[3:] for k, v in tr["a"].items()}                         # dropped valid support at the start
    cx2 = json.loads(json.dumps(cx))
    cx2["player_a"].update(complete_before=False, cut_before="implausible_jump")
    r = audit(c, q, rec, dict(tr, a=cut), cx2)
    assert "cut_not_justified" in r["reasons"]
    cx3 = json.loads(json.dumps(cx))
    cx3["nominal_start_s"] += 0.4                                        # an integer-second-style window
    assert "window_not_centred" in audit(c, q, rec, tr, cx3)["reasons"]
    cx4 = json.loads(json.dumps(cx))
    cx4["player_b"]["cut_before"] = "implausible_jump"                   # wrong reason for a gap
    assert "cut_reason_mismatch" in audit(c, q, rec, tr, cx4)["reasons"]


# ---------------------------------------------------------------- real development data (written outputs)

def _need(p: Path) -> None:
    if not p.exists():
        pytest.skip(f"{p} not written")


def test_v3_dataset_provenance_and_reference_mapping():
    d = approved_output_root(ROOT) / C.SCHEMA
    _need(d / "development-index" / "manifest.json")
    man = json.loads((d / "development-index" / "manifest.json").read_text(encoding="utf-8"))
    assert man["matches"] == C.development(ROOT)
    with open(d / "development-index" / "event_mapping.csv", encoding="utf-8") as f:
        mp = list(csv.DictReader(f))
    v2 = json.loads((approved_output_root(ROOT) / C.SOURCE / "development-index" / "summary.json").read_text(encoding="utf-8"))
    assert len(mp) == v2["episodes"] and all(r["v2_line"] == r["v3_line"] for r in mp)
    for mid in ("3812", "10517"):
        assert C.verify_existing(d / mid) is None
        m = json.loads((d / mid / "root.manifest.json").read_text(encoding="utf-8"))
        assert m["code_sha256"] == C._code_sha256()                     # written by this code
        ev = approved_output_root(ROOT) / C.SOURCE / mid / "root.events-0.25m.jsonl"
        assert hashlib.sha256(ev.read_bytes()).hexdigest() == m["source"]["events_sha256"]
        ids = [json.loads(x)["event_id"] for x in ev.read_text(encoding="utf-8").splitlines()]
        assert [json.loads(x)["event_id"] for x in (d / mid / "root.contexts.jsonl").read_text(encoding="utf-8").splitlines()] == ids
        assert P.verify_existing(approved_output_root(ROOT) / C.SOURCE / mid, P.ProximityConfig()) is None   # v2 untouched
    old = approved_output_root(ROOT) / A.SCHEMA
    om = json.loads((old / "manifest.json").read_text(encoding="utf-8"))
    assert all(hashlib.sha256((old / n).read_bytes()).hexdigest() == h for n, h in om["outputs_sha256"].items())   # v2 audit untouched


def test_v3_audit_written_and_consistent():
    d = approved_output_root(ROOT) / V.SCHEMA
    _need(d / "summary.json")
    s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    ds = json.loads((d / "dataset_checks.json").read_text(encoding="utf-8"))
    assert s["mathematically_valid"] == s["tracking_support_valid"] == s["encounters"]
    assert ds["all_match_checks_pass"] and ds["mapping_keys_equal_events"] and ds["mapping_lines_identity"]
    assert ds["presentation_ready_not_in_v2_high_quality_explained_by_v2_context"]   # v2's subset needs its own v2 context complete
    with open(d / "event_validation.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert sum(r["presentation_ready"] == "True" for r in rows) == s["presentation_ready"]
    for r in rows:
        if r["presentation_ready"] == "True":                            # complete, observed, no artifact, centred window
            assert r["reasons"] == "" and float(r["window_start_rel_s"]) == -2.5 and float(r["window_end_rel_s"]) == 2.5
            assert r["pair_quality"] == "observed/observed"
            assert float(r["a_pre_s"]) == pytest.approx(2.5) and float(r["b_post_s"]) == pytest.approx(2.5)


@pytest.mark.parametrize("eid,expect", [("7d69f374b095e733", ("identical_positions", "period_boundary", "incoming_incomplete_b")),
                                        ("548ad35e88e51867", ("incoming_incomplete_a", "incoming_incomplete_b")),
                                        ("22092c0d35089442", ("estimated_at_t_star",))])
def test_reported_cases_on_v3(eid, expect):
    d = approved_output_root(ROOT) / V.SCHEMA / "event_validation.csv"
    _need(d)
    with open(d, encoding="utf-8") as f:
        r = next(x for x in csv.DictReader(f) if x["event_id"] == eid)
    assert r["mathematically_valid"] == r["tracking_support_valid"] == "True" and r["presentation_ready"] == "False"
    assert all(e in r["reasons"].split(";") for e in expect), r["reasons"]
    assert float(r["window_start_rel_s"]) == -2.5


def test_v3_refuses_validation_and_test_matches():
    split = json.loads((Path("data/processed/m11-encounters-v1/splits.json")).read_text(encoding="utf-8"))
    for mid in (str(split["splits"]["validation"][0]), str(split["splits"]["test"][0])):
        with pytest.raises(Exception, match="development|never accessed"):
            C.run_match(ROOT, mid)
        with pytest.raises(RuntimeError):
            V.audit_match(ROOT, mid)
        with pytest.raises(RuntimeError):
            S1.verify_match(ROOT, mid, False)
        assert not (approved_output_root(ROOT) / C.SCHEMA / mid).exists()
