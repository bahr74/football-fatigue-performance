"""M11 Stage 3: supervised MATCHED / SWAPPED samples from the Stage 2 identity-free candidates (``m11-stage3-samples-v1``).

Design: ``docs/research/M11_STAGE3_SUPERVISED_DATASET_DESIGN_v1.md``; approved population: every Stage 2 ACCEPTED candidate of the
32 train/CV matches of the frozen split ``m11-stage2-split-v1``. Stage 2 decisions are read, never changed. Held-out samples are not
built here: the builder refuses held-out matches (their Stage 2 files are only hash-checked), and the model-input adapter refuses
them too.

**Task.** Two incoming tracklets (before a hidden gap) and two anonymous outgoing tracklets (after it). ``y = 0`` MATCHED
(in0 -> out0, in1 -> out1), ``y = 1`` SWAPPED (in0 -> out1, in1 -> out0). The label comes only from the Stage 2 reference chains:
incoming and outgoing tracklet of a player are cut from that player's one continuous chain. It is re-verified against the raw
positions of the two provider slots. Crossing class, F1-F4 states, distances and motion never enter the label.

**Temporal grid** (one convention, ``DT`` = 1/15 s, the native update rate; ``T`` = 30 points = 2.0 s). For gap centre ``tc`` and gap
length ``g``: g0 = tc - g/2, g1 = tc + g/2; the hidden gap is the closed interval [g0, g1].

- incoming grid: g0 + k DT for k = -30 .. -1 (``in_dt`` = -2.0 .. -DT; the last point is nearest the gap);
- outgoing grid: g1 + k DT for k = 1 .. 30 (``out_dt`` = DT .. 2.0; the first point is nearest the gap).

Positions are linear interpolation of the chain's native vertices (the 15 Hz update instants; the Stage 2 geometry), using **only
vertices strictly before g0** for the incoming grid and **only vertices strictly after g1** for the outgoing grid. A grid point
outside the span of those vertices is masked (no extrapolation, no bridging to another chain). No hidden-gap vertex can therefore
influence any input value (checked by a perturbation test on every match). Masked coordinates are 0.

**Frame** (per sample, incoming tracklets only): origin = midpoint of the two last valid incoming points; x axis = summed incoming
velocity (least squares over the last ``FIT_S`` of each tracklet). Fallbacks, all order-invariant and independent of outgoing
tracks, identities, crossing or gap positions: mode 1 if |v0 + v1| < ``V_MIN`` but |v0 - v1| >= ``V_MIN`` (opposing motion):
axis = +-(v0 - v1), sign fixed so the axis points to +x of the pitch (then +y); mode 2 otherwise (both nearly stationary): the
pitch axes. Absolute pitch coordinates are not an input.

**Order.** Incoming and outgoing order are independent bits of ``sha256(ORDER_SEED | candidate_id | table)``; nothing else.

**Tables** (separate files; never pooled, never counted as independent data): ``g0p80`` primary, ``g0p40`` and ``g1p60``
sensitivity, ``g0p80_shift`` gap centre shifted by a seeded offset in -0.3 .. +0.3 s. For shifted gaps the full Stage 2 label-quality
procedure is re-run at the shifted centre; only shifts that are ACCEPTED again are built, all others are reported with reasons.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from football_fatigue.encounters import quality as Q

from . import candidates as C
from . import core as W
from . import local as L

SCHEMA = "m11-stage3-supervised-samples-v1"
STAGE2 = "m11-stage2-candidates-v1"
SPLIT = "m11-stage2-split-v1"
ORDER_SEED = "m11-stage3-order-v1"
SHIFT_SEED = "m11-stage3-shift-v1"
DT = 1.0 / 15.0
T = 30
CONTEXT_S = T * DT
FIT_S = 0.5
V_MIN = 0.3
MIN_POINTS = 3
MATCHED, SWAPPED = 0, 1
SHIFTS_S = (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3)
TABLES = {"g0p80": (0.8, False), "g0p40": (0.4, False), "g1p60": (1.6, False), "g0p80_shift": (0.8, True)}
PRIMARY = "g0p80"
INPUT_FIELDS = ("in_pos", "in_mask", "in_dt", "out_pos", "out_mask", "out_dt", "gap_s")
ARRAY_FIELDS = INPUT_FIELDS + ("y",)
IN_REL = np.arange(-T, 0) * DT                    # incoming grid relative to g0
OUT_REL = np.arange(1, T + 1) * DT                # outgoing grid relative to g1
FORBIDDEN_INPUT_WORDS = ("slot", "player", "jersey", "team", "chain", "status", "estimated", "visible", "crossing", "cohort",
                         "candidate", "match", "period", "split", "overlap", "independent", "order", "label")
PERTURB_PER_TABLE = 25


# ---------------------------------------------------------------- label conventions

def recovery_y_to_stage3(y_recovery: int | np.ndarray) -> int | np.ndarray:
    """``m11_recovery`` uses y = 1 for H0 (b0 -> a0, b1 -> a1, matched). Stage 3 uses 0 = MATCHED, 1 = SWAPPED."""
    return 1 - y_recovery


def stage3_y_to_recovery(y: int | np.ndarray) -> int | np.ndarray:
    return 1 - y


# ---------------------------------------------------------------- one tracklet, one frame, one sample (pure functions)

def chain_vertices(seg, ch, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Native vertices (times, x, y, cache rows) of chain ``k``: the 15 Hz update instants joined by valid segments."""
    a, b = int(ch["first"][k]), int(ch["last"][k])
    return (np.r_[seg["t0"][a:b + 1], seg["t1"][b]], np.r_[seg["x0"][a:b + 1], seg["x1"][b]],
            np.r_[seg["y0"][a:b + 1], seg["y1"][b]], np.r_[seg["row0"][a:b + 1], seg["row1"][b]])


def tracklet(tv: np.ndarray, xv: np.ndarray, yv: np.ndarray, grid: np.ndarray, side: str, g0: float, g1: float
             ) -> tuple[np.ndarray, np.ndarray]:
    """(T, 2) positions and (T,) validity on absolute grid times. Only vertices strictly before g0 (side 'in') or strictly after
    g1 (side 'out') are used; grid points outside their span are masked."""
    keep = tv < g0 if side == "in" else tv > g1
    pos, ok = np.zeros((len(grid), 2)), np.zeros(len(grid), bool)
    if keep.sum() >= 1:
        t, x, y = tv[keep], xv[keep], yv[keep]
        ok = (grid >= t[0]) & (grid <= t[-1])
        if side == "in":
            ok &= grid < g0
        else:
            ok &= grid > g1
        if ok.any():
            pos[ok, 0], pos[ok, 1] = np.interp(grid[ok], t, x), np.interp(grid[ok], t, y)
    return pos, ok


def _slope(t: np.ndarray, p: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Least-squares velocity over the last FIT_S of the valid points (zero with fewer than two)."""
    if ok.sum() < 2:
        return np.zeros(2)
    tt, pp = t[ok], p[ok]
    k = tt >= tt[-1] - FIT_S - 1e-9
    tt, pp = tt[k], pp[k]
    if len(tt) < 2:
        return np.zeros(2)
    A = np.c_[np.ones(len(tt)), tt - tt[-1]]
    return np.linalg.lstsq(A, pp, rcond=None)[0][1]


def sample_frame(in_pos: np.ndarray, in_mask: np.ndarray, in_t: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """(origin, rotation R, mode) from the incoming tracklets only; p_frame = R (p - origin). Order-invariant in the two tracklets."""
    last = [in_pos[i][np.flatnonzero(in_mask[i])[-1]] for i in (0, 1)]
    origin = 0.5 * (last[0] + last[1])
    v = [_slope(in_t, in_pos[i], in_mask[i]) for i in (0, 1)]
    s, d = v[0] + v[1], v[0] - v[1]
    if np.hypot(*s) >= V_MIN:
        axis, mode = s, 0
    elif np.hypot(*d) >= V_MIN:
        axis, mode = d, 1
        if axis[0] < -1e-9 or (abs(axis[0]) <= 1e-9 and axis[1] < 0):
            axis = -axis
    else:
        axis, mode = np.array([1.0, 0.0]), 2
    th = float(np.arctan2(axis[1], axis[0]))
    c, si = np.cos(th), np.sin(th)
    return origin, np.array([[c, si], [-si, c]]), mode


def order_bits(candidate_id: str, table: str) -> tuple[int, int]:
    h = int.from_bytes(hashlib.sha256(f"{ORDER_SEED}|{candidate_id}|{table}".encode()).digest()[:8], "little")
    return h & 1, (h >> 1) & 1


def shift_for(candidate_id: str) -> float:
    h = int.from_bytes(hashlib.sha256(f"{SHIFT_SEED}|{candidate_id}".encode()).digest()[:8], "little")
    return float(SHIFTS_S[h % len(SHIFTS_S)])


def build_one(VA, VB, tc: float, g: float, in_order: int, out_order: int) -> dict[str, Any]:
    """One sample from the two players' chain vertices VA, VB = (t, x, y) (player A, player B). Returns the model arrays (frame
    coordinates), the pitch-frame tracklets (verification only), the player behind each tracklet (verification only) and the frame."""
    g0, g1 = tc - g / 2, tc + g / 2
    gin, gout = g0 + IN_REL, g1 + OUT_REL
    trk = {("in", p): tracklet(*V, gin, "in", g0, g1) for p, V in (("A", VA), ("B", VB))}
    trk |= {("out", p): tracklet(*V, gout, "out", g0, g1) for p, V in (("A", VA), ("B", VB))}
    who_in = ("B", "A") if in_order else ("A", "B")
    who_out = ("B", "A") if out_order else ("A", "B")
    pin = np.stack([trk[("in", p)][0] for p in who_in])
    min_ = np.stack([trk[("in", p)][1] for p in who_in])
    pout = np.stack([trk[("out", p)][0] for p in who_out])
    mout = np.stack([trk[("out", p)][1] for p in who_out])
    y = MATCHED if who_in[0] == who_out[0] else SWAPPED
    res: dict[str, Any] = dict(who_in=who_in, who_out=who_out, y=y, pitch_in=pin, pitch_out=pout, in_mask=min_, out_mask=mout,
                               grid_in=gin, grid_out=gout, g0=g0, g1=g1)
    counts = [int(m.sum()) for m in (*min_, *mout)]
    res["n_valid"] = counts
    if min(counts) < MIN_POINTS:
        res["failure"] = "too_few_valid_points"
        return res
    origin, R, mode = sample_frame(pin, min_, IN_REL)
    res.update(origin=origin, R=R, frame_mode=mode)
    res["in_pos"] = np.where(min_[..., None], (pin - origin) @ R.T, 0.0).astype(np.float32)
    res["out_pos"] = np.where(mout[..., None], (pout - origin) @ R.T, 0.0).astype(np.float32)
    return res


def model_arrays(s: dict[str, Any], g: float) -> dict[str, np.ndarray]:
    return dict(in_pos=s["in_pos"], in_mask=s["in_mask"].astype(bool), in_dt=IN_REL.astype(np.float32), out_pos=s["out_pos"],
                out_mask=s["out_mask"].astype(bool), out_dt=OUT_REL.astype(np.float32), gap_s=np.float32(g), y=np.int8(s["y"]))


# ---------------------------------------------------------------- symmetry helpers (training-time augmentation / tests)

def swap_outgoing(x: dict[str, np.ndarray], y):
    """Exchange the two outgoing tracklets: the label flips."""
    x2 = dict(x, out_pos=x["out_pos"][..., ::-1, :, :].copy(), out_mask=x["out_mask"][..., ::-1, :].copy())
    return x2, 1 - y


def swap_incoming(x: dict[str, np.ndarray], y):
    """Exchange the two incoming tracklets: the label flips (the frame is order-invariant, so no re-normalisation)."""
    x2 = dict(x, in_pos=x["in_pos"][..., ::-1, :, :].copy(), in_mask=x["in_mask"][..., ::-1, :].copy())
    return x2, 1 - y


# ---------------------------------------------------------------- deterministic storage

def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Uncompressed .npz with fixed zip timestamps (byte-identical for identical arrays), written atomically."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for k in sorted(arrays):
            b = io.BytesIO()
            np.lib.format.write_array(b, np.ascontiguousarray(arrays[k]), allow_pickle=False)
            z.writestr(zipfile.ZipInfo(f"{k}.npy", date_time=(1980, 1, 1, 0, 0, 0)), b.getvalue())
    L.atomic_write_bytes(path, buf.getvalue())


def empty_arrays(n: int = 0) -> dict[str, np.ndarray]:
    return dict(in_pos=np.zeros((n, 2, T, 2), np.float32), in_mask=np.zeros((n, 2, T), bool), in_dt=np.zeros((n, T), np.float32),
                out_pos=np.zeros((n, 2, T, 2), np.float32), out_mask=np.zeros((n, 2, T), bool), out_dt=np.zeros((n, T), np.float32),
                gap_s=np.zeros(n, np.float32), y=np.zeros(n, np.int8))


def stack(rows: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not rows:
        return empty_arrays()
    return {k: np.stack([r[k] for r in rows]) for k in ARRAY_FIELDS}


def check_arrays(A: dict[str, np.ndarray]) -> dict[str, Any]:
    n = len(A["y"])
    shapes = dict(in_pos=(n, 2, T, 2), in_mask=(n, 2, T), in_dt=(n, T), out_pos=(n, 2, T, 2), out_mask=(n, 2, T), out_dt=(n, T),
                  gap_s=(n,), y=(n,))
    dtypes = dict(in_pos=np.float32, in_mask=np.bool_, in_dt=np.float32, out_pos=np.float32, out_mask=np.bool_, out_dt=np.float32,
                  gap_s=np.float32, y=np.int8)
    r = dict(n=n, fields_ok=set(A) == set(ARRAY_FIELDS),
             shapes_ok=all(A[k].shape == shapes[k] for k in shapes), dtypes_ok=all(A[k].dtype == dtypes[k] for k in dtypes),
             finite_ok=all(bool(np.isfinite(A[k]).all()) for k in ("in_pos", "out_pos", "in_dt", "out_dt", "gap_s")),
             masked_zero_ok=bool((A["in_pos"][~A["in_mask"]] == 0).all() and (A["out_pos"][~A["out_mask"]] == 0).all()),
             labels_ok=bool(np.isin(A["y"], (MATCHED, SWAPPED)).all()),
             dt_ok=bool(n == 0 or (np.allclose(A["in_dt"], IN_REL, atol=1e-6) and np.allclose(A["out_dt"], OUT_REL, atol=1e-6))),
             dt_sign_ok=bool((A["in_dt"] < 0).all() and (A["out_dt"] > 0).all()),
             min_points_ok=bool(n == 0 or (A["in_mask"].sum(-1).min() >= MIN_POINTS and A["out_mask"].sum(-1).min() >= MIN_POINTS)))
    r["all_pass"] = all(v for k, v in r.items() if k.endswith("_ok"))
    return r


# ---------------------------------------------------------------- per match

def _band(v: float, edges, labels) -> str:
    if not np.isfinite(v):
        return "nan"
    return labels[int(np.searchsorted(edges, v, "right")) - 1] if v >= edges[0] else labels[0]


D_BANDS = ([0.0, 0.05, 0.25, 0.5], ["<0.05", "0.05-0.25", "0.25-0.5"])
SPEED_BANDS = ([0.0, 1.0, 2.0, 4.0], ["<1", "1-2", "2-4", ">=4"])


def verify_label(s: dict[str, Any], raw: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]) -> str:
    """Independent label check from the raw cache: assign every tracklet to the provider slot (A or B) whose raw vertex positions
    reproduce it; the label implied by those assignments must equal ``y``. Returns 'ok', 'mismatch' or 'ambiguous'."""
    who = {}
    for side, P, M, grid in (("in", s["pitch_in"], s["in_mask"], s["grid_in"]), ("out", s["pitch_out"], s["out_mask"], s["grid_out"])):
        for i in (0, 1):
            err = {}
            for p, (tr, xr, yr) in raw.items():
                pr, okr = tracklet(tr, xr, yr, grid, side, s["g0"], s["g1"])
                m = M[i]
                err[p] = float(np.abs(pr[m] - P[i][m]).max()) if (okr[m]).all() else np.inf
            good = [p for p, e in err.items() if e <= 1e-9]
            if len(good) != 1:
                return "ambiguous" if len(good) > 1 else "mismatch"
            who[(side, i)] = good[0]
    y_raw = MATCHED if who[("in", 0)] == who[("out", 0)] else SWAPPED
    ok = y_raw == s["y"] and who[("in", 0)] != who[("in", 1)] and who[("out", 0)] != who[("out", 1)]
    return "ok" if ok else "mismatch"


def gap_perturbation_ok(VA, VB, s: dict[str, Any], tc: float, g: float, io_: tuple[int, int]) -> bool:
    """Rebuild with every vertex inside the closed hidden gap moved by +1000 m: all model arrays must be identical."""
    def pert(V):
        t, x, y = V
        inside = (t >= tc - g / 2) & (t <= tc + g / 2)
        return t, np.where(inside, x + 1000.0, x), np.where(inside, y + 1000.0, y)
    s2 = build_one(pert(VA), pert(VB), tc, g, *io_)
    if "failure" in s2:
        return False
    return all(np.array_equal(s[k], s2[k]) for k in ("in_pos", "out_pos", "in_mask", "out_mask"))


META_FROM_CAND = ("match_id", "period", "split", "candidate_id", "t_star", "d_min", "strict_pass", "crossing_class", "team_relation",
                  "chord_speed_min_m_s", "rel_speed_m_s", "n_cross_local", "n_cross_window", "time_within_radius_s")


def build_match(c, mask, dup, seg, ch, cand: pd.DataFrame, ccfg: C.CandidateConfig, perturb_seed: int = 0
                ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """All tables of one match: arrays per table, metadata per table (row-aligned), failures, the LOWER_CONFIDENCE_IDENTITY audit
    table, and the per-match summary (counts and checks)."""
    cx, cy, _ = W.coefficients(seg, "linear")
    est = (mask & Q.ESTIMATED) != 0
    cohort_cols = [k for k in cand.columns if k.startswith("cohort_")]
    periods = {}
    for p in sorted(int(v) for v in np.unique(cand["period"])) if len(cand) else []:
        rows_p = np.flatnonzero(c.canonical_period == p)
        tp = c.video_time_s[rows_p]
        periods[p] = (rows_p, tp, float(tp[0]), float(tp[-1]))
    arrays: dict[str, list] = {t: [] for t in TABLES}
    metas: dict[str, list] = {t: [] for t in TABLES}
    fails: list[dict[str, Any]] = []
    checks: dict[str, Counter] = {t: Counter() for t in TABLES}
    rng = np.random.default_rng(perturb_seed)
    cache: dict[int, tuple] = {}

    def verts(k: int):
        if k not in cache:
            cache[k] = chain_vertices(seg, ch, k)
        return cache[k]

    for r in cand.itertuples(index=False):
        r = r._asdict()
        ka, kb = int(r["chain_a"]), int(r["chain_b"])
        if not (ch["slot"][ka] == r["slot_a"] and ch["slot"][kb] == r["slot_b"] and ch["period"][ka] == r["period"] == ch["period"][kb]):
            raise RuntimeError(f"{r['candidate_id']}: Stage 2 chain ids do not match slots / period")
        ta, xa, ya, rowa = verts(ka)
        tb, xb, yb, rowb = verts(kb)
        VA, VB = (ta, xa, ya), (tb, xb, yb)
        raw = {"A": (c.video_time_s[rowa], c.x[rowa, r["slot_a"]], c.y[rowa, r["slot_a"]]),
               "B": (c.video_time_s[rowb], c.x[rowb, r["slot_b"]], c.y[rowb, r["slot_b"]])}
        for table, (g, shifted) in TABLES.items():
            tag = L.key(g)
            if r[f"g{tag}_outcome"] != C.ACCEPTED:
                continue
            shift, q = 0.0, None
            if shifted:
                shift = shift_for(r["candidate_id"])
                rows_p, tp, pstart, pend = periods[r["period"]]
                q = C.label_quality(c, mask, dup, est, seg, ch, cx, cy, dict(r, t_star=r["t_star"] + shift), rows_p, tp, pstart, pend, g, ccfg)
                if q["outcome"] != C.ACCEPTED:
                    why = json.loads(q["reasons"]) or json.loads(q["lower_reasons"])
                    fails.append(dict(table=table, candidate_id=r["candidate_id"], match_id=r["match_id"], shift_s=shift,
                                      stage="shift_revalidation", outcome=q["outcome"], reason="|".join(why)))
                    continue
            tc = r["t_star"] + shift
            io_ = order_bits(r["candidate_id"], table)
            s = build_one(VA, VB, tc, g, *io_)
            if "failure" in s:
                fails.append(dict(table=table, candidate_id=r["candidate_id"], match_id=r["match_id"], shift_s=shift, stage="construction",
                                  outcome="", reason=s["failure"]))
                continue
            lab = verify_label(s, raw)
            checks[table][f"label_{lab}"] += 1
            gin_ok = bool((s["grid_in"][s["in_mask"].any(0)] < s["g0"]).all() and (s["grid_out"][s["out_mask"].any(0)] > s["g1"]).all())
            checks[table]["gap_boundary_ok" if gin_ok else "gap_boundary_bad"] += 1
            inv = max(float(np.abs((s["in_pos"][i][s["in_mask"][i]].astype(np.float64) @ s["R"]) + s["origin"]
                                   - s["pitch_in"][i][s["in_mask"][i]]).max()) for i in (0, 1))
            checks[table]["inverse_transform_ok" if inv < 1e-3 else "inverse_transform_bad"] += 1
            if lab != "ok" or not gin_ok or inv >= 1e-3:
                fails.append(dict(table=table, candidate_id=r["candidate_id"], match_id=r["match_id"], shift_s=shift, stage="verification",
                                  outcome="", reason=f"label_{lab}" if lab != "ok" else ("gap_boundary" if not gin_ok else "inverse_transform")))
                continue
            if rng.random() < 0.05 or checks[table]["perturbation_tested"] < 3:
                if checks[table]["perturbation_tested"] < PERTURB_PER_TABLE:
                    checks[table]["perturbation_tested"] += 1
                    checks[table]["perturbation_identical" if gap_perturbation_ok(VA, VB, s, tc, g, io_) else "perturbation_changed"] += 1
            arrays[table].append(model_arrays(s, g))
            ia, ib = ("a", "b") if s["who_in"][0] == "A" else ("b", "a")
            gq = q if shifted else {k[len(f"g{tag}_"):]: v for k, v in r.items() if k.startswith(f"g{tag}_")}
            diag = json.loads(gq["diagnostics"])
            m = {k: r[k] for k in META_FROM_CAND}
            m.update(table=table, row=len(metas[table]), gap_s=g, shift_s=shift, gap_centre_s=tc, y=s["y"], in_order=io_[0], out_order=io_[1],
                     frame_mode=s["frame_mode"], overlap_group=int(r[f"g{tag}_overlap_group"]), independent=bool(r[f"g{tag}_independent"]),
                     d_band=_band(r["d_min"], *D_BANDS), speed_band=_band(r["chord_speed_min_m_s"], *SPEED_BANDS),
                     near_coincidence="near_coincidence" in diag, jump_transfer_risk="jump_transfer_risk" in diag,
                     jump_transfer_risk_sides=gq["jump_transfer_risk"], boundary_jumps_in_window=int(gq["boundary_jumps_in_window"]),
                     est_frac_used_in0=gq[f"est_frac_used_{ia}"], est_frac_used_in1=gq[f"est_frac_used_{ib}"],
                     est_frac_in_gap_in0=gq[f"est_frac_in_gap_{ia}"], est_frac_in_gap_in1=gq[f"est_frac_in_gap_{ib}"],
                     n_valid_in0=s["n_valid"][0], n_valid_in1=s["n_valid"][1], n_valid_out0=s["n_valid"][2], n_valid_out1=s["n_valid"][3],
                     **{k: r[k] for k in cohort_cols})
            e0, e1 = m["est_frac_in_gap_in0"], m["est_frac_in_gap_in1"]
            m["status_in_gap"] = ("both_fully_estimated" if e0 == 1 and e1 == 1 else "both_fully_visible" if e0 == 0 and e1 == 0
                                  else "mixed" if np.isfinite(e0) and np.isfinite(e1) else "unknown")
            metas[table].append(m)
    lower = cand[(cand[f"g{L.key(ccfg.primary_gap_s)}_outcome"] == C.LOWER) | (cand["g0p40_outcome"] == C.LOWER) | (cand["g1p60_outcome"] == C.LOWER)]
    lower = lower[[k for k in META_FROM_CAND] + [k for k in lower.columns if k.endswith(("_outcome", "_lower_reasons", "_diagnostics"))]]
    A = {t: stack(arrays[t]) for t in TABLES}
    M = {t: pd.DataFrame(metas[t]) for t in TABLES}
    F = pd.DataFrame(fails, columns=["table", "candidate_id", "match_id", "shift_s", "stage", "outcome", "reason"])
    summ: dict[str, Any] = {}
    for t, (g, shifted) in TABLES.items():
        acc = int((cand[f"g{L.key(g)}_outcome"] == C.ACCEPTED).sum())
        ft = F[F["table"] == t]
        summ[t] = dict(accepted_candidates=acc, built=int(len(A[t]["y"])), failed=int(len(ft)),
                       failure_reasons={str(k): int(v) for k, v in ft["reason"].value_counts().items()},
                       swapped=int(A[t]["y"].sum()), arrays=check_arrays(A[t]), checks=dict(checks[t]),
                       candidates_hash=hashlib.sha256("|".join(M[t]["candidate_id"]).encode()).hexdigest() if len(M[t]) else "")
        if len(M[t]) and not (M[t]["row"].to_numpy() == np.arange(len(M[t]))).all():
            raise RuntimeError("metadata rows are not aligned with the arrays")
    summ["lower_confidence_identity_rows"] = int(len(lower))
    return A, M, F, lower, summ


# ---------------------------------------------------------------- model-input adapter (the only reader for model development)

class HeldOutRefused(PermissionError):
    pass


def load_model_inputs(out: Path, table: str, matches: list[str] | None = None, proc: Path | None = None,
                      expected_split_sha256: str | None = None) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """(inputs, y, groups) for model development. ``inputs`` has exactly ``INPUT_FIELDS``; ``groups`` are match ids for grouped
    folds (never an input). Refuses any held-out match of the frozen split and verifies every file against its ``match_done.json``.
    No metadata column is returned."""
    from .split import load_split

    if table not in TABLES:
        raise ValueError(table)
    proc = proc if proc is not None else out.parent
    sp = load_split(proc, SPLIT)
    if split_sha256(proc) != (expected_split_sha256 or SPLIT_SHA256):
        raise HeldOutRefused("the frozen split differs from the approved split")
    held = set(sp["heldout"])
    avail = sorted((p.name for p in (out / "matches").iterdir() if p.is_dir() and not p.name.endswith(".partial")), key=int)
    matches = avail if matches is None else [str(m) for m in matches]
    bad = sorted(set(matches) & held) + sorted(m for m in avail if m in held)
    if bad:
        raise HeldOutRefused(f"held-out matches are not readable during model development: {bad}")
    xs, ys, gs = [], [], []
    for m in matches:
        if m not in sp["train_cv"]:
            raise HeldOutRefused(f"{m} is not a train/CV match of the frozen split")
        d = out / "matches" / m
        L.verify_match_dir(d)
        with np.load(d / f"samples_{table}.npz", allow_pickle=False) as z:
            xs.append({k: z[k] for k in INPUT_FIELDS})
            ys.append(z["y"])
        gs.append(np.full(len(ys[-1]), int(m), np.int64))
    if not xs:
        return {k: v for k, v in empty_arrays().items() if k in INPUT_FIELDS}, np.zeros(0, np.int8), np.zeros(0, np.int64)
    return {k: np.concatenate([x[k] for x in xs]) for k in INPUT_FIELDS}, np.concatenate(ys), np.concatenate(gs)


# ---------------------------------------------------------------- CLI

def _stats(M: pd.DataFrame) -> dict[str, Any]:
    if not len(M):
        return dict(n=0)
    vc = lambda s: {str(k): int(v) for k, v in s.value_counts().sort_index().items()}  # noqa: E731
    grp = M.groupby(["match_id", "overlap_group"]).size()
    return dict(n=int(len(M)), swapped=int(M["y"].sum()), swapped_share=float(M["y"].mean()), independent=int(M["independent"].sum()),
                independent_swapped_share=float(M.loc[M["independent"], "y"].mean()) if M["independent"].any() else None,
                overlap_groups=int(len(grp)), overlap_group_size=vc(grp.clip(upper=10).rename("size")),
                largest_overlap_group=int(grp.max()), per_match=vc(M["match_id"]), crossing_class=vc(M["crossing_class"]),
                d_band=vc(M["d_band"]), speed_band=vc(M["speed_band"]), team_relation=vc(M["team_relation"]),
                near_coincidence=int(M["near_coincidence"].sum()), jump_transfer_risk=int(M["jump_transfer_risk"].sum()),
                status_in_gap=vc(M["status_in_gap"]), frame_mode=vc(M["frame_mode"]), shift_s=vc(M["shift_s"]),
                swapped_share_by_status={k: float(v) for k, v in M.groupby("status_in_gap")["y"].mean().items()},
                swapped_share_by_crossing={k: float(v) for k, v in M.groupby("crossing_class")["y"].mean().items()},
                masked_points={k: int((T - M[k]).sum()) for k in ("n_valid_in0", "n_valid_in1", "n_valid_out0", "n_valid_out1")})


def code_sha256() -> dict[str, str]:
    base = Path(__file__).parents[1]
    files = [base / "window_intersections" / n for n in ("samples.py", "candidates.py", "local.py", "core.py", "split.py")]
    files += [base / "encounters" / "quality.py", base / "matching" / "benchmark.py", base / "intersections" / "exact.py"]
    return {str(p.relative_to(base)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def main(argv: list[str] | None = None) -> int:
    from .__main__ import PROC, RAW, dev_matches, dumps, finish_manifest
    from .split import load_split
    from football_fatigue.intersections.catalogue import load_steps
    from football_fatigue.matching import validate
    from football_fatigue.matching.benchmark import load_source
    import psutil

    ap = argparse.ArgumentParser(description="M11 Stage 3: supervised MATCHED/SWAPPED samples from Stage 2 (train/CV matches only)")
    ap.add_argument("--matches", nargs="+", required=True, help="train/CV match ids, or 'train_cv' for all 32")
    ap.add_argument("--out", default="m11-stage3-samples-v1")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--simulate-crash-after", type=int, default=None)
    a = ap.parse_args(argv)
    sp = load_split(PROC, SPLIT)                                     # verifies the split file against its manifest
    sp["sha256"] = split_sha256(PROC)
    if sp["sha256"] != SPLIT_SHA256:
        raise SystemExit("frozen split hash differs from the approved split")
    if a.matches == ["train_cv"]:
        a.matches = sorted(sp["train_cv"], key=int)
    held = sorted(set(a.matches) & set(sp["heldout"]), key=int)
    if held:
        raise SystemExit(f"held-out matches are not built in Stage 3 (deferred until the final protocol is frozen): {held}")
    devs = dev_matches()
    if any(m not in devs or m not in sp["train_cv"] for m in a.matches):
        raise SystemExit("only train/CV development matches of the frozen split are allowed")
    s2 = PROC / STAGE2
    s2man = json.loads((s2 / "manifest.json").read_text(encoding="utf-8"))
    for m in sp["heldout"]:                                          # held-out integrity: hash only, contents never parsed
        rel = f"matches/{m}/candidates.parquet"
        if L._sha(s2 / rel) != s2man["outputs_sha256"][rel]:
            raise SystemExit(f"Stage 2 held-out file {rel} does not match its manifest")
    ccfg = C.CandidateConfig()
    if json.dumps(ccfg.as_dict(), sort_keys=True) != json.dumps(s2man["candidate_config"], sort_keys=True):
        raise SystemExit("candidate configuration differs from Stage 2")
    out = PROC / a.out
    run_cfg = dumps(dict(schema=SCHEMA, stage2=STAGE2, stage2_manifest_sha256=L._sha(s2 / "manifest.json"), split=SPLIT, split_sha256=sp["sha256"],
                         matches=a.matches, tables={k: dict(gap_s=v[0], shifted=v[1]) for k, v in TABLES.items()}, dt_s=DT, T=T, fit_s=FIT_S,
                         v_min=V_MIN, min_points=MIN_POINTS, shifts_s=SHIFTS_S, order_seed=ORDER_SEED, shift_seed=SHIFT_SEED,
                         label_convention={"0": "MATCHED", "1": "SWAPPED"}, candidate_config=ccfg.as_dict(), code_sha256=code_sha256()))
    if a.resume:
        if not out.exists() or (out / "manifest.json").exists():
            raise SystemExit(f"{out}: nothing to resume (missing, or complete and immutable)")
        if (out / "run_config.json").read_text(encoding="utf-8") != run_cfg:
            raise SystemExit("run configuration (arguments, config or code) differs from the interrupted run; refusing to mix")
    else:
        if out.exists():
            raise SystemExit(f"{out} exists; versioned outputs are never overwritten (use --resume for an interrupted run)")
        out.mkdir(parents=True)
        L.atomic_write_text(out / "run_config.json", run_cfg)
    proc = psutil.Process()
    snap0 = validate.raw_snapshot(RAW)
    T0 = time.perf_counter()

    def process(m: str, d: Path) -> dict[str, Any]:
        t = time.perf_counter()
        rel = f"matches/{m}/candidates.parquet"
        if L._sha(s2 / rel) != s2man["outputs_sha256"][rel]:
            raise RuntimeError(f"Stage 2 file {rel} does not match its manifest")
        cand = pd.read_parquet(s2 / rel)
        if set(cand["split"]) != {"train_cv"}:
            raise RuntimeError(f"{m}: Stage 2 rows are not all train_cv")
        c, mask, _e, _sm, _p = load_source(str(RAW), m)
        seg, _cnt, dup = W.build_segments(c, mask, load_steps(str(RAW), m))
        ch = W.chain_table(seg)
        t_load = time.perf_counter() - t
        t = time.perf_counter()
        A, M, F, lower, summ = build_match(c, mask, dup, seg, ch, cand, ccfg, perturb_seed=int(m))
        for tname in TABLES:
            save_npz(d / f"samples_{tname}.npz", A[tname])
            L.atomic_parquet(M[tname], d / f"meta_{tname}.parquet")
        L.atomic_parquet(F, d / "failures.parquet")
        L.atomic_parquet(lower.reset_index(drop=True), d / "lower_confidence_identity_audit.parquet")
        summ["runtime_s"] = dict(load_s=round(t_load, 1), build_s=round(time.perf_counter() - t, 1))
        summ["rss_mb_after"] = round(proc.memory_info().rss / 2**20, 1)
        print(f"{m}: " + ", ".join(f"{k} {v['built']}/{v['accepted_candidates']}" for k, v in summ.items() if k in TABLES)
              + f" | {summ['runtime_s']} rss {summ['rss_mb_after']} MB", flush=True)
        return summ

    try:
        per = L.run_matches(out, a.matches, process, a.simulate_crash_after)
    except L.SimulatedCrash as ex:
        print(f"SIMULATED CRASH: {ex}", flush=True)
        os._exit(3)
    # ---- aggregation (metadata only; arrays stay on disk)
    t = time.perf_counter()
    MM = {tn: pd.concat([pd.read_parquet(out / "matches" / m / f"meta_{tn}.parquet") for m in a.matches], ignore_index=True) for tn in TABLES}
    FF = pd.concat([pd.read_parquet(out / "matches" / m / "failures.parquet") for m in a.matches], ignore_index=True)
    summary: dict[str, Any] = {}
    for tn in TABLES:
        ft = FF[FF["table"] == tn]
        ck: Counter = Counter()
        for r in per.values():
            ck.update(r[tn]["checks"])
        summary[tn] = dict(_stats(MM[tn]), accepted_candidates=sum(r[tn]["accepted_candidates"] for r in per.values()),
                           built=sum(r[tn]["built"] for r in per.values()), failed=int(len(ft)),
                           failure_reasons={str(k): int(v) for k, v in ft["reason"].value_counts().items()},
                           failure_outcomes={str(k): int(v) for k, v in ft["outcome"].value_counts().items()}, checks=dict(ck))
    summary["lower_confidence_identity_rows"] = sum(r["lower_confidence_identity_rows"] for r in per.values())
    summary["population"] = "train_cv matches of m11-stage2-split-v1 only; held-out supervised shards deferred"
    arrays_ok = all(r[tn]["arrays"]["all_pass"] for r in per.values() for tn in TABLES)
    lab_ok = all(set(r[tn]["checks"]) <= {"label_ok", "gap_boundary_ok", "inverse_transform_ok", "perturbation_tested", "perturbation_identical"}
                 for r in per.values() for tn in TABLES)
    built_ok = all(summary[tn]["built"] + summary[tn]["failed"] == summary[tn]["accepted_candidates"] for tn in TABLES)
    snap1 = validate.raw_snapshot(RAW)
    checks = dict(arrays_all_pass=arrays_ok, verification_all_pass=lab_ok, counts_reconcile=built_ok, raw_unchanged=snap0 == snap1,
                  split_sha256=sp["sha256"], split_verified=True, heldout_built=False)
    checks["all_pass"] = arrays_ok and lab_ok and built_ok and snap0 == snap1
    L.atomic_write_text(out / "summary.json", dumps(summary))
    L.atomic_write_text(out / "checks.json", dumps(dict(checks, per_match={m: {tn: dict(arrays=r[tn]["arrays"], checks=r[tn]["checks"])
                                                                                for tn in TABLES} for m, r in per.items()})))
    mi = proc.memory_info()
    runtime = dict(per_match={m: r["runtime_s"] for m, r in per.items()}, aggregation_s=round(time.perf_counter() - t, 1),
                   total_this_invocation_s=round(time.perf_counter() - T0, 1), resumed=bool(a.resume),
                   peak_memory_mb=round(getattr(mi, "peak_wset", mi.rss) / 2**20, 1))
    L.atomic_write_text(out / "runtime.json", dumps(runtime))
    finish_manifest(out, dict(schema=SCHEMA, matches=a.matches, split=SPLIT, split_sha256=sp["sha256"], heldout_built=False,
                              stage2=STAGE2, stage2_manifest_sha256=L._sha(s2 / "manifest.json"), code_sha256=code_sha256(),
                              raw_unchanged=snap0 == snap1, raw_snapshot=snap1, created_unix=int(time.time()), argv=sys.argv[1:]))
    print(dumps(dict(checks=checks, built={tn: (summary[tn]["built"], summary[tn]["accepted_candidates"]) for tn in TABLES},
                     runtime=runtime)))
    return 0 if checks["all_pass"] else 1


def split_sha256(proc: Path) -> str:
    return hashlib.sha256((proc / SPLIT / "split.json").read_bytes()).hexdigest()


SPLIT_SHA256 = "66b2321a58f3845194757f943c2d36b92f18695322fb5490917f8a0144ff56c0"

if __name__ == "__main__":
    raise SystemExit(main())
