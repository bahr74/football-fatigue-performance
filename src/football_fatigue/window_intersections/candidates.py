"""Identity-free proximity candidates and label-quality tiers (M11 Stage 1): ``m11-identity-free-candidates``.

**Why identity-free.** The supervised task is tracklet association (MATCHED vs SWAPPED). Selecting samples by a geometric crossing
leaks the answer, because "the paths crossed" is computed from identity-continuous trajectories, and for the anonymous outgoing
tracklets it is equivalent to one of the two associations. Candidates here are selected only by the synchronized separation
d(t) = |A(t) - B(t)|. At every instant d(t) depends only on the *unordered* pair of positions {A(t), B(t)}, so exchanging the
identities of the two players after any instant leaves every candidate unchanged (tested: ``test_selection_is_invariant_to_swapping``).
The population therefore contains crossing and non-crossing encounters, and its selection carries no information about the label.

**Offline reference use (allowed, never a model input).** Identity-continuous reference trajectories are still used offline to
(a) segment each player's valid trajectory (Step 1 quality policy, as v1), (b) attach crossing metadata and F1-F4 cohort links for
stratification, and (c) decide label reliability. None of these selects the identity-free population.

**Candidates.** For every pair of players and every interval where both have a valid chain, d(t) is piecewise quadratic between
knots (both positions linear in time), so the set {t : d(t) <= R} is found exactly. Each maximal interval is an *episode*. Episodes
of one pair separated by at most ``merge_gap_s`` form one *encounter* (deduplication: weaving or repeated approaches are one
candidate). The candidate is the encounter's exact minimum: time t*, distance d_min.

**Label quality (configurable; M11 Stage 2 policy).** For each candidate and hidden-gap length g the sample window is
[t* - g/2 - context, t* + g/2 + context] and the *used* samples are each player's candidate chain inside that window. VISIBLE and
ESTIMATED samples are treated identically: tracking status never decides an outcome and is kept only as audit metadata.

- REJECTED (any of, with reasons): window outside the period; the minimum's episode truncated by coverage; no single chain covering
  the gap +- ``continuity_margin_s`` (no interpolation across missing or discontinuous regions); a duplicate-position or
  unattributed/ambiguous used sample; an identity-risk flag (label transfer, reciprocal exchange, discontinuity candidate,
  substitution window) on a used sample; an implausible-jump or frame-wide-shift flag on a used sample other than the chain's own
  first sample (a jump flag there is the step *into* the chain, which is why the chain starts there); a chord speed above
  ``implausible_speed_m_s``.
- LOWER_CONFIDENCE_IDENTITY: not rejected, but an identity-risk flag lies in the window outside the used samples (e.g. a label
  transfer just before the chain starts). The used samples are clean, yet the reference identity of the chain is in doubt.
- ACCEPTED: everything else.

Diagnostics (never change the outcome): ``near_coincidence`` (d_min below ``near_coincidence_m``); ``jump_transfer_risk`` (a jump
landing that starts the candidate chain, or the player's next chain, within ``jump_risk_time_s`` of the gap and within
``jump_risk_dist_m`` of the partner); chain-boundary jumps inside the window. Accepted candidates whose sample windows overlap and share
a player form an ``overlap_group``; a greedy time-ordered selection marks a non-overlapping ``independent`` subset.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from football_fatigue.encounters import quality as Q
from football_fatigue.encounters.trajectories import STATE_AVAILABLE
from football_fatigue.intersections.detect import UNUSABLE

from . import core as W
from . import local as L

SCHEMA = "m11-identity-free-candidates-v1"


@dataclass(frozen=True)
class CandidateConfig:
    radius_m: float = 0.50                   # candidate radius: episodes where d(t) <= radius
    strict_radius_m: float = 0.25            # primary ambiguity level (reported per candidate)
    merge_gap_s: float = 1.0                 # episodes of one pair at most this far apart -> one encounter
    crossing_window_s: float = 2.5           # crossing metadata: intersections with both arrivals within t* +- this
    local_crossing_s: float = 0.8            # local crossing: both arrivals within t* +- this (half the longest gap)
    gaps_s: tuple[float, ...] = (0.4, 0.8, 1.6)
    primary_gap_s: float = 0.8
    context_s: float = 2.0                   # observable context before / after the gap (up to)
    continuity_margin_s: float = 0.5         # one valid chain must cover the gap +- this (each player)
    implausible_speed_m_s: float = 12.0      # chord speed above this is implausible motion
    chord_half_s: float = 0.2
    shared_player_s: float = 1.0
    link_window_s: float = 2.5               # v1 events of the same pair within t* +- this are linked (nearest kept)
    near_coincidence_m: float = 0.05         # diagnostic: d_min below this (assignment barely constrained by motion)
    jump_risk_time_s: float = 2.0            # diagnostic: chain-boundary jump landing within this of the gap ...
    jump_risk_dist_m: float = 2.0            # ... and within this of the partner -> potential identity transfer

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


IDENTITY_RISK_BITS = {"label_transfer_candidate": Q.LABEL_TRANSFER_CANDIDATE, "reciprocal_exchange": Q.RECIPROCAL_EXCHANGE,
                      "discontinuity_candidate": Q.DISCONTINUITY_CANDIDATE, "substitution_window": Q.SUBSTITUTION_WINDOW}
MOTION_BITS = {"implausible_jump": Q.IMPLAUSIBLE_JUMP, "frame_wide_shift": Q.FRAME_WIDE_SHIFT}
FUNNEL = ("window_in_period", "episode_not_truncated", "continuous_reference", "no_corruption", "no_identity_risk",
          "no_jump_or_shift_flag", "chord_speed_plausible")          # every failure -> REJECTED
ACCEPTED, REJECTED, LOWER = "ACCEPTED", "REJECTED", "LOWER_CONFIDENCE_IDENTITY"
OUTCOMES = (ACCEPTED, LOWER, REJECTED)


# ---------------------------------------------------------------- episodes on one common interval

def episodes_on(A, B, R: float) -> list[dict[str, Any]]:
    """Maximal intervals with |A(t) - B(t)| <= R for two knot tracks on the same interval (exact), each with its minimum, the time
    of the minimum and the number of local minima of d inside it. Depends only on {A(t), B(t)}: symmetric in A and B."""
    TA, XA, YA = A
    TB, XB, YB = B
    T = np.union1d(TA, TB)
    rx = np.interp(T, TA, XA) - np.interp(T, TB, XB)
    ry = np.interp(T, TA, YA) - np.interp(T, TB, YB)
    if len(T) == 1:
        d = float(math.hypot(rx[0], ry[0]))
        return [dict(start=float(T[0]), end=float(T[0]), d_min=d, t_star=float(T[0]), n_local_minima=1)] if d <= R else []
    wx, wy = np.diff(rx), np.diff(ry)
    ww = wx * wx + wy * wy
    dot = rx[:-1] * wx + ry[:-1] * wy
    u1, u2 = L._inside_interval(ww, 2 * dot, rx[:-1] ** 2 + ry[:-1] ** 2 - R * R)
    ne = np.flatnonzero(u1 <= u2)
    if not len(ne):
        return []
    dT = np.diff(T)
    um = np.where(ww > 1e-18, np.clip(-dot / np.where(ww > 1e-18, ww, 1.0), 0.0, 1.0), 0.0)
    dm = np.hypot(rx[:-1] + um * wx, ry[:-1] + um * wy)
    interior = (ww > 1e-18) & (um > 1e-12) & (um < 1 - 1e-12)
    knot_min = np.zeros(len(T), bool)                        # knot k is a local minimum: decreasing into it, increasing out of it
    knot_min[1:-1] = (dot[:-1] + ww[:-1] < 0) & (dot[1:] > 0)
    out = []
    i = 0
    while i < len(ne):
        j = i
        while j + 1 < len(ne) and ne[j + 1] == ne[j] + 1 and u2[ne[j]] >= 1 - 1e-12 and u1[ne[j + 1]] <= 1e-12:
            j += 1
        p = ne[i:j + 1]
        k = p[int(np.argmin(dm[p]))]
        nloc = int(interior[p].sum() + knot_min[p[0] + 1:p[-1] + 1].sum())
        out.append(dict(start=float(T[p[0]] + u1[p[0]] * dT[p[0]]), end=float(T[p[-1]] + u2[p[-1]] * dT[p[-1]]), d_min=float(dm[k]),
                        t_star=float(T[k] + um[k] * dT[k]), n_local_minima=max(1, nloc)))
        i = j + 1
    return out


def merge_encounters(eps: list[dict[str, Any]], gap: float) -> list[list[dict[str, Any]]]:
    """Group a pair's episodes (any order) into encounters: consecutive episodes at most ``gap`` apart."""
    eps = sorted(eps, key=lambda e: (e["start"], e["end"]))
    out: list[list[dict]] = []
    for e in eps:
        if out and e["start"] - max(x["end"] for x in out[-1]) <= gap + 1e-12:
            out[-1].append(e)
        else:
            out.append([e])
    return out


def _chains_of(ch, slot: int, period: int) -> np.ndarray:
    return np.flatnonzero((ch["slot"] == slot) & (ch["period"] == period))


def overlaps(ch, ka: np.ndarray, kb: np.ndarray) -> list[tuple[int, int, float, float]]:
    """(chain a, chain b, lo, hi) for every pair of chains of two players that overlap in time (both sorted by start)."""
    out = []
    i = j = 0
    while i < len(ka) and j < len(kb):
        a, b = ka[i], kb[j]
        lo, hi = max(ch["t0"][a], ch["t0"][b]), min(ch["t1"][a], ch["t1"][b])
        if hi >= lo:
            out.append((int(a), int(b), float(lo), float(hi)))
        if ch["t1"][a] < ch["t1"][b]:
            i += 1
        else:
            j += 1
    return out


def pair_encounters(seg, ch, cx, cy, sa: int, sb: int, period: int, cfg: CandidateConfig) -> list[dict[str, Any]]:
    eps = []
    for a, b, lo, hi in overlaps(ch, _chains_of(ch, sa, period), _chains_of(ch, sb, period)):
        KA = L.chain_knots(seg, cx, cy, int(ch["first"][a]), int(ch["last"][a]), lo, hi, 1)
        KB = L.chain_knots(seg, cx, cy, int(ch["first"][b]), int(ch["last"][b]), lo, hi, 1)
        if KA is None or KB is None:
            continue
        # cheap rejection: bounding boxes farther apart than R
        if (KA[1].min() - KB[1].max() > cfg.radius_m or KB[1].min() - KA[1].max() > cfg.radius_m
                or KA[2].min() - KB[2].max() > cfg.radius_m or KB[2].min() - KA[2].max() > cfg.radius_m):
            continue
        for e in episodes_on(KA, KB, cfg.radius_m):
            e.update(chain_a=a, chain_b=b, cov_lo=lo, cov_hi=hi, trunc_before=e["start"] <= lo + 1e-9, trunc_after=e["end"] >= hi - 1e-9)
            eps.append(e)
    out = []
    for grp in merge_encounters(eps, cfg.merge_gap_s):
        best = min(grp, key=lambda e: (e["d_min"], e["t_star"]))
        out.append(dict(best, enc_start=min(e["start"] for e in grp), enc_end=max(e["end"] for e in grp), n_episodes=len(grp),
                        n_local_minima_total=int(sum(e["n_local_minima"] for e in grp)),
                        time_within_radius_s=float(sum(e["end"] - e["start"] for e in grp)),
                        any_episode_truncated=bool(any(e["trunc_before"] or e["trunc_after"] for e in grp))))
    return out


# ---------------------------------------------------------------- per match

def _hash(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def _rows(tp: np.ndarray, rows_p: np.ndarray, lo: float, hi: float, open_left: bool = False) -> np.ndarray:
    return rows_p[np.searchsorted(tp, lo, "right" if open_left else "left"):np.searchsorted(tp, hi, "right")]


def label_quality(c, mask, dup, est, seg, ch, cx, cy, cand: dict[str, Any], rows_p: np.ndarray, tp: np.ndarray, pstart: float, pend: float,
                  g: float, cfg: CandidateConfig) -> dict[str, Any]:
    """Outcome (ACCEPTED / LOWER_CONFIDENCE_IDENTITY / REJECTED) of one candidate for hidden gap g, with reasons, diagnostics and the
    tracking-status audit. Status (VISIBLE / ESTIMATED) is never used for the outcome."""
    ts, C, mg = cand["t_star"], cfg.context_s, cfg.continuity_margin_s
    g0, g1 = ts - g / 2, ts + g / 2
    w0, w1 = g0 - C, g1 + C
    ok: dict[str, bool] = {}
    info: dict[str, Any] = {}
    ok["window_in_period"] = w0 >= pstart - 1e-9 and w1 <= pend + 1e-9
    ok["episode_not_truncated"] = not (cand["trunc_before"] or cand["trunc_after"])
    cont, motion_ok, corrupt = True, True, False
    rr = _rows(tp, rows_p, w0, w1)
    bits = {k: False for k in list(IDENTITY_RISK_BITS) + list(MOTION_BITS)}
    near_bits = {k: False for k in IDENTITY_RISK_BITS}
    jump_risk, boundary_jumps = [], 0
    fdt = 1.0 / float(c.fps)
    partner = {"a": cand["slot_b"], "b": cand["slot_a"]}
    for tag, s, k in (("a", cand["slot_a"], cand["chain_a"]), ("b", cand["slot_b"], cand["chain_b"])):
        c0, c1 = float(ch["t0"][k]), float(ch["t1"][k])
        cont &= c0 <= g0 - mg + 1e-9 and c1 >= g1 + mg - 1e-9
        info[f"context_before_s_{tag}"] = float(np.clip(g0 - c0, 0, C))
        info[f"context_after_s_{tag}"] = float(np.clip(c1 - g1, 0, C))
        ru = _rows(tp, rows_p, max(c0, w0), min(c1, w1))                          # used samples (chain inside the window)
        rj = _rows(tp, rows_p, max(c0, w0), min(c1, w1), open_left=c0 >= w0)      # ... without the chain's own first sample
        mu, mj = mask[ru, s], mask[rj, s]
        corrupt |= bool(dup[ru, s].any()) or bool(((mu & UNUSABLE) != 0).any())
        used = np.isin(rr, ru)
        for name, bit in IDENTITY_RISK_BITS.items():
            hit = (mask[rr, s] & bit) != 0
            bits[name] |= bool(hit[used].any())
            near_bits[name] |= bool(hit[~used].any())
        for name, bit in MOTION_BITS.items():
            bits[name] |= bool(((mj & bit) != 0).any())
        vmax = L.max_chord_speed(L.chain_knots(seg, cx, cy, int(ch["first"][k]), int(ch["last"][k]), w0, w1, 1), cfg.chord_half_s)
        info[f"max_chord_speed_m_s_{tag}"] = vmax
        motion_ok &= not (np.isfinite(vmax) and vmax > cfg.implausible_speed_m_s)
        # chain-boundary jump audit: the jump landing that starts this chain, and the one that starts the player's next chain
        nxt = k + 1 if k + 1 < len(ch["t0"]) and ch["slot"][k + 1] == s and ch["period"][k + 1] == ch["period"][k] else None
        for side, kk, near_gap in (("start", k, g0 - float(ch["t0"][k])), ("next", nxt, None if nxt is None else float(ch["t0"][nxt]) - g1)):
            if kk is None:
                continue
            r0 = int(seg["row0"][ch["first"][kk]])
            if not (mask[r0, s] & Q.IMPLAUSIBLE_JUMP):
                continue
            t_land = float(ch["t0"][kk])
            boundary_jumps += int(w0 <= t_land <= w1)
            ps = partner[tag]
            dp = float(np.hypot(c.x[r0, s] - c.x[r0, ps], c.y[r0, s] - c.y[r0, ps]))
            if near_gap <= cfg.jump_risk_time_s + 1e-9 and np.isfinite(dp) and dp <= cfg.jump_risk_dist_m + 1e-9:
                jump_risk.append(f"{tag}_{side}")
        # tracking-status audit (metadata only)
        av = c.state[ru, s] == STATE_AVAILABLE
        info[f"est_frac_used_{tag}"] = float(est[ru, s][av].mean()) if av.any() else float("nan")
        rg = _rows(tp, rows_p, g0, g1)
        avg = c.state[rg, s] == STATE_AVAILABLE
        info[f"est_frac_in_gap_{tag}"] = float(est[rg, s][avg].mean()) if avg.any() else float("nan")
        for side, lo, hi in (("before", max(c0, w0), g0), ("after", g1, min(c1, w1))):
            r2 = _rows(tp, rows_p, lo, hi) if hi > lo else rows_p[:0]
            info[f"visible_{side}_s_{tag}"] = float(((c.state[r2, s] == STATE_AVAILABLE) & ~est[r2, s]).sum()) * fdt
    ok["continuous_reference"] = cont
    ok["no_corruption"] = not corrupt
    ok["no_identity_risk"] = not any(bits[k] for k in IDENTITY_RISK_BITS)
    ok["no_jump_or_shift_flag"] = not any(bits[k] for k in MOTION_BITS)
    ok["chord_speed_plausible"] = motion_ok
    reasons = [k for k in FUNNEL if not ok[k]]
    lower = sorted(f"{k}_near_sample" for k, v in near_bits.items() if v)
    outcome = REJECTED if reasons else LOWER if lower else ACCEPTED
    diag = (["near_coincidence"] if cand["d_min"] < cfg.near_coincidence_m else []) + (["jump_transfer_risk"] if jump_risk else [])
    return dict(info, outcome=outcome, reasons=json.dumps(reasons), lower_reasons=json.dumps(lower),
                flags=json.dumps(sorted([k for k, v in bits.items() if v] + lower)), diagnostics=json.dumps(diag),
                jump_transfer_risk=json.dumps(jump_risk), boundary_jumps_in_window=boundary_jumps,
                **{f"ok_{k}": v for k, v in ok.items()})


def match_candidates(c, mask, dup, seg, ch, cfg: CandidateConfig, wcfg: W.WindowConfig) -> pd.DataFrame:
    """All identity-free candidates of one match (linear geometry), with crossing metadata and label quality. Selection uses only
    d(t); crossing metadata and label quality are computed after selection and never remove a candidate."""
    cx, cy, _ = W.coefficients(seg, "linear")
    est = (mask & Q.ESTIMATED) != 0
    rows_out: list[dict[str, Any]] = []
    for p in sorted(int(v) for v in np.unique(seg["period"])):
        rows_p = np.flatnonzero(c.canonical_period == p)
        tp = c.video_time_s[rows_p]
        pstart, pend = float(tp[0]), float(tp[-1])
        slots = sorted(int(s) for s in np.unique(ch["slot"][ch["period"] == p]))
        for i, sa in enumerate(slots):
            for sb in slots[i + 1:]:
                for e in pair_encounters(seg, ch, cx, cy, sa, sb, p, cfg):
                    r: dict[str, Any] = dict(match_id=int(c.match_id), period=p, slot_a=sa, slot_b=sb, **{k: e[k] for k in (
                        "t_star", "d_min", "start", "end", "enc_start", "enc_end", "n_episodes", "n_local_minima", "n_local_minima_total",
                        "time_within_radius_s", "trunc_before", "trunc_after", "any_episode_truncated", "chain_a", "chain_b", "cov_lo", "cov_hi")})
                    r["candidate_id"] = _hash(c.match_id, p, sa, sb, round(e["t_star"] * 1e6))
                    r["t_star_rel_s"] = e["t_star"] - pstart
                    r["strict_pass"] = e["d_min"] <= cfg.strict_radius_m + 1e-12
                    w_lo, w_hi = min(e["start"], e["t_star"] - 3), max(e["end"], e["t_star"] + 3)
                    KA = L.chain_knots(seg, cx, cy, int(ch["first"][e["chain_a"]]), int(ch["last"][e["chain_a"]]), w_lo, w_hi, 1)
                    KB = L.chain_knots(seg, cx, cy, int(ch["first"][e["chain_b"]]), int(ch["last"][e["chain_b"]]), w_lo, w_hi, 1)
                    lo, hi = max(e["start"], KA[0][0], KB[0][0]), min(e["end"], KA[0][-1], KB[0][-1])
                    _, _, dur = L.pair_min(L.clip_track(KA, lo, hi), L.clip_track(KB, lo, hi), cfg.strict_radius_m)
                    r["time_within_strict_s"] = dur
                    pa, pb = L.at(KA, e["t_star"]), L.at(KB, e["t_star"])
                    r.update(x_a=pa[0], y_a=pa[1], x_b=pb[0], y_b=pb[1])
                    va, vb = L.velocity(KA, e["t_star"]), L.velocity(KB, e["t_star"])
                    r["rel_speed_m_s"] = float(np.hypot(*(va - vb)))
                    r["chord_speed_a_m_s"] = L.chord_speed_at(KA, e["t_star"], cfg.chord_half_s)
                    r["chord_speed_b_m_s"] = L.chord_speed_at(KB, e["t_star"], cfg.chord_half_s)
                    r["chord_speed_min_m_s"] = float(np.nanmin([r["chord_speed_a_m_s"], r["chord_speed_b_m_s"]])) \
                        if np.isfinite([r["chord_speed_a_m_s"], r["chord_speed_b_m_s"]]).any() else float("nan")
                    r["team_relation"] = "same_team" if c.slot_side[sa] == c.slot_side[sb] else "opposite_teams"
                    # crossing metadata (offline reference; never a selection criterion)
                    hits = W.brute_force_window(seg, cx, cy, sa, sb, p, e["t_star"] - cfg.crossing_window_s, e["t_star"] + cfg.crossing_window_s, 1, wcfg)
                    off = [max(abs(ta - e["t_star"]), abs(tb - e["t_star"])) for ta, tb in hits]
                    r["n_cross_window"] = len(hits)
                    r["n_cross_local"] = int(sum(o <= cfg.local_crossing_s + 1e-12 for o in off))
                    r["nearest_cross_offset_s"] = float(min(off)) if off else float("nan")
                    r["nearest_cross_abs_dt_s"] = float(abs(hits[int(np.argmin(off))][0] - hits[int(np.argmin(off))][1])) if off else float("nan")
                    r["crossing_class"] = "local_crossing" if r["n_cross_local"] else ("window_crossing_only" if hits else "no_crossing")
                    for g in cfg.gaps_s:
                        q = label_quality(c, mask, dup, est, seg, ch, cx, cy, r | dict(trunc_before=e["trunc_before"], trunc_after=e["trunc_after"]),
                                          rows_p, tp, pstart, pend, g, cfg)
                        tag = L.key(g)
                        for k, v in q.items():
                            r[f"g{tag}_{k}"] = v
                    rows_out.append(r)
    df = pd.DataFrame(rows_out)
    if len(df):
        df = df.sort_values(["period", "t_star", "slot_a", "slot_b"]).reset_index(drop=True)
        df["shared_player_candidates"] = _shared(df, cfg.shared_player_s)
        for g in cfg.gaps_s:
            grp, ind = dependency(df, g, cfg)
            df[f"g{L.key(g)}_overlap_group"], df[f"g{L.key(g)}_independent"] = grp, ind
    return df


def dependency(df: pd.DataFrame, g: float, cfg: CandidateConfig) -> tuple[np.ndarray, np.ndarray]:
    """Among ACCEPTED candidates of one match: overlap groups (connected components of samples whose windows overlap and that share a
    player; -1 for non-accepted) and a greedy time-ordered set of mutually non-overlapping samples. Groups never span matches, so a
    match-level split (held-out, grouped CV) never separates two dependent samples."""
    tag = L.key(g)
    span = g + 2 * cfg.context_s                                  # windows [t* - span/2, t* + span/2] overlap if |dt*| < span
    grp, ind = np.full(len(df), -1), np.zeros(len(df), bool)
    acc = np.flatnonzero(df[f"g{tag}_outcome"].to_numpy() == ACCEPTED)
    parent = {int(i): int(i) for i in acc}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for _, sub_ in df.iloc[acc].groupby("period"):
        idx = sub_.index.to_numpy()
        t, sa, sb = sub_["t_star"].to_numpy(), sub_["slot_a"].to_numpy(), sub_["slot_b"].to_numpy()
        o = np.argsort(t, kind="stable")
        for x in range(len(o)):
            for y in range(x + 1, len(o)):
                i, j = o[x], o[y]
                if t[j] - t[i] >= span - 1e-12:
                    break
                if {sa[i], sb[i]} & {sa[j], sb[j]}:
                    parent[find(int(idx[i]))] = find(int(idx[j]))
        last: dict[int, float] = {}
        for x in o:
            if all(t[x] - last.get(int(pl), -np.inf) >= span - 1e-12 for pl in (sa[x], sb[x])):
                ind[idx[x]] = True
                last[int(sa[x])] = last[int(sb[x])] = t[x]
    roots = {}
    for i in acc:
        grp[i] = roots.setdefault(find(int(i)), len(roots))
    return grp, ind


_RANK = {L.PASS: 3, L.UNDETERMINED: 2, L.FAIL: 1}
_UNRANK = {v: k for k, v in _RANK.items()}


def link_events(cands: pd.DataFrame, ev: pd.DataFrame, D: pd.DataFrame, S: pd.DataFrame, cohorts: dict[str, str], cfg: CandidateConfig,
                h_s: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach the F1-F4 states of the v1 crossing events (linear) to the candidates (metadata only; never a selection rule).

    Each event is mapped to the candidate of the same pair whose encounter [enc_start, enc_end] contains the event's local minimum
    time (arr neighbourhood, primary h), else to the candidate with the nearest t* within ``link_window_s``. A candidate's cohort
    state is PASS if any linked event passes, else UNDETERMINED if any is undetermined, else FAIL; -1 without a linked event.
    Coverage: every event whose local minimum is <= the candidate radius must lie inside a candidate encounter with d_min <= its
    local minimum (so the candidate population contains every F2 / F3 arr event)."""
    hk = L.key(h_s)
    t_loc = (ev["t_event_s"] + D[f"arr_h{hk}_t_rel_s"].fillna(0.0)).to_numpy(float)
    d_loc = D[f"arr_h{hk}_d_m"].to_numpy(float)
    cand_of = np.full(len(ev), -1)
    contained = np.zeros(len(ev), bool)
    by = {tuple(int(x) for x in k): g for k, g in cands.groupby(["period", "slot_a", "slot_b"])} if len(cands) else {}
    for i, (p, sa, sb) in enumerate(zip(ev["period"].to_numpy(int), ev["slot_a"].to_numpy(int), ev["slot_b"].to_numpy(int))):
        g = by.get((int(p), int(min(sa, sb)), int(max(sa, sb))))
        if g is None:
            continue
        inside = (g["enc_start"].to_numpy() - 1e-9 <= t_loc[i]) & (t_loc[i] <= g["enc_end"].to_numpy() + 1e-9)
        if inside.any():
            cand_of[i], contained[i] = int(g.index[np.flatnonzero(inside)[0]]), True
            continue
        dt = np.abs(g["t_star"].to_numpy() - t_loc[i])
        j = int(np.argmin(dt))
        if dt[j] <= cfg.link_window_s:
            cand_of[i] = int(g.index[j])
    out = cands.copy()
    n = len(out)
    lk = cand_of >= 0
    out["n_linked_events"] = np.bincount(cand_of[lk], minlength=n)[:n] if n else np.zeros(0, int)
    ids = ev["event_id"].to_numpy()
    nearest: dict[int, tuple[float, str]] = {}
    for i in np.flatnonzero(lk):
        dt = abs(t_loc[i] - float(out.at[cand_of[i], "t_star"]))
        if cand_of[i] not in nearest or dt < nearest[cand_of[i]][0]:
            nearest[cand_of[i]] = (dt, str(ids[i]))
    out["linked_event_id"] = [nearest[i][1] if i in nearest else None for i in range(n)]
    for lab, col in cohorts.items():
        best = np.zeros(n, int)
        st = S[col].to_numpy()
        for i in np.flatnonzero(lk):
            best[cand_of[i]] = max(best[cand_of[i]], _RANK[int(st[i])])
        out[f"cohort_{lab}"] = np.array([_UNRANK.get(int(b), -1) for b in best], dtype=np.int8)
    near = np.isfinite(d_loc) & (d_loc <= cfg.radius_m)
    ok_cov = np.zeros(len(ev), bool)
    for i in np.flatnonzero(near & contained):
        ok_cov[i] = float(out.at[cand_of[i], "d_min"]) <= d_loc[i] + 1e-9
    cov: dict[str, Any] = dict(events=int(len(ev)), events_local_min_le_radius=int(near.sum()), covered=int(ok_cov[near].sum()),
                               uncovered_event_ids=[str(x) for x in ids[near & ~ok_cov][:20]], events_linked=int(lk.sum()),
                               events_linked_by_containment=int(contained.sum()))
    cov["primary_pass_covered"] = {lab: dict(pass_events=int((S[col].to_numpy() == L.PASS).sum()),
                                             covered=int(((S[col].to_numpy() == L.PASS) & ok_cov).sum()))
                                   for lab, col in cohorts.items() if col.startswith(("F2_arr", "F3_"))}
    cov["all_covered"] = bool(ok_cov[near].all()) and all(v["pass_events"] == v["covered"] for v in cov["primary_pass_covered"].values())
    return out, cov


def _shared(df: pd.DataFrame, win: float) -> np.ndarray:
    out = np.zeros(len(df), int)
    for (p,), g in df.groupby(["period"]):
        idx = g.index.to_numpy()
        t, sa, sb = g["t_star"].to_numpy(), g["slot_a"].to_numpy(), g["slot_b"].to_numpy()
        for k, i in enumerate(idx):
            near = (np.abs(t - t[k]) <= win) & (np.arange(len(t)) != k)
            out[i] = int((near & ((sa == sa[k]) | (sb == sa[k]) | (sa == sb[k]) | (sb == sb[k]))).sum())
    return out


# ---------------------------------------------------------------- summaries

def funnel(df: pd.DataFrame, g: float) -> dict[str, Any]:
    tag = L.key(g)
    alive = np.ones(len(df), bool)
    seq, indep = {}, {}
    for k in FUNNEL:
        okk = df[f"g{tag}_ok_{k}"].to_numpy(bool)
        alive &= okk
        seq[k] = int(alive.sum())
        indep[k] = int((~okk).sum())
    vc = df[f"g{tag}_outcome"].value_counts()
    return dict(candidates=int(len(df)), surviving_after_each_reject_check=seq, failing_each_reject_check=indep,
                **{o: int(vc.get(o, 0)) for o in OUTCOMES})


def _vc(s: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in s.value_counts().sort_index().items()}


def gap_report(df: pd.DataFrame, g: float, cohorts: list[str], cfg: CandidateConfig) -> dict[str, Any]:
    """Everything the Stage 2 report needs for one gap length (all matches in ``df``)."""
    t = L.key(g)
    out_ = df[f"g{t}_outcome"]
    acc = (out_ == ACCEPTED).to_numpy()
    r: dict[str, Any] = dict(funnel=funnel(df, g))
    r["outcome_by_match"] = {str(m): _vc(x[f"g{t}_outcome"]) for m, x in df.groupby("match_id")}
    r["outcome_by_match_period"] = {f"{m}-{p}": _vc(x[f"g{t}_outcome"]) for (m, p), x in df.groupby(["match_id", "period"])}
    r["crossing_by_outcome"] = {o: _vc(df.loc[out_ == o, "crossing_class"]) for o in OUTCOMES}
    names = lambda s_: _vc(s_.map(lambda v: L.STATE_NAMES.get(int(v), "no_linked_event")))  # noqa: E731
    r["cohort_states_by_outcome"] = {o: {c: names(df.loc[out_ == o, f"cohort_{c}"]) for c in cohorts} for o in OUTCOMES}
    r["reject_reasons"] = _explode(df.loc[out_ == REJECTED, f"g{t}_reasons"])
    r["lower_confidence_reasons"] = _explode(df.loc[out_ == LOWER, f"g{t}_lower_reasons"])
    r["flags_all"] = _explode(df[f"g{t}_flags"])
    r["diagnostics_by_outcome"] = {o: _explode(df.loc[out_ == o, f"g{t}_diagnostics"]) for o in OUTCOMES}
    r["jump_transfer_risk_sides_accepted"] = _explode(df.loc[acc, f"g{t}_jump_transfer_risk"])
    r["boundary_jumps_in_window_accepted"] = int((df.loc[acc, f"g{t}_boundary_jumps_in_window"] > 0).sum())
    nc = df.loc[acc & (df["d_min"] < cfg.near_coincidence_m).to_numpy()]
    r["near_coincidence_accepted"] = dict(n=int(len(nc)), crossing=_vc(nc["crossing_class"]), team_relation=_vc(nc["team_relation"]),
                                          both_fully_estimated_in_gap=int(((nc[f"g{t}_est_frac_in_gap_a"] > 0.999) & (nc[f"g{t}_est_frac_in_gap_b"] > 0.999)).sum()))
    a = df.loc[acc]
    ea, eb = a[f"g{t}_est_frac_in_gap_a"], a[f"g{t}_est_frac_in_gap_b"]
    r["status_audit_accepted"] = dict(
        mean_estimated_share_used_a=round(float(a[f"g{t}_est_frac_used_a"].mean()), 4), mean_estimated_share_used_b=round(float(a[f"g{t}_est_frac_used_b"].mean()), 4),
        gap_both_fully_estimated=int(((ea > 0.999) & (eb > 0.999)).sum()), gap_both_fully_visible=int(((ea < 1e-9) & (eb < 1e-9)).sum()),
        gap_mixed=int(len(a) - ((ea > 0.999) & (eb > 0.999)).sum() - ((ea < 1e-9) & (eb < 1e-9)).sum()))
    sizes = a[f"g{t}_overlap_group"].groupby([a["match_id"], a[f"g{t}_overlap_group"]]).size()
    r["dependency_accepted"] = dict(accepted=int(len(a)), independent=int(a[f"g{t}_independent"].sum()), overlap_groups=int(len(sizes)),
                                    singleton_groups=int((sizes == 1).sum()), largest_group=int(sizes.max()) if len(sizes) else 0,
                                    in_groups_of_2_plus=int(sizes[sizes > 1].sum()))
    if "split" in df.columns:
        r["by_split"] = {sp: dict(outcomes=_vc(x[f"g{t}_outcome"]), accepted_independent=int((x[f"g{t}_independent"]).sum()))
                         for sp, x in df.groupby("split")}
    return r


def summary(df: pd.DataFrame, cfg: CandidateConfig, cohorts: list[str] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = dict(candidates=int(len(df)), per_match=_vc(df["match_id"]), crossing_class=_vc(df["crossing_class"]),
                               strict_pass=int(df["strict_pass"].sum()), team_relation=_vc(df["team_relation"]))
    out["dedup"] = dict(episodes_total=int(df["n_episodes"].sum()), encounters=int(len(df)), merged_encounters=int((df["n_episodes"] > 1).sum()),
                        local_minima_total=int(df["n_local_minima_total"].sum()), encounters_with_multiple_minima=int((df["n_local_minima_total"] > 1).sum()))
    out["crossing_by_strict"] = {("d<=%.2f" % cfg.strict_radius_m if s else "%.2f<d<=%.2f" % (cfg.strict_radius_m, cfg.radius_m)): _vc(g["crossing_class"])
                                 for s, g in df.groupby("strict_pass")}
    out["per_gap"] = {L.key(g): gap_report(df, g, cohorts or [], cfg) for g in cfg.gaps_s}
    return out


def _explode(series: pd.Series) -> dict[str, int]:
    cnt: dict[str, int] = {}
    for s in series:
        for r in json.loads(s):
            cnt[r] = cnt.get(r, 0) + 1
    return dict(sorted(cnt.items(), key=lambda kv: -kv[1]))


SPEED_BANDS = ([-0.01, 1, 2, 4, np.inf], ["<1", "1-2", "2-4", ">=4"])


def control_availability(df: pd.DataFrame, cohort_cols: dict[str, str], g: float, cfg: CandidateConfig) -> dict[str, Any]:
    """Pooled controls across the matches in ``df`` (callers pass training matches only). Cases: ACCEPTED local crossings whose cohort
    state is PASS (``identity_free``: every ACCEPTED local crossing). Controls: ACCEPTED non-crossing candidates. Strata: d_min band x
    slower chord-speed band. Per stratum at most one control per case (ratio <= 1:1, no duplication); where controls are fewer than
    cases each control gets weight cases / controls so the stratum is balanced in weight. Feasibility only; nothing is drawn."""
    t = L.key(g)
    d = df[df[f"g{t}_outcome"] == ACCEPTED].copy()
    d["d_band"] = np.where(d["strict_pass"], "le_strict", "loose")
    d["v_band"] = pd.cut(d["chord_speed_min_m_s"], SPEED_BANDS[0], labels=SPEED_BANDS[1]).astype(str)
    ctrl = d[d["crossing_class"] == "no_crossing"].groupby(["d_band", "v_band"]).size()
    out: dict[str, Any] = dict(controls_total=int(ctrl.sum()), controls_by_stratum={f"{a}|{b}": int(v) for (a, b), v in ctrl.items()})
    for lab, col in dict(cohort_cols, identity_free=None).items():
        sel = d["crossing_class"] == "local_crossing"
        if col is not None:
            sel &= d[col] == L.PASS
        need = d[sel].groupby(["d_band", "v_band"]).size()
        avail = ctrl.reindex(need.index).fillna(0).astype(int)
        used = np.minimum(need, avail)
        w = (need / avail.where(avail > 0)).where(avail < need)
        out[lab] = dict(cases=int(need.sum()), strata=int(len(need)), cases_in_strata_with_controls=int(need[avail >= 1].sum()),
                        controls_usable_at_most_1to1=int(used.sum()), achieved_ratio=round(float(used.sum() / max(1, need.sum())), 4),
                        max_control_weight=round(float(w.max()), 2) if w.notna().any() else 1.0,
                        by_stratum={f"{a}|{b}": dict(cases=int(n), controls=int(avail[(a, b)])) for (a, b), n in need.items()})
    return out


def check_candidates(df: pd.DataFrame, cfg: CandidateConfig) -> dict[str, Any]:
    """Internal consistency of one match's candidates: unique ids; ordered pairs; minimum inside its episode and encounter and within
    the radius; distinct encounters of a pair more than ``merge_gap_s`` apart (deduplication); per gap the outcome follows from the
    checks (REJECTED iff a check failed; LOWER only with a lower-confidence reason); overlap groups only on ACCEPTED rows."""
    ok: dict[str, bool] = {}
    if not len(df):
        return dict(all_pass=True, candidates=0)
    ok["unique_ids"] = bool(df["candidate_id"].is_unique)
    ok["slot_a_lt_slot_b"] = bool((df["slot_a"] < df["slot_b"]).all())
    ok["d_min_le_radius"] = bool((df["d_min"] <= cfg.radius_m + 1e-9).all())
    ok["t_star_in_episode"] = bool(((df["start"] - 1e-9 <= df["t_star"]) & (df["t_star"] <= df["end"] + 1e-9)).all())
    ok["episode_in_encounter"] = bool(((df["enc_start"] <= df["start"] + 1e-12) & (df["end"] <= df["enc_end"] + 1e-12)).all())
    sep = True
    for _, g in df.groupby(["period", "slot_a", "slot_b"]):
        g = g.sort_values("enc_start")
        sep &= bool((g["enc_start"].to_numpy()[1:] - g["enc_end"].to_numpy()[:-1] > cfg.merge_gap_s - 1e-9).all())
    ok["encounters_separated_by_merge_gap"] = sep
    oc = grp_ok = True
    for g in cfg.gaps_s:
        tag = L.key(g)
        allok = np.stack([df[f"g{tag}_ok_{k}"].to_numpy(bool) for k in FUNNEL], 1).all(1)
        o = df[f"g{tag}_outcome"].to_numpy()
        lower = df[f"g{tag}_lower_reasons"].map(lambda x: len(json.loads(x)) > 0).to_numpy()
        oc &= bool(((o == REJECTED) == ~allok).all() and ((o == LOWER) == (allok & lower)).all())
        grp_ok &= bool(((df[f"g{tag}_overlap_group"].to_numpy() >= 0) == (o == ACCEPTED)).all()
                       and (~df[f"g{tag}_independent"].to_numpy(bool) | (o == ACCEPTED)).all())
    ok["outcome_follows_checks"] = oc
    ok["dependency_only_on_accepted"] = grp_ok
    ok["crossing_is_metadata_only"] = bool(set(df["crossing_class"]) <= {"local_crossing", "window_crossing_only", "no_crossing"})
    return dict(ok, all_pass=all(ok.values()), candidates=int(len(df)), crossing_class=_vc(df["crossing_class"]))


# ---------------------------------------------------------------- deployability audit: framewise identity-free detection

def framewise_audit(c, mask, dup, seg, df: pd.DataFrame, cfg: CandidateConfig, quarantine: tuple[int, ...] = ()) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Detect close pairs frame by frame from the raw AVAILABLE positions (VISIBLE or ESTIMATED; no chains, no quality cuts, no
    interpolation), merge per pair within ``merge_gap_s``, and compare with the chain-based candidates. The slot pair only books which
    two points were close; the per-frame test uses the unordered point set. Unmatched framewise encounters get the reason why the
    chain-based generator could not see them (at the frame of minimum distance)."""
    cx, cy, _ = W.coefficients(seg, "linear")
    rows_out = []
    ns = c.x.shape[1]
    ia, ib = np.triu_indices(ns, 1)
    for p in sorted(int(v) for v in np.unique(c.canonical_period) if v > 0):
        rows = np.flatnonzero(c.canonical_period == p)
        t = c.video_time_s[rows]
        X, Y = c.x[rows], c.y[rows]
        near_any = np.zeros(len(ia), bool)
        CH = 4096
        hits: dict[int, list] = {}
        for a0 in range(0, len(rows), CH):
            dx = X[a0:a0 + CH, ia] - X[a0:a0 + CH, ib]
            dy = Y[a0:a0 + CH, ia] - Y[a0:a0 + CH, ib]
            d = np.hypot(dx, dy)
            fr, pr = np.nonzero(d <= cfg.radius_m)
            for f, q in zip(fr, pr):
                hits.setdefault(int(q), []).append((a0 + int(f), float(d[f, q])))
            near_any |= (d <= cfg.radius_m).any(0)
        for q, lst in hits.items():
            fi = np.array([x[0] for x in lst])
            dd = np.array([x[1] for x in lst])
            tt = t[fi]
            brk = np.r_[True, np.diff(tt) > cfg.merge_gap_s + 1e-9]
            gid = np.cumsum(brk) - 1
            for k in range(gid[-1] + 1):
                sel = gid == k
                j = int(np.argmin(np.where(sel, dd, np.inf)))
                rows_out.append(dict(match_id=int(c.match_id), period=p, slot_a=int(ia[q]), slot_b=int(ib[q]), start=float(tt[sel][0]),
                                     end=float(tt[sel][-1]), t_min=float(tt[j]), d_min=float(dd[j]), n_frames=int(sel.sum()), row=int(rows[fi[j]])))
    F = pd.DataFrame(rows_out)
    rep: dict[str, Any] = dict(framewise_encounters=int(len(F)), chain_candidates=int(len(df)))
    if not len(F):
        return F, rep
    # match framewise encounters to chain-based candidates (same pair and period, intervals overlap)
    by = {k: g for k, g in df.groupby(["period", "slot_a", "slot_b"])} if len(df) else {}
    matched = np.zeros(len(F), bool)
    cand_hit = np.zeros(len(df), bool)
    for i, r in enumerate(F.itertuples(index=False)):
        g = by.get((r.period, r.slot_a, r.slot_b))
        if g is None:
            continue
        ov = (g["enc_start"].to_numpy() <= r.end + 1e-9) & (g["enc_end"].to_numpy() >= r.start - 1e-9)
        if ov.any():
            matched[i] = True
            cand_hit[g.index.to_numpy()[ov]] = True
    F["matched_to_candidate"] = matched
    reasons = []
    for r in F.itertuples(index=False):
        if r.matched_to_candidate:
            reasons.append("")
            continue
        why = []
        for tag, s in (("a", r.slot_a), ("b", r.slot_b)):
            m = int(mask[r.row, s])
            px, _ = W.positions_at(seg, cx, cy, s, r.period, np.array([r.t_min]))
            if r.period in quarantine:
                why.append("quarantine_period")
            elif dup[r.row, s]:
                why.append(f"duplicate_position_{tag}")
            elif m & UNUSABLE:
                why.append(f"unattributed_or_ambiguous_{tag}")
            elif np.isnan(px[0]):
                rr = np.flatnonzero((c.canonical_period == r.period) & (np.abs(c.video_time_s - r.t_min) <= 1.0))
                why.append(f"no_valid_segment_jump_nearby_{tag}" if ((mask[rr, s] & Q.IMPLAUSIBLE_JUMP) != 0).any() else f"no_valid_segment_{tag}")
        reasons.append(";".join(sorted(set(why))) or "covered_but_chain_distance_above_radius")
    F["unmatched_reason"] = reasons
    risk = np.zeros(len(F), bool)
    for i, r in enumerate(F.itertuples(index=False)):
        if r.matched_to_candidate:
            continue
        rr = np.flatnonzero((c.canonical_period == r.period) & (np.abs(c.video_time_s - r.t_min) <= 1.0))
        bitsall = Q.IMPLAUSIBLE_JUMP | sum(IDENTITY_RISK_BITS.values())
        risk[i] = bool(((mask[rr][:, [r.slot_a, r.slot_b]] & bitsall) != 0).any())
    F["identity_or_jump_flag_within_1s"] = risk
    um = ~F["matched_to_candidate"]
    rep.update(framewise_matched=int((~um).sum()), framewise_unmatched=int(um.sum()),
               framewise_unmatched_reasons=_vc(F.loc[um, "unmatched_reason"].str.split(";").explode()),
               framewise_unmatched_with_identity_or_jump_flag_1s=int((um & F["identity_or_jump_flag_within_1s"]).sum()),
               candidates_with_framewise_match=int(cand_hit.sum()), candidates_without_framewise_frame=int((~cand_hit).sum()),
               candidates_without_framewise_frame_d_min=dict(median=float(np.median(df.loc[~cand_hit, "d_min"])) if (~cand_hit).any() else None))
    return F, rep
