"""Opt-in learned, deferred global association of tracklets (M6.3-A).

Used by :func:`football_fatigue.vision.identity.solve` when ``IdentityConfig.association_mode == "deferred"``;
the M6 / M6.1 / M6.2 default (``"m61"``: gated greedy linking with forced split repairs) is not touched.

1. *Candidates*: every ordered pair (a, b) of non-ambiguous tracklets with at least ``min_tracklet_frames`` boxes where b
   starts 1..``association_max_gap`` frames after a ends. Hard exclusions (never scored): both teams known and different,
   goalkeeper/referee/player role mismatch, pieces of a track split at a contact swap (``blocked``).
2. *Score*: link probability from a logistic model on scale-free kinematic, size, team, role, image-border, competition
   and colour-descriptor features (:data:`ASSOCIATION_FEATURES`). The weights are fitted on development data only
   (leave-one-match-out for every reported number) and supplied through the config; they never see reference ids at
   inference.
3. *Global assignment*: the maximum-weight bipartite matching of tracklet ends to tracklet starts with weight
   ``logit(p) - logit(tau)`` for ``p >= tau`` (each tracklet gets at most one successor and one predecessor, every link
   points forward in time, so chains are time-disjoint by construction).
4. *Deferral*: a matched link is kept only if its logit beats the logit of every other candidate of either endpoint by at
   least ``association_delta``. A matched link without that separation is **not made**; it is recorded as a competing
   hypothesis (``decision == "deferred"``), as are matched-out candidates with ``p >= tau`` (``"abstained"``).
5. *Chain consistency*: links are applied in descending logit; a link that would give one chain two different known teams
   or overlap in time is rejected and recorded.

Separation of concerns: this module decides which tracklets share a global id. It does not assign trust. The caller
(`identity.solve`) keeps the existing confidence layer and discounts every tracklet touched by a link made here, so a newly
linked segment is never ``confirmed`` until its confidence has been independently validated.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

ASSOCIATION_FEATURES = (
    "bias", "log_gap", "log1p_disp_h", "log_speed_h_per_frame", "log1p_resid_fwd", "log1p_resid_bwd", "abs_log_size_ratio",
    "team_same_known", "team_unknown", "both_goalkeepers", "border_exit", "border_entry", "log_len_a", "log_len_b",
    "log1p_out_candidates", "log1p_in_candidates", "app_torso_dist", "app_missing",
)
BORDER_FRACTION = 150.0 / 4096.0      # image-border band as a share of the image width (the dev panoramas are 4096 px wide)
ENDPOINT_FRAMES = 10                  # boxes used for the end-point position, velocity and height
DESCRIPTOR_SAMPLES = 8                # isolated colour descriptors per tracklet end
DESCRIPTOR_SPACING = 5                # frames between them
MIN_DESCRIPTORS = 3
EXTRAPOLATION_CAP = 25                # frames over which the constant-velocity prediction is extended


def _logit(p: np.ndarray | float) -> np.ndarray | float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _end_kinematics(t: Any, at_end: bool) -> tuple[np.ndarray, np.ndarray, float]:
    """Foot point at the tracklet end, per-frame foot velocity (least squares over the end-point boxes) and median height."""
    obs = t.obs[-ENDPOINT_FRAMES:] if at_end else t.obs[:ENDPOINT_FRAMES]
    f = np.array([o.frame for o in obs], float)
    x = np.array([(o.box[0] + o.box[2]) / 2.0 for o in obs])
    y = np.array([o.box[3] for o in obs])
    h = np.array([max(1.0, o.box[3] - o.box[1]) for o in obs])
    if len(f) >= 3 and f[-1] > f[0]:
        v = np.array([np.polyfit(f, x, 1)[0], np.polyfit(f, y, 1)[0]])
    else:
        v = np.zeros(2)
    i = -1 if at_end else 0
    return np.array([x[i], y[i]]), v, float(np.median(h))


def _robust(X: np.ndarray) -> np.ndarray:
    """Per-input median/MAD standardisation (unsupervised; the input is the single match being processed)."""
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0) * 1.4826
    return (X - med) / np.maximum(mad, 1e-3 + 0.05 * np.abs(med).mean())


def descriptor_endpoints(tracklets: Sequence[Any], inter: dict[tuple[int, int], bool]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """``{tracklet_id: (descriptor at the end, descriptor at the start)}`` from isolated, described boxes.

    A tracklet end is summarised by the median of up to ``DESCRIPTOR_SAMPLES`` standardised colour descriptors taken at
    least ``DESCRIPTOR_SPACING`` frames apart, walking inwards from that end. Tracklets with fewer than
    ``MIN_DESCRIPTORS`` samples are absent (the feature is then marked missing, never imputed).
    """
    rows, owner = [], []
    for t in tracklets:
        for o in t.obs:
            if o.appearance is not None and not inter.get((o.frame, o.index)):
                rows.append(o.appearance)
                owner.append((t.tracklet_id, o.frame))
    if not rows:
        return {}
    Z = _robust(np.array(rows, float))
    per: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for (tid, fr), z in zip(owner, Z):
        per[tid].append((fr, z))
    out = {}
    for tid, items in per.items():
        items.sort(key=lambda x: x[0])
        if len(items) < MIN_DESCRIPTORS:
            continue

        def pick(seq):
            kept, last = [], None
            for fr, z in seq:
                if last is None or abs(fr - last) >= DESCRIPTOR_SPACING:
                    kept.append(z)
                    last = fr
                if len(kept) == DESCRIPTOR_SAMPLES:
                    break
            return np.median(np.array(kept), axis=0) if len(kept) >= MIN_DESCRIPTORS else None
        end, start = pick(reversed(items)), pick(items)
        if end is not None and start is not None:
            out[tid] = (end, start)
    return out


def candidates(tracklets: Sequence[Any], cfg: Any, blocked: frozenset[tuple[int, int]] = frozenset()) -> list[tuple[Any, Any]]:
    """Ordered candidate pairs after the hard exclusions (see module docstring), sorted by (a start, b start, ids)."""
    ok = [t for t in tracklets if not t.ambiguous and len(t.obs) >= cfg.min_tracklet_frames]
    starts = sorted(ok, key=lambda t: (t.start, t.tracklet_id))
    out = []
    for a in sorted(ok, key=lambda t: (t.start, t.tracklet_id)):
        for b in starts:
            gap = b.start - a.end
            if gap < 1:
                continue
            if gap > cfg.association_max_gap:
                break
            if cfg.use_team and a.team and b.team and a.team != b.team:
                continue
            if a.role != b.role:
                continue
            if (a.local_track_id, b.local_track_id) in blocked:
                continue
            out.append((a, b))
    return out


def candidate_features(cands: Sequence[tuple[Any, Any]], tracklets: Sequence[Any], cfg: Any,
                       inter: dict[tuple[int, int], bool]) -> np.ndarray:
    """Feature matrix, columns :data:`ASSOCIATION_FEATURES` (the bias column is 1)."""
    if not cands:
        return np.zeros((0, len(ASSOCIATION_FEATURES)))
    width = cfg.association_image_width
    if width is None:
        width = max(o.box[2] for t in tracklets for o in t.obs)
    band = BORDER_FRACTION * width
    kin_end, kin_start = {}, {}
    for t in {id(x): x for pair in cands for x in pair}.values():
        kin_end[t.tracklet_id] = _end_kinematics(t, True)
        kin_start[t.tracklet_id] = _end_kinematics(t, False)
    out_n, in_n = defaultdict(int), defaultdict(int)
    for a, b in cands:
        out_n[a.tracklet_id] += 1
        in_n[b.tracklet_id] += 1
    desc = descriptor_endpoints(tracklets, inter)
    X = np.zeros((len(cands), len(ASSOCIATION_FEATURES)))
    for i, (a, b) in enumerate(cands):
        pa, va, ha = kin_end[a.tracklet_id]
        pb, vb, hb = kin_start[b.tracklet_id]
        gap = b.start - a.end
        hm = (ha + hb) / 2.0
        disp = float(np.linalg.norm(pb - pa)) / hm
        ge = min(gap, EXTRAPOLATION_CAP)
        rf = float(np.linalg.norm(pb - (pa + va * ge))) / hm
        rb = float(np.linalg.norm(pa - (pb - vb * ge))) / hm
        known = bool(a.team and b.team)
        app = float(np.linalg.norm(desc[a.tracklet_id][0] - desc[b.tracklet_id][1])) \
            if a.tracklet_id in desc and b.tracklet_id in desc else None
        X[i] = (1.0, math.log(gap), math.log1p(disp), math.log(disp / gap + 0.01), math.log1p(rf), math.log1p(rb),
                abs(math.log(hb / ha)), float(known and a.team == b.team), float(not known),
                float(a.role == "goalkeeper" and b.role == "goalkeeper"),
                float(min(pa[0], width - pa[0]) < band), float(min(pb[0], width - pb[0]) < band),
                math.log(len(a.obs)), math.log(len(b.obs)), math.log1p(out_n[a.tracklet_id]), math.log1p(in_n[b.tracklet_id]),
                0.0 if app is None else app, float(app is None))
    return X


def link_probability(X: np.ndarray, weights: Sequence[float]) -> np.ndarray:
    w = np.asarray(weights, float)
    if w.shape != (X.shape[1],):
        raise ValueError(f"association_weights has {w.size} coefficients, expected {X.shape[1]} ({ASSOCIATION_FEATURES})")
    return 1.0 / (1.0 + np.exp(-(X @ w)))


def global_assignment(cands: Sequence[tuple[Any, Any]], p: np.ndarray, tau: float) -> list[int]:
    """Indices of the maximum-weight matching of ends to starts over candidates with ``p >= tau`` (weight logit(p)-logit(tau))."""
    keep = np.flatnonzero(p >= tau)
    if not len(keep):
        return []
    outs = sorted({cands[k][0].tracklet_id for k in keep})
    ins = sorted({cands[k][1].tracklet_id for k in keep})
    oi, ii = {t: i for i, t in enumerate(outs)}, {t: i for i, t in enumerate(ins)}
    W = np.zeros((len(outs), len(ins)))
    K = -np.ones((len(outs), len(ins)), int)
    w = _logit(p[keep]) - _logit(tau)
    for k, wk in zip(keep, w):
        a, b = cands[k]
        W[oi[a.tracklet_id], ii[b.tracklet_id]] = wk
        K[oi[a.tracklet_id], ii[b.tracklet_id]] = k
    rows, cols = linear_sum_assignment(-W)
    return sorted(int(K[i, j]) for i, j in zip(rows, cols) if K[i, j] >= 0 and W[i, j] > 0)


def defer(cands: Sequence[tuple[Any, Any]], p: np.ndarray, matched: Sequence[int], delta: float
          ) -> tuple[list[int], list[tuple[int, float, int | None]]]:
    """Split matched links into kept ones and deferred ones.

    The margin of a matched link is its logit minus the best logit among all *other* candidates of its tail or head
    (candidates below ``tau`` included). Returns ``(kept, [(index, margin, best competitor index)])``.
    """
    lg = _logit(p)
    best_o: dict[int, list[int]] = defaultdict(list)
    best_i: dict[int, list[int]] = defaultdict(list)
    for k, (a, b) in enumerate(cands):
        best_o[a.tracklet_id].append(k)
        best_i[b.tracklet_id].append(k)
    kept, held = [], []
    for k in matched:
        a, b = cands[k]
        alt = [kk for kk in best_o[a.tracklet_id] + best_i[b.tracklet_id] if kk != k]
        if alt:
            top = max(alt, key=lambda kk: (lg[kk], -kk))
            margin = float(lg[k] - lg[top])
        else:
            top, margin = None, math.inf
        if margin >= delta:
            kept.append(k)
        else:
            held.append((k, margin, top))
    return kept, held


def associate(tracklets: Sequence[Any], cfg: Any, inter: dict[tuple[int, int], bool],
              blocked: frozenset[tuple[int, int]] = frozenset()) -> tuple[dict[int, int], list[dict[str, Any]]]:
    """``({tracklet_id: chain id}, link records)`` - the drop-in counterpart of ``identity.link_tracklets``.

    Record decisions: ``linked`` (source ``association``), ``deferred`` (matched, but the margin to a competing candidate is
    below ``association_delta``; no link is made), ``abstained`` (candidate with ``p >= tau`` that lost the assignment) and
    ``rejected`` (link dropped by the chain-consistency check).
    """
    if cfg.association_weights is None:
        raise ValueError("association_mode='deferred' needs association_weights (fitted on development data)")
    chain = {t.tracklet_id: t.tracklet_id for t in tracklets}
    records: list[dict[str, Any]] = []
    cands = candidates(tracklets, cfg, blocked)
    if not cands:
        return chain, records
    X = candidate_features(cands, tracklets, cfg, inter)
    p = link_probability(X, cfg.association_weights)
    matched = global_assignment(cands, p, cfg.association_tau)
    kept, held = defer(cands, p, matched, cfg.association_delta)
    lg = _logit(p)

    def rec(k: int, decision: str, **extra: Any) -> dict[str, Any]:
        a, b = cands[k]
        return dict({"from": a.tracklet_id, "to": b.tracklet_id, "gap": b.start - a.end, "p": round(float(p[k]), 4),
                     "decision": decision, "source": "association"}, **extra)

    mset = set(matched)
    for k in np.flatnonzero(p >= cfg.association_tau):
        if int(k) not in mset:
            records.append(rec(int(k), "abstained", reason="competing_link"))
    for k, margin, top in held:
        c = cands[top] if top is not None else None
        records.append(rec(k, "deferred", reason="insufficient_margin", margin=round(margin, 3),
                           competitor=None if c is None else {"from": c[0].tracklet_id, "to": c[1].tracklet_id,
                                                              "p": round(float(p[top]), 4)}))
    by_id = {t.tracklet_id: t for t in tracklets}
    members = {t.tracklet_id: [t.tracklet_id] for t in tracklets}
    for k in sorted(kept, key=lambda k: (-lg[k], cands[k][0].tracklet_id, cands[k][1].tracklet_id)):
        a, b = cands[k]
        ca, cb = chain[a.tracklet_id], chain[b.tracklet_id]
        if ca == cb:
            records.append(rec(k, "rejected", reason="same_chain"))
            continue
        ta = {by_id[m].team for m in members[ca] if by_id[m].team}
        tb = {by_id[m].team for m in members[cb] if by_id[m].team}
        if cfg.use_team and len(ta | tb) > 1:
            records.append(rec(k, "rejected", reason="chain_team_conflict"))
            continue
        if max(by_id[m].end for m in members[ca]) >= min(by_id[m].start for m in members[cb]):
            records.append(rec(k, "rejected", reason="simultaneous_overlap"))
            continue
        for m in members[cb]:
            chain[m] = ca
        members[ca].extend(members.pop(cb))
        records.append(rec(k, "linked", cost=round(float(-lg[k]), 4)))
    return chain, records
