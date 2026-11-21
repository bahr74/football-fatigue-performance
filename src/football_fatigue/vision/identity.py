"""Opt-in global player identity layer on top of a local tracker's output (M6).

Local tracker IDs are kept as given (``local_track_id``). This module:

1. cuts every local track into *tracklets*, splitting it where evidence says the track has been
   handed from one person to another (swap-aware splitting). A split needs two independent cues:
   a confident team-evidence change point plus an interaction, a motion jump or a tracker gap
   near it; or a motion jump during an interaction. Single-cue candidates are rejected and logged.
   Frames of the interaction around a split become their own ``ambiguous`` tracklet;
2. links tracklets into *global identities* across short gaps with a constant-velocity gate whose
   spread grows with the gap (quantiles fitted on development windows), team/role gates that only
   act when both sides are confident, a size gate, a uniqueness/non-overlap constraint and an
   abstention margin against competing explanations. Every accepted, rejected and abstained link
   is recorded;
3. gives every box an identity status (``confirmed`` / ``provisional`` / ``ambiguous`` /
   ``unknown``) and a confidence.

No box is ever removed or moved, so detection metrics are unchanged; only identity changes. Two
boxes visible in the same frame never share a global ID. The online mode re-solves a sliding
window every ``online_step`` frames and only emits identities ``online_latency`` frames after the
fact, so its output is causal with a fixed latency. Ground truth is never read here.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any, Iterable, Sequence

import numpy as np

from .team_appearance import OTHER, TEAM_LABELS, aggregate_votes

IDENTITY_SCHEMA_VERSION = 1
STATUSES = ("confirmed", "provisional", "ambiguous", "unknown")

# Undamped constant-velocity foot-point error in box heights, p90 per gap (frames), fitted on the
# M6 development windows (data/processed/vision_eval/m6-global-identity-v1/motion_gate_fit_dev.json).
DEFAULT_MOTION_P90 = ((1, 0.052), (2, 0.086), (3, 0.115), (5, 0.164), (8, 0.25), (12, 0.399), (18, 0.684),
                      (25, 1.093), (37, 1.949), (50, 3.031), (75, 5.584))


@dataclass(frozen=True)
class IdentityConfig:
    split: bool = True                 # swap-aware track splitting
    link: bool = True                  # global tracklet linking
    use_team: bool = True              # team evidence in splitting and linking gates
    team_isolated_only: bool = True    # ignore team votes of boxes in an interaction
    split_window: int = 12             # decided team votes on each side of a change point
    split_share: float = 0.75          # majority share required on each side
    split_support_frames: int = 25     # second cue must lie within this many frames of the change
    split_on_motion: bool = True       # allow motion-jump + interaction splits (no team cue)
    unpartnered_splits: str = "restore"  # restore | open: splits without a mirror-image partner
    jump_ratio: float = 2.5            # motion jump: residual > jump_ratio * p90(1-frame) * gap scale
    interaction_iou: float = 0.05
    interaction_foot: float = 0.6      # foot-point distance in box heights counted as interaction
    max_link_gap: int = 75
    link_require_team_gap: int | None = None  # M6.1: links over gaps > N frames need a known team on both sides
    motion_p90: tuple[tuple[int, float], ...] = DEFAULT_MOTION_P90
    motion_inflation: float = 1.25     # widen the reference-fitted spread for detection noise
    motion_floor: float = 0.05         # box heights
    motion_gate: float = 1.0           # accept when residual / spread <= gate
    velocity_frames: int = 10
    size_ratio: float = 1.5
    link_margin: float = 0.5           # abstain unless the next-best competitor costs this much more
    team_min_votes: int = 5
    team_min_share: float = 0.7
    role_share: float = 0.6            # share of goalkeeper (or referee) detections that sets a tracklet's role
    min_tracklet_frames: int = 5       # shorter tracklets keep their own id with status unknown
    confirm_threshold: float = 0.9
    provisional_threshold: float = 0.5
    calibration: tuple[float, ...] | None = None  # logistic coefficients (see CONFIDENCE_FEATURES)
    confidence_features: str = "m6"    # "m6" (CONFIDENCE_FEATURES) | "m61" (+ CONFIDENCE_FEATURES_M61_EXTRA)
    online_window: int = 250
    online_step: int = 25
    online_latency: int = 50
    # M6.2 contact-aware decisions (default off: M6 / M6.1 behaviour is unchanged)
    contact_mode: str = "off"          # off | evidence
    contact_weights: tuple[float, ...] | None = None  # intercept + CONTACT_FEATURES logistic weights on raw features
    contact_keep_below: float = 0.0    # p(swap) <= this: episode is a verified keep
    contact_swap_above: float = 1.0    # p(swap) >= this: split the track at the episode
    contact_window: int = 10           # boxes on each side used as evidence
    contact_min_side: int = 5          # episodes with fewer isolated boxes on a side stay undecided
    # M6.3-A learned deferred global association (default off: "m61" is the M6 / M6.1 / M6.2 linker)
    association_mode: str = "m61"      # m61 | deferred (see association.py)
    association_weights: tuple[float, ...] | None = None  # intercept + ASSOCIATION_FEATURES logistic weights on raw features
    association_tau: float = 0.3       # minimum link probability for the assignment
    association_delta: float = 0.5     # a link is made only if it beats every competitor of either end by this logit margin
    association_max_gap: int = 750     # frames
    association_image_width: int | None = None  # border-feature scale; None = widest observed box edge
    association_cap: str = "multiply"  # "multiply": confidence x discount (M6.3-A benchmark); "status": confidence kept, 'confirmed' -> 'provisional'
    association_confidence_discount: float | None = None  # None -> 0.9 (confirm_threshold > 0.9) else 0.9 * confirm_threshold; multiplies the confidence of tracklets touched by an association link

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IdentityConfig":
        d = {k: v for k, v in data.items() if not k.startswith("_")}  # "_"-keys are comments
        if "motion_p90" in d:
            d["motion_p90"] = tuple(tuple(x) for x in d["motion_p90"])
        if d.get("calibration") is not None:
            d["calibration"] = tuple(d["calibration"])
        if d.get("contact_weights") is not None:
            d["contact_weights"] = tuple(d["contact_weights"])
        if d.get("association_weights") is not None:
            d["association_weights"] = tuple(d["association_weights"])
        return cls(**d)

    @property
    def effective_association_discount(self) -> float:
        """The discount actually applied; the default is derived so that a default config is always valid."""
        if self.association_confidence_discount is not None:
            return self.association_confidence_discount
        return 0.9 if self.confirm_threshold > 0.9 else 0.9 * self.confirm_threshold

    def validate_association(self) -> None:
        """Raise ``ValueError`` for an unusable association setting (called by :func:`solve`)."""
        if self.association_mode not in ("m61", "deferred"):
            raise ValueError(f"unknown association_mode {self.association_mode!r}")
        if self.association_cap not in ("multiply", "status"):
            raise ValueError(f"unknown association_cap {self.association_cap!r}")
        if self.association_mode == "deferred":
            if self.association_weights is None:
                raise ValueError("association_mode='deferred' needs association_weights")
            if not 0.0 < self.association_tau < 1.0 or self.association_delta < 0.0 or self.association_max_gap < 1:
                raise ValueError("association_tau must be in (0, 1), association_delta >= 0, association_max_gap >= 1")
            if self.association_cap == "multiply" and not 0.0 < self.effective_association_discount < self.confirm_threshold:
                # a discounted box must be unable to reach 'confirmed': newly linked segments are never auto-trusted
                raise ValueError("association_confidence_discount must lie in (0, confirm_threshold)")


@dataclass(frozen=True)
class Observation:
    """One local-tracker box. ``index`` is the position in the frame's non-ball box list."""
    frame: int
    index: int
    local_track_id: int
    box: tuple[float, float, float, float]
    detected_class: str = "player"
    team_vote: str | None = None
    appearance: tuple[float, ...] | None = field(default=None, compare=False)  # box colour descriptor (M6.2 contact evidence)


@dataclass
class Tracklet:
    tracklet_id: int
    local_track_id: int
    obs: list[Observation]
    ambiguous: bool = False
    team: str | None = None
    team_confidence: float = 0.0
    team_votes: int = 0
    role: str = "player"               # "goalkeeper" | "referee" | "player" (detector-class majority)

    @property
    def goalkeeper(self) -> bool:
        return self.role == "goalkeeper"

    @property
    def start(self) -> int:
        return self.obs[0].frame

    @property
    def end(self) -> int:
        return self.obs[-1].frame

    def frames(self) -> set[int]:
        return {o.frame for o in self.obs}


def _foot(box: Sequence[float]) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2.0, box[3]])


def _height(box: Sequence[float]) -> float:
    return max(1.0, box[3] - box[1])


def _iou_matrix(b: np.ndarray) -> np.ndarray:
    ix = np.clip(np.minimum(b[:, None, 2], b[None, :, 2]) - np.maximum(b[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(b[:, None, 3], b[None, :, 3]) - np.maximum(b[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area[:, None] + area[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


@lru_cache(maxsize=4096)
def motion_spread(gap: int, cfg: IdentityConfig) -> float:
    """Allowed foot-point residual scale (box heights) after ``gap`` frames, log-interpolated."""
    gs = np.array([g for g, _ in cfg.motion_p90], float)
    ps = np.array([p for _, p in cfg.motion_p90], float)
    g = float(min(max(gap, gs[0]), gs[-1]))
    p = float(np.exp(np.interp(np.log(g), np.log(gs), np.log(ps))))
    if gap > gs[-1]:
        p *= gap / gs[-1]
    return p * cfg.motion_inflation + cfg.motion_floor


def _velocity(points: list[tuple[int, np.ndarray]], k: int) -> np.ndarray:
    """Per-frame foot velocity from the first/last points of a short frame-indexed run."""
    if len(points) < 2:
        return np.zeros(2)
    pts = points[-(k + 1):]
    df = pts[-1][0] - pts[0][0]
    return (pts[-1][1] - pts[0][1]) / df if df > 0 else np.zeros(2)


def interaction_partners(obs_by_frame: dict[int, list[Observation]], cfg: IdentityConfig) -> dict[tuple[int, int], list[int]]:
    """``{(frame, index): [partner indices]}`` with the same closeness rule as :func:`interactions`."""
    out: dict[tuple[int, int], list[int]] = {}
    for f, obs in obs_by_frame.items():
        if len(obs) < 2:
            continue
        b = np.array([o.box for o in obs], float)
        feet = np.column_stack([(b[:, 0] + b[:, 2]) / 2.0, b[:, 3]])
        hs = np.maximum(1.0, b[:, 3] - b[:, 1])
        d = np.linalg.norm(feet[:, None] - feet[None], axis=2) / np.minimum(hs[:, None], hs[None])
        close = (d < cfg.interaction_foot) | (_iou_matrix(b) > cfg.interaction_iou)
        np.fill_diagonal(close, False)
        for i in np.flatnonzero(close.any(axis=1)):
            out[(f, obs[i].index)] = [obs[j].index for j in np.flatnonzero(close[i])]
    return out


def interactions(obs_by_frame: dict[int, list[Observation]], cfg: IdentityConfig) -> dict[tuple[int, int], bool]:
    """``{(frame, index): True}`` for boxes overlapping or standing close to another box."""
    out: dict[tuple[int, int], bool] = {}
    for f, obs in obs_by_frame.items():
        n = len(obs)
        if n < 2:
            continue
        b = np.array([o.box for o in obs], float)
        feet = np.column_stack([(b[:, 0] + b[:, 2]) / 2.0, b[:, 3]])
        hs = np.maximum(1.0, b[:, 3] - b[:, 1])
        d = np.linalg.norm(feet[:, None] - feet[None], axis=2) / np.minimum(hs[:, None], hs[None])
        close = (d < cfg.interaction_foot) | (_iou_matrix(b) > cfg.interaction_iou)
        np.fill_diagonal(close, False)
        for i in np.flatnonzero(close.any(axis=1)):
            out[(f, obs[i].index)] = True
    return out


def _team_change_points(obs: list[Observation], cfg: IdentityConfig) -> list[dict[str, Any]]:
    decided = [(i, o.team_vote) for i, o in enumerate(obs) if o.team_vote in TEAM_LABELS]
    k = cfg.split_window
    c0 = np.concatenate([[0], np.cumsum([v == TEAM_LABELS[0] for _, v in decided])])

    def majority(lo: int) -> tuple[str, int]:
        n0 = int(c0[lo + k] - c0[lo])
        if n0 * 2 == k:  # tie: first vote of the window, as Counter.most_common would return
            return decided[lo][1], n0
        return (TEAM_LABELS[0], n0) if n0 * 2 > k else (TEAM_LABELS[1], k - n0)

    cands = []
    for j in range(k, len(decided) - k + 1):
        before, after = majority(j - k), majority(j)
        sb, sa = before[1] / k, after[1] / k
        if before[0] != after[0] and sb >= cfg.split_share and sa >= cfg.split_share:
            cands.append((j, (sb + sa) / 2, before[0], after[0]))
    # one change point per run of consecutive candidates: best score, then the middle one
    out, run = [], []
    for c in cands + [None]:
        if c is not None and (not run or c[0] == run[-1][0] + 1):
            run.append(c)
            continue
        if run:
            best = max(r[1] for r in run)
            top = [r for r in run if r[1] == best]
            j, score, lb, la = top[len(top) // 2]
            out.append({"obs_index": decided[j][0], "prev_obs_index": decided[j - 1][0], "score": round(score, 3),
                        "team_before": lb, "team_after": la})
        run = [c] if c is not None else []
    return out


def _motion_jumps(obs: list[Observation], cfg: IdentityConfig) -> list[tuple[int, float]]:
    """Observation indices where the box jumps beyond the constant-velocity gate: (index, ratio)."""
    out = []
    pts: list[tuple[int, np.ndarray]] = []
    for i, o in enumerate(obs):
        p = _foot(o.box)
        if len(pts) >= 3:
            gap = o.frame - pts[-1][0]
            pred = pts[-1][1] + _velocity(pts, cfg.velocity_frames) * gap
            r = float(np.linalg.norm(p - pred)) / _height(o.box) / motion_spread(gap, cfg)
            if r > cfg.jump_ratio:
                out.append((i, r))
        pts.append((o.frame, p))
    return out


def split_tracks(tracks: dict[int, list[Observation]], inter: dict[tuple[int, int], bool],
                 cfg: IdentityConfig) -> tuple[list[Tracklet], list[dict[str, Any]]]:
    """Cut local tracks into tracklets; returns tracklets and split decision records."""
    tracklets: list[Tracklet] = []
    records: list[dict[str, Any]] = []
    for tid in sorted(tracks):
        obs = tracks[tid]
        cuts: list[tuple[int, int, dict[str, Any]]] = []  # (first ambiguous idx, first idx after, record)
        if cfg.split and len(obs) >= 2:
            frames = [o.frame for o in obs]
            near = [inter.get((o.frame, o.index), False) for o in obs]
            jumps = _motion_jumps(obs, cfg)
            changes = _team_change_points(obs, cfg) if cfg.use_team else []
            used_jumps: set[int] = set()
            for c in changes:
                lo, hi = c["prev_obs_index"], c["obs_index"]
                f_lo, f_hi = frames[lo] - cfg.split_support_frames, frames[hi] + cfg.split_support_frames
                zone = [i for i in range(len(obs)) if f_lo <= frames[i] <= f_hi]
                inter_idx = [i for i in zone if near[i]]
                jump_idx = [i for i, _ in jumps if f_lo <= frames[i] <= f_hi]
                gap_idx = [i for i in zone if i > 0 and frames[i] - frames[i - 1] > 1]
                cues = ["team_change"] + (["interaction"] if inter_idx else []) + (["motion_jump"] if jump_idx else []) \
                    + (["tracker_gap"] if gap_idx else [])
                rec = {"local_track_id": tid, "frame_before": frames[lo], "frame_after": frames[hi],
                       "cues": cues, "team_score": c["score"], "team_before": c["team_before"],
                       "team_after": c["team_after"]}
                if len(cues) < 2:
                    records.append(dict(rec, decision="rejected", reason="single_cue"))
                    continue
                if inter_idx:
                    # the contiguous interaction run nearest to the change becomes ambiguous
                    centre = min(inter_idx, key=lambda i: min(abs(i - lo), abs(i - hi)))
                    a = b = centre
                    while a - 1 in inter_idx and a - 1 >= 0:
                        a -= 1
                    while b + 1 in inter_idx:
                        b += 1
                    a, b = min(a, hi), max(b + 1, hi)
                    a = max(a, 1)
                else:
                    a = b = hi
                used_jumps.update(jump_idx)
                cuts.append((a, b, dict(rec, decision="split", score=round(c["score"] + 0.5 * (len(cues) - 1), 3))))
            if cfg.split_on_motion:
                for i, r in jumps:
                    if i in used_jumps:
                        continue
                    lo_f, hi_f = frames[i - 1] - 2, frames[i] + 2
                    if any(near[j] for j in range(len(obs)) if lo_f <= frames[j] <= hi_f):
                        cuts.append((i, i, {"local_track_id": tid, "frame_before": frames[i - 1], "frame_after": frames[i],
                                            "cues": ["motion_jump", "interaction"], "jump_ratio": round(r, 2),
                                            "decision": "split", "score": round(r, 2)}))
                    else:
                        records.append({"local_track_id": tid, "frame_before": frames[i - 1], "frame_after": frames[i],
                                        "cues": ["motion_jump"], "jump_ratio": round(r, 2),
                                        "decision": "rejected", "reason": "single_cue"})
        # apply cuts in order, dropping overlapping ones (keep the higher score)
        cuts.sort(key=lambda c: (c[0], -c[2]["score"]))
        kept: list[tuple[int, int, dict[str, Any]]] = []
        for c in cuts:
            if kept and c[0] < kept[-1][1] + 1:
                if c[2]["score"] > kept[-1][2]["score"]:
                    records.append(dict(kept[-1][2], decision="rejected", reason="overlapping_split"))
                    kept[-1] = c
                else:
                    records.append(dict(c[2], decision="rejected", reason="overlapping_split"))
                continue
            kept.append(c)
        pieces: list[Tracklet] = []
        meta: list[tuple[dict[str, Any], int | None, int | None]] = []
        pos = 0
        for a, b, rec in kept:
            if a > pos:
                pieces.append(Tracklet(0, tid, obs[pos:a]))
            pre = len(pieces) - 1 if pieces and not pieces[-1].ambiguous else None
            amb = None
            if b > a:
                pieces.append(Tracklet(0, tid, obs[a:b], ambiguous=True))
                amb = len(pieces) - 1
            rec["ambiguous_frames"] = [obs[a].frame, obs[b - 1].frame] if b > a else None
            meta.append((rec, pre, amb))
            records.append(rec)
            pos = b
        if pos < len(obs):
            pieces.append(Tracklet(0, tid, obs[pos:]))
        for rec, pre, amb in meta:
            nxt = (amb if amb is not None else pre if pre is not None else -1) + 1
            rec["_pieces"] = (pieces[pre] if pre is not None else None, pieces[amb] if amb is not None else None,
                              pieces[nxt] if nxt < len(pieces) and not pieces[nxt].ambiguous else None)
        tracklets.extend(pieces)
    for n, t in enumerate(sorted(tracklets, key=lambda t: (t.start, t.local_track_id))):
        t.tracklet_id = n + 1
        ev = aggregate_votes([o.team_vote for o in t.obs], cfg.team_min_votes, cfg.team_min_share)
        t.team, t.team_confidence, t.team_votes = ev.label, ev.confidence, ev.votes
        roles = Counter(o.detected_class for o in t.obs)
        t.role = next((r for r in ("goalkeeper", "referee") if roles[r] / len(t.obs) >= cfg.role_share), "player")
    tracklets.sort(key=lambda t: t.tracklet_id)
    for rec in records:
        if "_pieces" in rec:
            rec["pre"], rec["ambiguous"], rec["post"] = (p.tracklet_id if p else None for p in rec.pop("_pieces"))
    return tracklets, records


def _end_gap_ratio(a: Tracklet, b: Tracklet, cfg: IdentityConfig) -> float:
    """Constant-velocity residual / allowed spread from the end of ``a`` to the start of ``b``."""
    fa, pa, va, ha = _endpoint(a, True, cfg.velocity_frames)
    fb, pb, vb, hb = _endpoint(b, False, cfg.velocity_frames)
    gap = max(1, fb - fa)
    ef = np.linalg.norm(pa + va * gap - pb)
    eb = np.linalg.norm(pb - vb * gap - pa)
    return float((ef + eb) / 2.0 / ((ha + hb) / 2.0) / motion_spread(gap, cfg))


def repair_splits(tracklets: list[Tracklet], split_records: list[dict[str, Any]],
                  cfg: IdentityConfig) -> list[tuple[int, int, str]]:
    """Forced links that resolve accepted splits.

    A split whose partner track shows the mirror-image split at the same time (opposite team change,
    or a simultaneous motion jump) and whose crossed continuations are kinematically plausible is a
    *swap*: the pieces are re-linked crosswise. A split without such a partner is either restored
    (pieces re-linked in their original order, the interaction interval stays ambiguous) or left
    open for the general linker, depending on ``cfg.unpartnered_splits``.
    """
    by_id = {t.tracklet_id: t for t in tracklets}
    acc = [r for r in split_records if r["decision"] == "split" and r.get("pre") and r.get("post")]
    forced: list[tuple[int, int, str]] = []
    paired: set[int] = set()
    for i, r in enumerate(acc):
        if i in paired:
            continue
        best = None
        for j, s in enumerate(acc):
            if j == i or j in paired or s["local_track_id"] == r["local_track_id"]:
                continue
            team_r, team_s = "team_change" in r["cues"], "team_change" in s["cues"]
            if team_r and team_s:
                if abs(s["frame_after"] - r["frame_after"]) > cfg.split_support_frames:
                    continue
                if not (s["team_before"] == r["team_after"] and s["team_after"] == r["team_before"]):
                    continue
            elif not team_r and not team_s:
                if abs(s["frame_after"] - r["frame_after"]) > 3:
                    continue
            else:
                continue
            ra, rb, sa, sb = by_id[r["pre"]], by_id[r["post"]], by_id[s["pre"]], by_id[s["post"]]
            if rb.start <= sa.end or sb.start <= ra.end:
                continue
            crossed = _end_gap_ratio(ra, sb, cfg) + _end_gap_ratio(sa, rb, cfg)
            straight = _end_gap_ratio(ra, rb, cfg) + _end_gap_ratio(sa, sb, cfg)
            if crossed > 4.0 * cfg.motion_gate or (not team_r and crossed >= straight):
                continue
            if best is None or crossed < best[0]:
                best = (crossed, j, s)
        if best is not None:
            _, j, s = best
            paired.update((i, j))
            r["repair"] = s["repair"] = "swap"
            r["partner_local_track_id"], s["partner_local_track_id"] = s["local_track_id"], r["local_track_id"]
            forced += [(r["pre"], s["post"], "swap_repair"), (s["pre"], r["post"], "swap_repair")]
    for i, r in enumerate(acc):
        if i in paired:
            continue
        if cfg.unpartnered_splits == "restore":
            r["repair"] = "restored"
            if r.get("ambiguous"):
                forced += [(r["pre"], r["ambiguous"], "restore"), (r["ambiguous"], r["post"], "restore")]
            else:
                forced.append((r["pre"], r["post"], "restore"))
        else:
            r["repair"] = "open"
    return forced


def _endpoint(t: Tracklet, at_end: bool, k: int) -> tuple[int, np.ndarray, np.ndarray, float]:
    """(frame, foot point, per-frame velocity pointing forward in time, box height) at one end."""
    obs = t.obs[-(k + 1):] if at_end else t.obs[:k + 1]
    pts = [(o.frame, _foot(o.box)) for o in obs]
    v = _velocity(pts, k)
    o = t.obs[-1] if at_end else t.obs[0]
    return o.frame, _foot(o.box), v, _height(o.box)


def link_candidates(tracklets: list[Tracklet], cfg: IdentityConfig,
                    blocked: frozenset[tuple[int, int]] = frozenset()) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Gated link candidates a->b (b starts after a ends) and the gate rejections.

    ``blocked`` holds (local track a, local track b) pairs that must not be linked (pieces of a track split at a
    contact the evidence says was a swap).
    """
    starts = sorted(tracklets, key=lambda t: t.start)
    cands, rejected = [], []
    for a in tracklets:
        if len(a.obs) < 1:
            continue
        fa, pa, va, ha = _endpoint(a, True, cfg.velocity_frames)
        for b in starts:
            gap = b.start - a.end
            if gap < 1:
                continue
            if gap > cfg.max_link_gap:
                break
            if (a.local_track_id, b.local_track_id) in blocked:
                rejected.append({"from": a.tracklet_id, "to": b.tracklet_id, "gap": gap, "decision": "rejected",
                                 "reason": "contact_swap_split"})
                continue
            fb, pb, vb, hb = _endpoint(b, False, cfg.velocity_frames)
            rec = {"from": a.tracklet_id, "to": b.tracklet_id, "gap": gap}
            reason = None
            if cfg.use_team and a.team and b.team and a.team != b.team:
                reason = "team_conflict"
            elif a.role != b.role and min(len(a.obs), len(b.obs)) >= cfg.min_tracklet_frames:
                reason = "role_conflict"
            elif cfg.link_require_team_gap is not None and gap > cfg.link_require_team_gap and not (a.team and b.team):
                reason = "team_unknown_long_gap"
            ratio = max(ha, hb) / min(ha, hb)
            if reason is None and ratio > cfg.size_ratio:
                reason = "size"
            # forward prediction from a and backward prediction from b, averaged residual
            ef = np.linalg.norm(pa + va * gap - pb)
            eb = np.linalg.norm(pb - vb * gap - pa)
            h = (ha + hb) / 2.0
            r = float((ef + eb) / 2.0 / h / motion_spread(gap, cfg))
            rec["motion_ratio"] = round(r, 3)
            if reason is None and r > cfg.motion_gate:
                reason = "motion"
            team_term = 0.0 if (a.team and a.team == b.team) else 0.25
            cost = r * r + math.log(ratio) ** 2 * 4.0 + team_term + 0.1 * a.ambiguous + 0.1 * b.ambiguous
            if reason == "team_unknown_long_gap" and ratio <= cfg.size_ratio and r <= cfg.motion_gate:
                # never linked, but a kinematically plausible alternative still makes competing links ambiguous
                rejected.append(dict(rec, decision="rejected", reason=reason))
                cands.append(dict(rec, cost=round(cost, 4), shadow=True))
                continue
            if reason:
                if reason != "motion" or r < 3 * cfg.motion_gate:
                    rejected.append(dict(rec, decision="rejected", reason=reason))
                continue
            cands.append(dict(rec, cost=round(cost, 4)))
    return cands, rejected


def link_tracklets(tracklets: list[Tracklet], cfg: IdentityConfig,
                   forced: Sequence[tuple[int, int, str]] = (),
                   blocked: frozenset[tuple[int, int]] = frozenset()) -> tuple[dict[int, int], list[dict[str, Any]]]:
    """Greedy unique linking with abstention; returns {tracklet_id: chain id} and link records.

    ``forced`` links (from split repair) are applied first; they skip the gates but not the
    uniqueness and non-overlap constraints.
    """
    by_id = {t.tracklet_id: t for t in tracklets}
    chain = {t.tracklet_id: t.tracklet_id for t in tracklets}
    members = {t.tracklet_id: [t.tracklet_id] for t in tracklets}
    frames_of = {t.tracklet_id: t.frames() for t in tracklets}
    records: list[dict[str, Any]] = []
    succ: dict[int, int] = {}
    pred: dict[int, int] = {}
    for a, b, why in forced:
        rec = {"from": a, "to": b, "gap": by_id[b].start - by_id[a].end, "source": why}
        ca, cb = chain[a], chain[b]
        fa = set().union(*(frames_of[m] for m in members[ca]))
        fb = set().union(*(frames_of[m] for m in members[cb]))
        if a in succ or b in pred or ca == cb or fa & fb:
            records.append(dict(rec, decision="rejected", reason="forced_link_conflict"))
            continue
        succ[a], pred[b] = b, a
        for m in members[cb]:
            chain[m] = ca
        members[ca].extend(members.pop(cb))
        records.append(dict(rec, decision="linked", cost=0.0))
    if not cfg.link:
        return chain, records
    cands, rejected = link_candidates(tracklets, cfg, blocked)
    records.extend(rejected)
    out_c: dict[int, list[float]] = defaultdict(list)
    in_c: dict[int, list[float]] = defaultdict(list)
    for c in cands:
        out_c[c["from"]].append(c["cost"])
        in_c[c["to"]].append(c["cost"])
    for c in sorted(cands, key=lambda c: (c["cost"], c["from"], c["to"])):
        a, b = c["from"], c["to"]
        if c.get("shadow") or a in succ or b in pred:
            continue
        ca, cb = chain[a], chain[b]
        if ca == cb:
            continue
        competitors = sorted(out_c[a])[1:2] + sorted(in_c[b])[1:2]
        if any(x < c["cost"] + cfg.link_margin for x in competitors):
            records.append(dict(c, decision="abstained", reason="competing_link"))
            continue
        fa = set().union(*(frames_of[m] for m in members[ca]))
        fb = set().union(*(frames_of[m] for m in members[cb]))
        if fa & fb:
            records.append(dict(c, decision="rejected", reason="simultaneous_overlap"))
            continue
        if cfg.use_team:
            ta = {by_id[m].team for m in members[ca] if by_id[m].team}
            tb = {by_id[m].team for m in members[cb] if by_id[m].team}
            if len(ta | tb) > 1:
                records.append(dict(c, decision="rejected", reason="chain_team_conflict"))
                continue
        succ[a], pred[b] = b, a
        for m in members[cb]:
            chain[m] = ca
        members[ca].extend(members.pop(cb))
        records.append(dict(c, decision="linked"))
    return chain, records


CONFIDENCE_FEATURES = ("bias", "log_chain_frames", "log_tracklet_frames", "team_known", "team_confidence",
                       "log1p_frames_to_link", "near_interaction", "log1p_frames_to_interaction",
                       "log1p_frames_to_tracklet_boundary", "local_team_agreement", "log1p_local_team_votes")


# M6.1 risk features. Most confirmed-but-wrong M6 boxes sit on a local track after an interaction in which the
# tracker passed the identity to another player. An interaction episode is "verified" only when the
# tracklet and all its partners have known, opposite teams (a cross-team swap would then show up as a team
# change); otherwise it is an unverified swap opportunity. A chain is cut into segments at unverified episodes
# and at tracklet boundaries; a box in the chain's dominant segment is more likely to carry the chain's
# majority identity.
CONFIDENCE_FEATURES_M61_EXTRA = ("segment_share", "log1p_chain_segments", "in_longest_segment",
                                 "team_evidence_available", "log1p_unverified_episodes")


def _segment_features(tracklets: list["Tracklet"], chains: dict[int, list["Tracklet"]],
                      partners: dict[tuple[int, int], list[int]],
                      keeps: dict[int, list[tuple[int, int]]] | None = None,
                      origin: dict[int, int] | None = None) -> dict[tuple[int, int], tuple[float, ...]]:
    tl_of = {(o.frame, o.index): t for t in tracklets for o in t.obs}
    team_available = float(any(o.team_vote in TEAM_LABELS for t in tracklets for o in t.obs))
    cuts: dict[int, list[int]] = {}
    unverified: dict[int, int] = {}
    for t in tracklets:
        eps: list[list[int]] = []
        for j, o in enumerate(t.obs):
            if (o.frame, o.index) in partners:
                if eps and o.frame - t.obs[eps[-1][-1]].frame <= 2:
                    eps[-1].append(j)
                else:
                    eps.append([j])
        cuts[t.tracklet_id] = []
        for ep in eps:
            mates = {tl_of[(t.obs[j].frame, p)].tracklet_id: tl_of[(t.obs[j].frame, p)] for j in ep
                     for p in partners[(t.obs[j].frame, t.obs[j].index)] if (t.obs[j].frame, p) in tl_of}
            verified = t.team in TEAM_LABELS and bool(mates) and all(
                m.team in TEAM_LABELS and m.team != t.team for m in mates.values())
            if not verified and keeps:  # M6.2: the contact decision said "keep" with high confidence
                f0, f1 = t.obs[ep[0]].frame, t.obs[ep[-1]].frame
                verified = any(a <= f1 and f0 <= b
                               for a, b in keeps.get((origin or {}).get(t.local_track_id, t.local_track_id), ()))
            if not verified:
                cuts[t.tracklet_id].append(ep[len(ep) // 2])
        unverified[t.tracklet_id] = len(cuts[t.tracklet_id])
    out: dict[tuple[int, int], tuple[float, ...]] = {}
    for ts in chains.values():
        segs: list[list[Observation]] = []
        seg_tl: list[int] = []
        for t in sorted(ts, key=lambda t: t.start):
            bounds = [0] + [c for c in cuts[t.tracklet_id] if c > 0] + [len(t.obs)]
            for a, b in zip(bounds, bounds[1:]):
                if b > a:
                    segs.append(t.obs[a:b]); seg_tl.append(t.tracklet_id)
        total = sum(len(s) for s in segs)
        longest = max(len(s) for s in segs)
        for s, tid in zip(segs, seg_tl):
            x = (len(s) / total, math.log1p(len(segs)), float(len(s) == longest), team_available,
                 math.log1p(unverified[tid]))
            for o in s:
                out[(o.frame, o.index)] = x
    return out


CONTACT_FEATURES = ("log_r_keep", "motion_pref_swap", "has_mate_motion", "log_gap", "team_flip", "team_same_known",
                    "team_mate_swap", "log_d_keep", "app_pref_swap", "has_app", "log_dur", "min_sep", "abs_log_h", "n_partners")


def contact_episodes(tracks: dict[int, list[Observation]], partners: dict[tuple[int, int], list[int]],
                     cfg: IdentityConfig) -> list[dict[str, Any]]:
    """Interaction episodes of each local track with inference-only evidence for a keep / swap decision.

    An episode is a run of boxes close to another box (gaps <= 2 frames). Evidence compares the isolated boxes
    ``contact_window`` before and after it: constant-velocity motion residual for the same track (keep) and for the
    contact partner (swap), team votes, colour-descriptor distance and episode geometry. No reference identity is read.
    """
    K, M = cfg.contact_window, cfg.contact_min_side
    by_key = {(o.frame, o.index): o for seq in tracks.values() for o in seq}
    tl_of = {k: o.local_track_id for k, o in by_key.items()}
    out: list[dict[str, Any]] = []

    def team(seq: list[Observation]) -> str | None:
        return aggregate_votes([o.team_vote for o in seq], cfg.team_min_votes, cfg.team_min_share).label if len(seq) >= M else None

    def med(seq: list[Observation]):
        d = [o.appearance for o in seq if o.appearance is not None]
        return np.median(np.array(d, float), 0) if len(d) >= 3 else None

    for tid, seq in tracks.items():
        near = [(o.frame, o.index) in partners for o in seq]
        eps: list[list[int]] = []
        cur: list[int] | None = None
        for j, nr in enumerate(near):
            if nr:
                if cur is not None and seq[j].frame - seq[cur[-1]].frame <= 2:
                    cur.append(j)
                else:
                    cur = [j]
                    eps.append(cur)
        for ep in eps:
            s, e = ep[0], ep[-1]
            pre = [o for o in seq[max(0, s - 3 * K):s] if (o.frame, o.index) not in partners][-K:]
            post = [o for o in seq[e + 1:e + 1 + 3 * K] if (o.frame, o.index) not in partners][:K]
            rec: dict[str, Any] = {"local_track_id": tid, "start": seq[s].frame, "end": seq[e].frame,
                                   "cut_frame": seq[ep[len(ep) // 2]].frame, "decidable": len(pre) >= M and len(post) >= M}
            if not rec["decidable"]:
                out.append(rec)
                continue
            gap = post[0].frame - pre[-1].frame
            pc: Counter = Counter()
            for j in ep:
                for p in partners[(seq[j].frame, seq[j].index)]:
                    pc[tl_of[(seq[j].frame, p)]] += 1
            mate = pc.most_common(1)[0][0] if pc else None
            mseq = tracks.get(mate, []) if mate is not None else []
            mpre = [o for o in mseq if o.frame < seq[s].frame and (o.frame, o.index) not in partners][-K:]
            mpost = [o for o in mseq if o.frame > seq[e].frame and (o.frame, o.index) not in partners][:K]
            h = max(1.0, float(np.median([_height(o.box) for o in pre[-5:]])))
            v = _velocity([(o.frame, _foot(o.box)) for o in pre], cfg.velocity_frames - 1)
            r_keep = float(np.linalg.norm(_foot(post[0].box) - (_foot(pre[-1].box) + v * gap)) / h / motion_spread(gap, cfg))
            r_swap = None
            if mpost and mpre:
                gm = mpost[0].frame - pre[-1].frame
                r_swap = float(np.linalg.norm(_foot(mpost[0].box) - (_foot(pre[-1].box) + v * gm)) / h / motion_spread(gm, cfg))
            tp, tq, tm = team(pre), team(post), team(mpost) if len(mpost) >= M else None
            ap_, aq, am = med(pre), med(post), med(mpost)
            d_keep = float(np.linalg.norm(ap_ - aq)) if ap_ is not None and aq is not None else None
            d_swap = float(np.linalg.norm(ap_ - am)) if ap_ is not None and am is not None else None
            seps = [float(np.linalg.norm(_foot(seq[j].box) - _foot(by_key[(seq[j].frame, p)].box)) / h)
                    for j in ep for p in partners[(seq[j].frame, seq[j].index)] if (seq[j].frame, p) in by_key]
            rec.update({"gap": gap, "dur": seq[e].frame - seq[s].frame + 1, "mate": mate, "r_keep": r_keep, "r_swap": r_swap,
                        "team_pre": tp, "team_post": tq, "mate_team_post": tm, "d_keep": d_keep, "d_swap": d_swap,
                        "min_sep": min(seps) if seps else None,
                        "h_ratio": float(np.median([_height(o.box) for o in post[:5]]) / h), "n_partners": len(pc)})
            out.append(rec)
    return out


def contact_feature_vector(r: dict[str, Any]) -> np.ndarray:
    """Raw evidence vector (CONTACT_FEATURES order) of a decidable episode record."""
    rk, rs = r["r_keep"], r["r_swap"]
    tp, tq, tm = r["team_pre"], r["team_post"], r["mate_team_post"]
    both = tp in TEAM_LABELS and tq in TEAM_LABELS
    dk, ds = r["d_keep"], r["d_swap"]
    f = (float(np.log(rk + 0.25)),
         float(-np.log((rs + 0.25) / (rk + 0.25))) if rs is not None else 0.0,
         float(rs is not None),
         float(np.log(max(1, r["gap"]))),
         float(both and tp != tq),
         float(both and tp == tq),
         float(both and tm in TEAM_LABELS and tp != tq and tm == tp),
         float(np.log(dk + 1e-3)) if dk is not None else 0.0,
         float(np.log((dk + 1e-3) / (ds + 1e-3))) if dk is not None and ds is not None else 0.0,
         float(dk is not None),
         float(np.log(r["dur"])),
         min(r["min_sep"] if r["min_sep"] is not None else 1.0, 1.5),
         abs(float(np.log(r["h_ratio"]))),
         float(r["n_partners"] > 1))
    return np.array(f, float)


def decide_contacts(episodes: list[dict[str, Any]], cfg: IdentityConfig) -> None:
    """Annotate episodes in place with ``p_swap`` and ``decision`` (keep | swap | ambiguous).

    Conservative by construction: only a low p(swap) verifies an episode, only a high p(swap) splits the track, everything
    else (including episodes without enough isolated boxes on either side) stays ambiguous.
    """
    w = np.array(cfg.contact_weights, float) if cfg.contact_weights is not None else None
    for r in episodes:
        r["p_swap"], r["decision"] = None, "ambiguous"
        if w is None or not r["decidable"]:
            continue
        p = _sigmoid(float(w[0] + w[1:] @ contact_feature_vector(r)))
        r["p_swap"] = round(p, 4)
        r["decision"] = "keep" if p <= cfg.contact_keep_below else "swap" if p >= cfg.contact_swap_above else "ambiguous"


def _apply_contact_splits(tracks: dict[int, list[Observation]], episodes: list[dict[str, Any]]
                          ) -> tuple[dict[int, list[Observation]], frozenset[tuple[int, int]], dict[int, int]]:
    """Split local tracks after every ``swap`` episode. Returns (tracks, blocked link pairs, new id -> original id)."""
    cuts: dict[int, list[int]] = defaultdict(list)
    for r in episodes:
        if r.get("decision") == "swap":
            cuts[r["local_track_id"]].append(r["cut_frame"])
    if not cuts:
        return tracks, frozenset(), {}
    nxt = max(tracks) + 1
    out: dict[int, list[Observation]] = {}
    blocked: set[tuple[int, int]] = set()
    origin: dict[int, int] = {}
    for tid, seq in tracks.items():
        if tid not in cuts:
            out[tid] = seq
            continue
        piece: list[Observation] = []
        ids = [tid]
        bounds = sorted(set(cuts[tid]))
        k = 0
        for o in seq:
            while k < len(bounds) and o.frame > bounds[k]:
                out[ids[-1]] = piece
                piece = []
                ids.append(nxt)
                origin[nxt] = tid
                blocked.add((ids[-2], ids[-1]))
                nxt += 1
                k += 1
            piece.append(o if ids[-1] == tid else
                         Observation(o.frame, o.index, ids[-1], o.box, o.detected_class, o.team_vote, o.appearance))
        out[ids[-1]] = piece
    return out, frozenset(blocked), origin


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def solve(observations: Iterable[Observation], cfg: IdentityConfig,
          contact_decisions: dict[tuple[int, int, int], str] | None = None) -> dict[str, Any]:
    """Offline identity solution for a batch of observations.

    Returns ``{"boxes": {(frame, index): info}, "tracklets": [...], "splits": [...], "links": [...], "contacts": [...]}``.
    ``info`` holds tracklet_id, global_id, identity_status, identity_confidence, team fields and the
    confidence feature vector.

    M6.2: with ``cfg.contact_mode == "evidence"`` every interaction episode of a local track gets a keep / swap /
    ambiguous decision (``contacts`` records the evidence, p(swap) and decision). ``swap`` splits the local track and blocks
    the pieces from being linked back together; ``keep`` marks the episode verified for the segment confidence features;
    ``ambiguous`` is left as in M6.1. ``contact_decisions`` ({(local track, start frame, end frame): decision}) overrides the
    evidence decisions - an experiment hook (e.g. oracle headroom), never used by the default path.
    """
    cfg.validate_association()
    obs = sorted(observations, key=lambda o: (o.frame, o.index))
    tracks: dict[int, list[Observation]] = defaultdict(list)
    by_frame: dict[int, list[Observation]] = defaultdict(list)
    for o in obs:
        tracks[o.local_track_id].append(o)
        by_frame[o.frame].append(o)
    contact_on = cfg.contact_mode != "off" or contact_decisions is not None
    if cfg.contact_mode not in ("off", "evidence"):
        raise ValueError(f"unknown contact_mode {cfg.contact_mode!r}")
    if cfg.confidence_features == "m61" or contact_on:
        partners = interaction_partners(by_frame, cfg)
        inter = {k: True for k in partners}
    elif cfg.confidence_features == "m6":
        partners, inter = {}, interactions(by_frame, cfg)
    else:
        raise ValueError(f"unknown confidence_features {cfg.confidence_features!r}")
    if cfg.confidence_features not in ("m6", "m61"):
        raise ValueError(f"unknown confidence_features {cfg.confidence_features!r}")
    contacts: list[dict[str, Any]] = []
    blocked: frozenset[tuple[int, int]] = frozenset()
    origin: dict[int, int] = {}
    keeps: dict[int, list[tuple[int, int]]] = defaultdict(list)
    if contact_on:
        contacts = contact_episodes(tracks, partners, cfg)
        decide_contacts(contacts, cfg)
        for r in contacts:
            ov = (contact_decisions or {}).get((r["local_track_id"], r["start"], r["end"]))
            if ov is not None:
                r["decision"], r["source"] = ov, "external"
            else:
                r["source"] = "evidence"
            if r["decision"] == "keep":
                keeps[r["local_track_id"]].append((r["start"], r["end"]))
        tracks, blocked, origin = _apply_contact_splits(tracks, contacts)
    if cfg.team_isolated_only:
        # a torso crop overlapping another person mixes both kits: keep team votes of isolated boxes only
        for tid, seq in tracks.items():
            tracks[tid] = [o if not inter.get((o.frame, o.index)) or o.team_vote is None else
                           Observation(o.frame, o.index, o.local_track_id, o.box, o.detected_class, None, o.appearance) for o in seq]
    tracklets, split_records = split_tracks(tracks, inter, cfg)
    forced = repair_splits(tracklets, split_records, cfg)
    if cfg.association_mode == "deferred":
        # M6.3-A: the forced split repairs are not applied; their pieces are ordinary candidates of the learned association
        from .association import associate

        chain, link_records = associate(tracklets, cfg, inter, blocked)
    else:
        chain, link_records = link_tracklets(tracklets, cfg, forced, blocked)
    by_id = {t.tracklet_id: t for t in tracklets}
    # tracklets whose global id rests on a learned association link are never auto-trusted (see the confidence loop)
    assoc_touched = {r[end] for r in link_records if r["decision"] == "linked" and r.get("source") == "association"
                     for end in ("from", "to")}
    # B: probability of the association link(s) a tracklet rests on (the weakest one); A is the local box confidence, C is
    # unvalidated for every chain containing such a link
    assoc_p: dict[int, float] = {}
    for r in link_records:
        if r["decision"] == "linked" and r.get("source") == "association":
            for end in ("from", "to"):
                assoc_p[r[end]] = min(assoc_p.get(r[end], 1.0), float(r["p"]))
    # chain -> global id, numbered by first appearance
    chains: dict[int, list[Tracklet]] = defaultdict(list)
    for t in tracklets:
        chains[chain[t.tracklet_id]].append(t)
    order = sorted(chains, key=lambda c: (min(t.start for t in chains[c]), c))
    gid_of_chain = {c: n + 1 for n, c in enumerate(order)}
    linked = [r for r in link_records if r["decision"] == "linked"]
    link_frames: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for r in linked:
        c = chain[r["from"]]
        link_frames[c].append((by_id[r["from"]].end, r["cost"]))
        link_frames[c].append((by_id[r["to"]].start, r["cost"]))
    n_feat = len(CONFIDENCE_FEATURES) + (len(CONFIDENCE_FEATURES_M61_EXTRA) if cfg.confidence_features == "m61" else 0)
    if cfg.calibration is not None and len(cfg.calibration) != n_feat:
        raise ValueError(f"calibration has {len(cfg.calibration)} coefficients, feature set "
                         f"{cfg.confidence_features!r} has {n_feat}")
    seg_x = _segment_features(tracklets, chains, partners, keeps, origin) if cfg.confidence_features == "m61" else {}
    boxes: dict[tuple[int, int], dict[str, Any]] = {}
    for c, ts in chains.items():
        cframes = sum(len(t.obs) for t in ts)
        teams = Counter()
        for t in ts:
            if t.team:
                teams[t.team] += t.team_votes
        cteam = teams.most_common(1)[0][0] if teams else None
        cconf = max((t.team_confidence for t in ts if t.team == cteam), default=0.0)
        lf = np.array(sorted(f for f, _ in link_frames.get(c, [])), float)
        for t in ts:
            frames = np.array([o.frame for o in t.obs], float)
            near = np.array([inter.get((o.frame, o.index), False) for o in t.obs])
            inter_frames = frames[near]
            # local team agreement with the chain team over +-W observations of the tracklet
            agree = np.array([o.team_vote == cteam for o in t.obs], float) if cteam else np.zeros(len(t.obs))
            other = np.array([o.team_vote in TEAM_LABELS and o.team_vote != cteam for o in t.obs], float) if cteam \
                else np.zeros(len(t.obs))
            ca, co = np.concatenate([[0], np.cumsum(agree)]), np.concatenate([[0], np.cumsum(other)])
            W = cfg.split_window
            for j, o in enumerate(t.obs):
                lo, hi = max(0, j - W), min(len(t.obs), j + W + 1)
                na, no = ca[hi] - ca[lo], co[hi] - co[lo]
                local = na / (na + no) if na + no else 0.5
                dlink = float(np.abs(lf - o.frame).min()) if len(lf) else 10_000.0
                dint = float(np.abs(inter_frames - o.frame).min()) if len(inter_frames) else 10_000.0
                dbound = float(min(o.frame - t.start, t.end - o.frame))
                x = (1.0, math.log(cframes), math.log(len(t.obs)), float(cteam is not None), cconf,
                     math.log1p(dlink), float(near[j]), math.log1p(dint), math.log1p(dbound), local,
                     math.log1p(na + no)) + seg_x.get((o.frame, o.index), ())
                if cfg.calibration is not None:
                    conf = _sigmoid(sum(w * v for w, v in zip(cfg.calibration, x)))
                else:  # uncalibrated heuristic until coefficients fitted on development data are supplied
                    conf = min(0.99, 0.5 + 0.05 * math.log(cframes) + 0.2 * (local - 0.5) - 0.1 * near[j])
                conf_uncapped = conf
                capped = t.tracklet_id in assoc_touched
                if capped and cfg.association_cap == "multiply":  # discount < confirm_threshold (validated in the config): such a box cannot become 'confirmed'
                    conf = conf * cfg.effective_association_discount
                if t.ambiguous:
                    status = "ambiguous"
                elif len(t.obs) < cfg.min_tracklet_frames:
                    status = "unknown"
                elif conf >= cfg.confirm_threshold:
                    status = "confirmed"
                elif conf >= cfg.provisional_threshold:
                    status = "provisional"
                else:
                    status = "unknown"
                if capped and status == "confirmed":  # association_cap == "status": local confidence stays, global trust is withheld
                    status = "provisional"
                boxes[(o.frame, o.index)] = {
                    "tracklet_id": t.tracklet_id, "global_id": gid_of_chain[c], "local_track_id": o.local_track_id,
                    "identity_status": status, "identity_confidence": round(conf, 4),
                    "team_cluster": cteam, "team_confidence": round(cconf, 3), "features": x}
                if cfg.association_mode == "deferred":
                    boxes[(o.frame, o.index)]["association"] = "linked_unvalidated" if capped else "none"
                    boxes[(o.frame, o.index)]["identity_confidence_uncapped"] = round(conf_uncapped, 4)  # A: local, unaffected by links
                    if capped:
                        boxes[(o.frame, o.index)]["association_link_confidence"] = round(assoc_p[t.tracklet_id], 4)  # B
    tl = [{"tracklet_id": t.tracklet_id, "local_track_id": t.local_track_id, "start": t.start, "end": t.end,
           "frames": len(t.obs), "ambiguous": t.ambiguous, "team": t.team, "team_confidence": round(t.team_confidence, 3),
           "team_votes": t.team_votes, "goalkeeper": t.goalkeeper, "role": t.role, "global_id": gid_of_chain[chain[t.tracklet_id]]}
          for t in tracklets]
    return {"boxes": boxes, "tracklets": tl, "splits": split_records, "links": link_records, "contacts": contacts}


def solve_online(observations: Iterable[Observation], cfg: IdentityConfig) -> dict[str, Any]:
    """Causal sliding-window identity with fixed latency.

    Every ``online_step`` frames the batch solver is re-run on the last ``online_window`` frames.
    Frames older than ``now - online_latency`` are frozen with the identities of the solve that first
    covered them at that latency; new chains are mapped onto already emitted global IDs by the
    number of frozen boxes they share (one-to-one), otherwise they get fresh IDs.
    """
    from scipy.optimize import linear_sum_assignment

    if cfg.association_mode != "m61":
        raise ValueError("association_mode='deferred' is offline-only (candidate gaps exceed the online window)")
    obs = sorted(observations, key=lambda o: (o.frame, o.index))
    if not obs:
        return {"boxes": {}, "solves": 0}
    first, last = obs[0].frame, obs[-1].frame
    frozen: dict[tuple[int, int], dict[str, Any]] = {}
    next_gid = 1
    solves = 0
    emitted_upto = first - 1
    now = first
    while emitted_upto < last:
        now = min(now + cfg.online_step, last + cfg.online_latency)
        horizon = min(now, last)
        win = [o for o in obs if horizon - cfg.online_window < o.frame <= horizon]
        res = solve(win, cfg)
        solves += 1
        boxes = res["boxes"]
        # map this solve's global ids onto frozen ids via shared frozen boxes
        local_ids = sorted({b["global_id"] for b in boxes.values()})
        prev_ids = sorted({frozen[k]["global_id"] for k in boxes if k in frozen})
        mapping: dict[int, int] = {}
        if prev_ids and local_ids:
            li = {g: i for i, g in enumerate(local_ids)}
            pi = {g: i for i, g in enumerate(prev_ids)}
            m = np.zeros((len(local_ids), len(prev_ids)))
            for k, b in boxes.items():
                if k in frozen:
                    m[li[b["global_id"]], pi[frozen[k]["global_id"]]] += 1
            for i, j in zip(*linear_sum_assignment(-m)):
                if m[i, j] > 0:
                    mapping[local_ids[i]] = prev_ids[j]
        emit_to = min(now - cfg.online_latency, last)
        # an emitted id may not appear twice in a frame: only reuse ids not active elsewhere in the frame
        for k in sorted(k for k in boxes if emitted_upto < k[0] <= emit_to):
            b = dict(boxes[k])
            g = b["global_id"]
            if g not in mapping:
                mapping[g] = next_gid
                next_gid += 1
            b["global_id"] = mapping[g]
            frozen[k] = b
        # ensure per-frame uniqueness after the mapping (ids from different solves can collide)
        for f in range(emitted_upto + 1, emit_to + 1):
            seen: set[int] = set()
            for k in sorted(k for k in frozen if k[0] == f):
                if frozen[k]["global_id"] in seen:
                    frozen[k]["global_id"] = next_gid
                    frozen[k]["identity_status"] = "unknown"
                    next_gid += 1
                seen.add(frozen[k]["global_id"])
        emitted_upto = max(emitted_upto, emit_to)
    return {"boxes": frozen, "solves": solves}


def interpolate_gaps(boxes: dict[tuple[int, int], dict[str, Any]], frame_boxes: dict[tuple[int, int], Sequence[float]],
                     max_gap: int = 12) -> dict[int, list[dict[str, Any]]]:
    """Linear in-chain gap fill for gaps of at most ``max_gap`` missing frames, flagged ``interpolated``.

    Returns ``{frame: [{global_id, bbox_xyxy_px, position_status}]}``. Only gaps between two
    non-ambiguous boxes of the same global id are filled; filled boxes are never mixed into the
    observed box list.
    """
    seq: dict[int, list[tuple[int, Sequence[float], str]]] = defaultdict(list)
    for k, b in boxes.items():
        seq[b["global_id"]].append((k[0], frame_boxes[k], b["identity_status"]))
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for g, s in seq.items():
        s.sort(key=lambda x: x[0])
        for (f1, b1, s1), (f2, b2, s2) in zip(s, s[1:]):
            if 1 < f2 - f1 <= max_gap + 1 and "ambiguous" not in (s1, s2):
                for f in range(f1 + 1, f2):
                    w = (f - f1) / (f2 - f1)
                    out[f].append({"global_id": g, "position_status": "interpolated",
                                   "bbox_xyxy_px": [round((1 - w) * a + w * b, 2) for a, b in zip(b1, b2)]})
    return dict(out)


def observations_from_records(records: Iterable[dict[str, Any]],
                              team_votes: dict[int, list[str | None]] | None = None,
                              descriptors: dict[int, Sequence[Any]] | None = None) -> list[Observation]:
    """Observations from ``frames.jsonl`` records (non-ball boxes in file order).

    ``descriptors`` ({frame: [colour descriptor | None per non-ball box]}) is only used by the M6.2 contact evidence.
    """
    out = []
    for r in records:
        f = int(r["frame_num"])
        players = [p for p in r["players"] if p.get("detected_class") != "ball" and p.get("bbox_xyxy_px")]
        votes = (team_votes or {}).get(f)
        desc = (descriptors or {}).get(f)
        for i, p in enumerate(players):
            out.append(Observation(f, i, int(p["track_id"]), tuple(float(v) for v in p["bbox_xyxy_px"]),
                                   p.get("detected_class") or "player", votes[i] if votes else None,
                                   tuple(float(v) for v in desc[i]) if desc is not None and desc[i] is not None else None))
    return out


def annotate_records(records: list[dict[str, Any]], boxes: dict[tuple[int, int], dict[str, Any]],
                     interpolated: dict[int, list[dict[str, Any]]] | None = None) -> list[dict[str, Any]]:
    """Copies of the records with identity fields; ``track_id`` becomes the global id."""
    out = []
    for r in records:
        f = int(r["frame_num"])
        players, i = [], 0
        for p in r["players"]:
            if p.get("detected_class") == "ball" or not p.get("bbox_xyxy_px"):
                players.append(p)
                continue
            b = boxes.get((f, i))
            i += 1
            if b is None:
                players.append(dict(p, local_track_id=p["track_id"], identity_status="unknown", position_status="observed"))
                continue
            entry = dict(p, track_id=b["global_id"], local_track_id=b["local_track_id"],
                         tracklet_id=b["tracklet_id"], global_id=b["global_id"],
                         identity_status=b["identity_status"], identity_confidence=b["identity_confidence"],
                         team_cluster=b["team_cluster"], team_confidence=b["team_confidence"],
                         position_status="observed")
            if "association" in b:  # M6.3-A only: whether the global id rests on a learned, not yet validated link
                entry["association_status"] = b["association"]
                entry["identity_confidence_local"] = b["identity_confidence_uncapped"]  # A: tracklet consistency, not link-dependent
                if "association_link_confidence" in b:
                    entry["association_link_confidence"] = b["association_link_confidence"]  # B: weakest incident link probability
            players.append(entry)
        rec = dict(r, players=players)
        if interpolated is not None:
            rec["identity_interpolated"] = interpolated.get(f, [])
        out.append(rec)
    return out


IDENTITY_ANALYTICS_POLICY = "m6-identity-analytics-v1"


def identity_trajectory_points(records: Iterable[dict[str, Any]], global_id: int, *, canonical_period: int,
                               statuses: Sequence[str] = ("confirmed",), change_guard_frames: int = 12):
    """Analytics trajectory of one global identity from identity-annotated ``frames.jsonl`` records.

    Policy ``IDENTITY_ANALYTICS_POLICY``: only *observed* boxes (never interpolated or predicted
    positions) with a usable pitch position, whose identity status is in ``statuses``, and that lie
    more than ``change_guard_frames`` frames away from any frame where this identity is ambiguous /
    unknown or passes from one tracklet to another (a link or split decision). Every box carries one
    global id, so per-identity selections never count a box twice. Returns ``(points, coverage)``.
    """
    if canonical_period not in (1, 2, 3, 4):
        raise ValueError("canonical_period must be explicitly mapped to 1, 2, 3, or 4.")
    from football_fatigue.analytics.movement import TrajectoryPoint

    rows = []
    for r in records:
        for p in r["players"]:
            if p.get("global_id") == global_id:
                rows.append((int(r["frame_num"]), float(r["timestamp_s"]), p))
    rows.sort(key=lambda x: x[0])
    risky = [f for f, _, p in rows if p.get("identity_status") in ("ambiguous", "unknown")]
    risky += [f for (f0, _, p0), (f, _, p) in zip(rows, rows[1:]) if p.get("tracklet_id") != p0.get("tracklet_id")
              for f in (f0, f)]
    risky_arr = np.array(sorted(set(risky)), float)
    cov = Counter({"observed_boxes": 0, "used": 0})
    points = []
    for f, ts, p in rows:
        cov["observed_boxes"] += 1
        if p.get("position_status", "observed") != "observed":
            cov["excluded_not_observed"] += 1
        elif p.get("identity_status") not in statuses:
            cov[f"excluded_status_{p.get('identity_status')}"] += 1
        elif len(risky_arr) and np.abs(risky_arr - f).min() <= change_guard_frames:
            cov["excluded_near_identity_change"] += 1
        elif not p.get("position_usable") or p.get("x") is None or p.get("y") is None:
            cov["excluded_no_pitch_position"] += 1
        else:
            cov["used"] += 1
            points.append(TrajectoryPoint(frame_num=f, video_time_s=ts, canonical_period=canonical_period,
                                          x=float(p["x"]), y=float(p["y"]), is_estimated=True,
                                          coord_source="cv_homography"))
    coverage = {"policy": IDENTITY_ANALYTICS_POLICY, "global_id": global_id, **dict(cov),
                "used_share": cov["used"] / cov["observed_boxes"] if cov["observed_boxes"] else 0.0}
    return points, coverage


def write_json(path, data: Any) -> None:
    with open(path, "x", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, default=str)


def _sha256(path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    """Opt-in identity pass over an existing tracking run; writes a new run directory, never in place."""
    import argparse
    import time
    from pathlib import Path

    from .team_appearance import extract_box_features, fit_balanced_team_model, fit_team_model, gated_team_votes

    ap = argparse.ArgumentParser(prog="python -m football_fatigue.vision.identity", description=main.__doc__)
    ap.add_argument("run_dir", type=Path, help="tracking run directory containing frames.jsonl")
    ap.add_argument("--output", type=Path, required=True, help="new run directory (must not exist)")
    ap.add_argument("--video", type=Path, help="source video, for per-video team appearance evidence")
    ap.add_argument("--config", type=Path, help="IdentityConfig JSON (e.g. the frozen M6 configuration)")
    ap.add_argument("--online", action="store_true", help="causal sliding-window mode with fixed latency")
    ap.add_argument("--team-method", choices=("m6", "balanced"), default="m6",
                    help="m6: 2-means + balance gate (default); balanced: M6.1 balance-selected k (use with the M6.1 config)")
    ap.add_argument("--interpolate-max-gap", type=int, default=0,
                    help="add flagged in-identity interpolated boxes for gaps up to N frames (separate field)")
    a = ap.parse_args(argv)
    cfg = IdentityConfig.from_dict(json.loads(a.config.read_text())) if a.config else IdentityConfig()
    frames_path = a.run_dir / "frames.jsonl"
    with frames_path.open(encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh]
    boxes_by_frame = {int(r["frame_num"]): [p["bbox_xyxy_px"] for p in r["players"]
                                            if p.get("detected_class") != "ball" and p.get("bbox_xyxy_px")] for r in records}
    votes, team_info = None, {"used": False}
    t0 = time.perf_counter()
    if a.video is not None and cfg.use_team:
        feats, _ = extract_box_features(str(a.video), boxes_by_frame)
        per_track: dict[int, list[np.ndarray]] = defaultdict(list)
        for r in records:
            f = int(r["frame_num"])
            players = [p for p in r["players"] if p.get("detected_class") != "ball" and p.get("bbox_xyxy_px")]
            for i, p in enumerate(players):
                if feats.get(f) and feats[f][i] is not None:
                    per_track[int(p["track_id"])].append(feats[f][i])
        if a.team_method == "balanced":
            model, votes, sel = fit_balanced_team_model(per_track, feats)
            team_info = {"used": votes is not None, "method": "balanced", "selection": sel,
                         "diagnostics": model.diagnostics if model else None}
        else:
            model = fit_team_model(per_track, method="kmeans2_box")
            if model is not None:
                votes, check = gated_team_votes({f: [model.vote(x) for x in fs] for f, fs in feats.items()})
                team_info = {"used": votes is not None, "method": "kmeans2_box", "diagnostics": model.diagnostics,
                             "balance_check": check}
    t_team = time.perf_counter() - t0
    # colour descriptors feed only the optional learned association (default path: unchanged observations)
    obs = observations_from_records(records, votes, feats if cfg.association_mode == "deferred" and a.video is not None
                                    and cfg.use_team else None)
    t0 = time.perf_counter()
    res = solve_online(obs, cfg) if a.online else solve(obs, cfg)
    t_solve = time.perf_counter() - t0
    interp = None
    if a.interpolate_max_gap > 0:
        fb = {}
        for o in obs:
            fb[(o.frame, o.index)] = o.box
        interp = interpolate_gaps(res["boxes"], fb, a.interpolate_max_gap)
    a.output.mkdir(parents=True, exist_ok=False)
    with (a.output / "frames.jsonl").open("x", encoding="utf-8") as fh:
        for r in annotate_records(records, res["boxes"], interp):
            fh.write(json.dumps(r) + "\n")
    if not a.online:
        write_json(a.output / "identity_graph.json", {k: res[k] for k in ("tracklets", "splits", "links")})
    statuses = Counter(b["identity_status"] for b in res["boxes"].values())
    n_frames = max(1, len(records))
    write_json(a.output / "manifest.json", {
        "identity_schema_version": IDENTITY_SCHEMA_VERSION, "mode": "online" if a.online else "offline",
        "source_frames": str(frames_path), "source_frames_sha256": _sha256(frames_path),
        "video": str(a.video) if a.video else None, "config": cfg.to_dict(), "team_evidence": team_info,
        "boxes": len(res["boxes"]), "global_ids": len({b["global_id"] for b in res["boxes"].values()}),
        "status_counts": dict(statuses), "interpolate_max_gap": a.interpolate_max_gap,
        "timing_s": {"team_features_and_model": round(t_team, 3), "identity_solve": round(t_solve, 3),
                     "identity_solve_ms_per_frame": round(1000 * t_solve / n_frames, 3)},
        "note": "track_id holds the global id; local_track_id the tracker id. Identity evidence, not jersey identity.",
    })
    print(f"wrote {a.output} ({len(res['boxes'])} boxes, {dict(statuses)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
