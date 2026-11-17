"""Four-segment identity samples from unique intersection events (advisor-directed M11 pipeline).

For an event between players P and Q with traversal times t_P, t_Q at the intersection:

- ``t_lo = min(t_P, t_Q)``, ``t_hi = max(t_P, t_Q)``, ``t_mid = (t_lo + t_hi) / 2``;
- **withheld interval** ``[t_lo, t_hi]`` (+ ``withheld_pad_s`` on each side): neither player's observations are model input;
- **incoming** span ``[t_mid - half_span_s, t_lo - pad)`` and **outgoing** span ``(t_hi + pad, t_mid + half_span_s]``, the same for
  both players, so time ranges and nominal lengths say nothing about who is who.

Model-facing instance (task-B-compatible schema, so the historical feature code can be applied unchanged): ``in-0`` is Player A
(the query), ``in-1`` Player B, ``out-0`` / ``out-1`` the two anonymous outgoing tracklets. Each tracklet holds every RAW frame
with a position in its span: ``t`` (s, relative to ``t_lo``), ``x``, ``y`` (m), ``visibility``, ``confidence``. Which player is A
and the order of the outgoing tracklets come from two independent bits of a seeded hash of the event key, so the label is
balanced and unrelated to slot numbers, time or file order. Instances carry an opaque ``instance_id`` and no slot, side,
jersey, player, team, event type or time-difference field.

The label (``target``: index of A's continuation among the outgoing tracklets) is provider trajectory continuity: the outgoing
tracklet cut from the same provider slot as A. It and all identities are written only to the separate references.

An event yields no sample when the four tracklets cannot be extracted reliably; the reason is recorded (:data:`EXCLUSIONS`).
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from football_fatigue.encounters import quality as Q
from football_fatigue.encounters.trajectories import RawCache

from .detect import Timeline, duplicate_positions

SCHEMA = "m11-intersections-v1"
TASK = "intersection_four_segment"
EXCLUSIONS = ("period_boundary", "segment_span_too_short", "attribution_unresolved_or_ambiguous", "duplicate_position_artifact",
              "substitution_window", "low_coverage", "internal_gap", "too_few_samples")
INSTANCE_KEYS = ("instance_id", "schema", "task", "units", "ambiguity_interval_s", "query", "incoming", "outgoing", "during")
TRACKLET_KEYS = ("tracklet_id", "t", "x", "y", "visibility", "confidence")


@dataclass(frozen=True)
class SampleConfig:
    half_span_s: float = 2.5         # segments lie within +-2.5 s of the event midpoint (the advisor's window)
    withheld_pad_s: float = 0.0      # extra withheld time on each side of [t_lo, t_hi]
    min_segment_s: float = 1.0       # nominal span of each segment
    min_coverage: float = 0.8        # share of the span's frames with a RAW position
    max_internal_gap_s: float = 0.5  # same continuity bound as the Step 1 steps
    min_samples: int = 10
    seed: str = "m11-intersections-v1"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bits(seed: str, key: str) -> tuple[int, int]:
    h = hashlib.sha256(f"{seed}|{key}".encode()).digest()
    return h[0] & 1, h[1] & 1


def _tracklet(c: RawCache, rows: np.ndarray, slot: int, t_ref: float, tid: str) -> dict[str, Any]:
    ok = rows[np.isfinite(c.x[rows, slot])]
    return dict(tracklet_id=tid, t=[round(float(v), 4) for v in c.video_time_s[ok] - t_ref], x=c.x[ok, slot].tolist(), y=c.y[ok, slot].tolist(),
                visibility=c.visibility[ok, slot].astype(int).tolist(), confidence=c.confidence[ok, slot].astype(int).tolist())


def _check(c: RawCache, rows: np.ndarray, slot: int, cfg: SampleConfig) -> str | None:
    ok = np.isfinite(c.x[rows, slot])
    if ok.sum() < cfg.min_samples:
        return "too_few_samples"
    if ok.mean() < cfg.min_coverage:
        return "low_coverage"
    tt = c.video_time_s[rows[ok]]
    edges = np.r_[c.video_time_s[rows[0]], tt, c.video_time_s[rows[-1]]]
    if np.diff(edges).max() > cfg.max_internal_gap_s:
        return "internal_gap"
    return None


def build_sample(c: RawCache, mask: np.ndarray, ev: dict, tl: Timeline, dup: np.ndarray,
                 cfg: SampleConfig = SampleConfig()) -> tuple[dict | None, dict | None, dict | None]:
    """(instance, reference, None) for a valid event, or (None, None, exclusion)."""
    p, sa, sb = ev["period"], ev["slot_a"], ev["slot_b"]
    t_lo, t_hi = min(ev["time_a_s"], ev["time_b_s"]), max(ev["time_a_s"], ev["time_b_s"])
    t_mid = (t_lo + t_hi) / 2
    w_lo, w_hi = t_lo - cfg.withheld_pad_s, t_hi + cfg.withheld_pad_s
    s_lo, s_hi = t_mid - cfg.half_span_s, t_mid + cfg.half_span_s

    def excluded(reason: str, **extra) -> tuple[None, None, dict]:
        return None, None, dict(event_id=ev["event_id"], reason=reason, period=p, type=ev["type"], abs_time_diff_s=ev["abs_time_diff_s"], **extra)

    p_lo, p_hi = tl.bounds(p)
    if s_lo < p_lo or s_hi > p_hi:
        return excluded("period_boundary")
    if min(w_lo - s_lo, s_hi - w_hi) < cfg.min_segment_s:
        return excluded("segment_span_too_short", span_s=round(min(w_lo - s_lo, s_hi - w_hi), 3))
    rows_in = tl.between(p, s_lo, w_lo, hi_open=True)
    rows_out = tl.between(p, w_hi, s_hi, lo_open=True)
    rows_all = tl.between(p, s_lo, s_hi)
    m = mask[rows_all][:, [sa, sb]]
    present = np.isfinite(c.x[rows_all][:, [sa, sb]]) | ((m & (Q.AMBIGUOUS_RAW | Q.MISSING_RAW)) != 0)
    if ((m & (Q.ATTRIBUTION_UNRESOLVED | Q.AMBIGUOUS_RAW | Q.SIDE_OVERCOUNT)) != 0)[present].any():
        return excluded("attribution_unresolved_or_ambiguous")
    if dup[rows_all][:, [sa, sb]].any():
        return excluded("duplicate_position_artifact")
    if ((m & Q.SUBSTITUTION_WINDOW) != 0).any():
        return excluded("substitution_window")
    for rows, part in ((rows_in, "incoming"), (rows_out, "outgoing")):
        for slot in (sa, sb):
            why = _check(c, rows, slot, cfg)
            if why:
                return excluded(why, segment=part)

    a_first, swap = _bits(cfg.seed, ev["event_key"])
    pa, pb = (sa, sb) if a_first else (sb, sa)                       # Player A (query) and Player B
    o0, o1 = (pb, pa) if swap else (pa, pb)                          # outgoing order
    target = 0 if o0 == pa else 1
    sid = hashlib.sha256(f"{cfg.seed}|sample|{ev['event_key']}".encode()).hexdigest()[:16]
    inst = dict(instance_id=sid, schema=SCHEMA, task=TASK, units=dict(t="s relative to the start of the withheld interval", x="m", y="m"),
                ambiguity_interval_s=[0.0, round(float(w_hi - w_lo), 4)], query="in-0",
                incoming=[_tracklet(c, rows_in, pa, w_lo, "in-0"), _tracklet(c, rows_in, pb, w_lo, "in-1")],
                outgoing=[_tracklet(c, rows_out, o0, w_lo, "out-0"), _tracklet(c, rows_out, o1, w_lo, "out-1")],
                during=dict(t=[], points=[]))

    def ident(s: int) -> dict:
        return dict(slot=int(s), side=str(c.slot_side[s]), jersey=int(c.slot_jersey[s]), player_id=int(c.slot_player_id[s]),
                    team_id=int(c.slot_team_id[s]))

    est = {tid: float(np.mean(np.asarray(tr["visibility"]) == 2)) for tid, tr in
           [(tr["tracklet_id"], tr) for tr in inst["incoming"] + inst["outgoing"]]}
    withheld = tl.between(p, w_lo, w_hi)
    ref = dict(instance_id=sid, schema=SCHEMA, event_id=ev["event_id"], event_key=ev["event_key"], match_id=ev["match_id"], period=p,
               target=target, query="in-0", expected={"in-0": f"out-{target}", "in-1": f"out-{1 - target}"},
               incoming={"in-0": ident(pa), "in-1": ident(pb)}, outgoing={"out-0": ident(o0), "out-1": ident(o1)},
               team_relation=ev["team_relation"], event_type=ev["type"], time_diff_s=ev["time_diff_s"], abs_time_diff_s=ev["abs_time_diff_s"],
               simultaneous=ev["simultaneous"], min_synchronous_distance_m=ev["min_synchronous_distance_m"],
               withheld_video_time_s=[float(w_lo), float(w_hi)], withheld_frames=[int(c.frame_num[withheld[0]]), int(c.frame_num[withheld[-1]])]
               if len(withheld) else None, span_video_time_s=[float(s_lo), float(s_hi)], estimated_share=est, quality=ev["quality"])
    return inst, ref, None


def build_samples(c: RawCache, mask: np.ndarray, events: list[dict], cfg: SampleConfig = SampleConfig(), *, tl: Timeline | None = None,
                  dup: np.ndarray | None = None) -> tuple[list[dict], list[dict], list[dict]]:
    tl = tl or Timeline(c)
    dup = duplicate_positions(c) if dup is None else dup
    insts, refs, excl = [], [], []
    for ev in events:
        i, r, x = build_sample(c, mask, ev, tl, dup, cfg)
        if x is not None:
            excl.append(x)
        else:
            insts.append(i)
            refs.append(r)
    order = np.argsort([i["instance_id"] for i in insts])           # file order by opaque ID, not by time or slot
    return [insts[k] for k in order], [refs[k] for k in order], excl
