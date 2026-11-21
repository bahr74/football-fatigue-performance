"""Non-destructive quality mask, update cadence and identity-discontinuity flags (M11 step 1).

Nothing here changes, removes or relabels an observation. Every flag is an extra bit next to the untouched raw value.

* Update cadence: the provider repeats raw positions (about 15 Hz in 29.97 fps files). A sample that exactly equals the
  same slot's sample in the immediately preceding frame is ``REPEAT_OF_PREVIOUS``; the first sample of each run of equal
  values is an ``UPDATE_INSTANT``. Runs longer than two frames are ``HELD_POSITION`` (a stationary player or a provider
  freeze: informational, never invalid).
* Motion is computed only between consecutive update instants of one slot, with their real timestamps, inside one period
  and only when the slot is unobserved for at most ``max_gap_s`` in between.
* A step is an ``IMPLAUSIBLE_JUMP`` when it moves more than ``jump_min_m`` faster than ``max_speed_m_s``. The speed of a
  jump uses the shortest interval the change can have taken (``change_dt``), so a hold followed by a jump is not diluted.
  Jumps are listed with full evidence and excluded from velocity and acceleration statistics; the samples stay.
* A step shorter than ``jump_min_m`` that still exceeds ``max_speed_m_s`` is an ``OVERSPEED_STEP`` (sub-metre noise): kept,
  flagged and excluded from velocity statistics, but not treated as a jump.
* When at least ``frame_wide_share`` of the observed slots (and ``frame_wide_min_slots``) jump in the same frame, those jumps
  are a ``FRAME_WIDE_SHIFT`` (a provider-wide displacement, not one player's identity).
* A jump *lands on another slot* when it ends within ``min(transfer_radius_m, distance / 2)`` of another slot's position just
  before the jump. Landing on another slot after a jump longer than ``transfer_min_m`` is a ``LABEL_TRANSFER_CANDIDATE``; two
  jumps of any length landing on each other's previous positions are a ``RECIPROCAL_EXCHANGE``.
* ``DISCONTINUITY_CANDIDATE`` marks the jumps that are evidence of a possible identity discontinuity: label-transfer
  candidates, reciprocal exchanges, and jumps longer than ``transfer_min_m`` outside a frame-wide shift. These are
  candidates, never confirmed swaps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np

from football_fatigue.ingestion import AttributionStatus, EventRow

from .trajectories import ATTRIBUTION, CONFIDENCE, STATE_ABSENT, STATE_AMBIGUOUS, STATE_AVAILABLE, STATE_MISSING, VISIBILITY, RawCache

# sample mask bits (uint32)
ESTIMATED = 1 << 0
LOW_CONFIDENCE = 1 << 1
REPEAT_OF_PREVIOUS = 1 << 2
HELD_POSITION = 1 << 3
UPDATE_INSTANT = 1 << 4
IMPLAUSIBLE_JUMP = 1 << 5            # set on the landing sample of the jump
LABEL_TRANSFER_CANDIDATE = 1 << 6
RECIPROCAL_EXCHANGE = 1 << 7
ATTRIBUTION_UNRESOLVED = 1 << 8      # quarantined, unresolved jersey, or no roster join
SUBSTITUTION_WINDOW = 1 << 9
SIDE_OVERCOUNT = 1 << 10             # more than ``max_side_players`` entries in the side's list in this frame
NO_MOTION_CONTEXT = 1 << 11          # an update instant without any valid incoming or outgoing step
AMBIGUOUS_RAW = 1 << 12
MISSING_RAW = 1 << 13
OVERSPEED_STEP = 1 << 14             # landing sample of a sub-metre step above max_speed_m_s
FRAME_WIDE_SHIFT = 1 << 15           # landing sample of a jump that is part of a frame-wide displacement
DISCONTINUITY_CANDIDATE = 1 << 16    # landing sample of a jump that is identity-discontinuity evidence
MASK_BITS = dict(ESTIMATED=ESTIMATED, LOW_CONFIDENCE=LOW_CONFIDENCE, REPEAT_OF_PREVIOUS=REPEAT_OF_PREVIOUS, HELD_POSITION=HELD_POSITION,
                 UPDATE_INSTANT=UPDATE_INSTANT, IMPLAUSIBLE_JUMP=IMPLAUSIBLE_JUMP, LABEL_TRANSFER_CANDIDATE=LABEL_TRANSFER_CANDIDATE,
                 RECIPROCAL_EXCHANGE=RECIPROCAL_EXCHANGE, ATTRIBUTION_UNRESOLVED=ATTRIBUTION_UNRESOLVED, SUBSTITUTION_WINDOW=SUBSTITUTION_WINDOW,
                 SIDE_OVERCOUNT=SIDE_OVERCOUNT, NO_MOTION_CONTEXT=NO_MOTION_CONTEXT, AMBIGUOUS_RAW=AMBIGUOUS_RAW, MISSING_RAW=MISSING_RAW,
                 OVERSPEED_STEP=OVERSPEED_STEP, FRAME_WIDE_SHIFT=FRAME_WIDE_SHIFT, DISCONTINUITY_CANDIDATE=DISCONTINUITY_CANDIDATE)

# exclusive per-sample categories, in precedence order
SAMPLE_CATEGORIES = ("identity_discontinuity_candidate", "implausible_jump", "overspeed_step", "ambiguous_or_unattributed", "missing",
                     "estimated", "insufficient_motion_context", "valid_raw")


@dataclass(frozen=True)
class M11Config:
    taus_m: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0)
    window_s: float = 2.0                # FOOTBALLTrace: 50 frames at 25 fps
    window_overlap: float = 0.5
    min_close_fraction: float = 0.5      # FOOTBALLTrace: distance <= tau for at least half of the window
    max_gap_s: float = 0.5               # unobserved time that still keeps a trajectory (and a segment) continuous
    context_s: float = 2.0               # before/after references of an encounter
    max_speed_m_s: float = 12.0          # same physiological bound as analytics.movement, used here to flag, not drop
    jump_min_m: float = 1.0
    transfer_min_m: float = 3.0
    transfer_radius_m: float = 1.0
    reciprocal_max_rows: int = 2
    frame_wide_share: float = 0.5
    frame_wide_min_slots: int = 5
    substitution_window_s: tuple[float, float] = (2.0, 5.0)   # before / after the SUB event time
    max_side_players: int = 11
    min_context_coverage: float = 0.5
    trusted_min_visible: float = 0.9
    trusted_min_coverage: float = 0.9

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QualityResult:
    mask: np.ndarray                  # (F, S) uint32
    speed: np.ndarray                 # (F, S) float32: valid step speed at the step's landing update instant, NaN otherwise
    steps: dict[str, np.ndarray]      # one entry per step between consecutive update instants of one slot
    jumps: list[dict[str, Any]]
    substitutions: list[dict[str, Any]]


def _contiguous_prev(c: RawCache) -> np.ndarray:
    """(F,) True when row r-1 is the immediately preceding source frame of the same period."""
    ok = np.zeros(c.n_frames, bool)
    if c.n_frames > 1:
        ok[1:] = (np.diff(c.frame_num) == 1) & (c.canonical_period[1:] == c.canonical_period[:-1]) & (c.canonical_period[1:] > 0)
    return ok


def cadence(c: RawCache) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(repeat, update, held) boolean (F, S) arrays."""
    avail = c.state == STATE_AVAILABLE
    rep = np.zeros_like(avail)
    prev = _contiguous_prev(c)[:, None]
    rep[1:] = avail[1:] & avail[:-1] & prev[1:] & (c.x[1:] == c.x[:-1]) & (c.y[1:] == c.y[:-1])
    upd = avail & ~rep
    held = np.zeros_like(avail)
    for s in range(c.n_slots):
        rows = np.flatnonzero(avail[:, s])
        if not len(rows):
            continue
        run = np.cumsum(upd[rows, s])                 # run id per available sample
        length = np.bincount(run)[run]
        held[rows[length >= 3], s] = True
    return rep, upd, held


def _substitutions(c: RawCache, events: Iterable[EventRow], cfg: M11Config) -> list[dict[str, Any]]:
    by_player = {int(p): s for s, p in enumerate(c.slot_player_id) if p >= 0}
    out = []
    for r in events:
        if r.game_event_type != "SUB":
            continue
        ge = r.attributes.get("gameEvents") or {}
        off, on = ge.get("playerOffId"), ge.get("playerOnId")
        out.append(dict(event_row_id=r.event_row_id, event_time_s=r.event_time_s, canonical_period=r.canonical_period,
                        player_off_id=off, player_on_id=on, slot_off=by_player.get(off), slot_on=by_player.get(on)))
    return out


def assess(c: RawCache, events: Iterable[EventRow], cfg: M11Config = M11Config()) -> QualityResult:
    F, S = c.n_frames, c.n_slots
    t = c.video_time_s
    nominal = 1.0 / c.fps
    mask = np.zeros((F, S), np.uint32)
    avail = c.state == STATE_AVAILABLE
    present = c.state != STATE_ABSENT
    mask[present & (c.visibility == VISIBILITY["ESTIMATED"])] |= ESTIMATED
    mask[present & (c.confidence == CONFIDENCE["LOW"])] |= LOW_CONFIDENCE
    mask[present & (c.attribution != ATTRIBUTION[AttributionStatus.RESOLVED])] |= ATTRIBUTION_UNRESOLVED
    mask[c.state == STATE_AMBIGUOUS] |= AMBIGUOUS_RAW
    mask[c.state == STATE_MISSING] |= MISSING_RAW
    rep, upd, held = cadence(c)
    mask[rep] |= REPEAT_OF_PREVIOUS
    mask[upd] |= UPDATE_INSTANT
    mask[held] |= HELD_POSITION
    for side in ("home", "away"):
        cols = np.flatnonzero(c.slot_side == side)
        over = np.flatnonzero(present[:, cols].sum(axis=1) > cfg.max_side_players)
        sub = mask[np.ix_(over, cols)]
        sub[present[np.ix_(over, cols)]] |= SIDE_OVERCOUNT
        mask[np.ix_(over, cols)] = sub
    subs = _substitutions(c, events, cfg)
    pre, post = cfg.substitution_window_s
    for sub in subs:
        if sub["event_time_s"] is None:
            continue
        rows = (t >= sub["event_time_s"] - pre) & (t <= sub["event_time_s"] + post) & (c.canonical_period == (sub["canonical_period"] or -1))
        for s in (sub["slot_off"], sub["slot_on"]):
            if s is not None:
                mask[rows & present[:, s], s] |= SUBSTITUTION_WINDOW

    # steps between consecutive update instants of each slot
    cols: dict[str, list] = {k: [] for k in ("slot", "row_from", "row_from_last", "row_to", "distance_m", "dt_s", "change_dt_s", "valid", "jump",
                                             "overspeed")}
    for s in range(S):
        rows = np.flatnonzero(avail[:, s])
        if len(rows) < 2:
            continue
        starts = np.flatnonzero(upd[rows, s])
        u = rows[starts]                                             # update instants
        last = rows[np.r_[starts[1:] - 1, len(rows) - 1]]            # last sample of each run
        a, b, la = u[:-1], u[1:], last[:-1]
        dist = np.hypot(c.x[b, s] - c.x[a, s], c.y[b, s] - c.y[a, s])
        dt = t[b] - t[a]
        change = np.minimum(dt, t[b] - t[la] + nominal)
        valid = (c.canonical_period[a] == c.canonical_period[b]) & (c.canonical_period[a] > 0) & (t[b] - t[la] <= cfg.max_gap_s) & (dt > 0)
        fast = valid & (dist / np.where(change > 0, change, np.inf) > cfg.max_speed_m_s)
        jump = fast & (dist > cfg.jump_min_m)
        for k, v in (("slot", np.full(len(a), s)), ("row_from", a), ("row_from_last", la), ("row_to", b), ("distance_m", dist), ("dt_s", dt),
                     ("change_dt_s", change), ("valid", valid), ("jump", jump), ("overspeed", fast & ~jump)):
            cols[k].append(v)
    steps = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in cols.items()}
    for k in ("slot", "row_from", "row_from_last", "row_to"):
        steps[k] = steps[k].astype(np.int64)
    for k in ("valid", "jump", "overspeed"):
        steps[k] = steps[k].astype(bool)
    speed = np.full((F, S), np.nan, np.float32)
    ok = steps["valid"] & ~steps["jump"] & ~steps["overspeed"]
    mask[steps["row_to"][steps["overspeed"]], steps["slot"][steps["overspeed"]]] |= OVERSPEED_STEP
    speed[steps["row_to"][ok], steps["slot"][ok]] = steps["distance_m"][ok] / steps["dt_s"][ok]
    # update instants with no valid incoming and no valid outgoing step
    has = np.zeros((F, S), bool)
    has[steps["row_to"][steps["valid"]], steps["slot"][steps["valid"]]] = True
    has[steps["row_from"][steps["valid"]], steps["slot"][steps["valid"]]] = True
    mask[upd & ~has] |= NO_MOTION_CONTEXT

    jumps = _jumps(c, steps, cfg)
    for j in jumps:
        for key, bit in (("label_transfer_candidate", LABEL_TRANSFER_CANDIDATE), ("reciprocal_exchange", RECIPROCAL_EXCHANGE),
                         ("frame_wide_shift", FRAME_WIDE_SHIFT), ("discontinuity_candidate", DISCONTINUITY_CANDIDATE)):
            if j[key]:
                mask[j["row_to"], j["slot"]] |= bit
        mask[j["row_to"], j["slot"]] |= IMPLAUSIBLE_JUMP
    return QualityResult(mask, speed, steps, jumps, subs)


def _nearest(c: RawCache, row: int, x: float, y: float, exclude: int) -> tuple[int | None, float | None]:
    d = np.hypot(c.x[row] - x, c.y[row] - y)
    d[exclude] = np.nan
    if np.all(np.isnan(d)):
        return None, None
    o = int(np.nanargmin(d))
    return o, float(d[o])


def _jumps(c: RawCache, steps: dict[str, np.ndarray], cfg: M11Config) -> list[dict[str, Any]]:
    out = []
    for i in np.flatnonzero(steps["jump"]):
        s, a, la, b = (int(steps[k][i]) for k in ("slot", "row_from", "row_from_last", "row_to"))
        x0, y0, x1, y1 = c.x[la, s], c.y[la, s], c.x[b, s], c.y[b, s]
        d = float(steps["distance_m"][i])
        partner, pd = _nearest(c, la, x1, y1, s)           # who stood where the label lands, just before the jump
        vacated_by, vd = _nearest(c, b, x0, y0, s)          # who stands at the origin when the label lands
        lands = pd is not None and pd <= min(cfg.transfer_radius_m, d / 2)
        transfer = lands and d > cfg.transfer_min_m
        out.append(dict(slot=s, side=str(c.slot_side[s]), jersey=int(c.slot_jersey[s]), period=int(c.canonical_period[b]),
                        row_from=la, row_to=b, frame_from=int(c.frame_num[la]), frame_to=int(c.frame_num[b]),
                        time_from_s=float(c.video_time_s[la]), time_to_s=float(c.video_time_s[b]),
                        x_from=float(x0), y_from=float(y0), x_to=float(x1), y_to=float(y1), distance_m=d,
                        change_dt_s=float(steps["change_dt_s"][i]), speed_m_s=d / float(steps["change_dt_s"][i]),
                        visibility_from=int(c.visibility[la, s]), visibility_to=int(c.visibility[b, s]),
                        confidence_from=int(c.confidence[la, s]), confidence_to=int(c.confidence[b, s]),
                        nearest_at_landing_slot=partner, nearest_at_landing_m=pd, nearest_at_origin_after_slot=vacated_by,
                        nearest_at_origin_after_m=vd, lands_on_other_slot=bool(lands), label_transfer_candidate=bool(transfer),
                        reciprocal_exchange=False, source_record_from=int(c.record_first[la]), source_record_to=int(c.record_first[b])))
    by = {(j["slot"], j["row_to"]): j for j in out}
    for j in out:
        if not j["lands_on_other_slot"]:
            continue
        p = j["nearest_at_landing_slot"]
        for dr in range(-cfg.reciprocal_max_rows, cfg.reciprocal_max_rows + 1):
            k = by.get((p, j["row_to"] + dr))
            if k and k["lands_on_other_slot"] and k["nearest_at_landing_slot"] == j["slot"]:
                j["reciprocal_exchange"] = k["reciprocal_exchange"] = True
    per_row = np.bincount([j["row_to"] for j in out], minlength=c.n_frames) if out else np.zeros(c.n_frames, int)
    observed = (c.state == STATE_AVAILABLE).sum(axis=1)
    for j in out:
        n = int(per_row[j["row_to"]])
        j["concurrent_jumps_in_frame"] = n
        j["frame_wide_shift"] = bool(n >= max(cfg.frame_wide_min_slots, cfg.frame_wide_share * int(observed[j["row_to"]])))
        j["discontinuity_candidate"] = bool(j["label_transfer_candidate"] or j["reciprocal_exchange"]
                                            or (j["distance_m"] > cfg.transfer_min_m and not j["frame_wide_shift"]))
    for n, j in enumerate(sorted(out, key=lambda j: (j["row_to"], j["slot"]))):
        j["jump_id"] = f"{c.match_id}-j{n:05d}"
    return sorted(out, key=lambda j: j["jump_id"])


def sample_categories(c: RawCache, mask: np.ndarray) -> np.ndarray:
    """(F, S) int8 index into SAMPLE_CATEGORIES (-1 where absent), first matching category in precedence order."""
    cat = np.full(mask.shape, -1, np.int8)
    present = c.state != STATE_ABSENT
    order = (DISCONTINUITY_CANDIDATE, IMPLAUSIBLE_JUMP, OVERSPEED_STEP, AMBIGUOUS_RAW | ATTRIBUTION_UNRESOLVED, MISSING_RAW, ESTIMATED,
             NO_MOTION_CONTEXT)
    left = present.copy()
    for i, bit in enumerate(order):
        hit = left & ((mask & bit) != 0)
        cat[hit] = i
        left &= ~hit
    cat[left] = len(order)
    return cat
