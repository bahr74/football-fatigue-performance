"""Real-first player identity matching benchmark over RAW FIFA encounters (M11 step 2, task A: identity-hidden association).

Input is a verified M11 step-1 catalogue (``m11-encounters-v2``): the RAW cache, the non-destructive quality mask and the
pairwise encounters at ``tau_m``. The tracking file is not re-read, and no observation is interpolated, removed or corrected.

**Events.** A pairwise encounter is a proximity event, not an occlusion. Encounters that share a participant (a pair member or a
third player within tau) and overlap or lie less than ``merge_gap_s`` apart within one segment are merged into one *event*: one
proximity episode, possibly with more than two players. An event has an ambiguity interval (the first to the last close frame
of its encounters) and a before and an after window of ``before_s`` and ``after_s``, cut at the segment edges (a period
boundary or an unobserved gap).

**Independence.** Events are taken in time order, and an event is rejected (``overlaps_earlier_instance``) when one of its
participants' windows overlaps a window already used by an earlier instance. No observation of a player therefore serves two
instances, and repeated contacts never become duplicate examples. Episodes longer than ``max_ambiguity_s``, such as sustained
marking, are rejected rather than treated as one encounter.

**Instances** (matcher-visible) hold, for each participant with enough observations:
- an *incoming* tracklet (the before window);
- an *outgoing* tracklet (the after window), under anonymous IDs, each list in its own seeded random order;
- the RAW observations of the ambiguity interval as unlabelled point sets, shuffled again in every frame.

Times are relative to the start of the ambiguity interval. No jersey, player, team, slot, side, frame number, period, match or
absolute time is included.

**Task B** (:func:`withhold_interval`) uses the same instances, IDs and order with every ambiguity-interval observation removed.
It is a controlled information-loss experiment on real trajectories, not a real occlusion.

**References** (evaluation-only, a separate file) hold the provider identity of every tracklet and the expected outgoing tracklet
for each incoming one. Provider identity is a reference label, not verified ground truth. The tiers describe how trustworthy that
reference is; they do not establish ground truth.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from football_fatigue.encounters import quality as Qm
from football_fatigue.encounters.catalogue import code_digest as encounters_code_digest
from football_fatigue.encounters.catalogue import peak_memory_mb, segments
from football_fatigue.encounters.quality import M11Config
from football_fatigue.encounters.trajectories import STATE_AVAILABLE, VISIBILITY, RawCache, load_cache
from football_fatigue.ingestion import Dataset, EventVersion, IngestionError, approved_output_root, safe_output_path

SCHEMA_VERSION = OUTPUT_DIR = "m11-matching-v3"   # v3: + task B and FOOTBALLTrace E0; v2 (task A) and v1 are kept unchanged
SOURCE_SCHEMA = SOURCE_DIR = "m11-encounters-v2"
TASK = "A_identity_hidden_association"
TASK_B = "B_withheld_interval_association"
EVENT_VERSION = EventVersion.ROOT
TIERS = ("high_confidence_reference", "lower_confidence_reference", "natural_suspicious_unverified")
REJECTIONS = ("too_many_participants", "ambiguity_too_long", "substitution_window", "side_overcount", "attribution_unresolved_or_ambiguous",
              "fewer_than_two_incoming", "no_outgoing", "overlaps_earlier_instance")
LOWER_REASONS = ("missing_incoming_or_outgoing", "segment_edge", "low_coverage", "estimated_observations", "implausible_jump",
                 "frame_wide_shift")
SOURCE_SUFFIXES = (".cache.npz", ".quality.npz", ".encounters.jsonl", ".manifest.json")
OUTPUT_SUFFIXES = (".config.json", ".eligibility.jsonl", ".instances.jsonl", ".instances_b.jsonl", ".references.jsonl", ".predictions.jsonl",
                   ".predictions_b.jsonl", ".footballtrace_literal.jsonl", ".summary.json", ".manifest.json")
MATCHER_VISIBLE = (".instances.jsonl", ".instances_b.jsonl")
EVALUATION_ONLY = (".references.jsonl", ".eligibility.jsonl", ".footballtrace_literal.jsonl")
# the complete matcher-visible schema; anything else in an instance is a leak (tests enforce it)
INSTANCE_KEYS = frozenset({"instance_id", "schema", "task", "units", "ambiguity_interval_s", "incoming", "outgoing", "during"})
TRACKLET_KEYS = frozenset({"tracklet_id", "t", "x", "y", "visibility", "confidence"})
DURING_KEYS = frozenset({"t", "points"})
UNITS = dict(t="s, relative to the start of the ambiguity interval", x="m (RAW pitch coordinates)", y="m (RAW pitch coordinates)",
             visibility="1 VISIBLE, 2 ESTIMATED, 0 unknown", confidence="1 LOW, 2 MEDIUM, 3 HIGH, 0 unknown",
             during_point="[x, y, visibility, confidence]")


@dataclass(frozen=True)
class MatchingConfig:
    tau_m: float = 2.0                   # default encounter threshold; 1 m and 3 m are later sensitivity checks only
    before_s: float = 2.0                # incoming window (= step-1 context)
    after_s: float = 2.0                 # outgoing window
    merge_gap_s: float = 1.0             # re-contact within this gap continues one proximity episode
    max_participants: int = 6
    max_ambiguity_s: float = 6.0         # longer proximity episodes (e.g. sustained marking) are not one encounter
    min_tracklet_s: float = 0.3          # a tracklet needs this much observed time (available samples) in its window
    high_min_coverage: float = 0.9       # high-confidence tier: share of each window observed, for every participant
    high_max_estimated_share: float = 0.1
    seed: str = "m11-matching-v1"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _r(v: float | None, n: int = 4) -> float | None:
    return None if v is None or not np.isfinite(v) else round(float(v), n)


def _slot_info(c: RawCache, s: int) -> dict[str, Any]:
    return dict(slot=s, side=str(c.slot_side[s]), jersey=int(c.slot_jersey[s]),
                player_id=None if c.slot_player_id[s] < 0 else int(c.slot_player_id[s]),
                team_id=None if c.slot_team_id[s] < 0 else int(c.slot_team_id[s]))


def shuffle_order(seed: str, match_id: object, key: str, purpose: str, n: int) -> list[int]:
    """Deterministic pseudo-random permutation, independent per instance and purpose, unrelated to slot order."""
    h = hashlib.sha256(f"{seed}|{match_id}|{key}|{purpose}".encode()).digest()
    return [int(i) for i in np.random.default_rng(int.from_bytes(h[:8], "little")).permutation(n)]


def tracklet_velocity(tr: dict[str, list], *, at_end: bool, span_s: float, max_speed_m_s: float) -> tuple[np.ndarray, int]:
    """Least-squares velocity (m/s) over a tracklet's update instants in its last (``at_end``) or first ``span_s``.

    An update instant is a sample whose position differs from the previous sample of the same tracklet, or the tracklet's first
    sample, so repeated 15 Hz values do not dilute the slope and a held position gives zero velocity. Returns ``(v, n_used)``."""
    t, x, y = (np.asarray(tr[k], float) for k in ("t", "x", "y"))
    if len(t) == 0:
        return np.zeros(2), 0
    upd = np.r_[True, (np.diff(x) != 0) | (np.diff(y) != 0)]
    sel = upd & ((t >= t[-1] - span_s) if at_end else (t <= t[0] + span_s))
    n = int(sel.sum())
    if n < 2:
        return np.zeros(2), n
    tt = t[sel] - t[sel].mean()
    v = np.array([(tt * (x[sel] - x[sel].mean())).sum(), (tt * (y[sel] - y[sel].mean())).sum()]) / (tt ** 2).sum()
    sp = float(np.hypot(*v))
    if sp > max_speed_m_s:
        v *= max_speed_m_s / sp
    return v, n


# ---------------------------------------------------------------- events

def _participants(e: dict) -> set[int]:
    return {p["slot"] for p in e["pair"]} | {t["slot"] for t in e["third_players"]}


def group_events(c: RawCache, encounters: list[dict], m11: M11Config, cfg: MatchingConfig) -> tuple[list[dict], list[tuple[int, int]]]:
    """Merge the pairwise encounters at ``cfg.tau_m`` into events (see the module docstring)."""
    segs = segments(c, m11)
    starts = np.array([a for a, _ in segs], dtype=np.int64)
    t = c.video_time_s
    items = []
    for e in encounters:
        if e["tau_m"] != cfg.tau_m:
            continue
        s, end = e["rows"]["during"]
        items.append(dict(seg=int(np.searchsorted(starts, s, "right") - 1), s=int(s), e=int(end), slots=_participants(e),
                          core={p["slot"] for p in e["pair"]}, members=[e]))
    while True:
        parent = list(range(len(items)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        by_slot: dict[tuple[int, int], list[int]] = {}
        for i, g in enumerate(items):
            for sl in g["slots"]:
                by_slot.setdefault((g["seg"], sl), []).append(i)
        merged = False
        for idx in by_slot.values():
            idx.sort(key=lambda i: items[i]["s"])
            run_end, run_i = -np.inf, -1
            for i in idx:
                if run_i >= 0 and t[items[i]["s"]] - run_end < cfg.merge_gap_s:
                    a, b = find(i), find(run_i)
                    if a != b:
                        parent[a] = b
                        merged = True
                end_t = t[items[i]["e"] - 1]
                if end_t > run_end:
                    run_end, run_i = end_t, i
        if not merged:
            break
        groups: dict[int, list[dict]] = {}
        for i, g in enumerate(items):
            groups.setdefault(find(i), []).append(g)
        items = [dict(seg=gs[0]["seg"], s=min(g["s"] for g in gs), e=max(g["e"] for g in gs), slots=set().union(*(g["slots"] for g in gs)),
                      core=set().union(*(g["core"] for g in gs)), members=[m for g in gs for m in g["members"]]) for gs in groups.values()]
    for g in items:
        g["members"].sort(key=lambda e: e["encounter_id"])
    items.sort(key=lambda g: (g["s"], g["e"], sorted(g["slots"])))
    return items, segs


def _window(c: RawCache, mask: np.ndarray, rows: slice, p: int, nominal_s: float) -> dict[str, Any]:
    av = c.state[rows, p] == STATE_AVAILABLE
    upd = av & ((mask[rows, p] & Qm.UPDATE_INSTANT) != 0)
    return dict(rows=[rows.start, rows.stop], available=int(av.sum()), updates=int(upd.sum()),
                coverage=_r(min(1.0, av.sum() / (nominal_s * c.fps))))


def assess_event(c: RawCache, mask: np.ndarray, g: dict, segs: list[tuple[int, int]], cfg: MatchingConfig) -> dict[str, Any]:
    """Eligibility audit of one event: rejection reasons, or a reference-quality tier and the reasons it is not higher."""
    t = c.video_time_s
    r0, r1 = segs[g["seg"]]
    s, e = g["s"], g["e"]
    b0 = max(r0, int(np.searchsorted(t, t[s] - cfg.before_s - 1e-9, "left")))
    a1 = min(r1, int(np.searchsorted(t, t[e - 1] + cfg.after_s + 1e-9, "right")))
    edge = t[s] - cfg.before_s < t[r0] - 1e-9 or t[e - 1] + cfg.after_s > t[r1 - 1] + 1e-9
    min_frames = cfg.min_tracklet_s * c.fps
    parts, bits_all = [], 0
    for p in sorted(g["slots"]):
        bits = int(np.bitwise_or.reduce(mask[b0:a1, p])) if a1 > b0 else 0
        bits_all |= bits
        bef, aft = _window(c, mask, slice(b0, s), p, cfg.before_s), _window(c, mask, slice(e, a1), p, cfg.after_s)
        av = c.state[b0:a1, p] == STATE_AVAILABLE
        est = av & (c.visibility[b0:a1, p] == VISIBILITY["ESTIMATED"])
        parts.append(dict(**_slot_info(c, p), core=p in g["core"], before=bef, after=aft,
                          incoming=bef["available"] >= min_frames, outgoing=aft["available"] >= min_frames,
                          estimated_share=_r(est.sum() / av.sum()) if av.any() else None,
                          mask_flags=[k for k, v in Qm.MASK_BITS.items() if bits & v]))
    rejected = []
    if len(parts) > cfg.max_participants:
        rejected.append("too_many_participants")
    if t[e - 1] - t[s] > cfg.max_ambiguity_s:
        rejected.append("ambiguity_too_long")
    if bits_all & Qm.SUBSTITUTION_WINDOW:
        rejected.append("substitution_window")
    if bits_all & Qm.SIDE_OVERCOUNT:
        rejected.append("side_overcount")
    if bits_all & (Qm.ATTRIBUTION_UNRESOLVED | Qm.AMBIGUOUS_RAW):
        rejected.append("attribution_unresolved_or_ambiguous")
    if sum(p["incoming"] for p in parts) < 2:
        rejected.append("fewer_than_two_incoming")
    if not any(p["outgoing"] for p in parts):
        rejected.append("no_outgoing")
    lower = []
    if not all(p["incoming"] and p["outgoing"] for p in parts):
        lower.append("missing_incoming_or_outgoing")
    if edge:
        lower.append("segment_edge")
    if any(min(p["before"]["coverage"] or 0.0, p["after"]["coverage"] or 0.0) < cfg.high_min_coverage for p in parts):
        lower.append("low_coverage")
    if any(p["estimated_share"] is None or p["estimated_share"] > cfg.high_max_estimated_share for p in parts):
        lower.append("estimated_observations")
    if bits_all & Qm.IMPLAUSIBLE_JUMP:
        lower.append("implausible_jump")
    if bits_all & Qm.FRAME_WIDE_SHIFT:
        lower.append("frame_wide_shift")
    if rejected:
        tier = None
    elif bits_all & Qm.DISCONTINUITY_CANDIDATE:
        tier = "natural_suspicious_unverified"
    else:
        tier = "lower_confidence_reference" if lower else "high_confidence_reference"
    m = g["members"]
    return dict(
        match_id=c.match_id, period=int(c.canonical_period[s]), accepted=tier is not None, tier=tier, rejection_reasons=rejected,
        primary_rejection=rejected[0] if rejected else None, below_high_reasons=lower,
        start_frame=int(c.frame_num[s]), end_frame=int(c.frame_num[e - 1]), start_time_s=_r(t[s]), end_time_s=_r(t[e - 1]),
        rows=dict(before=[b0, s], during=[s, e], after=[e, a1]), segment_edge=bool(edge),
        n_participants=len(parts), n_core=len(g["core"]), participants=parts,
        encounter_ids=[x["encounter_id"] for x in m], step1_categories=dict(Counter(x["category"] for x in m)),
        min_distance_m=min(x["min_distance_m"] for x in m), discontinuity_evidence=sorted({j for x in m for j in x["discontinuity_evidence"]}))


# ---------------------------------------------------------------- instances and references

def _tracklet(c: RawCache, rows: slice, p: int, t_ref: float, tid: str) -> dict[str, Any]:
    r = np.arange(rows.start, rows.stop)
    r = r[c.state[r, p] == STATE_AVAILABLE]
    return dict(tracklet_id=tid, t=[round(float(v), 4) for v in c.video_time_s[r] - t_ref], x=c.x[r, p].tolist(), y=c.y[r, p].tolist(),
                visibility=c.visibility[r, p].astype(int).tolist(), confidence=c.confidence[r, p].astype(int).tolist())


def build_instance(c: RawCache, row: dict, cfg: MatchingConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Matcher-visible instance and evaluation-only reference of one accepted event."""
    (b0, s), (_, e), (_, a1) = row["rows"]["before"], row["rows"]["during"], row["rows"]["after"]
    slots = [p["slot"] for p in row["participants"]]
    key = f"{row['start_frame']}|{row['end_frame']}|{','.join(map(str, slots))}"
    iid = hashlib.sha256(f"{cfg.seed}|{c.match_id}|{key}".encode()).hexdigest()[:16]
    inc = [p["slot"] for p in row["participants"] if p["incoming"]]
    out = [p["slot"] for p in row["participants"] if p["outgoing"]]
    in_ids = {inc[k]: f"in-{i}" for i, k in enumerate(shuffle_order(cfg.seed, c.match_id, key, "incoming", len(inc)))}
    out_ids = {out[k]: f"out-{i}" for i, k in enumerate(shuffle_order(cfg.seed, c.match_id, key, "outgoing", len(out)))}
    t = c.video_time_s
    t_ref = float(t[s])
    during_t, points = [], []
    for r in range(s, e):
        av = [p for p in slots if c.state[r, p] == STATE_AVAILABLE]
        order = shuffle_order(cfg.seed, c.match_id, key, f"during-{r - s}", len(av))
        during_t.append(round(float(t[r] - t_ref), 4))
        points.append([[float(c.x[r, av[k]]), float(c.y[r, av[k]]), int(c.visibility[r, av[k]]), int(c.confidence[r, av[k]])] for k in order])
    instance = dict(
        instance_id=iid, schema=SCHEMA_VERSION, task=TASK, units=UNITS, ambiguity_interval_s=[0.0, round(float(t[e - 1] - t_ref), 4)],
        incoming=[_tracklet(c, slice(b0, s), p, t_ref, in_ids[p]) for p in sorted(in_ids, key=in_ids.get)],
        outgoing=[_tracklet(c, slice(e, a1), p, t_ref, out_ids[p]) for p in sorted(out_ids, key=out_ids.get)],
        during=dict(t=during_t, points=points))
    by_slot = {p["slot"]: p for p in row["participants"]}
    expected = {in_ids[p]: out_ids.get(p) for p in sorted(in_ids, key=in_ids.get)}
    core_in = [p for p in in_ids if by_slot[p]["core"]]
    if len(in_ids) == 2 and len(core_in) == 2:
        relation = "same_team" if c.slot_side[core_in[0]] == c.slot_side[core_in[1]] else "opposite_team"
    else:
        relation = "multi_player"
    reference = dict(
        instance_id=iid, schema=SCHEMA_VERSION, match_id=c.match_id, period=row["period"], tier=row["tier"],
        below_high_reasons=row["below_high_reasons"], start_frame=row["start_frame"], end_frame=row["end_frame"],
        start_time_s=row["start_time_s"], end_time_s=row["end_time_s"], rows=row["rows"], encounter_ids=row["encounter_ids"],
        step1_categories=row["step1_categories"], discontinuity_evidence=row["discontinuity_evidence"],
        incoming={in_ids[p]: dict(_slot_info(c, p), core=by_slot[p]["core"]) for p in sorted(in_ids, key=in_ids.get)},
        outgoing={out_ids[p]: dict(_slot_info(c, p), core=by_slot[p]["core"]) for p in sorted(out_ids, key=out_ids.get)},
        expected=expected, without_incoming=[_slot_info(c, p) for p in slots if p not in in_ids],
        without_outgoing=[_slot_info(c, p) for p in slots if p not in out_ids], n_participants=len(slots), team_relation=relation,
        difficulty=difficulty(instance, expected, row))
    return instance, reference


def difficulty(instance: dict, expected: dict[str, str | None], row: dict, *, span_s: float = 0.5, turn_deg: float = 45.0,
               long_s: float = 3.0, moving_m_s: float = 1.0) -> dict[str, Any]:
    """Algorithm-independent motion difficulty from the reference mapping (evaluation-side; never used for selection)."""
    inc = {tr["tracklet_id"]: tr for tr in instance["incoming"]}
    out = {tr["tracklet_id"]: tr for tr in instance["outgoing"]}
    pairs = [(inc[i], out[o]) for i, o in expected.items() if o is not None]
    turns, speeds = [], []
    for a, b in pairs:
        va, _ = tracklet_velocity(a, at_end=True, span_s=span_s, max_speed_m_s=12.0)
        vb, _ = tracklet_velocity(b, at_end=False, span_s=span_s, max_speed_m_s=12.0)
        sa, sb = float(np.hypot(*va)), float(np.hypot(*vb))
        speeds.append((sa + sb) / 2)
        if sa >= moving_m_s and sb >= moving_m_s:
            turns.append(float(np.degrees(np.arccos(np.clip(va @ vb / (sa * sb), -1.0, 1.0)))))
    crossing = False
    for k in range(len(pairs)):
        for m in range(k + 1, len(pairs)):
            (a1, b1), (a2, b2) = pairs[k], pairs[m]
            rel_in = np.array([a1["x"][-1] - a2["x"][-1], a1["y"][-1] - a2["y"][-1]])
            rel_out = np.array([b1["x"][0] - b2["x"][0], b1["y"][0] - b2["y"][0]])
            crossing |= bool(rel_in @ rel_out < 0)
    dur = instance["ambiguity_interval_s"][1]
    reasons = [k for k, v in (("crossing", crossing), ("sharp_turn", bool(turns) and max(turns) > turn_deg),
                              ("multi_player", len(instance["incoming"]) > 2), ("long_interval", dur > long_s)) if v]
    return dict(label="challenging" if reasons else "standard", reasons=reasons, ambiguity_s=dur, min_distance_m=row["min_distance_m"],
                max_heading_change_deg=_r(max(turns), 1) if turns else None, mean_speed_m_s=_r(float(np.mean(speeds)), 2) if speeds else None,
                crossing=crossing)


def withhold_interval(instance: dict) -> dict:
    """Task B view of a task A instance: the same incoming and outgoing tracklets, IDs and order, with every observation of
    the ambiguity interval removed. A controlled information-loss experiment, not a real occlusion. The input is not modified."""
    return dict(instance, task=TASK_B, during=dict(t=[], points=[]))


def build_benchmark(c: RawCache, mask: np.ndarray, encounters: list[dict], m11: M11Config = M11Config(),
                    cfg: MatchingConfig = MatchingConfig()) -> tuple[list[dict], list[dict], list[dict]]:
    """(eligibility rows for every event, matcher-visible instances, evaluation-only references). Inputs are not modified."""
    events, segs = group_events(c, encounters, m11, cfg)
    rows, instances, references = [], [], []
    claimed: dict[int, list[tuple[int, int]]] = {}        # slot -> row spans already used by an earlier instance
    for g in events:                                      # time order; the choice never depends on any matcher
        row = assess_event(c, mask, g, segs, cfg)
        row["instance_id"] = None
        b0, a1 = row["rows"]["before"][0], row["rows"]["after"][1]
        if row["accepted"] and any(b0 < y1 and y0 < a1 for p in row["participants"] for y0, y1 in claimed.get(p["slot"], ())):
            row.update(accepted=False, tier=None, rejection_reasons=["overlaps_earlier_instance"], primary_rejection="overlaps_earlier_instance")
        if row["accepted"]:
            for p in row["participants"]:
                claimed.setdefault(p["slot"], []).append((b0, a1))
            inst, ref = build_instance(c, row, cfg)
            row["instance_id"] = inst["instance_id"]
            instances.append(inst)
            references.append(ref)
        rows.append(row)
    for i, row in enumerate(rows):
        row["event_index"] = i
    order = sorted(range(len(instances)), key=lambda k: instances[k]["instance_id"])    # file order carries no time information
    return rows, [instances[k] for k in order], [references[k] for k in order]


def audit(rows: list[dict], references: list[dict], encounters: list[dict], cfg: MatchingConfig) -> dict[str, Any]:
    cand = [e for e in encounters if e["tau_m"] == cfg.tau_m]
    by_eid = {eid: r for r in rows for eid in r["encounter_ids"]}
    trusted = [e for e in cand if e["category"] == "trusted_reference_candidate"]
    acc = [r for r in rows if r["accepted"]]
    rej = [r for r in rows if not r["accepted"]]

    def stats(refs: list[dict]) -> dict[str, Any]:
        return dict(
            instances=len(refs), incoming_tracklets=sum(len(r["incoming"]) for r in refs), outgoing_tracklets=sum(len(r["outgoing"]) for r in refs),
            expected_absent_outgoing=sum(v is None for r in refs for v in r["expected"].values()),
            participants=dict(sorted(Counter(str(r["n_participants"]) for r in refs).items())),
            team_relation=dict(Counter(r["team_relation"] for r in refs)), difficulty=dict(Counter(r["difficulty"]["label"] for r in refs)),
            difficulty_reasons=dict(Counter(x for r in refs for x in r["difficulty"]["reasons"])),
            ambiguity_s_p50=_r(float(np.median([r["difficulty"]["ambiguity_s"] for r in refs]))) if refs else None)

    return dict(
        tau_m=cfg.tau_m, candidate_encounters=len(cand), candidate_encounters_by_step1_category=dict(Counter(e["category"] for e in cand)),
        events=len(rows), events_by_participants=dict(sorted(Counter(str(r["n_participants"]) for r in rows).items(), key=lambda kv: int(kv[0]))),
        accepted=len(acc), accepted_by_tier={k: sum(r["tier"] == k for r in acc) for k in TIERS}, rejected=len(rej),
        rejected_primary={k: sum(r["primary_rejection"] == k for r in rej) for k in REJECTIONS},
        rejected_any={k: sum(k in r["rejection_reasons"] for r in rej) for k in REJECTIONS},
        lower_tier_reasons={k: sum(k in r["below_high_reasons"] for r in acc if r["tier"] == "lower_confidence_reference") for k in LOWER_REASONS},
        step1_trusted_encounters=len(trusted),
        step1_trusted_by_event_outcome=dict(Counter((by_eid[e["encounter_id"]]["tier"] or "rejected") for e in trusted)),
        instances={k: stats([r for r in references if r["tier"] == k]) for k in TIERS} | {"all": stats(references)})


# ---------------------------------------------------------------- source, outputs, run

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_source(raw_root: str | Path, match_id: object, source_root: str | Path | None = None) -> tuple[RawCache, np.ndarray, list[dict], dict, dict]:
    """Read a step-1 catalogue after checking every file against its manifest and the manifest against the dataset."""
    base = Path(source_root) if source_root is not None else approved_output_root(raw_root) / SOURCE_DIR
    paths = {sfx: safe_output_path(raw_root, match_id, EVENT_VERSION, base, suffix=sfx) for sfx in SOURCE_SUFFIXES}
    m = json.loads(paths[".manifest.json"].read_text(encoding="utf-8"))
    if m.get("schema") != SOURCE_SCHEMA or str(m.get("match_id")) != str(match_id):
        raise IngestionError(f"{paths['.manifest.json']} is not a {SOURCE_SCHEMA} catalogue of match {match_id}")
    if m.get("coordinate_policy") != "raw" or m.get("provider_smoothed_coordinates_used") is not False:
        raise IngestionError("the source catalogue is not RAW-only")
    digests = {}
    for sfx in SOURCE_SUFFIXES[:-1]:
        d = _sha256(paths[sfx])
        if m.get("outputs_sha256", {}).get(paths[sfx].name) != d:
            raise IngestionError(f"{paths[sfx]} does not match the checksum in its manifest; not used")
        digests[paths[sfx].name] = d
    src, ref = Dataset(raw_root).tracking_source(str(match_id))
    if ref.path != m["tracking_source"]["path"] or src.stat().st_size != m["tracking_source"]["bytes"]:
        raise IngestionError(f"the catalogue was built from {m['tracking_source']}, which is no longer the dataset's tracking source")
    c = load_cache(paths[".cache.npz"])
    with np.load(paths[".quality.npz"]) as z:
        mask = z["mask"]
    with open(paths[".encounters.jsonl"], encoding="utf-8") as f:
        encounters = [json.loads(line) for line in f]
    provenance = dict(directory=str(base / str(match_id)), schema=m["schema"], sha256=digests, manifest_sha256=_sha256(paths[".manifest.json"]),
                      code_sha256=m.get("code_sha256"), derived_from_cache=m.get("derived_from_cache"), tracking_source=m["tracking_source"])
    return c, mask, encounters, m, provenance


def output_paths(raw_root: str | Path, match_id: object, out_root: str | Path | None = None) -> dict[str, Path]:
    base = Path(out_root) if out_root is not None else approved_output_root(raw_root) / OUTPUT_DIR
    return {sfx: safe_output_path(raw_root, match_id, EVENT_VERSION, base, suffix=sfx) for sfx in OUTPUT_SUFFIXES}


def jsonl(rows: list[dict]) -> bytes:
    return "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows).encode()


def write_outputs(raw_root: str | Path, match_id: object, blobs: dict[str, bytes], manifest: dict, out_root: str | Path | None = None) -> dict[str, Path]:
    """Write every output or none. Existing outputs are never replaced."""
    paths = output_paths(raw_root, match_id, out_root)
    existing = [str(p) for p in paths.values() if p.exists()]
    if existing:
        raise IngestionError(f"outputs already exist and are never overwritten: {existing}")
    next(iter(paths.values())).parent.mkdir(parents=True, exist_ok=True)
    paths = output_paths(raw_root, match_id, out_root)
    base = paths[".manifest.json"].parent.parent
    tmp = {sfx: safe_output_path(raw_root, match_id, EVENT_VERSION, base, suffix=sfx + ".tmp") for sfx in OUTPUT_SUFFIXES}
    digests = {}
    try:
        for sfx in OUTPUT_SUFFIXES[:-1]:
            with open(tmp[sfx], "xb") as f:
                f.write(blobs[sfx])
            digests[paths[sfx].name] = hashlib.sha256(blobs[sfx]).hexdigest()
        with open(tmp[".manifest.json"], "xb") as f:
            f.write(json.dumps(dict(manifest, outputs_sha256=digests), indent=1, sort_keys=True).encode())
        for sfx in OUTPUT_SUFFIXES:
            if paths[sfx].exists():
                raise IngestionError(f"output appeared while writing; not overwritten: {paths[sfx]}")
            os.rename(tmp[sfx], paths[sfx])
    finally:
        for p in tmp.values():
            if p.exists():
                p.unlink()
    return paths


def code_digest() -> dict[str, str]:
    here = Path(__file__).parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(here.glob("*.py"))}


def run_match(raw_root: str | Path, match_id: object, *, source_root: str | Path | None = None, out_root: str | Path | None = None,
              cfg: MatchingConfig = MatchingConfig(), baseline_cfg=None, split: dict[str, Any] | None = None) -> tuple[dict[str, Path], dict[str, Any]]:
    from .baselines import BASELINES, BaselineConfig, run_baselines
    from .evaluation import evaluate
    from .footballtrace import FootballTraceConfig, protocol_l, summarize_protocol_l

    bc = baseline_cfg or BaselineConfig()
    ft = FootballTraceConfig()
    t0 = time.perf_counter()
    output_paths(raw_root, match_id, out_root)                                   # fail on an unsafe destination before any work
    c, mask, encounters, src_manifest, provenance = load_source(raw_root, match_id, source_root)
    if cfg.tau_m not in src_manifest["config"]["taus_m"]:
        raise IngestionError(f"the source catalogue has no encounters at tau = {cfg.tau_m} m")
    m11 = M11Config(max_gap_s=src_manifest["config"]["max_gap_s"])
    rows, instances, references = build_benchmark(c, mask, encounters, m11, cfg)
    t_build = time.perf_counter() - t0
    inst_bytes = jsonl(instances)
    inst_b_bytes = jsonl([withhold_interval(i) for i in instances])
    t1 = time.perf_counter()
    # every method sees exactly the written matcher-visible file of its task, and nothing else
    predictions = run_baselines([json.loads(x) for x in inst_bytes.decode().splitlines()], bc)
    predictions_b = run_baselines([json.loads(x) for x in inst_b_bytes.decode().splitlines()], bc)
    t_base = time.perf_counter() - t1
    literal = protocol_l(c, references, ft)
    summary = dict(schema=SCHEMA_VERSION, match_id=c.match_id, tasks=[TASK, TASK_B], audit=audit(rows, references, encounters, cfg), baselines=list(BASELINES),
                   results=evaluate(predictions, references), results_b=evaluate(predictions_b, references),
                   footballtrace_literal=summarize_protocol_l(literal),
                   note="all rates are agreement with the provider identity reference, not verified identity accuracy")
    config = dict(schema=SCHEMA_VERSION, tasks=[TASK, TASK_B], matching=cfg.as_dict(), baselines=bc.as_dict(), footballtrace=ft.as_dict(),
                  source_step1_config=src_manifest["config"],
                  tiers=list(TIERS), rejections=list(REJECTIONS), lower_reasons=list(LOWER_REASONS), units=UNITS,
                  matcher_visible_files=[f"root{s}" for s in MATCHER_VISIBLE], evaluation_only_files=[f"root{s}" for s in EVALUATION_ONLY],
                  instance_schema=dict(instance=sorted(INSTANCE_KEYS), tracklet=sorted(TRACKLET_KEYS), during=sorted(DURING_KEYS)))
    blobs = {".config.json": json.dumps(config, indent=1, sort_keys=True).encode(), ".eligibility.jsonl": jsonl(rows),
             ".instances.jsonl": inst_bytes, ".instances_b.jsonl": inst_b_bytes, ".references.jsonl": jsonl(references),
             ".predictions.jsonl": jsonl(predictions), ".predictions_b.jsonl": jsonl(predictions_b), ".footballtrace_literal.jsonl": jsonl(literal),
             ".summary.json": json.dumps(summary, indent=1, sort_keys=True).encode()}
    manifest = dict(schema=SCHEMA_VERSION, match_id=c.match_id, tasks=[TASK, TASK_B], coordinate_policy="raw", provider_smoothed_coordinates_used=False,
                    synthetic_swaps_injected=False, split=split, source=provenance, config=config,
                    code_sha256=dict(matching=code_digest(), encounters=encounters_code_digest()),
                    processing=dict(source="step1_catalogue", build_s=round(t_build, 1), baselines_s=round(t_base, 1),
                                    seconds_before_write=round(time.perf_counter() - t0, 1), peak_memory_mb_before_write=peak_memory_mb()),
                    python=platform.python_version(), numpy=np.__version__)
    return write_outputs(raw_root, c.match_id, blobs, manifest, out_root), summary
