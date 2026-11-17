"""Sliding-window pairwise trajectory-intersection detection on one match (advisor-directed M11 pipeline).

Source: the checksum-verified Step 1 RAW cache and quality steps (``m11-encounters-v2``), read through
``matching.benchmark.load_source``; nothing there is written.

**Polylines.** A player (slot) is represented by its *update instants* (the provider repeats positions at ~15 Hz in 29.97 fps
files; repeats are not vertices). Consecutive update instants are joined only by a Step 1 *valid* step: same period, unobserved
for at most ``max_gap_s`` (0.5 s) in between, positive duration. Steps that are implausible jumps (> 12 m/s and > 1 m) are not
joined, because a teleport is not a path. Steps touching an unattributed, ambiguous or side-overcount sample are not joined either.
Each exclusion is counted. Coordinates are RAW metres, never interpolated, smoothed or transformed.

**Windows.** Per period, window centres advance by ``stride_s`` from ``period_start + half_window_s`` while the whole window fits
in the period. A player is available in a window when at least one of its segments lies entirely inside the window; every
unordered pair of available players is compared (a pair-window check). Bounding boxes reject far-apart pairs first.

**Deduplication.** A segment pair found in several overlapping windows is one hit (its windows are kept). Hits of one player pair
are then merged into one *event* only when they describe the same contact: adjacent segments on both sides and the same point
(within ``merge_tol_m``), or adjacent segments of a collinear overlap. Separate crossings of the same pair stay separate.

**Event types.** ``proper_crossing`` (interior point of both segments); ``vertex_crossing`` / ``touch`` / ``vertex_undetermined``
(contact at a sample vertex; crossing when the paths change sides, decided from positions ``side_probe_s`` before and after);
``collinear_overlap``. Traversal times are interpolated linearly along each segment between its update instants. The time
difference is metadata, never a filter, and no distance threshold selects events.
"""

from __future__ import annotations

import hashlib
import itertools
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from football_fatigue.encounters import quality as Q
from football_fatigue.encounters.trajectories import RawCache

from .geometry import COLLINEAR, ENDPOINT, PROPER, classify_vertex, intersect

UNUSABLE = Q.ATTRIBUTION_UNRESOLVED | Q.AMBIGUOUS_RAW | Q.SIDE_OVERCOUNT
EVENT_TYPES = ("proper_crossing", "vertex_crossing", "touch", "vertex_undetermined", "collinear_overlap")


def duplicate_positions(c: RawCache) -> np.ndarray:
    """(F, S) True where a slot's RAW position is bit-identical to another slot's in the same frame.

    Two players cannot occupy exactly the same floating-point position, so this is a provider artifact (found in match 3812:
    seven fixed slot pairs share coordinates for up to ~2 min). It is an exact-equality validity rule, not a proximity criterion:
    such samples are never joined into segments, and coincident paths would otherwise yield thousands of spurious intersections."""
    dup = np.zeros(c.x.shape, bool)
    ok = np.isfinite(c.x) & np.isfinite(c.y)
    for a, b in itertools.combinations(range(c.n_slots), 2):
        eq = ok[:, a] & ok[:, b] & (c.x[:, a] == c.x[:, b]) & (c.y[:, a] == c.y[:, b])
        dup[eq, a] = dup[eq, b] = True
    return dup


class Timeline:
    """Row lookups by period and video time (rows of one period are in time order)."""

    def __init__(self, c: RawCache):
        self.t = c.video_time_s
        self.rows = {int(p): np.flatnonzero(c.canonical_period == p) for p in np.unique(c.canonical_period) if p > 0}
        for p, r in self.rows.items():
            if np.any(np.diff(self.t[r]) <= 0):
                raise ValueError(f"period {p}: video time is not strictly increasing")

    def between(self, p: int, lo: float, hi: float, *, lo_open: bool = False, hi_open: bool = False) -> np.ndarray:
        r = self.rows[p]
        tp = self.t[r]
        a = np.searchsorted(tp, lo, "right" if lo_open else "left")
        b = np.searchsorted(tp, hi, "left" if hi_open else "right")
        return r[a:b]

    def nearest(self, p: int, t: float) -> int:
        r = self.rows[p]
        k = int(np.clip(np.searchsorted(self.t[r], t), 1, len(r) - 1))
        return int(r[k - 1] if abs(self.t[r[k - 1]] - t) <= abs(self.t[r[k]] - t) else r[k])

    def bounds(self, p: int) -> tuple[float, float]:
        r = self.rows[p]
        return float(self.t[r[0]]), float(self.t[r[-1]])


@dataclass(frozen=True)
class DetectConfig:
    window_s: float = 5.0            # advisor: +-2.5 s around the centre
    stride_s: float = 1.0            # advisor: centre advances by 1 s
    tol_m: float = 1e-6              # geometric tolerance (endpoint / collinear / degenerate)
    merge_tol_m: float = 1e-3        # two hits at the same contact point (shared vertex) are one event
    side_probe_s: float = 0.2        # vertex/overlap classification: positions this long before and after the contact
    simultaneous_s: float = 0.2      # |time difference| at or below which a crossing is also labelled simultaneous (metadata)

    @property
    def half(self) -> float:
        return self.window_s / 2

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------- segments

def segments(c: RawCache, mask: np.ndarray, steps: dict[str, np.ndarray], dup: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Polyline segments (one per usable Step 1 step) and counts of the steps that were not joined, by reason (exclusive, in
    this order: invalid, jump, unattributed, duplicate position, zero length)."""
    s, a, b = steps["slot"], steps["row_from"], steps["row_to"]
    valid = steps["valid"].copy()
    jump = valid & steps["jump"]
    unusable = valid & ~jump & (((mask[a, s] | mask[b, s]) & UNUSABLE) != 0)
    duplicate = valid & ~jump & ~unusable & (dup[a, s] | dup[b, s])
    deg = valid & ~jump & ~unusable & ~duplicate & (steps["distance_m"] <= 0)
    keep = valid & ~jump & ~unusable & ~duplicate & ~deg
    counts = dict(steps_total=int(len(s)), invalid_gap_or_period=int((~valid).sum()), implausible_jump=int(jump.sum()),
                  unattributed_or_ambiguous=int(unusable.sum()), duplicate_position=int(duplicate.sum()), zero_length=int(deg.sum()),
                  segments=int(keep.sum()))
    k = np.flatnonzero(keep)
    k = k[np.lexsort((c.video_time_s[a[k]], s[k]))]
    t = c.video_time_s
    seg = dict(slot=s[k], row0=a[k], row1=b[k], t0=t[a[k]], t1=t[b[k]], period=c.canonical_period[a[k]].astype(int),
               x0=c.x[a[k], s[k]], y0=c.y[a[k], s[k]], x1=c.x[b[k], s[k]], y1=c.y[b[k], s[k]])
    return seg, counts


def period_windows(c: RawCache, period: int, cfg: DetectConfig) -> np.ndarray:
    """Window centres of one period (full windows only)."""
    t = c.video_time_s[c.canonical_period == period]
    if not len(t):
        return np.zeros(0)
    n = int(np.floor((t.max() - t.min() - cfg.window_s) / cfg.stride_s + 1e-9)) + 1
    return t.min() + cfg.half + cfg.stride_s * np.arange(max(n, 0))


# ---------------------------------------------------------------- window scan

def scan(c: RawCache, seg: dict[str, np.ndarray], periods: list[int], cfg: DetectConfig) -> tuple[list[dict], dict[str, Any]]:
    """Every (window, pair, segment pair) contact, and the scan counters."""
    hits: list[dict] = []
    st = dict(windows=0, windows_lt2_players=0, pair_checks=0, pair_bbox_rejected=0, pair_windows_with_hit=0, segment_pair_tests=0,
              available_players_per_window=[])
    for p in periods:
        in_p = np.flatnonzero(seg["period"] == p)
        slots = np.unique(seg["slot"][in_p])
        per = {}
        for s in slots:
            ix = in_p[seg["slot"][in_p] == s]                       # sorted by time (segments() sorts by slot, then time)
            per[int(s)] = dict(ix=ix, t0=seg["t0"][ix], t1=seg["t1"][ix],
                               p0=np.c_[seg["x0"][ix], seg["y0"][ix]], p1=np.c_[seg["x1"][ix], seg["y1"][ix]])
        for wc in period_windows(c, p, cfg):
            st["windows"] += 1
            lo_t, hi_t = wc - cfg.half, wc + cfg.half
            sl, box = {}, {}
            for s, d in per.items():
                lo, hi = np.searchsorted(d["t0"], lo_t - 1e-9, "left"), np.searchsorted(d["t1"], hi_t + 1e-9, "right")
                if hi > lo:
                    pts = np.r_[d["p0"][lo:hi], d["p1"][lo:hi]]
                    sl[s] = (lo, hi)
                    box[s] = (*pts.min(0), *pts.max(0))
            av = sorted(sl)
            st["available_players_per_window"].append(len(av))
            if len(av) < 2:
                st["windows_lt2_players"] += 1
                continue
            for sa, sb in itertools.combinations(av, 2):
                st["pair_checks"] += 1
                A, B = box[sa], box[sb]
                if A[0] > B[2] + cfg.tol_m or B[0] > A[2] + cfg.tol_m or A[1] > B[3] + cfg.tol_m or B[1] > A[3] + cfg.tol_m:
                    st["pair_bbox_rejected"] += 1
                    continue
                (la, ha), (lb, hb) = sl[sa], sl[sb]
                da, db = per[sa], per[sb]
                st["segment_pair_tests"] += int((ha - la) * (hb - lb))
                h = intersect(da["p0"][la:ha], da["p1"][la:ha], db["p0"][lb:hb], db["p1"][lb:hb], cfg.tol_m)
                if not len(h):
                    continue
                st["pair_windows_with_hit"] += 1
                for k in range(len(h)):
                    hits.append(dict(window=float(wc), period=p, sa=sa, sb=sb, ga=int(da["ix"][la + h.i[k]]), gb=int(db["ix"][lb + h.j[k]]),
                                     x=float(h.x[k]), y=float(h.y[k]), ta=float(h.ta[k]), tb=float(h.tb[k]), kind=str(h.kind[k])))
    return hits, st


# ---------------------------------------------------------------- deduplication into events

def unique_hits(hits: list[dict]) -> list[dict]:
    """One record per (pair, segment pair) with the windows that found it (stage 1)."""
    by: dict[tuple, dict] = {}
    for h in hits:
        key = (h["sa"], h["sb"], h["ga"], h["gb"])
        if key in by:
            by[key]["windows"].append(h["window"])
        else:
            by[key] = dict(h, windows=[h["window"]])
            by[key].pop("window")
    return sorted(by.values(), key=lambda h: (h["period"], h["sa"], h["sb"], h["ga"], h["gb"]))


def _contiguous(seg: dict[str, np.ndarray], g1: int, g2: int) -> bool:
    """Segments g1 and g2 (same slot) are consecutive pieces of one unbroken path."""
    lo, hi = min(g1, g2), max(g1, g2)
    return hi - lo <= 1 and (hi == lo or (seg["slot"][lo] == seg["slot"][hi] and seg["row1"][lo] == seg["row0"][hi]))


def merge(uh: list[dict], seg: dict[str, np.ndarray], cfg: DetectConfig) -> list[list[dict]]:
    """Group stage-1 hits into physical contacts (stage 2; see the module docstring)."""
    groups: list[list[dict]] = []
    by_pair: dict[tuple, list[dict]] = {}
    for h in uh:
        by_pair.setdefault((h["period"], h["sa"], h["sb"]), []).append(h)
    for hs in by_pair.values():
        parent = list(range(len(hs)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i, j in itertools.combinations(range(len(hs)), 2):
            a, b = hs[i], hs[j]
            if abs(a["ga"] - b["ga"]) > 1 or abs(a["gb"] - b["gb"]) > 1:
                continue
            if not (_contiguous(seg, a["ga"], b["ga"]) and _contiguous(seg, a["gb"], b["gb"])):
                continue
            same_point = np.hypot(a["x"] - b["x"], a["y"] - b["y"]) <= cfg.merge_tol_m
            overlap_chain = COLLINEAR in (a["kind"], b["kind"]) and PROPER not in (a["kind"], b["kind"])
            if same_point or overlap_chain:
                parent[find(i)] = find(j)
        comp: dict[int, list[dict]] = {}
        for i, h in enumerate(hs):
            comp.setdefault(find(i), []).append(h)
        groups.extend(comp.values())
    return groups


def _position(c: RawCache, tl: Timeline, slot: int, period: int, t: float) -> np.ndarray | None:
    """RAW position of a slot at the frame nearest ``t`` in the period, or None when absent there."""
    r = tl.nearest(period, t)
    if abs(c.video_time_s[r] - t) > 0.05 or not np.isfinite(c.x[r, slot]):
        return None
    return np.array([c.x[r, slot], c.y[r, slot]])


def event_record(c: RawCache, mask: np.ndarray, seg: dict[str, np.ndarray], members: list[dict], cfg: DetectConfig, tl: Timeline,
                 dup: np.ndarray) -> dict[str, Any]:
    """One canonical event from its member hits: geometry, traversal times, type, sources and audit metadata."""
    m0 = members[0]
    p, sa, sb = m0["period"], m0["sa"], m0["sb"]
    t = c.video_time_s

    def trav(g: int, u: float) -> float:
        return float(seg["t0"][g] + u * (seg["t1"][g] - seg["t0"][g]))

    ta = np.array([trav(h["ga"], h["ta"]) for h in members])
    tb = np.array([trav(h["gb"], h["tb"]) for h in members])
    rep = int(np.argsort(ta)[len(ta) // 2])                          # median member represents the event
    kinds = {h["kind"] for h in members}
    probe = []
    for slot, tt in ((sa, ta.min() - cfg.side_probe_s), (sa, ta.max() + cfg.side_probe_s), (sb, tb.min() - cfg.side_probe_s),
                     (sb, tb.max() + cfg.side_probe_s)):
        probe.append(_position(c, tl, slot, p, tt))
    side = "undetermined" if any(v is None for v in probe) else classify_vertex(*probe, tol_m=cfg.tol_m)
    if kinds == {PROPER}:
        etype = "proper_crossing"
    elif COLLINEAR in kinds:
        etype = "collinear_overlap"
    else:
        etype = {"crossing": "vertex_crossing", "touch": "touch"}.get(side, "vertex_undetermined")
    windows = sorted({w for h in members for w in h["windows"]})
    tA, tB = float(ta[rep]), float(tb[rep])
    ga, gb = members[rep]["ga"], members[rep]["gb"]
    lo, hi = min(tA, tB), max(tA, tB)
    rows = tl.between(p, lo - 0.5, hi + 0.5)
    d = np.hypot(c.x[rows, sa] - c.x[rows, sb], c.y[rows, sa] - c.y[rows, sb])
    near = rows[np.flatnonzero(np.isfinite(d))]
    # source rows of the first segment on each side: stable across code versions, unlike internal segment indices
    key = f"{c.match_id}|{p}|{sa}|{sb}|{int(seg['row0'][min(h['ga'] for h in members)])}|{int(seg['row0'][min(h['gb'] for h in members)])}"
    span = tl.between(p, lo - cfg.half, hi + cfg.half)
    ms = mask[span][:, [sa, sb]]
    flags = {name: bool(((ms & bit) != 0).any()) for name, bit in
             (("implausible_jump_within_2_5s", Q.IMPLAUSIBLE_JUMP), ("discontinuity_candidate_within_2_5s", Q.DISCONTINUITY_CANDIDATE),
              ("substitution_window_within_2_5s", Q.SUBSTITUTION_WINDOW))}
    flags["duplicate_position_within_2_5s"] = bool(dup[span][:, [sa, sb]].any())
    return dict(
        event_id=hashlib.sha256(key.encode()).hexdigest()[:16], event_key=key, match_id=int(c.match_id), period=int(p),
        slot_a=int(sa), slot_b=int(sb), side_a=str(c.slot_side[sa]), side_b=str(c.slot_side[sb]), jersey_a=int(c.slot_jersey[sa]),
        jersey_b=int(c.slot_jersey[sb]), player_id_a=int(c.slot_player_id[sa]), player_id_b=int(c.slot_player_id[sb]),
        team_relation="same_team" if c.slot_side[sa] == c.slot_side[sb] else "opposite_teams",
        type=etype, member_kinds=sorted(kinds), n_members=len(members), side_classification=side,
        x=float(members[rep]["x"]), y=float(members[rep]["y"]),
        time_a_s=tA, time_b_s=tB, time_diff_s=tB - tA, abs_time_diff_s=abs(tB - tA), simultaneous=abs(tB - tA) <= cfg.simultaneous_s,
        min_synchronous_distance_m=float(np.nanmin(d)) if len(near) else None,
        segment_a=dict(row0=int(seg["row0"][ga]), row1=int(seg["row1"][ga]), frame0=int(c.frame_num[seg["row0"][ga]]),
                       frame1=int(c.frame_num[seg["row1"][ga]]), t0=float(seg["t0"][ga]), t1=float(seg["t1"][ga]), u=float(members[rep]["ta"])),
        segment_b=dict(row0=int(seg["row0"][gb]), row1=int(seg["row1"][gb]), frame0=int(c.frame_num[seg["row0"][gb]]),
                       frame1=int(c.frame_num[seg["row1"][gb]]), t0=float(seg["t0"][gb]), t1=float(seg["t1"][gb]), u=float(members[rep]["tb"])),
        source_records=dict(a=int(c.record_first[seg["row0"][ga]]), b=int(c.record_first[seg["row0"][gb]])),
        n_windows=len(windows), windows=[round(w, 3) for w in windows], quality=flags)


def assign_clusters(events: list[dict], gap_s: float) -> dict[str, Any]:
    """Audit metadata, not deduplication: chain events of one pair whose crossing times are within ``gap_s`` of the previous
    one into a *pair cluster* (repeated crossings of the same two players). Sets ``cluster_id`` / ``cluster_size`` per event."""
    by: dict[tuple, list[dict]] = {}
    for e in events:
        by.setdefault((e["period"], e["slot_a"], e["slot_b"]), []).append(e)
    n = 0
    sizes = []
    for key in sorted(by):
        es = sorted(by[key], key=lambda e: min(e["time_a_s"], e["time_b_s"]))
        start, last = 0, None
        for k, e in enumerate(es + [None]):
            tm = None if e is None else min(e["time_a_s"], e["time_b_s"])
            if e is None or (last is not None and tm - last > gap_s):
                for f in es[start:k]:
                    f["cluster_id"], f["cluster_size"] = f"{f['match_id']}-c{n:05d}", k - start
                sizes.append(k - start)
                n, start = n + 1, k
            if e is not None:
                last = max(e["time_a_s"], e["time_b_s"])
    s = np.array(sizes) if sizes else np.zeros(0, int)
    return dict(clusters=len(sizes), single_event_clusters=int((s == 1).sum()), median_size=float(np.median(s)) if len(s) else None,
                p90_size=float(np.quantile(s, 0.9)) if len(s) else None, max_size=int(s.max()) if len(s) else None)


def detect(c: RawCache, mask: np.ndarray, steps: dict[str, np.ndarray], cfg: DetectConfig = DetectConfig(),
           quarantine: tuple[int, ...] = ()) -> tuple[list[dict], dict[str, Any]]:
    """Events of one match and detection statistics (counts, timings)."""
    t0 = time.perf_counter()
    dup = duplicate_positions(c)
    tl = Timeline(c)
    seg, seg_counts = segments(c, mask, steps, dup)
    t1 = time.perf_counter()
    periods = [int(p) for p in np.unique(c.canonical_period) if p > 0 and int(p) not in quarantine]
    hits, st = scan(c, seg, periods, cfg)
    t2 = time.perf_counter()
    uh = unique_hits(hits)
    groups = merge(uh, seg, cfg)
    events = sorted((event_record(c, mask, seg, g, cfg, tl, dup) for g in groups), key=lambda e: (e["period"], e["time_a_s"], e["slot_a"], e["slot_b"]))
    clusters = assign_clusters(events, cfg.window_s)
    t3 = time.perf_counter()
    apw = np.array(st.pop("available_players_per_window"))
    stats = dict(periods_scanned=periods, periods_quarantined=sorted(quarantine), segments=seg_counts, **st,
                 available_players_per_window=dict(min=int(apw.min()) if len(apw) else None, median=float(np.median(apw)) if len(apw) else None,
                                                   max=int(apw.max()) if len(apw) else None),
                 duplicate_position_samples=int(dup.sum()),
                 segment_pair_hits_per_window=len(hits), unique_segment_pair_hits=len(uh), unique_events=len(events),
                 hits_merged_into_other_events=len(uh) - len(events), pair_clusters=clusters,
                 seconds=dict(segments=round(t1 - t0, 2), scan=round(t2 - t1, 2), dedup_and_events=round(t3 - t2, 2)))
    return events, stats
