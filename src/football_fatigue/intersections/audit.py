"""Stage S1: temporal-intersection quality audit and complete-trajectory store (M11 advisor-directed pipeline, v2).

Starts from the immutable v1 geometric event catalogue (``m11-intersections-v1/<match>/root.events.jsonl``, checksum-verified)
and the Step 1 RAW cache. Nothing in v1 is changed; every v1 event gets exactly one eligibility record.

**Measurements** (per event; each player at its own crossing time ``t_i``; ``t* = (t_A + t_B) / 2``):

- ``abs_time_diff_s`` = |t_B - t_A| (from v1);
- velocity of each player: least-squares line through its update instants within ``t_i +- velocity_half_s``; speed, and the
  crossing angle between the two velocity vectors;
- local position noise ``sigma_i``: RMS of second differences of update-instant positions within ``t_i +- local_s`` divided by
  sqrt(6) per axis. Acceleration contributes too, so it is an upper bound of the measurement noise, and it is large for jitter;
- **timing uncertainty** ``sigma_dt = sqrt(sum_i (sigma_i / (max(v_i, v_floor) * max(sin(angle), sin_floor)))^2 + 2 (res / sqrt 12)^2)``:
  an along-path position error moves the crossing time by sigma/v, amplified by 1/sin(angle) for shallow crossings; ``res`` is
  the median update interval (quantisation). First-order propagation, not a validated error model;
- same-time distance between the players (minimum within ``t* +- window_half_s`` and at ``t*``);
- per player within ``t* +- window_half_s``: coverage, largest gap, VISIBLE / ESTIMATED shares, Step 1 implausible jumps,
  discontinuity candidates, substitution window, unresolved attribution, duplicate-position artifact;
- stationarity (net displacement within ``t_i +- local_s``) and jitter (straightness = net / path length; ``sigma_i``).

**Classification** at each threshold ``tau`` (first matching rule decides the category; *all* failing checks are recorded):

1. ``invalid_tracking_artifact``: duplicate-position artifact, unresolved / ambiguous attribution, the window leaves the period,
   coverage below ``min_coverage`` or an internal gap above ``max_gap_s`` for either player (the trajectory cannot be preserved
   completely or is not one player's);
2. ``non_time_aligned_crossing``: |dt| > tau + z * min(sigma_dt, sigma_cap_s);
3. ``uncertain_candidate``: tau < |dt| <= tau + z * min(sigma_dt, sigma_cap_s) (``timing_boundary``), or time-aligned but with
   any of: timing ill-conditioned (sigma_dt > timing_ratio * tau, speed below v_floor, angle below the floor, no velocity),
   stationary player, jitter, implausible jump, discontinuity candidate, substitution window, all-ESTIMATED player;
4. ``clean_candidate``: |dt| <= tau and none of the above.

ESTIMATED is the provider's visibility flag: it is recorded and can make an event uncertain, but it is never treated as evidence
of visual occlusion. Geometric and temporal coincidence is not verified occlusion.

**Trajectory store** (``root.trajectories.npz``): for every event that is a clean or uncertain candidate at any evaluated
threshold, every RAW frame of both players within ``t* +- window_half_s``, copied without change (no smoothing, no
interpolation, nothing masked or truncated inside the window). Tracks are ``track_0`` / ``track_1`` in a seeded order; their
identities and the continuity reference are only in ``root.references.jsonl``. Visibility, confidence and quality bits are
stored as metadata; whether any of them may be a model input is a later, separate decision.

Run: ``python -m football_fatigue.intersections.audit --match <development id> [--dry-run]``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from football_fatigue.encounters import quality as Q
from football_fatigue.encounters.trajectories import RawCache
from football_fatigue.ingestion import EventVersion, IngestionError, approved_output_root, safe_output_path
from football_fatigue.matching import ml, validate
from football_fatigue.matching.benchmark import jsonl, load_source

from .__main__ import peak_memory_mb, write_dir
from .detect import UNUSABLE, Timeline, duplicate_positions
from .samples import SCHEMA as V1_SCHEMA

SCHEMA = "m11-intersections-v2"
CATEGORIES = ("clean_candidate", "uncertain_candidate", "invalid_tracking_artifact", "non_time_aligned_crossing")
INVALID_REASONS = ("duplicate_position", "attribution_unresolved_or_ambiguous", "period_boundary", "low_coverage", "internal_gap")
UNCERTAIN_REASONS = ("timing_boundary", "timing_ill_conditioned", "stationary_player", "jitter", "implausible_jump",
                     "discontinuity_candidate", "substitution_window", "all_estimated_player")
STORE_ROW_KEYS = ("frame_num", "video_time_s", "record_first")
STORE_TRACK_KEYS = ("x", "y", "state", "visibility", "confidence", "quality_mask", "update_instant", "duplicate_position")


@dataclass(frozen=True)
class AuditConfig:
    thresholds_s: tuple[float, ...] = (0.10, 0.20, 0.30)
    primary_s: float = 0.20                # provisional; awaits advisor confirmation
    window_half_s: float = 2.5             # the existing +-2.5 s window, not expanded
    velocity_half_s: float = 0.25
    local_s: float = 1.0                   # stationarity / jitter / noise window around each player's crossing
    v_floor_m_s: float = 0.5               # below: speed too low for a defined crossing time
    angle_floor_deg: float = 15.0          # below: crossing too shallow for a defined crossing time
    z: float = 2.0
    sigma_cap_s: float = 0.5               # timing uncertainty above this means the crossing time is undefined at these scales
    timing_ratio: float = 0.5              # sigma_dt above timing_ratio * tau: ill-conditioned at that tau
    min_coverage: float = 0.8              # same completeness rules as the v1 samples
    max_gap_s: float = 0.5
    stationary_net_m: float = 0.5
    jitter_straightness: float = 0.5
    jitter_sigma_m: float = 0.10
    cluster_gap_s: float = 5.0
    seed: str = "m11-intersections-v2"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _f(v, nd=4):
    return None if v is None or not np.isfinite(v) else round(float(v), nd)


def load_v1(raw_root, match_id) -> tuple[list[dict], dict]:
    """The v1 events after checking them, and every other v1 file, against the v1 manifest."""
    d = approved_output_root(raw_root) / V1_SCHEMA / str(match_id)
    m = json.loads((d / "root.manifest.json").read_text(encoding="utf-8"))
    bad = [n for n, h in m["outputs_sha256"].items() if hashlib.sha256((d / n).read_bytes()).hexdigest() != h]
    if bad or m.get("schema") != V1_SCHEMA:
        raise IngestionError(f"v1 outputs of match {match_id} do not match their manifest: {bad}")
    events = [json.loads(line) for line in (d / "root.events.jsonl").read_text(encoding="utf-8").splitlines()]
    prov = dict(directory=str(d), manifest_sha256=hashlib.sha256((d / "root.manifest.json").read_bytes()).hexdigest(),
                events_sha256=m["outputs_sha256"]["root.events.jsonl"], files_verified=len(m["outputs_sha256"]))
    return events, prov


# ---------------------------------------------------------------- measurements

class Context:
    def __init__(self, c: RawCache, mask: np.ndarray, cfg: AuditConfig):
        self.c, self.mask, self.cfg = c, mask, cfg
        self.tl, self.dup = Timeline(c), duplicate_positions(c)
        self.upd = ((mask & Q.UPDATE_INSTANT) != 0) & np.isfinite(c.x)
        gaps = [np.diff(c.video_time_s[self.upd[:, s]]) for s in range(c.n_slots)]
        g = np.concatenate([v[v < 0.2] for v in gaps if len(v)])
        self.update_interval_s = float(np.median(g)) if len(g) else 1 / c.fps


def _updates(cx: Context, slot: int, p: int, lo: float, hi: float) -> np.ndarray:
    r = cx.tl.between(p, lo, hi)
    return r[cx.upd[r, slot]]


def player_motion(cx: Context, slot: int, p: int, t: float) -> dict[str, Any]:
    c, cfg = cx.c, cx.cfg
    r = _updates(cx, slot, p, t - cfg.velocity_half_s, t + cfg.velocity_half_s)
    vel = None
    if len(r) >= 3:
        tt = c.video_time_s[r] - t
        vel = np.array([np.polyfit(tt, c.x[r, slot], 1)[0], np.polyfit(tt, c.y[r, slot], 1)[0]])
    r = _updates(cx, slot, p, t - cfg.local_s, t + cfg.local_s)
    x, y, tt = c.x[r, slot], c.y[r, slot], c.video_time_s[r]
    path = float(np.hypot(np.diff(x), np.diff(y)).sum()) if len(r) > 1 else 0.0
    net = float(np.hypot(x[-1] - x[0], y[-1] - y[0])) if len(r) > 1 else 0.0
    sigma = None
    if len(r) >= 3:
        d1 = np.diff(tt)
        ok = np.abs(d1[1:] - d1[:-1]) <= 0.1 * np.maximum(d1[1:], d1[:-1])           # near-uniform triples only
        if ok.any():
            d2 = np.r_[(x[2:] - 2 * x[1:-1] + x[:-2])[ok], (y[2:] - 2 * y[1:-1] + y[:-2])[ok]]
            sigma = float(np.sqrt(np.mean(d2 ** 2) / 6))
    return dict(velocity=vel, speed=None if vel is None else float(np.hypot(*vel)), net_m=net, path_m=path,
                straightness=net / path if path > 0 else 0.0, sigma_m=sigma, n_updates=int(len(r)))


def window_quality(cx: Context, slot: int, rows: np.ndarray) -> dict[str, Any]:
    c, m = cx.c, cx.mask[rows, slot]
    ok = np.isfinite(c.x[rows, slot])
    t = c.video_time_s[rows]
    edges = np.r_[t[0], t[ok], t[-1]] if len(rows) else np.zeros(1)
    present = ok | ((m & (Q.AMBIGUOUS_RAW | Q.MISSING_RAW)) != 0)
    vis = c.visibility[rows, slot][ok]
    return dict(coverage=float(ok.mean()) if len(rows) else 0.0, max_gap_s=float(np.diff(edges).max()) if len(edges) > 1 else 0.0,
                n_frames=int(len(rows)), n_present=int(ok.sum()),
                visible_share=float(np.mean(vis == 1)) if len(vis) else None, estimated_share=float(np.mean(vis == 2)) if len(vis) else None,
                implausible_jumps=int(((m & Q.IMPLAUSIBLE_JUMP) != 0).sum()), discontinuity_candidates=int(((m & Q.DISCONTINUITY_CANDIDATE) != 0).sum()),
                substitution_window=bool(((m & Q.SUBSTITUTION_WINDOW) != 0).any()), unresolved_attribution=int(((m & UNUSABLE) != 0)[present].sum()),
                duplicate_positions=int(cx.dup[rows, slot].sum()))


def measure(cx: Context, ev: dict) -> dict[str, Any]:
    c, cfg = cx.c, cx.cfg
    p, sa, sb, ta, tb = ev["period"], ev["slot_a"], ev["slot_b"], ev["time_a_s"], ev["time_b_s"]
    t_star = (ta + tb) / 2
    lo, hi = cx.tl.bounds(p)
    rows = cx.tl.between(p, t_star - cfg.window_half_s, t_star + cfg.window_half_s)
    ma, mb = player_motion(cx, sa, p, ta), player_motion(cx, sb, p, tb)
    sin = None
    if ma["speed"] and mb["speed"]:
        va, vb = ma["velocity"], mb["velocity"]
        sin = abs(va[0] * vb[1] - va[1] * vb[0]) / (ma["speed"] * mb["speed"])
    sin_floor = math.sin(math.radians(cfg.angle_floor_deg))
    q = cx.update_interval_s / math.sqrt(12)                         # quantisation of the update instants
    sigma_dt = None
    if sin is not None and ma["sigma_m"] is not None and mb["sigma_m"] is not None:
        terms = [mm["sigma_m"] / (max(mm["speed"], cfg.v_floor_m_s) * max(sin, sin_floor)) for mm in (ma, mb)]
        sigma_dt = math.sqrt(terms[0] ** 2 + terms[1] ** 2 + 2 * q ** 2)
    d = np.hypot(c.x[rows, sa] - c.x[rows, sb], c.y[rows, sa] - c.y[rows, sb])
    fin = np.isfinite(d)
    r_star = cx.tl.nearest(p, t_star)
    d_star = float(np.hypot(c.x[r_star, sa] - c.x[r_star, sb], c.y[r_star, sa] - c.y[r_star, sb]))

    def motion(mm):
        return dict(speed_m_s=_f(mm["speed"], 3), net_displacement_m=_f(mm["net_m"], 3), path_length_m=_f(mm["path_m"], 3),
                    straightness=_f(mm["straightness"], 3), local_sigma_m=_f(mm["sigma_m"], 4), n_updates=mm["n_updates"])

    return dict(
        t_star_s=round(t_star, 4), abs_time_diff_s=round(ev["abs_time_diff_s"], 4), time_diff_s=round(ev["time_diff_s"], 4),
        crossing_angle_deg=None if sin is None else round(math.degrees(math.asin(min(sin, 1.0))), 2),
        timing_sigma_s=_f(sigma_dt, 4), update_interval_s=round(cx.update_interval_s, 4),
        min_same_time_distance_m=_f(float(d[fin].min()) if fin.any() else None, 3),
        min_same_time_distance_at_s=_f(float(c.video_time_s[rows][fin][np.argmin(d[fin])] - t_star) if fin.any() else None, 3),
        distance_at_t_star_m=_f(d_star, 3), v1_min_synchronous_distance_m=_f(ev["min_synchronous_distance_m"], 3),
        window_s=[round(t_star - cfg.window_half_s, 4), round(t_star + cfg.window_half_s, 4)],
        window_in_period=bool(t_star - cfg.window_half_s >= lo and t_star + cfg.window_half_s <= hi),
        motion_a=motion(ma), motion_b=motion(mb), window_a=window_quality(cx, sa, rows), window_b=window_quality(cx, sb, rows),
        _sin=sin)


# ---------------------------------------------------------------- classification

def checks(m: dict, cfg: AuditConfig) -> dict[str, bool]:
    """Threshold-independent checks (True = problem present)."""
    wa, wb, ma, mb = m["window_a"], m["window_b"], m["motion_a"], m["motion_b"]
    both = (wa, wb)
    sin_floor = math.sin(math.radians(cfg.angle_floor_deg))
    speeds = [mm["speed_m_s"] for mm in (ma, mb)]
    out = dict(
        duplicate_position=any(w["duplicate_positions"] for w in both),
        attribution_unresolved_or_ambiguous=any(w["unresolved_attribution"] for w in both),
        period_boundary=not m["window_in_period"],
        low_coverage=any(w["coverage"] < cfg.min_coverage for w in both),
        internal_gap=any(w["max_gap_s"] > cfg.max_gap_s for w in both),
        velocity_unavailable=None in speeds or m["timing_sigma_s"] is None,
        slow_at_crossing=any(s is not None and s < cfg.v_floor_m_s for s in speeds),
        shallow_angle=m["_sin"] is not None and m["_sin"] < sin_floor,
        stationary_player=any(mm["net_displacement_m"] is not None and mm["net_displacement_m"] < cfg.stationary_net_m for mm in (ma, mb)),
        jitter=any((mm["straightness"] is not None and mm["straightness"] < cfg.jitter_straightness)
                   or (mm["local_sigma_m"] is not None and mm["local_sigma_m"] > cfg.jitter_sigma_m) for mm in (ma, mb)),
        implausible_jump=any(w["implausible_jumps"] for w in both),
        discontinuity_candidate=any(w["discontinuity_candidates"] for w in both),
        substitution_window=any(w["substitution_window"] for w in both),
        all_estimated_player=any(w["estimated_share"] is not None and w["estimated_share"] >= 1.0 for w in both))
    return {k: bool(v) for k, v in out.items()}


def classify(m: dict, ck: dict[str, bool], tau: float, cfg: AuditConfig) -> dict[str, Any]:
    """Category and reasons at one threshold (see the module docstring)."""
    dt = m["abs_time_diff_s"]
    sigma = m["timing_sigma_s"] if m["timing_sigma_s"] is not None else cfg.sigma_cap_s
    band = tau + cfg.z * min(sigma, cfg.sigma_cap_s)
    invalid = [r for r in INVALID_REASONS if ck[r]]
    timing = "aligned" if dt <= tau else "boundary" if dt <= band else "not_aligned"
    ill = ck["velocity_unavailable"] or ck["slow_at_crossing"] or ck["shallow_angle"] or sigma > cfg.timing_ratio * tau
    unc = (["timing_boundary"] if timing == "boundary" else []) + (["timing_ill_conditioned"] if ill else []) + \
          [r for r in UNCERTAIN_REASONS[2:] if ck[r]]
    if invalid:
        cat = "invalid_tracking_artifact"
    elif timing == "not_aligned":
        cat = "non_time_aligned_crossing"
    elif unc:
        cat = "uncertain_candidate"
    else:
        cat = "clean_candidate"
    reasons = invalid + (["abs_time_diff_exceeds_tau_plus_z_sigma"] if timing == "not_aligned" else []) + unc
    return dict(category=cat, timing=timing, band_s=round(band, 4), reasons=reasons)


def candidate_clusters(recs: list[dict], key: str, gap_s: float) -> None:
    """Chain clean / uncertain candidates of one pair whose times are within ``gap_s`` (sets ``candidate_cluster`` per threshold)."""
    by: dict[tuple, list[dict]] = {}
    for r in recs:
        r.setdefault("candidate_cluster", {})[key] = None
        if r["status"][key]["category"] in ("clean_candidate", "uncertain_candidate"):
            by.setdefault((r["period"], r["pair_id"]), []).append(r)
    n = 0
    for k in sorted(by):
        last = None
        for r in sorted(by[k], key=lambda r: r["measurements"]["t_star_s"]):
            t = r["measurements"]["t_star_s"]
            if last is None or t - last > gap_s:
                n += 1
            r["candidate_cluster"][key] = f"{r['match_id']}-{key}-k{n:05d}"
            last = t


def tkey(tau: float) -> str:
    return f"{tau:.2f}"


def audit_events(cx: Context, events: list[dict]) -> list[dict]:
    cfg = cx.cfg
    recs = []
    for ev in events:
        m = measure(cx, ev)
        ck = checks(m, cfg)
        m.pop("_sin")
        lo, hi = sorted((ev["slot_a"], ev["slot_b"]))
        status = {tkey(t): classify(m, ck, t, cfg) for t in cfg.thresholds_s}
        recs.append(dict(event_id=ev["event_id"], event_key=ev["event_key"], match_id=ev["match_id"], period=ev["period"],
                         pair_id=f"{ev['match_id']}:{lo}-{hi}", v1_type=ev["type"], v1_cluster_id=ev["cluster_id"],
                         v1_cluster_size=ev["cluster_size"], v1_n_windows=ev["n_windows"], measurements=m, checks=ck, status=status,
                         primary_category=status[tkey(cfg.primary_s)]["category"], primary_reasons=status[tkey(cfg.primary_s)]["reasons"]))
    for t in cfg.thresholds_s:
        candidate_clusters(recs, tkey(t), cfg.cluster_gap_s)
    return recs


# ---------------------------------------------------------------- complete-trajectory store

def _bit(seed: str, key: str) -> int:
    return hashlib.sha256(f"{seed}|{key}".encode()).digest()[0] & 1


def build_store(cx: Context, events: list[dict], recs: list[dict]) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Lossless RAW slices of both players for every clean / uncertain candidate at any threshold, plus the separate references."""
    c, cfg = cx.c, cx.cfg
    by_id = {e["event_id"]: e for e in events}
    keep = [r for r in recs if any(s["category"] in ("clean_candidate", "uncertain_candidate") for s in r["status"].values())]
    keep.sort(key=lambda r: r["event_id"])
    offsets, rows_all, slots_all, ids, tstar, period = [0], [], [], [], [], []
    refs = []

    def ident(s: int) -> dict:
        return dict(slot=int(s), side=str(c.slot_side[s]), jersey=int(c.slot_jersey[s]), player_id=int(c.slot_player_id[s]),
                    team_id=int(c.slot_team_id[s]))

    for k, r in enumerate(keep):
        ev = by_id[r["event_id"]]
        rows = cx.tl.between(ev["period"], r["measurements"]["window_s"][0], r["measurements"]["window_s"][1])
        first_a = _bit(cfg.seed, ev["event_key"])
        s0, s1 = (ev["slot_a"], ev["slot_b"]) if first_a else (ev["slot_b"], ev["slot_a"])
        t0, t1 = (ev["time_a_s"], ev["time_b_s"]) if first_a else (ev["time_b_s"], ev["time_a_s"])
        rows_all.append(rows)
        slots_all.append(np.tile([s0, s1], (len(rows), 1)))
        offsets.append(offsets[-1] + len(rows))
        ids.append(r["event_id"])
        tstar.append(r["measurements"]["t_star_s"])
        period.append(ev["period"])
        flags = sorted({x for s in r["status"].values() for x in s["reasons"]})
        refs.append(dict(event_id=r["event_id"], store_index=k, schema=SCHEMA, match_id=r["match_id"], period=ev["period"],
                         tracks={"track_0": ident(s0), "track_1": ident(s1)},
                         crossing_time_s={"track_0": round(float(t0), 4), "track_1": round(float(t1), 4)},
                         continuity_reference=dict(source="provider slot continuity (RAW tracking labels); not independently verified",
                                                   track_0="one provider slot throughout the window", track_1="one provider slot throughout the window"),
                         reliability="flagged" if flags else "no_flags", flags=flags,
                         primary_category=r["primary_category"]))
    R = np.concatenate(rows_all) if rows_all else np.zeros(0, int)
    S = np.concatenate(slots_all) if slots_all else np.zeros((0, 2), int)
    take = lambda a: a[R[:, None], S] if len(R) else np.zeros((0, 2), a.dtype)  # noqa: E731
    store = dict(event_id=np.array(ids, "<U16"), t_star_s=np.array(tstar, float), period=np.array(period, np.int8),
                 offsets=np.array(offsets, np.int64), frame_num=c.frame_num[R], video_time_s=c.video_time_s[R], record_first=c.record_first[R],
                 x=take(c.x), y=take(c.y), state=take(c.state), visibility=take(c.visibility), confidence=take(c.confidence),
                 quality_mask=take(cx.mask), update_instant=take(cx.upd), duplicate_position=take(cx.dup))
    return store, refs


def npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:      # fixed member order and timestamps: deterministic bytes
        for k in sorted(arrays):
            b = io.BytesIO()
            np.lib.format.write_array(b, np.asanyarray(arrays[k]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{k}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, b.getvalue())
    return buf.getvalue()


# ---------------------------------------------------------------- summary

def _q(v, qs=(0.1, 0.5, 0.9)):
    v = np.array([x for x in v if x is not None], float)
    return {f"q{int(q * 100)}": round(float(np.quantile(v, q)), 3) for q in qs} if len(v) else None


def summarize(recs: list[dict], cfg: AuditConfig) -> dict[str, Any]:
    out = {}
    for t in cfg.thresholds_s:
        k = tkey(t)
        cats = Counter(r["status"][k]["category"] for r in recs)
        reasons = {cat: dict(Counter(x for r in recs if r["status"][k]["category"] == cat for x in r["status"][k]["reasons"]))
                   for cat in CATEGORIES}
        cand = [r for r in recs if r["status"][k]["category"] in ("clean_candidate", "uncertain_candidate")]
        clean = [r for r in cand if r["status"][k]["category"] == "clean_candidate"]
        pc = Counter(r["pair_id"] for r in cand)

        def strata(sub):
            m = [r["measurements"] for r in sub]
            est = [np.mean([x["window_a"]["estimated_share"] or 0, x["window_b"]["estimated_share"] or 0]) for x in m]
            return dict(n=len(sub), estimated_share=_q(est), all_visible_both=int(sum(e == 0 for e in est)),
                        min_same_time_distance_m=_q(x["min_same_time_distance_m"] for x in m),
                        timing_sigma_s=_q(x["timing_sigma_s"] for x in m),
                        speed_min_m_s=_q(min(x["motion_a"]["speed_m_s"] or 0, x["motion_b"]["speed_m_s"] or 0) for x in m),
                        crossing_angle_deg=_q(x["crossing_angle_deg"] for x in m))

        out[k] = dict(categories={cat: cats.get(cat, 0) for cat in CATEGORIES}, reasons_by_category=reasons,
                      timing=dict(Counter(r["status"][k]["timing"] for r in recs)),
                      candidates=dict(clean_or_uncertain=len(cand), distinct_pairs=len(pc), candidate_clusters=len({r["candidate_cluster"][k] for r in cand}),
                                      clean_distinct_pairs=len({r["pair_id"] for r in clean}),
                                      clean_candidate_clusters=len({r["candidate_cluster"][k] for r in clean}),
                                      top_pair_events=max(pc.values()) if pc else 0,
                                      top10pct_pairs_share=round(sum(sorted(pc.values(), reverse=True)[:max(1, len(pc) // 10)]) / max(1, len(cand)), 3)),
                      measurements_by_category={cat: strata([r for r in recs if r["status"][k]["category"] == cat]) for cat in CATEGORIES})
    ck = Counter(n for r in recs for n, v in r["checks"].items() if v)
    return dict(schema=SCHEMA, events=len(recs), primary_threshold_s=cfg.primary_s, by_threshold=out, checks_any_threshold=dict(ck),
                note=("Categories describe tracking quality and temporal alignment of geometric intersections. None is a verified visual "
                      "occlusion or identity swap; ESTIMATED is the provider's visibility flag, not occlusion evidence."))


# ---------------------------------------------------------------- run

def run(raw_root: str, match_id: str, dry_run: bool, cfg: AuditConfig = AuditConfig()) -> dict[str, Any]:
    from . import review

    t0 = time.perf_counter()
    split = json.loads((approved_output_root(raw_root) / ml.SPLITS_DIR / "splits.json").read_text(encoding="utf-8"))
    validate.check_split(split, [str(match_id)], "development")
    base = approved_output_root(raw_root) / SCHEMA
    out = safe_output_path(raw_root, match_id, EventVersion.ROOT, base, suffix=".manifest.json").parent
    if not dry_run and out.exists():
        raise IngestionError(f"{out} already exists and is never overwritten")
    snap0 = validate.raw_snapshot(raw_root)
    events, v1prov = load_v1(raw_root, match_id)
    c, mask, _enc, src_manifest, prov = load_source(raw_root, match_id)
    cx = Context(c, mask, cfg)
    t1 = time.perf_counter()
    recs = audit_events(cx, events)
    store, refs = build_store(cx, events, recs)
    in_store = {x["event_id"] for x in refs}
    for r in recs:
        r["in_trajectory_store"] = r["event_id"] in in_store
    t2 = time.perf_counter()
    summary = summarize(recs, cfg)
    summary["trajectory_store"] = dict(events=len(refs), rows=int(store["offsets"][-1]), window_half_s=cfg.window_half_s,
                                       selection="clean or uncertain candidate at any evaluated threshold")
    summary["seconds"] = dict(load=round(t1 - t0, 1), audit=round(t2 - t1, 1))
    summary["peak_memory_mb"] = peak_memory_mb()
    if dry_run:
        return summary
    figs, examples, worksheet, key = review.build(cx, events, recs, cfg)
    t3 = time.perf_counter()
    summary["seconds"]["figures"] = round(t3 - t2, 1)
    summary["examples"], summary["review"] = examples, dict(items=len(key), worksheet="root.review_worksheet.csv", key="root.review_key.jsonl")
    blobs = {"root.eligibility.jsonl": jsonl(recs), "root.references.jsonl": jsonl(refs), "root.trajectories.npz": npz_bytes(store),
             "root.review_worksheet.csv": worksheet, "root.review_key.jsonl": jsonl(key),
             "root.summary.json": json.dumps(summary, indent=1, sort_keys=True).encode(),
             "root.config.json": json.dumps(cfg.as_dict(), indent=1, sort_keys=True).encode(), **figs}
    snap1 = validate.raw_snapshot(raw_root)
    _, v1prov_after = load_v1(raw_root, match_id)
    manifest = dict(schema=SCHEMA, match_id=int(c.match_id), split=split["matches"][str(match_id)], config=cfg.as_dict(),
                    code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))},
                    v1_source=v1prov, v1_unchanged=v1prov == v1prov_after, step1_source=prov, step1_schema=src_manifest["schema"],
                    coordinate_policy="raw", provider_smoothed_coordinates_used=False, raw_unchanged=snap0 == snap1,
                    outputs_sha256={n: hashlib.sha256(b).hexdigest() for n, b in blobs.items()}, python=platform.python_version(),
                    numpy=np.__version__, seconds=summary["seconds"],
                    file_roles=dict(audit_only=["root.eligibility.jsonl", "root.review_key.jsonl", "root.fig-*.png"],
                                    identity_references=["root.references.jsonl"],
                                    trajectory_store_without_identities=["root.trajectories.npz"],
                                    blinded_review=["root.review_worksheet.csv", "root.review-*.png"]))
    blobs["root.manifest.json"] = json.dumps(manifest, indent=1, sort_keys=True).encode()
    write_dir(out, blobs)
    summary["written_to"] = str(out)
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m football_fatigue.intersections.audit")
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--match", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    try:
        out = run(str(Path(a.root).resolve()), a.match, a.dry_run)
    except IngestionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
