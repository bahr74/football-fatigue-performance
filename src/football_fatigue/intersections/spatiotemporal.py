"""Spatiotemporal occlusion-proxy dataset (M11): ``m11-occlusion-spatiotemporal-v1``.

Project decision (2026-11-09): a geometric crossing alone is not an occlusion. A candidate counts only when the two players were
at the intersection point P at the same time, up to the uncertainty of the tracking data itself. Inputs, read and checksum-verified,
never changed:

- the geometric candidates ``m11-occlusion-catalogue-v1`` (exact P, t_A, t_B from the source timestamps, windows, Before / After);
- the per-event measurements of ``m11-intersections-v3`` (S1 audit code, unchanged): speeds, local position noise, crossing angle,
  window completeness and the first-order timing uncertainty ``sigma_dt`` of each candidate.

**Temporal coincidence rule** (per candidate; ``h`` = the match's native provider update interval, 0.0667 s = 2 video frames):

- ``tau = h``: the time resolution of the positions. Two arrivals closer than one update cannot be ordered from the data.
- ``tol = z * sigma_dt`` (z = 2): the band within which measurement noise (position jitter / speed, amplified at shallow angles)
  plus quantisation could make truly simultaneous arrivals look different. ``sigma_dt >= 0.027 s`` from quantisation alone.
- **rejected** (``sequential_pass``): |t_A - t_B| > max(tau, tol), the difference exceeds both the resolution and the noise.
- **accepted**: |t_A - t_B| <= tol <= tau (consistent with simultaneity *and* timing precise to one update), the crossing time is
  well conditioned (speed >= 0.5 m/s, angle >= 15 deg, velocity available; S1 rules), no missing sample at the crossing and no
  tracking-artifact veto (duplicate position, unresolved attribution, incomplete or period-crossing window, speed spike within
  0.5 s of the crossing; S1 / S1.1 rules).
- **ambiguous** otherwise, with every reason: ``timing_precision_insufficient`` (tol > tau: noise wider than the resolution),
  ``beyond_noise_band_within_resolution`` (tol < |dt| <= tau), ``timing_ill_conditioned``, ``missing_sample_at_crossing``, vetoes.

Nothing is tuned for event count; the grid tau in {0.5, 1, 1.5, 2, 3} x h and z in {1, 2, 3} is reported as sensitivity.
Provider ESTIMATED positions at the crossing are flagged, never a decision input (the data cannot say whether they are reliable).
Accepted events are tracking-derived occlusion / encounter proxies, not verified camera occlusions.

**Smoothing study** (``--smoothing``; a derived copy only): a quadratic Savitzky-Golay filter over 5 update instants (0.27 s) on
each contiguous valid run of each player (never across gaps, jumps, invalid steps or periods; run ends unchanged), the unchanged
detector on the smoothed polylines, and a one-to-one match with the raw candidates. The dataset itself is built from raw positions.

Run: ``python -m football_fatigue.intersections.spatiotemporal --matches 3812 [...] [--smoothing] [--dry-run]``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from football_fatigue.ingestion import IngestionError, approved_output_root
from football_fatigue.matching import ml, validate
from football_fatigue.matching.benchmark import jsonl, load_source

from . import audit as AU
from .__main__ import peak_memory_mb
from .catalogue import SCHEMA as CATALOGUE, figure, load_steps, paths, segment_index, write_retry
from .detect import DetectConfig, duplicate_positions, merge, scan, segments, unique_hits

SCHEMA = "m11-occlusion-spatiotemporal-v1"
V3 = "m11-intersections-v3"
VETOES = ("duplicate_position", "attribution_unresolved_or_ambiguous", "period_boundary", "low_coverage", "internal_gap",
          "speed_spike_near_crossing")
ILL = ("velocity_unavailable", "slow_at_crossing", "shallow_angle")
FLAGS = ("stationary_player", "jitter", "implausible_jump", "discontinuity_candidate", "substitution_window", "all_estimated_player")
SG5 = np.array([-3.0, 12.0, 17.0, 12.0, -3.0]) / 35.0
DECISIONS = ("accepted", "ambiguous", "rejected")


@dataclass(frozen=True)
class CoincidenceConfig:
    z: float = 2.0
    resolution_multiple: float = 1.0                 # tau = multiple * native update interval
    gap_factor: float = 1.5                          # a crossing segment longer than 1.5 updates spans a missing sample
    sensitivity_multiples: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0)
    sensitivity_z: tuple[float, ...] = (1.0, 2.0, 3.0)
    match_t_s: float = 0.25                          # raw / smoothed comparison only: same pair, both times and P close
    match_p_m: float = 0.5

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------- inputs

def _verified_lines(d: Path, name: str) -> tuple[list[dict], str]:
    m = json.loads((d / "root.manifest.json").read_text(encoding="utf-8"))
    b = (d / name).read_bytes()
    h = hashlib.sha256(b).hexdigest()
    if h != m["outputs_sha256"][name]:
        raise IngestionError(f"{d / name} does not match its manifest")
    return [json.loads(x) for x in b.decode("utf-8").splitlines()], h


def load_inputs(raw_root: str, match_id: str) -> tuple[list[dict], dict[str, dict], dict[str, str]]:
    base = approved_output_root(raw_root)
    cat, hc = _verified_lines(base / CATALOGUE / str(match_id), "root.catalogue.jsonl")
    elig, he = _verified_lines(base / V3 / str(match_id), "root.eligibility.jsonl")
    by = {e["event_id"]: e for e in elig}
    if sorted(by) != sorted(r["event_id"] for r in cat):
        raise IngestionError(f"match {match_id}: catalogue and v3 eligibility describe different events")
    return cat, by, {f"{CATALOGUE}/root.catalogue.jsonl": hc, f"{V3}/root.eligibility.jsonl": he}


# ---------------------------------------------------------------- decision

def timing_class(r: dict) -> str:
    a, b = r["segment_a"], r["segment_b"]
    if r["timing"]["numerical_tie"]:
        return "exact_equality"
    if (a["t0"], a["t1"]) == (b["t0"], b["t1"]):
        return "same_sample_interval"
    return "overlapping_intervals" if r["timing"]["concurrent_crossing_segments"] else "disjoint_intervals"


def decide(abs_dt: float, sigma: float, h: float, gap: bool, ill: list[str], vetoes: list[str], z: float, mult: float) -> tuple[str, list[str]]:
    """Decision and reasons of one candidate (see the module docstring)."""
    tau, tol = mult * h, z * sigma
    if abs_dt > max(tau, tol):
        return "rejected", ["sequential_pass"]
    reasons = []
    if tol > tau:
        reasons.append("timing_precision_insufficient")
    elif abs_dt > tol:
        reasons.append("beyond_noise_band_within_resolution")
    if ill:
        reasons.append("timing_ill_conditioned")
    if gap:
        reasons.append("missing_sample_at_crossing")
    reasons += vetoes
    return ("accepted" if not reasons else "ambiguous"), reasons


def evaluate(r: dict, e: dict, cfg: CoincidenceConfig) -> dict[str, Any]:
    """Spatiotemporal record of one geometric candidate ``r`` with its v3 measurements ``e``."""
    m, ck = e["measurements"], e["checks"]
    h = m["update_interval_s"]
    sigma = m["timing_sigma_s"]
    gap = max(r["segment_a"]["t1"] - r["segment_a"]["t0"], r["segment_b"]["t1"] - r["segment_b"]["t0"]) > cfg.gap_factor * h
    ill = [k for k in ILL if ck[k]]
    vet = [k for k in VETOES if ck[k]]
    abs_dt = r["timing"]["abs_dt_s"]
    decision, reasons = decide(abs_dt, sigma if sigma is not None else np.inf, h, gap, ill, vet, cfg.z, cfg.resolution_multiple)
    tr = r["trajectories"]
    est = {s: 2 in tr[s]["visibility_at_crossing"] for s in ("a", "b")}
    tm = r["timing"]
    sens = {}
    for mult in cfg.sensitivity_multiples:
        for z in cfg.sensitivity_z:
            sens[f"{mult:g}h_z{z:g}"] = decide(abs_dt, sigma if sigma is not None else np.inf, h, gap, ill, vet, z, mult)[0]
    keep = ("event_id", "event_key", "match_id", "period", "slot_a", "slot_b", "side_a", "side_b", "jersey_a", "jersey_b", "player_id_a",
            "player_id_b", "team_relation", "type", "x", "y", "time_a_s", "time_b_s", "segment_a", "segment_b", "source_records",
            "windows", "n_windows", "window", "trajectories", "dedup", "cluster_id", "cluster_size")
    return dict(
        {k: r[k] for k in keep},
        coincidence=dict(decision=decision, reasons=reasons, abs_dt_s=abs_dt, dt_s=tm["dt_s"], tau_s=cfg.resolution_multiple * h,
                         tol_s=None if sigma is None else cfg.z * sigma, sigma_dt_s=sigma, z=cfg.z, update_interval_s=h,
                         timing_class=timing_class(r), crossing_segments_overlap_s=tm["crossing_segments_overlap_s"],
                         b_distance_to_p_when_a_at_p_m=tm["b_distance_to_p_when_a_at_p_m"],
                         a_distance_to_p_when_b_at_p_m=tm["a_distance_to_p_when_b_at_p_m"], sensitivity=sens),
        quality=dict(estimated_at_crossing_a=est["a"], estimated_at_crossing_b=est["b"],
                     observed_both_at_crossing=not (est["a"] or est["b"]), missing_sample_at_crossing=gap,
                     speed_a_m_s=m["motion_a"]["speed_m_s"], speed_b_m_s=m["motion_b"]["speed_m_s"],
                     local_sigma_a_m=m["motion_a"]["local_sigma_m"], local_sigma_b_m=m["motion_b"]["local_sigma_m"],
                     crossing_angle_deg=m["crossing_angle_deg"], timing_ill_conditioned=ill, vetoes=vet,
                     flags=[k for k in FLAGS if ck[k]], catalogue_quality=r["quality"]),
        provenance=dict(catalogue=CATALOGUE, eligibility=V3, advisor_recipe_agrees=r["advisor_recipe"]["agrees"]))


# ---------------------------------------------------------------- smoothing study

def smooth_segments(seg: dict[str, np.ndarray], gap_factor: float = 1.5) -> dict[str, np.ndarray]:
    """Derived copy of the segments with each uniformly sampled run's vertices smoothed (SG5 quadratic; the two vertices at each
    run end and runs shorter than five vertices unchanged). A run also ends at any segment longer than ``gap_factor`` native
    update intervals (a missing sample), so nothing is smoothed across a gap. Times, rows and the segments are untouched."""
    s = {k: v.copy() for k, v in seg.items()}
    n = len(seg["slot"])
    if not n:
        return s
    dur = seg["t1"] - seg["t0"]
    long = dur > gap_factor * float(np.median(dur))
    brk = np.r_[True, (seg["slot"][1:] != seg["slot"][:-1]) | (seg["row0"][1:] != seg["row1"][:-1]) | long[1:] | long[:-1]]
    starts = np.flatnonzero(brk)
    ends = np.r_[starts[1:], n] - 1
    for i, j in zip(starts, ends):
        if j - i + 2 < 5:
            continue
        for a0, a1 in (("x0", "x1"), ("y0", "y1")):
            v = np.r_[seg[a0][i:j + 1], seg[a1][j]]
            w = v.copy()
            w[2:-2] = np.convolve(v, SG5[::-1], mode="valid")
            s[a0][i:j + 1], s[a1][i:j + 1] = w[:-1], w[1:]
    return s


def light_events(seg: dict[str, np.ndarray], groups: list[list[dict]]) -> list[dict]:
    out = []
    for g in groups:
        ta = np.array([seg["t0"][h["ga"]] + h["ta"] * (seg["t1"][h["ga"]] - seg["t0"][h["ga"]]) for h in g])
        tb = np.array([seg["t0"][h["gb"]] + h["tb"] * (seg["t1"][h["gb"]] - seg["t0"][h["gb"]]) for h in g])
        k = int(np.argsort(ta)[len(ta) // 2])
        out.append(dict(period=g[k]["period"], sa=g[k]["sa"], sb=g[k]["sb"], x=g[k]["x"], y=g[k]["y"], ta=float(ta[k]), tb=float(tb[k])))
    return out


def match_events(raw: list[dict], sm: list[dict], cfg: CoincidenceConfig) -> dict[int, int]:
    """One-to-one raw -> smoothed matches: same pair and period, both times within ``match_t_s`` and P within ``match_p_m``."""
    by: dict[tuple, list[int]] = {}
    for j, e in enumerate(sm):
        by.setdefault((e["period"], e["sa"], e["sb"]), []).append(j)
    cand = []
    for i, r in enumerate(raw):
        for j in by.get((r["period"], r["slot_a"], r["slot_b"]), []):
            e = sm[j]
            da, db = abs(e["ta"] - r["time_a_s"]), abs(e["tb"] - r["time_b_s"])
            dp = float(np.hypot(e["x"] - r["x"], e["y"] - r["y"]))
            if da <= cfg.match_t_s and db <= cfg.match_t_s and dp <= cfg.match_p_m:
                cand.append((da + db, dp, i, j))
    out, used = {}, set()
    for _, _, i, j in sorted(cand):
        if i not in out and j not in used:
            out[i] = j
            used.add(j)
    return out


def smoothing_study(c, seg, periods: list[int], recs: list[dict], cfg: CoincidenceConfig) -> tuple[dict[str, Any], dict, list[dict], dict[int, int]]:
    dcfg = DetectConfig()
    ss = smooth_segments(seg)
    hits, _ = scan(c, ss, periods, dcfg)
    sm = light_events(ss, merge(unique_hits(hits), ss, dcfg))
    mt = match_events(recs, sm, cfg)
    h = recs[0]["coincidence"]["update_interval_s"] if recs else 1 / 15
    shift = np.array([[sm[j]["ta"] - recs[i]["time_a_s"], sm[j]["tb"] - recs[i]["time_b_s"],
                       abs(sm[j]["tb"] - sm[j]["ta"]) - recs[i]["coincidence"]["abs_dt_s"],
                       np.hypot(sm[j]["x"] - recs[i]["x"], sm[j]["y"] - recs[i]["y"])] for i, j in mt.items()]).reshape(-1, 4)
    dec = np.array([r["coincidence"]["decision"] for r in recs])
    out = dict(raw_candidates=len(recs), smoothed_candidates=len(sm), matched=len(mt), raw_only=len(recs) - len(mt), smoothed_only=len(sm) - len(mt),
               raw_only_by_decision={d: int(sum(dec[i] == d for i in range(len(recs)) if i not in mt)) for d in DECISIONS},
               matched_by_decision={d: int(sum(dec[i] == d for i in mt)) for d in DECISIONS},
               raw_within_h=int(sum(r["coincidence"]["abs_dt_s"] <= h for r in recs)),
               smoothed_within_h=int(sum(abs(e["tb"] - e["ta"]) <= h for e in sm)),
               accepted_still_within_h=int(sum(dec[i] == "accepted" and abs(sm[j]["tb"] - sm[j]["ta"]) <= h for i, j in mt.items())),
               rejected_now_within_h=int(sum(dec[i] == "rejected" and abs(sm[j]["tb"] - sm[j]["ta"]) <= h for i, j in mt.items())),
               matched_time_a_shift_s=_q(np.abs(shift[:, 0])), matched_abs_dt_change_s=_q(shift[:, 2]), matched_point_shift_m=_q(shift[:, 3]),
               matched_abs_dt_change_accepted_s=_q(np.array([shift[k, 2] for k, i in enumerate(mt) if dec[i] == "accepted"])))
    return out, ss, sm, mt


# ---------------------------------------------------------------- summaries

def _q(v, qs=(0, 5, 25, 50, 75, 95, 100)) -> dict[str, float] | None:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return {f"q{q}": round(float(np.quantile(v, q / 100)), 4) for q in qs} if len(v) else None


def summarize(recs: list[dict], cfg: CoincidenceConfig) -> dict[str, Any]:
    co = [r["coincidence"] for r in recs]
    qu = [r["quality"] for r in recs]
    out = dict(candidates=len(recs), decisions={d: sum(x["decision"] == d for x in co) for d in DECISIONS})
    reasons: dict[str, int] = {}
    for x in co:
        if x["decision"] == "ambiguous":
            for k in x["reasons"]:
                reasons[k] = reasons.get(k, 0) + 1
    out["ambiguous_reasons"] = dict(sorted(reasons.items()))
    out["timing_class"] = {k: sum(x["timing_class"] == k for x in co) for k in ("exact_equality", "same_sample_interval", "overlapping_intervals", "disjoint_intervals")}
    out["timing_class_by_decision"] = {d: {k: sum(x["timing_class"] == k and x["decision"] == d for x in co) for k in out["timing_class"]} for d in DECISIONS}
    out["observed_both_at_crossing_by_decision"] = {d: sum(q["observed_both_at_crossing"] and x["decision"] == d for x, q in zip(co, qu)) for d in DECISIONS}
    out["sensitivity"] = {k: {d: sum(x["sensitivity"][k] == d for x in co) for d in DECISIONS} for k in co[0]["sensitivity"]} if co else {}
    for d in DECISIONS:
        sub = [x for x in co if x["decision"] == d]
        out[f"{d}_abs_dt_s"] = _q([x["abs_dt_s"] for x in sub])
        out[f"{d}_sigma_dt_s"] = _q([x["sigma_dt_s"] for x in sub])
        out[f"{d}_other_player_distance_to_p_m"] = _q([max(v for v in (x["b_distance_to_p_when_a_at_p_m"], x["a_distance_to_p_when_b_at_p_m"]) if v is not None)
                                                      for x in sub if (x["b_distance_to_p_when_a_at_p_m"], x["a_distance_to_p_when_b_at_p_m"]) != (None, None)])
    acc = [r for r in recs if r["coincidence"]["decision"] == "accepted"]
    out["accepted_by_type"] = {k: sum(r["type"] == k for r in acc) for k in sorted({r["type"] for r in acc})}
    out["accepted_by_team_relation"] = {k: sum(r["team_relation"] == k for r in acc) for k in ("same_team", "opposite_teams")}
    out["accepted_distinct_pairs"] = len({(r["period"], r["slot_a"], r["slot_b"]) for r in acc})
    out["accepted_in_repeated_pair_cluster"] = sum(r["cluster_size"] > 1 for r in acc)
    return out


def trajectory_store(seg, key, recs: list[dict]) -> tuple[dict[str, np.ndarray], int]:
    """Before / After vertices (x, y, t) of both players for every accepted and ambiguous record, rebuilt from the RAW segments.
    ``event_index`` is the line number in ``root.events.jsonl``; ``part`` 0..3 = a_before, a_after, b_before, b_after."""
    ev, part, xs = [], [], []
    for k, r in enumerate(recs):
        if r["coincidence"]["decision"] == "rejected":
            continue
        p = paths(seg, r, key)
        for n, name in enumerate(("a_before", "a_after", "b_before", "b_after")):
            ev.append(k)
            part.append(n)
            xs.append(p[name])
    lens = np.array([len(v) for v in xs], np.int64)
    v = np.concatenate(xs) if xs else np.zeros((0, 3))
    return dict(event_index=np.array(ev, np.int64), part=np.array(part, np.int8), offsets=np.r_[0, np.cumsum(lens)].astype(np.int64),
                x=v[:, 0], y=v[:, 1], t=v[:, 2]), len(set(ev))


# ---------------------------------------------------------------- figures

def pick_examples(recs: list[dict]) -> dict[str, str]:
    """Deterministic: the accepted event with both players observed and with an ESTIMATED player (smallest |dt|), the ambiguous
    event with the largest noise band among |dt| <= tau, and the median sequential pass."""
    acc = [r for r in recs if r["coincidence"]["decision"] == "accepted"]
    out = {}
    obs = sorted((r for r in acc if r["quality"]["observed_both_at_crossing"]), key=lambda r: (r["coincidence"]["abs_dt_s"], r["event_id"]))
    est = sorted((r for r in acc if not r["quality"]["observed_both_at_crossing"]), key=lambda r: (r["coincidence"]["abs_dt_s"], r["event_id"]))
    if obs:
        out["accepted_observed"] = obs[0]["event_id"]
    if est:
        out["accepted_estimated"] = est[0]["event_id"]
    amb = sorted((r for r in recs if r["coincidence"]["decision"] == "ambiguous" and r["coincidence"]["abs_dt_s"] <= r["coincidence"]["tau_s"]),
                 key=lambda r: (-(r["coincidence"]["sigma_dt_s"] or 0), r["event_id"]))
    if amb:
        out["ambiguous_noise"] = amb[len(amb) // 2]["event_id"]
    rej = sorted((r for r in recs if r["coincidence"]["decision"] == "rejected"), key=lambda r: (r["coincidence"]["abs_dt_s"], r["event_id"]))
    if rej:
        out["rejected_sequential"] = rej[len(rej) // 2]["event_id"]
    return out


def smoothing_figure(seg, ss, key, r: dict, sm_event: dict | None) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    for s, col in (("a", "tab:blue"), ("b", "tab:red")):
        for data, ls, lab in ((seg, "-", "raw"), (ss, ":", "smoothed (derived)")):
            pr = paths(data, r, key)
            v = np.r_[pr[f"{s}_before"], pr[f"{s}_after"][1:]]
            ax.plot(v[:, 0], v[:, 1], ls, color=col, lw=1.8 if ls == "-" else 2.2, label=f"{s.upper()} {lab}")
    ax.plot(r["x"], r["y"], "ko", ms=7, label=f"raw P (dt={r['coincidence']['dt_s']:+.3f} s)")
    if sm_event is not None:
        ax.plot(sm_event["x"], sm_event["y"], "kx", ms=9, mew=2, label=f"smoothed P (dt={sm_event['tb'] - sm_event['ta']:+.3f} s)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m, RAW)")
    ax.set_ylabel("y (m, RAW)")
    ax.set_title(f"{r['match_id']} smoothing study: event {r['event_id']} ({r['coincidence']['decision']} on raw), "
                 f"{'kept' if sm_event is not None else 'no intersection'} after smoothing", fontsize=8)
    ax.legend(fontsize=7)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------- CLI

def build(raw_root: str, match_id: str, cfg: CoincidenceConfig, with_smoothing: bool) -> tuple[list[dict], dict[str, Any], dict[str, bytes], dict]:
    cat, elig, inputs = load_inputs(raw_root, match_id)
    recs = [evaluate(r, elig[r["event_id"]], cfg) for r in cat]
    summary = dict(schema=SCHEMA, match_id=int(match_id), config=cfg.as_dict(), dataset=summarize(recs, cfg))
    c, mask, _enc, src_manifest, prov = load_source(raw_root, match_id)
    steps = load_steps(raw_root, match_id)
    seg, _ = segments(c, mask, steps, duplicate_positions(c))
    key = segment_index(seg)
    store, n_store = trajectory_store(seg, key, recs)
    summary["trajectory_store"] = dict(events=n_store, vertices=int(store["offsets"][-1]))
    blobs = {"root.trajectories.npz": AU.npz_bytes(store)}
    by_id = {r["event_id"]: r for r in recs}
    cat_by_id = {r["event_id"]: r for r in cat}
    ex = pick_examples(recs)
    for name, eid in ex.items():
        r = by_id[eid]
        co = r["coincidence"]
        title = (f"{match_id} {name}: #{r['jersey_a']} (A) x #{r['jersey_b']} (B); |dt|={co['abs_dt_s']:.3f} s, tau={co['tau_s']:.3f} s, "
                 f"2 sigma={co['tol_s']:.3f} s; {co['decision']} {','.join(co['reasons'])}")
        blobs[f"root.fig-{name}.png"] = figure(c, seg, key, [cat_by_id[eid]], title)
    summary["examples"] = ex
    if with_smoothing:
        periods = sorted({r["period"] for r in recs})
        st, ss, sm, mt = smoothing_study(c, seg, periods, recs, cfg)
        summary["smoothing"] = st
        for i, r in enumerate(recs):                     # per-event jitter sensitivity (uncertainty flag, not a decision input)
            j = mt.get(i)
            r["quality"]["smoothing"] = dict(intersection_survives=j is not None,
                                             smoothed_abs_dt_s=None if j is None else abs(sm[j]["tb"] - sm[j]["ta"]))
        st["by_decision_survives"] = {d: sum(r["coincidence"]["decision"] == d and r["quality"]["smoothing"]["intersection_survives"] for r in recs)
                                      for d in DECISIONS}
        lost = sorted((i for i, r in enumerate(recs) if i not in mt and r["coincidence"]["decision"] == "accepted"), key=lambda i: recs[i]["event_id"])
        kept = sorted((i for i in mt if recs[i]["coincidence"]["decision"] == "accepted"), key=lambda i: recs[i]["event_id"])
        for name, ix, f in (("smoothing_removes_accepted", lost, lambda i: None), ("smoothing_keeps_accepted", kept, lambda i: sm[mt[i]])):
            if ix:
                blobs[f"root.fig-{name}.png"] = smoothing_figure(seg, ss, key, recs[ix[0]], f(ix[0]))
                summary["examples"][name] = recs[ix[0]]["event_id"]
    summary["inputs_sha256"] = inputs
    return recs, summary, blobs, dict(step1=prov, step1_schema=src_manifest["schema"])


def run_match(raw_root: str, match_id: str, cfg: CoincidenceConfig, dry_run: bool, with_smoothing: bool) -> dict[str, Any]:
    t0 = time.perf_counter()
    split = json.loads((approved_output_root(raw_root) / ml.SPLITS_DIR / "splits.json").read_text(encoding="utf-8"))
    validate.check_split(split, [str(match_id)], "development")
    out = approved_output_root(raw_root) / SCHEMA / str(match_id)
    if not dry_run and out.exists():
        raise IngestionError(f"{out} already exists and is never overwritten")
    snap0 = validate.raw_snapshot(raw_root)
    recs, summary, blobs, src = build(raw_root, match_id, cfg, with_smoothing)
    summary["seconds"] = round(time.perf_counter() - t0, 1)
    summary["peak_memory_mb"] = peak_memory_mb()
    if dry_run:
        return summary
    blobs = {"root.events.jsonl": jsonl(recs), "root.summary.json": json.dumps(summary, indent=1, sort_keys=True).encode(), **blobs}
    manifest = dict(schema=SCHEMA, match_id=int(match_id), split=split["matches"][str(match_id)], config=cfg.as_dict(),
                    code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))},
                    inputs_sha256=summary["inputs_sha256"], **src, raw_unchanged=snap0 == validate.raw_snapshot(raw_root),
                    coordinate_policy="raw (smoothing only in the comparison study, never in the dataset)",
                    outputs_sha256={n: hashlib.sha256(b).hexdigest() for n, b in blobs.items()}, python=platform.python_version(),
                    numpy=np.__version__,
                    note="Tracking-derived occlusion / encounter proxies: exact trajectory intersections with temporal coincidence within "
                         "the data's sampling resolution and measurement noise. Not verified camera occlusions; provider slots, not "
                         "verified identities.")
    blobs["root.manifest.json"] = json.dumps(manifest, indent=1, sort_keys=True).encode()
    write_retry(out, blobs)
    summary["written_to"] = str(out)
    return summary


def cross_match(raw_root: str, matches: list[str], dry_run: bool) -> dict[str, Any]:
    rows, sens, smo = {}, {}, {}
    for mid in matches:
        s = json.loads((approved_output_root(raw_root) / SCHEMA / mid / "root.summary.json").read_text(encoding="utf-8"))
        d = s["dataset"]
        rows[mid] = dict(candidates=d["candidates"], **d["decisions"],
                         accepted_observed_both=d["observed_both_at_crossing_by_decision"]["accepted"],
                         accepted_distinct_pairs=d["accepted_distinct_pairs"])
        for k, v in d["sensitivity"].items():
            for dd, n in v.items():
                sens.setdefault(k, dict.fromkeys(DECISIONS, 0))[dd] += n
        for k, v in (s.get("smoothing") or {}).items():
            if isinstance(v, int):
                smo[k] = smo.get(k, 0) + v
    tot = {k: sum(r[k] for r in rows.values()) for k in next(iter(rows.values()))}
    rep = dict(schema=SCHEMA, matches=sorted(rows), per_match=rows, totals=tot, sensitivity_totals=sens, smoothing_totals=smo)
    if not dry_run:
        write_retry(approved_output_root(raw_root) / SCHEMA / "cross-match", {"summary.json": json.dumps(rep, indent=1, sort_keys=True).encode()})
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m football_fatigue.intersections.spatiotemporal")
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--matches", nargs="+", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoothing", action="store_true", help="also run the raw vs smoothed comparison (derived copy)")
    ap.add_argument("--cross-match-only", action="store_true")
    a = ap.parse_args(argv)
    root = str(Path(a.root).resolve())
    cfg = CoincidenceConfig()
    try:
        if not a.cross_match_only:
            for mid in a.matches:
                s = run_match(root, mid, cfg, a.dry_run, a.smoothing)
                print(mid, json.dumps(dict(decisions=s["dataset"]["decisions"], ambiguous=s["dataset"]["ambiguous_reasons"],
                                           smoothing={k: v for k, v in (s.get("smoothing") or {}).items() if isinstance(v, int)},
                                           seconds=s["seconds"])), flush=True)
        else:
            print(json.dumps(cross_match(root, a.matches, a.dry_run), indent=1))
    except IngestionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
