"""Advisor occlusion-candidate catalogue (M11, Dr. Basha's extraction procedure): ``m11-occlusion-catalogue-v1``.

The advisor's procedure: every second, each player's trajectory from 2.5 s before to 2.5 s after; every unique pair of
trajectories goes to a curve-intersection algorithm; pairs without an intersection are discarded; the exact intersection
point P is kept, its time is read on each player's own trajectory, and repeats from overlapping windows are removed.

The detection is the unchanged v1 detector (:func:`.detect.detect`; same events as ``m11-intersections-v3``, checked). This
module only adds, per event, what is needed to reconstruct and audit it, and it applies **no** time or distance threshold:

- **window**: every detecting window centre and the canonical window (centre nearest to (t_A + t_B) / 2), plus a check that the
  recorded windows are exactly those whose +-2.5 s span contains both crossing segments (deduplication provenance);
- **trajectories**: for each player, the contiguous piece of its polyline in the canonical window that contains P, split at P
  into Before and After (source row bounds, so the RAW cache reconstructs the vertices exactly; :func:`paths`), with provider
  visibility (VISIBLE / ESTIMATED) at the crossing segment and the ESTIMATED share of the piece;
- **timing evidence**: t_A, t_B and their difference; who reached P first; a numerical tie (``|t_A - t_B| <= TIE_S``, floating
  point only); the overlap of the two crossing segments' time intervals (the data's own sampling resolution: when they overlap,
  both players were on their crossing segments at the same time, so the order of arrival is below the sampling resolution); and
  the other player's distance from P at the moment each player is at P (interpolated along its piece);
- **advisor recipe**: t_A, t_B by ``project`` / cumulative length / ``searchsorted`` (:mod:`.advisor`) on the canonical pieces,
  and whether they agree with the segment-indexed times (they differ only when ``project`` picks another pass of a loop).

The v1 ``simultaneous`` field (|dt| <= 0.2 s) is not carried over; no event is removed for its timing. Development matches only.
Run: ``python -m football_fatigue.intersections.catalogue --matches 3812 [...] [--dry-run] [--sensitivity]``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from football_fatigue.encounters.trajectories import RawCache
from football_fatigue.ingestion import EventVersion, IngestionError, approved_output_root, safe_output_path
from football_fatigue.matching import ml, validate
from football_fatigue.matching.benchmark import SOURCE_DIR, jsonl, load_source

from .__main__ import peak_memory_mb, write_dir
from .advisor import advisor_times, run_bounds
from .detect import DetectConfig, detect, duplicate_positions, merge, period_windows, scan, segments, unique_hits

SCHEMA = "m11-occlusion-catalogue-v1"
TIE_S = 1e-9          # numerical equality of two interpolated times (timestamps carry 1e-6 s; float64 rounding here is ~1e-12 s)
PAIRS_22 = math.comb(22, 2)


def load_steps(raw_root: str, match_id: str) -> dict[str, np.ndarray]:
    with np.load(safe_output_path(raw_root, match_id, EventVersion.ROOT, approved_output_root(raw_root) / SOURCE_DIR, suffix=".quality.npz")) as z:
        return {k[5:]: z[k] for k in z.files if k.startswith("step_")}


def piece(seg: dict[str, np.ndarray], g: int, u: float, lo: float, hi: float) -> dict[str, np.ndarray]:
    """The run of segments containing ``g`` inside [lo, hi], split at the point at parameter ``u`` of segment ``g``:
    ``before`` and ``after`` are (n, 3) arrays of x, y, t, both containing P (the last and the first vertex respectively)."""
    i, j = run_bounds(seg, g, lo, hi)
    px = seg["x0"][g] + u * (seg["x1"][g] - seg["x0"][g])
    py = seg["y0"][g] + u * (seg["y1"][g] - seg["y0"][g])
    pt = seg["t0"][g] + u * (seg["t1"][g] - seg["t0"][g])
    bi, ai = np.arange(i, g + 1), np.arange(g, j + 1)
    before = np.c_[np.r_[seg["x0"][bi], px], np.r_[seg["y0"][bi], py], np.r_[seg["t0"][bi], pt]]
    after = np.c_[np.r_[px, seg["x1"][ai]], np.r_[py, seg["y1"][ai]], np.r_[pt, seg["t1"][ai]]]
    return dict(before=before, after=after, i=i, j=j)


def position_at(before: np.ndarray, after: np.ndarray, t: float) -> np.ndarray | None:
    """Position on the piece at time ``t`` (linear between update instants), None outside the piece."""
    v = np.r_[before, after[1:]]
    if not v[0, 2] - 1e-9 <= t <= v[-1, 2] + 1e-9:
        return None
    return np.array([np.interp(t, v[:, 2], v[:, 0]), np.interp(t, v[:, 2], v[:, 1])])


def segment_index(seg: dict[str, np.ndarray]) -> dict[tuple[int, int], int]:
    return {(int(seg["slot"][g]), int(seg["row0"][g])): g for g in range(len(seg["slot"]))}


def paths(seg: dict[str, np.ndarray], rec: dict[str, Any], key: dict[tuple[int, int], int] | None = None) -> dict[str, np.ndarray]:
    """Before / After vertices (x, y, t) of both players of a catalogue record, rebuilt from the source segments."""
    key = key if key is not None else segment_index(seg)
    w = rec["window"]
    out = {}
    for side, slot in (("a", rec["slot_a"]), ("b", rec["slot_b"])):
        s = rec[f"segment_{side}"]
        pc = piece(seg, key[(slot, s["row0"])], s["u"], w["start_s"], w["end_s"])
        out[f"{side}_before"], out[f"{side}_after"] = pc["before"], pc["after"]
    return out


def describe(c: RawCache, seg: dict[str, np.ndarray], key: dict, e: dict[str, Any], centres: np.ndarray, half: float) -> dict[str, Any]:
    """Catalogue record of one v1 event (see the module docstring)."""
    ga, gb = key[(e["slot_a"], e["segment_a"]["row0"])], key[(e["slot_b"], e["segment_b"]["row0"])]
    tA, tB = e["time_a_s"], e["time_b_s"]
    mid = (tA + tB) / 2
    wins = np.array(e["windows"])
    w = float(centres[np.argmin(np.abs(centres - wins[np.argmin(np.abs(wins - mid))]))])     # unrounded centre
    lo, hi = w - half, w + half
    pa = piece(seg, ga, e["segment_a"]["u"], lo, hi)
    pb = piece(seg, gb, e["segment_b"]["u"], lo, hi)
    span_lo, span_hi = min(seg["t0"][ga], seg["t0"][gb]), max(seg["t1"][ga], seg["t1"][gb])
    expected = [round(float(x), 3) for x in centres if x - half <= span_lo + 1e-9 and span_hi <= x + half + 1e-9]
    consistent = expected == e["windows"] if e["n_members"] == 1 else set(expected) <= set(e["windows"])
    ovl = float(min(seg["t1"][ga], seg["t1"][gb]) - max(seg["t0"][ga], seg["t0"][gb]))
    p = np.array([e["x"], e["y"]])
    qb, qa = position_at(pb["before"], pb["after"], tA), position_at(pa["before"], pa["after"], tB)
    va = np.r_[pa["before"], pa["after"][1:]]
    vb = np.r_[pb["before"], pb["after"][1:]]
    xa, xb = advisor_times(va[:, :2], va[:, 2], vb[:, :2], vb[:, 2], p)
    dt = tB - tA

    def traj(pc: dict, g: int, slot: int) -> dict[str, Any]:
        rows = np.r_[seg["row0"][pc["i"]:pc["j"] + 1], seg["row1"][pc["j"]]]
        vis = c.visibility[rows, slot]
        return dict(piece_first_row=int(rows[0]), piece_last_row=int(rows[-1]), n_vertices_before=len(pc["before"]) - 1,
                    n_vertices_after=len(pc["after"]) - 1, start_s=float(pc["before"][0, 2]), end_s=float(pc["after"][-1, 2]),
                    covers_window=bool(pc["before"][0, 2] - lo < 0.2 and hi - pc["after"][-1, 2] < 0.2),
                    visibility_at_crossing=[int(c.visibility[seg["row0"][g], slot]), int(c.visibility[seg["row1"][g], slot])],
                    estimated_share=float(np.mean(vis == 2)))

    rec = {k: v for k, v in e.items() if k != "simultaneous"}
    rec.update(
        timing=dict(t_a_s=tA, t_b_s=tB, dt_s=dt, abs_dt_s=abs(dt), first="tie" if abs(dt) <= TIE_S else ("a" if dt > 0 else "b"),
                    numerical_tie=abs(dt) <= TIE_S, crossing_segments_overlap_s=ovl, concurrent_crossing_segments=ovl >= 0,
                    b_distance_to_p_when_a_at_p_m=None if qb is None else float(np.hypot(*(qb - p))),
                    a_distance_to_p_when_b_at_p_m=None if qa is None else float(np.hypot(*(qa - p)))),
        advisor_recipe=dict(t_a_s=xa, t_b_s=xb, agrees=max(abs(xa - tA), abs(xb - tB)) <= 1e-6),
        window=dict(canonical_centre_s=round(w, 3), start_s=lo, end_s=hi, centres=e["windows"], n_windows=e["n_windows"],
                    expected_centres=expected, consistent=bool(consistent)),
        dedup=dict(segment_pair_hits=e["n_members"], windows=e["n_windows"],
                   rule="one hit per segment pair over all windows; hits on adjacent segments at the same point (<= 1 mm) or one "
                        "collinear overlap form one event; separate crossings of a pair stay separate"),
        trajectories=dict(a=traj(pa, ga, e["slot_a"]), b=traj(pb, gb, e["slot_b"])))
    return rec


def accounting(c: RawCache, stats: dict[str, Any], periods: list[int], cfg: DetectConfig) -> dict[str, Any]:
    per = {}
    for p in periods:
        t = c.video_time_s[c.canonical_period == p]
        w = period_windows(c, p, cfg)
        per[str(p)] = dict(start_s=float(t.min()), end_s=float(t.max()), duration_s=round(float(t.max() - t.min()), 3), windows=len(w),
                           uncovered_tail_s=round(float(t.max() - (w[-1] + cfg.half)), 3) if len(w) else None)
    n = stats["windows"]
    return dict(periods=per, windows=n, theoretical_pair_windows_22_players=PAIRS_22 * n, pair_windows_checked=stats["pair_checks"],
                available_players_per_window=stats["available_players_per_window"], pair_windows_bbox_rejected=stats["pair_bbox_rejected"],
                pair_windows_with_intersection=stats["pair_windows_with_hit"], window_segment_pair_hits=stats["segment_pair_hits_per_window"],
                unique_segment_pair_hits=stats["unique_segment_pair_hits"], events=stats["unique_events"],
                hits_merged_into_other_events=stats["hits_merged_into_other_events"], segments=stats["segments"])


def _q(v) -> dict[str, float] | None:
    v = np.asarray([x for x in v if x is not None], float)
    return {f"q{q}": round(float(np.quantile(v, q / 100)), 4) for q in (0, 1, 5, 10, 25, 50, 75, 90, 100)} if len(v) else None


def summarize(recs: list[dict]) -> dict[str, Any]:
    tm = [r["timing"] for r in recs]
    n = len(recs)
    closest = [min(x for x in (t["b_distance_to_p_when_a_at_p_m"], t["a_distance_to_p_when_b_at_p_m"]) if x is not None)
               if (t["b_distance_to_p_when_a_at_p_m"], t["a_distance_to_p_when_b_at_p_m"]) != (None, None) else None for t in tm]
    return dict(events=n, by_type={k: sum(r["type"] == k for r in recs) for k in sorted({r["type"] for r in recs})},
                abs_dt_s=_q([t["abs_dt_s"] for t in tm]), numerical_ties=sum(t["numerical_tie"] for t in tm),
                concurrent_crossing_segments=sum(t["concurrent_crossing_segments"] for t in tm),
                other_player_distance_to_p_m=_q(closest),
                advisor_recipe_agrees=sum(r["advisor_recipe"]["agrees"] for r in recs),
                windows_consistent=sum(r["window"]["consistent"] for r in recs),
                windows_per_event={str(k): sum(r["n_windows"] == k for r in recs) for k in sorted({r["n_windows"] for r in recs})},
                both_pieces_cover_window=sum(r["trajectories"]["a"]["covers_window"] and r["trajectories"]["b"]["covers_window"] for r in recs),
                repeated_pair_events=sum(r["cluster_size"] > 1 for r in recs))


def sensitivity(c: RawCache, mask: np.ndarray, steps: dict[str, np.ndarray], periods: list[int], cfg: DetectConfig) -> dict[str, int]:
    """Events if the Step 1 validity exclusions were dropped one at a time (counts only; nothing is written as events)."""
    out = {}
    dup = duplicate_positions(c)
    no = np.zeros_like(steps["jump"])
    for name, st, d in (("as_built", steps, dup), ("implausible_jumps_joined", {**steps, "jump": no}, dup),
                        ("duplicate_positions_joined", steps, np.zeros_like(dup))):
        seg, _ = segments(c, mask, st, d)
        hits, _ = scan(c, seg, periods, cfg)
        out[name] = len(merge(unique_hits(hits), seg, cfg))
    return out


def build(raw_root: str, match_id: str, with_sensitivity: bool = False) -> tuple[list[dict], dict[str, Any], dict, Any]:
    c, mask, _enc, src_manifest, prov = load_source(raw_root, match_id)
    steps = load_steps(raw_root, match_id)
    cfg = DetectConfig()
    events, stats = detect(c, mask, steps, cfg, quarantine=tuple(c.quarantine_periods))
    seg, _ = segments(c, mask, steps, duplicate_positions(c))
    key = segment_index(seg)
    centres = {p: period_windows(c, p, cfg) for p in stats["periods_scanned"]}
    recs = [describe(c, seg, key, e, centres[e["period"]], cfg.half) for e in events]
    summary = dict(schema=SCHEMA, match_id=int(match_id), accounting=accounting(c, stats, stats["periods_scanned"], cfg), catalogue=summarize(recs),
                   periods_quarantined=stats["periods_quarantined"])
    v3 = approved_output_root(raw_root) / "m11-intersections-v3" / str(match_id) / "root.events.jsonl"
    summary["v3_identical"] = (jsonl(events) == v3.read_bytes()) if v3.exists() else None
    if summary["v3_identical"] is False:
        raise IngestionError(f"match {match_id}: detection does not reproduce the m11-intersections-v3 events")
    if with_sensitivity:
        summary["validity_sensitivity_events"] = sensitivity(c, mask, steps, stats["periods_scanned"], cfg)
    return recs, summary, dict(step1=prov, step1_schema=src_manifest["schema"]), (c, seg, key)


# ---------------------------------------------------------------- examples

def pick_examples(recs: list[dict]) -> dict[str, str]:
    """Deterministic illustrations: the closest-to-simultaneous and a median-|dt| proper crossing (clean windows, no quality flag),
    a pair with three crossings, and a loop where ``project`` picks another pass."""
    ok = [r for r in recs if r["type"] == "proper_crossing" and not any(r["quality"].values())
          and r["trajectories"]["a"]["covers_window"] and r["trajectories"]["b"]["covers_window"]]
    out = {}
    if ok:
        out["closest_to_same_time"] = min(ok, key=lambda r: (r["timing"]["abs_dt_s"], r["event_id"]))["event_id"]
        med = float(np.median([r["timing"]["abs_dt_s"] for r in recs]))
        out["median_time_difference"] = min(ok, key=lambda r: (abs(r["timing"]["abs_dt_s"] - med), r["event_id"]))["event_id"]
    tri = sorted((r for r in recs if r["cluster_size"] == 3), key=lambda r: (r["cluster_id"], r["timing"]["t_a_s"]))
    if tri:
        out["repeated_crossings_of_one_pair"] = tri[0]["cluster_id"]
    loop = sorted((r for r in recs if not r["advisor_recipe"]["agrees"]), key=lambda r: r["event_id"])
    if loop:
        out["project_picks_other_pass"] = loop[0]["event_id"]
    return out


def figure(c: RawCache, seg, key, recs: list[dict], title: str) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for n, r in enumerate(recs):
        pt = paths(seg, r, key)
        for side, col in (("a", "tab:blue"), ("b", "tab:red")):
            b, a = pt[f"{side}_before"], pt[f"{side}_after"]
            lab = n == 0
            ax.plot(b[:, 0], b[:, 1], "-", color=col, lw=2, label=f"{side.upper()} before" if lab else None)
            ax.plot(a[:, 0], a[:, 1], "--", color=col, lw=2, label=f"{side.upper()} after" if lab else None)
            ax.plot(*b[0, :2], "o", color=col, ms=4)
            ax.plot(*a[-1, :2], ">", color=col, ms=6)
        tm = r["timing"]
        ax.plot(r["x"], r["y"], "ko", ms=7, label="intersection P" if n == 0 else None)
        ax.annotate(f"P{n + 1}: tA={tm['t_a_s']:.3f} s, tB={tm['t_b_s']:.3f} s, dt={tm['dt_s']:+.3f} s", (r["x"], r["y"]),
                    textcoords="offset points", xytext=(8, -14 - 12 * n), fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m, RAW)")
    ax.set_ylabel("y (m, RAW)")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8, loc="best")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def example_blobs(c, seg, key, recs: list[dict]) -> tuple[dict[str, bytes], list[dict]]:
    by_id = {r["event_id"]: r for r in recs}
    blobs, meta = {}, []
    for name, ref in pick_examples(recs).items():
        rs = [r for r in recs if r["cluster_id"] == ref] if name == "repeated_crossings_of_one_pair" else [by_id[ref]]
        r0 = rs[0]
        title = (f"{c.match_id} {name}: slots {r0['slot_a']} (A, #{r0['jersey_a']}) x {r0['slot_b']} (B, #{r0['jersey_b']}), "
                 f"window {r0['window']['canonical_centre_s']:.1f} s +- 2.5 s")
        blobs[f"root.fig-{name}.png"] = figure(c, seg, key, rs, title)
        pts = [paths(seg, r, key) for r in rs]
        meta.append(dict(name=name, file=f"root.fig-{name}.png", event_ids=[r["event_id"] for r in rs],
                         vertices={r["event_id"]: {k: np.round(v, 4).tolist() for k, v in p.items()} for r, p in zip(rs, pts)}))
    return blobs, meta


# ---------------------------------------------------------------- CLI

def run_match(raw_root: str, match_id: str, dry_run: bool, with_sensitivity: bool) -> dict[str, Any]:
    t0 = time.perf_counter()
    split = json.loads((approved_output_root(raw_root) / ml.SPLITS_DIR / "splits.json").read_text(encoding="utf-8"))
    validate.check_split(split, [str(match_id)], "development")
    out = approved_output_root(raw_root) / SCHEMA / str(match_id)
    if not dry_run and out.exists():
        raise IngestionError(f"{out} already exists and is never overwritten")
    snap0 = validate.raw_snapshot(raw_root)
    recs, summary, src, (c, seg, key) = build(raw_root, match_id, with_sensitivity)
    figs, examples = example_blobs(c, seg, key, recs)
    summary["examples"] = [{k: v for k, v in m.items() if k != "vertices"} for m in examples]
    summary["seconds"] = round(time.perf_counter() - t0, 1)
    summary["peak_memory_mb"] = peak_memory_mb()
    if dry_run:
        return summary
    blobs = {"root.catalogue.jsonl": jsonl(recs), "root.examples.json": json.dumps(examples, indent=1).encode(),
             "root.summary.json": json.dumps(summary, indent=1, sort_keys=True).encode(), **figs}
    snap1 = validate.raw_snapshot(raw_root)
    manifest = dict(schema=SCHEMA, match_id=int(match_id), split=split["matches"][str(match_id)], detect_config=DetectConfig().as_dict(),
                    tie_s=TIE_S, code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))},
                    **src, raw_unchanged=snap0 == snap1, coordinate_policy="raw",
                    outputs_sha256={n: hashlib.sha256(b).hexdigest() for n, b in blobs.items()}, python=platform.python_version(), numpy=np.__version__,
                    note="Geometric trajectory intersections (occlusion candidates by the advisor's definition); identity-labelled provider "
                         "slots, not verified identities; no time or distance threshold selects events.")
    blobs["root.manifest.json"] = json.dumps(manifest, indent=1, sort_keys=True).encode()
    write_retry(out, blobs)
    summary["written_to"] = str(out)
    return summary


def write_retry(out: Path, blobs: dict[str, bytes], attempts: int = 5) -> None:
    """:func:`write_dir`, retried when Windows denies the final rename (a scanner holding a fresh file; the partial directory
    is removed by ``write_dir`` before each retry)."""
    for k in range(attempts):
        try:
            return write_dir(out, blobs)
        except PermissionError:
            if k == attempts - 1:
                raise
            time.sleep(3.0 * (k + 1))


def cross_match(raw_root: str, summaries: dict[str, dict], dry_run: bool) -> dict[str, Any]:
    rows = {}
    for mid, s in summaries.items():
        a, k = s["accounting"], s["catalogue"]
        rows[mid] = dict(windows=a["windows"], theoretical_pair_windows_22_players=a["theoretical_pair_windows_22_players"],
                         pair_windows_checked=a["pair_windows_checked"], pair_windows_with_intersection=a["pair_windows_with_intersection"],
                         window_segment_pair_hits=a["window_segment_pair_hits"], unique_segment_pair_hits=a["unique_segment_pair_hits"],
                         events=k["events"], numerical_ties=k["numerical_ties"], concurrent_crossing_segments=k["concurrent_crossing_segments"],
                         advisor_recipe_agrees=k["advisor_recipe_agrees"], windows_consistent=k["windows_consistent"],
                         median_abs_dt_s=k["abs_dt_s"]["q50"], v3_identical=s["v3_identical"])
    tot = {k: sum(r[k] for r in rows.values()) for k in next(iter(rows.values())) if k not in ("median_abs_dt_s", "v3_identical")}
    rep = dict(schema=SCHEMA, matches=sorted(rows), per_match=rows, totals=tot)
    if not dry_run:
        write_retry(approved_output_root(raw_root) / SCHEMA / "cross-match", {"summary.json": json.dumps(rep, indent=1, sort_keys=True).encode()})
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m football_fatigue.intersections.catalogue")
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--matches", nargs="+", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sensitivity", action="store_true", help="also count events without each Step 1 validity exclusion")
    ap.add_argument("--no-cross-match", action="store_true")
    ap.add_argument("--cross-match-only", action="store_true", help="only build the cross-match summary from written match summaries")
    a = ap.parse_args(argv)
    root = str(Path(a.root).resolve())
    summaries = {}
    try:
        for mid in a.matches if not a.cross_match_only else []:
            s = summaries[mid] = run_match(root, mid, a.dry_run, a.sensitivity)
            print(mid, json.dumps(dict(accounting={k: v for k, v in s["accounting"].items() if k not in ("periods", "segments")},
                                       catalogue=s["catalogue"], sensitivity=s.get("validity_sensitivity_events"),
                                       v3_identical=s["v3_identical"], seconds=s["seconds"])), flush=True)
        if a.cross_match_only:
            summaries = {mid: json.loads((approved_output_root(root) / SCHEMA / mid / "root.summary.json").read_text(encoding="utf-8"))
                         for mid in a.matches}
        if not a.no_cross_match:
            print(json.dumps(cross_match(root, summaries, a.dry_run)["totals"], indent=1))
    except IngestionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
