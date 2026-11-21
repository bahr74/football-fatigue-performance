"""Development encounter dataset over ``m11-sync-proximity-v2`` outputs: integrity validation, quality subsets, index (M11).

Reads only the written per-match outputs (never raw data; ``--determinism`` re-renders named matches from their Step 1 caches) and
writes ``<processed>/m11-sync-proximity-v2/development-index/``:

- ``encounters_index.csv``: one row per primary (0.25 m) episode, with identities, timing, quality, context completeness, subset
  membership and the location of its record (events file and line) and trajectory arrays (npz file and event index);
- ``subset_<name>.csv``: (match_id, event_id) of each subset; every subset is a view of the full population, never a replacement;
- ``match_summary.csv``: one row per match; ``summary.json``: quality-stratified totals; ``validation.json``: every integrity check
  per match; ``schema.json``: field definitions; ``manifest.json``: inputs, code and output hashes.

Subsets (metadata views; no new thresholds):

- ``all_encounters``: every 0.25 m episode;
- ``observed_observed`` / ``mixed_quality`` / ``estimated_estimated``: both players VISIBLE at t* / one / neither (status of the two
  samples bracketing t* for each player);
- ``non_coincident``: not a coincident interval (d = 0 over an interval: cloned provider estimates);
- ``high_quality_candidates``: observed/observed, non-coincident, no duplicated provider position in the episode, no support cut at
  the episode, both players' valid trajectories covering the whole canonical 5 s window, and no critical Step 1 flag (identity or
  integrity: implausible jump, label transfer, reciprocal exchange, unresolved attribution, ambiguous / missing raw, side
  over-count, frame-wide shift, discontinuity) on any context sample.

Run: ``python -m football_fatigue.intersections.proximity_dataset --matches <development ids> [--determinism 3812 ...]``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from football_fatigue.ingestion import IngestionError, approved_output_root
from football_fatigue.matching import ml, validate

from . import proximity as P
from .catalogue import write_retry

OUT = "development-index"
PRIMARY = "0.25m"
MAX_STEP_S = 0.5 + 1 / 29.97 + 1e-6          # longest valid moving segment (Step 1 max_gap_s plus one frame)
REQUIRED = ("event_id", "event_key", "schema", "threshold_m", "match_id", "period", "pair_id", "player_a", "player_b", "team_relation",
            "t_start_s", "t_end_s", "duration_s", "t_star_s", "d_min_m", "midpoint", "contact_class", "sample_instants_within",
            "exact_coincidence", "coincident_interval", "pair_quality", "support_cut_at_start", "support_cut_at_end", "windows_game_s",
            "canonical_window_game_s", "period_start_s", "period_end_s")
PLAYER = ("slot", "side", "team_id", "jersey", "player_id", "status_at_t_star", "quality_flags", "duplicate_provider_position",
          "segment_at_t_star", "position_at_t_star")


def dup_pos(r: dict) -> bool:
    return r["player_a"]["duplicate_provider_position"] or r["player_b"]["duplicate_provider_position"]


SUBSETS = {
    "all_encounters": lambda r: True,
    "observed_observed": lambda r: r["pair_quality"] == "observed/observed",
    "mixed_quality": lambda r: r["pair_quality"] == "observed/estimated",
    "estimated_estimated": lambda r: r["pair_quality"] == "estimated/estimated",
    "non_coincident": lambda r: not r["coincident_interval"],
    "high_quality_candidates": lambda r: (r["pair_quality"] == "observed/observed" and not r["coincident_interval"] and not dup_pos(r)
                                          and not r["support_cut_at_start"] and not r["support_cut_at_end"] and r["context"]["complete"]
                                          and not r["context"]["critical_flags"]),
}


def load(match_dir: Path, t: str) -> list[dict]:
    return [json.loads(x) for x in (match_dir / f"root.events-{t}.jsonl").read_text(encoding="utf-8").splitlines()]


# ---------------------------------------------------------------- integrity validation

def validate_match(match_dir: Path, cfg: P.ProximityConfig = P.ProximityConfig()) -> dict[str, Any]:
    """Every integrity check of one match; ``violations`` maps check name -> number of failing items (0 = pass)."""
    v: dict[str, int] = {}

    def check(name: str, bad: int) -> None:
        v[name] = v.get(name, 0) + int(bad)

    why = P.verify_existing(match_dir, cfg)
    check("manifest_hashes_schema_config_code", why is not None)
    mid = int(match_dir.name)
    ts = [f"{r:g}m" for r in cfg.thresholds_m]
    recs = {t: load(match_dir, t) for t in ts}
    ids = {t: {r["event_id"] for r in recs[t]} for t in ts}
    for t in ts:
        R = float(t[:-1])
        check("unique_event_ids", len(recs[t]) - len(ids[t]))
        for r in recs[t]:
            check("required_fields", sum(k not in r for k in REQUIRED) + sum(k not in r[p] for p in ("player_a", "player_b") for k in PLAYER))
            check("stable_event_id", hashlib.sha256(r["event_key"].encode()).hexdigest()[:16] != r["event_id"])
            a, b = r["player_a"], r["player_b"]
            check("match_identity", r["match_id"] != mid or r["pair_id"] != f"{mid}|{r['period']}|{a['slot']}|{b['slot']}" or a["slot"] >= b["slot"])
            check("team_relation", r["team_relation"] != ("same_team" if a["side"] == b["side"] else "opposite_teams"))
            check("timing_order", not (r["period_start_s"] - 1e-9 <= r["t_start_s"] <= r["t_star_s"] <= r["t_end_s"] <= r["period_end_s"] + 1e-9))
            check("duration", r["duration_s"] < 0 or abs(r["duration_s"] - (r["t_end_s"] - r["t_start_s"])) > 1e-9)
            check("d_min_within_threshold", not (0 <= r["d_min_m"] <= R + 1e-12))
            check("midpoint", max(abs(r["midpoint"][k] - (a["position_at_t_star"][k] + b["position_at_t_star"][k]) / 2) for k in (0, 1)) > 1e-9)
            check("d_min_equals_positions", abs(np.hypot(a["position_at_t_star"][0] - b["position_at_t_star"][0],
                                                         a["position_at_t_star"][1] - b["position_at_t_star"][1]) - r["d_min_m"]) > 1e-6)
        by_pair: dict[str, list[dict]] = {}
        for r in recs[t]:
            by_pair.setdefault(r["pair_id"], []).append(r)
        for eps in by_pair.values():                                     # deduplication: one pair's episodes are disjoint, never touching
            eps.sort(key=lambda r: r["t_start_s"])
            check("episodes_disjoint_per_pair", sum(n["t_start_s"] <= p["t_end_s"] for p, n in zip(eps, eps[1:])))
    for t, t2 in zip(ts, ts[1:]):                                        # nesting
        par = {r["event_id"]: r for r in recs[t2]}
        for r in recs[t]:
            p = par.get(r.get("parent_event_id"))
            check("nesting", p is None or p["pair_id"] != r["pair_id"] or not (p["t_start_s"] <= r["t_start_s"] + 1e-9 and p["t_end_s"] >= r["t_end_s"] - 1e-9))
    prim = recs[PRIMARY]
    with np.load(match_dir / "root.trajectories.npz") as z:
        Z = {k: z[k] for k in z.files}
    n = len(prim)
    check("npz_event_coverage", set(np.unique(Z["sample_event_index"]).tolist()) != set(range(n))
          or set(np.unique(Z["sync_event_index"]).tolist()) != set(range(n)))
    so, yo = np.argsort(Z["sample_event_index"], kind="stable"), np.argsort(Z["sync_event_index"], kind="stable")
    sb = np.searchsorted(Z["sample_event_index"][so], np.arange(n + 1))
    yb = np.searchsorted(Z["sync_event_index"][yo], np.arange(n + 1))
    for k, r in enumerate(prim):
        cx = r.get("context")
        check("context_fields", cx is None)
        if cx is None:
            continue
        s, y = so[sb[k]:sb[k + 1]], yo[yb[k]:yb[k + 1]]
        lo, hi = cx["window_start_s"], cx["window_end_s"]
        for side, key in ((0, "player_a"), (1, "player_b")):
            w = s[Z["sample_side"][s] == side]
            t = Z["sample_t"][w]
            check("samples_in_window", ((t < lo - 1e-9) | (t > hi + 1e-9)).sum())
            check("samples_in_period", ((t < r["period_start_s"] - 1e-9) | (t > r["period_end_s"] + 1e-9)).sum())
            check("samples_time_ordered", (np.diff(t) <= 0).sum())
            dt = np.diff(t)
            same = (np.diff(Z["sample_x"][w]) == 0) & (np.diff(Z["sample_y"][w]) == 0)
            check("no_interpolation_through_gaps", ((dt > MAX_STEP_S) & ~same).sum())     # long steps only where the player is held
            check("source_rows", ((Z["sample_source_row"][w] < 0) | (Z["sample_frame"][w] < 0)).sum())
            check("visibility_codes", (~np.isin(Z["sample_visibility"][w], (1, 2))).sum())
            check("before_after_split", (Z["sample_before_t_star"][w] != (t < r["t_star_s"])).sum())
            info = cx[key]
            check("context_matches_samples", len(w) != info["n_samples"] or not len(w)
                  or t[0] != info["first_sample_s"] or t[-1] != info["last_sample_s"])
            seg = r[key]["segment_at_t_star"]
            check("segment_at_t_star_in_samples", not (np.isin(seg["row0"], Z["sample_source_row"][w]) and np.isin(seg["row1"], Z["sample_source_row"][w])))
        ty = Z["sync_t"][y]
        i = np.flatnonzero(np.abs(ty - r["t_star_s"]) <= 1e-9)
        check("sync_has_t_star", len(i) == 0)
        if len(i):
            j = y[i[0]]
            check("sync_d_at_t_star", abs(Z["sync_d"][j] - r["d_min_m"]) > 1e-6)
            check("sync_positions_at_t_star", max(abs(Z["sync_xa"][j] - r["player_a"]["position_at_t_star"][0]), abs(Z["sync_ya"][j] - r["player_a"]["position_at_t_star"][1]),
                                                  abs(Z["sync_xb"][j] - r["player_b"]["position_at_t_star"][0]), abs(Z["sync_yb"][j] - r["player_b"]["position_at_t_star"][1])) > 1e-6)
        inside = (ty >= r["t_start_s"] - 1e-9) & (ty <= r["t_end_s"] + 1e-9)
        check("sync_within_threshold_inside_episode", (Z["sync_d"][y][inside] > float(PRIMARY[:-1]) + 1e-6).sum())
        check("sync_time_ordered", (np.diff(ty) <= 0).sum())
    return dict(match_id=mid, events={t: len(recs[t]) for t in ts}, violations=v, passed=not any(v.values()))


def determinism(raw_root: str, match_id: str) -> dict[str, Any]:
    """Re-render one match from its Step 1 cache and compare every hashed output with the written manifest."""
    out = approved_output_root(raw_root) / P.SCHEMA / str(match_id)
    m = json.loads((out / "root.manifest.json").read_text(encoding="utf-8"))
    blobs, _ = P.render(raw_root, match_id)
    diff = [n for n, b in blobs.items() if hashlib.sha256(b).hexdigest() != m["outputs_sha256"].get(n)]
    return dict(match_id=int(match_id), outputs=len(blobs), identical=not diff, differing=diff)


# ---------------------------------------------------------------- index and summaries

def index_row(r: dict, line: int, rel: str) -> dict[str, Any]:
    a, b, cx = r["player_a"], r["player_b"], r["context"]
    row = dict(match_id=r["match_id"], event_id=r["event_id"], period=r["period"], pair_id=r["pair_id"])
    for s, p in (("a", a), ("b", b)):
        row.update({f"slot_{s}": p["slot"], f"player_id_{s}": p["player_id"], f"jersey_{s}": p["jersey"], f"side_{s}": p["side"],
                    f"team_id_{s}": p["team_id"], f"status_{s}": p["status_at_t_star"], f"x_{s}": p["position_at_t_star"][0],
                    f"y_{s}": p["position_at_t_star"][1]})
    row.update(team_relation=r["team_relation"], t_start_s=r["t_start_s"], t_end_s=r["t_end_s"], duration_s=r["duration_s"], t_star_s=r["t_star_s"],
               d_min_m=r["d_min_m"], midpoint_x=r["midpoint"][0], midpoint_y=r["midpoint"][1], contact_class=r["contact_class"],
               pair_quality=r["pair_quality"], exact_coincidence=r["exact_coincidence"], coincident_interval=r["coincident_interval"],
               duplicate_provider_position=dup_pos(r), support_cut_at_start=r["support_cut_at_start"], support_cut_at_end=r["support_cut_at_end"],
               window_centre_s=cx["window_centre_s"], window_start_s=cx["window_start_s"], window_end_s=cx["window_end_s"],
               context_complete=cx["complete"], context_critical_flags="|".join(cx["critical_flags"]),
               cut_before_a=cx["player_a"].get("cut_before"), cut_after_a=cx["player_a"].get("cut_after"),
               cut_before_b=cx["player_b"].get("cut_before"), cut_after_b=cx["player_b"].get("cut_after"),
               parent_event_id_0_5m=r.get("parent_event_id"), children_0_1m=len(r.get("children", {}).get("0.1m", [])))
    row.update({f"in_{k}": f(r) for k, f in SUBSETS.items()})
    row.update(events_file=f"{rel}/root.events-{PRIMARY}.jsonl", events_line=line, npz_file=f"{rel}/root.trajectories.npz", npz_event_index=line)
    return row


def _csv(rows: list[dict]) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode()


def tally(recs: list[dict]) -> dict[str, Any]:
    cnt = lambda f: sum(1 for r in recs if f(r))  # noqa: E731
    cuts: dict[str, int] = {}
    crit: dict[str, int] = {}
    for r in recs:
        for p in ("player_a", "player_b"):
            for k in ("cut_before", "cut_after"):
                c = r["context"][p].get(k)
                if c:
                    cuts[c] = cuts.get(c, 0) + 1
        for f in r["context"]["critical_flags"]:
            crit[f] = crit.get(f, 0) + 1
    return dict(
        subsets={k: cnt(f) for k, f in SUBSETS.items()},
        pair_quality={q: cnt(lambda r, q=q: r["pair_quality"] == q) for q in ("observed/observed", "observed/estimated", "estimated/estimated")},
        team_relation={q: cnt(lambda r, q=q: r["team_relation"] == q) for q in ("same_team", "opposite_teams")},
        contact_class={q: cnt(lambda r, q=q: r["contact_class"] == q) for q in ("point", "isolated", "sustained")},
        coincident_interval=cnt(lambda r: r["coincident_interval"]), exact_coincidence=cnt(lambda r: r["exact_coincidence"]),
        duplicate_provider_position=cnt(dup_pos), support_cut=cnt(lambda r: r["support_cut_at_start"] or r["support_cut_at_end"]),
        context_complete=cnt(lambda r: r["context"]["complete"]), context_incomplete=cnt(lambda r: not r["context"]["complete"]),
        context_with_critical_flags=cnt(lambda r: bool(r["context"]["critical_flags"])), context_cut_reasons_player_sides=cuts,
        context_critical_flags=crit,
        by_quality_and_team={f"{q}|{t}": cnt(lambda r, q=q, t=t: r["pair_quality"] == q and r["team_relation"] == t)
                             for q in ("observed/observed", "observed/estimated", "estimated/estimated") for t in ("same_team", "opposite_teams")},
        high_quality_by_team={t: cnt(lambda r, t=t: SUBSETS["high_quality_candidates"](r) and r["team_relation"] == t) for t in ("same_team", "opposite_teams")},
        d_min_m=P._q([r["d_min_m"] for r in recs]), duration_s=P._q([r["duration_s"] for r in recs]))


SCHEMA_DOC = dict(
    unit="one row / record = one synchronized proximity episode of one player pair at R = 0.25 m (d(t) <= R with both players at the same t)",
    record_file="<match>/root.events-0.25m.jsonl (line = npz_event_index); sensitivity episodes in root.events-0.1m.jsonl / -0.5m.jsonl",
    episode_fields=dict(
        event_id="sha256(event_key)[:16]; event_key = match|period|slot_a|slot_b|R_mm|start_us", pair_id="match|period|slot_a|slot_b (slot_a < slot_b)",
        player_a_b="slot, side (home/away), team_id, jersey, player_id, status_at_t_star (observed/estimated), estimated_share, quality_flags (Step 1 bits on the "
                   "episode's samples), spans_missing_samples, held_position, duplicate_provider_position, segment_at_t_star (raw rows, frames, t, x, y, "
                   "visibility), position_at_t_star [x, y] m",
        t_start_s="first instant with d <= R (s, video time)", t_end_s="last instant with d <= R", t_star_s="earliest instant of minimum distance",
        d_min_m="minimum synchronized distance", midpoint="(A(t*) + B(t*)) / 2, indexing / plots only",
        contact_class="point (zero duration) / isolated (within R at <= 1 sample instant) / sustained (>= 2)",
        pair_quality="observed/observed, observed/estimated, estimated/estimated at t*", exact_coincidence="d = 0 somewhere (exact arithmetic)",
        coincident_interval="d = 0 over an interval: cloned provider estimates", support_cut_at_start_end="a valid trajectory ends at the episode boundary",
        windows_game_s="integer game seconds whose +-2.5 s window overlaps the episode", canonical_window_game_s="centre nearest t*",
        context="window_centre_s / window_start_s / window_end_s (clipped to the period), complete, critical_flags, and per player complete_before / "
                "complete_after, cut_before / cut_after (no_adjacent_sample, gap_period_or_time, implausible_jump, unattributed_or_ambiguous), "
                "first_sample_s, last_sample_s, n_samples, context_flags, critical_flags",
        parent_event_id="episode containing this one at the next larger threshold", children="contained episodes at smaller thresholds"),
    trajectory_npz=dict(
        sample_="original provider samples (trajectory vertices) of both players in the canonical window on the valid run containing the episode: "
                "event_index, side (0 = A, 1 = B), t (s), x, y (m, RAW pitch frame), source_row, frame, visibility (1 VISIBLE, 2 ESTIMATED), "
                "quality_bits (Step 1 mask), before_t_star",
        sync_="synchronized grid (union of both players' sample instants plus start, t*, end, where both have support): event_index, t, xa, ya, xb, yb, "
              "d, a_is_sample, b_is_sample (False = linearly interpolated at that t)"),
    no_labels="no identity-switch labels, no swaps, no synthetic occlusions; provider slots are the identity reference")


def build_index(raw_root: str, matches: list[str]) -> dict[str, bytes]:
    base = approved_output_root(raw_root) / P.SCHEMA
    rows, per_match, val, allrecs, by_match = [], [], [], [], {}
    for mid in matches:
        d = base / mid
        v = validate_match(d)
        val.append(v)
        recs = load(d, PRIMARY)
        allrecs += recs
        by_match[mid] = recs
        rows += [index_row(r, k, mid) for k, r in enumerate(recs)]
        s = json.loads((d / "root.summary.json").read_text(encoding="utf-8"))
        rt = json.loads((d / "root.runtime.json").read_text(encoding="utf-8")) if (d / "root.runtime.json").exists() else {}
        t = tally(recs)
        per_match.append(dict(match_id=int(mid), windows=s["windows"]["windows"], pair_window_comparisons=s["windows"]["pair_window_comparisons"],
                              pair_interval_comparisons=s["comparisons"]["pair_interval_comparisons"],
                              episodes_0m=v["events"]["0m"], episodes_0_1m=v["events"]["0.1m"], episodes_0_25m=v["events"]["0.25m"],
                              episodes_0_5m=v["events"]["0.5m"], **{k: t["pair_quality"][q] for k, q in (("observed_observed", "observed/observed"),
                              ("observed_estimated", "observed/estimated"), ("estimated_estimated", "estimated/estimated"))},
                              coincident_interval=t["coincident_interval"], same_team=t["team_relation"]["same_team"],
                              opposite_teams=t["team_relation"]["opposite_teams"], non_coincident=t["subsets"]["non_coincident"],
                              high_quality_candidates=t["subsets"]["high_quality_candidates"], context_complete=t["context_complete"],
                              context_incomplete=t["context_incomplete"], validation_passed=v["passed"], seconds=rt.get("seconds"),
                              peak_memory_mb=rt.get("peak_memory_mb")))
    total = tally(allrecs)
    sens = {k: sum(p[k] for p in per_match) for k in ("episodes_0m", "episodes_0_1m", "episodes_0_25m", "episodes_0_5m")}
    for k, f in SUBSETS.items():                                         # subsets reconcile with the full population
        assert sum(r[f"in_{k}"] for r in rows) == total["subsets"][k]
    q = total["pair_quality"]
    assert sum(q.values()) == len(rows) == total["subsets"]["all_encounters"]
    assert total["subsets"]["non_coincident"] + total["coincident_interval"] == len(rows)
    blobs = {"encounters_index.csv": _csv(rows), "match_summary.csv": _csv(per_match), "schema.json": json.dumps(SCHEMA_DOC, indent=1).encode()}
    for k in SUBSETS:
        blobs[f"subset_{k}.csv"] = _csv([dict(match_id=r["match_id"], event_id=r["event_id"]) for r in rows if r[f"in_{k}"]])
    summary = dict(schema=P.SCHEMA, primary_threshold_m=0.25, matches=len(matches), episodes=len(rows), sensitivity_episodes=sens,
                   totals=total, per_match={str(p["match_id"]): p for p in per_match}, validation_all_passed=all(v["passed"] for v in val))
    summary["subset_definitions"] = dict(
        all_encounters="every 0.25 m episode", observed_observed="both players observed at t*", mixed_quality="one observed, one estimated at t*",
        estimated_estimated="both estimated at t*", non_coincident="not a coincident interval (cloned provider estimates excluded)",
        high_quality_candidates="observed/observed, non-coincident, no duplicated provider position, no support cut at the episode, complete 5 s "
                                "context for both players, no critical Step 1 flag on any context sample")
    blobs["summary.json"] = json.dumps(summary, indent=1, sort_keys=True).encode()
    blobs["validation.json"] = json.dumps(val, indent=1).encode()
    split = approved_output_root(raw_root) / ml.SPLITS_DIR / "splits.json"
    blobs["manifest.json"] = json.dumps(dict(
        schema=P.SCHEMA, index=OUT, matches=matches, split_manifest_sha256=hashlib.sha256(split.read_bytes()).hexdigest(),
        match_manifests_sha256={m: hashlib.sha256((base / m / "root.manifest.json").read_bytes()).hexdigest() for m in matches},
        code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(P.__file__), Path(__file__))},
        outputs_sha256={n: hashlib.sha256(b).hexdigest() for n, b in blobs.items()},
        config=P.ProximityConfig().as_dict(),
        reproduce=["python -m football_fatigue.encounters build --root data/raw --match <id>   (Step 1 cache, development ids only)",
                   "python -m football_fatigue.intersections.proximity --matches <ids> --jobs 4",
                   "python -m football_fatigue.intersections.proximity --matches <ids> --cross-match-only",
                   "python -m football_fatigue.intersections.proximity_dataset --matches <ids> --determinism <ids>"]), indent=1).encode()
    return blobs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m football_fatigue.intersections.proximity_dataset")
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--matches", nargs="+", required=True)
    ap.add_argument("--determinism", nargs="*", default=[], help="matches to re-render and compare byte-for-byte")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    root = str(Path(a.root).resolve())
    try:
        split = json.loads((approved_output_root(root) / ml.SPLITS_DIR / "splits.json").read_text(encoding="utf-8"))
        validate.check_split(split, [str(m) for m in a.matches + a.determinism], "development")
        det = [determinism(root, m) for m in a.determinism]
        for d in det:
            print(json.dumps(d), flush=True)
        blobs = build_index(root, [str(m) for m in a.matches])
        if det:
            blobs["determinism.json"] = json.dumps(det, indent=1).encode()
        out = approved_output_root(root) / P.SCHEMA / OUT
        if out.exists():
            raise IngestionError(f"{out} already exists and is never overwritten")
        s = json.loads(blobs["summary.json"])
        print(json.dumps(dict(episodes=s["episodes"], validation_all_passed=s["validation_all_passed"], subsets=s["totals"]["subsets"],
                              context_complete=s["totals"]["context_complete"], context_incomplete=s["totals"]["context_incomplete"]), indent=1))
        if not a.dry_run:
            write_retry(out, blobs)
            print(out)
        return 0 if s["validation_all_passed"] and all(d["identical"] for d in det) else 1
    except IngestionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
