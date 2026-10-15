"""Shared, read-only helpers for the Milestone 1B audit verification scripts.

Audit tooling only: this is not a production loader, validator, or synchronization engine.
It never writes inside the raw data root, never decompresses to disk, and streams tracking
files one record at a time. Standard library plus the project's own inventory module.
"""
from __future__ import annotations

import bz2
import json
import math
import multiprocessing
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterator

from football_fatigue import inventory as inv

# Event Data versions: label used in outputs -> inventory version (None is the unlabelled root folder).
VERSIONS: dict[str, str | None] = {
    "root": None,
    "march_14_2025": "March 14, 2025",
    "may_1_2025": "May 1, 2025",
}

# Provisional temporal alignment policy (documented in docs/dataset_schema_audit.md, section 6).
# One tolerance only: one frame interval. A candidate frame must have the same period as the event.
TIE_EPSILON_S = 1e-9
# Provisional coordinate-agreement rule (temporal alignment is a separate, weaker status).
COORD_EXACT_M = 0.0005  # source precision is 0.001 m
COORD_NEAR_M = 0.01
COORD_AGREE_FRACTION = 0.9
COORD_MIN_COMPARED = 11


class AuditError(Exception):
    """A required input is missing or ambiguous; the check is reported as failed, never guessed."""


def locate(root: str | Path, match_id: str, _cache: dict = {}) -> dict[str, Any]:
    """Locate one match's files through the project inventory (explicit about missing/duplicate)."""
    validate_match_id(match_id)
    root = str(Path(root))
    if root not in _cache:
        _cache[root] = inv.scan(root)
    mi = _cache[root].matches.get(str(match_id))
    if mi is None:
        raise AuditError(f"match {match_id}: not found under {root}")

    def one(paths: list[str], what: str) -> Path:
        if len(paths) != 1:
            raise AuditError(f"match {match_id}: {len(paths)} {what} candidates (expected exactly 1): {paths}")
        return Path(root) / paths[0]

    events = {}
    for label, version in VERSIONS.items():
        events[label] = one(mi.events.get(version, []), f"event[{label}]")
    return dict(
        match_id=str(match_id),
        tracking=one(mi.tracking, "tracking"),
        metadata=one(mi.metadata, "metadata"),
        rosters=one(mi.rosters, "roster"),
        events=events,
    )


def match_ids(root: str | Path) -> list[str]:
    return inv.scan(str(Path(root))).match_ids


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_metadata(path: Path) -> dict:
    d = load_json(path)
    if isinstance(d, list):
        if len(d) != 1:
            raise AuditError(f"{path}: metadata list has {len(d)} objects")
        d = d[0]
    return d


def iter_tracking(path: Path) -> Iterator[dict]:
    """Stream tracking records from a .jsonl.bz2 file; nothing is written to disk."""
    with bz2.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# ---------------------------------------------------------------- statistics helpers

def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    return v[min(len(v) - 1, int(q * (len(v) - 1)))]


def dist_stats(values: list[float]) -> dict:
    return dict(n=len(values), p50=pct(values, 0.5), p90=pct(values, 0.9), p99=pct(values, 0.99),
                max=(max(values) if values else None))


def dist_xy(a: tuple, b: tuple) -> float | None:
    if a is None or b is None or None in (a[0], a[1], b[0], b[1]):
        return None
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ---------------------------------------------------------------- smoothed/raw duplicate jerseys

def classify_duplicate_jerseys(entries: list[dict] | None) -> dict:
    """Classify one player list. Never selects a candidate.

    status: 'missing' (None), 'unique' (no repeated jerseyNum), 'duplicate_identical'
    (every repeated group has identical x, y and attributes), 'duplicate_conflicting'
    (some repeated group has differing x or y). groups: jerseyNum -> list of candidate entries
    in source order (list order carries no meaning).
    """
    if entries is None:
        return dict(status="missing", groups={}, max_spread_m=None)
    by: dict[Any, list[dict]] = {}
    for e in entries:
        by.setdefault(e.get("jerseyNum"), []).append(e)
    groups = {j: g for j, g in by.items() if len(g) > 1}
    if not groups:
        return dict(status="unique", groups={}, max_spread_m=None)
    conflicting = False
    spread = 0.0
    for g in groups.values():
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                ci, cj = (g[i].get("x"), g[i].get("y")), (g[j].get("x"), g[j].get("y"))
                if ci != cj:
                    conflicting = True
                d = dist_xy(ci, cj)
                if d is not None:
                    spread = max(spread, d)
    return dict(status="duplicate_conflicting" if conflicting else "duplicate_identical", groups=groups, max_spread_m=spread)


# ---------------------------------------------------------------- event rows: collision-safe matching

def pair_key(row: dict) -> tuple:
    return (row.get("gameEventId"), row.get("possessionEventId"))


def signature(row: dict) -> tuple:
    """Row signature used for matching: ids plus times and types. Not unique on its own."""
    ge = row.get("gameEvents") or {}
    pe = row.get("possessionEvents") or {}
    return (row.get("gameEventId"), row.get("possessionEventId"), row.get("startTime"),
            ge.get("gameEventType"), pe.get("possessionEventType"))


def keyed_rows(rows: list[dict]) -> list[tuple]:
    """(signature, ordinal) per row, where ordinal counts repeats of the same signature in file order."""
    seen: Counter = Counter()
    out = []
    for r in rows:
        s = signature(r)
        out.append((s, seen[s]))
        seen[s] += 1
    return out


def match_rows(rows_a: list[dict], rows_b: list[dict]) -> dict:
    """Collision-safe one-to-one matching of rows between two versions of the same match.

    Rows are matched on (signature, ordinal-within-signature). Every row is preserved:
    unmatched rows of either side are returned, never dropped. 'ambiguous_signatures' lists
    signatures whose multiplicity differs between the two sides (their surplus rows are unmatched).
    """
    ka, kb = keyed_rows(rows_a), keyed_rows(rows_b)
    index_b = {k: j for j, k in enumerate(kb)}
    pairs, used_b = [], set()
    unmatched_a = []
    for i, k in enumerate(ka):
        j = index_b.get(k)
        if j is None:
            unmatched_a.append(i)
        else:
            pairs.append((i, j))
            used_b.add(j)
    unmatched_b = [j for j in range(len(rows_b)) if j not in used_b]
    ca, cb = Counter(k[0] for k in ka), Counter(k[0] for k in kb)
    ambiguous = sorted((s for s in ca.keys() & cb.keys() if ca[s] != cb[s]), key=repr)
    return dict(pairs=pairs, unmatched_a=unmatched_a, unmatched_b=unmatched_b, ambiguous_signatures=ambiguous)


def repeated_pair_rows(rows: list[dict]) -> int:
    """Rows whose (gameEventId, possessionEventId) pair occurs more than once in the file."""
    c = Counter(pair_key(r) for r in rows)
    return sum(n for n in c.values() if n > 1)


EXCLUDED_TOP = ("homePlayers", "awayPlayers", "ball")


def flatten(row: dict) -> dict[str, Any]:
    """Leaf paths of an event row, excluding player/ball position lists (compared separately)."""
    out: dict[str, Any] = {}

    def walk(v: Any, p: str) -> None:
        if isinstance(v, dict):
            for k, x in v.items():
                walk(x, f"{p}.{k}")
        else:
            out[p] = v

    for k, v in row.items():
        if k not in EXCLUDED_TOP:
            walk(v, k)
    return out


def compare_fields(a: dict, b: dict) -> dict:
    """Common-field comparison of two matched rows.

    changed: paths present in both rows whose values differ (null vs value counts as changed).
    only_a / only_b: paths present in one row only (schema differences, not value changes).
    """
    fa, fb = flatten(a), flatten(b)
    common = fa.keys() & fb.keys()
    changed = sorted(p for p in common if fa[p] != fb[p])
    return dict(changed=changed, only_a=sorted(fa.keys() - fb.keys()), only_b=sorted(fb.keys() - fa.keys()), n_common=len(common))


def player_map(row: dict) -> dict[tuple, dict]:
    out = {}
    for side in ("homePlayers", "awayPlayers"):
        for p in row.get(side) or []:
            out[(side, p.get("jerseyNum"))] = p
    return out


def compare_positions(a: dict, b: dict) -> dict:
    pa, pb = player_map(a), player_map(b)
    both = pa.keys() & pb.keys()
    xy_changed = 0
    max_d = 0.0
    for k in both:
        d = dist_xy((pa[k].get("x"), pa[k].get("y")), (pb[k].get("x"), pb[k].get("y")))
        if d is None or d > 0:
            xy_changed += 1
            if d is not None:
                max_d = max(max_d, d)
    return dict(players_compared=len(both), player_set_differs=set(pa) != set(pb), xy_changed=xy_changed,
                max_xy_diff_m=max_d, ball_differs=(a.get("ball") or []) != (b.get("ball") or []))


# ---------------------------------------------------------------- temporal alignment (provisional policy)

class Aligner:
    """Streaming nearest-frame selection for a list of events (one aligner per event version).

    Policy (PROVISIONAL, see docs/dataset_schema_audit.md section 6):
      * units: event time in seconds of video time; frame time = videoTimeMs / 1000;
      * a candidate frame must belong to the same source period as the event;
      * the nearest candidate (smallest |event - frame| time) is chosen; on an exact tie
        (difference below TIE_EPSILON_S) the earlier frame wins, because frames are offered
        in increasing time order and only a strictly smaller residual replaces the choice;
      * the single maximum acceptable residual is one frame interval (1 / fps);
      * no extrapolation: events outside a period's frame range stay unmatched, with a reason;
      * duplicate source records of one frameNum are the caller's job to collapse (the first
        record is the one offered), and the selected frameNum is recorded as provenance.
    """

    def __init__(self, events: list[tuple[int, float | None, Any]], fps: float):
        self.tol = 1.0 / fps
        self.events = [(i, t, p) for i, t, p in events if t is not None]
        self.events.sort(key=lambda e: e[1])
        self.no_time = [i for i, t, p in events if t is None]
        self.times = [e[1] for e in self.events]
        self.ptr = 0
        self.best: dict[int, tuple] = {}      # idx -> (abs_dt, signed_dt, frame_num, payload)
        self.best_any: dict[int, tuple] = {}  # idx -> (abs_dt, period) ignoring the period rule
        self.period_range: dict[Any, list[float]] = {}

    def offer(self, frame_num: int, t_s: float, period: Any, payload: Callable[[], Any]) -> None:
        rng = self.period_range.setdefault(period, [t_s, t_s])
        rng[0], rng[1] = min(rng[0], t_s), max(rng[1], t_s)
        while self.ptr < len(self.times) and self.times[self.ptr] < t_s - self.tol - TIE_EPSILON_S:
            self.ptr += 1
        j = self.ptr
        while j < len(self.times) and self.times[j] <= t_s + self.tol + TIE_EPSILON_S:
            idx, et, ep = self.events[j]
            dt = et - t_s
            a = abs(dt)
            if a <= self.tol + TIE_EPSILON_S:
                cur_any = self.best_any.get(idx)
                if cur_any is None or a < cur_any[0] - TIE_EPSILON_S:
                    self.best_any[idx] = (a, period)
                if ep == period:
                    cur = self.best.get(idx)
                    if cur is None or a < cur[0] - TIE_EPSILON_S:
                        self.best[idx] = (a, dt, frame_num, payload())
            j += 1

    def outcome(self, idx: int, event_time: float | None, event_period: Any) -> tuple[str, str | None]:
        """('matched', None) or ('unmatched', reason)."""
        if event_time is None:
            return "unmatched", "missing_event_time"
        if idx in self.best:
            return "matched", None
        if event_period not in (1, 2, 3, 4):
            return "unmatched", "invalid_source_period"
        if idx in self.best_any:
            return "unmatched", "period_label_differs_from_nearest_frame"
        rng = self.period_range.get(event_period)
        if rng is None:
            return "unmatched", "no_tracking_for_period"
        if event_time < rng[0] - self.tol - TIE_EPSILON_S or event_time > rng[1] + self.tol + TIE_EPSILON_S:
            return "unmatched", "outside_period_frame_range"
        return "unmatched", "no_frame_within_tolerance"


def coordinate_status(event_players: dict, tracking_players: dict) -> dict:
    """Coordinate agreement of one event row with the chosen frame (raw tracking positions).

    Status is 'coordinate_agreeing' when at least COORD_AGREE_FRACTION of >= COORD_MIN_COMPARED
    compared players lie within COORD_NEAR_M; 'coordinate_disagreeing' when compared but not;
    'coordinate_not_assessable' otherwise. This corroborates the timestamp only; it does not
    validate the player, team, or physical action.
    """
    d = [dist_xy((p.get("x"), p.get("y")), tracking_players[k]) for k, p in event_players.items() if k in tracking_players]
    d = [x for x in d if x is not None]
    n = len(d)
    if n < COORD_MIN_COMPARED:
        status = "coordinate_not_assessable"
    else:
        near = sum(1 for x in d if x <= COORD_NEAR_M)
        status = "coordinate_agreeing" if near / n >= COORD_AGREE_FRACTION else "coordinate_disagreeing"
    return dict(status=status, compared=n, exact=sum(1 for x in d if x <= COORD_EXACT_M),
                near=sum(1 for x in d if x <= COORD_NEAR_M), distances=d)


# ---------------------------------------------------------------- batch runner

def _worker(args: tuple) -> tuple[str, dict]:
    func, root, mid = args
    t0 = time.time()
    try:
        res = func(root, mid)
        res = dict(res, ok=True)
    except Exception as e:  # reported, never swallowed silently
        res = dict(ok=False, error=f"{type(e).__name__}: {e}")
    res["match_id"] = mid
    res["seconds"] = round(time.time() - t0, 1)
    return mid, res


MATCH_ID_RE = re.compile(r"[0-9]{1,9}")  # canonical inventory format: digits only (see inventory match-file patterns)


def validate_match_id(mid: object) -> str:
    """Accept only a canonical match id (ASCII digits). Rejects separators, traversal, absolute paths."""
    if not isinstance(mid, str) or not MATCH_ID_RE.fullmatch(mid):
        raise AuditError(f"invalid match id {mid!r}: expected 1-9 ASCII digits")
    return mid


def _inside(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def guard_output_dir(root: str | Path, out_dir: Path) -> Path:
    """Resolve out_dir (following symlinks/junctions) and refuse any location inside the raw root."""
    out_dir = Path(out_dir).resolve()
    raw = Path(root).resolve()
    if _inside(out_dir, raw):
        raise AuditError(f"refusing to write inside the raw data root: {out_dir}")
    return out_dir


def safe_destination(root: str | Path, out_dir: Path, mid: str, suffix: str = ".json") -> Path:
    """Final write path for one match, validated immediately before use (also for temporary files).

    The id must be canonical; the destination, resolved through symlinks, must be a direct child of the
    resolved output directory, must not be a symlink itself, and must not resolve into the raw root.
    """
    validate_match_id(mid)
    out_dir = guard_output_dir(root, out_dir)
    dest = out_dir / f"{mid}{suffix}"
    if dest.is_symlink():
        raise AuditError(f"refusing to write through a symlink: {dest}")
    resolved = dest.resolve()
    if resolved.parent != out_dir or _inside(resolved, Path(root).resolve()):
        raise AuditError(f"destination escapes the audit output directory: {dest}")
    return dest


def run_matches(func: Callable[[str, str], dict], root: str, mids: list[str], out_dir: Path, workers: int, force: bool) -> dict:
    """Run func(root, match_id) per match; one JSON file per match; existing files are kept unless force.

    All ids and the output directory are validated before any work is submitted. Returns
    {'attempted': n, 'failed': {match_id: error}} for the matches attempted in this call.
    """
    mids = [validate_match_id(m) for m in mids]
    out_dir = guard_output_dir(root, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    for m in mids:
        target = safe_destination(root, out_dir, m)
        if target.exists() and not force:
            print(f"keep existing {target.name} (use --force to recompute)")
        else:
            todo.append((func, root, m))
    t0 = time.time()
    failed: dict = {}
    results = []
    if workers > 1 and len(todo) > 1:
        with multiprocessing.Pool(workers) as pool:
            for mid, res in pool.imap_unordered(_worker, todo):
                results.append((mid, res))
                _save(root, out_dir, mid, res)
    else:
        for a in todo:
            mid, res = _worker(a)
            results.append((mid, res))
            _save(root, out_dir, mid, res)
    for mid, res in results:
        if not res.get("ok"):
            failed[mid] = res.get("error")
    print(f"attempted {len(todo)} match(es) in {time.time() - t0:.0f}s ({len(failed)} failed) -> {out_dir}")
    return dict(attempted=len(todo), failed=failed)


def _save(root: str | Path, out_dir: Path, mid: str, res: dict) -> None:
    dest = safe_destination(root, out_dir, mid)
    tmp = safe_destination(root, out_dir, mid, suffix=".json.tmp")   # same protection for the temporary file
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)
    safe_destination(root, out_dir, mid)                             # re-validate right before the replace
    os.replace(tmp, dest)
    print(f"{'ok  ' if res.get('ok') else 'FAIL'} {mid} ({res.get('seconds')}s)" + ("" if res.get("ok") else f" {res.get('error')}"), flush=True)


def load_outputs(out_dir: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(Path(out_dir).glob("*.json")):
        try:
            out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            out[p.stem] = dict(ok=False, error=f"unreadable output: {e}")
    return out


def check_outputs(results: dict[str, dict], requested: list[str]) -> dict:
    """Requested matches that failed, have no output, or have incomplete output -> {match_id: reason}."""
    bad = {}
    for m in requested:
        r = results.get(m)
        if r is None:
            bad[m] = "missing output"
        elif not r.get("ok"):
            bad[m] = r.get("error") or "incomplete output"
    return bad


def add_common_args(ap) -> None:
    ap.add_argument("--root", default="data/raw", help="raw data root (read-only)")
    ap.add_argument("--matches", nargs="*", help="match ids (default: all found by the inventory)")
    ap.add_argument("--out", default=None, help="output directory (default data/processed/audit_1b/<script>)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="recompute existing per-match outputs")
    ap.add_argument("--aggregate", action="store_true", help="only summarise existing per-match outputs")


def cli_main(doc: str, default_out: str, func: Callable[[str, str], dict], aggregate: Callable[[dict], None], argv: list[str] | None = None) -> int:
    """Shared CLI body. Exit status: 0 only if every requested match has complete, successful output;
    1 if any requested match failed or its output is missing/incomplete; 2 for invalid input or
    unsafe output paths (controlled error, nothing written)."""
    import argparse
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    a = ap.parse_args(argv)
    try:
        mids = [validate_match_id(m) for m in a.matches] if a.matches else match_ids(a.root)
        out = guard_output_dir(a.root, Path(a.out or default_out))
        if not a.aggregate:
            run_matches(func, a.root, mids, out, a.workers, a.force)
        results = load_outputs(out)
        failures = check_outputs(results, mids)
        requested = {m: results[m] for m in mids if m in results}
        if any(r.get("ok") for r in requested.values()):
            aggregate(requested)
        else:
            print("no successful per-match results to aggregate")
    except AuditError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    if failures:
        print(f"FAILED: {len(failures)} of {len(mids)} requested match(es) failed or have missing/incomplete output: {failures}", file=sys.stderr)
        return 1
    print(f"all {len(mids)} requested match(es) have complete, successful output")
    return 0
