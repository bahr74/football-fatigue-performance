"""Independent verification of M11 Step 1 (RAW cache, quality mask, steps) and of the trajectory segmentation built on it.

Read-only; development matches only; writes ``m11-step1-verification-v1``. Nothing here changes a threshold or a source record.
Every check is re-derived with code that does not call :mod:`football_fatigue.encounters.quality` (only the production
segmentation, :func:`football_fatigue.intersections.exact.trajectories`, is run, so that its output can be checked):

* **parse**: the RAW ``.jsonl.bz2`` is re-read with plain ``json`` and compared, frame by frame and player by player, with the
  cache (frame number, video time, period, side / jersey slot, state, raw x / y, visibility);
* **time**: video time strictly increasing, periods contiguous, duplicate timestamps, frame gaps;
* **identity**: one team per side, unique (side, jersey) slots and roster players;
* **visibility / cadence**: the ESTIMATED bit equals the provider visibility; repeats, update instants and held runs recomputed;
* **steps**: unobserved gap (time between the last available sample and the next update), period, and the jump rule
  (> ``max_speed_m_s`` over the shortest change interval and > ``jump_min_m``) recomputed and compared exactly;
* **segmentation**: every joined segment is a usable step, joined runs are continuous, no segment spans more than ``max_gap_s``
  without an available sample, crosses a period or an implausible jump; held (stationary) segments are constant;
* **diagnostics** for the thresholds (gap and speed distributions) and provider clones (bit-identical positions).

Run: ``python -m football_fatigue.step1_verify --root data/raw [--matches ...] [--jobs 3] [--no-parse]``.
"""

from __future__ import annotations

import argparse
import bz2
import csv
import hashlib
import io
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from football_fatigue.encounters import quality as Q
from football_fatigue.encounters.catalogue import OUTPUT_DIR as STEP1_DIR
from football_fatigue.encounters.trajectories import STATE_AMBIGUOUS, STATE_AVAILABLE, STATE_MISSING, VISIBILITY, load_cache
from football_fatigue.ingestion import approved_output_root

SCHEMA = "m11-step1-verification-v1"
CFG = Q.M11Config()
GAP_BINS_S = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0, math.inf)
SPEED_BINS = (0.0, 8.0, 10.0, 11.0, 12.0, 14.0, 20.0, math.inf)


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def development(raw_root: str) -> list[str]:
    from football_fatigue.matching import ml
    split = json.loads((approved_output_root(raw_root) / ml.SPLITS_DIR / "splits.json").read_text(encoding="utf-8"))
    return [str(m) for m in split["splits"]["development"]]


# ---------------------------------------------------------------- parse

def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def parse_check(raw_root: str, c) -> dict[str, Any]:
    """Re-read the RAW file and compare every frame / player with the cache (first record of a repeated frameNum, as ingestion)."""
    slot = {(str(s), int(j)): k for k, (s, j) in enumerate(zip(c.slot_side, c.slot_jersey))}
    path = Path(raw_root) / c.tracking_path
    bad = dict(frames=0, frame_num=0, video_time=0, period=0, state=0, xy=0, visibility=0, unknown_slot=0, slot_absent_in_raw=0)
    seen = np.zeros((c.n_frames, c.n_slots), bool)
    i, prev, rep_records, rep_conflicts = -1, None, 0, 0
    with bz2.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            fn = int(r["frameNum"])
            if prev is not None and fn == prev[0]:
                rep_records += 1
                rep_conflicts += any(r.get(k) != prev[1].get(k) for k in ("homePlayers", "awayPlayers"))
                if r.get("homePlayers") != prev[1].get("homePlayers"):
                    prev[2].add("home")
                if r.get("awayPlayers") != prev[1].get("awayPlayers"):
                    prev[2].add("away")
                continue
            if prev is not None:
                _frame(c, i, prev, slot, bad, seen)
            i += 1
            prev = (fn, r, set())
    if prev is not None:
        _frame(c, i, prev, slot, bad, seen)
    n = i + 1
    bad["frames"] = int(n != c.n_frames)
    absent_ok = int(((c.state != 0) & ~seen[:min(n, c.n_frames)]).sum()) if n == c.n_frames else -1
    bad["slot_absent_in_raw"] = absent_ok
    return dict(frames_raw=n, frames_cache=int(c.n_frames), repeated_records=rep_records, repeated_records_conflicting=rep_conflicts,
                mismatches=bad, ok=not any(bad.values()))


def _frame(c, i: int, prev, slot, bad, seen) -> None:
    fn, r, conflict = prev
    if i >= c.n_frames:
        return
    bad["frame_num"] += int(c.frame_num[i] != fn)
    bad["video_time"] += int(c.video_time_s[i] != float(r["videoTimeMs"]) / 1000)
    p = r.get("period")
    bad["period"] += int(int(c.source_period[i]) != (-1 if p is None else int(p)))
    for side in ("home", "away"):
        pl = r.get(f"{side}Players") or []
        jer: dict[int, list] = {}
        for q in pl:
            try:
                jer.setdefault(int(q.get("jerseyNum")), []).append(q)
            except (TypeError, ValueError):
                continue
        for j, qs in jer.items():
            s = slot.get((side, j))
            if s is None:
                bad["unknown_slot"] += 1
                continue
            seen[i, s] = True
            x, y = _num(qs[0].get("x")), _num(qs[0].get("y"))
            if side in conflict or len(qs) > 1:
                want = STATE_AMBIGUOUS
            elif x is None or y is None:
                want = STATE_MISSING
            else:
                want = STATE_AVAILABLE
            bad["state"] += int(c.state[i, s] != want)
            if want == STATE_AVAILABLE:
                bad["xy"] += int(c.x[i, s] != x or c.y[i, s] != y)
            if want != STATE_AMBIGUOUS:
                bad["visibility"] += int(c.visibility[i, s] != VISIBILITY.get(qs[0].get("visibility"), 0))


# ---------------------------------------------------------------- time, identity, cadence, steps

def time_checks(c) -> dict[str, Any]:
    t, p = c.video_time_s, c.canonical_period.astype(int)
    same = p[1:] == p[:-1]
    changes = p[np.r_[True, p[1:] != p[:-1]]]
    return dict(periods=sorted(int(x) for x in np.unique(p)), periods_contiguous=len(changes) == len(set(changes.tolist())),
                time_strictly_increasing_within_period=bool((np.diff(t)[same] > 0).all()),
                duplicate_timestamps=int((np.diff(t)[same] == 0).sum()), frame_num_increasing=bool((np.diff(c.frame_num) > 0).all()),
                frame_gaps=int((np.diff(c.frame_num)[same] > 1).sum()), rows_outside_periods=int((p == 0).sum()),
                max_time_step_within_period_s=round(float(np.diff(t)[same].max()), 6))


def identity_checks(c) -> dict[str, Any]:
    keys = list(zip(c.slot_side.tolist(), c.slot_jersey.tolist()))
    team = {s: set(c.slot_team_id[c.slot_side == s].tolist()) - {-1} for s in ("home", "away")}
    pid = [p for p in c.slot_player_id.tolist() if p >= 0]
    return dict(slots=int(c.n_slots), unique_side_jersey=len(set(keys)) == len(keys), one_team_per_side=all(len(v) == 1 for v in team.values()),
                sides_differ=team["home"] != team["away"], unique_player_ids=len(set(pid)) == len(pid),
                slots_without_roster=int((c.slot_player_id < 0).sum()), ambiguous_samples=int((c.state == STATE_AMBIGUOUS).sum()),
                duplicate_slot_samples=int(c.duplicate_slot_samples))


def cadence_mine(c) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Repeats (identical to the immediately preceding frame of the same period), update instants and held runs (>= 3 frames)."""
    av = c.state == STATE_AVAILABLE
    contig = np.r_[False, (c.frame_num[1:] - c.frame_num[:-1] == 1) & (c.canonical_period[1:] == c.canonical_period[:-1])
                   & (c.canonical_period[1:] > 0)]
    rep = np.zeros_like(av)
    rep[1:] = contig[1:, None] & av[1:] & av[:-1] & (c.x[1:] == c.x[:-1]) & (c.y[1:] == c.y[:-1])
    upd = av & ~rep
    held = np.zeros_like(av)
    for s in range(c.n_slots):
        r = np.flatnonzero(av[:, s])
        if len(r):
            starts = np.flatnonzero(upd[r, s])
            ends = np.r_[starts[1:], len(r)]
            for a, b in zip(starts, ends):
                if b - a >= 3:
                    held[r[a:b], s] = True
    return rep, upd, held


def steps_mine(c, upd: np.ndarray) -> dict[str, np.ndarray]:
    t, nominal = c.video_time_s, 1.0 / c.fps
    out: dict[str, list] = {k: [] for k in ("slot", "row_from", "row_from_last", "row_to", "gap", "dist", "change", "valid", "jump")}
    av = c.state == STATE_AVAILABLE
    for s in range(c.n_slots):
        r = np.flatnonzero(av[:, s])
        u_pos = np.flatnonzero(upd[r, s])
        if len(u_pos) < 2:
            continue
        a, b = r[u_pos[:-1]], r[u_pos[1:]]
        la = r[u_pos[1:] - 1]                                         # last available sample before the next update
        gap = t[b] - t[la]
        d = np.sqrt((c.x[b, s] - c.x[a, s]) ** 2 + (c.y[b, s] - c.y[a, s]) ** 2)
        ch = np.minimum(t[b] - t[a], gap + nominal)
        valid = (c.canonical_period[a] > 0) & (c.canonical_period[a] == c.canonical_period[b]) & (gap <= CFG.max_gap_s) & (t[b] > t[a])
        with np.errstate(divide="ignore", invalid="ignore"):
            jump = valid & (d > CFG.jump_min_m) & (d / ch > CFG.max_speed_m_s)
        for k, v in (("slot", np.full(len(a), s)), ("row_from", a), ("row_from_last", la), ("row_to", b), ("gap", gap), ("dist", d),
                     ("change", ch), ("valid", valid), ("jump", jump)):
            out[k].append(v)
    return {k: np.concatenate(v) for k, v in out.items()}


def _hist(v: np.ndarray, bins) -> dict[str, int]:
    h, _ = np.histogram(v, bins=np.array(bins, float))
    return {f"{lo:g}-{hi:g}": int(n) for lo, hi, n in zip(bins[:-1], bins[1:], h)}


def segmentation_checks(c, mask: np.ndarray, steps: dict[str, np.ndarray]) -> dict[str, Any]:
    from football_fatigue.intersections import exact as X
    from football_fatigue.intersections.detect import UNUSABLE, duplicate_positions

    seg, counts = X.trajectories(c, mask, steps)
    t, av = c.video_time_s, c.state == STATE_AVAILABLE
    usable = {(int(s), int(b)): int(a) for s, a, b, ok in zip(steps["slot"], steps["row_from"], steps["row_to"],
                                                                   steps["valid"] & ~steps["jump"]) if ok}
    last_of = {(int(s), int(a)): int(la) for s, a, la in zip(steps["slot"], steps["row_from"], steps["row_from_last"])}
    bad = dict(not_a_usable_step=0, unusable_endpoint=0, spans_gap=0, crosses_period=0, jump_in_segment=0, stationary_not_constant=0,
               discontinuous_join=0, time_order=0, endpoint_not_available=0)
    n = len(seg["slot"])
    for g in range(n):
        s, r0, r1 = int(seg["slot"][g]), int(seg["row0"][g]), int(seg["row1"][g])
        if not (av[r0, s] and av[r1, s]):
            bad["endpoint_not_available"] += 1
        if t[r1] <= t[r0]:
            bad["time_order"] += 1
        if c.canonical_period[r0] != c.canonical_period[r1] or c.canonical_period[r0] <= 0:
            bad["crosses_period"] += 1
        rr = np.flatnonzero(av[r0:r1 + 1, s]) + r0
        if len(rr) > 1 and np.diff(t[rr]).max() > CFG.max_gap_s + 1e-9:
            bad["spans_gap"] += 1
        if seg["stationary"][g]:
            if not ((c.x[rr, s] == c.x[r0, s]).all() and (c.y[rr, s] == c.y[r0, s]).all() and len(rr) == r1 - r0 + 1):
                bad["stationary_not_constant"] += 1
            continue
        a = usable.get((s, r1))
        la = last_of.get((s, a)) if a is not None else None
        if a is None or r0 not in (a, la - 1 if la is not None else a):
            bad["not_a_usable_step"] += 1
            continue
        if (mask[a, s] | mask[r1, s]) & UNUSABLE:
            bad["unusable_endpoint"] += 1
        d = math.hypot(c.x[r1, s] - c.x[r0, s], c.y[r1, s] - c.y[r0, s])
        if d > CFG.jump_min_m and d / min(t[r1] - t[a], t[r1] - t[la] + 1 / c.fps) > CFG.max_speed_m_s:
            bad["jump_in_segment"] += 1
    joined = (seg["slot"][1:] == seg["slot"][:-1]) & (seg["row0"][1:] == seg["row1"][:-1])
    bad["discontinuous_join"] = int((joined & ((seg["X0"][1:] != seg["X1"][:-1]) | (seg["Y0"][1:] != seg["Y1"][:-1]))).sum())
    # runs and what bounds them
    run_end = np.r_[~joined, True]
    last_rows = seg["row1"][run_end]
    tail_dropped = 0
    for g in np.flatnonzero(run_end):
        s, r1 = int(seg["slot"][g]), int(seg["row1"][g])
        if seg["stationary"][g] and r1 + 1 < c.n_frames and av[r1 + 1, s] and c.x[r1 + 1, s] == c.x[r1, s] and c.y[r1 + 1, s] == c.y[r1, s]:
            tail_dropped += 1
    dup = duplicate_positions(c)
    clone_seg = int((dup[seg["row0"], seg["slot"]] & dup[seg["row1"], seg["slot"]] & ~seg["stationary"]).sum())
    est_end = (c.visibility[seg["row0"], seg["slot"]] == VISIBILITY["ESTIMATED"]) | (c.visibility[seg["row1"], seg["slot"]] == VISIBILITY["ESTIMATED"])
    return dict(segments=int(n), runs=int(run_end.sum()), segment_counts=counts, defects=bad, ok=not any(bad.values()),
                runs_ending_one_frame_before_a_held_sample=tail_dropped, run_last_rows=int(len(last_rows)),
                segments_with_an_estimated_endpoint=int(est_end.sum()), duplicate_position_samples=int(dup.sum()),
                moving_segments_between_duplicated_positions=clone_seg)


def verify_match(raw_root: str, mid: str, parse: bool) -> dict[str, Any]:
    if mid not in development(raw_root):
        raise RuntimeError(f"{mid} is not a development match")
    d = approved_output_root(raw_root) / STEP1_DIR / mid
    man = json.loads((d / "root.manifest.json").read_text(encoding="utf-8"))
    hashes = {n: _sha(d / n) == man["outputs_sha256"][n] for n in ("root.cache.npz", "root.quality.npz")}
    c = load_cache(d / "root.cache.npz")
    with np.load(d / "root.quality.npz") as z:
        mask = z["mask"]
        steps = {k[5:]: z[k] for k in z.files if k.startswith("step_")}
    out: dict[str, Any] = dict(match_id=mid, step1_hashes=hashes, time=time_checks(c), identity=identity_checks(c))
    pres = c.state != 0
    out["visibility"] = dict(
        estimated_bit_equals_visibility=bool(np.array_equal((mask & Q.ESTIMATED) != 0, pres & (c.visibility == VISIBILITY["ESTIMATED"]))),
        available_without_visibility=int((c.state == STATE_AVAILABLE).sum() - ((c.state == STATE_AVAILABLE) & (c.visibility > 0)).sum()),
        available_samples=int((c.state == STATE_AVAILABLE).sum()),
        estimated_share_of_available=round(float((c.visibility[c.state == STATE_AVAILABLE] == VISIBILITY["ESTIMATED"]).mean()), 4))
    rep, upd, held = cadence_mine(c)
    out["cadence"] = dict(repeat_equal=bool(np.array_equal(rep, (mask & Q.REPEAT_OF_PREVIOUS) != 0)),
                          update_equal=bool(np.array_equal(upd, (mask & Q.UPDATE_INSTANT) != 0)),
                          held_equal=bool(np.array_equal(held, (mask & Q.HELD_POSITION) != 0)))
    mine = steps_mine(c, upd)
    same = {k: bool(np.array_equal(mine[k], steps[k])) for k in ("slot", "row_from", "row_from_last", "row_to", "valid", "jump")}
    same["distance"] = bool(np.allclose(mine["dist"], steps["distance_m"], rtol=0, atol=1e-9))
    same["change_dt"] = bool(np.allclose(mine["change"], steps["change_dt_s"], rtol=0, atol=1e-9))
    v, j = mine["valid"], mine["jump"]
    sp = np.divide(mine["dist"], mine["change"], out=np.full(len(v), np.inf), where=mine["change"] > 0)
    vis = (c.visibility[mine["row_from"], mine["slot"]] == 1) & (c.visibility[mine["row_to"], mine["slot"]] == 1)
    ok = v & ~j & (mine["dist"] > 0)
    out["steps"] = dict(equal_to_stored=same, ok=all(same.values()), steps=int(len(v)), invalid=int((~v).sum()),
                        invalid_by_gap=int(((mine["gap"] > CFG.max_gap_s) & (c.canonical_period[mine["row_from"]] == c.canonical_period[mine["row_to"]])).sum()),
                        invalid_by_period=int((c.canonical_period[mine["row_from"]] != c.canonical_period[mine["row_to"]]).sum()),
                        jumps=int(j.sum()), jumps_visible_to_visible=int((j & vis).sum()),
                        unobserved_gap_hist_s=_hist(mine["gap"][mine["gap"] > 1.5 / c.fps], GAP_BINS_S),
                        speed_hist_valid_moving_m_s=_hist(sp[ok], SPEED_BINS), speed_hist_visible_m_s=_hist(sp[ok & vis], SPEED_BINS),
                        speed_quantiles_visible_m_s=[round(float(q), 3) for q in np.quantile(sp[ok & vis], [0.5, 0.99, 0.999, 0.9999])],
                        jump_distance_hist_m=_hist(mine["dist"][j], (1.0, 1.5, 2.0, 3.0, 5.0, 10.0, math.inf)),
                        frame_wide_shift_jumps=int(((mask[mine["row_to"][j], mine["slot"][j]] & Q.FRAME_WIDE_SHIFT) != 0).sum()))
    out["segmentation"] = segmentation_checks(c, mask, steps)
    if parse:
        out["parse"] = parse_check(raw_root, c)
    return out


def _one(a: tuple[str, str, bool]) -> dict[str, Any]:
    return verify_match(*a)


def flat(r: dict[str, Any]) -> dict[str, Any]:
    p = r.get("parse")
    return dict(match_id=r["match_id"], step1_hashes=all(r["step1_hashes"].values()), parse_ok=None if p is None else p["ok"],
                repeated_records=None if p is None else p["repeated_records"],
                repeated_records_conflicting=None if p is None else p["repeated_records_conflicting"],
                time_ok=r["time"]["time_strictly_increasing_within_period"] and r["time"]["periods_contiguous"],
                duplicate_timestamps=r["time"]["duplicate_timestamps"], frame_gaps=r["time"]["frame_gaps"],
                identity_ok=all(r["identity"][k] for k in ("unique_side_jersey", "one_team_per_side", "sides_differ", "unique_player_ids")),
                estimated_bit_ok=r["visibility"]["estimated_bit_equals_visibility"], estimated_share=r["visibility"]["estimated_share_of_available"],
                cadence_ok=all(r["cadence"].values()), steps_ok=r["steps"]["ok"], steps=r["steps"]["steps"], invalid_steps=r["steps"]["invalid"],
                jumps=r["steps"]["jumps"], segmentation_ok=r["segmentation"]["ok"], segments=r["segmentation"]["segments"],
                runs=r["segmentation"]["runs"], held_tail_frames_not_in_trajectory=r["segmentation"]["runs_ending_one_frame_before_a_held_sample"],
                duplicate_position_samples=r["segmentation"]["duplicate_position_samples"],
                cloned_moving_segments=r["segmentation"]["moving_segments_between_duplicated_positions"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--matches", nargs="*")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--no-parse", action="store_true", help="skip the RAW re-parse (about 40 s per match)")
    a = ap.parse_args(argv)
    ms = a.matches or development(a.root)
    with ProcessPoolExecutor(max(1, a.jobs)) as ex:
        res = list(ex.map(_one, [(a.root, m, not a.no_parse) for m in ms]))
    rows = [flat(r) for r in res]
    tot = lambda f: sum(int(f(r)) for r in res)  # noqa: E731
    agg = dict(matches=len(res), all_checks_pass=all(r["parse_ok"] in (True, None) and r["time_ok"] and r["identity_ok"] and r["estimated_bit_ok"]
                                                     and r["cadence_ok"] and r["steps_ok"] and r["segmentation_ok"] and r["step1_hashes"] for r in rows),
               steps=tot(lambda r: r["steps"]["steps"]), invalid_steps=tot(lambda r: r["steps"]["invalid"]),
               invalid_by_gap=tot(lambda r: r["steps"]["invalid_by_gap"]), invalid_by_period=tot(lambda r: r["steps"]["invalid_by_period"]),
               jumps=tot(lambda r: r["steps"]["jumps"]), frame_wide_shift_jumps=tot(lambda r: r["steps"]["frame_wide_shift_jumps"]),
               segments=tot(lambda r: r["segmentation"]["segments"]), runs=tot(lambda r: r["segmentation"]["runs"]),
               duplicate_timestamps=tot(lambda r: r["time"]["duplicate_timestamps"]),
               repeated_records=tot(lambda r: (r.get("parse") or {}).get("repeated_records", 0)),
               repeated_records_conflicting=tot(lambda r: (r.get("parse") or {}).get("repeated_records_conflicting", 0)),
               held_tail_frames_not_in_trajectory=tot(lambda r: r["segmentation"]["runs_ending_one_frame_before_a_held_sample"]),
               duplicate_position_samples=tot(lambda r: r["segmentation"]["duplicate_position_samples"]),
               cloned_moving_segments=tot(lambda r: r["segmentation"]["moving_segments_between_duplicated_positions"]))
    for key in ("unobserved_gap_hist_s", "speed_hist_valid_moving_m_s", "speed_hist_visible_m_s", "jump_distance_hist_m"):
        agg[key] = {b: sum(r["steps"][key][b] for r in res) for b in res[0]["steps"][key]}
    out = approved_output_root(a.root) / SCHEMA
    out.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    files = {"per_match.csv": buf.getvalue().encode(), "details.json": json.dumps(res, indent=1, default=int).encode(),
             "summary.json": json.dumps(agg, indent=1).encode()}
    for n, b in files.items():
        (out / n).write_bytes(b)
    (out / "manifest.json").write_text(json.dumps(dict(schema=SCHEMA, matches=ms, config=CFG.as_dict(), code_sha256=_sha(Path(__file__)),
                                                       outputs_sha256={n: hashlib.sha256(b).hexdigest() for n, b in files.items()}), indent=1),
                                       encoding="utf-8")
    print(json.dumps(dict(out=str(out), **{k: v for k, v in agg.items() if not isinstance(v, dict)}), indent=1))
    return 0 if agg["all_checks_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
