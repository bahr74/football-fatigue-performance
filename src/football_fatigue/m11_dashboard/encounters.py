"""Data access for the encounter explorer (``encounters_app.py``): synchronized proximity encounters of ``m11-sync-proximity-v2``
with the event-centred 5 s contexts of ``m11-sync-proximity-v3`` (detection unchanged; same event IDs and order).

Read-only over the dataset; the only file written is the separate manual-review CSV (:func:`save_review`).

- matches: development matches of the frozen split manifest that have extraction outputs (validation / test ids are refused);
- events: ``<match>/root.events-<R>m.jsonl`` for R = 0, 0.1, 0.25, 0.5 m; high-quality membership from the published
  ``development-index/subset_high_quality_candidates.csv``;
- contexts: ``m11-sync-proximity-v3/<match>/root.contexts.jsonl`` and ``root.trajectories.npz`` hold the event-centred context
  [t* - 2.5 s, t* + 2.5 s] of the **0.25 m** episodes only (row k = line k of the v2 0.25 m events; the v3 context replaces the
  record's v2 ``context``, which is kept as ``context_v2``); window-edge points are interpolations on a valid segment, not samples. An event of another threshold is shown with the context of a related 0.25 m episode (its ancestor for
  0 / 0.1 m, its child for 0.5 m) only when that context window contains the event's t* *and* both players' positions there
  reproduce the event's recorded A(t*), B(t*); otherwise no trajectory is shown (:func:`context_for`).

Each player's samples form one valid Step 1 run (no gap > 0.5 s, jump, attribution problem or period change inside), so linear
interpolation between consecutive samples is the trajectory model of the extraction; the synchronized grid holds both players
at the same instants. Manual reviews are audit notes, not identity labels or verified occlusions.
"""

from __future__ import annotations

import csv
import io
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from football_fatigue.ingestion import approved_output_root
from football_fatigue.matching import ml

SCHEMA = "m11-sync-proximity-v2"
CONTEXT_SCHEMA = "m11-sync-proximity-v3"
HALF_S = 2.5
THRESHOLDS = ("0m", "0.1m", "0.25m", "0.5m")
PRIMARY = "0.25m"
REVIEW_STATUSES = ("accept", "reject", "uncertain")
REVIEW_FIELDS = ("match_id", "threshold", "event_id", "status", "note", "reviewed_at_utc")
POS_TOL_M = 1e-6


class ExplorerError(RuntimeError):
    pass


def base(raw_root: str) -> Path:
    return approved_output_root(raw_root) / SCHEMA


def context_dir(raw_root: str) -> Path:
    return approved_output_root(raw_root) / CONTEXT_SCHEMA


def index_dir(raw_root: str) -> Path:
    return base(raw_root) / "development-index"


def reviews_path(raw_root: str) -> Path:
    """Separate review file outside every dataset directory (override: ``M11_REVIEWS_PATH``)."""
    p = os.environ.get("M11_REVIEWS_PATH")
    return Path(p) if p else approved_output_root(raw_root) / "m11-encounter-reviews" / "reviews.csv"


def development_matches(raw_root: str) -> list[str]:
    """Development matches (frozen split manifest) with written 0.25 m events, in manifest order; nothing else is listed."""
    split = json.loads((approved_output_root(raw_root) / ml.SPLITS_DIR / "splits.json").read_text(encoding="utf-8"))
    dev = split["splits"]["development"]
    return [m for m in dev if (base(raw_root) / m / f"root.events-{PRIMARY}.jsonl").exists()]


def check_development(raw_root: str, match_id: str) -> None:
    if str(match_id) not in development_matches(raw_root):
        raise ExplorerError(f"match {match_id} is not an extracted development match")


def summary(raw_root: str) -> dict[str, Any]:
    return json.loads((index_dir(raw_root) / "summary.json").read_text(encoding="utf-8"))


def high_quality_ids(raw_root: str) -> set[tuple[str, str]]:
    with open(index_dir(raw_root) / "subset_high_quality_candidates.csv", encoding="utf-8") as f:
        return {(r["match_id"], r["event_id"]) for r in csv.DictReader(f)}


def load_events(raw_root: str, match_id: str, threshold: str) -> list[dict]:
    check_development(raw_root, match_id)
    if threshold not in THRESHOLDS:
        raise ExplorerError(f"unknown threshold {threshold}")
    p = base(raw_root) / str(match_id) / f"root.events-{threshold}.jsonl"
    recs = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
    if threshold == PRIMARY:                                             # event-centred v3 contexts, line for line
        cp = context_dir(raw_root) / str(match_id) / "root.contexts.jsonl"
        if not cp.exists():
            raise ExplorerError(f"{cp} is missing: run python -m football_fatigue.proximity_context")
        ctx = [json.loads(x) for x in cp.read_text(encoding="utf-8").splitlines()]
        if [x["event_id"] for x in ctx] != [r["event_id"] for r in recs]:
            raise ExplorerError(f"{cp} does not list the v2 events of match {match_id} in order")
        for r, x in zip(recs, ctx):
            r["context_v2"], r["context"] = r.get("context"), x["context"]
    return recs


def load_npz(raw_root: str, match_id: str) -> dict[str, np.ndarray]:
    check_development(raw_root, match_id)
    with np.load(context_dir(raw_root) / str(match_id) / "root.trajectories.npz") as z:
        return {k: z[k] for k in z.files}


def mtime(p: Path) -> float:
    """Cache key part: changes when a compatible output is rewritten, so cached loaders reread it."""
    return p.stat().st_mtime if p.exists() else 0.0


def rows(recs: list[dict], hq: set[tuple[str, str]], reviews: dict[tuple[str, str, str], dict], threshold: str,
         validation: dict[tuple[str, str], dict] | None = None) -> list[dict]:
    """Compact table rows (one per event); ``validation``: audit rows (0.25 m only)."""
    out = []
    for r in recs:
        a, b = r["player_a"], r["player_b"]
        cx = r.get("context")
        key = (str(r["match_id"]), threshold, r["event_id"])
        out.append(dict(
            event_id=r["event_id"], period=r["period"], t_star_s=round(r["t_star_s"], 3),
            player_a=f"{a['side']} #{a['jersey']}", player_b=f"{b['side']} #{b['jersey']}",
            team_relation="same" if r["team_relation"] == "same_team" else "opposite",
            d_min_cm=round(r["d_min_m"] * 100, 2), duration_s=round(r["duration_s"], 3), quality=r["pair_quality"],
            contact_class=r["contact_class"], coincident=r["coincident_interval"],
            context_complete=None if cx is None else cx["complete"],
            high_quality=(str(r["match_id"]), r["event_id"]) in hq if threshold == PRIMARY else None,
            presentation_ready=(validation or {}).get((str(r["match_id"]), r["event_id"]), {}).get("presentation_ready")
            if threshold == PRIMARY and validation else None,
            review=reviews.get(key, {}).get("status", "")))
    return out


def filter_rows(table: list[dict], *, qualities: set[str] | None = None, relation: str = "all", coincident: str = "all",
                high_quality_only: bool = False, ready_only: bool = False, search: str = "") -> list[dict]:
    s = search.strip().lower()
    out = []
    for r in table:
        if qualities is not None and r["quality"] not in qualities:
            continue
        if relation != "all" and r["team_relation"] != relation:
            continue
        if coincident == "exclude" and r["coincident"] or coincident == "only" and not r["coincident"]:
            continue
        if high_quality_only and not r["high_quality"]:
            continue
        if ready_only and not r.get("presentation_ready"):
            continue
        if s and s not in r["event_id"] and s not in r["player_a"].lower() and s not in r["player_b"].lower():
            continue
        out.append(r)
    return out


# ---------------------------------------------------------------- trajectories

@dataclass
class Context:
    """Trajectory context shown for one event. ``kind``: ``exported`` (its own 0.25 m export), ``related`` (the export of the
    0.25 m episode ``source_event_id``, verified at the event's t*) or ``none``."""
    kind: str
    note: str
    source_event_id: str | None = None
    record: dict | None = None                   # the 0.25 m record whose context is shown
    a: dict[str, np.ndarray] | None = None       # per player: t, x, y, visibility, row, frame, bits, before, edge
    b: dict[str, np.ndarray] | None = None
    sync: dict[str, np.ndarray] | None = None    # t, xa, ya, xb, yb, d, a_is_sample, b_is_sample


def arrays_for(npz: dict[str, np.ndarray], k: int) -> tuple[dict, dict, dict]:
    """Samples of A and B and the synchronized grid of export row ``k`` (time ordered)."""
    m = npz["sample_event_index"] == k
    sides = []
    for side in (0, 1):
        w = np.flatnonzero(m & (npz["sample_side"] == side))
        w = w[np.argsort(npz["sample_t"][w], kind="stable")]
        sides.append(dict(t=npz["sample_t"][w], x=npz["sample_x"][w], y=npz["sample_y"][w], visibility=npz["sample_visibility"][w],
                          row=npz["sample_source_row"][w], frame=npz["sample_frame"][w], bits=npz["sample_quality_bits"][w],
                          before=npz["sample_before_t_star"][w], edge=npz["sample_is_boundary"][w]))
    w = np.flatnonzero(npz["sync_event_index"] == k)
    w = w[np.argsort(npz["sync_t"][w], kind="stable")]
    sync = {k2: npz[f"sync_{k2}"][w] for k2 in ("t", "xa", "ya", "xb", "yb", "d", "a_is_sample", "b_is_sample")}
    return sides[0], sides[1], sync


def position(side: dict[str, np.ndarray], t: float) -> tuple[float, float] | None:
    """Linear position on a player's valid run at ``t``; None outside its samples (never extrapolated)."""
    if not len(side["t"]) or t < side["t"][0] - 1e-9 or t > side["t"][-1] + 1e-9:
        return None
    return float(np.interp(t, side["t"], side["x"])), float(np.interp(t, side["t"], side["y"]))


def _verified(rec: dict, a: dict, b: dict) -> bool:
    pa, pb = position(a, rec["t_star_s"]), position(b, rec["t_star_s"])
    if pa is None or pb is None:
        return False
    ea, eb = rec["player_a"]["position_at_t_star"], rec["player_b"]["position_at_t_star"]
    return max(abs(pa[0] - ea[0]), abs(pa[1] - ea[1]), abs(pb[0] - eb[0]), abs(pb[1] - eb[1])) <= POS_TOL_M


def context_for(raw_root: str, match_id: str, threshold: str, rec: dict, *, primary: list[dict] | None = None,
                npz: dict[str, np.ndarray] | None = None, events: dict[str, list[dict]] | None = None) -> Context:
    """The trajectory context of an event of any threshold (see the module docstring); never another event's trajectory unless
    it reproduces this event's positions at t*."""
    events = events or {}
    primary = primary if primary is not None else events.get(PRIMARY) or load_events(raw_root, match_id, PRIMARY)
    line = {r["event_id"]: k for k, r in enumerate(primary)}
    if threshold == PRIMARY:
        cand = [rec["event_id"]]
    elif threshold in ("0m", "0.1m"):
        cur, thr = rec, threshold                                        # follow parents up to 0.25 m
        while thr != PRIMARY and cur is not None:
            nxt = THRESHOLDS[THRESHOLDS.index(thr) + 1]
            ev = events.get(nxt) or load_events(raw_root, match_id, nxt)
            cur = next((r for r in ev if r["event_id"] == cur.get("parent_event_id")), None)
            thr = nxt
        cand = [cur["event_id"]] if cur is not None else []
    else:
        cand = list(rec.get("children", {}).get(PRIMARY, []))
    if not cand:
        return Context("none", f"No 0.25 m episode is related to this {threshold} event (its minimum distance is "
                               f"{rec['d_min_m'] * 100:.1f} cm); trajectory context is exported for 0.25 m episodes only.")
    npz = npz if npz is not None else load_npz(raw_root, match_id)
    for eid in cand:
        k = line[eid]
        a, b, sync = arrays_for(npz, k)
        if _verified(rec, a, b):
            kind = "exported" if threshold == PRIMARY else "related"
            note = ("Event-centred 5 s context of this 0.25 m episode." if kind == "exported" else
                    f"Context of the related 0.25 m episode {eid} (same pair; its window contains this event's t* and reproduces "
                    f"A(t*), B(t*) to {POS_TOL_M:g} m). Window and completeness refer to that episode.")
            return Context(kind, note, eid, primary[k], a, b, sync)
    if threshold == PRIMARY:
        raise ExplorerError(f"0.25 m event {rec['event_id']} does not reproduce its own t* positions from its trajectory export")
    return Context("none", f"The related 0.25 m episode(s) {', '.join(cand)} have exported windows that do not contain this event's "
                           f"t*; its trajectory context has not been exported.")


# ---------------------------------------------------------------- manual reviews (audit notes, not labels)

def load_reviews(path: Path) -> dict[tuple[str, str, str], dict]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8", newline="") as f:
        return {(r["match_id"], r["threshold"], r["event_id"]): r for r in csv.DictReader(f)}


def reviews_csv(reviews: dict[tuple[str, str, str], dict]) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(REVIEW_FIELDS), lineterminator="\n")
    w.writeheader()
    for k in sorted(reviews):
        w.writerow({f: reviews[k].get(f, "") for f in REVIEW_FIELDS})
    return buf.getvalue().encode("utf-8")


def save_review(path: Path, match_id: str, threshold: str, event_id: str, status: str, note: str = "") -> dict:
    """Insert or replace one review (key: match, threshold, event) and rewrite the review file atomically."""
    if status not in REVIEW_STATUSES:
        raise ExplorerError(f"review status must be one of {REVIEW_STATUSES}")
    if threshold not in THRESHOLDS:
        raise ExplorerError(f"unknown threshold {threshold}")
    reviews = load_reviews(path)
    row = dict(match_id=str(match_id), threshold=threshold, event_id=event_id, status=status, note=" ".join(note.split())[:500],
               reviewed_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    reviews[(row["match_id"], threshold, event_id)] = row
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(reviews_csv(reviews))
    os.replace(tmp, path)
    return row


# ---------------------------------------------------------------- navigation and plain-language descriptions

PITCH_HALF = (52.5, 34.0)                        # 105 x 68 m, origin at the centre spot
STEP_S = 0.1
NOMINAL_OFFSETS = [round(-HALF_S + k * STEP_S, 1) for k in range(int(round(2 * HALF_S / STEP_S)) + 1)]   # -2.5 ... +2.5 s
PHASE_OFFSET_S = 1.0                             # "Before" / "After": t* -/+ 1 s on the nominal axis (the same for every event)
REASON_TEXT = dict(implausible_jump="a tracking jump (faster than 12 m/s)",
                   gap_period_or_time="a tracking gap (> 0.5 s) or a period boundary",
                   unattributed_or_ambiguous="an ambiguous provider sample",
                   no_adjacent_sample="the player is not tracked there (entry, exit or substitution)",
                   period_boundary="the start / end of the match period",
                   outside_period="outside the match period")


def neighbour(ids: list[str], cur: str | None, step: int) -> str | None:
    """Previous (step -1) / next (+1) event of the filtered list, clamped at the ends; the first one when ``cur`` is not listed."""
    if not ids:
        return None
    if cur not in ids:
        return ids[0]
    return ids[min(max(ids.index(cur) + step, 0), len(ids) - 1)]


def phase_offset(phase: str) -> float:
    """Offset from t* of the Before / At encounter / After buttons (fixed: the same for every event)."""
    return {"before": -PHASE_OFFSET_S, "at": 0.0, "after": PHASE_OFFSET_S}[phase]


def unavailable_reason(cx: dict, key: str, t: float) -> str | None:
    """Why ``player_a`` / ``player_b`` has no valid position at ``t`` (None when it has one)."""
    c = cx[key]
    if c["covered_start_s"] - 1e-9 <= t <= c["covered_end_s"] + 1e-9:
        return None
    for t0, t1, why in c["unavailable"]:
        if t0 - 1e-9 <= t <= t1 + 1e-9:
            return why
    return "outside_context"


def state_at(ctx: Context, t: float) -> dict[str, Any]:
    """Both players at time ``t``: the position on the player's valid run (linear between its samples, the extraction's model),
    or None with the reason it is unavailable; the synchronized distance only when both exist. Nothing is extrapolated."""
    cx = ctx.record["context"]
    out: dict[str, Any] = {}
    for k, side in (("a", ctx.a), ("b", ctx.b)):
        p = position(side, t)
        out[k] = p
        out[f"why_{k}"] = None if p is not None else (unavailable_reason(cx, f"player_{k}", t) or "outside_context")
    out["d"] = None if out["a"] is None or out["b"] is None else float(np.hypot(out["a"][0] - out["b"][0], out["a"][1] - out["b"][1]))
    return out


def why_text(why: str | None) -> str:
    return "outside the exported context" if why == "outside_context" else REASON_TEXT.get(why or "", (why or "").replace("_", " "))


def identical_seconds(ctx: Context) -> float:
    """Time of the context on which both players have identical coordinates (provider clone); d = 0 at both ends of a
    synchronized interval means zero throughout (relative motion is linear)."""
    if ctx.kind == "none" or len(ctx.sync["t"]) < 2:
        return 0.0
    z = ctx.sync["d"] <= 1e-9
    return float(np.diff(ctx.sync["t"])[z[:-1] & z[1:]].sum())


def encounter_kind(rec: dict, ctx: Context | None = None) -> str:
    if rec["coincident_interval"] or (ctx is not None and identical_seconds(ctx) > 0):
        return "Potential tracking ambiguity (identical provider positions)"
    return "Exact synchronized coincidence" if rec["d_min_m"] == 0 else "Closest synchronized approach"


# ---------------------------------------------------------------- validation (``m11-sync-proximity-v3-audit``, read-only)

VALIDATION_SCHEMA = "m11-sync-proximity-v3-audit"
_BOOL = ("mathematically_valid", "tracking_support_valid", "incoming_complete", "outgoing_complete", "context_complete",
         "provider_artifact_suspected", "presentation_ready", "exact_coincidence", "coincident_interval")


def validation_path(raw_root: str) -> Path:
    return approved_output_root(raw_root) / VALIDATION_SCHEMA / "event_validation.csv"


def load_validation(raw_root: str) -> dict[tuple[str, str], dict]:
    """Per-event audit statuses of the 0.25 m episodes on their v3 contexts, keyed (match, event); empty when not written."""
    p = validation_path(raw_root)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        out = {}
        for r in csv.DictReader(f):
            for k in _BOOL:
                r[k] = r[k] == "True"
            out[(r["match_id"], r["event_id"])] = r
    return out


def reason_texts(v: dict | None) -> list[str]:
    from football_fatigue.proximity_audit_v3 import REASONS

    return [REASONS.get(x, x) for x in (v or {}).get("reasons", "").split(";") if x]


def out_of_pitch(ctx: Context) -> int:
    """Trajectory samples outside the pitch lines (drawn as they are, never clipped or rescaled)."""
    if ctx.kind == "none":
        return 0
    return int(sum(np.sum(((np.abs(s["x"]) > PITCH_HALF[0]) | (np.abs(s["y"]) > PITCH_HALF[1])) & ~s["edge"]) for s in (ctx.a, ctx.b)))


def context_summary(cx: dict | None) -> str:
    if cx is None:
        return "not available"
    if cx["complete"]:
        return "complete: both players tracked over the whole window t* ± 2.5 s"
    parts = [f"{p[-1].upper()}'s track {word} ({why_text(v)})"
             for p in ("player_a", "player_b") for k, word in (("cut_before", "starts late"), ("cut_after", "ends early"))
             if (v := cx[p].get(k))]
    return "incomplete: " + ("; ".join(parts) or "a track is shorter than the window")


def coverage_texts(rec: dict, ctx: Context) -> list[str]:
    """Where a player's valid track does not cover the nominal window: how much is there and why it stops (never bridged)."""
    if ctx.kind == "none":
        return []
    cx, out = ctx.record["context"], []
    for key in ("player_a", "player_b"):
        p, c = key[-1].upper(), cx[key]
        for k, word, have in (("cut_before", "incoming", c["incoming_s"]), ("cut_after", "outgoing", c["outgoing_s"])):
            if (v := c.get(k)) is None:
                continue
            if v == "period_boundary":
                out.append(f"Player {p}'s {word} track covers {have:.2f} s of the {HALF_S:.1f} s window: the match period "
                           f"{'starts' if word == 'incoming' else 'ends'} there.")
                continue
            out.append(f"No {word} track for Player {p}: the valid track {'starts' if word == 'incoming' else 'ends'} at the encounter "
                       f"(cut by {why_text(v)})." if have <= 1e-6 else
                       f"Player {p}'s {word} track covers {have:.2f} s of the {HALF_S:.1f} s window (cut by {why_text(v)}); it is not "
                       "joined across the cut.")
    return out


def notes(rec: dict, ctx: Context, v: dict | None = None) -> list[tuple[str, str]]:
    """Plain-language reliability notes as (level, text), level ``warning`` or ``info``. None is hidden in presentation mode.
    ``v``: the event's audit row (optional)."""
    out = []
    same = identical_seconds(ctx)
    if rec["coincident_interval"] or same > 0:
        out.append(("warning", "Potential tracking ambiguity: the provider gives both players identical coordinates "
                               f"{f'for {same:.2f} s of the context ' if same > 0 else ''}(cloned estimates); the two tracks are drawn "
                               "on top of each other there. This is a data artifact, not evidence of physical contact or occlusion."))
    elif rec["d_min_m"] == 0:
        out.append(("info", "Exact synchronized coincidence: both players at the same coordinates at the same instant (d = 0 on "
                            "the provider's whole-millimetre positions), not over an interval."))
    est = [k.upper() for k in "ab" if rec[f"player_{k}"]["status_at_t_star"] != "observed"]
    if est:
        out.append(("warning", f"Player {' and '.join(est)} {'is' if len(est) == 1 else 'are'} estimated by the provider at the "
                               "encounter, not directly observed: the position is a fill-in, not a measurement."))
    cx = (ctx.record or {}).get("context") if ctx.kind != "none" else rec.get("context")
    out += [("warning", t) for t in coverage_texts(rec, ctx)]
    if cx and not cx["complete"] and ctx.kind == "none":
        out.append(("warning", f"Context {context_summary(cx)}. The trajectory is not joined across the cut."))
    if cx and cx["critical_flags"]:
        out.append(("warning", "Tracking-quality flags in the context: "
                               f"{', '.join(f.lower().replace('_', ' ') for f in cx['critical_flags'])}."))
    if n := out_of_pitch(ctx):
        out.append(("warning", f"{n} trajectory sample(s) lie outside the 105 × 68 m pitch lines; they are drawn at the provider "
                               "coordinates (marked ×), not clipped."))
    if ctx.kind == "related":
        out.append(("info", ctx.note))
    elif ctx.kind == "none":
        out.append(("warning", ctx.note))
    if v is not None and not v["presentation_ready"]:
        out.append(("info", "Not presentation-ready (audit): " + "; ".join(reason_texts(v)) + "."))
    return out


# ---------------------------------------------------------------- figures (Plotly)

COLORS = dict(a="#00e5ff", b="#ffd600")          # distinct on the green pitch; colours never encode a pairing decision
WIDTH = dict(a=5.0, b=2.5)                       # A under a narrower B: where the tracks coincide both stay visible
VIS = {1: "VISIBLE", 2: "ESTIMATED"}
GREY = "#9e9e9e"
AXIS = [-HALF_S - 0.05, HALF_S + 0.05]           # the same relative time axis for every event


def _arrows(path: np.ndarray, k: int = 2) -> list[tuple[float, float, float]]:
    """``k`` direction markers evenly along a path: (x, y, angle in degrees clockwise from +y, Plotly's marker convention)."""
    if len(path) < 2:
        return []
    seg = np.diff(path, axis=0)
    ln = np.hypot(seg[:, 0], seg[:, 1])
    if ln.sum() < 0.3:                                                   # (nearly) standing still: no meaningful direction
        return []
    cum = np.r_[0.0, np.cumsum(ln)]
    out = []
    for f in np.arange(1, k + 1) / (k + 1):
        s = f * cum[-1]
        i = min(int(np.searchsorted(cum, s, side="right")) - 1, len(seg) - 1)
        p = path[i] + (s - cum[i]) / ln[i] * seg[i] if ln[i] > 0 else path[i]
        out.append((float(p[0]), float(p[1]), float(np.degrees(np.arctan2(seg[i, 0], seg[i, 1])))))
    return out


def _split(side: dict[str, np.ndarray], t_star: float) -> tuple[np.ndarray, np.ndarray]:
    """Path before (up to and including the position at t*) and after t*, on the player's own valid run."""
    p = position(side, t_star)
    pre = side["t"] < t_star
    bx, by = side["x"][pre], side["y"][pre]
    ax, ay = side["x"][~pre], side["y"][~pre]
    if p is not None:
        bx, by = np.r_[bx, p[0]], np.r_[by, p[1]]
        ax, ay = np.r_[p[0], ax], np.r_[p[1], ay]
    return np.c_[bx, by], np.c_[ax, ay]


def pitch_figure(ctx: Context, rec: dict, t_now: float, *, r_m: float, full_pitch: bool = False, height: int = 560):
    import plotly.graph_objects as go

    from .pitch import PITCH_COLOR, pitch_shapes

    fig = go.Figure()
    ts = rec["t_star_s"]
    pts = []
    cx = ctx.record["context"]
    for key, side in (("a", ctx.a), ("b", ctx.b)):
        pl = rec[f"player_{key}"]
        col, name = COLORS[key], f"{key.upper()} · {pl['side']} #{pl['jersey']}"
        w = WIDTH[key]
        pre, post = _split(side, ts)                                     # both on the player's single valid run: no gap is bridged
        fig.add_scatter(x=pre[:, 0], y=pre[:, 1], mode="lines", line=dict(color=col, width=w), name=f"{name} incoming", legendgroup=key,
                        hoverinfo="skip")
        fig.add_scatter(x=post[:, 0], y=post[:, 1], mode="lines", line=dict(color=col, width=w, dash="dot"),
                        name=f"{name} outgoing", legendgroup=key, hoverinfo="skip")
        for end, k in ((0, "cut_before"), (-1, "cut_after")):            # the valid track stops here: nothing is drawn beyond
            why = cx[f"player_{key}"].get(k)
            if why and why != "period_boundary":
                fig.add_scatter(x=[side["x"][end]], y=[side["y"][end]], mode="markers", legendgroup="cut", showlegend=False, name=f"{name} cut",
                                marker=dict(symbol="line-ns", size=14, angle=0, line=dict(color="#ff1744", width=3)),
                                hovertext=f"{name}: valid track {'starts' if end == 0 else 'ends'} here ({why_text(why)})", hoverinfo="text")
        o = ~side["edge"]                                                # provider samples only (window-edge points are interpolations)
        hover = [f"{name}<br>t* {t - ts:+.3f} s<br>row {r}, frame {f}<br>{VIS.get(int(v), 'unknown')}"
                 for t, r, f, v in zip(side["t"][o], side["row"][o], side["frame"][o], side["visibility"][o])]
        sym = np.where(side["visibility"][o] == 1, "circle", "circle-open")
        fig.add_scatter(x=side["x"][o], y=side["y"][o], mode="markers", marker=dict(color=col, size=3, symbol=sym, line=dict(color=col, width=1)),
                        opacity=0.55, hovertext=hover, hoverinfo="text", name=f"{name} samples", legendgroup=key, showlegend=False)
        arr = _arrows(pre) + _arrows(post)
        if arr:
            fig.add_scatter(x=[p[0] for p in arr], y=[p[1] for p in arr], mode="markers", hoverinfo="skip", showlegend=False,
                            marker=dict(symbol="triangle-up", size=10, angle=[p[2] for p in arr], color=col, line=dict(color="black", width=1)),
                            name=f"{name} direction", legendgroup=key)
        pos = pl["position_at_t_star"]
        fig.add_scatter(x=[pos[0]], y=[pos[1]], mode="markers", opacity=0.9, name=f"{key.upper()}(t*)", legendgroup=key,
                        marker=dict(color=col, size=11, symbol="diamond", line=dict(color="black", width=1.5)),
                        hovertext=f"{key.upper()}(t*) = ({pos[0]:.3f}, {pos[1]:.3f}) m", hoverinfo="text")
        pts.append(np.r_[pre, post])
    for name, sym in (("observed sample", "circle"), ("estimated sample", "circle-open")):   # legend keys only
        fig.add_scatter(x=[None], y=[None], mode="markers", marker=dict(color="#616161", size=7, symbol=sym, line=dict(color="#616161", width=1.5)),
                        name=name)
    if any(cx[f"player_{k}"].get(c) not in (None, "period_boundary") for k in "ab" for c in ("cut_before", "cut_after")):
        fig.add_scatter(x=[None], y=[None], mode="markers", marker=dict(symbol="line-ns", size=12, line=dict(color="#ff1744", width=3)),
                        name="valid track cut (not joined)", legendgroup="cut")
    sy = ctx.sync
    z = sy["d"] <= 1e-9
    same = np.flatnonzero(z[:-1] & z[1:])
    if len(same):                                                        # provider-cloned stretch: both tracks lie on top of each other
        xs, ys = [], []
        for i in same:
            xs += [sy["xa"][i], sy["xa"][i + 1], None]
            ys += [sy["ya"][i], sy["ya"][i + 1], None]
        fig.add_scatter(x=xs, y=ys, mode="lines", line=dict(color="#ff1744", width=1.5, dash="dash"), hoverinfo="skip",
                        name=f"A and B identical ({float(np.diff(sy['t'])[same].sum()):.2f} s, provider clone)")
    mid, d = rec["midpoint"], rec["d_min_m"]
    label = ("P* exact synchronized coincidence (d = 0)" if d == 0 else
             f"P* midpoint of A(t*), B(t*) at the closest synchronized approach (d_min {d * 100:.1f} cm; not a path intersection)")
    fig.add_scatter(x=[mid[0]], y=[mid[1]], mode="markers", marker=dict(color="white", size=12, symbol="star", line=dict(color="black", width=1)),
                    opacity=0.9, name=label, hovertext=label, hoverinfo="text")
    fig.add_shape(type="circle", x0=mid[0] - r_m, y0=mid[1] - r_m, x1=mid[0] + r_m, y1=mid[1] + r_m, line=dict(color="white", width=1, dash="dot"))
    pts = np.concatenate(pts)
    off = (np.abs(pts[:, 0]) > PITCH_HALF[0]) | (np.abs(pts[:, 1]) > PITCH_HALF[1])
    if off.any():                                                        # shown at the provider coordinates, never clipped
        fig.add_scatter(x=pts[off, 0], y=pts[off, 1], mode="markers", marker=dict(symbol="x-thin", size=9, line=dict(color="#ff1744", width=2)),
                        name="outside the pitch lines", hoverinfo="skip")
    now = state_at(ctx, t_now)                                           # selected time: only players with a valid position
    have = [(k, now[k]) for k in "ab" if now[k] is not None]
    if have:
        txt = (f"selected time: t* {t_now - ts:+.2f} s, d = {now['d'] * 100:.1f} cm" if now["d"] is not None else
               f"selected time: t* {t_now - ts:+.2f} s, " + ", ".join(f"{k.upper()} unavailable" for k in "ab" if now[k] is None))
        fig.add_scatter(x=[p[0] for _, p in have], y=[p[1] for _, p in have], mode="lines+markers" if len(have) == 2 else "markers",
                        line=dict(color="white", width=1.5, dash="dot"), hoverinfo="skip", name=txt,
                        marker=dict(size=17, color=[COLORS[k] for k, _ in have], symbol="circle-open", line=dict(width=3)))
    if full_pitch:
        lo, hi = np.minimum(pts.min(axis=0), (-55.0, -36.0)), np.maximum(pts.max(axis=0), (55.0, 36.0))
    else:                                                                # auto-zoom on the context (at least 8 m wide)
        lo, hi = pts.min(axis=0) - 1.5, pts.max(axis=0) + 1.5
        c, half = (lo + hi) / 2, np.maximum((hi - lo) / 2, 4.0)
        lo, hi = c - half, c + half
    fig.update_layout(shapes=list(fig.layout.shapes) + pitch_shapes(), plot_bgcolor=PITCH_COLOR, height=height,
                      margin=dict(l=10, r=10, t=10, b=10), hovermode="closest",
                      legend=dict(orientation="h", x=0, xanchor="left", y=-0.13, yanchor="top", font=dict(size=11)),
                      xaxis=dict(range=[lo[0], hi[0]], title="x (m)", showgrid=False, zeroline=False),
                      yaxis=dict(range=[lo[1], hi[1]], title="y (m)", showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1))
    return fig


def coverage_figure(ctx: Context, rec: dict, t_now: float, *, height: int = 150):
    """Valid support of Player A, Player B and both (shared) on the fixed axis t* -2.5 ... +2.5 s; the rest of the nominal window
    grey (hover: why it is unavailable). Relative to the shown event's t*."""
    import plotly.graph_objects as go

    ts, cx = rec["t_star_s"], ctx.record["context"]
    fig = go.Figure()
    ns, ne = cx["nominal_start_s"] - ts, cx["nominal_end_s"] - ts
    for name, c0, c1, col in (("A", cx["player_a"]["covered_start_s"], cx["player_a"]["covered_end_s"], COLORS["a"]),
                              ("B", cx["player_b"]["covered_start_s"], cx["player_b"]["covered_end_s"], COLORS["b"]),
                              ("both", cx["shared_start_s"], cx["shared_end_s"], "#43a047")):
        fig.add_scatter(x=[ns, ne], y=[name, name], mode="lines", line=dict(color=GREY, width=14), hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=[c0 - ts, c1 - ts], y=[name, name], mode="lines", line=dict(color=col, width=14), showlegend=False,
                        name=f"{name} valid", hovertemplate=f"{name}: valid t* {c0 - ts:+.2f} … {c1 - ts:+.2f} s<extra></extra>")
    for key, name in (("player_a", "A"), ("player_b", "B")):
        for t0, t1, why in cx[key]["unavailable"]:
            fig.add_scatter(x=[(t0 + t1) / 2 - ts], y=[name], mode="markers", marker=dict(size=14, color="rgba(0,0,0,0)"), showlegend=False,
                            name=f"{name} unavailable",
                            hovertemplate=f"{name} unavailable t* {t0 - ts:+.2f} … {t1 - ts:+.2f} s: {why_text(why)}<extra></extra>")
    fig.add_vline(x=0, line=dict(color="#212121", width=1.5), annotation_text="t* = 0", annotation_position="top")
    fig.add_vline(x=t_now - ts, line=dict(color="#d81b60", width=2))
    fig.add_scatter(x=[None], y=[None], mode="markers", marker=dict(symbol="square", size=10, color=GREY), name="no valid track (nothing drawn)")
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=24, b=10), xaxis=dict(range=AXIS, dtick=0.5),
                      yaxis=dict(categoryorder="array", categoryarray=["both", "B", "A"]),
                      legend=dict(orientation="h", x=1, xanchor="right", y=1.0, yanchor="bottom", font=dict(size=10)))
    return fig


def distance_figure(ctx: Context, rec: dict, t_now: float, *, r_m: float, height: int = 300):
    import plotly.graph_objects as go

    ts = rec["t_star_s"]
    s = ctx.sync
    rel = s["t"] - ts
    both = s["a_is_sample"] & s["b_is_sample"]
    cx = ctx.record["context"]
    fig = go.Figure()
    gaps = [(x0, x1) for x0, x1 in ((cx["nominal_start_s"] - ts, rel[0]), (rel[-1], cx["nominal_end_s"] - ts)) if x1 - x0 > 1e-6]
    for x0, x1 in gaps:                                                  # nominal window without shared valid tracks
        fig.add_vrect(x0=x0, x1=x1, fillcolor=GREY, opacity=0.3, line_width=0)
    if gaps:
        fig.add_scatter(x=[None], y=[None], mode="markers", marker=dict(symbol="square", size=10, color="rgba(158,158,158,0.5)"),
                        name="no shared valid track (not drawn)")
    fig.add_vrect(x0=rec["t_start_s"] - ts, x1=max(rec["t_end_s"] - ts, rec["t_start_s"] - ts + 1e-3), fillcolor="#43a047", opacity=0.2,
                  line_width=0)
    fig.add_scatter(x=[None], y=[None], mode="markers", marker=dict(symbol="square", size=10, color="rgba(67,160,71,0.35)"),
                    name="encounter (d ≤ R)")
    fig.add_scatter(x=rel, y=s["d"], mode="lines", line=dict(color="#212121", width=2), name="distance d(t), same instant",
                    hovertemplate="t* %{x:+.3f} s<br>d = %{y:.3f} m<extra></extra>")
    fig.add_scatter(x=rel[both], y=s["d"][both], mode="markers", marker=dict(color="#212121", size=4), name="both at an original sample",
                    hoverinfo="skip")
    now = state_at(ctx, t_now)
    if now["d"] is not None:
        fig.add_scatter(x=[t_now - ts], y=[now["d"]], mode="markers", marker=dict(color="#d81b60", size=12, line=dict(color="white", width=1.5)),
                        name=f"selected time (d = {now['d'] * 100:.1f} cm)", hoverinfo="skip")
    fig.add_hline(y=r_m, line=dict(color="#2e7d32", dash="dash"), annotation_text=f"R = {r_m * 100:.0f} cm", annotation_position="top right")
    fig.add_vline(x=0, line=dict(color="#757575"), annotation_text="t*", annotation_position="top left")
    fig.add_vline(x=t_now - ts, line=dict(color="#d81b60", width=2))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=36, b=10), xaxis_title="time relative to the encounter t* (s)",
                      xaxis=dict(range=AXIS, dtick=0.5),
                      yaxis_title="distance (m)", yaxis=dict(rangemode="tozero", range=[0, max(1.0, min(float(s["d"].max()), 8.0))]),
                      legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom", font=dict(size=10)))
    return fig
