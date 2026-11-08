"""Scoring detected events against the PFF event labels (root version).

PFF labels are human annotations, not physical ground truth; they are the reference the detector is scored against.
Matching is per event type, one-to-one, greedy by smallest time difference, same period, |dt| <= tolerance. Truth events
outside the tracked time range (excluded periods, event-only extra time, period boundaries) are counted and reported as
``not_scoreable``, never silently dropped.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from football_fatigue.ingestion import EventRow

from .classify import CROSS, PASS, SHOT, DetectedEvent
from .config import DetectorConfig
from .inputs import MatchPositions

TOLERANCES_S = (0.5, 1.0, 2.0)
PRIMARY_TOLERANCE_S = 1.0
SCORED_TYPES = ("PA", "CR", "SH", "GOAL", "KEY_PASS", "PASS_ANY")
_ACTOR = {"PA": "passerPlayerId", "CR": "crosserPlayerId", "SH": "shooterPlayerId", "CL": "clearerPlayerId"}
_OUTCOME = {"PA": "passOutcomeType", "CR": "crossOutcomeType", "SH": "shotOutcomeType", "CL": "clearanceOutcomeType"}


@dataclass
class LabelledEvent:
    type: str
    t_s: float
    period: int
    team_id: int | None
    player_id: int | None
    outcome: str | None
    key_pass: bool
    ref: str                 # source event row id (truth) or detector index
    receiver_id: int | None = None
    x: float | None = None   # labelled ball position (truth: the event file's ball snapshot), for display only
    y: float | None = None


def truth_events(rows: Iterable[EventRow], cfg: DetectorConfig) -> list[LabelledEvent]:
    """PA, CR, SH, CL rows of the PFF file, with a derived key-pass flag (same rule as for detections)."""
    rows = sorted((r for r in rows if r.event_time_s is not None and r.canonical_period is not None), key=lambda r: r.event_time_s)
    out: list[LabelledEvent] = []
    for r in rows:
        typ = r.possession_event_type
        if typ not in _ACTOR:
            continue
        pe = r.attributes.get("possessionEvents") or {}
        balls = (getattr(r, "positions_snapshot", None) or {}).get("ball") or []
        b = balls[0] if len(balls) == 1 and isinstance(balls[0], dict) else {}
        out.append(LabelledEvent(typ, float(r.event_time_s), int(r.canonical_period), r.source_team_id, pe.get(_ACTOR[typ]) or r.source_player_id,
                                 pe.get(_OUTCOME[typ]), False, r.event_row_id, pe.get("receiverPlayerId"), b.get("x"), b.get("y")))
    # key pass: last completed PA/CR of the shooting team before a shot, no opponent possession event in between, within the window
    actions = [(float(r.event_time_s), int(r.canonical_period), r.source_team_id) for r in rows if r.possession_event_type not in (None, "FO")]
    times = np.array([a[0] for a in actions])
    for sh in (e for e in out if e.type == "SH"):
        for prev in reversed([e for e in out if e.t_s < sh.t_s and e.period == sh.period and sh.t_s - e.t_s <= cfg.key_pass_window_s]):
            if prev.team_id != sh.team_id:
                break
            if prev.type in ("PA", "CR") and prev.outcome == "C":
                lo, hi = np.searchsorted(times, prev.t_s, "right"), np.searchsorted(times, sh.t_s, "left")
                if all(actions[i][2] == sh.team_id for i in range(lo, hi)):
                    prev.key_pass = True
                break
    return out


def detected_as_labelled(pos: MatchPositions, events: Sequence[DetectedEvent]) -> list[LabelledEvent]:
    out = []
    for i, e in enumerate(events):
        pid = int(pos.player_id[e.touch.player])
        out.append(LabelledEvent(e.type, e.touch.t_s, e.touch.period, pos.team_id_of_side(e.team_side), pid if pid >= 0 else None,
                                 e.outcome, e.key_pass, f"det-{i}"))
    return out


def scoreable(truth: list[LabelledEvent], pos: MatchPositions) -> tuple[list[LabelledEvent], list[LabelledEvent]]:
    """Split truth into events inside the tracked time range of their period and the rest."""
    ranges = {int(p): (float(pos.t[pos.period == p].min()), float(pos.t[pos.period == p].max())) for p in np.unique(pos.period)}
    inside, outside = [], []
    for e in truth:
        r = ranges.get(e.period)
        (inside if r is not None and r[0] - 0.5 <= e.t_s <= r[1] + 0.5 else outside).append(e)
    return inside, outside


def _select(events: list[LabelledEvent], kind: str) -> list[LabelledEvent]:
    if kind == "GOAL":
        return [e for e in events if e.type == "SH" and e.outcome == "G"]
    if kind == "KEY_PASS":
        return [e for e in events if e.key_pass]
    if kind == "PASS_ANY":
        return [e for e in events if e.type in ("PA", "CR")]
    return [e for e in events if e.type == kind]


def match_events(pred: list[LabelledEvent], truth: list[LabelledEvent], tol: float) -> list[tuple[int, int, float]]:
    """One-to-one greedy matching by |dt|; returns (pred index, truth index, dt = pred - truth)."""
    pairs = []
    t_times = np.array([e.t_s for e in truth])
    order = np.argsort(t_times) if truth else np.array([], int)
    ts = t_times[order]
    for i, p in enumerate(pred):
        lo, hi = np.searchsorted(ts, p.t_s - tol, "left"), np.searchsorted(ts, p.t_s + tol, "right")
        for jj in range(lo, hi):
            j = int(order[jj])
            if truth[j].period == p.period:
                pairs.append((abs(p.t_s - truth[j].t_s), i, j))
    pairs.sort()
    used_p, used_t, out = set(), set(), []
    for d, i, j in pairs:
        if i in used_p or j in used_t:
            continue
        used_p.add(i)
        used_t.add(j)
        out.append((i, j, pred[i].t_s - truth[j].t_s))
    return out


def score(pred: list[LabelledEvent], truth: list[LabelledEvent]) -> dict[str, Any]:
    """Counts per type and tolerance (tp, fp, fn), attribute agreement on matched pairs, a confusion table, timing errors."""
    res: dict[str, Any] = {"counts": {}, "attributes": {}, "timing_dt_s": {}}
    for kind in SCORED_TYPES:
        P, T = _select(pred, kind), _select(truth, kind)
        for tol in TOLERANCES_S:
            m = match_events(P, T, tol)
            res["counts"][f"{kind}@{tol}"] = dict(tp=len(m), fp=len(P) - len(m), fn=len(T) - len(m), n_pred=len(P), n_truth=len(T))
            if tol == PRIMARY_TOLERANCE_S:
                res["timing_dt_s"][kind] = [round(d, 3) for _, _, d in m]
                if kind in ("PASS_ANY", "SH"):
                    a = Counter()
                    for i, j, _ in m:
                        p, t = P[i], T[j]
                        a["matched"] += 1
                        a["team_agree"] += p.team_id == t.team_id
                        if t.player_id is not None and p.player_id is not None:
                            a["player_comparable"] += 1
                            a["player_agree"] += p.player_id == t.player_id
                        if kind == "PASS_ANY" and t.outcome is not None and p.outcome is not None:
                            a["outcome_comparable"] += 1
                            a["outcome_agree"] += (p.outcome == "C") == (t.outcome == "C")
                    res["attributes"][kind] = dict(a)
    # confusion: truth PA/CR/SH/CL vs any detection (one-to-one across types) at the primary tolerance
    T = [e for e in truth if e.type in ("PA", "CR", "SH", "CL")]
    m = match_events(pred, T, PRIMARY_TOLERANCE_S)
    conf = Counter()
    hit_t = {j for _, j, _ in m}
    for i, j, _ in m:
        conf[(T[j].type, pred[i].type)] += 1
    for j, t in enumerate(T):
        if j not in hit_t:
            conf[(t.type, "missed")] += 1
    hit_p = {i for i, _, _ in m}
    for i, p in enumerate(pred):
        if i not in hit_p:
            conf[("no_event", p.type)] += 1
    res["confusion"] = {f"{a}->{b}": c for (a, b), c in sorted(conf.items())}
    return res


def merge_scores(scores: Iterable[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"counts": {}, "attributes": {}, "timing_dt_s": {}, "confusion": Counter()}
    for s in scores:
        for k, c in s["counts"].items():
            acc = out["counts"].setdefault(k, Counter())
            acc.update(c)
        for k, c in s["attributes"].items():
            out["attributes"].setdefault(k, Counter()).update(c)
        for k, v in s["timing_dt_s"].items():
            out["timing_dt_s"].setdefault(k, []).extend(v)
        out["confusion"].update(s["confusion"])
    out["counts"] = {k: dict(v) for k, v in out["counts"].items()}
    out["attributes"] = {k: dict(v) for k, v in out["attributes"].items()}
    out["confusion"] = dict(sorted(out["confusion"].items()))
    return out


def prf(c: dict[str, int]) -> tuple[float | None, float | None, float | None]:
    p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else None
    r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else None
    f = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
    return p, r, f


_NAMES = {"PA": "Pass (excl. cross)", "CR": "Cross", "SH": "Shot", "GOAL": "Goal", "KEY_PASS": "Key pass", "PASS_ANY": "Pass incl. cross"}


def fmt(v: float | None) -> str:
    return "-" if v is None else f"{v:.3f}"


def render_markdown(title: str, merged: dict[str, Any], per_match: dict[int, dict[str, Any]], meta: dict[str, Any]) -> str:
    """Human-readable scorecard. Every number is computed from ``merged``/``per_match``; nothing is typed in."""
    L = [f"# {title}", "", f"Detector `{meta['detector_version']}`; matches: {len(per_match)} ({meta['split']}); reference labels: PFF event data "
         f"(root version), scored on events inside the tracked time range ({meta['not_scoreable']} truth events outside it are not scored).", "",
         f"## Detection quality at ±{PRIMARY_TOLERANCE_S:g} s", "", "| Event | Truth | Detected | Precision | Recall | F1 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for k in SCORED_TYPES:
        c = merged["counts"][f"{k}@{PRIMARY_TOLERANCE_S}"]
        p, r, f = prf(c)
        L.append(f"| {_NAMES[k]} | {c['n_truth']} | {c['n_pred']} | {fmt(p)} | {fmt(r)} | {fmt(f)} |")
    L += ["", "## F1 by time tolerance", "", "| Event | " + " | ".join(f"±{t:g} s" for t in TOLERANCES_S) + " |", "| --- |" + " ---: |" * len(TOLERANCES_S)]
    for k in SCORED_TYPES:
        L.append(f"| {_NAMES[k]} | " + " | ".join(fmt(prf(merged["counts"][f"{k}@{t}"])[2]) for t in TOLERANCES_S) + " |")
    L += ["", "## Attributes of matched events", ""]
    for k, a in merged["attributes"].items():
        parts = [f"team correct {a.get('team_agree', 0)}/{a.get('matched', 0)}"]
        if a.get("player_comparable"):
            parts.append(f"player correct {a.get('player_agree', 0)}/{a['player_comparable']}")
        if a.get("outcome_comparable"):
            parts.append(f"completed-vs-not correct {a.get('outcome_agree', 0)}/{a['outcome_comparable']}")
        L.append(f"- {_NAMES[k]}: " + ", ".join(parts))
    dts = merged["timing_dt_s"].get("PASS_ANY", [])
    if dts:
        q = np.percentile(np.abs(dts), [50, 90])
        L.append(f"- Timing (passes, detected minus labelled): median |dt| {q[0]:.2f} s, 90th percentile {q[1]:.2f} s")
    L += ["", f"## Confusion (truth -> detection, ±{PRIMARY_TOLERANCE_S:g} s)", "", "| Truth \\ Detected | PA | CR | SH | missed |", "| --- | ---: | ---: | ---: | ---: |"]
    conf = merged["confusion"]
    for t in ("PA", "CR", "SH", "CL", "no_event"):
        L.append(f"| {t} | " + " | ".join(str(conf.get(f"{t}->{d}", 0)) for d in ("PA", "CR", "SH", "missed")) + " |")
    L += ["", "`CL` = PFF clearance (not a detector class); `no_event` = detections with no labelled PA/CR/SH/CL nearby (false alarms).", "",
          "## Per match (F1 at ±1 s)", "", "| Match | Pass incl. cross | Shot | Goals (truth/detected/hit) | Flags |", "| --- | ---: | ---: | --- | --- |"]
    for m, s in sorted(per_match.items()):
        g = s["counts"][f"GOAL@{PRIMARY_TOLERANCE_S}"]
        flags = ", ".join(s.get("flags", []))
        L.append(f"| {m} | {fmt(prf(s['counts'][f'PASS_ANY@{PRIMARY_TOLERANCE_S}'])[2])} | {fmt(prf(s['counts'][f'SH@{PRIMARY_TOLERANCE_S}'])[2])} | "
                 f"{g['n_truth']}/{g['n_pred']}/{g['tp']} | {flags} |")
    L += ["", "## Method and limits", "", *[f"- {x}" for x in meta["notes"]], ""]
    return "\n".join(L)
