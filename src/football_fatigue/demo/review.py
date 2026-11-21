"""M7.2 human-in-the-loop identity review: an append-only, versioned annotation layer beside (never inside) the predictions.

Events are stored one JSON object per line. They are never edited or deleted; ``undo`` is itself an event.
:func:`resolve` replays them into a reviewed state:

* ``confirm``: these safe segments are the person ``identity`` (a reviewer label such as "Blue 7").
* ``unknown``: these segments have no usable identity.
* ``reject_association``: the global-id link of this tracklet is rejected, so its segments fall back to their own tracker identity.
* ``split``: a tracklet holds different people before and after ``frame``.
* ``merge``: identity ``from`` is the same person as identity ``into``. This is accepted only with ``explicit: true``, after a review.
* ``undo``: withdraws an earlier event, identified by ``target_event``.

A contradiction is reported as a conflict and the affected segments are left unresolved. Contradictions are: one segment
confirmed as two identities, one identity in two places in the same frame, or a confirmation across a split.

Reviews are annotations, not model-validation evidence. They enable level-L2 aggregation only for segments resolved to a single
identity without conflict.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

REVIEW_SCHEMA = "m72-identity-review-v1"
ACTIONS = ("confirm", "unknown", "reject_association", "split", "merge", "undo", "handover_decision")
HANDOVER_DECISIONS = ("accept", "reject", "unsure")     # M7.3: a reviewer's decision on a suggested hand-over (never applied to predictions)


class ReviewError(ValueError):
    pass


def validate_event(e: dict[str, Any], segments: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Normalised event or ReviewError. ``segments``: segment id -> {trk, local, f0, f1}."""
    a = e.get("action")
    if a not in ACTIONS:
        raise ReviewError(f"action must be one of {', '.join(ACTIONS)}")
    reviewer = str(e.get("reviewer") or "").strip()
    if not reviewer:
        raise ReviewError("reviewer is required (provenance)")
    out = {"action": a, "reviewer": reviewer[:80], "note": str(e.get("note") or "")[:500]}
    if a in ("confirm", "unknown"):
        segs = e.get("segments")
        if not isinstance(segs, list) or not segs or not all(isinstance(s, int) and s in segments for s in segs):
            raise ReviewError("segments must be a non-empty list of existing segment ids")
        out["segments"] = sorted(set(segs))
        if a == "confirm":
            ident = str(e.get("identity") or "").strip()
            if not ident:
                raise ReviewError("confirm needs an identity label")
            out["identity"] = ident[:40]
    elif a in ("reject_association", "split"):
        trk = e.get("trk")
        if not isinstance(trk, int) or trk not in {s["trk"] for s in segments.values()}:
            raise ReviewError("trk must be an existing tracklet id")
        out["trk"] = trk
        if a == "split":
            f = e.get("frame")
            span = [s for s in segments.values() if s["trk"] == trk]
            if not isinstance(f, int) or not min(s["f0"] for s in span) < f <= max(s["f1"] for s in span):
                raise ReviewError("split frame must lie inside the tracklet")
            out["frame"] = f
    elif a == "merge":
        if e.get("explicit") is not True:
            raise ReviewError("merge requires explicit: true (merge only after an explicit review)")
        fr, into = str(e.get("from") or "").strip(), str(e.get("into") or "").strip()
        if not fr or not into or fr == into:
            raise ReviewError("merge needs two different identity labels 'from' and 'into'")
        out.update({"from": fr[:40], "into": into[:40], "explicit": True})
    elif a == "handover_decision":
        sug, dec, segs = str(e.get("suggestion") or "").strip(), e.get("decision"), e.get("segments")
        if not sug or dec not in HANDOVER_DECISIONS:
            raise ReviewError(f"handover_decision needs a suggestion id and decision in {', '.join(HANDOVER_DECISIONS)}")
        if not isinstance(segs, list) or len(segs) != 2 or not all(isinstance(x, int) and x in segments for x in segs) or segs[0] == segs[1]:
            raise ReviewError("handover_decision needs the two segment ids it is about")
        out.update({"suggestion": sug[:40], "decision": dec, "segments": segs})
    else:
        if not isinstance(e.get("target_event"), int):
            raise ReviewError("undo needs target_event (an event id)")
        out["target_event"] = e["target_event"]
    return out


class ReviewStore:
    """Append-only JSONL log. Event ids are 1-based line numbers; existing lines are never rewritten."""

    def __init__(self, path: Path, match_id: str, overlay_schema: str):
        self.path, self.match_id, self.overlay_schema = Path(path), match_id, overlay_schema

    def events(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def append(self, e: dict[str, Any], segments: dict[int, dict[str, Any]]) -> dict[str, Any]:
        ev = validate_event(e, segments)
        prior = self.events()
        if ev["action"] == "undo":
            ids = {p["id"] for p in prior}
            if ev["target_event"] not in ids:
                raise ReviewError("target_event does not exist")
            if any(p["action"] == "undo" and p["target_event"] == ev["target_event"] for p in prior):
                raise ReviewError("that event is already undone")
        ev.update({"id": len(prior) + 1, "schema": REVIEW_SCHEMA, "match_id": self.match_id, "overlay_schema": self.overlay_schema,
                   "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(ev) + "\n")
        return ev


def resolve(events: Iterable[dict[str, Any]], segments: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Reviewed state from the event log. Predictions are not read or changed."""
    evs = list(events)
    undone = {e["target_event"] for e in evs if e["action"] == "undo"}
    live = [e for e in evs if e["action"] != "undo" and e["id"] not in undone]
    alias: dict[str, str] = {}
    for e in live:
        if e["action"] == "merge":
            alias[e["from"]] = e["into"]

    def root(x: str) -> str:
        seen = set()
        while x in alias and x not in seen:
            seen.add(x); x = alias[x]
        return x

    claims: dict[int, list[tuple[str, int]]] = defaultdict(list)    # segment -> [(identity or '?', event id)]
    rejected, splits = set(), defaultdict(list)
    for e in live:
        if e["action"] == "confirm":
            for s in e["segments"]:
                claims[s].append((e["identity"], e["id"]))
        elif e["action"] == "unknown":
            for s in e["segments"]:
                claims[s].append(("?", e["id"]))
        elif e["action"] == "reject_association":
            rejected.add(e["trk"])
        elif e["action"] == "split":
            splits[e["trk"]].append(e["frame"])
    # a later claim on the same segment by the same reviewer revises an earlier one; different reviewers disagreeing is a conflict
    by_event = {e["id"]: e for e in live}
    state, conflicts = {}, []
    for s, cl in claims.items():
        latest: dict[str, tuple[str, int]] = {}
        for ident, eid in cl:
            latest[by_event[eid]["reviewer"]] = (ident, eid)
        ids = {root(i) if i != "?" else "?" for i, _ in latest.values()}
        if len(ids) > 1:
            conflicts.append({"type": "contradictory_labels", "segment": s, "labels": sorted(ids), "events": sorted(e for _, e in latest.values())})
            continue
        ident = ids.pop()
        state[s] = {"identity": None if ident == "?" else ident, "status": "unknown" if ident == "?" else "confirmed",
                    "events": sorted(e for _, e in latest.values())}
    # a confirmed identity across a split of one tracklet
    for trk, fs in splits.items():
        for f in fs:
            before = {state[s]["identity"] for s in state if segments[s]["trk"] == trk and segments[s]["f1"] < f and state[s]["identity"]}
            after = {state[s]["identity"] for s in state if segments[s]["trk"] == trk and segments[s]["f0"] >= f and state[s]["identity"]}
            for ident in before & after:
                bad = [s for s in state if segments[s]["trk"] == trk and state[s]["identity"] == ident]
                conflicts.append({"type": "identity_across_split", "trk": trk, "frame": f, "identity": ident, "segments": bad})
                for s in bad:
                    state[s] = dict(state[s], status="conflict")
    # one identity in two places at once
    per_id = defaultdict(list)
    for s, st in state.items():
        if st["status"] == "confirmed":
            per_id[st["identity"]].append(s)
    for ident, ss in per_id.items():
        ss.sort(key=lambda s: segments[s]["f0"])
        for i, a in enumerate(ss):
            for b in ss[i + 1:]:
                if segments[b]["f0"] <= segments[a]["f1"]:     # segments of one tracker id never overlap, so this is two people
                    conflicts.append({"type": "overlapping_identity", "identity": ident, "segments": [a, b]})
                    state[a] = dict(state[a], status="conflict"); state[b] = dict(state[b], status="conflict")
    identities = defaultdict(list)
    for s, st in state.items():
        if st["status"] == "confirmed":
            identities[st["identity"]].append(s)
    # hand-over decisions: the latest per reviewer counts; reviewers that disagree are a conflict (the suggestion stays undecided)
    ho: dict[str, dict[str, tuple[str, int, list]]] = defaultdict(dict)
    for e in live:
        if e["action"] == "handover_decision":
            ho[e["suggestion"]][e["reviewer"]] = (e["decision"], e["id"], e["segments"])
    handovers = {}
    for sug, by in ho.items():
        decs = {d for d, _, _ in by.values()}
        segs_ = next(iter(by.values()))[2]
        if len(decs) > 1:
            conflicts.append({"type": "contradictory_handover", "suggestion": sug, "labels": sorted(decs), "segments": segs_})
        handovers[sug] = {"decision": decs.pop() if len(decs) == 1 else "conflict", "segments": segs_, "reviewers": sorted(by),
                          "events": sorted(i for _, i, _ in by.values())}
    return {"schema": REVIEW_SCHEMA, "segments": {str(s): v for s, v in sorted(state.items())}, "conflicts": conflicts,
            "rejected_associations": sorted(rejected), "splits": {str(k): sorted(v) for k, v in splits.items()},
            "merges": dict(alias), "identities": {k: sorted(v) for k, v in sorted(identities.items())},
            "handover_decisions": handovers,
            "events_total": len(evs), "events_live": len(live), "events_undone": len(undone),
            "note": "human review annotations; not model-validation evidence and never written into the predictions"}
