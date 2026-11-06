"""Encounter-level evaluation against the provider identity reference (M11 step 2). Evaluation-only: reads references.

Per incoming tracklet, with ``expected`` being the reference outgoing tracklet (``None`` when the player has no outgoing
tracklet), the outcomes are:

- ``correct_match``: MATCHED to ``expected``;
- ``wrong_match``: MATCHED to anything else, including any match when ``expected`` is None;
- ``correct_unmatched``: UNMATCHED when ``expected`` is None (genuinely absent);
- ``missed_unmatched``: UNMATCHED although ``expected`` exists;
- ``uncertain``: UNCERTAIN (an abstention, never counted as right or wrong).

The rates are shares of incoming tracklets:
- correct match rate;
- wrong match rate;
- unmatched rate;
- uncertain rate;
- abstention rate (unmatched + uncertain);
- coverage (the MATCHED share).

The false-association rate is wrong matches divided by MATCHED decisions. Complete assignment accuracy is the share of instances
where every incoming outcome is correct_match or correct_unmatched. A conflict is an instance in which two incoming tracklets
are MATCHED to one outgoing tracklet.

All of these are agreement with the provider reference, not verified identity accuracy. Precision, recall, F1, IDF1 and HOTA are
not reported: the benchmark has no independent identity annotation to support them.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .baselines import BASELINES

OUTCOMES = ("correct_match", "wrong_match", "correct_unmatched", "missed_unmatched", "uncertain")


def outcome(decision: dict, expected: str | None) -> str:
    if decision["decision"] == "MATCHED":
        return "correct_match" if expected is not None and decision["outgoing"] == expected else "wrong_match"
    if decision["decision"] == "UNMATCHED":
        return "correct_unmatched" if expected is None else "missed_unmatched"
    if decision["decision"] == "UNCERTAIN":
        return "uncertain"
    raise ValueError(f"unknown decision {decision['decision']}")


def _rate(a: int, b: int) -> float | None:
    return round(a / b, 4) if b else None


def metrics(per_instance: list[tuple[list[str], bool]]) -> dict[str, Any]:
    """``per_instance``: (outcomes of its incoming tracklets, has a conflict)."""
    k = Counter(o for outs, _ in per_instance for o in outs)
    n = sum(k.values())
    matched = k["correct_match"] + k["wrong_match"]
    complete = sum(all(o in ("correct_match", "correct_unmatched") for o in outs) for outs, _ in per_instance)
    return dict(
        instances=len(per_instance), incoming=n, counts={o: k[o] for o in OUTCOMES},
        correct_match_rate=_rate(k["correct_match"], n), wrong_match_rate=_rate(k["wrong_match"], n),
        unmatched_rate=_rate(k["correct_unmatched"] + k["missed_unmatched"], n), uncertain_rate=_rate(k["uncertain"], n),
        abstention_rate=_rate(k["correct_unmatched"] + k["missed_unmatched"] + k["uncertain"], n), coverage=_rate(matched, n),
        false_association_rate=_rate(k["wrong_match"], matched), complete_assignment_accuracy=_rate(complete, len(per_instance)),
        instances_with_wrong_match=sum("wrong_match" in outs for outs, _ in per_instance),
        conflicts=sum(c for _, c in per_instance))


GAP_BINS_S = (1.5, 2.5, 4.0)        # fixed a priori: <1.5, 1.5-2.5, 2.5-4, >=4 s of ambiguity (withheld in task B)


def gap_bin(s: float) -> str:
    edges = (0.0, *GAP_BINS_S)
    for lo, hi in zip(edges, edges[1:]):
        if s < hi:
            return f"{lo:g}-{hi:g}s"
    return f">={GAP_BINS_S[-1]:g}s"


def _strata(ref: dict) -> dict[str, str]:
    n = len(ref["incoming"])
    return dict(tier=ref["tier"], incoming=str(n) if n < 3 else "3+", difficulty=ref["difficulty"]["label"], team_relation=ref["team_relation"],
                gap=gap_bin(ref["difficulty"]["ambiguity_s"]), sharp_turn="yes" if "sharp_turn" in ref["difficulty"]["reasons"] else "no",
                absent_tracklet="yes" if ref["without_incoming"] or ref["without_outgoing"] else "no",
                crossing="yes" if ref["difficulty"]["crossing"] else "no",
                contains_step1_trusted="yes" if ref["step1_categories"].get("trusted_reference_candidate") else "no")


def evaluate(predictions: list[dict], references: list[dict], methods: tuple[str, ...] = BASELINES) -> dict[str, Any]:
    refs = {r["instance_id"]: r for r in references}
    out: dict[str, Any] = {}
    for name in methods:
        if not any(p["baseline"] == name for p in predictions):
            continue
        rows = []
        for p in (p for p in predictions if p["baseline"] == name):
            ref = refs[p["instance_id"]]
            outs = [outcome(d, ref["expected"][d["incoming"]]) for d in p["decisions"]]
            if sorted(d["incoming"] for d in p["decisions"]) != sorted(ref["expected"]):
                raise ValueError(f"{name} did not decide every incoming tracklet of {p['instance_id']}")
            claimed = Counter(d["outgoing"] for d in p["decisions"] if d["decision"] == "MATCHED")
            rows.append((ref, outs, any(v > 1 for v in claimed.values())))
        res: dict[str, Any] = dict(all=metrics([(o, c) for _, o, c in rows]))
        for key in ("tier", "incoming", "difficulty", "team_relation", "gap", "sharp_turn", "absent_tracklet", "crossing", "contains_step1_trusted"):
            vals = sorted({_strata(r)[key] for r, _, _ in rows})
            res[f"by_{key}"] = {v: metrics([(o, c) for r, o, c in rows if _strata(r)[key] == v]) for v in vals}
        out[name] = res
    return out
