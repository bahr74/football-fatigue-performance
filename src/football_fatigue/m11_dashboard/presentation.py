"""Display-only logic for the Step 8.1 advisor views: decision explanations, demo examples, feature categories, candidate charts.

Nothing here changes a prediction. Every function reads either the public context or the output of
:func:`inference.infer_instance` (the unchanged Step 6 decision records and pair table); none takes a provider reference, so
nothing here can leak it before "Reveal Provider Reference".

How the frozen rule decides (``ml.assign`` / ``ml.decide``), which :func:`explain_decision` follows step by step:

1. **Reachability.** An incoming tracklet with no physically reachable outgoing tracklet is UNMATCHED.
2. **Global assignment** (threshold-free, the same at both operating points). The one-to-one assignment maximising the summed
   log-odds of linked pairs is chosen jointly for the whole encounter. Leaving a tracklet unlinked costs 0, so a pair with
   p < 0.5 is never linked, and an outgoing tracklet can be given to only one incoming tracklet. A tracklet the assignment
   leaves unlinked is UNCERTAIN; its decision record then names its highest-probability reachable candidate, with no margin.
3. **Operating point.** An assigned link is MATCHED only if its probability clears the threshold (relaxed 0.76, conservative
   0.87) and its margin (how much worse, in log-odds, the best assignment without this link is) is at least 1.0; otherwise it
   is UNCERTAIN.
"""

from __future__ import annotations

from typing import Any

from football_fatigue.matching import ml

# ---------------------------------------------------------------- tracklet colours (one per tracklet, same in every view)

# incoming: cool hues with circles; outgoing: warm hues with diamonds. in-k and out-k never share a colour, because the
# tracklet order is random and a shared colour would suggest a pairing the model has not made.
IN_PALETTE = ("#29b6f6", "#ce93d8", "#4dd0e1", "#9fa8da", "#80cbc4", "#b39ddb")
OUT_PALETTE = ("#ffa726", "#f06292", "#fff176", "#ff8a65", "#ffcc80", "#f48fb1")
STATUS_BADGE = dict(MATCHED=":green-badge[MATCHED]", UNCERTAIN=":orange-badge[UNCERTAIN]", UNMATCHED=":gray-badge[UNMATCHED]")


def tracklet_colors(inst: dict) -> dict[str, str]:
    return {**{a["tracklet_id"]: IN_PALETTE[n % len(IN_PALETTE)] for n, a in enumerate(inst["incoming"])},
            **{b["tracklet_id"]: OUT_PALETTE[n % len(OUT_PALETTE)] for n, b in enumerate(inst["outgoing"])}}


# ---------------------------------------------------------------- demo examples (public metadata only)

DEMO_RULES = (
    ("A", "Two players, clear decision", "first high-confidence encounter with 2 incoming tracklets and a withheld gap over 1 s",
     lambda c: c["tier"] == ml.LABEL_TIER and c["n_incoming"] == 2 and c["ambiguity_s"] > 1.0),
    ("B", "Three or more players", "first high-confidence encounter with 3 or more incoming tracklets",
     lambda c: c["tier"] == ml.LABEL_TIER and c["n_incoming"] >= 3),
    ("C", "Naturally suspicious reference", "first encounter in the natural-suspicious (unverified) tier",
     lambda c: c["tier"] == ml.UNVERIFIED_TIER),
)
DEMO_KEYS = ("tier", "n_incoming", "ambiguity_s", "period", "start_time_s", "instance_id")


def demo_examples(contexts: dict[str, dict]) -> list[dict[str, Any]]:
    """Examples A, B, C by the fixed Step 8 rules, in time order. Uses only public-context keys (:data:`DEMO_KEYS`): never model
    output, the provider mapping, or validation / test data. ``instance_id`` is None when a match has no such encounter."""
    cs = sorted(({k: c[k] for k in DEMO_KEYS} for c in contexts.values()), key=lambda c: (c["period"], c["start_time_s"]))
    return [dict(key=key, title=title, rule=rule, instance_id=next((c["instance_id"] for c in cs if test(c)), None))
            for key, title, rule, test in DEMO_RULES]


# ---------------------------------------------------------------- decision explanation (inference output only)

def _p(x: float | None) -> str:
    return "–" if x is None else f"{x:.3f}"        # 3 decimals, so 0.998 does not read as a certain 1.00


def explain_decision(d: dict, pairs: list[dict], decisions: list[dict], op: dict) -> dict[str, Any]:
    """Why the frozen rule gave this decision, from ``infer_instance`` outputs only.

    ``d``: one decision record of one operating point; ``decisions``: all records of that point; ``op``: its threshold and margin.
    Returns ``assigned`` (the global assignment linked this tracklet), ``assigned_to``, ``best`` (highest-probability reachable
    candidate), the checks (``reachable``, ``threshold_ok``, ``margin_ok``; None when not applicable), ``implied`` (the decision
    those checks imply), ``consistent`` (``implied`` equals the record) and plain-language ``reasons``."""
    i = d["incoming"]
    mine = sorted((q for q in pairs if q["incoming"] == i), key=lambda q: -q["probability"])
    reach = [q for q in mine if q["reachable"]]
    best = reach[0] if reach else None
    assigned = d["margin"] is not None                    # ml.assign gives a margin only to a tracklet the assignment linked
    prob = {q["outgoing"]: q["probability"] for q in mine}
    others = {x["candidate"]: x["incoming"] for x in decisions if x is not d and x["margin"] is not None}  # links of the assignment
    thr, mrg = op["threshold"], op["margin"]
    reasons: list[str] = []
    threshold_ok = margin_ok = None
    if not reach:
        implied = "UNMATCHED"
        reasons.append(f"No outgoing tracklet is physically reachable from where {i} was last seen within the gap, so there is "
                       "nothing to link.")
    elif not assigned:
        implied = "UNCERTAIN"
        c, pc = d["candidate"], prob.get(d["candidate"])
        if pc is not None and pc < 0.5:
            reasons.append(f"Its best reachable candidate {c} scores only p = {_p(pc)}. The global assignment never links a pair "
                           "below 0.5, because that would lower the encounter's total log-odds, so it leaves this tracklet unlinked.")
        elif c in others:
            reasons.append(f"Its best candidate {c} (p = {_p(pc)}) is linked to {others[c]} instead. Each outgoing tracklet can be "
                           f"used once, and the joint assignment of the whole encounter scores higher that way, so {i} is left unlinked.")
        else:
            reasons.append(f"The global assignment leaves {i} unlinked; its best reachable candidate is {c} (p = {_p(pc)}).")
    else:
        c, pc = d["candidate"], prob[d["candidate"]]
        if best is not None and best["outgoing"] != c:
            taken = others.get(best["outgoing"])
            reasons.append(f"The global assignment links {i} to {c} (p = {_p(pc)}), not to its highest-scoring candidate "
                           f"{best['outgoing']} (p = {_p(best['probability'])})"
                           + (f", which goes to {taken}: the joint assignment of all tracklets scores higher." if taken else "."))
        else:
            reasons.append(f"The global assignment links {i} to {c}, its highest-scoring candidate (p = {_p(pc)}).")
        threshold_ok, margin_ok = pc >= thr, d["margin"] >= mrg
        reasons.append(f"p = {_p(pc)} {'clears' if threshold_ok else 'is below'} the threshold {thr:.2f}.")
        reasons.append(f"Margin {d['margin']:.2f} log-odds: the best assignment without this link is "
                       + (f"clearly worse (≥ {mrg:.1f})." if margin_ok else
                          f"only {d['margin']:.2f} worse, below the required {mrg:.1f}, so a close alternative exists."))
        implied = "MATCHED" if threshold_ok and margin_ok else "UNCERTAIN"
    verdict = dict(MATCHED=f"→ MATCHED: the link {i} → {d['outgoing']} is committed.",
                   UNCERTAIN="→ UNCERTAIN: the model abstains rather than risk a wrong link.",
                   UNMATCHED="→ UNMATCHED: no continuation can be proposed.")[d["decision"]]
    reasons.append(verdict)
    return dict(incoming=i, decision=d["decision"], assigned=assigned, assigned_to=d["candidate"] if assigned else None,
                best=None if best is None else best["outgoing"], best_probability=None if best is None else best["probability"],
                reachable=bool(reach), threshold_ok=threshold_ok, margin_ok=margin_ok, implied=implied,
                consistent=implied == d["decision"], reasons=reasons)


def assignment_rows(result: dict) -> list[dict[str, Any]]:
    """One row per incoming tracklet: the global assignment (shared by both points) and each point's decision."""
    pts = list(result["operating_points"])
    rows = []
    for n, d0 in enumerate(result["decisions"][pts[0]]):
        ex = {pt: explain_decision(result["decisions"][pt][n], result["pairs"], result["decisions"][pt], result["operating_points"][pt])
              for pt in pts}
        e0 = ex[pts[0]]
        rows.append(dict(incoming=d0["incoming"], assigned_to=e0["assigned_to"], best=e0["best"], best_probability=e0["best_probability"],
                         probability=d0["probability"] if e0["assigned"] else None, margin=d0["margin"],
                         reachable=sum(q["reachable"] for q in result["pairs"] if q["incoming"] == d0["incoming"]),
                         decisions={pt: result["decisions"][pt][n]["decision"] for pt in pts}, explanations=ex))
    return rows


# ---------------------------------------------------------------- candidate chart (plotly figure dict)

def candidate_figure(incoming: str, pairs: list[dict], assigned_to: str | None, ops: dict[str, dict], colors: dict[str, str]) -> dict:
    """Horizontal bars of one incoming tracklet's calibrated candidate probabilities, with the 0.5 assignment floor and both
    operating thresholds. The assigned candidate is outlined; unreachable candidates are grey."""
    mine = [q for q in pairs if q["incoming"] == incoming]
    ys = [q["outgoing"] for q in mine]
    text = [f"{q['probability']:.2f}" + ("  ★ assigned" if q["outgoing"] == assigned_to else "") + ("" if q["reachable"] else "  ✕ unreachable")
            for q in mine]
    bars = dict(type="bar", orientation="h", x=[q["probability"] for q in mine], y=ys, text=text, textposition="outside",
                marker=dict(color=[colors.get(q["outgoing"], "#bbbbbb") if q["reachable"] else "#5f6368" for q in mine],
                            line=dict(color=["#ffffff" if q["outgoing"] == assigned_to else "rgba(0,0,0,0)" for q in mine], width=3)),
                hovertext=[f"{incoming} → {q['outgoing']}<br>calibrated p = {q['probability']:.3f}<br>"
                           + ("reachable" if q["reachable"] else "unreachable (too far for the gap)") for q in mine],
                hoverinfo="text", showlegend=False, cliponaxis=False)
    lines = [(0.5, "assignment floor 0.5", "dot", "#9e9e9e")] + [(op["threshold"], f"{pt} {op['threshold']:.2f}", "dash", c)
                                                                for (pt, op), c in zip(ops.items(), ("#66bb6a", "#26a69a"))]
    shapes = [dict(type="line", x0=x, x1=x, yref="paper", y0=0, y1=1, line=dict(color=c, dash=dash, width=1.5)) for x, _, dash, c in lines]
    ann = [dict(x=x, y=1.0, yref="paper", yanchor="bottom", yshift=13 * k, text=t, showarrow=False, font=dict(size=10, color=c))
           for k, (x, t, _, c) in enumerate(lines)]
    return dict(data=[bars], layout=dict(height=110 + 34 * len(mine), margin=dict(l=10, r=10, t=56, b=10), shapes=shapes, annotations=ann,
                                          xaxis=dict(range=[0, 1.5], title=None, tickvals=[0, 0.25, 0.5, 0.75, 1.0]),
                                          yaxis=dict(autorange="reversed", title=None), bargap=0.35,
                                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"))


# ---------------------------------------------------------------- feature categories (display only)

CATEGORIES: dict[str, dict[str, Any]] = {
    "Position and distance": dict(
        about="Where the incoming tracklet ends and the outgoing one starts, how far apart, and how close other tracklets are.",
        features=("dist_m", "dist_per_s", "reach_ratio", "in_sep_m", "out_sep_m")),
    "Speed and acceleration": dict(
        about="Speeds and accelerations at both ends, and the speed or acceleration a single player would need to join them.",
        features=("speed_in", "speed_out", "speed_diff", "vel_diff", "speed_consistency", "hermite_peak_speed", "acc_in", "acc_out",
                  "req_acc", "hermite_max_acc", "hermite_rms_acc")),
    "Direction and turning": dict(
        about="Running directions at both ends, the turn needed to join them, and how curved each end is.",
        features=("heading_diff_deg", "both_moving", "turn_in_deg", "turn_out_deg", "turn_acc_demand", "turn_rate_in", "turn_rate_out",
                  "straight_in", "straight_out")),
    "Gap and trajectory continuity": dict(
        about="Length of the withheld gap, how far simple motion extrapolations miss across it, and how many endpoint samples "
              "were observed or provider-ESTIMATED (observation quality).",
        features=("gap_s", "cv_fwd_err", "cv_bwd_err", "cv_mid_err", "ca_fwd_err", "ca_bwd_err", "ca_mid_err", "n_upd_in", "n_upd_out",
                  "est_share_in", "est_share_out")),
    "Assignment ambiguity": dict(
        about="How many tracklets compete, how much better the next alternative fits, and how much forcing this pair would cost "
              "the rest of the encounter.",
        features=("n_incoming", "n_outgoing", "row_gap_dist_m", "col_gap_dist_m", "row_gap_cv_fwd_err", "col_gap_cv_fwd_err", "row_gap_cv_mid_err",
                  "col_gap_cv_mid_err", "row_gap_hermite_rms_acc", "col_gap_hermite_rms_acc", "in_closing_m_s", "n_reach_row", "n_reach_col",
                  "assign_regret_cv", "assign_regret_hermite")),
}


def feature_category(name: str) -> str:
    return next((c for c, v in CATEGORIES.items() if name in v["features"]), "Other")
