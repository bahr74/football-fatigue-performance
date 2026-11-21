"""Streamlit app: M11 identity recovery on real FIFA World Cup 2022 RAW tracking (development matches, frozen Step 6 model).

Run with ``python -m football_fatigue.m11_dashboard`` (or ``streamlit run`` this file ``-- --root data/raw``).

Reference hiding: ``MD.provider_reference`` and ``D.hidden_interval`` are called only inside ``if revealed`` branches. Every view
before that is built from the matcher-visible instance, the public context and the frozen model's output.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from football_fatigue.m11_dashboard import data as D
from football_fatigue.m11_dashboard import inference as I
from football_fatigue.m11_dashboard import pitch as P
from football_fatigue.m11_dashboard import presentation as PR
from football_fatigue.m11_dashboard import research as R
from football_fatigue.matching.evaluation import GAP_BINS_S, gap_bin, outcome

DISCLAIMER = ("Agreement is measured against the data provider's identity labels. Those labels are a reference, not independently "
              "verified ground truth, so agreement is **not** verified identity accuracy.")
IN_SAMPLE = ("**Development data, in-sample.** The frozen model was fitted on these 10 development matches. Examples here explain "
             "*how* the model behaves; they are not independent evidence that it generalises. General performance claims rest on "
             "the saved **locked-test** results (Research Results tab).")
PROB_NOTE = ("p is the frozen model's Platt-calibrated score that two tracklets are the same player *according to the provider's "
             "labels*. It is not a verified physical identity confidence.")
GLOSSARY = {
    "Tracking": "The data provider (PFF) records every player's position on the pitch about 30 times per second, in metres. Each "
                "position is marked VISIBLE (seen by the cameras) or ESTIMATED (filled in by the provider).",
    "Encounter": "Two or more players come within 2 m of each other. It is a proximity event; it is not necessarily a visual "
                 "occlusion, and nobody checked video.",
    "Tracklet": "A short anonymous piece of trajectory: up to 2 s *incoming* (before the encounter) or *outgoing* (after it). "
                "Tracklets are named `in-0`, `in-1`, … and `out-0`, `out-1`, … in random order, so `in-0` and `out-0` are not "
                "necessarily the same player.",
    "Identity masking": "Names, shirt numbers and teams are removed, and every observation of the players during the encounter is "
                        "withheld (task B). The model sees only the incoming and outgoing tracklets and must say which belong together. "
                        "This is a controlled experiment on real trajectories, not a real occlusion.",
    "Candidate pair": "Every incoming × outgoing combination. For each pair the model estimates the probability that both "
                      "tracklets belong to the same player, from 51 motion features.",
    "Global assignment": "All links of an encounter are chosen together, one-to-one, maximising the total log-odds, so two incoming "
                         "tracklets can never take the same outgoing tracklet, and a pair below p = 0.5 is never linked. A high pair "
                         "probability alone is therefore not a committed match.",
    "Abstention": "A link is committed (MATCHED) only if the assignment chose it, its probability clears the threshold, and an "
                  "alternative assignment is clearly worse (log-odds margin ≥ 1.0). Otherwise the model answers UNCERTAIN instead of "
                  "guessing. UNMATCHED means no outgoing tracklet is physically reachable.",
}
PROVENANCE = [
    ("RAW observation", "filled marker", "a position the provider marked VISIBLE"),
    ("Provider-estimated position", "hollow marker", "a position the provider marked ESTIMATED"),
    ("Incoming / outgoing tracklet", "cool-coloured circles / warm-coloured diamonds", "what the model sees; colours never imply a pairing"),
    ("Withheld interval (task B)", "brown banner and 'WITHHELD' label, no encounter-player markers", "real observations deliberately removed from the model's input"),
    ("Model prediction", "white line: solid = MATCHED, dashed = assigned but UNCERTAIN, dotted = best candidate not assigned", "output of the frozen classifier and assignment"),
    ("Provider reference", "identity colours and dotted 'withheld path', only after reveal", "the provider's labels; evaluation-only"),
]
DECISION_WORDS = dict(correct_match="agrees with provider reference", wrong_match="disagrees with provider reference",
                      uncertain="abstained (UNCERTAIN)", correct_unmatched="UNMATCHED, provider has no continuation",
                      missed_unmatched="UNMATCHED, provider has a continuation")
TIER_WORDS = {D.LABEL_TIER: "high-confidence reference", D.EXPLORATORY_TIER: "lower-confidence reference (exploratory)",
              D.UNVERIFIED_TIER: "natural suspicious, unverified"}
TIER_BADGE = {D.LABEL_TIER: ":green-badge[High-confidence reference]", D.EXPLORATORY_TIER: ":blue-badge[Lower-confidence reference]",
              D.UNVERIFIED_TIER: ":red-badge[Natural suspicious · unverified]"}
TIER_HELP = {D.LABEL_TIER: "Clean provider labels on both sides; used for training and for all headline metrics.",
             D.EXPLORATORY_TIER: "Provider labels with quality flags (for example ESTIMATED endpoints); results are exploratory.",
             D.UNVERIFIED_TIER: "The provider's own labels look inconsistent here (for example an implausible jump). Never trained "
                                "on; disagreement here is not a confirmed swap."}
SPEEDS = {"0.25×": 0.25, "0.5×": 0.5, "1× (real time)": 1.0, "2×": 2.0}
ZOOM_WORDS = {"close": "Close-up", "wide": "Wide", "full": "Full pitch"}
FILTER_KEYS = ("f_players", "f_gap", "f_tier", "f_sharp", "f_susp")
DEMO_PREFERRED = "3829"      # the documented Step 8 walkthrough match; examples inside it come from fixed public-metadata rules


def _root() -> str:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--root", default=os.environ.get("M11_DASHBOARD_ROOT", "data/raw"))
    a, _ = ap.parse_known_args(sys.argv[1:])
    return str(Path(a.root).resolve())


ROOT = _root()
META: dict | None = None


# ---------------------------------------------------------------- cached loaders

@st.cache_resource(show_spinner="Loading and verifying the frozen model…")
def frozen(root: str) -> I.FrozenModel:
    return I.load_frozen_model(root)


@st.cache_data(show_spinner=False)
def dev_matches(root: str) -> list[str]:
    return D.development_matches(root)


@st.cache_resource(max_entries=3, show_spinner="Loading task B instances…")
def match(root: str, mid: str) -> D.MatchData:
    return D.load_match(root, mid)


@st.cache_resource(max_entries=1, show_spinner="Loading RAW tracking cache…")
def cache(root: str, mid: str) -> dict:
    return D.load_cache(root, mid)


@st.cache_data(show_spinner=False)
def metadata(root: str, mid: str) -> dict | None:
    m = D.match_metadata(root, mid)
    if m is None:
        return None
    return dict(length=m.pitch_length_m, width=m.pitch_width_m, home=m.home_team_name, away=m.away_team_name,
                starts={k: getattr(m, f"start_period_{k}_s", None) for k in (1, 2)}, competition=m.competition_name)


@st.cache_data(show_spinner=False)
def research(root: str) -> dict:
    return R.research_results(root)


def clock(seconds: float | None) -> str:
    if seconds is None:
        return "–"
    m, s = divmod(max(seconds, 0.0), 60)
    return f"{int(m):02d}:{s:04.1f}"


def since(c: dict, meta: dict | None = None) -> float | None:
    """Seconds since the start of the encounter's period (metadata and tracking share the provider video clock)."""
    starts = (meta if meta is not None else META or {}).get("starts", {})
    s = starts.get(c["period"])
    return None if s is None else c["start_time_s"] - s


def pct(v: float | None, n: int = 1) -> str:
    return "–" if v is None else f"{100 * v:.{n}f}%"


def ci(v) -> str:
    return "" if not v else f" [{100 * v[0]:.1f}, {100 * v[1]:.1f}]"


def auc_ci(v) -> str:
    return "" if not v else f"95% CI {v[0]:.3f}–{v[1]:.3f}"


def plot(fig, key: str) -> None:
    st.plotly_chart(go.Figure(fig) if isinstance(fig, dict) else fig, key=key, config=dict(displayModeBar=False))


# ---------------------------------------------------------------- page

st.set_page_config(page_title="M11 Identity Recovery", page_icon="⚽", layout="wide")
st.title("Player identity recovery after close encounters")
st.caption("FIFA World Cup 2022 RAW tracking (PFF) · frozen M11 Step 6 Random Forest · development matches only")

try:
    FM = frozen(ROOT)
    MATCHES = dev_matches(ROOT)
except D.DashboardError as e:
    st.error(f"Required artifacts are missing or invalid, so nothing is shown.\n\n{e}")
    st.stop()
if not MATCHES:
    st.error("No development match with Step 2-3 benchmark outputs was found.")
    st.stop()
DEMO_MATCH = DEMO_PREFERRED if DEMO_PREFERRED in MATCHES else MATCHES[0]


def load_example(m: str, iid: str) -> None:
    """Advisor Demo: select one example encounter and clear the filters so it is in the list."""
    st.session_state["sel_match"] = m
    for k in FILTER_KEYS + (f"f_period_{m}",):
        st.session_state.pop(k, None)
    st.session_state[f"enc_{m}"] = iid


# ---------------------------------------------------------------- sidebar: selection and filters

with st.sidebar:
    st.header("Select an encounter")
    mid = st.selectbox("Match (development split)", MATCHES, format_func=lambda m: f"Match {m}", key="sel_match")
    try:
        MD = match(ROOT, mid)
    except D.DashboardError as e:
        st.error(str(e))
        st.stop()
    META = meta = metadata(ROOT, mid)
    if meta:
        st.caption(f"{meta['competition']}")
    ctxs = list(MD.contexts.values())
    periods = sorted({c["period"] for c in ctxs})
    period = st.selectbox("Period", ["All"] + periods, key=f"f_period_{mid}")
    st.subheader("Filters")
    players = st.radio("Incoming tracklets", ["All", "2 players", "3 or more"], horizontal=True, key="f_players")
    gaps = [gap_bin(0.0)] + [gap_bin(e) for e in GAP_BINS_S]
    gap_sel = st.pills("Withheld gap", gaps, selection_mode="multi", default=gaps, key="f_gap",
                       help="Duration of the withheld interval, in the fixed Step 2 bins (short: <1.5 s; long: ≥2.5 s).")
    tiers = list(TIER_WORDS)
    tier_sel = st.multiselect("Reference tier", tiers, default=tiers, format_func=TIER_WORDS.get, key="f_tier")
    sharp = st.checkbox("Sharp turn only", key="f_sharp", help="Reference-derived: computed by Step 2 from the provider's mapping "
                        "(heading change > 45° of a matched player). Used for selection only; never shown before reveal and never "
                        "given to the model.")
    susp = st.checkbox("Suspicious discontinuity only", key="f_susp", help="Natural-suspicious tier or a Step 1 identity-discontinuity candidate.")
    sel = [c for c in ctxs if (period == "All" or c["period"] == period)
           and (players == "All" or (c["n_incoming"] == 2 if players == "2 players" else c["n_incoming"] >= 3))
           and c["gap_bin"] in (gap_sel or []) and c["tier"] in (tier_sel or []) and (not sharp or c["filter_sharp_turn"]) and (not susp or c["suspicious"])]
    sel.sort(key=lambda c: (c["period"], c["start_time_s"]))
    st.caption(f"{len(sel)} of {len(ctxs)} encounters match the filters.")
    if not sel:
        st.warning("No encounter matches these filters.")
        st.stop()

    def enc_label(c: dict) -> str:
        return (f"P{c['period']} {clock(since(c))} · {c['n_incoming']} in / {c['n_outgoing']} out"
                f" · gap {c['ambiguity_s']:.1f} s · {c['instance_id'][:6]}")

    enc_key = f"enc_{mid}"
    options = [c["instance_id"] for c in sel]
    if st.session_state.get(enc_key) not in options:
        st.session_state.pop(enc_key, None)
    iid = st.selectbox("Encounter", options, format_func=lambda i: enc_label(MD.contexts[i]), key=enc_key)
    st.divider()
    st.caption("The frozen model was fitted on these 10 development matches, so results here are in-sample. Unbiased performance "
               "is in **Research Results** (locked test).")

CTX, INST = MD.contexts[iid], MD.instances[iid]
REVEAL_KEY, RESULT_KEY, FRAME_KEY = f"reveal_{mid}_{iid}", f"result_{mid}_{iid}", f"frame_{mid}_{iid}"
st.session_state.setdefault(REVEAL_KEY, False)
revealed = st.session_state[REVEAL_KEY]
result = st.session_state.get(RESULT_KEY)
TRAIN = D.training_pairs(INST, CTX["tier"])
COLORS = PR.tracklet_colors(INST)
LENGTH, WIDTH_M = (meta["length"], meta["width"]) if meta else (105.0, 68.0)
try:
    CACHE = cache(ROOT, mid)
    WINDOW = D.raw_window(CACHE, CTX, MD.participant_slots(iid))      # participants excluded; their slots are never displayed
    CACHE_NOTE = None
except D.DashboardError as e:
    CACHE, WINDOW, CACHE_NOTE = None, None, str(e)
DEMO_EXAMPLES = PR.demo_examples(match(ROOT, DEMO_MATCH).contexts)
EXAMPLE_OF = {e["instance_id"]: e["key"] for e in DEMO_EXAMPLES if e["instance_id"]} if mid == DEMO_MATCH else {}


def run_recovery(inst: dict, key: str) -> None:
    """Button callback, so every tab of the same rerun sees the new result."""
    st.session_state.pop(f"error_{key}", None)
    try:
        res = I.infer_instance(inst, FM)
        problems = I.check_assignment(res)
        if problems:
            raise D.DashboardError("assignment invariants failed: " + "; ".join(problems))
        st.session_state[key] = res
    except Exception as e:      # never fall back to cached or invented scores
        st.session_state.pop(key, None)
        st.session_state[f"error_{key}"] = str(e)


def set_reveal(key: str, value: bool) -> None:
    st.session_state[key] = value


def run_button(where: str) -> None:
    st.button("Run Identity Recovery", type="primary", on_click=run_recovery, args=(INST, RESULT_KEY), key=f"run_{where}",
              help="Runs the checksum-verified frozen model on the anonymous tracklets only.")
    if err := st.session_state.get(f"error_{RESULT_KEY}"):
        st.error(f"Exact frozen inference could not be reproduced for this encounter, so no prediction is shown.\n\n{err}")


def reveal_controls(where: str) -> None:
    if st.session_state[REVEAL_KEY]:
        st.button("Hide provider reference", key=f"hide_{where}", on_click=set_reveal, args=(REVEAL_KEY, False))
    else:
        st.button("Reveal Provider Reference", key=f"reveal_{where}", on_click=set_reveal, args=(REVEAL_KEY, True), type="secondary",
                  help="Shows the provider's identity labels for this encounter. They are evaluation-only and never reach the model.")


# ---------------------------------------------------------------- shared views

def encounter_header() -> None:
    """Short, untruncated summary metrics of the selected encounter (public context only)."""
    c = st.columns(5)
    c[0].metric("Period", f"P{CTX['period']}")
    c[1].metric("Clock", clock(since(CTX)), help="Time since the start of the period (provider video clock).")
    c[2].metric("Tracklets", f"{CTX['n_incoming']} in · {CTX['n_outgoing']} out")
    c[3].metric("Withheld gap", f"{CTX['ambiguity_s']:.2f} s", help=f"Step 2 bin {CTX['gap_bin']}")
    c[4].metric("Closest", f"{CTX['min_distance_m']:.2f} m", help="Minimum distance between two participants (Step 1); symmetric, "
                "carries no identity mapping.")
    tag = f"  ·  :violet-badge[Demo example {EXAMPLE_OF[iid]}]" if iid in EXAMPLE_OF else ""
    st.markdown(f"{TIER_BADGE[CTX['tier']]}{tag}  ·  match {mid} (development)  \n"
                f"<small>{TIER_HELP[CTX['tier']]}</small>", unsafe_allow_html=True)


def board(where: str, *, start: int, zoom: str, animate: bool, decisions=None, height: int = 600, speed: float = 1.0) -> None:
    rv = None
    if revealed:
        ref = MD.provider_reference(iid)
        colors = {**{t: D.identity_label(v) for t, v in ref["incoming"].items()}, **{t: D.identity_label(v) for t, v in ref["outgoing"].items()}}
        rv = dict(colors=colors, hidden=D.hidden_interval(CACHE, CTX, ref) if CACHE is not None else [])
    probs = {(q["incoming"], q["outgoing"]): q["probability"] for q in result["pairs"]} if result else None
    fig = P.replay_figure(INST, WINDOW, length=LENGTH, width=WIDTH_M, step=2, speed=speed, start=start, zoom=zoom, reveal=rv,
                          decisions=decisions, probabilities=probs, animate=animate, height=height)
    plot(fig, f"board_{where}")


def render_assignment(where: str) -> None:
    """The frozen model's answer: global assignment, both operating points, and why, per incoming tracklet."""
    ops = result["operating_points"]
    rows = PR.assignment_rows(result)
    st.markdown("**Global assignment and decisions** (the assignment is threshold-free, so it is the same at both points; only the "
                "commit rule differs)")
    st.dataframe(pd.DataFrame([{
        "Incoming": r["incoming"], "Assigned to": r["assigned_to"] or "– (left unlinked)",
        "p of assigned link": None if r["probability"] is None else round(r["probability"], 3),
        "Margin (log-odds)": None if r["margin"] is None else round(r["margin"], 2),
        "Highest-p candidate": f"{r['best']} ({r['best_probability']:.2f})" if r["best"] else "none reachable",
        f"Relaxed (p ≥ {ops['relaxed']['threshold']:.2f})": r["decisions"]["relaxed"],
        f"Conservative (p ≥ {ops['conservative']['threshold']:.2f})": r["decisions"]["conservative"],
        "In-sample pair": "yes" if r["assigned_to"] and (r["incoming"], r["assigned_to"]) in TRAIN else "no"} for r in rows]), hide_index=True)
    st.caption(PROB_NOTE)
    for r in rows:
        with st.container(border=True):
            left, right = st.columns([2, 3], gap="large")
            with left:
                st.markdown(f"<span style='color:{COLORS[r['incoming']]};font-weight:700;font-size:1.1em'>● {r['incoming']}</span> — "
                            f"{r['reachable']} reachable candidate(s)", unsafe_allow_html=True)
                plot(PR.candidate_figure(r["incoming"], result["pairs"], r["assigned_to"], ops, COLORS), f"cand_{where}_{r['incoming']}")
            with right:
                for pt in I.POINTS:
                    ex = r["explanations"][pt]
                    st.markdown(f"**{pt.capitalize()}** (p ≥ {ops[pt]['threshold']:.2f}, margin ≥ {ops[pt]['margin']:.1f}): "
                                f"{PR.STATUS_BADGE[ex['decision']]}")
                    st.markdown("\n".join(f"- {s}" for s in ex["reasons"]))


def reference_rows() -> tuple[list[dict], dict]:
    """After reveal only: per incoming tracklet, the provider continuation and each point's outcome."""
    ref = MD.provider_reference(iid)
    rows, tally = [], {pt: dict(agree=0, disagree=0, abstain=0, unmatched=0, best_is_ref=0) for pt in I.POINTS}
    for a_ in INST["incoming"]:
        i = a_["tracklet_id"]
        exp = ref["expected"].get(i)
        row = {"Incoming": i, "Provider identity": D.identity_label(ref["incoming"][i]), "Provider continuation": exp or "none (left the area)"}
        if result is not None:
            for pt in I.POINTS:
                d = next(x for x in result["decisions"][pt] if x["incoming"] == i)
                o = outcome(d, exp)
                row[pt.capitalize()] = f"{d['decision']}{' → ' + d['outgoing'] if d['outgoing'] else ''}: {DECISION_WORDS[o]}"
                t = tally[pt]
                t[{"correct_match": "agree", "wrong_match": "disagree", "uncertain": "abstain"}.get(o, "unmatched")] += 1
                if o == "uncertain" and exp is not None and d["candidate"] == exp:
                    t["best_is_ref"] += 1
        rows.append(row)
    return rows, tally


def render_reference(where: str) -> None:
    reveal_controls(where)
    if not revealed:
        st.info("The provider's identity labels are hidden. Reveal them only after you have looked at the model's answer.")
        return
    ref = MD.provider_reference(iid)
    st.warning(DISCLAIMER)
    rows, tally = reference_rows()
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    if result is not None:
        for pt, t in tally.items():
            committed = t["agree"] + t["disagree"]
            s = f"**{pt.capitalize()}**: {committed} committed link(s): {t['agree']} agree, {t['disagree']} disagree with the provider reference"
            if t["abstain"]:
                s += (f"; {t['abstain']} abstention(s), in {t['best_is_ref']} of which the proposed candidate was the provider's "
                      "continuation (abstaining there cost coverage, not accuracy)")
            st.markdown(s + ".")
    else:
        st.info("Run Identity Recovery to compare the model with this reference.")
    diff = ref["difficulty"]
    notes = [f"Reference tier: **{TIER_WORDS[ref['tier']]}**."]
    if ref["tier"] == D.EXPLORATORY_TIER:
        notes.append("Lower-confidence reference: " + (", ".join(ref["below_high_reasons"]) or "see flags") + "; results here are exploratory.")
    if ref["tier"] == D.UNVERIFIED_TIER:
        notes.append("Natural suspicious encounter: the provider's own labels may be inconsistent here, and disagreement does not "
                     "establish which side is right. This is not a confirmed identity swap.")
    if ref["discontinuity_evidence"]:
        notes.append("Step 1 flagged identity-discontinuity evidence nearby: " + ", ".join(ref["discontinuity_evidence"]) + ".")
    notes.append(f"Reference-derived motion: crossing = {'yes' if diff['crossing'] else 'no'}, largest heading change = "
                 f"{diff['max_heading_change_deg'] if diff['max_heading_change_deg'] is not None else '–'}°, team relation = "
                 f"{ref['team_relation'].replace('_', ' ')}.")
    if TRAIN:
        notes.append("In-sample: the frozen model was fitted on this encounter's provider labels, so agreement here is optimistic.")
    notes.append("One encounter is not evidence of general performance; see the locked-test results in Research Results.")
    st.markdown("\n".join(f"- {n}" for n in notes))


tab_demo, tab_explore, tab_replay, tab_recover, tab_explain, tab_research = st.tabs(
    ["★ Advisor Demo", "1 · Match Explorer", "2 · Tactical Replay", "3 · Identity Recovery", "4 · Model Explanation", "5 · Research Results"])

# ---------------------------------------------------------------- 0. advisor demo

with tab_demo:
    st.markdown("A guided walk through one encounter: **Select → Watch → Run model → Associations → Reveal reference → Understand**.")
    st.info(IN_SAMPLE)
    with st.container(border=True):
        st.markdown("#### 1 · Select an encounter")
        st.caption(f"Three development examples from match {DEMO_MATCH}, picked by fixed rules on public metadata in time order, never "
                   "by model output, validation or test results. Any other encounter can be chosen in the sidebar.")
        meta_demo = metadata(ROOT, DEMO_MATCH) or {}
        cols = st.columns(3)
        for col, ex in zip(cols, DEMO_EXAMPLES):
            with col:
                st.markdown(f"**{ex['key']} · {ex['title']}**  \n<small>Rule: {ex['rule']}.</small>", unsafe_allow_html=True)
                if ex["instance_id"] is None:
                    st.caption("No such encounter in this match.")
                    continue
                c = match(ROOT, DEMO_MATCH).contexts[ex["instance_id"]]
                st.caption(f"P{c['period']} {clock(since(c, meta_demo))} · {c['n_incoming']} in / {c['n_outgoing']} out · "
                           f"gap {c['ambiguity_s']:.2f} s")
                st.button(f"Load example {ex['key']}", key=f"ex_{ex['key']}", on_click=load_example, args=(DEMO_MATCH, ex["instance_id"]),
                          width="stretch", type="primary" if ex["instance_id"] == iid else "secondary")
        st.markdown("**Selected encounter**")
        encounter_header()
    with st.container(border=True):
        st.markdown("#### 2 · Watch: before → withheld → after")
        if CACHE_NOTE:
            st.warning(f"RAW cache unavailable, so only the matcher-visible tracklets are drawn.\n\n{CACHE_NOTE}")
        sb = P.storyboard_times(INST, WINDOW, 2)
        cols = st.columns(3)
        for col, (ph, k), cap in zip(cols, sb.items(), (
                "Incoming tracklets approach. The model sees these.",
                "The encounter players' observations are withheld; other players keep moving.",
                "Anonymous outgoing tracklets appear. Which incoming one continues as which?")):
            with col:
                board(f"demo_{ph}", start=k, zoom="close", animate=False, height=330)
                st.caption(cap)
        st.caption("Open **Tactical Replay** for the animated version, legend and zoom.")
    with st.container(border=True):
        st.markdown("#### 3 · Run the frozen model")
        run_button("demo")
        if result is None:
            st.caption("The model receives only the anonymous incoming and outgoing tracklets: no names, numbers, teams or withheld positions.")
        else:
            st.success("Frozen inference ran on the matcher-visible tracklets only (no reference, no fitting).")
    with st.container(border=True):
        st.markdown("#### 4 · View the associations")
        if result is None:
            st.caption("Run the model first.")
        else:
            render_assignment("demo")
    with st.container(border=True):
        st.markdown("#### 5 · Reveal the provider reference")
        st.caption("Only now are the provider's identity labels loaded into the page.")
        reveal_controls("demo")
        if revealed:
            board("demo_reveal", start=sb["after"], zoom="close", animate=False, height=380,
                  decisions=result["decisions"]["relaxed"] if result else None)
            st.caption("Identity colours and dotted withheld paths are the provider's reference, never given to the model. "
                       "White lines are the model's relaxed-point answer.")
    with st.container(border=True):
        st.markdown("#### 6 · Understand the result")
        if not revealed:
            st.caption("Reveal the reference in step 5 to compare.")
        else:
            render_reference("demo_cmp")
        st.markdown("- **Abstaining is a feature:** UNCERTAIN means the model declined to link because the assignment, the threshold "
                    "or the margin did not support a confident link.\n"
                    "- **This is one development encounter.** It shows how the model behaves; it is not evidence of generalisation.\n"
                    "- **General performance** comes only from the saved locked-test results (12 unseen matches, evaluated once) in "
                    "**Research Results**.")

# ---------------------------------------------------------------- 1. explorer

with tab_explore:
    st.subheader("Selected encounter")
    encounter_header()
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("**Reference tiers**")
        st.markdown("\n".join(f"- {TIER_BADGE[t]} {TIER_HELP[t]}" for t in TIER_WORDS))
        with st.expander("Technical metadata"):
            info = {
                "Match": f"{mid} (development split)",
                "Encounter ID (anonymous)": CTX["instance_id"],
                "Provider video time": f"{CTX['start_time_s']:.2f} – {CTX['end_time_s']:.2f} s",
                "Frame range (withheld interval)": f"{CTX['start_frame']} – {CTX['end_frame']}",
                "Players involved": CTX["n_participants"],
                "Reference tier": TIER_WORDS[CTX["tier"]],
                "Quality flags (why not high tier)": ", ".join(CTX["below_high_reasons"]) or "none",
                "Step 1 encounter categories": ", ".join(f"{k} ×{v}" for k, v in CTX["step1_categories"].items()),
                "Identity-discontinuity evidence": ", ".join(CTX["discontinuity_evidence"]) or "none",
                "Pairs the model was fitted on": f"{len(TRAIN)} of {CTX['n_incoming'] * CTX['n_outgoing']} (in-sample)",
            }
            st.dataframe(pd.DataFrame(info.items(), columns=["Field", "Value"]).astype(str), hide_index=True)
            st.caption("Crossing and turn statistics are computed from the provider mapping and appear only after the reference is revealed.")
    with right:
        st.markdown("**Key ideas**")
        for k, v in GLOSSARY.items():
            with st.expander(k):
                st.markdown(v)
        with st.expander("What the colours and markers mean"):
            st.dataframe(pd.DataFrame(PROVENANCE, columns=["Kind", "Shown as", "Meaning"]), hide_index=True)
    st.subheader(f"Encounters in match {mid} that match the filters ({len(sel)})")
    st.dataframe(pd.DataFrame([dict(period=f"P{c['period']}", clock=clock(since(c)), players=f"{c['n_incoming']} in / {c['n_outgoing']} out",
                                    gap_s=round(c["ambiguity_s"], 2), closest_m=round(c["min_distance_m"], 2), tier=TIER_WORDS[c["tier"]],
                                    suspicious="yes" if c["suspicious"] else "", example=EXAMPLE_OF.get(c["instance_id"], ""),
                                    encounter=c["instance_id"][:8]) for c in sel]), hide_index=True, height=260)

# ---------------------------------------------------------------- 2. replay

with tab_replay:
    if CACHE_NOTE:
        st.warning(f"RAW cache unavailable, so only the matcher-visible tracklets are drawn (no other players).\n\n{CACHE_NOTE}")
    if not meta:
        st.info("Match metadata unavailable: the pitch is drawn at 105 × 68 m.")
    ctl = st.columns([2, 3, 2, 3])
    speed = SPEEDS[ctl[0].selectbox("Playback speed", list(SPEEDS), index=2)]
    zoom = ctl[1].segmented_control("View", list(ZOOM_WORDS), format_func=ZOOM_WORDS.get, default="close", key="zoom") or "close"
    show_pred = ctl[2].toggle("Show model decisions", value=result is not None, disabled=result is None,
                              help="Run Identity Recovery first.")
    point = ctl[3].radio("Operating point", list(I.POINTS), horizontal=True, key="pt_replay", format_func=str.capitalize)
    times = P.frame_times(INST, WINDOW, 2)
    first_seen = int(np.searchsorted(times, min(a["t"][0] for a in INST["incoming"]) - 1e-6))   # first frame with an incoming tracklet
    st.session_state.setdefault(FRAME_KEY, min(first_seen, len(times) - 1))

    def step(d: int) -> None:
        st.session_state[FRAME_KEY] = int(np.clip(st.session_state[FRAME_KEY] + d, 0, len(times) - 1))

    def jump(phase: str) -> None:
        st.session_state[FRAME_KEY] = P.storyboard_times(INST, WINDOW, 2)[phase] if phase != "start" else min(first_seen, len(times) - 1)

    b = st.columns([1, 1, 1, 1, 1, 4])
    b[0].button("◀ Step", on_click=step, args=(-1,), width="stretch")
    b[1].button("Step ▶", on_click=step, args=(1,), width="stretch")
    b[2].button("Before", on_click=jump, args=("start",), width="stretch")
    b[3].button("Withheld", on_click=jump, args=("during",), width="stretch")
    b[4].button("After", on_click=jump, args=("after",), width="stretch")
    with b[5]:
        reveal_controls("replay")
    st.slider("Timeline (s, relative to the start of the withheld interval)", 0, len(times) - 1, key=FRAME_KEY,
              format=" ", help="Drag to scrub; Play/Pause inside the board animates from the selected frame.")
    k = st.session_state[FRAME_KEY]
    amb = INST["ambiguity_interval_s"][1]
    tl = go.Figure()
    for x0, x1, color, name in ((times[0], 0.0, P.PHASE_COLOR["before"], "Before: observed"), (0.0, amb, P.PHASE_COLOR["during"], "WITHHELD from model"),
                                (amb, times[-1], P.PHASE_COLOR["after"], "After: observed")):
        tl.add_trace(go.Bar(x=[x1 - x0], base=[x0], y=[""], orientation="h", marker_color=color, name=name, hoverinfo="name",
                            text=[f"{name} ({x1 - x0:.1f} s)"], textposition="inside", insidetextanchor="middle"))
    tl.add_vline(x=float(times[k]), line_color="white", line_width=3)
    tl.update_layout(barmode="stack", height=70, margin=dict(l=10, r=10, t=5, b=5), showlegend=False,
                     xaxis=dict(title=None, range=[times[0], times[-1]], ticksuffix=" s"), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    plot(tl, "timeline")
    board("replay", start=k, zoom=zoom, animate=True, speed=speed, decisions=result["decisions"][point] if (result and show_pred) else None)
    st.caption("Circles = incoming tracklets, diamonds = outgoing; filled = VISIBLE, hollow = provider ESTIMATED; small grey = other "
               "players (RAW); yellow dashed circle = encounter region. Colours identify tracklets only and never imply a pairing. "
               "During the withheld interval the encounter players are not drawn, because the model receives nothing from it. Being "
               "close is not proof of a visual occlusion. No video frames or ball positions are shown (none are in this data).")
    if revealed:
        st.warning("**Provider reference revealed.** Tracklet colours now show the provider's identity labels, and the dotted "
                   "paths are the provider-labelled positions inside the withheld interval. They are reference-only and were never "
                   "given to the model.")

# ---------------------------------------------------------------- 3. recovery

with tab_recover:
    st.markdown("The model scores every **candidate pair**, links tracklets with a one-to-one **global assignment**, and commits a "
                "link only if it clears the operating point; otherwise it **abstains**. A high pair probability is therefore not "
                "automatically a committed match.")
    ops = FM.operating_points
    cols = st.columns(3)
    cols[0].metric("Relaxed point", f"p ≥ {ops['relaxed']['threshold']:.2f}", f"margin ≥ {ops['relaxed']['margin']:.1f} log-odds", delta_color="off", delta_arrow="off")
    cols[1].metric("Conservative point", f"p ≥ {ops['conservative']['threshold']:.2f}", f"margin ≥ {ops['conservative']['margin']:.1f} log-odds",
                   delta_color="off", delta_arrow="off")
    cols[2].metric("Frozen model", f"{len(FM.features)} features", f"SHA-256 {FM.model_sha256[:10]}… verified", delta_color="off", delta_arrow="off")
    run_button("recover")
    if result is None:
        st.info("Press **Run Identity Recovery** to run the frozen model on this encounter.")
    else:
        st.success("Frozen inference ran on the matcher-visible tracklets only (no reference, no fitting).")
        with st.expander("Probability matrix (all candidate pairs)"):
            inc = [a["tracklet_id"] for a in INST["incoming"]]
            outg = [b["tracklet_id"] for b in INST["outgoing"]]
            pm = {(q["incoming"], q["outgoing"]): q for q in result["pairs"]}
            M = np.array([[pm[(i, o)]["probability"] for o in outg] for i in inc])
            txt = [[f"{pm[(i, o)]['probability']:.2f}" + ("" if pm[(i, o)]["reachable"] else " ✕") for o in outg] for i in inc]
            hm = go.Figure(go.Heatmap(z=M, x=outg, y=inc, zmin=0, zmax=1, colorscale="Blues", text=txt, texttemplate="%{text}", colorbar=dict(title="p")))
            hm.update_layout(height=120 + 50 * len(inc), margin=dict(l=10, r=10, t=30, b=10), title="Calibrated pair probabilities (✕ = unreachable)",
                             yaxis=dict(autorange="reversed"))
            plot(hm, "heat")
        render_assignment("recover")
        with st.expander("How the decision rule works"):
            st.markdown("1. **Reachability:** an incoming tracklet with no physically reachable outgoing tracklet is UNMATCHED.\n"
                        "2. **Global assignment** (same at both points): the one-to-one set of links with the highest total log-odds. "
                        "A pair below p = 0.5 is never linked, and each outgoing tracklet goes to at most one incoming tracklet, so a "
                        "tracklet whose best candidate is taken by a better-fitting tracklet is left unlinked → UNCERTAIN.\n"
                        "3. **Operating point:** an assigned link is MATCHED only if p clears the threshold *and* its margin (how "
                        "much worse the best assignment without it is) is ≥ 1.0 log-odds; otherwise UNCERTAIN.")
    st.divider()
    st.subheader("Prediction vs provider reference")
    render_reference("recover")

# ---------------------------------------------------------------- 4. explanation

with tab_explain:
    if result is None:
        st.info("Run Identity Recovery first; explanations use the values computed for that run.")
    else:
        pairs = result["pairs"]
        default = next((n for n, q in enumerate(pairs) if any(d["incoming"] == q["incoming"] and d["candidate"] == q["outgoing"]
                                                              for d in result["decisions"]["relaxed"])), 0)
        n = st.selectbox("Candidate pair", range(len(pairs)), index=default,
                         format_func=lambda j: f"{pairs[j]['incoming']} → {pairs[j]['outgoing']} (p = {pairs[j]['probability']:.2f})")
        q = pairs[n]
        f = q["features"]
        st.caption("The probability below belongs to this pair alone. Whether it becomes a committed link also depends on the global "
                   "assignment and the operating point (see the decision for this incoming tracklet at the bottom).")
        st.markdown("#### What the model measured for this pair")
        g = st.columns(5)
        g[0].metric("Distance", f"{f['dist_m']:.2f} m", f"{f['dist_per_s']:.1f} m/s needed", delta_color="off", delta_arrow="off",
                    help="Position and distance: incoming end to outgoing start, and the mean speed needed to cover it.")
        g[1].metric("Velocity change", f"{f['vel_diff']:.2f} m/s", f"{f['speed_in']:.1f} → {f['speed_out']:.1f} m/s", delta_color="off", delta_arrow="off",
                    help="Speed and acceleration: difference of the velocity vectors at both ends.")
        g[2].metric("Direction change", f"{f['heading_diff_deg']:.0f}°", "0 if either end is slow", delta_color="off", delta_arrow="off",
                    help="Direction and turning: angle between the running directions at both ends.")
        g[3].metric("Gap · extrapolation", f"{f['gap_s']:.2f} s", f"paths miss by {f['cv_mid_err']:.2f} m", delta_color="off", delta_arrow="off",
                    help="Gap and trajectory continuity: withheld time, and how far constant-velocity paths from both ends miss each other.")
        g[4].metric("Assignment regret", f"{f['assign_regret_cv']:.2f} m", f"{f['n_reach_row']:.0f} reachable candidates", delta_color="off", delta_arrow="off",
                    help="Assignment ambiguity: 0 means this pair is in the best constant-velocity assignment; larger values mean forcing it "
                         "costs that much extra extrapolation error elsewhere.")
        pc = I.path_contributions(FM, result["Xf"][n])
        st.markdown("#### How this forest reached its score")
        if pc["additive_error"] > 1e-9:
            st.error("The path decomposition does not add up for this pair, so it is not shown.")
        else:
            contrib = pd.DataFrame([dict(feature=k, category=PR.feature_category(k), contribution=v, value=f[k],
                                         meaning=I.DESCRIPTIONS.get(k, "")) for k, v in pc["contributions"].items()])
            cat = contrib.groupby("category")["contribution"].sum().reindex(list(PR.CATEGORIES))
            c1, c2 = st.columns([3, 2], gap="large")
            cb = go.Figure(go.Bar(x=cat.values, y=cat.index, orientation="h", marker_color=np.where(cat.values > 0, "#43a047", "#e53935"),
                                  text=[f"{v:+.3f}" for v in cat.values], textposition="outside", cliponaxis=False))
            cb.update_layout(height=280, margin=dict(l=10, r=40, t=30, b=10), title="Contribution by feature category (raw probability units)",
                             yaxis=dict(autorange="reversed"), xaxis_title="← lowers same-player score · raises it →")
            with c1:
                plot(cb, "catbar")
            c2.markdown(f"Forest average before any split: **{pc['bias']:.3f}**  \n"
                        f"\\+ path contributions: **{sum(pc['contributions'].values()):+.3f}**  \n"
                        f"= raw forest probability **{pc['raw_probability']:.3f}**  \n"
                        f"→ after the frozen Platt calibration: **{q['probability']:.3f}**")
            c2.caption("Saabas path attribution: exact and additive for this forest, but descriptive, not causal, and biased towards "
                       "features split near the root. It says how this forest's arithmetic arrived at the score, not why the players "
                       "moved as they did. Model behaviour is unchanged.")
            top = contrib.reindex(contrib["contribution"].abs().sort_values(ascending=False).index).head(12).iloc[::-1]
            bar = go.Figure(go.Bar(x=top["contribution"], y=top["feature"], orientation="h",
                                   marker_color=np.where(top["contribution"] > 0, "#43a047", "#e53935"),
                                   hovertext=[f"{r.category}<br>value {r.value:.3g}<br>{r.meaning}" for r in top.itertuples()], hoverinfo="text+x"))
            bar.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10), title="Top 12 individual features",
                              xaxis_title="← lowers same-player score · raises it →")
            with st.expander("Top 12 individual feature contributions"):
                plot(bar, "contrib")
            st.markdown("#### All 51 features by category")
            for name, spec in PR.CATEGORIES.items():
                sub = contrib[contrib["category"] == name]
                with st.expander(f"{name} · {len(sub)} features · contribution {sub['contribution'].sum():+.3f}"):
                    st.caption(spec["about"])
                    st.dataframe(sub.assign(value=sub["value"].round(4), contribution=sub["contribution"].round(4))
                                 [["feature", "value", "contribution", "meaning"]], hide_index=True)
        st.markdown(f"#### Decision for {q['incoming']}")
        for pt in I.POINTS:
            ds = result["decisions"][pt]
            d = next(x for x in ds if x["incoming"] == q["incoming"])
            ex = PR.explain_decision(d, pairs, ds, result["operating_points"][pt])
            st.markdown(f"**{pt.capitalize()}:** {PR.STATUS_BADGE[ex['decision']]}  \n" + "  \n".join(ex["reasons"]))
        with st.expander("Global feature importance of the frozen model (training data, not this decision)"):
            gi = pd.Series(I.global_importance(FM)).sort_values(ascending=False).head(15)
            st.bar_chart(gi, horizontal=True)
            st.caption("Impurity-based (MDI) importance from training; it favours continuous features and says nothing about one decision.")

# ---------------------------------------------------------------- 5. research results

with tab_research:
    st.info("These are the **historical** Step 6 validation and Step 7 locked-test results, read from the saved, checksummed scoring "
            "files. Nothing here is recomputed, and nothing is computed from the encounter selected above. " + DISCLAIMER)
    try:
        RR = research(ROOT)
    except D.DashboardError as e:
        st.error(str(e))
        RR = None
    if RR:
        s = st.columns(3)
        s[0].markdown(":blue-badge[Development · 10 matches]  \nThe model was **fitted** on these. Numbers are leave-one-match-out "
                      "estimates. The dashboard's demo matches are these matches, so anything seen in the demo is **in-sample**.")
        s[1].markdown(":orange-badge[Validation · 12 matches]  \nUnseen; scored once with the frozen model (Step 6) to decide "
                      "whether the locked test was worth running.")
        s[2].markdown(":green-badge[Locked test · 12 matches]  \nUnseen; evaluated **once** (Step 7) and never re-run. **The only "
                      "source for general performance claims.**")
        t = RR["rows"].get("test")
        if t is None:
            st.warning("Locked-test scoring artifacts not found.")
        else:
            st.subheader("Locked test (12 unseen matches, evaluated once)")
            m = st.columns(5)
            pcnt = t.get("pairs")
            m[0].metric("Pairwise ROC-AUC", f"{t['roc_auc']:.3f}", auc_ci(t["roc_auc_ci95"]) or None, delta_color="off", delta_arrow="off")
            for col, pt in zip((m[1:3], m[3:5]), I.POINTS):
                c = t[pt]["counts"]
                col[0].metric(f"{pt.capitalize()} coverage", pct(t[pt]["coverage"]), f"{c['matched']:,} of {c['incoming']:,} tracklets linked",
                              delta_color="off", delta_arrow="off")
                col[1].metric(f"{pt.capitalize()} false assoc.", pct(t[pt]["false_association_rate"]), f"{c['wrong']:,} of {c['matched']:,} links disagree",
                              delta_color="off", delta_arrow="off")
            with st.expander("How to read these numbers", expanded=True):
                st.markdown(
                    f"- **Pairwise ROC-AUC** ({t['roc_auc']:.3f}): over candidate pairs of high-confidence encounters"
                    + (f" ({pcnt['pairs']:,} pairs: {pcnt['positives']:,} same-player, {pcnt['negatives']:,} different-player by the "
                       "provider reference)" if pcnt else "")
                    + ", the probability that a random same-player pair scores higher than a random different-player pair. It "
                      "measures ranking only, before the assignment and thresholds.\n"
                    "- **Coverage** = committed (MATCHED) links ÷ all incoming tracklets. UNCERTAIN abstentions count in the denominator "
                    "but not in the numerator.\n"
                    "- **False association** = committed links that disagree with the provider reference ÷ all committed links. "
                    "Abstentions are never counted as errors.\n"
                    "- **The trade-off:** a lower threshold commits more links (higher coverage) but more of them disagree (higher false "
                    "association). The relaxed point targets ≤ 10% false association, the conservative point ≤ 5%. Rule-based "
                    "methods rarely abstain, so they cover more and disagree much more often (chart below).\n"
                    "- Brackets are 95% match-level bootstrap intervals; all rates are agreement with the provider reference.")
            tr = go.Figure()
            place = {"frozen@conservative": "bottom center", "step4_rf@frozen": "top left", "chain_during": "top right",
                     "hungarian_cv": "bottom center", "nearest": "top center", "constant_velocity": "bottom right"}
            for name, v in t["baselines"].items():
                frozen_pt = name.startswith("frozen@")
                tr.add_trace(go.Scatter(x=[v["coverage"]], y=[v["false_association_rate"]], mode="markers+text", text=[v["label"]],
                                        textposition=place.get(name, "top center"), name=v["label"], showlegend=False,
                                        marker=dict(size=16 if frozen_pt else 10, color="#43a047" if frozen_pt else "#9e9e9e",
                                                    symbol="star" if frozen_pt else "circle")))
            tr.add_hline(y=0.10, line=dict(color="#66bb6a", dash="dash"), annotation_text="10% target (relaxed)")
            tr.add_hline(y=0.05, line=dict(color="#26a69a", dash="dot"), annotation_text="5% target (conservative)")
            tr.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10), title="Locked test: coverage vs false association (saved values)",
                             xaxis=dict(title="coverage (share of incoming tracklets linked)", tickformat=".0%", range=[0, 1.05]),
                             yaxis=dict(title="false association (share of links that disagree)", tickformat=".0%", rangemode="tozero"))
            plot(tr, "tradeoff")
            st.caption("E0-Adapted is a FOOTBALLTrace-inspired matcher built for this task; the published FOOTBALLTrace method corrects "
                       "already-labelled tracks and was not reproduced or outperformed here.")
        table = []
        for name, label in (("development", "Development (LOMO)"), ("validation", "Validation (once)"), ("test", "Locked test (once)")):
            r = RR["rows"].get(name)
            if r is None:
                continue
            for pt in I.POINTS:
                c = r[pt]["counts"]
                table.append({"Split": label, "Matches": r["matches"], "ROC-AUC": f"{r['roc_auc']:.3f} {auc_ci(r['roc_auc_ci95'])}".strip(),
                              "Point": f"{pt} ({FM.operating_points[pt]['threshold']:.2f})",
                              "Coverage [95% CI]": f"{pct(r[pt]['coverage'])}{ci(r[pt]['coverage_ci95'])}",
                              "Linked / incoming": f"{c['matched']:,} / {c['incoming']:,}",
                              "False assoc. [95% CI]": f"{pct(r[pt]['false_association_rate'])}{ci(r[pt]['false_association_ci95'])}",
                              "Disagree / linked": f"{c['wrong']:,} / {c['matched']:,}"})
        st.subheader("Development vs validation vs test")
        st.dataframe(pd.DataFrame(table), hide_index=True)
        st.caption("LOMO = leave-one-match-out within the 10 development matches (the demo matches, so in-sample for the final model). "
                   "High-confidence reference tier. Coverage = linked incoming tracklets / all incoming; false association = links that "
                   "disagree with the provider / all links. Brackets: 95% bootstrap intervals as saved with each split's results.")
        if t is not None:
            with st.expander("Locked test: all comparison methods (table)"):
                st.dataframe(pd.DataFrame([dict(Method=v["label"], Coverage=pct(v["coverage"]), **{"False association": pct(v["false_association_rate"])})
                                           for v in t["baselines"].values()]), hide_index=True)
        st.caption("Sources: " + " · ".join(f"{k}: `{v}`" for k, v in RR["sources"].items()))
