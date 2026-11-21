"""M11 Trajectory Recovery Lab (Streamlit). Launch: ``python -m football_fatigue.m11_dashboard lab``.

Development data only (the 32 train/CV FIFA matches). Every "Run prediction" calls the GRU C and HGB C checkpoints of the outer
fold that never saw the match (``m11_infer``). The true pairing lives only in ``st.session_state["truth"]`` and is rendered only
after "Reveal ground truth".
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from football_fatigue.m11_benchmark import scenarios as SC
from football_fatigue.m11_infer import bundle as B
from football_fatigue.m11_infer import predict as P
from football_fatigue.m11_dashboard import lab as L
from football_fatigue.m11_dashboard import lab_board as LB

TITLE = "M11 Trajectory Recovery Lab"
CAVEAT = ("Trajectory-based identity recovery under **simulated** missing intervals in real FIFA World Cup 2022 tracking data "
          "(32 development matches, unseen-match fold models). Not a video occlusion demo; the frozen final held-out evaluation has not been run.")
MODEL_NAME = {"gru_c": "GRU C", "hgb_c": "HGB C"}
SUPPORT_TEXT = dict(primary="primary (GRU C)", fallback="fallback (HGB C)", unsupported="unsupported input")
OUTCOME_TEXT = dict(accepted_correct="✅ accepted and correct", accepted_wrong="❌ accepted but WRONG",
                    abstain_low_confidence="⏸ abstained (low confidence)", abstain_unsupported="🚫 abstained (unsupported input)")
DEMO_STEPS = ("Select a scenario", "Watch the approach", "The hidden interval", "Run GRU and HGB", "Compare predictions",
              "Reveal the truth", "Try another")
DEMO_GAPS = (0.8, 0.4, 0.6, 1.0, 1.2, 1.6, 2.0, 2.4, 0.2)
FACTORS = ("baseline",) + tuple(SC.FACTORS)
NOISE_LEVELS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50)
GAP_STEPS = [k / 10 for k in range(2, 25)]           # plain Python floats (exact 0.8, 0.0, ...)
SHIFT_STEPS = [k / 10 for k in range(-5, 6)]
W_KEYS = ("w_match", "w_cid", "w_vmode", "w_factor", "w_level", "w_gap", "w_shift", "w_noise", "w_drop", "w_ctx", "w_seed", "w_rseed",
          "demo_gap", "w_zoom", "w_play", "w_mode")

st.set_page_config(page_title=TITLE, page_icon="⚽", layout="wide")
ss = st.session_state
for k in W_KEYS:                                       # keep widget values when a widget is not rendered (mode switches)
    if k in ss:
        ss[k] = ss[k]


# ---------------------------------------------------------------- cached resources

@st.cache_resource(max_entries=4, show_spinner="Loading match tracking data (about 10 s the first time)…")
def match_data(m: str) -> SC.MatchData:
    return SC.MatchData(m)


@st.cache_resource(show_spinner="Loading the fold's GRU C and HGB C checkpoints…")
def predictor(fold: int) -> P.Predictor:
    return P.Predictor(fold, device="cpu")


@st.cache_data(show_spinner=False)
def encounters(m: str) -> pd.DataFrame:
    return L.encounter_table(match_data(m))


@st.cache_data(show_spinner=False)
def factor_table(f: str) -> pd.DataFrame:
    return L.factor_table(f)


MATCHES = L.development_matches()


# ---------------------------------------------------------------- scenario spec (source of truth for what is shown)

def default_spec() -> dict:
    m = MATCHES[0]
    return dict(match=m, cid=None, vmode="m2", factor="baseline", level="base", custom=dict(gap=0.8, shift=0.0, noise=0.0, dropout="none",
                                                                                           context=2.0), seed=0, origin="default")


def set_spec(**kw) -> None:
    """Change the scenario (also mirrors the sidebar widgets so they show the new selection)."""
    sp = dict(ss.get("spec") or default_spec())
    sp.update(kw)
    ss["spec"] = sp
    ss["w_match"], ss["w_vmode"], ss["w_factor"], ss["w_level"] = sp["match"], sp["vmode"], sp["factor"], sp["level"]
    if sp["cid"] is not None:
        ss["w_cid"] = sp["cid"]
    c = sp["custom"]
    ss["w_gap"], ss["w_shift"], ss["w_noise"], ss["w_drop"], ss["w_ctx"], ss["w_seed"] = c["gap"], c["shift"], c["noise"], c["dropout"], c["context"], sp["seed"]


def random_scenario(seed: int, origin: str, factor: str = "baseline", level: str = "base") -> None:
    """A reproducible random development match and encounter from ``seed``."""
    rng = np.random.default_rng(seed)
    m = MATCHES[int(rng.integers(len(MATCHES)))]
    enc = encounters(m)
    set_spec(match=m, cid=str(enc["candidate_id"].iloc[int(rng.integers(len(enc)))]), vmode="m2", factor=factor, level=level,
             origin=f"{origin} (seed {seed})")


def demo_gap_spec(g: float) -> dict:
    return dict(factor="baseline", level="base") if g == 0.8 else dict(factor="gap", level=SC.level_key(float(g)))


def ensure_scenario() -> None:
    sp = ss.get("spec")
    if sp is None:
        return
    if sp["cid"] is None:
        sp["cid"] = str(encounters(sp["match"])["candidate_id"].iloc[0])
    key = json.dumps({k: sp[k] for k in ("match", "cid", "vmode", "factor", "level", "custom", "seed")}, sort_keys=True)
    if key == ss.get("built_key"):
        return
    md = match_data(sp["match"])
    try:
        if sp["vmode"] == "m2":
            pub, truth = L.make(md, sp["cid"], factor=sp["factor"], level=sp["level"])
        else:
            pub, truth = L.make(md, sp["cid"], custom=sp["custom"], seed=int(sp["seed"]))
        err = None
    except ValueError as e:
        pub, truth, err = None, None, str(e)
    ss.update(pub=pub, truth=truth, build_error=err, built_key=key, result=None, revealed=False, frame=0, history=ss.get("history", []))
    if pub is not None:
        ss["history"] = (ss["history"] + [dict(match=pub.match_id, encounter=pub.candidate_id, variant=f"{pub.factor}={pub.level}",
                                                scenario_id=pub.scenario_id, valid=pub.valid)])[-12:]


def run_prediction() -> None:
    pub = ss.get("pub")
    if pub is None or not pub.valid:
        return
    ss["result"] = L.run_inference(pub, predictor(pub.fold))
    ss["n_runs"] = ss.get("n_runs", 0) + 1


def reveal() -> None:
    ss["revealed"] = True


# ---------------------------------------------------------------- display pieces

def scenario_header(pub: L.Public) -> None:
    p = pub.params
    variant = "baseline (0.8 s gap, clean)" if pub.factor == "baseline" else (
        f"custom: gap {p['gap']} s, shift {p['shift']:+} s, noise {p['noise'] * 100:g} cm, dropout {p['dropout']}, context {p['context']} s"
        if pub.factor == "custom" else f"{pub.factor} = {pub.level} (one factor changed from the baseline)")
    st.markdown(f"**Match {pub.match_id}** · encounter `{pub.candidate_id}` · {variant} · models: outer fold **{pub.fold}** "
                f"(never trained on match {pub.match_id})")


def playback(pub: L.Public, truth, result, *, key: str, demo_start: int | None = None, height: int = 600) -> None:
    c1, c2 = st.columns([3, 2])
    zoom = c1.radio("View", list(LB.ZOOMS), horizontal=True, key=f"{key}_zoom")
    mode = c2.radio("Playback", ["Animation", "Step by step"], horizontal=True, key=f"{key}_play",
                    help="Animation: ▶ Play / ⏸ Pause and the time slider under the pitch. Step by step: one frame per click.")
    times = LB.frame_times(pub)
    kf = LB.key_frames(pub)
    if mode == "Animation":
        start = demo_start if demo_start is not None else 0
        st.plotly_chart(LB.board(pub, truth, result, zoom=LB.ZOOMS[zoom], start=start, height=height), width="stretch", key=f"{key}_fig_a",
                        config=dict(displaylogo=False))
        return
    fk = f"{key}_frame"
    if fk not in ss or ss.get(f"{fk}_sid") != pub.scenario_id:
        ss[fk] = demo_start if demo_start is not None else 0
        ss[f"{fk}_sid"] = pub.scenario_id
    b = st.columns(7)
    def go_to(v): ss[fk] = int(np.clip(v, 0, len(times) - 1))  # noqa: E301,E704
    b[0].button("⏮ Start", key=f"{key}_s0", on_click=go_to, args=(0,), width="stretch")
    b[1].button("◀ −1 frame", key=f"{key}_m1", on_click=go_to, args=(ss[fk] - 1,), width="stretch")
    b[2].button("+1 frame ▶", key=f"{key}_p1", on_click=go_to, args=(ss[fk] + 1,), width="stretch")
    b[3].button("+0.5 s ⏩", key=f"{key}_p8", on_click=go_to, args=(ss[fk] + int(L.FRAME_HZ / 2),), width="stretch")
    b[4].button("Before gap", key=f"{key}_ka", on_click=go_to, args=(kf["approach"],), width="stretch")
    b[5].button("In gap", key=f"{key}_kg", on_click=go_to, args=(kf["gap"],), width="stretch")
    b[6].button("After gap", key=f"{key}_kb", on_click=go_to, args=(kf["after"],), width="stretch")
    st.slider("Time (frame)", 0, len(times) - 1, key=fk, format="%d",
              help=f"{L.FRAME_HZ:g} frames per second; t = 0 is the start of the hidden interval")
    st.plotly_chart(LB.board(pub, truth, result, zoom=LB.ZOOMS[zoom], start=ss[fk], animate=False, height=height), width="stretch",
                    key=f"{key}_fig_s", config=dict(displaylogo=False))


def prob_bar(p_swapped: float) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(y=[""], x=[1 - p_swapped], orientation="h", name="MATCHED", marker_color="#2e7d32", text=f"MATCHED {1 - p_swapped:.3f}",
                textposition="inside", insidetextanchor="start")
    fig.add_bar(y=[""], x=[p_swapped], orientation="h", name="SWAPPED", marker_color="#c62828", text=f"SWAPPED {p_swapped:.3f}",
                textposition="inside", insidetextanchor="end")
    fig.update_layout(barmode="stack", height=70, margin=dict(l=0, r=0, t=0, b=0), showlegend=False, xaxis=dict(range=[0, 1], visible=False),
                      yaxis=dict(visible=False))
    return fig


def mapping_text(mp: dict[str, str]) -> str:
    return " · ".join(f"{a} → {b}" for a, b in mp.items())


def model_card(col, v: dict, pred: dict, key: str) -> None:
    name = MODEL_NAME[v["model"]]
    with col.container(border=True):
        st.markdown(f"#### {name}")
        if not v.get("available"):
            st.warning("No probability: the input has too few observations for the model grid.")
            return
        st.plotly_chart(prob_bar(v["p_swapped"]), width="stretch", key=f"{key}_{v['model']}", config=dict(displayModeBar=False))
        st.markdown(f"Model's pairing: **{v['pairing']}**  \n{mapping_text(v['mapping'])}")
        st.markdown(f"Confidence **{v['confidence']:.3f}** vs its 5 % threshold {v['threshold']:.4f} → "
                    + ("clears the threshold" if v["clears_threshold"] else "**below threshold** (would abstain)"))
        if pred["support"] == "unsupported":
            st.error("Input outside the supported range: this probability is **diagnostic only, not validated**.")
        elif v["routed"]:
            st.success("This is the model the routing policy used.")
        else:
            st.info("Not the routed model for this input: shown for comparison only.")


def routing_card(col, result: dict, pub: L.Public) -> None:
    d = result["prediction"]
    with col.container(border=True):
        st.markdown("#### Operational decision")
        st.caption(f"routing policy `{P.POLICY}` (unchanged, measured limitations in the Evaluation tab)")
        if d["model"] is None:
            st.error(f"**ABSTAIN: unsupported input** ({SUPPORT_TEXT['unsupported']})")
        elif d["abstained"]:
            st.warning(f"**ABSTAIN: low confidence** ({MODEL_NAME[d['model']]} confidence {d['confidence']:.3f} < {d['threshold']:.4f})")
        else:
            st.success(f"**ACCEPTED: {d['decision']}** by {MODEL_NAME[d['model']]} ({d['support']} route)  \n"
                       f"{mapping_text(result['routed_mapping'])}")
        st.markdown("**Why:**\n" + "\n".join(f"- {r}" for r in d["reasons"]))
        st.caption(f"gap {d['gap_s']:.3f} s · declared noise {result['declared_noise_m'] * 100:g} cm · operating point 5 % error · "
                   f"inference ran {result['ran_at']} in {result['runtime_ms']} ms on CPU")
        lim = measured_limit(pub)
        if lim:
            st.caption(lim)


def measured_limit(pub: L.Public) -> str | None:
    """Milestone 2's measured coverage / realised error for the same declared condition (all 1,920 encounters)."""
    if pub.source != "fifa" or pub.factor == "custom":
        return None
    f = "gap" if pub.factor == "baseline" else pub.factor
    t = L.metrics()["by_factor"][f]["base" if pub.factor == "baseline" else pub.level]["routed"]
    if t["coverage"] in (None, 0, 0.0):
        return f"Milestone 2, this condition (n = {t['n']}): the policy abstains on all of them (unsupported)."
    return (f"Milestone 2, this condition (n = {t['n']}): coverage {t['coverage']:.1%}, realised error among accepted "
            f"{t['realized_error']:.1%} (target 5 %).")


def results_block(pub: L.Public, key: str) -> None:
    res = ss.get("result")
    if res is None:
        return
    if "error" in res:
        st.error(f"The model could not be run on this input: {res['error']}")
        return
    cols = st.columns(3)
    model_card(cols[0], res["models"]["gru_c"], res["prediction"], key)
    model_card(cols[1], res["models"]["hgb_c"], res["prediction"], key)
    routing_card(cols[2], res, pub)


def truth_block(pub: L.Public) -> None:
    truth, res = ss.get("truth"), ss.get("result")
    if not ss.get("revealed") or truth is None:
        return
    with st.container(border=True):
        st.markdown(f"### Ground truth: **{truth.pairing}**  ·  {mapping_text(truth.mapping)}")
        st.caption("From the two players' own continuous provider tracks (label re-verified against the raw positions). "
                   + " · ".join(f"{k}: jersey {v['jersey']} ({v['team_side']})" for k, v in truth.players.items()))
        if res and "prediction" in res:
            ev = L.evaluate(res, truth)
            c = st.columns(3)
            for i, m in enumerate(("gru_c", "hgb_c")):
                if m in ev["models"]:
                    c[i].metric(MODEL_NAME[m], "correct ✅" if ev["models"][m]["correct"] else "wrong ❌")
            c[2].markdown("Operational decision  \n**" + OUTCOME_TEXT[ev["outcome"]] + "**")
            if ev["abstention_avoided_error"] is not None:
                st.caption("The abstention " + ("avoided an error: the routed model leaned the wrong way." if ev["abstention_avoided_error"]
                                                else "withheld a correct answer: the routed model leaned the right way."))
        else:
            st.caption("No prediction was run for this scenario.")


def advanced(pub: L.Public) -> None:
    res = ss.get("result")
    with st.expander("Advanced information (match, fold, checkpoints, provenance)"):
        fm = B.fold_matches(pub.fold) if pub.fold >= 0 else []
        st.markdown(f"- Match **{pub.match_id}**, encounter `{pub.candidate_id}`, period {pub.period}; scenario id `{pub.scenario_id}`\n"
                    f"- Outer fold **{pub.fold}**: models trained on the 28 development matches outside this fold; unseen matches of this fold: "
                    f"{', '.join(fm)}\n"
                    f"- Parameters: `{json.dumps(pub.params)}`; label source: {pub.label_source}\n"
                    f"- Hidden interval [g0, g1] = [{pub.g0:.3f}, {pub.g1:.3f}] s (provider video time)\n"
                    f"- Routing policy `{P.POLICY}`, thresholds (5 % / 2 %): {json.dumps(B.THRESHOLDS)}")
        if res is not None:
            st.markdown("**Checkpoints used in the last run** (Stage 4C seed 0, read-only):")
            st.json(res["checkpoints"], expanded=False)
            if "prediction" in res:
                st.markdown("**Model input summary** (from the adapter):")
                st.json(res["prediction"]["input_summary"], expanded=False)
                cc = L.saved_crosscheck(pub, res)
                if cc:
                    st.markdown("**Cross-check with the saved Milestone 2 run** (the live result above was computed now; this is only a comparison):")
                    st.dataframe(pd.DataFrame([dict(model=MODEL_NAME[m], saved=cc[m]["saved"], live=cc[m]["live"], abs_diff=cc[m]["abs_diff"],
                                                    tolerance=cc[m]["tolerance"]) for m in ("gru_c", "hgb_c")]), hide_index=True)
        st.download_button("Download model-visible input (JSON)", json.dumps(pub.input_json(), indent=1), file_name=f"m11_input_{pub.scenario_id}.json",
                           mime="application/json")
        if ss.get("revealed") and ss.get("truth") is not None:
            st.download_button("Download ground truth (JSON)", json.dumps(dict(scenario_id=pub.scenario_id, **ss["truth"].to_json()), indent=1),
                               file_name=f"m11_truth_{pub.scenario_id}.json", mime="application/json")


def invalid_block(pub: L.Public) -> None:
    st.error(f"**Invalid scenario: not shown to any model.** Reason: `{pub.invalid_reason}`.  \n"
             "The Stage 2 label check was re-run for this gap / placement and did not accept it (for example, one player's provider "
             "track is not continuous enough to give a trustworthy pairing). Labels are never invented. Try another encounter or level.")


def action_row(pub: L.Public, key: str) -> None:
    b = st.columns([1, 1, 2])
    b[0].button("▶ Run prediction", type="primary", key=f"{key}_run", on_click=run_prediction, width="stretch",
                help="Runs the GRU C and HGB C checkpoints now, on the model-visible tracklets only")
    b[1].button("👁 Reveal ground truth", key=f"{key}_reveal", on_click=reveal, disabled=ss.get("revealed", False), width="stretch")
    if ss.get("result") is None:
        b[2].caption("Prediction not run yet.")
    else:
        b[2].caption(f"Prediction computed at {ss['result']['ran_at']} (live model call #{ss.get('n_runs', 0)} this session).")


# ---------------------------------------------------------------- sidebar

def sidebar() -> str:
    st.sidebar.title("⚽ " + TITLE)
    mode = st.sidebar.radio("Mode", ["Demo", "Research lab"], key="w_mode",
                            help="Demo: a guided 7-step walkthrough. Research lab: all controls.")
    if mode == "Demo":
        st.sidebar.markdown("Follow the steps on the **Scenario** tab.")
        st.sidebar.caption(CAVEAT)
        return mode
    sp = ss["spec"]
    st.sidebar.subheader("1 · Match and encounter")

    def on_match():
        set_spec(match=ss["w_match"], cid=None, origin="chosen")
        ss.pop("w_cid", None)

    st.sidebar.selectbox("Development match", MATCHES, key="w_match", on_change=on_match,
                         format_func=lambda m: f"{m} · fold {B.fold_for_match(m)} models")
    enc = encounters(ss["w_match"])
    labels = {r.candidate_id: L.encounter_label(r) for r in enc.itertuples()}      # public fields only (no pre-reveal hints)
    if ss.get("w_cid") not in labels:
        ss["w_cid"] = sp["cid"] if sp["cid"] in labels else enc["candidate_id"].iloc[0]
    st.sidebar.selectbox("Encounter (Milestone 2 selection, 60 per match)", list(labels), key="w_cid", format_func=labels.get,
                         on_change=lambda: set_spec(cid=ss["w_cid"], origin="chosen"))
    c = st.sidebar.columns([2, 3])
    ss.setdefault("w_rseed", 1)
    c[0].number_input("Seed", 0, 10**9, key="w_rseed")
    c[1].button("🎲 Seeded random", width="stretch", on_click=lambda: random_scenario(int(ss["w_rseed"]), "seeded random"),
                help="Random development match and encounter from this seed (reproducible)")
    st.sidebar.button("🎲 Truly random encounter", width="stretch", on_click=lambda: random_scenario(L.fresh_seed(), "truly random"),
                      help="Draws a new seed from the operating system; the seed is shown so it can be reproduced")
    st.sidebar.subheader("2 · Stress test")
    vm = st.sidebar.radio("Variant", ["m2", "custom"], key="w_vmode", horizontal=True,
                          format_func={"m2": "One factor (Milestone 2)", "custom": "Custom combination"}.get,
                          on_change=lambda: set_spec(vmode=ss["w_vmode"]))
    if vm == "m2":
        def on_factor():
            f = ss["w_factor"]
            set_spec(factor=f, level="base" if f == "baseline" else SC.level_key(SC.FACTORS[f][0]))

        st.sidebar.selectbox("Factor", FACTORS, key="w_factor", on_change=on_factor,
                             format_func=dict(baseline="baseline (0.8 s, clean)", gap="gap duration", shift="gap placement (shift)",
                                              noise="positional noise", dropout="observation dropout", context="context length").get)
        f = ss["w_factor"]
        if f != "baseline":
            lv = [SC.level_key(v) for v in SC.FACTORS[f]]
            if ss.get("w_level") not in lv:
                ss["w_level"] = lv[0]
            st.sidebar.select_slider("Level", lv, key="w_level", on_change=lambda: set_spec(level=ss["w_level"]),
                                     format_func=lambda v: dict(gap=f"{v} s", shift=f"{float(v):+g} s", noise=f"{float(v) * 100:g} cm",
                                                                context=f"{v} s").get(f, v))
        st.sidebar.caption("Only this factor differs from the baseline, exactly as in the Milestone 2 suite, so the saved result of "
                           "the same scenario is available as a cross-check.")
    else:
        def on_custom():
            set_spec(custom=dict(gap=float(ss["w_gap"]), shift=float(ss["w_shift"]), noise=float(ss["w_noise"]), dropout=ss["w_drop"],
                                 context=float(ss["w_ctx"])), seed=int(ss["w_seed"]))

        st.sidebar.select_slider("Gap duration (s)", GAP_STEPS, key="w_gap", on_change=on_custom)
        st.sidebar.select_slider("Gap placement: shift from closest approach (s)", SHIFT_STEPS,
                                 key="w_shift", on_change=on_custom)
        st.sidebar.select_slider("Positional noise σ", NOISE_LEVELS, key="w_noise", on_change=on_custom, format_func=lambda v: f"{v * 100:g} cm")
        st.sidebar.selectbox("Observation dropout", SC.DROPOUT_MODES, key="w_drop", on_change=on_custom)
        st.sidebar.select_slider("Context per side (s)", [0.5, 1.0, 1.5, 2.0], key="w_ctx", on_change=on_custom)
        st.sidebar.number_input("Noise / dropout / order seed", 0, 10**9, key="w_seed", on_change=on_custom)
        st.sidebar.caption("Several factors at once are allowed here; such results are not Milestone 2 comparisons and have no saved cross-check.")
    st.sidebar.divider()
    st.sidebar.caption(CAVEAT)
    return mode


# ---------------------------------------------------------------- tabs

def scenario_tab_lab() -> None:
    pub = ss.get("pub")
    if ss.get("build_error"):
        st.error(f"Scenario could not be generated: {ss['build_error']}")
        return
    if pub is None:
        st.info("Choose a match and an encounter in the sidebar.")
        return
    scenario_header(pub)
    st.caption(f"Selected by: {ss['spec'].get('origin', 'chosen')}")
    if not pub.valid:
        invalid_block(pub)
        return
    action_row(pub, "lab")
    playback(pub, ss["truth"] if ss.get("revealed") else None, ss.get("result"), key="lab")
    results_block(pub, "lab")
    truth_block(pub)
    evidence_blocks(pub, "lab")
    advanced(pub)
    with st.expander("Scenarios generated this session"):
        st.dataframe(pd.DataFrame(ss.get("history", [])), hide_index=True)


DEMO_TEXT = {
    1: "We take a real moment from a World Cup 2022 match in which **two players come within half a metre of each other**. "
       "The models used were trained **without this match** (unseen-match fold).",
    2: "Press **▶ Play**. The two coloured circles are **Player 1** and **Player 2**: we know who they are while they approach "
       "(*incoming tracklets*). Grey dots are the other players.",
    3: "Now we **simulate an occlusion**: both players' positions are removed for a short interval (brown badge). After it, two "
       "**anonymous tracks** appear, **Track X** and **Track Y** (*outgoing tracklets*, diamonds). Their colours say nothing about "
       "who they are. The question: **which track is which player?**",
    4: "Press **Run prediction**. The real GRU C (a recurrent neural network) and HGB C (gradient-boosted trees on motion features) "
       "checkpoints are executed now, on the trajectories before and after the gap only.",
    5: "Each model gives a probability for **MATCHED** (Player 1 → Track X, Player 2 → Track Y) versus **SWAPPED** (Player 1 → "
       "Track Y, Player 2 → Track X). The **routing policy** decides which model is trusted for this kind of input and "
       "**abstains** if that model's confidence is below its threshold.",
    6: "Press **Reveal ground truth**. The true pairing comes from the provider's continuous track of each player; the dotted lines "
       "show where they really went while hidden.",
    7: "Try another random scenario, or keep this encounter and **change the gap**: longer gaps are harder, and very short or very "
       "long gaps are outside the validated range (the policy abstains).",
}


def scenario_tab_demo() -> None:
    step = ss.setdefault("demo_step", 1)
    st.progress(step / len(DEMO_STEPS), text=f"Step {step} of {len(DEMO_STEPS)}: **{DEMO_STEPS[step - 1]}**")
    nav = st.columns([1, 1, 6])

    def go(s):
        ss["demo_step"] = int(np.clip(s, 1, len(DEMO_STEPS)))

    nav[0].button("◀ Back", on_click=go, args=(step - 1,), disabled=step == 1, width="stretch")
    nav[1].button("Next ▶", on_click=go, args=(step + 1,), disabled=step == len(DEMO_STEPS) or ss.get("pub") is None, type="primary",
                  width="stretch")
    st.info(DEMO_TEXT[step])

    def new_random():
        g = float(ss.get("demo_gap", 0.8))
        random_scenario(L.fresh_seed(), "demo: truly random", **demo_gap_spec(g))
        ss["demo_step"] = 2

    pub = ss.get("pub")
    if step == 1 or pub is None:
        st.button("🎲 Pick a random unseen-match scenario", type="primary", on_click=new_random)
        if pub is None:
            return
    if step == 7:
        c = st.columns([2, 2, 3])
        c[0].button("🎲 Another random scenario", on_click=new_random, width="stretch")
        c[1].selectbox("Gap for this encounter (s)", DEMO_GAPS, key="demo_gap")

        def same_with_gap():
            set_spec(vmode="m2", origin="demo: same encounter, new gap", **demo_gap_spec(float(ss["demo_gap"])))
            ss["demo_step"] = 2

        c[2].button("↻ Same encounter with this gap", on_click=same_with_gap, width="stretch")
    scenario_header(pub)
    if not pub.valid:
        invalid_block(pub)
        return
    if step >= 4:
        action_row(pub, "demo")
    kf = LB.key_frames(pub)
    start = {1: 0, 2: 0, 3: kf["gap"], 4: kf["end"], 5: kf["end"], 6: kf["end"], 7: kf["end"]}[step]
    playback(pub, ss["truth"] if ss.get("revealed") else None, ss.get("result"), key="demo",
             demo_start=start, height=560)
    if step >= 4:
        results_block(pub, "demo")
        truth_block(pub)
    evidence_blocks(pub, "demo")
    with st.expander("Legend and terms"):
        legend()
    advanced(pub)


def legend() -> None:
    st.markdown(
        "- **Incoming tracklets** (circles, Player 1 / Player 2): the last 2 s of each player before the gap; identities known.\n"
        "- **Simulated occlusion** (brown badge, dashed circle): an interval in which **both** players' positions are removed from "
        "the model input. It imitates a tracker losing two players in a crossing; it is not a real video occlusion.\n"
        "- **Outgoing tracklets** (diamonds, Track X / Track Y): the first 2 s after the gap; identity unknown to the model. Their "
        "colours are neutral and their order is random.\n"
        "- **MATCHED**: Player 1 → Track X and Player 2 → Track Y. **SWAPPED**: Player 1 → Track Y and Player 2 → Track X.\n"
        "- **Probability**: each model's P(SWAPPED); P(MATCHED) = 1 − P(SWAPPED).\n"
        "- **Confidence**: max(P, 1 − P), between 0.5 (a coin flip) and 1.\n"
        "- **Abstention**: the system declines to decide. *Low confidence*: the routed model's confidence is below its threshold "
        "(set on development data so that about 5 % of accepted answers are wrong at the 0.8 s baseline). *Unsupported*: the input "
        "is outside the range the models were validated for (gap < 0.4 s or > 1.6 s, noise > 5 cm, too few points).\n"
        "- **Routing policy `m11-routing-v1`**: GRU C for clean 0.75–0.85 s gaps; HGB C for gaps 0.4–1.6 s or noise up to 5 cm.\n"
        "- **White line** after a prediction: the operational decision (solid = accepted, dashed = abstained lean). **Dotted lines** "
        "after reveal: the true hidden paths.")


def gallery_tab() -> None:
    st.markdown("Examples come from the saved Milestone 2 run. **Fixed examples** are the first six scenarios of each category in "
                "scenario-id order (a hash), not chosen for looking good. **Random from category** draws any scenario of the category. "
                "Loading an example regenerates the scenario; press *Run prediction* to compute it live. Category names are hidden by "
                "default because the real name reveals the outcome.")
    reveal = st.toggle("Show real category names (reveals the outcome)", value=False, key="gal_reveal_names")
    cats = dict(L.GALLERY) if reveal else dict(L.GALLERY_BLIND)
    cat = st.selectbox("Category", list(cats), format_func=lambda c: f"{cats[c]}  ({len(L.category_frames()[c]):,} scenarios)")
    rows = L.gallery_fixed(cat)

    def load(r, origin):
        set_spec(match=str(r["match_id"]), cid=str(r["candidate_id"]), vmode="m2", factor=r["factor"], level=r["level"], origin=origin)
        ss["w_mode"] = "Research lab"

    tbl = pd.DataFrame([dict(example=i + 1, match=r["match_id"], variant=f"{r['factor']}={r['level']}", scenario_id=r["scenario_id"]) for i, r in enumerate(rows)])
    st.dataframe(tbl, hide_index=True)
    c = st.columns(len(rows) + 1)
    for i, r in enumerate(rows):
        c[i].button(f"Load #{i + 1}", key=f"gal_{cat}_{i}", on_click=load, args=(r, f"gallery: {cats[cat]} #{i + 1}"), width="stretch")

    def rnd():
        seed = L.fresh_seed()
        load(L.gallery_random(cat, seed), f"gallery: random from '{cats[cat]}' (seed {seed})")

    c[-1].button("🎲 Random from category", key=f"gal_{cat}_rnd", on_click=rnd, width="stretch")
    st.caption("After loading, open the **Scenario** tab (the app switches to Research lab mode).")


def evaluation_tab() -> None:
    m = L.metrics()
    st.markdown("All figures below are **saved Milestone 2 results** (`m11-scenarios-v2`): 1,920 encounters from the 32 development "
                "matches, each scored by the fold models that never saw its match.")
    with st.container(border=True):
        st.markdown("**What kind of evidence is this?**\n" + "\n".join(f"- {v}" for v in L.EVIDENCE.values()))
    o = m["overall_routed"]
    c = st.columns(5)
    c[0].metric("Scenarios", f"{m['scenarios']:,}")
    c[1].metric("Valid (label accepted)", f"{m['valid']:,}")
    c[2].metric("Routed coverage", f"{o['coverage']:.1%}")
    c[3].metric("Realised error (target 5 %)", f"{o['realized_error']:.1%}", help="Share of accepted (non-abstained) answers that are wrong")
    c[4].metric("Accepted wrong", f"{o['accepted_wrong']:,}", help=f"of {o['accepted_correct'] + o['accepted_wrong']:,} accepted answers")
    st.warning(f"Measured limitation of `{P.POLICY}`: across all tested scenarios about **{o['realized_error']:.0%}** of accepted answers "
               "are wrong, not the intended 5 %. The policy cannot see dropout or short context, trusts the declared noise, and its "
               "thresholds were set at the 0.8 s gap. It is shown unchanged; a revised policy has not been approved.")
    f = st.radio("Factor", list(SC.FACTORS), horizontal=True, format_func=L.FACTOR_UNITS.get)
    t = factor_table(f)
    st.plotly_chart(factor_chart(f, t), width="stretch")
    st.dataframe(t, hide_index=True)
    st.caption("BA = balanced accuracy, 95 % CI from a match-grouped bootstrap (1,000 resamples). 'routed model: none' rows are outside "
               "the policy's supported range: their raw BA is diagnostic only. 'invalid' = variants whose label could not be verified.")
    st.markdown("#### Movement families (baseline 0.8 s)")
    fam = m["families"]["baseline=base"]
    rows = [dict(family=k.replace("_", " "), value=str(val).replace("_", " "), n=v["n"], GRU_BA=v["gru"].get("balanced_accuracy"),
                 HGB_BA=v["hgb"].get("balanced_accuracy"), routed_error=v["routed"]["realized_error"])
            for k, vals in fam.items() for val, v in vals.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("Families are descriptive (computed from the clean provider tracks); small groups have wide uncertainty. Pairs whose "
               "positions are VISIBLE inside the gap are clearly harder than provider-ESTIMATED ones.")
    st.download_button("Download all saved metrics (JSON)", L.metrics_path().read_bytes(), file_name="m11_scenarios_metrics_v1.json")


def factor_chart(f: str, t: pd.DataFrame) -> go.Figure:
    x = t.iloc[:, 0].astype(str).tolist()
    order = x
    if f in ("gap", "noise", "shift", "context"):
        base = dict(gap="0.8", noise="0", shift="0", context="2")[f]
        vals = [float(base) if v == "baseline" else float(v) for v in x]
        idx = np.argsort(vals)
        t = t.iloc[idx]
        order = [f"{vals[i]:g}" + (" (baseline)" if x[i] == "baseline" else "") for i in idx]
    fig = go.Figure()
    for col, ci, name, color in (("GRU BA", "GRU 95% CI", "GRU C", "#1e88e5"), ("HGB BA", "HGB 95% CI", "HGB C", "#f4511e")):
        lo = [float(c.strip("[]").split(",")[0]) if c else np.nan for c in t[ci]]
        hi = [float(c.strip("[]").split(",")[1]) if c else np.nan for c in t[ci]]
        y = t[col].to_numpy(float)
        fig.add_scatter(x=order, y=y, name=name, mode="lines+markers", line=dict(color=color, width=2), marker=dict(size=9),
                        error_y=dict(type="data", array=np.array(hi) - y, arrayminus=y - np.array(lo), visible=True, thickness=1.2))
    unsup = [o for o, r in zip(order, t["routed model"]) if r.startswith("none")]
    if unsup:
        fig.add_scatter(x=unsup, y=[0.5] * len(unsup), mode="markers", name="unsupported by the policy (diagnostic)",
                        marker=dict(symbol="x", size=11, color="#757575"))
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10), yaxis=dict(title="balanced accuracy", range=[0.45, 1.0]),
                      xaxis=dict(title=L.FACTOR_UNITS[f], type="category"), legend=dict(orientation="h", y=-0.3, x=0),
                      title=dict(text=f"GRU C vs HGB C by {L.FACTOR_UNITS[f]} (0.5 = chance)", font=dict(size=14)))
    return fig


def json_tab() -> None:
    st.markdown("Run the models on **four manually prepared tracklets**: two incoming and two outgoing, each with timestamps (s) and "
                "metric pitch coordinates x, y (m). Format `m11-trajectory-input-v1`, as written by *Download model-visible input* or "
                "`python -m football_fatigue.m11_infer export-fifa`. Raw video (MP4) cannot be analysed here: detection, tracking and "
                "pitch calibration are not part of this tool.")
    src = st.radio("Input", ["Upload a JSON file", "Use the current scenario (model-visible part)"], horizontal=True)
    data = None
    if src.startswith("Upload"):
        f = st.file_uploader("Trajectory JSON", type=["json"])
        if f is not None:
            try:
                data = json.loads(f.getvalue().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                st.error(f"Not valid JSON: {e}")
    elif ss.get("pub") is not None and ss["pub"].valid:
        data = ss["pub"].input_json()
    if data is None:
        return
    try:
        pub = L.from_json(data)
    except Exception as e:                            # noqa: BLE001 - shown to the user, never hidden
        st.error(f"Input rejected: {e}")
        return
    mid = pub.match_id
    c = st.columns(3)
    if mid in B.heldout_matches():
        st.error(f"Match {mid} is a frozen held-out match: refused.")
        return
    if mid in MATCHES:
        fold = B.fold_for_match(mid)
        c[0].markdown(f"Development match {mid}: using fold **{fold}** (never trained on it).")
    else:
        fold = c[0].selectbox("Fold models (external input: none of them saw it)", list(range(8)))
    noise = c[1].number_input("Declared positional noise σ (m)", 0.0, 2.0, value=float(data.get("noise_sigma_m") or 0.0), step=0.01,
                              help="Video-derived positions are typically 0.4–1 m off: far outside the supported 5 cm")
    tr = c[2].selectbox("Known true pairing", ["unknown", "MATCHED", "SWAPPED"],
                        help="Only if a trustworthy pairing is available (e.g. manual annotation)")
    pub.fold = fold
    if st.button("▶ Run prediction on this input", type="primary"):
        ss["json_result"] = (pub.scenario_id, L.run_inference(pub, predictor(fold), noise_sigma_m=noise))
    jr = ss.get("json_result")
    if jr and jr[0] == pub.scenario_id:
        res = jr[1]
        if "error" in res:
            st.error(f"The model could not be run: {res['error']}")
        else:
            cols = st.columns(3)
            model_card(cols[0], res["models"]["gru_c"], res["prediction"], "json")
            model_card(cols[1], res["models"]["hgb_c"], res["prediction"], "json")
            routing_card(cols[2], res, pub)
            if tr == "unknown":
                st.warning("**Diagnostic only**: no ground-truth pairing was given, so this result cannot be evaluated.")
            else:
                ok = {m: v["pairing"] == tr for m, v in res["models"].items() if v.get("available")}
                st.info(f"Against the user-supplied pairing ({tr}, not verified by this tool): "
                        + ", ".join(f"{MODEL_NAME[m]} {'correct' if v else 'wrong'}" for m, v in ok.items())
                        + ". Trustworthy only if the pairing is verified and the coordinates are accurate metric positions.")
    st.plotly_chart(LB.board(pub, None, (jr[1] if jr and jr[0] == pub.scenario_id else None), zoom="close", animate=True, height=520),
                    width="stretch", key="json_fig")


def guide_tab() -> None:
    st.markdown("### What this lab shows")
    st.markdown(CAVEAT)
    legend()
    st.markdown("### A 3–5 minute live demonstration")
    st.markdown(
        "1. **(30 s) The problem.** Mode *Demo*, step 1: *Pick a random unseen-match scenario*. Explain that two players "
        "came close in a real match and that the models never trained on this match.\n"
        "2. **(45 s) Approach.** Step 2: press ▶ Play; point out Player 1 / Player 2 and the grey context players.\n"
        "3. **(30 s) The hidden interval.** Step 3: both players disappear; two anonymous tracks reappear. Ask the audience to guess.\n"
        "4. **(45 s) Live inference.** Steps 4–5: *Run prediction*; compare GRU C and HGB C probabilities, confidence and the "
        "operational decision (accept or abstain) and its reasons.\n"
        "5. **(30 s) Truth.** Step 6: *Reveal ground truth*; the dotted lines show the hidden paths; the verdict for each model.\n"
        "6. **(60 s) Stress.** Step 7: the same encounter with gap 1.6 s (HGB fallback) and 2.4 s (unsupported: the policy abstains), "
        "or another random scenario. Close with the **Evaluation** tab: 92 % balanced accuracy for GRU C at the 0.8 s baseline, "
        "falling with longer gaps and noise, and the measured ~8 % routed error across the stress suite (target 5 %).\n\n"
        "Honest framing: synthetic interruptions of real tracking data; development matches with unseen-match fold models; the final "
        "held-out evaluation has not been run.")


# ---------------------------------------------------------------- validity, raw data, failure analysis, model explanation

def evidence_blocks(pub: L.Public, key: str) -> None:
    """Validity of the test (any time), raw provider data and failure analysis (after reveal only: both reveal the pairing)."""
    if pub.source != "fifa" or not pub.valid:
        return
    md = match_data(pub.match_id)
    with st.expander("✅ Is this a valid test? (unseen match, verified checkpoints, verified label)"):
        v = L.validity(pub, predictor(pub.fold))
        st.dataframe(pd.DataFrame([dict(result="✅ yes" if x["ok"] else "❌ NO", check=x["check"], detail=x["detail"]) for x in v]),
                     hide_index=True, width="stretch")
        r = md.cand.loc[pub.candidate_id]
        st.markdown(f"- **Real close encounter:** in the raw data these two players come within **{float(r['d_min']):.2f} m** of each other "
                    f"(Stage 2 selects encounters closer than 0.5 m) at match clock {L._clock(md, int(pub.period), float(r['t_star']))}.\n"
                    f"- **Simulated occlusion:** the provider tracked both players continuously; the dashboard removes "
                    f"{pub.g1 - pub.g0:.2f} s of their positions from the model input. It is not a real camera occlusion.\n"
                    f"- **Raw source file:** `{L.raw_source(md)}`")
    if not (ss.get("revealed") and ss.get("truth") is not None):
        st.caption("🔒 Raw provider data and the failure analysis unlock after **Reveal ground truth**: they show where each player "
                   "really went, so they would give the answer away.")
        return
    truth = ss["truth"]
    with st.expander("📄 Raw provider data of the two players (World Cup tracking file)", expanded=key == "lab"):
        rw = L.raw_window(md, pub, truth)
        d = rw[rw["player"] == "Player 1"]
        fig = go.Figure()
        fig.add_vrect(x0=0, x1=pub.g1 - pub.g0, fillcolor="rgba(109,76,65,0.25)", line_width=0, annotation_text="hidden from the model",
                      annotation_position="top left")
        fig.add_hline(y=0.5, line_dash="dash", line_color="#9e9e9e", annotation_text="0.5 m encounter radius", annotation_position="bottom right")
        fig.add_scatter(x=d["t_rel_s"], y=d["distance_between_players_m"], mode="lines+markers", marker=dict(size=4), line=dict(color="#1e88e5"),
                        name="distance between the players")
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="time relative to the start of the hidden interval (s)",
                          yaxis_title="distance (m)", title=dict(text="Distance between the two players in the raw data", font=dict(size=14)),
                          showlegend=False)
        st.plotly_chart(fig, width="stretch", key=f"{key}_rawdist")
        st.caption(f"Every row is one raw frame ({md.c.fps:g} per second) of the provider file for the two players, 2 s before to 2 s after "
                   "the hidden interval. `hidden_from_model = True` rows were removed from the model input. VISIBLE = the provider saw "
                   "the player; ESTIMATED = the provider filled in the position.")
        st.dataframe(rw, hide_index=True, height=300, width="stretch")
        st.download_button("Download these raw rows (CSV)", rw.to_csv(index=False), file_name=f"m11_raw_{pub.scenario_id}.csv", key=f"{key}_rawcsv")
    with st.expander("🔍 Why could the models fail here?", expanded=key == "lab"):
        failure_analysis(pub, truth)


def failure_analysis(pub: L.Public, truth: L.Truth) -> None:
    res = ss.get("result")
    reasons = []
    p = pub.params
    if abs(float(p["gap"]) - 0.8) > 1e-6:
        reasons.append(f"The gap is {p['gap']} s. **GRU C was trained only on 0.8 s gaps** (its gap input never varied), so it extrapolates "
                       "here; HGB C uses the gap as a feature and degrades more smoothly.")
    if float(p["noise"]) > 0:
        reasons.append(f"Positions carry {float(p['noise']) * 100:g} cm of added noise. Both models were trained on clean provider data; "
                       "noise distorts the velocity estimated at the gap edges, which drives both models.")
    if p.get("dropout") not in (None, "none", "as given"):
        reasons.append(f"Observations were removed ({p['dropout']}). The adapter interpolates linearly between the remaining points, so the "
                       "input differs from the regular 15 Hz grid seen in training.")
    if p.get("context") is not None and float(p["context"]) < 2.0:
        reasons.append(f"Only {p['context']} s of context per side: fewer points to estimate speed and direction.")
    if p.get("shift") and abs(float(p["shift"])) > 0.3:
        reasons.append(f"The gap is shifted {float(p['shift']):+g} s from the closest approach, beyond the ±0.3 s used in training.")
    dif = L.difficulty(pub)
    for d in dif:
        if d["routed_error"] is not None and d["routed_error"] >= 0.08:
            reasons.append(f"This encounter is **{d['this_encounter']}** ({d['descriptor']}): at the clean 0.8 s baseline, encounters like it "
                           f"have a routed error of {d['routed_error']:.0%} (n = {d['encounters_like_it']}).")
    phy = L.physics(pub)
    if phy["pairing"] != truth.pairing:
        reasons.append("Simple physics (straight-line extrapolation at constant velocity) points to the **wrong** pairing here: the players "
                       "turned, slowed or accelerated while hidden. The models must learn when to overrule this intuition.")
    if res and "prediction" in res and res["prediction"]["confidence"] is not None and res["prediction"]["confidence"] < 0.75:
        reasons.append("The routed model's confidence is low: the evidence on the two sides of the gap is genuinely ambiguous.")
    if reasons:
        st.markdown("\n".join(f"- {r}" for r in reasons))
    else:
        st.markdown("- No specific risk factor found: clean input at the trained gap, and a movement type with low error in Milestone 2.")
    st.markdown("**Physics reference: constant-velocity mismatch (m)**")
    st.dataframe(pd.DataFrame([dict(link=k, mismatch_m=v) for k, v in phy["cost"].items()]), hide_index=True)
    st.caption(f"MATCHED total {phy['matched_cost']} m, SWAPPED total {phy['swapped_cost']} m → physics says **{phy['pairing']}** "
               f"(truth: {truth.pairing}). Each player is extrapolated forward and each track backward across the gap at constant "
               "velocity; the smaller total mismatch wins. This reference is not one of the deployed models.")
    st.markdown("**This encounter's movement type and Milestone 2 accuracy of similar encounters (0.8 s baseline)**")
    if dif:
        st.dataframe(pd.DataFrame(dif), hide_index=True, width="stretch")


def models_tab() -> None:
    st.markdown("### What problem the models solve")
    st.markdown(
        "Two players are seen **before** a hidden interval (incoming tracklets, *Player 1* and *Player 2*) and two anonymous tracks are "
        "seen **after** it (outgoing tracklets, *Track X* and *Track Y*). There are only two possible answers: **MATCHED** (Player 1 → "
        "Track X, Player 2 → Track Y) or **SWAPPED** (Player 1 → Track Y, Player 2 → Track X). Both models output one number, "
        "**p = P(SWAPPED)**, between 0 and 1. They see only positions and times: no video, no appearance, no team or jersey.")
    st.markdown("### Step 1 · From positions to model input (shared by both models)")
    st.markdown(
        "- The last **2.0 s before** the gap and the first **2.0 s after** it are resampled at **15 Hz** → 30 points per tracklet, "
        "using only observations strictly outside the hidden interval (no extrapolation; missing points are masked).\n"
        "- Positions are rotated and translated into an **encounter frame**: origin at the midpoint of the two last incoming "
        "points, x-axis along the players' combined incoming direction. So the model sees the *relative* geometry, not where on the "
        "pitch it happened.\n"
        "- Tracklet order is random, and both models are built so that exchanging Track X and Track Y turns p into 1 − p exactly.")
    st.markdown("### Step 2a · GRU C (primary model): a recurrent neural network")
    st.graphviz_chart("""digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#e3f2fd", fontname="Helvetica", fontsize=11];
      in1 [label="Player 1\n30 × 4"]; in2 [label="Player 2\n30 × 4"]; o1 [label="Track X\n30 × 4"]; o2 [label="Track Y\n30 × 4"];
      ei [label="Incoming encoder\nGRU (48) → 48-d", fillcolor="#bbdefb"]; eo [label="Outgoing encoder\nGRU (48) → 48-d", fillcolor="#ffe0b2"];
      h [label="Pair head (MLP 193→96→48→1)\nscore s(i, j) for each\nPlayer–Track link", fillcolor="#c8e6c9"];
      out [label="logit = s(1,Y) + s(2,X) − s(1,X) − s(2,Y)\np = sigmoid(logit) = P(SWAPPED)", fillcolor="#fff9c4"];
      in1 -> ei; in2 -> ei; o1 -> eo; o2 -> eo; ei -> h; eo -> h; h -> out; }""")
    st.markdown(
        "- **Input per tracklet:** 30 time steps × 4 channels = (x / 10 m, y / 10 m, mask, time to the gap edge in s). Incoming "
        "tracklets are read so that the point nearest the gap comes last.\n"
        "- **Encoders:** one GRU (hidden size 48) for incoming and one for outgoing tracklets; the GRU state at the last valid step is "
        "projected to a 48-number embedding (a learned summary of position, speed, direction and turning).\n"
        "- **Pair head:** for each of the four Player–Track combinations it combines the two embeddings [a, b, a·b, |a − b|, gap] "
        "(193 numbers) in a small network (96 → 48 → 1) into a compatibility score.\n"
        "- **Decision:** the SWAPPED links minus the MATCHED links give the logit; p = sigmoid(logit). **43,585** trainable parameters.\n"
        "- **Training:** binary cross-entropy, AdamW (learning rate 1e-3, weight decay 1e-4), batch 256, gradient clip 1.0, dropout "
        "0.1, early stopping on 3 held-aside training matches (patience 10), at most 160 epochs. Each batch is randomly mirrored "
        "(y → −y) and the tracklet orders randomly exchanged (labels flipped accordingly). **Config C:** half of the samples have the "
        "gap centred on the closest approach, half have it shifted (up to ±0.3 s). All training gaps are 0.8 s.")
    st.markdown("### Step 2b · HGB C (fallback model): gradient-boosted decision trees on 36 motion features")
    st.markdown(
        "- **Features (36):** 25 *pairing* features, each = (cost of SWAPPED links) − (cost of MATCHED links) for one motion cost: "
        "distance between extrapolated and observed positions with several motion models (nearest point, constant velocity forward / "
        "both ways, constant acceleration, 1-step / 2-step variants, also on a log scale), differences in velocity, speed, heading and "
        "acceleration, and implied-speed consistency; plus 11 *context* features (gap, separation before / after, min / max speeds, "
        "angles between the two players, best constant-velocity fit and its margin).\n"
        "- **Model:** scikit-learn HistGradientBoostingClassifier, 400 trees, learning rate 0.05, at most 31 leaves, min 40 samples per "
        "leaf, L2 1.0. Hyperparameters were chosen by inner cross-validation on training matches only.\n"
        "- **Symmetry:** trained on every sample and its mirrored copy (Track X ↔ Track Y, label flipped); prediction "
        "p = (f(x) + 1 − f(swapped x)) / 2.")
    st.markdown("### Step 3 · Routing and abstention (policy `m11-routing-v1`)")
    st.markdown("- GRU C for clean inputs with a 0.75–0.85 s gap; HGB C for gaps 0.4–1.6 s or noise up to 5 cm; anything else is "
                "**unsupported** → ABSTAIN.\n- The chosen model's confidence max(p, 1 − p) must reach its threshold (GRU 0.6425, HGB "
                "0.7875: set on development predictions for about 5 % error at the 0.8 s baseline), otherwise ABSTAIN (low confidence).")
    st.markdown("### Step 4 · How training and testing were separated")
    st.markdown("- 40 FIFA World Cup 2022 matches: **8 frozen held-out matches never used**, 32 development matches.\n"
                "- The 32 are split into 8 folds of 4 matches. For each fold, a GRU C and an HGB C are trained on the other 28 matches "
                "(25 for fitting, 3 for early stopping) and tested on the 4 unseen matches. The lab always uses the fold model that "
                "never saw the selected match.\n"
                "- Training samples: 25,331 two-player encounters (players within 0.5 m) with labels from the provider's continuous "
                "tracks; the hidden interval is simulated.")
    pub = ss.get("pub")
    if pub is None or not pub.valid:
        return
    st.markdown(f"### Live: exactly what the models receive for the current scenario (match {pub.match_id}, fold {pub.fold})")
    try:
        mi = L.model_inputs(pub)
    except Exception as e:                                # noqa: BLE001 - shown, never hidden
        st.error(f"Model input cannot be built for this scenario: {e}")
        return
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**GRU C input tensors** (encounter frame; x, y in units of 10 m; dt in s; mask 1 = observed)")
        which = st.selectbox("Tracklet", ["Player 1", "Player 2", "Track X", "Track Y"], key="mt_trk")
        arr = {"Player 1": mi["gru_in"][0], "Player 2": mi["gru_in"][1], "Track X": mi["gru_out"][0], "Track Y": mi["gru_out"][1]}[which]
        st.dataframe(pd.DataFrame(arr, columns=["x/10m", "y/10m", "mask", "dt_s"]).round(4), height=300, width="stretch")
        st.caption("Incoming tracklets are listed with the point nearest the gap first (the network reads them in reverse).")
    with c2:
        st.markdown("**HGB C input: the 36 motion features**")
        st.dataframe(pd.DataFrame([dict(feature=k, value=round(v, 4)) for k, v in mi["features"].items()]), height=340, hide_index=True,
                     width="stretch")
        st.caption("perm_* = cost(SWAPPED) − cost(MATCHED): positive values favour MATCHED.")
    h = L.training_history(pub.fold)
    if h:
        hd = pd.DataFrame(h)
        fig = go.Figure()
        fig.add_scatter(x=hd["epoch"], y=hd["train_loss"], name="training loss", line=dict(color="#1e88e5"))
        fig.add_scatter(x=hd["epoch"], y=hd["val_log_loss"], name="early-stopping matches log loss", line=dict(color="#f4511e"))
        best = int(hd["val_log_loss"].idxmin())
        fig.add_vline(x=int(hd["epoch"][best]), line_dash="dot", annotation_text=f"best epoch {int(hd['epoch'][best])} (kept)")
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="epoch", yaxis_title="binary cross-entropy",
                          title=dict(text=f"How the fold-{pub.fold} GRU C used here was trained", font=dict(size=14)),
                          legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(fig, width="stretch", key="mt_hist")
        st.caption(f"{len(hd)} epochs run. If the best epoch is near 160 the epoch budget was binding (the model was still improving).")


# ---------------------------------------------------------------- page

if "spec" not in ss:
    set_spec(**default_spec())
mode = sidebar()
ensure_scenario()
st.title(TITLE)
st.caption(CAVEAT)
tabs = st.tabs(["🎬 Scenario", "🖼 Example gallery", "📊 Evaluation (Milestone 2)", "🧠 How the models work", "📄 Load JSON tracklets",
                "❓ Guide & legend"])
with tabs[0]:
    if mode == "Demo":
        scenario_tab_demo()
    else:
        scenario_tab_lab()
with tabs[1]:
    gallery_tab()
with tabs[2]:
    evaluation_tab()
with tabs[3]:
    models_tab()
with tabs[4]:
    json_tab()
with tabs[5]:
    guide_tab()
