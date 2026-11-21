"""M11 Studio: Match & Interaction Explorer, Identity Recovery Lab, Multi-player (N x M) lab, Review queue, Evidence & provenance.

Launch: ``python -m football_fatigue.m11_dashboard --port 8620 studio`` (or ``scripts\\m11_studio.cmd``), then http://127.0.0.1:8620.
Development (train/CV) matches only; every loader checks match membership first. The older dashboards are unchanged."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from football_fatigue.m11_benchmark import scenarios as SC
from football_fatigue.m11_dashboard import lab as L
from football_fatigue.m11_dashboard import lab_board as LB
from football_fatigue.m11_dashboard import studio_figs as FG
from football_fatigue.m11_explorer import events as EV
from football_fatigue.m11_explorer import multiplayer as MP
from football_fatigue.m11_explorer import physics_live as PL
from football_fatigue.m11_explorer import review as RV
from football_fatigue.m11_explorer import tracks as TR
from football_fatigue.m11_infer import bundle as BU
from football_fatigue.m11_infer import predict as P
from football_fatigue.m11_phase2 import common as C

st.set_page_config(page_title="M11 Studio", page_icon="⚽", layout="wide")
ss = st.session_state
MATCHES = C.allowed()


@st.cache_resource(max_entries=2, show_spinner="Loading match tracking (about 15 s the first time)…")
def tracks(m: str) -> TR.MatchTracks:
    return TR.load(m)


@st.cache_resource(max_entries=4, show_spinner="Loading the Recovery Lab match data…")
def lab_match(m: str):
    return SC.MatchData(m)


@st.cache_resource(show_spinner="Loading the fold's GRU C and HGB C checkpoints…")
def predictor(fold: int) -> P.Predictor:
    return P.Predictor(fold, device="cpu")


@st.cache_data(show_spinner=False)
def jload(path: str):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def clock(t: float, p0: float) -> str:
    s = max(0.0, t - p0)
    return f"{int(s // 60):02d}:{s % 60:04.1f}"


# ======================================================================== Workspace A: Match & Interaction Explorer

def explorer() -> None:
    st.header("Match & Interaction Explorer")
    st.caption("Every catalogued interaction in the chosen window (Phase 1 catalogue, canonical event ids). Proximity is geometry, not "
               "occlusion; ESTIMATED is provider metadata, not verified visibility. Development matches only.")
    c = st.columns([1.2, 0.8, 2.6, 1, 1, 1])
    m = c[0].selectbox("Match", MATCHES, key="ex_match")
    M = tracks(m)
    per = M.periods()
    p = c[1].selectbox("Period", per["period"].tolist(), key="ex_period")
    p0, p1 = float(per.set_index("period").loc[p, "start"]), float(per.set_index("period").loc[p, "end"])
    if ss.get("ex_jump") is not None:
        ss["ex_start"] = float(np.clip(ss.pop("ex_jump"), 0, p1 - p0 - 5))
    start_rel = c[2].slider("Window start (match clock in the period, s)", 0.0, float(p1 - p0 - 5), key="ex_start", value=float(ss.get("ex_start", 600.0)), step=1.0,
                            format="%.0f s")
    length = c[3].selectbox("Window", [5, 10, 20, 30, 60], index=1, key="ex_len", format_func=lambda v: f"{v} s")
    radius = c[4].selectbox("Radius", EV.RADII, index=1, key="ex_radius", format_func=lambda v: f"{v} m")
    hz = c[5].selectbox("Playback", [5, 10, 15], index=0, key="ex_hz", format_func=lambda v: f"{v} Hz")
    t0, t1 = p0 + start_rel, p0 + start_rel + length
    with st.expander("Filters", expanded=False):
        f = st.columns(4)
        teams = f[0].radio("Team relation", ["all", "same", "opposite"], horizontal=True, key="ex_teams")
        mind = f[1].number_input("Minimum proximity duration (s)", 0.0, 10.0, 0.0, 0.1, key="ex_mind")
        sl = M.slots()
        players = f[2].multiselect("Players (slot · side · shirt)", sl["slot"].tolist(), key="ex_players",
                                   format_func=lambda s: f"{s} · {sl.loc[s, 'side']} · #{sl.loc[s, 'jersey']}")
        ctypes = f[3].multiselect("Crossing types", ["crossing_synchronous", "crossing_near_synchronous", "crossing_asynchronous"],
                                  default=["crossing_synchronous", "crossing_near_synchronous", "crossing_asynchronous"], key="ex_ctypes")
        show_self = st.checkbox("Show self-crossings (one player's own loops; never an interaction)", value=True, key="ex_self")
    W = EV.window_events(m, int(p), t0, t1, radius, teams=None if teams == "all" else teams, min_duration_s=mind, slots=set(players) or None,
                         crossing_types=tuple(ctypes))
    selfx = M.self_crossings(int(p), t0 - 0.0, t1, slots=set(players) or None) if show_self else None
    cnt = W.counts()
    if selfx is not None:
        cnt["self_crossing"] = len(selfx)
    st.markdown(" · ".join(f"**{v}** {k.replace('_', ' ')}" for k, v in cnt.items()) + f"  ·  query {W.query_ms} ms")

    sel = ss.get("ex_sel")
    xr = yr = None
    hl: set[int] = set(players)
    if sel is not None and ss.get("ex_zoom", True):
        xr, yr = FG.zoom_box(sel["x"], sel["y"], ss.get("ex_zoom_half", 10.0))
        hl |= set(sel["slots"])
    left, right = st.columns([3, 2])
    with left:
        fr = M.frames(int(p), t0, t1, hz)
        fig = FG.match_figure(fr, sl, crossings=W.crossings, edges=W.edges, selfx=selfx, highlight=hl, pitch=M.pitch, xr=xr, yr=yr,
                              frame_ms=int(1000 / hz))
        fig.update_layout(title=f"Match {m} · period {p} · {clock(t0, p0)}–{clock(t1, p0)} · all {fr['slot'].nunique()} players")
        st.plotly_chart(fig, key="ex_fig", config=dict(scrollZoom=True, toImageButtonOptions=dict(format="png", filename=f"m11_{m}_p{p}_{int(t0)}")))
        z = st.columns(3)
        z[0].toggle("Zoom to the selected event", value=True, key="ex_zoom")
        z[1].slider("Zoom half-width (m)", 3.0, 30.0, 10.0, 1.0, key="ex_zoom_half")
        if z[2].button("Clear selection"):
            ss.pop("ex_sel", None)
            st.rerun()
        st.plotly_chart(FG.timeline_figure(W.edges, W.crossings, selfx, t0, t1), key="ex_tl")
    with right:
        tabs = st.tabs(["Proximity", "Crossings", "Groups", "Players & pairs", "Self-crossings", "Definitions"])
        with tabs[0]:
            cols = ["edge_id", "slot_a", "slot_b", "start", "end", "duration_s", "d_min", "context", "status_composition", "cv_tier", "label_risk",
                    "estimated_comotion", "boundary_uncertain"]
            ev = st.dataframe(W.edges[cols], hide_index=True, on_select="rerun", selection_mode="single-row", key="ex_tab_e", height=260)
            if ev.selection.rows:
                r = W.edges.iloc[ev.selection.rows[0]]
                ss["ex_sel"] = dict(kind="edge", id=r["edge_id"], x=float((r["x_a"] + r["x_b"]) / 2), y=float((r["y_a"] + r["y_b"]) / 2),
                                    slots=[int(r["slot_a"]), int(r["slot_b"])])
        with tabs[1]:
            cc = ["crossing_id", "event_type", "slot_a", "slot_b", "ta", "tb", "abs_dt_s", "d_sync_m", "x", "y"] + (["within_radius"] if "within_radius" in W.crossings else [])
            cv = st.dataframe(W.crossings[cc], hide_index=True, on_select="rerun", selection_mode="single-row", key="ex_tab_c", height=260)
            if cv.selection.rows:
                r = W.crossings.iloc[cv.selection.rows[0]]
                ss["ex_sel"] = dict(kind="crossing", id=r["crossing_id"], x=float(r["x"]), y=float(r["y"]), slots=[int(r["slot_a"]), int(r["slot_b"])])
        with tabs[2]:
            st.caption("link 0 s = simultaneous group; link 1 s = temporally connected group. **n_players** is the connected size; "
                       "**max_simultaneous_players** is how many were close at one instant. `inflated` = connected size exceeds it.")
            g = W.groups[["link_s", "group_id", "n_players", "max_simultaneous_players", "inflated", "players", "start", "end", "n_edges", "n_merges",
                          "n_splits"]].copy()
            g["players"] = g["players"].map(lambda v: ", ".join(map(str, EV.group_players(v))))
            gv = st.dataframe(g, hide_index=True, on_select="rerun", selection_mode="single-row", key="ex_tab_g", height=260)
            if gv.selection.rows:
                r = W.groups.iloc[gv.selection.rows[0]]
                ss["mp_group"] = dict(match=m, period=int(p), slots=EV.group_players(r["players"]), t=float((r["start"] + r["end"]) / 2),
                                      max_sim=r["max_simultaneous_players"])
                st.info("Group sent to the **Multi-player (N×M)** workspace (exploratory scenario).")
        with tabs[3]:
            st.caption("Players with several partners (sequential interactions) and pairs that meet more than once in this window.")
            st.dataframe(W.players.sort_values("n_edges", ascending=False), hide_index=True, height=150)
            st.dataframe(W.pairs.sort_values("n_edges", ascending=False), hide_index=True, height=150)
        with tabs[4]:
            st.caption("Detected per continuous chain after 0.25 m arc-length thinning; ≥ 1 s between visits, ≥ 2 m path, ≥ 0.5 m loop extent. "
                       "A self-crossing is never a two-player encounter or occlusion.")
            if selfx is not None:
                sv = st.dataframe(selfx, hide_index=True, on_select="rerun", selection_mode="single-row", key="ex_tab_s", height=260)
                if sv.selection.rows:
                    r = selfx.iloc[sv.selection.rows[0]]
                    ss["ex_sel"] = dict(kind="self", id=f"self-{r['slot']}-{r['t_first']:.2f}", x=float(r["x"]), y=float(r["y"]), slots=[int(r["slot"])])
        with tabs[5]:
            for k, v in EV.EVENT_TYPES.items():
                st.markdown(f"- **{k.replace('_', ' ')}**: {v}")
        inspector(M, W, radius, p0)
        b = st.columns(2)
        if b[0].button("◀ Previous window", width="stretch"):
            ss["ex_jump"] = start_rel - length
            st.rerun()
        if b[1].button("Next window ▶", width="stretch"):
            ss["ex_jump"] = start_rel + length
            st.rerun()


def inspector(M: TR.MatchTracks, W: EV.WindowEvents, radius: float, p0: float) -> None:
    sel = ss.get("ex_sel")
    with st.container(border=True):
        st.subheader("Event inspector")
        if sel is None:
            st.caption("Select a row in a table to inspect it and zoom the pitch to it.")
            return
        st.markdown(f"**{sel['kind']}** `{sel['id']}` · slots {', '.join(map(str, sel['slots']))}")
        if sel["kind"] == "edge":
            r = W.edges[W.edges["edge_id"] == sel["id"]]
            if not len(r):
                st.caption("Not in the current window.")
                return
            r = r.iloc[0]
            for line in EV.explain_edge(r):
                st.markdown(f"- {line}")
            a = M.path(int(r["slot_a"]), W.period, r["start"] - 3, r["end"] + 3)
            b = M.path(int(r["slot_b"]), W.period, r["start"] - 3, r["end"] + 3)
            st.plotly_chart(FG.distance_figure(a, b, radius, r["start"], r["end"]), key="ex_dist")
            if st.button("Use this encounter in the Recovery Lab", help="Opens the lab on the Stage 2 candidate of this pair if one exists"):
                send_to_lab(W.match, int(r["slot_a"]), int(r["slot_b"]), float(r["t_star"]))
        elif sel["kind"] == "crossing":
            r = W.crossings[W.crossings["crossing_id"] == sel["id"]].iloc[0]
            st.markdown(f"- {EV.EVENT_TYPES[r['event_type']]}\n- Player A passes at {r['ta']:.2f} s, player B at {r['tb']:.2f} s "
                        f"(|Δt| {r['abs_dt_s']:.2f} s); when the first passes, the other is {r['d_sync_m']:.2f} m away.")
            a = M.path(int(r["slot_a"]), W.period, min(r["ta"], r["tb"]) - 3, max(r["ta"], r["tb"]) + 3)
            b = M.path(int(r["slot_b"]), W.period, min(r["ta"], r["tb"]) - 3, max(r["ta"], r["tb"]) + 3)
            st.plotly_chart(FG.distance_figure(a, b, radius), key="ex_dist_c")
        else:
            st.markdown("- One player's path returns over itself. This is **not** an interaction with another player, not occlusion and never "
                        "an association candidate.")


def send_to_lab(m: str, sa: int, sb: int, t: float) -> None:
    md = lab_match(m)
    c = md.cand.reset_index(drop=True)
    k = c[((c["slot_a"] == sa) & (c["slot_b"] == sb)) | ((c["slot_a"] == sb) & (c["slot_b"] == sa))]
    if not len(k):
        st.warning("No Stage 2 candidate for this pair (the lab uses validated 0.5 m / 0.8 s candidates only).")
        return
    k = k.iloc[int(np.argmin(np.abs(k["t_star"].to_numpy() - t)))]
    if abs(float(k["t_star"]) - t) > 5:
        st.warning("No validated candidate within 5 s of this event.")
        return
    ss["rl_spec"] = dict(match=m, cid=str(k["candidate_id"]), seed=0)
    ss.pop("rl_result", None)
    ss["rl_reveal"] = False
    st.success("Sent. Open **Identity Recovery Lab** in the sidebar.")


# ======================================================================== Workspace B: Identity Recovery Lab

AVAILABILITY = [
    dict(model="GRU C (gap-last, shift-mix, 160 ep)", live="yes", source="Stage 4C seed-0 outer-fold checkpoints (SHA-256 checked)"),
    dict(model="HGB C", live="yes", source="Stage 4C outer-fold models (SHA-256 checked)"),
    dict(model="Kalman / CV / CA / nearest / direction (deterministic)", live="yes", source="Phase 2 baselines, no learned parameters"),
    dict(model="LSTM", live="UNAVAILABLE", source="no validated fair-benchmark checkpoint; needs approved GPU training (m11_fair train)"),
    dict(model="SVM", live="UNAVAILABLE", source="no saved model; historical OOF and CPU-pilot results are summaries on Stage 3 samples only"),
]


def recovery_lab() -> None:
    st.header("Identity Recovery Lab")
    st.caption("Before → hidden interval → after → prediction → ground truth. The hidden interval is a **synthetic observation gap** on real "
               "provider tracking, not a camera occlusion. Each match is scored by the fold models that never saw it.")
    spec = ss.setdefault("rl_spec", dict(match=MATCHES[0], cid=None, seed=0))
    c = st.columns([1, 2.4, 1, 1])
    m = c[0].selectbox("Match", MATCHES, index=MATCHES.index(spec["match"]), key="rl_m")
    md = lab_match(m)
    enc = L.encounter_table(md)
    labels = {r.candidate_id: L.encounter_label(r) for r in enc.itertuples()}           # public fields only
    extra = spec["cid"] if spec["cid"] and spec["cid"] not in labels and m == spec["match"] else None
    opts = list(labels) + ([extra] if extra else [])
    cur = spec["cid"] if spec["cid"] in opts and m == spec["match"] else opts[0]
    cid = c[1].selectbox("Encounter (labels show only public information)", opts, index=opts.index(cur),
                         format_func=lambda k: labels.get(k, f"sent from explorer · {k[:8]}"), key="rl_cid")
    seed = c[2].number_input("Scenario seed", 0, 10**6, int(spec.get("seed", 0)), key="rl_seed")
    if c[3].button("🎲 Random encounter", width="stretch"):
        s = L.fresh_seed()
        mm = MATCHES[L.pick(len(MATCHES), s)]
        e2 = L.encounter_table(lab_match(mm))
        ss["rl_spec"] = dict(match=mm, cid=str(e2["candidate_id"].iloc[L.pick(len(e2), s + 1)]), seed=0)
        ss.pop("rl_result", None)
        ss["rl_reveal"] = False
        st.rerun()
    with st.expander("Scenario parameters (gap, placement, noise, dropout)", expanded=False):
        q = st.columns(5)
        gap = q[0].select_slider("Gap (s)", [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0, 2.4], value=0.8, key="rl_gap")
        shift = q[1].select_slider("Gap centre offset from closest approach (s)", [-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5], value=0.0, key="rl_shift")
        noise = q[2].select_slider("Noise σ (m)", [0.0, 0.01, 0.02, 0.05, 0.1, 0.25], value=0.0, key="rl_noise")
        drop = q[3].selectbox("Dropout", ["none", "random10", "random30", "random50", "burst0p5", "rate7p5"], key="rl_drop")
        ctx = q[4].select_slider("Context (s)", [0.5, 1.0, 1.5, 2.0], value=2.0, key="rl_ctx")
    key = (m, cid, int(seed), gap, shift, noise, drop, ctx)
    if ss.get("rl_key") != key:
        ss["rl_key"] = key
        ss["rl_spec"] = dict(match=m, cid=cid, seed=int(seed))
        ss.pop("rl_result", None)
        ss.pop("rl_phys", None)
        ss["rl_reveal"] = False
    custom = dict(gap=gap, shift=shift, noise=noise, dropout=drop, context=ctx)
    try:
        pub, truth = (L.make(md, cid) if custom == dict(gap=0.8, shift=0.0, noise=0.0, dropout="none", context=2.0) and seed == 0
                      else L.make(md, cid, custom=custom, seed=int(seed)))
    except Exception as e:
        st.error(f"Scenario could not be generated: {e}")
        return
    if not pub.valid:
        st.warning(f"No valid scenario for these parameters: {pub.invalid_reason}")
        return
    reveal = bool(ss.get("rl_reveal"))
    result = ss.get("rl_result")
    left, right = st.columns([3, 2])
    with left:
        zoom = st.radio("View", ["close", "wide", "full"], horizontal=True, key="rl_zoom")
        fig = LB.board(pub, truth if reveal else None, result, zoom=zoom)
        st.plotly_chart(fig, key="rl_board")
        st.caption("Player 1 / Player 2 (blue, purple) are the incoming tracklets with known identity. Track X / Track Y (orange, yellow) "
                   "are anonymous outgoing tracklets; their colours say nothing about identity until reveal.")
    with right:
        b = st.columns(3)
        if b[0].button("▶ Run prediction", type="primary", width="stretch"):
            ss["rl_result"] = L.run_inference(pub, predictor(md.fold), noise_sigma_m=noise)
            ss["rl_phys"] = PL.run(pub.incoming, pub.outgoing, pub.g0, pub.g1)
            st.rerun()
        if b[1].button("👁 Reveal ground truth", width="stretch", disabled=result is None, help="Available after a prediction has been run"):
            ss["rl_reveal"] = True
            st.rerun()
        if b[2].button("🙈 Hide truth", width="stretch", disabled=not reveal):
            ss["rl_reveal"] = False
            st.rerun()
        model_panel(result, ss.get("rl_phys"), truth if reveal else None)
        with st.expander("Model availability", expanded=False):
            st.dataframe(pd.DataFrame(AVAILABILITY), hide_index=True)
        with st.expander("What the models receive (features and assumptions)"):
            try:
                mi = L.model_inputs(pub)
                st.markdown(f"- Hidden interval {pub.g1 - pub.g0:.2f} s; context {ctx} s per side; 15 Hz grid; sample frame from incoming data only.\n"
                            "- GRU: 2 × 2 tracklet tensors (x, y, mask, dt). HGB: 36 motion features from the same tracklets.\n"
                            "- Never used: identities, provider status, catalogue difficulty tiers, any sample inside the gap.")
                st.dataframe(pd.DataFrame(mi["features"].items(), columns=["feature", "value"]), hide_index=True, height=220)
            except Exception as e:
                st.caption(f"inputs unavailable: {e}")
        with st.expander("Validity checks"):
            st.dataframe(pd.DataFrame(L.validity(pub)), hide_index=True)
        if reveal and truth is not None:
            with st.container(border=True):
                st.subheader("Ground truth (provider identities)")
                st.markdown(f"**{truth.pairing}**: " + ", ".join(f"{k} → {v}" for k, v in truth.mapping.items()))
                st.dataframe(pd.DataFrame(truth.players).T, height=110)
                st.caption("Provider chains define the truth; they are not human-verified (see the Review queue).")


def model_panel(result, phys, truth) -> None:
    with st.container(border=True):
        st.subheader("Model answers")
        if result is None:
            st.caption("Press **Run prediction**. Answers appear before the truth can be revealed.")
            return
        if "error" in result:
            st.error(result["error"])
            return
        d = result["prediction"]
        ev = L.evaluate(result, truth) if truth is not None else None
        rows = []
        for k, v in result["models"].items():
            if not v.get("available"):
                rows.append(dict(model=k, answer="n/a"))
                continue
            rows.append(dict(model=k.upper().replace("_", " "), answer=v["pairing"], **{"P(SWAPPED)": round(v["p_swapped"], 3)},
                             confidence=round(v["confidence"], 3), threshold=v["threshold"], routed="yes" if v["routed"] else "",
                             correct=("✔" if ev["models"][k]["correct"] else "✘") if ev else "?"))
        if phys and phys.get("available"):
            for r in phys["rows"]:
                ok = (r["decision"] == truth.pairing) if truth is not None else None
                rows.append(dict(model=r["name"], answer=r["decision"], **{"P(SWAPPED)": None}, confidence=None, threshold=None,
                                 **{"physics margin": round(r["margin"], 3)}, routed="", correct=("✔" if ok else "✘") if truth is not None else "?"))
        rows += [dict(model="LSTM", answer="unavailable", correct="—"), dict(model="SVM", answer="unavailable", correct="—")]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        dec = "ABSTAIN" if d["abstained"] or d["model"] is None else d["predicted_pairing"]
        st.markdown(f"**Routed decision ({d['model'] or 'none'})**: {dec} · support: {d['support']}"
                    + (f" · outcome: **{ev['outcome'].replace('_', ' ')}**" if ev else ""))
        if phys and phys.get("available"):
            st.caption(f"Kalman 0.99 gate: {phys['kalman_gate']}. Deterministic margins are cost differences, not probabilities. "
                       "Phase 2 independent BA on the 0.8 s population: Kalman forward 0.667, GRU (historical OOF) 0.900.")
        st.caption("Probabilities are model scores; their calibration was measured on development OOF (ECE ≈ 0.01), not on this scenario.")


# ======================================================================== Workspace C: Multi-player N x M

def multiplayer() -> None:
    st.header("Multi-player association (N × M)")
    st.warning("No learned model supports N × M yet; the binary MATCHED / SWAPPED models are NOT applied here. The solver is the deterministic "
               "Kalman + Hungarian with a null option (gate χ²₂ 0.99). Synthetic scenarios test the code; real-group scenarios are exploratory "
               "and use provider identities (not human-verified) as truth.")
    src = st.radio("Scenario source", ["Synthetic curriculum level", "Real simultaneous group (from the explorer)", "Upload JSON"], horizontal=True, key="mp_src")
    d = None
    if src.startswith("Synthetic"):
        c = st.columns(3)
        lv = c[0].selectbox("Level", list(MP.LEVELS), format_func=lambda k: f"L{k}: {MP.LEVELS[k]['name']}", key="mp_lv")
        sd = c[1].number_input("Seed", 0, 10**6, 0, key="mp_seed")
        d = MP.generate(int(lv), int(sd))
    elif src.startswith("Real"):
        g = ss.get("mp_group")
        if not g:
            st.info("Select a group row in the explorer's **Groups** tab first.")
            return
        c = st.columns(3)
        gap = c[0].select_slider("Gap (s)", [0.4, 0.8, 1.2, 1.6], value=0.8, key="mp_gap")
        asy = c[1].select_slider("Asynchrony (± s)", [0.0, 0.2, 0.4], value=0.0, key="mp_async")
        sd = c[2].number_input("Seed", 0, 10**6, 0, key="mp_rseed")
        st.caption(f"Match {g['match']} p{g['period']} · players {g['slots']} · max simultaneously close: {g['max_sim']}")
        try:
            d = MP.from_real_group(tracks(g["match"]), g["slots"], g["period"], g["t"], gap, asy, int(sd))
        except ValueError as e:
            st.error(str(e))
            return
    else:
        up = st.file_uploader("m11-multiplayer-association-v1 JSON", type="json", key="mp_up")
        if up is None:
            st.caption("Example: `docs/research/m11_multiplayer_examples/` (synthetic).")
            return
        d = json.loads(up.getvalue().decode("utf-8"))
    probs = MP.validate(d)
    if probs:
        st.error("Invalid scenario: " + "; ".join(probs))
        return
    dec = MP.solve(d)
    reveal = st.toggle("Reveal truth", value=False, key="mp_reveal", disabled=d.get("truth") is None)
    l, r = st.columns([3, 2])
    l.plotly_chart(FG.nxm_figure(d, dec.links, reveal), key="mp_fig")
    with r:
        st.plotly_chart(FG.assignment_heatmap(d, dec.cost, dec.feasible, dec.links), key="mp_heat")
        st.dataframe(pd.DataFrame([dict(incoming=i, linked_to=o or "unmatched (null)") for i, o in dec.links.items()]), hide_index=True)
        st.caption(f"Unmatched outgoing: {', '.join(dec.unmatched_outgoing) or 'none'} · solver {dec.runtime_ms} ms")
        if reveal and d.get("truth") is not None:
            s = MP.score(dec, d["truth"])
            st.markdown(f"**Exact assignment:** {'✔' if s['exact'] else '✘'} · links correct {s['links_tp']}/{s['links_true']} · "
                        f"null correct {s['null_tp']}/{s['null_true']} · identity switches {s['id_switches']}")
        st.markdown("**Binary GRU / HGB:** not applicable (two-player models).")
    with st.expander("Synthetic curriculum benchmark (saved)"):
        b = jload(str(C.PROC / "m11-nxm-synthetic-benchmark-v1" / "results.json"))
        if b:
            st.dataframe(pd.DataFrame([dict(level=k, name=v["level"], **v["kalman_hungarian_null"]) for k, v in b["levels"].items()]), hide_index=True)
            st.caption(b.get("note", ""))
        else:
            st.caption("Not built: `python -m football_fatigue.m11_explorer nxm-benchmark`.")


# ======================================================================== Workspace D: Review queue

def review_queue() -> None:
    st.header("Human review queue (125 events)")
    st.caption("100 Phase 2 stratified review events + 25 severe motion contradictions. Nothing is marked reviewed unless a named reviewer saves "
               "a decision. Figures are aids, not ground truth.")
    q = RV.queue()
    rv = st.text_input("Reviewer name (required to save)", key="rv_name")
    dec = RV.decisions()
    mine = dec[dec["reviewer"] == RV._safe(rv)] if rv.strip() else dec.iloc[0:0]
    st.progress(len(mine) / len(q), text=f"{len(mine)} / {len(q)} reviewed by {rv or '—'}")
    i = int(ss.setdefault("rv_i", 0))
    c = st.columns([1, 1, 3])
    if c[0].button("◀ Previous", width="stretch"):
        ss["rv_i"] = (i - 1) % len(q)
        st.rerun()
    if c[1].button("Next ▶", width="stretch"):
        ss["rv_i"] = (i + 1) % len(q)
        st.rerun()
    pick = c[2].selectbox("Jump to", list(range(len(q))), index=i, format_func=lambda k: f"{q.loc[k, 'review_id']} · {q.loc[k, 'stratum']}", key="rv_jump")
    if pick != i:
        ss["rv_i"] = pick
        st.rerun()
    r = q.loc[i]
    st.markdown(f"**{r['review_id']}** · {r['source']} · match {r['match_id']} p{r['period']} · slots {r['slot_a']}–{r['slot_b']} · "
                f"status **{r['status']}** · tier {r['cv_tier']} · label risk {r['label_risk']} · clone-like {r['clone_like']}")
    l, rr = st.columns([3, 2])
    with l:
        M = tracks(str(r["match_id"]))
        t0, t1 = float(r["t_star"]) - 4, float(r["t_star"]) + 4
        fr = M.frames(int(r["period"]), t0, t1, 10)
        fr = fr[fr["slot"].isin([r["slot_a"], r["slot_b"]]) | (np.hypot(fr["x"] - fr["x"].median(), fr["y"] - fr["y"].median()) < 15)]
        x0 = fr[fr["slot"] == r["slot_a"]][["x", "y"]].mean()
        xr, yr = FG.zoom_box(float(x0["x"]), float(x0["y"]), 12)
        st.plotly_chart(FG.match_figure(fr, M.slots(), highlight={int(r["slot_a"]), int(r["slot_b"])}, pitch=M.pitch, xr=xr, yr=yr, frame_ms=100, height=480),
                        key="rv_fig")
        a, b = M.path(int(r["slot_a"]), int(r["period"]), t0, t1), M.path(int(r["slot_b"]), int(r["period"]), t0, t1)
        st.plotly_chart(FG.distance_figure(a, b, float(r["radius_m"]), float(r["start"]), float(r["end"])), key="rv_dist")
        if r["figure"]:
            with st.expander("Phase 2 static figure"):
                st.image(r["figure"])
    with rr:
        with st.form("rv_form"):
            prev = mine[mine["review_id"] == r["review_id"]]
            ans = {}
            for k, (qq, opts) in RV.QUESTIONS.items():
                cur = prev.iloc[0][k] if len(prev) and prev.iloc[0][k] in opts else opts[0]
                ans[k] = st.radio(qq, opts, index=opts.index(cur), key=f"rv_{k}_{i}")
            notes = st.text_area("Notes", value=prev.iloc[0]["notes"] if len(prev) else "", key=f"rv_notes_{i}")
            if st.form_submit_button("💾 Save decision", type="primary", disabled=not rv.strip()):
                RV.save_decision(rv, r["review_id"], ans, notes)
                st.success("Saved (appended).")
                st.rerun()
        st.markdown(f"State: **{'human reviewed' if len(prev) else 'not reviewed'}**")
        if st.button("Export review manifest (new CSV)", disabled=not rv.strip()):
            st.success(f"Written: {RV.export(rv)}")


# ======================================================================== Workspace E: Evidence & provenance

def evidence() -> None:
    st.header("Evidence & provenance")
    st.caption("Historical results are saved evaluations (development OOF; selection-optimistic where noted). Live predictions are only made in "
               "the Recovery Lab. No held-out or official FIFA test data is used anywhere.")
    amb = jload(str(C.PROC / "m11-phase2-ambiguity-benchmark-v1" / "ambiguity_results.json"))
    t = st.tabs(["Model comparison", "Physics by gap", "Strata", "Fair benchmark status", "Provenance"])
    with t[0]:
        rows = [dict(model="GRU gap-last (Stage 4, 80 ep)", kind="historical OOF", independent_BA=0.900, note="24/24 fits near the epoch cap"),
                dict(model="LSTM gap-last (Stage 4, 80 ep)", kind="historical OOF", independent_BA=0.858, note="unfair: under-trained, untuned"),
                dict(model="SVM Nystroem (Stage 4)", kind="historical OOF", independent_BA=0.859, note="grid edge-bound"),
                dict(model="HGB (Stage 4)", kind="historical OOF", independent_BA=0.870, note="largest leaf count always chosen")]
        if amb:
            for k, v in amb["tables"]["g0p80"]["methods"].items():
                rows.append(dict(model=f"physics: {k}", kind="deterministic (Phase 2)", independent_BA=v["independent"]["balanced_accuracy"],
                                 note=""))
        pil = jload(str(C.PROC / "m11-fair-cpu-pilot-v1" / "summary.json"))
        if pil:
            for k, v in pil["evaluation"]["models"].items():
                if k.startswith("pilot_"):
                    rows.append(dict(model=k, kind="CPU PILOT (reduced budget)", independent_BA=v["independent"]["balanced_accuracy"], note="no saved model"))
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    with t[1]:
        if amb:
            st.caption("Independent balanced accuracy by gap table (g0p40 0.4 s, g0p80 0.8 s, g1p60 1.6 s, g0p80_shift centre off t*).")
            st.dataframe(pd.DataFrame(amb["by_gap_table_independent_ba"]))
    with t[2]:
        if amb:
            s = amb["strata_independent"]
            pick = st.selectbox("Stratum (distance, speed, status, context, group size, clone-like, …)", list(s), key="evd_s")
            st.dataframe(pd.DataFrame(s[pick]).T)
            st.markdown("**Learned models (historical seed-0 OOF) by physics tier** (physics-hard = all five core physics methods wrong):")
            st.dataframe(pd.DataFrame({k: v["accuracy_by_tier_independent"] for k, v in amb["saved_oof_models_seed0"].items()}).T)
            st.caption("Why learned models win on physics-hard cases is unexplained (motion cues vs provider texture); see the fair protocol controls.")
    with t[3]:
        plan = jload(str(C.PROC / "m11-fair-benchmark-plan-v1" / "plan.json"))
        st.markdown("**Status:** " + (plan["status"] if plan else "plan not written"))
        if plan:
            st.json(dict(partition_validation=plan["partition_validation"], compute_estimate_gpu_h=plan["compute_estimate_gpu_h"]), expanded=False)
    with t[4]:
        man = jload(str(C.PROC / BU.OUT / "bundles_v1.json"))
        if man:
            st.dataframe(pd.DataFrame([dict(model=m, fold=f, checkpoint=v["checkpoint"], sha256=v["checkpoint_sha256"][:16] + "…")
                                       for m, fs in man["bundles"].items() for f, v in fs.items()]), hide_index=True, height=300)
        for nm in ("m11-interaction-catalogue-v1", "m11-phase2-ambiguity-benchmark-v1", "m11-phase2-review-sample-v1", "m11-fair-benchmark-plan-v1",
                   "m11-fair-cpu-pilot-v1", "m11-nxm-synthetic-benchmark-v1"):
            p = C.PROC / nm
            st.markdown(f"- `{nm}`: {'present' if p.exists() else 'not built'}")


# ======================================================================== layout

PAGES = {"Match & Interaction Explorer": explorer, "Identity Recovery Lab": recovery_lab, "Multi-player (N×M)": multiplayer,
         "Review queue": review_queue, "Evidence & provenance": evidence}
st.sidebar.title("⚽ M11 Studio")
page = st.sidebar.radio("Workspace", list(PAGES), key="page")
st.sidebar.caption("Development matches only (32 train/CV). Held-out and official FIFA validation/test matches are refused by every loader.\n\n"
                   "Older apps are unchanged: `lab` (port 8610), `encounters`, `app`.")
PAGES[page]()
