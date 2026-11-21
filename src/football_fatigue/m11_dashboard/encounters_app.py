"""Streamlit Football Trajectory Explorer: synchronized player-proximity encounters (``m11-sync-proximity-v2``) with their
event-centred 5 s contexts (``m11-sync-proximity-v3``) and audit (``m11-sync-proximity-v3-audit``); development matches only.

Run with ``python -m football_fatigue.m11_dashboard encounters`` (or ``streamlit run`` this file ``-- --root data/raw``).
Data access and figures: :mod:`.encounters`. The only file written is the manual-review CSV (audit notes, never labels).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from football_fatigue.m11_dashboard import encounters as E

CAVEAT = ("**Tracking-derived proximity, not a visually confirmed occlusion.** Both players are compared at exactly the same "
          "timestamp. Player identities are the provider's slots (a reference, not independently verified). ESTIMATED positions "
          "are provider fill-ins, not measurements.")
REVIEW_NOTE = ("Manual review = an audit note on whether this encounter looks plausible and usable. It is **not** an identity label, "
               "not verified occlusion ground truth and not a training target.")
HOW_TO_READ = """
- **Incoming** (solid) and **outgoing** (dotted) paths of Player A (cyan) and Player B (yellow) over the 5 s context; small
  triangles show the direction of travel. Filled dots are observed samples, hollow dots provider estimates.
- **A(t\\*)**, **B(t\\*)** (diamonds): both players at the instant t\\* of their **closest approach**; **P\\*** (star) is the
  midpoint of the two positions (a single point only for an exact synchronized coincidence, d = 0). The dotted circle has radius R.
- Every encounter uses the same **window t\\* − 2.5 s … t\\* + 2.5 s**. The **support bars** show where Player A, Player B and
  both have a valid track; grey = no valid track (a tracking gap or jump, the player not tracked, or outside the match period).
  Nothing is drawn or invented there.
- The **distance chart** shows d(t) with both players taken at the same instant. The slider and the Before (−1 s) / At encounter /
  After (+1 s) buttons move along the fixed axis; at a time without a valid track the player is reported unavailable.
- **Why it matters for identity recovery:** when two tracked players come this close, a tracker can swap their identities.
  Comparing who arrives (incoming) with who leaves (outgoing) at such encounters is the evidence a later identity-recovery
  step can use; reliable encounters (both observed, complete context, no provider cloning) are the ones to trust.
"""
QUALITY = ("observed/observed", "observed/estimated", "estimated/estimated")
QUALITY_WORDS = {"observed/observed": "both observed", "observed/estimated": "mixed", "estimated/estimated": "both estimated"}
READY, HQ = "Presentation-ready (audited)", "High-quality candidates"
PRESETS = {READY: None, HQ: None, "All encounters (incl. incomplete / problematic)": set(QUALITY), "Both observed": {"observed/observed"},
           "Mixed (one estimated)": {"observed/estimated"}, "Both estimated": {"estimated/estimated"}}
READY_HELP = ("Presentation-ready = audited: detection and tracking support re-verified, both players' incoming and outgoing tracks "
              "cover the whole window t* ± 2.5 s, no provider artifact (cloned / duplicated positions, critical tracking flags), both observed at t* with "
              "at least half of their context observed, and both moving at least 1 m. Other options show every event, including "
              "incomplete and problematic ones; nothing is removed from the dataset.")
STATUS_WORDS = {"accept": "Accept: plausible encounter", "reject": "Reject: artifact / unsuitable", "uncertain": "Uncertain"}
# widget states kept when a widget is not drawn (presentation mode hides the advanced filters)
PERSIST = ("mid", "quality", "event", "present", "thr", "rel", "coin", "search", "full")


def _root() -> str:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--root", default=os.environ.get("M11_DASHBOARD_ROOT", "data/raw"))
    a, _ = ap.parse_known_args(sys.argv[1:])
    return str(Path(a.root).resolve())


ROOT = _root()
REVIEWS = E.reviews_path(ROOT)


# ---------------------------------------------------------------- cached loaders (keyed by file mtimes: rewritten outputs reload)

@st.cache_data(show_spinner=False)
def matches(root: str, mtime_key: float) -> list[str]:
    return E.development_matches(root)


@st.cache_data(show_spinner=False)
def overview(root: str, mtime_key: float) -> dict:
    return E.summary(root)


@st.cache_data(show_spinner=False)
def hq(root: str, mtime_key: float) -> set:
    return E.high_quality_ids(root)


@st.cache_data(show_spinner="Loading events…", max_entries=16)
def events(root: str, mid: str, thr: str, mtime_key: float) -> list[dict]:
    return E.load_events(root, mid, thr)


@st.cache_data(show_spinner=False)
def ctx_summary(root: str, mtime_key: float) -> dict:
    p = E.context_dir(root) / "development-index" / "summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@st.cache_data(show_spinner="Loading the audit…")
def validation(root: str, mtime_key: float) -> dict:
    return E.load_validation(root)


@st.cache_resource(show_spinner="Loading trajectories…", max_entries=3)
def npz(root: str, mid: str, mtime_key: float) -> dict:
    return E.load_npz(root, mid)


def stamp(*parts: str) -> float:
    return E.mtime(E.base(ROOT).joinpath(*parts))


def cstamp(*parts: str) -> float:
    return E.mtime(E.context_dir(ROOT).joinpath(*parts))


def game_clock(rec: dict, t: float) -> str:
    s = max(t - rec["period_start_s"], 0.0)
    return f"P{rec['period']} {int(s // 60):02d}:{s % 60:06.3f}"


def show_notes(items: list[tuple[str, str]]) -> None:
    for level, text in items:
        (st.warning if level == "warning" else st.info)(text)


# ---------------------------------------------------------------- callbacks (run before the next script pass)

def step_event(ids: list[str], step: int) -> None:
    st.session_state["event"] = E.neighbour(ids, st.session_state.get("event"), step)


def pick_from_table(ids: list[str], key: str) -> None:
    rows = st.session_state[key].selection.rows
    if rows:
        st.session_state["event"] = ids[rows[0]]


def jump(key: str, t: float) -> None:
    st.session_state[key] = t


# ---------------------------------------------------------------- page

st.set_page_config(page_title="Football Trajectory Explorer", layout="wide")
for k in PERSIST:                                                        # keep hidden widgets' values across reruns
    if k in st.session_state:
        st.session_state[k] = st.session_state[k]
st.title("Football Trajectory Explorer")
st.caption(CAVEAT)
try:
    MATCHES = matches(ROOT, stamp("development-index", "summary.json"))
except Exception as e:                                                   # noqa: BLE001 - shown to the user
    st.error(f"Dataset not found or unreadable under {E.base(ROOT)}: {e}")
    st.stop()
tab_b, tab_a, tab_c = st.tabs(["Trajectory explorer", "Dataset overview", "Review log"])

with tab_a:
    s = overview(ROOT, stamp("development-index", "summary.json"))
    t = s["totals"]
    c = st.columns(5)
    c[0].metric("Development matches", s["matches"])
    c[1].metric("Encounters (R = 0.25 m)", f"{s['episodes']:,}")
    c[2].metric("High-quality candidates", f"{t['subsets']['high_quality_candidates']:,}")
    c[3].metric("Coincident intervals (provider-cloned)", f"{t['coincident_interval']:,}")
    c[4].metric("Complete context (t* ± 2.5 s)", f"{ctx_summary(ROOT, E.mtime(E.context_dir(ROOT) / 'development-index' / 'summary.json')).get('v3_complete', 0):,}")
    left, right = st.columns(2)
    q = t["pair_quality"]
    fig = go.Figure(go.Bar(x=[QUALITY_WORDS[k] for k in QUALITY], y=[q[k] for k in QUALITY], marker_color=["#2e7d32", "#f9a825", "#9e9e9e"],
                           text=[f"{q[k]:,}" for k in QUALITY], textposition="outside"))
    fig.update_layout(title="Quality at t* (status of both players)", height=320, margin=dict(l=10, r=10, t=40, b=10), yaxis_title="encounters")
    left.plotly_chart(fig, width="stretch")
    pm = pd.DataFrame(s["per_match"].values())
    fig = go.Figure()
    for col, name, color in (("observed_observed", "both observed", "#2e7d32"), ("observed_estimated", "mixed", "#f9a825"),
                             ("estimated_estimated", "both estimated", "#9e9e9e")):
        fig.add_bar(x=pm["match_id"].astype(str), y=pm[col], name=name, marker_color=color)
    fig.update_layout(barmode="stack", title="Encounters per match", height=320, margin=dict(l=10, r=10, t=40, b=10),
                      xaxis=dict(type="category", tickfont=dict(size=8)), legend=dict(orientation="h", y=-0.25))
    right.plotly_chart(fig, width="stretch")
    sens = sorted((float(k.split("_", 1)[1].replace("_", ".").rstrip("m")), v) for k, v in s["sensitivity_episodes"].items())
    st.caption(f"Episodes by threshold: {', '.join(f'{r:.2f} m: {v:,}' for r, v in sens)}. "
               f"Source: `{E.index_dir(ROOT)}`.")


def tick(ok: bool) -> str:
    return "✓" if ok else "✗"


def details(mid: str, rec: dict, ctx: E.Context, n: int, total: int, v: dict | None) -> None:
    a, b = rec["player_a"], rec["player_b"]
    cx = rec.get("context")
    lines = [
        "| | |", "|---|---|",
        f"| Player A | {a['side']} #{a['jersey']} · {a['status_at_t_star']} at the encounter |",
        f"| Player B | {b['side']} #{b['jersey']} · {b['status_at_t_star']} at the encounter |",
        f"| Match · event | {mid} · event {n} of {total} |",
        f"| Encounter type | {E.encounter_kind(rec, ctx)} |",
        f"| Minimum synchronized distance | **{rec['d_min_m'] * 100:.1f} cm** at {game_clock(rec, rec['t_star_s'])} |",
        f"| Duration | {rec['duration_s']:.3f} s ({rec['contact_class']}) |",
        f"| Quality | {QUALITY_WORDS[rec['pair_quality']]} |",
        f"| Context t* ± 2.5 s | {E.context_summary(cx) if cx else 'not available for this threshold'} |"]
    if v is not None:
        lines += [f"| Audit | detection {tick(v['mathematically_valid'])} · tracking {tick(v['tracking_support_valid'])} · "
                  f"incoming {tick(v['incoming_complete'])} · outgoing {tick(v['outgoing_complete'])} · "
                  f"artifact {'suspected' if v['provider_artifact_suspected'] else 'none'} |",
                  f"| Presentation-ready | {'**yes**' if v['presentation_ready'] else 'no'} |"]
    st.markdown("\n".join(lines))


def advanced_details(mid: str, rec: dict, ctx: E.Context) -> None:
    a, b = rec["player_a"], rec["player_b"]
    cx = (ctx.record or {}).get("context") if ctx.kind != "none" else rec.get("context")
    with st.expander("Advanced details"):
        flags = sorted(set(a["quality_flags"]) | set(b["quality_flags"]))
        st.markdown("\n".join([
            "| | |", "|---|---|",
            f"| Event ID | `{rec['event_id']}` |",
            f"| Player IDs (provider slots) | A {a['player_id']} · B {b['player_id']} |",
            f"| Teams | {rec['team_relation'].replace('_', ' ')} |",
            f"| Start → end | {game_clock(rec, rec['t_start_s'])} → {game_clock(rec, rec['t_end_s'])} |",
            f"| t* (video time) | {rec['t_star_s']:.3f} s |",
            f"| Provider coincidence | {'yes (cloned positions)' if rec['coincident_interval'] else ('exact touch' if rec['exact_coincidence'] else 'no')} |",
            f"| Critical flags in context | {', '.join(cx['critical_flags']) if cx and cx['critical_flags'] else 'none'} |",
            f"| Provider flags (episode) | {', '.join(flags) if flags else 'none'} |",
            f"| Trajectory source | {ctx.kind}{f' (0.25 m episode `{ctx.source_event_id}`)' if ctx.kind == 'related' else ''} |"]))
        st.json(dict(event_key=rec["event_key"], windows_game_s=rec["windows_game_s"], v2_canonical_window_game_s=rec["canonical_window_game_s"],
                     sample_instants_within=rec["sample_instants_within"], support_cut_at_start=rec["support_cut_at_start"],
                     support_cut_at_end=rec["support_cut_at_end"], parent_event_id=rec.get("parent_event_id"), children=rec.get("children"),
                     segment_a_at_t_star=a["segment_at_t_star"], segment_b_at_t_star=b["segment_at_t_star"],
                     context_source=ctx.source_event_id, context=cx), expanded=False)


def review_form(mid: str, thr: str, eid: str, reviews: dict) -> None:
    st.markdown("**Manual review**")
    st.caption(REVIEW_NOTE)
    cur = reviews.get((str(mid), thr, eid))
    st.write(f"Current: **{STATUS_WORDS[cur['status']]}** ({cur['reviewed_at_utc']})" + (f" — {cur['note']}" if cur and cur["note"] else "")
             if cur else "Current: not reviewed")
    with st.form(f"review-{mid}-{thr}-{eid}", clear_on_submit=False):
        status = st.radio("Status", E.REVIEW_STATUSES, format_func=STATUS_WORDS.get, horizontal=True,
                          index=E.REVIEW_STATUSES.index(cur["status"]) if cur else 2)
        note = st.text_input("Note (optional)", value=cur["note"] if cur else "", max_chars=500)
        if st.form_submit_button("Save review", key="save_review"):
            E.save_review(REVIEWS, mid, thr, eid, status, note)
            st.rerun()


def explorer() -> None:
    top = st.columns([1.0, 1.7, 3.4, 0.55, 0.55, 1.1], vertical_alignment="bottom")
    mid = top[0].selectbox("Match", MATCHES, key="mid")
    present = top[5].toggle("Presentation mode", key="present", help="Audited presentation-ready encounters only (R = 0.25 m).")
    val = validation(ROOT, E.mtime(E.validation_path(ROOT)))
    preset = top[1].selectbox("Encounters", list(PRESETS), key="quality", help=READY_HELP, disabled=present)
    adv = None
    if not present:
        adv = st.expander("Advanced filters and event table")
        c = adv.columns([1.6, 1, 1, 2])
        st.session_state.setdefault("thr", E.PRIMARY)
        c[0].radio("Threshold R", E.THRESHOLDS, horizontal=True, key="thr",
                   format_func=lambda x: f"{float(x[:-1]):.2f} m")
        c[1].selectbox("Teams", ("all", "opposite", "same"), key="rel")
        c[2].selectbox("Provider-cloned intervals", ("all", "exclude", "only"), key="coin")
        c[3].text_input("Search (event ID or jersey, e.g. #7)", key="search")
    ss = st.session_state
    thr, rel, coin, search = ss.get("thr", E.PRIMARY), ss.get("rel", "all"), ss.get("coin", "all"), ss.get("search", "")
    if present:                                                          # presentation: the audited list, never filtered further
        preset, thr, rel, coin, search = READY, E.PRIMARY, "all", "all", ""
    recs = events(ROOT, mid, thr, stamp(mid, f"root.events-{thr}.jsonl") + cstamp(mid, "root.contexts.jsonl"))
    reviews = E.load_reviews(REVIEWS)
    table = E.rows(recs, hq(ROOT, stamp("development-index", "subset_high_quality_candidates.csv")), reviews, thr, validation=val)
    ready_only, hq_only = preset == READY, preset == HQ
    if ready_only and not val:
        st.warning(f"The audit file `{E.validation_path(ROOT)}` is missing: showing high-quality candidates instead. "
                   "Run `python -m football_fatigue.proximity_audit_v3`.")
        ready_only, hq_only = False, True
    if (ready_only or hq_only) and thr != E.PRIMARY:
        st.caption(f"{preset} is defined at R = 0.25 m only; showing all encounters at this threshold.")
        ready_only = hq_only = False
    shown = E.filter_rows(table, qualities=PRESETS[preset] or set(QUALITY), relation=rel, coincident=coin, high_quality_only=hq_only,
                          ready_only=ready_only, search=search)
    if not shown:
        top[2].selectbox("Encounter", ["no encounter matches the filters"], disabled=True)
        st.info("No encounter matches the filters." + (" This match has no presentation-ready encounter." if ready_only else ""))
        return
    by_id = {r["event_id"]: r for r in recs}
    number = {r["event_id"]: k + 1 for k, r in enumerate(recs)}
    ids = [r["event_id"] for r in shown]
    if ss.get("event") not in ids:
        ss["event"] = ids[0]

    def label(eid: str) -> str:
        r = by_id[eid]
        a, b = r["player_a"], r["player_b"]
        return (f"#{number[eid]} · {game_clock(r, r['t_star_s'])} · {a['side']} #{a['jersey']} × {b['side']} #{b['jersey']} · "
                f"{r['d_min_m'] * 100:.1f} cm")

    eid = top[2].selectbox(f"Encounter ({len(ids):,} of {len(table):,} at R = {float(thr[:-1]):.2f} m)", ids, key="event",
                           format_func=label)
    pos = ids.index(eid)
    top[3].button("◀ Prev", on_click=step_event, args=(ids, -1), disabled=pos == 0, key="prev", width="stretch")
    top[4].button("Next ▶", on_click=step_event, args=(ids, 1), disabled=pos == len(ids) - 1, key="next", width="stretch")
    if adv is not None:
        tkey = f"table-{mid}-{thr}"
        adv.dataframe(pd.DataFrame(shown), hide_index=True, height=230, on_select=lambda: pick_from_table(ids, tkey),
                      selection_mode="single-row", key=tkey,
                      column_config=dict(t_star_s=st.column_config.NumberColumn("t* (s)", format="%.3f"),
                                         d_min_cm=st.column_config.NumberColumn("d_min (cm)", format="%.2f")))

    rec = by_id[eid]
    a, b = rec["player_a"], rec["player_b"]
    v = val.get((str(mid), eid)) if thr == E.PRIMARY else None
    ctx = E.context_for(ROOT, mid, thr, rec, npz=npz(ROOT, mid, cstamp(mid, "root.trajectories.npz")),
                        events={k: events(ROOT, mid, k, stamp(mid, f"root.events-{k}.jsonl") + (cstamp(mid, "root.contexts.jsonl") if k == E.PRIMARY else 0))
                                for k in E.THRESHOLDS})
    r_m = float(thr[:-1])
    notes = E.notes(rec, ctx, v)
    st.subheader(f"{E.encounter_kind(rec, ctx)} · A {a['side']} #{a['jersey']} × B {b['side']} #{b['jersey']}")

    if present:
        m = st.columns(5)
        m[0].metric("Event time", game_clock(rec, rec["t_star_s"]))
        m[1].metric("Min. synchronized distance", f"{rec['d_min_m'] * 100:.1f} cm")
        m[2].metric("Quality", QUALITY_WORDS[rec["pair_quality"]])
        cx = rec.get("context")
        m[3].metric("Context t* ± 2.5 s", "—" if cx is None else ("complete" if cx["complete"] else "incomplete"))
        m[4].metric("Encounter", f"{pos + 1} of {len(ids)} ready")
        show_notes([x for x in notes if x[0] == "warning"])              # warnings always stay visible
        if ctx.kind != "none":
            trajectory(mid, thr, eid, rec, ctx, r_m, pitch_h=640, dist_h=250)
        with st.expander("How to read this view"):
            st.markdown(HOW_TO_READ)
        return

    left, right = st.columns([1.6, 1])
    with left:
        if ctx.kind != "none":
            trajectory(mid, thr, eid, rec, ctx, r_m, pitch_h=560, dist_h=270)
        else:
            st.warning(ctx.note)
        with st.expander("How to read this view"):
            st.markdown(HOW_TO_READ)
    with right:
        details(mid, rec, ctx, number[eid], len(recs), v)
        show_notes([x for x in notes if not (ctx.kind == "none" and x[1] == ctx.note)])
        advanced_details(mid, rec, ctx)
        review_form(mid, thr, eid, reviews)


def trajectory(mid: str, thr: str, eid: str, rec: dict, ctx: E.Context, r_m: float, *, pitch_h: int, dist_h: int) -> None:
    """Fixed nominal axis t* -2.5 ... +2.5 s for every event (0.1 s steps plus t* itself); at a selected time each player is shown
    only where the player has a valid track, otherwise it is reported unavailable with the reason (never invented)."""
    ts = rec["t_star_s"]
    key = f"t-{mid}-{thr}-{eid}"
    if key not in st.session_state:
        st.session_state[key] = 0.0
    c = st.columns([1, 1, 1, 1.2])
    for col, phase, text in zip(c, ("before", "at", "after"), (f"◀ Before (t* {-E.PHASE_OFFSET_S:+.1f} s)", "● At encounter",
                                                               f"After (t* {E.PHASE_OFFSET_S:+.1f} s) ▶")):
        col.button(text, on_click=jump, args=(key, E.phase_offset(phase)), key=f"{phase}-{key}", width="stretch")
    full = c[3].toggle("Full pitch", key="full")
    off = st.select_slider("Time relative to the encounter (fixed 5 s window, t* ± 2.5 s)", options=E.NOMINAL_OFFSETS, key=key,
                           format_func=lambda o: f"t* {o:+.1f} s · {game_clock(rec, ts + o)}")
    t_now = ts + off
    now = E.state_at(ctx, t_now)
    parts = [f"**{game_clock(rec, t_now)}** · t* {off:+.1f} s"]
    for k in "ab":
        p = now[k]
        parts.append(f"{k.upper()} ({p[0]:.2f}, {p[1]:.2f}) m" if p is not None else f"{k.upper()} **unavailable** ({E.why_text(now[f'why_{k}'])})")
    parts.append(f"synchronized distance **{now['d'] * 100:.1f} cm**" if now["d"] is not None else "no synchronized distance")
    st.markdown(" · ".join(parts))
    st.plotly_chart(E.pitch_figure(ctx, rec, t_now, r_m=r_m, full_pitch=full, height=pitch_h), width="stretch", key="pitch")
    st.plotly_chart(E.coverage_figure(ctx, rec, t_now), width="stretch", key="coverage")
    st.plotly_chart(E.distance_figure(ctx, rec, t_now, r_m=r_m, height=dist_h), width="stretch", key="dist")


with tab_b:
    explorer()

with tab_c:
    st.caption(REVIEW_NOTE)
    reviews = E.load_reviews(REVIEWS)
    st.write(f"{len(reviews)} reviewed events · file `{REVIEWS}`")
    if reviews:
        st.dataframe(pd.DataFrame([reviews[k] for k in sorted(reviews)]), hide_index=True)
        st.download_button("Export reviews (CSV)", E.reviews_csv(reviews), file_name="m11_encounter_reviews.csv", mime="text/csv")
