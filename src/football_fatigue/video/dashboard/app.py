"""Occlusion Review dashboard (Streamlit). Launch with ``python -m football_fatigue.video.dashboard``.

Reads saved runs only (no detector, no GPU). Frame access is exact by decoded index.

Real-time playback is the browser-native player (``football_fatigue.video.player``: HTML5 video + canvas overlay, synced
per presented frame), embedded in the Viewer tab. The Streamlit part is a frame *inspector* (exact still frames, stepping,
event follow, manual region) and has no rerun-based play loop.

Widget state: the current frame, the selected event and the view mode live in internal session keys (``cur``,
``selected``, ``view_mode``). Widgets that show them (``frame_slider``, ``view_radio``) are synced from those keys
*before* they are created and write back through ``on_change`` callbacks; buttons use ``on_click`` callbacks. Nothing
assigns to a widget key after the widget exists (that raised ``StreamlitAPIException`` in the old play loop).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from football_fatigue.video import overlay as ov
from football_fatigue.video.dashboard import data as D
from football_fatigue.video.dashboard import state as S

st.set_page_config(page_title="Occlusion Review", layout="wide", page_icon="⚽")


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--player-url", default="", help="URL of the native video player (started by the launcher)")
    return ap.parse_args(sys.argv[1:])


ARGS = _args()
DATA_ROOT, OUT_ROOT = Path(ARGS.data_root), Path(ARGS.out_root)

st.markdown("""<style>
.block-container {padding-top: 1.2rem; padding-bottom: 1rem; max-width: 1700px;}
.badge {display:inline-block; padding:2px 10px; border-radius:12px; font-weight:600; font-size:0.85rem; color:#111;}
.muted {color:#8a8f98; font-size:0.85rem;}
div[data-testid="stMetricValue"] {font-size: 1.25rem;}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------------------------- cached loaders
@st.cache_data(show_spinner=False)
def runs_cached(out_root: str):
    return D.discover_runs(Path(out_root))


@st.cache_resource(show_spinner="Loading run…", max_entries=4)
def run_cached(run_dir: str):
    info = next(r for r in D.discover_runs(OUT_ROOT) if str(r.run_dir) == run_dir)
    return D.load_run(info)


@st.cache_resource(show_spinner="Loading tracker output…", max_entries=4)
def tracks_cached(replay: str, lo: int, hi: int):
    return D.load_tracks(Path(replay), lo, hi)


@st.cache_resource(show_spinner="Loading ground truth…", max_entries=4)
def gt_cached(source_id: str):
    return D.load_gt(DATA_ROOT, source_id)


@st.cache_resource(show_spinner="Opening video…", max_entries=2)
def reader_cached(path: str):
    return D.LockedReader(Path(path))


@st.cache_data(show_spinner=False)
def report_cached(out_root: str):
    from football_fatigue.video.occlusion.report import collect

    return collect(Path(out_root))


def rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def badge(cat: str) -> str:
    return f'<span class="badge" style="background:{ov.CATEGORY_HEX[cat]}">{ov.CATEGORY_LABELS[cat]}</span>'


# ---------------------------------------------------------------------------------------------- run selection
runs = runs_cached(str(OUT_ROOT))
st.sidebar.title("Occlusion Review")
st.sidebar.caption("Development matches only · saved runs · no GPU inference")
if not runs:
    st.error(f"No completed runs under {OUT_ROOT}. Create one with `python -m football_fatigue.video.occlusion run …`.")
    st.stop()
sources = sorted({r.source_id for r in runs})
src = st.sidebar.selectbox("Match / source", sources, index=sources.index("soccertrack-117093") if "soccertrack-117093" in sources else 0)
src_runs = [r for r in runs if r.source_id == src]
run_titles = [f"{r.tracker_input} · {r.profile} · frames {r.window[0]}–{r.window[1]} · engine {r.engine_version}"
              + (" · video" if r.rendered else "") + f" · {r.run_id.rsplit('-', 1)[-1]}" for r in src_runs]
choice = st.sidebar.selectbox("Run", run_titles, key=f"run_pick-{src}",
                              help="profile v2 = Step 2 engine; step1 = Step 1 baseline on identical inputs; "
                                   "B0 = production tracker input, T0 = turf filter off (recommended)")
info = src_runs[run_titles.index(choice)]
run = run_cached(str(info.run_dir))
lo, hi = info.window
W, H = run.size
fps = run.fps
labels = S.labels_for(run.events)
by_id = {e["event_id"]: e for e in run.events}
reviews = S.ReviewStore(OUT_ROOT / "reviews" / info.source_id / "reviews.jsonl")
baseline_runs = [r for r in src_runs if r.profile == "step1" and r.window[0] <= lo and r.window[1] >= hi and r.run_id != info.run_id]

# session state per run (internal keys only; widget keys are synced from them before the widgets are created)
if st.session_state.get("run_id") != info.run_id:
    st.session_state.update(run_id=info.run_id, cur=lo, selected=None, view_mode="Full frame")
    for _k in ("_last_tl_pick", "_last_table_pick", "_last_ex_pick", "_last_miss_pick", "goto_frame"):
        st.session_state.pop(_k, None)
if st.session_state.get("selected") not in by_id:
    st.session_state.selected = None
st.session_state.cur = S.clamp(int(st.session_state.get("cur", lo)), lo, hi)

# ---------------------------------------------------------------------------------------------- filters & overlays
with st.sidebar.expander("Event filters", expanded=True):
    uncertain_row = st.toggle("Group uncertain events in their own row", value=True,
                              help="Off: uncertain events stay in their class row (still marked uncertain).")
    cats = st.multiselect("Categories", ov.CATEGORIES, default=["box_overlap", "probable_occlusion", "uncertain"]
                          if len(run.events) > 600 else list(ov.CATEGORIES), format_func=lambda c: ov.CATEGORY_LABELS[c])
    max_d = max((S.duration_frames(e) for e in run.events), default=1)
    dmin, dmax = st.slider("Duration (frames)", 1, max(2, max_d), (1, max(2, max_d)))
    levels = st.multiselect("Evidence level", ["high", "medium", "low"], default=["high", "medium", "low"],
                            help="Engine's own evidence grade (not a calibrated probability).")
    unc = st.radio("Uncertain events", ["all", "only", "exclude"], horizontal=True)
    track_txt = st.text_input("Temporary track ID", "", placeholder="e.g. 107")
filters = S.Filters(tuple(cats), dmin, dmax, tuple(levels), unc, int(track_txt) if track_txt.strip().isdigit() else None, uncertain_row)
shown = S.filter_events(run.events, filters)
shown_ids = {e["event_id"] for e in shown}

with st.sidebar.expander("Overlays", expanded=True):
    o_tracks = st.toggle("Other tracked players", True)
    o_ids = st.toggle("Temporary track IDs (T##)", True)
    o_events = st.toggle("Event boxes", True)
    o_pred = st.toggle("Predicted (hidden) boxes", True)
    o_dim = st.toggle("Dim non-selected events", True)
    o_trails = st.slider("Motion trail (frames)", 0, 50, 15)
    o_gt = st.toggle("Ground-truth boxes (dev only)", False, disabled=gt_cached(info.source_id) is None)
st.sidebar.caption(ov.TRACK_ID_NOTE + ".")

tracks = tracks_cached(str(D.replay_dir(run, DATA_ROOT)), max(0, lo - 60), hi + 60)
reader = reader_cached(str(D.video_path(run, DATA_ROOT)))


def view_options(frame: int, member_tracks: set[int] | None = None) -> ov.ViewOptions:
    trails = {}
    if o_trails:
        want = member_tracks if member_tracks is not None else set(tracks.get(frame, {}))
        for t in want:
            pts = [((tracks[f][t][0][0] + tracks[f][t][0][2]) / 2, tracks[f][t][0][3])
                   for f in range(frame - o_trails, frame + 1) if f in tracks and t in tracks[f]]
            trails[t] = pts
    gtb = []
    if o_gt:
        gtb = [b for _gid, b in (gt_cached(info.source_id) or {}).get(frame, [])]
    return ov.ViewOptions(show_tracks=o_tracks, show_track_ids=o_ids, show_events=o_events, dim_others=o_dim,
                          show_predicted=o_pred, trails=trails, gt_boxes=gtb)


def render(frame: int, rect, out_w: int, selected: str | None, member_tracks: set[int] | None = None) -> np.ndarray | None:
    img = reader.frame(frame)
    if img is None:
        return None
    tr = {t: v[0] for t, v in tracks.get(frame, {}).items()}
    rows = S.active_at(run.rows_by_frame, frame, shown_ids | ({selected} if selected else set()))
    return ov.render_view(img, rect, out_w, tr, rows, by_id, labels, selected, view_options(frame, member_tracks))


def select_event(eid: str, where: str = "peak") -> None:
    """Safe from anywhere (callbacks or script body): only internal keys change; script-body callers then rerun."""
    if eid not in by_id:
        return
    st.session_state.selected = eid
    st.session_state.cur = S.clamp(S.focus_frame(by_id[eid], where), lo, hi)
    st.session_state.view_mode = "Follow selected event"


def go_to(frame: int) -> None:
    st.session_state.cur = S.clamp(int(frame), lo, hi)


def step_by(n: int) -> None:
    go_to(st.session_state.cur + n)


def adjacent(direction: int) -> None:
    e = S.adjacent_event(shown, st.session_state.cur, direction)
    if e:
        select_event(e["event_id"], "start")


def _slider_changed() -> None:
    go_to(st.session_state.frame_slider)


def _goto_changed() -> None:
    if st.session_state.get("goto_frame") is not None:
        go_to(st.session_state.goto_frame)


def _view_changed() -> None:
    st.session_state.view_mode = st.session_state.view_radio


# ---------------------------------------------------------------------------------------------- header
sel = st.session_state.selected
head_l, head_r = st.columns([3, 2])
head_l.markdown(f"### {info.source_id} · profile `{info.profile}`")
head_l.markdown(f'<span class="muted">run {info.run_id} · {len(run.events)} events '
                f'({len(shown)} shown) · {W}×{H} @ {fps:g} fps · '
                f'{"evaluated against SoccerTrack GT" if run.evaluation else "no evaluation"}</span>', unsafe_allow_html=True)
counts = {c: sum(ov.category_of(e, uncertain_row) == c for e in run.events) for c in ov.CATEGORIES}
head_r.markdown(" ".join(f"{badge(c)} {counts[c]}" for c in ov.CATEGORIES), unsafe_allow_html=True)

tab_view, tab_events, tab_results, tab_detect, tab_bench, tab_upload = st.tabs(
    ["🎞 Viewer", "📋 Events", "📊 Results", "🔬 Detection diagnostics", "🏁 Detector & Tracker Benchmark", "⬆ Upload (planned)"])


# ---------------------------------------------------------------------------------------------- viewer
def timeline(frame: int) -> None:
    items = S.timeline_items(shown, fps, labels, uncertain_row)
    fig = go.Figure()
    for cat in ov.CATEGORIES:
        its = [i for i in items if i["category"] == cat]
        if not its:
            continue
        fig.add_trace(go.Bar(
            y=[i["row"] for i in its], x=[max(i["dur_s"], 1 / fps) for i in its], base=[i["start_s"] for i in its],
            orientation="h", marker=dict(color=ov.CATEGORY_HEX[cat],
                                         line=dict(width=[3 if i["event_id"] == st.session_state.selected else 0 for i in its], color="#ffffff")),
            customdata=[i["event_id"] for i in its], name=ov.CATEGORY_LABELS[cat],
            hovertemplate=[f"{i['text']}<br>frames {i['start_frame']}–{i['end_frame']}<extra></extra>" for i in its]))
    t = frame / fps
    fig.add_vline(x=t, line_width=2, line_color="#ffffff")
    win = max(20.0, (hi - lo + 1) / fps) if (hi - lo) < 2000 else 60.0
    x0 = max(lo / fps, t - win / 2)
    fig.update_layout(barmode="overlay", height=230, margin=dict(l=10, r=10, t=10, b=30), showlegend=False,
                      xaxis=dict(title="match clock (s)", range=[x0, x0 + win]),
                      yaxis=dict(categoryorder="array", categoryarray=[ov.CATEGORY_LABELS[c] for c in reversed(ov.CATEGORIES)]),
                      dragmode="pan", clickmode="event+select")
    ev = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="points", key=f"tl-{info.run_id}")
    pts = (ev or {}).get("selection", {}).get("points", []) if isinstance(ev, dict) or hasattr(ev, "get") else []
    if pts:
        eid = pts[0].get("customdata")
        if isinstance(eid, list):
            eid = eid[0]
        if eid in by_id and eid != st.session_state.get("_last_tl_pick"):
            st.session_state._last_tl_pick = eid
            select_event(eid)
            st.rerun()


VIEW_MODES = ["Full frame", "Follow selected event", "Manual region"]

with tab_view:
    if ARGS.player_url:
        from urllib.parse import urlencode

        _q = urlencode({"run": info.run_id, "frame": st.session_state.cur})
        st.markdown(f"**Real-time player** (browser-native video; overlay synced to each presented frame) · "
                    f"[open in its own tab ↗]({ARGS.player_url}/?{_q})")
        with st.expander("Real-time player", expanded=True):
            components.iframe(f"{ARGS.player_url}/?{_q}&embed=1", height=820, scrolling=True)
    else:
        st.info("Real-time player not running. Start the dashboard with `python -m football_fatigue.video.dashboard` "
                "(it starts the player too) or run `python -m football_fatigue.video.player` on its own.")
    st.markdown("#### Frame inspector")
    st.caption("Exact still frames decoded by index. There is no play loop here; use the real-time player for playback.")
    nav = st.columns([1, 1, 1, 1, 1.3, 1.3, 1.2])
    nav[0].button("⏮ −25", width="stretch", on_click=step_by, args=(-25,), key="nav_m25")
    nav[1].button("◀ −1", width="stretch", on_click=step_by, args=(-1,), key="nav_m1")
    nav[2].button("+1 ▶", width="stretch", on_click=step_by, args=(1,), key="nav_p1")
    nav[3].button("+25 ⏭", width="stretch", on_click=step_by, args=(25,), key="nav_p25")
    nav[4].button("◀ prev event", width="stretch", on_click=adjacent, args=(-1,), key="nav_prev_ev")
    nav[5].button("next event ▶", width="stretch", on_click=adjacent, args=(1,), key="nav_next_ev")
    nav[6].number_input("go to frame", lo, hi, value=None, key="goto_frame", label_visibility="collapsed",
                        placeholder="go to frame…", on_change=_goto_changed)

    st.session_state.frame_slider = st.session_state.cur            # sync before the widget exists
    st.slider("Frame", lo, hi, key="frame_slider", label_visibility="collapsed", on_change=_slider_changed)

    zl, zr = st.columns([2, 3])
    if st.session_state.get("view_mode") not in VIEW_MODES:
        st.session_state.view_mode = "Full frame"
    st.session_state.view_radio = st.session_state.view_mode         # sync before the widget exists
    zoom_mode = zl.radio("View", VIEW_MODES, horizontal=True, key="view_radio", on_change=_view_changed)
    if zoom_mode == "Manual region":
        c1, c2, c3 = zr.columns(3)
        mx = c1.slider("centre x (px)", 0, W, W // 2, step=16, key="mx")
        my = c2.slider("centre y (px)", 0, H, H // 2, step=8, key="my")
        mz = c3.slider("zoom", 1.0, 8.0, 3.0, step=0.5, key="mz")

    frame = st.session_state.cur
    selected = st.session_state.selected
    if zoom_mode == "Follow selected event" and selected:
        rect = ov.event_zoom_rect(by_id[selected], run.rows_by_event.get(selected, []), W, H)
        members = {int(m["track_id"]) for m in by_id[selected]["members"]}
    elif zoom_mode == "Manual region":
        rect, members = ov.region_rect(mx, my, mz, W, H), None
    else:
        rect, members = (0, 0, W, H), None
        if zoom_mode == "Follow selected event":
            st.caption("No event selected: showing the full frame.")
    t0 = time.perf_counter()
    img = render(frame, rect, 1600, selected, members)
    ms = (time.perf_counter() - t0) * 1000
    clock = S.frame_to_clock(frame, fps)
    st.markdown(f"**frame {frame}** · clock {S.format_clock(clock)} · container PTS "
                f"{S.frame_to_pts(frame, fps, run.pts_offset):.3f} s"
                f"<span class='muted'> · decoded + drawn in {ms:.0f} ms</span>", unsafe_allow_html=True)
    if img is None:
        st.warning("frame could not be decoded")
    else:
        st.image(rgb(img), width="stretch")
    active = S.active_at(run.rows_by_frame, frame, shown_ids)
    if active:
        st.markdown("Active here: " + " ".join(
            f'{badge(ov.category_of(by_id[r["event_id"]], uncertain_row))} {labels[r["event_id"]]}' for r in active[:12]),
            unsafe_allow_html=True)
    timeline(frame)
    st.caption("Click a bar in the timeline to select an event (pan with drag, zoom with scroll). "
               "Solid boxes = tracker output (each is a measured detection). Dashed boxes = the occlusion engine's "
               "predicted position where nothing was detected.")


# ---------------------------------------------------------------------------------------------- inspector
def inspector(eid: str) -> None:
    e = by_id[eid]
    cat = ov.category_of(e, uncertain_row)
    st.markdown(f"#### {labels[eid]} &nbsp; {badge(cat)} &nbsp; <code>{eid}</code>", unsafe_allow_html=True)
    st.markdown(f"**{e['event_class']} / {e['subtype']}** — {ov.describe(e)}")
    c = st.columns(6)
    c[0].metric("Start", f"f{e['start_frame']}", S.format_clock(e["start_frame"] / fps), delta_color="off")
    c[1].metric("End", f"f{e['end_frame']}", S.format_clock(e["end_frame"] / fps), delta_color="off")
    c[2].metric("Duration", f"{S.duration_frames(e)} fr", f"{S.duration_frames(e) / fps:.2f} s", delta_color="off")
    c[3].metric("Peak", f"f{e['peak_frame']}")
    c[4].metric("Evidence level", e["evidence_level"])
    c[5].metric("Uncertain", "yes" if e["uncertain"] else "no")
    j = st.columns(4)
    if j[0].button("Jump to start", key=f"js-{eid}"):
        select_event(eid, "start"); st.rerun()
    if j[1].button("Jump to peak", key=f"jp-{eid}"):
        select_event(eid, "peak"); st.rerun()
    if j[2].button("Jump to end", key=f"je-{eid}"):
        select_event(eid, "end"); st.rerun()
    if j[3].button("Clear selection", key=f"jc-{eid}"):
        st.session_state.selected = None; st.rerun()

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Why it fired**")
        st.markdown("\n".join(f"- {line}" for line in S.explain(e)))
        st.caption("Evidence level: " + S.LEVEL_TEXT[e["evidence_level"]] + ". It is the engine's own grade, not a calibrated probability.")
        if e["uncertainty_reasons"]:
            st.markdown("**Reasons for doubt**")
            st.markdown("\n".join(f"- {r}" for r in e["uncertainty_reasons"]))
        if e.get("flags"):
            st.markdown("Flags: " + ", ".join(f"`{f}`" for f in e["flags"]))
    with right:
        st.markdown("**Players involved** (temporary track IDs)")
        mem = pd.DataFrame([{"track": f"T{m['track_id']}", "role": m["role"], "frames": f"{m['first_frame']}–{m['last_frame']}",
                             "hidden": ", ".join(f"{a}–{b}" for a, b in m["hidden"]) or "–",
                             "reappeared?": {"reappeared_same_track": "yes, same track", "new_track_at_prediction": "new track at prediction",
                                             "not_reappeared": "no (within 1 s)", "truncated": "window ends", None: "–"}.get(m["outcome"], m["outcome"])}
                            for m in e["members"]])
        st.dataframe(mem, hide_index=True, width="stretch")
        st.markdown("**Measurements**")
        st.dataframe(pd.DataFrame([{"measure": k, "value": v} for k, v in e["measurements"].items()]), hide_index=True,
                     width="stretch", height=210)

    # ground truth
    ge = run.eval_by_event.get(eid)
    st.markdown("**Ground-truth comparison**")
    if ge is None:
        st.info("No evaluation for this run (or the event lies outside the evaluated window).")
    else:
        txt = S.GT_OUTCOME_TEXT.get(ge["outcome"], ge["outcome"])
        mapping = ", ".join(f"T{t} → GT player {g if g is not None else '— (not a GT player)'}" for t, g in ge["member_gt"].items())
        st.markdown(f"Outcome **`{ge['outcome']}`**: {txt}.  \nTrack → GT player: {mapping}.")
        if ge.get("matched_episode"):
            me = ge["matched_episode"]
            st.markdown(f"Matched GT {me['tier']} episode: players {me['pair'][0]} & {me['pair'][1]}, frames {me['start']}–{me['end']} "
                        f"(peak cover {me['peak_cover']:.2f}); start error {ge['start_error']:+d}, end error {ge['end_error']:+d} frames.")
        st.caption("GT = SoccerTrack MOT boxes (22 players, template sizes, no referees/staff): geometric projection occlusion, a proxy for visual occlusion.")

    # snapshots
    st.markdown("**Before / during / after** (fixed crop around the event)")
    rect = ov.event_zoom_rect(e, run.rows_by_event.get(eid, []), W, H)
    snaps = S.snapshot_frames(e, max(0, lo - 50), hi + 30)
    members = {int(m["track_id"]) for m in e["members"]}
    cols = st.columns(5)
    for col, (name, fr) in zip(cols, snaps.items()):
        img = render(fr, rect, 520, eid, members)
        if img is not None:
            col.image(rgb(img), caption=f"{name} · f{fr}", width="stretch")

    # enlarged player crops
    st.markdown("**Enlarged players** (at the peak frame; dashed = predicted position, not detected)")
    peak_rows = [r for r in run.rows_by_event.get(eid, []) if r["frame"] == e["peak_frame"]]
    img = reader.frame(e["peak_frame"])
    if img is not None and peak_rows:
        pcols = st.columns(max(1, len(peak_rows[0]["boxes"])))
        for col, (tid, info_) in zip(pcols, sorted(peak_rows[0]["boxes"].items(), key=lambda kv: int(kv[0]))):
            b = info_["box"]
            bh = b[3] - b[1]
            crop_rect = ov.clamp_rect(b[0] - bh * 0.6, b[1] - bh * 0.35, (b[2] - b[0]) + bh * 1.2, bh * 1.7, W, H)
            v = ov.render_view(img, crop_rect, 260, {}, peak_rows, by_id, labels, eid, ov.ViewOptions(show_tracks=False))
            col.image(rgb(v), caption=f"T{tid} · {info_['state'].replace('_', ' ')}", width="stretch")

    # baseline comparison
    if baseline_runs:
        base = run_cached(str(baseline_runs[0].run_dir))
        bm = S.baseline_matches(e, base.events)
        st.markdown(f"**Step 1 baseline** (same input, run `{baseline_runs[0].run_id}`): {len(bm)} overlapping event(s) sharing a track")
        if bm:
            st.dataframe(pd.DataFrame([{"event": b["event_id"], "class": f"{b['event_class']}/{b['subtype']}",
                                        "frames": f"{b['start_frame']}–{b['end_frame']}", "tracks": len(b["members"])} for b in bm]),
                         hide_index=True, width="stretch")

    # review
    st.markdown("**Your review** (saved separately from model output)")
    latest = reviews.latest(info.run_id).get(eid)
    if latest:
        st.caption(f"Current label: **{latest['label']}** ({latest['created_at_utc'][:19]} UTC){' — ' + latest['note'] if latest['note'] else ''}")
    with st.form(key=f"rev-{eid}", clear_on_submit=True):
        lab = st.radio("Label", S.REVIEW_LABELS, horizontal=True, index=None)
        note = st.text_input("Note (optional)")
        if st.form_submit_button("Save review"):
            if lab is None:
                st.warning("choose a label first")
            else:
                reviews.append(info.run_id, eid, lab, note)
                st.success(f"saved to {reviews.path}")


with tab_view:
    st.divider()
    if st.session_state.selected:
        inspector(st.session_state.selected)
    else:
        st.info("Select an event on the timeline, in the Events tab, or with ‘next event’ to inspect it.")


# ---------------------------------------------------------------------------------------------- events table
with tab_events:
    latest = reviews.latest(info.run_id)
    table = pd.DataFrame([{
        "label": labels[e["event_id"]], "category": ov.CATEGORY_LABELS[ov.category_of(e, uncertain_row)],
        "class/subtype": f"{e['event_class']}/{e['subtype']}", "start": e["start_frame"], "end": e["end_frame"],
        "frames": S.duration_frames(e), "tracks": " ".join(f"T{m['track_id']}" for m in e["members"]),
        "evidence": e["evidence_level"], "uncertain": e["uncertain"],
        "GT outcome": run.eval_by_event.get(e["event_id"], {}).get("outcome", ""),
        "review": latest.get(e["event_id"], {}).get("label", ""), "event_id": e["event_id"]} for e in shown])
    st.caption(f"{len(shown)} events match the filters. Select a row to open it in the Viewer tab.")
    if not table.empty:
        out = st.dataframe(table, hide_index=True, width="stretch", height=520, on_select="rerun",
                           selection_mode="single-row", key=f"evtable-{info.run_id}",
                           column_config={"event_id": st.column_config.TextColumn(width="small")})
        rows_sel = out.selection.rows if hasattr(out, "selection") else []
        if rows_sel:
            eid = table.iloc[rows_sel[0]]["event_id"]
            if eid != st.session_state.get("_last_table_pick"):
                st.session_state._last_table_pick = eid
                select_event(eid)
                st.rerun()
        st.download_button("Download filtered table (CSV)", table.to_csv(index=False).encode(), f"{info.run_id}-events.csv")


# ---------------------------------------------------------------------------------------------- results
with tab_results:
    st.markdown("#### This run")
    c1, c2 = st.columns([2, 3])
    cnt = pd.DataFrame([{"category": ov.CATEGORY_LABELS[c], "events": counts[c]} for c in ov.CATEGORIES])
    c1.dataframe(cnt, hide_index=True, width="stretch")
    if run.manifest.get("stats", {}).get("suppressed_off_pitch_events") is not None:
        c1.caption(f"{run.manifest['stats']['suppressed_off_pitch_events']} events with off-pitch people were suppressed "
                   f"(kept in suppressed.jsonl for review).")
    if run.evaluation:
        s = run.evaluation["summary"]
        rows = []
        for cls, key in (("probable_occlusion", "vs_detectable_occlusion_episodes"), ("box_overlap", "vs_detectable_overlap_episodes")):
            m = s[cls][key]
            rows.append({"class": cls, "events": s[cls]["events"], "precision": m["precision"], "recall": m["recall"], "F1": m["f1"],
                         "|start err| (fr)": s[cls]["timing_frames"].get("start_abs_error_median"),
                         "|end err| (fr)": s[cls]["timing_frames"].get("end_abs_error_median")})
        c2.markdown("Event-level scores vs ground-truth episodes (detectable = both players tracked nearby)")
        c2.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        pc = s.get("pair_claims", {})
        c2.markdown(f"Claim-level precision (every pair an event asserts, independent of grouping): occlusion "
                    f"**{pc.get('occlusion', {}).get('precision')}**, overlap **{pc.get('overlap', {}).get('precision')}**. "
                    f"Occlusion episodes found by any event: **{s['occlusion_episode_recall_any_class']['detectable']}** "
                    f"of {s['occlusion_episode_recall_any_class']['detectable_episodes']}.")
        c2.markdown(f"Detection losses that are not occlusions in GT: **{s['detection_loss_not_occlusion_precision']}**.")
        st.caption(run.evaluation.get("note", ""))

        st.markdown("#### Successes and failures")
        oc = sorted({e["outcome"] for e in run.evaluation["events"]})
        pick = st.selectbox("Outcome", oc, index=oc.index("tp") if "tp" in oc else 0,
                            format_func=lambda o: f"{o} — {S.GT_OUTCOME_TEXT.get(o, '')}")
        ex = [e for e in run.evaluation["events"] if e["outcome"] == pick and e["event_id"] in by_id]
        ext = pd.DataFrame([{"label": labels[e["event_id"]], "class": f"{e['event_class']}/{e['subtype']}", "start": e["start_frame"],
                             "end": e["end_frame"], "GT players": ", ".join(str(v) for v in e["member_gt"].values()),
                             "event_id": e["event_id"]} for e in ex])
        st.caption(f"{len(ex)} events. Select one to inspect it in the Viewer tab.")
        if not ext.empty:
            o2 = st.dataframe(ext, hide_index=True, width="stretch", height=260, on_select="rerun",
                              selection_mode="single-row", key=f"ex-{info.run_id}-{pick}")
            if o2.selection.rows:
                eid = ext.iloc[o2.selection.rows[0]]["event_id"]
                if eid != st.session_state.get("_last_ex_pick"):
                    st.session_state._last_ex_pick = eid
                    select_event(eid); st.rerun()

        st.markdown("#### Missed ground-truth occlusion episodes")
        missed = [m for m in run.evaluation["episodes"] if m["tier"] == "occlusion" and not m["matched_events"]]
        mt = pd.DataFrame([{"GT players": f"{m['pair'][0]} & {m['pair'][1]}", "start": m["start"], "end": m["end"],
                            "peak cover": m["peak_cover"], "cause": m.get("miss_cause")} for m in missed])
        st.caption(f"{len(missed)} missed. Select one to jump there (switch on ‘Ground-truth boxes’ in Overlays).")
        if not mt.empty:
            o3 = st.dataframe(mt, hide_index=True, width="stretch", height=240, on_select="rerun",
                              selection_mode="single-row", key=f"miss-{info.run_id}")
            if o3.selection.rows:
                m = missed[o3.selection.rows[0]]
                if (m["start"], tuple(m["pair"])) != st.session_state.get("_last_miss_pick"):
                    st.session_state._last_miss_pick = (m["start"], tuple(m["pair"]))
                    go_to(m["start"])
                    st.session_state.selected = None
                    st.session_state.view_mode = "Full frame"
                    st.rerun()
    else:
        c2.info("This run has no saved evaluation. Run `python -m football_fatigue.video.occlusion evaluate --run-dir …`.")

    st.markdown("#### Before / after: Step 1 baseline vs v2 (all development matches, identical inputs)")
    rep = report_cached(str(OUT_ROOT))
    if rep:
        cols = ["match", "role", "profile", "events", "po_events", "po_precision", "po_recall", "po_f1", "claim_occ_precision",
                "po_start_abs_err", "po_end_abs_err", "bo_events", "bo_precision", "bo_recall", "claim_ovl_precision",
                "occ_recall_any", "loss_not_occlusion_precision"]
        st.dataframe(pd.DataFrame(rep)[cols], hide_index=True, width="stretch")
        st.caption("tune = 117093, 118575 (thresholds chosen there); check = 117092, 118576 (scored once with frozen settings). "
                   "All four are development data; the check matches are not independent final validation.")
    rv = reviews.load()
    st.markdown(f"#### Review labels: {len(rv)} saved for {info.source_id}")
    if rv:
        st.dataframe(pd.DataFrame(rv)[["created_at_utc", "run_id", "event_id", "label", "note"]], hide_index=True, width="stretch")


# ---------------------------------------------------------------------------------------------- detection diagnostics
with tab_detect:
    from football_fatigue.video.dashboard import detection_tab

    detection_tab.render(DATA_ROOT, OUT_ROOT, reader_cached)


# ---------------------------------------------------------------------------------------------- benchmark (research)
with tab_bench:
    from football_fatigue.video.dashboard import bench_tab

    bench_tab.render(DATA_ROOT, OUT_ROOT, ARGS.player_url)


# ---------------------------------------------------------------------------------------------- upload (design only)
with tab_upload:
    st.markdown("#### Local video upload — planned for the next milestone (not available yet)")
    st.warning("Arbitrary-video analysis is **not supported** in this version. This dashboard only opens saved runs of "
               "the development matches. No detector runs here.")
    st.markdown("""
Planned workflow (design in `docs/video_occlusion_step2.md`):
1. **Upload** an MP4 from this computer → copied to `data/processed/video/uploads/<upload_id>/source.mp4` (original untouched), SHA-256 and probe recorded.
2. **Check**: decodability, frame rate, resolution, exact frame indexing (sequential-decode check), optional clip range.
3. **Detect** (GPU, existing detector + tile settings) → detection cache in the upload folder; progress and ETA shown here.
4. **Track** (existing ByteTrack replay) → `frames.jsonl`.
5. **Occlusion engine** (this engine, chosen profile; no play-area filter unless a pitch outline is drawn) → a normal run folder.
6. **Review** in this dashboard, exactly like the development runs (no ground-truth scores for uploads).
""")
