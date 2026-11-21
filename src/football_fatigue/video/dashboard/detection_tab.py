"""Detection diagnostics tab: one ground-truth player shown at every pipeline stage, side by side and zoomed.

Read-only; uses the diag-v1 outputs (``python -m football_fatigue.video.detection diagnose``) and, when present, the
tracker-input replay variants (``replay``).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from football_fatigue.video.detection import DIAG_VERSION
from football_fatigue.video.detection.stages import LOST_STAGES, STAGE_TEXT, STAGES, VISIBILITY
from football_fatigue.video.detection.visual import (
    CYAN, GREEN, MAGENTA, ORANGE, RED, WHITE, YELLOW, compose, crop_rect, draw_layer, stage_layers,
)

STAGE_BGR = {"tracked": CYAN, "detected_untracked": GREEN, "below_threshold": ORANGE, "turf_filtered": MAGENTA,
             "poor_overlap": WHITE, "merge_suspect": YELLOW, "no_detection": RED}


def diagnosed_sources(out_root: Path) -> list[str]:
    return sorted(p.parents[2].name for p in Path(out_root).glob(f"*/detection/{DIAG_VERSION}/summary.json"))


@st.cache_resource(show_spinner="Indexing cache, replay and ground truth…", max_entries=2)
def store_cached(data_root: str, out_root: str, source_id: str):
    from football_fatigue.video.detection.framestore import FrameStore

    return FrameStore(Path(data_root), Path(out_root), source_id)


@st.cache_data(show_spinner=False)
def lost_rows(out_root: str, source_id: str) -> pd.DataFrame:
    d = pd.read_csv(Path(out_root) / source_id / "detection" / DIAG_VERSION / "gt_rows.csv.gz")
    return d[d.stage != "tracked"][["frame", "gid", "stage", "visibility", "size", "cover", "det_conf"]]


def stage_table(summary: dict) -> pd.DataFrame:
    sv = summary["stage_visibility"]
    df = pd.DataFrame({v: [sv.get(s, {}).get(v, 0) for s in STAGES] for v in VISIBILITY}, index=list(STAGES))
    df = df.loc[:, df.sum() > 0]
    tot = df.sum()
    return (df / tot * 100).round(2).astype(str) + "%"


def replay_table(out_root: Path, source_id: str) -> pd.DataFrame | None:
    base = Path(out_root) / source_id / "detection" / "replay-v1"
    rows = []
    for p in sorted(base.glob("*/result.json")) if base.exists() else []:
        r = json.loads(p.read_text())
        m, mp = r["metrics_all_tracks"], r["metrics_on_pitch_tracks"]
        rows.append({"variant": r["variant"]["name"], "what": r["variant"]["note"],
                     "HOTA": m["hota"]["HOTA"], "DetRe": m["hota"]["DetRe"], "DetPr": m["hota"]["DetPr"],
                     "IDF1": m["identity"]["IDF1"], "IDSW": m["clear"]["IDSW"],
                     "HOTA on-pitch": mp["hota"]["HOTA"], "DetPr on-pitch": mp["hota"]["DetPr"]})
    return pd.DataFrame(rows) if rows else None


def overview(img: np.ndarray, rows: list[dict], focus: int | None, width: int = 1600) -> np.ndarray:
    scale = width / img.shape[1]
    small = cv2.resize(img, (width, int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    rect = (0, 0, img.shape[1], img.shape[0])
    items = [{"box": r["box"], "colour": STAGE_BGR[r["stage"]], "thick": 3 if r["gid"] == focus else 1,
              "label": f"GT{r['gid']}" if r["stage"] != "tracked" or r["gid"] == focus else None} for r in rows]
    return draw_layer(small, rect, scale, items)


def render(data_root: Path, out_root: Path, reader_cached) -> None:
    sources = diagnosed_sources(out_root)
    if not sources:
        st.info("No detection diagnosis yet. Run `python -m football_fatigue.video.detection diagnose …` first.")
        return
    c1, c2, c3 = st.columns([1.3, 1, 1])
    source = c1.selectbox("Match", sources, key="dx_source")
    fs = store_cached(str(data_root), str(out_root), source)
    variant = c2.selectbox("Tracker output shown", fs.available_variants(), key="dx_variant",
                           help="B0 = production tracker replay; other variants are the tracker-input A/B replays")
    zoom = c3.slider("Crop size (player heights)", 3.0, 12.0, 5.0, 0.5, key="dx_zoom")

    lost = lost_rows(str(out_root), source)
    f1, f2, f3 = st.columns([1.2, 1.2, 1])
    stages = f1.multiselect("Jump filter: stage", list(LOST_STAGES), default=["turf_filtered", "no_detection"],
                            key="dx_stages")
    vis = f2.multiselect("Jump filter: visibility", list(VISIBILITY), default=["visible"], key="dx_vis")
    st.session_state.setdefault("dx_frame", int(lost.frame.iloc[0]) if len(lost) else 0)
    if "dx_frame_req" in st.session_state:          # set by the example picker (before the frame widget exists)
        st.session_state["dx_frame"] = int(st.session_state.pop("dx_frame_req"))
    sel = lost[lost.stage.isin(stages) & lost.visibility.isin(vis)]
    frames = np.unique(sel.frame.to_numpy())

    b = st.columns(7)
    cur = int(st.session_state["dx_frame"])
    if b[0].button("◀◀ prev lost", key="dx_prev", disabled=not len(frames)):
        prior = frames[frames < cur]
        st.session_state["dx_frame"] = int(prior[-1] if len(prior) else frames[-1])
    for col, d, k in ((b[1], -25, "dx_m25"), (b[2], -1, "dx_m1"), (b[4], 1, "dx_p1"), (b[5], 25, "dx_p25")):
        if col.button(f"{d:+d}", key=k):
            st.session_state["dx_frame"] = int(min(max(0, cur + d), fs.last_frame))
    if b[6].button("next lost ▶▶", key="dx_next", disabled=not len(frames)):
        later = frames[frames > cur]
        st.session_state["dx_frame"] = int(later[0] if len(later) else frames[0])
    b[3].number_input("Frame", 0, fs.last_frame, key="dx_frame", label_visibility="collapsed")
    frame = int(st.session_state["dx_frame"])

    ex_path = Path(out_root) / source / "detection" / DIAG_VERSION / "examples" / "index.json"
    if ex_path.exists():
        ex = json.loads(ex_path.read_text())
        labels = ["—"] + [f"{e['stage']} · frame {e['frame']} · GT{e['gid']} ({e['frames']} frames)" for e in ex]
        pick = st.selectbox("Curated example (longest runs per stage)", labels, key="dx_example")
        if pick != "—" and st.session_state.get("dx_example_applied") != pick:
            e = ex[labels.index(pick) - 1]
            st.session_state["dx_example_applied"] = pick
            st.session_state["dx_focus_req"] = e["gid"]
            st.session_state["dx_frame_req"] = e["frame"]
            st.rerun()

    att = fs.attribution(frame, variant)
    rows = att["gt"]
    if not rows:
        st.warning("No ground-truth players in this frame.")
        return
    gids = [r["gid"] for r in rows]
    req = st.session_state.pop("dx_focus_req", None)
    default = req if req in gids else next((r["gid"] for r in rows if r["stage"] != "tracked"
                                            and r["visibility"] == "visible"), gids[0])
    focus = st.selectbox("Focus player", gids, index=gids.index(default), key=f"dx_focus_{frame}",
                         format_func=lambda g: next(f"GT{g} · {r['stage']} · {r['visibility']}" for r in rows
                                                    if r["gid"] == g))
    r = next(x for x in rows if x["gid"] == focus)
    img = reader_cached(str(fs.video)).frame(frame)
    dets = fs.dets(frame)
    layers = stage_layers(fs.gt.get(frame, []), focus, dets, fs.tracks(frame, variant),
                          fs.predicted.get(frame, []), None)
    layers[f"Tracker output ({variant})"] = layers.pop("Tracker output")
    layers["Engine predicted"] = layers.pop("Engine predicted")
    rect = crop_rect(r["box"], fs.size, side_heights=zoom)
    conf = "" if r["det_conf"] is None else f" · matched box conf {r['det_conf']:.2f}, IoU {r['det_iou']:.2f}"
    header = [f"frame {frame} · GT{focus} · {r['visibility']} ({int(r['cover'] * 100)}% covered by nearer players) · "
              f"{r['height']:.0f} px", f"Lost at: {r['stage']} — {STAGE_TEXT[r['stage']]}{conf}"]
    st.image(cv2.cvtColor(compose(img, rect, layers, header, panel=300), cv2.COLOR_BGR2RGB), width="stretch")
    verdict = {"tracked": "OK: tracked", "turf_filtered": "POST-PROCESSING: turf filter removed a detected player",
               "below_threshold": "POST-PROCESSING: confidence threshold removed a detected player",
               "detected_untracked": "TRACKER: detection present, tracker output none",
               "merge_suspect": "POSSIBLE MERGE SUPPRESSION: needs a raw per-tile capture to confirm",
               "no_detection": "DETECTOR: no box near the player after merging (model miss)",
               "poor_overlap": "UNCERTAIN: a box exists but disagrees with the annotation"}[r["stage"]]
    (st.success if r["stage"] == "tracked" else st.warning if r["stage"] != "no_detection" else st.error)(verdict)

    with st.expander("Whole frame (ground truth coloured by stage)", expanded=False):
        st.image(cv2.cvtColor(overview(img, rows, focus), cv2.COLOR_BGR2RGB), width="stretch")
        st.caption("cyan tracked · green detected-untracked · orange below threshold · magenta turf filter · "
                   "yellow merge suspect · red no detection · white poor overlap")
    table = pd.DataFrame([{"GT": x["gid"], "stage": x["stage"], "visibility": x["visibility"],
                           "covered %": int(x["cover"] * 100), "height px": x["height"], "box conf": x["det_conf"],
                           "track": x["track_id"]} for x in rows])
    st.dataframe(table, hide_index=True, width="stretch", height=240)

    s1, s2 = st.columns(2)
    summary = json.loads((Path(out_root) / source / "detection" / DIAG_VERSION / "summary.json").read_text())
    s1.markdown("**Where ground-truth players are lost (whole match, share of each visibility class)**")
    s1.dataframe(stage_table(summary), width="stretch")
    rt = replay_table(out_root, source)
    if rt is not None:
        s2.markdown("**Tracker-input A/B (identical cached detections, CPU replay)**")
        s2.dataframe(rt, hide_index=True, width="stretch")
