"""Streamlit section "Detector & Tracker Benchmark" (research track; reads saved outputs only, never runs a model).

Detectors without a completed run are listed with their status and are never drawn. Every number shown comes from
``data/processed/video/_bench/`` or the saved engine evaluations.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from football_fatigue.video.bench import detectors as BD
from football_fatigue.video.bench import evaluate as BE
from football_fatigue.video.bench import trackers as BT
from football_fatigue.video.bench import visual as V
from football_fatigue.video.bench.frames import load as load_manifest
from football_fatigue.video.bench.registry import CANDIDATES, as_rows
from football_fatigue.video.sources import SOURCES, ExactFrameReader
from football_fatigue.video.tracking import DESCRIPTIONS, TRACKER_INPUTS, tracks_dir


@st.cache_resource(show_spinner="Loading benchmark ground truth…", max_entries=1)
def _gt(data_root: str, out_root: str, sha: str):
    return BE.load_gt(Path(data_root), Path(out_root))


@st.cache_resource(show_spinner=False, max_entries=12)
def _dets(out_root: str, key: str, mtime: float):
    return BD.load_dets(Path(out_root), key)


@st.cache_resource(show_spinner=False, max_entries=2)
def _reader(path: str):
    from football_fatigue.video.dashboard.data import LockedReader

    return LockedReader(Path(path), cache_frames=4)


@st.cache_resource(show_spinner="Loading full-match ground truth…", max_entries=2)
def _full_gt(data_root: str, sid: str):
    from football_fatigue.vision.mot_eval import load_reference

    return load_reference(Path(data_root) / SOURCES[sid]["gt"], 0)[0]


@st.cache_resource(show_spinner="Loading tracks…", max_entries=6)
def _bench_tracks(out_root: str, sid: str, run: str, lo: int, hi: int):
    return BT.load_tracks(Path(out_root), sid, run, lo, hi)


def _rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _fmt(v, n=3):
    return "–" if v is None else (f"{v:.{n}f}" if isinstance(v, float) else str(v))


# ----------------------------------------------------------------------------------------------- detector comparison
def _detector_comparison(data_root: Path, out_root: Path) -> None:
    try:
        man = load_manifest(out_root)
    except FileNotFoundError:
        st.info("No benchmark frame manifest yet: `python -m football_fatigue.video.bench manifest --data-root …`")
        return
    gt = _gt(str(data_root), str(out_root), man["frames_sha256"])
    runs = BD.load_runs(out_root)
    with_dets = sorted(k for k in runs if (BD.runs_root(out_root) / k / "dets.jsonl").exists())
    no_run = [f"⛔ {r['key']} — {r['status']}" for r in as_rows(data_root.parent, runs)
              if not any(runs[k]["detector"] == r["key"] for k in with_dets)]
    c = st.columns([1.2, 2.2, 1.6, 1.6, 1, 1])
    sources = sorted({e["source_id"] for e in man["frames"]})
    sid = c[0].selectbox("Match", sources, key="bm_src",
                         format_func=lambda s: f"{s[12:]} ({SOURCES[s]['role']})")
    entries = [e for e in man["frames"] if e["source_id"] == sid]
    ent = c[1].selectbox("Benchmark frame", entries, key="bm_frame",
                         format_func=lambda e: f"f{e['frame']} · chosen for {e['selected_for']} · {', '.join(e['strata'])}")
    opts = with_dets + no_run
    default_b = next((i for i, k in enumerate(opts) if k.startswith("yolov8m-coco__tile1280-img1280")), min(1, len(opts) - 1))
    a_key = c[2].selectbox("Detector A", opts, index=next((i for i, k in enumerate(opts) if k.endswith("production-cache")), 0), key="bm_a")
    b_key = c[3].selectbox("Detector B", opts, index=default_b, key="bm_b")
    view = c[4].selectbox("View", ["on_pitch", "all", "no_referee"], key="bm_view",
                          help="on_pitch: drop boxes whose feet are > 0.5 heights outside the annotated pitch; no_referee: also drop "
                               "boxes the model labelled referee (football models only). GT has no referees.")
    conf = c[5].slider("Conf >", 0.1, 0.9, 0.4, 0.05, key="bm_conf")
    st.caption("Same decoded frame for both detectors. Green = matched a GT player (IoU ≥ 0.5) · red = false positive "
               "(may be a referee, coach or spectator: they are not annotated) · magenta = duplicate · orange dashed = "
               "missed GT player · thin white = GT.")
    bgr = _reader(str(data_root / SOURCES[sid]["video"])).frame(int(ent["frame"]))
    g = gt[(sid, int(ent["frame"]))]
    poly = np.asarray(__import__("football_fatigue.video.sources", fromlist=["play_area_polygon"]).play_area_polygon(
        data_root / SOURCES[sid]["keypoints"]), dtype=np.float32).reshape(-1, 1, 2)
    splits = {}
    for label, key in (("A", a_key), ("B", b_key)):
        if key.startswith("⛔"):
            st.warning(f"Detector {label}: {key[2:]}. No detections exist for it, so nothing is drawn.")
            continue
        r = runs[key]
        dets = _dets(str(out_root), key, (BD.runs_root(out_root) / key / "dets.jsonl").stat().st_mtime)
        rec = dets.get((sid, int(ent["frame"])))
        if rec is None:
            st.warning(f"Detector {label} ({key}): this frame was not processed (run status {r['status']}).")
            continue
        preds = BE._view(rec["boxes"], view, poly)
        sp = V.classify(g, preds, conf)
        splits[label] = (key, sp)
        img = V.draw_detections(bgr, sp, scale=bgr.shape[1] / 2048)
        st.markdown(f"**{label}: `{key}`** · {r['device']} · {len(sp['tp'])} matched · {len(sp['fp'])} FP · "
                    f"{len(sp['dup'])} duplicates · {len(sp['missed'])} missed of {len(g)} GT · "
                    f"{rec['seconds']:.2f} s/frame {'(cache read)' if r['device'] == 'cache' else ''}")
        st.image(_rgb(img), width="stretch")
    if splits and g:
        st.markdown("**Small-player and missed-player zooms** (smallest GT players, plus every player missed by either detector)")
        missed_ids = {m["gid"] for _k, sp in splits.values() for m in sp["missed"]}
        pick = sorted(g, key=lambda x: x["box"][3] - x["box"][1])[:3]
        pick += [x for x in g if x["gid"] in missed_ids and x not in pick][:5]
        for x in pick:
            cols = st.columns(len(splits) + 1)
            cols[0].markdown(f"GT {x['gid']} · {x['size']} · annotated {x['vis']}")
            for col, (label, (key, sp)) in zip(cols[1:], splits.items()):
                col.image(_rgb(V.crop(V.draw_detections(bgr, sp, labels=True), x["box"])), caption=f"{label}: {key.split('__')[0]}")


# ----------------------------------------------------------------------------------------------- tracker comparison
def _tracker_comparison(data_root: Path, out_root: Path, player_url: str) -> None:
    res = BT.load_results(out_root)
    sources = [s for s in SOURCES if SOURCES[s].get("role") in ("tune", "check")]
    c = st.columns([1.2, 1.4, 1.4, 1])
    sid = c[0].selectbox("Match", sources, key="bt_src", format_func=lambda s: f"{s[12:]} ({SOURCES[s]['role']})")
    built = [n for n in TRACKER_INPUTS if (tracks_dir(data_root, out_root, sid, n) / "frames.jsonl").is_file()]
    a = c[1].selectbox("Profile A (top)", built, index=built.index("T0") if "T0" in built else 0, key="bt_a",
                       format_func=lambda n: f"{n} · {DESCRIPTIONS[n][:60]}")
    others = [n for n in built if n != a] or built
    b = c[2].selectbox("Profile B (bottom)", others, key="bt_b", format_func=lambda n: f"{n} · {DESCRIPTIONS[n][:60]}")
    start = c[3].number_input("Start frame", 0, 6000, 1500, 25, key="bt_start")
    if player_url:
        url = f"{player_url}/?{urlencode({'source': sid, 'ti': a, 'cmp': b, 'frame': int(start), 'embed': 1})}"
        st.components.v1.iframe(url, height=820, scrolling=True)
        st.markdown(f"[open the synchronized A | B player in its own tab]({url.replace('&embed=1', '')})")
    else:
        st.info("Start the dashboard with `python -m football_fatigue.video.dashboard` to get the synchronized player.")

    st.markdown("##### ID switches and fragment recoveries against GT (benchmark tracker runs, this match)")
    runs = sorted({f"{r['input']}-{r['tracker']}-b{r['buffer_frames']}" for r in res if r["source_id"] == sid})
    if len(runs) < 1:
        st.info("No benchmark tracker runs for this match yet.")
        return
    cc = st.columns([1.5, 1.5, 1])
    ra = cc[0].selectbox("Tracker run A", runs, index=next((i for i, r in enumerate(runs) if r == "T0-sv-bytetrack-b75"), 0), key="bt_ra")
    rb = cc[1].selectbox("Tracker run B", runs, index=min(1, len(runs) - 1), key="bt_rb")
    length = cc[2].selectbox("Window (frames)", [250, 750, 1500], index=1, key="bt_len")
    lo, hi = int(start), int(start) + int(length) - 1
    gt = _full_gt(str(data_root), sid)
    tabs = {}
    for r in (ra, rb):
        tr = _bench_tracks(str(out_root), sid, r, lo, hi)
        ev = V.id_events(gt, {f: [x for x in v] for f, v in tr.items()}, lo, hi)
        tabs[r] = (tr, ev)
    st.dataframe(pd.DataFrame([{"run": r, "id_switches": sum(e["kind"] == "id_switch" for e in ev),
                                "fragments_recovered_same_id": sum(e["kind"] == "fragment_recovered" and e["same_id"] for e in ev),
                                "fragments_recovered_new_id": sum(e["kind"] == "fragment_recovered" and not e["same_id"] for e in ev)}
                               for r, (_t, ev) in tabs.items()]), hide_index=True, width="stretch")
    st.caption("Frame-level Hungarian matching (IoU ≥ 0.5) inside the window: an approximation of CLEAR-MOT IDSW for "
               "locating examples; the full-match TrackEval numbers are in Research results.")
    sw = [dict(e, run=r) for r, (_t, ev) in tabs.items() for e in ev if e["kind"] == "id_switch"]
    if sw:
        df = pd.DataFrame(sw)[["run", "frame", "gt_id", "from", "to"]]
        st.dataframe(df, hide_index=True, width="stretch", height=220)
        pick = st.selectbox("Show an ID switch", range(len(sw)), key="bt_sw",
                            format_func=lambda i: f"{sw[i]['run']} · frame {sw[i]['frame']} · GT {sw[i]['gt_id']}: T{sw[i]['from']} → T{sw[i]['to']}")
        e = sw[pick]
        fr = e["frame"]
        rd = _reader(str(data_root / SOURCES[sid]["video"]))
        gtb = dict(gt.get(fr, []))[e["gt_id"]]
        cols = st.columns(4)
        for col, f in zip(cols, (fr - 10, fr - 1, fr, fr + 10)):
            img = rd.frame(max(0, f)).copy()
            for x1, y1, x2, y2 in [b for _g, b in gt.get(f, [])]:
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), V.WHITE, 1)
            for run, col_c in ((ra, V.GREEN), (rb, V.YELLOW)):
                for t in tabs[run][0].get(f, []):
                    tid, x1, y1, x2, y2 = t[0], *t[1:5]
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), col_c, 2)
                    cv2.putText(img, f"{'A' if run == ra else 'B'}{tid}", (int(x1), int(y1) - 4 if run == ra else int(y2) + 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col_c, 1, cv2.LINE_AA)
            col.image(_rgb(V.crop(img, gtb, heights=6.0, out_h=260)), caption=f"frame {f}")
        st.caption("Green = run A ids, yellow = run B ids, white = GT boxes (centred on the GT player whose id switched).")


# ----------------------------------------------------------------------------------------------- occlusion comparison
def _occlusion_comparison(out_root: Path, player_url: str) -> None:
    from football_fatigue.video.player.compare import _engine

    rows = []
    for sid in [s for s in SOURCES if SOURCES[s].get("role") in ("tune", "check")]:
        for ti in TRACKER_INPUTS:
            e = _engine(out_root, sid, ti)
            if not e or "eval" not in e:
                continue
            ev = e["eval"]
            rows.append({"match": sid[12:], "role": SOURCES[sid]["role"], "profile": ti, "events": e["events_total"],
                         "probable occl.": ev["po_events"], "PO precision": ev["po_precision"], "PO recall": ev["po_recall"],
                         "PO F1": ev["po_f1"], "PO false (fp_*)": ev["po_false"], "PO non-GT person": ev["po_non_gt_person"],
                         "overlap precision": ev["bo_precision"], "overlap recall": ev["bo_recall"],
                         "detector-miss events": e["by_category"].get("detection_loss", 0),
                         "loss-not-occlusion precision": ev["loss_not_occlusion_precision"], "run": e["run_id"]})
    if not rows:
        st.info("No evaluated engine runs.")
        return
    st.markdown("Engine profile v2 (config ff8815ed), eval-v1 against SoccerTrack GT. Only the tracking profile differs "
                "between rows of one match. A box overlap is a *candidate* visual-occlusion signal; precision/recall are "
                "measured against GT-derived occlusion episodes (GT has no visibility labels: coverage is computed from "
                "overlapping GT boxes).")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if player_url:
        st.markdown("Open a run in the player (events, timelines, predicted boxes, GT outcome per event):")
        for r in rows:
            q = urlencode({"source": f"soccertrack-{r['match']}", "run": r["run"]})
            st.markdown(f"- [{r['match']} · {r['profile']}]({player_url}/?{q})")


# ----------------------------------------------------------------------------------------------- research results
def _results(data_root: Path, out_root: Path) -> None:
    runs = BD.load_runs(out_root)
    st.markdown("##### Detector candidates and status")
    st.dataframe(pd.DataFrame([{k: r[k] for k in ("status", "key", "family", "track", "params_m", "gflops_640", "licence",
                                                  "training_data", "eval_overlap_risk", "source")} for r in as_rows(data_root.parent, runs)]),
                 hide_index=True, width="stretch")
    ev = BE.load_eval(out_root)
    if ev:
        st.markdown("##### Detector accuracy on the benchmark frames (view: on_pitch)")
        rows = []
        for k, r in ev.items():
            v = r["views"]["on_pitch"]
            for grp in ("tune", "check", "all"):
                if grp not in v:
                    continue
                g = v[grp]
                p = g["at_production_conf"]
                t = g["at_tune_selected_conf"] or {}
                rows.append({"run": k, "frames": g["frames"], "group": grp, "AP50": g["AP50"], "AP50:95": g["AP50_95"],
                             "AP50 (hidden GT ignored)": g["AP50_ignore_hidden"], "P@0.4": p["precision"], "R@0.4": p["recall"],
                             "FP/frame@0.4": p["fp_per_frame"], "dup/100TP": p["duplicates_per_100_tp"],
                             "tune conf": v.get("tune_selected_conf"), "P@tune": t.get("precision"), "R@tune": t.get("recall"),
                             "F1@tune": t.get("f1"),
                             **{f"R {s}": p["recall_by_size"].get(s, {}).get("recall") for s in ("<35 px", "35-50 px", "50-80 px", ">=80 px")},
                             **{f"R {s}": p["recall_by_visibility"].get(s, {}).get("recall") for s in ("visible", "partial", "heavy", "hidden")},
                             "s/frame": np.mean([x["mean_s_per_frame"] for x in r["latency"].values()]), "device": r["device"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("CPU runs: latency is CPU time on this laptop (8 threads), not GPU. 'production-cache' reads stored GPU outputs.")
    _failure_tables(out_root)
    res = BT.load_results(out_root)
    if res:
        st.markdown("##### Trackers on identical detections (YOLO26-L production cache), full matches")
        st.dataframe(pd.DataFrame([{"match": r["source_id"][12:], "role": r["role"], "input": r["input"], "tracker": r["tracker"],
                                    "buffer": r["buffer_frames"], "HOTA": r["metrics_all_tracks"]["hota"]["HOTA"],
                                    "AssA": r["metrics_all_tracks"]["hota"]["AssA"], "DetA": r["metrics_all_tracks"]["hota"]["DetA"],
                                    "IDF1": r["metrics_all_tracks"]["identity"]["IDF1"], "IDSW": r["metrics_all_tracks"]["clear"]["IDSW"],
                                    "Frag": r["metrics_all_tracks"]["clear"]["Frag"], "track ids": r["track_ids"],
                                    "ms/frame": r["ms_per_frame"], "RSS +MB": r["rss_increase_mb"],
                                    "dropped rows": r["unmeasured_rows_dropped"]} for r in res]),
                     hide_index=True, width="stretch")


def _failure_tables(out_root: Path) -> None:
    agg_p = Path(out_root) / "_diag" / "pipeline-v1" / "aggregate.json"
    st.markdown("##### Failure decomposition (pipeline-v1): why GT players are not tracked")
    if not agg_p.exists():
        st.info("No decomposition yet: `python -m football_fatigue.video.failures decompose --source …`")
        return
    agg = json.loads(agg_p.read_text(encoding="utf-8"))
    rows = []
    for prof, P in agg["profiles"].items():
        for sid, v in P["per_match"].items():
            c = v["codes"]
            rows.append({"profile": prof, "match": sid[12:], "role": v["role"], "GT obs (evaluable)": v["evaluable"],
                         "lost %": v["lost_share_of_evaluable_pct"], "E no box after merge": c.get("miss", 0),
                         "B conf ≤ threshold": c.get("rej_conf", 0), "B turf filter": c.get("rej_turf", 0),
                         "C passed, no track": c.get("untracked", 0), "D ID changes": v["id_changes"]["total"],
                         "episodes": v["episodes"]["total"], "missing players / frame": v["coverage"]["mean_missing_players"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("One outcome per SoccerTrack GT player per frame, in pipeline order. 'No box after merge' cannot be split "
               "into model miss vs merge-NMS suppression: the cache stores post-merge boxes only. Inspect cases in the "
               "player's 🔍 Debugger.")
    eng_p = Path(out_root) / "_diag" / "engine-t0l-v1" / "summary.json"
    if eng_p.exists():
        eng = json.loads(eng_p.read_text(encoding="utf-8"))
        st.markdown("##### B0 vs T0 vs T0L: tracking and occlusion engine v2 (same detections, same ByteTrack)")
        st.dataframe(pd.DataFrame([{"role": role, "profile": prof, **{k: v for k, v in x.items() if k != "matches"}}
                                   for role, P in eng["by_role"].items() for prof, x in P.items()]), hide_index=True, width="stretch")


# ----------------------------------------------------------------------------------------------- failure gallery
def _gallery(out_root: Path) -> None:
    idx = Path(out_root) / "_bench" / "gallery-v1" / "index.json"
    if not idx.exists():
        st.info("No gallery yet: `python -m football_fatigue.video.bench gallery --data-root …`")
        return
    g = json.loads(idx.read_text(encoding="utf-8"))
    st.caption(g.get("selection_rule", ""))
    cats = sorted({i["category"] for i in g["items"]})
    cat = st.selectbox("Category", cats, key="bg_cat")
    for it in [i for i in g["items"] if i["category"] == cat]:
        p = idx.parent / it["file"]
        st.markdown(f"**{it['title']}** — {it['caption']}")
        if p.suffix == ".png":
            st.image(str(p), width="stretch")
        elif p.suffix == ".mp4":
            st.video(str(p))


def render(data_root: Path, out_root: Path, player_url: str = "") -> None:
    st.markdown("#### Detector & Tracker Benchmark (development matches only; research track — production unchanged)")
    if not st.toggle("Load the benchmark section", key="bench_on",
                     help="Off by default: Streamlit runs every tab on each interaction, and this section decodes and draws "
                          "full-resolution frames."):
        st.info("Switch on to compare detectors and trackers on the saved benchmark outputs.")
        return
    t = st.tabs(["Detector comparison", "Tracker comparison", "Occlusion comparison", "Research results", "Failure gallery"])
    with t[0]:
        _detector_comparison(data_root, out_root)
    with t[1]:
        _tracker_comparison(data_root, out_root, player_url)
    with t[2]:
        _occlusion_comparison(out_root, player_url)
    with t[3]:
        _results(data_root, out_root)
    with t[4]:
        _gallery(out_root)
