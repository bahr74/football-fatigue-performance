"""Failure-case gallery from executed runs only (PNGs and short MP4 clips), chosen by fixed rules, not by eye.

Selection rule (recorded in index.json): for every category the qualifying benchmark frames / events are listed in a
fixed order and two are drawn with a seeded RNG. Success and failure categories use the same rule, so the gallery
shows typical cases of each kind rather than the best or worst one.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from football_fatigue.video.bench import detectors as BD
from football_fatigue.video.bench import evaluate as BE
from football_fatigue.video.bench import trackers as BT
from football_fatigue.video.bench import visual as V
from football_fatigue.video.bench.frames import load as load_manifest
from football_fatigue.video.sources import SOURCES, ExactFrameReader, play_area_polygon

GALLERY_VERSION = "gallery-v1"
SEED = 7
PRODUCTION = "yolo26l-football__production-cache"


def _poly(data_root: Path, sid: str):
    return np.asarray(play_area_polygon(Path(data_root) / SOURCES[sid]["keypoints"]), dtype=np.float32).reshape(-1, 1, 2)


def _panel(img: np.ndarray, title: str, width: int = 2048) -> np.ndarray:
    s = width / img.shape[1]
    out = cv2.resize(img, (width, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    bar = np.full((28, width, 3), 24, np.uint8)
    cv2.putText(bar, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, V.WHITE, 1, cv2.LINE_AA)
    return np.vstack([bar, out])


def _hstack(imgs: list[np.ndarray]) -> np.ndarray:
    h = max(i.shape[0] for i in imgs)
    return np.hstack([cv2.copyMakeBorder(i, 0, h - i.shape[0], 0, 6, cv2.BORDER_CONSTANT, value=(24, 24, 24)) for i in imgs])


def build_gallery(data_root: Path, out_root: Path, alt: str | None = None) -> Path:
    data_root, out_root = Path(data_root), Path(out_root)
    dest = out_root / "_bench" / GALLERY_VERSION
    if (dest / "index.json").exists():
        raise SystemExit(f"gallery already exists (write-once): {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    man = load_manifest(out_root)
    gt = BE.load_gt(data_root, out_root)
    runs = BD.load_runs(out_root)
    ev = BE.load_eval(out_root)
    if alt is None:      # best non-production detector by AP50 on the check matches (on_pitch), among complete runs
        cands = [k for k, r in ev.items() if runs[k]["detector"] != runs[PRODUCTION]["detector"] and runs[k].get("subset") is None
                 and "check" in r["views"]["on_pitch"]]       # a different model, not the production weights at another size
        alt = max(cands, key=lambda k: ev[k]["views"]["on_pitch"]["check"]["AP50"]) if cands else None
    dets = {k: BD.load_dets(out_root, k) for k in (PRODUCTION, alt) if k}
    readers: dict[str, ExactFrameReader] = {}

    def frame(sid, f):
        if sid not in readers:
            readers.clear()
            readers[sid] = ExactFrameReader(data_root / SOURCES[sid]["video"], cache_frames=2)
        return readers[sid].frame(int(f))

    def split(key, sid, f, conf=0.4):
        rec = dets[key][(sid, f)]
        return V.classify(gt[(sid, f)], BE._view(rec["boxes"], "on_pitch", _poly(data_root, sid)), conf)

    items: list[dict[str, Any]] = []

    def pick(pool: list, n: int = 2) -> list:
        if not pool:
            return []
        idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
        return [pool[i] for i in sorted(idx)]

    def det_png(cat: str, sid: str, f: int, note: str, zoom_gid: int | None = None) -> None:
        img = frame(sid, f)
        panels, crops = [], []
        for key in [k for k in (PRODUCTION, alt) if k]:
            sp = split(key, sid, f)
            drawn = V.draw_detections(img, sp, scale=img.shape[1] / 2048)
            panels.append(_panel(drawn, f"{key}  |  {len(sp['tp'])} matched, {len(sp['fp'])} FP, {len(sp['dup'])} dup, "
                                        f"{len(sp['missed'])} missed of {len(sp['gt'])} GT  (conf > 0.4, on-pitch view)"))
            if zoom_gid is not None:
                g = next((x for x in gt[(sid, f)] if x["gid"] == zoom_gid), None)
                if g:
                    crops.append(V.crop(drawn, g["box"], heights=5.0, out_h=300))
        out = np.vstack(panels + [V.legend_strip(2048, f"{sid[12:]} frame {f}")])
        if crops:
            z = _hstack(crops)
            out = np.vstack([out, cv2.copyMakeBorder(z, 4, 0, 0, max(0, 2048 - z.shape[1]), cv2.BORDER_CONSTANT, value=(24, 24, 24))[:, :2048]])
        name = f"{cat}_{sid[12:]}_f{f}.png"
        cv2.imwrite(str(dest / name), out)
        items.append({"category": cat, "file": name, "title": f"{sid[12:]} frame {f}", "caption": note})

    entries = man["frames"]
    stats = {}
    for e in entries:
        sid, f = e["source_id"], e["frame"]
        if (sid, f) not in dets[PRODUCTION] or (alt and (sid, f) not in dets[alt]):
            continue
        sp, sa = split(PRODUCTION, sid, f), (split(alt, sid, f) if alt else None)
        stats[(sid, f)] = (e, sp, sa)
    keys = sorted(stats)
    r = lambda sp: len(sp["tp"]) / max(len(sp["gt"]), 1)  # noqa: E731
    rules = [
        ("yolo26l_success", lambda e, sp, sa: r(sp) >= 0.95 and len(sp["fp"]) <= 1, "YOLO26-L recall >= 0.95 with <= 1 FP"),
        ("yolo26l_failure", lambda e, sp, sa: len(sp["missed"]) >= 3, "YOLO26-L misses >= 3 GT players"),
        ("alt_success", lambda e, sp, sa: sa is not None and r(sa) >= 0.9, f"{alt} recall >= 0.9"),
        ("alt_failure", lambda e, sp, sa: sa is not None and len(sa["missed"]) >= 4, f"{alt} misses >= 4 GT players"),
        ("small_players", lambda e, sp, sa: "small" in e["strata"], "frame in the small-player stratum"),
        ("heavy_overlap", lambda e, sp, sa: "heavy" in e["strata"], "frame in the heavy-occlusion stratum (annotated)"),
        ("low_light", lambda e, sp, sa: "low_light" in e["strata"], "117092 (low light)"),
        ("goalkeepers", lambda e, sp, sa: "goalkeeper" in e["strata"], "heuristic goalkeeper near another player"),
        ("referee_labels", lambda e, sp, sa: any(p[6] == "referee" for p in sp["fp"] + sp["tp"]),
         "YOLO26-L labelled a box 'referee' (GT has no referees: such boxes count as FP)"),
    ]
    for cat, rule, desc in rules:
        pool = [k for k in keys if rule(*stats[k])]
        for sid, f in pick(pool):
            e, sp, sa = stats[(sid, f)]
            zoom = None
            if cat in ("small_players", "heavy_overlap", "goalkeepers"):
                cand = sp["missed"] + (sa["missed"] if sa else []) or sorted(sp["gt"], key=lambda x: x["box"][3] - x["box"][1])
                zoom = cand[0]["gid"] if cand else None
            det_png(cat, sid, f, f"rule: {desc}; {len(pool)} qualifying frames, 2 drawn (seed {SEED}). Strata: {', '.join(e['strata'])}", zoom)

    # ID switches: T0 production ByteTrack vs the best benchmark tracker run on a tune match
    res = BT.load_results(out_root)
    if res:
        def name(x):
            return f"{x['input']}-{x['tracker']}-b{x['buffer_frames']}"
        tune = [x for x in res if x["role"] == "tune" and x["input"] == "T0"]
        best = max((x for x in tune if x["tracker"] != "sv-bytetrack"), key=lambda x: x["metrics_all_tracks"]["hota"]["HOTA"], default=None)
        for x in [y for y in tune if y["tracker"] == "sv-bytetrack"] + ([best] if best else []):
            sid = x["source_id"]
            from football_fatigue.vision.mot_eval import load_reference

            full_gt = load_reference(data_root / SOURCES[sid]["gt"], 0)[0]
            tr = BT.load_tracks(out_root, sid, name(x), 0, 10**6)
            evs = [e for e in V.id_events(full_gt, tr, 0, max(tr)) if e["kind"] == "id_switch"]
            for e in pick(evs, 1):
                f = e["frame"]
                gb = dict(full_gt[f])[e["gt_id"]]
                strip = []
                for ff in (f - 15, f - 5, f, f + 10):
                    img = frame(sid, max(0, ff)).copy()
                    for t in tr.get(ff, []):
                        col = V.YELLOW if t[0] in (e["from"], e["to"]) else V.GREEN
                        cv2.rectangle(img, (int(t[1]), int(t[2])), (int(t[3]), int(t[4])), col, 2)
                        cv2.putText(img, f"T{t[0]}", (int(t[1]), int(t[2]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
                    c = V.crop(img, gb, heights=6.0, out_h=320)
                    cv2.putText(c, f"frame {ff}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, V.WHITE, 1, cv2.LINE_AA)
                    strip.append(c)
                nm = f"id_switch_{name(x)}_{sid[12:]}_f{f}.png"
                cv2.imwrite(str(dest / nm), _hstack(strip))
                items.append({"category": "id_switches", "file": nm, "title": f"{name(x)} · {sid[12:]} frame {f}",
                              "caption": f"GT player {e['gt_id']}: track T{e['from']} -> T{e['to']} (yellow). Drawn 1 of "
                                         f"{len(evs)} frame-level switches (seed {SEED})."})
                clip = f"id_switch_{name(x)}_{sid[12:]}_f{f}.mp4"
                _clip(frame, tr, sid, f - 40, f + 40, gb, dest / clip, (e["from"], e["to"]))
                items.append({"category": "id_switches", "file": clip, "title": f"clip · {name(x)} · {sid[12:]} frames {f - 40}–{f + 40}",
                              "caption": "annotated clip (mp4v), 25 fps, crop centred on the GT player at the switch"})

    # false occlusion alerts: probable-occlusion events of the T0 engine run with an fp_* GT outcome
    from football_fatigue.video.player.compare import _engine_run

    for sid in ("soccertrack-117093", "soccertrack-118576"):
        run = _engine_run(out_root, sid, "T0")
        if run is None:
            continue
        evp = run.parent.parent / "evaluation" / f"{run.name}.eval-v1.json"
        if not evp.exists():
            continue
        e_eval = json.loads(evp.read_text(encoding="utf-8"))["events"]
        events = {json.loads(line)["event_id"]: json.loads(line) for line in (run / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()}
        bad = [x for x in e_eval if x["event_class"] == "probable_occlusion" and x["outcome"].startswith("fp_")]
        for x in pick(bad, 1):
            evr = events[x["event_id"]]
            f = int(evr.get("peak_frame", evr["start_frame"]))
            img = frame(sid, f).copy()
            row = None
            with (run / "frames.jsonl").open(encoding="utf-8") as fh:
                for line in fh:
                    if x["event_id"] in line:
                        rr = json.loads(line)
                        if rr["event_id"] == x["event_id"] and rr["frame"] == f:
                            row = rr
                            break
            boxes = []
            from football_fatigue.vision.mot_eval import load_reference

            for _gid, gbx in load_reference(data_root / SOURCES[sid]["gt"], 0)[0].get(f, []):
                cv2.rectangle(img, (int(gbx[0]), int(gbx[1])), (int(gbx[2]), int(gbx[3])), V.WHITE, 1)
            for tid, m in ((row or {}).get("boxes") or {}).items():
                bx = m["box"]
                boxes.append(bx)
                col = V.RED if m["state"] == "visible" else V.ORANGE
                cv2.rectangle(img, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), col, 2)
                cv2.putText(img, f"T{tid} {m['state']}", (int(bx[0]), int(bx[1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
            if not boxes:
                continue
            ub = [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]
            nm = f"false_occlusion_{sid[12:]}_f{f}.png"
            cv2.imwrite(str(dest / nm), V.crop(img, ub, heights=3.0, out_h=420))
            items.append({"category": "false_occlusion_alerts", "file": nm, "title": f"{sid[12:]} · event {x['event_id']}",
                          "caption": f"engine T0 probable occlusion, frames {x['start_frame']}–{x['end_frame']}; eval-v1 outcome "
                                     f"{x['outcome']} (GT boxes of the matched players did not form an occlusion episode). "
                                     f"Drawn 1 of {len(bad)} (seed {SEED})."})
    index = {"version": GALLERY_VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
             "production_run": PRODUCTION, "alternative_run": alt,
             "selection_rule": __doc__.strip().splitlines()[0] + " Two items per category drawn with seed "
                               f"{SEED} from all qualifying cases (rules in each caption).", "items": items}
    (dest / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    return dest


def _clip(frame, tracks, sid, a, b, centre_box, path: Path, highlight: tuple[int, int]) -> None:
    w = None
    for f in range(max(0, a), b + 1):
        img = frame(sid, f).copy()
        for t in tracks.get(f, []):
            col = V.YELLOW if t[0] in highlight else V.GREEN
            cv2.rectangle(img, (int(t[1]), int(t[2])), (int(t[3]), int(t[4])), col, 2)
            cv2.putText(img, f"T{t[0]}", (int(t[1]), int(t[2]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
        c = V.crop(img, centre_box, heights=7.0, out_h=360)
        cv2.putText(c, f"frame {f}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, V.WHITE, 1, cv2.LINE_AA)
        if w is None:
            size = (c.shape[1], c.shape[0])
            w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, size)
        w.write(c if (c.shape[1], c.shape[0]) == size else cv2.resize(c, size))
    if w is not None:
        w.release()
