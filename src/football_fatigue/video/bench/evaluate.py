"""Detector evaluation on the benchmark frames (same frames, same ground truth, same matcher for every model).

Ground truth: SoccerTrack MOT boxes of the 22 players (goalkeepers included, no class labels). Referees, coaches,
substitutes and spectators are NOT annotated, so a correct detection of them counts as a false positive. That is the
same for every model; three views make its size visible:

* ``all``         - every person-like prediction (player / goalkeeper / referee / COCO person)
* ``on_pitch``    - predictions whose foot point is at most 0.5 box heights outside the annotated play area (primary)
* ``no_referee``  - ``on_pitch`` minus predictions the model itself labelled ``referee`` (football models only)

Matching: greedy by descending confidence, IoU >= t, one prediction per GT box. Annotated-visibility classes come from
the GT boxes alone (share of each GT box covered by nearer GT boxes: visible < 0.1 <= partial < 0.5 <= heavy < 0.9 <=
hidden); they describe the annotation, not what a model saw. AP is all-point interpolated (not COCO's 101-point).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from football_fatigue.video.bench.detectors import RUNS_VERSION, load_dets, load_runs, runs_root
from football_fatigue.video.bench.frames import load as load_manifest
from football_fatigue.video.detection.stages import (Params, average_precision, gt_cover, iou_matrix, outside_distance,
                                                     size_bin, visibility)
from football_fatigue.video.sources import SOURCES, play_area_polygon

EVAL_VERSION = "detector-eval-v1"
PERSON = {"player", "goalkeeper", "referee", "person"}
THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))
PRODUCTION_CONF = 0.4
TUNE = ("soccertrack-117093", "soccertrack-118575")
CHECK = ("soccertrack-117092", "soccertrack-118576")


def match_frame(gt: Sequence[Sequence[float]], preds: Sequence[Sequence[float]], thr: float = 0.5
                ) -> tuple[list[int], list[int]]:
    """Greedy matching. ``preds`` rows start with x1, y1, x2, y2, conf. Returns (gt index per pred or -1, pred index per
    gt or -1)."""
    order = sorted(range(len(preds)), key=lambda i: -preds[i][4])
    m = iou_matrix([tuple(g[:4]) for g in gt], [tuple(preds[i][:4]) for i in order]) if gt and preds else np.zeros((len(gt), len(preds)))
    p2g, g2p = [-1] * len(preds), [-1] * len(gt)
    for j, i in enumerate(order):
        best, bi = -1, thr
        for g in range(len(gt)):
            if g2p[g] < 0 and m[g, j] >= bi:
                best, bi = g, m[g, j]
        if best >= 0:
            g2p[best], p2g[i] = i, best
    return p2g, g2p


def _gt_frame(source_id: str, frame: int, gt_rows, poly, size) -> list[dict[str, Any]]:
    p = Params()
    boxes = [b for _g, b in gt_rows]
    cover = gt_cover(boxes, p.cover_grid) if boxes else []
    return [{"gid": gid, "box": list(b), "vis": visibility(b, float(c), outside_distance(b, poly), size, p),
             "size": size_bin(b[3] - b[1])} for (gid, b), c in zip(gt_rows, cover)]


def load_gt(data_root: Path, out_root: Path) -> dict[tuple[str, int], list[dict[str, Any]]]:
    from football_fatigue.vision.mot_eval import load_reference
    from football_fatigue.video.bench.frames import _frame_size

    man = load_manifest(out_root)
    out: dict[tuple[str, int], list] = {}
    by_src = defaultdict(list)
    for e in man["frames"]:
        by_src[e["source_id"]].append(e["frame"])
    for sid, frames in by_src.items():
        spec = SOURCES[sid]
        gt, _ = load_reference(Path(data_root) / spec["gt"], 0)
        poly = np.asarray(play_area_polygon(Path(data_root) / spec["keypoints"]), dtype=np.float32).reshape(-1, 1, 2)
        size = _frame_size(data_root, sid)
        for f in frames:
            out[(sid, f)] = _gt_frame(sid, f, gt.get(f, []), poly, size)
    return out


def _view(boxes: list[list], view: str, poly) -> list[list]:
    keep = [b for b in boxes if b[6] in PERSON]
    if view in ("on_pitch", "no_referee"):
        keep = [b for b in keep if (d := outside_distance(tuple(b[:4]), poly)) is None or d >= -0.5 * (b[3] - b[1])]
    if view == "no_referee":
        keep = [b for b in keep if b[6] != "referee"]
    return keep


def _scores(frames: list[tuple[list[dict], list[list]]], conf: float) -> dict[str, Any]:
    """P/R/F1, FP per frame, duplicates and recall by size / visibility at one confidence threshold (IoU 0.5)."""
    tp = fp = fn = dup = 0
    by_size: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_vis: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for gt, preds in frames:
        ps = [p for p in preds if p[4] > conf]
        p2g, g2p = match_frame([g["box"] for g in gt], ps)
        tp += sum(1 for x in p2g if x >= 0)
        fn += sum(1 for x in g2p if x < 0)
        if ps and gt:
            m = iou_matrix([tuple(g["box"]) for g in gt], [tuple(p[:4]) for p in ps])
            for i, x in enumerate(p2g):
                if x < 0:
                    if m[:, i].max() >= 0.5:
                        dup += 1
                    fp += 1
        else:
            fp += len(ps)
        for g, x in zip(gt, g2p):
            by_size[g["size"]][0] += x >= 0
            by_size[g["size"]][1] += 1
            by_vis[g["vis"]][0] += x >= 0
            by_vis[g["vis"]][1] += 1
    n = max(len(frames), 1)
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"conf": conf, "tp": tp, "fp": fp, "fn": fn, "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4), "fp_per_frame": round(fp / n, 3),
            "duplicates": dup, "duplicates_per_100_tp": round(100 * dup / max(tp, 1), 2),
            "recall_by_size": {k: {"recall": round(v[0] / v[1], 4), "n": v[1]} for k, v in sorted(by_size.items())},
            "recall_by_visibility": {k: {"recall": round(v[0] / v[1], 4), "n": v[1]} for k, v in sorted(by_vis.items())}}


def _ap(frames: list[tuple[list[dict], list[list]]], thr: float, ignore_hidden: bool = False) -> float:
    scored: list[tuple[float, bool]] = []
    n_gt = 0
    for gt, preds in frames:
        keep = [g for g in gt if not (ignore_hidden and g["vis"] == "hidden")]
        ign = [g for g in gt if ignore_hidden and g["vis"] == "hidden"]
        n_gt += len(keep)
        p2g, _ = match_frame([g["box"] for g in keep], preds, thr)
        rest = [p for p, x in zip(preds, p2g) if x < 0]
        if ign and rest:
            m = iou_matrix([tuple(g["box"]) for g in ign], [tuple(p[:4]) for p in rest])
            ignored = {id(p) for j, p in enumerate(rest) if m[:, j].max() >= thr}
        else:
            ignored = set()
        scored += [(p[4], x >= 0) for p, x in zip(preds, p2g) if id(p) not in ignored]
    return average_precision(scored, n_gt)


def _best_f1_conf(frames, grid=tuple(round(0.1 + 0.05 * i, 2) for i in range(17))) -> float:
    return max(grid, key=lambda c: (_scores(frames, c)["f1"], -c))


def evaluate_run(data_root: Path, out_root: Path, key: str, gt=None, manifest=None) -> dict[str, Any]:
    runs = load_runs(out_root)
    run = runs[key]
    dets = load_dets(out_root, key)
    gt = gt or load_gt(data_root, out_root)
    manifest = manifest or load_manifest(out_root)
    polys = {sid: np.asarray(play_area_polygon(Path(data_root) / SOURCES[sid]["keypoints"]), dtype=np.float32).reshape(-1, 1, 2)
             for sid in {e["source_id"] for e in manifest["frames"]}}
    entries = [e for e in manifest["frames"] if (e["source_id"], e["frame"]) in dets]
    out: dict[str, Any] = {"eval_version": EVAL_VERSION, "run": key, "detector": run["detector"], "protocol": run["protocol"],
                           "device": run["device"], "frames_evaluated": len(entries),
                           "frames_in_manifest": len(manifest["frames"]), "views": {}}
    has_ref = any(b[6] == "referee" for r in dets.values() for b in r["boxes"])
    for view in ("all", "on_pitch") + (("no_referee",) if has_ref else ()):
        def frames_of(sel):
            return [(gt[(e["source_id"], e["frame"])], _view(dets[(e["source_id"], e["frame"])]["boxes"], view, polys[e["source_id"]]))
                    for e in entries if sel(e)]
        groups = {"all": lambda e: True, "tune": lambda e: e["source_id"] in TUNE, "check": lambda e: e["source_id"] in CHECK}
        groups |= {sid: (lambda s: lambda e: e["source_id"] == s)(sid) for sid in sorted(polys)}
        tune_conf = _best_f1_conf(frames_of(groups["tune"])) if any(groups["tune"](e) for e in entries) else None
        res = {"tune_selected_conf": tune_conf}
        for name, sel in groups.items():
            fr = frames_of(sel)
            if not fr:
                continue
            aps = [_ap(fr, t) for t in THRESHOLDS]
            res[name] = {"frames": len(fr), "AP50": round(aps[0], 4), "AP50_95": round(float(np.mean(aps)), 4),
                         "AP75": round(aps[5], 4), "AP50_ignore_hidden": round(_ap(fr, 0.5, True), 4),
                         "at_production_conf": _scores(fr, PRODUCTION_CONF),
                         "at_tune_selected_conf": _scores(fr, tune_conf) if tune_conf is not None else None}
        strata = defaultdict(list)
        for e in entries:
            for s in e["strata"]:
                strata[s].append(e)
        res["by_stratum"] = {}
        for s, es in sorted(strata.items()):
            fr = [(gt[(e["source_id"], e["frame"])], _view(dets[(e["source_id"], e["frame"])]["boxes"], view, polys[e["source_id"]]))
                  for e in es]
            sc = _scores(fr, PRODUCTION_CONF)
            res["by_stratum"][s] = {"frames": len(fr), "AP50": round(_ap(fr, 0.5), 4), "recall": sc["recall"],
                                    "precision": sc["precision"], "fp_per_frame": sc["fp_per_frame"]}
        out["views"][view] = res
    secs = defaultdict(list)
    for e in entries:
        secs[e["source_id"]].append(dets[(e["source_id"], e["frame"])]["seconds"])
    out["latency"] = {sid: {"mean_s_per_frame": round(float(np.mean(v)), 3), "median_s_per_frame": round(float(np.median(v)), 3),
                            "fps": round(1 / max(float(np.mean(v)), 1e-9), 3), "frames": len(v)} for sid, v in sorted(secs.items())}
    out["latency_note"] = ("cache runs read stored boxes: latency is the read time, not inference; production GPU speed "
                           "was 1.77 s/frame (GTX 1650 Ti)") if run["device"] == "cache" else f"measured on {run['device']}"
    out["hardware"] = run.get("hardware")
    out["peak_rss_mb"] = run.get("peak_rss_mb")
    out["peak_vram_mb"] = run.get("peak_vram_mb")
    return out


def evaluate_all(data_root: Path, out_root: Path) -> Path:
    runs = load_runs(out_root)
    gt = load_gt(data_root, out_root)
    manifest = load_manifest(out_root)
    dest = Path(out_root) / "_bench" / RUNS_VERSION / "eval"
    dest.mkdir(parents=True, exist_ok=True)
    summary = {}
    for key, run in sorted(runs.items()):
        if run.get("status") != "complete":
            summary[key] = {"status": run.get("status"), "skipped": "run not complete"}
            continue
        res = evaluate_run(data_root, out_root, key, gt, manifest)
        (dest / f"{key}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        summary[key] = {"status": "evaluated", "frames": res["frames_evaluated"], "subset": run.get("subset")}
    (dest / "summary.json").write_text(json.dumps({"eval_version": EVAL_VERSION, "runs": summary,
                                                   "frames_sha256": manifest["frames_sha256"]}, indent=1), encoding="utf-8")
    return dest


def load_eval(out_root: Path) -> dict[str, dict[str, Any]]:
    d = Path(out_root) / "_bench" / RUNS_VERSION / "eval"
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json")) if p.stem != "summary"} if d.exists() else {}


__all__ = ["match_frame", "evaluate_run", "evaluate_all", "load_eval", "load_gt", "runs_root"]


def paired_bootstrap(data_root: Path, out_root: Path, key_a: str, key_b: str, n: int = 1000, seed: int = 0,
                     view: str = "on_pitch", sources: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Frame-level paired bootstrap of AP50 and recall@0.4 differences (B - A) on the frames both runs processed.
    Frames from one match are correlated, so these intervals are optimistic; they show whether a gap is within noise."""
    man = load_manifest(out_root)
    gt = load_gt(data_root, out_root)
    da, db = load_dets(out_root, key_a), load_dets(out_root, key_b)
    polys = {sid: np.asarray(play_area_polygon(Path(data_root) / SOURCES[sid]["keypoints"]), dtype=np.float32).reshape(-1, 1, 2)
             for sid in {e["source_id"] for e in man["frames"]}}
    keys = [(e["source_id"], e["frame"]) for e in man["frames"] if (e["source_id"], e["frame"]) in da
            and (e["source_id"], e["frame"]) in db and (sources is None or e["source_id"] in sources)]
    fa = [(gt[k], _view(da[k]["boxes"], view, polys[k[0]])) for k in keys]
    fb = [(gt[k], _view(db[k]["boxes"], view, polys[k[0]])) for k in keys]
    rng = np.random.default_rng(seed)
    d_ap, d_rec = [], []
    for _ in range(n):
        idx = rng.integers(0, len(keys), len(keys))
        A, B = [fa[i] for i in idx], [fb[i] for i in idx]
        d_ap.append(_ap(B, 0.5) - _ap(A, 0.5))
        d_rec.append(_scores(B, PRODUCTION_CONF)["recall"] - _scores(A, PRODUCTION_CONF)["recall"])
    q = lambda v: [round(float(np.quantile(v, 0.025)), 4), round(float(np.quantile(v, 0.975)), 4)]  # noqa: E731
    return {"a": key_a, "b": key_b, "frames": len(keys), "view": view, "resamples": n,
            "ap50_diff": round(_ap(fb, 0.5) - _ap(fa, 0.5), 4), "ap50_diff_ci95": q(d_ap),
            "recall_diff": round(_scores(fb, PRODUCTION_CONF)["recall"] - _scores(fa, PRODUCTION_CONF)["recall"], 4),
            "recall_diff_ci95": q(d_rec)}
