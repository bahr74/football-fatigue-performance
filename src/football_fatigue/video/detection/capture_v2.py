"""A2-cap v2: stage-resolved raw-tile capture (GPU, gated) + its CPU plan and CPU evaluator (M13).

Why v2. The production checkpoint (YOLO26-L fine-tune, sha 05ee5d6c...) runs with ``end2end=False``: every tile goes
through Ultralytics' class-aware NMS (IoU 0.7, max_det 300) *inside* the model call, and only then through the
cross-tile ``edge_aware`` merge (IoU 0.5). Capture v1 stored the per-tile boxes *after* that NMS, so it could not tell a
proposal removed by per-tile NMS from one never proposed. v2 stores four stages per frame:

1. ``raw``    every candidate anchor of every tile with max class score >= 0.05 (the NMS input), all class scores,
              box in full-frame pixels, tile index, anchor index;
2. ``nms``    which raw candidates survive the per-tile NMS (recomputed with ``return_idxs`` from the exact tensor the
              predictor post-processed, and checked against the predictor's own Results boxes);
3. ``merge_in`` = the per-tile NMS survivors with conf > 0.1 (production cache floor) in full-frame coordinates;
4. ``final``  the production ``edge_aware`` merge replayed offline (``detection.merge.merge('A0')``, Session A's own
              merge code) - compared with the cached boxes of the same frame as a reproduction check.

Only the GPU capture needs approval (``--approved``). ``--smoke`` (<= 2 frames, any device) is a readiness test whose
output goes to a separate ``smoke`` folder and is never used as evidence.

Plan (CPU): :func:`make_plan` picks windows from the saved pipeline-v1 T0 decomposition, two targeted and two seeded
random per match (see ``PLAN_RULES``). Evaluation (CPU): :func:`evaluate` attributes every evaluable GT observation of
the captured frames to one stage, using documented IoU rules (``EVAL_RULES``).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from football_fatigue.video.sources import SOURCES, ExactFrameReader, guard_not_reserved

CAPTURE_VERSION = "capture-v2"
PLAN_VERSION = "a2cap-plan-v1"
EVAL_VERSION = "a2cap-eval-v1"
FLOOR = 0.05                      # raw candidate floor (below the 0.1 cache floor, so near-misses are visible)
CACHE_FLOOR = 0.1
WINDOW = 30
SEED = 13
DEV = ("soccertrack-117093", "soccertrack-118575", "soccertrack-117092", "soccertrack-118576")
PERSON_LABELS = ("player", "referee", "goalkeeper")
PLAN_RULES = {
    "overlap": "targeted: the 30-frame window with most T0 'no box' GT-frames whose episode visibility is heavy/hidden",
    "small_or_clear": "targeted: the window with most T0 'no box' GT-frames of visible/partial players (mostly < 35 px "
                      "or low light) - misses NOT explained by GT overlap",
    "random_a": f"seeded uniform random window (seed {SEED}), not chosen by any failure - unbiased control",
    "random_b": f"second seeded uniform random window (seed {SEED}) - unbiased control",
}
EVAL_RULES = {
    "final_match": "one-to-one Hungarian match of GT boxes to final (merged) person boxes with conf > 0.1, IoU >= 0.5",
    "raw_assignment": "a raw person candidate is assigned to the GT box it overlaps most, if that IoU >= 0.5 "
                      "(many candidates per GT allowed: raw output is dense)",
    "poor_localisation": "no assigned candidate, best raw person candidate IoU in [0.3, 0.5) with the GT box",
    "shared_box_given_to_neighbour": "the only candidates/final boxes with IoU >= 0.5 belong to an overlapping GT "
                                     "neighbour (one box for two players)",
    "tile_nms_suppressor": "the highest-scoring kept candidate of the same tile and class with IoU > 0.7 "
                           "(exact for greedy NMS)",
    "merge_suppressor": "the final box with the highest IoU > 0.5 to the removed candidate (approximate: edge_aware "
                        "also resolves truncated boxes)",
}
CATEGORIES = ("detected_and_passed", "rejected_downstream_conf", "no_raw_proposal", "poor_localisation",
              "removed_by_tile_nms", "removed_by_cross_tile_merge", "below_cache_floor", "shared_box_given_to_neighbour",
              "ambiguous")


# ----------------------------------------------------------------------------------------------- helpers
def _iou_mat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    a, b = np.asarray(a, float)[:, None, :], np.asarray(b, float)[None, :, :]
    ix = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    iy = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = ix * iy
    ua = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1]) + (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1]) - inter
    return np.where(ua > 0, inter / np.where(ua > 0, ua, 1), 0.0)


def _hungarian(iou: np.ndarray, thr: float) -> dict[int, int]:
    from scipy.optimize import linear_sum_assignment

    if not iou.size:
        return {}
    r, c = linear_sum_assignment(-iou)
    return {int(i): int(j) for i, j in zip(r, c) if iou[i, j] >= thr}


# ----------------------------------------------------------------------------------------------- plan (CPU)
def make_plan(out_root: Path, diag_version: str = "pipeline-v1") -> Path:
    from football_fatigue.video.failures.pipeline import CODES, load_episodes

    dest = Path(out_root) / "_diag" / PLAN_VERSION
    if (dest / "plan.json").exists():
        raise SystemExit(f"plan already exists (write-once): {dest}")
    rng = np.random.default_rng(SEED)
    windows, strata = [], {}
    for sid in DEV:
        eps = load_episodes(out_root, sid, "T0")["episodes"]
        diag = Path(out_root) / "_diag" / diag_version / sid / "T0" / "frames.jsonl"
        n = sum(1 for _ in diag.open(encoding="utf-8"))
        w_ov, w_sm = np.zeros(n), np.zeros(n)
        for e in eps:
            if e["cause"] != "miss":
                continue
            tgt = w_ov if e["visibility"] in ("heavy", "hidden") else w_sm
            tgt[e["start"]:e["end"] + 1] += 1
        cs = lambda w: np.convolve(w, np.ones(WINDOW), "valid")  # noqa: E731
        taken: list[tuple[int, int]] = []

        def free(s: int) -> bool:
            return all(s + WINDOW <= a or s >= a + WINDOW for a, _ in taken)

        chosen = {}
        for name, w in (("overlap", w_ov), ("small_or_clear", w_sm)):
            order = np.argsort(-cs(w), kind="stable")
            s = int(next(x for x in order if free(int(x))))
            taken.append((s, WINDOW))
            chosen[name] = s
        for name in ("random_a", "random_b"):
            while True:
                s = int(rng.integers(0, n - WINDOW))
                if free(s):
                    break
            taken.append((s, WINDOW))
            chosen[name] = s
        # what each window contains (from the saved decomposition, T0)
        rows = [json.loads(x) for x in diag.open(encoding="utf-8")]
        for name, s in chosen.items():
            c = Counter(CODES[o[1]] for r in rows[s:s + WINDOW] for o in r["o"])
            ov = int(w_ov[s:s + WINDOW].sum())
            sm = int(w_sm[s:s + WINDOW].sum())
            windows.append({"source_id": sid, "role": SOURCES[sid]["role"], "kind": name, "start": s, "frames": WINDOW,
                            "end": s + WINDOW - 1, "t0_codes": dict(c), "miss_gt_frames_heavy_or_hidden": ov,
                            "miss_gt_frames_visible_or_partial": sm})
        strata[sid] = {"frames_in_match": n}
    total = sum(w["frames"] for w in windows)
    plan = {"version": PLAN_VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED,
            "rules": PLAN_RULES, "windows": windows, "frames_total": total, "strata": strata,
            "note": "targeted windows give conditional attributions for those miss types; only the random windows give "
                    "unbiased match-level shares; report them separately",
            "command": "python -m football_fatigue.video.detection capture2 --approved --data-root "
                       "F:/Football-Fatigue-Project/data --plan data/processed/video/_diag/a2cap-plan-v1/plan.json"}
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "plan.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
    return dest / "plan.json"


# ----------------------------------------------------------------------------------------------- capture (GPU, gated)
class _Grab:
    """Keeps the exact tensors DetectionPredictor.postprocess receives (raw predictions + letterboxed input shape)."""

    def __init__(self):
        self.calls: list[tuple[Any, tuple, list]] = []

    def __enter__(self):
        from ultralytics.models.yolo.detect import predict as P

        self._cls, self._orig = P.DetectionPredictor, P.DetectionPredictor.postprocess
        grab = self

        def hooked(pred_self, preds, img, orig_imgs, **kw):
            p = preds[0] if isinstance(preds, (list, tuple)) else preds
            shapes = [o.shape[:2] for o in orig_imgs] if isinstance(orig_imgs, list) else [orig_imgs.shape[2:]] * len(p)
            grab.calls.append((p.detach().float().cpu(), tuple(img.shape[2:]), shapes,
                               dict(conf=pred_self.args.conf, iou=pred_self.args.iou, agnostic=pred_self.args.agnostic_nms,
                                    max_det=pred_self.args.max_det, classes=pred_self.args.classes,
                                    end2end=bool(getattr(pred_self.model, "end2end", False)))))
            return grab._orig(pred_self, preds, img, orig_imgs, **kw)

        self._cls.postprocess = hooked
        return self

    def __exit__(self, *exc):
        self._cls.postprocess = self._orig


def _tile_records(call, tiles: list, results: list, tile_base: int) -> tuple[list, list, dict]:
    """Raw candidates + NMS keep flags for the tiles of one model call."""
    import torch
    from ultralytics.utils import nms, ops

    pred, img_hw, shapes, args = call
    if args["end2end"]:
        raise RuntimeError("model is end2end: there is no per-tile NMS stage to capture (v2 assumes end2end=False)")
    _out, keepi = nms.non_max_suppression(pred, args["conf"], args["iou"], args["classes"], args["agnostic"],
                                          max_det=args["max_det"], nc=0, end2end=False, return_idxs=True)
    raws, kept_check, stats = [], [], {"nms_equal_results": True}
    nc = pred.shape[1] - 4
    for b in range(pred.shape[0]):
        ti = tile_base + b
        x0, y0 = tiles[ti][0], tiles[ti][1]
        p = pred[b]                                       # (4 + nc, N)
        scores = p[4:4 + nc].T                            # (N, nc)
        cand = torch.nonzero(scores.amax(1) > args["conf"]).view(-1)
        boxes = ops.xywh2xyxy(p[:4, cand].T.clone())
        boxes = ops.scale_boxes(img_hw, boxes, shapes[b])
        keep = set(int(k) for k in keepi[b].view(-1).tolist())
        for j, a in enumerate(cand.tolist()):
            s = scores[a]
            x1, y1, x2, y2 = (float(v) for v in boxes[j])
            raws.append([ti, int(a), int(s.argmax()), round(float(s.max()), 5), [round(float(v), 5) for v in s],
                         round(x1 + x0, 3), round(y1 + y0, 3), round(x2 + x0, 3), round(y2 + y0, 3), int(a in keep)])
        # the predictor's own per-tile output must equal the NMS survivors recomputed here
        res = results[b].boxes
        mine = sorted((round(r[5], 1), round(r[6], 1)) for r in raws if r[0] == ti and r[9])
        theirs = sorted((round(v[0] + x0, 1), round(v[1] + y0, 1)) for v in res.xyxy.tolist()) if res is not None else []
        if len(mine) != len(theirs) or any(abs(a[0] - c[0]) > 0.6 or abs(a[1] - c[1]) > 0.6 for a, c in zip(mine, theirs)):
            stats["nms_equal_results"] = False
        kept_check.append(len(mine))
    return raws, kept_check, stats


def capture(data_root: Path, out_root: Path, plan_path: Path | None, device: str = "cuda:0", smoke: int = 0,
            source_id: str | None = None, start: int | None = None) -> Path:
    from football_fatigue.video.detection.diagnose import cache_dir, iter_cache
    from football_fatigue.video.detection.merge import merge
    from football_fatigue.vision.detector import YOLOPlayerDetector

    if smoke:
        if smoke > 2 or source_id is None or start is None:
            raise SystemExit("--smoke takes <= 2 frames and needs --source and --start")
        windows = [{"source_id": source_id, "start": start, "frames": smoke, "kind": "smoke"}]
        dest_root = Path(out_root) / "_diag" / f"{CAPTURE_VERSION}-smoke"
    else:
        plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
        windows = plan["windows"]
        dest_root = Path(out_root) / "_diag" / CAPTURE_VERSION
    for w in windows:
        spec = SOURCES[w["source_id"]]
        guard_not_reserved(w["source_id"], spec["match_id"], spec["video"], spec["cache_run"])
    dets: dict[str, Any] = {}
    for w in windows:
        sid, s0, n = w["source_id"], int(w["start"]), int(w["frames"])
        dest = dest_root / f"{sid}-{s0}-{n}"
        if dest.exists():
            raise FileExistsError(f"write-once: {dest}")
        spec = SOURCES[sid]
        cm = json.loads((cache_dir(data_root, sid) / "cache_manifest.json").read_text(encoding="utf-8"))
        dc = cm["detector"]
        weights = Path(data_root).parent / dc["model_path"]
        key = dc["weights_sha256"] + device
        if key not in dets:
            if hashlib.sha256(weights.read_bytes()).hexdigest() != dc["weights_sha256"]:
                raise RuntimeError("detector weights differ from the cache's weights")
            det = YOLOPlayerDetector(model_path=str(weights), confidence_threshold=FLOOR, device=device,
                                     image_size=dc["image_size"], tile_width=dc["tile_width"], tile_height=dc["tile_height"],
                                     tile_overlap=dc["tile_overlap"], half=dc["half"], merge_iou=dc["merge_iou"],
                                     tile_batch=dc["tile_batch"], tile_merge=dc["tile_merge"]["policy"],
                                     cross_role_iou=dc["tile_merge"]["cross_role_iou"])
            det.load_model()
            dets[key] = det
        det = dets[key]
        names = dict(det._model.names)
        cached = {fr: [[d.label, round(d.conf, 5), *[round(v, 3) for v in d.box]] for d in ds]
                  for fr, ds in _cache_window(iter_cache, data_root, sid, s0, n)}
        tmp = dest.with_name(dest.name + ".partial")
        tmp.mkdir(parents=True, exist_ok=True)
        reader = ExactFrameReader(Path(data_root) / spec["video"])
        t0 = time.perf_counter()
        repro = Counter()
        nms_ok = True
        try:
            import torch

            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            with gzip.open(tmp / "frames.jsonl.gz", "wt", encoding="utf-8") as fh:
                for fr in range(s0, s0 + n):
                    bgr = reader.frame(fr)
                    h, wd = bgr.shape[:2]
                    tiles = det._tile_grid(wd, h)
                    crops = [np.ascontiguousarray(bgr[y:y + th, x:x + tw]) for x, y, tw, th in tiles]
                    kw = dict(conf=FLOOR, device=det.device, imgsz=det.image_size, half=det.half, verbose=False)
                    raws: list = []
                    with _Grab() as g:
                        results = det._run_tiles(crops, kw)
                    base = 0
                    for call in g.calls:
                        r, _k, st = _tile_records(call, tiles, results[base:base + call[0].shape[0]], base)
                        raws += r
                        nms_ok &= st["nms_equal_results"]
                        base += call[0].shape[0]
                    merge_in = [i for i, r in enumerate(raws) if r[9] and r[3] > CACHE_FLOOR]
                    items = [(raws[i][2], raws[i][3], raws[i][5:9]) for i in merge_in]
                    rects = [tuple(tiles[raws[i][0]]) for i in merge_in]
                    final = merge("A0", items, rects, (wd, h), names, dc["merge_iou"], dc["tile_merge"]["cross_role_iou"])
                    final_idx = [_find(items, f, merge_in) for f in final]
                    rc = _repro(final, cached.get(fr, []), names)
                    repro.update(rc)
                    fh.write(json.dumps({"frame": fr, "size": [wd, h], "tiles": tiles, "raw": raws,
                                         "merge_in": merge_in, "final": final_idx,
                                         "final_boxes": [[int(c), round(float(s), 5), *[round(float(v), 3) for v in b]]
                                                         for c, s, b in final],
                                         "cached": cached.get(fr, []), "repro": rc}, separators=(",", ":")) + "\n")
            vram = round(torch.cuda.max_memory_allocated() / 2**30, 3) if device.startswith("cuda") else None
        finally:
            reader.close()
        secs = time.perf_counter() - t0
        man = {"version": CAPTURE_VERSION, "status": "complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "source_id": sid, "window": w, "device": device, "smoke": bool(smoke), "floor": FLOOR,
               "cache_floor": CACHE_FLOOR, "detector": dc, "class_names": names, "seconds": round(secs, 1),
               "seconds_per_frame": round(secs / n, 2), "peak_vram_gib": vram,
               "checks": {"per_tile_nms_recomputed_equals_predictor_output": nms_ok,
                          "final_vs_cache": dict(repro)},
               "stages": ["raw (pre-NMS candidates, conf >= 0.05)", "nms (per-tile NMS keep flag)",
                          "merge_in (NMS survivors, conf > 0.1)", "final (edge_aware merge replayed, Session A code)"],
               "raw_row": "[tile, anchor, class, conf, class_scores[], x1, y1, x2, y2, kept_by_tile_nms]"}
        (tmp / "manifest.json").write_text(json.dumps(man, indent=1, default=str), encoding="utf-8")
        tmp.rename(dest)
        print(json.dumps({"window": f"{sid}-{s0}-{n}", "s_per_frame": man["seconds_per_frame"], "vram_gib": vram,
                          "nms_ok": nms_ok, "repro": dict(repro)}), flush=True)
    return dest_root


def _cache_window(iter_cache, data_root, sid, s0, n):
    for fr, ds in iter_cache(data_root, sid, s0):
        if fr >= s0 + n:
            break
        yield fr, ds


def _find(items: list, f, idx: list[int]) -> int | None:
    for k, it in enumerate(items):
        if it[0] == f[0] and abs(it[1] - f[1]) < 1e-6 and max(abs(a - b) for a, b in zip(it[2], f[2])) < 1e-3:
            return idx[k]
    return None


def _repro(final: list, cached: list, names: dict) -> dict[str, int]:
    """Final (replayed merge) vs cached boxes of the same frame: matched at IoU > 0.95, or extra on either side."""
    fb = np.array([list(b) for _c, _s, b in final]) if final else np.zeros((0, 4))
    cb = np.array([c[2:6] for c in cached]) if cached else np.zeros((0, 4))
    m = _hungarian(_iou_mat(fb, cb), 0.95)
    return {"matched": len(m), "only_in_capture": len(fb) - len(m), "only_in_cache": len(cb) - len(m)}


# ----------------------------------------------------------------------------------------------- evaluation (CPU)
def attribute_frame(gt: list[tuple[int, list[float]]], rec: dict, names: dict[int, str], profile_conf: float = 0.4,
                    tile_iou: float = 0.7, merge_iou: float = 0.5) -> list[dict[str, Any]]:
    """One category per GT box of one captured frame (rules in ``EVAL_RULES``)."""
    raw = rec["raw"]
    person = [i for i, r in enumerate(raw) if names.get(r[2]) in PERSON_LABELS]
    final = [i for i in rec["final"] if i is not None and names.get(raw[i][2]) in PERSON_LABELS]
    gids = [g for g, _b in gt]
    gb = np.array([b for _g, b in gt], float) if gt else np.zeros((0, 4))
    rb = np.array([raw[i][5:9] for i in person]) if person else np.zeros((0, 4))
    fb = np.array([raw[i][5:9] for i in final]) if final else np.zeros((0, 4))
    m_final = _hungarian(_iou_mat(gb, fb), 0.5)
    iou_raw = _iou_mat(gb, rb)
    owner = {}
    if iou_raw.size:
        best = iou_raw.argmax(0)
        for j in range(len(person)):
            if iou_raw[best[j], j] >= 0.5:
                owner.setdefault(int(best[j]), []).append(person[j])
    final_set = set(final)
    out = []
    for gi, gid in enumerate(gids):
        row: dict[str, Any] = {"gid": gid}
        if gi in m_final:
            r = raw[final[m_final[gi]]]
            row.update(category="detected_and_passed" if r[3] > profile_conf else "rejected_downstream_conf", conf=r[3])
        else:
            mine = owner.get(gi, [])
            if not mine:
                near = float(iou_raw[gi].max()) if iou_raw.size else 0.0
                # a candidate with IoU >= 0.5 that overlaps a neighbour even more: one box for two players
                row["category"] = ("shared_box_given_to_neighbour" if near >= 0.5 else
                                   "poor_localisation" if near >= 0.3 else "no_raw_proposal")
                row["best_raw_iou"] = round(near, 3)
            else:
                kept = [i for i in mine if raw[i][9]]
                if not kept:
                    row["category"] = "removed_by_tile_nms"
                    top = max(mine, key=lambda i: raw[i][3])
                    row["best_removed_conf"] = raw[top][3]
                    row["suppressor_gid"] = _suppressor_gid(top, raw, gb, gids, tile_iou)
                elif not any(i in final_set for i in kept):
                    above = [i for i in kept if raw[i][3] > CACHE_FLOOR]
                    if not above:
                        row["category"] = "below_cache_floor"
                    else:
                        row["category"] = "removed_by_cross_tile_merge"
                        top = max(above, key=lambda i: raw[i][3])
                        row["best_removed_conf"] = raw[top][3]
                        row["suppressor_gid"] = _merge_suppressor_gid(top, raw, final, gb, gids, merge_iou)
                else:
                    row["category"] = "shared_box_given_to_neighbour"
        out.append(row)
    return out


def _gid_of(box, gb, gids):
    if not len(gb):
        return None
    iou = _iou_mat(np.array([box]), gb)[0]
    return int(gids[int(iou.argmax())]) if iou.max() >= 0.5 else None


def _suppressor_gid(i: int, raw: list, gb, gids, thr: float):
    r = raw[i]
    cands = [k for k, q in enumerate(raw) if q[9] and q[0] == r[0] and q[2] == r[2] and q[3] >= r[3]
             and _iou_mat(np.array([q[5:9]]), np.array([r[5:9]]))[0, 0] > thr]
    if not cands:
        return None
    k = max(cands, key=lambda k: raw[k][3])
    return _gid_of(raw[k][5:9], gb, gids)


def _merge_suppressor_gid(i: int, raw: list, final: list, gb, gids, thr: float):
    if not final:
        return None
    iou = _iou_mat(np.array([raw[i][5:9]]), np.array([raw[k][5:9] for k in final]))[0]
    if iou.max() <= thr:
        return None
    return _gid_of(raw[final[int(iou.argmax())]][5:9], gb, gids)


def evaluate(data_root: Path, out_root: Path, capture_root: Path | None = None) -> Path:
    from football_fatigue.vision.mot_eval import load_reference

    capture_root = Path(capture_root or Path(out_root) / "_diag" / CAPTURE_VERSION)
    dest = Path(out_root) / "_diag" / EVAL_VERSION
    if (dest / "summary.json").exists():
        raise SystemExit(f"output already exists (write-once): {dest}")
    rows = []
    gts: dict[str, dict] = {}
    for d in sorted(p for p in capture_root.iterdir() if (p / "manifest.json").exists()):
        man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        sid = man["source_id"]
        spec = SOURCES[sid]
        guard_not_reserved(sid, spec["match_id"], spec["gt"])
        if sid not in gts:
            gts[sid], _ = load_reference(Path(data_root) / spec["gt"], 0)
        names = {int(k): v for k, v in man["class_names"].items()}
        with gzip.open(d / "frames.jsonl.gz", "rt", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                for r in attribute_frame(gts[sid].get(rec["frame"], []), rec, names):
                    rows.append({"source_id": sid, "kind": man["window"].get("kind"), "frame": rec["frame"], **r})
    summ: dict[str, Any] = defaultdict(Counter)
    for r in rows:
        summ[f"{r['source_id']}|{r['kind']}"][r["category"]] += 1
        summ[f"all|{r['kind']}"][r["category"]] += 1
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "observations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (dest / "summary.json").write_text(json.dumps({"version": EVAL_VERSION, "rules": EVAL_RULES, "categories": CATEGORIES,
                                                   "counts": {k: dict(v) for k, v in summ.items()}}, indent=1),
                                       encoding="utf-8")
    return dest
