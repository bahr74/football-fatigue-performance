"""Standard MOT metrics (HOTA, DetA, AssA, CLEAR, Identity/IDF1) for saved CV runs.

This is an adapter around TrackEval's metric classes. It does not re-implement a metric:
reference and predicted boxes are converted to the per-frame id / IoU-similarity structure that
TrackEval's ``HOTA``, ``CLEAR`` and ``Identity`` consume, and TrackEval's own sequence
combination produces the final numbers. ``tracking_eval.py`` is the separate, approximate
project evaluator (greedy per-frame Hungarian at one IoU; no HOTA). Its IDF1 is reported under
``approximate_project_metrics`` and must not be read as the standard IDF1 below.

Conventions
* Frames are 0-based decoded-frame indices on both sides. ``gt_frame_base`` converts a reference
  file that numbers frames from 1. The evaluation window is ``[start, end)``; every frame in it is
  a timestep, including frames with no reference boxes or no prediction record.
* Reference rows are ``frame,id,x,y,w,h[,conf,class,...]`` (MOTChallenge). Rows whose conf field
  is exactly 0 are "ignore" rows in MOTChallenge and are dropped; ``-1`` means unspecified and is
  kept. The class column is not used: the reference is treated as one class of tracked people.
* Predictions come from ``frames.jsonl`` (``players[].bbox_xyxy_px``, ``track_id``,
  ``detected_class``, ``confidence``). Class filtering and a strict ``confidence > min_confidence``
  filter are explicit options. Records without a box, and boxes that are non-finite or have
  non-positive width or height, are dropped and counted. A repeated track id within one frame
  is an error, not silently resolved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from football_fatigue.vision.tracking_eval import _iou_matrix

REPORT_SCHEMA_VERSION = 1
ADAPTER_VERSION = "1"
HOTA_ALPHA_INDEX_05 = 9  # TrackEval alphas are 0.05, 0.10 ... 0.95

CLASS_VIEWS: dict[str, frozenset[str] | None] = {
    "all_non_ball": None,  # everything except the ball
    "player_goalkeeper": frozenset({"player", "goalkeeper"}),
}


class MotEvalError(ValueError):
    """Raised for input that cannot be evaluated unambiguously."""


Box = tuple[float, float, float, float]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_box(box: Iterable[float]) -> Box | None:
    values = tuple(float(v) for v in box)
    if len(values) != 4 or not all(np.isfinite(values)) or values[2] <= values[0] or values[3] <= values[1]:
        return None
    return values  # type: ignore[return-value]


def load_reference(path: Path, frame_base: int = 0) -> tuple[dict[int, list[tuple[int, Box]]], dict[str, int]]:
    """Read MOTChallenge reference rows into ``{0-based frame: [(id, xyxy)]}``."""
    frames: dict[int, list[tuple[int, Box]]] = {}
    seen: set[tuple[int, int]] = set()
    stats = {"rows": 0, "ignored_conf_zero": 0, "invalid_boxes": 0}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = line.split(",")
        if len(row) < 6:
            raise MotEvalError(f"{path}:{number}: expected at least 6 comma-separated fields")
        stats["rows"] += 1
        if len(row) > 6 and row[6].strip() and float(row[6]) == 0:
            stats["ignored_conf_zero"] += 1
            continue
        frame, ident = int(float(row[0])) - frame_base, int(float(row[1]))
        if frame < 0:
            raise MotEvalError(f"{path}:{number}: frame {row[0]} is below gt_frame_base={frame_base}")
        x, y, w, h = (float(v) for v in row[2:6])
        box = _valid_box((x, y, x + w, y + h))
        if box is None:
            stats["invalid_boxes"] += 1
            continue
        if (frame, ident) in seen:
            raise MotEvalError(f"{path}:{number}: reference id {ident} appears twice in frame {frame}")
        seen.add((frame, ident))
        frames.setdefault(frame, []).append((ident, box))
    return frames, stats


def load_predictions(
    frames_path: Path,
    start: int,
    end: int | None,
    view: str = "all_non_ball",
    min_confidence: float | None = None,
) -> tuple[dict[int, list[tuple[int, Box, float]]], dict[str, Any]]:
    """Read ``frames.jsonl`` predictions inside ``[start, end)`` as ``{frame: [(id, xyxy, conf)]}``."""
    if view not in CLASS_VIEWS:
        raise MotEvalError(f"unknown class view {view!r}; expected one of {sorted(CLASS_VIEWS)}")
    allowed = CLASS_VIEWS[view]
    frames: dict[int, list[tuple[int, Box, float]]] = {}
    stats = {
        "records_in_window": 0, "predictions": 0, "dropped_ball": 0, "dropped_other_class": 0,
        "dropped_no_box": 0, "dropped_invalid_box": 0, "dropped_below_confidence": 0,
    }
    seen_frames: set[int] = set()
    with frames_path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            frame = int(record["frame_num"])
            if frame < start or (end is not None and frame >= end):
                continue
            if frame in seen_frames:
                raise MotEvalError(f"{frames_path}: frame {frame} appears twice")
            seen_frames.add(frame)
            stats["records_in_window"] += 1
            ids_in_frame: set[int] = set()
            for player in record["players"]:
                label = player.get("detected_class")
                if label == "ball":
                    stats["dropped_ball"] += 1
                    continue
                if allowed is not None and label not in allowed:
                    stats["dropped_other_class"] += 1
                    continue
                raw_box = player.get("bbox_xyxy_px")
                if not raw_box:
                    stats["dropped_no_box"] += 1
                    continue
                box = _valid_box(raw_box)
                if box is None:
                    stats["dropped_invalid_box"] += 1
                    continue
                confidence = float(player.get("confidence") or 0.0)
                if min_confidence is not None and not confidence > min_confidence:
                    stats["dropped_below_confidence"] += 1
                    continue
                ident = int(player["track_id"])
                if ident in ids_in_frame:
                    raise MotEvalError(f"{frames_path}: track id {ident} appears twice in frame {frame}")
                ids_in_frame.add(ident)
                frames.setdefault(frame, []).append((ident, box, confidence))
                stats["predictions"] += 1
    stats["frames_with_record"] = len(seen_frames)
    return frames, stats


def resolve_window(
    start: int, end: int | None, reference: dict[int, Any], predicted_frames: Iterable[int]
) -> tuple[int, int]:
    """Return the evaluated ``[start, end)``; an open end covers every frame seen on either side."""
    if start < 0:
        raise MotEvalError("start must be >= 0")
    if end is None:
        seen = [*reference, *predicted_frames]
        if not seen:
            raise MotEvalError("no frames to evaluate and no explicit end")
        end = max(seen) + 1
    if end <= start:
        raise MotEvalError(f"empty evaluation window [{start}, {end})")
    return start, end


def build_sequence_data(
    reference: dict[int, list[tuple[int, Box]]],
    predictions: dict[int, list[tuple[int, Box, float]]],
    start: int,
    end: int,
) -> dict[str, Any]:
    """Convert both sides to TrackEval's per-timestep structure (ids remapped to 0..n-1, IoU similarity)."""
    gt_ids_all = sorted({i for f in range(start, end) for i, _ in reference.get(f, [])})
    tr_ids_all = sorted({i for f in range(start, end) for i, _, _ in predictions.get(f, [])})
    gt_map = {i: n for n, i in enumerate(gt_ids_all)}
    tr_map = {i: n for n, i in enumerate(tr_ids_all)}
    gt_ids, tr_ids, similarity = [], [], []
    for frame in range(start, end):
        ref = sorted(reference.get(frame, []), key=lambda r: r[0])
        pred = sorted(predictions.get(frame, []), key=lambda p: p[0])
        gt_ids.append(np.array([gt_map[i] for i, _ in ref], dtype=int))
        tr_ids.append(np.array([tr_map[i] for i, _, _ in pred], dtype=int))
        similarity.append(
            _iou_matrix(
                np.array([b for _, b in ref], dtype=float).reshape(-1, 4),
                np.array([b for _, b, _ in pred], dtype=float).reshape(-1, 4),
            ).reshape(len(ref), len(pred))
        )
    return {
        "num_timesteps": end - start,
        "num_gt_ids": len(gt_ids_all),
        "num_tracker_ids": len(tr_ids_all),
        "num_gt_dets": int(sum(len(a) for a in gt_ids)),
        "num_tracker_dets": int(sum(len(a) for a in tr_ids)),
        "gt_ids": gt_ids,
        "tracker_ids": tr_ids,
        "similarity_scores": similarity,
    }


def trackeval_version() -> str:
    return metadata.version("trackeval")


def standard_metrics(data: dict[str, Any], iou_threshold: float = 0.5) -> dict[str, Any]:
    """Run TrackEval HOTA, CLEAR and Identity on prepared sequence data."""
    try:
        from trackeval.metrics import CLEAR, HOTA, Identity
    except ImportError as exc:  # pragma: no cover - dev dependency
        raise MotEvalError("TrackEval is required: pip install -e \".[dev]\"") from exc

    def combined(metric: Any) -> dict[str, Any]:
        return metric.combine_sequences({"sequence": metric.eval_sequence(data)})

    hota = combined(HOTA())
    clear = combined(CLEAR({"THRESHOLD": iou_threshold, "PRINT_CONFIG": False}))
    identity = combined(Identity({"THRESHOLD": iou_threshold, "PRINT_CONFIG": False}))
    at05 = HOTA_ALPHA_INDEX_05

    def num(value: Any) -> float:
        return round(float(value), 6)

    return {
        "hota": {
            "HOTA": num(np.mean(hota["HOTA"])), "DetA": num(np.mean(hota["DetA"])), "AssA": num(np.mean(hota["AssA"])),
            "DetRe": num(np.mean(hota["DetRe"])), "DetPr": num(np.mean(hota["DetPr"])),
            "AssRe": num(np.mean(hota["AssRe"])), "AssPr": num(np.mean(hota["AssPr"])), "LocA": num(np.mean(hota["LocA"])),
            "alpha_0.5": {k: num(hota[k][at05]) for k in ("HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA")},
            "averaged_over": "19 localisation thresholds alpha = 0.05 ... 0.95",
        },
        "clear": {
            "MOTA": num(clear["MOTA"]), "MOTP": num(clear["MOTP"]), "IDSW": int(clear["IDSW"]), "Frag": int(clear["Frag"]),
            "CLR_TP": int(clear["CLR_TP"]), "CLR_FP": int(clear["CLR_FP"]), "CLR_FN": int(clear["CLR_FN"]),
            "CLR_Re": num(clear["CLR_Re"]), "CLR_Pr": num(clear["CLR_Pr"]),
            "MT": int(clear["MT"]), "PT": int(clear["PT"]), "ML": int(clear["ML"]),
            "iou_threshold": iou_threshold,
        },
        "identity": {
            "IDF1": num(identity["IDF1"]), "IDR": num(identity["IDR"]), "IDP": num(identity["IDP"]),
            "IDTP": int(identity["IDTP"]), "IDFN": int(identity["IDFN"]), "IDFP": int(identity["IDFP"]),
            "iou_threshold": iou_threshold,
        },
        "counts": {
            "gt_ids": data["num_gt_ids"], "tracker_ids": data["num_tracker_ids"],
            "gt_dets": data["num_gt_dets"], "tracker_dets": data["num_tracker_dets"],
            "timesteps": data["num_timesteps"],
        },
    }


def export_motchallenge(
    reference: dict[int, list[tuple[int, Box]]],
    predictions: dict[int, list[tuple[int, Box, float]]],
    start: int,
    end: int,
    out_dir: Path,
    sequence: str = "sequence",
    tracker: str = "tracker",
) -> dict[str, str]:
    """Write a TrackEval ``MotChallenge2DBox`` layout for the window (frames renumbered from 1)."""
    def rows(items: dict[int, list], with_conf: bool) -> list[str]:
        lines = []
        for frame in range(start, end):
            for item in sorted(items.get(frame, []), key=lambda r: r[0]):
                ident, (x1, y1, x2, y2) = item[0], item[1]
                conf = item[2] if with_conf else 1.0
                lines.append(f"{frame - start + 1},{ident},{x1:.3f},{y1:.3f},{x2 - x1:.3f},{y2 - y1:.3f},{conf:.6f},"
                             f"{-1 if with_conf else 1},{-1 if with_conf else 1}")
        return lines

    paths = {
        "gt": out_dir / "gt" / sequence / "gt" / "gt.txt",
        "seqinfo": out_dir / "gt" / sequence / "seqinfo.ini",
        "tracker": out_dir / "trackers" / tracker / "data" / f"{sequence}.txt",
        "seqmap": out_dir / "seqmap.txt",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    paths["gt"].write_text("\n".join(rows(reference, False)) + "\n", encoding="utf-8")
    paths["tracker"].write_text("\n".join(rows(predictions, True)) + "\n", encoding="utf-8")
    paths["seqinfo"].write_text(
        f"[Sequence]\nname={sequence}\nseqLength={end - start}\n", encoding="utf-8"
    )
    paths["seqmap"].write_text(f"name\n{sequence}\n", encoding="utf-8")
    return {k: str(v) for k, v in paths.items()}


def evaluate_run(
    frames_path: Path,
    reference_path: Path,
    start: int = 0,
    end: int | None = None,
    view: str = "all_non_ball",
    iou_threshold: float = 0.5,
    min_confidence: float | None = None,
    gt_frame_base: int = 0,
    include_approximate: bool = True,
) -> dict[str, Any]:
    reference, ref_stats = load_reference(reference_path, gt_frame_base)
    predictions, pred_stats = load_predictions(frames_path, 0, None, view, min_confidence)
    start, end = resolve_window(start, end, reference, predictions)
    in_window = {f: p for f, p in predictions.items() if start <= f < end}
    pred_stats["predictions_in_window"] = sum(len(v) for v in in_window.values())
    record_frames = _record_frames(frames_path)
    pred_stats["window_frames_without_prediction_record"] = sum(1 for f in range(start, end) if f not in record_frames)
    data = build_sequence_data(reference, in_window, start, end)
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": "standard_mot_evaluation",
        "evaluator": {
            "adapter": "football_fatigue.vision.mot_eval", "adapter_version": ADAPTER_VERSION,
            "trackeval": trackeval_version(), "numpy": np.__version__,
        },
        "inputs": {
            "frames": {"path": str(frames_path), "sha256": file_sha256(frames_path)},
            "reference": {"path": str(reference_path), "sha256": file_sha256(reference_path)},
        },
        "configuration": {
            "window": {"start": start, "end_exclusive": end, "frames": end - start, "frame_indexing": "0-based decoded frame index"},
            "gt_frame_base": gt_frame_base, "class_view": view, "iou_threshold": iou_threshold,
            "min_confidence": min_confidence, "min_confidence_comparison": "confidence > min_confidence",
        },
        "input_handling": {"reference": ref_stats, "predictions": pred_stats},
        "standard_metrics": standard_metrics(data, iou_threshold),
    }
    if include_approximate:
        report["approximate_project_metrics"] = _approximate(frames_path, reference_path, start, end, view, iou_threshold)
    return report


def _record_frames(frames_path: Path) -> set[int]:
    with frames_path.open(encoding="utf-8") as stream:
        return {int(json.loads(line)["frame_num"]) for line in stream}


def _approximate(frames_path: Path, reference_path: Path, start: int, end: int, view: str, iou: float) -> dict[str, Any]:
    """The project's own greedy evaluator on the same class view; kept separate and labelled approximate."""
    import tempfile

    from football_fatigue.vision.tracking_eval import evaluate

    allowed = CLASS_VIEWS[view]
    with tempfile.TemporaryDirectory() as tmp:
        filtered = Path(tmp) / "frames.jsonl"
        with frames_path.open(encoding="utf-8") as src, filtered.open("w", encoding="utf-8") as dst:
            for line in src:
                record = json.loads(line)
                record["players"] = [
                    p for p in record["players"]
                    if p.get("detected_class") != "ball" and (allowed is None or p.get("detected_class") in allowed)
                ]
                dst.write(json.dumps(record) + "\n")
        result = evaluate(filtered, reference_path, iou, start, end)
    return {
        "evaluator": "football_fatigue.vision.tracking_eval",
        "note": "Approximate: greedy per-frame assignment at one IoU. 'idf1_approximate' is not the standard IDF1.",
        "idf1_approximate": result["idf1"],
        **{k: result[k] for k in (
            "precision", "recall", "tp", "fp", "fn", "id_switches", "fragmentations", "mota_like",
            "predicted_track_ids", "reference_identities",
        )},
        "small_reference_boxes_h_lt_35px": result["fn_breakdown"]["reference_small_boxes_h_lt_35px"],
        "small_reference_boxes_missed": result["fn_breakdown"]["missed_small_boxes_h_lt_35px"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Standard MOT metrics (TrackEval HOTA/CLEAR/Identity) for a saved CV run.")
    ap.add_argument("run_dir", type=Path, help="run directory containing frames.jsonl")
    ap.add_argument("reference", type=Path, help="MOTChallenge reference file")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, help="exclusive; default: every frame seen")
    ap.add_argument("--view", choices=sorted(CLASS_VIEWS), default="all_non_ball")
    ap.add_argument("--iou", type=float, default=0.5, help="threshold for CLEAR and Identity (HOTA averages 0.05-0.95)")
    ap.add_argument("--min-confidence", type=float)
    ap.add_argument("--gt-frame-base", type=int, default=0, choices=(0, 1))
    ap.add_argument("--output", type=Path, required=True, help="report JSON to write (never inside the run directory unless you say so)")
    ap.add_argument("--export-motchallenge", type=Path, help="also write a TrackEval MotChallenge2DBox layout here")
    ap.add_argument("--no-approximate", action="store_true")
    args = ap.parse_args(argv)
    if args.output.exists():
        print(f"error: {args.output} already exists; reports are never overwritten", file=sys.stderr)
        return 2
    try:
        report = evaluate_run(
            args.run_dir / "frames.jsonl", args.reference, args.start, args.end, args.view, args.iou,
            args.min_confidence, args.gt_frame_base, not args.no_approximate,
        )
        if args.export_motchallenge:
            window = report["configuration"]["window"]
            reference, _ = load_reference(args.reference, args.gt_frame_base)
            predictions, _ = load_predictions(args.run_dir / "frames.jsonl", 0, None, args.view, args.min_confidence)
            report["export"] = export_motchallenge(
                reference, predictions, window["start"], window["end_exclusive"], args.export_motchallenge
            )
    except MotEvalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"standard_metrics": report["standard_metrics"], "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
