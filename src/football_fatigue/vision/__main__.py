"""Run football video tracking and write a provenance-stamped JSONL result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from football_fatigue.vision import (
    DetectedClass,
    HomographyCalibrator,
    MultiObjectTracker,
    VisionTrackingPipeline,
    YOLOPlayerDetector,
)
from football_fatigue.analytics.cv import build_cv_analytics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version(package: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(package)
    except Exception:
        return None


def _load_calibration(path: Path) -> HomographyCalibrator:
    payload = json.loads(path.read_text(encoding="utf-8"))
    calibrator = HomographyCalibrator.from_dict(payload)
    if not calibrator.is_calibrated:
        raise ValueError(f"Calibration file has no matrix: {path}")
    return calibrator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Input video file")
    parser.add_argument("--calibration", type=Path, help="JSON from HomographyCalibrator.to_dict()")
    parser.add_argument(
        "--detection-only", action="store_true",
        help="Run image-space detection/tracking without projecting positions to pitch coordinates",
    )
    parser.add_argument("--weights", required=True, type=Path, help="Football detector checkpoint (.pt); no implicit model download")
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/vision"))
    parser.add_argument("--run-id", required=True, help="New output directory name; existing outputs are never overwritten")
    parser.add_argument("--confidence", type=float, default=0.40)
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size in pixels (larger can improve small-object recall at higher memory cost)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--max-seconds", type=float, help="Stop after this much decoded video time")
    parser.add_argument("--tracker", choices=("bytetrack", "iou"), default="bytetrack")
    parser.add_argument(
        "--calibration-check", type=Path,
        help="Optional JSON with independent image_points and pitch_points arrays for held-out residuals",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.detection_only == (args.calibration is not None):
        print("error: provide --calibration for pitch tracking, or use --detection-only without calibration", file=sys.stderr)
        return 2
    if args.calibration_check is not None and args.detection_only:
        print("error: --calibration-check cannot be used with --detection-only", file=sys.stderr)
        return 2
    required_files = [("video", args.video), ("weights", args.weights)]
    if args.calibration is not None:
        required_files.append(("calibration", args.calibration))
    if args.calibration_check is not None:
        required_files.append(("calibration-check", args.calibration_check))
    for name, path in required_files:
        if not path.is_file():
            print(f"error: {name} file does not exist: {path}", file=sys.stderr)
            return 2
    if not args.run_id or args.run_id in {".", ".."} or Path(args.run_id).name != args.run_id:
        print("error: --run-id must be one safe path component", file=sys.stderr)
        return 2
    if not 0 < args.confidence <= 1:
        print("error: --confidence must be within (0, 1]", file=sys.stderr)
        return 2
    if args.imgsz < 32:
        print("error: --imgsz must be at least 32", file=sys.stderr)
        return 2
    if args.max_frames is not None and args.max_frames < 1:
        print("error: --max-frames must be positive", file=sys.stderr)
        return 2
    if args.max_seconds is not None and (not math.isfinite(args.max_seconds) or args.max_seconds <= 0):
        print("error: --max-seconds must be finite and positive", file=sys.stderr)
        return 2

    output_root = args.output_root.resolve()
    raw_root = (Path.cwd() / "data" / "raw").resolve()
    try:
        output_root.relative_to(raw_root)
    except ValueError:
        pass
    else:
        print("error: output root cannot be inside data/raw", file=sys.stderr)
        return 2
    run_dir = output_root / args.run_id
    if run_dir.exists():
        print(f"error: output already exists; choose another --run-id: {run_dir}", file=sys.stderr)
        return 2
    try:
        calibrator = _load_calibration(args.calibration) if args.calibration is not None else None
        calibration_check = None
        if args.calibration_check is not None:
            check_data = json.loads(args.calibration_check.read_text(encoding="utf-8"))
            check_pixels = [tuple(map(float, pair)) for pair in check_data["image_points"]]
            check_pitch = [tuple(map(float, pair)) for pair in check_data["pitch_points"]]
            check_errors = calibrator.reprojection_errors_m(check_pixels, check_pitch)
            if not check_errors:
                raise ValueError("calibration check must include at least one independent point")
            calibration_check = {
                "source_sha256": _sha256(args.calibration_check),
                "point_count": len(check_errors),
                "errors_m": check_errors,
                "mean_error_m": sum(check_errors) / len(check_errors),
                "max_error_m": max(check_errors),
                "independent_held_out_points": True,
            }
        tracker = MultiObjectTracker(use_supervision=args.tracker == "bytetrack")
        if args.tracker == "bytetrack" and tracker.backend != "bytetrack":
            print("error: ByteTrack could not initialize; install the vision extra or select --tracker iou", file=sys.stderr)
            return 2
        pipeline = VisionTrackingPipeline(
            calibrator=calibrator,
            detector=YOLOPlayerDetector(
                str(args.weights), args.confidence, args.device, image_size=args.imgsz,
            ),
            tracker=tracker,
        )
    except Exception as exc:
        print(f"error: configuration failed: {exc}", file=sys.stderr)
        return 2

    output_root.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=False, exist_ok=False)
    frames_path = run_dir / "frames.jsonl"
    count = 0
    processed_frames = []
    player_observations = 0
    other_non_ball_observations = 0
    usable_player_positions = 0
    ball_observations = 0
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        with frames_path.open("x", encoding="utf-8", newline="\n") as stream:
            for frame in pipeline.process_video_generator(
                str(args.video), max_frames=args.max_frames, max_seconds=args.max_seconds,
            ):
                stream.write(json.dumps(frame.to_dict(), allow_nan=False, separators=(",", ":")) + "\n")
                if (count + 1) % 25 == 0:
                    stream.flush()
                processed_frames.append(frame)
                count += 1
                player_observations += sum(
                    p.detected_class in (DetectedClass.PLAYER, DetectedClass.GOALKEEPER)
                    for p in frame.players
                )
                other_non_ball_observations += sum(
                    p.detected_class not in (DetectedClass.PLAYER, DetectedClass.GOALKEEPER)
                    for p in frame.players
                )
                usable_player_positions += sum(p.is_usable for p in frame.players)
                ball_observations += frame.ball is not None
        if count == 0:
            raise ValueError("video yielded no decodable frames")
        cv_analytics = build_cv_analytics(processed_frames)
        (run_dir / "analytics.json").write_text(
            json.dumps(cv_analytics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        manifest: dict[str, Any] = {
            "status": "processed_unvalidated",
            "created_at_utc": started_at,
            "inputs": {
                "video_sha256": _sha256(args.video),
                "calibration_sha256": _sha256(args.calibration) if args.calibration else None,
                "weights_sha256": _sha256(args.weights),
                "calibration_check_sha256": (
                    _sha256(args.calibration_check) if args.calibration_check else None
                ),
            },
            "configuration": {
                "confidence_threshold": args.confidence,
                "image_size": args.imgsz,
                "mode": "detection_only" if args.detection_only else "pitch_tracking",
                "device": args.device,
                "requested_tracker": args.tracker,
                "selected_tracker": tracker.backend,
                "max_frames": args.max_frames,
                "max_seconds": args.max_seconds,
                "calibration": calibrator.to_dict() if calibrator else None,
                "video_fps": pipeline.fps,
                "video_width_px": pipeline._video_width,
                "video_height_px": pipeline._video_height,
            },
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "ultralytics": _version("ultralytics"),
                "supervision": _version("supervision"),
                "opencv": _version("opencv-python"),
            },
            "summary": {
                "frames_written": count,
                "player_observations": player_observations,
                "other_non_ball_observations": other_non_ball_observations,
                "usable_estimated_player_positions": usable_player_positions,
                "ball_observations_unusable_for_ground_position": ball_observations,
                "analytics_file": "analytics.json",
                "movement_analytics_status": cv_analytics["movement_analytics"]["status"],
                "stage_5": cv_analytics["stage_5_tactical_metrics"],
            },
            "calibration_validation": calibration_check or {
                "status": "not_applicable_detection_only" if args.detection_only else "not_supplied",
                "fit_point_residuals_are_not_reported_as_accuracy": True,
            },
            "limitations": [
                "No accuracy claim: compare against licensed annotations before reporting metrics.",
                (
                    "Detection-only mode: pixel boxes are retained; pitch coordinates and movement metrics are withheld."
                    if args.detection_only else
                    "Pitch coordinates are homography estimates; calibration quality is not independently validated."
                ),
                "Player identity and jersey number are not inferred; team side may remain unknown.",
                "Ball ground-plane coordinates are withheld because a planar homography cannot recover airborne height.",
                "Tracking-derived movement is not a physiological fatigue measurement.",
            ],
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except Exception as exc:
        manifest = {
            "status": "failed_or_incomplete",
            "created_at_utc": started_at,
            "error": f"{type(exc).__name__}: {exc}",
            "inputs": {
                "video_sha256": _sha256(args.video),
                "calibration_sha256": _sha256(args.calibration) if args.calibration else None,
                "weights_sha256": _sha256(args.weights),
                "calibration_check_sha256": (
                    _sha256(args.calibration_check) if args.calibration_check else None
                ),
            },
            "configuration": {
                "confidence_threshold": args.confidence,
                "image_size": args.imgsz,
                "device": args.device,
                "requested_tracker": args.tracker,
                "selected_tracker": tracker.backend,
                "max_frames": args.max_frames,
                "max_seconds": args.max_seconds,
                "mode": "detection_only" if args.detection_only else "pitch_tracking",
                "calibration": calibrator.to_dict() if calibrator else None,
                "video_fps": pipeline.fps,
                "video_width_px": pipeline._video_width,
                "video_height_px": pipeline._video_height,
            },
            "runtime": {
                "python": platform.python_version(),
                "ultralytics": _version("ultralytics"),
                "supervision": _version("supervision"),
                "opencv": _version("opencv-python"),
            },
            "summary": {
                "frames_written": count,
                "player_observations": player_observations,
                "other_non_ball_observations": other_non_ball_observations,
                "usable_estimated_player_positions": usable_player_positions,
                "ball_observations_unusable_for_ground_position": ball_observations,
            },
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(f"error: processing failed: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {count} frames to {run_dir}")
    print(f"Tracker: {tracker.backend}; run status: processed_unvalidated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
