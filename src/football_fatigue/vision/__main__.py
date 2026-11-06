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
from football_fatigue.vision.detection_cache import (
    CACHE_SCHEMA_VERSION,
    REPLAY_COMPATIBILITY,
    REPLAY_MODES,
    DetectionCacheError,
    DetectionCacheReader,
    DetectionCacheWriter,
    resolve_replay,
)

_CACHE_DIRNAME = "detection_cache"
# Detector settings that a replay takes from the cache; passing different ones is an error.
_REPLAY_FIXED_DETECTOR_FLAGS = (
    "imgsz", "device", "mask_margin", "tile_merge", "tile_merge_cross_role_iou",
)


def _probe_video(video: Path, start_frame: int, max_frames: int | None, max_seconds: float | None) -> dict[str, Any]:
    """Frame rate, size and the number of frames this run will process."""
    import cv2 as _cv2
    probe = _cv2.VideoCapture(str(video))
    expected = int(probe.get(_cv2.CAP_PROP_FRAME_COUNT)) - start_frame
    fps = float(probe.get(_cv2.CAP_PROP_FPS)) or 25.0
    width, height = int(probe.get(_cv2.CAP_PROP_FRAME_WIDTH)), int(probe.get(_cv2.CAP_PROP_FRAME_HEIGHT))
    probe.release()
    if max_frames is not None:
        expected = min(expected, max_frames)
    if max_seconds is not None:
        expected = min(expected, int(max_seconds * fps))
    return {"fps": fps, "width": width, "height": height, "expected": expected}


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
    parser.add_argument("--weights", type=Path, help="Football detector checkpoint (.pt); no implicit model download. Required unless --replay-detections is used")
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/vision"))
    parser.add_argument("--run-id", required=True, help="New output directory name; existing outputs are never overwritten")
    parser.add_argument("--confidence", type=float, default=0.40)
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size in pixels (larger can improve small-object recall at higher memory cost)")
    parser.add_argument("--tile-width", type=int, help="Tiled inference: full-height strips of this width (overlapping), boxes merged by NMS")
    parser.add_argument("--tile-height", type=int, help="With --tile-width: tile height for a 2D grid of overlapping tiles (default: full-height strips)")
    parser.add_argument("--tile-batch", type=int, help="With --tile-width: tiles per model call, to bound peak GPU memory (default: all tiles of a frame)")
    parser.add_argument(
        "--tile-merge", choices=("class_aware", "person_agnostic", "edge_aware"), default="class_aware",
        help="Duplicate merging across tiles: class_aware (default), person_agnostic (cross-role duplicates and tile-edge truncation) or edge_aware (person_agnostic, with tile-edge truncation resolved before NMS; see docs)",
    )
    parser.add_argument("--tile-merge-cross-role-iou", type=float, default=0.7, help="person_agnostic: IoU at which different person roles count as one object")
    parser.add_argument("--pitch-keypoints", type=Path, help="Keypoints JSON; detections with feet outside the dilated pitch outline are dropped (image-space mask, not metric calibration)")
    parser.add_argument("--mask-margin", type=int, default=60)
    parser.add_argument("--track-activation", type=float, default=0.5, help="ByteTrack high-confidence activation threshold")
    parser.add_argument("--track-buffer", type=int, default=75, help="ByteTrack lost-track buffer in frames")
    parser.add_argument("--track-match", type=float, default=0.95, help="ByteTrack IoU-distance matching threshold (higher is more permissive)")
    parser.add_argument("--half", action="store_true", help="FP16 inference (CUDA only)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--start-frame", type=int, default=0, help="First source frame to process (timestamps stay absolute)")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--max-seconds", type=float, help="Stop after this much decoded video time")
    parser.add_argument("--tracker", choices=("bytetrack", "iou"), default="bytetrack")
    parser.add_argument(
        "--role-policy", choices=("stable", "latest"), default="stable",
        help="stable: a track's reported role is a decayed evidence estimate; latest: the current frame's raw label "
             "(association is class-agnostic and the ball is never tracked either way)",
    )
    parser.add_argument(
        "--calibration-check", type=Path,
        help="Optional JSON with independent image_points and pitch_points arrays for held-out residuals",
    )
    parser.add_argument(
        "--save-detections", action="store_true",
        help=f"Also write every raw detector observation to <run>/{_CACHE_DIRNAME}/ for later replay",
    )
    parser.add_argument(
        "--cache-confidence-floor", type=float,
        help="With --save-detections: run the detector at this lower confidence and store everything above it, while "
             "--confidence stays the tracking-input threshold (default: equal to --confidence)",
    )
    parser.add_argument(
        "--replay-detections", type=Path,
        help="Detection cache (or a run directory containing one): track stored detections without running the detector; --confidence is the tracking-input threshold of the replay",
    )
    parser.add_argument(
        "--replay-mode", choices=REPLAY_MODES, default=REPLAY_COMPATIBILITY,
        help="compatibility: --confidence must equal the saving run's tracking-input threshold (reconstructs its tracker input). "
             "experimental: any --confidence at or above the cache floor; results are labelled experimental",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    replaying = args.replay_detections is not None
    if replaying and args.save_detections:
        print("error: --save-detections and --replay-detections cannot be combined", file=sys.stderr)
        return 2
    if args.cache_confidence_floor is not None and not args.save_detections:
        print(
            "error: --cache-confidence-floor only applies with --save-detections"
            + ("; a replay uses the floor stored in the cache" if replaying else ""),
            file=sys.stderr,
        )
        return 2
    if not replaying and args.weights is None:
        print("error: --weights is required unless --replay-detections is used", file=sys.stderr)
        return 2
    if replaying:
        conflicting = [
            f"--{name.replace('_', '-')}" for name in _REPLAY_FIXED_DETECTOR_FLAGS
            if getattr(args, name) != parser.get_default(name)
        ] + [
            flag for flag, value in (("--tile-width", args.tile_width), ("--tile-height", args.tile_height),
                                     ("--tile-batch", args.tile_batch), ("--half", args.half or None),
                                     ("--pitch-keypoints", args.pitch_keypoints)) if value is not None
        ]
        if conflicting:
            print(f"error: detector settings are fixed by the cache; remove {', '.join(conflicting)}", file=sys.stderr)
            return 2
    if args.detection_only == (args.calibration is not None):
        print("error: provide --calibration for pitch tracking, or use --detection-only without calibration", file=sys.stderr)
        return 2
    if args.calibration_check is not None and args.detection_only:
        print("error: --calibration-check cannot be used with --detection-only", file=sys.stderr)
        return 2
    required_files = [("video", args.video)]
    if args.weights is not None:
        required_files.append(("weights", args.weights))
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
    if args.cache_confidence_floor is not None and not (
        0 < args.cache_confidence_floor <= args.confidence
    ):
        print("error: --cache-confidence-floor must be within (0, --confidence]", file=sys.stderr)
        return 2
    if args.tile_width is None and (args.tile_height is not None or args.tile_batch is not None
                                    or args.tile_merge != "class_aware") and not replaying:
        print("error: --tile-height, --tile-batch and --tile-merge require --tile-width", file=sys.stderr)
        return 2
    if args.tile_height is not None and args.tile_height < 64 or args.tile_batch is not None and args.tile_batch < 1:
        print("error: --tile-height must be at least 64 and --tile-batch at least 1", file=sys.stderr)
        return 2
    if not 0.5 <= args.tile_merge_cross_role_iou <= 1:
        print("error: --tile-merge-cross-role-iou must be within [0.5, 1]", file=sys.stderr)
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
        _mask_polygon = None
        if args.pitch_keypoints is not None:
            from football_fatigue.vision.calibration import pitch_polygon_from_keypoints
            _mask_polygon = pitch_polygon_from_keypoints(str(args.pitch_keypoints), args.mask_margin)
        reader = replay_info = None
        video_sha256 = _sha256(args.video) if (args.save_detections or replaying) else None
        weights_sha256 = _sha256(args.weights) if args.weights is not None else None
        if replaying:
            # Refuse a mismatched cache before any output is created.
            reader = DetectionCacheReader(args.replay_detections)
            replay_info = resolve_replay(reader.manifest, args.replay_mode, args.confidence)
            replay_probe = _probe_video(args.video, args.start_frame, args.max_frames, args.max_seconds)
            if replay_probe["expected"] < 1:
                raise ValueError("replay window contains no frames")
            reader.validate_for_replay(
                video_sha256=video_sha256, fps=replay_probe["fps"],
                width=replay_probe["width"], height=replay_probe["height"],
                first_frame=args.start_frame, last_frame=args.start_frame + replay_probe["expected"] - 1,
                weights_sha256=weights_sha256,
            )
            weights_sha256 = reader.manifest["detector"]["weights_sha256"]
        tracker = MultiObjectTracker(
            use_supervision=args.tracker == "bytetrack",
            track_activation_threshold=args.track_activation,
            track_buffer=args.track_buffer,
            matching_threshold=args.track_match,
            role_policy=args.role_policy,
        )
        if args.tracker == "bytetrack" and tracker.backend != "bytetrack":
            print("error: ByteTrack could not initialize; install the vision extra or select --tracker iou", file=sys.stderr)
            return 2
        inference_floor = args.cache_confidence_floor if args.cache_confidence_floor is not None else args.confidence
        detector = None if replaying else YOLOPlayerDetector(
            str(args.weights), inference_floor, args.device, image_size=args.imgsz,
            tile_width=args.tile_width, half=args.half, tile_height=args.tile_height,
            tile_batch=args.tile_batch, tile_merge=args.tile_merge, cross_role_iou=args.tile_merge_cross_role_iou,
        )
        pipeline = VisionTrackingPipeline(
            calibrator=calibrator,
            detector=detector,
            tracker=tracker,
            pitch_mask_polygon=_mask_polygon,
            tracking_confidence=args.confidence,
        )
    except DetectionCacheError as exc:
        print(f"error: detection cache rejected: {exc}", file=sys.stderr)
        return 2
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
    import time as _time
    t_start = _time.perf_counter()
    writer: DetectionCacheWriter | None = None
    try:
        probe = _probe_video(args.video, args.start_frame, args.max_frames, args.max_seconds)
        expected, _probe_fps, _w, _h = probe["expected"], probe["fps"], probe["width"], probe["height"]
        (run_dir / "manifest.json").write_text(json.dumps({
            "status": "processing",
            "configuration": {"video_fps": _probe_fps, "video_width_px": _w or 4096, "video_height_px": _h or 1080},
            "summary": {"total_frames_expected": expected},
        }, indent=2) + "\n", encoding="utf-8")
        if args.save_detections:
            writer = DetectionCacheWriter(run_dir / _CACHE_DIRNAME, {
                "run_id": args.run_id,
                "video": {
                    "path": str(args.video), "sha256": video_sha256, "fps": _probe_fps,
                    "width": _w, "height": _h,
                },
                "detector": detector.describe() | {
                    "weights_sha256": weights_sha256,
                    "pitch_mask": {
                        "keypoints": str(args.pitch_keypoints), "keypoints_sha256": _sha256(args.pitch_keypoints),
                        "margin_px": args.mask_margin,
                    } if args.pitch_keypoints else None,
                },
                "tracking_input": {
                    "confidence_threshold": args.confidence, "comparison": "confidence > threshold",
                },
                "window": {
                    "start_frame": args.start_frame, "max_frames": args.max_frames,
                    "max_seconds": args.max_seconds, "expected_frames": expected,
                },
                "runtime": {
                    "python": platform.python_version(), "ultralytics": _version("ultralytics"),
                    "opencv": _version("opencv-python"),
                },
            })
            pipeline.observation_sink = writer.write_frame
        detection_source = reader.detection_source(args.start_frame, args.confidence) if replaying else None
        with frames_path.open("x", encoding="utf-8", newline="\n") as stream:
            for frame in pipeline.process_video_generator(
                str(args.video), max_frames=args.max_frames, max_seconds=args.max_seconds,
                start_frame=args.start_frame, detection_source=detection_source,
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
        detection_cache_info: dict[str, Any] | None = None
        if writer is not None:
            writer.close(
                complete=True, class_names=detector.describe()["class_names"],
                detector_extra={"merge_statistics": dict(detector.merge_statistics)},
            )
            cache_reader = DetectionCacheReader(writer.directory)
            detection_cache_info = {
                "role": "saved", "path": _CACHE_DIRNAME, "schema_version": CACHE_SCHEMA_VERSION,
                "cache_confidence_floor": cache_reader.confidence_floor,
                "tracking_confidence": cache_reader.tracking_confidence,
                "interpretation": (
                    "A tracking run: its tracker saw only observations with confidence > tracking_confidence. "
                    "Observations between the cache floor and that threshold are stored but were not tracked."
                ),
                "frames_written": cache_reader.manifest["frames_written"],
                "records_sha256": cache_reader.manifest["records_sha256"],
                "manifest_sha256": cache_reader.manifest_sha256,
                "counts": cache_reader.manifest["counts"],
            }
        elif replaying:
            detection_cache_info = {
                "role": "replayed", "path": str(reader.directory), "schema_version": CACHE_SCHEMA_VERSION,
                "saved_by_run": reader.manifest.get("run_id"),
                "frames_in_cache": reader.manifest["frames_written"],
                "records_sha256": reader.manifest["records_sha256"],
                "manifest_sha256": reader.manifest_sha256,
                "interpretation": (
                    "Experimental replay: a tracking-input threshold other than the saving run's; not comparable "
                    "with the saving run or the historical baseline as if equivalent."
                    if replay_info["experimental"] else
                    "Compatibility replay: reconstructs the saving run's tracker input from the cache."
                ),
                "replay": replay_info,
            }
        cached_detector = reader.manifest["detector"] if replaying else None
        cv_analytics = build_cv_analytics(processed_frames)
        (run_dir / "analytics.json").write_text(
            json.dumps(cv_analytics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        manifest: dict[str, Any] = {
            "status": "processed_unvalidated",
            "created_at_utc": started_at,
            "inputs": {
                "video_sha256": video_sha256 or _sha256(args.video),
                "calibration_sha256": _sha256(args.calibration) if args.calibration else None,
                "weights_sha256": weights_sha256,
                "calibration_check_sha256": (
                    _sha256(args.calibration_check) if args.calibration_check else None
                ),
            },
            "configuration": {
                "confidence_threshold": args.confidence,  # tracking-input threshold
                "cache_confidence_floor": (
                    cached_detector.get("confidence_floor") if replaying else
                    (detector.confidence_threshold if args.save_detections else None)
                ),
                "image_size": cached_detector.get("image_size") if replaying else args.imgsz,
                "tile_width": cached_detector.get("tile_width") if replaying else args.tile_width,
                "tile_height": cached_detector.get("tile_height") if replaying else args.tile_height,
                "tile_batch": cached_detector.get("tile_batch") if replaying else args.tile_batch,
                "tile_merge": cached_detector.get("tile_merge") if replaying else detector.describe()["tile_merge"],
                "tracker_params": {"activation": args.track_activation, "buffer_frames": args.track_buffer, "matching_threshold": args.track_match},
                "class_safety": {
                    "person_tracker_inputs": "player, goalkeeper, referee, person; the ball is never submitted",
                    "association": "class-agnostic across person roles",
                    "role_policy": args.role_policy,
                    "role_half_life_s": tracker._role_half_life_s,
                    "role_switch_margin": tracker._role_switch_margin,
                    "reported_role": "detected_class = the track's role estimate (a label, not a verified identity); "
                                     "observed_class = this frame's raw detector label",
                },
                "kit_classifier": "adaptive saturation split (blue/white), other=GK/referee kits, unknown otherwise; colour labels, not home/away",
                "pitch_mask": (
                    cached_detector.get("pitch_mask") if replaying else
                    {"keypoints": str(args.pitch_keypoints), "margin_px": args.mask_margin} if args.pitch_keypoints else None
                ),
                "half": cached_detector.get("half") if replaying else args.half,
                "mode": "detection_only" if args.detection_only else "pitch_tracking",
                "device": "replay (detector not run)" if replaying else args.device,
                "requested_tracker": args.tracker,
                "selected_tracker": tracker.backend,
                "start_frame": args.start_frame,
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
                "total_frames_expected": expected,
                "wall_time_s": round(_time.perf_counter() - t_start, 1),
                "stage_seconds": {k: round(v, 2) for k, v in pipeline.timings.items()} | {"write_and_analytics": round(_time.perf_counter() - t_start - sum(pipeline.timings.values()), 2)},
                "processing_fps": round(count / max(1e-9, _time.perf_counter() - t_start), 2),
                "player_observations": player_observations,
                "other_non_ball_observations": other_non_ball_observations,
                "usable_estimated_player_positions": usable_player_positions,
                "ball_observations_unusable_for_ground_position": ball_observations,
                "ball_detections_not_tracked": pipeline.ball_detections_seen,
                "detector_merge_statistics": None if replaying else dict(detector.merge_statistics),
                "role_statistics": tracker.roles.stats(),
                "analytics_file": "analytics.json",
                "movement_analytics_status": cv_analytics["movement_analytics"]["status"],
                "stage_5": cv_analytics["stage_5_tactical_metrics"],
            },
            **({"detection_cache": detection_cache_info} if detection_cache_info else {}),
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
                "The ball is not tracked: each frame's `ball` is its highest-confidence detection, not a track.",
                "Track roles (player/goalkeeper/referee) are evidence-weighted labels, not verified identities.",
                "Tracking-derived movement is not a physiological fatigue measurement.",
                *(["Detections were replayed from a cache; the detector did not run."] if replaying else []),
                *(
                    ["Experimental replay: not a reproduction of the saving run's detector setting; "
                     "do not compare with compatibility results as if equivalent."]
                    if replaying and replay_info["experimental"] else []
                ),
            ],
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except Exception as exc:
        if writer is not None:
            writer.close(complete=False, error=f"{type(exc).__name__}: {exc}")
        manifest = {
            "status": "failed_or_incomplete",
            "created_at_utc": started_at,
            "error": f"{type(exc).__name__}: {exc}",
            "inputs": {
                "video_sha256": video_sha256 or _sha256(args.video),
                "calibration_sha256": _sha256(args.calibration) if args.calibration else None,
                "weights_sha256": weights_sha256,
                "calibration_check_sha256": (
                    _sha256(args.calibration_check) if args.calibration_check else None
                ),
            },
            "configuration": {
                "confidence_threshold": args.confidence,
                "image_size": args.imgsz,
                "device": args.device,
                "save_detections": args.save_detections,
                **({"replay": replay_info} if replaying else {}),
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
