"""End-to-end Computer Vision Tracking & Pitch Calibration Pipeline."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Callable, Generator, Sequence
from football_fatigue.vision.calibration import HomographyCalibrator
from football_fatigue.vision.detector import YOLOPlayerDetector
from football_fatigue.vision.models import (
    BoundingBox,
    DetectedClass,
    Detection,
    TeamSide,
    TrackedBall,
    TrackedPlayer,
    VisionFrame,
)
from football_fatigue.vision.team_classifier import TeamColorClassifier
from football_fatigue.vision.tracker import ActiveTrack, MultiObjectTracker

if TYPE_CHECKING:
    from football_fatigue.analytics.movement import TrajectoryPoint


def player_trajectory_points(
    frames: Sequence[VisionFrame],
    track_id: int,
    *,
    canonical_period: int,
) -> list[TrajectoryPoint]:
    """Adapt usable CV positions to the analytics trajectory contract.

    Period mapping is an explicit caller input because a video timestamp alone
    does not establish which match period it represents.
    """
    if canonical_period not in (1, 2, 3, 4):
        raise ValueError("canonical_period must be explicitly mapped to 1, 2, 3, or 4.")
    from football_fatigue.analytics.movement import TrajectoryPoint

    points = []
    for frame in frames:
        player = next((p for p in frame.players if p.track_id == track_id), None)
        if (
            player is None or not player.is_usable
            or player.pitch_x is None or player.pitch_y is None
        ):
            continue
        points.append(TrajectoryPoint(
            frame_num=frame.frame_num,
            video_time_s=frame.timestamp_s,
            canonical_period=canonical_period,
            x=player.pitch_x,
            y=player.pitch_y,
            is_estimated=True,
            coord_source="cv_homography",
        ))
    return points


class VisionTrackingPipeline:
    """Orchestrates detection/tracking, with optional pitch calibration."""

    def __init__(
        self,
        calibrator: HomographyCalibrator | None,
        detector: YOLOPlayerDetector | None = None,
        tracker: MultiObjectTracker | None = None,
        team_classifier: TeamColorClassifier | None = None,
        fps: float = 25.0,
    ) -> None:
        self.calibrator = calibrator
        self.detector = detector or YOLOPlayerDetector()
        self.tracker = tracker or MultiObjectTracker()
        self.team_classifier = team_classifier or TeamColorClassifier()
        self.fps = float(fps)
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be finite and greater than zero.")
        self._prev_player_positions: dict[int, tuple[float, float, float]] = {}  # track_id -> (x, y, time)
        self._last_frame_num: int | None = None
        self._last_timestamp_s: float | None = None
        self._last_video_path: str | None = None
        self._video_width: int | None = None
        self._video_height: int | None = None

    def set_video_fps(self, fps: float) -> None:
        """Set the decoded video frame rate before processing begins."""
        if self._last_frame_num is not None:
            raise RuntimeError("Cannot change video fps after frame processing has started.")
        if not math.isfinite(float(fps)) or fps <= 0:
            raise ValueError("fps must be finite and greater than zero.")
        self.fps = float(fps)
        self.tracker.set_frame_rate(self.fps)

    def process_frame(
        self,
        frame_rgb: Any,  # numpy array (H, W, 3)
        frame_num: int,
        timestamp_s: float | None = None,
        detections: Sequence[Detection] | None = None,
    ) -> VisionFrame:
        """Process one frame; pitch coordinates are absent without calibration."""
        if timestamp_s is None:
            timestamp_s = frame_num / self.fps
        if frame_num < 0 or not math.isfinite(float(timestamp_s)) or timestamp_s < 0:
            raise ValueError("frame_num and timestamp_s must be non-negative; timestamp_s must be finite.")
        if self._last_frame_num is not None and frame_num <= self._last_frame_num:
            raise ValueError("frame_num must increase for each processed frame.")
        if self._last_timestamp_s is not None and timestamp_s <= self._last_timestamp_s:
            raise ValueError("timestamp_s must increase for each processed frame.")

        # 1. Detect entities if not pre-supplied
        if detections is None:
            if frame_rgb is None:
                raise ValueError("Either frame_rgb or detections must be provided.")
            detections = self.detector.detect(frame_rgb)

        # 2. Update multi-object tracks
        tracks = self.tracker.update(detections, frame_num)

        # 3. Separate ball and players
        tracked_players: list[TrackedPlayer] = []
        tracked_ball: TrackedBall | None = None

        for trk in tracks:
            if trk.class_name == DetectedClass.BALL:
                # A planar homography cannot recover airborne ball height. Keep
                # the lower box edge's ground-plane projection as an estimate,
                # but do not expose it as a usable pitch position without an
                # independent grounded-ball annotation.
                center = trk.bbox.bottom_center
                pitch_coords = (
                    self.calibrator.pixel_to_pitch(center[0], center[1])
                    if self.calibrator is not None else None
                )
                px, py = pitch_coords if pitch_coords is not None else (None, None)
                candidate = TrackedBall(
                    pixel_x=center[0],
                    pixel_y=center[1],
                    pitch_x=px,
                    pitch_y=py,
                    confidence=trk.confidence,
                    is_estimated=True,
                    is_usable=False,
                )
                if tracked_ball is None or candidate.confidence > tracked_ball.confidence:
                    tracked_ball = candidate
            else:
                # Player ground contact point is bottom-center of bounding box
                foot_pt = trk.bbox.bottom_center
                pitch_coords = (
                    self.calibrator.pixel_to_pitch(foot_pt[0], foot_pt[1])
                    if self.calibrator is not None else None
                )
                px, py = pitch_coords if pitch_coords is not None else (None, None)
                is_on_pitch = (
                    self.calibrator is not None and px is not None and py is not None
                    and self.calibrator.is_within_pitch(px, py, margin_m=3.0)
                )

                # Team classification via jersey color
                jersey_color = (
                    self.team_classifier.extract_jersey_color_hsv(frame_rgb, trk.bbox)
                    if frame_rgb is not None
                    else None
                )
                team_side = (
                    TeamSide.REFEREE if trk.class_name == DetectedClass.REFEREE
                    else self.team_classifier.classify_player(jersey_color, track_id=trk.track_id)
                )

                # Velocity calculation
                speed_m_s: float | None = None
                valid_position = px is not None and py is not None and is_on_pitch
                accept_for_speed = valid_position
                if valid_position and trk.track_id in self._prev_player_positions:
                    prev_x, prev_y, prev_t = self._prev_player_positions[trk.track_id]
                    dt = timestamp_s - prev_t
                    if dt > 0.0:
                        dx = px - prev_x
                        dy = py - prev_y
                        inst_speed = math.sqrt(dx * dx + dy * dy) / dt
                        if inst_speed <= 12.0:  # screening threshold, not a validated measurement
                            speed_m_s = inst_speed
                        else:
                            accept_for_speed = False

                if accept_for_speed:
                    self._prev_player_positions[trk.track_id] = (px, py, timestamp_s)

                tracked_players.append(TrackedPlayer(
                    track_id=trk.track_id,
                    team_side=team_side,
                    detected_class=trk.class_name,
                    jersey_number=None,
                    bbox=trk.bbox,
                    confidence=trk.confidence,
                    pixel_x=foot_pt[0],
                    pixel_y=foot_pt[1],
                    pitch_x=px,
                    pitch_y=py,
                    speed_m_s=speed_m_s,
                    is_estimated=True,
                    is_usable=is_on_pitch,
                ))

        vision_frame = VisionFrame(
            frame_num=frame_num,
            timestamp_s=timestamp_s,
            players=tracked_players,
            ball=tracked_ball,
            fps=self.fps,
        )
        self._last_frame_num = frame_num
        self._last_timestamp_s = timestamp_s
        return vision_frame

    def process_video_generator(
        self,
        video_path: str,
        max_frames: int | None = None,
        max_seconds: float | None = None,
        callback: Callable[[int, int], None] | None = None,
    ) -> Generator[VisionFrame, None, None]:
        """Stream and process frames from a video file using OpenCV."""
        import cv2

        if max_frames is not None and max_frames < 0:
            raise ValueError("max_frames must be non-negative or None.")
        if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
            raise ValueError("max_seconds must be finite and greater than zero.")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = float(cap.get(cv2.CAP_PROP_FPS))
        self._last_video_path = str(video_path)
        self._video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.set_video_fps(video_fps if math.isfinite(video_fps) and video_fps > 0 else self.fps)

        frame_idx = 0
        try:
            while True:
                if max_frames is not None and frame_idx >= max_frames:
                    break
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                if max_seconds is not None and frame_idx / self.fps >= max_seconds:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                vf = self.process_frame(frame_rgb, frame_num=frame_idx, timestamp_s=frame_idx / self.fps)
                yield vf

                frame_idx += 1
                if callback is not None:
                    callback(frame_idx, total_frames)

        finally:
            cap.release()
