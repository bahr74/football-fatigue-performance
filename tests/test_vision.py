"""Tests for the Computer Vision Tracking and Pitch Calibration module."""

import math
import pytest
from football_fatigue.vision.calibration import HomographyCalibrator
from football_fatigue.vision.detector import YOLOPlayerDetector
from football_fatigue.vision.models import (
    BoundingBox,
    DetectedClass,
    Detection,
    TeamSide,
    TrackedPlayer,
    VisionFrame,
)
from football_fatigue.vision.pipeline import VisionTrackingPipeline
from football_fatigue.vision.pipeline import player_trajectory_points
from football_fatigue.vision.team_classifier import TeamColorClassifier
from football_fatigue.vision.tracker import MultiObjectTracker
from football_fatigue.vision.__main__ import main as vision_cli_main
from football_fatigue.analytics.cv import build_cv_analytics, summarize_vision_movement


class TestHomographyCalibrator:
    def test_fit_and_transform_four_corners(self):
        calibrator = HomographyCalibrator(pitch_length_m=105.0, pitch_width_m=68.0)

        # 4 corners of standard pitch in image pixels:
        # e.g. top-left (100, 100), bottom-left (100, 800), top-right (1200, 100), bottom-right (1200, 800)
        img_points = [(100.0, 100.0), (100.0, 800.0), (1200.0, 100.0), (1200.0, 800.0)]
        pitch_points = [(-52.5, 34.0), (-52.5, -34.0), (52.5, 34.0), (52.5, -34.0)]

        calibrator.fit_from_points(img_points, pitch_points)
        assert calibrator.is_calibrated

        # Test center spot (midpoint of corners: (650, 450) -> should be close to (0, 0))
        center_pitch = calibrator.pixel_to_pitch(650.0, 450.0)
        assert center_pitch is not None
        assert math.isclose(center_pitch[0], 0.0, abs_tol=0.1)
        assert math.isclose(center_pitch[1], 0.0, abs_tol=0.1)

        # Test round-trip projection
        pixel_back = calibrator.pitch_to_pixel(0.0, 0.0)
        assert pixel_back is not None
        assert math.isclose(pixel_back[0], 650.0, abs_tol=1.0)
        assert math.isclose(pixel_back[1], 450.0, abs_tol=1.0)

    def test_pitch_boundaries(self):
        calibrator = HomographyCalibrator(pitch_length_m=105.0, pitch_width_m=68.0)
        assert calibrator.is_within_pitch(0.0, 0.0)
        assert calibrator.is_within_pitch(50.0, 30.0)
        assert calibrator.is_within_pitch(53.0, 35.0, margin_m=3.0)
        assert not calibrator.is_within_pitch(60.0, 40.0, margin_m=2.0)

    def test_serialization(self):
        calibrator = HomographyCalibrator(pitch_length_m=100.0, pitch_width_m=64.0)
        calibrator._matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        d = calibrator.to_dict()
        assert d["is_calibrated"] is True
        loaded = HomographyCalibrator.from_dict(d)
        assert loaded.pitch_length_m == 100.0
        assert loaded.matrix == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        assert loaded.ransac_reprojection_threshold_m == pytest.approx(0.25)

    def test_reprojection_errors_report_pitch_space_residuals(self):
        calibrator = HomographyCalibrator(matrix_3x3=[
            [0.1, 0.0, -10.0], [0.0, -0.1, 40.0], [0.0, 0.0, 1.0],
        ])
        errors = calibrator.reprojection_errors_m(
            [(100, 100), (200, 200)], [(0, 30), (11, 20)],
        )
        assert errors == pytest.approx([0.0, 1.0])
        with pytest.raises(ValueError, match="greater than zero"):
            HomographyCalibrator(ransac_reprojection_threshold_m=0)

    def test_rejects_degenerate_or_non_finite_calibration_points(self):
        calibrator = HomographyCalibrator()
        with pytest.raises(ValueError, match="At least 4"):
            calibrator.fit_from_points([(0, 0), (1, 0), (0, 1)], [(0, 0), (1, 0), (0, 1)])
        with pytest.raises(ValueError, match="finite"):
            calibrator.fit_from_points(
                [(0, 0), (1, 0), (0, 1), (float("nan"), 1)],
                [(0, 0), (1, 0), (0, 1), (1, 1)],
            )
        with pytest.raises(ValueError, match="invertible"):
            HomographyCalibrator(matrix_3x3=[[1, 0, 0], [0, 0, 0], [0, 0, 1]])
        with pytest.raises(ValueError, match="collinear"):
            calibrator.fit_from_points(
                [(0, 0), (1, 0), (2, 0), (3, 0)],
                [(0, 0), (1, 0), (2, 0), (3, 0)],
            )


class TestBoundingBox:
    def test_geometry(self):
        box = BoundingBox(x1=100.0, y1=200.0, x2=150.0, y2=300.0)
        assert box.width == 50.0
        assert box.height == 100.0
        assert box.center == (125.0, 250.0)
        assert box.bottom_center == (125.0, 300.0)

    def test_rejects_invalid_geometry_and_confidence(self):
        with pytest.raises(ValueError, match="positive width"):
            BoundingBox(1, 1, 1, 2)
        with pytest.raises(ValueError, match="confidence"):
            Detection(DetectedClass.PLAYER, 1.1, BoundingBox(0, 0, 1, 1))


class TestYOLOClassMapping:
    def test_field_turf_box_is_rejected(self):
        import numpy as np

        field = np.zeros((100, 100, 3), dtype=np.uint8)
        field[:, :] = (40, 150, 40)

        assert YOLOPlayerDetector._looks_like_player(
            field, BoundingBox(20, 20, 40, 70),
        ) is False

    def test_generic_person_is_not_claimed_to_be_a_player(self):
        assert YOLOPlayerDetector._map_class(0, {0: "person"}) == DetectedClass.PERSON
        assert YOLOPlayerDetector._map_class(32, {32: "sports ball"}) == DetectedClass.BALL
        assert YOLOPlayerDetector._map_class(3, {3: "referee"}) == DetectedClass.REFEREE
        assert YOLOPlayerDetector._map_class(8, {8: "unknown object"}) is None

    def test_external_football_checkpoint_label_map(self):
        names = {0: "ball", 1: "player", 2: "referee", 3: "goalkeeper"}
        assert [YOLOPlayerDetector._map_class(idx, names) for idx in range(4)] == [
            DetectedClass.BALL,
            DetectedClass.PLAYER,
            DetectedClass.REFEREE,
            DetectedClass.GOALKEEPER,
        ]


class TestMultiObjectTracker:
    def test_backend_is_explicit_and_rate_is_validated(self):
        tracker = MultiObjectTracker(use_supervision=False)
        assert tracker.backend == "iou_fallback"
        with pytest.raises(ValueError, match="frame_rate"):
            MultiObjectTracker(frame_rate=0, use_supervision=False)

    def test_supervision_adapter_preserves_ids_classes_and_boxes(self, monkeypatch):
        import sys
        import types
        import numpy as np

        class FakeDetections:
            def __init__(self, xyxy, confidence, class_id):
                self.xyxy, self.confidence, self.class_id = xyxy, confidence, class_id
                self.tracker_id = np.array([77])

            def __len__(self):
                return len(self.xyxy)

            @classmethod
            def empty(cls):
                return cls(np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int))

        class FakeByteTrack:
            def __init__(self, **kwargs):
                assert kwargs["frame_rate"] == 29.97

            def update_with_detections(self, detections):
                return detections

        monkeypatch.setitem(sys.modules, "supervision", types.SimpleNamespace(
            ByteTrack=FakeByteTrack, Detections=FakeDetections,
        ))
        tracker = MultiObjectTracker(frame_rate=29.97)
        assert tracker.backend == "bytetrack"
        detection = Detection(DetectedClass.REFEREE, 0.9, BoundingBox(2, 3, 8, 12))
        result = tracker.update([detection], frame_idx=0)
        assert result[0].track_id == 77
        assert result[0].class_name == DetectedClass.REFEREE
        assert result[0].bbox == detection.bbox

    def test_set_frame_rate_reinitializes_bytetrack(self, monkeypatch):
        import sys
        import types

        requested_rates = []

        class FakeByteTrack:
            def __init__(self, **kwargs):
                requested_rates.append(kwargs["frame_rate"])

        monkeypatch.setitem(sys.modules, "supervision", types.SimpleNamespace(ByteTrack=FakeByteTrack))
        tracker = MultiObjectTracker(frame_rate=25.0)
        tracker.set_frame_rate(29.97)
        assert requested_rates == [25.0, 29.97]

    def test_track_creation_and_matching(self):
        tracker = MultiObjectTracker(max_lost_frames=5, iou_threshold=0.2, use_supervision=False)

        # Frame 1: 2 players
        det1 = Detection(DetectedClass.PLAYER, 0.9, BoundingBox(100, 100, 150, 200))
        det2 = Detection(DetectedClass.PLAYER, 0.85, BoundingBox(300, 300, 350, 400))
        active1 = tracker.update([det1, det2], frame_idx=0)
        assert len(active1) == 2
        id1 = active1[0].track_id
        id2 = active1[1].track_id

        # Frame 2: slightly moved players
        det1_moved = Detection(DetectedClass.PLAYER, 0.92, BoundingBox(105, 102, 155, 202))
        det2_moved = Detection(DetectedClass.PLAYER, 0.88, BoundingBox(302, 301, 352, 401))
        active2 = tracker.update([det1_moved, det2_moved], frame_idx=1)
        assert len(active2) == 2
        assert {t.track_id for t in active2} == {id1, id2}

    def test_preserves_distinct_detected_classes(self):
        tracker = MultiObjectTracker(use_supervision=False)
        detections = [
            Detection(DetectedClass.GOALKEEPER, 0.9, BoundingBox(1, 1, 10, 20)),
            Detection(DetectedClass.REFEREE, 0.9, BoundingBox(20, 1, 30, 20)),
            Detection(DetectedClass.BALL, 0.9, BoundingBox(40, 1, 45, 6)),
        ]
        tracks = tracker.update(detections, frame_idx=0)
        assert [track.class_name for track in tracks] == [
            DetectedClass.GOALKEEPER,
            DetectedClass.REFEREE,
            DetectedClass.BALL,
        ]


class TestTeamColorClassifier:
    def test_classification_and_distance(self):
        classifier = TeamColorClassifier()
        # Mock centers: home = blue (100, 180, 200), away = red (10, 200, 220)
        classifier.home_center_hsv = (100.0, 180.0, 200.0)
        classifier.away_center_hsv = (10.0, 200.0, 220.0)

        # Blue sample
        side_blue = classifier.classify_player((98.0, 175.0, 195.0), track_id=1)
        assert side_blue == TeamSide.HOME

        # Red sample
        side_red = classifier.classify_player((12.0, 190.0, 210.0), track_id=2)
        assert side_red == TeamSide.AWAY

    def test_unknown_when_no_verified_team_color_mapping(self):
        classifier = TeamColorClassifier()
        assert classifier.classify_player(None, track_id=44) == TeamSide.UNKNOWN
        assert classifier.classify_player((100.0, 150.0, 200.0), track_id=44) == TeamSide.UNKNOWN

    def test_cluster_mapping_requires_explicit_home_assignment(self):
        classifier = TeamColorClassifier()
        classifier.cluster_centers_hsv = ((100.0, 180.0, 200.0), (10.0, 200.0, 220.0))
        assert classifier.classify_player((100.0, 180.0, 200.0)) == TeamSide.UNKNOWN
        classifier.map_cluster_to_home(0)
        assert classifier.classify_player((100.0, 180.0, 200.0)) == TeamSide.HOME
        assert classifier.classify_player((10.0, 200.0, 220.0)) == TeamSide.AWAY

    def test_tied_track_history_is_unknown(self):
        assert TeamColorClassifier._majority_side(
            [TeamSide.HOME, TeamSide.AWAY]
        ) == TeamSide.UNKNOWN


class TestVisionTrackingPipeline:
    def test_cli_rejects_missing_inputs_without_creating_output(self, tmp_path):
        output = tmp_path / "out"
        result = vision_cli_main([
            str(tmp_path / "missing.mp4"),
            "--calibration", str(tmp_path / "calibration.json"),
            "--weights", str(tmp_path / "model.pt"),
            "--output-root", str(output),
            "--run-id", "demo",
        ])
        assert result == 2
        assert not output.exists()

    def test_cli_rejects_outputs_inside_raw_data(self, tmp_path):
        from pathlib import Path
        import os

        old_cwd = Path.cwd()
        os.chdir(tmp_path)
        try:
            raw = tmp_path / "data" / "raw"
            raw.mkdir(parents=True)
            inputs = []
            for name in ("video.mp4", "calibration.json", "model.pt"):
                path = tmp_path / name
                path.write_bytes(b"synthetic")
                inputs.append(path)
            result = vision_cli_main([
                str(inputs[0]), "--calibration", str(inputs[1]),
                "--weights", str(inputs[2]), "--output-root", str(raw),
                "--run-id", "bad-output",
            ])
            assert result == 2
            assert list(raw.iterdir()) == []
        finally:
            os.chdir(old_cwd)

    def test_pipeline_single_frame(self):
        calibrator = HomographyCalibrator(pitch_length_m=105.0, pitch_width_m=68.0)
        # Simple identity homography for testing
        calibrator._matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

        pipeline = VisionTrackingPipeline(
            calibrator=calibrator,
            tracker=MultiObjectTracker(use_supervision=False),
        )

        # Provide synthetic detections
        det_player = Detection(
            DetectedClass.PLAYER,
            0.95,
            BoundingBox(10.0, 10.0, 20.0, 30.0),  # bottom-center at (15, 30)
        )
        det_ball = Detection(
            DetectedClass.BALL,
            0.90,
            BoundingBox(0.0, 0.0, 2.0, 2.0),      # center at (1, 1)
        )

        vf = pipeline.process_frame(
            frame_rgb=None,
            frame_num=1,
            timestamp_s=0.04,
            detections=[det_player, det_ball],
        )

        assert isinstance(vf, VisionFrame)
        assert len(vf.players) == 1
        assert vf.ball is not None
        assert math.isclose(vf.players[0].pitch_x, 15.0)
        assert math.isclose(vf.players[0].pitch_y, 30.0)

        d = vf.to_dict()
        assert d["frame_num"] == 1
        assert len(d["players"]) == 1
        assert d["players"][0]["coord_source"] == "cv_tracking"
        assert d["players"][0]["jersey"] is None
        assert d["players"][0]["confidence"] == pytest.approx(0.95)

    def test_unprojectable_positions_remain_missing(self):
        calibrator = HomographyCalibrator(
            matrix_3x3=[[1, 0, 0], [0, 1, 0], [1, 0, 1]],
        )
        pipeline = VisionTrackingPipeline(
            calibrator=calibrator,
            tracker=MultiObjectTracker(use_supervision=False),
        )
        detections = [
            Detection(DetectedClass.PLAYER, 0.9, BoundingBox(-2, -10, 0, 0)),
            Detection(DetectedClass.BALL, 0.9, BoundingBox(-2, 0, 0, 2)),
        ]
        result = pipeline.process_frame(None, frame_num=0, detections=detections)
        output = result.to_dict()
        assert len(output["players"]) == 1
        assert output["players"][0]["x"] is None
        assert output["players"][0]["y"] is None
        assert output["players"][0]["position_usable"] is False
        assert output["ball"]["x"] is None
        assert output["ball"]["y"] is None
        assert output["ball"]["position_usable"] is False

    def test_cv_trajectory_can_feed_existing_movement_analytics(self):
        from football_fatigue.analytics.movement import compute_player_movement
        from football_fatigue.vision.models import TrackedPlayer

        frames = [
            VisionFrame(
                frame_num=idx,
                timestamp_s=idx * 0.04,
                players=[TrackedPlayer(
                    track_id=9,
                    team_side=TeamSide.UNKNOWN,
                    pitch_x=idx * 0.12,
                    pitch_y=0.0,
                    is_usable=True,
                    is_estimated=True,
                )],
            )
            for idx in range(2)
        ]
        points = player_trajectory_points(frames, 9, canonical_period=1)
        metrics, _ = compute_player_movement(points)
        assert len(points) == 2
        assert metrics.total_distance_m == pytest.approx(0.12)
        assert metrics.coverage.estimated_points == 2

    def test_trajectory_adapter_requires_explicit_period_mapping(self):
        with pytest.raises(ValueError, match="explicitly mapped"):
            player_trajectory_points([], 1, canonical_period=0)

    def test_cv_stage_four_uses_all_usable_frames(self):
        frames = [VisionFrame(
            frame_num=index,
            timestamp_s=index * 0.1,
            players=[TrackedPlayer(
                track_id=7,
                team_side=TeamSide.UNKNOWN,
                pitch_x=index * 0.5,
                pitch_y=0.0,
                is_usable=True,
                is_estimated=True,
            )],
        ) for index in range(4)]

        report = summarize_vision_movement(frames)

        assert report["status"] == "computed"
        assert report["frame_count"] == 4
        assert report["tracks"]["7"]["total_distance_m"] == pytest.approx(1.5)

    def test_cv_stage_five_keeps_missing_inputs_unavailable(self):
        frames = [VisionFrame(frame_num=0, timestamp_s=0.0)]

        report = build_cv_analytics(frames)["stage_5_tactical_metrics"]

        assert report["pc"]["status"] == "unavailable"
        assert "stable home/away team attribution" in report["pc"]["reasons"]
        assert report["epv"]["status"] == "unavailable"
        assert report["das"]["status"] == "unavailable"
