"""Pitch Calibration and Homography Module.

Maps camera image pixel coordinates (u, v) to real-world pitch coordinates (X, Y)
in meters, referenced to the standard pitch-centre Cartesian system:
- Origin (0, 0) is at the center spot.
- Pitch length (X-axis) runs from -length/2 to +length/2 (standard: -52.5m to +52.5m).
- Pitch width (Y-axis) runs from -width/2 to +width/2 (standard: -34.0m to +34.0m).
- Standard pitch dimensions: 105.0m x 68.0m.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

# Standard FIFA pitch landmarks in meters relative to pitch-centre (0, 0)
PITCH_LANDMARKS_105x68: dict[str, tuple[float, float]] = {
    "center_spot": (0.0, 0.0),
    "halfway_line_top": (0.0, 34.0),
    "halfway_line_bottom": (0.0, -34.0),
    "center_circle_top": (0.0, 9.15),
    "center_circle_bottom": (0.0, -9.15),
    "corner_top_left": (-52.5, 34.0),
    "corner_bottom_left": (-52.5, -34.0),
    "corner_top_right": (52.5, 34.0),
    "corner_bottom_right": (52.5, -34.0),
    "penalty_spot_left": (-41.5, 0.0),
    "penalty_spot_right": (41.5, 0.0),
    "penalty_box_left_top": (-36.0, 20.16),
    "penalty_box_left_bottom": (-36.0, -20.16),
    "penalty_box_right_top": (36.0, 20.16),
    "penalty_box_right_bottom": (36.0, -20.16),
    "six_yard_left_top": (-47.0, 9.16),
    "six_yard_left_bottom": (-47.0, -9.16),
    "six_yard_right_top": (47.0, 9.16),
    "six_yard_right_bottom": (47.0, -9.16),
    "goal_left": (-52.5, 0.0),
    "goal_right": (52.5, 0.0),
}


@dataclass(frozen=True)
class LandmarkPair:
    """A calibration point pairing a pixel position to a pitch coordinate."""
    name: str
    pixel_x: float
    pixel_y: float
    pitch_x_m: float
    pitch_y_m: float


class HomographyCalibrator:
    """Calculates and applies 2D planar homography between image pixels and pitch coordinates in meters."""

    def __init__(
        self,
        pitch_length_m: float = 105.0,
        pitch_width_m: float = 68.0,
        matrix_3x3: Sequence[Sequence[float]] | None = None,
        ransac_reprojection_threshold_m: float = 0.25,
    ) -> None:
        self.pitch_length_m = float(pitch_length_m)
        self.pitch_width_m = float(pitch_width_m)
        if not math.isfinite(self.pitch_length_m) or self.pitch_length_m <= 0:
            raise ValueError("pitch_length_m must be finite and greater than zero.")
        if not math.isfinite(self.pitch_width_m) or self.pitch_width_m <= 0:
            raise ValueError("pitch_width_m must be finite and greater than zero.")
        self.ransac_reprojection_threshold_m = float(ransac_reprojection_threshold_m)
        if not math.isfinite(self.ransac_reprojection_threshold_m) or self.ransac_reprojection_threshold_m <= 0:
            raise ValueError("ransac_reprojection_threshold_m must be finite and greater than zero.")
        self._matrix: list[list[float]] | None = [list(row) for row in matrix_3x3] if matrix_3x3 else None
        self._inv_matrix: list[list[float]] | None = None
        if self._matrix:
            if len(self._matrix) != 3 or any(len(row) != 3 for row in self._matrix):
                raise ValueError("matrix_3x3 must have shape 3 by 3.")
            if not all(math.isfinite(value) for row in self._matrix for value in row):
                raise ValueError("matrix_3x3 values must be finite.")
            self._compute_inverse()
            if self._inv_matrix is None:
                raise ValueError("matrix_3x3 must be invertible.")

    @property
    def is_calibrated(self) -> bool:
        return self._matrix is not None

    @property
    def matrix(self) -> list[list[float]] | None:
        return self._matrix

    def reprojection_errors_m(
        self,
        image_points: Sequence[tuple[float, float]],
        pitch_points: Sequence[tuple[float, float]],
    ) -> list[float]:
        """Return Euclidean image-to-pitch residuals for supplied correspondences."""
        if len(image_points) != len(pitch_points):
            raise ValueError("image_points and pitch_points must have identical length.")
        if not self.is_calibrated:
            raise RuntimeError("Calibrator is not yet calibrated.")
        errors = []
        for (u, v), (expected_x, expected_y) in zip(image_points, pitch_points):
            projected = self._project(self._matrix, float(u), float(v))
            if projected is None:
                raise ValueError("A calibration point projected to an invalid coordinate.")
            errors.append(math.hypot(projected[0] - expected_x, projected[1] - expected_y))
        return errors

    @staticmethod
    def _project(
        matrix: Sequence[Sequence[float]], u: float, v: float
    ) -> tuple[float, float] | None:
        x_prime = matrix[0][0] * u + matrix[0][1] * v + matrix[0][2]
        y_prime = matrix[1][0] * u + matrix[1][1] * v + matrix[1][2]
        w = matrix[2][0] * u + matrix[2][1] * v + matrix[2][2]
        if abs(w) < 1e-8:
            return None
        x, y = x_prime / w, y_prime / w
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        return x, y

    def fit_from_points(
        self,
        image_points: Sequence[tuple[float, float]],
        pitch_points: Sequence[tuple[float, float]],
    ) -> list[list[float]]:
        """Compute the 3x3 homography matrix H mapping image (u, v) -> pitch (X, Y).
        
        Requires at least 4 non-collinear point correspondences.
        """
        if len(image_points) < 4 or len(pitch_points) < 4:
            raise ValueError(f"At least 4 point pairs required for homography, got {len(image_points)}.")
        if len(image_points) != len(pitch_points):
            raise ValueError("image_points and pitch_points must have identical length.")
        if not all(math.isfinite(float(v)) for point in (*image_points, *pitch_points) for v in point):
            raise ValueError("Calibration coordinates must be finite numbers.")
        if self._points_collinear(image_points) or self._points_collinear(pitch_points):
            raise ValueError("Calibration points must span two dimensions; collinear points are degenerate.")

        # Try using OpenCV if available
        try:
            import cv2
            import numpy as np

            src = np.array(image_points, dtype=np.float32).reshape(-1, 1, 2)
            dst = np.array(pitch_points, dtype=np.float32).reshape(-1, 1, 2)
            h, status = cv2.findHomography(
                src, dst, cv2.RANSAC, self.ransac_reprojection_threshold_m
            )
            if h is None:
                # Fall back to standard method if RANSAC fails with few points
                h, _ = cv2.findHomography(src, dst, 0)
            if h is not None:
                if not np.isfinite(h).all() or abs(float(np.linalg.det(h))) < 1e-12:
                    raise ValueError("Calibration points produced a singular homography.")
                self._matrix = h.tolist()
                self._compute_inverse()
                return self._matrix
        except ImportError:
            pass

        # Fallback pure-math Direct Linear Transformation (DLT) solver
        self._matrix = self._solve_dlt(image_points, pitch_points)
        self._compute_inverse()
        if self._inv_matrix is None:
            self._matrix = None
            raise ValueError("Calibration points are degenerate; homography is not invertible.")
        return self._matrix

    @staticmethod
    def _points_collinear(points: Sequence[tuple[float, float]], tolerance: float = 1e-9) -> bool:
        """Detect point sets whose spatial extent is only one-dimensional."""
        unique = list(dict.fromkeys((float(x), float(y)) for x, y in points))
        if len(unique) < 4:
            return True
        x0, y0 = unique[0]
        x1, y1 = next(
            ((x, y) for x, y in unique[1:] if math.hypot(x - x0, y - y0) > tolerance),
            unique[0],
        )
        if (x0, y0) == (x1, y1):
            return True
        for x, y in unique[2:]:
            area_twice = abs((x1 - x0) * (y - y0) - (y1 - y0) * (x - x0))
            baseline = math.hypot(x1 - x0, y1 - y0)
            if area_twice / baseline > tolerance:
                return False
        return True

    def pixel_to_pitch(self, u: float, v: float) -> tuple[float, float] | None:
        """Transform image coordinates (u, v) to pitch coordinates (X, Y) in meters."""
        if self._matrix is None:
            raise RuntimeError("Calibrator is not yet calibrated. Call fit_from_points first.")
        if not math.isfinite(float(u)) or not math.isfinite(float(v)):
            return None

        projected = self._project(self._matrix, u, v)
        if projected is None:
            return None
        x, y = projected
        return (round(x, 3), round(y, 3))

    def pitch_to_pixel(self, x: float, y: float) -> tuple[float, float] | None:
        """Transform pitch coordinates (X, Y) in meters back to image pixel coordinates (u, v)."""
        if self._inv_matrix is None:
            if self._matrix is None:
                raise RuntimeError("Calibrator is not yet calibrated.")
            self._compute_inverse()
        if not math.isfinite(float(x)) or not math.isfinite(float(y)):
            return None

        inv = self._inv_matrix
        if inv is None:
            return None

        u_prime = inv[0][0] * x + inv[0][1] * y + inv[0][2]
        v_prime = inv[1][0] * x + inv[1][1] * y + inv[1][2]
        w = inv[2][0] * x + inv[2][1] * y + inv[2][2]

        if abs(w) < 1e-8:
            return None

        u = u_prime / w
        v = v_prime / w
        if not math.isfinite(u) or not math.isfinite(v):
            return None
        return (round(u, 1), round(v, 1))

    def is_within_pitch(self, x: float, y: float, margin_m: float = 3.0) -> bool:
        """Check if a coordinate lies on or near the pitch boundaries."""
        if not all(math.isfinite(float(v)) for v in (x, y, margin_m)) or margin_m < 0:
            return False
        half_l = (self.pitch_length_m / 2.0) + margin_m
        half_w = (self.pitch_width_m / 2.0) + margin_m
        return (-half_l <= x <= half_l) and (-half_w <= y <= half_w)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pitch_length_m": self.pitch_length_m,
            "pitch_width_m": self.pitch_width_m,
            "ransac_reprojection_threshold_m": self.ransac_reprojection_threshold_m,
            "is_calibrated": self.is_calibrated,
            "matrix": self._matrix,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HomographyCalibrator:
        return cls(
            pitch_length_m=data.get("pitch_length_m", 105.0),
            pitch_width_m=data.get("pitch_width_m", 68.0),
            matrix_3x3=data.get("matrix"),
            ransac_reprojection_threshold_m=data.get("ransac_reprojection_threshold_m", 0.25),
        )

    def _compute_inverse(self) -> None:
        """Compute the 3x3 inverse matrix using cofactor formula."""
        m = self._matrix
        if m is None:
            return
        det = (
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        )
        if abs(det) < 1e-12:
            self._inv_matrix = None
            return

        inv_det = 1.0 / det
        inv = [
            [
                (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
                (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
                (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
            ],
            [
                (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
                (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
                (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
            ],
            [
                (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
                (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
                (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
            ],
        ]
        self._inv_matrix = inv

    @staticmethod
    def _solve_dlt(
        img_pts: Sequence[tuple[float, float]],
        pitch_pts: Sequence[tuple[float, float]],
    ) -> list[list[float]]:
        """Pure-Python Direct Linear Transformation solver for 4+ point pairs."""
        try:
            import numpy as np

            # Formulate Ah = 0
            a_rows = []
            for (u, v), (x, y) in zip(img_pts, pitch_pts):
                a_rows.append([-u, -v, -1.0, 0.0, 0.0, 0.0, u * x, v * x, x])
                a_rows.append([0.0, 0.0, 0.0, -u, -v, -1.0, u * y, v * y, y])

            a = np.array(a_rows, dtype=np.float64)
            # SVD decomposition
            _, _, vh = np.linalg.svd(a)
            # Last row of Vh corresponds to smallest singular value
            h = vh[-1].reshape(3, 3)
            # Normalize h so h[2, 2] = 1.0 if possible
            if abs(h[2, 2]) > 1e-8:
                h = h / h[2, 2]
            return h.tolist()
        except ImportError as exc:
            raise RuntimeError("NumPy is required to calculate a homography without OpenCV.") from exc


def pitch_polygon_from_keypoints(
    keypoints_path: str,
    margin_px: int = 60,
    frame_size: tuple[int, int] = (4096, 1080),
) -> list[tuple[float, float]]:
    """Image-space pitch outline (convex hull of labelled pitch keypoints, dilated).

    Used only as an image-space false-positive mask; it is not a metric
    calibration. ``margin_px`` keeps players standing on or just outside the
    touchlines.
    """
    import json

    import cv2
    import numpy as np

    with open(keypoints_path, encoding="utf-8") as stream:
        raw = json.load(stream)
    pts = np.array([v for v in raw.values()], dtype=np.float32)
    if len(pts) < 4:
        raise ValueError("At least four keypoints are required for a pitch outline.")
    width, height = frame_size
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(pts).astype(np.int32), 255)
    if margin_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin_px + 1, 2 * margin_px + 1))
        mask = cv2.dilate(mask, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    contour = cv2.approxPolyDP(contour, 2.0, True)
    return [(float(x), float(y)) for x, y in contour.reshape(-1, 2)]
