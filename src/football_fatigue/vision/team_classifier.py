"""Team classification based on jersey color clustering."""

from __future__ import annotations

import math
from typing import Any, Sequence
from football_fatigue.vision.models import BoundingBox, TeamSide


class TeamColorClassifier:
    """Classifies detected players into Home, Away, or Referee based on jersey colors."""

    def __init__(self, home_color_name: str = "Home", away_color_name: str = "Away") -> None:
        self.home_color_name = home_color_name
        self.away_color_name = away_color_name
        self.home_center_hsv: tuple[float, float, float] | None = None
        self.away_center_hsv: tuple[float, float, float] | None = None
        self.cluster_centers_hsv: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
        self._cluster_labels: list[int] | None = None
        self._track_team_history: dict[int, list[TeamSide]] = {}

    def extract_jersey_color_hsv(
        self,
        frame_rgb: Any,  # numpy array (H, W, 3) or None
        bbox: BoundingBox,
    ) -> tuple[float, float, float] | None:
        """Extract average HSV color of the player's jersey (upper half of bounding box)."""
        if frame_rgb is None:
            return None

        try:
            import cv2
            import numpy as np

            h, w = frame_rgb.shape[:2]
            x1 = max(0, int(bbox.x1))
            y1 = max(0, int(bbox.y1))
            x2 = min(w, int(bbox.x2))
            y2 = min(h, int(bbox.y1 + bbox.height * 0.50))  # top half = jersey

            if x2 <= x1 or y2 <= y1:
                return None

            jersey_crop = frame_rgb[y1:y2, x1:x2]
            hsv = cv2.cvtColor(jersey_crop, cv2.COLOR_RGB2HSV)

            # Filter out grass green pixels: Hue in [35, 85], Saturation > 40
            mask_non_green = ~((hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85) & (hsv[:, :, 1] >= 40))
            valid_pixels = hsv[mask_non_green]

            if len(valid_pixels) < 10:
                # If too few non-green pixels, take the central region
                valid_pixels = hsv.reshape(-1, 3)

            median_hsv = np.median(valid_pixels, axis=0)
            return (float(median_hsv[0]), float(median_hsv[1]), float(median_hsv[2]))
        except Exception:
            return None

    def fit_teams(self, samples_hsv: Sequence[tuple[float, float, float]]) -> list[int]:
        """Cluster jersey samples; labels are arbitrary until home is identified."""
        if len(samples_hsv) < 2:
            self.cluster_centers_hsv = None
            self._cluster_labels = None
            self.home_center_hsv = None
            self.away_center_hsv = None
            raise ValueError("At least two jersey color samples are required to cluster teams.")

        try:
            import numpy as np
            from sklearn.cluster import KMeans

            data = np.array(samples_hsv)
            if data.ndim != 2 or data.shape[1] != 3 or not np.isfinite(data).all():
                raise ValueError("Jersey samples must be finite HSV triples.")
            if np.any(data[:, 0] < 0) or np.any(data[:, 0] > 180):
                raise ValueError("OpenCV hue samples must be within 0-180.")
            if np.any(data[:, 1:] < 0) or np.any(data[:, 1:] > 255):
                raise ValueError("OpenCV saturation and value samples must be within 0-255.")
            # Embed hue on a circle so red near 0 and 180 stays nearby.
            hue_radians = data[:, 0] * math.pi / 90.0
            hue_scale = 180.0 / math.pi
            weighted_data = np.column_stack((
                np.cos(hue_radians) * hue_scale,
                np.sin(hue_radians) * hue_scale,
                data[:, 1] * 1.5,
                data[:, 2] * 0.8,
            ))

            kmeans = KMeans(n_clusters=2, random_state=42, n_init=5)
            kmeans.fit(weighted_data)

            centers = []
            for cx, cy, saturation, value in kmeans.cluster_centers_:
                hue = (math.atan2(cy, cx) % (2 * math.pi)) * 90.0 / math.pi
                centers.append((hue, saturation / 1.5, value / 0.8))
            c1, c2 = centers
            self.cluster_centers_hsv = ((c1[0], c1[1], c1[2]), (c2[0], c2[1], c2[2]))
            self._cluster_labels = [int(label) for label in kmeans.labels_]
            # Cluster numbers have no intrinsic home/away meaning.
            self.home_center_hsv = None
            self.away_center_hsv = None
            return list(self._cluster_labels)
        except ImportError as exc:
            self.cluster_centers_hsv = None
            self._cluster_labels = None
            self.home_center_hsv = None
            self.away_center_hsv = None
            raise RuntimeError("NumPy and scikit-learn are required for jersey clustering.") from exc

    def map_cluster_to_home(self, cluster_id: int) -> None:
        """Apply the user's verified home/away mapping to the two color clusters."""
        if self.cluster_centers_hsv is None:
            raise RuntimeError("Fit jersey clusters before mapping one to home.")
        if cluster_id not in (0, 1):
            raise ValueError("cluster_id must be 0 or 1.")
        self.home_center_hsv = self.cluster_centers_hsv[cluster_id]
        self.away_center_hsv = self.cluster_centers_hsv[1 - cluster_id]

    def classify_player(
        self,
        color_hsv: tuple[float, float, float] | None,
        track_id: int | None = None,
    ) -> TeamSide:
        """Classify a single player color sample into HOME or AWAY."""
        if color_hsv is None:
            # Use majority vote from history if available
            if track_id is not None and track_id in self._track_team_history:
                return self._majority_side(self._track_team_history[track_id])
            return TeamSide.UNKNOWN

        if self.home_center_hsv is None or self.away_center_hsv is None:
            return TeamSide.UNKNOWN

        d_home = self._hsv_distance(color_hsv, self.home_center_hsv)
        d_away = self._hsv_distance(color_hsv, self.away_center_hsv)

        if math.isclose(d_home, d_away, abs_tol=1e-9):
            return TeamSide.UNKNOWN
        side = TeamSide.HOME if d_home < d_away else TeamSide.AWAY

        # Update history for temporal smoothing
        if track_id is not None:
            if track_id not in self._track_team_history:
                self._track_team_history[track_id] = []
            hist = self._track_team_history[track_id]
            hist.append(side)
            if len(hist) > 30:
                hist.pop(0)
            return self._majority_side(hist)

        return side

    @staticmethod
    def _hsv_distance(c1: tuple[float, float, float], c2: tuple[float, float, float]) -> float:
        """Angular distance for hue (0-180 in OpenCV) + Euclidean for saturation & value."""
        # Hue difference on a circle of diameter 180
        h_diff = abs(c1[0] - c2[0])
        h_dist = min(h_diff, 180.0 - h_diff) * 2.0  # normalize scale
        s_dist = abs(c1[1] - c2[1])
        v_dist = abs(c1[2] - c2[2]) * 0.5
        return math.sqrt(h_dist * h_dist + s_dist * s_dist + v_dist * v_dist)

    @staticmethod
    def _majority_side(history: list[TeamSide]) -> TeamSide:
        if not history:
            return TeamSide.UNKNOWN
        home_count = sum(1 for s in history if s == TeamSide.HOME)
        away_count = sum(1 for s in history if s == TeamSide.AWAY)
        if home_count == away_count:
            return TeamSide.UNKNOWN
        return TeamSide.HOME if home_count > away_count else TeamSide.AWAY
