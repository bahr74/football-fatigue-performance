"""Team classification based on jersey color clustering."""

from __future__ import annotations

import math
from collections import Counter, deque
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


class KitColorClassifier:
    """Track-level kit-colour label: ``blue``, ``white``, ``other`` or ``unknown``.

    This is a colour label, not a home/away identity.

    * ``other``: goalkeeper/referee-style kits (yellow, red, dark torso pixels
      dominate). Never assigned to either team.
    * ``blue`` / ``white``: torso saturation (grass removed) split by an
      *adaptive* 1-D 2-means threshold over recent boxes of all players, so
      lighting drift across the match (white shirts looked bluish in shade)
      does not break a fixed threshold. The more saturated cluster is called
      ``blue``; this assumes a blue-vs-white fixture. If the two clusters are
      not clearly separated, or a box lies in the ambiguous band, no vote is cast.
    * ``unknown``: not enough votes, or no clear majority over the track's
      recent frames.
    """

    def __init__(
        self,
        min_observations: int = 8,
        other_min: float = 0.30,
        window: int = 40,
        min_share: float = 0.6,
        buffer_size: int = 800,
        min_buffer: int = 100,
        min_separation: float = 50.0,
        band: float = 0.12,
        refit_every: int = 20,
    ) -> None:
        self.min_observations = min_observations
        self.other_min = other_min
        self.window = window
        self.min_share = min_share
        self.min_buffer = min_buffer
        self.min_separation = min_separation
        self.band = band
        self.refit_every = refit_every
        self._sat_buffer: deque[float] = deque(maxlen=buffer_size)
        self._since_fit = 0
        self._threshold: float | None = None
        self._half_band = 0.0
        self.separation = 0.0
        self._state: dict[int, deque] = {}
        self.last_vote: str | None = None  # raw per-frame vote of the most recent observe()

    def torso_features(self, frame_rgb: Any, bbox: BoundingBox) -> tuple[float, float, float, float] | None:
        """(median saturation, yellow, red, dark fractions) of non-grass torso pixels, or None."""
        try:
            import cv2
            import numpy as np

            h, w = frame_rgb.shape[:2]
            bw, bh = bbox.x2 - bbox.x1, bbox.y2 - bbox.y1
            y1, y2 = max(0, int(bbox.y1 + 0.18 * bh)), min(h, int(bbox.y1 + 0.55 * bh))
            x1, x2 = max(0, int(bbox.x1 + 0.25 * bw)), min(w, int(bbox.x2 - 0.25 * bw))
            if y2 - y1 < 3 or x2 - x1 < 2:
                return None
            hsv = cv2.cvtColor(frame_rgb[y1:y2, x1:x2], cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(int)
            hue, sat, val = hsv[:, 0], hsv[:, 1], hsv[:, 2]
            keep = ~((hue >= 35) & (hue <= 85) & (sat >= 40))  # drop grass
            if keep.sum() < 8:
                return None
            hue, sat, val = hue[keep], sat[keep], val[keep]
            yellow = ((hue >= 18) & (hue <= 35) & (sat >= 100) & (val >= 120)).mean()
            red = (((hue <= 8) | (hue >= 170)) & (sat >= 100) & (val >= 80)).mean()
            dark = (val <= 60).mean()
            return float(np.median(sat)), float(yellow), float(red), float(dark)
        except Exception:
            return None

    def _refit(self) -> None:
        import numpy as np

        x = np.fromiter(self._sat_buffer, dtype=float)
        if len(x) < self.min_buffer:
            self._threshold = None
            return
        lo, hi = np.percentile(x, 25), np.percentile(x, 75)
        for _ in range(20):  # 1-D 2-means
            thr = (lo + hi) / 2
            a, b = x[x <= thr], x[x > thr]
            if len(a) == 0 or len(b) == 0:
                break
            lo, hi = a.mean(), b.mean()
        self.separation = float(hi - lo)
        if self.separation < self.min_separation:
            self._threshold = None  # no clear two-colour split in view
            return
        self._threshold = float((lo + hi) / 2)
        self._half_band = self.band * self.separation

    def _frame_label(self, feats: tuple[float, float, float, float]) -> str | None:
        sat, yellow, red, dark = feats
        if max(yellow, red, dark) >= self.other_min:
            return "other"
        if self._threshold is None:
            return None
        if sat >= self._threshold + self._half_band:
            return "blue"
        if sat <= self._threshold - self._half_band:
            return "white"
        return None

    def observe(self, track_id: int, frame_rgb: Any, bbox: BoundingBox) -> tuple[str, float]:
        """Add one observation for a track and return (label, confidence 0-1)."""
        feats = self.torso_features(frame_rgb, bbox) if frame_rgb is not None else None
        votes = self._state.setdefault(track_id, deque(maxlen=self.window))
        if feats is not None:
            if max(feats[1:]) < self.other_min:
                self._sat_buffer.append(feats[0])
                self._since_fit += 1
                if self._since_fit >= self.refit_every or self._threshold is None and len(self._sat_buffer) == self.min_buffer:
                    self._since_fit = 0
                    self._refit()
            self.last_vote = self._frame_label(feats)
            votes.append(self.last_vote)
        else:
            self.last_vote = None
        return self.label(track_id)

    def label(self, track_id: int) -> tuple[str, float]:
        """Plurality vote over the track's recent frames; ``unknown`` without a clear majority."""
        votes = self._state.get(track_id)
        if not votes:
            return "unknown", 0.0
        decided = [v for v in votes if v is not None]
        if len(decided) < self.min_observations:
            return "unknown", 0.0
        best, n = Counter(decided).most_common(1)[0]
        share = n / len(votes)  # undecided frames count against the label
        if share < self.min_share:
            return "unknown", 0.0
        return best, round(share, 3)
