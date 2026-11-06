"""Multi-object tracker for the people on the pitch (players, goalkeepers, referees).

The tracker is person-only: ball detections are never associated here (a ball is a different kind
of object with its own motion; see ``VisionTrackingPipeline`` for how ball observations are kept
separately). Association is class-agnostic across the person roles, so a detector label that
flickers between player, goalkeeper and referee never breaks a track. Each track reports a
*role estimate* built from decayed, confidence-weighted label evidence, plus the raw label of the
current frame, so the role is never silently equal to "whatever was detected last".
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Sequence
from football_fatigue.vision.models import BoundingBox, DetectedClass, Detection

PERSON_CLASSES = frozenset({
    DetectedClass.PERSON, DetectedClass.PLAYER, DetectedClass.GOALKEEPER, DetectedClass.REFEREE,
})
ROLE_POLICIES = ("stable", "latest")


@dataclass
class ActiveTrack:
    """Internal state for an active object track."""
    track_id: int
    class_name: DetectedClass
    bbox: BoundingBox
    confidence: float
    age: int = 1
    hits: int = 1
    lost_frames: int = 0
    vx: float = 0.0
    vy: float = 0.0
    observed_class: DetectedClass | None = None  # raw label of the matched detection in this frame
    role_confidence: float | None = None         # share of decayed evidence behind class_name


class RoleEstimator:
    """Per-track role estimate from decayed, confidence-weighted label votes.

    Votes decay with a half-life in seconds (converted with the frame rate), so a few contrary
    frames cannot flip a long-observed role, while sustained contrary evidence eventually does:
    the challenger must also exceed the current role's evidence by ``switch_margin``. Nothing here
    is a verified identity; ``confidence`` is only the current role's share of the evidence.
    """

    def __init__(self, frame_rate: float, half_life_s: float = 2.0, switch_margin: float = 1.25) -> None:
        if not math.isfinite(float(half_life_s)) or half_life_s <= 0:
            raise ValueError("half_life_s must be finite and greater than zero.")
        if not math.isfinite(float(switch_margin)) or switch_margin < 1:
            raise ValueError("switch_margin must be finite and at least 1.")
        self.half_life_frames = float(half_life_s) * float(frame_rate)
        self.switch_margin = float(switch_margin)
        self._state: dict[int, dict[str, Any]] = {}
        self.observations = 0
        self.observed_label_changes = 0   # raw label differs from the track's previous raw label
        self.role_changes = 0             # reported role changed
        self.tracks_with_label_change: set[int] = set()
        self.tracks_with_role_change: set[int] = set()

    def observe(self, track_id: int, label: DetectedClass, confidence: float, frame_idx: int) -> tuple[DetectedClass, float]:
        state = self._state.get(track_id)
        if state is None:
            state = self._state[track_id] = {"votes": {}, "role": label, "last_raw": label, "last_frame": frame_idx}
        else:
            decay = 0.5 ** (max(0, frame_idx - state["last_frame"]) / self.half_life_frames)
            state["votes"] = {k: v * decay for k, v in state["votes"].items()}
            if label != state["last_raw"]:
                self.observed_label_changes += 1
                self.tracks_with_label_change.add(track_id)
        votes = state["votes"]
        votes[label] = votes.get(label, 0.0) + max(float(confidence), 1e-6)
        state["last_raw"], state["last_frame"] = label, frame_idx
        # deterministic: ties resolve to the current role
        best = max(votes, key=lambda k: (votes[k], k == state["role"]))
        if best != state["role"] and votes[best] > self.switch_margin * votes.get(state["role"], 0.0):
            state["role"] = best
            self.role_changes += 1
            self.tracks_with_role_change.add(track_id)
        self.observations += 1
        return state["role"], votes.get(state["role"], 0.0) / sum(votes.values())

    def prune(self, frame_idx: int, max_age_frames: int) -> None:
        for track_id in [t for t, s in self._state.items() if frame_idx - s["last_frame"] > max_age_frames]:
            del self._state[track_id]

    def stats(self) -> dict[str, Any]:
        return {
            "observations": self.observations,
            "observed_label_changes": self.observed_label_changes,
            "role_changes": self.role_changes,
            "tracks_with_label_change": len(self.tracks_with_label_change),
            "tracks_with_role_change": len(self.tracks_with_role_change),
        }


class MultiObjectTracker:
    """Tracks the people on the pitch across consecutive video frames using ByteTrack or IoU tracking."""

    def __init__(
        self,
        max_lost_frames: int = 25,
        iou_threshold: float = 0.25,
        use_supervision: bool = True,
        frame_rate: float = 25.0,
        track_activation_threshold: float = 0.5,
        track_buffer: int = 75,
        matching_threshold: float = 0.95,
        role_policy: str = "stable",
        role_half_life_s: float = 2.0,
        role_switch_margin: float = 1.25,
    ) -> None:
        if role_policy not in ROLE_POLICIES:
            raise ValueError(f"role_policy must be one of {ROLE_POLICIES}.")
        if max_lost_frames < 0:
            raise ValueError("max_lost_frames must be non-negative.")
        if not math.isfinite(float(iou_threshold)) or not 0 <= iou_threshold <= 1:
            raise ValueError("iou_threshold must be finite and within 0-1.")
        if not math.isfinite(float(frame_rate)) or frame_rate <= 0:
            raise ValueError("frame_rate must be finite and greater than zero.")
        self.max_lost_frames = max_lost_frames
        self.iou_threshold = iou_threshold
        self.frame_rate = float(frame_rate)
        # ByteTrack settings tuned on SoccerTrack 117093 dev clips (see docs/cv_detection_tracking_report.md):
        # a permissive IoU-distance gate (0.95) suits ~20x45 px players, a 3 s buffer re-links occlusions.
        self.track_activation_threshold = track_activation_threshold
        self.track_buffer = track_buffer
        self.matching_threshold = matching_threshold
        self.next_track_id = 1
        self.tracks: dict[int, ActiveTrack] = {}
        self._supervision_tracker: Any = None
        self.role_policy = role_policy
        self._role_half_life_s, self._role_switch_margin = role_half_life_s, role_switch_margin
        self.roles = RoleEstimator(self.frame_rate, role_half_life_s, role_switch_margin)
        self.ignored_non_person_detections = 0  # e.g. a ball handed to the person tracker by mistake

        if use_supervision:
            try:
                import supervision as sv
                self._supervision_tracker = self._new_bytetrack(sv)
            except Exception:
                self._supervision_tracker = None

    def _new_bytetrack(self, sv: Any) -> Any:
        return sv.ByteTrack(
            track_activation_threshold=self.track_activation_threshold,
            lost_track_buffer=self.track_buffer,
            minimum_matching_threshold=self.matching_threshold,
            frame_rate=self.frame_rate,
        )

    @property
    def backend(self) -> str:
        """Return the tracker implementation actually selected."""
        return "bytetrack" if self._supervision_tracker is not None else "iou_fallback"

    def set_frame_rate(self, frame_rate: float) -> None:
        """Set ByteTrack's rate before it has consumed a frame."""
        if not math.isfinite(float(frame_rate)) or frame_rate <= 0:
            raise ValueError("frame_rate must be finite and greater than zero.")
        if self._supervision_tracker is not None:
            # Reinitialize to ensure ByteTrack scales its lost-track buffer to
            # the input video's actual temporal sampling.
            import supervision as sv
            self.frame_rate = float(frame_rate)
            self._supervision_tracker = self._new_bytetrack(sv)
        else:
            self.frame_rate = float(frame_rate)
        self.roles = RoleEstimator(self.frame_rate, self._role_half_life_s, self._role_switch_margin)

    def update(self, detections: Sequence[Detection], frame_idx: int) -> list[ActiveTrack]:
        """Update tracks with person detections and return the current active tracks.

        Non-person detections (the ball) are not tracked; they are counted and dropped.
        """
        people = [d for d in detections if d.class_name in PERSON_CLASSES]
        self.ignored_non_person_detections += len(detections) - len(people)
        if self._supervision_tracker is not None:
            tracks = self._update_supervision(people)
        else:
            tracks = [dataclasses.replace(t) for t in self._update_builtin(people)]
        return self._apply_roles(tracks, frame_idx)

    def _apply_roles(self, tracks: list[ActiveTrack], frame_idx: int) -> list[ActiveTrack]:
        for track in tracks:
            track.observed_class = track.class_name
            role, share = self.roles.observe(track.track_id, track.class_name, track.confidence, frame_idx)
            track.role_confidence = share
            if self.role_policy == "stable":
                track.class_name = role
        self.roles.prune(frame_idx, max_age_frames=max(self.track_buffer, self.max_lost_frames) * 2)
        return tracks

    def _update_supervision(self, detections: Sequence[Detection]) -> list[ActiveTrack]:
        """Update using Roboflow supervision ByteTrack."""
        import numpy as np
        import supervision as sv

        if not detections:
            # Empty frame
            empty_sv = sv.Detections.empty()
            self._supervision_tracker.update_with_detections(empty_sv)
            return []

        xyxy = np.array([[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2] for d in detections])
        confidence = np.array([d.confidence for d in detections])
        classes = list(DetectedClass)
        class_id = np.array([classes.index(d.class_name) for d in detections])

        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )

        tracked = self._supervision_tracker.update_with_detections(sv_detections)

        active: list[ActiveTrack] = []
        for i in range(len(tracked)):
            t_id = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else (i + 1)
            b = tracked.xyxy[i]
            conf = float(tracked.confidence[i]) if tracked.confidence is not None else 1.0
            cls = classes[int(tracked.class_id[i])]

            active.append(ActiveTrack(
                track_id=t_id,
                class_name=cls,
                bbox=BoundingBox(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                confidence=conf,
            ))
        return active

    def _update_builtin(self, detections: Sequence[Detection]) -> list[ActiveTrack]:
        """Pure-Python IoU and spatial proximity tracker."""
        # Keep the last observed box as the association anchor. Predicted boxes
        # can be added later with a validated motion model.
        for track in self.tracks.values():
            track.age += 1
            track.lost_frames += 1

        unmatched_detections = list(range(len(detections)))
        matched_tracks = set()

        if self.tracks and detections:
            # Match by class and IoU
            for track_id, track in list(self.tracks.items()):
                best_iou = -1.0
                best_det_idx = -1

                for d_idx in unmatched_detections:
                    det = detections[d_idx]
                    # association is class-agnostic across person roles
                    iou = self._compute_iou(track.bbox, det.bbox)
                    if iou > best_iou and iou >= self.iou_threshold:
                        best_iou = iou
                        best_det_idx = d_idx

                if best_det_idx >= 0:
                    det = detections[best_det_idx]
                    # Update velocity
                    new_center = det.bbox.center
                    old_center = track.bbox.center
                    track.vx = new_center[0] - old_center[0]
                    track.vy = new_center[1] - old_center[1]

                    track.bbox = det.bbox
                    track.confidence = det.confidence
                    track.class_name = det.class_name  # latest raw label; the role estimate is applied on output
                    track.hits += 1
                    track.lost_frames = 0
                    matched_tracks.add(track_id)
                    unmatched_detections.remove(best_det_idx)

        # Create new tracks for unmatched detections
        for d_idx in unmatched_detections:
            det = detections[d_idx]
            new_id = self.next_track_id
            self.next_track_id += 1
            self.tracks[new_id] = ActiveTrack(
                track_id=new_id,
                class_name=det.class_name,
                bbox=det.bbox,
                confidence=det.confidence,
            )

        # Prune expired tracks
        expired = [t_id for t_id, t in self.tracks.items() if t.lost_frames > self.max_lost_frames]
        for t_id in expired:
            del self.tracks[t_id]

        return [t for t in self.tracks.values() if t.lost_frames == 0]

    @staticmethod
    def _compute_iou(b1: BoundingBox, b2: BoundingBox) -> float:
        """Compute Intersection over Union between two bounding boxes."""
        x_left = max(b1.x1, b2.x1)
        y_top = max(b1.y1, b2.y1)
        x_right = min(b1.x2, b2.x2)
        y_bottom = min(b1.y2, b2.y2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = b1.width * b1.height
        area2 = b2.width * b2.height
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0
