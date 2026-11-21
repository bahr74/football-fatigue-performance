"""Object detection wrapper for players, goalkeepers, referees, and the ball."""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence
from football_fatigue.vision.models import BoundingBox, DetectedClass, Detection, RawObservation

TILE_MERGE_POLICIES = ("class_aware", "person_agnostic", "edge_aware")
_PERSON_CLASSES = frozenset({
    DetectedClass.PERSON, DetectedClass.PLAYER, DetectedClass.GOALKEEPER, DetectedClass.REFEREE,
})
PERSON_AGNOSTIC_POLICIES = ("person_agnostic", "edge_aware")
# person_agnostic / edge_aware merge constants (recorded in the detector provenance)
EDGE_PX = 2.0               # a box within this distance of a tile cut is treated as touching it
EDGE_CONTAINMENT = 0.8      # share of the truncated box that must lie inside the other box
EDGE_CONFIDENCE_MARGIN = 0.2  # the larger box may be at most this much less confident


class YOLOPlayerDetector:
    """YOLOv8/v11 wrapper for detecting football entities in match video frames."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.40,
        device: str = "cpu",
        image_size: int = 640,
        tile_width: int | None = None,
        tile_overlap: float = 0.2,
        half: bool = False,
        merge_iou: float = 0.5,
        tile_height: int | None = None,
        tile_batch: int | None = None,
        tile_merge: str = "class_aware",
        cross_role_iou: float = 0.7,
    ) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        if not isinstance(image_size, int) or image_size < 32:
            raise ValueError("image_size must be an integer of at least 32 pixels.")
        self.image_size = image_size
        if tile_width is not None and tile_width < 64:
            raise ValueError("tile_width must be at least 64 pixels.")
        if not 0 <= tile_overlap < 1:
            raise ValueError("tile_overlap must be in [0, 1).")
        # Tiled inference: full-height vertical strips of tile_width pixels so
        # small players keep their native scale instead of being downsized.
        self.tile_width = tile_width
        self.tile_overlap = tile_overlap
        self.half = half
        self.merge_iou = merge_iou
        # Optional 2D tiling (tile_height), tile micro-batching and an alternative merge policy; all
        # default to the original behaviour (full-height strips, one batch, class-aware NMS).
        if tile_height is not None and tile_height < 64:
            raise ValueError("tile_height must be at least 64 pixels.")
        if tile_batch is not None and tile_batch < 1:
            raise ValueError("tile_batch must be at least 1.")
        if tile_merge not in TILE_MERGE_POLICIES:
            raise ValueError(f"tile_merge must be one of {TILE_MERGE_POLICIES}.")
        if not 0 < merge_iou <= cross_role_iou <= 1:
            raise ValueError("cross_role_iou must be within [merge_iou, 1].")
        if tile_width is None and (tile_height is not None or tile_batch is not None or tile_merge != "class_aware"):
            raise ValueError("tile_height, tile_batch and tile_merge require tile_width.")
        self.tile_height = tile_height
        self.tile_batch = tile_batch
        self.tile_merge = tile_merge
        self.cross_role_iou = cross_role_iou
        self.merge_statistics: Counter[str] = Counter()
        self._model: Any = None

    def load_model(self) -> None:
        """Load YOLO model lazily on first inference."""
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model from {self.model_path}: {e}")

    def describe(self) -> dict[str, Any]:
        """Detector settings that determine its raw output (recorded in detection-cache provenance)."""
        names = getattr(self._model, "names", None)
        merge: dict[str, Any] = {"policy": self.tile_merge, "merge_iou": self.merge_iou}
        if self.tile_merge in PERSON_AGNOSTIC_POLICIES:
            merge |= {
                "cross_role_iou": self.cross_role_iou, "edge_px": EDGE_PX, "edge_containment": EDGE_CONTAINMENT,
                "edge_confidence_margin": EDGE_CONFIDENCE_MARGIN,
                "ball_and_unmapped_labels": "merged class-aware, never with person roles",
            }
            if self.tile_merge == "edge_aware":
                merge["edge_truncation"] = "resolved before same-role NMS, then re-checked on the survivors"
        return {
            "model_path": self.model_path,
            "confidence_floor": self.confidence_threshold,
            "image_size": self.image_size,
            "tile_width": self.tile_width,
            "tile_height": self.tile_height,
            "tile_overlap": self.tile_overlap,
            "tile_batch": self.tile_batch,
            "tile_merge": merge,
            "merge_iou": self.merge_iou,
            "half": self.half,
            "device": self.device,
            "preprocessing": {
                "input_colour": "RGB frame converted to BGR for Ultralytics",
                "resize": "Ultralytics letterbox of each tile to image_size on its longer side; "
                          "equal-size tiles are padded rectangularly",
                "tiles": "full-height strips" if self.tile_width is not None and self.tile_height is None
                         else "2D grid" if self.tile_width is not None else "whole frame",
            },
            "turf_filter": {"name": "looks_like_player", "version": 1},
            "class_names": {str(k): str(v) for k, v in names.items()} if isinstance(names, dict) else None,
        }

    def observe(
        self,
        frame_rgb: Any,
        pitch_mask_polygon: Sequence[tuple[float, float]] | None = None,
    ) -> list[RawObservation]:
        """Every detector output on an RGB frame with the filter decisions recorded.

        ``detect`` keeps exactly the observations whose ``kept`` property is true.
        """
        self.load_model()

        observations: list[RawObservation] = []
        names = getattr(self._model, "names", {})
        for cls_id, conf, xyxy in self._infer(frame_rgb):
            cls_name = self._map_class(cls_id, names)
            box = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
            passes_turf: bool | None = None
            inside_mask: bool | None = None
            if cls_name is not None and cls_name != DetectedClass.BALL:
                bbox = BoundingBox(*box)
                passes_turf = self._looks_like_player(frame_rgb, bbox)
                # Optional pitch polygon filtering
                if passes_turf and pitch_mask_polygon is not None:
                    pt = bbox.bottom_center
                    inside_mask = self._point_in_polygon(pt[0], pt[1], pitch_mask_polygon)
            observations.append(RawObservation(
                class_id=int(cls_id),
                class_label=self._label(cls_id, names),
                mapped_class=cls_name,
                confidence=conf,
                xyxy=box,
                passes_turf_filter=passes_turf,
                inside_pitch_mask=inside_mask,
            ))
        return observations

    def detect(
        self,
        frame_rgb: Any,
        pitch_mask_polygon: Sequence[tuple[float, float]] | None = None,
    ) -> list[Detection]:
        """Run object detection on an RGB frame (numpy array or image path)."""
        detections = (o.to_detection() for o in self.observe(frame_rgb, pitch_mask_polygon))
        return [d for d in detections if d is not None]

    @staticmethod
    def _label(class_id: int, names: Any) -> str:
        """The model's own label for a class id (recorded raw, before class mapping)."""
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    def _axis_origins(self, length: int, tile_size: int) -> list[int]:
        tile = min(tile_size, length)
        step = max(1, int(tile * (1 - self.tile_overlap)))
        starts = list(range(0, max(1, length - tile + 1), step))
        if starts[-1] + tile < length:
            starts.append(length - tile)
        return starts

    def _tile_origins(self, width: int) -> list[int]:
        return self._axis_origins(width, self.tile_width)

    def _tile_grid(self, width: int, height: int) -> list[tuple[int, int, int, int]]:
        """Tiles as (x, y, w, h), row by row; without tile_height each tile is a full-height strip."""
        tile_w = min(self.tile_width, width)
        if self.tile_height is None:
            rows, tile_h = [0], height
        else:
            rows, tile_h = self._axis_origins(height, self.tile_height), min(self.tile_height, height)
        return [(x, y, tile_w, tile_h) for y in rows for x in self._tile_origins(width)]

    def _run_tiles(self, crops: list[Any], kwargs: dict[str, Any]) -> list[Any]:
        """One model call per ``tile_batch`` crops (all crops in a single call by default)."""
        if self.tile_batch is None or self.tile_batch >= len(crops):
            return self._model(crops, **kwargs)
        results: list[Any] = []
        for start in range(0, len(crops), self.tile_batch):
            results.extend(self._model(crops[start:start + self.tile_batch], **kwargs))
        return results

    def _infer(self, frame_rgb: Any) -> list[tuple[int, float, list[float]]]:
        """Raw (class_id, confidence, xyxy) in full-frame pixel coordinates."""
        # Ultralytics treats numpy arrays as BGR; the pipeline supplies RGB.
        import numpy as np

        bgr = np.ascontiguousarray(frame_rgb[:, :, ::-1])
        kwargs = dict(
            conf=self.confidence_threshold, device=self.device, imgsz=self.image_size,
            half=self.half, verbose=False,
        )
        height, width = bgr.shape[:2]
        if self.tile_width is None:
            results = self._model(bgr, **kwargs)
            tiles = [(0, 0, width, height)]
        else:
            tiles = self._tile_grid(width, height)
            results = self._run_tiles(
                [np.ascontiguousarray(bgr[y:y + h, x:x + w]) for x, y, w, h in tiles], kwargs,
            )
        out: list[tuple[int, float, list[float]]] = []
        tile_of: list[int] = []
        for index, (r, (x_off, y_off, _w, _h)) in enumerate(zip(results, tiles)):
            b = r.boxes
            if b is None or len(b) == 0:
                continue
            for cls, conf, xyxy in zip(b.cls.tolist(), b.conf.tolist(), b.xyxy.tolist()):
                out.append((int(cls), float(conf), [xyxy[0] + x_off, xyxy[1] + y_off, xyxy[2] + x_off, xyxy[3] + y_off]))
                tile_of.append(index)
        if self.tile_width is not None and len(tiles) > 1:
            before = len(out)
            if self.tile_merge in PERSON_AGNOSTIC_POLICIES:
                out = self._merge_person_agnostic(out, [tiles[i] for i in tile_of], (width, height))
            else:
                out = self._merge_tiles(out)
                self.merge_statistics["suppressed_class_aware"] += before - len(out)
        return out

    def _merge_tiles(self, items: list[tuple[int, float, list[float]]]) -> list[tuple[int, float, list[float]]]:
        """Class-aware NMS to remove duplicates from overlapping tiles."""
        import torch
        from torchvision.ops import batched_nms

        if not items:
            return items
        boxes = torch.tensor([i[2] for i in items], dtype=torch.float32)
        scores = torch.tensor([i[1] for i in items], dtype=torch.float32)
        classes = torch.tensor([i[0] for i in items])
        keep = batched_nms(boxes, scores, classes, self.merge_iou).tolist()
        return [items[k] for k in keep]

    def _merge_person_agnostic(
        self,
        items: list[tuple[int, float, list[float]]],
        tile_rects: list[tuple[int, int, int, int]],
        frame_size: tuple[int, int],
    ) -> list[tuple[int, float, list[float]]]:
        """Conservative duplicate suppression across tiles; roles are kept apart unless boxes coincide.

        1. Ball and unmapped labels are merged class-aware and never interact with person boxes.
        2. Person boxes of the same label are merged by NMS at ``merge_iou`` (as the default policy).
        3. Person boxes of different roles are merged only when they nearly coincide
           (IoU >= ``cross_role_iou``); the more confident box keeps its own role. The
           suppressed role is counted in ``merge_statistics`` but is not stored per observation.
        4. A box cut by a tile edge is dropped only when a more complete box from another tile
           extends beyond that cut, contains it (>= ``EDGE_CONTAINMENT`` of its area), and is not
           much less confident; otherwise both are kept.

        With ``tile_merge="edge_aware"`` rule 4 is applied to all person boxes *before* the
        same-role NMS (and again on the survivors). Under ``person_agnostic`` a truncated box that is
        more confident than the complete box wins the NMS and the complete box is gone before rule 4
        can see it.
        """
        import torch
        from torchvision.ops import batched_nms, nms

        if not items:
            return items
        names = getattr(self._model, "names", {})
        boxes = torch.tensor([i[2] for i in items], dtype=torch.float32)
        scores = torch.tensor([i[1] for i in items], dtype=torch.float32)
        classes = torch.tensor([i[0] for i in items])
        person = [self._map_class(i[0], names) in _PERSON_CLASSES for i in items]
        person_idx = [k for k, p in enumerate(person) if p]
        other_idx = [k for k, p in enumerate(person) if not p]
        kept: list[int] = []
        if other_idx:
            sel = torch.tensor(other_idx)
            kept += [other_idx[k] for k in batched_nms(boxes[sel], scores[sel], classes[sel], self.merge_iou).tolist()]
        if person_idx:
            if self.tile_merge == "edge_aware":
                person_idx = self._resolve_edge_truncation(items, tile_rects, frame_size, person_idx, names)
            sel = torch.tensor(person_idx)
            same_role = [person_idx[k] for k in batched_nms(boxes[sel], scores[sel], classes[sel], self.merge_iou).tolist()]
            self.merge_statistics["suppressed_same_role"] += len(person_idx) - len(same_role)
            sel = torch.tensor(same_role)
            survivors = [same_role[k] for k in nms(boxes[sel], scores[sel], self.cross_role_iou).tolist()]
            survivor_set = set(survivors)
            for dropped in same_role:
                if dropped in survivor_set:
                    continue
                winner = max(survivors, key=lambda s: self._box_iou(items[s][2], items[dropped][2]))
                pair = f"{self._label(items[winner][0], names)}>{self._label(items[dropped][0], names)}"
                self.merge_statistics[f"suppressed_cross_role:{pair}"] += 1
            truncated = {
                a for a in survivors
                if any(
                    b != a and self._truncated_by(items[a], tile_rects[a], items[b], tile_rects[b], frame_size)
                    for b in survivors
                )
            }
            self.merge_statistics["suppressed_edge_truncated"] += len(truncated)
            kept += [s for s in survivors if s not in truncated]
        return [items[k] for k in sorted(kept)]

    def _resolve_edge_truncation(
        self,
        items: list[tuple[int, float, list[float]]],
        tile_rects: list[tuple[int, int, int, int]],
        frame_size: tuple[int, int],
        person_idx: list[int],
        names: Any,
    ) -> list[int]:
        """Person indices left after dropping boxes cut by a tile edge and completed by another tile's box."""
        survivors = []
        for a in person_idx:
            completer = next(
                (b for b in person_idx
                 if b != a and self._truncated_by(items[a], tile_rects[a], items[b], tile_rects[b], frame_size)),
                None,
            )
            if completer is None:
                survivors.append(a)
                continue
            self.merge_statistics["suppressed_edge_truncated_before_nms"] += 1
            if self._label(items[a][0], names) != self._label(items[completer][0], names):
                self.merge_statistics["edge_truncated_role_differs"] += 1
        return survivors

    @staticmethod
    def _box_iou(a: Sequence[float], b: Sequence[float]) -> float:
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _truncated_by(
        small: tuple[int, float, list[float]],
        small_tile: tuple[int, int, int, int],
        big: tuple[int, float, list[float]],
        big_tile: tuple[int, int, int, int],
        frame_size: tuple[int, int],
    ) -> bool:
        """Whether ``small`` is cut by an interior edge of its tile and ``big`` (another tile) completes it."""
        if small_tile == big_tile:
            return False
        a, b = small[2], big[2]
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        if area_b <= area_a or big[1] < small[1] - EDGE_CONFIDENCE_MARGIN:
            return False
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        if area_a <= 0 or ix * iy / area_a < EDGE_CONTAINMENT:
            return False
        tx, ty, tw, th = small_tile
        width, height = frame_size
        cut_and_completed = (
            (tx > 0 and a[0] <= tx + EDGE_PX and b[0] < tx - EDGE_PX),
            (tx + tw < width and a[2] >= tx + tw - EDGE_PX and b[2] > tx + tw + EDGE_PX),
            (ty > 0 and a[1] <= ty + EDGE_PX and b[1] < ty - EDGE_PX),
            (ty + th < height and a[3] >= ty + th - EDGE_PX and b[3] > ty + th + EDGE_PX),
        )
        return any(cut_and_completed)

    @staticmethod
    def _looks_like_player(frame_rgb: Any, bbox: BoundingBox) -> bool:
        """Reject boxes whose proposed upper body is almost entirely field turf."""
        if frame_rgb is None:
            return True
        try:
            import cv2
            import numpy as np

            height, width = frame_rgb.shape[:2]
            x1 = max(0, min(width, int(bbox.x1)))
            x2 = max(0, min(width, int(bbox.x2)))
            y1 = max(0, min(height, int(bbox.y1)))
            y2 = max(0, min(height, int(bbox.y1 + bbox.height * 0.65)))
            if x2 <= x1 or y2 <= y1:
                return False
            hsv = cv2.cvtColor(frame_rgb[y1:y2, x1:x2], cv2.COLOR_RGB2HSV)
            green = ((hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85) & (hsv[:, :, 1] >= 40))
            return float(np.mean(green)) <= 0.72
        except Exception:
            return True

    @staticmethod
    def _map_class(class_id: int, names: Any) -> DetectedClass | None:
        """Map model labels conservatively; COCO person is not a known role."""
        if isinstance(names, dict):
            raw_name = names.get(class_id, str(class_id))
        elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            raw_name = names[class_id]
        else:
            raw_name = str(class_id)
        label = str(raw_name).strip().lower().replace(" ", "_")
        aliases = {
            "person": DetectedClass.PERSON,
            "player": DetectedClass.PLAYER,
            "goalkeeper": DetectedClass.GOALKEEPER,
            "goalie": DetectedClass.GOALKEEPER,
            "referee": DetectedClass.REFEREE,
            "official": DetectedClass.REFEREE,
            "ball": DetectedClass.BALL,
            "sports_ball": DetectedClass.BALL,
        }
        if class_id == 32 and label in {"32", "sports_ball"}:
            return DetectedClass.BALL
        return aliases.get(label)

    @staticmethod
    def _point_in_polygon(x: float, y: float, poly: Sequence[tuple[float, float]]) -> bool:
        """Ray-casting algorithm to test if a point is within a polygonal mask."""
        n = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
