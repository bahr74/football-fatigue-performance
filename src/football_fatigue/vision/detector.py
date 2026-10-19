"""Object detection wrapper for players, goalkeepers, referees, and the ball."""

from __future__ import annotations

from typing import Any, Sequence
from football_fatigue.vision.models import BoundingBox, DetectedClass, Detection


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

    def detect(
        self,
        frame_rgb: Any,
        pitch_mask_polygon: Sequence[tuple[float, float]] | None = None,
    ) -> list[Detection]:
        """Run object detection on an RGB frame (numpy array or image path)."""
        self.load_model()

        detections: list[Detection] = []
        names = getattr(self._model, "names", {})
        for cls_id, conf, xyxy in self._infer(frame_rgb):
            cls_name = self._map_class(cls_id, names)
            if cls_name is None:
                continue

            bbox = BoundingBox(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))

            if cls_name != DetectedClass.BALL and not self._looks_like_player(frame_rgb, bbox):
                continue

            # Optional pitch polygon filtering
            if pitch_mask_polygon is not None and cls_name != DetectedClass.BALL:
                pt = bbox.bottom_center
                if not self._point_in_polygon(pt[0], pt[1], pitch_mask_polygon):
                    continue

            detections.append(Detection(
                class_name=cls_name,
                confidence=conf,
                bbox=bbox,
            ))

        return detections

    def _tile_origins(self, width: int) -> list[int]:
        tile = min(self.tile_width, width)
        step = max(1, int(tile * (1 - self.tile_overlap)))
        xs = list(range(0, max(1, width - tile + 1), step))
        if xs[-1] + tile < width:
            xs.append(width - tile)
        return xs

    def _infer(self, frame_rgb: Any) -> list[tuple[int, float, list[float]]]:
        """Raw (class_id, confidence, xyxy) in full-frame pixel coordinates."""
        # Ultralytics treats numpy arrays as BGR; the pipeline supplies RGB.
        import numpy as np

        bgr = np.ascontiguousarray(frame_rgb[:, :, ::-1])
        kwargs = dict(
            conf=self.confidence_threshold, device=self.device, imgsz=self.image_size,
            half=self.half, verbose=False,
        )
        if self.tile_width is None:
            results = self._model(bgr, **kwargs)
            offsets = [0]
        else:
            width = bgr.shape[1]
            offsets = self._tile_origins(width)
            tile = min(self.tile_width, width)
            results = self._model([np.ascontiguousarray(bgr[:, x:x + tile]) for x in offsets], **kwargs)
        out: list[tuple[int, float, list[float]]] = []
        for r, x_off in zip(results, offsets):
            b = r.boxes
            if b is None or len(b) == 0:
                continue
            for cls, conf, xyxy in zip(b.cls.tolist(), b.conf.tolist(), b.xyxy.tolist()):
                out.append((int(cls), float(conf), [xyxy[0] + x_off, xyxy[1], xyxy[2] + x_off, xyxy[3]]))
        if self.tile_width is not None and len(offsets) > 1:
            out = self._merge_tiles(out)
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
