"""Random frame access to the stage inputs (cache detections, production tracker output, replay variants, GT) for the
dashboard's detection-diagnostic view. Read-only."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from football_fatigue.video.detection.diagnose import cache_dir, find_engine_run, frame_size, load_predicted, replay_path
from football_fatigue.video.detection.replay import REPLAY_VERSION
from football_fatigue.video.detection.stages import Det, Params, attribute_frame
from football_fatigue.video.sources import SOURCES, guard_not_reserved, play_area_polygon


class LineIndex:
    """Byte offsets of a JSONL file keyed by an integer frame field (read lazily, file never modified)."""

    def __init__(self, path: Path, key: str) -> None:
        self.path = Path(path)
        self.offsets: dict[int, int] = {}
        pos = 0
        with self.path.open("rb") as fh:
            for line in fh:
                if line.strip():
                    i = line.find(b'"' + key.encode() + b'":')
                    j = i + len(key) + 3
                    k = j
                    while line[k:k + 1].isdigit() or line[k:k + 1] == b" ":
                        k += 1
                    self.offsets[int(line[j:k])] = pos
                pos += len(line)

    def get(self, frame: int) -> dict[str, Any] | None:
        off = self.offsets.get(frame)
        if off is None:
            return None
        with self.path.open("rb") as fh:
            fh.seek(off)
            return json.loads(fh.readline())


@dataclass
class FrameStore:
    data_root: Path
    out_root: Path
    source_id: str

    def __post_init__(self) -> None:
        from football_fatigue.vision.mot_eval import load_reference

        spec = SOURCES[self.source_id]
        guard_not_reserved(self.source_id, spec["cache_run"], spec["replay_run"])
        cdir = cache_dir(self.data_root, self.source_id)
        self.columns = json.loads((cdir / "cache_manifest.json").read_text(encoding="utf-8"))["columns"]
        self.cache = LineIndex(cdir / "detections.jsonl", "frame_num")
        self.replay = LineIndex(replay_path(self.data_root, self.source_id), "frame_num")
        self.gt, _ = load_reference(Path(self.data_root) / spec["gt"], 0)
        self.poly = np.asarray(play_area_polygon(Path(self.data_root) / spec["keypoints"]),
                               dtype=np.float32).reshape(-1, 1, 2)
        self.size = frame_size(self.data_root, self.source_id)
        self.predicted, _ = load_predicted(find_engine_run(self.out_root, self.source_id))
        self.video = Path(self.data_root) / spec["video"]
        self._variants: dict[str, dict[int, dict[int, tuple]]] = {}
        self.last_frame = max(self.cache.offsets)

    def dets(self, frame: int) -> list[Det]:
        rec = self.cache.get(frame)
        if rec is None:
            return []
        out = []
        for row in rec["observations"]:
            o = dict(zip(self.columns, row)) if isinstance(row, list) else row
            turf = o["passes_turf_filter"]   # stored as 1 / 0 / null
            out.append(Det((float(o["x1"]), float(o["y1"]), float(o["x2"]), float(o["y2"])), float(o["confidence"]),
                           str(o["class_label"]), o["mapped_class"], None if turf is None else bool(turf)))
        return out

    def tracks(self, frame: int, variant: str = "B0") -> dict[int, tuple]:
        if variant == "B0":
            rec = self.replay.get(frame)
            return {} if rec is None else {int(p["track_id"]): tuple(float(v) for v in p["bbox_xyxy_px"])
                                           for p in rec["players"]}
        if variant not in self._variants:
            path = Path(self.out_root) / self.source_id / "detection" / REPLAY_VERSION / variant / "tracks.jsonl.gz"
            v: dict[int, dict[int, tuple]] = {}
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    r = json.loads(line)
                    v[r["frame"]] = {int(t[0]): tuple(t[1:5]) for t in r["tracks"]}
            self._variants[variant] = v
        return self._variants[variant].get(frame, {})

    def available_variants(self) -> list[str]:
        base = Path(self.out_root) / self.source_id / "detection" / REPLAY_VERSION
        names = sorted(p.name for p in base.glob("*") if (p / "tracks.jsonl.gz").exists()) if base.exists() else []
        return ["B0"] + [n for n in names if n != "B0"]

    def attribution(self, frame: int, variant: str = "B0", p: Params = Params()) -> dict[str, Any]:
        """Stage attribution of one frame against the chosen tracker output."""
        return attribute_frame(self.gt.get(frame, []), self.dets(frame), self.tracks(frame, variant), self.poly,
                               self.size, p)
