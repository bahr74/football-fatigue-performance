"""Detector adapters with one interface, and a resumable, write-once runner over the benchmark frame manifest.

Every adapter returns, for one decoded frame, boxes as ``[x1, y1, x2, y2, confidence, model_label, mapped_class]`` in
full-frame pixel coordinates (top-left origin, x right, y down, xyxy, exclusive of nothing - the convention of the
detection cache). ``mapped_class`` is Session A's conservative class mapping (COCO ``person`` -> ``person``; football
labels -> player / goalkeeper / referee / ball); evaluation treats every person-like class as "person".

Adapters
* ``cache``              - the production detection cache (YOLO26-L, GPU, imgsz 1920, 1280-px tiles, edge-aware merge)
* ``ultralytics_tiled``  - Session A's ``YOLOPlayerDetector`` (imported, not modified) with any Ultralytics checkpoint,
                           the production tile grid and merge, and a chosen inference size
* ``unavailable``        - raises ``DetectorUnavailable`` with the registry status; never produces boxes
"""

from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from football_fatigue.video.bench.frames import load as load_manifest
from football_fatigue.video.bench.registry import CANDIDATES, resolve, sha256_of, status
from football_fatigue.video.sources import SOURCES, ExactFrameReader, guard_not_reserved

RUNS_VERSION = "detector-v1"
CONF_FLOOR = 0.1                      # same floor as the production cache, so AP is computed over the same range


class DetectorUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class Protocol:
    name: str
    image_size: int | None
    tile: int | None = 1280
    overlap: float = 0.2
    merge: str = "edge_aware"
    note: str = ""


PROTOCOLS = {p.name: p for p in (
    Protocol("production-cache", 1920, note="read from the production detection cache (no inference in this run)"),
    Protocol("tile1280-img1280", 1280, note="production tile grid and merge; inference size 1280 for every model (B1 common protocol)"),
    Protocol("tile1280-img1920", 1920, note="production settings exactly; used on CPU only as an adapter-equivalence check"),
    Protocol("tile1280-img640", 640, note="each COCO model's native training size"),
)}


def run_key(detector: str, protocol: str) -> str:
    return f"{detector}__{protocol}"


def runs_root(out_root: Path) -> Path:
    return Path(out_root) / "_bench" / RUNS_VERSION / "runs"


class CacheAdapter:
    device = "cache"

    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
        self._cache: dict[str, dict[int, list]] = {}

    def describe(self) -> dict[str, Any]:
        from football_fatigue.video.detection.diagnose import cache_dir

        out = {}
        for sid in self._cache:
            cm = json.loads((cache_dir(self.data_root, sid) / "cache_manifest.json").read_text(encoding="utf-8"))
            out[sid] = {"cache_run": cm["run_id"], "detector": cm["detector"], "records_sha256": cm["records_sha256"]}
        return {"adapter": "cache", "caches": out}

    def detect(self, source_id: str, frame: int, _bgr) -> list[list]:
        from football_fatigue.video.detection.diagnose import iter_cache

        if source_id not in self._cache:
            self._cache[source_id] = {fr: [[*d.box, d.conf, d.label, d.mapped] for d in ds if d.conf > CONF_FLOOR]
                                      for fr, ds in iter_cache(self.data_root, source_id)}
        return self._cache[source_id][frame]


class UltralyticsTiledAdapter:
    def __init__(self, weights: Path, protocol: Protocol, device: str = "cpu", threads: int | None = None):
        import torch
        from football_fatigue.vision.detector import YOLOPlayerDetector

        if device == "cpu":
            torch.set_num_threads(threads or os.cpu_count() or 1)
        self.device = device
        self.weights = Path(weights)
        self.protocol = protocol
        self.det = YOLOPlayerDetector(model_path=str(weights), confidence_threshold=CONF_FLOOR, device=device,
                                      image_size=protocol.image_size, tile_width=protocol.tile, tile_height=protocol.tile,
                                      tile_overlap=protocol.overlap, tile_batch=1, tile_merge=protocol.merge,
                                      cross_role_iou=0.7)
        self.det.load_model()

    def describe(self) -> dict[str, Any]:
        m = self.det._model.model
        return {"adapter": "ultralytics_tiled", "detector_settings": self.det.describe(),
                "params_m_measured": round(sum(p.numel() for p in m.parameters()) / 1e6, 3),
                "note": "tile_batch 1 (production used 2; batching does not change Ultralytics outputs per tile)"}

    def detect(self, _source_id: str, _frame: int, bgr) -> list[list]:
        import numpy as np

        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        return [[*o.xyxy, float(o.confidence), o.class_label, o.mapped_class.value if o.mapped_class else None]
                for o in self.det.observe(rgb)]


def make_adapter(detector: str, protocol: str, data_root: Path, project_root: Path, device: str = "cpu"):
    c = CANDIDATES[detector]
    p = PROTOCOLS[protocol]
    if protocol == "production-cache":
        if detector != "yolo26l-football":
            raise DetectorUnavailable("only the production detector has a detection cache")
        return CacheAdapter(data_root)
    if c.adapter != "ultralytics_tiled":
        raise DetectorUnavailable(f"{detector}: {status(detector, project_root)} (adapter {c.adapter} not installed)")
    weights = resolve(detector, project_root)
    if weights is None:
        raise DetectorUnavailable(f"{detector}: {status(detector, project_root)} - weights not on this machine")
    if device.startswith("cuda") and os.environ.get("FF_GPU_APPROVED") != "1":
        raise DetectorUnavailable("GPU inference needs explicit approval (set FF_GPU_APPROVED=1 only after it is given)")
    return UltralyticsTiledAdapter(weights, p, device)


def _hardware(device: str) -> dict[str, Any]:
    import torch

    hw = {"platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
          "torch": torch.__version__, "torch_threads": torch.get_num_threads(), "device": device}
    if torch.cuda.is_available():
        hw["gpu"] = torch.cuda.get_device_name(0)
    return hw


def run(data_root: Path, out_root: Path, project_root: Path, detector: str, protocol: str, device: str = "cpu",
        max_frames: int | None = None, sources: list[str] | None = None) -> Path:
    """Run one detector on the manifest frames (resumable; a completed run is never rewritten)."""
    import psutil

    manifest = load_manifest(out_root)
    frames = [e for e in manifest["frames"] if not sources or e["source_id"] in sources]
    if max_frames is not None:
        frames = frames[:max_frames]
    dest = runs_root(out_root) / run_key(detector, protocol)
    mpath = dest / "run_manifest.json"
    if mpath.exists() and json.loads(mpath.read_text(encoding="utf-8"))["status"] == "complete":
        raise SystemExit(f"run already complete (write-once): {dest}")
    adapter = make_adapter(detector, protocol, data_root, project_root, device)
    dest.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, int]] = set()
    dpath = dest / "dets.jsonl"
    if dpath.exists():
        for line in dpath.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["source_id"], r["frame"]))
    proc = psutil.Process()
    peak_rss = proc.memory_info().rss
    readers: dict[str, ExactFrameReader] = {}
    t_start = time.perf_counter()
    c = CANDIDATES[detector]
    head = {"status": "partial", "detector": detector, "protocol": protocol, "protocol_spec": PROTOCOLS[protocol].__dict__,
            "device": adapter.device, "candidate": {k: getattr(c, k) for k in ("family", "track", "checkpoint", "licence",
                                                                                 "training_data", "source", "sha256")},
            "frame_manifest": {"version": manifest["version"], "frames_sha256": manifest["frames_sha256"],
                               "frames_requested": len(frames), "sources": sorted({e["source_id"] for e in frames})},
            "confidence_floor": CONF_FLOOR, "box_convention": "xyxy pixels, full frame, top-left origin",
            "hardware": _hardware(adapter.device), "started_at_utc": datetime.now(timezone.utc).isoformat()}
    if isinstance(adapter, UltralyticsTiledAdapter):
        head["checkpoint_sha256_verified"] = sha256_of(adapter.weights)
    mpath.write_text(json.dumps(head, indent=1), encoding="utf-8")
    with dpath.open("a", encoding="utf-8", newline="\n") as fh:
        for e in frames:
            sid, fr = e["source_id"], int(e["frame"])
            if (sid, fr) in done:
                continue
            spec = SOURCES[sid]
            guard_not_reserved(sid, spec["match_id"], spec["video"])
            bgr = None
            if not isinstance(adapter, CacheAdapter):
                if sid not in readers:
                    readers = {sid: ExactFrameReader(Path(data_root) / spec["video"], cache_frames=2)}   # one source, 2 frames: low memory
                bgr = readers[sid].frame(fr)
            t0 = time.perf_counter()
            boxes = adapter.detect(sid, fr, bgr)
            dt = time.perf_counter() - t0
            peak_rss = max(peak_rss, proc.memory_info().rss)
            fh.write(json.dumps({"source_id": sid, "frame": fr, "seconds": round(dt, 4),
                                 "boxes": [[round(float(v), 2) for v in b[:4]] + [round(float(b[4]), 4), b[5], b[6]]
                                           for b in boxes]}, separators=(",", ":")) + "\n")
            fh.flush()
            done.add((sid, fr))
    head.update(adapter.describe() | {
        "status": "complete" if len(done) >= len(frames) else "partial", "frames_done": len(done),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(), "wall_seconds_this_session": round(time.perf_counter() - t_start, 1),
        "peak_rss_mb": round(peak_rss / 2**20, 1)})
    if adapter.device.startswith("cuda"):
        import torch

        head["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    if max_frames is not None:
        head["status"] = "complete" if len(done) >= len(frames) else "partial"
        head["subset"] = f"first {max_frames} manifest frames (bounded pilot)"
    mpath.write_text(json.dumps(head, indent=1), encoding="utf-8")
    return dest


def load_runs(out_root: Path) -> dict[str, dict[str, Any]]:
    root = runs_root(out_root)
    out = {}
    for m in sorted(root.glob("*/run_manifest.json")) if root.exists() else []:
        out[m.parent.name] = json.loads(m.read_text(encoding="utf-8"))
    return out


def load_dets(out_root: Path, key: str) -> dict[tuple[str, int], dict[str, Any]]:
    out = {}
    for line in (runs_root(out_root) / key / "dets.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[(r["source_id"], r["frame"])] = r
    return out
