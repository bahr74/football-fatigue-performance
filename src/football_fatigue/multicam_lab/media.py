"""Media access for the lab: exact frames by index (PyAV, PTS-checked), video validation, and HTTP range streaming.

Frames are decoded from the original files read-only; nothing is re-encoded or written next to the originals.
"""

from __future__ import annotations

import mimetypes
import threading
from collections import OrderedDict
from pathlib import Path

import av
import cv2
import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".m4v", ".mov", ".mkv", ".avi", ".webm"}
MIME = {".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm", ".mkv": "video/x-matroska",
        ".avi": "video/x-msvideo", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".json": "application/json",
        ".js": "text/javascript", ".css": "text/css", ".html": "text/html; charset=utf-8", ".csv": "text/csv", ".md": "text/markdown; charset=utf-8"}


def mime_type(path: Path) -> str:
    return MIME.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def sniff_container(head: bytes) -> str | None:
    """Container from magic bytes (not from the file name)."""
    if len(head) >= 12 and head[4:8] == b"ftyp":
        return "mp4/mov"
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return "matroska/webm"
    if head[:4] == b"RIFF" and head[8:12] == b"AVI ":
        return "avi"
    return None


def probe_video(path: Path) -> dict:
    """Validate a video by opening it and decoding its first frame. Raises ValueError with a readable reason."""
    path = Path(path)
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"unsupported file extension {path.suffix!r}; allowed: {', '.join(sorted(VIDEO_EXTENSIONS))}")
    with open(path, "rb") as fh:
        head = fh.read(16)
    container = sniff_container(head)
    if container is None:
        raise ValueError("file content is not a recognised video container (MP4/MOV, MKV/WebM or AVI)")
    try:
        c = av.open(str(path))
    except Exception as exc:                                   # noqa: BLE001 - surface the decoder's reason
        raise ValueError(f"video cannot be opened: {exc}") from exc
    try:
        if not c.streams.video:
            raise ValueError("file has no video stream")
        vs = c.streams.video[0]; tb = float(vs.time_base) if vs.time_base else None
        fps = float(vs.average_rate) if vs.average_rate else None
        frame = next(c.decode(vs), None)
        if frame is None:
            raise ValueError("no decodable video frame")
        duration = float(c.duration / 1e6) if c.duration else (float(vs.duration * tb) if vs.duration and tb else None)
        if not fps or fps > 240 or not duration or duration <= 0:   # broken timestamps (e.g. every frame at pts 0) would misplace every frame
            raise ValueError(f"video timestamps are not usable (frame rate {fps or 0:g} fps, duration {duration or 0:g} s); re-encode the file with valid timestamps")
        frames = vs.frames or (int(round(duration * fps)) if duration and fps else None)
        return {"container": container, "codec": vs.codec_context.name, "width": frame.width, "height": frame.height, "fps": fps,
                "duration_s": duration, "frame_count": frames, "first_pts_s": float(frame.pts * tb) if frame.pts is not None and tb else 0.0,
                "has_audio": bool(c.streams.audio), "size_bytes": path.stat().st_size}
    finally:
        c.close()


class FrameReader:
    """Exact frame access by index for constant-frame-rate files (frame i has PTS first_pts + i/fps). Keeps one open decoder
    per video and decodes forward when the request is just ahead; otherwise seeks to the preceding keyframe."""

    def __init__(self, path: Path, cache_frames: int = 240):
        self.path = Path(path); self._lock = threading.Lock(); self._cache: OrderedDict[int, np.ndarray] = OrderedDict(); self._max = cache_frames
        self._c = av.open(str(self.path)); self._vs = self._c.streams.video[0]; self._tb = float(self._vs.time_base)
        self.fps = float(self._vs.average_rate); self._dec = None; self._last = -10

    def _index_of(self, frame) -> int:
        return int(round(frame.pts * self._tb * self.fps))

    def frame(self, index: int) -> np.ndarray | None:
        with self._lock:
            if index in self._cache:
                self._cache.move_to_end(index); return self._cache[index]
            if self._dec is None or not (self._last < index <= self._last + 90):
                self._c.seek(int(max(0.0, index / self.fps - 1.0) / self._tb), stream=self._vs, backward=True)
                self._dec = self._c.decode(self._vs); self._last = -10
            for fr in self._dec:
                i = self._index_of(fr); self._last = i
                if i < index - 1:                               # skip without converting
                    continue
                img = fr.to_ndarray(format="bgr24"); self._cache[i] = img
                if len(self._cache) > self._max:
                    self._cache.popitem(last=False)
                if i >= index:
                    break
            return self._cache.get(index)

    def jpeg(self, index: int, quality: int = 85, max_width: int | None = None) -> bytes | None:
        img = self.frame(index)
        if img is None:
            return None
        if max_width and img.shape[1] > max_width:
            img = cv2.resize(img, (max_width, int(round(img.shape[0] * max_width / img.shape[1]))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None

    def close(self) -> None:
        with self._lock:
            self._c.close()


def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """'bytes=a-b' -> (start, end) inclusive, clamped; None for no/invalid range (serve whole file)."""
    if not header or not header.startswith("bytes="):
        return None
    spec = header[6:].split(",")[0].strip()
    a, _, b = spec.partition("-")
    try:
        if a == "":
            n = int(b); start, end = max(0, size - n), size - 1
        else:
            start = int(a); end = int(b) if b else size - 1
    except ValueError:
        return None
    if start >= size or start > end:
        return (size, size)                                    # unsatisfiable
    return start, min(end, size - 1)
