"""Versioned cache of raw detector observations, for deterministic tracker replay.

A cache is a directory with two files:

* ``detections.jsonl``: one record per decoded frame, in order and without gaps.
  Each record holds every detector output at or above the cache's confidence floor
  (column order: ``OBSERVATION_COLUMNS``), including labels the pipeline does not map and
  boxes the turf or pitch-mask filters would reject, with each filter decision stored.
* ``cache_manifest.json``: schema version, completion status, record count, SHA-256 of the
  records, and the provenance needed to refuse a replay against the wrong video, window
  or detector (video hash, weights hash, detector configuration, frame window).

Two thresholds are recorded separately. The **cache floor** (``detector.confidence_floor``) is
the confidence the detector ran with: everything above it is stored. The **tracking-input
threshold** (``tracking_input.confidence_threshold``) is what the saving run actually passed
to the tracker; it is never below the floor. Both comparisons are strict (``confidence >
threshold``), as in Ultralytics' non-maximum suppression.

Replay feeds stored observations back through ``VisionTrackingPipeline.process_frame`` instead
of running the detector. A cache cannot contain anything below its floor, so a replay can
never recover detections the saving run did not keep.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from football_fatigue.vision.models import DetectedClass, Detection, RawObservation

CACHE_SCHEMA_VERSION = 2
MANIFEST_FILENAME = "cache_manifest.json"
RECORDS_FILENAME = "detections.jsonl"
OBSERVATION_COLUMNS = (
    "class_id", "class_label", "mapped_class", "confidence",
    "x1", "y1", "x2", "y2", "passes_turf_filter", "inside_pitch_mask",
)
REPLAY_COMPATIBILITY = "compatibility"
REPLAY_EXPERIMENTAL = "experimental"
REPLAY_MODES = (REPLAY_COMPATIBILITY, REPLAY_EXPERIMENTAL)

_STATUS_WRITING = "writing"
_STATUS_COMPLETE = "complete"
_STATUS_INCOMPLETE = "incomplete"
_MAPPED_VALUES = {c.value for c in DetectedClass}


class DetectionCacheError(ValueError):
    """The cache is missing, incomplete, corrupted, or does not match the requested replay."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detections_from_observations(
    observations: Sequence[RawObservation], min_confidence: float | None = None,
) -> list[Detection]:
    """Tracker input from stored observations: kept by the recorded filters and above the threshold.

    The threshold is exclusive, matching Ultralytics' ``conf`` semantics.
    """
    detections = []
    for observation in observations:
        if min_confidence is not None and not observation.confidence > min_confidence:
            continue
        detection = observation.to_detection()
        if detection is not None:
            detections.append(detection)
    return detections


def resolve_replay(manifest: dict[str, Any], mode: str, confidence: float) -> dict[str, Any]:
    """Validate a replay mode and tracking-input threshold against the cache; describe the replay.

    * ``compatibility`` reconstructs the saving run's tracking input, so the threshold must equal
      the saving run's tracking-input threshold.
    * ``experimental`` allows any threshold at or above the cache floor (for example a cache saved
      with floor 0.10 for a 0.40 run, replayed at 0.25); results are labelled experimental.
    * Either mode rejects a threshold below the floor: those detections were never stored.
    """
    if mode not in REPLAY_MODES:
        raise DetectionCacheError(f"unknown replay mode {mode!r}; expected one of {REPLAY_MODES}")
    floor = float(manifest["detector"]["confidence_floor"])
    saved = float(manifest["tracking_input"]["confidence_threshold"])
    if not math.isfinite(confidence) or confidence < floor:
        raise DetectionCacheError(
            f"confidence {confidence} is below the cache floor {floor}: detections under the floor "
            "were never stored and cannot be recovered by replay"
        )
    same_as_saved = math.isclose(confidence, saved, rel_tol=0.0, abs_tol=1e-9)
    if mode == REPLAY_COMPATIBILITY and not same_as_saved:
        raise DetectionCacheError(
            f"compatibility replay reconstructs the saving run's tracking input and needs confidence == {saved} "
            f"(cache floor {floor}); got {confidence}. Use --replay-mode {REPLAY_EXPERIMENTAL} to choose another "
            "threshold at or above the floor"
        )
    return {
        "mode": mode,
        "experimental": mode == REPLAY_EXPERIMENTAL,
        "tracking_confidence": confidence,
        "saving_run_tracking_confidence": saved,
        "cache_confidence_floor": floor,
        "equals_saving_run_tracking_confidence": same_as_saved,
    }


def _encode_tri(value: bool | None) -> int | None:
    return None if value is None else int(value)


def _decode_tri(value: Any, line: int, name: str) -> bool | None:
    if value is None:
        return None
    if value in (0, 1) and not isinstance(value, bool):
        return bool(value)
    raise DetectionCacheError(f"line {line}: {name} must be 0, 1 or null")


def _encode_observation(o: RawObservation) -> list[Any]:
    return [
        o.class_id, o.class_label, o.mapped_class.value if o.mapped_class is not None else None,
        o.confidence, *o.xyxy, _encode_tri(o.passes_turf_filter), _encode_tri(o.inside_pitch_mask),
    ]


def _decode_observation(row: Any, line: int) -> RawObservation:
    if not isinstance(row, list) or len(row) != len(OBSERVATION_COLUMNS):
        raise DetectionCacheError(f"line {line}: observation must have {len(OBSERVATION_COLUMNS)} columns")
    class_id, label, mapped, confidence, x1, y1, x2, y2, turf, mask = row
    if not isinstance(class_id, int) or isinstance(class_id, bool) or not isinstance(label, str):
        raise DetectionCacheError(f"line {line}: invalid class_id or class_label")
    if mapped is not None and mapped not in _MAPPED_VALUES:
        raise DetectionCacheError(f"line {line}: unknown mapped_class {mapped!r}")
    numbers = (confidence, x1, y1, x2, y2)
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in numbers):
        raise DetectionCacheError(f"line {line}: confidence and box must be finite numbers")
    if not 0.0 <= confidence <= 1.0:
        raise DetectionCacheError(f"line {line}: confidence must be within 0-1")
    return RawObservation(
        class_id=class_id, class_label=label,
        mapped_class=DetectedClass(mapped) if mapped is not None else None,
        confidence=float(confidence), xyxy=(float(x1), float(y1), float(x2), float(y2)),
        passes_turf_filter=_decode_tri(turf, line, "passes_turf_filter"),
        inside_pitch_mask=_decode_tri(mask, line, "inside_pitch_mask"),
    )


class DetectionCacheWriter:
    """Streams observations for consecutive frames into a new cache directory.

    The directory must not exist (nothing is overwritten) and its parent must. The manifest
    reads ``writing`` until ``close(complete=True)``; a crashed or failed run therefore
    leaves a cache that readers refuse.
    """

    def __init__(self, directory: Path, provenance: dict[str, Any]) -> None:
        for section, keys in (
            ("video", ("sha256", "fps", "width", "height")),
            ("detector", ("weights_sha256", "confidence_floor")),
            ("tracking_input", ("confidence_threshold",)),
            ("window", ("start_frame",)),
        ):
            missing = [k for k in keys if k not in provenance.get(section, {})]
            if missing:
                raise DetectionCacheError(f"provenance {section} is missing {missing}")
        if provenance["tracking_input"]["confidence_threshold"] < provenance["detector"]["confidence_floor"]:
            raise DetectionCacheError("the tracking-input threshold cannot be below the cache floor")
        self._tracking_threshold = float(provenance["tracking_input"]["confidence_threshold"])
        self.directory = Path(directory)
        if self.directory.exists():
            raise DetectionCacheError(f"detection cache already exists; it is never overwritten: {self.directory}")
        self._provenance = provenance
        self._next_frame = int(provenance["window"]["start_frame"])
        self._first_frame: int | None = None
        self._frames = 0
        self._counts: Counter[str] = Counter()
        self._closed = False
        self.directory.mkdir(parents=False, exist_ok=False)
        self._stream = (self.directory / RECORDS_FILENAME).open("x", encoding="utf-8", newline="\n")
        self._write_manifest(_STATUS_WRITING)

    def write_frame(self, frame_num: int, timestamp_s: float, observations: Sequence[RawObservation]) -> None:
        if self._closed:
            raise DetectionCacheError("cache writer is closed")
        if frame_num != self._next_frame:
            raise DetectionCacheError(
                f"frames must be written consecutively: expected {self._next_frame}, got {frame_num}"
            )
        record = {
            "frame_num": frame_num,
            "timestamp_s": timestamp_s,
            "observations": [_encode_observation(o) for o in observations],
        }
        line = json.dumps(record, allow_nan=False, separators=(",", ":"))
        self._stream.write(line + "\n")
        if self._first_frame is None:
            self._first_frame = frame_num
        self._frames += 1
        self._next_frame = frame_num + 1
        self._counts["observations"] += len(observations)
        for o in observations:
            if o.kept:
                self._counts["kept"] += 1
                if o.confidence > self._tracking_threshold:
                    self._counts["tracking_input"] += 1
            self._counts[f"mapped_{o.mapped_class.value if o.mapped_class else 'unmapped'}"] += 1

    def close(
        self, *, complete: bool, error: str | None = None, class_names: dict[str, str] | None = None,
        detector_extra: dict[str, Any] | None = None,
    ) -> None:
        """Finalise the manifest. Idempotent; the first call decides the status."""
        if self._closed:
            return
        self._closed = True
        self._stream.close()
        if class_names is not None:
            self._provenance["detector"]["class_names"] = class_names
        if detector_extra:
            self._provenance["detector"].update(detector_extra)  # e.g. merge statistics gathered during the run
        self._write_manifest(
            _STATUS_COMPLETE if complete and self._frames > 0 else _STATUS_INCOMPLETE,
            error=error if error else (None if complete and self._frames > 0 else "no frames were written"),
        )

    def _write_manifest(self, status: str, error: str | None = None) -> None:
        records = self.directory / RECORDS_FILENAME
        finished = status != _STATUS_WRITING
        manifest = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "status": status,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "columns": list(OBSERVATION_COLUMNS),
            "confidence_semantics": "stored observations satisfy confidence > detector.confidence_floor; "
                                    "the saving run tracked observations with confidence > tracking_input.confidence_threshold",
            **self._provenance,
            "frames_written": self._frames,
            "first_frame": self._first_frame,
            "last_frame": (self._next_frame - 1) if self._frames else None,
            "counts": dict(self._counts),
            "records_bytes": records.stat().st_size if finished else None,
            "records_sha256": file_sha256(records) if finished else None,
            "error": error,
        }
        target = self.directory / MANIFEST_FILENAME
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, target)


class DetectionCacheReader:
    """A validated, complete detection cache.

    Construction checks the schema version, completion status, columns, file size, record
    count and SHA-256, and raises ``DetectionCacheError`` for anything incomplete or altered.
    ``path`` may be the cache directory or a run directory containing ``detection_cache/``.
    """

    def __init__(self, path: Path, *, verify_hash: bool = True) -> None:
        directory = Path(path)
        if not (directory / MANIFEST_FILENAME).is_file() and (directory / "detection_cache" / MANIFEST_FILENAME).is_file():
            directory = directory / "detection_cache"
        self.directory = directory
        manifest_path = directory / MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise DetectionCacheError(f"not a detection cache (no {MANIFEST_FILENAME}): {directory}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise DetectionCacheError(f"cache manifest is not valid JSON: {exc}") from exc
        if not isinstance(manifest, dict):
            raise DetectionCacheError("cache manifest must be a JSON object")
        if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise DetectionCacheError(
                f"unsupported cache schema version {manifest.get('schema_version')!r}; "
                f"this code reads version {CACHE_SCHEMA_VERSION}"
            )
        if manifest.get("status") != _STATUS_COMPLETE:
            raise DetectionCacheError(
                f"cache is not complete (status {manifest.get('status')!r}: {manifest.get('error')})"
            )
        if manifest.get("columns") != list(OBSERVATION_COLUMNS):
            raise DetectionCacheError("cache columns do not match this reader")
        for section, keys in (
            ("video", ("sha256", "fps", "width", "height")),
            ("detector", ("weights_sha256", "confidence_floor")),
            ("tracking_input", ("confidence_threshold",)),
            ("window", ("start_frame",)),
        ):
            missing = [k for k in keys if k not in (manifest.get(section) or {})]
            if missing:
                raise DetectionCacheError(f"cache manifest {section} is missing {missing}")
        if manifest["tracking_input"]["confidence_threshold"] < manifest["detector"]["confidence_floor"]:
            raise DetectionCacheError("cache manifest has a tracking-input threshold below its floor")
        self.manifest: dict[str, Any] = manifest
        self.records_path = directory / RECORDS_FILENAME
        if not self.records_path.is_file():
            raise DetectionCacheError(f"cache records file is missing: {self.records_path}")
        frames = manifest.get("frames_written")
        first, last = manifest.get("first_frame"), manifest.get("last_frame")
        if (
            not all(isinstance(v, int) and not isinstance(v, bool) for v in (frames, first, last))
            or frames < 1 or last != first + frames - 1
        ):
            raise DetectionCacheError("cache frame window is inconsistent")
        if self.records_path.stat().st_size != manifest.get("records_bytes"):
            raise DetectionCacheError("cache records size differs from the manifest (truncated or modified)")
        if verify_hash:
            if file_sha256(self.records_path) != manifest.get("records_sha256"):
                raise DetectionCacheError("cache records SHA-256 differs from the manifest (corrupted or modified)")
        self.manifest_sha256 = file_sha256(manifest_path)

    @property
    def first_frame(self) -> int:
        return int(self.manifest["first_frame"])

    @property
    def last_frame(self) -> int:
        return int(self.manifest["last_frame"])

    @property
    def confidence_floor(self) -> float:
        """Lowest confidence the stored observations can contain (the detector's inference floor)."""
        return float(self.manifest["detector"]["confidence_floor"])

    @property
    def tracking_confidence(self) -> float:
        """The threshold the saving run applied to build its tracker input."""
        return float(self.manifest["tracking_input"]["confidence_threshold"])

    def validate_for_replay(
        self, *, video_sha256: str, fps: float, width: int, height: int,
        first_frame: int, last_frame: int, weights_sha256: str | None = None,
    ) -> None:
        """Refuse a replay against a different video, geometry, detector weights or frame window."""
        video = self.manifest["video"]
        if video["sha256"] != video_sha256:
            raise DetectionCacheError("cache was saved from a different video (SHA-256 mismatch)")
        if (int(video["width"]), int(video["height"])) != (width, height):
            raise DetectionCacheError("cache video dimensions differ from the replay video")
        if not math.isclose(float(video["fps"]), float(fps), rel_tol=0.0, abs_tol=1e-6):
            raise DetectionCacheError("cache video frame rate differs from the replay video")
        if weights_sha256 is not None and weights_sha256 != self.manifest["detector"]["weights_sha256"]:
            raise DetectionCacheError("--weights differs from the model that produced the cache (SHA-256 mismatch)")
        if first_frame < self.first_frame or last_frame > self.last_frame:
            raise DetectionCacheError(
                f"replay frames {first_frame}-{last_frame} are outside the cached window "
                f"{self.first_frame}-{self.last_frame}"
            )

    def iter_records(self, start_frame: int | None = None) -> Iterator[tuple[int, float, list[RawObservation]]]:
        """Validated ``(frame_num, timestamp_s, observations)`` records, optionally from ``start_frame``."""
        expected = self.first_frame
        with self.records_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    record = json.loads(line)
                    frame_num, timestamp_s, rows = record["frame_num"], record["timestamp_s"], record["observations"]
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise DetectionCacheError(f"line {line_number}: unreadable record ({exc})") from exc
                if frame_num != expected:
                    raise DetectionCacheError(f"line {line_number}: expected frame {expected}, found {frame_num}")
                expected += 1
                if start_frame is not None and frame_num < start_frame:
                    continue
                yield frame_num, float(timestamp_s), [_decode_observation(r, line_number) for r in rows]
        if expected - 1 != self.last_frame:
            raise DetectionCacheError("cache holds fewer records than its manifest states")

    def detection_source(self, start_frame: int, min_confidence: float) -> Callable[[int], list[Detection]]:
        """A sequential per-frame source for ``VisionTrackingPipeline.process_video_generator``."""
        records = self.iter_records(start_frame)

        def source(frame_idx: int) -> list[Detection]:
            try:
                frame_num, _, observations = next(records)
            except StopIteration:
                raise DetectionCacheError(f"cache has no record for frame {frame_idx}") from None
            if frame_num != frame_idx:
                raise DetectionCacheError(f"replay is out of step: asked for frame {frame_idx}, cache gave {frame_num}")
            return detections_from_observations(observations, min_confidence)

        return source
