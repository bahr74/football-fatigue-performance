"""Production ingestion layer (Milestone 2A): typed, read-only, streaming loaders and quality reporting.

    from football_fatigue.ingestion import Dataset, EventVersion
    ds = Dataset("data/raw")
    meta = ds.metadata("3812"); roster = ds.roster("3812")
    events = ds.load_events("3812", EventVersion.ROOT)          # the version must be named
    for frame in ds.iter_tracking_frames("3812", roster=roster):  # streaming, bounded memory
        ...

Out of scope here: event-to-tracking synchronization, smoothing or interpolation, metrics, playback.
"""

from .loaders import Dataset, as_float, as_int, validate_match_id
from .models import (
    AttributionStatus,
    BallSample,
    EventRow,
    EventVersion,
    IngestionError,
    MatchMetadata,
    PlayerSample,
    QualityFlag,
    Roster,
    RosterEntry,
    SmoothedCandidate,
    SmoothedStatus,
    SourceRef,
    TrackingFrame,
)
from .validation import MatchQualityReport, approved_output_root, quality_report, safe_output_path, write_quality_report

__all__ = [
    "AttributionStatus", "BallSample", "Dataset", "EventRow", "EventVersion", "IngestionError", "MatchMetadata",
    "MatchQualityReport", "PlayerSample", "QualityFlag", "Roster", "RosterEntry", "SmoothedCandidate", "SmoothedStatus",
    "SourceRef", "TrackingFrame", "approved_output_root", "as_float", "as_int", "quality_report", "safe_output_path",
    "validate_match_id", "write_quality_report",
]
