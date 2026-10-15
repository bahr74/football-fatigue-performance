"""Event-to-tracking temporal alignment (Milestone 2B), built on the ingestion API.

Policy (carried over from the Milestone 1B audit, ``scripts/audit/audit_lib.py::Aligner``; provisional):

* time base: event ``event_time_s`` and frame ``video_time_s``, both seconds of video time;
* an event may match only frames of the **same canonical period** (invalid or absent periods never match);
* the nearest frame in time wins; an exact tie (difference below 1 ns) goes to the **earlier** frame;
* the single maximum acceptable residual is one frame interval (``1 / fps``) unless a tolerance is given;
* no extrapolation, no interpolation, no invented frames: events outside a period's frame range stay unmatched;
* every event is preserved, matched or not, in source order, with its provenance and the selected frame's.

Three different claims are kept apart and never merged:

1. **temporal coverage** - ``MatchStatus.MATCHED``: a same-period frame exists within the tolerance;
2. **coordinate agreement** - ``SpatialAssessment`` (optional): the event's embedded player/ball coordinates equal the
   frame's *raw* tracking positions. It uses raw observations only, skips quarantined players and frames whose raw
   data conflict between repeated records, and corrects nothing;
3. **validated physical correspondence** - never assessed here (``VALIDATED_CORRESPONDENCE``); it needs independent
   ground truth such as video.

Cost: tracking frames are consumed once, in file order, with O(1) frame memory; events are buffered, which is
O(E) memory (plus O(E x players) only when spatial assessment is enabled) and O(E log E) to sort; each frame touches
only the events within one tolerance of it, so total time is O(F + E log E + sum of window sizes).
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .ingestion import Dataset, EventRow, EventVersion, IngestionError, QualityFlag, SourceRef, TrackingFrame, as_float, as_int, validate_match_id
from .ingestion.models import AttributionStatus

POLICY_VERSION = "2b-1"
VALIDATED_CORRESPONDENCE = "not_assessed"
TIE_EPSILON_S = 1e-9
SPATIAL_NEAR_M = 0.01          # agreement tolerance (source precision is 0.001 m)
SPATIAL_AGREE_FRACTION = 0.9
SPATIAL_MIN_PLAYERS = 11


class SynchronizationError(IngestionError):
    """The inputs cannot be aligned under the policy (for example frames that are not in increasing time order)."""


class _ContextError(SynchronizationError):
    """Inputs of another match or version: always raised to the caller, never absorbed as a failed scan."""


class MatchStatus(str, Enum):
    MATCHED = "matched"
    UNMATCHED = "unmatched"


class UnmatchedReason(str, Enum):
    MISSING_EVENT_TIME = "missing_event_time"
    INVALID_SOURCE_PERIOD = "invalid_source_period"
    NO_TRACKING_FOR_PERIOD = "no_tracking_for_period"
    OUTSIDE_PERIOD_FRAME_RANGE = "outside_period_frame_range"
    NO_FRAME_WITHIN_TOLERANCE = "no_frame_within_tolerance"
    PERIOD_LABEL_DIFFERS_FROM_NEAREST_FRAME = "period_label_differs_from_nearest_frame"
    TRACKING_SCAN_INCOMPLETE = "tracking_scan_incomplete"   # the tracking stream failed before this event could be settled


class SpatialStatus(str, Enum):
    NOT_ASSESSED = "not_assessed"                    # spatial assessment was not requested, or the event is unmatched
    COORDINATE_AGREEING = "coordinate_agreeing"
    COORDINATE_DISAGREEING = "coordinate_disagreeing"
    NOT_ASSESSABLE = "not_assessable"                # requested, but valid comparable observations are missing


class TemporalNote(str, Enum):
    NEAR_PERIOD_BOUNDARY = "NEAR_PERIOD_BOUNDARY"    # within one tolerance of the first or last frame of the period


@dataclass(frozen=True, slots=True)
class SpatialAssessment:
    status: SpatialStatus = SpatialStatus.NOT_ASSESSED
    reason: str | None = None
    players_compared: int = 0
    players_within_tolerance: int = 0
    ball: str = "not_assessed"        # not_assessed | agrees | differs | not_assessable
    tolerance_m: float = SPATIAL_NEAR_M
    players_excluded_ambiguous: int = 0   # (side, jersey) groups with conflicting duplicate coordinates, excluded from the comparison


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    # event side (provenance preserved)
    event_row_id: str
    match_id: int
    event_version: EventVersion
    source_row_index: int
    event_source: SourceRef | None
    event_time_s: float | None
    source_period: int | None
    canonical_period: int | None
    event_quality_flags: frozenset[QualityFlag]
    # temporal result
    status: MatchStatus
    reason: UnmatchedReason | None
    residual_s: float | None              # event time minus frame time (signed)
    tolerance_s: float
    # selected frame (None when unmatched; no frame is ever invented)
    frame_num: int | None = None
    frame_video_time_s: float | None = None
    frame_period: int | None = None
    frame_source: SourceRef | None = None
    frame_source_record_indices: tuple[int, ...] = ()
    frame_quality_flags: frozenset[QualityFlag] = frozenset()
    temporal_notes: frozenset[TemporalNote] = frozenset()
    spatial: SpatialAssessment = field(default_factory=SpatialAssessment)
    validated_correspondence: str = VALIDATED_CORRESPONDENCE
    policy_version: str = POLICY_VERSION

    @property
    def quality_label(self) -> str:
        """Strongest claim supported here: unmatched < temporal_only < coordinate_agreeing (never 'validated')."""
        if self.status is MatchStatus.UNMATCHED:
            return "unmatched"
        if self.spatial.status is SpatialStatus.COORDINATE_AGREEING:
            return "coordinate_agreeing"
        if self.spatial.status is SpatialStatus.COORDINATE_DISAGREEING:
            return "temporal_only_coordinates_disagree"
        return "temporal_only"


@dataclass(frozen=True, slots=True)
class MatchSynchronization:
    match_id: int
    event_version: EventVersion
    fps: float
    tolerance_s: float
    alignments: tuple[AlignmentResult, ...]        # one per event row, in source order, none dropped
    scan_complete: bool
    fatal_error: str | None
    tracking_period_ranges: dict[int, tuple[float, float, int]]   # canonical period -> (first s, last s, frames)
    spatial_assessed: bool
    policy_version: str = POLICY_VERSION
    ordering_violation: bool = False   # tracking frames went backwards in time: no result of this run is definitive

    def summary(self) -> dict[str, Any]:
        a = self.alignments
        matched = [x for x in a if x.status is MatchStatus.MATCHED]
        res = [abs(x.residual_s) * 1000 for x in matched]
        return dict(
            match_id=self.match_id, event_version=self.event_version.value, events=len(a), matched=len(matched), unmatched=len(a) - len(matched),
            unmatched_reasons=dict(Counter(x.reason.value for x in a if x.reason)), scan_complete=self.scan_complete, fatal_error=self.fatal_error, ordering_violation=self.ordering_violation,
            tolerance_ms=self.tolerance_s * 1000, residual_abs_ms_median=statistics.median(res) if res else None, residual_abs_ms_max=max(res) if res else None,
            near_period_boundary=sum(TemporalNote.NEAR_PERIOD_BOUNDARY in x.temporal_notes for x in a),
            quality_labels=dict(Counter(x.quality_label for x in a)), spatial_assessed=self.spatial_assessed,
            validated_correspondence=VALIDATED_CORRESPONDENCE, policy_version=self.policy_version,
            tracking_periods={str(k): dict(first_video_s=v[0], last_video_s=v[1], frames=v[2]) for k, v in sorted(self.tracking_period_ranges.items())},
        )


# ---------------------------------------------------------------- spatial comparison (raw observations only)

def _resolve_groups(groups: dict[tuple[str, int], set[tuple[float, float]]]) -> tuple[dict[tuple[str, int], tuple[float, float]], set[tuple[str, int]]]:
    """One position per (side, jersey) when all valid duplicates agree; groups with conflicting coordinates are ambiguous.
    The outcome depends only on the set of coordinates, never on entry order."""
    return ({k: next(iter(v)) for k, v in groups.items() if len(v) == 1}, {k for k, v in groups.items() if len(v) > 1})


def _event_players(row: EventRow) -> tuple[dict[tuple[str, int], tuple[float, float]], set[tuple[str, int]]]:
    groups: dict[tuple[str, int], set[tuple[float, float]]] = {}
    for side, key in (("home", "homePlayers"), ("away", "awayPlayers")):
        lst = row.positions_snapshot.get(key)
        for e in lst if isinstance(lst, list) else []:
            if isinstance(e, dict):
                j, x, y = as_int(e.get("jerseyNum")), as_float(e.get("x")), as_float(e.get("y"))
                if j is not None and x is not None and y is not None:     # invalid coordinates are skipped, as before
                    groups.setdefault((side, j), set()).add((x, y))
    return _resolve_groups(groups)


def _frame_snapshot(frame: TrackingFrame) -> dict[str, Any]:
    """Compact, raw-only observations of the selected frame (players of quarantined periods are excluded)."""
    unreliable = set(frame.repeat_conflict_fields)
    groups: dict[tuple[str, int], set[tuple[float, float]]] = {}
    if not unreliable & {"homePlayers", "awayPlayers"}:
        for p in frame.players:
            if p.jersey_number is None or p.x_raw_m is None or p.y_raw_m is None or p.attribution_status is AttributionStatus.QUARANTINED:
                continue
            groups.setdefault((p.source_side, p.jersey_number), set()).add((p.x_raw_m, p.y_raw_m))
    players, ambiguous = _resolve_groups(groups)
    ball = None
    if (frame.ball is not None and frame.ball.x_raw_m is not None and frame.ball.y_raw_m is not None and "balls" not in unreliable
            and frame.ball.raw_count == 1):          # several raw balls would make "the" ball order-dependent
        ball = (frame.ball.x_raw_m, frame.ball.y_raw_m)
    return dict(players=players, ambiguous=ambiguous, ball=ball, players_conflict=bool(unreliable & {"homePlayers", "awayPlayers"}), ball_conflict="balls" in unreliable)


def assess_spatial(row: EventRow, snap: dict[str, Any], tol_m: float = SPATIAL_NEAR_M) -> SpatialAssessment:
    """Coordinate agreement of one matched event with its frame; raw observations only, reported separately from timing."""
    ev, ev_amb = _event_players(row)
    both = (ev.keys() | ev_amb) & (snap["players"].keys() | snap["ambiguous"])   # jerseys present on both sides
    excluded = both & (ev_amb | snap["ambiguous"])                               # ...of which at least one side is ambiguous
    d = [math.hypot(x - snap["players"][k][0], y - snap["players"][k][1]) for k, (x, y) in ev.items() if k in snap["players"]]
    near = sum(1 for v in d if v <= tol_m)
    eb = row.positions_snapshot.get("ball")
    pts = {(as_float(b.get("x")), as_float(b.get("y"))) for b in (eb if isinstance(eb, list) else []) if isinstance(b, dict)}
    pts = {q for q in pts if None not in q}
    if len(pts) != 1 or snap["ball"] is None:        # no valid event ball, or conflicting event balls, or no reliable frame ball
        ball = "not_assessable"
    else:
        (ebx, eby), = pts
        ball = "agrees" if math.hypot(ebx - snap["ball"][0], eby - snap["ball"][1]) <= tol_m else "differs"
    if len(d) < SPATIAL_MIN_PLAYERS:
        if snap["players_conflict"]:
            reason = "frame_player_data_conflict"
        elif excluded:
            reason = "ambiguous_duplicate_jerseys_excluded; fewer_than_%d_comparable_players" % SPATIAL_MIN_PLAYERS
        else:
            reason = "fewer_than_%d_comparable_players" % SPATIAL_MIN_PLAYERS
        return SpatialAssessment(SpatialStatus.NOT_ASSESSABLE, reason, len(d), near, ball, tol_m, len(excluded))
    status = SpatialStatus.COORDINATE_AGREEING if near / len(d) >= SPATIAL_AGREE_FRACTION else SpatialStatus.COORDINATE_DISAGREEING
    return SpatialAssessment(status, None, len(d), near, ball, tol_m, len(excluded))


# ---------------------------------------------------------------- the temporal join

def align_events(events: Iterable[EventRow], frames: Iterable[TrackingFrame], *, match_id: int, event_version: EventVersion, fps: float,
                 tolerance_s: float | None = None, spatial: bool = False, spatial_tolerance_m: float = SPATIAL_NEAR_M) -> MatchSynchronization:
    """Streaming temporal join of event rows to tracking frames (frames must arrive in increasing video-time order).

    Events must belong to ``match_id`` and ``event_version`` and frames to ``match_id``; otherwise a ``SynchronizationError``
    is raised (an alignment of foreign inputs is never returned). If frames go backwards in time, nothing is definitive:
    every event that depends on tracking becomes ``TRACKING_SCAN_INCOMPLETE`` and ``ordering_violation`` is set.

    A tracking error raised while iterating ``frames`` is not propagated: the result records it (``scan_complete=False``,
    ``fatal_error``), events that the unread part of the stream could still change are ``TRACKING_SCAN_INCOMPLETE``,
    and no absence claim (missing period, outside range) is made.
    """
    if not (isinstance(fps, (int, float)) and math.isfinite(fps) and fps > 0):
        raise SynchronizationError(f"fps must be a positive finite number, got {fps!r}")
    tol = (1.0 / fps) if tolerance_s is None else float(tolerance_s)
    if not (math.isfinite(tol) and tol > 0):
        raise SynchronizationError(f"tolerance must be a positive finite number of seconds, got {tolerance_s!r}")
    if not isinstance(event_version, EventVersion):
        raise _ContextError("event_version must be an EventVersion")
    rows = list(events)                                                       # O(E) buffer
    for r in rows:                                                            # context is enforced on the public API itself
        if r.match_id != match_id:
            raise _ContextError(f"event {r.event_row_id} belongs to match {r.match_id}, not the requested match {match_id}")
        if r.source_version is not event_version:
            raise _ContextError(f"event {r.event_row_id} has version {r.source_version.value}, not the requested {event_version.value}")
    order = sorted((i for i, r in enumerate(rows) if r.event_time_s is not None), key=lambda i: (rows[i].event_time_s, i))
    times = [rows[i].event_time_s for i in order]
    best: dict[int, tuple] = {}        # row index -> (abs_dt, signed_dt, frame summary, snapshot|None)
    best_any: set[int] = set()         # rows with some frame (any period) within tolerance
    ranges: dict[int, list] = {}       # canonical period -> [first_s, last_s, frames]
    ptr = 0
    last_t: float | None = None
    scan_error: str | None = None
    ordering_violation = False
    try:
        for fr in frames:
            if fr.match_id != match_id:
                raise _ContextError(f"tracking frame {fr.frame_num} belongs to match {fr.match_id}, not the requested match {match_id}")
            t = fr.video_time_s
            if last_t is not None and t < last_t - TIE_EPSILON_S:
                ordering_violation = True
                raise SynchronizationError(f"tracking frames are not in increasing time order (frame {fr.frame_num} at {t} s after {last_t} s)")
            last_t = t
            if fr.canonical_period is not None:
                rg = ranges.setdefault(fr.canonical_period, [t, t, 0])
                rg[1] = t
                rg[2] += 1
            while ptr < len(times) and times[ptr] < t - tol - TIE_EPSILON_S:
                ptr += 1
            snap = None
            j = ptr
            while j < len(times) and times[j] <= t + tol + TIE_EPSILON_S:
                i = order[j]
                dt = times[j] - t
                if abs(dt) <= tol + TIE_EPSILON_S:
                    best_any.add(i)
                    if fr.canonical_period is not None and rows[i].canonical_period == fr.canonical_period:
                        cur = best.get(i)
                        if cur is None or abs(dt) < cur[0] - TIE_EPSILON_S:     # strictly closer only: ties keep the earlier frame
                            if spatial and snap is None:
                                snap = _frame_snapshot(fr)
                            best[i] = (abs(dt), dt, (fr.frame_num, t, fr.canonical_period, fr.source, fr.source_record_indices, fr.quality_flags), snap)
                j += 1
    except _ContextError:
        raise
    except IngestionError as e:       # includes SynchronizationError and tracking structural errors
        scan_error = f"{type(e).__name__}: {e}"
    complete = scan_error is None
    settled_before = None if complete or last_t is None else last_t - tol
    out: list[AlignmentResult] = []
    for i, r in enumerate(rows):
        base = dict(event_row_id=r.event_row_id, match_id=r.match_id, event_version=r.source_version, source_row_index=r.source_row_index,
                    event_source=r.source, event_time_s=r.event_time_s, source_period=r.source_period, canonical_period=r.canonical_period,
                    event_quality_flags=r.quality_flags, tolerance_s=tol)
        if i in best and not ordering_violation and (complete or r.event_time_s < settled_before):
            ad, dt, (fn, ft, fp, fsrc, fidx, fflags), snap = best[i]
            rg = ranges.get(fp)
            notes = frozenset({TemporalNote.NEAR_PERIOD_BOUNDARY}) if rg and (r.event_time_s <= rg[0] + tol or r.event_time_s >= rg[1] - tol) else frozenset()
            sp = assess_spatial(r, snap, spatial_tolerance_m) if spatial and snap is not None else SpatialAssessment()
            out.append(AlignmentResult(**base, status=MatchStatus.MATCHED, reason=None, residual_s=dt, frame_num=fn, frame_video_time_s=ft, frame_period=fp,
                                       frame_source=fsrc, frame_source_record_indices=tuple(fidx), frame_quality_flags=fflags, temporal_notes=notes, spatial=sp))
            continue
        out.append(AlignmentResult(**base, status=MatchStatus.UNMATCHED, reason=_reason(r, i, best_any, ranges, tol, complete, settled_before, ordering_violation), residual_s=None))
    return MatchSynchronization(match_id, event_version, fps, tol, tuple(out), complete, scan_error,
                                {k: (v[0], v[1], v[2]) for k, v in ranges.items()}, spatial, ordering_violation=ordering_violation)


def _reason(r: EventRow, i: int, best_any: set[int], ranges: dict, tol: float, complete: bool, settled_before: float | None,
            ordering_violation: bool = False) -> UnmatchedReason:
    if r.event_time_s is None:
        return UnmatchedReason.MISSING_EVENT_TIME
    if r.canonical_period is None:
        return UnmatchedReason.INVALID_SOURCE_PERIOD
    if ordering_violation:   # a backward jump breaks the sorted-stream assumption behind every match and absence conclusion
        return UnmatchedReason.TRACKING_SCAN_INCOMPLETE
    if not complete and (settled_before is None or r.event_time_s >= settled_before):
        return UnmatchedReason.TRACKING_SCAN_INCOMPLETE     # absence cannot be asserted from a failed scan
    if i in best_any:
        return UnmatchedReason.PERIOD_LABEL_DIFFERS_FROM_NEAREST_FRAME
    rg = ranges.get(r.canonical_period)
    if rg is None:
        return UnmatchedReason.NO_TRACKING_FOR_PERIOD if complete else UnmatchedReason.TRACKING_SCAN_INCOMPLETE
    if r.event_time_s < rg[0] - tol - TIE_EPSILON_S or r.event_time_s > rg[1] + tol + TIE_EPSILON_S:
        return UnmatchedReason.OUTSIDE_PERIOD_FRAME_RANGE if complete else UnmatchedReason.TRACKING_SCAN_INCOMPLETE
    return UnmatchedReason.NO_FRAME_WITHIN_TOLERANCE


def synchronize_match(dataset: Dataset, match_id: object, version: EventVersion, *, tolerance_s: float | None = None, spatial: bool = False,
                      spatial_tolerance_m: float = SPATIAL_NEAR_M, quarantine_periods: Iterable[int] = ()) -> MatchSynchronization:
    """Align every event row of one explicit event version to the match's tracking frames (one streaming pass).

    ``spatial=True`` adds the separate coordinate-agreement assessment (raw observations only). ``quarantine_periods``
    withholds team/player attribution of those periods from spatial assessment; nothing is swapped or corrected.
    """
    if not isinstance(version, EventVersion):
        raise SynchronizationError("an EventVersion must be named explicitly")
    mid = validate_match_id(match_id)
    meta = dataset.metadata(mid)
    events = dataset.load_events(mid, version)                                # source order, duplicates preserved
    quarantine = tuple(quarantine_periods)
    roster = dataset.roster(mid, meta) if (spatial and quarantine) else None   # identity is needed only to honour quarantine
    frames = dataset.iter_tracking_frames(mid, roster=roster, metadata=meta, quarantine_periods=quarantine)
    return align_events(events, frames, match_id=int(mid), event_version=version, fps=meta.fps, tolerance_s=tolerance_s, spatial=spatial,
                        spatial_tolerance_m=spatial_tolerance_m)
