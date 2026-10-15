"""Unified match data interface (Milestone 2C): one entry point over ingestion and synchronization.

    from football_fatigue.ingestion import EventVersion
    from football_fatigue.match_data import MatchData, CoordinatePolicy

    md = MatchData.open("data/raw", "3812", EventVersion.ROOT, policy=CoordinatePolicy.RAW)
    md.metadata, md.roster, md.events()              # cheap: no tracking file is read
    for pf in md.frames():                           # lazy, chronological source order, bounded memory
        pf.frame.frame_num, pf.frame.source_record_indices
        for p in pf.players:
            p.selected.x, p.selected.state, p.position_usable, p.identity_usable
    sync = md.synchronize()                          # one more streaming pass; every event preserved
    md.unmatched_events()                            # never hidden
    for pf in md.frames(attach_events=True):         # reads tracking twice (synchronization first), O(E) memory
        pf.events                                    # matched AlignmentResults whose selected frame is this one

Data preparation is deliberately conservative. Nothing is interpolated, extrapolated, smoothed, or invented; no frame is
manufactured; missing stays missing (never (0, 0)); several conflicting candidates are never reduced to one; sides and
identities are never swapped or resolved beyond the roster join. Raw measurements and the source-supplied smoothed
coordinates are separate representations (``raw`` and ``smoothed``); ``selected`` applies one explicit
``CoordinatePolicy`` to them and says which representation it used.

Usability, precisely: ``position_usable`` means the selected coordinate is present, finite, free of an unresolved conflict,
and in a frame whose source period is 1-4. ``identity_usable`` means the team/player attribution was resolved through the
roster join and not quarantined. Neither is a statement about physical accuracy: ``ESTIMATED`` positions (about 63 % of raw
positions in the audit), low jersey confidence, and the tracking system's own error remain, and are listed in ``caveats``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .ingestion import (AttributionStatus, Dataset, EventRow, EventVersion, IngestionError, MatchMetadata, PlayerSample, QualityFlag as Q, Roster,
                        SmoothedStatus, TrackingFrame, validate_match_id)
from .ingestion.validation import TrackingQuality
from .synchronization import AlignmentResult, MatchStatus, MatchSynchronization, synchronize_match


# Coordinate system of every ``x``/``y``/``z`` exposed here (source values, never transformed or normalized).
COORDINATE_SYSTEM = dict(
    name="pitch_center_m", unit="meter", origin="pitch centre", x="along the touchlines, increasing to the right in the main-camera view",
    y="along the end lines, increasing upwards in the main-camera view", z="height above the pitch (ball only)",
    pitch_size="MatchMetadata.pitch_length_m x pitch_width_m (105 x 68 in all 64 matches)",
    attack_direction="not normalized: it differs by period and is not provided here",
)
BALL_FLAGS = frozenset({Q.BALL_MISSING, Q.MULTIPLE_BALLS, Q.BALL_SMOOTHED_XY_NULL, Q.BALL_SMOOTHED_UNRESOLVED, Q.INVALID_COORDINATE})


class CoordState(str, Enum):
    AVAILABLE = "available"      # a finite coordinate exists and is not in conflict
    MISSING = "missing"          # absent, null, or malformed in the source: never replaced by a number
    AMBIGUOUS = "ambiguous"      # several conflicting candidates (or records): none is selected


class CoordinatePolicy(str, Enum):
    RAW = "raw"                              # raw measurement only (default: the evidence layer)
    SMOOTHED = "smoothed"                    # source-supplied smoothed coordinate only; no fallback
    SMOOTHED_THEN_RAW = "smoothed_then_raw"  # smoothed if available, else the raw position explicitly labelled 'raw_fallback'


@dataclass(frozen=True, slots=True)
class Coordinate:
    """One representation of one observation. ``x``/``y``/``z`` are None unless the state is AVAILABLE."""

    state: CoordState
    x: float | None = None
    y: float | None = None
    z: float | None = None
    reason: str | None = None
    candidates: tuple[tuple[float | None, float | None], ...] = ()   # conflicting alternatives when AMBIGUOUS (may be empty: see provenance)


@dataclass(frozen=True, slots=True)
class SelectedCoordinate:
    state: CoordState
    x: float | None
    y: float | None
    z: float | None
    source: str | None        # 'raw', 'smoothed', 'raw_fallback', or None when nothing is selected
    reason: str | None


@dataclass(frozen=True, slots=True)
class PreparedPlayer:
    # quality flags and provenance of the observation: ``sample.quality_flags``, ``sample.smoothed_candidates``; frame provenance on PreparedFrame.frame
    source_side: str          # the source list, preserved; never swapped
    jersey_number: int | None
    team_id: int | None
    player_id: int | None
    attribution_status: AttributionStatus
    identity_usable: bool
    raw: Coordinate
    smoothed: Coordinate
    selected: SelectedCoordinate
    position_usable: bool
    caveats: frozenset[str]
    sample: PlayerSample      # the canonical ingestion sample, with all flags and smoothed candidates


@dataclass(frozen=True, slots=True)
class PreparedBall:
    raw: Coordinate
    smoothed: Coordinate
    selected: SelectedCoordinate
    position_usable: bool
    caveats: frozenset[str]
    quality_flags: frozenset[Q] = frozenset()   # ball-related flags of the frame


@dataclass(frozen=True, slots=True)
class PreparedFrame:
    frame: TrackingFrame                  # canonical frame: timestamps, periods, provenance, quality flags
    players: tuple[PreparedPlayer, ...]
    ball: PreparedBall
    period_usable: bool                   # canonical period 1-4
    policy: CoordinatePolicy
    events: tuple[AlignmentResult, ...] = ()   # matched events aligned to this frame (only with attach_events=True)

    # flat, stable access for consumers (the canonical TrackingFrame stays available as ``frame``)
    @property
    def frame_num(self) -> int:
        return self.frame.frame_num

    @property
    def video_time_s(self) -> float:
        return self.frame.video_time_s

    @property
    def canonical_period(self) -> int | None:
        return self.frame.canonical_period

    @property
    def source_period(self) -> int | None:
        return self.frame.source_period

    @property
    def quality_flags(self) -> frozenset[Q]:
        return self.frame.quality_flags


# ---------------------------------------------------------------- the explicit selection policy

def select(raw: Coordinate, smoothed: Coordinate, policy: CoordinatePolicy) -> SelectedCoordinate:
    """Apply one explicit policy. Conflicting candidates are never reduced to one; there is no interpolation."""
    def pick(c: Coordinate, source: str, reason: str | None = None) -> SelectedCoordinate:
        return SelectedCoordinate(c.state, c.x, c.y, c.z, source if c.state is CoordState.AVAILABLE else None, reason or c.reason)

    if policy is CoordinatePolicy.RAW:
        return pick(raw, "raw")
    if policy is CoordinatePolicy.SMOOTHED:
        return pick(smoothed, "smoothed")
    if smoothed.state is CoordState.AVAILABLE:
        return pick(smoothed, "smoothed")
    if raw.state is CoordState.AVAILABLE:
        return SelectedCoordinate(raw.state, raw.x, raw.y, raw.z, "raw_fallback", f"smoothed_{smoothed.state.value}")
    state = CoordState.AMBIGUOUS if CoordState.AMBIGUOUS in (smoothed.state, raw.state) else CoordState.MISSING
    return SelectedCoordinate(state, None, None, None, None, f"smoothed_{smoothed.state.value}_raw_{raw.state.value}")


# ---------------------------------------------------------------- preparation of one frame (pure, no I/O)

def _raw_players(frame: TrackingFrame) -> dict[tuple[str, int], set[tuple[float, float]]]:
    groups: dict[tuple[str, int], set[tuple[float, float]]] = {}
    for p in frame.players:
        if p.jersey_number is not None and p.x_raw_m is not None and p.y_raw_m is not None:
            groups.setdefault((p.source_side, p.jersey_number), set()).add((p.x_raw_m, p.y_raw_m))
    return groups


def _player_coords(p: PlayerSample, frame: TrackingFrame, raw_groups: dict) -> tuple[Coordinate, Coordinate]:
    side_raw = "homePlayers" if p.source_side == "home" else "awayPlayers"
    conflicts = set(frame.repeat_conflict_fields)
    if side_raw in conflicts:
        raw = Coordinate(CoordState.AMBIGUOUS, reason="repeated_records_disagree")
    elif p.jersey_number is not None and len(raw_groups.get((p.source_side, p.jersey_number), ())) > 1:
        raw = Coordinate(CoordState.AMBIGUOUS, reason="conflicting_duplicate_raw_jerseys", candidates=tuple(sorted(raw_groups[(p.source_side, p.jersey_number)])))
    elif p.x_raw_m is None or p.y_raw_m is None:
        raw = Coordinate(CoordState.MISSING, reason="null_or_invalid_coordinate")
    else:
        raw = Coordinate(CoordState.AVAILABLE, p.x_raw_m, p.y_raw_m)
    if side_raw + "Smoothed" in conflicts:
        sm = Coordinate(CoordState.AMBIGUOUS, reason="repeated_records_disagree")
    elif p.smoothed_status is SmoothedStatus.AMBIGUOUS_CONFLICTING:
        sm = Coordinate(CoordState.AMBIGUOUS, reason="conflicting_smoothed_candidates", candidates=tuple((c.x_m, c.y_m) for c in p.smoothed_candidates))
    elif p.smoothed_status is SmoothedStatus.MISSING:
        sm = Coordinate(CoordState.MISSING, reason="no_smoothed_entry")
    elif p.x_smooth_m is None or p.y_smooth_m is None:
        sm = Coordinate(CoordState.MISSING, reason="null_or_invalid_coordinate")
    else:
        sm = Coordinate(CoordState.AVAILABLE, p.x_smooth_m, p.y_smooth_m)
    return raw, sm


def _caveats(flags: Iterable[Q], source: str | None) -> frozenset[str]:
    out = {f.value for f in flags if f in (Q.PLAYER_ESTIMATED, Q.JERSEY_CONFIDENCE_LOW)}
    if source == "raw_fallback":
        out.add("RAW_FALLBACK_FOR_SMOOTHED")
    return frozenset(out)


def prepare_frame(frame: TrackingFrame, policy: CoordinatePolicy = CoordinatePolicy.RAW, events: tuple[AlignmentResult, ...] = ()) -> PreparedFrame:
    """Wrap a canonical frame with explicit coordinate states, a policy-selected coordinate, and usability indicators."""
    period_ok = frame.canonical_period is not None
    groups = _raw_players(frame)
    players = []
    for p in frame.players:
        raw, sm = _player_coords(p, frame, groups)
        sel = select(raw, sm, policy)
        ident = p.attribution_status is AttributionStatus.RESOLVED
        players.append(PreparedPlayer(p.source_side, p.jersey_number, p.team_id, p.player_id, p.attribution_status, ident, raw, sm, sel,
                                      sel.state is CoordState.AVAILABLE and period_ok, _caveats(p.quality_flags, sel.source), p))
    players_t = tuple(players)
    b = frame.ball
    conflicts = set(frame.repeat_conflict_fields)
    # Repeated-record conflicts are decided first, from the conflict fields alone, so the prepared state does not depend on
    # which conflicting record ingestion kept (including the case where the kept record has no ball at all).
    raw_conflict = "balls" in conflicts
    sm_conflict = "ballsSmoothed" in conflicts or Q.BALL_SMOOTHED_UNRESOLVED in frame.quality_flags   # 2A: unresolved, never a record's value
    if raw_conflict:
        raw_b = Coordinate(CoordState.AMBIGUOUS, reason="repeated_records_disagree")
    elif b is None:
        raw_b = Coordinate(CoordState.MISSING, reason="no_ball_observation")
    elif b.raw_count > 1:
        raw_b = Coordinate(CoordState.AMBIGUOUS, reason="multiple_raw_balls")
    elif b.x_raw_m is None or b.y_raw_m is None:
        raw_b = Coordinate(CoordState.MISSING, reason="no_raw_ball" if b.raw_count == 0 else "null_or_invalid_coordinate")
    else:
        raw_b = Coordinate(CoordState.AVAILABLE, b.x_raw_m, b.y_raw_m, b.z_raw_m)
    if sm_conflict:
        sm_b = Coordinate(CoordState.AMBIGUOUS, reason="repeated_records_disagree")
    elif b is None:
        sm_b = Coordinate(CoordState.MISSING, reason="no_ball_observation")
    elif b.x_smooth_m is None or b.y_smooth_m is None:
        sm_b = Coordinate(CoordState.MISSING, reason="null_smoothed_coordinate")
    else:
        sm_b = Coordinate(CoordState.AVAILABLE, b.x_smooth_m, b.y_smooth_m, b.z_smooth_m)
    sel_b = select(raw_b, sm_b, policy)
    ball = PreparedBall(raw_b, sm_b, sel_b, sel_b.state is CoordState.AVAILABLE and period_ok, _caveats(frame.quality_flags & {Q.BALL_MISSING}, sel_b.source),
                        frame.quality_flags & BALL_FLAGS)
    return PreparedFrame(frame, players_t, ball, period_ok, policy, events)


# ---------------------------------------------------------------- the entry point

class MatchData:
    """One match, one explicit event version, one explicit coordinate policy. Construction is cheap (no tracking read)."""

    def __init__(self, dataset: Dataset, match_id: object, event_version: EventVersion, *, policy: CoordinatePolicy = CoordinatePolicy.RAW,
                 quarantine_periods: Iterable[int] = ()):
        if not isinstance(event_version, EventVersion):
            raise IngestionError("an EventVersion must be named explicitly (root, march_14_2025, may_1_2025)")
        if not isinstance(policy, CoordinatePolicy):
            raise IngestionError("policy must be a CoordinatePolicy")
        self.dataset = dataset
        self.match_id = validate_match_id(match_id)
        self.event_version = event_version
        self.policy = policy
        self.quarantine_periods = tuple(quarantine_periods)
        self._metadata: MatchMetadata | None = None
        self._roster: Roster | None = None
        self._events: list[EventRow] | None = None
        self._sync: dict[bool, MatchSynchronization] = {}

    @classmethod
    def open(cls, dataset_or_root: Dataset | str | Path, match_id: object, event_version: EventVersion, **kw: Any) -> "MatchData":
        ds = dataset_or_root if isinstance(dataset_or_root, Dataset) else Dataset(dataset_or_root)
        return cls(ds, match_id, event_version, **kw)

    def __repr__(self) -> str:
        return f"MatchData(match={self.match_id}, events={self.event_version.value}, policy={self.policy.value})"

    # -- cheap, cached, no tracking I/O
    @property
    def metadata(self) -> MatchMetadata:
        if self._metadata is None:
            self._metadata = self.dataset.metadata(self.match_id)
        return self._metadata

    @property
    def roster(self) -> Roster:
        if self._roster is None:
            self._roster = self.dataset.roster(self.match_id, self.metadata)
        return self._roster

    def events(self) -> list[EventRow]:
        """Rows of the requested version only, in source order, duplicates preserved."""
        if self._events is None:
            self._events = self.dataset.load_events(self.match_id, self.event_version)
        return list(self._events)

    # -- synchronization (one streaming pass, cached)
    def synchronize(self, *, spatial: bool = False) -> MatchSynchronization:
        if spatial not in self._sync:
            self._sync[spatial] = synchronize_match(self.dataset, self.match_id, self.event_version, spatial=spatial, quarantine_periods=self.quarantine_periods)
        return self._sync[spatial]

    def unmatched_events(self) -> tuple[AlignmentResult, ...]:
        """Every event the synchronization could not place, with its reason; never hidden."""
        return tuple(a for a in self.synchronize().alignments if a.status is MatchStatus.UNMATCHED)

    # -- explicit quality scan
    def quality_summary(self, *, max_frames: int | None = None, include_synchronization: bool = False) -> "QualitySummary":
        """Quality summary of this match. NOT run when a match is opened: it streams the tracking file (``max_frames`` bounds it).

        ``include_synchronization=True`` adds the event coverage and unmatched reasons, which costs one more full tracking pass
        unless :meth:`synchronize` was already called. A partial or failed scan never asserts that a period is missing.
        """
        return build_quality_summary(self, max_frames=max_frames, include_synchronization=include_synchronization)

    # -- tracking
    def frames(self, *, attach_events: bool = False, max_frames: int | None = None) -> Iterator[PreparedFrame]:
        """Lazy, chronological (source order) prepared frames; memory stays bounded unless ``attach_events`` is set.

        ``attach_events=True`` first runs (or reuses) the synchronization, which is a full extra streaming pass, and
        buffers the matched alignments grouped by frame (O(E)). Unmatched events are not attached to any frame; use
        :meth:`unmatched_events`.
        """
        by_frame: dict[tuple[int, ...], list[AlignmentResult]] = {}   # key: the selected frame's source record indices (unique per frame occurrence)
        if attach_events:
            for a in self.synchronize().alignments:
                if a.status is MatchStatus.MATCHED:
                    by_frame.setdefault(tuple(a.frame_source_record_indices), []).append(a)
        roster = self.roster
        for fr in self.dataset.iter_tracking_frames(self.match_id, roster=roster, metadata=self.metadata, quarantine_periods=self.quarantine_periods,
                                                    max_frames=max_frames):
            hit = by_frame.pop(tuple(fr.source_record_indices), ())       # popped: an occurrence can be attached to only once
            yield prepare_frame(fr, self.policy, tuple(a for a in hit if a.frame_num == fr.frame_num and a.frame_period == fr.canonical_period))


# ---------------------------------------------------------------- data quality summary (explicit, never automatic)

SCAN_COMPLETE, SCAN_PARTIAL, SCAN_FAILED = "complete", "partial", "failed"


@dataclass(frozen=True, slots=True)
class Metric:
    """A count with its denominator and counting unit stated; ``denominator`` None means there is no meaningful base."""

    count: int
    denominator: int | None
    unit: str            # frames | player_observations | events

    @property
    def fraction(self) -> float | None:
        return None if not self.denominator else self.count / self.denominator


@dataclass(frozen=True, slots=True)
class ScanInfo:
    status: str                      # complete | partial | failed
    frames_scanned: int              # unique frames counted (repeated source records collapse into one frame)
    max_frames: int | None           # the bound that was requested, if any
    fatal_error: str | None
    note: str


@dataclass(frozen=True, slots=True)
class QualitySummary:
    """Quality of one match as seen through MatchData. Counting units are in each metric; see docs/team_handoff.md."""

    match_id: str
    event_version: str
    policy: str
    scan: ScanInfo
    metrics: dict[str, Metric]
    breakdowns: dict[str, dict[str, int]]          # name -> {key: count}; the unit is in ``breakdown_units``
    breakdown_units: dict[str, str]
    tracking_frames_by_canonical_period: dict[str, int]
    event_rows_by_canonical_period: dict[str, int]
    event_periods_without_tracking: tuple[int, ...] | None   # None unless the scan is complete: absence is never asserted otherwise
    quarantined_periods: tuple[int, ...]
    synchronization: dict[str, Any] | None         # MatchSynchronization.summary() when requested
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return dict(match_id=self.match_id, event_version=self.event_version, policy=self.policy,
                    scan=dict(status=self.scan.status, frames_scanned=self.scan.frames_scanned, max_frames=self.scan.max_frames,
                              fatal_error=self.scan.fatal_error, note=self.scan.note),
                    metrics={k: dict(count=m.count, denominator=m.denominator, unit=m.unit) for k, m in self.metrics.items()},
                    breakdowns=self.breakdowns, breakdown_units=self.breakdown_units,
                    tracking_frames_by_canonical_period=self.tracking_frames_by_canonical_period,
                    event_rows_by_canonical_period=self.event_rows_by_canonical_period,
                    event_periods_without_tracking=None if self.event_periods_without_tracking is None else list(self.event_periods_without_tracking),
                    quarantined_periods=list(self.quarantined_periods), synchronization=self.synchronization, warnings=list(self.warnings))


def build_quality_summary(md: "MatchData", *, max_frames: int | None, include_synchronization: bool) -> QualitySummary:
    tq = TrackingQuality()                       # reuses the ingestion accumulator for flag counts and identity status
    c: Counter = Counter()
    sel_src: Counter = Counter()
    per_side: Counter = Counter()
    status, fatal = SCAN_COMPLETE, None
    scanned = 0
    try:
        limit = None if max_frames is None else max_frames + 1     # one extra frame tells whether the stream really ended
        for pf in md.frames(max_frames=limit):
            if max_frames is not None and scanned >= max_frames:
                status = SCAN_PARTIAL
                break
            scanned += 1
            tq.add(pf.frame)
            f = pf.frame
            c["frames"] += 1
            c["period_unusable"] += not pf.period_usable
            c["gap"] += f.frame_gap_before is not None
            c["repeated"] += f.source_record_count > 1
            c["repeat_conflict"] += bool(f.repeat_conflict_fields)
            for side in ("home", "away"):
                per_side[f"{side}:{sum(1 for p in pf.players if p.source_side == side)}"] += 1
            for p in pf.players:
                c["obs"] += 1
                c["raw_" + p.raw.state.value] += 1
                c["sm_" + p.smoothed.state.value] += 1
                c["sel_unusable"] += not p.position_usable
                sel_src[p.selected.source or "none"] += 1
                v = p.sample.visibility
                c["vis_" + (v.lower() if v in ("VISIBLE", "ESTIMATED") else "unknown")] += 1
                c["low_conf"] += Q.JERSEY_CONFIDENCE_LOW in p.sample.quality_flags
            b = pf.ball
            c["ball_raw_" + b.raw.state.value] += 1
            c["ball_sm_" + b.smoothed.state.value] += 1
            c["ball_sel_unusable"] += not b.position_usable
    except IngestionError as e:
        status, fatal = SCAN_FAILED, str(e)
    n, o = c["frames"], c["obs"]
    note = {SCAN_COMPLETE: "the whole tracking file was read successfully",
            SCAN_PARTIAL: f"bounded scan of the first {max_frames} frames; whole-match totals and period absence are not asserted",
            SCAN_FAILED: "the tracking stream failed; counts cover only the frames read before the failure and period absence is not asserted"}[status]
    M = Metric
    metrics: dict[str, Metric] = {
        "frames_period_unusable": M(c["period_unusable"], n, "frames"), "frames_with_frame_gap": M(c["gap"], n, "frames"),
        "frames_with_repeated_source_records": M(c["repeated"], n, "frames"), "frames_with_repeat_conflicts": M(c["repeat_conflict"], n, "frames"),
        "ball_raw_available": M(c["ball_raw_available"], n, "frames"), "ball_raw_missing": M(c["ball_raw_missing"], n, "frames"),
        "ball_raw_ambiguous": M(c["ball_raw_ambiguous"], n, "frames"), "ball_smoothed_available": M(c["ball_sm_available"], n, "frames"),
        "ball_smoothed_missing": M(c["ball_sm_missing"], n, "frames"), "ball_smoothed_ambiguous": M(c["ball_sm_ambiguous"], n, "frames"),
        "ball_selected_unusable": M(c["ball_sel_unusable"], n, "frames"),
        "player_raw_available": M(c["raw_available"], o, "player_observations"), "player_raw_missing": M(c["raw_missing"], o, "player_observations"),
        "player_raw_ambiguous": M(c["raw_ambiguous"], o, "player_observations"), "player_smoothed_available": M(c["sm_available"], o, "player_observations"),
        "player_smoothed_missing": M(c["sm_missing"], o, "player_observations"), "player_smoothed_ambiguous": M(c["sm_ambiguous"], o, "player_observations"),
        "player_selected_unusable": M(c["sel_unusable"], o, "player_observations"),
        "player_estimated": M(c["vis_estimated"], o, "player_observations"), "player_measured_visible": M(c["vis_visible"], o, "player_observations"),
        "player_visibility_unknown": M(c["vis_unknown"], o, "player_observations"), "player_jersey_confidence_low": M(c["low_conf"], o, "player_observations"),
    }
    for k, v in tq.attribution_status.items():
        metrics["identity_" + k] = M(v, o, "player_observations")
    breakdowns = {"selected_coordinate_source": dict(sel_src), "players_per_side_per_frame": dict(per_side),
                  "frame_quality_flags": dict(tq.frame_flags), "player_quality_flags": dict(tq.sample_flags)}
    units = {"selected_coordinate_source": "player_observations", "players_per_side_per_frame": "frame_sides (frames x 2 sides)",
             "frame_quality_flags": "frames", "player_quality_flags": "player_observations"}
    # events (cheap: events are loaded, tracking is not touched here)
    ev_by_period: Counter = Counter()
    events_error = None
    try:
        rows = md.events()
        ev_flags: Counter = Counter(fl.value for r in rows for fl in r.quality_flags)
        ev_by_period.update(str(r.canonical_period) for r in rows if r.canonical_period is not None)
        metrics["events_total"] = M(len(rows), None, "events")
        metrics["events_invalid_source_period"] = M(ev_flags.get("INVALID_SOURCE_PERIOD", 0), len(rows), "events")
        metrics["events_missing_time"] = M(ev_flags.get("EVENT_TIME_MISSING", 0), len(rows), "events")
        metrics["events_in_repeated_key_pairs"] = M(ev_flags.get("EVENT_KEY_PAIR_REPEATED", 0), len(rows), "events")
        breakdowns["event_quality_flags"], units["event_quality_flags"] = dict(ev_flags), "events"
    except IngestionError as e:
        rows, events_error = [], str(e)
    sync = None
    if include_synchronization:
        try:
            s = md.synchronize()
            sync = s.summary()
            metrics["events_temporally_matched"] = M(sync["matched"], sync["events"], "events")
            metrics["events_unmatched"] = M(sync["unmatched"], sync["events"], "events")
            breakdowns["unmatched_reasons"], units["unmatched_reasons"] = dict(sync["unmatched_reasons"]), "events"
        except IngestionError as e:
            sync = dict(error=str(e))
    missing = tuple(sorted(int(k) for k in ev_by_period if k not in tq.frames_by_canonical_period)) if status == SCAN_COMPLETE else None
    warn: list[str] = []
    if status != SCAN_COMPLETE:
        warn.append(f"{status} tracking scan: {note}")
    if events_error:
        warn.append(f"events could not be loaded: {events_error}")
    if missing:
        warn.append(f"events exist for periods {list(missing)} that have no tracking frames (event-only periods)")
    if md.quarantine_periods:
        warn.append(f"team/player attribution is withheld (quarantined) for periods {list(md.quarantine_periods)}")
    if c["vis_estimated"]:
        warn.append(f"{c['vis_estimated']} of {o} player observations are ESTIMATED by the tracking source, not measured in the video")
    if c["raw_ambiguous"] or c["sm_ambiguous"] or c["ball_raw_ambiguous"] or c["ball_sm_ambiguous"]:
        warn.append("ambiguous observations exist: candidates are kept and no coordinate is selected for them")
    if sync and sync.get("scan_complete") is False:
        warn.append("the synchronization scan did not complete: unmatched events may be tracking_scan_incomplete")
    return QualitySummary(md.match_id, md.event_version.value, md.policy.value, ScanInfo(status, scanned, max_frames, fatal, note), metrics, breakdowns, units,
                          dict(tq.frames_by_canonical_period), dict(ev_by_period), missing, md.quarantine_periods, sync, tuple(warn))
