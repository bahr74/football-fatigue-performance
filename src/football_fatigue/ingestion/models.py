"""Canonical, immutable data models for Milestone 2A ingestion (provisional contracts in docs/data_contracts.md).

Source values are preserved next to any derived value. Missing data is ``None`` (never 0). Quality flags
describe a record; they never delete or repair it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

CANONICAL_PERIODS = (1, 2, 3, 4)


class IngestionError(Exception):
    """Fatal structural error: a source file or record cannot be interpreted. Never used for data-quality issues."""


class EventVersion(str, Enum):
    """Event Data version. There is deliberately no default: callers must name a version."""

    ROOT = "root"
    MARCH_14_2025 = "march_14_2025"
    MAY_1_2025 = "may_1_2025"

    @property
    def folder(self) -> str | None:
        """Inventory version key: None is the unlabelled Event Data root folder."""
        return {EventVersion.ROOT: None, EventVersion.MARCH_14_2025: "March 14, 2025", EventVersion.MAY_1_2025: "May 1, 2025"}[self]


class QualityFlag(str, Enum):
    """Recoverable data-quality flags. Fatal structural problems raise IngestionError instead."""

    # tracking frame
    REPEATED_RECORDS_PRESENT = "REPEATED_RECORDS_PRESENT"
    REPEATED_RECORDS_DIFFER = "REPEATED_RECORDS_DIFFER"
    NONMONOTONIC_FRAME = "NONMONOTONIC_FRAME"
    FRAME_GAP = "FRAME_GAP"
    INVALID_SOURCE_PERIOD = "INVALID_SOURCE_PERIOD"
    RAW_LIST_MISSING = "RAW_LIST_MISSING"
    DUPLICATE_RAW_JERSEY = "DUPLICATE_RAW_JERSEY"
    SMOOTHED_LIST_MISSING = "SMOOTHED_LIST_MISSING"
    SMOOTHED_JERSEY_WITHOUT_RAW = "SMOOTHED_JERSEY_WITHOUT_RAW"
    SMOOTHED_JERSEY_UNPARSEABLE = "SMOOTHED_JERSEY_UNPARSEABLE"
    SMOOTHED_ENTRY_MALFORMED = "SMOOTHED_ENTRY_MALFORMED"
    BALL_MISSING = "BALL_MISSING"
    MULTIPLE_BALLS = "MULTIPLE_BALLS"
    BALL_SMOOTHED_XY_NULL = "BALL_SMOOTHED_XY_NULL"
    BALL_SMOOTHED_UNRESOLVED = "BALL_SMOOTHED_UNRESOLVED"  # repeated records disagree on ballsSmoothed; canonical smoothed ball is None
    INVALID_COORDINATE = "INVALID_COORDINATE"
    # player sample
    PLAYER_ESTIMATED = "PLAYER_ESTIMATED"
    JERSEY_CONFIDENCE_LOW = "JERSEY_CONFIDENCE_LOW"
    JERSEY_INVALID = "JERSEY_INVALID"
    SMOOTHED_DUPLICATE_IDENTICAL = "SMOOTHED_DUPLICATE_IDENTICAL"
    SMOOTHED_AMBIGUOUS_CONFLICTING = "SMOOTHED_AMBIGUOUS_CONFLICTING"
    SMOOTHED_MISSING = "SMOOTHED_MISSING"
    IDENTITY_UNRESOLVED_JERSEY = "IDENTITY_UNRESOLVED_JERSEY"
    IDENTITY_QUARANTINED = "IDENTITY_QUARANTINED"
    # event row
    EVENT_KEY_PAIR_REPEATED = "EVENT_KEY_PAIR_REPEATED"
    EVENT_TIME_MISSING = "EVENT_TIME_MISSING"
    EVENT_ID_MISSING = "EVENT_ID_MISSING"
    # report level
    NO_TRACKING_FOR_PERIOD = "NO_TRACKING_FOR_PERIOD"


class SmoothedStatus(str, Enum):
    UNIQUE = "unique"
    DUPLICATE_IDENTICAL = "duplicate_identical"
    AMBIGUOUS_CONFLICTING = "ambiguous_conflicting"
    MISSING = "missing"


class AttributionStatus(str, Enum):
    RESOLVED = "resolved"
    QUARANTINED = "quarantined"
    UNRESOLVED_JERSEY = "unresolved_jersey"
    NOT_ATTEMPTED = "not_attempted"  # no roster supplied


@dataclass(frozen=True, slots=True)
class SourceRef:
    """Where a record came from: path relative to the dataset root (POSIX), kind, version label, row/record index."""

    path: str
    kind: str
    version: str | None = None
    index: int | None = None


@dataclass(frozen=True, slots=True)
class MatchMetadata:
    match_id: int
    kickoff_date: str
    competition_id: int
    competition_name: str
    season: str
    matchweek: int | None
    home_team_id: int
    away_team_id: int
    home_team_name: str
    away_team_name: str
    stadium_id: int | None
    stadium_name: str | None
    pitch_length_m: float
    pitch_width_m: float
    fps: float
    home_team_start_left: bool | None
    home_team_start_left_extra_time: bool | None
    # video-time seconds; None when the source is null (all five extra-time matches)
    start_period_1_s: float | None
    end_period_1_s: float | None
    start_period_2_s: float | None
    end_period_2_s: float | None
    source: SourceRef

    def side_of_team(self, team_id: int) -> str | None:
        return "home" if team_id == self.home_team_id else "away" if team_id == self.away_team_id else None

    def team_of_side(self, side: str) -> int:
        return self.home_team_id if side == "home" else self.away_team_id


@dataclass(frozen=True, slots=True)
class RosterEntry:
    match_id: int
    team_id: int
    roster_side: str  # derived from metadata home/away
    player_id: int
    shirt_number: int
    position_group: str | None
    started: bool
    nickname: str | None
    source: SourceRef


@dataclass(frozen=True, slots=True)
class Roster:
    match_id: int
    entries: tuple[RosterEntry, ...]
    issues: tuple[str, ...]  # recoverable roster problems (for example repeated shirt numbers)

    def index(self) -> dict[tuple[str, int], RosterEntry]:
        """(roster_side, shirt_number) -> entry, omitting ambiguous (repeated) shirt numbers."""
        seen: dict[tuple[str, int], list[RosterEntry]] = {}
        for e in self.entries:
            seen.setdefault((e.roster_side, e.shirt_number), []).append(e)
        return {k: v[0] for k, v in seen.items() if len(v) == 1}


@dataclass(frozen=True, slots=True)
class SmoothedCandidate:
    """One source smoothed entry. List order carries no meaning and is never used to select a player."""

    x_m: float | None
    y_m: float | None
    visibility: str | None
    confidence: str | None


@dataclass(frozen=True, slots=True)
class PlayerSample:
    source_side: str  # 'home' or 'away': the source list the entry came from, preserved verbatim
    jersey_number: int | None
    jersey_raw: Any
    x_raw_m: float | None
    y_raw_m: float | None
    visibility: str | None
    confidence: str | None
    speed_raw: float | None
    smoothed_status: SmoothedStatus
    smoothed_candidates: tuple[SmoothedCandidate, ...]
    x_smooth_m: float | None  # populated only for unique or duplicate_identical; None when ambiguous or missing
    y_smooth_m: float | None
    team_id: int | None
    player_id: int | None
    attribution_status: AttributionStatus
    quality_flags: frozenset[QualityFlag]


@dataclass(frozen=True, slots=True)
class BallSample:
    x_raw_m: float | None
    y_raw_m: float | None
    z_raw_m: float | None
    visibility_raw: str | None
    x_smooth_m: float | None
    y_smooth_m: float | None
    z_smooth_m: float | None
    visibility_smooth: str | None
    raw_count: int  # number of entries in the source ``balls`` list


@dataclass(frozen=True, slots=True)
class TrackingFrame:
    match_id: int
    frame_num: int
    video_time_ms_source: float
    video_time_s: float
    source_period: int | None  # verbatim
    canonical_period: int | None  # None unless the source period is 1-4
    period_elapsed_s: float | None
    game_clock_nominal_s: float | None  # display only
    source_record_indices: tuple[int, ...]  # every source record with this frameNum
    source_record_count: int
    players: tuple[PlayerSample, ...]
    unmatched_smoothed: tuple[tuple[str, SmoothedCandidate, Any], ...]  # (side, candidate, original jerseyNum or malformed entry): smoothed entries that cannot be joined to a raw entry
    ball: BallSample | None  # None means a missing observation, never a zero position
    frame_gap_before: int | None  # frames skipped since the previous frame of the same period
    game_event_id: int | None  # cross-reference overlay only
    possession_event_id: int | None
    quality_flags: frozenset[QualityFlag]
    source: SourceRef
    repeat_conflict_fields: tuple[str, ...] = ()  # source fields that differ between repeated records (published values are then unresolved; re-read via source_record_indices)


@dataclass(frozen=True, slots=True)
class EventRow:
    event_row_id: str
    match_id: int
    source_version: EventVersion
    source_row_index: int
    game_event_id: int | None
    possession_event_id: int | None
    game_event_type: str | None
    possession_event_type: str | None
    source_period: int | None
    canonical_period: int | None
    event_time_s: float | None
    start_time_s: float | None
    end_time_s: float | None
    duration_s: float | None
    source_team_id: int | None
    source_player_id: int | None
    home_team: bool | None
    sequence: float | None
    attack_direction: str | None  # root version only; the source string "nan" becomes None
    attributes: dict[str, Any]  # source sections (gameEvents, possessionEvents, initialTouch, fouls, grades), verbatim
    positions_snapshot: dict[str, Any]  # homePlayers/awayPlayers/ball as in the source; NOT a position source
    quality_flags: frozenset[QualityFlag] = field(default_factory=frozenset)
    source: SourceRef | None = None
