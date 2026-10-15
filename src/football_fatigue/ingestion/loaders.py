"""Read-only loaders: metadata, rosters, Event Data (explicit version), and streaming tracking frames.

Fatal structural problems raise IngestionError; recoverable data-quality problems become QualityFlags on the
returned records. Nothing here writes to disk, selects an event version implicitly, resolves ambiguous smoothed
coordinates, infers team identity from a jersey number alone, or turns missing observations into zeros.
"""

from __future__ import annotations

import bz2
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

from football_fatigue import inventory as inv

from .models import (
    CANONICAL_PERIODS,
    AttributionStatus,
    BallSample,
    EventRow,
    EventVersion,
    IngestionError,
    MatchMetadata,
    PlayerSample,
    QualityFlag as Q,
    Roster,
    RosterEntry,
    SmoothedCandidate,
    SmoothedStatus,
    SourceRef,
    TrackingFrame,
)

_MATCH_ID = re.compile(r"[0-9]{1,9}")


def validate_match_id(match_id: object) -> str:
    """Canonical match id: 1-9 ASCII digits (the inventory's file-name format). Rejects separators and traversal."""
    if isinstance(match_id, int) and not isinstance(match_id, bool):
        match_id = str(match_id)
    if not isinstance(match_id, str) or not _MATCH_ID.fullmatch(match_id):
        raise IngestionError(f"invalid match id {match_id!r}: expected 1-9 ASCII digits")
    return match_id


# ---------------------------------------------------------------- scalar coercion (never fabricates values)

def as_float(v: Any) -> float | None:
    """Finite number or None. Booleans, strings, NaN and infinities are not numbers here."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def as_int(v: Any) -> int | None:
    """Integer, integral float, or ASCII-digit string; otherwise None. Booleans are rejected."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v) if math.isfinite(v) and v == int(v) else None
    if isinstance(v, str) and re.fullmatch(r"-?[0-9]+", v.strip()):
        return int(v.strip())
    return None


def _req(d: Any, path: str, what: str) -> Any:
    cur = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur or cur[key] is None:
            raise IngestionError(f"{what}: required field '{path}' is missing or null")
        cur = cur[key]
    return cur


def _read_json(path: Path) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        raise IngestionError(f"{path.name}: cannot read as JSON ({type(e).__name__}: {e})") from e


def check_identity(match_id: int, meta: "MatchMetadata | None", roster: "Roster | None") -> None:
    """Reject metadata/rosters of another match and rosters whose teams disagree with the metadata home/away teams."""
    if meta is not None and meta.match_id != match_id:
        raise IngestionError(f"metadata belongs to match {meta.match_id}, not the requested match {match_id}")
    if roster is None:
        return
    if meta is None:
        raise IngestionError("a roster join needs the match metadata (home/away team ids)")
    if roster.match_id != match_id:
        raise IngestionError(f"roster belongs to match {roster.match_id}, not the requested match {match_id}")
    for e in roster.entries:
        if e.match_id != match_id or meta.side_of_team(e.team_id) != e.roster_side:
            raise IngestionError(f"roster entry (team {e.team_id}, side {e.roster_side}, player {e.player_id}) is inconsistent with the "
                                 f"home/away teams ({meta.home_team_id}/{meta.away_team_id}) of match {match_id}")


# ---------------------------------------------------------------- dataset access

class Dataset:
    """Match-ID based access to a raw dataset root (read-only; files are located through the project inventory)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not self.root.is_dir():
            raise IngestionError(f"dataset root is not a directory: {self.root}")
        self._inventory: inv.Inventory | None = None

    # -- location
    def _inv(self) -> inv.Inventory:
        if self._inventory is None:
            try:
                self._inventory = inv.scan(self.root)
            except inv.InventoryScanError as e:
                raise IngestionError(f"incomplete dataset scan: {'; '.join(e.errors)}") from e
        return self._inventory

    def match_ids(self) -> list[str]:
        return self._inv().match_ids

    def _one(self, match_id: str, paths: list[str], what: str) -> tuple[Path, str]:
        if len(paths) != 1:
            raise IngestionError(f"match {match_id}: {len(paths)} {what} file(s) found, expected exactly 1: {paths}")
        return self.root / paths[0], paths[0]

    def _match(self, match_id: object) -> tuple[str, inv.MatchInventory]:
        mid = validate_match_id(match_id)
        mi = self._inv().matches.get(mid)
        if mi is None:
            raise IngestionError(f"match {mid}: not found under {self.root}")
        return mid, mi

    # -- metadata / roster
    def metadata(self, match_id: object) -> MatchMetadata:
        mid, mi = self._match(match_id)
        path, rel = self._one(mid, mi.metadata, "metadata")
        return parse_metadata(_read_json(path), mid, SourceRef(rel, "metadata"))

    def roster(self, match_id: object, metadata: MatchMetadata | None = None) -> Roster:
        mid, mi = self._match(match_id)
        meta = metadata or self.metadata(mid)
        check_identity(int(mid), meta, None)
        path, rel = self._one(mid, mi.rosters, "roster")
        return parse_roster(_read_json(path), meta, SourceRef(rel, "roster"))

    # -- events
    def load_events(self, match_id: object, version: EventVersion) -> list[EventRow]:
        return list(self.iter_events(match_id, version))

    def iter_events(self, match_id: object, version: EventVersion) -> Iterator[EventRow]:
        """Rows of exactly the requested version, in source order, none dropped or merged."""
        if not isinstance(version, EventVersion):
            raise IngestionError("an EventVersion must be named explicitly (root, march_14_2025, may_1_2025)")
        mid, mi = self._match(match_id)
        paths = mi.events.get(version.folder, [])
        path, rel = self._one(mid, paths, f"event[{version.value}]")
        yield from parse_events(_read_json(path), mid, version, rel)

    # -- tracking
    def tracking_source(self, match_id: object) -> tuple[Path, SourceRef]:
        mid, mi = self._match(match_id)
        path, rel = self._one(mid, mi.tracking, "tracking")
        return path, SourceRef(rel, "tracking")

    def iter_tracking_frames(
        self,
        match_id: object,
        *,
        roster: Roster | None = None,
        metadata: MatchMetadata | None = None,
        collapse_repeats: bool = True,
        quarantine_periods: Iterable[int] = (),
        max_frames: int | None = None,
    ) -> Iterator[TrackingFrame]:
        """Stream tracking frames with bounded memory (one frame plus its repeats' comparison state).

        Records with the same consecutive ``frameNum`` collapse into one frame that keeps every source record
        index (``collapse_repeats=False`` yields one frame per record instead). ``quarantine_periods`` names
        periods whose team/player attribution is to be withheld (for example periods flagged by an external
        side-identity review); nothing is swapped. A roster enables the (side, shirt number) join; the jersey
        number alone is never used.
        """
        mid, _ = self._match(match_id)
        path, src = self.tracking_source(mid)
        meta = metadata
        if roster is not None and meta is None:
            meta = self.metadata(mid)
        check_identity(int(mid), meta, roster)   # eager: raised here, before any frame is read
        return iter_tracking_frames(path, src, int(mid), roster=roster, metadata=meta, collapse_repeats=collapse_repeats,
                                    quarantine_periods=frozenset(quarantine_periods), max_frames=max_frames)


# ---------------------------------------------------------------- metadata and roster

def parse_metadata(doc: Any, match_id: str, source: SourceRef) -> MatchMetadata:
    try:
        return _parse_metadata(doc, match_id, source)
    except (AttributeError, TypeError, KeyError, ValueError) as e:
        raise IngestionError(f"metadata {match_id}: malformed structure ({type(e).__name__}: {e})") from e


def _parse_metadata(doc: Any, match_id: str, source: SourceRef) -> MatchMetadata:
    m = doc[0] if isinstance(doc, list) and len(doc) == 1 else doc
    if not isinstance(m, dict):
        raise IngestionError(f"metadata {match_id}: expected one JSON object (or a one-element list)")
    what = f"metadata {match_id}"
    mid = as_int(_req(m, "id", what))
    if mid != int(match_id):
        raise IngestionError(f"{what}: id {m['id']!r} does not match the requested match id")
    fps = as_float(_req(m, "fps", what))
    if fps is None or fps <= 0:
        raise IngestionError(f"{what}: fps must be a positive finite number")
    home, away = as_int(_req(m, "homeTeam.id", what)), as_int(_req(m, "awayTeam.id", what))
    if home is None or away is None or home == away:
        raise IngestionError(f"{what}: home/away team ids must be two distinct integers")
    pitches = _req(m, "stadium.pitches", what)
    if not isinstance(pitches, list) or len(pitches) != 1:
        raise IngestionError(f"{what}: expected exactly one pitch entry, got {len(pitches) if isinstance(pitches, list) else pitches!r}")
    if not isinstance(pitches[0], dict):
        raise IngestionError(f"{what}: stadium.pitches[0] must be an object, got {type(pitches[0]).__name__}")
    length, width = as_float(pitches[0].get("length")), as_float(pitches[0].get("width"))
    if length is None or width is None or length <= 0 or width <= 0:
        raise IngestionError(f"{what}: pitch length/width must be positive finite numbers")
    comp = as_int(_req(m, "competition.id", what))
    if comp is None:
        raise IngestionError(f"{what}: competition id is not an integer")
    start_left = m.get("homeTeamStartLeft")
    et = m.get("homeTeamStartLeftExtraTime")
    return MatchMetadata(
        match_id=mid,
        kickoff_date=str(_req(m, "date", what)),
        competition_id=comp,
        competition_name=str(_req(m, "competition.name", what)),
        season=str(_req(m, "season", what)),
        matchweek=as_int(m.get("week")),
        home_team_id=home,
        away_team_id=away,
        home_team_name=str((m["homeTeam"] or {}).get("name")),
        away_team_name=str((m["awayTeam"] or {}).get("name")),
        stadium_id=as_int(m["stadium"].get("id")),
        stadium_name=m["stadium"].get("name"),
        pitch_length_m=length,
        pitch_width_m=width,
        fps=fps,
        home_team_start_left=start_left if isinstance(start_left, bool) else None,
        home_team_start_left_extra_time=et if isinstance(et, bool) else None,
        start_period_1_s=as_float(m.get("startPeriod1")),
        end_period_1_s=as_float(m.get("endPeriod1")),
        start_period_2_s=as_float(m.get("startPeriod2")),
        end_period_2_s=as_float(m.get("endPeriod2")),
        source=source,
    )


def parse_roster(doc: Any, meta: MatchMetadata, source: SourceRef) -> Roster:
    try:
        return _parse_roster(doc, meta, source)
    except (AttributeError, TypeError, KeyError, ValueError) as e:
        raise IngestionError(f"roster {meta.match_id}: malformed structure ({type(e).__name__}: {e})") from e


def _parse_roster(doc: Any, meta: MatchMetadata, source: SourceRef) -> Roster:
    if not isinstance(doc, list):
        raise IngestionError(f"roster {meta.match_id}: expected a JSON list")
    what = f"roster {meta.match_id}"
    entries: list[RosterEntry] = []
    for i, e in enumerate(doc):
        if not isinstance(e, dict):
            raise IngestionError(f"{what}: entry {i} is not an object")
        team = as_int(_req(e, "team.id", f"{what} entry {i}"))
        pid = as_int(_req(e, "player.id", f"{what} entry {i}"))
        shirt = as_int(_req(e, "shirtNumber", f"{what} entry {i}"))
        side = meta.side_of_team(team) if team is not None else None
        if side is None or pid is None or shirt is None:
            raise IngestionError(f"{what}: entry {i} has a team that is not home/away of the match, or a non-integer id/shirt number")
        entries.append(RosterEntry(meta.match_id, team, side, pid, shirt, e.get("positionGroupType"), bool(e.get("started")),
                                   (e.get("player") or {}).get("nickname"), SourceRef(source.path, source.kind, None, i)))
    issues: list[str] = []
    counts = Counter((e.roster_side, e.shirt_number) for e in entries)
    issues += [f"repeated shirt number {k[1]} for {k[0]} ({n} entries): excluded from the identity join" for k, n in sorted(counts.items()) if n > 1]
    pcount = Counter(e.player_id for e in entries)
    issues += [f"player id {k} appears {n} times" for k, n in sorted(pcount.items()) if n > 1]
    return Roster(meta.match_id, tuple(entries), tuple(issues))


# ---------------------------------------------------------------- events

def _canonical(period: int | None) -> int | None:
    return period if period in CANONICAL_PERIODS else None


def parse_events(doc: Any, match_id: str, version: EventVersion, rel_path: str) -> Iterator[EventRow]:
    if not isinstance(doc, list):
        raise IngestionError(f"events {match_id} [{version.value}]: expected a JSON list of rows")
    what = f"events {match_id} [{version.value}]"
    pair_counts = Counter()
    for i, r in enumerate(doc):
        if isinstance(r, dict):
            for k in ("gameEventId", "possessionEventId"):
                if not isinstance(r.get(k), (type(None), int, float, str)):
                    raise IngestionError(f"{what}: row {i}: {k} must be a scalar, got {type(r.get(k)).__name__}")
            pair_counts[(r.get("gameEventId"), r.get("possessionEventId"))] += 1
    for i, r in enumerate(doc):
        if not isinstance(r, dict):
            raise IngestionError(f"{what}: row {i} is not an object")
        if r.get("gameId") is not None and as_int(r.get("gameId")) != int(match_id):
            raise IngestionError(f"{what}: row {i}: gameId {r.get('gameId')!r} does not belong to match {match_id}")
        try:
            yield _event_row(r, i, match_id, version, rel_path, pair_counts)
        except (AttributeError, TypeError, KeyError, ValueError) as e:
            raise IngestionError(f"{what}: row {i}: malformed structure ({type(e).__name__}: {e})") from e


def _event_row(r: dict, i: int, match_id: str, version: EventVersion, rel_path: str, pair_counts: Counter) -> EventRow:
    ge = r.get("gameEvents") if isinstance(r.get("gameEvents"), dict) else {}
    pe = r.get("possessionEvents") if isinstance(r.get("possessionEvents"), dict) else {}
    flags: set[Q] = set()
    src_period = as_int(ge.get("period"))
    if _canonical(src_period) is None:
        flags.add(Q.INVALID_SOURCE_PERIOD)
    t = as_float(r.get("eventTime"))
    if t is None:
        flags.add(Q.EVENT_TIME_MISSING)
    gid = as_int(r.get("gameEventId"))
    if gid is None:
        flags.add(Q.EVENT_ID_MISSING)
    if pair_counts[(r.get("gameEventId"), r.get("possessionEventId"))] > 1:
        flags.add(Q.EVENT_KEY_PAIR_REPEATED)
    direction = (r.get("stadiumMetadata") or {}).get("teamAttackingDirection") if isinstance(r.get("stadiumMetadata"), dict) else None
    ht = ge.get("homeTeam")
    attrs = {k: r[k] for k in ("gameEvents", "initialTouch", "possessionEvents", "fouls", "grades", "stadiumMetadata") if k in r}
    return EventRow(
        event_row_id=f"{match_id}:{version.value}:{i}", match_id=int(match_id), source_version=version, source_row_index=i,
        game_event_id=gid, possession_event_id=as_int(r.get("possessionEventId")),
        game_event_type=ge.get("gameEventType"), possession_event_type=pe.get("possessionEventType"),
        source_period=src_period, canonical_period=_canonical(src_period),
        event_time_s=t, start_time_s=as_float(r.get("startTime")), end_time_s=as_float(r.get("endTime")), duration_s=as_float(r.get("duration")),
        source_team_id=as_int(ge.get("teamId")), source_player_id=as_int(ge.get("playerId")),
        home_team=ht if isinstance(ht, bool) else None, sequence=as_float(r.get("sequence")),
        attack_direction=direction if direction in ("L", "R") else None,
        attributes=attrs, positions_snapshot={k: r.get(k) for k in ("homePlayers", "awayPlayers", "ball")},
        quality_flags=frozenset(flags), source=SourceRef(rel_path, "events", version.value, i),
    )


# ---------------------------------------------------------------- tracking

def _iter_records(path: Path) -> Iterator[tuple[int, dict]]:
    """Stream (record_index, object) from a .jsonl.bz2 file; the index counts non-blank records from 0."""
    idx = 0
    try:
        with bz2.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except ValueError as e:
                    raise IngestionError(f"{path.name}: record {idx} is not valid JSON ({e})") from e
                if not isinstance(rec, dict):
                    raise IngestionError(f"{path.name}: record {idx} is not a JSON object")
                yield idx, rec
                idx += 1
    except (OSError, EOFError) as e:
        raise IngestionError(f"{path.name}: cannot read compressed stream ({type(e).__name__}: {e})") from e


def _jersey(v: Any) -> int | None:
    return as_int(v)


def _candidate(e: dict) -> SmoothedCandidate:
    return SmoothedCandidate(as_float(e.get("x")), as_float(e.get("y")), e.get("visibility"), e.get("confidence"))


def _smoothed_status(cands: list[SmoothedCandidate]) -> SmoothedStatus:
    if not cands:
        return SmoothedStatus.MISSING
    if len(cands) == 1:
        return SmoothedStatus.UNIQUE
    first = (cands[0].x_m, cands[0].y_m)
    same = first[0] is not None and first[1] is not None and all((c.x_m, c.y_m) == first for c in cands)
    return SmoothedStatus.DUPLICATE_IDENTICAL if same else SmoothedStatus.AMBIGUOUS_CONFLICTING


def _players(rec: dict, ctx: dict) -> tuple[list[PlayerSample], list[tuple[str, SmoothedCandidate, Any]], set[Q]]:
    samples: list[PlayerSample] = []
    unmatched: list[tuple[str, SmoothedCandidate, Any]] = []
    fflags: set[Q] = set()
    period = ctx["period"]
    for side, rk, sk in (("home", "homePlayers", "homePlayersSmoothed"), ("away", "awayPlayers", "awayPlayersSmoothed")):
        raw, sm = rec.get(rk), rec.get(sk)
        if not isinstance(raw, list):
            fflags.add(Q.RAW_LIST_MISSING)
            raw = []
        groups: dict[Any, list[tuple[SmoothedCandidate, Any]]] = {}
        if isinstance(sm, list):
            for e in sm:
                if isinstance(e, dict):
                    groups.setdefault(_jersey(e.get("jerseyNum")), []).append((_candidate(e), e.get("jerseyNum")))
                else:  # keep even a malformed smoothed entry as an unjoinable observation
                    fflags.add(Q.SMOOTHED_ENTRY_MALFORMED)
                    unmatched.append((side, SmoothedCandidate(None, None, None, None), e))
        else:
            fflags.add(Q.SMOOTHED_LIST_MISSING)
        raw_jerseys = Counter(_jersey(e.get("jerseyNum")) for e in raw if isinstance(e, dict))
        if any(n > 1 and j is not None for j, n in raw_jerseys.items()):
            fflags.add(Q.DUPLICATE_RAW_JERSEY)
        for e in raw:
            if not isinstance(e, dict):
                continue
            flags: set[Q] = set()
            jr = e.get("jerseyNum")
            j = _jersey(jr)
            if j is None:
                flags.add(Q.JERSEY_INVALID)
            x, y = as_float(e.get("x")), as_float(e.get("y"))
            if x is None or y is None:
                flags.add(Q.INVALID_COORDINATE)
            if e.get("visibility") == "ESTIMATED":
                flags.add(Q.PLAYER_ESTIMATED)
            if e.get("confidence") == "LOW":
                flags.add(Q.JERSEY_CONFIDENCE_LOW)
            cands = [c for c, _ in groups.get(j, [])] if j is not None else []
            status = _smoothed_status(cands) if isinstance(sm, list) else SmoothedStatus.MISSING
            xs = ys = None
            if status in (SmoothedStatus.UNIQUE, SmoothedStatus.DUPLICATE_IDENTICAL):
                xs, ys = cands[0].x_m, cands[0].y_m
                if xs is None or ys is None:
                    flags.add(Q.INVALID_COORDINATE)
            if status is SmoothedStatus.AMBIGUOUS_CONFLICTING:
                flags.add(Q.SMOOTHED_AMBIGUOUS_CONFLICTING)
            elif status is SmoothedStatus.DUPLICATE_IDENTICAL:
                flags.add(Q.SMOOTHED_DUPLICATE_IDENTICAL)
            elif status is SmoothedStatus.MISSING:
                flags.add(Q.SMOOTHED_MISSING)
            team = player = None
            attr = AttributionStatus.NOT_ATTEMPTED
            roster_index = ctx["roster_index"]
            if roster_index is not None:
                if period in ctx["quarantine"]:
                    attr = AttributionStatus.QUARANTINED
                    flags.add(Q.IDENTITY_QUARANTINED)
                else:
                    entry = roster_index.get((side, j)) if j is not None else None
                    if entry is None:
                        attr = AttributionStatus.UNRESOLVED_JERSEY
                        flags.add(Q.IDENTITY_UNRESOLVED_JERSEY)
                    else:
                        attr, team, player = AttributionStatus.RESOLVED, entry.team_id, entry.player_id
            samples.append(PlayerSample(side, j, jr, x, y, e.get("visibility"), e.get("confidence"), as_float(e.get("speed")),
                                        status, tuple(cands), xs, ys, team, player, attr, frozenset(flags)))
        raw_keys = set(raw_jerseys)
        for j, cs in groups.items():
            if j is None:   # unparseable jersey: never joinable, never dropped
                fflags.add(Q.SMOOTHED_JERSEY_UNPARSEABLE)
                unmatched.extend((side, c, jr_raw) for c, jr_raw in cs)
            elif j not in raw_keys:
                fflags.add(Q.SMOOTHED_JERSEY_WITHOUT_RAW)
                unmatched.extend((side, c, jr_raw) for c, jr_raw in cs)
    return samples, unmatched, fflags


def _ball(rec: dict, flags: set[Q]) -> BallSample | None:
    balls = rec.get("balls")
    raw = [b for b in balls if isinstance(b, dict)] if isinstance(balls, list) else []
    sm = rec.get("ballsSmoothed") if isinstance(rec.get("ballsSmoothed"), dict) else None
    if not raw and sm is None:
        flags.add(Q.BALL_MISSING)
        return None
    if not raw:
        flags.add(Q.BALL_MISSING)
    if len(raw) > 1:
        flags.add(Q.MULTIPLE_BALLS)
    b = raw[0] if raw else {}
    s = sm or {}
    sx, sy = as_float(s.get("x")), as_float(s.get("y"))
    if sm is not None and (sx is None or sy is None):
        flags.add(Q.BALL_SMOOTHED_XY_NULL)
    rx, ry, rz = as_float(b.get("x")), as_float(b.get("y")), as_float(b.get("z"))
    if raw and (rx is None or ry is None):
        flags.add(Q.INVALID_COORDINATE)
    return BallSample(rx, ry, rz, b.get("visibility") if raw else None, sx, sy, as_float(s.get("z")),
                      s.get("visibility") if sm is not None else None, len(raw))


# Repeated records of one frameNum must agree on time and period (otherwise the frame is structurally ambiguous: fatal);
# differences in the other canonical fields are kept visible as unresolved (flag + field names), never published silently.
_REPEAT_FATAL = ("videoTimeMs", "period")
_REPEAT_COMPARE = ("periodElapsedTime", "periodGameClockTime", "homePlayers", "awayPlayers", "homePlayersSmoothed",
                   "awayPlayersSmoothed", "balls", "ballsSmoothed")


def _build_frame(match_id: int, src: SourceRef, rec: dict, indices: list[int], conflicts: set[str], gap: int | None, nonmono: bool,
                 ctx_base: dict) -> TrackingFrame:
    fn = as_int(rec.get("frameNum"))
    vt_ms = as_float(rec.get("videoTimeMs"))
    src_period = as_int(rec.get("period"))
    flags: set[Q] = set()
    if len(indices) > 1:
        flags.add(Q.REPEATED_RECORDS_PRESENT)
    if conflicts:
        flags.add(Q.REPEATED_RECORDS_DIFFER)
    if nonmono:
        flags.add(Q.NONMONOTONIC_FRAME)
    if gap:
        flags.add(Q.FRAME_GAP)
    if _canonical(src_period) is None:
        flags.add(Q.INVALID_SOURCE_PERIOD)
    ctx = dict(ctx_base, period=src_period)
    samples, unmatched, pflags = _players(rec, ctx)
    flags |= pflags
    ball = _ball(rec, flags)
    if "ballsSmoothed" in conflicts:
        # Policy: repeated records disagree on the smoothed ball, so no record's value is canonical. The smoothed
        # fields stay None (raw ball untouched); the alternatives are read from the source via source_record_indices.
        flags.discard(Q.BALL_SMOOTHED_XY_NULL)
        flags.add(Q.BALL_SMOOTHED_UNRESOLVED)
        if ball is not None:
            ball = replace(ball, x_smooth_m=None, y_smooth_m=None, z_smooth_m=None, visibility_smooth=None)
    return TrackingFrame(
        match_id=match_id, frame_num=fn, video_time_ms_source=vt_ms, video_time_s=vt_ms / 1000.0,
        source_period=src_period, canonical_period=_canonical(src_period),
        period_elapsed_s=as_float(rec.get("periodElapsedTime")), game_clock_nominal_s=as_float(rec.get("periodGameClockTime")),
        source_record_indices=tuple(indices), source_record_count=len(indices), players=tuple(samples),
        unmatched_smoothed=tuple(unmatched), ball=ball, frame_gap_before=gap,
        game_event_id=as_int(rec.get("game_event_id")), possession_event_id=as_int(rec.get("possession_event_id")),
        quality_flags=frozenset(flags), source=SourceRef(src.path, src.kind, None, indices[0]),
        repeat_conflict_fields=tuple(sorted(conflicts)),
    )


def iter_tracking_frames(path: Path, src: SourceRef, match_id: int, *, roster: Roster | None, metadata: MatchMetadata | None,
                         collapse_repeats: bool, quarantine_periods: frozenset[int], max_frames: int | None = None) -> Iterator[TrackingFrame]:
    ctx = dict(roster_index=roster.index() if roster is not None else None, quarantine=quarantine_periods)
    check_identity(match_id, metadata, roster)
    cur: dict | None = None  # {'rec','indices','conflicts','gap','nonmono'}
    prev_fn: int | None = None
    prev_period: Any = None
    emitted = 0

    def check(idx: int, rec: dict) -> None:
        if as_int(rec.get("frameNum")) is None:
            raise IngestionError(f"{path.name}: record {idx}: required field 'frameNum' is missing or not an integer")
        if as_float(rec.get("videoTimeMs")) is None:
            raise IngestionError(f"{path.name}: record {idx}: required field 'videoTimeMs' is missing or not a finite number")
        ref = rec.get("gameRefId")
        if ref is not None and as_int(ref) != match_id:
            raise IngestionError(f"{path.name}: record {idx}: gameRefId {ref!r} does not belong to match {match_id}")

    def build(c: dict) -> TrackingFrame:
        try:
            return _build_frame(match_id, src, c["rec"], c["indices"], c["conflicts"], c["gap"], c["nonmono"], ctx)
        except (AttributeError, TypeError, KeyError, ValueError) as e:
            raise IngestionError(f"{path.name}: frame {c['fn']} (records {c['indices'][0]}..{c['indices'][-1]}): malformed structure ({type(e).__name__}: {e})") from e

    pending: IngestionError | None = None  # a structural error is raised only after the last good frame was yielded
    records = _iter_records(path)
    while True:
        try:
            idx, rec = next(records)
            check(idx, rec)
        except StopIteration:
            break
        except IngestionError as e:
            pending = e
            break
        fn = as_int(rec["frameNum"])
        if collapse_repeats and cur is not None and fn == cur["fn"]:
            cur["indices"].append(idx)
            first = cur["rec"]
            bad = [k for k in _REPEAT_FATAL if rec.get(k) != first.get(k)]
            if bad:
                pending = IngestionError(f"{path.name}: records {cur['indices'][0]} and {idx} share frameNum {fn} but differ in "
                                         f"{', '.join(bad)}: incompatible repeated records")
                break
            cur["conflicts"].update(k for k in _REPEAT_COMPARE if rec.get(k) != first.get(k))
            continue
        if cur is not None:
            yield build(cur)
            emitted += 1
            if max_frames is not None and emitted >= max_frames:
                return
        period = as_int(rec.get("period"))
        gap = None
        nonmono = False
        if prev_fn is not None:
            if fn < prev_fn:
                nonmono = True
            elif period == prev_period and fn - prev_fn > 1:
                gap = fn - prev_fn - 1
        cur = dict(rec=rec, fn=fn, indices=[idx], conflicts=set(), gap=gap, nonmono=nonmono)
        prev_fn, prev_period = fn, period
    if cur is not None and not (max_frames is not None and emitted >= max_frames):
        yield build(cur)
    if pending is not None:
        raise pending
