"""Deterministic validation and per-match quality reporting for Milestone 2A.

Fatal structural errors (unreadable or structurally unusable sources) are reported separately from recoverable
data-quality issues (flag counts with explicit denominators). Nothing is repaired, swapped, interpolated, or
invented: periods without tracking are reported as such, never filled.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .loaders import Dataset, validate_match_id
from .models import CANONICAL_PERIODS, EventVersion, IngestionError, MatchMetadata, QualityFlag as Q, Roster, TrackingFrame

REPORT_SCHEMA_VERSION = 1


@dataclass(slots=True)
class TrackingQuality:
    """Streaming accumulator: constant memory regardless of match length."""

    frames: int = 0
    source_records: int = 0
    first_video_time_s: float | None = None
    last_video_time_s: float | None = None
    player_samples: int = 0
    frame_flags: Counter = field(default_factory=Counter)
    sample_flags: Counter = field(default_factory=Counter)
    smoothed_status: Counter = field(default_factory=Counter)
    attribution_status: Counter = field(default_factory=Counter)
    frames_by_source_period: Counter = field(default_factory=Counter)
    frames_by_canonical_period: Counter = field(default_factory=Counter)
    frames_without_ball: int = 0
    frames_with_ball_smoothed_xy_null: int = 0
    frames_with_ball_smoothed_unresolved: int = 0
    unmatched_smoothed_entries: int = 0

    def add(self, f: TrackingFrame) -> None:
        self.frames += 1
        self.source_records += f.source_record_count
        t = f.video_time_s
        self.first_video_time_s = t if self.first_video_time_s is None else self.first_video_time_s
        self.last_video_time_s = t
        for fl in f.quality_flags:
            self.frame_flags[fl.value] += 1
        self.frames_by_source_period["null" if f.source_period is None else str(f.source_period)] += 1
        if f.canonical_period is not None:
            self.frames_by_canonical_period[str(f.canonical_period)] += 1
        self.frames_without_ball += f.ball is None or f.ball.x_raw_m is None
        self.frames_with_ball_smoothed_xy_null += Q.BALL_SMOOTHED_XY_NULL in f.quality_flags
        self.frames_with_ball_smoothed_unresolved += Q.BALL_SMOOTHED_UNRESOLVED in f.quality_flags
        self.unmatched_smoothed_entries += len(f.unmatched_smoothed)
        for p in f.players:
            self.player_samples += 1
            self.smoothed_status[p.smoothed_status.value] += 1
            self.attribution_status[p.attribution_status.value] += 1
            for fl in p.quality_flags:
                self.sample_flags[fl.value] += 1


@dataclass(slots=True)
class MatchQualityReport:
    match_id: str
    event_version: str
    fatal_errors: list[str] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)
    tracking: dict[str, Any] = field(default_factory=dict)
    events: dict[str, Any] = field(default_factory=dict)
    periods: dict[str, Any] = field(default_factory=dict)
    roster_issues: list[str] = field(default_factory=list)
    scope: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        if self.fatal_errors:
            return "fatal"
        counts = [*self.tracking.get("frame_flag_counts", {}).values(), *self.tracking.get("sample_flag_counts", {}).values(),
                  *self.events.get("flag_counts", {}).values(), *self.periods.get("flag_counts", {}).values()]
        return "ok_with_quality_issues" if any(counts) or self.roster_issues else "ok"

    def to_dict(self) -> dict[str, Any]:
        return dict(schema_version=REPORT_SCHEMA_VERSION, match_id=self.match_id, event_version=self.event_version, status=self.status,
                    fatal_errors=self.fatal_errors, scope=self.scope, sources=self.sources, tracking=self.tracking, events=self.events,
                    periods=self.periods, roster_issues=self.roster_issues)

    def summary_text(self) -> str:
        t, e, p = self.tracking, self.events, self.periods
        lines = [f"match {self.match_id} [events: {self.event_version}] status: {self.status}"]
        lines += [f"  FATAL: {x}" for x in self.fatal_errors]
        if t:
            lines.append(f"  tracking: {t['source_records']} records -> {t['frames']} frames; repeated-record frames {t['frame_flag_counts'].get('REPEATED_RECORDS_PRESENT', 0)}; "
                         f"frames without raw ball {t['frames_without_raw_ball']} of {t['frames']}; player samples {t['player_samples']}; "
                         f"smoothed ambiguous {t['smoothed_status'].get('ambiguous_conflicting', 0)}; identity {t['attribution_status']}")
        if e:
            lines.append(f"  events: {e['rows']} rows; invalid source period {e['invalid_source_period_rows']}; rows in repeated key pairs {e['rows_in_repeated_key_pairs']}")
        if p:
            lines.append(f"  periods: events {p['event_rows_by_canonical_period']}, tracking frames {p['tracking_frames_by_canonical_period']}; "
                         f"event periods with no tracking: {p['event_periods_without_tracking']}")
        if self.scope.get("max_frames") is not None:
            lines.append(f"  NOTE: partial scan (first {self.scope['max_frames']} frames only); tracking counts are not whole-match totals")
        return "\n".join(lines)


def quality_report(dataset: Dataset, match_id: object, version: EventVersion, *, max_frames: int | None = None,
                   quarantine_periods: Iterable[int] = ()) -> MatchQualityReport:
    """Stream the match's tracking once and load the requested event version; never raises for source problems."""
    mid = validate_match_id(match_id)
    if not isinstance(version, EventVersion):
        raise IngestionError("an EventVersion must be named explicitly")
    rep = MatchQualityReport(mid, version.value)
    rep.scope = dict(max_frames=max_frames, quarantine_periods=sorted(set(quarantine_periods)), note="denominators are stated per count; periods without tracking are reported, never filled")
    meta: MatchMetadata | None = None
    roster: Roster | None = None
    try:
        meta = dataset.metadata(mid)
        rep.sources["metadata"] = meta.source.path
        roster = dataset.roster(mid, meta)
        rep.sources["roster"] = roster.entries[0].source.path if roster.entries else ""
        rep.roster_issues = list(roster.issues)
    except IngestionError as e:
        rep.fatal_errors.append(f"metadata/roster: {e}")
    # events
    ev_periods: Counter = Counter()
    try:
        rows = dataset.load_events(mid, version)
        flag_counts: Counter = Counter()
        for r in rows:
            ev_periods["null" if r.source_period is None else str(r.source_period)] += 1
            for fl in r.quality_flags:
                flag_counts[fl.value] += 1
        rep.sources["events"] = rows[0].source.path if rows and rows[0].source else ""
        rep.events = dict(rows=len(rows), by_source_period=dict(ev_periods), invalid_source_period_rows=flag_counts.get("INVALID_SOURCE_PERIOD", 0),
                          rows_in_repeated_key_pairs=flag_counts.get("EVENT_KEY_PAIR_REPEATED", 0), flag_counts=dict(flag_counts),
                          denominator="rows")
    except IngestionError as e:
        rep.fatal_errors.append(f"events: {e}")
        rows = []
    # tracking
    q = TrackingQuality()
    complete = False
    try:
        _, src = dataset.tracking_source(mid)
        rep.sources["tracking"] = src.path
        for fr in dataset.iter_tracking_frames(mid, roster=roster, metadata=meta, quarantine_periods=quarantine_periods, max_frames=max_frames):
            q.add(fr)
        complete = max_frames is None   # a bounded scan is complete only by construction of its limit, so it is never "complete"
    except IngestionError as e:
        rep.fatal_errors.append(f"tracking: {e}")
    rep.tracking = dict(
        scan_complete=complete, frames=q.frames, source_records=q.source_records, player_samples=q.player_samples, first_video_time_s=q.first_video_time_s,
        last_video_time_s=q.last_video_time_s, frame_flag_counts=dict(q.frame_flags), frame_flag_denominator="frames",
        sample_flag_counts=dict(q.sample_flags), sample_flag_denominator="player_samples", smoothed_status=dict(q.smoothed_status),
        attribution_status=dict(q.attribution_status), frames_by_source_period=dict(q.frames_by_source_period),
        frames_without_raw_ball=q.frames_without_ball, frames_with_ball_smoothed_xy_null=q.frames_with_ball_smoothed_xy_null,
        frames_with_ball_smoothed_unresolved=q.frames_with_ball_smoothed_unresolved,
        unmatched_smoothed_entries=q.unmatched_smoothed_entries,
    )
    # periods: never fabricate missing tracking
    ev_canon = Counter(str(r.canonical_period) for r in rows if r.canonical_period is not None)
    # absence of a period can be asserted only after a scan that read the whole tracking file successfully
    missing = None if not complete else sorted(int(k) for k in ev_canon if k not in q.frames_by_canonical_period)
    rep.periods = dict(
        event_rows_by_canonical_period=dict(ev_canon), tracking_frames_by_canonical_period=dict(q.frames_by_canonical_period),
        event_periods_without_tracking=missing, flag_counts={Q.NO_TRACKING_FOR_PERIOD.value: len(missing)} if missing else {},
        note=None if complete else ("not assessed: partial scan" if max_frames is not None else "not assessed: tracking scan failed or is incomplete"),
        tracking_periods_without_events=sorted(int(k) for k in q.frames_by_canonical_period if k not in ev_canon),
        canonical_periods=list(CANONICAL_PERIODS),
    )
    return rep


# ---------------------------------------------------------------- safe output

def _is_link(p: Path) -> bool:
    return p.is_symlink() or (hasattr(os.path, "isjunction") and os.path.isjunction(p))


def _under(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def approved_output_root(raw_root: str | Path) -> Path:
    """Trusted boundary, taken lexically: ``<resolved dataset root's parent>/processed``.

    The boundary is deliberately *not* resolved through ``processed`` itself, so a link or junction named
    ``processed`` cannot redirect it (see ``_assert_no_links``). Outputs may be written only below it.
    """
    return Path(os.path.abspath(Path(raw_root).resolve().parent / "processed"))


def _assert_no_links(anchor: Path, target: Path) -> None:
    """Every existing component from ``anchor`` (a resolved directory) down to ``target`` must be a real path:
    not a symlink or junction, and resolving to itself."""
    cur = anchor
    for part in target.relative_to(anchor).parts:
        cur = cur / part
        if _is_link(cur):
            raise IngestionError(f"output path component is a link or junction: {cur}")
        if cur.exists() and cur.resolve() != cur:
            raise IngestionError(f"output path component is redirected (resolves to {cur.resolve()}): {cur}")


def safe_output_path(raw_root: str | Path, match_id: object, version: EventVersion, out_root: str | Path | None = None, suffix: str = ".json") -> Path:
    """Validated destination for one report (or its temporary file, ``suffix=".json.tmp"``), checked immediately before use."""
    mid = validate_match_id(match_id)
    raw = Path(raw_root).resolve()
    data_dir = raw.parent
    approved = approved_output_root(raw_root)
    base = Path(os.path.abspath(out_root)) if out_root is not None else approved / "ingestion_quality"
    if _under(base, raw) or not _under(base, approved):
        raise IngestionError(f"output location {base} is not inside the approved output root {approved} (or lies inside the raw data)")
    dest = base / mid / f"{version.value}{suffix}"
    _assert_no_links(data_dir, dest)
    return dest


def write_quality_report(report: MatchQualityReport, raw_root: str | Path, out_root: str | Path | None = None, *, overwrite: bool = False) -> Path:
    """Write the report JSON under the approved output root. Existing reports are never replaced silently."""
    version = EventVersion(report.event_version)
    dest = safe_output_path(raw_root, report.match_id, version, out_root)
    tmp = safe_output_path(raw_root, report.match_id, version, out_root, suffix=".json.tmp")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest = safe_output_path(raw_root, report.match_id, version, out_root)      # re-validate after directories exist
    tmp = safe_output_path(raw_root, report.match_id, version, out_root, suffix=".json.tmp")
    if dest.exists() and not overwrite:
        raise IngestionError(f"report already exists (use overwrite=True): {dest}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=1)
    safe_output_path(raw_root, report.match_id, version, out_root)
    safe_output_path(raw_root, report.match_id, version, out_root, suffix=".json.tmp")
    os.replace(tmp, dest)
    return dest
