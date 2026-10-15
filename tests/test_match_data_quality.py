"""Integration tests for MatchData quality summaries and the playback contract. They use the real 2A ingestion and 2B
synchronization interfaces on SYNTHETIC data (30 fps); nothing here is real source data."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from football_fatigue.ingestion import EventVersion, IngestionError, QualityFlag as Q
from football_fatigue.match_data import COORDINATE_SYSTEM, CoordinatePolicy as P, CoordState as S, MatchData

_spec = importlib.util.spec_from_file_location("ingestion_test_helpers", Path(__file__).with_name("test_ingestion.py"))
H = importlib.util.module_from_spec(_spec)
sys.modules["ingestion_test_helpers"] = H
_spec.loader.exec_module(H)

FPS = 30.0


def T(n):
    return n / FPS


def counted_records():
    """3 frames, 13 player observations with a known set of problems (see test_summary_counts_and_denominators)."""
    est = dict(H.player(3, -3.0, -3.0), visibility="ESTIMATED")
    f10 = H.rec(10, home=[H.player(1, "abc", 1.0), H.player(2, 2.0, 2.0)], away=[H.player(1, -1.0, -1.0), est])
    f11 = H.rec(11, hs=[H.player(1, 1.0, 1.0), H.player(2, 2.0, 2.0), H.player(2, 2.5, 2.0)])
    nov = dict(H.player(1, 1.0, 1.0), visibility=None)
    f12 = H.rec(12, home=[nov, H.player(2, 2.0, 2.0), H.player(2, 9.0, 9.0)], balls=[], bs=None)
    return [f10, f11, f12]


def open_md(tmp_path, records, events=None, version=EventVersion.ROOT, **kw):
    raw = H.build(tmp_path, records, events=events, versions=("root", "march", "may"))
    return raw, MatchData.open(raw, "100", version, **kw)


# ---------------------------------------------------------------- denominators and counts

def test_summary_counts_and_denominators(tmp_path):
    _, md = open_md(tmp_path, counted_records())
    q = md.quality_summary()
    m = q.metrics
    assert q.scan.status == "complete" and q.scan.frames_scanned == 3 and q.scan.fatal_error is None
    assert (m["ball_raw_missing"].count, m["ball_raw_missing"].denominator, m["ball_raw_missing"].unit) == (1, 3, "frames")
    assert (m["ball_raw_available"].count, m["ball_smoothed_missing"].count, m["ball_smoothed_available"].count) == (2, 1, 2)
    obs = 13
    for name, count in (("player_raw_missing", 1), ("player_raw_ambiguous", 2), ("player_raw_available", 10), ("player_smoothed_missing", 1),
                        ("player_smoothed_ambiguous", 3), ("player_smoothed_available", 9), ("player_selected_unusable", 3),
                        ("player_estimated", 1), ("player_visibility_unknown", 1), ("player_measured_visible", 11), ("identity_resolved", 13)):
        assert (m[name].count, m[name].denominator, m[name].unit) == (count, obs, "player_observations"), name
    assert m["player_raw_missing"].fraction == pytest.approx(1 / 13)
    assert q.breakdowns["players_per_side_per_frame"] == {"home:2": 2, "home:3": 1, "away:2": 3}                  # absent records are not zeros
    assert q.breakdowns["selected_coordinate_source"] == {"raw": 10, "none": 3}
    assert q.breakdown_units["players_per_side_per_frame"].startswith("frame_sides") and q.breakdown_units["selected_coordinate_source"] == "player_observations"
    assert q.tracking_frames_by_canonical_period == {"1": 3} and q.event_periods_without_tracking == ()
    assert any("ESTIMATED" in w for w in q.warnings) and any("ambiguous" in w for w in q.warnings)


def test_coordinate_source_selection_is_reflected_in_the_summary(tmp_path):
    raw, _ = open_md(tmp_path, counted_records())
    sm = MatchData.open(raw, "100", EventVersion.ROOT, policy=P.SMOOTHED).quality_summary()
    fb = MatchData.open(raw, "100", EventVersion.ROOT, policy=P.SMOOTHED_THEN_RAW).quality_summary()
    assert sm.breakdowns["selected_coordinate_source"] == {"smoothed": 9, "none": 4} and sm.policy == "smoothed"
    assert fb.breakdowns["selected_coordinate_source"] == {"smoothed": 9, "raw_fallback": 1, "none": 3}      # 3 raw-ambiguous/ambiguous stay unselected
    assert fb.metrics["player_selected_unusable"].count == 3


def test_no_false_zero_coordinates_anywhere(tmp_path):
    _, md = open_md(tmp_path, counted_records())
    for policy in P:
        md.policy = policy
        for pf in md.frames():
            coords = [c for p in pf.players for c in (p.raw, p.smoothed, p.selected)] + [pf.ball.raw, pf.ball.smoothed, pf.ball.selected]
            for c in coords:
                if c.state is not S.AVAILABLE:
                    assert (c.x, c.y) == (None, None)
            assert all(p.selected.x != 0.0 or p.selected.state is S.AVAILABLE for p in pf.players)


# ---------------------------------------------------------------- scan completeness

def test_partial_complete_and_failed_scans(tmp_path):
    events = [H.event(1, 101, T(11)), H.event(2, 102, 5000.0, period=3)]
    _, md = open_md(tmp_path / "a", counted_records(), events=events)
    full = md.quality_summary()
    assert full.scan.status == "complete" and full.event_periods_without_tracking == (3,) and any("event-only" in w for w in full.warnings)
    part = md.quality_summary(max_frames=2)
    assert part.scan.status == "partial" and part.scan.frames_scanned == 2 and part.scan.max_frames == 2
    assert part.event_periods_without_tracking is None and part.metrics["ball_raw_missing"].denominator == 2    # denominators follow the scanned frames
    assert md.quality_summary(max_frames=3).scan.status == "complete" and md.quality_summary(max_frames=99).scan.status == "complete"
    _, bad = open_md(tmp_path / "b", [H.rec(10), H.rec(11), "{broken", H.rec(30, period=3)], events=events)
    failed = bad.quality_summary()
    assert failed.scan.status == "failed" and failed.scan.frames_scanned == 2 and "record 2" in failed.scan.fatal_error
    assert failed.event_periods_without_tracking is None                                                      # never claims absence after a failure
    assert failed.metrics["ball_raw_available"].count == 2 and any("failed tracking scan" in w for w in failed.warnings)


def test_corrupt_tracking_does_not_prevent_metadata_roster_events_or_a_failed_summary(tmp_path):
    raw = H.build(tmp_path, [H.rec(10)])
    next(raw.rglob("100.jsonl.bz2")).write_bytes(b"not bz2")
    md = MatchData.open(raw, "100", EventVersion.ROOT)
    assert md.metadata.fps == 30.0 and len(md.roster.entries) == 4 and len(md.events()) == 2
    q = md.quality_summary()
    assert q.scan.status == "failed" and q.scan.frames_scanned == 0 and q.metrics["events_total"].count == 2


def test_opening_a_match_never_scans_tracking(tmp_path):
    raw = H.build(tmp_path, [H.rec(10)])
    next(raw.rglob("100.jsonl.bz2")).write_bytes(b"not bz2")
    md = MatchData.open(raw, "100", EventVersion.ROOT)       # would raise if opening scanned
    assert md.metadata is md.metadata and md._sync == {}


# ---------------------------------------------------------------- extra time, versions, events

def test_missing_extra_time_tracking_and_explicit_event_versions(tmp_path):
    events = [H.event(1, 101, T(11)), H.event(2, 102, 6000.0, period=3), H.event(3, 103, 6100.0, period=4)]
    raw, root = open_md(tmp_path, [H.rec(n) for n in range(10, 14)], events=events)
    q = root.quality_summary(include_synchronization=True)
    assert q.event_periods_without_tracking == (3, 4) and q.event_rows_by_canonical_period == {"1": 1, "3": 1, "4": 1}
    assert q.metrics["events_total"].count == 3 and q.synchronization["unmatched_reasons"] == {"no_tracking_for_period": 2}
    assert q.breakdowns["unmatched_reasons"] == {"no_tracking_for_period": 2} and q.breakdown_units["unmatched_reasons"] == "events"
    assert (q.metrics["events_temporally_matched"].count, q.metrics["events_unmatched"].count, q.metrics["events_unmatched"].denominator) == (1, 2, 3)
    may = MatchData.open(raw, "100", EventVersion.MAY_1_2025).quality_summary()
    assert may.event_version == "may_1_2025" and may.metrics["events_total"].count == 4 and may.synchronization is None   # IT row; no sync unless asked
    assert "events_temporally_matched" not in may.metrics


def test_event_attachment_loses_and_duplicates_no_event(tmp_path):
    events = [H.event(i, 100 + i, T(10 + i % 6)) for i in range(1, 12)] + [H.event(20, 120, 9e3, period=3), H.event(21, 121, T(12), period=0)]
    _, md = open_md(tmp_path, [H.rec(n) for n in range(10, 20)], events=events)
    ids = [a.event_row_id for pf in md.frames(attach_events=True) for a in pf.events]
    un = [a.event_row_id for a in md.unmatched_events()]
    assert len(ids) == len(set(ids)) == 11 and sorted(un) == ["100:root:11", "100:root:12"]
    assert sorted(ids + un, key=lambda s: int(s.split(":")[-1])) == [e.event_row_id for e in md.events()]
    for pf in md.frames(attach_events=True):
        for a in pf.events:
            assert a.frame_num == pf.frame_num and a.status.value == "matched" and a.validated_correspondence == "not_assessed"


def test_identity_quarantine_is_counted_and_reported(tmp_path):
    _, md = open_md(tmp_path, [H.rec(10), H.rec(11, period=3)], quarantine_periods=[3])
    q = md.quality_summary()
    assert q.metrics["identity_quarantined"].count == 4 and q.metrics["identity_resolved"].count == 4 and q.metrics["identity_quarantined"].denominator == 8
    assert q.quarantined_periods == (3,) and any("quarantined" in w for w in q.warnings)


# ---------------------------------------------------------------- contract, provenance, determinism, immutability

def test_playback_contract_exposes_stable_typed_fields_without_defaults_for_missing_data(tmp_path):
    _, md = open_md(tmp_path, [H.rec(10, period=0), H.rec(11, balls=[], bs=None)], events=[H.event(1, 101, T(11))])
    a, b = md.frames(attach_events=True)
    assert (a.frame_num, a.video_time_s, a.source_period, a.canonical_period) == (10, pytest.approx(T(10)), 0, None) and not a.period_usable
    assert a.frame.source_record_indices == (0,) and a.frame.source.path.endswith("100.jsonl.bz2") and Q.INVALID_SOURCE_PERIOD in a.quality_flags
    p = a.players[0]
    assert (p.source_side, p.jersey_number, p.selected.source, p.position_usable, p.identity_usable) == ("home", 1, "raw", False, True)   # period invalid -> not usable
    assert p.sample.quality_flags is not None and p.team_id == 10 and p.player_id == 1001
    assert b.ball.selected.state is S.MISSING and b.ball.selected.x is None and Q.BALL_MISSING in b.ball.quality_flags and not b.ball.position_usable
    (ev,) = b.events
    assert (ev.event_row_id, ev.event_version, ev.status.value, ev.frame_num) == ("100:root:0", EventVersion.ROOT, "matched", 11)
    assert ev.residual_s == pytest.approx(0, abs=1e-9) and ev.frame_source_record_indices == (1,) and ev.validated_correspondence == "not_assessed"
    assert COORDINATE_SYSTEM["name"] == "pitch_center_m" and COORDINATE_SYSTEM["unit"] == "meter"


def test_summaries_and_frames_are_deterministic_and_leave_raw_data_untouched(tmp_path):
    raw, md = open_md(tmp_path, counted_records(), events=[H.event(1, 101, T(11))])
    before = H.digest(raw)
    s1 = md.quality_summary(include_synchronization=True)
    s2 = MatchData.open(raw, "100", EventVersion.ROOT).quality_summary(include_synchronization=True)
    assert s1 == s2 and json.dumps(s1.to_dict(), sort_keys=True) == json.dumps(s2.to_dict(), sort_keys=True)
    f1 = [(pf.frame_num, tuple(p.selected for p in pf.players)) for pf in md.frames(attach_events=True)]
    f2 = [(pf.frame_num, tuple(p.selected for p in pf.players)) for pf in md.frames(attach_events=True)]
    assert f1 == f2 and H.digest(raw) == before and not (tmp_path / "processed").exists()


def test_source_record_provenance_survives_quality_scans(tmp_path):
    records = [H.rec(10), H.rec(11), H.rec(11), H.rec(12)]
    _, md = open_md(tmp_path, records)
    q = md.quality_summary()
    assert q.metrics["frames_with_repeated_source_records"].count == 1 and q.scan.frames_scanned == 3
    assert [pf.frame.source_record_indices for pf in md.frames()] == [(0,), (1, 2), (3,)]
