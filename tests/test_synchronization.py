"""Tests for football_fatigue.synchronization. All data is SYNTHETIC (30 fps, 105x68-style toy values) and only mimics
the shape of the real files; it is not real source data and does not define the real schemas."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from football_fatigue.ingestion import Dataset, EventVersion, IngestionError, QualityFlag as Q
from football_fatigue.synchronization import (
    MatchStatus,
    SpatialStatus,
    SynchronizationError,
    TemporalNote,
    UnmatchedReason as U,
    VALIDATED_CORRESPONDENCE,
    synchronize_match,
)

_spec = importlib.util.spec_from_file_location("ingestion_test_helpers", Path(__file__).with_name("test_ingestion.py"))
H = importlib.util.module_from_spec(_spec)
sys.modules["ingestion_test_helpers"] = H
_spec.loader.exec_module(H)

FPS = 30.0
TOL = 1 / FPS


def T(frame: float) -> float:
    """Video time in seconds of a (possibly fractional) frame number."""
    return frame / FPS


def run(tmp_path, records, events, version=EventVersion.ROOT, **kw):
    raw = H.build(tmp_path, records, events=events, versions=("root", "march", "may"))
    return raw, synchronize_match(Dataset(raw), "100", version, **kw)


def frames(first, last, period=1):
    return [H.rec(n, period=period) for n in range(first, last + 1)]


def ev(ge, t, period=1, **kw):
    return H.event(ge, ge + 100, t, period=period, **kw)


# ---------------------------------------------------------------- temporal matching

def test_exact_match_and_provenance(tmp_path):
    raw, s = run(tmp_path, frames(10, 20), [ev(1, T(12)), ev(2, T(15))])
    a = s.alignments[0]
    assert (a.status, a.reason, a.frame_num) == (MatchStatus.MATCHED, None, 12)
    assert a.residual_s == pytest.approx(0.0, abs=1e-9) and a.tolerance_s == pytest.approx(TOL)
    assert (a.event_row_id, a.event_version, a.source_row_index, a.event_source.index) == ("100:root:0", EventVersion.ROOT, 0, 0)
    assert a.frame_source.path.endswith("100.jsonl.bz2") and a.frame_source_record_indices == (2,) and a.frame_period == 1
    assert a.frame_video_time_s == pytest.approx(T(12)) and a.canonical_period == 1 and a.policy_version
    assert s.fps == FPS and s.scan_complete and s.fatal_error is None and s.tracking_period_ranges[1][2] == 11


def test_nearest_frame_selection_and_signed_residual(tmp_path):
    _, s = run(tmp_path, frames(10, 20), [ev(1, T(12.4)), ev(2, T(12.6)), ev(3, T(14.9))])
    assert [a.frame_num for a in s.alignments] == [12, 13, 15]
    assert s.alignments[0].residual_s == pytest.approx(T(0.4)) and s.alignments[1].residual_s == pytest.approx(-T(0.4))


def test_exact_tie_goes_to_the_earlier_frame(tmp_path):
    _, s = run(tmp_path, frames(10, 20), [ev(1, (T(12) + T(13)) / 2)])
    assert s.alignments[0].frame_num == 12


def test_tolerance_boundary_is_one_frame_interval(tmp_path):
    last = T(20)
    events = [ev(1, last + TOL), ev(2, last + TOL + 1e-6), ev(3, T(10) - TOL), ev(4, T(10) - TOL - 1e-6)]
    _, s = run(tmp_path, frames(10, 20), events)
    a = s.alignments
    assert a[0].status is MatchStatus.MATCHED and a[0].frame_num == 20 and TemporalNote.NEAR_PERIOD_BOUNDARY in a[0].temporal_notes
    assert a[1].reason is U.OUTSIDE_PERIOD_FRAME_RANGE and a[1].frame_num is None and a[1].residual_s is None
    assert a[2].status is MatchStatus.MATCHED and a[2].frame_num == 10
    assert a[3].reason is U.OUTSIDE_PERIOD_FRAME_RANGE
    assert s.alignments[0].tolerance_s == s.alignments[1].tolerance_s == pytest.approx(TOL)


def test_custom_tolerance_and_invalid_arguments(tmp_path):
    _, s = run(tmp_path, frames(10, 20), [ev(1, T(12) + 0.02)], tolerance_s=0.01)
    assert s.alignments[0].reason is U.NO_FRAME_WITHIN_TOLERANCE and s.tolerance_s == 0.01
    raw = H.build(tmp_path / "x", frames(10, 12), versions=("root",))
    for bad in (0, -1, float("nan"), float("inf")):
        with pytest.raises(SynchronizationError):
            synchronize_match(Dataset(raw), "100", EventVersion.ROOT, tolerance_s=bad)


def test_events_before_and_after_the_tracking_range(tmp_path):
    _, s = run(tmp_path, frames(30, 40), [ev(1, 0.0), ev(2, 100.0), ev(3, T(35))])
    assert [a.reason for a in s.alignments] == [U.OUTSIDE_PERIOD_FRAME_RANGE, U.OUTSIDE_PERIOD_FRAME_RANGE, None]


# ---------------------------------------------------------------- periods

def test_same_period_matching_only(tmp_path):
    recs = frames(10, 14, 1) + frames(15, 19, 2)
    events = [ev(1, T(16), period=2), ev(2, T(16), period=1), ev(3, T(16), period=3), ev(4, T(12), period=1)]
    _, s = run(tmp_path, recs, events)
    r = s.alignments
    assert (r[0].status, r[0].frame_num, r[0].frame_period) == (MatchStatus.MATCHED, 16, 2)
    assert r[1].reason is U.PERIOD_LABEL_DIFFERS_FROM_NEAREST_FRAME and r[1].frame_num is None     # period-2 frame is not a period-1 candidate
    assert r[2].reason is U.PERIOD_LABEL_DIFFERS_FROM_NEAREST_FRAME
    assert r[3].frame_num == 12 and set(s.tracking_period_ranges) == {1, 2}


def test_invalid_periods_and_missing_time_are_preserved_not_matched(tmp_path):
    events = [ev(1, T(12), period=0), ev(2, T(12), period=9), H.event(3, 103, None), ev(4, T(12))]
    _, s = run(tmp_path, frames(10, 20), events)
    r = s.alignments
    assert (r[0].reason, r[0].source_period, r[0].canonical_period) == (U.INVALID_SOURCE_PERIOD, 0, None)
    assert (r[1].reason, r[1].source_period) == (U.INVALID_SOURCE_PERIOD, 9)
    assert r[2].reason is U.MISSING_EVENT_TIME and r[2].event_time_s is None and Q.EVENT_TIME_MISSING in r[2].event_quality_flags
    assert r[3].status is MatchStatus.MATCHED and len(r) == 4                                      # nothing dropped


def test_event_only_extra_time_has_no_tracking_period_and_no_invented_frames(tmp_path):
    events = [ev(1, T(12), period=1), ev(2, 5000.0, period=3), ev(3, 5100.0, period=4)]
    _, s = run(tmp_path, frames(10, 20) + frames(21, 25, 2), events)
    assert [a.reason for a in s.alignments] == [None, U.NO_TRACKING_FOR_PERIOD, U.NO_TRACKING_FOR_PERIOD]
    assert all(a.frame_num is None and a.frame_source is None and a.frame_source_record_indices == () for a in s.alignments[1:])
    assert s.summary()["unmatched_reasons"] == {"no_tracking_for_period": 2}


# ---------------------------------------------------------------- repeated records, event versions, determinism

def test_repeated_tracking_records_are_one_frame_with_all_indices(tmp_path):
    recs = [H.rec(10), H.rec(11), H.rec(11), H.rec(11, bs=dict(visibility="VISIBLE", x=9.0, y=9.0, z=0.0)), H.rec(12)]
    _, s = run(tmp_path, recs, [ev(1, T(11))])
    a = s.alignments[0]
    assert a.frame_num == 11 and a.frame_source_record_indices == (1, 2, 3)
    assert Q.REPEATED_RECORDS_PRESENT in a.frame_quality_flags and Q.BALL_SMOOTHED_UNRESOLVED in a.frame_quality_flags


def test_event_version_is_explicit_and_rows_are_never_substituted(tmp_path):
    raw = H.build(tmp_path, frames(10, 20), events=[ev(1, T(12)), ev(2, T(13))], versions=("root", "may"))
    ds = Dataset(raw)
    root = synchronize_match(ds, "100", EventVersion.ROOT)
    may = synchronize_match(ds, "100", EventVersion.MAY_1_2025)
    assert len(root.alignments) == 2 and len(may.alignments) == 3                                 # May has the extra IT row
    assert {a.event_version for a in may.alignments} == {EventVersion.MAY_1_2025} and may.event_version is EventVersion.MAY_1_2025
    assert may.alignments[2].event_row_id == "100:may_1_2025:2"
    with pytest.raises(IngestionError, match="march_14_2025"):
        synchronize_match(ds, "100", EventVersion.MARCH_14_2025)                                  # absent: no fallback
    with pytest.raises(SynchronizationError):
        synchronize_match(ds, "100", "root")
    with pytest.raises(IngestionError):
        synchronize_match(ds, "../100", EventVersion.ROOT)


def test_every_event_is_preserved_in_source_order_including_duplicate_keys(tmp_path):
    events = [ev(1, T(15)), ev(1, T(12), gtype="FOUL", ptype="FO") | {"possessionEventId": 101}, ev(3, T(10)), ev(4, T(99), period=3)]
    _, s = run(tmp_path, frames(10, 20), events)
    assert [a.source_row_index for a in s.alignments] == [0, 1, 2, 3]
    assert [a.frame_num for a in s.alignments] == [15, 12, 10, None]                              # out-of-order times, order preserved


def test_output_is_deterministic(tmp_path):
    raw = H.build(tmp_path, frames(10, 40), events=[ev(n, T(10 + n * 0.5)) for n in range(1, 40)], versions=("root",))
    one = synchronize_match(Dataset(raw), "100", EventVersion.ROOT, spatial=True)
    two = synchronize_match(Dataset(raw), "100", EventVersion.ROOT, spatial=True)
    assert one == two and json.dumps(one.summary(), sort_keys=True) == json.dumps(two.summary(), sort_keys=True)


# ---------------------------------------------------------------- temporal versus spatial labels

def squad(shift=0.0):
    return [H.player(j, float(j) + shift, 0.0) for j in range(1, 12)]


def spatial_event(ge, t, shift=0.0, **kw):
    return H.event(ge, ge + 100, t, homePlayers=squad(shift), awayPlayers=[], ball=[dict(visibility="VISIBLE", x=0.5 + shift, y=0.5, z=0.1)], **kw)


def spatial_frames(n0, n1, **kw):
    return [H.rec(n, home=squad(), away=[], **kw) for n in range(n0, n1 + 1)]


def test_temporal_match_is_not_spatial_verification(tmp_path):
    events = [spatial_event(1, T(12)), spatial_event(2, T(13), shift=5.0)]
    _, plain = run(tmp_path / "a", spatial_frames(10, 20), events)
    for a in plain.alignments:
        assert a.status is MatchStatus.MATCHED and a.spatial.status is SpatialStatus.NOT_ASSESSED and a.quality_label == "temporal_only"
        assert a.validated_correspondence == VALIDATED_CORRESPONDENCE == "not_assessed"
    _, sp = run(tmp_path / "b", spatial_frames(10, 20), events, spatial=True)
    good, bad = sp.alignments
    assert good.spatial.status is SpatialStatus.COORDINATE_AGREEING and good.quality_label == "coordinate_agreeing"
    assert good.spatial.players_compared == 11 and good.spatial.players_within_tolerance == 11 and good.spatial.ball == "agrees"
    assert bad.status is MatchStatus.MATCHED and bad.spatial.status is SpatialStatus.COORDINATE_DISAGREEING            # not corrected, just reported
    assert bad.spatial.ball == "differs" and bad.quality_label == "temporal_only_coordinates_disagree"
    assert all(a.validated_correspondence == "not_assessed" for a in sp.alignments)
    assert sp.summary()["quality_labels"] == {"coordinate_agreeing": 1, "temporal_only_coordinates_disagree": 1}


def test_spatial_is_not_assessable_without_comparable_valid_observations(tmp_path):
    ev_few = H.event(1, 101, T(12))                                                              # one embedded player only
    ev_nan = spatial_event(2, T(13))
    ev_nan["homePlayers"][0]["x"] = float("nan")
    _, s = run(tmp_path, spatial_frames(10, 20), [ev_few, ev_nan], spatial=True)
    assert [a.spatial.status for a in s.alignments] == [SpatialStatus.NOT_ASSESSABLE] * 2
    assert s.alignments[1].spatial.players_compared == 10                                         # the malformed observation is not compared


def test_conflicting_frame_data_is_never_used_for_spatial_assessment(tmp_path):
    recs = spatial_frames(10, 11) + [H.rec(12, home=squad(), away=[]), H.rec(12, home=squad(3.0), away=[], balls=[]), H.rec(13, home=squad(), away=[])]
    _, s = run(tmp_path, recs, [spatial_event(1, T(12))], spatial=True)
    a = s.alignments[0]
    assert a.spatial.status is SpatialStatus.NOT_ASSESSABLE and a.spatial.reason == "frame_player_data_conflict" and a.spatial.ball == "not_assessable"
    assert Q.REPEATED_RECORDS_DIFFER in a.frame_quality_flags


def test_missing_frame_ball_makes_only_the_ball_comparison_unassessable(tmp_path):
    recs = [H.rec(n, home=squad(), away=[], balls=[], bs=None) for n in range(10, 15)]
    _, s = run(tmp_path, recs, [spatial_event(1, T(12))], spatial=True)
    a = s.alignments[0]
    assert a.spatial.status is SpatialStatus.COORDINATE_AGREEING and a.spatial.ball == "not_assessable" and Q.BALL_MISSING in a.frame_quality_flags


def test_quarantined_periods_are_excluded_from_spatial_assessment(tmp_path):
    raw = H.build(tmp_path, spatial_frames(10, 20), events=[spatial_event(1, T(12))], versions=("root",))
    ds = Dataset(raw)
    plain = synchronize_match(ds, "100", EventVersion.ROOT, spatial=True)
    held = synchronize_match(ds, "100", EventVersion.ROOT, spatial=True, quarantine_periods=[1])
    assert plain.alignments[0].spatial.status is SpatialStatus.COORDINATE_AGREEING
    assert held.alignments[0].status is MatchStatus.MATCHED                                       # timing is unaffected by identity quarantine
    assert held.alignments[0].spatial.status is SpatialStatus.NOT_ASSESSABLE and held.alignments[0].spatial.players_compared == 0


def test_event_coordinates_are_reference_observations_and_the_flagged_discrepancy_is_not_corrected(tmp_path):
    _, s = run(tmp_path, spatial_frames(10, 20), [spatial_event(1, T(12), shift=2.0)], spatial=True)
    a = s.alignments[0]
    assert a.spatial.status is SpatialStatus.COORDINATE_DISAGREEING and a.frame_num == 12 and a.residual_s == pytest.approx(0, abs=1e-9)


# ---------------------------------------------------------------- failed scans and read-only behaviour

def test_failed_scan_makes_no_absence_claims_and_keeps_all_events(tmp_path):
    events = [ev(1, T(11)), ev(2, T(30)), ev(3, T(500), period=2), ev(4, T(12), period=3)]
    recs = frames(10, 13) + ["{broken"] + frames(30, 33)
    _, s = run(tmp_path, recs, events)
    r = s.alignments
    assert not s.scan_complete and "record 4" in s.fatal_error and len(r) == 4
    assert r[0].status is MatchStatus.MATCHED and r[0].frame_num == 11                           # settled before the failure point
    assert [x.reason for x in r[1:]] == [U.TRACKING_SCAN_INCOMPLETE] * 3
    assert all(x.reason not in (U.NO_TRACKING_FOR_PERIOD, U.OUTSIDE_PERIOD_FRAME_RANGE) for x in r)


def test_non_monotonic_frames_are_rejected_not_silently_misaligned(tmp_path):
    recs = frames(10, 12) + [H.rec(11)] + frames(13, 14)
    _, s = run(tmp_path, recs, [ev(1, T(10.5)), ev(2, T(13))])
    assert not s.scan_complete and "increasing time order" in s.fatal_error
    assert s.alignments[1].reason is U.TRACKING_SCAN_INCOMPLETE


def test_synchronization_never_modifies_raw_data(tmp_path):
    raw = H.build(tmp_path, spatial_frames(10, 20), events=[spatial_event(1, T(12))], versions=("root", "march", "may"))
    before = H.digest(raw)
    ds = Dataset(raw)
    for v in EventVersion:
        synchronize_match(ds, "100", v, spatial=True)
    assert H.digest(raw) == before and not (tmp_path / "processed").exists()


# ================================================================ final blocking fixes (Codex review of 2B)

from football_fatigue.synchronization import align_events  # noqa: E402


# ---------------------------------------------------------------- 1. non-monotonic tracking is never definitive

def test_backward_timestamp_makes_every_tracking_dependent_result_non_definitive(tmp_path):
    recs = [H.rec(10), H.rec(20), H.rec(11)]                                    # 10 -> 20 -> 11
    events = [ev(1, T(11)), ev(2, T(11.2)), ev(3, T(10)), ev(4, T(15)), ev(5, T(20)), ev(6, T(12), period=0), H.event(7, 107, None)]
    _, s = run(tmp_path, recs, events)
    assert s.ordering_violation is True and s.scan_complete is False and "increasing time order" in s.fatal_error
    r = s.alignments
    assert len(r) == 7
    assert all(x.status is MatchStatus.UNMATCHED and x.frame_num is None and x.residual_s is None for x in r[:5])    # nothing finalized
    assert [x.reason for x in r[:5]] == [U.TRACKING_SCAN_INCOMPLETE] * 5                                           # not "matched", not "no_frame_within_tolerance"
    assert [x.reason for x in r[5:]] == [U.INVALID_SOURCE_PERIOD, U.MISSING_EVENT_TIME]                            # independent of tracking order
    assert not any(x.reason in (U.NO_FRAME_WITHIN_TOLERANCE, U.NO_TRACKING_FOR_PERIOD, U.OUTSIDE_PERIOD_FRAME_RANGE) for x in r)
    assert s.summary()["ordering_violation"] is True


def test_ordinary_failed_scan_is_not_an_ordering_violation_and_ordered_streams_are_unchanged(tmp_path):
    recs = frames(10, 13) + ["{broken"]
    _, s = run(tmp_path / "a", recs, [ev(1, T(11)), ev(2, T(500))])
    assert s.scan_complete is False and s.ordering_violation is False
    assert s.alignments[0].status is MatchStatus.MATCHED and s.alignments[1].reason is U.TRACKING_SCAN_INCOMPLETE   # prefix results still settle
    _, ok = run(tmp_path / "b", frames(10, 20), [ev(1, T(12)), ev(2, T(500))])
    assert ok.scan_complete and not ok.ordering_violation and ok.alignments[0].frame_num == 12 and ok.alignments[1].reason is U.OUTSIDE_PERIOD_FRAME_RANGE


# ---------------------------------------------------------------- 2. match and version context on the public API

def _two(tmp_path):
    raw1 = H.build(tmp_path / "m100", frames(10, 20), events=[ev(1, T(12)), ev(2, T(13))], versions=("root", "may"))
    ev200 = [H.event(1, 101, T(12)) | {"gameId": 200}, H.event(2, 102, T(13)) | {"gameId": 200}]
    raw2 = H.build(tmp_path / "m200", frames(10, 20), events=ev200, versions=("root",), metadata=[dict(H.METADATA[0], id="200")], mid="200")
    d1, d2 = Dataset(raw1), Dataset(raw2)
    return (d1.load_events("100", EventVersion.ROOT), d1.load_events("100", EventVersion.MAY_1_2025), list(d1.iter_tracking_frames("100")),
            d2.load_events("200", EventVersion.ROOT), list(d2.iter_tracking_frames("200")))


def test_align_events_rejects_foreign_events_frames_and_mixed_versions(tmp_path):
    root100, may100, frames100, root200, frames200 = _two(tmp_path)
    ok = align_events(root100, frames100, match_id=100, event_version=EventVersion.ROOT, fps=FPS)
    assert [a.frame_num for a in ok.alignments] == [12, 13]                                           # valid inputs still work
    with pytest.raises(SynchronizationError, match="belongs to match 200, not the requested match 100"):
        align_events(root200, frames100, match_id=100, event_version=EventVersion.ROOT, fps=FPS)       # foreign events
    with pytest.raises(SynchronizationError, match="tracking frame .* belongs to match 200"):
        align_events(root100, frames200, match_id=100, event_version=EventVersion.ROOT, fps=FPS)       # foreign frames
    with pytest.raises(SynchronizationError, match="tracking frame .* belongs to match 200"):
        align_events(root100, frames100[:5] + frames200[5:], match_id=100, event_version=EventVersion.ROOT, fps=FPS)   # foreign frame mid-stream
    with pytest.raises(SynchronizationError, match="has version may_1_2025, not the requested root"):
        align_events(root100 + may100, frames100, match_id=100, event_version=EventVersion.ROOT, fps=FPS)  # mixed versions
    with pytest.raises(SynchronizationError, match="has version root, not the requested may_1_2025"):
        align_events(root100, frames100, match_id=100, event_version=EventVersion.MAY_1_2025, fps=FPS)     # version differs from the request
    with pytest.raises(SynchronizationError):
        align_events(root100, frames100, match_id=100, event_version="root", fps=FPS)                      # not an EventVersion
    assert issubclass(SynchronizationError, IngestionError)


# ---------------------------------------------------------------- 3. ambiguous duplicate jerseys in spatial assessment

def squad14(shift=0.0):
    return [H.player(j, float(j) + shift, 0.0) for j in range(1, 15)]


def conflicting(players, jerseys, delta=3.0):
    """Add a conflicting duplicate (shifted coordinates) for each listed jersey."""
    return players + [H.player(j, float(j) + delta, 0.0) for j in jerseys]


def spatial_run(tmp_path, event_home, frame_home, reverse=False):
    if reverse:
        event_home, frame_home = list(reversed(event_home)), list(reversed(frame_home))
    ev_ = H.event(1, 101, T(12), homePlayers=event_home, awayPlayers=[], ball=[])
    recs = [H.rec(n, home=frame_home, away=[]) for n in range(10, 15)]
    _, s = run(tmp_path, recs, [ev_], spatial=True)
    return s.alignments[0]


@pytest.mark.parametrize("reverse", [False, True])
def test_all_jerseys_conflicting_on_the_event_side_is_not_assessable_and_order_independent(tmp_path, reverse):
    a = spatial_run(tmp_path, conflicting(squad(), range(1, 12)), squad(), reverse)
    assert a.status is MatchStatus.MATCHED and a.frame_num == 12                                   # temporal result preserved
    assert a.spatial.status is SpatialStatus.NOT_ASSESSABLE and a.spatial.players_compared == 0 and a.spatial.players_excluded_ambiguous == 11
    assert "ambiguous_duplicate_jerseys_excluded" in a.spatial.reason and a.quality_label == "temporal_only"


@pytest.mark.parametrize("reverse", [False, True])
def test_all_jerseys_conflicting_on_the_tracking_side_is_not_assessable_and_order_independent(tmp_path, reverse):
    a = spatial_run(tmp_path, squad(), conflicting(squad(), range(1, 12)), reverse)
    assert a.status is MatchStatus.MATCHED and a.spatial.status is SpatialStatus.NOT_ASSESSABLE and a.spatial.players_excluded_ambiguous == 11
    assert Q.DUPLICATE_RAW_JERSEY in a.frame_quality_flags


@pytest.mark.parametrize("reverse", [False, True])
def test_enough_unambiguous_players_remain_assessable_and_results_do_not_depend_on_order(tmp_path, reverse):
    # 14 players; 3 conflicting duplicates on each side (different jerseys) leave 8 unambiguous ones -> not assessable;
    # 1 conflicting jersey leaves 13 -> assessable, and the ambiguous jersey never decides the outcome
    few = spatial_run(tmp_path / "few", conflicting(squad14(), [1, 2, 3]), conflicting(squad14(), [4, 5, 6]), reverse)
    assert few.spatial.status is SpatialStatus.NOT_ASSESSABLE and few.spatial.players_compared == 8 and few.spatial.players_excluded_ambiguous == 6
    one = spatial_run(tmp_path / "one", conflicting(squad14(), [1]), squad14(), reverse)
    assert one.spatial.status is SpatialStatus.COORDINATE_AGREEING and one.spatial.players_compared == 13 and one.spatial.players_excluded_ambiguous == 1
    assert one.spatial.players_within_tolerance == 13


def test_identical_duplicates_are_not_ambiguous_and_invalid_duplicates_do_not_poison_a_group(tmp_path):
    same = spatial_run(tmp_path / "same", squad() + [H.player(3, 3.0, 0.0)], squad())
    assert same.spatial.status is SpatialStatus.COORDINATE_AGREEING and same.spatial.players_compared == 11 and same.spatial.players_excluded_ambiguous == 0
    bad = H.player(3, float("nan"), 0.0)
    inval = spatial_run(tmp_path / "inval", squad() + [bad], squad())
    assert inval.spatial.status is SpatialStatus.COORDINATE_AGREEING and inval.spatial.players_compared == 11      # invalid entry skipped, as before


def test_conflicting_duplicate_event_balls_are_not_compared(tmp_path):
    two_balls = [dict(visibility="VISIBLE", x=0.5, y=0.5, z=0.1), dict(visibility="VISIBLE", x=7.0, y=7.0, z=0.1)]
    for balls in (two_balls, list(reversed(two_balls))):
        ev_ = H.event(1, 101, T(12), homePlayers=squad(), awayPlayers=[], ball=balls)
        _, s = run(tmp_path / str(balls[0]["x"]), [H.rec(n, home=squad(), away=[]) for n in range(10, 15)], [ev_], spatial=True)
        assert s.alignments[0].spatial.ball == "not_assessable" and s.alignments[0].spatial.status is SpatialStatus.COORDINATE_AGREEING
