"""Tests for football_fatigue.match_data. All data is SYNTHETIC (30 fps) and only mimics the shape of the real files."""

import bz2
import importlib.util
import sys
from pathlib import Path

import pytest

from football_fatigue.ingestion import AttributionStatus, Dataset, EventVersion, IngestionError, QualityFlag as Q
from football_fatigue.match_data import CoordinatePolicy as P, CoordState as S, MatchData, prepare_frame, select, Coordinate

_spec = importlib.util.spec_from_file_location("ingestion_test_helpers", Path(__file__).with_name("test_ingestion.py"))
H = importlib.util.module_from_spec(_spec)
sys.modules["ingestion_test_helpers"] = H
_spec.loader.exec_module(H)

FPS = 30.0


def T(n):
    return n / FPS


def open_md(tmp_path, records, events=None, version=EventVersion.ROOT, **kw):
    raw = H.build(tmp_path, records, events=events, versions=("root", "march", "may"))
    return raw, MatchData.open(raw, "100", version, **kw)


def only(md, **kw):
    return next(iter(md.frames(**kw)))


def home(pf, jersey):
    return next(p for p in pf.players if p.source_side == "home" and p.jersey_number == jersey)


# ---------------------------------------------------------------- entry point, laziness, provenance

def test_event_version_and_policy_must_be_explicit(tmp_path):
    raw = H.build(tmp_path, [H.rec(10)])
    with pytest.raises(TypeError):
        MatchData.open(raw, "100")                                   # no default version
    with pytest.raises(IngestionError):
        MatchData.open(raw, "100", "root")
    with pytest.raises(IngestionError):
        MatchData.open(raw, "100", EventVersion.ROOT, policy="raw")
    with pytest.raises(IngestionError):
        MatchData.open(raw, "../100", EventVersion.ROOT)
    assert "events=root" in repr(MatchData.open(raw, "100", EventVersion.ROOT))


def test_basic_information_does_not_read_the_tracking_file(tmp_path):
    raw = H.build(tmp_path, [H.rec(10)])
    next(raw.rglob("100.jsonl.bz2")).write_bytes(b"not a bz2 stream")          # corrupt tracking
    md = MatchData.open(raw, "100", EventVersion.MAY_1_2025)
    assert md.metadata.match_id == 100 and len(md.roster.entries) == 4
    assert len(md.events()) == 3 and {e.source_version for e in md.events()} == {EventVersion.MAY_1_2025}   # May has the extra IT row
    with pytest.raises(IngestionError):
        next(iter(md.frames()))                                                # only the tracking access fails


def test_frames_are_lazy_chronological_and_never_manufactured(tmp_path):
    records = [H.rec(n) for n in (10, 11, 11, 14, 15)]                         # a gap (12, 13) and a repeated record
    _, md = open_md(tmp_path, records)
    it = md.frames()
    assert iter(it) is it
    frames = list(md.frames())
    assert [f.frame.frame_num for f in frames] == [10, 11, 14, 15]            # no frame is inserted into the gap
    assert [f.frame.source_record_indices for f in frames] == [(0,), (1, 2), (3,), (4,)]
    assert frames[2].frame.frame_gap_before == 2 and Q.FRAME_GAP in frames[2].frame.quality_flags
    assert [f.frame.frame_num for f in md.frames(max_frames=2)] == [10, 11]
    f = frames[0]
    assert f.frame.video_time_s == pytest.approx(T(10)) and f.frame.canonical_period == 1 and f.frame.source.path.endswith("100.jsonl.bz2") and f.period_usable


# ---------------------------------------------------------------- raw / smoothed / missing / ambiguous

def test_raw_and_smoothed_are_distinct_representations(tmp_path):
    hs = [H.player(1, 1.5, 1.5), H.player(2, 2.0, 2.0)]
    _, md = open_md(tmp_path, [H.rec(10, hs=hs)])
    p = home(only(md), 1)
    assert (p.raw.state, p.raw.x, p.raw.y) == (S.AVAILABLE, 1.0, 1.0) and (p.smoothed.state, p.smoothed.x, p.smoothed.y) == (S.AVAILABLE, 1.5, 1.5)
    assert p.selected.source == "raw" and p.selected.x == 1.0                  # the default policy is RAW
    for policy, src, x in ((P.SMOOTHED, "smoothed", 1.5), (P.SMOOTHED_THEN_RAW, "smoothed", 1.5)):
        sel = home(only(MatchData.open(tmp_path / "raw", "100", EventVersion.ROOT, policy=policy)), 1).selected
        assert (sel.source, sel.x) == (src, x)


def test_missing_coordinates_stay_missing_never_zero(tmp_path):
    bad = [H.player(1, "abc", 1.0), H.player(2, float("nan"), float("nan"))]
    _, md = open_md(tmp_path, [H.rec(10, home=bad, balls=[], bs=None)])
    pf = only(md)
    for j in (1, 2):
        p = home(pf, j)
        assert p.raw.state is S.MISSING and p.raw.x is None and p.raw.y is None and p.selected.x is None and p.selected.y is None
        assert p.selected.state is S.MISSING and p.selected.source is None and not p.position_usable
    assert pf.ball.raw.state is S.MISSING and pf.ball.selected.x is None and not pf.ball.position_usable and Q.BALL_MISSING in pf.frame.quality_flags


def test_ambiguous_smoothed_candidates_are_kept_and_never_selected(tmp_path):
    hs = [H.player(1, 1.5, 1.0), H.player(1, 1.6, 1.0), H.player(2, 2.0, 2.0)]
    raw, md = open_md(tmp_path, [H.rec(10, hs=hs)])
    p = home(only(md), 1)
    assert p.smoothed.state is S.AMBIGUOUS and sorted(p.smoothed.candidates) == [(1.5, 1.0), (1.6, 1.0)] and p.smoothed.x is None
    smoothed = home(only(MatchData.open(raw, "100", EventVersion.ROOT, policy=P.SMOOTHED)), 1)
    assert smoothed.selected.state is S.AMBIGUOUS and smoothed.selected.x is None and not smoothed.position_usable          # no candidate, no fallback
    fb = home(only(MatchData.open(raw, "100", EventVersion.ROOT, policy=P.SMOOTHED_THEN_RAW)), 1)
    assert (fb.selected.source, fb.selected.x, fb.selected.reason) == ("raw_fallback", 1.0, "smoothed_ambiguous") and "RAW_FALLBACK_FOR_SMOOTHED" in fb.caveats
    assert fb.position_usable and fb.sample.smoothed_candidates                                                            # candidates stay on the canonical sample


def test_conflicting_duplicate_raw_jerseys_are_ambiguous_under_every_policy(tmp_path):
    dup = [H.player(1, 1.0, 1.0), H.player(1, 9.0, 9.0), H.player(2, 2.0, 2.0)]
    raw, _ = open_md(tmp_path, [H.rec(10, home=dup, hs=[])])
    for policy in P:
        pf = only(MatchData.open(raw, "100", EventVersion.ROOT, policy=policy))
        for p in (x for x in pf.players if x.source_side == "home" and x.jersey_number == 1):
            assert p.raw.state is S.AMBIGUOUS and sorted(p.raw.candidates) == [(1.0, 1.0), (9.0, 9.0)]
            assert p.selected.state is not S.AVAILABLE and p.selected.x is None and not p.position_usable
        assert home(pf, 2).selected.state is S.AVAILABLE or policy is P.SMOOTHED


def test_repeated_frame_smoothed_ball_conflict_stays_unresolved_and_raw_ball_is_unchanged(tmp_path):
    first = H.rec(10, bs=dict(visibility="VISIBLE", x=1.25, y=2.5, z=0.3))
    second = H.rec(10, bs=dict(visibility="VISIBLE", x=9.0, y=8.0, z=0.0))
    raw, md = open_md(tmp_path, [first, second])
    b = only(md).ball
    assert b.smoothed.state is S.AMBIGUOUS and b.smoothed.x is None and b.smoothed.reason == "repeated_records_disagree"
    assert (b.raw.state, b.raw.x, b.raw.y) == (S.AVAILABLE, 0.5, 0.5) and b.selected.source == "raw" and b.position_usable
    sm = only(MatchData.open(raw, "100", EventVersion.ROOT, policy=P.SMOOTHED)).ball
    assert sm.selected.state is S.AMBIGUOUS and sm.selected.x is None
    fb = only(MatchData.open(raw, "100", EventVersion.ROOT, policy=P.SMOOTHED_THEN_RAW)).ball.selected
    assert (fb.source, fb.x) == ("raw_fallback", 0.5)


def test_select_policy_function_is_pure_and_total():
    av, miss, amb = Coordinate(S.AVAILABLE, 1.0, 2.0), Coordinate(S.MISSING), Coordinate(S.AMBIGUOUS)
    assert select(av, miss, P.RAW).source == "raw" and select(miss, av, P.RAW).state is S.MISSING
    assert select(miss, av, P.SMOOTHED).x == 1.0 and select(av, amb, P.SMOOTHED).state is S.AMBIGUOUS
    assert select(amb, miss, P.SMOOTHED_THEN_RAW).state is S.AMBIGUOUS and select(miss, miss, P.SMOOTHED_THEN_RAW).state is S.MISSING
    assert select(miss, miss, P.RAW).x is None                                    # never (0, 0)


# ---------------------------------------------------------------- usability and identity

def test_usability_is_not_physical_validation_and_carries_caveats(tmp_path):
    est = [dict(H.player(1, 1.0, 1.0), visibility="ESTIMATED", confidence="LOW"), H.player(2, 2.0, 2.0)]
    _, md = open_md(tmp_path, [H.rec(10, home=est), H.rec(11, period=0)])
    a, b = md.frames()
    p = home(a, 1)
    assert p.position_usable and {"PLAYER_ESTIMATED", "JERSEY_CONFIDENCE_LOW"} <= p.caveats                      # usable, but not "validated"
    assert not home(b, 1).position_usable and not b.period_usable and b.frame.source_period == 0 and b.frame.canonical_period is None


def test_identity_is_not_resolved_beyond_the_roster_and_quarantine_is_respected(tmp_path):
    recs = [H.rec(10, home=[H.player(1, 1, 1), H.player(99, 2, 2)]), H.rec(11, period=3)]
    _, md = open_md(tmp_path, recs, quarantine_periods=[3])
    a, b = md.frames()
    assert home(a, 1).identity_usable and home(a, 1).player_id == 1001 and home(a, 1).team_id == 10
    u = home(a, 99)
    assert not u.identity_usable and u.attribution_status is AttributionStatus.UNRESOLVED_JERSEY and u.player_id is None and u.position_usable
    assert all(p.attribution_status is AttributionStatus.QUARANTINED and not p.identity_usable and p.player_id is None and p.team_id is None for p in b.players)
    assert all(p.position_usable for p in b.players) and {p.source_side for p in b.players} == {"home", "away"}    # positions kept; sides never swapped


# ---------------------------------------------------------------- synchronized events

def test_events_are_attached_to_their_frames_and_unmatched_events_are_never_hidden(tmp_path):
    events = [H.event(1, 101, T(12)), H.event(2, 102, T(12)), H.event(3, 103, T(15)), H.event(4, 104, 5000.0, period=3), H.event(5, 105, T(12), period=0)]
    _, md = open_md(tmp_path, [H.rec(n) for n in range(10, 20)], events=events)
    attached = {pf.frame.frame_num: [a.event_row_id for a in pf.events] for pf in md.frames(attach_events=True) if pf.events}
    assert attached == {12: ["100:root:0", "100:root:1"], 15: ["100:root:2"]}
    un = md.unmatched_events()
    assert [(a.event_row_id, a.reason.value) for a in un] == [("100:root:3", "no_tracking_for_period"), ("100:root:4", "invalid_source_period")]
    assert sum(len(v) for v in attached.values()) + len(un) == len(md.events()) == 5          # nothing lost
    assert all(a.validated_correspondence == "not_assessed" for a in md.synchronize().alignments)
    assert all(pf.events == () for pf in md.frames())                                         # events are opt-in; plain iteration is single-pass


def test_synchronization_is_cached_and_uses_the_requested_version(tmp_path):
    _, md = open_md(tmp_path, [H.rec(n) for n in range(10, 20)], events=[H.event(1, 101, T(12))], version=EventVersion.MAY_1_2025)
    s1 = md.synchronize()
    assert md.synchronize() is s1 and s1.event_version is EventVersion.MAY_1_2025 and len(s1.alignments) == 2


def test_match_data_never_modifies_raw_data(tmp_path):
    raw, md = open_md(tmp_path, [H.rec(n) for n in range(10, 15)], events=[H.event(1, 101, T(12))])
    before = H.digest(raw)
    list(md.frames(attach_events=True))
    md.unmatched_events(), md.events(), md.roster
    assert H.digest(raw) == before and not (tmp_path / "processed").exists()


# ---------------------------------------------------------------- blocker regressions (Codex review of 2C)

def test_events_attach_to_their_selected_source_frame_not_to_every_frame_with_the_same_number(tmp_path):
    # frame numbers 10, 11, 10 again (a later, separate occurrence at 0.4 s); one event at each frame-10 timestamp
    records = [H.rec(10), H.rec(11), H.rec(10, videoTimeMs=400.0)]
    events = [H.event(1, 101, 10 / 30), H.event(2, 102, 0.4), H.event(3, 103, 9000.0, period=3)]
    _, md = open_md(tmp_path, records, events=events)
    frames = list(md.frames(attach_events=True))
    assert [pf.frame.source_record_indices for pf in frames] == [(0,), (1,), (2,)] and [pf.frame_num for pf in frames] == [10, 11, 10]
    assert [[a.event_row_id for a in pf.events] for pf in frames] == [["100:root:0"], [], ["100:root:1"]]       # 2 events -> exactly 2 attachments
    for pf in frames:
        for a in pf.events:
            assert a.frame_source_record_indices == pf.frame.source_record_indices and a.frame_num == pf.frame_num   # the selected occurrence
    assert [a.event_row_id for a in md.unmatched_events()] == ["100:root:2"]
    assert "100:root:2" not in {a.event_row_id for pf in frames for a in pf.events}                          # unmatched events are never attached
    assert [[a.event_row_id for a in pf.events] for pf in md.frames(attach_events=True)] == [[a.event_row_id for a in pf.events] for pf in frames]  # repeatable


def test_reversed_or_repeated_frame_numbers_never_create_extra_attachments(tmp_path):
    # frame numbers go backwards while time advances; consecutive duplicates collapse; several events share one frame
    records = [H.rec(12, videoTimeMs=100.0), H.rec(12, videoTimeMs=100.0), H.rec(11, videoTimeMs=200.0), H.rec(12, videoTimeMs=300.0)]
    events = [H.event(1, 101, 0.1), H.event(2, 102, 0.1), H.event(3, 103, 0.2), H.event(4, 104, 0.3)]
    _, md = open_md(tmp_path, records, events=events)
    frames = list(md.frames(attach_events=True))
    assert [pf.frame.source_record_indices for pf in frames] == [(0, 1), (2,), (3,)]
    assert [[a.event_row_id for a in pf.events] for pf in frames] == [["100:root:0", "100:root:1"], ["100:root:2"], ["100:root:3"]]   # source order kept
    assert sum(len(pf.events) for pf in frames) == len(events) == len({a.event_row_id for pf in frames for a in pf.events})
    assert all(a.frame_source_record_indices == pf.frame.source_record_indices for pf in frames for a in pf.events)


def ball_conflict_records(order):
    no_ball = H.rec(10, balls=[], bs=None)
    with_ball = H.rec(10)
    return [no_ball, with_ball] if order == "no_ball_first" else [with_ball, no_ball]


@pytest.mark.parametrize("order", ["no_ball_first", "ball_first"])
@pytest.mark.parametrize("policy", list(P))
def test_repeated_ball_conflict_is_ambiguous_in_both_record_orders(tmp_path, order, policy):
    records = ball_conflict_records(order) + [H.rec(11), H.rec(12, balls=[], bs=None)]
    raw, _ = open_md(tmp_path, records)
    md = MatchData.open(raw, "100", EventVersion.ROOT, policy=policy)
    pf, normal, missing = md.frames()
    assert pf.frame.repeat_conflict_fields == ("balls", "ballsSmoothed") and pf.frame.source_record_indices == (0, 1)   # provenance and flags kept
    b = pf.ball
    assert b.raw.state is S.AMBIGUOUS and b.smoothed.state is S.AMBIGUOUS                                         # independent of record order
    assert (b.raw.x, b.raw.y, b.smoothed.x, b.smoothed.y) == (None, None, None, None)
    assert b.selected.state is S.AMBIGUOUS and b.selected.x is None and b.selected.source is None and not b.position_usable   # nothing selected, no fallback
    assert normal.ball.raw.state is S.AVAILABLE and normal.ball.position_usable                                   # unrelated frames unchanged
    assert missing.ball.raw.state is S.MISSING and missing.ball.smoothed.state is S.MISSING and missing.ball.selected.x is None
    m = md.quality_summary().metrics
    assert (m["ball_raw_ambiguous"].count, m["ball_smoothed_ambiguous"].count) == (1, 1)
    assert (m["ball_raw_missing"].count, m["ball_smoothed_missing"].count) == (1, 1)                              # only the genuinely missing frame
    assert (m["ball_raw_available"].count, m["ball_smoothed_available"].count, m["ball_selected_unusable"].count) == (1, 1, 2)
    assert all(m[k].denominator == 3 and m[k].unit == "frames" for k in ("ball_raw_ambiguous", "ball_raw_missing"))


def test_ball_conflict_summaries_are_identical_for_both_orders(tmp_path):
    out = []
    for order in ("no_ball_first", "ball_first"):
        _, md = open_md(tmp_path / order, ball_conflict_records(order))
        pf = next(iter(md.frames()))
        out.append((pf.ball.raw.state, pf.ball.smoothed.state, pf.ball.selected, pf.ball.position_usable,
                    {k: (v.count, v.denominator) for k, v in md.quality_summary().metrics.items() if k.startswith("ball_")}))
    assert out[0] == out[1]
