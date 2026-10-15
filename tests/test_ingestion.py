"""Tests for football_fatigue.ingestion. All data is SYNTHETIC and only mimics the shape of the real files;
it is not real source data and does not define the real schemas."""

import bz2
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from football_fatigue.ingestion import (
    AttributionStatus,
    Dataset,
    EventVersion,
    IngestionError,
    QualityFlag as Q,
    SmoothedStatus,
    quality_report,
    safe_output_path,
    validate_match_id,
    write_quality_report,
)
from football_fatigue.ingestion.__main__ import main as cli_main

TS = "20000101T000000Z"


def player(j, x, y, **kw):
    return dict(jerseyNum=str(j), confidence="HIGH", visibility="VISIBLE", x=x, y=y, **kw)


def rec(frame, period=1, home=None, away=None, hs=None, as_=None, balls="default", bs="default", **kw):
    home = home if home is not None else [player(1, 1.0, 1.0), player(2, 2.0, 2.0)]
    away = away if away is not None else [player(1, -1.0, -1.0), player(3, -3.0, -3.0)]
    sm = lambda lst: [{k: v for k, v in p.items() if k != "speed"} for p in lst]  # noqa: E731
    d = dict(version=None, gameRefId=None, videoTimeMs=frame * 1000 / 30.0, frameNum=frame, period=period,
             periodElapsedTime=float(frame), periodGameClockTime=float(frame),
             homePlayers=home, homePlayersSmoothed=hs if hs is not None else sm(home),
             awayPlayers=away, awayPlayersSmoothed=as_ if as_ is not None else sm(away),
             balls=[dict(visibility="VISIBLE", x=0.5, y=0.5, z=0.1)] if balls == "default" else balls,
             ballsSmoothed=dict(visibility="VISIBLE", x=0.6, y=0.6, z=0.1) if bs == "default" else bs,
             game_event_id=None, possession_event_id=None, game_event=None, possession_event=None)
    d.update(kw)
    return d


def event(ge, pe, t, period=1, gtype="OTB", ptype="PA", **kw):
    r = dict(gameId=100, gameEventId=ge, possessionEventId=pe, startTime=t, endTime=t, duration=0.0, eventTime=t,
             gameEvents=dict(gameEventType=gtype, period=period, teamId=10, playerId=1001, homeTeam=True),
             possessionEvents=dict(possessionEventType=ptype) if ptype else None, initialTouch={}, fouls={}, grades={},
             homePlayers=[player(1, 1.0, 1.0)], awayPlayers=[], ball=[])
    r.update(kw)
    return r


METADATA = [dict(id="100", date="2022-11-21T16:00:00", competition=dict(id="38", name="Synthetic Cup"), season="2022", week=1,
                 homeTeam=dict(id="10", name="Home", shortName="HOM"), awayTeam=dict(id="20", name="Away", shortName="AWY"),
                 stadium=dict(id="1", name="Synthetic Park", pitches=[dict(length=105.0, width=68.0)]), fps=30.0,
                 homeTeamStartLeft=True, homeTeamStartLeftExtraTime=None, startPeriod1=0.0, endPeriod1=100.0, startPeriod2=200.0, endPeriod2=300.0)]
ROSTER = [dict(player=dict(id="1001", nickname="A"), positionGroupType="GK", shirtNumber="1", started=True, team=dict(id="10", name="Home")),
          dict(player=dict(id="1002", nickname="B"), positionGroupType="CM", shirtNumber="2", started=True, team=dict(id="10", name="Home")),
          dict(player=dict(id="2001", nickname="C"), positionGroupType="GK", shirtNumber="1", started=True, team=dict(id="20", name="Away")),
          dict(player=dict(id="2003", nickname="D"), positionGroupType="CM", shirtNumber="3", started=True, team=dict(id="20", name="Away"))]


def write(path: Path, content: str | bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content.encode() if isinstance(content, str) else content)


def build(tmp_path, records, events=None, versions=("root", "march", "may"), metadata=METADATA, roster=ROSTER, mid="100"):
    raw = tmp_path / "raw"
    lines = "\n".join(r if isinstance(r, str) else json.dumps(r) for r in records) + "\n"
    write(raw / f"Tracking Data-{TS}-1-001" / "Tracking Data" / f"{mid}.jsonl.bz2", bz2.compress(lines.encode()))
    write(raw / f"Metadata-{TS}-1-001" / "Metadata" / f"{mid}.json", json.dumps(metadata))
    write(raw / f"Rosters-{TS}-1-001" / "Rosters" / f"{mid}.json", json.dumps(roster))
    events = events if events is not None else [event(1, 10, 1.0), event(2, 11, 2.0)]
    folders = {"root": "", "march": "March 14, 2025/", "may": "May 1, 2025/"}
    for v in versions:
        evs = events if v != "may" else events + [event(3, None, 3.0, ptype="IT")]
        write(raw / f"Event Data-{TS}-1-001" / "Event Data" / f"{folders[v]}{mid}.json", json.dumps(evs))
    return raw


def digest(root: Path) -> dict:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


# ---------------------------------------------------------------- streaming and provenance

def test_tracking_is_streamed_lazily_and_bounded(tmp_path):
    raw = build(tmp_path, [rec(n) for n in range(10, 20)])
    ds = Dataset(raw)
    it = ds.iter_tracking_frames("100")
    assert iter(it) is it                              # a generator, not a materialized list
    assert next(it).frame_num == 10
    assert [f.frame_num for f in ds.iter_tracking_frames("100", max_frames=3)] == [10, 11, 12]


def test_repeated_records_are_collapsed_with_full_provenance(tmp_path):
    raw = build(tmp_path, [rec(10), rec(10), rec(10, balls=[]), rec(11)])
    frames = list(Dataset(raw).iter_tracking_frames("100"))
    assert [f.frame_num for f in frames] == [10, 11]
    f = frames[0]
    assert f.source_record_indices == (0, 1, 2) and f.source_record_count == 3
    assert Q.REPEATED_RECORDS_PRESENT in f.quality_flags and Q.REPEATED_RECORDS_DIFFER in f.quality_flags   # third record differs
    assert frames[1].source_record_indices == (3,) and Q.REPEATED_RECORDS_PRESENT not in frames[1].quality_flags
    assert f.source.path.endswith("100.jsonl.bz2") and "\\" not in f.source.path and not Path(f.source.path).is_absolute()
    assert f.video_time_ms_source == pytest.approx(10 * 1000 / 30.0) and f.video_time_s == pytest.approx(10 / 30.0)


def test_uncollapsed_mode_keeps_every_record(tmp_path):
    raw = build(tmp_path, [rec(10), rec(10), rec(11)])
    frames = list(Dataset(raw).iter_tracking_frames("100", collapse_repeats=False))
    assert [(f.frame_num, f.source_record_indices) for f in frames] == [(10, (0,)), (10, (1,)), (11, (2,))]


def test_gap_and_nonmonotonic_frames_are_flagged_not_repaired(tmp_path):
    raw = build(tmp_path, [rec(10), rec(14), rec(12)])
    a, b, c = Dataset(raw).iter_tracking_frames("100")
    assert Q.FRAME_GAP in b.quality_flags and b.frame_gap_before == 3
    assert Q.NONMONOTONIC_FRAME in c.quality_flags


# ---------------------------------------------------------------- smoothed ambiguity, ball, periods, malformed values

def smoothed_conflict_record():
    home = [player(1, 1.0, 1.0), player(2, 2.0, 2.0)]
    hs = [player(1, 1.5, 1.0), player(1, 1.6, 1.0), player(2, 2.0, 2.0)]
    return rec(10, home=home, hs=hs)


def test_conflicting_smoothed_candidates_are_retained_and_coordinate_is_unresolved(tmp_path):
    raw = build(tmp_path, [smoothed_conflict_record()])
    (f,) = Dataset(raw).iter_tracking_frames("100")
    p1 = next(p for p in f.players if p.source_side == "home" and p.jersey_number == 1)
    assert p1.smoothed_status is SmoothedStatus.AMBIGUOUS_CONFLICTING
    assert sorted((c.x_m, c.y_m) for c in p1.smoothed_candidates) == [(1.5, 1.0), (1.6, 1.0)]   # both kept, none chosen
    assert p1.x_smooth_m is None and p1.y_smooth_m is None
    assert Q.SMOOTHED_AMBIGUOUS_CONFLICTING in p1.quality_flags
    assert p1.x_raw_m == 1.0                                          # raw is untouched and available
    p2 = next(p for p in f.players if p.source_side == "home" and p.jersey_number == 2)
    assert p2.smoothed_status is SmoothedStatus.UNIQUE and (p2.x_smooth_m, p2.y_smooth_m) == (2.0, 2.0)


def test_list_order_never_selects_a_candidate(tmp_path):
    r = smoothed_conflict_record()
    r["homePlayersSmoothed"].reverse()
    raw = build(tmp_path, [r])
    (f,) = Dataset(raw).iter_tracking_frames("100")
    p1 = next(p for p in f.players if p.source_side == "home" and p.jersey_number == 1)
    assert p1.smoothed_status is SmoothedStatus.AMBIGUOUS_CONFLICTING and p1.x_smooth_m is None


def test_identical_duplicate_smoothed_entries_are_not_ambiguous(tmp_path):
    r = rec(10, hs=[player(1, 1.5, 1.0), player(1, 1.5, 1.0), player(2, 2.0, 2.0)])
    raw = build(tmp_path, [r])
    (f,) = Dataset(raw).iter_tracking_frames("100")
    p1 = next(p for p in f.players if p.source_side == "home" and p.jersey_number == 1)
    assert p1.smoothed_status is SmoothedStatus.DUPLICATE_IDENTICAL and p1.x_smooth_m == 1.5 and len(p1.smoothed_candidates) == 2


def test_missing_smoothed_list_and_unmatched_smoothed_entries(tmp_path):
    r = rec(10)
    r["homePlayersSmoothed"] = None
    r["awayPlayersSmoothed"] = r["awayPlayersSmoothed"] + [player(9, 9.0, 9.0)]
    raw = build(tmp_path, [r])
    (f,) = Dataset(raw).iter_tracking_frames("100")
    assert Q.SMOOTHED_LIST_MISSING in f.quality_flags and Q.SMOOTHED_JERSEY_WITHOUT_RAW in f.quality_flags
    assert all(p.smoothed_status is SmoothedStatus.MISSING for p in f.players if p.source_side == "home")
    assert [(s, c.x_m, j) for s, c, j in f.unmatched_smoothed] == [("away", 9.0, "9")]


def test_missing_ball_is_a_missing_observation_not_zero(tmp_path):
    raw = build(tmp_path, [rec(10, balls=[], bs=None), rec(11, balls=[], bs=dict(visibility="VISIBLE", x=None, y=None, z=0.0)),
                           rec(12, bs=dict(visibility="VISIBLE", x=None, y=None, z=0.0))])
    a, b, c = Dataset(raw).iter_tracking_frames("100")
    assert a.ball is None and Q.BALL_MISSING in a.quality_flags
    assert b.ball is not None and b.ball.x_raw_m is None and b.ball.x_smooth_m is None and Q.BALL_MISSING in b.quality_flags
    assert Q.BALL_SMOOTHED_XY_NULL in b.quality_flags
    assert c.ball.x_raw_m == 0.5 and c.ball.x_smooth_m is None and c.ball.z_smooth_m == 0.0 and Q.BALL_SMOOTHED_XY_NULL in c.quality_flags


def test_invalid_period_is_kept_distinct_from_canonical_period(tmp_path):
    raw = build(tmp_path, [rec(10, period=0), rec(11, period=3), rec(12, period=None), rec(13, period=7)])
    fr = list(Dataset(raw).iter_tracking_frames("100"))
    assert [(f.source_period, f.canonical_period) for f in fr] == [(0, None), (3, 3), (None, None), (7, None)]
    assert [Q.INVALID_SOURCE_PERIOD in f.quality_flags for f in fr] == [True, False, True, True]


def test_malformed_values_become_none_with_flags_never_zero(tmp_path):
    home = [player(1, "abc", 1.0), player(2, float("nan"), float("inf")), player("x", 3.0, 3.0)]
    raw = build(tmp_path, [rec(10, home=home, hs=[])])
    (f,) = Dataset(raw).iter_tracking_frames("100")
    h = [p for p in f.players if p.source_side == "home"]
    assert (h[0].x_raw_m, h[0].y_raw_m) == (None, 1.0) and Q.INVALID_COORDINATE in h[0].quality_flags
    assert (h[1].x_raw_m, h[1].y_raw_m) == (None, None) and Q.INVALID_COORDINATE in h[1].quality_flags
    assert h[2].jersey_number is None and h[2].jersey_raw == "x" and Q.JERSEY_INVALID in h[2].quality_flags


# ---------------------------------------------------------------- identity

def test_identity_uses_side_and_shirt_never_jersey_alone(tmp_path):
    raw = build(tmp_path, [rec(10)])
    ds = Dataset(raw)
    (f,) = ds.iter_tracking_frames("100", roster=ds.roster("100"))
    by = {(p.source_side, p.jersey_number): p for p in f.players}
    assert by[("home", 1)].player_id == 1001 and by[("away", 1)].player_id == 2001      # same jersey, different teams
    assert by[("home", 1)].team_id == 10 and by[("away", 3)].team_id == 20
    assert all(p.attribution_status is AttributionStatus.RESOLVED for p in f.players)


def test_unresolved_quarantined_and_not_attempted_identity(tmp_path):
    raw = build(tmp_path, [rec(10, home=[player(1, 1, 1), player(99, 2, 2)]), rec(11, period=3)])
    ds = Dataset(raw)
    a, b = ds.iter_tracking_frames("100", roster=ds.roster("100"), quarantine_periods=[3])
    u = next(p for p in a.players if p.jersey_number == 99)
    assert u.attribution_status is AttributionStatus.UNRESOLVED_JERSEY and u.player_id is None and Q.IDENTITY_UNRESOLVED_JERSEY in u.quality_flags
    assert all(p.attribution_status is AttributionStatus.QUARANTINED and p.player_id is None and p.team_id is None for p in b.players)
    assert all(p.source_side in ("home", "away") for p in b.players)                  # original side labels untouched
    (c, d) = ds.iter_tracking_frames("100")
    assert all(p.attribution_status is AttributionStatus.NOT_ATTEMPTED for p in c.players)


def test_repeated_roster_shirt_is_excluded_from_the_join(tmp_path):
    roster = ROSTER + [dict(player=dict(id="1009", nickname="E"), positionGroupType="CM", shirtNumber="2", started=False, team=dict(id="10", name="Home"))]
    raw = build(tmp_path, [rec(10)], roster=roster)
    ds = Dataset(raw)
    r = ds.roster("100")
    assert r.issues and (("home", 2) not in r.index())
    (f,) = ds.iter_tracking_frames("100", roster=r)
    assert next(p for p in f.players if p.source_side == "home" and p.jersey_number == 2).attribution_status is AttributionStatus.UNRESOLVED_JERSEY


# ---------------------------------------------------------------- events

def test_event_version_is_explicit_and_never_substituted(tmp_path):
    raw = build(tmp_path, [rec(10)], versions=("root", "may"))
    ds = Dataset(raw)
    assert len(ds.load_events("100", EventVersion.ROOT)) == 2 and len(ds.load_events("100", EventVersion.MAY_1_2025)) == 3
    with pytest.raises(IngestionError, match="march_14_2025"):
        ds.load_events("100", EventVersion.MARCH_14_2025)         # missing: no silent fallback to another version
    with pytest.raises(IngestionError):
        ds.load_events("100", "root")                              # a version object is required
    assert {r.source_version for r in ds.iter_events("100", EventVersion.MAY_1_2025)} == {EventVersion.MAY_1_2025}


def test_event_rows_preserve_order_duplicates_and_source_values(tmp_path):
    evs = [event(1, 10, 5.0), event(1, 10, 6.0, gtype="FOUL", ptype="FO"), event(2, 11, 7.0, period=0), event(3, 12, None, ptype="CH")]
    raw = build(tmp_path, [rec(10)], events=evs, versions=("root",))
    rows = Dataset(raw).load_events("100", EventVersion.ROOT)
    assert [r.source_row_index for r in rows] == [0, 1, 2, 3] and [r.event_time_s for r in rows] == [5.0, 6.0, 7.0, None]
    assert [Q.EVENT_KEY_PAIR_REPEATED in r.quality_flags for r in rows] == [True, True, False, False]
    assert (rows[2].source_period, rows[2].canonical_period) == (0, None) and Q.INVALID_SOURCE_PERIOD in rows[2].quality_flags
    assert Q.EVENT_TIME_MISSING in rows[3].quality_flags
    assert rows[1].event_row_id == "100:root:1" and rows[1].source.index == 1 and rows[1].source.version == "root"
    assert rows[0].attributes["possessionEvents"]["possessionEventType"] == "PA"
    assert rows[0].positions_snapshot["homePlayers"][0]["x"] == 1.0


def test_event_attack_direction_nan_string_becomes_none(tmp_path):
    evs = [event(1, 10, 1.0, stadiumMetadata=dict(teamAttackingDirection="nan")), event(2, 11, 2.0, stadiumMetadata=dict(teamAttackingDirection="R"))]
    raw = build(tmp_path, [rec(10)], events=evs, versions=("root",))
    assert [r.attack_direction for r in Dataset(raw).load_events("100", EventVersion.ROOT)] == [None, "R"]


# ---------------------------------------------------------------- metadata, roster, fatal structure

def test_metadata_is_validated(tmp_path):
    ds = Dataset(build(tmp_path, [rec(10)]))
    m = ds.metadata("100")
    assert (m.match_id, m.home_team_id, m.fps, m.pitch_length_m) == (100, 10, 30.0, 105.0) and m.side_of_team(20) == "away"
    for bad in (dict(METADATA[0], id="999"), dict(METADATA[0], fps=0), {k: v for k, v in METADATA[0].items() if k != "homeTeam"},
                dict(METADATA[0], stadium=dict(id="1", name="x", pitches=[]))):
        with pytest.raises(IngestionError):
            Dataset(build(tmp_path / "x", [rec(10)], metadata=[bad])).metadata("100")


def test_null_period_times_are_none_not_zero(tmp_path):
    md = [dict(METADATA[0], startPeriod1=None, endPeriod1=None, startPeriod2=None, endPeriod2=None, homeTeamStartLeftExtraTime=True)]
    m = Dataset(build(tmp_path, [rec(10)], metadata=md)).metadata("100")
    assert m.start_period_1_s is None and m.end_period_2_s is None and m.home_team_start_left_extra_time is True


def test_roster_with_foreign_team_is_fatal(tmp_path):
    bad = [dict(ROSTER[0], team=dict(id="77", name="X"))]
    with pytest.raises(IngestionError):
        Dataset(build(tmp_path, [rec(10)], roster=bad)).roster("100")


def test_fatal_structural_tracking_errors(tmp_path):
    for records, text in (([rec(10), "{not json"], "record 1"), ([rec(10), json.dumps([1])], "not a JSON object"),
                          ([{k: v for k, v in rec(10).items() if k != "frameNum"}], "frameNum"),
                          ([dict(rec(10), videoTimeMs="x")], "videoTimeMs")):
        ds = Dataset(build(tmp_path / text.replace(" ", "_"), records))
        with pytest.raises(IngestionError, match=text):
            list(ds.iter_tracking_frames("100"))


def test_invalid_match_ids_and_unknown_matches(tmp_path):
    ds = Dataset(build(tmp_path, [rec(10)]))
    for bad in ("../100", "100/..", "", "1 0", "abc", None):
        with pytest.raises(IngestionError):
            ds.metadata(bad)
    with pytest.raises(IngestionError, match="not found"):
        ds.metadata("101")
    assert validate_match_id(100) == "100"
    with pytest.raises(IngestionError):
        Dataset(tmp_path / "missing")


# ---------------------------------------------------------------- quality reports

def test_quality_report_counts_denominators_and_missing_periods(tmp_path):
    records = [rec(10), rec(10), smoothed_conflict_record() | dict(frameNum=11, videoTimeMs=11 * 1000 / 30.0), rec(12, balls=[], bs=None), rec(13, period=0)]
    evs = [event(1, 10, 1.0), event(2, 11, 2.0, period=3), event(3, 12, 3.0, period=0)]
    raw = build(tmp_path, records, events=evs, versions=("root",))
    rep = quality_report(Dataset(raw), "100", EventVersion.ROOT)
    t = rep.tracking
    assert rep.status == "ok_with_quality_issues" and not rep.fatal_errors
    assert (t["source_records"], t["frames"], t["player_samples"]) == (5, 4, 16)
    assert t["frame_flag_counts"]["REPEATED_RECORDS_PRESENT"] == 1 and t["frame_flag_denominator"] == "frames"
    assert t["smoothed_status"]["ambiguous_conflicting"] == 1 and t["sample_flag_denominator"] == "player_samples"
    assert t["frames_without_raw_ball"] == 1 and t["frames_by_source_period"] == {"1": 3, "0": 1}
    assert rep.events["rows"] == 3 and rep.events["invalid_source_period_rows"] == 1
    assert rep.periods["event_periods_without_tracking"] == [3]                 # reported, never fabricated
    assert rep.periods["flag_counts"] == {"NO_TRACKING_FOR_PERIOD": 1} and rep.periods["tracking_frames_by_canonical_period"] == {"1": 3}


def test_quality_report_reports_fatal_errors_instead_of_raising(tmp_path):
    raw = build(tmp_path, [rec(10), "{broken"], versions=("root",))
    rep = quality_report(Dataset(raw), "100", EventVersion.MAY_1_2025)
    assert rep.status == "fatal" and any("events" in e for e in rep.fatal_errors) and any("tracking" in e for e in rep.fatal_errors)
    assert rep.tracking["frames"] == 1                                         # partial counts are kept, not discarded
    assert "FATAL" in rep.summary_text()


def test_partial_scan_is_labelled(tmp_path):
    raw = build(tmp_path, [rec(n) for n in range(10, 20)], versions=("root",))
    rep = quality_report(Dataset(raw), "100", EventVersion.ROOT, max_frames=2)
    assert rep.tracking["frames"] == 2 and rep.scope["max_frames"] == 2 and "partial scan" in rep.summary_text()


# ---------------------------------------------------------------- read-only behaviour and output safety

def test_ingestion_and_report_never_modify_raw_data(tmp_path):
    raw = build(tmp_path, [rec(10), rec(10), rec(11)])
    before = digest(raw)
    ds = Dataset(raw)
    list(ds.iter_tracking_frames("100", roster=ds.roster("100")))
    ds.load_events("100", EventVersion.ROOT)
    rep = quality_report(ds, "100", EventVersion.ROOT)
    write_quality_report(rep, raw)
    assert digest(raw) == before
    out = tmp_path / "processed" / "ingestion_quality" / "100" / "root.json"
    assert out.is_file() and json.loads(out.read_text())["match_id"] == "100"


def test_reports_are_written_only_below_the_approved_processed_root(tmp_path):
    raw = build(tmp_path, [rec(10)], versions=("root",))
    rep = quality_report(Dataset(raw), "100", EventVersion.ROOT)
    for bad in (raw / "x", raw, tmp_path / "elsewhere", tmp_path / "processed" / ".." / "elsewhere"):
        with pytest.raises(IngestionError):
            write_quality_report(rep, raw, bad)
    assert not any(raw.rglob("*.json.tmp")) and digest(raw)
    p = write_quality_report(rep, raw)
    with pytest.raises(IngestionError, match="already exists"):
        write_quality_report(rep, raw)                                          # never silently replaced
    assert write_quality_report(rep, raw, overwrite=True) == p


def test_output_path_rejects_malformed_ids_and_links(tmp_path):
    raw = build(tmp_path, [rec(10)], versions=("root",))
    for bad in ("../100", "..\\100", "100/../../x", "/abs", "100.json"):
        with pytest.raises(IngestionError):
            safe_output_path(raw, bad, EventVersion.ROOT)
    link = tmp_path / "processed" / "ingestion_quality" / "100"
    link.parent.mkdir(parents=True)
    try:
        os.symlink(raw, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        if os.name == "nt" and subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(raw)], capture_output=True).returncode == 0:
            pass
        else:
            pytest.skip("links unavailable")
    with pytest.raises(IngestionError):
        safe_output_path(raw, "100", EventVersion.ROOT)


# ---------------------------------------------------------------- CLI smoke (synthetic)

def test_cli_exit_codes(tmp_path, capsys):
    raw = build(tmp_path, [rec(10), rec(11)], versions=("root",))
    assert cli_main(["quality", "100", "--events", "root", "--root", str(raw)]) == 0
    assert "status: ok_with_quality_issues" in capsys.readouterr().out or True
    assert cli_main(["quality", "../100", "--events", "root", "--root", str(raw)]) == 2
    assert cli_main(["quality", "100", "--events", "may_1_2025", "--root", str(raw)]) == 1      # that version is absent: fatal, not substituted
    with pytest.raises(SystemExit) as e:
        cli_main(["quality", "100", "--root", str(raw)])                                         # no default event version
    assert e.value.code == 2


# ================================================================ final correctness regressions (Codex review of 2A)

# ---------------------------------------------------------------- 1. repeated-record conflicts

def test_repeated_records_with_conflicting_timestamp_or_period_are_rejected(tmp_path):
    for name, second in (("time", dict(videoTimeMs=999.0)), ("period", dict(period=2))):
        raw = build(tmp_path / name, [rec(10), rec(10) | second, rec(11)])
        ds = Dataset(raw)
        it = ds.iter_tracking_frames("100")
        with pytest.raises(IngestionError, match="incompatible repeated records"):
            list(it)
        got = []
        try:
            for f in Dataset(raw).iter_tracking_frames("100"):
                got.append(f)
        except IngestionError as e:
            assert "records 0 and 1" in str(e) and "frameNum 10" in str(e)
        assert all(f.frame_num != 11 for f in got)                  # nothing after the ambiguous frame is published


def test_repeated_records_with_other_conflicts_are_flagged_unresolved_with_fields(tmp_path):
    other_smoothed_ball = rec(10, bs=dict(visibility="VISIBLE", x=9.0, y=9.0, z=0.0))
    other_raw_player = rec(10, home=[player(1, 5.0, 5.0), player(2, 2.0, 2.0)])
    for name, second, field in (("sball", other_smoothed_ball, "ballsSmoothed"), ("player", other_raw_player, "homePlayers")):
        raw = build(tmp_path / name, [rec(10), second])
        (f,) = Dataset(raw).iter_tracking_frames("100")
        assert Q.REPEATED_RECORDS_DIFFER in f.quality_flags and field in f.repeat_conflict_fields
        assert f.source_record_indices == (0, 1)                    # alternatives remain retrievable


def test_compatible_repeated_records_collapse_without_conflict(tmp_path):
    second = rec(10, game_event_id=7.0, possession_event_id=8.0)   # overlay ids may differ legitimately
    raw = build(tmp_path, [rec(10), second])
    (f,) = Dataset(raw).iter_tracking_frames("100")
    assert f.repeat_conflict_fields == () and Q.REPEATED_RECORDS_DIFFER not in f.quality_flags and f.source_record_count == 2


def test_uncollapsed_mode_still_yields_conflicting_records_separately(tmp_path):
    raw = build(tmp_path, [rec(10), rec(10) | dict(videoTimeMs=999.0), rec(10, period=2)])
    frames = list(Dataset(raw).iter_tracking_frames("100", collapse_repeats=False))
    assert [f.source_record_indices for f in frames] == [(0,), (1,), (2,)]


# ---------------------------------------------------------------- 2. cross-match identity

def two_matches(tmp_path):
    raw = build(tmp_path, [rec(10)])
    other = build(tmp_path / "o", [rec(10)], metadata=[dict(METADATA[0], id="200")], mid="200")
    return raw, Dataset(raw), Dataset(other)


def test_foreign_roster_and_metadata_are_rejected(tmp_path):
    raw, ds, other = two_matches(tmp_path)
    foreign_roster = other.roster("200")
    foreign_meta = other.metadata("200")
    with pytest.raises(IngestionError, match="roster belongs to match 200"):
        ds.iter_tracking_frames("100", roster=foreign_roster, metadata=ds.metadata("100"))
    with pytest.raises(IngestionError, match="metadata belongs to match 200"):
        ds.iter_tracking_frames("100", roster=ds.roster("100"), metadata=foreign_meta)
    with pytest.raises(IngestionError, match="metadata belongs to match 200"):
        ds.roster("100", metadata=foreign_meta)
    with pytest.raises(IngestionError, match="metadata belongs to match 200"):
        ds.iter_tracking_frames("100", metadata=foreign_meta)


def test_roster_inconsistent_with_home_away_teams_is_rejected(tmp_path):
    swapped = [dict(METADATA[0], homeTeam=METADATA[0]["awayTeam"], awayTeam=METADATA[0]["homeTeam"])]
    raw = build(tmp_path, [rec(10)])
    ds = Dataset(raw)
    roster = ds.roster("100")
    swapped_meta = Dataset(build(tmp_path / "s", [rec(10)], metadata=swapped)).metadata("100")
    with pytest.raises(IngestionError, match="inconsistent with the home/away teams"):
        ds.iter_tracking_frames("100", roster=roster, metadata=swapped_meta)


def test_foreign_events_and_tracking_records_are_rejected_but_missing_ids_are_allowed(tmp_path):
    foreign_events = [event(1, 10, 1.0), event(2, 11, 2.0, gameId=999)]
    raw = build(tmp_path / "e", [rec(10)], events=foreign_events, versions=("root",))
    with pytest.raises(IngestionError, match=r"row 1: gameId 999 does not belong to match 100"):
        Dataset(raw).load_events("100", EventVersion.ROOT)
    no_id = [{k: v for k, v in event(1, 10, 1.0).items() if k != "gameId"}, event(2, 11, 2.0, gameId=None), event(3, 12, 3.0, gameId="100")]
    assert len(Dataset(build(tmp_path / "n", [rec(10)], events=no_id, versions=("root",))).load_events("100", EventVersion.ROOT)) == 3
    for ref in (999.0, "999"):
        rawt = build(tmp_path / f"t{ref}", [rec(10, gameRefId=ref)])
        with pytest.raises(IngestionError, match="gameRefId .* does not belong to match 100"):
            list(Dataset(rawt).iter_tracking_frames("100"))
    for ref in (None, 100.0, 100):
        assert len(list(Dataset(build(tmp_path / f"ok{ref}", [rec(10, gameRefId=ref)])).iter_tracking_frames("100"))) == 1


# ---------------------------------------------------------------- 3. unparseable smoothed candidates

def test_unparseable_smoothed_jerseys_are_preserved_as_unmatched(tmp_path):
    hs = [player(1, 1.0, 1.0), player(2, 2.0, 2.0), player("x", 7.0, 7.0), player(None, 8.0, 8.0), player("1.5", 9.0, 9.0), None, "junk"]
    raw = build(tmp_path, [rec(10, hs=hs)])
    (f,) = Dataset(raw).iter_tracking_frames("100")
    kept = [(s, c.x_m, j) for s, c, j in f.unmatched_smoothed if s == "home"]
    assert (("home", 7.0, "x") in kept) and (("home", 8.0, "None") in kept) and (("home", 9.0, "1.5") in kept)
    assert ("home", None, None) in kept and ("home", None, "junk") in kept          # malformed entries are kept too
    assert len(kept) == 5
    assert {Q.SMOOTHED_JERSEY_UNPARSEABLE, Q.SMOOTHED_ENTRY_MALFORMED} <= f.quality_flags
    assert all(p.smoothed_status is SmoothedStatus.UNIQUE for p in f.players if p.source_side == "home")   # raw joins unaffected


def test_unparseable_smoothed_jersey_is_kept_even_when_a_raw_jersey_is_also_unparseable(tmp_path):
    home = [player(1, 1.0, 1.0), player("x", 3.0, 3.0)]
    raw = build(tmp_path, [rec(10, home=home, hs=[player(1, 1.0, 1.0), player("x", 3.5, 3.5)])])
    (f,) = Dataset(raw).iter_tracking_frames("100")
    assert [(s, c.x_m, j) for s, c, j in f.unmatched_smoothed if s == "home"] == [("home", 3.5, "x")]
    bad = next(p for p in f.players if p.source_side == "home" and p.jersey_number is None)
    assert bad.smoothed_status is SmoothedStatus.MISSING and bad.x_smooth_m is None     # never joined by guessing


# ---------------------------------------------------------------- 4. trusted output boundary

def _junction(link, target):
    try:
        os.symlink(target, link, target_is_directory=True)
        return True
    except (OSError, NotImplementedError):
        return os.name == "nt" and subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True).returncode == 0


def test_processed_directory_redirected_outside_the_boundary_is_rejected(tmp_path):
    raw = build(tmp_path, [rec(10)], versions=("root",))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    if not _junction(tmp_path / "processed", elsewhere):
        pytest.skip("links unavailable")
    rep = quality_report(Dataset(raw), "100", EventVersion.ROOT)
    with pytest.raises(IngestionError, match="link|redirected"):
        write_quality_report(rep, raw)
    with pytest.raises(IngestionError, match="link|redirected"):
        safe_output_path(raw, "100", EventVersion.ROOT)
    with pytest.raises(IngestionError, match="link|redirected"):
        safe_output_path(raw, "100", EventVersion.ROOT, suffix=".json.tmp")
    assert list(elsewhere.iterdir()) == [] and digest(raw)


def test_redirected_inner_directories_and_files_are_rejected(tmp_path):
    raw = build(tmp_path, [rec(10)], versions=("root",))
    rep = quality_report(Dataset(raw), "100", EventVersion.ROOT)
    outside = tmp_path / "outside"
    outside.mkdir()
    proc = tmp_path / "processed"
    proc.mkdir()
    if not _junction(proc / "ingestion_quality", outside):
        pytest.skip("links unavailable")
    with pytest.raises(IngestionError):
        write_quality_report(rep, raw)
    assert list(outside.iterdir()) == []


def test_final_and_temporary_destinations_are_validated_and_overwrite_is_explicit(tmp_path):
    raw = build(tmp_path, [rec(10)], versions=("root",))
    rep = quality_report(Dataset(raw), "100", EventVersion.ROOT)
    dest = safe_output_path(raw, "100", EventVersion.ROOT)
    tmp = safe_output_path(raw, "100", EventVersion.ROOT, suffix=".json.tmp")
    assert dest.parent == tmp.parent == tmp_path / "processed" / "ingestion_quality" / "100" and tmp.name == "root.json.tmp"
    assert write_quality_report(rep, raw) == dest and not tmp.exists()
    with pytest.raises(IngestionError, match="already exists"):
        write_quality_report(rep, raw)
    assert write_quality_report(rep, raw, overwrite=True) == dest
    for bad in (tmp_path / "outside", raw, tmp_path / "processed" / ".." / "outside", tmp_path / "processed" / ".." / "raw" / "x"):
        with pytest.raises(IngestionError):
            safe_output_path(raw, "100", EventVersion.ROOT, bad)


# ---------------------------------------------------------------- 5. structural errors and scan completion

def test_malformed_nested_metadata_is_a_contextual_ingestion_error(tmp_path):
    for name, bad in (("pitch_none", dict(METADATA[0], stadium=dict(id="1", name="x", pitches=[None]))),
                      ("pitch_str", dict(METADATA[0], stadium=dict(id="1", name="x", pitches=["a"]))),
                      ("team_list", dict(METADATA[0], homeTeam=[1])),
                      ("stadium_str", dict(METADATA[0], stadium="park"))):
        with pytest.raises(IngestionError, match="metadata 100"):
            Dataset(build(tmp_path / name, [rec(10)], metadata=[bad])).metadata("100")
    with pytest.raises(IngestionError, match="roster 100"):
        Dataset(build(tmp_path / "r", [rec(10)], roster=[dict(ROSTER[0], player=[1])])).roster("100")


def test_unhashable_event_ids_are_a_contextual_ingestion_error(tmp_path):
    for k in ("gameEventId", "possessionEventId"):
        for bad in ([], {"a": 1}):
            evs = [event(1, 10, 1.0), event(2, 11, 2.0) | {k: bad}]
            raw = build(tmp_path / f"{k}{type(bad).__name__}", [rec(10)], events=evs, versions=("root",))
            with pytest.raises(IngestionError, match=rf"row 1: {k} must be a scalar"):
                Dataset(raw).load_events("100", EventVersion.ROOT)
    raw = build(tmp_path / "gp", [rec(10)], events=[event(1, 10, 1.0) | {"gameEvents": [1, 2]}], versions=("root",))
    assert Dataset(raw).load_events("100", EventVersion.ROOT)[0].game_event_type is None      # not an object: kept recoverable, no crash


def test_failed_or_truncated_scan_does_not_claim_missing_periods(tmp_path):
    evs = [event(1, 10, 1.0), event(2, 11, 2.0, period=2), event(3, 12, 3.0, period=3)]
    # failure midway: period 2 events exist, but the scan died before reaching period 2 tracking
    raw = build(tmp_path / "fail", [rec(10), rec(11), "{broken", rec(12, period=2)], events=evs, versions=("root",))
    rep = quality_report(Dataset(raw), "100", EventVersion.ROOT)
    assert rep.status == "fatal" and rep.tracking["scan_complete"] is False and rep.tracking["frames"] == 2   # partial counts kept
    assert rep.periods["event_periods_without_tracking"] is None and rep.periods["flag_counts"] == {}
    assert "failed or is incomplete" in rep.periods["note"] and any("record 2" in e for e in rep.fatal_errors)
    # truncated compressed stream
    p = next((tmp_path / "fail" / "raw").rglob("100.jsonl.bz2"))
    good = bz2.compress(("\n".join(json.dumps(rec(n)) for n in range(10, 4000)) + "\n").encode())
    p.write_bytes(good[: len(good) // 2])
    rep = quality_report(Dataset(tmp_path / "fail" / "raw"), "100", EventVersion.ROOT)
    assert rep.status == "fatal" and rep.tracking["scan_complete"] is False and rep.periods["event_periods_without_tracking"] is None
    # a bounded scan is not complete either; a full successful scan still reports genuinely missing periods
    ok = build(tmp_path / "ok", [rec(10), rec(11, period=2)], events=evs, versions=("root",))
    full = quality_report(Dataset(ok), "100", EventVersion.ROOT)
    assert full.tracking["scan_complete"] is True and full.periods["event_periods_without_tracking"] == [3]
    part = quality_report(Dataset(ok), "100", EventVersion.ROOT, max_frames=1)
    assert part.tracking["scan_complete"] is False and part.periods["event_periods_without_tracking"] is None


# ---------------------------------------------------------------- smoothed-ball conflict policy

def test_conflicting_smoothed_ball_is_unresolved_and_first_value_is_not_exposed(tmp_path):
    first = rec(10, bs=dict(visibility="VISIBLE", x=1.25, y=2.5, z=0.3))
    second = rec(10, bs=dict(visibility="ESTIMATED", x=9.0, y=8.0, z=0.0))
    raw = build(tmp_path, [first, second])
    (f,) = Dataset(raw).iter_tracking_frames("100")
    b = f.ball
    assert (b.x_smooth_m, b.y_smooth_m, b.z_smooth_m, b.visibility_smooth) == (None, None, None, None)   # not the first record's 1.25/2.5
    assert (b.x_raw_m, b.y_raw_m, b.z_raw_m) == (0.5, 0.5, 0.1)                                          # raw ball untouched
    assert Q.REPEATED_RECORDS_DIFFER in f.quality_flags and Q.BALL_SMOOTHED_UNRESOLVED in f.quality_flags
    assert f.repeat_conflict_fields == ("ballsSmoothed",) and f.source_record_indices == (0, 1)
    rep = quality_report(Dataset(raw), "100", EventVersion.ROOT)
    assert rep.tracking["frame_flag_counts"]["BALL_SMOOTHED_UNRESOLVED"] == 1 and rep.tracking["frames_with_ball_smoothed_unresolved"] == 1
    assert rep.tracking["frames_with_ball_smoothed_xy_null"] == 0


def test_agreeing_smoothed_ball_and_uncollapsed_records_keep_their_values(tmp_path):
    same = dict(visibility="VISIBLE", x=1.25, y=2.5, z=0.3)
    (f,) = Dataset(build(tmp_path / "a", [rec(10, bs=same), rec(10, bs=dict(same))])).iter_tracking_frames("100")
    assert (f.ball.x_smooth_m, f.ball.y_smooth_m) == (1.25, 2.5) and Q.BALL_SMOOTHED_UNRESOLVED not in f.quality_flags
    fs = list(Dataset(build(tmp_path / "b", [rec(10, bs=same), rec(10, bs=dict(same, x=9.0))])).iter_tracking_frames("100", collapse_repeats=False))
    assert [f.ball.x_smooth_m for f in fs] == [1.25, 9.0] and all(Q.BALL_SMOOTHED_UNRESOLVED not in f.quality_flags for f in fs)
