"""Tests for Playback Data Adapter (Z2).

Uses synthetic fixtures from test_ingestion.py to verify that PreparedFrames
are transformed into renderable FrameState dicts strictly according to scientific contracts:
- Unusable coordinates (missing/ambiguous) are omitted; never drawn at (0,0).
- Estimated positions carry estimated: True.
- Ball heights and usability are preserved.
- Temporal events carry their alignment quality labels, never verified.
- Gaps and period boundaries are preserved.
"""

import importlib.util
from pathlib import Path
import sys
import pytest

from football_fatigue.ingestion import EventVersion, QualityFlag as Q
from football_fatigue.match_data import CoordinatePolicy, MatchData
from football_fatigue.playback.adapter import (
    build_event_index,
    build_roster_lookup,
    build_roster_names,
    frame_to_state,
)
from football_fatigue.playback.server import (
    get_matches_catalog,
    load_match_data,
)

_spec = importlib.util.spec_from_file_location("ingestion_test_helpers", Path(__file__).with_name("test_ingestion.py"))
H = importlib.util.module_from_spec(_spec)
sys.modules["ingestion_test_helpers"] = H
_spec.loader.exec_module(H)


def test_adapter_omits_unusable_player_positions(tmp_path):
    # home has 1 valid player and 1 player with None coordinates
    home_players = [
        H.player(1, 10.0, 20.0),
        H.player(2, None, None),
    ]
    rec = H.rec(10, home=home_players, away=[])
    raw = H.build(tmp_path, [rec])
    md = MatchData.open(raw, "100", EventVersion.ROOT)
    pf = next(iter(md.frames()))

    state = frame_to_state(pf)
    assert len(state["players"]) == 1
    assert state["players"][0]["jersey"] == 1
    assert state["players"][0]["x"] == pytest.approx(10.0)
    assert state["players"][0]["y"] == pytest.approx(20.0)
    assert not any(p["jersey"] == 2 for p in state["players"])


def test_adapter_flags_estimated_positions(tmp_path):
    # Player 1 is marked ESTIMATED, Player 2 is VISIBLE
    home_players = [
        dict(jerseyNum="1", confidence="HIGH", visibility="ESTIMATED", x=-5.0, y=15.0),
        H.player(2, 0.0, 0.0),
    ]
    rec = H.rec(15, home=home_players, away=[])
    raw = H.build(tmp_path, [rec])
    md = MatchData.open(raw, "100", EventVersion.ROOT)
    pf = next(iter(md.frames()))

    state = frame_to_state(pf)
    p1 = next(p for p in state["players"] if p["jersey"] == 1)
    p2 = next(p for p in state["players"] if p["jersey"] == 2)

    assert p1["estimated"] is True
    assert p2["estimated"] is False


def test_adapter_preserves_ball_coordinates_and_height(tmp_path):
    ball_entry = [dict(visibility="VISIBLE", x=12.5, y=-8.0, z=1.85)]
    ball_smoothed = dict(visibility="VISIBLE", x=12.5, y=-8.0, z=1.85)
    rec = H.rec(20, balls=ball_entry, bs=ball_smoothed)
    raw = H.build(tmp_path, [rec])
    md = MatchData.open(raw, "100", EventVersion.ROOT)
    pf = next(iter(md.frames()))

    state = frame_to_state(pf)
    assert state["ball"] is not None
    assert state["ball"]["x"] == pytest.approx(12.5)
    assert state["ball"]["y"] == pytest.approx(-8.0)
    assert state["ball"]["z"] == pytest.approx(1.85)


def test_adapter_omits_unusable_ball(tmp_path):
    rec = H.rec(25, balls=[], bs=None)
    raw = H.build(tmp_path, [rec])
    md = MatchData.open(raw, "100", EventVersion.ROOT)
    pf = next(iter(md.frames()))

    state = frame_to_state(pf)
    assert state["ball"] is None


def test_adapter_surfaces_frame_gap(tmp_path):
    # Frames 10 and 15 (gap of 4 missing frames)
    records = [H.rec(10), H.rec(15)]
    raw = H.build(tmp_path, records)
    md = MatchData.open(raw, "100", EventVersion.ROOT)
    frames = list(md.frames())

    s0 = frame_to_state(frames[0])
    s1 = frame_to_state(frames[1])
    assert s0["has_gap"] is False
    assert s1["has_gap"] is True


def test_adapter_event_index_contains_matched_and_unmatched(tmp_path):
    records = [H.rec(10), H.rec(11)]
    # Event 1 matches frame 10 (t=10/30=0.333s); Event 2 is in period 2 (no tracking frames)
    events = [
        H.event(1, 101, 10 / 30, period=1, gtype="PASS", ptype="PA"),
        H.event(2, 102, 50.0, period=2, gtype="SHOT", ptype="SH"),
    ]
    raw = H.build(tmp_path, records, events=events)
    md = MatchData.open(raw, "100", EventVersion.ROOT)
    sync = md.synchronize()

    ev_index = build_event_index(sync, md.events())
    assert len(ev_index) == 2
    matched = [e for e in ev_index if e["matched"]]
    unmatched = [e for e in ev_index if not e["matched"]]

    assert len(matched) == 1
    assert matched[0]["frame_num"] == 10
    assert matched[0]["quality_label"] == "temporal_only"
    assert matched[0]["game_event_type"] == "PASS"
    assert matched[0]["possession_event_type"] == "PA"

    assert len(unmatched) == 1
    assert unmatched[0]["reason"] == "no_tracking_for_period"
    assert unmatched[0]["game_event_type"] == "SHOT"


def test_adapter_event_index_enriches_player_and_team(tmp_path):
    records = [H.rec(10)]
    events = [H.event(1, 101, 10 / 30, period=1, gtype="PASS", ptype="PA")]
    raw = H.build(tmp_path, records, events=events)
    md = MatchData.open(raw, "100", EventVersion.ROOT)
    sync = md.synchronize()

    roster_lookup = build_roster_lookup(md.roster)
    ev_index = build_event_index(sync, md.events(), roster_lookup=roster_lookup, meta=md.metadata)
    assert len(ev_index) == 1
    e0 = ev_index[0]
    assert e0["player_name"] is not None  # enriched from roster
    assert e0["team_name"] == "Home"


def test_get_matches_catalog(tmp_path):
    records = [H.rec(10)]
    raw = H.build(tmp_path, records)
    catalog = get_matches_catalog(str(raw))
    assert len(catalog) == 1
    assert catalog[0]["match_id"] == "100"
    assert catalog[0]["home_team"] == "Home"
    assert catalog[0]["away_team"] == "Away"
    assert "stage" in catalog[0]


def test_load_match_data_scope_half1(tmp_path):
    records = [H.rec(10, period=1), H.rec(11, period=1), H.rec(20, period=2)]
    raw = H.build(tmp_path, records)
    data = load_match_data(str(raw), "100", scope="half1")
    # Only period 1 frames should be loaded
    assert len(data["frames"]) == 2
    assert all(f["canonical_period"] == 1 for f in data["frames"])
    assert data["scope"] == "half1"

