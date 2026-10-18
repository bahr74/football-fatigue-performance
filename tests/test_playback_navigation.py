"""Acceptance Tests for Event-Driven Navigation and Playback Controller (Z2/Z3).

Directly validates the 8 required acceptance criteria from TASK 3 Section 5.D:
1. Next Event and Previous Event select the correct event in chronological order.
2. Event-type filtering skips events outside the selected category.
3. Selecting an event moves playback to its associated frame when a valid temporal match exists.
4. Multiple events associated with the same frame are handled consistently.
5. Unmatched events remain visible and cannot trigger fabricated frame positions.
6. Switching between timestamp and event modes preserves a consistent playback position.
7. Resuming playback after an event jump advances chronologically without state desynchronization.
8. Period boundaries, tracking gaps, and first/last-event boundaries are handled correctly.
"""

import pytest

from football_fatigue.playback.controller import (
    EventItem,
    NavigationMode,
    PlaybackController,
)


@pytest.fixture
def sample_frames():
    """Create a deterministic synthetic sequence of 10 frames across two periods with a gap."""
    frames = []
    # Period 1: frames 100 to 104 (t=0.0 to t=0.133s)
    for i in range(5):
        frames.append({
            "frame_num": 100 + i,
            "video_time_s": i * (1.0 / 30.0),
            "canonical_period": 1,
            "source_period": 1,
            "period_usable": True,
            "has_gap": False,
            "players": [],
            "ball": None,
            "events": [],
        })
    # Gap: skip frames 105-109
    # Frame 110 has a gap
    frames.append({
        "frame_num": 110,
        "video_time_s": 0.333,
        "canonical_period": 1,
        "source_period": 1,
        "period_usable": True,
        "has_gap": True,
        "players": [],
        "ball": None,
        "events": [],
    })
    # Period 2: frames 200 to 203
    for i in range(4):
        frames.append({
            "frame_num": 200 + i,
            "video_time_s": 100.0 + i * (1.0 / 30.0),
            "canonical_period": 2,
            "source_period": 2,
            "period_usable": True,
            "has_gap": (i == 0),
            "players": [],
            "ball": None,
            "events": [],
        })
    return frames


@pytest.fixture
def sample_events():
    """Create a sequence of matched and unmatched events, including multiple events on one frame."""
    return [
        # Frame 100 has TWO events: a PASS and a SHOT
        EventItem(
            event_row_id="ev-1",
            game_event_type="PASS",
            possession_event_type="PA",
            canonical_period=1,
            event_time_s=0.001,
            matched=True,
            frame_num=100,
            residual_s=0.001,
        ),
        EventItem(
            event_row_id="ev-2",
            game_event_type="SHOT",
            possession_event_type="SH",
            canonical_period=1,
            event_time_s=0.002,
            matched=True,
            frame_num=100,
            residual_s=0.002,
        ),
        # Frame 103 has a FOUL
        EventItem(
            event_row_id="ev-3",
            game_event_type="FOUL",
            possession_event_type="FO",
            canonical_period=1,
            event_time_s=0.100,
            matched=True,
            frame_num=103,
            residual_s=0.000,
        ),
        # Unmatched event in period 1 (e.g. outside tracking)
        EventItem(
            event_row_id="ev-unmatched-p1",
            game_event_type="PASS",
            possession_event_type="PA",
            canonical_period=1,
            event_time_s=50.0,
            matched=False,
            frame_num=None,
            reason="outside_tracking_window",
        ),
        # Frame 201 in period 2 has a GOAL
        EventItem(
            event_row_id="ev-4",
            game_event_type="GOAL",
            possession_event_type="G",
            canonical_period=2,
            event_time_s=100.033,
            matched=True,
            frame_num=201,
            residual_s=0.000,
        ),
    ]


# ---------------------------------------------------------------- Test 1: Next / Prev chronological order
def test_1_next_and_previous_event_select_in_chronological_order(sample_frames, sample_events):
    ctrl = PlaybackController(frames=sample_frames, events=sample_events, fps=30.0)

    # Initially at frame 100
    assert ctrl.state.frame_num == 100

    # Advance to next event -> ev-1
    ev1 = ctrl.next_event()
    assert ev1 is not None and ev1.event_row_id == "ev-1"
    assert ctrl.state.frame_num == 100

    # Next event -> ev-2 (second event on frame 100)
    ev2 = ctrl.next_event()
    assert ev2 is not None and ev2.event_row_id == "ev-2"

    # Next event -> ev-3 (frame 103)
    ev3 = ctrl.next_event()
    assert ev3 is not None and ev3.event_row_id == "ev-3"
    assert ctrl.state.frame_num == 103

    # Next event -> ev-4 (frame 201)
    ev4 = ctrl.next_event()
    assert ev4 is not None and ev4.event_row_id == "ev-4"
    assert ctrl.state.frame_num == 201

    # Prev event -> back to ev-3 (frame 103)
    p_ev3 = ctrl.prev_event()
    assert p_ev3 is not None and p_ev3.event_row_id == "ev-3"
    assert ctrl.state.frame_num == 103


# ---------------------------------------------------------------- Test 2: Event-type filtering
def test_2_event_type_filtering_skips_events_outside_category(sample_frames, sample_events):
    ctrl = PlaybackController(frames=sample_frames, events=sample_events, fps=30.0)

    # Filter only FOUL (FO)
    ctrl.set_event_filter("FO")
    filtered = ctrl.get_filtered_events()
    assert len(filtered) == 1
    assert filtered[0].event_row_id == "ev-3"

    # next_event should jump directly to ev-3, skipping ev-1 and ev-2
    ev = ctrl.next_event()
    assert ev is not None
    assert ev.event_row_id == "ev-3"
    assert ctrl.state.frame_num == 103

    # Calling next_event again should return None (no other fouls)
    assert ctrl.next_event() is None
    assert ctrl.state.frame_num == 103  # state unchanged


# ---------------------------------------------------------------- Test 3: Select event moves playback to associated frame
def test_3_selecting_event_moves_playback_to_associated_frame(sample_frames, sample_events):
    ctrl = PlaybackController(frames=sample_frames, events=sample_events, fps=30.0)

    ok = ctrl.select_event("ev-4")
    assert ok is True
    assert ctrl.state.frame_num == 201
    assert ctrl.state.canonical_period == 2
    assert ctrl.state.selected_event_id == "ev-4"
    assert ctrl.state.mode == NavigationMode.EVENT


# ---------------------------------------------------------------- Test 4: Multiple events associated with the same frame
def test_4_multiple_events_on_same_frame_handled_consistently(sample_frames, sample_events):
    ctrl = PlaybackController(frames=sample_frames, events=sample_events, fps=30.0)

    # Select ev-1 on frame 100
    assert ctrl.select_event("ev-1") is True
    assert ctrl.state.frame_num == 100
    assert ctrl.state.selected_event_id == "ev-1"

    # Next event from ev-1 is ev-2, still on frame 100
    ev2 = ctrl.next_event()
    assert ev2 is not None and ev2.event_row_id == "ev-2"
    assert ctrl.state.frame_num == 100
    assert ctrl.state.selected_event_id == "ev-2"

    # Next event after ev-2 is ev-3 on frame 103
    ev3 = ctrl.next_event()
    assert ev3 is not None and ev3.event_row_id == "ev-3"
    assert ctrl.state.frame_num == 103


# ---------------------------------------------------------------- Test 5: Unmatched events remain visible and never fabricate frames
def test_5_unmatched_events_remain_visible_and_cannot_trigger_fabricated_positions(sample_frames, sample_events):
    ctrl = PlaybackController(frames=sample_frames, events=sample_events, fps=30.0)

    # Unmatched events exist and are discoverable
    unmatched = ctrl.get_unmatched_events()
    assert len(unmatched) == 1
    assert unmatched[0].event_row_id == "ev-unmatched-p1"
    assert unmatched[0].reason == "outside_tracking_window"

    # Move to frame 102
    ctrl.seek_frame_num(102)
    assert ctrl.state.frame_num == 102

    # Attempting to select unmatched event MUST return False and NOT change frame
    success = ctrl.select_event("ev-unmatched-p1")
    assert success is False
    assert ctrl.state.frame_num == 102  # Not fabricated or changed!
    assert ctrl.state.selected_event_id == "ev-unmatched-p1"  # preserved for UI display


# ---------------------------------------------------------------- Test 6: Mode switching preserves playback position
def test_6_switching_modes_preserves_consistent_playback_position(sample_frames, sample_events):
    ctrl = PlaybackController(frames=sample_frames, events=sample_events, fps=30.0)

    # Event mode jump to ev-3 (frame 103)
    ctrl.select_event("ev-3")
    assert ctrl.state.mode == NavigationMode.EVENT
    assert ctrl.state.frame_num == 103

    # Switch to timestamp mode by seeking or stepping
    ctrl.step(0)  # zero delta step
    assert ctrl.state.mode == NavigationMode.TIMESTAMP
    assert ctrl.state.frame_num == 103  # preserved

    # Seeking time keeps timestamp mode
    ctrl.seek_time(ctrl.state.video_time_s)
    assert ctrl.state.mode == NavigationMode.TIMESTAMP
    assert ctrl.state.frame_num == 103


# ---------------------------------------------------------------- Test 7: Resuming playback after event jump advances chronologically
def test_7_resuming_playback_after_event_jump_advances_chronologically(sample_frames, sample_events):
    ctrl = PlaybackController(frames=sample_frames, events=sample_events, fps=30.0)

    # Jump to event ev-3 (frame 103, index 3)
    ctrl.select_event("ev-3")
    assert ctrl.state.frame_index == 3
    assert ctrl.state.frame_num == 103

    # Resuming / stepping forward advances to next chronological frame without desynchronization
    ctrl.step(1)
    assert ctrl.state.frame_index == 4
    assert ctrl.state.frame_num == 104
    assert ctrl.state.mode == NavigationMode.TIMESTAMP


# ---------------------------------------------------------------- Test 8: Boundaries (periods, gaps, first/last)
def test_8_boundaries_and_gaps_handled_correctly(sample_frames, sample_events):
    ctrl = PlaybackController(frames=sample_frames, events=sample_events, fps=30.0)

    # First event boundary: calling prev_event at start returns None and does not crash
    ctrl.restart()
    assert ctrl.prev_event() is None
    assert ctrl.state.frame_num == 100

    # Step backward at frame 0 clamps cleanly at frame 0
    ctrl.step(-1)
    assert ctrl.state.frame_index == 0
    assert ctrl.state.frame_num == 100

    # Frame gap handling: step through frame 104 -> frame 110 (which has gap)
    ctrl.seek_frame_num(104)
    assert ctrl.state.has_gap is False
    ctrl.step(1)
    assert ctrl.state.frame_num == 110
    assert ctrl.state.has_gap is True  # gap surfaced

    # Period boundary: step across frame 110 (period 1) -> frame 200 (period 2)
    assert ctrl.state.canonical_period == 1
    ctrl.step(1)
    assert ctrl.state.frame_num == 200
    assert ctrl.state.canonical_period == 2

    # Last frame boundary: stepping past the end clamps at last frame
    for _ in range(10):
        ctrl.step(1)
    assert ctrl.state.frame_index == len(sample_frames) - 1
    assert ctrl.state.frame_num == 203
    assert ctrl.next_event() is None
