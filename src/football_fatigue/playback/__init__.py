"""Tactical board playback and event navigation modules (Z2/Z3).

This package provides:
- adapter.py    : converts PreparedFrame objects into lightweight renderable FrameState dicts
- controller.py : single authoritative playback state and event-driven navigation controller
- server.py     : stdlib HTTP server streaming match data to the HTML/JS tactical board
"""

from football_fatigue.playback.adapter import (
    build_event_index,
    build_roster_lookup,
    build_roster_names,
    frame_to_state,
)
from football_fatigue.playback.controller import (
    EventItem,
    NavigationMode,
    PlaybackController,
    PlaybackState,
)

__all__ = [
    "EventItem",
    "NavigationMode",
    "PlaybackController",
    "PlaybackState",
    "build_event_index",
    "build_roster_lookup",
    "build_roster_names",
    "frame_to_state",
]
