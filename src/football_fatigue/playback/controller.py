"""Playback Controller and Event Navigator (Z2/Z3).

Provides the authoritative playback state and event-driven navigation logic
over MatchData frames and synchronized events.

Guarantees:
- Single authoritative state (frame index, frame_num, timestamp, period, active event).
- Strict separation of observed/temporal event alignments from physical ground truth.
- Unmatched events are preserved, queryable, and never fabricate frame positions.
- Event navigation is deterministic, chronological, and supports category/type filtering.
- Seamless mode switching: timestamp seek -> event jump -> chronological resume.
- Period boundaries and tracking gaps are explicitly surfaced, never interpolated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from football_fatigue.match_data import MatchData, PreparedFrame, CoordinatePolicy
from football_fatigue.playback.adapter import (
    build_event_index,
    build_roster_names,
    frame_to_state,
)


class NavigationMode(str, Enum):
    TIMESTAMP = "timestamp"
    EVENT = "event"


@dataclass(frozen=True)
class EventItem:
    """Standardized representation of an event for playback navigation."""
    event_row_id: str
    game_event_type: str | None
    possession_event_type: str | None
    canonical_period: int
    event_time_s: float | None
    matched: bool
    frame_num: int | None = None
    residual_s: float | None = None
    quality_label: str = "unmatched"
    reason: str | None = None
    source_team_id: int | None = None
    source_player_id: int | None = None

    @property
    def display_type(self) -> str:
        return self.possession_event_type or self.game_event_type or "UNKNOWN"

    def matches_filter(self, filter_type: str | None) -> bool:
        """Check if this event matches the requested category or event type."""
        if not filter_type or filter_type.upper() in ("ALL", "*"):
            return True
        f = filter_type.upper()
        return (
            (self.game_event_type is not None and self.game_event_type.upper() == f)
            or (self.possession_event_type is not None and self.possession_event_type.upper() == f)
        )


@dataclass
class PlaybackState:
    """The single authoritative playback state."""
    frame_index: int = 0
    frame_num: int = 0
    video_time_s: float = 0.0
    canonical_period: int = 1
    period_usable: bool = True
    has_gap: bool = False
    is_playing: bool = False
    playback_speed: float = 1.0
    mode: NavigationMode = NavigationMode.TIMESTAMP
    selected_event_id: str | None = None
    active_filter: str | None = None


class PlaybackController:
    """Controls match playback and event-driven navigation.

    Consumes prepared frames and event alignments from the existing
    MatchData and synchronization interfaces.
    """

    def __init__(
        self,
        frames: Sequence[PreparedFrame | dict[str, Any]],
        events: Sequence[EventItem | dict[str, Any]],
        fps: float = 29.97,
    ) -> None:
        if not frames:
            raise ValueError("PlaybackController requires at least one frame.")

        self._fps = float(fps)
        self._frames: list[dict[str, Any]] = []
        self._frame_num_to_index: dict[int, int] = {}

        # Normalize frames into standardized dictionaries
        for idx, f in enumerate(frames):
            if isinstance(f, dict):
                fd = f
            elif isinstance(f, PreparedFrame):
                fd = frame_to_state(f)
            else:
                raise TypeError(f"Unsupported frame type: {type(f)}")
            self._frames.append(fd)
            # Map first occurrence of frame_num to index
            f_num = fd["frame_num"]
            if f_num not in self._frame_num_to_index:
                self._frame_num_to_index[f_num] = idx

        # Normalize events
        self._events: list[EventItem] = []
        self._events_by_id: dict[str, EventItem] = {}
        for ev in events:
            if isinstance(ev, EventItem):
                item = ev
            elif isinstance(ev, dict):
                item = EventItem(
                    event_row_id=str(ev.get("event_row_id")),
                    game_event_type=ev.get("game_event_type"),
                    possession_event_type=ev.get("possession_event_type"),
                    canonical_period=int(ev.get("canonical_period", 1)),
                    event_time_s=ev.get("event_time_s"),
                    matched=bool(ev.get("matched", False)),
                    frame_num=ev.get("frame_num"),
                    residual_s=ev.get("residual_s"),
                    quality_label=ev.get("quality_label", "unmatched"),
                    reason=ev.get("reason"),
                    source_team_id=ev.get("source_team_id"),
                    source_player_id=ev.get("source_player_id"),
                )
            else:
                raise TypeError(f"Unsupported event type: {type(ev)}")
            self._events.append(item)
            self._events_by_id[item.event_row_id] = item

        # Initial state at first frame
        first_frame = self._frames[0]
        self._state = PlaybackState(
            frame_index=0,
            frame_num=first_frame["frame_num"],
            video_time_s=first_frame["video_time_s"],
            canonical_period=first_frame["canonical_period"],
            period_usable=first_frame["period_usable"],
            has_gap=first_frame.get("has_gap", False),
            is_playing=False,
            playback_speed=1.0,
            mode=NavigationMode.TIMESTAMP,
            selected_event_id=None,
            active_filter=None,
        )

    @classmethod
    def from_match_data(
        cls,
        match_data: MatchData,
        max_frames: int | None = 1500,
        policy: CoordinatePolicy = CoordinatePolicy.SMOOTHED_THEN_RAW,
    ) -> PlaybackController:
        """Factory method constructing a controller directly from MatchData."""
        roster_names = build_roster_names(match_data.roster)
        sync = match_data.synchronize(spatial=False)
        event_index = build_event_index(sync)

        frames = []
        for pf in match_data.frames(attach_events=True, max_frames=max_frames):
            frames.append(frame_to_state(pf, roster_names=roster_names))

        return cls(frames=frames, events=event_index, fps=match_data.metadata.fps)

    # ---------------------------------------------------------------- Properties

    @property
    def state(self) -> PlaybackState:
        return self._state

    @property
    def total_frames(self) -> int:
        return len(self._frames)

    @property
    def current_frame_state(self) -> dict[str, Any]:
        return self._frames[self._state.frame_index]

    @property
    def all_events(self) -> list[EventItem]:
        return list(self._events)

    @property
    def fps(self) -> float:
        return self._fps

    # ---------------------------------------------------------------- Core State Sync

    def _sync_state_to_index(self, new_idx: int) -> None:
        """Internal helper to update authoritative state when frame index changes."""
        idx = max(0, min(new_idx, len(self._frames) - 1))
        f = self._frames[idx]
        self._state.frame_index = idx
        self._state.frame_num = f["frame_num"]
        self._state.video_time_s = f["video_time_s"]
        self._state.canonical_period = f["canonical_period"]
        self._state.period_usable = f["period_usable"]
        self._state.has_gap = f.get("has_gap", False)

    # ---------------------------------------------------------------- Timestamp Navigation

    def play(self) -> None:
        """Start or resume playback."""
        if self._state.frame_index >= len(self._frames) - 1:
            self._sync_state_to_index(0)
        self._state.is_playing = True

    def pause(self) -> None:
        """Pause playback."""
        self._state.is_playing = False

    def toggle_play(self) -> bool:
        """Toggle between play and pause."""
        if self._state.is_playing:
            self.pause()
        else:
            self.play()
        return self._state.is_playing

    def set_speed(self, speed: float) -> None:
        """Set playback speed multiplier (e.g. 0.5, 1.0, 2.0)."""
        if speed <= 0:
            raise ValueError("Playback speed must be positive.")
        self._state.playback_speed = float(speed)

    def restart(self) -> None:
        """Rewind to the first frame and pause."""
        self.pause()
        self._sync_state_to_index(0)
        self._state.selected_event_id = None
        self._state.mode = NavigationMode.TIMESTAMP

    def step(self, delta_frames: int = 1) -> None:
        """Advance or step backward by delta_frames.

        Preserves chronological position and switches mode to TIMESTAMP.
        """
        self._state.mode = NavigationMode.TIMESTAMP
        self._sync_state_to_index(self._state.frame_index + delta_frames)

    def seek_frame_num(self, frame_num: int) -> bool:
        """Seek to a specific frame number.

        Returns True if found, False if the frame number does not exist.
        """
        idx = self._frame_num_to_index.get(frame_num)
        if idx is None:
            return False
        self._state.mode = NavigationMode.TIMESTAMP
        self._sync_state_to_index(idx)
        return True

    def seek_time(self, timestamp_s: float) -> None:
        """Seek to the nearest frame corresponding to timestamp_s."""
        self._state.mode = NavigationMode.TIMESTAMP
        if timestamp_s <= self._frames[0]["video_time_s"]:
            self._sync_state_to_index(0)
            return
        if timestamp_s >= self._frames[-1]["video_time_s"]:
            self._sync_state_to_index(len(self._frames) - 1)
            return

        # Binary search for nearest frame by video_time_s
        low, high = 0, len(self._frames) - 1
        best_idx = 0
        min_diff = float("inf")

        while low <= high:
            mid = (low + high) // 2
            t = self._frames[mid]["video_time_s"]
            diff = abs(t - timestamp_s)
            if diff < min_diff:
                min_diff = diff
                best_idx = mid
            if t < timestamp_s:
                low = mid + 1
            elif t > timestamp_s:
                high = mid - 1
            else:
                best_idx = mid
                break

        self._sync_state_to_index(best_idx)

    # ---------------------------------------------------------------- Event Navigation

    def set_event_filter(self, filter_type: str | None) -> None:
        """Set the active event filter category (e.g. 'PA', 'SH', 'FO', 'G', 'ALL')."""
        self._state.active_filter = None if filter_type in (None, "ALL", "*") else filter_type

    def get_filtered_events(self, include_unmatched: bool = True) -> list[EventItem]:
        """Return all events matching the active filter in chronological order."""
        filt = self._state.active_filter
        out = []
        for ev in self._events:
            if not ev.matches_filter(filt):
                continue
            if not include_unmatched and not ev.matched:
                continue
            out.append(ev)
        return out

    def get_unmatched_events(self) -> list[EventItem]:
        """Return unmatched events matching the active filter, with reasons."""
        filt = self._state.active_filter
        return [ev for ev in self._events if not ev.matched and ev.matches_filter(filt)]

    def select_event(self, event_row_id: str) -> bool:
        """Jump to the frame aligned with the specified event.

        Requirements:
        - If the event is matched and its aligned frame is loaded, jump to it.
        - If the event is unmatched, DO NOT fabricate any frame or position.
          Leave playback position unchanged and return False.
        """
        ev = self._events_by_id.get(str(event_row_id))
        if ev is None:
            return False

        if not ev.matched or ev.frame_num is None:
            # Unmatched event: do not fabricate frame; update selected event for UI
            self._state.selected_event_id = ev.event_row_id
            return False

        frame_idx = self._frame_num_to_index.get(ev.frame_num)
        if frame_idx is None:
            # Frame is outside the bounded segment loaded in memory
            self._state.selected_event_id = ev.event_row_id
            return False

        # Valid temporal match found in loaded frames
        self._state.mode = NavigationMode.EVENT
        self._state.selected_event_id = ev.event_row_id
        self._sync_state_to_index(frame_idx)
        return True

    def next_event(self) -> EventItem | None:
        """Navigate to the next matched event in chronological order.

        Deterministic tie-breaking:
        - If multiple events are associated with the current frame, advances to the next
          event in the event index sequence.
        - Otherwise, advances to the earliest event with event_time_s > current time (or
          frame_num > current frame_num).
        """
        candidates = self.get_filtered_events(include_unmatched=False)
        if not candidates:
            return None

        cur_time = self._state.video_time_s
        cur_frame = self._state.frame_num
        cur_ev_id = self._state.selected_event_id

        # If an event is currently selected, check if there is a subsequent event on the same frame
        if cur_ev_id:
            try:
                cur_pos = next(i for i, e in enumerate(candidates) if e.event_row_id == cur_ev_id)
                # Check next event in candidates list
                if cur_pos + 1 < len(candidates):
                    next_ev = candidates[cur_pos + 1]
                    if self.select_event(next_ev.event_row_id):
                        return next_ev
            except StopIteration:
                pass

        # Otherwise find first event with frame_num > current frame or event_time_s > cur_time + 1e-4
        for ev in candidates:
            if ev.frame_num is not None and ev.frame_num > cur_frame:
                if self.select_event(ev.event_row_id):
                    return ev
            elif ev.event_time_s is not None and ev.event_time_s > cur_time + 1e-4:
                if self.select_event(ev.event_row_id):
                    return ev

        return None

    def prev_event(self) -> EventItem | None:
        """Navigate to the previous matched event in reverse chronological order."""
        candidates = self.get_filtered_events(include_unmatched=False)
        if not candidates:
            return None

        cur_frame = self._state.frame_num
        cur_ev_id = self._state.selected_event_id

        # If an event is currently selected, check previous event in candidates
        if cur_ev_id:
            try:
                cur_pos = next(i for i, e in enumerate(candidates) if e.event_row_id == cur_ev_id)
                if cur_pos > 0:
                    prev_ev = candidates[cur_pos - 1]
                    if self.select_event(prev_ev.event_row_id):
                        return prev_ev
            except StopIteration:
                pass

        # Otherwise find the last event with frame_num < current frame
        for ev in reversed(candidates):
            if ev.frame_num is not None and ev.frame_num < cur_frame:
                if self.select_event(ev.event_row_id):
                    return ev

        return None
