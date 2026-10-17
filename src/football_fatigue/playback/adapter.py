"""Playback Data Adapter (Z2).

Converts ``PreparedFrame`` objects from the MatchData API into lightweight,
JSON-serialisable ``FrameState`` dicts that the tactical board renderer
consumes.  No positional data is invented, fabricated, or interpolated.
Scientific invariants from the MatchData layer are preserved:

- Missing or ambiguous coordinates are omitted from the render list entirely.
- ``position_usable`` is the only gate for rendering; it is never overridden.
- ``identity_usable`` gates whether a player label carries a confirmed
  player_id / name; falling back to jersey number with an explicit flag.
- Estimated positions are forwarded with ``estimated: true`` so the renderer
  can distinguish them visually.
- Period, frame_num, and video_time_s are copied verbatim; the adapter adds
  no clock arithmetic of its own.
- Events attached to a frame (from ``attach_events=True``) are forwarded with
  their alignment quality label; they are never re-labelled as verified.
- Frame gaps (``FRAME_GAP`` quality flag) and period changes are surfaced so
  the controller can decide how to display them.
"""

from __future__ import annotations

from typing import Any, Sequence

from football_fatigue.ingestion.models import QualityFlag as Q
from football_fatigue.match_data import PreparedFrame


def frame_to_state(
    pf: PreparedFrame,
    roster_names: dict[int, str] | None = None,
    event_lookup: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert one PreparedFrame to a JSON-serialisable FrameState dict.

    Parameters
    ----------
    pf:
        A prepared frame from ``MatchData.frames()``.
    roster_names:
        Optional ``{player_id: display_name}`` lookup built from the match
        roster. When supplied and ``p.identity_usable`` is True, the player
        entry receives a ``name`` field.
    event_lookup:
        Optional ``{event_row_id: EventRow}`` lookup to enrich attached
        events with event types and player/team identifiers.

    Returns
    -------
    dict with keys:
        frame_num, video_time_s, canonical_period, source_period,
        period_usable, has_gap, players, ball, events
    """
    has_gap = Q.FRAME_GAP in pf.quality_flags

    players: list[dict[str, Any]] = []
    for p in pf.players:
        if not p.position_usable:
            # Missing or ambiguous: omit entirely; never draw at (0,0).
            continue
        c = p.selected
        entry: dict[str, Any] = {
            "side": p.source_side,          # 'home' or 'away'
            "jersey": p.jersey_number,
            "x": c.x,
            "y": c.y,
            "coord_source": c.source,        # 'raw', 'smoothed', 'raw_fallback'
            "estimated": "PLAYER_ESTIMATED" in p.caveats,
            "jersey_confidence_low": "JERSEY_CONFIDENCE_LOW" in p.caveats,
            "identity_usable": p.identity_usable,
        }
        if p.identity_usable:
            entry["player_id"] = p.player_id
            if roster_names and p.player_id and p.player_id in roster_names:
                entry["name"] = roster_names[p.player_id]
        players.append(entry)

    ball: dict[str, Any] | None = None
    if pf.ball.position_usable:
        bc = pf.ball.selected
        ball = {
            "x": bc.x,
            "y": bc.y,
            "z": bc.z,          # height in metres; may be None for raw when z is absent
            "coord_source": bc.source,
        }

    events: list[dict[str, Any]] = []
    for a in pf.events:
        raw_ev = event_lookup.get(a.event_row_id) if event_lookup else None
        g_type = getattr(raw_ev, "game_event_type", None) if raw_ev else None
        p_type = getattr(raw_ev, "possession_event_type", None) if raw_ev else None
        s_team = getattr(raw_ev, "source_team_id", None) if raw_ev else None
        s_player = getattr(raw_ev, "source_player_id", None) if raw_ev else None

        events.append({
            "event_row_id": a.event_row_id,
            "game_event_type": g_type,
            "possession_event_type": p_type,
            "canonical_period": a.canonical_period,
            "event_time_s": a.event_time_s,
            "residual_s": a.residual_s,
            "quality_label": a.quality_label,       # 'temporal_only' etc.; never 'validated'
            "validated_correspondence": a.validated_correspondence,  # always 'not_assessed'
            "source_team_id": s_team,
            "source_player_id": s_player,
        })

    return {
        "frame_num": pf.frame_num,
        "video_time_s": pf.video_time_s,
        "canonical_period": pf.canonical_period,
        "source_period": pf.source_period,
        "period_usable": pf.period_usable,
        "has_gap": has_gap,
        "players": players,
        "ball": ball,
        "events": events,
    }


def build_roster_names(roster) -> dict[int, str]:
    """Build a ``{player_id: display_name}`` map from a ``Roster`` object."""
    names: dict[int, str] = {}
    for e in roster.entries:
        if e.player_id and e.nickname:
            names[e.player_id] = e.nickname
    return names


def build_roster_lookup(roster) -> dict[int, dict[str, Any]]:
    """Build a ``{player_id: {name, jersey, side, position_group, team_id}}`` map."""
    lookup: dict[int, dict[str, Any]] = {}
    for e in roster.entries:
        if e.player_id:
            lookup[e.player_id] = {
                "name": e.nickname,
                "jersey": e.shirt_number,
                "side": e.roster_side,
                "position_group": e.position_group,
                "team_id": e.team_id,
            }
    return lookup


def build_event_index(
    match_sync,
    events: Sequence[Any] | None = None,
    roster_lookup: dict[int, dict[str, Any]] | None = None,
    meta: Any | None = None,
) -> list[dict[str, Any]]:
    """Produce an ordered list of all event alignment results for the event panel.

    Includes both matched and unmatched events. Unmatched ones carry
    ``matched: false`` and a ``reason`` explaining why they have no frame.
    Enriched with player names, jerseys, and team designations.
    """
    from football_fatigue.synchronization import MatchStatus

    event_lookup: dict[str, Any] = {}
    if events:
        for ev in events:
            eid = getattr(ev, "event_row_id", None)
            if eid is None and isinstance(ev, dict):
                eid = ev.get("event_row_id")
            if eid is not None:
                event_lookup[str(eid)] = ev

    results = []
    for a in match_sync.alignments:
        matched = a.status is MatchStatus.MATCHED
        raw_ev = event_lookup.get(a.event_row_id)
        g_type = getattr(raw_ev, "game_event_type", None) if raw_ev else None
        p_type = getattr(raw_ev, "possession_event_type", None) if raw_ev else None
        s_team = getattr(raw_ev, "source_team_id", None) if raw_ev else None
        s_player = getattr(raw_ev, "source_player_id", None) if raw_ev else None

        attrs = getattr(raw_ev, "attributes", None) or {}
        fouls_dict = attrs.get("fouls") if isinstance(attrs.get("fouls"), dict) else {}
        is_foul = bool(fouls_dict and any(v is not None for v in fouls_dict.values()))
        foul_type = fouls_dict.get("foulType") if is_foul else None

        pe_dict = attrs.get("possessionEvents") if isinstance(attrs.get("possessionEvents"), dict) else {}
        is_goal = (g_type == "G") or (p_type == "SH" and pe_dict.get("shotOutcomeType") == "G")

        player_info = roster_lookup.get(s_player) if (roster_lookup and s_player) else None
        player_name = player_info.get("name") if player_info else None
        jersey = player_info.get("jersey") if player_info else None
        player_side = player_info.get("side") if player_info else None

        team_name = None
        if meta:
            if player_side == "home" or s_team == getattr(meta, "home_team_id", None):
                team_name = getattr(meta, "home_team_name", None)
            elif player_side == "away" or s_team == getattr(meta, "away_team_id", None):
                team_name = getattr(meta, "away_team_name", None)

        entry: dict[str, Any] = {
            "event_row_id": a.event_row_id,
            "game_event_type": g_type,
            "possession_event_type": p_type,
            "canonical_period": a.canonical_period,
            "source_period": a.source_period,
            "event_time_s": a.event_time_s,
            "source_team_id": s_team,
            "source_player_id": s_player,
            "player_name": player_name,
            "jersey": jersey,
            "player_side": player_side,
            "team_name": team_name,
            "is_foul": is_foul,
            "foul_type": foul_type,
            "is_goal": is_goal,
            "matched": matched,
            "quality_label": a.quality_label,
        }
        if matched:
            entry["frame_num"] = a.frame_num
            entry["residual_s"] = a.residual_s
        else:
            entry["reason"] = a.reason.value if a.reason else None
        results.append(entry)
    return results
