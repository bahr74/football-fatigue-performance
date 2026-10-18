"""Tactical Board HTTP Server (Z2).

A lightweight stdlib HTTP server that:
- Serves the tactical board web frontend.
- Starts instantly with an interactive Match Selection & Setup Card (no hardcoded default match).
- Allows users to select any of the 64 FIFA World Cup 2022 tournament matches.
- Configures duration/scope (1st Half, 2nd Half, Full Match, or custom minutes/frames) and coordinate policy.
- Dynamically loads and synchronises positional tracking frames and match events.
- Tracks match scoreline and goal events in real time.
- Opens the tactical board in the default browser.

Usage (module entry point)::

    python -m football_fatigue.playback

Or to preload a specific match::

    python -m football_fatigue.playback --match 10517 --scope half1

No third-party dependencies; uses only ``http.server``, ``json``,
``threading``, ``webbrowser``, and the existing ``football_fatigue`` package.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from football_fatigue.ingestion import Dataset, EventVersion
from football_fatigue.match_data import MatchData, CoordinatePolicy
from football_fatigue.analytics import summarize_match_movement
from football_fatigue.playback.adapter import (
    build_event_index,
    build_roster_lookup,
    build_roster_names,
    frame_to_state,
)

# ------------------------------------------------------------------ tournament matches catalog

_CATALOG_CACHE: dict[str, list[dict[str, Any]]] = {}

STAGE_ORDER = {
    8: (1, "🏆 Final"),
    7: (2, "🥉 3rd Place Play-off"),
    6: (3, "Semi-finals"),
    5: (4, "Quarter-finals"),
    4: (5, "Round of 16"),
    3: (6, "Group Stage — Matchday 3"),
    2: (7, "Group Stage — Matchday 2"),
    1: (8, "Group Stage — Matchday 1"),
}


def get_matches_catalog(root: str) -> list[dict[str, Any]]:
    """Scan dataset root for all matches, categorized by tournament stage."""
    if root in _CATALOG_CACHE:
        return _CATALOG_CACHE[root]

    try:
        ds = Dataset(root)
        matches = []
        for mid in ds.match_ids():
            try:
                meta = ds.metadata(mid)
                mw = meta.matchweek or 0
                order, stage_name = STAGE_ORDER.get(mw, (9, f"Matchweek {mw}"))
                matches.append({
                    "match_id": str(mid),
                    "home_team": meta.home_team_name,
                    "away_team": meta.away_team_name,
                    "home_team_id": meta.home_team_id,
                    "away_team_id": meta.away_team_id,
                    "matchweek": mw,
                    "stage_order": order,
                    "stage": stage_name,
                    "kickoff_date": meta.kickoff_date,
                    "stadium_name": meta.stadium_name,
                    "label": f"{meta.home_team_name} vs {meta.away_team_name}",
                })
            except Exception:
                continue

        # Sort: tournament hierarchy (Final down to Group Stage), then alphabetically
        matches.sort(key=lambda m: (m["stage_order"], m["home_team"]))
        _CATALOG_CACHE[root] = matches
        return matches
    except Exception as e:
        print(f"[server] Warning: Failed to catalog matches from {root!r}: {e}")
        return []


# ------------------------------------------------------------------ data loading

def load_match_data(
    root: str,
    match_id: str,
    max_frames: int | None = None,
    *,
    scope: str = "half1",
    period: int | None = None,
    minutes: float | None = None,
    policy: CoordinatePolicy = CoordinatePolicy.SMOOTHED_THEN_RAW,
) -> dict[str, Any]:
    """Load a match segment and event index into memory.

    Parameters
    ----------
    root:
        Dataset raw root folder.
    match_id:
        Identifier of the match.
    max_frames:
        Optional hard upper bound on number of frames.
    scope:
        Segment scope: 'half1' (1st Half), 'half2' (2nd Half), 'full', or 'custom'.
    period:
        Explicit period filter (1, 2, 3, 4).
    minutes:
        Optional limit on elapsed minutes of match time to load.
    policy:
        Coordinate selection policy.
    """
    print(f"[server] Opening match {match_id} (scope={scope!r}, period={period}, minutes={minutes}, max_frames={max_frames}, policy={policy.value}) ...")
    md = MatchData.open(root, match_id, EventVersion.ROOT, policy=policy)
    meta = md.metadata
    roster = md.roster
    roster_lookup = build_roster_lookup(roster)
    roster_names = build_roster_names(roster)

    print(f"[server]   {meta.home_team_name} vs {meta.away_team_name}  |  "
          f"pitch {meta.pitch_length_m}x{meta.pitch_width_m}m  |  {meta.fps} fps")

    # Synchronize events first (one tracking pass, cached inside md).
    print("[server] Synchronising events ...")
    t0 = time.perf_counter()
    sync = md.synchronize(spatial=False)
    t1 = time.perf_counter()
    print(f"[server]   sync complete in {t1 - t0:.1f}s  |  "
          f"{sync.summary()['matched']} matched, {sync.summary()['unmatched']} unmatched")

    # Build event index (all events, matched + unmatched) enriched with roster & team
    events = md.events()
    event_lookup = {e.event_row_id: e for e in events}
    event_index = build_event_index(sync, events, roster_lookup=roster_lookup, meta=meta)

    # Track match goals and running score
    home_score = 0
    away_score = 0
    goal_events: list[dict[str, Any]] = []
    for ev in event_index:
        if ev.get("is_goal"):
            is_home = (ev.get("player_side") == "home") or (ev.get("source_team_id") == meta.home_team_id)
            if is_home:
                home_score += 1
            else:
                away_score += 1
            goal_events.append({
                "time_s": ev.get("event_time_s"),
                "canonical_period": ev.get("canonical_period"),
                "player_name": ev.get("player_name"),
                "jersey": ev.get("jersey"),
                "team_side": "home" if is_home else "away",
                "team_name": meta.home_team_name if is_home else meta.away_team_name,
                "home_score": home_score,
                "away_score": away_score,
            })

    # Determine target period
    target_period: int | None = None
    if scope == "half1":
        target_period = 1
    elif scope == "half2":
        target_period = 2
    elif scope == "full":
        target_period = None
    elif period is not None:
        target_period = int(period)

    max_duration_s = (float(minutes) * 60.0) if (minutes is not None and minutes > 0) else None

    # Load frames
    print(f"[server] Loading frames (target_period={target_period}, minutes={minutes}, max_frames={max_frames}) ...")
    t2 = time.perf_counter()
    frames: list[dict[str, Any]] = []
    raw_prepared_frames = []
    prev_time: float | None = None
    prev_period: int | None = None
    start_time: float | None = None

    for pf in md.frames(attach_events=True):
        cp = pf.canonical_period

        # Target period boundary
        if target_period is not None:
            if cp is None or cp < target_period:
                continue
            if cp > target_period:
                # Target period finished; stop streaming immediately
                break

        # Time duration limit (video time)
        if start_time is None:
            start_time = pf.video_time_s
        if max_duration_s is not None and (pf.video_time_s - start_time) > max_duration_s:
            break

        raw_prepared_frames.append(pf)
        state = frame_to_state(pf, roster_names, event_lookup=event_lookup)
        if prev_time is not None:
            state["gap_s"] = round(pf.video_time_s - prev_time, 4)
            state["period_changed"] = pf.canonical_period != prev_period
        else:
            state["gap_s"] = 0.0
            state["period_changed"] = False
        prev_time = pf.video_time_s
        prev_period = pf.canonical_period
        frames.append(state)

        # Max frames count filter
        if max_frames is not None and max_frames > 0 and len(frames) >= max_frames:
            break

    t3 = time.perf_counter()
    print(f"[server]   {len(frames)} frames loaded in {t3 - t2:.1f}s")

    # Compute physical tracking analytics
    print("[server] Computing movement analytics ...")
    summaries = summarize_match_movement(raw_prepared_frames)
    analytics_payload = {
        f"{k[0]}_{k[1]}": v.as_dict() for k, v in summaries.items()
    }
    print(f"[server]   computed metrics for {len(analytics_payload)} players")

    return {
        "loaded": True,
        "match_id": match_id,
        "home_team": meta.home_team_name,
        "away_team": meta.away_team_name,
        "home_team_id": meta.home_team_id,
        "away_team_id": meta.away_team_id,
        "pitch_length_m": meta.pitch_length_m,
        "pitch_width_m": meta.pitch_width_m,
        "fps": meta.fps,
        "policy": policy.value,
        "scope": scope,
        "target_period": target_period,
        "max_frames_loaded": len(frames),
        "final_home_score": home_score,
        "final_away_score": away_score,
        "goal_events": goal_events,
        "frames": frames,
        "events": event_index,
        "analytics": analytics_payload,
        "coordinate_system": "pitch_center_m",
        "x_axis": "+x toward right in main-camera view",
        "y_axis": "+y toward top in main-camera view",
    }


# ------------------------------------------------------------------ HTTP handler

class BoardHandler(BaseHTTPRequestHandler):
    """Serves the tactical board HTML and all JSON endpoints."""

    payload: dict[str, Any] = {}
    root: str = "data/raw/FIFA World Cup 2022"
    lock: threading.Lock = threading.Lock()
    board_html_path: Path = Path(__file__).parent.parent / "board" / "index.html"

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self._serve_file(self.board_html_path, "text/html; charset=utf-8")

        elif path == "/api/matches":
            matches = get_matches_catalog(self.root)
            self._json(matches)

        elif path == "/api/match":
            if not self.payload or not self.payload.get("loaded"):
                self._json({"loaded": False})
                return
            meta = {k: v for k, v in self.payload.items() if k not in ("frames", "analytics")}
            self._json(meta)

        elif path == "/api/frames":
            self._json(self.payload.get("frames", []))

        elif path == "/api/events":
            self._json(self.payload.get("events", []))

        elif path == "/api/analytics":
            self._json(self.payload.get("analytics", {}))

        elif path == "/api/load":
            qs = parse_qs(parsed.query)
            match_id = qs.get("match", ["3812"])[0]
            scope = qs.get("scope", ["half1"])[0]
            period_val = qs.get("period", [None])[0]
            period = int(period_val) if (period_val and period_val.isdigit()) else None
            minutes_val = qs.get("minutes", [None])[0]
            minutes = float(minutes_val) if minutes_val else None
            max_frames_val = qs.get("max_frames", [None])[0]
            max_frames = int(max_frames_val) if (max_frames_val and max_frames_val.isdigit()) else None
            policy_val = qs.get("policy", ["smoothed_then_raw"])[0]
            pol = CoordinatePolicy.RAW if policy_val == "raw" else (
                CoordinatePolicy.SMOOTHED if policy_val == "smoothed" else CoordinatePolicy.SMOOTHED_THEN_RAW
            )

            with self.lock:
                self.__class__.payload = load_match_data(
                    root=self.root,
                    match_id=match_id,
                    max_frames=max_frames,
                    scope=scope,
                    period=period,
                    minutes=minutes,
                    policy=pol,
                )

            self._json({
                "status": "ok",
                "match_id": match_id,
                "scope": scope,
                "frames_loaded": len(self.payload.get("frames", [])),
                "events_count": len(self.payload.get("events", [])),
                "home_team": self.payload.get("home_team"),
                "away_team": self.payload.get("away_team"),
                "final_home_score": self.payload.get("final_home_score", 0),
                "final_away_score": self.payload.get("final_away_score", 0),
            })

        else:
            self.send_error(404, "Not Found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/load":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len) if content_len > 0 else b"{}"
            try:
                data = json.loads(body.decode("utf-8")) if body else {}
            except Exception:
                data = {}

            match_id = str(data.get("match", "3812"))
            scope = data.get("scope", "half1")
            period = data.get("period")
            minutes = data.get("minutes")
            max_frames = data.get("max_frames")
            policy_val = data.get("policy", "smoothed_then_raw")
            pol = CoordinatePolicy.RAW if policy_val == "raw" else (
                CoordinatePolicy.SMOOTHED if policy_val == "smoothed" else CoordinatePolicy.SMOOTHED_THEN_RAW
            )

            with self.lock:
                self.__class__.payload = load_match_data(
                    root=self.root,
                    match_id=match_id,
                    max_frames=int(max_frames) if max_frames else None,
                    scope=scope,
                    period=int(period) if period else None,
                    minutes=float(minutes) if minutes else None,
                    policy=pol,
                )

            self._json({
                "status": "ok",
                "match_id": match_id,
                "scope": scope,
                "frames_loaded": len(self.payload.get("frames", [])),
                "events_count": len(self.payload.get("events", [])),
                "home_team": self.payload.get("home_team"),
                "away_team": self.payload.get("away_team"),
                "final_home_score": self.payload.get("final_home_score", 0),
                "final_away_score": self.payload.get("final_away_score", 0),
            })
        else:
            self.send_error(404, "Not Found")

    def _json(self, data: Any) -> None:
        body = json.dumps(data, allow_nan=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self.send_error(404, f"File not found: {path}")
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        pass


# ------------------------------------------------------------------ entry point

def run(
    root: str = "data/raw/FIFA World Cup 2022",
    match_id: str | None = None,
    max_frames: int | None = None,
    scope: str = "half1",
    period: int | None = 1,
    minutes: float | None = None,
    policy: str = "smoothed_then_raw",
    port: int = 5780,
    open_browser: bool = True,
) -> None:
    """Start the HTTP server. If match_id is None, prompts in the web interface."""
    BoardHandler.root = root

    if match_id:
        pol = CoordinatePolicy.RAW if policy == "raw" else (
            CoordinatePolicy.SMOOTHED if policy == "smoothed" else CoordinatePolicy.SMOOTHED_THEN_RAW
        )
        BoardHandler.payload = load_match_data(
            root=root,
            match_id=match_id,
            max_frames=max_frames,
            scope=scope,
            period=period,
            minutes=minutes,
            policy=pol,
        )
    else:
        BoardHandler.payload = {}

    server = HTTPServer(("127.0.0.1", port), BoardHandler)
    url = f"http://127.0.0.1:{port}/"
    print(f"\n[server] Tactical board server ready at {url}")
    print("[server] Press Ctrl+C to stop.\n")

    if open_browser:
        threading.Thread(target=lambda: (time.sleep(0.4), webbrowser.open(url)), daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] Stopped.")


def _cli() -> None:
    p = argparse.ArgumentParser(description="Football Fatigue Tactical Board Server")
    p.add_argument("--root", default="data/raw/FIFA World Cup 2022",
                   help="Path to the raw dataset root (default: 'data/raw/FIFA World Cup 2022')")
    p.add_argument("--match", default=None,
                   help="Optional match ID to preload (default: None, opens Match Selection Card)")
    p.add_argument("--scope", default="half1", choices=["half1", "half2", "full", "custom"],
                   help="Segment scope to load (default: half1 for First Half)")
    p.add_argument("--minutes", type=float, default=None,
                   help="Minutes to load from segment (default: None for full segment)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Max tracking frames to load (default: None; use 0 for unlimited)")
    p.add_argument("--policy", default="smoothed_then_raw", choices=["raw", "smoothed", "smoothed_then_raw"],
                   help="Coordinate policy (default: smoothed_then_raw)")
    p.add_argument("--port", type=int, default=5780, help="HTTP port (default: 5780)")
    p.add_argument("--no-browser", action="store_true", help="Do not open a browser tab automatically")
    args = p.parse_args()

    max_frames = None if (args.max_frames == 0 or args.max_frames is None) else args.max_frames
    run(
        root=args.root,
        match_id=args.match,
        max_frames=max_frames,
        scope=args.scope,
        minutes=args.minutes,
        policy=args.policy,
        port=args.port,
        open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    _cli()
