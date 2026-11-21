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
import csv
import io
import json
import math
import re
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from football_fatigue.ingestion import Dataset, EventVersion
from football_fatigue.match_data import MatchData, CoordinatePolicy
from football_fatigue.analytics import summarize_match_movement, compute_spatial_analytics
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
    spatial_payload = compute_spatial_analytics(
        raw_prepared_frames,
        pitch_length_m=meta.pitch_length_m,
        pitch_width_m=meta.pitch_width_m,
    )

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
        "spatial_analytics": spatial_payload,
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
    home_html_path: Path = Path(__file__).parent.parent / "board" / "home.html"
    cv_html_path: Path = Path(__file__).parent.parent / "board" / "cv.html"
    vision_demo_dir: Path = Path("data/processed/vision/soccertrack-117093-detection-only-30f-20261018")
    vision_video_path: Path = Path("data/external/soccertrack-v2/raw/117093/117093.mp4")
    vision_mot_path: Path = Path("data/external/soccertrack-v2/mot/117093.txt")
    vision_export_dir: Path = Path("data/processed/vision_exports/full")
    _mot_frames: dict[int, list[dict[str, float | int]]] | None = None
    _mot_summary: dict[str, Any] | None = None
    _cv_analytics_cache: Any = None
    _cv_coach_cache: Any = None

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
            self._serve_file(self.home_html_path, "text/html; charset=utf-8")

        elif path in ("/tactical", "/tactical/"):
            self._serve_file(self.board_html_path, "text/html; charset=utf-8")

        elif path in ("/cv", "/cv/"):
            self._serve_file(self.cv_html_path, "text/html; charset=utf-8", no_store=True)

        elif path == "/api/cv/demo":
            self._cv_demo()

        elif path == "/api/cv/frames":
            self._cv_frames(parsed)

        elif path == "/api/cv/tracks.csv":
            self._cv_tracks_csv()

        elif path == "/api/cv/analytics":
            self._cv_analytics()

        elif path == "/api/cv/coach":
            self._cv_coach()

        elif path in self._CV_DOWNLOADS:
            self._cv_download(path, "download" in parse_qs(parsed.query))

        elif path == "/api/cv/mot":
            self._cv_mot(parsed)

        elif path == "/media/soccertrack/117093.mp4":
            self._serve_video(self.vision_video_path)

        elif path == "/api/matches":
            matches = get_matches_catalog(self.root)
            self._json(matches)

        elif path == "/api/match":
            if not self.payload or not self.payload.get("loaded"):
                self._json({"loaded": False})
                return
            meta = {k: v for k, v in self.payload.items() if k not in ("frames", "analytics", "spatial_analytics")}
            self._json(meta)

        elif path == "/api/frames":
            self._json(self.payload.get("frames", []))

        elif path == "/api/events":
            self._json(self.payload.get("events", []))

        elif path == "/api/analytics":
            self._json(self.payload.get("analytics", {}))

        elif path == "/api/spatial":
            self._json(self.payload.get("spatial_analytics", {}))

        elif path == "/api/report.csv":
            self._csv_report()

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

    def _csv_report(self) -> None:
        """Download derived movement and spatial summaries as a local CSV response."""
        if not self.payload or not self.payload.get("loaded"):
            self.send_error(409, "Load a match before downloading its report")
            return
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer)
        writer.writerow([
            "record_type", "match_id", "period", "side", "jersey", "frame_num",
            "window_s", "observations", "x_m", "y_m", "distance_m", "speed_m_s",
            "length_span_m", "width_span_m", "convex_hull_area_m2", "estimated_count",
            "value", "unit", "notes",
        ])
        match_id = self.payload.get("match_id")
        for key, metrics in self.payload.get("analytics", {}).items():
            side, jersey = key.rsplit("_", 1)
            writer.writerow(["player_movement", match_id, "", side, jersey, "", "", "", "", "",
                             metrics.get("total_distance_m"), metrics.get("max_speed_m_s"), "", "", "", "",
                             metrics.get("coverage_fraction"), "fraction",
                             f"estimated_fraction={metrics.get('estimated_fraction')}; reliable={metrics.get('is_reliable')}; "
                             f"distance_m={metrics.get('total_distance_m')}; max_speed_m_s={metrics.get('max_speed_m_s')}; "
                             f"not physiological fatigue"])
        spatial = self.payload.get("spatial_analytics", {})
        for row in spatial.get("team_shape", []):
            writer.writerow(["team_shape", match_id, row.get("canonical_period"), row.get("side"), "",
                             row.get("frame_num"), "", row.get("observed_player_count"),
                             row.get("centroid_x_m"), row.get("centroid_y_m"), "", "",
                             row.get("length_span_m"), row.get("width_span_m"),
                             row.get("convex_hull_area_m2"), row.get("estimated_player_count"),
                             "", "metres", "Observed positions only; roster visibility may vary"])
        for row in spatial.get("positional_changes", []):
            writer.writerow(["positional_change", match_id, row.get("canonical_period"), row.get("side"),
                             row.get("jersey"), "", row.get("window_s"),
                             f"{row.get('from_observations')};{row.get('to_observations')}",
                             row.get("mean_x_to_m"), row.get("mean_y_to_m"),
                             "", "", "", "",
                             "", f"{row.get('from_estimated_observations')};{row.get('to_estimated_observations')}",
                             row.get("mean_position_change_m"), "metres",
                             f"from_window={row.get('from_window')}; to_window={row.get('to_window')}; "
                             f"mean_position_change_m={row.get('mean_position_change_m')}; distance between adjacent window means"])
        heatmap = spatial.get("heatmap", {})
        for side, matrix in heatmap.get("counts", {}).items():
            for row_idx, row in enumerate(matrix):
                for col_idx, count in enumerate(row):
                    if count:
                        writer.writerow(["heatmap_cell", match_id, "", side, "", "", "", count,
                                         "", "", "", "", "", "", "",
                                         heatmap.get("estimated_observations_in_grid", {}).get(side),
                                         count, "usable observations", f"row={row_idx};column={col_idx}; observation count, not dwell time"])
        body = buffer.getvalue().encode("utf-8-sig")
        self.send_response(200)
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header("Content-Disposition", f"attachment; filename=match-{match_id}-analytics.csv")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, path: Path, content_type: str, no_store: bool = False) -> None:
        if not path.exists():
            self.send_error(404, f"File not found: {path}")
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if no_store:
            self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _cv_run_dir(self) -> Path:
        demo_dir = self.vision_demo_dir
        candidates = sorted(demo_dir.parent.glob("soccertrack-117093-*/frames.jsonl"))
        if candidates:
            # Prefer the newest run, including an actively processing run.
            demo_dir = max(candidates, key=lambda path: path.stat().st_mtime).parent
        return demo_dir

    def _cv_demo(self) -> None:
        """Return a bounded summary of the selected local CV run.

        Frames are not returned here; use ``/api/cv/frames`` windows. The
        endpoint names one known sample and never exposes arbitrary files.
        """
        from football_fatigue.vision import run_index

        demo_dir = self._cv_run_dir()
        manifest_path = demo_dir / "manifest.json"
        frames_path = demo_dir / "frames.jsonl"
        analytics_path = demo_dir / "analytics.json"
        if not self.vision_video_path.is_file() or not frames_path.is_file():
            self._json({
                "available": False,
                "message": "The SoccerTrack video or its saved CV demo output is not available.",
            })
            return
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {
                "status": "processing",
                "configuration": {"video_fps": 25.0, "video_width_px": 4096, "video_height_px": 1080},
                "summary": {},
            }
            analytics = json.loads(analytics_path.read_text(encoding="utf-8")) if analytics_path.is_file() else None
            cfg = manifest.get("configuration") or {}
            index = run_index.get_index(
                frames_path, int(cfg.get("video_width_px") or 4096), int(cfg.get("video_height_px") or 1080)
            )
            summary = run_index.summary(index)
            first = run_index.frame_window(frames_path, index, 0, 10**9, limit=25)
        except (OSError, ValueError, KeyError) as exc:
            self._json({"available": False, "message": f"Could not read CV demo artifacts: {exc}"})
            return
        self._json({
            "available": True,
            "sample_id": "soccertrack-117093",
            "video_url": "/media/soccertrack/117093.mp4",
            "frames": first,  # compatibility: first 25 frames only; use frames_endpoint
            "video_start_time_s": run_index.video_start_time_s(self.vision_video_path),
            "frames_endpoint": "/api/cv/frames",
            "csv_url": "/api/cv/tracks.csv",
            "manifest": manifest,
            "analytics": analytics,
            "image_space": summary,
            "notice": (
                "Detector output covers the saved run. Boxes and track IDs are image-space estimates; "
                "pitch analytics are shown only when calibrated coordinates pass the input checks."
            ),
        })

    # Fixed whitelist of saved exports; no client-supplied path ever reaches the filesystem.
    _CV_DOWNLOADS = {
        "/media/soccertrack/annotated.mp4": ("annotated_full_117093.mp4", "Annotated full video"),
        "/media/soccertrack/preview.mp4": ("preview_30s_2048.mp4", "30 s preview (frames 3000-3749)"),
    }

    def _cv_export_files(self) -> list[dict[str, Any]]:
        out = []
        for url, (name, label) in self._CV_DOWNLOADS.items():
            f = self.vision_export_dir / name
            out.append({
                "url": url, "label": label, "filename": name, "available": f.is_file(),
                "size_bytes": f.stat().st_size if f.is_file() else None,
            })
        return out

    def _cv_download(self, path: str, as_attachment: bool) -> None:
        name = self._CV_DOWNLOADS[path][0]
        target = self.vision_export_dir / name
        self._serve_video(target, attachment=name if as_attachment else None)

    def _cv_coach(self) -> None:
        """Coach-facing team-shape insights for the selected run.

        Uses the saved ``coach_insights.json`` (written by ``python -m football_fatigue.vision.team_shape``,
        includes reference validation). Without it the shape analytics are computed from ``frames.jsonl``
        and marked unvalidated.
        """
        from football_fatigue.vision import run_index, team_shape

        run_dir = self._cv_run_dir()
        frames_path = run_dir / "frames.jsonl"
        if not frames_path.is_file():
            self._json({"available": False, "message": "No saved CV run."})
            return
        saved = run_dir / "coach_insights.json"
        try:
            if saved.is_file():
                insights = json.loads(saved.read_text(encoding="utf-8"))
                insights["validated"] = "validation" in insights
            else:
                key = (str(frames_path), frames_path.stat().st_size)
                cached = BoardHandler._cv_coach_cache
                if cached is None or cached[0] != key:
                    BoardHandler._cv_coach_cache = cached = (key, team_shape.compute(frames_path))
                insights = dict(cached[1], validated=False)
        except (OSError, ValueError) as exc:
            self._json({"available": False, "message": f"Could not build coach insights: {exc}"})
            return
        self._json({
            "available": True, "run": run_dir.name, **insights,
            "video_start_time_s": run_index.video_start_time_s(self.vision_video_path),
            "downloads": self._cv_export_files(),
        })

    def _cv_analytics(self) -> None:
        """Cached team-level image-space analytics for the selected saved run (no detector run)."""
        from football_fatigue.vision import run_index

        try:
            frames_path, index = self._cv_index()
        except (ValueError, OSError) as exc:
            self._json({"available": False, "message": str(exc)})
            return
        if index is None:
            self._json({"available": False, "message": "No saved CV run."})
            return
        run_dir = frames_path.parent

        def load(name: str):
            f = run_dir / name
            try:
                return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else None
            except (OSError, ValueError):
                return None

        manifest = load("manifest.json") or {}
        reference_eval = load("tracking_eval.json")
        key = (str(frames_path), index.consumed, bool(reference_eval))
        cached = BoardHandler._cv_analytics_cache
        if cached is None or cached[0] != key:
            result = run_index.team_analytics(index, manifest, reference_eval)
            result["comparison"] = load("comparison.json")
            BoardHandler._cv_analytics_cache = cached = (key, result)
        self._json({
            "available": True,
            "run": run_dir.name,
            **cached[1],
            "downloads": self._cv_export_files(),
            "advanced_metrics": {
                "status": "unavailable",
                "reasons": ["Camera calibration is not validated, so metric distance, speed, Pitch Control, "
                            "EPV, OBSO, DAS and fatigue are not computed."],
            },
        })

    def _cv_index(self):
        from football_fatigue.vision import run_index

        demo_dir = self._cv_run_dir()
        frames_path = demo_dir / "frames.jsonl"
        if not frames_path.is_file():
            return None, None
        manifest_path = demo_dir / "manifest.json"
        cfg = {}
        if manifest_path.is_file():
            cfg = (json.loads(manifest_path.read_text(encoding="utf-8")).get("configuration") or {})
        return frames_path, run_index.get_index(
            frames_path, int(cfg.get("video_width_px") or 4096), int(cfg.get("video_height_px") or 1080)
        )

    def _cv_frames(self, parsed: Any) -> None:
        """Bounded frame window: ``?start=<frame>&end=<frame>&limit=<n>``."""
        from football_fatigue.vision import run_index

        qs = parse_qs(parsed.query)
        try:
            start = int(qs.get("start", ["0"])[0])
            end = int(qs.get("end", [str(start + 100)])[0])
            limit = int(qs.get("limit", [str(run_index.MAX_WINDOW_FRAMES)])[0])
            frames_path, index = self._cv_index()
        except (ValueError, OSError) as exc:
            self._json({"available": False, "message": str(exc)})
            return
        if index is None:
            self._json({"available": False, "message": "No saved CV run."})
            return
        self._json({
            "available": True,
            "start": start,
            "frames": run_index.frame_window(frames_path, index, start, end, limit),
            "frames_indexed": len(index.offsets),
        })

    def _cv_tracks_csv(self) -> None:
        from football_fatigue.vision import run_index

        try:
            frames_path, index = self._cv_index()
        except (ValueError, OSError):
            frames_path = index = None
        if index is None:
            self.send_error(404, "No saved CV run")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header("Content-Disposition", 'attachment; filename="cv_tracks_pixel_space.csv"')
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        for chunk in run_index.tracks_csv(frames_path, index):
            self.wfile.write(chunk.encode("utf-8"))

    @classmethod
    def _load_mot_annotations(cls) -> tuple[dict[int, list[dict[str, float | int]]], dict[str, Any]]:
        """Load the one paired, local MOT file into a frame index once.

        These are reference annotations, never detector predictions.  Keeping
        the source separate makes it possible to inspect full-video alignment
        without misrepresenting labels as model output.
        """
        if cls._mot_frames is not None and cls._mot_summary is not None:
            return cls._mot_frames, cls._mot_summary
        if not cls.vision_mot_path.is_file():
            raise FileNotFoundError("Paired SoccerTrack MOT annotations are unavailable")
        frames: dict[int, list[dict[str, float | int]]] = {}
        trajectories: dict[int, list[tuple[int, float, float]]] = {}
        with cls.vision_mot_path.open("r", encoding="utf-8", newline="") as stream:
            for row_number, row in enumerate(csv.reader(stream), start=1):
                if len(row) < 6:
                    raise ValueError(f"MOT row {row_number} has fewer than six columns")
                try:
                    frame = int(float(row[0]))
                    track_id = int(float(row[1]))
                    x, y, width, height = (float(value) for value in row[2:6])
                except ValueError as exc:
                    raise ValueError(f"MOT row {row_number} has invalid numeric values") from exc
                if width <= 0 or height <= 0:
                    continue
                record = {"track_id": track_id, "x": x, "y": y, "width": width, "height": height}
                frames.setdefault(frame, []).append(record)
                trajectories.setdefault(track_id, []).append((frame, x + width / 2, y + height / 2))
        speeds: list[float] = []
        for points in trajectories.values():
            points.sort()
            for (previous_frame, previous_x, previous_y), (frame, x, y) in zip(points, points[1:]):
                delta = frame - previous_frame
                if delta > 0:
                    speeds.append(math.hypot(x - previous_x, y - previous_y) * 25.0 / delta)
        sorted_speeds = sorted(speeds)
        summary = {
            "source": "SoccerTrack paired MOT reference annotations",
            "unit": "pixels_per_second",
            "frame_count": len(frames),
            "first_frame": min(frames),
            "last_frame": max(frames),
            "detection_count": sum(len(records) for records in frames.values()),
            "track_count": len(trajectories),
            "mean_boxes_per_frame": round(sum(len(records) for records in frames.values()) / len(frames), 2),
            "median_image_speed_px_s": round(sorted_speeds[len(sorted_speeds) // 2], 2) if sorted_speeds else None,
        }
        cls._mot_frames, cls._mot_summary = frames, summary
        return frames, summary

    def _cv_mot(self, parsed: Any) -> None:
        """Return one frame of paired reference boxes, plus image-space summary."""
        requested = parse_qs(parsed.query).get("frame", [None])[0]
        try:
            frames, summary = self._load_mot_annotations()
            frame = int(requested) if requested is not None else summary["first_frame"]
        except (FileNotFoundError, ValueError) as exc:
            self._json({"available": False, "message": str(exc)})
            return
        self._json({
            "available": True,
            "frame_num": frame,
            "boxes": frames.get(frame, []),
            "summary": summary,
            "notice": "Reference annotation overlay. These boxes and IDs are not detector output.",
        })

    def _serve_video(self, path: Path, attachment: str | None = None) -> None:
        """Stream the local demonstration video with single-range support."""
        if not path.is_file():
            self.send_error(404, "Video file not found")
            return
        size = path.stat().st_size
        start, end = 0, size - 1
        range_header = self.headers.get("Range")
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if not match:
                self.send_error(416, "Invalid Range header")
                return
            start = int(match.group(1)) if match.group(1) else 0
            end = int(match.group(2)) if match.group(2) else size - 1
            if start > end or start >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            end = min(end, size - 1)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        length = end - start + 1
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(length))
        if attachment:
            self.send_header("Content-Disposition", f'attachment; filename="{attachment}"')
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        with path.open("rb") as stream:
            stream.seek(start)
            remaining = length
            while remaining:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (ConnectionError, OSError):
                    return  # the browser closed this range request (seek/stall); not an error
                remaining -= len(chunk)

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
    workspace = Path.cwd()
    BoardHandler.vision_demo_dir = workspace / "data" / "processed" / "vision" / "soccertrack-117093-detection-only-30f-20261018"
    BoardHandler.vision_video_path = workspace / "data" / "external" / "soccertrack-v2" / "raw" / "117093" / "117093.mp4"
    BoardHandler.vision_export_dir = workspace / "data" / "processed" / "vision_exports" / "full"
    BoardHandler.vision_mot_path = workspace / "data" / "external" / "soccertrack-v2" / "mot" / "117093.txt"
    BoardHandler._mot_frames = None
    BoardHandler._mot_summary = None

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

    # Threaded: a long-lived video range request must not block JSON API calls.
    server = ThreadingHTTPServer(("127.0.0.1", port), BoardHandler)
    server.daemon_threads = True
    url = f"http://127.0.0.1:{port}/"
    print(f"\n[server] Football analytics workspace ready at {url}")
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
