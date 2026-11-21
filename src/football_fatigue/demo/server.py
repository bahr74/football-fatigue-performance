"""Local server for the Football Vision Demo. Reuses the existing playback server's handler (range video streaming, JSON helpers)
but exposes only a fixed whitelist of routes; no client-supplied path ever reaches the filesystem."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import threading
import time
import webbrowser
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from football_fatigue.analytics import coach
from football_fatigue.demo import review as RV
from football_fatigue.playback.server import BoardHandler

DEFAULT_MATCH = "117093"
PAGE = Path(__file__).with_name("demo.html")
PAGE_M7 = Path(__file__).with_name("demo_m7.html")   # previous (M7 sprint) layout, kept for comparison at /m7
MATCHES = {
    # v5 = v4 + M7.2 opt-in team assignment (windowed tracklet votes + touchline rule); v4 kept for comparison
    "117093": {"video": Path("data/external/soccertrack-v2/raw/117093/117093.mp4"), "overlay": Path("data/processed/demo/117093/overlay_v5.json"),
               "reviews": Path("data/processed/demo/117093/reviews/identity_reviews_v1.jsonl"),
               # M7.3 opt-in: continuous-inverse positions + per-sample sigma, and short-gap hand-over review suggestions (never applied)
               "motion": Path("data/processed/demo/117093/motion_v1.json"),
               "suggestions": Path("data/processed/demo/117093/handover_suggestions_v1.json")},
}
PRESET_DIR = Path("data/processed/demo/presets")
PRESET_NAME = re.compile(r"^[A-Za-z0-9 _-]{1,40}$")


class DemoHandler(BoardHandler):
    match_id = DEFAULT_MATCH
    video_path = MATCHES[DEFAULT_MATCH]["video"]
    overlay_path = MATCHES[DEFAULT_MATCH]["overlay"]
    selftest_dir: Path | None = None
    review_path = MATCHES[DEFAULT_MATCH]["reviews"]
    preset_dir = PRESET_DIR
    motion_path: Path | None = MATCHES[DEFAULT_MATCH]["motion"]
    suggestions_path: Path | None = MATCHES[DEFAULT_MATCH]["suggestions"]
    _overlay_gz: bytes | None = None
    _physical = None
    _lock = threading.Lock()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html", "/demo"):
            self._serve_file(PAGE, "text/html; charset=utf-8", no_store=True)
        elif path == "/m7":
            self._serve_file(PAGE_M7, "text/html; charset=utf-8", no_store=True)
        elif path == "/api/overlay":
            self._overlay()
        elif path == f"/media/{self.match_id}.mp4":
            self._serve_video(self.video_path)
        elif path == "/api/presets":
            self._json({"builtin": coach.PRESETS, "saved": self._saved_presets(), "schema": coach.SETTINGS_SCHEMA,
                        "defaults": coach.CoachSettings().as_dict()})
        elif path == "/api/reviews":
            ev = self._review_store().events()
            self._json({"events": ev, "state": RV.resolve(ev, self._physical_index().segments_dict())})
        elif path == "/api/suggestions":
            p = self.suggestions_path
            d = json.loads(p.read_text(encoding="utf-8")) if p is not None and p.is_file() else None
            self._json(d if d and d.get("match_id") == str(self.match_id) else {"suggestions": [], "note": "no suggestion file for this match"})
        elif path == "/api/health":
            self._json({"ok": True, "match": self.match_id, "video": self.video_path.is_file(), "overlay": self.overlay_path.is_file()})
        else:
            self.send_error(404, "Not found")

    def do_POST(self) -> None:
        body = self.rfile.read(min(int(self.headers.get("Content-Length") or 0), 4_000_000))   # drain first: closing with unread data resets the socket on Windows
        path = urlparse(self.path).path
        if path in ("/api/analytics", "/api/presets", "/api/reviews"):
            try:
                req = json.loads(body.decode("utf-8") or "{}")
                if not isinstance(req, dict):
                    raise ValueError
            except ValueError:
                self._json_status({"errors": ["request body must be a JSON object"]}, 400)
                return
            {"/api/analytics": self._analytics, "/api/presets": self._save_preset, "/api/reviews": self._add_review}[path](req)
            return
        if path != "/api/selftest" or self.selftest_dir is None:
            self.send_error(404, "Not found")
            return
        out = self.selftest_dir / f"selftest_{time.strftime('%Y%m%d-%H%M%S')}.json"
        with open(out, "x", encoding="utf-8") as fh:
            fh.write(body.decode("utf-8"))
        self._json({"saved": out.name})

    def _json_status(self, data, status: int) -> None:
        b = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _physical_index(self):
        cls = type(self)
        with cls._lock:
            if cls._physical is None:
                from football_fatigue.demo.physical import PhysicalIndex
                mp = self.motion_path
                ov = json.loads(self.overlay_path.read_text(encoding="utf-8"))
                motion = json.loads(mp.read_text(encoding="utf-8")) if mp is not None and mp.is_file() else None
                if motion is not None and (motion.get("match_id") != ov.get("match_id") or motion.get("rows") != len(ov["cols"]["src"])):
                    motion = None       # a sidecar of another match / overlay is never used
                cls._physical = PhysicalIndex(ov, motion)
            return cls._physical

    def _review_store(self) -> RV.ReviewStore:
        return RV.ReviewStore(self.review_path, self.match_id, self._physical_index().ov.get("schema", "unknown"))

    def _analytics(self, req: dict) -> None:
        settings, errors = coach.settings_from_dict(req.get("settings"))
        q = {}
        for k, typ in (("t0", (int, float)), ("t1", (int, float)), ("local", int), ("team", int)):
            v = req.get(k)
            if v is not None and (isinstance(v, bool) or not isinstance(v, typ)):
                errors.append(f"{k} has the wrong type")
            q[k] = v
        P = self._physical_index()
        q["positions"] = req.get("positions") or "stored"
        if q["positions"] not in P.sources:
            errors.append(f"positions must be one of {sorted(P.sources)}")
        if errors:
            self._json_status({"errors": errors}, 400)
            return
        ev = self._review_store().events()
        state = RV.resolve(ev, P.segments_dict()) if ev else None
        with type(self)._lock:
            res = P.analyse(settings, review_state=state, series=req.get("local") is not None, diagnostic=req.get("diagnostic") is True, **q)
        self._json(res)

    def _saved_presets(self) -> dict:
        out = {}
        if self.preset_dir.is_dir():
            for f in sorted(self.preset_dir.glob("*.json")):
                try:
                    d = json.loads(f.read_text(encoding="utf-8"))
                    out[d["name"]] = d
                except (ValueError, KeyError):
                    continue
        return out

    def _save_preset(self, req: dict) -> None:
        name = str(req.get("name") or "").strip()
        settings, errors = coach.settings_from_dict(req.get("settings"))
        if not PRESET_NAME.match(name):
            errors.append("name: 1-40 letters, digits, spaces, '_' or '-'")
        if name in coach.PRESETS:
            errors.append("a built-in example preset cannot be replaced")
        if errors:
            self._json_status({"errors": errors}, 400)
            return
        self.preset_dir.mkdir(parents=True, exist_ok=True)
        f = self.preset_dir / (re.sub(r"[^A-Za-z0-9_-]", "_", name) + ".json")
        prev = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else None
        hist = ([{k: prev[k] for k in ("settings", "saved_utc")}] + prev.get("history", []))[:20] if prev else []
        rec = {"name": name, "schema": coach.SETTINGS_SCHEMA, "settings": settings.as_dict(), "label": "coach-selected definition",
               "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "history": hist}
        f.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        self._json({"saved": name, "preset": rec})

    def _add_review(self, req: dict) -> None:
        segs = self._physical_index().segments_dict()
        try:
            ev = self._review_store().append(req, segs)
        except RV.ReviewError as e:
            self._json_status({"errors": [str(e)]}, 400)
            return
        self._json({"event": ev, "state": RV.resolve(self._review_store().events(), segs)})

    def _overlay(self) -> None:
        if not self.overlay_path.is_file():
            self.send_error(404, "Overlay not built; see the demo README")
            return
        cls = type(self)
        if cls._overlay_gz is None:
            cls._overlay_gz = gzip.compress(verified_overlay_bytes(self.overlay_path, self.video_path), 6)
        body = cls._overlay_gz
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def first_frame_pts_s(video: Path) -> float | None:
    """Presentation time of the first video frame from the container; None if it cannot be read (never a silent 0.0)."""
    try:
        import av

        with av.open(str(video)) as container:
            stream = container.streams.video[0]
            return float(stream.start_time * stream.time_base)
    except Exception:
        return None


def verified_overlay_bytes(overlay_path: Path, video_path: Path) -> bytes:
    """Overlay JSON with ``video.start_time_s`` checked against the real container. The frame<->time mapping is
    PTS(f) = start + f/fps; a wrong stored start shifts every overlay by (error x fps) frames, so the container wins."""
    ov = json.loads(overlay_path.read_text(encoding="utf-8"))
    stored = ov.get("video", {}).get("start_time_s")
    actual = first_frame_pts_s(video_path) if video_path.is_file() else None
    if actual is None:
        ov["video"]["start_time_source"] = "overlay_unverified"
    elif stored is None or abs(stored - actual) > 1e-3:
        ov["video"]["start_time_source"] = f"container (overlay stored {stored!r}, corrected)"
        ov["video"]["start_time_s"] = actual
    else:
        ov["video"]["start_time_source"] = "container_verified"
    return json.dumps(ov, separators=(",", ":")).encode("utf-8")


def run(match: str = DEFAULT_MATCH, port: int = 5790, open_browser: bool = True, selftest_dir: Path | None = None,
        review_file: Path | None = None, preset_dir: Path | None = None) -> None:
    if match not in MATCHES:
        raise SystemExit(f"unknown demo match {match!r}; available: {sorted(MATCHES)}")
    DemoHandler.match_id = match
    DemoHandler.video_path = MATCHES[match]["video"]
    DemoHandler.overlay_path = MATCHES[match]["overlay"]
    DemoHandler.selftest_dir = selftest_dir
    DemoHandler.review_path = review_file or MATCHES[match]["reviews"]
    DemoHandler.preset_dir = preset_dir or PRESET_DIR
    DemoHandler.motion_path = MATCHES[match].get("motion")
    DemoHandler.suggestions_path = MATCHES[match].get("suggestions")
    DemoHandler._overlay_gz = None
    DemoHandler._physical = None
    for label, p in (("overlay", DemoHandler.overlay_path), ("video", DemoHandler.video_path), ("motion sidecar (opt-in)", DemoHandler.motion_path),
                     ("hand-over suggestions", DemoHandler.suggestions_path)):
        print(f"[demo] {label}: {p} {'OK' if p is not None and p.is_file() else 'MISSING'}")
    server = ThreadingHTTPServer(("127.0.0.1", port), DemoHandler)
    server.daemon_threads = True
    url = f"http://127.0.0.1:{port}/"
    print(f"[demo] Football Vision Demo ({match}) at {url}   (Ctrl+C to stop)")
    if open_browser:
        threading.Thread(target=lambda: (time.sleep(0.5), webbrowser.open(url)), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[demo] stopped")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m football_fatigue.demo", description="Football Vision Demo (local, display-only)")
    ap.add_argument("--match", default=DEFAULT_MATCH)
    ap.add_argument("--port", type=int, default=5790)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--selftest-dir", type=Path, help="enable POST /api/selftest and save browser self-test reports here")
    ap.add_argument("--review-file", type=Path, help="identity-review log (append-only JSONL); default: the match's reviews file")
    ap.add_argument("--preset-dir", type=Path, help="folder for saved coach presets; default data/processed/demo/presets")
    a = ap.parse_args(argv)
    run(a.match, a.port, not a.no_browser, a.selftest_dir, a.review_file, a.preset_dir)
    return 0
