"""Native video player server (stdlib HTTP, read-only on every input).

Routes
  /                      the player page (player.html)
  /media/<key>           whitelisted source video with HTTP range support (206) - the browser decodes and plays it
  /api/sources           videos, tracker-input profiles and engine runs
  /api/meta?source=      fps, size, frame count, first-frame PTS
  /api/frames?source=&start=&n=&tracker=&run=     overlay chunk (gzip JSON), see player.data
  /api/events?source=&run=                         engine events of a run (+ GT outcome)
  /api/compare?source=                             B0 vs T0 metrics from saved outputs
  /api/diag?source=&profile=                       debugger: pipeline failure episodes, ID changes, view exits
  /api/strip?source=&gid=&frames=a,b,c&profile=    debugger: before/during/after PNG around one GT player
  /api/bench?source=&run=                          debugger: detector-benchmark boxes (benchmark frames only)
  /api/gallery, /gallery/<file>                    reproducible failure gallery (gallery-v2)
  /api/health
  POST /api/selftest     (only with --selftest-dir) browser self-test report, written to that folder

Video serving follows the M10.8 lab (whitelist + range) and the M7 demo (HTML5 video as the master clock).
"""

from __future__ import annotations

import argparse
import gzip
import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from football_fatigue.multicam_lab.media import mime_type, parse_range
from football_fatigue.video.player.data import Catalog, PlayerError, compare
from football_fatigue.video.sources import SourceError

PAGE = Path(__file__).with_name("player.html")


def make_handler(cat: Catalog, selftest_dir: Path | None = None):
    class H(BaseHTTPRequestHandler):
        server_version = "VideoPlayer/1.0"

        def log_message(self, fmt, *args):
            return

        def _send(self, body: bytes, ctype: str, code: int = 200, gz: bool = False, cache: str = "no-store") -> None:
            if gz and "gzip" in (self.headers.get("Accept-Encoding") or "") and len(body) > 1024:
                body = gzip.compress(body, 5)
                gz = True
            else:
                gz = False
            try:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", cache)
                if gz:
                    self.send_header("Content-Encoding", "gzip")
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)
            except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                pass                                        # the page navigated away; nothing to answer

        def _json(self, data, code: int = 200) -> None:
            self._send(json.dumps(data, separators=(",", ":")).encode(), "application/json", code, gz=True)

        def _err(self, code: int, msg: str) -> None:
            self._json({"error": msg}, code)

        def _video(self, path: Path) -> None:
            if not path.is_file():
                return self._err(404, f"video not found on this machine: {path.name}")
            size = path.stat().st_size
            rng = parse_range(self.headers.get("Range"), size)
            if rng == (size, size):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            start, end = rng if rng else (0, size - 1)
            self.send_response(206 if rng else 200)
            self.send_header("Content-Type", mime_type(path))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Cache-Control", "max-age=3600")
            if rng:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if self.command == "HEAD":
                return
            with open(path, "rb") as fh:
                fh.seek(start)
                left = end - start + 1
                try:
                    while left > 0:
                        chunk = fh.read(min(1 << 18, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
                except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                    pass                                    # the browser cancels range requests while seeking

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            u = urlparse(self.path)
            path = unquote(u.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if path in ("/", "/index.html"):
                    return self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
                if path.startswith("/media/"):
                    return self._video(cat.video_path(path[len("/media/"):]))
                if path == "/api/health":
                    return self._json({"ok": True, "data_root": str(cat.data_root), "out_root": str(cat.out_root),
                                       "selftest": selftest_dir is not None})
                if path == "/api/sources":
                    return self._json(cat.sources())
                if path == "/api/meta":
                    return self._json(cat.source_meta(q["source"]))
                if path == "/api/frames":
                    t0 = time.perf_counter()
                    d = cat.chunk(q["source"], int(q.get("start", 0)), int(q.get("n", 250)), q.get("tracker") or None,
                                  q.get("run") or None)
                    d["server_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                    return self._json(d)
                if path == "/api/events":
                    return self._json(cat.dev(q["source"]).events(q["run"]))
                if path == "/api/compare":
                    return self._json(compare(cat, q["source"]))
                if path == "/api/diag":
                    return self._json(cat.debug.diag(q["source"], q.get("profile", "T0")))
                if path == "/api/strip":
                    png = cat.debug.strip(q["source"], int(q["gid"]), [int(x) for x in q["frames"].split(",")],
                                          q.get("profile", "T0"), (q.get("labels") or "").split(",") if q.get("labels") else [])
                    return self._send(png, "image/png")
                if path == "/api/bench":
                    return self._json(cat.debug.bench(q["source"], q.get("run") or None))
                if path == "/api/gallery":
                    return self._json(cat.debug.gallery())
                if path.startswith("/gallery/"):
                    f = cat.debug.gallery_file(path[len("/gallery/"):])
                    return self._send(f.read_bytes(), "image/png" if f.suffix == ".png" else "video/mp4", cache="max-age=600")
                if path == "/api/thumb":
                    crop = tuple(int(q[k]) for k in ("x", "y", "cw", "ch")) if "cw" in q else None
                    return self._json(cat.thumb(q["source"], int(q["frame"]), int(q.get("w", 64)), int(q.get("h", 18)), crop))
                return self._err(404, f"no route {path}")
            except KeyError as exc:
                return self._err(400, f"missing parameter {exc}")
            except (PlayerError, SourceError, ValueError) as exc:
                return self._err(400, str(exc))
            except Exception as exc:                        # noqa: BLE001 - readable error instead of a dropped socket
                return self._err(500, f"{type(exc).__name__}: {exc}")

        def do_POST(self):
            n = min(int(self.headers.get("Content-Length") or 0), 4_000_000)
            body = self.rfile.read(n)                       # drain first: Windows resets sockets closed with unread data
            if urlparse(self.path).path != "/api/selftest" or selftest_dir is None:
                return self._err(404, "not found")
            selftest_dir.mkdir(parents=True, exist_ok=True)
            out = selftest_dir / f"selftest_{time.strftime('%Y%m%d-%H%M%S')}_{int(time.time() * 1000) % 1000:03d}.json"
            out.write_bytes(body)
            return self._json({"saved": str(out)})
    return H


def start(data_root: Path, out_root: Path, host: str = "127.0.0.1", port: int = 5831,
          selftest_dir: Path | None = None) -> ThreadingHTTPServer:
    cat = Catalog(Path(data_root).resolve(), Path(out_root).resolve())
    srv = ThreadingHTTPServer((host, port), make_handler(cat, selftest_dir))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True, name="video-player").start()
    return srv


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m football_fatigue.video.player",
                                 description="Native video player with synchronized detection/tracking/event overlays")
    ap.add_argument("--data-root", required=True, help="read-only data root (videos, caches, replays, ground truth)")
    ap.add_argument("--out-root", default="data/processed/video")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5831)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--selftest-dir", type=Path, help="enable POST /api/selftest (browser QA reports)")
    a = ap.parse_args(argv)
    srv = start(Path(a.data_root), Path(a.out_root), a.host, a.port, a.selftest_dir)
    url = f"http://{a.host}:{a.port}/"
    print(f"Video player at {url}  (Ctrl+C to stop)", flush=True)
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        srv.shutdown()
    return 0
