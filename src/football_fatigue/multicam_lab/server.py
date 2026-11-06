"""Local HTTP server of the multi-camera lab (stdlib only).

Routes (all read-only on originals):
  /, /review, /sync                 pages (static/)
  /static/<file>                    page assets
  /media/<key>                      whitelisted videos/images with HTTP range support (206), correct MIME types
  /api/sessions, /api/session/<id>  prepared sessions (cached experiment outputs) and finished job sessions
  /api/frame?video=<key>&idx=<n>    exact source frame (JPEG) decoded by index; headers carry the frame time
  /api/player, /api/heatmap         player analytics from observed positions only
  /api/research                     study tables for Research mode
  /api/videos, /api/profiles        processing inputs; POST /api/upload stores a validated video
  /api/jobs[/<id>[/cancel]]         bounded processing jobs (production detector + ByteTrack)
  /api/review/items, /api/review    benchmark review: provisional labels (read-only) and an append-only human review log
  /api/sync-review                  append-only manual sync decisions
"""

from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from football_fatigue.multicam_lab import analytics
from football_fatigue.multicam_lab.jobs import CameraInput, JobManager, JobSpec
from football_fatigue.multicam_lab.media import FrameReader, mime_type, parse_range, probe_video

STATIC = Path(__file__).parent / "static"
M108 = Path("data/processed/vision_eval/m108-unified-multicamera-analytics-v1")
LAB = Path("data/processed/multicam_lab")
RAW = Path("data/external/wadi-degla/raw")


class Lab:
    def __init__(self, repo: Path, m108: Path, lab_root: Path):
        self.repo = repo; self.m108 = m108; self.sessions_dir = m108 / "results/sessions"; self.lab_root = lab_root
        self.jobs = JobManager(lab_root, m108 / "results/profiles", repo); self.reviews = lab_root / "reviews"; self.reviews.mkdir(parents=True, exist_ok=True)
        self._readers: dict[str, FrameReader] = {}; self._rlock = threading.Lock(); self._sessions: dict[str, dict] = {}
        media = json.loads((m108 / "results/media_manifest.json").read_text(encoding="utf-8")) if (m108 / "results/media_manifest.json").exists() else {}
        self.media = {k: Path(v["path"]) for k, v in media.items()}
        self.media_meta = media

    # videos that may be decoded/streamed: the two originals of the pair, uploads, and job inputs
    def videos(self) -> dict[str, Path]:
        v = {"wadi_degla_moving": RAW / "wadi_degla_moving.mp4", "wadi_degla_fixed_same_match": RAW / "wadi_degla_fixed_same_match.mp4"}
        for p in sorted(self.jobs.uploads.glob("*")):
            if p.is_file():
                v[f"upload:{p.name}"] = p
        return {k: p for k, p in v.items() if (self.repo / p).is_file() or p.is_file()}

    def video_key(self, path: str) -> str | None:
        rp = (self.repo / path).resolve()
        return next((k for k, p in self.videos().items() if (self.repo / p).resolve() == rp), None)

    def reader(self, key: str) -> FrameReader | None:
        vids = self.videos()
        if key not in vids:
            return None
        with self._rlock:
            if key not in self._readers:
                self._readers[key] = FrameReader(self.repo / vids[key])
            return self._readers[key]

    def session_list(self) -> list[dict]:
        out = []
        idx = self.sessions_dir / "index.json"
        if idx.exists():
            out += json.loads(idx.read_text(encoding="utf-8"))
        for j in sorted(self.jobs.jobs.values(), key=lambda j: j.created_utc, reverse=True):
            if j.status == "done" and j.outputs.get("session"):
                out.append({"id": j.outputs["session"], "title": f"Job {j.id} · {j.spec.mode.replace('_', ' ')} · {j.spec.duration_s:g} s from {j.spec.start_s:g} s",
                            "group": "Processing jobs (fresh runs)", "kind": j.spec.mode, "source": "fresh processing job"})
        return out

    def session(self, sid: str) -> dict | None:
        if sid in self._sessions:
            return self._sessions[sid]
        if sid.startswith("job-"):
            p = self.jobs.jobs_dir / sid[4:] / "session.json"
        else:
            p = self.sessions_dir / f"{Path(sid).name}.json"
        if not p.is_file():
            return None
        s = json.loads(p.read_text(encoding="utf-8"))
        for name, cam in s["cameras"].items():
            cam["video_key"] = self.video_key(cam["video"])
        self._sessions[sid] = s
        return s


def player_payload(s: dict, cam: str, tracker: str, tid: int) -> dict:
    """All observations of one short-term track in a session, with trajectory segments, heatmap and metrics."""
    obs = []; prev_partner = None
    other = [c for c in s["camera_order"] if c != cam]
    for inst in s["instants"]:
        for o in inst["obs"].get(cam, []):
            if o.get("track", {}).get(tracker) != tid:
                continue
            link = next((l for l in inst.get("links", []) if (l["cams"][0] == cam and l["ka"] == o["k"]) or (l["cams"][1] == cam and l["kb"] == o["k"])), None)
            partner = None
            if link:
                ok = link["kb"] if link["cams"][0] == cam else link["ka"]; oc = other[0]
                po = next((x for x in inst["obs"].get(oc, []) if x["k"] == ok), None)
                partner = (po or {}).get("track", {}).get("bytetrack_prod")
            brk = prev_partner is not None and partner is not None and partner != prev_partner
            obs.append({"t": inst["T"], "p": o.get("p"), "conf": o.get("conf"), "state": o.get("state", "observed"), "k": o["k"],
                        "partner_track": partner, "link_ambiguous": bool(link and link["ambiguous"]), "break_before": brk,
                        "fused": link["fused"] if link else None})
            if partner is not None:
                prev_partner = partner
    step = s["master"]["step_s"]; interval = s["instants"][-1]["T"] - s["instants"][0]["T"] + step if s["instants"] else 0   # each instant covers one step
    pts = [x for x in obs if x["p"] is not None]
    segs = analytics.trajectory_segments([{"t": x["t"], "p": x["p"], "break_before": x["break_before"]} for x in pts], max_gap_s=2.5 * step if step > 0.1 else 0.45)
    pitch = s.get("pitch") or {}
    hm = analytics.heatmap([tuple(x["p"]) for x in pts], (pitch.get("x_min", -62), pitch.get("x_max", 62)), (pitch.get("y_far", -31) - 2, pitch.get("y_near", 36) + 2), 3.0) if pts else None
    return {"camera": cam, "tracker": tracker, "track_id": tid, "observations": obs, "segments": [[{"t": p["t"], "p": p["p"]} for p in sg] for sg in segs],
            "heatmap": hm, "metrics": analytics.player_metrics(obs, interval, step), "interval_s": round(interval, 3), "step_s": step,
            "identity_note": "Short-term track of one camera. It restarts in every clip and is not a verified player identity."}


def heatmap_all(s: dict) -> dict:
    """Observed-position density of all on-pitch people: one point per linked pair (fused) plus each unlinked observation."""
    pts = []; n_inst = 0
    for inst in s["instants"]:
        n_inst += 1; linked = set()
        for l in inst.get("links", []):
            pts.append(tuple(l["fused"])); linked.add((l["cams"][0], l["ka"])); linked.add((l["cams"][1], l["kb"]))
        for cam, obs in inst["obs"].items():
            for o in obs:
                if o.get("p") is None or (cam, o["k"]) in linked or str(o.get("state", "")).startswith("off-pitch"):
                    continue
                pts.append(tuple(o["p"]))
    pitch = s.get("pitch") or {}
    hm = analytics.heatmap(pts, (pitch.get("x_min", -62), pitch.get("x_max", 62)), (pitch.get("y_far", -31) - 2, pitch.get("y_near", 36) + 2), 3.0)
    hm["instants"] = n_inst
    hm["note"] = "Observed positions only (both cameras, linked pairs counted once). Areas outside both camera views cannot appear."
    return hm


def make_handler(lab: Lab):
    class H(BaseHTTPRequestHandler):
        server_version = "MulticamLab/1.0"

        def log_message(self, fmt, *args):            # quiet; errors still reach stderr through send_error paths
            return

        # ---- helpers
        def _json(self, data, code: int = 200):
            b = json.dumps(data).encode(); self.send_response(code)
            self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.send_header("Cache-Control", "no-store")
            self.end_headers(); self.wfile.write(b)

        def _err(self, code: int, msg: str):
            self._json({"error": msg}, code)

        def _file(self, path: Path, cache: bool = True):
            if not path.is_file():
                return self._err(404, f"file not found: {path.name}")
            size = path.stat().st_size; rng = parse_range(self.headers.get("Range"), size); ctype = mime_type(path)
            if rng == (size, size):
                self.send_response(416); self.send_header("Content-Range", f"bytes */{size}"); self.end_headers(); return
            start, end = rng if rng else (0, size - 1)
            self.send_response(206 if rng else 200); self.send_header("Content-Type", ctype); self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1)); self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
            if rng:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if self.command == "HEAD":
                return
            with open(path, "rb") as fh:
                fh.seek(start); left = end - start + 1
                try:
                    while left > 0:
                        chunk = fh.read(min(1 << 18, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk); left -= len(chunk)
                except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                    pass                                    # browsers cancel range requests while seeking

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        # ---- GET
        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            u = urlparse(self.path); path = unquote(u.path); q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if path in ("/", "/index.html"):
                    return self._file(STATIC / "index.html", cache=False)
                if path in ("/review", "/review.html"):
                    return self._file(STATIC / "review.html", cache=False)
                if path in ("/sync", "/sync.html"):
                    return self._file(STATIC / "sync.html", cache=False)
                if path.startswith("/static/"):
                    p = (STATIC / path[len("/static/"):]).resolve()
                    return self._file(p, cache=False) if p.is_relative_to(STATIC.resolve()) else self._err(403, "forbidden")
                if path.startswith("/media/"):
                    key = path[len("/media/"):]
                    if key not in lab.media:
                        return self._err(404, f"unknown media item {key!r} (only whitelisted assets are served)")
                    return self._file(lab.repo / lab.media[key])
                if path == "/api/media":
                    return self._json({k: {**v, "available": (lab.repo / v["path"]).is_file()} for k, v in lab.media_meta.items()})
                if path == "/api/sessions":
                    return self._json(lab.session_list())
                if path.startswith("/api/session/"):
                    s = lab.session(path.rsplit("/", 1)[1])
                    return self._json(s) if s else self._err(404, "unknown session")
                if path == "/api/frame":
                    r = lab.reader(q.get("video", ""))
                    if r is None:
                        return self._err(404, "unknown video key")
                    idx = int(q.get("idx", -1)); jpg = r.jpeg(idx, max_width=int(q["w"]) if q.get("w") else None)
                    if jpg is None:
                        return self._err(404, f"frame {idx} not available")
                    self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.send_header("Content-Length", str(len(jpg)))
                    self.send_header("X-Frame-Index", str(idx)); self.send_header("X-Source-Time", f"{idx / r.fps:.4f}")
                    self.send_header("Cache-Control", "max-age=86400"); self.end_headers(); self.wfile.write(jpg); return
                if path == "/api/player":
                    s = lab.session(q.get("sid", ""))
                    if not s:
                        return self._err(404, "unknown session")
                    return self._json(player_payload(s, q["cam"], q.get("tracker", "bytetrack_prod"), int(q["tid"])))
                if path == "/api/heatmap":
                    s = lab.session(q.get("sid", ""))
                    return self._json(heatmap_all(s)) if s else self._err(404, "unknown session")
                if path == "/api/research":
                    p = lab.m108 / "results/research.json"
                    return self._json(json.loads(p.read_text(encoding="utf-8")) if p.exists() else {})
                if path == "/api/videos":
                    out = []
                    for k, p in lab.videos().items():
                        try:
                            info = probe_video(lab.repo / p)
                        except ValueError as exc:
                            info = {"error": str(exc)}
                        out.append({"key": k, "path": str(p).replace("\\", "/"), "probe": info, "profiles": lab.jobs.profiles_for(lab.repo / p)})
                    return self._json(out)
                if path == "/api/jobs":
                    return self._json([j.to_dict() for j in sorted(lab.jobs.jobs.values(), key=lambda j: j.created_utc, reverse=True)])
                if path.startswith("/api/jobs/"):
                    j = lab.jobs.jobs.get(path.split("/")[3])
                    return self._json(j.to_dict()) if j else self._err(404, "unknown job")
                if path == "/api/review/items":
                    p = lab.m108 / "results/benchmark/review_items.json"
                    return self._json(json.loads(p.read_text(encoding="utf-8")) if p.exists() else [])
                if path == "/api/review":
                    p = lab.reviews / "benchmark_reviews_v1.jsonl"
                    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []
                    return self._json(rows)
                if path == "/api/sync-review":
                    p = lab.reviews / "sync_reviews_v1.jsonl"
                    return self._json([json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else [])
                return self._err(404, f"no route {path}")
            except Exception as exc:                        # noqa: BLE001 - return a readable error instead of a dropped socket
                return self._err(500, f"{type(exc).__name__}: {exc}")

        # ---- POST
        def do_POST(self):
            u = urlparse(self.path); path = unquote(u.path)
            try:
                if path == "/api/upload":
                    name = unquote(self.headers.get("X-Filename") or "upload.mp4"); n = int(self.headers.get("Content-Length") or 0)
                    return self._json(lab.jobs.store_upload(name, self.rfile, n))
                if path == "/api/jobs":
                    d = self._body(); cams = [CameraInput(**c) for c in d.pop("cameras", [])]
                    return self._json(lab.jobs.submit(JobSpec(cameras=cams, **d)).to_dict())
                if path.startswith("/api/jobs/") and path.endswith("/cancel"):
                    return self._json({"cancelled": lab.jobs.cancel(path.split("/")[3])})
                if path in ("/api/review", "/api/sync-review"):
                    d = self._body()
                    if not str(d.get("reviewer", "")).strip():
                        return self._err(400, "reviewer name is required (provenance)")
                    d["saved_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                    d["provenance"] = "human review in the M10.8 lab" if path == "/api/review" else "manual sync review in the M10.8 lab"
                    p = lab.reviews / ("benchmark_reviews_v1.jsonl" if path == "/api/review" else "sync_reviews_v1.jsonl")
                    with open(p, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(d) + "\n")
                    return self._json({"saved": True, "file": str(p).replace("\\", "/")})
                return self._err(404, f"no route {path}")
            except ValueError as exc:
                return self._err(400, str(exc))
            except Exception as exc:                        # noqa: BLE001
                return self._err(500, f"{type(exc).__name__}: {exc}")
    return H


def run(host: str = "127.0.0.1", port: int = 5810, open_browser: bool = True, repo: Path = Path.cwd(), m108: Path = M108, lab_root: Path = LAB) -> None:
    lab = Lab(repo, m108, lab_root); srv = ThreadingHTTPServer((host, port), make_handler(lab)); url = f"http://{host}:{port}/"
    print(f"Multi-camera lab (EXPERIMENTAL) at {url}  - Ctrl+C to stop", flush=True)
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Experimental multi-camera analysis lab (M10.8)")
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=5810); ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--m108-dir", type=Path, default=M108); ap.add_argument("--lab-root", type=Path, default=LAB)
    a = ap.parse_args(); run(a.host, a.port, not a.no_browser, Path.cwd(), a.m108_dir, a.lab_root)
