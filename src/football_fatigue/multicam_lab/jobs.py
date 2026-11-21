"""Bounded local processing jobs.

A job runs the *production* detector + ByteTrack (``python -m football_fatigue.vision --detection-only``, unchanged) on a
bounded clip of each camera, then optionally maps foot points to the pitch with a camera profile that matches the
recording, and (two cameras, both mapped, sync not manual) associates locations across cameras. Results become a lab
session. Originals are never modified; every job lives in its own directory with a reproducible record.
"""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from football_fatigue.multicam_lab import fusion
from football_fatigue.multicam_lab.geometry import Profile, apply_h
from football_fatigue.multicam_lab.media import FrameReader, probe_video

MAX_CLIP_S = 30.0
MAX_UPLOAD_BYTES = 4 * 1024 ** 3
WEIGHTS = Path("models/vision/player_detector.pt")
SYNC_STATES = ("same_video", "estimated_independent", "manual_unverified", "unknown")


@dataclass
class CameraInput:
    name: str                      # "main" | "fixed" | "moving"
    video: str
    profile: str | None = None     # profile file name, only applied when the fingerprint matches
    offset_s: float = 0.0          # source time = master time + offset_s (master = first camera)


@dataclass
class JobSpec:
    mode: str                      # "single_camera" | "two_camera"
    cameras: list[CameraInput]
    start_s: float
    duration_s: float
    confidence: float = 0.25
    imgsz: int = 640
    same_match_confirmed: bool = False
    sync_state: str = "unknown"
    sync_sigma_s: float | None = None
    calibration_variant: str | None = None

    def validate(self) -> list[str]:
        e = []
        if self.mode not in ("single_camera", "two_camera"):
            e.append(f"unknown mode {self.mode!r}")
        if self.mode == "single_camera" and len(self.cameras) != 1:
            e.append("single-camera mode needs exactly one video")
        if self.mode == "two_camera":
            if len(self.cameras) != 2:
                e.append("two-camera mode needs exactly two videos")
            if not self.same_match_confirmed:
                e.append("confirm that both recordings show the same match")
        if not (0 < self.duration_s <= MAX_CLIP_S):
            e.append(f"clip duration must be within (0, {MAX_CLIP_S:g}] s (bounded offline analysis)")
        if self.start_s < 0:
            e.append("start time must be >= 0")
        if not (0 < self.confidence <= 1):
            e.append("confidence must be within (0, 1]")
        if self.sync_state not in SYNC_STATES:
            e.append(f"sync state must be one of {SYNC_STATES}")
        return e


@dataclass
class Job:
    id: str
    spec: JobSpec
    status: str = "queued"          # queued | validating | detecting | mapping | associating | done | failed | cancelled
    stage: str = ""
    progress: float = 0.0
    created_utc: str = ""
    finished_utc: str | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    outputs: dict = field(default_factory=dict)
    reproduction: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class JobManager:
    def __init__(self, root: Path, profiles_dir: Path, repo_root: Path = Path.cwd()):
        self.root = Path(root); self.jobs_dir = self.root / "jobs"; self.uploads = self.root / "uploads"; self.profiles_dir = Path(profiles_dir)
        self.repo = Path(repo_root); self.jobs: dict[str, Job] = {}; self._lock = threading.Lock(); self._queue: list[str] = []
        self._procs: dict[str, subprocess.Popen] = {}; self._cancel: set[str] = set()
        for d in (self.jobs_dir, self.uploads):
            d.mkdir(parents=True, exist_ok=True)
        for p in sorted(self.jobs_dir.glob("*/job.json")):
            try:
                d = json.loads(p.read_text(encoding="utf-8")); spec = JobSpec(**{**d["spec"], "cameras": [CameraInput(**c) for c in d["spec"]["cameras"]]})
                j = Job(**{**d, "spec": spec})
                if j.status not in ("done", "failed", "cancelled"):
                    j.status, j.errors = "failed", j.errors + ["interrupted: the lab server stopped while this job was running"]
                self.jobs[j.id] = j
            except Exception:                                   # noqa: BLE001 - a broken record must not stop the server
                continue
        threading.Thread(target=self._worker, daemon=True).start()

    # ---------------- uploads
    def store_upload(self, filename: str, stream, length: int) -> dict:
        safe = "".join(ch for ch in Path(filename).name if ch.isalnum() or ch in "._-")[:120] or "upload.mp4"
        if length <= 0:
            raise ValueError("empty upload")
        if length > MAX_UPLOAD_BYTES:
            raise ValueError(f"file too large ({length / 1e9:.2f} GB > {MAX_UPLOAD_BYTES / 1e9:.0f} GB limit)")
        dest = self.uploads / f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{secrets.token_hex(3)}-{safe}"
        left = length
        with open(dest, "wb") as fh:
            while left > 0:
                chunk = stream.read(min(1 << 20, left))
                if not chunk:
                    break
                fh.write(chunk); left -= len(chunk)
        if left:
            dest.unlink(missing_ok=True); raise ValueError("upload interrupted before the whole file arrived")
        try:
            info = probe_video(dest)
        except ValueError:
            dest.unlink(missing_ok=True); raise
        return {"path": str(dest.relative_to(self.repo)) if dest.is_relative_to(self.repo) else str(dest), "probe": info}

    def profiles_for(self, video: Path) -> list[str]:
        out = []
        for p in sorted(self.profiles_dir.glob("*.json")):
            try:
                if Profile.load(p).matches(video):
                    out.append(p.name)
            except Exception:                                   # noqa: BLE001
                continue
        return out

    # ---------------- jobs
    def submit(self, spec: JobSpec) -> Job:
        errs = spec.validate()
        for c in spec.cameras:
            v = self.repo / c.video
            if not v.is_file():
                errs.append(f"video not found: {c.video}")
                continue
            try:
                info = probe_video(v)
            except ValueError as exc:
                errs.append(f"{c.name}: {exc}"); continue
            if info["duration_s"] is not None and spec.start_s + spec.duration_s > info["duration_s"] + c.offset_s + 1e-3:
                errs.append(f"{c.name}: requested range ends after the video ({info['duration_s']:.1f} s)")
        if errs:
            raise ValueError("; ".join(errs))
        jid = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"
        job = Job(id=jid, spec=spec, created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        with self._lock:
            self.jobs[jid] = job; self._queue.append(jid)
        self._save(job); return job

    def cancel(self, jid: str) -> bool:
        job = self.jobs.get(jid)
        if not job or job.status in ("done", "failed", "cancelled"):
            return False
        self._cancel.add(jid); p = self._procs.get(jid)
        if p and p.poll() is None:
            p.terminate()
        if job.status == "queued":
            with self._lock:
                if jid in self._queue:
                    self._queue.remove(jid)
            job.status = "cancelled"; self._save(job)
        return True

    def _save(self, job: Job) -> None:
        d = self.jobs_dir / job.id; d.mkdir(parents=True, exist_ok=True)
        (d / "job.json").write_text(json.dumps(job.to_dict(), indent=1), encoding="utf-8")

    def _worker(self) -> None:
        while True:
            jid = None
            with self._lock:
                if self._queue:
                    jid = self._queue.pop(0)
            if jid is None:
                time.sleep(0.3); continue
            job = self.jobs[jid]
            try:
                self._run(job)
                job.status = "cancelled" if jid in self._cancel else "done"
            except Exception as exc:                            # noqa: BLE001 - report every failure on the job
                job.status = "cancelled" if jid in self._cancel else "failed"
                if job.status == "failed":
                    job.errors.append(f"{type(exc).__name__}: {exc}"); job.reproduction["traceback"] = traceback.format_exc()[-4000:]
            job.finished_utc = datetime.now(timezone.utc).isoformat(timespec="seconds"); self._save(job)

    def _set(self, job: Job, status: str, stage: str, progress: float) -> None:
        job.status, job.stage, job.progress = status, stage, round(progress, 3); self._save(job)

    def _run(self, job: Job) -> None:
        spec = job.spec; jdir = self.jobs_dir / job.id; runs = jdir / "runs"; runs.mkdir(parents=True, exist_ok=True)
        if not WEIGHTS.is_file():
            raise FileNotFoundError(f"detector weights missing: {WEIGHTS} (no automatic download)")
        self._set(job, "validating", "probing inputs", 0.02)
        probes = {c.name: probe_video(self.repo / c.video) for c in spec.cameras}
        job.reproduction.update({"python": sys.version.split()[0], "weights": str(WEIGHTS), "probes": probes, "commands": []})
        per_cam: dict[str, dict] = {}
        n = len(spec.cameras)
        for ci, cam in enumerate(spec.cameras):
            info = probes[cam.name]; fps = info["fps"] or 30.0
            start_frame = max(0, int(round((spec.start_s + cam.offset_s) * fps)))
            cmd = [sys.executable, "-m", "football_fatigue.vision", str(self.repo / cam.video), "--weights", str(WEIGHTS), "--detection-only",
                   "--run-id", cam.name, "--output-root", str(runs), "--start-frame", str(start_frame), "--max-seconds", f"{spec.duration_s:g}",
                   "--device", "cpu", "--confidence", f"{spec.confidence:g}", "--imgsz", str(spec.imgsz)]
            job.reproduction["commands"].append(" ".join(cmd))
            self._set(job, "detecting", f"{cam.name}: production detector + ByteTrack", 0.05 + 0.75 * ci / n)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "YOLO_AUTOINSTALL": "False", "YOLO_OFFLINE": "True"}
            log = open(jdir / f"detect_{cam.name}.log", "w", encoding="utf-8")
            p = subprocess.Popen(cmd, cwd=self.repo, stdout=log, stderr=subprocess.STDOUT, env=env); self._procs[job.id] = p
            expected = max(1, int(round(spec.duration_s * fps))); frames_file = runs / cam.name / "frames.jsonl"
            while p.poll() is None:
                time.sleep(1.0)
                if frames_file.exists():
                    with open(frames_file, "rb") as fh:
                        done = sum(1 for _ in fh)
                    self._set(job, "detecting", f"{cam.name}: {done}/{expected} frames", 0.05 + 0.75 * (ci + min(1.0, done / expected)) / n)
            log.close()
            if job.id in self._cancel:
                raise RuntimeError("cancelled by user")
            if p.returncode != 0:
                tail = (jdir / f"detect_{cam.name}.log").read_text(encoding="utf-8", errors="replace")[-1500:]
                raise RuntimeError(f"production detector failed for {cam.name} (exit {p.returncode}): {tail}")
            per_cam[cam.name] = {"frames": [json.loads(l) for l in frames_file.read_text(encoding="utf-8").splitlines() if l.strip()],
                                 "fps": fps, "info": info, "cam": cam}
        self._set(job, "mapping", "applying camera profiles", 0.82)
        session = self._build_session(job, per_cam)
        (jdir / "session.json").write_text(json.dumps(session), encoding="utf-8")
        job.outputs = {"session": f"job-{job.id}", "session_file": str((jdir / "session.json").relative_to(self.repo)) if jdir.is_relative_to(self.repo) else str(jdir / "session.json"),
                       "runs": str(runs), "pitch_mapped": session["calibration"]["status"], "fusion": session["fusion"]["status"]}
        self._set(job, "done", "finished", 1.0)

    # ---------------- results -> session
    def _build_session(self, job: Job, per_cam: dict) -> dict:
        spec = job.spec; names = [c.name for c in spec.cameras]
        profiles: dict[str, Profile | None] = {}
        for c in spec.cameras:
            prof = None
            if c.profile:
                pp = self.profiles_dir / Path(c.profile).name
                if not pp.is_file():
                    job.warnings.append(f"{c.name}: profile {c.profile} not found; no pitch coordinates")
                else:
                    cand = Profile.load(pp)
                    if cand.matches(self.repo / c.video):
                        prof = cand
                    else:
                        job.warnings.append(f"{c.name}: profile {c.profile} does not match this recording (fingerprint); no pitch coordinates")
            profiles[c.name] = prof
        cams_out = {}; obs_by_cam: dict[str, dict[int, list]] = {}; H_by_cam: dict[str, dict[int, list]] = {}
        for c in spec.cameras:
            pc = per_cam[c.name]; prof = profiles[c.name]; reader = FrameReader(self.repo / c.video) if prof and prof.model == "rotating_pinhole" else None
            by_idx: dict[int, list] = {}; reg_ok = 0; h_by_idx: dict[int, list] = {}
            for fr in pc["frames"]:
                idx = int(fr["frame_num"]); Hm = None
                if prof is not None:
                    Hm = prof.pitch_to_image(spec.calibration_variant) if prof.model == "static_homography" else prof.pitch_to_image(frame_bgr=reader.frame(idx))
                    reg_ok += Hm is not None
                    if Hm is not None:
                        h_by_idx[idx] = np.round(Hm, 8).tolist()
                obs = []
                for k, pl in enumerate(fr.get("players", [])):
                    b = pl["bbox_xyxy_px"]; foot = ((b[0] + b[2]) / 2, b[3]); o = {"k": k, "box": [round(v, 1) for v in b], "cls": (pl.get("detected_class") or "?")[0],
                                                                                    "conf": round(float(pl.get("confidence") or 0), 3), "track": {"bytetrack_prod": pl.get("track_id")},
                                                                                    "p": None, "cv": None}
                    if Hm is not None:
                        r = prof.observe(foot, Hm)
                        if r is not None:
                            o["p"] = [round(float(v), 2) for v in r[0]]; o["cv"] = [round(float(r[1][0, 0]), 3), round(float(r[1][0, 1]), 3), round(float(r[1][1, 1]), 3)]
                    obs.append(o)
                by_idx[idx] = obs
            if reader:
                reader.close()
            obs_by_cam[c.name] = by_idx; H_by_cam[c.name] = h_by_idx
            cams_out[c.name] = {"video": c.video, "fps": pc["fps"], "frame_count": pc["info"]["frame_count"], "offset_s": c.offset_s,
                                "first_pts_s": pc["info"]["first_pts_s"], "profile": prof.data.get("name") if prof else None,
                                "pitch_mapped_frames": reg_ok, "processed_frames": len(pc["frames"])}
        master = names[0]; fps_m = per_cam[master]["fps"]
        mapped = all(profiles[n] is not None for n in names)
        fusion_ok = spec.mode == "two_camera" and mapped and spec.sync_state in ("same_video", "estimated_independent")
        fusion_reason = ("single camera" if spec.mode == "single_camera" else
                         "blocked: both cameras need a matching calibration profile" if not mapped else
                         "blocked: synchronisation is manual/unverified" if not fusion_ok else "computed (experimental)")
        unc = fusion.Uncertainty(sync_sigma_s=spec.sync_sigma_s or 0.15)
        instants = []
        for idx in sorted(obs_by_cam[master]):
            T = round(idx / fps_m - spec.cameras[0].offset_s, 4); inst = {"T": T, "cams": {}, "obs": {}, "links": []}
            for c in spec.cameras:
                fps = per_cam[c.name]["fps"]; want = T + c.offset_s; j = int(round(want * fps)); t_src = j / fps
                have = j in obs_by_cam[c.name]
                inst["cams"][c.name] = {"idx": j, "t_src": round(t_src, 4), "residual_s": round(t_src - want, 4), "status": "ok" if have else "no-observation"}
                if j in H_by_cam[c.name]:
                    inst["cams"][c.name]["H"] = H_by_cam[c.name][j]
                inst["obs"][c.name] = [dict(o) for o in obs_by_cam[c.name].get(j, [])]
            if fusion_ok:
                a, b = names
                A = [{"k": o["k"], "pitch": o["p"], "cov": [[o["cv"][0], o["cv"][1]], [o["cv"][1], o["cv"][2]]]} for o in inst["obs"][a] if o["p"] is not None]
                B = [{"k": o["k"], "pitch": o["p"], "cov": [[o["cv"][0], o["cv"][1]], [o["cv"][1], o["cv"][2]]]} for o in inst["obs"][b] if o["p"] is not None]
                inst["links"] = [{"cams": [a, b], "ka": l["m"], "kb": l["f"], **{k: v for k, v in l.items() if k not in ("m", "f")}} for l in fusion.associate(A, B, unc)]
            for c in spec.cameras:                               # per-observation state
                side = 0 if c.name == names[0] else 1
                for o in inst["obs"][c.name]:
                    l = next((x for x in inst["links"] if (x["ka"] if side == 0 else x["kb"]) == o["k"]), None)
                    o["state"] = ("observed (no pitch position)" if o["p"] is None else
                                  ("ambiguous" if l["ambiguous"] else "matched") if l else
                                  (f"{c.name}-only" if spec.mode == "two_camera" and fusion_ok else "observed (single camera)"))
            instants.append(inst)
        pitch = next((p.data.get("pitch") for p in profiles.values() if p is not None), None)
        return {"id": f"job-{job.id}", "title": f"Job {job.id} ({spec.mode.replace('_', ' ')})", "kind": spec.mode, "source": "fresh processing job", "pitch": pitch,
                "cameras": cams_out, "camera_order": names, "master": {"camera": master, "step_s": round(1 / fps_m, 6)},
                "sync": {"state": spec.sync_state, "offsets_s": {c.name: c.offset_s for c in spec.cameras}, "sigma_s": spec.sync_sigma_s},
                "calibration": {"status": "mapped" if mapped else ("partial" if any(profiles.values()) else "none"),
                                "profiles": {n: (profiles[n].data.get("name") if profiles[n] else None) for n in names},
                                "note": "No pitch coordinates are produced without a calibration profile that matches the recording."},
                "fusion": {"status": fusion_reason}, "detector": {"weights": str(WEIGHTS), "confidence": spec.confidence, "imgsz": spec.imgsz,
                                                                   "tracker": "production ByteTrack (football_fatigue.vision)"},
                "instants": instants, "job_id": job.id}
