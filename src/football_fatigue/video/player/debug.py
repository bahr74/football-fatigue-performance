"""Read-only data for the player's "Player Detection & Identity Debugger" (development matches only).

* ``outcomes``  per-frame pipeline outcome of every GT player for one profile (``_diag/pipeline-v1``), merged into the
  overlay chunk as layer ``o`` = [gt id, code index, temporary track id | null, matched cached box conf | null].
* ``diag``      failure episodes, temporary-ID changes and view exits of one profile, plus the code legend.
* ``strip``     a PNG of before / during / after crops around one GT player, drawn on exactly decoded frames.
* ``bench``     boxes of a completed detector-benchmark run on its benchmark frames (detector comparison).
* ``gallery``   the reproducible failure gallery index (``_diag/gallery-v2``).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

from football_fatigue.video.detection.framestore import LineIndex
from football_fatigue.video.failures.pipeline import CODE_TEXT, CODES, GROUP, out_dir
from football_fatigue.video.sources import SOURCES, ExactFrameReader, guard_not_reserved

CODE_HEX = {"ok": "#4ade80", "id_changed": "#facc15", "untracked": "#a78bfa", "rej_conf": "#fb923c",
            "rej_turf": "#f472b6", "rej_class": "#94a3b8", "miss": "#ef4444", "not_evaluable": "#64748b"}
GALLERY_VERSION = "gallery-v2"


class Debug:
    def __init__(self, data_root: Path, out_root: Path) -> None:
        self.data_root, self.out_root = Path(data_root), Path(out_root)
        self._idx: dict[tuple[str, str], LineIndex | None] = {}
        self._diag: dict[tuple[str, str], dict] = {}
        self._reader: tuple[str, ExactFrameReader] | None = None
        self._lines: dict[tuple[str, str], LineIndex] = {}
        self._lock = threading.Lock()

    def _line(self, path: Path, key: str) -> LineIndex:
        k = (str(path), key)
        with self._lock:
            if k not in self._lines:
                self._lines[k] = LineIndex(path, key)
            return self._lines[k]

    def _check(self, sid: str) -> None:
        if sid not in SOURCES:
            raise ValueError(f"unknown development source {sid!r}")
        spec = SOURCES[sid]
        guard_not_reserved(sid, spec["match_id"], spec["gt"])

    # ------------------------------------------------------------------ per-frame outcomes
    def outcomes(self, sid: str, profile: str, frame: int) -> list | None:
        key = (sid, profile)
        with self._lock:
            if key not in self._idx:
                p = out_dir(self.out_root, sid) / profile / "frames.jsonl"
                self._idx[key] = LineIndex(p, "frame") if p.is_file() else None
            idx = self._idx[key]
        if idx is None:
            return None
        r = idx.get(frame)
        return r["o"] if r else []

    # ------------------------------------------------------------------ episodes
    def diag(self, sid: str, profile: str) -> dict[str, Any]:
        self._check(sid)
        key = (sid, profile)
        with self._lock:
            if key in self._diag:
                return self._diag[key]
        base = out_dir(self.out_root, sid)
        ep, sm, mf = base / profile / "episodes.json", base / profile / "summary.json", base / "manifest.json"
        if not ep.exists():
            out = {"available": False, "profile": profile,
                   "reason": f"no pipeline decomposition for {sid} / {profile} "
                             f"(python -m football_fatigue.video.failures decompose --source {sid})"}
        else:
            e = json.loads(ep.read_text(encoding="utf-8"))
            out = {"available": True, "profile": profile, "codes": list(CODES), "code_text": CODE_TEXT, "groups": GROUP,
                   "colours": CODE_HEX, "episodes": e["episodes"], "id_changes": e["id_changes"],
                   "exits": e.get("exits", []), "summary": json.loads(sm.read_text(encoding="utf-8")),
                   "manifest": {k: v for k, v in json.loads(mf.read_text(encoding="utf-8")).items()
                                if k in ("version", "created_at_utc", "frames", "params", "limitations", "inputs")}}
        with self._lock:
            self._diag[key] = out
        return out

    # ------------------------------------------------------------------ before / during / after strip
    def _frame(self, sid: str, f: int):
        with self._lock:
            if self._reader is None or self._reader[0] != sid:
                self._reader = (sid, ExactFrameReader(self.data_root / SOURCES[sid]["video"], cache_frames=4))
            return self._reader[1].frame(int(f))

    def strip(self, sid: str, gid: int, frames: list[int], profile: str, labels: list[str]) -> bytes:
        """PNG: one crop per frame centred on GT player ``gid``, GT boxes (white, G##), the profile's tracker output
        (cyan, T##), the cached boxes (thin, conf) and the player's pipeline outcome as a caption."""
        import cv2

        from football_fatigue.video.tracking import tracks_dir

        self._check(sid)
        frames = [int(f) for f in frames][:6]
        gt, _ = _gt_cached(str(self.data_root), sid)
        tix = self._line(tracks_dir(self.data_root, self.out_root, sid, profile) / "frames.jsonl", "frame_num")
        from football_fatigue.video.detection.diagnose import cache_dir

        cache = self._line(cache_dir(self.data_root, sid) / "detections.jsonl", "frame_num")
        cols = json.loads((cache_dir(self.data_root, sid) / "cache_manifest.json").read_text(encoding="utf-8"))["columns"]
        boxes = {f: dict(gt.get(f, [])).get(gid) for f in frames}
        known = [b for b in boxes.values() if b is not None]
        if not known:
            raise ValueError(f"GT player {gid} is not annotated in frames {frames}")
        centre = known[len(known) // 2]
        h = max(40.0, centre[3] - centre[1])
        out = []
        for f, lab in zip(frames, labels + [""] * len(frames)):
            img = self._frame(sid, f)
            if img is None:
                continue
            img = img.copy()
            b = boxes[f] or centre
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            x1, y1 = int(max(0, cx - 3.2 * h)), int(max(0, cy - 2.0 * h))
            x2, y2 = int(min(img.shape[1], cx + 3.2 * h)), int(min(img.shape[0], cy + 2.0 * h))
            rec = cache.get(f)
            for row in (rec or {}).get("observations", []):
                o = dict(zip(cols, row)) if isinstance(row, list) else row
                cv2.rectangle(img, (int(o["x1"]), int(o["y1"])), (int(o["x2"]), int(o["y2"])), (120, 120, 120), 1)
                cv2.putText(img, f"{float(o['confidence']):.2f}", (int(o["x1"]), int(o["y2"]) + 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.33, (170, 170, 170), 1, cv2.LINE_AA)
            for g_id, gb in gt.get(f, []):
                c = (255, 255, 255) if g_id == gid else (200, 200, 200)
                cv2.rectangle(img, (int(gb[0]), int(gb[1])), (int(gb[2]), int(gb[3])), c, 2 if g_id == gid else 1)
                cv2.putText(img, f"G{g_id}", (int(gb[0]), int(gb[3]) + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1, cv2.LINE_AA)
            for p in (tix.get(f) or {"players": []})["players"]:
                tb = p["bbox_xyxy_px"]
                cv2.rectangle(img, (int(tb[0]), int(tb[1])), (int(tb[2]), int(tb[3])), (238, 211, 34), 2)
                cv2.putText(img, f"T{p['track_id']}", (int(tb[0]), int(tb[1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (238, 211, 34), 1, cv2.LINE_AA)
            crop = img[y1:y2, x1:x2]
            s = 300 / max(1, crop.shape[0])
            crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), 300), interpolation=cv2.INTER_LINEAR)
            oc = self.outcomes(sid, profile, f) or []
            mine = next((r for r in oc if r[0] == gid), None)
            code = CODES[mine[1]] if mine else "not annotated"
            bar = np.full((40, crop.shape[1], 3), 20, np.uint8)
            cv2.putText(bar, f"{lab} frame {f}", (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
            hexc = CODE_HEX.get(code, "#cccccc").lstrip("#")
            col = (int(hexc[4:6], 16), int(hexc[2:4], 16), int(hexc[0:2], 16))
            txt = f"G{gid}: {code}" + (f" T{mine[2]}" if mine and mine[2] is not None else "") + \
                  (f" conf {mine[3]:.2f}" if mine and mine[3] is not None else "")
            cv2.putText(bar, txt, (6, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
            out.append(np.vstack([bar, crop]))
        hh = max(o.shape[0] for o in out)
        row = np.hstack([cv2.copyMakeBorder(o, 0, hh - o.shape[0], 0, 4, cv2.BORDER_CONSTANT, value=(12, 12, 12)) for o in out])
        ok, buf = cv2.imencode(".png", row)
        return buf.tobytes()

    # ------------------------------------------------------------------ detector benchmark boxes
    def bench(self, sid: str, run: str | None) -> dict[str, Any]:
        from football_fatigue.video.bench.detectors import load_dets, load_runs

        self._check(sid)
        runs = load_runs(self.out_root)
        done = {k: v for k, v in runs.items() if v.get("status") == "complete" and v.get("subset") is None}
        listing = [{"run": k, "detector": v["detector"], "protocol": v["protocol"], "device": v["device"]} for k, v in sorted(done.items())]
        if not run:
            return {"runs": listing}
        if run not in done:
            raise ValueError(f"no completed benchmark run {run!r}")
        d = load_dets(self.out_root, run)
        prod = load_dets(self.out_root, "yolo26l-football__production-cache") if "yolo26l-football__production-cache" in done else {}
        frames = {}
        for (s, f), rec in d.items():
            if s != sid or (s, f) not in prod:
                continue
            frames[f] = [[round(b[0], 1), round(b[1], 1), round(b[2], 1), round(b[3], 1), round(b[4], 3), b[6] or b[5]]
                         for b in rec["boxes"]]
        return {"runs": listing, "run": run, "frames": frames, "provenance": {k: done[run][k] for k in ("detector", "protocol", "device")}
                | {"checkpoint_sha256": done[run].get("checkpoint_sha256_verified"), "frame_manifest": done[run]["frame_manifest"]},
                "note": "boxes only on benchmark frames where this run AND the production run are complete"}

    # ------------------------------------------------------------------ gallery
    def gallery(self) -> dict[str, Any]:
        p = self.out_root / "_diag" / GALLERY_VERSION / "index.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"items": [], "missing": str(p)}

    def gallery_file(self, name: str) -> Path:
        base = (self.out_root / "_diag" / GALLERY_VERSION).resolve()
        p = (base / name).resolve()
        if base not in p.parents or p.suffix.lower() not in (".png", ".mp4"):
            raise ValueError("not a gallery file")
        return p


_GT: dict[str, Any] = {}


def _gt_cached(data_root: str, sid: str):
    from football_fatigue.vision.mot_eval import load_reference

    if sid not in _GT:
        _GT.clear()
        _GT[sid] = load_reference(Path(data_root) / SOURCES[sid]["gt"], 0)
    return _GT[sid]
