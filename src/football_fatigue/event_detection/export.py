"""Write detected events in a PFF-like JSON shape plus a run manifest (derived data; never under the raw root).

The record layout follows the PFF event files where a field has a direct equivalent (``gameId``, ``eventTime``,
``gameEvents.period/teamId/playerId``, ``possessionEvents.possessionEventType/passOutcomeType/...``, ``ball``) so the same
consumers (playback, analytics) can read either file. Detector-only information lives under ``detector``.
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .classify import CROSS, PASS, SHOT, DetectedEvent
from .config import DETECTOR_VERSION, DetectorConfig
from .inputs import MatchPositions


def _pid(pos: MatchPositions, col: int | None) -> int | None:
    if col is None:
        return None
    v = int(pos.player_id[col])
    return v if v >= 0 else None


def to_pff_like(pos: MatchPositions, events: Sequence[DetectedEvent], version: str = DETECTOR_VERSION) -> list[dict[str, Any]]:
    out = []
    for i, e in enumerate(events):
        t = e.touch
        actor = _pid(pos, t.player)
        pe: dict[str, Any] = {"possessionEventType": e.type}
        if e.type == PASS:
            pe.update(passerPlayerId=actor, receiverPlayerId=_pid(pos, e.receiver), passOutcomeType=e.outcome)
        elif e.type == CROSS:
            pe.update(crosserPlayerId=actor, receiverPlayerId=_pid(pos, e.receiver), crossOutcomeType=e.outcome)
        elif e.type == SHOT:
            pe.update(shooterPlayerId=actor, shotOutcomeType=e.outcome)
        z = pos.ball[t.frame, 2]
        out.append({
            "gameId": pos.match_id,
            "detectedEventId": i,
            "eventTime": round(t.t_s, 3),
            "frameNum": int(pos.frame_num[t.frame]),
            "gameEvents": {"gameEventType": "G" if e.is_goal else "OTB", "period": t.period, "teamId": pos.team_id_of_side(e.team_side),
                           "playerId": actor, "homeTeam": e.team_side == 0},
            "possessionEvents": pe,
            "ball": [{"x": round(t.x, 2), "y": round(t.y, 2), "z": None if np.isnan(z) else round(float(z), 2)}],
            "detector": {"version": version, "keyPass": e.key_pass, "confidence": e.confidence, "playerKey": pos.player_key[t.player],
                         "receiverKey": pos.player_key[e.receiver] if e.receiver is not None else None,
                         "endTime": round(e.end_touch.t_s, 3) if e.end_touch is not None else None, "reason": e.reason},
        })
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest(pos: MatchPositions, cfg: DetectorConfig, sources: dict[str, Path], ball_stats: dict[str, int], n_touches: int,
             n_events: int, version: str = DETECTOR_VERSION, model: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "match_id": pos.match_id,
        "detector_version": version,
        "model": model,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input": {"source": pos.source, "frames": pos.n_frames, "excluded_periods": list(pos.excluded_periods), "notes": pos.notes,
                  "leakage_controls": ["raw ball only (ballsSmoothed is built from event labels)", "tracking event overlays not read",
                                       "event labels used for evaluation only"]},
        "sources": {k: {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)} for k, p in sources.items()},
        "parameters": cfg.to_dict(),
        "ball_cleaning": ball_stats,
        "counts": {"touches": n_touches, "events": n_events},
        "software": {"python": platform.python_version(), "numpy": np.__version__},
    }


def write_run(out_root: Path, raw_root: Path, match_id: int, records: list[dict], run_manifest: dict, *, overwrite: bool = False,
              version: str = DETECTOR_VERSION) -> Path:
    """Write ``<out_root>/<match>/<detector version>/events.json`` and ``manifest.json``; refuses the raw tree and existing runs."""
    out_root, raw_root = Path(out_root).resolve(), Path(raw_root).resolve()
    if out_root == raw_root or raw_root in out_root.parents:
        raise ValueError(f"output {out_root} lies inside the raw data root {raw_root}")
    dest = out_root / str(match_id) / version
    if (dest / "events.json").exists() and not overwrite:
        raise FileExistsError(f"a run already exists (use --overwrite): {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    for name, obj in (("events.json", records), ("manifest.json", run_manifest)):
        tmp = dest / (name + ".tmp")
        tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
        tmp.replace(dest / name)
    return dest
