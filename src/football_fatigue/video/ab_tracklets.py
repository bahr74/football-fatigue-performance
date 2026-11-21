"""Session B side of the Session A/B tracklet contract (``m11-session-ab-candidate-event-v1``), observations only.

M13 preparation, NOT an integration: nothing here imports or calls Session A. It turns a tracker-input profile's
``frames.jsonl`` into per-tracklet observation arrays in the contract's terms:

* ``timestamp_s`` = ``pts_offset_s + frame / fps`` (video clock; Session A never uses frame indices);
* ``pixel`` = foot point (bottom-centre of the box), ``box_h_px``, ``det_conf`` - image space only;
* ``pitch`` = ``None`` unless a calibration with an uncertainty estimate is supplied. Session B has no validated
  pitch calibration for SoccerTrack, so raw pixels are **never** placed in the metric ``pitch`` field;
* ``position_status`` = ``"observed"`` for every exported sample: the tracks hold measured cached boxes only.
  Frames without a box are absent (gaps), never filled with ByteTrack predictions or interpolation;
* identity: temporary tracker IDs are ``local_only``; outgoing candidates are ``anonymous`` with ``global_id = None``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

CONTRACT = "m11-session-ab-candidate-event-v1"


def tracklets(frames_jsonl: Path, fps: float, pts_offset_s: float,
              calibrate: Callable[[float, float], tuple[float, float, float]] | None = None,
              frames: range | None = None) -> dict[int, dict[str, Any]]:
    """Per temporary track ID: parallel observation arrays (``calibrate`` maps a foot pixel to (x_m, y_m, sigma_m))."""
    out: dict[int, dict[str, Any]] = {}
    with Path(frames_jsonl).open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            f = int(r["frame_num"])
            if frames is not None and f not in frames:
                continue
            for p in r["players"]:
                if p.get("measured") is False:          # never export a tracker prediction as an observation
                    continue
                x1, y1, x2, y2 = p["bbox_xyxy_px"]
                fx, fy = (x1 + x2) / 2, y2
                t = out.setdefault(int(p["track_id"]), {"frame": [], "timestamp_s": [], "pixel": [], "pitch": [],
                                                       "position_status": []})
                t["frame"].append(f)
                t["timestamp_s"].append(round(pts_offset_s + f / fps, 6))
                t["pixel"].append({"x_px": round(fx, 2), "y_px": round(fy, 2), "box_h_px": round(y2 - y1, 2),
                                   "det_conf": round(float(p["confidence"]), 4)})
                if calibrate is None:
                    t["pitch"].append(None)
                else:
                    xm, ym, sm = calibrate(fx, fy)
                    t["pitch"].append({"x_m": xm, "y_m": ym, "sigma_m": sm})
                t["position_status"].append("observed")
    return out


def candidate(track_id: int, obs: dict[str, Any], incoming: bool) -> dict[str, Any]:
    """One tracklet entry; outgoing candidates are anonymous (the contract forbids a pre-filled id)."""
    ident = ({"global_id": None, "identity_status": "local_only", "identity_confidence": None,
              "source": f"session-b temporary track {track_id}"} if incoming else
             {"global_id": None, "identity_status": "anonymous", "identity_confidence": None, "source": None})
    return {"tracklet_id": f"b-t{track_id}", "identity": ident, "observations": obs}


def hidden_intervals(obs: dict[str, Any], fps: float, min_gap_frames: int = 2) -> list[tuple[float, float]]:
    """Gaps inside one tracklet (no observed sample for >= ``min_gap_frames``), in video seconds."""
    fr, ts = obs["frame"], obs["timestamp_s"]
    return [(ts[i] + 1 / fps, ts[i + 1] - 1 / fps) for i in range(len(fr) - 1) if fr[i + 1] - fr[i] - 1 >= min_gap_frames]
