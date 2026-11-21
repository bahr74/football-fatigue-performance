"""GT-free MOT-format export of a tracker-input profile's tracks (M13, preparation for the GTA offline refiner).

Reads only ``tracking.tracks_dir(...)/frames.jsonl`` + its ``manifest.json`` (tracker output: temporary ByteTrack IDs,
measured cached boxes, confidences). Ground truth is never opened, so no SoccerTrack identity can leak into a model
input. One MOT text line per tracked box, the layout gta-link's ``generate_tracklets.py`` unpacks::

    <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,-1,-1,-1

* ``frame`` is 1-based (MOT convention): ``mot_frame = source_frame + 1`` (recorded in the sidecar);
* ``id`` is the temporary tracker ID, unchanged (not a player identity);
* boxes are pixels of the source video (x, y, w, h), 3 decimals; ``conf`` is the detection confidence, 4 decimals.

Output (write-once): ``<dest>/<source_id>.txt`` + ``<dest>/<source_id>.mot.json`` (provenance, sha256 of input and
output, frame mapping). :func:`read_mot` reads such a file (e.g. GTA's refined output) back into the
``{frame: [(id, (x1, y1, x2, y2), conf, None)]}`` form that ``detection.replay.score`` takes, so T0 and T0+GTA are
scored by the same evaluator.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from football_fatigue.video.sources import SOURCES, guard_not_reserved
from football_fatigue.video.tracking import tracks_dir

MOT_VERSION = "mot-export-v1"
FRAME_OFFSET = 1


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def mot_lines(frames_jsonl: Path):
    """Yield MOT lines from a tracks ``frames.jsonl`` (tracker output only)."""
    with Path(frames_jsonl).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            fr = int(r["frame_num"]) + FRAME_OFFSET
            for p in sorted(r["players"], key=lambda q: int(q["track_id"])):
                x1, y1, x2, y2 = (float(v) for v in p["bbox_xyxy_px"])
                if x2 <= x1 or y2 <= y1:
                    raise ValueError(f"degenerate box at frame {r['frame_num']}: {p['bbox_xyxy_px']}")
                yield (f"{fr},{int(p['track_id'])},{x1:.3f},{y1:.3f},{x2 - x1:.3f},{y2 - y1:.3f},"
                       f"{float(p['confidence']):.4f},-1,-1,-1")


def export_dir(src_dir: Path, source_id: str, dest: Path, profile: str = "?") -> Path:
    """Export ``src_dir/frames.jsonl`` to ``dest/<source_id>.txt`` (+ sidecar). ``src_dir`` holds tracker output only."""
    src_dir, dest = Path(src_dir), Path(dest)
    out = dest / f"{source_id}.txt"
    if out.exists():
        raise FileExistsError(f"write-once: {out}")
    dest.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".txt.partial")
    n = 0
    ids: set[int] = set()
    frames: set[int] = set()
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for line in mot_lines(src_dir / "frames.jsonl"):
            fh.write(line + "\n")
            f, i = line.split(",", 2)[:2]
            frames.add(int(f))
            ids.add(int(i))
            n += 1
    tmp.rename(out)
    man_p = src_dir / "manifest.json"
    man: dict[str, Any] = json.loads(man_p.read_text(encoding="utf-8")) if man_p.exists() else {}
    side = {"version": MOT_VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_id": source_id,
            "profile": profile, "input": {"frames_jsonl": str(src_dir / "frames.jsonl"),
                                          "sha256": _sha(src_dir / "frames.jsonl"),
                                          "tracker_manifest_configuration": man.get("configuration")},
            "output": {"file": out.name, "sha256": _sha(out), "lines": n, "frames_with_boxes": len(frames),
                       "temporary_ids": len(ids)},
            "format": "<frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,-1,-1,-1",
            "frame_mapping": f"mot_frame = source_frame + {FRAME_OFFSET}",
            "ids": "temporary ByteTrack IDs of the profile (not player identities)",
            "ground_truth_used": False}
    (dest / f"{source_id}.mot.json").write_text(json.dumps(side, indent=1), encoding="utf-8")
    return out


def export(data_root: Path, out_root: Path, source_id: str, profile: str = "T0") -> Path:
    spec = SOURCES[source_id]
    guard_not_reserved(source_id, spec["match_id"], spec["cache_run"], spec["replay_run"])
    dest = Path(out_root) / "_bench" / MOT_VERSION / profile
    return export_dir(tracks_dir(data_root, out_root, source_id, profile), source_id, dest, profile)


def read_mot(path: Path) -> dict[int, list]:
    """MOT text -> ``{source_frame: [(id, (x1, y1, x2, y2), conf, None), ...]}`` (inverse of the export)."""
    out: dict[int, list] = {}
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            v = line.strip().split(",")
            f, i = int(float(v[0])) - FRAME_OFFSET, int(float(v[1]))
            x, y, w, h, c = (float(t) for t in v[2:7])
            out.setdefault(f, []).append((i, (x, y, x + w, y + h), c, None))
    return out
