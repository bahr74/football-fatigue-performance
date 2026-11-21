"""M7-0: opt-in pixel-space identity-provenance export with segment guards.

Turns (optionally identity-annotated) ``frames.jsonl`` records into *safe observation segments*: maximal runs of
consecutive observed foot-point samples that belong to one tracklet, one local tracker id, one global-id hypothesis,
one camera and one trust class, with no coordinate discontinuity. The export never bridges

* a tracklet boundary (a global id that rests on a link or split decision is kept as separate segments),
* missing observations (``max_missing_frames``, default 0),
* interpolated positions (stored apart in ``interpolated_samples``; never inside a segment),
* a camera / view change, a global-id change, a change of association provenance or trust class,
* a foot-point jump larger than ``max_jump_box_heights`` box heights per frame (e.g. a tracker swap).

Coordinates are image pixels (``coordinate_system``); nothing here is metric, and no per-player total is produced.
``global_id_use == "attributable"`` only says that every sample is ``confirmed`` with no unvalidated association link;
it is not a claim that the global identity is correct. Swaps inside a tracklet are not detectable (M6.3-D) and remain
a documented limitation. The module reads records only; it changes no identity code, threshold or default output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

EXPORT_SCHEMA_VERSION = "m7-0-identity-export-v1"
EXPORT_POLICY = "m7-0-pixel-space-segment-guard-v1"

COORDINATE_SYSTEM: dict[str, Any] = {
    "name": "image_px",
    "unit": "pixel",
    "origin": "top-left corner of the source video frame",
    "x_axis": "right",
    "y_axis": "down",
    "point": "bbox bottom-centre (foot point)",
    "metric_valid": False,
    "pitch_calibration": None,
}

KNOWN_LIMITATIONS = [
    "Pixel coordinates only: no metres, no speed, no sprint or distance totals are provided or implied.",
    "A global id is an identity hypothesis; 'attributable' means confirmed with no unvalidated association link, not verified correct.",
    "Identity swaps inside a tracklet are not detectable with the available signals (M6.3-D); segments assume one identity per tracklet.",
    "Segments never carry a movement total per player; any consumer must keep segment_id as the unit of attribution.",
    "Metres per pixel varies across the panorama; pixel displacements are not comparable between image positions.",
]

TRUSTED_STATUSES = ("confirmed",)


@dataclass(frozen=True)
class ExportConfig:
    """Segment-guard parameters. Defaults are conservative; none changes identity behaviour."""

    max_missing_frames: int = 0            # missing frames tolerated inside a segment (0: strictly consecutive)
    max_jump_box_heights: float = 1.5      # foot-point jump per frame, in own box heights, above which a segment is cut
    min_attributable_samples: int = 25     # shorter trusted segments stay 'local_only'
    trusted_statuses: tuple[str, ...] = TRUSTED_STATUSES
    require_provenance: bool = True        # records without association_status are never treated as trusted

    def __post_init__(self) -> None:
        if self.max_missing_frames < 0 or self.max_jump_box_heights <= 0 or self.min_attributable_samples < 1:
            raise ValueError("invalid ExportConfig")


def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _rows(records: Iterable[dict[str, Any]], source: dict[str, Any], cfg: ExportConfig):
    """Observed player rows, interpolated samples and exclusion counts from records."""
    rows: list[dict[str, Any]] = []
    interp: list[dict[str, Any]] = []
    excl: Counter = Counter()
    for r in records:
        f = int(r["frame_num"])
        ts = _num(r.get("timestamp_s"))
        cam = r.get("camera_id", source.get("camera_id"))
        for p in r.get("players", []):
            if p.get("detected_class") == "ball":
                continue
            bb = p.get("bbox_xyxy_px")
            if not bb or any(_num(v) is None for v in bb):
                excl["no_valid_bbox"] += 1
                continue
            if p.get("position_status", "observed") != "observed":
                excl["not_observed_position"] += 1
                continue
            x1, y1, x2, y2 = (float(v) for v in bb)
            annotated = "tracklet_id" in p and "global_id" in p
            local = int(p.get("local_track_id", p.get("track_id")))
            assoc = p.get("association_status")
            status = p.get("identity_status", "unannotated")
            rows.append({
                "frame": f, "ts": ts, "camera": cam, "local": local,
                "tracklet": p["tracklet_id"] if annotated else None, "gid": p["global_id"] if annotated else None,
                "annotated": annotated, "status": status, "conf": _num(p.get("identity_confidence")),
                "conf_local": _num(p.get("identity_confidence_local", p.get("identity_confidence"))),
                "link_conf": _num(p.get("association_link_confidence")), "assoc": assoc,
                "x": (x1 + x2) / 2.0, "y": y2, "h": max(y2 - y1, 1e-6),
            })
        for q in r.get("identity_interpolated", []) or []:
            bb = q.get("bbox_xyxy_px")
            if bb and all(_num(v) is not None for v in bb):
                interp.append({"frame": f, "global_id": q["global_id"], "x_px": round((bb[0] + bb[2]) / 2.0, 2),
                               "y_px": round(float(bb[3]), 2), "position_status": "interpolated"})
    return rows, interp, excl


def _trusted(row: dict[str, Any], cfg: ExportConfig) -> tuple[bool, str | None]:
    """Box-level trust for global attribution and the reason it is withheld."""
    if not row["annotated"]:
        return False, "not_identity_annotated"
    if row["status"] not in cfg.trusted_statuses:
        return False, "status_not_confirmed"
    if row["assoc"] is None:
        return (False, "provenance_not_recorded") if cfg.require_provenance else (True, None)
    if row["assoc"] != "none":
        return False, "linked_unvalidated"
    return True, None


def _break_reason(a: dict[str, Any], b: dict[str, Any], cfg: ExportConfig) -> str | None:
    gap = b["frame"] - a["frame"] - 1
    if gap < 0:
        return "duplicate_or_unordered_frame"
    if gap > cfg.max_missing_frames:
        return "missing_observations"
    if a["gid"] != b["gid"]:
        return "global_id_change"
    if a["camera"] != b["camera"]:
        return "camera_change"
    if a["assoc"] != b["assoc"]:
        return "association_change"
    if a["trust"] != b["trust"]:
        return "trust_change"
    dt = b["frame"] - a["frame"]
    if math.hypot(b["x"] - a["x"], b["y"] - a["y"]) > cfg.max_jump_box_heights * max(a["h"], b["h"]) * dt:
        return "coordinate_discontinuity"
    return None


def build_export(records: Iterable[dict[str, Any]], *, source: dict[str, Any] | None = None,
                 config: ExportConfig | None = None) -> dict[str, Any]:
    """Build the export from records (identity-annotated or plain tracker output). Pure function of its inputs."""
    cfg = config or ExportConfig()
    source = dict(source or {})
    rows, interp, excl = _rows(records, source, cfg)
    for r in rows:
        r["trust"], r["why"] = _trusted(r, cfg)
    groups: dict[tuple[Any, int], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["tracklet"], r["local"])].append(r)

    segments: list[dict[str, Any]] = []
    mid = source.get("match_id") or "m"
    for (tracklet, local), g in sorted(groups.items(), key=lambda kv: (min(r["frame"] for r in kv[1]), str(kv[0]))):
        g.sort(key=lambda r: r["frame"])
        runs: list[tuple[list[dict[str, Any]], str, str]] = []
        cur, start_reason = [g[0]], "start_of_track"
        for a, b in zip(g, g[1:]):
            why = _break_reason(a, b, cfg)
            if why is None:
                cur.append(b)
            else:
                runs.append((cur, start_reason, why))
                cur, start_reason = [b], why
        runs.append((cur, start_reason, "end_of_track"))
        for k, (run, s_reason, e_reason) in enumerate(runs):
            reasons = sorted({r["why"] for r in run if r["why"]})
            ok = not reasons
            if ok and len(run) < cfg.min_attributable_samples:
                ok, reasons = False, ["short_segment"]
            segments.append({
                "segment_id": f"{mid}:{'t' + str(tracklet) if tracklet is not None else 'l' + str(local)}:{k}",
                "local_track_id": local, "tracklet_id": tracklet, "global_id": run[0]["gid"],
                "association_status": run[0]["assoc"], "camera_id": run[0]["camera"],
                "global_id_use": "attributable" if ok else "local_only", "global_id_use_reasons": reasons,
                "n_samples": len(run), "frame_start": run[0]["frame"], "frame_end": run[-1]["frame"],
                "t_start_s": run[0]["ts"], "t_end_s": run[-1]["ts"], "start_break": s_reason, "end_break": e_reason,
                "frames": [r["frame"] for r in run], "timestamp_s": [r["ts"] for r in run],
                "x_px": [round(r["x"], 2) for r in run], "y_px": [round(r["y"], 2) for r in run],
                "box_height_px": [round(r["h"], 2) for r in run], "identity_status": [r["status"] for r in run],
                "identity_confidence": [r["conf"] for r in run], "identity_confidence_local": [r["conf_local"] for r in run],
                "association_link_confidence": [r["link_conf"] for r in run],
            })
    segments.sort(key=lambda s: (s["frame_start"], s["segment_id"]))

    by_gid: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for s in segments:
        if s["global_id"] is not None:
            by_gid[s["global_id"]].append(s)
    transitions: list[dict[str, Any]] = []
    for gid, ss in sorted(by_gid.items(), key=lambda kv: str(kv[0])):
        ss.sort(key=lambda s: (s["frame_start"], s["frame_end"]))
        for a, b in zip(ss, ss[1:]):
            gap = b["frame_start"] - a["frame_end"]
            if gap <= 0:
                kind = "overlap_conflict"      # one global id on two segments at once: it cannot be one trajectory
            elif a["tracklet_id"] != b["tracklet_id"]:
                kind = "tracklet_transition"
            else:
                kind = "within_tracklet_break"
            transitions.append({
                "global_id": gid, "from_segment": a["segment_id"], "to_segment": b["segment_id"], "kind": kind,
                "gap_frames": gap, "from_association": a["association_status"], "to_association": b["association_status"],
                "to_break_reason": b["start_break"], "bridged": False,
            })

    n_samples = sum(s["n_samples"] for s in segments)
    attributable = [s for s in segments if s["global_id_use"] == "attributable"]
    export = {
        "schema_version": EXPORT_SCHEMA_VERSION, "policy": EXPORT_POLICY, "source": source,
        "coordinate_system": dict(COORDINATE_SYSTEM), "params": asdict(cfg), "segments": segments,
        "transitions": transitions, "interpolated_samples": interp,
        "summary": {
            "segments": len(segments), "samples": n_samples, "attributable_segments": len(attributable),
            "attributable_samples": sum(s["n_samples"] for s in attributable),
            "transitions": dict(Counter(t["kind"] for t in transitions)),
            "transitions_blocked": len(transitions), "transitions_bridged": 0,
            "overlap_conflicts": sum(1 for t in transitions if t["kind"] == "overlap_conflict"),
            "segment_breaks": dict(Counter(s["end_break"] for s in segments)),
            "excluded": dict(excl), "interpolated_samples": len(interp),
        },
        "claims": {"metric_distance": False, "global_identity_validated": False, "fatigue_or_load_metric": False},
        "known_limitations": list(KNOWN_LIMITATIONS),
    }
    return export


def segment_path_length_px(segment: dict[str, Any]) -> float:
    """Diagnostic polyline length of one segment in PIXELS (not metres); used for reconciliation, not as a load metric."""
    xs, ys = segment["x_px"], segment["y_px"]
    return float(sum(math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i]) for i in range(len(xs) - 1)))


# ------------------------------------------------------------------------------------------------ data contract
_NUM_NULL = ["number", "null"]
EXPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["schema_version", "policy", "source", "coordinate_system", "params", "segments", "transitions",
                 "interpolated_samples", "summary", "claims", "known_limitations"],
    "properties": {
        "schema_version": {"type": "string", "enum": [EXPORT_SCHEMA_VERSION]},
        "policy": {"type": "string"},
        "source": {"type": "object"},
        "coordinate_system": {"type": "object", "required": ["name", "unit", "origin", "metric_valid"],
                              "properties": {"name": {"type": "string", "enum": ["image_px"]}, "unit": {"type": "string", "enum": ["pixel"]},
                                             "origin": {"type": "string"}, "metric_valid": {"type": "boolean", "enum": [False]}}},
        "params": {"type": "object", "required": ["max_missing_frames", "max_jump_box_heights", "min_attributable_samples"]},
        "segments": {"type": "array", "items": {
            "type": "object",
            "required": ["segment_id", "local_track_id", "tracklet_id", "global_id", "association_status", "global_id_use",
                         "global_id_use_reasons", "n_samples", "frame_start", "frame_end", "start_break", "end_break", "frames",
                         "timestamp_s", "x_px", "y_px", "box_height_px", "identity_status", "identity_confidence",
                         "identity_confidence_local", "association_link_confidence"],
            "properties": {
                "segment_id": {"type": "string"}, "local_track_id": {"type": "integer"},
                "tracklet_id": {"type": ["integer", "null"]}, "global_id": {"type": ["integer", "null"]},
                "association_status": {"type": ["string", "null"]},
                "global_id_use": {"type": "string", "enum": ["attributable", "local_only"]},
                "global_id_use_reasons": {"type": "array", "items": {"type": "string"}},
                "n_samples": {"type": "integer", "minimum": 1}, "frame_start": {"type": "integer"}, "frame_end": {"type": "integer"},
                "start_break": {"type": "string"}, "end_break": {"type": "string"},
                "frames": {"type": "array", "items": {"type": "integer"}},
                "timestamp_s": {"type": "array", "items": {"type": _NUM_NULL}},
                "x_px": {"type": "array", "items": {"type": "number"}}, "y_px": {"type": "array", "items": {"type": "number"}},
                "box_height_px": {"type": "array", "items": {"type": "number"}},
                "identity_status": {"type": "array", "items": {"type": "string"}},
                "identity_confidence": {"type": "array", "items": {"type": _NUM_NULL}},
                "identity_confidence_local": {"type": "array", "items": {"type": _NUM_NULL}},
                "association_link_confidence": {"type": "array", "items": {"type": _NUM_NULL}},
            }}},
        "transitions": {"type": "array", "items": {
            "type": "object", "required": ["global_id", "from_segment", "to_segment", "kind", "gap_frames", "bridged"],
            "properties": {"kind": {"type": "string", "enum": ["tracklet_transition", "within_tracklet_break", "overlap_conflict"]},
                           "gap_frames": {"type": "integer"}, "bridged": {"type": "boolean", "enum": [False]}}}},
        "interpolated_samples": {"type": "array", "items": {
            "type": "object", "required": ["frame", "global_id", "x_px", "y_px", "position_status"],
            "properties": {"position_status": {"type": "string", "enum": ["interpolated"]}}}},
        "summary": {"type": "object", "required": ["segments", "samples", "transitions_blocked", "transitions_bridged"]},
        "claims": {"type": "object", "required": ["metric_distance", "global_identity_validated", "fatigue_or_load_metric"],
                   "properties": {"metric_distance": {"type": "boolean", "enum": [False]},
                                  "global_identity_validated": {"type": "boolean", "enum": [False]},
                                  "fatigue_or_load_metric": {"type": "boolean", "enum": [False]}}},
        "known_limitations": {"type": "array", "items": {"type": "string"}},
    },
}

_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}


def _is(value: Any, t: str) -> bool:
    if t == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, _TYPES[t])


def _check(schema: dict[str, Any], value: Any, path: str, errors: list[str]) -> None:
    ts = schema.get("type")
    if ts is not None:
        ts = [ts] if isinstance(ts, str) else ts
        if not any(_is(value, t) for t in ts):
            errors.append(f"{path}: expected {ts}, got {type(value).__name__}")
            return
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in {schema['enum']}")
    if "minimum" in schema and isinstance(value, (int, float)) and value < schema["minimum"]:
        errors.append(f"{path}: {value} < {schema['minimum']}")
    if isinstance(value, dict):
        for k in schema.get("required", []):
            if k not in value:
                errors.append(f"{path}: missing '{k}'")
        for k, sub in schema.get("properties", {}).items():
            if k in value:
                _check(sub, value[k], f"{path}.{k}", errors)
    elif isinstance(value, list) and "items" in schema:
        for i, v in enumerate(value):
            _check(schema["items"], v, f"{path}[{i}]", errors)
            if len(errors) > 50:
                return


def validate_export(export: dict[str, Any]) -> list[str]:
    """Schema and invariant problems of an export (empty list = valid). Invariants are the segment guards themselves."""
    errors: list[str] = []
    _check(EXPORT_SCHEMA, export, "$", errors)
    if errors:
        return errors
    max_gap = export["params"]["max_missing_frames"]
    seen: set[str] = set()
    for s in export["segments"]:
        sid = s["segment_id"]
        n = s["n_samples"]
        if sid in seen:
            errors.append(f"{sid}: duplicate segment_id")
        seen.add(sid)
        for k in ("frames", "timestamp_s", "x_px", "y_px", "box_height_px", "identity_status", "identity_confidence",
                  "identity_confidence_local", "association_link_confidence"):
            if len(s[k]) != n:
                errors.append(f"{sid}: {k} has {len(s[k])} entries, n_samples={n}")
        fr = s["frames"]
        if fr and (fr[0] != s["frame_start"] or fr[-1] != s["frame_end"]):
            errors.append(f"{sid}: frame_start/frame_end disagree with frames")
        if any(b - a - 1 > max_gap or b <= a for a, b in zip(fr, fr[1:])):
            errors.append(f"{sid}: bridges a gap larger than max_missing_frames or is not increasing")
        if s["global_id_use"] == "attributable" and (s["global_id_use_reasons"] or any(
                st not in export["params"].get("trusted_statuses", TRUSTED_STATUSES) for st in s["identity_status"])):
            errors.append(f"{sid}: attributable but carries untrusted samples")
    if export["summary"]["transitions_bridged"] != 0 or any(t["bridged"] for t in export["transitions"]):
        errors.append("a transition is marked bridged")
    return errors


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def write_export(path: str | Path, export: dict[str, Any]) -> None:
    """Write a validated export to a NEW file (never overwrites)."""
    problems = validate_export(export)
    if problems:
        raise ValueError("invalid export: " + "; ".join(problems[:5]))
    with open(path, "x", encoding="utf-8") as fh:
        json.dump(export, fh, separators=(",", ":"))


def read_export(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        export = json.load(fh)
    problems = validate_export(export)
    if problems:
        raise ValueError("invalid export: " + "; ".join(problems[:5]))
    return export


def main(argv: Sequence[str] | None = None) -> int:
    """Opt-in export from an identity-annotated (or plain) run; writes a new file, never in place."""
    ap = argparse.ArgumentParser(prog="python -m football_fatigue.vision.identity_export", description=main.__doc__)
    ap.add_argument("frames", type=Path, help="frames.jsonl (identity-annotated or tracker output)")
    ap.add_argument("--output", type=Path, required=True, help="new JSON file (must not exist)")
    ap.add_argument("--match-id")
    ap.add_argument("--camera-id", default="panorama")
    ap.add_argument("--max-missing-frames", type=int, default=0)
    ap.add_argument("--max-jump-box-heights", type=float, default=1.5)
    a = ap.parse_args(argv)
    with a.frames.open(encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh]
    exp = build_export(records, source={"match_id": a.match_id, "camera_id": a.camera_id, "frames_path": str(a.frames),
                                         "frames_sha256": sha256_file(a.frames)},
                       config=ExportConfig(max_missing_frames=a.max_missing_frames, max_jump_box_heights=a.max_jump_box_heights))
    write_export(a.output, exp)
    s = exp["summary"]
    print(f"wrote {a.output}: {s['segments']} segments, {s['samples']} samples, {s['attributable_segments']} attributable, "
          f"{s['transitions_blocked']} transitions blocked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
