"""M7-DEMO: precomputed, display-only overlay data for the Football Vision Demo.

Turns identity-annotated ``frames.jsonl`` records (``identity.annotate_records``) plus the M7-0 safe-segment export into one compact,
columnar JSON the browser can index by frame number without per-frame server work. Everything is image-space (pixels):
nothing here is metric, and no speed, distance, workload or fatigue value is computed.

Display-only interpolation (``interp_max_gap``) fills short gaps *inside one tracklet* with ``est=1`` boxes that are stored beside, never
instead of, the observed samples. It never crosses a tracklet, local-id, global-id, identity-status, association or camera boundary, and
its boxes are excluded from every count and from the M7-0 segments.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

from football_fatigue.vision.identity_export import build_export

OVERLAY_SCHEMA = "m7-demo-overlay-v1"
STATUSES = ["confirmed", "provisional", "unknown", "ambiguous"]
ASSOC = ["none", "linked_unvalidated"]
TEAMS = ["team_0", "team_1", "other"]          # index into the 'team' column; -1 = no team evidence
TEAM_RELIABLE = 0.7                            # team colour is drawn only at or above this team_confidence
CONF_BINS = 20
ROLES = ["player_candidate", "goalkeeper", "official_candidate", "unknown"]
PITCH_EDGE_M = 1.0        # a foot point up to 1 m outside the lines still counts as on-pitch (line width, throw-ins)
PITCH_NEAR_M = 5.0        # 1-5 m outside: 'near edge' (drawn grey); beyond: outside the calibrated region (not drawn on the board)
FIELDS = ("local", "gid", "trk", "x1", "y1", "x2", "y2", "st", "conf", "conf_local", "link", "as", "team", "tconf", "cls", "seg", "est")


def _idx(values: list[str], v: Any, default: int = -1) -> int:
    return values.index(v) if v in values else default


def _box_rows(records: Iterable[dict[str, Any]]):
    """Observed player rows in file order: (frame, row dict). Ball rows and boxes without a bbox are not overlaid."""
    for r in records:
        f = int(r["frame_num"])
        for p in r.get("players", []):
            if p.get("detected_class") == "ball" or not p.get("bbox_xyxy_px") or "tracklet_id" not in p:
                continue
            if p.get("position_status", "observed") != "observed":
                continue
            bb = [float(v) for v in p["bbox_xyxy_px"]]
            yield f, {
                "local": int(p["local_track_id"]), "gid": int(p["global_id"]), "trk": int(p["tracklet_id"]), "bbox": bb,
                "st": _idx(STATUSES, p.get("identity_status")), "conf": p.get("identity_confidence"),
                "conf_local": p.get("identity_confidence_local"), "link": p.get("association_link_confidence"),
                "as": _idx(ASSOC, p.get("association_status")), "team": _idx(TEAMS, p.get("team_cluster")),
                "tconf": p.get("team_confidence"), "cls": p.get("detected_class") or "player", "est": 0,
            }


def display_interpolation(rows_by_frame: dict[int, list[dict[str, Any]]], max_gap: int) -> dict[int, list[dict[str, Any]]]:
    """Estimated boxes for gaps of at most ``max_gap`` missing frames inside one safe local tracklet.

    A gap is filled only if both ends share tracklet, local id, global id, identity status, association status and class, and
    every frame in the gap is empty for that local id. The filled box is flagged ``est=1`` and carries no confidence.
    """
    if max_gap <= 0:
        return {}
    by_key: dict[tuple[int, int], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for f, rows in rows_by_frame.items():
        for r in rows:
            by_key[(r["trk"], r["local"])].append((f, r))
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for seq in by_key.values():
        seq.sort(key=lambda x: x[0])
        for (f1, a), (f2, b) in zip(seq, seq[1:]):
            gap = f2 - f1 - 1
            if not 0 < gap <= max_gap:
                continue
            if any(a[k] != b[k] for k in ("gid", "st", "as", "cls")):
                continue
            for f in range(f1 + 1, f2):
                w = (f - f1) / (f2 - f1)
                out[f].append(dict(a, bbox=[(1 - w) * u + w * v for u, v in zip(a["bbox"], b["bbox"])], est=1,
                                   conf=None, conf_local=None, link=None))
    return dict(out)


def pitch_validity(x_m: float, y_m: float, length: float = 105.0, width: float = 68.0) -> int:
    """0 = on the pitch (within PITCH_EDGE_M of the lines), 1 = near the edge (<= PITCH_NEAR_M outside), 2 = outside the calibrated region."""
    out = max(-x_m, x_m - length, -y_m, y_m - width, 0.0)
    return 0 if out <= PITCH_EDGE_M else (1 if out <= PITCH_NEAR_M else 2)


def role_of(detected_class: str, team: str | None, validity: int | None) -> int:
    """Conservative display role (index into ROLES). Off-pitch people and third-colour-cluster 'players' stay unknown; the tracker's
    'referee' class is only a candidate (a development-sample inspection found 6/10 plausible officials)."""
    if validity is not None and validity != 0:
        return 3
    if detected_class == "goalkeeper":
        return 1
    if detected_class == "referee":
        return 2
    if detected_class == "player" and team != "other":
        return 0
    return 3


def build_overlay(records: list[dict[str, Any]], *, match_id: str, fps: float, video: dict[str, Any],
                  interp_max_gap: int = 5, image_width: int = 4096, image_height: int = 1080,
                  extra: dict[str, Any] | None = None, pitch_mapper: Any = None, landmarks_m: Any = None,
                  calibration: dict[str, Any] | None = None) -> dict[str, Any]:
    """Columnar overlay for the demo. ``records`` must be identity-annotated; reference ids are never involved."""
    n_frames = max(int(r["frame_num"]) for r in records) + 1
    rows_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for f, row in _box_rows(records):
        rows_by_frame[f].append(row)
    export = build_export(records, source={"match_id": match_id, "camera_id": "panorama"})
    seg_of: dict[tuple[int, int], int] = {}
    segments = []
    for k, s in enumerate(export["segments"]):
        segments.append({"id": s["segment_id"], "local": s["local_track_id"], "trk": s["tracklet_id"], "gid": s["global_id"],
                         "f0": s["frame_start"], "f1": s["frame_end"], "n": s["n_samples"], "use": s["global_id_use"],
                         "why": s["global_id_use_reasons"], "start": s["start_break"], "end": s["end_break"]})
        for f in s["frames"]:
            seg_of[(f, s["local_track_id"])] = k
    est_by_frame = display_interpolation(rows_by_frame, interp_max_gap)

    cols: dict[str, list[Any]] = {k: [] for k in FIELDS}
    offsets = [0]
    det, per_status = [], {s: [] for s in STATUSES}
    hist = {s: [0] * CONF_BINS for s in STATUSES}
    track_frames: dict[int, list[int]] = defaultdict(list)
    track_meta: dict[int, dict[str, Any]] = defaultdict(lambda: {"trk": set(), "gid": set(), "conf": Counter()})
    xs, ys = [], []
    for f in range(n_frames):
        rows = sorted(rows_by_frame.get(f, []), key=lambda r: r["local"])
        counts = Counter(r["st"] for r in rows)
        det.append(len(rows))
        for i, s in enumerate(STATUSES):
            per_status[s].append(counts.get(i, 0))
        for r in rows:
            track_frames[r["local"]].append(f)
            m = track_meta[r["local"]]
            m["trk"].add(r["trk"])
            m["gid"].add(r["gid"])
            if r["st"] >= 0 and r["conf"] is not None:
                hist[STATUSES[r["st"]]][min(CONF_BINS - 1, int(r["conf"] * CONF_BINS))] += 1
            xs.append((r["bbox"][0] + r["bbox"][2]) / 2)
            ys.append(r["bbox"][3])
        for r in rows + sorted(est_by_frame.get(f, []), key=lambda r: r["local"]):
            bb = r["bbox"]
            cols["local"].append(r["local"]); cols["gid"].append(r["gid"]); cols["trk"].append(r["trk"])
            for name, v in zip(("x1", "y1", "x2", "y2"), bb):
                cols[name].append(round(v, 1))
            cols["st"].append(r["st"])
            for name in ("conf", "conf_local", "link", "tconf"):
                cols[name].append(None if r[name] is None else round(float(r[name]), 4))
            cols["as"].append(r["as"]); cols["team"].append(r["team"]); cols["cls"].append(r["cls"])
            cols["seg"].append(seg_of.get((f, r["local"]), -1)); cols["est"].append(r["est"])
        offsets.append(len(cols["local"]))
    if pitch_mapper is not None:   # opt-in: approximate pitch projection of the foot point (metres, corner origin) + validity + role
        import numpy as np
        from scipy.spatial import cKDTree

        foot = np.c_[(np.array(cols["x1"]) + np.array(cols["x2"])) / 2, np.array(cols["y2"])]
        pm = pitch_mapper(foot) if len(foot) else np.zeros((0, 2))
        dk = cKDTree(np.asarray(landmarks_m, float)).query(pm)[0] if landmarks_m is not None and len(pm) else np.full(len(pm), -1.0)
        cols["pxm"] = [round(float(v), 2) for v in pm[:, 0]]
        cols["pym"] = [round(float(v), 2) for v in pm[:, 1]]
        cols["pv"] = [pitch_validity(a, b) for a, b in pm]
        cols["kpd"] = [round(float(v), 1) for v in dk]
        cols["role"] = [role_of(c, TEAMS[t] if t >= 0 else None, v) for c, t, v in zip(cols["cls"], cols["team"], cols["pv"])]
    ys_sorted = sorted(ys)
    lo, hi = ys_sorted[int(0.005 * (len(ys_sorted) - 1))], ys_sorted[int(0.995 * (len(ys_sorted) - 1))]

    tracks = {}
    for loc, frames in track_frames.items():
        gaps = sum(1 for a, b in zip(frames, frames[1:]) if b - a > 1)
        m = track_meta[loc]
        tracks[str(loc)] = {"first": frames[0], "last": frames[-1], "n": len(frames), "interruptions": gaps,
                            "tracklets": sorted(m["trk"]), "gids": sorted(m["gid"])}
    summary = {
        "frames": n_frames, "boxes_observed": sum(det), "boxes_estimated_display_only": sum(cols["est"]),
        "mean_detections_per_frame": round(sum(det) / n_frames, 2), "local_tracks": len(tracks),
        "status_boxes": {s: sum(per_status[s]) for s in STATUSES},
        "segments": len(segments), "attributable_segments": sum(1 for s in segments if s["use"] == "attributable"),
        "attributable_samples": sum(s["n"] for s in segments if s["use"] == "attributable"),
        "transitions_blocked": export["summary"]["transitions_blocked"], "transitions_bridged": export["summary"]["transitions_bridged"],
    }
    return {
        "schema": OVERLAY_SCHEMA, "match_id": match_id, "fps": fps, "n_frames": n_frames, "video": video,
        "image": {"width_px": image_width, "height_px": image_height},
        "coordinate_system": {"name": "image_px", "unit": "pixel", "origin": "top-left of the video frame", "x_axis": "right",
                              "y_axis": "down", "point": "bbox bottom-centre (foot point)", "metric_valid": False},
        "board": {"mode": "image_space_non_metric", "x_range_px": [0, image_width], "y_range_px": [round(lo, 1), round(hi, 1)],
                  "note": "Approximate image-space layout. Not registered to the pitch, not metric: no distance, speed or workload may be read from it."},
        "statuses": STATUSES, "assoc": ASSOC, "teams": TEAMS, "team_reliable": TEAM_RELIABLE, "interp_max_gap": interp_max_gap,
        "frame_offsets": offsets, "cols": cols, "segments": segments, "tracks": tracks,
        "series": {"det": det, **{s: per_status[s] for s in STATUSES}}, "conf_hist": hist, "conf_bins": CONF_BINS,
        "summary": summary, "limitations": LIMITATIONS_CALIBRATED if calibration else LIMITATIONS, "roles": ROLES, "calibration": calibration, **(extra or {}),
    }


LIMITATIONS = [
    "Boxes, IDs and statuses come from saved detector/tracker/identity outputs (no new inference); image-space only.",
    "The tactical board is an APPROXIMATE, NON-METRIC image-space layout: no distances, speeds, workload, sprints or fatigue.",
    "Global IDs are hypotheses; only 'confirmed' with no unvalidated link is shown as solid. Provisional/unknown are never treated as confirmed.",
    "Estimated (display-interpolated) boxes are visual aids, excluded from all counts and never measurements.",
    "Interior identity swaps inside a tracklet are not detectable with the available signals (M6.3-D).",
    "Development match only; no independent validation; whole-match identity reliability is not claimed.",
    "No ball, no jersey numbers, no pitch calibration (best held-out 2.04 m median is below the acceptance bar).",
]
# used when an approximate pitch calibration is supplied (overlay v3): the board may be a pitch projection, still not valid for speed/distance
LIMITATIONS_CALIBRATED = LIMITATIONS[:1] + [
    "Pitch positions come from an APPROXIMATE calibration of this one match (hybrid fisheye + TPS; held-out keypoint error median 0.45 m, "
    "p90 1.07 m; interior n=11: 0.52 / 1.75 m). Valid for display, team shape and coarse zones; NOT for distance, speed, workload or sprints.",
] + LIMITATIONS[2:6] + [
    "Roles are conservative: detector class + team cluster + pitch validity. Off-pitch people and the third colour cluster stay 'unknown'; "
    "the detector's referee class also fires on sideline staff (visual audit, not human ground truth).",
    "No ball, no jersey numbers, no player names. Pitch analytics describe the visible players only, with their counts.",
]
