"""Bounded, cached access to a saved CV run's ``frames.jsonl``.

The dashboard must never need the whole JSONL in memory or in one response.
A single streaming pass builds a byte-offset index and image-space summaries;
frame windows are then served by seeking.  All values are pixel-space only.
"""

from __future__ import annotations

import csv
import io
import json
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

MAX_WINDOW_FRAMES = 500
TIMELINE_POINTS = 300
TRAJECTORY_TRACKS = 25
TRAJECTORY_POINTS = 80
HEATMAP_COLS = 64
CONF_BINS = 20
KITS = ("blue", "white", "other", "unknown")


@dataclass
class _Track:
    first: int
    last: int
    count: int = 0
    gaps: int = 0
    conf_sum: float = 0.0
    path: list[tuple[int, float, float]] = field(default_factory=list)
    kits: Counter = field(default_factory=Counter)


@dataclass
class RunIndex:
    size: int
    mtime_ns: int
    width: int
    height: int
    offsets: list[tuple[int, int]]  # (frame_num, byte offset), ascending
    fps: float
    counts: list[tuple[int, int]]  # (frame_num, players)
    tracks: dict[int, _Track]
    heat: list[list[int]]
    consumed: int = 0  # bytes of complete lines already indexed
    heat_kit: dict = field(default_factory=dict)  # kit colour -> grid
    kit_frames: list = field(default_factory=list)  # (frame, blue, white, other, unknown) box counts
    conf_hist: list = field(default_factory=lambda: [0] * CONF_BINS)  # box confidence, equal bins on [0, 1]
    heat_min: dict = field(default_factory=dict)  # (minute, kit) -> grid


_cache: dict[str, RunIndex] = {}
_lock = threading.Lock()


_start_cache: dict[str, float] = {}


def video_start_time_s(path: Path) -> float:
    """Presentation time (s) of the first video frame, from the container.

    Browsers play container timestamps: this video's first frame is at ~0.979 s,
    so ``currentTime * fps`` is ~24.5 frames ahead of the decoded frame index.
    Returns 0.0 when it cannot be read.
    """
    key = str(path)
    if key in _start_cache:
        return _start_cache[key]
    value = 0.0
    try:
        import av

        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            if stream.start_time is not None and stream.time_base is not None:
                value = float(stream.start_time * stream.time_base)
    except Exception:
        value = 0.0
    _start_cache[key] = value
    return value


def get_index(path: Path, width: int = 4096, height: int = 1080) -> RunIndex:
    """Return a cached index; a growing (active) run is scanned incrementally."""
    stat = path.stat()
    key = f"{path.resolve()}|{width}x{height}"
    with _lock:
        index = _cache.get(key)
        if index is None or stat.st_size < index.consumed:  # new or truncated/replaced file
            rows = max(1, round(HEATMAP_COLS * height / width))
            index = RunIndex(0, 0, width, height, [], 25.0, [], {}, [[0] * HEATMAP_COLS for _ in range(rows)])
            _cache[key] = index
        if stat.st_size != index.size or stat.st_mtime_ns != index.mtime_ns:
            _scan(index, path)
            index.size, index.mtime_ns = stat.st_size, stat.st_mtime_ns
        return index


def _scan(index: RunIndex, path: Path) -> None:
    """Index complete lines after ``index.consumed``; a partial last line is left for later."""
    width, height = index.width, index.height
    rows = len(index.heat)
    with path.open("rb") as stream:
        stream.seek(index.consumed)
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line or not line.endswith(b"\n"):
                break
            index.consumed = stream.tell()
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            frame = int(rec["frame_num"])
            index.fps = float(rec.get("fps") or index.fps)
            players = rec.get("players") or []
            index.offsets.append((frame, offset))
            index.counts.append((frame, len(players)))
            per_kit = dict.fromkeys(KITS, 0)
            minute = int(frame / index.fps // 60)
            for p in players:
                k = p.get("kit_color")
                per_kit[k if k in per_kit else "unknown"] += 1
                c = p.get("confidence")
                if c is not None:
                    index.conf_hist[min(CONF_BINS - 1, max(0, int(float(c) * CONF_BINS)))] += 1
            index.kit_frames.append((frame, *(per_kit[k] for k in KITS)))
            for p in players:
                tid = int(p["track_id"])
                trk = index.tracks.get(tid)
                if trk is None:
                    trk = index.tracks[tid] = _Track(first=frame, last=frame)
                elif frame - trk.last > 1:
                    trk.gaps += 1
                trk.last = frame
                trk.count += 1
                trk.conf_sum += float(p.get("confidence") or 0.0)
                trk.kits[p.get("kit_color") or "unknown"] += 1
                px, py = p.get("pixel_x"), p.get("pixel_y")
                if px is not None and py is not None:
                    trk.path.append((frame, float(px), float(py)))
                    col = min(HEATMAP_COLS - 1, max(0, int(px / width * HEATMAP_COLS)))
                    row = min(rows - 1, max(0, int(py / height * rows)))
                    index.heat[row][col] += 1
                    kit = p.get("kit_color") or "unknown"
                    grid = index.heat_kit.get(kit)
                    if grid is None:
                        grid = index.heat_kit[kit] = [[0] * HEATMAP_COLS for _ in range(rows)]
                    grid[row][col] += 1
                    mgrid = index.heat_min.get((minute, kit))
                    if mgrid is None:
                        mgrid = index.heat_min[(minute, kit)] = [[0] * HEATMAP_COLS for _ in range(rows)]
                    mgrid[row][col] += 1


def _downsample(items: list, limit: int) -> list:
    if len(items) <= limit:
        return items
    step = len(items) / limit
    return [items[int(i * step)] for i in range(limit)]


def _movement_px(path: list[tuple[int, float, float]], fps: float) -> dict[str, float | None]:
    """Image-space movement of a track's foot point (pixels, not metres).

    Positions are smoothed with a 5-frame moving average before summing step
    lengths, so detector jitter does not inflate path length. Frame gaps
    longer than 5 frames are not bridged.
    """
    if len(path) < 2:
        return {"path_px": None, "net_px": None, "mean_speed_px_s": None}
    k = 5
    xs, ys = [p[1] for p in path], [p[2] for p in path]
    sx = [sum(xs[max(0, i - k // 2): i + k // 2 + 1]) / len(xs[max(0, i - k // 2): i + k // 2 + 1]) for i in range(len(xs))]
    sy = [sum(ys[max(0, i - k // 2): i + k // 2 + 1]) / len(ys[max(0, i - k // 2): i + k // 2 + 1]) for i in range(len(ys))]
    length = 0.0
    seconds = 0.0
    for i in range(1, len(path)):
        df = path[i][0] - path[i - 1][0]
        if df > 5:
            continue
        length += ((sx[i] - sx[i - 1]) ** 2 + (sy[i] - sy[i - 1]) ** 2) ** 0.5
        seconds += df / fps
    net = ((sx[-1] - sx[0]) ** 2 + (sy[-1] - sy[0]) ** 2) ** 0.5
    return {
        "path_px": round(length, 1),
        "net_px": round(net, 1),
        "mean_speed_px_s": round(length / seconds, 1) if seconds > 0 else None,
    }


def _track_rows(index: RunIndex) -> list[dict[str, Any]]:
    rows = []
    for tid, t in index.tracks.items():
        span = t.last - t.first + 1
        rows.append({
            "track_id": tid,
            "first_frame": t.first,
            "last_frame": t.last,
            "frames_seen": t.count,
            "duration_s": round(span / index.fps, 2),
            "continuity": round(t.count / span, 3),
            "gaps": t.gaps,
            "mean_confidence": round(t.conf_sum / t.count, 3) if t.count else None,
            "kit_color": t.kits.most_common(1)[0][0] if t.kits else "unknown",
            **_movement_px(t.path, index.fps),
        })
    rows.sort(key=lambda r: -r["frames_seen"])
    return rows


def summary(index: RunIndex) -> dict[str, Any]:
    """Compact image-space summary: counts timeline, track continuity, trajectories, heatmap."""
    n = len(index.offsets)
    timeline = [
        {"frame": f, "t": round(f / index.fps, 2), "players": c}
        for f, c in _downsample(index.counts, TIMELINE_POINTS)
    ]
    track_rows = _track_rows(index)
    top = {r["track_id"] for r in track_rows[:TRAJECTORY_TRACKS]}
    trajectories = {
        str(tid): [[round(x, 1), round(y, 1)] for _, x, y in _downsample(t.path, TRAJECTORY_POINTS)]
        for tid, t in index.tracks.items() if tid in top
    }
    durations = sorted(r["duration_s"] for r in track_rows)
    counts = [c for _, c in index.counts]
    return {
        "frames_indexed": n,
        "first_frame": index.offsets[0][0] if n else None,
        "last_frame": index.offsets[-1][0] if n else None,
        "fps": index.fps,
        "players_per_frame": {
            "mean": round(sum(counts) / n, 2) if n else None,
            "min": min(counts) if counts else None,
            "max": max(counts) if counts else None,
        },
        "track_count": len(track_rows),
        "short_tracks_lt_1s": sum(1 for d in durations if d < 1.0),
        "median_track_duration_s": durations[len(durations) // 2] if durations else None,
        "tracks": track_rows[:200],
        "timeline": timeline,
        "kit_track_counts_ge50_frames": dict(Counter(r["kit_color"] for r in track_rows if r["frames_seen"] >= 50)),
        "trajectories_px": trajectories,
        "heatmap": {
            "space": "image_pixels",
            "width_px": index.width,
            "height_px": index.height,
            "rows": len(index.heat),
            "cols": HEATMAP_COLS,
            "cells": index.heat,
            "by_kit": {k: g for k, g in index.heat_kit.items()},
        },
    }


def _compact(rec: dict) -> dict:
    """Trim a saved record to what the dashboard draws; pixel values to 0.1 px."""
    r1 = lambda v: None if v is None else round(float(v), 1)
    players = [{
        "track_id": p["track_id"],
        "detected_class": p.get("detected_class"),
        "confidence": p.get("confidence"),
        "kit_color": p.get("kit_color"),
        "bbox_xyxy_px": [r1(v) for v in p["bbox_xyxy_px"]] if p.get("bbox_xyxy_px") else None,
        "pixel_x": r1(p.get("pixel_x")),
        "pixel_y": r1(p.get("pixel_y")),
    } for p in rec.get("players") or []]
    ball = rec.get("ball")
    return {
        "frame_num": rec["frame_num"],
        "timestamp_s": rec.get("timestamp_s"),
        "players": players,
        "ball": {"pixel_x": r1(ball.get("pixel_x")), "pixel_y": r1(ball.get("pixel_y"))} if ball else None,
    }


def frame_window(path: Path, index: RunIndex, start: int, end: int, limit: int = MAX_WINDOW_FRAMES) -> list[dict]:
    """Return records with ``start <= frame_num < end``, capped at ``limit`` (<= MAX_WINDOW_FRAMES)."""
    limit = max(1, min(limit, MAX_WINDOW_FRAMES))
    import bisect

    keys = [f for f, _ in index.offsets]
    lo = bisect.bisect_left(keys, start)
    out: list[dict] = []
    with path.open("rb") as stream:
        for frame, offset in index.offsets[lo:]:
            if frame >= end or len(out) >= limit:
                break
            stream.seek(offset)
            out.append(_compact(json.loads(stream.readline())))
    return out


def tracks_csv(path: Path, index: RunIndex) -> Iterator[str]:
    """Stream one CSV row per tracked box; pixel-space only."""
    header = ["frame_num", "timestamp_s", "track_id", "class", "confidence",
              "x1_px", "y1_px", "x2_px", "y2_px", "foot_x_px", "foot_y_px", "kit_color"]
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerow(header)
    yield buf.getvalue()
    with path.open("rb") as stream:
        for _, offset in index.offsets:
            stream.seek(offset)
            rec = json.loads(stream.readline())
            buf = io.StringIO()
            w = csv.writer(buf, lineterminator="\n")
            for p in rec.get("players") or []:
                box = p.get("bbox_xyxy_px") or [None] * 4
                w.writerow([rec["frame_num"], rec.get("timestamp_s"), p["track_id"],
                            p.get("detected_class"), p.get("confidence"), *box,
                            p.get("pixel_x"), p.get("pixel_y"), p.get("kit_color")])
            text = buf.getvalue()
            if text:
                yield text


# --------------------------------------------------------------------------------------------
# Team-level image-space analytics. Everything here is derived from the saved records only
# (no detector run) and is expressed in camera-view pixels or counts, never metres.
# --------------------------------------------------------------------------------------------

MIN_STABLE_FRAMES = 100
MIN_STABLE_CONTINUITY = 0.8
MIN_REGION_BOXES = 50  # per-minute region shares below this many boxes are reported as null
DURATION_EDGES_S = (1, 2, 5, 10, 30)


def _median(values: list[float]) -> float | None:
    values = sorted(values)
    if not values:
        return None
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2


def _extent(cells: list[list[int]], lo_q: float = 0.025, hi_q: float = 0.975) -> dict | None:
    """Occupied extent of a heat grid in cell units: the 2.5-97.5 percentile of foot points per axis."""
    total = sum(sum(r) for r in cells)
    if total == 0:
        return None

    def span(marginal: list[int]) -> tuple[float, float]:
        acc, lo, hi = 0, 0, len(marginal) - 1
        found = False
        for i, v in enumerate(marginal):
            acc += v
            if not found and acc >= lo_q * total:
                lo, found = i, True
            if acc >= hi_q * total:
                hi = i
                break
        return float(lo), float(hi + 1)

    cols = [sum(r[c] for r in cells) for c in range(len(cells[0]))]
    rows = [sum(r) for r in cells]
    x0, x1 = span(cols)
    y0, y1 = span(rows)
    return {"x0": x0, "x1": x1, "y0": y0, "y1": y1}


def _region(c: int, r: int, ext: dict) -> tuple[int, int]:
    """(column third 0=left..2=right, row half 0=far/top..1=near/bottom) of a cell centre."""
    fx = (c + 0.5 - ext["x0"]) / max(1e-9, ext["x1"] - ext["x0"])
    fy = (r + 0.5 - ext["y0"]) / max(1e-9, ext["y1"] - ext["y0"])
    return min(2, max(0, int(fx * 3))), (0 if fy < 0.5 else 1)


def _region_counts(grid: list[list[int]], ext: dict) -> list[list[int]]:
    out = [[0, 0, 0], [0, 0, 0]]  # [far, near][left, centre, right]
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            if v:
                ci, ri = _region(c, r, ext)
                out[ri][ci] += v
    return out


def _share(counts: list[list[int]]) -> list[list[float | None]]:
    total = sum(map(sum, counts))
    return [[round(v / total, 4) if total else None for v in row] for row in counts]


def team_analytics(index: RunIndex, manifest: dict | None = None, reference_eval: dict | None = None,
                   bin_s: float = 2.0) -> dict[str, Any]:
    """Match overview, per-kit time series, image-space occupancy, camera-view regions,
    tracking diagnostics and pixel-space movement summaries for the indexed run."""
    manifest = manifest or {}
    base = summary(index)
    n = len(index.kit_frames)
    fps = index.fps or 25.0
    expected = (manifest.get("summary") or {}).get("total_frames_expected")
    sums = [sum(f[1 + i] for f in index.kit_frames) for i in range(4)]
    total_boxes = sum(sums)
    kit_boxes = {k: sums[i] for i, k in enumerate(KITS)}
    share = {k: (round(v / total_boxes, 4) if total_boxes else None) for k, v in kit_boxes.items()}
    mean_pf = {k: (round(sums[i] / n, 2) if n else None) for i, k in enumerate(KITS)}
    zero_frames = sum(1 for f in index.kit_frames if sum(f[1:]) == 0)

    # time series: mean boxes per kit within fixed bins; empty bins are null, never zero
    per_bin = max(1, int(round(bin_s * fps)))
    series: dict[str, list] = {"t": [], **{k: [] for k in KITS}}
    if n:
        first = index.kit_frames[0][0]
        buckets: dict[int, list] = {}
        for f in index.kit_frames:
            buckets.setdefault((f[0] - first) // per_bin, []).append(f)
        for b in range(max(buckets) + 1):
            rows = buckets.get(b)
            series["t"].append(round((first + b * per_bin) / fps, 2))
            for i, k in enumerate(KITS):
                series[k].append(round(sum(r[1 + i] for r in rows) / len(rows), 2) if rows else None)

    # occupancy and camera-view regions
    rows_n = len(index.heat)
    all_ext = _extent(index.heat)
    regions: dict[str, Any] = {"definition": None, "by_kit": {}, "by_minute": {}}
    if all_ext:
        w, h = index.width / HEATMAP_COLS, index.height / rows_n
        regions["definition"] = {
            "extent_px": {"x0": round(all_ext["x0"] * w), "x1": round(all_ext["x1"] * w),
                          "y0": round(all_ext["y0"] * h), "y1": round(all_ext["y1"] * h)},
            "columns": ["left", "centre", "right"], "rows": ["far", "near"],
            "basis": "2.5-97.5 percentile extent of all detected foot points, split in thirds (x) and halves (y)",
        }
        for k in KITS:
            grid = index.heat_kit.get(k)
            if grid and sum(map(sum, grid)):
                counts = _region_counts(grid, all_ext)
                regions["by_kit"][k] = {"counts": counts, "share": _share(counts)}
        minutes = sorted({m for m, _ in index.heat_min})
        for k in ("blue", "white"):
            rows_out = []
            for m in minutes:
                grid = index.heat_min.get((m, k))
                counts = _region_counts(grid, all_ext) if grid else [[0] * 3, [0] * 3]
                tot = sum(map(sum, counts))
                rows_out.append({
                    "minute": m, "boxes": tot,
                    "share": _share(counts) if tot >= MIN_REGION_BOXES else None,
                })
            regions["by_minute"][k] = rows_out

    # tracking diagnostics (unsupervised: no reference data)
    durations = [t.last - t.first + 1 for t in index.tracks.values()]
    dur_hist = []
    edges = [0, *[int(e * fps) for e in DURATION_EDGES_S], None]
    for lo, hi in zip(edges, edges[1:]):
        count = sum(1 for d in durations if d >= lo and (hi is None or d < hi))
        label = f"<{DURATION_EDGES_S[0]} s" if lo == 0 else (
            f"{lo / fps:g}+ s" if hi is None else f"{lo / fps:g}-{hi / fps:g} s")
        dur_hist.append({"label": label, "tracks": count})
    gaps = [t.gaps for t in index.tracks.values()]
    conf_total = sum(index.conf_hist)
    conf = [{"lo": round(i / CONF_BINS, 2), "hi": round((i + 1) / CONF_BINS, 2), "boxes": v}
            for i, v in enumerate(index.conf_hist)]
    cont = [t.count / (t.last - t.first + 1) for t in index.tracks.values()]
    box_total = sum(t.count for t in index.tracks.values())
    diagnostics = {
        "track_ids": len(index.tracks),
        "median_duration_s": base["median_track_duration_s"],
        "shorter_than_1s": base["short_tracks_lt_1s"],
        "duration_histogram": dur_hist,
        "confidence_histogram": conf if conf_total else None,
        "mean_confidence": round(sum(t.conf_sum for t in index.tracks.values()) / box_total, 3) if box_total else None,
        "fragmentation": {
            "tracks_with_gaps": sum(1 for g in gaps if g),
            "total_gaps": sum(gaps),
            "mean_continuity": round(sum(cont) / len(cont), 3) if cont else None,
        },
        "refinement": manifest.get("refinement"),
        "reference_evaluation": reference_eval,
    }

    # pixel-space movement on tracks stable enough for a smoothed foot-point path
    all_rows = _track_rows(index)
    stable = [r for r in all_rows
              if r["frames_seen"] >= MIN_STABLE_FRAMES and r["continuity"] >= MIN_STABLE_CONTINUITY
              and r.get("mean_speed_px_s") is not None]
    by_kit_mv = {}
    for k in KITS:
        rows = [r for r in stable if r["kit_color"] == k]
        by_kit_mv[k] = {
            "tracks": len(rows),
            "median_px_s": _median([r["mean_speed_px_s"] for r in rows]),
            "median_path_px": _median([r["path_px"] for r in rows]),
        }
    movement = {
        "criteria": f">= {MIN_STABLE_FRAMES} frames seen and >= {MIN_STABLE_CONTINUITY:.0%} continuity",
        "stable_tracks": len(stable),
        "of_tracks": len(index.tracks),
        "by_kit": by_kit_mv,
        "rows": sorted(stable, key=lambda r: -r["frames_seen"])[:40],
    }

    return {
        "overview": {
            "frames_processed": n,
            "frames_expected": expected,
            "duration_s": round(n / fps, 2) if n else None,
            "fps": fps,
            "frames_with_no_detections": zero_frames,
            "boxes_total": total_boxes,
            "boxes_by_kit": kit_boxes,
            "share_by_kit": share,
            "team_label_coverage": round((kit_boxes["blue"] + kit_boxes["white"] + kit_boxes["other"]) / total_boxes, 4) if total_boxes else None,
            "mean_boxes_per_frame": mean_pf,
            "mean_boxes_per_frame_all": round(total_boxes / n, 2) if n else None,
            "track_ids": len(index.tracks),
        },
        "warnings": _warnings(index, kit_boxes, total_boxes, mean_pf, base, reference_eval),
        "series": {"bin_s": bin_s, **series},
        "heatmaps": {
            "space": "image_pixels", "width_px": index.width, "height_px": index.height,
            "cols": HEATMAP_COLS, "rows": rows_n,
            "by_kit": {k: index.heat_kit[k] for k in KITS if k in index.heat_kit},
            "region_extent_px": (regions["definition"] or {}).get("extent_px"),
        },
        "regions": regions,
        "diagnostics": diagnostics,
        "movement": movement,
    }


def _warnings(index, kit_boxes, total, mean_pf, base, ref) -> list[dict]:
    out = [{"level": "info", "text": "Track IDs are tracker labels, not verified player identities. "
                                     "Kit labels are colour labels, not verified home/away."}]
    if not total:
        return [{"level": "warn", "text": "No detections in the saved run."}, *out]
    if kit_boxes["unknown"] / total > 0.10:
        out.append({"level": "warn", "text": f"{kit_boxes['unknown'] / total:.0%} of boxes have no usable kit colour."})
    mean_all = sum(v for v in mean_pf.values() if v is not None)
    if mean_all < 20:
        out.append({"level": "warn", "text": f"Only {mean_all:.1f} boxes per frame on average; a match has 22 players on "
                                              "the pitch, so detection recall is incomplete (far players and clusters are missed)."})
    if len(index.tracks) > 2 * 22:
        out.append({"level": "warn", "text": f"{len(index.tracks)} track IDs for about 22 players: tracks are fragmented."})
    if base["track_count"] and base["short_tracks_lt_1s"] / base["track_count"] > 0.2:
        out.append({"level": "warn", "text": f"{base['short_tracks_lt_1s']} tracks last under 1 s (fragments or false detections)."})
    b, w = mean_pf.get("blue"), mean_pf.get("white")
    if b is not None and w is not None and abs(b - w) > 2:
        out.append({"level": "info", "text": f"Blue and white are detected at different rates ({b} vs {w} per frame); "
                                              "this can come from position, occlusion or lighting, not only from play."})
    if ref and ref.get("idf1") is not None and ref["idf1"] < 0.5:
        out.append({"level": "warn", "text": f"Reference IDF1 is {ref['idf1']:.2f}: long-term identity continuity is low."})
    out.append({"level": "info", "text": "Referees, goalkeepers and staff are detected too; counts are detections, not verified players."})
    return out
