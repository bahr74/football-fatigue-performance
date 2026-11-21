"""Advisor-recipe cross-check of the geometric intersection detector (M11, Dr. Basha's definition).

The advisor's recipe: give the two trajectories (polylines) to a curve-intersection algorithm, which returns the
intersection (occlusion) point P; locate P along each trajectory (``LineString.project``: arc length to the nearest point
on the line), find the containing segment from the cumulative segment lengths (``np.searchsorted``) and interpolate that
player's time from its real timestamps.

The v1 detector (:mod:`.detect`, unchanged) implements the same mathematics segment by segment: P from the exact
segment-pair intersection, and the time ``t0 + u (t1 - t0)`` on the containing segment. This module re-derives every
v1 event independently:

1. **Point**: the two source segments' supporting lines are intersected with a 2x2 linear solve (a different formula from
   :func:`.geometry.intersect`).
2. **Time**: the advisor's projection recipe, applied to each player's whole contiguous polyline in the detecting window.

``project`` returns the *first* along-line position nearest to P. When a player's own path passes through P more than once
in the window (a loop), the recipe can return another pass than the one that crossed the other player. Such events are
counted separately as ``self_revisit``; they are not errors of either method, since the exact segment index removes the
ambiguity. Shapely itself is not installed; this numpy version follows its documented semantics.
Run: ``python -m football_fatigue.intersections.advisor --match 3812 [...] [--write]``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from football_fatigue.ingestion import EventVersion, approved_output_root, safe_output_path
from football_fatigue.matching.benchmark import SOURCE_DIR, load_source

from .detect import DetectConfig, duplicate_positions, segments


def project(verts: np.ndarray, p: np.ndarray) -> float:
    """Arc length along the polyline ``verts`` (n, 2) of the nearest point to ``p``; the first one on ties (Shapely semantics).
    Zero-length segments (repeated points) contribute no length and are never the nearest-point segment."""
    a, b = verts[:-1], verts[1:]
    d = b - a
    L = np.hypot(d[:, 0], d[:, 1])
    cum = np.r_[0.0, np.cumsum(L)]
    ok = L > 0
    u = np.where(ok, ((p - a) * d).sum(1) / np.where(ok, L ** 2, 1.0), 0.0).clip(0, 1)
    dist = np.hypot(*(a + u[:, None] * d - p).T)
    dist = np.where(ok, dist, np.inf)
    k = int(np.argmin(dist))
    return float(cum[k] + u[k] * L[k])


def time_at(verts: np.ndarray, times: np.ndarray, s: float) -> tuple[float, int]:
    """The advisor's step: segment containing arc length ``s`` (``searchsorted`` on cumulative lengths), linear time inside it."""
    L = np.hypot(*np.diff(verts, axis=0).T)
    cum = np.r_[0.0, np.cumsum(L)]
    k = int(np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(L) - 1))
    while L[k] == 0 and k > 0:                                     # repeated point: step back to a real segment
        k -= 1
    f = 0.0 if L[k] == 0 else (s - cum[k]) / L[k]
    return float(times[k] + f * (times[k + 1] - times[k])), k


def advisor_times(verts_a, times_a, verts_b, times_b, p) -> tuple[float, float]:
    """t_A and t_B of point ``p`` by the advisor's recipe (no equality is imposed)."""
    return time_at(verts_a, times_a, project(verts_a, p))[0], time_at(verts_b, times_b, project(verts_b, p))[0]


def line_point(a0, a1, b0, b1) -> np.ndarray | None:
    """Intersection of the supporting lines of two segments by a linear solve (None when parallel)."""
    M = np.array([[a1[0] - a0[0], b0[0] - b1[0]], [a1[1] - a0[1], b0[1] - b1[1]]])
    if abs(np.linalg.det(M)) < 1e-12:
        return None
    u, _ = np.linalg.solve(M, np.asarray(b0) - np.asarray(a0))
    return np.asarray(a0) + u * (np.asarray(a1) - np.asarray(a0))


def run_bounds(seg: dict[str, np.ndarray], g: int, lo: float, hi: float) -> tuple[int, int]:
    """First and last index of the contiguous run of segments (same slot, joined end to start) containing segment ``g`` and
    lying inside [lo, hi]."""
    s = seg["slot"][g]
    i = g
    while i - 1 >= 0 and seg["slot"][i - 1] == s and seg["row1"][i - 1] == seg["row0"][i] and seg["t0"][i - 1] >= lo - 1e-9:
        i -= 1
    j = g
    while j + 1 < len(seg["slot"]) and seg["slot"][j + 1] == s and seg["row0"][j + 1] == seg["row1"][j] and seg["t1"][j + 1] <= hi + 1e-9:
        j += 1
    return i, j


def polyline(seg: dict[str, np.ndarray], g: int, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and times of the run of :func:`run_bounds`."""
    i, j = run_bounds(seg, g, lo, hi)
    ix = np.arange(i, j + 1)
    verts = np.r_[np.c_[seg["x0"][ix], seg["y0"][ix]], [[seg["x1"][j], seg["y1"][j]]]]
    times = np.r_[seg["t0"][ix], seg["t1"][j]]
    return verts, times


def crosscheck(raw_root: str, match_id: str, tol_t: float = 1e-6, tol_m: float = 1e-6) -> dict[str, Any]:
    c, mask, _enc, _m, _prov = load_source(raw_root, match_id)
    with np.load(safe_output_path(raw_root, match_id, EventVersion.ROOT, approved_output_root(raw_root) / SOURCE_DIR, suffix=".quality.npz")) as z:
        steps = {k[5:]: z[k] for k in z.files if k.startswith("step_")}
    seg, _ = segments(c, mask, steps, duplicate_positions(c))
    key = {(int(seg["slot"][g]), int(seg["row0"][g])): g for g in range(len(seg["slot"]))}
    half = DetectConfig().half
    path = approved_output_root(raw_root) / "m11-intersections-v3" / str(match_id) / "root.events.jsonl"     # = v1 events (checked in S1.1)
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    out = dict(events=len(events), point_checked=0, point_max_err_m=0.0, time_agree=0, self_revisit=0, time_max_err_s_agree=0.0,
               revisit_examples=[], by_type={})
    for e in events:
        ga, gb = key[(e["slot_a"], e["segment_a"]["row0"])], key[(e["slot_b"], e["segment_b"]["row0"])]
        p = np.array([e["x"], e["y"]])
        if e["type"] == "proper_crossing":
            q = line_point((seg["x0"][ga], seg["y0"][ga]), (seg["x1"][ga], seg["y1"][ga]), (seg["x0"][gb], seg["y0"][gb]), (seg["x1"][gb], seg["y1"][gb]))
            if q is not None:
                out["point_checked"] += 1
                out["point_max_err_m"] = max(out["point_max_err_m"], float(np.hypot(*(q - p))))
        w = e["windows"][0]
        va, ta = polyline(seg, ga, w - half, w + half)
        vb, tb = polyline(seg, gb, w - half, w + half)
        tA, tB = advisor_times(va, ta, vb, tb, p)
        err = max(abs(tA - e["time_a_s"]), abs(tB - e["time_b_s"]))
        bt = out["by_type"].setdefault(e["type"], dict(n=0, agree=0))
        bt["n"] += 1
        if err <= tol_t:
            out["time_agree"] += 1
            bt["agree"] += 1
            out["time_max_err_s_agree"] = max(out["time_max_err_s_agree"], err)
        else:
            out["self_revisit"] += 1
            if len(out["revisit_examples"]) < 5:
                out["revisit_examples"].append(dict(event_id=e["event_id"], v1=[e["time_a_s"], e["time_b_s"]], advisor=[tA, tB]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m football_fatigue.intersections.advisor")
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--match", nargs="+", required=True)
    ap.add_argument("--write", action="store_true", help="write m11-advisor-check-v1/<match>.json (never overwritten)")
    a = ap.parse_args(argv)
    root = str(Path(a.root).resolve())
    for mid in a.match:
        r = crosscheck(root, mid) | dict(match=mid)
        print(json.dumps({k: v for k, v in r.items() if k != "revisit_examples"}, default=str), flush=True)
        if a.write:
            out = approved_output_root(root) / "m11-advisor-check-v1" / f"{mid}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "x", encoding="utf-8") as f:
                json.dump(r, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
