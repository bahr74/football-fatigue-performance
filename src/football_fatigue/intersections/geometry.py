"""Vectorised intersection of two polylines' line segments (M11 advisor-directed intersection pipeline).

Segments are given as start and end points ``(n, 2)``. All pairs ``A[i] x B[j]`` are tested; for every touching pair the result
gives the intersection point, the parameters ``ta`` / ``tb`` in [0, 1] along each segment and a kind:

- ``proper``: the segments cross at a point strictly inside both;
- ``endpoint``: they meet at a point that is an endpoint of at least one of them (a vertex touch or a crossing through a vertex;
  :func:`classify_vertex` decides which, from the neighbouring vertices);
- ``collinear``: they lie on one line and overlap along a stretch of positive length; the point is the overlap's midpoint.

Tolerances are absolute, in metres (``tol_m``): a point within ``tol_m`` of a segment end counts as that end, and segments
shorter than ``tol_m`` are degenerate and never tested (the caller counts them). Coordinates are used as given.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PROPER, ENDPOINT, COLLINEAR = "proper", "endpoint", "collinear"


@dataclass(frozen=True)
class Hits:
    i: np.ndarray          # index into A's segments
    j: np.ndarray          # index into B's segments
    x: np.ndarray          # intersection point (overlap midpoint for collinear)
    y: np.ndarray
    ta: np.ndarray         # parameter along A[i], 0 = start, 1 = end
    tb: np.ndarray
    kind: np.ndarray       # PROPER / ENDPOINT / COLLINEAR

    def __len__(self) -> int:
        return len(self.i)


def _cross(ax, ay, bx, by):
    return ax * by - ay * bx


def bbox_overlap(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray, tol_m: float) -> np.ndarray:
    """(nA, nB) True where the segments' bounding boxes (grown by ``tol_m``) overlap."""
    axl, axh = np.minimum(a0[:, 0], a1[:, 0])[:, None] - tol_m, np.maximum(a0[:, 0], a1[:, 0])[:, None] + tol_m
    ayl, ayh = np.minimum(a0[:, 1], a1[:, 1])[:, None] - tol_m, np.maximum(a0[:, 1], a1[:, 1])[:, None] + tol_m
    bxl, bxh = np.minimum(b0[:, 0], b1[:, 0])[None], np.maximum(b0[:, 0], b1[:, 0])[None]
    byl, byh = np.minimum(b0[:, 1], b1[:, 1])[None], np.maximum(b0[:, 1], b1[:, 1])[None]
    return (axl <= bxh) & (bxl <= axh) & (ayl <= byh) & (byl <= ayh)


def intersect(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray, tol_m: float = 1e-6) -> Hits:
    """Every touching segment pair of A x B (see the module docstring). Degenerate segments (length < ``tol_m``) are skipped."""
    a0, a1, b0, b1 = (np.asarray(v, float).reshape(-1, 2) for v in (a0, a1, b0, b1))
    la, lb = np.hypot(*(a1 - a0).T), np.hypot(*(b1 - b0).T)
    cand = bbox_overlap(a0, a1, b0, b1, tol_m) & (la >= tol_m)[:, None] & (lb >= tol_m)[None]
    i, j = np.nonzero(cand)
    empty = Hits(*(np.zeros(0, int),) * 2, *(np.zeros(0),) * 4, np.zeros(0, "<U9"))
    if not len(i):
        return empty
    p, r = a0[i], a1[i] - a0[i]
    q, s = b0[j], b1[j] - b0[j]
    lr, ls = la[i], lb[j]
    qp = q - p
    den = _cross(r[:, 0], r[:, 1], s[:, 0], s[:, 1])
    # |den| = |r||s| |sin(angle)|: treated as parallel when |sin(angle)| <= 1e-10 (exactly or numerically parallel)
    par = np.abs(den) <= 1e-10 * lr * ls
    out_i, out_j, out_x, out_y, out_ta, out_tb, out_k = [], [], [], [], [], [], []

    # non-parallel: unique line intersection
    n = ~par
    if n.any():
        ta = _cross(qp[n, 0], qp[n, 1], s[n, 0], s[n, 1]) / den[n]
        tb = _cross(qp[n, 0], qp[n, 1], r[n, 0], r[n, 1]) / den[n]
        ea, eb = tol_m / lr[n], tol_m / ls[n]                     # tolerance in parameter units
        ok = (ta >= -ea) & (ta <= 1 + ea) & (tb >= -eb) & (tb <= 1 + eb)
        at_end = (ta <= ea) | (ta >= 1 - ea) | (tb <= eb) | (tb >= 1 - eb)
        ta, tb = np.clip(ta, 0, 1), np.clip(tb, 0, 1)
        pts = p[n] + ta[:, None] * r[n]
        out_i.append(i[n][ok]), out_j.append(j[n][ok]), out_x.append(pts[ok, 0]), out_y.append(pts[ok, 1])
        out_ta.append(ta[ok]), out_tb.append(tb[ok]), out_k.append(np.where(at_end[ok], ENDPOINT, PROPER))

    # parallel: collinear only if B's start lies on A's line; then intersect the projections on A
    if par.any():
        k = np.flatnonzero(par)
        off = np.abs(_cross(qp[k, 0], qp[k, 1], r[k, 0], r[k, 1])) / lr[k]          # distance of b0 from A's line
        k = k[off <= tol_m]
        if len(k):
            rr = (r[k] ** 2).sum(1)
            t0 = (qp[k] * r[k]).sum(1) / rr                                          # b0 and b1 as parameters on A
            t1 = t0 + (s[k] * r[k]).sum(1) / rr
            lo, hi = np.maximum(np.minimum(t0, t1), 0.0), np.minimum(np.maximum(t0, t1), 1.0)
            span = (hi - lo) * lr[k]
            keep = span >= -tol_m
            k, lo, hi, span, t0, t1 = k[keep], lo[keep], hi[keep], span[keep], t0[keep], t1[keep]
            mid = (lo + hi) / 2
            pts = p[k] + mid[:, None] * r[k]
            tb = (mid - t0) / np.where(t1 != t0, t1 - t0, 1.0)
            out_i.append(i[k]), out_j.append(j[k]), out_x.append(pts[:, 0]), out_y.append(pts[:, 1])
            out_ta.append(mid), out_tb.append(np.clip(tb, 0, 1)), out_k.append(np.where(span > tol_m, COLLINEAR, ENDPOINT))
    if not out_i:
        return empty
    cat = [np.concatenate(v) for v in (out_i, out_j, out_x, out_y, out_ta, out_tb, out_k)]
    order = np.lexsort((cat[1], cat[0]))
    return Hits(*(v[order] for v in cat))


def side(p: np.ndarray, a: np.ndarray, b: np.ndarray, tol_m: float = 1e-6) -> int:
    """+1 / -1 for the side of point ``p`` relative to the directed line a -> b, 0 when within ``tol_m`` of it."""
    d = b - a
    n = np.hypot(*d)
    if n < tol_m:
        return 0
    v = _cross(d[0], d[1], p[0] - a[0], p[1] - a[1]) / n
    return 0 if abs(v) <= tol_m else int(np.sign(v))


def classify_vertex(a_before: np.ndarray, a_after: np.ndarray, b_before: np.ndarray, b_after: np.ndarray, tol_m: float = 1e-6) -> str:
    """For a contact at a vertex (or along an overlap): ``crossing`` when A's points before and after the contact lie on opposite
    sides of B's local line (b_before -> b_after), ``touch`` when on the same side, ``undetermined`` when either lies on it."""
    s0, s1 = side(a_before, b_before, b_after, tol_m), side(a_after, b_before, b_after, tol_m)
    if s0 == 0 or s1 == 0:
        return "undetermined"
    return "crossing" if s0 != s1 else "touch"
