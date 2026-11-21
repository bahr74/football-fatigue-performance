"""Pixel-space box geometry (xyxy floats, origin top-left, y down)."""

from __future__ import annotations

from typing import Sequence

Box = tuple[float, float, float, float]


def area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection(a: Box, b: Box) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def iou(a: Box, b: Box) -> float:
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def ios(a: Box, b: Box) -> float:
    """Intersection over the smaller box: 1.0 when one box lies inside the other."""
    smaller = min(area(a), area(b))
    return intersection(a, b) / smaller if smaller > 0 else 0.0


def covered_fraction(target: Box, cover: Box) -> float:
    """Share of ``target`` covered by ``cover``."""
    a = area(target)
    return intersection(target, cover) / a if a > 0 else 0.0


def widen(b: Box, fraction: float) -> Box:
    """Widen horizontally by ``fraction`` of the width in total (half on each side), for limbs."""
    dx = (b[2] - b[0]) * fraction / 2.0
    return (b[0] - dx, b[1], b[2] + dx, b[3])


def union_box(boxes: Sequence[Box]) -> Box:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def nearer_camera(a: Box, b: Box) -> int:
    """0 if ``a`` is nearer the camera, 1 if ``b`` is: the larger bottom y is nearer for a raised camera.

    This is an image-space heuristic (both players on the ground, camera above the pitch); ties go to ``a``.
    """
    return 0 if a[3] >= b[3] else 1


def outside_fraction(b: Box, width: int, height: int) -> float:
    """Share of the box outside the image."""
    a = area(b)
    if a <= 0:
        return 0.0
    return 1.0 - intersection(b, (0.0, 0.0, float(width), float(height))) / a


def predict_constant_velocity(history: Sequence[tuple[int, Box]], frame: int, max_samples: int = 5) -> Box:
    """Extrapolate the last observed box to ``frame`` with the mean velocity of the last samples.

    ``history`` holds (frame, box) observations in increasing frame order; the box size is kept from the
    last observation. This is the engine's own prediction, not the tracker's Kalman state.
    """
    if not history:
        raise ValueError("history is empty")
    recent = list(history)[-max_samples:]
    last_f, last = recent[-1]
    if len(recent) >= 2:
        first_f, first = recent[0]
        span = last_f - first_f
        vx = ((last[0] + last[2]) - (first[0] + first[2])) / 2.0 / span if span else 0.0
        vy = ((last[1] + last[3]) - (first[1] + first[3])) / 2.0 / span if span else 0.0
    else:
        vx = vy = 0.0
    dt = frame - last_f
    return (last[0] + vx * dt, last[1] + vy * dt, last[2] + vx * dt, last[3] + vy * dt)


def predict_damped(history: Sequence[tuple[int, Box]], frame: int, full_frames: int = 5,
                   max_step_heights: float = 0.25, max_samples: int = 6) -> Box:
    """Bounded extrapolation for long gaps: median per-frame velocity of the last samples, capped at
    ``max_step_heights`` box heights per frame, applied for at most ``full_frames`` frames, then held."""
    if not history:
        raise ValueError("history is empty")
    recent = list(history)[-max_samples:]
    last_f, last = recent[-1]
    vxs, vys = [], []
    for (f0, b0), (f1, b1) in zip(recent, recent[1:]):
        span = f1 - f0
        if span > 0:
            vxs.append(((b1[0] + b1[2]) - (b0[0] + b0[2])) / 2.0 / span)
            vys.append(((b1[1] + b1[3]) - (b0[1] + b0[3])) / 2.0 / span)
    vx = sorted(vxs)[len(vxs) // 2] if vxs else 0.0
    vy = sorted(vys)[len(vys) // 2] if vys else 0.0
    cap = max_step_heights * max(last[3] - last[1], 1.0)
    speed = (vx * vx + vy * vy) ** 0.5
    if speed > cap:
        vx, vy = vx * cap / speed, vy * cap / speed
    dt = min(frame - last_f, full_frames)
    return (last[0] + vx * dt, last[1] + vy * dt, last[2] + vx * dt, last[3] + vy * dt)
