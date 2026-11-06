"""Master timeline: one clock for every panel.

Master time ``T`` is the reference camera's presentation time. A camera with offset ``offset_s`` shows the same physical
instant at source time ``T + offset_s`` (for the Wadi Degla pair: fixed = moving + tau). Every rendered instant records the
requested source time, the source frame actually used, and the residual; a frame further away than ``tolerance_s`` is
reported as unavailable instead of being substituted silently.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraClock:
    name: str
    fps: float
    frame_count: int
    offset_s: float = 0.0          # source time = master time + offset_s
    first_pts_s: float = 0.0       # PTS of frame 0 (constant-frame-rate files: frame i at first_pts_s + i / fps)

    def frame_time(self, index: int) -> float:
        return self.first_pts_s + index / self.fps


@dataclass(frozen=True)
class ResolvedFrame:
    camera: str
    requested_source_s: float
    index: int | None
    source_s: float | None
    residual_s: float | None
    status: str                    # "ok" | "unavailable"

    def to_dict(self) -> dict:
        return {"camera": self.camera, "requested_source_s": round(self.requested_source_s, 4), "index": self.index,
                "source_s": None if self.source_s is None else round(self.source_s, 4),
                "residual_s": None if self.residual_s is None else round(self.residual_s, 4), "status": self.status}


def resolve(clock: CameraClock, master_s: float, tolerance_s: float | None = None) -> ResolvedFrame:
    """Nearest source frame to the master instant; unavailable when outside the file or beyond the tolerance."""
    tol = 0.5 / clock.fps + 1e-6 if tolerance_s is None else tolerance_s
    want = master_s + clock.offset_s
    idx = round((want - clock.first_pts_s) * clock.fps)
    if idx < 0 or idx >= clock.frame_count:
        return ResolvedFrame(clock.name, want, None, None, None, "unavailable")
    t = clock.frame_time(idx); res = t - want
    if abs(res) > tol:
        return ResolvedFrame(clock.name, want, idx, t, res, "unavailable")
    return ResolvedFrame(clock.name, want, idx, t, res, "ok")


def observation_state(obs_time_s: float | None, source_s: float | None, tolerance_s: float) -> str:
    """Overlay freshness: 'current' when the observation was made on (or within tolerance of) the displayed source frame,
    'stale' when it is older/newer, 'missing' when there is no observation or no frame."""
    if obs_time_s is None or source_s is None:
        return "missing"
    return "current" if abs(obs_time_s - source_s) <= tolerance_s + 1e-6 else "stale"


def master_grid(t0: float, t1: float, step_s: float) -> list[float]:
    n = int(round((t1 - t0) / step_s))
    return [round(t0 + i * step_s, 6) for i in range(n + 1)]
