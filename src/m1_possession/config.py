"""All parameters of the M1 ball smoother, with defaults (one place)."""
from __future__ import annotations

from dataclasses import asdict, dataclass

METHODS = ("linear", "savgol", "rts", "rts_adaptive")


@dataclass(frozen=True)
class SmootherConfig:
    # a. repeated samples: in a run of identical consecutive positions only every `repeat_keep_every`-th frame is a
    #    real update (the raw ball is held for two frames, about 15 Hz). A ball at rest keeps samples at that rate.
    repeat_keep_every: int = 2
    # b. impossible jumps
    max_speed_mps: float = 40.0         # implied speed above this is impossible
    spike_passes: int = 3               # repeated one-sample spike removal passes
    # c. gaps: interpolate only when two kept samples are at most this far apart; longer gaps stay missing
    max_gap_s: float = 0.5
    # d. smoothing method: 'linear' (interpolation + central differences, no smoothing), 'savgol',
    #    'rts' (constant-velocity Kalman + Rauch-Tung-Striebel), 'rts_adaptive' (rts that raises the process
    #    noise for one step when the innovation is too large, and lets the velocity jump at kicks detected in the
    #    track itself: keeps kicks sharp)
    method: str = "rts_adaptive"
    # Savitzky-Golay (on the frame grid, after interpolation)
    savgol_window: int = 9              # frames, odd
    savgol_poly: int = 2
    # Kalman / RTS (white-noise acceleration model, per axis)
    rts_q: float = 30.0                 # process noise spectral density (m^2/s^3)
    rts_r_m: float = 0.25               # measurement noise standard deviation (m)
    rts_adapt_nis: float = 13.8         # chi-square (2 dof, 99.9 %) threshold on the x/y innovation
    rts_adapt_factor: float = 1000.0    # process noise multiplier for the flagged step
    # rts_adaptive also lets the velocity jump at kicks found in the track itself: frames where the change between
    # the mean velocity over the next and the previous `kick_frames` frames (linear interpolation of the kept
    # samples) is a local maximum and at least `kick_dv_mps`; there the velocity variance gets `kick_var` added.
    kick_dv_mps: float = 3.0
    kick_frames: int = 3
    kick_var: float = 400.0             # (m/s)^2
    # input
    drop_estimated: bool = False        # drop raw samples whose visibility is ESTIMATED

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")
        if self.repeat_keep_every < 1:
            raise ValueError("repeat_keep_every must be >= 1")
        if not self.max_speed_mps > 0 or not self.max_gap_s > 0:
            raise ValueError("max_speed_mps and max_gap_s must be positive")
        if self.savgol_window < 3 or self.savgol_window % 2 == 0 or not 0 <= self.savgol_poly < self.savgol_window:
            raise ValueError("savgol_window must be odd >= 3 and savgol_poly < savgol_window")
        if not (self.rts_q > 0 and self.rts_r_m > 0 and self.rts_adapt_factor >= 1):
            raise ValueError("rts_q, rts_r_m must be positive and rts_adapt_factor >= 1")
        if not (self.kick_dv_mps > 0 and self.kick_frames >= 1 and self.kick_var >= 0):
            raise ValueError("kick_dv_mps must be positive, kick_frames >= 1, kick_var >= 0")

    def as_dict(self) -> dict:
        return asdict(self)


# Scratch space for long runs (one place). C: is nearly full, so TEMP / TMP, joblib's temp folder, LightGBM / numpy
# temporary files and any Optuna storage go to M1_TMP (override with the M1_TMP environment variable).
M1_TMP_DEFAULT = r"D:\m1_tmp"


def use_scratch() -> str:
    """Point temporary files of this process and its child processes to the M1 scratch folder; returns its path."""
    import os
    import tempfile

    root = os.environ.get("M1_TMP", M1_TMP_DEFAULT)
    os.makedirs(root, exist_ok=True)
    for k in ("TMP", "TEMP", "TMPDIR", "JOBLIB_TEMP_FOLDER", "M1_TMP"):
        os.environ[k] = root
    tempfile.tempdir = root
    return root


def optuna_storage(name: str) -> str:
    """SQLite storage URL for an Optuna study kept in the scratch folder (use with optuna.create_study(storage=...))."""
    import os

    return "sqlite:///" + os.path.join(use_scratch(), f"{name}.db").replace("\\", "/")
