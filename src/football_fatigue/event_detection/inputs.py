"""Source-agnostic position input for event detection.

The detector sees only positions and time: per frame a ball (x, y, z or missing) and the players of each side. Nothing
else from the source reaches it. In particular, for PFF tracking:

- the ball is the **raw** ``balls`` measurement only. ``ballsSmoothed`` is built by PFF *using the event labels* (the ball
  is placed at the carrier between events), so using it would leak the answer into the detector;
- the ``game_event`` / ``possession_event`` overlays embedded in tracking frames are never read;
- player positions are the raw measurements (``CoordinatePolicy.RAW``).

``MatchPositions`` is columnar (numpy): one row per unique frame, one column per player key seen in the match. Missing is
NaN, never zero. A later CV adapter only has to fill the same structure (``side`` from kit/team classification, ``key``
from the track id).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from football_fatigue.ingestion import Dataset, EventVersion, MatchMetadata
from football_fatigue.match_data import CoordinatePolicy, CoordState, MatchData

HOME, AWAY = 0, 1
SIDES = ("home", "away")
CACHE_FORMAT = 1


@dataclass
class MatchPositions:
    match_id: int
    source: str                      # e.g. "pff_tracking_raw" or "cv"
    fps: float
    home_team_id: int | None
    away_team_id: int | None
    home_attacks_right: dict[int, bool]   # period -> True when the home team attacks +x
    t: np.ndarray                     # (F,) seconds of video time
    period: np.ndarray                # (F,) int
    frame_num: np.ndarray             # (F,) int
    ball: np.ndarray                  # (F, 3) x, y, z in metres; NaN = missing (z may be NaN alone)
    px: np.ndarray                    # (F, P) player x; NaN = not observed in that frame
    py: np.ndarray                    # (F, P)
    player_side: np.ndarray           # (P,) HOME / AWAY
    player_key: list[str]             # (P,) stable key within the match, e.g. "home:10"
    player_id: np.ndarray             # (P,) source player id, -1 when unknown
    excluded_periods: tuple[int, ...] = ()
    notes: list[str] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return int(self.t.shape[0])

    def team_id_of_side(self, side: int) -> int | None:
        return self.home_team_id if side == HOME else self.away_team_id

    def attacks_right(self, side: int, period: int) -> bool | None:
        home_right = self.home_attacks_right.get(int(period))
        if home_right is None:
            return None
        return home_right if side == HOME else not home_right

    # -- cache (derived data; never written into data/raw)
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(format=CACHE_FORMAT, match_id=self.match_id, source=self.source, fps=self.fps, home_team_id=self.home_team_id,
                    away_team_id=self.away_team_id, home_attacks_right={str(k): v for k, v in self.home_attacks_right.items()},
                    player_key=self.player_key, excluded_periods=list(self.excluded_periods), notes=self.notes)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, t=self.t, period=self.period, frame_num=self.frame_num, ball=self.ball, px=self.px, py=self.py,
                            player_side=self.player_side, player_id=self.player_id, meta=np.array(json.dumps(meta)))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "MatchPositions":
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            if meta.get("format") != CACHE_FORMAT:
                raise ValueError(f"unsupported positions cache format in {path}")
            return cls(match_id=meta["match_id"], source=meta["source"], fps=meta["fps"], home_team_id=meta["home_team_id"],
                       away_team_id=meta["away_team_id"], home_attacks_right={int(k): v for k, v in meta["home_attacks_right"].items()},
                       t=z["t"], period=z["period"], frame_num=z["frame_num"], ball=z["ball"], px=z["px"], py=z["py"],
                       player_side=z["player_side"], player_key=list(meta["player_key"]), player_id=z["player_id"],
                       excluded_periods=tuple(meta["excluded_periods"]), notes=list(meta["notes"]))


def attack_directions(meta: MatchMetadata) -> dict[int, bool]:
    """Period -> home attacks +x. From ``homeTeamStartLeft`` (verified 64/64 in the audit); extra time from its own flag."""
    out: dict[int, bool] = {}
    if meta.home_team_start_left is not None:
        out[1] = bool(meta.home_team_start_left)
        out[2] = not out[1]
    if meta.home_team_start_left_extra_time is not None:
        out[3] = bool(meta.home_team_start_left_extra_time)
        out[4] = not out[3]
    return out


def from_match_data(md: MatchData, *, exclude_periods: Iterable[int] = (), max_frames: int | None = None) -> MatchPositions:
    """Build positions from PFF tracking: raw players, raw ball, no event overlay, frames of ``exclude_periods`` dropped."""
    if md.policy is not CoordinatePolicy.RAW:
        raise ValueError("event detection reads raw positions only (CoordinatePolicy.RAW)")
    meta = md.metadata
    excluded = tuple(sorted(set(int(p) for p in exclude_periods)))
    keys: dict[str, int] = {}
    sides: list[int] = []
    pids: list[int] = []
    t, period, fnum, ball = [], [], [], []
    cols: list[dict[int, tuple[float, float]]] = []
    dropped = 0
    for pf in md.frames(max_frames=max_frames):
        per = pf.canonical_period
        if per is None or per in excluded:
            dropped += 1
            continue
        t.append(pf.video_time_s)
        period.append(per)
        fnum.append(pf.frame_num)
        r = pf.ball.raw
        if r.state is CoordState.AVAILABLE:
            ball.append((r.x, r.y, np.nan if r.z is None else r.z))
        else:
            ball.append((np.nan, np.nan, np.nan))       # missing or several raw balls: never resolved here
        row: dict[int, tuple[float, float]] = {}
        for p in pf.players:
            c = p.raw
            if c.state is not CoordState.AVAILABLE or p.jersey_number is None:
                continue
            key = f"{p.source_side}:{p.jersey_number}"
            j = keys.get(key)
            if j is None:
                j = keys[key] = len(sides)
                sides.append(HOME if p.source_side == "home" else AWAY)
                pids.append(p.player_id if (p.identity_usable and p.player_id is not None) else -1)
            row[j] = (c.x, c.y)
        cols.append(row)
    n_f, n_p = len(t), len(sides)
    px = np.full((n_f, n_p), np.nan, dtype=np.float32)
    py = np.full((n_f, n_p), np.nan, dtype=np.float32)
    for i, row in enumerate(cols):
        for j, (x, y) in row.items():
            px[i, j] = x
            py[i, j] = y
    notes = [f"{dropped} frames dropped (invalid or excluded period)"] if dropped else []
    return MatchPositions(match_id=int(md.match_id), source="pff_tracking_raw", fps=float(meta.fps), home_team_id=meta.home_team_id,
                          away_team_id=meta.away_team_id, home_attacks_right=attack_directions(meta), t=np.asarray(t, dtype=np.float64),
                          period=np.asarray(period, dtype=np.int8), frame_num=np.asarray(fnum, dtype=np.int64),
                          ball=np.asarray(ball, dtype=np.float32).reshape(n_f, 3), px=px, py=py, player_side=np.asarray(sides, dtype=np.int8),
                          player_key=list(keys), player_id=np.asarray(pids, dtype=np.int64), excluded_periods=excluded, notes=notes)


def load_positions(root: str | Path, match_id: object, *, cache_dir: Path | None, exclude_periods: Iterable[int] = (),
                   refresh: bool = False) -> MatchPositions:
    """Positions for one match, read from the cache when present (the tracking pass takes about a minute per match)."""
    excluded = tuple(sorted(set(int(p) for p in exclude_periods)))
    md = MatchData.open(Dataset(root), match_id, EventVersion.ROOT, policy=CoordinatePolicy.RAW)
    path = None
    if cache_dir is not None:
        suffix = "_excl" + "-".join(map(str, excluded)) if excluded else ""
        path = cache_dir / f"{md.match_id}{suffix}.positions.npz"
        if path.exists() and not refresh:
            return MatchPositions.load(path)
    pos = from_match_data(md, exclude_periods=excluded)
    if path is not None:
        pos.save(path)
    return pos
