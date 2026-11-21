"""Match trajectories for playback (RAW provider coordinates, the same verified Step 1 source and segments as the catalogue).

Coordinates: pitch metres centred on the centre spot (x along the length, y across), as produced by Step 1 and drawn by
``m11_dashboard.pitch.pitch_shapes``. Times: the source video clock in seconds (``video_time_s``), the catalogue's time base."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from football_fatigue.m11_phase2 import common as C

from . import selfcross as SX


@dataclass
class MatchTracks:
    match: str
    c: object                  # encounters.trajectories.RawCache
    mask: np.ndarray           # Step 1 quality bitmask (frames x slots)
    seg: dict[str, np.ndarray]  # catalogue segments (valid native segments with chain ids)
    pitch: tuple[float, float]

    def periods(self) -> pd.DataFrame:
        rows = []
        for p in sorted(int(v) for v in np.unique(self.c.canonical_period) if v > 0):
            t = self.c.video_time_s[self.c.canonical_period == p]
            rows.append(dict(period=p, start=float(t[0]), end=float(t[-1])))
        return pd.DataFrame(rows)

    def slots(self) -> pd.DataFrame:
        c = self.c
        return pd.DataFrame(dict(slot=np.arange(len(c.slot_side)), side=c.slot_side.astype(str), jersey=c.slot_jersey.astype(int),
                                 player_id=c.slot_player_id.astype(np.int64)))

    def frames(self, period: int, t0: float, t1: float, hz: float = 5.0) -> pd.DataFrame:
        """Long table (t, slot, x, y, estimated) resampled to ``hz`` by nearest source frame; unavailable samples are dropped."""
        from football_fatigue.encounters import quality as Q
        from football_fatigue.encounters.trajectories import STATE_AVAILABLE
        c = self.c
        rows = np.flatnonzero((c.canonical_period == period) & (c.video_time_s >= t0) & (c.video_time_s <= t1))
        if not len(rows):
            return pd.DataFrame(columns=["t", "slot", "x", "y", "estimated"])
        tt = c.video_time_s[rows]
        grid = np.arange(tt[0], tt[-1] + 1e-9, 1.0 / hz)
        pick = rows[np.clip(np.searchsorted(tt, grid), 0, len(rows) - 1)]
        n_s = c.x.shape[1]
        av = c.state[pick] == STATE_AVAILABLE
        df = pd.DataFrame(dict(t=np.repeat(np.round(grid, 3), n_s), slot=np.tile(np.arange(n_s), len(pick)), x=c.x[pick].ravel(),
                               y=c.y[pick].ravel(), estimated=((self.mask[pick] & Q.ESTIMATED) != 0).ravel(), ok=av.ravel()))
        return df[df["ok"] & np.isfinite(df["x"])].drop(columns="ok").reset_index(drop=True)

    def path(self, slot: int, period: int, t0: float, t1: float) -> pd.DataFrame:
        """Full-rate positions of one player (for event zooms and hidden-path reveal)."""
        from football_fatigue.encounters import quality as Q
        c = self.c
        rows = np.flatnonzero((c.canonical_period == period) & (c.video_time_s >= t0) & (c.video_time_s <= t1))
        d = pd.DataFrame(dict(t=c.video_time_s[rows], x=c.x[rows, slot], y=c.y[rows, slot],
                              estimated=(self.mask[rows, slot] & Q.ESTIMATED) != 0))
        return d[np.isfinite(d["x"])].reset_index(drop=True)

    def self_crossings(self, period: int, t0: float, t1: float, slots=None, cfg: SX.SelfCrossConfig = SX.SelfCrossConfig()) -> pd.DataFrame:
        from football_fatigue.encounters import quality as Q
        c = self.c

        def est_of(slot, t):
            r = np.clip(np.searchsorted(c.video_time_s, t), 0, len(c.video_time_s) - 1)
            return (self.mask[r, slot] & Q.ESTIMATED) != 0
        return SX.segments_self_crossings(self.seg, est_of, slots=slots, period=period, t0=t0, t1=t1, cfg=cfg)


@lru_cache(maxsize=2)
def load(match: str) -> MatchTracks:
    """Load one development match (guard first; RAW Step 1 cache verified against its manifest by ``load_source``)."""
    m = C.checked([str(match)])[0]
    from football_fatigue.ingestion import Dataset
    from football_fatigue.intersections.catalogue import load_steps
    from football_fatigue.interaction_catalogue.run import RAW
    from football_fatigue.matching.benchmark import load_source
    from football_fatigue.window_intersections import core as W
    c, mask, _enc, _man, _prov = load_source(str(RAW), m)
    if str(c.match_id) != m:
        raise SystemExit(f"loaded cache is match {c.match_id}, expected {m}")
    seg, _counts, _dup = W.build_segments(c, mask, load_steps(str(RAW), m))
    md = Dataset(str(RAW)).metadata(m)
    return MatchTracks(m, c, mask, seg, (float(md.pitch_length_m), float(md.pitch_width_m)))
