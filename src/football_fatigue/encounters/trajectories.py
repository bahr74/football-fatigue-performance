"""One-pass compact cache of RAW player observations for one match (M11 step 1).

Frames are read once through ``MatchData(policy=CoordinatePolicy.RAW)``, so parsing, repeated-record collapsing, the roster
join, quarantine and the raw coordinate states all come from the existing ingestion layer. Provider-smoothed coordinates are
never read. Values are copied, not transformed: ``x``/``y`` are the source floats, NaN unless the raw state is AVAILABLE.

Layout: per frame (F,), per slot (S,) and per frame x slot (F, S). A slot is one ``(source_side, jersey_number)`` as given by
the provider; it is an identity *label*, not a verified person.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from football_fatigue.ingestion import AttributionStatus, IngestionError, QualityFlag
from football_fatigue.match_data import CoordinatePolicy, CoordState, MatchData

STATE_ABSENT, STATE_AVAILABLE, STATE_MISSING, STATE_AMBIGUOUS = 0, 1, 2, 3
VISIBILITY = {"VISIBLE": 1, "ESTIMATED": 2}          # 0 = absent/unknown
CONFIDENCE = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}      # 0 = absent/unknown
ATTRIBUTION = {AttributionStatus.RESOLVED: 1, AttributionStatus.QUARANTINED: 2, AttributionStatus.UNRESOLVED_JERSEY: 3,
               AttributionStatus.NOT_ATTEMPTED: 4}
QFLAG_BIT = {f: 1 << i for i, f in enumerate(QualityFlag)}   # bit i = i-th QualityFlag member (order recorded in the manifest)
_STATE = {CoordState.AVAILABLE: STATE_AVAILABLE, CoordState.MISSING: STATE_MISSING, CoordState.AMBIGUOUS: STATE_AMBIGUOUS}
_CHUNK = 8192


def qflag_bits(flags) -> int:
    return sum(QFLAG_BIT[f] for f in flags)


@dataclass
class RawCache:
    match_id: int
    fps: float
    tracking_path: str
    quarantine_periods: tuple[int, ...]
    # per frame
    frame_num: np.ndarray          # int64
    video_time_s: np.ndarray       # float64, videoTimeMs / 1000 (source clock)
    source_period: np.ndarray      # int16, -1 when null
    canonical_period: np.ndarray   # int8, 0 when not 1-4
    record_first: np.ndarray       # int64, first source record index (collapsed records are contiguous)
    record_count: np.ndarray       # int16
    frame_gap_before: np.ndarray   # int32, -1 when none
    frame_qflags: np.ndarray       # int64 bitmask of QualityFlag
    # per slot
    slot_side: np.ndarray          # '<U4' home / away
    slot_jersey: np.ndarray        # int16
    slot_player_id: np.ndarray     # int64 from the roster (side, shirt) index, -1 when not in it
    slot_team_id: np.ndarray       # int64, -1 when not in it
    # per frame x slot
    x: np.ndarray                  # float64 raw x (m), NaN unless state AVAILABLE
    y: np.ndarray
    state: np.ndarray              # int8 STATE_*
    visibility: np.ndarray         # int8 VISIBILITY
    confidence: np.ndarray         # int8 CONFIDENCE
    attribution: np.ndarray        # int8 ATTRIBUTION
    qflags: np.ndarray             # int64 bitmask of the sample's QualityFlag set
    # counters
    samples_without_jersey: int = 0   # samples with an unparseable jersey: no slot, counted (never silently dropped)
    duplicate_slot_samples: int = 0   # a second sample of the same slot in one frame: both kept as AMBIGUOUS

    @property
    def n_frames(self) -> int:
        return len(self.frame_num)

    @property
    def n_slots(self) -> int:
        return len(self.slot_side)

    def arrays(self) -> dict[str, np.ndarray]:
        """Every field as an array (scalars as 0-d arrays), the content of ``root.cache.npz``."""
        return {k: (v if isinstance(v, np.ndarray) else np.asarray(v)) for k, v in vars(self).items()}


def load_cache(path, defaults: dict | None = None) -> RawCache:
    """Read a ``root.cache.npz`` back into a :class:`RawCache`. ``defaults`` supplies scalar fields that an older cache
    (arrays only) does not store; a value stored in the file always wins."""
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    for k, v in (defaults or {}).items():
        d.setdefault(k, v)
    scalars = dict(match_id=int, fps=float, tracking_path=str, samples_without_jersey=int, duplicate_slot_samples=int)
    missing = [k for k in (*scalars, "quarantine_periods") if k not in d]
    if missing:
        raise IngestionError(f"{path}: cache lacks {missing}")
    for k, f in scalars.items():
        d[k] = f(d[k])
    d["quarantine_periods"] = tuple(int(p) for p in np.atleast_1d(d["quarantine_periods"]))
    return RawCache(**d)


_FRAME_COLS = dict(frame_num=(np.int64, 0), video_time_s=(np.float64, np.nan), source_period=(np.int16, -1), canonical_period=(np.int8, 0),
                   record_first=(np.int64, -1), record_count=(np.int16, 0), frame_gap_before=(np.int32, -1), frame_qflags=(np.int64, 0))
_SAMPLE_COLS = dict(x=(np.float64, np.nan), y=(np.float64, np.nan), state=(np.int8, STATE_ABSENT), visibility=(np.int8, 0),
                    confidence=(np.int8, 0), attribution=(np.int8, 0), qflags=(np.int64, 0))


def _new_chunk(width: int) -> dict[str, np.ndarray]:
    c = {k: np.full(_CHUNK, fill, dt) for k, (dt, fill) in _FRAME_COLS.items()}
    c.update({k: np.full((_CHUNK, width), fill, dt) for k, (dt, fill) in _SAMPLE_COLS.items()})
    return c


def _widen(c: dict[str, np.ndarray], width: int) -> None:
    for k, (dt, fill) in _SAMPLE_COLS.items():
        old = c[k]
        c[k] = np.full((_CHUNK, width), fill, dt)
        c[k][:, :old.shape[1]] = old


def build_cache(md: MatchData, *, max_frames: int | None = None) -> RawCache:
    """Stream the match once and keep the raw observations. ``md`` must use ``CoordinatePolicy.RAW``."""
    if md.policy is not CoordinatePolicy.RAW:
        raise IngestionError("the encounter cache reads RAW coordinates only (CoordinatePolicy.RAW)")
    meta, index = md.metadata, md.roster.index()
    _, src = md.dataset.tracking_source(md.match_id)
    slots: dict[tuple[str, int], int] = {}
    chunks: list[dict[str, np.ndarray]] = []
    width, n, no_jersey, dup = 32, 0, 0, 0
    bits: dict[frozenset, int] = {}
    for pf in md.frames(max_frames=max_frames):
        fr = pf.frame
        j = n % _CHUNK
        if j == 0:
            chunks.append(_new_chunk(width))
        c = chunks[-1]
        idx = fr.source_record_indices
        if list(idx) != list(range(idx[0], idx[0] + len(idx))):
            raise IngestionError(f"frame {fr.frame_num}: collapsed source records are not contiguous {idx}")
        c["frame_num"][j], c["video_time_s"][j] = fr.frame_num, fr.video_time_s
        c["source_period"][j] = -1 if fr.source_period is None else fr.source_period
        c["canonical_period"][j] = fr.canonical_period or 0
        c["record_first"][j], c["record_count"][j] = idx[0], fr.source_record_count
        c["frame_gap_before"][j] = -1 if fr.frame_gap_before is None else fr.frame_gap_before
        c["frame_qflags"][j] = bits.setdefault(fr.quality_flags, qflag_bits(fr.quality_flags))
        for p in pf.players:
            if p.jersey_number is None:
                no_jersey += 1
                continue
            s = slots.setdefault((p.source_side, p.jersey_number), len(slots))
            if s >= width:
                width *= 2
                _widen(c, width)
            if c["state"][j, s] != STATE_ABSENT:      # two samples for one slot in one frame: keep neither value
                c["state"][j, s], c["x"][j, s], c["y"][j, s] = STATE_AMBIGUOUS, np.nan, np.nan
                dup += 1
                continue
            st = _STATE[p.raw.state]
            c["state"][j, s] = st
            if st == STATE_AVAILABLE:
                c["x"][j, s], c["y"][j, s] = p.raw.x, p.raw.y
            sm = p.sample
            c["visibility"][j, s] = VISIBILITY.get(sm.visibility, 0)
            c["confidence"][j, s] = CONFIDENCE.get(sm.confidence, 0)
            c["attribution"][j, s] = ATTRIBUTION[sm.attribution_status]
            c["qflags"][j, s] = bits.setdefault(sm.quality_flags, qflag_bits(sm.quality_flags))
        n += 1
    S = len(slots)
    out: dict[str, np.ndarray] = {}
    for k, (dt, fill) in _FRAME_COLS.items():
        out[k] = np.concatenate([c[k] for c in chunks])[:n] if chunks else np.zeros(0, dt)
    for k, (dt, fill) in _SAMPLE_COLS.items():
        a = np.full((n, S), fill, dt)
        for i, c in enumerate(chunks):
            rows = min(_CHUNK, n - i * _CHUNK)
            w = min(S, c[k].shape[1])
            a[i * _CHUNK:i * _CHUNK + rows, :w] = c[k][:rows, :w]
            c[k] = None                                   # release the chunk column as soon as it is copied
        out[k] = a
    keys = sorted(slots, key=slots.get)
    entries = [index.get(k) for k in keys]
    return RawCache(
        match_id=int(md.match_id), fps=float(meta.fps), tracking_path=src.path, quarantine_periods=tuple(md.quarantine_periods),
        slot_side=np.array([k[0] for k in keys], dtype="<U4"), slot_jersey=np.array([k[1] for k in keys], dtype=np.int16),
        slot_player_id=np.array([-1 if e is None else e.player_id for e in entries], dtype=np.int64),
        slot_team_id=np.array([-1 if e is None else e.team_id for e in entries], dtype=np.int64),
        samples_without_jersey=no_jersey, duplicate_slot_samples=dup, **out)
