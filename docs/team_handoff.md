# Developer handoff: the tracking-data interface

Status: **Milestone 2C, implemented and pending independent review; not formally accepted.** This guide is for the developer who builds tactical playback or tracking analytics on top of the data layer. Rules stay in [PROJECT_RULES.md](../PROJECT_RULES.md); the data semantics are in [data_contracts.md](data_contracts.md), [dataset_schema_audit.md](dataset_schema_audit.md), and [event_version_comparison.md](event_version_comparison.md). The Python examples below are executed by `tests/test_team_handoff_docs.py` against synthetic data, so they match the real API.

## 1. Setup

```powershell
Set-Location F:\Football-Fatigue-Project
py -3.12 -m venv .venv                                         # once; the environment is not committed
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest                           # no dataset needed
```

The raw dataset stays untouched in `data/raw/` (ignored by Git, read-only in practice). Generated files go only under `data/processed/`. There are no runtime dependencies beyond the standard library.

## 2. Open a match

A match is opened with an explicit match ID, an **explicit event version** (there is no default), and a coordinate policy (section 5). Opening reads nothing: metadata, roster and events are loaded on first access, and **tracking is never read until you iterate frames or request a quality scan**.

```python
import os
from football_fatigue.ingestion import EventVersion
from football_fatigue.match_data import COORDINATE_SYSTEM, CoordinatePolicy, CoordState, MatchData

ROOT = os.environ.get("FF_RAW_ROOT", "data/raw")      # the dataset root (read-only)
MATCH = os.environ.get("FF_MATCH", "3812")            # any match ID found by the inventory
md = MatchData.open(ROOT, MATCH, EventVersion.ROOT, policy=CoordinatePolicy.RAW)
print(md)
print(md.metadata.home_team_name, md.metadata.away_team_name, md.metadata.fps, md.metadata.pitch_length_m, md.metadata.pitch_width_m)
print(len(md.roster.entries), "roster entries;", len(md.events()), "event rows of version", md.event_version.value)
```

`EventVersion.ROOT` (June 3, 2025 content) is the provisional default recommendation for event content; positions always come from the tracking files. `MARCH_14_2025` and `MAY_1_2025` are separate versions: a missing version raises an error and is never replaced by another.

## 3. Iterate frames lazily

`md.frames()` is a generator in source order with bounded memory. **Frame numbers are not guaranteed to be consecutive**: nothing is interpolated or inserted into a gap, and several source records with the same frame number are collapsed into one frame whose `frame.source_record_indices` keeps every record.

```python
first = next(iter(md.frames()))
print(first.frame_num, first.video_time_s, "source period", first.source_period, "canonical period", first.canonical_period)
print("source records:", first.frame.source_record_indices, "file:", first.frame.source.path)
bounded = [pf.frame_num for pf in md.frames(max_frames=50)]        # a bounded look, nothing else is read
print(len(bounded), "frames read")
```

Coordinates are in the **pitch-centre system, in meters** (`COORDINATE_SYSTEM`): origin at the centre, +x to the right and +y upwards in the main-camera view, `z` upwards (ball). They are source values, never normalized for attack direction.

## 4. Read player and ball coordinates

Every player observation has three views: `raw` (the tracking measurement), `smoothed` (the source-supplied smoothed coordinate), and `selected` (what the policy chose, with `selected.source`). A `Coordinate` is `x`/`y`/`z` only when its state is `AVAILABLE`.

```python
pf = first
for p in pf.players[:3]:
    c = p.selected
    print(p.source_side, p.jersey_number, c.state.value, c.x, c.y, c.source, "position_usable:", p.position_usable,
          "identity_usable:", p.identity_usable, "player_id:", p.player_id, sorted(p.caveats))
b = pf.ball.selected
print("ball", b.state.value, b.x, b.y, b.z, b.source, "usable:", pf.ball.position_usable, sorted(f.value for f in pf.ball.quality_flags))
print(COORDINATE_SYSTEM["name"], COORDINATE_SYSTEM["unit"], first.quality_flags and sorted(f.value for f in first.quality_flags))
```

Provenance: the frame number, `frame.source_record_indices`, `frame.source.path`, the canonical `frame` itself, and the full ingestion sample `p.sample` (all `quality_flags` and `smoothed_candidates`) stay reachable from every prepared object.

## 5. Choose a coordinate policy

| Policy | `selected` is |
| --- | --- |
| `RAW` (default) | the raw measurement, or missing/ambiguous |
| `SMOOTHED` | the supplied smoothed coordinate only; **no fallback** when it is missing or ambiguous |
| `SMOOTHED_THEN_RAW` | smoothed when available, otherwise the raw position labelled `source="raw_fallback"` (with the reason and the caveat `RAW_FALLBACK_FOR_SMOOTHED`) |

```python
for policy in CoordinatePolicy:
    other = MatchData.open(ROOT, MATCH, EventVersion.ROOT, policy=policy)
    p = next(iter(other.frames(max_frames=1))).players[0]
    print(policy.value, "raw:", p.raw.state.value, "smoothed:", p.smoothed.state.value, "selected from:", p.selected.source)
```

No policy interpolates, extrapolates, smooths, or fills anything.

## 6. Interpret missing, ambiguous, and estimated observations

* **`MISSING`**: absent, null, or malformed in the source. `x`/`y` are `None`: **never draw a missing coordinate at the pitch centre or at (0, 0)**. Ball positions are missing in roughly a third of frames.
* **`AMBIGUOUS`**: several conflicting candidates (conflicting smoothed candidates, conflicting duplicate raw jerseys, several raw balls, or repeated source records that disagree). The candidates stay in `coordinate.candidates` / `p.sample.smoothed_candidates`; **none is selected**. A conflicting smoothed ball of a repeated frame is unresolved by design.
* **`ESTIMATED`**: the tracking source itself estimated the player's position instead of seeing the player (about 63 % of raw player positions). The coordinate is still `AVAILABLE`; the caveat `PLAYER_ESTIMATED` is the signal. Low jersey confidence appears as `JERSEY_CONFIDENCE_LOW`.
* **`position_usable`**: the selected coordinate is present, finite, free of an unresolved conflict, and the frame's source period is 1-4. **`identity_usable`**: the team/player was resolved through the roster join (side + shirt number) and is not quarantined. **Neither means physically validated.**

```python
counts = {}
for pf in md.frames(max_frames=500):
    for p in pf.players:
        key = (p.raw.state.value, p.smoothed.state.value, "estimated" if "PLAYER_ESTIMATED" in p.caveats else "other")
        counts[key] = counts.get(key, 0) + 1
print(sorted(counts.items(), key=lambda kv: -kv[1])[:5])
```

## 7. Events: aligned and unmatched

`md.synchronize()` aligns every event row of the requested version to the tracking by video time (same period, nearest frame, one-frame tolerance, earlier frame on a tie). **It streams the whole tracking file once** (about 80 s per match) and is cached. Every event is preserved; unmatched events keep their reason.

```python
sync = md.synchronize()                                   # expensive and explicit
print(sync.summary()["matched"], "matched,", sync.summary()["unmatched"], "unmatched; scan complete:", sync.scan_complete)
for a in md.unmatched_events():                           # never hidden
    print(a.event_row_id, a.reason.value, a.source_period, a.canonical_period)
attached = {}
for pf in md.frames(attach_events=True):                  # a second pass; matched events appear on their frame
    for a in pf.events:
        attached[a.event_row_id] = (pf.frame_num, a.residual_s, a.status.value, a.validated_correspondence)
print(len(attached), "events attached;", len(md.events()) - len(attached), "not attached (unmatched)")
```

A temporal match means only that a same-period frame exists within one frame interval. It is **not** physical verification: `a.validated_correspondence` is always `"not_assessed"`. Coordinate agreement is a separate, optional check (`md.synchronize(spatial=True)`), and the known ten-match event-coordinate discrepancies show up there as disagreement, uncorrected.

## 8. Quality summaries and scan completeness

`md.quality_summary(max_frames=None, include_synchronization=False)` is explicit and streams the tracking file; opening a match never does it. Its scan is **`complete`**, **`partial`** (a `max_frames` bound that stopped before the end of the file), or **`failed`** (the stream raised an error; partial counts are kept). **Period absence is asserted only for a complete scan**: otherwise `event_periods_without_tracking` is `None`.

```python
q = md.quality_summary(max_frames=2000)                   # bounded; use max_frames=None for the whole match
print(q.scan.status, q.scan.frames_scanned, q.scan.note)
for name in ("ball_raw_missing", "player_raw_missing", "player_raw_ambiguous", "player_smoothed_ambiguous", "player_estimated"):
    m = q.metrics[name]
    print(f"{name}: {m.count} of {m.denominator} {m.unit}", None if m.fraction is None else round(m.fraction, 4))
print(q.event_periods_without_tracking, q.warnings)
```

Counting units: `frames` = unique frames (repeated source records collapse); `player_observations` = one raw player entry in one frame (home or away); `events` = event rows of the chosen version. Metrics: `frames_period_unusable`, `frames_with_frame_gap`, `frames_with_repeated_source_records`, `frames_with_repeat_conflicts`, `ball_raw_*` and `ball_smoothed_*` (`available`/`missing`/`ambiguous`), `ball_selected_unusable` (frames); `player_raw_*`, `player_smoothed_*`, `player_selected_unusable`, `player_estimated`, `player_measured_visible`, `player_visibility_unknown`, `player_jersey_confidence_low`, `identity_<status>` (player observations); `events_total`, `events_invalid_source_period`, `events_missing_time`, `events_in_repeated_key_pairs`, and with `include_synchronization=True` `events_temporally_matched`/`events_unmatched` plus the `unmatched_reasons` breakdown (events). Breakdowns carry their own unit in `q.breakdown_units`. A player who has no record in a frame is *absent*, not a zero position: see `players_per_side_per_frame`.

## 9. Building tactical playback

Do not assume continuous frames. Use `video_time_s` as the clock, treat each period separately, and decide how a consumer shows a gap.

```python
def draw_list(pf):
    """What a renderer may draw for one frame: only usable positions, in the pitch-centre system."""
    out = [dict(kind="player", side=p.source_side, jersey=p.jersey_number, x=p.selected.x, y=p.selected.y, estimated="PLAYER_ESTIMATED" in p.caveats)
           for p in pf.players if p.position_usable]
    if pf.ball.position_usable:
        out.append(dict(kind="ball", x=pf.ball.selected.x, y=pf.ball.selected.y, z=pf.ball.selected.z))
    return out                                            # missing and ambiguous observations are simply not drawn

last_time, last_period = None, None
for pf in md.frames(max_frames=200):
    if last_period is not None and pf.canonical_period != last_period:
        pass                                              # a period change: there are no frames for the break between periods
    wait = None if last_time is None else pf.video_time_s - last_time     # real elapsed video time since the previous frame
    last_time, last_period = pf.video_time_s, pf.canonical_period
    drawn = draw_list(pf)
print(len(drawn), "items drawn in the last frame; last gap in seconds:", wait)
```

Rules for a board: never draw `MISSING` or `AMBIGUOUS` coordinates; mark estimated players visibly if you show them; show events only from `pf.events` (matched) and offer `md.unmatched_events()` separately; never label an aligned event as verified. Seeking by time currently means iterating from the start (the layer streams): build your own index if you need random access.

## 10. Known limitations

* **No imputation:** missing positions stay missing; nothing is interpolated, extrapolated, smoothed, or invented, and no frame is created for a gap.
* **Event correspondence is temporal only:** physical agreement of an event with the video, and the right player, team and action, were never validated; no ground truth exists in the data.
* **Identity anomalies:** periods 3 and 4 of four extra-time matches show a `suspected_side_identity_anomaly` (audit section 7), and extra time of two matches has no tracking at all. The interface does not detect or correct this; you can withhold attribution with `quarantine_periods=[3, 4]`. Sides are never swapped.
* **Event-coordinate discrepancy** in ten matches (May/June event coordinates vs tracking) is reported, not corrected; use tracking for positions.
* **Source quality:** about 63 % of raw player positions are estimated by the tracking source, raw positions repeat about every second frame in the audited windows, and the ball is missing in about a third of frames.
* **Licensing and checksums are unverified:** do not copy dataset content into the repository, notebooks, tests, or reports that leave the team.
* **Not implemented (by design):** speed, acceleration, distance and workload metrics, fatigue indicators or prediction, tactical board UI, and any model. Tracking-derived movement is not a physiological fatigue measure.
