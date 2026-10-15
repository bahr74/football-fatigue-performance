# Milestone 1B audit verification scripts

Read-only, standard-library tooling that reproduces the quantitative findings in [docs/dataset_schema_audit.md](../../docs/dataset_schema_audit.md) and [docs/event_version_comparison.md](../../docs/event_version_comparison.md). These are **audit scripts, not production code**: no ingestion, cleaning, preprocessing, or synchronization engine. They locate files through `football_fatigue.inventory`, never write inside `data/raw/` (writing there is refused), never decompress to disk, and stream tracking files one record at a time.

Per-match results go to `data/processed/audit_1b/<script>/<match>.json` (ignored by Git; existing files are kept unless `--force`). `--aggregate` prints pooled totals with explicit denominators, kept separate from per-match ranges, and lists failed or missing matches. Run from the repository root with the project interpreter (`.\.venv\Scripts\python.exe`, Python 3.12).

| Script | Reproduces | Typical cost (this machine, 4-7 workers) |
| --- | --- | --- |
| `event_versions.py` | Row counts per version; repeated `(gameEventId, possessionEventId)` pairs; collision-safe row matching between versions; changed rows vs changed field values; position differences | about 1 minute for all 64 matches |
| `tracking_scan.py` | Records, unique frames, repeated-frame records, period coverage (incl. missing extra time), missing ball, estimated positions, `speed` presence, smoothed-list duplicates (identical vs conflicting) | about 3-4 CPU-minutes per match; about 40 minutes for 64 matches with 7 workers |
| `event_alignment.py` | Temporal coverage under the provisional alignment policy; coordinate agreement of event rows (all three versions) with raw tracking at the selected frame | about 1-4 CPU-minutes per match (tracking is streamed once) |
| `extra_time_identity.py` | Extra-time coverage; carrier-to-ball test that flags `suspected_side_identity_anomaly` (events for all matches, tracking for extra-time matches) | seconds for events; about 4 minutes per extra-time match for tracking |

```powershell
# one match (fast): replace 3812 with any match id
.\.venv\Scripts\python.exe scripts\audit\event_versions.py --matches 3812 --workers 1 --force
.\.venv\Scripts\python.exe scripts\audit\tracking_scan.py show-frame 3812 5578     # smoothed-list conflict, early exit
.\.venv\Scripts\python.exe scripts\audit\tracking_scan.py --matches 3814 --workers 1 --force

# all matches, then the pooled summary (can also be re-run alone with --aggregate)
.\.venv\Scripts\python.exe scripts\audit\event_versions.py --workers 4
.\.venv\Scripts\python.exe scripts\audit\tracking_scan.py --workers 7
.\.venv\Scripts\python.exe scripts\audit\event_alignment.py --workers 7
.\.venv\Scripts\python.exe scripts\audit\extra_time_identity.py --workers 4
```

Unit tests for the verification logic (synthetic data only): `.\.venv\Scripts\python.exe -m pytest tests\test_audit_lib.py`.

## Definitions used by the scripts

- **Frame:** a unique `frameNum`. Repeated records of one `frameNum` collapse to the first record; whether they are identical is counted.
- **Event row signature:** `(gameEventId, possessionEventId, startTime, gameEventType, possessionEventType)`. Rows with an identical signature are numbered in file order; rows match across versions on `(signature, ordinal)`. Unmatched and ambiguous rows are reported, never dropped.
- **Common-field comparison:** leaf paths present in both matched rows; null vs value is a change; paths present in only one row are schema differences. `homePlayers`, `awayPlayers`, `ball` are compared separately.
- **Temporal alignment (provisional):** event `eventTime` (seconds of video time) against tracking `videoTimeMs / 1000`; same source period only; nearest frame; one tolerance of one frame interval (`1 / fps`); exact ties choose the earlier frame; no extrapolation beyond a period's frame range; every unmatched event gets a reason.
- **Coordinate agreement (separate, weaker than validation):** at least 90 % of at least 11 compared players within 0.01 m of the raw tracking positions at the selected frame. Validated correspondence (right player, team, period, physical action) is not assessed.

## Limits

Tracking scans over all 64 files are expensive; `--matches` runs subsets. A subset is never reported as the full dataset: each `--aggregate` prints how many matches it covers. Raw-derived outputs stay under the ignored `data/processed/` and are not committed.
