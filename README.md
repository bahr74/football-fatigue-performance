# Football Fatigue & Performance Prediction

Biomedical Engineering graduation project: a modular football analytics platform that studies player movement, workload, fatigue-related indicators, and potentially injury risk from multiple validated data sources.

Our current responsibility is **Task 2: Software Architecture & Tactical Board Streaming**. With no match video available, the first objective is to reconstruct matches from positional tracking records, synchronized with event data, and replay them on an interactive 2D tactical board.

Repository: <https://github.com/bahr74/football-fatigue-performance>

## Current status

| Area | Status |
| --- | --- |
| Repository foundation: installable `football_fatigue` package, import test, CI, documentation | **Implemented** (Milestone 0) |
| Dataset discovery and multi-match inventory (file names and sizes only) | **Implemented** (Milestone 1A, pending final independent review) |
| Schema audit and proposed data contracts (documentation plus read-only reproducible audit scripts in [scripts/audit](scripts/audit/README.md): [dataset schema audit](docs/dataset_schema_audit.md), [event version comparison](docs/event_version_comparison.md), [data contracts](docs/data_contracts.md)) | **Implemented** (Milestone 1B, pending independent review; not formally complete; version policy and contracts are provisional) |
| Production streaming ingestion and quality reporting (`football_fatigue.ingestion`) | **Implemented** (Milestone 2A, pending independent review) |
| Event-to-tracking temporal synchronization (`football_fatigue.synchronization`) | **Implemented** (Milestone 2B, pending independent review) |
| Checksums/provenance manifest, preprocessing | Planned |
| Tracking–event synchronization | Planned |
| Interactive 2D tactical board | Planned |
| Movement analytics (distance, speed, acceleration, workload indicators where scientifically supported) | Planned |
| Video detection/tracking, ball trajectories, out-of-frame estimation, wearables/physiology, fatigue prediction, injury-risk research, genetics | Future, not started, no code |

The package contains the metadata-only dataset inventory (`football_fatigue.inventory`) and nothing else. The package parses no file contents, and no schema validation, checksums, backend, frontend, analytics, or model exists. The Milestone 1B audit's key quantitative findings are reproducible with the read-only scripts in `scripts/audit/`; they are audit tooling, not an ingestion or synchronization implementation.

## Event-to-tracking synchronization (Milestone 2B)

`football_fatigue.synchronization` aligns each event row of one **explicitly named** event version to the match's tracking frames by video time. It applies the alignment policy validated in Milestone 1B and never interpolates, extrapolates, or invents frames or positions.

```python
from football_fatigue.ingestion import Dataset, EventVersion
from football_fatigue.synchronization import synchronize_match

sync = synchronize_match(Dataset("data/raw"), "3812", EventVersion.ROOT, spatial=False)   # one streaming tracking pass
for a in sync.alignments:                      # one AlignmentResult per event row, source order, none dropped
    if a.status.value == "matched":
        a.frame_num, a.residual_s, a.frame_source_record_indices
    else:
        a.reason                               # UnmatchedReason
print(sync.summary())                          # counts, reasons, residuals, tracking period ranges, scan_complete
```

- **Policy (provisional):** event `event_time_s` vs frame `video_time_s`; only frames of the **same canonical period**; the nearest frame wins, exact ties go to the **earlier** frame; the single maximum residual is one frame interval (`1 / fps` from the metadata, or an explicit `tolerance_s`); no extrapolation.
- **Result model:** `AlignmentResult` carries the event provenance (row id, version, source row index, source reference, source and canonical period, event quality flags), the temporal result (`MatchStatus`, `UnmatchedReason`, signed `residual_s`, `tolerance_s`), the selected frame's provenance (frame number, video time, period, source reference, **all** source record indices, frame quality flags), `temporal_notes` (`NEAR_PERIOD_BOUNDARY`), a separate `SpatialAssessment`, and `validated_correspondence = "not_assessed"`. `quality_label` is `unmatched`, `temporal_only`, `coordinate_agreeing`, or `temporal_only_coordinates_disagree`; it never says "validated".
- **Unmatched reasons:** `missing_event_time`, `invalid_source_period` (the source period is kept next to a `None` canonical period), `no_tracking_for_period` (for example event-only extra time), `outside_period_frame_range`, `no_frame_within_tolerance`, `period_label_differs_from_nearest_frame`, and `tracking_scan_incomplete` (the tracking stream failed: no absence is claimed and the error is kept in `fatal_error`). Frames that go backwards in time set `ordering_violation` and make every tracking-dependent result `tracking_scan_incomplete`. Events and frames of another match, or events of another version than requested, are rejected with a `SynchronizationError` (an `IngestionError`).
- **Three separate claims:** temporal coverage (`MATCHED`), coordinate agreement (`spatial=True`: the event's embedded coordinates against the frame's *raw* positions, tolerance 0.01 m, at least 90 % of at least 11 players; quarantined players, frames whose repeated records conflict, malformed values, and (side, jersey) groups with conflicting duplicate coordinates on either side are excluded, independent of entry order; nothing is corrected, and the known ten-match event-coordinate discrepancies appear as `coordinate_disagreeing`), and validated physical correspondence (not assessed; it needs independent ground truth such as video).
- **Performance:** tracking is consumed once in file order with O(1) frame memory; events are buffered (O(E), plus O(E x players) with `spatial=True`) and each frame touches only the events within one tolerance, so time is O(F + E log E). A whole match took about 75-80 s in the smoke tests.
- **Limitations:** the one-frame tolerance is a temporal-coverage rule, not an accuracy claim (event times coincide with the frame grid to about 0.25 ms in the audit, which shows grid agreement, not physical timing); no identity swap or correction is applied for the suspected extra-time side anomaly; a tracking error leaves the unsettled events as `tracking_scan_incomplete` rather than guessing.

## Ingestion (Milestone 2A)

`football_fatigue.ingestion` provides typed, read-only loaders and per-match quality reports. It implements the provisional contracts of [docs/data_contracts.md](docs/data_contracts.md); it does not synchronize events with tracking, smooth or interpolate, compute metrics, or render anything.

```python
from football_fatigue.ingestion import Dataset, EventVersion, quality_report, write_quality_report

ds = Dataset("data/raw")                                    # files are located through the project inventory
meta, roster = ds.metadata("3812"), ds.roster("3812")        # validated; fatal problems raise IngestionError
events = ds.load_events("3812", EventVersion.ROOT)           # a version must be named; no fallback; source order and duplicates kept
for frame in ds.iter_tracking_frames("3812", roster=roster): # streaming, bounded memory; repeated frameNum records collapse
    ball = frame.ball                                        # None = missing observation (never 0)
    for p in frame.players:                                  # raw x/y, all smoothed candidates, flagged status, side+shirt identity
        ...
report = quality_report(ds, "3812", EventVersion.ROOT)       # one tracking pass; use max_frames=N for a bounded smoke test
write_quality_report(report, "data/raw")                     # only below data/processed/ingestion_quality/<match>/<version>.json
```

```powershell
.\.venv\Scripts\python.exe -m football_fatigue.ingestion quality 3812 --events root [--max-frames 3000] [--write [--overwrite]]
```

Conflicting smoothed candidates are kept and the canonical smoothed coordinate is left `None` (never the first candidate); an invalid source period is kept next to a `None` canonical period; identity uses the roster join on (side, shirt number), never the jersey number alone, and periods named in `quarantine_periods` are withheld instead of swapped. Exit codes of the CLI: 0 report produced, 1 fatal structural error in a source, 2 invalid arguments or an unsafe output location.

## Dataset inventory

`football_fatigue.inventory` scans a raw-data root and reports which files exist for each match. It reads only directory listings and file sizes; it never opens, decompresses, hashes, or modifies a file, and it does not choose an authoritative Event Data version. Files are classified from their directory context (download folder name, optional bare inner folder of the same name, and for events the version subfolder) and a strict file name, not by extension alone. Conflicting family folders, nested download wrappers, and other unsupported nesting are reported as unrecognised files and never counted toward match coverage.

```powershell
.\.venv\Scripts\python.exe -m football_fatigue.inventory                       # scan data/raw (default)
.\.venv\Scripts\python.exe -m football_fatigue.inventory D:\path\to\raw        # other root
.\.venv\Scripts\python.exe -m football_fatigue.inventory --match 3812          # one match
.\.venv\Scripts\python.exe -m football_fatigue.inventory --format json         # machine-readable
```

- **Text report:** match count, per-modality coverage, event versions with per-version coverage, specifications, global reference files, then issues grouped as missing files, duplicate or ambiguous candidates, unrecognised files, and skipped entries. `--match` lists that match's files per modality and event version.
- **JSON report:** the same summary plus a `files` list (relative path, kind, size in bytes, match ID, event version, classification note) and per-match files. The Event Data root version is `{"root": true, "name": null}`, which is distinct from a real folder named `(root)`. `--match` JSON includes `scan_wide_issue_count` for issues not tied to a match.
- **Exit codes:** `0` scan completed (issues, if any, are in the report); `2` the root is not a directory; `3` the scan failed (unreadable directory or file metadata, or the root is a symlink/junction), in which case nothing is printed to stdout and an error list goes to stderr.
- **Links:** symbolic links and Windows junctions are never followed or counted as dataset files; they are reported as skipped entries. A scan root that is itself a link is rejected.

## Dataset

The planned dataset is FIFA World Cup 2022, obtained from [this Drive folder](https://drive.google.com/drive/folders/1_a_q1e9CXeEPJ3GdCv_3-rNO3gPqacfa) and stored locally, unchanged, at `data/raw/` (ignored by Git). It is expected to contain tracking, events, rosters, metadata, and global reference files. Downloaded top-level folders carry timestamp suffixes; keep them exactly as downloaded.

By file inventory (names, extensions, sizes, and counts), the local copy covers **64 matches** with tracking, events, metadata, and rosters for each. Event Data exists in **three distinct versions** (root, `March 14, 2025`, `May 1, 2025`), each covering all 64 matches, and which version is authoritative has **not yet been determined**. A read-only readability check (all tracking `.bz2` files decompress; all event, metadata, and roster files parse as JSON) passed with no errors. The Milestone 1B audit then inspected the actual files and the PFF specifications: schemas, frame rate (29.97 fps), coordinate system, time bases, identifier relationships, event version differences, and several data problems are documented in [docs/dataset_schema_audit.md](docs/dataset_schema_audit.md) and [docs/event_version_comparison.md](docs/event_version_comparison.md), with a provisional (not final) Event Data version policy. **Licensing terms and checksums remain unverified.** See [docs/data_policy.md](docs/data_policy.md) for the observed layout and handling rules. Match `3812` is the initial development sample, not a hardcoded limitation; all processing must accept any match ID. There are no match videos.

## Setup (Windows, Python 3.12)

The virtual environment lives at `.venv` inside the repository (ignored by Git). To create it once, then install and test:

```powershell
Set-Location F:\Football-Fatigue-Project
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
```

Activation is optional; invoking the environment's interpreter directly works. Use your own Python 3.12 if `py` is unavailable. Do not commit or copy environments.

## Repository structure

| Path | Purpose |
| --- | --- |
| `src/football_fatigue/` | Python package (currently the metadata-only dataset inventory) |
| `tests/` | Tests (installed-package check, synthetic inventory tests, and synthetic tests of the audit verification logic; no dataset required) |
| `src/football_fatigue/ingestion/` | Milestone 2A loaders, canonical models, and quality reports (see below) |
| `src/football_fatigue/synchronization.py` | Milestone 2B event-to-tracking temporal alignment (see above) |
| `scripts/audit/` | Read-only Milestone 1B audit verification scripts (not production code); see its README for reproduction commands |
| `docs/` | [Architecture](docs/architecture.md), [data policy](docs/data_policy.md), and the Milestone 1B [schema audit](docs/dataset_schema_audit.md), [event version comparison](docs/event_version_comparison.md), and proposed [data contracts](docs/data_contracts.md) |
| `data/raw/` | Original dataset, immutable, ignored by Git |
| `data/processed/` | Reproducible derived data, ignored by Git |
| `data/external/` | Other external data, ignored by Git |
| `PROJECT_RULES.md` | Authoritative shared rules for developers and agents |
| `CLAUDE.md`, `AGENTS.md` | Short persistent instructions for Claude Code and Codex |
| `.github/` | CI workflow and pull request template |

Directories such as `configs/`, `models/`, `reports/`, `notebooks/`, and `references/` are created only when needed; `models/` and `reports/` are ignored by Git. Git does not keep empty directories.

## Development workflow

- Claude Code is the primary implementation assistant; Codex is an independent reviewer (and may debug or implement targeted changes on request).
- Work on short-lived feature branches from `main` (for example `feat/dataset-inventory`), one milestone at a time.
- **All Git operations that modify repository state (add, commit, push, pull, merge, branch creation or switching) are performed manually by the user.** Agents may use read-only Git commands.
- Pull requests explain behavior and validation, are reviewed by another developer, pass checks, and are squash-merged. Do not force-push `main`.
- Read [PROJECT_RULES.md](PROJECT_RULES.md) before implementing anything.

## Roadmap

1. Dataset discovery and multi-match inventory.
2. Schema inspection, focused on match 3812 and cross-checked against other matches.
3. Tracking, event, roster, and metadata ingestion.
4. Data validation and quality assessment.
5. Reproducible preprocessing and optimized storage.
6. Tracking–event synchronization.
7. Interactive 2D tactical board: player/ball movement, play/pause, seeking, playback speed, event navigation.
8. Player movement analytics.

Each step requires explicit authorization before it starts. Later research modules (listed as Future above) will be designed only after the tracking and event foundation is validated.
