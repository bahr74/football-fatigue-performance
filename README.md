# Football Fatigue & Performance Prediction

Biomedical Engineering graduation project: a modular football analytics platform that studies player movement, workload, fatigue-related indicators, and potentially injury risk from multiple validated data sources.

Our current responsibility is **Task 2: Software Architecture & Tactical Board Streaming**. With no match video available, the first objective is to reconstruct matches from positional tracking records, synchronized with event data, and replay them on an interactive 2D tactical board.

Repository: <https://github.com/bahr74/football-fatigue-performance>

## Current status

| Area | Status |
| --- | --- |
| Repository foundation: installable `football_fatigue` package, import test, CI, documentation | **Implemented** (Milestone 0) |
| Dataset discovery and multi-match inventory (file names and sizes only) | **Implemented** (Milestone 1A, pending final independent review) |
| Schema inspection (next, Milestone 1B), ingestion, content validation, checksums, preprocessing | Planned |
| Tracking–event synchronization | Planned |
| Interactive 2D tactical board | Planned |
| Movement analytics (distance, speed, acceleration, workload indicators where scientifically supported) | Planned |
| Video detection/tracking, ball trajectories, out-of-frame estimation, wearables/physiology, fatigue prediction, injury-risk research, genetics | Future, not started, no code |

The package contains the metadata-only dataset inventory (`football_fatigue.inventory`) and nothing else. No file contents are parsed, and no schema validation, checksums, backend, frontend, analytics, or model exists.

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

By file inventory (names, extensions, sizes, and counts), the local copy covers **64 matches** with tracking, events, metadata, and rosters for each. Event Data exists in **three distinct versions** (root, `March 14, 2025`, `May 1, 2025`), each covering all 64 matches, and which version is authoritative has **not yet been determined**. A read-only readability check (all tracking `.bz2` files decompress; all event, metadata, and roster files parse as JSON) passed with no errors; this is not schema validation. **Beyond that check, no file schemas, sampling frequency, coordinate system, time conventions, or licensing terms have been verified.** See [docs/data_policy.md](docs/data_policy.md) for the observed layout and handling rules. Match `3812` is the initial development sample, not a hardcoded limitation; all processing must accept any match ID. There are no match videos.

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
| `tests/` | Tests (installed-package check and synthetic inventory tests; no dataset required) |
| `docs/` | [Architecture](docs/architecture.md) and [data policy](docs/data_policy.md) |
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
