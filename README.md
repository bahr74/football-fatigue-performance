# Football Fatigue & Performance Prediction

Biomedical Engineering graduation project: a modular football analytics platform that studies player movement, workload, fatigue-related indicators, and potentially injury risk from multiple validated data sources.

Our current responsibility is **Task 2: Software Architecture & Tactical Board Streaming**. With no match video available, the first objective is to reconstruct matches from positional tracking records, synchronized with event data, and replay them on an interactive 2D tactical board.

Repository: <https://github.com/bahr74/football-fatigue-performance>

## Current status

| Area | Status |
| --- | --- |
| Repository foundation: installable `football_fatigue` package, import test, CI, documentation | **Implemented** (Milestone 0) |
| Dataset discovery and multi-match inventory | Planned (next, Milestone 1) |
| Schema inspection, ingestion, validation, preprocessing | Planned |
| Tracking–event synchronization | Planned |
| Interactive 2D tactical board | Planned |
| Movement analytics (distance, speed, acceleration, workload indicators where scientifically supported) | Planned |
| Video detection/tracking, ball trajectories, out-of-frame estimation, wearables/physiology, fatigue prediction, injury-risk research, genetics | Future, not started, no code |

The package currently contains only a package marker. No dataset parsing, backend, frontend, analytics, or model exists.

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
| `src/football_fatigue/` | Python package (currently a package marker only) |
| `tests/` | Tests (currently an installed-package check) |
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
