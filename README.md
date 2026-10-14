# Football Fatigue & Performance Prediction

Biomedical Engineering graduation project. Current work is Task 2: football tracking data processing, event synchronization, and an interactive 2D tactical board.

## Current milestone

Milestone 0 provides repository documentation, an installable Python package, a package installation test, and CI. No dataset processing, tactical board, backend, or predictive model is implemented. There are no local datasets or match videos at foundation setup.

The planned dataset is FIFA World Cup 2022, supplied through [this Drive folder](https://drive.google.com/drive/folders/1_a_q1e9CXeEPJ3GdCv_3-rNO3gPqacfa). Its contents, schemas, sampling frequency, coordinates, time conventions, and licensing have not been validated. Match `3812` is the initial development sample; future processing must accept any available match ID.

## Windows setup

Use Python 3.12. From PowerShell, the environment for this workstation is a sibling of the repository on drive F:

```powershell
Set-Location F:\Football-Fatigue-Project
# Run once; explicitly select Python 3.12 rather than the launcher default.
& "C:\Users\DELL\AppData\Local\Programs\Python\Python312\python.exe" -m venv F:\Football-Fatigue-Project.venv
& "F:\Football-Fatigue-Project.venv\Scripts\python.exe" -m pip install --no-cache-dir -e ".[dev]"
& "F:\Football-Fatigue-Project.venv\Scripts\python.exe" -m pytest
& "F:\Football-Fatigue-Project.venv\Scripts\python.exe" -c "import football_fatigue; print(football_fatigue.__file__)"
```

Other developers should substitute their own Python 3.12 executable and environment path. Activation is optional; use the environment's interpreter directly. Do not commit or copy environments. Installation uses `--no-cache-dir` to avoid persistent pip downloads on C; pip may still use temporary OS storage during builds.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/football_fatigue/` | Python package; currently only a package marker |
| `tests/` | Tests; currently installed-package verification |
| `docs/` | Architecture and data handling decisions |
| `configs/` | Future shared configuration once requirements are validated |
| `data/raw/` | Original source dataset, unchanged; ignored by Git |
| `data/processed/` | Reproducible derived data; ignored by Git |
| `data/external/` | Other external data; ignored by Git |
| `models/`, `reports/` | Local generated artifacts; ignored by Git |
| `notebooks/` | Future reviewed analysis notebooks, with outputs cleared |
| `references/` | Approved small reference notes and links, not dataset copies |

Git does not retain empty directories. After a fresh clone, recreate local directories as needed:

```powershell
New-Item -ItemType Directory -Force -Path configs,data/raw,data/processed,data/external,models,notebooks,references,reports,.agents/skills,.claude/skills | Out-Null
```

## Working together

Read [PROJECT_RULES.md](PROJECT_RULES.md), [architecture](docs/architecture.md), and [data policy](docs/data_policy.md) before implementing a feature. Codex uses [AGENTS.md](AGENTS.md); Claude Code uses [CLAUDE.md](CLAUDE.md).

Three developers should use separate clones and short-lived branches from `main`, for example `feat/match-discovery`, `fix/time-validation`, or `docs/data-provenance`. Pull requests should explain behavior and validation, receive one other developer's review, pass checks, and use squash merges. Do not force-push `main`.

## Private GitHub setup (pending approval)

Local Git is established at Milestone 0. Remote creation, remote configuration, pushing, invitations, and GitHub settings require separate authorization. Once approved, create an empty private repository without generated files, authenticate with Git Credential Manager or SSH, add its URL as `origin`, and push `main`. Then invite the other developers and enable PR controls where available for the chosen account.

Video CV, physiological integration, fatigue prediction, injury-risk modeling, and genetics are future work outside this milestone.
