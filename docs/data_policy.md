# Data handling policy

Follow [PROJECT_RULES.md](../PROJECT_RULES.md). This document explains the local storage and provenance workflow.

## Source and access

The planned FIFA World Cup 2022 dataset is supplied through [this Drive folder](https://drive.google.com/drive/folders/1_a_q1e9CXeEPJ3GdCv_3-rNO3gPqacfa). No files have been downloaded at Milestone 0. Contents, formats, licensing, and redistribution permissions remain unverified. Do not treat private GitHub visibility as permission to redistribute data.

## Local storage

- `data/raw/`: place the original dataset here later, preserving its full directory hierarchy, filenames, and bytes. Do not split or rename original files for match discovery.
- `data/processed/`: derived outputs only; separate runs by match ID and processing version once the output contract is defined.
- `data/external/`: other external data, with its own provenance and access terms.
- `models/`, `reports/`, `outputs/`, `artifacts/`, and `logs/`: local generated material, excluded from Git.

All `data/` contents are ignored. A configurable data root will allow storage elsewhere without changing reusable code. The initial environment is `F:\Football-Fatigue-Project.venv`, outside the Git working tree; environments inside the tree are ignored too.

## Inventory and provenance after data arrives

Before implementing parsing, record the source, acquisition date, access terms, original relative paths, sizes, and cryptographic checksums. Keep any inventory containing restricted source details locally. Commit only approved small documentation summaries without private links or credentials.

Validate match ID discovery against actual files and metadata. Treat global references separately, record missing/ambiguous matches, and do not assume every match has every modality. Before processing, establish schema, units, frequency, coordinate and clock conventions, period boundaries, and missing-data behavior.

Derived runs should record source checksums, match ID, processing version, parameters, environment versions, transformations, validation failures, exclusions, and output paths. Do not silently replace sources or previous outputs.

## Git review

Before each commit, inspect `git status`, `git diff --cached`, and staged file sizes. Use `git check-ignore -v <path>` to diagnose exclusions. Ignore rules do not remove already tracked files. Clear notebook outputs and inspect embedded data, metadata, and secrets before staging notebooks. Do not add dataset copies to `references/` or test directories.

Do not introduce Git LFS or a data-versioning service at this milestone. Decide later if operational requirements justify them. Dataset backups and access management remain separate from source-code version control.
