# Data handling policy

Follow [PROJECT_RULES.md](../PROJECT_RULES.md). This document covers local storage, provenance, and safe handling.

## Source and licensing

The dataset is FIFA World Cup 2022 data, downloaded from [this Drive folder](https://drive.google.com/drive/folders/1_a_q1e9CXeEPJ3GdCv_3-rNO3gPqacfa). **Licensing, access terms, and redistribution rights are unverified.** Shared-folder access, or a private GitHub repository, is not permission to redistribute. Verify and record the terms (including anything in the supplied specification and change-log documents) before any sharing beyond the team. There are no match videos.

## Local raw layout (observed from file names, sizes, and hashes)

Location: `data/raw/` (ignored by Git). Observed on 2026-10-15 after a fresh download. Beyond the readability check below, file contents have not been inspected; schemas, units, frequency, coordinates, and clocks are unverified. Google Drive split each source family into numbered download parts.

| Folder (timestamp suffix as downloaded) | Files | Content |
| --- | --- | --- |
| `Tracking Data-20261015T150539Z-1-001/Tracking Data/` | 33 | 32 `<id>.jsonl.bz2` plus `PFF FC Tracking Data Specification v2.2.pdf` |
| `Tracking Data-20261015T150539Z-1-002/Tracking Data/` | 32 | 32 `<id>.jsonl.bz2` |
| `Event Data-20261015T150543Z-1-001/Event Data/` | 97 | `<id>.json` at the top level and in `March 14, 2025/` and `May 1, 2025/` |
| `Event Data-20261015T150543Z-1-002/Event Data/` | 97 | same layout, plus specification PDFs v2.5 (top level) and v2.4 (`May 1, 2025/`) |
| `Metadata-20261015T150542Z-1-001/Metadata/` | 64 | `<id>.json` |
| `Rosters-20261015T150541Z-1-001/Rosters/` | 64 | `<id>.json`, same IDs as metadata |
| root files | 3 | `competitions.csv`, `players.csv`, `PFF FC Change Log.docx` |

**Coverage (file inventory):** 64 matches (IDs 3812 to 3859 and 10502 to 10517), each present in tracking, events, metadata, and rosters. Tracking parts `-001` and `-002` are disjoint (32 matches each); metadata and rosters hold all 64 IDs in one part each. Match 3812 is present in all four families. No zero-byte or unusually small files were observed. `competitions.csv` appears to list match IDs (unverified).

**Event Data versions:** there are **three distinct versions** of the event files: top level (unlabelled), `March 14, 2025`, and `May 1, 2025`. Each version covers all 64 matches, distributed across the two download parts (a given ID and version lives in exactly one part, so both parts are needed). The three versions of an ID differ in size (averaging about 18.0 MB, 10.9 MB, and 16.6 MB), so they are different content, not copies. **Which version is authoritative has not been determined.** Do not treat versions as interchangeable; resolve this using the specification PDFs, the change log, and file contents before ingestion.

**Provenance of this copy:** an earlier download included a second tracking copy (timestamp `150751Z`). Its 64 files were verified byte-identical (SHA-256) to those in `150539Z`, and the user has manually removed it. Only `150539Z` parts `-001` and `-002` remain.

**Integrity check (2026-10-15): passed, readability only.** A read-only pass fully decompressed all 64 tracking `.bz2` files (first and last lines parsed as JSON) and parsed as JSON all 192 event files and all 64 metadata and 64 roster files, with no errors. This confirms the files are not truncated or corrupt at the compression/JSON level. It does **not** validate schemas, completeness of records, coordinates, or timing. The 64 `.bz2` files in the earlier second tracking copy were byte-identical, so they are covered by the same result.

Preserve all original files and folder names, including timestamp suffixes, subfolders, and `.bz2` compression. Do not extract, rename, or edit in place; any decompression writes to derived storage.

## Match ID mapping

Mapping is discovered, never assumed: inventory filenames, then confirm against metadata contents once inspected. Record per match which modalities exist, duplicate or multi-version candidates, and conflicts. `3812` is the development sample, not a special case. Global files (`competitions.csv`, `players.csv`) stay separate from match records, and their links to match files must be validated.

## Provenance and checksums

Before parsing, record: source URL, acquisition date, access terms, relative path, size, and a cryptographic checksum (for example SHA-256) for every file, plus tool and version. Keep this manifest locally (for example under `reports/`, ignored); commit only an approved summary without private links or credentials. Re-verify checksums when files are re-downloaded; a changed hash must be surfaced, not overwritten.

## Derived data

- `data/processed/`: derived outputs only, separated by match ID and processing version, never into `data/raw/`.
- Each run records source checksums, match ID, processing version, parameters, environment versions, transformations, validation failures, exclusions, and output paths.
- Never silently replace sources or earlier outputs.
- `data/external/`: other external data with its own provenance and terms.
- `models/`, `reports/`, `outputs/`, `artifacts/`, `logs/`: generated, ignored by Git. A configurable data root lets storage live elsewhere.

## Missing data

Do not assume every match has every modality or that every field is populated. Represent missing as missing (not zero), report coverage per match, and document each exclusion, interpolation, or imputation with its justification. Validation must report, not hide, gaps.

## Git and sharing

- No raw dataset files, samples, or copies in GitHub, tests, `references/`, notebooks, or PRs unless redistribution rights are verified and approved. All of `data/` is ignored, but ignore rules do not remove already-tracked files; check `git status`, `git diff --cached`, and staged sizes before each commit (`git check-ignore -v <path>` diagnoses exclusions). Git commands that change state are run by the user.
- Generated reports may embed source values, player identifiers, or private paths. Keep them local by default; review and redact before sharing. Do not share outputs containing source data beyond the team until licensing is verified.
- Clear notebook outputs and check for embedded data and secrets.
- No Git LFS or data-versioning service for now. Dataset backups and access control are handled separately from source control.
