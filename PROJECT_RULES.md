# Shared project rules

This is the authoritative rules document for developers and coding agents (Claude Code, Codex). `CLAUDE.md` and `AGENTS.md` point here and must not weaken or duplicate it. Follow the currently authorized milestone; clarify conflicts or scope changes before taking dependent actions.

## Milestones

- **Milestone 0 (complete):** repository foundation: documentation, installable package, import test, CI.
- **Milestone 1 (in progress, requires explicit authorization for each part):**
  - **1A, dataset discovery and metadata-only inventory:** implemented (`football_fatigue.inventory`: file names and sizes only), pending final independent review. It is not a schema, content, or checksum validation.
  - **1B, schema audit:** implemented as documentation plus read-only audit verification scripts and their tests (`scripts/audit/`, `tests/test_audit_lib.py`; no production ingestion, cleaning, or synchronization) ([dataset_schema_audit.md](docs/dataset_schema_audit.md), [event_version_comparison.md](docs/event_version_comparison.md), [data_contracts.md](docs/data_contracts.md)), pending independent review; not formally complete. The Event Data version policy and the data contracts are provisional proposals. Checksums and provenance manifest, confirmation of the authoritative Event Data version, licensing verification, and the open data questions listed in those documents remain, so Milestone 1 as a whole is not complete.
  - **2A, production streaming ingestion and validation:** implemented (`football_fatigue.ingestion`: read-only typed loaders, canonical models, quality reports), pending independent review; not formally complete. It implements the provisional contracts of [data_contracts.md](docs/data_contracts.md) and does not do event-to-tracking synchronization, smoothing, metrics, or playback.
  - **2B, event-to-tracking temporal synchronization:** implemented (`football_fatigue.synchronization`), pending independent review; not formally complete. It aligns by video time under the 1B policy, reports temporal coverage, coordinate agreement, and (never assessed) validated correspondence separately, and performs no interpolation, smoothing, metrics, or playback.
- Roadmap steps (see [README.md](README.md)) are done one at a time. Do not begin a later step, or add code, modules, or dependencies for it, without authorization. Documentation-only tasks must not change code, tests, dependencies, data, or Git state.

## Scientific and data integrity

- **Raw immutability.** `data/raw/` is read-only in practice. Never rename, edit, move, re-encode, decompress in place, or overwrite source files or their folder names (including timestamp suffixes) to suit a parser.
- **Provenance.** Record source, acquisition date, licensing/access terms, file inventory, and checksums. Access to a shared folder does not establish redistribution rights.
- **No fabrication.** Do not invent schemas, fields, units, sampling rates, coordinate conventions, or measurements. A statement about the data is "verified" only if the relevant file was actually inspected; otherwise label it unverified or observed-only.
- **Validate before transforming.** Check structure, identifiers, periods, timestamps, sampling frequency, units, coordinate origin/orientation, and missingness before any transformation or metric.
- **Match IDs.** Discover files by a supplied match ID after validating actual naming and metadata. Never hardcode `3812`. Handle absent, duplicate, or ambiguous files (including multiple versions of a file) explicitly; do not pick one silently.
- **Missing data and uncertainty.** Keep missing distinct from zero. Do not silently resample, interpolate, drop, or assume clock alignment. Record every exclusion, interpolation, transformation, and validation failure, and carry uncertainty into outputs.
- **Global references** (competitions, players, change logs, specifications) stay separate from match-specific records; validate their relationships rather than assuming them.
- **Tracking is not fatigue measurement.** Positional tracking and derived movement metrics (distance, speed, acceleration, workload indicators) are not validated physiological fatigue labels. Do not present them, or exploratory biomarkers/genetics, as deterministic fatigue or injury predictors. Distinguish measured results, hypotheses, and recommendations, and state limits.

## Engineering

- **Modular boundaries:** ingestion, validation, preprocessing, synchronization, analytics, and playback are separate responsibilities (see [docs/architecture.md](docs/architecture.md)). Add modules only for approved, implemented behavior; no empty stubs.
- Accept configurable data roots and match IDs; no machine-specific absolute paths in reusable code.
- Write derived data separately from raw, grouped by match and processing version, with source hashes, parameters, software versions, and transformation provenance. Never silently overwrite prior outputs. Keep cross-match state from leaking between matches.
- Make processing deterministic and reproducible; control random seeds if randomness is introduced.
- Keep dependencies minimal; add one only for a demonstrated need, with a compatible version range.
- Test meaningful behavior and failure cases. Fixtures must be small, legally shareable, secret-free, and synthetic ones clearly labelled so they do not masquerade as real source schemas.
- Follow existing conventions, keep changes small and focused, and report only checks that actually ran. Never claim tests passed unless they were executed.

## Security and Git hygiene

- Keep datasets, environments, credentials, caches, and generated outputs out of Git. Ignore rules are not a guarantee: review staged changes and sizes.
- Never commit or print secrets or tokens. Keep authentication material outside project files.
- No raw dataset (or copy or sample of it) in the repository, tests, `references/`, notebooks, or PRs unless redistribution rights are verified and approved.
- Clear notebook outputs and inspect for embedded data before committing.
- Do not delete, reset, or overwrite unrelated work; no destructive cleanup without approval.

## Manual Git operations

All Git operations that modify repository state (add, commit, push, pull, merge, rebase, branch creation or switching, tag, stash, reset, remote changes) are performed manually by the user. Agents may run read-only commands (`status`, `diff`, `log`, `branch --show-current`, `check-ignore`) and may prepare suggested commit messages and PR text, but must not execute state-changing Git or GitHub commands, or change GitHub settings, unless the user explicitly asks in that instance. Workflow: short-lived feature branches from `main`, focused commits, PRs reviewed by another developer, squash merge, no force-push to `main`.

## Later research

For prediction work, prevent subject/match leakage, document validation splits, and state scientific limitations before claiming performance. Video, wearable/physiological, injury-risk, and genetics work each need their own validated ingestion, time mapping, and authorization.
