# Shared project rules

These rules apply to developers and coding agents. Follow the current authorized task scope; clarify conflicts or changes in scope before taking dependent actions.

## Scientific and data integrity

- Preserve original dataset directories, filenames, and file contents. Never rename, edit, or overwrite sources to suit a parser.
- Record source location, acquisition date, licensing/access terms, file inventory, and checksums when data is obtained. Access to a shared folder does not establish redistribution rights.
- Discover matching files by a supplied match ID after validating actual filename conventions and metadata. Handle absent or ambiguous matches explicitly. Do not hardcode `3812` into processing logic.
- Validate actual schemas, entity identifiers, periods, timestamps, sampling frequency, units, coordinate origin/orientation, and missingness before transformations or metrics.
- Do not invent fields, substitute assumed frequencies, silently resample, or infer undocumented clock alignment. Record uncertainty and unresolved assumptions.
- Keep missing values distinct from zero. Record exclusions, interpolation, transformations, and validation failures.
- Separate original global references from match-specific records; validate their relationships instead of assuming a folder layout.
- Tracking and tactical metrics are not validated physiological fatigue labels. Do not present exploratory genetics or biomarkers as deterministic fatigue or injury predictors.
- Distinguish measured results, hypotheses, and recommendations. Report exactly which checks ran and their limits.

## Engineering

- Separate ingestion, validation, preprocessing, synchronization, analytics, and playback responsibilities. Add modules when they implement approved behavior, not as empty future stubs.
- Accept configurable data roots and match IDs. Do not embed machine-specific absolute paths in reusable Python code.
- Write derived data separately, grouped by match and processing version; record source hashes, parameters, software versions, and transformation provenance.
- Make processing reproducible and deterministic where applicable; explicitly control random seeds if randomness is introduced.
- Avoid silent overwrite of prior artifacts. Keep global references and cross-match state from leaking into unrelated matches.
- Start with minimal dependencies. Add a dependency only for a demonstrated requirement, with compatible versions documented.
- Test meaningful behavior and failure cases. Small fixtures must be verified, legally shareable, and free of secrets. Synthetic fixtures must be clearly identified and must not masquerade as real source schemas.
- Current Milestone 0 includes documentation, packaging, an import test, CI, environment setup, and local Git only. No dataset download, parser, data processing, frontend, backend, analytics, or ML implementation is authorized.

## Repository and collaboration

- Keep raw, processed, and external datasets, environments, credentials, caches, and generated outputs out of Git. Review staged changes; ignore rules alone are not a security or size guarantee.
- Clear notebook outputs before committing and inspect notebooks for embedded data and secrets. Keep only approved small reference documents in Git.
- Preserve existing files and directories. Do not perform destructive cleanup, reset history, or overwrite unrelated work.
- Use short-lived branches, focused commits, and PRs reviewed by another developer. Run appropriate checks and state any untested behavior.
- Do not create or modify remote repositories, add remotes, push, or change GitHub settings without explicit authorization. Keep authentication material outside project files.
- For later prediction work, prevent subject/match leakage and document validation splits and scientific limitations before claiming predictive performance.
