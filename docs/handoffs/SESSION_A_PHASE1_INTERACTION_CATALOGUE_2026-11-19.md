# Session A, Phase 1: Interaction Catalogue v1, handoff (2026-11-19)

**Branch:** `feature/m11-identity-finalization` (HEAD `1bd016d`; nothing committed: the user performs all Git operations).

**Predecessors:**
- `SESSION_A_RESEARCH_AND_DASHBOARD_AUDIT_2026-11-19.md`
- `SESSION_A_DATASET_AND_MODEL_CRITICAL_REVIEW_2026-11-19.md`

## 1. Completed

| Item | Path | Status |
|---|---|---|
| Implementation | `src/football_fatigue/interaction_catalogue/` (`core.py`, `run.py`, `summary.py`, `__main__.py`, `__init__.py`) | New, untracked |
| Tests | `tests/test_interaction_catalogue.py` (14 synthetic tests) | New, untracked; 14 passed |
| Full catalogue | `data/processed/m11-interaction-catalogue-v1/` (32 match folders, `run_manifest_*.json`, `summary/`) | Built; all checks passed |
| Pilot (one match, 3816) | `data/processed/m11-interaction-catalogue-pilot-v1/` | Kept; built with `max_accel_m_s2 = 15` (the full run uses 25). Superseded |
| Methodology | `docs/research/m11_interaction_catalogue_v1_methodology.md` | New |
| Counts and findings | `docs/research/m11_interaction_catalogue_v1_findings.md` | New |

### Per-match output files

| File | Contents |
|---|---|
| `nodes.parquet` | Player chains |
| `edges.parquet` | All radii; `radius_m` column |
| `edge_chains.parquet` | — |
| `crossings.parquet` | Every crossing |
| `pair_windows.parquet` | Crossings per complete 5 s window |
| `groups.parquet` | δ = 0 and δ = 1 s |
| `group_timeline.parquet` | Simultaneous groups of ≥ 3 players |
| `window_stats.parquet` | — |
| `match_summary.json` | Segment exclusions, timestamps, coordinates, sample-quality totals, reproduction checks |

### Commands

```powershell
.\.venv\Scripts\python.exe -m football_fatigue.interaction_catalogue run --layout full --workers 4   # refuses existing match folders
.\.venv\Scripts\python.exe -m football_fatigue.interaction_catalogue summarize --layout full         # refuses an existing summary
.\.venv\Scripts\python.exe -m pytest tests/test_interaction_catalogue.py
```

**Tests run.** The full repository suite gave **1126 passed, 6 skipped** in 303 s (2026-11-19, 23:51–23:56 +02:00). The 14 new
catalogue tests are included.

## 2. Safeguards observed

- **Matches processed:** only the 32 `train_cv` matches.
  - `run.assert_allowed` refuses every other id before loading anything.
  - The 8 internal held-out and 24 official validation/test ids are refused.
  - A test checks this with a loader that fails if called.
- **Untouched:** no Session B file; no model or checkpoint; no training; no GPU.
- **Earlier outputs read only:** Stage 1–4 and M1–M3 outputs were not modified. Stage 2 candidates and window-intersections v1 were
  read only, for reproduction checks.
- **0.5 m benchmark unchanged:** it is reproduced exactly as a projection of the catalogue (edges = 45,989 episodes; edge chains =
  29,861 candidates; identical t\* and d_min).
- **Status:** VISIBLE and ESTIMATED are both eligible. Status is audit metadata only and is tested not to change eligibility,
  reliability or label risk by itself.
- **No occlusion claims:** proximity and crossings are never called occlusion. Visual occlusion, box overlap, tracker loss and
  identity switches are documented as not observable.

## 3. Incident during the run (resolved)

- **What happened.** The first full launch failed at once. On Windows, `multiprocessing` with spawn cannot pickle a worker function
  defined in `__main__.py`, so workers died and were respawned.
- **Fix.**
  - The task was stopped.
  - No match output had been written; only the empty `m11-interaction-catalogue-v1/` folder existed.
  - The runner moved to `run.py`, with `__main__.py` as a thin entry point, and the run was relaunched successfully.

## 4. Key results

All values are exact; details are in the findings document.

- **Radius sensitivity:** edges 21,882 / 45,989 / 104,842 / 161,663 / 216,551 at 0.25 / 0.5 / 1 / 1.5 / 2 m. Temporally connected
  groups 13,423 / 24,667 / 48,053 / 62,260 / 69,194.
- **Crossings:**
  - 581,786 crossings, of which only 33,887 (5.8 %) are synchronous; median separation 3.3 m at the crossing.
  - 246,532 pair-windows hold more than one crossing; 23,306 hold more than one synchronous crossing.
  - All crossings are now kept.
- **Multi-player structure at 0.5 m:** 1,199 simultaneous groups of ≥ 3 players. 10.7 % of edges are simultaneous with another
  partner and 12.7 % are sequential within 1 s.
- **Status:** both players are VISIBLE in only 7.5 % of sub-25 cm contacts, against 22 % at 2 m. At 0.5 m, ESTIMATED contacts are
  constant-velocity-confusable half as often (16 %) as VISIBLE ones (34 %).
- **Ambiguity:** 24,369 constant-velocity-confusable edges at 2 m, against 10,627 at 0.5 m.

## 5. Not done (by instruction or out of scope)

- No model retraining.
- No dataset v2 selection.
- No supervised radius choice.
- No real video work.
- No held-out evaluation.
- No Git operations.
- No PCHIP variant of the catalogue (v1 compared it previously).
- No human review of example events (no figures or dashboard view were requested in Phase 1).

## 6. Decisions for the user and the advisor

1. **Paper focus.** Should the paper centre on (a) two-player recovery with a broader catalogue, (b) a catalogue / provider-artifact
   study, or (c) N×M association? The catalogue supports all three.
2. **Supervised population.** Should it be defined by a pre-specified ambiguity criterion (for example constant-velocity confusable
   or marginal at ≤ 2 m) rather than by distance? Approve the criterion before any sample building.
3. **ESTIMATED-dominated close contacts.** How should the paper treat them: stratify only, or add a VISIBLE-only reference analysis?
4. **Independence unit.** Connected groups within matches, with the match for confidence intervals?
5. **Clone-like pairs.** Should they be reported with and without these pairs, or should a dedicated provider-artifact flag be added
   in a later dataset version?
6. **Human audit.** Is an audit of a stratified sample of edges and groups (for example 100 events) wanted before definitions are
   frozen?

## 7. Suggested commit (user to run)

Files:

```
src/football_fatigue/interaction_catalogue/
tests/test_interaction_catalogue.py
docs/research/m11_interaction_catalogue_v1_methodology.md
docs/research/m11_interaction_catalogue_v1_findings.md
docs/handoffs/SESSION_A_PHASE1_INTERACTION_CATALOGUE_2026-11-19.md
```

Message: `feat(m11): add Interaction Catalogue v1 (descriptive multi-radius interaction graph, dev matches only)`

Data outputs stay out of Git.
