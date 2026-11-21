# Session A, Phase 2: overnight research sprint handoff (2026-11-20)

**Branch:** `feature/m11-identity-finalization` (HEAD `1bd016d`). **Nothing committed:** the user performs all Git operations.

**Predecessor:** `SESSION_A_PHASE1_INTERACTION_CATALOGUE_2026-11-19.md`

## 0. Safeguards observed

- **Data:** only the 32 `train_cv` matches.
  - Every command validates match membership before loading (`m11_phase2/common.checked` → the Phase 1 guard).
  - Stage 3 inputs go through the Stage 3 adapter, which refuses held-out matches.
  - No held-out, official FIFA validation/test or Session B data was opened.
- **Nothing trained or run:**
  - no model trained, fine-tuned or run;
  - no GPU used;
  - no checkpoint opened or changed;
  - saved OOF predictions were read only.
- **Earlier outputs untouched:**
  - no earlier output modified (verified: no file in the catalogue, Stage 3, Stage 4 or Stage 4C folders is newer than the Phase 1
    handoff);
  - no Session B file changed;
  - no download.
- **New outputs:** only new versioned folders (write-once; refuse existing) and new docs, code and tests.
- **No human review was performed.** The review manifest's reviewer fields are empty.

## 1. Deliverables

| Task | Code | Outputs (`data/processed/…`) | Report |
|---|---|---|---|
| A1–A3 quality audit | `m11_phase2/audit.py` | `m11-phase2-quality-audit-v1/` (112 KB) | `docs/research/m11_phase2_quality_audit_v1.md` |
| A4 review sample | `m11_phase2/review.py` | `m11-phase2-review-sample-v1/` (100 events, 100 PNG, 5.7 MB) | same, §6 |
| B ambiguity benchmark | `m11_phase2/baselines.py`, `ambiguity.py` | `m11-phase2-ambiguity-benchmark-v1/` (3.8 MB) | `docs/research/m11_phase2_ambiguity_benchmark_v1.md` |
| C dataset v2 design | — | — | `docs/research/m11_dataset_v2_design.md` |
| D model readiness | `m11_phase2/readiness.py` | `m11-phase2-model-readiness-v1/` | `docs/research/m11_four_model_benchmark_readiness.md` |
| E Session A/B interface | — | — | `docs/research/m11_session_ab_interface_v1.md` + `m11_session_ab_interface_v1_examples/*.json` |
| F paper map | — | — | `docs/research/m11_paper_experiment_map_v1.md` |
| Shared | `m11_phase2/common.py`, `__main__.py`, `__init__.py` | — | — |
| Tests | `tests/test_m11_phase2.py` (13 tests) | — | — |

**Each output folder has a `manifest.json`** with:
- input hashes: catalogue run manifest `a19f0f81…`, Stage 3 manifest `6016fa6c…`, split `66b2321a…`;
- parameters;
- code sha256;
- output sha256.

**Commands:**
```powershell
.\.venv\Scripts\python.exe -m football_fatigue.m11_phase2 audit
.\.venv\Scripts\python.exe -m football_fatigue.m11_phase2 review --figures
.\.venv\Scripts\python.exe -m football_fatigue.m11_phase2 ambiguity
.\.venv\Scripts\python.exe -m football_fatigue.m11_phase2 readiness
.\.venv\Scripts\python.exe -m pytest tests/test_m11_phase2.py tests/test_interaction_catalogue.py
```

Re-running a command refuses to overwrite its folder. Use a new version name.

## 2. Tests

- **Focused:** `tests/test_m11_phase2.py` + `tests/test_interaction_catalogue.py`: **27 passed** (2026-11-20).
- **Full suite:** 1139 passed, 6 skipped (§8).

## 3. Key results

| | Result | Label |
|---|---|---|
| **A** | Phase 1 counts re-verified exactly | [V] |
| | Counting invariants pass at every radius | [V] |
| | **Transitive inflation:** 60 % of connected 3+ groups at 0.5 m never have 3 players simultaneously | [V] |
| | A 5 cm bias flips 26 % of 0.5 m membership | [V] |
| | ESTIMATED contacts are kinematically smoother; VISIBLE carries about 2.5× the jump rate | [V] |
| | 302 clone-like pair-periods (5,139 edges) | [V] |
| | Merge gap 1 s is on a plateau; synchrony tolerance has none | [V] |
| **B** | Best deterministic baseline (forward Kalman, label-free noise σ_a 8 m/s², σ_m 0.05 m): **0.667 BA** [0.656, 0.676] | [V] |
| | Physics-hard (all five core methods wrong): **10.1 %** of independent samples | [V] |
| | Best-method-wrong: 33.3 % | [V] |
| | Hardest: fast crossings, mixed status, longer gaps | [V] |
| | Centring at t\* defeats bidirectional methods | [V] |
| | Learned historical models get 87–91 % on physics-hard samples | [V analysis of DOC predictions] |
| | 25 severe contradictions are unflagged suspected provider swaps | [HYP] |
| **C** | Recommended (not frozen): stratified mixed sampling with population weights; schemas A (pairwise + null + distractors) and B (N × M) specified; leakage table | [REC] |
| **D** | Historical LSTM comparison not fair (budget-bound, untuned, smaller hidden size, no extension); tabular grids boundary-bound | [V] |
| | Proposed partition validated (8 / 8 rules) | [V] |
| | About 36 GPU-h for the centred priority-1 networks; about 291 GPU-h with shift augmentation (budgeted tuning) | [EST] |
| **E** | Contract and examples written; blockers: calibration, noise / domain shift, asynchronous loss, no video GT | [REC] |
| **F** | Contribution ranking (feasibility × significance): 1 > 4 > 3 > 2 > 5 | [HYP] |

## 4. Not done (by instruction or out of scope)

- No training.
- No dataset v2 build or selection.
- No held-out evaluation.
- No video work.
- No faithful OC-SORT reproduction (only a direction cue).
- No N × M benchmark (no labelled population yet).
- No human review.
- No PCHIP variant.

## 5. Approvals needed before the next stage

1. **Scope:** primary paper direction (the advisor memo in the paper map, §5).
2. **Human audit** of the 100-event sample plus the 25 severe failures (labels / artefacts).
3. **Fair re-benchmark protocol** (Task D2) and the first GPU budget (about 36 GPU-h centred).
4. **Code changes for training:** parameterise the model kind and sizes, calibration partition, equal searches, convergence flags.
5. **Dataset v2 strategy** (stratified mixed) and gates G-v2-0…2 before any build.
6. **Status policy:** headline population and a VISIBLE-only reference analysis.
7. **Confirmation that the held-out 8 stay untouched** until a pre-registered final evaluation.

## 6. Suggested commit (user to run)

**Files:**
```
src/football_fatigue/m11_phase2/
tests/test_m11_phase2.py
docs/research/m11_phase2_quality_audit_v1.md
docs/research/m11_phase2_ambiguity_benchmark_v1.md
docs/research/m11_dataset_v2_design.md
docs/research/m11_four_model_benchmark_readiness.md
docs/research/m11_session_ab_interface_v1.md
docs/research/m11_session_ab_interface_v1_examples/
docs/research/m11_paper_experiment_map_v1.md
docs/handoffs/SESSION_A_PHASE2_OVERNIGHT_HANDOFF_2026-11-20.md
```

**Message:** `feat(m11): Phase 2 catalogue validation, deterministic ambiguity benchmark, v2/readiness/interface designs (dev only)`

Data outputs stay out of Git.

## 7. Known issues / caveats

- `review.py` figures read RAW Step 1 caches of 31 train/CV matches (about 8 min).
- The catalogue `cv_margin` uses provider identities, so it is label-derived. It is a difficulty stratum, not a trigger.
- The broader-radius difficulty counts are transfer estimates.
- The learned-model tier comparison uses selection-optimistic historical predictions.

## 8. Full-suite result

**1139 passed, 6 skipped, 2 warnings** in 281 s (2026-11-20, 07:02–07:07 +02:00). This is 1,126 earlier + 13 new.

The warnings are pre-existing sklearn Nystroem warnings in `test_m11_benchmark.py`.
