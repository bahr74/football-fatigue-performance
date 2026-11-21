# Session A handoff: M12 Scientific validation, efficient fair training and multi-player identity recovery (2026-11-21)

Branch `feature/m11-identity-finalization`. **Nothing committed** (the user commits).

- **Constraints respected:**
  - Session B not modified;
  - no held-out or FIFA validation/test access;
  - no GPU training;
  - no downloads or installs;
  - no historical output overwritten;
  - no dashboard redesign.
- **Test suite:** **1,207 passed, 6 skipped** (full suite, 330 s, run once at the end; 1,172 before this milestone).

## Commands (always through the launcher)

```bat
scripts\session_a_python.cmd -m football_fatigue.worktree                       REM import isolation check
scripts\m11_studio.cmd                                                          REM Studio :8620 (refuses on failed check)
scripts\session_a_python.cmd -m football_fatigue.m11_audit validity             REM audit (done; output exists, write-once)
scripts\session_a_python.cmd -m football_fatigue.m11_fair pilot-plan            REM GPU pilot package (CPU, seconds)
scripts\session_a_python.cmd -m football_fatigue.m11_fair gpu-pilot --approval <id>      REM REFUSED without approval
scripts\session_a_python.cmd -m football_fatigue.m11_fair train --family gru --dist shift_mix --search halving --approval <id>
scripts\session_a_python.cmd -m football_fatigue.m11_explorer nxm-dataset       REM exploratory N×M prototype (done)
scripts\session_a_python.cmd -m football_fatigue.m11_explorer nxm-compare       REM solver + learned scorer comparison (done)
scripts\session_a_python.cmd -m football_fatigue.m11_explorer review-plan       REM review queue proposal (done)
.venv\Scripts\python.exe -m pytest
```

## Reports (`docs/research/`)

| # | File | Topic |
|---|---|---|
| 1 | `m12_import_isolation_v1.md` | Import isolation fix |
| 2 | `m12_scientific_audit_v1.md` | Audit of the historical GRU result |
| 3 | `m12_fair_benchmark_readiness_v1.md` | Distributions, compute matching, convergence, efficiency |
| 4 | `m12_gpu_pilot_approval_package_v1.md` | GPU pilot proposal |
| 5 | `m12_multiplayer_prototype_v1.md` | Exploratory N×M dataset and quality report |
| 6 | `m12_nxm_solver_comparison_v1.md` | Deterministic N×M solver variants |
| 7 | `m12_learned_link_scorer_v1.md` | Learned link scorer |
| 8 | `m12_review_readiness_v1.md` | Review plan, v2 acceptance criteria, reviewer guide |
| 9 | `m12_session_ab_compatibility_v1.md` | A/B contract compatibility |
| 10 | `m12_remaining_blockers_v1.md` | Blockers and approvals |

## Files

**New code:**
- `src/football_fatigue/worktree.py`;
- `scripts/session_a_python.cmd`;
- `src/football_fatigue/m11_audit/` (`__init__`, `__main__`, `validity`);
- `src/football_fatigue/m11_fair/` (`distributions.py`, `halving.py`, `evaluation.py`, `fold_v2.py`, `gpu_pilot.py`);
- `src/football_fatigue/m11_explorer/` (`nxm_dataset.py`, `nxm_compare.py`, `ab_contract.py`, `review_plan.py`).

**Modified:**
- `scripts/m11_studio.cmd`;
- `pyproject.toml` (pytest `pythonpath`);
- `m11_dashboard/__main__.py` (`child_env` for the Streamlit child);
- `m11_fair/train.py`: convergence diagnostics, write-once checkpoints with sidecars, bit-identical resume, sample weights,
  non-finite loss stop. The default path is still identical to the historical trainer (tested);
- `m11_fair/__main__.py` (`pilot-plan`, `gpu-pilot`, v2 `train`);
- `m11_explorer/__main__.py` (`nxm-dataset`, `nxm-compare`, `review-plan`).

**New tests:**
- `tests/test_worktree_isolation.py` (5);
- `tests/test_m11_audit.py` (8);
- `tests/test_m11_fair_v2.py` (10);
- `tests/test_m11_nxm.py` (12).

**New outputs (`data/processed/`, all write-once):**

| Folder | Status |
|---|---|
| `m11-scientific-audit-v1` | Done |
| `m11-fair-gpu-pilot-plan-v1` | Plan only |
| `m11-nxm-exploratory-v1` | EXPLORATORY |
| `m11-nxm-solver-comparison-v1` | EXPLORATORY |
| `m11-review-priority-v1` | Proposal, 175 items, all unreviewed |

## Benchmark tables (key numbers)

**Two-player historical result, audited** (independent n = 18,777; match-level CIs):

| Model | BA overall | BA both-VISIBLE |
|---|---|---|
| GRU gap-last | 0.900 [0.895, 0.905] | 0.812 |
| HGB | 0.870 | 0.795 |
| GRU − HGB | +0.031 [0.025, 0.036] | +0.017 [−0.000, 0.031] |
| Label-free texture rule | 0.680 | 0.540 (0.719 both-ESTIMATED) |

**N×M, exploratory** (8,540 scenarios, 32 matches, cross-fitted by match):

| System | Exact [CI] | Link F1 | ID switches | Null recall |
|---|---|---|---|---|
| S0 original Kalman + Hungarian | 0.416 [0.404, 0.427] | 0.610 | 1.05 | 0.083 |
| S1 bidirectional motion | 0.483 [0.472, 0.492] | 0.676 | 0.85 | 0.086 |
| S2 bidirectional + tuned gate/nulls | 0.488 [0.475, 0.500] | 0.680 | 0.83 | 0.131 |
| L1 HGB scorer + Hungarian | 0.544 [0.534, 0.553] | 0.745 | 0.65 | 0.100 |
| L2 HGB scorer + Hungarian + null | **0.593** [0.579, 0.607] | 0.758 | 0.58 | 0.423 |

L2 HGB vs S2: +0.105 [0.094, 0.118].

## Compute

**Actual (all CPU):**

| Job | Time |
|---|---|
| Audit | about 10 s |
| Real-pool and evaluation-set loading | Seconds each |
| CPU timing benchmark | Minutes |
| Fold runner v2 real-data smoke (LSTM and HGB, tiny) | 20 s + 16 s |
| N×M dataset build | 411 s |
| N×M comparison | 850 s |
| Review plan | Under 1 min |
| Full test suite | 330 s |

- **GPU actually used:** none.
- **Estimated (not run):** GPU pilot 1.6 to 2.5 GPU-h [EST].

## What was genuinely trained

- Logistic regression and HGB link scorers (CPU), 8 cross-fitted folds, exploratory.
- Tiny smoke-test networks and models (CPU) for infrastructure checks only. Their numbers are not results.
- **No GRU or LSTM was trained for any reported result**; all two-player numbers are re-read historical predictions.

## Exploratory datasets

- `m11-nxm-exploratory-v1`: provider-derived truth, not human-reviewed, not frozen, not dataset v2, not video.
- Every N×M number in this handoff is exploratory.

## Final answers

1. **Is the historical GRU trustworthy?**
   - As a measurement, yes: no leak in splits, labels, early stopping, selection, duplicates or player overlap; seed-stable
     0.899 to 0.907.
   - As evidence of identity recovery, no. Its lead over HGB comes mostly from reading provider ESTIMATED interpolation texture:
     a label-free texture rule gets 0.68 to 0.72; in both-VISIBLE the GRU is 0.81 and its lead is not significant; with texture
     neutralised it is at or below HGB.
2. **What prevents a final fair comparison?**
   - No GPU run under the v2 protocol (convergence and cost unknown).
   - No approved compute budget.
   - No texture-neutral headline training (`robust_mix` / `visible_only`).
   - Shared GPU scheduling with Session B.
   - The infrastructure itself is ready and tested.
3. **What is the smallest useful GPU pilot?**
   - 7 configurations: GRU h48, LSTM h40, LSTM h42 (parameter-matched), GRU h96 l2 and LSTM h96 l2 on `centred`, plus GRU h48
     and LSTM h42 on `shift_mix`.
   - Outer fold 0, seed 0, cap 400, patience 25.
   - Cost about 1.6 to 2.5 GPU-h [EST], < 1.5 GB.
   - Resumable, with stop conditions.
   - It yields measured s/epoch, best epochs and memory, which size the full budget.
4. **Can the system do real N×M?**
   - Partially, and only exploratorily. On provider-derived real groups the best system is exact in 59 % of scenarios:
     73 % for N = 2 and 52 to 55 % for N = 3 to 5.
   - Null + distractor cases are solved in only 37 %.
   - Physics alone is 42 to 49 %.
   - Nothing is validated on video or on human-reviewed truth.
5. **Does learned link scoring beat physics?**
   - Yes, on this exploratory data: +0.105 exact over the best tuned physics solver (CI clear of zero), fewer identity switches,
     and a much better confidence ranking (0.91 link precision at 60 % coverage vs 0.71).
   - Texture features add only about 0.007, so it is not mainly a texture reader.
   - Still provider-domain and exploratory.
6. **What evidence is needed for dataset v2?**
   - Acceptance criteria V1 to V8 (report 8), in particular:
     - a human-estimated provider-label error with upper 95 % bound ≤ 5 % (about 120 reviewed items);
     - ≥ 80 % of severe contradictions explained;
     - two-reviewer κ ≥ 0.6;
     - shortcut baselines (texture, timing) reported, with visibility-stratified headlines;
     - match balance and per-N counts;
     - provenance labels.
   - None is met yet: 0 human decisions exist.
7. **What next?**
   - (a) Approve and run the GPU pilot (A1).
   - (b) Start human review of the 175-item queue, with a second reviewer (A3).
   - (c) N×M v1.1: fix the `repeated_interaction` flag, add N > M coverage, null-structured training, and a calibration-partition
     protocol for abstention (A5).
   - (d) Then a compute-matched fair benchmark with `robust_mix` and both-VISIBLE headlines (A2).
   - Suggested next milestone: **M13: GPU pilot + first human review round + N×M v1.1**.
