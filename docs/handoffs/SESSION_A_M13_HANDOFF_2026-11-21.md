# Session A handoff: M13 multi-player dataset corrections, reliable association and validation readiness (2026-11-21)

Branch `feature/m11-identity-finalization`. **Nothing committed** (the user commits).

**Constraints respected.**
- Session A only; Session B not modified.
- Only the 32 development matches; no held-out or FIFA test data.
- **No GPU used.**
- No downloads or installs.
- No historical output overwritten (15 M12 output files re-hashed: byte-identical).
- No dashboard changes.
- No occlusion claims.

Every N×M number below is **EXPLORATORY**: provider-derived truth, not human-reviewed, not frozen, not video.

## Commands (always through the launcher)

```bat
scripts\session_a_python.cmd -m football_fatigue.worktree                         REM import isolation check (OK)
scripts\session_a_python.cmd -m football_fatigue.m11_explorer nxm-dataset-v2      REM corrected N×M dataset (done, 937 s)
scripts\session_a_python.cmd -m football_fatigue.m11_explorer nxm-assoc-v2        REM benchmark + calibration on v2 and v1 (done, 2,468 s)
scripts\session_a_python.cmd -m football_fatigue.m11_explorer review-export       REM review queue v2 + export (done, < 1 min)
scripts\session_a_python.cmd -m football_fatigue.m11_fair gpu-pilot --approval <id> --device cuda   REM REFUSED without approval (checked: exit 2)
.venv\Scripts\python.exe -m pytest
```

All outputs are write-once: a second run refuses an existing folder.

## Reports

| File | Topic |
|---|---|
| `docs/research/m13_nxm_dataset_quality_v1.md` | Root causes, corrections, coverage, shortcut audit |
| `docs/research/m13_nxm_association_benchmark_v1.md` | Physics vs learned vs shortcut-only, on v1 and v2 |
| `docs/research/m13_calibration_and_abstention_v1.md` | Calibration partitions, risk–coverage, operating points, stress |
| `docs/research/m13_human_validation_readiness_v1.md` | Queue v2, label definitions, statistics, gates |

## Files

**New code**
- `src/football_fatigue/m11_explorer/nxm_dataset_v2.py`
- `src/football_fatigue/m11_explorer/nxm_assoc_v2.py`
- `src/football_fatigue/m11_explorer/review_v2.py`

**Modified**
- `m11_explorer/__main__.py`: three new commands.
- `m11_fair/gpu_pilot.py`: `gpu_busy()` now **fails closed**. If `nvidia-smi` is missing or fails, the pilot refuses to start;
  before, an unknown GPU state counted as free.

**Unchanged on purpose:** `nxm_dataset.py`, `nxm_compare.py`, `multiplayer.py`, `review_plan.py`. v1 stays reproducible.

**New tests**
- `tests/test_m11_nxm_v2.py` (24): repeat flag, duplicates, nulls, new players, one-to-one truth, seed reproducibility, timing
  independence, model view, `assign_v`, partitions, calibration metrics, review statistics, historical manifests.
- `tests/test_m13_pilot_safety.py` (4).

**New outputs (`data/processed/`)**

| Folder | Status | Contents |
|---|---|---|
| `m11-nxm-exploratory-v2` | EXPLORATORY, not frozen | 20,489 scenarios, 3,360 encounters |
| `m11-nxm-association-v2` | EXPLORATORY | `v2/` and `v1/` benchmark, calibration, quality, coverage, per-scenario and test decisions |
| `m11-review-priority-v2` | Proposal | 190 items, all `unreviewed`; blinded N×M items + coordinator key |

## 1. Root causes of the M12 defects (verified)

| Defect | Cause |
|---|---|
| `repeated_interaction` always true | `order_in_chain` is a 1-based rank, so `max > 0` is always true |
| Duplicate events | One continuous encounter appears as several link-0 catalogue groups. 15.5 % of v1 events overlapped another selected event; one edge chain spans up to 23 groups |
| Timing shortcut | One gap length per player made the true pair's `t_first − t_end` distinctive. On v1, a shortcut-only HGB beats chance by **+0.104** exact (0.487 on `async_var`) |
| N > M missing for N ≥ 4 | `unequal` almost always added a distractor (4×3: 7 scenarios), and no variant removed two tracks |
| Count shortcut | Nulls occurred only in near-N×N `unequal` scenarios |

## 2. Corrections

- **Canonical encounters:** union of shared edge chains and overlapping windows with ≥ 2 shared members; one representative each.
  83,428 candidates → 48,092 encounters.
- **A real repeat flag:** a distinct earlier chain 1 to 60 s before the encounter. True for 52.5 % of encounters.
- **Independent timing:** disappearance and reappearance times drawn independently, so the timing AUC is 0.485.
- **Seven variants:** up to 2 nulls and 2 new players, with null and new counts drawn independently.
- **Opaque outgoing ids** (hashes).
- **Model view:** systems see only anonymous {id, t, x, y} tracks.

## 3. New dataset shape

- **Size:** 20,489 scenarios, 3,360 encounters, 32 matches (73 to 144 encounters per match).
- **Encounters by N:** N = 2: 800, N = 3: 1,982, N = 4: 334, N = 5: 244.
- **Shapes:** every N×M cell from 2×1 to 5×7 has ≥ 240 scenarios, all 32 matches and ≥ 30 encounters. 4×3 has 413 (v1: 7);
  5×3 and 5×4 are new.
- **Too rare:** N ≥ 4 with fully visible contexts has 6 encounters.
- **ESTIMATED dominance** (a data property, not a bug):
  - contexts are 67 % ESTIMATED on average;
  - only 5.5 % of scenarios are all visible.

## 4 to 7. Results (exact assignment, match-level 95 % CI)

| System | M12 (v1) | M13 protocol, v1 | **M13, v2** |
|---|---|---|---|
| Shortcut-only HGB | n/a | 0.249 | **0.152** (chance 0.150) |
| S0 forward Kalman | 0.416 | 0.416 | 0.377 [0.370, 0.386] |
| S1 bidirectional | 0.483 | 0.483 | 0.446 [0.436, 0.456] |
| S2 tuned + nulls | 0.488 | 0.492 | 0.461 [0.451, 0.471] |
| L1 HGB (no null) | 0.544 | 0.541 | 0.502 [0.493, 0.512] |
| **L2 HGB + null** | 0.593 | 0.588 | **0.516** [0.506, 0.527] |
| L2t (+ texture) | 0.600 | 0.595 | 0.527 |
| H hybrid | n/a | 0.589 | 0.517 |

**Paired differences on v2.**

| Comparison | Exact difference |
|---|---|
| L2 HGB − S2 | **+0.055 [0.048, 0.062]** |
| L2 HGB − S2, N ≥ 4 | +0.047 [0.030, 0.064] |
| L2 HGB − S2, with a null | +0.090 |
| L2t − L2 (texture) | +0.011 |
| Hybrid − L2 | +0.002 |

**Nulls.**
- `null_new` (null and new player together): S2 0.001, L1 0.000, **L2 0.206**.
- At M = N, null recall is 0.001 for S2 and 0.312 for L2.
- New-player recall: S2 0.557, L2 0.688.

**Physics diagnosis.**
- The χ² gate admits 100 % of wrong pairs, so S0 and S1 choose a null only when the shape forces one.
- The true-pair constant-velocity miss is 0.97 m median, the same order as player spacing.
- That miss is 1.75 m with visible contexts against 0.68 m with ESTIMATED contexts.

**Shortcut baselines.**
- On v2, order, timing and the full shortcut-only HGB are all at chance.
- Texture-only reaches 0.210 (+0.060 over chance): a small real signal, kept out of the L2 headline.

## 8. Calibration and abstention (v2, untouched test folds)

Everything below is chosen on match-disjoint calibration folds.

**L2 HGB decision confidence.**
- ECE 0.037, Brier 0.173 (physics S2: confidence flat, uninformative).
- The threshold set on calibration folds for 95 % accuracy gives, on test:
  - coverage 32.1 %;
  - selective accuracy **0.949**;
  - false-link rate 4.9 % among accepted links;
  - null precision 0.927 among accepted nulls.

**Scenario level.** The 90 % exact target chosen on calibration gives 0.894 exact at 31.1 % coverage.

**Where it breaks.** The same threshold, by stratum:

| Stratum | Selective accuracy (target 0.95) |
|---|---|
| All-visible contexts | 0.869 |
| Partly visible contexts | 0.854 |
| Null + new | 0.920 |
| **Position noise σ = 0.15 m** | **0.801** (accuracy 0.66 → 0.49, ECE 0.095) |

## 9. Unresolved data-quality problems

- Provider identity is unverified: 0 human decisions.
- ESTIMATED interpolation dominates and makes motion more predictable than real measurement.
- Large visible groups are too rare to assess.
- Selection on continuous provider chains excludes the provider's own identity losses.
- 6.2 % of scenarios contain speeds > 12 m/s.
- Test-fold players appear in other folds (50 to 89 %, same teams). Matches are disjoint.
- The two-player GRU texture issue from M12 is unchanged; it was not part of this milestone.

## 10. Human-review readiness

`m11-review-priority-v2` holds 190 items: tiers 1 to 5 carried over unchanged, plus new N×M tiers 6 (30) and 7 (15) with one item
per v2 encounter. 35 items are flagged for double review.

**Exports**
- `queue.csv`;
- `review_template.csv`;
- blinded `nxm_items/*.json`;
- the coordinator-only `nxm_key.json`.

**Statistics**
- Exact Clopper-Pearson intervals. An upper bound ≤ 5 % needs 72, 110 or 142 non-unsure items for 0, 1 or 2 errors. This corrects
  M12's "about 120".
- κ ≥ 0.6 on ≥ 30 shared items.
- Gates V1 to V10, of which V1, V2, V3, V4 and V9 are not met.

**Decisions:** 0 human decisions exist.

## 11. Tests and runtime

**Tests**

| Run | Result |
|---|---|
| New tests (`test_m11_nxm_v2.py`, `test_m13_pilot_safety.py`) | 28 passed |
| `test_m11_fair_v2.py` | 10 passed |
| Full suite (one run at the end) | **1,235 passed, 6 skipped, 2 sklearn warnings** in 303 s. M12: 1,207 passed, 6 skipped |

**Checks**
- Import isolation: OK.
- Historical M12 outputs: byte-identical.

**Runtime (CPU only, sequential)**

| Job | Time |
|---|---|
| Dataset build | 937 s |
| Benchmark, v2 + v1 | 2,468 s |
| Review export | < 1 min |
| Full test suite | 303 s |

**Compute note.** Session B started a CPU job (`football_fatigue.video.failures activation`, about one core) at 08:08 while the
benchmark ran. I tried to restart my job with fewer threads, but stopping processes was denied, so both ran concurrently on the 8
logical CPUs. Session B was not touched.

## 12. GPU pilot: ready for separate approval (unchanged from M12)

**Verified**
- The 7 configurations and estimates are identical to the saved `m11-fair-gpu-pilot-plan-v1/pilot_plan.json`.
- `test_m11_fair_v2.py` passes (10).
- Refused without `--approval` (exit 2).
- No pilot output exists.
- `gpu_busy()` is now fail-closed.

**The plan**

| | |
|---|---|
| Configurations | GRU h48, LSTM h40, LSTM h42 (parameter-matched), GRU h96 l2, LSTM h96 l2 (all `centred`); GRU h48 and LSTM h42 (`shift_mix`) |
| Scope | Outer fold 0, seed 0, cap 400 epochs, patience 25 |
| Estimate | **1.57 GPU-h likely, 2.51 worst case [EST]**; peak < 1.5 GB [EST]; stop at 3.5 GB or 4 h |
| Resume | Re-run the same command |
| Command | `scripts\session_a_python.cmd -m football_fatigue.m11_fair gpu-pilot --approval <id> --device cuda` |

**Current blocker:** `nvidia-smi` lists the ChatGPT desktop app (`ChatGPT.exe`) as a GPU compute process, so the pilot would refuse
to start now. Close it, and make sure Session B is not on the GPU, before running.

## Final answers

1. **Is the corrected N×M dataset more scientifically defensible?** Yes.
   - Duplicates are removed: 0 overlapping events vs 202 pairs in v1.
   - The repeat flag is real (52.5 %).
   - Timing, order, length and count shortcuts are closed; the shortcut-only model is at chance (+0.002 vs +0.104 on v1).
   - Nulls and new players are covered for every N.
   - Effective sample sizes are stated as encounters.

   It remains exploratory: provider truth, mostly ESTIMATED contexts, synthetic gaps.
2. **Did HGB still outperform the physics solvers?** Yes, by less.
   - L2 HGB 0.516 vs S2 0.461: +0.055 [0.048, 0.062]; +0.047 for N ≥ 4; +0.090 with nulls.
   - It also has fewer identity switches (0.60 vs 0.77).
   - Part of the M12 margin (+0.105) came from the v1 timing shortcut.
   - Adding physics to HGB (hybrid) adds nothing. Texture adds +0.011 and is excluded from the headline.
3. **Which N×M scenarios remain unreliable?**
   - Null and new player together: 0.21 for the best system, about 0 for physics.
   - Partly visible contexts: 0.36.
   - Scenarios with new players: 0.36 to 0.39.
   - Synchronous crossings: 0.43.
   - N ≥ 4 with fully visible contexts: only 6 encounters, not assessable.
   - Anything video-like: position noise of 0.15 m cuts decision accuracy from 0.66 to 0.49.
4. **Can the model abstain with calibrated confidence?** In-domain only.
   - A threshold chosen on separate calibration matches gives 0.949 decision accuracy at 32 % coverage on untouched test matches.
   - The guarantee fails on visible contexts (0.85 to 0.87) and under video-grade noise (0.80).
   - Physics margins are useless for abstention.
   - It is not deployable without calibration data from the deployment domain.
5. **Is the evidence sufficient to freeze dataset v2?** **No.**
   - 0 human decisions.
   - Gates V1 (label error), V2, V3, V4 (agreement) and V9 (N×M link error) are unmet.
   - Visible large-group coverage is inadequate.
6. **Is the GPU pilot ready for separate approval?** Yes.
   - It is unchanged, verified, refused without approval, and now fail-closed.
   - About 1.6 to 2.5 GPU-h [EST].
   - Close the desktop app that currently shows on the GPU first.
7. **What should be done next?** Suggested **M14: first human review round + GPU pilot + visible-domain stress**:
   - (a) A3: review the 190-item queue, with a second reviewer on the 35 double-review items.
   - (b) A1: run the GPU pilot after freeing the GPU.
   - (c) A4: small Studio change to load `m11-review-priority-v2` and the N×M link question.
   - (d) CPU: train and calibrate the N×M scorer with visible-only weighting and position-noise augmentation. Re-measure on the
     visible stratum and under σ = 0.15 m noise.
   - (e) Only after V1, V2, V4 and V9 pass: freeze dataset v2.

## Approvals needed (none given)

| ID | Request | Cost |
|---|---|---|
| A1 | GPU pilot | About 1.6 to 2.5 GPU-h [EST] |
| A2 | Full fair benchmark, sized from A1 | To be sized |
| A3 | Reviewer time: 190 items + second reviewer on 35 | Reviewer hours |
| A4 | Studio queue loading (`m11-review-priority-v2`) and N×M link question | Small UI change |
| A5 | Done in M13 | — |
| A6 (new) | Visible-domain / noise-augmented N×M training and calibration | CPU, about 1 h |
