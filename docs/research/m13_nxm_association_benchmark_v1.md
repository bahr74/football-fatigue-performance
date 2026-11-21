# M13 report 2: N×M association benchmark on the corrected dataset (v1)

Status: **EXPLORATORY.**
- Truth is provider identity: not human-reviewed, not video.
- Every number is from untouched test folds of the 32 development matches.

Date 2026-11-21.

- Code: `src/football_fatigue/m11_explorer/nxm_assoc_v2.py`, which reuses M12's `nxm_compare` cost, feature and bootstrap
  functions unchanged.
- Run with `scripts\session_a_python.cmd -m football_fatigue.m11_explorer nxm-assoc-v2`. CPU only, 2,468 s for v2 and v1 together.
- Output: `data/processed/m11-nxm-association-v2/{v2,v1}/{benchmark,calibration,quality,coverage}.json`,
  `per_scenario.parquet`, `test_decisions.parquet`, `summary.json` and `manifest.json`.
- Tests: `tests/test_m11_nxm_v2.py`.

## 1. Protocol

**Partitions.** For each outer test fold f (frozen Stage 4 folds, match-disjoint):

| Partition | Folds | Used for |
|---|---|---|
| Fit | 5 folds | Fitting LR / HGB link scorers |
| Calibration | (f+1, f+2) mod 8 | Null τ, hybrid β, calibrators, abstention thresholds |
| Test | f | Scored once |

- **Physics S2** is tuned on a fixed bounded subset of the 7 non-test folds (σ_a × incoming null × outgoing null grid, as in M12).
- **Nothing** is selected on a test fold.
- **CIs** are match-level bootstraps (32 matches, 1,000 draws) on single systems and on paired differences.
- **Inputs** are `view(d)` only: anonymous tracks {id, t, x, y}. There is no provider status, no hidden path and no meta (tested).

**Systems.**

| Group | ID | Definition |
|---|---|---|
| Shortcut-only | RAND | Random one-to-one assignment |
| | SC_order | Outgoing id order |
| | SC_timing | Rank of disappearance ↔ rank of reappearance |
| | **SC_hgb** | HGB on non-motion features: time difference, relative times, sample counts, N, M; with tuned null |
| | SC_tex | HGB on provider-texture features only; with null |
| Physics | S0 | Forward Kalman + Hungarian, 0.99 gate (M12 original) |
| | S1 | Bidirectional Kalman |
| | S2 | S1 with tuned σ_a and separate null costs |
| Learned | L1 | LR / HGB link scorer + Hungarian, no explicit null |
| | L2 | + explicit null −log τ |
| | L2t | + 2 texture features (ablation) |
| | H | Hybrid: −log p_HGB + β·d2_bidirectional, with null; β ∈ {0.02 to 0.2} chosen on calibration |

**Global-consistency reranking was not implemented.** It is optional in the brief, and a useful reranker needs a trained
scenario-level model over k-best assignments. The hybrid result below (+0.002) suggests little headroom from re-weighting physics.

## 2. Comparison with M12, on v1 and on v2 (exact assignment, match-level 95 % CI)

| System | M12 report (v1) | M13 protocol on v1 (8,540) | M13 protocol on **v2** (20,489) |
|---|---|---|---|
| RAND | n/a | 0.145 [0.137, 0.153] | 0.150 [0.145, 0.156] |
| **SC_hgb (shortcut-only)** | n/a | **0.249** [0.237, 0.261] | **0.152** [0.148, 0.157] |
| SC_timing | (timing oracle 0.45 async) | 0.225 | 0.152 |
| SC_tex | (texture rule near chance) | 0.203 | 0.210 [0.203, 0.216] |
| S0 | 0.416 | 0.416 | 0.377 [0.370, 0.386] |
| S1 | 0.483 | 0.483 | 0.446 [0.436, 0.456] |
| S2 | 0.488 | 0.492 | 0.461 [0.451, 0.471] |
| L1 LR / HGB | 0.479 / 0.544 | 0.479 / 0.541 | 0.443 / 0.502 [0.493, 0.512] |
| L2 LR | 0.505 | 0.506 | 0.445 |
| **L2 HGB** | **0.593** | **0.588** [0.575, 0.600] | **0.516** [0.506, 0.527] |
| L2t HGB (+ texture) | 0.600 | 0.595 | 0.527 [0.517, 0.538] |
| H HGB (hybrid) | n/a | 0.589 | 0.517 [0.507, 0.529] |

- The M13 protocol reproduces M12 on v1 within 0.005, even though τ now comes from separate calibration folds and the scorers are
  fitted on 5 instead of 7 folds.
- **v2 is harder for every system.** That is expected:
  - duplicate encounters are gone;
  - timing carries no information;
  - null + new cases are 14 % of scenarios;
  - all N ≥ 3 encounters are used.
- **The v1 headline was inflated by its shortcut.** The shortcut-only model beats chance by **+0.104 [0.094, 0.113]** on v1. On the
  v1 `async_var` variant alone it is exact in 0.487 of scenarios (physics S2: 0.646). On v2 the gain is
  **+0.002 [−0.003, 0.007]**, so the timing, order, length and count shortcuts are closed.

## 3. Learned vs deterministic on v2 (paired, match-level CI)

| Comparison | Exact difference |
|---|---|
| **L2 HGB − S2 (best physics)** | **+0.055 [0.048, 0.062]** |
| L2 HGB − S2, N ≥ 4 | +0.047 [0.030, 0.064] |
| L2 HGB − S2, scenarios with a null | +0.090 [0.081, 0.099] |
| L2 HGB − S0 | +0.138 [0.130, 0.147] |
| S2 − S0 (bidirectional + tuning) | +0.083 [0.075, 0.092] |
| S1 − S0 (bidirectional only) | +0.069 [0.062, 0.076] |
| L2 HGB − L2 LR | +0.071 [0.065, 0.077] |
| L2 HGB − SC_hgb (motion vs shortcut) | +0.364 [0.353, 0.374] |
| **L2t − L2 (texture reliance)** | **+0.011 [0.008, 0.015]** |
| H − L2 (physics added to HGB) | +0.002 [0.001, 0.003] |

**Full metrics (v2, test folds).**

| System | Exact | Link P | Link R | F1 | ID switches / scenario | False-link rate | Null recall | Null precision | New-player recall | New precision |
|---|---|---|---|---|---|---|---|---|---|---|
| S0 | 0.377 | 0.525 | 0.559 | 0.541 | 0.94 | 0.475 | 0.467 | 0.635 | 0.541 | 0.762 |
| S1 | 0.446 | 0.588 | 0.626 | 0.606 | 0.79 | 0.412 | 0.494 | 0.672 | 0.552 | 0.779 |
| S2 | 0.461 | 0.597 | 0.635 | 0.616 | 0.77 | 0.403 | 0.506 | 0.685 | 0.557 | 0.783 |
| L1 HGB | 0.502 | 0.647 | 0.689 | 0.667 | 0.65 | 0.353 | 0.526 | 0.715 | 0.576 | 0.811 |
| **L2 HGB** | **0.516** | **0.669** | 0.679 | **0.674** | **0.60** | **0.331** | **0.602** | 0.643 | **0.688** | 0.740 |
| L2t HGB | 0.527 | 0.672 | 0.691 | 0.681 | 0.59 | 0.328 | 0.596 | 0.671 | 0.676 | 0.771 |
| SC_hgb | 0.152 | 0.269 | 0.285 | 0.277 | 1.56 | 0.731 | 0.382 | 0.511 | 0.254 | 0.352 |

## 4. Breakdowns (v2, exact)

| Stratum | S0 | S2 | L1 HGB | L2 HGB | L2t | SC_hgb |
|---|---|---|---|---|---|---|
| N = 2 | 0.560 | 0.592 | 0.631 | 0.654 | 0.661 | 0.346 |
| N = 3 | 0.353 | 0.434 | 0.477 | 0.489 | 0.498 | 0.136 |
| N = 4 | 0.292 | 0.423 | 0.470 | 0.485 | 0.500 | 0.034 |
| N = 5 | 0.268 | 0.421 | 0.447 | 0.448 | 0.479 | 0.005 |
| M < N | 0.375 | 0.459 | 0.494 | 0.504 | 0.512 | 0.220 |
| M = N | 0.429 | 0.501 | 0.553 | 0.571 | 0.580 | 0.178 |
| M > N | 0.307 | 0.405 | 0.438 | 0.451 | 0.468 | 0.038 |
| 0 / 1 / 2 nulls | 0.45 / 0.27 / 0.30 | 0.55 / 0.33 / 0.37 | 0.60 / 0.36 / 0.39 | 0.57 / 0.44 / 0.43 | 0.59 / 0.45 / 0.43 | n/a |
| 0 / 1 / 2 new players | 0.47 / 0.23 / 0.22 | 0.56 / 0.30 / 0.29 | 0.62 / 0.32 / 0.32 | 0.60 / 0.39 / 0.36 | 0.62 / 0.39 / 0.38 | n/a |
| `complete` (sync, no dropout) | 0.600 | 0.669 | **0.744** | 0.724 | 0.744 | 0.212 |
| `async` | 0.446 | 0.552 | **0.605** | 0.571 | 0.587 | 0.221 |
| `null1` / `null2` | 0.40 / 0.44 | 0.49 / 0.53 | 0.54 / 0.56 | 0.53 / 0.56 | 0.55 / 0.57 | n/a |
| `new1` / `new2` | 0.36 / 0.33 | 0.48 / 0.44 | 0.51 / 0.48 | 0.50 / 0.46 | 0.51 / 0.48 | n/a |
| **`null_new`** | **0.000** | **0.001** | **0.000** | **0.206** | 0.183 | 0.000 |
| Gap 0.7 to 1.0 / 1.0 to 1.3 / > 1.3 s | 0.54 / 0.33 / 0.32 | 0.61 / 0.42 / 0.41 | 0.68 / 0.45 / 0.44 | 0.67 / 0.47 / 0.46 | 0.69 / 0.48 / 0.47 | n/a |
| Context all visible (263 encounters) | 0.415 | 0.452 | 0.480 | 0.486 | 0.493 | 0.255 |
| Context ESTIMATED ≤ 50 % | 0.275 | 0.300 | 0.358 | 0.357 | 0.368 | 0.154 |
| Context ESTIMATED > 50 % | 0.415 | 0.524 | 0.561 | 0.581 | 0.592 | 0.143 |
| Synchronous crossing | 0.302 | 0.387 | 0.424 | 0.432 | 0.446 | 0.117 |
| Label-risk edge | 0.318 | 0.454 | 0.482 | 0.488 | 0.512 | 0.061 |

**By shape (n scenarios; S2 → L2 HGB).**

| Shape | n | S2 | L2 HGB |
|---|---|---|---|
| 2×1 | 800 | 0.69 | 0.73 |
| 2×2 | 1,935 | 0.61 | 0.68 |
| 3×1 | 1,982 | 0.55 | 0.58 |
| 3×2 | 2,498 | 0.35 | 0.40 |
| 3×3 | 4,790 | 0.47 | 0.54 |
| 3×4 | 2,052 | 0.36 | 0.42 |
| 3×5 | 1,381 | 0.41 | 0.43 |
| 4×2 | 334 | 0.44 | 0.53 |
| **4×3** | 413 | 0.35 | 0.43 |
| 4×4 | 847 | 0.47 | 0.52 |
| 4×5 | 398 | 0.37 | 0.46 |
| 4×6 | 308 | 0.44 | 0.45 |
| **5×3** | 244 | 0.44 | 0.48 |
| **5×4** | 299 | 0.34 | 0.39 |
| 5×5 | 620 | 0.45 | 0.49 |
| 5×6 | 300 | 0.40 | 0.43 |
| 5×7 | 240 | **0.44** | **0.40** |

Physics is better only at 5×7 (240 scenarios). The 4×3, 5×3 and 5×4 rows were absent or nearly absent in v1.

## 5. Null and new-player performance

**Null + new together (`null_new`, 2,790 encounters) is the decisive failure.** Physics and forced matching are exact in essentially
0 % of these scenarios. L2 HGB reaches 0.206: the explicit null is the only mechanism that ever gets these right.

**At M = N** a null and a new player coexist, so the shape gives no hint:

| System | Null recall | Null precision | New-player recall |
|---|---|---|---|
| S2 | 0.001 | 0.08 | 0.002 |
| L2 HGB | 0.312 | 0.428 | 0.391 |

Physics never predicts a null that the shape does not force; it links the distractor instead.

**Diagnosed causes (measured on 3,000 v2 scenarios, 7,181 true pairs).**
1. **The gate rejects nothing.** With σ_a = 8 the χ²(2, 0.99) gate admits **100 %** of wrong pairs and rejects **0 %** of true pairs.
   S0 and S1 therefore never choose a null voluntarily. S2's tuned null costs sit at the grid edges in several folds (incoming
   null 0.5, outgoing null 16 to 36): the tuning drives nulls cheap for incoming and expensive for outgoing.
2. **Motion prediction error is comparable to player separation.**
   - The constant-velocity miss for true pairs has median 0.97 m and p90 3.9 m, while the group radius is 1.0 m.
   - The miss grows with the gap: 0.77, 0.99 and 1.29 m for gaps ≤ 1.0 s, 1.0 to 1.3 s and > 1.3 s.
   - It is larger for crossings (1.17 vs 0.79 m).
   - It is **2.6× larger with visible contexts (1.75 m) than with ESTIMATED contexts (0.68 m).** Provider interpolation is smoother
     than measured motion. That is why every system scores higher on ESTIMATED-dominated scenarios than on partly visible ones
     (0.58 vs 0.36 for L2 HGB).
3. **A trade-off between nulls and no-null cases.** L1 (no null) beats L2 where no null exists (`complete` 0.744 vs 0.724; `async`
   0.605 vs 0.571). L2 wins whenever one does. A single τ, chosen on calibration matches, is the honest compromise, as in M12.
4. **Asynchronous reappearance** costs 0.12 to 0.15 exact (`complete` → `async`) for every motion system.
5. **Group consistency** was not enforced beyond one-to-one assignment. No reranker was trained (section 1).

## 6. Does any improvement come from shortcuts or artefacts?

| Shortcut | Evidence on v2 | Verdict |
|---|---|---|
| Timing, order, track length, counts | SC_hgb, which sees all of them, is at chance: +0.002 [−0.003, 0.007] over RAND. On v1 it was +0.104 | **Closed in v2.** The v1 learned scores partly rode on this |
| Count / null structure | Null recall at M = N: S2 0.001, L2 0.312. SC_hgb has no null skill beyond chance | The learned null skill comes from motion |
| Provider texture | SC_tex − RAND = +0.060 [0.052, 0.068]; L2t − L2 = +0.011 [0.008, 0.015] | **A real but small signal.** L2 (headline) excludes texture features. L2t's gain is not a motion gain |
| ESTIMATED predictability | All systems score higher on ESTIMATED-dominated contexts; true-pair CV miss is 0.68 m vs 1.75 m visible | Not a model shortcut but a **data property**. Headline numbers are dominated (68 %) by provider-interpolated motion |
| Correlated variants | Paired match-level bootstrap; 3,360 encounters | Differences are robust to match resampling |

## 7. Unreliable scenarios (exact accuracy, or too few independent encounters)

| Scenario | Why it is unreliable |
|---|---|
| **Null + new player together** | L2 HGB 0.21; physics about 0 |
| **Partly visible contexts** (ESTIMATED ≤ 50 %) | 0.36 for L2 HGB; the hardest real-data stratum |
| **N ≥ 3 with new players** | 0.36 to 0.39 |
| **Synchronous crossings** | 0.43 |
| **N ≥ 4 with fully visible contexts** | Only 6 encounters: no conclusion possible |
| **Video domain** | Untested. Position noise of 0.15 m cuts decision accuracy from 0.66 to 0.49 (report 3) |
