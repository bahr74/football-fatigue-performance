# Phase 2, Task D: four-model benchmark readiness (GRU / LSTM / SVM / HGB + deterministic baselines)

**Date:** 2026-11-20

**Nature:**
- A read-only audit of saved training records, plus a CPU dry-run validation of a proposed partition and experiment matrix.
- **Nothing was trained, no checkpoint was opened or changed, and no model was run.**

**Code:** `src/football_fatigue/m11_phase2/readiness.py`

**Command:** `.\.venv\Scripts\python.exe -m football_fatigue.m11_phase2 readiness`

**Outputs:** `data/processed/m11-phase2-model-readiness-v1/`
- `readiness_results.json`
- `neural_fit_records.json`: per-fit epochs, losses, timing
- `proposed_partition.json`
- `proposed_experiment_matrix.json`
- `manifest.json`

**Sources read:**
- `m11-stage4-benchmarks-v1/preds/g0p80/<model>/seed*.json` and `folds.json`;
- `m11-stage4c-robustness-v1/neural/<cfg>/seed*/fold*.json`;
- code in `m11_benchmark/{models,run,folds,stage4c}.py`.

---

## D1. Historical training audit [V from records]

### Neural models (Stage 4A/4B, table g0p80, 8 outer folds × seeds)

| Model | Params | Config | Fits | Best epoch min / median / max | Best epoch ≥ cap − 2 | ≥ cap − 10 | Stopped early | Val-LL slope, last 10 epochs (median) | Train − val BA | s / fit (median) |
|---|---|---|---|---|---|---|---|---|---|---|
| **GRU gap-last** | 43,585 | hidden 48, emb 48, lr 1e-3, wd 1e-4, dropout 0.1, batch 256, cap 80, patience 10 | 24 (3 seeds) | 76 / 79 / 80 | **19** | **24** | 0 | −0.0025 / epoch | 0.013 | 99 s |
| **LSTM gap-last** | 41,985 | hidden **40**, otherwise identical | 24 (3 seeds) | 39 / 78 / 80 | **13** | **21** | 3 | −0.0009 / epoch | 0.015 | 231 s |
| MLP | 55,873 | 96 → 48 | 24 | 40 / 65.5 / 79 | 4 | 6 | 18 | −0.0002 | 0.015 | 80 s |
| GRU (far-end summary, superseded) | 43,585 | — | 8 | 26 / 46.5 / 78 | 1 | 2 | 6 | — | 0.002 | 71 s |
| LSTM (far-end summary, superseded) | 41,985 | — | 8 | 16 / 28 / 69 | 0 | 0 | 8 | — | −0.001 | 49 s |

All fits ran on CUDA (GTX 1650 Ti). Seconds per epoch: **GRU 1.24, LSTM 3.15** (×2.5).

### Stage 4C (GRU only), best early-stopping state within caps 80 / 120 / 160

| Config | Fits | At cap (80 / 120 / 160) | Validation BA mean (80 / 120 / 160) |
|---|---|---|---|
| A (centred) | 24 | 19 / 5 / 1 | 0.902 / 0.918 / 0.919 |
| B | 24 | 12 / 6 / 6 | 0.863 / 0.874 / 0.883 |
| C (shift mix) | 24 | 21 / 12 / **5** | 0.890 / 0.916 / **0.925** |
| Anoise / Asmooth | 8 each | 4/2/0 · 3/1/0 | 0.879/0.888/0.888 · 0.861/0.869/0.874 |

### Tabular models

| Model | Grid | Seeds | Selected (24 fold fits) | Train − val BA | s / fit (median) |
|---|---|---|---|---|---|
| SVM (Nystroem 1,000 + LinearSVC + Platt) | 6 (C ∈ {0.1, 1}, γ ∈ {0.5, 1, 2}) | 3 (≈ identical) | **C = 1, γ = 0.5 in 24 / 24: both on the grid edge** | 0.007 | 52 s |
| HGB | 4 (lr ∈ {0.05, 0.1}, leaves ∈ {15, 31}, iters 400 / 200) | 3 (identical) | lr 0.05 / 31 leaves / 400 iters ×15; lr 0.1 / 31 / 200 ×9: **always the largest leaf count** | 0.051 | 4.5 s |
| RF | 2 | 3 | max_depth None ×24 | 0.119 | 17 s |
| LR | 4 | 1 | C = 1 ×7, C = 10 ×1 | — | 1.3 s |

**Other records:**
- **Calibration:**
  - SVM: Platt scaling fitted inside the training partition.
  - Neural and HGB: none beyond the loss.
  - ECE about 0.008–0.010 [DOC].
- **Model selection:** "selected models" for Stage 4C and M1–M3 were chosen on the same OOF results that are reported
  (selection-optimistic) [DOC].
- **Seeds:**
  - Neural: 3 (0, 1, 2).
  - Tabular: 3, deterministic, so effectively 1.
- **Folds:** 8 frozen outer folds of 4 matches (`folds.json`, salt `m11-stage3-folds-v1`). 3 early-stop matches per outer fold;
  3 inner folds for tabular grids.

### Is the historical LSTM comparison fair? **No** [V]

**Fair:** the same samples, folds, seeds, augmentation, early-stopping matches, gap-last orientation and roughly equal parameters.

**Not fair:**
1. **Budget-bound for both, worse for the LSTM in time.**
   - GRU: 24 / 24 fits had their best epoch within 10 of the 80-epoch cap.
   - LSTM: 21 / 24 fits did. Its validation loss was still falling (median slope −0.0009 per epoch).
   - The LSTM was never given 120 / 160 epochs. The GRU was, and gained +0.017 BA (A, 80 → 160).
2. **Smaller hidden size** (40 vs 48), chosen to equalise parameter count. Defensible, but it was never searched.
3. **No hyperparameter search for either network**, while SVM, HGB, RF and LR had inner-CV grids.
4. **No shift (config C) training, no stress tests and no texture ablation for the LSTM.**
5. **Tabular grids are also boundary-bound** (SVM γ and C at the edge; HGB at the largest leaf count). The tabular results are not
   converged in the hyperparameter sense either.

**Defensible statement:** "At an identical 80-epoch budget without tuning, the gap-last GRU outperformed the gap-last LSTM (0.902 vs
0.857 BA); neither had converged, and the LSTM was not given the extended training the GRU later received."

## D2. Fair comparison protocol (proposed)

1. **Population and splits**
   - The same eligible events (Stage 3 g0p80 first; dataset v2 only after its gates).
   - Frozen outer folds.
   - Per outer fold: **fit (22 matches) / early-stop (3) / calibration (3) / test (4)**, all match-disjoint.
   - The partition is `proposed_partition.json`. Its validation passes all eight rules (§D3).
2. **Feature access, stated per family**
   - Sequence models: observed tracklets.
   - SVM / HGB: the 36 engineered motion features from the same observed tracklets.
   - Status, ids and label-derived difficulty are never inputs (dataset v2 §5).
3. **Equal tuning budget:** 24 configurations per family, drawn by a fixed-seed random or Sobol search.

   | Family | Search space |
   |---|---|
   | GRU / LSTM | lr [3e-4, 3e-3], hidden {32, 48, 64, 96}, layers {1, 2}, dropout [0, 0.3], wd [1e-5, 1e-3], batch {128, 256} |
   | SVM | C [0.03, 30], γ [0.05, 4], Nystroem {1k, 2k} |
   | HGB | lr [0.02, 0.2], leaves {15, 31, 63, 127}, min-leaf [10, 80], l2 [0, 2], iterations up to 2,000 with early stopping |

   Grid edges are checked: a selected value on an edge triggers one extension round, which is recorded.
4. **Convergence:**
   - Neural cap 400 epochs, patience 25.
   - A fit whose best epoch lies within 10 of the cap is flagged, and the cap is raised for that family.
   - HGB uses early stopping on the early-stop matches.
5. **Seeds:**
   - Neural: ≥ 5 seeds × 8 folds; report fold × seed variance.
   - Tabular: 3 seeds where stochastic.
6. **Training distributions offered to every family:** centred, shift-augmented, variable-gap, noise/dropout. Every family receives
   the same augmented samples.
7. **Calibration and abstention:**
   - Temperature or Platt fitted on the **calibration matches only**.
   - Abstention thresholds fitted there too.
   - Risk–coverage reported on test.
8. **Model selection:** the protocol-level choice (which configuration per family) uses inner folds only. **No selection on test
   OOF.** The four-family comparison reports all four, with paired match-bootstrap differences.
9. **Metrics and strata:**
   - BA, log loss, Brier, ECE, AURC.
   - Strata: d_band, speed, status, context, physics tier (Task B), clone-like.
   - The Task B physics baselines as reference rows.
10. **Compute ledger:** GPU / CPU seconds per fit, from the training records.

## D3. Implementation readiness (code inspection) [V]

| Requirement | Current code | Change needed |
|---|---|---|
| LSTM with extended caps / config C | `stage4c.py` hard-codes `KIND = "gru_gaplast"` | Parameterise the kind; keep the snapshot logic |
| Hyperparameter search for neural models | `Neural` accepts lr, wd, dropout, batch, epochs, patience; sizes come from the fixed `NEURAL` dict in `models.py` | Pass hidden / emb / layers through; add `num_layers` to `make_net` |
| Separate calibration partition | Not present (only early-stop matches) | Add a calibration role to the fold logic (the proposal reuses `folds._rank` with new salts; frozen outer folds unchanged) |
| Equal tabular search | Fixed small grids in `run.py` | Replace with a seeded search of the same size per family |
| Five neural seeds | `--seeds` CLI accepts any list | None |
| Convergence flagging | Records contain `best_epoch` / `epochs_run` | Add an automatic at-cap check (`readiness.py` already computes it) |
| Selection without test leakage | `robust/selected_models.json` is chosen on OOF | Define selection on inner folds |
| Deterministic baselines as reference rows | Task B module | Wire into evaluation |
| Write-once outputs, held-out refusal | Present (`write_json_once`, Stage 3 adapter) | None |

**Hidden assumptions found:**
- Early-stop matches are excluded from fitting but double as model-selection data inside `Neural.fit`. This is acceptable.
- The outgoing and incoming exchange augmentations assume a 2 × 2 structure. They will not carry over to K-candidate schemas
  unchanged.
- Training uses all rows (25,331) including dependent ones; evaluation uses the independent subset. This is acceptable, but it must
  be stated.

**Dry-run validator (CPU):**

| Rule | Result |
|---|---|
| All four families present at priority 1 | ✔ |
| Neural seeds ≥ 5 | ✔ |
| Equal tuning budget per purpose | ✔ |
| Neural cap ≥ 2 × historical best epoch, with patience recorded | ✔ |
| Calibration disjoint from fit / early-stop / test | ✔ |
| Every partition match is train/CV | ✔ |
| Outer folds equal the frozen Stage 4 folds | ✔ |
| Test folds partition the 32 matches | ✔ |

## D4. Experiment matrix and compute [EST]

**Matrix:** `proposed_experiment_matrix.json`; 29 rows.

| Priority | Purpose | Families |
|---|---|---|
| 1 | Fair re-benchmark, centred | GRU, LSTM, SVM, HGB + 5 deterministic baselines |
| 1 | Shift-augmented training for all | GRU, LSTM, SVM, HGB |
| 2 | Variable-gap training 0.2–2.4 s | All four |
| 2 | Noise / dropout (texture-controlled) training | All four |
| 2 | Clone-like exclusion sensitivity | All four |
| 3 | Ambiguity-enriched v2 population (after dataset v2 gates) | All four |

**Evaluation strata on every row:** distance band, gap, noise, dropout, VISIBLE / ESTIMATED composition, isolated / group context,
clone-like, physics tier.

**GPU-hours (neural only; tabular is CPU-minutes):**

Seconds per epoch are taken from the records; the mean epochs are assumed to be 0.5 × cap; augmentation multiplies the cost (shift
×7, gap or noise ×4).

| Scenario | Priority 1 | All |
|---|---|---|
| Full nested tuning (24 configurations × 3 inner folds × 8 outer folds) | 1,200 GPU-h | 2,700 GPU-h |
| **Budgeted** (successive halving η = 3 on one inner split per outer fold) | **291 GPU-h** | 655 GPU-h |

**Priority-1 cost by run** (budgeted tuning + final):

| Run | GPU-h |
|---|---|
| GRU centred | 7.5 + 2.7 |
| LSTM centred | 19.1 + 7.0 |
| GRU shift | 52.6 + 19.2 |
| LSTM shift | 133.9 + 49.0 |

**Recommendation [REC]:**
- Run priority 1 *centred* first: about 36 GPU-h for GRU + LSTM, plus CPU for SVM / HGB. This gives a fair answer to the GRU vs
  LSTM question.
- Run the shift-augmented arm after an approval that accounts for its 255 GPU-h.
- Consider subsampling the shift variants: one random shift per sample per epoch instead of all seven, which costs ×1 instead of ×7.

## What must change before fair GRU / LSTM / SVM / HGB training

1. Approve the partition (fit / early-stop / calibration / test) and the 24-configuration equal budget.
2. Parameterise `stage4c.py` (model kind) and `make_net` (sizes, layers).
3. Add the calibration role and no-OOF selection.
4. Replace the tabular grids with equal-size seeded searches with edge checks.
5. Add automatic convergence flags.
6. Approve the GPU budget.
7. Decide the population: v1 now; v2 later.
