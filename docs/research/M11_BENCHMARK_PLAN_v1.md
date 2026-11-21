# M11 benchmark plan: MATCHED/SWAPPED models (v1, plan only)

Status: **plan for approval. No model is trained, no Stage 3 sample is built.**

Depends on:
- `docs/research/M11_STAGE3_SUPERVISED_DATASET_DESIGN_v1.md` (samples, schema, leakage controls);
- `docs/research/m11_stage2_results_v1.md` (counts);
- the frozen split `m11-stage2-split-v1` (32 train/CV matches, 8 held-out matches).

## 1. Task

Given two incoming and two outgoing tracklets around a hidden gap, predict p(SWAPPED). Class 0 is MATCHED and class 1 is SWAPPED. The prediction is symmetrised over the outgoing order.

- **Primary table:** gap 0.8 s, gap centred on t*.
- **Secondary tables:** gaps 0.4 s and 1.6 s, and the shifted-centre table.

## 2. Model ladder (run in this order, stop adding complexity when it stops paying)

| # | Family | Model | Input | Notes |
|---|---|---|---|---|
| B0 | rule | nearest endpoint | last / first observed points | no fitting |
| B1 | rule | constant velocity, forward and bidirectional | `motion` fits | existing |
| B2 | rule | constant acceleration, bidirectional | `motion` fits | existing |
| B3 | calibrated rule | best of B0–B2, cost margin → logistic | 1 feature | existing `models.Baseline`; calibration fitted in each training fold |
| T1 | tabular | logistic regression (L2) | ~60 antisymmetric features from `motion.features` | standardised in fold |
| T2 | tabular | k-NN | same | k chosen by inner CV from {15, 31, 63} |
| T3 | tabular | RBF SVM | same, Nyström 1,000 components + linear SVM, Platt-calibrated | O(n) memory |
| T4 | tabular | random forest | same | 300 trees; fixed depth grid {None, 12} |
| T5 | tabular | HistGradientBoosting | same | existing `models.GBM`; xgboost/LightGBM only if installation is approved (decision B1) |
| S1 | sequence | MLP on flattened masked sequences | `in_pos`, `out_pos`, masks, dt | small (≈30k parameters) |
| S2 | sequence | GRU / LSTM pair encoder (PairNet) | sequences + physics pair score | existing `models.Hybrid`; hidden 64 |

- **Hyperparameters.** Each model gets one small predeclared grid of at most 6 settings, chosen by inner grouped CV inside the training folds. There are no broad searches.
- **Same inputs for every model.** No identity, team, status, crossing class, F1–F4 state or pitch position. A test checks the feature list.

## 3. Cross-validation (train/CV matches only)

- **Folds.** Grouped 8-fold CV by match: 32 matches, 4 per fold, assigned by `sha256("m11-stage3-folds-v1|match")`. The assignment is frozen in a `folds.json` with a hash before any model runs.
  - Leave-one-match-out is kept as a check for the best two models only, because of cost.
- **Out-of-fold predictions.** Every train/CV sample gets exactly one, written per model. All model comparisons, calibration plots and threshold choices use these predictions only.
- **Fold-local fitting.** Scalers, calibrators and early stopping (neural models: one inner validation match drawn from the training fold) are fitted inside each fold.
- **Seeds.** Every model gets 3 seeds; neural models get 3 initialisations. Means and spreads are reported.

## 4. Imbalance and strata

- **Swap labels.** Balanced by construction (independent hashed orders), so no reweighting is needed for y.
- **Encounter mix.** Most accepted encounters are not local crossings. The headline metric is reported on all samples, and separately per crossing class. Crossing class is never used as an input or a training target.
- **Controls.** Matched non-intersection controls are an evaluation stratum, used to check false-alarm behaviour of the abstention rule. Their weights are capped (decision B2; proposed cap 10, with uncapped as a sensitivity check).

## 5. Metrics

**Primary:**
- **Balanced accuracy on the independent subset**, at the symmetric threshold 0.5. This is the headline.
- **Log loss**, as the probability-quality headline.

**Secondary:**
- accuracy, F1 for SWAPPED, ROC AUC, Brier score, ECE (15 bins);
- **selective-prediction curve:** coverage vs. error; coverage at 1% and 5% error;
- swap-recovery rate: the fraction of truly SWAPPED cases correctly flagged.

**Uncertainty:** 95% match-level bootstrap intervals (2,000 resamples), using the existing `evaluate.match_bootstrap`.

**Paired comparisons:** model vs. B3 and model vs. T5, as the bootstrap of the per-match difference, using the existing `evaluate.paired_difference`. A model "beats" another only if the 95% interval of the difference excludes 0.

## 6. Reported strata (all from metadata, never inputs)

- Crossing class.
- d_min band: <0.05, 0.05–0.25, 0.25–0.5 m.
- Slower chord-speed band: <1, 1–2, 2–4, ≥4 m/s.
- Angle between paths.
- Same team vs. opposing team.
- Tracking-status combination in the gap: both VISIBLE, mixed, both fully ESTIMATED.
- Near-coincidence (d_min < 0.05 m) yes/no.
- Jump-transfer-risk yes/no.
- Overlap-group size: 1, 2–3, ≥4.
- Each F1–F4 cohort state.
- Per match.

## 7. Planned analyses

1. **Learning curves.** The best tabular and best sequence models on 4, 8, 16, 24 and 28 training matches (nested inside the folds), to test the sample-size claim in the dataset design §10.
2. **Status-texture ablation** (dataset design §7). Re-evaluate the best two models with:
   - (a) all tracks lightly smoothed (Savitzky–Golay, 5 points);
   - (b) Gaussian noise σ = 5 cm added to all tracks.

   A large drop on mixed-status samples and none on uniform ones would show a status shortcut.
3. **Gap-length transfer.** Train on 0.8 s and test on 0.4 s and 1.6 s. Also train on each gap separately.
4. **Shift robustness.** The centred model evaluated on the shifted-centre table.
5. **Dependency sensitivity.** Unweighted vs. 1/overlap-group weighting (dataset design §8).
6. **Error audit.** 50 highest-confidence errors per best model, rendered with the existing trajectory explorer. Reviewed visually, not used for tuning.

## 8. Held-out protocol (one time only)

1. All of the following are frozen on train/CV evidence and written to `final_protocol.json` with hashes:
   - the final model or models (at most 2: best tabular, best sequence);
   - the threshold and abstention rule;
   - the calibrator;
   - the evaluation script.
2. One command builds or opens the held-out shards (8 matches), retrains the frozen models on all 32 train/CV matches, and evaluates once. Results are written to a new versioned directory.
3. Held-out results are reported as they are. They are never used to change a model, threshold or feature. A second run would need a new, approved split.

## 9. Compute budget

- **Tabular models.** All tabular models × 8 folds × 3 seeds on ~40k rows: minutes on 8 CPUs, with SVM via Nyström.
- **Sequence models.** GRU/LSTM on CUDA (available, torch 2.9.1): about 1–3 min per fold.
- **Total.** Well under 2 hours. Every run is resumable per fold, with outputs in versioned directories.

## 10. Decisions needed before implementation

The full list, combined with the dataset-design decisions, is in `docs/research/m11_stage2_results_v1.md` §Approvals.
