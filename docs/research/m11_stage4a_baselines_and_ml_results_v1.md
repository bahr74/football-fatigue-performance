# M11 Stage 4A — physics baselines and traditional ML (v1)

Date: 2026-11-14. Scope: the 32 train/CV matches of the frozen Stage 2 split only. No held-out match was loaded, trained on,
tuned on or evaluated. Stage 5 has not started.

Companion reports: [Stage 4B neural networks](m11_stage4b_neural_network_results_v1.md),
[Stage 4 combined comparison](m11_stage4_combined_comparison_v1.md).

## 1. Task and data

- Task: given two incoming tracklets (before a hidden interval) and two outgoing tracklets (after it), predict
  P(SWAPPED) = the probability that the outgoing order is crossed relative to the incoming order. Positive class = SWAPPED (y = 1).
- Primary table: `m11-stage3-samples-v1`, gap 0.8 s (`g0p80`): 25,331 samples, 18,777 in the **independent subset**
  (one per overlap group). All headline numbers below are on the independent subset. Numbers on all samples are in the
  metric files.
- Inputs: only the 7 Stage 3 input fields (`in_pos`, `in_mask`, `in_dt`, `out_pos`, `out_mask`, `out_dt`, `gap_s`), read through
  `load_model_inputs`. Slot, player, jersey, team, tracking status, crossing class, cohort and order bits are never model inputs.
  This was checked by gate g5 (forbidden-word scan of all 36 feature names).
- Folds: 8 outer folds × 4 matches, grouped by match (`FOLD_SALT = "m11-stage3-folds-v1"`, `folds.json`). Tabular
  hyperparameters were chosen per outer fold by 3 inner match-grouped folds, using inner log loss. Every prediction is out-of-fold
  at match level.

## 2. Features (36, deterministic, `features/g0p80.npz`)

- 25 `perm_*` features are antisymmetric under exchanging the outgoing order: the swap is exact (gate g6, max error 0.0).
  They are: pairing-cost differences for 8 motion rules, in raw and log form (nearest endpoint; constant velocity
  forward/bidirectional; constant acceleration bidirectional; one-point-velocity and two-point-acceleration variants), plus
  velocity, speed, heading, acceleration and implied-speed differences, mid-gap CV error, and CA/CV forward/backward costs.
- 11 symmetric context features: gap length, separation at the gap edges, min/max speeds before and after, crossing angles
  before and after, and CV-bidirectional min pair cost and absolute margin.
- Tabular models are trained on the data plus its swapped copy (labels flipped). Predictions are symmetrised:
  p(x) = ½[p(x) + 1 − p(swap x)], so swapping the outgoing order gives exactly 1 − p (unit tests, all five models).

## 3. Integrity gates and controls (all PASS)

| Gate | Result |
|---|---|
| g1 manifests | Stage 2 (410 files) and Stage 3 (356 files) manifests verified; raw cache unchanged |
| g2 split | frozen split sha `66b2321a…c56c0` approved; folds disjoint, 8 × 4 matches |
| g3 loader | all 8 held-out matches refused (inputs and metadata) |
| g4 labels/order | y = in_order XOR out_order on every row; SWAPPED share balanced across crossing, distance, speed and frame-mode strata (chi² p ≥ 0.4) |
| g5 inputs | only the 7 input fields; NaN masks and sample times correct; 0 forbidden feature names |
| g6 symmetry | exact swap equivariance (400 rows, max error 0.0) |
| g7 identity audit | 160/160 labels independently re-derived from the raw cache by rigid (Kabsch) alignment, RMS ≤ 1.7e-7 m, 0 ambiguous |
| g9 LOWER reconciliation | 25 train/CV LOWER candidates = union over the three gaps (20/17/13 per gap); none built as a sample |

Controls (HGB on the same folds; `gates/controls.json`):

| Control | Indep. BA | Expected |
|---|---|---|
| Label permutation within match (20 permutations) | mean 0.5097, SE 0.0038, range 0.484–0.544 | chance |
| True labels, same model | 0.8696 | exceeds the null max by 0.326 |
| Mask pattern only | 0.4945 | chance |
| Metadata only (no order bits) | 0.4982 | chance |
| Context (symmetric) features only | 0.5017 | chance |
| Positive control: order bits | 1.000 | 1.0 (shows the leak path exists in metadata; the adapter blocks it) |

### Documented corrections (failed attempts are kept on disk)

1. **g5 gate redesign.** The first g5 run failed (`gates/gates_FAILED_1794617435.json`). Its synthetic check placed both
   players through the same meeting point, so bidirectional costs tie and margins are about 1e-16. The check could not
   discriminate (ill-posed), and the data was not at fault. It was redesigned with parallel, non-coincident lanes, then passed.
2. **Permutation-null redesign.** The first control used a single label permutation and gave BA 0.524, CI [0.518, 0.531]
   (`gates/controls_FAILED_1794617606.json`). Repeated single permutations spread from 0.479 to 0.552 (scratch probe), so one
   draw cannot test "at chance". It was replaced by a 20-permutation null, with the decision on the null mean (within 3 SE of 0.5).
   A coin-flip-label probe gave mean 0.4993 (SE 0.0032).

## 4. Physics baselines

Each rule gives a cost difference d (crossed minus straight pairing). It is mapped to P(SWAPPED) = σ(−w·d/(0.25+gap)),
with w fitted per outer fold on the training folds only.

**Correction:** the first physics version (`preds/g0p80/B*_*`, superseded, kept on disk) clipped w at 0. When a rule's sign
was anti-informative, it collapsed to a constant 0.5 (BA 0.5000). The B0–B2 rules now use w ≥ 1e-4 (the "physics rule",
fixed sign). B3 learns the sign and magnitude:

- B3_best_single: the single best rule by inner log loss.
- B3_combo: a symmetric logistic combination of all 8 margins.

| Model | Indep. BA | 95% CI (match bootstrap) | Log loss | AUC | ECE |
|---|---|---|---|---|---|
| phys_B0_nearest | 0.3538 | [0.346, 0.362] | 0.6932 | 0.281 | 0.0005 |
| phys_B1_cv_fwd | 0.6108 | [0.601, 0.621] | 0.6861 | 0.628 | 0.032 |
| phys_B1_cv_bidir | 0.5252 | [0.515, 0.536] | 0.6932 | 0.502 | 0.0005 |
| phys_B1_cv1_bidir | 0.4925 | [0.481, 0.505] | 0.6932 | 0.453 | 0.0005 |
| phys_B2_ca_bidir | 0.5967 | [0.587, 0.606] | 0.6909 | 0.609 | 0.007 |
| phys_B2_ca2_bidir | 0.5058 | [0.495, 0.517] | 0.6932 | 0.480 | 0.0005 |
| phys_B3_best_single (selected "nearest" with learned negative sign in 8/8 folds) | 0.6462 | [0.638, 0.654] | 0.5972 | 0.719 | 0.020 |
| **phys_B3_combo** | **0.6958** | [0.688, 0.703] | **0.5685** | 0.768 | 0.024 |

Measured findings:

- **Nearest-endpoint matching is anti-informative.** Its raw sign accuracy is 0.356. This is how the samples are built: the
  gap is centred on closest approach, so at the gap edges the players have often already passed each other. Learning the
  sign turns it into the best single rule (BA 0.646).
- **Bidirectional extrapolation is nearly degenerate.** Comparing mid-gap positions from both sides is almost symmetric
  when the gap is centred on closest approach (cv_bidir raw sign accuracy 0.517). The forward CV and CA rules are weakly
  informative (0.59–0.61).
- The best physics model (B3_combo) is far below every learned model (−0.14 to −0.20 BA). Simple kinematics explain only
  a small part of the association signal.

## 5. Traditional ML

Hyperparameter grids have ≤ 6 configurations per model and are selected per outer fold by inner-fold log loss:

| Model | Implementation | Grid | Selected (8 outer folds) |
|---|---|---|---|
| lr | StandardScaler + LogisticRegression | C ∈ {0.01, 0.1, 1, 10} | C=1 (7), C=10 (1) |
| knn | StandardScaler + KNN | k ∈ {15, 31, 63} | k=31 (8) |
| svm | StandardScaler + Nystroem RBF (1000 components) + LinearSVC; Platt weight from inner-OOF decisions | γ·n_features ∈ {0.5, 1, 2} × C ∈ {0.1, 1} | γ=0.5, C=1 (8) |
| rf | RandomForest, 300 trees, min_samples_leaf 5 | max_depth ∈ {None, 12} | None (8) |
| hgb | HistGradientBoosting, min_samples_leaf 40, L2 1.0, no early stopping | lr 0.05 / 400 iterations or 0.1 / 200 iterations, leaves {15, 31} | (0.05, 400, 31) ×5; (0.1, 200, 31) ×3 |

Results on the independent subset. Seeds are 0, 1 and 2 for the stochastic models; values are mean ± sd over seeds. The CI
is the seed-0 match-level bootstrap (2,000 resamples).

| Model | Seeds | Indep. BA | 95% CI | Log loss | AUC | Brier | ECE | Train LL / Val LL |
|---|---|---|---|---|---|---|---|---|
| **hgb** | 3 | **0.8695 ± 0.0000** | [0.863, 0.875] | **0.2847 ± 0.0000** | 0.950 | 0.091 | 0.010 | 0.212 / 0.287 |
| svm | 3 | 0.8587 ± 0.0002 | [0.853, 0.864] | 0.3067 ± 0.0002 | 0.941 | 0.098 | 0.009 | 0.298 / 0.308 |
| rf | 3 | 0.8574 ± 0.0002 | [0.852, 0.863] | 0.3309 ± 0.0003 | 0.942 | 0.103 | 0.062 | 0.195 / 0.333 |
| lr | 1 (deterministic) | 0.8373 | [0.830, 0.844] | 0.3437 | 0.925 | 0.111 | 0.010 | 0.343 / 0.344 |
| knn | 1 (deterministic) | 0.8224 | [0.815, 0.830] | 0.3729 | 0.915 | 0.120 | 0.028 | 0.347 / 0.373 |

Paired match-bootstrap differences against HGB (seed 0, independent subset). All are significant: the CI excludes 0.

| Model | ΔBA vs HGB [95% CI] | ΔLL vs HGB [95% CI] |
|---|---|---|
| svm | −0.0109 [−0.0147, −0.0071] | +0.0218 [+0.0182, +0.0256] |
| rf | −0.0119 [−0.0148, −0.0088] | +0.0460 [+0.0433, +0.0488] |
| lr | −0.0322 [−0.0364, −0.0278] | +0.0590 [+0.0542, +0.0635] |
| phys_B3_combo | −0.1737 [−0.1811, −0.1661] | +0.2838 [+0.2761, +0.2913] |

HGB details (seed 0):

- Precision / recall / F1 (SWAPPED): 0.868 / 0.872 / 0.870.
- Confusion matrix: TN 8149, FP 1249, FN 1202, TP 8177.
- Per-fold BA: 0.8697 ± 0.0076. Per-match BA ranges from 0.834 to 0.911 (median 0.874).

Measured findings:

- **HGB is the best tabular model** on both BA and log loss, and it is well calibrated (ECE 0.010).
- **RF is clearly overconfident** (ECE 0.062): it has the lowest train LL of the tabular models but a worse validation LL.
- **HGB seeds are identical.** With `early_stopping=False` and no row or feature subsampling, scikit-learn's HGB does not use
  its random state, so its 3 seeds carry no seed variance. The 3-seed coverage is formally complete but uninformative.
  SVM (Nystroem sampling) and RF show real but tiny seed variance (sd ≤ 0.0003).
- **Overfitting gap:** HGB train LL 0.212 vs validation LL 0.287, and RF 0.195 vs 0.333. These are moderate gaps; the
  validation metrics are what count. LR and KNN have almost no gap, which suggests they are capacity-limited.

## 6. Runtime (one consumer GPU workstation; tabular models run on the CPU)

| Model | Seed 0 (s) | Seeds 1–2 (s each) | Note |
|---|---|---|---|
| physics (all 8) | 4 | – | |
| lr | 12 | – | |
| knn | 110 | – | |
| svm | 1782 | 421–423 | seed 0 includes the inner-CV grid search (6 configs × 3 inner folds × 8 outer folds); later seeds reuse the selected config |
| rf | 594 | 145 | fold models are 614 MB of joblib |
| hgb | 225 | 30 | |

Peak RAM was not instrumented (NOT MEASURED).

## 7. Outputs

- Predictions: `data/processed/m11-stage4-benchmarks-v1/preds/g0p80/<model>/seed<k>.npz` and `.json`, plus fold models and
  per-fold selection files in `folds/`.
- Metrics: `metrics/g0p80/v1/` (interim, partial neural seeds) and `metrics/g0p80/v2/` (final; see the combined report).
- Gates and controls: `gates/`. Progress log: `PROGRESS.md`.

## 8. Limitations specific to 4A

- The engineered features are hand-designed kinematic summaries. A ceiling of about 0.87 BA may reflect the features rather
  than the information in the data, since the sequence model does better (Stage 4B).
- All results are out-of-fold on 32 development matches. Generalisation to the 8 held-out matches is untested by design.
