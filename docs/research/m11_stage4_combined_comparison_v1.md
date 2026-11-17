# M11 Stage 4 — combined model comparison and robustness (v1)

Date: 2026-11-14. Status: **Stage 4A + 4B complete on train/CV. STOPPED before Stage 5.** No held-out match was loaded,
trained on, used for selection or tuning, or evaluated. Detail reports:
[4A baselines and traditional ML](m11_stage4a_baselines_and_ml_results_v1.md) and
[4B neural networks](m11_stage4b_neural_network_results_v1.md).

**Headline (development out-of-fold, independent subset):**

- **gru_gaplast** has BA **0.902 ± 0.005** over 3 seeds; seed 0 BA is 0.900, 95% match-bootstrap CI [0.896, 0.905]. Its log
  loss is 0.228 and ECE 0.008.
- **hgb** has BA 0.870, CI [0.863, 0.875], with log loss 0.285.
- The best physics model reaches BA 0.696.
- The GRU's lead over HGB (+0.031 BA) is statistically clear on the 0.8 s table. It disappears under centimetre-scale input
  smoothing or noise and at 1.6 s gaps, and it is smallest on VISIBLE-only encounters. See §11–§12.

## 1. Dataset, split and folds

| Item | Value |
|---|---|
| Source | `data/processed/m11-stage3-samples-v1` (manifest sha `6016fa6c…`); Stage 2 `m11-stage2-candidates-v1` (manifest sha `b6ab0355…`) |
| Split | frozen `m11-stage2-split-v1/split.json`, sha `66b2321a58f3845194757f943c2d36b92f18695322fb5490917f8a0144ff56c0` |
| Matches used | 32 train/CV. Held out and never loaded: 3822 3824 3834 3839 3844 10510 10514 10517 |
| Primary table | `g0p80` (0.8 s gap): 25,331 samples; **18,777 independent** (one per overlap group); SWAPPED share ≈ 0.50 |
| Other tables | `g0p40` 25,668 (19,342 indep.), `g1p60` 24,661 (17,759 indep.), `g0p80_shift` 25,172. Used only in robustness analyses, never pooled |
| Outer folds (8 × 4 matches, salt `m11-stage3-folds-v1`, `folds.json` sha `d9c07ea4…`) | f0: 3815 3826 3835 3852 · f1: 10507 3819 3838 3859 · f2: 3821 3829 3831 3847 · f3: 3816 3818 3823 3830 · f4: 3812 3833 3843 3855 · f5: 10503 10512 3814 3840 · f6: 10506 3828 3836 3837 · f7: 10502 3849 3851 3858 |
| Inner selection | tabular: 3 match-grouped inner folds per outer fold (inner log loss); neural: 3 early-stop matches per outer fold |
| Primary metric | balanced accuracy and log loss on the independent subset; positive class = SWAPPED |

## 2. Integrity and leakage audit

All gates pass (`gates/gates.json`, sha `10b69bd0…`). Details are in 4A §3. In brief:

- Manifests verified and the raw cache unchanged.
- Split approved and folds disjoint.
- The loader refuses all 8 held-out matches.
- Labels satisfy y = in XOR out order on every row, with balance across strata.
- Inputs are restricted to 7 fields, with 0 forbidden feature names.
- Swap symmetry is exact.
- An independent raw-cache identity audit confirmed 160/160 labels (RMS ≤ 1.7e-7 m).
- LOWER candidates were reconciled.

Controls pass (`gates/controls.json`, sha `b547bce2…`):

- A 20-permutation within-match null gives BA mean 0.5097 (SE 0.0038, max 0.544), vs 0.8696 with true labels.
- Mask-only, metadata-only and context-only controls give 0.4945, 0.4982 and 0.5017.
- The order-bit positive control gives 1.0.

**Corrections, documented and kept on disk:**

1. The g5 synthetic check was ill-posed: coincident meeting point and tied costs (`gates_FAILED_1794617435.json`). It was
   redesigned with parallel lanes.
2. The single-permutation control (0.524) could not test "at chance" because single-draw spread is 0.48–0.55
   (`controls_FAILED_1794617606.json`). It was replaced by a 20-permutation null.
3. The physics v1 calibration clipped the weight at 0, collapsing anti-informative rules to 0.5. That version is superseded
   (`preds/g0p80/B*_`, not evaluated) and replaced by fixed-sign w ≥ 1e-4 plus learned-sign B3 models.
4. The gap-last RNN variants were added after the original GRU/LSTM underfit. The originals stay in the comparison.

## 3. Model configurations

| Family | Model | Configuration |
|---|---|---|
| Physics | B0–B2 (6 rules) | cost difference d of the rule; P = σ(−w·d/(0.25+gap)); w ≥ 1e-4 fitted on training folds |
| Physics | B3_best_single / B3_combo | learned sign: best single rule by inner LL / symmetric logistic on all 8 margins |
| Tabular | lr, knn, svm (Nystroem RBF + LinearSVC + Platt), rf (300 trees), hgb | 36 features, antisymmetric augmentation, symmetrised prediction; grids ≤ 6 configs (4A §5) |
| Neural | mlp, gru, lstm, gru_gaplast, lstm_gaplast | per-tracklet encoder + antisymmetric pair head; about 42–56k parameters; AdamW 1e-3, ≤ 80 epochs, patience 10 (4B §2) |

Seeds: 3 for the stochastic models (svm, rf, hgb, mlp, gru_gaplast, lstm_gaplast). lr and knn are deterministic (1 seed).
The original gru and lstm have **1 seed only (INCOMPLETE seed coverage)**: they are dominated by their gap-last variants and
the GPU budget went to the leading models. HGB's 3 seeds are bit-identical: with no subsampling, its randomness is unused.

## 4. Physics baselines

| Model | Indep. BA [95% CI] | LL |
|---|---|---|
| phys_B3_combo | 0.6958 [0.688, 0.703] | 0.5685 |
| phys_B3_best_single (nearest, learned negative sign) | 0.6462 [0.638, 0.654] | 0.5972 |
| phys_B1_cv_fwd | 0.6108 [0.601, 0.621] | 0.6861 |
| phys_B2_ca_bidir | 0.5967 [0.587, 0.606] | 0.6909 |
| phys_B1_cv_bidir / cv1_bidir / ca2_bidir | 0.525 / 0.493 / 0.506 | 0.6932 |
| phys_B0_nearest (fixed physical sign) | 0.3538 [0.346, 0.362] | 0.6932 |

The nearest-endpoint heuristic is anti-informative (raw sign accuracy 0.356). The likely reason is that the gap is centred
on closest approach, so players have often passed each other by the gap edges. Mid-gap bidirectional extrapolation is
nearly degenerate.

## 5. Traditional ML

| Model | Seeds | Indep. BA | 95% CI | LL | AUC | ECE |
|---|---|---|---|---|---|---|
| hgb | 3 (identical) | 0.8695 | [0.863, 0.875] | 0.2847 | 0.950 | 0.010 |
| svm | 3 | 0.8587 ± 0.0002 | [0.853, 0.864] | 0.3067 | 0.941 | 0.009 |
| rf | 3 | 0.8574 ± 0.0002 | [0.852, 0.863] | 0.3309 | 0.942 | 0.062 |
| lr | 1 | 0.8373 | [0.830, 0.844] | 0.3437 | 0.925 | 0.010 |
| knn | 1 | 0.8224 | [0.815, 0.830] | 0.3729 | 0.915 | 0.028 |

## 6. Neural networks

| Model | Seeds | Indep. BA | 95% CI (seed 0) | LL | AUC | ECE |
|---|---|---|---|---|---|---|
| gru_gaplast | 3 | 0.9021 ± 0.0045 | [0.896, 0.905] | 0.2276 ± 0.0091 | 0.969 | 0.008 |
| lstm_gaplast | 3 | 0.8570 ± 0.0014 | [0.854, 0.863] | 0.3106 ± 0.0026 | 0.940 | 0.008 |
| mlp | 3 | 0.8447 ± 0.0020 | [0.837, 0.848] | 0.3351 ± 0.0029 | 0.930 | 0.011 |
| gru (original) | 1 (INCOMPLETE) | 0.7647 | [0.746, 0.784] | 0.4496 | 0.866 | 0.020 |
| lstm (original) | 1 (INCOMPLETE) | 0.7366 | [0.730, 0.744] | 0.4852 | 0.839 | 0.020 |

Paired differences (seed 0, match bootstrap):

- gru_gaplast − hgb: ΔBA **+0.0308 [+0.0251, +0.0364]**, ΔLL −0.0540 [−0.0611, −0.0467].
- lstm_gaplast − hgb: ΔBA −0.0112 [−0.0163, −0.0060].
- mlp − hgb: ΔBA −0.0271 [−0.0325, −0.0217].

## 7. Training versus validation (seed-0 fold means; LL)

| Model | Train LL | Validation LL | Reading |
|---|---|---|---|
| gru_gaplast | 0.202 | 0.231 | small gap; best epoch 76–80 of 80 in all 24 fold fits, so the **epoch cap binds (mild under-training)** |
| lstm_gaplast | 0.276 | 0.307 | small gap; cap often binding |
| mlp | 0.310 | 0.338 | mild overfit, stopped by early stopping |
| hgb | 0.212 | 0.287 | moderate gap |
| rf | 0.195 | 0.333 | largest gap; overconfident (ECE 0.062) |
| svm / lr / knn | 0.298 / 0.343 / 0.347 | 0.308 / 0.344 / 0.373 | little gap; capacity-limited |
| gru / lstm (original) | 0.446 / 0.488 | 0.449 / 0.486 | **underfit** |

## 8. Learning curves

Protocol (`robust/learning_curve/`, sha `4904009e…`):

- For each outer fold, the first n of its 28 training matches are taken in a seeded order (`m11-stage3-folds-v1|lc|<fold>`).
- Evaluation always uses the full outer validation fold (independent subset), with seed 0.
- HGB uses the configuration selected for that fold in the main run.
- The GRU holds out max(1, n/8) of the n matches for early stopping; the main run (n = 28) holds out 3.

| Training matches n | hgb train LL | hgb BA / LL | gru_gaplast train LL | gru_gaplast BA / LL |
|---|---|---|---|---|
| 4 | 0.065 | 0.853 / 0.322 | 0.460 | 0.742 / 0.480 |
| 8 | 0.124 | 0.859 / 0.300 | 0.405 | 0.782 / 0.427 |
| 16 | 0.178 | 0.865 / 0.289 | 0.292 | 0.852 / 0.327 |
| 24 | 0.204 | 0.867 / 0.286 | 0.235 | 0.883 / 0.267 |
| 28 (main run) | 0.212 | 0.870 / 0.285 | 0.202 | 0.900 / 0.231 |

Findings:

- **HGB is close to saturation.** BA rises by only 0.016 from 4 to 28 matches and by 0.003 from 24 to 28. Its train/validation
  gap shrinks with more data: it overfits strongly at n = 4 (train LL 0.065) and converges by n = 24.
- **The GRU is data-hungry and still improving.** Its BA rises by 0.158 from 4 to 28 matches and by 0.017 from 24 to 28.
  Train and validation LL fall together, so the GRU is in an underfitting / data-limited regime, not an overfitting one.
- The GRU overtakes HGB between 16 and 24 matches. More training matches would likely widen its lead, but that is an
  extrapolation and is not measured.

## 9. Independent versus overlapping samples

| Model | Indep. BA (n = 18,777) | All-samples BA (n = 25,331) |
|---|---|---|
| gru_gaplast | 0.9021 | 0.9019 |
| hgb | 0.8695 | 0.8667 |
| phys_B3_combo | 0.6958 | 0.6875 |

BA by overlap-group size on the independent subset, seed 0:

| Model | Group size 1 | 2–3 | ≥ 4 |
|---|---|---|---|
| hgb | 0.871 | 0.870 | 0.857 |
| gru_gaplast | 0.899 | 0.905 | 0.903 |

**Dependency weighting** (1/overlap-group-size sample weights; seed 0; `robust/dependency/dependency.json`):

| Model | Weighted BA / LL | Unweighted BA / LL | ΔBA [95% CI] | ΔLL [95% CI] |
|---|---|---|---|---|
| hgb | 0.8662 / 0.2861 | 0.8695 / 0.2847 | −0.0033 [−0.0059, −0.0010] | +0.0014 [−0.0003, +0.0030] |
| gru_gaplast | 0.8840 / 0.2659 | 0.9003 / 0.2307 | −0.0163 [−0.0217, −0.0112] | +0.0352 [+0.0257, +0.0448] |

Down-weighting overlapping encounters does not help either model; it reduces the effective training size. There is no
evidence that the overlap structure inflates the independent-subset scores.

## 10. Crossing and difficulty strata (independent subset, seed 0; BA with match-bootstrap 95% CI)

| Stratum | n | phys_B3_combo | hgb | gru_gaplast |
|---|---|---|---|---|
| local crossing | 12,193 | 0.831 | 0.889 [0.881, 0.896] | 0.914 [0.909, 0.920] |
| window crossing only | 3,473 | 0.540 | 0.856 [0.846, 0.866] | 0.889 [0.881, 0.898] |
| no crossing | 3,111 | 0.342 | 0.809 [0.795, 0.824] | 0.858 [0.848, 0.869] |
| d_min < 0.05 m (near-coincident) | 2,129 | 0.797 | 0.850 [0.830, 0.868] | 0.889 [0.876, 0.903] |
| d_min 0.05–0.25 m | 6,910 | 0.750 | 0.885 [0.877, 0.893] | 0.917 [0.911, 0.924] |
| d_min 0.25–0.5 m | 9,738 | 0.635 | 0.863 [0.857, 0.869] | 0.891 [0.886, 0.896] |
| speed < 1 m/s | 8,111 | 0.608 | 0.875 | 0.905 |
| speed ≥ 4 m/s | 1,078 | 0.797 | 0.885 | 0.899 |
| crossing angle < 30° | 2,377 | 0.688 | 0.841 [0.821, 0.860] | 0.862 [0.848, 0.874] |
| crossing angle 60–120° | 6,813 | 0.765 | 0.889 | 0.916 |
| no linked v1 event | 2,142 | 0.318 | 0.798 [0.780, 0.814] | 0.852 [0.838, 0.866] |
| same team | 6,107 | 0.692 | 0.880 | 0.914 |
| opposite teams | 12,670 | 0.698 | 0.864 | 0.894 |
| jump/transfer risk | 311 | 0.592 | 0.823 [0.783, 0.868] | 0.855 [0.819, 0.889] |

Findings:

- The hardest cases for the learned models are no-crossing encounters, shallow crossing angles (< 30°) and encounters
  without a linked v1 event.
- Physics fails badly exactly where there is no geometric crossing.
- The GRU's advantage holds in every stratum.

Team relation is metadata only and is not a model input.

## 11. Tracking-status robustness

BA by tracking status inside the gap (independent subset, seed 0):

| Status in gap | n | phys_B3_combo | hgb | gru_gaplast |
|---|---|---|---|---|
| both fully ESTIMATED | 9,730 | 0.662 | 0.875 [0.867, 0.883] | 0.912 [0.906, 0.917] |
| mixed | 7,166 | 0.749 | 0.881 [0.873, 0.889] | 0.908 [0.901, 0.915] |
| both fully VISIBLE | 1,881 | 0.669 | **0.795** [0.778, 0.812] | **0.812** [0.796, 0.829] |
| estimated-heavy context (≥ 0.5) | 13,458 | 0.690 | 0.876 | 0.913 |
| visible-heavy context (< 0.5) | 5,319 | 0.710 | 0.852 | 0.868 |

Positional texture: the median second difference per frame is 0.0885 m for ESTIMATED, 0.0357 m for mixed and 0.0216 m for
VISIBLE. ESTIMATED tracks are about 4× rougher, not smoother.

**Texture ablation** (`robust/texture/`). Transforms are applied identically to all four tracklets; masks, labels and groups
are unchanged.

- *smooth5*: centred 5-point mask-aware average.
- *noise5cm*: seeded Gaussian noise, σ = 5 cm.

| Variant | Model, protocol | All BA | ESTIMATED | mixed | VISIBLE |
|---|---|---|---|---|---|
| original | hgb | 0.870 | 0.875 | 0.881 | 0.795 |
| original | gru_gaplast | 0.900 | 0.912 | 0.908 | 0.812 |
| smooth5 | hgb, inference only | 0.864 | 0.871 | 0.875 | 0.784 |
| smooth5 | hgb, **retrained** | 0.869 | 0.877 | 0.879 | 0.784 |
| smooth5 | gru_gaplast, inference only | 0.836 | 0.820 | 0.873 | 0.783 |
| smooth5 | gru_gaplast, **retrained** | 0.853 | 0.850 | 0.881 | 0.767 |
| noise5cm | hgb, inference only | 0.851 | 0.845 | 0.875 | 0.788 |
| noise5cm | hgb, **retrained** | 0.853 | 0.850 | 0.874 | 0.785 |
| noise5cm | gru_gaplast, inference only | 0.786 | 0.748 | 0.849 | 0.743 |
| noise5cm | gru_gaplast, **retrained** | 0.850 | 0.846 | 0.882 | 0.753 |

Measured findings:

- After retraining on either transform, the GRU no longer beats HGB: smooth 0.853 vs 0.869; noise 0.850 vs 0.853.
- HGB barely changes under smoothing (−0.001) and loses 0.017 under noise.
- The GRU's original +0.031 lead therefore relies on centimetre-scale positional structure, which both transforms destroy.
- Under both conditions, the learned models do much worse on fully VISIBLE gaps than on ESTIMATED or mixed gaps. Physics
  does not show this pattern (0.669 VISIBLE vs 0.662 ESTIMATED).

Hypothesis (not established): ESTIMATED positions are produced by the provider, possibly using identity-aware
interpolation through real occlusions. If so, they may carry a continuity pattern that learned models exploit and that real
gaps will not have. If true, the fully VISIBLE stratum (BA 0.81 GRU / 0.80 HGB) is the conservative estimate for
deployment-like conditions. It is an equally valid reading that the fine structure is genuine motion signal. The ablations
alone cannot separate the two, and a performance drop is not taken as proof of shortcut learning.

## 12. Gap-length and shift robustness

**Gap transfer** (`robust/gap_transfer/`). Transfer means the 0.8 s fold models predicting the same outer validation matches
of another gap table. Retrained means the same model family trained with the same folds on that table (seed 0). The tables
share encounters and are never pooled.

| Table | Model | Transfer BA / LL | Retrained BA / LL | Retrained − transfer ΔBA [95% CI] |
|---|---|---|---|---|
| 0.4 s | hgb | 0.910 / 0.228 | 0.949 / 0.130 | +0.039 [0.035, 0.043] |
| 0.4 s | gru_gaplast | 0.801 / 0.477 | 0.963 / 0.100 | +0.161 [0.149, 0.172] |
| 1.6 s | hgb | 0.799 / 0.469 | 0.822 / 0.383 | +0.023 [0.018, 0.027] |
| 1.6 s | gru_gaplast | 0.767 / 0.815 | 0.815 / 0.406 | +0.048 [0.041, 0.054] |

Retrained GRU − retrained HGB (post-hoc paired bootstrap, `paired_gru_vs_hgb_retrained_by_gap.json`):

| Gap | ΔBA [95% CI] | ΔLL [95% CI] |
|---|---|---|
| 0.4 s | +0.0144 [+0.0114, +0.0176] | −0.0298 [−0.0366, −0.0227] |
| 0.8 s | +0.0308 [+0.0251, +0.0364] | −0.0540 [−0.0611, −0.0467] |
| 1.6 s | **−0.0074 [−0.0146, −0.0001]** | +0.0232 [+0.0125, +0.0333] |

Findings:

- Difficulty rises steeply with gap length.
- Neither model transfers across gap lengths without retraining. The GRU transfers far worse than HGB, which uses explicit
  gap-normalised features.
- At 1.6 s, HGB is marginally better than the GRU.

**Shift robustness** (`robust/shift/`). Centred-gap fold models predict gaps whose centre is offset from closest approach.

| Shift (s) | −0.3 | −0.2 | −0.1 | 0 | +0.1 | +0.2 | +0.3 |
|---|---|---|---|---|---|---|---|
| hgb shifted BA | 0.735 | 0.840 | 0.874 | 0.864 | 0.874 | 0.821 | 0.714 |
| hgb centred BA (same candidates) | 0.864 | 0.869 | 0.882 | 0.863 | 0.871 | 0.863 | 0.872 |
| gru_gaplast shifted BA | 0.713 | 0.823 | 0.856 | 0.899 | 0.828 | 0.825 | 0.810 |
| gru_gaplast centred BA | 0.893 | 0.899 | 0.904 | 0.899 | 0.913 | 0.896 | 0.898 |

Overall shifted vs centred BA: hgb 0.818 vs 0.869; GRU 0.822 vs 0.900.

Both models have learned that the gap sits on closest approach, and both lose 0.09–0.18 BA at ±0.3 s. HGB is insensitive to
±0.1 s shifts; the GRU loses 0.05–0.09. A real tracking failure is not centred on closest approach, so this is a deployment
risk. A shift-augmented training set is the obvious remedy, but it was not tested because it would change the approved
training design.

## 13. Calibration and abstention

Calibration (ECE, 15 bins, independent subset): gru_gaplast 0.008, hgb 0.010, phys_B3_combo 0.024. All are well
calibrated except rf (0.062).

Nested abstention (`robust/abstention/`): for each outer fold, the confidence threshold is chosen on the out-of-fold
predictions of the other 7 folds, then applied to that fold. Values are pooled realised error / coverage.

| Target error | hgb coverage (error) | gru_gaplast coverage (error) | phys_B3_combo coverage (error) |
|---|---|---|---|
| 1% | 0.482 (0.010) | 0.579 (0.010) | 0.049 (0.014) |
| 2% | 0.582 (0.020) | 0.683 (0.020) | 0.065 (0.021) |
| 5% | 0.738 (0.050) | **0.849 (0.050)** | 0.131 (0.051) |
| 10% | 0.913 (0.100) | 0.999 (0.100) | 0.230 (0.100) |
| AURC | 0.031 | 0.019 | 0.181 |

The nested thresholds are stable across folds (GRU 5% thresholds 0.710–0.725) and realise the target error almost exactly.
Abstaining on the least confident 15% of encounters halves the GRU's error from 10% to 5%. Recovered share (correct and
accepted) at 5% is 0.806 for the GRU and 0.701 for HGB.

## 14. High-confidence error analysis (`robust/errors/`)

High-confidence wrong means confidence ≥ 0.9 and wrong, on the independent subset, seed 0.

- **GRU:** 254 of 1,872 errors (13.6%).
- **HGB:** 176 of 2,451 errors (7.2%).
- 45 encounters are high-confidence errors for both models.
- **Match concentration:** none. The worst match has 23 per 1,000 (GRU) or 19 per 1,000 (HGB) high-confidence errors,
  against medians of 13 and 9.

How the high-confidence errors differ from the whole independent subset:

| Feature | All independent | GRU HC-wrong | HGB HC-wrong |
|---|---|---|---|
| no crossing | 16.6% | **35.0%** | **35.2%** |
| local crossing | 64.9% | 45.7% | 39.2% |
| d_min 0.25–0.5 m | 51.9% | 63.8% | 61.4% |
| both fully VISIBLE | 10.0% | **22.1%** | 13.1% |
| median separation at gap start (m) | 0.89 | 0.83 | 0.88 |
| speed < 1 m/s | 43.2% | 41.3% | 46.6% |

Recurring patterns:

- **Non-crossing, larger-separation encounters.** Two players come within 0.25–0.5 m without the paths crossing; the label
  then depends on subtle direction changes.
- **Abrupt direction changes inside or just after the gap.**
- **Near-stationary players with jittery tracks.**
- **Sparse context.** About 7.5% of samples have fewer than 30 valid points.

Visual inspection of the 12 highest-confidence HGB errors (`high_confidence_errors_hgb.png`): several show a reference
continuation that needs a sharp turn or a jump right after the gap, while the alternative pairing would continue smoothly.
This pattern fits either genuine sudden manoeuvres or **provider reference-identity errors**. It was not verified against
video, and no sample was relabelled or excluded.

Error rates are lowest for local crossings, which were the selection target. The selection toward clean continuous
encounters means the hardest real cases (long occlusions, crowds) are under-represented.

## 15. Runtime, CPU/GPU and memory

| Phase | Wall time | Device |
|---|---|---|
| Gates + controls | about 2 + 7 min | CPU |
| Physics | 4 s | CPU |
| Tabular (all 5 models, 3 seeds where stochastic) | 3,918 s | CPU (n_jobs = −1 for rf) |
| Neural seed 0 (5 kinds) | 1,599 + 1,970 s | CUDA GPU (PyTorch 2.9.1) |
| Neural seeds 1–2 (gru_gaplast, mlp, lstm_gaplast) | 7,550 s (partly concurrent) | GPU |
| Evaluation v1 / v2 | 1,713 / 1,974 s | CPU (bootstraps) |
| Robustness: texture / shift / gap transfer / dependency / texture_nn / learning curve | 247 / 17 / 5,383 / 3,240 / 3,966 / 1,666 s | GPU + CPU, with 2–3 jobs concurrent |

- Python 3.12.10.
- GPU memory in use was about 2 GB per neural job (nvidia-smi snapshot).
- **Peak RAM and peak GPU memory were not instrumented (NOT MEASURED).**
- Concurrent GPU jobs roughly doubled per-seed wall time; results are deterministic per seed and unaffected.
- Disk: 1.1 GB in total (predictions and fold models 999 MB, of which rf joblib is 614 MB).

## 16. Output paths and hashes

Root: `data/processed/m11-stage4-benchmarks-v1/` (written once; a re-run skips finished units). Hashes are SHA-256 prefixes.

| File | sha256 (16) |
|---|---|
| `metrics/g0p80/v2/comparison.json` (final) | `32f34a0521f06298` |
| `metrics/g0p80/v1/comparison.json` (interim, partial neural seeds) | `526cb48bb7a96c08` |
| `folds.json` | `d9c07ea4f449e590` |
| `gates/gates.json` / `gates/controls.json` | `10b69bd00fe69a89` / `b547bce2899e6dcf` |
| `robust/selected_models.json` (hgb, gru_gaplast, phys_B3_combo; from v1; identical choice under v2) | `805e2fe6215d5c6e` |
| `robust/abstention/abstention.json` | `0e6b897b9831ba9f` |
| `robust/errors/errors.json` | `1d8ce177d6e4ba6f` |
| `robust/texture/texture.json` / `texture_nn_retrained.json` | `093d4bdb514f3d63` / `b17a0e7df084b703` |
| `robust/shift/shift.json` | `4784f2fe9defe290` |
| `robust/gap_transfer/gap_transfer.json` / `paired_gru_vs_hgb_retrained_by_gap.json` | `afa8f06cf235d32f` / `659ea238ec084228` |
| `robust/dependency/dependency.json` | `1239f677cc3873c3` |
| `robust/learning_curve/learning_curve.json` | `4904009e7f5968dc` |
| `features/g0p80.npz` | `f20ae18756db7802` |
| `preds/g0p80/{hgb,gru_gaplast,phys_B3_combo}/seed0.npz` | `d1697e11caf487e0` / `940627bb62868f07` / `c2117b0be54e07e0` |

Code: `src/football_fatigue/m11_benchmark/` (`data`, `folds`, `metrics`, `models`, `gates`, `run`, `evaluate`, `robust`).
Tests: `tests/test_m11_benchmark.py` (20 tests). Full suite run on 2026-11-14: **1024 passed, 6 skipped**.

Run order:

```
python -m football_fatigue.m11_benchmark gates
python -m football_fatigue.m11_benchmark controls
python -m football_fatigue.m11_benchmark physics
python -m football_fatigue.m11_benchmark tabular
python -m football_fatigue.m11_benchmark neural --models … --seeds …
python -m football_fatigue.m11_benchmark evaluate --version v2 --models …
python -m football_fatigue.m11_benchmark robust --analyses …
```

## 17. Remaining scientific limitations

1. **Synthetic gaps.** Gaps are cut out of continuous provider tracks and centred on closest approach. Real failures are not
   centred on closest approach (shift drops of up to 0.18 BA) and vary in length (the retrained 1.6 s models lose 0.05–0.09 BA vs 0.8 s).
2. **Provider ESTIMATED positions.** About 52% of independent samples are fully ESTIMATED inside the gap. Learned models
   score 0.08–0.10 BA higher there than on fully VISIBLE gaps, while physics does not. The GRU's edge depends on fine
   positional structure. Whether that structure is real motion or an artefact of the provider's estimation is unresolved.
3. **Reference identities are the provider's.** The independent audit confirms that labels reproduce the raw cache, not that
   the provider's identities are correct. Some high-confidence errors look like possible reference errors (unverified).
4. **Selection bias** toward clean, continuous two-player encounters (Stage 2 acceptance). Crowded multi-player occlusions are
   out of scope.
5. **Dependence.** Overlapping encounters exist. Headline numbers use the independent subset; weighting did not change the
   conclusions.
6. **Development-only estimates.** All numbers are out-of-fold on 32 matches. The ranking was chosen on these same
   out-of-fold numbers (v1 selection = v2), so the leading model's score is mildly optimistic. Stage 5 held-out evaluation is
   needed.
7. **Under-training and seed coverage.** The GRU epoch cap binds. HGB seeds carry no variance. The original gru/lstm have
   1 seed.
8. **Not measured:** peak memory.

## 18. Best-performing models (development out-of-fold, independent subset)

| Rank | Model | BA (3-seed mean ± sd) | 95% CI (seed 0) | LL | AUC | ECE | Coverage at 5% error |
|---|---|---|---|---|---|---|---|
| 1 | **gru_gaplast** | **0.9021 ± 0.0045** | [0.896, 0.905] | 0.2276 ± 0.0091 | 0.969 | 0.008 | 0.849 |
| 2 | hgb | 0.8695 ± 0.0000 | [0.863, 0.875] | 0.2847 | 0.950 | 0.010 | 0.738 |
| 3 | svm | 0.8587 ± 0.0002 | [0.853, 0.864] | 0.3067 | 0.941 | 0.009 | – |
| – | best physics (phys_B3_combo) | 0.6958 | [0.688, 0.703] | 0.5685 | 0.768 | 0.024 | 0.131 |

The GRU is best at 0.4 s and 0.8 s gaps. At 1.6 s, and under texture removal, it is no better than HGB. HGB is the more
robust second model (it transfers better across gaps and is less sensitive to ±0.1 s shifts and texture).

## 19. Recommendations for the Stage 5 held-out protocol (requires approval)

1. **Freeze before access.** Before the 8 held-out matches are built, record the following:
   - Final candidates: gru_gaplast and hgb, both trained on all 32 train/CV matches. phys_B3_combo is the physics reference.
   - Seeds: gru_gaplast seeds 0–2 (report mean and per seed); HGB 1 run.
   - Hyperparameters: HGB as selected by inner CV on all 32 matches; GRU unchanged, 80 epochs, 3 early-stop matches.
   - Abstention thresholds from the development out-of-fold predictions: 5% target, GRU 0.7175 / HGB 0.765.
   - Freeze the code commit and the evaluation script.
2. **Build held-out shards once** with the unchanged Stage 3 builder and an approved one-time override. Evaluate once:
   no re-tuning, no threshold changes, no model swaps after seeing results.
3. **Primary endpoint:** independent-subset BA and log loss at 0.8 s, with match-level bootstrap CIs (8 matches; expect wide
   intervals), plus the paired GRU − HGB difference.
4. **Mandatory secondary reports:**
   - Results by tracking status, with fully VISIBLE as the conservative estimate.
   - Results by crossing class.
   - Results at 0.4 and 1.6 s with models retrained on the corresponding development tables, plus the shifted table.
   - Coverage and error at the frozen thresholds.
5. **Pre-declared success criterion** (suggested): GRU held-out BA ≥ 0.85 with the CI excluding the physics baseline. Also
   report whether the GRU − HGB difference replicates. If it does not, prefer HGB for robustness.
6. Do not deploy on real tracking failures without a shift-augmented and gap-length-diverse model. The development evidence
   shows strong sensitivity to both.
