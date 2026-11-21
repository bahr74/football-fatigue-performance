# M11 Stage 4C: robustness improvement, controlled experiments and pre-held-out model selection (v1)

Date: 2026-11-14. Scope: the 32 train/CV matches of the frozen split `m11-stage2-split-v1` (sha `66b2321a…c56c0`), the frozen Stage 4
outer folds (8 × 4 matches) and the Stage 3 label definition. **No held-out match was read, built or evaluated; Stage 5 has not
started.** Stage 2, Stage 3 and Stage 4 outputs were only read.

All numbers are **out-of-fold development estimates** on the independent subset (18,777 centred encounters). Several Stage 4C
choices were made after seeing outer-fold results of earlier Stage 4 runs on the same folds (see §8, "Development-selection
optimism"). None of this establishes performance on unseen FIFA validation or test data.

Outputs:
- `data/processed/m11-stage4c-shiftaug-v1` (195 MB): shift-augmentation samples.
- `data/processed/m11-stage4c-robustness-v1` (218 MB): fold units, predictions, metrics, audit, runtime.

Code (new, uncommitted):
- `src/football_fatigue/m11_benchmark/shiftaug.py`
- `src/football_fatigue/m11_benchmark/stage4c.py`
- `src/football_fatigue/m11_benchmark/stage4c_eval.py`
- `tests/test_m11_stage4c.py`

## Executive summary

- **Audit.** No leakage or implementation defect in the models was found (10 / 10 checks pass, §1).
  - Stage 4B `gru_gaplast` was retrained from scratch and reproduced **bit-identically**: max |Δp| = 0.0 for all three seeds.
  - One audit check was itself wrong in its first version and was corrected (§1).
- **Texture probe (new evidence).** Tracking *texture* alone pairs tracklets far above chance:
  - It uses only second/third-difference statistics: no positions, velocities, distances or headings.
  - BA 0.701 overall; 0.739 when both players are ESTIMATED in the gap; 0.525 (≈ chance) when both are VISIBLE.
  - Dose-response with the ESTIMATED share of input points: 0.603 → 0.756.
  - This is strong evidence that ESTIMATED positions carry an identity-linked, provider-specific signal. It does not identify the
    mechanism.
- **Longer training** (best early-stopping state within 120 / 160 epochs) improves the GRU on centred gaps.
  - BA 0.902 → 0.920 (3 seeds); +0.022 [+0.017, +0.027] for seed 0.
  - Early stopping now ends most fold fits before the cap.
  - The gain appears in every tracking-status stratum, including fully VISIBLE (0.818 → 0.828).
  - It makes noise and gap-length transfer slightly worse.
- **Shift-robust training works.** Config C trains on 50 % centred and 50 % shifted gaps (±0.1–0.3 s), with a 160-epoch budget.
  - Centred BA: 0.926 ± 0.003, with no centred cost vs config A: +0.002 [−0.003, +0.006].
  - Shifted BA: 0.910 ± 0.001 vs 0.832 for A (+0.076 [+0.070, +0.081]) and 0.815 for the Stage 4B GRU.
  - Uniform-shift training (B) costs 0.04 centred BA and is dominated by C.
- **Ensembles.** The predeclared GRU/HGB ensemble, with weights chosen on each fold's training partition, selected **w = 1 (GRU only)
  in 24 / 24 folds** for config C. It is therefore not a distinct candidate.
  - A fixed 50/50 average is more robust to gap-length change and noise but poorly calibrated (ECE 0.052). It was not selected by
    the nested procedure.
- **Main weaknesses of the GRU** (unchanged by shift training):
  - Poor transfer to other gap lengths: 0.4 s BA 0.840 vs HGB 0.928; 1.6 s 0.745 vs 0.803.
  - Fragility under 5 cm positional noise: 0.757 vs 0.842.
  - Severe over-confidence in both cases: log loss 0.79–0.87 vs 0.34–0.46 for HGB.
  - Training with noise or smoothing augmentation buys noise robustness at a centred cost of 0.03–0.05 BA.
- **Recommendation: two candidates for the one-time held-out evaluation.**
  1. **GRU gap-last, config C, 160-epoch budget** (primary).
  2. **HGB with the same 50/50 shift-mixture training (HGB C)**: the simple, noise- and gap-length-robust reference.

  The proposed protocol is in [`m11_stage5_final_protocol_PROPOSED_v1.md`](m11_stage5_final_protocol_PROPOSED_v1.md). It is
  **not activated**.

## 1. Independent code and leakage audit

`python -m football_fatigue.m11_benchmark.stage4c audit` → `audit/audit_v2.json`; `all_pass = true`.

`audit/audit.json` (v1) is kept as superseded evidence. Its check 4 failed because **the check** anchored the 0.5 s velocity window
at the grid end, not at each tracklet's last valid point as `samples._slope` does. With the corrected check, the frame error is
9.5e-7.

| # | Check | Result |
|---|---|---|
| 1 | All 30 Stage 4 v2 prediction files carry the identical frozen outer-fold vector; `folds.json` equals the salted rank; 8 folds × 4 matches. All Stage 4C runs use `Ctx.fold` from the same function. | PASS |
| 2, 10 | The adapter returns exactly the 7 input fields. No feature name contains an identity, team, status, label or order word. Sequence channels are x/10 m, y/10 m, mask, dt. No status column reaches any model or any sample-acceptance decision. | PASS |
| 3 | `gru_gaplast`: reversal and scaling exact; masked steps all zero (positions and dt). The summary is invariant to values at masked gap-adjacent steps (max change 0.0), and the nearest valid point changes every embedding. 36 incoming / 22 outgoing of 50,662 tracklets have a masked gap-adjacent point; their summary is the nearest *valid* point. Fixed T = 30, no padding. | PASS |
| 4 | Normalisation uses incoming tracklets only. Origin = midpoint of the last valid incoming points (error 0.0 m). Mode-0 axis = summed incoming velocity (residual y-component ≤ 9.5e-7). Position scale fixed at 10 m; tabular scaler fitted inside each training fold. Stage 3 hidden-gap perturbation tests: all identical. | PASS (v2) |
| 5 | Every shifted row belongs to its encounter's match and outer fold. Reflection, order-exchange, shift, noise and smoothing augmentation act only on training batches or training partitions. Early-stop matches stay inside the outer training partition. | PASS |
| 6 | Complementarity under outgoing-order exchange (max \|p(swap) − (1 − p)\|): HGB 1.1e-16; GRU gap-last 4.2e-7 (float32); physics combo 1.1e-16; every Stage 4C GRU fold ≤ 5.1e-7. | PASS |
| 7 | 15 Hz grids (dt = 1/15 s, T = 30), the sequence dt channel and feature-path sample times (relative to the gap centre) agree; all `gap_s` = 0.8 in the primary table. | PASS |
| 8 | Superseded physics v1 (`B*_`), robustness variants (`__*`) and failed gate files stay on disk but are not ranked. | PASS |
| 9 | Stage 4B v2 headline metrics recomputed from the saved predictions (max difference < 1e-3). **Retraining reproduction**: Stage 4C config A at the 80-epoch cap reproduces the Stage 4B `gru_gaplast` predictions with max \|Δp\| = 0.0 for seeds 0, 1, 2. | PASS |
| – | Stage 4B "best epoch" equals the epoch of the loaded early-stopping state in 24 / 24 fold fits. | OK |

**No leakage path or model-implementation defect was found.**

### Shortcut investigation: tracking texture

The texture probe (`audit/texture_probe_v2.json`) uses symmetric HGB on the same folds.

- **Inputs:** per-tracklet texture statistics only.
  - Median and 90th percentile of |second difference|.
  - Median |third difference|.
  - Share of near-zero second differences.
- **Pair features:** antisymmetric cost differences of those statistics.
- **Deliberately excluded:** position, velocity, heading or distance.

| Stratum | n | BA [95 % CI] |
|---|---|---|
| All (independent) | 18,777 | **0.701** [0.691, 0.712] |
| Both fully ESTIMATED in gap | 9,730 | 0.739 [0.728, 0.749] |
| Mixed | 7,166 | 0.697 [0.675, 0.717] |
| Both fully VISIBLE in gap | 1,881 | **0.525** [0.504, 0.545] |
| ESTIMATED share of input points < 0.1 | 2,100 | 0.603 [0.583, 0.623] |
| 0.1–0.5 | 4,054 | 0.674 [0.641, 0.702] |
| 0.5–0.9 | 4,764 | 0.679 [0.664, 0.694] |
| ≥ 0.9 | 7,859 | 0.756 [0.744, 0.766] |

Median |second difference| per step: ESTIMATED 0.070 m vs VISIBLE 0.017 m.

**Interpretation.** Where positions are provider estimates, the fine texture of the incoming tracklet predicts which outgoing
tracklet is the same player. Where both are VISIBLE it does not.

- This is consistent with an estimation process whose smoothness or jitter signature is player-specific and persists across the gap.
  Such a signature would not exist with a different tracker.
- A genuine kinematic explanation (acceleration continuity) would be expected to work on VISIBLE data too, and it does not.
- This is **not proof** of the mechanism, and the label itself is unaffected (it comes from provider reference chains).

It does mean that any model's ESTIMATED-stratum performance may include a provider-specific component. Fully VISIBLE gaps are
the conservative estimate (§8).

## 2. Shift-augmentation dataset and experiment

### 2.1 Dataset (`m11-stage4c-shiftaug-v1`)

For every candidate of the primary table and each shift in {−0.3, −0.2, −0.1, +0.1, +0.2, +0.3} s, the Stage 3 procedure for shifted
gaps was re-run by calling the Stage 3 functions themselves (`shiftaug.py`):

1. Full Stage 2 label-quality procedure at the shifted centre. Only ACCEPTED shifts are built: continuous reference chains,
   chord-speed plausibility, discontinuity / label-transfer checks, window inside the period.
2. Construction from the two players' own chain vertices only: no interpolation through the hidden gap, no bridging between chains,
   incoming-only frame.
3. Independent label verification against the raw provider positions, gap-boundary and inverse-transform checks, and the hidden-gap
   perturbation test (10 per shift per match).

Results:

| | Value |
|---|---|
| Shifted samples built | **150,895** of 151,986 (25,023 – 25,258 per shift) |
| Rejected at revalidation | 1,091: continuous reference 936, chord-speed plausibility 137 (+1 both), discontinuity / label-transfer candidate 7, no identity risk 5, window outside period 5 |
| Label verification | 150,895 / 150,895 ok |
| Hidden-gap perturbation tests | 1,920 / 1,920 identical |
| Stage 3 reproduction | 21,657 / 21,657 byte-identical to the frozen `g0p80_shift` samples (same candidate, same shift, Stage 3 order bits) |
| Raw data | unchanged (snapshot before and after) |

- **Order bits** are drawn independently per shift (`order_bits(id, "g0p80_aug±s")`), so neither order nor row position identifies
  the shift.
- **Bookkeeping.** Every row keeps its candidate id, match, `g0p80` overlap group and independence flag.

### 2.2 Training distributions (predeclared; one draw per encounter per epoch)

- **A:** centred only (Stage 4B).
- **B:** each epoch, each training encounter is drawn uniformly from {centred} ∪ its valid shifted versions (1/7 centred when all six
  shifts are valid).
- **C:** each epoch, centred with probability 0.5, otherwise a uniformly drawn valid shifted version.

Common rules:
- Early stopping uses the same distribution, on the 3 early-stop matches of each outer training partition.
- Shift draws use a separate random stream, so A reproduces Stage 4B exactly.
- Shifted versions never add encounters or weight; augmented copies are never counted as independent observations.
- For HGB, the same distributions are expressed as sample weights: B = 1/(1 + n_shifts) per version; C = 0.5 centred, 0.5 / n_shifts
  per shifted version.
- HGB reuses the Stage 4 per-fold inner-selected configuration.

### 2.3 Evaluation

All metrics use the same outer folds and the independent encounters only.

- **Centred:** the frozen `g0p80` table.
- **Shifted:** all six shifts of the shift-augmentation set, 111,749 rows of 18,777 independent encounters, stratified by shift.
  Encounters repeat across shifts; CIs resample matches.
- **Comparability with Stage 4B:** the frozen `g0p80_shift` table (one seeded shift per candidate).

### 2.4 Results (mean ± sd over 3 seeds; GRU caps explained in §3)

| Model | Centred BA | Centred LL | Shifted BA (±0.1–0.3 s) | Shifted LL | `g0p80_shift` BA |
|---|---|---|---|---|---|
| GRU A, 80 ep (= Stage 4B) | 0.902 ± 0.005 | 0.228 | 0.815 ± 0.008 | 0.451 | 0.827 |
| GRU A, 160 ep | 0.920 ± 0.003 | 0.194 | 0.832 ± 0.002 | 0.424 | 0.845 |
| GRU B, 160 ep | 0.883 ± 0.011 | 0.266 | 0.890 ± 0.005 | 0.253 | 0.891 |
| **GRU C, 160 ep** | **0.926 ± 0.003** | **0.178** | **0.910 ± 0.001** | **0.214** | **0.910** |
| GRU C, 80 ep | 0.890 ± 0.010 | 0.254 | 0.877 ± 0.004 | 0.279 | 0.878 |
| HGB A (= Stage 4A) | 0.870 | 0.285 | 0.810 | 0.388 | 0.818 |
| HGB B | 0.856 | 0.310 | 0.883 | 0.265 | 0.877 |
| **HGB C** | 0.861 | 0.298 | 0.878 | 0.273 | 0.874 |

By shift, seed 0 (BA):

| Shift (s) | −0.3 | −0.2 | −0.1 | 0 | +0.1 | +0.2 | +0.3 |
|---|---|---|---|---|---|---|---|
| GRU A 80 ep | 0.718 | 0.819 | 0.858 | 0.900 | 0.834 | 0.832 | 0.806 |
| GRU A 160 ep | 0.749 | 0.844 | 0.876 | 0.922 | 0.852 | 0.851 | 0.826 |
| **GRU C 160 ep** | 0.905 | 0.915 | 0.907 | **0.924** | 0.903 | 0.912 | 0.911 |
| HGB A | 0.726 | 0.841 | 0.875 | 0.870 | 0.868 | 0.826 | 0.721 |
| **HGB C** | 0.887 | 0.883 | 0.875 | 0.861 | 0.873 | 0.876 | 0.878 |

Paired differences (seed 0, match bootstrap, 1,000 resamples):

| Comparison | Centred ΔBA | Shifted ΔBA | Shifted ΔLL |
|---|---|---|---|
| GRU C160 − GRU A160 | +0.002 [−0.003, +0.006] | **+0.076 [+0.070, +0.081]** | −0.205 [−0.217, −0.191] |
| GRU C160 − GRU A80 (Stage 4B) | +0.024 [+0.018, +0.029] | +0.098 [+0.091, +0.104] | −0.238 [−0.255, −0.220] |
| GRU B160 − GRU A160 | −0.040 [−0.048, −0.030] | +0.057 [+0.051, +0.063] | −0.168 |
| GRU C160 − GRU B160 | +0.041 [+0.031, +0.051] | +0.019 [+0.013, +0.024] | −0.037 |
| HGB C − HGB A | −0.009 [−0.012, −0.005] | **+0.069 [+0.064, +0.073]** | −0.116 [−0.121, −0.110] |
| HGB C − HGB B | +0.005 [+0.003, +0.007] | −0.004 [−0.005, −0.003] | +0.008 |
| GRU C160 − HGB C | +0.063 [+0.057, +0.069] | +0.031 [+0.027, +0.034] | −0.056 [−0.062, −0.051] |

**Conclusion.** The 50/50 mixture (C) makes both model families robust to a misplaced gap centre (±0.3 s). Shifted accuracy becomes
essentially flat across shifts.

- **GRU:** no measurable centred cost, provided the longer training budget is available. At 80 epochs C costs 0.006–0.012
  centred BA.
- **HGB:** C costs 0.009 centred BA and gains 0.069 shifted.
- **Uniform shifts (B)** cost the GRU 0.04 centred BA and are dominated by C.

## 3. Longer GRU training

Design (predeclared):
- Architecture, optimiser, weight decay, dropout, batch, patience 10, early-stop matches and outer folds are those of Stage 4B.
- One run per seed with a 160-epoch budget records the best early-stopping state reached within 80, 120 and 160 epochs.
- Because the main random stream is consumed exactly as in Stage 4B, the 80-epoch state *is* the Stage 4B model (bit-identical, §1).
- No outer-fold information is used to choose the epoch; inner early stopping chooses it within the budget.

| Config | Cap | Centred BA (3 seeds) | Centred LL | Train LL / fold-val LL (seed 0) | Best epoch per fold (seed 0) | Folds stopped by patience |
|---|---|---|---|---|---|---|
| A | 80 | 0.902 ± 0.005 | 0.228 | 0.202 / 0.231 | 76–80 | 0 / 8 (cap binding) |
| A | 120 | 0.919 ± 0.003 | 0.197 | 0.155 / 0.193 | 92–120 | 4 / 8 |
| A | 160 | 0.920 ± 0.003 | 0.194 | 0.153 / 0.191 | 92–127 | 8 / 8 |
| C | 80 | 0.890 ± 0.010 | 0.254 | – | 77–80 | 0 / 8 |
| C | 160 | 0.926 ± 0.003 | 0.178 | 0.162 / 0.184 | 114–159 | 5 / 8 (seed 0) |
| B | 160 | 0.883 ± 0.011 | 0.266 | – | up to 160 | cap binding in several folds |

- **Convergence.** For A, patience ended every fold between epochs 102 and 137, so 160 is not binding. The 80-epoch Stage 4B models
  were under-trained.
  - Config A 120 → 160 changes centred BA by +0.000 to +0.001.
  - Under C the harder mixed task uses more epochs: best epoch 159 in two seed-0 folds, so the 160 budget is borderline binding there.
- **Generalisation gap.** Train LL 0.15 vs validation LL 0.19, a moderate gap; validation keeps improving, so there is no
  overfitting signal on the in-distribution task.
- **Effect.** A160 − A80 (seed 0):
  - Centred: +0.022 [+0.017, +0.027] BA, −0.042 LL.
  - Shifted: +0.022 BA.
  - Gain in every status stratum: VISIBLE 0.818 → 0.828, ESTIMATED 0.913 → 0.934 (3-seed means).
- **Cost.** Longer training makes the GRU slightly *more* sensitive to inputs it was not trained on:
  - 5 cm noise: −0.020 BA, +0.25 LL.
  - 1.6 s gaps: −0.018 BA, +0.11 LL.
  - 0.4 s gaps: 0.797 → 0.759.

  This is a sign of fitting the training distribution's fine structure more closely (§4, §5).

**Conclusion:** longer training is a meaningful in-distribution improvement and is adopted (160-epoch budget, patience 10). It does
not fix, and slightly worsens, transfer to other gap lengths and noise.

## 4. Positional-texture robustness

Tracking status is evaluation metadata only. Configurations and test inputs:

- **Transforms applied to test inputs** (masks and timestamps preserved; identical across models):
  - Original.
  - `smooth5`: mask-aware 5-point moving average.
  - `noise5cm`: i.i.d. Gaussian, σ = 5 cm, on valid points.
- **Training-time augmentation** (centred training, seed 0, 160-epoch budget):
  - `Anoise`: σ ~ U(0, 5 cm) per sample and batch.
  - `Asmooth`: the smoothed copy with probability 0.5.
- **HGB equivalents:** original + transformed features at 0.5 weight each.
- **Stage 4B train-and-test transformed rows** (retrained on the transform) are reused for comparison.

| Model | Original | Smooth5 (test) | Noise5cm (test) | VISIBLE orig. | ESTIMATED orig. | Noise: VISIBLE / ESTIMATED |
|---|---|---|---|---|---|---|
| GRU A 80 (Stage 4B) | 0.900 | 0.836 | 0.786 | 0.812 | 0.912 | 0.743 / 0.748 |
| GRU A 160 | 0.922 | 0.827 | 0.767 | 0.830 | 0.939 | 0.728 / 0.725 |
| GRU C 160 | 0.924 | 0.821 | 0.757 | 0.841 | 0.939 | 0.710 / 0.723 |
| GRU Anoise 160 (1 seed) | 0.888 | 0.853 | **0.852** | 0.795 | 0.901 | 0.783 / 0.845 |
| GRU Asmooth 160 (1 seed) | 0.872 | **0.861** | 0.838 | 0.789 | 0.876 | 0.766 / 0.822 |
| GRU retrained on smooth5 / on noise5cm (Stage 4B, 80 ep) | – | 0.853 (train+test) | 0.850 (train+test) | – | – | – |
| HGB A | 0.870 | 0.864 | 0.851 | 0.795 | 0.875 | 0.788 / 0.845 |
| HGB C | 0.861 | 0.853 | 0.842 | 0.793 | 0.875 | 0.786 / 0.841 |
| HGB Anoise / Asmooth | 0.868 / 0.869 | 0.865 / 0.868 | 0.854 / 0.852 | 0.796 / 0.795 | 0.874 / 0.877 | – |

Status counts (independent, centred): both fully ESTIMATED 9,730; mixed 7,166; both fully VISIBLE 1,881.

Findings:
1. **The GRU depends on fine-scale positional detail; HGB does not.**
   - 5 cm noise costs the GRU 0.11–0.17 BA (0.11 for Stage 4B, 0.15–0.17 for the 160-epoch models; VISIBLE as well as ESTIMATED), and its log loss rises to 0.62–0.87, i.e. confidently
     wrong.
   - The same noise costs HGB 0.02.
   - The Stage 4B retrained-on-transform rows (0.850–0.853) show that the drop is not only a distribution shift: a GRU that never
     sees fine detail is no better than HGB.
2. **Augmentation trades centred accuracy for robustness.**
   - Noise augmentation restores noise robustness (0.852) at a centred cost of −0.035 [−0.039, −0.030] vs A160, ending at the HGB
     level (0.888 vs 0.870).
   - Smoothing augmentation costs −0.050.
   - Neither is recommended as a candidate: they sacrifice the GRU's advantage for robustness HGB already has.
3. **ESTIMATED vs VISIBLE.** Every learned model is 0.08–0.10 BA better on ESTIMATED than on VISIBLE gaps; physics is not (Stage 4B).
   - The texture probe (§1) shows ESTIMATED texture carries pairing information by itself.
   - The GRU's lead over HGB nevertheless **persists on fully VISIBLE gaps**: GRU C 0.841 [0.823, 0.856] vs HGB A 0.795
     [0.778, 0.812] and HGB C 0.793 [0.772, 0.812]. So the lead is not explained by ESTIMATED texture alone.
   - Under noise the GRU's VISIBLE advantage disappears (0.710 vs 0.786).
   - The available evidence does **not** establish that the GRU exploits provider artifacts. It establishes that the signal exists
     and that the GRU relies on fine-scale detail generally.

## 5. HGB / GRU ensemble

- **Selected-weight ensemble** (predeclared):
  - Weight grid: w ∈ {0, 0.25, 0.5, 0.75, 1} for w · p_GRU + (1 − w) · p_HGB.
  - Selection: separately per outer fold, by log loss on that fold's 3 early-stop matches (training partition only).
  - Inputs to selection: GRU early-stop predictions, and an HGB refitted without those matches.
- **Fixed 50/50 average**: the second predeclared combination.

| Ensemble (3 seeds) | Weights chosen | Centred BA | Centred LL | ECE | Shifted BA | 0.4 s | 1.6 s | Noise5cm | VISIBLE |
|---|---|---|---|---|---|---|---|---|---|
| ens A (GRU A160 + HGB A) | 0.75–1.0 | 0.923 ± 0.003 | 0.190 | 0.020 | 0.841 | 0.795 | 0.765 | 0.780 | 0.829 |
| ens B (GRU B160 + HGB B) | 0.5–1.0 | 0.887 ± 0.009 | 0.257 | 0.019 | 0.898 | 0.917 | 0.787 | 0.826 | 0.818 |
| **ens C (GRU C160 + HGB C)** | **1.0 in 24 / 24 folds** | = GRU C | = GRU C | = GRU C | = GRU C | = GRU C | = GRU C | = GRU C | = GRU C |
| fixed 50/50 (C members) | 0.5 | 0.925 ± 0.003 | 0.209 | **0.052** | 0.917 | 0.927 | 0.789 | 0.815 | 0.840 |

- Inner selection rejects HGB for config C.
- ens A − GRU A160: +0.004 [+0.002, +0.005] centred BA; ens A is less well calibrated (ECE 0.020 vs 0.009). The improvement is
  not useful.
- The **fixed 50/50** C average looks attractive outside the training distribution:
  - Shifted: +0.006 vs GRU C.
  - 0.4 s: +0.09.
  - 1.6 s: +0.04.
  - Noise: +0.06.
  - However, it is clearly worse calibrated (ECE 0.052 vs 0.007; LL +0.031).
  - Its benefits were observed on outer predictions, and the nested procedure (the only unbiased selector here) did not choose it.
  - Recommending it would be outer-fold selection, so it is **not** proposed. It is reported as evidence that HGB carries
    complementary robustness.

## 6. Model comparison with uncertainty (seed 0 unless "±")

| Model | Centred BA [95 % CI] | Centred LL | ECE | Shifted BA | VISIBLE BA [CI] | d_min < 5 cm (n = 2,129) | 0.4 s | 1.6 s | Noise | Coverage at 5 % error |
|---|---|---|---|---|---|---|---|---|---|---|
| **GRU C 160** | **0.924** [0.919, 0.928]; 3 seeds 0.926 ± 0.003 | **0.183** | 0.006 | **0.909** | **0.841** [0.823, 0.856] | **0.910** | 0.838 | 0.747 | 0.757 | **0.928** |
| GRU A 160 | 0.922 [0.918, 0.927] | 0.189 | 0.007 | 0.833 | 0.830 [0.815, 0.844] | 0.920 | 0.753 | 0.748 | 0.767 | 0.922 |
| GRU A 80 (Stage 4B best) | 0.900 [0.896, 0.905] | 0.231 | 0.008 | 0.811 | 0.812 [0.796, 0.829] | 0.889 | 0.801 | 0.767 | 0.786 | 0.849 |
| GRU B 160 | 0.883 [0.874, 0.893] | 0.267 | 0.010 | 0.890 | 0.815 | 0.863 | 0.889 | 0.767 | 0.807 | 0.785 |
| HGB A (Stage 4A best) | 0.870 [0.863, 0.875] | 0.285 | 0.010 | 0.810 | 0.795 [0.778, 0.812] | 0.850 | 0.910 | 0.799 | **0.851** | 0.738 |
| **HGB C** | 0.861 [0.855, 0.867] | 0.298 | 0.009 | 0.878 | 0.793 [0.772, 0.812] | 0.836 | **0.928** | **0.803** | 0.842 | 0.717 |
| Best physics (phys_B3_combo, Stage 4A) | 0.696 | 0.569 | 0.024 | – | 0.669 | – | – | – | – | – |

Seed stability (sd of centred BA over 3 seeds):
- GRU C: 0.003; GRU A: 0.003–0.005; GRU B: 0.011.
- HGB is deterministic (three identical seeds, Stage 4A).
- GRU C seeds: 0.924 / 0.925 / 0.929 centred; 0.909 / 0.909 / 0.911 shifted.

## 7. Failure analysis

1. **Gap-length mismatch.** A model trained on 0.8 s gaps and applied to a 0.4 s gap is *worse* for the GRU (0.838) than for HGB
   (0.928), although shorter gaps are easier.
   - The GRU receives the gap length as an input but has only ever seen 0.8 s, so it cannot use it.
   - At 1.6 s: GRU C 0.747 vs HGB C 0.803. Log loss 0.79 vs 0.46: the GRU is confidently wrong (ECE 0.128).
   - Stage 4B showed that retraining at each gap length fixes most of this (0.963 / 0.815).
2. **Positional noise.** See §4. GRU confidence does not drop when the input is degraded (log loss 0.87).
3. **Misplaced gap centre.** Fixed by C for both families (§2).
4. **ESTIMATED-dependent signal.** All learned models lose 0.08–0.10 BA on fully VISIBLE gaps; part of the ESTIMATED advantage is
   provider texture (§1).
5. **Stage 4B high-confidence errors** remain characterised as enriched for no-crossing encounters and VISIBLE gaps, with no match
   concentration. Possible provider identity errors were not adjudicated against video. This was not re-run for C.
6. **Close encounters** (d_min < 5 cm): GRU C 0.910 centred / 0.914 shifted; HGB C 0.836 / 0.878. No collapse.

## 8. Tracking-status, gap duration and shift: summary

**Status** (centred BA; seed 0 GRU, deterministic HGB):

| Model | Both ESTIMATED (9,730) | Mixed (7,166) | Both VISIBLE (1,881) | VISIBLE ECE | Shifted: ESTIMATED / VISIBLE |
|---|---|---|---|---|---|
| GRU C 160 | 0.939 [0.934, 0.943] | 0.926 | 0.841 [0.823, 0.856] | 0.038 | 0.921 / 0.862 |
| GRU A 160 | 0.939 | 0.924 | 0.830 | 0.057 | 0.840 / 0.768 |
| HGB A | 0.875 | 0.881 | 0.795 | 0.034 | 0.846 / 0.726 |
| HGB C | 0.875 | 0.860 | 0.793 | 0.042 | 0.906 / 0.816 |

**Gap duration and shift.** See §2.4 and §7. Every model has seen only 0.8 s gaps in training.

**Development-selection optimism.** Stage 4C choices were informed by Stage 4A / 4B outer-fold results on the same 32 matches:
- the GRU family and gap-last variant;
- the decision to try longer training (the binding cap);
- shift augmentation (the shift sensitivity).

Within Stage 4C, the configurations, caps, weight grid and abstention target were predeclared. However, the final pick among A / B / C
and the cap budget were made after seeing outer-fold results. The 0.926 / 0.910 estimates are therefore optimistic by an unknown
small amount. The held-out evaluation is the only unbiased estimate.

## 9. Calibration and abstention

- **Calibration** (centred, independent): no post-hoc calibration is needed in-distribution.
  - ECE: GRU C 0.006–0.009 (3 seeds 0.007); HGB C 0.009.
  - Shifted: GRU C 0.007; HGB C 0.014.
  - Both degrade on VISIBLE gaps (ECE 0.038–0.042) and badly for the GRU under gap-length or noise shift (§7).
- **Abstention, nested:** each fold's threshold comes from the other seven folds' OOF predictions at a target error of 5 %.

| Model | Centred coverage / error | Shifted coverage / error (same thresholds) | AURC |
|---|---|---|---|
| GRU C 160 (3 seeds) | 0.928 / 0.050; 0.935; 0.948 | 0.911 / 0.060; 0.922 / 0.063; 0.935 / 0.065 | 0.012 |
| GRU A 160 | 0.922 / 0.050 | 0.867 / 0.127 | 0.013 |
| GRU A 80 (Stage 4B) | 0.849 / 0.050 | 0.774 / 0.124 | 0.019 |
| HGB A | 0.738 / 0.050 | 0.652 / 0.087 | 0.031 |
| HGB C | 0.717 / 0.050 | 0.728 / 0.042 | 0.034 |

- Shift training keeps abstention thresholds valid under centre shift: GRU C's realised shifted error is 6.0–6.5 %, vs 12.7 % for A.
- Pooled development thresholds at 5 % target error (all 8 folds, independent):
  - GRU C seed 0: confidence ≥ 0.6425, coverage 0.927.
  - HGB C: confidence ≥ 0.7875, coverage 0.716.
- At 2 %: GRU C 0.8525 (coverage 0.788); HGB C 0.9029 (coverage 0.546).

## 10. Runtime and resource use

- **Hardware:** GTX 1650 Ti (4 GB), 8-thread Intel CPU. PyTorch on CUDA; cuDNN benchmark off.
- **GRU training** (8 folds, single job, 160-epoch budget):
  - A: 1,152–1,416 s per seed.
  - B / C: 1,152–1,564 s per seed.
  - Texture configs: 1,009–1,082 s.
  - Total Stage 4C GPU time ≈ 4.0 h (14,297 s), run sequentially (no concurrent GPU jobs).
- **HGB fits:**
  - A: 10–13 s.
  - B / C: 21–42 s per fold (6× data).
  - Stage 4C HGB phase: 15 min including the shift-set motion features (345 s for 150,895 samples).
- **Shift dataset build:** 32 matches, ~20 s each, ~11 min.
- **Inference per sample** (`runtime/runtime.json`, 2,000 samples):
  - GRU 0.007 ms (GPU) / 0.053 ms (CPU).
  - HGB 0.012–0.015 ms, plus 1.68 ms for motion-feature extraction.
  - Both are negligible for offline use.
- **Model size:** GRU 43,585 parameters (≈ 0.2 MB per fold state). Peak memory was not instrumented (**NOT MEASURED**).

## 11. Reproducibility checks and hashes

- Stage 4B `gru_gaplast` predictions reproduced bit-identically by a fresh Stage 4C run (§1).
- Stage 3 shifted samples reproduced byte-identically (21,657 / 21,657).
- All Stage 4C units are written once and resumable. Evaluation versions are never overwritten (`metrics/v1`).

SHA-256 (full values in the proposed protocol):

| Artifact | SHA-256 (prefix) |
|---|---|
| `m11-stage2-split-v1/split.json` | 66b2321a58f33845 |
| `m11-stage3-samples-v1/manifest.json` | 6016fa6c00814ef4 |
| `m11-stage4c-shiftaug-v1/manifest.json` | 926584d773aae51c |
| `m11-stage4-benchmarks-v1/folds.json` | d9c07ea4f449e590 |
| `m11-stage4c-robustness-v1/metrics/v1/comparison.json` | ec10a7c68ef99fda |
| `m11-stage4c-robustness-v1/audit/audit_v2.json` | c7f7835f6debba36 |
| `m11-stage4c-robustness-v1/audit/texture_probe_v2.json` | 5efa23a28ab5e42e |

**Tests:**
- New `tests/test_m11_stage4c.py`: 5 passed.
- Full repository suite after the Stage 4C implementation: **1029 passed, 6 skipped** (237 s).
- Lint (ruff) not run: not installed.

**Not completed / not run:**
- Texture-augmentation GRUs ran with 1 seed only.
- The high-confidence error figure was not regenerated for GRU C.
- Peak memory was not measured.
- No video adjudication of suspected provider identity errors.
- No multi-gap-length training (not authorised; see §14).

## 12. Remaining scientific limitations

1. **Provider-specific texture.**
   - ESTIMATED positions contain identity-linked texture (§1).
   - The VISIBLE stratum (n = 1,881 independent, 10 %) is the conservative estimate: GRU C ≈ 0.84, HGB ≈ 0.79.
2. **Single training gap length.** Real gaps vary. Both candidates are trained only at 0.8 s; the GRU transfers badly and
   over-confidently to other lengths.
3. **Noise sensitivity of the GRU.** Different trackers or noisier sensors could cost it 0.15 BA with confident errors.
4. **Reference identities are the provider's.** Label errors were not video-adjudicated.
5. **Development-selection optimism** (§8). Development data is one tournament from one provider.
6. **Shift distribution.** The shifted evaluation is uniform over ±0.1–0.3 s; real gap placement relative to closest approach is
   unknown.

## 13. Final model recommendation and rationale

The recommendation balances centred and shifted accuracy, the VISIBLE stratum, calibration, abstention, robustness, stability and
complexity.

**Candidate 1 (primary): GRU gap-last, shift mixture C, 160-epoch budget.**
- Best on every in-distribution criterion:
  - Centred 0.926 ± 0.003; shifted 0.910 ± 0.001, flat across ±0.3 s.
  - VISIBLE 0.841; close encounters 0.910.
  - ECE 0.007; 93 % coverage at 5 % error, still 6 % error under shift.
- Stable across seeds (sd 0.003).
- Its advantage over HGB is robust to gap placement and present on VISIBLE gaps.
- It is **not** robust to gap-length change or added positional noise.

**Candidate 2 (reference): HGB with shift mixture C (HGB C).**
- Simpler, deterministic, and far more robust:
  - Noise 0.842; 0.4 s 0.928; 1.6 s 0.803.
  - Calibrated under transfer.
- Shift-robust (0.878), at a 0.06 centred-BA deficit to the GRU.
- Including it lets the held-out evaluation test whether the GRU's development advantage holds on unseen matches, and gives a fallback
  if it does not.

**Not recommended:**
- **Ensembles:** the nested procedure selected the GRU alone; the fixed 50/50 average is outer-selected and poorly calibrated.
- **GRU B:** dominated by C.
- **Texture-augmented GRUs:** they give up the advantage over HGB.
- **Centred-only models (A):** not shift-robust.

## 14. Decisions requiring your approval before Stage 5

1. **Approve or amend the two candidates** and the [proposed final protocol](m11_stage5_final_protocol_PROPOSED_v1.md): training on
   all 32 train/CV matches, seeds, thresholds, metrics.
2. **Authorise building held-out supervised data once.** This needs a guarded code change (both builders currently refuse held-out
   matches). The data to build:
   - Stage 3 tables `g0p80`, `g0p40`, `g1p60`, `g0p80_shift` for the 8 frozen held-out matches.
   - The shift-augmentation set for the same matches.
3. **Decide whether the held-out report treats fully VISIBLE gaps as a co-primary stratum**, given the texture finding.
4. **Optional, before freezing (each needs authorisation and adds development-selection optimism):**
   - Multi-gap-length training (0.4 / 0.8 / 1.6 s mixture) to address the GRU's gap-length failure.
   - Video adjudication of high-confidence errors.
   - A 3-seed run of the texture-augmented GRU.
5. Git: all new and modified files are uncommitted and left for your review.
