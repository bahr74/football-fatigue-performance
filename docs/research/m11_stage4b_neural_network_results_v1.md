# M11 Stage 4B — neural networks (MLP, GRU, LSTM) (v1)

Date: 2026-11-14. Scope: the 32 train/CV matches only, with the same table (`g0p80`), folds, independent subset and metric
code as [Stage 4A](m11_stage4a_baselines_and_ml_results_v1.md). No held-out data was used. Stage 5 has not started.

## 1. Inputs

- Only the 7 Stage 3 input fields are used. For each tracklet, a (T=30, 4) sequence is built: x/10 m, y/10 m, valid-mask and
  dt (s), in the sample frame (`data.seq_tensor`).
- Incoming tracklets are time-reversed, so every sequence **starts at the gap edge** and runs away from the gap.
- Masked steps are explicitly zeroed (unit test `test_seq_tensor_masks_and_reversal`).
- The gap length is a scalar input to the pair head.
- No identity, team, status or crossing metadata is used.

## 2. Architecture (small, fixed; no architecture search)

- **Encoders.** The incoming and outgoing tracklets each have their own encoder.
  - MLP: flatten → 96 → 48.
  - GRU: hidden 48, then a projection to 48.
  - LSTM: hidden 40, then a projection to 48.
- **Pair head.** For each of the four (incoming i, outgoing j) pairings, a shared head scores
  [a, b, a·b, |a−b|, gap] → s_ij.
- **Output.** logit(SWAPPED) = s01 + s10 − s00 − s11. This is exactly antisymmetric: exchanging either the incoming or the
  outgoing order negates the logit. Unit tests cover all 5 kinds; per-fold symmetry errors are stored in the fold JSONs.
- **Training.**
  - Optimiser: AdamW (lr 1e-3, weight decay 1e-4), dropout 0.1, gradient clip 1.0, batch 256, at most 80 epochs.
  - Early stopping: patience 10 on the log loss of 3 early-stop matches held out from each outer training fold.
  - Augmentation: reflection y → −y, plus random outgoing and incoming order exchanges, each with its label flip.
- **Parameters:** mlp 55,873; gru / gru_gaplast 43,585; lstm / lstm_gaplast 41,985.

### Documented correction: gap-last recurrent variants

The original `gru` and `lstm` read each sequence from the gap edge outward and summarise at the **last valid step**, which is
the point farthest from the gap. They underfit: train LL ≈ validation LL ≈ 0.45–0.49, and early stopping triggered after
16–78 epochs.

The `*_gaplast` variants make one change: the sequence is flipped before the RNN, so the summary state is taken at the
gap-adjacent point, where the association evidence is. No other hyperparameter was changed.

Both original models are **kept in the comparison** at 1 seed. They were not removed for performing poorly, and their
seeds 1–2 were not run, to save GPU time.

## 3. Results (independent subset, n = 18,777)

Values are mean ± sd over seeds; the 95% CI is the seed-0 match-level bootstrap (2,000 resamples). Final evaluation:
`metrics/g0p80/v2/`.

| Model | Seeds | Indep. BA | 95% CI (seed 0) | Log loss | AUC | Brier | ECE | Train LL / Val LL (seed 0) |
|---|---|---|---|---|---|---|---|---|
| **gru_gaplast** | 3 | **0.9021 ± 0.0045** | [0.896, 0.905] | **0.2276 ± 0.0091** | 0.969 | 0.070 | 0.008 | 0.202 / 0.231 |
| lstm_gaplast | 3 | 0.8570 ± 0.0014 | [0.854, 0.863] | 0.3106 ± 0.0026 | 0.940 | 0.099 | 0.008 | 0.276 / 0.307 |
| mlp | 3 | 0.8447 ± 0.0020 | [0.837, 0.848] | 0.3351 ± 0.0029 | 0.930 | 0.107 | 0.011 | 0.310 / 0.338 |
| gru (original) | 1 — INCOMPLETE seeds | 0.7647 | [0.746, 0.784] | 0.4496 | 0.866 | 0.149 | 0.020 | 0.446 / 0.449 |
| lstm (original) | 1 — INCOMPLETE seeds | 0.7366 | [0.730, 0.744] | 0.4852 | 0.839 | 0.162 | 0.020 | 0.488 / 0.486 |

Per seed (BA / LL):

| Model | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| gru_gaplast | 0.9003 / 0.2307 | 0.9072 / 0.2174 | 0.8987 / 0.2348 |
| lstm_gaplast | 0.8583 / 0.3113 | 0.8556 / 0.3128 | 0.8572 / 0.3078 |
| mlp | 0.8424 / 0.3379 | 0.8455 / 0.3321 | 0.8461 / 0.3352 |

Paired match-bootstrap differences (seed 0, independent subset). The CI excludes 0 in every case.

| Comparison | ΔBA [95% CI] | ΔLL [95% CI] |
|---|---|---|
| gru_gaplast − hgb (best tabular) | **+0.0308** [+0.0251, +0.0364] | **−0.0540** [−0.0611, −0.0467] |
| gru_gaplast − phys_B3_combo | +0.2045 [+0.1976, +0.2115] | −0.3378 [−0.3464, −0.3293] |
| lstm_gaplast − hgb | −0.0112 [−0.0163, −0.0060] | +0.0266 [+0.0198, +0.0330] |
| mlp − hgb | −0.0271 [−0.0325, −0.0217] | +0.0531 [+0.0453, +0.0607] |
| gru (original) − gru_gaplast | −0.1356 [−0.1535, −0.1176] | +0.2188 [+0.1986, +0.2371] |

gru_gaplast details (seed 0):

- Precision / recall / F1 (SWAPPED): 0.900 / 0.901 / 0.900.
- Confusion matrix: TN 8458, FP 940, FN 932, TP 8447.
- Per-fold BA: 0.9004 ± 0.0077. Per-match BA ranges from 0.880 to 0.929.

## 4. Training versus validation

- **gru_gaplast:** train LL 0.18–0.22 vs fold validation LL 0.21–0.25, a small generalisation gap.
  - The best epoch was 76–80 of 80 in all 24 fold fits (3 seeds × 8 folds), so **the 80-epoch cap was binding**.
  - The model was still improving when training stopped, so it is mildly under-trained. Longer training was not run: it was
    not in the approved budget and would be a tuning change.
- **lstm_gaplast:** mostly the same pattern (best epoch ≥ 76 in 16 of 24 fold fits, range 39–80); train LL 0.25–0.33 vs validation 0.28–0.35.
- **mlp:** best epoch 40–79 (early stopping active); train LL 0.28–0.34 vs validation 0.31–0.36. This is mild overfitting
  and is controlled by early stopping.
- **gru / lstm (original):** train LL ≈ validation LL ≈ 0.45–0.49. They **underfit**, consistent with the summary-position
  problem described in §2.

## 5. Runtime and hardware

- PyTorch 2.9.1 on CUDA (one consumer GPU), batch 256. cuDNN benchmark was off for determinism.
- Single-job fold fits took 50–160 s.
- Seed-level runtimes, single job: mlp 529 s; gru 607 s; lstm 459 s; gru_gaplast 780–819 s; lstm_gaplast 1147 s.
- When run concurrently with other GPU jobs, the same seeds took 1256–2367 s. These are wall-clock contention effects;
  results are unaffected (deterministic seeds).
- GPU memory in use was about 2 GB (nvidia-smi snapshot during a run). Peak memory was not instrumented (NOT MEASURED).

## 6. Robustness of the neural model

Robustness results for gru_gaplast are in the [combined report](m11_stage4_combined_comparison_v1.md): gap transfer, shift,
texture ablations (inference-time and retrained), dependency weighting, learning curve, abstention and errors.

The three findings most relevant to the neural model are:

1. **Retraining on a smoothed copy of the inputs costs the GRU more than HGB.** GRU BA falls from 0.900 to 0.853; HGB falls
   from 0.870 to 0.869.
   - This hints that part of the GRU's advantage comes from fine-scale positional detail.
   - Whether that detail is genuine motion signal or provider-specific noise texture is **not established**.
2. **The GRU transfers poorly across gap lengths without retraining.**
   - A model trained at 0.8 s scores BA 0.801 on 0.4 s gaps and 0.767 on 1.6 s gaps.
   - HGB scores 0.910 and 0.799 under the same transfer.
   - At 1.6 s, the retrained GRU no longer beats the retrained HGB: ΔBA −0.0074 [−0.0146, −0.0001].
3. **The GRU is more sensitive to the gap not being centred on closest approach.** Shifted-centre BA is 0.822 for the GRU
   vs 0.818 for HGB, but at ±0.1 s shifts the GRU drops much further (0.828 / 0.856 vs centred 0.913 / 0.904).

## 7. Outputs

- `preds/g0p80/{mlp,gru,lstm,gru_gaplast,lstm_gaplast}/seed*.npz|json`.
- Fold weights and per-fold histories are in `folds/seed<k>_fold<f>.pt|npz|json`, under
  `data/processed/m11-stage4-benchmarks-v1/`.
