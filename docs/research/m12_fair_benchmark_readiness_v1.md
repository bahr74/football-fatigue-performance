# M12 report 3: Fair four-model benchmark readiness (v1)

Status: infrastructure built and CPU-tested; **no GPU training was run**. Date 2026-11-21. This extends the protocol
`m11_fair_four_model_protocol_v1.md`; that document still governs the partition, budgets and calibration.

## A. Training distributions, kept separate from evaluation (`m11_fair/distributions.py`, `m11_fair/evaluation.py`)

Every distribution draws **exactly one row per encounter per epoch** (the Stage 4C rule): variants are augmentations, never extra
independent samples. Training distributions:

| Name | Rows drawn from | P(centred) | Online perturbation | Purpose |
|---|---|---|---|---|
| `centred` | g0p80 | 1.0 | none | Historical setting |
| `shift_mix` | shiftaug (±0.1/0.2/0.3 s), g0p80_shift | 0.5 | none | Removes the centring prior |
| `gap_mix` | g0p40, g1p60 | 1/3 | none | Variable gap length |
| `robust_mix` | all four tables | 0.4 | Position noise σ ∈ {0, 0, 2, 5} cm, context dropout ∈ {0, 0, 10, 30} % | Texture and missing observations |
| `visible_only` | g0p80 rows with both players VISIBLE in the gap | 1.0 | none | Removes the ESTIMATED shortcut |

Perturbation never touches the gap-nearest point or masked steps. These properties are tested.

Evaluation stress tables are fixed and identical for every model:

- `g0p80` (centred);
- `g0p80_shift`;
- `g0p40`;
- `g1p60`;
- `smooth5`;
- `noise5cm`.

Predictions on every stress table are saved per fold and seed. Stress tables are never used for training or selection.

**Measured on the real pool (CPU):**
- `centred` has 25,331 rows (46 MB of sequences).
- `shift_mix` has 201,398 rows (369 MB).
- `robust_mix` has 251,371 rows (460 MB), with a peak RSS of 825 MB.
- The drawn variant share is 0.50 for `shift_mix` and 0.61 for `robust_mix`, matching the design.
- Loading all six evaluation tables takes 8 s (580 MB).

## B. Comparable search budgets (`m11_fair/halving.py`, `m11_fair/fold_v2.py`)

| Requirement | Implementation |
|---|---|
| Same configuration budget | Every family samples from its own 24-configuration space (`spaces.py`) with the same seeds and salt |
| **Compute matching ≠ equal configuration count** | `configs_for_budget(budget_s, s_per_epoch)` converts one GPU-seconds budget into a configuration count per family from **measured** s/epoch. Historical GPU: GRU 1.24 s/epoch, LSTM 3.15 s/epoch. At equal seconds the LSTM gets fewer configurations, and reports must say so |
| Parameter matching | `param_matched_hidden`: LSTM h42 has 43,585 parameters, identical to GRU h48 (43,585). Tested within 2 % |
| Capacity comparison | Pilot includes the largest search sizes (GRU h96 l2, 203 k; LSTM h96 l2, 260 k) |
| Stopping | Same `patience` and `epoch cap` for both RNNs; early stopping on centred primary rows of 3 training matches (`early_stop` role) |
| Calibration | Temperature (neural) and Platt (SVM) fitted only on the `calibration` role matches; never the test fold |
| Uncertainty | Match-level bootstrap CIs on paired differences; 5 neural and 3 tabular seeds per fold (protocol) |
| Tabular models | Variant rows weighted 1/count per encounter (`FairTabular.fit(sample_weight=...)`), so augmentation does not inflate effective sample size |

## C. Convergence evidence (`m11_fair/train.py`)

Every neural fit records:

- **History:** per-epoch training loss, early-stopping log-loss, and `elapsed_s`.
- **Convergence** (`convergence()`):
  - `best_epoch`, `at_cap`, `stopped_by_patience`, `still_improving`;
  - `overfitting`, `es_rise_after_best`, `generalisation_gap_at_best`, loss slopes.
- **Cost:**
  - parameter count, `fit_s`, `s_per_epoch`;
  - peak process RSS (psutil), peak GPU memory.
- **Checkpoint integrity:**
  - written once with a sha256 sidecar (parameter count, configuration, metadata);
  - overwrite is refused;
  - tampering is detected by `verify_checkpoint`.
- **Resume:** per-epoch state (network, optimiser, best state, patience counter, history, NumPy, torch and generator RNG states)
  is written atomically (temporary file + `os.replace`). A resumed run is **bit-identical** to an uninterrupted one, including the
  random draws and perturbation (tested).
- **Failure:** a non-finite training loss raises an error instead of training on.

## D. Efficiency

- **Successive halving:**
  - rungs (20, 60, 180, 400) epochs, eta 3;
  - 24 configurations cost 24×20 + 8×40 + 3×120 + 1×220 = 1,380 epochs instead of 24×400 = 9,600 (about 7× less);
  - rung states are resumed, not retrained;
  - seed 0 of the final fit reuses the finished search state.
- **Pilot first:** a one-fold, one-seed GPU pilot (report 4) measures s/epoch and best epochs before any full budget is requested.
- **Labelling:**
  - any reduced budget (fewer folds, seeds or configurations, or halving instead of a full search) is written into the output
    path (`<search>_<n>cfg`) and must be stated in every table;
  - a halving result must not be described as a "full fair comparison".

## CPU verification actually run

| Check | Result |
|---|---|
| `tests/test_m11_fair_v2.py` (10 tests) | Pass: historical equivalence (v2 default path equals `m11_benchmark.models.Neural`, atol 1e-6), bit-identical resume, checkpoints, convergence flags, halving schedule and selection, parameter matching, pool draws, perturbation, end-to-end fold runner (gru, hgb) resumable |
| Real-data fold runner, CPU, fold 0, `shift_mix` | LSTM (tiny, 2 epochs, 2 configurations): runs end to end, 17,504 fit encounters, predictions for all 6 stress tables, 20 s. HGB (30 iterations, 2 configurations): 16 s. **These BA values are smoke values from deliberately tiny models and are not results** |
| CPU s/epoch, 5,000 rows, 4 threads | GRU h48: 11.9 s; LSTM h40: 3.9 s; LSTM h42: 4.4 s; GRU h96 l2: 10.8 s; LSTM h96 l2: 4.0 s. **CPU ordering is the opposite of the GPU measurements, so CPU timing is not used for GPU estimates** |

## What still prevents a final fair comparison

1. **No GPU run under v2.** Convergence (best epoch versus the cap) and real s/epoch for the larger configurations are unknown.
   This needs pilot approval.
2. **The budget is not approved.** A compute-matched search over 8 folds × 5 seeds × 2 RNN families is several GPU-days (estimate
   after the pilot).
3. **No texture-neutral headline yet.** The fair comparison must report both-VISIBLE and texture-robust (`robust_mix`) results
   (report 2), or it will rank models by how well they read provider interpolation.
4. **Shared GPU.** Session B also uses the GPU; runs must not overlap (the pilot refuses to start while another compute process
   is on the GPU).
