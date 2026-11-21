# Fair GRU / LSTM / SVM / HGB benchmark: protocol and approval package (v1)

**Date:** 2026-11-20

**Status:**
- Infrastructure is implemented and tested.
- The plan package is written.
- A CPU tabular pilot was run (labelled as a pilot).
- **No GPU training has been run.** Neural training is refused without an explicit approval id.

**Code:** `src/football_fatigue/m11_fair/` (`nets.py`, `spaces.py`, `train.py`, `runner.py`, `protocol.py`, `pilot.py`, `__main__.py`)

**Tests:** `tests/test_m11_studio.py` (fair-benchmark section)

**Plan output:** `data/processed/m11-fair-benchmark-plan-v1/plan.json` plus `manifest.json` (code and split hashes)

**Labels:**

| Label | Meaning |
|---|---|
| [V] | Verified |
| [EST] | Estimate |
| [REC] | Recommendation |

## 1. What was unfair before [V, Phase 2 Task D]

| Family | Historical problem |
|---|---|
| GRU and LSTM | 80-epoch cap. Best epoch within 10 of the cap in 24/24 (GRU) and 21/24 (LSTM) fits. |
| LSTM | Smaller (hidden 40); never extended, tuned or shift-trained. |
| GRU | Only the GRU later got 160 epochs. |
| SVM | Fixed Nystroem-1000 approximation; 6-point grid; the selected C = 1, γ = 0.5 sits on the grid edge in 24/24 fits. |
| HGB | 4-point grid; always the largest leaf count. |
| Model selection (Stage 4C, M1) | Made on the reported OOF results. |

## 2. Protocol

| Item | Rule |
|---|---|
| Population | Stage 3 `m11-stage3-samples-v1`. g0p80 primary; g0p40 / g1p60 / g0p80_shift for evaluation. 32 train/CV matches. |
| Roles per outer fold | test = frozen Stage 4 outer fold (4 matches); calibration 3; early-stop 3; fit 22 (3 inner folds). All match-disjoint. Salts `m11-phase2-cal-v1` / `m11-phase2-es-v1`. |
| Partition validation | 7 / 7 rules pass [V] (`validate_partition`): test folds partition the 32; equal the frozen folds; roles disjoint; cover exactly the 32; sizes 22/3/3/4; inner folds = fit; no protected match. |
| Input access | GRU / LSTM: observed tracklet sequences (x, y, mask, dt). SVM / HGB: the 36 motion features computed from the same observed tracklets. Never inputs: identities, provider status, label-derived tiers. |
| Search | Scrambled Sobol, **24 configurations per family**, salt `m11-fair-search-v1`, deterministic. Edge flags trigger one documented extension round. |
| Seeds | Neural 5 (0–4); tabular 3 where stochastic. |
| Selection | Per outer fold, on early-stop matches only: neural by early-stop log loss; tabular by early-stop AUC (scale-free). **Test is never used for a choice.** |
| Calibration | Symmetric temperature / Platt scale fitted on the 3 calibration matches only. |
| Abstention | Thresholds from calibration matches; risk–coverage reported on test. |
| Comparisons | Best-of-search per family. Parameter-matched GRU vs LSTM. Compute-matched (equal GPU seconds). Paired match-bootstrap differences (2,000). |
| Controls | Independent subset. Both-VISIBLE reference. Mixed / ESTIMATED strata. Clone-like exclusion. Texture/smoothing ablation. Noise/dropout stress. Centred vs shifted. Short/long gaps. Physics-hard / contested / easy tiers (evaluation only). Deterministic baselines as reference rows. |

**Search spaces** (`spaces.py`):

| Family | Search space |
|---|---|
| GRU, LSTM | lr 3e-4–3e-3 (log); hidden {32, 48, 64, 96}; layers {1, 2}; head dropout 0–0.3; weight decay 1e-5–1e-3 (log); batch {128, 256}; cap 400 epochs; patience 25; AdamW |
| SVM, Nystroem | C 0.03–30; γ 0.05–4 (× 1/n_features); n_components {1000, 2000, 4000} |
| SVM, exact RBF | `sklearn.svm.SVC` on a seeded ≤ 6,000-row subsample of the fit rows. Same C / γ. Reported separately from the approximation. |
| SVM, linear | C 0.001–10 |
| HGB | lr 0.02–0.2; leaves {15, 31, 63, 127}; min_samples_leaf 10–80; l2 0–2; up to 2,000 iterations; best iteration on early-stop matches |

**Deviation from the full nested plan (documented):** configuration selection uses the single early-stop split, not 3 inner folds, to save compute. Calibration and test matches are never involved.

## 3. Implementation checks [V]

| Check | Result |
|---|---|
| `nets.make_pair_net` at 1 layer and the historical sizes vs `m11_benchmark.models.make_net` | Identical state-dict keys, identical seeded weights, identical outputs; 43,585 / 41,985 parameters (GRU / LSTM) |
| Existing checkpoints | Untouched (`models.py` not modified) |
| Synthetic smoke run of every family (`python -m football_fatigue.m11_fair smoke`, CPU, 17 s) | Selection, calibration and test prediction all run; outputs finite |
| `python -m football_fatigue.m11_fair train --family lstm` without `--approval` | **REFUSED** (exit 2) |
| Write-once | Checkpoints and fold results are never overwritten |

## 4. CPU tabular pilot (PILOT, reduced budget) [V]

**Command:** `python -m football_fatigue.m11_fair cpu-pilot`

**Output:** `data/processed/m11-fair-cpu-pilot-v1/`

**Budget:**
- 8 configurations per family; seed 0.
- HGB capped at 1,000 iterations.
- No model file saved, so there is no live inference.

**Run:** 2 workers, 55 min, **0 failed trials** (an earlier 5-worker attempt hit MemoryError on a memory-constrained machine and
was discarded before writing anything). Independent subset, 18,777 samples, 32 matches, OOF over the 8 outer folds [V]:

| Model | Independent BA | Paired vs historical same family (match bootstrap 95 % CI) | vs historical GRU gap-last |
|---|---|---|---|
| **PILOT HGB** (8 configs, ES best iteration) | **0.873** (AUC 0.952, ECE 0.008) | +0.0035 [+0.001, +0.006] vs HGB 0.870 | −0.027 [−0.033, −0.022] |
| PILOT SVM Nystroem | 0.862 | +0.003 [−0.001, +0.006] vs SVM 0.859 | −0.039 [−0.045, −0.033] |
| PILOT SVM exact RBF (≤ 6,000 rows) | 0.852 | −0.006 [−0.009, −0.004] | −0.048 [−0.053, −0.043] |
| PILOT SVM linear | 0.837 | −0.021 [−0.025, −0.018] | −0.063 [−0.069, −0.057] |
| Kalman forward (reference) | 0.667 | | |

Strata (PILOT HGB / historical GRU): both-VISIBLE 0.807 / 0.812; mixed 0.880 / 0.908; both-ESTIMATED 0.880 / 0.912;
physics-contested 0.846 / 0.883; physics consensus-wrong 0.872 / 0.910.

**Reading.** A wider, early-stop-selected tabular search moves HGB by +0.004 BA and SVM by +0.003 (not significant): the historical
tabular baselines were *not* materially under-tuned. The GRU lead (~0.03–0.04) survives the fairer tabular search, and it is
concentrated in mixed / ESTIMATED and physics-hard samples; on both-VISIBLE samples HGB and GRU are equal (0.807 vs 0.812). The
exact RBF SVM (subsampled to 6,000 rows for tractability) is below the Nystroem approximation, so the approximation is not the
bottleneck. Selection edge flags: HGB chose lr ≈ 0.021 (range edge) and 127 leaves (list end) in 6/8 folds; Nystroem chose 1,000
components (list end) in 8/8 → the fair run should extend those ranges once. The historical rows are seed-0 OOF and were part of
earlier selection decisions (selection-optimistic). Caveat: CPU pilot, seed 0, 8 configurations; not the fair benchmark.

## 5. GPU training package (needs approval)

**Command per family** (after approval; resumable, write-once):

```powershell
.\.venv\Scripts\python.exe -m football_fatigue.m11_fair train --family lstm --approval <id> --device cuda
.\.venv\Scripts\python.exe -m football_fatigue.m11_fair train --family gru  --approval <id> --device cuda
.\.venv\Scripts\python.exe -m football_fatigue.m11_fair train --family svm_nystroem --approval <id>
.\.venv\Scripts\python.exe -m football_fatigue.m11_fair train --family hgb --approval <id>
```

**Outputs:** `data/processed/m11-fair-benchmark-v1/<family>/fold<k>.json` and `checkpoints/<family>_fold<k>_seed<s>.pt`. These are new files; Stage 4 / 4C checkpoints are never touched.

**Compute** [EST]:
- Per-epoch seconds are the measured GTX 1650 Ti values (GRU 1.24, LSTM 3.15 s/epoch at historical size).
- They are scaled by parameter count.
- Mean epochs are assumed to be half the 400 cap.

| Family | Runner as implemented (24 configs × 8 folds, single split) | + final 5 seeds | Full 3-inner-fold nesting | Successive-halving variant (not implemented) |
|---|---|---|---|---|
| GRU | ≈ 24.8 GPU-h | + 5.2 | 74.3 | 13.8 |
| LSTM | ≈ 79.3 GPU-h | + 16.5 | 237.9 | 44.1 |
| **Total GRU + LSTM** | **≈ 126 GPU-h** | | | |

- `--configs N` reduces the search (for example `--configs 8` is about one third of the tuning cost).
- SVM and HGB are CPU-hours.
- The larger figure compared with Phase 2's estimate (36 GPU-h) comes from the new search including 2-layer and 96-unit networks (up to about 260 k parameters).

**Recommendation [REC]:**
1. Approve LSTM and GRU with `--configs 12` first: ≈ 63 GPU-h including finals.
2. Then decide whether the remaining 12 configurations are needed, based on the edge flags and the best-epoch distribution.

## 6. What is still missing before a fair claim

1. Shift-augmented and variable-gap training distributions in the runner. Stage 4C data exist for config C; the runner currently trains on centred g0p80 only.
2. Texture / smoothing ablation inputs.
3. Successive halving, if the compute budget requires it.
4. GPU approval.

Until then, no new LSTM / SVM model exists for live inference.
