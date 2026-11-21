# M11 Stage 5: final held-out evaluation protocol (PROPOSED v1, NOT ACTIVE)

Status: **proposal awaiting approval.** Nothing in this document has been executed:
- no held-out data has been read or built;
- no final model has been trained;
- no threshold has been applied to held-out data.

Evidence: [Stage 4C report](m11_stage4c_robustness_and_model_selection_v1.md).

Once approved, every item below is frozen **before** any held-out file is opened. Any later change makes the evaluation exploratory
and must be reported as such.

## 1. Held-out data

- **Matches:** the 8 internal held-out matches of `m11-stage2-split-v1`: 3822, 3824, 3834, 3839, 3844, 10510, 10514, 10517.
- **Frozen split:** `split.json` SHA-256 `66b2321a58f3845194757f943c2d36b92f18695322fb5490917f8a0144ff56c0`.
- **Unchanged and not used:** the official FIFA validation and test data.
- **Builds (once):** with the unchanged Stage 3 code (`samples.py`, SHA-256 `8930014ff2a8783d8366c540a9a1524f1723107418e21374d7fbcde9e354a7b9`)
  and the Stage 4C shift builder (`shiftaug.py`, the version hashed in §8), into new versioned directories:
  - the four Stage 3 tables (`g0p80`, `g0p40`, `g1p60`, `g0p80_shift`);
  - the six-shift augmentation set.
- **Required code change (needs approval):** both builders refuse held-out matches. They need an explicit, logged
  `--final-heldout` mode that writes to separate `*-heldout-v1` directories and records its own manifest.
- **Same rules as development:** Stage 2 candidate acceptance, label verification, independence and overlap groups. No held-out
  sample is filtered by model output, tracking status or crossing class.

## 2. Selected models (at most two)

| | Candidate 1 (primary) | Candidate 2 (reference) |
|---|---|---|
| Model | GRU gap-last pair encoder (`gru_gaplast`, 43,585 parameters) | Symmetric HistGradientBoosting on the 36 motion features (`hgb`) |
| Inputs | (2, 30, 4) incoming (reversed) and outgoing sequences: x/10 m, y/10 m, mask, dt; plus gap length | `motion.features` (perm_* antisymmetric + context), from the same 7 input fields |
| Training distribution | **Config C**: each epoch, each encounter centred with probability 0.5, else a uniformly drawn valid shifted version (±0.1, ±0.2, ±0.3 s) | **Config C** as sample weights: centred 0.5, each valid shifted version 0.5 / n_shifts; plus antisymmetric order copies |
| Optimisation | AdamW lr 1e-3, weight decay 1e-4, dropout 0.1, gradient clip 1.0, batch 256; augmentation: reflection, both order exchanges | learning_rate 0.05, max_iter 400, max_leaf_nodes 31, min_samples_leaf 40, l2 1.0, no early stopping (inner-CV choice in 5 / 8 folds, lowest mean inner log loss) |
| Stopping | Early stopping, patience 10 (improvement > 1e-4) on 3 early-stop matches, at most 160 epochs | – |
| Seeds | 0 (primary), 1, 2 (variance) | 0 (deterministic) |
| Calibration | None (development ECE 0.006–0.009) | None (development ECE 0.009) |
| Decision | SWAPPED if p > 0.5 (antisymmetric model, no threshold tuning) | same |
| Abstention | Predict only if max(p, 1 − p) ≥ **0.6425** (5 % target); secondary 0.8525 (2 %) | ≥ **0.7875** (5 %); secondary 0.9029 (2 %) |

Abstention thresholds are pooled development OOF thresholds (independent subset, 32 matches, seed 0) from Stage 4C `metrics/v1`.

## 3. Final training (before opening held-out data)

- Train each candidate once on **all 32 train/CV matches**: centred `g0p80` plus `m11-stage4c-shiftaug-v1`.
- **GRU early-stop matches:** the first 3 of `folds._rank(all 32 matches, "m11-stage3-folds-v1|early|final")`; training uses the
  other 29.
- **Record and hash before the held-out build:** model files (GRU state dicts, HGB joblib), training histories, best epochs and the
  exact code version.
- **Sanity check (development only):** the final models' predictions on the 32 development matches are in-sample. They are
  reported for integrity only, never as performance.

## 4. Evaluation procedure (run once)

1. Build the held-out data (§1); verify manifests, label checks and perturbation tests.
2. Load the frozen models; verify their hashes.
3. Predict every held-out table once. Store predictions write-once with hashes.
4. Compute the metrics in §5. No retraining, recalibration, threshold change or model choice after step 3.
5. Report both candidates in full, whichever looks better. Their paired difference is a secondary, descriptive result.

## 5. Reporting metrics (independent subset; 95 % CIs by match bootstrap, 2,000 resamples, 8 matches)

**Primary:**
- Balanced accuracy and log loss on centred 0.8 s gaps (`g0p80`).
- Balanced accuracy on shifted 0.8 s gaps (all six shifts).

**Secondary:**
- ROC AUC, Brier, ECE (10 bins), confusion matrix, F1 (SWAPPED).
- Per-shift BA.
- Per-match BA (8 values).
- Seed variance (GRU).
- Paired GRU − HGB differences.

**Strata (pre-specified):**
- `status_in_gap`, with **both fully VISIBLE** reported alongside the overall figure as the conservative estimate. Proposed as
  co-primary; your decision.
- ESTIMATED share of input points.
- Crossing class.
- d_min band (including < 5 cm).
- Speed band.
- Team relation.
- Overlap-group size.

**Transfer (descriptive, models trained at 0.8 s only):** `g0p40`, `g1p60`, with log loss and ECE.

**Abstention:** coverage, realised error and recovered share at the frozen thresholds, centred and shifted; AURC.

**Pre-specified interpretation:**
- A held-out result is consistent with development if it falls inside the development 95 % CI ± 0.02 BA.
- The GRU is preferred as the deployed model only if its held-out centred **and** shifted BA exceed HGB's, with paired CIs excluding 0.
- If the CIs overlap, HGB is preferred as the simpler, more robust model.

## 6. What the evaluation cannot show

- Generalisation to other providers, tournaments or trackers.
- Robustness to gap lengths other than 0.8 s; these are reported only as transfer.
- Freedom from provider-texture effects (Stage 4C §1).
- Correctness of provider reference identities.

## 7. Development reference values (for comparison only)

| Metric | GRU C 160 (3 seeds) | HGB C |
|---|---|---|
| Centred BA | 0.926 ± 0.003 | 0.861 [0.855, 0.867] |
| Shifted BA | 0.910 ± 0.001 | 0.878 [0.874, 0.883] |
| Both-VISIBLE BA | 0.845 ± 0.006 | 0.793 [0.772, 0.812] |
| Centred LL / ECE | 0.178 / 0.007 | 0.298 / 0.009 |
| Coverage at 5 % error (nested) | 0.928–0.948 | 0.717 |
| 0.4 s / 1.6 s transfer BA | 0.840 / 0.745 | 0.928 / 0.803 |

These are optimistic development estimates (Stage 4C §8).

## 8. Artifact hashes (SHA-256)

| Artifact | SHA-256 |
|---|---|
| `data/processed/m11-stage2-split-v1/split.json` | 66b2321a58f3845194757f943c2d36b92f18695322fb5490917f8a0144ff56c0 |
| `data/processed/m11-stage2-candidates-v1/manifest.json` | b6ab035520aa5d81676e8c73ce0d3415e7cd07f96849c207fead7741dde38d19 |
| `data/processed/m11-stage3-samples-v1/manifest.json` | 6016fa6c00814ef4fa3d94f5e2c1c66a23d89a5cd35d91277376ce0c830ade8f |
| `data/processed/m11-stage4c-shiftaug-v1/manifest.json` | 926584d773aae51c54a16c569f6fadb26afb448c3ca3b82b2a947430dd8aa598 |
| `data/processed/m11-stage4-benchmarks-v1/folds.json` | d9c07ea4f449e5903f0313a285e0a77ade30df167f86b03688d58389c131ff6d |
| `data/processed/m11-stage4-benchmarks-v1/gates/gates.json` | 10b69bd00fe69a893eebb2826b46df373210518c4f498946d8b161a975716792 |
| `data/processed/m11-stage4-benchmarks-v1/gates/controls.json` | b547bce2899e6dcf8a5980389529e6ecbcdb45f6721692b4717b96172da4fa12 |
| `data/processed/m11-stage4c-robustness-v1/metrics/v1/comparison.json` | ec10a7c68ef99fda94ae31ce386f3c0674e9a9ef9d7d2c16b0ac4b4e26cae4ac |
| `data/processed/m11-stage4c-robustness-v1/audit/audit_v2.json` | c7f7835f6debba36aca0aa2cfc1bb76bc54eba348f38b2503f0e4372bbca3aff |
| `data/processed/m11-stage4c-robustness-v1/audit/texture_probe_v2.json` | 5efa23a28ab5e42e024636b42e5f5315c754e78136f011a67701f5ec5d44e7c9 |
| `src/football_fatigue/window_intersections/samples.py` | 8930014ff2a8783d8366c540a9a1524f1723107418e21374d7fbcde9e354a7b9 |
| `src/football_fatigue/window_intersections/candidates.py` | 3a058d4d61b5a830bde87295fb68023119b6025a47bd2867ae6a1201b5b5a862 |
| `src/football_fatigue/m11_recovery/motion.py` | 73776b963af8b2c63cecda1442252394f619e0353da73532a6b286889bb49178 |
| `src/football_fatigue/m11_benchmark/data.py` | 3f27c920fb54bb28d8ad8cd9f4d36d1c92a6e03f4bcd39785136e1b31dbb9ae0 |
| `src/football_fatigue/m11_benchmark/models.py` | 3712efa9aa69cbd9de3c3a40faaca7dddccdcfff58571078ca24ea4c28750acb |
| `src/football_fatigue/m11_benchmark/folds.py` | e8161d9887f4acf8a33686f5dc3affa9b25c414f9d7cf7305ec6f9282f4d8222 |
| `src/football_fatigue/m11_benchmark/metrics.py` | fb1959b7c24db641d04c266f8c33ee36a0d4510bb24509fefcc69f61f169d569 |
| `src/football_fatigue/m11_benchmark/shiftaug.py` | 869e5466b5f55048c4d8ace42644ed65ee26311f17dce45ab38b543bc28fe184 |
| `src/football_fatigue/m11_benchmark/stage4c.py` | a2a1f532687547fb893678a7243218f9e7490da12110db7e3234957b0d6bd8b0 |
| `src/football_fatigue/m11_benchmark/stage4c_eval.py` | 5fa0355beb2b7d1bf6035b1c634a7c639a6c4ee968045668f047a457df7c70f5 |

The code is uncommitted. Hashes must be re-verified, and the final training code (§3) added and hashed, once the code is committed
and the protocol approved.

## 9. Approval checklist

- [ ] Candidates and configurations (§2)
- [ ] Final training on all 32 matches (§3)
- [ ] Guarded held-out build mode (§1)
- [ ] Primary metrics; VISIBLE stratum as co-primary or not (§5)
- [ ] Decision rule between candidates (§5)
