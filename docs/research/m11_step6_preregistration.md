# M11 Step 6: preregistration of the frozen validation evaluation

Written 2026-11-06, **after** the freeze (`data/processed/m11-frozen-v1`) and **before** any validation match was processed. At
this point no validation catalogue, benchmark, prediction or outcome existed.

The machine-readable specification is `data/processed/m11-frozen-v1/spec.json` (SHA-256
`6fa1df5c0067c1671aa35329dfcfc6c55794a3059dde3777b8ebfa423d032d49`). This file restates it.

## Splits (frozen manifest `m11-encounters-v1/splits.json`, 40 / 12 / 12)

| Split | Matches | Role |
|---|---|---|
| Development (used) | 3812, 3816, 10506, 3814, 3847, 3836, 3833, 3815, 3851, 3829 | Fitting the frozen model |
| Validation | 3817, 3820, 3825, 3827, 3832, 3841, 3842, 3846, 3856, 10508, 10511, 10513 | Evaluated once |
| Test | 3813, 3845, 3848, 3850, 3853, 3854, 3857, 10504, 10505, 10509, 10515, 10516 | **Locked**, never accessed |

## Frozen model

**Identity.**
- File: `m11-frozen-v1/model.joblib`
- SHA-256: `629cca77c4692c3e37d3d6c6ce0c6eff7421c4ac2fbd70c063d7cdd3abeb9187`

**Estimator.** `RandomForestClassifier`:
- 300 trees;
- `max_features` 0.33, `min_samples_leaf` 5;
- class weight `balanced_subsample`;
- seed 0.

These are the Step 5 hyperparameters, chosen in every Step 5 fold with and without `conf_*`. No new search was run.

**Training data.** The clean high-tier pairs (no ESTIMATED endpoints) of the ten development matches.

**Features.** 51: the Step 4 core set and the 14 Step 5 football-aware features.
- Excluded: `conf_in`, `conf_out` (provider confidence codes).
- Not included: the FOOTBALLTrace-inspired features.
- The exact list is `spec.json → model.features`.

**Calibration.** Platt (a logistic fit on logit p), fitted on inner grouped out-of-fold predictions. The groups are {3812, 3816,
3836, 10506}, {3814, 3829, 3847} and {3815, 3833, 3851}. Result: a = 1.1632, b = −0.0478.

**Decision.** Unchanged from Steps 4 and 5:
- one-to-one assignment that maximises summed log-odds over reachable pairs;
- **MATCHED** if assigned, p ≥ threshold and margin ≥ margin;
- **UNMATCHED** only when no outgoing tracklet is reachable;
- otherwise **UNCERTAIN**.

## Operating points: check and resolution (development data only)

**The resolution rule**, fixed in `validate.freeze_core` before the freeze was run:
- keep a preselected point if its inner out-of-fold false association meets its target;
- otherwise use the inner selection for that target, which has maximum inner coverage under the target.

| Point | Preselected | Inner coverage / false assoc. at the preselected point | Compatible | **Frozen** |
|---|---|---|---|---|
| Conservative (target 5%) | 0.87 / margin 1.0 | 0.272 / 0.046 | yes | **0.87 / 1.0** |
| Relaxed (target 10%) | 0.75 / margin 1.0 | 0.423 / **0.103** | **no** | **0.76 / 1.0** (inner 0.409 / 0.0995) |

**Why the relaxed point moved.** The preselected relaxed point came from the Step 5 model *with* `conf_*`. Without them, it
misses the 10% target on the inner folds by 0.3 percentage points. Under the rule it moves by 0.01. Nothing about validation was
known when this happened.

## Development reference (leave-one-match-out at the frozen points)

| Measure | Conservative 0.87 / 1.0 | Relaxed 0.76 / 1.0 |
|---|---|---|
| Coverage (95% match CI) | 0.269 [0.254, 0.282] | 0.399 [0.383, 0.413] |
| False association (95% match CI) | 0.043 [0.026, 0.060] | 0.086 [0.067, 0.101] |

Pairwise: AUC 0.839, top-1 0.718, ECE 0.0075, Brier 0.162.

## Comparators (all frozen before validation; SHA-256 values in `spec.json`)

**Primary baseline.** The frozen Step 4 RF: `m11-ml-v1/models/final/rf.joblib`, threshold 0.86, margin 0.5, uncalibrated.

**The other frozen Step 4 models:**
- `rf_ft`: with FOOTBALLTrace-inspired features; threshold 0.84;
- `lr` and `lr_ft`: threshold 0.98;
- `lr_distance`: threshold 0.99.

**The Step 5 model with `conf_*`** (`m11-ml-v2` final) at its own points: 0.87 / 1.0 and 0.75 / 1.0.

**The Step 2–3 baselines**, as produced by the unchanged benchmark builder:
- nearest;
- constant velocity;
- Hungarian + constant velocity;
- chaining;
- E0-Adapted, the FOOTBALLTrace-*inspired* matcher.

**The published FOOTBALLTrace**, as E0-Literal protocol L. It is evaluation-side only: it needs provider-labelled trajectories
and an operator, and it is not a matcher.

## Metrics

All metrics are agreement with the provisional provider reference.
- **A. Pairwise:** ROC-AUC, average precision, Brier, ECE, log-loss and top-1 per incoming tracklet, over high-tier pairs.
- **B. Encounter-level:** correct and wrong match rates per incoming tracklet.
- **C. Selective:** coverage (MATCHED / incoming) and false association (wrong / MATCHED).
- **D. Complete encounters:** the share of instances in which every incoming outcome is correct.
- **E. Strata:** gap (< 1.5, 1.5–2.5, 2.5–4, ≥ 4 s), sharp turn, crossing, player count (2 / 3+) and tier. The lower tier is
  exploratory, and natural suspicious cases are reported as decision counts only.
- **Uncertainty:** a match-level bootstrap of the 12 validation matches (2,000 resamples, seed 0), paired for differences.

## Criteria (high tier, pooled over the 12 validation matches)

- **Primary.** Both conditions must hold at the relaxed point (0.76 / 1.0):
  - false association ≤ 0.10;
  - coverage higher than the frozen Step 4 RF, with the lower bound of the 95% match-bootstrap CI of the paired coverage
    difference > 0.

  If either fails, the primary criterion fails.
- **Secondary**, reported separately: at the conservative point (0.87 / 1.0), false association ≤ 0.05.
- **Reported, not gated:**
  - confidence intervals;
  - per-match rates and their spread;
  - development against validation;
  - calibration shift;
  - strata;
  - failure cases.

**Commitments:**
- no threshold, margin, calibration, feature or model change after validation is opened;
- no retraining on validation;
- the test split stays locked.

## Execution order

1. Step 1 catalogues for the 12 validation matches, using the unchanged pipeline with `--allow-validation`, into
   `m11-validation-v1/encounters`.
2. The unchanged Step 2–3 benchmark, run by `validate.build_validation_match`, into `m11-validation-v1/matching`. Only the
   eligibility audit is looked at.
3. `predict`: reads only `root.instances_b.jsonl` and the manifest, and writes predictions with checksums.
4. `score`: verifies the prediction checksums first, then reads the references.
