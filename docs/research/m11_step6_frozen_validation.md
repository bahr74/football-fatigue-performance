# M11 Step 6: frozen classifier and independent validation (12 validation matches)

Status: 2026-11-06.
- **Inputs:** the Step 3, 4 and 5 reports.
- **Preregistration:** `docs/research/m11_step6_preregistration.md` (SHA-256 `a0c606b1…b126c2`), written before any validation
  match was processed.
- **Outputs:**
  - `data/processed/m11-frozen-v1/` (26 MB);
  - `data/processed/m11-validation-v1/{encounters, matching, predictions, scoring}` (1.3 GB, mostly Step 1 caches).

**Not done:**
- no test match was accessed;
- no model, threshold or calibration was fitted or changed on validation data;
- no new model or feature search;
- nothing was committed.

Every rate is **agreement with the provisional provider identity reference**, on task B. Task B withholds the encounter
interval, which is a controlled removal of information, not a real occlusion.

## Answer first

The frozen model **transfers to the 12 unseen validation matches** and **meets both preregistered criteria**. Validation is pooled
over 5,039 incoming tracklets in 2,422 high-tier encounters.

| | Development (leave-one-match-out, same frozen points) | **Validation** | Preregistered criterion |
|---|---|---|---|
| Pairwise ROC-AUC | 0.839 | **0.848** | – |
| Top-1 per incoming | 0.718 | 0.727 | – |
| ECE (calibrated) | 0.0075 | 0.0072 | – |
| **Relaxed point (0.76 / 1.0): coverage** | 0.399 | **0.428** [0.414, 0.445] | > frozen Step 4 RF (0.154); difference CI [+0.264, +0.283] ✔ |
| **Relaxed point: false association** | 0.086 | **0.083** [0.072, 0.094] | ≤ 0.10 ✔ (**primary met**) |
| Conservative point (0.87 / 1.0): coverage | 0.269 | 0.294 [0.281, 0.309] | – |
| Conservative point: false association | 0.043 | **0.047** [0.041, 0.054] | ≤ 0.05 ✔ (**secondary met**, point estimate) |

Brackets are 95% match-bootstrap intervals (12 matches, 2,000 resamples).

**Caveats:**
- The conservative point's interval reaches 0.054, so its 5% target is not met with certainty.
- Per match, the relaxed false association ranges from 0.042 to 0.118. Two matches exceed 10%: 10511 at 0.118 and 3841 at 0.104.
- Long gaps (2.5 s or more) and encounters with 3+ players remain essentially unsolved.

## 1. Frozen model specification (`m11-frozen-v1/spec.json`, model SHA-256 `629cca77…eb9187`)

| Item | Frozen value |
|---|---|
| Estimator | `RandomForestClassifier`: 300 trees, `max_features` 0.33, `min_samples_leaf` 5, `class_weight="balanced_subsample"`, seed 0. These are the Step 5 hyperparameters, chosen in every Step 5 fold with and without `conf_*`; no new search was run |
| Training data | Clean high-tier pairs (no ESTIMATED endpoint) of the 10 development matches: 3812, 3816, 10506, 3814, 3847, 3836, 3833, 3815, 3851, 3829 |
| Features (51) | The Step 4 core set and the 14 Step 5 football-aware features, **minus `conf_in` and `conf_out`**. No FOOTBALLTrace-inspired features. The exact list is `spec.json → model.features`, and `validate.FROZEN_FEATURES` in code |
| Calibration | Platt (a logistic fit on logit p), fitted on inner grouped out-of-fold predictions. Groups: {3812, 3816, 3836, 10506}, {3814, 3829, 3847}, {3815, 3833, 3851}. Result: a = 1.1632, b = −0.0478 |
| Decision rule | One-to-one assignment that maximises summed log-odds over reachable pairs. **MATCHED** if assigned, p ≥ threshold and margin ≥ margin; **UNMATCHED** only if nothing is reachable; otherwise **UNCERTAIN** |
| Conservative point | Threshold **0.87**, margin **1.0** (target 5%) |
| Relaxed point | Threshold **0.76**, margin **1.0** (target 10%; see below) |

**Checks of the earlier sensitivity experiment.**
- The Step 5 no-`conf` sensitivity run gave AUC 0.839 and coverage / false association of 0.278 / 0.045 and 0.423 / 0.095.
- The same parameters were chosen in all of its folds.
- The freeze reproduces this procedure: inner out-of-fold AUC 0.838 and development leave-one-match-out AUC 0.839.

**One operating point moved**, resolved on development data only, before validation existed.
- The preselected relaxed point 0.75 / 1.0 came from the Step 5 model *with* `conf_*`.
- For the frozen no-`conf` model, its inner out-of-fold false association is **0.1028**, which misses the 10% target.
- The resolution rule was coded in `validate.freeze_core` before the freeze ran: keep a point if it meets its target, otherwise
  take the inner selection. It moved the relaxed point to **0.76 / 1.0** (inner 0.409 / 0.0995).
- The conservative point 0.87 / 1.0 was compatible: inner 0.272 / 0.046.
- Both checks are recorded in `spec.json → operating_check`.

## 2. Preregistered protocol

The protocol is in full in `m11_step6_preregistration.md` and `spec.json`.

**Splits** (frozen manifest, 40 / 12 / 12):
- development: the 10 matches listed in section 1;
- validation: 3817, 3820, 3825, 3827, 3832, 3841, 3842, 3846, 3856, 10508, 10511, 10513;
- test, locked: 3813, 3845, 3848, 3850, 3853, 3854, 3857, 10504, 10505, 10509, 10515, 10516.

**Metrics.** A: pairwise ROC-AUC and top-1. B: correct and wrong match rates. C: coverage and false association. D: complete
encounters. E: strata.

**Criteria.**
- **Primary:** at the relaxed point, pooled high-tier false association ≤ 0.10, and coverage above the frozen Step 4 RF with the
  lower bound of the paired match-bootstrap CI > 0.
- **Secondary:** at the conservative point, false association ≤ 0.05.

**Execution, in order and enforced in code:**
1. **Freeze.** It refuses any non-development match.
2. **Build.** The unchanged Step 1 pipeline with `--allow-validation`, then the unchanged Step 2–3 builder through
   `validate.build_validation_match`. Only the eligibility audit was looked at.
3. **Predict** (`validate.run_predict`):
   - it reads only `root.instances_b.jsonl` and the manifest; the predictions manifest records `reference_files_read: false`,
     `model_fitting: false` and `threshold_tuning: false`;
   - it checks the SHA-256 of the frozen model and of every comparator before use;
   - it writes probabilities and decisions with checksums.
4. **Score** (`validate.run_score`): it verifies the prediction checksums and the frozen-model hash first, and only then reads
   references.

Predictions (`m11-validation-v1/predictions`) and references (`m11-validation-v1/matching/*/root.references.jsonl`) are stored
separately. Test matches are refused by every entry point.

## 3. Validation matches and eligibility

Events are candidate encounter events from the unchanged Step 2 audit:

| Match | Events | High | Lower | Suspicious | Rejected |
|---|---|---|---|---|---|
| 3817 | 941 | **76** | 567 | 0 | 298 |
| 3820 | 1,107 | 223 | 560 | 7 | 317 |
| 3825 | 1,016 | 143 | 530 | 2 | 341 |
| 3827 | 1,211 | 267 | 611 | 1 | 332 |
| 3832 | 1,100 | 143 | 605 | 3 | 349 |
| 3841 | 937 | **71** | 555 | 5 | 306 |
| 3842 | 1,175 | 297 | 516 | 0 | 362 |
| 3846 | 959 | 197 | 484 | 6 | 272 |
| 3856 | 1,229 | 219 | 630 | 0 | 380 |
| 10508 | 1,451 | 343 | 656 | 0 | 452 |
| 10511 | 1,310 | 288 | 605 | 3 | 414 |
| 10513 | 1,017 | 155 | 535 | 1 | 326 |
| **Total** | 13,453 | **2,422** | 6,854 | 28 | 4,149 |

**Labelled pairs.**
- High tier: 5,039 incoming tracklets and 10,785 scored pairs.
- Lower tier, exploratory: 14,816 incoming and 33,568 pairs.

**Small high tiers.** 3817 and 3841 have small high tiers (71–76 instances), so their per-match rates are noisy.

## 4. Results at both operating points (validation, high tier, pooled)

**B, C and D** (rates per incoming tracklet):

| Method | Correct | Wrong | Uncertain | Coverage | False assoc. | Complete enc. |
|---|---|---|---|---|---|---|
| **Frozen, relaxed (0.76 / 1.0)** | **0.392** | **0.035** | 0.572 | 0.428 | **0.083** | 0.346 |
| **Frozen, conservative (0.87 / 1.0)** | 0.280 | 0.014 | 0.706 | 0.294 | 0.047 | 0.227 |

- UNMATCHED is 0 for both; no high-tier incoming tracklet lacked a reachable candidate.
- Conflicts are 0 for both (the assignment is one-to-one).

**A, pairwise:**

| | ROC-AUC | Average precision | Top-1 | Brier | ECE | Log-loss |
|---|---|---|---|---|---|---|
| High tier | 0.848 | 0.838 | 0.727 | 0.158 | 0.007 | 0.469 |
| Lower tier (exploratory) | 0.777 | 0.740 | 0.656 | 0.190 | 0.019 | 0.556 |

**E, strata.** Coverage / false association at the relaxed point, then the conservative point:

| Stratum | n incoming | Relaxed | Conservative | Frozen Step 4 RF |
|---|---|---|---|---|
| Gap < 1.5 s | 1,897 | **0.687 / 0.073** | 0.529 / 0.039 | 0.313 / 0.047 |
| Gap 1.5–2.5 s | 1,793 | 0.428 / 0.089 | 0.255 / 0.066 | 0.102 / 0.104 |
| Gap 2.5–4 s | 935 | 0.090 / **0.179** | 0.021 / 0.050 | 0.002 / 0 |
| Gap ≥ 4 s | 414 | 0.002 / 1.0 (1 match) | 0 / – | 0 / – |
| 2 players | 4,540 | 0.467 / 0.081 | 0.324 / 0.046 | 0.171 / 0.061 |
| 3+ players | 499 | 0.070 / **0.200** | 0.024 / 0.167 | 0.004 / 0 |
| Sharp turn | 3,241 | 0.317 / **0.122** | 0.191 / 0.070 | 0.079 / 0.114 |
| No sharp turn | 1,798 | 0.627 / 0.048 | 0.480 / 0.031 | 0.291 / 0.034 |
| Crossing | 2,088 | 0.409 / 0.064 | 0.268 / 0.039 | 0.140 / 0.044 |
| No crossing | 2,951 | 0.441 / 0.095 | 0.312 / 0.052 | 0.165 / 0.070 |
| **Lower tier** (exploratory) | 14,816 | 0.235 / **0.113** | 0.135 / 0.075 | 0.070 / 0.064 |

**Natural suspicious cases** (unverified, 73 incoming): relaxed 4 MATCHED, 67 UNCERTAIN, 2 UNMATCHED. They are reported as decision
counts only.

## 5. Development versus validation

| | Development (leave-one-match-out, frozen config and points) | Validation | Shift |
|---|---|---|---|
| ROC-AUC, pooled | 0.839 | 0.848 | +0.009 |
| ROC-AUC, per match | 0.815–0.867 | 0.795–0.913 | 10511 is lowest (0.795) |
| Top-1 | 0.718 | 0.727 | +0.009 |
| ECE / Brier | 0.0075 / 0.162 | 0.0072 / 0.158 | none |
| Relaxed coverage / false assoc. | 0.399 / 0.086 | 0.428 / 0.083 | +0.029 / −0.003 |
| Relaxed false assoc., per match | 0.027–0.133 | 0.042–0.118 | similar spread |
| Conservative coverage / false assoc. | 0.269 / 0.043 | 0.294 / 0.047 | +0.025 / +0.004 |
| Conservative false assoc., per match | 0–0.092 | 0.022–0.067 | similar |

**Calibration shift.** The validation reliability curve tracks the diagonal in every bin: mean p against observed rate is
0.966 / 0.964 in the top bin, 0.550 / 0.534 in the middle and 0.026 / 0.023 at the bottom. There is no calibration shift, and the
frozen Platt map transfers.

## 6. Baseline comparison (validation, high tier, each method at its frozen operating point)

| Method | What it is | Correct | Wrong | Coverage | False assoc. [95% CI] | Complete |
|---|---|---|---|---|---|---|
| Nearest | Distance only, no one-to-one constraint (1,421 conflicts) | 0.412 | 0.351 | 0.763 | 0.461 | 0.041 |
| Constant velocity | Extrapolation, no one-to-one constraint (1,082 conflicts) | 0.452 | 0.331 | 0.783 | 0.423 | 0.149 |
| Hungarian + CV | One-to-one, gate only | 0.412 | 0.216 | 0.628 | 0.344 [0.330, 0.360] | 0.396 |
| Chaining | One-to-one | 0.423 | 0.223 | 0.646 | 0.345 [0.323, 0.369] | **0.420** |
| **E0-Adapted** | FOOTBALLTrace-*inspired* matcher (Step 3) | 0.290 | 0.197 | 0.487 | 0.403 [0.383, 0.422] | 0.283 |
| Step 4 LR / LR+FT | Frozen, threshold 0.98 | 0.015 | 0.000 | 0.015 | 0.026 / 0.038 | 0.005 |
| Step 4 LR distance | Frozen proximity control | 0 | 0 | 0 | – | 0 |
| **Step 4 RF** (primary baseline) | Frozen, threshold 0.86, margin 0.5 | 0.145 | 0.009 | 0.154 | 0.060 [0.045, 0.078] | 0.081 |
| Step 4 RF + FT features | Frozen, threshold 0.84 | 0.163 | 0.011 | 0.174 | 0.064 [0.050, 0.081] | 0.094 |
| Step 5 with `conf_*`, relaxed | Frozen, 0.75 / 1.0 | 0.399 | 0.039 | 0.438 | 0.089 [0.078, 0.100] | 0.355 |
| **Frozen, relaxed** | 0.76 / 1.0 | 0.392 | **0.035** | 0.428 | **0.083** [0.072, 0.094] | 0.346 |
| **Frozen, conservative** | 0.87 / 1.0 | 0.280 | 0.014 | 0.294 | 0.047 [0.041, 0.054] | 0.227 |

**The published FOOTBALLTrace (protocol L, evaluation-side).**
- It was applied to the 2,270 provider-labelled two-player high-tier pairs.
- It **always** exchanges the trajectory tails at its breakpoint, so its agreement with a no-swap reference is 0, as in Step 3.
- Its breakpoint fell inside the interval in 50% of cases (min-angle rule).
- It is not a matcher. It needs labelled trajectories and an operator, and it cannot be compared in the tables above.

**Paired differences against the frozen Step 4 RF** (match bootstrap):
- **Frozen, relaxed:** coverage +0.264 to +0.283; false association +0.003 to +0.038. More commitments bring slightly more false
  association, but it stays within the 10% budget.
- **Frozen, conservative:** coverage +0.130 to +0.149; false association −0.033 to +0.003. This is more coverage at no measurable
  cost.
- **Step 4 RF + FT against RF:** coverage +0.016 to +0.023, which comes only from its lower frozen threshold (0.84 against 0.86).
  Pairwise AUC is 0.786 against 0.787: **the paper-inspired features add nothing on validation either.**

**Why coverage differs.**
- **The rule-based matchers abstain rarely,** so they commit 49–78% of tracklets at 34–46% false association.
- **The ML models abstain explicitly** whenever the calibrated probability or the assignment margin is weak.
- **At the relaxed point** the frozen model makes **93–95% as many correct matches** as Hungarian or chaining (0.392 against
  0.412 and 0.423), with **16% of their wrong matches** (0.035 against 0.216 and 0.223).
- **Complete-encounter accuracy favours methods that commit everything,** and chaining is still highest on it (0.420 against 0.346).
- **The Step 5 model with `conf_*`** behaves essentially the same as the frozen model (AUC 0.848 for both). That confirms that
  removing the confidence codes cost nothing.

## 7. Per-match uncertainty (validation, high tier)

| Match | Pairwise AUC | Relaxed coverage / false assoc. (MATCHED) | Conservative coverage / false assoc. | Step 4 RF |
|---|---|---|---|---|
| 3817 | 0.849 | 0.416 / 0.045 (67) | 0.280 / 0.022 | 0.174 / 0.036 |
| 3820 | 0.862 | 0.423 / 0.062 (192) | 0.302 / 0.044 | 0.152 / 0.043 |
| 3825 | 0.842 | 0.403 / 0.075 (120) | 0.279 / 0.036 | 0.164 / 0.061 |
| 3827 | 0.840 | 0.428 / 0.074 (242) | 0.300 / 0.035 | 0.141 / 0.113 |
| 3832 | 0.913 | 0.483 / 0.042 (144) | 0.319 / 0.032 | 0.178 / 0.019 |
| 3841 | 0.827 | 0.456 / **0.104** (67) | 0.306 / 0.067 | 0.150 / 0.091 |
| 3842 | 0.850 | 0.414 / 0.088 (261) | 0.281 / 0.056 | 0.130 / 0.037 |
| 3846 | 0.866 | 0.438 / 0.091 (176) | 0.326 / 0.046 | 0.162 / 0.031 |
| 3856 | 0.842 | 0.480 / 0.096 (219) | 0.325 / 0.054 | 0.184 / 0.048 |
| 10508 | 0.860 | 0.404 / 0.087 (287) | 0.263 / 0.043 | 0.139 / 0.051 |
| 10511 | **0.795** | 0.398 / **0.118** (238) | 0.273 / 0.067 | 0.140 / 0.095 |
| 10513 | 0.838 | 0.451 / 0.077 (143) | 0.319 / 0.050 | 0.199 / 0.095 |

- **Coverage is very stable:** 0.40–0.48 at the relaxed point and 0.26–0.33 at the conservative point. This is why the bootstrap
  intervals are narrow.
- **False association varies more.** At the relaxed point 10 of 12 matches are at ≤ 0.096, and at the conservative point 7 of 12
  are ≤ 0.05.
- **The targets hold for the pooled rate, not for every match.**

## 8. Overfitting and generalisation

- **No sign of overfitting to the development matches.** Validation AUC, top-1, calibration, coverage and false association all
  equal or slightly exceed the development leave-one-match-out values. The validation numbers sit inside the development per-match
  ranges.
- **The operating points carried over.** The thresholds were fixed on development only and land on their targets on validation:
  0.083 against a 0.10 target, and 0.047 against 0.05. The Step 5 finding, that thresholds transfer between matches for the tree
  models, holds on new matches.
- **The improvement over Step 4 replicates out of sample.** Coverage is ×2.8 at the relaxed point and ×1.9 at the conservative
  point, and the step-4 → step-5 gain (pairwise AUC +0.06) is the same as in development.
- **The slightly better validation numbers are within match-to-match variation.** The selection rule used development outer folds,
  so a small downward bias was possible, and none was observed.
- **What does not generalise is what never worked in development:**
  - gaps of 2.5 s or more: relaxed coverage 0.09, with false association 0.18;
  - 3+ players: coverage 0.07, false association 0.20;
  - sharp turns: false association 0.12 at the relaxed point;
  - the lower tier: false association 0.113 at the relaxed point.

  These are the limits of the information available in task B, not a generalisation failure.

## 9. Failure cases (frozen model, relaxed point, high tier)

**Counts.** 179 wrong matches and 1,176 abstentions whose top candidate was *not* the reference. In those abstentions, committing
would have produced an error.

**Most confident wrong matches** (all two-player encounters, all with assignment regret 0):

| Match / instance | p / reference-pair p | Pattern |
|---|---|---|
| 10508 `90cb6c526423f58b` | 0.998 / 0.030 | Sharp turn. The chosen continuation is smooth (CV meet-in-the-middle 0.40 m, Hermite RMS 1.0 m/s²) |
| 3827 `5a589d10a16e9376` | 0.993 / 0.607 | Sharp turn. Both continuations are plausible |
| 10513 `427a4334787674f9` | 0.990 / 0.651 | Crossing. Both are plausible |
| 3827 `ab365e0f76220c79` | 0.988 / 0.360 | Crossing. The chosen path is almost straight (6.9°, 0.26 m) |
| 3827 `66c73502cb05f722` | 0.969 / 0.029 | Crossing with a sharp turn. The chosen continuation has 1.6° heading change |
| 10513 `bac746c5a129d47e` | 0.961 / 0.254 | No difficulty flag. The chosen pair is near-perfect (0.09 m) |
| 3842 `6cb1ec40c3843914` | 0.958 / 0.054 | Crossing; 8.3 m within 1.2 s |

**Two kinds of failure:**
- **The reference pair also has a high probability** (3827 `5a58…`, 10513 `427a…`). Here the model is ambiguous, but the margin
  rule did not catch it.
- **The reference continuation is physically strained, while the chosen one is nearly perfect.** These are either genuine swerves
  that task B endpoints cannot reveal, or provider-reference errors.

**Possible reference errors.** **14** wrong matches meet the preregistered-style flag: p ≥ 0.8, reference-pair p ≤ 0.2 and chosen
CV meet-in-the-middle error < 0.5 m. They are listed in `scoring/summary.json → cases.possible_reference_errors`. They are
**candidates for review, not verified swaps.**

**Correct abstentions.** The highest-probability wrong candidates were blocked by the margin rule. Examples:
- 10513 `e778f0bda7f16cc7`: p 0.80, but the reference pair has 0.68 and the margin is 0.47;
- 3817 `a5b0d30b4401fdcd`: p 0.76 against 0.75;
- 3825 `8d3b7fbcb53b2c9d`: p 0.76, just below the threshold.

The assignment margin is doing its intended job.

## 10. Provider-label limitations

- **The reference is the provider's identity assignment.** It is provisional, not independent ground truth. Every "wrong match"
  is a disagreement with the provider.
- **Some disagreements may be provider errors.** Section 9 shows 14 flagged cases, and Step 5 found the same pattern in
  development. This would bias the false-association rate *upwards*. The effect cannot be measured without independent labels.
- **The high tier is a selection.** It excludes ESTIMATED endpoints, more than 6 players, encounters longer than 6 s and
  attribution problems. Validation high-tier sizes vary from 71 to 343 instances per match.
- **The lower tier has higher false association** (0.113 relaxed) and is exploratory only.
- **The known 3816 swap and other natural suspicious cases** remain unverified. On validation, the 73 suspicious incoming tracklets
  got 4 MATCHED decisions at the relaxed point.

## 11. Tests and integrity

**Code changes.**
- `src/football_fatigue/matching/validate.py` (new): freeze, validation build, predict and score.
- `__main__.py`: subcommands `freeze`, `valbuild`, `predict` and `score`.
- `__init__.py`: docstring.

No Step 1–5 module was changed, and no unrelated module was touched.

**New tests: `tests/test_matching_validate.py`, 10 tests.** They cover:
- **schema:** the frozen feature schema (51 features, no `conf_*`, no FT features);
- **features without references:**
  - `features_only` equals the training features without reading references;
  - identity fields do not change the features;
- **split membership:** test matches are refused from every function, and so are wrong-split matches;
- **operating points:** the preselected point is kept, or resolved by the rule;
- **frozen prediction:**
  - fits nothing — `fit_model`, `fit_platt` and `RandomForest.fit` are patched to fail;
  - is reproducible;
  - is one-to-one;
  - follows the MATCHED, UNMATCHED and UNCERTAIN rules;
- **criteria evaluation;**
- **outputs:**
  - written atomically, with no overwrite;
  - tampered files are detected by checksum;
- **raw data:** the raw snapshot detects writes;
- **references:** the visible-file loader works with no reference file present and rejects a bad checksum.

**Results.**
- `pytest tests/test_matching_validate.py`: 10 passed.
- Full suite: **724 passed, 6 skipped, 1 warning** (an unrelated vision deprecation warning).

**Integrity checks:**
- **Checksums:** every file in `m11-frozen-v1`, `predictions` and `scoring` matches its manifest. `m11-ml-v1` (31 files),
  `m11-ml-v2` (26 files) and all 22 matching manifests (10 development, 12 validation) verify unchanged.
- **Raw data:** the snapshot of `data/raw` (394 files, sizes and modification times) is identical at freeze, predict, score and
  now.
- **Test split:** no directory for any of the 12 test matches exists under `data/processed`.
- **Reproducibility:** recomputing match 3832's features from its instance file and applying the frozen model and calibrator
  reproduces all 3,709 stored probabilities exactly.
- **Harmless warning:** scoring printed a numpy all-NaN quantile warning. It comes from the bootstrap of `step4_lr_distance`, which
  never commits.

## 12. GO / NO-GO for the later final test evaluation

**GO, conditionally, for one final test-set evaluation** of exactly this frozen model, with the same operating points, after your
approval.
- **The evidence:** both preregistered validation criteria were met on 12 unseen matches, development and validation agree, and
  calibration did not shift.
- **Conditions:**
  - re-use `m11-frozen-v1` unchanged (SHA-256 `629cca77…`);
  - preregister the same primary and secondary criteria before opening test;
  - run test once;
  - report results as provider-reference agreement.

**NO-GO for these claims:**
- identity accuracy (no independent labels);
- real-occlusion or video-tracking performance (task B is a withheld-interval benchmark);
- per-match guarantees (two validation matches exceed 10% at the relaxed point);
- long gaps of 2.5 s or more, 3+ player encounters, and the lower tier.

**No change** to the model, features or thresholds is justified by these results. Making any change now would turn the test run
into a second validation.

## 13. Remaining need for independently reviewed identity annotations

They are still needed. Every number here is agreement with the provider, and the most confident disagreements look like either real
swerves or provider errors. A small, independently reviewed annotation set (video review by two annotators) would allow true
precision to be estimated for the first time. It should cover:
- the 14 flagged validation cases and a random sample of MATCHED decisions at both points;
- natural suspicious cases, including the known 3816 swap;
- a sample of 3+ player and long-gap encounters.

This needs approval and annotation effort. It does not use the test split.
