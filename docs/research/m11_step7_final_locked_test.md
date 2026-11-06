# M11 Step 7: final locked test evaluation of the frozen classifier (12 test matches, run once)

Status: 2026-11-06.
- **Inputs:** the Step 1–6 reports and the frozen Step 6 artifacts (`data/processed/m11-frozen-v1`).
- **Outputs:** `data/processed/m11-test-v1/{audit, encounters, matching, predictions, scoring}` (1.17 GB, mostly Step 1 caches).
- **Code:** `src/football_fatigue/matching/locked_test.py` (new) and `tests/test_matching_locked_test.py` (new). No other file
  changed.

**What was not done:**
- No retraining, recalibration, threshold change, feature change or model selection.
- No baseline was tuned.
- No synthetic swaps.
- No stage was re-run after scores were seen, and there is a single test evaluation.
- Nothing was committed.

Every rate here is **agreement with the provisional provider identity reference** on task B (withheld-interval trajectory
association). It is not verified identity accuracy, and a withheld interval is not proof of a real visual occlusion.

## Answer first

On the 12 locked test matches (5,555 incoming tracklets in 2,665 high-tier encounters), the frozen model **reproduced its
development and validation behaviour** and **met both preregistered criteria on the point estimate.**

| High tier | Development (LOMO) | Validation | **Test** [95% match bootstrap] |
|---|---|---|---|
| Pairwise ROC-AUC | 0.839 | 0.848 | **0.838** [0.826, 0.848] |
| Relaxed (0.76 / 1.0): coverage | 0.399 | 0.428 | **0.403** [0.386, 0.419] |
| Relaxed: false association | 0.086 | 0.083 | **0.088** [0.072, 0.107] |
| Conservative (0.87 / 1.0): coverage | 0.269 | 0.294 | **0.270** [0.255, 0.284] |
| Conservative: false association | 0.043 | 0.047 | **0.046** [0.034, 0.062] |
| Frozen Step 4 RF: coverage / false association | – | 0.154 / 0.060 | 0.152 / 0.056 |

- **Primary criterion: met.**
  - Relaxed false association is 0.088, within the 0.10 limit.
  - Coverage beats the frozen Step 4 RF: 0.403 against 0.152, with a paired difference CI of [+0.238, +0.264].
  - The upper end of the false-association interval (0.107) is above 0.10, so the 10% limit is not met with certainty.
- **Secondary criterion: met on the point estimate.** False association is 0.046 against the 0.05 target, but its interval reaches
  0.062.
- **Per-match variation is real.**
  - Relaxed point: 4 of 12 matches exceed 10% (10504 at 0.142, 10505 at 0.137, 3853 at 0.126, 10515 at 0.117).
  - Conservative point: 6 of 12 exceed 5%.
- **Still unsolved:** gaps of 2.5 s or more, encounters with 3+ players, and sharp turns of 90° or more.

## 1. Pre-test integrity audit

`locked_test audit` ran **before any test-match file was read or written**. It writes its record only if every check passes,
and all **25 of 25 passed**.

**Audit record** (`m11-test-v1/audit`):
- written 2026-11-06T21:11:21+0200;
- `audit.json` SHA-256 `9a2513ff…e34c29d`;
- manifest SHA-256 `23515929…69b975`.

| # | Check | Result |
|---|---|---|
| 1 | Split manifest unchanged since the freeze | SHA-256 `165fa3e0…aaf1` equals `spec.json` |
| 2 | The 12 test IDs are exact | 3813, 3845, 3848, 3850, 3853, 3854, 3857, 10504, 10505, 10509, 10515, 10516 |
| 3 | Test IDs disjoint from development (40) and validation (12) | yes. The specification also records `test_access: never` |
| 4 | Frozen model and specification hashes | model `629cca77…eb9187`; spec `6fa1df5c…d49`; frozen directory manifest verified |
| 5 | Feature schema, hyperparameters, calibration | 51 features with no `conf_*`; RF with 300 trees, `max_features` 0.33, leaf 5, `balanced_subsample`, seed 0; Platt a = 1.16317, b = −0.04777 |
| 5b | Assignment implementation | Re-predicting validation match 3832 reproduced all 3,709 stored Step 6 probabilities and 1,502 decision sets exactly |
| 5c | Frozen code unchanged since the Step 6 predictions | `baselines`, `benchmark`, `evaluation`, `footballtrace`, `ml`, `ml2` and `validate` hashes are equal. The Step 1 `encounters` code hashes equal the Step 6 build |
| 6 | Preregistration and specification checksums | `m11_step6_preregistration.md` = `a0c606b1…b126c2`. The Step 6 report is present (`38934ada…`) |
| 7 | No earlier test output or test-derived selection | No path named after a test match exists under any `m11-*` directory. `m11-test-v1` did not exist. The matches used by `m11-ml-v1`, `m11-ml-v2` and `m11-frozen-v1` contain no test ID. Frozen comparator hashes verified |
| 8 | RAW source unchanged | Raw snapshot of 394 files with digest `a6d610de…7856`, equal to the freeze and to Step 6 predict and score |
| 9 | Operating points | conservative 0.87 / 1.0; relaxed 0.76 / 1.0 (bundle = spec = Step 6) |
| – | Step 6 summary numbers re-read from saved outputs | Development: AUC 0.8391, relaxed 0.3994 / 0.0855, conservative 0.2692 / 0.0434. Validation: AUC 0.8479, relaxed 0.4279 / 0.0830, conservative 0.2941 / 0.0472. These match the table supplied with the Step 7 brief |

The audit also fixed, before any test data existed:
- the strata definitions (section 8);
- that the Step 6 primary and secondary criteria apply unchanged;
- the bootstrap settings (2,000 resamples, seed 0).

**First audit attempt.** It crashed on its own hyperparameter check: the frozen model is a one-step sklearn `Pipeline`, and the
check read the parameters from the wrong object. **Nothing was written, and no test file was read.** I corrected the check
(parameters are read from the pipeline's single RF step) and re-ran the audit, which passed.

## 2. Frozen model configuration and checksums

This is the Step 6 freeze, unchanged.

| Item | Value |
|---|---|
| Model | `m11-frozen-v1/model.joblib`, SHA-256 `629cca77c4692c3e37d3d6c6ce0c6eff7421c4ac2fbd70c063d7cdd3abeb9187` |
| Specification | `m11-frozen-v1/spec.json`, SHA-256 `6fa1df5c0067c1671aa35329dfcfc6c55794a3059dde3777b8ebfa423d032d49` |
| Estimator | `Pipeline([clf: RandomForestClassifier(300 trees, max_features 0.33, min_samples_leaf 5, balanced_subsample, seed 0)])` |
| Training | Clean high-tier pairs of 10 development matches: 3812, 3816, 10506, 3814, 3847, 3836, 3833, 3815, 3851, 3829 |
| Features | 51: the Step 4 core set and 14 football-aware features, without `conf_in` / `conf_out`, and without the FOOTBALLTrace-inspired features |
| Calibration | Platt on inner grouped out-of-fold predictions: a = 1.1632, b = −0.0478 |
| Decision | One-to-one log-odds assignment over reachable pairs. **MATCHED** if assigned, p ≥ threshold and margin ≥ 1.0; **UNMATCHED** only when nothing is reachable; otherwise **UNCERTAIN** |
| Operating points | Conservative 0.87 / 1.0; relaxed 0.76 / 1.0 |

**Execution order, enforced in code.** Every stage after the audit refuses to run unless the audit verified and passed.
1. **Audit.**
2. **Build:** the unchanged Step 1 catalogue (`encounters.catalogue.run_match`, the call the Step 1 CLI makes) and the unchanged
   Step 2–3 builder, per match. The builder writes both the matcher-visible files and the evaluation-only reference files, as in
   Step 6. Only eligibility counts were printed.
3. **Predict:** reads only `root.instances_b.jsonl` and the manifest. It checks completeness and writes `predictions`:
   - `predictions.jsonl` `6e78a88b…`;
   - `pair_probabilities.jsonl` `0742ec6b…`;
   - the manifest records `reference_files_read: false`, `model_fitting: false` and `threshold_tuning: false`.
4. **Score:** verifies the prediction checksums first, then reads references (`scoring/summary.json` `1bfd2fc6…`).

**Completeness check (prediction stage).** All 9 frozen methods decided all 19,116 incoming tracklets in 8,930 instances exactly
once, over 42,598 candidate pairs.

## 3. Test matches and benchmark sizes

Events are candidate encounter events from the unchanged Step 2 audit:

| Match | Events | High | Lower | Suspicious | Rejected | High-tier incoming |
|---|---|---|---|---|---|---|
| 3813 | 1,020 | 153 | 545 | 5 | 317 | 312 |
| 3845 | **483** | 109 | 225 | 0 | 149 | 226 |
| 3848 | 1,236 | 214 | 624 | 5 | 393 | 455 |
| 3850 | 1,098 | **398** | 391 | 2 | 307 | 840 |
| 3853 | 1,099 | 130 | 612 | 5 | 352 | 269 |
| 3854 | 1,174 | 366 | 503 | 0 | 305 | 755 |
| 3857 | 1,051 | 201 | 530 | 3 | 317 | 414 |
| 10504 | 1,042 | 179 | 560 | 9 | 294 | 375 |
| 10505 | 1,137 | 268 | 521 | 1 | 347 | 550 |
| 10509 | 1,208 | 257 | 602 | 3 | 346 | 537 |
| 10515 | 1,085 | 220 | 545 | 1 | 319 | 462 |
| 10516 | 1,096 | 170 | 566 | 7 | 353 | 360 |
| **Total** | 12,729 | **2,665** | 6,224 | 41 | 3,799 | **5,555** |

**Totals by tier:**

| Tier | Incoming | Pair rows |
|---|---|---|
| High | 5,555 | 11,869 |
| Lower | 13,456 | 30,430 |
| Natural suspicious (unverified) | 105 | – |

**Match 3845** has about half the candidate events of the other matches, although it has a similar number of frames (178,153). Its
high tier (109) is still in the normal range. I did not investigate the cause, and did not exclude or change anything because of
it.

## 4. Pairwise classification (high tier; separate from association)

Pairs are every incoming × outgoing pair of a high-tier instance (11,869 pairs, 5,555 positive). Precision, recall and F1 are pair
decisions at a probability cut-off. They are not encounter decisions.

| | ROC-AUC [match CI] | Avg. precision | Brier | ECE | Log-loss | Precision / recall / F1 at p ≥ 0.5 |
|---|---|---|---|---|---|---|
| **Frozen** | **0.838** [0.826, 0.848] | 0.826 | 0.163 | **0.007** | 0.485 | 0.735 / 0.715 / 0.725 |
| Step 5 with `conf_*` | 0.837 | 0.826 | 0.164 | 0.011 | 0.485 | 0.731 / 0.710 / 0.720 |
| Step 4 RF | 0.775 | 0.752 | 0.191 | 0.019 | 0.554 | 0.664 / 0.692 / 0.678 |
| Step 4 RF + FT features | 0.775 | 0.751 | 0.191 | 0.022 | 0.555 | 0.663 / 0.693 / 0.678 |
| Step 4 LR / LR + FT | 0.746 / 0.746 | 0.710 | 0.204 | 0.029 | 0.590 | 0.641 / 0.689 / 0.664 |
| Step 4 LR distance only | 0.535 | 0.503 | 0.249 | 0.035 | 0.692 | 0.485 / 0.656 / 0.558 |

**Frozen model at the operating thresholds (pairs, before assignment):**

| Threshold | Precision | Recall | F1 | True positives | False positives |
|---|---|---|---|---|---|
| p ≥ 0.87 | 0.953 | 0.258 | 0.406 | 1,431 | 70 |
| p ≥ 0.76 | 0.909 | 0.369 | 0.525 | 2,048 | 206 |

**Calibration** holds on test. Mean p against observed rate is 0.968 / 0.962 in the top bin, 0.550 / 0.543 at 0.5–0.6, and
0.027 / 0.029 in the bottom bin.

**Lower tier** (exploratory): frozen AUC 0.781, against 0.732 for Step 4 RF.

**Per-match AUC** ranges from 0.797 (3853) to 0.885 (3813).

## 5. Association results (high tier, pooled, rates per incoming tracklet)

| | Top-1 agreement | Correct | Wrong | Uncertain | Unmatched | **Coverage** | **False assoc.** | Complete encounters |
|---|---|---|---|---|---|---|---|---|
| **Frozen, relaxed (0.76 / 1.0)** | 0.719 | 0.368 | 0.036 | 0.597 | 0 | **0.403** | **0.088** | 0.327 |
| **Frozen, conservative (0.87 / 1.0)** | 0.719 | 0.257 | 0.012 | 0.730 | 0 | **0.270** | **0.046** | 0.207 |

- **Coverage and false association together:**
  - At the relaxed point, the model commits on 40% of tracklets and 8.8% of those commitments disagree with the provider.
  - At the conservative point, it commits on 27%, with 4.6% disagreement.
- **The correct-match rate (0.368 / 0.257) is not overall accuracy.** The model abstains (UNCERTAIN) on 60–73% of tracklets.
- **Top-1** (0.719) is the share of incoming tracklets whose most probable candidate is the reference. It is reported separately
  because it ignores abstention and the one-to-one constraint.
- **UNMATCHED is 0** because every high-tier incoming tracklet had a reachable candidate.

**Lower tier** (exploratory):

| Point | Coverage | False association |
|---|---|---|
| Relaxed | 0.227 | 0.100 |
| Conservative | 0.132 | 0.060 |

**Natural suspicious tier** (105 incoming, unverified; decision counts only): at the relaxed point 9 MATCHED, 94 UNCERTAIN and 2
UNMATCHED.

## 6. Development versus validation versus test

| High tier | Development | Validation | Test | Test − validation | Test − development |
|---|---|---|---|---|---|
| ROC-AUC | 0.839 | 0.848 | 0.838 | −0.010 | −0.001 |
| Top-1 | 0.718 | 0.727 | 0.719 | −0.008 | +0.001 |
| ECE / Brier | 0.0075 / 0.162 | 0.0072 / 0.158 | 0.0068 / 0.163 | ≈ 0 | ≈ 0 |
| Relaxed coverage | 0.399 | 0.428 | 0.403 | −0.025 | +0.004 |
| Relaxed false association | 0.086 | 0.083 | 0.088 | +0.005 | +0.002 |
| Conservative coverage | 0.269 | 0.294 | 0.270 | −0.024 | +0.001 |
| Conservative false association | 0.043 | 0.047 | 0.046 | −0.001 | +0.003 |
| Relaxed complete encounters | – | 0.346 | 0.327 | −0.019 | – |
| Per-match AUC range | 0.815–0.867 | 0.795–0.913 | 0.797–0.885 | | |
| Per-match relaxed false association | 0.027–0.133 | 0.042–0.118 | 0.035–0.142 | | |

**Distribution shift.** The share of high-tier incoming tracklets per stratum hardly moves between splits:

| Stratum | Development | Validation | Test |
|---|---|---|---|
| Gap < 1.5 s | 0.355 | 0.377 | 0.365 |
| Gap ≥ 2.5 s | 0.298 | 0.268 | 0.288 |
| 3+ players | 0.118 | 0.099 | 0.108 |
| Sharp turn | 0.639 | 0.643 | 0.668 |
| Crossing | 0.436 | 0.414 | 0.430 |

- Calibration did not shift.
- Test is slightly harder than validation: more long gaps and more sharp turns. This plausibly explains the small coverage drop
  against validation. I did not test that explanation.

## 7. Baseline comparison (test, high tier, identical instances and information access)

Each method is at its frozen operating point, and none was tuned on test. Brackets are 95% match-bootstrap intervals.

| Method | What it is | Correct | Wrong | Coverage | False assoc. [CI] | Complete | Frozen relaxed minus method: coverage / false assoc. |
|---|---|---|---|---|---|---|---|
| Nearest position | Distance only, not one-to-one (1,603 conflicts) | 0.404 | 0.362 | 0.766 | 0.473 [0.462, 0.484] | 0.041 | [−0.385, −0.343] / [−0.404, −0.365] |
| Constant velocity | Extrapolation, not one-to-one (1,175 conflicts) | 0.443 | 0.327 | 0.770 | 0.425 [0.408, 0.437] | 0.137 | [−0.382, −0.351] / [−0.355, −0.313] |
| Hungarian + CV | One-to-one, gate only | 0.400 | 0.206 | 0.606 | 0.341 [0.309, 0.365] | 0.380 | [−0.222, −0.187] / [−0.279, −0.222] |
| Chaining (task B) | One-to-one | 0.411 | 0.231 | 0.642 | 0.360 [0.332, 0.387] | **0.403** | [−0.267, −0.212] / [−0.306, −0.239] |
| **E0-Adapted** | FOOTBALLTrace-*inspired* matcher (Step 3) | 0.302 | 0.193 | 0.496 | 0.390 [0.367, 0.418] | 0.291 | [−0.123, −0.060] / [−0.332, −0.274] |
| Step 4 LR / LR + FT | Frozen, threshold 0.98 | 0.016 | 0.001 | 0.017 | 0.032 | 0.008 | [+0.369, +0.403] / [+0.025, +0.086] |
| Step 4 LR distance | Proximity control | 0 | 0 | 0 | – | 0 | – |
| **Step 4 RF** | Primary baseline (0.86 / 0.5) | 0.144 | 0.009 | 0.152 | 0.056 [0.046, 0.067] | 0.090 | [+0.238, +0.264] / [+0.014, +0.051] |
| Step 4 RF + FT features | Frozen, threshold 0.84 | 0.156 | 0.011 | 0.167 | 0.064 [0.053, 0.077] | 0.100 | [+0.224, +0.249] / [+0.008, +0.044] |
| Step 5 with `conf_*`, relaxed | 0.75 / 1.0 | 0.375 | 0.037 | 0.412 | 0.090 [0.074, 0.107] | 0.330 | [−0.014, −0.004] / [−0.006, +0.003] |
| **Frozen, relaxed** | 0.76 / 1.0 | 0.368 | **0.036** | 0.403 | 0.088 [0.072, 0.107] | 0.327 | – |
| **Frozen, conservative** | 0.87 / 1.0 | 0.257 | **0.012** | 0.270 | 0.046 [0.034, 0.062] | 0.207 | – |

**Reading the comparison:**
- **The rule-based matchers commit on 50–77% of tracklets, at 34–47% false association.** The frozen model at the relaxed point
  gets 0.368 correct against 0.400–0.411 for Hungarian or chaining. That is about 90% of their correct matches, with 16–17% of
  their wrong ones (0.036 against 0.206–0.231).
- **The rule-based matchers rarely abstain** (Hungarian's UNCERTAIN comes from its gate). Their higher coverage is therefore not a
  like-for-like advantage, and coverage and false association have to be read together.
- **Chaining is highest on complete encounters** (0.403 against 0.327), because it commits nearly everything.
- **Against the frozen Step 4 RF,** the conservative point has 0.106–0.128 more coverage at no measurable cost in false association
  (paired difference [−0.021, +0.005]). The relaxed point has 0.238–0.264 more coverage for 1.4–5.1 points more false association.
- **The paper-inspired (FT) features again add nothing.** Pairwise AUC is 0.775 with or without them, and RF + FT's slightly higher
  coverage comes only from its lower frozen threshold.
- **Removing the provider confidence codes cost nothing.** The model that keeps them (`conf_*`) has the same AUC (0.837) and the
  same association rates.

## 8. Confidence intervals and per-match variability

**Uncertainty method.** All intervals are 95% match-level bootstrap intervals (2,000 resamples of the 12 matches, seed 0, paired
for differences). Matches, not pairs or tracklets, are the resampled unit. This respects within-match and within-encounter
correlation. The AUC interval is computed the same way.

**Per match, high tier:**

| Match | Pairwise AUC | Relaxed coverage / false assoc. (MATCHED) | Conservative coverage / false assoc. | Step 4 RF coverage / false assoc. |
|---|---|---|---|---|
| 3813 | **0.885** | 0.458 / 0.035 (143) | 0.304 / 0.011 | 0.186 / 0.034 |
| 3845 | 0.840 | 0.425 / 0.062 (96) | 0.265 / 0.067 | 0.168 / 0.053 |
| 3848 | 0.842 | 0.347 / 0.070 (158) | 0.240 / 0.028 | 0.114 / 0.077 |
| 3850 | 0.857 | 0.418 / 0.066 (351) | 0.293 / 0.024 | 0.175 / 0.041 |
| 3853 | **0.797** | 0.323 / **0.126** (87) | 0.208 / 0.071 | 0.123 / 0.030 |
| 3854 | 0.840 | 0.387 / 0.089 (292) | 0.249 / 0.032 | 0.154 / 0.052 |
| 3857 | 0.832 | 0.399 / 0.085 (165) | 0.244 / 0.030 | 0.111 / 0.022 |
| 10504 | 0.809 | 0.395 / **0.142** (148) | 0.256 / 0.083 | 0.181 / 0.088 |
| 10505 | 0.815 | 0.413 / **0.137** (227) | 0.271 / 0.087 | 0.144 / 0.051 |
| 10509 | 0.847 | 0.436 / 0.068 (234) | 0.302 / 0.049 | 0.158 / 0.082 |
| 10515 | 0.841 | 0.426 / **0.117** (197) | 0.294 / 0.051 | 0.165 / 0.053 |
| 10516 | 0.815 | 0.394 / 0.077 (142) | 0.281 / 0.059 | 0.136 / 0.082 |

- **Coverage is stable:** 0.32–0.46 at the relaxed point and 0.21–0.30 at the conservative point.
- **False association varies about three- to eightfold between matches** (0.035–0.142 relaxed; 0.011–0.087 conservative).
- **The four matches above 10% at the relaxed point** are three 105xx matches and 3853. They also have the lowest per-match AUC
  (0.797–0.841).
- **The pooled targets do not hold match by match.** No per-match guarantee should be claimed.
- **Several per-match rates rest on fewer than 100 committed decisions** (e.g. 3853 with 87), so individual-match rates are
  imprecise.

## 9. Failure analysis (frozen model, relaxed point, high tier unless stated)

All cases are stored in `scoring/summary.json → cases` with:
- match, instance, period and time;
- anonymous incoming and outgoing tracklet IDs;
- decision, top candidate, provider-reference tracklet, probability, margin and reference-pair probability;
- quality flags (tier, below-high reasons, difficulty, gap, player count, heading change, visibility/confidence class);
- the reason the case was included.

No player identity is copied into the cases.

| Category | Count | Representative case: match, instance (period, time) | Note |
|---|---|---|---|
| Correct confident | 2,042 | 10509 `d381681582951e8c` (P2, 4219.3 s): in-1 → out-1, p 1.00 | Correct despite a 137° turn. Many p ≈ 1.0 commitments are simple 2-player pass-bys |
| Incorrect confident | 198 | 3853 `375a527679f5dd98` (P2, 3839.2 s): in-0 → out-1, p 0.998; reference pair p 0.029 | Crossing. The chosen continuation is smooth |
| Appropriate abstention | 1,352 | 3813 `cd35c5c1ea7de054` (P2, 6862.1 s): top candidate out-0 has p 0.88, the reference out-1 has 0.76, margin 0.29 → UNCERTAIN | The margin rule blocked a would-be error. Similar: 10505 `01b4482c4e480ec9` (p 0.84 against 0.71) |
| Long-gap failure (≥ 2.5 s) | 19 | 3850 `61fd9569119d5e54` (P2, 6068.5 s, gap 2.64 s): p 0.906, reference pair p 0.13 | Few, because the model rarely commits on long gaps (relaxed coverage 0.105 at 2.5–4 s and 0.006 at ≥ 4 s) |
| Sharp-turn failure | 143 | 3850 `b9eb73dab38ce312` (P1, 1735.4 s, 162° turn): p 0.996, reference pair p 0.37 | The reference continuation reverses direction, and the model prefers the straight continuation |
| Multi-player ambiguity | 295 (wrong or abstained with a wrong top candidate, 3+ players) | 3848 `7091d094f2284e84` (P2, 3676.7 s, 3 players): in-1 → out-2, p 0.963, reference pair p 0.12 | Also flagged as a possible reference inconsistency |
| Possible provider-label inconsistency | 13 (Step 6 rule) | 10504 `e636b8ebebd3b132` (P2, 3679.7 s) | **Both** incoming tracklets were matched crosswise at p 0.99 and 0.95, with reference pairs at p 0.000 and 0.001 and smooth chosen paths (0.34 m, 0.19 m). This is the pattern an identity swap in the provider data would produce, but it is **unverified** |
| Natural suspicious (unverified tier) | 105 incoming | e.g. 10504 `0539142af56d2f2e` (3 players): all UNCERTAIN | Listed for independent review. Not relabelled |

**Notable failure patterns:**
- **Most confident wrong matches are two-player encounters** where the chosen continuation is physically smoother than the
  provider's. Either the player genuinely swerved (task B endpoints cannot show this), or the provider label is wrong.
- **The margin rule caught 1,352 high-probability wrong candidates.** It failed when the wrong pair was far more probable than the
  reference pair.

## 10. FOOTBALLTrace's exact scientific role

- **The published FOOTBALLTrace** corrects identity *given provider-labelled trajectories*. It splits two labelled tracks at a
  breakpoint and swaps their tails.
  - We ran it only as evaluation-side **protocol L** (E0-Literal) on the 2,477 two-player high-tier pairs of the test set.
  - It always swaps (2,477 of 2,477), so its agreement with a no-swap reference is 0.
  - Its min-angle breakpoint fell inside the withheld interval in 51% of cases (2,524 of 4,954).
  - It is **not a matcher of anonymous tracklets.** It cannot be compared with the methods in section 7, and **no claim is made
    that the published paper was outperformed.**
- **E0-Adapted** is our FOOTBALLTrace-*inspired* task B matcher (Step 3). On test it reaches 0.496 coverage at 0.390 false
  association. It is a baseline under our protocol, not the paper's method.
- **The FootballTrace-inspired features** in the Step 4 RF + FT and LR + FT models add nothing on test (AUC 0.775 against 0.775).

## 11. Overfitting and generalisation

- **No overfitting is visible.**
  - Test equals development within 0.004 on every headline metric.
  - Test is 0.01–0.03 below validation, inside the validation and test match-bootstrap intervals.
  - Calibration (ECE 0.007) is unchanged.
- **The operating points transferred.** Both were fixed on development and land on target on two independent sets of 12 matches.
  - Relaxed: 0.083 on validation and 0.088 on test.
  - Conservative: 0.047 and 0.046.
- **The improvement over Step 4 replicated on both sets:**
  - pairwise AUC +0.06 against Step 4 RF (0.838 against 0.775 on test);
  - coverage ×2.6 at the relaxed point and ×1.8 at the conservative point.
- **What does not generalise is what never worked:**
  - **Gap 2.5–4 s:** relaxed coverage 0.105, false association 0.161.
  - **Gap ≥ 4 s:** coverage 0.006.
  - **3+ players:** coverage 0.128 with 3 players and 0.015 with 4+.
  - **Sharp turns:** false association 0.180 at 90–135° and 0.158 above 135°, against 0.035 for turns of 45° or less.
  - **Crossings closer than 0.5 m:** false association 0.177.
  - **Estimated samples present:** false association 0.107, against 0.073 for all-HIGH confidence.

  These limits were already seen in development and validation, so the model is generalising its weaknesses as well as its
  strengths.

**Encounter-level strata, test, high tier.** Each cell is coverage / false association. n is incoming tracklets.

| Stratum | n | Relaxed | Conservative | Step 4 RF | Hungarian |
|---|---|---|---|---|---|
| Gap < 1.5 s | 2,026 | 0.682 / 0.065 | 0.508 / 0.033 | 0.330 / 0.054 | 0.705 / 0.218 |
| Gap 1.5–2.5 s | 1,927 | 0.383 / 0.121 | 0.223 / 0.079 | 0.088 / 0.059 | 0.604 / 0.371 |
| Gap 2.5–4 s | 1,125 | 0.105 / 0.161 | 0.036 / 0.025 | 0.008 / 0.111 | 0.479 / 0.477 |
| Gap ≥ 4 s | 477 | 0.006 / 0 | 0 / – | 0 / – | 0.493 / 0.625 |
| Duration < 1 s | 319 | 0.759 / 0.054 | 0.611 / 0.021 | 0.423 / 0.044 | 0.743 / 0.169 |
| Duration 1–2 s | 2,970 | 0.576 / 0.086 | 0.403 / 0.053 | 0.231 / 0.058 | 0.677 / 0.277 |
| Duration 2–3 s | 1,183 | 0.222 / 0.126 | 0.085 / 0.020 | 0.021 / 0.040 | 0.518 / 0.442 |
| Duration 3–4 s | 606 | 0.035 / 0.191 | 0.010 / 0 | 0.002 / 0 | 0.446 / 0.493 |
| Duration 4–6 s | 477 | 0.006 / 0 | 0 / – | 0 / – | 0.493 / 0.625 |
| 2 players | 4,954 | 0.440 / 0.087 | 0.297 / 0.045 | 0.169 / 0.055 | 0.611 / 0.314 |
| 3 players | 468 | 0.128 / 0.133 | 0.051 / 0.125 | 0.024 / 0.091 | 0.586 / 0.536 |
| 4+ players | 133 | 0.015 / 0 (2 matched) | 0.007 / 0 | 0 / – | 0.496 / 0.742 |
| Turn ≤ 45° | 1,491 | 0.631 / 0.035 | 0.486 / 0.015 | 0.322 / 0.017 | 0.767 / 0.121 |
| Turn 45–90° | 1,348 | 0.449 / 0.093 | 0.284 / 0.050 | 0.134 / 0.061 | 0.619 / 0.333 |
| Turn 90–135° | 1,128 | 0.271 / 0.180 | 0.142 / 0.138 | 0.066 / 0.216 | 0.529 / 0.538 |
| Turn > 135° | 1,230 | 0.164 / 0.158 | 0.080 / 0.102 | 0.031 / 0.158 | 0.468 / 0.607 |
| Turn unmeasured | 358 | 0.520 / 0.118 | 0.371 / 0.053 | 0.207 / 0.081 | 0.603 / 0.282 |
| No crossing | 3,169 | 0.413 / 0.107 | 0.285 / 0.053 | 0.163 / 0.051 | 0.591 / 0.457 |
| Crossing, ≥ 1 m | 805 | 0.533 / 0.030 | 0.383 / 0.013 | 0.200 / 0.031 | 0.662 / 0.141 |
| Crossing, 0.5–1 m | 1,308 | 0.336 / 0.077 | 0.193 / 0.051 | 0.115 / 0.080 | 0.599 / 0.209 |
| Crossing, < 0.5 m | 273 | 0.227 / 0.177 | 0.124 / 0.118 | 0.077 / 0.191 | 0.645 / 0.301 |
| All HIGH confidence | 1,585 | 0.456 / 0.073 | 0.310 / 0.026 | 0.189 / 0.047 | 0.633 / 0.320 |
| Minimum MEDIUM confidence | 1,604 | 0.415 / 0.101 | 0.289 / 0.067 | 0.160 / 0.055 | 0.586 / 0.318 |
| LOW or unknown confidence | 1,564 | 0.383 / 0.085 | 0.250 / 0.036 | 0.138 / 0.056 | 0.607 / 0.345 |
| ESTIMATED sample present | 802 | 0.316 / 0.107 | 0.192 / 0.071 | 0.095 / 0.092 | 0.591 / 0.422 |
| Same team | 116 | 0.672 / 0.064 | 0.569 / 0.030 | 0.414 / 0.062 | 0.681 / 0.228 |
| Opposite team | 4,838 | 0.434 / 0.088 | 0.291 / 0.045 | 0.163 / 0.055 | 0.609 / 0.317 |
| Lower tier (reference quality) | 13,456 | 0.227 / 0.100 | 0.132 / 0.060 | 0.065 / 0.052 | 0.591 / 0.341 |

**Notes on the strata:**
- **Small groups.** Groups with fewer than 50 committed decisions should not support conclusions: 4+ players, gap ≥ 4 s, and
  conservative long gaps.
- **Duration is the gap in task B.** The withheld encounter interval is the gap, so the duration rows are the gap in finer bins.
- **Turn and crossing strata are evaluation-only.** They come from the reference continuation, so they describe the reference's
  path, not information the matcher had.

## 12. Scientific limitations

1. **Provider references are not ground truth.** Every "correct" or "wrong" is agreement or disagreement with provider IDs. Some
   confident disagreements look like provider errors: 13 are flagged, including one full crosswise pair (10504 `e636…`). This
   would bias false association upwards, by an unknown amount.
2. **Task B is a withheld-interval benchmark.** It removes the encounter interval from otherwise good tracking. It is not a
   real-occlusion or video-tracking evaluation, and it does not show that occlusions occurred.
3. **The high tier is a selection:** no ESTIMATED endpoints, at most 6 players, at most 6 s, no attribution problems. Results do
   not transfer to all encounters. On the lower tier, false association is 0.100 at the relaxed point.
4. **Pooled targets, not per-match guarantees.** 4 of 12 test matches exceed 10% at the relaxed point.
5. **Coverage is selective.** 60% of high-tier tracklets (73% at the conservative point) stay UNCERTAIN, and long gaps and
   multi-player encounters are mostly left unresolved.
6. **The data is one competition and one provider,** at 30 Hz RAW positions with no appearance information. There is no evidence
   about other leagues, providers or video-derived tracks.
7. **Twelve test matches** give match-bootstrap intervals about ±0.015 wide for coverage and ±0.017 for false association. Per-match
   numbers are imprecise.

## 13. Final research conclusions

1. **The frozen model is a well-calibrated pairwise scorer.** On task B, with RAW positions only, it reaches ROC-AUC 0.84 with
   ECE below 0.01 on three disjoint sets of matches: development (leave-one-match-out), validation and test.
2. **Combined with global one-to-one assignment and margin-based abstention,** it resolves about 40% of high-tier incoming
   tracklets at about 9% false association against provider references, or 27% at under 5%. That is 1.8–2.6 times the coverage of
   the Step 4 classifier, at comparable false association.
3. **Rule-based matchers,** including our FOOTBALLTrace-inspired E0-Adapted, commit more but disagree with the provider on 34–47%
   of their commitments.
4. **The useful signal is football-aware motion plausibility and assignment context.** Neither the FOOTBALLTrace-inspired features
   nor the provider confidence codes add measurable value.
5. **The approach does not resolve** long gaps, crowded encounters or sharp direction changes. These need information that task B
   endpoints do not contain, for example appearance from video.
6. **What may be claimed:** "provider-reference association agreement on a withheld-interval benchmark". Real-world identity
   recovery accuracy may not be claimed.

## 14. Recommended independent-annotation protocol

This is needed before any accuracy claim. It is not executed here and needs approval.

1. **Sample.** Draw the sample from development and validation matches if possible. Test cases may be annotated, but only as a
   one-off audit of this run. **Strata:**
   - (a) all flagged possible inconsistencies: 13 test cases and 14 validation cases;
   - (b) a random sample of MATCHED decisions at each operating point, about 150 per point, stratified by match;
   - (c) wrong matches with reference-pair p ≤ 0.2;
   - (d) natural suspicious cases, including the known 3816 swap;
   - (e) about 50 3+ player and long-gap encounters.
2. **Material.** Broadcast or tactical video around the encounter (±5 s), with the RAW trajectories overlaid using anonymous IDs.
   Annotators do not see provider IDs or model output.
3. **Labels.** Each annotator chooses one of:
   - same player continues as incoming → outgoing *k*;
   - cannot tell;
   - provider swap evident.

   Free-text notes are allowed.
4. **Procedure.** Two independent annotators, with Cohen's κ reported. Disagreements are adjudicated by a third reviewer, and the
   protocol and analysis plan are written before labelling starts.
5. **Analysis.**
   - true precision of MATCHED decisions, with match-level intervals;
   - the provider-reference error rate in each stratum;
   - a corrected false-association estimate.

   Provider labels are never overwritten. Corrections are stored as a separate, versioned annotation layer.

## 15. The locked test has now been consumed

The 12 locked test matches (3813, 3845, 3848, 3850, 3853, 3854, 3857, 10504, 10505, 10509, 10515, 10516) have been processed and
scored exactly once, with the frozen model `629cca77…`.
- **Any later change** to the model, features, thresholds, calibration or eligibility makes these matches development or validation
  data for that change. They can no longer serve as an unbiased test.
- **Future unbiased claims** need new matches or independently annotated data.
- **The outputs** are in `data/processed/m11-test-v1` (audit, encounters, matching, predictions and scoring, each checksummed) and
  must not be regenerated or overwritten.

## Tests and integrity (checks that actually ran)

**New tests: `tests/test_matching_locked_test.py`, 7 tests.**
- **Split isolation:** only locked test matches are accepted.
- **Stage order:**
  - every stage refuses to run without a passed audit;
  - a failed audit writes nothing.
- **Matcher-visible input:**
  - the test loader works with no reference file present;
  - it rejects a manifest whose split is not test.
- **Completeness:** every incoming tracklet must be decided exactly once.
- **Metric denominators and strata bins:** the high-tier subset and the all-tier "tier" stratum, the 3-player rates, and the
  turn, crossing and visibility bins.
- **Thresholded precision, recall and F1 counts;** the match bootstrap of AUC is deterministic and bounded.

**Results.**
- New tests: 7 passed.
- Full suite (after scoring): **731 passed, 6 skipped, 1 warning** (an unrelated vision deprecation warning), 101 s.

**Integrity checks, after scoring.**
- **Checksums:**
  - `m11-frozen-v1`, `m11-ml-v1` (31 files), `m11-ml-v2` (26 files), the Step 6 predictions and scoring, and the Step 7 audit,
    predictions and scoring all verify against their manifests;
  - 71 per-match `root.manifest.json` files under `m11-*` (493 files) have 0 mismatches.
- **Raw data:** the snapshot (394 files, `a6d610de…`) is identical at the audit, predict, score and after scoring.
- **Determinism:** after scoring, match 3813's predictions were recomputed from its matcher-visible file. They equal the stored
  test predictions exactly (3,469 pairs with a maximum difference of 0.0, and 1,406 identical decision sets). Nothing was
  overwritten.
- **Harmless warning:** scoring printed one numpy all-NaN quantile warning. It comes from the bootstrap of `step4_lr_distance`,
  which never commits.
