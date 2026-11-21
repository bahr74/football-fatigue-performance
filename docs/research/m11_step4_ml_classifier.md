# M11 Step 4: first supervised player identity-matching classifier (task B)

Status: bounded pilot complete. 2026-11-06.

Every accuracy number below is **agreement with the provider identity reference**. None is independently verified identity accuracy.

The evaluation is **leave-one-match-out over four development matches** (3812, 3814, 3816, 10506):
- every reported number comes from a model that never saw the match it is scored on;
- the decision threshold was also chosen without that match;
- no validation or test match was used.

This is a match-disjoint *development* estimate, not a validation result.

## 1. What was built

### Code changes

| File | Change |
|---|---|
| `src/football_fatigue/matching/ml.py` | **New.** Features, label protocol, models, global assignment, metrics, leave-one-match-out experiment, outputs |
| `src/football_fatigue/matching/evaluation.py` | `evaluate(..., methods=BASELINES)`: optional method list, so the existing evaluator scores the ML methods too. The default behaviour is unchanged |
| `src/football_fatigue/matching/__main__.py` | New `ml` subcommand |
| `src/football_fatigue/matching/__init__.py` | Docstring only |
| `tests/test_matching_ml.py` | **New.** 17 tests on synthetic fixtures |

### Commands

```powershell
.\.venv\Scripts\python.exe -m football_fatigue.encounters build --root data/raw --match 3816   # also 10506 and 3814
.\.venv\Scripts\python.exe -m football_fatigue.matching build --root data/raw --match 3816     # also 10506 and 3814
.\.venv\Scripts\python.exe -m football_fatigue.matching ml --matches 3812 3816 10506 3814
```

### Dependency

scikit-learn 1.9.1 was already installed in `.venv`. It is declared only under the `vision` extra, and `pyproject.toml` was not changed because it holds unrelated pending edits. `ml.py` imports scikit-learn lazily, and its tests are skipped when it is missing.

### Pipeline

1. **Load.** `load_match` reads `m11-matching-v3/<match>/root.instances_b.jsonl`, `root.references.jsonl` and `root.predictions_b.jsonl`. It checks each file's sha256 against the v3 manifest. It refuses a source that is not a development, RAW, swap-free output.
2. **Features.** `instance_features(inst)` reads one matcher-visible task B instance and nothing else.
3. **Labels.** `pair_labels(reference, instance)` is the only function that reads the reference. Labels are joined to features by `(instance_id, incoming, outgoing)` only after both have been built.
4. **Train.** One model per held-out match is fitted on the other three matches, using clean high-tier pairs only.
5. **Choose the threshold.** An inner leave-one-match-out runs over the three training matches. The threshold is the lowest probability at which the pooled out-of-fold false-association rate is at most 5%, which gives the most coverage under that cap. The target was fixed in advance.
6. **Decide.** Each held-out encounter gets a global one-to-one assignment (section 7 of this report covers the decision rule).
7. **Final models.** A final model per type is fitted on all four matches and saved for later use. It is not evaluated, because no held-out match remains.

### Outputs in `data/processed/m11-ml-v1/`

The directory was written all at once; an existing directory is never replaced.

| File | Content |
|---|---|
| `config.json` | Feature definitions, configuration, label protocol, decision rules |
| `summary.json` | Dataset statistics, folds, pairwise and encounter-level results with strata, thresholds, trade-off curves, model weights, top failures |
| `pairs.jsonl` | Matcher-side: every pair's features and held-out probability per model |
| `pair_labels.jsonl` | Evaluation-only: label, tier and label status per pair |
| `predictions.jsonl` | Encounter-level decisions per model, with probability, margin and evidence |
| `failures.jsonl` | Most confident held-out wrong matches |
| `models/heldout-<match>/<model>.joblib` | The 20 fold models that produced the reported numbers |
| `models/final/<model>.joblib` | Final models trained on all four matches |
| `manifest.json` | sha256 of all 31 outputs, sources, split manifest, code; versions |

### Other new outputs

All previous outputs are unchanged.
- `m11-encounters-v2/{3816,10506,3814}`: Step 1 catalogues. 10506 quarantines periods 3–4 (extra time), as the frozen split manifest's extra-time rule requires.
- `m11-matching-v3/{3816,10506,3814}`: task A, task B, baselines and E0 for each match.

## 2. Labelling protocol

The provider identity mapping is a provisional reference.

| Pair class | Definition |
|---|---|
| Positive | Incoming and outgoing tracklets carry the same provider identity within one encounter |
| Negative | They carry different provider identities **within the same encounter** |

Every candidate in an encounter is a nearby player, so negatives are hard by construction: the model has to separate identities among close players, not near from far.

| Status | Rule | Use |
|---|---|---|
| `train` | `high_confidence_reference` tier, and no ESTIMATED observation in either endpoint window (last or first 0.5 s) | Supervised fitting, threshold choice, primary evaluation |
| `excluded_estimated_endpoint` | High tier, but ESTIMATED inside an endpoint window | Not trained on. The encounter is still evaluated |
| `exploratory` | `lower_confidence_reference` | Separate exploratory evaluation only |
| `unverified` | `natural_suspicious_unverified` | Never labelled, trained on or scored. Decisions are saved |

The high tier already requires ≥ 90% coverage, ≤ 10% ESTIMATED observations, no implausible jump, no frame-wide shift, no segment edge, and all incoming and outgoing tracklets present (Step 2 design note). Suspicious cases are never treated as confirmed swaps.

### Eligibility per match (task B, τ = 2 m)

| Match | Events | Accepted (high / lower / suspicious) | Rejected | Main rejections |
|---|---|---|---|---|
| 3812 | 1,274 | 214 / 628 / 3 | 429 | ambiguity > 6 s 191, overlap 183, > 6 players 47 |
| 3816 | 1,173 | 178 / 590 / 6 | 399 | ambiguity 193, overlap 162, > 6 players 31 |
| 10506 | 1,558 | 257 / 524 / 0 | 777 | attribution unresolved 354, ambiguity 221, overlap 142 |
| 3814 | 921 | 183 / 442 / 3 | 293 | ambiguity 140, overlap 120, > 6 players 29 |

### Class distribution

| Tier | Instances | Incoming | Pairs | Positives | Negatives | Notes |
|---|---|---|---|---|---|---|
| High | 832 | 1,746 | 3,768 | 1,746 | 2,022 | 21 pairs withheld for ESTIMATED endpoints; 3,747 training pairs (1,736 positive, 2,011 negative) |
| Lower (exploratory) | 2,184 | 4,696 | 10,552 | 4,696 | 5,856 | — |
| Suspicious (unverified) | 12 | — | 85 | — | — | Not labelled |

**Imbalance.** About 46% of training pairs are positive, because negatives are restricted to the encounter. Both models still use balanced class weights, and calibration is measured on the held-out matches.

## 3. Features

All features are computed from a task B instance only:
- no identity, roster, jersey, slot, team, match ID, frame or absolute time;
- no observation from the withheld interval;
- no Step 1 continuity flag;
- no tracklet order (a test checks that permuting the order leaves every pair's features unchanged).

| Group | Features |
|---|---|
| Gap and reachability | `gap_s`, `dist_m`, `dist_per_s`, `reach_ratio` (distance ÷ (0.5 m + 12 m/s × gap)) |
| Endpoint kinematics | `speed_in`, `speed_out`, `speed_diff`, `vel_diff`, `heading_diff_deg`, `both_moving` |
| Extrapolation | `cv_fwd_err`, `cv_bwd_err`, `cv_mid_err` (forward and backward half-gap extrapolations meeting), `ca_fwd_err` (constant acceleration, capped at 8 m/s²) |
| Path physics | `req_acc` (constant acceleration needed from the incoming state); `hermite_max_acc`, `hermite_rms_acc` (cubic Hermite path joining both endpoint states) |
| Within-tracklet shape (1 s) | `acc_in`, `acc_out`, `turn_rate_in`, `turn_rate_out`, `straight_in`, `straight_out` |
| Quality and coverage | `n_upd_in`, `n_upd_out`, `est_share_in`, `est_share_out`, `conf_in`, `conf_out` |
| Context | `n_incoming`, `n_outgoing` |
| Competition | `row_gap_*` and `col_gap_*` for `dist_m`, `cv_fwd_err`, `cv_mid_err` and `hermite_rms_acc`: each value minus the best competing candidate in the same row or column |
| FOOTBALLTrace-inspired (ablation only) | `ft_sharpness` (the E0-Adapted junction sharpness on the 25 Hz Savitzky–Golay paper clock) and its `row_gap` |

The FOOTBALLTrace-inspired features are **not the published method** (see the Step 3 report).

## 4. Models

All models are scikit-learn pipelines with deterministic seeds. Complexity is fixed in advance and was not tuned.

| Model | Description |
|---|---|
| `lr` | `StandardScaler` + `LogisticRegression(C=1, class_weight=balanced)`, core features |
| `rf` | `RandomForestClassifier(300 trees, min_samples_leaf=5, class_weight=balanced_subsample, random_state=0)`, core features |
| `lr_ft`, `rf_ft` | The same models plus the FOOTBALLTrace-inspired features (ablation) |
| `lr_distance` | Proximity-only control: `gap_s` and `dist_m` only |

## 5. Match-disjoint split

- **Frozen split:** the 40/12/12 manifest `m11-encounters-v1/splits.json` was not changed. All four matches are in the development split, and three of them (3812, 3816, 10506) are its fixed development matches.
- **Outer folds:** four, each holding out one match. Each fold model is trained on 2,530–2,939 pairs from three matches. A test checks that the scaler of each fold model saw exactly the training pairs of the other matches.
- **Inner folds:** three, over the training matches. They are used only to choose the threshold.
- **Separation:** instance IDs are unique across matches, which the code checks. No tracklet, encounter or time window is shared between partitions.

## 6. Results: pairwise classification (held-out, pooled, high tier)

These are pair-level metrics at p ≥ 0.5. They are separate from encounter-level decisions.

| Model | Precision | Recall | F1 | ROC-AUC | Avg. precision | Brier | ECE | Top-1 per incoming |
|---|---|---|---|---|---|---|---|---|
| lr | 0.657 | 0.713 | 0.684 | 0.760 | 0.720 | 0.200 | 0.033 | 0.679 |
| rf | 0.671 | 0.690 | 0.680 | **0.785** | **0.761** | **0.188** | 0.035 | **0.690** |
| lr_ft | 0.658 | 0.710 | 0.683 | 0.760 | 0.720 | 0.200 | 0.033 | 0.680 |
| rf_ft | 0.673 | 0.683 | 0.678 | 0.785 | 0.760 | 0.188 | 0.030 | 0.688 |
| lr_distance | 0.487 | 0.642 | 0.554 | 0.539 | 0.501 | 0.249 | 0.039 | 0.511 |

### By held-out match (ROC-AUC)

| Model | 3812 | 3816 | 10506 | 3814 |
|---|---|---|---|---|
| lr | 0.754 | 0.783 | 0.771 | 0.729 |
| rf | 0.782 | 0.798 | 0.797 | 0.758 |
| lr_distance | 0.528 | 0.532 | 0.560 | 0.532 |

- **Exploratory lower tier:** LR AUC 0.689 and RF 0.723; top-1 per incoming 0.606 and 0.609.
- **Proximity is not the signal.** The distance-only control is near chance (AUC 0.54, top-1 0.51), so the classifiers are not learning "close means same player".
- **What the models use.** The heaviest LR weights are the competition features of the Hermite RMS acceleration, followed by `vel_diff`, `cv_fwd_err` and `cv_mid_err`. The top RF importances are the `cv_mid_err` and `hermite_rms_acc` competition features.

### Calibration (pooled high tier, 10 bins)

| Model | ECE | Where it deviates |
|---|---|---|
| LR | 0.033 | Overconfident at the top: bin 0.9–1.0 has mean p 0.949 against 0.884 positive; bin 0.8–0.9 has 0.849 against 0.756 |
| RF | 0.035 | Underconfident at the extremes: 0.944 against 0.988, and 0.058 against 0.005 |

No probability recalibration was applied.

## 7. Results: encounter-level decisions (held-out, pooled, high tier: 832 instances, 1,746 incoming)

### Decision rule

1. Solve a one-to-one assignment that maximises the summed log-odds over physically reachable pairs. Leaving a row unassigned costs nothing, so a pair below p = 0.5 is never taken.
2. **MATCHED** requires p ≥ threshold and a margin ≥ 0.5 log-odds over the best assignment without this pair.
3. **UNCERTAIN** applies when a reachable candidate exists but is unassigned, below the threshold, or within the margin.
4. **UNMATCHED** applies only when no outgoing tracklet is reachable within 12 m/s. A low probability alone never means the player disappeared.

### Results

ML models are shown at their thresholds chosen on training matches.

| Method | Correct | Wrong | Uncertain | Unmatched | Coverage | False assoc. | Complete |
|---|---|---|---|---|---|---|---|
| Nearest position | 0.400 | 0.365 | 0.149 | 0.085 | 0.766 | 0.477 | 0.037 |
| Constant velocity | 0.430 | 0.330 | 0.174 | 0.066 | 0.759 | 0.434 | 0.142 |
| Hungarian + CV | 0.378 | 0.220 | 0.316 | 0.086 | 0.598 | 0.368 | 0.358 |
| Chaining (no interval) | 0.413 | 0.221 | 0.365 | 0.000 | 0.635 | 0.348 | **0.403** |
| E0-Adapted (FOOTBALLTrace-inspired) | 0.279 | 0.188 | 0.533 | 0.000 | 0.467 | 0.403 | 0.260 |
| **rf** (threshold 0.82–0.86) | 0.140 | **0.008** | 0.852 | 0.000 | 0.148 | **0.054** | 0.083 |
| rf_ft (0.81–0.85) | 0.151 | 0.008 | 0.841 | 0.000 | 0.159 | 0.050 | 0.088 |
| lr (0.95–0.98) | 0.027 | 0.003 | 0.970 | 0.000 | 0.030 | 0.096 | 0.012 |
| lr_ft | 0.026 | 0.003 | 0.971 | 0.000 | 0.029 | 0.098 | 0.012 |
| lr_distance | 0 | 0 | 1.000 | 0 | 0 | — | 0 |

- "False assoc." is wrong matches ÷ MATCHED decisions. "Complete" is the share of encounters with every decision correct.
- Conflicts (two incoming tracklets matched to one outgoing) are 0 for every one-to-one method. Nearest has 497 and CV 371.
- **Per held-out match**, RF's false-association rate was 0.036 (3812), 0.051 (3816), 0.066 (10506) and 0.057 (3814), at coverage 0.13–0.17.
- **LR's threshold did not transfer.** It met the target on the training matches, but on held-out data its false-association rate was 0.096 pooled and 0.138 on 3814 (small numbers).
- **Exploratory lower tier** (4,696 incoming): RF had coverage 0.078 and false association 0.052; LR 0.036 and 0.150; Hungarian 0.595 and 0.342; chaining 0.651 and 0.393.

## 8. Coverage versus false association

This curve is held out and pooled over the high tier. It is descriptive only and was never used to choose anything.

| Threshold | RF coverage | RF false assoc. | RF complete | LR coverage | LR false assoc. |
|---|---|---|---|---|---|
| 0.50 (maximum) | 0.376 | 0.108 | 0.234 | 0.375 | 0.125 |
| 0.70 | 0.286 | 0.098 | 0.160 | 0.294 | 0.128 |
| 0.80 | 0.178 | 0.061 | 0.097 | 0.193 | 0.137 |
| 0.85 | 0.139 | 0.054 | 0.081 | 0.144 | 0.127 |
| 0.90 | 0.095 | 0.012 | 0.058 | 0.097 | 0.107 |

- **Ceiling.** RF coverage cannot exceed 0.38, because of the 0.5-probability floor and the a priori margin.
- **Comparison at RF's highest coverage.** At 0.5, RF commits 0.335 of incoming tracklets correctly and 0.041 wrongly. Hungarian commits 0.378 correctly and 0.220 wrongly. That is about 89% of Hungarian's correct matches with about 19% of its wrong matches.
- **Comparison at the chosen operating point.** RF trades most of its coverage for a false-association rate of about 5%.
- **Complete-encounter accuracy** is lower for the ML models than for chaining or Hungarian, because abstaining on one tracklet makes the whole encounter incomplete.

## 9. Difficulty strata (held-out, high tier, at the chosen thresholds)

| Stratum | RF coverage / false assoc. | Hungarian coverage / false assoc. | Chaining coverage / false assoc. |
|---|---|---|---|
| Gap < 1.5 s (587 incoming) | 0.315 / 0.027 | 0.704 / 0.196 | 0.741 / 0.361 |
| Gap 1.5–2.5 s (592) | 0.125 / 0.122 | 0.610 / 0.385 | 0.633 / 0.360 |
| Gap 2.5–4 s (378) | 0.000 / — | 0.481 / 0.621 | 0.574 / 0.290 |
| Gap ≥ 4 s (189) | 0.000 / — | 0.466 / 0.580 | 0.429 / 0.383 |
| Two players (1,528) | 0.165 / 0.052 | 0.607 / 0.332 | 0.632 / 0.325 |
| Three or more players (218) | 0.032 / 0.143 | 0.537 / 0.650 | 0.651 / 0.507 |
| No sharp turn (638) | 0.282 / 0.033 | 0.735 / 0.188 | 0.649 / 0.382 |
| Sharp turn > 45° (1,108) | 0.071 / 0.101 | 0.519 / 0.515 | 0.626 / 0.329 |
| Crossing (780) | 0.141 / 0.045 | 0.626 / 0.232 | 0.558 / 0.788 |
| No crossing (966) | 0.154 / 0.060 | 0.576 / 0.487 | 0.697 / 0.064 |

- **The classifier abstains as difficulty grows.** It makes no commitments beyond 2.5 s gaps.
- **Hard cases cost the most.** Sharp turns and three or more players carry its highest false-association rates (0.10–0.14, on small numbers).

## 10. Critical failure cases

The full list is in `failures.jsonl`.

1. **A sharp turn at a crossing makes the wrong continuation look smoother.**
   - 3814 `de791660603dbebc` (LR p 0.979; the true pair has p 0.547): the true path turns 174° inside a 1.0 s gap.
   - 10506 `468f8100c0de9cef` (RF p 0.899; true pair 0.328).
   - 3812 `a3983a592ade84d8` (RF p 0.899; true pair 0.530).
   - These are 7 of the 15 most confident wrong matches for `lr` and `rf`.
2. **Multi-player turns.** 10506 `b62a1c2a4197488a` (three players, 177° turn; the true pair has p 0.11). 10506 `aeb1e491520c2881` (four players, crossing, 3 s gap, 160° heading difference).
3. **A near-perfect wrong continuation.** In 3816 `0ca96cb82c11e789` (RF p 0.96, no difficulty flag), the chosen outgoing tracklet starts 0.19 m from the incoming end, with a 0.97 m extrapolation error. The provider reference says it is the other player. This is either a real hand-over that motion cannot resolve or a reference error. It cannot be decided without independent labels.
4. **The known 3816 swap (P2 04:00.2, away #5 / #17).**
   - Its encounter is `1f98c38f556f0c95`: P2, about 4 min after the first P2 event. It has three players (away 5, away 17, home 22), a 5.9 s interval, a crossing, a sharp turn and ESTIMATED observations.
   - Step 1 classed it as ordinary, so it sits in the **lower (exploratory) tier**. Its provider reference may itself be the swapped one.
   - Every model returns UNCERTAIN for every tracklet; RF probabilities are 0.35–0.42. The models neither confirm nor refute the swap. It is not scored.
5. **LR's operating point did not transfer** from the training matches to 3814 (section 7).

## 11. Tests and regression

- **Engineering only.** `tests/test_matching_ml.py` has 17 tests on synthetic hand-built task B instances. They show the pipeline is wired correctly, not that it is accurate on real data.
- **What the tests cover:**
  - pair labels;
  - withholding ESTIMATED endpoints from training;
  - feature values on straight, standing, turning and unreachable players;
  - invariance to tracklet order;
  - refusal of task A or interval data;
  - separation of features from the reference (permuting the reference changes labels but not features; no identity-like feature names);
  - one-to-one assignment;
  - UNMATCHED only without a reachable candidate;
  - UNCERTAIN on ambiguity and below the threshold;
  - deterministic LR and RF training and joblib round trips;
  - training on clean high-tier pairs only;
  - pairwise metric definitions (precision, recall, F1, AUC, ECE, top-1);
  - threshold selection;
  - grouped split integrity, including fold models seeing only other matches' pairs;
  - rejection of duplicate IDs and of fewer than three matches;
  - refusal of validation and test matches and of an existing output.
- **Bug found while writing them:** the exhaustive `solve` prunes on partial totals and needs non-negative costs. `assign` adds one constant to every pair and to abstaining, which leaves the optimum and the margins unchanged.
- **Real artefacts:**
  - all 31 output sha256 values match the manifest;
  - for match 3814, recomputing the features from the instances and reloading the saved fold models (lr, rf, rf_ft) reproduces the stored held-out probabilities to within their 4-decimal rounding;
  - the `m11-matching-v2` manifest (`ec4075daf6bd…`) and the `m11-encounters-v2/3812` cache (`988992f7a664…`) hashes are unchanged.
- **Full suite:** `pytest` gives 696 passed, 6 skipped, 1 warning (110 s), including the 27 Step 2–3 matching tests, the encounters tests and the FIFA tests.

## 12. Scientific limitations

- **The reference is the provider's.** There are no independent identity labels. A confident "wrong" match may be a reference error, as in case 3 of section 10, and the known swap sits in the exploratory tier as an unflagged reference.
- **Development data only.** This is leave-one-match-out over four development matches. No complexity tuning was done, and no validation or test estimate exists.
- **Selection bias.** The high tier excludes encounters longer than 6 s, more than 6 players, attribution problems and overlapping events. Its intervals are proximity, not verified occlusion, and task B's withheld interval is a controlled removal of information, not a real occlusion.
- **Low coverage.** At about 5% false association the classifier commits only 14–16% of incoming tracklets, and none in gaps of 2.5 s or more. It does not solve the long-gap problem.
- **Imperfect calibration.** LR is overconfident at the top, and its threshold did not transfer. No recalibration was applied.
- **One encounter-level rule.** The margin (0.5 log-odds) and the 5% target were fixed in advance and not explored.
- **No gain from FOOTBALLTrace-inspired features** (AUC 0.785 against 0.785).

## 13. GO / NO-GO

- **GO for further model development on task B.**
  - The pairwise identity signal is real and transfers across matches: RF AUC is 0.76–0.80 on every held-out match, against 0.53–0.56 for proximity alone.
  - At an operating point chosen without the held-out match, RF cuts the false-association rate from 0.35–0.48 (all baselines) to 0.05.
  - RF is preferred over LR: higher AUC, and a threshold that transferred.
- **NO-GO for any accuracy, deployment or generalisation claim**, and for dropping the abstention. Coverage is low, long gaps, sharp turns and multi-player encounters remain unsolved, and the evaluation is development-only against provider labels.
- **Drop the FOOTBALLTrace-inspired features** from the next round.

### Recommended next experiment (needs approval)

1. Build 4–6 more development matches, for 8–10 in total.
2. Choose model complexity and recalibration with grouped (match-level) cross-validation on development matches only.
3. Add turn-aware or reachable-set motion features and an encounter-level joint score.
4. Use false association by gap length as the primary safety metric.
5. Then run one frozen evaluation on the validation split.
