# M11 Step 5: diagnosis, football-aware features and model comparison (task B, ten development matches)

Status: 2026-11-06.
- **Inputs:** the Step 1 to 4 reports (`m11_step1_handoff.md`, `m11_step2_matching_design.md`, `m11_step3_taskb_footballtrace.md`,
  `m11_step4_ml_classifier.md`) and the frozen split `m11-encounters-v1/splits.json`.
- **Outputs:**
  - `data/processed/m11-ml-v2/`;
  - Step 1 catalogues and task A/B benchmarks for six more development matches.
- **Unchanged:** `m11-ml-v1` and all earlier outputs. Their manifest hashes were re-verified (section 11).

**Not done in this step:**
- no validation or test match was read;
- no synthetic swap was injected;
- no deep sequence model was built;
- nothing was committed.

Every rate below is **agreement with the provider identity reference**, not verified identity accuracy. The benchmark is task B:
the encounter interval is withheld, which is a controlled removal of information, not a real occlusion.

## Answer first

- **The new model improves on the Step 4 Random Forest.** The improvement comes almost entirely from one new feature family,
  football-aware features (G), and above all from two **global-assignment regret** features. The choice of model family hardly matters.
- **Selected model:** `rf_reg/core_g`, a Random Forest with `max_features = 0.33` and `min_samples_leaf = 5`. Its inputs are the
  Step 4 core features plus family G, with Platt calibration. The rule for choosing it was fixed before any Step 5 result.
- **Results:** held out on ten development matches, pooled over the high-confidence tier.

  | | Step 4 RF (`rf/core`) | Selected (`rf_reg/core_g`) | Match-bootstrap 95% CI of the difference |
  |---|---|---|---|
  | Pairwise ROC-AUC | 0.790 | **0.840** | higher in every match (0.814–0.867 vs 0.758–0.810) |
  | Top-1 per incoming | 0.696 | 0.716 | |
  | Log-loss / ECE (calibrated) | 0.539 / 0.009 | 0.478 / 0.006 | |
  | Coverage at the 5% target | 0.150 | **0.276** | +0.115 to +0.137 |
  | False association at the 5% target | 0.037 | 0.045 | −0.006 to +0.020 |
  | Coverage at the 10% target | 0.295 | **0.426** | +0.110 to +0.151 |
  | False association at the 10% target | 0.096 | 0.095 | −0.023 to +0.022 |

- **The 50% coverage goal at under 10% false association was not reached.**
  - Using thresholds chosen on the training matches only, the best result is 42.6% coverage at 9.5% false association.
  - Even the best point on the held-out curve, which is an oracle and not deployable, reaches only 44% coverage at false
    association ≤ 10%; reaching 50% coverage costs 12.6% false association.
- **Overfitting is not the limiting factor.**
  - Held-out-match AUC equals the within-training-match encounter-grouped AUC: 0.791 vs 0.789.
  - The learning curves are nearly flat.
  - The bottleneck was what the features represent, and it still is for long gaps, multi-player encounters and sharp turns.
- **FOOTBALLTrace-inspired features (family F) add nothing measurable.** They are excluded from the selected model.
- **New risk found:** the provider confidence codes (`conf_in`, `conf_out`, present since Step 4) carry a weak identity hint
  (section 5.4). Removing them changes nothing (AUC 0.839, coverage 0.278 at 0.045 false association).
- **Verdict: CONDITIONAL GO** for one later frozen validation evaluation; the conditions are in section 13.

## 1. New development matches and sample sizes

**Selection.**
- Rule: the 36 not-yet-processed `development` matches of the frozen split are ranked by `sha256("m11-ml-v2|" + match_id)`, and
  the first six are taken: **3847, 3836, 3833, 3815, 3851, 3829**.
- No match attribute was used, and none of the six has an extra-time anomaly.
- 3829 is flagged `release_disagreement`. Like 3816 in Step 4, it was kept: that flag is a split attribute, not an exclusion
  rule.
- Processing reused Step 1 (`football_fatigue.encounters build`) and Steps 2–3 (`football_fatigue.matching build`) with
  unchanged definitions. Each match took about 3 minutes.

**Eligibility** (`m11-matching-v3` audits; events = candidate encounter events; tiers = accepted instances):

| Match | Events | High | Lower | Suspicious | Rejected |
|---|---|---|---|---|---|
| 3812 | 1,274 | 214 | 628 | 3 | 429 |
| 3816 | 1,173 | 178 | 590 | 6 | 399 |
| 10506 | 1,558 | 257 | 524 | 0 | 777 |
| 3814 | 921 | 183 | 442 | 3 | 293 |
| **3847** | 1,273 | 163 | 726 | 2 | 382 |
| **3836** | 1,141 | 266 | 522 | 3 | 350 |
| **3833** | 1,057 | 146 | 534 | 6 | 371 |
| **3815** | 1,210 | 319 | 549 | 5 | 337 |
| **3851** | 1,272 | 194 | 626 | 12 | 440 |
| **3829** | 1,163 | 123 | 693 | 7 | 340 |

**Why lower-tier instances are lower tier.** The main reasons are ESTIMATED observations (506–713 per match) and implausible jumps
(85–197 per match).

**Labelled data, high-confidence tier, all ten matches:**
- 2,043 instances and 4,276 incoming tracklets;
- 9,198 pairs, of which 4,276 are positive;
- 68 pairs left out for ESTIMATED endpoints;
- 9,130 training pairs (4,246 positive, 4,884 negative; 46.5% positive).

**Exploratory and unverified data:**
- lower tier: 5,834 instances and 12,544 incoming (exploratory only, never trained on);
- natural suspicious: 47 instances and 115 incoming (unverified, never scored).

**High-tier profile per match** (instances / incoming / training pairs; encounters with 3+ players; sharp turn; crossing; median gap):

| Match | Inst. | Inc. | Train pairs | 3+ players | Sharp turn | Crossing | Median gap (s) |
|---|---|---|---|---|---|---|---|
| 3812 | 214 | 439 | 906 | 9 | 129 | 69 | 1.84 |
| 3816 | 178 | 376 | 808 | 19 | 111 | 74 | 1.84 |
| 10506 | 257 | 549 | 1,217 | 27 | 156 | 120 | 1.84 |
| 3814 | 183 | 382 | 816 | 13 | 122 | 94 | 1.64 |
| 3847 | 163 | 340 | 722 | 13 | 109 | 70 | 1.57 |
| 3836 | 266 | 548 | 1,141 | 16 | 166 | 104 | 1.90 |
| 3833 | 146 | 310 | 666 | 16 | 82 | 61 | 1.64 |
| 3815 | 319 | 680 | 1,513 | 28 | 195 | 144 | 1.77 |
| 3851 | 194 | 400 | 825 | 12 | 132 | 74 | 1.77 |
| 3829 | 123 | 252 | 516 | 5 | 81 | 52 | 1.77 |

**Gap distribution** (high tier, incoming): <1.5 s 1,519; 1.5–2.5 s 1,482; 2.5–4 s 902; ≥4 s 373.

**Units of analysis.** Pairs within an encounter are correlated. All uncertainty is therefore reported per match or by a
**match-level bootstrap** (2,000 resamples of the ten matches), never by treating pairs as independent.

## 2. Protocol (`src/football_fatigue/matching/ml2.py`)

**Outer loop.** Leave-one-match-out over the ten matches. The held-out match is used only to score.

**Inner loop.** The nine training matches are dealt into three match groups (deterministic). The out-of-fold predictions of
that grouped split are used to:
1. choose hyperparameters from a bounded grid, fixed a priori, by lowest pooled log-loss;
2. fit a Platt calibrator;
3. choose the operating point that maximises inner coverage under each false-association target (5% and 10%). The operating
   point is a threshold in 0.50–0.99 combined with an assignment margin in {0, 0.25, 0.5, 1, 2} log-odds.

The model is then refitted on all nine matches and scored once on the held-out match.

**Decision rule.** Step 4's rule is unchanged except that the margin is now chosen rather than fixed at 0.5:
- one-to-one log-odds assignment;
- **MATCHED** only if p ≥ threshold and margin ≥ the chosen margin;
- **UNMATCHED** only when no outgoing tracklet is reachable;
- otherwise **UNCERTAIN**.

**Grids:**
- `lr` and `rf`: the unchanged Step 4 configurations;
- `rf_reg`: `min_samples_leaf` ∈ {5, 20, 50} × `max_features` ∈ {sqrt, 0.33};
- `hgb` (HistGradientBoosting, 200 iterations, L2 = 1): `max_leaf_nodes` ∈ {7, 15} × `min_samples_leaf` ∈ {20, 50};
- `mlp` (scaled; one hidden layer of 32, or two of 32 and 16): α ∈ {1e-3, 1e-2, 1e-1}, 400 iterations.

All models are seeded. Ablations, learning curves and permutation importance use fixed configurations
(`FIXED` in `ml2.py`).

**Selection rule, written into `ML2Config` before the run:**
- among the models whose pooled held-out false association at the 5% target is ≤ 0.075, take the one with the highest pooled
  held-out coverage;
- count it as improving on Step 4 only if the bootstrap CI of the coverage difference excludes 0 and the upper bound of the
  false-association difference is ≤ +0.02;
- otherwise keep the Step 4 RF.

**This selection uses the outer development folds.** The reported numbers for the selected model are therefore slightly
optimistic, and the frozen validation run is the unbiased check.

**Run.** `python -m football_fatigue.matching ml2 --matches 3812 3816 10506 3814 3847 3836 3833 3815 3851 3829` took 61 minutes,
with outer folds run in four processes. The results do not depend on the number of processes (tested).

## 3. Train-versus-held-out diagnosis (fixed Step 4 configurations, core features)

Means over the ten outer folds:

| Model | Train AUC (in-sample) | Train F1 | Within-training encounter-grouped 5-fold AUC | RF out-of-bag AUC | **Held-out-match AUC** | Held-out F1 | Held-out log-loss | Held-out Brier | Held-out top-1 |
|---|---|---|---|---|---|---|---|---|---|
| LR | 0.755 | 0.672 | 0.754 | – | **0.753** | 0.675 | 0.586 | 0.202 | 0.677 |
| RF | 0.994 | 0.965 | 0.789 | 0.771 | **0.791** | 0.688 | 0.543 | 0.186 | 0.699 |
| HGB (fixed) | 0.948 | 0.864 | 0.788 | – | **0.791** | 0.695 | 0.540 | 0.185 | 0.709 |

**Per held-out match, RF:**
- AUC 0.758 (3847) to 0.813 (3833);
- top-1 0.649 to 0.735.

**Interpretation:**
- **LR underfits.** Its training and held-out AUC are the same, about 0.75, and do not move with more data (section 4). A linear
  score on these features has too little capacity.
- **RF's in-sample AUC of 0.994 is memorisation, not harmful overfitting.**
  - Held-out-match AUC (0.791) equals the AUC on unseen encounters from the training matches (0.789).
  - Out-of-bag AUC (0.771) is lower, because OOB scores pairs whose encounter partners were seen in training.
  - So the gap between matches is about 0: there is **no evidence of overfitting to particular matches**, and no harmful
    distribution shift between them.
- **Calibration.** Raw RF probabilities are slightly overconfident at the top (ECE 0.028 pooled). Platt scaling fitted on the
  inner folds brings this to 0.009, so calibration is now a minor issue.
- **Thresholds transfer for the tree models.**
  - The inner-chosen RF thresholds are identical in every fold: 0.88 and margin 2 for the 5% target; 0.71–0.75 for the 10% target.
  - LR's thresholds do not transfer: held-out false association at the 5% target is 0.077 against an inner value of about 0.04.
- **Low coverage was not a threshold artefact.** At any threshold, the core-feature RF cannot commit much more without a steep
  rise in false association (section 7).

## 4. Learning curves (held-out-match AUC against the number of training matches)

Random match subsets with seed 0; three subsets per size and per held-out match; mean ± SD over held-out matches and subsets:

| Training matches (≈ pairs) | LR | RF | HGB (fixed) |
|---|---|---|---|
| 1 (≈ 940) | 0.747 ± 0.022 | 0.766 ± 0.020 | 0.753 ± 0.021 |
| 2 (≈ 1,850) | 0.750 | 0.776 | 0.768 |
| 3 (≈ 2,650) | 0.752 | 0.780 | 0.773 |
| 5 (≈ 4,520) | 0.753 | 0.786 | 0.782 |
| 7 (≈ 6,480) | 0.753 | 0.788 | 0.788 |
| 9 (≈ 8,220) | 0.753 ± 0.022 | 0.791 ± 0.018 | 0.791 ± 0.018 |

- **LR is flat.** More data will not help a model that underfits.
- **RF and HGB are still rising, but slowly.** RF gains +0.003 AUC from 7 to 9 matches, and HGB's training AUC falls towards its
  held-out AUC.
- **Extrapolation.** Many more development matches might add about 0.01 AUC with the core features. By contrast, adding family G
  to the same nine matches adds **+0.049**.
- **Conclusion: the core-feature models were limited by trajectory and assignment information, not by sample size.**

**Label noise.**
- No high-tier reference pair is physically unreachable (0 of 4,276).
- Only 2 of 4,276 meet a strict "smooth alternative, strained reference" pattern. The rule is in `summary.json → label_noise`.
- The most confident errors (section 10) often look smoother than the reference continuation. That is consistent with either
  real swerves or reference errors.
- Without independent labels, label noise can be neither excluded nor estimated. It is a plausible part of the irreducible error,
  but it is not shown to be the main one.

## 5. Feature analysis and ablation

### 5.1 Families

| Family | Features |
|---|---|
| A spatial and reach | `gap_s`, `dist_m`, `dist_per_s`, `reach_ratio` |
| B velocity and extrapolation | speeds, `speed_diff`, `vel_diff`, `cv_fwd/bwd/mid_err`, `ca_fwd_err` |
| C heading, acceleration and turn | `heading_diff_deg`, `both_moving`, `req_acc`, `hermite_max/rms_acc`, `acc_in/out`, `turn_rate_in/out`, `straight_in/out` |
| D quality | `n_upd_in/out`, `est_share_in/out`, `conf_in/out` |
| E competitive | `n_incoming/outgoing`, `row_gap_*`, `col_gap_*` |
| F FOOTBALLTrace-inspired | `ft_sharpness`, `row_gap_ft_sharpness` |
| G football-aware (new, 14 features) | see 5.3 |

### 5.2 Ablations (fixed RF and HGB, leave-one-match-out, held-out high tier)

The baseline here is "full" (all families A–G, 55 features). "ΔAUC" is the per-match mean difference from full, with the number
of matches (of 10) where the variant is better. The coverage column is the best point on the held-out curve at false
association ≤ 10%; it is **descriptive only** (oracle).

| Variant | RF AUC | RF ΔAUC (better/10) | HGB AUC | HGB ΔAUC (better/10) | RF coverage at FA ≤ 10% |
|---|---|---|---|---|---|
| full (A–G) | 0.839 | – | 0.840 | – | 0.432 |
| drop A | 0.839 | +0.0004 (5) | 0.841 | −0.0005 (6) | 0.417 |
| drop B | 0.838 | −0.0003 (5) | 0.838 | −0.0023 (4) | 0.419 |
| drop C | 0.840 | +0.0009 (6) | 0.843 | +0.0022 (7) | 0.438 |
| drop D | 0.838 | −0.0001 (5) | 0.841 | +0.0001 (6) | 0.435 |
| drop E | 0.829 | **−0.0096 (0)** | 0.830 | **−0.0095 (0)** | 0.409 |
| drop F | 0.839 | +0.0007 (4) | 0.839 | −0.0004 (3) | 0.431 |
| drop G | 0.790 | **−0.0487 (0)** | 0.792 | **−0.0501 (0)** | 0.351 |
| only A | 0.512 | | 0.524 | | 0.000 |
| only B | 0.643 | | 0.640 | | 0.175 |
| only C | 0.607 | | 0.597 | | 0.003 |
| only D | 0.618 | | 0.612 | | 0.101 |
| only E | 0.777 | | 0.773 | | 0.254 |
| only F | 0.552 | | 0.573 | | – |
| only G | 0.828 | | 0.825 | | 0.391 |
| distance only | 0.526 | | 0.529 | | 0.000 |

**Grouped permutation importance.**
- Method: each fold's model never saw its held-out match. Columns are permuted within that match's clean high-tier pairs, with
  three repeats. The value is the AUC drop, as a mean over ten folds.
- Families, RF / HGB:
  - G: 0.237 / 0.275 (10 of 10 folds positive);
  - E: 0.037 / 0.033 (10 of 10);
  - B: 0.013 / 0.013 (10 of 10);
  - A, C, D and F: ≤ 0.003 each.
- Single features, RF:
  - `assign_regret_cv` is rank 1 in every fold (AUC drop 0.076 ± 0.013);
  - `assign_regret_hermite` is rank 2 in every fold (0.033);
  - after these, the features with positive drops come from families E and B.

**Redundant or degenerate features** (Spearman |ρ| ≥ 0.95 on clean high-tier pairs):
- `dist_per_s` ≡ `reach_ratio` (ρ = 1.0);
- `hermite_max_acc` ≈ `hermite_rms_acc` (0.98);
- `n_incoming` ≡ `n_outgoing` ≡ `n_reach_row` ≡ `n_reach_col`: in clean high-tier encounters every candidate is reachable and
  the encounters are square;
- `est_share_in/out` are constant (0) on training pairs by construction, because ESTIMATED endpoints are left out.

**Unstable features.** Most features in A, C and D have rank SDs of 10–17 across folds: they act as interchangeable weak signals.

**Not useful.** No family other than E and G is individually necessary.
- C, the turn and heading family, never helps on its own (AUC 0.60).
- The new turn-feasibility features (`turn_in/out_deg`, `turn_acc_demand`) are among the least important.

### 5.3 Football-aware features (family G; physical constants fixed from Steps 1–4, none estimated from data)

**Fixed assumptions:**
- maximum speed 12 m/s;
- reach base 0.5 m;
- "moving" threshold 1 m/s;
- acceleration cap 8 m/s².

| Feature | Physical idea | Effect |
|---|---|---|
| `assign_regret_cv`, `assign_regret_hermite` | **Global assignment ambiguity.** The cost of the best one-to-one assignment that is forced to contain this pair, minus the unconstrained best. Costs are capped constant-velocity meet-in-the-middle error or Hermite RMS acceleration; abstaining costs the cap | Dominant. They give the classifier what Hungarian matching knows (whether this pairing is consistent with a good global explanation), and the classifier adds calibrated abstention |
| `hermite_peak_speed` | Peak speed of the smooth connecting path; reachability in speed rather than distance | Small |
| `ca_bwd_err`, `ca_mid_err` | Bidirectional constant-acceleration extrapolation | Small (family B overlap) |
| `turn_in_deg`, `turn_out_deg`, `turn_acc_demand` | Turning feasibility: (speed × turn angle) / gap, so a sharp turn at low speed is cheap and is **not** treated as an error by itself | About 0 |
| `speed_consistency` | Mean junction speed against the endpoint speeds (piecewise consistency) | Small |
| `in_sep_m`, `in_closing_m_s`, `out_sep_m` | Relative motion of competing players (separation, closing speed) | Small; carry context about difficulty |
| `n_reach_row`, `n_reach_col` | Number of reachable competitors | Redundant on the high tier |

All 14 are computed from the matcher-visible task B instance only.

**Tests:**
- order invariance;
- task B only (no interval data);
- identity tokens and identity fields do not change the features;
- changing the labels changes no feature.

### 5.4 Risk found: provider confidence codes

`conf_in` and `conf_out` are in the core feature set since Step 4. They are the mean provider confidence code at each endpoint.

**The finding.**
- On clean high-tier pairs, *equality* of the incoming and outgoing codes separates same from different identity with AUC 0.60.
- The codes are equal for 76% of positive pairs and 58% of negative pairs. Neither code alone separates (AUC 0.50).
- The provider's confidence code therefore carries a weak trace of its own identity continuity. That is close to the "provider
  continuity flag" the protocol forbids.

**Its effect on the models is negligible:**
- dropping the whole family D changes RF AUC by −0.0001;
- family D's permutation drop is 0.0002 for RF.

**Sensitivity re-run.** The selected `rf_reg/core_g` under the same nested protocol without `conf_in` and `conf_out`
(scratchpad, not a project output):

| | AUC | Log-loss | Top-1 | ECE | Coverage / false assoc. at 5% target | Coverage / false assoc. at 10% target |
|---|---|---|---|---|---|---|
| With `conf_*` (selected) | 0.840 | 0.478 | 0.716 | 0.006 | 0.276 / 0.045 | 0.426 / 0.095 |
| Without `conf_*` | 0.839 | 0.479 | 0.718 | 0.008 | 0.278 / 0.045 | 0.423 / 0.095 |

The results are practically identical, and the same hyperparameters were chosen in every fold. **The selected model's gain does
not depend on the provider confidence codes.** The script and its output are in the session scratchpad (`step5_noconf.py`,
`step5_noconf.json`); they are not part of `m11-ml-v2`.

**Recommendation.** Remove `conf_*` from the frozen feature set before any validation run (needs approval).

## 6. FOOTBALLTrace

**Three different things.** They are not equivalent, and none of them is the published system run on our data.

| | What it is | Status here |
|---|---|---|
| **A. Published FOOTBALLTrace correction** (`data/raw/122268.pdf`, transcribed in the Step 3 report) | Savitzky-Golay smoothing and resampling of *tracker-labelled* trajectories. A turn-cosine breakpoint is found near an occlusion that an **operator has confirmed on video**, and the tails are **always** exchanged at that breakpoint. | Not reproducible as an automatic matcher. It needs labelled trajectories through the encounter, which task B removes, plus operator confirmation. It has no decision of its own (Step 3: E0-Literal, evaluation-side only). |
| **B. E0-Adapted** (Step 3, a task B matcher) | The SG plus junction-sharpness cost on our anonymous tracklets, with a one-to-one assignment and abstention | Comparison baseline: false association 0.384, coverage 0.485 on the ten matches |
| **C. Features inspired by FOOTBALLTrace inside the classifier** (family F) | `ft_sharpness` and its competitor-relative version | Ablation only |

**Ablation result for C** (paired, all ten matches):

| Comparison | AUC | Coverage / false association at the 5% target | Coverage / false association at the 10% target |
|---|---|---|---|
| `rf/core` → `rf/core_f` | 0.790 → 0.790 | 0.150 / 0.037 → 0.149 / 0.038 | 0.295 / 0.096 → 0.286 / 0.088 |
| `hgb/core` → `hgb/core_f` | 0.791 → 0.790 | 0.144 / 0.036 → 0.142 / 0.038 | 0.278 / 0.078 → 0.276 / 0.070 |
| `rf/core_g` → `rf/core_fg` | 0.839 → 0.839 | 0.274 / 0.050 → 0.272 / 0.051 | 0.426 / 0.097 → 0.430 / 0.098 |
| `hgb/core_g` → `hgb/core_fg` | 0.842 → 0.842 | 0.266 / 0.043 → 0.275 / 0.042 | 0.424 / 0.089 → 0.416 / 0.091 |

- **Drop-F ablation:** RF +0.0007 (better in 4 of 10 matches), HGB −0.0004 (3 of 10).
- **Permutation drop:** at most 0.0014.
- **Conclusion:** no measurable benefit, so F is not in the selected model.

**Why there is no benefit.** The paper's premise is that a wrong connection shows up as a sharp turn. Here, 63% of high-tier
encounters contain a legitimate sharp turn, so sharpness alone separates poorly (only F: AUC 0.55–0.57).

**FOOTBALLTrace still has a role:**
- as the candidate-encounter methodology behind Step 1;
- as a published correction reference;
- as a source of smoothing and breakpoint ideas;
- as a baseline wherever a protocol with labelled trajectories (protocol L) applies.

**The scientific difference.**
- FOOTBALLTrace *corrects* trajectories after a human has confirmed a swap.
- Our system *decides* MATCHED / UNMATCHED / UNCERTAIN automatically, with calibrated probabilities, and abstains instead of
  forcing a correction.

## 7. Model comparison and coverage against false association

Held-out pooled high tier (4,276 incoming). The operating points were chosen on the inner training folds only.

| Model / features | AUC | AP | Top-1 | Log-loss | ECE (Platt) | Coverage / false assoc. at 5% target | Coverage / false assoc. at 10% target | Lower-tier AUC |
|---|---|---|---|---|---|---|---|---|
| LR / core (Step 4) | 0.753 | 0.713 | 0.673 | 0.584 | 0.025 | 0.033 / 0.077 | 0.196 / 0.087 | 0.689 |
| **RF / core (Step 4)** | 0.790 | 0.767 | 0.696 | 0.539 | 0.009 | 0.150 / 0.037 | 0.295 / 0.096 | 0.729 |
| RF-reg / core | 0.791 | 0.769 | 0.701 | 0.538 | 0.010 | 0.143 / 0.034 | 0.311 / 0.094 | 0.725 |
| HGB / core | 0.791 | 0.770 | 0.709 | 0.539 | 0.011 | 0.144 / 0.036 | 0.278 / 0.078 | 0.723 |
| MLP / core | 0.799 | 0.776 | 0.729 | 0.537 | 0.021 | 0.123 / 0.025 | 0.274 / 0.065 | 0.646 |
| LR / core+G | 0.800 | 0.767 | 0.707 | 0.540 | 0.029 | 0.055 / 0.064 | 0.260 / 0.088 | 0.743 |
| RF / core+G | 0.839 | 0.825 | 0.718 | 0.481 | 0.010 | 0.274 / 0.050 | 0.426 / 0.097 | 0.774 |
| **RF-reg / core+G (selected)** | **0.840** | 0.827 | 0.716 | **0.478** | **0.006** | **0.276 / 0.045** | **0.426 / 0.095** | 0.771 |
| HGB / core+G | 0.842 | 0.829 | 0.727 | 0.477 | 0.018 | 0.266 / 0.043 | 0.424 / 0.089 | 0.768 |
| MLP / core+G | 0.836 | 0.818 | **0.750** | 0.490 | 0.023 | 0.201 / **0.022** | 0.386 / 0.075 | 0.690 |

**Notes on the comparison.**
- **The hyperparameter search moved little.** `rf_reg` chose `max_features` 0.33 with leaf 5 in every fold, and the grid's
  log-loss range was only 0.484–0.509.
- **On the same features, RF, RF-reg and HGB are statistically indistinguishable.**
- **The MLP** has the best top-1 and the lowest false association, but less coverage at the 5% target. It is also clearly worse
  on the lower tier (AUC 0.690) and needs more calibration (raw ECE 0.068). The sample of 9 matches does not support preferring it.

**Uncertainty, selected model** (match bootstrap, 95%):
- coverage at the 5% target [0.263, 0.291], false association [0.029, 0.061];
- coverage at the 10% target [0.406, 0.444], false association [0.074, 0.116].

**Matched budgets and coverage levels** (held-out curve; descriptive, oracle, not used for any choice):

| Model | Coverage at FA ≤ 2% | at ≤ 5% | at ≤ 10% | at ≤ 15% | FA at coverage ≥ 30% | ≥ 40% | ≥ 50% |
|---|---|---|---|---|---|---|---|
| RF / core | 0.073 | 0.150 | 0.297 | 0.433 | 0.101 | 0.121 | 0.165 |
| RF-reg / core+G | 0.177 | 0.281 | 0.437 | 0.565 | 0.053 | 0.089 | 0.126 |
| HGB / core+G | 0.180 | 0.287 | 0.447 | 0.597 | 0.052 | 0.081 | 0.121 |
| MLP / core+G | 0.176 | 0.294 | 0.471 | 0.669 | 0.051 | 0.085 | 0.110 |

**Trade-off curve, selected model** (held-out, margin 0.5):

| Threshold | Coverage | False association |
|---|---|---|
| 0.95 | 0.162 | 0.020 |
| 0.90 | 0.232 | 0.035 |
| 0.85 | 0.292 | 0.050 |
| 0.80 | 0.355 | 0.069 |
| 0.75 | 0.418 | 0.095 |
| 0.70 | 0.483 | 0.118 |
| ≤ 0.60 | 0.612 | 0.163 |

At the same threshold the Step 4 RF commits 0.141 at 0.065 (threshold 0.90), and 0.276 at 0.093 (threshold 0.75).

**Aspirational target.** **No model reaches 50% coverage at < 10% false association**, even with oracle thresholds. The best is
the MLP curve, at 47% coverage for 9.9%.

**Comparison with the Step 2–3 baselines on the same ten matches** (encounter level, high tier):

| Method | Correct | Wrong | Coverage | False assoc. | Complete encounters |
|---|---|---|---|---|---|
| Nearest | 0.397 | 0.358 | 0.755 | 0.474 | 0.033 |
| Constant velocity | 0.433 | 0.338 | 0.771 | 0.439 | 0.132 |
| Hungarian + CV | 0.384 | 0.224 | 0.608 | 0.368 | 0.368 |
| Chaining | 0.417 | 0.223 | 0.640 | 0.349 | **0.408** |
| E0-Adapted | 0.299 | 0.187 | 0.485 | 0.384 | 0.285 |
| Step 4 RF, 5% / 10% | 0.144 / 0.267 | 0.006 / 0.028 | 0.150 / 0.295 | 0.037 / 0.096 | 0.095 / 0.180 |
| **Selected, 5% / 10%** | **0.264 / 0.385** | **0.012 / 0.041** | 0.276 / 0.426 | 0.045 / 0.095 | 0.214 / 0.343 |

At the 10% operating point the selected model makes **as many correct matches as Hungarian + CV** (0.385 vs 0.384), with **18% of
its wrong matches** (0.041 vs 0.224).

Complete-encounter accuracy is still lower than for chaining or Hungarian. It rewards committing on every incoming tracklet, so it
is not used to rank the models.

## 8. Results by match and by difficulty (selected model, held out, high tier)

**Per match at the 5% target** (coverage / false association):

| 3812 | 3816 | 10506 | 3814 | 3847 | 3836 | 3833 | 3815 | 3851 | 3829 |
|---|---|---|---|---|---|---|---|---|---|
| 0.280 / 0.041 | 0.266 / 0.000 | 0.266 / 0.034 | 0.251 / **0.083** | 0.309 / **0.076** | 0.319 / 0.063 | 0.274 / 0.012 | 0.269 / 0.033 | 0.268 / **0.084** | 0.238 / 0.000 |

- At the 10% target the per-match false association is 0.033–0.167, with 3847 at 0.167 and 3836 at 0.136.
- **The 5% target is a pooled tendency, not a per-match guarantee.** 3 of 10 matches exceed 7.5%, with small counts: each match has
  60–183 MATCHED decisions.
- **The thresholds themselves are stable:** 0.85–0.88 for the 5% target and 0.73–0.76 for the 10% target, met in every inner fold.

**Strata** (coverage / false association; the 5% target, then the 10% target; Step 4 RF at the 5% target in brackets):

| Stratum | n incoming | 5% target | 10% target | Step 4 RF, 5% |
|---|---|---|---|---|
| Gap < 1.5 s | 1,519 | 0.508 / 0.032 | 0.695 / 0.071 | [0.321 / 0.031] |
| Gap 1.5–2.5 s | 1,482 | 0.257 / 0.073 | 0.449 / 0.122 | [0.103 / 0.059] |
| Gap 2.5–4 s | 902 | 0.031 / 0.000 | 0.110 / 0.172 | [0.002 / 0.000] |
| Gap ≥ 4 s | 373 | 0.000 / – | 0.003 / 0.000 | [0.000 / –] |
| 2 players | 3,770 | 0.307 / 0.043 | 0.469 / 0.093 | [0.168 / 0.036] |
| 3+ players | 506 | 0.047 / 0.125 | 0.103 / 0.154 | [0.018 / 0.111] |
| Sharp turn | 2,732 | 0.168 / 0.083 | 0.309 / 0.143 | [0.080 / 0.082] |
| No sharp turn | 1,544 | 0.466 / 0.021 | 0.633 / 0.053 | [0.273 / 0.014] |
| Crossing | 1,865 | 0.254 / 0.051 | 0.409 / 0.085 | [0.128 / 0.038] |
| No crossing | 2,411 | 0.293 / 0.041 | 0.439 / 0.102 | [0.167 / 0.037] |
| Lower tier (exploratory) | 12,544 | 0.123 / 0.070 | 0.231 / 0.116 | [0.064 / 0.030] |

**Reading the strata.**
- **Coverage roughly doubles in every stratum.**
- **Sharp turns:** coverage doubles at the same false association (0.083 against 0.082). The difficulty is unchanged, but more
  sharp-turn cases are committed.
- **Long gaps are not solved.** Above 2.5 s the model abstains almost always, and the few commitments it makes at the 10% target
  have high false association.
- **Multi-player encounters are not solved.** Coverage is 5–10% and false association 12–15% (small n).
- **Lower tier:** the model transfers, with AUC 0.771, but false association is above target. This is expected, because the lower
  tier has more ESTIMATED observations and jumps.

**Natural suspicious cases (unverified, qualitative; 47 instances, 115 incoming):**
- At the 5% target: 9 MATCHED, 106 UNCERTAIN.
- **The known 3816 swap** (instance `1f98c38f556f0c95`, lower tier, three players, 5.9 s gap): all incoming are UNCERTAIN (p 0.31–0.43).
  It is neither confirmed nor refuted.

## 9. Selected model and threshold

**The model.** `rf_reg/core_g`:
- RandomForest with 300 trees, `max_features` 0.33, `min_samples_leaf` 5, class weight `balanced_subsample`, seed 0;
- inputs: the 39 core features and the 14 family G features, 53 in total;
- Platt calibration.

**Why it was chosen.** It came first under the a-priori rule. More importantly, the gain over the Step 4 RF passes the a-priori
improvement test (the bootstrap CIs in "Answer first"), and it is consistent across all ten matches: AUC is higher in every match.

**The final development model.** `models/final/rf_reg__core_g.joblib`, fitted on all ten matches with the same nested procedure:
- inner groups: {3812, 3816, 3836, 10506}, {3814, 3829, 3847}, {3815, 3833, 3851};
- Platt a = 1.156, b = −0.049.

**Operating points.**
- **Primary:** threshold 0.87, margin 1.0 log-odds, for the 5% false-association target. Inner out-of-fold: coverage 0.270, false
  association 0.048.
- **Relaxed alternative:** threshold 0.75, margin 1.0, for the 10% target. Inner: 0.417 / 0.099.

**Practical tie.** HGB / core+G and RF / core+G are statistically tied with the selected model. If a later step prefers one of
them, for example for speed or size, nothing here argues against it.

## 10. Failure cases

The most confident wrong matches of the selected model, held out at the 10% operating point, are in `failures.jsonl`. All 15 are
2-player encounters.

| Match / instance | p chosen / p reference | Pattern |
|---|---|---|
| 3814 `aa8a047724c6ce01` | 0.990 / 0.084 | Crossing with a sharp turn. The chosen continuation is smoother (CV meet-in-the-middle error 0.24 m against 1.26 m) |
| 3847 `a92a075c440793fe` | 0.990 / 0.059 | Crossing with a sharp turn. The reference path needs Hermite RMS 6.5 m/s² against 3.3 |
| 3851 `0bf0f7112383fb62` | 0.978 / 0.319 | Crossing. The chosen path is almost perfectly straight (0.14 m, 0.6 m/s²). This could be a reference error or a genuine swerve |
| 3836 `8770f91ff775c685` | 0.974 / 0.150 | No difficulty flag. The chosen continuation is smoother on every measure |
| 3836 `7826568f6b9d265f` | 0.966 / 0.083 | Crossing with a sharp turn. The reference path implies a 107° turn |
| 3812 `a3983a592ade84d8` | 0.969 / 0.143 | Also a top Step 4 failure (crossing with a sharp turn) |

**The common pattern.** The reference continuation is physically *less* plausible than the chosen one. Either the players really
swerved, which our features cannot express from task B endpoints, or the provider reference is wrong at these encounters.

- **12 of the 15** involve a crossing or a sharp turn.
- **These cases are the first candidates for independent (video) verification.**
- **The remaining failure modes are structural:** gaps of 2.5 s or more and encounters with 3 or more players produce almost no
  commitments.

## 11. Tests and verification

**Code changes.**
- `src/football_fatigue/matching/ml2.py` (new): football-aware features, nested protocol, diagnostics, ablations, learning curves,
  importance, bootstrap and outputs.
- `__main__.py`: the `ml2` subcommand.
- `__init__.py`: docstring.
- `ml.py`, `evaluation.py` and all Step 1–4 code are unchanged.

**New tests: `tests/test_matching_ml2.py`, 18 tests on synthetic instances.** They cover:
- **features:** football-aware feature values and row alignment; order invariance; rejection of task A and of interval data;
- **sharp turns:** a turn is cheap at low speed and costly at high speed;
- **long gaps:** reachability counts;
- **multi-player encounters:** assignment regret;
- **leakage:**
  - identity fields and labels do not change the features;
  - feature names contain no identity tokens;
  - inputs are not mutated;
- **families and ablation variants;**
- **grouped isolation:**
  - no held-out row reaches any fit;
  - inner fits exclude their own group;
  - flipping the held-out labels leaves the held-out probabilities and operating points unchanged;
- **reproducibility:** serial equals parallel;
- **operating points:** selection on an inner table; the vectorised rates agree with the Step 4 decision path; abstention and
  UNMATCHED;
- **calibration:** Platt;
- **bootstrap;**
- **persistence:** joblib round trip;
- **output safety:** refusal of non-development matches and of an existing output.

**Results.**
- `pytest tests/test_matching_ml2.py`: 18 passed.
- Full suite: **714 passed, 6 skipped, 1 warning** (an unrelated vision deprecation warning; 158 s).

**Output verification.**
- All 26 output hashes match the manifest of `m11-ml-v2`.
- Reloading the saved held-out-3814 fold model and calibrator and recomputing the features reproduces all 3,032 stored
  probabilities exactly (to 4-decimal rounding).
- Re-verified unchanged against their manifests: `m11-ml-v1` (31 files) and `m11-matching-v3` for 3812, 3814, 3816 and 10506.

**Housekeeping.**
- `m11-ml-v2` is 637 MB, mostly forest models in `models/`.
- A smoke run first wrote by mistake to `F:\c\Users\...` (a shell path translation). It was moved into the session scratchpad,
  which leaves an empty `F:\c\Users\DELL\AppData\Local\Temp\claude\...` folder tree outside the repository. Nothing was deleted.

## 12. Scientific limitations

**Data and protocol:**
- **References are not ground truth.** The provider identities are a provisional reference; label noise can be neither excluded nor
  measured (section 4).
- **Development only.** Ten development matches. The model selection used the outer folds, so the selected model's numbers are
  mildly optimistic. No validation or test match was used.
- **Selection bias.** The high tier excludes encounters longer than 6 s, encounters with more than 6 players, ESTIMATED endpoints
  and attribution problems. Results do not transfer to the hardest real situations, and the lower tier shows higher false
  association.
- **Task B is not an occlusion.** The withheld interval is a controlled removal of data from real trajectories. No claim about
  real occlusion or video tracking accuracy follows.

**Model behaviour:**
- **The target is pooled, not per match.** Per-match false association at the 5% target reached 0.084, and per-match counts are small.
- **The gain comes from global context.** The assignment-regret features make each pair's score depend on the whole encounter. That
  is legitimate and matcher-visible, but pairwise probabilities are therefore not independent across pairs.
- **The aspirational goal was not met.** Coverage of 50% at under 10% false association was not reached.
- **Unsolved difficulty.** Long gaps (2.5 s or more), multi-player encounters and sharp-turn crossings remain unsolved.

**Inputs:**
- **The provider confidence feature risk** in section 5.4.
- **The FOOTBALLTrace paper was not re-read for this step.** The PDF could not be rendered here (no PDF renderer is installed), so
  section 6 relies on the Step 3 transcription of `data/raw/122268.pdf`.

## 13. GO / NO-GO for the later frozen validation evaluation

**CONDITIONAL GO.** The improvement is:
- large: coverage roughly ×1.8 at the same false-association budget;
- consistent across all ten matches;
- produced by a protocol whose thresholds transfer between matches;
- not explained by overfitting.

That is enough to justify **one** frozen validation evaluation. The conditions, all of which need your approval:

1. **Freeze the configuration** before touching validation:
   - features: core + G, with `conf_in` and `conf_out` removed (section 5.4), refitted by the same nested procedure on these
     ten development matches;
   - operating points 5% and 10% as chosen on the inner folds.
2. **Pre-register the validation gate.** For example: pooled false association ≤ 7.5% at the 5% target, with coverage ≥ 20%.
   Report per match and stratum results, plus a match bootstrap.
3. **Run validation once.** No re-tuning afterwards; the test split stays locked.
4. **Report validation results as provider-reference agreement only.**

**NO-GO** for:
- any claim of identity accuracy, deployment or real occlusion;
- any claim about long gaps or multi-player encounters;
- further feature or model search on validation data.

**Recommended next research, before or alongside validation and needing approval:**
- independent verification of the top-confidence disagreements in section 10;
- dedicated work on encounters with 3 or more players and gaps over 2.5 s; the current features give almost no evidence there.
