# M13 report 3: Confidence calibration and abstention for N×M association (v1)

Status: **EXPLORATORY.** The data are provider-derived (`m11-nxm-exploratory-v2`). Nothing here is video or human-verified.
Date 2026-11-21.

- Code: `nxm_assoc_v2.py` (same run as report 2).
- Output: `data/processed/m11-nxm-association-v2/{v2,v1}/calibration.json` and `test_decisions.parquet`.

## 1. Protocol: separate, match-disjoint partitions

For outer test fold f (frozen Stage 4 folds, 4 matches each):

| Partition | Folds | Used for |
|---|---|---|
| Fit | The other 5 folds | Fitting the HGB / LR link scorers only |
| Calibration | (f+1, f+2) mod 8 | Choosing everything else, listed below |
| Test | f | Touched once, to score |

**Chosen on the calibration folds only:**
- null τ;
- the hybrid weight;
- the isotonic link-probability calibrator;
- the decision-confidence model;
- the scenario-confidence model;
- every abstention threshold.

Partitions are disjoint and each scenario is tested exactly once (tested). *Physics S2* has no fitted model; its parameters are tuned
on a bounded subset of the 7 non-test folds, as in M12. M12 tuned τ on in-sample training predictions; that is replaced here.

**Four concepts, kept separate.**

| Concept | What it is | Example |
|---|---|---|
| **Link probability** | p(same player) for an (incoming, outgoing) pair | HGB score, isotonic-calibrated |
| **Null prediction** | A *decision*: the incoming player did not reappear | Hungarian picks the null at cost −log τ |
| **Decision confidence** | P(this decision, link *or* null, is correct) | Logistic on log margin, is-null, calibrated p, N, M |
| **Abstention** | *Refusing* to output a decision because its confidence is below a threshold | Threshold chosen on calibration folds |

A null is an answer ("the player left"); abstention is the absence of an answer. They are scored separately: null precision among
accepted decisions vs abstained share.

Margin = increase of the optimal total assignment cost when the decision is forbidden (for a null: when the row must link). It
measures how much the whole one-to-one solution depends on that decision.

## 2. Results on untouched test folds (v2; 63,390 decisions in 20,489 scenarios)

### 2.1 Calibration quality

| System | Decision accuracy | ECE (10 bins) | Brier | AURC (error) |
|---|---|---|---|---|
| S2 physics (margin-based confidence) | 0.610 | 0.016 | 0.237 | 0.364 |
| L2 LR | 0.605 | 0.027 | 0.200 | 0.217 |
| **L2 HGB** | **0.664** | 0.037 | **0.173** | **0.142** |

- Raw isotonic link probabilities, read as the confidence of a *chosen* link, are miscalibrated (HGB: ECE 0.088). Pair-level
  calibration does not transfer to selected links, because selection changes the distribution. The decision-level model is needed.
- S2's low ECE is uninformative. Its confidence hardly varies (mean about 0.61 everywhere), so it is "calibrated" only by
  predicting the base rate. **A calibrated but non-discriminating confidence is useless for abstention.**

### 2.2 Risk–coverage (selective decision accuracy when only the most confident decisions are kept)

| System | 100 % | 90 % | 80 % | 60 % | 40 % | 20 % |
|---|---|---|---|---|---|---|
| S2 | 0.610 | 0.615 | 0.616 | 0.616 | 0.632 | 0.662 |
| L2 LR | 0.605 | 0.638 | 0.671 | 0.741 | 0.806 | 0.905 |
| **L2 HGB** | 0.664 | 0.700 | 0.736 | 0.819 | 0.914 | 0.986 |

### 2.3 Operating points (threshold chosen on calibration folds, evaluated once on test)

| System | Rule (calibration folds) | Test coverage | Test selective accuracy | False-link rate (accepted links) | Null precision (accepted) |
|---|---|---|---|---|---|
| **L2 HGB** | Decision accuracy ≥ 0.95 (thresholds 0.837 to 0.864 across folds) | **32.1 %** | **0.949** | **4.9 %** | 0.927 |
| L2 HGB | 80 % coverage | 80.0 % | 0.736 | 25.0 % | 0.679 |
| L2 HGB | 60 % coverage | 60.1 % | 0.819 | 16.9 % | 0.762 |
| L2 LR | Accuracy ≥ 0.95 | 10.4 % | 0.945 | 5.1 % | 0.658 |
| S2 | Accuracy ≥ 0.95 | 0.01 % | n/a | n/a | n/a. The target was unreachable on calibration data in 5 of 8 folds |

**Scenario level** (the whole assignment correct; L2 HGB):
- ECE 0.039, Brier 0.169.
- The 90 % exact target chosen on calibration gives **0.894 exact at 31.1 % coverage** on test.
- Risk–coverage: 0.516 at 100 %, 0.699 at 60 %, 0.960 at 20 %.

**Null vs abstain (L2 HGB, all test decisions).**
- 11,611 null decisions with accuracy 0.643, and 51,779 links with accuracy 0.669.
- At the 95 % operating point, 67.9 % of decisions are abstentions, and accepted nulls are 92.7 % correct.

**Forced matching vs null-aware vs abstention.**
- L1 (forced, no null) is exact in 0.502 of scenarios.
- L2 (null-aware) is exact in 0.516.
- L2 with abstention keeps about 1 in 3 decisions at about 95 % accuracy.

## 3. Domain-shift stress (L2 HGB; the same calibration-fold threshold applied to every stratum)

| Stratum | Decisions | Accuracy | Mean confidence | ECE | Coverage at threshold | Selective accuracy (target 0.95) |
|---|---|---|---|---|---|---|
| Context ESTIMATED > 50 % | 44,352 | 0.719 | 0.702 | 0.033 | 40.0 % | 0.962 |
| Context ESTIMATED ≤ 50 % (partly visible) | 16,137 | 0.525 | 0.573 | 0.061 | 12.8 % | **0.854** |
| All visible | 2,901 | 0.592 | 0.607 | 0.059 | 17.3 % | **0.869** |
| Gap 0.7 to 1.0 s | 14,082 | 0.763 | 0.736 | 0.036 | 45.3 % | 0.957 |
| Gap > 1.3 s | 12,776 | 0.613 | 0.623 | 0.049 | 25.1 % | 0.940 |
| N ≥ 4 | 17,715 | 0.685 | 0.685 | 0.049 | 39.6 % | 0.960 |
| Null + new | 8,670 | 0.516 | 0.590 | 0.081 | 18.8 % | **0.920** |
| Label-risk edge | 2,766 | 0.703 | 0.695 | 0.048 | 41.1 % | 0.964 |
| **Position noise σ = 0.15 m on every sample** (video-like; every 2nd test scenario) | 31,699 | **0.490** (0.664 clean) | 0.570 | **0.095** | 11.4 % | **0.801** |

**Reading.**
- **The 95 % operating point holds on average only because 70 % of decisions are ESTIMATED-dominated.**
  - On visible or partly visible contexts the same threshold gives 0.85 to 0.87.
  - The confidence model is over-confident exactly where provider interpolation is absent.
  - This is the provider-texture problem again (M12 reports 2 and 6): confidence learned on interpolated data does not transfer
    to measured data.
- **Position noise of 0.15 m**, the M12 Session B envelope p90:
  - decision accuracy drops from 0.66 to 0.49;
  - selective accuracy at the operating point falls to 0.80, with ECE 0.095.
  - The models and their confidence are not robust to video-grade noise.
- Null + new scenarios are over-confident (mean confidence 0.59 vs accuracy 0.52).

## 4. Answer: can the model abstain with calibrated confidence?

**Partially, and only in-domain.**
- **What works.** On provider-derived, mostly-ESTIMATED test matches, a threshold chosen on separate calibration matches gives about
  95 % decision accuracy at about 32 % coverage (0.949 measured), and about 90 % whole-assignment accuracy at 31 % coverage. That
  is a genuine out-of-sample operating point; M12 had only descriptive curves.
- **What fails.**
  - The guarantee does **not** hold on visible contexts (0.85 to 0.87), on null + new scenarios (0.92) or under 0.15 m noise
    (0.80).
  - Visible data are scarce: 263 encounters, 2,901 decisions. Calibrating on them separately would rest on few matches and was not
    attempted.
- **Deployable?** No. A deployable threshold needs calibration data from the deployment domain: video tracks, with human-checked
  identity.
- **Physics confidence** (margins of the tuned Kalman solver) is not a usable abstention signal.
