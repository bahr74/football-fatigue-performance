# Phase 2, Task B: deterministic motion-ambiguity benchmark (development matches only)

**Date:** 2026-11-20 · CPU only · no learned parameter · no training.

**Code:**
- `src/football_fatigue/m11_phase2/baselines.py`: methods, Kalman filter, Hungarian with null, gating.
- `src/football_fatigue/m11_phase2/ambiguity.py`: runner and metrics.
- Tests: `tests/test_m11_phase2.py`.

**Command:** `.\.venv\Scripts\python.exe -m football_fatigue.m11_phase2 ambiguity` (216 s, about 0.6 GB RSS).

**Outputs:** `data/processed/m11-phase2-ambiguity-benchmark-v1/`
- `ambiguity_results.json`
- `per_sample_g0p80.parquet`: margins, tiers and context per sample; no tracking data
- `severe_failures_top25.parquet`
- `manifest.json`: input hashes, including Stage 3 manifest `6016fa6c…`, split `66b2321a…` and catalogue run manifest
  `a19f0f81…`; parameters; code hashes; output hashes

**Labels:**

| Label | Meaning |
|---|---|
| [V] | Computed in this run |
| [DOC] | Historical report |
| [EST] | Estimate |
| [HYP] | Interpretation |
| [REC] | Recommendation |

---

## 1. Protocol

**Population:**
- The validated Stage 3 samples `m11-stage3-samples-v1` on the 32 train/CV matches, loaded only through the Stage 3 adapter. It
  refuses held-out matches and verifies hashes.

| Table | Samples | MATCHED / SWAPPED | Independent | Matches |
|---|---|---|---|---|
| **g0p80** (gap 0.8 s centred on t\*) | **25,331** | 12,702 / 12,629 | **18,777** | 32 |
| g0p40 (0.4 s) | 25,668 | 12,771 / 12,897 | 19,342 | 32 |
| g1p60 (1.6 s) | 24,661 | 12,295 / 12,366 | 17,759 | 32 |
| g0p80_shift (centre shifted by ±0.1–0.3 s) | 25,172 | 12,504 / 12,668 | 18,648 | 32 |

The classes are balanced by the Stage 3 order randomisation. Headline metrics use the **independent subset** (greedy
non-overlapping per shared player). Confidence intervals resample **matches** (2,000 replicates).

**Input contract:**
- Each method sees only:
  - two incoming tracklets (≤ 2 s before the gap);
  - two anonymous outgoing tracklets (≤ 2 s after the gap);
  - masks, times and the gap length.
- No method reads the hidden interval, any label or any metadata.
- Every cost is pairwise, so exchanging the outgoing order negates the margin exactly. This is tested.
- **Cross-check:** `nearest`, `cv_fwd` and `cv_bidir` reproduce the historical `m11_recovery.motion` margins to within 1e-7 m on
  match 3812 [V].
- The `nearest` independent BA, 0.3538, equals the historical `phys_B0` 0.354 [DOC].

**Methods** (details in the `baselines.py` docstring):

| Method | Definition |
|---|---|
| `nearest` | Last incoming vs first outgoing point |
| `const_pos` | Mean of the last / first 0.5 s |
| `cv_fwd` / `cv_bidir` | Least-squares constant velocity over 0.5 s; forward to the first outgoing point, or both sides to the gap centre |
| `ca_fwd` / `ca_bidir` | Quadratic over 1 s with stability constraints (≥ 5 points over ≥ 0.5 s, else CV fallback, used in 140 samples; \|a\| ≤ 7 m/s²; speed ≤ 12 m/s); `ca_fwd_raw` unconstrained |
| `kf_fwd` | Constant-velocity Kalman filter on the incoming tracklet, predicted to the first outgoing observation. Cost = Mahalanobis d² + log det S; gate χ²₂(0.99) = 9.21 |
| `kf_bidir` | Forward filter of incoming, backward filter of outgoing, compared at the gap centre |
| Hungarian | `assign()` (`linear_sum_assignment`, optional null cost) is the solver for every cost; for 2 × 2 it equals comparing the two permutations (tested against brute force) |
| `ocm_dir` | OC-SORT-*inspired* direction-consistency cue only: angle between the observed heading and the direction to the candidate. **Not OC-SORT**: no boxes, IoU, ORU or OCR |

**Kalman noise, chosen without labels:**
- Minimum one-step predictive NLL per observation of all observed tracklets over σ_a ∈ {1, 2, 4, 8, 16, 32} m/s² and σ_m ∈ {0.02,
  0.05, 0.1, 0.2} m.
- Selected: **σ_a = 8 m/s², σ_m = 0.05 m**, an interior optimum (157 s).
- No tuning on labels; no threshold fitted on evaluation data.

## 2. Main results (g0p80, independent subset) [V]

| Method | BA | 95 % CI (match bootstrap) | Per-match BA, min–max | Ties |
|---|---|---|---|---|
| nearest | 0.354 | 0.346–0.362 | 0.298–0.398 | 0 |
| const_pos | 0.331 | 0.324–0.338 | 0.275–0.381 | 0 |
| cv_fwd | 0.611 | 0.601–0.621 | 0.555–0.703 | 0 |
| cv_bidir | 0.525 | 0.515–0.536 | 0.460–0.611 | 0 |
| ca_fwd (constrained) | 0.655 | 0.647–0.663 | 0.609–0.723 | 0 |
| ca_fwd_raw | 0.656 | 0.648–0.665 | 0.611–0.728 | 0 |
| ca_bidir | 0.592 | 0.583–0.601 | 0.545–0.690 | 0 |
| **kf_fwd** | **0.667** | **0.656–0.676** | 0.609–0.725 | 0 |
| kf_bidir | 0.553 | 0.541–0.563 | 0.483–0.607 | 0 |
| ocm_dir | 0.643 | 0.638–0.647 | 0.614–0.669 | 571 |

**Confusion matrices:** in `ambiguity_results.json`. For `kf_fwd`: TN 6,302, FP 3,096, FN 3,166, TP 6,213. Errors are symmetric
between the classes.

**Kalman gating:**
- 25,327 of 25,331 samples admit a full assignment inside the gate, so it abstains on 4. Only 28 links are gated out.
- **The 0.99 gate is almost never binding at 0.5 m.** All candidates are within metres, so gating cannot resolve close-contact
  ambiguity.

**For reference: saved learned-model OOF predictions** [DOC, read only; seed 0; the same samples, folds and independent subset]:

| Model | Independent BA |
|---|---|
| GRU gap-last | 0.900 |
| HGB | 0.870 |
| SVM | 0.859 |
| LSTM gap-last | 0.858 |

**The best deterministic baseline is 0.23 BA below the GRU.**

**Robustness note: the centred design defeats bidirectional methods [V/HYP].**
- With the gap centred on the closest approach, both players' extrapolations meet near the same point at t = 0. The bidirectional
  cost difference is then small and noise-dominated.
- `cv_bidir` (0.525) and `kf_bidir` (0.553) lose to their forward versions.
- When the gap centre is shifted off t\* (g0p80_shift), the ordering reverses: `ca_bidir` 0.701 and `kf_bidir` 0.690 vs `kf_fwd`
  0.641.
- **The difficulty of this benchmark is partly a product of the centring convention.** Any paper must state it, and a variable-centre
  evaluation is needed (§7).

## 3. Gap length [V]

Same encounters; independent BA.

| Method | 0.4 s | 0.8 s | 1.6 s | 0.8 s shifted |
|---|---|---|---|---|
| cv_fwd | 0.757 | 0.611 | 0.561 | 0.640 |
| ca_bidir | **0.855** | 0.592 | 0.505 | **0.701** |
| kf_fwd | 0.808 | **0.667** | 0.533 | 0.641 |
| kf_bidir | 0.825 | 0.553 | **0.448** | 0.690 |
| ocm_dir | 0.750 | 0.643 | **0.587** | 0.636 |

- Physical association degrades quickly with the gap.
- At 1.6 s, bidirectional Kalman is **worse than chance** (0.448): extrapolated meeting points are systematically misleading.
- Simple direction consistency is the most robust cue at long gaps.

## 4. Strata (g0p80, independent; kf_fwd BA unless noted) [V]

| Stratum | Levels: n → kf_fwd BA |
|---|---|
| d_min band | < 0.05 m: 2,129 → 0.603 · 0.05–0.25: 6,910 → 0.640 · 0.25–0.5: 9,738 → 0.699 |
| Relative-speed band | < 1 m/s: 8,111 → 0.728 · 1–2: 5,376 → 0.641 · 2–4: 4,212 → 0.604 · ≥ 4: 1,078 → 0.573 |
| Crossing class | local crossing: 12,193 → 0.645 (nearest **0.142**) · window-only: 3,473 → 0.715 · none: 3,111 → 0.697 (nearest **0.813**) |
| Status in gap (Stage 3) | both fully ESTIMATED: 9,730 → **0.745** · both fully VISIBLE: 1,881 → 0.675 · mixed: 7,166 → **0.558** |
| Catalogue status (edge) | both ESTIMATED: 10,452 → 0.736 · both VISIBLE: 2,674 → 0.647 · mixed: 5,651 → 0.548 |
| Context (catalogue) | isolated: 16,479 → 0.671 · sequential: 1,457 → 0.598 · simultaneous: 841 → 0.695 |
| Simultaneous group size | 2: 17,936 → 0.665 · 3: 643 → 0.689 · 4+: 198 → 0.713 |
| Reliability | nominal: 16,350 → 0.677 · reduced: 2,427 → 0.593 |
| Label risk | no: 18,508 → 0.668 · yes: 269 → 0.540 |
| Clone-like pair-period | no: 17,745 → 0.664 · yes: 1,032 → 0.707 |
| Boundary-uncertain edge | no: 16,445 → 0.657 · yes: 2,332 → 0.737 |
| Team relation | opposite: 12,670 → 0.663 · same: 6,107 → 0.674 |

**Readings [V/HYP]:**
1. **Difficulty rises with closeness and with relative speed.** It is highest for sub-5 cm contacts and for fast crossings.
   - The "nearest endpoint" rule is anti-informative exactly for local crossings: the players have passed each other.
2. **Status is the largest single modifier.**
   - Fully ESTIMATED gaps are the *easiest* (0.745); mixed ones the hardest (0.558).
   - Consistent with Task A, interpolated motion is smoother and more predictable.
   - A mixed gap also contains a status transition, often with a position correction [HYP].
3. **Multi-player context does not make the two-player decision harder for physics baselines** at 0.5 m.
   - Simultaneous groups are slightly easier.
   - **This does not mean multi-player association is easy.** Stage 3 still presents only the correct two candidates; distractors are
     not offered.
4. **Clone-like and boundary-uncertain samples are easier for physics:** grazing, slow, near-parallel contacts.
5. **Per match:** kf_fwd BA ranges 0.609–0.725. The match bootstrap CI width is ±0.01.

## 5. How many encounters are genuinely ambiguous for simple physical association?

**Definitions** (labels are used only to score, so these are *evaluation strata, not deployable triggers*):

| Definition | Independent (18,777) | All (25,331) |
|---|---|---|
| Best single method (kf_fwd) wrong | **6,262 (33.3 %)** | 8,764 (34.6 %) |
| Majority of the five core methods wrong (≥ 3 of cv_fwd, cv_bidir, ca_bidir, kf_fwd, kf_bidir) | 7,236 (38.5 %) | 10,105 (39.9 %) |
| **All five core methods wrong ("physics-hard")** | **1,897 (10.1 %)** | **2,701 (10.7 %)** |
| All five correct ("physics-easy") | 4,308 (22.9 %) | 5,687 (22.5 %) |
| Contested (methods disagree) | 12,572 (67.0 %) | 16,943 (66.9 %) |

**Where are the physics-hard cases?** (independent)

| Stratum | Physics-hard share |
|---|---|
| d_min < 0.05 m | 11.4 % |
| 0.05–0.25 m | 12.2 % |
| 0.25–0.5 m | 8.3 % |
| Mixed status | 18.8 % |
| Both VISIBLE | 9.2 % |
| Both ESTIMATED | **5.6 %** |
| Relative speed ≥ 4 m/s | 18.7 % |
| Relative speed < 1 m/s | 5.7 % |
| Catalogue cv_confusable | 29.1 % |
| Catalogue marginal | 2.5 % |
| Catalogue separable | 0.1 % |

**Answer at 0.5 m [V]:**
- About **one encounter in ten** is misassigned by *every* simple physical model.
- About **one in three** is misassigned by the best one (a forward Kalman filter).
- Only **about one in four or five** is solved by all of them.
- The hard cases are concentrated in fast crossings and mixed-status gaps. They are under-represented among ESTIMATED-only contacts.

**The catalogue `cv_margin` indicator is not independent ground truth [V].**
- It ranks difficulty well: 1,726 of the 1,897 physics-hard samples fall in its `cv_confusable` tier, and only 9 in `cv_separable`.
- But it is computed from the provider-chain identities: margin = cost(swap) − cost(provider assignment). **It therefore uses the
  label.**
- It is a valid *descriptive difficulty stratum* for evaluation and for dataset design. It is **not** a deployable pre-event trigger,
  and it is not model-independent.
- The agreement above is partly by construction: it is a CV-family method scored on similar quantities.

**Broader radii [EST]:** no labelled samples exist beyond 0.5 m, so none were built.
- Assume the conditional rates P(physics-hard | catalogue tier) measured at 0.5 m transfer to other radii:
  - confusable 0.291;
  - marginal 0.025;
  - separable 0.0014.
- Applied to the catalogue tier counts, the implied physics-hard *edges* are about **2.1 k / 3.5 k / 5.9 k / 7.2 k / 7.9 k** at
  0.25 / 0.5 / 1 / 1.5 / 2 m.
- The implied kf_fwd-wrong edges are about 7.1 k / 12.0 k / 20.8 k / 26.1 k / 29.5 k.

**Caveats:**
- The unit is edges, not independent samples.
- The 0.5 m calibration of these rates almost certainly does not transfer exactly: wider contacts are slower and longer.
- They are an order-of-magnitude guide for dataset v2 only.
- **The qualitative conclusion is robust:** the number of physically hard encounters keeps growing beyond 0.5 m, slowly, while the
  population grows fast. A distance-only expansion dilutes difficulty.

## 6. Learned models within physics tiers [DOC predictions, V analysis]

**Saved Stage 4 OOF predictions** (seed 0), accuracy on the independent subset:

| Model | Physics-easy | Contested | Physics-hard | cv_confusable | cv_separable |
|---|---|---|---|---|---|
| GRU gap-last | 0.947 | 0.883 | **0.910** | 0.858 | 0.956 |
| HGB | 0.951 | 0.842 | 0.868 | 0.806 | 0.948 |
| SVM | 0.945 | 0.828 | 0.866 | 0.797 | 0.947 |
| LSTM gap-last | 0.918 | 0.834 | 0.880 | 0.810 | 0.936 |

**Finding [V], interpretation [HYP]:**
- All four learned models are **more accurate on physics-hard samples than on contested ones**, and are 87–91 % accurate where every
  physics model fails.
- Two explanations need to be separated before any claim:
  - **(a)** the models learned genuine non-linear motion cues (turns, deceleration);
  - **(b)** they exploit provider-specific signals: interpolation texture [DOC texture probe], or systematic provider smoothing at
    crossings.
- This is the most important open question for the paper's central claim.
- **[REC]** Repeat this table with the texture-controlled inputs (noise / smoothing ablations) in the fair re-benchmark (Task D,
  E013–E014). Report it on both-VISIBLE samples separately: at 0.5 m there are 2,674 independent both-VISIBLE samples.

## 7. Severe failures [V]

- `severe_failures_top25.parquet` lists the physics-hard samples whose CV bidirectional margin points most strongly to the wrong
  assignment: 9.2 to 11.8 m of summed endpoint error.
- 19 of 25 are at relative speed ≥ 4 m/s; 23 of 25 are local crossings; 20 of 25 have mixed status.
- **None carries a label-risk flag.**
- **[HYP]** A 10 m motion contradiction within 0.8 s at sprint speed is hard to explain physically. These are prime suspects for
  **provider identity swaps at crossings** that the current flags do not catch. They should be human-reviewed before they are kept
  as labels.

## 8. Runtime and memory [V]

| Item | Value |
|---|---|
| g0p80, all ten methods | 9.5 s for 25,331 samples |
| …motion | 2.3 s |
| …Kalman | 7.2 s |
| …OCM | 0.04 s |
| Per sample (all methods) | 375 µs |
| RSS | ≈ 0.58 GB (includes loading all four tables) |
| Label-free Kalman selection | 157 s (24 configurations) |
| Whole run | 216 s |

Single CPU process.

## 9. Limitations

- Two-player, simultaneous hiding, correct candidates only. **No distractors, no null outgoing tracks, no asynchronous loss.**
  Deterministic methods with gating and null (`assign(..., null_cost)`) are implemented and tested, but not yet benchmarked, because
  no N × M population exists.
- Baselines see Stage 3's sample frame (rotation/translation from incoming data only). It is invariant for all methods here.
- The constant-acceleration constraints (`A_MAX` 7 m/s², `V_MAX` 12 m/s) were set a priori. The constrained and unconstrained
  versions differ by 0.001 BA, so the choice is immaterial here.
- `ocm_dir` ties (571 independent samples, slow incoming players) score 0.5.
- The learned-model comparison uses historical predictions on the same population. The models were selected on these same OOF
  results [DOC], so their numbers are selection-optimistic.

## 10. Answer to the research question

**How many football encounters are genuinely identity-ambiguous for simple physical association, and how does that change with
distance, interaction structure and trajectory quality?**

**At the validated 0.5 m / 0.8 s population [V]:**
- 10 % are wrong for every physical model.
- 33 % are wrong for the best one.
- Only 23 % are unanimous and correct.

**Difficulty increases with:**
- closeness (d_min < 0.25 m);
- relative speed;
- local crossings;
- gap length (best BA 0.86 at 0.4 s, 0.67 at 0.8 s, 0.59 at 1.6 s);
- *mixed* VISIBLE/ESTIMATED status.

**Difficulty decreases with:**
- fully ESTIMATED status;
- clone-like and grazing contacts;
- a gap centre that is not exactly at closest approach.

**Group context does not raise two-candidate difficulty.**

**Beyond 0.5 m [EST]:**
- Hard cases continue to exist: about 8 k hard edges at 2 m by the transfer estimate.
- They form a shrinking fraction of the population.
- **A dataset defined by an ambiguity criterion fixed before labelling, not by distance alone, is needed to study them efficiently**
  (see `m11_dataset_v2_design.md`).
