# M12 report 7: Learned link scorer for N×M association (v1)

Status: **EXPLORATORY** (same data, folds and caveats as report 6; CPU only; lightweight models, no GPU, no neural embedding).
Date 2026-11-21.

- Code: `nxm_compare.py` (same run as report 6).
- Output: `data/processed/m11-nxm-solver-comparison-v1/comparison.json`.

## Design

**Pair features.** Each (incoming, outgoing) pair gets 18 motion features, computed from the public scenario only (truth never
read; tested):

- forward, backward and bidirectional log-d2;
- endpoint distance;
- Δt and required speed;
- incoming and outgoing speeds and their difference;
- heading cosine;
- constant-velocity miss distances, forward and backward;
- row- and column-relative cost and rank;
- N and M.

The texture ablation adds 2 provider-texture features (log second-difference roughness and its difference).

**Models.** Logistic regression (standardised, C = 1) and HistGradientBoosting (300 iterations, learning rate 0.05). Both are
fitted per outer fold on the 7 training folds' pairs; label = "same provider identity".

**Assignment.** Cost = −log p, solved with Hungarian, one-to-one.

| ID | Scorer | Null |
|---|---|---|
| L1 | learned + Hungarian | none (only the rectangular shape leaves tracks unmatched) |
| L2 | learned + Hungarian + explicit null | null cost −log τ per side, τ ∈ {0.05 … 0.9} tuned on training folds (τ = 0.3 in 22 of 24 fold-model choices) |
| L2t | L2, HGB, + texture features | Ablation |

**Not done:** GRU embeddings were not reused. The two-player GRU is not trained on N×M pairs, its lead is partly texture
(report 2), and no neural training was authorised.

## Results (8,540 scenarios, 32 matches, cross-fitted by match)

| # | System | Exact [95 % CI] | Link P | Link R | F1 | ID switches | False-link rate | Null recall | Null precision |
|---|---|---|---|---|---|---|---|---|---|
| 1 | S0 original Kalman + Hungarian | 0.416 [0.404, 0.427] | 0.588 | 0.633 | 0.610 | 1.05 | 0.412 | 0.083 | 0.544 |
| 2 | S1 improved motion + Hungarian | 0.483 [0.472, 0.492] | 0.652 | 0.702 | 0.676 | 0.85 | 0.348 | 0.086 | 0.570 |
| 2b | S2 improved + tuned nulls | 0.488 [0.475, 0.500] | 0.660 | 0.702 | 0.680 | 0.83 | 0.340 | 0.131 | 0.457 |
| 3 | L1 LR + Hungarian | 0.479 [0.468, 0.490] | 0.654 | 0.704 | 0.678 | 0.85 | 0.346 | 0.080 | 0.569 |
| 3 | L1 HGB + Hungarian | 0.544 [0.534, 0.553] | 0.718 | 0.773 | 0.745 | 0.65 | 0.282 | 0.100 | 0.714 |
| 4 | L2 LR + Hungarian + null | 0.505 [0.493, 0.516] | 0.688 | 0.701 | 0.695 | 0.74 | 0.312 | 0.278 | 0.349 |
| 4 | **L2 HGB + Hungarian + null** | **0.593** [0.579, 0.607] | 0.753 | 0.763 | 0.758 | 0.58 | 0.247 | 0.423 | 0.497 |
| ab | L2t HGB + texture | 0.600 [0.587, 0.613] | 0.757 | 0.774 | 0.765 | 0.57 | 0.243 | 0.405 | 0.534 |

**Paired differences.**
- L2 HGB vs S0: **+0.177** [0.164, 0.189].
- L2 HGB vs S2 (best physics): **+0.105** [0.094, 0.118].
- L1 HGB vs S0: +0.127 [0.118, 0.137].
- L2t vs L2 (texture): about +0.007. This is small, so the learned N×M scorer is *not* mainly a texture reader, unlike the
  two-player GRU.

## Breakdowns (exact: S2 / L1 HGB / L2 HGB)

| Stratum | S2 | L1 HGB | L2 HGB |
|---|---|---|---|
| N = 2 / 3 / 4 / 5 | 0.62 / 0.45 / 0.43 / 0.42 | 0.68 / 0.51 / 0.48 / 0.47 | 0.73 / 0.55 / 0.54 / 0.52 |
| complete | 0.672 | **0.747** | 0.726 |
| async_var | 0.645 | **0.735** | 0.691 |
| extra_out | 0.522 | **0.605** | 0.580 |
| unequal (null + distractor) | 0.117 | 0.096 | **0.373** |
| gap > 1.3 s | 0.318 | 0.380 | 0.425 |
| context visible | 0.525 | 0.535 | 0.562 |
| crossing | 0.399 | 0.457 | 0.500 |

**Explicit null trade-off.** It lifts `unequal` from 0.10 to 0.37, but costs about 0.02 to 0.04 exact where no null exists
(complete / async). A null mechanism tuned per scenario type is not possible without knowing the type. A single τ is the honest
setting.

## Confidence (descriptive precision vs coverage, margin ranking, no test-tuned threshold)

| System | 100 % | 80 % | 60 % | 40 % | 20 % |
|---|---|---|---|---|---|
| S2 physics | 0.660 | 0.695 | 0.709 | 0.719 | 0.705 |
| L1 HGB | 0.718 | 0.794 | 0.870 | 0.934 | 0.969 |
| L2 HGB | 0.753 | 0.831 | **0.912** | **0.973** | 0.995 |

Learned margins are an informative abstention signal: keeping the most confident 60 % of links gives 0.91 link precision. The
physics margin is not informative. A deployable threshold must still be fitted on a separate calibration partition, which has not
been done.

## Does learned link scoring beat physics?

**On this exploratory provider-derived prototype, yes:**
- +0.105 exact over the best tuned physics solver, with a match-level CI clear of zero;
- fewer identity switches (0.58 vs 0.83 per scenario);
- far better confidence ranking.

**Caveats:**
1. Truth is provider identity (not human-reviewed).
2. Contexts are 70 % ESTIMATED.
3. Selection on continuous provider chains.
4. Models are fitted on provider tracking, so they are out of domain for video (report 9).
5. Exact accuracy is still only 0.52 to 0.55 for 3 to 5 players.

This is **exploratory evidence**, not a validated result.
