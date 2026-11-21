# M11 Stage 2: 40-match candidate extraction, results (v1)

Status: **complete and verified, 2026-11-13.** No Stage 3 samples were built and nothing was trained.

| Item | Value |
|---|---|
| Output | `data/processed/m11-stage2-candidates-v1` (560 MB, 410 hashed files; manifest sha256 prefix `b6ab035520aa5d81`) |
| Command | `python -m football_fatigue.window_intersections.local --matches all --out m11-stage2-candidates-v1 --split m11-stage2-split-v1` |
| Split | `m11-stage2-split-v1/split.json`, sha256 `66b2321a…6c0` (unchanged, verified by `load_split`) |
| Matches | the 40 FIFA development matches. No validation or test match was read (`dev_matches()` refuses them). |
| Runtime | 67.9 min in one invocation with no crash (median 93.5 s per match, max 116.6 s); peak RSS 3.1 GB |

Planning documents written from these results:
- `M11_STAGE3_SUPERVISED_DATASET_DESIGN_v1.md`
- `M11_BENCHMARK_PLAN_v1.md`

## 1. Verification

| Check | Result |
|---|---|
| `matches/<m>/match_done.json` present | 40 / 40; no `.partial` or temporary files remain |
| `checks.json` all_pass | True (descriptor, mask, state and dense checks) |
| `candidates_all_pass` (outcome follows checks; dependency on accepted rows only) | True |
| Regression vs. pilot-v2 (3812, 3816) | identical on common columns |
| Raw data unchanged (hashes before and after) | True |
| Coverage: every event with local minimum ≤ radius is linked to a candidate | True for all 40 matches (44,375 events) |
| `split` column | 32 matches `train_cv`, 8 matches `heldout` = 3822 3824 3834 3839 3844 10510 10514 10517 (exactly the approved set) |

## 2. Candidates and outcomes

- **Candidates.** 36,732 identity-free encounters, deduplicated from 56,672 episodes and 139,370 local minima (9,125 merged).
- **Crossing class:** local crossing 23,416; window-only crossing 6,522; no crossing 6,794.

| Gap | ACCEPTED | LOWER_CONFIDENCE_IDENTITY | REJECTED | accepted % | accepted, independent |
|---|---|---|---|---|---|
| 0.4 s | 31,635 | 27 | 5,070 | 86.1% | 23,852 |
| **0.8 s (primary)** | **31,219** | **23** | **5,490** | **85.0%** | **23,166** |
| 1.6 s | 30,406 | 15 | 6,311 | 82.8% | 21,920 |

- **Accepted per match** (0.8 s): minimum 541, median 771, maximum 1,078. Both periods are present in all 40 matches.
- **Against the projection.** The pilot-based projection was about 40,100 accepted at 0.8 s. The actual count is 31,219, or 78% of the projection. The two pilot matches (3812, 3816) are among the densest: 1,003 accepted per match against a 40-match median of 771. So the shortfall is match density, not a new rejection cause. The reject-reason mix matches the pilot.

### Rejection funnel at 0.8 s (each check alone; a candidate can fail several)

| Check | Fails |
|---|---|
| continuous_reference (one valid chain per player across gap ± 0.5 s) | 4,753 |
| episode_not_truncated | 2,209 |
| chord_speed_plausible (≤ 12 m/s) | 1,417 |
| no_identity_risk (on used samples) | 89 |
| window_in_period | 57 |
| no_corruption, no_jump_or_shift_flag (inside chain) | 0, 0 |

- **LOWER_CONFIDENCE_IDENTITY** (23): discontinuity candidate near the used samples 22, label transfer 2, substitution window 1.
- **What the gap length changes.** Rejection grows with the gap almost entirely through `continuous_reference`: 4,297 → 4,753 → 5,589.

## 3. Accepted samples at 0.8 s

### Split

| Split | Matches | ACCEPTED | independent | LOWER | REJECTED |
|---|---|---|---|---|---|
| train_cv | 32 | 25,331 | 18,777 | 17 | 4,513 |
| heldout | 8 | 5,888 | 4,389 | 6 | 977 |

### Crossing class and cohort states

- **Crossing class:** local 19,985; window-only 5,920; no crossing 5,314.
- **Cohort states:**

| Cohort | PASS | FAIL | UNDETERMINED | no linked event |
|---|---|---|---|---|
| F1 τ0.30 | 13,112 | 14,256 | – | 3,851 |
| F2 primary | 12,696 | 14,270 | 402 | 3,851 |
| F2 R0.50 | 23,605 | 3,581 | 182 | 3,851 |
| F3 primary | 8,397 | 18,958 | 13 | 3,851 |
| F4 ε0.25 | 20,240 | 7,038 | 90 | 3,851 |
| F4 tier | 15,808 | 11,547 | 13 | 3,851 |

### Diagnostics (never change the outcome)

- **near_coincidence (d_min < 0.05 m): 3,502 (11.2%).**
  - 3,395 of them are local crossings.
  - 2,008 are fully ESTIMATED for both players in the gap.
  - 2,114 are opposing teams and 1,388 are the same team.
  - At d ≈ 0, the motion before and after is the only evidence for these, so they are expected to be the hardest stratum.
- **jump_transfer_risk: 471 (1.5%).**
  - Sides: a_start 175, b_start 165, a_next 90, b_next 90.
  - 1,892 accepted candidates have a chain-boundary jump somewhere in the window, outside the used samples.

### Tracking status (audit only, never an input)

| Status in gap | Accepted |
|---|---|
| both fully ESTIMATED | 16,054 (51.4%) |
| mixed | 12,237 |
| both fully VISIBLE | 2,928 (9.4%) |

The mean ESTIMATED share of the used samples is 0.67 per player.

Half of the accepted gaps are entirely estimated by the provider. The provider's estimate across an occlusion is itself an interpolation, so motion in those gaps may be smoother than reality. In those cases the "ground truth" continuity is the provider's own association. This is why the benchmark plan reports metrics by status combination and includes a status-texture ablation (dataset design §7; benchmark §7.2).

### Dependency

| Measure | Count |
|---|---|
| Overlap groups | 20,892 |
| Singleton groups | 16,745 |
| Candidates in groups of 2 or more | 14,474 |
| Largest group | 51 |
| Greedy independent subset | 23,166 (74.2%) |

Groups never cross matches, so grouped CV by match is leakage-safe.

## 4. Non-intersection controls (train/CV matches only, 0.8 s)

- **Available.** 4,323 controls in total. The strata are d band × slower chord-speed band: d ≤ strict radius 1,006; loose 3,317.

| Cohort | Cases | Controls usable at ≤ 1:1 | Achieved ratio | Max control weight |
|---|---|---|---|---|
| identity_free | 16,177 | 3,460 | 0.21 | 35.8 |
| F2_primary | 9,268 | 1,006 | 0.11 | 32.9 |
| F2_R0.50 | 15,468 | 3,393 | 0.22 | 33.8 |
| F3_primary | 6,780 | 1,006 | 0.15 | 32.1 |
| F4_tier | 12,311 | 2,394 | 0.19 | 34.4 |

- **Ratio.** Controls cover every case stratum, but only at about 1:5 to 1:9. Weights reach about 33–37 in the slow, close strata.
- **Recommendation.** Use controls as an evaluation stratum with capped weights, not as balanced training data (decision B2).

## 5. Framewise selection-bias audit

- **Purpose.** An independent framewise scan of raw AVAILABLE positions (all slot pairs ≤ 0.5 m, merged within 1 s), matched to the chain-based candidates.

| Quantity | Count |
|---|---|
| Framewise encounters | 40,639 |
| Matched to a candidate | 35,899 (88.3%) |
| Unmatched | 4,740 (11.7%) |
| Candidates with no framewise frame (sub-frame minima between frames) | 637 |

**Unmatched reasons:**

| Reason | Count |
|---|---|
| quarantine period (excluded by design) | 1,316 |
| duplicate position (provider writes identical xy for both) | 2,328 player-sides |
| no valid segment, jump nearby | 1,952 player-sides |
| unattributed or ambiguous | 32 |

- **Flags near unmatched encounters.** 2,979 of the 4,740 unmatched encounters (63%) have an identity or jump flag within 1 s. The pilot figure was 98%. Across 40 matches the unmatched set is less dominated by flags, mainly because quarantine periods account for 1,316.
- **Median unmatched encounter:** d_min 0.28 m, 4 frames.
- **Conclusion.**
  - The candidate set under-represents the messiest encounters: duplicates, jumps and identity-flagged periods.
  - The trained model's performance will therefore describe clean, continuous encounters. It must not be claimed for encounters the provider itself failed to track.
  - Unmatched framewise encounters are not used for training, because they have no reliable reference chain (decision D7).

## 6. Data-retention summary (goal: keep as much valid data as possible)

- **What was retained.** VISIBLE and ESTIMATED are equally eligible; the earlier VISIBLE-minimum check was removed and no label-reliability check was weakened. Across 40 matches, 85.0% of the identity-free candidates (31,219) are accepted at 0.8 s.
- **What was removed.** Nearly all removals are reference-continuity, truncation or plausibility failures (8,525 check failures among 5,490 rejected). Identity-risk removals are only 89. These are exactly the cases where the label cannot be trusted.
- **What sample size this gives.** 25,331 train/CV samples at 0.8 s (18,777 independent). That is enough for the planned tabular and small recurrent models (dataset design §10). Learning curves will confirm it.

## 7. Limitations

- **Projection miss.** Accepted is 22% below the projection, because the pilot matches are denser than typical.
- **Label circularity.** 51% of the accepted gaps are fully ESTIMATED on both sides (§3).
- **Selection bias.** The framewise audit shows a bias toward clean encounters (§5).
- **Thin controls.** The control ratio is about 0.2:1 (§4).
- **Gap tables are not independent.** The three gap tables are versions of the same encounters, not independent data.

## 8. Approvals needed before Stage 3

| # | Decision | Recommendation |
|---|---|---|
| D1 | LOWER_CONFIDENCE_IDENTITY (23) | evaluation-only table, excluded from training |
| D2 | Resampling grid | native 1/15 s, T = 30 (alternative: 0.1 s, T = 20, reusing existing code unchanged) |
| D3 | Pitch coordinates as input | no (sample frame from incoming tracks only) |
| D4 | Near-coincidence (3,502) and jump-risk (471) | keep in training; report with and without |
| D5 | Status-texture shortcut | no status input; ablations with smoothing and noise injection |
| D6 | Dependency weighting | unweighted default; 1/group-size as an ablation; independent subset for headline metrics |
| D7 | Framewise-unmatched encounters | not used (no reliable reference) |
| B1 | Install xgboost / LightGBM | not needed; use sklearn HistGradientBoosting (no install) |
| B2 | Controls | evaluation stratum only, weight cap 10, uncapped as a sensitivity check |
| B3 | CV scheme | grouped 8-fold by match (hashed, frozen) on 32 train/CV matches; LOMO check for the top 2 |
| B4 | Label convention | 0 = MATCHED, 1 = SWAPPED (flip the existing `m11_recovery` convention, with a test) |
| B5 | Optional identity-masking task | defer; not in v1 |
| B6 | Held-out protocol | one-time evaluation after `final_protocol.json` is frozen (benchmark §8) |
| — | Authorise Stage 3 implementation (sample builder) and then the benchmark | awaiting user |
