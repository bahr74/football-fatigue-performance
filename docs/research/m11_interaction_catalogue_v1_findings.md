# Interaction Catalogue v1: counts and findings (32 train/CV development matches)

**Source:** `data/processed/m11-interaction-catalogue-v1/summary/catalogue_summary.json` and `summary/tables.md`.

**Definitions:** `docs/research/m11_interaction_catalogue_v1_methodology.md`.

**Status of the numbers:**
- **Every number is an exact count** from the catalogue built on 2026-11-19 (the run manifest is in the catalogue root).
- No estimate is used, except where a line is marked **[EST]**.
- No held-out, official validation/test or Session B data was accessed.
- No model was trained or evaluated.

**Labels:**

| Label | Meaning |
|---|---|
| [V] | Exact count from the catalogue |
| [HYP] | Interpretation |
| [REC] | Recommendation |

**Not observable:** nothing below is a statement about visual occlusion.

---

## 0. Validation status [V]

**Reproduction checks:**

| Check | Result |
|---|---|
| Historical Stage 2 reproduction at R = 0.5 m | **Exact**: 45,989 episodes = 45,989 edges; 29,861 candidates = 29,861 edge chains. Every key, t\* and d_min identical, in 32/32 matches |
| Window-intersections v1 reproduction | **Exact**: 581,786 crossings; 767,067 pair-windows with ≥ 1 crossing; 246,532 pair-windows with > 1, in 32/32 matches |
| Structure | Passed in 32/32 matches: unique ids, d_min ≤ R, t\* inside the edge, radii nested, every edge in exactly one group per (R, δ) |

**Tests and runtime:**
- 14 focused synthetic tests pass (`tests/test_interaction_catalogue.py`).
- Runtime: 3,622 CPU-s in total, 113 s median per match, 4 workers, about 16 min wall time. Output size 199 MB.

**Data and conventions:**

| Item | Value |
|---|---|
| Frames in catalogued periods | 5,864,182 |
| Player chains (nodes) | 87,011; median duration 0.6 s, p90 156 s |
| Pitch | 105 × 68 m in all 32 matches; the centred origin is supported in all 32 |
| Timestamps | Strictly increasing; no frame gaps; max step 0.0334 s |

---

## 1. Counts by radius (exact) [V]

| Quantity | 0.25 m | **0.5 m** | 1.0 m | 1.5 m | 2.0 m |
|---|---|---|---|---|---|
| Proximity edges | 21,882 | **45,989** | 104,842 | 161,663 | 216,551 |
| Edge chains (same pair, gap ≤ 1 s) | 14,769 | **29,861** | 68,345 | 104,885 | 138,950 |
| Simultaneous groups (δ = 0) | 21,308 | 42,257 | 83,428 | 110,953 | 128,082 |
| Temporally connected groups (δ = 1 s) | 13,423 | 24,667 | 48,053 | 62,260 | 69,194 |
| Edges per match, min / median / max | 457 / 648 / 1,101 | 1,003 / 1,378 / 2,273 | 2,516 / 3,253 / 5,172 | 4,019 / 4,900 / 7,457 | 5,362 / 6,621 / 9,822 |
| Distinct pairs (match level) | 5,629 | 7,404 | 9,159 | 10,059 | 10,593 |
| Distinct player-id pairs (across matches) | 5,094 | 6,554 | 7,961 | 8,650 | 9,041 |
| Players / player-match identities involved | 554 / 967 | 557 / 977 | 560 / 983 | 560 / 984 | 560 / 984 |
| Same-team edges | 9,452 (43 %) | 18,918 (41 %) | 37,484 (36 %) | 56,030 (35 %) | 75,482 (35 %) |

### Sensitivity relative to 0.5 m

| Quantity | 0.25 m | 0.5 m | 1.0 m | 1.5 m | 2.0 m |
|---|---|---|---|---|---|
| Edges | 0.48 | 1 | 2.28 | 3.52 | 4.71 |
| Edge chains | 0.49 | 1 | 2.29 | 3.51 | 4.65 |
| Connected groups | 0.54 | 1 | 1.95 | 2.52 | 2.81 |
| Distinct pairs | 0.76 | 1 | 1.24 | 1.36 | 1.43 |

**Reading [V/HYP]:**
- Counts roughly double per radius step up to 1 m and then grow sub-linearly.
- *Independent-ish* units (connected groups) grow far more slowly than edges: ×2.8 against ×4.7 at 2 m. Wider radii mostly add
  repeated or concurrent contacts within the same play.
- Distinct pairs saturate: at 2 m, 10,593 of the ≈ 15,000 possible slot pairs [EST: about 31 slots per match].

### Exact versus estimated

The review's projection was 50–70 k merged candidates at 1 m and 150–250 k at 2 m [EST].

| Radius | Estimate | Exact (edge chains) | Assessment |
|---|---|---|---|
| 1 m | 50–70 k | 68,345 | Inside the range |
| 2 m | 150–250 k | 138,950 | Slightly below the range |

## 2. Repeated interactions per pair and per player [V]

| Quantity | 0.25 m | 0.5 m | 1.0 m | 2.0 m |
|---|---|---|---|---|
| Pair-periods with 1 / 2 / 3–4 / 5–9 / 10+ edges | 2,855 / 1,436 / 1,304 / 1,018 / 355 | 2,833 / 1,685 / 1,975 / 1,980 / 1,213 | 2,298 / 1,679 / 2,230 / 3,049 / 3,456 | 1,680 / 1,258 / 2,105 / 3,378 / 6,819 |
| Edges in pair-periods with ≥ 10 edges | 5,227 (24 %) | 20,095 (44 %) | 71,128 (68 %) | 182,119 (84 %) |
| Edge chains made of ≥ 2 edges | 3,393 | 7,384 | 17,148 | 36,866 |
| Edges per player-match, p50 / p90 / max-ish (p99) | 41 / 88 / 142 | 89 / 178 / 259 | 212 / 392 / 543 | 453 / 787 / 1,048 |

**Repeated contact is the norm, not the exception.**
- At 0.5 m, 44 % of all edges come from pairs that meet ≥ 10 times in a half.
- At 2 m this rises to 84 %: markers and defensive partners [HYP].
- Statistics on edges therefore have strong within-pair dependence.

## 3. Crossings and multiple intersections per window [V]

| Quantity | Value |
|---|---|
| Geometric crossings (\|tA − tB\| ≤ 5 s) | **581,786** (13,491–22,524 per match), 11,199 distinct pairs |
| …synchronous (\|tA − tB\| ≤ 0.3 s) | **33,887 (5.8 %)** |
| …median \|tA − tB\| | 2.44 s (p10 0.50 s) |
| …median separation when the first player is at the point | **3.29 m** (p10 0.50 m) |
| …separation ≤ 0.25 / 0.5 / 1.0 / 2.0 m | 29,545 / 57,957 / 111,108 / 201,565 |
| Pair-windows with 1 / 2 / 3 / 4 / 5+ crossings | 520,535 / 126,095 / 52,568 / 21,269 / 46,600 |
| Pair-windows with > 1 crossing | **246,532** (max 252 crossings in one pair-window) |
| Pair-windows with a synchronous crossing / with > 1 synchronous crossing | 101,147 / **23,306** |
| Crossings that v1's first-intersection rule never used | 243,280; **all kept here** |

**Finding [V/HYP].** Most geometric crossings are **not interactions**.
- 94 % are asynchronous: one player passes a point seconds after the other.
- The median separation at the crossing is 3.3 m.
- The 243 k crossings v1 discarded are overwhelmingly of this kind.
- What the first-intersection rule genuinely lost is the **23,306 pair-windows with several synchronous crossings** (weaving or
  repeated passes). The catalogue now keeps each one.
- Pair-windows with up to 252 crossings come from two players jittering around the same spot. They are noise in the crossing
  definition, not 252 events.

**Edges with a crossing inside them:**

| Quantity | 0.25 m | 0.5 m | 1.0 m | 2.0 m |
|---|---|---|---|---|
| Edges containing ≥ 1 synchronous crossing | 12,156 (56 %) | 16,341 (36 %) | 20,329 (19 %) | 21,464 (10 %) |

## 4. Multi-player groups, simultaneous and sequential interactions [V]

**Groups by number of players:**

| Quantity | 0.25 m | 0.5 m | 1.0 m | 1.5 m | 2.0 m |
|---|---|---|---|---|---|
| Simultaneous groups with 2 / 3 / 4 / 5+ players | 21,073 / 219 / 15 / 1 | 41,058 / 1,004 / 145 / 50 | 79,115 / 3,268 / 561 / 484 | 102,955 / 5,946 / 1,114 / 938 | 115,747 / 9,108 / 1,783 / 1,444 |
| Largest instantaneous component 3 / 4 / 5+ | 225 / 9 / 1 | 1,060 / 116 / 23 | 3,492 / 519 / 302 | 6,399 / 948 / 651 | 9,824 / 1,584 / 927 |
| Connected groups (1 s) with 2 / 3 / 4 / 5+ players | 12,554 / 701 / 126 / 42 | 22,259 / 1,736 / 393 / 279 | 41,915 / 4,330 / 909 / 899 | 52,313 / 6,668 / 1,777 / 1,502 | 55,708 / 8,342 / 2,561 / 2,583 |
| Largest group, simultaneous / connected | 5 / 8 | 8 / 14 | 15 / 18 | 19 / 21 | 21 / 21 |

**Edges by context and group dynamics:**

| Quantity | 0.25 m | 0.5 m | 1.0 m | 1.5 m | 2.0 m |
|---|---|---|---|---|---|
| Edges inside simultaneous groups of ≥ 3 players | 809 (3.7 %) | **4,931 (10.7 %)** | 25,727 (24.5 %) | 58,708 (36 %) | 100,804 (47 %) |
| Edges sequential within 1 s with another partner (not simultaneous) | 2,105 | 5,856 | 14,350 | 21,345 | 26,631 |
| Isolated edges (no other partner within 1 s) | 18,968 (87 %) | 35,202 (77 %) | 64,765 (62 %) | 81,610 (50 %) | 89,116 (41 %) |
| Groups with a merge / with a split | 4 / 0 | 61 / 69 | 481 / 483 | 952 / 944 | 1,410 / 1,391 |
| Joins / leaves (simultaneous groups) | 393 / 386 | 2,602 / 2,597 | 13,098 / 13,088 | 27,591 / 27,573 | 43,125 / 43,118 |

The "edges inside simultaneous groups" counts equal the counts of edges that overlap in time with an edge to a different partner,
by construction.

**Findings:**
- **[V]** At the historical 0.5 m radius, 23 % of edges are not isolated pairwise events: 10.7 % are simultaneous with another
  partner and 12.7 % are sequential within 1 s.
- **[V]** At 1 m, 38 % are not isolated. At 2 m most edges belong to multi-player structure.
- **[HYP]** Groups of 10–21 players at 1.5–2 m are transitive chains in set pieces or congested play, not 21-way interactions.
  `max_simultaneous_players` is the better size measure there.

## 5. Durations, distances and kinematics [V]

| Quantity | 0.25 m | 0.5 m | 1.0 m | 2.0 m |
|---|---|---|---|---|
| Edge duration p50 / p90 (s) | 0.11 / 0.49 | 0.18 / 1.04 | 0.35 / 2.14 | 0.64 / 4.37 |
| d_min p50 (m) | 0.16 | 0.36 | 0.76 | 1.50 |
| Relative speed at t\*, p50 (m/s) | 1.21 | 1.18 | 1.22 | 1.21 |
| Relative heading p50 (°) | 41 | 40 | 36 | 35 |
| Boundary uncertain (d_min within 5 cm of R) | 7,237 (33 %) | 12,185 (26 %) | 21,419 (20 %) | 35,116 (16 %) |

**Short contacts and sensitivity to coordinate error.**
- At 0.5 m, half of the edges last under 0.2 s.
- A quarter of them would appear or vanish under a 5 cm coordinate change.
- **[HYP]** Any population defined at a small radius therefore inherits provider-coordinate noise in its *membership*, not only in
  its features.

## 6. VISIBLE / ESTIMATED composition [V]

| Quantity | 0.25 m | 0.5 m | 1.0 m | 1.5 m | 2.0 m |
|---|---|---|---|---|---|
| Both VISIBLE / mixed / both ESTIMATED | 1,631 / 4,953 / 15,298 | 4,468 / 10,861 / 30,660 | 20,748 / 25,449 / 58,645 | 35,678 / 41,978 / 84,007 | 48,158 / 59,909 / 108,484 |
| Share of edges where both players are VISIBLE | **7.5 %** | 9.7 % | 19.8 % | 22.1 % | 22.2 % |
| Mean ESTIMATED fraction | 0.81 | 0.78 | 0.68 | 0.64 | 0.64 |

**Finding [V].**
- The closer the contact, the more it is provider-estimated.
- Both players are VISIBLE in only 7.5 % of sub-25 cm contacts, against 22 % at 2 m. The match-wide ESTIMATED share is 62 %.

**[HYP]** Two explanations are possible: either the provider marks players in close contact as estimated (occluded in its own
camera system), or its estimation pulls near players together. **Both mean that close-contact geometry in this dataset is largely
provider-modelled.** This is the strongest data-validity concern for any small-radius study.

## 7. Clone-like and suspicious patterns [V]

| Quantity | 0.25 m | 0.5 m | 1.0 m | 2.0 m |
|---|---|---|---|---|
| Edges with long ESTIMATED co-motion (≥ 1 s within 0.25 m, both ≥ 95 % ESTIMATED) | 855 | 1,079 | 1,036 | 879 |
| Pair-periods with ≥ 10 edges | 355 | 1,213 | 3,456 | 6,819 |
| …of which ≥ 95 % ESTIMATED | 141 | 302 | 418 | 436 |
| …ESTIMATED and same team | 94 | 211 | 338 | 356 |

**Cloned positions removed upstream.**
- Bit-identical (cloned) positions affect **2,220,332 samples (1.7 % of available samples)**.
- They removed 1.10 M segments, with no interpolation across them.
- Near-identical clones, which are not bit-identical, remain and are flagged as ESTIMATED co-motion.

## 8. Ambiguous versus clearly separable scenarios (constant-velocity indicator) [V]

**Tiers by radius:**

| cv_tier | 0.25 m | 0.5 m | 1.0 m | 1.5 m | 2.0 m |
|---|---|---|---|---|---|
| Confusable (CV would exchange) | 6,247 (29 %) | 10,627 (23 %) | 18,388 (18 %) | 22,456 (14 %) | 24,369 (11 %) |
| Marginal (0–0.5 m margin) | 9,270 (42 %) | 13,929 (30 %) | 19,545 (19 %) | 21,748 (13 %) | 22,197 (10 %) |
| Separable (≥ 0.5 m) | 4,411 (20 %) | 17,240 (37 %) | 56,959 (54 %) | 100,365 (62 %) | 144,464 (67 %) |
| Unavailable (< 1.2 s chain coverage) | 1,954 | 4,193 | 9,950 | 17,094 | 25,521 |

**At 0.5 m by status:**

| Status | Edges | Confusable |
|---|---|---|
| Both VISIBLE | 4,468 | **1,503 (34 %)** |
| Mixed | 10,861 | 4,308 (40 %) |
| Both ESTIMATED | 30,660 | 4,816 (16 %) |

**Findings:**
- **[V]** Ambiguity falls with radius, but **it does not vanish at 2 m**: 24,369 edges at 2 m are still CV-confusable.
  - That is *more* confusable scenarios than the whole 0.5 m catalogue's 10,627.
  - So a 0.5 m population misses most of the scenarios where the simplest physical model fails.
- **[V]** Both-ESTIMATED contacts are much *less* often CV-confusable (16 %) than VISIBLE (34 %) or mixed (40 %) ones at 0.5 m.
- **[HYP]** This matches the earlier model finding (lower error on ESTIMATED): provider-estimated motion is smoother and more
  predictable. **Model performance on ESTIMATED-dominated close contacts may overstate performance on real, visible
  interactions.**

## 9. Independent groups versus overlapping observations (deduplication) [V]

| Unit | 0.25 m | 0.5 m | 1.0 m | 1.5 m | 2.0 m |
|---|---|---|---|---|---|
| Window detections (5 s windows, 1 s stride) | 114,576 | 250,796 | 616,868 | 1,021,352 | **1,456,983** |
| Unique pair-windows | 78,145 | 166,271 | 418,884 | 707,347 | 1,023,339 |
| **Edges (catalogue events)** | 21,882 | **45,989** | 104,842 | 161,663 | 216,551 |
| Edge chains | 14,769 | 29,861 | 68,345 | 104,885 | 138,950 |
| Simultaneous groups | 21,308 | 42,257 | 83,428 | 110,953 | 128,082 |
| Connected groups (closest to "independent situations") | 13,423 | 24,667 | 48,053 | 62,260 | 69,194 |

**Inflation from sliding windows.**
- A sliding-window detector inflates counts by ×5.5 (0.5 m) to ×6.7 (2 m) over the true edges.
- This explains the "about a million" expectation: 1.46 M window detections at 2 m correspond to 217 k edges and 69 k connected
  situations.

## 10. Data-quality concerns [V]

| Concern | Count | Comment |
|---|---|---|
| Bit-identical duplicate (cloned) positions | 2,220,332 samples; 1,100,953 segments excluded | Largest exclusion; mostly ESTIMATED tracks [HYP] |
| Unattributed / ambiguous samples excluded | 683,976 segments | — |
| Implausible jumps (chain breaks) | 290,783 | — |
| Over-speed sub-metre steps | 169,555 samples | Flag 13.7 % of 0.5 m edges (6,316); provider update jitter |
| Smoothed acceleration, per-edge max p50 / p99 | 6.6 / 37 m/s² | Implausible tails; 1,234 edges (2.7 %) > 25 m/s² at 0.5 m |
| Speed > 12 m/s near the edge | 544 edges at 0.5 m | — |
| Out-of-pitch samples (> 3 m outside) | 11,296 (0.009 %) | 10–28 edges; negligible |
| Non-finite coordinates / INVALID_COORDINATE | 0 / 0 | — |
| Timestamp problems | 0 | — |
| Possible label corruption | 5,406 edges at 0.5 m (11.8 %) | Implausible jump ± 1 s 4,440; chain boundary 2,198; duplicate position 1,055; frame shift 315; identity-risk flag 156 |
| Reliability reduced | 7,571 at 0.5 m (16.5 %) | Over-speed 6,316; accel 1,234; ESTIMATED co-motion 1,079; speed 544 |

**Status alone never reduced reliability or eligibility.** This is enforced in the code and tested.

## 11. Previously overlooked interaction types (now captured)

1. **Multi-player simultaneous groups** [V]. 1,199 groups of ≥ 3 players at 0.5 m (4,931 edges); 4,313 groups at 1 m. Earlier they
   were represented only as unrelated binary pairs.
2. **Sequential hand-offs** [V]. One player meets a different partner within 1 s: 5,856 edges at 0.5 m. These are relevant to
   staggered-gap N×M association.
3. **Group merges and splits** [V]. 61 / 69 groups at 0.5 m; about 1,400 at 2 m.
4. **Repeated weaving within a window** [V]. 23,306 pair-windows with several synchronous crossings. These were discarded by v1's
   first-intersection rule.
5. **Ambiguous contacts beyond 0.5 m** [V]. 18,388 CV-confusable edges at 1 m and 24,369 at 2 m, absent from the 0.5 m population.
6. **Provider clone-like co-motion** [V]. About 1 k edges; 302 repeated same-pair clusters at 0.5 m.

## 12. What this does and does not support

**Supports:**
- **[V]** a complete, deduplicated, reproducible count of proximity and crossing events at five radii, with the historical
  benchmark embedded exactly;
- **[V]** quantified multi-player structure, repetition and provider-status composition.

**Does not support:**
- **[HYP]** any claim about occlusion or visibility in video;
- **[HYP]** a choice of supervised radius. The catalogue is descriptive. The constant-velocity indicator is only one simple
  physical descriptor.

**Recommendations [REC], for decision only:**
1. If a broader supervised population is pursued, define it by an *ambiguity criterion* fixed in advance, not by sample count.
   The constant-velocity confusable/marginal tiers at ≤ 2 m are one candidate.
2. Report every result stratified by status composition and by isolated / sequential / group context.
3. Use connected groups, or matches, as the unit of statistical independence. Never use windows.
