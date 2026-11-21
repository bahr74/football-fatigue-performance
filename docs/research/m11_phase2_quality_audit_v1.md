# Phase 2, Task A: Interaction Catalogue v1, scientific validation (development matches only)

**Date:** 2026-11-20

**Inputs:** `data/processed/m11-interaction-catalogue-v1/` (read only; run manifest `run_manifest_2026-11-19T23-34-58.json`, sha256
`a19f0f81…`). The catalogue was not modified.

**Outputs:**
- `data/processed/m11-phase2-quality-audit-v1/`:
  - `audit_results.json`
  - `tables.md`
  - `per_match_radius.parquet`
  - `clone_like_pair_periods.parquet`
  - `manifest.json`, with code and output hashes
- `data/processed/m11-phase2-review-sample-v1/`:
  - `review_manifest.parquet` / `.csv`
  - 100 PNG figures
  - `review_summary.json`
  - `manifest.json`

**Code:**
- `src/football_fatigue/m11_phase2/audit.py`
- `review.py`
- `common.py` (guard, write-once outputs, provenance)

**Commands:**
```powershell
.\.venv\Scripts\python.exe -m football_fatigue.m11_phase2 audit             # 48 s
.\.venv\Scripts\python.exe -m football_fatigue.m11_phase2 review --figures  # ~8 min (reads RAW Step 1 caches of 31 matches)
```

**Data boundary:** only the 32 `train_cv` matches. `common.checked()` runs the Phase 1 guard before any file is opened. A test
monkeypatches the parquet reader to prove that nothing is read for a protected id.

**Labels:**

| Label | Meaning |
|---|---|
| [V] | Computed exactly in this run |
| [EST] | Estimate under a stated assumption |
| [HYP] | Interpretation |
| [REC] | Recommendation |

No human review was performed. The review manifest has empty reviewer fields.

---

## 1. Verification of the Phase 1 figures before reuse [V]

| Figure in the Phase 1 report | Recomputed | Status |
|---|---|---|
| Edges 21,882 / 45,989 / 104,842 / 161,663 / 216,551 | Identical | ✔ |
| Edge chains 14,769 / 29,861 / 68,345 / 104,885 / 138,950 | Identical | ✔ |
| Crossings 581,786; synchronous 33,887 | Identical | ✔ |
| Simultaneous / connected groups (all radii) | Re-derived with `core.link_components` (δ = 0 and 1 s): identical to the stored tables | ✔ |
| Edge chains at a 1 s merge gap | Re-derived independently by sort and cumulative max: identical | ✔ |

## 2. Counting-inflation invariants [V]

All pass at every radius (`audit_results.json → invariants.all_passed = true`).

| Invariant | Result |
|---|---|
| `window_stats.edges` sums to the edge count; edge ids unique | ✔ |
| Σ per-edge `n_windows` = window detections | ✔ |
| Every edge belongs to exactly one group per δ; group `n_edges` agree | ✔ (δ = 0 and δ = 1) |
| Simultaneous (δ = 0) groups refine connected (δ = 1) groups | ✔ |
| Crossing ids unique | ✔ |

**Window inflation** (window detections per edge): 5.24 / 5.45 / 5.88 / 6.32 / 6.73 at 0.25 … 2 m. Sliding windows never enter any
event count.

**Transitive merging inflates "group size" [V].** This is the important new finding of this section.

| R | Connected (δ = 1 s) groups with ≥ 3 players | …never ≥ 3 players **at the same instant** | Simultaneous (δ = 0) 3+ groups | …never 3 at once |
|---|---|---|---|---|
| 0.25 | 869 | 659 (76 %) | 235 | 0 |
| **0.5** | **2,408** | **1,438 (60 %)** | 1,199 | 0 |
| 1.0 | 6,138 | 3,091 (50 %) | 4,313 | 0 |
| 2.0 | 13,486 | 4,699 (35 %) | 12,335 | 0 |

- At 0.5 m the connected group's player count exceeds its largest instantaneous component in 1,825 of 2,408 groups: median ratio
  1.5, p90 2.0, max 4.0.
- 90 of the 1,199 simultaneous 3+ groups also contain more players than ever touch at once. The δ = 0 linkage chains edges that
  only touch end-to-start.

**Consequences:**
- "Connected 3+ groups" are mostly **sequential hand-offs**, not true multi-player contacts.
- **[REC]** Any multi-player claim should use `max_simultaneous_players`, the instantaneous component size. At 0.5 m that gives
  1,199 situations with 3+ players simultaneously within 0.5 m of a linked partner (1,060 / 116 / 23 at 3 / 4 / 5+).

## 3. Structure and distributions (A1) [V]

### 3.1 Per match

| Quantity at 0.5 m | Min | Median | Max | CV across matches |
|---|---|---|---|---|
| Edges | 1,003 | 1,378 | 2,273 | 0.20 |
| Edge chains | 663 | 944 | 1,277 | 0.18 |
| Connected groups | 544 | 784 | 995 | 0.17 |
| Both-VISIBLE share | 5.6 % | 10.4 % | 15.3 % | 0.21 |
| CV-confusable share | 16.2 % | 23.4 % | 29.0 % | 0.11 |
| Label-risk share | 5.8 % | 11.2 % | 18.3 % | 0.28 |

- **No match is a robust outlier at 0.5 m** (|robust z| > 3 on any quantity).
- At 1–2 m, match **3851** has excess edge counts (z = 3.2–3.8).
- Matches 3819, 3851 and 10507 have a low CV-confusable share at 1.5–2 m (z ≈ −3.1 to −3.8).

These are candidates for inspection, not exclusion. Crossings per match: 13,491–22,524 (CV 0.13).

### 3.2 Players and pairs: dependence structure

| | 0.5 m | 2.0 m |
|---|---|---|
| Player-matches involved | 977 | 984 |
| Edges per player-match p10 / p50 / p90 / p99 | 15 / 89 / 178 / 259 | 76 / 453 / 787 / 1,048 |
| Gini over players | 0.37 | 0.34 |
| Top 10 % of players: share of edge ends | 22.8 % | 20.6 % |
| Pairs (match level); edges per pair p50 / p90 / p99 | 7,404; 4 / 14 / 38 | 10,593; 11 / 49 / 129 |
| Gini over pairs; top 10 % pairs' share of edges | 0.54; 39.3 % | 0.57; 41.1 % |
| Same-team share | 41.1 % | 34.9 % |

**Pair-level concentration is strong:** 10 % of pairs hold about 40 % of the edges. Any estimate that treats edges as independent
overstates the evidence. **[REC]** Cluster at least by pair and by match.

### 3.3 Repetition: time between consecutive edges of the same pair

| R | Gaps | ≤ 1 s | 1–5 s | 5–30 s | 30–120 s | > 120 s | Median |
|---|---|---|---|---|---|---|---|
| 0.5 m | 36,303 | 16,128 (44 %) | 2,745 | 3,003 | 3,324 | 11,103 | 3.3 s |
| 2.0 m | 201,311 | 77,601 (39 %) | 25,059 | 30,493 | 28,725 | 39,433 | 4.4 s |

**The distribution is bimodal [V/HYP]:**
- Re-contacts within 1 s are mostly the same physical contact fragmented by jitter at the radius boundary. They are what the 1 s
  chain merge absorbs.
- Long gaps (> 2 min) are genuinely separate meetings.
- The 1–30 s band (5,748 gaps at 0.5 m) is where "repeated interaction" in the tactical sense lives.

### 3.4 Durations, distances, crossings

**At 0.5 m:**
- Duration p10 / p50 / p90 / p99: 0.04 / 0.18 / 1.04 / 4.8 s.
- 14,283 edges (31 %) last < 0.1 s, and 8,979 (20 %) are shorter than one 15 Hz step.
- d_min / R has median 0.71 and p90 0.97, so **the distribution is piled against the radius**. This is consistent with many grazing
  contacts [HYP].

**Crossings:**
- 581,340 proper, 375 at an endpoint, 71 collinear.
- Synchronous crossings have median separation 0.36 m (p90 1.22 m); asynchronous ones 3.59 m (p90 11.3 m).
- 37,605 *asynchronous* crossings still have the two players within 0.5 m when the first reaches the point. These are slow
  co-located movement, not synchronous passes.

### 3.5 Context of each edge

| R | Isolated | Sequential (another partner within 1 s) | Simultaneous (overlapping another partner) |
|---|---|---|---|
| 0.5 m | 35,202 (77 %) | 5,856 (13 %) | 4,931 (11 %) |
| 2.0 m | 89,116 (41 %) | 26,631 (12 %) | 100,804 (47 %) |

Mixed-team 3+ groups (δ = 0): 799 of 1,199 at 0.5 m.

### 3.6 Rare and extreme events

Listed in `audit_results.json → extremes`:
- the longest edges;
- the most negative CV margins;
- the highest accelerations;
- the largest simultaneous groups at 1 m;
- the pair-windows with the most crossings (up to 252, jitter about one spot).

## 4. Provider-artifact evidence in four separate families (A2) [V]

Percentages of edges at **0.5 m**. Every radius is in `tables.md`.

| Family | Indicator | Both VISIBLE (4,468) | Mixed (10,861) | Both ESTIMATED (30,660) |
|---|---|---|---|---|
| **1. Measurement quality** | Over-speed step in ±1 s | 21.0 | 19.9 | **10.5** |
| | Smoothed accel > 25 m/s² | 5.2 | 4.4 | **1.7** |
| | Max accel, median (m/s²) | 8.8 | 8.9 | **5.1** |
| **2. Identity-label reliability** | Any label-risk flag | 16.9 | 15.4 | 9.7 |
| | Implausible jump in ±1 s | 16.9 | 15.3 | 6.6 |
| | Chain truncated at the edge | 4.1 | 3.8 | 5.2 |
| **3. Event-boundary uncertainty** | d_min within 5 cm of R | 25.7 | **12.7** | **31.5** |
| | Duration < 0.1 s | 29.1 | 23.0 | 34.2 |
| **4. Interpolation artifacts** | Long ESTIMATED co-motion | 0.0 | 0.2 | 3.5 |
| | In a clone-like pair-period | 0.2 | 1.5 | **16.2** |
| | d_min < 5 cm | 5.3 | 10.5 | 6.4 |
| (difficulty, for reference) | CV-confusable | 33.6 | 39.7 | 15.7 |

**Findings [V]:**
1. **ESTIMATED close contacts are kinematically smoother.**
   - The median max acceleration is 5.1 vs 8.8 m/s² for VISIBLE.
   - The jump and over-speed flag rates are about half.
   - They are less often CV-confusable.

   This is exactly the signature of model-based interpolation. It supports the earlier texture-probe concern [DOC]: a model can
   learn that "smooth" means "predictable".
2. **The VISIBLE subset carries more identity-label risk.** It has about 2.5× the jump rate. Measured provider positions are noisier,
   and jumps cluster near contact.
3. **Boundary uncertainty is highest for ESTIMATED edges** (31.5 %): membership at a fixed radius is least stable exactly there.
4. **Clone-like structure.**
   - **302 pair-periods** (≥ 10 edges at 0.5 m, mean ESTIMATED ≥ 0.95) hold **5,139 edges (11.2 % of 0.5 m edges)**. 211 of the
     pairs are same-team, and they appear in 32 of 32 matches.
   - The edge-level long-co-motion flag (1,079 edges) overlaps with them in only 134 edges.

   **The two flags capture different artifacts:**
   - long co-located drift (co-motion);
   - many short repeated touches of a mostly estimated pair (clone-pair).

   Diagnostic figure P2R-027 (match 3833) shows the second pattern: two near-stationary, fully ESTIMATED tracks jittering by about
   0.3 m around each other for 3.6 s.
5. **Identity breaks:**
   - The shorter of the two provider chains around a 0.5 m edge has median 123 s and p10 20 s.
   - 2,334 edges (5.1 %) involve a chain shorter than 5 s, so the identity label is short-lived on one side.
   - Label-risk reasons at 0.5 m: implausible jump 4,440; chain boundary 2,198; duplicate position 1,055; frame shift 315; identity
     flag 156.
   - 3,201 edges carry both a label risk and reduced reliability. **25,317 (55 %) carry no flag at all.**

## 5. Sensitivity (A3) [V unless marked]

### 5.1 Radius
Unchanged from Phase 1 (§1). The connected situations, rather than the edges, are the slowly growing quantity.

### 5.2 Group-link tolerance δ

**Groups at 0.5 m (number of groups / of which 3+ players / largest):**

| δ (s) | 0 | 0.25 | 0.5 | 1 | 2 | 3 |
|---|---|---|---|---|---|---|
| Groups / 3+ / largest | 42,257 / 1,199 / 8 | 29,981 / 1,704 / 10 | 26,596 / 2,035 / 13 | 24,667 / 2,408 / 14 | 22,192 / 2,769 / 14 | 20,512 / 2,948 / 16 |

- **The number of "multi-player" groups depends strongly on δ:** ×2.5 from δ = 0 to 3 s at 0.5 m.
- The share that is truly simultaneous falls accordingly (§2).
- **[REC]** Never report a multi-player count without δ and the simultaneous-size definition.

### 5.3 Chain merge gap (repeated contact → one chain)

| R | 0 s | 0.5 s | 1 s | 2 s | 5 s |
|---|---|---|---|---|---|
| 0.5 | 45,989 | 30,795 | **29,861** | 28,720 | 27,116 |
| 2.0 | 216,551 | 145,567 | 138,950 | 129,120 | 113,891 |

- Between 0.5 s and 5 s the count changes by only 12 % (0.5 m).
- Most of the merging happens below 0.5 s (fragmentation by jitter).
- **The historical 1 s choice is on a plateau. It is not a sensitive parameter.**

### 5.4 Crossing synchrony tolerance τ

| τ (s) | 0.1 | 0.2 | **0.3** | 0.5 | 1 | 2 |
|---|---|---|---|---|---|---|
| Crossings | 10,304 | 21,561 | **33,887** | 58,736 | 120,556 | 239,079 |
| Share | 1.8 % | 3.7 % | 5.8 % | 10.1 % | 20.7 % | 41.1 % |
| Median separation at crossing (m) | 0.13 | 0.24 | 0.36 | 0.58 | 1.00 | 1.67 |

- The synchronous-crossing count is **roughly linear in τ**: there is no natural plateau.
- "Synchronous crossing" is therefore a continuum.
- **[REC]** Use `abs_dt_s` and `d_sync_m` as continuous descriptors rather than a binary class.

### 5.5 Boundary jitter (uniform separation bias ±ε)

| R | ε = 2 cm lost / gained [EST] | ε = 5 cm | ε = 10 cm |
|---|---|---|---|
| 0.25 | 16.0 % / 6.0 % | 33.1 % / 15.4 % | 53.1 % / 32.6 % |
| **0.5** | **13.2 % / 3.0 %** | **26.5 % / 8.0 %** | **40.9 % / 16.7 %** |
| 1.0 | 10.1 % / 1.8 % | 20.4 % / 4.5 % | 31.4 % / 9.1 % |
| 1.5 | 8.9 % / 1.2 % | 17.8 % / 3.0 % | 26.8 % / 6.2 % |

**How these were computed:**
- "Lost" counts edges whose minimum would exceed R under +ε. It is exact for a uniform bias, ignoring edge splitting.
- "Gained" counts next-radius edges with minimum in (R, R + ε]. It is a lower bound [EST].

**Reading:**
- A 5 cm systematic coordinate error, smaller than typical provider jitter, changes the 0.5 m membership by about a quarter.
- **The small-radius populations are defined by noise-level geometry.**
- Wider radii are proportionally more stable.

### 5.6 Clone-like exclusion (sensitivity only; the catalogue is unchanged)

| R | Variant | Edges | Edge chains | CV-confusable | Both VISIBLE |
|---|---|---|---|---|---|
| 0.5 | All | 45,989 | 29,861 | 23.1 % | 9.7 % |
| 0.5 | Excluding co-motion edges | 44,910 | 29,469 | 23.0 % | 9.9 % |
| 0.5 | Excluding clone-like pair-periods | 40,850 | 28,048 | 24.4 % | 10.9 % |
| 0.5 | Excluding both | 39,905 | 27,696 | 24.4 % | 11.2 % |
| 2.0 | Excluding both | 205,922 | 133,450 | 11.4 % | 22.9 % |

- Excluding both flags removes 13 % of 0.5 m edges but only 7 % of edge chains: clone-like pairs are many short touches.
- Their effect on aggregate shares is small (≤ 1.5 points).
- **Their effect on model evaluation is larger.** In Task B, physics baselines are *more* accurate on clone-pair samples: forward
  Kalman 0.707 vs 0.664.

## 6. Stratified review sample (A4) [V; not human-reviewed]

**Sampler:**
- Deterministic: sha256(`m11-phase2-review-v1` | stratum | edge id) order.
- At most one event per match per stratum.
- An event is never chosen twice. `all_strata` lists every stratum an event satisfies, and 78 events satisfy more than one.

**Composition:**
- 100 events over 31 matches.
- 92 at 0.5 m and 8 at 2 m.
- Status: 16 both VISIBLE, 21 mixed, 63 both ESTIMATED.

**Quotas:**

| Stratum | n |
|---|---|
| clean_visible | 10 |
| clone_like | 10 |
| mixed_status | 8 |
| estimated_dominated | 8 |
| repeated_sync_crossings | 8 |
| group_3_simultaneous | 8 |
| group_4plus_simultaneous | 8 |
| high_ambiguity_cv | 8 |
| broad_radius_ambiguous (2 m, d_min > 1 m, CV-confusable) | 8 |
| short_event | 6 |
| long_event | 6 |
| low_ambiguity_cv | 6 |
| boundary_sensitive | 6 |

**Figures** (`figures/P2R-###.png`):
- Raw positions of both players for t\* ± 3 s, with ESTIMATED samples hollow.
- The separation curve with the edge interval and R.

They are **diagnostics for a reviewer, not evidence of visual occlusion**.

**Reviewer fields** (empty):
- `positions_plausible`
- `identity_continuity_plausible`
- `players_actually_close`
- `provider_artifact_suspected`
- `visual_occlusion_checkable`
- `notes`

**Observation, not a validation.** The "clean_visible" stratum uses status over [start − 0.1 s, end + 0.1 s]. ESTIMATED samples can
appear just outside that pad: figure P2R-001 shows a few around the crossing. A reviewer should judge the whole ±3 s view.

**[REC]** Add the 25 Task B "severe failure" samples to the review queue (`m11-phase2-ambiguity-benchmark-v1/severe_failures_top25.parquet`).
Their size, motion-margin errors of 9–12 m in a 0.8 s gap, suggests possible provider identity swaps that no current flag catches.

## 7. Trustworthiness verdict

**Trustworthy [V]:**
- Counting is correct.
- Events are deduplicated: no window or group inflation enters event counts.
- The catalogue reproduces every earlier dataset.
- All derived structures re-derive identically.

**Not trustworthy as-is:**

| Use | Problem | Remedy |
|---|---|---|
| (a) "Group size" from connected groups | Transitive inflation | Use simultaneous size |
| (b) Small-radius membership | A quarter flips under 5 cm bias | Treat R as a continuum or use an ambiguity criterion |
| (c) Close contacts as measurements of real motion | 67 % of 0.5 m edges are both ESTIMATED, and they are kinematically smoother | Stratify by status; do not pool |
| (d) Provider identity labels | Uncertain in 5–17 % of edges per status (label-risk flags), and some severe motion contradictions are unflagged | Human audit of a sample is required before any label-quality claim |
