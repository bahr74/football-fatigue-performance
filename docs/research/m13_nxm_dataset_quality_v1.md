# M13 report 1: N×M dataset defects, corrections and quality (v1)

Status: **EXPLORATORY.**
- The data are provider-derived; no scenario is human-reviewed.
- The dataset is not frozen and is not the project's dataset v2.
- It is not video, and nothing in it is a verified occlusion.

Date 2026-11-21.

**Code**
- `src/football_fatigue/m11_explorer/nxm_dataset_v2.py` (new). v1's `nxm_dataset.py` is unchanged and only reused for helpers.
- Run with `scripts\session_a_python.cmd -m football_fatigue.m11_explorer nxm-dataset-v2` (CPU, 937 s).

**Output:** `data/processed/m11-nxm-exploratory-v2/` (write-once), containing `scenarios.jsonl.gz`, `index.parquet`,
`encounters.parquet`, `build.json` and `manifest.json`.

**Audit:** `data/processed/m11-nxm-association-v2/{v1,v2}/{quality,coverage}.json` (`nxm_assoc_v2.audit` / `coverage`).

**Tests:** `tests/test_m11_nxm_v2.py`.

**v1 is preserved.** `m11-nxm-exploratory-v1` and `m11-nxm-solver-comparison-v1` are byte-identical to their manifests (tested).

## 1. Root causes of the M12 defects (verified facts)

| # | Defect | Root cause (verified in code and data) | Consequence in v1 |
|---|---|---|---|
| D1 | `repeated_interaction` = 1.0 on every scenario | v1 computed `max(order_in_chain) > 0`. The catalogue sets `order_in_chain = rank(start, method="first")` inside an edge chain (`interaction_catalogue/core.py`), so it is ≥ 1 by construction. | The flag carried no information |
| D2 | Duplicate events | One continuous encounter is fragmented into several link-0 catalogue groups. In match 3812 one edge chain spans up to 23 groups, and 1,476 of 3,339 candidate groups are fragments of another encounter. v1 used one event per group. | 343 of 2,213 v1 events (15.5 %) share ≥ 2 members with another selected event inside an overlapping window (202 pairs; at most 2,024 independent encounters). Concentrated at N = 3 (10.5 %) and N = 4 (12.4 %) |
| D3 | Timing shortcut | v1 drew one gap length per player plus a per-player centre offset. The true pair's `t_first − t_end` equals that player's gap, while wrong pairs mix two offsets and have a wider spread. | A label-free timing oracle was exact in 0.45 of async scenarios. Pair-level AUC of \|dt − median\| for true vs wrong is 0.58 on a v1 subset |
| D4 | N > M missing for N ≥ 4 | `unequal` removed exactly one outgoing track *and* added a distractor whenever one was within 6 m, which is almost always. No variant removed two. | 4×3: 7 scenarios; 5×4 and 5×3: none; no scenario with two nulls |
| D5 | Count shortcut | Nulls occurred only in `unequal`, and its shape was almost always N×N. | P(null \| M < N) = 1.0 and P(null \| M > N) = 0.0, so the shape almost determined the null structure |

**Hypothesis, not verified:** provider ESTIMATED interpolation makes motion more predictable. That would explain why physics does
better with ESTIMATED contexts (M12 report 6). It is not a defect of the generator and remains a data property (section 4).

## 2. Corrections implemented in `m11-nxm-exploratory-v2`

1. **Canonical encounters (D2).**
   - Link-0 groups at R = 1.0 m are united when they share an edge chain. The chain is the catalogue's same-pair contact with
     gaps under 1.0 s.
   - Candidate events of one period are also united when their scenario windows (5.8 s) overlap and they share ≥ 2 members.
   - One representative is kept per encounter: the largest eligible N, ties broken by a seeded hash (never by a model or a label).
   - `n_catalogue_groups` records how many catalogue rows were merged (median 2, p90 8, p99 19).
   - Across the 32 matches: 83,428 candidates form 48,092 encounters, so 35,336 rows were merged; 38,472 encounters are
     eligible (continuous provider chain over the window).
2. **Repeated encounters (D1).**
   - `repeated_encounter` is true when a member pair had a *distinct* earlier edge chain that is not part of this encounter and
     ended 1 to 60 s before the encounter started.
   - Fragments of the same contact never count. That includes chains ending less than the 1.0 s merge gap before the encounter
     (tested).
   - Also recorded: `n_prior_encounters` and `prior_gap_s`.
   - Result: true for **52.5 %** of encounters (no longer constant). The prior gap is p10 3.3 s, median 12.0 s, p90 42.8 s.
3. **Decoupled timing (D3).** Each player's disappearance time `tc − U(0.3, 0.9)` and reappearance time `tc + U(0.3, 0.9)` are
   drawn independently. New outgoing players use the same reappearance distribution, so true and wrong pairs have the same
   time-difference distribution (tested; measured below).
4. **Null and new-player coverage (D4, D5).** Seven variants per encounter:

   | Variant | Construction | Shape |
   |---|---|---|
   | `complete` | Synchronous 0.8 s, no dropout | N×N |
   | `async` | Independent times, 20 % context dropout | N×N |
   | `null1` | `async`, 1 outgoing removed | N×(N−1) |
   | `null2` | `async`, 2 removed (N ≥ 3) | N×(N−2) |
   | `new1` | `async` + 1 new outgoing | N×(N+1) |
   | `new2` | `async` + 2 new outgoing | N×(N+2) |
   | `null_new` | k ∈ {1, 2} removed and d ∈ {1, 2} new, drawn independently | Varies |

   A variant that cannot be built is skipped, never filled with invented data.
5. **Opaque outgoing ids.** Outgoing ids are seeded hashes (`out-3fa9c2d1`), not numbers, so a missing track cannot be seen from
   an id sequence.
6. **Unchanged guarantees.**
   - **One-to-one truth:** truth is a map from incoming id to outgoing id or null. New outgoing tracks are never a truth target
     (tested).
   - **Continuous provider chains:** every member has one chain over the window, and distractors are checked the same way.
   - **Observed-only contexts:** contexts use only samples outside each player's hidden interval, without extrapolation.
   - **Development data only:** the 32 train/CV matches, guarded by `C.checked`.
   - **Provenance:** `provenance = "provider-derived"` and `human_reviewed = false` on every scenario.
7. **Model view.** The benchmark gives systems only `view(d)`: tracks with {id, t, x, y}. Truth, meta (slots, player ids,
   provider status, hidden intervals, variant) and incoming labels are removed (tested: features are invariant to them).

## 3. Shape and coverage

**Totals.** 20,489 scenarios from **3,360 canonical encounters** in 32 matches, 73 to 144 encounters per match (median 106).
Variants per encounter: median 7, p10 4.

**Encounters by N.**

| N | v2 encounters | v1 events |
|---|---|---|
| 2 | 800 (capped at 25 per match) | 611 |
| 3 | 1,982 | 939 |
| 4 | 334 | 404, with duplicates |
| 5 | 244 | 259 |

v1 capped N = 3 at 40 events per match; v2 uses every eligible N ≥ 3 encounter.

**Scenarios by shape (credible = ≥ 30 distinct encounters from ≥ 10 matches; every cell below qualifies).**

| N \ M | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| 2 | 800 | 1,935 | 704 | 344 | | | |
| 3 | 1,982 | 2,498 | 4,790 | 2,052 | 1,381 | | |
| 4 | | 334 | **413** | 847 | 398 | 308 | |
| 5 | | | **244** | **299** | 620 | 300 | 240 |

In v1, 4×3 had 7 scenarios and 5×3 and 5×4 had none.

**Other dimensions (scenarios / encounters).**

| Dimension | Breakdown |
|---|---|
| Nulls | 0: 11,779 / 3,360; 1: 5,014 / 3,360; 2: 3,696 / 2,560 |
| New outgoing | 0: 12,640 / 3,360; 1: 4,422 / 2,804; 2: 3,427 / 2,284 |
| Shape | M < N: 6,570; M = N: 8,192; M > N: 5,727 |
| Mean gap | 0.7 to 1.0 s: 4,749; 1.0 to 1.3 s: 11,521; > 1.3 s: 4,219. No ≤ 0.7 s band: the minimum per-player gap is 0.6 s |
| Visibility (context ESTIMATED share) | All visible: 1,135 scenarios / 263 encounters; ≤ 50 %: 5,458 / 1,090; > 50 %: 13,896 / 2,349 |
| Match balance | Every cell draws on all 32 matches |

**Too rare for credible conclusions:**
- **N ≥ 4 with fully visible contexts: 6 encounters from 5 matches** (5 of them with a null). No statement about large groups
  under visible tracking is supported.
- Every other cross-tabulation (N band × gap, N band × difficulty, visibility × difficulty) has ≥ 30 encounters from ≥ 10
  matches.
- Variants of an encounter are augmentations. Effective sample sizes are the encounter counts, never the scenario counts.

## 4. Shortcut and label-validity audit (v2; measured)

| Check | Result | Reading |
|---|---|---|
| Duplicate events (≥ 2 shared members, overlapping windows) | **0** (v1: 202 pairs) | Fixed |
| Correlated variants | Median 7 scenarios per encounter | All CIs are match-level bootstraps; variants are never counted as independent |
| Timing: \|dt − median\| AUC, true vs wrong pairs | **0.485** (dt median 1.254 true vs 1.280 wrong; v1 subset 0.58) | No timing signal |
| Outgoing track length (samples) AUC | 0.526 | Small residual. Wrong pairs include new players' tracks; dropout is identical. Measured again through the shortcut-only model in report 2 |
| Outgoing id order | Ids are opaque hashes, sorted lexically | No order information (SC_order baseline in report 2) |
| Count: P(null \| shape) | M < N: 1.0 (structural); **M = N: 0.18; M > N: 0.12** (v1: 0.30 / 0.0) | The null count is now only partly implied by the shape. M < N still forces at least N − M nulls; that is a property of the task, not a leak |
| Match overlap across folds | 0 matches in more than one fold | Match-disjoint |
| Player overlap across folds | 50 to 89 % of a test fold's players appear in other folds | Expected: the same teams play several matches. Models see only anonymous motion, so player memorisation can only act through individual movement style. Not tested further |
| Provider label uncertainty | 3.3 % of scenarios touch a label-risk edge; 10.6 % ESTIMATED co-motion | Labels are not verified (report 4) |
| ESTIMATED dominance | Mean context ESTIMATED 0.67; 88 % have a mostly-ESTIMATED track; 5.5 % all visible | **Unchanged data property**; headline numbers are stratified by visibility in report 2 |
| Implausible motion | 6.2 % have a sample-to-sample speed > 12 m/s | Provider jumps; review candidates |

**Verified facts vs hypotheses.**
- *Verified:* D1 to D5 and their corrections, the measured audit numbers above, and that v1 is unchanged.
- *Hypotheses:*
  - that provider ESTIMATED texture inflates motion predictability (consistent with M12 reports 2 and 6, not proven);
  - that the 0.526 length AUC is benign (to be checked by the shortcut-only model).

## 5. What v2 still is not

- **Truth:** provider identity, not human-verified. Hidden intervals are synthetic cuts.
- **Selection:** continuous provider chains (the same selection as Stage 3) exclude exactly the cases where the provider itself
  lost the identity.
- **Augmentations:** variants are augmentations of 3,360 real encounters, not 20,489 independent events.
- **Not frozen:** it must not be used for video claims.
