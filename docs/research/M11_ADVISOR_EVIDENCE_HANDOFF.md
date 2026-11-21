# M11 — Advisor evidence handoff (input for the advisor discussion report)

Date 2026-11-11. Branch `feature/wadi-degla-demo`. This is a research and evidence document only. Nothing was implemented,
regenerated, retrained or modified for it. Every number below was either re-read from a stored output in this session
(read-only scripts in the session scratchpad, outside the repository) or is explicitly labelled as reported.

**Scope rule.** Only development matches were inspected. No validation or test match file was opened for this handoff.

**Labels used throughout:**

| Label | Meaning |
|---|---|
| **[V] VERIFIED** | Re-read or recomputed in this session from the stored artifact, source file or code named next to it |
| **[R] REPORTED** | Stated in a repository report; the underlying artifact was not re-checked in this session |
| **[I] INFERENCE** | Our interpretation; not a fact in the data or a statement by the advisor |
| **[U] UNRESOLVED** | Open; needs the advisor or further evidence |

**Two scopes must never be mixed:**

| Scope | Matches | Used by |
|---|---|---|
| **S10** | 10 development matches: 3812, 3814, 3815, 3816, 3829, 3833, 3836, 3847, 3851, 10506 | 2D catalogue (166,819), advisor cross-check, spatiotemporal tolerance (284), exact solver (566), proximity v1 (8,215), identity experiments a1–a3 |
| **S40** | All 40 development matches of the frozen split (`m11-encounters-v1/splits.json`) | Step 1 verification, proximity v2 (30,438), v3 contexts, v3 audit (1,766 ready) |

Path abbreviation: `DP/` = `data/processed/`.

---

## 1. The advisor's original requirements

### 1.1 Direct evidence available in the repository

| Source | What it contains | Status |
|---|---|---|
| `data/raw/Screenshot 2026-11-09 175517.png` (code) | `find_intersections(A, B)`: `LineString(A).intersection(LineString(B))`; keeps `Point` / `MultiPoint`, **returns `[]` for any other geometry** (e.g. collinear overlap); for each point `project` → cumulative segment lengths → `np.searchsorted(..., side='right') - 1`; returns `point`, `index_A`, `index_B` | **[V]** viewed in this session |
| `data/raw/Screenshot 2026-11-09 175508.png` (diagram) | Player A and Player B trajectories crossing at a black dot labelled "Occlusion Point"; labels "Segment A_Before", "Segment A_After", "Segment B_After" | **[V]** viewed in this session |
| The advisor's written messages | Quoted in `docs/research/m11_exact_spatiotemporal_extraction.md` §1 (lines 15–27) and `m11_occlusion_catalogue_realignment.md` §1 | **[R]** — the original messages are not stored as a file in the repo; quotes are as transcribed in those reports |

Two facts visible in the code screenshot matter. **[V]**

- The code is purely 2D (x, y). Time does not enter it.
- The code returns segment **indices** only. It does not compute a timestamp. Converting the index into a time
  (t = t0 + u·(t1 − t0)) is our completion of his recipe (`advisor.py:65 advisor_times`).

Quoted advisor phrases, as transcribed in `m11_exact_spatiotemporal_extraction.md:17-26` **[R]**:

- "for each 1second ... 2.5 seconds before ... 2.5 seconds after"
- "4 seconds trajectory"
- "22 players", "all possible couples", "121x60x90 = 653,400"
- "Keep only the trajectories that intersect"
- "The occ point here is one point at one time ... This time will be exactly the same for both players"
- "don't try to estimate it yourself ... curves intersection algorithm"
- "much robust than ... same point or in close points within x-meter distance"
- "cleaning from repeated ones (... trajectories 1 to 4 and also 2 to 5)"

Two arithmetic slips in his text **[V by arithmetic]**:

- ±2.5 s is a 5 s window, not "4 seconds".
- C(22, 2) = 231 pairs, not 121. So 231 × 5,400 = 1,247,400 pair-windows per 90 min, not 653,400.

### 1.2 Requirement-by-requirement status

| # | Requirement | Evidence type | Status | Where / note |
|---|---|---|---|---|
| R1 | Use the trajectories of all 22 players (all unique pairs) | Text [R] + code [V] (pairwise function) | **Implemented** | `detect.py:327`, `exact.py:185 _pairs_by_group`; S10 pair-window comparisons 14,059,734 (catalogue) / 14,321,863 (exact) [V `DP/m11-occlusion-catalogue-v1/cross-match/summary.json`, `DP/m11-exact-spatiotemporal-v1/cross-match/summary.json`]. Substitutes are extra slots, so pairs per window vary |
| R2 | 5 s sliding windows (±2.5 s), one per second | Text [R] | **Implemented** (5 s, not 4 s) | v1/catalogue: 61,873 windows (centres from period start + 2.5 s, full windows only). exact/proximity: 61,973 windows (every integer game second, partial windows at period edges included) [V both summaries]. The two window grids differ slightly; detection results do not depend on the windows (§2) |
| R3 | Curve intersection of each pair inside a window; discard non-intersecting pairs | Code [V] | **Implemented** exactly as his code (2D) | `geometry.py:51 intersect`; S10: 239,218 of 14,059,734 pair-windows have ≥ 1 intersection [V catalogue summary] |
| R4 | "Same time **and** same position" ("one point at one time ... exactly the same for both players") | Text [R] | **Tested and rejected as an exact rule**; replaced by synchronized proximity | Exact solver gives 566 events in S10 = 551 cloned intervals + 15 lattice points; 0 survive ±0.5 mm jitter (§2.4, §4) |
| R5 | Intersection coordinates P and the timestamps | Code [V] (P, indices) | **Implemented** (P exact; t_A, t_B per player) | catalogue record fields `x`, `y`, `time_a_s`, `time_b_s`, `time_diff_s` [V `DP/m11-occlusion-catalogue-v1/3812/root.catalogue.jsonl`] |
| R6 | Deduplicate repeats from overlapping windows | Text [R] | **Implemented** | 2D: 468,859 window-level hits → 168,818 unique segment-pair hits → 166,819 events. Exact: 12,004 → 566 [V summaries] |
| R7 | Before / After segments around P | Diagram [V] | **Implemented** | catalogue `trajectories.{a,b}.n_vertices_before/after`, `catalogue.py:85 paths`; v3 contexts give the incoming/outgoing valid track for [t* − 2.5, t* + 2.5] s |
| R8 | Identity preservation before and after the occlusion | Not found as an explicit advisor statement in the inspected material | **Partial** — studied as the M11 identity-recovery goal; no validated final model for the advisor population | [I] The requirement is inferred from the diagram (A_Before → A_After) and the project goal. Experiments in §6 |
| R9 | "Do not estimate P yourself" / no x-metre distance rule | Text [R] | **Followed for R3/R4**; **departed from** in the current proximity dataset (R = 0.25 m), because R4 has no non-artifact solutions | The departure is a project decision and needs his approval [U] |

---

## 2. Audit of every intersection algorithm

### 2.0 Shared input: Step 1 trajectories (S40) [V `DP/m11-step1-verification-v1/summary.json`]

- Provider positions are integer mm, updated at about 15 Hz in 29.97 fps files [R `m11_exact_spatiotemporal_extraction.md:49,53`].
- Each player's trajectory is a piecewise-linear path through its update instants.
- A step is not joined when any of these holds:
  - it crosses a period;
  - the gap is > 0.5 s unobserved;
  - it is an implausible jump (> 12 m/s and > 1 m);
  - it is unattributed.
- Counts over S40: 81,826,714 steps; 938 invalid, **all by period change, 0 by gap**; 385,029 jumps (45,943 in frame-wide shifts);
  2,530,865 duplicate-position samples; 1,243,070 cloned moving segments.
- The provider fills unobserved time with ESTIMATED positions instead of leaving gaps.
- These validity rules are ours, not the advisor's [R `m11_occlusion_catalogue_realignment.md:60-74`]. On 3812, joining
  jump steps raises the 2D count from 18,469 to 22,144; joining duplicate-position samples raises it to 63,118.

### 2.1 2D geometric intersection (advisor's code) — `detect.py`, `geometry.py`, `catalogue.py`

| Item | Detail |
|---|---|
| Functions | `detect.py:327 detect`, `geometry.py:51 intersect`, `geometry.py:117 classify_vertex`, `catalogue.py:85 paths`, `catalogue.py:280 run_match` |
| Equation | Segments A(u) = A0 + u·(A1 − A0) and B(v) = B0 + v·(B1 − B0), u, v ∈ [0, 1]. Solve A(u) = B(v) with 2D cross products (proper / vertex / collinear). P = A(u). Times: t_A = t0,A + u·(t1,A − t0,A) and t_B = t0,B + v·(t1,B − t0,B) |
| Time used? | **No.** Space only; each player's time at P is read afterwards |
| Interpolation / thresholds | Linear segments between samples; geometric tolerance 1e-6 m; **no** time or distance threshold; no smoothing |
| Dedup | One record per segment pair over all windows; hits at the same vertex (≤ 1 mm, adjacent segments) merged; separate crossings of one pair stay separate |
| Output | `DP/m11-intersections-v3/` (identical events) and `DP/m11-occlusion-catalogue-v1/<match>/root.catalogue.jsonl` |
| Scope / count | **S10: 166,819 events** [V catalogue summary `totals.events`]; 3812: 18,469; per match 11,894–19,913 [R `m11_occlusion_catalogue_realignment.md:132-145`] |
| Δt | Exact ties (\|Δt\| ≤ 1e-9 s): 15 [V]; concurrent crossing segments: 3,485 [V]; median \|Δt\| 2.16 s [R `realignment.md:218`] |
| Retained? | Kept as the **immutable geometric baseline** of the advisor's code. **Not** used as the occlusion definition, because it ignores simultaneity |
| Limitations | Measures path crossings at different times; ~82 % of events are in pairs that re-cross within 5 s (jitter/weaving) [R `realignment.md:150`]; 21.7 % disappear under light SG smoothing [R `m11_spatiotemporal_occlusion_dataset.md:179`] |

**The 4.2 % figure, re-derived [V from `DP/m11-occlusion-catalogue-v1/<match>/root.catalogue.jsonl`, S10]:**

| \|t_A − t_B\| ≤ | events | share of 166,819 |
|---|---|---|
| 1e-9 s (tie) | 15 | 0.009 % |
| 0.05 s | 1,730 | 1.04 % |
| 0.10 s | 3,476 | 2.08 % |
| **0.20 s** | **7,085** | **4.25 %** (per match 3.86 %–4.96 %) |
| 0.50 s | 19,032 | 11.41 % |
| 1.00 s | 39,168 | 23.48 % |

- The 4.2 % is a **raw \|Δt\| ≤ 0.20 s subset that we derived**. It is not an output of the advisor's method
  [R `realignment.md:56`].
- Do not confuse it with the S1 audit class "time-aligned @0.20" in `DP/m11-intersections-v3/cross-match/summary.json`. That
  class uses a σ-dependent tolerance: clean 1,382 + uncertain 14,025 + invalid 1,416, with 149,996 non-aligned [V].

### 2.2 Advisor-recipe cross-check — `advisor.py`

| Item | Detail |
|---|---|
| Functions | `advisor.py:65 advisor_times` (`project` → cumsum → `searchsorted` → linear time); `advisor.py:79 run_bounds` |
| Purpose | Independent re-implementation of his Shapely recipe to check our P and times. It is a check, not a detector |
| Output | `DP/m11-advisor-check-v1/<match>.json` |
| Result | **S10, summed over the 10 files** [V]: 166,819 events; points re-solved 166,699 with a maximum error of 7.1e-15 m; times agree on **166,812**; 7 "self-revisit" loops where `project` returns the first nearest pass of the same player |
| Discrepancy | The catalogue summary reports `advisor_recipe_agrees = 166,814` (5 loops) [V]. The two runs use different agreement tests. Both numbers are stored; neither indicates a geometry error [I] |

### 2.3 Near-synchronous crossing with a temporal tolerance — `spatiotemporal.py`

| Item | Detail |
|---|---|
| Functions | `spatiotemporal.py:115 decide`, `:395 run_match` |
| Rule | h = 0.0667 s (one provider update); σ_dt is first-order timing uncertainty; z = 2. Reject if \|Δt\| > max(h, 2σ_dt). Accept if \|Δt\| ≤ 2σ_dt ≤ h **and** speed ≥ 0.5 m/s, angle ≥ 15°, no missing sample, no artifact vetoes. Otherwise ambiguous |
| Time used? | Yes, as a tolerance on the 2D crossing times |
| Scope / count | S10: 166,819 → **284 accepted (0.17 %)**, 8,841 ambiguous, 157,694 rejected; 103 accepted with both positions observed [R `m11_spatiotemporal_occlusion_dataset.md:117`; spot-checked per event in §3] |
| Sensitivity | Accepted ranges **0 to 1,454** over defensible (τ, z) [R `:91-99`] |
| Retained? | Kept as an analysis set. **Superseded** as the primary definition: the count is resolution-limited and threshold-dependent, and it still requires a 2D crossing, so it misses meetings whose sampled paths do not cross [R `:219`] |
| Output | `DP/m11-occlusion-spatiotemporal-v1/` |

### 2.4 Exact spatiotemporal solution A(t*) = B(t*) — `exact.py`

| Item | Detail |
|---|---|
| Functions | `exact.py:89 trajectories`, `:144 classify`, `:168 solve_overlap`, `:339 windows_of`, `:505 detect_exact`, `:590 run_match` |
| Equation | On each common time interval of two valid segments, D(t) = A(t) − B(t) is linear. With end values D0 and D1: if D0 = D1 = 0 → **interval**; if exactly one is 0 → **sample-instant point**; if cross(D0, D1) = 0 and D0·D1 < 0 → **interior point** at s* = D0·(D0 − D1)/\|D0 − D1\|²; otherwise no solution |
| Arithmetic | Integer mm, integer µs; float pre-screen then exact `Fraction` verification. **No tolerance** |
| Scope / counts | **S10** [V exact summary]: 212,653,306 synchronous pair-intervals → 139,471 with a solution → 12,004 window-level detections → **566 unique** = **15 interior points + 0 sample-instant points + 551 coincident intervals** |
| Perturbation | ±0.5 mm uniform jitter → `perturbed_exact_solutions = 0` in all 10 matches [V] |
| What the 566 are | 551 intervals: all same-team, ESTIMATED, provider cloning (`v1_duplicate_rule_would_exclude = 551` [V]). 15 points: the same 15 numerical ties as §2.1; all same-team, all with identical x at both segment ends (D_x ≡ 0) [V: the 15 points are listed in this session; for each, segment_a x0/x1 equals segment_b x0/x1] |
| Retained? | **Tested and rejected** as the occlusion definition (§4). Kept as evidence and as the R = 0 limit of proximity (566 = 566 [R `m11_sync_proximity_encounters.md:71`]) |
| Output | `DP/m11-exact-spatiotemporal-v1/` |

### 2.5 Synchronized proximity d(t) ≤ R — `proximity.py`, then v2 / v3

| Item | Detail |
|---|---|
| Functions | `proximity.py:470 detect_all`, `:191 episodes`, `:134 exact_row`, `:146 exact_border`; v2 dataset `proximity_dataset.py`; v3 contexts `proximity_context.py`; v3 audit `proximity_audit_v3.py` |
| Equation | On each pair-interval [lo, hi], r(s) = r0 + s·e for s ∈ [0, 1], so d²(s) = \|r0\|² + 2s(r0·e) + s²\|e\|². The minimum is at s* = clip(−r0·e/\|e\|², 0, 1). The within-R part comes from the roots of d²(s) = R². Values within 1e-3 mm² of R² are re-decided exactly. **A(t) is always compared with B(t) at the same t** |
| Episode | Contiguous within-R time of one pair. t* is the earliest instant of minimum distance. Classes: point / isolated (within R at ≤ 1 sample instant) / sustained |
| Threshold | Primary R = 0.25 m; sensitivity 0, 0.10, 0.50 m; thresholds nest exactly [R `m11_sync_proximity_encounters.md:210`] |
| v1 scope (S10) | 0.25 m: **8,215** episodes; 0 m: 566 [R `:186`] |
| **v2 scope (S40)** | **30,438** episodes at 0.25 m; 0 m: 2,750; 0.10 m: 13,915; 0.50 m: 61,106; coincident intervals 2,697; high-quality candidates 1,917 [V `DP/m11-sync-proximity-v2/development-index/summary.json`] |
| v3 context (S40) | Window [t* − 2.5, t* + 2.5] s, clipped to the period. Edge points are interpolated on the valid straddling segment. Unavailable intervals are listed with reasons. Context complete: **24,052** (v2: 24,088); 173 windows clipped by the period [V `DP/m11-sync-proximity-v3/development-index/summary.json`] |
| v3 audit (S40) | 30,438 mathematically valid and tracking-support valid; incoming complete 25,573; outgoing complete 26,540; context complete 24,052; provider artifact suspected **6,634**; **presentation-ready 1,766**; types: closest approach 27,670 / exact coincidence 40 / potential tracking ambiguity 2,728 [V `DP/m11-sync-proximity-v3-audit/summary.json`] |
| Retained? | **Current primary definition** (same time, near the same place). It requires no 2D crossing: 27,063 of 30,438 events have a spatial path crossing in context, and 2,595 of those a simultaneous one [V audit summary] |
| Limitations | R is a choice; d_min is near-uniform below 0.5 m, so there is no natural break [R `m11_sync_proximity_encounters.md:212`]. 27,819 events are ESTIMATED at t* [V audit `reasons.estimated_at_t_star`]. Isolated contacts are interpolation inferences. **No episode is a verified occlusion** |

### 2.6 Comparison summary

| Algorithm | Space | Time | Threshold | Scope | Count | Role now |
|---|---|---|---|---|---|---|
| 2D crossing (advisor code) | exact | ignored | none | S10 | 166,819 | baseline / candidates |
| \|Δt\| ≤ 0.20 s subset | exact | ≤ 0.20 s | 0.20 s (ours) | S10 | 7,085 (4.25 %) | historical (a1 population source) |
| Spatiotemporal tolerance | exact | ≤ 2σ ≤ h | h, z = 2, floors | S10 | 284 | analysis set |
| Exact A(t*) = B(t*) | exact | exact | none | S10 | 566 (551 clones + 15 lattice) | rejected |
| Proximity v1 | ≤ 0.25 m | synchronized | R | S10 | 8,215 | superseded by v2 |
| **Proximity v2 + v3 context** | ≤ 0.25 m | synchronized | R | **S40** | **30,438** (1,766 ready) | **current** |

---

## 3. Verified real examples (all from development match 3812 unless stated)

Clock = time since the period's first frame. Period starts in 3812 [V v2 records]: P1 141.007674 s, P2 3040.340340 s.
VISIBLE = provider visibility 1 (observed); ESTIMATED = 2.

### A — Path crossing at different arrival times (rejected as occlusion) [V]

| Field | Value |
|---|---|
| Event | catalogue `751439e3b92975d5`, `proper_crossing` |
| Players | home #3 (A) × away #10 (B), opposite teams, P2 |
| P | (−4.711, −8.637) m |
| Arrival | t_A = 6219.1846 s (P2 52:58.844); t_B = 6221.2875 s (P2 53:00.947) |
| Δt | **+2.103 s** |
| Distance | B is 3.92 m from P when A is there; A is 3.92 m from P when B is there; the closest same-time distance in ±2.5 s is 0.744 m |
| Status at crossing | A VISIBLE, B ESTIMATED |
| Flags | none (no jump, duplicate or substitution within 2.5 s) |
| Spatiotemporal decision | `rejected`, reason `sequential_pass`, in all 15 sensitivity settings |
| Source | `DP/m11-occlusion-catalogue-v1/3812/root.catalogue.jsonl`, `DP/m11-occlusion-spatiotemporal-v1/3812/root.events.jsonl`; figure `DP/m11-exact-spatiotemporal-v1/contrast-3812/fig-751439e3b92975d5.png` |
| Interpretation | Satisfies the advisor's code but not his "same time" sentence. The players never meet. This is the typical case: median \|Δt\| is 2.16 s |

### B — Nearly equal arrival times, both observed [V]

| Field | Value |
|---|---|
| Event | catalogue `e699a1a7e0b85335`, `proper_crossing` |
| Players | home #8 × away #7, opposite teams, P1 |
| P | (44.794, −3.012) m |
| Arrival | t_A = 374.504597 s, t_B = 374.500966 s (P1 03:53.497 / 03:53.493) |
| Δt | **−0.0036 s** (B first) |
| Distance | A is 1.6 cm from P when B is there; B is 0.8 cm from P when A is there |
| Status | both VISIBLE at the crossing; crossing angle 64.6°, speeds 4.9 and 2.5 m/s |
| Spatiotemporal decision | `accepted` at the primary rule (τ = h, z = 2, σ_dt 0.028 s); `ambiguous` at (0.5h, z ≥ 2) and (1h, z = 3) |
| Proximity | v2 episode `dcd56d806105a6c1`: d_min 0.008 m, 0.122 s, isolated, observed/observed |
| Interpretation | A physically plausible simultaneous crossing with observed tracking. It is still **not exactly** simultaneous (3.6 ms), so the exact solver does not find it. This is the strongest kind of example; there were 6 with these properties in 3812 (opposite teams, both visible, 0 < \|Δt\| ≤ 0.01 s, no flags) |

### C — Exact same-time, same-position solution [V]

**C1 (3812, estimated)** — exact `3dc953ace4a43656` = catalogue `b04199215143947c` = proximity `22092c0d35089442`.

| Field | Value |
|---|---|
| Players | home #16 × home #3 (same team), P2 05:12.288 (t* = 3352.628446 s) |
| P | (−45.9147, 0.8791) m; A(t*) = B(t*) |
| Δt | 0 (numerical tie); both segments span the same frames 100477 → 100479 with u = 0.6373 |
| Structure | identical x at both ends for both players (−45.981 → −45.877); only y crosses |
| Status | both ESTIMATED (A ESTIMATED for the whole window, observed share 0.0) |
| Spatiotemporal decision | `ambiguous` (`timing_precision_insufficient`) |
| v3 audit | `exact_synchronized_coincidence`; context complete; **not** presentation-ready (`estimated_at_t_star`, `low_observed_share`) |

**C2 (3816, observed)** — exact `40f1f9388a619423`.

| Field | Value |
|---|---|
| Players | away #21 × away #13 (same team), P1 26:57.083 (t* = 1719.0515 s) |
| P | (40.8906, −6.6371) m (exact rational mm: 1512954/37, −245573/37) |
| Structure | identical x interval for both (40.752 → 41.037) on the same frames; y crosses (A −6.641 → −6.633, B −6.677 → −6.595) |
| Status | both VISIBLE |
| Proximity | v2 episode `0d7f6b64ff15cb93`: 0.713 s sustained, observed/observed |

**Interpretation (C1 and C2).** True solutions of A(t*) = B(t*) exist, but all 15 in S10 share the D_x ≡ 0 lattice pattern
and are same-team, and none survives ±0.5 mm jitter. They are provider-structured coincidences, not evidence of a physical
occlusion **[V pattern; I meaning]**.

### D — Cloned provider track [V]

| Field | Value |
|---|---|
| Event | proximity v2 `548ad35e88e51867`; exact interval `1803eb7e8a5b7a57` |
| Players | home #6 (player 214) × home #8 (player 474), P1 47:40.460 → end of period |
| Position | identical (19.450, 1.421) at t*; identical moving coordinates for **2.202 s** (33 merged exact intervals) |
| Status | both ESTIMATED; `duplicate_provider_position`, `IMPLAUSIBLE_JUMP`, `FRAME_WIDE_SHIFT` flags |
| Support | both incoming tracks cut at t* by an implausible jump; outgoing ends at 3003.670 s (period end 3003.704 s) |
| v3 audit | `potential_tracking_ambiguity`; provider artifact suspected; not ready; 82 spatial path crossings and 64 simultaneous ones in context |
| Context | Over S10 the 551 intervals start at shared instants. At these instants all 22 players are ESTIMATED, and the longest clone is 109 s, home #6/#8 over 190.6 m [R `m11_sync_proximity_encounters.md:150-161`] |
| Interpretation | A provider artifact. Counted as an exact solution, but it is **not** an occlusion |

### E — Nonzero synchronized proximity, observed, presentation-ready [V]

**E1** — v2 `bea2f3e495f944ec`.

| Field | Value |
|---|---|
| Players | home #21 (13228) × away #21 (1522), opposite teams, P1 35:53.343 (t* = 2294.3503 s) |
| A(t*), B(t*) | (34.2547, 7.1962), (34.1498, 7.1587) m |
| d_min | **0.111 m**; within 0.25 m for 0.298 s (sustained) |
| Status | both observed at t*; observed share 0.97 / 1.00 |
| Context | complete, 2.5 s incoming and outgoing for both; no unavailable interval |
| v3 audit | `closest_synchronized_approach`; **presentation-ready**; 3 spatial path crossings, 0 simultaneous |

**E2** — v2 `4e66a0a40b832021`.

| Field | Value |
|---|---|
| Players | home #4 × away #5, P1 06:48.108 |
| d_min | 0.052 m; isolated, 0.078 s |
| Status | both observed; observed share 0.95 / 0.76 |
| v3 audit | presentation-ready; 2 spatial crossings, 0 simultaneous |

**Interpretation.** These are the intended population: same time, about 5–11 cm apart, observed tracking. Their paths cross,
but never at the same instant. **This is still not a verified occlusion**: there is no video check and no camera geometry.

### F — Interrupted by a jump, missing support and a period boundary [V]

| Field | Value |
|---|---|
| Event | v2 `7d69f374b095e733` |
| Players | home #4 × home #3, P1 47:40.460 |
| Position | identical (17.592, −0.001); exact coincidence and coincident interval (clone) |
| Support | A: incoming 2.5 s, outgoing 2.202 s. B: **incoming only 0.467 s**, cut by an `implausible_jump` over [t* − 2.5, t* − 0.467] s |
| Period end | after +2.202 s both are `gap_period_or_time`, then `outside_period`; the window is clipped by the period |
| Status | both ESTIMATED; B has `OVERSPEED_STEP` |
| v3 audit | `potential_tracking_ambiguity`; provider artifact suspected; not ready |
| Interpretation | The context cannot show a full before/after for B. v3 reports the unsupported time with a reason instead of inventing positions |

Also in category F: `548ad35e88e51867` (example D) has **both** incoming tracks cut exactly at t* (incoming 0.0 s).

**Category coverage.** All six categories have real examples. No category lacks a trustworthy example.

- B and E are the only ones with observed tracking on both players.
- C's examples are trustworthy as *solutions* but not as *occlusions*.

---

## 4. Why exact intersection was not selected

| Factor | Finding | Label | Source |
|---|---|---|---|
| Measure-zero condition | A(t) = B(t) is two equations (x, y) in one unknown t; generic motion has no solution | **[V]** mathematics; confirmed by data | `exact.py:144`; §2.4 |
| Millimetre rounding | All 4,067,102 coordinate values are integer mm, so exact equality is a lattice event | [R] | `m11_exact_spatiotemporal_extraction.md:53` |
| Perturbation | ±0.5 mm jitter → 0 exact solutions (from 566) | **[V]** | exact summary `perturbed_exact_solutions = 0` |
| Clones | 551 / 566 are coincident intervals of cloned ESTIMATED tracks | **[V]** | exact summary; example D |
| Remaining points | 15 / 15 have D_x ≡ 0 and are same-team | **[V]** | 15 points listed in this session |
| Estimated positions | ESTIMATED share 53–73 % per match (S40). 0.25 m encounters are est/est in 74 % of cases, against 42–58 % for random pairs | [V] share; [R] 74 % | Step 1 summary; `m11_sync_proximity_encounters.md:255-261` |
| Sparse updates | About 15 Hz updates: arrivals closer than one update (0.067 s) cannot be ordered | [R] | `m11_spatiotemporal_occlusion_dataset.md:39-48` |
| Different arrival times | 2D crossings: median \|Δt\| 2.16 s; 95.75 % have \|Δt\| > 0.20 s | **[V]** | §2.1 table |
| Sensitivity of exact equality | A 3.6 ms offset (example B) makes a physically simultaneous observed crossing invisible to the exact solver | **[V]** | example B |
| Threshold false positives (for the alternative) | Proximity at R picks up pass-bys; d_min is near-uniform below 0.5 m; 0.50 m adds mostly sustained side-by-side movement | [R] | `m11_sync_proximity_encounters.md:212-215` |
| Physical meetings are frequent | Same-time episodes ≤ 10 cm: 3,159; ≤ 50 cm: 13,786 (S10) | **[V]** | exact summary `near_0.1m`, `near_0.5m` |

**Hypotheses, not demonstrated** [I]:

- That provider ESTIMATED positions are least reliable exactly during occlusions.
- That the D_x ≡ 0 pattern comes from a provider-side snapping or estimation step.
- That 0.25 m between centre points corresponds to visual occlusion in broadcast video.

None of these was tested against video.

---

## 5. FOOTBALLTrace audit

### 5.1 Bibliography

Mostafa, A. H., Rushdi, M. A., **Basha, T. A.**, and Sayed, K. (2023). *FOOTBALLTrace: An AI-Based System for Football
Player Tracking with Occlusion Detection and Trajectory Correction.* In Proceedings of the 11th International Conference on
Sport Sciences Research and Technology Support (icSPORTS 2023), pp. 194–201. SCITEPRESS. ISBN 978-989-758-673-6;
ISSN 2184-3201. DOI: 10.5220/0012226800003587. Link: https://doi.org/10.5220/0012226800003587.

- Local copy: `data/raw/122268.pdf`. Text extracted in this session, read-only.
- **[I]** Tamer A. Basha (Cairo University) is a co-author. Whether he is the project advisor "Dr. Basha" is not established
  by the repository [U].

### 5.2 What the paper specifies [V, read from the PDF]

| Element | Paper | Page / section |
|---|---|---|
| Video and tracker | 3 cameras at 25 fps stitched into a panorama; homography to a pitch top view; YOLOv7 + Deep Affinity Network (DAN); a long occlusion is δb = 15 frames | p. 196–198, §3 |
| Candidate detection | Sliding window of **50 frames (2 s)** with **50 % overlap**; mutual Euclidean distance matrix per frame; a possible occlusion if the distance ≤ **τ = 4** "for the purpose of reproducibility" for **≥ 50 % of the window**; τ optimisation left to future work | p. 197–198, §3.3 |
| Units | Top-view image units (Table 1 coordinates 264–269 / 201–206); **no metre scale** | p. 196, Table 1 |
| Operator | An operator confirms each candidate on the video: 211 candidates → **40 events (80 trajectories)** confirmed | p. 198 §3.3; p. 199 §4 |
| Analysis window | 5 s around the occlusion event | p. 198, §3.4 |
| Smoothing | Savitzky–Golay (window w, order n), then resampling with step s | p. 198, §3.4 |
| Breakpoint (Algorithm 1) | v1 = P(i) − P(i−1), v2 = P(i) − P(i+1), "Angle" = v1·v2 / (\|v1\|\|v2\|), take the index of the minimal value | p. 198, Algorithm 1, Eq. 1 |
| Correction | Split both trajectories at their breakpoints; connect T_before(occluded) → T_after(occluding) and vice versa (always exchanges tails) | p. 198–199, §3.4, Fig. 9 |
| Tuning | 80 % (64 traj.) train / 20 % (16) test; grid search with 5-fold CV; chosen w = 7, n = 2, s = 7 by majority vote; mean validation 90.5 % | p. 199, §3.5, Table 2 |
| Result | **87.5 % (70 / 80)** disrupted trajectories corrected; association error 20 % | p. 199, §4 |

**Internal issues in the paper [V reading; I significance]:**

- "Angle" is computed as a cosine, and the minimum cosine is the straightest point, while the text and Fig. 8 describe a sharp
  turn.
- The 87.5 % is stated over all 80 trajectories, which include the 64 used for tuning. No separate held-out figure for the 16
  test trajectories was found in the extracted text.
- The temporal condition is a per-frame distance over ≥ 50 % of a 2 s window. There is no same-instant intersection.

### 5.3 Comparison with our implementation

| Aspect | FOOTBALLTrace | M11 | Source |
|---|---|---|---|
| Candidate rule | distance ≤ τ (image units) for ≥ 50 % of a 2 s window | synchronized d(t) ≤ 0.25 m, analytic within each sample interval, any duration | `proximity.py`; §2.5 |
| Temporal condition | windowed duration | exact same t; duration stored, not required | — |
| Intersection | not used | 2D and exact versions tested (§2.1, §2.4) | — |
| Smoothing | SG w = 7, n = 2, s = 7 | none in detection; SG studied only as sensitivity (§2.3) | — |
| Angle / breakpoint | Algorithm 1 | E0-Literal reproduction: breakpoint within the interval 0.560 / 0.565 vs 0.555 for a random point, i.e. **at chance** (3812, 761 pairs) | [R] `m11_step3_taskb_footballtrace.md:148-152`; `matching/footballtrace.py` |
| ID-switch correction | always exchange tails after operator confirmation | E0-Literal exchanges in 761 / 761, so agreement with provider labels is 0.0 by construction. E0-Adapted (cost-based) is the weakest one-to-one method on task B (high tier complete 0.243) | [R] same report §5 |
| Evaluation | 80 operator-confirmed trajectories | provider labels as reference; no human swap labels | [R] |
| Verdict | — | **NO-GO** as an identity-recovery method on FIFA RAW (Step 3) | [R] `m11_step3_taskb_footballtrace.md:198-201` |

---

## 6. Earlier identity-recovery experiments (75.0 %, 79.7 %, 50.7 %)

| Figure | What it is | Label | Source |
|---|---|---|---|
| **75.0 %** (0.7501) | Accuracy of the "assume they crossed" prior (fitted nearest rule, inverted) on the a1 population | **[V]** | `DP/m11-recovery-v1/experiments/a1-advisor-intersections/results.json` → `results.nearest.report.overall.accuracy` |
| **79.7 %** (0.7966) | GBM trained and evaluated (LOMO) on the a1 population; reported CI [0.789, 0.804] | **[V]** value; [R] CI | same file → `results.gbm`; CI from `m11_advisor_intersection_correction.md:87` |
| **50.7 %** (0.5065) | The a1-trained GBM applied to identity-free contacts whose labelled paths do **not** cross (n = 162,600 rows) | **[V]** | `.../a3-transfer-intersections-to-contacts/results.json` → `by_geometry.no_crossing.accuracy`; same file: crossing not aligned 0.7262, aligned 0.8455 |

**Design** [V `results.json` fields; R for the details]:

- Population: v3 2D intersections with \|Δt\| ≤ 0.20 s, tiers clean + uncertain.
- 6,936 interactions × 15 synthetic gap/noise scenarios = 104,040 rows (S10).
- Leave-one-match-out within development.
- Task: given two "before" tracks and two anonymous "after" tracks across a synthetic gap, decide the association.
- Labels: **provider identities**, not human-verified identity.

**Status:**

- **Historical exploratory diagnostics. Not a validated final model.**
- The a1 population is a crossing-only selection, so a model can learn the prior "assume they crossed". a3 shows this
  directly: chance on non-crossings.
- No untouched final evaluation exists for this pipeline. Validation was examined (Step 6 and the contact run), and the locked
  test is consumed [R `m11_advisor_intersection_correction.md:133`].
- Not to be confused with:
  - the contact-formulation validation GBM, 0.767 [R];
  - the Step 7 locked-test pairwise AUC 0.838 [R `m11_step7_final_locked_test.md:26`]. That is a different task, population
    and metric.

---

## 7. Current state

### 7.1 Confirmation of the requested figures

| Item | Value | Label | Source |
|---|---|---|---|
| Development matches | **40** | [V] | v2, v3 and Step 1 summaries `matches = 40` |
| Events (0.25 m) | **30,438** | [V] | `DP/m11-sync-proximity-v2/development-index/summary.json` `episodes`; v3 audit `encounters` |
| Complete contexts (v3, t* ± 2.5 s) | **24,052** | [V] | v3 index `v3_complete`; audit `context_complete` |
| Provider-artifact-suspected | **6,634** | [V] | audit `provider_artifact_suspected` |
| Presentation-ready | **1,766** | [V] | audit `presentation_ready` |
| Tests | **106 passed** | [V] run 2026-11-11 in this session: `tests/test_proximity_context.py`, `test_m11_encounter_explorer.py`, `test_proximity_audit.py`, `test_m11_dashboard.py`, `test_intersections_proximity.py` | pytest output "106 passed in 53.02s" |
| Window | t* ± 2.5 s, event-centred, clipped to the period (173 clipped) | [V] | v3 context records `definition: event_centred`, `half_width_s: 2.5` |

- "6,634 artifacts" is the count of events with **suspected** provider artifacts (clones, identical positions, jumps, shifts).
  It is not 6,634 confirmed artifacts.
- The 106 passing tests show code consistency with its own specification and stored outputs. They do **not** show that any
  identity is correct or that any event is an occlusion.

### 7.2 Presentation-ready is not a validated occlusion label

"Presentation-ready" (`proximity_audit_v3.py`) means all of the following:

- the context is complete;
- observed share and path length are above the audit minimums;
- there is no suspected provider artifact;
- there is no period boundary;
- there is no critical flag.

It is a **display-quality filter**. No event in any M11 dataset is a human-verified or video-verified occlusion. Labels
throughout are provider slots.

### 7.3 Remaining limitations

1. **No ground truth.** There is no video, no camera calibration and no human swap labels for the FIFA RAW matches.
2. **ESTIMATED dominance.** 27,819 / 30,438 events are ESTIMATED at t* for at least one player [V audit reasons].
   `low_observed_share` affects 26,975 [V].
3. **Threshold choice.** R = 0.25 m has no natural break in the data and needs the advisor's approval.
4. **Interpolation.** Contacts shorter than one update (isolated) are inferred by linear interpolation.
5. **Selection bias.** Any training population restricted to crossings teaches the "crossed" prior. A label-symmetric
   population is required.
6. **Repeated contacts.** Repeated contacts of one pair are separate episodes. Episode grouping is undecided.
7. **No untouched evaluation.** There is no untouched final evaluation for an advisor-aligned identity model.
8. **Uncommitted work.** The v3 context, v3 audit, Step 1 verifier and dashboard changes are uncommitted (git status at session
   start).

---

## 8. Compliance matrix

| Requirement | Implemented | Partial | Tested and rejected | Not implemented | Evidence |
|---|---|---|---|---|---|
| 22 trajectories / all pairs | ✔ | | | | §1.2 R1 |
| 5 s sliding windows, 1 s stride | ✔ | | | | R2 (5 s, not 4 s) |
| Curve intersection (his code) | ✔ | | | | §2.1; 166,819 |
| Same time **and** same position (exact) | | | ✔ | | §2.4, §4; 566 artifacts, 0 after jitter |
| Same time, near the same position (proximity) | ✔ (project decision) | | | | §2.5; needs his approval |
| Intersection coordinates and timestamps | ✔ | | | | catalogue fields; exact µs/mm |
| Deduplication across windows | ✔ | | | | 468,859 → 166,819; 12,004 → 566 |
| Before/After segments | ✔ | | | | catalogue `paths`, v3 contexts |
| Identity preservation before/after | | ✔ | | | §6: exploratory only; no validated model |
| Collinear-overlap handling as in his code | | | | ✔ (deliberately) | his code returns `[]`; we keep overlaps as events |
| `project`-based time | ✔ (cross-check only) | | | | 166,812 agree; 7 loops |

---

## 9. Unresolved questions for the advisor [U]

1. **Same time or same place?** Does an occlusion require the same instant? If yes, may we use a spatial tolerance (proximity,
   R = 0.25 m), given that the exact rule yields only artifacts? Or does a sequential path crossing (example A) count?
2. **ESTIMATED positions.** Do events with ESTIMATED positions count, and should they be analysed separately?
3. **Validity rules.** Should jumps, gaps over 0.5 s and duplicate positions break trajectories (current), or be joined as in a
   raw `LineString`?
4. **Repeated contacts.** Should repeated contacts within seconds be one episode?
5. **Label-symmetric training.** May non-crossing same-time meetings be added to avoid the selection prior?
6. **Window length.** 5 s (±2.5 s, used) or 4 s?
7. **Ground truth.** Is any video or annotation source available to verify a sample of events as true occlusions?
8. **FOOTBALLTrace.** Is the co-author T. A. Basha the advisor? Is FOOTBALLTrace's proximity-window candidate stage (not the
   breakpoint correction) the intended reference?

---

## 10. Concise summary

**Verified in this session:**

- The advisor's code is 2D and returns segment indices only. His diagram shows Before/After segments.
- 166,819 2D crossings (S10). 4.25 % have \|Δt\| ≤ 0.20 s (7,085), and only 15 are exact ties.
- The advisor-recipe agreement is 166,812 (advisor-check files) vs 166,814 (catalogue summary): two stored tests that differ.
- Exact solver: 566 = 551 clones + 15 D_x ≡ 0 points, and 0 after ±0.5 mm jitter.
- Proximity on S40: 30,438 events, 24,052 complete contexts, 6,634 artifact-suspected, 1,766 presentation-ready.
- The 106 tests pass.
- 75.0 / 79.7 / 50.7 % re-read from the a1/a3 result files.
- FOOTBALLTrace details read from the PDF.
- Real examples A–F, all with stored IDs.

**Uncertain or reported only:**

- The advisor's message wording (transcribed in reports; the originals are not in the repo).
- The 284 / 8,215 counts and the CI of 79.7 %.
- Whether "identity preservation" was stated by the advisor.
- Whether the FOOTBALLTrace co-author is the advisor.
- Every claim that an event is an occlusion. None is verified.

**File:** `docs/research/M11_ADVISOR_EVIDENCE_HANDOFF.md`
