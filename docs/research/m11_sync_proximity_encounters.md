# M11 — Synchronized player-proximity encounters (`m11-sync-proximity-v1`)

| Item | Detail |
|---|---|
| Code | `src/football_fatigue/intersections/proximity.py` |
| Tests | `tests/test_intersections_proximity.py` |
| Outputs | `data/processed/m11-sync-proximity-v1/` |
| Data used | Development matches only: 3812, 3814, 3815, 3816, 3829, 3833, 3836, 3847, 3851, 10506 |

Not done in this stage:

- no validation or test match was opened;
- no model was trained;
- no labels were created;
- raw data and earlier outputs are unchanged (each manifest records `raw_unchanged`);
- nothing is committed.

The events are **tracking-derived identity-ambiguity proxies**, not visually verified occlusions.

## 1. Repository audit: what each earlier M11 pipeline did

| Definition | Implementation | 3812 | 10 dev matches |
|---|---|---|---|
| (1) 2D path intersection: each player passes P at its own time | `detect.py` (v1/v3), `catalogue.py` → `m11-occlusion-catalogue-v1` (the advisor's Shapely-style code) | 18,469 | 166,819 |
| (5) Spatial crossing at different arrival times | Same population as (1): median \|t_A − t_B\| 2.16 s | ≈ all of (1) | ≈ all of (1) |
| Crossing plus a time tolerance (τ = 0.0667 s, 2σ, quality vetoes) | `spatiotemporal.py` → `m11-occlusion-spatiotemporal-v1` | — | 284 accepted |
| (2) Exact A(t*) = B(t*) | `exact.py` → `m11-exact-spatiotemporal-v1` | 18 | 566 |
| (4) Sustained coincident trajectories | The 551 intervals in (2) | 17 | 551 |
| **(3) Synchronized proximity, d(t) ≤ 0.25 m** | **`proximity.py` → `m11-sync-proximity-v1` (this report)** | **844** | **8,215** |

The advisor's diagram and code are `data/raw/Screenshot 2026-11-09 175508.png` and `175517.png`. His code computes (1). His text ("one point at one time ... exactly the same for both players") asks for simultaneity. The definition in this report is (3), as instructed.

### Inherited filtering rules

The trajectories are those of `exact.trajectories`, unchanged. Each rule is classified below.

| Rule | Effect (3812) | Classification |
|---|---|---|
| Segments join consecutive provider update instants; repeated frames are not vertices | 2,019,347 segments | Mathematically necessary: this is the provider's 15 Hz cadence; repeated frames carry no new position |
| Held positions (runs of 3 or more equal frames) are stationary, then moving | 424 held runs | Data integrity: avoids smearing a stop over the hold |
| Gap of more than 0.5 s unobserved: not joined | 22 steps (with period changes) | Configurable quality restriction (Step 1 `max_gap_s`); interpolating long gaps would invent motion |
| Period change: not joined | Included above | Mathematically necessary: there is no motion across periods |
| Implausible jump (> 12 m/s and > 1 m): not joined | 14,658 steps | Configurable quality restriction: a teleport, often an identity discontinuity. The trajectory is split, not dropped; the jump is kept as a cut (see example 4) |
| Unattributed or ambiguous raw sample, side over-count: not joined | 1,504 steps | Data integrity: the slot identity is unknown |
| Repeated timestamps (dt ≤ 0) | Step invalid, so no zero-length segment | Data integrity (tested) |
| Coordinate precision | Integer mm (checked); time integer µs (checked) | — |
| ESTIMATED positions | **Kept and flagged** | Not a filter |
| Low confidence, overspeed sub-metre steps, substitution window | **Kept and flagged** | Not a filter |
| v1 duplicate-position exclusion | **Not applied**; flagged `duplicate_provider_position` | It was an unnecessary heuristic for this goal; it is now a flag |
| Speed, angle, direction, confidence, shape filters | **None** | — |
| Window membership | Labelling only; it never removes an encounter | — |

Window deduplication is global per match, by pair and contiguous synchronized time.

## 2. Corrected objective

The target is two players near the same position at exactly the same timestamp: A(t) is compared with B(t), never with B(t + Δt).

- An exact coincidence, A(t*) = B(t*), is two equations in one unknown. In this data it arises only from provider artifacts: a ±0.5 mm perturbation leaves 0 of 566.
- A 2D crossing has no simultaneity at all.
- Synchronized proximity, d(t) ≤ R, is the practical same-time, same-place criterion. The trajectories may have any shape, and they need not overlap before or after the encounter.

## 3. Algorithm (continuous time, analytic)

1. **Trajectories.** Each player is a time-parameterised piecewise-linear path through its own samples: hundreds of short segments per 5 s, never one line per window.
2. **Pair-intervals.** Every pair of segments of two players with overlapping time, [lo, hi] = [max t0, min t1], is one pair-interval. Time is thereby split at the union of both players' sample instants: 21,050,439 synchronous plus 1,516 partial overlaps in 3812. Both positions are evaluated at the same lo and hi, and a test checks this against independent interpolation.
3. **Relative position.** r(s) = r0 + s·e for s ∈ [0, 1], so d²(s) = |r0|² + 2s·(r0·e) + s²·|e|².
   - Minimum: s* = clip(−r0·e / |e|², 0, 1).
   - Within-R part: the roots of d²(s) = R², clipped to [0, 1]. A sub-interval that touches the end of a pair-interval snaps to that exact instant.
4. **Precision.** The computation uses float64 on integer mm. Any pair-interval whose minimum lies within 1e-3 mm² of R² is re-decided with exact rationals. R = 0 therefore reproduces the exact solver: 566 = 566, per match.
5. **Episodes.** Within-R sub-intervals of one pair that touch or overlap form one episode. A later approach by the same pair is a new episode, so the result is never one closest approach per pair.
6. **Episode fields:**
   - start, end, duration;
   - t* (earliest instant of minimum distance) and d_min;
   - A(t*), B(t*) and the midpoint P* (used for indexing and plots only);
   - `exact_coincidence` and `coincident_interval`;
   - class:
     - `point`: duration 0;
     - `isolated`: within R at no more than one sample instant, so the contact lies between samples;
     - `sustained`: within R across at least two sample instants.

   The class is descriptive; durations are stored so other splits remain possible.
7. **Windows.**
   - Integer-second centres, ±2.5 s, clipped at the period edges.
   - An episode is detected by every window it overlaps (the window-level count).
   - Each episode keeps the context of its canonical window, the one with the centre nearest t*.
   - Brute force over every window gives the same unique set (test).
8. **Rounding.** Full float64 internally and in JSON (no rounding). Times are in seconds and positions in metres, RAW frame.

**Higher-order interpolation was not adopted.** VISIBLE tracks deviate from their neighbours' midpoint by a median of 5–7 mm (§10). Over 0.067 s steps a spline would change positions by millimetres, while adding overshoot near turns and stops. Piecewise linear is exact to the samples and reproducible.

**Limitation.** Between samples the true path is unknown. A within-R contact shorter than one step (an `isolated` episode) is an interpolation inference.

## 4. Match 3812 results (measured)

Context shared by all thresholds:

- 6,169 windows of 1 s, with 22 to 23 players available (median 22);
- 1,425,655 pair-window comparisons (231 × windows = 1,425,039);
- 21,051,955 pair-interval comparisons;
- 630 period-slot pairs with common support.

| Statistic (unit) | 0.00 m | 0.10 m | **0.25 m** | 0.50 m |
|---|---|---|---|---|
| Segment-level detections (qualifying pair-intervals) | 12,088 | 12,740 | **15,089** | 23,942 |
| Window-level detections (episode × windows) | 880 | 2,452 | **5,164** | 10,178 |
| Repeated detections removed | 862 | 2,125 | **4,320** | 8,441 |
| **Unique episodes** | 18 | 327 | **844** | 1,737 |
| point / isolated / sustained | 1 / 0 / 17 | 0 / 247 / 80 | **0 / 427 / 417** | 0 / 515 / 1,222 |
| Exact coincidence / coincident interval | 18 / 17 | 18 / 17 | 18 / 17 | 18 / 17 |
| Distinct pairs / players | 15 / 13 | 168 / 30 | **265 / 30** | 335 / 30 |
| Same team / opposite teams | 18 / 0 | 143 / 184 | **331 / 513** | 646 / 1,091 |
| d_min median [p10, p90] (m) | 0 | 0.049 [0.005, 0.092] | **0.154 [0.022, 0.237]** | 0.338 [0.064, 0.482] |
| Duration median [p90] (s) | 8.98 [109.0] | 0.046 [0.229] | **0.100 [0.408]** | 0.184 [0.857] |
| observed/observed | 0 | 62 | **148** | 356 |
| observed/estimated | 0 | 54 | **162** | 323 |
| estimated/estimated | 18 | 211 | **534** | 1,058 |
| Support cut (gap, jump, period or substitution at an episode end) | 16 | 19 | **39** | 94 |
| Spans unobserved samples (interpolated gap) | 0 | 0 | 0 | 0 |
| Held position involved | 0 | 0 | 0 | 4 |
| Duplicate provider position | 17 | 17 | 17 | 17 |

## 5. Historical comparison (verified)

- **18,469 2D crossings (3812).** Re-read from `m11-occlusion-catalogue-v1`. They are path crossings at different times; for example, catalogue event `751439e3b92975d5` (example 7) has Δt = 2.10 s and a closest same-time distance of 0.74 m. Only a small minority are simultaneous.
- **18 exact events (3812).** Reproduced exactly by R = 0: 18 = 18 and 566 = 566 across matches.
- **844 proximity episodes at 0.25 m (3812).** These are the intended population. They include all 18 exact events (§6).
- **The 13,786 figure.**
  - **Coverage:** the ten development matches; for 3812 it is 1,472.
  - **Unit:** runs of consecutive **synchronous** pair-intervals of one pair whose float minimum is at most 0.5 m. It is continuous-time (analytic within each interval) and same-time.
  - **Exclusions:** partial overlaps (held positions, irregular cadence) are not part of it. It is an episode count, not samples, windows or pairs.
  - **Reproduced** exactly by `replica_near`, both per match and in total (13,786 = 13,786).
  - **Rigorous count at 0.50 m: 16,583 episodes.** The difference comes from:
    - partial overlaps are now included;
    - the old runs joined consecutive intervals whose minima were both ≤ 0.5 m, even when d rose above 0.5 m between them. Episodes now break wherever d(t) > R, which separates approaches.

## 6. The previous 18 exact events of 3812

All 18 are in `3812/root.summary.json` under `historical.exact_v1_events`. Each entry carries the segment rows, frames, coordinates and visibility of both players.

**What they have in common:**

- all 18 are same-team pairs;
- all have both players ESTIMATED;
- all qualify at 0.25 m;
- each lies inside one 0.25 m episode.

**The interior point** `3dc953ace4a43656` (#16 × #3, home, t* = 3352.628446 s) lies inside a 0.696 s sustained episode, `22092c0d35089442` (example 1).

**The 17 coincident intervals** start at four instants:

| Start (s) | Pairs | Duration (s) |
|---|---|---|
| 3001.468 | 7 | 2.202 |
| 3925.792 | 1 | 9.009 |
| 3935.802 | 2 | 8.976 |
| 4485.786 | 7 | 109.009 |

At 0.25 m each becomes an episode of the same length or slightly longer (up to +0.04 s, the approach and separation).

**Provider evidence.** At these instants **all 22 players are ESTIMATED**, and the provider copies one estimated track onto a teammate. The two then follow identical moving paths: home #6 and #8 cover 190.6 m in 109 s; away #5, #4 and #3 share one path. These are **cloned provider estimates, not occlusions**.

They are kept with `coincident_interval = true`, `pair_quality = estimated/estimated` and `duplicate_provider_position = true`, so they can be excluded or studied explicitly.

## 7. Results for the 10 development matches

**Per match:**

| Match | Windows | Episodes 0.00 m | Episodes 0.10 m | **Episodes 0.25 m** | Episodes 0.50 m | 0.25 m observed/observed | 0.25 m opposite teams |
|---|---|---|---|---|---|---|---|
| 3812 | 6,169 | 18 | 327 | **844** | 1,737 | 148 | 513 |
| 3814 | 6,022 | 84 | 279 | **636** | 1,238 | 60 | 345 |
| 3815 | 6,271 | 106 | 352 | **731** | 1,411 | 60 | 387 |
| 3816 | 6,612 | 30 | 357 | **908** | 1,855 | 84 | 536 |
| 3829 | 6,186 | 15 | 371 | **974** | 1,951 | 98 | 524 |
| 3833 | 6,427 | 52 | 321 | **715** | 1,526 | 83 | 438 |
| 3836 | 5,838 | 138 | 473 | **883** | 1,656 | 63 | 439 |
| 3847 | 6,418 | 28 | 341 | **759** | 1,570 | 66 | 438 |
| 3851 | 6,231 | 53 | 527 | **1,140** | 2,350 | 104 | 643 |
| 10506 | 5,799 | 42 | 248 | **625** | 1,289 | 63 | 362 |

**Totals.** 61,973 windows; 14,321,863 pair-window comparisons; 212,666,084 pair-interval comparisons.

| Total (10 matches) | 0.00 m | 0.10 m | **0.25 m** | 0.50 m |
|---|---|---|---|---|
| Unique episodes | 566 | 3,596 | **8,215** | 16,583 |
| Window-level detections | 12,004 | 27,483 | **52,077** | 99,298 |
| Segment-level detections | 139,471 | 147,111 | **174,319** | 263,234 |
| point / isolated / sustained | 15 / 0 / 551 | 0 / 2,225 / 1,371 | **1 / 3,651 / 4,563** | 1 / 5,064 / 11,518 |
| Coincident intervals | 551 | 551 | 551 | 551 |
| obs/obs · obs/est · est/est | 11 · 0 · 555 | 311 · 510 · 2,775 | **829 · 1,319 · 6,067** | 2,165 · 2,646 · 11,772 |
| Opposite / same team | 0 / 566 | 1,867 / 1,729 | **4,625 / 3,590** | 9,933 / 6,650 |
| Distinct pairs (summed over matches) | 150 | 1,464 | **2,370** | 3,183 |
| d_min median (m) | 0 | 0.044 | **0.150** | 0.348 |
| Duration median / p90 (s) | 8.98 / 40.9 | 0.067 / 1.51 | **0.117 / 0.68** | 0.189 / 1.19 |
| Support cut / held / duplicate position | 487 / 8 / 551 | 508 / 9 / 673 | **607 / 13 / 806** | 973 / 36 / 1,147 |

The 0.25 m episodes without coincident intervals number 7,664:

| Quality | d = 0 | d ≤ 0.10 m | 0.10 < d ≤ 0.25 m |
|---|---|---|---|
| observed/observed | 10 | 295 | 524 |
| observed/estimated | 0 | 483 | 836 |
| estimated/estimated | 4 | 1,498 | 4,014 |

Their median duration is 0.108 s (p90 0.48 s, p99 2.17 s). By team: 4,625 opposite-team and 3,039 same-team. By class: 3,651 isolated, 4,012 sustained and 1 point.

## 8. Sensitivity

- **The thresholds nest exactly.** Every episode has exactly one parent at the next threshold (checked at run time).
- **Growth from 0.10 m to 0.25 m:** 2.3× (3,596 → 8,215). **From 0.25 m to 0.50 m:** 2.0× (→ 16,583).
- **Below 0.5 m, d_min is close to uniform** (3812: 263 / 502 / 644 / 1,074 / 1,737 episodes with d_min ≤ 0.1 / 0.2 / 0.25 / 0.4 / 0.5 m). This is the signature of geometric pass-bys, which have a uniform miss distance. The 0.25 m count therefore scales with R and has no natural break.
- **The exception is the bottom bin**, which is over-represented: d_min ≤ 1 cm contains 58 episodes in 3812, about 1.7× the uniform expectation of 34. These are the provider coincidences.
- **Going from 0.25 m to 0.10 m mostly removes ESTIMATED pairs:** est/est 6,067 → 2,775 and obs/obs 829 → 311. The share of observed/observed stays about 9–10 %, so the narrower threshold does not purify quality.
- **Going from 0.25 m to 0.50 m adds mostly sustained same-place proximity** (sustained 4,563 → 11,518), such as walking together or marking, rather than new contact events.

## 9. Visual evidence (match 3812)

The figures and `examples.json` are in `data/processed/m11-sync-proximity-v1/examples-3812/`. `examples.json` gives, for each example, the original samples nearest t* (row, frame, t, x, y, VISIBLE/ESTIMATED), A(t*), B(t*), P*, d_min and the duration.

Each figure has two panels:

- **Left:** the 5 s paths. Lines are interpolated; filled dots are VISIBLE samples and hollow dots ESTIMATED samples. Same-time markers at t* ± 0.5 s and ± 1 s are joined by dotted lines, and A(t*), B(t*), P* and the R circle are drawn.
- **Right:** d(t) at the same t, with the R line, the episode shaded, and the instants where both players are at an original sample.

| # | Category | Event | Players | t* (s) | d_min | Duration, class | Quality |
|---|---|---|---|---|---|---|---|
| 1 | Previous exact interior point | `22092c0d35089442` | #16 × #3 home | 3352.628446 | 0 | 0.696 s, sustained | est/est |
| 2 | Previous coincident interval | `548ad35e88e51867` | #6 × #8 home | 3001.468135 | 0 | 2.202 s, sustained | est/est |
| 3 | Nonidentical trajectories | `4e66a0a40b832021` | #4 home × #5 away | 549.115782 | 5.20 cm | 0.078 s, isolated | obs/obs |
| 4 | Changing direction | `ab4e2fd9c1210bea` | #5 home × #3 away | 5574.420784 | 11.35 cm | 0.069 s, isolated | obs/obs |
| 5 | Brief threshold crossing | `0ef54f44fffe8aad` | #21 away × #7 home | 4981.554958 | 24.92 cm | 0.0074 s, isolated | obs/obs |
| 6 | Sustained proximity | `364220553110194c` | #21 × #8 home | 4182.916250 | 13.16 cm | 0.798 s, sustained | obs/obs |
| 7 | 2D crossing at different times (rejected) | catalogue `751439e3b92975d5` | #3 home × #10 away | P at 6219.185 / 6221.288 | closest same-time 0.74 m | — | — |
| 8 | Low quality, estimated | `1ca860ed5f4ee588` | #9 home × #22 away | 2585.602978 | 3.22 cm | 0.035 s, isolated | est/est |

Notes on individual examples:

- **Example 3:** two different curved paths meet once; the players are 5 cm apart at t* and more than 0.4 m apart one sample before and after.
- **Example 4:** A turns sharply at the meeting point. B's valid trajectory ends 0.43 s later because the next step was an implausible jump (not joined). This is a genuine identity-ambiguity case the dataset should keep.
- **Example 7:** the two paths cross, but the closest same-time distance in ± 2.5 s is 0.74 m, so it is not an encounter.
- **Selection rules** are in `examples.json`. They are for illustration only and never used for acceptance.

## 10. Quality analysis

**Provider precision.** Deviation of each vertex from its neighbours' midpoint, 10 matches:

| Positions | Median | p90 | p99 |
|---|---|---|---|
| VISIBLE | 5–7 mm | 1.9–3.4 cm | 15–22 cm |
| ESTIMATED | 3.7–5.1 cm | 10–15 cm | 23–42 cm |

This measures local smoothness, not absolute accuracy: there is no ground truth. Coordinates are integer mm, updated at about 15 Hz (median moving segment 0.0667 s).

**ESTIMATED positions are over-represented in encounters.**

- 56–70 % of samples are ESTIMATED.
- A random same-time player pair is est/est in 42–58 % of cases.
- 0.25 m encounters are est/est in **74 %** of cases (72 % excluding coincident intervals).

The provider's estimates place players closer together than observed tracking does. ESTIMATED-only encounters must therefore be analysed separately and **never pooled silently**.

**Gaps and interpolation.**

- No 0.25 m episode spans unobserved samples: the provider fills gaps with ESTIMATED positions instead of leaving holes.
- 607 episodes are cut at a support end: a long gap, an implausible jump, a period edge, or a substitution or attribution problem.
- 13 episodes involve a held position.

**Other flags on the non-coincident 0.25 m episodes:**

| Flag | Episodes |
|---|---|
| Duplicate provider position | 255 |
| Overspeed step in the episode | 390 |
| Implausible-jump landing | 148 |
| Substitution window | 6 |

All flags are kept per player in each record.

**Artifacts.**

- 551 coincident intervals are cloned estimated tracks (§6).
- Equal-x lattice ties produce the 15 exact points.
- Isolated contacts shorter than one step are interpolation inferences.

## 11. Dataset structure (per match, `m11-sync-proximity-v1/<match>/`)

| File | Contents |
|---|---|
| `root.events-0.25m.jsonl` | Primary: one record per episode (fields below) |
| `root.events-{0,0.1,0.5}m.jsonl` | Sensitivity episodes with the same fields |
| `root.trajectories.npz` | For 0.25 m only; arrays described below |
| `root.summary.json` | Configuration, windows, comparison counts, per-threshold statistics, historical reconciliation (18 exact events, 13,786 replica, catalogue counts) |
| `root.manifest.json` | Code and output hashes, `raw_unchanged`, split, configuration |

**Fields of each episode record:**

- **Identifiers:** `event_id` (sha256 of match | period | slots | R | start µs), `pair_id`, match, period, threshold.
- **Players:** for each of A and B, slot, side, team_id, jersey, player_id, `status_at_t_star`, `estimated_share`, `quality_flags`, `spans_missing_samples`, `held_position`, `duplicate_provider_position`, `segment_at_t_star` (raw rows, frames, t, x, y, visibility) and `position_at_t_star`.
- **Timing and geometry:** `t_start_s`, `t_end_s`, `duration_s`, `t_star_s`, `d_min_m`, `midpoint`.
- **Classification:** `contact_class`, `sample_instants_within`, `exact_coincidence`, `coincident_interval`, `pair_quality`, `support_cut_at_start` / `support_cut_at_end`, `synchronous_only`.
- **Windows:** `windows_game_s`, `canonical_window_game_s`, period bounds.
- **Nesting:** `parent_event_id` and `children`.

**Arrays in `root.trajectories.npz`:**

- `sample_*`: the original samples of both players in the canonical 5 s window, on the valid run containing the encounter. Columns are side, t, x, y, source row, frame, visibility, quality bits, and `before_t_star`, which splits before from after.
- `sync_*`: the synchronized grid (union of both players' sample instants plus start, t* and end). Columns are t, A(t), B(t), d(t), plus `a_is_sample` / `b_is_sample` masks; false means interpolated.

The dataset contains no labels, no swaps and no synthetic occlusions.

## 12. Limitations (what cannot be inferred without video)

- Whether a camera occlusion actually happened. Proximity in pitch coordinates is a proxy.
- Whether ESTIMATED positions are where the players were. 74 % of encounters rely on them, and the cloned tracks show they can be fabricated.
- True identity. Slots are the provider's identity, not verified.
- Motion between 15 Hz samples. Isolated contacts depend on linear interpolation.
- Body extent. 0.25 m between centre points is body contact or overlap in reality; whether it occludes depends on the camera geometry.

## 13. Recommendation on 0.25 m

0.25 m is appropriate as the primary threshold, under two conditions:

- the analysis is stratified by `pair_quality`;
- `coincident_interval` (cloned estimates) is excluded from any training population.

The reasons:

1. **For VISIBLE pairs, both thresholds are resolvable.** Position noise is at the cm level (p90 2–3 cm), so 0.10 m and 0.25 m are both measurable, and 0.25 m keeps 2.7× more observed/observed events (829 vs 311).
2. **For ESTIMATED pairs, 0.10 m is inside the noise** (p90 10–15 cm). A smaller threshold therefore does not mean more accurate events there.
3. **Neither threshold is a natural break in the data.** d_min is close to uniform below 0.5 m.
4. **0.25 m is close to body contact** between two players' centre points, which matches the occlusion intent.

0.10 m is better only as a high-precision subset of observed/observed episodes. The primary threshold was not changed.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m football_fatigue.intersections.proximity --matches 3812 3814 3815 3816 3829 3833 3836 3847 3851 10506
.\.venv\Scripts\python.exe -m football_fatigue.intersections.proximity --matches 3812 3814 3815 3816 3829 3833 3836 3847 3851 10506 --cross-match-only
.\.venv\Scripts\python.exe -m football_fatigue.intersections.proximity --matches 3812 --examples --catalogue-event 751439e3b92975d5
.\.venv\Scripts\python.exe -m pytest tests/test_intersections_proximity.py -q
```

Tests: 24 proximity tests, covering every case in the brief plus the dense-reference and same-timestamp checks. All intersection suites together: **112 passed**.
