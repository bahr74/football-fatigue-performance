# M11: Four-method local-descriptor pilot (F1–F4), FIFA development matches 3812 and 3816

Status: **pilot complete. All 40 development matches are not yet run, and no model was trained.** Written 2026-11-13.

**Scope:**

- Development matches 3812 and 3816 only. No validation or test match was read, and RAW data is unchanged (`raw_unchanged: true`).
- The v1 window-intersection events were reused read-only. Their SHA-256 values were checked against the v1 manifest.
- The sliding-window scan was not repeated.
- Labels: **[V]** verified by computation, **[I]** inference.

## 1. Code, outputs, how to reproduce

| Item | Path |
|---|---|
| Descriptors, methods, checks, CLI | `src/football_fatigue/window_intersections/local.py` |
| Figures | `src/football_fatigue/window_intersections/local_figures.py` |
| Unit tests (9, synthetic, closed-form) | `tests/test_window_local.py` |
| **Pilot output (valid)** | `data/processed/m11-window-local-methods-pilot-v2/` |
| Superseded pilot output | `data/processed/m11-window-local-methods-pilot-v1/` (see §8) |
| Visual index | `data/processed/m11-window-local-methods-pilot-v2/visual_index.html` |

The output directory contains:

- `descriptors/<match>/descriptors_{linear,pchip}.parquet`: one row per v1 event, keyed by `event_id`;
- `descriptors/<match>/method_masks_{linear,pchip}.parquet`: 210 method configurations;
- `counts_grid.json`, `counts_primary.json`, `analysis.json` (overlap, stratified counts, redundancy, guard effect, linear vs PCHIP), `checks.json`, `runtime.json`, `config.json`;
- `figures/` and `manifest.json`.

To reproduce:

```
python -m football_fatigue.window_intersections.local --matches 3812 3816 --out <new directory name>
```

The CLI refuses non-development matches and refuses existing output directories.

## 2. Definitions as implemented

The selected intersection is P. Each player has an arrival time at P (tA, tB). The earlier one is t−, the later one is t+, and d(t) = |A(t) − B(t)| is the synchronized distance.

**Geometry**

- Positions are piecewise linear in time. For linear, the knots are the native 15 Hz vertices. For PCHIP, they are the 10 sub-points per native segment, exactly the polyline on which v1 found the intersections.
- Each player is used only on its continuous valid chain at the intersection. Nothing is bridged.

**Methods** (all four require the v1 geometric intersection):

| Method | Condition |
|---|---|
| F1(τ) | \|tA − tB\| ≤ τ. Identical to v1 A (checked). |
| F2(nb, h, R[, guard]) | Exact minimum of d(t) over the neighbourhood is ≤ R. |
| F3 | F1(τ) **and** F2. |
| F4(ε, τ) | Each player's occupancy interval around its own arrival comes within τ of the other's: gap = max(0, later start − earlier end) ≤ τ. |

**F2 neighbourhoods**

- **arr:** [t− − h, t+ + h].
- **pt:** [tA − h, tA + h] ∪ [tB − h, tB + h].
- Both are clipped to the span where both chains exist.

**F2 guard:** the pair midpoint at the minimum time t\* lies within ρ = 1.0 m of P.

**F4 occupancy:** the maximal interval during which the player stays within ε of P, found exactly per piece and searched at most 2.5 s either side.

**The two F4 interpretations**

| Interpretation | ε |
|---|---|
| Fixed physical region | 0.05, 0.10 or 0.25 m |
| Measurement uncertainty ("tier") | per player: 0.05 m if VISIBLE at the intersection, 0.10 m if ESTIMATED |

The tier values come from the measured jitter: VISIBLE about 1 cm, ESTIMATED about 8–10 cm, which is a lower bound because the absolute error is unknown. As ε → 0, F4 becomes F1.

**Grid**

- τ ∈ {0.10, 0.20, 0.30} s.
- R ∈ {0.25, plus 0.50 and 1.00 as sensitivities} m.
- ε ∈ {0.05, 0.10, 0.25} m, plus the tier version.
- h ∈ {0, 0.10, 0.25, 0.50} s.
- Guard on or off.

## 3. Integrity checks [V]

All checks pass on both matches and both variants (`checks.json`):

- **Every event placed back on P:**
  - linear: at most 7×10⁻¹² m;
  - PCHIP: at most 9.5×10⁻⁷ m, within the v1 geometric tolerance of 10⁻⁶ m.
- **F1 equals v1 `a_tau_*`.**
- **The minimum never increases as h grows.**
- **arr ≤ pt** for the same h.
- **arr(h=0) is at most both arrival distances,** and pt(h=0) equals the smaller of the two.
- **The F4 gap** never increases with ε and never exceeds \|dt\|.
- **F3 = F1 ∧ F2** for all 144 F3 configurations.
- **Independent dense check:** 400 random events per match and variant.
  - Distance: the sampled minimum matches the closed form within one sampling step.
  - Occupancy end points: on a 1 ms grid plus the knots, within 1 ms.
- **Unit tests:** 9 tests with closed-form answers (perpendicular crossing, time within R, slow vs fast, chain break, grazing exit, standing player, random tracks vs dense sampling, mask logic). With the 8 v1 window tests: **17 passed.**

## 4. Results, both matches, linear reference [V]

Primary events in scope: 23,482 (3812: 11,882; 3816: 11,600).

| Configuration | Events | Accepted (= sample-ready) | Both VISIBLE / mixed / both ESTIMATED | Median \|dt\| (s) | Median slower speed (m/s) | Windows with > 1 intersection |
|---|---|---|---|---|---|---|
| F1 τ 0.20 | 956 | 671 | 289 / 254 / 413 | 0.10 | 3.06 | 478 |
| **F1 τ 0.30** | **1,511** | 1,073 | 505 / 373 / 633 | 0.16 | 3.04 | 757 |
| F2 arr h0.25 R0.25 | 1,730 | 1,441 | 294 / 427 / 1,009 | 0.57 | 1.60 | 1,313 |
| **F2 arr h0.25 R0.25 + guard** | **1,564** | 1,317 | 247 / 364 / 953 | 0.45 | 1.55 | 1,159 |
| F2 pt h0.25 R0.25 | 1,505 | 1,235 | 258 / 361 / 886 | 0.37 | 1.64 | 1,105 |
| F2 pt h0.25 R0.25 + guard | 1,451 | 1,209 | 243 / 339 / 869 | 0.35 | 1.62 | 1,056 |
| F3 τ 0.20, arr + guard | 566 | 420 | 138 / 163 / 265 | 0.06 | 2.25 | 294 |
| **F3 τ 0.30, arr + guard** | **692** | 527 | 154 / 194 / 344 | 0.08 | 2.08 | 380 |
| F3 τ 0.30, arr (no guard) = F3 τ 0.30, pt | 705 | 532 | 158 / 199 / 348 | 0.09 | 2.11 | 391 |
| F4 ε 0.05, τ 0.30 | 1,699 | 1,237 | 560 / 418 / 721 | 0.18 | 2.88 | 868 |
| F4 ε 0.10, τ 0.30 | 1,933 | 1,436 | 628 / 466 / 839 | 0.20 | 2.76 | 1,022 |
| **F4 ε 0.25, τ 0.30** | **2,878** | 2,245 | 919 / 641 / 1,318 | 0.29 | 2.38 | 1,619 |
| F4 tier, τ 0.30 | 1,838 | 1,361 | 560 / 439 / 839 | 0.19 | 2.76 | 972 |

**Speeds above 10 m/s** on the intersection segment, which are likely artifacts: 192 events in F1 τ 0.30 (12.7 %), 117 in F2 + guard, 65 in F3 and 253 in F4 ε 0.25.

**Sensitivity grids** (`counts_grid.json`):

| Setting | arr | arr + guard | pt | pt + guard |
|---|---|---|---|---|
| F2 R0.25, h = 0 / 0.10 / 0.25 / 0.50 s | 1,454 / 1,605 / 1,730 / 1,873 | 1,325 / 1,473 / 1,564 / 1,612 | 891 / 1,259 / 1,505 / 1,722 | 891 / 1,256 / 1,451 / 1,544 |
| F2 h0.25, R = 0.25 / 0.50 / 1.00 m | 1,730 / 3,065 / 5,906 | 1,564 / 2,684 / 4,782 | 1,505 / 2,721 / 5,398 | 1,451 / 2,558 / 4,711 |

| F4 events by τ (0.10 / 0.20 / 0.30 s) | Events |
|---|---|
| ε 0.05 | 629 / 1,163 / 1,699 |
| ε 0.10 | 859 / 1,393 / 1,933 |
| ε 0.25 | 1,598 / 2,232 / 2,878 |
| tier | 784 / 1,308 / 1,838 |

**Exclusive overlap** (F1 τ 0.30, F2 arr + guard, F2 pt, F4 ε 0.25):

| Combination | Events |
|---|---|
| All four | 692 |
| F1 and F4 only | 806 |
| F4 only | 987 |
| Both F2 variants only | 388 |
| Both F2 variants and F4 | 366 |
| F2 arr only | 107 |
| F2 pt only | 43 |
| Other combinations | ≤ 13 each |

## 5. Findings

### 5.1 The spatial condition is not redundant with F1 [V]

- Only 46 % of the F1 events at τ 0.30 also pass F2; 59 % at τ 0.20 and 85 % at τ 0.10.
- **The pass share falls steadily with speed** (slower player at P, τ 0.30):

| Speed (m/s) | < 1 | 1–2 | 2–4 | 4–7 | ≥ 7 |
|---|---|---|---|---|---|
| Share of F1 that passes F2 | 0.93 | 0.73 | 0.41 | 0.26 | 0.13 |

- **The region where F1 guarantees F2 is tiny.** Straight-line motion would make F1(τ) imply F2(R) only when the slower speed is at most R/τ (0.83 m/s at τ 0.30). Only 71 of the 1,511 F1 events are that slow, and 67 of them pass F2. The other 4 fail because the trajectories are not straight.
- **The straight-line bound itself is only approximate:** d(arr, h 0) ≤ v_min·\|dt\| + 1 cm holds for 76 % of all events.
- **F3 therefore acts as F1 with a time tolerance that adapts to speed,** roughly \|dt\| ≲ R / v. Its median \|dt\| is 0.08 s, against 0.16 s for F1.
- **F3 removes mainly fast crossings.** These are real crossings, but the players are never close at the same moment. Gallery example: VISIBLE players at 4.7 m/s, \|dt\| = 0.19 s, local minimum 0.74 m.

### 5.2 F2 is not redundant with F1 either [V]

- 872 of the 1,564 F2 + guard events fail F1 at τ 0.30 (median \|dt\| 0.45 s).
- These are slow, standing or weaving pairs. 74 % of F2 events come from windows with several intersections, against 50 % for F1.
- F2 is also dominated by both-ESTIMATED events (61 %), against 42 % for F1. **F2 therefore has the weakest labels.**

### 5.3 The neighbourhood definition matters for F2, not for F3 [V]

- **For F2:** arr finds 1,730 events and pt finds 1,505 at h 0.25.
  - pt at h = 0 uses only the two arrival instants: 891 events.
  - arr is more inclusive because it contains the closest approach *between* the two arrivals. That is where the true minimum lies for crossing motion (the closed-form test case).
- **For F3, the choice is immaterial:** arr and pt both give 705 events at τ 0.30, because \|dt\| ≤ 0.3 s keeps both neighbourhoods short.
- **h** changes F2 by about ±10 % from h 0.10 to h 0.50. **R** changes F2 strongly: 0.25 → 0.50 → 1.00 m gives 1,730 → 3,065 → 5,906 events. It is the dominant parameter.

### 5.4 The 1.0 m guard matters only for F2 with the arr neighbourhood and a large |dt| [V]

| | h 0 | h 0.10 | h 0.25 | h 0.50 |
|---|---|---|---|---|
| arr: events removed by the guard | 129 | 132 | 166 | 261 |
| of which with \|dt\| > 0.30 s | 129 | 132 | 153 | 226 |
| removed inside F3 | 0 | 0 | 13 | 35 |

- For pt, the guard removes 0 at h 0, 3 at h 0.10 and 54 at h 0.25.
- **The guard has two sources:**
  - long arrival intervals, where a closest approach elsewhere enters;
  - inside F3, only the margin h. With h ≤ 0.10 s it removes nothing.
- **The removed events are genuinely different encounters.** Gallery example: \|dt\| = 3.77 s, minimum 0.19 m, but 3.49 m from P.
- **Conclusion:** the guard is necessary for F2 with arr, harmless for F3, and unnecessary for pt with h ≤ 0.10 s.

### 5.5 F4: the two interpretations behave very differently [V]

**Measurement uncertainty** (ε 0.05 / 0.10 / tier) adds 188 / 422 / 327 events to F1 at τ 0.30 (+12 % / +28 % / +22 %). These are mostly events whose \|dt\| lies just above τ. They stay close to F1 in composition: tier mix, speed (median 2.8 m/s) and \|dt\| (median 0.18–0.20 s).

**Fixed physical region** (ε 0.25) adds 1,367 events (+90 %). **987 events are in F4 only:** they pass neither F1 nor F2.

- These are pairs that both pass through the 0.25 m disc around P within 0.3 s of each other, without arriving within 0.3 s at P itself, and without being within 0.25 m of each other at the same moment.
- Gallery example: \|dt\| = 0.43 s, local minimum 0.57 m.

**Which interpretation is scientifically meaningful?** [I]

- As **measurement uncertainty**, F4 is a defensible correction of F1: it does not reject a crossing because of position noise of the order of the jitter.
  - The tier values are a lower bound on ESTIMATED error, not a calibrated uncertainty.
- As a **fixed physical region**, F4 is a third, looser proximity concept: "both near the same spot at about the same time". It neither strengthens the crossing evidence nor implies same-moment closeness.
  - Its 0.25 m disc makes the players at most about 0.5 m plus the speed-gap term apart.
  - It is the largest pool and has the highest share of VISIBLE events (32 %).

**Occupancy caveats**

- **Long occupancy is truncated by the 2.5 s cap:** 540 events at ε 0.25 (75 at ε 0.05) involve a player who stays inside the disc that long, usually a standing player.
- **The interval is the exact connected component.** A path that grazes out of the disc by 0.2 mm ends the interval there (unit test "grazing exit"; found by the dense check, §8). This makes F4 sensitive to sub-millimetre geometry at ε edges.
  - A tolerance (for example, merging re-entries within one native step) would need a decision.

### 5.6 Linear vs PCHIP [V]

- Membership flips between the two variants are 0.7–4.8 % of the linear events in every primary configuration. The largest is F3 τ 0.20, with 27 of 566.
- PCHIP passes 0–2 % more events.
- **Linear is a sound reference.** PCHIP changes no conclusion.

## 6. Runtime [V] (sequential, this machine)

| Step | Per match |
|---|---|
| Load the Step 1 cache and rebuild segments | 7–8 s |
| Descriptors, linear | 28 s |
| Descriptors, PCHIP | 38 s |
| Checks (400-event dense check) | about 1 s |
| **Total per match** | **about 75 s** |

- Analysis takes about 1 s, and the figures for two matches about 40 s.
- Total pilot: **190 s**.
- **Estimate for all 40 development matches** [I]: about 50 min sequential, plus figures. Linear only would be about 25 min. The v1 scan took about 30 min.

## 7. Scientific limitations and risks

1. **Label leakage.** Every method selects by a crossing computed from identity-labelled tracks.
   - Earlier M11 work found that "assume they crossed" alone reaches about 75 %, and an advisor-population GBM was at chance on non-crossings.
   - None of F1–F4 is a fair training or evaluation population on its own.
   - **Matched non-crossing controls are required:** close approaches without a crossing, matched on local distance, speed and tier. They are documented here and **not built**.
2. **Weak labels where proximity is high.** 61 % of F2 events and 50 % of F3 events have both players ESTIMATED at P, so the provider identity there is itself an estimate.
3. **Speed** is taken from one 15 Hz segment. It is noisy for ESTIMATED tracks, and 7.5–12.9 % of events per method exceed 10 m/s.
4. **Proximity is not occlusion.** The earlier SoccerTrack study found image occlusion at a median ground distance of 1.7–2.9 m. R = 0.25 m selects extreme proximity.
5. **First-intersection rule.** All methods inherit the v1 primary-intersection selection. Weaving pairs, especially in F2, have several crossings per window, and only the first is evaluated.
6. **Two matches only.** These matches were also the v1 pilot and the earlier prototype, so the counts are not independent evidence. Counts must not be extrapolated beyond order of magnitude.
7. **Availability.** The F2 neighbourhood is unavailable for 167 events in 3812 and 59 in 3816 (linear), where the two chains do not overlap in time. These events can never pass F2 or F3.

## 8. Run notes

- The first pilot run (`-pilot-v1`) reported one failed independent check: the occupancy dense check on 3812 linear.
  - **Diagnosis:** a knot 0.10021 m from P (0.2 mm outside ε = 0.10 m) makes the player leave the disc for about 0.5 ms. The exact algorithm stops there correctly; the 1 ms grid skipped over it.
  - **Fix:** the check now also samples the knots. Distance to a point is convex on each linear piece, so an exit can only straddle a knot.
  - A unit test was added, and the pilot was re-run as `-pilot-v2`. All checks pass.
  - **All data files of v1 and v2 are byte-identical** (manifest SHA-256 values compared); only `checks.json` and the code hash differ.
  - `-pilot-v1` was not deleted: versioned outputs are never overwritten or removed without approval.
- `local.py` keeps the shared v1 module and `core.py` unchanged. The v1 datasets are untouched.

## 9. Recommendation [I]

1. **Keep all four methods for the full run, defined as:**
   - F1 at τ 0.20 and 0.30;
   - F2 with arr, h 0.25 and the guard (pt reported as a sensitivity);
   - F3 = F1 ∧ F2 (arr; the guard is immaterial there);
   - F4 in both interpretations, reported separately and not merged: tier for measurement uncertainty, and ε 0.25 as a separate proximity concept.
2. **Run the full 40-match descriptor pass with the same code**, into a new versioned directory. Use linear plus PCHIP, or linear only to halve the runtime.
3. **Before any training design,** specify the matched non-crossing controls and the common held-out population.
