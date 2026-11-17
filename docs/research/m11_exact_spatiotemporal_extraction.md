# M11 — Exact spatiotemporal occlusion extraction (advisor's instructions revisited)

Schema `m11-exact-spatiotemporal-v1`, code `src/football_fatigue/intersections/exact.py`, tests
`tests/test_intersections_exact.py`. Ten development matches only (3812, 3814, 3815, 3816, 3829, 3833, 3836, 3847, 3851,
10506). No validation or test match was opened. No model was trained. Raw files and earlier outputs were not changed.
Nothing is committed.

## 1. Compliance audit: the advisor's words vs. later additions

The advisor's material is in the repository: `data/raw/Screenshot 2026-11-09 175508.png` (diagram) and
`data/raw/Screenshot 2026-11-09 175517.png` (code). Both were re-read for this audit, together with his messages.

### What he wrote

| Advisor's instruction (quoted or paraphrased from his text, diagram or code) | Status in this rebuild |
|---|---|
| "for each 1second ... 2.5 seconds before ... 2.5 seconds after" | Implemented. Windows are centred on every integer game second from the period's first frame. Partial windows at period edges are included. |
| "4 seconds trajectory" | His arithmetic slip: ±2.5 s gives 5 s. 5 s is used. |
| "22 players", "all possible couples" | Implemented. 22·21/2 = **231** pairs (he wrote 121). |
| "121x60x90 = 653,400" | With 231 pairs: 231 × 5,400 = **1,247,400** pair-windows per 90 min. Measured on 3812: 1,425,655, because the match is longer than 90 min and substitutes add slots. |
| "Keep only the trajectories that intersect" | Implemented. Intersection here means the exact solution of A(t*) = B(t*); see the next row. |
| "The occ point here is one point at one time ... This time will be exactly the same for both players" | This is the requirement A(t*) = B(t*), solved exactly. |
| "don't try to estimate it yourself ... curves intersection algorithm" | Implemented. The solve is exact: integer µs/mm with `Fraction` verification. Nothing is estimated or fitted. |
| "much robust than ... same point or in close points within x-meter distance" | Implemented. No distance threshold is used. |
| "cleaning from repeated ones (... trajectories 1 to 4 and also 2 to 5)" | Implemented. Duplicates are removed by an exact key (pair, t*, P). |
| Diagram: Occlusion Point, A_Before, A_After, B_After | Implemented. Both local trajectories are exported before and after P, with P and t*. |
| Code: `LineString(A).intersection(LineString(B))` on (x, y), then `project`/`searchsorted` | **Inconsistent with his text.** The code is 2D only and does not include time, so it returns path crossings where each player passes P at a different time. This was the root of the earlier divergence (§2). |

### What the AI added earlier (not from the advisor)

None of these are used in v1 of this schema:

- 0.20 s τ;
- 1 m contact radius;
- h = 0.0667 s with a 2σ band;
- spike / quality vetoes;
- SG5 smoothing;
- the v1 duplicate-position exclusion;
- the |Δt| > 3 s cut.

## 2. Why the previous pipelines diverged, and why 284 was small

| Candidate cause | Finding |
|---|---|
| Use of the advisor's 2D code | **Main cause.** v1/v3 and the catalogue reproduce his code exactly: 166,819 crossings. Each player reaches P at its own time (median \|Δt\| 2.16 s). These are spatial crossings, not simultaneous occupancy. |
| Spatiotemporal v1 tolerance and rules | The 284 accepted events came from crossings with \|Δt\| ≤ 2σ ≤ 0.0667 s plus quality rules. That count depends on thresholds we set: across the grid it ranges from 0 to 1,454. |
| v1 duplicate-position rule | It removed bit-identical slot positions. That rule excludes **exactly** the coincident events: 551 of the 551 intervals. |
| Trajectory construction | Verified. Segments join provider update instants linearly. Held positions (runs of 3 or more repeated frames) are now represented as stationary and then moving, instead of being smeared. Trajectories do not join across gaps over 0.5 s, period changes, unusable states or implausible jumps. |
| Time alignment | Verified. `videoTimeMs`/1000 has float noise of at most 9.1e-13 s, and the integer µs value is recovered exactly. Update instants are synchronised across players: 92,520 distinct t0 for 1,999,746 segments, with mixed frames at 0.25 %. |
| Interpolation | Linear between updates, as the provider samples. No smoothing. |
| Fragmentation, windows | Windows only label events; they do not create or remove solutions. Brute force over every window gives the same unique set (test). |
| Deduplication | Exact key. Distinct events of the same pair are never merged (test). |
| Provider interpretation | Coordinates are integer mm (all 4,067,102 values). Same-time equality is therefore a lattice event. |
| **Genuine spatial-vs-simultaneous difference** | **Decisive.** A(t) = B(t) is two equations (x and y) in one unknown t. For generic motion it has no solution. Two paths cross in the plane all the time, but two players occupy the same point at the same instant only by coincidence of measure zero. |

## 3. The algorithm

For every pair (A, B), on every common time interval where both players have a valid linear segment, the code defines
D(t) = A(t) − B(t), which is linear. With D0 and D1 the values of D at the two ends of the common interval:

- D0 = D1 = 0: coincident motion, reported as an **interval**.
- Exactly one of them is zero: a **point at a sample instant**.
- cross(D0, D1) = 0 and D0·D1 < 0: an **interior point**, at s* = D0·(D0 − D1)/|D0 − D1|². P is A(t*), which equals B(t*).
- Otherwise there is no solution.

**Arithmetic.** Positions are integer mm and times integer µs. Intervals with synchronous end points are tested with
integers. Partial overlaps are pre-screened in floating point (1e-9) and then verified exactly with `Fraction`. No physical
tolerance enters the decision. The float screen only selects candidates, and every reported event is exact.

**Windows and deduplication.**

- A solution belongs to every window centred at c with |t* − c| ≤ 2.5 s.
- Window-level detections are counted, then reduced to unique events:
  - points by (pair, t*, P);
  - coincidences by maximal contiguous intervals;
  - end-point solutions that touch an interval are absorbed into it.

**Export.** Each event carries:

- identities, team, jersey, team relation;
- t* (exact µs) and P (exact mm);
- source rows, frames, visibility and quality bits of both segments;
- the windows that contain it and the canonical window;
- before/after trajectories of both players ±2.5 s with P inserted.

## 4. Synthetic tests (`tests/test_intersections_exact.py`, 14 tests)

The tests cover:

- an exact simultaneous crossing (t* = 104.970000 s; windows 3–7; canonical 5);
- a spatial crossing at different times: 0 events, although the 2D v1 code finds 1;
- the same time at different positions: 0;
- a stationary player;
- a shared sample point (merged from 2 detections; the v1 rule would have excluded it);
- parallel and coincident paths (interval 104–106 s);
- multiple intersections: the count equals the number of sign changes;
- overlapping windows, checked against an independent brute force over every window, pair and segment pair;
- missing data: a 1.1 s gap gives 0 and a 0.4 s gap gives 1;
- different sampling instants (non-synchronous) giving the exact t*;
- repeated encounters of one pair kept as separate events;
- export content;
- `windows_of`;
- the written 3812 output.

**Run:** all intersection suites (exact, spatiotemporal, catalogue, advisor, intersections, v3, audit) gave
**88 passed**.

## 5. Results, match 3812

| Stage | Count |
|---|---|
| Windows | 6,169 |
| Pair-window comparisons | 1,425,655 |
| Windows × 231 (theoretical) | 1,425,039 |
| Pair-windows with a 2D crossing (advisor code) | 26,662 |
| 2D crossings, window-level / unique | 52,155 / 18,469 |
| Synchronous pair-intervals solved | 21,050,439 |
| Pair-intervals with an exact solution | 12,088 |
| Window-level exact detections | 880 |
| Duplicates removed | 862 |
| **Unique exact events** | **18** |
| — interior points | 1 |
| — coincident intervals | 17 (805 s in total) |

The one interior point is `3dc953ace4a43656`: #16 × #3 (home), t* = 3352.628446 s. Its figure shows both players
converging on the same point at the same instant.

## 6. Results, 10 development matches

| Match | Windows | Comparisons | 2D unique crossings | Exact solutions | Window-level | Unique | Points | Intervals |
|---|---|---|---|---|---|---|---|---|
| 3812 | 6,169 | 1,425,655 | 18,469 | 12,088 | 880 | 18 | 1 | 17 |
| 3814 | 6,022 | 1,391,588 | 11,894 | 38,255 | 2,938 | 84 | 1 | 83 |
| 3815 | 6,271 | 1,449,261 | 15,685 | 10,309 | 1,205 | 106 | 0 | 106 |
| 3816 | 6,612 | 1,528,076 | 18,305 | 4,693 | 452 | 30 | 1 | 29 |
| 3829 | 6,186 | 1,429,739 | 19,648 | 12,846 | 927 | 15 | 0 | 15 |
| 3833 | 6,427 | 1,485,143 | 15,661 | 22,366 | 1,745 | 52 | 0 | 52 |
| 3836 | 5,838 | 1,349,150 | 15,067 | 15,369 | 1,703 | 138 | 1 | 137 |
| 3847 | 6,418 | 1,483,199 | 18,825 | 10,240 | 819 | 28 | 1 | 27 |
| 3851 | 6,231 | 1,439,955 | 19,913 | 9,669 | 888 | 53 | 6 | 47 |
| 10506 | 5,799 | 1,340,097 | 13,352 | 3,636 | 447 | 42 | 4 | 38 |
| **Total** | **61,973** | **14,321,863** | **166,819** | **139,471** | **12,004** | **566** | **15** | **551** |

Every per-match field, including the near-meeting and perturbation counts, is in `cross-match/summary.json`.

## 7. Dataset

All outputs are under `data/processed/m11-exact-spatiotemporal-v1/`. Each match has its own directory, `<match>/`, with:

- `root.events.jsonl`: one record per unique event;
- `root.trajectories.npz`: before/after samples of both players with P;
- `root.summary.json`: per-stage counts;
- `root.manifest.json`: source hashes and parameters;
- `root.fig-*.png`: figures.

`cross-match/summary.json` aggregates all matches. `contrast-3812/` holds the 2D-crossing contrast figures.

The writer never overwrites existing outputs and refuses non-development matches.

## 8. Per-stage counts (10 matches)

| Stage | Count |
|---|---|
| pair-window comparisons | 14,321,863 (theoretical 231 × windows: 14,315,763) |
| 2D spatial crossings (advisor code), window-level / unique | 468,859 / 166,819 |
| synchronous pair-intervals examined | 212,653,306 |
| exact same-time, same-position solutions (pair-intervals) | 139,471 |
| window-level exact detections | 12,004 |
| duplicates removed | 11,438 |
| **final unique events** | **566** = 15 interior points + 0 sample-instant points + 551 coincident intervals |

## 9. Visual and scientific explanation

**Figures.**

- `3812/root.fig-point-3dc953ace4a43656.png` shows a true exact event: the markers at the same instants converge to P.
- `3812/root.fig-interval-1803eb7e8a5b7a57.png` shows #6 and #8 (home) with identical coordinates for 2.2 s.
- `contrast-3812/fig-751439e3b92975d5.png` shows a 2D crossing from the advisor's code: #3 reaches P at 6219.185 s and #10
  at 6221.288 s (Δt = +2.10 s). At every common instant the two players are apart; the dotted lines join same-time
  positions.

**What the 566 events are.**

- **551 coincident intervals.**
  - All are same-team pairs with ESTIMATED positions.
  - Median duration 8.98 s, maximum 249 s, 9,279.7 s in total.
  - The provider writes the same coordinates into two slots, so these are **artifacts, not occlusions**.
- **15 interior points.**
  - They are exactly the 15 numerical ties of the 2D catalogue (set equality verified).
  - All are same-team, and all have identical x at both ends of the interval (D_x ≡ 0) while y changes sign. This is a
    provider-structured pattern; many involve goalkeeper #1 at |x| 39–52 m.
  - Equal-x pair-intervals are frequent: 1,310 in 3812 and 1,161 in 3851, nearly all same-team.
  - 11 of the 15 are VISIBLE and 4 ESTIMATED.

**Perturbation test.** Jittering every coordinate uniformly by ±0.5 mm (below the provider's resolution) leaves **0** exact
solutions in all 10 matches. The exact events therefore depend entirely on lattice and duplicate structure, not on player
motion.

**Context (descriptive only, not a selection).** Same-time near-meeting episodes in the 10 matches, by closest approach:

| Closest approach | Episodes |
|---|---|
| ≤ 1 mm | 616 |
| ≤ 1 cm | 912 |
| ≤ 10 cm | 3,159 |
| ≤ 50 cm | 13,786 |

Physical meetings are frequent, but exact equality is not.

**Answer to the expectation of many events.** Under A(t*) = B(t*) with no tolerance, the data contain essentially no
physical simultaneous crossings. The count is not small because of a bug or a filter: the tests and the brute-force
equivalence rule that out. It is small because the condition is non-generic. The 2D crossings (166,819) satisfy the
advisor's code but not his "same time" sentence. The two cannot both hold without a tolerance, and the requirement
forbids one. The requirement was not relaxed.

## 10. Full-dataset processing and training plan

1. **Remaining development matches (30).**
   - Build Step 1 caches first; only 10 of the 40 have them.
   - Then run `python -m football_fatigue.intersections.exact --matches <ids>`.
   - Processing is deterministic and per match. A match that already has outputs is skipped, which acts as the checkpoint.
   - Then run `--cross-match-only`.
2. **Validation (12) and test (12).** Not accessed. **Authorization is required** before either is processed.
3. **Advisor decision needed before training.** The exact rule yields 15 artifact-structured points and 551 duplicate
   intervals, so it cannot supply a training set. Options to put to Dr. Basha (none is adopted here):
   - (a) keep the 2D crossings, i.e. his code, as the occlusion events;
   - (b) state a same-time tolerance himself;
   - (c) define occlusion by same-time proximity.
4. **Training.** Only after (3) is decided, using development matches with leave-one-match-out, a label-symmetric
   counterpart, and validation once.

**Reproduction:**

```powershell
.\.venv\Scripts\python.exe -m football_fatigue.intersections.exact --matches 3812 3814 3815 3816 3829 3833 3836 3847 3851 10506
.\.venv\Scripts\python.exe -m football_fatigue.intersections.exact --matches 3812 --cross-match-only
.\.venv\Scripts\python.exe -m football_fatigue.intersections.exact --matches 3812 --contrast 751439e3b92975d5 e699a1a7e0b85335
.\.venv\Scripts\python.exe -m pytest tests/test_intersections_exact.py -q
```
