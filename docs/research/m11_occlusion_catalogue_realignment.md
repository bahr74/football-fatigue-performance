# M11 — Re-alignment with Dr. Basha's occlusion extraction procedure

Date 2026-11-09. Development matches only (3812, 3814, 3815, 3816, 3829, 3833, 3836, 3847, 3851, 10506). No validation or test
match was opened, no model was trained, and no historical artifact was changed or deleted. Nothing is committed.

- New output: `data/processed/m11-occlusion-catalogue-v1/<match>/` and `.../cross-match/summary.json`.
- New code: `src/football_fatigue/intersections/catalogue.py`, plus a small refactor in `advisor.py` (`run_bounds`).
- New tests: `tests/test_intersections_catalogue.py`.

## 1. What Dr. Basha requested

His messages, diagram and Shapely code describe one procedure:

1. For every second of play, take each player's trajectory from 2.5 s before to 2.5 s after.
2. Pair the trajectories of the players on the pitch in all unique pairs.
3. Give each pair to a curve-intersection function (`LineString.intersection`).
4. Discard pairs without an intersection. Keep the intersecting pairs: these are the occlusion dataset.
5. Take the exact intersection point P returned by the algorithm; do not estimate it.
6. Find P's segment on each trajectory (`project`, cumulative lengths, `searchsorted`) and from it P's time.
7. Remove the repeats caused by overlapping windows (a crossing at t = 3 appears in the windows 1–4 and 2–5).

The diagram splits each trajectory at P into Before and After segments. He states that the occlusion is "one point at one time"
and that the time "will be exactly the same for both players".

## 2. What the existing implementation already did correctly

The v1 detector (`intersections/detect.py` + `geometry.py`, reused unchanged by v3) already implements this procedure. Each
point below was checked in code and confirmed in the stored data.

| advisor step | existing implementation | verified |
|---|---|---|
| ±2.5 s windows | `window_s = 5.0`, a full window per centre | yes |
| every 1 s | `stride_s = 1.0`, per period from period start + 2.5 s | 6,159 windows in 3812 |
| all pairs | every unordered pair of players present in the window; the bounding box only skips pairs that cannot touch | 1,404,228 pair-windows in 3812 |
| curve intersection | exact segment-pair intersection (proper / vertex / collinear) on the RAW polylines; no distance threshold | point re-solved to 7e-15 m |
| exact P | the returned point is stored, not an average or a nearest point | — |
| segment and time of P | the segment index is known exactly; t = t0 + u·(t1 − t0) on that segment, from the source timestamps | equals the advisor recipe on 166,814 / 166,819 |
| no time filter | Δt is recorded, never used to drop an event | all 166,819 kept |
| window deduplication | one hit per segment pair over all windows; hits at a shared vertex merged | windows of every event equal the windows containing both segments (166,819 / 166,819) |
| repeated visits | the exact segment removes the ambiguity of `project` | 5 loop cases (below) |
| halftime | windows never cross a period | per period |
| substitutions / missing data | a player is in a window only where it has valid segments | 11–23 available players per window |

The catalogue rebuilt today from the RAW cache is byte-identical to the stored `m11-intersections-v3` events on all 10 matches.

## 3. What was misunderstood or filtered incorrectly

The geometric catalogue was never filtered. The problems are in the layers built **on top of** it:

- **0.20 s time threshold.** The v1 records carry a `simultaneous` flag (|Δt| ≤ 0.20 s). S1/S1.1 classified events as
  "time-aligned" at τ = 0.20 s, and the a1 experiment trained only on that 4.2 % subset. The threshold was ours, not the
  advisor's.
- **1 m contact threshold.** The contact catalogue (`m11-contacts-v1`) replaced the intersection with a distance rule. This
  redefines the task and is not his method. All its results stay historical.
- **v1 samples.** The four-segment samples excluded 5,657 events with |Δt| > 3 s (sample stage only; the events were kept).
- **The "4.2 %"** is a derived temporal subset (|Δt| ≤ 0.20 s) of the geometric catalogue. It is not the advisor's output.

In the new catalogue the `simultaneous` field is dropped. Timing is described without any chosen threshold (section 8).

**Validity conditions that are ours, not his.** These are data-quality rules, not occlusion filters, and they need his
confirmation. Step 1 does not join two samples into a segment when:

- the gap between them is > 0.5 s unobserved;
- the step is an implausible jump (> 12 m/s and > 1 m);
- either sample has unresolved attribution;
- either sample's position is bit-identical to another player's (a provider artifact).

A literal Shapely run on the raw sample sequence would join them. Sensitivity on 3812 (events):

| variant | events |
|---|---|
| as built | 18,469 |
| + implausible-jump steps joined | 22,144 (+3,675) |
| + duplicate-position samples joined | 63,118 (+44,649; two slots sharing one position intersect everywhere) |

## 4. Window and pair counts (corrected arithmetic)

- A window of ±2.5 s lasts **5 s**, not 4 s. We use 5 s, as instructed.
- 22 players give C(22, 2) = 22 × 21 / 2 = **231** pairs, not 121.
- A 90-minute match has 90 × 60 = 5,400 one-second centres, so 231 × 5,400 = **1,247,400** pair-window checks (not 653,400).
  These are comparisons, not intersections.

Actual counts:

| | 3812 | 10 development matches |
|---|---|---|
| playing time (both periods) | 102.8 min | 1,032.7 min |
| windows (1 s stride) | 6,159 | 61,873 |
| 231 × windows | 1,422,729 | 14,292,663 |
| pair-windows actually checked (players present) | 1,404,228 | 14,059,734 |
| pair-windows with ≥ 1 intersection | 26,662 (1.9 %) | 239,218 (1.7 %) |
| intersection hits counted per window | 52,155 | 468,859 |
| unique segment-pair intersections (window repeats removed) | 18,643 | 168,818 |
| events (hits at one shared vertex or overlap merged) | **18,469** | **166,819** |

Notes on the counts:

- Matches last longer than 90 minutes (stoppage time), and the number of players present varies (substitutions, missing data).
- The last full window of a period ends up to 1 s before the period ends (3812: 0.70 s and 0.31 s uncovered).

## 5. The pipeline, step by step

1. **Source.** The checksum-verified Step 1 RAW cache (`m11-encounters-v2`), read only.
2. **Trajectories.** Each player's polyline consists of its update instants (the provider updates at about 15 Hz in 29.97 fps
   files) joined by valid steps. Coordinates are RAW metres; nothing is smoothed or interpolated.
3. **Windows.** Within each period, centres are every 1 s; each window spans [c − 2.5, c + 2.5].
4. **Pairs.** All unique pairs of players present in the window.
5. **Intersection.** Exact segment-pair intersection, which gives P, the segment on each player and the parameter u.
6. **Times.** t_A and t_B are each read on that player's own crossing segment from the source timestamps. Both are kept.
7. **Deduplication.** One record per segment pair across all windows, with every window kept as provenance. Two hits at the
   same vertex (≤ 1 mm, adjacent segments) are one event. Separate crossings of a pair stay separate.
8. **Catalogue record** (`root.catalogue.jsonl`):
   - match, period, slots, jersey and player IDs;
   - P; the source rows, frames, timestamps and u of both crossing segments;
   - t_A, t_B, Δt;
   - windows: all centres, the expected centres, a consistency flag and the canonical window;
   - Before/After: the row bounds of each player's piece in the canonical window (`catalogue.paths()` rebuilds the vertices
     exactly);
   - provider visibility (VISIBLE / ESTIMATED) at the crossing and the ESTIMATED share;
   - quality flags, the advisor-recipe times, and pair-cluster metadata.

Source trajectories are never shortened; Before and After are derived views.

## 6. Verified results

**Match 3812:**

- 18,469 events: 18,455 proper crossings, 8 vertex crossings, 6 touches.
- Byte-identical to v3. Window provenance is consistent for 18,469 / 18,469.
- The advisor recipe agrees on 18,468; one is a loop.

**All 10 development matches:** 166,819 events, 11,894 to 19,913 per match:

| match | events | median \|Δt\| (s) |
|---|---|---|
| 3814 | 11,894 | 2.17 |
| 10506 | 13,352 | 2.17 |
| 3836 | 15,067 | 2.21 |
| 3833 | 15,661 | 2.22 |
| 3815 | 15,685 | 2.16 |
| 3816 | 18,305 | 2.17 |
| 3812 | 18,469 | 2.10 |
| 3847 | 18,825 | 2.16 |
| 3829 | 19,648 | 2.18 |
| 3851 | 19,913 | 2.10 |

Every match is identical to v3, and window provenance is consistent for all events. The advisor recipe agrees on 166,814;
the 5 others are loops where `project` returns another pass of the same player.

**Repeated crossings.** 82 % of events belong to a pair that crosses again within 5 s, mostly slow or jittering players
weaving around each other. These are distinct intersection points, not window duplicates, so they are kept.

## 7. Real examples (3812; figures in `m11-occlusion-catalogue-v1/3812/`)

**A. Closest to the same time** (`root.fig-closest_to_same_time.png`; event `b04199215143947c`).

| | |
|---|---|
| players | #16 (A) and #3 (B), period 2 |
| P | (−45.915, 0.879) |
| A's crossing segment | frames 100477 → 100479, t = 3352.585919 → 3352.652653 s, u = 0.6373 |
| B's crossing segment | same frames and timestamps, u = 0.6373 |
| times | t_A = t_B = 3352.6284 s (a numerical tie) |
| provider status | both ESTIMATED at the crossing |
| Before / After | 34 / 41 vertices each |

The two players pass the same point in the same frame interval. However, both tracks are provider-estimated there, so this is
not observed evidence of a meeting.

**B. Median time difference** (`root.fig-median_time_difference.png`; event `751439e3b92975d5`).

| | |
|---|---|
| players | #3 (A) and #10 (B) |
| P | (−4.711, −8.637) |
| A's crossing segment | 6219.119 → 6219.186 s, u = 0.982, so t_A = 6219.185 s |
| B's crossing segment | 6221.255 → 6221.321 s, u = 0.494, so t_B = 6221.288 s |
| Δt | +2.10 s |
| distance from P | when A is at P, B is 3.92 m away; when B is at P, A is 3.92 m away |

The paths cross, but the players never meet: A had passed P 2.1 s before B arrived.

**C. Repeated crossings of one pair** (`root.fig-repeated_crossings_of_one_pair.png`): #21 and #6.

| | t_A (s) | t_B (s) | Δt (s) |
|---|---|---|---|
| P1 | 1939.190 | 1936.273 | −2.92 |
| P2 | 1942.656 | 1938.731 | −3.93 |
| P3 | 1942.957 | 1938.939 | −4.02 |

Three separate events. Each is drawn in its own canonical window, so the curves overlap in the figure.

**D. Loop** (`root.fig-project_picks_other_pass.png`; event `bcef557ba8706495`).

| | |
|---|---|
| crossing segment | t_A = 300.1485 s |
| advisor's `project` | 300.1184 s: A's path passes near P twice, and `project` returns the first nearest pass |
| exact segment index | resolves it |

## 8. Does the same-time assumption hold? Mostly not

No threshold is used here. Over the 166,819 intersections:

| evidence | value |
|---|---|
| exact numerical tie, \|t_A − t_B\| ≤ 1e-9 s | 15 events (0.009 %) |
| crossing segments overlap in time (arrival order below the ~0.067 s sampling resolution) | 3,485 (2.1 %) |

Quantiles of \|t_A − t_B\| and of the other player's distance from P at the moment one player is at P:

| quantile | \|Δt\| (s) | distance from P (m) |
|---|---|---|
| q1 | 0.048 | 0.04 |
| q5 | 0.23 | 0.16 |
| q10 | 0.44 | 0.32 |
| q25 | 1.06 | 0.91 |
| median | 2.16 | 2.23 |
| q75 | 3.29 | 4.81 |
| q90 | 3.98 | 8.12 |
| max | 4.93 | — |

Under "one point at one time", Δt would be zero up to timing noise, and the other player would be at P. In the data, the median
crossing is a sequential pass: the second player arrives about 2 s after the first, which by then is about 2 m away.

**The two statements cannot both hold.** A 2D curve intersection ignores time. The time read on A's trajectory and the time
read on B's are equal only in the rare simultaneous cases. In general they are different numbers, and the data shows this.

## 9. Questions for Dr. Basha

1. **Same point at the same time, or same point at any time?** Should a sequential pass (A at P 2 s before B) count as an
   occlusion?
   - If yes, the catalogue is complete as built.
   - If no, which criterion separates them: only the 15 exact ties, the 2.1 % with concurrent crossing segments, or a tolerance
     he chooses? We will not pick one.
   - An alternative is a space-time intersection (curves in x, y, t), which matches "one point at one time" directly.
2. **Window length:** ±2.5 s (5 s, used) or 4 s?
3. **Validity rules:** should implausible jumps, duplicate-position artifacts and gaps over 0.5 s break a trajectory (current),
   or be joined like a raw `LineString` (+20 % events in 3812 for jumps)?
4. **Repeated crossings:** should a pair's repeated crossings within seconds (82 % of events) stay separate events or form one
   episode? His deduplication only covers overlapping windows.
5. **Two details of his code:**
   - it returns nothing for collinear overlaps (`LineString` results); we keep them as events;
   - `project` can choose the wrong pass of a loop; we use the exact segment.

## 10. Recommended next step

Send Dr. Basha section 8, with examples A and B, and ask question 1, the same-time criterion. It decides which part of the
complete catalogue is the occlusion dataset, and no further extraction work should be done before it is answered.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m football_fatigue.intersections.catalogue --matches 3812 --sensitivity --no-cross-match
.\.venv\Scripts\python.exe -m football_fatigue.intersections.catalogue --matches 3814 3815 3816 3829 3833 3836 3847 3851 10506 --no-cross-match
.\.venv\Scripts\python.exe -m football_fatigue.intersections.catalogue --matches 3812 3814 3815 3816 3829 3833 3836 3847 3851 10506 --cross-match-only
.\.venv\Scripts\python.exe -m pytest tests/test_intersections_catalogue.py
```
