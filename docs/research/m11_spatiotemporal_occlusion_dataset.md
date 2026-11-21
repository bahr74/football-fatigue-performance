# M11 — Advisor-aligned spatiotemporal occlusion-proxy dataset

Date 2026-11-09. Development matches only (3812, 3814, 3815, 3816, 3829, 3833, 3836, 3847, 3851, 10506). No validation or test
match was used, no model was trained, and no raw or earlier artifact was changed. Nothing is committed.

- Output: `data/processed/m11-occlusion-spatiotemporal-v1/<match>/` and `.../cross-match/summary.json`.
- Code: `src/football_fatigue/intersections/spatiotemporal.py`.
- Tests: `tests/test_intersections_spatiotemporal.py`.

**Inputs** (read and checksum-verified against their manifests; unchanged):

- `m11-occlusion-catalogue-v1`: the 166,819 exact geometric candidates;
- `m11-intersections-v3` eligibility: the S1 per-event measurements (speeds, local position noise, angle, window completeness,
  timing uncertainty).

## 1. What the advisor requested

For every second, each player's trajectory over ±2.5 s; all unique pairs; an exact curve intersection; non-intersecting pairs
discarded; window duplicates removed; the time of P read from each original trajectory. He describes the occlusion as "one
point at one time".

**Project decision (2026-11-09):** a geometric crossing alone is not an occlusion. The two players must be at P at the same
time, up to the uncertainty of the tracking data.

## 2. What the existing detector already did correctly

All six steps, verified in the 2026-11-09 realignment:

- the 166,819 candidates are byte-identical to v3;
- window provenance is consistent for every event;
- the advisor's `project`/`searchsorted` time agrees on 166,814, and the other 5 are loops where `project` picks another pass.

That catalogue is the immutable geometric baseline. This dataset adds a decision per candidate and removes nothing.

## 3. Temporal coincidence rule

**What the data shows:**

- The provider updates every player every h = **0.0667 s** (2 video frames). The update interval is identical in all 10
  matches, and 99.998 % of crossing segments last exactly h.
- Local position noise: median 0.034 m, q95 0.14 m. This is the second-difference estimate; it includes real acceleration, so
  it is an upper bound.
- Speed at the crossing: median 2.4 m/s; 11 % of players move below 0.5 m/s.
- σ_dt is the S1 timing uncertainty of t_A − t_B, propagated to first order from:
  - quantisation, h/√12 per player;
  - position noise / speed, amplified by 1/sin(angle).

  Quantisation alone gives σ_dt ≥ 0.027 s.

**Rule** (per candidate; z = 2; τ = h):

| decision | condition | meaning |
|---|---|---|
| **rejected** | \|t_A − t_B\| > max(τ, 2σ_dt) | sequential pass: the difference exceeds both the data's time resolution and its noise |
| **accepted** | \|t_A − t_B\| ≤ 2σ_dt ≤ τ, and the conditions listed below | consistent with simultaneity, with timing determined to within one provider update |
| **ambiguous** | everything else, with every reason recorded | cannot be decided from the data |

An accepted candidate must also meet these conditions:

- the crossing time is well conditioned: speed ≥ 0.5 m/s, angle ≥ 15°, velocity available;
- no sample is missing at the crossing (crossing segment > 1.5 h);
- no tracking-artifact veto:
  - duplicate position;
  - unresolved attribution;
  - window incomplete or crossing a period boundary;
  - speed spike within 0.5 s of the crossing.

**Justification:**

- τ is the native temporal resolution; arrivals closer than one update cannot be ordered from the data.
- 2σ_dt is the 95 % band of measurement uncertainty.
- Accepting only when the noise band fits inside one update makes the rule **conservative**:
  - a noisy, slow or shallow crossing is not accepted on a lucky small Δt;
  - a large Δt is not rejected when the noise could explain it.
- The 0.20 s threshold is not used. Neither is "same second".
- The floors (0.5 m/s, 15°, spike within 0.5 s) and z = 2 are inherited unchanged from the S1 / S1.1 audit, not re-tuned here.

**Descriptive timing classes** (recorded, never a decision on their own):

| class | accepted | ambiguous | rejected |
|---|---|---|---|
| exact equality (\|Δt\| ≤ 1e-9 s) | 3 | 12 | 0 |
| same sample interval | 164 | 947 | 0 |
| overlapping intervals | 117 | 1,839 | **403** |
| disjoint intervals | 0 | 6,043 | 157,291 |

Overlapping intervals alone do not decide: 403 such candidates are rejected.

**Sensitivity** (accepted / ambiguous / rejected over 166,819 candidates):

| τ | z = 1 | z = 2 | z = 3 |
|---|---|---|---|
| 0.5 h (0.033 s) | 150 / 4,413 / 162,256 | 0 / 9,092 / 157,727 | 0 / 13,613 / 153,206 |
| **1 h (0.067 s)** | 477 / 4,552 / 161,790 | **284 / 8,841 / 157,694** | 0 / 13,613 / 153,206 |
| 1.5 h (0.100 s) | 624 / 5,095 / 161,100 | 736 / 8,742 / 157,341 | 430 / 13,248 / 153,141 |
| 2 h (0.133 s) | 720 / 5,864 / 160,235 | 974 / 9,118 / 156,727 | 920 / 13,141 / 152,758 |
| 3 h (0.200 s) | 853 / 7,679 / 158,287 | 1,264 / 10,347 / 155,208 | 1,454 / 13,780 / 151,585 |

**The rejected count is stable** (151.6k–162.3k, ≥ 91 %). **The accepted count is not** (0–1,454). The primary rule sits close
to the data's resolution limit: with quantisation alone, 2σ_dt is 0.054 s against τ = 0.067 s. This is the main limitation
(section 6).

## 4. From geometric candidates to accepted spatiotemporal events

| match | candidates | accepted | of which both observed | ambiguous | rejected |
|---|---|---|---|---|---|
| 3812 | 18,469 | 40 | 21 | 1,071 | 17,358 |
| 3814 | 11,894 | 22 | 10 | 431 | 11,441 |
| 3815 | 15,685 | 31 | 10 | 705 | 14,949 |
| 3816 | 18,305 | 35 | 7 | 780 | 17,490 |
| 3829 | 19,648 | 31 | 16 | 1,149 | 18,468 |
| 3833 | 15,661 | 25 | 8 | 609 | 15,027 |
| 3836 | 15,067 | 30 | 9 | 762 | 14,275 |
| 3847 | 18,825 | 27 | 8 | 975 | 17,823 |
| 3851 | 19,913 | 25 | 10 | 1,644 | 18,244 |
| 10506 | 13,352 | 18 | 4 | 715 | 12,619 |
| **total** | **166,819** | **284 (0.17 %)** | **103** | **8,841 (5.3 %)** | **157,694 (94.5 %)** |

**Accepted events (all 10 matches):**

| measure | median | q95 | max |
|---|---|---|---|
| \|Δt\| | 0.029 s | 0.057 s | 0.066 s |
| σ_dt | 0.030 s | 0.033 s | — |
| the other player's distance from P when one player is there | 0.15 m | 0.40 m | 0.72 m |

So spatial coincidence follows. Other characteristics:

- all 284 are proper crossings, in 258 distinct pairs;
- 187 involve opposite teams and 97 the same team;
- 219 are in a pair that crosses repeatedly within 5 s.

**Ambiguous reasons** (an event can have several):

| reason | events |
|---|---|
| timing precision insufficient (noise band wider than one update) | 8,699 |
| timing ill-conditioned (slow or shallow) | 5,770 |
| speed spike near the crossing | 2,175 |
| duplicate position | 183 |
| attribution | 46 |
| within resolution but beyond the noise band | 33 |
| period boundary / gap / coverage | 8 |

The rejected events have a median \|Δt\| of 2.23 s, and the other player is a median 4.0 m from P.

**Per-event record** (`root.events.jsonl`, all candidates):

- match, period, slots, jersey and player IDs; P;
- t_A, t_B, Δt;
- decision, reasons, τ, 2σ_dt, σ_dt and the timing class;
- the sensitivity decision at every grid point;
- quality: ESTIMATED / observed at the crossing, missing sample, speeds, local noise, angle, vetoes, flags, smoothing survival
  and smoothed Δt;
- source segments (rows, frames, timestamps, u) and the canonical and all detecting windows;
- deduplication and catalogue provenance.

`root.trajectories.npz` holds the Before / After vertices (x, y, t) of both players for all accepted and ambiguous events
(3812: 1,111 events, 150,163 vertices), rebuilt from the RAW segments.

**Figures:** `root.fig-{accepted_observed, accepted_estimated, ambiguous_noise, rejected_sequential, smoothing_keeps_accepted}.png`
(3812 example accepted_observed: #8 and #7, |Δt| = 0.004 s, P at (44.85, −3.03)). The earlier "same time" example (both
ESTIMATED, Δt = 0) is now **ambiguous**: the crossing timing is ill-conditioned there.

## 5. Noise and smoothing

**Without smoothing (primary):** the decisions above use the raw positions only.

**Smoothing study** (a derived copy; quadratic Savitzky–Golay over 5 update instants = 0.27 s):

- applied to each uniformly sampled run of one player;
- never across gaps, jumps, invalid steps, missing samples or periods; run ends unchanged;
- the detector is unchanged, and smoothed events are matched one-to-one with raw candidates (same pair, both times within
  0.25 s, P within 0.5 m; these tolerances are used only for this comparison).

| | 10 matches |
|---|---|
| raw candidates → smoothed candidates | 166,819 → 132,062 |
| raw candidates that disappear (jitter crossings) | 36,269 (21.7 %): 32,004 rejected, 4,265 ambiguous, **0 accepted** |
| new candidates after smoothing | 1,512 |
| accepted that keep an intersection | **284 / 284** |
| accepted still within one update after smoothing | 281 / 284 |
| rejected that move within one update | 14 |
| 3812 matched accepted: change of \|Δt\| | median 0.0006 s, q5–q95 −0.005 to +0.017 s |
| 3812 matched: shift of P | median 8 mm, q95 8 cm |

**Decision: smoothing is not applied to the dataset.**

- Jitter mainly creates spurious crossings of slow, weaving players far apart in time; the coincidence rule already rejects or
  holds back nearly all of them.
- The accepted set is robust to smoothing.
- Smoothing would change the exact original-trajectory geometry the advisor asked for.

Each record carries `quality.smoothing` (survives / smoothed \|Δt\|) as a jitter-sensitivity flag.

## 6. Limitations and scientific assumptions

1. **Proxies, not verified occlusions.** Accepted events are tracking-derived occlusion / encounter proxies. There is no video
   or camera calibration, so whether the players overlapped in any camera image is unknown. Labels are provider slots, not
   verified identities.
2. **The accepted count is resolution-limited and model-dependent.** σ_dt is a first-order propagation, not a validated error
   model, and its quantisation term (uniform over a whole update) is conservative. The count ranges from 0 to 1,454 across
   defensible (τ, z). The ambiguous class (8,841) is kept so that a different, justified rule can be applied without
   rebuilding.
3. **ESTIMATED positions.** 64 % of accepted events (181 / 284) have an ESTIMATED position at the crossing, and 71 % of all
   candidates do.
   - ESTIMATED is flagged, not used in the decision.
   - The data cannot say whether estimated positions are reliable at the crossing, even though occlusion is exactly when a
     provider estimates.
   - Needs an advisor decision.
4. **Speed-spike veto.** 2,175 ambiguous events are held back by the S1.1 spike rule. Spikes near a close encounter may be the
   provider's own confusion, that is, the phenomenon itself. The rule is flagged, not re-tuned.
5. **Repeated crossings.** 77 % of accepted events are in a pair that crosses repeatedly within 5 s. Whether these are one
   occlusion episode is an open question.
6. **Inherited rules.**
   - the S1 floors (0.5 m/s, 15°), the spike window (0.5 s) and z = 2;
   - Step 1 validity (gap ≤ 0.5 s, 12 m/s jump);
   - ±2.5 s windows (the advisor wrote "4 seconds").
7. **Near misses.** Exact intersection cannot see two players who meet but whose sampled paths do not cross, for example a
   head-on pass 2 cm apart (tested). Such meetings are outside the advisor's definition and are not in the dataset.
8. Output size: 499 MB (`root.events.jsonl` keeps every candidate in full).

## 7. Is it ready for identity-recovery model development?

**Not yet as a training set by itself.**

- **For:** it is clean, fully auditable and faithful to the advisor's definition plus the project's same-time decision. It is
  suitable as an **evaluation / analysis set of advisor-defined occlusion proxies**.
- **Against:**
  - Size: 284 accepted events (103 with both positions observed) in 258 pairs over 10 matches is small. The accepted count
    depends on the tolerance choice.
  - Selection prior: these are crossings by construction, and crossing is computed from identity-labelled tracks. A model
    trained only on them learns "assume they crossed". This was measured earlier: the intersection-trained model was at chance
    on non-crossings. A label-symmetric counterpart (meetings that do not cross) is needed before training.
  - The ESTIMATED question (limitation 3) affects 64 % of accepted events.

**Recommended next step:** ask Dr. Basha to confirm three points.

1. τ = one provider update and z = 2, or another value from the sensitivity table.
2. Whether ESTIMATED crossings count.
3. Whether a label-symmetric set of same-time non-crossing meetings may be added for training.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m football_fatigue.intersections.spatiotemporal --matches 3812 3814 3815 3816 3829 3833 3836 3847 3851 10506 --smoothing
.\.venv\Scripts\python.exe -m football_fatigue.intersections.spatiotemporal --matches 3812 3814 3815 3816 3829 3833 3836 3847 3851 10506 --cross-match-only
.\.venv\Scripts\python.exe -m pytest tests/test_intersections_spatiotemporal.py
```
