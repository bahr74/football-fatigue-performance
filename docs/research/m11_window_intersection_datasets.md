# M11: Advisor-aligned sliding-window intersection datasets (methods A, B, C)

Status: **extraction, cleaning/validation, visualisation and comparison done. No model was trained.** Written 2026-11-13.

**Scope:**

- Data: the 40 FIFA WC 2022 development matches only. No validation or test match was read.
- Pairs: two-player pairs only.
- RAW data was read-only, and its size/mtime digest is identical before and after (`raw_unchanged: true`).

All numbers below were computed by the code named here. Labels: **[V]** verified by computation, **[I]** inference.

## 1. Code, outputs and how to reproduce

| Item | Path |
|---|---|
| Extraction core | `src/football_fatigue/window_intersections/core.py` |
| CLI, dataset writers and checks | `src/football_fatigue/window_intersections/__main__.py` |
| Figures | `src/football_fatigue/window_intersections/figures.py` |
| Unit tests (9, synthetic) | `tests/test_window_intersections.py` |
| Pilot (3812, 3816) | `data/processed/m11-window-intersections-pilot-v1/` (`shared/`, `method_a/`, `method_b/`, `method_c/`) |
| Shared layer (all 40) | `data/processed/m11-window-intersections-v1/`, with per match `events_{linear,pchip}.parquet`, `pair_windows_{linear,pchip}.parquet`, `match_summary.json`; at the root `summary.json`, `figures/`, `manifest.json` |
| Method A dataset | `data/processed/m11-window-method-a-intersection-time-v1/` |
| Method B dataset | `data/processed/m11-window-method-b-intersection-anchor-distance-v1/` |
| Method C dataset | `data/processed/m11-window-method-c-intersection-time-and-distance-v1/` |
| Visual inspection index | `data/processed/m11-window-intersections-v1/visual_index.html` (also `.md`) |

Each method directory contains:

- `candidates_{linear,pchip}.{parquet,jsonl}`
- `config.json`, `counts.json`, `rejections.json`, `linear_vs_pchip.json`
- `figures/` and `manifest.json` (SHA-256 of every output and of the code)

To reproduce:

```
python -m football_fatigue.window_intersections --layout pilot --matches 3812 3816 --workers 1
python -m football_fatigue.window_intersections --layout full --workers 1 --brute 100
```

Runtime and run notes:

- About 45 s per match for both variants; the 40 matches took about 30 min.
- `--workers > 1` hung at pool start-up in this environment, so the runs used the sequential mode.
- The pilot was produced before a small change to the figure-example picker (distinct examples per category). Its figures differ slightly; its data do not.

## 2. Exact construction (as implemented)

**Windows**

- Anchors: every whole second k of period time, where the period's first frame is second 0.
- Window: [k − 2.5, k + 2.5] on the actual frame timestamps (29.97 fps files, positions updating at about 15 Hz).
- Windows that do not fit inside the period are *incomplete*: they are counted and not evaluated.

**Players and pairs**

- Active player: has at least one AVAILABLE sample in the window.
- Every unordered pair of active players is one pair-window evaluation. A pair is always stored as `slot_a < slot_b`, so (A, B) and (B, A) are never both formed. Uniqueness is asserted per match.

**Trajectories (quality policy)**

- Built from Step 1 valid segments (`intersections.exact.trajectories`).
- Never joined across an invalid step (gap > 0.5 s or a period change), an implausible jump, or an unattributed or ambiguous sample. Held positions become stationary segments.
- Segments touching a bit-identical duplicate position (confirmed corruption) are also removed.
- Nothing is joined across a chain break, a substitution or a period boundary.

**Intersection**

- The two players' spatial paths meet, at any times tA and tB.
- It is solved exactly on the segments, once per pair and period. Window membership then requires both tA and tB inside the window, which is the same as clipping both trajectories to the window.
- An independent brute-force re-computation (clipping plus `intersections.geometry.intersect`) found **0 mismatches**:
  - pilot: 800 pair-windows per match and variant;
  - full run: 100 per match and variant, 8,000 in total;
  - unit tests: synthetic tracks.

**First-intersection rule.** The primary intersection of a pair-window is the first by:

1. min(tA, tB), the earliest arrival of either player;
2. then max(tA, tB);
3. then x;
4. then y.

Further intersections are counted, and up to 8 are stored as diagnostics (`extra` in `pair_windows_*.parquet`).

**Methods** (on the primary intersection):

- **A:** |tA − tB| ≤ τ, with τ ∈ {0.10, 0.20, 0.30} s, nested.
- **B:** ‖A(k) − B(k)‖ ≤ 0.25 m at the *exact anchor time* k. Positions come from the variant curve. The rule is unavailable when a player has no valid segment at k.
- **C:** A **AND** B (asserted per event: C = A ∧ B).
- The minimum distance anywhere in the window is stored separately as `window_min_raw_dist_m`. It is a diagnostic only and never used for selection.
- B at 0.50 and 1.00 m is a reported sensitivity only; no dataset is built from it.

**Deduplication**

- One physical intersection selected by several overlapping windows is one **event**, and keeps its list of windows (`windows`; for B and C, `b_pass_windows`).
- For B and C, an event qualifies when the anchor rule holds in at least one of its windows.
- **Encounters** group the events of one pair within 1.0 s and 1.0 m. They are an audit grouping and never remove events.

**Variants**

- `linear` (baseline).
- `pchip`: `scipy.interpolate.PchipInterpolator` per chain, for chains with ≥ 3 vertices. Geometry is evaluated on 10 sub-pieces per native segment.
- Check: PCHIP stayed inside every interval's sample box, with a maximum excursion of **0.000 m** over 50,000 sampled segments per match.

## 3. Scan volumes, 40 matches, linear [V]

| Quantity | Value |
|---|---|
| Complete windows evaluated | 243,498 (241,781 have exactly 22 active players) |
| Incomplete windows (period edges, not evaluated) | 437 |
| Player-windows: active / valid trajectory / complete trajectory (one chain covers the whole 5 s) | 5,358,856 / 5,282,215 / 5,007,477 |
| Pair-window evaluations (all active pairs) | **56,290,029** |
| Pair-windows where both players have valid trajectories / complete trajectories | 54,899,163 / 49,832,274 |
| Step 1 steps → segments used | 81,826,714 → 78,840,769 in 113,082 chains |
| Excluded steps: implausible jump / invalid gap or period / unattributed or ambiguous / duplicate position (corruption) | 385,029 / 938 / 1,368,000 / 1,245,516 |
| Spatial intersections with \|tA − tB\| ≤ 5 s | 719,497 |
| Pair-window detections (pair-windows with ≥ 1 intersection) | 951,814 |
| Deduplicated primary events | 419,526 |
| Intersections never primary in any window | 299,971 |
| Pair-windows with more than one intersection | 303,917 (31.9 %); median 1, maximum 252 |

**How often the first-intersection simplification applies:** in 31.9 % of detected pair-windows. Within the method datasets:

| Method | Passing pair-windows with > 1 intersection |
|---|---|
| A, τ = 0.30 | 31,529 of 87,786 |
| B | 4,816 of 5,810 (83 %) |
| C, τ = 0.30 | 1,517 of 2,240 |

For B the share is high because a pair that is close at the anchor time tends to weave around each other, so their paths cross several times.

## 4. Candidates per method, 40 matches, linear [V]

| Method / threshold | Raw pair-window detections | Duplicates removed | Events merged from > 1 window | **Independent events** | Distinct encounters | Player pairs | Valid after cleaning (not rejected) | Accepted (no flags) | ≥ 1 s context both players, both sides | Full ±2.5 s context | Sample-ready* |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A, τ 0.10 s | 25,504 | 18,138 | 5,707 | **7,366** | 7,274 | 4,193 | 7,156 | 5,360 | 5,998 | 5,344 | 5,360 |
| A, τ 0.20 s | 54,547 | 38,752 | 12,349 | **15,795** | 15,492 | 6,246 | 15,427 | 11,596 | 13,012 | 11,572 | 11,594 |
| A, τ 0.30 s | 87,786 | 62,323 | 20,007 | **25,463** | 24,821 | 7,604 | 24,954 | 18,949 | 21,225 | 18,912 | 18,946 |
| B, R 0.25 m | 5,810 | 833 | 634 | **4,977** | 4,612 | 2,878 | 4,891 | 4,539 | 4,745 | 4,542 | 4,539 |
| C, τ 0.10 & R 0.25 | 1,098 | 123 | 94 | **975** | 972 | 820 | 945 | 828 | 880 | 828 | 828 |
| C, τ 0.20 & R 0.25 | 1,717 | 224 | 168 | **1,493** | 1,475 | 1,162 | 1,456 | 1,302 | 1,370 | 1,302 | 1,302 |
| C, τ 0.30 & R 0.25 | 2,240 | 324 | 241 | **1,916** | 1,880 | 1,416 | 1,874 | 1,694 | 1,778 | 1,694 | 1,694 |

\*Sample-ready = status accepted **and** ≥ 1 s of unbroken valid trajectory before and after the event time for both players.

Per match, the median number of events is 626 (A, τ 0.30), 123 (B) and 46.5 (C, τ 0.30).

**Why A loses ~70 % to deduplication and B/C only ~15 %.** In A, the same intersection is the first of up to 5 overlapping windows. In B and C, the anchor-distance rule usually holds in only one of those windows.

**Threshold rejections among the 419,526 primary events:**

| Method | Rejections |
|---|---|
| A | 394,063 have \|dt\| > 0.30 s |
| B | 412,306 have anchor distance > 0.25 m in every window; 2,243 have no anchor position in any window (gap at the anchor) |
| C | 23,547 fail only the distance rule; 3,061 fail only the time rule; 391,002 fail both |

**Quality tier at the intersection** (A, τ 0.30 / B / C, τ 0.30):

| Tier | A | B | C |
|---|---|---|---|
| VISIBLE/VISIBLE | 7,915 | 313 | 129 |
| Mixed | 5,558 | 702 | 286 |
| ESTIMATED/ESTIMATED | 11,990 | 3,962 | 1,501 |

Methods B and C are dominated (80 %, 78 %) by intersections where both players are provider-ESTIMATED. This is the most important data finding of this run [V]. A plausible cause is that the provider's estimated positions of two nearby players converge, but that is an [I], not verified.

**Cleaning status, all 419,526 primary events:**

- Accepted: 344,209.
- Flagged: 73,033 (suspected artifacts in the ±2.5 s context).
- Rejected: 2,284 (confirmed corruption).

**Reasons, A at τ 0.30** (events with the reason for either player):

| Class | Reasons |
|---|---|
| Reject | duplicate position 704, unattributed/ambiguous 49 |
| Flag | implausible jump 7,483, frame-wide shift 343, discontinuity candidate 169, substitution window 15 |

The same reasons for B and C are in `rejections.json` and `counts.json`.

**Information flags (kept, never rejected):**

- ESTIMATED at the intersection; over-speed sub-metre steps; context crossing a period boundary (19 in A).
- Speed > 10 m/s and acceleration > 10 m/s². The acceleration flag fires on most events (about 50k player-flags in A). At 15 Hz with 1–10 cm noise it is noise-driven and should not be used as a filter without recalibration [I].

**Multi-player diagnostics** (no multi-player model was built):

| Diagnostic | A, τ 0.30 | B | C, τ 0.30 |
|---|---|---|---|
| A third player within 2 m of the pair at the event time | 6,219 | 2,239 | 636 |
| Another event of a different pair shares a player within 1 s | 15,300 | 1,853 | 763 |

## 5. Linear vs PCHIP [V]

| | Linear | PCHIP |
|---|---|---|
| Intersections | 719,497 | 727,241 |
| Pair-window detections | 951,814 | 953,269 |
| Primary events | 419,526 | 420,852 |

Event matching (same pair, ≤ 0.2 s, ≤ 0.5 m, one-to-one): 417,890 matched; 1,636 linear-only; 2,962 PCHIP-only.

On matched events:

| Measure | Median | p90 | p99 |
|---|---|---|---|
| Intersection location change | 0.27 cm | 1.3 cm | 5.0 cm |
| \|dt\| change | 1.3 ms | 7.4 ms | 34 ms |

Method membership:

| Method | Linear | PCHIP | Linear-passing events whose PCHIP twin fails or is missing |
|---|---|---|---|
| A, τ 0.30 | 25,463 | 25,605 | 274 (1.1 %) |
| B | 4,977 | 5,005 | 206 (4.1 %) |
| C, τ 0.30 | 1,916 | 1,908 | 72 (3.8 %) |

Numerical artifacts:

- PCHIP overshoot: 0.000 m.
- PCHIP-unsupported segments (one-segment chains) stay linear.
- PCHIP-only intersections (2,962) are 0.7 % of events.

**Conclusion: PCHIP does not materially change counts, locations or arrival times.** It moves about 1–4 % of method memberships across a threshold edge.

## 6. Visualisations (actual FIFA samples)

Open `data/processed/m11-window-intersections-v1/visual_index.html`.

**Per method** (example match 3812):

- 2 accepted examples with VISIBLE samples, and 1 accepted example with ESTIMATED samples;
- 1 example that passes the threshold but has quality status rejected or flagged;
- threshold-rejected examples (C also shows "passes R but fails τ");
- 1 linear-vs-PCHIP overlay (the largest location change);
- 1 duplicate-window mapping, showing which windows merge into one event.

Each event figure shows:

- the 5 s window and its anchor;
- both trajectories, with raw samples (VISIBLE filled, ESTIMATED hollow);
- the first intersection (star) and each player's arrival time;
- the anchor-time positions and distance;
- the synchronized distance curve against R;
- the pass/fail of each rule, the quality status, and context availability.

**Method-level figures** (`m11-window-intersections-v1/figures/`): counts, quality tiers, status, rejection and flag reasons, threshold sensitivity, A/B overlap, context completeness, and the PCHIP location and |dt| change histograms.

**What the figures show:**

- Accepted events are real path crossings with clear before and after paths.
- Most B and C candidates are weaving or close-marking situations with several crossings in the window.
- The quality-rejected example shows exactly what the layer is meant to catch: an ESTIMATED track that teleports (an 11 m distance drop in under 0.1 s), next to duplicate positions.
- PCHIP and linear paths are visually indistinguishable at this scale, except where samples are sparse.

## 7. Known issues

1. **`pchip_supported_a/b` metadata in linear-variant v1 records.** In the linear-variant records of the v1 outputs, these columns are always False, and the info flag `pchip_unsupported_segment` is set on every linear event.
   - The PCHIP-variant values are correct.
   - No selection, status, tier or count depends on these columns.
   - The code was fixed after the run (the support flag is now the same for both variants, covered by a unit test). The v1 manifests record the pre-fix code hashes.
   - The v1 outputs were not rewritten, because versioned outputs are never overwritten.
2. Multiprocessing (`--workers > 1`) hung at start-up in this environment, so the runs were sequential.
3. The acceleration info flag is noise-dominated (§4).

## 8. Training readiness

**These are extraction datasets, not model-ready samples.** What is available:

- Anonymous before and after trajectories: `context_min` / `context_full` and `before_s_*` / `after_s_*` per player.
- Supervision through provider slot continuity: `label_reliable`, which excludes rejected events and those flagged for substitution or discontinuity.
- A common schema across A, B and C, so all three can feed one sample builder.

What still blocks training:

- **No sample construction yet.** The gap design (which part around the intersection is hidden), the input lengths, the keep/swap target, the normalisation and the masks are all undefined.
- **Supervision quality is weak in the B and C pools.** 78–80 % of B/C events are ESTIMATED at the intersection, so the provider's own identity there is an estimate.
- **Pool sizes differ by more than 10×.** Sample-ready counts are 18,946 (A, τ 0.30), 4,539 (B) and 1,694 (C, τ 0.30). "Same training budget" therefore needs a subsampling rule.
- **Samples overlap.** Events share players and time (15,300 of 25,463 in A), so splits must be grouped by match. Overlapping samples must not be counted as independent.
- **No common held-out population is defined yet.**

## 9. Decisions needed before model training

1. Which threshold is each method's dataset? For example A τ = 0.20 or 0.30 s, B 0.25 m, C 0.20/0.25.
2. Linear or PCHIP as the primary geometry. The data say it does not matter; linear is simpler.
3. Quality inclusion: accepted only, or accepted + flagged. Also how the ESTIMATED/ESTIMATED tier is treated.
4. The sample design (gap, input/output windows, target, normalisation) and a sample builder.
5. The common held-out evaluation population and the grouped-by-match split; and whether FIFA validation can be used (it has already been used twice).
6. The equal-budget rule (subsample A and B to C's size, or a fixed number of steps).
