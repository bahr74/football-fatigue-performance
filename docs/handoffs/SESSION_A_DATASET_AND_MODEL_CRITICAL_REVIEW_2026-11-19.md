# Session A: dataset and model critical review (2026-11-19)

**Repository:** `F:\Football-Fatigue-Project` · **Branch:** `feature/m11-identity-finalization` · **HEAD:** `1bd016d`

**Predecessor:** `docs/handoffs/SESSION_A_RESEARCH_AND_DASHBOARD_AUDIT_2026-11-19.md` (read first; its findings are not repeated
except where needed).

**Nature of this document:** a read-only scientific review. No code, dataset, output or report was changed. No model was trained,
nothing ran on a GPU and nothing was downloaded. All counts come from **existing development caches of the 32 train/CV matches
only**. The match list was taken from `m11-stage2-split-v1/split.json`, and every per-match file read was asserted to belong to the
train/CV split. No held-out match, no official FIFA validation or test match, and no Session B file was opened.

## Evidence labels

| Label | Meaning |
|---|---|
| **[V]** | Verified today from code, or computed today from saved development artifacts (read-only aggregation) |
| **[DOC]** | Reported in a historical project report; not recomputed |
| **[HYP]** | Hypothesis or interpretation |
| **[REC]** | Recommendation |
| **[NV]** | Not yet verified; would need new processing or approval |
| **[EST]** | Estimate derived from verified counts under stated assumptions |

---

## 1. Executive summary

1. **The 25,331 figure is narrowly defined, not too small and not a bug** [V].
   - The 32 development matches contain 5.92 M frames, 130 M player-position samples, 991 player-match identities and 561 unique
     players.
   - At a 0.5 m synchronous-distance radius they contain 49,640 proximity episodes over 9,905 distinct player pairs.
   - Chain-based extraction merges these into 29,861 pair encounters, of which 25,331 pass the 0.8 s label-quality check
     (84.8 %).
   - The per-player overlap rule leaves 18,777 independent ones.
   - The number of distinct encounters **is not hundreds of thousands at 0.5 m**. It approaches 10⁵ only at radii of about 2 m,
     where most events are no longer identity-ambiguous.
2. **Most of the loss is by definition, not by filtering.**
   - The 0.5 m radius (a definition) and the per-pair merge (deduplication) are the dominant reductions.
   - The 0.8 s gap check removes only 15.2 % of candidates. The gap length itself matters little: 0.4 s keeps 25,668,
     0.8 s 25,331 and 1.6 s 24,661.
   - The biggest quality reason is a broken provider identity chain across the window (3,892 candidates).
3. **Multi-player and repeated interactions are present but encoded as independent binary pairs.**
   - 18.8 % of candidates share a player with another candidate within 1 s.
   - 5,968 player pairs meet more than once.
   - At the 1 m Step 1 level, 37 % of encounters have a third player within the radius.
   - The window-intersection layer keeps only the first intersection per pair-window: 243,280 of 581,786 geometric intersections
     (41.8 %) never become an event.
   - Nothing in the current pipeline represents a *group* event.
4. **The historical GRU-vs-LSTM comparison is not conclusive** [V].
   - At the shared 80-epoch budget both gap-last models were still improving: best epoch 76–80 of 80 in 23/24 GRU fold fits and
     about 19/24 LSTM fold fits.
   - Only the GRU later received 160 epochs and shift augmentation (Stage 4C).
   - No neural hyperparameter search was ever run, while every tabular model received an inner-CV grid search.
   - LSTM and SVM were never stress-tested.
5. **The existing models do not generalise to N players as they are** [V by architecture]. Their output is a single
   antisymmetric MATCHED/SWAPPED logit. The GRU's pair head and HGB's per-link costs *are* pairwise compatibility scorers and
   could be reused inside an N×M assignment, but only after retraining as calibrated per-link scorers.
6. **The real bottleneck is not data quantity.** It is the task definition, label and provider-texture confounds, optimisation
   budget and selection fairness [HYP, supported by the learning curves and Stage 4C records].
7. **Recommended direction [REC].**
   - First paper: **two-player recovery done properly**, with a fair four-family benchmark (GRU, LSTM, SVM, HGB, plus Kalman /
     Hungarian / physics), embedded in a **complete pairwise interaction catalogue** with threshold sensitivity.
   - Multi-player N×M association: a controlled second contribution or a follow-up paper. It should not be the primary claim of
     the first paper.

---

## 2. Verified baseline research state

The predecessor audit is still accurate.
- Stages 1–4C are complete.
- Milestones 1–3 are implemented and uncommitted.
- The 16 checkpoints match their recorded hashes.
- The focused tests pass (83 passed on 2026-11-19).
- No final 32-match training and no held-out evaluation have happened.

Nothing in source or data has changed since then [V: `git status` identical].

---

## 3. Complete dataset-count funnel (32 train/CV matches)

**Sources:**
- Step 1 caches `m11-encounters-v2/<m>/root.summary.json`;
- sync-proximity `m11-sync-proximity-v2/<m>/root.summary.json`;
- window intersections `m11-window-intersections-v1/<m>/match_summary.json`;
- Stage 2 `m11-stage2-candidates-v1/matches/<m>/{candidates,framewise_audit}.parquet`;
- Stage 3 `m11-stage3-samples-v1/matches/<m>/meta_g0p80.parquet`.

All values are **exact** aggregations [V] unless marked.

### 3.1 Raw observation level

| Quantity | Value | Denominator / note |
|---|---|---|
| Approved development (train/CV) matches | **32** | split SHA `66b2321a…6c0` |
| Frames (29.97 fps records) | **5,923,415** | per match 172,575–233,020 (median 184,232) |
| Tracked time | 197,643 s = **54.9 h** | — |
| Player-match identities (slots) | **991** | 31.0 per match (starters + substitutes) |
| Unique provider player ids | **561** | across 32 matches |
| Player-position samples available | **130,323,267** | frames × present slots |
| …provider update instants (≈ 15 Hz) | 65,215,362 | each update is repeated in the next 29.97 fps record (`REPEAT_OF_PREVIOUS`) |
| …ESTIMATED samples | 80,835,389 (**62.0 %**) | status is metadata, not a filter |
| …valid VISIBLE ("valid_raw") samples | 47,749,734 (36.6 %) | — |
| Implausible-jump samples (> 12 m/s and ≥ 1 m) | 281,467 (0.22 %) | flagged, not dropped |
| Over-speed sub-metre steps | 169,555 | flagged, not dropped |
| Identity-discontinuity candidates | 9,316 | flagged |
| Synchronous pair-interval comparisons | 671,904,700 | sync-proximity v2 |

### 3.2 Proximity level (synchronous distance d(t) ≤ R, identity-free; sync-proximity v2)

| R | Segment-level detections | Window-level detections (5 s windows, 1 s stride) | **Unique episodes** | Distinct pairs | est/est | obs/est | obs/obs | same team |
|---|---|---|---|---|---|---|---|---|
| 0 m | 675,340 | 54,441 | 1,992 | 507 | 1,970 | 0 | 22 | 1,973 |
| 0.10 m | 698,600 | 100,914 | 11,081 | 4,412 | 8,657 | 1,594 | 830 | 5,859 |
| 0.25 m | 782,818 | 173,435 | 24,634 | 7,237 | 18,571 | 3,986 | 2,077 | 11,616 |
| **0.50 m** | **1,061,885** | **315,418** | **49,640** | **9,905** | 35,765 | 8,017 | 5,858 | 21,330 |

**Notes:**
- The R = 0 row is almost entirely **duplicate provider positions** (1,963 of 1,992), that is, cloned ESTIMATED tracks.
- Window-level detections overcount by a factor of about 6 at 0.5 m because of overlapping windows: 265,778 are repeated
  detections of the same episode.
- **These are the "hundreds of thousands to a million" rows the user anticipated. They are not independent events.**

### 3.3 Candidate and sample level (Stage 2 → Stage 3, R = 0.5 m)

| Step | Count | Retained | Rule (code) |
|---|---|---|---|
| Raw episodes on continuous provider chains, d ≤ 0.5 m | 45,989 (Σ `n_episodes`) | — | `candidates.episodes_on`, `candidates.py:94,176` |
| Merged per pair (episodes ≤ 1.0 s apart → one encounter) | **29,861** candidates | — | `merge_gap_s = 1.0`, `candidates.py:64,180` |
| 0.8 s label-quality outcome | **ACCEPTED 25,331** / LOWER 17 / REJECTED 4,513 | 84.8 % | `label_quality`, `candidates.py:199` |
| Stage 3 samples built (`g0p80`) | **25,331** | 100 % | `samples.py` |
| Independent (greedy non-overlapping per shared player, span = g + 4 s = 4.8 s) | **18,777** | 74.1 % | `dependency`, `candidates.py:333-360` |
| Overlap groups (connected components) | 16,935 | — | same |
| Per-fold GRU fitting / early-stop / test rows | 19.4–20.3 k / 2.2–2.8 k / 2.7–3.4 k | — | predecessor §3.8 |
| Shift-augmented variants (Stage 4C) | 150,895 | **variants, not events** | `shiftaug.py` |
| M2 stress variants | 55,680 (1,920 encounters × 29) | **variants, not events** | `scenarios.py` |

**Reconciliation, 49,640 → 45,989** [V]. The framewise audit compares slot-based episodes with chain-based candidates over 32,561
frame-level episodes:

| Outcome | Episodes |
|---|---|
| Matched to a candidate | 29,153 |
| **Unmatched** | **3,408 (10.5 %)** |
| …a jump without a valid segment nearby | 1,419 |
| …duplicate-position clones | 1,461 |
| …quarantined period | 520 |
| …unattributed | 21 |

2,169 of the unmatched episodes have an identity or jump flag within 1 s. These exclusions are **intended** (no trustworthy
continuous chain, so no trustworthy label), but they are a selection effect: they remove the most chaotic moments.

### 3.4 Reasons for rejection at 0.8 s (flags can co-occur) [V]

| `g0p80_ok_*` flag failing | Candidates | Meaning |
|---|---|---|
| `continuous_reference` | 3,892 | a player's provider chain does not cover the context window ±2 s around the gap |
| `episode_not_truncated` | 1,831 | the proximity episode touches a chain boundary |
| `chord_speed_plausible` | 1,197 | implied speed across the window > 12 m/s |
| `no_identity_risk` | 69 | discontinuity or label-transfer risk |
| `window_in_period` | 48 | the window leaves the period |
| `no_corruption`, `no_jump_or_shift_flag` | 0 | — |

LOWER_CONFIDENCE (17) reasons: discontinuity candidate (120 hits), substitution window (19), label-transfer (17), reciprocal
exchange (12).

### 3.5 Effect of each assumption

| Assumption | Effect on count | Evidence |
|---|---|---|
| **0.5 m radius** | Defines the population. Halving R to 0.25 m halves unique episodes (24,634); doubling to 1 m roughly doubles them again (§5) | §3.2 [V] |
| **Fixed 0.8 s gap** | Small: accepted 25,668 / 25,331 / 24,661 at 0.4 / 0.8 / 1.6 s. Gap length is not what limits the data | §3.3 [V] |
| **Continuous-chain requirement** | Main quality loss (3,892) plus 3,408 framewise episodes never become candidates | §3.3–3.4 [V] |
| **Two-player-only** | **Nothing is removed.** Multi-player situations are kept and represented as several binary pair samples (§6). The third player is neither hidden nor modelled | `shared_player_candidates` [V] |
| **Simultaneous hiding** | Nothing is removed. Every sample hides both players over the same [g0, g1], an assumption of the *task*, not of the data | `samples.build_one` [V] |
| **Independence rule** | 25,331 → 18,777 for evaluation; training uses all rows | [V] |

### 3.6 Implementation-error check

- **No counting bug found.** Stage 2 has built-in checks:
  - `encounters_separated_by_merge_gap`;
  - `d_min_le_radius`;
  - a coverage check that every local minimum ≤ R lies in a candidate (`candidates.py:382-423, 550-573`);
  - overlap groups match outcomes.
- **Stage 3:** built 100 % of accepted candidates, with labels verified 100,832/100,832 [DOC].
- **Boundary semantics:** documented in `m11_stage4c_correction_v1.md` §2.
- **Two design choices behave like filters and deserve explicit justification in a paper [HYP]:**
  1. Merging episodes of a pair within 1 s turns weaving or repeated contact into one sample: 7,384 candidates merge ≥ 2
     episodes, up to 8+.
  2. The independence span of 4.8 s with a shared player declares a player's A–B and A–C interactions dependent whenever they are
     less than 4.8 s apart: 24.3 % of successive interactions of a player.

---

## 4. Per-match and per-player distributions [V]

| Unit | Min | Median | p90 | Max |
|---|---|---|---|---|
| Stage 2 candidates per match | 663 | 944 | — | 1,277 |
| Stage 3 samples per match | 541 | 778 | — | 1,078 |
| 0.5 m sync episodes per match | 1,049 | 1,545 | — | 2,350 |
| Stage 3 samples per unique player (555 players appear) | 1 | 74 | 202 | 361 |
| Stage 3 samples per player-match identity (970 appear) | 1 | 51 | 97 | 182 |
| Stage 3 samples per player pair (6,939 pairs) | 1 | — | — | **63** |

### Repetition

- 5,968 of 9,686 candidate pair-periods have more than one candidate, and those account for 26,143 candidates.
- **440 pairs have ≥ 10 candidates and supply 6,349 candidates (21 %).**
- The most repeated pairs (59, 53, 46, … per half) are **same-team pairs that are ~100 % ESTIMATED in the gap**, with median
  d_min ≈ 0.1–0.3 m.
  - Example: match 10502, period 2, slots 6/23: 59 candidates, est 1.00.
  - [HYP] These are provider-filled positions of two off-camera players drifting together, not 59 genuine football interactions.
    They are distinct in time, but probably not distinct as *visual* events, and they are the population on which models score
    best (ESTIMATED error 3.1 % vs VISIBLE 14.7 %).
- **Same pair, multiple events:** yes. Repeats > 1 s apart are separate candidates, so they are genuinely distinct in time.
  Repeats within 4.8 s fall into one overlap group, so they are not counted as independent.

---

## 5. Assessment of the 0.5 m threshold and broader radii

### 5.1 Available counts

Exact for ≤ 0.5 m; for larger radii a *different* encounter definition from Step 1 is used [V].

Step 1 uses the FOOTBALLTrace window rule, not synchronous episodes, so its counts are comparable only *between radii*, not with
§3.2 (`encounters/catalogue.py:1-15`).

| Step 1 radius | Encounters | Same team | With a third player within R | Median of per-match median duration |
|---|---|---|---|---|
| 1 m | 26,629 | 9,853 | 9,837 (37 %) | 2.19 s |
| 2 m | 90,042 | 27,322 | 46,027 (51 %) | 2.57 s |
| 3 m | 154,857 | 49,015 | 103,554 (67 %) | 3.10 s |
| 4 m | 213,908 | 73,180 | 170,795 (80 %) | 3.65 s |

**[EST] Projected chain-based Stage 2 candidates at larger R:**
- Step 1 grows ×3.4 from 1 m to 2 m. Synchronous episodes grow about ×2 per doubling below 0.5 m.
- A 1 m Stage 2 run would plausibly yield ~50–70 k merged candidates.
- A 2 m run would plausibly yield ~150–250 k merged candidates.
- **Independent counts will grow much less.** Players already have successive 0.5 m interactions with a median spacing of 22.6 s,
  and denser interactions per player collide under the 4.8 s per-player overlap rule.
- **[NV]:** exact values need a Stage 2 rerun at new radii (CPU; the 0.5 m run took 68 min for 40 matches).

### 5.2 What a larger radius changes

| Aspect | Effect of increasing R | Evidence |
|---|---|---|
| Episodes / events | Superlinear growth | §3.2, §5.1 [V] |
| Duration | Longer (median 2.2 → 3.6 s from 1 to 4 m) | [V] |
| Multi-player share | Rises steeply (37 % → 80 % with a third player) | [V] |
| Identity ambiguity | Falls: at 1.5–2 m the trajectories are separated by more than a body width, and the nearest-endpoint rule increasingly suffices | [HYP]; testable: physics-baseline BA by distance band |
| Proximity vs occlusion | FIFA tracking has no image, so it cannot show occlusion. SoccerTrack evidence: 97–98 % of severe image occlusions occur at ground distance > 0.25 m (median 1.7–2.9 m, perpendicular 0.15–0.19 m) [DOC: `M11_VIDEO_GROUNDED_OCCLUSION_RESEARCH.md`] | So **a small ground radius has low recall for camera occlusion**, and occlusion depends on camera geometry, not ground distance alone |
| False "proximity events" | More pairs that are merely near, for example in defensive lines | [HYP] |
| Compute | Linear in candidates; candidate generation is O(pairs × time) with bounding-box pruning (`candidates.py:173-174`) | [V] |

### 5.3 Recommended threshold methodology [REC]

1. **Do not choose R by sample count.**
2. **Define the target first:**
   - *(a) identity-ambiguous encounters*, where motion alone could confuse a tracker;
   - *(b) camera-occlusion candidates*, which need camera geometry and are not definable from FIFA alone.
3. **For (a):** catalogue at R ∈ {0.25, 0.5, 1.0, 1.5, 2.0} m and report, per band:
   - nearest-endpoint and constant-velocity baseline accuracy;
   - learned-model accuracy;
   - third-player share and duration.

   Choose the band where the physics baseline is no longer near ceiling. The inclusion criterion is then *empirical ambiguity*,
   not distance.
4. **Report sensitivity.** All results by distance band, with the primary analysis pre-specified.
5. **Keep the catalogue (all R) separate from the supervised task population** (a subset with a stated rule).

---

## 6. Multiple intersections per window, repeated and overlapping interactions

### 6.1 What the current code does [V]

| Mechanism | Where | Behaviour |
|---|---|---|
| Synchronous episodes | `intersections/proximity.py` (sync-proximity v2) | Every pair, every episode ≤ R; repeated windows recorded as `repeated_window_detections` |
| Stage 2 per-pair merge | `candidates.py:64,180` | Episodes of one pair ≤ 1 s apart → one candidate (weaving collapses) |
| Shared-player link | `shared_player_candidates`, `shared_player_s = 1.0` | Counts other candidates sharing a player within 1 s: **5,620 / 29,861 (18.8 %) have ≥ 1**, up to 7+ |
| Dependency | `dependency()` | Union-find over accepted candidates that share a player and are < 4.8 s apart → overlap groups (16,935); greedy independent subset |
| Window intersections (geometric path crossings in 5 s windows) | `window_intersections/core.py`; outputs `m11-window-intersections-v1` | **First intersection per pair-window only.** Over 32 matches: 581,786 intersections, 338,506 events; **246,532 pair-windows have multiple intersections; 243,280 intersections are never primary** |
| Linked events | `n_linked_events`, `linked_event_id` | Links candidates to window events (metadata) |

### 6.2 What is missing

- **Group events:** no representation of an A–B–C group and its formation or dissolution.
- **Group-level ordering:** no ordering of interactions inside a group.
- **Labels and features beyond two players:** no label or feature describes the third player. In a Stage 3 sample the nearby
  third player is neither hidden nor provided to the model.
- **Within-window crossings:** every crossing after the first in a pair-window is dropped from the window layer.

### 6.3 Proposed representation [REC]

A **temporal interaction graph** per match period.

- **Nodes:** player-chain segments (player id, provider chain id, time span), so identity breaks are explicit.
- **Pairwise interaction edges** (one per synchronous episode at the *catalogue* radius):
  - `t_start`, `t_end`, `t*`, `d_min`, duration;
  - crossing flag and crossing times (all, not only the first);
  - relative speed and heading;
  - status composition (VISIBLE / ESTIMATED), as audit metadata;
  - quality flags;
  - `episode_id`.
- **Deduplication rule:** an edge is the maximal synchronous interval with d ≤ R. Window detections map many-to-one onto edges
  (already checked in sync-proximity: `every_detection_maps_to_one_event`). **Count edges, never windows.**
- **Repeated pair contact:** consecutive edges of one pair separated by less than a merge gap form an *edge chain* (weaving).
  Keep both the chain and its member episodes.
- **Group events:** connected components of the graph restricted to edges overlapping in time (±δ) → `group_id`, membership over
  time, formation and dissolution times, size, and the ordered list of edges.
- **Ambiguity:**
  - boundary uncertainty is recorded as the d_min margin to R and the closeness of the status to ESTIMATED;
  - an event is `uncertain` if its existence flips under ±0.05 m or ±1 provider step.
- **Independence for statistics:** the match is the unit for CIs. Within a match, group events (not edges) are the unit for
  "independent encounters". Edges are the unit for the pairwise catalogue.

This preserves every interaction, never counts a window twice, and turns the current two-player candidates into one
**projection** (pair edges with no concurrent third-player edge).

---

## 7. From two players to N players

### 7.1 How the binary task differs [V/HYP]

**The current task:**
- p(SWAPPED) for a 2×2 problem with a known one-to-one structure;
- both players hidden over the same interval;
- both outgoing tracks present.

**The general problem:**
- N identified incoming tracks and M anonymous outgoing tracks (M ≠ N possible);
- unmatched entries on both sides (missed, left the area, spurious);
- staggered disappearance and reappearance times;
- a solution space of partial injections, which grows factorially.

### 7.2 Methods

| Method | Role | Notes |
|---|---|---|
| Pairwise compatibility score s(i, j) | Core building block | Learned or physical |
| N×M cost matrix + **Hungarian** (with dummy rows/columns for null assignment) | Global one-to-one assignment | Standard in MOT; needs calibrated costs and a null cost |
| **Kalman** prediction (constant velocity / constant acceleration) with Mahalanobis gating | Classical baseline; principled uncertainty growth with gap length | Not implemented in M11 yet [V] |
| Learned pairwise matching | Scores per link | |
| Bipartite graph matching / min-cost flow | Generalises Hungarian to multiple time steps | |
| Multi-hypothesis association (MHT) | Keeps the k-best assignments across successive events | Heavier; useful when groups chain |
| Explicit null / unmatched | Required for M ≠ N | |
| Confidence-based abstention | Per-assignment margin (best − second-best total cost) or marginal probabilities | Generalises the current max(p, 1 − p) |

### 7.3 Can the GRU or HGB be reused? [V architecture / HYP performance]

- **GRU C.** The pair head `head([a, b, a·b, |a−b|, gap])` is already a per-link score s(i, j) (`models.py:171-187`). The binary
  logit is s01 + s10 − s00 − s11.
  - It was trained only through this *difference*, so each s(i, j) is identified only up to per-row and per-column offsets. **The
    absolute link scores are not calibrated and cannot serve directly as N×M costs or as null-assignment thresholds.**
  - Reuse requires retraining with a per-link objective, for example a softmax over each row including a null column.
  - The encoders (incoming / outgoing) are reusable as initialisation.
- **HGB C.** Its features are permutation *differences* of per-link costs (`motion.py:153-165`), so the model is intrinsically
  2×2. The underlying per-link costs (`pair_costs`, `pair_matrices`) are reusable. A new per-link classifier (link valid or not)
  on link-level features would be needed.
- **SVM / LSTM.** The same reasoning applies as for HGB and the GRU respectively.
- **Conclusion:** **a binary two-player model does not generalise automatically.** Components can be reused, but the N-player
  models must be retrained and evaluated as new models.

### 7.4 Staged progression [REC]

| Stage | Data | Labels | Algorithms | Metrics | Feasibility / risk |
|---|---|---|---|---|---|
| **A. Correct and fully benchmark two-player recovery** | Existing Stage 3 (+ distance-band extension) | Provider chains (verified) | GRU, LSTM, SVM, HGB, Kalman+Hungarian (2×2), physics | BA, log loss, ECE, risk-coverage, by stratum | High feasibility; moderate GPU |
| **B. Catalogue all pairwise interactions** | 32 dev matches, temporal interaction graph at R ∈ {0.25…2} m | None (descriptive) | Edge extraction, deduplication, sensitivity | Counts, durations, sensitivity, inter-annotator agreement if any reviewed | High feasibility (CPU); risk of scope creep |
| **C. Group overlapping interactions** | Graph from B | Group membership from the provider (descriptive) | Connected components / temporal clustering | Group size and duration distributions; stability under δ, R | Medium; definitional choices dominate |
| **D. N incoming / M outgoing association** | Simulated hiding of k ∈ {2, 3, 4} group members with staggered gaps, plus distractor and missing tracks | Provider chains | Per-link scorers (retrained GRU / LSTM / HGB / SVM), Kalman costs, Hungarian with null, MHT | Assignment accuracy, per-link precision/recall, IDF1-style, null-detection F1, risk-coverage | Medium-low; new label logic, new sample builder, new evaluation; large design space |
| **E. Validation** | Controlled (D) → held-out (once) → real video pilot (§12) | Provider / human annotation | Frozen models | As above + MOT identity metrics | Video requires rights, calibration and annotation; long |

---

## 8. VISIBLE / ESTIMATED quality-control policy

### 8.1 Current policy [V]

- **Eligibility is status-blind.** `label_quality` docstring: "Status … is never used for the outcome" (`candidates.py:199-202`).
- **Status is never a model input.** It is kept as audit metadata: `est_frac_*`, `status_in_gap`.

### 8.2 Existing checks [V]

| Check | Implementation | Action |
|---|---|---|
| Speed > 12 m/s with step ≥ 1 m | `IMPLAUSIBLE_JUMP` (`encounters/quality.py:11,43,74-75`) | Flag; chain break; jump-transfer risk |
| Sub-metre over-speed | `OVERSPEED_STEP` | Flag only (kept) |
| Frame-wide shift | `FRAME_WIDE_SHIFT` | Flag |
| Identity discontinuity / label transfer / reciprocal exchange | mask bits + LOWER outcome | Downgrade / reject |
| Duplicate positions (cloned tracks) | `duplicate_position_samples`; window layer excludes them; framewise unmatched 1,461 | Exclude from chains |
| Held positions, repeats, substitution windows, side over-count | mask bits | Flag / LOWER |
| Timestamps strictly increasing; period boundaries | Window layer `timestamps.strictly_increasing`; `window_in_period` | Reject when violated |
| Chord-speed plausibility across the sample window | `chord_speed_plausible` (12 m/s) | Reject (1,197) |
| Continuous provider chain | `continuous_reference` | Reject (3,892) |
| Near-coincidence (d_min < 0.05 m) | `near_coincidence` flag | Metadata only |

### 8.3 Missing checks [V by search]

- **Out-of-pitch coordinates:** no bounds check found in `encounters/` or `window_intersections/`.
- **Acceleration plausibility:** acceleration is computed but declared "diagnostic only"; raw p99 is 89 m/s² in 3812 because of
  update-step noise.
- **Per-player adaptive speed limits:** a per-player robust high quantile instead of a fixed 12 m/s.
- **A candidate-level flag for clone-like pairs:** long same-team ESTIMATED co-motion (§4).
- **Smoothing-aware kinematics:** derivatives from a local polynomial fit over ≥ 3 updates, not raw finite differences.

### 8.4 Recommended policy [REC]

1. **Keep both statuses eligible.** Never use status as a feature or an exclusion rule.
2. **Separate eligibility from reliability.**
   - Eligibility: identity continuity, timestamp validity, in-bounds, no hard corruption.
   - Reliability: a per-sample score from status, update cadence, local residual to a smooth fit, and flags. It is stored and used
     only for stratified reporting and possibly for weighting experiments, not for selection.
3. **Robust anomaly detection.**
   - Fit a local quadratic or Savitzky–Golay curve to provider updates.
   - Flag points whose residual exceeds k·MAD and whose implied acceleration exceeds a per-player quantile.
   - Require two independent indicators before treating a point as corrupt, so that genuine sprints and sharp cuts are not
     deleted.
4. **Bias measurement experiments** (no eligibility change):
   - report all metrics by status composition (both VISIBLE / mixed / both ESTIMATED), with match-level CIs;
   - texture-neutralisation tests: smoothing or jittering ESTIMATED to VISIBLE-like residual spectra at *test* time and at
     *train* time;
   - texture-only probe per model family (already BA 0.70 for HGB-style features [DOC]);
   - train-VISIBLE-only / test-VISIBLE as a reference model;
   - check whether the clone-like repeated pairs (§4) inflate scores: evaluate with and without them, and with
     pair-level-deduplicated weighting.

---

## 9. Full GRU / LSTM / SVM / HGB benchmark review

**Common setup (Stage 4A/4B)** [V: `m11_benchmark/run.py`, `models.py`, `metrics/g0p80/v2/comparison.md`, `PROGRESS.md`]:
- Task: the 2×2 MATCHED/SWAPPED on `g0p80`, centred, gap 0.8 s.
- Folds: 8 match-grouped outer folds.
- Metrics: independent subset (18,777).
- Neural models: 3 early-stop matches per fold, 80-epoch cap, patience 10.
- Tabular models: trained on both outgoing orders with symmetric prediction.

| Model | Input | Architecture / size | Tuning | Seeds | Budget / convergence | Indep. BA (mean ± sd) | Log loss | ECE | Stress-tested? |
|---|---|---|---|---|---|---|---|---|---|
| **GRU gap-last** | (2, 30, 4) per side + gap | GRU 48 → 48, pair head; **43,585** params | **none** (fixed lr 1e-3, sizes) | 3 | Best epoch 76–80/80 in **23 of 24** fold fits (budget-bound) | **0.9021 ± 0.0045** | 0.228 | 0.008 | Yes (Stage 4B robust, 4C, M2) |
| **LSTM gap-last** | same | LSTM 40 → 48, same head; **41,985** params | **none** | 3 | Best epoch ≥ 71/80 in 22 of 24 fold fits (exceptions: s1 f0 57, s1 f7 39) | **0.8570 ± 0.0014** | 0.311 | 0.008 | **No** |
| GRU / LSTM (far-end summary) | same, summary at far end | as above | none | 1 | Stopped early (16–78) | 0.765 / 0.737 | 0.450 / 0.485 | — | No (superseded design) |
| MLP | flattened (2 × 30 × 4) | 96 → 48; 55,873 params | none | 3 | 41–80 | 0.8447 ± 0.0020 | 0.335 | 0.011 | No |
| **SVM** | 36 motion features | RBF approximated by Nystroem (1,000 components) + LinearSVC; Platt-type scaling of the decision function | 6-point grid (gamma, C) by inner 3-fold match CV | 3 (≈ identical) | Deterministic fit | **0.8587 ± 0.0002** | 0.307 | 0.009 | **No** |
| **HGB** | 36 motion features | HistGB, leaf 40, l2 1.0 | 4-point grid by inner CV | 3 (identical) | No early stopping; 200/400 iterations | **0.8695** | 0.285 | 0.010 | Yes |
| RF | 36 features | 300 trees, leaf 5 | 2-point grid | 3 | — | 0.8574 | 0.331 | **0.062** | No |
| LR | 36 features | standardised logistic | 4-point C grid | 1 | — | 0.8373 | 0.344 | 0.010 | No |
| KNN | 36 features | k ∈ {15, 31, 63} | grid | 1 | — | 0.8224 | 0.373 | 0.028 | No |
| Physics combo / best single | 8 baseline margins | LR on margins | inner CV | 1 | — | 0.6958 / 0.6462 | 0.569 / 0.597 | — | Yes (combo) |
| Single physics rules | — | fixed | — | 1 | — | 0.35–0.61 | ≈ 0.693 | — | — |

**After Stage 4B, only GRU and HGB continued** (selected by `robust/selected_models.json` on the same OOF results) [V]:
- Stage 4C gave the GRU 160 epochs and config C, reaching BA 0.926 centred.
- HGB C got config-C weighting (0.861 centred).
- Milestones 1–3 use only these two.

**Other notes:**
- **Inference cost:** not measured systematically [NV]. Training wall time from logs: LSTM gap-last 105–305 s per fold; GRU
  gap-last of the same order (seed 0 total 819 s vs LSTM 1,147 s).
- **Physics baselines:** a "nearest" rule at BA 0.354 means the closest endpoint is *anti*-informative under this population
  (players continue past each other). This is useful evidence that the task is not trivial at 0.5 m [V].

### 9.1 Was GRU vs LSTM conclusive? No [V + HYP]

**Verified facts:**
- equal budget (80 epochs), equal training data, equal folds and seeds;
- roughly equal capacity (43.6 k vs 42.0 k parameters);
- the same gap-last orientation and the same masking (summary at the last valid step; masked steps are fed as zeros to both);
- the same augmentations;
- GRU leads by 0.045 BA, consistently across all 8 folds and 3 seeds.

**Why it is not conclusive:**
1. **Both were budget-bound at 80 epochs.**
   - The GRU later gained +0.018 BA (0.902 → 0.920 for config A at 160 epochs [DOC]).
   - The LSTM was never given a longer budget. LSTMs with the same lr often converge more slowly [HYP].
2. **No hyperparameter search for either network** (learning rate, hidden size, depth, dropout), while tabular models were tuned.
3. **No config C (shift) training for the LSTM**, so the post-4C comparison is GRU C vs nothing.
4. **The LSTM was not evaluated under gap, noise or texture stress.**
5. **One architecture variant each.** No bidirectional encoder, no layer normalisation, no gradient diagnostics.

**Defensible statement today:** "Under an identical 80-epoch budget without tuning, the gap-last GRU outperformed the gap-last LSTM
(0.902 vs 0.857 BA); both were still improving when stopped."

### 9.2 Fairness assessment

| Issue | Status |
|---|---|
| Same encounters, folds, labels | Fair [V] |
| Tuning budgets | **Unfair:** tabular grids by inner CV, neural models none |
| Convergence | **Unfair:** neural budgets binding; only the GRU was extended |
| Training distribution | **Unfair after Stage 4C:** only GRU and HGB got shift training |
| Model selection | On the same OOF results that are reported (selection-optimistic) |
| Stress evaluation | Only GRU / HGB / physics |
| Feature access | Tabular models see engineered physics features; sequence models see raw sequences. This is a legitimate design contrast but must be stated |

### 9.3 Future fair benchmark protocol [REC]

1. **Splits.** Fixed outer folds by match (keep the current 8). Inside each outer training partition:
   - a *tuning* inner split (3 match groups);
   - a separate *calibration* match group for thresholds;
   - early-stop matches drawn inside the tuning split.
2. **Equal tuning budget per family**, for example 24–32 configurations each by random or Bayesian search on inner folds.
   - Neural: learning rate, hidden size, layers {1, 2}, dropout, weight decay, batch, epoch cap up to 400 with patience.
   - SVM: C, gamma, Nystroem size, or an exact RBF on a subsample.
   - HGB: learning rate, leaves, iterations, min-leaf, l2.
3. **Convergence check.** Report best-epoch / cap per fit. A fit at the cap triggers a budget increase, or is flagged.
4. **Seeds.** ≥ 5 for neural models; report fold × seed variance.
5. **The same training distributions** (centred, shift-augmented, variable-gap) offered to all families.
6. **The same metrics**, with match-bootstrap CIs and paired differences:
   - BA, log loss, Brier, ECE;
   - risk-coverage with thresholds fitted on calibration matches;
   - stress-suite and status strata.
7. **A compute ledger:** GPU-hours and CPU-hours per family.
8. **For N×M:** every family is converted to a **per-link scorer** under the same assignment solver and null handling.
   - Neural encoders score link (i, j) from both tracklets.
   - SVM / HGB score link-level features.
   - Kalman gives Mahalanobis costs.

   The comparison is then of scorers, not of incompatible output formats.

---

## 10. Training convergence and dataset-adequacy diagnosis

| Candidate cause | Evidence | Verdict |
|---|---|---|
| Too little data | GRU learning curve: 0.742 @ 4 matches → 0.883 @ 24 → 0.900 @ 28; still rising. HGB saturated (0.853 → 0.870) [DOC §8]. **Confound:** with fewer matches at a fixed 80 epochs, the GRU also gets fewer gradient steps | **Partly**, for sequence models; not for HGB |
| Too narrow a task | One gap, ≤ 0.5 m, 2×2, simultaneous hiding; models fail off-distribution (M2) | **Yes**, main limitation for generality |
| Poor labels | Labels verified against raw chains; provider identity itself unverified; 3,301 candidates with d_min ≤ 0.05 m, partly clones | Labels internally consistent; **external label correctness unknown** |
| Poor convergence | Binding budgets (GRU 4B, GRU C 13/24 fits, LSTM 4B) | **Yes** |
| Weak architecture | Gap-last fix +0.14 BA; no search done | Unknown |
| Overfitting | GRU train/val LL 0.20/0.23; HGB 0.21/0.29 | Not dominant |
| Domain shift | Gap, noise, dropout, context (M2) | **Yes**, for deployment claims |
| Provider-interpolation bias | Texture probe 0.70; VISIBLE error ×4.7 | **Yes, serious** |
| Unfair comparisons | §9.2 | **Yes** |
| Evaluation methodology | Match-grouped, independent subset (good); in-sample thresholds; selection on reported OOF | **Partly flawed** |

### 10.1 Growth that is scientifically meaningful [EST / REC]

| Expansion | Expected new rows | Expected new *independent* evidence | Value |
|---|---|---|---|
| Radius 0.5 → 1 m | ~+20–40 k candidates [EST] | Moderate (new pairs, but more multi-player overlap) | Moderate: tests the ambiguity band |
| Radius → 2 m | ~+120–220 k [EST] | Lower per row (80 % with third player at 4 m) | Mostly easy or multi-player cases |
| Variable gaps 0.2–2.4 s | ×k variants of the same encounters | **None** (same encounters) | High for robustness, not for independence |
| Multiple interactions per window / within-window crossings | Up to the 243 k non-primary intersections are *geometric* crossings, mostly asynchronous | Small for the synchronous task | Matters for the catalogue |
| Multi-player groups | Group samples from existing edges | New task, new evidence type | High novelty, high risk |
| More matches | Only 32 development matches remain clean; 8 held-out reserved; 24 official val/test consumed by an earlier track (re-use would need an explicit decision) | Each match adds ~780 samples / ~580 independent | The strongest lever for independent evidence, but none is available without policy decisions |

**Bottom line:** the number of *independent matches* (32) bounds the statistical power far more than the number of rows.
Expanding the rows by widening R or adding variants does not add independent matches.

---

## 11. Real World Cup video validation: feasibility plan [REC; nothing done]

1. **Rights.** Broadcast footage is owned by FIFA and the broadcasters. Research use needs a licence or a clear exception; check
   institutional legal advice. The PFF release contains tracking, events and metadata, but **no video** [V: data inventory].
2. **View type.**
   - Broadcast: a moving, zooming camera; players leave the frame.
   - Tactical / wide: better, rarely public.
   - Fixed or panoramic: best, but not available for World Cup matches.
   - Identify the type before planning.
3. **Detection:** person detection (YOLO-class) with pitch masking; the earlier Track B experience shows that turf filtering
   matters [DOC: memory, Session B].
4. **Multi-object tracking:** ByteTrack-class baseline; identity switches, fragmentation and ID recovery as targets.
5. **Camera motion and calibration:** per-frame homography from pitch lines (broadcast requires frame-wise estimation). Expect
   0.4–1 m metric error [DOC: earlier CV milestones]. **This is far outside the current models' supported noise (≤ 5 cm).**
6. **Ground truth for occlusion:** human annotation of the occlusion interval (start/end), the players involved and the correct
   identity after reappearance; two annotators on a subset for agreement.
7. **Ground-truth identity:** jersey numbers and team; cross-check with PFF tracking *only if* the match and time can be aligned.
8. **Synchronisation:** PFF tracking has a video-time field relative to *their* source video. A public broadcast may differ in
   start offset, cuts, replays and frame rate. **Alignment must be verified** with events (kick-offs, goals) and positions; do not
   assume it.
9. **Video vs provider trajectories:** after calibration and alignment, measure position error distributions and status
   agreement.
10. **Metrics:** IDF1, ID switches, fragmentation, HOTA association accuracy; occlusion-recovery accuracy per annotated event;
    risk-coverage of abstention.
11. **Off-screen players.** Broadcast views often lack the players needed for 2 s of context on both sides. Many events will be
    unusable, and that must be reported.

**Concepts that must not be conflated:**

| Concept | Definition |
|---|---|
| Spatial proximity | ground distance ≤ R |
| Crossing trajectories | ground paths intersect, possibly at different times |
| Bounding-box overlap | image IoU > 0 |
| Visual occlusion | one player's pixels are hidden by another |
| Tracking loss | the tracker emits no box |
| Identity switch | the tracker's identity changes owner |

**FIFA tracking alone provides only the first two.**

**Minimum pilot:**
- one match segment of 10–15 minutes, with rights cleared;
- 30–50 human-annotated occlusion or ambiguity events;
- the tracker run, the pitch calibrated and the trajectories exported in `m11-trajectory-input-v1`;
- `m11_infer` run with the declared noise (expected: mostly "unsupported");
- a report of feasibility, not accuracy.

---

## 12. Single model vs multi-model systems

Ratings are qualitative.

| Option | Evidence-based accuracy (2×2, 0.8 s) | Gap / missing / noise robustness | Calibration | Selective risk | Runtime | Interpretability | Stability / maintenance | Assessment |
|---|---|---|---|---|---|---|---|---|
| A. GRU only | Best (0.902–0.926) | Poor off-gap and with noise | Good in-distribution | Good at 0.8 s only | Fast (43 k params) | Low | Medium | Strong in-distribution, fragile |
| B. LSTM only | 0.857 (under-trained) | Untested | Good (ECE 0.008) | Untested | Fast | Low | Medium | **Unknown until fairly re-run** |
| C. SVM only | 0.859 | Untested | Good | Untested | Feature cost dominates | Medium | High | Reasonable baseline |
| D. HGB only | 0.861–0.870 | Best gap transfer; fails at large shift | Good | Moderate | Feature cost dominates | Medium (features) | High | Robust baseline |
| E. Kalman + Hungarian | Not measured [NV] | Principled uncertainty growth with the gap | Needs tuning | Natural (gating) | Very fast | High | High | **Required baseline; may be surprisingly competitive off-distribution** [HYP] |
| F. GRU/HGB routed hybrid | 0.921 at baseline; 8 % routed error under stress | Routing helps for gap only | Mixed | Fails the 5 % target | Both models | Medium | Low (two models + router) | Current system; needs a shift-aware router |
| G. Multi-model ensemble (GRU, LSTM, SVM, HGB) | Nested ensemble picked GRU-only in 24/24 [DOC] | Possibly better under shift | Needs recalibration | Possibly better | Highest | Low | Lowest | Justify only if fair evidence shows a gain |
| H. Learned pairwise scorer + global assignment | N/A for 2×2 (equivalent) | Depends on training | Needs per-link calibration | Natural via assignment margin | Moderate | Medium | Medium | **The only option that scales to N×M** |

**Recommendation [REC]:**
- The paper compares A–E (plus physics) fairly.
- Deployment: choose the simplest option within the CI of the best on the *preregistered* criteria (risk at a target coverage
  under the stress mix).
- H is the architecture to build if Stage D proceeds.

---

## 13. Major publication risks

1. **Simulated interruptions only:** reviewers will ask for real occlusions (§11).
2. **Provider texture:** results may reflect PFF interpolation, not football motion. This must be addressed head-on.
3. **Provider labels as ground truth:** identity errors in the source are possible.
4. **Selection on reported data and in-sample thresholds.**
5. **Unfair model comparison** (§9.2).
6. **Narrow population** (≤ 0.5 m) without sensitivity analysis.
7. **Few independent units:** 32 matches; CIs must be match-level.
8. **Novelty:** MOT association (Kalman/Hungarian, learned affinities) is mature. The novelty must come from the controlled
   football benchmark, robustness and provider-artifact analysis, and selective prediction under shift. **A literature review is
   still outstanding.**
9. **Licensing:** PFF data terms were not verified [NV]. Video rights are a separate issue.
10. **Scope creep:** a full N-player system plus video would delay any defensible result.

---

## 14. Recommended research direction

**First paper (recommended):** *Two-player trajectory-only identity re-association across tracking interruptions, within a complete
pairwise interaction catalogue.* It contains:

- **the catalogue:** a temporal interaction graph on 32 matches, with threshold sensitivity, repeated and multi-player statistics
  and status composition;
- **the task:** the 2×2 benchmark with distance bands, variable gaps and status strata;
- **the benchmark:** GRU, LSTM, SVM, HGB, Kalman+Hungarian and physics under equal tuning and convergence budgets and ≥ 5 neural
  seeds;
- **the analyses:**
  - provider-texture analysis;
  - selective prediction with thresholds fitted on calibration matches;
  - one preregistered held-out evaluation.
- **Multi-player:** as a descriptive section (group statistics) plus at most a small controlled 3-player pilot.

**Second paper or extension:** N×M association with null assignments and per-link scorers, then video validation.

This is the minimum path that is novel enough to be interesting (catalogue + texture + fair benchmark + selective risk), feasible on
the available data, and defensible.

---

## 15. Prioritised roadmap with decision gates

All phases use only the 32 development matches. **The frozen held-out set stays untouched until Phase 7.**

| Phase | Objective | Dependencies | Deliverable | Scientific benefit | Compute / annotation cost | Risks | Existing results valid? | Training? | GPU approval? |
|---|---|---|---|---|---|---|---|---|---|
| 0 | Housekeeping: commit M1–M3 (user); fix documentation errors from the predecessor audit | — | Commits; errata | Integrity | None | None | Yes | No | No |
| 1 | **Interaction catalogue v1**: temporal interaction graph at R ∈ {0.25, 0.5, 1, 1.5, 2} m; all crossings per pair-window; group components; deduplication rules | Gate 1 | New `m11-interaction-graph-v1` + report | Answers "how many events exist"; basis for the threshold choice | CPU, ~1–3 h for 32 matches [EST] | Definitional disputes | Yes (new version) | No | No |
| 2 | Threshold and ambiguity study: physics and existing-model accuracy by distance band, using existing checkpoints on new development candidates | 1, Gate 2 | Report + chosen supervised population | Evidence-based R | CPU | Choosing R post hoc (pre-specify the criterion) | Yes | No | No |
| 3 | Quality v2: out-of-bounds, robust kinematics, clone-pair flag, reliability score (metadata only) | 1 | Flags + report | Cleaner labels without status bias | CPU | Over-flagging | Yes | No | No |
| 4 | Two-player dataset v2 (Stage 3 builder at the chosen R band, variable gaps, status strata) | 2, 3, Gate 3 (for a multi-player pilot) | `m11-stage3-samples-v2` | Broader, documented population | CPU | Leakage via new rules (pre-specify) | Old results remain valid for the old definition | No | No |
| 5 | **Fair four-family benchmark** + Kalman/Hungarian + physics; equal tuning; ≥ 5 seeds; calibration matches; texture experiments | 4, Gate 4 | Benchmark report | Core paper result | **GPU**: ~dozens of GPU-hours [EST] | Compute; budget creep | Old results become historical | **Yes** | **Yes** |
| 6 | Model / system selection on development (preregistered criterion); selective policy v2 | 5, Gate 5 | Selection memo + frozen protocol v2 | Defensible choice | CPU | Selection bias (nested) | — | Final training | Yes |
| 6b (optional) | 3-player controlled pilot (Stage D) | 1, 5, Gate 3 | Pilot report | Feasibility of N×M | GPU, modest | Scope creep | — | Yes | Yes |
| 6c (optional) | Video pilot (§11) | Rights, Gate 6 | Feasibility report | External validity | Annotation 30–50 events; CPU/GPU for the tracker | Rights, alignment | — | No (inference only) | Yes |
| 7 | **Final held-out evaluation** (once) | 6, Gate 7 | Held-out report | Unbiased estimate | Small | Irreversible | — | No | Yes |
| 8 | Manuscript | 7, literature review, Gate 8 | Draft | — | — | Novelty | — | — | — |

### Go / no-go gates [REC]

| Gate | Before | Criteria |
|---|---|---|
| 1 | Redefining the dataset | Written definitions of edge, chain, group, independence and the catalogue radius set; approved |
| 2 | Expanding proximity thresholds for *training* | The Phase 1–2 sensitivity report shows the band where ambiguity is real; criterion stated before looking at model results |
| 3 | Building multi-player samples | Group statistics show enough k = 3 events with complete chains in ≥ 20 matches; a label rule for staggered gaps is approved |
| 4 | Training new models | Frozen dataset v2 manifest; tuning budgets and compute ledger approved; no held-out access in code paths (test) |
| 5 | Selecting the final architecture | Preregistered criterion; paired match-bootstrap CIs; convergence flags clear |
| 6 | Real video inference | Rights confirmed; view type known; annotation protocol and alignment test passed |
| 7 | Held-out evaluation | Protocol v2 signed and hashed; models and thresholds frozen; analysis script hashed; explicit user approval |
| 8 | Claiming publication readiness | Literature review done; all mandatory baselines; held-out result reported regardless of outcome; licence terms verified |

---

## 16. Questions requiring user approval

1. **Paper scope:** two-player first paper with a catalogue and a descriptive multi-player section (recommended), or a full N-player
   formulation now?
2. **Catalogue radii:** approve {0.25, 0.5, 1.0, 1.5, 2.0} m for the *descriptive* catalogue, with the supervised radius chosen by
   the pre-specified ambiguity criterion?
3. **Event definitions:**
   - approve the temporal interaction graph (edges = synchronous episodes; chains; groups);
   - recording of all crossings per pair-window.
4. **Independence unit:** group events within a match, and the match for CIs?
5. **Quality v2:** add out-of-bounds, robust kinematics, a clone-pair flag and a reliability score, all as metadata, with
   eligibility unchanged for both statuses?
6. **Benchmark budget:** GPU hours for a fair GRU / LSTM / SVM / HGB benchmark with equal tuning, ≥ 5 seeds and longer epoch caps?
7. **Official validation/test (24 matches):** keep them permanently excluded, or decide explicitly whether they may become extra
   development data (they are consumed only for the old RF track)? This is a scientific-policy decision; the default is to keep
   them excluded.
8. **Video:** which match and footage source do you intend to use, and are the rights clarified?

---

## 17. Direct answers

**Is the current dataset too small, or merely narrowly defined?**
Narrowly defined.
- At 0.5 m, the 32 matches contain about 49.6 k proximity episodes → 29.9 k merged encounters → 25.3 k labelled samples → 18.8 k
  independent ones. This is close to the realistic count for that definition [V].
- Rows in the hundreds of thousands exist only as frame or window detections (1.06 M / 315 k) or at about 2 m radii [V/EST].
- The binding statistical limit is **32 independent matches**, not rows.

**Are our event-extraction rules missing important football interactions?**
Partly, by design.
- Interactions between 0.5 m and ~2 m: 90 k Step 1 encounters at 2 m vs 26.6 k at 1 m.
- Every crossing after the first within a pair-window: 243 k intersections.
- Group structure: 19–37 % involve a third player.
- 10.5 % of close episodes are excluded for lack of a continuous provider chain.

None of this is a bug. But the paper must report it, and the catalogue should include it [V].

**Can our existing models generalise beyond two players?**
Not as they are. Their outputs are 2×2 antisymmetric logits whose per-link scores are uncalibrated. The encoders and the motion
costs are reusable components for a retrained per-link scorer inside an assignment solver [V architecture / HYP performance].

**Was the historical GRU-versus-LSTM comparison conclusive?**
No.
- Both were budget-bound at 80 epochs, and neither was tuned.
- Only the GRU later received 160 epochs and shift training.
- The LSTM was never stress-tested.

The GRU's 0.045 BA lead at equal budget is consistent, but it is not proof of superiority [V].

**Should we expand the dataset before retraining?**
Partly.
- First define and build the catalogue and run the threshold and ambiguity study (no training).
- Then freeze dataset v2 and run *one* fair benchmark on it.

Retraining now on the old definition would have to be redone [REC].

**What is the minimum scientifically defensible path toward a strong paper?**
1. Catalogue.
2. Evidence-based population.
3. Quality v2 as metadata.
4. Fair four-family + Kalman/Hungarian benchmark with calibration-match thresholds and texture analyses.
5. Preregistered selection.
6. One held-out evaluation.
7. Literature review and licence check.

Multi-player association and video come afterwards, as pilots or a follow-up [REC].

---

## Appendix A: what this review ran

All runs were read-only Python aggregations over the train/CV per-match files listed in §3, each asserted to belong to
`split == train_cv` or taken from the frozen split list. Scripts are in the session scratchpad (`r1.py`–`r3.py` and inline
commands), not in the repository. No file in the repository was modified. This report is the only file created.

## Appendix B: Git status after this review

Identical to the predecessor audit §2.1. The new file lies inside the already-untracked `docs/handoffs/`.
