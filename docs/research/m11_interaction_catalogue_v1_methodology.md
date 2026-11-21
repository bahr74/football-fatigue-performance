# Interaction Catalogue v1: methodology and event definitions

**Schema:** `m11-interaction-catalogue-v1`.

**Code:** `src/football_fatigue/interaction_catalogue/` (`core.py` definitions, `run.py` runner and checks, `summary.py`
aggregation).

**Outputs:** `data/processed/m11-interaction-catalogue-v1/`.

**Status:** Session A, Phase 1 (2026-11-19). A descriptive catalogue on the 32 train/CV development matches.

**What this catalogue is not:**

- It is not a training dataset.
- It selects no supervised radius.
- It makes no claim about visual occlusion.

## 1. Purpose

The catalogue builds an auditable foundation for detecting, representing and counting player interactions in FIFA (PFF) tracking.
It does not commit to a two-player or a multi-player model.

Every proximity and crossing event is kept and counted once. The historical 0.5 m benchmark population is reproduced exactly as one
projection of the catalogue (§8).

## 2. Data and protected boundary

- **Matches.** Only the 32 matches in `train_cv` of the frozen `m11-stage2-split-v1/split.json` (SHA recorded in the run manifest)
  are processed.
- **Refusal.** The runner refuses any other match *before anything is loaded* (`run.assert_allowed`, tested).
- **Matches never opened:**
  - the 8 internal held-out matches (3822, 3824, 3834, 3839, 3844, 10510, 10514, 10517);
  - the 24 official validation / test matches (non-development in `m11-encounters-v1/splits.json`).
- **Source.** The checksummed Step 1 RAW catalogue (`matching.benchmark.load_source`, which verifies every file against its
  manifest), plus the pitch size from the match metadata.
- **Coordinate policy.** Raw coordinates only. Provider-smoothed coordinates are never used.
- **Read-only.** Earlier M11 outputs are only read: Stage 2 candidates and window-intersections v1, for the reproduction checks.
- **No overwriting.** Every match directory is created with `exist_ok=False`.

## 3. Trajectories (unchanged from Stage 2)

The trajectories are the valid segments of `window_intersections.core.build_segments`, the same trajectories as the historical
Stage 2 candidates.

- **Segments.** Linear between consecutive 15 Hz provider updates of one player in one period.
- **Never joined across:**
  - a gap > 0.5 s;
  - an implausible jump (> 12 m/s over ≥ 1 m);
  - an unattributed or ambiguous sample;
  - a duplicate-position (bit-identical, cloned) sample;
  - period 0, a quarantined period or a period boundary.
- **Chain.** A maximal run of joined segments. **Nothing is interpolated across a chain break.**
- **Status.** VISIBLE and ESTIMATED samples are treated identically.

## 4. Event types, and what is *not* observable

| Type | Definition | Observable in FIFA coordinates? |
|---|---|---|
| **Proximity edge** (radius R) | Maximal interval with synchronous separation d(t) = \|A(t) − B(t)\| ≤ R on two overlapping chains | Yes (geometric) |
| **Geometric crossing** | The two paths meet in space at times tA, tB with \|tA − tB\| ≤ 5 s | Yes (geometric) |
| Synchronous crossing | A crossing with \|tA − tB\| ≤ 0.3 s | Yes |
| **Edge chain** | Repeated contact of one pair: edges whose gap to the chain so far is ≤ 1.0 s | Yes |
| **Group** (multi-player) | Connected component of edges sharing a player within a link tolerance δ | Yes |
| Candidate identity-ambiguity indicator | Constant-velocity margin `cv_margin_m` (§6) | Descriptive indicator only |
| Visual occlusion, bounding-box overlap, tracker loss, identity switch | Require images | **No.** Never inferred here |

**Proximity edges are found exactly.** Both positions are linear between knots, so d(t)² is piecewise quadratic and the set
{t : d(t) ≤ R} is solved exactly (`window_intersections.candidates.episodes_on`). For each edge the catalogue records:

- start, end, duration;
- t\* (time of the minimum) and d_min;
- the number of local minima;
- truncation by a chain boundary;
- positions at t\*;
- speeds and relative speed (0.4 s central chord);
- relative heading (only when both speeds > 0.2 m/s);
- time within 0.25 m;
- the crossings attached to the edge (§5).

**Radii.** These are descriptive analysis radii, **not** training thresholds: R ∈ {0.25, 0.5, 1.0, 1.5, 2.0} m.

- Edges are nested: every edge at a smaller R lies inside an edge of the same chain pair at the next larger R (checked per match).
- One wider-R edge may contain several smaller-R edges.

**Crossings.**
- All crossings come from `window_intersections.core.period_hits` (linear), as in window-intersections v1, but **none is
  discarded**. The v1 "first intersection per pair-window" rule is not applied.
- Each crossing records:
  - |tA − tB| and kind (proper / endpoint / collinear);
  - the chains involved;
  - `d_sync_m`: the separation between the players when the first of them is at the crossing point;
  - `within_rX`: whether `d_sync_m` ≤ R.
- A crossing is attached to an edge of the same pair when both crossing times lie in [start − 0.5 s, end + 0.5 s].

## 5. Temporal interaction graph

| Element | Key | Content |
|---|---|---|
| Node | `node_id = match:period:slot:chain` | One player chain (an identity break is a node boundary): slot, player id, side, t0, t1, segments |
| Edge | `edge_id` (hash of match, period, pair, R, start, end) | Proximity edge with descriptors, `node_a`, `node_b`, provenance (`chain_a`, `chain_b`) |
| Edge chain | `edge_chain_id` | Repeated same-pair contact; best (lowest d_min) t\*, n_edges, total time within R, crossings |
| Group | `group_id` per (R, δ) | Players, pairs, home/away count, start, end, size; for δ = 0 also the lifecycle |
| Group timeline | (group_id, state) | Membership state changes (components through time), for simultaneous groups of ≥ 3 players |
| Crossing | `crossing_id` | Every geometric crossing |
| Pair-window | (pair, anchor) | Number of crossings (and synchronous crossings) inside each complete 5 s window |

**Event ordering.** Edges have start and end. `order_in_chain` and `order_in_group_d0p00` give the order of edges inside a chain or a
simultaneous group.

**Groups.** Two edges are linked when they share a player and the gap between their intervals is ≤ δ. Groups are the connected
components (per-player sweep with union-find, `core.link_components`).

- δ = 0 gives **simultaneous groups**: edges that overlap in time.
- δ = 1.0 s gives **temporally connected groups**: sequential interactions are included.
- Every edge belongs to exactly one group per (R, δ). This is checked.
- Several edges involving the same player inside one window all remain, each as its own edge, linked through the group.

**Lifecycle (δ = 0).**
- *States.* At every edge boundary and between boundaries, the active edges define components of players. A state change is
  recorded whenever the components change.
- *Counted events:*
  - formation: the first state;
  - join: a player appears;
  - leave: a player disappears;
  - merge: one component formed from two or more earlier components;
  - split: one component becomes two or more;
  - dissolution: the last state.
- *Summary measures:* `max_simultaneous_players` (largest instantaneous component) and `max_concurrent_edges`.

**Deduplication policy (reproducible).**
1. *Count edges, never windows.* A sliding window detector reports an edge once per overlapping 5 s window (`n_windows`). These
   window detections are reported separately (`window_stats`) and never counted as events.
2. Overlapping windows of the same pair are reported as unique pair-windows, a diagnostic only.
3. Repeated contact of a pair is kept as separate edges *and* summarised as one edge chain.
4. Group membership never removes an edge.
5. A crossing is counted once per (pair, tA, tB) after merging hits at shared vertices.

**Interaction context per edge** (other partners only, at the same R):
- `n_concurrent_other`: edges sharing a player and overlapping in time;
- `n_sequential_other`: edges sharing a player, not overlapping, within 1 s.

## 6. Ambiguity indicators (descriptive, not labels)

**`cv_margin_m`.**
- *Procedure.* Hide [t\* − 0.4, t\* + 0.4] s. Extrapolate each player with the chord velocity of the preceding 0.4 s.
- *Value.* margin = (summed endpoint error of the exchanged assignment) − (error of the provider-chain assignment).
- *Tiers:*
  - `cv_confusable` (margin < 0): constant velocity would exchange them;
  - `marginal` (0 to 0.5 m);
  - `cv_separable` (≥ 0.5 m);
  - `unavailable` (a chain lacks 1.2 s of coverage).
- *Status.* The provider chain is the reference. The indicator describes how hard a scenario is for the simplest physical model. It
  is not a prediction, not a label and not a selection rule.

**`boundary_uncertain`.** d_min lies within 0.05 m of R, so the edge's existence depends on centimetre-level coordinate error.

## 7. Quality metadata (three separate concepts; none removes an event)

| Concept | Column(s) | Rule |
|---|---|---|
| Eligibility | `eligible` | Always True: an edge exists only on validated chains. Excluded data are counted per match (`segments`, framewise exclusions) |
| Measurement reliability | `reliability` (nominal / reduced), `reliability_reasons` | See the reasons below the table |
| Possible label corruption | `label_risk`, `label_risk_reasons` | See the reasons below the table |
| Provider status | `est_frac_a`, `est_frac_b`, `status_composition` | Fraction of ESTIMATED update instants during the edge (± 0.1 s). **Audit metadata only. Never affects eligibility, reliability or label risk by itself (tested)** |

**Reasons for `reliability = reduced`:**
- `speed_anomaly`: max smoothed speed > 12 m/s within edge ± 1 s;
- `accel_anomaly`: max smoothed acceleration > 25 m/s²;
- `overspeed_step`: a provider sub-metre over-speed flag;
- `out_of_pitch`: a sample more than 3 m outside the touch or goal lines;
- `estimated_comotion`: both players ≥ 95 % ESTIMATED **and** ≥ 1 s within 0.25 m. This is the clone-like pattern; the co-motion,
  not the status, is the trigger.

**Reasons for `label_risk`:**
- an identity-risk flag (label transfer, reciprocal exchange, discontinuity candidate, substitution window), within edge ± 1 s;
- an implausible jump;
- a frame-wide shift;
- a duplicate-position sample;
- an edge truncated by a chain boundary.

**Notes:**
- *Acceleration threshold.* On the 3816 pilot, 15 m/s² flagged 14 % of edges: median per-edge maximum 7.2 m/s², p99 30 m/s². Cuts
  and turns produce large smoothed accelerations, so only gross anomalies (> 25 m/s²) are flagged, and the full distribution is
  reported. The pilot output (`m11-interaction-catalogue-pilot-v1`) was built with 15 m/s². The full catalogue uses 25 m/s²; each
  value is recorded in the run manifest.
- *Out of pitch.* This assumes a centred origin. The convention is checked per match from the observed coordinate quantiles
  (`coordinates.centred_origin_supported`) before the flag is computed.
- *Timestamps.* Timestamps must be strictly increasing and finite in every period, otherwise the match is refused. Frame gaps and
  the largest step are recorded.

## 8. Built-in consistency checks (per match, `match_summary.json`)

1. **Stage 2 reproduction.** At R = 0.5 m with a 1.0 s chain gap, the edges equal the historical Stage 2 episodes and the edge
   chains equal the Stage 2 candidates. Keys, t\*, d_min and episodes per candidate must all be identical. The historical
   benchmark is therefore unchanged and reproducible as a projection of the catalogue.
2. **Window-intersections v1 reproduction.** The catalogue must reproduce three v1 totals: crossings = v1 intersections, pair-windows
   with ≥ 1 crossing = v1 pair-window detections, and multi-crossing pair-windows = v1 count.
3. **Structure:**
   - unique edge ids;
   - slot_a < slot_b;
   - t\* ∈ [start, end];
   - d_min ≤ R;
   - radii nested;
   - every edge in exactly one group per (R, δ);
   - group edge counts partition the edges.

## 9. Known limitations

- **Chain gaps.** Interactions during chain gaps (missing data, cloned positions, jumps) cannot be catalogued. At 0.5 m about 10 %
  of slot-level close episodes have no valid chain (Stage 2 framewise audit). They are counted, not recovered.
- **Interpolation.** Linear interpolation between 15 Hz updates. A PCHIP variant was compared in v1 and is not repeated here.
- **Group transitivity.** At large R, transitive groups can grow to 10–20 players (set pieces). `max_simultaneous_players` and the
  timelines separate genuine clusters from chains of pairwise contacts.
- **Ambiguity indicator.** `cv_margin_m` uses one hidden length (0.8 s) and one history (0.4 s). It is a scenario descriptor, not an
  evaluation of any model.
- **Not observable.** No statement here concerns camera visibility. Proximity ≠ occlusion.
