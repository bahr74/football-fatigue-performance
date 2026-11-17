# M11 intersection pipeline — Stage S1: temporal-intersection quality audit (match 3812, v2)

Status: **S1 complete for development match 3812; awaiting review.** No model trained; no other match processed. v1 outputs,
earlier M11 artifacts (654 files, checked by size and mtime before and after), the frozen model (`629cca77…`) and the dashboard
are unchanged; RAW snapshot unchanged. New code: `src/football_fatigue/intersections/audit.py`, `review.py`; tests
`tests/test_intersections_audit.py`; outputs `data/processed/m11-intersections-v2/3812/` (written once, never overwritten).

```powershell
.\.venv\Scripts\python.exe -m football_fatigue.intersections.audit --match 3812 [--dry-run]
```

> Geometric and temporal coincidence is **not** verified visual occlusion; ESTIMATED is the provider's visibility flag, not
> occlusion evidence; the continuity reference is provider slot continuity, not verified identity. The 0.20 s primary threshold
> is a **provisional heuristic** awaiting advisor confirmation, not a validated threshold.

## 1. Inputs and outputs

Inputs: the 18,469 v1 events (all 14 v1 files verified against the v1 manifest before and after) and the Step 1 RAW cache and
quality mask (checksum-verified by `load_source`).

| file | role | content |
|---|---|---|
| `root.eligibility.jsonl` | audit only | one record per v1 event (same order): measurements, threshold-independent checks, category + reasons at 0.10 / 0.20 / 0.30 s, pair ID, v1 cluster, candidate cluster per threshold, `in_trajectory_store` |
| `root.trajectories.npz` | store, no identities | for 2,192 events (clean or uncertain at any threshold): every RAW frame of both players in `t* ± 2.5 s` (328,465 rows): `frame_num, video_time_s, record_first` and per track `x, y, state, visibility, confidence, quality_mask, update_instant, duplicate_position`; `event_id, t_star_s, period, offsets` |
| `root.references.jsonl` | identity references | per stored event: `track_0` / `track_1` (seeded order) → slot, side, jersey, player, team; each track's crossing time; continuity reference; reliability (`no_flags` / `flagged` + flags) |
| `root.fig-<category>-<k>.png` | audit figures | 7 categories × 3 (identities and measurements shown) |
| `root.review-<id>.png`, `root.review_worksheet.csv` | blinded review | 50 items; no identity, Δt, threshold, measurement or category |
| `root.review_key.jsonl` | review key | review ID → event, stratum, track mapping, Δt, categories (do not give to the reviewer) |
| `root.summary.json`, `root.config.json`, `root.manifest.json` | | counts, configuration, checksums and provenance |

Visibility, confidence and quality bits are stored as metadata; whether any may be a model input is a later decision.

## 2. Measurements (per event)

`t* = (t_A + t_B)/2`; each player is measured at its own crossing time `t_i`.

- **Δt** = |t_B − t_A| (v1); **velocity**: least-squares line through the player's update instants within `t_i ± 0.25 s`;
  **crossing angle** between the two velocity vectors.
- **Local noise** σ_i: RMS second difference of update-instant positions within `t_i ± 1 s` / √6 (per axis; includes
  acceleration, so an upper bound of measurement noise, and large for jitter).
- **Timing uncertainty** σ_Δt = √( Σ_i [σ_i / (max(v_i, 0.5 m/s) · max(sin θ, sin 15°))]² + 2 (u/√12)² ), u = median update
  interval (0.0667 s). First-order propagation, not a validated error model.
- Same-time distance (minimum within `t* ± 2.5 s`, its time, and at `t*`); per player in the window: coverage, largest gap,
  VISIBLE / ESTIMATED shares, Step 1 implausible jumps and discontinuity candidates, substitution window, unresolved
  attribution, duplicate-position artifact; net displacement, path length and straightness within `t_i ± 1 s`.

## 3. Classification (configurable, `AuditConfig`; fixed before the run)

At each threshold τ, the first matching rule decides; **all** failing checks are recorded as reasons.

1. **invalid_tracking_artifact**: duplicate-position artifact; unresolved/ambiguous attribution; window leaves the period;
   coverage < 0.8; internal gap > 0.5 s (either player). The trajectory cannot be preserved completely or is not one player's.
2. **non_time_aligned_crossing**: Δt > τ + 2·min(σ_Δt, 0.5 s).
3. **uncertain_candidate**: τ < Δt ≤ τ + 2·min(σ_Δt, 0.5 s) (`timing_boundary`), or Δt ≤ τ with any of:
   `timing_ill_conditioned` (σ_Δt > τ/2, speed < 0.5 m/s, angle < 15°, or no velocity), `stationary_player` (net
   displacement < 0.5 m in ±1 s), `jitter` (straightness < 0.5 or σ_i > 0.10 m), `implausible_jump`,
   `discontinuity_candidate`, `substitution_window`, `all_estimated_player`.
4. **clean_candidate**: Δt ≤ τ and none of the above.

Noise never makes an event invalid by itself (jitter, jumps, ESTIMATED → uncertain). The rules are monotone in τ (tested).

## 4. Results (3812)

| | τ = 0.10 s | **τ = 0.20 s (primary)** | τ = 0.30 s |
|---|---|---|---|
| clean candidate | 60 | **180** | 302 |
| uncertain candidate | 1,279 | **1,579** | 1,890 |
| — of which time-aligned (Δt ≤ τ) | 334 | 618 | 933 |
| — of which timing boundary | 945 | 961 | 957 |
| invalid tracking artifact (time-aligned / boundary / not aligned) | 53 (0 / 8 / 45) | **53** (0 / 12 / 41) | 53 (5 / 11 / 37) |
| non-time-aligned crossing | 17,077 | **16,657** | 16,224 |
| time-aligned / boundary / not aligned | 394 / 953 / 17,122 | 798 / 973 / 16,698 | 1,240 / 968 / 16,261 |
| candidates (clean + uncertain): distinct pairs / clusters | 199 / 520 | 213 / 742 | 233 / 964 |
| clean only: distinct pairs / clusters | 47 / 58 | 92 / 161 | 127 / 268 |
| largest pair (candidates) / top-10 % pairs' share | 86 / 0.45 | 90 / 0.42 | 94 / 0.42 |

Per-reason counts for every threshold and category are in `root.summary.json`.

**Reasons at 0.20 s** (an event can have several): uncertain — timing ill-conditioned 1,125, jitter 1,116, all-ESTIMATED player
1,049, timing boundary 961, implausible jump 636, stationary player 211, discontinuity candidate 21, substitution window 3.
Time-aligned uncertain (618): timing ill-conditioned 376, jitter 370, all-ESTIMATED 333, implausible jump 295, stationary 46,
discontinuity 12; 189 have exactly one reason (jump 80, timing 46, all-ESTIMATED 37, jitter 26). Invalid (53, identical at all τ;
none time-aligned: 41 not aligned, 12 boundary): duplicate position 28, unresolved attribution 22, period boundary 7.

**Clean candidates at 0.20 s** (180): ESTIMATED share median 0.26 (26 events fully VISIBLE for both players); minimum same-time
distance median 0.24 m (p90 0.60); σ_Δt median 0.033 s (p90 0.052); slower player's speed median 2.7 m/s; crossing angle
median 44°. They come from 92 pairs and 161 candidate clusters; 27 are the only v1 event of their pair cluster.

**Finding after visual inspection (not part of the pre-specified rules):** 61 of the 180 clean candidates (18/60 at 0.10,
95/302 at 0.30) contain at least one Step 1 `OVERSPEED_STEP` — a sub-metre step faster than 12 m/s, e.g. the speed spikes of
away #22 in `root.fig-clean_near_simultaneous-1.png`. They are not implausible jumps by the Step 1 definition (> 1 m), so the
rules above do not flag them. Whether they should make an event uncertain is an open decision (§7); the rules were not changed
after seeing the results. Among uncertain candidates, extreme provider artifacts remain *uncertain* by design: e.g. review item
`Rbc083839` shows two tracks moving at ~20 m/s for over a second (a run of flagged jumps).

Runtime: load 4 s, audit 30 s, figures ~29 s; peak memory ≈ 0.6 GB; outputs 59 MB (eligibility 46 MB, store 4.7 MB).

## 5. Figures (`data/processed/m11-intersections-v2/3812/`)

Chosen by a seeded hash within each set (set sizes at 0.20 s in brackets): `fig-clean_near_simultaneous-{1,2,3}` (180),
`fig-temporal_mismatch-{1,2,3}` (non-aligned with Δt ≤ 1 s; 2,611), `fig-tracking_jitter-*` (370), `fig-implausible_jump-*`
(295), `fig-estimated_heavy-*` (333), `fig-repeated_pair_crossings-*` (aligned, v1 cluster ≥ 10; 241),
`fig-uncertain_crossing_time-*` (1,337). No invalid artifact is time-aligned at 0.20 s, so there is no invalid-artifact figure;
the v1 figure `m11-intersections-v1/3812/root.fig-duplicate_position_artifact.png` shows the main artifact.

**Blinded review** (`root.review_worksheet.csv`, 50 items in hashed order): clean 15, time-aligned uncertain 15, timing
boundary 10, non-aligned with Δt ≤ 0.5 s 10 (the "invalid, time-aligned" stratum is empty). Questions per item: same-time
passage of the intersection (yes/no/unclear), tracking plausible, identity continuity plausible, notes. The key is in
`root.review_key.jsonl`; the reviewer must not see it.

## 6. Tests

`tests/test_intersections_audit.py` (14): threshold boundaries (Δt = τ aligned; band edge; σ cap; unknown σ), τ-dependent
timing ill-conditioning, precedence and complete reason lists, noisy events never invalid, σ_Δt for noise-free fast
perpendicular crossings (= quantisation only) and against the formula with injected noise, slow/shallow crossings flagged,
Δt and same-time distance on a known lag, lossless store (every frame of the window, equal to the cache, interaction not
masked), identity separation (store keys, seeded track order), one record per v1 event and candidate-cluster accounting,
monotonicity in τ, byte-identical outputs on repeat, v1 integrity, written v2 outputs (checksums, record order = v1,
category totals, lossless spot check against the cache, worksheet free of identity/category tokens). Full suite: 799 passed,
6 skipped.

## 7. Assumptions and open decisions (advisor)

1. Primary τ = 0.20 s (provisional); report 0.10 / 0.30 as sensitivity.
2. Timing model: σ_Δt formula, floors (0.5 m/s, 15°), z = 2, σ cap 0.5 s, ill-conditioned when σ_Δt > τ/2.
3. Whether `OVERSPEED_STEP` (sub-metre > 12 m/s) should make a candidate uncertain (would leave 119 clean at 0.20 s).
4. Whether all-ESTIMATED players should be uncertain (1,049 at 0.20 s) — ESTIMATED is not occlusion evidence, but the positions
   are provider-generated, not measured.
5. Jitter and stationarity thresholds (straightness 0.5, σ 0.10 m, 0.5 m net); repeated pair crossings are metadata only.
6. ±2.5 s window kept; whether the training view needs a longer context.
7. Provider continuity as the reference; how flagged references are used.
