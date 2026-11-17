# M11 Stage 3: supervised MATCHED/SWAPPED dataset, results (v1)

Status: **complete and verified, 2026-11-14.**
- Built: the 32 train/CV matches only.
- Not built: held-out supervised shards, deferred until the final protocol is frozen.
- Not done: no model trained, no benchmark run.

Design: `M11_STAGE3_SUPERVISED_DATASET_DESIGN_v1.md`. Benchmark plan: `M11_BENCHMARK_PLAN_v1.md` (review in §15).

## 1. Implementation summary

**New module** `src/football_fatigue/window_intersections/samples.py`.
- Run with `python -m football_fatigue.window_intersections.samples --matches train_cv --out m11-stage3-samples-v1`, plus `--resume` and `--simulate-crash-after`.
- Reads only immutable inputs:
  - the Stage 2 `candidates.parquet` of each match, hash-checked against the Stage 2 manifest;
  - the Step 1 caches;
  - the frozen split, whose hash is checked against the approved `66b2321a…6c0`.
- **Refuses held-out matches.** The Stage 2 files of the 8 held-out matches are only hash-checked; their contents are never parsed.
- **Refuses a changed configuration.** The candidate configuration must equal the one recorded in the Stage 2 manifest.

**Per match:**
1. Rebuild the Stage 2 valid segments and chains (linear geometry).
2. For every Stage 2 ACCEPTED candidate and each table, build one sample from the two reference chains.
3. Verify the sample (§10).
4. Write the outputs atomically under `matches/<m>.partial` with `match_done.json` SHA-256 hashes, then rename the directory. This reuses the Stage 2 runner `local.run_matches`.

Stage 2 outcomes are never changed. LOWER_CONFIDENCE_IDENTITY and REJECTED candidates are never built.

**Temporal grid.** Δ = 1/15 s and T = 30 points, a fixed convention. The hidden gap is the closed interval [g0, g1].
- Incoming points: g0 + kΔ, k = −30…−1.
- Outgoing points: g1 + kΔ, k = 1…30.
- **Interpolation.** Positions are linear interpolation of the chain's native 15 Hz vertices. Incoming points use only vertices strictly before g0; outgoing points use only vertices strictly after g1. A grid point outside those vertices is masked; nothing is extrapolated or bridged.
- **Result.** No vertex inside or on the boundary of the gap can influence any input. This is verified by a perturbation test (§11).

**Frame.** Computed from the incoming tracklets only.
- Origin: the midpoint of the last valid incoming points.
- Axis, mode 0: the summed incoming velocity (least squares over the last 0.5 s).
- Axis, mode 1, used when the speeds cancel (opposing motion): ±(v0 − v1), with the sign fixed to point to pitch +x.
- Axis, mode 2, used when both players are nearly stationary: the pitch axes.
- Every mode is invariant to the incoming order (tested).

**Order and label.**
- The incoming and outgoing order are independent bits of `sha256("m11-stage3-order-v1|candidate_id|table")`.
- y = 0 MATCHED, y = 1 SWAPPED.
- `recovery_y_to_stage3` / `stage3_y_to_recovery` convert explicitly between this and the old `m11_recovery` convention (y = 1 meant matched). This is tested against the real `views.make_sample`.
- The existing `m11_recovery` code and its stored outputs are unchanged.

**Model-input adapter** `samples.load_model_inputs(out, table)`. It returns `(inputs, y, groups)`:
- `inputs`: exactly the 7 approved fields.
- `groups`: match ids, for grouped folds only.

It refuses to run if:
- any held-out match is requested;
- a held-out shard exists in the directory;
- the split differs from the approved one;
- any shard fails its hash.

It never returns metadata.

## 2. Output schema

`data/processed/m11-stage3-samples-v1/matches/<match>/`, with 10 files per match.

| File | Content |
|---|---|
| `samples_<table>.npz` | model arrays + `y` (uncompressed, fixed zip timestamps → byte-deterministic) |
| `meta_<table>.parquet` | row-aligned metadata (`row` = array index; never read by the adapter) |
| `failures.parquet` | every accepted candidate not built, with table, stage and reason |
| `lower_confidence_identity_audit.parquet` | metadata of LOWER_CONFIDENCE_IDENTITY candidates (any gap); audit only, never built |
| `match_done.json` | SHA-256 of every file + per-match summary |

**Tables:** `g0p80` (primary), `g0p40`, `g1p60` and `g0p80_shift`.

**Model arrays (per sample):**

| Field | Shape | Dtype |
|---|---|---|
| `in_pos` | (2, 30, 2) | float32 |
| `in_mask` | (2, 30) | bool |
| `in_dt` | (30,) | float32; −2.0 … −1/15 s |
| `out_pos` | (2, 30, 2) | float32 |
| `out_mask` | (2, 30) | bool |
| `out_dt` | (30,) | float32; 1/15 … 2.0 s |
| `gap_s` | () | float32 |
| `y` | () | int8 |

Masked coordinates are 0. There are no NaN or Inf values.

**Metadata columns:**
- Identifiers: `match_id`, `period`, `split`, `candidate_id`, `table`, `row`.
- Gap: `gap_s`, `shift_s`, `gap_centre_s`, `t_star`.
- Label and order: `y`, `in_order`, `out_order`.
- Frame and dependency: `frame_mode`, `overlap_group`, `independent`.
- Geometry and strata: `d_min`, `d_band`, `chord_speed_min_m_s`, `speed_band`, `rel_speed_m_s`, `strict_pass`, `crossing_class`, `n_cross_local`, `n_cross_window`, `team_relation`, `time_within_radius_s`.
- Diagnostics: `near_coincidence`, `jump_transfer_risk`, `jump_transfer_risk_sides`, `boundary_jumps_in_window`.
- Status, reordered to the incoming order: `est_frac_used_in{0,1}`, `est_frac_in_gap_in{0,1}`, `status_in_gap`.
- Point counts: `n_valid_{in,out}{0,1}`.
- Cohorts: the 8 `cohort_*` F1–F4 states.

**Slot, chain, player, jersey and team identities are not written at all**, not even to the metadata.

## 3. Constructed sample counts (32 train/CV matches)

| Table | Stage 2 ACCEPTED | Built | Failed | Independent | Overlap groups | Largest group |
|---|---|---|---|---|---|---|
| **g0p80 (primary)** | 25,331 | **25,331** | 0 | 18,777 | 16,935 | 51 |
| g0p40 | 25,668 | 25,668 | 0 | 19,342 | 17,566 | 51 |
| g1p60 | 24,661 | 24,661 | 0 | 17,759 | 15,729 | 62 |
| g0p80_shift | 25,331 | 25,172 | 159 | 18,648 | 16,826 | 51 |

- **Gap tables are not independent data.** The three gap tables are versions of the same encounters, and the shift table is a re-cut of the primary one. None of them is counted as additional independent training data. They have different order draws per table, so they must never be pooled across tables.

## 4. Per-match and per-gap counts

Samples per match (min / median / max):

| Table | Min | Median | Max |
|---|---|---|---|
| g0p80 | 541 | 782 | 1,078 |
| g0p40 | 544 | 789 | 1,093 |
| g1p60 | 522 | 767 | 1,055 |
| g0p80_shift | 540 | 779 | 1,071 |

The full per-match counts are in `summary.json` (`<table>.per_match`).

**Shift distribution** (g0p80_shift): −0.3 s 3,574, −0.2 s 3,699, −0.1 s 3,778, 0.0 s 3,515, +0.1 s 3,544, +0.2 s 3,499, +0.3 s 3,563.

## 5. MATCHED/SWAPPED balance

| Table | SWAPPED share (all) | SWAPPED share (independent) |
|---|---|---|
| g0p80 | 0.4986 (12,629 / 25,331) | 0.4995 |
| g0p40 | 0.5025 | 0.5025 |
| g1p60 | 0.5014 | 0.5051 |
| g0p80_shift | 0.5033 | 0.5002 |

- **Overall balance.** All shares are within ±0.6 percentage points of 0.5. The binomial SE is about 0.3 points.
- **Balance by stratum** (g0p80):
  - status: both fully ESTIMATED 0.497, mixed 0.500, both fully VISIBLE 0.501;
  - crossing: local 0.495, window-only 0.501, none 0.508.
- **Order is unrelated to strata.** These are what an order independent of the strata should produce.
- **Held-out balance** will be checked when those shards are built.

## 6. Independent vs. overlapping samples

**g0p80:**
- 18,777 samples are in the greedy independent subset (74.1%).
- 13,561 overlap groups are singletons.
- Groups of size 2–9: 1,974 / 633 / 263 / 128 / 71 / 65 / 39 / 37.
- 164 groups have 10 or more samples; the largest has 51.
- Every overlap group lies within one match (tested). So grouped CV by match never separates dependent samples.

## 7. Failed construction

- **Gap tables:** 0 failures in all three.
  - Every Stage 2 ACCEPTED candidate produced a sample with at least 3 valid points per tracklet.
  - Every sample passed label, boundary and inverse-transform verification.
- **Shift table:** 159 failures (0.63%), all rejected by the full Stage 2 label-quality procedure re-run at the shifted centre:
  - `continuous_reference`: 135 (a chain no longer covers the shifted gap ± 0.5 s);
  - `chord_speed_plausible`: 24.
  - None became LOWER_CONFIDENCE_IDENTITY.
  - All 159 are listed in `failures.parquet` with candidate id, shift and reason.
- **Masked points** (g0p80): about 1.0% of incoming points and 0.7% of outgoing points are masked. These are chains that start or end inside the 2 s context, so the tail is masked and never bridged.

## 8. Near-coincidence and jump-risk retention

Both are retained as Stage 2 accepted them, with their flags in the metadata.

| Table | Near-coincidence (d_min < 0.05 m) | Jump-transfer-risk |
|---|---|---|
| g0p80 | 2,814 (11.1%) | 377 |
| g0p40 | 2,847 | 444 |
| g1p60 | 2,730 | 270 |
| g0p80_shift | 2,790 | 333 |

d_band (g0p80): <0.05 m 2,814; 0.05–0.25 m 9,389; 0.25–0.5 m 13,128.

## 9. Tracking-status composition (audit only, never an input)

g0p80: both fully ESTIMATED in the gap 12,991 (51.3%), mixed 10,061, both fully VISIBLE 2,279 (9.0%).

**Status never changes the inputs.** A test rebuilds a match with all samples ESTIMATED and with random status: the arrays are byte-identical.

## 10. Label verification

For every built sample, all 100,832 across the four tables:
- **Independent label check.** Each of the four tracklets was re-derived from the raw cache positions of both provider slots. A tracklet is assigned to the slot that reproduces it exactly (error ≤ 1e-9 m). The label implied by those assignments must equal `y`.
  - Result: **label_ok for 100,832 / 100,832**, with 0 mismatches and 0 ambiguous.
- **Gap boundaries.** Every valid incoming grid time is < g0 and every valid outgoing grid time is > g1: 100,832 / 100,832.
- **Inverse transform.** Frame coordinates map back to pitch coordinates within 1e-3 m: 100,832 / 100,832.

Synthetic tests check the label semantics for all four order combinations, with known identities:
- swapping the outgoing order flips the label;
- swapping the incoming order flips it;
- swapping both preserves it;
- swapping the incoming order equals rebuilding with the incoming order bit flipped.

## 11. Leakage tests

23 new tests are in `tests/test_stage3_samples.py`.

| # | Risk | Protection, and the test or check |
|---|---|---|
| 1 | Hidden-gap positions | strict vertex selection. Gap-vertex perturbation (±1000 m, random): identical arrays, both in tests and on real data (800 samples per table, 3,200 in all, **all identical**). Vertices exactly on g0 or g1 are excluded (tested). |
| 2 | Reference identity | identities are used only offline, for the label and its verification. Not stored in the arrays or the metadata (tested). |
| 3 | Provider slot | the hashed order depends only on candidate id and table. Balance and independence of the bits tested on 4,000 ids. |
| 4 | Tracking status | not an input. ESTIMATED / VISIBLE invariance tested (byte-identical arrays). The texture risk remains (§14). |
| 5 | Crossing class | metadata only; the label never depends on it. Balance by crossing class is 0.495–0.508. |
| 6 | Metadata | the adapter returns exactly the 7 input fields (tested). Field names are checked against forbidden words. |
| 7 | Split | the builder and the adapter refuse held-out matches. The adapter also refuses if a held-out shard exists, if the split hash is wrong, or if a shard was tampered with (3 tests). No held-out shard exists. |
| 8 | Overlap groups | groups lie within a match (tested); `match_id` is returned as the grouping key. |
| 9 | Preprocessing | each sample is normalised by its own incoming tracklets only. There are no dataset-level statistics. A sample built alone is identical to the same sample built in a batch (tested). The frame does not change when outgoing positions change (tested). |
| 10 | Augmentation and ordering | the swap helpers are exact (tested). No augmented copies are stored, so no duplicate example can cross folds. Each gap table uses its own order draw. |

Scaling, calibration and model selection are not done here. They must be fitted inside the training folds in Stage 4.

## 12. Runtime and memory

| Run | Time | Notes |
|---|---|---|
| Full run | 373 s (6.2 min) | median 11.4 s per match (7 s loading the Step 1 cache, about 4.5 s building); peak working set 1.8 GB |
| Pilot | 2 matches | simulated crash after match 1, then `--resume`. Then a clean rebuild in a separate directory. **All 20 per-match data files are byte-identical**; only runtime fields in `match_done.json` and `runtime.json` differ. |

**Pilot directories** (kept, immutable): `m11-stage3-samples-pilot-v1` (crash, then resume) and `m11-stage3-samples-pilot-rebuild-v1`.

**Tests:**
- Focused: 46 passed (Stage 2 + Stage 3).
- **Full suite: 1004 passed, 6 skipped** (981 before, plus 23 new).

## 13. Output paths and hashes

| Item | Value |
|---|---|
| Dataset | `data/processed/m11-stage3-samples-v1` (141 MB, 356 hashed files, manifest sha256 prefix `6016fa6c00814ef4`) |
| Checks | `checks.json`: all_pass, arrays_all_pass, verification_all_pass, counts_reconcile, raw_unchanged = True; heldout_built = False |
| Split | `m11-stage2-split-v1/split.json`, sha256 `66b2321a58f3845194757f943c2d36b92f18695322fb5490917f8a0144ff56c0` (unchanged) |
| Stage 2 inputs | `m11-stage2-candidates-v1` (unchanged; its manifest hash is recorded in the Stage 3 manifest) |
| Code | hashes in `manifest.json` → `code_sha256` |

Re-verification: every file was re-hashed against `manifest.json` (0 mismatches), and every match directory against its `match_done.json`.

## 14. Remaining risks

1. **Label circularity.** 51% of the primary gaps are entirely ESTIMATED by the provider for both players. There, the "truth" is the provider's own continuity across its own interpolation. Benchmark metrics must be reported by `status_in_gap`.
2. **Status-texture shortcut.** VISIBLE jitter is about 1 cm and ESTIMATED about 8 cm. Status is not an input, but its texture is in the positions. This is not testable without a model; the smoothing and noise ablations are planned.
3. **Selection bias** (Stage 2 framewise audit). Messy encounters (duplicates, jumps, identity flags) are under-represented. Results describe clean, continuous encounters.
4. **Near-coincidence.** 11% of samples have the two players closer than 5 cm at closest approach. Motion is the only evidence there, and some of these may be provider-merged tracks.
5. **Fixed grid.** in_dt and out_dt are constant across samples, so they carry no information. Native timing irregularities appear only as masked points (about 1%).
6. **The frame-mode fallbacks** (mode 1: 2.3%, mode 2: 2.7% of g0p80) use pitch-axis orientation. That is incoming-only information, but it is a weak cue to the absolute pitch direction.

## 15. Readiness for model benchmarking, and benchmark-plan review

**Ready**, given approval for Stage 4. Changes needed in `M11_BENCHMARK_PLAN_v1.md` before training:

1. **Adapter into `m11_recovery`.** `motion.Kin`, `features` and the baselines read a `views.Sample` with NaN for missing points and times relative to the gap centre.
   - A converter is needed: positions → NaN where masked; `t_before = in_dt − g/2`; `t_after = out_dt + g/2`; `y_recovery = stage3_y_to_recovery(y)`.
   - `dataset.T_PAD` and the Hybrid model assume the 0.1 s grid; they must use T = 30.
2. **Angle stratum.** The crossing angle (plan §6) is not in the Stage 3 metadata. Either join it offline from the Stage 2 candidates by `candidate_id`, or drop it.
3. **Controls.** Plan §4 refers to "matched controls". In Stage 3 the non-crossing encounters are already inside the population (`crossing_class = no_crossing`, 4,323 in g0p80). Results should be stratified by crossing class. A separate weighted control table is not needed for training. The weight cap (B2) applies only if a control-weighted evaluation is still wanted.
4. **Held-out build.** The final protocol (plan §8) needs one extra, explicitly authorised step: building the held-out shards with the same frozen code and configuration, after `final_protocol.json` is frozen. That step is not implemented.
5. **Headline subset.** Report the independent subset per table (§6). The three gap tables share encounters, so the gap-transfer analysis (plan §7.3) must compare paired results, not pooled ones.
6. **Shift table.** It has 0.6% fewer samples than the centred table (§7). Compare the shift table with the centred table on the same candidate ids.
7. **Tree-model features.** `motion.features` gives speeds and separations in the per-sample frame, consistent with this design. Absolute pitch features remain excluded (D3).

Open decisions from `m11_stage2_results_v1.md` §8 (D1–D7, B1–B6) were implemented as recommended where they affect Stage 3:
- D1: LOWER excluded and kept for audit;
- D2: 1/15 s grid, T = 30;
- D3: no pitch coordinates;
- D4: near-coincidence and jump-risk kept and flagged;
- D7: framewise-unmatched encounters not used;
- B4: label convention 0 = MATCHED, 1 = SWAPPED.

The training-time decisions (D5, D6, B1–B3, B5, B6) are open for Stage 4.

**Stopped here.** Waiting for approval to begin Stage 4 (model training).
