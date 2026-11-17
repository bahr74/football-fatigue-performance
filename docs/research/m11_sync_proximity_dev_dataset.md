# M11 — Synchronized proximity encounters: complete development dataset (`m11-sync-proximity-v2`)

| Item | Detail |
|---|---|
| Date | 2026-11-10 |
| Code | `src/football_fatigue/intersections/proximity.py` (extraction), `src/football_fatigue/intersections/proximity_dataset.py` (index, subsets, validation) |
| Tests | `tests/test_intersections_proximity.py` |
| Method | As in [m11_sync_proximity_encounters.md](m11_sync_proximity_encounters.md) (unchanged) |
| Split | All 40 development matches of the frozen split (`m11-encounters-v1/splits.json`) |

Not done in this stage:

- no validation or test match was read or processed;
- no model was trained;
- no labels were created;
- raw data and every earlier output, including `m11-sync-proximity-v1`, are unchanged;
- nothing is committed.

## 1. Final audit (Phase 1)

| # | Check | Result |
|---|---|---|
| 1 | Same timestamp for A and B | Both players are evaluated at the same lo/hi of each pair-interval. Unit test `test_same_timestamp_for_both_players`; the validator checks d(t*) and A(t*), B(t*) on the synchronized grid for every episode |
| 2 | Criterion d ≤ 0.25 m | Analytic quadratic minimum, with exact re-decision at the threshold. Validator `d_min_within_threshold`, `sync_within_threshold_inside_episode` |
| 3 | Original time-parameterized trajectories | Vertices are the provider samples (rows, frames, t, x, y, visibility, quality bits) in `root.trajectories.npz` |
| 4 | ±2.5 s windows at integer seconds | `windows_game_s`; the canonical window is the centre nearest t* |
| 5 | No duplicates across windows | Brute-force window test; validator `episodes_disjoint_per_pair` (no two episodes of a pair touch or overlap) |
| 6 | Repeated encounters kept separate | Test `test_repeated_encounters_of_one_pair` |
| 7 | Observed/estimated status and source rows | `status_at_t_star`, `segment_at_t_star` and npz `sample_visibility` / `sample_source_row` / `sample_frame` |
| 8 | Context before and after t* | **New in v2:** `context.complete`, per-player `complete_before` / `complete_after` and `cut_before` / `cut_after` with the reason |
| 9 | The 18 exact events of 3812 | All 18 lie inside 0.25 m episodes. v2 episode IDs equal v1 on all 10 earlier matches and all thresholds (regression test) |
| 10 | Cloned coincident intervals | `coincident_interval = true` (2,697 in total); excluded from `non_coincident` and `high_quality_candidates`, kept in `all_encounters` |

### Effects of the inherited Step 1 rules (40 matches)

These rules split trajectories; they never drop an encounter.

| Rule | Steps not joined | Effect on 0.25 m episodes |
|---|---|---|
| Gap of more than 0.5 s, period change or timestamp | 938 of 81,826,714 | 42 player-side context cuts |
| Jump of more than 12 m/s and more than 1 m | 385,029 (0.47 %) | **12,122** player-side context cuts: the main cause of incomplete context. The jump landing is flagged `IMPLAUSIBLE_JUMP` |
| Unattributed or ambiguous sample, side over-count | 1,368,000 | 295 cuts. Of these steps, 1,309,180 are the **quarantined extra time** (periods 3–4) of 10506 and 10517, excluded by the frozen split's extra-time rule |
| Substitution or player absent | — | 70 cuts (`no_adjacent_sample`) |
| Support cut at the episode boundary | — | 2,846 episodes |

There was no change to the mathematical definition. **v2 differs from v1 only by:**

- context-completeness and cut-reason metadata;
- the timing record moved to an unhashed `root.runtime.json`, so the hashed outputs are byte-deterministic;
- verified skipping of complete outputs;
- `--jobs` parallelism.

## 2. Dataset policy (Phase 2)

Every subset is a view of the full population, written as an index column `in_<subset>` and a `subset_<name>.csv` file. There are no new thresholds; same-team and opposite-team episodes are both kept, with `team_relation`. The sensitivity thresholds 0.10 m and 0.50 m are kept as separate record files.

| Subset | Definition | Episodes |
|---|---|---|
| `all_encounters` | Every 0.25 m episode | **30,438** |
| `observed_observed` | Both players VISIBLE at t* | 2,619 |
| `mixed_quality` | One observed, one estimated | 4,761 |
| `estimated_estimated` | Both estimated | 23,058 |
| `non_coincident` | Not a cloned coincident interval | 27,741 |
| `high_quality_candidates` | Observed/observed and non-coincident, with all of the integrity conditions listed below | **1,917** (1,513 opposite team, 404 same team) |

The integrity conditions for `high_quality_candidates` are:

- no duplicated provider position;
- no support cut at the episode;
- both players' valid trajectories cover the whole canonical 5 s window;
- no critical Step 1 flag on any context sample. The critical flags are implausible jump, label transfer, reciprocal exchange, unresolved attribution, ambiguous or missing raw sample, side over-count, frame-wide shift and discontinuity.

## 3. Development-split extraction (Phase 3)

| Item | Result |
|---|---|
| Authorized development matches | 40 (from the split manifest) |
| Processed successfully | **40** |
| Failed | 0 |
| Skipped | 0. The skip logic was tested on 3812 (`skipped_complete` after verifying hashes, schema, configuration and code) |
| Step 1 caches built | 30 (3818, then 29 in parallel); all exit code 0; about 140–340 s each, about 0.55 GB each |
| Extraction | 23–42 s per match (1,235 s in total; 2 or 3 workers); peak 2.2 GB per worker |
| Index and validation | 86 s, including three determinism re-renders |

**Bottleneck:** building the Step 1 caches from raw tracking takes about 3–4 minutes per match, against about 30 s for extraction.

## 4. Integrity validation (Phase 4)

`development-index/validation.json` runs 29 checks per match. **All 40 matches pass, with 0 violations.** The checks cover:

- the manifest (hashes, schema, configuration, code);
- unique and stable IDs;
- match, pair, slot and team identity;
- timing order and duration;
- d_min ≤ R;
- d_min equal to the distance between A(t*) and B(t*);
- the midpoint;
- disjoint episodes per pair;
- 0.10 ⊂ 0.25 ⊂ 0.50 nesting;
- npz coverage of every event;
- samples inside the window and inside the period (no cross-period interpolation);
- strictly ordered sample times;
- no moving step longer than 0.5 s plus one frame (no interpolation through gaps);
- source rows and frames;
- visibility codes;
- the before/after split;
- context fields equal to the samples;
- the segment at t* present in the samples;
- the synchronized grid containing t* with the same d and positions;
- the grid within R throughout the episode.

**Negative check.** A tampered copy (scratch directory, since removed) was reported on 8 checks.

**Determinism.** Re-rendering 3812, 3840 and 10517 gave byte-identical outputs: 6 of 6 files each.

**Reconciliation.** These assertions hold in `build_index`:

- subset counts equal the index columns;
- the three quality classes sum to all episodes;
- non-coincident plus coincident equals all episodes.

## 5. Results (0.25 m primary, 40 matches)

**Scale:** 243,935 windows, 56,375,283 pair-window comparisons, 837,503,477 pair-interval comparisons.

| Count | Episodes |
|---|---|
| **Unique episodes** | **30,438** |
| observed/observed · observed/estimated · estimated/estimated | 2,619 · 4,761 · 23,058 |
| Coincident intervals (cloned provider estimates) · any exact coincidence | 2,697 · 2,737 |
| Same team · opposite teams | 14,350 · 16,088 |
| point · isolated · sustained | 6 · 13,048 · 17,384 |
| Context complete · incomplete · with critical flags | **24,088 · 6,350** · 4,719 |
| High-quality candidates | **1,917** |
| d_min median [p10, p90] | 0.146 m [0.003, 0.237] |
| Duration median [p90] | 0.122 s [0.87 s] |

**Sensitivity.** The thresholds nest exactly.

| Threshold | 0 m | 0.10 m | **0.25 m** | 0.50 m |
|---|---|---|---|---|
| Episodes | 2,750 | 13,915 | **30,438** | 61,106 |

Per-match counts are in `development-index/match_summary.csv` (episodes at each threshold, quality classes, coincident intervals, team relation, high-quality candidates, context completeness, runtime). Range at 0.25 m: 416 (3822) to 1,232 (10514).

**10514 stands out:** 428 coincident intervals, which is heavy provider track cloning, and 650 incomplete contexts.

## 6. ML-ready layout (Phase 5)

`development-index/encounters_index.csv` has one row per 0.25 m episode (15.5 MB):

- identities and teams of A and B;
- period, start, end, t*, d_min, A(t*), B(t*), midpoint;
- class and quality;
- coincidence and duplicate flags;
- support cuts;
- the canonical window (centre, start, end);
- context completeness, cut reasons and critical flags;
- the parent at 0.50 m and the number of 0.10 m children;
- subset membership;
- `events_file`, `events_line`, `npz_file` and `npz_event_index`, which locate the full record and its arrays.

The trajectory arrays are not duplicated: they stay in each match's `root.trajectories.npz`.

- **`sample_*`:** original samples of A and B, with t, x, y, source row, frame, visibility, quality bits and `before_t_star`.
- **`sync_*`:** A(t), B(t) and d(t) on the synchronized grid, with `a_is_sample` / `b_is_sample` masks; false means interpolated.

`development-index/schema.json` documents every field. There are no labels, swaps or synthetic occlusions.

## 7. Deliverables

All paths are under `F:\Football-Fatigue-Project\data\processed\m11-sync-proximity-v2\`.

| Deliverable | Path |
|---|---|
| Per-match outputs (40) | `<match>/root.events-{0,0.1,0.25,0.5}m.jsonl`, `root.trajectories.npz`, `root.summary.json`, `root.manifest.json`, `root.runtime.json` |
| Cross-match aggregate | `cross-match/summary.json` |
| Consolidated index | `development-index/encounters_index.csv` |
| Subsets | `development-index/subset_*.csv` |
| Quality-stratified summary | `development-index/summary.json` |
| Per-match CSV | `development-index/match_summary.csv` |
| Schema | `development-index/schema.json` |
| Reproducibility manifest | `development-index/manifest.json` (split hash, per-match manifest hashes, code hashes, output hashes, commands); plus `development-index/determinism.json` and `development-index/validation.json` |
| Step 1 caches (30 new) | `data/processed/m11-encounters-v2/<match>/` |

**Total size:** about 301 MB.

**Tests:** 25 proximity tests (2 new: context completeness and cut reasons; subset definitions), plus the written-output and v1-equivalence test. All intersection suites together: **114 passed**.

## 8. Failures, skips and risks

- **No failures or skips.**
- **Extra-time periods 3–4 of 10506 and 10517** are quarantined by the frozen split rule, so they contain no episodes.
- **The data are dominated by estimates:** 76 % of episodes are estimated/estimated, and 2,697 coincident intervals are provider cloning. Only 1,917 episodes are high-quality candidates. A training design must stratify by quality and must not pool these populations silently.
- **Context completeness is the canonical 5 s window only.** A model needing longer context must re-derive it from the Step 1 caches.
- **Identity reference:** the provider slots are the identity reference, and are not verified. The encounters are tracking-derived proxies, not visual occlusions.

**Reproduction:**

```powershell
python -m football_fatigue.encounters build --root data/raw --match <id>
python -m football_fatigue.intersections.proximity --matches <ids> --jobs 3
python -m football_fatigue.intersections.proximity --matches <ids> --cross-match-only
python -m football_fatigue.intersections.proximity_dataset --matches <ids> --determinism 3812 3840 10517
```

The Step 1 cache build is for development ids only. `<ids>` is the development list in the split manifest.
