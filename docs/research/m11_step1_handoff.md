# M11 Step 1 handoff: FIFA encounter catalogue (RAW-first)

Status at handoff: 2026-11-06. Step 1 is implemented and validated. Step 2 has **not** started.

## 1. Objective and constraints

- **M11 goal:** occlusion-aware preservation and recovery of player identity on FIFA World Cup 2022 (PFF) tracking.
  - First reproduce FOOTBALLTrace (Mostafa et al., icSPORTS 2023; paper at `data/raw/122268.pdf`): occlusion-candidate detection and swap correction.
  - Only then improve on it.
- **Supervisor instruction:** use **RAW coordinates only**. Provider-smoothed coordinates are never used for identity analysis or as ground truth.
- **Scientific rules:**
  - Proximity is not confirmed occlusion.
  - A suspicious jump is not a confirmed swap.
  - Hard cases are preserved.
  - False corrections are measured.
  - Matches stay disjoint across development, validation and test.

## 2. Architecture (IMPLEMENTED)

**Reused unchanged:**
- `Dataset` / `iter_tracking_frames`;
- `MatchData(policy=CoordinatePolicy.RAW)`, giving raw coordinate states, the roster join and quarantine;
- `PlayerSample` / `TrackingFrame`, `QualityFlag`;
- `safe_output_path` / `approved_output_root`.

No legacy module was modified (`ingestion`, `match_data`, `synchronization`, `analytics`, `playback`, `vision`, `multicam_lab`).

**New package `src/football_fatigue/encounters/`:**

| Module | Content |
|---|---|
| `trajectories.py` | `build_cache` (one raw pass → `RawCache` arrays, per frame (F,), per slot (S,), per frame × slot (F,S)); `load_cache(path, defaults)` |
| `quality.py` | `M11Config` (all thresholds), `assess` (non-destructive uint32 mask, update-instant steps, jumps, substitutions), `sample_categories` |
| `catalogue.py` | `find_encounters`, `summarize`, `write_outputs` (all-or-nothing, never overwrites), `run_match` (raw pass), `run_from_cache` (checksum- and provenance-verified rebuild), `peak_memory_mb` |
| `splits.py` | `assign` (deterministic 40/12/12), `write_manifest` (frozen, never replaced), `default_quarantine` |
| `__main__.py` | CLI `splits` and `build` (`--from-cache`, `--max-frames`, `--allow-validation`; test-split matches are refused) |

Tests: `tests/test_encounters.py` (13 synthetic-fixture tests).

## 3. Git status at handoff (uncommitted; the user commits)

- **New, untracked:**
  - `src/football_fatigue/encounters/`
  - `tests/test_encounters.py`
  - `docs/research/m11_step1_handoff.md`
- **Pre-existing and unrelated** (not touched by M11):
  - modified: `docs/vision_module.md`, `pyproject.toml`, `src/football_fatigue/vision/{__main__,detector,models,pipeline,tracker}.py`, `tests/test_vision.py`
  - untracked: the vision, demo, multicam and analytics work, several docs, and `wadi-degla-demo-assets.zip`
- Branch: `feature/wadi-degla-demo`.

## 4. Outputs, schemas and splits

| Path | Schema | Note |
|---|---|---|
| `data/processed/m11-encounters-v1/splits.json` | `m11-encounters-v1-splits` | **Frozen.** Seed `m11-encounters-v1`. 3812, 3816 and 10506 fixed in development. `release_disagreement` (10 matches) and `extra_time_anomaly` (10506, 10510, 10511, 10517; periods 3–4 quarantined) are attributes, not splits |
| `data/processed/m11-encounters-v1/3812/root.*` | `m11-encounters-v1` | **Stale, kept for comparison.** Raw pass, before the fixes |
| `data/processed/m11-encounters-v2/3812/root.*` | `m11-encounters-v2` | **Current.** Rebuilt from the v1 cache (sha256 `eb3e9916…`, recorded under `derived_from_cache`) |

**Files per match:**
- `root.cache.npz`: raw arrays. In v2 it also holds its scalar fields.
- `root.quality.npz`: the mask, the speed array and the step arrays.
- `root.encounters.jsonl`: one record per encounter and τ.
- `root.jumps.jsonl`
- `root.summary.json`
- `root.manifest.json`: config, code sha256, tracking source path and size, split, processing time and memory, output checksums, and `provider_smoothed_coordinates_used: false`.

**The split:**
- Development: 3812, 3814, 3815, 3816, 3818, 3819, 3821, 3822, 3823, 3824, 3826, 3828, 3829, 3830, 3831, 3833, 3834, 3835, 3836, 3837, 3838, 3839, 3840, 3843, 3844, 3847, 3849, 3851, 3852, 3855, 3858, 3859, 10502, 10503, 10506, 10507, 10510, 10512, 10514, 10517.
- Validation: 3817, 3820, 3825, 3827, 3832, 3841, 3842, 3846, 3856, 10508, 10511, 10513.
- **Locked test:** 3813, 3845, 3848, 3850, 3853, 3854, 3857, 10504, 10505, 10509, 10515, 10516.

## 5. Validated results

- **Tests:** FIFA regression plus encounters: **269 passed, 6 skipped, 0 failed** (13 of them encounter tests).
- **Match 3812:** 184,857 frames, 30 slots, 6,168 s of tracked time.

  | | 1 m | 2 m | 3 m | 4 m |
  |---|---|---|---|---|
  | Encounters | 882 | **3,157** | 5,115 | 6,968 |
  | Suspicious identity | 5 | **15** | 29 | 53 |
  | Unresolved | 5 | 19 | 37 | 42 |
  | Trusted reference candidates | 116 | **342** | 323 | 224 |
  | Ordinary | 756 | 2,781 | 4,726 | 6,649 |
  | With a third player within τ | 266 | 1,522 | 3,371 | 5,647 |

- **Samples:**
  - valid raw 1,729,225; ESTIMATED 2,314,756;
  - implausible jump 14,451; over-speed step 8,463; possible identity break 207;
  - repeat share 49.96%; median update interval 0.0667 s (15 Hz);
  - valid speed max 12.00 m/s, p99 8.79 m/s.
- **Jumps:** 14,658, of which:
  - 396 are part of a frame-wide shift (31 frames);
  - 207 are possible identity breaks (3 label transfers + 204 isolated jumps over 3 m);
  - 0 are reciprocal exchanges.
- **v1 → v2:** encounter detections are identical. Suspicious at 2 m went from 646 to 15 and trusted from 385 to 342 (lost to over-speed noise). **This is a labelling change, not a measured accuracy gain.**
- **Performance:**
  - raw pass: 172 s;
  - cache rebuild: 42 s, peak 609 MB.

## 6. Quality-mask logic and thresholds (`M11Config` defaults)

- **Cadence:**
  - `REPEAT_OF_PREVIOUS`: exactly equal to the same slot in the immediately preceding frame of the same period.
  - `UPDATE_INSTANT`: the first sample of each run of equal values.
  - `HELD_POSITION`: a run of 3 frames or more. Informational only, never invalid.
- **Motion:** computed only between consecutive update instants of one slot, using real timestamps, within one period, and with at most `max_gap_s = 0.5` s unobserved between them.
  - `change_dt = min(dt, t_new − t_last_old + 1/fps)`
- **`IMPLAUSIBLE_JUMP`:** distance > `jump_min_m = 1.0` **and** distance / change_dt > `max_speed_m_s = 12.0`.
- **`OVERSPEED_STEP`:** over the speed limit but distance ≤ 1 m.
- Jumps and over-speed steps are flagged, kept and excluded from velocity statistics.
- **`FRAME_WIDE_SHIFT`:** at least max(5, 0.5 × observed slots) jumps in the same frame.
- **Landing on another slot:** the label lands within min(`transfer_radius_m = 1.0`, distance / 2) of another slot's position just before the jump.
  - `LABEL_TRANSFER_CANDIDATE` = lands on another slot and distance > `transfer_min_m = 3.0`.
  - `RECIPROCAL_EXCHANGE` = two jumps of any length within ±2 rows, each landing on the other's previous position.
- **`DISCONTINUITY_CANDIDATE`:** a label transfer, a reciprocal exchange, or a jump over 3 m that is not part of a frame-wide shift.
- **Other flags:**
  - `SUBSTITUTION_WINDOW`: off/on players from root `SUB` events, 2 s before to 5 s after.
  - `SIDE_OVERCOUNT`: more than 11 players listed for a side.
  - `ATTRIBUTION_UNRESOLVED`, `AMBIGUOUS_RAW`, `MISSING_RAW`, `ESTIMATED`, `LOW_CONFIDENCE`, `NO_MOTION_CONTEXT`.
- **Encounters:** FOOTBALLTrace rule.
  - τ ∈ {1, 2, 3, 4} m; 2 s windows, 50% overlap; distance ≤ τ for at least 50% of the window **in time** (frame weights are capped at 1/fps).
  - Windows live inside segments: one period with no gap over 0.5 s, so nothing crosses a period boundary or a gap.
  - Consecutive qualifying windows are merged; the encounter runs from the first to the last close frame.
  - Context: ±2 s.
  - Third players: anyone within τ of either player.
- **Encounter category** (first match wins):
  1. `suspicious_identity`: a discontinuity candidate in the context.
  2. `unresolved`: ambiguous or unattributed, substitution window, over-full side, or coverage below 0.5.
  3. `trusted_reference_candidate`: no jump or over-speed step, no third player, coverage and VISIBLE share both at least 0.9.
  4. `ordinary`: everything else.

## 7. Anomaly examples (reproduce from the v2 outputs)

- **Away #3 label jump:** away #3, frames 109377→109378, `jump_id 3812-j07977`.
  - 29.5 m, VISIBLE to VISIBLE, confidence HIGH→LOW.
  - Label transfer: it lands 0.92 m from another slot; nobody is within 4.41 m of the origin afterwards.
  - Its encounters (`3812-t2-01936`, `3812-t2-01937`) are suspicious.
- **Frame-wide shift:** frame 109378, 15 of 15 jumps, the others moving a median of 2.18 m. Clusters at frames 89926–89952 (14–16 players each).
- **Normal encounters:**
  - `3812-t2-00003`: trusted. Home #3 vs away #7, 4.57 s, closest 0.88 m.
  - Home #3 vs away #8, P2: never suspicious.
- **Largest isolated jump:** `3812-j06677`, away #22, 7.75 m, ESTIMATED to ESTIMATED. Probably the provider re-positioning a player it couldn't see.
- **To inspect:** filter `root.jumps.jsonl` / `root.encounters.jsonl` by `jump_id` or `encounter_id`; raw values are in `root.cache.npz` (`x`, `y`, `frame_num`, `slot_side`, `slot_jersey`).
- **Known real swap in another match** (from earlier M11 research, not yet processed): 3816 P2 04:00.2, away #5 / #17.

## 8. Commands

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_encounters.py -p no:cacheprovider
.\.venv\Scripts\python.exe -m pytest -q tests\test_ingestion.py tests\test_match_data.py tests\test_match_data_quality.py tests\test_synchronization.py tests\test_analytics.py tests\test_analytics_motion.py tests\test_analytics_observed.py tests\test_audit_lib.py tests\test_playback_adapter.py tests\test_playback_navigation.py tests\test_inventory.py tests\test_package_installation.py tests\test_encounters.py -p no:cacheprovider
# raw pass (about 3 min per match; refuses existing outputs):
.\.venv\Scripts\python.exe -m football_fatigue.encounters build --root data/raw --match <id>
# verified rebuild from a cache (no raw pass):
.\.venv\Scripts\python.exe -m football_fatigue.encounters build --root data/raw --match 3812 --from-cache data/processed/m11-encounters-v1/3812/root.cache.npz --out <new output root>
```

## 9. Scientific risks and limitations

- **No independent real swap labels.** Suspicious encounters are candidates. Precision and recall on real data are unknown.
- **ESTIMATED observations** (57% of samples):
  - 126 of the 207 identity-break jumps go from ESTIMATED to ESTIMATED, and 172 have LOW confidence afterwards;
  - this is a likely source of false alarms.
- **Third players:** 48% of encounters at 2 m have one, so the two-player assumption often fails. Groups are flagged, not resolved.
- **Class imbalance:** about 15 suspicious against 3,157 encounters (2 m), and 0 reciprocal exchanges in 3812. Real swaps are rare.
- **Missed swaps:**
  - A swap inside a frame-wide shift with a jump of 3 m or less may be missed, because landing is checked against positions before the shift.
  - Swaps spread over several updates are not modelled.
- **Untuned thresholds:** 12 m/s, 1 m, 3 m, 1 m radius, 50% frame-wide share.
- **Acceleration from raw differences is noise** (p95 about 45 m/s²): diagnostic only.
- **Stationary vs provider-held** samples can't be told apart from the data alone.
- **Missing checks:** release agreement is not checked (the 10 matches remain an attribute only), and there is no ball or event evidence.
- **One match processed.**

## 10. Proposed Step 2 (NOT IMPLEMENTED; needs approval)

1. **Build catalogues for more development matches** with a raw pass each (the approval to process more matches is pending).
2. **Synthetic benchmark generator**, reading `root.cache.npz` through `load_cache`:
   - **Swaps:** tail exchange at a real encounter time, on the 15 Hz update grid.
   - **Fragments:** a deleted interval followed by a new anonymous ID.
   - **Where:** only in `trusted_reference_candidate` encounters (VISIBLE, no jumps, no third player); variants for same team / opposite team and for distance.
   - **Untouched negatives:** a matched set of encounters, so the false-correction rate can be measured.
   - **No leakage:** anonymous per-window IDs; no jersey numbers, roster data or smoothed positions.
   - **Separate reporting:** natural candidates (the 207 identity-break jumps, the 3816 swap) form a separately reported, reviewed set.
3. **Preparation for FOOTBALLTrace E0** (later):
   - The exact algorithm: Savitzky–Golay with w = 7, n = 2, s = 7; θ = arccos breakpoint; tail swap.
   - Also report the literal min-cosine variant and the time-matched s (about 4 updates).
   - Five-fold search grouped by encounter and match.
   - Metrics: candidate recall and precision, swap precision, recall and F1, assignment accuracy, false-correction rate.

Not started: synthetic swaps, FOOTBALLTrace, baselines, any model training.
