# M11: Step 1 verification, event-centred 5 s contexts (v3) and dashboard corrections

Date: 2026-11-11. Development matches only (40). Nothing committed.

## 1. Summary

- **Step 1 cleaning is correct.** It was verified independently on all 40 development matches, including a full re-parse of the RAW files. No defect was found, so Step 1 was not changed.
- **No code bug in detection.**
- **The context window was asymmetric by design.** v2 centred each context on the nearest whole game second (the advisor's window grid), not on t*. It also exported only segments lying entirely inside the window.
- **v3 re-extracts only the trajectory contexts.** Each context now covers exactly [t* − 2.5 s, t* + 2.5 s]:
  - The window is clipped only to the match period.
  - At the window edges, positions are interpolated on the valid segment that straddles the edge.
  - Detection, thresholds, event IDs and event records are unchanged (reused from v2).
- **All 30,438 encounters pass the detection and tracking-support checks** on their new contexts.
- **1,766 encounters are presentation-ready** (v2: 1,622).
- **The dashboard** now uses one fixed axis, −2.5 … +2.5 s, for every event. At times without valid support it reports positions as unavailable instead of inventing them.

## 2. Phase 1: Step 1 verification

The verifier is `python -m football_fatigue.step1_verify --root data/raw --jobs 3`. It writes to `data/processed/m11-step1-verification-v1/` (per_match.csv, details.json, summary.json, manifest.json).

Only the production segmentation (`intersections.exact.trajectories`) is executed. Everything else is re-derived without calling `encounters.quality`.

| Cleaning rule | Implementation | Verification | Result | Defect | Action |
|---|---|---|---|---|---|
| RAW parsing | `MatchData(RAW)` → `RawCache` | Re-read every `.jsonl.bz2` record with plain `json`; compare frame, video time, period, (side, jersey) slot, state, x/y, visibility | 7,432,290 frames, 0 mismatches in any field | none | none |
| Repeated records | consecutive equal `frameNum` collapsed; first record kept; conflicts → AMBIGUOUS | counted in the re-parse | 27,084 repeated records, 0 conflicting | none | none |
| Timestamps / periods | `videoTimeMs/1000`, canonical period | strictly increasing within each period, periods contiguous | 0 duplicate timestamps, 0 non-monotonic | none | none |
| IDs, teams, jerseys | slot = (side, jersey), roster join | unique slots, one team per side, unique player IDs, jerseys checked in the re-parse | all pass; 0 slots without a roster entry; 0 ambiguous samples | none | none |
| Observed / estimated | provider `visibility`; `ESTIMATED` mask bit | bit equals visibility for every sample | pass; ESTIMATED share is 53–73 % of available samples per match | none | none |
| Repeats / held positions | equal to the immediately preceding frame → repeat; runs ≥ 3 frames → held | recomputed independently | identical masks | none | none |
| Unobserved-gap rule (0.5 s) | step valid if ≤ 0.5 s without an available sample, same period | steps recomputed independently, compared exactly | identical (81,826,714 steps). All 938 invalid steps are period changes; **no within-period gap above 0.05 s exists** | none | none (see §2.1) |
| Jump rule (> 12 m/s and > 1 m) | shortest change interval | recomputed, compared exactly | identical; 385,029 jumps (45,943 in frame-wide shifts) | none | none |
| Segmentation | joined only through valid, non-jump steps between attributed samples | every segment checked: is a usable step, endpoints available, no gap, no period crossing, no jump, held segments constant, joins continuous | 80,086,285 segments, 115,375 runs, 0 defects | none | none |
| Interpolation within support | linear inside a segment only | as above; v3 audit re-derives every window-edge point from RAW | 0 defects | none | none |
| Provider clones | bit-identical positions are kept and flagged (`duplicate_positions`), never treated as verified | counted; flagged downstream in the audit | 2,530,865 duplicated samples; 1,243,070 moving segments between duplicated positions | none (flagged by design) | kept as an artifact reason |
| Flag propagation | context flags = OR of the Step 1 bits of the vertices | v3 audit checks every vertex against the cache (visibility, period, state) and reproduces the bits | pass | none | none |

**Representation note (not a defect).** When a valid run ends in a held run, the run's last repeated frame is not a vertex. This happens 9 times in 40 matches and shortens support by at most one frame. It is conservative, so it was left unchanged.

### 2.1 Thresholds against the data

- **The 0.5 s gap rule never applies inside a period.** The provider fills every unobserved interval with ESTIMATED positions, so trajectories are cut by jumps, attribution problems and period changes, never by gaps.
- **"Missing tracking" therefore usually means "estimated", not absent.** This is why ESTIMATED share drives readiness.
- **Valid visible steps:**
  - median speed 1.9 m/s;
  - p99 8.8 m/s;
  - p99.9 13.9 m/s.
- **Speeds between 12 and 20 m/s:**
  - 220,209 of these steps are joined;
  - all are sub-metre over-speed steps (noise), which is the documented reason for the 1 m component of the jump rule.
- **Jump distances:**
  - 56 % of jumps are 1–1.5 m;
  - 136 exceed 5 m.
- The thresholds are plausible and were not changed.

## 3. Phase 2: why the old window was asymmetric

`intersections/proximity.py` follows the advisor's window framework:
- Windows are centred on every whole game second c of the period (period-relative), spanning [c − 2.5, c + 2.5].
- An episode is counted in every window it overlaps.
- Each episode kept the context of its *canonical window*, the one whose centre is nearest t*.

This had three consequences:

1. **The context was off-centre by up to 0.5 s.** It ran from t* −2.0 … −3.0 s to t* +2.0 … +3.0 s; the most extreme relative bounds were −3.18 s and +3.00 s (near period edges). Only 2,880 of 30,438 v2 contexts started exactly at t* − 2.5 s.
2. **Up to one segment (≤ 0.5 s + 1 frame) could be missing at each edge.** `advisor.run_bounds` keeps only segments lying entirely inside the window, so a joined segment straddling an edge was dropped. This is why the earlier audit needed a tolerance of 0.53 s.
3. **The windows never affected detection.** Episodes come from all pair-intervals of the full trajectories; windows only label window-level detections and select the exported context. Re-centring is therefore safe.

All source data needed to rebuild the contexts is available: the checksum-verified Step 1 cache, mask and steps, and the v2 events.

Example events:
- **P1 35:53.343 (3812):** v2 window −2.825 … +2.175 s; v3 −2.5 … +2.5 s, complete.
- **Case A (3812 `7d69f374b095e733`):**
  - v2 window −2.96 … +2.04 s;
  - v3 −2.5 … +2.236 s, because period 1 ends there;
  - Player B's valid incoming track still starts at −0.467 s (a tracking jump), and nothing is bridged.

## 4. Phase 3: the event-centred context (`m11-sync-proximity-v3`)

**Command.** `python -m football_fatigue.proximity_context --root data/raw --jobs 3`. Code: `src/football_fatigue/proximity_context.py`. It lives outside `intersections/` so that v2's code hashes stay valid.

**Window**
- Nominal window = [t* − 2.5 s, t* + 2.5 s] around the continuous t*, never rounded.
- The in-period window is clipped to the period's first and last frame (`window_clipped_by_period`; 173 events).

**Each player's trajectory**
- It is the Step 1 valid run containing the player's segment at t*: the same run as v2, no other runs added, nothing joined across a cut.
- Original samples inside the window are kept with their source row, frame, visibility and quality bits.
- Where a joined segment straddles a window edge, one point is interpolated at the edge on that segment:
  - `sample_is_boundary`, source row −1;
  - ESTIMATED if either end is ESTIMATED;
  - the flags of both ends.
- Nothing is extrapolated.

**Unavailable parts of the nominal window.** They are listed per player as `[start, end, reason]`. The reasons are:
- `implausible_jump`;
- `gap_period_or_time`;
- `unattributed_or_ambiguous`;
- `no_adjacent_sample`;
- `period_boundary`;
- `outside_period`.

Each player record also gives the covered interval and the incoming / outgoing seconds.

**Synchronized grid.** Both players at the union of their vertex times plus t_start, t* and t_end, inside the shared support, as in v2.

**Checks during the build.** Every context reproduces the recorded A(t*) and B(t*) to within 1e-6 m.

**Outputs, per match**
- `root.contexts.jsonl`: line k = v2 line k, same `event_id`, with the v2 window kept for reference.
- `root.trajectories.npz`: v2 layout plus `sample_is_boundary`.
- `root.summary.json` and `root.manifest.json`: code hashes, the v2 events/manifest hashes, Step 1 provenance, and `raw_unchanged`.

**Development index**
- `event_mapping.csv`: v2 → v3 line identity, plus old and new windows.
- `summary.json` and `manifest.json`.

v3 occupies 112 MB; events are not duplicated. Raw data, Step 1 caches, `m11-sync-proximity-v2` and `m11-sync-proximity-v2-audit` were not modified; their hashes are re-checked by tests.

## 5. Phase 5: revalidation (`m11-sync-proximity-v3-audit`)

**Command.** `python -m football_fatigue.proximity_audit_v3 --root data/raw --jobs 3`. Optional: `--min-observed-share 0.5 --min-path-m 1.0`.

The audit uses the v2 audit's geometry, which shares no code with the builder. In addition it re-derives the following from the RAW cache:
- the window (period limits recomputed);
- every window-edge point (it must lie on a valid joined step of the RAW update instants);
- every cut inside the window. The next step must be unjoinable; if it were joinable, valid support would have been lost.

| Status | v2 (integer-second window) | v3 (t* ± 2.5 s) |
|---|---|---|
| Encounters | 30,438 | 30,438 |
| Mathematically valid | 30,438 | 30,438 |
| Tracking support valid | 30,438 | 30,438 |
| Incoming complete (both players) | 25,602 | 25,573 |
| Outgoing complete (both players) | 26,607 | 26,540 |
| Context complete | 24,088 | 24,052 (106 gained, 142 lost) |
| Window clipped by the period | n/a | 173 |
| Provider artifact suspected | 6,578 | 6,634 |
| **Presentation-ready** | **1,622** | **1,766** (168 gained, 24 lost) |

**Why the counts change**
- The context-complete changes are the window shift: up to ±0.5 s of extra or less context on each side.
- Artifacts can appear or disappear inside the moved window (99 gained, 43 lost).
- **Presentation-ready gains (168)** were previously excluded by:
  - observed share < 50 % (158), a threshold-sensitive rule: a 0.5 s shift changes about 7 of about 75 samples;
  - incomplete context (11);
  - a jump flag (4).
- **Presentation-ready losses (24)** fail under v3 because of:
  - observed share (12);
  - a jump flag inside the new window (8);
  - incomplete context (11);
  - period clipping (3).

  These reason counts overlap; one event can have several.

**Presentation-ready**
- All 1,766 ready encounters are observed/observed (0 mixed or estimated).
- Per match: from 20 (10512) to 98 (3812).
- 11 ready events are outside v2's high-quality subset. That subset requires the *v2* context to be complete, and these 11 had incomplete v2 contexts (all 11 checked).

**Encounter types**
- 27,670 closest synchronized approaches.
- 40 exact synchronized coincidences (single instant).
- 2,728 potential tracking ambiguities (identical provider positions).

**Paths that also cross in space:** 27,063 events have the two paths crossing; only 2,595 of those crossings happen at the same instant.

**Dataset checks** (all pass): v3 manifest hashes; v2 events unchanged; same events in the same order; t* unchanged; npz rows; unique IDs; mapping identity; index hashes.

**Readiness rules** (configurable; recorded in the manifest). An event is presentation-ready when:
- it is mathematically valid and its tracking support is valid;
- both players' incoming and outgoing tracks cover the full t* ± 2.5 s;
- no artifact is suspected (identical or duplicated positions, critical Step 1 flags in the context);
- both players are observed at t*;
- at least 50 % of each player's original context samples are observed;
- each player moves at least 1 m.

## 6. Phase 6: dashboard (same app, `encounters_app.py`)

- **Data.** v2 events; the v3 context replaces the record's `context` (v2 kept as `context_v2`); statuses come from the v3 audit.
- **Fixed axis.** The slider is the nominal axis −2.5 … +2.5 s in 0.1 s steps, with 0 = t*, for every event. Before / At encounter / After jump to −1, 0 and +1 s.
  - At a selected time, each player's position on its valid run is shown; otherwise the player is shown as **unavailable (reason)** and no distance is given.
  - I chose this over a valid-only slider because the axis is then identical for every event, while nothing is invented: interpolation happens only on valid segments, the extraction's own model.
- **New support bars** (Player A, Player B, both), drawn on the same axis. Grey marks no valid track; hovering gives the reason.
- **Distance chart.** Fixed x-range. Grey where there is no shared support; the selected time is marked.
- **Pitch**
  - unchanged: equal scaling, auto-zoom, A drawn wider under B, direction arrows, cut markers;
  - window-edge interpolations are not drawn as samples;
  - period ends are not drawn as cuts.
- **Presentation mode** shows only the v3 presentation-ready events.
- Wording: "proximity, not a visually confirmed occlusion" is kept. Nothing refers to a trained identity model.
- **Browser check.** Headless Edge at 1600 px; screenshots in `data/processed/m11-sync-proximity-v3-audit/screenshots/`:
  - case A;
  - case C at t* −1 s, both players shown unavailable because of the jump;
  - case D;
  - presentation mode at t* and at +1 s.

## 7. Tests

`pytest tests/test_proximity_context.py tests/test_m11_encounter_explorer.py tests/test_proximity_audit.py tests/test_m11_dashboard.py tests/test_intersections_proximity.py`: **106 passed** (15 + 20 + 16 + 29 + 26).

The new and updated tests cover:
- **Step 1 rules:**
  - gap segmentation, including the independent re-derivation equal to Step 1;
  - jump vs over-speed segmentation;
  - observed/estimated preservation;
  - no bridging across gaps or jumps.
- **Window and context:**
  - exact event-centred window at a non-integer t*;
  - edge interpolation checked against RAW;
  - partial coverage;
  - period clipping.
- **Encounter types:** exact coincidence; nonzero proximity; provider clones.
- **Audit tamper detection:** moved edge point, dropped support, integer-second window, wrong cut reason.
- **Dashboard:**
  - fixed axis on all figures;
  - unsupported slider times (AppTest);
  - Plotly paths limited to valid runs;
  - presentation filtering.
- **Provenance:** v2 → v3 reference identity; v3 / v2 / v2-audit hashes and code provenance; refusal of validation and test matches.

## 8. Limitations

- **Detection is unchanged.** The integer-second windows remain only in v2's window-level counts (`windows_game_s`, `canonical_window_game_s`).
- **Readiness thresholds are presentation rules, not scientific labels.** The 50 % observed share is close to many events and is sensitive to the window (§5).
- **Positions are provider-estimated more than half the time** (53–73 % per match). Estimated tracks are shown hollow but cannot be verified.
- **Proximity is not occlusion.** Slots are provider labels, not verified identities. No model is trained here.
- **Provider clones and jump-cut tracks are data artifacts.** They are flagged and excluded from presentation, never repaired.

## 9. Launch

```powershell
.\.venv\Scripts\python.exe -m football_fatigue.m11_dashboard --port 8503 encounters
```

Options go *before* the subcommand (`... encounters --port 8503`, as given in the previous summary, is rejected by the CLI). Then open http://127.0.0.1:8503.
