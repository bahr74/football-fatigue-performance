# Session B M13: failure case studies (2026-11-21)

**Source:** `python -m football_fatigue.video.failures cases --data-root F:/Football-Fatigue-Project/data` → `data/processed/video/_diag/cases-m13-v1/{cases.json,cases.md}`.

This table uses saved pipeline-v1 observations only; no new inference was run.

**Selection:**
- Each category has a fixed rule (below).
- 2 cases are drawn per category with seed 17 from **all** qualifying cases (pool sizes listed below).
- The table also includes the fixed reference case from earlier handoffs.
- Nothing was picked by eye.

**Labels:**
- `T##` is a temporary ByteTrack ID of the named profile. `G##` is a SoccerTrack GT player.
- "GT visibility" comes from GT-GT box cover. It is **not** a confirmed visual occlusion.
- None of these cases has been labelled by a human as occluded.

| category | rule | qualifying cases |
|---|---|---|
| heavy-overlap disappearance | T0 no-box episode ≥ 5 frames, GT-derived heavy/hidden, not 117092 | 550 |
| turf-filter false rejection | B0 episode caused by the turf filter, ≥ 5 frames | 1,450 |
| low-confidence interruption | T0 episode caused by conf ≤ 0.4, ≥ 3 frames, visible/partial | 1,356 |
| confirmed identity switch | T0 temporary-ID change while the previous ID continued on another box | 733 |
| small-player missed detection | T0 no-box ≥ 5 frames, median GT height < 35 px, visible, not 117092 | 91 |
| low-light failure | T0 no-box ≥ 5 frames in 117092, visible | 490 |

## Table

Column definitions:
- **"recovered":** the same GT player was tracked again after the episode. "changed ID" means under a different temporary ID.
- **"old T on other player":** the previous temporary ID was output on a different GT player, either during the gap or within 2 s after it.

| category | match | profile | GT | frames | prev T | next T | conf before → after | height px | GT visibility, typical (max cover) | stage | recovered | old T on other player | open |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reference (earlier handoffs) | 117093 | T0 | G4 | 31–37 | T3 | T5 | 0.554 → 0.649 | 37.1 | heavy (0.764) | miss | yes, changed ID | G9 @f38 | `?source=soccertrack-117093&ti=T0&debug=1&gid=4&frame=30` |
| heavy_overlap_disappearance | 118576 | T0 | G20 | 2066–2076 | T17 | T67 | 0.462 → 0.672 | 45.0 | heavy (0.861) | miss | yes, changed ID | G21 @f2067 | `?source=soccertrack-118576&ti=T0&debug=1&gid=20&frame=2065` |
| heavy_overlap_disappearance | 118576 | T0 | G20 | 4657–4665 | T69 | T69 | 0.724 → 0.503 | 47.7 | heavy (0.917) | miss | yes, same ID | no | `?source=soccertrack-118576&ti=T0&debug=1&gid=20&frame=4656` |
| turf_filter_false_rejection | 117093 | B0 | G9 | 2314–2320 | T106 | T106 | 0.886 → 0.871 | 59.8 | visible (0.0) | rej_turf (T0: ok) | yes, same ID | no | `?source=soccertrack-117093&ti=B0&debug=1&gid=9&frame=2313` |
| turf_filter_false_rejection | 118575 | B0 | G21 | 2093–2098 | T9 | T9 | 0.837 → 0.667 | 46.5 | visible (0.0) | rej_turf (T0: ok) | yes, same ID | no | `?source=soccertrack-118575&ti=B0&debug=1&gid=21&frame=2092` |
| low_confidence_interruption | 117092 | T0 | G1 | 3473–3475 | T206 | T206 | 0.521 → 0.622 | 34.5 | visible (0.0) | rej_conf | yes, same ID | no | `?source=soccertrack-117092&ti=T0&debug=1&gid=1&frame=3472` |
| low_confidence_interruption | 118575 | T0 | G11 | 894–896 | T44 | T44 | 0.558 → 0.57 | 26.0 | partial (0.292) | rej_conf | yes, same ID | G19 @f896 | `?source=soccertrack-118575&ti=T0&debug=1&gid=11&frame=893` |
| confirmed_identity_switch | 117092 | T0 | G5 | 5898–5898 | T354 | T363 | 0.551 → 0.644 | – | – (–) | id_changed | yes, changed ID | G15 @f5881 | `?source=soccertrack-117092&ti=T0&debug=1&gid=5&frame=5874` |
| confirmed_identity_switch | 117093 | T0 | G21 | 4795–4795 | T176 | T83 | 0.552 → 0.782 | – | – (–) | id_changed | yes, changed ID | G19 @f4793 | `?source=soccertrack-117093&ti=T0&debug=1&gid=21&frame=4787` |
| small_player_missed_detection | 118575 | T0 | G12 | 782–792 | T52 | T52 | 0.629 → 0.501 | 32.0 | visible (0.0) | miss | yes, same ID | G14 @f822 | `?source=soccertrack-118575&ti=T0&debug=1&gid=12&frame=781` |
| small_player_missed_detection | 118576 | T0 | G16 | 2968–2983 | T14 | T14 | 0.663 → 0.591 | 29.85 | visible (0.5) | miss | yes, same ID | no | `?source=soccertrack-118576&ti=T0&debug=1&gid=16&frame=2967` |
| low_light_failure | 117092 | T0 | G12 | 443–596 | T3 | T69 | 0.62 → 0.572 | 32.05 | visible (1.0) | miss | yes, changed ID | no | `?source=soccertrack-117092&ti=T0&debug=1&gid=12&frame=442` |
| low_light_failure | 117092 | T0 | G2 | 3723–3793 | T198 | T234 | 0.631 → 0.726 | 22.2 | visible (0.5) | miss | yes, changed ID | no | `?source=soccertrack-117092&ti=T0&debug=1&gid=2&frame=3722` |

To open a case: start the player (handoff launch command), then append the "open" query to `http://127.0.0.1:5831/`. Use **before / during / after** in the Debugger's failure panel.

## What the cases show (from the saved observations above)

1. **Heavy-overlap disappearance.** Examples: 117093 G4 f31–37; 118576 G20 f2066–2076 and f4657–4665.
   - The player's box vanishes **after the merge** while a neighbouring GT player keeps a box. Every frame carries the `merge_suspect` hint (G4: 7/7; G20: 7 of 8).
   - In two of the three cases the player came back under a new temporary ID. In two of the three, **the old ID went to the neighbour** (G4's T3 → G9 at f38; G20's T17 → G21 at f2067).
   - This is the single most damaging pattern for identity.
   - Whether the model never proposed the hidden player, or per-tile NMS (IoU 0.7) / the cross-tile merge (IoU 0.5) removed it, is **unknown until A2-cap**.
2. **Turf-filter false rejection.** Examples: 117093 G9, 118575 G21.
   - Confident (0.84–0.89), unoccluded (cover 0.0) players of 47–60 px are deleted by the grass-colour filter in B0.
   - In T0 the same frames are `ok`. This is a pure post-processing loss and is why T0 is the baseline.
3. **Low-confidence interruption.** Examples: 117092 G1, 118575 G11.
   - 3-frame dips below 0.4 on small players (26–35 px).
   - Both kept the same ID afterwards: ByteTrack's 62-frame buffer bridges short dips.
   - In the 118575 case, G11's T44 was nevertheless output on G19 at f896, a near-swap during a partial overlap.
4. **Confirmed identity switch.** Examples: 117092 G5 (after a 19-frame gap), 117093 G21 (3-frame gap).
   - The new ID was assigned while the old ID continued on another GT player (G15, G19).
   - Both switches follow a gap. This matches the M12 finding that 72 % of ID changes come after a detector gap.
5. **Small-player missed detection.** Examples: 118575 G12 (32 px), 118576 G16 (30 px).
   - No box, with no overlap hint (`none`). These are not occlusions.
   - ByteTrack bridged both with the same ID.
   - In the G12 case, T52 still appeared on G14 at f822, about 1.2 s after the episode.
6. **Low-light failure.** Examples: 117092 G12 f443–596 (154 frames), G2 f3723–3793.
   - Long no-box runs on 22–32 px players with mostly `none` hints. Both players returned under new IDs.
   - These gaps are longer than any tracker buffer, so no association change fixes them. Only better detection does.
   - In the M13 complementarity analysis, YOLOv8m found 73 of 96 production misses in 117092 benchmark frames.
