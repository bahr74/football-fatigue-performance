# Dataset schema audit (Milestone 1B)

Status: **implemented, pending independent review** (revised after the Codex review). Milestone 1B is not formally complete. Everything below was derived by reading the actual files in `data/raw/` and the supplied PFF specifications, on 2026-10-15. Read-only; no raw file was changed. Quantitative findings marked *reproducible* are produced by the read-only scripts in [scripts/audit](../scripts/audit/README.md); findings marked *exploratory* come from temporary scripts that are not retained in the repository and should be re-checked before being relied on. Companion documents: [event_version_comparison.md](event_version_comparison.md) (Event Data versions and the provisional version policy) and [data_contracts.md](data_contracts.md) (proposed internal structures). File layout and licensing caveats stay in [data_policy.md](data_policy.md).

**Evidence labels.** **[SPEC]** stated by a PFF document. **[OBS]** directly observed in the files. **[INF]** inference or provisional recommendation. **[OPEN]** unresolved. A statement carries no label only if it is plain context.

## 1. Method and coverage

The key quantitative checks are reproducible with the standard-library scripts in [scripts/audit](../scripts/audit/README.md) (they locate files through `football_fatigue.inventory`, never write inside `data/raw/`, never decompress to disk, and stream tracking files one record at a time). Everything else was inspected with temporary read-only Python 3.12 scripts (`bz2`, `json`, `csv`, `zipfile`; `pdftotext` for the PDFs) that are **not** retained; those results are labelled *exploratory*. No dependency was installed. The earlier session's scratch results were used only as leads.

| Check | Scope (denominator) | Status | Runtime on this machine |
| --- | --- | --- | --- |
| Tracking: records, unique frames, repeated-frame records, period coverage, missing ball, estimated positions, `speed` presence, smoothed-list duplicates (`tracking_scan.py`) | **All 64 files, every record** (11,892,765 records; 11,849,815 unique frames) | reproducible | about 12-17 minutes with 4-7 workers |
| Event version row counts, repeated keys, collision-safe matching, changed rows/values, position differences (`event_versions.py`) | **All 64 matches × 3 versions** (144,541 + 97,444 + 144,541 rows) | reproducible | about 1 minute |
| Event ↔ tracking temporal coverage and coordinate agreement, every row of every version (`event_alignment.py`) | **All 64 matches** | reproducible | about 20 minutes with 4 workers |
| Suspected side/identity anomalies and extra-time coverage (`extra_time_identity.py`) | **All 64 matches** (events); tracking for the five extra-time matches (three have extra-time frames) | reproducible | about 4-5 minutes |
| Cadence of raw positions (first 10,000 records per file), implied-speed counts, `speed` magnitudes, kickoff/goalkeeper/shot-side orientation tests, event↔roster id relations, metadata/roster/`players.csv` relations | all 64 matches or samples as stated in each section | **exploratory, not retained** | - |
| Specification text vs files | Tracking v2.2; Event v2.4 and v2.5; change log, read in full | manual | - |

**Not done (explicitly out of scope or expensive):** checksums and provenance manifest; licensing verification; any physiological or fatigue validation; any cleaning, resampling, or production synchronization; per-frame jump analysis beyond the aggregate counts below; video (none exists).

**Sample matches used for narrative examples:** `3812` (reference, group stage week 1, 2 periods, 105×68 pitch, frame 5578 for the smoothed-list conflict), `3813`, `3814` (smoothed-list conflicts and null smoothed ball), `10506`, `10508`, `10517` (extra time with tracking), `10510`, `10511` (extra time without tracking), `10502`, `3816`, `3859` (event coordinates that disagree with tracking, see the event comparison).

## 2. Dataset overview

- **[OBS]** 64 matches, all with tracking, events (3 versions), metadata, rosters. IDs `3812`–`3859` (48, metadata `week` 1–3, group stage) and `10502`–`10517` (16, `week` 4–8, knockout). `competitions.csv` lists exactly these 64 IDs (season `2022`, competition `38`, name "FIFA Men's World Cup"). Match IDs in tracking file names, `gameRefId`, metadata `id`, event `gameId`, and `competitions.csv` agree for all 64.
- **Stages [INF]:** metadata gives `week` only (16/16/16 for weeks 1–3; 8, 4, 2, 1, 1 for weeks 4–8). No explicit stage field exists; the mapping to round of 16, quarter-finals, semi-finals, third place, and final is inferred from the counts.
- **Extra time [OBS]:** five matches have periods 3 and 4 in the events: `10506`, `10508`, `10510`, `10511`, `10517` (metadata `homeTeamStartLeftExtraTime` is non-null for exactly these five). Tracking contains periods 3 and 4 for only **three** of them (`10506`, `10508`, `10517`); tracking for `10510` and `10511` ends after period 2 while events continue (738 and 694 extra-time event rows fall in periods that tracking lacks). **In periods 3 and 4 of `10506`, `10510`, `10511`, `10517` the position lists show a `suspected_side_identity_anomaly` relative to the teams named by the events (§7: a flag, not a confirmed swap); `10508` shows none.** For `10510` and `10511` extra time is **event-only** (no tracking frames).
- **Match length [OBS]:** per-match tracking has 170,717 (`10510`, no extra time in tracking) to 254,242 (`10517`, with extra time) unique frames. Period 1 runs 2,758–3,542 s of `periodElapsedTime`, period 2 2,931–3,529 s.
- **Substitutions [OBS]:** 5–13 `SUB` game events per match; rosters have 45–52 entries (11 starters per team in every match).

## 3. Sources reviewed

| Source | Location | Version/date | Notes |
| --- | --- | --- | --- |
| PFF FC Tracking Data Specification | `Tracking Data-…-001/Tracking Data/PFF FC Tracking Data Specification v2.2.pdf` | v2.2, October 2023 | Eight sections, 8 pages (§2 data fields p.3, §3 players p.4, §4 ball p.4, §5–6 events p.5–6, §7 coordinates p.7, §8 metadata p.8) |
| PFF FC Event Data Specification v2.5 | `Event Data-…-002/Event Data/PFF FC Event Data Specification v2.5.pdf` | v2.5, June 2025 | Sits beside the unlabelled root event files |
| PFF FC Event Data Specification v2.4 | `Event Data-…-002/Event Data/May 1, 2025/…v2.4.pdf` | v2.4, April 2025 | Sits in the `May 1, 2025` folder |
| PFF FC Change Log | `data/raw/PFF FC Change Log.docx` | entries 2025-03-14, 2025-05-01, 2025-06-03 | Read from the document XML; compared with the files in the event comparison |

The tracking specification v2.2 predates the observed tracking format in several respects (§9).

## 4. Tracking files

Path pattern: `data/raw/Tracking Data-<ts>-00{1,2}/Tracking Data/<matchId>.jsonl.bz2`, one JSON object per line, UTF-8, bz2. Every record has the same 19 top-level keys; no extra keys exist.

### 4.1 Top-level fields (`$.<name>`, observed in 11,892,765 records)

| Field | Type observed | Spec meaning [SPEC] | Observed [OBS] | Presence | Project use [INF] |
| --- | --- | --- | --- | --- | --- |
| `version` | string, **first record of each file only**, else `null` | Version of Sportlogiq tracking output | `"4.1.0"` in all 64 | 64 non-null of 11.9 M | Provenance only |
| `gameRefId` | float (e.g. `3812.0`), first record only, else `null` | Unique game id (spec says int) | equals file name in 64/64 | first record only | Cross-check against file name |
| `generatedTime`, `smoothedTime` | ISO-8601 string, first record only | Generation and smoothing times | `generatedTime` 2023-07/08 for 50 files, 2024-01/08 for 14 (exploratory) | first record only | Provenance; see `speed` below |
| `videoTimeMs` | float | Milliseconds since start of video | equals `frameNum / 29.97 × 1000` (e.g. frame 6211 → 207240.574) | always | **Primary time key** |
| `frameNum` | int | Frames since video start, from 0 | starts at 1,214–6,313 (first frame = period-1 kickoff, not 0); steps are 0 or +1 within a period | always | Frame id |
| `period` | int | Half number; 3 and 4 are extra time | 1, 2 (61 matches); 1–4 (3 matches) | always | Period id |
| `periodElapsedTime` | float seconds | Seconds since start of the half (floor 0) | 0 at each period start, strictly consistent with frame steps | always | Within-period clock |
| `periodGameClockTime` | float seconds | Estimated game-clock time | equals elapsed time in period 1; **starts at 2700.0 in period 2** (59 matches with metadata), 5400.0 / 6300.0 in periods 3 / 4 | always | Display clock only (nominal offsets) |
| `homePlayers`, `awayPlayers` | list of dict | Player info incl. jersey, confidence, visibility, x, y | 11 entries in 99.94 % of frames; 12–15 around substitutions | always | Raw positions |
| `homePlayersSmoothed`, `awayPlayersSmoothed` | list of dict, **`null` in the first record of period 2 (61 files) and period 4 (3 files): 128 lists** | Kalman-smoothed positions | length 11 in 99.9 % of lists; **repeated jersey numbers with conflicting coordinates in 6,094 lists (3,047 frames) and in 85,524 of the 85,900 lists of repeated-frame records (§4.2b)**; 8–10 entries in a few lists | non-null 11,892,701 records | Smoothed positions, **ambiguous where jerseys repeat**; see §4.2b |
| `balls` | list of dict | Only the estimated in-play ball | **empty in 3,855,434 of 11,849,815 unique frames (32.54 %; 21.3–47.1 % per match)**, otherwise length 1 | always | Raw ball |
| `ballsSmoothed` | dict or `null` | Ball smoothed using events | `null` in 3,855,460 unique frames; **`x` and `y` are `null` while `z` is set in 1,404,715 of the 7,994,355 present dicts (17.57 %)** | 7,994,355 present | Use only after null checks |
| `game_event_id`, `possession_event_id` | float or `null` | Event ids | non-null in 3,263,344 / 128,700 frames | optional | Cross-reference only (§6) |
| `game_event`, `possession_event` | dict or `null` | See §5 | non-null in the same frames as the ids | optional | Cross-reference only |

### 4.2 Player entries (`$.homePlayers[]`, `$.awayPlayers[]`)

| Path | Type | Meaning | Observed |
| --- | --- | --- | --- |
| `jerseyNum` | **string** (spec: int) | Estimated jersey number | no `playerId` exists in tracking; no repeated jersey number occurs in any of the 23,699,630 raw lists (frame × side); repeats occur only in smoothed lists (§4.2b) |
| `confidence` | string `LOW`/`MEDIUM`/`HIGH` | Confidence of the jersey identification | `LOW` 68.2 %, `HIGH` 24.3 %, `MEDIUM` 7.4 % of 260,712,096 raw entries (reproducible) |
| `visibility` | string `VISIBLE`/`ESTIMATED` | Tracked in video vs estimated from context | `ESTIMATED` **62.70 % pooled (163,458,531 of 260,712,096 raw entries); 48.8–74.9 % per match** (reproducible) |
| `x`, `y` | float, metres | Pitch coordinates (§7) | never `null` in any player entry, raw or smoothed (all 64 files) |
| `speed` | float | **not in the specification** | present on every raw player entry of 14 files and none of the other 50 (file-level, never partial; reproducible: 59,619,422 entries); never on smoothed entries; the 14 files are those with `generatedTime` in 2024 (exploratory); magnitudes (median ≈ 1.4, p99 ≈ 6.5, max ≈ 12.2 in match 3812) are consistent with m/s but **the unit is [OPEN]** |

Raw `x,y` are held across frames (**exploratory**, not reproduced by `scripts/audit`): in the first 10,000 records of each of the 64 files, 49.95–49.98 % of consecutive same-period frame pairs have **identical raw (x, y)** for the same jersey, in runs of exactly two frames (e.g. match 3812/3813 probes), i.e. raw positions update about every second frame (≈15 Hz effective, not 29.97 Hz). Smoothed positions change nearly every frame (0.0–1.3 % identical). **[OBS]** for the sampled windows only; whether the cadence holds for the whole match was not tested.

Raw positions are also noisy (**exploratory**): of 260,708,625 consecutive raw position pairs, 5,363,961 (2.06 %) imply more than 12 m/s, with single-frame steps up to 53.1 m. The earlier smoothed implied-speed counts were contaminated by the repeated smoothed jerseys and are withdrawn.

### 4.2b Repeated jerseys in smoothed lists (reproducible: `tracking_scan.py`, all 64 files)

Reproduce: `scripts\audit\tracking_scan.py show-frame 3812 5578` (the home and the away list each have 22 smoothed entries for 11 jerseys) and `tracking_scan.py --matches 3814`.

- **Lists and denominators.** 23,699,630 smoothed lists (11,849,815 unique frames × 2 sides): 23,693,408 have unique jerseys, **6,094 contain repeated jerseys, 128 are `null`**. List lengths: 11 (23,678,065), 22 (6,090), 33 (2), 44 (2); 8–10 and 12–15 entries occur around substitutions.
- **Identical vs conflicting.** 67,034 repeated-jersey groups: **67,017 are conflicting** (the candidates have different `x` or `y`) and 17 are identical. Group sizes: 2 (66,990), 3 (22), 4 (22). The candidates of a conflicting group lie within 0.05 m (30,845 groups), 0.10 m (24,342), 0.50 m (11,822), and farther than 0.5 m in 8 groups. Every conflicting group has a raw entry for the same jersey at the same frame; the distance of a candidate to that raw entry has a per-match median of 0.45–1.02 m and a maximum of 32.9 m.
- **Where it occurs.** Repeated jerseys occur only in frames that also have repeated source records (same `frameNum`): of 3,235 such frames, 3,047 have a repeated-jersey smoothed list and 188 do not; none of the 11,846,580 frames without repeated records does. In the 42,950 repeated-frame records the smoothed lists have repeated jerseys in 85,524 of 85,900 lists (99.6 %) and equal the first record's smoothed lists in all cases.
- **Examples [OBS].** Match `3812`, frame 5578 (period 1): home jersey 3 has raw (−0.641, −8.792) and smoothed candidates (−0.276, −9.221) and (−0.325, −9.194), both `ESTIMATED`/`LOW`; home jersey 5 has candidates (−34.428, −7.060) and (−34.571, −6.991) against raw (−34.122, −5.219). Match `3814`, frame 3577: home jersey 1 has raw (−28.514, 0.109) and candidates (−28.357, 0.114) and (−28.344, 0.129); in `3814`, 84 of 360,886 smoothed lists are conflicting (924 groups, all conflicting).
- **Consequence.** List order carries no information that identifies the correct player entry; no candidate may be selected by default (for example "the first entry"). The earlier suggestion to use the first complete entry is withdrawn; the proposed contract keeps all candidates and marks the smoothed position unavailable, with an explicitly flagged raw fallback ([data_contracts.md](data_contracts.md) §6).

### 4.3 Ball entries (`$.balls[]`, `$.ballsSmoothed`)

`{visibility: "VISIBLE"|"ESTIMATED", x, y, z}`, metres. Raw ball `x` spans −80.68…76.27, `y` −48.56…44.44, `z` −7.75…26.32 across all files, so the ball can lie well outside the 105×68 pitch and can have negative `z` (negative height is physically implausible; **[OPEN]** whether it is estimation error). `ballsSmoothed` is not simply a smoothed copy of `balls`: at the kickoff of match 3812 raw is (−0.97, 0.29) and smoothed (2.46, 1.58), consistent with the specification (smoothed ball is placed at the carrier between events).

## 5. Events embedded in tracking

`$.game_event` (3,263,344 frames; present on every frame of the game event's duration) and `$.possession_event` (128,700 frames; the frame of the possession event):

- `game_event`: `game_id` (int), `game_event_type` (str), `formatted_game_clock`, `player_id`/`player_name`/`shirt_number`/`position_group_type`/`team_id`/`team_name` (**strings**, `null` on 9,310/8,058 frames), `start_time`, `end_time` (`null` on 6,280 frames), `duration`, `inserted_at`, `updated_at`, `video_url`, `home_team` (**int 0/1**, spec says boolean), `sequence`, `home_ball` (bool), `start_frame`, `end_frame`. Types seen: `OTB`, `OUT`, `SUB`, `OFF`, `ON`, `END`, `FIRSTKICKOFF`, `SECONDKICKOFF`, `THIRDKICKOFF`, `FOURTHKICKOFF`, `G`, and **`CLK` (6 frames), which the specification does not list**.
- `possession_event`: `game_id`, `game_event_id`, `possession_event_type` (`PA`, `CH`, `RE`, `CL`, `CR`, `SH`, `BC`), `formatted_game_clock`, `start_time`, `inserted_at`, `updated_at`, `start_frame`.
- **[INF]** This is a thin event overlay. It lacks event positions, outcomes, and the `IT`/`TC`/`FO` types of the Event Data files. The Event Data files are the event source; the tracking overlay is only a cross-reference.

## 6. Timing and synchronization feasibility

**Clocks identified [OBS]:**

| Clock | Where | Unit/reference | Semantics |
| --- | --- | --- | --- |
| `videoTimeMs` / `frameNum` | tracking | ms / frames since video start | Monotonic across the whole file; jumps only between periods (half-time gap). `videoTimeMs = frameNum / fps × 1000` with fps 29.97 (frame interval 33.367 ms; all 11,849,681 within-period steps round to 33.4 ms) |
| `periodElapsedTime` | tracking | s since the period's first frame | Resets to 0 at every period |
| `periodGameClockTime` | tracking | s, nominal | Period 1 = elapsed; period 2 starts at 2700; periods 3/4 start at 5400/6300. Not a stoppage-corrected match clock |
| `startPeriod1/2`, `endPeriod1/2`, `halfPeriod`, `period1/2` | metadata | s in video, s, minutes | `startPeriod1` equals the tracking period-1 start video time to ±1 ms in all 59 matches that have it; `startPeriod2` equals the period-2 start to within one frame (0–33 ms); `endPeriod1`/`endPeriod2` to within one frame (−33…+14 ms). `halfPeriod` = `startPeriod2` − `endPeriod1` (3812: 36.603 s). **Null in all five extra-time matches** |
| `startTime`, `endTime`, `eventTime`, `duration` | event | s since video start | `eventTime` is the time of the possession event; `startTime` is the start of the enclosing game event (different from `eventTime` in 56,123 of 144,541 events, by up to 70.3 s) |
| `startGameClock` | event | integer seconds, game clock | differs from tracking `periodGameClockTime` at the matched frame by a per-match median of up to 1.8 s (p99 up to 8 s, max 70.9 s) |

**Three statuses, kept apart.** (1) *Temporal coverage*: a same-period tracking frame exists within the alignment tolerance of the event time. (2) *Coordinate agreement*: the event's player coordinates equal the raw tracking positions at the selected frame. (3) *Validated correspondence*: the event is reliably associated with the correct player, team, period, and physical action; this needs independent ground truth (for example video) and **was not assessed**. Statuses (1) and (2) are necessary, not sufficient, for (3).

**Provisional temporal alignment policy** (implemented in `scripts/audit/audit_lib.py`, `Aligner`; one policy, one tolerance):

| Element | Definition |
| --- | --- |
| Units | event `eventTime` and frame `videoTimeMs / 1000`, both seconds of video time |
| Same-period matching | only frames whose source `period` equals the event's `gameEvents.period` are candidates |
| Nearest frame | the candidate with the smallest absolute time difference |
| Maximum residual | **one frame interval, `1 / fps` = 33.37 ms** (a single tolerance; there is no second, looser threshold) |
| Tie-breaking | exact ties (difference below 1 ns) choose the **earlier** frame (deterministic: frames are processed in increasing time and only a strictly smaller residual replaces a choice) |
| Duplicate frame records | records with a repeated `frameNum` collapse to the first record; the selected `frameNum` is the frame provenance |
| Period boundaries / missing frames | no extrapolation: an event outside its period's first-to-last frame range (± tolerance) is unmatched (`outside_period_frame_range`); a period with no frames is `no_tracking_for_period`; a source period outside 1–4 is `invalid_source_period`; a frame in another period but none in the event's period is `period_label_differs_from_nearest_frame`; no frame in tolerance inside the range is `no_frame_within_tolerance` |
| Status | `unmatched`, `matched` (temporal only), and on top of that `coordinate_agreeing` when at least 90 % of at least 11 compared players lie within 0.01 m of the raw tracking positions at the selected frame (`coordinate_disagreeing`, `coordinate_not_assessable` otherwise). `validated` is never assigned |

**Results [OBS, reproducible: `event_alignment.py`, all 64 matches, every event row of each version]:**

- **Temporal coverage (root and May: 144,541 rows each; March: 97,444).** Root and May: 142,990 rows matched (98.93 %); 1,551 unmatched: 1,432 in periods without tracking frames (the extra time of `10510` and `10511`: 738 and 694), 118 outside a period's frame range (including period-boundary events and, in `10506`, `10508`, `10517`, extra-time events beyond the tracking range), 1 with source period 0 (match `3833`). March: 96,351 matched (98.88 %), 1,093 unmatched. 38 matches have at least one unmatched row.
- **Residuals.** Per-match median absolute residual 0.24–0.28 ms; 140,746 of 142,990 matched root rows are within 1 ms, 2,168 within half a frame (16.7 ms), 76 within one frame; the worst per-match 99th percentile is 15.6 ms; the maximum is 33.35 ms (a boundary event just inside the tolerance). **These residuals measure agreement with the timestamp grid; event times are almost always on or next to a frame time. They show that a frame exists near the stated time, not that the stated time is the physical moment of the action.**
- **Coordinate agreement (a separate status, over the temporally matched rows).** March: all 96,351 rows `coordinate_agreeing` (2,120,280 of 2,120,280 players exact). May: 120,854 agreeing and 22,136 disagreeing, the latter all in the ten flagged matches. Root: 120,805 agreeing and 22,185 disagreeing (the same ten matches plus 49 isolated rows in 40 other matches). Ball: exact in 115,052 of 135,929 May rows with a ball in both sources (the rest are the ten flagged matches).
- **Coordinate agreement corroborates the timestamp and the jersey-keyed join of positions** for 54 matches, where the event coordinates are exactly the tracking coordinates at that frame (so event and tracking carry the same position stream at that frame). It does **not** validate which team, player, or action the event describes.

**Provisional conclusion [INF]:** the shared video-time convention is **provisionally supported** by the timestamp structure (`eventTime` coincides with tracking frame times, and `startPeriod1/2` of the metadata coincide with period start frames) and by the coordinate comparison; it has no independent physical ground truth. Align events to tracking by video time under the policy above, never by game clock, frame id, or event ids (the embedded tracking overlay is not a reliable frame index: exploratory result, not retained). The game clock is display metadata. **[OPEN]** treatment of unmatched events in navigation, the missing extra-time tracking of `10510`/`10511`, and whether one frame interval is the right tolerance once ground truth exists.

- Tracking begins at the period-1 kickoff (no pre-kickoff frames); between periods the half-time gap is not represented. Unique frames: strictly increasing `frameNum` and `videoTimeMs`, no gap within a period, in all 64 files, every record (reproducible); `videoTimeMs` equals `frameNum / fps × 1000` to 0.000 ms in every frame.

## 7. Coordinates and the tactical board

**[SPEC]** (tracking §7): origin at the pitch centre, x along the touchlines increasing left to right as seen by the main camera, y along the end lines increasing bottom to top, z vertical up, metres; for a 105×68 pitch the markings are as drawn in the spec. **[OBS]** consistent: metadata gives pitch 105.0×68.0 m for all 64 matches (`stadium.pitches[0]`); player `x` ranges −60.0…60.3 and `y` −41.7…40.9 (players and ball may leave the pitch); event `stadiumMetadata.pitchLength/pitchWidth` agree with metadata in all versions that carry them.

**Attacking direction [OBS]:**

- Regular time (**exploratory**, not retained): the home goalkeeper's mean `x` in period 1 is negative iff `homeTeamStartLeft` is true, and flips in period 2, in **64/64** matches for both periods (event player lists, root version). Kickoff positions agree in 64/64 (first half) and 63/64 (second half; match `3818` has both teams' mean `x` on the same side at the second-half kickoff event, a kickoff snapshot anomaly).
- The root version's `stadiumMetadata.teamAttackingDirection` (`R`/`L`, `nan` when no team has the ball) agrees with the shooter's side in all 1,390 regular-time shots (**exploratory**).
- **Extra time, five matches: a suspected side/identity anomaly in four [OBS, reproducible: `extra_time_identity.py`].** The ball carrier named by the event (`gameEvents.playerId`/`teamId`, root version, possession type other than `IT`) should be near the ball in the position list of his own team. Per period the median own-side carrier-to-ball distance is at most 3.7 m in all 128 regular-time match-periods of all 64 matches (none flagged) and 1.9 m in periods 3 and 4 of `10508`, but **10.5–22.1 m in periods 3 and 4 of `10506` (22.1, 17.7), `10510` (10.5, 11.1), `10511` (20.5, 13.3), and `10517` (16.6, 20.6)**, where the nearest player of the *opposite* list is 1.7–2.4 m from the ball. The tracking files show the same pattern (`10506`: embedded `game_event` team and shirt number are 25.0 and 27.9 m from the ball on their own side and 2.5 and 2.4 m on the opposite side in periods 3 and 4; `10517`: 14.9 and 20.5 m versus 2.5 and 3.6 m; `10508`: 1.9 and 2.1 m versus 17.9 and 22.4 m, i.e. normal). `10510` and `10511` have no extra-time tracking.
- **Provisional flag: `suspected_side_identity_anomaly`** for periods 3 and 4 of `10506`, `10510`, `10511`, `10517`. It is derived from a heuristic (n ≥ 30, median own side > 8 m and median opposite side < 4 m) and is **not a confirmed label swap**: no correction or reassignment of teams, players, or positions is made, and the original side labels are preserved. Supporting exploratory observations (not retained): third-kickoff positions (home list mean x +9.6…+13.2 m although `homeTeamStartLeftExtraTime` is true), goalkeeper positions, and shooter locations are consistent with the event team labels and the metadata flag rather than with the side labels of the position lists. **[INF]** a likely explanation is that the `homePlayers`/`awayPlayers` lists are attached to the opposite team in those periods; this cannot be proven from the files, and video or a PFF statement is needed **[OPEN]**. Until resolved, team and player attribution of positions in those periods is quarantined; ball coordinates are not shown to be affected.
- The roster/ID relations in §8 pass in these periods because ids were assigned by side label; they cannot detect this problem.

**Same reference system [OBS]:** player and ball coordinates are in one system; event coordinates equal tracking raw coordinates wherever they agree (§6); `ballsSmoothed` and `homePlayersSmoothed` are in the same frame as raw (no offsets found).

**Proposed internal convention [INF, provisional]:** keep source coordinates untouched as `x_m, y_m` (origin centre, +x right, +y up, metres, camera view). Store separately, per match and period, `home_attacks_toward` (`+x` or `−x`) with its evidence source. A rendering board draws the raw frame as is (105×68 plus a margin for off-pitch positions); any "both teams attack right" view is a derived, labelled transformation `(x, y) → (−x, −y)` for the team attacking left, never written over raw. No transformation is required to render a 2D board in the raw frame. Axis orientation for screen rendering (screen y grows downward) is a rendering detail to be documented by the board implementation.

## 8. Metadata, rosters, global files, and identifiers

**Metadata** (`Metadata-…/Metadata/<id>.json`): a JSON **list with one object** in all 64 files. Keys (identical in all files): `awayTeam{id,name,shortName}`, `awayTeamKit`, `competition{id,name}`, `date`, `endPeriod1`, `endPeriod2`, `fps`, `halfPeriod`, `homeTeam{…}`, `homeTeamKit`, `homeTeamStartLeft`, `homeTeamStartLeftExtraTime`, `id` (**string**), `period1`, `period2` (minutes), `season`, `stadium{id,name,pitches[{id,length,width,startDate,endDate}]}`, `startPeriod1`, `startPeriod2`, `videoUrl`, `week`. `fps` = 29.97 in all 64; pitch 105×68 in all 64; `homeTeamStartLeft` true in 54, false in 10.

**Rosters** (`Rosters-…/Rosters/<id>.json`): list of `{player{id:str, nickname}, positionGroupType, shirtNumber:str, started:bool, team{id:str, name}}`; 45–52 entries; exactly 11 `started` per team in every match; player ids unique per match; shirt numbers unique per team; the two roster teams equal metadata home/away in all 64.

**Global files:** `competitions.csv` (1 row, columns `games,id,name`; `games` is a Python-literal list of 64 `{id, season}` equal to the metadata IDs). `players.csv`: columns `dob,firstName,height,id,lastName,nickname,positionGroupType`, **2,321 rows for 829 unique ids**: 673 ids repeat in rows that are fully identical (up to 8 copies), so de-duplication on `id` is lossless. Its `positionGroupType` is coarse (`D`, `M`, `F`, `GK`, and a few specific values) and differs from the per-match roster value for most entries (match for 497 of 3,230 roster entries). Every roster player id exists in `players.csv`, and `players.csv` contains no id that never appears in a roster (829 = distinct roster players). `dob` and `height` are personal data; keep them out of shared artifacts.

**Identifier relationships [OBS]:**

| Link | Result |
| --- | --- |
| Tracking player → roster | Tracking has **no `playerId`**, only `jerseyNum` (string) per side. The link is `(side, jerseyNum) → roster (team, shirtNumber)`, where side maps to team through metadata home/away. Shirt numbers are unique per team in every roster, so the link is unambiguous per match. **Not tested directly against tracking positions for all 64 matches**; it was tested through the events (next rows) |
| Event player lists → roster | In all three versions, every `homePlayers[]/awayPlayers[]` entry has `playerId` that exists in the roster, a `jerseyNum` equal to the roster `shirtNumber`, and a team equal to the metadata side: 0 mismatches over 3.18 M (root), 3.18 M (May), 2.14 M (March) player entries in 64 matches. This verifies id/side/shirt consistency only; it cannot detect the extra-time position inversion of §7 |
| Event `gameEvents.playerId`/`teamId` and every `…PlayerId` field in `possessionEvents` → roster | 0 ids missing from the roster; `teamId` always home or away; `homeTeam` flag consistent with `teamId`: 0 mismatches |
| Event `jerseyNum` (int) ↔ tracking `jerseyNum` (string) | Same number after integer conversion; event positions equal tracking positions per jersey (§6), which supports the jersey-based join for the compared events in the unflagged periods (exploratory) |
| Embedded tracking events (`player_id`, `team_id` strings) | Same ids as in event files for the matched events (string vs int types) |
| Types | roster/metadata ids are **strings**; event ids are **ints** (after the March 2025 change); tracking `gameRefId` and event ids are floats |

**[OPEN]** `jerseyNum` reuse after substitutions within a team in a match was not found (roster shirt numbers are unique per team), but tracking jersey identity during frames with 12–15 entries (substitution windows) was not examined individually.

## 9. Data quality findings and spec discrepancies

| # | Finding | Class | Evidence | Recommended rule (not implemented) |
| --- | --- | --- | --- | --- |
| 1 | Raw positions repeat every second frame (≈15 Hz effective) | Expected data behaviour **[OBS, exploratory]**, sampled | 49.95–49.98 % identical pairs, 64 files, first 10,000 records | Do not treat raw as 29.97 Hz; document effective rate before speed/acceleration |
| 2 | Raw implied speed > 12 m/s in 2.06 % of consecutive pairs; steps up to 53 m | Confirmed issue (noise), exploratory | 260.7 M pairs | Prefer smoothed or filtered input for metrics; flag jumps; never differentiate raw positions blindly |
| 3 | 62.70 % of raw player positions are `ESTIMATED` (48.8–74.9 % per match); 68.2 % `LOW` jersey confidence | Expected, important limit | all 64 files, reproducible (260,712,096 raw entries) | Carry `visibility`/`confidence` per sample into every metric; report estimated share |
| 4 | Ball list empty in 32.54 % of unique frames (3,855,434 of 11,849,815); `ballsSmoothed.x/y` null in 17.57 % of present dicts (1,404,715 of 7,994,355) | Confirmed issue | all 64 files, reproducible | Missing ≠ zero; propagate as missing; no interpolation without a recorded rule |
| 5 | Smoothed player lists repeat jerseys with **conflicting** coordinates (6,094 lists in 3,047 frames; 85,524 of 85,900 lists in repeated-frame records); 128 smoothed lists are `null` | Confirmed issue | §4.2b; 61 of 64 files; reproducible | Preserve all candidates; mark the smoothed position ambiguous/unavailable; an explicitly flagged raw fallback only; never select by list order |
| 6 | `jerseyNum` is a string in tracking, int in events/spec | Spec–data discrepancy | types | Normalise at ingestion into one declared type |
| 7 | `version`, `gameRefId`, timestamps only on the first record | Spec–data discrepancy | 64 files | Read from first record, store as match provenance |
| 8 | `frameNum` does not start at 0 (first value 1,214–6,313) | Spec–data discrepancy | 64 files | Use `videoTimeMs`; do not assume a zero base |
| 9 | Repeated-`frameNum` records: 42,950 (3,235 frames); raw players identical in all, smoothed lists equal to the first record's in all | Expected per spec ("frames with multiple associated possession events are duplicated") | reproducible, all 64 files | Collapse by `frameNum` keeping every source record index and the count (provenance); keep the event overlay; flag any difference between repeats |
| 10 | Extra-time tracking missing for `10510`, `10511`; metadata period times null for all five extra-time matches | Confirmed issue | §2, §6 | Mark extra time unavailable per match; do not infer |
| 11 | `speed` exists in 14 files only | Expected variation | file-level, generation date | Do not rely on it; unit unverified |
| 12 | Ball `x` to −80.7 m, `z` below 0 | Potential issue | ranges | Allow out-of-pitch values; validate `z` before any use |
| 13 | `CLK` game-event type; `home_team` int; ids as strings | Spec–data discrepancy | §5 | Treat unknown types as opaque |
| 13b | `suspected_side_identity_anomaly`: carrier far from the ball in its own side's list and ≈2 m on the opposite list in periods 3 and 4 of `10506`, `10510`, `10511`, `10517` | **Confirmed observation, unconfirmed cause** | §7; reproducible, events (64 matches) and tracking (`10506`, `10517`) | Flag per match and period; quarantine team/player attribution of positions there; preserve original labels; never swap automatically |
| 13c | In ten matches (`10502`, `10512`, `3816`, `3817`, `3829`, `3835`, `3841`, `3845`, `3852`, `3859`) the event files of May 1 and June 3 carry player coordinates that differ from the supplied tracking (and from the March 14 version, which equals tracking) by a median 0.5–3.0 m and up to 73 m | **Confirmed discrepancy**; cause [OPEN] (May/root also carry `speed` for exactly these ten, which their tracking lacks: consistent with a newer tracking release behind the event files [INF]) | event comparison document §3.4 (reproducible: 0 of 22,136 matched May events coordinate-agreeing); an exploratory carrier-to-ball comparison favours the May coordinates (median 2.0 m / p90 6.1 m vs 3.1 m / 17.9 m with tracking-raw coordinates) but does not establish which is correct | Treat player attribution in tracking for these ten matches as less certain; validate carrier-to-ball distance; decide the position source per match after PFF clarification |
| 14 | One event row with source `period` 0, `startTime` 0.0 in match `3833` (`G`, game event 6543671) | Confirmed anomaly (1 row, all versions) | event files; `invalid_source_period` in the alignment | Preserve the source value; canonical period null; do not remap; exclude from period logic and log |
| 15 | `periodGameClockTime` is nominal (2700/5400/6300 resets) and differs from event `startGameClock` | Expected / [OPEN] | §6 | Use video time for alignment |

No non-monotonic `frameNum` or `videoTimeMs` exists in any file; no within-period frame gap exists among the 11,849,815 unique frames; no null `x`/`y` exists in any raw player entry (reproducible, all 64 files, every record).

## 10. Open questions and decisions needed

1. **Licensing and redistribution** remain unverified; this audit adds no data to the repository beyond field names, types, and aggregate counts.
2. Checksums/provenance manifest (Milestone 1 open item) not done here.
3. Extra-time handling: event-only extra time in `10510` and `10511`, null metadata period times, `suspected_side_identity_anomaly` in four matches (§7). Decide: exclude extra time, expose it with the flag, or correct with an approved, documented rule after confirmation.
4. Unit and meaning of `speed`; whether the 15 Hz raw cadence is a property of the whole match.
5. Which event version to use: see [event_version_comparison.md](event_version_comparison.md); provisional only.
6. Why event coordinates in ten matches (May and June 3 versions) differ from the tracking files supplied; the March 14 version matches in all 64.
7. Whether the project wants smoothed or raw positions as the default playback source, and whether a smoothed position may be published when conflicting candidates agree within a stated tolerance (§4.2b).
8. Independent ground truth for event timing and event-to-player correspondence (status 3 of §6); none exists in the supplied data.

## 11. Recommendations for Milestone 2 (scope, not started)

Ingestion of tracking, events, metadata, and rosters into the contracts in [data_contracts.md](data_contracts.md), with: one validation report per match covering the findings in §9; the provisional alignment policy of §6 (temporal coverage, coordinate agreement, and validation kept as separate statuses); the coordinate convention of §7; `visibility`/`confidence` carried through; the event version policy applied explicitly. Still out of scope: cleaning, interpolation, metrics, board, prediction.
