# Proposed data contracts (Milestone 1B)

Status: **proposal for the next implementation milestone, derived from the audit, not finalized and not implemented.** Nothing here is code. Every choice that rests on an unresolved question is marked **PROVISIONAL**. Evidence is in [dataset_schema_audit.md](dataset_schema_audit.md) (*audit §n*) and [event_version_comparison.md](event_version_comparison.md) (*versions §n*); the numbers are reproducible with [scripts/audit](../scripts/audit/README.md). Intended consumers: tactical playback and tracking analytics.

All examples are **synthetic** illustrations of shape, not source data.

## 1. Cross-cutting conventions and invariants

| # | Invariant | Basis |
| --- | --- | --- |
| I1 | **Source values are preserved verbatim next to any derived value.** Source `period` (including the observed value 0), source side labels (`homePlayers`/`awayPlayers`), source ids, and source versions are never overwritten. Derived values live in separate, named fields | PROJECT_RULES |
| I2 | **Four distinct data states** are never conflated: `available` (present, no known problem), `missing` (absent or null in the source), `ambiguous` (several conflicting source candidates; none selected), `quarantined` (present but its attribution to a team/player/period is uncertain). A value in state `ambiguous` or `quarantined` is exposed as candidates plus a flag, not as a single value | audit §4.2, §7 |
| I3 | **Four validation levels** are never conflated: `source_present` < `temporally_aligned` (a same-period frame lies within the alignment tolerance) < `coordinate_agreeing` (event coordinates match the raw tracking positions at that frame) < `validated` (right player, team, period, and physical action, confirmed by independent ground truth such as video). Ingestion may assign at most `coordinate_agreeing`; `validated` is never assigned automatically | audit §6 |
| I4 | **No silent selection, swap, interpolation, or fill.** Conflicting duplicates, suspected side/identity anomalies, and missing frames are flagged and carried; any fallback is an explicit, flagged field naming its source | PROJECT_RULES |
| I5 | **Provenance survives deduplication.** A collapsed frame keeps the record indices of every source record it came from and how many there were; an event keeps `source_version` and its source row index | audit §4 |
| I6 | Identifiers: match, team, player ids are **integers** in contracts; the original source text and type stay in provenance | audit §8 |
| I7 | Primary time key: **`video_time_s`** (float seconds of video time) = tracking `videoTimeMs / 1000` = event `eventTime`; `frame_num` is secondary. Frame interval is `1 / fps` with `fps` from metadata (29.97 in all 64 matches). The shared video-time convention is **provisionally supported** by timestamp structure and coordinate comparisons; it is not independently validated physical ground truth | audit §6 |
| I8 | Coordinates: source meters, origin at the pitch centre, +x right, +y up (main-camera view), `z` up; stored untouched as `x_m`, `y_m`, `z_m`. Off-pitch values are legal. A "both teams attack +x" view is a derived, labelled transform | audit §7 |
| I9 | Version policy is an explicit run parameter and is recorded: **PROVISIONAL** default root (June 3, 2025) for event content, supplied tracking files for positions, March 14 event coordinates as a compatibility reference (not ground truth), all three versions retained | versions §7 |
| I10 | `null` means missing, never 0. Every record carries `quality_flags` (a possibly empty set, §9) | audit §9 |

## 2. Match

| Field | Type | Null | Source / note |
| --- | --- | --- | --- |
| `match_id` | int | no | metadata `id` (string in source); equals file names, `gameRefId`, event `gameId` |
| `kickoff_date` | datetime (as given) | no | metadata `date`; time zone not stated, **[OPEN]** |
| `competition_id`, `competition_name`, `season` | int, str, str | no | metadata |
| `matchweek` | int | no | metadata `week`; no stage field exists, a stage label is a derived PROVISIONAL mapping |
| `home_team_id`, `away_team_id` | int | no | metadata |
| `stadium_id`, `stadium_name`, `pitch_length_m`, `pitch_width_m` | int, str, float, float | no | metadata `stadium` (one pitch entry in all 64; 105 × 68) |
| `fps` | float | no | metadata (29.97) |
| `periods_in_events`, `periods_in_tracking` | sets of source period values | no | derived; e.g. `10510` has {1,2,3,4} in events and {1,2} in tracking |
| `source_refs` | list | no | relative paths of metadata, roster, tracking, and each event version used |

## 3. Team, player, roster

| Contract | Fields (type, nullable) |
| --- | --- |
| **Team** | `team_id` int, `name`, `short_name` str, `kit` (optional dict) |
| **Player** (global, `players.csv` de-duplicated on `id`; 2,321 rows, 829 ids, duplicates identical) | `player_id` int, `first_name`, `last_name`, `nickname`, `date_of_birth`, `height_cm`. Personal data: keep out of shared artifacts |
| **RosterEntry** (per match) | `match_id`, `team_id`, `roster_side` ∈ {`home`,`away`} (from metadata home/away), `player_id`, `shirt_number` int, `position_group` str, `started` bool |

`(match_id, team_id, shirt_number)` is unique per roster (verified for all 64) and is the join from a tracking jersey to a player **only when the side label is trusted** (§4).

## 4. Period (one row per `(match_id, source_period)`)

| Field | Type | Null | Note |
| --- | --- | --- | --- |
| `source_period` | int | no | verbatim, including values outside 1–4 (the observed period 0 of one event of match `3833`) |
| `canonical_period` | int 1–4 | yes | null when `source_period` is not 1–4; never silently remapped |
| `tracking_status` | enum | no | `available` or `missing_period` (periods 3 and 4 of `10510` and `10511` exist in events only: "event-only extra time") |
| `start_video_time_s`, `end_video_time_s` | float | yes | from tracking first/last frame; null when `missing_period`. Metadata `startPeriod*/endPeriod*` agree within one frame where present and are null in the five extra-time matches |
| `home_attacks_toward` | enum `+x`/`-x`/`unknown` with `evidence` | no | periods 1–2 from `homeTeamStartLeft`; periods 3–4 PROVISIONAL `unknown` unless separately supported |
| `side_identity_status` | enum | no | `no_anomaly_detected`, **`suspected_side_identity_anomaly`**, `insufficient_events`, or `not_assessed`. Set from the carrier-to-ball test of audit §7 (periods 3 and 4 of `10506`, `10510`, `10511`, `10517` are `suspected_side_identity_anomaly`). It is a flag with evidence, **not a decision that labels are swapped**; the cause is unconfirmed |
| `side_identity_evidence` | dict | yes | test name, n, medians, script and version |

## 5. Tracking frame (one row per unique `frame_num`)

| Field | Type | Null | Source / note |
| --- | --- | --- | --- |
| `match_id`, `frame_num` | int | no | `frameNum` (does not start at 0) |
| `video_time_s` | float | no | `videoTimeMs / 1000` |
| `source_period` | int | no | verbatim |
| `period_elapsed_s` | float | no | |
| `game_clock_nominal_s` | float | no | display only (resets to 2700 / 5400 / 6300) |
| `source_record_indices` | list of int | no | zero-based line numbers of **every** source record with this `frameNum`; `source_record_count = len(...)`. In the audit, repeated records carry identical raw players in all 42,950 cases, but identical smoothed lists are not guaranteed by the contract; any difference is flagged `REPEATED_RECORDS_DIFFER` |
| `ball` | BallSample | yes | null (state `missing`) when `balls` is empty |
| `quality_flags` | set | no | |

## 6. Player position (one row per `(match_id, frame_num, source_side, jersey_number)`)

| Field | Type | Null | Source / note |
| --- | --- | --- | --- |
| `source_side` | enum `home`/`away` | no | the list the entry came from, preserved; never rewritten |
| `jersey_number` | int | no | tracking `jerseyNum` (string in source) |
| `team_id`, `player_id` | int | yes | resolved through the roster `(roster_side, shirt_number)` **only if** the period's `side_identity_status` is not `suspected_side_identity_anomaly`; otherwise null with `attribution_status = quarantined`. Never swapped automatically |
| `attribution_status` | enum | no | `resolved`, `quarantined`, `unresolved_jersey` |
| `x_raw_m`, `y_raw_m` | float | no | raw `x`, `y` (never null in the source); update about every second frame (`raw_repeated_from_previous_frame` is derived) |
| `smoothed_status` | enum | no | `unique` (exactly one smoothed entry for the jersey), `duplicate_identical` (several entries with identical `x`,`y`), `ambiguous_conflicting` (several entries with different `x` or `y`), `missing` (smoothed list null or jersey absent) |
| `smoothed_candidates` | list of {`x_m`, `y_m`, `visibility`, `confidence`} | no | **all** source candidates; order carries no meaning and is not used to select |
| `x_smooth_m`, `y_smooth_m` | float | yes | populated only when `smoothed_status` is `unique` or `duplicate_identical`; **null when `ambiguous_conflicting`** |
| `position_fallback` | {`source`, `x_m`, `y_m`, `flag`} | yes | for ambiguous or missing smoothed positions a consumer may use the raw position of the same jersey **as an explicitly flagged fallback** (`SMOOTHED_AMBIGUOUS_RAW_FALLBACK`); raw was present in all 67,017 conflicting groups of the audit. No other fallback (no averaging of candidates, no "first entry") exists |
| `visibility`, `jersey_confidence` | enums | no | from the raw entry |
| `speed_raw` | float | yes | only in 14 files; unit unverified, not used as a metric input (PROVISIONAL) |
| `quality_flags` | set | no | |

Candidate spread (audit §4.2): the conflicting candidates differ by at most 0.05 m in about 46 % of groups, 0.10 m in 82 %, and 0.5 m in all but 8 groups; the small spread does not make a candidate the correct one, so no candidate is promoted by default. Whether a smoothed position within a stated tolerance of the candidates may be published is a decision for the review (**open decision 1**).

## 7. Ball sample

| Field | Type | Null | Note |
| --- | --- | --- | --- |
| `x_raw_m`, `y_raw_m`, `z_raw_m`, `visibility_raw` | float, float, float, enum | yes | `balls[0]`; null when the list is empty. May lie far outside the pitch; `z` can be negative (**[OPEN]**) |
| `x_smooth_m`, `y_smooth_m`, `z_smooth_m`, `visibility_smooth` | float, float, float, enum | yes | `ballsSmoothed`; `x`,`y` are null in 17.6 % of present dicts |
| `ball_status` | enum | no | `available`, `missing_raw`, `missing_smooth_xy`, `missing` |

## 8. Event row (one row per source row)

| Field | Type | Null | Source / note |
| --- | --- | --- | --- |
| `event_row_id` | str | no | derived, stable: `match_id` + `source_version` + zero-based source row index |
| `match_id` | int | no | `gameId` |
| `source_version` | enum `root`/`march_14_2025`/`may_1_2025` | no | provenance of this row |
| `game_event_id`, `possession_event_id` | int, int | no, yes | the pair is **not unique** (25 repeated pairs covering 53 rows); do not use it as a key |
| `game_event_type`, `possession_event_type` | str, str | no, yes | unknown values preserved |
| `source_period` | int | no | verbatim (one row has 0) |
| `canonical_period` | int 1–4 | yes | null when `source_period` is not 1–4 |
| `event_time_s` | float | no | `eventTime` |
| `start_time_s`, `end_time_s`, `duration_s` | float | yes | game-event interval |
| `source_team_id`, `source_player_id` | int | yes | as in the source; **not corrected** in periods flagged `suspected_side_identity_anomaly` |
| `home_team`, `sequence`, `attack_direction` | bool, number, enum | yes | May/root only (`attack_direction`: root only; the string `"nan"` becomes null). `sequence` differs between root and May in 9,763 rows |
| `positions_snapshot` | embedded player/ball list | yes | **not a position source**; kept for comparison with `coordinate_status` |
| `attributes` | dict | yes | remaining source fields verbatim; nested key sets vary by match and version |
| `quality_flags`, `source_ref` | set, ref | no | |

## 9. Timeline entry (derived) and quality flags

**TimelineEntry** is derived and never replaces a source event. Alignment follows one provisional policy (audit §6):

| Policy element | Definition |
| --- | --- |
| Units | event `event_time_s` and frame `video_time_s`, both seconds of video time |
| Candidate frames | frames of the **same source period** as the event only |
| Selection | nearest frame by `|event_time_s − video_time_s|` |
| Maximum residual | **one frame interval, `1 / fps`** (33.37 ms at 29.97 fps); a single tolerance, no second threshold |
| Tie-breaking | exact tie (difference below 1 ns): the earlier frame |
| Missing frames / boundaries | no extrapolation: an event outside the period's first-to-last frame range (± the tolerance) is unmatched; a period with no tracking frames gives `no_tracking_for_period`; periods outside 1–4 give `invalid_source_period`; no frame within tolerance gives `no_frame_within_tolerance` |
| Provenance | `frame_num` of the selected frame, its `source_record_indices`, signed `residual_s`, `tolerance_s`, `policy_version` |
| Status | `unmatched`; `matched_temporal_only` (temporal coverage only); `matched_coordinate_agreeing` (also ≥ 90 % of ≥ 11 compared players within 0.01 m of the raw tracking positions at the selected frame). `validated` is not assigned (I3) |

| Field | Type | Null |
| --- | --- | --- |
| `event_row_id`, `match_id` | str, int | no |
| `alignment_status` | enum above | no |
| `frame_num`, `residual_s` | int, float | yes (null when unmatched) |
| `reason` | enum | yes (set when unmatched) |
| `coordinate_status` | `coordinate_agreeing` / `coordinate_disagreeing` / `coordinate_not_assessable` | yes |
| `validated_correspondence` | constant `not_assessed` | no |

**Quality flags (initial, extendable):** `REPEATED_RECORDS_PRESENT`, `REPEATED_RECORDS_DIFFER`, `RAW_POSITION_REPEATED`, `PLAYER_ESTIMATED`, `JERSEY_CONFIDENCE_LOW`, `SMOOTHED_LIST_MISSING`, `SMOOTHED_DUPLICATE_IDENTICAL`, `SMOOTHED_AMBIGUOUS_CONFLICTING`, `SMOOTHED_AMBIGUOUS_RAW_FALLBACK`, `BALL_MISSING`, `BALL_SMOOTHED_XY_NULL`, `OFF_PITCH`, `SUSPECTED_SIDE_IDENTITY_ANOMALY`, `NO_TRACKING_FOR_PERIOD`, `EVENT_UNMATCHED_<reason>`, `EVENT_COORDS_DISAGREE_TRACKING`, `INVALID_SOURCE_PERIOD`, `EVENT_KEY_PAIR_REPEATED`.

## 10. Provenance record (per derived output)

`match_id`, `processing_version`, `event_version_used`, list of `source_ref` (relative path, size, **sha256 to be computed in the next milestone; not computed here**), `tool_versions`, `parameters` (alignment tolerance, coordinate thresholds), `policy_version`, `validation_findings`, `exclusions`, `created_at`. Derived data goes under `data/processed/<match-id>/<processing-version>/`, never over earlier outputs.

## 11. Consistency with the audit (reviewed after the Codex review)

| Contract statement | Status |
| --- | --- |
| Raw `x,y` never null; frame steps 0 or +1; `videoTimeMs = frameNum/fps*1000` exactly | verified by `tracking_scan.py` for all 64 files, every record |
| Repeated `frameNum` records carry identical raw players (42,950 of 42,950) | verified by `tracking_scan.py`, 64 files |
| Smoothed duplicates: conflicting (not identical) in all but 17 duplicate groups; confined to frames with repeated records | verified by `tracking_scan.py`, 64 files (see audit §4.2) |
| `(match, team, shirt)` unique per roster | all 64 rosters (exploratory check, not in `scripts/audit`) |
| Event ↔ tracking temporal coverage and coordinate agreement per version | `event_alignment.py`, 64 matches, every event row |
| Event version differences by matched row | `event_versions.py`, 64 matches |
| Side/identity anomaly flag | `extra_time_identity.py`, 64 matches (events), extra-time matches (tracking) |
| Event ↔ roster id consistency (3 versions) | exploratory check, not retained in `scripts/audit`; it cannot detect side-identity anomalies |

## 12. Open contract decisions (PROVISIONAL until resolved)

1. Whether a smoothed position may be published when conflicting candidates agree within a stated tolerance; until decided the contract publishes none and offers the flagged raw fallback.
2. Default playback source (smoothed with flagged fallback vs raw).
3. Extra time: exclude, expose with the `suspected_side_identity_anomaly` flag, or correct after confirmation; the contract never swaps silently. Extra time of `10510` and `10511` is event-only.
4. Event version for event content (root proposed) and coordinates (tracking proposed); `sequence` source (root or May).
5. Unit and use of `speed`; whether the raw update rate holds for whole matches.
6. Treatment of unmatched events in event navigation.
7. Stage labels and kickoff time zone.
8. Whether the alignment tolerance of one frame interval is appropriate once any independent ground truth exists; it is a temporal-coverage rule, not an accuracy claim.
