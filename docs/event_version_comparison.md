# Event Data version comparison (Milestone 1B)

Status: **implemented, pending independent review.** Revised after the Codex review: the earlier comparison collapsed repeated `(gameEventId, possessionEventId)` pairs; this version uses collision-safe row matching and publishes denominators. The version policy at the end is **provisional**. Companion documents: [dataset_schema_audit.md](dataset_schema_audit.md) and [data_contracts.md](data_contracts.md). Evidence labels: **[SPEC]** PFF document, **[OBS]** observed in the files, **[INF]** inference, **[OPEN]** unresolved.

**Reproduce:** `.\.venv\Scripts\python.exe scripts\audit\event_versions.py --workers 4` (about 1 minute, all 64 matches, 192 files) and `scripts\audit\event_alignment.py` (see [scripts/audit/README.md](../scripts/audit/README.md)). Both read the raw files read-only through `football_fatigue.inventory` and keep per-match results separate from pooled totals.

## 1. Method (the row collections compared)

- **Row collections:** for each of the 64 matches, every row of the root, `March 14, 2025`, and `May 1, 2025` file: **144,541 / 97,444 / 144,541 rows**. No row is dropped, deduplicated, or collapsed at any step.
- **Why collision-safe matching is needed [OBS]:** `(gameEventId, possessionEventId)` is not a unique key. In every version there are **25 repeated key pairs covering 53 rows in 13 matches** (28 surplus rows beyond the first of each pair; 144,513 distinct pairs for 144,541 rows in root and May, 97,416 for 97,444 in March). Each repeated pair is one `OTB` row (possession type `CH` 20, `CR` 2, `SH` 2, `PA` 1) plus one to three `FOUL`/`FO` rows (23 pairs of two rows, one of three, one of four). Three `FOUL` rows share even the full signature below (matches `3838`, `3858`), so a file-order ordinal is needed.
- **Matching strategy:** the signature `(gameEventId, possessionEventId, startTime, gameEventType, possessionEventType)` plus the ordinal of identical signatures in file order. Two rows (one per version) match when signature and ordinal are equal; matching is one-to-one. Unmatched rows of either side and signatures whose multiplicity differs (ambiguous) are counted and reported, never dropped.
- **Field comparison scope:** all leaf paths present in **both** matched rows, except the `homePlayers`, `awayPlayers`, and `ball` position lists, which are compared separately. A null versus a value is a change. Paths present in only one version are **schema differences**, not value changes. Changed **rows** (at least one changed common value) and changed **field values** are counted separately.
- **Coordinate comparison with tracking** uses the provisional alignment policy of the audit (§6): nearest same-period frame within one frame interval; players matched by side and jersey number; agreement tolerance 0.01 m. It tests coordinates only.

## 2. Which version is which

The folder names carry no June label, so root is identified from documents and content.

| Version | Spec beside it [SPEC] | Change-log entries reflected in the content [OBS] | Identification |
| --- | --- | --- | --- |
| `March 14, 2025` | none | 2025-03-14. No `IT` rows, no `sequence`, `homeTeam`, `stadiumMetadata`, `teamAttackingDirection`, `positionGroupType` | consistent with the 2025-03-14 entry **[INF: strong]** |
| `May 1, 2025` | v2.4 (April 2025) in the same folder | 2025-03-14 and 2025-05-01 (`IT` rows, `homeTeam`, `sequence`, `stadiumMetadata`, with `homeTeamStartLeftExtraTime` in the five extra-time matches). No `teamAttackingDirection`, no `positionGroupType` | consistent with the 2025-05-01 entry **[INF: strong]** |
| root (unlabelled) | v2.5 (June 2025) in the same part | May's content plus `stadiumMetadata.teamAttackingDirection` and `positionGroupType` in every player entry, **and different `sequence` values and extra-time coordinates (below)** | consistent with the 2025-06-03 entry and v2.5 **[INF: strong]**; no PFF document states the folder-to-entry mapping |

The v2.4 → v2.5 differences (diffed in full) are exactly `playerId`/`positionGroupType` in §7.1 and `teamAttackingDirection` in §8.

## 3. Results (all 64 matches, denominators stated)

### 3.1 Row counts and keys

| | March 14 | May 1 | Root |
| --- | --- | --- | --- |
| Rows | 97,444 (1,278–2,066 per match) | 144,541 (1,789–3,160) | 144,541 (1,789–3,160) |
| Distinct signatures / distinct key pairs | 97,441 / 97,416 | 144,538 / 144,513 | 144,538 / 144,513 |
| Rows in repeated key pairs | 53 (25 pairs, 13 matches) | 53 | 53 |
| `IT` rows | 0 | 47,097 | 47,097 |
| Game events: `OTB` / `OUT` / `SUB` / `OFF` / `ON` / `END` / `FOUL` / `G` / kickoffs | 89,990 / 6,389 / 587 / 84 / 67 / 138 / 28 / 23 / 138 | 137,087 / same | 137,087 / same |
| Possession events: `PA` 66,715, `CH` 11,555, `CL` 2,956, `RE` 2,544, `CR` 2,412, `SH` 1,559, `TC` 1,290, `BC` 855, `FO` 28; no possession event 7,530 | same | same + `IT` 47,097 | same as May |
| Source `period` values | 1: 47,504 · 2: 47,438 · 3: 1,231 · 4: 1,270 · **0: 1** | 71,381 · 69,554 · 1,844 · 1,761 · **1** | same as May |

### 3.2 Matching and changed values

| Comparison (a vs b) | Matched / rows a / rows b | Unmatched a / b (ambiguous signatures) | Rows with ≥ 1 changed common value | Changed common field values | Schema-only paths (in a, not b) |
| --- | --- | --- | --- | --- | --- |
| May vs March | **97,444** / 144,541 / 97,444 (100 % of March, 67.4 % of May) | 47,097 (all `IT`) / 0 (0) | 1,225 (1.26 % of matched) | 3,932 over 68 paths | 34 (e.g. `gameEvents.homeTeam`, `sequence`, `stadiumMetadata.*`, some optional nested keys) |
| Root vs May | **144,541** / 144,541 / 144,541 (100 %) | 0 / 0 (0) | 9,763 (6.75 %) | 9,763, all `sequence` | 1 (`stadiumMetadata.teamAttackingDirection`) |
| Root vs March | 97,444 / 144,541 / 97,444 | 47,097 (all `IT`) / 0 (0) | 1,225 | 3,932 over 68 paths | 35 |

**Reconciliation with the independent review.** The review's earlier figure of 3,954 changed values was wrong because its row dictionary was keyed on `(gameEventId, possessionEventId)`, which is not unique: three March `FOUL` rows were paired with the wrong May rows, producing false differences (match `3838`: 6; match `3858`: 8 + 8; 22 in total). Removing them gives 3,954 − 22 = **3,932**, the value of the signature-plus-ordinal comparison above, which preserves every duplicate row. The 97,444 matched March rows and the 144,541 rows with 28 surplus rows in repeated pairs also agree with the review. The 3,932 is therefore verified, not an open scope difference.

Largest changed fields May vs March (matched rows): `possessionEvents.touchPlayerName` 701, `grades.touchGrade` 698, `possessionEvents.touchPlayerId` 698, `grades.passerGrade` 180, `gameEvents.playerName` 152, `possessionEvents.targetPlayerName` 130, `possessionEvents.pressurePlayerName` 128, `grades.movementGrade` 104. `startTime`, `endTime`, `eventTime`, `duration`, and period do not change between versions in any matched row.

### 3.3 Position lists in matched rows

| Pair | Player entries compared | Rows with a different player set | Player entries with changed `x`/`y` | Rows with any changed `x`/`y` | Rows with a different ball | Matches affected |
| --- | --- | --- | --- | --- | --- | --- |
| May vs March | 2,144,180 | 152 | 327,629 | 14,895 | 14,036 | 10 (all rows of exactly the ten flagged matches) |
| Root vs May | 3,178,023 | 697 | 31,061 | 1,513 | 1,441 | 47 |
| Root vs March | 2,142,576 | 650 | 349,046 | 15,941 | 15,015 | 50 |

Root vs May: 1,434 of the 1,513 rows with changed coordinates are the extra-time rows of `10510` (739 rows) and `10511` (695 rows), whose coordinates and player sets differ in root; the other 79 rows (1,746 player entries, 45 matches) are small differences at isolated events (up to a few metres).

### 3.4 Event coordinates versus tracking (all 64 matches, every event row, three versions)

Temporal denominator: 142,990 of 144,541 root rows (98.93 %) have a same-period frame within one frame interval (1,551 unmatched: 1,432 in periods with no tracking, 118 outside a period's frame range, 1 invalid source period). Coordinate agreement is assessed on those temporally matched rows.

| Subset | March 14 | May 1 | Root |
| --- | --- | --- | --- |
| 54 unflagged matches | 81,461 of 81,461 events coordinate-agreeing; 1,792,605 of 1,792,605 players exact; ball 76,504 of 76,504 | 120,854 of 120,854 events; 2,659,251 of 2,659,251 players exact; ball 115,027 of 115,027 | 120,805 of 120,854 events agreeing, 49 disagreeing (isolated events, 1–2 per match); 2,658,170 of 2,659,251 players exact |
| 10 flagged matches (`10502`, `10512`, `3816`, `3817`, `3829`, `3835`, `3841`, `3845`, `3852`, `3859`) | 14,890 of 14,890 events agreeing; 327,675 of 327,675 exact; ball 13,950 of 13,950 | **0 of 22,136 events agreeing**; 5 of 486,877 players exact (2,477 within 0.01 m); ball exact in 25 of 20,902 | identical to May |

**Interpretation [INF]:** March event coordinates equal the supplied raw tracking positions in all 64 matches, so March is a **compatibility reference** for the tracking files, not ground truth. In ten matches May and root carry different player and ball coordinates; these ten are also the only matches whose May/root rows carry `speed` while their tracking files do not (and whose tracking was generated in 2023, while the 14 tracking files with `speed` were generated in 2024), which is consistent with a newer tracking release behind the May/root events; this is not proven. An exploratory carrier-to-ball comparison (not retained in `scripts/audit`) found the event carrier closer to the ball with May/root coordinates than with the tracking coordinates in these matches; it does not establish which is physically correct **[OPEN]**.

**Extra time of `10510` and `10511` [OBS]:** tracking has no extra-time frames. In March and May every extra-time row of `10510` has an empty ball list and identical positions in both versions; root carries different positions and ball coordinates in these rows (§3.3). There is nothing to verify any version against. In both matches, and in `10506` and `10517`, the extra-time side labels are suspected anomalous in all versions (audit §7): `suspected_side_identity_anomaly`, an unconfirmed flag, not a swap.

## 4. Verification of the change log against the files

| Change-log claim | Result |
| --- | --- |
| 2025-03-14: `playerId` added; `jerseyNum` integer; identifiers converted to integers; `touches`, `touchesInBox`, `startGameClock`, `gameClock` integer; `'nan'` → `null` for `trickType` | present in all three versions as stated (integers, nullable; no `"nan"` `trickType`). A change cannot be confirmed because no earlier version was supplied |
| 2025-03-14: event XY are raw broadcast tracking, not smoothed | **confirmed for March in all 64 matches** (§3.4); not true for May and root in ten matches, an undocumented difference |
| 2025-05-01: `IT` events; `homeTeam`; `sequence`; `stadiumMetadata` with pitch size, `homeTeamStartLeft`, `homeTeamStartLeftExtraTime` (extra-time games only) | confirmed (47,097 `IT` rows, absent in March; the extra-time key exists in the five extra-time matches only) |
| 2025-06-03: `teamAttackingDirection` (`R`/`L`/string `"nan"`) in `stadiumMetadata`; `positionGroupType` | confirmed in root only; **not documented in the change log:** the 9,763 changed `sequence` rows and the changed extra-time coordinates of `10510`/`10511` |
| Spec v2.4/v2.5 player fields | confirmed; `speed` (not in the spec) appears in 14 matches in all versions and in 24 matches in May and root |

## 5. Provisional version policy

**Provisional recommendation, with the uncertainty stated:**

1. **Event content: root (June 3, 2025) as the primary source.** It is the newest by the change log, carries the fields of the newest specification, and has the same rows as May (144,541 of 144,541 matched). **It is not an unchanged superset of March or of May:** relative to May it changes 9,763 `sequence` values and the extra-time rows of two matches; relative to March it adds 47,097 rows and many columns **and** changes 3,932 field values in 1,225 matched rows. Whether root's changes are corrections cannot be decided from the files.
2. **Positions for the tactical board and analytics: the supplied tracking files.** Event-embedded coordinates are a cross-check only.
3. **March 14 event coordinates are a compatibility reference** (they equal tracking everywhere), **not ground truth.**
4. **All three versions are retained** untouched and independently loadable; every derived output records the version used.
5. **Flag, do not hide:** the ten coordinate-disagreement matches; event-only extra time in `10510` and `10511`; suspected side-identity anomalies in extra time of `10506`, `10510`, `10511`, `10517`; the repeated key pairs; the period-0 row of `3833`.
6. These are provisional policies, not irreversible decisions. They should be confirmed at the independent review and, ideally, with PFF's data owner.

**Uncertainties [OPEN]:** root's folder identification rests on content and documents; the undocumented `sequence` and coordinate changes; the cause of the ten-match disagreement; no ground truth for any version; licensing and delivery provenance unverified.
