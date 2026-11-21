# Original FIFA World Cup 2022 System — Complete Technical Walkthrough

**Inspection date:** 2026-10-20, Africa/Cairo.  
**Repository:** F:\Football-Fatigue-Project.  
**Inspected HEAD:** de0687e; branch feature/wadi-degla-demo.  
**Original board baseline:** d89d0e6c9f73ae08b2531c9c34c87aa186c18007.  
**Purpose:** source-based technical explanation for a graduation-project supervisor discussion.  
**Scope:** original FIFA data engineering, synchronization, MatchData, original movement analytics, HTTP adapter, tactical board, and playback. Later video-CV implementation is excluded.

## Reading guide and evidence rules

This is a static reverse-engineering report, not a runtime acceptance certificate. Source files, tests, configuration, documentation, Git history, a bounded raw-data sample, and selected existing audit artifacts were inspected. **No tests, servers, browser application, builds, inference, or processing pipelines were executed.** Python was used only for read-only text/AST inspection, reading existing JSON/document XML, and streaming three raw tracking records. The only project write is this report.

Evidence vocabulary used throughout:

| Label | Meaning |
| --- | --- |
| **SOURCE VERIFIED** | Implemented behavior established by reading current source; not a claim that it executed successfully in this audit. |
| **DOCUMENTED** | Existing project documentation or stored audit output states it; expensive underlying computation was not repeated. |
| **DIRECT SAMPLE** | Observed directly in the bounded raw-file inspection described below. |
| **INFERRED** | Explanation or consequence derived from source, not a documented historical motive or physical validation. |
| **PLANNED** | Intent exists but implementation is absent from the inspected original path. |
| **NOT VERIFIED** | Evidence or execution is insufficient. |
| **NOT FOUND IN CURRENT REPOSITORY** | No implementation was found within the inspected original-system files. |

References use repository-relative path:Lstart-Lend. They refer to current working-tree line numbers unless explicitly prefixed with a Git revision. Appendix A inventories all 23 relevant source/script files; Appendix B records exact Python signatures, model fields, returns, callers/callees and selected control-flow evidence; Appendix C indexes 192 test-function definitions in 11 test files. Test-definition counts are not parametrized pytest item counts or passing-test counts.

The initial working tree contained one unrelated untracked file, wadi-degla-demo-assets.zip. It was neither opened nor changed. No repository branch, index, commit, remote or Git configuration was modified.

### Table of contents

1. Executive technical overview
2. Repository architecture and file map
3. FIFA dataset architecture
4. Milestone 1: investigation and audit
5. Milestone 2A: ingestion and parsing
6. Milestone 2B: synchronization
7. Milestone 2C: preparation and integration
8. MatchData and public APIs
9. Tactical board implementation
10. Playback and temporal architecture
11. Frontend/backend integration
12. Algorithms, structures and mathematics
13. Tests, validation and reliability
14. End-to-end worked example
15. Ahmed and Ziad: responsibilities and evidence
16. Decisions, problems and fixes
17. Supervisor discussion: 40 questions and answers
18. Final architecture, status and evidence index

# Chapter 1 — Executive Technical Overview

## 1.1 What the original system does

**SOURCE VERIFIED:** it reconstructs a football match from supplied tracking coordinates and event annotations, prepares quality-aware frame objects, aligns event rows to tracking frames by video time, and displays a selectable match segment on an HTML Canvas tactical board. It does not need to detect players in video. Tracking coordinates are inputs, not outputs of a detector in this workstream.

The implemented path is:

1. Discover original files using directory context and numeric match filenames.
2. Read metadata, roster, and an explicitly selected event version.
3. Stream compressed JSONL tracking records.
4. Validate required structure and preserve uncertainty, duplicates and provenance.
5. Join shirt numbers to roster identities through source side and metadata team IDs.
6. Align event rows to same-period frames within one metadata-derived frame interval.
7. Prepare raw, smoothed and selected coordinates under an explicit policy.
8. Convert prepared frames to JSON-friendly render states.
9. Load the selected segment into backend memory, calculate original movement summaries, and expose HTTP responses.
10. Fetch arrays into browser memory, maintain browser playback state, transform meters to pixels, and draw players/ball and event details.

Evidence: src/football_fatigue/ingestion/loaders.py:L110-L200; src/football_fatigue/synchronization.py:L228-L351; src/football_fatigue/match_data.py:L250-L332; src/football_fatigue/playback/server.py:L106-L275; src/football_fatigue/board/index.html:L1530-L1701.

## 1.2 Why there are separate layers

A tracking record answers “what positions were supplied at this video time?” An event answers “what action was annotated at this time?” A roster answers “which player belongs to this team and shirt number?” Rendering answers “where should a marker appear on this canvas?” These questions have different failure modes. Keeping them separate prevents a renderer from inventing missing coordinates, a shirt number from becoming a globally unique player identifier, or temporal proximity from becoming a claim of physical correctness.

**INFERRED design benefit**, grounded in the separation actually implemented: a future renderer can consume PreparedFrame or FrameState without implementing bz2 parsing, version discovery or time matching. A future ingestion improvement need not rewrite Canvas drawing. Shared engineering rules demand these boundaries: PROJECT_RULES.md, engineering section; exact rule anchors appear in the evidence index.

## 1.3 End-to-end architecture

~~~mermaid
flowchart TD
    RAW["Original FIFA raw files: tracking .jsonl.bz2, metadata, rosters, events"] --> INV["inventory.scan: filenames, directory context, sizes"]
    INV --> DS["Dataset: require exactly one requested source"]
    DS --> META["MatchMetadata + Roster"]
    DS --> EV["EventRow list: explicit EventVersion"]
    DS --> TRACK["Streaming TrackingFrame iterator"]
    META --> TRACK
    TRACK --> SYNC["align_events: same period, nearest video time, tolerance"]
    EV --> SYNC
    TRACK --> PREP["prepare_frame: coordinate policy, states, identity flags"]
    SYNC --> ATT["Source-record-provenance event attachment"]
    PREP --> ATT
    ATT --> MD["MatchData Python interface"]
    MD --> AD["frame_to_state: render-safe dictionaries"]
    MD --> MOV["Original derived movement summaries"]
    AD --> LOAD["load_match_data: selected segment in memory"]
    MOV --> LOAD
    LOAD --> HTTP["BoardHandler: /api/match, /api/frames, /api/events, /api/analytics"]
    HTTP --> JS["Browser arrays + PB state"]
    JS --> CANVAS["Canvas pitch, player/ball markers, event panel"]
    MD -. separate optional consumer .-> CTRL["Python PlaybackController: not used by browser server path"]
~~~

This diagram describes the original FIFA path. In the current shared server, later spatial/CV additions exist alongside it; those are not original components of this diagram.

## 1.4 Implemented versus incomplete

Implemented: metadata-only inventory, reproducible audit tooling, typed production ingestion, source quality flags, temporal synchronization, MatchData preparation and summaries, adapter, Python navigation controller, stdlib HTTP service, browser match setup, player/ball drawing, frame-index playback, event panel, and original movement summaries.

Incomplete or unverified: physical event correspondence, authority/licensing confirmation, complete source-hash manifest, automatic identity-anomaly detection/quarantine in the board service, production-grade error/concurrency contracts, and deployment/wheel asset validation. Temporal-gap playback is only visually flagged, not faithfully scheduled. Fatigue models and physiological validation are **NOT FOUND IN CURRENT REPOSITORY** for this original path.

Evidence: docs/data_contracts.md:L147-L173; docs/team_handoff.md:L157-L165; source-specific limitations in Chapters 10–16. Documentation status strings still say several milestones are pending review; a commit proves code existence, not formal sign-off.

# Chapter 2 — Repository Architecture and Complete File Map

## 2.1 Structural map

~~~text
Football-Fatigue-Project/
  PROJECT_RULES.md, AGENTS.md, CLAUDE.md
  README.md, pyproject.toml, .gitignore, .gitattributes
  .github/workflows/checks.yml
  .github/pull_request_template.md
  data/raw/                         original sources, ignored
  data/processed/audit_1b/           existing audit outputs, ignored
  src/football_fatigue/
    __init__.py
    inventory.py
    ingestion/
      __init__.py, __main__.py, models.py, loaders.py, validation.py
    synchronization.py
    match_data.py
    playback/
      __init__.py, __main__.py, adapter.py, controller.py, server.py
    board/
      __init__.py, index.html        original board; later additions isolated
    analytics/
      __init__.py, movement.py       original movement functionality
  scripts/audit/
    README.md, audit_lib.py, event_versions.py, tracking_scan.py,
    event_alignment.py, extra_time_identity.py
  tests/
    test_package_installation.py, test_inventory.py, test_audit_lib.py,
    test_ingestion.py, test_synchronization.py, test_match_data.py,
    test_match_data_quality.py, test_team_handoff_docs.py,
    test_playback_adapter.py, test_playback_navigation.py, test_analytics.py
  docs/
    architecture.md, data_policy.md, dataset_schema_audit.md,
    event_version_comparison.md, data_contracts.md, team_handoff.md
~~~

Existing configs/, references/, notebooks/, models/ are organizational directories, not evidence that corresponding runtime components exist. No original separate JavaScript bundle, CSS bundle, Node package manifest, frontend build system, FastAPI application, database schema or WebSocket implementation was found. The original styles and JavaScript live inside board/index.html.

## 2.2 Main responsibility table

| File/group | Layer | Principal inputs → outputs | Direct integration |
| --- | --- | --- | --- |
| inventory.py | Discovery | root path → Inventory, FileRecord, Issue | Dataset and audit_lib call scan; never parses content |
| ingestion/models.py | Domain contracts | typed constructor fields → immutable records/enums | loaders, validation, synchronization, preparation |
| ingestion/loaders.py | Ingestion | explicit match/version + sources → metadata/roster/event rows/frame iterator | MatchData and synchronize_match |
| ingestion/validation.py | Quality/output boundary | canonical frames/events → counters/report; explicit report writer | ingestion CLI and MatchData accumulator |
| ingestion/__main__.py | CLI | quality command arguments → console/report + exit code | Dataset, quality_report, optional writer |
| synchronization.py | Synchronization | EventRows + chronological frames + FPS → MatchSynchronization | MatchData.synchronize; audit policy is conceptual predecessor |
| match_data.py | Consumer Python API | Dataset + match/version/policy → PreparedFrames and QualitySummary | adapter, server, movement and controller |
| playback/adapter.py | Serialization adapter | PreparedFrame + roster/event lookups → FrameState dictionary; alignments → event index | server; controller |
| playback/controller.py | Optional Python playback/navigation | materialized frames/events → PlaybackState | tests and Python consumers; not called by BoardHandler |
| playback/server.py | HTTP/service | root + selection → stored payload and JSON | browser fetch; MatchData and adapter |
| board/index.html | Browser visualization | metadata/frames/events/analytics → Canvas and DOM | fetches original endpoints; owns PB state |
| analytics/movement.py | Original derived analytics | usable prepared positions → trajectories, kinematics, summaries | server and analytics tests |
| audit scripts | Investigation only | source files → per-match diagnostic JSON | audit_lib runner; no production importing of audit logic |
| tests | Verification specifications | synthetic fixtures → assertions | inspected, not run here |

Exact definitions, signatures, imports, callers and line ranges are indexed in Appendices A–C rather than guessed from names.

## 2.3 Dependency and communication map

~~~mermaid
flowchart LR
    I[inventory] --> L[ingestion.loaders]
    M[ingestion.models] --> L
    M --> V[ingestion.validation]
    L --> V
    L --> S[synchronization]
    S --> D[match_data]
    V --> D
    L --> D
    D --> A[playback.adapter]
    D --> C[playback.controller]
    A --> C
    D --> R[analytics.movement]
    D --> H[playback.server]
    A --> H
    R --> H
    H --> B[board/index.html via HTTP JSON]
~~~

Arrows mean “supplies/used by,” not Python import direction. There is no call from the browser to PlaybackController. Python imports its class into playback/__init__.py for consumers; server.py performs its own load orchestration and the JavaScript implements its own state/navigation logic. Evidence: playback/__init__.py:L10-L31; server.py:L40-L48; board/index.html:L898-L904,L1343-L1484.

## 2.4 Shared-file boundary and version history

Read-only Git evidence:

| Commit | Recorded author | Scope established by changed files |
| --- | --- | --- |
| 582c703 | bahr74 | Foundation |
| 891c0f4 | bahr74 | Dataset inventory |
| 3708c84 | bahr74 | Dataset schema audit |
| dbc1736 | bahr74 | Streaming ingestion and validation |
| a405e8b | bahr74 | Event synchronization |
| ed11919 | bahr74 | MatchData integration and handoff |
| b134dc2 | Ahmed Bahr | Merge of the data branch |
| d89d0e6 | zeyadkkhaled | First committed board, playback adapter/controller/server, movement module and tests |
| 665577a, de0687e | later contributors | CV-era commits; not analyzed as original-system implementation |

A comparison of d89d0e6 to HEAD shows the original inventory, ingestion, synchronization, MatchData, adapter, controller and movement files unchanged. Shared server.py and board/index.html changed. Originally "/" served the board and run used HTTPServer; currently "/" serves a new landing page, "/tactical" serves the original board, and the service uses ThreadingHTTPServer. Current spatial endpoints, report download and board overlay additions are outside the original baseline and are mentioned only to prevent historical misattribution. Their algorithms and CV routes are deliberately excluded.

Evidence: git show --stat d89d0e6; git diff --name-status d89d0e6 HEAD; git diff d89d0e6 HEAD -- src/football_fatigue/board/index.html; current server.py:L305-L366,L868-L918. No branches were switched.

# Chapter 3 — FIFA Dataset Architecture

## 3.1 Origin and physical organization

The project documents a Google Drive dataset source at https://drive.google.com/drive/folders/1_a_q1e9CXeEPJ3GdCv_3-rNO3gPqacfa. The link was not fetched during this task; local contents and project evidence were used. Sharing access does not establish redistribution rights. Licensing remains **NOT VERIFIED**.

**DIRECT SAMPLE:** data/raw currently contains the six timestamped download-part directories and three global files shown below. The server's default nested root does not match this inspected layout; use a correctly configured root rather than moving the dataset.

~~~text
data/raw/
  Tracking Data-20261015T150539Z-1-001/Tracking Data/
  Tracking Data-20261015T150539Z-1-002/Tracking Data/
  Event Data-20261015T150543Z-1-001/Event Data/
  Event Data-20261015T150543Z-1-002/Event Data/
  Metadata-20261015T150542Z-1-001/Metadata/
  Rosters-20261015T150541Z-1-001/Rosters/
  competitions.csv
  players.csv
  PFF FC Change Log.docx
~~~

Tracking filename: numeric ID + .jsonl.bz2. Metadata, roster and events: numeric ID + .json. Event files have root, March 14, 2025 and May 1, 2025 locations. Download-part suffixes are preserved; ingestion discovers sources instead of renaming them.

Evidence: docs/data_policy.md:L9-L31; inventory.py:L44-L59,L155-L197; server.py:L284-L284,L921-L949.

## 3.2 Coverage and what was not recomputed

**DOCUMENTED, historical all-match audit:** 64 matches, IDs 3812–3859 and 10502–10517, each with tracking, metadata, rosters and three event versions. The original tracking population is reported as 11,892,765 records collapsing to 11,849,815 frames. That full scan was not repeated here.

**DIRECT SAMPLE:** match 3812 root events contain 2,010 rows; roster contains 52 entries. Metadata identifies Senegal as home (372), Netherlands as away (366), FPS 29.97, pitch 105×68 m. Existing stored match-3812 tracking audit reports 185,746 records, 184,857 frames and 889 repeated records; those are stored results, not this task's measured scan totals.

Four audit output directories currently each contain 64 JSON files: event_versions, tracking_scan, event_alignment and extra_time_identity. Presence alone does not prove every stored result is current or successful; selected 3812 outputs were read and report ok=true.

Evidence: docs/dataset_schema_audit.md:L9-L30; data/processed/audit_1b/tracking_scan/3812.json fields counts and ok. JSON evidence uses field/index anchors because minified raw files do not provide stable per-record source-code line numbers.

## 3.3 Tracking schema

A tracking file is a bz2 compressed UTF-8 JSONL stream. One nonblank line is one dictionary. Production parsing never extracts the file to disk.

| Source field | Canonical field/use | Important detail |
| --- | --- | --- |
| frameNum | TrackingFrame.frame_num | Source frame identity, not zero-based array index |
| videoTimeMs | video_time_ms_source; video_time_s = /1000 | Primary alignment clock |
| period | source_period; canonical_period | Only 1–4 canonical; preserve invalid original value |
| periodElapsedTime | period_elapsed_s | Period-relative clock |
| periodGameClockTime | game_clock_nominal_s | Display/nominal match clock; not alignment clock |
| homePlayers, awayPlayers | PlayerSample tuples | Side is source-list membership |
| homePlayersSmoothed, awayPlayersSmoothed | candidates/status/smoothed XY | Source supplied; project does not recompute smoothing |
| balls | raw BallSample values | Empty means missing; more than one means ambiguity at preparation |
| ballsSmoothed | smoothed BallSample values | Dict/null; XY can be null despite valid Z |
| gameRefId | contextual identity check | Often non-null only at file start |
| game_event_id, possession_event_id | optional cross-references | Not a unique event-row key |
| version/generatedTime/smoothedTime | source metadata | Not all retained as canonical MatchMetadata fields |

Evidence: ingestion/models.py:L165-L244; ingestion/loaders.py:L358-L540,L552-L559. **DOCUMENTED** schema description calls the top level “19 keys,” but also discusses generatedTime and smoothedTime; treat the actual input keys and parser as authoritative rather than using a prose key count as a validator. The loader accepts extra keys; it is not a strict exhaustive JSON Schema validator.

### Directly observed first tracking record

At data/raw/Tracking Data-20261015T150539Z-1-001/Tracking Data/3812.jsonl.bz2, first nonblank record:

| Item | Observed value |
| --- | --- |
| frameNum / period | 4226 / 1 |
| videoTimeMs | 141007.674 |
| periodElapsedTime | 0.0 |
| gameRefId / version | 3812.0 / "4.1.0" |
| First home player | jersey "21", raw (13.975,19.244), VISIBLE, HIGH |
| Same jersey's source-smoothed position | (14.357,18.306) |
| Raw ball | (-0.97,0.29,0.0), VISIBLE |

Only three records were streamed. The next two raw positions for the first home player both equal (13.953,19.228), while smoothed coordinates change. This corroborates repeated raw positions locally; it does not establish a match-wide effective measurement rate.

## 3.4 Events and versions

Events are JSON arrays, not JSONL. A source row includes identifiers, times, gameEvents, possessionEvents, initialTouch, fouls, grades, stadiumMetadata, and player/ball snapshots. EventRow retains selected typed fields and verbatim source sections; it is not a flattened lossless representation of every possible top-level field.

**DOCUMENTED historical counts:**

| Property | March | May | Root |
| --- | ---: | ---: | ---: |
| Rows | 97,444 | 144,541 | 144,541 |
| IT rows | 0 | 47,097 | 47,097 |
| Rows in repeated ID pairs | 53 | 53 | 53 |
| Repeated pairs | 25 | 25 | 25 |

Common-field comparison: May versus March reports 3,932 changed values in 1,225 matched rows. Root versus May reports 9,763 sequence changes. The earlier 3,954 number is explained in the comparison document as 22 false changes from pairing colliding ID pairs incorrectly. This report did not rerun those comparisons.

Root content is provisionally associated with June 3, 2025 because of fields and the supplied change log; a root directory is not itself a dated release identifier. The DOCX XML was read in memory and confirms dated entries for March 14, May 1 and June 3 and their stated additions. Folder-to-release authority remains **NOT VERIFIED**. Evidence: docs/event_version_comparison.md:L7-L97; audit_lib.py:L151-L233.

## 3.5 Identifiers and roster relationships

Tracking jerseys are strings in raw data; event jerseys and player IDs may be numeric, while roster/metadata IDs are strings. The loader converts accepted integral forms to integers, rejecting bool and invalid forms.

A jersey is not a person ID. The safe join is:

~~~text
match → metadata home/away team IDs
tracking source_side + normalized jersey
    → roster (side, shirt_number)
    → team_id + player_id
~~~

Duplicate roster shirts for a side are excluded from the join. Unknown shirts remain unresolved; no “nearest player” inference is attempted. Caller-specified quarantine periods withhold team/player IDs while preserving source side and coordinates. Global players.csv is not used by the production board path; DOB/height are not needed for rendering and should not enter shared reports.

Evidence: ingestion/models.py:L95-L151; ingestion/loaders.py:L92-L105,L272-L293,L396-L471; docs/dataset_schema_audit.md:L157-L176.

## 3.6 Coordinates, periods, missingness and limitations

**DOCUMENTED/SOURCE convention:** pitch-centered meters, +X right and +Y up in main-camera view. Screen Y grows down, so the renderer flips that axis only. No attacking-direction normalization is performed. Ball Z is retained but not used to give the marker a three-dimensional projection.

Source period 0 is not half 1: canonical_period=None. Extra-time tracking is historically absent in 10510 and 10511, while events continue. Suspected side identity anomalies in four extra-time matches are hypotheses, not authority for automatic swaps. The board service does not automatically configure quarantine_periods.

Raw missing, ambiguous and estimated are distinct. A coordinate (0,0) is a legitimate pitch-center observation; None is absence. Player estimated/low-confidence caveats are carried into prepared objects. Source speed is optional and its unit remains unresolved by the audit; original derived movement computes from coordinates instead.

Ten matches have documented May/root event-coordinate disagreement with supplied tracking. Policy is to render tracking, use root provisionally for event content, retain all versions, and use March as a compatibility reference rather than physical ground truth. Evidence: docs/event_version_comparison.md:L63-L97; match_data.py:L44-L63,L175-L245; server.py:L135-L148.

# Chapter 4 — Milestone 1: Dataset Investigation and Audit

## 4.1 Inventory is filesystem evidence, not a schema audit

inventory.scan walks deterministically, uses lstat-style metadata, classifies whole path context, and records paths/sizes. It never opens dataset content. It rejects linked scan roots, skips symlinks/junction entries, and raises InventoryScanError if any traversal/stat operation fails. Missing modalities, unknown files and duplicate candidates become issues; an incomplete traversal does not become a clean inventory.

classify supports a timestamped download wrapper, an optional bare inner family folder and, for events only, one version folder. Conflicting family nesting and nested timestamped wrappers are unknown rather than guessed. Root events use None internally, remaining distinct from a literal folder named "(root)". JSON serialization uses an explicit root boolean/name object to avoid conflation.

Evidence: inventory.py:L155-L197,L209-L326,L332-L377. Tests: test_inventory.py:L253-L302,L327-L430,L436-L644.

## 4.2 Audit modules and artifacts

| Script | What it examines | What it establishes and does not establish |
| --- | --- | --- |
| event_versions.py | Three event arrays, collision-safe row pairs, common fields, coordinate snapshots | Structural/content differences; not which version is physically correct |
| tracking_scan.py | Every streamed record, first occurrence per consecutive frame, duplicate groups, flags/counters | Historical source distributions and candidate conflicts; no production cleaning |
| event_alignment.py | Every event version against one tracking pass | Temporal coverage and coordinate agreement; not physical event correspondence |
| extra_time_identity.py | Event carrier-to-ball distances and selected tracking evidence | Suspicion flags; not confirmed team-side correction |
| audit_lib.py | Discovery, row matching, flattening, Aligner, output runner | Shared reproducibility methods; not imported into production synchronization |

Existing outputs live under data/processed/audit_1b/<script>/<match>.json. The scripts support --matches, --workers, --force, --aggregate. **These commands were not executed during this task.** They are not pure no-write commands: their sources read raw files but their runner writes diagnostic JSON, usually with a temporary file and replace. This distinction explains why the task used existing artifacts rather than rerunning “read-only audits.” Evidence: scripts/audit/README.md:L1-L39; audit_lib.py:L374-L504.

## 4.3 Duplicate-safe event comparison

An event ID pair can repeat. Dictionary indexing only by (gameEventId,possessionEventId) overwrites rows and can manufacture differences. Audit matching instead creates:

~~~text
signature = (gameEventId, possessionEventId, startTime,
             gameEventType, possessionEventType)
key = (signature, ordinal_among_equal_signatures_in_source_order)
~~~

A source row gets one key, and one counterpart at most. Surplus rows and differing multiplicities remain explicit. Leaf paths existing in both rows are compared; null versus value counts as a value difference, absent versus present counts as a schema difference. Position lists are compared separately.

Evidence: audit_lib.py:L151-L256; event_versions.py:L27-L93. Production event IDs are instead match:version:source_row_index, so production does not need to collapse colliding annotation IDs.

## 4.4 Scientific lessons driving later code

The audit exposed repeated raw samples, estimated players, missing balls, null smoothed XY, conflicting smoothed jerseys, mixed identifier types, event versions and incomplete extra time. Ingestion therefore validates before use; preparation refuses arbitrary candidates; synchronization names its policy and carries provenance. These are technically supported relationships between findings and implementation, not a claim about a private debugging discussion.

Reference match 3812 is an example, not an ingestion special case. Other documented examples include 3814 for ambiguity, 3816 for version-coordinate disagreement, and 10510/10511 for event-only extra time. Evidence: docs/dataset_schema_audit.md:L20-L30,L178-L215; loaders.py:L43-L49.

## 4.5 Completion evidence and limits

Git commits establish that investigation code and reports were committed. Existing audit outputs establish saved results exist. Tests specify expected behavior. None alone proves formal milestone acceptance; PROJECT_RULES and audit documents still contain pending-review labels. Expensive historical totals are **DOCUMENTED**, not independently regenerated here. Licensing, checksums and exploratory scratch-only results remain separate open matters.

# Chapter 5 — Milestone 2A: Data Ingestion and Parsing

## 5.1 Domain models and scalar validation

The data models are dataclasses, mostly frozen/slot-based, not database tables. Enums prevent a typo such as "ROOT" from silently selecting a different version. SourceRef records relative path, source kind, version and optional index. TrackingFrame retains every collapsed source index. PlayerSample separates raw XY, smoothed candidates, status and identity. BallSample separates raw and smoothed coordinates, visibility and raw candidate count.

All exact fields appear in Appendix B, with definitions at ingestion/models.py:L20-L244.

Scalar rules:

| Helper | Behavior | Consequence |
| --- | --- | --- |
| validate_match_id(match_id) | int excluding bool or string of 1–9 ASCII digits | rejects separators/traversal/absolute-path strings |
| as_float(v) | finite int/float excluding bool; otherwise None | numeric strings are not coordinate values |
| as_int(v) | int, integral finite float, or signed ASCII-digit string | schema string IDs can be normalized |
| _req(d,path,what) | follows dotted object path; raises on absent/null | required metadata cannot default to invented values |
| _read_json(path) | UTF-8 json.load, wraps I/O/JSON errors | small JSON sources are materialized |

Evidence: loaders.py:L40-L105.

## 5.2 Dataset discovery and public interface

Dataset(root) checks that root is a directory, but delays inventory until needed. _inv caches the inventory. _match validates ID and finds the match grouping. _one requires exactly one candidate; zero and duplicates both raise contextual IngestionError.

metadata(match_id) reads and parses one source. roster(match_id,metadata=None) loads or uses metadata, verifies identity, and parses one roster. load_events(match_id,version) materializes iter_events. iter_events yields typed rows but first calls json.load on the entire JSON array: **it is not a streaming JSON-array decoder**. tracking_source returns a Path plus relative SourceRef. iter_tracking_frames accepts optional roster/metadata, repeat-collapse flag, quarantine periods and a frame limit and returns the canonical tracking generator.

Evidence: loaders.py:L110-L200. Parameter-by-parameter exact signatures, return expressions, callers and callees are in Appendix B.

## 5.3 Metadata and roster parsing

Metadata accepts one dictionary or a one-element list. It verifies requested ID, positive finite FPS, two distinct integer team IDs, exactly one pitch with positive dimensions and an integer competition ID. Optional period times become None when absent; home-side booleans are accepted only as booleans.

Roster requires a list of objects with team.id, player.id and shirtNumber. Each team must equal one metadata team. Entries preserve source row indices. Duplicate side-shirt groups are reported and Roster.index excludes them rather than selecting one. Repeated player IDs become issues. A roster with a foreign match/team context is rejected before tracking is joined.

Evidence: loaders.py:L205-L293; models.py:L141-L151. The parser normalizes identifiers, not coordinates or team orientation.

## 5.4 Event parsing

parse_events first requires an array, counts repeated scalar ID pairs, rejects unhashable ID structures, then yields each row in original order. A supplied gameId that disagrees with the requested match is fatal. Missing IDs/time/invalid period become flags when recoverable.

_event_row derives source/canonical period from gameEvents.period; event_time_s from eventTime; identity from gameEvents; attack direction only from "L"/"R". "nan" becomes None. The row identifier includes source array index, so duplicate ID pairs retain separate identities. Source sections are preserved in attributes and positions_snapshot. No ranking, deduplication or version merging is performed.

Evidence: loaders.py:L298-L353. An event's startTime is not substituted for eventTime. This matters when a possession event occurs inside a longer game event.

## 5.5 Compressed tracking state machine

_iter_records opens bz2 in text-read mode, skips blank lines, decodes one JSON object per record and yields (nonblank_record_index,dict). Thus index is not necessarily a physical line number if blank lines occur. JSON/structural/compression failures become IngestionError.

The canonical generator maintains one current record group:

~~~text
PSEUDOCODE — explanatory, not copied implementation
for each source record:
    validate frameNum, finite videoTimeMs, optional gameRefId
    if collapsing and frameNum equals current group's frameNum:
        append source index
        if timestamp or period differs: retain pending fatal error; stop
        record which compared fields differ
    else:
        yield current canonical frame if one exists
        enforce emitted-frame bound
        start next group; calculate gap/nonmonotonic-frame flag
after EOF or error:
    yield remaining current good group, then raise pending error if any
~~~

Consecutive equal frame numbers collapse. Nonconsecutive reused frame numbers remain separate occurrences. The generator does not maintain a global deduplication set or sort the entire file. Frame number decreases are flagged, not automatically repaired. Synchronization independently rejects decreasing video timestamps.

Repeated fields compared include period clocks, raw/smoothed player lists and raw/smoothed ball. Timestamp and period disagreements are fatal structural conflicts. Other disagreements preserve the first canonical sample plus conflict-field names and all source indices, allowing later preparation to withhold unreliable coordinates.

Evidence: loaders.py:L497-L607. Memory is independent of whole-match frame count, but scales with the current record and the source-index list for its repeats; it is not an absolute constant under adversarial unlimited record/repeat sizes.

## 5.6 Player and ball extraction

_players groups smoothed entries by normalized jersey, retaining candidate objects in source order. A single candidate or identical finite XY candidates can populate smoothed XY; conflicting candidates populate no canonical smoothed position. Unjoinable/malformed smoothed entries remain in unmatched_smoothed with flags. Every raw dict entry creates a PlayerSample; raw entries are not silently reduced to one jersey.

The identity join is separate from coordinate validity. A missing coordinate can have known player identity. A quarantined identity can have usable geometry. ESTIMATED and LOW flags refer to source raw-player attributes and remain caveats, not uncertainty probabilities.

_ball reads valid raw dictionaries and an optional smoothed dictionary. It records raw_count and flags absence/multiple balls/null smoothed XY. The canonical BallSample may retain the first raw entry, but preparation refuses a unique selected raw ball when raw_count>1. Repeated smoothed-ball conflicts explicitly clear canonical smoothed values and set BALL_SMOOTHED_UNRESOLVED.

Evidence: loaders.py:L378-L540; match_data.py:L206-L245.

## 5.7 Worked ingestion example

For the observed match-3812 first record:

1. _iter_records yields source index 0 and frameNum 4226.
2. check accepts gameRefId 3812.0 as requested ID 3812 and videoTimeMs as finite.
3. _build_frame converts 141007.674 ms to 141.007674 s; source/canonical periods both 1.
4. A raw jersey "21" becomes integer 21; XY stay exactly (13.975,19.244).
5. Smoothed candidate coordinates stay distinct at (14.357,18.306).
6. Roster joining, when enabled, uses ("home",21), not shirt 21 across both teams.
7. TrackingFrame exposes source_record_indices=(0,), source path, raw/smoothed fields and flags.

This is a **source-derived walk-through of DIRECT SAMPLE values**, not a production ingestion run during this task. Calls and exact type definitions: loaders.py:L396-L540; models.py:L165-L217.

## 5.8 Quality reporting and controlled writes

TrackingQuality accumulates frames, source records, flags, period counts, identity status and observation counts. quality_report reads metadata/events/tracking and stores fatal_errors instead of returning a misleading clean report. A bounded 2A scan is conservatively not declared complete, even if the requested bound exceeds EOF; 2C uses a more precise extra-frame mechanism.

safe_output_path places reports under the lexical approved processed boundary, rejects raw/outside destinations and linked components, validates final and temporary files, and write_quality_report refuses existing reports unless overwrite=True. The writer intentionally uses a temporary JSON and os.replace. **None was invoked here.** Ingestion CLI exit codes: 0 report without fatal errors, 1 fatal source error reported, 2 argument/context/output-location error.

Evidence: ingestion/validation.py:L25-L175,L180-L238; ingestion/__main__.py:L1-L48. This is path validation, not a claim of race-free defense against a hostile process replacing directories concurrently.

# Chapter 6 — Milestone 2B: Tracking and Event Synchronization

## 6.1 Inputs, clocks and outputs

align_events(events,frames,*,match_id,event_version,fps,tolerance_s=None,spatial=False,spatial_tolerance_m=0.01) consumes EventRows and a chronological TrackingFrame iterable. It returns MatchSynchronization containing one AlignmentResult for every input event in source order, scan_complete/fatal_error, period ranges, tolerance, and ordering_violation.

Times are floating-point seconds: eventTime versus videoTimeMs/1000. FPS comes from metadata, not a hardcoded 29.97 in the algorithm. Period elapsed and game clocks are preserved elsewhere but not used for this matching.

The accepted common timebase is a **provisional data compatibility convention**, not independent physical validation. Finding a nearby frame in a dense timeline proves temporal coverage only.

Evidence: synchronization.py:L1-L23,L93-L159,L228-L351; audit_lib.py:L261-L322.

## 6.2 Candidate selection mechanics

1. Validate positive finite FPS and tolerance, explicit EventVersion, and all event match/version contexts.
2. Buffer events; create a stable sorted index of rows with time. Keep original rows for final output ordering.
3. Stream frames once. Reject foreign match IDs. Detect backward timestamp order.
4. Maintain a forward pointer into sorted event times. Discard from the candidate window events more than tolerance behind the current frame.
5. For each event in the frame-time ± tolerance window, record any-period proximity for diagnostics.
6. Only same canonical period and valid frame period can improve the event's best match.
7. Replace current best only if the new absolute residual is smaller by more than 1 ns. With increasing frame times, ties retain the earlier frame.
8. Copy a compact frame summary, source indices and flags, plus an optional raw snapshot for spatial comparison.
9. Restore source order and produce a matched result or explicit unmatched reason.

Evidence: synchronization.py:L228-L312. Signed residual is event_time_s − frame_video_time_s: positive means the event timestamp follows the selected frame.

## 6.3 Tolerance and boundary semantics

Default tolerance:

~~~text
tau = 1 / fps seconds
at 29.97: tau = 0.0333667000333667 s = 33.3667000333667 ms
accept if |event − frame| <= tau + 1e-9 s
~~~

The 1 ns epsilon handles floating-point boundaries/ties; it is not measurement precision. There is no synthesized frame beyond a period. However, contrary to literal “no extrapolation beyond the range” wording in some documents, an event slightly outside the first/last sample can still attach to that real boundary sample if within tolerance. Events farther than the expanded range are outside_period_frame_range. No position or timestamp is extrapolated.

Evidence: synchronization.py:L248-L296,L315-L332; tests/test_synchronization.py:L73-L96.

### Example 1: real first event

Tracking frame 4226: 141.007674 s. Event row 0: 141.008 s, period 1.

Residual = 141.008 − 141.007674 = **+0.000326 s = +0.326 ms**. It is far below 33.3667 ms. Frame 4227 is at 141.041041 s, 33.041 ms away, so frame 4226 wins. This uses actual inspected timestamps; selection is inferred directly from code, not executed here.

### Example 2: deterministic exact midpoint

**Synthetic teaching values at metadata FPS 30:** frames at 10.000000000 s and 10.033333333 s, same period. Event at 10.0166666665 s has equal residual magnitude. Earlier-frame tie rule retains the first frame; residual is approximately +16.667 ms. Reversing frame order violates the chronological precondition, rather than authorizing arbitrary tie handling.

### Example 3: inclusive boundary

**Synthetic:** one same-period frame at 10.0 s, fps=30, event at 10.0+1/30. It matches that real frame. At 10.0+1/30+0.000001, it does not: the added microsecond exceeds the 1 ns arithmetic epsilon. No extra frame is created.

### Example 4: correct time, wrong period

**Synthetic:** an event at 100 s in period 2; only a period-1 frame at 100 s exists. That frame is never selected. If the scan completes, the any-period proximity diagnostic yields period_label_differs_from_nearest_frame. A period-2 frame at 100.02 s would be valid if within tolerance, even though the period-1 frame is closer.

## 6.4 Missing and failed scan semantics

| Situation | Outcome |
| --- | --- |
| Missing event time | missing_event_time |
| Invalid canonical event period | invalid_source_period |
| Matching depends on unread tracking | tracking_scan_incomplete |
| Other-period frame near the event, no valid same-period match | period_label_differs_from_nearest_frame |
| Complete scan, no frames for event period | no_tracking_for_period |
| Complete scan, event beyond period range plus tolerance | outside_period_frame_range |
| Interior gap with no candidate in tolerance | no_frame_within_tolerance |

Precedence matters: missing time is checked before invalid period; incomplete tracking must not yield a false absence claim. On an ordinary read failure after last timestamp T, a previously matched event can remain settled if its time is strictly less than T−tau. On a backward timestamp violation this watermark is no longer trustworthy, so **all tracking-dependent event outcomes become incomplete**, including apparently early matches.

Context mismatch raises SynchronizationError to the caller instead of being absorbed into an incomplete result. Evidence: synchronization.py:L51-L52,L228-L332; tests/test_synchronization.py:L242-L326.

## 6.5 Duplicate timestamps, records and provenance

Consecutive repeated source records collapse upstream and selected provenance contains every index. Equal-time distinct frames are processed in source order; strictly-closer replacement does not arbitrarily replace the first equal-time candidate. The guaranteed earlier-time tie rule cannot distinguish identical timestamps except by source order.

A match result carries source file and selected source_record_indices, not only frame_num. That distinction later prevents reused frame numbers from receiving duplicate event attachments.

Evidence: synchronization.py:L93-L119,L283-L312; match_data.py:L316-L332.

## 6.6 Optional spatial assessment

Spatial assessment compares embedded event coordinates with **raw** tracking coordinates, independent of playback coordinate policy. Conflicting duplicate (side,jersey) groups are excluded; identical finite duplicates may agree. Quarantined players and repeated-frame raw conflicts are not used as if trustworthy.

At least 11 comparable players are required. At least 90% must be within 0.01 m to label coordinate_agreeing. Ball agreement is separate, and missing/conflicting balls are not assessable. An agreeing player label does not imply that the ball agrees.

validated_correspondence remains "not_assessed" in every outcome. This is essential because both coordinate copies may derive from the same tracking source; equality is not independent truth. Evidence: synchronization.py:L164-L223; tests/test_synchronization.py:L184-L237,L350-L388.

## 6.7 Complexity and tactical significance

For F tracking frames and E events, event sort costs O(E log E); frame scanning is O(F + total candidate-window work). Memory is O(E) compact best matches, with O(E×players) snapshots when spatial assessment is requested. Tracking frames are not all materialized by synchronization itself.

The board needs this mapping so selecting an event can navigate to a real observed frame rather than guessing from event startTime, annotation IDs or rounded displayed minutes. It also needs unmatched rows to remain visible when no trustworthy position exists.

# Chapter 7 — Milestone 2C: Preparation and Integration

## 7.1 Preparation is not a new smoothing engine

prepare_frame wraps a canonical TrackingFrame; it does not generate new positions. Source raw and source-smoothed coordinates are retained separately. Type normalization happened in ingestion; there is no rotation, pitch-size normalization, clock reset, resampling, imputation or coordinate averaging in this layer.

CoordState is AVAILABLE, MISSING or AMBIGUOUS. Coordinate objects contain numeric XY only when available; ambiguous objects may retain candidate XY. SelectedCoordinate additionally states raw, smoothed, raw_fallback or no source. Identity quarantine remains an independent AttributionStatus.

Evidence: match_data.py:L44-L143,L148-L245.

## 7.2 Policy table

| Raw | Smoothed | RAW | SMOOTHED | SMOOTHED_THEN_RAW |
| --- | --- | --- | --- | --- |
| Available | Available | Raw | Smoothed | Smoothed |
| Available | Missing | Raw | Missing | Raw fallback with reason |
| Available | Ambiguous | Raw | Ambiguous | Raw fallback with reason/caveat |
| Missing | Available | Missing | Smoothed | Smoothed |
| Ambiguous | Missing | Ambiguous | Missing | Ambiguous |
| Missing | Missing | Missing | Missing | Missing |
| Ambiguous | Ambiguous | Ambiguous | Ambiguous | Ambiguous |

These states refer to each representation independently. A known-good raw coordinate is not a selected member of an ambiguous smoothed candidate set. Fallback therefore explicitly says raw_fallback and adds RAW_FALLBACK_FOR_SMOOTHED.

Evidence: match_data.py:L148-L162; tests/test_match_data.py:L79-L142.

## 7.3 Player conflicts and identity

_raw_players groups valid raw XY by side and jersey. Multiple distinct positions for the same key make its raw representation ambiguous. Repeated-record side-list disagreements also make raw representation ambiguous. Smoothed-list conflicts are handled independently. PreparedPlayer keeps the canonical PlayerSample so a consumer can inspect flags, raw visibility and all smoothed candidates.

position_usable means selected coordinate is available and canonical period is valid. identity_usable means roster attribution is RESOLVED. Neither guarantees correct physical identity or measurement quality. An ESTIMATED source player can be position_usable, with an explicit caveat. This avoids confusing “draw-able” with “measured and independently validated.”

Evidence: match_data.py:L167-L216; tests/test_match_data.py:L147-L164.

## 7.4 Ball ambiguity, including a missing retained canonical ball

Repeated-ball conflicts are evaluated before testing whether frame.ball is None. Thus if one repeated record lacks the ball and another supplies it, raw/smoothed prepared state is AMBIGUOUS regardless of which source record was first. A remaining BALL_MISSING source flag is historical source quality, not an instruction to override the prepared state.

When several raw balls are present, raw is ambiguous; smoothed can independently be usable if the source supplies one unconflicted smoothed coordinate. With no ball evidence and no conflict, states are missing, never zeros. Conflicting smoothed repeat values are not used even if ingestion retained an earlier sample.

Evidence: match_data.py:L218-L245; tests/test_match_data.py:L225-L259.

## 7.5 Event attachment

MatchData.frames(attach_events=True) first uses cached synchronization or performs a full synchronization pass. It groups matched results by selected frame_source_record_indices. While streaming the second pass it pops the matching group once, then verifies frame number and period. Unmatched results remain available via unmatched_events.

A reused frame number at a later source occurrence therefore receives only its own selected events. Several events at the same occurrence remain several rows in source order.

**Synthetic example:** frame numbers [10,11,10], increasing video times [10/30,11/30,0.4], indices [(0,),(1,),(2,)]. Events at 10/30 and 0.4 attach to indices (0,) and (2,), respectively. They do not both attach to every number 10.

Evidence: match_data.py:L316-L332; tests/test_match_data.py:L197-L222.

## 7.6 Quality summaries and counting units

QualitySummary is opt-in, never part of open(). It states scan status, processed count, bound, fatal error, metrics, breakdown units, periods, quarantine context and optional synchronization summary.

| Metric family | Numerator basis | Denominator |
| --- | --- | --- |
| frame/gap/repeat/conflict/ball state | Canonical frame occurrences yielded | Scanned canonical frames |
| player coordinate/visibility/confidence/identity | Raw player entries in canonical frames | Scanned player observations |
| event flags | Event rows of selected version | Loaded event rows |
| synchronization coverage | Alignment rows | All event rows in that synchronization |
| players_per_side_per_frame | Number of observed entries per source side | Breakdown unit is frame-sides |
| selected_coordinate_source | Chosen representation per player entry | Player observations |

A missing player record is not a sample at (0,0). It is absent from the observation denominator and affects the side-entry-count distribution. Thus “player_raw_missing=0” does not imply every expected player was observed. Side counts are not an active-lineup model.

Repeated consecutive records contribute one canonical frame, not repeat-count times its players. The implementation's notion of “unique frame” is consecutive collapse, not global uniqueness under arbitrary nonconsecutive frame-number reuse.

Evidence: match_data.py:L341-L496; tests/test_match_data_quality.py:L43-L69,L180-L185.

## 7.7 Complete, partial and failed

For max_frames=N, the summary requests up to N+1 frames. If an extra canonical frame exists, count only N and report partial. If EOF arrives after exactly N, report complete. A generator failure reports failed with retained partial counts. Missing-period assertions are allowed only for a complete scan.

The probe may read upstream lookahead beyond the counted N frames; “bounded” does not mean precisely N compressed lines or zero decompression buffering. include_synchronization=True explicitly incurs a separate full pass unless cached, even if the quality scan itself is bounded.

Evidence: match_data.py:L307-L313,L392-L433,L470-L496; tests/test_match_data_quality.py:L86-L115.

# Chapter 8 — MatchData API: Complete Technical Walkthrough

## 8.1 What “API” actually means

MatchData is a **Python object API**. It has no HTTP listener, route decorators, sockets or background server. The separate playback server exposes an HTTP/JSON API built on it. The browser consumes FrameState dictionaries, not dataclass instances or a Python iterator.

A WebSocket interface is **NOT FOUND IN CURRENT REPOSITORY** for this original path. So are server-side paginated frame fetching, a persistent seek index and per-client playback sessions. Exact signatures for every relevant Python method are reproduced in Appendix B.

## 8.2 MatchData public contract

| Member | Input | Output/cost | Validation, cache and edge cases |
| --- | --- | --- | --- |
| MatchData.open(dataset_or_root,match_id,event_version,**kw) | Dataset or root; explicit version; policy/quarantine options | MatchData handle, no tracking scan | Dataset root directory check; ID format, EventVersion and CoordinatePolicy validated by constructor; source existence resolved lazily |
| metadata | No argument | MatchMetadata | Cached after first metadata read; not a tracking read |
| roster | No argument | Roster | Uses metadata; cached; identity conflicts remain issues/unresolved |
| events() | No argument | New list of cached EventRow references | Requested version only; duplicate/source order preserved; list is copied, nested dictionaries are not deeply frozen |
| frames(attach_events=False,max_frames=None) | Opt-in attachment and optional bound | Generator of PreparedFrame | Lazy body; cached metadata/roster; no list of all tracking frames unless consumer builds one |
| synchronize(spatial=False) | Optional spatial comparison | MatchSynchronization | Cached separately by spatial boolean; explicit full streaming tracking pass |
| unmatched_events() | No argument | Tuple of unmatched alignments | Uses nonspatial synchronization cache; may trigger full scan on first call |
| quality_summary(max_frames=None,include_synchronization=False) | Quality bound and sync inclusion | QualitySummary | Explicit scan, status/count units; optional extra synchronization pass |
| prepare_frame(frame,policy=RAW,events=()) | Canonical frame | PreparedFrame | Pure wrapper; no file I/O |
| select(raw,smoothed,policy) | Two representations + enum | SelectedCoordinate | Pure policy selection, not interpolation |

Evidence: match_data.py:L148-L162,L206-L332. Changing public mutable attributes after cached reads is not cache invalidation; create a new MatchData for a new match/version/quarantine context. The intended contract is one configured handle per context, not a multi-match mutable session object.

## 8.3 Example invocation and returned structure

~~~python
# Documentation example only; not executed in this audit.
from football_fatigue.ingestion import EventVersion
from football_fatigue.match_data import MatchData, CoordinatePolicy

md = MatchData.open("data/raw", "3812", EventVersion.ROOT,
                    policy=CoordinatePolicy.RAW)
meta = md.metadata
roster = md.roster
rows = md.events()
pf = next(md.frames(max_frames=1))
# pf.frame_num, pf.video_time_s, pf.source_period, pf.canonical_period
# pf.frame.source_record_indices; pf.players; pf.ball; pf.events
~~~

For the observed first source record, a source-derived expected subset is frame_num=4226, video_time_s=141.007674, canonical_period=1, source_record_indices=(0,), a player selected from raw at (13.975,19.244), and raw ball at (-0.97,0.29,0). Prepared events are empty unless attachment is requested. This is not represented as an executed API response.

## 8.4 Loading a match: implemented sequence

~~~mermaid
sequenceDiagram
    participant User as Python caller
    participant MD as MatchData
    participant DS as Dataset
    participant INV as Inventory
    participant FS as Raw files
    User->>MD: open(root, match_id, EventVersion, policy)
    MD->>DS: construct(root), directory check
    MD-->>User: handle (no tracking scan)
    User->>MD: metadata
    MD->>DS: metadata(match)
    DS->>INV: scan(root), once lazily
    INV-->>DS: unique-source candidates
    DS->>FS: read metadata JSON
    DS-->>MD: validated MatchMetadata
    MD-->>User: cached object
~~~

## 8.5 Requesting tracking frames

~~~mermaid
sequenceDiagram
    participant C as Consumer
    participant M as MatchData
    participant D as Dataset
    participant T as bz2 source stream
    C->>M: frames(max_frames=N)
    M-->>C: lazy generator
    C->>M: next(generator)
    M->>D: metadata + roster (cached when available)
    M->>D: iter_tracking_frames(...)
    D->>T: read one current group plus lookahead
    T-->>D: source dictionaries
    D-->>M: TrackingFrame
    M->>M: prepare_frame(policy)
    M-->>C: PreparedFrame
~~~

## 8.6 Requesting synchronized events

~~~mermaid
sequenceDiagram
    participant C as Consumer
    participant M as MatchData
    participant S as synchronize_match
    participant D as Dataset
    C->>M: synchronize(spatial=False)
    alt uncached
        M->>S: dataset, ID, version, quarantine
        S->>D: metadata + selected event rows
        S->>D: tracking iterator
        loop one chronological tracking pass
            D-->>S: TrackingFrame
            S->>S: update nearest same-period candidates
        end
        S-->>M: all source-ordered AlignmentResults + scan status
    end
    M-->>C: cached MatchSynchronization
~~~

## 8.7 Seeking: separate real interfaces

**MatchData seek method:** **NOT FOUND IN CURRENT REPOSITORY**. Re-read frames from the start or build a consumer index.

**Python controller:** seek_time(timestamp_s) binary-searches a materialized list; seek_frame_num uses a first-occurrence map. It clamps outside the loaded timestamp range. It does not stream from raw on each seek.

**Browser:** jumpToIdx selects an array index; jumpToFrame uses findIndex on source frame numbers. The timeline range input is an index slider, not a timestamp HTTP query.

~~~mermaid
sequenceDiagram
    participant U as User
    participant JS as Browser controls
    participant F as In-memory FRAMES array
    participant R as Canvas renderer
    U->>JS: timeline input(index)
    JS->>JS: pauseIfPlaying()
    JS->>F: clamp index, select existing frame
    JS->>R: render()
    R->>R: toCanvas meters to pixels
    Note over JS,F: No HTTP seek request; no MatchData seek index
~~~

Evidence: controller.py:L251-L293; board/index.html:L1343-L1360,L1469-L1472.

## 8.8 Adapter API and exact payload roles

frame_to_state(pf,roster_names=None,event_lookup=None) returns:

~~~text
{
 frame_num, video_time_s, canonical_period, source_period,
 period_usable, has_gap,
 players: [{side, jersey, x, y, coord_source, estimated,
            jersey_confidence_low, identity_usable,
            player_id? , name?}],
 ball: null | {x,y,z,coord_source},
 events: [{event_row_id, game_event_type, possession_event_type,
           canonical_period, event_time_s, residual_s, quality_label,
           validated_correspondence, source_team_id, source_player_id}]
}
~~~

Only position_usable players/ball are rendered. Confirmed labels require identity_usable; otherwise jersey remains a source identifier. Full SourceRef/candidate/flag collections are not copied to FrameState, although they remain accessible in Python PreparedFrame. A browser JSON payload therefore has less provenance detail than MatchData.

build_roster_names maps player_id to nickname. build_roster_lookup maps player_id to name/jersey/side/position_group/team_id. build_event_index preserves every alignment, adds matched boolean and frame/residual or unmatched reason, enriches names/types/team, classifies goals and foul sections. It does not sort rows or verify physical action.

Evidence: adapter.py:L30-L224.

## 8.9 Python PlaybackController API

Constructor accepts a nonempty sequence of PreparedFrames or FrameState dictionaries and EventItems/dictionaries. It builds a list and a first-occurrence frame-number map; events preserve input order. State properties expose current frame, total frames, all events and FPS.

play/pause/toggle_play/set_speed/restart/step update state. The class has no real-time scheduling loop: a caller must call step or seek. set_speed stores a positive multiplier; it does not itself animate.

seek_frame_num returns False for unavailable numbers. seek_time binary-searches timestamps; equal-distance tie behavior is traversal-dependent because only strictly smaller differences replace best, unlike synchronization's explicit earlier-frame policy. set_event_filter and get_filtered_events filter, but do not reorder input chronology. select_event leaves position unchanged for unmatched/outside-segment events while recording selection. next_event/prev_event traverse filtered source sequence and loaded-frame availability.

from_match_data(match_data,max_frames=1500,policy=SMOOTHED_THEN_RAW) synchronizes, builds an event index and materializes adapted frames. Its policy argument is unused: the supplied MatchData.policy actually governs positions. The factory does not supply raw events to build_event_index, so type/name enrichment differs from the server's richer call.

Evidence: controller.py:L88-L173,L178-L413. This controller is implemented and tested as a Python utility, but it is not the authoritative state of the actual browser board.

# Chapter 9 — Ziad's Tactical Board Implementation

## 9.1 Files, technology and visual structure

Original implementation resides in board/index.html (inline CSS, HTML and JavaScript), playback/server.py, playback/adapter.py and the optional Python controller. The board uses CanvasRenderingContext2D and DOM elements, not React, SVG-based player rendering, WebGL or a Python GUI. The embedded page includes match setup, fixture badges, scoreboard, pitch canvas, player list, event filters/details, timeline, speed control and keyboard bindings.

Git d89d0e6 records these files under author zeyadkkhaled. Whether that account corresponds to the user-named Ziad is contextual, not independently authenticated. Chapter 15 separates intended responsibility from account evidence.

Evidence: board/index.html:L1-L879,L934-L1037; Git d89d0e6 --stat.

## 9.2 Canvas sizing and pitch drawing

resizeCanvas derives available center-panel dimensions, subtracts scoreboard space, and preserves metadata pitch_length_m/pitch_width_m aspect ratio before flooring canvas dimensions. Canvas backing size is not explicitly multiplied by devicePixelRatio.

drawPitch fills a grass background, ten alternating stripe divisions, touchline rectangle, halfway line, center circle of radius 9.15 m, center mark, penalty boxes of depth 16.5 m and width 40.32 m, goal symbols and penalty spots 11 m from each end. These dimensions are drawing constants, not inferred from player positions. There is no comprehensive football pitch rules engine: small goal-area boxes and penalty arcs are not drawn by this function.

Evidence: board/index.html:L937-L1002.

## 9.3 Exact coordinate transformation

For pitch length L, width B and canvas pixel dimensions W,H:

~~~text
scaleX = W / L
scaleY = H / B
pixelX = (x_m + L/2) * scaleX
pixelY = (B/2 - y_m) * scaleY
~~~

toCanvas implements these formulas exactly. Translation moves the centered coordinate system to the top-left canvas frame; subtraction flips vertical orientation. No homography or camera calibration is required because inputs already use pitch meters.

For the observed player raw (13.975,19.244), use an **illustrative**, not observed browser, canvas 1050×680 on a 105×68 pitch:

~~~text
scaleX = scaleY = 10 px/m
pixelX = (13.975 + 52.5)*10 = 664.75
pixelY = (34 - 19.244)*10 = 147.56
~~~

Pitch center becomes (525,340). A smoothed-policy drawing would use the different source coordinates; it must not be described as the same measurement. Ball raw (-0.97,0.29) maps to (515.3,337.1).

Evidence: board/index.html:L959-L965; direct tracking sample index 0.

## 9.4 Player markers, ball and missing data

drawPlayer creates a circle of radius 10 pixels, shadow, team color and jersey label. Home is cyan (#22d3ee), away orange (#f97316). ESTIMATED players use opacity 0.55, dashed outline and a dark fill; labels remain jersey numbers, not global IDs. Names/player metrics are shown in DOM lists/details, not written as IDs above every marker.

drawBall creates a yellow marker of radius 7 pixels with radial glow and shadow. Ball Z is delivered by JSON but not used to change size/height. Missing or ambiguous selected ball becomes null upstream, so render simply omits it. No “last-known ball” hold or interpolation is implemented.

position_usable is enforced by the adapter, not recomputed by JavaScript. Unknown identity can still be displayed as a source-side jersey marker. Coordinates outside the pitch are not clamped; the pitch-sized canvas may clip off-pitch observations rather than showing an exterior margin. Goals partly outside the canvas may likewise be clipped.

Evidence: adapter.py:L61-L98; board/index.html:L880-L884,L1004-L1037,L1097-L1109.

## 9.5 Rendering cycle and interactions

render chooses FRAMES[PB.idx], clears the canvas, draws pitch and then players/ball, updates HUD and scoreboard. Current shared code inserts later overlay calls in this function; they are not original board behavior and are excluded here.

updatePlayerList rebuilds DOM rows and binds clicks. showPlayerAnalytics presents original precomputed segment-level movement summaries; the values are not live instantaneous measurements recomputed at every PB.idx. Selecting an event builds detail text and may jump to its frame; selecting a player changes the detail panel.

No positional interpolation is implemented in the original Canvas loop. Source-smoothed coordinates are pre-existing dataset values, not interpolation performed by the browser. Evidence: board/index.html:L1097-L1217,L1260-L1360.

# Chapter 10 — Playback and Temporal Architecture

## 10.1 Two playback implementations, not one shared state

Python PlaybackState and browser PB are different objects in different runtimes. The browser never calls Python controller play(), step() or select_event(). Server routes expose arrays, not controller state endpoints. PB contains idx, playing, speed, lastTs and frameAcc. META, FRAMES, EVENTS and ANALYTICS hold the loaded segment and context.

Evidence: controller.py:L66-L78; board/index.html:L887-L904; server.py:L342-L404.

## 10.2 Match selection and segment loading

The browser fetches a catalog, groups by stage and chooses a fixture. The UI supports first half, second half, full or custom duration/frame count and coordinate policy. Launch posts to /api/load; backend synchronization scans the whole match even when a short segment is requested, then a second tracking pass collects the chosen segment. This is explicit source behavior, not a claim of subsecond loading.

scope half1 selects period 1, half2 period 2, full all periods; otherwise an explicit period argument can select a period. minutes limits elapsed video time from the first accepted frame, not sum of within-period playing time. max_frames limits retained frames, not synchronization cost. Full scope can include inter-period timestamp jumps in its duration budget.

Evidence: server.py:L135-L233; board/index.html:L1573-L1644.

## 10.3 Animation math

tick(ts) receives requestAnimationFrame's browser timestamp in milliseconds:

~~~text
elapsed_ms = ts - PB.lastTs
PB.frameAcc += elapsed_ms * PB.speed
frame_duration_ms = 1000 / META.fps
while accumulated_time >= frame_duration_ms:
    subtract one frame duration
    increment PB.idx, unless at the end
render selected observed frame
~~~

If the browser is slow, several indices can advance before one draw, so displayed frames can be skipped; no positions are synthesized. Play resets accumulator/lastTs and restarts from zero if already at the end. Pause changes state and icon; step and timeline input clamp to valid indices.

Evidence: board/index.html:L1362-L1394,L1443-L1484.

## 10.4 Critical distinction: sample-rate playback versus timestamp-gap playback

The loop does **not** schedule each next frame using its actual video_time_s difference. A jump from 10.0 s to 20.0 s in adjacent retained entries is traversed in one FPS interval. Period breaks and gaps are labelled by HUD, but elapsed real video duration is compressed.

The displayed scoreboard clock uses floor(video_time_s/60) and seconds modulo 60, not periodGameClockTime or elapsed match minutes. For frame 4226, the source-derived display is approximately 2:21, while periodElapsedTime is 0 at kickoff. “Match Time” labels must therefore be explained as video-time display in current implementation.

These are **SOURCE VERIFIED limitations**, not hypothetical missing performance enhancements. Evidence: board/index.html:L1112-L1155,L1362-L1377; ingestion/models.py:L198-L217.

## 10.5 Event navigation details and divergences

Event click: find by event_row_id, show details, highlight, then jumpToFrame if matched. Unmatched stays visible and never fabricates a frame. A matched event outside the loaded segment produces a toast.

Browser next/previous select the first/reversed-first filtered matched event with event_time beyond current frame time ±0.001 s. They do not track a selected-event ordinal like the Python controller. Several events on the same frame can therefore be skipped or repeatedly selected depending on residual times; Python controller tests for same-frame progression do not prove browser parity.

Both browser findIndex and Python first-occurrence maps seek by frame number, not source-record provenance. MatchData attachment is provenance-safe, but that stronger identity does not survive into the event-navigation JSON contract. Reused frame numbers remain a downstream risk.

Evidence: board/index.html:L1260-L1430; controller.py:L318-L413; adapter.py:L147-L224.

## 10.6 Memory, buffering and performance

Data ingestion and synchronization are streaming. load_match_data is not: it retains both raw_prepared_frames and JSON frame dictionaries during load, then computes movement summaries. Browser fetch then holds the JSON-decoded FRAMES array as another full segment copy. /api/frames has no pagination, query time-window or chunk protocol. json.dumps creates a complete response byte string.

O(F×players + E) backend/browser storage is a consumer decision; do not promise whole-system bounded memory merely because the parser streams. Startup without a preload avoids tracking loading until selection. Catalog caching is keyed by root; synchronization caching is per MatchData instance and not persistent across fresh /api/load requests.

Evidence: server.py:L52-L101,L193-L275,L456-L463; match_data.py:L253-L300; board/index.html:L1646-L1688.

# Chapter 11 — Frontend/Backend Integration

## 11.1 Original endpoint contract

The following table describes actual original routes retained in current BoardHandler. Later /api/spatial and /api/report.csv additions are outside the original baseline.

| Method/route | Request | Actual success response | Handler/notes |
| --- | --- | --- | --- |
| GET /tactical or /tactical/ | No query contract | HTML board | Current route; original baseline used / or /index.html |
| GET /api/matches | No query contract | Array of catalog objects | get_matches_catalog(root); source failures can silently omit matches |
| GET /api/match | No query contract | {loaded:false}, or metadata/context plus event index and goals | Excludes frames and analytics; current also excludes spatial_analytics |
| GET /api/frames | No query contract | Entire retained FrameState array | Empty array before load; no timestamp parameter |
| GET /api/events | No query contract | Entire event index, matched and unmatched | Not restricted to loaded segment |
| GET /api/analytics | No query contract | Dictionary keyed side_jersey → movement metrics | Empty object before load |
| GET /api/load | match, scope, period, minutes, max_frames, policy query parameters | status, match_id, scope, frames_loaded, events_count, teams, scores | Synchronous data loading, protected by shared lock |
| POST /api/load | JSON with same selection fields | Same success acknowledgement | Browser's actual launch path |
| OPTIONS | CORS request | 200, GET/POST/OPTIONS and Content-Type allowed | No separate resource-specific schema |

GET load defaults match to "3812", scope half1 and policy smoothed_then_raw. Unknown policy strings fall back to smoothed_then_raw instead of explicit rejection. POST malformed JSON is caught and replaced with {}, so it may initiate the default match load. Query numeric conversions and load exceptions do not have a consistent structured error envelope.

_json always emits 200 on success using json.dumps(allow_nan=False), UTF-8, Content-Type application/json, Content-Length and wildcard Access-Control-Allow-Origin. Unknown routes use 404. There are no authentication/session mechanisms in the original handler. Loading errors may end as handler exceptions rather than intentional 400/422 JSON responses.

Evidence: server.py:L298-L463. These are source observations, not HTTP requests performed in this audit.

## 11.2 Load request and response example

~~~json
{
  "match": "3812",
  "scope": "custom",
  "minutes": null,
  "max_frames": 100,
  "policy": "raw"
}
~~~

This is a valid **conceptual request using actual field names**. A successful acknowledgement has status="ok", match_id="3812", scope="custom", frames_loaded, events_count, team names and final score fields. Numeric counts are deliberately not fabricated here: no server request was made.

get_matches_catalog scans IDs, reads metadata, infers stage labels from matchweek and caches the sorted list. It is not based on a tournament API. “All 64” is the documented dataset population, not a hardcoded catalog loop.

## 11.3 Actual load-to-board sequence

~~~mermaid
sequenceDiagram
    participant U as User
    participant B as board/index.html
    participant H as BoardHandler
    participant L as load_match_data
    participant M as MatchData
    participant A as adapter/movement
    U->>B: choose match, scope and policy; launch
    B->>H: POST /api/load JSON
    H->>L: call under shared lock
    L->>M: open + metadata + roster + events
    L->>M: synchronize(False), full pass
    L->>M: frames(attach_events=True), second pass
    loop selected segment
        M-->>L: PreparedFrame
        L->>A: frame_to_state
    end
    L->>A: summarize_match_movement
    L-->>H: full payload dictionary
    H-->>B: success acknowledgement
    B->>H: GET /api/match
    H-->>B: context, events and goals
    B->>H: GET /api/frames
    H-->>B: retained frame array
    B->>H: GET /api/events
    H-->>B: complete event index
    B->>H: GET /api/analytics
    H-->>B: original movement summaries
    B->>B: reset PB; build list; resize; render
~~~

Current shared server also computes a later spatial payload; that addition is deliberately not traced as original behavior.

## 11.4 Serialization, state and safety boundaries

The adapter creates plain dictionaries before json.dumps. Enum-like concepts become simple strings/bools; positions already passed preparation. Detailed raw candidates/provenance are reduced, so a web consumer cannot reconstruct the full audit trail from FrameState alone.

payload is a class-level shared active match, not per-user state. A lock serializes loads; GETs do not participate in an atomic versioned multi-endpoint snapshot. In the current threaded server, a reload can occur between separate metadata/frames/events requests, mixing match contexts for concurrent clients. The original single-threaded baseline avoided simultaneous requests but still had one shared active payload.

Server binds to 127.0.0.1 by default. This is a local demonstration architecture, not a production multi-user API. There is no database, message broker, background job ID, progress streaming or websocket. Evidence: server.py:L280-L285,L383-L452,L905-L918.

## 11.5 Static assets and browser errors

_serve_file reads configured HTML paths and responds with content length and content type; an unavailable asset yields 404. The original board is one file, with no frontend rebuild required to interpret its source. Current "/" landing is a later routing addition; original report examples should use /tactical on HEAD.

fetchJSON throws on non-2xx responses. loadMatchesCatalog logs failures; loadData reopens setup on error; launchSelectedMatch shows a short error/loading message before restoring the form. These are basic recovery behaviors, not retry/backoff, cancellation or validation UI.

Potential web-hardening gaps include source names inserted through innerHTML and broad CORS with state-changing GET load. They were observed statically, not exploited. The original system's local controlled-data use does not establish safety for untrusted uploaded datasets or external deployment.

# Chapter 12 — Algorithms, Data Structures and Mathematical Foundations

## 12.1 Actual algorithms and tradeoffs

| Algorithm/structure | Location | Why useful; complexity/limits |
| --- | --- | --- |
| Sorted filesystem walk and path-context classifier | inventory.py:L155-L287 | Deterministic inventory, O(files) storage plus directory sorting; no content validation |
| Dictionaries of match/version → candidate lists | inventory.py:L275-L326 | Detect zero/multiple sources without arbitrary selection |
| Generator over bz2 JSONL | loaders.py:L358-L375 | Bounded decompression/record memory; time proportional to input read |
| Consecutive-repeat state machine | loaders.py:L543-L607 | One frame group plus indices, preserves differing fields; no global sort/dedup |
| Counter/set grouping by jersey | loaders.py:L396-L471; match_data.py:L167-L196 | Identity/candidate ambiguity, average O(players+candidates) per frame |
| Event signature + ordinal | audit_lib.py:L151-L195 | Duplicate-preserving cross-version pairing, expected O(rows) indexing |
| Sorted events + moving window | synchronization.py:L228-L312 | O(E log E + F + window work), avoids frame materialization |
| Selected-source-index map and pop | match_data.py:L316-L332 | Expected O(events+frames) attachment, no reuse duplication |
| Counter-based quality accumulation | match_data.py:L392-L496 | Counts/denominators with no full frame list |
| Binary search in Python controller | controller.py:L263-L293 | O(log loaded_frames), assumes timestamp-sorted materialized data |
| Browser frame-number findIndex | board/index.html:L1343-L1352 | O(loaded_frames), simple but ambiguous for reused frame numbers |
| Browser accumulator scheduling | board/index.html:L1362-L1377 | FPS-based frame index advance; does not preserve timestamp-gap duration |
| Meter-to-pixel affine map | board/index.html:L959-L965 | O(1) per object, no perspective calibration |
| Original windowed movement | movement.py:L168-L351 | Path-length/time window and derivatives with heuristic rejection |

## 12.2 Original movement formulas and limitations

The original board includes movement.py, committed with the board baseline. It is distinct from later video-derived analytics.

extract_player_trajectories groups position_usable samples by (source_side,jersey), preserving time, period, selected coordinate source and estimated caveat. It does not require identity_usable, so output is a source-side/jersey trajectory, not guaranteed biological identity.

compute_player_movement splits at nonpositive delta time, gaps greater than max_gap_s (default 0.5 s), and period changes. For valid consecutive points:

~~~text
d_i = hypot(x_i - x_(i-1), y_i - y_(i-1)) meters
dt_i = t_i - t_(i-1) seconds
window_speed_i = sum(window distances) / sum(window dt) m/s
acceleration_i = (filtered_speed_i - previous_filtered_speed) / dt_i m/s²
~~~

The speed filter uses half_w=max(1,window_frames//2), so even window_frames=1 does not disable neighborhood averaging. It is a centered distance/time window, not a Kalman filter. Source positions are not smoothed by this function.

Speeds above 12 m/s are excluded from distance/speed accumulation and flagged. Absolute acceleration above 10 m/s² is cleared, while the associated speed/distance may remain valid. speed_spikes_rejected also counts acceleration rejections. High acceleration/deceleration counters count qualifying derivative samples (threshold ±2.5), not grouped bursts or independent physical events.

Zones: walking <2; jogging [2,4); running [4,5.5); HSR [5.5,7); sprint ≥7 m/s. Zone distance uses each raw step distance assigned to its filtered-speed zone. mean_speed is the arithmetic average of accepted filtered speeds, not total distance divided by accepted duration when dt varies.

Coverage fraction = usable points / total retained segment frames; estimated fraction = estimated usable points / usable points. The is_reliable boolean means coverage≥0.60, a software heuristic, not validated scientific reliability. Substitutes absent before/after their appearance are included in the match-segment frame denominator; it is not playing-time-adjusted coverage.

Evidence: movement.py:L40-L165,L168-L398. The docstring phrases “scientifically validated” and “physiological limits” exceed evidence from synthetic tests alone. These metrics are derived movement descriptors, never fatigue measurements.

## 12.3 Caching is contextual

Inventory is cached inside Dataset, metadata/roster/events in MatchData, synchronization by spatial boolean, catalog by root in the server, and frames in server/browser lists. None is a persisted source-hash invalidated cache. Refreshing source data during a session is not automatically detected. Create a new configured handle/reload for a changed dataset; this is an operational recommendation, not implemented change.

Evidence: loaders.py:L113-L126; match_data.py:L253-L300; server.py:L52-L101.

## 12.4 Alternatives: not historical claims

Possible alternatives include indexed columnar storage, paged frame APIs, binary-search browser indexes, actual timestamp scheduling and event-ID provenance transport. They are **future design options**, not code found in the original system. A WebSocket would not fix coordinate ambiguity; a database would not prove event correspondence; interpolation would create estimated intermediate positions requiring a new scientific policy.

# Chapter 13 — Tests, Validation and Reliability

## 13.1 Inspected evidence, not executed results

Eleven original-system test files contain **192 test-function definitions** by static AST counting. Parametrization expands definitions into pytest items, so this number must not be compared directly with historical “passed” counts. **Zero tests were executed during this audit**, as explicitly required.

Earlier conversation reports of 202 passed/6 skipped or 35 focused passes are not current-run evidence. No historical browser visual acceptance or current CI run was retrieved.

## 13.2 Coverage by test file

| Test file | Behaviors actually asserted |
| --- | --- |
| test_package_installation.py | Installed distribution/module import identity |
| test_inventory.py | Strict paths/names, splits, root/literal-root versions, duplicate candidates, complete JSON records, traversal/stat failures, symlink/junction roots/entries, no file-content reads or mutation |
| test_audit_lib.py | Duplicate candidate classification, signature/ordinal pairing, common fields versus schema differences, matching tolerance/ties/periods, output protections and incomplete-result exits |
| test_ingestion.py | Streaming limits, repeat collapse/uncollapsed mode, conflicts, candidate retention, missing/malformed data, period handling, safe identities/versions, structural errors, reports/output paths/CLI |
| test_synchronization.py | Residual sign, nearest/tie/inclusive tolerance, source ordering, periods, repeated indices, context mismatch, scan failure/order violation, spatial ambiguity/quarantine |
| test_match_data.py | Laziness, explicit version/policy, coordinate states/fallbacks, identity, provenance attachment, reversed ball conflict ordering, immutability |
| test_match_data_quality.py | Known numerators/denominators, partial/complete/failed, event-only periods, observation absence, quarantine, deterministic summary/attachment |
| test_team_handoff_docs.py | Extracts/executes Python examples against synthetic files, verifies metric names and raw digest |
| test_playback_adapter.py | Omission of unusable positions, estimated flag, ball Z, gap propagation, matched/unmatched and enrichment, catalog and half1 loader |
| test_playback_navigation.py | Python controller next/previous, filtering, same-frame events, unmatched selection, modes, chronological stepping, boundaries |
| test_analytics.py | Constant velocity, stationary positions, cadence artifacts, gaps/periods, spike handling, coverage/estimated fraction and zones |

Appendix C lists every test function with exact line numbers and selected assertions. “Synthetic” is stated in test headers/fixtures. They imitate shapes; they do not declare provider truth.

## 13.3 Important missing coverage

No original browser JavaScript harness or browser-render screenshot test was found. Python navigation tests do not exercise PB, tick, jumpToFrame, eventLabel or browser DOM handlers. Existing later CV JavaScript harnesses are a separate workstream and cannot fill this gap.

Gaps visible from source include browser timestamp-gap scheduling, same-frame next/previous parity, HTTP malformed input/status contracts, threaded multi-endpoint consistency, scoreline correctness under duplicate goal representations, period-0 dictionaries entering PlaybackController, reused-frame-number navigation, and package distribution including HTML assets.

The original pyproject has source-package discovery but no explicit package-data declaration for board HTML. Editable source runs can access files by path; wheel/static-asset inclusion is **NOT VERIFIED** because no build was permitted.

Evidence: pyproject.toml:L1-L31; controller.py:L88-L155; server.py:L409-L463; board/index.html:L1220-L1430.

## 13.4 CI and integrity checks

.github/workflows/checks.yml configures Ubuntu/Python 3.12, installs .[dev] with --no-cache-dir and runs pytest on main pushes and pull requests. This is a workflow definition, not a successful-run certificate. The current dev extra adds numpy after the original baseline; original runtime dependencies were empty and remain declared empty.

.gitignore excludes data, virtual environments, caches, credentials and generated output directories; .gitattributes defines line endings. They do not prevent accidental staging of an arbitrarily named archive: the unrelated root zip is visible as untracked. No ignore rules were changed.

No raw-file hash manifest was created, no tests were run and no “clean diff check” result is claimed from this documentation audit.

# Chapter 14 — Complete End-to-End Worked Example

## 14.1 The observed source event

**DIRECT SAMPLE:** root event array row index 0 for match 3812:

| Field | Value |
| --- | --- |
| gameId | 3812 |
| gameEventId / possessionEventId | 6497466 / 6373793 |
| eventTime / startTime | 141.008 / 141.008 s |
| gameEvents.gameEventType / period | FIRSTKICKOFF / 1 |
| possessionEvents.possessionEventType | PA |
| gameEvents.teamId | 366, Netherlands, away |
| gameEvents.playerId | 3315 |
| sequence | 1.0 |

Only necessary technical fields are reproduced. Dataset video links and unrelated personal fields are intentionally omitted. Raw anchor: data/raw/Event Data-20261015T150543Z-1-002/Event Data/3812.json, array index 0. This file was read directly, not through a running server.

## 14.2 Parsing and identity

parse_events verifies the source is an array, counts duplicate key pairs, checks gameId and constructs EventRow. Source-derived row identity is **3812:root:0**, source_version ROOT, source_row_index 0, source/canonical period 1 and event_time_s 141.008. attributes retain the annotation sections and positions_snapshot retains embedded coordinates.

Metadata says home=372, away=366. It therefore supports assigning the event's team 366 to away. This is ID consistency, not proof the physical action happened as annotated. Roster enrichment for the event uses player ID; tracking identity uses side+shirt.

Evidence: loaders.py:L302-L353; adapter.py:L147-L224.

## 14.3 Tracking record and matching

The first tracking frame is 4226 at 141.007674 s, source record 0, same period. The next observed frame is 4227 at 141.041041 s. By the actual nearest-frame rule, 4226 wins, residual +0.326 ms, tolerance 33.3667 ms.

A source-derived expected AlignmentResult subset:

~~~text
event_row_id = "3812:root:0"
status = MATCHED
frame_num = 4226
frame_video_time_s = 141.007674
frame_period = 1
frame_source_record_indices = (0,)
residual_s ≈ 0.000326
validated_correspondence = "not_assessed"
~~~

This calculation is not a stored/new execution result. A full synchronize_match was not run. Source selection rule: synchronization.py:L248-L312.

## 14.4 Preparation and attachment

For a RAW MatchData handle, the first observed home jersey 21 keeps raw (13.975,19.244), distinct from smoothed (14.357,18.306). Raw ball stays (-0.97,0.29,0). With valid period and finite positions, the corresponding selected positions are usable under source-derived logic.

When attachment is requested, the alignment group key is (0,), matching the selected frame occurrence. The event is attached once; source frame numbers alone are not used for this operation. MatchData exposes the full canonical frame/sample and flags.

Evidence: match_data.py:L175-L245,L316-L332.

## 14.5 Backend and frontend continuation: verified source, not executed observation

All remaining steps exist in source, but **no server/browser execution was performed**:

1. load_match_data opens ROOT with the requested policy, synchronizes and iterates the requested segment.
2. frame_to_state omits unusable objects and serializes the prepared usable coordinates.
3. FrameState includes the event's residual and nonvalidated quality label.
4. build_event_index enriches the event panel from EventRow and roster.
5. BoardHandler stores payload and returns arrays via /api/frames and /api/events.
6. loadData fetches and stores arrays, resets PB.idx=0, builds event list and renders.
7. At illustrative canvas 1050×680, jersey-21 raw position draws at (664.75,147.56); raw ball at (515.3,337.1).
8. eventLabel sees possession PA and displays “Pass,” even though its enclosing game event is FIRSTKICKOFF. This label priority is implemented, not invented.
9. Selecting that row jumps via frame_num 4226; details show temporal quality and approximately 0.3 ms residual after display rounding.
10. Play starts requestAnimationFrame scheduling over retained frame indices.

Evidence: adapter.py:L30-L120,L147-L224; server.py:L135-L275; board/index.html:L1220-L1236,L1293-L1339,L1343-L1387,L1646-L1688.

**Verified execution trace stops at raw-file inspection.** The downstream steps above are a source-derived trace through implemented handlers/functions, not a claim that rendering was observed, the network response succeeded, or the event was physically validated.

# Chapter 15 — Ahmed vs Ziad: Technical Responsibility and Integration

## 15.1 Intended responsibility versus defensible authorship

The user assigns Ahmed the data-engineering/integration responsibility and Ziad the board/playback responsibility. Git supports a file-boundary distinction, but account names and commit authorship are not conclusive proof of who typed each line or whether an assistant contributed.

| Responsibility | Intended owner | Git evidence | Limit |
| --- | --- | --- | --- |
| Foundation, inventory, audit, loaders, synchronization, MatchData/handoff | Ahmed | bahr74 commits 582c703 through ed11919; merge authored Ahmed Bahr | Account-to-person identification partly contextual; no line-by-line personal authorship certificate |
| Original board, adapter, controller, server, movement and tests | Ziad | d89d0e6 authored zeyadkkhaled | User says “Ziad”; recorded account is “zeyadkkhaled”; mapping not independently authenticated |
| Shared contracts and integration behavior | Both roles | imports/calls cross the MatchData–adapter boundary | Architecture responsibility is not exclusive ownership |
| Later shared-file changes | Outside this report | diff d89d0e6..HEAD | Not attributed to original implementation |

No authorship was inferred from “Z2,” filenames, chat instructions or a function's naming style alone.

## 15.2 What Ahmed supplies

The data-layer handoff is a configured MatchData object and its contracts: metadata, roster, version-specific EventRows, lazy PreparedFrames, explicit raw/smoothed selection, position and identity usability, source indices, optional synchronization results, unmatched rows and quality summaries.

Ahmed's boundary should answer whether a coordinate exists and is ambiguous, what time/period it belongs to, where it came from and how strongly identity was established. It does not decide team colors, screen scaling or play-button behavior.

Evidence: docs/team_handoff.md:L16-L156; match_data.py:L89-L143,L250-L332.

## 15.3 What Ziad consumes and adds

The adapter converts those objects into render-safe dictionaries; server chooses segment and serves arrays; JavaScript controls playback and Canvas presentation. The board should not parse raw JSONL, merge event versions, replace None with zero or resolve candidate conflicts. It does choose marker appearance, event label/filter behavior and user navigation.

Original movement.py also belongs to the board-era commit. It adds derived metrics; its scientific assumptions require explanation separately from ingestion correctness.

Evidence: adapter.py:L1-L120; server.py:L106-L275; board/index.html:L934-L1430.

## 15.4 Where the handoff weakens

Full source-record provenance exists in MatchData and alignment results but is not transported in the event index/FrameState for browser event seeking. Position versus identity usability is separate in Python, but source-side color can still be interpreted as confirmed team attribution unless the UI is careful. The server does not automatically quarantine historically suspect extra-time periods. Frame gaps are exposed but not scheduled with real timestamp duration.

These are integration limits visible in source; they must not be hidden by saying “the data layer validates everything.”

# Chapter 16 — Architecture Decisions, Problems and Fixes

## 16.1 Reconstructable decisions

| Problem supported by evidence | Implemented solution | Why it works | Remaining limit |
| --- | --- | --- | --- |
| Split downloads and duplicate source candidates | Inventory groups by match/version, Dataset requires exactly one | Avoids picking an arbitrary file | Snapshot inventory is cached |
| Root event version collides with a literal "(root)" folder label | Internal None and explicit JSON root/name | Separates identifiers from display labels | Root release date still inferred |
| Repeated event ID pairs overwrite rows | Audit signature+ordinal; production row-index identity | Preserves multiplicity and one-to-one pairs | Reordering equal signatures between releases still deserves care |
| Conflicting smoothed jerseys | Preserve candidates; canonical smooth XY absent | Does not choose arbitrary first candidate | Raw fallback remains an explicit policy |
| Repeated records disagree in timestamp/period | Fatal contextual error | Prevents scientifically incoherent canonical frame | Partial good frames can precede propagated error |
| Nonmonotonic tracking invalidates a synchronization watermark | ordering_violation invalidates tracking-dependent matches | Avoids false definitive early matches | Browser plain frame iteration itself is not an ordering validator |
| Cross-match low-level alignment | Context guards | Stops false matches between unrelated games | HTTP payload still lacks per-request snapshot versioning |
| Reused frame number duplicates attachments | Source-record-index grouping + pop | Attaches once to selected occurrence | Browser/Python navigation still keys by frame number |
| Missing-first repeated-ball conflict misreported as missing | Conflict decision before canonical-ball-None branch | Order-independent prepared states/counts | Source BALL_MISSING flag remains correctly historical |
| Browser action classification and setup | Possession-first event labels, enriched event index, dynamic catalog | Separates many action rows from enclosing game categories | Label semantics are not uniformly correct; see below |

Evidence: inventory.py:L97-L103,L155-L377; audit_lib.py:L151-L233; synchronization.py:L228-L332; match_data.py:L218-L245,L316-L332; tests/test_match_data.py:L197-L259; Git d89d0e6 commit subject.

These are code/test-supported solutions. The report does not invent dates or conversations for individual debugging steps.

## 16.2 Current risks to explain honestly

1. **Browser timing compression:** gaps/period breaks are traversed by one index step; source timestamps are displayed but not used for interval scheduling (board/index.html:L1362-L1377).
2. **Clock labels:** scoreboard uses video time, not nominal football game clock (board/index.html:L1112-L1139).
3. **Controller/JS divergence:** Python same-frame navigation tests do not cover browser logic (controller.py:L347-L413 versus board/index.html:L1398-L1430).
4. **Period-0 controller construction:** EventItem dictionary normalization uses int(ev.get("canonical_period",1)); an explicit None raises TypeError. The browser avoids this Python controller path, but the utility cannot ingest every valid unmatched EventRow-derived index dictionary unchanged (controller.py:L119-L139).
5. **Factory policy ignored:** PlaybackController.from_match_data declares a policy parameter but never applies it (controller.py:L158-L173).
6. **Source-provenance reduction:** attachment is safe in MatchData, but browser frame-number lookup can choose the wrong reused occurrence (adapter.py:L147-L224; board/index.html:L1343-L1352).
7. **Failure status not promoted by server:** synchronize returns scan_complete/fatal_error, but load_match_data logs matched counts and does not require a complete scan before proceeding; the HTTP payload does not expose the whole synchronization status (server.py:L145-L156,L251-L275).
8. **Identity quarantine not configured automatically:** the service opens MatchData without quarantine periods, even for documented suspect extra time (server.py:L135-L136).
9. **Goal scoreboard:** goal_events and score increments are counted per event row; there is no explicit unique-goal identity/deduplication or own-goal logic in this loop. Physical scoring accuracy is NOT VERIFIED (server.py:L158-L178).
10. **Event label semantics:** JavaScript maps OTB to “Out of bounds,” while audit extra_time_identity explicitly uses OTB as a carrier-associated in-play game event. The label should not be defended as provider-verified; exact provider definition was not independently rechecked in PDF here (board/index.html:L1220-L1226; scripts/audit/extra_time_identity.py:L1-L17).
11. **HTTP input/error handling:** malformed JSON can fall back to default load, unknown policies fall back, unhandled conversions/source failures lack structured errors (server.py:L368-L454).
12. **Memory/session limits:** full arrays and one active shared payload; current threaded requests can mix contexts across endpoints (server.py:L196-L275,L280-L285,L905-L918).
13. **Movement confidence:** source-side/jersey trajectories and ≥60% coverage do not establish validated human physiology or identity (movement.py:L79-L105,L354-L398).
14. **Packaging/default root:** no verified packaged HTML asset contract, and default nested raw path does not exist in inspected layout (pyproject.toml; server.py:L921-L949).
15. **Documentation drift:** docs/team_handoff still calls metrics/UI “not implemented,” while board-era source implements them; README and PROJECT_RULES retain earlier-stage pending-review labels, while current README and architecture now acknowledge the implemented board. Treat stage-specific handoff text as historical, not as proof that source is absent.

These are static risks/defects, not runtime exploit demonstrations or a request to redesign the system. No fixes were implemented.

# Chapter 17 — Technical Discussion Preparation

The answers below intentionally separate software behavior from physical validation. Follow-up prompts expose common misconceptions.

## Q1. What problem does the data pipeline solve?
It turns several differently structured sources into consistent time/period-aware objects with explicit uncertainty. Without it, the board would have to guess file versions, convert IDs and resolve missingness itself. Follow-up: a visually plausible board is not evidence that those guesses were correct. Evidence: loaders.py:L110-L200; match_data.py:L206-L332.

## Q2. Is the original system detecting players from video?
No. It consumes supplied FIFA/PFF pitch coordinates. Video detection and tracking are another workstream. Follow-up: jerseyNum here is a source observation; it is not a ByteTrack track ID. Evidence: ingestion/models.py:L165-L217; board/index.html:L959-L1024.

## Q3. Why are tracking files compressed JSONL?
Each line can be decoded independently while bz2 decompresses a stream, avoiding a full match object in parser memory. Follow-up: JSONL streaming does not imply browser data is streamed; the server later materializes the chosen segment. Evidence: loaders.py:L358-L375; server.py:L196-L229.

## Q4. What makes metadata loading “cheap”?
It reads a small JSON source and may trigger filename inventory, but never reads tracking content. Follow-up: MatchData.open checks root/arguments, not that every requested modality already exists; missing sources can fail lazily. Evidence: match_data.py:L253-L294; loaders.py:L113-L167.

## Q5. Why not hardcode match 3812?
It is only a development sample. Discovery validates any supplied numeric match and requires unambiguous source files. Follow-up: the server's legacy default 3812 exists, but data-layer algorithms do not special-case that ID. Evidence: loaders.py:L43-L49,L136-L141; server.py:L370-L370.

## Q6. What is the unique player identifier?
player_id is the provider identifier, but tracking gives side+jersey rather than player_id. A roster join supplies the ID. Follow-up: shirt 9 can exist on both sides; jersey alone is not safe. Evidence: models.py:L128-L181; loaders.py:L396-L471.

## Q7. How are duplicate roster shirts handled?
They are flagged and excluded from Roster.index, leaving attribution unresolved. Follow-up: choosing the first duplicate would silently assert identity and is not implemented. Evidence: models.py:L146-L151; loaders.py:L288-L293.

## Q8. Why retain source and canonical period?
The original value is evidence. Period 0/invalid values must not silently become first half. Canonical 1–4 supports valid period logic; None prevents invalid alignment. Follow-up: a field being int-shaped does not make it a valid half. Evidence: loaders.py:L298-L350.

## Q9. Which tracking clock is authoritative for alignment?
videoTimeMs/1000. eventTime is used on the event side. periodElapsedTime and periodGameClockTime have different references. Follow-up: converting units does not prove physical event correspondence. Evidence: synchronization.py:L228-L312; docs/dataset_schema_audit.md:L109-L140.

## Q10. Is FPS exactly 30?
Metadata reports 29.97 for the audited files. Code uses metadata FPS; examples at 30 are explicitly synthetic. Follow-up: a frame interval is 33.3667 ms here, not exactly 33.3333 ms. Evidence: loaders.py:L220-L222; synchronization.py:L248-L252.

## Q11. Why does nearest-frame selection need a tolerance?
Otherwise every event would get a frame, even in a long dropout. Tolerance limits temporal distance. Follow-up: a one-frame cutoff is provisional policy, not measured annotation precision. Evidence: synchronization.py:L248-L296.

## Q12. How does tie-breaking work?
Synchronization only replaces a best candidate when strictly closer by more than epsilon; chronological input means the earlier frame remains on a tie. Follow-up: the separate Python controller's binary search has no identical explicit earlier-tie guarantee. Evidence: synchronization.py:L283-L296; controller.py:L263-L293.

## Q13. Can a nearer frame from another half win?
No, same canonical period is mandatory. Any-period proximity only diagnoses period-label disagreement. Follow-up: period matching is a hard constraint, not a soft score. Evidence: synchronization.py:L283-L296,L315-L332.

## Q14. What happens when tracking fails halfway through?
Synchronization reports incomplete/fatal status and avoids missing-period conclusions; only provably settled prefix matches can remain under ordinary truncation. An ordering violation invalidates every tracking-dependent result. Follow-up: the board service currently does not promote all this status into its payload. Evidence: synchronization.py:L297-L332; server.py:L145-L156.

## Q15. Why preserve all event rows?
Annotation ID pairs repeat and several actions can share a frame. Deduplicating by ID pair can lose legitimate rows or compare wrong release rows. Follow-up: a row's stable within-file identity is match:version:index. Evidence: loaders.py:L302-L353; audit_lib.py:L151-L195.

## Q16. How is event attachment different from synchronization?
Synchronization chooses a frame occurrence. Attachment joins its result to a streamed PreparedFrame on selected source-record indices, once. Follow-up: frame_num is not sufficient under reused numbers. Evidence: match_data.py:L316-L332.

## Q17. What happens to unmatched events?
They remain in synchronization.alignments and unmatched_events; the event index carries matched=false and a reason. The browser displays them without inventing coordinates. Follow-up: being unmatched does not erase the annotation. Evidence: adapter.py:L147-L224; board/index.html:L1260-L1339.

## Q18. Does coordinate agreement verify the action?
No. It establishes agreement between two coordinate representations, potentially sharing a source. It does not validate which player touched the ball or that the event happened. Follow-up: validated_correspondence is always not_assessed. Evidence: synchronization.py:L164-L223.

## Q19. Why keep three event versions?
They differ in content, fields and some coordinate snapshots. Explicit selection preserves provenance and enables comparison. Follow-up: the newest folder is not automatically physically best; root's June mapping remains inferred. Evidence: docs/event_version_comparison.md:L15-L97.

## Q20. What is the default position source?
MatchData defaults RAW; the board service defaults SMOOTHED_THEN_RAW. They are different defaults at different layers. Follow-up: raw fallback is explicitly labelled, not a smoothed measurement. Evidence: match_data.py:L60-L63,L253-L267; server.py:L106-L115.

## Q21. Why not choose the first smoothed candidate?
List order does not identify truth. Conflicting candidates remain ambiguous; a permitted raw fallback uses a different representation with caveats. Follow-up: close coordinates are still conflicting under current exact-equality policy. Evidence: loaders.py:L386-L393; match_data.py:L175-L196.

## Q22. What is the difference between a missing ball and ambiguous ball?
Missing means no valid observation; ambiguous means conflicting alternatives or repeated records disagree. Both can be unusable, but quality metrics must distinguish them. Follow-up: a missing first repeated record cannot override later conflict evidence. Evidence: match_data.py:L218-L245; tests/test_match_data.py:L233-L259.

## Q23. Does (0,0) mean the ball is missing?
No. It is pitch center, a real coordinate. Missing uses None and explicit state; adapter omits it. Follow-up: replacing None with zero visually fabricates a center-pitch object. Evidence: match_data.py:L54-L85; adapter.py:L61-L98.

## Q24. Can a coordinate be usable when identity is quarantined?
Yes. Geometry and identity validity are separate. The point can be drawn with source-side caution without claiming the roster identity. Follow-up: board service does not automatically activate known extra-time quarantine. Evidence: match_data.py:L206-L216; server.py:L135-L136.

## Q25. What does ESTIMATED mean?
The source tracking system estimated a raw player's position rather than observing it directly. Preparation retains a caveat and the board draws a dashed translucent marker. Follow-up: estimated does not mean absent, and available does not mean measured. Evidence: loaders.py:L428-L431; board/index.html:L1004-L1024.

## Q26. Is MatchData a REST API?
No: it is a Python facade. HTTP endpoints belong to BoardHandler and serialize adapter dictionaries. Follow-up: changing a Python method signature does not automatically change a REST route. Evidence: match_data.py:L250-L332; server.py:L280-L463.

## Q27. Does the frontend use a WebSocket?
No original WebSocket implementation was found. It fetches whole JSON arrays after loading. Follow-up: an iterator inside the server is not a network streaming protocol. Evidence: board/index.html:L1530-L1534,L1646-L1688.

## Q28. How do meters become pixels?
Translate centered X by half pitch length, flip Y around half width, then scale each by canvas dimension divided by pitch dimension. Follow-up: this is affine display mapping, not camera homography. Evidence: board/index.html:L959-L965.

## Q29. Does the board interpolate positions?
No. It draws the retained frame at PB.idx. Source smoothing is precomputed by the data provider; movement speed filtering is a separate derived operation. Follow-up: a smooth-looking animation does not prove interpolation. Evidence: board/index.html:L1097-L1109,L1362-L1377.

## Q30. What controls playback speed?
PB.speed multiplies browser elapsed time before accumulating frame intervals. At 2×, twice as many frame steps fit the same wall time. Follow-up: sparse/gap time is still compressed because stepping uses FPS, not adjacent source time differences. Evidence: board/index.html:L1362-L1387,L1465-L1467.

## Q31. What does seeking do?
Browser timeline picks an existing array index; event jumps find a frame number; Python controller separately offers binary-search time seek. MatchData itself has no seek index. Follow-up: browser seek does not request a new backend time window. Evidence: board/index.html:L1343-L1360,L1469-L1472; controller.py:L263-L293.

## Q32. Why can a matched event still be outside the loaded segment?
Synchronization covers the whole match, while the service may retain only one half or a bounded slice. The full event index remains visible. Follow-up: matched means source-frame availability in the full scan, not necessarily browser-segment availability. Evidence: server.py:L145-L156,L180-L233; board/index.html:L1343-L1352.

## Q33. Is all memory bounded?
The ingestion iterator is bounded by current group/record size; synchronization buffers events. The service and browser materialize a selected frame segment. Follow-up: saying “streaming system” without naming the layer is misleading. Evidence: loaders.py:L543-L607; server.py:L196-L275.

## Q34. Are movement metrics fatigue measurements?
No. They are derived distance/speed/acceleration descriptors from selected positions. Physiological fatigue requires additional validated labels and design. Follow-up: the 60% coverage flag is only a code heuristic, not clinical validation. Evidence: movement.py:L79-L105,L168-L351; PROJECT_RULES.md.

## Q35. Why filter speed over a window?
Repeated raw coordinates can alternate zero and doubled finite-difference speed. A windowed path/time ratio reduces this artifact. Follow-up: it can still blur peaks, and the source cadence evidence is sampled; it is not a complete noise model. Evidence: movement.py:L265-L318; tests/test_analytics.py:L69-L96.

## Q36. What validates the implementation?
Synthetic tests specify structural, numerical and failure-case behavior. Existing audit artifacts support dataset investigation. This documentation task ran no tests and no browser. Follow-up: Python controller tests do not validate Canvas behavior or physical annotations. Evidence: Chapter 13 and Appendix C.

## Q37. What is the Ahmed–Ziad interface?
PreparedFrame and alignment contracts on the Python side, converted by frame_to_state into render-safe JSON. Ahmed's role supplies defensible data states; Ziad's role handles presentation/playback. Follow-up: integration can lose provenance if the adapter contract is narrower. Evidence: match_data.py:L89-L143; adapter.py:L30-L224.

## Q38. Why not automatically swap teams in suspicious extra time?
The carrier-distance test suggests a side-identity anomaly but does not prove its cause. Automatic correction would create unsupported attribution. Follow-up: preserve labels, quarantine identity where configured, and seek provider/independent confirmation. Evidence: docs/dataset_schema_audit.md:L145-L151.

## Q39. What should be fixed before calling this production-ready?
Clarify clock/gap scheduling, propagate scan status, preserve occurrence identity through web navigation, validate HTTP inputs, isolate match snapshots, and test browser behavior and packaged assets. Follow-up: these are original-system integration concerns, not a reason to introduce ML or CV. Evidence: Chapter 16.

## Q40. What can we confidently tell the supervisor today?
The original system has a working implementation path in source from immutable compressed tracking and events to a quality-aware Python API and an HTTP/Canvas board. Exact algorithms and tests exist, but runtime visual behavior, physical correspondence and scientific fatigue validity are separate evidence gaps. Follow-up: distinguish code existence, historical test outcomes, this task's static inspection, and independent ground truth.

# Chapter 18 — Final Architecture and Implementation Status

## 18.1 Feature matrix

| Component/capability | Status in original workstream | Confidence boundary |
| --- | --- | --- |
| Multi-match filesystem inventory | Implemented | Source verified; not rerun |
| Read-only production ingestion | Implemented | Source verified; bounded raw sample inspected directly |
| Event version policy | Implemented explicit selection; policy provisional | Root release authority/physical superiority not established |
| Candidate/missingness/identity states | Implemented | No imputation or automatic side correction |
| Same-period temporal alignment | Implemented | Not physical validation |
| Optional raw coordinate agreement | Implemented | Not independent truth |
| MatchData facade and quality summary | Implemented | Lazy base access; explicit expensive scans |
| Provenance-safe Python event attachment | Implemented | Not preserved in browser event seeking |
| Python navigation controller | Implemented utility | Not actual browser playback engine |
| Local HTTP data service | Implemented | Shared payload, materialized arrays, limited error contract |
| Canvas pitch/player/ball rendering | Implemented | No browser runtime/render test in this audit |
| Play/pause/speed/index seek/event panel | Implemented | FPS stepping compresses gaps; controller/JS parity incomplete |
| Original movement summaries | Implemented | Heuristic derived kinematics; not fatigue or validated physiology |
| Timestamp-faithful gap scheduling | Not implemented in original browser loop | Gaps merely surfaced |
| MatchData random-access seek index | Not implemented | Consumer materialization/indexing separate |
| Automatic identity anomaly resolution | Not implemented | Caller quarantine only |
| Source checksum manifest and licensing verification | Open | No fresh manifest or legal authority established |
| Physiological fatigue/injury/genetic models | NOT FOUND IN CURRENT REPOSITORY for this path | Out of scope; no scientific claims |

## 18.2 Actual contract summary

The data-layer contract is configured by dataset root, match ID, explicit EventVersion and CoordinatePolicy. It exposes immutable typed frames with source occurrence provenance, explicit missing/ambiguous coordinates, separate geometry and identity usability, and source-ordered alignment rows.

The web contract is a shared loaded payload containing metadata, frame dictionaries, all event-index rows and movement summaries. It deliberately carries fewer raw details than MatchData. The browser owns its state and renders index-selected frames. This three-part distinction—Python data API, HTTP serialization/service, JavaScript visualization—is the key architectural explanation.

## 18.3 Suggested next milestones, not implementation performed

1. Correct source/video/game-clock labels and decide/document gap-duration playback policy.
2. Carry synchronization completeness and source-occurrence identity through the web contract.
3. Add original-board JavaScript/browser coverage for event navigation, missing data and timing.
4. Validate HTTP selection inputs and give failures intentional JSON/status contracts.
5. Decide extra-time quarantine behavior and make attribution limits visible.
6. Verify packaging/static HTML availability and correct root configuration.
7. Produce a source manifest and confirm access/licensing terms.
8. Validate movement descriptors against appropriate ground truth before physiological interpretation.

These recommendations are intentionally restricted to the original system. No video/CV, ML, imputation or UI implementation was performed.

## 18.4 One-page supervisor explanation

Our original FIFA system reconstructs matches from supplied pitch tracking rather than analyzing video. The dataset has separate compressed tracking files, event releases, metadata and rosters. These sources are not interchangeable: tracking supplies positions, metadata supplies pitch/FPS/team context, rosters link side-and-shirt to identity, and events supply annotated actions.

The ingestion layer reads bz2 JSONL incrementally and converts accepted identifiers and finite numeric values into typed records. It preserves source timestamps, periods, record indices and quality flags. Repeated records can be collapsed without erasing their source indices, and incompatible time/period repeats are rejected. Conflicting smoothed candidates are retained as ambiguity rather than arbitrarily chosen.

Synchronization compares eventTime in seconds with tracking videoTimeMs divided by 1000. It chooses the nearest frame in the same canonical period, accepts residuals within one metadata-derived frame interval and retains the earlier frame on a tie. Every event remains in source order, including duplicates and unmatched events. Temporal matching and optional coordinate agreement are explicitly weaker than validated physical correspondence.

MatchData is the Python handoff facade. It loads reference objects lazily, streams prepared frames, applies an explicit raw/smoothed/fallback coordinate policy and provides opt-in quality summaries. Missing positions remain None; quarantined identity does not erase usable geometry. Events attach to selected source-record occurrences once, avoiding duplication when frame numbers recur.

The board layer converts prepared objects into lightweight dictionaries. A local stdlib HTTP service loads a selected segment, serves JSON, and computes original movement summaries. The browser stores these arrays, draws a 105×68-like metadata-sized pitch using Canvas, translates centered meters into pixels and shows team-colored jersey markers and a ball only when coordinates are usable. Estimated players have distinct visual styling. Events remain visible even when unmatched.

Important limitations remain. The browser plays at metadata FPS by array index; it flags gaps but does not wait their true timestamp duration. Its displayed clock is video time. Python controller tests are not browser tests. Movement metrics are derived heuristics, not physiological fatigue measures. Some event coordinates differ from tracking, extra-time identity is uncertain, and licensing/checksum questions are still open. This report establishes the implementation from source and bounded input inspection; it does not claim a fresh runtime, physical or clinical validation.

## 18.5 Evidence index and inspection manifest

The appendices below are part of this same report, not additional files. They make the explanation auditable against the inspected revision. AST extraction parses text only: it never imports production modules, runs tests, starts a service or reads datasets beyond the stated bounded samples.

**Raw evidence anchors inspected:**

- data/raw/Tracking Data-20261015T150539Z-1-001/Tracking Data/3812.jsonl.bz2 — first three nonblank records, frameNum 4226–4228.
- data/raw/Event Data-20261015T150543Z-1-002/Event Data/3812.json — array length and first two rows; explanatory trace uses row 0 only.
- data/raw/Metadata-20261015T150542Z-1-001/Metadata/3812.json — object index 0.
- data/raw/Rosters-20261015T150541Z-1-001/Rosters/3812.json — array length and first row.
- data/raw/PFF FC Change Log.docx — word/document.xml text, in memory.
- data/processed/audit_1b/{event_versions,event_alignment,tracking_scan,extra_time_identity}/3812.json — selected existing stored output keys and success flags; no recomputation.

Specification PDFs were located/documented by existing schema audit, but **NOT independently reread in this task**. Their historical scientific claims are cited as DOCUMENTED, not newly verified. Global player/competition CSV contents and all-match raw distributions were not re-audited. Browser visuals, real HTTP response behavior, test execution and wheel packaging are NOT VERIFIED during this task.


## Appendix A — Complete inspected file/dependency manifest
This appendix lists 23 original-related source/script files, 11 test files and 16 supporting documents/configuration files. Shared-file exclusions are explicit. Imports below are static text evidence, not proof a branch executes; optional CV/spatial import details are omitted by scope.

### src/football_fatigue/__init__.py

**Source:** src/football_fatigue/__init__.py:L1-L1. **Responsibility:** Package entry/export surface; constructor-free namespace.
**Imports/callees at module level:** No imports.
**Classes:** None. **Top-level functions:** None.
**Lexically detected importing files within this manifest:** No explicit module-name match; facade/re-export/relative or runtime use is described in chapters.

### src/football_fatigue/inventory.py

**Source:** src/football_fatigue/inventory.py:L1-L492. **Responsibility:** Data discovery; input root and filesystem metadata, output Inventory and reports.
**Imports/callees at module level:** __future__: annotations; argparse; collections: defaultdict; dataclasses: dataclass, field; json; os; pathlib: Path; re; stat; sys
**Classes:** FileRecord, Issue, MatchInventory, InventoryScanError, Inventory **Top-level functions:** version_label, _version_key, _family_of, _is_download_wrapper, _unknown, classify, _describe, scan, _find_issues, _version_json, to_dict, format_report, format_match, main
**Lexically detected importing files within this manifest:** scripts/audit/audit_lib.py, src/football_fatigue/ingestion/loaders.py, tests/test_inventory.py

### src/football_fatigue/ingestion/__init__.py

**Source:** src/football_fatigue/ingestion/__init__.py:L1-L37. **Responsibility:** Package entry/export surface; constructor-free namespace.
**Imports/callees at module level:** .loaders: Dataset, as_float, as_int, validate_match_id; .models: AttributionStatus, BallSample, EventRow, EventVersion, IngestionError, MatchMetadata, PlayerSample, QualityFlag, Roster, RosterEntry, SmoothedCandidate, SmoothedStatus, SourceRef, TrackingFrame; .validation: MatchQualityReport, approved_output_root, quality_report, safe_output_path, write_quality_report
**Classes:** None. **Top-level functions:** None.
**Lexically detected importing files within this manifest:** No explicit module-name match; facade/re-export/relative or runtime use is described in chapters.

### src/football_fatigue/ingestion/__main__.py

**Source:** src/football_fatigue/ingestion/__main__.py:L1-L43. **Responsibility:** Module CLI entry; arguments to explicit command orchestration and exit result.
**Imports/callees at module level:** .loaders: Dataset; .models: EventVersion, IngestionError; .validation: quality_report, write_quality_report; __future__: annotations; argparse; sys
**Classes:** None. **Top-level functions:** main
**Lexically detected importing files within this manifest:** tests/test_ingestion.py

### src/football_fatigue/ingestion/models.py

**Source:** src/football_fatigue/ingestion/models.py:L1-L244. **Responsibility:** Domain records/enums; input typed fields, output canonical model instances.
**Imports/callees at module level:** __future__: annotations; dataclasses: dataclass, field; enum: Enum; typing: Any
**Classes:** IngestionError, EventVersion, QualityFlag, SmoothedStatus, AttributionStatus, SourceRef, MatchMetadata, RosterEntry, Roster, SmoothedCandidate, PlayerSample, BallSample, TrackingFrame, EventRow **Top-level functions:** None.
**Lexically detected importing files within this manifest:** src/football_fatigue/ingestion/__init__.py, src/football_fatigue/ingestion/__main__.py, src/football_fatigue/ingestion/loaders.py, src/football_fatigue/ingestion/validation.py, src/football_fatigue/playback/adapter.py, src/football_fatigue/synchronization.py

### src/football_fatigue/ingestion/loaders.py

**Source:** src/football_fatigue/ingestion/loaders.py:L1-L607. **Responsibility:** Ingestion; input discovered raw sources and context, output typed metadata/roster/events/tracking iterator.
**Imports/callees at module level:** .models: CANONICAL_PERIODS, AttributionStatus, BallSample, EventRow, EventVersion, IngestionError, MatchMetadata, PlayerSample, QualityFlag, Roster, RosterEntry, SmoothedCandidate, SmoothedStatus, SourceRef, TrackingFrame; __future__: annotations; bz2; collections.abc: Iterable, Iterator; collections: Counter; dataclasses: replace; football_fatigue: inventory; json; math; pathlib: Path; re; typing: Any
**Classes:** Dataset **Top-level functions:** validate_match_id, as_float, as_int, _req, _read_json, check_identity, parse_metadata, _parse_metadata, parse_roster, _parse_roster, _canonical, parse_events, _event_row, _iter_records, _jersey, _candidate, _smoothed_status, _players, _ball, _build_frame, iter_tracking_frames
**Lexically detected importing files within this manifest:** src/football_fatigue/ingestion/__init__.py, src/football_fatigue/ingestion/__main__.py, src/football_fatigue/ingestion/validation.py

### src/football_fatigue/ingestion/validation.py

**Source:** src/football_fatigue/ingestion/validation.py:L1-L238. **Responsibility:** Quality/output boundary; input canonical observations, output counters/reports and explicit guarded writer.
**Imports/callees at module level:** .loaders: Dataset, validate_match_id; .models: CANONICAL_PERIODS, EventVersion, IngestionError, MatchMetadata, QualityFlag, Roster, TrackingFrame; __future__: annotations; collections.abc: Iterable; collections: Counter; dataclasses: dataclass, field; json; os; pathlib: Path; typing: Any
**Classes:** TrackingQuality, MatchQualityReport **Top-level functions:** quality_report, _is_link, _under, approved_output_root, _assert_no_links, safe_output_path, write_quality_report
**Lexically detected importing files within this manifest:** src/football_fatigue/ingestion/__init__.py, src/football_fatigue/ingestion/__main__.py, src/football_fatigue/match_data.py

### src/football_fatigue/synchronization.py

**Source:** src/football_fatigue/synchronization.py:L1-L351. **Responsibility:** Temporal/spatial alignment; input rows, frames and FPS, output MatchSynchronization.
**Imports/callees at module level:** .ingestion.models: AttributionStatus; .ingestion: Dataset, EventRow, EventVersion, IngestionError, QualityFlag, SourceRef, TrackingFrame, as_float, as_int, validate_match_id; __future__: annotations; collections.abc: Iterable; collections: Counter; dataclasses: dataclass, field; enum: Enum; math; statistics; typing: Any
**Classes:** SynchronizationError, _ContextError, MatchStatus, UnmatchedReason, SpatialStatus, TemporalNote, SpatialAssessment, AlignmentResult, MatchSynchronization **Top-level functions:** _resolve_groups, _event_players, _frame_snapshot, assess_spatial, align_events, _reason, synchronize_match
**Lexically detected importing files within this manifest:** src/football_fatigue/match_data.py, src/football_fatigue/playback/adapter.py, tests/test_synchronization.py

### src/football_fatigue/match_data.py

**Source:** src/football_fatigue/match_data.py:L1-L496. **Responsibility:** Integration/Python API; input Dataset, ID, version and policy, output PreparedFrames and summaries.
**Imports/callees at module level:** .ingestion.validation: TrackingQuality; .ingestion: AttributionStatus, Dataset, EventRow, EventVersion, IngestionError, MatchMetadata, PlayerSample, QualityFlag, Roster, SmoothedStatus, TrackingFrame, validate_match_id; .synchronization: AlignmentResult, MatchStatus, MatchSynchronization, synchronize_match; __future__: annotations; collections.abc: Iterable, Iterator; collections: Counter; dataclasses: dataclass; enum: Enum; pathlib: Path; typing: Any
**Classes:** CoordState, CoordinatePolicy, Coordinate, SelectedCoordinate, PreparedPlayer, PreparedBall, PreparedFrame, MatchData, Metric, ScanInfo, QualitySummary **Top-level functions:** select, _raw_players, _player_coords, _caveats, prepare_frame, build_quality_summary
**Lexically detected importing files within this manifest:** src/football_fatigue/analytics/movement.py, src/football_fatigue/playback/adapter.py, src/football_fatigue/playback/controller.py, src/football_fatigue/playback/server.py, tests/test_match_data.py, tests/test_match_data_quality.py, tests/test_playback_adapter.py

### src/football_fatigue/playback/__init__.py

**Source:** src/football_fatigue/playback/__init__.py:L1-L31. **Responsibility:** Package entry/export surface; constructor-free namespace.
**Imports/callees at module level:** football_fatigue.playback.adapter: build_event_index, build_roster_lookup, build_roster_names, frame_to_state; football_fatigue.playback.controller: EventItem, NavigationMode, PlaybackController, PlaybackState
**Classes:** None. **Top-level functions:** None.
**Lexically detected importing files within this manifest:** No explicit module-name match; facade/re-export/relative or runtime use is described in chapters.

### src/football_fatigue/playback/__main__.py

**Source:** src/football_fatigue/playback/__main__.py:L1-L3. **Responsibility:** Module CLI entry; arguments to explicit command orchestration and exit result.
**Imports/callees at module level:** football_fatigue.playback.server: _cli
**Classes:** None. **Top-level functions:** None.
**Lexically detected importing files within this manifest:** tests/test_ingestion.py

### src/football_fatigue/playback/adapter.py

**Source:** src/football_fatigue/playback/adapter.py:L1-L224. **Responsibility:** Serialization boundary; input prepared frames/alignments/lookups, output render dictionaries and event index.
**Imports/callees at module level:** __future__: annotations; football_fatigue.ingestion.models: QualityFlag; football_fatigue.match_data: PreparedFrame; football_fatigue.synchronization: MatchStatus; typing: Any, Sequence
**Classes:** None. **Top-level functions:** frame_to_state, build_roster_names, build_roster_lookup, build_event_index
**Lexically detected importing files within this manifest:** src/football_fatigue/playback/__init__.py, src/football_fatigue/playback/controller.py, src/football_fatigue/playback/server.py, tests/test_playback_adapter.py

### src/football_fatigue/playback/controller.py

**Source:** src/football_fatigue/playback/controller.py:L1-L413. **Responsibility:** Materialized Python navigation; input frame/event sequences, output mutable PlaybackState.
**Imports/callees at module level:** __future__: annotations; dataclasses: dataclass, field; enum: Enum; football_fatigue.match_data: MatchData, PreparedFrame, CoordinatePolicy; football_fatigue.playback.adapter: build_event_index, build_roster_names, frame_to_state; typing: Any, Sequence
**Classes:** NavigationMode, EventItem, PlaybackState, PlaybackController **Top-level functions:** None.
**Lexically detected importing files within this manifest:** src/football_fatigue/playback/__init__.py, tests/test_playback_navigation.py

### src/football_fatigue/playback/server.py

**Source:** src/football_fatigue/playback/server.py:L1-L953. **Responsibility:** Shared HTTP service, original FIFA branches only; input selection, output payload and HTML/JSON.
Original-relevant current regions: imports/context L24-L101; FIFA load L106-L244 and payload L251-L275 (later spatial fields excluded); original route branches L342-L361,L368-L463; shared HTML helper L523-L535; run/CLI L868-L949 with CV setup assignments excluded. Later routes/helpers are not part of the source catalog below.
**Imports/callees at module level:** __future__: annotations; argparse; csv; football_fatigue.analytics: summarize_match_movement; football_fatigue.ingestion: Dataset, EventVersion; football_fatigue.match_data: MatchData, CoordinatePolicy; football_fatigue.playback.adapter: build_event_index, build_roster_lookup, build_roster_names, frame_to_state; http.server: BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer; io; json; math; pathlib: Path; re; threading; time; typing: Any; urllib.parse: parse_qs, urlparse; webbrowser
**Classes:** BoardHandler **Top-level functions:** get_matches_catalog, load_match_data, run, _cli
**Lexically detected importing files within this manifest:** src/football_fatigue/playback/__main__.py, tests/test_playback_adapter.py

### src/football_fatigue/board/__init__.py

**Source:** src/football_fatigue/board/__init__.py:L1-L4. **Responsibility:** Package entry/export surface; constructor-free namespace.
**Imports/callees at module level:** No imports.
**Classes:** None. **Top-level functions:** None.
**Lexically detected importing files within this manifest:** No explicit module-name match; facade/re-export/relative or runtime use is described in chapters.

### src/football_fatigue/board/index.html

**Source:** src/football_fatigue/board/index.html:L1-L1704. **Responsibility:** Original browser presentation; input JSON arrays and controls, output Canvas and DOM.
**Inputs:** /api/matches, /api/match, /api/frames, /api/events and /api/analytics plus user controls. **Output:** Canvas and DOM. **Served by:** BoardHandler. Later overlay/import additions are excluded from the original trace.

### src/football_fatigue/analytics/__init__.py

**Source:** src/football_fatigue/analytics/__init__.py:L1-L49. **Responsibility:** Package entry/export surface; constructor-free namespace.
Original surface: movement exports only. Later spatial/CV exports in this shared file are excluded. Its package import can still execute those later imports in the current working tree; the historical original baseline did not have them.
**Imports/callees at module level:** football_fatigue.analytics.movement: DerivedKinematics, MovementCoverage, MovementMetrics, TrajectoryPoint, classify_speed_zone, compute_player_movement, extract_player_trajectories, summarize_match_movement
**Classes:** None. **Top-level functions:** None.
**Lexically detected importing files within this manifest:** No explicit module-name match; facade/re-export/relative or runtime use is described in chapters.

### src/football_fatigue/analytics/movement.py

**Source:** src/football_fatigue/analytics/movement.py:L1-L398. **Responsibility:** Original derived analytics; input usable prepared positions, output trajectories/kinematics/summary metrics.
**Imports/callees at module level:** __future__: annotations; dataclasses: dataclass, field; enum: Enum; football_fatigue.match_data: PreparedFrame, PreparedPlayer; math; typing: Any, Iterable, Sequence
**Classes:** TrajectoryPoint, DerivedKinematics, MovementCoverage, MovementMetrics **Top-level functions:** classify_speed_zone, compute_player_movement, extract_player_trajectories, summarize_match_movement
**Lexically detected importing files within this manifest:** src/football_fatigue/analytics/__init__.py, tests/test_analytics.py

### scripts/audit/audit_lib.py

**Source:** scripts/audit/audit_lib.py:L1-L504. **Responsibility:** Audit-only helpers; input sources/results, output comparison/alignment diagnostics and guarded saved results.
**Imports/callees at module level:** __future__: annotations; argparse; bz2; collections: Counter; football_fatigue: inventory; json; math; multiprocessing; os; pathlib: Path; re; sys; time; typing: Any, Callable, Iterator
**Classes:** AuditError, Aligner **Top-level functions:** locate, match_ids, load_json, load_metadata, iter_tracking, pct, dist_stats, dist_xy, classify_duplicate_jerseys, pair_key, signature, keyed_rows, match_rows, repeated_pair_rows, flatten, compare_fields, player_map, compare_positions, coordinate_status, _worker, validate_match_id, _inside, guard_output_dir, safe_destination, run_matches, _save, load_outputs, check_outputs, add_common_args, cli_main
**Lexically detected importing files within this manifest:** scripts/audit/event_alignment.py, scripts/audit/event_versions.py, scripts/audit/extra_time_identity.py, scripts/audit/tracking_scan.py

### scripts/audit/event_versions.py

**Source:** scripts/audit/event_versions.py:L1-L141. **Responsibility:** Audit-only version comparison; three row collections to per-match comparison JSON.
**Imports/callees at module level:** __future__: annotations; argparse; audit_lib; collections: Counter; pathlib: Path; sys
**Classes:** None. **Top-level functions:** version_summary, compare_versions, run_match, aggregate, main
**Lexically detected importing files within this manifest:** No explicit module-name match; facade/re-export/relative or runtime use is described in chapters.

### scripts/audit/tracking_scan.py

**Source:** scripts/audit/tracking_scan.py:L1-L233. **Responsibility:** Audit-only streaming source distributions; tracking records to counters/candidate diagnostics.
**Imports/callees at module level:** __future__: annotations; argparse; audit_lib; collections: Counter; json; pathlib: Path; sys
**Classes:** None. **Top-level functions:** bucket, scan_match, aggregate, show_frame, main
**Lexically detected importing files within this manifest:** No explicit module-name match; facade/re-export/relative or runtime use is described in chapters.

### scripts/audit/event_alignment.py

**Source:** scripts/audit/event_alignment.py:L1-L150. **Responsibility:** Audit-only temporal/coordinate comparison; event versions plus tracking to diagnostics.
**Imports/callees at module level:** __future__: annotations; argparse; audit_lib; collections: Counter; pathlib: Path; sys
**Classes:** None. **Top-level functions:** jersey, frame_snapshot, run_match, aggregate, main
**Lexically detected importing files within this manifest:** No explicit module-name match; facade/re-export/relative or runtime use is described in chapters.

### scripts/audit/extra_time_identity.py

**Source:** scripts/audit/extra_time_identity.py:L1-L138. **Responsibility:** Audit-only side-identity suspicion; carrier and ball observations to heuristic flags.
**Imports/callees at module level:** __future__: annotations; argparse; audit_lib; collections: Counter; pathlib: Path; statistics; sys
**Classes:** None. **Top-level functions:** med, flag, run_match, aggregate, main
**Lexically detected importing files within this manifest:** No explicit module-name match; facade/re-export/relative or runtime use is described in chapters.

### Supporting documentation and configuration

- **PROJECT_RULES.md:L1-L51** — L1 Shared project rules; L5 Milestones; L16 Scientific and data integrity; L27 Engineering; L37 Security and Git hygiene; L45 Manual Git operations; L49 Later research.

- **AGENTS.md:L1-L28** — L1 Codex instructions; L5 Role; L10 Review focus; L18 Working agreement.

- **CLAUDE.md:L1-L22** — L1 Claude Code instructions; L5 Working agreement; L15 Environment.

- **README.md:L1-L185** — L1 Football Fatigue & Performance Prediction; L9 Current status; L29 Unified match data interface (Milestone 2C); L56 Event-to-tracking synchronization (Milestone 2B); L80 Ingestion (Milestone 2A); L104 Dataset inventory; L120 Dataset; L126 Setup (Windows, Python 3.12); L141 Repository structure; L162 Development workflow; L170 Roadmap.

- **pyproject.toml:L1-L28**.

- **.gitignore:L1-L46** — L1 Original, processed, and external datasets; all generated outputs; L12 Environments and Python artifacts; L29 Secrets and local configuration; L40 Machine-specific files.

- **.gitattributes:L1-L9**.

- **.github/workflows/checks.yml:L1-L22**.

- **.github/pull_request_template.md:L1-L11** — L1 Problem and resulting behavior; L5 Validation; L9 Data and scientific implications.

- **docs/architecture.md:L1-L75** — L1 Task 2 architecture; L5 Status; L21 Data flow (planned); L40 Components; L55 Multi-match design; L59 Synchronization and coordinates; L63 Future extension points; L71 Next gates.

- **docs/data_policy.md:L1-L58** — L1 Data handling policy; L5 Source and licensing; L9 Local raw layout (observed from file names, sizes, and hashes); L33 Match ID mapping; L37 Provenance and checksums; L41 Derived data; L49 Missing data; L53 Git and sharing.

- **docs/dataset_schema_audit.md:L1-L215** — L1 Dataset schema audit (Milestone 1B); L7 1. Method and coverage; L24 2. Dataset overview; L32 3. Sources reviewed; L43 4. Tracking files; L94 5. Events embedded in tracking; L102 6. Timing and synchronization feasibility; L141 7. Coordinates and the tactical board; L157 8. Metadata, rosters, global files, and identifiers; L178 9. Data quality findings and spec discrepancies; L202 10. Open questions and decisions needed; L213 11. Recommendations for Milestone 2 (scope, not started).

- **docs/event_version_comparison.md:L1-L97** — L1 Event Data version comparison (Milestone 1B); L7 1. Method (the row collections compared); L15 2. Which version is which; L27 3. Results (all 64 matches, denominators stated); L76 4. Verification of the change log against the files; L86 5. Provisional version policy.

- **docs/data_contracts.md:L1-L173** — L1 Proposed data contracts (Milestone 1B); L11 1. Cross-cutting conventions and invariants; L26 2. Match; L40 3. Team, player, roster; L50 4. Period (one row per `(match_id, source_period)`); L62 5. Tracking frame (one row per unique `frame_num`); L75 6. Player position (one row per `(match_id, frame_num, source_side, jersey_number)`); L94 7. Ball sample; L102 8. Event row (one row per source row); L121 9. Timeline entry (derived) and quality flags; L147 10. Provenance record (per derived output); L151 11. Consistency with the audit (reviewed after the Codex review); L164 12. Open contract decisions (PROVISIONAL until resolved).

- **docs/team_handoff.md:L1-L165** — L1 Developer handoff: the tracking-data interface; L5 1. Setup; L16 2. Open a match; L35 3. Iterate frames lazily; L49 4. Read player and ball coordinates; L66 5. Choose a coordinate policy; L83 6. Interpret missing, ambiguous, and estimated observations; L99 7. Events: aligned and unmatched; L117 8. Quality summaries and scan completeness; L132 9. Building tactical playback; L157 10. Known limitations.

- **scripts/audit/README.md:L1-L39** — L1 Milestone 1B audit verification scripts; L15 one match (fast): replace 3812 with any match id; L20 all matches, then the pooled summary (can also be re-run alone with --aggregate); L29 Definitions used by the scripts; L37 Limits.

These are inspection references, not sources of runtime truth when they conflict with implementation. No secret/local-settings contents were inspected. No frontend dependency lock or original frontend build configuration was found.

## Appendix B — Exact Python symbol, model and call-chain catalog
Signatures below are AST-rendered equivalents of the inspected definitions (formatting/quote style may differ; parameter names, kinds, defaults and annotations are preserved). Dataclass constructors use the declared fields. Returns/guards are source expressions, not evaluated results. “Caller candidates” are terminal-name lexical matches and can include unrelated same-named methods; chapters give the confirmed architectural call paths. Framework-dispatched HTTP methods have BaseHTTPRequestHandler/browser callers even without lexical Python calls. Full behavior and numerical examples are in Chapters 4–12.

### Symbols in src/football_fatigue/__init__.py

### Symbols in src/football_fatigue/inventory.py

#### FileRecord — src/football_fatigue/inventory.py:L63-L69

~~~python
path: str
kind: str
size: int
match_id: str | None = None
version: str | None = None
note: str | None = None
~~~

#### Issue — src/football_fatigue/inventory.py:L73-L77

~~~python
kind: str
detail: str
match_id: str | None = None
paths: tuple[str, ...] = ()
~~~

#### MatchInventory — src/football_fatigue/inventory.py:L81-L86

~~~python
match_id: str
tracking: list[str] = field(default_factory=list)
metadata: list[str] = field(default_factory=list)
rosters: list[str] = field(default_factory=list)
events: dict[str | None, list[str]] = field(default_factory=dict)
~~~

#### InventoryScanError — src/football_fatigue/inventory.py:L89-L94
The scan could not enumerate or stat everything, so no inventory is valid.
**Methods:** __init__.

#### Inventory — src/football_fatigue/inventory.py:L111-L139

~~~python
root: str
matches: dict[str, MatchInventory]
files: list[FileRecord]
issues: list[Issue]
~~~
**Methods:** match_ids, event_versions, coverage, event_version_coverage, files_of_kind.

#### InventoryScanError.__init__ — src/football_fatigue/inventory.py:L92-L94

~~~python
def InventoryScanError.__init__(self, errors: list[str])
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** len; super; super().__init__
**Output:** no explicit return/yield; mutation, exception/command action, or implicit None. See signature and module context.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### version_label — src/football_fatigue/inventory.py:L97-L103

~~~python
def version_label(version: str | None) -> str
~~~
**Declared purpose:** Human-readable event version label (never used as an identifier).
**Caller candidates:** src/football_fatigue/inventory.py:_find_issues@L312; src/football_fatigue/inventory.py:format_match@L432; src/football_fatigue/inventory.py:format_report@L390; src/football_fatigue/inventory.py:to_dict@L343
**Callees visible in body:** No explicit calls; field/literal/attribute computation.
**Output expressions (selected excerpts):** L103: Return version; L100: Return ROOT_LABEL; L102: Return f"'{ROOT_LABEL}' (literal folder)".
**Important source branches:** L99: version is None; L101: version == ROOT_LABEL.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### _version_key — src/football_fatigue/inventory.py:L106-L107

~~~python
def _version_key(version: str | None) -> tuple[bool, str]
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/inventory.py:to_dict@L354
**Callees visible in body:** No explicit calls; field/literal/attribute computation.
**Output expressions (selected excerpts):** L107: Return (version is not None, version or '').
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### Inventory.match_ids — src/football_fatigue/inventory.py:L118-L119

~~~python
def Inventory.match_ids(self) -> list[str]
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** scripts/audit/audit_lib.py:cli_main@L486; src/football_fatigue/playback/server.py:get_matches_catalog@L74
**Callees visible in body:** sorted
**Output expressions (selected excerpts):** L119: Return sorted(self.matches, key=int).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### Inventory.event_versions — src/football_fatigue/inventory.py:L122-L123

~~~python
def Inventory.event_versions(self) -> list[str | None]
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** self.matches.values; sorted
**Output expressions (selected excerpts):** L123: Return sorted({v for m in self.matches.values() for v in m.events}, key=_version_key).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### Inventory.coverage — src/football_fatigue/inventory.py:L125-L133

~~~python
def Inventory.coverage(self) -> dict[str, int]
~~~
**Declared purpose:** Number of matches with at least one file per modality.
**Caller candidates:** src/football_fatigue/inventory.py:format_report@L385; src/football_fatigue/inventory.py:to_dict@L341
**Callees visible in body:** bool; self.matches.values
**Output expressions (selected excerpts):** L133: Return counts.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### Inventory.event_version_coverage — src/football_fatigue/inventory.py:L135-L136

~~~python
def Inventory.event_version_coverage(self) -> dict[str | None, int]
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/inventory.py:format_report@L389; src/football_fatigue/inventory.py:to_dict@L344
**Callees visible in body:** self.matches.values; sum
**Output expressions (selected excerpts):** L136: Return {v: sum((v in m.events for m in self.matches.values())) for v in self.event_versions}.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### Inventory.files_of_kind — src/football_fatigue/inventory.py:L138-L139

~~~python
def Inventory.files_of_kind(self, kind: str) -> list[FileRecord]
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/inventory.py:_find_issues@L324; src/football_fatigue/inventory.py:format_report@L394; src/football_fatigue/inventory.py:to_dict@L371; src/football_fatigue/inventory.py:to_dict@L372
**Callees visible in body:** No explicit calls; field/literal/attribute computation.
**Output expressions (selected excerpts):** L139: Return [f for f in self.files if f.kind == kind].
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### _family_of — src/football_fatigue/inventory.py:L142-L144

~~~python
def _family_of(directory_name: str) -> str | None
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/inventory.py:classify@L165; src/football_fatigue/inventory.py:classify@L181
**Callees visible in body:** _DOWNLOAD_SUFFIX.sub; _DOWNLOAD_SUFFIX.sub('', directory_name).lower; _FAMILY_DIRS.get
**Output expressions (selected excerpts):** L144: Return _FAMILY_DIRS.get(base).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### _is_download_wrapper — src/football_fatigue/inventory.py:L147-L148

~~~python
def _is_download_wrapper(directory_name: str) -> bool
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/inventory.py:classify@L178; src/football_fatigue/inventory.py:classify@L182
**Callees visible in body:** _DOWNLOAD_SUFFIX.search
**Output expressions (selected excerpts):** L148: Return _DOWNLOAD_SUFFIX.search(directory_name) is not None.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### _unknown — src/football_fatigue/inventory.py:L151-L152

~~~python
def _unknown(note: str) -> tuple[str, None, None, str]
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/inventory.py:classify@L170; src/football_fatigue/inventory.py:classify@L172; src/football_fatigue/inventory.py:classify@L175; src/football_fatigue/inventory.py:classify@L183; src/football_fatigue/inventory.py:classify@L184; src/football_fatigue/inventory.py:classify@L188; src/football_fatigue/inventory.py:classify@L191; src/football_fatigue/inventory.py:classify@L197
**Callees visible in body:** No explicit calls; field/literal/attribute computation.
**Output expressions (selected excerpts):** L152: Return (UNKNOWN, None, None, note).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### classify — src/football_fatigue/inventory.py:L155-L197

~~~python
def classify(parts: tuple[str, ...]) -> tuple[str, str | None, str | None, str | None]
~~~
**Declared purpose:** Classify a path (relative components, file name last).
**Caller candidates:** src/football_fatigue/inventory.py:scan@L260
**Callees visible in body:** ', '.join; '/'.join; Path; Path(name).suffix.lower; _MATCH_FILE[family].fullmatch; _family_of; _is_download_wrapper; _unknown; found.group; iter; len; list; next; sorted
**Output expressions (selected excerpts):** L197: Return _unknown(f'unexpected file name in {family} folder'); L170: Return _unknown('not inside a recognised dataset folder'); L172: Return _unknown('conflicting family directories: ' + ', '.join(sorted(present))); L175: Return _unknown(f'unsupported nesting: {family} folder is not a top-level directory').
**Important source branches:** L167: not present; L171: len(present) > 1; L174: families[0] is None; L178: rest and families[1] == family and (not _is_download_wrapper(rest[0])); L186: family == EVENTS; L193: found; L195: Path(name).suffix.lower() == '.pdf'.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### _describe — src/football_fatigue/inventory.py:L200-L206

~~~python
def _describe(error: OSError, root: Path) -> str
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/inventory.py:scan.on_walk_error@L237; src/football_fatigue/inventory.py:scan@L237; src/football_fatigue/inventory.py:scan@L251
**Callees visible in body:** Path; Path(target).relative_to; Path(target).relative_to(root).as_posix; getattr; str
**Output expressions (selected excerpts):** L206: Return f'{target}: {error.strerror or error}'.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### scan — src/football_fatigue/inventory.py:L209-L287

~~~python
def scan(root: str | os.PathLike[str]) -> Inventory
~~~
**Declared purpose:** Scan ``root`` read-only and build an inventory. Never opens file contents.
**Caller candidates:** scripts/audit/audit_lib.py:locate@L49; scripts/audit/audit_lib.py:match_ids@L72; src/football_fatigue/ingestion/loaders.py:Dataset._inv@L123; src/football_fatigue/inventory.py:main@L458
**Callees visible in body:** FileRecord; Inventory; InventoryScanError; Issue; MatchInventory; NotADirectoryError; Path; _describe; _find_issues; classify; dirnames.remove; dirnames.sort; errors.append; files.append; full.relative_to; full.relative_to(root_path).as_posix; getattr; getattr(match, record.kind).append; list; match.events.setdefault; match.events.setdefault(record.version, []).append; matches.setdefault; os.lstat; os.path.isjunction; os.path.islink; os.path.normpath; os.walk; relative.as_posix; root_path.is_dir; skip; skipped.append; sorted; stat.S_ISLNK; stat.S_ISREG; str
**Output expressions (selected excerpts):** L287: Return inventory.
**Important source branches:** L220: os.path.islink(checked) or os.path.isjunction(checked); L225: not root_path.is_dir(); L272: errors; L277: record.match_id is None; L280: record.kind == EVENTS; L242: os.path.islink(full) or os.path.isjunction(full); L253: stat.S_ISLNK(mode.st_mode).
**Explicit failure paths (excerpts):** L221: InventoryScanError([f'{root_path}: the scan root is a symbolic link or junction; pass the real raw-data directory instead']); L226: NotADirectoryError(f'dataset root is not a directory: {root_path}'); L273: InventoryScanError(errors).

#### scan.skip — src/football_fatigue/inventory.py:L232-L234

~~~python
def scan.skip(full: Path, reason: str) -> None
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/inventory.py:scan@L244; src/football_fatigue/inventory.py:scan@L254; src/football_fatigue/inventory.py:scan@L257
**Callees visible in body:** Issue; full.relative_to; full.relative_to(root_path).as_posix; skipped.append
**Output:** no explicit return/yield; mutation, exception/command action, or implicit None. See signature and module context.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### scan.on_walk_error — src/football_fatigue/inventory.py:L236-L237

~~~python
def scan.on_walk_error(error: OSError) -> None
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** _describe; errors.append
**Output:** no explicit return/yield; mutation, exception/command action, or implicit None. See signature and module context.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### _find_issues — src/football_fatigue/inventory.py:L290-L326

~~~python
def _find_issues(inventory: Inventory) -> list[Issue]
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/inventory.py:scan@L286
**Callees visible in body:** Issue; getattr; inventory.files_of_kind; issues.append; len; m.events.get; sorted; tuple; version_label
**Output expressions (selected excerpts):** L326: Return issues.
**Important source branches:** L308: not m.events; L297: not paths; L313: m.events and (not paths); L299: len(paths) > 1; L315: len(paths) > 1.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### _version_json — src/football_fatigue/inventory.py:L332-L334

~~~python
def _version_json(version: str | None) -> dict
~~~
**Declared purpose:** Unambiguous version: root is {"root": true}; a folder named "(root)" is not.
**Caller candidates:** src/football_fatigue/inventory.py:to_dict@L343; src/football_fatigue/inventory.py:to_dict@L352; src/football_fatigue/inventory.py:to_dict@L366
**Callees visible in body:** No explicit calls; field/literal/attribute computation.
**Output expressions (selected excerpts):** L334: Return {'root': version is None, 'name': version}.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### to_dict — src/football_fatigue/inventory.py:L337-L377

~~~python
def to_dict(inventory: Inventory) -> dict
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/ingestion/validation.py:write_quality_report@L234; src/football_fatigue/inventory.py:main@L473
**Callees visible in body:** _version_json; _version_key; inventory.coverage; inventory.event_version_coverage; inventory.event_version_coverage().items; inventory.files_of_kind; inventory.matches[mid].events.items; len; list; sorted; version_label
**Output expressions (selected excerpts):** L338: Return {'root': inventory.root, 'match_count': len(inventory.matches), 'coverage': inventory.coverage(), 'event_versions': [{'version': _version_json(v), 'display': version_label(v), 'matches': count} for v, count in inventory.event_version_coverage().items()], 'matches': {mid: {'tracking': inventory.matches[mid].tracking, 'metadata': inventory.matches[mid].meta....
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### format_report — src/football_fatigue/inventory.py:L380-L417

~~~python
def format_report(inventory: Inventory) -> str
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/inventory.py:main@L487
**Callees visible in body:** '\n'.join; by_kind.get; by_kind[issue.kind].append; defaultdict; inventory.coverage; inventory.coverage().items; inventory.event_version_coverage; inventory.event_version_coverage().items; inventory.files_of_kind; len; lines.append; lines.extend; titles.items; version_label
**Output expressions (selected excerpts):** L417: Return '\n'.join(lines).
**Important source branches:** L388: inventory.event_versions; L410: not issues.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### format_match — src/football_fatigue/inventory.py:L420-L444

~~~python
def format_match(inventory: Inventory, match_id: str) -> str
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/inventory.py:main@L485
**Callees visible in body:** '\n'.join; getattr; inventory.matches.get; len; lines.append; lines.extend; match.events.get; match.events.values; sum; version_label
**Output expressions (selected excerpts):** L444: Return '\n'.join(lines); L423: Return f'Match {match_id}: not found in {inventory.root}'.
**Important source branches:** L422: match is None; L442: scan_wide.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### main — src/football_fatigue/inventory.py:L447-L488

~~~python
def main(argv: list[str] | None=None) -> int
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** argparse.ArgumentParser; data['matches'].get; format_match; format_report; json.dumps; parser.add_argument; parser.parse_args; print; scan; sum; to_dict
**Output expressions (selected excerpts):** L488: Return 0; L461: Return 2; L470: Return 3.
**Important source branches:** L472: args.format == 'json'; L474: args.match; L484: args.match.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

### Symbols in src/football_fatigue/ingestion/__init__.py

### Symbols in src/football_fatigue/ingestion/__main__.py

#### main — src/football_fatigue/ingestion/__main__.py:L19-L39

~~~python
def main(argv: list[str] | None=None) -> int
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** Dataset; EventVersion; ap.add_subparsers; ap.parse_args; argparse.ArgumentParser; print; q.add_argument; quality_report; rep.summary_text; sub.add_parser; write_quality_report
**Output expressions (selected excerpts):** L39: Return 1 if rep.fatal_errors else 0; L38: Return 2.
**Important source branches:** L34: a.write.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

### Symbols in src/football_fatigue/ingestion/models.py

#### IngestionError — src/football_fatigue/ingestion/models.py:L16-L17
Fatal structural error: a source file or record cannot be interpreted. Never used for data-quality issues.

#### EventVersion — src/football_fatigue/ingestion/models.py:L20-L30
Event Data version. There is deliberately no default: callers must name a version.

~~~python
ROOT = 'root'
MARCH_14_2025 = 'march_14_2025'
MAY_1_2025 = 'may_1_2025'
~~~
**Methods:** folder.

#### QualityFlag — src/football_fatigue/ingestion/models.py:L33-L67
Recoverable data-quality flags. Fatal structural problems raise IngestionError instead.

~~~python
REPEATED_RECORDS_PRESENT = 'REPEATED_RECORDS_PRESENT'
REPEATED_RECORDS_DIFFER = 'REPEATED_RECORDS_DIFFER'
NONMONOTONIC_FRAME = 'NONMONOTONIC_FRAME'
FRAME_GAP = 'FRAME_GAP'
INVALID_SOURCE_PERIOD = 'INVALID_SOURCE_PERIOD'
RAW_LIST_MISSING = 'RAW_LIST_MISSING'
DUPLICATE_RAW_JERSEY = 'DUPLICATE_RAW_JERSEY'
SMOOTHED_LIST_MISSING = 'SMOOTHED_LIST_MISSING'
SMOOTHED_JERSEY_WITHOUT_RAW = 'SMOOTHED_JERSEY_WITHOUT_RAW'
SMOOTHED_JERSEY_UNPARSEABLE = 'SMOOTHED_JERSEY_UNPARSEABLE'
SMOOTHED_ENTRY_MALFORMED = 'SMOOTHED_ENTRY_MALFORMED'
BALL_MISSING = 'BALL_MISSING'
MULTIPLE_BALLS = 'MULTIPLE_BALLS'
BALL_SMOOTHED_XY_NULL = 'BALL_SMOOTHED_XY_NULL'
BALL_SMOOTHED_UNRESOLVED = 'BALL_SMOOTHED_UNRESOLVED'
INVALID_COORDINATE = 'INVALID_COORDINATE'
PLAYER_ESTIMATED = 'PLAYER_ESTIMATED'
JERSEY_CONFIDENCE_LOW = 'JERSEY_CONFIDENCE_LOW'
JERSEY_INVALID = 'JERSEY_INVALID'
SMOOTHED_DUPLICATE_IDENTICAL = 'SMOOTHED_DUPLICATE_IDENTICAL'
SMOOTHED_AMBIGUOUS_CONFLICTING = 'SMOOTHED_AMBIGUOUS_CONFLICTING'
SMOOTHED_MISSING = 'SMOOTHED_MISSING'
IDENTITY_UNRESOLVED_JERSEY = 'IDENTITY_UNRESOLVED_JERSEY'
IDENTITY_QUARANTINED = 'IDENTITY_QUARANTINED'
EVENT_KEY_PAIR_REPEATED = 'EVENT_KEY_PAIR_REPEATED'
EVENT_TIME_MISSING = 'EVENT_TIME_MISSING'
EVENT_ID_MISSING = 'EVENT_ID_MISSING'
NO_TRACKING_FOR_PERIOD = 'NO_TRACKING_FOR_PERIOD'
~~~

#### SmoothedStatus — src/football_fatigue/ingestion/models.py:L70-L74

~~~python
UNIQUE = 'unique'
DUPLICATE_IDENTICAL = 'duplicate_identical'
AMBIGUOUS_CONFLICTING = 'ambiguous_conflicting'
MISSING = 'missing'
~~~

#### AttributionStatus — src/football_fatigue/ingestion/models.py:L77-L81

~~~python
RESOLVED = 'resolved'
QUARANTINED = 'quarantined'
UNRESOLVED_JERSEY = 'unresolved_jersey'
NOT_ATTEMPTED = 'not_attempted'
~~~

#### SourceRef — src/football_fatigue/ingestion/models.py:L85-L91
Where a record came from: path relative to the dataset root (POSIX), kind, version label, row/record index.

~~~python
path: str
kind: str
version: str | None = None
index: int | None = None
~~~

#### MatchMetadata — src/football_fatigue/ingestion/models.py:L95-L124

~~~python
match_id: int
kickoff_date: str
competition_id: int
competition_name: str
season: str
matchweek: int | None
home_team_id: int
away_team_id: int
home_team_name: str
away_team_name: str
stadium_id: int | None
stadium_name: str | None
pitch_length_m: float
pitch_width_m: float
fps: float
home_team_start_left: bool | None
home_team_start_left_extra_time: bool | None
start_period_1_s: float | None
end_period_1_s: float | None
start_period_2_s: float | None
end_period_2_s: float | None
source: SourceRef
~~~
**Methods:** side_of_team, team_of_side.

#### RosterEntry — src/football_fatigue/ingestion/models.py:L128-L137

~~~python
match_id: int
team_id: int
roster_side: str
player_id: int
shirt_number: int
position_group: str | None
started: bool
nickname: str | None
source: SourceRef
~~~

#### Roster — src/football_fatigue/ingestion/models.py:L141-L151

~~~python
match_id: int
entries: tuple[RosterEntry, ...]
issues: tuple[str, ...]
~~~
**Methods:** index.

#### SmoothedCandidate — src/football_fatigue/ingestion/models.py:L155-L161
One source smoothed entry. List order carries no meaning and is never used to select a player.

~~~python
x_m: float | None
y_m: float | None
visibility: str | None
confidence: str | None
~~~

#### PlayerSample — src/football_fatigue/ingestion/models.py:L165-L181

~~~python
source_side: str
jersey_number: int | None
jersey_raw: Any
x_raw_m: float | None
y_raw_m: float | None
visibility: str | None
confidence: str | None
speed_raw: float | None
smoothed_status: SmoothedStatus
smoothed_candidates: tuple[SmoothedCandidate, ...]
x_smooth_m: float | None
y_smooth_m: float | None
team_id: int | None
player_id: int | None
attribution_status: AttributionStatus
quality_flags: frozenset[QualityFlag]
~~~

#### BallSample — src/football_fatigue/ingestion/models.py:L185-L194

~~~python
x_raw_m: float | None
y_raw_m: float | None
z_raw_m: float | None
visibility_raw: str | None
x_smooth_m: float | None
y_smooth_m: float | None
z_smooth_m: float | None
visibility_smooth: str | None
raw_count: int
~~~

#### TrackingFrame — src/football_fatigue/ingestion/models.py:L198-L217

~~~python
match_id: int
frame_num: int
video_time_ms_source: float
video_time_s: float
source_period: int | None
canonical_period: int | None
period_elapsed_s: float | None
game_clock_nominal_s: float | None
source_record_indices: tuple[int, ...]
source_record_count: int
players: tuple[PlayerSample, ...]
unmatched_smoothed: tuple[tuple[str, SmoothedCandidate, Any], ...]
ball: BallSample | None
frame_gap_before: int | None
game_event_id: int | None
possession_event_id: int | None
quality_flags: frozenset[QualityFlag]
source: SourceRef
repeat_conflict_fields: tuple[str, ...] = ()
~~~

#### EventRow — src/football_fatigue/ingestion/models.py:L221-L244

~~~python
event_row_id: str
match_id: int
source_version: EventVersion
source_row_index: int
game_event_id: int | None
possession_event_id: int | None
game_event_type: str | None
possession_event_type: str | None
source_period: int | None
canonical_period: int | None
event_time_s: float | None
start_time_s: float | None
end_time_s: float | None
duration_s: float | None
source_team_id: int | None
source_player_id: int | None
home_team: bool | None
sequence: float | None
attack_direction: str | None
attributes: dict[str, Any]
positions_snapshot: dict[str, Any]
quality_flags: frozenset[QualityFlag] = field(default_factory=frozenset)
source: SourceRef | None = None
~~~

#### EventVersion.folder — src/football_fatigue/ingestion/models.py:L28-L30

~~~python
def EventVersion.folder(self) -> str | None
~~~
**Declared purpose:** Inventory version key: None is the unlabelled Event Data root folder.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** No explicit calls; field/literal/attribute computation.
**Output expressions (selected excerpts):** L30: Return {EventVersion.ROOT: None, EventVersion.MARCH_14_2025: 'March 14, 2025', EventVersion.MAY_1_2025: 'May 1, 2025'}[self].
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### MatchMetadata.side_of_team — src/football_fatigue/ingestion/models.py:L120-L121

~~~python
def MatchMetadata.side_of_team(self, team_id: int) -> str | None
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/ingestion/loaders.py:_parse_roster@L283; src/football_fatigue/ingestion/loaders.py:check_identity@L103
**Callees visible in body:** No explicit calls; field/literal/attribute computation.
**Output expressions (selected excerpts):** L121: Return 'home' if team_id == self.home_team_id else 'away' if team_id == self.away_team_id else None.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### MatchMetadata.team_of_side — src/football_fatigue/ingestion/models.py:L123-L124

~~~python
def MatchMetadata.team_of_side(self, side: str) -> int
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** No explicit calls; field/literal/attribute computation.
**Output expressions (selected excerpts):** L124: Return self.home_team_id if side == 'home' else self.away_team_id.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### Roster.index — src/football_fatigue/ingestion/models.py:L146-L151

~~~python
def Roster.index(self) -> dict[tuple[str, int], RosterEntry]
~~~
**Declared purpose:** (roster_side, shirt_number) -> entry, omitting ambiguous (repeated) shirt numbers.
**Caller candidates:** src/football_fatigue/ingestion/loaders.py:iter_tracking_frames@L545
**Callees visible in body:** len; seen.items; seen.setdefault; seen.setdefault((e.roster_side, e.shirt_number), []).append
**Output expressions (selected excerpts):** L151: Return {k: v[0] for k, v in seen.items() if len(v) == 1}.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

### Symbols in src/football_fatigue/ingestion/loaders.py

#### Dataset — src/football_fatigue/ingestion/loaders.py:L110-L200
Match-ID based access to a raw dataset root (read-only; files are located through the project inventory).
**Methods:** __init__, _inv, match_ids, _one, _match, metadata, roster, load_events, iter_events, tracking_source, iter_tracking_frames.

#### validate_match_id — src/football_fatigue/ingestion/loaders.py:L43-L49

~~~python
def validate_match_id(match_id: object) -> str
~~~
**Declared purpose:** Canonical match id: 1-9 ASCII digits (the inventory's file-name format). Rejects separators and traversal.
**Caller candidates:** scripts/audit/audit_lib.py:cli_main@L486; scripts/audit/audit_lib.py:locate@L46; scripts/audit/audit_lib.py:run_matches@L406; scripts/audit/audit_lib.py:safe_destination@L389; scripts/audit/tracking_scan.py:main@L225; src/football_fatigue/ingestion/loaders.py:Dataset._match@L137; src/football_fatigue/ingestion/validation.py:quality_report@L113; src/football_fatigue/ingestion/validation.py:safe_output_path@L211; src/football_fatigue/match_data.py:MatchData.__init__@L260; src/football_fatigue/synchronization.py:synchronize_match@L344
**Callees visible in body:** IngestionError; _MATCH_ID.fullmatch; isinstance; str
**Output expressions (selected excerpts):** L49: Return match_id.
**Important source branches:** L45: isinstance(match_id, int) and (not isinstance(match_id, bool)); L47: not isinstance(match_id, str) or not _MATCH_ID.fullmatch(match_id).
**Explicit failure paths (excerpts):** L48: IngestionError(f'invalid match id {match_id!r}: expected 1-9 ASCII digits').

#### as_float — src/football_fatigue/ingestion/loaders.py:L54-L59

~~~python
def as_float(v: Any) -> float | None
~~~
**Declared purpose:** Finite number or None. Booleans, strings, NaN and infinities are not numbers here.
**Caller candidates:** src/football_fatigue/ingestion/loaders.py:_ball@L487; src/football_fatigue/ingestion/loaders.py:_ball@L490; src/football_fatigue/ingestion/loaders.py:_ball@L493; src/football_fatigue/ingestion/loaders.py:_build_frame@L507; src/football_fatigue/ingestion/loaders.py:_build_frame@L534; src/football_fatigue/ingestion/loaders.py:_candidate@L383; src/football_fatigue/ingestion/loaders.py:_event_row@L331; src/football_fatigue/ingestion/loaders.py:_event_row@L347; src/football_fatigue/ingestion/loaders.py:_event_row@L349; src/football_fatigue/ingestion/loaders.py:_parse_metadata@L220; src/football_fatigue/ingestion/loaders.py:_parse_metadata@L231; src/football_fatigue/ingestion/loaders.py:_parse_metadata@L257; src/football_fatigue/ingestion/loaders.py:_parse_metadata@L258; src/football_fatigue/ingestion/loaders.py:_parse_metadata@L259
**Callees visible in body:** float; isinstance; math.isfinite
**Output expressions (selected excerpts):** L59: Return f if math.isfinite(f) else None; L57: Return None.
**Important source branches:** L56: isinstance(v, bool) or not isinstance(v, (int, float)).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### as_int — src/football_fatigue/ingestion/loaders.py:L62-L72

~~~python
def as_int(v: Any) -> int | None
~~~
**Declared purpose:** Integer, integral float, or ASCII-digit string; otherwise None. Booleans are rejected.
**Caller candidates:** src/football_fatigue/ingestion/loaders.py:_build_frame@L506; src/football_fatigue/ingestion/loaders.py:_build_frame@L508; src/football_fatigue/ingestion/loaders.py:_build_frame@L537; src/football_fatigue/ingestion/loaders.py:_event_row@L328; src/football_fatigue/ingestion/loaders.py:_event_row@L334; src/football_fatigue/ingestion/loaders.py:_event_row@L344; src/football_fatigue/ingestion/loaders.py:_event_row@L348; src/football_fatigue/ingestion/loaders.py:_jersey@L379; src/football_fatigue/ingestion/loaders.py:_parse_metadata@L217; src/football_fatigue/ingestion/loaders.py:_parse_metadata@L223; src/football_fatigue/ingestion/loaders.py:_parse_metadata@L234; src/football_fatigue/ingestion/loaders.py:_parse_metadata@L245; src/football_fatigue/ingestion/loaders.py:_parse_metadata@L250; src/football_fatigue/ingestion/loaders.py:_parse_roster@L280
**Callees visible in body:** int; isinstance; math.isfinite; re.fullmatch; v.strip
**Output expressions (selected excerpts):** L72: Return None; L65: Return None; L67: Return v; L69: Return int(v) if math.isfinite(v) and v == int(v) else None.
**Important source branches:** L64: isinstance(v, bool); L66: isinstance(v, int); L68: isinstance(v, float); L70: isinstance(v, str) and re.fullmatch('-?[0-9]+', v.strip()).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### _req — src/football_fatigue/ingestion/loaders.py:L75-L81

~~~python
def _req(d: Any, path: str, what: str) -> Any
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/ingestion/loaders.py:_parse_metadata@L217; src/football_fatigue/ingestion/loaders.py:_parse_metadata@L220; src/football_fatigue/ingestion/loaders.py:_parse_metadata@L223; src/football_fatigue/ingestion/loaders.py:_parse_metadata@L226; src/football_fatigue/ingestion/loaders.py:_parse_metadata@L234; src/football_fatigue/ingestion/loaders.py:_parse_metadata@L241; src/football_fatigue/ingestion/loaders.py:_parse_metadata@L243; src/football_fatigue/ingestion/loaders.py:_parse_metadata@L244; src/football_fatigue/ingestion/loaders.py:_parse_roster@L280; src/football_fatigue/ingestion/loaders.py:_parse_roster@L281; src/football_fatigue/ingestion/loaders.py:_parse_roster@L282
**Callees visible in body:** IngestionError; isinstance; path.split
**Output expressions (selected excerpts):** L81: Return cur.
**Important source branches:** L78: not isinstance(cur, dict) or key not in cur or cur[key] is None.
**Explicit failure paths (excerpts):** L79: IngestionError(f"{what}: required field '{path}' is missing or null").

#### _read_json — src/football_fatigue/ingestion/loaders.py:L84-L89

~~~python
def _read_json(path: Path) -> Any
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/ingestion/loaders.py:Dataset.iter_events@L167; src/football_fatigue/ingestion/loaders.py:Dataset.metadata@L147; src/football_fatigue/ingestion/loaders.py:Dataset.roster@L154
**Callees visible in body:** IngestionError; json.load; open; type
**Output expressions (selected excerpts):** L87: Return json.load(f).
**Explicit failure paths (excerpts):** L89: IngestionError(f'{path.name}: cannot read as JSON ({type(e).__name__}: {e})').

#### check_identity — src/football_fatigue/ingestion/loaders.py:L92-L105

~~~python
def check_identity(match_id: int, meta: 'MatchMetadata | None', roster: 'Roster | None') -> None
~~~
**Declared purpose:** Reject metadata/rosters of another match and rosters whose teams disagree with the metadata home/away teams.
**Caller candidates:** src/football_fatigue/ingestion/loaders.py:Dataset.iter_tracking_frames@L198; src/football_fatigue/ingestion/loaders.py:Dataset.roster@L152; src/football_fatigue/ingestion/loaders.py:iter_tracking_frames@L546
**Callees visible in body:** IngestionError; meta.side_of_team
**Output expressions (selected excerpts):** L97: Return None.
**Important source branches:** L94: meta is not None and meta.match_id != match_id; L96: roster is None; L98: meta is None; L100: roster.match_id != match_id; L103: e.match_id != match_id or meta.side_of_team(e.team_id) != e.roster_side.
**Explicit failure paths (excerpts):** L95: IngestionError(f'metadata belongs to match {meta.match_id}, not the requested match {match_id}'); L99: IngestionError('a roster join needs the match metadata (home/away team ids)'); L101: IngestionError(f'roster belongs to match {roster.match_id}, not the requested match {match_id}'); L104: IngestionError(f'roster entry (team {e.team_id}, side {e.roster_side}, player {e.player_id}) is inconsistent with the home/away teams ({meta.home_team_id}/{meta.away_team_id}) of match {match_id}').

#### Dataset.__init__ — src/football_fatigue/ingestion/loaders.py:L113-L117

~~~python
def Dataset.__init__(self, root: str | Path)
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/inventory.py:InventoryScanError.__init__@L94
**Callees visible in body:** IngestionError; Path; self.root.is_dir
**Output:** no explicit return/yield; mutation, exception/command action, or implicit None. See signature and module context.
**Important source branches:** L115: not self.root.is_dir().
**Explicit failure paths (excerpts):** L116: IngestionError(f'dataset root is not a directory: {self.root}').

#### Dataset._inv — src/football_fatigue/ingestion/loaders.py:L120-L126

~~~python
def Dataset._inv(self) -> inv.Inventory
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/ingestion/loaders.py:Dataset._match@L138; src/football_fatigue/ingestion/loaders.py:Dataset.match_ids@L129
**Callees visible in body:** '; '.join; IngestionError; inv.scan
**Output expressions (selected excerpts):** L126: Return self._inventory.
**Important source branches:** L121: self._inventory is None.
**Explicit failure paths (excerpts):** L125: IngestionError(f"incomplete dataset scan: {'; '.join(e.errors)}").

#### Dataset.match_ids — src/football_fatigue/ingestion/loaders.py:L128-L129

~~~python
def Dataset.match_ids(self) -> list[str]
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** scripts/audit/audit_lib.py:cli_main@L486; src/football_fatigue/playback/server.py:get_matches_catalog@L74
**Callees visible in body:** self._inv
**Output expressions (selected excerpts):** L129: Return self._inv().match_ids.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### Dataset._one — src/football_fatigue/ingestion/loaders.py:L131-L134

~~~python
def Dataset._one(self, match_id: str, paths: list[str], what: str) -> tuple[Path, str]
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/ingestion/loaders.py:Dataset.iter_events@L166; src/football_fatigue/ingestion/loaders.py:Dataset.metadata@L146; src/football_fatigue/ingestion/loaders.py:Dataset.roster@L153; src/football_fatigue/ingestion/loaders.py:Dataset.tracking_source@L172
**Callees visible in body:** IngestionError; len
**Output expressions (selected excerpts):** L134: Return (self.root / paths[0], paths[0]).
**Important source branches:** L132: len(paths) != 1.
**Explicit failure paths (excerpts):** L133: IngestionError(f'match {match_id}: {len(paths)} {what} file(s) found, expected exactly 1: {paths}').

#### Dataset._match — src/football_fatigue/ingestion/loaders.py:L136-L141

~~~python
def Dataset._match(self, match_id: object) -> tuple[str, inv.MatchInventory]
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/ingestion/loaders.py:Dataset.iter_events@L164; src/football_fatigue/ingestion/loaders.py:Dataset.iter_tracking_frames@L193; src/football_fatigue/ingestion/loaders.py:Dataset.metadata@L145; src/football_fatigue/ingestion/loaders.py:Dataset.roster@L150; src/football_fatigue/ingestion/loaders.py:Dataset.tracking_source@L171
**Callees visible in body:** IngestionError; self._inv; self._inv().matches.get; validate_match_id
**Output expressions (selected excerpts):** L141: Return (mid, mi).
**Important source branches:** L139: mi is None.
**Explicit failure paths (excerpts):** L140: IngestionError(f'match {mid}: not found under {self.root}').

#### Dataset.metadata — src/football_fatigue/ingestion/loaders.py:L144-L147

~~~python
def Dataset.metadata(self, match_id: object) -> MatchMetadata
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/ingestion/loaders.py:Dataset.iter_tracking_frames@L197; src/football_fatigue/ingestion/loaders.py:Dataset.roster@L151; src/football_fatigue/ingestion/validation.py:quality_report@L121; src/football_fatigue/match_data.py:MatchData.metadata@L281; src/football_fatigue/playback/server.py:get_matches_catalog@L76; src/football_fatigue/synchronization.py:synchronize_match@L345
**Callees visible in body:** SourceRef; _read_json; parse_metadata; self._match; self._one
**Output expressions (selected excerpts):** L147: Return parse_metadata(_read_json(path), mid, SourceRef(rel, 'metadata')).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### Dataset.roster — src/football_fatigue/ingestion/loaders.py:L149-L154

~~~python
def Dataset.roster(self, match_id: object, metadata: MatchMetadata | None=None) -> Roster
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/ingestion/validation.py:quality_report@L123; src/football_fatigue/match_data.py:MatchData.roster@L287; src/football_fatigue/synchronization.py:synchronize_match@L348
**Callees visible in body:** SourceRef; _read_json; check_identity; int; parse_roster; self._match; self._one; self.metadata
**Output expressions (selected excerpts):** L154: Return parse_roster(_read_json(path), meta, SourceRef(rel, 'roster')).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### Dataset.load_events — src/football_fatigue/ingestion/loaders.py:L157-L158

~~~python
def Dataset.load_events(self, match_id: object, version: EventVersion) -> list[EventRow]
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/ingestion/validation.py:quality_report@L131; src/football_fatigue/match_data.py:MatchData.events@L293; src/football_fatigue/synchronization.py:synchronize_match@L346
**Callees visible in body:** list; self.iter_events
**Output expressions (selected excerpts):** L158: Return list(self.iter_events(match_id, version)).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### Dataset.iter_events — src/football_fatigue/ingestion/loaders.py:L160-L167

~~~python
def Dataset.iter_events(self, match_id: object, version: EventVersion) -> Iterator[EventRow]
~~~
**Declared purpose:** Rows of exactly the requested version, in source order, none dropped or merged.
**Caller candidates:** src/football_fatigue/ingestion/loaders.py:Dataset.load_events@L158
**Callees visible in body:** IngestionError; _read_json; isinstance; mi.events.get; parse_events; self._match; self._one
**Output expressions (selected excerpts):** L167: YieldFrom parse_events(_read_json(path), mid, version, rel).
**Important source branches:** L162: not isinstance(version, EventVersion).
**Explicit failure paths (excerpts):** L163: IngestionError('an EventVersion must be named explicitly (root, march_14_2025, may_1_2025)').

#### Dataset.tracking_source — src/football_fatigue/ingestion/loaders.py:L170-L173

~~~python
def Dataset.tracking_source(self, match_id: object) -> tuple[Path, SourceRef]
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/ingestion/loaders.py:Dataset.iter_tracking_frames@L194; src/football_fatigue/ingestion/validation.py:quality_report@L148
**Callees visible in body:** SourceRef; self._match; self._one
**Output expressions (selected excerpts):** L173: Return (path, SourceRef(rel, 'tracking')).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### Dataset.iter_tracking_frames — src/football_fatigue/ingestion/loaders.py:L175-L200

~~~python
def Dataset.iter_tracking_frames(self, match_id: object, *, roster: Roster | None=None, metadata: MatchMetadata | None=None, collapse_repeats: bool=True, quarantine_periods: Iterable[int]=(), max_frames: int | None=None) -> Iterator[TrackingFrame]
~~~
**Declared purpose:** Stream tracking frames with bounded memory (one frame plus its repeats' comparison state).
**Additional declaration:** Records with the same consecutive ``frameNum`` collapse into one frame that keeps every source record index (``collapse_repeats=False`` yields one frame per record instead). ``quarantine_periods`` names periods whose team/player attribution is to be withheld (for example periods flagged by an external side-identity review); nothing is swapped. A roster enables the (side, shirt number) join; the jersey number alone is never used.
**Caller candidates:** src/football_fatigue/ingestion/validation.py:quality_report@L150; src/football_fatigue/match_data.py:MatchData.frames@L329; src/football_fatigue/synchronization.py:synchronize_match@L349
**Callees visible in body:** check_identity; frozenset; int; iter_tracking_frames; self._match; self.metadata; self.tracking_source
**Output expressions (selected excerpts):** L199: Return iter_tracking_frames(path, src, int(mid), roster=roster, metadata=meta, collapse_repeats=collapse_repeats, quarantine_periods=frozenset(quarantine_periods), max_frames=max_frames).
**Important source branches:** L196: roster is not None and meta is None.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### parse_metadata — src/football_fatigue/ingestion/loaders.py:L205-L209

~~~python
def parse_metadata(doc: Any, match_id: str, source: SourceRef) -> MatchMetadata
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/ingestion/loaders.py:Dataset.metadata@L147
**Callees visible in body:** IngestionError; _parse_metadata; type
**Output expressions (selected excerpts):** L207: Return _parse_metadata(doc, match_id, source).
**Explicit failure paths (excerpts):** L209: IngestionError(f'metadata {match_id}: malformed structure ({type(e).__name__}: {e})').

#### _parse_metadata — src/football_fatigue/ingestion/loaders.py:L212-L262

~~~python
def _parse_metadata(doc: Any, match_id: str, source: SourceRef) -> MatchMetadata
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/ingestion/loaders.py:parse_metadata@L207
**Callees visible in body:** (m['awayTeam'] or {}).get; (m['homeTeam'] or {}).get; IngestionError; MatchMetadata; _req; as_float; as_int; int; isinstance; len; m.get; m['stadium'].get; pitches[0].get; str; type
**Output expressions (selected excerpts):** L239: Return MatchMetadata(match_id=mid, kickoff_date=str(_req(m, 'date', what)), competition_id=comp, competition_name=str(_req(m, 'competition.name', what)), season=str(_req(m, 'season', what)), matchweek=as_int(m.get('week')), home_team_id=home, away_team_id=away, home_team_name=str((m['homeTeam'] or {}).get('name')), away_team_name=str((m['awayTeam'] or {}).get('n....
**Important source branches:** L214: not isinstance(m, dict); L218: mid != int(match_id); L221: fps is None or fps <= 0; L224: home is None or away is None or home == away; L227: not isinstance(pitches, list) or len(pitches) != 1; L229: not isinstance(pitches[0], dict); L232: length is None or width is None or length <= 0 or (width <= 0).
**Explicit failure paths (excerpts):** L215: IngestionError(f'metadata {match_id}: expected one JSON object (or a one-element list)'); L219: IngestionError(f"{what}: id {m['id']!r} does not match the requested match id"); L222: IngestionError(f'{what}: fps must be a positive finite number'); L225: IngestionError(f'{what}: home/away team ids must be two distinct integers').

#### parse_roster — src/football_fatigue/ingestion/loaders.py:L265-L269

~~~python
def parse_roster(doc: Any, meta: MatchMetadata, source: SourceRef) -> Roster
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/ingestion/loaders.py:Dataset.roster@L154
**Callees visible in body:** IngestionError; _parse_roster; type
**Output expressions (selected excerpts):** L267: Return _parse_roster(doc, meta, source).
**Explicit failure paths (excerpts):** L269: IngestionError(f'roster {meta.match_id}: malformed structure ({type(e).__name__}: {e})').

#### _parse_roster — src/football_fatigue/ingestion/loaders.py:L272-L293

~~~python
def _parse_roster(doc: Any, meta: MatchMetadata, source: SourceRef) -> Roster
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/ingestion/loaders.py:parse_roster@L267
**Callees visible in body:** (e.get('player') or {}).get; Counter; IngestionError; Roster; RosterEntry; SourceRef; _req; as_int; bool; counts.items; e.get; entries.append; enumerate; isinstance; meta.side_of_team; pcount.items; sorted; tuple
**Output expressions (selected excerpts):** L293: Return Roster(meta.match_id, tuple(entries), tuple(issues)).
**Important source branches:** L273: not isinstance(doc, list); L278: not isinstance(e, dict); L284: side is None or pid is None or shirt is None.
**Explicit failure paths (excerpts):** L274: IngestionError(f'roster {meta.match_id}: expected a JSON list'); L279: IngestionError(f'{what}: entry {i} is not an object'); L285: IngestionError(f'{what}: entry {i} has a team that is not home/away of the match, or a non-integer id/shirt number').

#### _canonical — src/football_fatigue/ingestion/loaders.py:L298-L299

~~~python
def _canonical(period: int | None) -> int | None
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/ingestion/loaders.py:_build_frame@L518; src/football_fatigue/ingestion/loaders.py:_build_frame@L533; src/football_fatigue/ingestion/loaders.py:_event_row@L329; src/football_fatigue/ingestion/loaders.py:_event_row@L346
**Callees visible in body:** No explicit calls; field/literal/attribute computation.
**Output expressions (selected excerpts):** L299: Return period if period in CANONICAL_PERIODS else None.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### parse_events — src/football_fatigue/ingestion/loaders.py:L302-L321

~~~python
def parse_events(doc: Any, match_id: str, version: EventVersion, rel_path: str) -> Iterator[EventRow]
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/ingestion/loaders.py:Dataset.iter_events@L167
**Callees visible in body:** Counter; IngestionError; _event_row; as_int; enumerate; int; isinstance; r.get; type
**Output expressions (selected excerpts):** L319: Yield _event_row(r, i, match_id, version, rel_path, pair_counts).
**Important source branches:** L303: not isinstance(doc, list); L308: isinstance(r, dict); L314: not isinstance(r, dict); L316: r.get('gameId') is not None and as_int(r.get('gameId')) != int(match_id); L310: not isinstance(r.get(k), (type(None), int, float, str)).
**Explicit failure paths (excerpts):** L304: IngestionError(f'events {match_id} [{version.value}]: expected a JSON list of rows'); L315: IngestionError(f'{what}: row {i} is not an object'); L317: IngestionError(f"{what}: row {i}: gameId {r.get('gameId')!r} does not belong to match {match_id}"); L321: IngestionError(f'{what}: row {i}: malformed structure ({type(e).__name__}: {e})').

#### _event_row — src/football_fatigue/ingestion/loaders.py:L324-L353

~~~python
def _event_row(r: dict, i: int, match_id: str, version: EventVersion, rel_path: str, pair_counts: Counter) -> EventRow
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/ingestion/loaders.py:parse_events@L319
**Callees visible in body:** (r.get('stadiumMetadata') or {}).get; EventRow; SourceRef; _canonical; as_float; as_int; flags.add; frozenset; ge.get; int; isinstance; pe.get; r.get; set
**Output expressions (selected excerpts):** L342: Return EventRow(event_row_id=f'{match_id}:{version.value}:{i}', match_id=int(match_id), source_version=version, source_row_index=i, game_event_id=gid, possession_event_id=as_int(r.get('possessionEventId')), game_event_type=ge.get('gameEventType'), possession_event_type=pe.get('possessionEventType'), source_period=src_period, canonical_period=_canonical(src_perio....
**Important source branches:** L329: _canonical(src_period) is None; L332: t is None; L335: gid is None; L337: pair_counts[r.get('gameEventId'), r.get('possessionEventId')] > 1.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### _iter_records — src/football_fatigue/ingestion/loaders.py:L358-L375

~~~python
def _iter_records(path: Path) -> Iterator[tuple[int, dict]]
~~~
**Declared purpose:** Stream (record_index, object) from a .jsonl.bz2 file; the index counts non-blank records from 0.
**Caller candidates:** src/football_fatigue/ingestion/loaders.py:iter_tracking_frames@L568
**Callees visible in body:** IngestionError; bz2.open; isinstance; json.loads; line.strip; type
**Output expressions (selected excerpts):** L372: Yield (idx, rec).
**Important source branches:** L364: not line.strip(); L370: not isinstance(rec, dict).
**Explicit failure paths (excerpts):** L375: IngestionError(f'{path.name}: cannot read compressed stream ({type(e).__name__}: {e})'); L371: IngestionError(f'{path.name}: record {idx} is not a JSON object'); L369: IngestionError(f'{path.name}: record {idx} is not valid JSON ({e})').

#### _jersey — src/football_fatigue/ingestion/loaders.py:L378-L379

~~~python
def _jersey(v: Any) -> int | None
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/ingestion/loaders.py:_players@L410; src/football_fatigue/ingestion/loaders.py:_players@L416; src/football_fatigue/ingestion/loaders.py:_players@L424
**Callees visible in body:** as_int
**Output expressions (selected excerpts):** L379: Return as_int(v).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### _candidate — src/football_fatigue/ingestion/loaders.py:L382-L383

~~~python
def _candidate(e: dict) -> SmoothedCandidate
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/ingestion/loaders.py:_players@L410
**Callees visible in body:** SmoothedCandidate; as_float; e.get
**Output expressions (selected excerpts):** L383: Return SmoothedCandidate(as_float(e.get('x')), as_float(e.get('y')), e.get('visibility'), e.get('confidence')).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### _smoothed_status — src/football_fatigue/ingestion/loaders.py:L386-L393

~~~python
def _smoothed_status(cands: list[SmoothedCandidate]) -> SmoothedStatus
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/ingestion/loaders.py:_players@L435
**Callees visible in body:** all; len
**Output expressions (selected excerpts):** L393: Return SmoothedStatus.DUPLICATE_IDENTICAL if same else SmoothedStatus.AMBIGUOUS_CONFLICTING; L388: Return SmoothedStatus.MISSING; L390: Return SmoothedStatus.UNIQUE.
**Important source branches:** L387: not cands; L389: len(cands) == 1.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### _players — src/football_fatigue/ingestion/loaders.py:L396-L471

~~~python
def _players(rec: dict, ctx: dict) -> tuple[list[PlayerSample], list[tuple[str, SmoothedCandidate, Any]], set[Q]]
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/ingestion/loaders.py:_build_frame@L521
**Callees visible in body:** Counter; PlayerSample; SmoothedCandidate; _candidate; _jersey; _smoothed_status; any; as_float; e.get; fflags.add; flags.add; frozenset; groups.get; groups.items; groups.setdefault; groups.setdefault(_jersey(e.get('jerseyNum')), []).append; isinstance; raw_jerseys.items; rec.get; roster_index.get; samples.append; set; tuple; unmatched.append; unmatched.extend
**Output expressions (selected excerpts):** L471: Return (samples, unmatched, fflags).
**Important source branches:** L403: not isinstance(raw, list); L407: isinstance(sm, list); L417: any((n > 1 and j is not None for j, n in raw_jerseys.items())); L420: not isinstance(e, dict); L425: j is None; L428: x is None or y is None; L430: e.get('visibility') == 'ESTIMATED'.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### _ball — src/football_fatigue/ingestion/loaders.py:L474-L494

~~~python
def _ball(rec: dict, flags: set[Q]) -> BallSample | None
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/ingestion/loaders.py:_build_frame@L523
**Callees visible in body:** BallSample; as_float; b.get; flags.add; isinstance; len; rec.get; s.get
**Output expressions (selected excerpts):** L493: Return BallSample(rx, ry, rz, b.get('visibility') if raw else None, sx, sy, as_float(s.get('z')), s.get('visibility') if sm is not None else None, len(raw)); L480: Return None.
**Important source branches:** L478: not raw and sm is None; L481: not raw; L483: len(raw) > 1; L488: sm is not None and (sx is None or sy is None); L491: raw and (rx is None or ry is None).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### _build_frame — src/football_fatigue/ingestion/loaders.py:L504-L540

~~~python
def _build_frame(match_id: int, src: SourceRef, rec: dict, indices: list[int], conflicts: set[str], gap: int | None, nonmono: bool, ctx_base: dict) -> TrackingFrame
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/ingestion/loaders.py:iter_tracking_frames.build@L563; src/football_fatigue/ingestion/loaders.py:iter_tracking_frames@L563
**Callees visible in body:** SourceRef; TrackingFrame; _ball; _canonical; _players; as_float; as_int; dict; flags.add; flags.discard; frozenset; len; rec.get; replace; set; sorted; tuple
**Output expressions (selected excerpts):** L531: Return TrackingFrame(match_id=match_id, frame_num=fn, video_time_ms_source=vt_ms, video_time_s=vt_ms / 1000.0, source_period=src_period, canonical_period=_canonical(src_period), period_elapsed_s=as_float(rec.get('periodElapsedTime')), game_clock_nominal_s=as_float(rec.get('periodGameClockTime')), source_record_indices=tuple(indices), source_record_count=len(indi....
**Important source branches:** L510: len(indices) > 1; L512: conflicts; L514: nonmono; L516: gap; L518: _canonical(src_period) is None; L524: 'ballsSmoothed' in conflicts; L529: ball is not None.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### iter_tracking_frames — src/football_fatigue/ingestion/loaders.py:L543-L607

~~~python
def iter_tracking_frames(path: Path, src: SourceRef, match_id: int, *, roster: Roster | None, metadata: MatchMetadata | None, collapse_repeats: bool, quarantine_periods: frozenset[int], max_frames: int | None=None) -> Iterator[TrackingFrame]
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/ingestion/loaders.py:Dataset.iter_tracking_frames@L199; src/football_fatigue/ingestion/validation.py:quality_report@L150; src/football_fatigue/match_data.py:MatchData.frames@L329; src/football_fatigue/synchronization.py:synchronize_match@L349
**Callees visible in body:** ', '.join; IngestionError; _build_frame; _iter_records; as_float; as_int; build; check; check_identity; cur['conflicts'].update; cur['indices'].append; dict; first.get; next; rec.get; roster.index; set; type
**Output expressions (selected excerpts):** L563: Return _build_frame(match_id, src, c['rec'], c['indices'], c['conflicts'], c['gap'], c['nonmono'], ctx); L605: Yield build(cur); L590: Yield build(cur); L593: Return None.
**Important source branches:** L604: cur is not None and (not (max_frames is not None and emitted >= max_frames)); L606: pending is not None; L553: as_int(rec.get('frameNum')) is None; L555: as_float(rec.get('videoTimeMs')) is None; L558: ref is not None and as_int(ref) != match_id; L579: collapse_repeats and cur is not None and (fn == cur['fn']); L589: cur is not None.
**Explicit failure paths (excerpts):** L607: pending; L554: IngestionError(f"{path.name}: record {idx}: required field 'frameNum' is missing or not an integer"); L556: IngestionError(f"{path.name}: record {idx}: required field 'videoTimeMs' is missing or not a finite number"); L559: IngestionError(f'{path.name}: record {idx}: gameRefId {ref!r} does not belong to match {match_id}').

#### iter_tracking_frames.check — src/football_fatigue/ingestion/loaders.py:L552-L559

~~~python
def iter_tracking_frames.check(idx: int, rec: dict) -> None
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/ingestion/loaders.py:iter_tracking_frames@L572
**Callees visible in body:** IngestionError; as_float; as_int; rec.get
**Output:** no explicit return/yield; mutation, exception/command action, or implicit None. See signature and module context.
**Important source branches:** L553: as_int(rec.get('frameNum')) is None; L555: as_float(rec.get('videoTimeMs')) is None; L558: ref is not None and as_int(ref) != match_id.
**Explicit failure paths (excerpts):** L554: IngestionError(f"{path.name}: record {idx}: required field 'frameNum' is missing or not an integer"); L556: IngestionError(f"{path.name}: record {idx}: required field 'videoTimeMs' is missing or not a finite number"); L559: IngestionError(f'{path.name}: record {idx}: gameRefId {ref!r} does not belong to match {match_id}').

#### iter_tracking_frames.build — src/football_fatigue/ingestion/loaders.py:L561-L565

~~~python
def iter_tracking_frames.build(c: dict) -> TrackingFrame
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/ingestion/loaders.py:iter_tracking_frames@L590; src/football_fatigue/ingestion/loaders.py:iter_tracking_frames@L605
**Callees visible in body:** IngestionError; _build_frame; type
**Output expressions (selected excerpts):** L563: Return _build_frame(match_id, src, c['rec'], c['indices'], c['conflicts'], c['gap'], c['nonmono'], ctx).
**Explicit failure paths (excerpts):** L565: IngestionError(f"{path.name}: frame {c['fn']} (records {c['indices'][0]}..{c['indices'][-1]}): malformed structure ({type(e).__name__}: {e})").

### Symbols in src/football_fatigue/ingestion/validation.py

#### TrackingQuality — src/football_fatigue/ingestion/validation.py:L25-L64
Streaming accumulator: constant memory regardless of match length.

~~~python
frames: int = 0
source_records: int = 0
first_video_time_s: float | None = None
last_video_time_s: float | None = None
player_samples: int = 0
frame_flags: Counter = field(default_factory=Counter)
sample_flags: Counter = field(default_factory=Counter)
smoothed_status: Counter = field(default_factory=Counter)
attribution_status: Counter = field(default_factory=Counter)
frames_by_source_period: Counter = field(default_factory=Counter)
frames_by_canonical_period: Counter = field(default_factory=Counter)
frames_without_ball: int = 0
frames_with_ball_smoothed_xy_null: int = 0
frames_with_ball_smoothed_unresolved: int = 0
unmatched_smoothed_entries: int = 0
~~~
**Methods:** add.

#### MatchQualityReport — src/football_fatigue/ingestion/validation.py:L68-L107

~~~python
match_id: str
event_version: str
fatal_errors: list[str] = field(default_factory=list)
sources: dict[str, str] = field(default_factory=dict)
tracking: dict[str, Any] = field(default_factory=dict)
events: dict[str, Any] = field(default_factory=dict)
periods: dict[str, Any] = field(default_factory=dict)
roster_issues: list[str] = field(default_factory=list)
scope: dict[str, Any] = field(default_factory=dict)
~~~
**Methods:** status, to_dict, summary_text.

#### TrackingQuality.add — src/football_fatigue/ingestion/validation.py:L44-L64

~~~python
def TrackingQuality.add(self, f: TrackingFrame) -> None
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** scripts/audit/audit_lib.py:match_rows@L191; scripts/audit/extra_time_identity.py:run_match@L85; src/football_fatigue/ingestion/loaders.py:_ball@L479; src/football_fatigue/ingestion/loaders.py:_ball@L482; src/football_fatigue/ingestion/loaders.py:_ball@L484; src/football_fatigue/ingestion/loaders.py:_ball@L489; src/football_fatigue/ingestion/loaders.py:_ball@L492; src/football_fatigue/ingestion/loaders.py:_build_frame@L511; src/football_fatigue/ingestion/loaders.py:_build_frame@L513; src/football_fatigue/ingestion/loaders.py:_build_frame@L515; src/football_fatigue/ingestion/loaders.py:_build_frame@L517; src/football_fatigue/ingestion/loaders.py:_build_frame@L519; src/football_fatigue/ingestion/loaders.py:_build_frame@L528; src/football_fatigue/ingestion/loaders.py:_event_row@L330
**Callees visible in body:** len; str
**Output:** no explicit return/yield; mutation, exception/command action, or implicit None. See signature and module context.
**Important source branches:** L53: f.canonical_period is not None.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### MatchQualityReport.status — src/football_fatigue/ingestion/validation.py:L80-L85

~~~python
def MatchQualityReport.status(self) -> str
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** any; self.events.get; self.events.get('flag_counts', {}).values; self.periods.get; self.periods.get('flag_counts', {}).values; self.tracking.get; self.tracking.get('frame_flag_counts', {}).values; self.tracking.get('sample_flag_counts', {}).values
**Output expressions (selected excerpts):** L85: Return 'ok_with_quality_issues' if any(counts) or self.roster_issues else 'ok'; L82: Return 'fatal'.
**Important source branches:** L81: self.fatal_errors.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### MatchQualityReport.to_dict — src/football_fatigue/ingestion/validation.py:L87-L90

~~~python
def MatchQualityReport.to_dict(self) -> dict[str, Any]
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/ingestion/validation.py:write_quality_report@L234; src/football_fatigue/inventory.py:main@L473
**Callees visible in body:** dict
**Output expressions (selected excerpts):** L88: Return dict(schema_version=REPORT_SCHEMA_VERSION, match_id=self.match_id, event_version=self.event_version, status=self.status, fatal_errors=self.fatal_errors, scope=self.scope, sources=self.sources, tracking=self.tracking, events=self.events, periods=self.periods, roster_issues=self.roster_issues).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### MatchQualityReport.summary_text — src/football_fatigue/ingestion/validation.py:L92-L107

~~~python
def MatchQualityReport.summary_text(self) -> str
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/ingestion/__main__.py:main@L33
**Callees visible in body:** '\n'.join; lines.append; self.scope.get; t['frame_flag_counts'].get; t['smoothed_status'].get
**Output expressions (selected excerpts):** L107: Return '\n'.join(lines).
**Important source branches:** L96: t; L100: e; L102: p; L105: self.scope.get('max_frames') is not None.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### quality_report — src/football_fatigue/ingestion/validation.py:L110-L175

~~~python
def quality_report(dataset: Dataset, match_id: object, version: EventVersion, *, max_frames: int | None=None, quarantine_periods: Iterable[int]=()) -> MatchQualityReport
~~~
**Declared purpose:** Stream the match's tracking once and load the requested event version; never raises for source problems.
**Caller candidates:** src/football_fatigue/ingestion/__main__.py:main@L32
**Callees visible in body:** Counter; IngestionError; MatchQualityReport; TrackingQuality; dataset.iter_tracking_frames; dataset.load_events; dataset.metadata; dataset.roster; dataset.tracking_source; dict; flag_counts.get; int; isinstance; len; list; q.add; rep.fatal_errors.append; set; sorted; str; validate_match_id
**Output expressions (selected excerpts):** L175: Return rep.
**Important source branches:** L114: not isinstance(version, EventVersion).
**Explicit failure paths (excerpts):** L115: IngestionError('an EventVersion must be named explicitly').

#### _is_link — src/football_fatigue/ingestion/validation.py:L180-L181

~~~python
def _is_link(p: Path) -> bool
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/ingestion/validation.py:_assert_no_links@L203
**Callees visible in body:** hasattr; os.path.isjunction; p.is_symlink
**Output expressions (selected excerpts):** L181: Return p.is_symlink() or (hasattr(os.path, 'isjunction') and os.path.isjunction(p)).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### _under — src/football_fatigue/ingestion/validation.py:L184-L185

~~~python
def _under(path: Path, parent: Path) -> bool
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/ingestion/validation.py:safe_output_path@L216
**Callees visible in body:** No explicit calls; field/literal/attribute computation.
**Output expressions (selected excerpts):** L185: Return path == parent or parent in path.parents.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### approved_output_root — src/football_fatigue/ingestion/validation.py:L188-L194

~~~python
def approved_output_root(raw_root: str | Path) -> Path
~~~
**Declared purpose:** Trusted boundary, taken lexically: ``<resolved dataset root's parent>/processed``.
**Caller candidates:** src/football_fatigue/ingestion/validation.py:safe_output_path@L214
**Callees visible in body:** Path; Path(raw_root).resolve; os.path.abspath
**Output expressions (selected excerpts):** L194: Return Path(os.path.abspath(Path(raw_root).resolve().parent / 'processed')).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### _assert_no_links — src/football_fatigue/ingestion/validation.py:L197-L206

~~~python
def _assert_no_links(anchor: Path, target: Path) -> None
~~~
**Declared purpose:** Every existing component from ``anchor`` (a resolved directory) down to ``target`` must be a real path: not a symlink or junction, and resolving to itself.
**Caller candidates:** src/football_fatigue/ingestion/validation.py:safe_output_path@L219
**Callees visible in body:** IngestionError; _is_link; cur.exists; cur.resolve; target.relative_to
**Output:** no explicit return/yield; mutation, exception/command action, or implicit None. See signature and module context.
**Important source branches:** L203: _is_link(cur); L205: cur.exists() and cur.resolve() != cur.
**Explicit failure paths (excerpts):** L204: IngestionError(f'output path component is a link or junction: {cur}'); L206: IngestionError(f'output path component is redirected (resolves to {cur.resolve()}): {cur}').

#### safe_output_path — src/football_fatigue/ingestion/validation.py:L209-L220

~~~python
def safe_output_path(raw_root: str | Path, match_id: object, version: EventVersion, out_root: str | Path | None=None, suffix: str='.json') -> Path
~~~
**Declared purpose:** Validated destination for one report (or its temporary file, ``suffix=".json.tmp"``), checked immediately before use.
**Caller candidates:** src/football_fatigue/ingestion/validation.py:write_quality_report@L226; src/football_fatigue/ingestion/validation.py:write_quality_report@L227; src/football_fatigue/ingestion/validation.py:write_quality_report@L229; src/football_fatigue/ingestion/validation.py:write_quality_report@L230; src/football_fatigue/ingestion/validation.py:write_quality_report@L235; src/football_fatigue/ingestion/validation.py:write_quality_report@L236
**Callees visible in body:** IngestionError; Path; Path(raw_root).resolve; _assert_no_links; _under; approved_output_root; os.path.abspath; validate_match_id
**Output expressions (selected excerpts):** L220: Return dest.
**Important source branches:** L216: _under(base, raw) or not _under(base, approved).
**Explicit failure paths (excerpts):** L217: IngestionError(f'output location {base} is not inside the approved output root {approved} (or lies inside the raw data)').

#### write_quality_report — src/football_fatigue/ingestion/validation.py:L223-L238

~~~python
def write_quality_report(report: MatchQualityReport, raw_root: str | Path, out_root: str | Path | None=None, *, overwrite: bool=False) -> Path
~~~
**Declared purpose:** Write the report JSON under the approved output root. Existing reports are never replaced silently.
**Caller candidates:** src/football_fatigue/ingestion/__main__.py:main@L35
**Callees visible in body:** EventVersion; IngestionError; dest.exists; dest.parent.mkdir; json.dump; open; os.replace; report.to_dict; safe_output_path
**Output expressions (selected excerpts):** L238: Return dest.
**Important source branches:** L231: dest.exists() and (not overwrite).
**Explicit failure paths (excerpts):** L232: IngestionError(f'report already exists (use overwrite=True): {dest}').

### Symbols in src/football_fatigue/synchronization.py

#### SynchronizationError — src/football_fatigue/synchronization.py:L47-L48
The inputs cannot be aligned under the policy (for example frames that are not in increasing time order).

#### _ContextError — src/football_fatigue/synchronization.py:L51-L52
Inputs of another match or version: always raised to the caller, never absorbed as a failed scan.

#### MatchStatus — src/football_fatigue/synchronization.py:L55-L57

~~~python
MATCHED = 'matched'
UNMATCHED = 'unmatched'
~~~

#### UnmatchedReason — src/football_fatigue/synchronization.py:L60-L67

~~~python
MISSING_EVENT_TIME = 'missing_event_time'
INVALID_SOURCE_PERIOD = 'invalid_source_period'
NO_TRACKING_FOR_PERIOD = 'no_tracking_for_period'
OUTSIDE_PERIOD_FRAME_RANGE = 'outside_period_frame_range'
NO_FRAME_WITHIN_TOLERANCE = 'no_frame_within_tolerance'
PERIOD_LABEL_DIFFERS_FROM_NEAREST_FRAME = 'period_label_differs_from_nearest_frame'
TRACKING_SCAN_INCOMPLETE = 'tracking_scan_incomplete'
~~~

#### SpatialStatus — src/football_fatigue/synchronization.py:L70-L74

~~~python
NOT_ASSESSED = 'not_assessed'
COORDINATE_AGREEING = 'coordinate_agreeing'
COORDINATE_DISAGREEING = 'coordinate_disagreeing'
NOT_ASSESSABLE = 'not_assessable'
~~~

#### TemporalNote — src/football_fatigue/synchronization.py:L77-L78

~~~python
NEAR_PERIOD_BOUNDARY = 'NEAR_PERIOD_BOUNDARY'
~~~

#### SpatialAssessment — src/football_fatigue/synchronization.py:L82-L89

~~~python
status: SpatialStatus = SpatialStatus.NOT_ASSESSED
reason: str | None = None
players_compared: int = 0
players_within_tolerance: int = 0
ball: str = 'not_assessed'
tolerance_m: float = SPATIAL_NEAR_M
players_excluded_ambiguous: int = 0
~~~

#### AlignmentResult — src/football_fatigue/synchronization.py:L93-L130

~~~python
event_row_id: str
match_id: int
event_version: EventVersion
source_row_index: int
event_source: SourceRef | None
event_time_s: float | None
source_period: int | None
canonical_period: int | None
event_quality_flags: frozenset[QualityFlag]
status: MatchStatus
reason: UnmatchedReason | None
residual_s: float | None
tolerance_s: float
frame_num: int | None = None
frame_video_time_s: float | None = None
frame_period: int | None = None
frame_source: SourceRef | None = None
frame_source_record_indices: tuple[int, ...] = ()
frame_quality_flags: frozenset[QualityFlag] = frozenset()
temporal_notes: frozenset[TemporalNote] = frozenset()
spatial: SpatialAssessment = field(default_factory=SpatialAssessment)
validated_correspondence: str = VALIDATED_CORRESPONDENCE
policy_version: str = POLICY_VERSION
~~~
**Methods:** quality_label.

#### MatchSynchronization — src/football_fatigue/synchronization.py:L134-L159

~~~python
match_id: int
event_version: EventVersion
fps: float
tolerance_s: float
alignments: tuple[AlignmentResult, ...]
scan_complete: bool
fatal_error: str | None
tracking_period_ranges: dict[int, tuple[float, float, int]]
spatial_assessed: bool
policy_version: str = POLICY_VERSION
ordering_violation: bool = False
~~~
**Methods:** summary.

#### AlignmentResult.quality_label — src/football_fatigue/synchronization.py:L122-L130

~~~python
def AlignmentResult.quality_label(self) -> str
~~~
**Declared purpose:** Strongest claim supported here: unmatched < temporal_only < coordinate_agreeing (never 'validated').
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** No explicit calls; field/literal/attribute computation.
**Output expressions (selected excerpts):** L130: Return 'temporal_only'; L125: Return 'unmatched'; L127: Return 'coordinate_agreeing'; L129: Return 'temporal_only_coordinates_disagree'.
**Important source branches:** L124: self.status is MatchStatus.UNMATCHED; L126: self.spatial.status is SpatialStatus.COORDINATE_AGREEING; L128: self.spatial.status is SpatialStatus.COORDINATE_DISAGREEING.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### MatchSynchronization.summary — src/football_fatigue/synchronization.py:L147-L159

~~~python
def MatchSynchronization.summary(self) -> dict[str, Any]
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/match_data.py:build_quality_summary@L473; src/football_fatigue/playback/server.py:load_match_data@L151
**Callees visible in body:** Counter; abs; dict; len; max; self.tracking_period_ranges.items; sorted; statistics.median; str; sum
**Output expressions (selected excerpts):** L151: Return dict(match_id=self.match_id, event_version=self.event_version.value, events=len(a), matched=len(matched), unmatched=len(a) - len(matched), unmatched_reasons=dict(Counter((x.reason.value for x in a if x.reason))), scan_complete=self.scan_complete, fatal_error=self.fatal_error, ordering_violation=self.ordering_violation, tolerance_ms=self.tolerance_s * 1000....
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### _resolve_groups — src/football_fatigue/synchronization.py:L164-L167

~~~python
def _resolve_groups(groups: dict[tuple[str, int], set[tuple[float, float]]]) -> tuple[dict[tuple[str, int], tuple[float, float]], set[tuple[str, int]]]
~~~
**Declared purpose:** One position per (side, jersey) when all valid duplicates agree; groups with conflicting coordinates are ambiguous. The outcome depends only on the set of coordinates, never on entry order.
**Caller candidates:** src/football_fatigue/synchronization.py:_event_players@L179; src/football_fatigue/synchronization.py:_frame_snapshot@L191
**Callees visible in body:** groups.items; iter; len; next
**Output expressions (selected excerpts):** L167: Return ({k: next(iter(v)) for k, v in groups.items() if len(v) == 1}, {k for k, v in groups.items() if len(v) > 1}).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### _event_players — src/football_fatigue/synchronization.py:L170-L179

~~~python
def _event_players(row: EventRow) -> tuple[dict[tuple[str, int], tuple[float, float]], set[tuple[str, int]]]
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/synchronization.py:assess_spatial@L201
**Callees visible in body:** _resolve_groups; as_float; as_int; e.get; groups.setdefault; groups.setdefault((side, j), set()).add; isinstance; row.positions_snapshot.get; set
**Output expressions (selected excerpts):** L179: Return _resolve_groups(groups).
**Important source branches:** L175: isinstance(e, dict); L177: j is not None and x is not None and (y is not None).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### _frame_snapshot — src/football_fatigue/synchronization.py:L182-L196

~~~python
def _frame_snapshot(frame: TrackingFrame) -> dict[str, Any]
~~~
**Declared purpose:** Compact, raw-only observations of the selected frame (players of quarantined periods are excluded).
**Caller candidates:** src/football_fatigue/synchronization.py:align_events@L288
**Callees visible in body:** _resolve_groups; bool; dict; groups.setdefault; groups.setdefault((p.source_side, p.jersey_number), set()).add; set
**Output expressions (selected excerpts):** L196: Return dict(players=players, ambiguous=ambiguous, ball=ball, players_conflict=bool(unreliable & {'homePlayers', 'awayPlayers'}), ball_conflict='balls' in unreliable).
**Important source branches:** L186: not unreliable & {'homePlayers', 'awayPlayers'}; L193: frame.ball is not None and frame.ball.x_raw_m is not None and (frame.ball.y_raw_m is not None) and ('balls' not in unreliable) and (frame.ball.raw_count == 1); L188: p.jersey_number is None or p.x_raw_m is None or p.y_raw_m is None or (p.attribution_status is AttributionStatus.QUARANTINED).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### assess_spatial — src/football_fatigue/synchronization.py:L199-L223

~~~python
def assess_spatial(row: EventRow, snap: dict[str, Any], tol_m: float=SPATIAL_NEAR_M) -> SpatialAssessment
~~~
**Declared purpose:** Coordinate agreement of one matched event with its frame; raw observations only, reported separately from timing.
**Caller candidates:** src/football_fatigue/synchronization.py:align_events@L306
**Callees visible in body:** SpatialAssessment; _event_players; as_float; b.get; ev.items; ev.keys; isinstance; len; math.hypot; row.positions_snapshot.get; snap['players'].keys; sum
**Output expressions (selected excerpts):** L223: Return SpatialAssessment(status, None, len(d), near, ball, tol_m, len(excluded)); L221: Return SpatialAssessment(SpatialStatus.NOT_ASSESSABLE, reason, len(d), near, ball, tol_m, len(excluded)).
**Important source branches:** L209: len(pts) != 1 or snap['ball'] is None; L214: len(d) < SPATIAL_MIN_PLAYERS; L215: snap['players_conflict']; L217: excluded.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### align_events — src/football_fatigue/synchronization.py:L228-L312

~~~python
def align_events(events: Iterable[EventRow], frames: Iterable[TrackingFrame], *, match_id: int, event_version: EventVersion, fps: float, tolerance_s: float | None=None, spatial: bool=False, spatial_tolerance_m: float=SPATIAL_NEAR_M) -> MatchSynchronization
~~~
**Declared purpose:** Streaming temporal join of event rows to tracking frames (frames must arrive in increasing video-time order).
**Caller candidates:** src/football_fatigue/synchronization.py:synchronize_match@L350
**Callees visible in body:** AlignmentResult; MatchSynchronization; SpatialAssessment; SynchronizationError; _ContextError; _frame_snapshot; _reason; abs; assess_spatial; best.get; best_any.add; dict; enumerate; float; frozenset; isinstance; len; list; math.isfinite; out.append; ranges.get; ranges.items; ranges.setdefault; set; sorted; tuple; type
**Output expressions (selected excerpts):** L311: Return MatchSynchronization(match_id, event_version, fps, tol, tuple(out), complete, scan_error, {k: (v[0], v[1], v[2]) for k, v in ranges.items()}, spatial, ordering_violation=ordering_violation).
**Important source branches:** L240: not (isinstance(fps, (int, float)) and math.isfinite(fps) and (fps > 0)); L243: not (math.isfinite(tol) and tol > 0); L245: not isinstance(event_version, EventVersion); L249: r.match_id != match_id; L251: r.source_version is not event_version; L302: i in best and (not ordering_violation) and (complete or r.event_time_s < settled_before); L264: fr.match_id != match_id.
**Explicit failure paths (excerpts):** L241: SynchronizationError(f'fps must be a positive finite number, got {fps!r}'); L244: SynchronizationError(f'tolerance must be a positive finite number of seconds, got {tolerance_s!r}'); L246: _ContextError('event_version must be an EventVersion'); L250: _ContextError(f'event {r.event_row_id} belongs to match {r.match_id}, not the requested match {match_id}').

#### _reason — src/football_fatigue/synchronization.py:L315-L332

~~~python
def _reason(r: EventRow, i: int, best_any: set[int], ranges: dict, tol: float, complete: bool, settled_before: float | None, ordering_violation: bool=False) -> UnmatchedReason
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/synchronization.py:align_events@L310
**Callees visible in body:** ranges.get
**Output expressions (selected excerpts):** L332: Return UnmatchedReason.NO_FRAME_WITHIN_TOLERANCE; L318: Return UnmatchedReason.MISSING_EVENT_TIME; L320: Return UnmatchedReason.INVALID_SOURCE_PERIOD; L322: Return UnmatchedReason.TRACKING_SCAN_INCOMPLETE.
**Important source branches:** L317: r.event_time_s is None; L319: r.canonical_period is None; L321: ordering_violation; L323: not complete and (settled_before is None or r.event_time_s >= settled_before); L325: i in best_any; L328: rg is None; L330: r.event_time_s < rg[0] - tol - TIE_EPSILON_S or r.event_time_s > rg[1] + tol + TIE_EPSILON_S.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### synchronize_match — src/football_fatigue/synchronization.py:L335-L351

~~~python
def synchronize_match(dataset: Dataset, match_id: object, version: EventVersion, *, tolerance_s: float | None=None, spatial: bool=False, spatial_tolerance_m: float=SPATIAL_NEAR_M, quarantine_periods: Iterable[int]=()) -> MatchSynchronization
~~~
**Declared purpose:** Align every event row of one explicit event version to the match's tracking frames (one streaming pass).
**Caller candidates:** src/football_fatigue/match_data.py:MatchData.synchronize@L299
**Callees visible in body:** SynchronizationError; align_events; dataset.iter_tracking_frames; dataset.load_events; dataset.metadata; dataset.roster; int; isinstance; tuple; validate_match_id
**Output expressions (selected excerpts):** L350: Return align_events(events, frames, match_id=int(mid), event_version=version, fps=meta.fps, tolerance_s=tolerance_s, spatial=spatial, spatial_tolerance_m=spatial_tolerance_m).
**Important source branches:** L342: not isinstance(version, EventVersion).
**Explicit failure paths (excerpts):** L343: SynchronizationError('an EventVersion must be named explicitly').

### Symbols in src/football_fatigue/match_data.py

#### CoordState — src/football_fatigue/match_data.py:L54-L57

~~~python
AVAILABLE = 'available'
MISSING = 'missing'
AMBIGUOUS = 'ambiguous'
~~~

#### CoordinatePolicy — src/football_fatigue/match_data.py:L60-L63

~~~python
RAW = 'raw'
SMOOTHED = 'smoothed'
SMOOTHED_THEN_RAW = 'smoothed_then_raw'
~~~

#### Coordinate — src/football_fatigue/match_data.py:L67-L75
One representation of one observation. ``x``/``y``/``z`` are None unless the state is AVAILABLE.

~~~python
state: CoordState
x: float | None = None
y: float | None = None
z: float | None = None
reason: str | None = None
candidates: tuple[tuple[float | None, float | None], ...] = ()
~~~

#### SelectedCoordinate — src/football_fatigue/match_data.py:L79-L85

~~~python
state: CoordState
x: float | None
y: float | None
z: float | None
source: str | None
reason: str | None
~~~

#### PreparedPlayer — src/football_fatigue/match_data.py:L89-L102

~~~python
source_side: str
jersey_number: int | None
team_id: int | None
player_id: int | None
attribution_status: AttributionStatus
identity_usable: bool
raw: Coordinate
smoothed: Coordinate
selected: SelectedCoordinate
position_usable: bool
caveats: frozenset[str]
sample: PlayerSample
~~~

#### PreparedBall — src/football_fatigue/match_data.py:L106-L112

~~~python
raw: Coordinate
smoothed: Coordinate
selected: SelectedCoordinate
position_usable: bool
caveats: frozenset[str]
quality_flags: frozenset[Q] = frozenset()
~~~

#### PreparedFrame — src/football_fatigue/match_data.py:L116-L143

~~~python
frame: TrackingFrame
players: tuple[PreparedPlayer, ...]
ball: PreparedBall
period_usable: bool
policy: CoordinatePolicy
events: tuple[AlignmentResult, ...] = ()
~~~
**Methods:** frame_num, video_time_s, canonical_period, source_period, quality_flags.

#### MatchData — src/football_fatigue/match_data.py:L250-L332
One match, one explicit event version, one explicit coordinate policy. Construction is cheap (no tracking read).
**Methods:** __init__, open, __repr__, metadata, roster, events, synchronize, unmatched_events, quality_summary, frames.

#### Metric — src/football_fatigue/match_data.py:L341-L350
A count with its denominator and counting unit stated; ``denominator`` None means there is no meaningful base.

~~~python
count: int
denominator: int | None
unit: str
~~~
**Methods:** fraction.

#### ScanInfo — src/football_fatigue/match_data.py:L354-L359

~~~python
status: str
frames_scanned: int
max_frames: int | None
fatal_error: str | None
note: str
~~~

#### QualitySummary — src/football_fatigue/match_data.py:L363-L389
Quality of one match as seen through MatchData. Counting units are in each metric; see docs/team_handoff.md.

~~~python
match_id: str
event_version: str
policy: str
scan: ScanInfo
metrics: dict[str, Metric]
breakdowns: dict[str, dict[str, int]]
breakdown_units: dict[str, str]
tracking_frames_by_canonical_period: dict[str, int]
event_rows_by_canonical_period: dict[str, int]
event_periods_without_tracking: tuple[int, ...] | None
quarantined_periods: tuple[int, ...]
synchronization: dict[str, Any] | None
warnings: tuple[str, ...]
~~~
**Methods:** to_dict.

#### PreparedFrame.frame_num — src/football_fatigue/match_data.py:L126-L127

~~~python
def PreparedFrame.frame_num(self) -> int
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** No explicit calls; field/literal/attribute computation.
**Output expressions (selected excerpts):** L127: Return self.frame.frame_num.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### PreparedFrame.video_time_s — src/football_fatigue/match_data.py:L130-L131

~~~python
def PreparedFrame.video_time_s(self) -> float
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** No explicit calls; field/literal/attribute computation.
**Output expressions (selected excerpts):** L131: Return self.frame.video_time_s.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### PreparedFrame.canonical_period — src/football_fatigue/match_data.py:L134-L135

~~~python
def PreparedFrame.canonical_period(self) -> int | None
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** No explicit calls; field/literal/attribute computation.
**Output expressions (selected excerpts):** L135: Return self.frame.canonical_period.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### PreparedFrame.source_period — src/football_fatigue/match_data.py:L138-L139

~~~python
def PreparedFrame.source_period(self) -> int | None
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** No explicit calls; field/literal/attribute computation.
**Output expressions (selected excerpts):** L139: Return self.frame.source_period.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### PreparedFrame.quality_flags — src/football_fatigue/match_data.py:L142-L143

~~~python
def PreparedFrame.quality_flags(self) -> frozenset[Q]
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** No explicit calls; field/literal/attribute computation.
**Output expressions (selected excerpts):** L143: Return self.frame.quality_flags.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### select — src/football_fatigue/match_data.py:L148-L162

~~~python
def select(raw: Coordinate, smoothed: Coordinate, policy: CoordinatePolicy) -> SelectedCoordinate
~~~
**Declared purpose:** Apply one explicit policy. Conflicting candidates are never reduced to one; there is no interpolation.
**Caller candidates:** src/football_fatigue/match_data.py:prepare_frame@L213; src/football_fatigue/match_data.py:prepare_frame@L242
**Callees visible in body:** SelectedCoordinate; pick
**Output expressions (selected excerpts):** L162: Return SelectedCoordinate(state, None, None, None, None, f'smoothed_{smoothed.state.value}_raw_{raw.state.value}'); L151: Return SelectedCoordinate(c.state, c.x, c.y, c.z, source if c.state is CoordState.AVAILABLE else None, reason or c.reason); L154: Return pick(raw, 'raw'); L156: Return pick(smoothed, 'smoothed').
**Important source branches:** L153: policy is CoordinatePolicy.RAW; L155: policy is CoordinatePolicy.SMOOTHED; L157: smoothed.state is CoordState.AVAILABLE; L159: raw.state is CoordState.AVAILABLE.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### select.pick — src/football_fatigue/match_data.py:L150-L151

~~~python
def select.pick(c: Coordinate, source: str, reason: str | None=None) -> SelectedCoordinate
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/match_data.py:select@L154; src/football_fatigue/match_data.py:select@L156; src/football_fatigue/match_data.py:select@L158
**Callees visible in body:** SelectedCoordinate
**Output expressions (selected excerpts):** L151: Return SelectedCoordinate(c.state, c.x, c.y, c.z, source if c.state is CoordState.AVAILABLE else None, reason or c.reason).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### _raw_players — src/football_fatigue/match_data.py:L167-L172

~~~python
def _raw_players(frame: TrackingFrame) -> dict[tuple[str, int], set[tuple[float, float]]]
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/match_data.py:prepare_frame@L209
**Callees visible in body:** groups.setdefault; groups.setdefault((p.source_side, p.jersey_number), set()).add; set
**Output expressions (selected excerpts):** L172: Return groups.
**Important source branches:** L170: p.jersey_number is not None and p.x_raw_m is not None and (p.y_raw_m is not None).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### _player_coords — src/football_fatigue/match_data.py:L175-L196

~~~python
def _player_coords(p: PlayerSample, frame: TrackingFrame, raw_groups: dict) -> tuple[Coordinate, Coordinate]
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/match_data.py:prepare_frame@L212
**Callees visible in body:** Coordinate; len; raw_groups.get; set; sorted; tuple
**Output expressions (selected excerpts):** L196: Return (raw, sm).
**Important source branches:** L178: side_raw in conflicts; L186: side_raw + 'Smoothed' in conflicts; L180: p.jersey_number is not None and len(raw_groups.get((p.source_side, p.jersey_number), ())) > 1; L188: p.smoothed_status is SmoothedStatus.AMBIGUOUS_CONFLICTING; L182: p.x_raw_m is None or p.y_raw_m is None; L190: p.smoothed_status is SmoothedStatus.MISSING; L192: p.x_smooth_m is None or p.y_smooth_m is None.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### _caveats — src/football_fatigue/match_data.py:L199-L203

~~~python
def _caveats(flags: Iterable[Q], source: str | None) -> frozenset[str]
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/match_data.py:prepare_frame@L216; src/football_fatigue/match_data.py:prepare_frame@L243
**Callees visible in body:** frozenset; out.add
**Output expressions (selected excerpts):** L203: Return frozenset(out).
**Important source branches:** L201: source == 'raw_fallback'.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### prepare_frame — src/football_fatigue/match_data.py:L206-L245

~~~python
def prepare_frame(frame: TrackingFrame, policy: CoordinatePolicy=CoordinatePolicy.RAW, events: tuple[AlignmentResult, ...]=()) -> PreparedFrame
~~~
**Declared purpose:** Wrap a canonical frame with explicit coordinate states, a policy-selected coordinate, and usability indicators.
**Caller candidates:** src/football_fatigue/match_data.py:MatchData.frames@L332
**Callees visible in body:** Coordinate; PreparedBall; PreparedFrame; PreparedPlayer; _caveats; _player_coords; _raw_players; players.append; select; set; tuple
**Output expressions (selected excerpts):** L245: Return PreparedFrame(frame, players_t, ball, period_ok, policy, events).
**Important source branches:** L224: raw_conflict; L234: sm_conflict; L226: b is None; L236: b is None; L228: b.raw_count > 1; L238: b.x_smooth_m is None or b.y_smooth_m is None; L230: b.x_raw_m is None or b.y_raw_m is None.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### MatchData.__init__ — src/football_fatigue/match_data.py:L253-L267

~~~python
def MatchData.__init__(self, dataset: Dataset, match_id: object, event_version: EventVersion, *, policy: CoordinatePolicy=CoordinatePolicy.RAW, quarantine_periods: Iterable[int]=())
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/inventory.py:InventoryScanError.__init__@L94
**Callees visible in body:** IngestionError; isinstance; tuple; validate_match_id
**Output:** no explicit return/yield; mutation, exception/command action, or implicit None. See signature and module context.
**Important source branches:** L255: not isinstance(event_version, EventVersion); L257: not isinstance(policy, CoordinatePolicy).
**Explicit failure paths (excerpts):** L256: IngestionError('an EventVersion must be named explicitly (root, march_14_2025, may_1_2025)'); L258: IngestionError('policy must be a CoordinatePolicy').

#### MatchData.open — src/football_fatigue/match_data.py:L270-L272

~~~python
def MatchData.open(cls, dataset_or_root: Dataset | str | Path, match_id: object, event_version: EventVersion, **kw: Any) -> 'MatchData'
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** scripts/audit/audit_lib.py:_save@L439; scripts/audit/audit_lib.py:iter_tracking@L91; scripts/audit/audit_lib.py:load_json@L76; src/football_fatigue/ingestion/loaders.py:_iter_records@L362; src/football_fatigue/ingestion/loaders.py:_read_json@L86; src/football_fatigue/ingestion/validation.py:write_quality_report@L233; src/football_fatigue/playback/server.py:load_match_data@L136; src/football_fatigue/playback/server.py:run@L913
**Callees visible in body:** Dataset; cls; isinstance
**Output expressions (selected excerpts):** L272: Return cls(ds, match_id, event_version, **kw).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### MatchData.__repr__ — src/football_fatigue/match_data.py:L274-L275

~~~python
def MatchData.__repr__(self) -> str
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** No explicit calls; field/literal/attribute computation.
**Output expressions (selected excerpts):** L275: Return f'MatchData(match={self.match_id}, events={self.event_version.value}, policy={self.policy.value})'.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### MatchData.metadata — src/football_fatigue/match_data.py:L279-L282

~~~python
def MatchData.metadata(self) -> MatchMetadata
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/ingestion/loaders.py:Dataset.iter_tracking_frames@L197; src/football_fatigue/ingestion/loaders.py:Dataset.roster@L151; src/football_fatigue/ingestion/validation.py:quality_report@L121; src/football_fatigue/playback/server.py:get_matches_catalog@L76; src/football_fatigue/synchronization.py:synchronize_match@L345
**Callees visible in body:** self.dataset.metadata
**Output expressions (selected excerpts):** L282: Return self._metadata.
**Important source branches:** L280: self._metadata is None.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### MatchData.roster — src/football_fatigue/match_data.py:L285-L288

~~~python
def MatchData.roster(self) -> Roster
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/ingestion/validation.py:quality_report@L123; src/football_fatigue/synchronization.py:synchronize_match@L348
**Callees visible in body:** self.dataset.roster
**Output expressions (selected excerpts):** L288: Return self._roster.
**Important source branches:** L286: self._roster is None.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### MatchData.events — src/football_fatigue/match_data.py:L290-L294

~~~python
def MatchData.events(self) -> list[EventRow]
~~~
**Declared purpose:** Rows of the requested version only, in source order, duplicates preserved.
**Caller candidates:** src/football_fatigue/match_data.py:build_quality_summary@L459; src/football_fatigue/playback/server.py:load_match_data@L154
**Callees visible in body:** list; self.dataset.load_events
**Output expressions (selected excerpts):** L294: Return list(self._events).
**Important source branches:** L292: self._events is None.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### MatchData.synchronize — src/football_fatigue/match_data.py:L297-L300

~~~python
def MatchData.synchronize(self, *, spatial: bool=False) -> MatchSynchronization
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/match_data.py:MatchData.frames@L325; src/football_fatigue/match_data.py:MatchData.unmatched_events@L304; src/football_fatigue/match_data.py:build_quality_summary@L472; src/football_fatigue/playback/controller.py:PlaybackController.from_match_data@L166; src/football_fatigue/playback/server.py:load_match_data@L148
**Callees visible in body:** synchronize_match
**Output expressions (selected excerpts):** L300: Return self._sync[spatial].
**Important source branches:** L298: spatial not in self._sync.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### MatchData.unmatched_events — src/football_fatigue/match_data.py:L302-L304

~~~python
def MatchData.unmatched_events(self) -> tuple[AlignmentResult, ...]
~~~
**Declared purpose:** Every event the synchronization could not place, with its reason; never hidden.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** self.synchronize; tuple
**Output expressions (selected excerpts):** L304: Return tuple((a for a in self.synchronize().alignments if a.status is MatchStatus.UNMATCHED)).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### MatchData.quality_summary — src/football_fatigue/match_data.py:L307-L313

~~~python
def MatchData.quality_summary(self, *, max_frames: int | None=None, include_synchronization: bool=False) -> 'QualitySummary'
~~~
**Declared purpose:** Quality summary of this match. NOT run when a match is opened: it streams the tracking file (``max_frames`` bounds it).
**Additional declaration:** ``include_synchronization=True`` adds the event coverage and unmatched reasons, which costs one more full tracking pass unless :meth:`synchronize` was already called. A partial or failed scan never asserts that a period is missing.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** build_quality_summary
**Output expressions (selected excerpts):** L313: Return build_quality_summary(self, max_frames=max_frames, include_synchronization=include_synchronization).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### MatchData.frames — src/football_fatigue/match_data.py:L316-L332

~~~python
def MatchData.frames(self, *, attach_events: bool=False, max_frames: int | None=None) -> Iterator[PreparedFrame]
~~~
**Declared purpose:** Lazy, chronological (source order) prepared frames; memory stays bounded unless ``attach_events`` is set.
**Additional declaration:** ``attach_events=True`` first runs (or reuses) the synchronization, which is a full extra streaming pass, and buffers the matched alignments grouped by frame (O(E)). Unmatched events are not attached to any frame; use :meth:`unmatched_events`.
**Caller candidates:** src/football_fatigue/match_data.py:build_quality_summary@L401; src/football_fatigue/playback/controller.py:PlaybackController.from_match_data@L170; src/football_fatigue/playback/server.py:load_match_data@L202
**Callees visible in body:** by_frame.pop; by_frame.setdefault; by_frame.setdefault(tuple(a.frame_source_record_indices), []).append; prepare_frame; self.dataset.iter_tracking_frames; self.synchronize; tuple
**Output expressions (selected excerpts):** L332: Yield prepare_frame(fr, self.policy, tuple((a for a in hit if a.frame_num == fr.frame_num and a.frame_period == fr.canonical_period))).
**Important source branches:** L324: attach_events; L326: a.status is MatchStatus.MATCHED.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### Metric.fraction — src/football_fatigue/match_data.py:L349-L350

~~~python
def Metric.fraction(self) -> float | None
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** No explicit calls; field/literal/attribute computation.
**Output expressions (selected excerpts):** L350: Return None if not self.denominator else self.count / self.denominator.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### QualitySummary.to_dict — src/football_fatigue/match_data.py:L380-L389

~~~python
def QualitySummary.to_dict(self) -> dict[str, Any]
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/ingestion/validation.py:write_quality_report@L234; src/football_fatigue/inventory.py:main@L473
**Callees visible in body:** dict; list; self.metrics.items
**Output expressions (selected excerpts):** L381: Return dict(match_id=self.match_id, event_version=self.event_version, policy=self.policy, scan=dict(status=self.scan.status, frames_scanned=self.scan.frames_scanned, max_frames=self.scan.max_frames, fatal_error=self.scan.fatal_error, note=self.scan.note), metrics={k: dict(count=m.count, denominator=m.denominator, unit=m.unit) for k, m in self.metrics.items()}, b....
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### build_quality_summary — src/football_fatigue/match_data.py:L392-L496

~~~python
def build_quality_summary(md: 'MatchData', *, max_frames: int | None, include_synchronization: bool) -> QualitySummary
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/match_data.py:MatchData.quality_summary@L313
**Callees visible in body:** Counter; M; QualitySummary; ScanInfo; TrackingQuality; bool; dict; ev_by_period.update; ev_flags.get; int; len; list; md.events; md.frames; md.synchronize; s.summary; sorted; str; sum; sync.get; tq.add; tq.attribution_status.items; tuple; v.lower; warn.append
**Output expressions (selected excerpts):** L495: Return QualitySummary(md.match_id, md.event_version.value, md.policy.value, ScanInfo(status, scanned, max_frames, fatal, note), metrics, breakdowns, units, dict(tq.frames_by_canonical_period), dict(ev_by_period), missing, md.quarantine_periods, sync, tuple(warn)).
**Important source branches:** L470: include_synchronization; L481: status != SCAN_COMPLETE; L483: events_error; L485: missing; L487: md.quarantine_periods; L489: c['vis_estimated']; L491: c['raw_ambiguous'] or c['sm_ambiguous'] or c['ball_raw_ambiguous'] or c['ball_sm_ambiguous'].
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

### Symbols in src/football_fatigue/playback/__init__.py

### Symbols in src/football_fatigue/playback/__main__.py

### Symbols in src/football_fatigue/playback/adapter.py

#### frame_to_state — src/football_fatigue/playback/adapter.py:L30-L120

~~~python
def frame_to_state(pf: PreparedFrame, roster_names: dict[int, str] | None=None, event_lookup: dict[str, Any] | None=None) -> dict[str, Any]
~~~
**Declared purpose:** Convert one PreparedFrame to a JSON-serialisable FrameState dict.
**Caller candidates:** src/football_fatigue/playback/controller.py:PlaybackController.__init__@L106; src/football_fatigue/playback/controller.py:PlaybackController.from_match_data@L171; src/football_fatigue/playback/server.py:load_match_data@L220
**Callees visible in body:** event_lookup.get; events.append; getattr; players.append
**Output expressions (selected excerpts):** L110: Return {'frame_num': pf.frame_num, 'video_time_s': pf.video_time_s, 'canonical_period': pf.canonical_period, 'source_period': pf.source_period, 'period_usable': pf.period_usable, 'has_gap': has_gap, 'players': players, 'ball': ball, 'events': events}.
**Important source branches:** L80: pf.ball.position_usable; L59: not p.position_usable; L73: p.identity_usable; L75: roster_names and p.player_id and (p.player_id in roster_names).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### build_roster_names — src/football_fatigue/playback/adapter.py:L123-L129

~~~python
def build_roster_names(roster) -> dict[int, str]
~~~
**Declared purpose:** Build a ``{player_id: display_name}`` map from a ``Roster`` object.
**Caller candidates:** src/football_fatigue/playback/controller.py:PlaybackController.from_match_data@L165; src/football_fatigue/playback/server.py:load_match_data@L140
**Callees visible in body:** No explicit calls; field/literal/attribute computation.
**Output expressions (selected excerpts):** L129: Return names.
**Important source branches:** L127: e.player_id and e.nickname.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### build_roster_lookup — src/football_fatigue/playback/adapter.py:L132-L144

~~~python
def build_roster_lookup(roster) -> dict[int, dict[str, Any]]
~~~
**Declared purpose:** Build a ``{player_id: {name, jersey, side, position_group, team_id}}`` map.
**Caller candidates:** src/football_fatigue/playback/server.py:load_match_data@L139
**Callees visible in body:** No explicit calls; field/literal/attribute computation.
**Output expressions (selected excerpts):** L144: Return lookup.
**Important source branches:** L136: e.player_id.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### build_event_index — src/football_fatigue/playback/adapter.py:L147-L224

~~~python
def build_event_index(match_sync, events: Sequence[Any] | None=None, roster_lookup: dict[int, dict[str, Any]] | None=None, meta: Any | None=None) -> list[dict[str, Any]]
~~~
**Declared purpose:** Produce an ordered list of all event alignment results for the event panel.
**Caller candidates:** src/football_fatigue/playback/controller.py:PlaybackController.from_match_data@L167; src/football_fatigue/playback/server.py:load_match_data@L156
**Callees visible in body:** any; attrs.get; bool; ev.get; event_lookup.get; fouls_dict.get; fouls_dict.values; getattr; isinstance; pe_dict.get; player_info.get; results.append; roster_lookup.get; str
**Output expressions (selected excerpts):** L224: Return results.
**Important source branches:** L162: events; L193: meta; L218: matched; L165: eid is None and isinstance(ev, dict); L167: eid is not None; L194: player_side == 'home' or s_team == getattr(meta, 'home_team_id', None); L196: player_side == 'away' or s_team == getattr(meta, 'away_team_id', None).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

### Symbols in src/football_fatigue/playback/controller.py

#### NavigationMode — src/football_fatigue/playback/controller.py:L29-L31

~~~python
TIMESTAMP = 'timestamp'
EVENT = 'event'
~~~

#### EventItem — src/football_fatigue/playback/controller.py:L35-L62
Standardized representation of an event for playback navigation.

~~~python
event_row_id: str
game_event_type: str | None
possession_event_type: str | None
canonical_period: int
event_time_s: float | None
matched: bool
frame_num: int | None = None
residual_s: float | None = None
quality_label: str = 'unmatched'
reason: str | None = None
source_team_id: int | None = None
source_player_id: int | None = None
~~~
**Methods:** display_type, matches_filter.

#### PlaybackState — src/football_fatigue/playback/controller.py:L66-L78
The single authoritative playback state.

~~~python
frame_index: int = 0
frame_num: int = 0
video_time_s: float = 0.0
canonical_period: int = 1
period_usable: bool = True
has_gap: bool = False
is_playing: bool = False
playback_speed: float = 1.0
mode: NavigationMode = NavigationMode.TIMESTAMP
selected_event_id: str | None = None
active_filter: str | None = None
~~~

#### PlaybackController — src/football_fatigue/playback/controller.py:L81-L413
Controls match playback and event-driven navigation.
**Methods:** __init__, from_match_data, state, total_frames, current_frame_state, all_events, fps, _sync_state_to_index, play, pause, toggle_play, set_speed, restart, step, seek_frame_num, seek_time, set_event_filter, get_filtered_events, get_unmatched_events, select_event, next_event, prev_event.

#### EventItem.display_type — src/football_fatigue/playback/controller.py:L51-L52

~~~python
def EventItem.display_type(self) -> str
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** No explicit calls; field/literal/attribute computation.
**Output expressions (selected excerpts):** L52: Return self.possession_event_type or self.game_event_type or 'UNKNOWN'.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### EventItem.matches_filter — src/football_fatigue/playback/controller.py:L54-L62

~~~python
def EventItem.matches_filter(self, filter_type: str | None) -> bool
~~~
**Declared purpose:** Check if this event matches the requested category or event type.
**Caller candidates:** src/football_fatigue/playback/controller.py:PlaybackController.get_filtered_events@L306; src/football_fatigue/playback/controller.py:PlaybackController.get_unmatched_events@L316
**Callees visible in body:** filter_type.upper; self.game_event_type.upper; self.possession_event_type.upper
**Output expressions (selected excerpts):** L59: Return self.game_event_type is not None and self.game_event_type.upper() == f or (self.possession_event_type is not None and self.possession_event_type.upper() == f); L57: Return True.
**Important source branches:** L56: not filter_type or filter_type.upper() in ('ALL', '*').
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### PlaybackController.__init__ — src/football_fatigue/playback/controller.py:L88-L155

~~~python
def PlaybackController.__init__(self, frames: Sequence[PreparedFrame | dict[str, Any]], events: Sequence[EventItem | dict[str, Any]], fps: float=29.97) -> None
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/inventory.py:InventoryScanError.__init__@L94
**Callees visible in body:** EventItem; PlaybackState; TypeError; ValueError; bool; enumerate; ev.get; first_frame.get; float; frame_to_state; int; isinstance; self._events.append; self._frames.append; str; type
**Output:** no explicit return/yield; mutation, exception/command action, or implicit None. See signature and module context.
**Important source branches:** L94: not frames; L103: isinstance(f, dict); L112: f_num not in self._frame_num_to_index; L119: isinstance(ev, EventItem); L105: isinstance(f, PreparedFrame); L121: isinstance(ev, dict).
**Explicit failure paths (excerpts):** L95: ValueError('PlaybackController requires at least one frame.'); L108: TypeError(f'Unsupported frame type: {type(f)}'); L137: TypeError(f'Unsupported event type: {type(ev)}').

#### PlaybackController.from_match_data — src/football_fatigue/playback/controller.py:L158-L173

~~~python
def PlaybackController.from_match_data(cls, match_data: MatchData, max_frames: int | None=1500, policy: CoordinatePolicy=CoordinatePolicy.SMOOTHED_THEN_RAW) -> PlaybackController
~~~
**Declared purpose:** Factory method constructing a controller directly from MatchData.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** build_event_index; build_roster_names; cls; frame_to_state; frames.append; match_data.frames; match_data.synchronize
**Output expressions (selected excerpts):** L173: Return cls(frames=frames, events=event_index, fps=match_data.metadata.fps).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### PlaybackController.state — src/football_fatigue/playback/controller.py:L178-L179

~~~python
def PlaybackController.state(self) -> PlaybackState
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** No explicit calls; field/literal/attribute computation.
**Output expressions (selected excerpts):** L179: Return self._state.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### PlaybackController.total_frames — src/football_fatigue/playback/controller.py:L182-L183

~~~python
def PlaybackController.total_frames(self) -> int
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** len
**Output expressions (selected excerpts):** L183: Return len(self._frames).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### PlaybackController.current_frame_state — src/football_fatigue/playback/controller.py:L186-L187

~~~python
def PlaybackController.current_frame_state(self) -> dict[str, Any]
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** No explicit calls; field/literal/attribute computation.
**Output expressions (selected excerpts):** L187: Return self._frames[self._state.frame_index].
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### PlaybackController.all_events — src/football_fatigue/playback/controller.py:L190-L191

~~~python
def PlaybackController.all_events(self) -> list[EventItem]
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** list
**Output expressions (selected excerpts):** L191: Return list(self._events).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### PlaybackController.fps — src/football_fatigue/playback/controller.py:L194-L195

~~~python
def PlaybackController.fps(self) -> float
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** No explicit calls; field/literal/attribute computation.
**Output expressions (selected excerpts):** L195: Return self._fps.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### PlaybackController._sync_state_to_index — src/football_fatigue/playback/controller.py:L199-L208

~~~python
def PlaybackController._sync_state_to_index(self, new_idx: int) -> None
~~~
**Declared purpose:** Internal helper to update authoritative state when frame index changes.
**Caller candidates:** src/football_fatigue/playback/controller.py:PlaybackController.play@L215; src/football_fatigue/playback/controller.py:PlaybackController.restart@L239; src/football_fatigue/playback/controller.py:PlaybackController.seek_frame_num@L260; src/football_fatigue/playback/controller.py:PlaybackController.seek_time@L267; src/football_fatigue/playback/controller.py:PlaybackController.seek_time@L270; src/football_fatigue/playback/controller.py:PlaybackController.seek_time@L293; src/football_fatigue/playback/controller.py:PlaybackController.select_event@L344; src/football_fatigue/playback/controller.py:PlaybackController.step@L249
**Callees visible in body:** f.get; len; max; min
**Output:** no explicit return/yield; mutation, exception/command action, or implicit None. See signature and module context.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### PlaybackController.play — src/football_fatigue/playback/controller.py:L212-L216

~~~python
def PlaybackController.play(self) -> None
~~~
**Declared purpose:** Start or resume playback.
**Caller candidates:** src/football_fatigue/playback/controller.py:PlaybackController.toggle_play@L227
**Callees visible in body:** len; self._sync_state_to_index
**Output:** no explicit return/yield; mutation, exception/command action, or implicit None. See signature and module context.
**Important source branches:** L214: self._state.frame_index >= len(self._frames) - 1.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### PlaybackController.pause — src/football_fatigue/playback/controller.py:L218-L220

~~~python
def PlaybackController.pause(self) -> None
~~~
**Declared purpose:** Pause playback.
**Caller candidates:** src/football_fatigue/playback/controller.py:PlaybackController.restart@L238; src/football_fatigue/playback/controller.py:PlaybackController.toggle_play@L225
**Callees visible in body:** No explicit calls; field/literal/attribute computation.
**Output:** no explicit return/yield; mutation, exception/command action, or implicit None. See signature and module context.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### PlaybackController.toggle_play — src/football_fatigue/playback/controller.py:L222-L228

~~~python
def PlaybackController.toggle_play(self) -> bool
~~~
**Declared purpose:** Toggle between play and pause.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** self.pause; self.play
**Output expressions (selected excerpts):** L228: Return self._state.is_playing.
**Important source branches:** L224: self._state.is_playing.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### PlaybackController.set_speed — src/football_fatigue/playback/controller.py:L230-L234

~~~python
def PlaybackController.set_speed(self, speed: float) -> None
~~~
**Declared purpose:** Set playback speed multiplier (e.g. 0.5, 1.0, 2.0).
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** ValueError; float
**Output:** no explicit return/yield; mutation, exception/command action, or implicit None. See signature and module context.
**Important source branches:** L232: speed <= 0.
**Explicit failure paths (excerpts):** L233: ValueError('Playback speed must be positive.').

#### PlaybackController.restart — src/football_fatigue/playback/controller.py:L236-L241

~~~python
def PlaybackController.restart(self) -> None
~~~
**Declared purpose:** Rewind to the first frame and pause.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** self._sync_state_to_index; self.pause
**Output:** no explicit return/yield; mutation, exception/command action, or implicit None. See signature and module context.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### PlaybackController.step — src/football_fatigue/playback/controller.py:L243-L249

~~~python
def PlaybackController.step(self, delta_frames: int=1) -> None
~~~
**Declared purpose:** Advance or step backward by delta_frames.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** self._sync_state_to_index
**Output:** no explicit return/yield; mutation, exception/command action, or implicit None. See signature and module context.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### PlaybackController.seek_frame_num — src/football_fatigue/playback/controller.py:L251-L261

~~~python
def PlaybackController.seek_frame_num(self, frame_num: int) -> bool
~~~
**Declared purpose:** Seek to a specific frame number.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** self._frame_num_to_index.get; self._sync_state_to_index
**Output expressions (selected excerpts):** L261: Return True; L258: Return False.
**Important source branches:** L257: idx is None.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### PlaybackController.seek_time — src/football_fatigue/playback/controller.py:L263-L293

~~~python
def PlaybackController.seek_time(self, timestamp_s: float) -> None
~~~
**Declared purpose:** Seek to the nearest frame corresponding to timestamp_s.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** abs; float; len; self._sync_state_to_index
**Output expressions (selected excerpts):** L268: Return None; L271: Return None.
**Important source branches:** L266: timestamp_s <= self._frames[0]['video_time_s']; L269: timestamp_s >= self._frames[-1]['video_time_s']; L282: diff < min_diff; L285: t < timestamp_s; L287: t > timestamp_s.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### PlaybackController.set_event_filter — src/football_fatigue/playback/controller.py:L297-L299

~~~python
def PlaybackController.set_event_filter(self, filter_type: str | None) -> None
~~~
**Declared purpose:** Set the active event filter category (e.g. 'PA', 'SH', 'FO', 'G', 'ALL').
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** No explicit calls; field/literal/attribute computation.
**Output:** no explicit return/yield; mutation, exception/command action, or implicit None. See signature and module context.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### PlaybackController.get_filtered_events — src/football_fatigue/playback/controller.py:L301-L311

~~~python
def PlaybackController.get_filtered_events(self, include_unmatched: bool=True) -> list[EventItem]
~~~
**Declared purpose:** Return all events matching the active filter in chronological order.
**Caller candidates:** src/football_fatigue/playback/controller.py:PlaybackController.next_event@L356; src/football_fatigue/playback/controller.py:PlaybackController.prev_event@L389
**Callees visible in body:** ev.matches_filter; out.append
**Output expressions (selected excerpts):** L311: Return out.
**Important source branches:** L306: not ev.matches_filter(filt); L308: not include_unmatched and (not ev.matched).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### PlaybackController.get_unmatched_events — src/football_fatigue/playback/controller.py:L313-L316

~~~python
def PlaybackController.get_unmatched_events(self) -> list[EventItem]
~~~
**Declared purpose:** Return unmatched events matching the active filter, with reasons.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** ev.matches_filter
**Output expressions (selected excerpts):** L316: Return [ev for ev in self._events if not ev.matched and ev.matches_filter(filt)].
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### PlaybackController.select_event — src/football_fatigue/playback/controller.py:L318-L345

~~~python
def PlaybackController.select_event(self, event_row_id: str) -> bool
~~~
**Declared purpose:** Jump to the frame aligned with the specified event.
**Caller candidates:** src/football_fatigue/playback/controller.py:PlaybackController.next_event@L371; src/football_fatigue/playback/controller.py:PlaybackController.next_event@L379; src/football_fatigue/playback/controller.py:PlaybackController.next_event@L382; src/football_fatigue/playback/controller.py:PlaybackController.prev_event@L402; src/football_fatigue/playback/controller.py:PlaybackController.prev_event@L410
**Callees visible in body:** self._events_by_id.get; self._frame_num_to_index.get; self._sync_state_to_index; str
**Output expressions (selected excerpts):** L345: Return True; L328: Return False; L333: Return False; L339: Return False.
**Important source branches:** L327: ev is None; L330: not ev.matched or ev.frame_num is None; L336: frame_idx is None.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### PlaybackController.next_event — src/football_fatigue/playback/controller.py:L347-L385

~~~python
def PlaybackController.next_event(self) -> EventItem | None
~~~
**Declared purpose:** Navigate to the next matched event in chronological order.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** enumerate; len; next; self.get_filtered_events; self.select_event
**Output expressions (selected excerpts):** L385: Return None; L358: Return None; L380: Return ev; L372: Return next_ev.
**Important source branches:** L357: not candidates; L365: cur_ev_id; L378: ev.frame_num is not None and ev.frame_num > cur_frame; L369: cur_pos + 1 < len(candidates); L379: self.select_event(ev.event_row_id); L381: ev.event_time_s is not None and ev.event_time_s > cur_time + 0.0001; L371: self.select_event(next_ev.event_row_id).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### PlaybackController.prev_event — src/football_fatigue/playback/controller.py:L387-L413

~~~python
def PlaybackController.prev_event(self) -> EventItem | None
~~~
**Declared purpose:** Navigate to the previous matched event in reverse chronological order.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** enumerate; next; reversed; self.get_filtered_events; self.select_event
**Output expressions (selected excerpts):** L413: Return None; L391: Return None; L411: Return ev; L403: Return prev_ev.
**Important source branches:** L390: not candidates; L397: cur_ev_id; L409: ev.frame_num is not None and ev.frame_num < cur_frame; L400: cur_pos > 0; L410: self.select_event(ev.event_row_id); L402: self.select_event(prev_ev.event_row_id).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

### Symbols in src/football_fatigue/playback/server.py

#### BoardHandler — src/football_fatigue/playback/server.py:L280-L863
Serves the tactical board HTML and all JSON endpoints.
Shared HTTP handler; only original FIFA state/routes are in scope. Fields: payload, root, lock, board_html_path. Original methods are catalogued individually below.

#### get_matches_catalog — src/football_fatigue/playback/server.py:L66-L101

~~~python
def get_matches_catalog(root: str) -> list[dict[str, Any]]
~~~
**Declared purpose:** Scan dataset root for all matches, categorized by tournament stage.
**Caller candidates:** src/football_fatigue/playback/server.py:BoardHandler.do_GET@L343
**Callees visible in body:** Dataset; STAGE_ORDER.get; ds.match_ids; ds.metadata; matches.append; matches.sort; print; str
**Input/output and edge cases:** see Chapters 8 and 11 for the original FIFO load, response schema, defaults, statuses and errors. Shared later-workstream branches are deliberately not expanded here.

#### load_match_data — src/football_fatigue/playback/server.py:L106-L275

~~~python
def load_match_data(root: str, match_id: str, max_frames: int | None=None, *, scope: str='half1', period: int | None=None, minutes: float | None=None, policy: CoordinatePolicy=CoordinatePolicy.SMOOTHED_THEN_RAW) -> dict[str, Any]
~~~
**Declared purpose:** Load a match segment and event index into memory.
**Caller candidates:** src/football_fatigue/playback/server.py:BoardHandler.do_GET@L384; src/football_fatigue/playback/server.py:BoardHandler.do_POST@L432; src/football_fatigue/playback/server.py:run@L893
**Callees visible in body:** MatchData.open; build_event_index; build_roster_lookup; build_roster_names; ev.get; float; frame_to_state; frames.append; goal_events.append; int; len; md.events; md.frames; md.synchronize; print; raw_prepared_frames.append; round; summaries.items; summarize_match_movement; sync.summary; time.perf_counter; v.as_dict
**Input/output and edge cases:** see Chapters 8 and 11 for the original FIFO load, response schema, defaults, statuses and errors. Shared later-workstream branches are deliberately not expanded here.

#### BoardHandler.do_OPTIONS — src/football_fatigue/playback/server.py:L298-L303

~~~python
def BoardHandler.do_OPTIONS(self) -> None
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** self.end_headers; self.send_header; self.send_response
**Input/output and edge cases:** see Chapters 8 and 11 for the original FIFO load, response schema, defaults, statuses and errors. Shared later-workstream branches are deliberately not expanded here.

#### BoardHandler.do_GET — src/football_fatigue/playback/server.py:L305-L407

~~~python
def BoardHandler.do_GET(self) -> None
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** float; get_matches_catalog; int; len; load_match_data; max_frames_val.isdigit; parse_qs; period_val.isdigit; qs.get; self._csv_report; self._json; self._serve_file; self._serve_video; self.payload.get; self.payload.items; self.send_error; urlparse
**Input/output and edge cases:** see Chapters 8 and 11 for the original FIFO load, response schema, defaults, statuses and errors. Shared later-workstream branches are deliberately not expanded here.

#### BoardHandler.do_POST — src/football_fatigue/playback/server.py:L409-L454

~~~python
def BoardHandler.do_POST(self) -> None
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** body.decode; data.get; float; int; json.loads; len; load_match_data; self._json; self.headers.get; self.payload.get; self.rfile.read; self.send_error; str; urlparse
**Input/output and edge cases:** see Chapters 8 and 11 for the original FIFO load, response schema, defaults, statuses and errors. Shared later-workstream branches are deliberately not expanded here.

#### BoardHandler._json — src/football_fatigue/playback/server.py:L456-L463

~~~python
def BoardHandler._json(self, data: Any) -> None
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/playback/server.py:BoardHandler.do_GET@L344; src/football_fatigue/playback/server.py:BoardHandler.do_GET@L348; src/football_fatigue/playback/server.py:BoardHandler.do_GET@L351; src/football_fatigue/playback/server.py:BoardHandler.do_GET@L354; src/football_fatigue/playback/server.py:BoardHandler.do_GET@L357; src/football_fatigue/playback/server.py:BoardHandler.do_GET@L360; src/football_fatigue/playback/server.py:BoardHandler.do_GET@L363; src/football_fatigue/playback/server.py:BoardHandler.do_GET@L394; src/football_fatigue/playback/server.py:BoardHandler.do_POST@L442
**Callees visible in body:** json.dumps; json.dumps(data, allow_nan=False).encode; len; self.end_headers; self.send_header; self.send_response; self.wfile.write; str
**Input/output and edge cases:** see Chapters 8 and 11 for the original FIFO load, response schema, defaults, statuses and errors. Shared later-workstream branches are deliberately not expanded here.

#### BoardHandler._serve_file — src/football_fatigue/playback/server.py:L523-L535

~~~python
def BoardHandler._serve_file(self, path: Path, content_type: str, no_store: bool=False) -> None
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/playback/server.py:BoardHandler.do_GET@L310; src/football_fatigue/playback/server.py:BoardHandler.do_GET@L313; src/football_fatigue/playback/server.py:BoardHandler.do_GET@L316
**Callees visible in body:** len; path.exists; path.read_bytes; self.end_headers; self.send_error; self.send_header; self.send_response; self.wfile.write; str
**Input/output and edge cases:** see Chapters 8 and 11 for the original FIFO load, response schema, defaults, statuses and errors. Shared later-workstream branches are deliberately not expanded here.

#### BoardHandler.log_message — src/football_fatigue/playback/server.py:L862-L863

~~~python
def BoardHandler.log_message(self, fmt: str, *args: Any) -> None
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** No explicit calls; field/literal/attribute computation.
**Input/output and edge cases:** see Chapters 8 and 11 for the original FIFO load, response schema, defaults, statuses and errors. Shared later-workstream branches are deliberately not expanded here.

#### run — src/football_fatigue/playback/server.py:L868-L918

~~~python
def run(root: str='data/raw/FIFA World Cup 2022', match_id: str | None=None, max_frames: int | None=None, scope: str='half1', period: int | None=1, minutes: float | None=None, policy: str='smoothed_then_raw', port: int=5780, open_browser: bool=True) -> None
~~~
**Declared purpose:** Start the HTTP server. If match_id is None, prompts in the web interface.
**Caller candidates:** src/football_fatigue/playback/server.py:_cli@L940
**Callees visible in body:** Path.cwd; ThreadingHTTPServer; load_match_data; print; server.serve_forever; threading.Thread; threading.Thread(target=lambda: (time.sleep(0.4), webbrowser.open(url)), daemon=True).start; time.sleep; webbrowser.open
**Input/output and edge cases:** see Chapters 8 and 11 for the original FIFO load, response schema, defaults, statuses and errors. Shared later-workstream branches are deliberately not expanded here.

#### _cli — src/football_fatigue/playback/server.py:L921-L949

~~~python
def _cli() -> None
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** argparse.ArgumentParser; p.add_argument; p.parse_args; run
**Input/output and edge cases:** see Chapters 8 and 11 for the original FIFO load, response schema, defaults, statuses and errors. Shared later-workstream branches are deliberately not expanded here.

### Symbols in src/football_fatigue/board/__init__.py

### Symbols in src/football_fatigue/analytics/__init__.py

### Symbols in src/football_fatigue/analytics/movement.py

#### TrajectoryPoint — src/football_fatigue/analytics/movement.py:L55-L63
A single spatial observation of a player along a trajectory.

~~~python
frame_num: int
video_time_s: float
canonical_period: int
x: float
y: float
is_estimated: bool = False
coord_source: str = 'raw'
~~~

#### DerivedKinematics — src/football_fatigue/analytics/movement.py:L67-L75
Instantaneous and windowed kinematic properties derived at a point.

~~~python
frame_num: int
video_time_s: float
speed_m_s: float
acceleration_m_s2: float | None
is_valid: bool
is_spike: bool
zone: str
~~~

#### MovementCoverage — src/football_fatigue/analytics/movement.py:L79-L105
Data quality and confidence metadata for derived metrics.

~~~python
total_frames_evaluated: int = 0
usable_points: int = 0
estimated_points: int = 0
missing_or_ambiguous_points: int = 0
gaps_detected: int = 0
speed_spikes_rejected: int = 0
~~~
**Methods:** coverage_fraction, estimated_fraction, is_reliable.

#### MovementMetrics — src/football_fatigue/analytics/movement.py:L109-L151
Aggregated movement metrics and workload indicators for a player or segment.

~~~python
total_distance_m: float = 0.0
walking_distance_m: float = 0.0
jogging_distance_m: float = 0.0
running_distance_m: float = 0.0
hsr_distance_m: float = 0.0
sprint_distance_m: float = 0.0
max_speed_m_s: float = 0.0
mean_speed_m_s: float = 0.0
max_acceleration_m_s2: float = 0.0
max_deceleration_m_s2: float = 0.0
high_accel_events: int = 0
high_decel_events: int = 0
coverage: MovementCoverage = field(default_factory=MovementCoverage)
~~~
**Methods:** high_intensity_distance_m, as_dict.

#### MovementCoverage.coverage_fraction — src/football_fatigue/analytics/movement.py:L89-L93

~~~python
def MovementCoverage.coverage_fraction(self) -> float
~~~
**Declared purpose:** Fraction of evaluated frames with usable tracking positions.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** No explicit calls; field/literal/attribute computation.
**Output expressions (selected excerpts):** L93: Return self.usable_points / self.total_frames_evaluated; L92: Return 0.0.
**Important source branches:** L91: self.total_frames_evaluated == 0.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### MovementCoverage.estimated_fraction — src/football_fatigue/analytics/movement.py:L96-L100

~~~python
def MovementCoverage.estimated_fraction(self) -> float
~~~
**Declared purpose:** Fraction of usable tracking points that were estimated by the tracking system.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** No explicit calls; field/literal/attribute computation.
**Output expressions (selected excerpts):** L100: Return self.estimated_points / self.usable_points; L99: Return 0.0.
**Important source branches:** L98: self.usable_points == 0.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### MovementCoverage.is_reliable — src/football_fatigue/analytics/movement.py:L103-L105

~~~python
def MovementCoverage.is_reliable(self) -> bool
~~~
**Declared purpose:** Heuristic check for scientific reliability: at least 60% coverage.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** No explicit calls; field/literal/attribute computation.
**Output expressions (selected excerpts):** L105: Return self.coverage_fraction >= 0.6.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### MovementMetrics.high_intensity_distance_m — src/football_fatigue/analytics/movement.py:L126-L128

~~~python
def MovementMetrics.high_intensity_distance_m(self) -> float
~~~
**Declared purpose:** Combined High-Speed Running and Sprinting distance (>= 5.5 m/s).
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** No explicit calls; field/literal/attribute computation.
**Output expressions (selected excerpts):** L128: Return self.hsr_distance_m + self.sprint_distance_m.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### MovementMetrics.as_dict — src/football_fatigue/analytics/movement.py:L130-L151

~~~python
def MovementMetrics.as_dict(self) -> dict[str, Any]
~~~
**Declared purpose:** Convert metrics to a standard serialisable dictionary.
**Caller candidates:** src/football_fatigue/playback/server.py:load_match_data@L242
**Callees visible in body:** round
**Output expressions (selected excerpts):** L132: Return {'total_distance_m': round(self.total_distance_m, 2), 'walking_distance_m': round(self.walking_distance_m, 2), 'jogging_distance_m': round(self.jogging_distance_m, 2), 'running_distance_m': round(self.running_distance_m, 2), 'hsr_distance_m': round(self.hsr_distance_m, 2), 'sprint_distance_m': round(self.sprint_distance_m, 2), 'high_intensity_distance_m':....
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### classify_speed_zone — src/football_fatigue/analytics/movement.py:L154-L165

~~~python
def classify_speed_zone(speed: float) -> str
~~~
**Declared purpose:** Classify instantaneous or windowed speed into standard football velocity bands.
**Caller candidates:** src/football_fatigue/analytics/movement.py:compute_player_movement@L303
**Callees visible in body:** No explicit calls; field/literal/attribute computation.
**Output expressions (selected excerpts):** L157: Return 'walking'; L159: Return 'jogging'; L161: Return 'running'; L163: Return 'hsr'.
**Important source branches:** L156: speed < ZONE_WALKING_MAX; L158: speed < ZONE_JOGGING_MAX; L160: speed < ZONE_RUNNING_MAX; L162: speed < ZONE_HSR_MAX.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### compute_player_movement — src/football_fatigue/analytics/movement.py:L168-L351

~~~python
def compute_player_movement(points: Sequence[TrajectoryPoint], total_frames_evaluated: int | None=None, max_gap_s: float=DEFAULT_MAX_GAP_S, window_frames: int=3, max_speed_m_s: float=PHYSIOLOGICAL_MAX_SPEED_M_S, max_accel_m_s2: float=PHYSIOLOGICAL_MAX_ACCEL_M_S2) -> tuple[MovementMetrics, list[DerivedKinematics]]
~~~
**Declared purpose:** Compute validated movement metrics for a chronological sequence of points.
**Caller candidates:** src/football_fatigue/analytics/movement.py:summarize_match_movement@L391
**Callees visible in body:** DerivedKinematics; MovementCoverage; MovementMetrics; abs; all_kinematics.append; classify_speed_zone; current_segment.append; len; math.hypot; max; min; range; segments.append; step_dists.append; step_dts.append; step_speeds.append; sum; valid_speeds.append
**Output expressions (selected excerpts):** L351: Return (metrics, all_kinematics); L208: Return (MovementMetrics(coverage=cov), []).
**Important source branches:** L207: len(points) < 2; L228: current_segment; L220: dt <= 0 or dt > max_gap_s or period_changed; L243: len(seg) < 2; L222: current_segment; L282: is_spike; L299: is_accel_spike.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### extract_player_trajectories — src/football_fatigue/analytics/movement.py:L354-L378

~~~python
def extract_player_trajectories(frames: Iterable[PreparedFrame]) -> dict[tuple[str, int], list[TrajectoryPoint]]
~~~
**Declared purpose:** Extract chronological trajectory points for each (source_side, jersey_number) pair.
**Caller candidates:** src/football_fatigue/analytics/movement.py:summarize_match_movement@L387
**Callees visible in body:** TrajectoryPoint; trajectories.setdefault; trajectories.setdefault(key, []).append
**Output expressions (selected excerpts):** L378: Return trajectories.
**Important source branches:** L365: not p.position_usable or p.jersey_number is None.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### summarize_match_movement — src/football_fatigue/analytics/movement.py:L381-L398

~~~python
def summarize_match_movement(frames: Sequence[PreparedFrame], window_frames: int=3) -> dict[tuple[str, int], MovementMetrics]
~~~
**Declared purpose:** Calculate movement metrics for all players across a sequence of PreparedFrames.
**Caller candidates:** src/football_fatigue/playback/server.py:load_match_data@L240
**Callees visible in body:** compute_player_movement; extract_player_trajectories; len; trajectories.items
**Output expressions (selected excerpts):** L398: Return summaries.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

### Symbols in scripts/audit/audit_lib.py

#### AuditError — scripts/audit/audit_lib.py:L40-L41
A required input is missing or ambiguous; the check is reported as failed, never guessed.

#### Aligner — scripts/audit/audit_lib.py:L261-L322
Streaming nearest-frame selection for a list of events (one aligner per event version).
**Methods:** __init__, offer, outcome.

#### locate — scripts/audit/audit_lib.py:L44-L68

~~~python
def locate(root: str | Path, match_id: str, _cache: dict={}) -> dict[str, Any]
~~~
**Declared purpose:** Locate one match's files through the project inventory (explicit about missing/duplicate).
**Caller candidates:** scripts/audit/event_alignment.py:run_match@L47; scripts/audit/event_versions.py:run_match@L88; scripts/audit/extra_time_identity.py:run_match@L44; scripts/audit/tracking_scan.py:scan_match@L38; scripts/audit/tracking_scan.py:show_frame@L203
**Callees visible in body:** AuditError; Path; VERSIONS.items; _cache[root].matches.get; dict; inv.scan; len; mi.events.get; one; str; validate_match_id
**Output expressions (selected excerpts):** L62: Return dict(match_id=str(match_id), tracking=one(mi.tracking, 'tracking'), metadata=one(mi.metadata, 'metadata'), rosters=one(mi.rosters, 'roster'), events=events); L57: Return Path(root) / paths[0].
**Important source branches:** L48: root not in _cache; L51: mi is None; L55: len(paths) != 1.
**Explicit failure paths (excerpts):** L52: AuditError(f'match {match_id}: not found under {root}'); L56: AuditError(f'match {match_id}: {len(paths)} {what} candidates (expected exactly 1): {paths}').

#### locate.one — scripts/audit/audit_lib.py:L54-L57

~~~python
def locate.one(paths: list[str], what: str) -> Path
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** scripts/audit/audit_lib.py:locate@L61; scripts/audit/audit_lib.py:locate@L64; scripts/audit/audit_lib.py:locate@L65; scripts/audit/audit_lib.py:locate@L66
**Callees visible in body:** AuditError; Path; len
**Output expressions (selected excerpts):** L57: Return Path(root) / paths[0].
**Important source branches:** L55: len(paths) != 1.
**Explicit failure paths (excerpts):** L56: AuditError(f'match {match_id}: {len(paths)} {what} candidates (expected exactly 1): {paths}').

#### match_ids — scripts/audit/audit_lib.py:L71-L72

~~~python
def match_ids(root: str | Path) -> list[str]
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** scripts/audit/audit_lib.py:cli_main@L486; src/football_fatigue/playback/server.py:get_matches_catalog@L74
**Callees visible in body:** Path; inv.scan; str
**Output expressions (selected excerpts):** L72: Return inv.scan(str(Path(root))).match_ids.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### load_json — scripts/audit/audit_lib.py:L75-L77

~~~python
def load_json(path: Path) -> Any
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** scripts/audit/audit_lib.py:load_metadata@L81; scripts/audit/event_alignment.py:run_match@L49; scripts/audit/event_versions.py:run_match@L89; scripts/audit/extra_time_identity.py:run_match@L47
**Callees visible in body:** json.load; open
**Output expressions (selected excerpts):** L77: Return json.load(f).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### load_metadata — scripts/audit/audit_lib.py:L80-L86

~~~python
def load_metadata(path: Path) -> dict
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** scripts/audit/event_alignment.py:run_match@L48; scripts/audit/extra_time_identity.py:run_match@L45; scripts/audit/tracking_scan.py:scan_match@L39
**Callees visible in body:** AuditError; isinstance; len; load_json
**Output expressions (selected excerpts):** L86: Return d.
**Important source branches:** L82: isinstance(d, list); L83: len(d) != 1.
**Explicit failure paths (excerpts):** L84: AuditError(f'{path}: metadata list has {len(d)} objects').

#### iter_tracking — scripts/audit/audit_lib.py:L89-L94

~~~python
def iter_tracking(path: Path) -> Iterator[dict]
~~~
**Declared purpose:** Stream tracking records from a .jsonl.bz2 file; nothing is written to disk.
**Caller candidates:** scripts/audit/event_alignment.py:run_match@L55; scripts/audit/extra_time_identity.py:run_match@L76; scripts/audit/tracking_scan.py:scan_match@L58; scripts/audit/tracking_scan.py:show_frame@L204
**Callees visible in body:** bz2.open; json.loads; line.strip
**Output expressions (selected excerpts):** L94: Yield json.loads(line).
**Important source branches:** L93: line.strip().
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### pct — scripts/audit/audit_lib.py:L99-L103

~~~python
def pct(values: list[float], q: float) -> float | None
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** scripts/audit/audit_lib.py:dist_stats@L107; scripts/audit/event_alignment.py:run_match@L113
**Callees visible in body:** int; len; min; sorted
**Output expressions (selected excerpts):** L103: Return v[min(len(v) - 1, int(q * (len(v) - 1)))]; L101: Return None.
**Important source branches:** L100: not values.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### dist_stats — scripts/audit/audit_lib.py:L106-L108

~~~python
def dist_stats(values: list[float]) -> dict
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** scripts/audit/event_alignment.py:run_match@L113; scripts/audit/tracking_scan.py:scan_match@L139
**Callees visible in body:** dict; len; max; pct
**Output expressions (selected excerpts):** L107: Return dict(n=len(values), p50=pct(values, 0.5), p90=pct(values, 0.9), p99=pct(values, 0.99), max=max(values) if values else None).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### dist_xy — scripts/audit/audit_lib.py:L111-L114

~~~python
def dist_xy(a: tuple, b: tuple) -> float | None
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** scripts/audit/audit_lib.py:classify_duplicate_jerseys@L143; scripts/audit/audit_lib.py:compare_positions@L250; scripts/audit/audit_lib.py:coordinate_status@L333; scripts/audit/event_alignment.py:run_match@L102; scripts/audit/extra_time_identity.py:run_match@L61; scripts/audit/extra_time_identity.py:run_match@L62; scripts/audit/extra_time_identity.py:run_match@L92; scripts/audit/tracking_scan.py:scan_match@L124
**Callees visible in body:** math.hypot
**Output expressions (selected excerpts):** L114: Return math.hypot(a[0] - b[0], a[1] - b[1]); L113: Return None.
**Important source branches:** L112: a is None or b is None or None in (a[0], a[1], b[0], b[1]).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### classify_duplicate_jerseys — scripts/audit/audit_lib.py:L119-L146

~~~python
def classify_duplicate_jerseys(entries: list[dict] | None) -> dict
~~~
**Declared purpose:** Classify one player list. Never selects a candidate.
**Caller candidates:** scripts/audit/tracking_scan.py:scan_match@L105; scripts/audit/tracking_scan.py:scan_match@L107; scripts/audit/tracking_scan.py:scan_match@L117; scripts/audit/tracking_scan.py:scan_match@L119; scripts/audit/tracking_scan.py:scan_match@L68; scripts/audit/tracking_scan.py:show_frame@L210; scripts/audit/tracking_scan.py:show_frame@L211
**Callees visible in body:** by.items; by.setdefault; by.setdefault(e.get('jerseyNum'), []).append; dict; dist_xy; e.get; g[i].get; g[j].get; groups.values; len; max; range
**Output expressions (selected excerpts):** L146: Return dict(status='duplicate_conflicting' if conflicting else 'duplicate_identical', groups=groups, max_spread_m=spread); L128: Return dict(status='missing', groups={}, max_spread_m=None); L134: Return dict(status='unique', groups={}, max_spread_m=None).
**Important source branches:** L127: entries is None; L133: not groups; L141: ci != cj; L144: d is not None.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### pair_key — scripts/audit/audit_lib.py:L151-L152

~~~python
def pair_key(row: dict) -> tuple
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** scripts/audit/audit_lib.py:repeated_pair_rows@L200; scripts/audit/event_versions.py:version_summary@L31
**Callees visible in body:** row.get
**Output expressions (selected excerpts):** L152: Return (row.get('gameEventId'), row.get('possessionEventId')).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### signature — scripts/audit/audit_lib.py:L155-L160

~~~python
def signature(row: dict) -> tuple
~~~
**Declared purpose:** Row signature used for matching: ids plus times and types. Not unique on its own.
**Caller candidates:** scripts/audit/audit_lib.py:keyed_rows@L168; scripts/audit/event_versions.py:version_summary@L35
**Callees visible in body:** ge.get; pe.get; row.get
**Output expressions (selected excerpts):** L159: Return (row.get('gameEventId'), row.get('possessionEventId'), row.get('startTime'), ge.get('gameEventType'), pe.get('possessionEventType')).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### keyed_rows — scripts/audit/audit_lib.py:L163-L171

~~~python
def keyed_rows(rows: list[dict]) -> list[tuple]
~~~
**Declared purpose:** (signature, ordinal) per row, where ordinal counts repeats of the same signature in file order.
**Caller candidates:** scripts/audit/audit_lib.py:match_rows@L181
**Callees visible in body:** Counter; out.append; signature
**Output expressions (selected excerpts):** L171: Return out.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### match_rows — scripts/audit/audit_lib.py:L174-L195

~~~python
def match_rows(rows_a: list[dict], rows_b: list[dict]) -> dict
~~~
**Declared purpose:** Collision-safe one-to-one matching of rows between two versions of the same match.
**Caller candidates:** scripts/audit/event_versions.py:compare_versions@L44
**Callees visible in body:** Counter; ca.keys; cb.keys; dict; enumerate; index_b.get; keyed_rows; len; pairs.append; range; set; sorted; unmatched_a.append; used_b.add
**Output expressions (selected excerpts):** L195: Return dict(pairs=pairs, unmatched_a=unmatched_a, unmatched_b=unmatched_b, ambiguous_signatures=ambiguous).
**Important source branches:** L187: j is None.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### repeated_pair_rows — scripts/audit/audit_lib.py:L198-L201

~~~python
def repeated_pair_rows(rows: list[dict]) -> int
~~~
**Declared purpose:** Rows whose (gameEventId, possessionEventId) pair occurs more than once in the file.
**Caller candidates:** scripts/audit/event_versions.py:version_summary@L33
**Callees visible in body:** Counter; c.values; pair_key; sum
**Output expressions (selected excerpts):** L201: Return sum((n for n in c.values() if n > 1)).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### flatten — scripts/audit/audit_lib.py:L207-L221

~~~python
def flatten(row: dict) -> dict[str, Any]
~~~
**Declared purpose:** Leaf paths of an event row, excluding player/ball position lists (compared separately).
**Caller candidates:** scripts/audit/audit_lib.py:compare_fields@L230
**Callees visible in body:** isinstance; row.items; v.items; walk
**Output expressions (selected excerpts):** L221: Return out.
**Important source branches:** L212: isinstance(v, dict); L219: k not in EXCLUDED_TOP.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### flatten.walk — scripts/audit/audit_lib.py:L211-L216

~~~python
def flatten.walk(v: Any, p: str) -> None
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** scripts/audit/audit_lib.py:flatten@L214; scripts/audit/audit_lib.py:flatten@L220; src/football_fatigue/inventory.py:scan@L239
**Callees visible in body:** isinstance; v.items; walk
**Output:** no explicit return/yield; mutation, exception/command action, or implicit None. See signature and module context.
**Important source branches:** L212: isinstance(v, dict).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### compare_fields — scripts/audit/audit_lib.py:L224-L233

~~~python
def compare_fields(a: dict, b: dict) -> dict
~~~
**Declared purpose:** Common-field comparison of two matched rows.
**Caller candidates:** scripts/audit/event_versions.py:compare_versions@L54
**Callees visible in body:** dict; fa.keys; fb.keys; flatten; len; sorted
**Output expressions (selected excerpts):** L233: Return dict(changed=changed, only_a=sorted(fa.keys() - fb.keys()), only_b=sorted(fb.keys() - fa.keys()), n_common=len(common)).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### player_map — scripts/audit/audit_lib.py:L236-L241

~~~python
def player_map(row: dict) -> dict[tuple, dict]
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** scripts/audit/audit_lib.py:compare_positions@L245; scripts/audit/event_alignment.py:run_match@L95
**Callees visible in body:** p.get; row.get
**Output expressions (selected excerpts):** L241: Return out.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### compare_positions — scripts/audit/audit_lib.py:L244-L256

~~~python
def compare_positions(a: dict, b: dict) -> dict
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** scripts/audit/event_versions.py:compare_versions@L64
**Callees visible in body:** a.get; b.get; dict; dist_xy; len; max; pa.keys; pa[k].get; pb.keys; pb[k].get; player_map; set
**Output expressions (selected excerpts):** L255: Return dict(players_compared=len(both), player_set_differs=set(pa) != set(pb), xy_changed=xy_changed, max_xy_diff_m=max_d, ball_differs=(a.get('ball') or []) != (b.get('ball') or [])).
**Important source branches:** L251: d is None or d > 0; L253: d is not None.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### Aligner.__init__ — scripts/audit/audit_lib.py:L276-L285

~~~python
def Aligner.__init__(self, events: list[tuple[int, float | None, Any]], fps: float)
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** src/football_fatigue/inventory.py:InventoryScanError.__init__@L94
**Callees visible in body:** self.events.sort
**Output:** no explicit return/yield; mutation, exception/command action, or implicit None. See signature and module context.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### Aligner.offer — scripts/audit/audit_lib.py:L287-L305

~~~python
def Aligner.offer(self, frame_num: int, t_s: float, period: Any, payload: Callable[[], Any]) -> None
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** scripts/audit/event_alignment.py:run_match@L72
**Callees visible in body:** abs; len; max; min; payload; self.best.get; self.best_any.get; self.period_range.setdefault
**Output:** no explicit return/yield; mutation, exception/command action, or implicit None. See signature and module context.
**Important source branches:** L297: a <= self.tol + TIE_EPSILON_S; L299: cur_any is None or a < cur_any[0] - TIE_EPSILON_S; L301: ep == period; L303: cur is None or a < cur[0] - TIE_EPSILON_S.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### Aligner.outcome — scripts/audit/audit_lib.py:L307-L322

~~~python
def Aligner.outcome(self, idx: int, event_time: float | None, event_period: Any) -> tuple[str, str | None]
~~~
**Declared purpose:** ('matched', None) or ('unmatched', reason).
**Caller candidates:** scripts/audit/event_alignment.py:run_match@L82
**Callees visible in body:** self.period_range.get
**Output expressions (selected excerpts):** L322: Return ('unmatched', 'no_frame_within_tolerance'); L310: Return ('unmatched', 'missing_event_time'); L312: Return ('matched', None); L314: Return ('unmatched', 'invalid_source_period').
**Important source branches:** L309: event_time is None; L311: idx in self.best; L313: event_period not in (1, 2, 3, 4); L315: idx in self.best_any; L318: rng is None; L320: event_time < rng[0] - self.tol - TIE_EPSILON_S or event_time > rng[1] + self.tol + TIE_EPSILON_S.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### coordinate_status — scripts/audit/audit_lib.py:L325-L342

~~~python
def coordinate_status(event_players: dict, tracking_players: dict) -> dict
~~~
**Declared purpose:** Coordinate agreement of one event row with the chosen frame (raw tracking positions).
**Caller candidates:** scripts/audit/event_alignment.py:run_match@L95
**Callees visible in body:** dict; dist_xy; event_players.items; len; p.get; sum
**Output expressions (selected excerpts):** L341: Return dict(status=status, compared=n, exact=sum((1 for x in d if x <= COORD_EXACT_M)), near=sum((1 for x in d if x <= COORD_NEAR_M)), distances=d).
**Important source branches:** L336: n < COORD_MIN_COMPARED.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### _worker — scripts/audit/audit_lib.py:L347-L357

~~~python
def _worker(args: tuple) -> tuple[str, dict]
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** scripts/audit/audit_lib.py:run_matches@L426
**Callees visible in body:** dict; func; round; time.time; type
**Output expressions (selected excerpts):** L357: Return (mid, res).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### validate_match_id — scripts/audit/audit_lib.py:L363-L367

~~~python
def validate_match_id(mid: object) -> str
~~~
**Declared purpose:** Accept only a canonical match id (ASCII digits). Rejects separators, traversal, absolute paths.
**Caller candidates:** scripts/audit/audit_lib.py:cli_main@L486; scripts/audit/audit_lib.py:locate@L46; scripts/audit/audit_lib.py:run_matches@L406; scripts/audit/audit_lib.py:safe_destination@L389; scripts/audit/tracking_scan.py:main@L225; src/football_fatigue/ingestion/loaders.py:Dataset._match@L137; src/football_fatigue/ingestion/validation.py:quality_report@L113; src/football_fatigue/ingestion/validation.py:safe_output_path@L211; src/football_fatigue/match_data.py:MatchData.__init__@L260; src/football_fatigue/synchronization.py:synchronize_match@L344
**Callees visible in body:** AuditError; MATCH_ID_RE.fullmatch; isinstance
**Output expressions (selected excerpts):** L367: Return mid.
**Important source branches:** L365: not isinstance(mid, str) or not MATCH_ID_RE.fullmatch(mid).
**Explicit failure paths (excerpts):** L366: AuditError(f'invalid match id {mid!r}: expected 1-9 ASCII digits').

#### _inside — scripts/audit/audit_lib.py:L370-L371

~~~python
def _inside(path: Path, parent: Path) -> bool
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** scripts/audit/audit_lib.py:guard_output_dir@L378; scripts/audit/audit_lib.py:safe_destination@L395
**Callees visible in body:** No explicit calls; field/literal/attribute computation.
**Output expressions (selected excerpts):** L371: Return path == parent or parent in path.parents.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### guard_output_dir — scripts/audit/audit_lib.py:L374-L380

~~~python
def guard_output_dir(root: str | Path, out_dir: Path) -> Path
~~~
**Declared purpose:** Resolve out_dir (following symlinks/junctions) and refuse any location inside the raw root.
**Caller candidates:** scripts/audit/audit_lib.py:cli_main@L487; scripts/audit/audit_lib.py:run_matches@L407; scripts/audit/audit_lib.py:safe_destination@L390
**Callees visible in body:** AuditError; Path; Path(out_dir).resolve; Path(root).resolve; _inside
**Output expressions (selected excerpts):** L380: Return out_dir.
**Important source branches:** L378: _inside(out_dir, raw).
**Explicit failure paths (excerpts):** L379: AuditError(f'refusing to write inside the raw data root: {out_dir}').

#### safe_destination — scripts/audit/audit_lib.py:L383-L397

~~~python
def safe_destination(root: str | Path, out_dir: Path, mid: str, suffix: str='.json') -> Path
~~~
**Declared purpose:** Final write path for one match, validated immediately before use (also for temporary files).
**Caller candidates:** scripts/audit/audit_lib.py:_save@L437; scripts/audit/audit_lib.py:_save@L438; scripts/audit/audit_lib.py:_save@L441; scripts/audit/audit_lib.py:run_matches@L411
**Callees visible in body:** AuditError; Path; Path(root).resolve; _inside; dest.is_symlink; dest.resolve; guard_output_dir; validate_match_id
**Output expressions (selected excerpts):** L397: Return dest.
**Important source branches:** L392: dest.is_symlink(); L395: resolved.parent != out_dir or _inside(resolved, Path(root).resolve()).
**Explicit failure paths (excerpts):** L393: AuditError(f'refusing to write through a symlink: {dest}'); L396: AuditError(f'destination escapes the audit output directory: {dest}').

#### run_matches — scripts/audit/audit_lib.py:L400-L433

~~~python
def run_matches(func: Callable[[str, str], dict], root: str, mids: list[str], out_dir: Path, workers: int, force: bool) -> dict
~~~
**Declared purpose:** Run func(root, match_id) per match; one JSON file per match; existing files are kept unless force.
**Caller candidates:** scripts/audit/audit_lib.py:cli_main@L489
**Callees visible in body:** _save; _worker; dict; guard_output_dir; len; multiprocessing.Pool; out_dir.mkdir; pool.imap_unordered; print; res.get; results.append; safe_destination; target.exists; time.time; todo.append; validate_match_id
**Output expressions (selected excerpts):** L433: Return dict(attempted=len(todo), failed=failed).
**Important source branches:** L419: workers > 1 and len(todo) > 1; L412: target.exists() and (not force); L430: not res.get('ok').
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### _save — scripts/audit/audit_lib.py:L436-L443

~~~python
def _save(root: str | Path, out_dir: Path, mid: str, res: dict) -> None
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** scripts/audit/audit_lib.py:run_matches@L423; scripts/audit/audit_lib.py:run_matches@L428
**Callees visible in body:** json.dump; open; os.replace; print; res.get; safe_destination
**Output:** no explicit return/yield; mutation, exception/command action, or implicit None. See signature and module context.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### load_outputs — scripts/audit/audit_lib.py:L446-L453

~~~python
def load_outputs(out_dir: Path) -> dict[str, dict]
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** scripts/audit/audit_lib.py:cli_main@L490
**Callees visible in body:** Path; Path(out_dir).glob; dict; json.loads; p.read_text; sorted
**Output expressions (selected excerpts):** L453: Return out.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### check_outputs — scripts/audit/audit_lib.py:L456-L465

~~~python
def check_outputs(results: dict[str, dict], requested: list[str]) -> dict
~~~
**Declared purpose:** Requested matches that failed, have no output, or have incomplete output -> {match_id: reason}.
**Caller candidates:** scripts/audit/audit_lib.py:cli_main@L491
**Callees visible in body:** r.get; results.get
**Output expressions (selected excerpts):** L465: Return bad.
**Important source branches:** L461: r is None; L463: not r.get('ok').
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### add_common_args — scripts/audit/audit_lib.py:L468-L474

~~~python
def add_common_args(ap) -> None
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** scripts/audit/audit_lib.py:cli_main@L483
**Callees visible in body:** ap.add_argument
**Output:** no explicit return/yield; mutation, exception/command action, or implicit None. See signature and module context.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### cli_main — scripts/audit/audit_lib.py:L477-L504

~~~python
def cli_main(doc: str, default_out: str, func: Callable[[str, str], dict], aggregate: Callable[[dict], None], argv: list[str] | None=None) -> int
~~~
**Declared purpose:** Shared CLI body. Exit status: 0 only if every requested match has complete, successful output; 1 if any requested match failed or its output is missing/incomplete; 2 for invalid input or unsafe output paths (controlled error, nothing written).
**Caller candidates:** scripts/audit/event_alignment.py:main@L146; scripts/audit/event_versions.py:main@L137; scripts/audit/extra_time_identity.py:main@L134; scripts/audit/tracking_scan.py:main@L229
**Callees visible in body:** Path; add_common_args; aggregate; any; ap.parse_args; argparse.ArgumentParser; check_outputs; guard_output_dir; len; load_outputs; match_ids; print; r.get; requested.values; run_matches; validate_match_id
**Output expressions (selected excerpts):** L504: Return 0; L502: Return 1; L499: Return 2.
**Important source branches:** L500: failures; L488: not a.aggregate; L493: any((r.get('ok') for r in requested.values())).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

### Symbols in scripts/audit/event_versions.py

#### version_summary — scripts/audit/event_versions.py:L27-L40

~~~python
def version_summary(rows: list[dict]) -> dict
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** scripts/audit/event_versions.py:run_match@L90
**Callees visible in body:** (r.get('gameEvents') or {}).get; (r.get('possessionEvents') or {}).get; Counter; L.pair_key; L.repeated_pair_rows; L.signature; dict; ge.items; keys.values; len; pe.items; periods.items; r.get; sorted; str; sum
**Output expressions (selected excerpts):** L32: Return dict(rows=len(rows), distinct_key_pairs=len(keys), rows_in_repeated_key_pairs=L.repeated_pair_rows(rows), repeated_key_pairs=sum((1 for n in keys.values() if n > 1)), distinct_signatures=len({L.signature(r) for r in rows}), game_event_types={str(k): v for k, v in ge.items()}, possession_event_types={str(k): v for k, v in pe.items()}, periods={str(k): v fo....
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### compare_versions — scripts/audit/event_versions.py:L43-L84

~~~python
def compare_versions(a: list[dict], b: list[dict]) -> dict
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** scripts/audit/event_versions.py:run_match@L92
**Callees visible in body:** (a[i].get('gameEvents') or {}).get; (a[i].get('possessionEvents') or {}).get; (b[j].get('gameEvents') or {}).get; (b[j].get('possessionEvents') or {}).get; Counter; L.compare_fields; L.compare_positions; L.match_rows; a[i].get; b[j].get; dict; len; max; ptype_a.items; ptype_b.items; str; type_a.items; type_b.items
**Output expressions (selected excerpts):** L76: Return dict(rows_a=len(a), rows_b=len(b), matched_pairs=len(m['pairs']), unmatched_a=len(m['unmatched_a']), unmatched_b=len(m['unmatched_b']), ambiguous_signatures=len(m['ambiguous_signatures']), unmatched_a_possession_types={str(k): v for k, v in ptype_a.items()}, unmatched_b_possession_types={str(k): v for k, v in ptype_b.items()}, unmatched_a_game_types={str(....
**Important source branches:** L55: c['changed'].
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### run_match — scripts/audit/event_versions.py:L87-L93

~~~python
def run_match(root: str, mid: str) -> dict
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** L.load_json; L.locate; compare_versions; dict; f['events'].items; rows.items; version_summary
**Output expressions (selected excerpts):** L93: Return out.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### aggregate — scripts/audit/event_versions.py:L96-L133

~~~python
def aggregate(res: dict[str, dict]) -> None
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** scripts/audit/audit_lib.py:cli_main@L494
**Callees visible in body:** Counter; c.update; ch.most_common; ch.update; dict; len; max; min; oa.most_common; oa.update; ob.most_common; ob.update; ok.values; p.update; pa.update; pb.update; print; r.get; res.items; sum; t; x['positions'].get; x['positions'].items
**Output:** no explicit return/yield; mutation, exception/command action, or implicit None. See signature and module context.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### main — scripts/audit/event_versions.py:L136-L137

~~~python
def main() -> int
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** L.cli_main
**Output expressions (selected excerpts):** L137: Return L.cli_main(__doc__, 'data/processed/audit_1b/event_versions', run_match, aggregate).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

### Symbols in scripts/audit/tracking_scan.py

#### bucket — scripts/audit/tracking_scan.py:L30-L34

~~~python
def bucket(d: float) -> str
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** scripts/audit/tracking_scan.py:scan_match@L120
**Callees visible in body:** No explicit calls; field/literal/attribute computation.
**Output expressions (selected excerpts):** L34: Return SPREAD_BUCKETS[-1][0]; L33: Return name.
**Important source branches:** L32: d <= hi.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### scan_match — scripts/audit/tracking_scan.py:L37-L140

~~~python
def scan_match(root: str, mid: str) -> dict
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** Counter; L.classify_duplicate_jerseys; L.dist_stats; L.dist_xy; L.iter_tracking; L.load_metadata; L.locate; abs; bs.get; bucket; dict; e.get; examples.append; float; len; max; periods.setdefault; r.get; rawmap.get; sc['groups'].items; sm_group_sizes.items; sm_len.items; sm_raw_dist.extend; str; sum; xtab.items
**Output expressions (selected excerpts):** L132: Return dict(frame_crosstab={f'{a}|{b}': n for (a, b), n in xtab.items()}, fps=fps, counts=dict(c), periods=periods, max_abs_videoTimeMs_minus_frame_formula_ms=max_time_formula_err_ms, visibility=dict(visibility), confidence=dict(confidence), raw_list_status=dict(raw_status), repeat_record_smoothed_list_status=dict(repeat_sm_status), smoothed_list_status=dict(sm_....
**Important source branches:** L61: prev is not None and fn == prev['fn']; L71: prev is not None; L74: prev is not None; L93: bs is not None; L75: per == prev['per']; L110: sm is not None; L77: step < 0 or vt < prev['vt'].
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### aggregate — scripts/audit/tracking_scan.py:L143-L199

~~~python
def aggregate(res: dict[str, dict]) -> None
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** scripts/audit/audit_lib.py:cli_main@L494
**Callees visible in body:** Counter; S; conf.update; dict; len; max; min; mm.values; ok.items; ok.values; print; r.get; r['counts'].get; r['smoothed_list_status'].get; r['visibility'].get; r['visibility'].values; res.items; rp.update; rs.update; sg.update; sl.update; sorted; spd.update; ss.update; ss.values; sum; sz.update; tuple; vis.update; vr.get; vr.update; xt.update
**Output expressions (selected excerpts):** L148: Return None.
**Important source branches:** L147: not ok; L195: dm.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### show_frame — scripts/audit/tracking_scan.py:L202-L217

~~~python
def show_frame(root: str, mid: str, frame: int) -> int
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** scripts/audit/tracking_scan.py:main@L225
**Callees visible in body:** '; '.join; L.classify_duplicate_jerseys; L.iter_tracking; L.locate; cl['groups'].items; int; len; print; sorted
**Output expressions (selected excerpts):** L217: Return 1; L215: Return 0.
**Important source branches:** L205: r['frameNum'] != frame.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### main — scripts/audit/tracking_scan.py:L220-L229

~~~python
def main() -> int
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** L.cli_main; L.validate_match_id; ap.add_argument; ap.parse_args; argparse.ArgumentParser; len; print; show_frame
**Output expressions (selected excerpts):** L229: Return L.cli_main(__doc__, 'data/processed/audit_1b/tracking_scan', scan_match, aggregate); L225: Return show_frame(a.root, L.validate_match_id(a.match), a.frame); L228: Return 2.
**Important source branches:** L221: len(sys.argv) > 1 and sys.argv[1] == 'show-frame'.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

### Symbols in scripts/audit/event_alignment.py

#### jersey — scripts/audit/event_alignment.py:L30-L34

~~~python
def jersey(v)
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** scripts/audit/event_alignment.py:frame_snapshot@L41
**Callees visible in body:** int
**Output expressions (selected excerpts):** L32: Return int(v); L34: Return v.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### frame_snapshot — scripts/audit/event_alignment.py:L37-L43

~~~python
def frame_snapshot(r: dict) -> dict
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** scripts/audit/event_alignment.py:run_match@L67
**Callees visible in body:** b.get; dict; e.get; jersey; r.get
**Output expressions (selected excerpts):** L43: Return dict(players=players, ball=(b.get('x'), b.get('y'), b.get('z')) if b else None).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### run_match — scripts/audit/event_alignment.py:L46-L116

~~~python
def run_match(root: str, mid: str) -> dict
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** (r.get('gameEvents') or {}).get; Counter; L.Aligner; L.coordinate_status; L.dist_stats; L.dist_xy; L.iter_tracking; L.load_json; L.load_metadata; L.locate; L.pct; L.player_map; a.offer; a.outcome; al.values; by_period.items; by_period.setdefault; dict; eb.get; enumerate; f['events'].items; float; frame_snapshot; len; memo.append; r.get; resid_ms.append; resid_signed.append; rows.items; str
**Output expressions (selected excerpts):** L116: Return out; L68: Return memo[0].
**Important source branches:** L58: fn == prev_fn; L66: not memo; L86: o == 'unmatched'; L101: eb and tb; L92: ad * 1000 <= hi + 1e-09; L105: eb and (not tb); L107: tb and (not eb).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### aggregate — scripts/audit/event_alignment.py:L119-L142

~~~python
def aggregate(res: dict[str, dict]) -> None
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** scripts/audit/audit_lib.py:cli_main@L494
**Callees visible in body:** Counter; bk.update; bl.update; cs.update; dict; len; max; min; ok.items; ok.values; pl.update; print; r.get; r['tracking'].get; r['versions'][v]['coordinate_status'].get; r['versions'][v]['temporal'].get; res.items; rs.update; sorted; sum; unm.items; x['temporal'].get
**Output expressions (selected excerpts):** L124: Return None.
**Important source branches:** L123: not ok.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### main — scripts/audit/event_alignment.py:L145-L146

~~~python
def main() -> int
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** L.cli_main
**Output expressions (selected excerpts):** L146: Return L.cli_main(__doc__, 'data/processed/audit_1b/event_alignment', run_match, aggregate).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

### Symbols in scripts/audit/extra_time_identity.py

#### med — scripts/audit/extra_time_identity.py:L33-L34

~~~python
def med(v)
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** scripts/audit/extra_time_identity.py:run_match@L70; scripts/audit/extra_time_identity.py:run_match@L99
**Callees visible in body:** round; statistics.median
**Output expressions (selected excerpts):** L34: Return round(statistics.median(v), 2) if v else None.
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### flag — scripts/audit/extra_time_identity.py:L37-L40

~~~python
def flag(n, own, opp)
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** scripts/audit/extra_time_identity.py:run_match@L100; scripts/audit/extra_time_identity.py:run_match@L71
**Callees visible in body:** No explicit calls; field/literal/attribute computation.
**Output expressions (selected excerpts):** L40: Return 'no_anomaly_detected' if n >= MIN_N else 'insufficient_events'; L39: Return 'suspected_side_identity_anomaly'.
**Important source branches:** L38: n >= MIN_N and own is not None and (opp is not None) and (own > OWN_FAR_M) and (opp < OPP_NEAR_M).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### run_match — scripts/audit/extra_time_identity.py:L43-L103

~~~python
def run_match(root: str, mid: str) -> dict
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** (r.get('gameEvents') or {}).get; Counter; L.dist_xy; L.iter_tracking; L.load_json; L.load_metadata; L.locate; a['opp'].append; a['own'].append; acc.items; acc.setdefault; any; dict; e.get; flag; ge.get; int; len; med; min; p.get; pe.get; per_rows.items; r.get; r['ball'][0].get; r['balls'][0].get; round; seen.add; set; sorted; str; sum; t_acc.items; t_acc.setdefault
**Output expressions (selected excerpts):** L103: Return out.
**Important source branches:** L73: out['has_extra_time_events']; L52: ge.get('gameEventType') != 'OTB' or not pe.get('possessionEventType') or pe.get('possessionEventType') == 'IT'; L54: ge.get('teamId') is None or ge.get('playerId') is None or (not r.get('ball')); L59: not own; L64: d_own is None or not d_opp; L77: r['frameNum'] == prev; L83: not ge or pid is None or pid in seen or (ge.get('game_event_type') != 'OTB') or (ge.get('team_id') is None) or (not r['balls']).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### aggregate — scripts/audit/extra_time_identity.py:L106-L130

~~~python
def aggregate(res: dict[str, dict]) -> None
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** scripts/audit/audit_lib.py:cli_main@L494
**Callees visible in body:** (r.get('tracking') or {}).items; Counter; any; dict; et.items; len; max; ok.items; print; r.get; r['event_rows_by_period'].items; r['events'].items; res.items; sorted
**Output:** no explicit return/yield; mutation, exception/command action, or implicit None. See signature and module context.
**Important source branches:** L124: '3' not in (r.get('tracking_frames_by_period') or {}); L128: any((k not in ('1', '2', '3', '4') for k in r['event_rows_by_period'])); L118: k in ('3', '4'); L122: k in ('3', '4').
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

#### main — scripts/audit/extra_time_identity.py:L133-L134

~~~python
def main() -> int
~~~
**Purpose/context:** helper or accessor within the module responsibility in Appendix A; the algorithm is traced in its chapter.
**Caller candidates:** Public/CLI/framework call, parent helper, property access, or no explicit lexical call in the inspected source manifest.
**Callees visible in body:** L.cli_main
**Output expressions (selected excerpts):** L134: Return L.cli_main(__doc__, 'data/processed/audit_1b/extra_time_identity', run_match, aggregate).
**Failure behavior:** no explicit raise in this body; failures may propagate from listed callees. Pure dataclass accessors simply expose their stored field.

### Original browser function index
JavaScript signatures and starting lines are literal source evidence. Callback bindings are explained in Chapters 9–11. This index deliberately omits drawObservationHeatmap and drawTeamShape introduced after the original baseline.
- function showToast(msg) — src/football_fatigue/board/index.html:L908.
- function periodLabel(p) — src/football_fatigue/board/index.html:L917.
- function periodShort(p) — src/football_fatigue/board/index.html:L925.
- function resizeCanvas() — src/football_fatigue/board/index.html:L937.
- function toCanvas(x_m, y_m) — src/football_fatigue/board/index.html:L959.
- function drawPitch() — src/football_fatigue/board/index.html:L967.
- function drawPlayer(p) — src/football_fatigue/board/index.html:L1004.
- function drawBall(ball) — src/football_fatigue/board/index.html:L1027.
- function render() — src/football_fatigue/board/index.html:L1097.
- function updateScoreboard(frame) — src/football_fatigue/board/index.html:L1112.
- function updateHUD(frame) — src/football_fatigue/board/index.html:L1131.
- function updatePlayerList(players) — src/football_fatigue/board/index.html:L1160.
- function showPlayerAnalytics(player, metrics) — src/football_fatigue/board/index.html:L1194.
- function eventLabel(ev) — src/football_fatigue/board/index.html:L1228.
- function eventColor(ev) — src/football_fatigue/board/index.html:L1238.
- function eventMatchesFilter(ev, filter) — src/football_fatigue/board/index.html:L1252.
- function buildEventList() — src/football_fatigue/board/index.html:L1260.
- function highlightEventItem(eid) — src/football_fatigue/board/index.html:L1307.
- function showEventDetail(ev) — src/football_fatigue/board/index.html:L1315.
- function jumpToFrame(frameNum) — src/football_fatigue/board/index.html:L1343.
- function jumpToIdx(idx) — src/football_fatigue/board/index.html:L1354.
- function step(delta) — src/football_fatigue/board/index.html:L1360.
- function tick(ts) — src/football_fatigue/board/index.html:L1362.
- function play() — src/football_fatigue/board/index.html:L1379.
- function pause() — src/football_fatigue/board/index.html:L1389.
- function pauseIfPlaying() — src/football_fatigue/board/index.html:L1396.
- function filteredMatchedEvents() — src/football_fatigue/board/index.html:L1398.
- function goNextEvent() — src/football_fatigue/board/index.html:L1402.
- function goPrevEvent() — src/football_fatigue/board/index.html:L1417.
- function openSetupModal() — src/football_fatigue/board/index.html:L1512.
- function closeSetupModal() — src/football_fatigue/board/index.html:L1526.
- async function fetchJSON(url) — src/football_fatigue/board/index.html:L1530.
- async function loadMatchesCatalog() — src/football_fatigue/board/index.html:L1536.
- async function launchSelectedMatch() — src/football_fatigue/board/index.html:L1573.
- async function loadData() — src/football_fatigue/board/index.html:L1646.

## Appendix C — Complete original-system test evidence index
**Not executed.** Each entry is a test definition inspected statically, not a passing result. Selected assertions below are actual unevaluated expressions. Fixtures and helper methods can introduce setup effects; this is why the suite was not run under the task restriction.

### tests/test_package_installation.py — 1 test definitions
**Dependencies:** importlib.metadata: distribution; importlib: import_module.

- **test_installed_distribution_is_importable** — tests/test_package_installation.py:L7-L11.
  Selected assertions: L10: package.__file__ is not None; L11: installed.metadata['Name'] == 'football-fatigue'.

### tests/test_inventory.py — 36 test definitions
**Dependencies:** builtins; football_fatigue: inventory; io; json; os; pytest; stat; subprocess; sys.

- **test_complete_match_has_no_issues** — tests/test_inventory.py:L48-L54.
  Selected assertions: L51: result.match_ids == ['100']; L52: result.coverage() == {'tracking': 1, 'events': 1, 'metadata': 1, 'rosters': 1}.

- **test_missing_modality_reported_without_crashing** — tests/test_inventory.py:L57-L66.
  Selected assertions: L61: result.match_ids == ['100', '200']; L63: ('200', 'no tracking file') in missing.

- **test_event_versions_stay_separate_and_missing_version_flagged** — tests/test_inventory.py:L69-L83.
  Selected assertions: L73: result.event_versions == [ROOT_VERSION, 'March 14, 2025', 'May 1, 2025']; L74: result.event_version_coverage() == {ROOT_VERSION: 2, 'March 14, 2025': 1, 'May 1, 2025': 2}.

- **test_split_download_parts_and_timestamps_combine** — tests/test_inventory.py:L86-L97.
  Selected assertions: L92: result.match_ids == ['1', '2']; L93: result.coverage()[inv.TRACKING] == 2.

- **test_empty_directory** — tests/test_inventory.py:L100-L103.
  Selected assertions: L102: result.matches == {} and result.issues == [] and (result.event_versions == []); L103: 'Match IDs discovered: 0' in inv.format_report(result).

- **test_missing_root_raises_and_cli_returns_error** — tests/test_inventory.py:L106-L110.
  Selected assertions: L109: inv.main([str(tmp_path / 'absent')]) == 2; L110: 'not a directory' in capsys.readouterr().err.
  Error expectations: pytest.raises(NotADirectoryError).

- **test_match_ids_sort_numerically** — tests/test_inventory.py:L113-L116.
  Selected assertions: L116: inv.scan(tmp_path).match_ids == ['3812', '3859', '10502'].

- **test_unknown_and_non_match_files** — tests/test_inventory.py:L119-L137.
  Selected assertions: L128: unknown_paths(result) == {'notes.txt', f'Event Data-{TS}-1-001/Event Data/readme.json', 'Elsewhere/100.json', f'Tracking Data-{TS}-1-001/Tracking Data/100.json'}; L134: [f.path for f in result.files_of_kind(inv.REFERENCE)] == ['competitions.csv'].

- **test_duplicate_candidates_for_same_match_and_version** — tests/test_inventory.py:L143-L155.
  Selected assertions: L149: {i.detail for i in duplicates} == {'2 candidates for events version V1', '2 tracking candidates'}; L153: all((len(i.paths) == 2 for i in duplicates)).

- **test_duplicate_metadata_and_roster_candidates_across_split_downloads** — tests/test_inventory.py:L161-L170.
  Selected assertions: L167: len(getattr(result.matches['5'], modality)) == 2; L169: duplicate.detail == f'2 {modality} candidates' and duplicate.match_id == '5'.

- **test_conflicting_family_directories_are_not_assigned** — tests/test_inventory.py:L185-L191.
  Selected assertions: L188: result.matches == {}; L189: unknown_paths(result) == {relative}.

- **test_unsupported_nesting_is_reported_not_guessed** — tests/test_inventory.py:L205-L210.
  Selected assertions: L208: result.matches == {}; L210: record.kind == inv.UNKNOWN and 'unsupported nesting' in record.note.

- **test_supported_layouts** — tests/test_inventory.py:L213-L221.
  Selected assertions: L219: result.issues != [] and (not issues(result, 'unknown_file')); L220: set(result.matches['1'].events) == {ROOT_VERSION, 'V', 'W'}.

- **test_strict_filename_matching** — tests/test_inventory.py:L240-L242.
  Selected assertions: L242: kind == inv.UNKNOWN and match_id is None and note.

- **test_filename_matching_accepts_exact_names_only** — tests/test_inventory.py:L245-L247.
  Selected assertions: L246: inv.classify((f'Metadata-{TS}-1-001', 'Metadata', '42.json'))[:2] == ('metadata', '42'); L247: inv.classify(('Tracking Data', '42.JSONL.BZ2'))[:2] == ('tracking', '42').

- **test_root_version_distinct_from_literal_root_folder** — tests/test_inventory.py:L253-L273.
  Selected assertions: L258: set(events) == {None, '(root)'}; L259: len(events[None]) == 1 and len(events['(root)']) == 1.

- **test_json_includes_file_level_metadata** — tests/test_inventory.py:L279-L302.
  Selected assertions: L284: inv.main([str(tmp_path), '--format', 'json']) == 0; L288: tracking == {'path': f'Tracking Data-{TS}-1-001/Tracking Data/5.jsonl.bz2', 'kind': 'tracking', 'size': 5, 'match_id': '5', 'event_version': None, 'note': None}.

- **test_json_match_view_includes_files** — tests/test_inventory.py:L305-L312.
  Selected assertions: L307: inv.main([str(tmp_path), '--format', 'json', '--match', '3812']) == 0; L309: data['found'] is True and len(data['files']) == 5.

- **test_cli_text_match_view** — tests/test_inventory.py:L315-L321.
  Selected assertions: L317: inv.main([str(tmp_path), '--match', '3812']) == 0; L319: 'Match 3812' in out and '[V1]' in out and ('[(root)]' in out) and ('MISSING' not in out).

- **test_enumeration_error_fails_the_scan** — tests/test_inventory.py:L327-L350.
  Selected assertions: L343: any(('cannot list directory' in e and 'Metadata-' in e for e in excinfo.value.errors)); L345: inv.main([str(tmp_path)]) == 3.
  Error expectations: pytest.raises(inv.InventoryScanError).

- **test_file_metadata_error_fails_the_scan** — tests/test_inventory.py:L360-L378.
  Selected assertions: L373: 'cannot read file metadata' in message and '100.jsonl.bz2' in message; L374: str(tmp_path) not in message.
  Error expectations: pytest.raises(inv.InventoryScanError).

- **test_all_scan_errors_are_collected** — tests/test_inventory.py:L381-L393.
  Selected assertions: L393: len(excinfo.value.errors) == 4.
  Error expectations: pytest.raises(inv.InventoryScanError).

- **test_nested_download_wrappers_are_not_confident_matches** — tests/test_inventory.py:L408-L417.
  Selected assertions: L414: result.matches == {} and result.coverage() == {m: 0 for m in inv.MATCH_MODALITIES}; L415: unknown_paths(result) == {nested, nested_with_inner}.

- **test_legitimate_drive_layout_still_accepted_with_bare_inner_folder** — tests/test_inventory.py:L420-L430.
  Selected assertions: L429: result.coverage() == {m: 1 for m in inv.MATCH_MODALITIES}; L430: not issues(result, 'unknown_file').

- **test_normal_directory_root_is_accepted** — tests/test_inventory.py:L436-L439.
  Selected assertions: L438: inv.scan(tmp_path).match_ids == ['100']; L439: inv.scan(str(tmp_path) + os.sep).match_ids == ['100'].

- **test_directory_symlink_root_is_rejected** — tests/test_inventory.py:L453-L457.

- **test_dangling_symlink_root_is_rejected** — tests/test_inventory.py:L460-L463.

- **test_junction_root_is_rejected** — tests/test_inventory.py:L467-L475.

- **test_linked_root_rejected_without_traversal_when_simulated** — tests/test_inventory.py:L479-L489.

- **test_match_view_reports_scan_wide_issues** — tests/test_inventory.py:L492-L499.
  Selected assertions: L495: inv.main([str(tmp_path), '--match', '100']) == 0; L496: 'scan-wide issues not tied to a match: 1' in capsys.readouterr().out.

- **test_file_symlink_to_external_file_is_skipped** — tests/test_inventory.py:L512-L524.
  Selected assertions: L520: result.match_ids == ['100']; L522: skipped.paths == (link.relative_to(root).as_posix(),).

- **test_dangling_symlink_is_reported_not_crashing** — tests/test_inventory.py:L527-L533.
  Selected assertions: L533: result.matches == {} and len(issues(result, 'skipped_entry')) == 1.

- **test_directory_symlink_is_not_followed** — tests/test_inventory.py:L536-L546.
  Selected assertions: L543: result.match_ids == ['100']; L545: [i.paths for i in skipped] == [('Linked',)].

- **test_link_detection_with_simulated_links** — tests/test_inventory.py:L549-L575.
  Platform-independent check of link handling, for machines that cannot symlink.
  Selected assertions: L571: result.match_ids == ['100']; L573: set(skipped) == {f'Tracking Data-{TS}-1-001/Tracking Data/9.jsonl.bz2', 'LinkedDir'}.

- **test_junction_is_not_followed** — tests/test_inventory.py:L579-L592.
  Selected assertions: L591: result.match_ids == ['100']; L592: [i.paths for i in issues(result, 'skipped_entry')] == [('Junction',)].

- **test_scan_never_opens_or_mutates_dataset_files** — tests/test_inventory.py:L605-L644.
  Selected assertions: L642: inv.main([str(tmp_path), '--format', 'json']) == 0; L644: snapshot(tmp_path) == before.

### tests/test_audit_lib.py — 23 test definitions
**Dependencies:** importlib.util; os; pathlib: Path; pytest; subprocess; sys.

- **test_duplicate_jersey_classification_and_order_independence** — tests/test_audit_lib.py:L26-L36.
  Selected assertions: L27: L.classify_duplicate_jerseys(None)['status'] == 'missing'; L28: L.classify_duplicate_jerseys([entry('1', 0, 0), entry('2', 1, 1)])['status'] == 'unique'.

- **test_repeated_pair_rows_counts_every_row** — tests/test_audit_lib.py:L44-L46.
  Selected assertions: L46: L.repeated_pair_rows(rows) == 2.

- **test_collision_safe_matching_preserves_rows_and_reports_unmatched** — tests/test_audit_lib.py:L49-L58.
  Selected assertions: L54: matched_a == [0, 1, 2]; L55: sorted(m['unmatched_a']) == [3, 4].

- **test_matching_is_one_to_one_for_identical_rows** — tests/test_audit_lib.py:L61-L64.
  Selected assertions: L64: sorted((j for _, j in m['pairs'])) == [0, 1].

- **test_field_comparison_scope_separates_changes_from_schema_differences** — tests/test_audit_lib.py:L67-L77.
  Selected assertions: L73: c['changed'] == ['possessionEvents.touchPlayerId']; L74: c['only_a'] == ['sequence'] and c['only_b'] == [].

- **test_alignment_nearest_frame_and_tolerance_is_one_frame** — tests/test_audit_lib.py:L91-L97.
  Selected assertions: L94: al.best[0][2] == 1; L95: al.outcome(0, 1.4 * FRAME, 1) == ('matched', None).

- **test_alignment_exact_tie_chooses_earlier_frame** — tests/test_audit_lib.py:L100-L102.
  Selected assertions: L102: al.best[0][2] == 0.

- **test_alignment_requires_same_period_and_reports_reasons** — tests/test_audit_lib.py:L105-L114.
  Selected assertions: L109: al.outcome(0, 0.0, 2) == ('unmatched', 'period_label_differs_from_nearest_frame'); L110: al.outcome(1, 0.0, 0) == ('unmatched', 'invalid_source_period').

- **test_coordinate_status_thresholds_and_not_assessable** — tests/test_audit_lib.py:L117-L123.
  Selected assertions: L120: L.coordinate_status(ev_players, same)['status'] == 'coordinate_agreeing'; L122: L.coordinate_status(ev_players, moved)['status'] == 'coordinate_disagreeing'.

- **test_run_matches_refuses_to_write_inside_raw_root** — tests/test_audit_lib.py:L126-L130.
  Error expectations: pytest.raises(L.AuditError).

- **test_valid_match_ids_are_accepted** — tests/test_audit_lib.py:L165-L166.
  Selected assertions: L166: L.validate_match_id(good) == good.

- **test_malformed_match_ids_are_rejected** — tests/test_audit_lib.py:L176-L178.
  Error expectations: pytest.raises(L.AuditError).

- **test_traversal_id_is_rejected_before_any_work_and_raw_file_is_untouched** — tests/test_audit_lib.py:L181-L190.
  Selected assertions: L188: calls == []; L189: raw_file.read_text() == RAW_CONTENT.
  Error expectations: pytest.raises(L.AuditError).

- **test_destination_must_stay_inside_output_dir_and_outside_raw** — tests/test_audit_lib.py:L193-L204.
  Selected assertions: L197: L.safe_destination(raw, out, '3812') == out.resolve() / '3812.json'; L198: L.safe_destination(raw, out, '3812', suffix='.json.tmp').name == '3812.json.tmp'.
  Error expectations: pytest.raises(L.AuditError); pytest.raises(L.AuditError); pytest.raises(L.AuditError).

- **test_output_dir_linked_into_raw_is_rejected** — tests/test_audit_lib.py:L207-L213.
  Selected assertions: L213: raw_file.read_text() == RAW_CONTENT.
  Error expectations: pytest.raises(L.AuditError).

- **test_destination_file_symlinked_to_raw_is_rejected** — tests/test_audit_lib.py:L216-L226.
  Selected assertions: L226: raw_file.read_text() == RAW_CONTENT.
  Error expectations: pytest.raises(L.AuditError).

- **test_valid_run_writes_only_inside_output_dir_and_leaves_no_temp_file** — tests/test_audit_lib.py:L229-L235.
  Selected assertions: L233: summary == dict(attempted=2, failed={}); L234: sorted((p.name for p in out.iterdir())) == ['10517.json', '3812.json'].

- **test_exit_status_zero_when_all_succeed** — tests/test_audit_lib.py:L262-L265.
  Selected assertions: L264: code == 0; L265: 'complete, successful output' in capsys.readouterr().out.

- **test_exit_status_nonzero_when_one_fails_and_diagnostics_are_kept** — tests/test_audit_lib.py:L268-L273.
  Selected assertions: L271: code == 1; L272: 'synthetic failure' in cap.out + cap.err and 'FAILED' in cap.err.

- **test_exit_status_nonzero_when_all_fail** — tests/test_audit_lib.py:L276-L279.
  Selected assertions: L278: code == 1; L279: 'complete, successful output' not in capsys.readouterr().out.

- **test_exit_status_nonzero_for_missing_requested_results** — tests/test_audit_lib.py:L282-L285.
  Selected assertions: L285: code == 1 and 'missing output' in cap.err.

- **test_invalid_match_id_is_a_controlled_error_with_nonzero_status** — tests/test_audit_lib.py:L288-L291.
  Selected assertions: L290: code == 2 and 'invalid match id' in capsys.readouterr().err; L291: not (tmp_path / 'out').exists().

- **test_check_outputs_reports_failed_missing_and_incomplete** — tests/test_audit_lib.py:L294-L296.
  Selected assertions: L296: L.check_outputs(res, ['1', '2', '3', '4']) == {'2': 'boom', '3': 'incomplete output', '4': 'missing output'}.

### tests/test_ingestion.py — 46 test definitions
**Dependencies:** bz2; football_fatigue.ingestion.__main__: main; football_fatigue.ingestion: AttributionStatus, Dataset, EventVersion, IngestionError, QualityFlag, SmoothedStatus, quality_report, safe_output_path, validate_match_id, write_quality_report; hashlib; json; os; pathlib: Path; pytest; subprocess.

- **test_tracking_is_streamed_lazily_and_bounded** — tests/test_ingestion.py:L93-L99.
  Selected assertions: L97: iter(it) is it; L98: next(it).frame_num == 10.

- **test_repeated_records_are_collapsed_with_full_provenance** — tests/test_ingestion.py:L102-L111.
  Selected assertions: L105: [f.frame_num for f in frames] == [10, 11]; L107: f.source_record_indices == (0, 1, 2) and f.source_record_count == 3.

- **test_uncollapsed_mode_keeps_every_record** — tests/test_ingestion.py:L114-L117.
  Selected assertions: L117: [(f.frame_num, f.source_record_indices) for f in frames] == [(10, (0,)), (10, (1,)), (11, (2,))].

- **test_gap_and_nonmonotonic_frames_are_flagged_not_repaired** — tests/test_ingestion.py:L120-L124.
  Selected assertions: L123: Q.FRAME_GAP in b.quality_flags and b.frame_gap_before == 3; L124: Q.NONMONOTONIC_FRAME in c.quality_flags.

- **test_conflicting_smoothed_candidates_are_retained_and_coordinate_is_unresolved** — tests/test_ingestion.py:L135-L145.
  Selected assertions: L139: p1.smoothed_status is SmoothedStatus.AMBIGUOUS_CONFLICTING; L140: sorted(((c.x_m, c.y_m) for c in p1.smoothed_candidates)) == [(1.5, 1.0), (1.6, 1.0)].

- **test_list_order_never_selects_a_candidate** — tests/test_ingestion.py:L148-L154.
  Selected assertions: L154: p1.smoothed_status is SmoothedStatus.AMBIGUOUS_CONFLICTING and p1.x_smooth_m is None.

- **test_identical_duplicate_smoothed_entries_are_not_ambiguous** — tests/test_ingestion.py:L157-L162.
  Selected assertions: L162: p1.smoothed_status is SmoothedStatus.DUPLICATE_IDENTICAL and p1.x_smooth_m == 1.5 and (len(p1.smoothed_candidates) == 2).

- **test_missing_smoothed_list_and_unmatched_smoothed_entries** — tests/test_ingestion.py:L165-L173.
  Selected assertions: L171: Q.SMOOTHED_LIST_MISSING in f.quality_flags and Q.SMOOTHED_JERSEY_WITHOUT_RAW in f.quality_flags; L172: all((p.smoothed_status is SmoothedStatus.MISSING for p in f.players if p.source_side == 'home')).

- **test_missing_ball_is_a_missing_observation_not_zero** — tests/test_ingestion.py:L176-L183.
  Selected assertions: L180: a.ball is None and Q.BALL_MISSING in a.quality_flags; L181: b.ball is not None and b.ball.x_raw_m is None and (b.ball.x_smooth_m is None) and (Q.BALL_MISSING in b.quality_flags).

- **test_invalid_period_is_kept_distinct_from_canonical_period** — tests/test_ingestion.py:L186-L190.
  Selected assertions: L189: [(f.source_period, f.canonical_period) for f in fr] == [(0, None), (3, 3), (None, None), (7, None)]; L190: [Q.INVALID_SOURCE_PERIOD in f.quality_flags for f in fr] == [True, False, True, True].

- **test_malformed_values_become_none_with_flags_never_zero** — tests/test_ingestion.py:L193-L200.
  Selected assertions: L198: (h[0].x_raw_m, h[0].y_raw_m) == (None, 1.0) and Q.INVALID_COORDINATE in h[0].quality_flags; L199: (h[1].x_raw_m, h[1].y_raw_m) == (None, None) and Q.INVALID_COORDINATE in h[1].quality_flags.

- **test_identity_uses_side_and_shirt_never_jersey_alone** — tests/test_ingestion.py:L205-L212.
  Selected assertions: L210: by['home', 1].player_id == 1001 and by['away', 1].player_id == 2001; L211: by['home', 1].team_id == 10 and by['away', 3].team_id == 20.

- **test_unresolved_quarantined_and_not_attempted_identity** — tests/test_ingestion.py:L215-L224.
  Selected assertions: L220: u.attribution_status is AttributionStatus.UNRESOLVED_JERSEY and u.player_id is None and (Q.IDENTITY_UNRESOLVED_JERSEY in u.quality_flags); L221: all((p.attribution_status is AttributionStatus.QUARANTINED and p.player_id is None and (p.team_id is None) for p in b.players)).

- **test_repeated_roster_shirt_is_excluded_from_the_join** — tests/test_ingestion.py:L227-L234.
  Selected assertions: L232: r.issues and ('home', 2) not in r.index(); L234: next((p for p in f.players if p.source_side == 'home' and p.jersey_number == 2)).attribution_status is AttributionStatus.UNRESOLVED_JERSEY.

- **test_event_version_is_explicit_and_never_substituted** — tests/test_ingestion.py:L239-L247.
  Selected assertions: L242: len(ds.load_events('100', EventVersion.ROOT)) == 2 and len(ds.load_events('100', EventVersion.MAY_1_2025)) == 3; L247: {r.source_version for r in ds.iter_events('100', EventVersion.MAY_1_2025)} == {EventVersion.MAY_1_2025}.
  Error expectations: pytest.raises(IngestionError, match='march_14_2025'); pytest.raises(IngestionError).

- **test_event_rows_preserve_order_duplicates_and_source_values** — tests/test_ingestion.py:L250-L260.
  Selected assertions: L254: [r.source_row_index for r in rows] == [0, 1, 2, 3] and [r.event_time_s for r in rows] == [5.0, 6.0, 7.0, None]; L255: [Q.EVENT_KEY_PAIR_REPEATED in r.quality_flags for r in rows] == [True, True, False, False].

- **test_event_attack_direction_nan_string_becomes_none** — tests/test_ingestion.py:L263-L266.
  Selected assertions: L266: [r.attack_direction for r in Dataset(raw).load_events('100', EventVersion.ROOT)] == [None, 'R'].

- **test_metadata_is_validated** — tests/test_ingestion.py:L271-L278.
  Selected assertions: L274: (m.match_id, m.home_team_id, m.fps, m.pitch_length_m) == (100, 10, 30.0, 105.0) and m.side_of_team(20) == 'away'.
  Error expectations: pytest.raises(IngestionError).

- **test_null_period_times_are_none_not_zero** — tests/test_ingestion.py:L281-L284.
  Selected assertions: L284: m.start_period_1_s is None and m.end_period_2_s is None and (m.home_team_start_left_extra_time is True).

- **test_roster_with_foreign_team_is_fatal** — tests/test_ingestion.py:L287-L290.
  Error expectations: pytest.raises(IngestionError).

- **test_fatal_structural_tracking_errors** — tests/test_ingestion.py:L293-L299.
  Error expectations: pytest.raises(IngestionError, match=text).

- **test_invalid_match_ids_and_unknown_matches** — tests/test_ingestion.py:L302-L311.
  Selected assertions: L309: validate_match_id(100) == '100'.
  Error expectations: pytest.raises(IngestionError, match='not found'); pytest.raises(IngestionError); pytest.raises(IngestionError).

- **test_quality_report_counts_denominators_and_missing_periods** — tests/test_ingestion.py:L316-L329.
  Selected assertions: L322: rep.status == 'ok_with_quality_issues' and (not rep.fatal_errors); L323: (t['source_records'], t['frames'], t['player_samples']) == (5, 4, 16).

- **test_quality_report_reports_fatal_errors_instead_of_raising** — tests/test_ingestion.py:L332-L337.
  Selected assertions: L335: rep.status == 'fatal' and any(('events' in e for e in rep.fatal_errors)) and any(('tracking' in e for e in rep.fatal_errors)); L336: rep.tracking['frames'] == 1.

- **test_partial_scan_is_labelled** — tests/test_ingestion.py:L340-L343.
  Selected assertions: L343: rep.tracking['frames'] == 2 and rep.scope['max_frames'] == 2 and ('partial scan' in rep.summary_text()).

- **test_ingestion_and_report_never_modify_raw_data** — tests/test_ingestion.py:L348-L358.
  Selected assertions: L356: digest(raw) == before; L358: out.is_file() and json.loads(out.read_text())['match_id'] == '100'.

- **test_reports_are_written_only_below_the_approved_processed_root** — tests/test_ingestion.py:L361-L371.
  Selected assertions: L367: not any(raw.rglob('*.json.tmp')) and digest(raw); L371: write_quality_report(rep, raw, overwrite=True) == p.
  Error expectations: pytest.raises(IngestionError, match='already exists'); pytest.raises(IngestionError).

- **test_output_path_rejects_malformed_ids_and_links** — tests/test_ingestion.py:L374-L389.
  Error expectations: pytest.raises(IngestionError); pytest.raises(IngestionError).

- **test_cli_exit_codes** — tests/test_ingestion.py:L394-L402.
  Selected assertions: L396: cli_main(['quality', '100', '--events', 'root', '--root', str(raw)]) == 0; L397: 'status: ok_with_quality_issues' in capsys.readouterr().out or True.
  Error expectations: pytest.raises(SystemExit).

- **test_repeated_records_with_conflicting_timestamp_or_period_are_rejected** — tests/test_ingestion.py:L409-L422.
  Selected assertions: L422: all((f.frame_num != 11 for f in got)); L421: 'records 0 and 1' in str(e) and 'frameNum 10' in str(e).
  Error expectations: pytest.raises(IngestionError, match='incompatible repeated records').

- **test_repeated_records_with_other_conflicts_are_flagged_unresolved_with_fields** — tests/test_ingestion.py:L425-L432.
  Selected assertions: L431: Q.REPEATED_RECORDS_DIFFER in f.quality_flags and field in f.repeat_conflict_fields; L432: f.source_record_indices == (0, 1).

- **test_compatible_repeated_records_collapse_without_conflict** — tests/test_ingestion.py:L435-L439.
  Selected assertions: L439: f.repeat_conflict_fields == () and Q.REPEATED_RECORDS_DIFFER not in f.quality_flags and (f.source_record_count == 2).

- **test_uncollapsed_mode_still_yields_conflicting_records_separately** — tests/test_ingestion.py:L442-L445.
  Selected assertions: L445: [f.source_record_indices for f in frames] == [(0,), (1,), (2,)].

- **test_foreign_roster_and_metadata_are_rejected** — tests/test_ingestion.py:L456-L467.
  Error expectations: pytest.raises(IngestionError, match='roster belongs to match 200'); pytest.raises(IngestionError, match='metadata belongs to match 200'); pytest.raises(IngestionError, match='metadata belongs to match 200').

- **test_roster_inconsistent_with_home_away_teams_is_rejected** — tests/test_ingestion.py:L470-L477.
  Error expectations: pytest.raises(IngestionError, match='inconsistent with the home/away teams').

- **test_foreign_events_and_tracking_records_are_rejected_but_missing_ids_are_allowed** — tests/test_ingestion.py:L480-L492.
  Selected assertions: L486: len(Dataset(build(tmp_path / 'n', [rec(10)], events=no_id, versions=('root',))).load_events('100', EventVersion.ROOT)) == 3; L492: len(list(Dataset(build(tmp_path / f'ok{ref}', [rec(10, gameRefId=ref)])).iter_tracking_frames('100'))) == 1.
  Error expectations: pytest.raises(IngestionError, match='row 1: gameId 999 does not belong to match 100'); pytest.raises(IngestionError, match='gameRefId .* does not belong to match 100').

- **test_unparseable_smoothed_jerseys_are_preserved_as_unmatched** — tests/test_ingestion.py:L497-L506.
  Selected assertions: L502: ('home', 7.0, 'x') in kept and ('home', 8.0, 'None') in kept and (('home', 9.0, '1.5') in kept); L503: ('home', None, None) in kept and ('home', None, 'junk') in kept.

- **test_unparseable_smoothed_jersey_is_kept_even_when_a_raw_jersey_is_also_unparseable** — tests/test_ingestion.py:L509-L515.
  Selected assertions: L513: [(s, c.x_m, j) for s, c, j in f.unmatched_smoothed if s == 'home'] == [('home', 3.5, 'x')]; L515: bad.smoothed_status is SmoothedStatus.MISSING and bad.x_smooth_m is None.

- **test_processed_directory_redirected_outside_the_boundary_is_rejected** — tests/test_ingestion.py:L528-L541.
  Selected assertions: L541: list(elsewhere.iterdir()) == [] and digest(raw).
  Error expectations: pytest.raises(IngestionError, match='link|redirected'); pytest.raises(IngestionError, match='link|redirected'); pytest.raises(IngestionError, match='link|redirected').

- **test_redirected_inner_directories_and_files_are_rejected** — tests/test_ingestion.py:L544-L555.
  Selected assertions: L555: list(outside.iterdir()) == [].
  Error expectations: pytest.raises(IngestionError).

- **test_final_and_temporary_destinations_are_validated_and_overwrite_is_explicit** — tests/test_ingestion.py:L558-L570.
  Selected assertions: L563: dest.parent == tmp.parent == tmp_path / 'processed' / 'ingestion_quality' / '100' and tmp.name == 'root.json.tmp'; L564: write_quality_report(rep, raw) == dest and (not tmp.exists()).
  Error expectations: pytest.raises(IngestionError, match='already exists'); pytest.raises(IngestionError).

- **test_malformed_nested_metadata_is_a_contextual_ingestion_error** — tests/test_ingestion.py:L575-L583.
  Error expectations: pytest.raises(IngestionError, match='roster 100'); pytest.raises(IngestionError, match='metadata 100').

- **test_unhashable_event_ids_are_a_contextual_ingestion_error** — tests/test_ingestion.py:L586-L594.
  Selected assertions: L594: Dataset(raw).load_events('100', EventVersion.ROOT)[0].game_event_type is None.
  Error expectations: pytest.raises(IngestionError, match=f'row 1: {k} must be a scalar').

- **test_failed_or_truncated_scan_does_not_claim_missing_periods** — tests/test_ingestion.py:L597-L616.
  Selected assertions: L602: rep.status == 'fatal' and rep.tracking['scan_complete'] is False and (rep.tracking['frames'] == 2); L603: rep.periods['event_periods_without_tracking'] is None and rep.periods['flag_counts'] == {}.

- **test_conflicting_smoothed_ball_is_unresolved_and_first_value_is_not_exposed** — tests/test_ingestion.py:L621-L633.
  Selected assertions: L627: (b.x_smooth_m, b.y_smooth_m, b.z_smooth_m, b.visibility_smooth) == (None, None, None, None); L628: (b.x_raw_m, b.y_raw_m, b.z_raw_m) == (0.5, 0.5, 0.1).

- **test_agreeing_smoothed_ball_and_uncollapsed_records_keep_their_values** — tests/test_ingestion.py:L636-L641.
  Selected assertions: L639: (f.ball.x_smooth_m, f.ball.y_smooth_m) == (1.25, 2.5) and Q.BALL_SMOOTHED_UNRESOLVED not in f.quality_flags; L641: [f.ball.x_smooth_m for f in fs] == [1.25, 9.0] and all((Q.BALL_SMOOTHED_UNRESOLVED not in f.quality_flags for f in fs)).

### tests/test_synchronization.py — 30 test definitions
**Dependencies:** football_fatigue.ingestion: Dataset, EventVersion, IngestionError, QualityFlag; football_fatigue.synchronization: MatchStatus, SpatialStatus, SynchronizationError, TemporalNote, UnmatchedReason, VALIDATED_CORRESPONDENCE, synchronize_match; football_fatigue.synchronization: align_events; importlib.util; json; pathlib: Path; pytest; sys.

- **test_exact_match_and_provenance** — tests/test_synchronization.py:L51-L59.
  Selected assertions: L54: (a.status, a.reason, a.frame_num) == (MatchStatus.MATCHED, None, 12); L55: a.residual_s == pytest.approx(0.0, abs=1e-09) and a.tolerance_s == pytest.approx(TOL).

- **test_nearest_frame_selection_and_signed_residual** — tests/test_synchronization.py:L62-L65.
  Selected assertions: L64: [a.frame_num for a in s.alignments] == [12, 13, 15]; L65: s.alignments[0].residual_s == pytest.approx(T(0.4)) and s.alignments[1].residual_s == pytest.approx(-T(0.4)).

- **test_exact_tie_goes_to_the_earlier_frame** — tests/test_synchronization.py:L68-L70.
  Selected assertions: L70: s.alignments[0].frame_num == 12.

- **test_tolerance_boundary_is_one_frame_interval** — tests/test_synchronization.py:L73-L82.
  Selected assertions: L78: a[0].status is MatchStatus.MATCHED and a[0].frame_num == 20 and (TemporalNote.NEAR_PERIOD_BOUNDARY in a[0].temporal_notes); L79: a[1].reason is U.OUTSIDE_PERIOD_FRAME_RANGE and a[1].frame_num is None and (a[1].residual_s is None).

- **test_custom_tolerance_and_invalid_arguments** — tests/test_synchronization.py:L85-L91.
  Selected assertions: L87: s.alignments[0].reason is U.NO_FRAME_WITHIN_TOLERANCE and s.tolerance_s == 0.01.
  Error expectations: pytest.raises(SynchronizationError).

- **test_events_before_and_after_the_tracking_range** — tests/test_synchronization.py:L94-L96.
  Selected assertions: L96: [a.reason for a in s.alignments] == [U.OUTSIDE_PERIOD_FRAME_RANGE, U.OUTSIDE_PERIOD_FRAME_RANGE, None].

- **test_same_period_matching_only** — tests/test_synchronization.py:L101-L109.
  Selected assertions: L106: (r[0].status, r[0].frame_num, r[0].frame_period) == (MatchStatus.MATCHED, 16, 2); L107: r[1].reason is U.PERIOD_LABEL_DIFFERS_FROM_NEAREST_FRAME and r[1].frame_num is None.

- **test_invalid_periods_and_missing_time_are_preserved_not_matched** — tests/test_synchronization.py:L112-L119.
  Selected assertions: L116: (r[0].reason, r[0].source_period, r[0].canonical_period) == (U.INVALID_SOURCE_PERIOD, 0, None); L117: (r[1].reason, r[1].source_period) == (U.INVALID_SOURCE_PERIOD, 9).

- **test_event_only_extra_time_has_no_tracking_period_and_no_invented_frames** — tests/test_synchronization.py:L122-L127.
  Selected assertions: L125: [a.reason for a in s.alignments] == [None, U.NO_TRACKING_FOR_PERIOD, U.NO_TRACKING_FOR_PERIOD]; L126: all((a.frame_num is None and a.frame_source is None and (a.frame_source_record_indices == ()) for a in s.alignments[1:])).

- **test_repeated_tracking_records_are_one_frame_with_all_indices** — tests/test_synchronization.py:L132-L137.
  Selected assertions: L136: a.frame_num == 11 and a.frame_source_record_indices == (1, 2, 3); L137: Q.REPEATED_RECORDS_PRESENT in a.frame_quality_flags and Q.BALL_SMOOTHED_UNRESOLVED in a.frame_quality_flags.

- **test_event_version_is_explicit_and_rows_are_never_substituted** — tests/test_synchronization.py:L140-L153.
  Selected assertions: L145: len(root.alignments) == 2 and len(may.alignments) == 3; L146: {a.event_version for a in may.alignments} == {EventVersion.MAY_1_2025} and may.event_version is EventVersion.MAY_1_2025.
  Error expectations: pytest.raises(IngestionError, match='march_14_2025'); pytest.raises(SynchronizationError); pytest.raises(IngestionError).

- **test_every_event_is_preserved_in_source_order_including_duplicate_keys** — tests/test_synchronization.py:L156-L160.
  Selected assertions: L159: [a.source_row_index for a in s.alignments] == [0, 1, 2, 3]; L160: [a.frame_num for a in s.alignments] == [15, 12, 10, None].

- **test_output_is_deterministic** — tests/test_synchronization.py:L163-L167.
  Selected assertions: L167: one == two and json.dumps(one.summary(), sort_keys=True) == json.dumps(two.summary(), sort_keys=True).

- **test_temporal_match_is_not_spatial_verification** — tests/test_synchronization.py:L184-L197.
  Selected assertions: L192: good.spatial.status is SpatialStatus.COORDINATE_AGREEING and good.quality_label == 'coordinate_agreeing'; L193: good.spatial.players_compared == 11 and good.spatial.players_within_tolerance == 11 and (good.spatial.ball == 'agrees').

- **test_spatial_is_not_assessable_without_comparable_valid_observations** — tests/test_synchronization.py:L200-L206.
  Selected assertions: L205: [a.spatial.status for a in s.alignments] == [SpatialStatus.NOT_ASSESSABLE] * 2; L206: s.alignments[1].spatial.players_compared == 10.

- **test_conflicting_frame_data_is_never_used_for_spatial_assessment** — tests/test_synchronization.py:L209-L214.
  Selected assertions: L213: a.spatial.status is SpatialStatus.NOT_ASSESSABLE and a.spatial.reason == 'frame_player_data_conflict' and (a.spatial.ball == 'not_assessable'); L214: Q.REPEATED_RECORDS_DIFFER in a.frame_quality_flags.

- **test_missing_frame_ball_makes_only_the_ball_comparison_unassessable** — tests/test_synchronization.py:L217-L221.
  Selected assertions: L221: a.spatial.status is SpatialStatus.COORDINATE_AGREEING and a.spatial.ball == 'not_assessable' and (Q.BALL_MISSING in a.frame_quality_flags).

- **test_quarantined_periods_are_excluded_from_spatial_assessment** — tests/test_synchronization.py:L224-L231.
  Selected assertions: L229: plain.alignments[0].spatial.status is SpatialStatus.COORDINATE_AGREEING; L230: held.alignments[0].status is MatchStatus.MATCHED.

- **test_event_coordinates_are_reference_observations_and_the_flagged_discrepancy_is_not_corrected** — tests/test_synchronization.py:L234-L237.
  Selected assertions: L237: a.spatial.status is SpatialStatus.COORDINATE_DISAGREEING and a.frame_num == 12 and (a.residual_s == pytest.approx(0, abs=1e-09)).

- **test_failed_scan_makes_no_absence_claims_and_keeps_all_events** — tests/test_synchronization.py:L242-L250.
  Selected assertions: L247: not s.scan_complete and 'record 4' in s.fatal_error and (len(r) == 4); L248: r[0].status is MatchStatus.MATCHED and r[0].frame_num == 11.

- **test_non_monotonic_frames_are_rejected_not_silently_misaligned** — tests/test_synchronization.py:L253-L257.
  Selected assertions: L256: not s.scan_complete and 'increasing time order' in s.fatal_error; L257: s.alignments[1].reason is U.TRACKING_SCAN_INCOMPLETE.

- **test_synchronization_never_modifies_raw_data** — tests/test_synchronization.py:L260-L266.
  Selected assertions: L266: H.digest(raw) == before and (not (tmp_path / 'processed').exists()).

- **test_backward_timestamp_makes_every_tracking_dependent_result_non_definitive** — tests/test_synchronization.py:L276-L287.
  Selected assertions: L280: s.ordering_violation is True and s.scan_complete is False and ('increasing time order' in s.fatal_error); L282: len(r) == 7.

- **test_ordinary_failed_scan_is_not_an_ordering_violation_and_ordered_streams_are_unchanged** — tests/test_synchronization.py:L290-L296.
  Selected assertions: L293: s.scan_complete is False and s.ordering_violation is False; L294: s.alignments[0].status is MatchStatus.MATCHED and s.alignments[1].reason is U.TRACKING_SCAN_INCOMPLETE.

- **test_align_events_rejects_foreign_events_frames_and_mixed_versions** — tests/test_synchronization.py:L310-L326.
  Selected assertions: L313: [a.frame_num for a in ok.alignments] == [12, 13]; L326: issubclass(SynchronizationError, IngestionError).
  Error expectations: pytest.raises(SynchronizationError, match='belongs to match 200, not the requested match 100'); pytest.raises(SynchronizationError, match='tracking frame .* belongs to match 200'); pytest.raises(SynchronizationError, match='tracking frame .* belongs to match 200').

- **test_all_jerseys_conflicting_on_the_event_side_is_not_assessable_and_order_independent** — tests/test_synchronization.py:L350-L354.
  Selected assertions: L352: a.status is MatchStatus.MATCHED and a.frame_num == 12; L353: a.spatial.status is SpatialStatus.NOT_ASSESSABLE and a.spatial.players_compared == 0 and (a.spatial.players_excluded_ambiguous == 11).

- **test_all_jerseys_conflicting_on_the_tracking_side_is_not_assessable_and_order_independent** — tests/test_synchronization.py:L358-L361.
  Selected assertions: L360: a.status is MatchStatus.MATCHED and a.spatial.status is SpatialStatus.NOT_ASSESSABLE and (a.spatial.players_excluded_ambiguous == 11); L361: Q.DUPLICATE_RAW_JERSEY in a.frame_quality_flags.

- **test_enough_unambiguous_players_remain_assessable_and_results_do_not_depend_on_order** — tests/test_synchronization.py:L365-L372.
  Selected assertions: L369: few.spatial.status is SpatialStatus.NOT_ASSESSABLE and few.spatial.players_compared == 8 and (few.spatial.players_excluded_ambiguous == 6); L371: one.spatial.status is SpatialStatus.COORDINATE_AGREEING and one.spatial.players_compared == 13 and (one.spatial.players_excluded_ambiguous == 1).

- **test_identical_duplicates_are_not_ambiguous_and_invalid_duplicates_do_not_poison_a_group** — tests/test_synchronization.py:L375-L380.
  Selected assertions: L377: same.spatial.status is SpatialStatus.COORDINATE_AGREEING and same.spatial.players_compared == 11 and (same.spatial.players_excluded_ambiguous == 0); L380: inval.spatial.status is SpatialStatus.COORDINATE_AGREEING and inval.spatial.players_compared == 11.

- **test_conflicting_duplicate_event_balls_are_not_compared** — tests/test_synchronization.py:L383-L388.
  Selected assertions: L388: s.alignments[0].spatial.ball == 'not_assessable' and s.alignments[0].spatial.status is SpatialStatus.COORDINATE_AGREEING.

### tests/test_match_data.py — 18 test definitions
**Dependencies:** bz2; football_fatigue.ingestion: AttributionStatus, Dataset, EventVersion, IngestionError, QualityFlag; football_fatigue.match_data: CoordinatePolicy, CoordState, MatchData, prepare_frame, select, Coordinate; importlib.util; pathlib: Path; pytest; sys.

- **test_event_version_and_policy_must_be_explicit** — tests/test_match_data.py:L40-L50.
  Selected assertions: L50: 'events=root' in repr(MatchData.open(raw, '100', EventVersion.ROOT)).
  Error expectations: pytest.raises(TypeError); pytest.raises(IngestionError); pytest.raises(IngestionError).

- **test_basic_information_does_not_read_the_tracking_file** — tests/test_match_data.py:L53-L60.
  Selected assertions: L57: md.metadata.match_id == 100 and len(md.roster.entries) == 4; L58: len(md.events()) == 3 and {e.source_version for e in md.events()} == {EventVersion.MAY_1_2025}.
  Error expectations: pytest.raises(IngestionError).

- **test_frames_are_lazy_chronological_and_never_manufactured** — tests/test_match_data.py:L63-L74.
  Selected assertions: L67: iter(it) is it; L69: [f.frame.frame_num for f in frames] == [10, 11, 14, 15].

- **test_raw_and_smoothed_are_distinct_representations** — tests/test_match_data.py:L79-L87.
  Selected assertions: L83: (p.raw.state, p.raw.x, p.raw.y) == (S.AVAILABLE, 1.0, 1.0) and (p.smoothed.state, p.smoothed.x, p.smoothed.y) == (S.AVAILABLE, 1.5, 1.5); L84: p.selected.source == 'raw' and p.selected.x == 1.0.

- **test_missing_coordinates_stay_missing_never_zero** — tests/test_match_data.py:L90-L98.
  Selected assertions: L98: pf.ball.raw.state is S.MISSING and pf.ball.selected.x is None and (not pf.ball.position_usable) and (Q.BALL_MISSING in pf.frame.quality_flags); L96: p.raw.state is S.MISSING and p.raw.x is None and (p.raw.y is None) and (p.selected.x is None) and (p.selected.y is None).

- **test_ambiguous_smoothed_candidates_are_kept_and_never_selected** — tests/test_match_data.py:L101-L110.
  Selected assertions: L105: p.smoothed.state is S.AMBIGUOUS and sorted(p.smoothed.candidates) == [(1.5, 1.0), (1.6, 1.0)] and (p.smoothed.x is None); L107: smoothed.selected.state is S.AMBIGUOUS and smoothed.selected.x is None and (not smoothed.position_usable).

- **test_conflicting_duplicate_raw_jerseys_are_ambiguous_under_every_policy** — tests/test_match_data.py:L113-L121.
  Selected assertions: L121: home(pf, 2).selected.state is S.AVAILABLE or policy is P.SMOOTHED; L119: p.raw.state is S.AMBIGUOUS and sorted(p.raw.candidates) == [(1.0, 1.0), (9.0, 9.0)].

- **test_repeated_frame_smoothed_ball_conflict_stays_unresolved_and_raw_ball_is_unchanged** — tests/test_match_data.py:L124-L134.
  Selected assertions: L129: b.smoothed.state is S.AMBIGUOUS and b.smoothed.x is None and (b.smoothed.reason == 'repeated_records_disagree'); L130: (b.raw.state, b.raw.x, b.raw.y) == (S.AVAILABLE, 0.5, 0.5) and b.selected.source == 'raw' and b.position_usable.

- **test_select_policy_function_is_pure_and_total** — tests/test_match_data.py:L137-L142.
  Selected assertions: L139: select(av, miss, P.RAW).source == 'raw' and select(miss, av, P.RAW).state is S.MISSING; L140: select(miss, av, P.SMOOTHED).x == 1.0 and select(av, amb, P.SMOOTHED).state is S.AMBIGUOUS.

- **test_usability_is_not_physical_validation_and_carries_caveats** — tests/test_match_data.py:L147-L153.
  Selected assertions: L152: p.position_usable and {'PLAYER_ESTIMATED', 'JERSEY_CONFIDENCE_LOW'} <= p.caveats; L153: not home(b, 1).position_usable and (not b.period_usable) and (b.frame.source_period == 0) and (b.frame.canonical_period is None).

- **test_identity_is_not_resolved_beyond_the_roster_and_quarantine_is_respected** — tests/test_match_data.py:L156-L164.
  Selected assertions: L160: home(a, 1).identity_usable and home(a, 1).player_id == 1001 and (home(a, 1).team_id == 10); L162: not u.identity_usable and u.attribution_status is AttributionStatus.UNRESOLVED_JERSEY and (u.player_id is None) and u.position_usable.

- **test_events_are_attached_to_their_frames_and_unmatched_events_are_never_hidden** — tests/test_match_data.py:L169-L178.
  Selected assertions: L173: attached == {12: ['100:root:0', '100:root:1'], 15: ['100:root:2']}; L175: [(a.event_row_id, a.reason.value) for a in un] == [('100:root:3', 'no_tracking_for_period'), ('100:root:4', 'invalid_source_period')].

- **test_synchronization_is_cached_and_uses_the_requested_version** — tests/test_match_data.py:L181-L184.
  Selected assertions: L184: md.synchronize() is s1 and s1.event_version is EventVersion.MAY_1_2025 and (len(s1.alignments) == 2).

- **test_match_data_never_modifies_raw_data** — tests/test_match_data.py:L187-L192.
  Selected assertions: L192: H.digest(raw) == before and (not (tmp_path / 'processed').exists()).

- **test_events_attach_to_their_selected_source_frame_not_to_every_frame_with_the_same_number** — tests/test_match_data.py:L197-L210.
  Selected assertions: L203: [pf.frame.source_record_indices for pf in frames] == [(0,), (1,), (2,)] and [pf.frame_num for pf in frames] == [10, 11, 10]; L204: [[a.event_row_id for a in pf.events] for pf in frames] == [['100:root:0'], [], ['100:root:1']].

- **test_reversed_or_repeated_frame_numbers_never_create_extra_attachments** — tests/test_match_data.py:L213-L222.
  Selected assertions: L219: [pf.frame.source_record_indices for pf in frames] == [(0, 1), (2,), (3,)]; L220: [[a.event_row_id for a in pf.events] for pf in frames] == [['100:root:0', '100:root:1'], ['100:root:2'], ['100:root:3']].

- **test_repeated_ball_conflict_is_ambiguous_in_both_record_orders** — tests/test_match_data.py:L233-L249.
  Selected assertions: L238: pf.frame.repeat_conflict_fields == ('balls', 'ballsSmoothed') and pf.frame.source_record_indices == (0, 1); L240: b.raw.state is S.AMBIGUOUS and b.smoothed.state is S.AMBIGUOUS.

- **test_ball_conflict_summaries_are_identical_for_both_orders** — tests/test_match_data.py:L252-L259.
  Selected assertions: L259: out[0] == out[1].

### tests/test_match_data_quality.py — 12 test definitions
**Dependencies:** football_fatigue.ingestion: EventVersion, IngestionError, QualityFlag; football_fatigue.match_data: COORDINATE_SYSTEM, CoordinatePolicy, CoordState, MatchData; importlib.util; json; pathlib: Path; pytest; sys.

- **test_summary_counts_and_denominators** — tests/test_match_data_quality.py:L43-L60.
  Selected assertions: L47: q.scan.status == 'complete' and q.scan.frames_scanned == 3 and (q.scan.fatal_error is None); L48: (m['ball_raw_missing'].count, m['ball_raw_missing'].denominator, m['ball_raw_missing'].unit) == (1, 3, 'frames').

- **test_coordinate_source_selection_is_reflected_in_the_summary** — tests/test_match_data_quality.py:L63-L69.
  Selected assertions: L67: sm.breakdowns['selected_coordinate_source'] == {'smoothed': 9, 'none': 4} and sm.policy == 'smoothed'; L68: fb.breakdowns['selected_coordinate_source'] == {'smoothed': 9, 'raw_fallback': 1, 'none': 3}.

- **test_no_false_zero_coordinates_anywhere** — tests/test_match_data_quality.py:L72-L81.
  Selected assertions: L81: all((p.selected.x != 0.0 or p.selected.state is S.AVAILABLE for p in pf.players)); L80: (c.x, c.y) == (None, None).

- **test_partial_complete_and_failed_scans** — tests/test_match_data_quality.py:L86-L99.
  Selected assertions: L90: full.scan.status == 'complete' and full.event_periods_without_tracking == (3,) and any(('event-only' in w for w in full.warnings)); L92: part.scan.status == 'partial' and part.scan.frames_scanned == 2 and (part.scan.max_frames == 2).

- **test_corrupt_tracking_does_not_prevent_metadata_roster_events_or_a_failed_summary** — tests/test_match_data_quality.py:L102-L108.
  Selected assertions: L106: md.metadata.fps == 30.0 and len(md.roster.entries) == 4 and (len(md.events()) == 2); L108: q.scan.status == 'failed' and q.scan.frames_scanned == 0 and (q.metrics['events_total'].count == 2).

- **test_opening_a_match_never_scans_tracking** — tests/test_match_data_quality.py:L111-L115.
  Selected assertions: L115: md.metadata is md.metadata and md._sync == {}.

- **test_missing_extra_time_tracking_and_explicit_event_versions** — tests/test_match_data_quality.py:L120-L130.
  Selected assertions: L124: q.event_periods_without_tracking == (3, 4) and q.event_rows_by_canonical_period == {'1': 1, '3': 1, '4': 1}; L125: q.metrics['events_total'].count == 3 and q.synchronization['unmatched_reasons'] == {'no_tracking_for_period': 2}.

- **test_event_attachment_loses_and_duplicates_no_event** — tests/test_match_data_quality.py:L133-L142.
  Selected assertions: L138: len(ids) == len(set(ids)) == 11 and sorted(un) == ['100:root:11', '100:root:12']; L139: sorted(ids + un, key=lambda s: int(s.split(':')[-1])) == [e.event_row_id for e in md.events()].

- **test_identity_quarantine_is_counted_and_reported** — tests/test_match_data_quality.py:L145-L149.
  Selected assertions: L148: q.metrics['identity_quarantined'].count == 4 and q.metrics['identity_resolved'].count == 4 and (q.metrics['identity_quarantined'].denominator == 8); L149: q.quarantined_periods == (3,) and any(('quarantined' in w for w in q.warnings)).

- **test_playback_contract_exposes_stable_typed_fields_without_defaults_for_missing_data** — tests/test_match_data_quality.py:L154-L166.
  Selected assertions: L157: (a.frame_num, a.video_time_s, a.source_period, a.canonical_period) == (10, pytest.approx(T(10)), 0, None) and (not a.period_usable); L158: a.frame.source_record_indices == (0,) and a.frame.source.path.endswith('100.jsonl.bz2') and (Q.INVALID_SOURCE_PERIOD in a.quality_flags).

- **test_summaries_and_frames_are_deterministic_and_leave_raw_data_untouched** — tests/test_match_data_quality.py:L169-L177.
  Selected assertions: L174: s1 == s2 and json.dumps(s1.to_dict(), sort_keys=True) == json.dumps(s2.to_dict(), sort_keys=True); L177: f1 == f2 and H.digest(raw) == before and (not (tmp_path / 'processed').exists()).

- **test_source_record_provenance_survives_quality_scans** — tests/test_match_data_quality.py:L180-L185.
  Selected assertions: L184: q.metrics['frames_with_repeated_source_records'].count == 1 and q.scan.frames_scanned == 3; L185: [pf.frame.source_record_indices for pf in md.frames()] == [(0,), (1, 2), (3,)].

### tests/test_team_handoff_docs.py — 2 test definitions
**Dependencies:** importlib.util; pathlib: Path; re; sys.

- **test_handoff_examples_execute_against_the_real_api** — tests/test_team_handoff_docs.py:L20-L37.
  Selected assertions: L28: len(blocks) >= 8; L33: 'complete' in out or 'partial' in out.

- **test_documented_metric_names_exist** — tests/test_team_handoff_docs.py:L40-L44.
  Selected assertions: L44: name in text.

### tests/test_playback_adapter.py — 9 test definitions
**Dependencies:** football_fatigue.ingestion: EventVersion, QualityFlag; football_fatigue.match_data: CoordinatePolicy, MatchData; football_fatigue.playback.adapter: build_event_index, build_roster_lookup, build_roster_names, frame_to_state; football_fatigue.playback.server: get_matches_catalog, load_match_data; importlib.util; pathlib: Path; pytest; sys.

- **test_adapter_omits_unusable_player_positions** — tests/test_playback_adapter.py:L36-L52.
  Selected assertions: L48: len(state['players']) == 1; L49: state['players'][0]['jersey'] == 1.

- **test_adapter_flags_estimated_positions** — tests/test_playback_adapter.py:L55-L71.
  Selected assertions: L70: p1['estimated'] is True; L71: p2['estimated'] is False.

- **test_adapter_preserves_ball_coordinates_and_height** — tests/test_playback_adapter.py:L74-L86.
  Selected assertions: L83: state['ball'] is not None; L84: state['ball']['x'] == pytest.approx(12.5).

- **test_adapter_omits_unusable_ball** — tests/test_playback_adapter.py:L89-L96.
  Selected assertions: L96: state['ball'] is None.

- **test_adapter_surfaces_frame_gap** — tests/test_playback_adapter.py:L99-L109.
  Selected assertions: L108: s0['has_gap'] is False; L109: s1['has_gap'] is True.

- **test_adapter_event_index_contains_matched_and_unmatched** — tests/test_playback_adapter.py:L112-L136.
  Selected assertions: L124: len(ev_index) == 2; L128: len(matched) == 1.

- **test_adapter_event_index_enriches_player_and_team** — tests/test_playback_adapter.py:L139-L151.
  Selected assertions: L148: len(ev_index) == 1; L150: e0['player_name'] is not None.

- **test_get_matches_catalog** — tests/test_playback_adapter.py:L154-L162.
  Selected assertions: L158: len(catalog) == 1; L159: catalog[0]['match_id'] == '100'.

- **test_load_match_data_scope_half1** — tests/test_playback_adapter.py:L165-L172.
  Selected assertions: L170: len(data['frames']) == 2; L171: all((f['canonical_period'] == 1 for f in data['frames'])).

### tests/test_playback_navigation.py — 8 test definitions
**Dependencies:** football_fatigue.playback.controller: EventItem, NavigationMode, PlaybackController; pytest.

- **test_1_next_and_previous_event_select_in_chronological_order** — tests/test_playback_navigation.py:L131-L159.
  Selected assertions: L135: ctrl.state.frame_num == 100; L139: ev1 is not None and ev1.event_row_id == 'ev-1'.

- **test_2_event_type_filtering_skips_events_outside_category** — tests/test_playback_navigation.py:L163-L180.
  Selected assertions: L169: len(filtered) == 1; L170: filtered[0].event_row_id == 'ev-3'.

- **test_3_selecting_event_moves_playback_to_associated_frame** — tests/test_playback_navigation.py:L184-L192.
  Selected assertions: L188: ok is True; L189: ctrl.state.frame_num == 201.

- **test_4_multiple_events_on_same_frame_handled_consistently** — tests/test_playback_navigation.py:L196-L213.
  Selected assertions: L200: ctrl.select_event('ev-1') is True; L201: ctrl.state.frame_num == 100.

- **test_5_unmatched_events_remain_visible_and_cannot_trigger_fabricated_positions** — tests/test_playback_navigation.py:L217-L234.
  Selected assertions: L222: len(unmatched) == 1; L223: unmatched[0].event_row_id == 'ev-unmatched-p1'.

- **test_6_switching_modes_preserves_consistent_playback_position** — tests/test_playback_navigation.py:L238-L254.
  Selected assertions: L243: ctrl.state.mode == NavigationMode.EVENT; L244: ctrl.state.frame_num == 103.

- **test_7_resuming_playback_after_event_jump_advances_chronologically** — tests/test_playback_navigation.py:L258-L270.
  Selected assertions: L263: ctrl.state.frame_index == 3; L264: ctrl.state.frame_num == 103.

- **test_8_boundaries_and_gaps_handled_correctly** — tests/test_playback_navigation.py:L274-L305.
  Selected assertions: L279: ctrl.prev_event() is None; L280: ctrl.state.frame_num == 100.

### tests/test_analytics.py — 7 test definitions
**Dependencies:** football_fatigue.analytics.movement: MovementMetrics, TrajectoryPoint, classify_speed_zone, compute_player_movement; pytest.

- **test_1_constant_velocity_trajectory** — tests/test_analytics.py:L26-L54.
  A player moves along the x-axis at a constant 5.0 m/s for 2.0 seconds (60 frames at 30 fps).
  Selected assertions: L48: metrics.total_distance_m == pytest.approx(expected_distance, abs=0.05); L49: metrics.mean_speed_m_s == pytest.approx(speed, abs=0.1).

- **test_2_stationary_player** — tests/test_analytics.py:L57-L74.
  A stationary player stays at (15.0, -10.0) for 50 frames.
  Selected assertions: L71: metrics.total_distance_m == pytest.approx(0.0, abs=1e-06); L72: metrics.mean_speed_m_s == pytest.approx(0.0, abs=1e-06).

- **test_3_repeated_cadence_samples_smoothed_against_sawtooth** — tests/test_analytics.py:L77-L101.
  Simulates 15 Hz tracking on a 30 Hz recording where coordinates repeat on alternate frames.
  Selected assertions: L99: metrics.total_distance_m == pytest.approx(3.73, abs=0.3); L101: metrics.max_speed_m_s < 6.0.

- **test_4_gap_and_period_break_exclusion** — tests/test_analytics.py:L104-L132.
  Points span across a period break (e.g. halftime) and a 3-second gap.
  Selected assertions: L128: metrics.total_distance_m == pytest.approx(11.0, abs=0.1); L130: metrics.coverage.gaps_detected == 2.

- **test_5_speed_spike_rejection** — tests/test_analytics.py:L135-L152.
  A sensor glitch places the player at x=500m for one frame (instant speed > 50 m/s).
  Selected assertions: L150: metrics.coverage.speed_spikes_rejected >= 1; L152: metrics.max_speed_m_s <= 12.0.

- **test_6_coverage_and_estimated_fraction** — tests/test_analytics.py:L155-L170.
  Checks calculation of coverage fraction and estimated fraction.
  Selected assertions: L165: metrics.coverage.total_frames_evaluated == 10; L166: metrics.coverage.usable_points == 3.

- **test_7_speed_zones_and_sprinting** — tests/test_analytics.py:L173-L188.
  Tests correct classification across speed zones: walking, jogging, running, HSR, sprinting.
  Selected assertions: L175: classify_speed_zone(1.0) == 'walking'; L176: classify_speed_zone(3.0) == 'jogging'.

**Static total:** 192 test-function definitions across 11 original-system test files. No pytest items were collected or executed.

## Appendix D — Source snapshot fingerprints and verification boundaries
SHA-256 below fingerprints inspected code/docs/configuration only, not datasets. These in-memory fingerprints make the report revision checkable without creating a separate manifest. They are not a replacement for the still-open raw-source provenance/checksum manifest.

| Inspected file | SHA-256 |
| --- | --- |
| src/football_fatigue/__init__.py | ccf4ddd841100322b0197cec3acf31fd38677b0a7d89ce563181b2792141492a |
| src/football_fatigue/inventory.py | 88f678617d93b93d0207f880ff4570b0470f37b2c1e73e41138eb764c67f2e60 |
| src/football_fatigue/ingestion/__init__.py | c39d1936dd84258f991f9a23095c1b6b587b7ff25e9e88dc8abee61345fb74b2 |
| src/football_fatigue/ingestion/__main__.py | cec8e2564806835f51c86bcfbc8b969a368050d8dc49cc9deadb4663747d7e35 |
| src/football_fatigue/ingestion/models.py | e6722425876901d573428e299e22ac1275cfc1db644ac4e5f24194bd97cd4fcb |
| src/football_fatigue/ingestion/loaders.py | 88921610e01948cc3cba17bef15d6f754f1bcfb5c217b5bb1c9d85517c44b904 |
| src/football_fatigue/ingestion/validation.py | bd15c677fe0ced811a7ba53991971007a2a60c23a0c5472d11e91f54814bad94 |
| src/football_fatigue/synchronization.py | 614aecef00f86b94d7183ce4f9bcdcac3bdbf3c3ac3a8a2d33dcedf21ef56593 |
| src/football_fatigue/match_data.py | 66a2ad9b109568b2385a2cfe398780f040c9d8d392fddddd8b81ee44d75b0e5a |
| src/football_fatigue/playback/__init__.py | c4c038b821b37cd3781735062f0245b2d5931214b31fb631dce117336cc53c33 |
| src/football_fatigue/playback/__main__.py | 8f959cf23fb5e05df13c087a4db47300996d5ad3211378648242ea9eede5f931 |
| src/football_fatigue/playback/adapter.py | b797612f07426dfdc44b05488cf9b5ae08f2cd81da787d73d56215db6d0f2ec5 |
| src/football_fatigue/playback/controller.py | 65ac7b9b6b5f15636c20d65eee0f570de2783c90fd10c31d06fdb0c36ef23a33 |
| src/football_fatigue/playback/server.py | df54416cc5d3b22456e8a858e0a396d1ad38437cac78e9eec1e4a964eac16682 |
| src/football_fatigue/board/__init__.py | 220dcbe28b3004ce4d165e009a5772678a11bca6f3d41f606cfbb26135987914 |
| src/football_fatigue/board/index.html | e6612d436a85b835bfa080be7e906acd546d992f522cd9bd9f54fa0ed268c8ef |
| src/football_fatigue/analytics/__init__.py | e445330a2730dbe1718cb1124c16b9b200a17c783aabc7518d964c1f9a155d3c |
| src/football_fatigue/analytics/movement.py | b47177a752aeacecb2684efeb8d56fae50348f0b2296a7066e2d5a6e80d56052 |
| scripts/audit/audit_lib.py | c5a54061ee2476d7b88374787b5d1b6f32e5d28c88d522efed9a3e16616e4831 |
| scripts/audit/event_versions.py | ee071188c74fc7e1d86fce1452e415911ab48694d0477617c55f6ed4bcf29ef8 |
| scripts/audit/tracking_scan.py | 1af6bba7ec926c7d8bb94372fbbe9ed68de4f65181db4a945cd4cad8bf494835 |
| scripts/audit/event_alignment.py | b5058afe995f453b84f65adcffa82de2015f36ea348da003d307b4dc5e3e89bc |
| scripts/audit/extra_time_identity.py | 527a1cab543c19a739ff60640d393a1da910433ccf3799c6fdd6d5368d7a0bd3 |
| tests/test_package_installation.py | f346cc068bc46fbfdd7f4767a1f53e596bc8ae02aeaf82a91fe7feb2a1584eaf |
| tests/test_inventory.py | 7eb5bb388f56ce879e2b1db558704d2888606d02eb9473d76dfc207a67f47c33 |
| tests/test_audit_lib.py | bf1201ee9e79a553eaa82a12a91e9f4022d982a0c024098f8be2eaffa6855bcb |
| tests/test_ingestion.py | bffbb7c1bba8e60344c042aa99ad19fe07abe90fe89b5c6edcb5f17dd6c1dc69 |
| tests/test_synchronization.py | 5dfbfeed73a3f54f60539fd33b25d0a97de1026d55b2fcab69ae8812223f40a6 |
| tests/test_match_data.py | 69e7dfe2e57af1f76b0e7ef771ea7fe0a052caed8bb89dbcd855d7ea5b72598a |
| tests/test_match_data_quality.py | f96a5aa52b587e244ff20bcf002366eb8cf7d700a7fa886eb4907f879273bf14 |
| tests/test_team_handoff_docs.py | 672af4e4dd782c6ffc17f6ed8ce3b7efda95cf40db4077db494382282484a89e |
| tests/test_playback_adapter.py | f824f5bc16c664d0467795322018bc81386ed1894be46dfeff072c6c1a061e97 |
| tests/test_playback_navigation.py | e2e55df3092b623023f2c762b946ae09128cdba62604a02cbb5e198fca55320f |
| tests/test_analytics.py | f56eaf56b689ac7f57e00e00b1bbd7b1cfdcf20f1b300b1f2bfbdf22f24a80c8 |
| PROJECT_RULES.md | 5a511023f719434f5f175e280926520a1efabfdc4486a9f9468b4dcabe058f38 |
| AGENTS.md | eeadf3594d8f549f0e304bc3d7b7fa3c90dfcc9f05ddd48071b6f3319e9748ce |
| CLAUDE.md | 92cd17f3d33c6a10f72c3a2afa85e668b51011c34748bbcc06bf12f726ce1209 |
| README.md | c5075b9fb2c48769b592c8fa7eba5d7f086641cd67168dabee9ab57022c05847 |
| pyproject.toml | ab864d8bcb03563710892722148eebe2f315f51118afdafa8fd389f099de070b |
| .gitignore | 310a73ad495153d42dd710c7272aad84a5f7d1b6f7e3452910a6388791d42eeb |
| .gitattributes | c4816aeb8103718b7705139cc0eb145304b1bf36b82cbd314e1dfb72e086b444 |
| .github/workflows/checks.yml | 99281083eb984aee642d82379341f1a225f2285aad66e63879c76401444b27c6 |
| .github/pull_request_template.md | 621522b72f89d45ffc0fdbbd491947ef7dad1e8ef813ae9610b6205498b2896f |
| docs/architecture.md | e1818eee83437739487db660090be315a2bb5e63415cb2a2602dff7aed36273c |
| docs/data_policy.md | bdd2fdf2fe780ec9e5d0fb3742fa6b855cbab0bdf83cb9e6c7dd49804440f748 |
| docs/dataset_schema_audit.md | a2fad1b52fe1d61a4d2829cf926b55b00445dbb82a62ee32f4cd98f2bcf12cd6 |
| docs/event_version_comparison.md | b467fe5faec74af854fa01387c403900438113f9c40575b842da120d47d98c36 |
| docs/data_contracts.md | 097dd9d61805bbcea4c21505cf39548aaf8192e4fea6bd8772c1f6e6132c8f01 |
| docs/team_handoff.md | 9e47253e2b22c1e286ddf266faae406c13d66ffcd212eda12d68916c09658b2a |
| scripts/audit/README.md | b8a2d4ae7387d14e41931d3aafe460aba3b2cdb45dd95db86d4f36d6d2c8f30c |

### Inspection completion statement

All 18 requested chapters are present. This report documents 23 source/script files, 11 test files (192 definitions), and 16 supporting files. Later CV implementations and their tests/assets are excluded. No tests, application, server, inference, frontend build or processing pipeline was executed. No dependency or dataset was downloaded. The sole authorized write is this Markdown report; the pre-existing untracked archive was preserved. Source references apply to the inspected HEAD/working tree identified above. Physical correspondence, browser runtime, real-data all-match recomputation, source licensing and packaged-asset availability remain NOT VERIFIED.
