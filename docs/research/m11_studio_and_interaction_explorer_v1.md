# M11 Studio: architecture, controls, interaction events and self-intersections (v1)

**Date:** 2026-11-20

**Scope:** development (train/CV) matches only. Every loader validates match membership before reading anything.

## 1. Launch

```powershell
cd F:\Football-Fatigue-Project
.\.venv\Scripts\python.exe -m football_fatigue.m11_dashboard --port 8620 studio
```

Or double-click `scripts\m11_studio.cmd`, then open **http://127.0.0.1:8620**.

The earlier apps are preserved:

| App | Command |
|---|---|
| Trajectory Recovery Lab | `--port 8610 lab` (`scripts\m11_lab.cmd`) |
| Encounter explorer | `--port 8503 encounters` |
| Step 8 dashboard | (no subcommand) |

## 2. Framework decision

Kept **Streamlit + Plotly**.

**Why:**
- The pitch drawing (`pitch.py`), the lab board (`lab_board.py`), inference, caching and the tests already exist.
- Plotly animation frames give smooth play / pause / scrub in the browser without server round-trips.
- Row selection in `st.dataframe` (Streamlit 1.67) gives click-to-inspect.
- No new dependency.

**Rejected for now:** a separate React / Vue front end with an API. It would offer frame-accurate canvas rendering of full 90-minute matches, but adds a second stack, a build step and an API surface with no measured need. Revisit if windows longer than about 2 minutes at 15 Hz become necessary.

## 3. Architecture

| Layer | Module | Role |
|---|---|---|
| Data guard | `m11_phase2.common.checked` → `interaction_catalogue.run.assert_allowed` | Refuses held-out / official / unknown matches before reading |
| Catalogue query | `m11_explorer/events.py` | `window_events()`: every canonical edge / crossing / group in a window, with filters and semantics |
| Tracks | `m11_explorer/tracks.py` | Verified Step 1 RAW cache plus catalogue segments; playback frames; player paths |
| Self-crossings | `m11_explorer/selfcross.py` | One-player loop detector (§6) |
| Live physics | `m11_explorer/physics_live.py` | Phase 2 Kalman / CV / CA / nearest / direction on the lab scenario's model inputs |
| N × M | `m11_explorer/multiplayer.py` | Schema, synthetic curriculum, Kalman + Hungarian-null solver, metrics, exploratory real groups |
| Review | `m11_explorer/review.py` | 125-item queue; append-only decisions; timestamped exports |
| Figures | `m11_dashboard/studio_figs.py` | Pitch animation, timeline, distance plot, N × M plot, assignment heatmap |
| App | `m11_dashboard/studio_app.py` | Five workspaces |
| Lab back end (reused) | `m11_dashboard/lab.py`, `lab_board.py`, `m11_infer` | Scenario generator, GRU C / HGB C inference, routing, truth separation |

## 4. Workspaces and controls

### A. Match & Interaction Explorer

**Controls:**
- match, period;
- window start (slider, period clock) and length (5–60 s);
- radius (0.25–2 m);
- playback rate (5 / 10 / 15 Hz);
- filters: team relation, minimum duration, players, crossing types, self-crossings on/off.

**Pitch:**
- All players are animated, coloured by team.
- Shirt numbers are shown.
- ESTIMATED samples are drawn hollow.
- Faint full-window paths are shown; highlighted players are brighter.
- Event markers:

| Marker | Event |
|---|---|
| ★ | Proximity (closest point) |
| Red ✕ | Synchronous crossing |
| Yellow ✕ | Near-synchronous crossing |
| ○ | Asynchronous crossing |
| Purple ◆ | Self-crossing |

- Play / pause / slider work client-side; scroll-zoom and box-zoom are available.
- The PNG export button is in the toolbar.

**Timeline:**
- One row per pair, with proximity bars and both crossing passage times joined by a dotted line.
- Self-crossings are on their own `self <slot>` rows.

**Tables (click a row to select it and zoom to it):**
- Proximity
- Crossings
- Groups (connected size **and** max simultaneous; `inflated` flag)
- Players & pairs (repeat and sequential counts)
- Self-crossings
- Definitions

**Event inspector:**
- A plain-language reason for inclusion and the quality flags (`explain_edge`).
- A distance-vs-time plot with the radius line and ESTIMATED samples.
- "Use this encounter in the Recovery Lab": sends the matching validated Stage 2 candidate, if one exists.
- A group row sends the group to the N × M workspace.

**Navigation:** previous / next window.

### B. Identity Recovery Lab

**Flow:** Before → hidden interval → after → prediction → ground truth.

**Steps:**
1. Choose a match and encounter, or a random one.
   - Dropdown labels show **public fields only**: number, clock, team relation.
2. Set the scenario parameters: gap 0.2–2.4 s, centre offset ±0.5 s, noise 0–25 cm, dropout mode, context 0.5–2 s, seed.
   - Generation is reproducible through the Milestone 2 generator; labels are re-validated when the gap or centre changes.
3. The board (`lab_board.board`) shows:
   - Player 1 / 2 (incoming, known identity);
   - Track X / Y (anonymous, neutral colours);
   - all other players.
4. **▶ Run prediction:**
   - live GRU C and HGB C (the fold that never saw the match) with the routing decision;
   - the seven deterministic baselines on the same model-visible arrays (margins, Kalman gate);
   - LSTM / SVM shown **unavailable**.
5. **👁 Reveal** (enabled only after a prediction):
   - provider pairing, hidden paths, ✔ / ✘ per model.
   - **🙈 Hide truth** hides it again.

**Expanders:**
- model availability;
- model inputs (the 36 features and the assumptions);
- validity checks.

**Leak fixes (audit D1 / D2):**
- `lab.encounter_label` builds dropdown labels from `PRE_REVEAL_LABEL_FIELDS`. Crossing / movement / closest distance / in-gap status are `REVEAL_ONLY_FIELDS`.
- The original lab (`lab_app.py`) uses the same function.
- Its gallery shows neutral names "Example set A–I" unless "Show real category names" is switched on.

### C. Multi-player (N × M)

**Sources:**
- synthetic curriculum level and seed;
- a real simultaneous group from the explorer (**exploratory**);
- an uploaded `m11-multiplayer-association-v1` JSON (examples in `docs/research/m11_multiplayer_examples/`).

**Displays:**
- Incoming tracks are coloured and labelled. Outgoing tracks are neutral grey with anonymous ids until reveal.
- Chosen links are drawn as arrows; after reveal they turn green (correct) or red (wrong).
- The assignment-cost heatmap shows Mahalanobis d², "gated" cells and ✔ for chosen links.
- After reveal: exact assignment, link and null counts, identity switches.

The binary models are explicitly **not applicable** here.

### D. Review queue

**Contents:** 100 Phase 2 stratified events plus 25 severe motion contradictions.

**Per item:**
- previous / next / jump;
- zoomed animation of both players and neighbours;
- distance plot;
- the Phase 2 static PNG.

**Questions:**
- positions plausible;
- players actually close;
- identity continuity;
- artefact type (smooth ESTIMATED, clone-like, jump, swap);
- unexplained contradiction;
- whether visual occlusion is checkable.

**Saving and export:**
- Decisions are saved per named reviewer, **append-only** (`data/processed/m11-human-review-v1/decisions/<reviewer>.jsonl`).
- Export writes a new timestamped CSV in which unsaved items stay "not reviewed".
- Nothing is ever marked reviewed automatically.

### E. Evidence & provenance

**Model comparison table:**
- historical OOF (GRU 0.900, LSTM 0.858, SVM 0.859, HGB 0.870), with fairness notes;
- the ten Phase 2 deterministic methods;
- CPU-pilot rows, labelled PILOT.

**Other views:**
- physics by gap table;
- Phase 2 strata;
- learned-model accuracy by physics tier;
- fair-benchmark plan status;
- checkpoint SHA-256 list (`m11-infer-v1/bundles_v1.json`);
- presence of every versioned output folder.

## 5. Interaction event definitions (`events.EVENT_TYPES`)

| Event | Definition | Source |
|---|---|---|
| proximity | Two players within R for a contiguous interval (edge; canonical `edge_id`) | `edges.parquet` |
| crossing_synchronous | Paths cross and both pass the point within 0.3 s | `crossings.parquet` (`abs_dt_s ≤ 0.3`) |
| crossing_near_synchronous | 0.3 s < \|tA − tB\| ≤ 1.0 s | same |
| crossing_asynchronous | \|tA − tB\| > 1.0 s: same place, different time | same |
| repeated_pair | The same pair has more than one edge in the window | derived (`pairs.n_edges > 1`) |
| sequential_player | One player has edges with ≥ 2 partners | derived (`players.n_partners > 1`) |
| simultaneous_group | Link-0 group; `max_simultaneous_players` counts players close at one instant | `groups.parquet` (link 0) |
| connected_group | Link-1 group (chained contacts); its size is **not** simultaneous | `groups.parquet` (link 1; max-simultaneous filled from member link-0 groups) |
| self_crossing | One player's path crosses an earlier part of itself | `selfcross.py` |
| synthetic_gap | Observations hidden on purpose (Recovery Lab) | lab generator |
| verified_occlusion | **Not available** in PFF tracking | — |

**How the window query works:**
- It returns *every* row overlapping [t0, t1] (edges by interval; crossings when either passage time is inside).
- Rows are deduplicated by canonical id. No first-only filtering.
- Typical query time: about 1 ms once the match's tables are cached (435 ms including the first load).

## 6. Self-intersections (`selfcross.py`)

**Method:**
1. Per continuous provider chain (never across a chain break), keep a vertex every 0.25 m of travel (arc-length thinning). This removes stationary jitter.
2. Candidate segment pairs share a 1 m grid neighbourhood; long segments are tested against their bounding box. Adjacent segments are never tested. Intersections must be strictly proper.
3. A crossing is kept when:
   - ≥ 1 s separates the two visits;
   - ≥ 2 m of path lies between them;
   - the loop reaches ≥ 0.5 m from the crossing point.
4. Hits within 0.5 s of an earlier kept hit are merged.
5. Each kept crossing reports: slot, chain, period, position, both visit times, time separation, path length, loop extent, merged hit count, ESTIMATED fraction over the loop, and a quality flag (`low` when ≥ 50 % ESTIMATED).

**Semantics:** a self-crossing is descriptive. It is never an association candidate, never a two-player event and never occlusion.

**Tests:**

| Case | Expected |
|---|---|
| Loop | 1 |
| Figure-eight | 1 |
| Spiral | 0 |
| Shared-endpoint zig-zag | 0 |
| Stationary jitter | 0 |
| Tiny 0.2 m wiggle | 0 |
| Chain break | 0 |
| Missing samples | Dropped, loop still found |

**Catalogue:** `python -m football_fatigue.m11_explorer selfcross` writes `data/processed/m11-self-crossings-v1/`.

**Result [V]** (32 development matches, CPU, 25 min): **927,181** self-crossings (18,323–42,568 per match). Time between the two
visits: median 81 s (p10 5.1 s, p90 598 s); path between visits median 129 m. Most are therefore *returns to the same place much
later*, which the geometric definition includes; tight loops are a minority: 90,659 (9.8 %) within 5 s, 166,676 (18 %) within 10 s.
67 % are `low` quality (≥ 50 % ESTIMATED over the loop) overall, 55–57 % among the short ones, so many short loops sit in provider
ESTIMATED segments. In the Studio the window bounds both visits, so a 10 s window only ever shows crossings with dt ≤ 10 s.
Recommendation: when a 'local loop' event set is needed, filter dt ≤ 5–10 s and report the ESTIMATED share; never treat either as
an interaction.

## 7. GUI acceptance status

| # | Requirement | Status |
|---|---|---|
| 1 | One Windows command | ✔ `scripts\m11_studio.cmd` |
| 2–4 | Select match; play / scrub; all players | ✔ (verified headlessly: renders without exception; Plotly frames) |
| 5–6 | Any 5 s window; every canonical event listed | ✔ window lengths 5–60 s; unit tests on synthetic catalogue tables |
| 7 | Select an event and zoom | ✔ row selection → zoom box |
| 8 | Several events of one player | ✔ Players & pairs table; tested |
| 9 | Simultaneous vs sequential | ✔ `context` column; group table with both sizes |
| 10 | Self-crossing not classified as an interaction | ✔ separate marker, row and table |
| 11–13 | Valid synthetic gap; before / gap / after; genuine GRU + HGB inference | ✔ headless click-through: prediction ran |
| 14 | Kalman on the same example | ✔ |
| 15 | Answer before truth | ✔ Reveal disabled until a prediction |
| 16 | Reveal pairing and hidden paths | ✔ |
| 17 | Successful / failed / uncertain events | Partial. Random and explorer-sent encounters give mixed outcomes. The blind gallery (outcome categories) is in the original lab app, not yet in the Studio. |
| 18 | Phase 2 reviewer examples | ✔ Review queue |
| 19 | Model and data provenance | ✔ Evidence tab |
| 20 | Unavailable families marked | ✔ LSTM / SVM unavailable |
| 21 | Synthetic N × M without the binary model | ✔ |

**Smoothness of playback in a real browser is not machine-verified.** Headless tests confirm that the pages render and that the clicks execute; they do not measure frame rate.
