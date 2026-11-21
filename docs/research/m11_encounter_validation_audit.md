# M11 — Encounter validation, trajectory integrity audit and explorer repair

| Item | Detail |
|---|---|
| Date | 2026-11-10 |
| Audited dataset | `m11-sync-proximity-v2`, all 40 development matches, unchanged (output hashes and v2 code provenance still verify) |
| Audit code | `src/football_fatigue/proximity_audit.py`. It sits outside `intersections/`, because `proximity.verify_existing` hashes every module in that package |
| Audit outputs | `data/processed/m11-sync-proximity-v2-audit/`: `event_validation.csv` (one row per 0.25 m episode), `per_match.csv`, `summary.json`, `dataset_checks.json`, `manifest.json` (tolerances, rules, reason codes, input and output hashes), `screenshots/` |
| Dashboard | `src/football_fatigue/m11_dashboard/encounters.py`, `encounters_app.py` |
| Tests | `tests/test_proximity_audit.py` (new), `tests/test_m11_encounter_explorer.py` (extended) |

Not done in this stage:

- no validation or test match was read (the audit refuses them; tested);
- no event record, threshold or extraction rule was changed;
- no model was trained, nothing was labelled, nothing is committed.

## 1. Reported problems and root causes

All three screenshot events are in match **3812**. They were identified uniquely by searching the synchronized grids of all 40 development matches for period, game clock and coordinates (to within 6 mm).

| Case | Event | Root cause | Kind | Fix |
|---|---|---|---|---|
| **A**: outgoing tracks "missing" at t\* +2.002 s, A = B ≈ (16.70, 0.10) | `7d69f374b095e733`, home #4 × home #3, t\* = P1 47:40.460 | The outgoing tracks exist (30 vertices each), but **after t\* the provider gives both players identical coordinates** (a cloned estimate; `coincident_interval`, 2.00 s identical). They move only 0.9 m, and the drawing put B exactly on top of A, under the encounter markers. A frame-wide shift and an implausible jump occur at t\* (Step 1 flags). | Data artifact + dashboard (no explanation, A hidden under B) | A is drawn wider underneath B; the identical stretch is marked as a red dashed line with its duration; "Potential tracking ambiguity" label and warning; not presentation-ready |
| **B**: slider only t\* −0.467 … +2.002 s | same event | (1) **Window policy (expected):** the canonical 5 s window is centred on the integer game second nearest t\*, here t\* −2.96 … +2.04 s, so the post side is only 2.04 s. (2) **Tracking support:** B's valid run starts at −0.467 s because the step before it is an implausible jump, and it is not joined across the jump. (3) The episode continues to +2.202 s, beyond the exported window. | Expected policy + genuine data cut | The distance chart now shows the full requested window, with the part lacking shared valid tracks in grey. A caption states requested vs available. The Before/After buttons show their real offsets and are disabled when nothing exists on that side. A warning gives B's coverage and the cut reason |
| **C**: only one point, A = B ≈ (18.73, 1.40) | `548ad35e88e51867`, home #6 × home #8, same t\* | **Both players' valid tracks start exactly at t\*.** Each is cut by an implausible jump in the same frame (frame-wide shift), so there is no incoming track at all. Afterwards both are identical clones moving 0.9 m. | Data artifact | "No incoming track for Player A / B" warnings with the reason; cut markers; Before disabled; identical stretch marked; not presentation-ready |
| **D**: convincing crossing, A = B ≈ (−45.91, 0.88) | `22092c0d35089442`, home #16 × home #3, t\* = P2 05:12.288 | The detection is valid: an **exact synchronized coincidence at one instant** (d = 0 in exact whole-millimetre arithmetic; not an interval), with complete context and incoming/outgoing tracks. But both players are provider **estimates** at t\* (A 0 % and B 33 % observed vertices in the context). | Expected behaviour; low observational evidence | Labelled "Exact synchronized coincidence"; P\* is labelled as the coincidence point; not presentation-ready (estimated at t\*, low observed share) |

**Not correctable without new source data:**

- the cloned coordinates and the jumps in A and C;
- B's missing 2.5 s before t\*;
- any trajectory outside the exported window. The NPZ stores only the canonical window; a longer context must be re-derived from the Step 1 caches as a new versioned export.

The dashboard never extends or bridges these.

## 2. Dataset validation summary

These are counts of the 30,438 0.25 m episodes. Reason codes and their texts are in `manifest.json["reasons"]`.

| Status | Events | Definition |
|---|---|---|
| Audited | **30,438** | All development episodes at R = 0.25 m |
| `mathematically_valid` | **30,438** | Independent recomputation agrees (§3) |
| `tracking_support_valid` | **30,438** | Every exported vertex equals the RAW cache; time order, period, gap and jump rules, grid references all hold |
| `incoming_complete` | 25,602 | Both players' valid tracks cover the window before t\* |
| `outgoing_complete` | 26,607 | Both players' valid tracks cover the window after t\* |
| `context_complete` | 24,088 | Both sides, both players |
| `provider_artifact_suspected` | **6,578** | Identical coordinates over time (2,719, including 22 outside the episode itself), duplicated provider position (4,014), or a critical Step 1 flag (implausible jump 4,719; frame-wide shift 975; discontinuity 38; label transfer 15; reciprocal exchange 2) |
| `presentation_ready` | **1,622** | All of the conditions listed below the table |
| Not presentation-ready | 28,816 | Kept in the dataset and shown under "All encounters" |
| Unresolved | 0 | No event fails detection or support checks |

An event is `presentation_ready` when all of these hold:

- it is mathematically valid and its tracking support is valid;
- incoming and outgoing tracks are complete for both players;
- no provider artifact is suspected;
- both players are observed at t\*;
- at least 50 % of each player's context vertices are observed;
- each player moves at least 1 m in the context.

**By quality at t\*:**

| Quality | Events | Ready | Artifact suspected |
|---|---|---|---|
| observed/observed | 2,619 | 1,622 | 499 |
| observed/estimated | 4,761 | 0 | 877 |
| estimated/estimated | 23,058 | 0 | 5,202 |

**Relation to the published `high_quality_candidates` (1,917).** Presentation-ready is a strict subset. The 295 high-quality candidates that are not ready all fail only the 50 % observed-share rule.

**Per match:** 19 (10512) to 89 (3812) ready events; every match has some.

**Boundaries:**

| Edge | d = R verified | Support cut | Episode continues beyond the exported window |
|---|---|---|---|
| Start | 28,110 | 2,245 | 83 |
| End | 27,403 | 1,353 | 1,682 |

An edge beyond the exported window cannot be checked from the export; it is consistent, not failed.

**Coverage** is shown as quantiles 0 / 10 / 50 / 90 / 100 %, in seconds:

| Measure | 0 % | 10 % | 50 % | 90 % | 100 % |
|---|---|---|---|---|---|
| Requested before t\* | 0 | 2.10 | 2.52 | 2.93 | 3.18 |
| Requested after t\* | 0.004 | 2.07 | 2.48 | 2.90 | 3.00 |
| Shared valid before t\* | 0 | 0.32 | 2.40 | 2.87 | 3.15 |
| Shared valid after t\* | 0 | 1.25 | 2.35 | 2.84 | 2.99 |

Requested values below 2.0 s or above 3.0 s come from windows clipped at a period boundary.

**Dataset-wide checks (`dataset_checks.json`, all pass):**

- per-match output hashes equal the manifests;
- index and match manifest hashes;
- NPZ rows equal JSONL lines;
- event IDs are unique per match and globally, and reproduce from `event_key`;
- no duplicate records;
- deterministic order: (period, slot A, slot B, t_start) at every threshold;
- nesting 0 ⊂ 0.10 ⊂ 0.25 ⊂ 0.50 m via parent/children;
- per-match and sensitivity counts equal `summary.json` (2,750 / 13,915 / 30,438 / 61,106);
- the index has exactly the audited keys, and `events_line = npz_event_index`.

## 3. Algorithm verification

**Detection (unchanged, `intersections/proximity.py`).**

Each provider slot is piecewise linear between its update instants. It is joined only across Step 1 valid steps: the same period, at most 0.5 s unobserved, no jump faster than 12 m/s and longer than 1 m, and no unattributed sample.

For two players A and B, time is split at the union of both players' sample instants. **Both positions are evaluated at the same t**, never at B(t + Δt). On each interval, the relative position r(s) = r₀ + s·e (s ∈ [0, 1]) is linear, so

  d²(s) = |r₀|² + 2s·(r₀·e) + s²|e|²,  s\* = clip(−r₀·e / |e|², 0, 1).

- **Episode:** the within-R parts {s : d²(s) ≤ R²} are merged while they touch or overlap. An episode is therefore a maximal interval with d(t) ≤ R. Its boundaries are where d = R, or where a player's valid support ends (a support cut).
- **t\*:** the instant of the minimum within the episode, earliest on ties.
- **A(t\*), B(t\*):** the interpolated positions at t\*.
- **P\* = (A(t\*) + B(t\*)) / 2** is a midpoint. It is a point on both paths only when d_min = 0. When d_min > 0, the paths need not intersect at all, or may cross at different times (§ below).
- **Exact decisions:** values within 1e-3 mm² of R² are decided again with exact rational arithmetic.

**Zero distance arises in two ways:**

- an **exact synchronized coincidence**: r passes exactly through 0 at one instant (40 events; 26 of them both observed);
- an **identical interval**: r₀ = e = 0, with both coordinates identical over time. These are provider clones (2,697 `coincident_interval` episodes).

**Geometric crossing ≠ encounter.** 26,985 event contexts contain a spatial crossing of the two paths. In 24,397 of these, the players pass the crossing point at *different* times, which is not a synchronized encounter. This is a descriptive audit field only; no crossing extraction was added to the pipeline.

**Independent re-derivation (audit).** This is separate numpy code that shares nothing with `proximity.py`. It works from the exported vertices after they have been checked against the RAW cache.

| Check | Tolerance |
|---|---|
| Vertex vs RAW cache: frame number, video time, x/y of the claimed slot, state AVAILABLE, visibility, period | t ≤ 1 µs; x, y ≤ 0.5 mm (vertices are whole mm) |
| Slot identity: side, jersey, player ID, team ID vs cache roster | exact |
| Joined steps: longest wait for an available sample, measured on the cache states | ≤ 0.5 s + 1 frame |
| Joined steps: implausible jump | not (> 1 m and > 12 m/s) |
| Recomputed episode minimum vs d_min | ≤ 1 µm |
| d at recorded t\* vs minimum | ≤ 1 µm |
| A(t\*), B(t\*) vs record | ≤ 1 µm |
| P\* vs midpoint of the record's positions | ≤ 1 nm |
| Within the episode, d ≤ R | R + 1 µm |
| d at an interior boundary vs R | ≤ 10 µm |
| t\* within the shared support and the episode | ≤ 1 µs |
| Duration and contact class | ≤ 1 ns |
| `exact_coincidence` ⇔ d_min = 0; `coincident_interval` ⇔ an identical interval within the episode | identical = d ≤ 1 nm |
| Synchronized grid = union of instants plus t_start, t\*, t_end; d = |A − B|; sample masks | 1 nm |
| A track flagged complete reaches the window edge | within one joinable segment (0.5 s + 1 frame) |

**Defects found during the audit (all in the audit code, fixed before the final run):**

1. The first version took the minimum over the whole 5 s context. In 257 events of 3812, another, closer approach of the same pair lies in the window. The minimum is now taken within the episode.
2. A completeness tolerance of 2 frames flagged one event (3851 `45b9985b5eb589f9`). Its track continues into the window, but the joined 0.18 s segment straddling the window start is not exported, because the export keeps segments fully inside the window. The tolerance is now one joinable segment.

No defect was found in the extraction, the exported records or the trajectories.

**What the source data do not support:** visual occlusion, verified identities (provider slots only), positions inside cloned or estimated stretches as measurements, and anything outside the exported windows.

## 4. Dashboard fixes

- **Default list:** *Presentation-ready (audited)*. "All encounters (incl. incomplete / problematic)" and the quality views keep every event available. Presentation mode always uses the audited list at 0.25 m, and its selector is locked.
- **Layering:** Player A (cyan, wide) is drawn under Player B (yellow, narrow), so coinciding tracks stay visible.
  - identical stretches: red dashed line, with the identical duration in the legend;
  - valid-track cut points: red bar markers, with the reason on hover;
  - the "selected time" marker is hollow;
  - A(t\*), B(t\*) and P\* are drawn above the paths.
- **Labels:** "Closest synchronized approach", "Exact synchronized coincidence" or "Potential tracking ambiguity". P\* reads "midpoint of A(t\*), B(t\*) … not a path intersection" whenever d_min > 0.
- **Coverage messages:** "No incoming track for Player A … (cut by a tracking jump)" and "Player B's incoming track covers 0.47 s of the 2.96 s requested …".
- **Time controls:**
  - the slider covers only instants where both players have valid tracks, labelled with absolute game time plus t\* offset;
  - Before/After jump to real instants about ±1 s from t\*; their labels show the actual offset, and they are disabled when there is none;
  - the distance chart spans the requested window, with grey shading where no shared valid track exists (above 0.1 s);
  - a caption states requested vs available.
- **Details:** a compact table (players, match/event, type, minimum synchronized distance and time, duration, quality, context). It also shows an audit line (detection / tracking / incoming / outgoing / artifact) and presentation readiness. Readable reasons appear when the event is not ready. Raw fields stay under "Advanced details".
- **Presentation mode:** metrics for event time, minimum synchronized distance, quality, 5 s context and "k of n ready". Warnings are always shown; the review form and advanced filters are hidden.

## 5. Evidence

**Commands**, run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m football_fatigue.proximity_audit --root data/raw --jobs 3
.\.venv\Scripts\python.exe -m pytest tests/test_proximity_audit.py tests/test_m11_encounter_explorer.py tests/test_m11_dashboard.py tests/test_intersections_proximity.py -q
```

**Results:**

- the audit wrote 30,438 rows;
- **89 passed**:
  - 16 audit tests;
  - 18 explorer tests;
  - 29 dashboard tests;
  - 26 proximity tests, including the v2 code-provenance check (`verify_existing` is None).

**The new tests cover:**

- missing outgoing support;
- incomplete windows;
- cloned estimated coordinates;
- a complete observed encounter (ready);
- an exact synchronized coincidence;
- nonzero synchronized proximity without intersection;
- a spatial crossing at different times;
- closest approach restricted to the episode;
- tampered vertex, relabelled visibility, wrong period, 0.6 s unobserved gap and 5 m jump, all detected;
- a valid 0.47 s gap kept;
- mismatched d_min / A(t\*) reported;
- the three screenshot cases (audit classification and rendering);
- layering, legend and zoom;
- slider bounds and disabled Before (case C);
- the presentation list equal to the audited ready set;
- navigation and state persistence;
- the written audit consistent with recomputation (3812 recomputed and compared field by field).

**Browser check.** Streamlit was driven in headless Edge at 1,600 px width. Screenshots are in `data/processed/m11-sync-proximity-v2-audit/screenshots/`:

- `caseA.png`, `caseC.png`, `caseD.png`;
- `presentation_ready.png` (event 1 of 89 ready, 3812);
- `presentation_before.png` (Before = t\* −1.02 s).

**Representative validated example:** 3812 `#16`, home #21 × away #21, P1 35:53.343. It is a closest synchronized approach of 11.1 cm, both observed, with complete incoming and outgoing tracks.

**Remaining scientific limitations:**

- Readiness thresholds (50 % observed, 1 m motion) are presentation rules, not dataset rules. They are recorded in the manifest and can be changed without touching the data.
- 24 ready events have d_min = 0 at a single instant. On whole-millimetre provider data such a touch can come from rounding, so they are labelled as coincidences, not as contact.
- Context is the canonical window only. The window is not centred on t\* (each side is 2–3 s).
- Provider slots are not verified identities. Proximity is not visual occlusion. Manual reviews are audit notes, not labels.
