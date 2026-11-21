# Session B: current-state handoff (video detection, tracking, occlusion, dashboard)

Date: 2026-11-18 · Author: Claude Code (Session B) · Repository: `F:\Football-Fatigue-Video`

This document records the repository as it actually is on 2026-11-18. It was written from read-only inspection of code,
Git metadata, saved artifacts, the pytest cache and the session transcript. Nothing was implemented, re-run or changed
to write it, apart from creating this file.

## Evidence labels

| label | meaning |
|---|---|
| **VERIFIED** | checked directly while writing this document (file present, value read from a saved artifact, Git output, code read) |
| **DOCUMENTED** | stated in an existing repository document written by Session B; not re-derived now |
| **HISTORICAL** | happened in an earlier Session B turn (for example a test run); there is a trace, but it was not re-executed now |
| **UNVERIFIED** | no evidence found |
| **PLANNED** | proposed or designed, not executed |

---

## 0. Answer first: was the approved T0 milestone implemented?

**No. The approved T0 milestone was not implemented.** Nothing in the code, artifacts or tests changed after the
detection diagnosis finished on 2026-11-18 at 02:14 local time. (VERIFIED)

The evidence:

1. **Newest files.** The newest Session B source file is `src/football_fatigue/video/dashboard/detection_tab.py`
   (02:02). The newest artifact is `data/processed/video/soccertrack-118576/detection/replay-v1/TG6C3/result.json`
   (02:14). The newest doc is `docs/research/VIDEO_DETECTION_DIAGNOSIS_v1.md` (02:14). Nothing was written after the
   diagnosis milestone. (VERIFIED)
2. **The approval never reached the session.** The task "SESSION B — APPROVED: APPLY THE VERIFIED DETECTION FIX AND
   VALIDATE THE VIDEO PIPELINE" does not appear as a user message in any Claude Code transcript under
   `C:\Users\DELL\.claude\projects\`. Its only occurrences are quotations inside the current verification request
   (transcript `358a4fca-…jsonl`, line 2781 onwards). The approval was therefore written but never delivered to
   Session B. (VERIFIED)
3. **No T0 configuration outside the diagnosis module.** `T0` exists only as a replay variant in
   `src/football_fatigue/video/detection/replay.py:40`. There is no T0 selection in the occlusion engine CLI
   (`occlusion/__main__.py`), in `sources.py`, or in the main dashboard viewer. (VERIFIED by grep)
4. **No T0 occlusion runs.** Only `step1` and `v2` engine runs exist, both made on 2026-11-17 from the production (B0)
   replay. (VERIFIED)

What *does* exist for T0 is the CPU **diagnostic replay** from the previous milestone. It holds T0 tracks and TrackEval
metrics for all four development matches, and T0 can be selected in the dashboard's *Detection diagnostics* tab.
That is experimental evidence for T0, not an adopted production configuration. Details are in §8.

---

## 1. Executive project context and Session B responsibilities

* **The project.** Football Fatigue measures player load from video and tracking data. Two Claude Code sessions work on
  it in parallel:
  * **Session A** (`F:\Football-Fatigue-Project`, branch `feature/m11-identity-finalization`): identity recovery (M11),
    multi-camera work, and the shared vision code (`football_fatigue.vision`: detector, detection cache, ByteTrack
    tracker, MOT evaluation).
  * **Session B** (this worktree, `F:\Football-Fatigue-Video`, branch `feature/video-occlusion-events`): the video
    **occlusion-event engine**, its review dashboard and evaluation, and, since 2026-11-18, the **detection-quality
    diagnosis**. (DOCUMENTED, VERIFIED via `git worktree list`)
* **Session B rules:**
  * Read Session A's shared data and checkpoints without modifying them.
  * Do not depend on Session A's M11 models or dashboard.
  * Never touch the reserved sequences.
  * No GPU inference, model downloads or training without explicit approval.
  * Git is read-only for the agent; the user commits.
  * Outputs only under `data/processed/video/` of this worktree.
* **Out of scope for Session B:**
  * persistent player identities, tactical roles and 22-player identity recovery;
  * pitch-metre coordinates and physical-load metrics.
  * Session B works in image pixels with temporary tracker IDs.

## 2. Current repository state and version information

All VERIFIED (`git` read-only commands, 2026-11-18).

| item | value |
|---|---|
| Repository root | `F:/Football-Fatigue-Video` (a Git worktree of the same repository as `F:/Football-Fatigue-Project`) |
| Branch | `feature/video-occlusion-events` |
| HEAD | `1bd016d8099e32d94807aaca805028e8621007f8` (shared with Session A's worktree, which is on `feature/m11-identity-finalization`) |
| Session B commits | **none**. All Session B work is untracked. |
| Staged changes | none |
| Modified tracked files | none |

**Latest 8 commits** (all of them precede Session B's video work):

```
1bd016d 2026-11-17 21:13 Add M11 identity recovery research, benchmarks and dashboards
1458e1b 2026-11-06 22:50 chore: add Step 1 handoff, ignore demo asset bundle, scrub local username
8505c5c 2026-11-06 22:46 docs: add project master context, technical walkthrough and advisor reports
4653047 2026-11-06 22:46 feat(m11): add encounter catalogue, identity matching and locked-test evaluation
3ec9441 2026-11-06 22:46 feat(multicam): add unified multi-camera lab
936bc07 2026-11-06 22:45 feat(demo): add coach dashboard demo with overlay continuity and review layer
1d020fc 2026-11-06 22:45 feat(analytics): add coach analytics, motion and observed-quantity modules
2e18001 2026-11-06 22:45 feat(vision): add detection cache, identity association and MOT evaluation
```

**`git status --short --branch`** (before this file was created):

```
## feature/video-occlusion-events
?? docs/research/VIDEO_DETECTION_DIAGNOSIS_v1.md
?? docs/research/VIDEO_ENGINE_DECISION_REPORT_v1.md
?? docs/research/VIDEO_V1_OCCLUSION_PLAN.md
?? docs/research/VIDEO_V1_STEP2_EVALUATION.md
?? docs/video_detection_diagnostics.md
?? docs/video_occlusion_step1.md
?? docs/video_occlusion_step2.md
?? src/football_fatigue/video/
?? tests/test_video_detection.py
?? tests/test_video_occlusion.py
?? tests/test_video_step2.py
```

This handoff adds `?? docs/handoffs/`.

**The untracked files and their purposes:**

| path | purpose | last written |
|---|---|---|
| `docs/research/VIDEO_ENGINE_DECISION_REPORT_v1.md` | engine choice report: YOLO26-L verified; merge NMS hides occlusions | 2026-11-17 21:35 |
| `docs/research/VIDEO_V1_OCCLUSION_PLAN.md` | V1 occlusion-engine plan | 2026-11-17 21:47 |
| `docs/video_occlusion_step1.md` | Step 1 engine guide (window demo) | 2026-11-17 22:10 |
| `docs/research/VIDEO_V1_STEP2_EVALUATION.md` | Step 2 evaluation: step1 vs v2 on 4 development matches | 2026-11-17 23:11 |
| `docs/video_occlusion_step2.md` | Step 2 dashboard guide, run-folder contract, upload design | 2026-11-17 23:11 |
| `docs/research/VIDEO_DETECTION_DIAGNOSIS_v1.md` | detection-quality diagnosis and B0/T0 replay results | 2026-11-18 02:14 |
| `docs/video_detection_diagnostics.md` | diagnosis commands and detection-tab guide | 2026-11-18 02:03 |
| `src/football_fatigue/video/` | all Session B code (map in §3) | 2026-11-17 22:02 to 2026-11-18 02:02 |
| `tests/test_video_occlusion.py` | Step 1 engine tests | 2026-11-17 22:06 |
| `tests/test_video_step2.py` | Step 2 engine, evaluation and dashboard tests | 2026-11-17 23:09 |
| `tests/test_video_detection.py` | diagnosis tests (15) | 2026-11-18 02:01 |

`data/processed/video/` holds the outputs. It is not listed by `git status`, so it is presumably ignored.

**The actual state versus the last historical Claude response.** The last substantive Session B response was the
detection-diagnosis report (deliverables A–H). It recommended T0 and asked for approval. Nothing happened in the
repository after it. The repository matches that report, and it does **not** reflect any post-approval
implementation. (VERIFIED)

## 3. Architecture and module/file map

### 3.1 End-to-end pipeline as implemented

```
video (MP4)
  │  decoded by Session A (PyAV); Session B re-decodes only for display/rendering (ExactFrameReader)
  ▼
YOLO26-L detection, tiled ── Session A: football_fatigue.vision.detector.YOLOPlayerDetector
  │  1280×1280 tiles, 20% overlap, imgsz 1920, batch 2, weights player_detector.pt (sha 05ee5d6c…)
  ▼
tile merge / NMS ── Session A: edge-aware person-agnostic merge, IoU 0.5, cross-role IoU 0.7
  ▼
detection cache (post-merge boxes only, conf > 0.1, each with passes_turf_filter flag)
  │  Session A: football_fatigue.vision.detection_cache  →  <data_root>/processed/vision/<cache_run>/detection_cache/
  ▼
tracker-input filter ── production (B0): person class, conf > 0.4, passes_turf_filter == True
  ▼
ByteTrack ── Session A: football_fatigue.vision.tracker.MultiObjectTracker (supervision)
  │  activation 0.5, lost buffer 75, match 0.95  →  <data_root>/processed/vision/<replay_run>/frames.jsonl
  ▼
occlusion engine (Session B) ── football_fatigue.video.occlusion, engine 0.2.0, profiles step1 / v2
  │  → data/processed/video/<source>/occlusion/0.2.0/<run_id>/{manifest.json, events.jsonl, frames.jsonl, suppressed*.jsonl}
  ▼
evaluation (Session B, dev only) ── occlusion/evaluate.py → occlusion/evaluation/<run_id>.eval-v1.json
  ▼
Streamlit dashboard (Session B) ── football_fatigue.video.dashboard, http://127.0.0.1:5830
```

All facts in this diagram are DOCUMENTED in the engine decision report and the diagnosis report. The input paths are
VERIFIED in `detection/diagnose.py:36-41` and `sources.py:31-53`.

Notes on each stage:

* **Video decoding.** Session B's `ExactFrameReader` (`sources.py`) and `multicam_lab.media.probe_video` are used for
  display, rendering and probing only. Session B never runs the detector in production. (VERIFIED)
* **Track refinement.** There is no separate refinement stage. The tracker's output is used as-is. The occlusion engine
  only adds *predicted* boxes for lost or hidden players. The v2 profile caps the predicted velocity at 0.25 box heights
  per frame for 5 frames and then holds the box. (DOCUMENTED)
* **Identity handling.** There is none beyond ByteTrack's temporary track IDs (`T##`). No persistent identity, no ReID,
  no Global ID. (VERIFIED: no identity module under `video/`)
* **Video-to-pitch visualisation.** **Not implemented in Session B.** The only pitch geometry is the image-space
  play-area polygon. It is built from SoccerTrack's annotated pitch keypoints (`sources.play_area_polygon`) and is used to
  suppress off-pitch tracks. There are no pitch-metre coordinates and no minimap. (VERIFIED by grep: no homography code
  under `video/`)

### 3.2 Module map (`src/football_fatigue/video/`)

| module | role |
|---|---|
| `__init__.py` | `ENGINE_VERSION = "0.2.0"` |
| `schema.py` | event schema `football_fatigue.video.occlusion_event` 1.1.0 (`OcclusionEvent`, `Member`, `SourceInfo`) |
| `sources.py` | `SOURCES` (4 development matches); `RESERVED_MATCH_IDS` and `guard_not_reserved`; `play_area_polygon`; `ExactFrameReader`; `load_source`; `load_inputs` (checks every tracked box against the cache) |
| `occlusion/engine.py` | `OcclusionEngine`; `PROFILES = {"step1": EngineConfig(), "v2": EngineConfig(grouping="pair", prediction="damped", use_play_area=True, pair_switch_cover=0.5)}` |
| `occlusion/geometry.py` | box geometry helpers |
| `occlusion/evaluate.py` | GT episodes, event and claim-level scoring, detection-loss scoring (`eval-v1`) |
| `occlusion/report.py` | step1 vs v2 tables |
| `occlusion/__main__.py` | CLI `run` / `evaluate`. The tracks always come from `SOURCES[...]["replay_run"]`, or `--replay-run` naming a Session A replay folder. There is **no T0 option**. |
| `render.py`, `render_focus.py`, `overlay.py` | annotated MP4 and strips (focus layout), shared overlay drawing |
| `dashboard/app.py`, `state.py`, `data.py`, `__main__.py` | Streamlit app. Tabs: Viewer, Events, Results, 🔬 Detection diagnostics, Upload (planned). |
| `dashboard/detection_tab.py` | the detection-diagnostics tab (§10) |
| `detection/__init__.py` | `DIAG_VERSION = "diag-v1"` |
| `detection/stages.py` | stage attribution (Hungarian, IoU ≥ 0.5), visibility classes, size bins, AP50 |
| `detection/diagnose.py` | writes `diag-v1/{summary.json, gt_rows.csv.gz, engine_xref.csv}` |
| `detection/replay.py` | tracker-input variants (`InputFilter`, `VARIANTS`: B0, T0, TG6, C3, T0C3, T0P, T0C3P, TG6C3), ByteTrack replay, B0 baseline check, TrackEval scoring → `replay-v1/<variant>/` |
| `detection/merge.py` | merge policies A0–A3 for the NMS A/B (unit-tested; never run on real data) |
| `detection/capture.py` | GPU raw per-tile capture; refuses to run without `--approved`; **never run** |
| `detection/visual.py` | comparison panels → `diag-v1/examples/*.png` |
| `detection/framestore.py` | per-frame random access to cache, replay variants, GT and engine predictions (for the dashboard) |
| `detection/report.py` | markdown tables of the diagnosis |
| `detection/__main__.py` | CLI: `diagnose`, `examples`, `replay`, `report`, `ab`, `capture` |

**Dependencies on Session A code** (imported read-only from the Session B worktree's own copy of
`football_fatigue.vision`, which is the same commit):
* `vision.detection_cache.DetectionCacheReader`;
* `vision.tracker.MultiObjectTracker`;
* `vision.mot_eval` (`load_reference`, `build_sequence_data`, `standard_metrics`; TrackEval);
* `vision.detector` (`YOLOPlayerDetector`, `_PERSON_CLASSES`; used only by `merge.py` and `capture.py`);
* `vision.models`;
* `multicam_lab.media.probe_video`.

(VERIFIED by grep)

**Environment:**
* Python is `F:\Football-Fatigue-Project\.venv\Scripts\python.exe` (execute-only; no package installs).
* Set `PYTHONPATH=F:\Football-Fatigue-Video\src` so that Session B's code is imported.
* The data root is `F:\Football-Fatigue-Project\data` (read-only).

### 3.3 Stored outputs

All VERIFIED (directory listing).

```
data/processed/video/
  soccertrack-<id>/occlusion/0.1.0/occl-soccertrack-117093-3000-3499-da3e0b5f/        (Step 1 window, 117093 only)
  soccertrack-<id>/occlusion/0.1.0/INVALID-render-pts-offset_occl-…/                   (kept, marked invalid)
  soccertrack-<id>/occlusion/0.2.0/occl-soccertrack-<id>-0-<last>-step1-ac611834/      (full match, profile step1)
  soccertrack-<id>/occlusion/0.2.0/occl-soccertrack-<id>-0-<last>-v2-ff8815ed/         (full match, profile v2)
  soccertrack-117093/occlusion/0.2.0/occl-soccertrack-117093-3000-3499-v2-ff8815ed/    (window with annotated.mp4)
  soccertrack-<id>/occlusion/evaluation/<run_id>.eval-v1.json
  soccertrack-<id>/detection/diag-v1/{summary.json, gt_rows.csv.gz, engine_xref.csv, examples/*.png, examples/index.json}
  soccertrack-<id>/detection/replay-v1/{B0,T0,TG6,C3,T0C3,T0P,T0C3P,TG6C3}/{tracks.jsonl.gz, result.json}
  _detection/turf_rejected_non_gt_high_conf.png, _detection/turf_nongt.py
```

There is no `reviews/` folder, so no human event reviews have been saved. (VERIFIED)

### 3.4 Separate concepts (do not conflate)

| concept | definition in this pipeline | where it is measured |
|---|---|---|
| **Detection miss (model)** | no box (conf > 0.1) near a visible GT player after merging | `diag-v1` stage `no_detection` |
| **Post-processing loss** | the model produced a box, but the tracker input dropped it: turf filter (`turf_filtered`) or conf ≤ 0.4 (`below_threshold`) | `diag-v1` |
| **Possible merge (NMS) loss** | no box, but a surviving neighbour overlaps the player (upper bound only) | `diag-v1` stage `merge_suspect` |
| **Tracker loss** | a detection went to the tracker, but there was no tracker output box | `detected_untracked` |
| **Tracker prediction** | an engine-generated box for a lost or hidden track (dashed in the UI); never an observation | `frames.jsonl` states `hidden_predicted` / `lost_predicted` |
| **Track fragmentation** | one GT player covered by several successive track IDs (CLEAR `Frag`; number of track IDs) | `replay-v1/*/result.json` |
| **Identity switch** | a GT player's matched track ID changes (CLEAR `IDSW`). The engine's `tracked_other_id` cases are ID switches, not occlusions. | `result.json`, `diag-v1` engine cross-reference |
| **True visual occlusion** | proxied by GT geometry only: ≥ 50% of a GT box covered by nearer GT boxes for ≥ 3 frames. There is no pixel-level occlusion label. | `occlusion/evaluate.py` |

## 4. Dataset and protected-split policy

* **Development sources** (`sources.py:31-53`, VERIFIED). All are SoccerTrack v2 panorama matches. The GT is MOT boxes
  for the 22 players only; there are no referees and the boxes are template-sized.

  | source | role | cache run | replay run |
  |---|---|---|---|
  | soccertrack-117093 | tune | `soccertrack-117093-m6-d1fixed-full-merged-20261025` | `…-full-stable-replay` |
  | soccertrack-118575 | tune | `soccertrack-118575-m62-d1fixed-full-merged-20261026` | `…-full-stable-replay` |
  | soccertrack-117092 | check | `soccertrack-117092-m6-d1ea-full-merged-20261025` | `…-full-stable-replay` |
  | soccertrack-118576 | check | `soccertrack-118576-m62-d1fixed-full-merged-20261026` | `…-full-stable-replay` |

* **Reserved, never opened by Session B:** `RESERVED_MATCH_IDS = {"118577", "118578", "128058", "128057", "132831",
  "132877"}`. `guard_not_reserved` is called by every loader, the CLI, the replay, the framestore and the capture.
  (VERIFIED in code)
* **The split is development-internal.** "Check" matches had no thresholds chosen on them, but they are still
  development data. **No Session B number is an independent validation result.** (DOCUMENTED)
* `data/raw` and Session A's data are read, never written. (DOCUMENTED; consistent with the output inventory)

## 5. Historical video milestones

| date | milestone | outcome | label |
|---|---|---|---|
| 2026-11-17 | Engine decision report (`VIDEO_ENGINE_DECISION_REPORT_v1.md`) | YOLO26-L weights verified. Merge NMS was identified as hiding occlusions. A V1 occlusion engine was proposed. | DOCUMENTED |
| 2026-11-17 | Step 1: occlusion engine 0.1.0 demo on 117093 frames 3000–3499 | Event classes: box overlap, probable occlusion, detector miss, tracker gap, uncertain. Annotated MP4. A first render had a PTS offset bug and is kept as `INVALID-render-pts-offset_…`. | DOCUMENTED, artifacts VERIFIED |
| 2026-11-17 | Step 2: engine 0.2.0 (profile `v2`), `eval-v1`, Streamlit dashboard on :5830 | v2 improved timing and readability, **not precision** (§9). | DOCUMENTED |
| 2026-11-18 01:48–02:14 | Detection diagnosis `diag-v1` + tracker-input replay `replay-v1` + dashboard detection tab | The turf filter is the main loss; T0 recommended and **not applied**; GPU work awaits approval. | DOCUMENTED, artifacts VERIFIED |
| (not delivered) | "APPROVED: APPLY THE VERIFIED DETECTION FIX AND VALIDATE THE VIDEO PIPELINE" | **not executed**; see §8 | VERIFIED absent |

## 6. Detection-quality investigation and root causes

Source: `docs/research/VIDEO_DETECTION_DIAGNOSIS_v1.md` and `data/processed/video/<source>/detection/diag-v1/summary.json`.

### 6.1 What was examined

* **Matches and frames.** All frames of the four development matches were examined: 5,976 / 6,000 / 6,000 / 5,975
  frames for 117093 / 118575 / 117092 / 118576. These are the B0 baseline-check frame counts in `result.json`.
  (VERIFIED)
* **Inputs.** The examined boxes are only the post-merge cache boxes at conf > 0.1. **Raw per-tile YOLO output does not
  exist on disk.** The caches never stored pre-merge boxes, so "raw YOLO" versus "merged" could not be compared. The
  comparison panels show an explicit "raw per-tile: not captured" panel. (VERIFIED in code `visual.compose`;
  DOCUMENTED)
* **The three detection levels:**
  * *raw YOLO* is unavailable;
  * *merged detections* are the cache (conf > 0.1, all classes);
  * *tracker inputs* are the merged person boxes with conf > 0.4 that passed the turf filter (B0).
* **Curated windows.** The comparison panels show the longest loss runs per stage, 12 PNGs per match, in
  `diag-v1/examples/` with `index.json`. (VERIFIED present)

### 6.2 Where visible GT players are lost before the tracker

Visible GT boxes are those less than 10% covered by nearer GT boxes. (DOCUMENTED)

| match | visible GT boxes | lost | turf filter | no detection (model) | below 0.4 | merge suspect | tracker | poor overlap |
|---|---|---|---|---|---|---|---|---|
| 117093 | 113,866 | 8.3% | **79%** | 5% | 8% | 5% | 2% | 2% |
| 118575 | 118,986 | 14.6% | **69%** | 8% | 13% | 3% | 4% | 2% |
| 117092 | 116,396 | 20.3% | 4% | **50%** | 31% | 2% | 8% | 6% |
| 118576 | 109,678 | 6.9% | **69%** | 4% | 8% | 13% | 3% | 3% |

* **The model's coverage.** It puts a box (conf > 0.1) on **98.0–99.1%** of visible players on 117093, 118575 and
  118576, and on 88.4% on 117092. (DOCUMENTED)
* **Engine "detector miss" events** (v2 runs) by their real cause:

  | match | turf-filtered GT player | model miss | not a GT player |
  |---|---|---|---|
  | 117093 | 72% | 0% | 23–27% |
  | 118575 | 74% | 3% | 23–27% |
  | 117092 | 22% | 42% | 33% |
  | 118576 | 74% | 1% | 23–27% |

  (DOCUMENTED)

### 6.3 Why the turf filter removes genuine players

* **The rule.** Session A's `_looks_like_player` rejects a box when more than 72% of its upper 65% is green (OpenCV
  hue 35–85, saturation ≥ 40). (DOCUMENTED)
* **Who it removes:**
  * lime or yellow-green kits fall inside that hue band;
  * thin, small players whose narrow box is mostly grass exceed 72%.
* **The removed boxes are confident.** Median conf 0.87; 10th percentile 0.62–0.84. (DOCUMENTED)
* **The goalkeeper example.** `soccertrack-117093/detection/diag-v1/examples/turf_filtered-f01585-gt6.png`: a lime-kit
  goalkeeper, conf 0.78, removed for **231 consecutive frames**. (File VERIFIED present; numbers DOCUMENTED)
* **The beige-kit example.** `soccertrack-118575/detection/diag-v1/examples/turf_filtered-f03600-gt19.png`: a thin
  player in a beige kit, conf 0.84, removed for 183 frames. (File VERIFIED present; numbers DOCUMENTED)
* **The same failure on the check match:** `soccertrack-118576/…/turf_filtered-f02790-gt14.png` and `…f03908-gt7.png`.
  (DOCUMENTED)
* **Non-GT boxes the filter removes are referees.** All 18 sampled high-confidence on-pitch non-GT turf-rejected boxes
  were referees in dark kits; none was grass. Montage: `data/processed/video/_detection/turf_rejected_non_gt_high_conf.png`.
  (DOCUMENTED; file VERIFIED present)

### 6.4 Genuine YOLO limitations (117092)

* **The footage.** It is dim, low-contrast and heavily compressed. 50% of visible losses are `no_detection`.
* **Small players are worst.** Tracker-input recall of visible players under 35 px is **0.593**, against 0.83–0.91 on
  the other matches.
* **Examples:** `soccertrack-117092/…/no_detection-f04382-gt16.png` and `…f04772-gt14.png`: visible players with no box
  for about 100 frames.
* **Other true misses:** a crouching 17 px player (117093, `no_detection-f01130-gt4.png`) and a goalkeeper lying down
  (118575, `no_detection-f04397-gt20.png`).

(DOCUMENTED)

### 6.5 Other diagnostics

AP50 and duplicates (DOCUMENTED):

| match | AP50 (COCO-style, truncated at 0.1) | duplicates per 100 TP |
|---|---|---|
| 117093 | 0.871 | 0.16–0.76 (range across matches) |
| 118575 | 0.823 | |
| 117092 | 0.837 | |
| 118576 | 0.902 | |

* **Confidence threshold.** Going from 0.4 to 0.3 adds 0.5–2.2 recall points and 30–150% more off-pitch boxes.
  (DOCUMENTED)
* **Merge NMS.** `merge_suspect` covers at most 2–13% of visible losses, and is mostly heavy or hidden players. This is
  an upper bound, not a measurement. (DOCUMENTED)

## 7. B0/T0 experimental evidence

### 7.1 Protocol

Code: `detection/replay.py`. (VERIFIED in code, DOCUMENTED in the report)

* **Identical inputs.** The same cached post-merge detections and the same Session A `MultiObjectTracker` parameters
  are used for every variant. Only the tracker-input filter changes:
  * `B0`: production; person, conf > 0.4, turf filter applied;
  * `T0`: person, conf > 0.4, turf flag ignored.
* **Baseline check.** B0 was compared box-for-box with the production replay `frames.jsonl`. The saved `result.json`
  files record `frames_differing = 0` on all four matches (5,976 / 6,000 / 6,000 / 5,975 frames). (VERIFIED in saved
  results)
* **Scoring.** TrackEval HOTA / CLEAR / Identity against the full-match GT (22 players), for all tracks and for on-pitch
  tracks only.

### 7.2 Results

Read from `data/processed/video/<source>/detection/replay-v1/{B0,T0}/result.json` on 2026-11-18. (VERIFIED)

| match | role | variant | HOTA | DetRe | DetPr | AssA | IDF1 | IDSW | Frag | track IDs\* | HOTA on-pitch | DetPr on-pitch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 117093 | tune | B0 | 0.371 | 0.763 | 0.753 | 0.210 | 0.377 | 373 | 2,692 | 262 | 0.375 | 0.772 |
| 117093 | tune | **T0** | **0.423** | 0.815 | 0.739 | 0.264 | 0.435 | 348 | 921 | 210 | 0.428 | 0.757 |
| 118575 | tune | B0 | 0.371 | 0.698 | 0.750 | 0.223 | 0.345 | 383 | 3,256 | 253 | 0.371 | 0.753 |
| 118575 | tune | **T0** | **0.409** | 0.783 | 0.748 | 0.247 | 0.391 | 266 | 1,452 | 154 | 0.410 | 0.750 |
| 117092 | check | B0 | 0.305 | 0.653 | 0.820 | 0.153 | 0.280 | 509 | 4,496 | 379 | 0.307 | 0.829 |
| 117092 | check | **T0** | 0.307 | 0.660 | 0.816 | 0.154 | 0.281 | 509 | 4,149 | 370 | 0.308 | 0.825 |
| 118576 | check | B0 | 0.384 | 0.773 | 0.781 | 0.215 | 0.373 | 387 | 2,399 | 182 | 0.385 | 0.783 |
| 118576 | check | **T0** | **0.419** | 0.808 | 0.758 | 0.253 | 0.445 | 350 | 996 | 133 | 0.420 | 0.760 |

\* Track-ID counts are DOCUMENTED (diagnosis report §5). The other columns are VERIFIED. 118576 T0 HOTA is 0.4195; the
report rounds it to 0.420.

**Reading:**
* **Gain.** On the three matches whose losses are mostly the turf filter, T0 raises HOTA by +0.035 to +0.053 and IDF1
  by +0.046 to +0.072. ID switches fall by 25–117.
* **Fragmentation.** CLEAR Frag falls 55–66% and track-ID counts fall 21–39%: tracks no longer break each time a player
  enters and leaves the filter.
* **117092 is essentially unchanged.** Its losses are model misses, not post-processing.

### 7.3 The cost of tracking more referees

* **Detection precision drops.** DetPr falls by 0.002–0.023 under T0 (on-pitch: 0.003–0.023), because referees and
  linesmen that the turf filter used to remove are now tracked. SoccerTrack GT has no referees, so every referee box is
  scored as a false positive. Part of this "precision loss" is therefore people detected correctly. (VERIFIED numbers;
  interpretation DOCUMENTED)
* **Consequences downstream.** More referees in the tracker input mean more `non_gt_person` occlusion events and more
  occlusion pairs that involve referees. Under T0 this has **not been measured**, because no T0 occlusion run exists.
  (UNVERIFIED)

### 7.4 Other variants

All DOCUMENTED; results are in `replay-v1/`.

| variant | change | effect |
|---|---|---|
| `TG6` | turf filter only for conf ≤ 0.6 | the same as T0 to within ±0.001 HOTA |
| `C3` | conf > 0.3 | at most +0.006 HOTA |
| `T0P` / `T0C3P` | play-area filter | at most +0.005 HOTA; needs annotated keypoints, so it does not transfer to uploads |

### 7.5 What is reproducible and what is checked

| result | status |
|---|---|
| `diag-v1` and `replay-v1` artifacts | Present, write-once, regenerable with the CLI commands in `docs/video_detection_diagnostics.md`. The replay takes about 2–3 minutes per variant per match on CPU. (VERIFIED present; reproducibility DOCUMENTED, not re-run now) |
| B0 = production replay | Recorded inside the saved results. (VERIFIED in saved files) |
| Visual causes of turf rejection (lime kit, beige kit, referees) | From inspection of the example PNGs by the agent. Not independently checked by a human. (DOCUMENTED, not independently verified) |
| Stage attribution | Automated with fixed IoU thresholds. `poor_overlap` mixes annotation offsets and bad boxes. (DOCUMENTED limitation) |
| Independent validation of T0 | **None.** All four matches are development data. (VERIFIED by split policy) |

## 8. Exact post-approval T0 implementation status

The questions from the verification request, answered one by one:

| # | question | answer | label |
|---|---|---|---|
| 1 | Was T0 implemented after approval? | **No.** The approval was never delivered to Session B, and no file changed after 02:14. | VERIFIED |
| 2 | Where is T0 selected or configured? Is B0 preserved? | Only as the diagnostic replay variant `InputFilter("T0", turf="ignore", note="turf filter off, conf > 0.4")` (`src/football_fatigue/video/detection/replay.py:40`). B0 is preserved: production is untouched, `replay.py:39` reproduces it and it verifies equal. There is no production or engine-level T0 setting. | VERIFIED |
| 3 | Were B0/T0 tracking outputs generated and versioned? | Yes, as **diagnostic** replays: `replay-v1/B0/` and `replay-v1/T0/` (`tracks.jsonl.gz`, `result.json` with `replay_version`) for all four matches, 2026-11-18 01:58–02:10. They were made by the diagnosis milestone, not by a post-approval rebuild. They are not in the Session A `frames.jsonl` format the engine reads. | VERIFIED |
| 4 | Are before/after evaluation metrics saved? | Tracking metrics: yes (§7.2). Occlusion-event metrics under T0: **no**. | VERIFIED |
| 5 | Was the occlusion engine run under both configurations? | **No.** Only B0-based `step1` and `v2` runs exist (2026-11-17). The engine CLI has no T0 input path. | VERIFIED |
| 6 | What happened to false probable occlusions, detection-loss events, tracker gaps and ID switches? | Under T0, ID switches fall (§7.2). For **events**, nothing has been measured. The expected effect: most turf-filtered "detector miss" events disappear (they are 72–74% of those events on 3 matches) and `tracked_other_id` false occlusions fall. This is a prediction, not a result. | ID switches VERIFIED; event effects UNVERIFIED |
| 7 | Does the dashboard support B0 vs T0, zoomed inspection and event comparison? | **Partly.** The 🔬 Detection diagnostics tab has a "Tracker output shown" selector over all replay variants, zoomed stage panels, and a B0/T0 metrics table. The Viewer, Events and Results tabs show only the B0-based engine runs. There is **no side-by-side B0/T0 event comparison**. | VERIFIED in code |
| 8 | Were the goalkeeper, beige-kit and 117092 examples made inspectable? | Yes: as PNGs in `diag-v1/examples/` and through the tab's "Curated example" picker, which reads the same `index.json`. This was done in the diagnosis milestone. | VERIFIED |
| 9 | What tests and browser validation were completed after the change? | **None after the change, because there was no change.** The last test run was at about 02:15 on 2026-11-18: 62 passed (`test_video_detection.py`, `test_video_step2.py`, `test_video_occlusion.py`); the pytest cache shows `lastfailed = {}`. The detection tab was checked only by a headless Streamlit `AppTest` smoke test. No browser or screenshot validation of the detection tab is recorded. | Test run HISTORICAL (the pytest cache is consistent); browser UNVERIFIED |
| 10 | Were any GPU NMS or low-light experiments executed? | **No.** `capture.py` was never run: no `captures/` folder exists. `merge-ab-v1/` does not exist. There is no low-light pilot output and no download. | VERIFIED |
| 11 | Unfinished changes, failed tests, missing artifacts, uncommitted files? | All Session B work is uncommitted (§2). There are no half-applied edits (no files newer than the milestone) and no recorded test failures. Missing artifacts: T0 occlusion runs, T0 occlusion evaluations, the raw per-tile capture and the merge A/B. | VERIFIED |

## 9. Occlusion-engine state and before/after evidence

* **Engine.** Version 0.2.0. Profiles `step1` (config `ac611834`) and `v2` (config `ff8815ed`, the default). The event
  schema is 1.1.0. (VERIFIED in code and run folder names)
* **Runs present.** Full matches for all four sources × {step1, v2}, plus the 117093 v2 window 3000–3499 with
  `annotated.mp4`. All are B0-based. (VERIFIED)
* **step1 → v2 evaluation** (`eval-v1`; full table in `VIDEO_V1_STEP2_EVALUATION.md`) (DOCUMENTED):
  * **Claim-level occlusion precision is unchanged on the tune matches** (0.666 → 0.663, 0.677 → 0.679) and slightly
    lower on the check matches (0.571 → 0.537, 0.708 → 0.687).
  * **Timing improved:** the median |end error| fell from 11–15.5 frames to 5–8 frames.
  * **Box-overlap recall improved:** 0.14–0.31 → 0.34–0.48.
  * **Detection-loss events are not occlusions:** 98.6–98.9% of scored detection-loss events are not occlusions in GT.
  * **The remaining false positives** are `gt_overlap_only` (partly the template-box proxy) and `duplicate_track`.
* **B0 → T0 for the occlusion engine:** **no evidence**; it was never run. (UNVERIFIED)
* **Caveats:**
  * The GT is a geometric proxy (template boxes, no referees).
  * The evidence level is a grade, not a calibrated probability.
  * `T##` IDs are temporary.

## 10. Dashboard capabilities and launch command

**Launch** (DOCUMENTED; it was used historically on 2026-11-17 and 2026-11-18; the server is stopped now):

```powershell
cd F:\Football-Fatigue-Video
$env:PYTHONPATH = "F:\Football-Fatigue-Video\src"
F:\Football-Fatigue-Project\.venv\Scripts\python.exe -m football_fatigue.video.dashboard --data-root F:\Football-Fatigue-Project\data
# http://127.0.0.1:5830   (--port, --headless available)
```

**Viewer / Events / Results tabs** (Step 2) (DOCUMENTED):
* **Viewer:**
  * a Match/Run picker over the engine runs;
  * a frame viewer with full frame, follow-event and manual zoom;
  * a clickable timeline;
  * an event inspector with evidence, a GT comparison and snapshots;
  * a human review form, saved to `data/processed/video/reviews/<source>/reviews.jsonl`. It is unused: no reviews
    exist.
* **Events:** a filtered table with CSV export.
* **Results:** scores, success and failure examples, missed GT episodes, and the step1 vs v2 table.
* **Playback** is stepped scrubbing at up to about 6 steps per second.

**🔬 Detection diagnostics tab** (`dashboard/detection_tab.py`; VERIFIED in code):
* **Selectors:** the match, the tracker output shown (`B0`, `T0`, `TG6`, … from `replay-v1`), and the crop zoom.
* **Navigation:**
  * jump filters by stage and visibility, with prev/next lost and ±1/±25 buttons;
  * a curated-example picker;
  * a focus-player picker.
* **Comparison strip:** GT | raw per-tile ("not captured") | post-merge cache (colour-coded: green = tracker input,
  orange = below threshold, magenta = turf filter) | tracker output of the chosen variant | engine predicted.
* **Verdict line, frame view and tables:**
  * a verdict line that names the responsible stage;
  * a whole-frame overview coloured by stage;
  * the per-frame GT table, the match-level stage × visibility table, and the replay A/B metrics table.
* **Limitation.** "Engine predicted" always comes from the B0-based v2 run. There is no T0 engine run to show.

## 11. Existing tests and validation evidence

| test file | scope | status |
|---|---|---|
| `tests/test_video_occlusion.py` | Step 1 engine | part of the 62-pass run (HISTORICAL) |
| `tests/test_video_step2.py` | v2 engine, evaluation, dashboard data/state | part of the 62-pass run (HISTORICAL) |
| `tests/test_video_detection.py` | 15 tests (see below) | part of the 62-pass run (HISTORICAL) |

The 15 detection tests cover:
* stage cascade, merge_suspect and poor_overlap, cover and visibility;
* the threshold sweep, precision and AP;
* replay filters;
* merge policies (A0 equals production);
* the capture refusing to run without approval;
* `LineIndex`, the stage table and the report;
* on real data:
  * B0 baseline identical;
  * diagnose window agreeing with `FrameStore`;
  * a dashboard-tab `AppTest` smoke test.

* **Last run.** 2026-11-18, about 02:15, with real data: **62 passed**. `.pytest_cache/v/cache/lastfailed` is `{}`
  (written 02:02), and `nodeids` was updated at 02:15. Both are consistent with a passing run. Not re-run for this
  handoff. (HISTORICAL)
* **The full repository suite was not re-run in the diagnosis milestone.** Earlier it had 2 environmental Session A
  failures. (HISTORICAL)
* **Browser validation:** none recorded for the detection tab. (UNVERIFIED)

## 12. Remaining problems and unsupported claims

1. **T0 is not the production tracker input.** The recommended fix exists only as a diagnostic replay.
2. **No occlusion-event evidence under T0.** Any claim that T0 reduces false "detector miss" or false probable-occlusion
   events is a prediction. (UNVERIFIED)
3. **Raw YOLO output was never observed.** The model-versus-NMS split for missing boxes is an upper bound.
4. **Referees.** T0 tracks more referees. GT cannot score them, and the engine flags but does not suppress them.
5. **117092 low-light misses are unsolved** by any post-processing change.
6. **Duplicate tracks** create false occlusion pairs (DOCUMENTED, unhandled).
7. **Scientific boundaries:**
   * **Detection versus tracking.** HOTA and IDF1 are tracking metrics. T0 improves them by feeding the tracker boxes
     the model already produced; the detector itself did not change.
   * **Occlusion versus loss.** Engine "detector miss" and "tracker gap" are losses, not occlusions. Occlusion truth is a
     GT-geometry proxy, not a visual label.
   * **Track IDs versus identities.** Higher HOTA and IDF1 and fewer ID switches **do not** mean persistent or global
     player identities. `T##` IDs remain temporary, and the remaining IDF1 is 0.28–0.45.
   * **Development versus validation.** All numbers come from four development matches; nothing is validated
     independently.
   * **Pixels versus metres.** All results are in image pixels. Nothing in Session B produces validated pitch-metre
     coordinates.
   * **Historical versus current.** `step1` and `v2` engine results and B0 are the current pipeline. T0 is an
     experimental result.
8. **Everything is uncommitted.** A lost worktree would lose all Session B code and docs.

## 13. GPU pilots proposed but not authorised

All PLANNED and not executed (VERIFIED: no outputs).

| # | pilot | cost estimate | purpose |
|---|---|---|---|
| 1 | Raw per-tile capture (`python -m football_fatigue.video.detection capture --approved`): 4 matches × 2 windows × 125 frames | about 30 min GPU (GTX 1650 Ti, 1.77 s/frame), under 3 GB VRAM, about 50 MB disk | the merge-policy A/B (A0–A3) with `ab --capture <dir>` on CPU |
| 2 | 117092 low-light test-time scaling: same weights, 640 px tiles at imgsz 1280, 2 × 125 frames | about 30 min GPU, no download | check whether upscaling recovers small or dark players |
| 3 | Optional RF-DETR-M zero-shot sanity check (Apache-2.0) | about a 130 MB download plus about 20 min GPU | only if 1–2 leave a large gap; zero-shot is not a fair comparison |

## 14. Prioritised next steps

1. **The user commits the current Session B work.**
2. **Re-issue the T0 approval to Session B** (it was never received). Then:
   * add a selectable tracker-input profile (B0 default-preserved, T0 recommended);
   * write T0 replays in the engine's input format under `data/processed/video/` (never into Session A's
     `processed/vision`);
   * add an engine `--tracks`/profile option;
   * run `v2` on B0 and T0 for the four development matches and evaluate both;
   * add a B0/T0 event comparison to the dashboard;
   * test, and validate in a browser.
3. **Decide on GPU pilot 1** (NMS capture), then pilot 2 (117092 low light).
4. **Handle referees** in the engine: flag or suppress `non_gt_person` events under T0.
5. **Independent validation** belongs to a later, explicitly authorised milestone. The reserved sequences stay closed
   until then.

## 15. Self-contained continuation brief for ChatGPT

> **Context.** Football Fatigue project, Session B (video), worktree `F:\Football-Fatigue-Video`, branch
> `feature/video-occlusion-events`, HEAD `1bd016d` (no Session B commits; all Session B work untracked). Python
> `F:\Football-Fatigue-Project\.venv\Scripts\python.exe` with `PYTHONPATH=F:\Football-Fatigue-Video\src`. Data root
> `F:\Football-Fatigue-Project\data` (read-only). Outputs only under `F:\Football-Fatigue-Video\data\processed\video\`.
> Never open reserved sequences 118577, 118578, 128058, 128057, 132831, 132877. No GPU inference, downloads or training
> without explicit user approval. Git is read-only for the agent.
>
> **Pipeline.** Production YOLO26-L tiled detections (Session A cache, post-merge only, conf > 0.1, turf flag) →
> tracker input (person, conf > 0.4, turf filter) → Session A ByteTrack replay → Session B occlusion engine 0.2.0
> (profiles step1 / v2) → `eval-v1` → Streamlit dashboard on :5830 (Viewer / Events / Results / 🔬 Detection
> diagnostics). Temporary track IDs only. Pixel space only.
>
> **Key finding (2026-11-18, development data).** Most engine "detector misses" are confident YOLO boxes (median conf
> 0.87) removed by Session A's turf (grass-colour) filter: lime goalkeeper kit, beige kit and thin players. Replaying
> identical detections with the turf filter off (variant T0, conf > 0.4) changes HOTA:
>
> | match | HOTA B0 → T0 |
> |---|---|
> | 117093 | 0.371 → 0.423 |
> | 118575 | 0.371 → 0.409 |
> | 118576 | 0.384 → 0.420 |
> | 117092 | 0.305 → 0.307 (low-light model misses) |
>
> IDF1 rises by 0.046–0.072, ID switches fall, fragmentation falls by 55–66%, and DetPr falls by 0.002–0.023 because
> referees are now tracked. B0 reproduces production exactly.
>
> **Status.** T0 exists only as a diagnostic replay (`src/football_fatigue/video/detection/replay.py`, outputs
> `replay-v1/`). The approved milestone "APPLY THE VERIFIED DETECTION FIX AND VALIDATE THE VIDEO PIPELINE" was **never
> delivered to or executed by** Session B:
> * no T0 tracker-input profile;
> * no T0 occlusion-engine runs or evaluations;
> * no B0/T0 event comparison in the dashboard;
> * no post-change tests.
>
> GPU pilots (NMS raw capture about 30 min; 117092 low-light about 30 min; optional RF-DETR-M) are proposed and not
> authorised. The last test run (2026-11-18) had 62 video tests passing.
>
> **Next action.** Re-send the T0 approval to Session B, specifying:
> * B0 stays the preserved default;
> * T0 becomes a selectable recommended profile;
> * the engine is run under both and evaluated;
> * the dashboard gets a B0/T0 event comparison;
> * tests plus browser validation.
>
> Do not interpret higher HOTA as persistent identity; all results are development-only and pixel-space.
