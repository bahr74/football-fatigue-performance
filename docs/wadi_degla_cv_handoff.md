# Wadi Degla CV Handoff

## Scope

This document records the work present in the `wadi-degla` worktree compared
with the earlier `main` baseline. It is a handoff, not a claim that the
experimental CV outputs are scientifically validated.

## Git and data handling

- Current branch observed: `wadi-degla`.
- Changes are present in the worktree and have not been staged, committed, or
  pushed by the coding agent.
- `data/external/`, `data/raw/`, model checkpoints, caches, and generated CV
  runs must remain local and ignored. They must not be added to a commit or
  pushed. The local SoccerTrack sample is gated external data; a future agent
  must verify its current access and redistribution terms before sharing it.
- The user should review `git status`, select only source/docs/tests/config
  files, then perform the manual commit and push to the Wadi Degla remote.

## What changed from the earlier main baseline

### Existing platform work now present on this branch

- Streaming ingestion, typed models, validation, quality summaries, event
  synchronization, and the unified `MatchData` interface.
- Tactical board playback with match selection, event navigation, movement
  summaries, spatial summaries, heatmaps, team-shape views, and CSV export.
- Movement analytics with distance, speed, acceleration, velocity zones, gap
  handling, spike rejection, coverage, and estimated-observation reporting.
- Spatial analytics with opt-in median smoothing, observation heatmaps, team
  shape, and adjacent-window positional changes.

### CV module

- Added `src/football_fatigue/vision/` with calibration, YOLO detection,
  ByteTrack/IoU tracking, team-color classification, data models, and the
  frame/video pipeline.
- Added the CV CLI at `python -m football_fatigue.vision`.
- Added JSONL frame output and provenance manifests under
  `data/processed/vision/`.
- Added `--max-seconds`, so a one-minute run can be made with
  `--max-seconds 60`; omitting both duration limits processes to EOF.
- Added confidence to saved player records and conservative filtering of
  boxes whose proposed upper body is almost entirely field turf.
- Pinned `supervision` to `>=0.25,<0.31`. Version `0.31` removed the
  `ByteTrack` API used by this project; version `0.30.9` exposes it.
- Added `accessible-space>=2.1,<3` as the optional `tactical` dependency.
- Added periodic JSONL flushing so long runs can be inspected while active.

### CV analytics and dashboard

- Added `src/football_fatigue/analytics/cv.py`.
- The real Stage 4-facing name is now `movement_analytics`, not
  `stage_4_movement`.
- `analytics.json` reports movement metrics when usable calibrated positions
  exist, while preserving missing and estimated values.
- Added dashboard cards for Movement Analytics, Pitch Control, Expected
  Possession Value, Off-Ball Scoring Opportunity, and Dangerous Accessible
  Space.
- The dashboard prefers the newest saved or actively-processing CV run and
  displays completed detector frames while a run is still running.
- Processing state is shown separately from genuinely unavailable analytics.
- The tactical board itself was left unchanged; CV presentation is in
  `src/football_fatigue/board/cv.html` and the CV endpoints in the playback
  server.

## Verified local runs

- The one-minute run completed successfully:

  ```text
  run: soccertrack-117093-first-minute-20261019
  frames: 1500 at 25 fps
  tracker: bytetrack
  confidence: 0.40
  image size: 640
  status: processed_unvalidated
  ```

- The run is detection-only. It contains image-space boxes and track IDs, but
  no validated pitch positions, stable team labels, grounded ball state, or
  possession state. Its `analytics.json` therefore correctly reports
  `movement_analytics` and all four tactical metrics as unavailable.
- The older 30-frame smoke output used IoU tracking and a lower confidence
  threshold. It should not be treated as the final detector result.
- The full-video CPU run was intentionally stopped because it was too slow for
  the environment. The duration-limited CLI now makes shorter reproducible
  runs practical.

## Validation run

The latest focused validation completed with:

```text
35 passed
```

Earlier full-suite validation completed with:

```text
262 passed, 6 skipped
```

Run the full suite again after any further changes:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Why tactical cards are currently unavailable

The current video is a panoramic SoccerTrack clip. Detection-only mode does
not create metre coordinates. The supplied raw homography was inspected and
does not provide sufficiently accurate raw-image-to-pitch coordinates for
movement or tactical claims; earlier held-out calibration diagnostics showed
metre-scale error that is too large for trustworthy analytics.

The required inputs are:

| Metric | Required inputs still missing |
| --- | --- |
| Movement analytics | Validated rectified-video pitch calibration in metres |
| PC | Calibrated player positions, team labels, velocities, ball position, and pitch clock |
| EPV | Possession/event labels, declared reward target, training data, held-out evaluation |
| OBSO | Ball/possession state, player identity, team, pitch direction, and scoring/control model |
| DAS | Calibrated tracking, team/possession state, attacking direction, velocity fields, pass assumptions, and validation |

Installing a package does not create those inputs. `accessible-space` is
installed, but publishing a DAS number before those inputs are audited would
be misleading.

## Detailed next-agent plan

### 1. Reproduce and audit the current state

1. Read `PROJECT_RULES.md`, `AGENTS.md`, and this document.
2. Run `git status --short`; preserve unrelated work.
3. Confirm that `data/external/` and generated outputs are ignored and are not
   staged.
4. Run the focused vision tests and then the full test suite.
5. Inspect the completed one-minute manifest and `analytics.json` before
   changing the pipeline.

### 2. Integrate the SoccerTrack camera correction

1. Inspect the actual shapes, dtypes, coordinate conventions, and image sizes
   of `117093_camera_intrinsics.npz`, `117093_mapx.npy`, `117093_mapy.npy`,
   `117093_homography.npy`, and `117093_keypoints.json`.
2. Add a reusable preprocessing component that applies the supplied remap
   arrays to decoded frames without modifying `data/external`.
3. Keep the original pixel coordinates and transformed coordinates separately
   in output provenance.
4. Verify whether the homography is forward or inverse using the actual
   keypoint pairs. Do not assume the direction from the filename.
5. Fit or adapt the transform only after confirming pitch origin, axis
   direction, units, and resolution.
6. Add independent held-out landmarks and reject the calibration if the
   agreed metre error threshold is exceeded.
7. Add tests using a small synthetic remap/calibration fixture; do not copy
   external data into tests.

### 3. Improve detection and association quality

1. Evaluate the detector against `mot/117093.txt` using a documented
   frame-level protocol: class mapping, IoU threshold, confidence threshold,
   precision, recall, and exclusions.
2. Compare `imgsz` values and tiled inference on the 4096px panoramic frame.
   Tiled inference may recover small players better than resizing the whole
   panorama to 640px.
3. Preserve the full-frame coordinate transform when stitching tile boxes.
4. Prefer the compatible ByteTrack release already pinned, and record the
   selected tracker and version in every manifest.
5. Tune false-positive rejection from measured validation data, not from one
   screenshot. Keep ball, referee, goalkeeper, and player thresholds
   configurable.
6. Add an evaluation report separate from prediction output. Never use MOT
   reference boxes as detector predictions.

### 4. Establish team and ball state

1. Collect explicit jersey-color samples for both teams and record which
   cluster is home/away; arbitrary cluster IDs are not team identities.
2. Add temporal confidence and an unknown/quarantine state for ambiguous
   team assignments.
3. Improve ball detection/tracking separately from player tracking. A planar
   homography must not mark airborne ball coordinates as grounded.
4. Define possession as an auditable heuristic or event-backed label, with
   confidence and unknown intervals.
5. Validate attacking direction per period and preserve periods rather than
   silently flipping coordinates.

### 5. Implement Stage 4 movement analytics

1. Convert only validated pitch positions into `TrajectoryPoint` records.
2. Use the existing gap, period-boundary, smoothing, speed-spike, and coverage
   policies in `analytics/movement.py`.
3. Keep raw, smoothed, estimated, missing, and rejected observations distinct.
4. Add CV dashboard summaries for total distance, max/mean speed,
   acceleration/deceleration, coverage, estimated fraction, and unreliable
   segments.
5. Add heatmap, team shape, and positional-change views only when their
   coordinate quality requirements pass.
6. State clearly that these are movement indicators, not physiological fatigue
   measurements.

### 6. Implement and validate Stage 5 in order

1. Implement a declared pitch-control formulation first, with explicit player
   response/velocity parameters and uncertainty.
2. Build the input adapter for `accessible-space` and call DAS only when its
   required columns and possession state are present.
3. Define and validate OBSO's scoring/control formulation before exposing a
   number.
4. Treat EPV as a learned model: define the terminal reward, construct
   leakage-safe training/validation splits, train, evaluate on held-out
   matches, and version the model.
5. Add metric-specific validation reports and dashboard coverage badges.
6. Never convert a missing tactical input into zero.

### 7. Finish dashboard and reproducibility work

1. Show run status, processed-frame count, calibration status, coverage, and
   uncertainty prominently.
2. Add a run selector so an incomplete run cannot silently replace a completed
   run without showing that state.
3. Keep the tactical board unchanged unless a separate request authorizes it.
4. Record source hashes, model hashes, dependency versions, parameters, and
   calibration validation in every output manifest.
5. Add a small integration test for the CV API's completed, processing, and
   failed-run responses.

## Copy-ready prompt for the next agent

```text
Continue the Wadi Degla football CV module from docs/wadi_degla_cv_handoff.md.
Read PROJECT_RULES.md, AGENTS.md, and the handoff first. Do not stage, commit,
push, modify data/raw, or copy data/external into Git. Preserve unrelated
work. Work only on the authorized CV milestone.

First reproduce the current one-minute run and tests. Then integrate the
SoccerTrack camera remap using the actual mapx/mapy and homography assets,
validate held-out pitch landmarks, and keep transformed coordinates and
uncertainty explicit. Improve detector recall/precision with a documented MOT
evaluation and tiled inference if needed. Preserve ByteTrack >=0.25,<0.31.

After calibration passes, connect validated pitch positions to the existing
movement analytics. Then establish audited team attribution, ball state,
possession, and attacking direction. Implement pitch control and the
accessible-space DAS adapter only with complete inputs. Define and validate
OBSO. Treat EPV as a learned, leakage-safe model requiring labelled targets
and held-out evaluation. Do not publish any metric when an input is missing.

Expose run status, coverage, uncertainty, movement metrics, and tactical
readiness/results in the CV dashboard. Add focused tests for every new data
contract, run the full pytest suite, and report exact checks and limitations.
``` 