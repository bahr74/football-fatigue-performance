# Computer vision tracking module

## Purpose and current state

The module is an experimental video-to-position pipeline for a **fixed camera**. It can decode a video frame-by-frame, call a caller-supplied Ultralytics checkpoint, map recognized classes into conservative football object types, track detections, estimate player ground-contact coordinates with a user-supplied planar homography, and write per-frame JSONL plus a run manifest. Estimated player positions can be adapted to the existing movement analytics API when the caller explicitly maps the video to a match period.

It does not automatically calibrate the pitch, recognize jersey numbers, decide which team is home, recover a ball's 3D trajectory, estimate off-camera positions, synchronize video with match events, or measure fatigue. A person detector trained on generic images may return `person`, not football `player`; a football checkpoint with labels matching `player`, `goalkeeper`, `referee`, and `ball` is required to produce those roles. IDs are track IDs, not verified player identities. Team sides remain unknown until a caller fits colors and verifies which cluster is home.

**Evidence status:** the checked-in tests use synthetic boxes, coordinates, and frames. They establish code behavior only. No model has been fine-tuned on project data; no training dataset, validation set, or accuracy score is claimed. Real-footage accuracy and calibration error are unmeasured pending licensed sample footage and annotations.

## Install

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -e ".[dev,vision]"
```

The command requires network access if the optional packages are not already installed. The model checkpoint is a separate artifact; it is not fetched implicitly and must be supplied with `--weights`. Store model files under the ignored `models/vision/` directory. Record the model source, license, version, and checksum before showing results to others.

## Prepare a pitch calibration

Choose at least four non-collinear, visible pitch landmarks. Four define the projective transform, so use more than four and distribute them across the image when possible. Record each image point `(u, v)` in pixels and its corresponding pitch point `(X, Y)` in metres using the stated coordinate convention: 105 m by 68 m, center spot `(0, 0)`, length on X, width on Y. Set aside independent landmarks not used to fit the transform. The optional CLI `--calibration-check` reports held-out residuals when given a JSON file with `image_points` and `pitch_points` arrays. The fit residual threshold defaults to 0.25 metres and is configurable on the calibrator. Avoid points hidden by players, distortion-heavy edges, or an incorrectly identified line intersection.

Create a calibration JSON using measured correspondences:

```python
import json
from football_fatigue.vision import HomographyCalibrator

calibration = HomographyCalibrator()
calibration.fit_from_points(
    image_points=[(u1, v1), (u2, v2), (u3, v3), (u4, v4)],
    pitch_points=[(X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4)],
)
with open("calibration.json", "w", encoding="utf-8") as stream:
    json.dump(calibration.to_dict(), stream, indent=2)
```

Replace the symbols with manually measured coordinates. Do not use a calibration for a different camera, crop, resolution, or camera zoom. Homography assumes a planar pitch and does not correct lens distortion or camera movement.

An independent check file looks like this (use landmarks excluded from the fit):

```json
{"image_points": [[u5, v5]], "pitch_points": [[X5, Y5]]}
```

Use actual numeric coordinates, and include several checks spread across the view. `--calibration-check checks.json` places each error, the mean, and the maximum in the run manifest. One point is accepted mechanically but is not enough to characterize calibration quality; report the sample count and distribution.

## Run on a short clip

Keep the sample in ignored external storage, not Git. For SoccerTrack v2, accept the dataset's access terms on Hugging Face and download one MOT clip with its paired annotations instead of the full dataset. The authors' setup guide currently shows a match-filtered command: `./scripts/download.sh --dest ./data --match 117092 --match 117093`; run its `--help` first to select a single match and see the files fetched. It requires Git, Python 3.12+, and the Hugging Face CLI (`huggingface_hub`). Authentication and gate acceptance must be done by you. Record the sample identifier, exact files, license/access terms, and date. The current Hugging Face page identifies CC BY 4.0 and requires agreeing to share contact information; review current terms before redistribution.

For CV validation, obtain an `mot` clip and its paired MOT annotations. For position validation, the matching `gsr` annotations may help, but the authors' docs note that released GSR files diverge from the nominal schema and can be very large. Inspect the actual shipped `images` and `annotations` records with a streaming parser; do not load a multi-gigabyte annotation file wholesale. MOT labels support image-space detection/association evaluation. GSR labels support pitch-space location/identity comparison under their coordinate convention. A MOT clip by itself cannot validate calibrated pitch coordinates. Confirm the paired sample's actual annotation availability and alignment after download.

After placing a video, calibration JSON, and compatible checkpoint on disk:

```powershell
.\.venv\Scripts\python.exe -m football_fatigue.vision `
  "data/external/soccertrack/sample.mp4" `
  --calibration "configs/calibration.json" `
  --calibration-check "configs/calibration-check.json" `
  --weights "models/vision/football_detector.pt" `
  --output-root "data/processed/vision" `
  --run-id "sample-first-pass" `
  --tracker bytetrack `
  --confidence 0.25 `
  --imgsz 1280 `
  --device cpu `
  # Omit --max-frames to process the complete video.
```

For image-space evaluation when no valid pitch calibration exists (including this panoramic SoccerTrack clip), use explicit detection-only mode. It writes bounding boxes in pixel coordinates, leaves pitch coordinates and movement metrics null/unusable, and records that limitation in the manifest:

```powershell
.\.venv\Scripts\python.exe -m football_fatigue.vision `
  "data/external/soccertrack-v2/mot/clips/117093.mp4" `
  --detection-only `
  --weights "models/vision/player_detector.pt" `
  --output-root "data/processed/vision" `
  --run-id "soccertrack-117093-detection-only" `
  --tracker iou `
  --confidence 0.25 `
  --imgsz 1280 `
  --device cpu `
  --max-frames 30
```

Choose a new run ID for every attempt. The run is a smoke sample and does not compare predictions to the MOT annotations automatically; use the paired `mot/117093.txt` for a separate, documented evaluation.

Use a new `--run-id` for every run. Existing run directories are rejected. `--tracker iou` selects the simple IoU fallback; it is less robust to occlusion and should be described as such. If ByteTrack is requested but cannot initialize, the CLI stops rather than silently switching. CPU is the portable default; GPU performance depends on the installed PyTorch build and hardware.

## Detector tiling options

`--tile-width` alone keeps the historical behaviour: full-height strips, every strip of a frame in one model call, boxes merged by class-aware NMS at IoU 0.5. Three opt-in options extend it (each requires `--tile-width`, is recorded in the detector provenance of the cache and run manifests, and is fixed by a cache on replay):

- `--tile-height H` makes a 2D grid of overlapping tiles (same overlap on both axes, tiles flush with the frame edges, row-major order). Per-tile letterboxing to `--imgsz` sets the effective scale: for example 1280x1280 tiles at `--imgsz 1920` upscale by 1.5.
- `--tile-batch N` sends at most N tiles per model call to bound peak GPU memory. It does not change which tiles are processed; results can still differ at floating-point level on GPU because batch composition affects the convolution kernels.
- `--tile-merge person_agnostic` (with `--tile-merge-cross-role-iou`, default 0.7) merges more conservatively than class-aware NMS: ball and unmapped labels are merged class-aware and never with people; person boxes of the same role are merged at IoU 0.5 as before; person boxes of *different* roles are merged only when they nearly coincide (IoU >= 0.7) and the more confident box keeps its own role; a box cut by an interior tile edge is dropped only when a box from another tile extends past that edge, contains at least 80% of it, and is not more than 0.2 less confident. Overlapping boxes below these limits stay as separate observations. Suppressed boxes are not stored per observation; counts per rule and role pair are written to `detector.merge_statistics` in the cache manifest and `summary.detector_merge_statistics` in the run manifest. A suppressed role label is therefore evidence that is counted but not kept.
- `--tile-merge edge_aware` is `person_agnostic` with the tile-edge rule applied to all person boxes *before* the same-role NMS (and again on the survivors). With `person_agnostic` a truncated box that is more confident than the complete box from the neighbouring tile wins the NMS, so the complete box is gone before the edge rule sees it (observed on the 117092 benchmark at the y=626 cut of the 2D grid; a diagnosis from one clip, not a general claim). The edge rule itself is unchanged (same constants, recorded in the detector provenance); the extra counters are `suppressed_edge_truncated_before_nms` and `edge_truncated_role_differs` (the dropped truncated box carried a different role label than the box that completes it). Boxes without a qualifying completing box are kept. Ball and unmapped labels are untouched.

None of this has been validated on video yet: the unit tests use a stub model, and whether the merge is correct on real footage needs the reference boxes or visual inspection.

## Detection cache and replay

Tracker, team and evaluation experiments should not each pay for detector inference. `--save-detections` additionally writes every raw detector observation to `<run>/detection_cache/`; `--replay-detections` tracks stored observations without loading the detector. Both are opt-in; without them the CLI behaves as before.

```powershell
# Save: same run as usual, plus the cache. Without --cache-confidence-floor the floor equals --confidence.
.\.venv\Scripts\python.exe -m football_fatigue.vision data/raw/soccertrack_v2/mot/117092.mp4 --detection-only --weights models/vision/player_detector.pt --run-id RUN --imgsz 1920 --tile-width 1280 --confidence 0.4 --device cuda:0 --max-frames 625 --save-detections

# Save with a lower floor: YOLO runs at 0.10 and everything above it is stored, but only confidence > 0.4 reaches the tracker.
.\.venv\Scripts\python.exe -m football_fatigue.vision ... --confidence 0.4 --cache-confidence-floor 0.1 --save-detections

# Replay (no --weights, no GPU needed; tracker settings may change freely).
.\.venv\Scripts\python.exe -m football_fatigue.vision data/raw/soccertrack_v2/mot/117092.mp4 --detection-only --replay-detections data/processed/vision/RUN --run-id RUN-replay --confidence 0.4 --max-frames 625
```

- **Cache format:** `cache_manifest.json` (schema version, `complete` status, frame window, record count and SHA-256, video and weights hashes, detector configuration) plus `detections.jsonl` (one record per frame; per observation: class id and label, mapped class, confidence, box, and the turf-filter and pitch-mask decisions). Unmapped labels and filtered boxes are kept with their decisions, so a replay reproduces the filtering without the pixels.
- **Integrity:** a replay is refused for an incomplete or altered cache, a different schema version, a different video (SHA-256, size, frame rate), different `--weights`, or a frame window outside the cached one. Detector flags (`--imgsz`, `--tile-width`, `--device`, `--half`, `--pitch-keypoints`, `--mask-margin`) come from the cache and cannot be overridden. A cache directory is never overwritten.
- **Two thresholds:** the *cache floor* (`--cache-confidence-floor`, default `--confidence`) is the confidence YOLO runs with; every returned observation above it is stored, including filtered and unmapped ones. The *tracking-input threshold* (`--confidence`) decides which kept observations reach the tracker (`confidence > threshold`, strict, as in Ultralytics NMS). They are recorded separately in the cache manifest (`detector.confidence_floor`, `tracking_input.confidence_threshold`) and the run manifest. The floor must satisfy `0 < floor <= --confidence`, applies only with `--save-detections`, and is rejected with a replay. Tracker activation, association and filtering are unchanged. Caveat: a lower floor feeds more candidates into NMS and the detector's per-image limits, so stored observations above 0.4 are not guaranteed identical to a plain 0.4 run; this must be measured, not assumed.
- **Replay modes:** `compatibility` (default) requires `--confidence` to equal the saving run's tracking-input threshold and reconstructs its tracker input. `experimental` allows any `--confidence` at or above the cache floor, for example a cache saved at floor 0.10 for a 0.40 run and replayed at 0.25; the manifest labels the run experimental. A confidence below the floor is rejected in both modes: those detections were never stored, and a cache cannot be created retroactively for an earlier run such as B2.
- **Three kinds of result, not to be conflated:** (1) the original B2 run is a historical baseline with no cache; (2) a new cache run is a fresh detector run whose agreement with B2 has to be measured, not assumed; (3) experimental replays are different tracker inputs and are not comparable with either as if equivalent.
- **What replay still needs:** the video is decoded again because the kit classifier reads frame pixels. Replay output is deterministic given the cache and the tracker settings; whether a new detector run reproduces an earlier run's detections exactly on GPU is a separate question that must be measured.

## Standard MOT evaluation

`football_fatigue.vision.mot_eval` computes HOTA (with DetA, AssA, LocA), CLEAR (MOTA, IDSW, Frag, MT/PT/ML) and Identity (IDF1/IDP/IDR) with TrackEval (dev dependency, `trackeval>=1.3,<2`), without re-implementing any metric. It works on saved `frames.jsonl` and a MOTChallenge reference file; it never runs a detector.

```powershell
.\.venv\Scripts\python.exe -m football_fatigue.vision.mot_eval data/processed/vision/RUN data/raw/soccertrack_v2/mot/117092.txt --start 0 --end 625 --view all_non_ball --output data/processed/vision_eval/NAME/RUN_all_non_ball.json
```

- **Conventions:** frames are 0-based decoded-frame indices (`--gt-frame-base 1` for a 1-based reference); the window is `[start, end)` and every frame in it is a timestep, including frames with no reference boxes or no prediction record. Views: `all_non_ball` or `player_goalkeeper`. HOTA is averaged over alpha 0.05-0.95; CLEAR and Identity use `--iou` (default 0.5). `--min-confidence` is a strict `>` filter.
- **Input handling (counted in the report):** predictions without a box, non-finite or non-positive-size boxes are dropped; a repeated track id within a frame is an error. Reference rows with conf field 0 are MOTChallenge "ignore" rows and are dropped; `-1` is kept. The reference class column is not used.
- **Reports:** versioned JSON (`schema_version`, TrackEval and adapter versions, SHA-256 of both inputs, full configuration), written to a path you choose and never overwritten. `--export-motchallenge DIR` also writes a TrackEval `MotChallenge2DBox` layout (frames renumbered from 1 inside the window); a test checks that TrackEval's own loader gives the same numbers as the in-memory path.
- **Approximate vs standard:** `tracking_eval.py` is kept unchanged. The report repeats its numbers under `approximate_project_metrics` (`idf1_approximate`) so the two are never mixed up. They differ because that evaluator re-matches every frame by IoU alone, whereas TrackEval's CLEAR matching prefers keeping the previous identity (fewer ID switches) and its IDF1 uses a global identity assignment.
- **Known TrackEval behaviour:** a timestep in which nothing at all is predicted does not reset the previous-match state, so a gap covering such frames is not counted as a fragmentation. HOTA's AssA counts missed frames against the identity.

## Class-safe person tracking

- **Ball isolation:** the person tracker only receives player, goalkeeper, referee and person detections. The pipeline removes ball detections before tracking (the tracker also drops and counts any that reach it). The ball is not tracked: each frame's `ball` is its highest-confidence detection, and `VisionTrackingPipeline.last_ball_detections` keeps all of that frame's ball detections for a future ball module. Because the ball no longer takes track ids, local track ids can differ from older runs even when every person is associated identically (the B2 clip: same tracks, ids shifted by the 3 ids the ball had used).
- **Association** is class-agnostic across person roles (ByteTrack always was; the `--tracker iou` fallback used to require equal classes and no longer does), so a changing detector label never splits a track.
- **Role estimate:** `detected_class` is the track's role, the class with the most decayed, confidence-weighted evidence (half-life 2 s of video, challenger needs 1.25x the current role's evidence). `observed_class` is this frame's raw label and `role_confidence` the role's share of the evidence (not a probability). A role is a label, never a verified identity; it can still be wrong, and nothing here uses kit colour, team, camera or dataset information. `--role-policy latest` reports the raw label as `detected_class` instead.
- **Outputs:** `frames.jsonl` players gain `observed_class` and `role_confidence` (all earlier keys unchanged); `manifest.json` records `configuration.class_safety`, `summary.role_statistics` and `summary.ball_detections_not_tracked`. Detection caches are unchanged and replay unchanged.

## Global player identity layer (opt-in, M6)

`football_fatigue.vision.identity` is a post-tracking pass over an existing run. It never changes a run in place and is not used by `/cv`, the tactical board or any default.

```powershell
.\.venv\Scripts\python.exe -m football_fatigue.vision.identity data/processed/vision/RUN --output data/processed/vision/RUN-identity --video VIDEO.mp4 --config data/processed/vision_eval/m6-global-identity-v1/frozen_config.json [--online] [--interpolate-max-gap 12]
```

- **Team evidence** (`team_appearance.py`): per-video and per-window, with no fixed kit colours.
  - A turf colour model is fitted in Lab from pixels around person boxes. Turf is removed from torso crops, and a 6-D colour descriptor describes each box.
  - The two team clusters come from k-means over the run's own boxes (`kmeans2_box`).
  - A box far from both clusters votes `other` (goalkeepers, referees). A box about equally close to both does not vote.
  - Team labels are cluster names (`team_0`/`team_1`), not home/away.
- **Tracklets and swap-aware splitting:**
  - A local track is split only on two independent cues: a confident team-vote change point (30 decided votes on each side, ≥95% share) plus an interaction, tracker gap or motion jump near it.
  - Single-cue candidates are rejected and logged.
  - The interaction interval becomes an `ambiguous` tracklet.
  - A split whose partner track shows the mirror-image change is repaired as a swap (crossed re-link). A split without a partner is restored.
- **Global linking:**
  - Tracklet ends are linked across gaps of ≤75 frames.
  - The motion gate is constant-velocity, with a spread that grows with the gap; the p90 error was fitted on development windows.
  - Team and role (goalkeeper) gates act only when both sides are confident. There is also a size gate.
  - A link must be unique and must not overlap in time. The solver abstains if a competing link is nearly as cheap.
  - Every link, rejection and abstention is written to `identity_graph.json`.
- **Outputs:**
  - Per player, `frames.jsonl` keeps every box. `track_id` is replaced by the global id, and the original is kept as `local_track_id`. New fields: `tracklet_id`, `global_id`, `identity_status` (`confirmed` / `provisional` / `ambiguous` / `unknown`), `identity_confidence` (logistic, calibrated on dev windows), `team_cluster`, `team_confidence` and `position_status` (`observed`).
  - Optional interpolated boxes go in a separate `identity_interpolated` list and never into `players`.
  - `manifest.json` holds the source hash, configuration, team-model diagnostics, status counts and timing.
- **Online mode:** a 250-frame window is re-solved every 25 frames, and identities are emitted 50 frames after the fact. The output is therefore causal, with a fixed 2 s latency.
- **Analytics:** `identity_trajectory_points` (policy `m6-identity-analytics-v1`) returns only observed boxes with usable pitch positions and the selected statuses. By default that means `confirmed`, more than 12 frames away from any ambiguous, unknown or tracklet-change frame of that identity. A coverage report comes with it. Each box has one global id, so the per-identity selections never double count.
- **Evidence and limits:**
  - See `data/processed/vision_eval/m6-global-identity-v1/README.md`: dev and held-out windows, ablations, full-length runs, calibration, runtime.
  - Identity evidence only. This is not jersey-number identity and not re-identification across matches.
  - It was validated on two fixed-camera SoccerTrack matches only, both blue-vs-white fixtures. It is not evidence of camera-agnostic generalisation.

## Outputs and interpretation

Each output run contains:

- `frames.jsonl`: one record per decoded frame, with timestamp, track IDs, detections, calibrated player coordinates, nullable speed, and ball observation. Missing values remain JSON `null`.
- `manifest.json`: input hashes, calibration, confidence/device/tracker configuration, video rate and dimensions, runtime versions, output counts, and explicit limitations. Held-out calibration residuals are included if supplied. The status remains `processed_unvalidated`; this means a run completed, not that its accuracy passed.

Player positions use the bottom-center of the detection box as a ground-contact proxy. They are marked estimated. The CLI's `usable_estimated_player_positions` means the projection fell within the configured pitch bounds; it is not a validation score. The current detector wrapper does not write annotated video. Ball positions are intentionally not marked usable: a ground-plane homography cannot recover an airborne ball's height.

## Relationship to movement analytics and the tactical board

The existing tracking-data analytics include distance/speed/acceleration summaries, opt-in median coordinate smoothing, observation-count heatmaps, sampled team-shape summaries, and per-player changes between adjacent five-minute mean-position windows. CV runs now also emit `analytics.json` with the same quality-aware Stage 4 movement contract for calibrated, usable player positions and a Stage 5 input-readiness report. These outputs keep missing, estimated, and unvalidated values distinct; a detection-only run therefore reports movement as unavailable rather than treating pixel motion as metres. The tactical board remains separate and unchanged.

The project-specific tactical metrics PC, EPV, OBSO, and DAS remain gated by their documented inputs. The CV report records each metric's readiness and missing requirements, but does not publish a score without calibrated pitch coordinates, stable team labels, possession/event definitions, and validation data. In particular, EPV still requires a declared labelled target and held-out evaluation, and DAS requires the approved dependency and pass assumptions. Event recognition and VAEP/OBV remain later work. No model was trained or fine-tuned on project data.

## Initial SoccerTrack v2 sample check

The local sample is `mot/clips/117093.mp4` with paired `mot/117093.txt`; both are kept under ignored `data/external/` and must not be committed. Observed video metadata: 4096 × 1080 pixels, 25 fps, 5,976 decoded frames. The annotation file has 131,472 rows over frame indices 0–5,975, 22 rows per frame, and no malformed rows in the complete streaming parse. SoccerTrack's v1.2 release also provides small, match-specific `raw/117093/` calibration assets: camera intrinsics, a homography, 65 keypoints, OpenCV remap arrays, and padding metadata (26.5 MB total). The repository documents these maps as its lens-correction/perspective-calibration inputs. Their integration into this project's CLI remains outstanding.

**Detector diagnostic (not a benchmark):** `mobadam/football-player-detection` loaded with classes `{ball, player, referee, goalkeeper}`. On 30 evenly spaced frames, CPU, confidence 0.25, inference size 1280, player and goalkeeper predictions only, and greedy one-to-one box matching at IoU ≥ 0.5, the diagnostic yielded 577 matches from 759 predictions and 660 annotation rows (micro precision 0.760, micro recall 0.874). This is a sparse frame sample, not AP, MOTA, IDF1, or a full-video score. The annotation-to-class mapping and greedy matching protocol are provisional. A 640px check showed much lower recall, consistent with small players being lost when the full 4096px panorama is resized.

**Detection-only CLI smoke run:** `data/processed/vision/soccertrack-117093-detection-only-30f-20261018/` contains 30 sequential frames from the real clip, processed with the IoU fallback, confidence 0.25, and inference size 1280. Its 805 non-ball tracked records comprise 777 `player`, 22 `referee`, and 6 `goalkeeper` records; pixel bounding boxes are present and pitch coordinates/speeds are null. The manifest status is `processed_unvalidated`. This checks the end-to-end CLI/output path, not tracker identity quality or calibrated positions.

The sample image is a strongly curved panoramic view. As a calibration diagnostic, the supplied fisheye camera model was applied to the 65 image landmarks, a planar homography was fit on 52, and 13 held-out landmarks were projected. Held-out error was median 2.61 m, mean 2.67 m, and 95th percentile 4.44 m. This is too large for trustworthy movement analytics; do **not** interpret pitch coordinates, distances, heatmaps, or positional analytics from this clip as validated. The current project CLI does not yet apply the SoccerTrack remap arrays. It does have an explicit detection-only mode that retains pixel boxes while withholding pitch coordinates and movement metrics.

SHA-256 fingerprints recorded during this review (2026-10-18):

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `mot/117093.txt` | 6,221,412 | `f081ed8f4c7feb76e0fb0982c645fe6ae2039cf69bf2ffeaa787896fa01e2331` |
| `mot/clips/117093.mp4` | 304,980,773 | `dfe2260aaed37dacc98204b76f6f8e0f31604f8e7342926b6586e677abc91e43` |
| `models/vision/player_detector.pt` | 53,140,901 | `05ee5d6c1ed37d92b0425a4d98cbb656bd71869860615590dbfaae913d75087f` |

Calibration assets in `data/external/soccertrack-v2/raw/117093/` (all obtained from release `v1.2`):

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `117093_camera_intrinsics.npz` | 1,996 | `fe3f14a96495aa6706e6bf716490d599e0818ed0a48e7bf8e4387e7fe985c720` |
| `117093_homography.npy` | 200 | `deb667816e2b0a98c536cf65d08d03d0c5d300cc0027425eb1bcee8f0e0426f7` |
| `117093_keypoints.json` | 3,852 | `bcc874620cfa1bcbd49b33a6132abf51f35525f25cafb677084e8ce3e803a94f` |
| `117093_mapx.npy` | 17,694,848 | `50328635d0121643037f91df51af3fc6e4be430ea304548a44df598f15ea47b6` |
| `117093_mapy.npy` | 8,847,488 | `30229e8ad1efecb062c1687acd2b9a5752f8d7b699b91c4d757a0a00fe28af8a` |
| `117093_padding_info.csv` | 154 | `0b7e73f40dad775807790860df61fcaada818fcbd634f105d454bb4fdd43ead5` |

The dataset page currently identifies the gated sample as CC BY 4.0; the detector model card identifies Apache-2.0. These licenses apply to those artifacts, not to the project as a whole; recheck the access and attribution terms before distributing demo outputs.

## Real-footage validation checklist

1. Confirm footage and annotation licenses allow local processing and mentor demonstration.
2. Verify the detector checkpoint's class names and license. Test one frame and inspect boxes against the image.
3. Measure pitch landmarks on that exact camera view. Hold out at least one visible landmark; report its reprojection error in metres and pixels.
4. Compare detections/tracks against MOT ground truth under a documented protocol. Report detection precision/recall or AP, association quality under a defined MOT metric, sample size, class mapping, and exclusions. Do not infer these metrics from a successful run.
5. Compare projected player footpoints against valid pitch-space GSR positions on matched frames. Report median and percentile position error, usable/omitted observations, annotation alignment procedure, and calibration assumptions.
6. Keep this output separate from physiological measurements. Tracking-derived distance/speed are movement indicators, not validated fatigue labels.

The repository also includes [a synthetic calibration JSON](vision_synthetic_calibration.json) for validating the CLI's calibration-file parsing only. It maps an invented rectangle to a pitch and must not be used on real footage.

The [SoccerTrack v2 Hugging Face page](https://huggingface.co/datasets/atomscott/soccertrack-v2) is gated and currently reports CC BY 4.0. Its [author setup guide](https://github.com/AtomScott/SoccerTrack-v2/blob/main/docs/setup.md) provides match filtering; its [GSR format guide](https://github.com/AtomScott/SoccerTrack-v2/blob/main/docs/format-gsr.md) records the released-schema caveats. The CV implementation and tests can be exercised without downloading it, but real-world validation cannot be claimed until the actual paired labels are inspected and evaluated.

One possible external detector is [mobadam/football-player-detection](https://huggingface.co/mobadam/football-player-detection), whose card lists YOLO26 football classes and Apache-2.0. Its local `player_detector.pt` checkpoint is 53,140,901 bytes (SHA-256 above); the card reports broadcast-footage training, not training on this project's data. It loaded under the project's installed Ultralytics 8.4.182 and exposed classes `ball`, `player`, `referee`, and `goalkeeper`. Ultralytics package licensing is separate from this checkpoint's license and should be reviewed for the intended mentor/demo use.
