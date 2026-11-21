# Coach analytics: feasibility audit, delivered insights and roadmap (SoccerTrack 117093)

Evidence date 2026-10-19. Run: `data/processed/vision/soccertrack-117093-full-v3-refined-20261019/`.
Everything reported as "measured" was computed from saved files in this repository; nothing was retrained or re-run on the video except a 40-frame ball probe.

## 1. Feasibility audit

| Metric | Inputs required | Dataset provides | Missing | Validation | Status |
|---|---|---|---|---|---|
| Team shape (length, width, compactness) | Per-frame team positions; metres for absolute values | Detected feet + colour labels (blue/white), image pixels | Validated pitch calibration | vs reference tracks (Pearson r of 1 s series) and vs prototype pitch projection | **Image-space delivered**; metres unavailable |
| Positional formation / roles | Stable identity per player + metric positions | Tracks fragmented (IDF1 0.27) | Identity, calibration | Needs labelled formations | Unavailable |
| Defensive line | Attacking direction per team + across-pitch depth in metres | Goalkeepers often undetected | Direction, calibration | Needs labelled direction | Unavailable |
| Spatial occupation | Foot points | Foot points + labels | none for image-space | Re-derived from JSONL (exact match) | **Delivered (image-space)** |
| High pressing / pressure | Ball carrier, metric distances, identities | none usable | Ball, calibration, identity | - | Unavailable |
| Possession % | Match-ball track + attribution | Ball is detectable but a static spare ball outranks it | Match-ball identification, possession labels | >= 85% vs manual labels | Unavailable |
| Distance, sprints, high-speed running | Metric positions at >= 0.5 m accuracy; one identity per player | Pixel tracks only | Calibration, identity | speed plausibility, reference tracks | Unavailable |
| Pitch Control, EPV, OBSO, DAS | Metric positions + velocities + ball + possession (+ trained value model) | none of these | All | - | Unavailable |

## 2. Camera and calibration investigation

Files: `117093_keypoints.json` (65 annotated pitch points: pitch metres -> video pixels), `117093_camera_intrinsics.npz` (fisheye `K`, `D`, `Knew`; its pickled extrinsics were deliberately not loaded), `117093_homography.npy`; `mapx.npy`/`mapy.npy` (undistortion maps) are absent.
Held-out (8-fold) error of each mapping, in metres on the 65 keypoints (`python -m football_fatigue.vision.pitch_prototype`):

| Mapping | Median | p90 |
|---|---|---|
| Supplied homography (either direction) | thousands of m / hundreds of px - not usable | |
| Single homography fitted to raw pixels | 6.33 | 11.06 |
| Supplied fisheye K,D + homography | 2.53 | 4.19 |
| Plain radial model + homography | 5.94-6.6 | 10.2-10.5 |
| **Fisheye + pose fitted to keypoints (prototype, 4 radial terms)** | **2.04** | **4.02** |

Train reprojection: 6.8 px median, 13 px max. The fisheye + pose model is the right family (the supplied intrinsics give the same pose by PnP), but about 2 m median error is not accurate enough for anything in metres.
Acceptance (held-out): team shape in metres median <= 1.0 m and p90 <= 2.0 m; distance/speed median <= 0.5 m and p90 <= 1.0 m. **Both not met.** Likely next steps: more or better keypoints (line intersections along the whole pitch, sub-pixel), a distortion model with additional terms validated on independent points, or recovering the original undistortion maps.

## 3. Ball probe (40 evenly spaced frames, YOLOv8n tiled, class "sports ball", conf >= 0.05)

Every frame has a high-confidence ball candidate (~0.7), but the top candidates are mostly 3 static spare balls on the touchline. A non-static candidate with conf >= 0.25 appears in only 9 of 40 sampled frames (22%). There are no ball labels, so recall of the match ball cannot be measured. Possession stays unavailable (criteria: >= 80% match-ball coverage with spare balls rejected, >= 85% possession accuracy vs manual labels).

## 4. Acceptance criteria for advanced metrics (evaluated in `coach_insights.json`, shown on the dashboard)

| Metric | Criteria | Measured | Met |
|---|---|---|---|
| Calibration for team shape (m) | median <= 1.0 m, p90 <= 2.0 m | 2.04 / 4.02 m | no |
| Calibration for distance/speed | median <= 0.5 m, p90 <= 1.0 m | 2.04 / 4.02 m | no |
| Identity continuity | IDF1 >= 0.80 and <= 1 switch per player-minute | IDF1 0.27, 7.5 switches per player-minute | no |
| Match-ball detection | >= 80% of frames, spare balls rejected | 22% of sampled frames show a non-static candidate; no labels | no |
| Possession attribution | >= 85% vs manual labels | no labelled data | not measurable |
| Attacking direction | known for every period | not available | no |

Each advanced metric (formation, defensive line, possession, pressing, distance, sprints, Pitch Control, EPV/OBSO/DAS) needs a subset of these and is therefore **unavailable**.

## 5. Delivered coach analytics (image-space)

Module `football_fatigue.vision.team_shape`; CLI:
```
python -m football_fatigue.vision.team_shape RUN_DIR --mot data/external/soccertrack-v2/mot/117093.txt \
  --kit-reference data/processed/vision_benchmarks/phase1/kit_reference_117093.json \
  --calibration data/processed/coach/calibration_prototype.json --ball-probe data/processed/coach/ball_probe_40frames.json
```
Per frame and colour group (>= 6 labelled players, otherwise skipped): length (10-90 percentile span along image x), width (along image y), spread (RMS distance from the median position, farthest player ignored when >= 8), block position; spans divided by the group's median box height ("player heights"). Notable moments (compact, spread out, stretched, squeezed, block shift, teams differ) come from each group's own distribution with a >= 20% deviation from its median, >= 3 s long, ranked and capped.

Validation (full video, reference = SoccerTrack MOT with derived colour labels, 10 players per group):

| Measure | Blue r | White r |
|---|---|---|
| Block position | 1.00 | 0.99 |
| Length along the pitch | 0.84 | 0.66 |
| Width across the pitch | 0.91 | 0.66 |
| Spread / compactness | 0.77 | 0.35 |

19 of 20 spread/compact/stretch/squeeze moments lie on the same side of the reference median; 18 of 27 moments are shown by default (reliable measure and reference agrees). White spread is the weakest measure (only ~7-8 of 10 players are detected), so those moments are hidden unless requested. The image-space spread correlates r = 0.83 (blue) and 0.84 (white) with the prototype-projected spread. 20% of frames have fewer than 14 of 20 outfield players labelled, 4% have doubtful colour labels, 0% have low mean confidence. Three moments were checked by eye on the annotated video (blue stretched ~2:16, blue squeezed ~3:56, white squeezed ~1:18) and match the description.

Perspective caution: a pixel is worth more metres far from the camera and the lens bends the lines. Values are depth-adjusted by player height, events are relative to each group's own median, near-edge moments are flagged, and none of it is presented as metres, speed, possession, pressing or fatigue.

## 6. Roadmap to unlock the rest

1. **Calibration**: add keypoints (or find the missing undistortion maps), fit the fisheye + pose model with more radial terms, and verify on independent points until median <= 0.5 m / p90 <= 1.0 m. Then re-compute team shape in metres and re-check the image-space proxies against it.
2. **Identity**: add appearance/re-identification to the tracker or an offline global association; criterion IDF1 >= 0.8 and <= 1 switch per player-minute. Unlocks distance, sprints, high-speed running.
3. **Ball and possession**: reject static balls (motion + pitch-region gating), track the match ball with a small ball-specific model, label ~200 frames of possession for evaluation. Unlocks possession %, pressing.
4. **Direction and roles**: detect goalkeepers reliably (the yellow kit is often missed) to fix attacking direction; then defensive line and formation.
5. **Pitch Control / OBSO / DAS**: after 1-3; EPV additionally needs a trained and validated value model on suitable event data.
