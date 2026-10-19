# CV detection, tracking and kit-colour report (SoccerTrack 117093)

Scope: Wadi Degla CV module only. All numbers are against the paired MOT file
(0-based frames, 5,976 frames, 22 labelled players per frame, referees/staff not
labelled), matched with the Hungarian algorithm at IoU >= 0.5. Dev clips: frames
0-249 (begin), 3000-3249 (middle), 5726-5975 (end). Reproduce with
`python -m football_fatigue.vision.tracking_eval RUN_DIR data/external/soccertrack-v2/mot/117093.txt --start A --end B --kit-reference data/processed/vision_benchmarks/phase1/kit_reference_117093.json`.

## Diagnosis of the saved full run (baseline)
- Over the full video IDF1 = 0.18 (1,313 track IDs for 22 players, 1,408 ID switches);
  10 s clips score 0.58-0.79. Identity fragmentation is the dominant failure.
- 90% of false positives have no MOT overlap. Crops show real people not in the
  MOT file (referee in black, bench staff, spectators), not hallucinated boxes on empty
  pitch. Only ~10% are duplicates/mislocalised boxes.
- Misses: players bunched together, small boxes (< 35 px tall), goalkeepers at distance.
- Detector boxes are geometrically accurate (median height ratio 1.01 vs MOT);
  coordinates and frame alignment are correct. ByteTrack never saw low-confidence
  detections (detector floor 0.40 > activation 0.30), and its IoU gate (0.75) was too strict for 20x45 px boxes.
- The team classifier existed but was never fitted, so every team was "unknown".

## Detection experiments (same 1,000 dev frames, conf >= 0.40, pitch-masked)
| Model / geometry | P | R | F1 | ms/frame |
|---|---|---|---|---|
| yolov8n, 1280 tiles @960 (current) | 0.846 | 0.780 | 0.812 | 124 |
| yolov8n, tiles @1280 | 0.777 | 0.846 | 0.810 | 162 |
| yolov8n, 1024 tiles @1024 | 0.748 | 0.843 | 0.792 | 206 |
| yolov8n, 1280 tiles, overlap 0.4 | 0.844 | 0.782 | 0.811 | 145 |
| yolov8s | 0.798 | 0.830 | 0.814 | 192 |
| yolov8m | 0.745 | 0.857 | 0.797 | 336 |
Geometry, overlap and larger COCO models do not materially change F1 (0.79-0.82);
the pretrained COCO detector is the ceiling. Detection configuration was therefore left unchanged.

## Tracking (ByteTrack, offline grid on cached detections, per-clip IDF1)
Tuned: activation 0.5, buffer 75 frames, matching threshold 0.95 (dev clips: IDF1 0.62 -> 0.70, ID switches 290 -> 108).

## Kit colour (blue / white / other / unknown)
Colour labels only, not home/away and not identity. Adaptive saturation split
(white shirts looked bluish in shade late in the match; fixed thresholds failed
there: 442 of 610 white-kit boxes mislabelled blue in the end clip). Goalkeeper
and referee kits are "other", never a team. Reference labels were derived from
colour statistics and visual inspection, not independently annotated.

## Result: baseline vs improved on short clips (full pipeline run)
| Clip | Version | P | R | IDF1 | ID sw. | Track IDs |
|---|---|---|---|---|---|---|
| begin | baseline | 0.877 | 0.907 | 0.789 | 25 | 65 |
| begin | improved | 0.921 | 0.885 | 0.840 | 14 | 32 |
| middle | baseline | 0.816 | 0.634 | 0.587 | 109 | 123 |
| middle | improved | 0.854 | 0.595 | 0.640 | 28 | 33 |
| end | baseline | 0.867 | 0.823 | 0.584 | 77 | 83 |
| end | improved | 0.895 | 0.803 | 0.709 | 39 | 36 |

Kit labels: coverage 81-86% of matched boxes; accuracy on labelled boxes 94.6-98.5%;
blue<->white swap rate 1.6-4.1%.

## Remaining failures
Recall in crowded scenes (middle clip 0.60); ~10-40 ID switches per 10 s; referee,
staff and spectators remain detected as persons (kit "other"/"unknown"); the
saved full-video demo run predates these changes.

## Synchronisation root cause, refinement and the annotated full video

**"Boxes ahead of players" was synchronisation, not tracking.** The source MP4 has a container
`start_time` of 88110/90000 = 0.979 s. Browsers play container timestamps, so `currentTime * 25` is
~24.5 frames ahead of the decoded index (cv2/PyAV 0..5975) that `frames.jsonl` uses. The dashboard now reads
the offset (`video_start_time_s`) and maps `floor((t - start) * 25)`; verified in Edge at 7 timestamps with 0
frame error. Raw YOLO and YOLO+ByteTrack have identical IoU vs MOT (0.841) and a lead of -0.05 frames, so
the tracker adds no lag. The burned-in MP4 is rendered by source frame index and its timeline starts at 0.

**Refinement (offline `postprocess`)**: tracklet stitching (gap <= 75 frames, plausible image-space
displacement, similar size, no kit conflict, one-to-one) plus short-flicker removal (< 8 frames). Kit labels
keep the online per-frame label and fall back to the tracklet's own plurality (>= 0.6) when it is unknown.
Pooling votes over a whole stitched chain was tried and rejected: on the full video it cut coverage to 0.55 and
accuracy to 0.926 (lighting drift plus chains bridged by unlabelled tracklets).

**Full video, 5,976 frames, vs MOT (IoU 0.5; MOT is evaluation-only)**

| Version | P | R | IDF1 | ID switches | Fragmentations | Track IDs |
|---|---|---|---|---|---|---|
| Original saved demo | 0.833 | 0.782 | 0.182 | 1,408 | n/a | 1,313 |
| Pass 1 (tuned tracker + online kit) | 0.871 | 0.756 | 0.243 | 702 | 3,566 | 287 |
| Refined (exported) | 0.871 | 0.756 | 0.270 | 657 | 3,566 | 120 |

Recall is lower than the original (0.756 vs 0.782) because the higher confidence floor and tracker activation
trade recall for precision; detection weights were not changed. 120 track IDs vs 22 players is still ~5.5
IDs per player.

**Kit colour vs derived reference (coverage / accuracy on labelled / blue-white swap)**: pass 1 0.898 / 0.963 /
3.6%; refined 0.978 / 0.949 / 5.0%. The reference is derived from torso colour plus visual inspection, not
independent annotation. 290 goalkeeper-kit boxes are assigned to a team (of ~6,600).

**Outputs** (all under `data/processed/`, gitignored): `vision_exports/full/annotated_full_117093.mp4`
(4096x1080, 25 fps, 5,976 frames, H.264 crf 20, 274 MB), `vision_exports/full/preview_30s_2048.mp4`
(frames 3000-3749, 2048 px wide), `vision_exports/sample20s/annotated_sample_20s.mp4`, dashboard run
`vision/soccertrack-117093-full-v3-refined-20261019/`. Export verification: same-index source frame is closest
(mean diff 2.19 vs 2.25), 96/96 sampled box edges carry the expected kit colour; frames inspected visually at
the start (10), middle (2988) and end (5970).

**Limitations**: yellow goalkeepers and tiny far-touchline players are often missed; crowded clusters lose
recall; referee/staff are detected (shown as "other"/unknown) ; kit labels are colours, not home/away; IDs are
tracker labels, not identities; no metric analytics (calibration not validated); the source container starts at
0.979 s while the export starts at 0.

## Team analytics dashboard (`/cv`)

Backed by `GET /api/cv/analytics` (cached, derived only from the saved `frames.jsonl`; no detector run):
match overview and quality warnings, per-kit detections over time (2 s bins; empty bins are null), per-kit
image-space occupancy, camera-view regions (thirds x near/far halves of the 2.5-97.5 percentile extent of foot
points; not tactical zones), tracking diagnostics (duration, gaps, confidence, refinement, reference
evaluation from `tracking_eval.json` / `comparison.json` in the run folder) and pixel-space movement for
stable tracks (>= 100 frames, >= 80% continuity). Saved videos are served only from a fixed whitelist
(`/media/soccertrack/annotated.mp4`, `/media/soccertrack/preview.mp4`, `?download=1` for attachment).
Validation: every overview, series, heat, region, minute, duration, gap and confidence figure was re-derived
independently from the JSONL and matched exactly. Advanced metrics remain unavailable.
