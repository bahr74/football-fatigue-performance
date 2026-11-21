# Session B · Detection quality diagnosis v1 (development data, CPU only)

Date: 2026-11-18 · `diag-v1`, `replay-v1` · branch `feature/video-occlusion-events` (uncommitted)
Code: `src/football_fatigue/video/detection/` · guide: `docs/video_detection_diagnostics.md`

## Answer first

**Most "detector misses" shown in the dashboard are not YOLO26-L failures.** They are visible players that YOLO26-L
detected with high confidence, after which our **turf (grass-colour) post-processing filter** removed them.

* **Where visible players are lost** (117093, 118575 and 118576): 69–79% of visible players lost before the tracker are
  removed by the turf filter.
* **What the engine's detector-miss events really were:** 72–74% of its "detector miss" events on those matches are a
  turf-filtered visible player. The model misses 0–3% of those events. The other 23–27% are not ground-truth players
  (mostly referees).
* **The model itself:** it puts a box (conf > 0.1) on **98.0–99.1% of visible players** on those three matches.
* **The fix is CPU-only.** Turning the turf filter off and replaying the same cached detections through the same ByteTrack
  raises tracking HOTA by:
  * 0.371 → 0.423 on 117093 and 0.371 → 0.409 on 118575 (tune);
  * **0.384 → 0.420 on the check match 118576**;
  * 117092 is unchanged (+0.002).

  IDF1 rises by 0.046–0.073, and ID switches go *down*. Results: §5.
* **117092 is the exception: a genuine model weakness.** It is dim, low-contrast, heavily compressed footage. There,
  50% of visible losses are model misses (`no_detection`), mostly small players (< 35 px: tracker-input recall 0.593).
  No post-processing change fixes that.
* **Tile-merge NMS** can explain at most a small share of *visible* losses (2–13%). It matters for heavily occluded
  players (it is the main loss stage for the `heavy` class). Confirming and fixing it needs a raw per-tile capture (GPU,
  approval required, §6).

**Recommendation: keep YOLO26-L.**
* Fix the post-processing now: drop the turf filter, or keep it only for low-confidence boxes.
* Run the NMS A/B on a short GPU capture.
* Treat low-light footage (117092) as the one open model problem, for a later fine-tuning or test-time-scaling
  experiment.

Replacing the detector is **not justified** by this evidence.

## 1. Data and method

* **Inputs (read-only, identical for every variant):**
  * production detection caches: YOLO26-L `player_detector.pt` (sha `05ee5d6c…`), 1280×1280 tiles, 20% overlap,
    imgsz 1920, edge-aware merge at NMS IoU 0.5, cache floor conf 0.1;
  * the production ByteTrack replays (input conf > 0.4 and turf filter passed; activation 0.5, buffer 75, match 0.95);
  * SoccerTrack MOT ground truth (22 players; no referees).
* **Not available:** raw per-tile boxes before merging. The caches store only the post-merge output, so the "raw" stage
  cannot be reconstructed on CPU. The comparison panels show an explicit "not captured" panel instead of inventing one.
* **Stage attribution** (`stages.py`):
  * every GT player in every frame goes to the first pipeline stage that still has a matching box (Hungarian matching,
    IoU ≥ 0.5);
  * the stages are `tracked`, `detected_untracked`, `below_threshold`, `turf_filtered`, `poor_overlap`, `merge_suspect`
    and `no_detection` (definitions in the guide).
  * `merge_suspect` is an **upper bound** for NMS losses. Greedy NMS can only remove a box that overlaps a surviving box
    by IoU > 0.5 (or is contained in it, the edge rule), so a GT player with no surviving neighbour cannot have been
    removed by the merge.
* **Visibility** (`visibility()`), from GT geometry:
  * share of the box covered by nearer GT players: visible < 10%, partial < 50%, heavy < 90%, hidden ≥ 90%;
  * off-pitch: foot more than 0.5 box heights outside the annotated pitch;
  * uncertain: the box touches the image border (none occurred).
  * Referees are absent from GT, so occlusion by referees is not counted.
  * **Hidden players are reported separately and are not counted as detector failures.**
* **Precision side:** every tracker-input box is TP, duplicate, poor overlap, off-pitch, referee label, or other
  on-pitch non-GT. Spot check: all 18 sampled high-confidence on-pitch non-GT boxes were referees.
* **Detection AP50:** COCO-style greedy matching over every cached person box. It is truncated at the cache floor 0.1.
* **Engine cross-reference:** every predicted box of the Step 2 occlusion engine (v2 run) is matched to GT (IoU ≥ 0.3),
  and each event gets the dominant stage of its GT player.
* **Tracking replay** (`replay.py`):
  * Session A's `MultiObjectTracker` (ByteTrack) is fed the identical cached detections under different tracker-input
    filters, and the result is scored with TrackEval (HOTA / CLEAR / Identity) against the full-match GT.
  * Variant `B0` reproduces the production replay **exactly** (all frames, same IDs and boxes: checked).
* **Split:** 117093 and 118575 are tune; 117092 and 118576 are check, with no threshold chosen on them. All four are
  development data. The reserved sequences were not accessed.

## 2. Where visible players are lost

| match | visible GT boxes | lost before tracker | detected, untracked | below threshold | **turf filter** | poor overlap | merge suspect | **no detection (model)** |
|---|---|---|---|---|---|---|---|---|
| 117093 | 113,866 | 9,492 (8.3%) | 210 (2%) | 743 (8%) | **7,465 (79%)** | 164 (2%) | 480 (5%) | 430 (5%) |
| 118575 | 118,986 | 17,423 (14.6%) | 759 (4%) | 2,239 (13%) | **12,061 (69%)** | 434 (2%) | 450 (3%) | 1,480 (8%) |
| 117092 | 116,396 | 23,620 (20.3%) | 1,787 (8%) | 7,405 (31%) | 944 (4%) | 1,347 (6%) | 395 (2%) | **11,742 (50%)** |
| 118576 | 109,678 | 7,565 (6.9%) | 224 (3%) | 620 (8%) | **5,184 (69%)** | 256 (3%) | 988 (13%) | 293 (4%) |

### What the engine's "detector miss" events really were (v2 full-match runs)

| match | events | turf filter | model miss | below threshold | merge suspect | other | not a GT player |
|---|---|---|---|---|---|---|---|
| 117093 | 1,574 | 1,138 (72%) | 6 (0%) | 0 | 0 | 2 | 428 (27%) |
| 118575 | 1,505 | 1,108 (74%) | 48 (3%) | 1 | 0 | 5 | 343 (23%) |
| 117092 | 915 | 204 (22%) | 388 (42%) | 12 (1%) | 1 | 11 | 299 (33%) |
| 118576 | 1,116 | 826 (74%) | 11 (1%) | 0 | 0 | 2 | 277 (25%) |

* **Engine "tracker gap" events** are mostly non-GT people (referees and staff). Otherwise they are visible players
  detected only at 0.1–0.4 confidence (83 / 327 / 1,084 / 64 events).
* **Engine "probable occlusion" events:** the largest single group is `tracked_other_id`. The "hidden" player is in fact
  visible and tracked under a *new* track ID (an ID switch), so the engine predicts a hidden player who is not hidden.
* **Turf-filtered boxes are confident:** median conf 0.87 (10th percentile 0.62–0.84).

## 3. Visual evidence (`data/processed/video/<source>/detection/diag-v1/examples/`)

Each PNG shows GT | raw (not captured) | post-merge cache | tracker output | engine prediction for one zoomed region.

| file | what it shows |
|---|---|
| `soccertrack-117093/…/turf_filtered-f01585-gt6.png` | A goalkeeper in a lime/yellow-green kit, detected at **0.78**, removed by the turf filter for **231 consecutive frames**: the tracker never sees him and the engine reports a "detector miss". |
| `soccertrack-118575/…/turf_filtered-f03600-gt19.png` | A thin player in a beige kit, detected at **0.84**, removed for 183 frames (most of a narrow box is grass). |
| `soccertrack-118576/…/turf_filtered-f02790-gt14.png`, `…f03908-gt7.png` | The same failure on the check match. |
| `soccertrack-117092/…/no_detection-f04382-gt16.png`, `…f04772-gt14.png` | Dim, noisy, compressed footage: clearly visible players get **no box at all** for ~100 frames. A genuine model miss. |
| `soccertrack-117093/…/no_detection-f01130-gt4.png` | A small (17 px) crouching player by the goal: a model miss on an unusual pose. |
| `soccertrack-118575/…/no_detection-f04397-gt20.png` | A goalkeeper lying on the ground: a model miss. |
| `soccertrack-118575/…/below_threshold-f00428-gt22.png` | A player against the cluttered stand background, conf 0.14: below the threshold. |
| `soccertrack-118576/…/merge_suspect-f04979-gt10.png` | A goal-mouth crowd: a heavily covered player has no box while overlapping neighbours do. A possible merge suppression; unconfirmed without the raw capture. |

High-confidence on-pitch non-GT boxes removed by the turf filter are referees in dark kits. This was checked on 18
sampled crops; none was grass. Montage: `data/processed/video/_detection/turf_rejected_non_gt_high_conf.png` (the script that made it is beside it).

**Why the turf filter fails.** `_looks_like_player` rejects a box when > 72% of its upper 65% is green (OpenCV H 35–85,
S ≥ 40). That hue band includes lime and yellow-green kits. A thin, small player whose ~20 px box is mostly grass also
exceeds 72%.

## 4. Recall, size, precision, confidence

**Recall by visibility** (stage reached; *any box* = a cached box with conf > 0.1 at any filter decision):

| match | visibility | GT boxes | tracker output | tracker input | any box |
|---|---|---|---|---|---|
| 117093 | visible / partial / heavy / hidden | 113,866 / 10,866 / 5,042 / 1,649 | 0.917 / 0.907 / 0.459 / 0.140 | 0.918 / 0.917 / 0.491 / 0.155 | 0.991 / 0.952 / 0.592 / 0.220 |
| 118575 | 〃 | 118,986 / 9,365 / 3,211 / 435 | 0.854 / 0.766 / 0.378 / 0.209 | 0.860 / 0.786 / 0.415 / 0.218 | 0.980 / 0.921 / 0.524 / 0.255 |
| 117092 | 〃 | 116,396 / 11,055 / 4,186 / 350 | 0.797 / 0.609 / 0.327 / 0.077 | 0.812 / 0.643 / 0.358 / 0.091 | 0.884 / 0.758 / 0.476 / 0.123 |
| 118576 | 〃 | 109,678 / 14,903 / 6,008 / 764 | 0.931 / 0.895 / 0.497 / 0.174 | 0.933 / 0.906 / 0.528 / 0.192 | 0.986 / 0.950 / 0.633 / 0.226 |

**Tracker-input recall of visible players by GT box height:**

| match | < 35 px | 35–50 px | 50–80 px | ≥ 80 px |
|---|---|---|---|---|
| 117093 | 0.909 | 0.932 | 0.895 | 0.943 |
| 118575 | 0.829 | 0.849 | 0.891 | 0.892 |
| 117092 | **0.593** | 0.847 | 0.960 | 0.918 |
| 118576 | 0.884 | 0.930 | 0.949 | 0.924 |

The 50–80 px dip on 117093 is the lime-kit goalkeeper (turf filter), not a size effect.

**Tracker-input precision split** (conf > 0.4, turf filter as in production; GT has no referees):

| match | AP50 | boxes | TP | duplicate | poor overlap | off pitch | referee label | other non-GT on pitch | TP share |
|---|---|---|---|---|---|---|---|---|---|
| 117093 | 0.871 | 136,550 | 117,320 | 202 | 680 | 4,306 | 602 | 13,440 | 0.859 |
| 118575 | 0.823 | 125,194 | 111,112 | 195 | 997 | 943 | 2,411 | 9,536 | 0.888 |
| 117092 | 0.837 | 110,983 | 103,197 | 786 | 1,627 | 2,773 | 1,038 | 1,562 | 0.930 |
| 118576 | 0.902 | 132,629 | 119,226 | 194 | 939 | 1,561 | 4,503 | 6,206 | 0.899 |

* **Duplicates are rare:** 0.16–0.76 per 100 TP.
* **"Other non-GT on pitch"** is dominated by referees and linesmen labelled `player`. They are people, but outside
  SoccerTrack's GT.

**Confidence threshold** (tracker-input recall, turf filter as in production):

| match | 0.1 | 0.2 | 0.3 | **0.4 (prod)** | 0.5 |
|---|---|---|---|---|---|
| 117093 | 0.905 | 0.901 | 0.897 | 0.892 | 0.885 |
| 118575 | 0.865 | 0.858 | 0.851 | 0.842 | 0.829 |
| 117092 | 0.851 | 0.826 | 0.804 | 0.782 | 0.755 |
| 118576 | 0.920 | 0.916 | 0.912 | 0.907 | 0.899 |

Going from 0.4 to 0.3 adds only 0.5–2.2 recall points (most on 117092) and 30–150% more off-pitch boxes.

## 5. Tracker-input A/B (CPU replay; identical detections, identical ByteTrack)

`B0` reproduced the production replay on **all frames of all four matches** (5,976 / 6,000 / 6,000 / 5,975 frames, 0
differing). Metrics are TrackEval against the full-match GT (22 players). Referees count as FP in every variant.

| match | role | variant | HOTA | DetRe | DetPr | AssA | IDF1 | IDSW | track IDs |
|---|---|---|---|---|---|---|---|---|---|
| 117093 | tune | **B0** production (turf filter, conf > 0.4) | 0.371 | 0.763 | 0.753 | 0.210 | 0.377 | 373 | 262 |
| 117093 | tune | C3 (turf filter, conf > 0.3) | 0.377 | 0.765 | 0.748 | 0.218 | 0.390 | 366 | 255 |
| 117093 | tune | **T0** (turf filter off, conf > 0.4) | **0.423** | 0.815 | 0.739 | 0.264 | 0.435 | 348 | 210 |
| 117093 | tune | TG6 (turf filter only for conf ≤ 0.6) | 0.423 | 0.815 | 0.740 | 0.263 | 0.434 | 348 | 209 |
| 117093 | tune | T0C3 (turf off, conf > 0.3) | 0.423 | 0.817 | 0.735 | 0.265 | 0.441 | 342 | 204 |
| 117093 | tune | T0P (turf off + play-area filter) | 0.428 | 0.815 | 0.757 | 0.264 | 0.440 | 348 | 189 |
| 118575 | tune | **B0** | 0.371 | 0.698 | 0.750 | 0.223 | 0.345 | 383 | 253 |
| 118575 | tune | C3 | 0.371 | 0.703 | 0.748 | 0.223 | 0.344 | 384 | 251 |
| 118575 | tune | **T0** | **0.409** | 0.783 | 0.748 | 0.247 | 0.391 | 266 | 154 |
| 118575 | tune | TG6 | 0.409 | 0.783 | 0.748 | 0.247 | 0.391 | 266 | 152 |
| 118575 | tune | T0C3 | 0.411 | 0.788 | 0.745 | 0.249 | 0.390 | 270 | 152 |
| 118575 | tune | T0P | 0.410 | 0.783 | 0.750 | 0.247 | 0.392 | 266 | 143 |
| 117092 | check | **B0** | 0.305 | 0.653 | 0.820 | 0.153 | 0.280 | 509 | 379 |
| 117092 | check | C3 | 0.309 | 0.663 | 0.816 | 0.155 | 0.282 | 514 | 376 |
| 117092 | check | **T0** | 0.307 | 0.660 | 0.816 | 0.154 | 0.281 | 509 | 370 |
| 117092 | check | TG6 | 0.307 | 0.659 | 0.818 | 0.154 | 0.280 | 509 | 376 |
| 117092 | check | T0C3 | 0.310 | 0.669 | 0.812 | 0.155 | 0.283 | 514 | 367 |
| 117092 | check | T0P | 0.308 | 0.660 | 0.825 | 0.154 | 0.282 | 509 | 343 |
| 118576 | check | **B0** | 0.384 | 0.773 | 0.781 | 0.215 | 0.373 | 387 | 182 |
| 118576 | check | C3 | 0.384 | 0.775 | 0.777 | 0.215 | 0.365 | 386 | 180 |
| 118576 | check | **T0** | **0.420** | 0.808 | 0.758 | 0.253 | 0.445 | 350 | 133 |
| 118576 | check | TG6 | 0.420 | 0.808 | 0.759 | 0.253 | 0.446 | 350 | 132 |
| 118576 | check | T0C3 | 0.418 | 0.811 | 0.755 | 0.252 | 0.438 | 351 | 131 |
| 118576 | check | T0P | 0.418 | 0.808 | 0.760 | 0.251 | 0.439 | 351 | 131 |

(The full table, with on-pitch-only scoring and T0C3P / TG6C3: `python -m football_fatigue.video.detection report`.)

**Reading.**
* **Turf filter off is the one change that matters:**
  * HOTA +0.053 / +0.039 on the tune matches and **+0.035 on the check match 118576**;
  * DetRe +3.5 to +8.5 points;
  * IDF1 +0.046 to +0.073;
  * ID switches −25 to −117;
  * 21–39% fewer track IDs: tracks no longer fragment when a player passes in and out of the filter.
* **The cost:** DetPr falls 0.002–0.023, because referees now get tracked (they are not in GT).
* **117092 is unchanged:** +0.002, because its losses are model misses.
* **`TG6` gives the same result as `T0`:** keeping the filter for conf ≤ 0.6 neither helps nor hurts.
* **Lowering the threshold to 0.3** adds ≤ 0.006 HOTA and slightly more ID switches.
* **The play-area filter** adds ≤ 0.005 HOTA, and it needs SoccerTrack's annotated pitch keypoints, so it does not
  transfer to uploads.
* **Recommended tracker input: `T0`** (or `TG6` if a low-confidence safety net is wanted), conf > 0.4.
* **Not applied yet:** this changes the production post-processing, which needs your approval. The Session A detector
  code is untouched.

## 6. Tile-merge NMS A/B: implemented, waiting for a raw capture

* **Upper bound from the cache:** `merge_suspect` covers 2–13% of visible losses. Merge-suspect players are mostly
  **heavy/hidden**: 1,097–1,736 heavy and 262–1,171 hidden GT-frames per match. So a merge change can matter for
  occlusion analysis, but it cannot fix the visible-player complaint.
* **Policies** (`merge.py`, unit-tested):
  * `A0`: the production edge-aware merge, computed by Session A's own `_merge_person_agnostic`, so it is identical by
    construction;
  * `A1`: cross-tile duplicate suppression only;
  * `A2`: overlap-region-aware: a pair is suppressed only if both boxes lie in the overlap of their two tiles;
  * `A3`: global NMS IoU 0.7.
* **Scoring** (`ab`): recall/precision, duplicates per 100 TP, and recall by visibility on the same GT. `A0` is also
  checked frame by frame against the cached boxes.
* **Missing input:** a raw per-tile capture (`capture --approved`). Request: see §8.

## 7. Alternative detectors: shortlist (not benchmarked; no downloads made)

| candidate | why | licence | concerns |
|---|---|---|---|
| **YOLO26-L (current) + fixed post-processing** | 98–99% of visible players already get a box on 3 of 4 matches | AGPL-3.0 (Ultralytics) | low light (117092), small players |
| YOLO26-L with test-time upscaling (smaller tiles / larger imgsz) on dark or small regions | targets the 117092 failure without a new model; SoccerNet 2026 winners used high-resolution / two-pass small-athlete detection | same | 2–4× GPU time |
| RF-DETR Medium/Large (`rfdetr`) | modern DETR, Apache-2.0 for N–L detection weights | Apache-2.0 (N–L); XL/2XL under Roboflow PML | COCO-pretrained "person" only; inputs of 576–704 px need tiling; no published small-object AP; a fair comparison needs fine-tuning on football (training not allowed in this milestone) |
| Football fine-tuned community checkpoints (for example RF-DETR fine-tuned on SoccerNet on Hugging Face; Roboflow Universe football datasets) | football-specific | varies: often CC BY 4.0 data; weights' licence unclear | unverified cards (one reports 85.7 mAP50 on test but 49.8 on validation); broadcast-view data differs from SoccerTrack's panorama |

A zero-shot COCO-pretrained model compared with a SoccerTrack-fine-tuned YOLO26-L would not be a fair test. No
candidate is recommended for download until the post-processing fix has been measured, and then only for the 117092
low-light failure.

## 8. Requests for approval (nothing run yet)

1. **Raw per-tile capture for the NMS A/B (GPU).**
   * 4 dev matches × 2 windows × 125 frames = 1,000 frames.
   * Windows: on the tune matches, the segments with the most `merge_suspect` frames; on the check matches, fixed
     windows chosen before looking.
   * Production speed is 1.77 s/frame on the GTX 1650 Ti, so about **30 min GPU** in total.
   * Under 3 GB VRAM (production settings, batch 2) and about 50 MB of disk.
   * Then the CPU A/B (`ab`): minutes.
2. **Low-light test-time scaling pilot on 117092 (GPU).**
   * Same weights; 640-px tiles at imgsz 1280 (about 4× the tiles) on 2 × 125 frames.
   * About **30 min GPU**.
   * No download.
3. **Optional, after 1–2: RF-DETR-M zero-shot sanity check on the same 500 frames.**
   * About a 130 MB download (Apache-2.0); about 20 min GPU at 576-px tiles.
   * Only worth doing if 1–2 leave a large gap.

## 9. Limitations

* There are no raw per-tile boxes, so the attribution between model and merge for missing boxes is bounded, not
  measured.
* GT has 22 players: referees count as non-GT. HOTA penalises any variant that tracks more referees, so the turf-off
  precision loss is partly people correctly detected.
* Visibility comes from GT boxes (template-like), not pixels.
* `poor_overlap` collects both annotation offsets and genuinely bad boxes.
* The play-area filter uses SoccerTrack's annotated pitch keypoints. Arbitrary uploads would need a drawn outline.
* The check matches are development data, not independent validation.

## 10. Reproduce

See `docs/video_detection_diagnostics.md`. Run times on this PC (CPU):
* `diagnose`: 29–34 s per match;
* `examples`: about 1 min per match;
* `replay`: about 2–3 min per variant per match, most of it TrackEval scoring.
