# Session B — detector & tracker comparison (2026-11-20)

Research track only. **The production detector, B0 and the historical T0 are unchanged.** Development matches only:
117093 and 118575 (tune), 117092 and 118576 (check). No reserved or FIFA data was read. No GPU inference, downloads,
installs or training. Generated tables: `data/processed/video/_bench/report-v1.md` (`python -m football_fatigue.video.bench report`).

## 1. What was run

| item | what | where |
|---|---|---|
| Frame manifest | `frames-v1`: 92 frames (22 / 22 / 24 / 24 per match), 11 strata, seed 20261120, GT used only to stratify, sha `fd55d7dd8f86…` | `_bench/frames/frames-v1.json` |
| Detector runs | 9 runs: production cache, plus 8 CPU runs (all with edge-aware tile merge, conf floor 0.1, 8 threads) | `_bench/detector-v1/runs/*` |
| Detector eval | AP50, AP50:95, AP75, P/R, FP/frame, duplicates, recall by size / visibility / stratum, latency; three views | `_bench/detector-v1/eval/` |
| Paired bootstrap | 500 frame resamples per comparison | `_bench/detector-v1/eval/bootstrap-v1.json.txt` |
| Trackers | 7 trackers × 4 full matches on identical T0 detections, plus 5 trackers on T0L (conf 0.1) | `_bench/tracker-v1/` |
| Occlusion engine | engine v2 on T0-BOT and T0-OCS track profiles, all 4 matches, evaluated with eval-v1 | `<match>/occlusion/0.2.0/…-v2-T0-BOT-*`, `…-T0-OCS-*` |
| Gallery | 26 items (PNG + 3 MP4), seeded selection rule, alternative detector = YOLOv8m | `_bench/gallery-v1/index.json` |

**Ground-truth limitation (applies to every detector number).** SoccerTrack GT annotates the 22 players only.
Referees, coaches, staff and spectators are not annotated, so a correct detection of them counts as a false positive.
The `on_pitch` view, used as the headline, drops boxes whose feet are more than 0.5 box-heights outside the annotated
play area. A referee on the pitch still counts as FP for every model. `no_referee` additionally drops boxes the
football model labelled `referee`; COCO models have no such class. Visibility classes are derived from GT box
overlap, not from visibility labels.

**Reproducibility checks that ran:**
- YOLO26-L on CPU at the production settings (img 1920, tiles 1280, overlap 0.2, edge-aware merge) reproduced the stored GPU cache exactly on 6 frames: same box counts, max abs difference 0.0.
- The benchmark sv-bytetrack harness reproduces the historical T0 tracks exactly (test, 300 frames of 118576).

## 2. Detectors (92 frames, on_pitch view, conf > 0.4 unless stated)

| run | AP50 tune / check / all | AP50:95 all | P@0.4 | R@0.4 | FP/frame | R small (<35 px) | R heavy occl. | s/frame (CPU) |
|---|---|---|---|---|---|---|---|---|
| **YOLO26-L football, production (img 1920)** | 0.888 / 0.851 / **0.866** | 0.572 | 0.897 | 0.865 | 2.19 | 0.626 | 0.429 | (GPU cache) |
| YOLO26-L football, img 1280 | 0.862 / 0.857 / 0.855 | 0.569 | 0.867 | 0.879 | 2.96 | 0.673 | 0.438 | 9.7 |
| YOLOv8m COCO, img 1280 | 0.812 / 0.810 / 0.807 | **0.591** | 0.749 | 0.868 | 6.39 | 0.686 | 0.324 | 6.4 |
| YOLOv8s COCO, img 1280 | 0.790 / 0.778 / 0.780 | 0.533 | 0.808 | 0.820 | 4.28 | 0.593 | 0.286 | 3.4 |
| YOLOv8n COCO, img 1280 | 0.763 / 0.750 / 0.752 | 0.475 | 0.838 | 0.750 | 3.19 | 0.482 | 0.286 | 1.8 |
| YOLOv8m COCO, img 640 | 0.742 / 0.748 / 0.744 | 0.456 | 0.869 | 0.694 | 2.30 | 0.357 | 0.267 | 2.7 |
| YOLOv8s COCO, img 640 | 0.653 / 0.668 / 0.659 | 0.366 | 0.847 | 0.577 | 2.29 | 0.166 | 0.171 | 1.1 |
| YOLOv8n COCO, img 640 | 0.560 / 0.599 / 0.579 | 0.277 | 0.829 | 0.469 | 2.13 | 0.113 | 0.095 | 0.8 |
| YOLO26-L football, img 1920 (6-frame equivalence subset only) | 0.903 (n=6) | – | – | – | – | – | – | 16.7 |

### Paired bootstrap (B − A, 95% CI over frames; frames within a match are correlated, so the intervals are optimistic)

| A → B | frames | ΔAP50 | ΔR@0.4 |
|---|---|---|---|
| YOLO26-L prod → YOLOv8m@1280 | 92 | −0.059 [−0.076, −0.041] | +0.003 [−0.012, +0.021] |
| YOLO26-L prod → YOLOv8m@1280, **117092 low light only** | 24 | −0.011 [−0.052, +0.043] | **+0.059 [+0.015, +0.105]** |
| YOLO26-L prod → YOLO26-L@1280 | 92 | −0.011 [−0.023, +0.001] | +0.015 [+0.006, +0.025] |
| YOLO26-L prod → YOLOv8s@1280 | 92 | −0.086 [−0.103, −0.068] | −0.044 [−0.060, −0.029] |
| YOLOv8m@640 → YOLOv8m@1280 | 92 | +0.063 [+0.049, +0.076] | +0.173 [+0.153, +0.194] |

### What the detector results show

1. **YOLO26-L (production) has the highest AP50**, because of its precision: 2.2 FP/frame vs 6.4 for YOLOv8m. Its
   recall is no higher than YOLOv8m@1280. YOLOv8m has a higher AP50:95 (0.591 vs 0.572), so its boxes fit
   the GT more tightly. The COCO models' extra FPs are largely off-GT people inside the play area (referees,
   staff; see `referee_labels` and `alt_failure` in the gallery). Some of this is the GT limitation, not a model error.
2. **Training data vs architecture:** this is not yet separable. The architecture-matched control (`yolo26l-coco`,
   the same base weights the production model was fine-tuned from) needs a download. Until it runs, the
   YOLO26-L vs YOLOv8m gap mixes fine-tuning data, architecture and inference size.
3. **Inference resolution matters most.** Tiling at 1280 instead of 640 raises YOLOv8m recall by 0.17, and small-player
   recall goes from 0.36 to 0.69. Any GPU comparison must use img ≥ 1280 per tile.
4. **Low light (117092) is YOLO26-L's weak spot:** recall 0.725 vs 0.784 for YOLOv8m, with a significant ΔR. It also
   has the weakest recall on small players (<35 px) of the 1280-class runs.
5. **Heavy occlusion:** every model is below 0.45 recall on GT players that are ≥ 50% covered. No detector solves this.
   YOLO26-L leads (0.43–0.44 vs ≤ 0.32 for COCO models).
6. Duplicates are negligible for all runs (≤ 0.2 per 100 TP) with edge-aware merge.

## 3. Detector status (every requested family)

| detector | track | status | reason |
|---|---|---|---|
| YOLO26-L football (production) | B2 | **BENCHMARKED** | GPU cache + CPU runs at 1280 / 1920 |
| YOLOv8n / s / m (COCO) | B1 | **CPU PILOT ONLY** | local weights (SHA verified); CPU at 640 and 1280 |
| YOLOv9m (COCO) | B1 | REQUIRES WEIGHTS/DOWNLOAD APPROVAL | not local |
| YOLOv10m (COCO) | B1 | REQUIRES WEIGHTS/DOWNLOAD APPROVAL | not local |
| YOLO11m (COCO) | B1 | REQUIRES WEIGHTS/DOWNLOAD APPROVAL | not local |
| YOLO12m (COCO) | B1 | REQUIRES WEIGHTS/DOWNLOAD APPROVAL | not local |
| YOLO26m (COCO) | B1 | REQUIRES WEIGHTS/DOWNLOAD APPROVAL | not local |
| YOLO26-L (COCO; architecture-matched control) | B1 | REQUIRES WEIGHTS/DOWNLOAD APPROVAL | not local; key for separating data vs architecture |
| RF-DETR medium | B1 | INCOMPATIBLE (current env) | `rfdetr` package not installed and no adapter: needs an install approval plus weights |
| easychamp YOLOv8 football (HF) | B2 | REQUIRES WEIGHTS/DOWNLOAD APPROVAL | card says MIT, but Ultralytics-trained weights may carry AGPL; training data undocumented |
| soccana YOLO11 football (HF) | B2 | REQUIRES WEIGHTS/DOWNLOAD APPROVAL | licence not stated; SoccerNet broadcast data, different domain |

Params / FLOPs (640, from the Ultralytics docs pages verified this sprint): v8m 25.9M / 78.9B; v10m 15.4M / 59.1B; 11m 20.1M / 68.0B;
12m 20.2M / 67.5B; 26m 20.4M / 68.2B; v9m 20.0M / 76.3B (not re-verified).

## 4. Trackers (identical T0 detections, full matches, track buffer 75, no ReID, no camera-motion compensation)

Mean over the two matches of each role (from `report-v1.md`):

| tracker | HOTA tune | HOTA check | AssA check | IDF1 check | IDSW check (sum) | ms/frame |
|---|---|---|---|---|---|---|
| **sv-bytetrack (production = T0)** | 0.4164 | **0.3632** | **0.2035** | **0.3631** | **859** | 6.7–7.8 |
| ul-botsort | **0.4198** | 0.3416 | 0.1809 | 0.3280 | 1059 | 3.7–6.4 |
| ul-ocsort | 0.4078 | 0.3465 | 0.1865 | 0.3284 | 1210 | 4.6–5.8 |
| ul-deepocsort (no ReID) | 0.4078 | 0.3465 | 0.1865 | 0.3284 | 1210 | 4.7–5.3 |
| ul-bytetrack | 0.4107 | 0.3439 | 0.1834 | 0.3170 | 1157 | 4.0–4.5 |
| ul-tracktrack | 0.4066 | 0.3132 | 0.1538 | 0.2834 | 997 | 3.3–4.3 |
| ul-fasttrack | 0.3763 | 0.3130 | 0.1512 | 0.2872 | 1260 | 2.6–3.3 |

- **No complex tracker beats production ByteTrack after T0.** BoT-SORT is +0.003 HOTA on tune, driven by one match
  (118575: 0.434 vs 0.409), and −0.022 on check. Every alternative has more ID switches on check (and more fragments, except TrackTrack: 4935 vs 5145, at much lower AssA).
  The Ultralytics ByteTrack is itself 0.02 below the supervision ByteTrack, so differences between implementations
  are as large as the differences between algorithms.
- DeepOC-SORT is identical to OC-SORT because ReID is off (no ReID weights; a ReID download needs approval). The
  rows are kept to show that no appearance cue was used.
- **T0L (conf 0.1 into the tracker):** HOTA changes by ≤ 0.015, while Frag drops 32–37% for sv-bytetrack, BoT-SORT and ByteTrack-UL (12–17% for OC-SORT). That is a
  fragmentation gain, not a HOTA gain.
- **DCTracker: UNAVAILABLE.** The paper is closed access (Knowledge-Based Systems 304, 112528, 2024) and no public code
  was found. **GTA (gta-link): REQUIRES DOWNLOAD APPROVAL.** The repo is public (licence not verified). It is an
  offline split-and-connect refinement reporting SoccerNet HOTA 79.4 → 83.1, which is our most promising untested
  option because association, not detection, dominates our error (AssA ≈ 0.15–0.28 vs DetA ≈ 0.61–0.69).
- Memory: all trackers < 6 MB extra RSS, except a ~530 MB first-process import cost on 117093.

## 5. Occlusion engine with other trackers (engine v2, eval-v1, full matches)

| match | profile | probable occl. events | PO precision | PO recall | PO F1 | overlap recall |
|---|---|---|---|---|---|---|
| 117093 (tune) | B0 / T0 / T0-BOT / T0-OCS | 493 / 511 / 602 / 616 | 0.706 / 0.699 / 0.678 / 0.665 | 0.648 / 0.652 / 0.675 / 0.661 | 0.676 / 0.674 / 0.676 / 0.663 | 0.359 / 0.362 / 0.416 / 0.417 |
| 118575 (tune) | B0 / T0 / T0-BOT / T0-OCS | 299 / 304 / 406 / 401 | 0.703 / 0.719 / 0.693 / 0.632 | 0.695 / 0.721 / 0.786 / 0.741 | 0.699 / 0.720 / 0.737 / 0.682 | 0.482 / 0.502 / 0.547 / 0.555 |
| 117092 (check) | B0 / T0 / T0-BOT / T0-OCS | 423 / 430 / 694 / 729 | 0.571 / 0.566 / 0.498 / 0.484 | 0.682 / 0.682 / 0.720 / 0.739 | 0.622 / 0.619 / 0.589 / 0.585 | 0.340 / 0.339 / 0.352 / 0.345 |
| 118576 (check) | B0 / T0 / T0-BOT / T0-OCS | 458 / 464 / 569 / 620 | 0.742 / 0.752 / 0.714 / 0.669 | 0.767 / 0.772 / 0.813 / 0.770 | 0.754 / 0.762 / 0.760 / 0.716 | 0.479 / 0.476 / 0.502 / 0.520 |

The alternative trackers raise probable-occlusion recall (+0.02 to +0.06) and lower precision by a similar or larger
amount. F1 is flat or worse on both check matches. **A different tracker does not improve event accuracy.** The engine's
remaining errors (fp_* outcomes, i.e. overlaps of tracks whose GT players were not occluding) come from detection
geometry, not tracker choice. No signal here supports identity resolution, and none was attempted.

## 6. Limitations

- **Hardware:** COCO models ran on CPU only. GPU latency, FPS and VRAM were not measured, so every detector except
  production is "CPU PILOT ONLY".
- **Size:** 92 frames is small. The intervals above treat frames as independent, but frames within a match are
  correlated.
- **Production resolution:** the production setting (img 1920) has only 6 CPU frames, because CPU is too slow
  (16.7 s/frame). The full 92-frame production numbers come from the GPU cache. CPU equivalence was demonstrated on those 6 frames.
- **Tracker hyperparameters:** Ultralytics YAML defaults with buffer 75, so the comparison is untuned. sv-bytetrack
  carries the production parameters.
- **Strata:** camera movement / zoom is not applicable (fixed SoccerTrack panorama). The edge stratum has 5 frames
  (118576 only).
