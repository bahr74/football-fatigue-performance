# Session B: staged multi-detector plan and GTA feasibility, 2026-11-21 (PROPOSED, nothing run)

This document records no downloads, no GPU runs and no installs. Candidate metadata comes from `video/bench/registry.py` (`python -m football_fatigue.video.bench registry`). Vendor figures are marked as such there.

## 1. What fair detector testing requires (lessons from pipeline-v1)

- **Same pipeline.** Use the same tiling (1280 tiles, 20 % overlap), the same merge (IoU 0.5), the same class mapping, the same conf grid and the same frames for every model. Re-score the existing caches with `bench evaluate`.
- **Report errors separately.** Detector error must be reported separately from the turf filter, the confidence threshold and the tracker, because those downstream stages cost about as much as the detector (§2.4 of the decomposition report).
- **Stratify the dev frame sample** by the strata where YOLO26-L fails: low light (117092), boxes under 35 px, and GT-heavy overlap. The 92-frame sample used on 2026-11-20 is too small for the heavy-overlap stratum. Stage 3 below enlarges it.
- **Compare across tracks with care.** Compare COCO models with each other (track B1), and football fine-tunes against `yolo26l-coco`, which is the architecture-matched control (track B2). Never compare a COCO model against the football fine-tune and attribute the gap to the architecture.
- **Do not cross-product.** Run detectors first, then track only the shortlist.

## 2. Stages

| stage | what | models | data | compute | approval |
|---|---|---|---|---|---|
| S1 | comparable sizes, COCO | yolov8m (local), yolov9m, yolov10m, yolo11m, yolo12m, yolo26m | existing 92-frame dev sample | CPU about 1–2 h per model, or GPU about 5 min per model | **A1** (weights), **A2** (if GPU) |
| S2 | architecture-matched control | yolo26l-coco vs yolo26l-football (production) | 92 frames, then S3 sample | as S1 | **A1**, **A2** |
| S3 | larger stratified dev sample | the 2–3 best from S1/S2 + production | about 400 frames: 100 per match, ≥ 30 % low-light / < 35 px / heavy-overlap frames | sampling is CPU and needs no approval; inference needs **A2** | A2 |
| S4 | football checkpoints | easychamp YOLOv8, soccana YOLO11 (Hugging Face); optional RF-DETR medium | S3 sample | as S1 | **A4** (licence check first), **A3** (RF-DETR install + weights) |
| S5 | full-sequence tracking, shortlist ≤ 2 | full dev matches → T0 sv-bytetrack + engine v2 | 4 dev matches, about 6,000 frames each | GPU about 3 h per model per match at production speed (1.77 s/frame) | **A2** |

Commands, using the existing tooling:

```bash
python -m football_fatigue.video.bench detect --data-root F:/Football-Fatigue-Project/data --detector <key> --protocol tile1280-img1280 --device cpu|0
python -m football_fatigue.video.bench evaluate --data-root F:/Football-Fatigue-Project/data
python -m football_fatigue.video.bench track --data-root ... --source <sid> --tracker sv-bytetrack --input T0
```

| key | source | licence (registry) | params (M) | note |
|---|---|---|---|---|
| yolov9m-coco | Ultralytics assets v8.4.0 | GPL-3.0 upstream, AGPL-3.0 port | 20.0 (vendor) | |
| yolov10m-coco | Ultralytics assets v8.4.0 | AGPL-3.0 | 15.4 (vendor) | NMS-free |
| yolo11m-coco | Ultralytics assets v8.4.0 | AGPL-3.0 | 20.1 (vendor) | |
| yolo12m-coco | Ultralytics assets v8.4.0 | AGPL-3.0 | 20.2 (vendor) | higher memory |
| yolo26m-coco | Ultralytics assets v8.4.0 | AGPL-3.0 | 20.4 (vendor) | NMS-free |
| yolo26l-coco | Ultralytics assets v8.4.0 | AGPL-3.0 | measure after download | S2 control |
| rfdetr-medium-coco | github.com/roboflow/rf-detr | Apache-2.0 (N–L) | measure | needs `pip install rfdetr` |
| easychamp YOLOv8 | huggingface.co/aabyzov/easychamp-player-detection-yolov8 | card: MIT; AGPL question open | – | training data undocumented |
| soccana YOLO11 | huggingface.co/Adit-jain/soccana | not stated | – | broadcast-domain data |

All weight URLs use the pattern `https://github.com/ultralytics/assets/releases/download/v8.4.0/<name>.pt`. On download, the registry records the sha256, size and measured parameter count.

VRAM is unmeasured for every model on this GPU (GTX 1650 Ti, 4 GB). Only YOLO26-L production was observed, at under 3 GB. Each GPU run must record peak VRAM.

## 3. Tracker association and GTA

**Association failures measured (T0, pipeline-v1):**
- 1,890 temporary-ID changes over 4 matches. 72 % of them come after a gap, usually a detector gap.
- 733 of these changes are *GT-verified swaps*, where the previous ID continued on another player.
- Untracked-despite-detection is small: 0.8 %. Most of it is boxes ≤ 0.5 that ByteTrack cannot start a track from.

So the identity problem is mostly **re-association after short detector gaps in overlap**. That is the case offline split-and-connect refinement targets.

**GTA (gta-link), feasibility (repo inspected via web fetch, not cloned):**

| item | finding |
|---|---|
| repo | github.com/sjc042/gta-link (paper arXiv 2411.08216, ACCV 2024 workshop) |
| licence | MIT (Copyright 2024 Jiacheng Sun) |
| ReID | required: OSNet checkpoint `reid_checkpoints/sports_model.pth.tar-60` (SportsMOT-trained). The README gives no download URL, and whether the file is committed is **unconfirmed** |
| dependencies | `torchreid` 1.4.0, torch, scikit-learn, lap, loguru, opencv-python, gdown. Python 3.8 is recommended, so a **separate venv** is likely needed |
| input | MOT txt per sequence (`pred_dir/seqN.txt`) plus video frames |
| steps | `generate_tracklets.py --model_path … --data_path … --pred_dir … --tracker …` → `refine_tracklets.py --use_split --min_len 100 --eps 0.6 --min_samples 10 --max_k 3 --use_connect --spatial_factor 1.0 --merge_dist_thres 0.4` |
| reported gain | SoccerNet HOTA 79.4 → 83.1 (paper abstract; not re-verified, different camera and domain) |

**Experiment specification (after approval A5):**
1. Export the T0 and T0L sv-bytetrack tracks of the 2 tune matches to MOT txt. This is CPU work and new code under `video/bench`.
2. Run GTA with default parameters, then a small grid on tune only (`merge_dist_thres` 0.3 / 0.4 / 0.5, `min_len` 50 / 100).
3. Freeze the configuration, then run once on check.
4. Score everything with `detection.replay.score` (HOTA, AssA, IDF1, IDSW, Frag) and pipeline-v1 ID changes, against the B0 / T0 / T0L sv-bytetrack baselines.
5. Re-run engine v2 + eval-v1 on the refined tracks.
6. Promote only if check AssA and IDF1 improve without an increase in GT-verified swaps.

GTA is **not** assumed better before this benchmark.

## 4. Approvals requested (none granted yet)

| id | what | size / cost |
|---|---|---|
| A1 | download COCO weights: yolov9m, yolov10m, yolo11m, yolo12m, yolo26m, yolo26l from Ultralytics assets v8.4.0 | about 6 files of roughly 40–100 MB |
| A2 | GPU inference: S1/S2 on 92 frames (about 5 min per model); S3 about 400 frames per model; optional S5 full matches (about 3 h per model per match) | GTX 1650 Ti |
| A2-cap | GPU raw per-tile capture for NMS (decomposition report §4) | about 16 min, < 3 GB VRAM |
| A3 | `pip install rfdetr` + RF-DETR medium weights | new dependency |
| A4 | Hugging Face football checkpoints (easychamp, soccana), after the licence check | 2 files |
| A5 | GTA: clone gta-link (MIT), separate venv (torch, torchreid 1.4.0), OSNet sports checkpoint | CPU or GPU, tune matches first |
| A6 | optional: Roboflow `trackers` package | new dependency |
