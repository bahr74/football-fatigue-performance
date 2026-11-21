# Session B M13: staged multi-detector benchmark (2026-11-21)

**Scope of this milestone:** audit, frame set, comparability rules, one new CPU analysis on *existing* runs,
resource table. No weights downloaded, no GPU used, no new detector inference run.

## 1. Local checkpoint audit (verified on disk)

| key | file | bytes | sha256 (registry) | architecture / head | classes | licence | benchmarked (frames-v1, 92 frames) |
|---|---|---|---|---|---|---|---|
| yolo26l-football (production) | `F:/Football-Fatigue-Project/models/vision/player_detector.pt` | 53,140,901 | 05ee5d6c…087f | YOLO26-L, **end2end = False** → per-tile class-aware NMS (IoU 0.7) | ball, player, referee, goalkeeper | AGPL-3.0 (checkpoint metadata) | cache (GPU, imgsz 1920), CPU 1280, CPU 1920 (6-frame subset) |
| yolov8m-coco | `models/yolov8m.pt` | 52,136,884 | 5d4a90cd…fe5 | YOLOv8m, NMS | COCO 80 (person = 0) | AGPL-3.0 | CPU 1280, CPU 640 |
| yolov8s-coco | `models/yolov8s.pt` | 22,588,772 | 1f47a78b…a78a | YOLOv8s, NMS | COCO 80 | AGPL-3.0 | CPU 1280, CPU 640 |
| yolov8n-coco | `models/yolov8n.pt` | 6,549,796 | f59b3d83…6b36 | YOLOv8n, NMS | COCO 80 | AGPL-3.0 | CPU 1280, CPU 640 |

No other `.pt/.pth/.onnx/.engine` file exists under `F:/Football-Fatigue-Project/models`. Everything else in the
registry (YOLOv9/10/11/12/26-m COCO, yolo26l-coco, RF-DETR, Hugging Face football models) **needs approval**.

## 2. New CPU analysis on existing runs: who finds the production misses? (`_bench/complement-m13-v1.json`)

On frames-v1 (92 frames, stratified towards hard cases, so not match-representative), GT players matched greedily at
IoU ≥ 0.5 with person boxes of conf > 0.1. "Base" = production cache (YOLO26-L, imgsz 1920).

| run (inference size) | base misses | of them found by run | run misses | of them found by base |
|---|---|---|---|---|
| yolov8m-coco (1280) | 195 | **119 (61 %)** | 123 | 47 |
| yolov8s-coco (1280) | 195 | 102 (52 %) | 179 | 86 |
| yolov8n-coco (1280) | 195 | 91 (47 %) | 237 | 133 |
| yolo26l-football CPU (1280) | 195 | 59 (30 %) | 170 | 34 |
| yolov8m-coco (640) | 195 | 84 (43 %) | 292 | 181 |

By stratum, YOLOv8m @1280 finds:

| stratum | base misses found by YOLOv8m |
|---|---|
| low light (117092) | 73 / 96 |
| small (< 35 px) | 68 / 101 |
| heavy overlap | 15 / 51 |
| hidden | 8 / 21 |

Readings (hypotheses for the benchmark, not conclusions):
- **Model-family diversity helps where production is weak:** low light and small players.
- **Heavy overlap stays hard for every local model.** Under 30 % of base misses are recovered there.
- **A large share of production misses are marginal.** The *same* checkpoint at a different inference size (1280 on CPU vs 1920 on GPU) recovers 30 % of them.
- **These counts are at conf > 0.1.** YOLOv8m's precision is much lower: 6.4 FP/frame at 0.4 vs 2.2 for production (`report-v1.md`).

## 3. Fixed frame sets

| set | frames | built from | use |
|---|---|---|---|
| frames-v1 (existing, sha fd55d7dd…) | 92 (44 tune, 48 check) | GT + pitch landmarks only, seed 20261120 | all completed runs; unchanged |
| **frames-v2 (new)** | 274 (130 tune, 144 check), 24 per hard stratum, 72 uniform, min gap 25 frames, seed 20261121; 2 frames shared with v1 | same GT-only rules (`bench.frames.build(version='frames-v2')`) | Stage A/B detector comparison |
| a2cap-plan-v1 (new) | 480 (16 windows × 30) | saved T0 decomposition (targeted) + seeded random | raw-tile stage attribution (production model only) |

Tune/check separation is kept in every set (per-frame `role`); thresholds are chosen on tune frames only.

## 4. Comparability rules (every stage)

1. **Identical tiling and merge for all models.** Use 1280 tiles, 20 % overlap, `edge_aware` merge IoU 0.5, conf floor 0.1, and the same person-class mapping (COCO person ↔ player/referee/goalkeeper).
2. **One common inference size, 1280, for the cross-model comparison (protocol `tile1280-img1280`).** The production 1920 is reported as a separate row and never mixed into the comparison.
3. **Score raw detection first**, separately from filtering and tracking:
   - AP50 and AP50:95;
   - recall at conf 0.1 and 0.4;
   - FP/frame;
   - recall per stratum and per GT-derived visibility.
4. **Then track** with the identical T0 input rule (conf > 0.4) and the same sv-ByteTrack, on full matches, for at most two shortlisted detectors. Every tracking row names its detector.
5. **Report the referee caveat.** GT has players only, so referees count as FP, equally for all models.

## 5. Stages

| stage | models | frames | compute | approval |
|---|---|---|---|---|
| **A** (local only) | yolov8m-coco @1280 first, then yolo26l-football @1280 on CPU as the same-size control | frames-v2 (274) | CPU measured: v8m 6.44 s/frame → **~30 min**; yolo26l @1280 9.67 s/frame → ~45 min | none (CPU). Needs one small change first: detector run folders keyed by frame-set version (today `runs/<key>` is frames-v1 only and write-once) |
| B | COCO m-size: yolo11m, yolo26m (+ yolov9m, yolov10m, yolo12m if B is cheap) and **yolo26l-coco** as the architecture-matched control for the football fine-tune | frames-v2 | GPU: 274 frames × ~1 s ≈ 5 min per model (unmeasured; production 1.77 s/frame at 1920) | A1 (download ≈ 40-100 MB each) + A2 (GPU) |
| C | football checkpoints (easychamp YOLOv8, soccana YOLO11) after licence check | frames-v2 | as B | A4 + A2 |
| D | RF-DETR medium | frames-v2 | install + GPU | A3 + A2 |
| E | ≤ 2 shortlisted detectors, full dev matches → T0 tracking + engine v2 | 4 × ~6,000 frames | GPU ≈ 3 h per model per match at 1.77 s/frame (production size) | A2 |

The smallest high-information comparison is Stage A plus yolo26l-coco from Stage B. It answers three questions:
- Does a COCO model with a different training distribution complement production in low light and on small players?
- How much of the production gap is due to the football fine-tune rather than the architecture?
- Does any model recover heavy-overlap players? Today the answer is "rarely", and A2-cap explains why.

Detector-level results must exist before any tracker is re-benchmarked on new detections.
