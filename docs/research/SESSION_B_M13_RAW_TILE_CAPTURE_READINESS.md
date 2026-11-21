# Session B M13: A2-cap raw-tile capture, readiness and approval gate (2026-11-21)

**Status: READY. Awaiting your explicit approval of A2-cap. The GPU capture was not run.**

What ran on CPU:
- the window plan;
- unit tests on synthetic data and the real Ultralytics NMS function;
- **one 1-frame CPU readiness test** with the production model. Its output went to a separate `capture-v2-smoke` folder and is never used as evidence.

## 1. What the old plan could not answer and why v2 is needed

The production checkpoint (`models/vision/player_detector.pt`, YOLO26-L fine-tune, sha 05ee5d6c…) was loaded on CPU to inspect it. It runs with **`end2end = False`**. So every tile passes through two suppression stages:

| stage | where | rule |
|---|---|---|
| per-tile NMS | inside the Ultralytics model call | class-aware, IoU 0.7, max_det 300, conf ≥ floor (defaults of Ultralytics 8.4.182) |
| cross-tile merge | `vision.detector._merge_person_agnostic` | `edge_aware`, merge IoU 0.5, cross-role IoU 0.7, edge-truncation rule |

Capture v1 (`detection/capture.py`, never run) saved the per-tile boxes **after** the first stage. It could therefore not separate "never proposed" from "removed by per-tile NMS".

v1 and `merge.py` also assumed YOLO26's one-to-one head de-duplicates inside a tile. That is **not** the production configuration, so the A1 "same tile never suppresses" rationale in `merge.py` needs re-reading after the capture.

**v2** is `video/detection/capture_v2.py`. It hooks `DetectionPredictor.postprocess`, the exact tensor the predictor post-processes, and stores four stages per frame:

1. **raw:** every candidate anchor with max class score ≥ 0.05. Stored per candidate: all 4 class scores, box in full-frame pixels, tile, anchor index.
2. **nms:** the per-tile NMS keep flag. It is recomputed with `return_idxs` using the predictor's own arguments and **checked against the predictor's Results boxes** on every frame.
3. **merge_in:** NMS survivors with conf > 0.1 (the cache floor).
4. **final:** the production merge replayed offline with Session A's own code (`merge.merge('A0')`), then **compared with the cached boxes of the same frame**.

Suppression relationships:
- Per-tile NMS suppressors are exactly recoverable, because greedy NMS removes a box by the highest-scoring kept box of the same class with IoU > 0.7.
- Merge suppressors are recorded approximately, because the edge-truncation rule also applies.

The output is versioned (`capture-v2`) and write-once (`.partial` staging, refusal if the folder exists). The weight sha256 is checked against the cache manifest before loading.

## 2. Readiness test (CPU, 1 frame, `capture-v2-smoke/soccertrack-117093-1599-1`)

| check | result |
|---|---|
| weights present, sha matches cache | yes (53,140,901 bytes) |
| dependencies | ultralytics 8.4.182, torch 2.9.1+cu126, CUDA available (GTX 1650 Ti), supervision 0.30.9 |
| per-tile NMS recomputed == predictor output | **true** |
| replayed merge vs cache (IoU > 0.95) | **26 matched, 0 only in capture, 0 only in cache** |
| frame contents | 4 tiles, 378 raw candidates, 53 kept by per-tile NMS, 43 merge inputs (> 0.1), 26 final |
| size | 10.6 KB gzipped per frame → about 5 MB for 480 frames |
| CPU time | 23.7 s/frame including model warm-up. The capture itself runs on GPU |

The CPU path and the GPU path share all code except the `device` argument.

## 3. Window plan (`_diag/a2cap-plan-v1/plan.json`, CPU, built from the saved T0 decomposition)

There are four 30-frame windows per match, non-overlapping, 480 frames in total.

| kind | rule | purpose |
|---|---|---|
| overlap | most T0 "no box" GT-frames with GT-derived heavy/hidden visibility | targeted: overlap misses |
| small_or_clear | most T0 "no box" GT-frames of visible/partial players | targeted: non-overlap misses (small, low light) |
| random_a, random_b | seeded uniform (seed 13), chosen by no failure | **unbiased controls**, including clear views |

| match | overlap | small_or_clear | random_a | random_b |
|---|---|---|---|---|
| 117093 (tune) | 1599–1628 (205 miss obs) | 1497–1526 (147) | 5327–5356 (0 miss: clear-view control) | 5142–5171 (0) |
| 118575 (tune) | 1413–1442 (53) | 4421–4450 (190) | 4893–4922 (15) | 5106–5135 (6) |
| 117092 (check, low light) | 5951–5980 (155) | 2984–3013 (351) | 415–444 (94) | 4841–4870 (183) |
| 118576 (check) | 5064–5093 (121) | 2987–3016 (56) | 5599–5628 (35) | 1554–1583 (5) |

Totals (T0 codes): 1,616 "no box" GT observations and about 8,900 with a cached box. Those include 8,218 tracked `ok` observations, which serve as controls for the same stage accounting.

**Reporting rule:**
- Targeted windows give *conditional* attributions (for overlap misses; for non-overlap misses).
- Only the random windows give unbiased match-level shares.
- The two are reported separately and never pooled.

## 4. Evaluation (CPU, implemented, `evaluate2`)

Every evaluable GT player in a captured frame gets exactly one category. Rules are in `EVAL_RULES`:

| category | rule |
|---|---|
| detected_and_passed / rejected_downstream_conf | one-to-one Hungarian match (IoU ≥ 0.5) to a final person box; conf > 0.4 or not |
| no_raw_proposal | no raw person candidate with IoU ≥ 0.3 |
| poor_localisation | best raw person candidate IoU in [0.3, 0.5) |
| shared_box_given_to_neighbour | candidates with IoU ≥ 0.5 exist, but every one overlaps another GT player more (one box for two players), or its final box went to the neighbour in the one-to-one match |
| removed_by_tile_nms | own candidates (IoU ≥ 0.5, best-overlap GT) exist, but none survived per-tile NMS. Records the suppressor's GT player |
| removed_by_cross_tile_merge | an own candidate survived per-tile NMS with conf > 0.1 but not the merge. Records the suppressor (approximate) |
| below_cache_floor | own candidates survived NMS only at conf 0.05–0.1 |

Tests (`tests/test_video_m13.py`) cover every category on synthetic frames, the NMS recomputation against Ultralytics' own function, and the refusal paths.

**Smoke-frame illustration (one frame, not evidence).** 117093 f1599 sits in the heaviest-overlap window. 13 GT players were detected and passed, 2 rejected by conf. The non-detected ones were:
- 4 `shared_box_given_to_neighbour`;
- 2 `poor_localisation`;
- 1 `removed_by_cross_tile_merge`;
- 0 `removed_by_tile_nms`.

If the capture confirms this pattern, most overlap misses are a **model-level "one box for two players"** outcome, not NMS deletions. Merge-NMS tuning would then help little, and the fix would be detector training or resolution. That is a hypothesis; the capture decides.

## 5. Approval request A2-cap (exact)

```bash
cd F:/Football-Fatigue-Video
set PYTHONPATH=F:\Football-Fatigue-Video\src;F:\Football-Fatigue-Project\src
F:\Football-Fatigue-Project\.venv\Scripts\python.exe -m football_fatigue.video.detection capture2 --approved ^
  --data-root F:/Football-Fatigue-Project/data --plan data/processed/video/_diag/a2cap-plan-v1/plan.json
F:\Football-Fatigue-Project\.venv\Scripts\python.exe -m football_fatigue.video.detection evaluate2 ^
  --data-root F:/Football-Fatigue-Project/data
```

| item | value |
|---|---|
| model | production YOLO26-L checkpoint, production tiling (1280 × 1280, 20 % overlap, imgsz 1920, batch 2, fp32) |
| frames | 480 (16 windows × 30) on dev matches only |
| GPU time | estimate **15–20 min**: 480 × 1.77 s/frame (production speed) plus one model load. This is an estimate, not a budget. The run records s/frame and peak VRAM per window |
| VRAM | expected < 3 GB on the 4 GB GTX 1650 Ti; measured and recorded |
| disk | about 5 MB |
| writes | only `data/processed/video/_diag/capture-v2/` (write-once) |
| stop condition | abort if any window's `per_tile_nms_recomputed_equals_predictor_output` is false or the replay-vs-cache mismatch exceeds 2 % of boxes |
| unchanged | production configuration and NMS parameters. Merge IoU sensitivity (0.5 / 0.6 / 0.7) is replayed offline on the captured candidates afterwards, with no new inference |
