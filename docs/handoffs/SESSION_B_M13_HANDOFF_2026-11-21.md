# Session B M13 handoff: detection recovery, ByteTrack activation, multi-YOLO staging, GTA readiness (2026-11-21)

**Scope executed:** CPU only, dev matches only (tune 117093, 118575; check 117092, 118576).

Not done in this milestone:
- no GPU inference or training;
- no downloads, installs or clones;
- no Session A change, no production default change, no commit or branch change.

Reserved matches were never opened. Historical outputs were checked after all runs: `baseline.verify()` reports **40 files, 0 changed**.

## Reports

| report | content |
|---|---|
| `docs/research/SESSION_B_M13_RAW_TILE_CAPTURE_READINESS.md` | capture v2 design, 1-frame CPU readiness test, 480-frame plan, attribution rules, **A2-cap request** |
| `docs/research/SESSION_B_M13_BYTETRACK_ACTIVATION_ABLATION.md` | semantics, tune grid, check, identity analysis, decision (not promoted) |
| `docs/research/SESSION_B_M13_DETECTOR_BENCHMARK_PLAN.md` | local checkpoint audit, complementarity of existing runs, frames-v2, stages A–E |
| `docs/research/SESSION_B_M13_GTA_FEASIBILITY.md` | GTA requirements and mechanism, MOT export adapter, protocol, **A5 request** |
| `docs/research/SESSION_B_M13_FAILURE_CASE_STUDIES.md` | 13 seeded cases in 6 categories plus the reference case |
| `docs/research/SESSION_B_M13_SESSION_AB_COMPATIBILITY.md` | Session A contract mapping, rules, synthetic-tested builder |

## Immutable M12 baseline (`_diag/m13-baseline-v1/manifest.json`)

The manifest records sources, detector sha, cache/tiling/merge configuration, tracker semantics, GT matching rules and the profile of every claimed number. It also stores sha256 of the 40 historical result files. Every share below uses **526,760 evaluable GT observations** as its denominator:

| claim | count | share | profile |
|---|---|---|---|
| no post-merge box | 33,579 | 6.37 % | profile-independent |
| 117092 no box | 18,649 / 131,987 | 14.13 % | profile-independent |
| rejected by confidence ≤ 0.4 | 15,826 | 3.00 % | B0 = T0 |
| turf filter | 26,732 | 5.07 % | B0 only |
| untracked | 4,209 | 0.80 % | T0 (B0 0.85 %, T0L 2.07 %) |
| temporary-ID changes | 1,890 (733 with the old ID continued on another player) | – | T0 |

## Code changes (all in Session B, all new versions)

| file | change |
|---|---|
| `video/tracking.py` | profiles `T0-A03`, `T0L-A03` (activation 0.3); `production_tracker_config`, `run_sv_bytetrack`; manifests record the activation actually used. T0 rebuild verified byte-identical. `RECOMMENDED_TRACKER_INPUT` unchanged |
| `video/failures/activation.py` (new) | activation grid, pre-declared selection, check scoring |
| `video/failures/baseline.py` (new) | baseline manifest + `verify()` |
| `video/failures/cases.py` (new) | seeded case study selection |
| `video/failures/recovery.py` (new) | confidence × size and gap tables (CPU) |
| `video/failures/pipeline.py`, `engine_compare.py`, `__main__.py` | `--version`, `--profile` options. Defaults unchanged |
| `video/detection/capture_v2.py` (new) and `detection/__main__.py` | raw pre-NMS capture, plan, attribution; `plan2` / `capture2` / `evaluate2`. Refuses GPU or non-smoke runs without `--approved` |
| `video/bench/mot_export.py` (new) | GT-free MOT export/import for GTA |
| `video/bench/complement.py` (new) | complementarity of existing detector runs |
| `video/bench/frames.py` | `version`/`seed` parameters (frames-v1 default unchanged); frames-v2 built |
| `video/ab_tracklets.py` (new) | Session A contract builder (observed pixels only; pitch null without calibration) |
| `tests/test_video_m13.py` (new) | semantics, selection, export, attribution, NMS recompute, CLI refusals, baseline, plan, A/B builder |

Known stale text, not changed: the `detection/merge.py` docstring says YOLO26's one-to-one head de-duplicates inside a tile. The production checkpoint has `end2end = False`, so per-tile NMS runs.

## Commands run (CPU)

```bash
export PYTHONPATH=/f/Football-Fatigue-Video/src:/f/Football-Fatigue-Project/src
python -m football_fatigue.video.failures activation --data-root F:/Football-Fatigue-Project/data
# per match x {T0-A03, T0L-A03}: tracking.build -> occlusion run --profile v2 --tracker-input P --start 0 --no-render -> occlusion evaluate
python -m football_fatigue.video.failures decompose --version pipeline-m13-v1 --profile T0 --profile T0L --profile T0-A03 --profile T0L-A03 (4 sources)
python -m football_fatigue.video.failures engine --version engine-activation-v1 --profile T0 --profile T0L --profile T0-A03 --profile T0L-A03
python -m football_fatigue.video.failures cases --data-root F:/Football-Fatigue-Project/data
python -m football_fatigue.video.detection plan2 --data-root F:/Football-Fatigue-Project/data
python -m football_fatigue.video.detection capture2 --device cpu --smoke 1 --source soccertrack-117093 --start 1599   # readiness only
# baseline.build, recovery.run, complement.run, frames.build(version='frames-v2', seed=20261121) via python -c
```

The chain took about 45 min on CPU (grid 15.6 min; each profile build plus engine run 64–92 s per match). The player lists the two new profiles (checked via `/api/sources`).

## Tests

- Focused: `tests/test_video_m13.py tests/test_video_failures.py tests/test_video_bench.py tests/test_video_detection.py`, run with `FOOTBALL_FATIGUE_DATA_ROOT` set. Result: **67 passed, 1 skipped**. The skipped test checks the activation manifests; it ran before those profiles were built.
- After the chain finished, `test_video_m13.py` alone: **15 passed**, including the activation-manifest test.
- Full suite: **1095 passed, 63 skipped, 3 failed** in 4 min 39 s.
  - `test_proximity_audit::test_validation_and_test_matches_refused` and `test_proximity_context::test_v3_refuses_validation_and_test_matches` are the known environmental failures, unchanged.
  - `test_video_failures::test_t0l_engine_event_timestamps` was a test bug caused by M13: its glob `-v2-T0L-*` also matched the new `T0L-A03` run. The glob is now exact.
  - After that fix, `test_video_failures.py` and `test_video_m13.py` give **32 passed**. The full suite was not re-run after the fix.

## Answers

1. **Largest verified cause of disappearance:** no post-merge detector box. That is 6.37 % of evaluable GT observations, and 14.13 % in low-light 117092.
   - It does not change under any tracker setting.
   - The turf filter (5.07 %) was larger, but only in B0, and T0 already removes it.
   - 72 % of ID changes follow a detector gap.
2. **Unknowable until raw-tile capture:** for a missing box, whether the model never proposed it, per-tile NMS (IoU 0.7) removed it, the cross-tile merge (IoU 0.5) removed it, or the box was shared with or mislocalised onto the neighbour. Consequently, whether merge/NMS tuning can help at all is also unknown.
3. **Does lowering activation genuinely improve tracking?** No.
   - Coverage and AssA rise, but verified swaps rise 9–25 %, FP up to ×2.2, and false alerts up to 44 %.
   - HOTA is −0.021 to +0.044 depending on the match.
   - The check gain of the tune-selected cell (+0.011) comes from 117092 alone.
4. **Production baseline:** T0 (sv-ByteTrack, activation 0.5, cached boxes conf > 0.4, no turf filter). Unchanged.
5. **First local detector to benchmark:** YOLOv8m-COCO at 1280, on frames-v2, CPU (~30 min).
   - It found 119 of 195 production misses on frames-v1, mainly in low light and on small players.
   - The yolo26l-football @1280 same-size control comes next.
6. **GTA: ready but blocked on approval A5.**
   - Code, MIT licence and the committed SportsMOT OSNet checkpoint were verified on GitHub.
   - It needs a separate Python 3.8 environment and torchreid, plus a ~12 GB frame export.
   - The MOT adapter and evaluation protocol are prepared.
   - Main risk: the connector is appearance-only with no gap limit, and the kits are identical.
7. **Smallest useful GPU experiment:** A2-cap, 480 frames with the production model and configuration, about 15–20 min of GPU, about 5 MB output.
8. **Next Session B milestone (M14):**
   - run A2-cap plus evaluation, including an offline merge-IoU replay;
   - run detector Stage A on CPU;
   - then decide between detector work (resolution, fine-tune, second model) and merge changes.
   - GTA follows if A5 is approved.

## Approvals needed (none were used)

| id | what | cost |
|---|---|---|
| **A2-cap** | GPU raw-tile capture, exact command in the readiness report | ~15–20 min GPU, ~5 MB |
| A1 | download COCO checkpoints for Stage B (yolo11m, yolo26m, yolo26l-coco, …) | ~40–100 MB each |
| A2 | GPU inference for Stages B–E | per the stage table |
| A3 | install RF-DETR | new dependency |
| A4 | Hugging Face football checkpoints, after licence check | download |
| A5 | clone gta-link, Python 3.8 env with torchreid 1.4.0, 12 GB frame export | env + disk |

Stage A (local CPU detectors on frames-v2) needs no approval. It needs one small code change first: detector run folders keyed by frame-set version.
