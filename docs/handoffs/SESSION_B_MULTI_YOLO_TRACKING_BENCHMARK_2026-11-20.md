# Session B handoff — multi-YOLO detection & tracking benchmark (2026-11-20)

Results and analysis: [SESSION_B_DETECTOR_TRACKER_COMPARISON_2026-11-20.md](../research/SESSION_B_DETECTOR_TRACKER_COMPARISON_2026-11-20.md).
Nothing is committed (Git is the user's). Production detector, B0 and historical T0 outputs are unchanged.

## Commands

All commands use `F:\Football-Fatigue-Project\.venv\Scripts\python.exe`, with
`PYTHONPATH=F:\Football-Fatigue-Video\src;F:\Football-Fatigue-Project\src`, run from `F:\Football-Fatigue-Video`.

```powershell
# dashboard (Streamlit :5830 + native player :5831); new tab "🏁 Detector & Tracker Benchmark", switch "Load the benchmark section" on
python -m football_fatigue.video.dashboard --data-root F:\Football-Fatigue-Project\data
# A | B synchronized player for any two tracking profiles
#   http://127.0.0.1:5831/?source=soccertrack-118575&ti=T0&cmp=T0-BOT&frame=2100

# benchmark CLI (all resumable / write-once; CPU unless --device cuda:0 with FF_GPU_APPROVED=1)
python -m football_fatigue.video.bench registry --data-root F:\Football-Fatigue-Project\data
python -m football_fatigue.video.bench detect   --data-root F:\Football-Fatigue-Project\data --detector yolov8m-coco --protocol tile1280-img1280
python -m football_fatigue.video.bench evaluate --data-root F:\Football-Fatigue-Project\data
python -m football_fatigue.video.bench track    --data-root F:\Football-Fatigue-Project\data --source soccertrack-117093 --tracker ul-botsort --input T0
python -m football_fatigue.video.bench gallery  --data-root F:\Football-Fatigue-Project\data   # write-once
python -m football_fatigue.video.bench report   --data-root F:\Football-Fatigue-Project\data

# T0 (and benchmark tracker profiles) for the occlusion engine
python -m football_fatigue.video.tracking  --data-root F:\Football-Fatigue-Project\data --source soccertrack-117093 --profile T0-BOT
python -m football_fatigue.video.occlusion run --data-root F:\Football-Fatigue-Project\data --source soccertrack-117093 --profile v2 --tracker-input T0-BOT --start 0 --no-render
```

## Files

**New:**
- `src/football_fatigue/video/bench/`: `registry`, `frames`, `detectors`, `evaluate`, `trackers`, `visual`, `gallery`, `report`, `__main__`
- `src/football_fatigue/video/dashboard/bench_tab.py`
- `tests/test_video_bench.py`
- this handoff and the research report

**Modified:**
- `video/tracking.py`: adds the T0-UBT / BOT / OCS / DOCS / FT / TT profiles (T0 detections + another tracker). B0 and T0 are unchanged.
- `video/player/player.html`: compare mode takes any two profiles via the `cmp` URL parameter / dropdown; the default is T0.
- `video/dashboard/app.py`: one new tab, inserted before Upload.
- `video/player/qa.py`: one added check (any-two-profile compare via URL).

**Outputs:**
- `data/processed/video/_bench/` (frames-v1, detector-v1, tracker-v1, gallery-v1, report-v1.md)
- per-match `tracking/T0-BOT-*`, `tracking/T0-OCS-*`
- per-match `occlusion/0.2.0/…-v2-T0-BOT-*` and `…-T0-OCS-*`, each with an eval

## Tests that ran

- `tests/test_video_bench.py`: 21 passed. Covers:
  - metrics: AP / matching / views
  - registry and SHA identity
  - unavailable models and the GPU guard
  - tracker idx mapping and IDs
  - manifest determinism / tamper detection
  - reserved refusal
  - cache coordinates and provenance
  - sv-bytetrack reproduces T0
  - event timestamps
  - Streamlit AppTest of the new section
  - Session A `vision` code and `models` unmodified
- Full suite: 1064 passed, 63 skipped, 2 failed. The two failures are `test_proximity_audit.py::test_validation_and_test_matches_refused` and `test_proximity_context.py::test_v3_refuses_validation_and_test_matches`. Both need `data/processed/m11-encounters-v1/splits.json`, which exists only in the project checkout. They are environmental; the files are untouched by this sprint.
- Native player browser QA (`python -m football_fatigue.video.player.qa`, headless Edge): 43/43 passed, including the new check that `?ti=T0&cmp=T0-BOT` opens A | B with both profiles loaded (`_qa/player-qa-v1-20261121-010341`). The Streamlit section was exercised through `streamlit.testing` AppTest (render, toggle, match switch), not in a real browser.

## Decision answers

1. **Is YOLO26-L still the best detector?** Among what could be run, yes. It has the highest AP50 (0.866 vs 0.807 for YOLOv8m@1280, Δ −0.059 [−0.076, −0.041]), mainly through precision. Recall is the same as YOLOv8m. It is unproven against YOLO11 / 12 / 26 COCO and RF-DETR (downloads not approved).
2. **Did any alternative YOLO beat it?** Not on AP50 or F1. YOLOv8m has higher AP50:95 (tighter boxes), higher small-player recall and significantly higher low-light recall on 117092 (+0.059 [+0.015, +0.105]), at about 3× the FP rate.
3. **Training data vs architecture?** Not separable yet. This needs the `yolo26l-coco` control (same base weights, no football fine-tune): one approval.
4. **Should production change?** No. There is no direct evidence that any tested model is better overall.
5. **Do complex trackers beat ByteTrack after T0?** No. On check, sv-bytetrack leads every metric. BoT-SORT's +0.003 tune HOTA is one match.
6. **Best tracker to keep:** production sv-bytetrack (T0). T0L (conf 0.1) cuts sv-bytetrack fragmentation by about 35% (Frag 5145 → 3232 on check) with flat HOTA. It is worth a dedicated engine test.
7. **Does a better tracker improve occlusion-event accuracy?** No. BoT-SORT and OC-SORT trade precision for recall, and F1 is flat or worse on check.
8. **Dominant error:** association (AssA 0.15–0.28 vs DetA 0.61–0.69), then heavy-occlusion detection (recall < 0.45 for every model).
9. **Most promising next step:** GTA-style offline tracklet refinement on T0 tracks (needs the repo download), then the GPU detector run.
10. **Is the historical pipeline intact?** Yes. B0 and T0 files are untouched. The CPU production settings reproduce the cache exactly (6 frames), and the benchmark sv-bytetrack reproduces the T0 tracks exactly (test).

## Approval package (nothing below has been done)

| # | request | source / licence | size | runtime / resources | output |
|---|---|---|---|---|---|
| A1 | Download COCO weights: yolov9m, yolov10m, yolo11m, yolo12m, yolo26m, yolo26l | `https://github.com/ultralytics/assets/releases/download/v8.4.0/<name>.pt`; AGPL-3.0 (v9 original GPL-3.0) | ≈ 20–55 MB each, ≈ 0.25 GB total | – | `F:\Football-Fatigue-Project\models\` (SHA recorded in the registry on first use) |
| A2 | GPU inference (GTX 1650 Ti 4 GB), `FF_GPU_APPROVED=1` | – | – | all local + A1 models × 92 frames × protocol tile1280-img1280 (+ img1920 for the medium models); tile_batch 1, < 3 GB VRAM expected; ≈ 1–3 s/frame → < 1 h total | `_bench/detector-v1/runs/<model>__<protocol>/` with VRAM / latency |
| A3 | RF-DETR: `pip install rfdetr` + adapter + medium weights | Apache-2.0 (N–L weights) | pip deps (torch-compatible; pulls transformers/timm, roughly hundreds of MB) | GPU as in A2 | same |
| A4 | Football checkpoints: easychamp YOLOv8, soccana YOLO11 (Hugging Face) | MIT per card / unstated; AGPL inheritance unclear; training data undocumented | ≈ 50–100 MB | GPU as in A2 | same; reported as B2, with the overlap risk stated |
| A5 | GTA offline refinement: clone `github.com/sjc042/gta-link` (licence to verify on download) | – | small repo; may need ReID weights (separate request) | CPU, minutes per match | `_bench/tracker-v1/<src>/T0-gta-*` |
| A6 | Optional: `pip install trackers` (Roboflow, Apache-2.0) as a second ByteTrack / OC-SORT implementation | Apache-2.0 | small | CPU | same |

DCTracker: no public code, so UNAVAILABLE (nothing to approve).

## Known issues

- **Memory:** Windows commit headroom is small (other apps). Run heavy jobs one at a time. The frame-reader cache in the bench is 2 frames.
- **Strata:** the edge stratum has only 5 frames (118576). Camera motion is N/A.
- The Ultralytics ByteTrack scores 0.02 HOTA below supervision ByteTrack on the same detections, so implementation details matter.
