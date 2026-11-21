# Video occlusion events — V1 Step 1

This is the first working, visual occlusion-event demo. It runs CPU only, from existing cached detections and the existing ByteTrack replay, on SoccerTrack match 117093, frames 3000–3499 (20 s). Plan: [research/VIDEO_V1_OCCLUSION_PLAN.md](research/VIDEO_V1_OCCLUSION_PLAN.md).

## What it does

The `football_fatigue.video` package reads two existing artefacts, read-only:

- **Detection cache:** `processed/vision/soccertrack-117093-m6-d1fixed-full-merged-20261025`.
- **Production tracker replay of that cache:** `…-full-stable-replay/frames.jsonl`. Every tracked box in the processed frames is checked against the cache before use.

It writes three event classes, which are kept apart:

| class | meaning | subtypes |
|---|---|---|
| `box_overlap` | Two visible boxes overlap (widened IoS ≥ 0.2 for ≥ 2 frames). No claim that anyone is hidden | `overlap` |
| `probable_occlusion` | Strong overlap plus evidence of hiding | `partial`: the far box loses confidence or height while overlapped. `full`: a track is lost while its constant-velocity prediction is ≥ 50% covered by a visible box. `merged`: `full`, and the occluder's box grows ≥ 1.4× |
| `detection_loss` | A track is lost with nobody in front of it | `detector_miss`: no cached person box at the predicted spot. `tracker_gap`: the cache had a box there but the tracker did not output it. `left_frame`. `near_neighbour_uncertain`: 20–50% covered |

**Lifecycle.** Events start, continue and end per connected group of players. An event closes after 5 frames without evidence. Groups that touch merge into one event; a group that splits creates a child with `parent_event_id`. A hidden player is followed for at most 25 frames (1 s). Each hidden member gets an outcome:

- `reappeared_same_track`;
- `new_track_at_prediction` (a location match, **not** an identity check);
- `not_reappeared`;
- `truncated`.

No identity, role or pitch coordinates are computed (`pitch.status = "not_computed"`).

**Schema.** `football_fatigue.video.occlusion_event` 1.0.0 (`src/football_fatigue/video/schema.py`). Readers ignore unknown fields and refuse a different major version.

## Reproduce

These commands are run from `F:\Football-Fatigue-Video`. Step 1 uses the existing interpreter of the main worktree **only to execute installed packages**. `PYTHONPATH` makes imports resolve to this worktree's `src`, and the run refuses to start otherwise; resolved paths are recorded in `manifest.json` → `import_roots`.

```powershell
$env:PYTHONPATH = "F:\Football-Fatigue-Video\src"
F:\Football-Fatigue-Project\.venv\Scripts\python.exe -m football_fatigue.video.occlusion run `
  --data-root F:\Football-Fatigue-Project\data --out-root data\processed\video `
  --source soccertrack-117093 `
  --cache-run soccertrack-117093-m6-d1fixed-full-merged-20261025 `
  --replay-run soccertrack-117093-m6-d1fixed-full-merged-20261025-full-stable-replay `
  --start 3000 --end 3499
```

Runs are write-once: rerunning into an existing run folder is refused. Use another `--out-root` to reproduce; `events.jsonl` and `frames.jsonl` come out byte-identical.

**Outputs.** They are written to `data\processed\video\soccertrack-117093\occlusion\0.1.0\occl-soccertrack-117093-3000-3499-da3e0b5f\` (Git-ignored):

| file | content |
|---|---|
| `events.jsonl` | one event per line |
| `frames.jsonl` | per-frame boxes, pairs and hidden predictions per event |
| `manifest.json` | input hashes, config, stats, timing, import roots, frame-index check |
| `annotated.mp4` | H.264, 4096×1080, 25 fps, frames 3000–3499 |
| `strips\*.png` | before / start / peak / end / after crops for 10 events |

**Video legend:**

- grey: tracks not in an event;
- yellow: `box_overlap`;
- orange: `probable_occlusion`;
- red: `detection_loss`;
- dashed boxes are predictions where nothing was detected;
- `E<n>` labels match the strip file names.

**Tests:** `tests/test_video_occlusion.py`. The real-data tests run only when `FOOTBALL_FATIGUE_DATA_ROOT` is set:

```powershell
$env:PYTHONPATH = "F:\Football-Fatigue-Video\src"; $env:FOOTBALL_FATIGUE_DATA_ROOT = "F:\Football-Fatigue-Project\data"
F:\Football-Fatigue-Project\.venv\Scripts\python.exe -m pytest tests\test_video_occlusion.py -q -p no:cacheprovider
```

## Known limitations (Step 1)

- **Events inherit the cache's blind spots.**
  - Players removed by the tile-merge NMS (IoU 0.5) or below the 0.1 floor are invisible.
  - An occlusion in which the hidden player never had a track cannot be detected.
  - Severe occlusions are under-detected by the cache itself (decision report §A.4).
- **Prediction while hidden is constant-velocity in pixels.** Over long gaps it can drift onto empty grass.
- **No pitch or touchline mask.** Off-pitch people (bench, staff, spectators) produce events, mostly `tracker_gap` and `detector_miss`.
- **Chained overlaps form long crowd events.** Example: 7 tracks over 110 frames. Their strips are hard to read.
- **Thresholds are untuned starting values** (`EngineConfig`). Scores are uncalibrated; `evidence_level` is a rule-based label.
- **`multicam_lab.media.FrameReader` indexes by absolute PTS.** That is 24 frames early for 117093, whose container starts at 0.979 s. Step 1 therefore uses its own `ExactFrameReader`, checked against sequential decoding on every run. The legacy reader was not changed.
