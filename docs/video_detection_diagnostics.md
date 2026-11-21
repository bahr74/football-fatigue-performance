# Detection diagnostics: commands and dashboard view

Session B, `diag-v1` / `replay-v1`. Development SoccerTrack matches only (117093, 118575 tune; 117092, 118576 check).
Everything here is CPU-only and reads the existing detection caches, tracker replays and ground truth without
changing them. The results are in `docs/research/VIDEO_DETECTION_DIAGNOSIS_v1.md`.

## Commands

```powershell
cd F:\Football-Fatigue-Video
$env:PYTHONPATH = "F:\Football-Fatigue-Video\src"
$PY = "F:\Football-Fatigue-Project\.venv\Scripts\python.exe"
$DR = "F:\Football-Fatigue-Project\data"
$S = "--source","soccertrack-117093","--source","soccertrack-118575","--source","soccertrack-117092","--source","soccertrack-118576"

& $PY -m football_fatigue.video.detection diagnose --data-root $DR @S   # stage attribution, ~30 s per match
& $PY -m football_fatigue.video.detection examples --data-root $DR @S --per-stage 2   # comparison panels (PNG)
& $PY -m football_fatigue.video.detection replay   --data-root $DR @S   # tracker-input A/B, ~2-3 min per variant
& $PY -m football_fatigue.video.detection report                        # markdown tables
```

Outputs are written once, to `data/processed/video/<source>/detection/`:

```
diag-v1/summary.json        stage x visibility x size counts, precision split, threshold sweep, AP50, engine cross-reference
diag-v1/gt_rows.csv.gz      one row per ground-truth player per frame (stage, visibility, size, matched box)
diag-v1/engine_xref.csv     every predicted box of the occlusion engine, matched to ground truth and its stage
diag-v1/examples/*.png      stage-comparison panels (+ index.json)
replay-v1/<variant>/        tracks.jsonl.gz + result.json (HOTA / CLEAR / IDF1; B0 is checked to equal production)
```

Not run without approval:
* `capture … --approved`: raw per-tile detector capture on the GPU;
* `ab --capture <dir>`: the CPU merge-policy A/B on that capture.

## Stages

Each ground-truth player in each frame is assigned to the first stage that still holds a matching box (one-to-one
Hungarian matching, IoU ≥ 0.5):

| stage | meaning | which component is responsible |
|---|---|---|
| `tracked` | a tracker output box matches | — (OK) |
| `detected_untracked` | detection above the tracker threshold, but the tracker output no box | tracker |
| `below_threshold` | only a cached box with 0.1 < conf ≤ 0.4 | post-processing (confidence threshold) |
| `turf_filtered` | the box was removed by the turf (grass-colour) filter | post-processing (turf filter) |
| `poor_overlap` | an unclaimed box overlaps it with IoU 0.2–0.5 | uncertain (annotation or box shape) |
| `merge_suspect` | no box, but a surviving neighbour overlaps it (IoU ≥ 0.3 or covers ≥ 70%) | possibly tile-merge NMS; confirming it needs a raw capture |
| `no_detection` | no box anywhere near the player after merging | detector (model) |

Visibility comes from ground truth only:
* **visible / partial / heavy / hidden**: < 10% / < 50% / < 90% / ≥ 90% of the box is covered by nearer players;
* **off_pitch**: the foot point is more than 0.5 box heights outside the annotated pitch;
* **uncertain**: the box touches the image border.

Referees are not in SoccerTrack's ground truth, so occlusion by a referee is not counted.

## Dashboard: 🔬 Detection diagnostics tab

Start the dashboard as before (`python -m football_fatigue.video.dashboard --data-root …`, http://127.0.0.1:5830) and
open the **Detection diagnostics** tab.

* **Match:** any match that has a diagnosis.
* **Tracker output shown:**
  * `B0` is the production tracker;
  * the other replay variants (for example `T0` = turf filter off) show what the tracker would have output from the
    same detections.
* **Crop size:** the zoom, in player heights.
* **Jump filters + ◀◀ prev lost / next lost ▶▶:** step through frames where a player of the chosen stage and visibility
  is lost. `±1`, `±25` and the frame box move freely.
* **Curated example:** the longest runs per stage (the same frames as the PNG panels).
* **Focus player:** any ground-truth player in the frame. It defaults to the first visible lost player.
* **Comparison strip:** side-by-side zoomed panels.
  1. *Ground truth* (focus player in yellow).
  2. *Raw per-tile*: an explicit "not captured" panel. The caches never stored pre-merge boxes.
  3. *Post-merge (cache)*:
     * green = tracker input;
     * orange = below the 0.4 threshold;
     * magenta = removed by the turf filter.
  4. *Tracker output* (cyan, `T##`).
  5. *Engine predicted* (dashed red: the occlusion engine's predicted box for a lost or hidden track).
* **Verdict line:** which stage lost the player:
  * POST-PROCESSING (turf filter or threshold);
  * TRACKER;
  * POSSIBLE MERGE SUPPRESSION;
  * DETECTOR (model miss);
  * UNCERTAIN.
* **Whole frame** expander: every ground-truth player coloured by stage.
* **Tables:**
  * all ground-truth players of the frame;
  * the whole-match stage-by-visibility shares;
  * the tracker-input A/B table.

`T##` are temporary tracker IDs, not player identities.
