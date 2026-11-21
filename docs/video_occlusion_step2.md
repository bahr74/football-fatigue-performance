# Occlusion engine Step 2: review dashboard, readable video, evaluation

Session B, engine 0.2.0. Development matches only; no GPU inference.

## Start the dashboard

```powershell
cd F:\Football-Fatigue-Video
$env:PYTHONPATH = "F:\Football-Fatigue-Video\src"      # makes Python import Session B's code
F:\Football-Fatigue-Project\.venv\Scripts\python.exe -m football_fatigue.video.dashboard --data-root F:\Football-Fatigue-Project\data
```

Open **http://127.0.0.1:5830**. The server is bound to this computer only. Use `--port` to change the port and
`--headless` to stop a browser opening.

The dashboard reads saved runs under `data/processed/video/` and the read-only videos, tracker replays and ground truth
under `--data-root`. It never runs the detector.

## Screen layout

**Sidebar**
* **Match / Run.**
  * `v2` is the Step 2 engine; `step1` is the Step 1 baseline on identical inputs.
  * Runs tagged `video` also have a rendered review MP4 and strips.
  * Full-match runs cover all ~6,000 frames.
* **Event filters:**
  * category: box overlap / probable occlusion / detector miss / tracker gap / uncertain;
  * duration in frames;
  * evidence level;
  * uncertain events: all / only / exclude;
  * temporary track ID.
* **"Group uncertain events in their own row"** puts every uncertain event in the *Uncertain* row. Switch it off to keep
  uncertain events in their class row.
* **Overlays:**
  * other tracked players;
  * `T##` labels;
  * event boxes;
  * predicted (hidden) boxes;
  * dimming of non-selected events;
  * motion-trail length;
  * ground-truth boxes (dashed white, development matches only).

**Viewer tab**
* **Navigation:** −25, −1, +1, +25 frames, Play/Pause (stride 1/2/5, optional loop), previous/next event, and a frame
  slider.
* **Time display:** the frame number, the match clock (`mm:ss.ff` from the first decoded frame) and the container PTS.
* **View:**
  * *Full frame*;
  * *Follow selected event*: a fixed, enlarged crop around the whole event, so the camera does not jitter;
  * *Manual region*: centre x/y and zoom 1–8×.
* **Timeline:** one row per category; bars are events; the white line is the current frame. **Click a bar** to select
  the event and jump to its peak. Drag to pan, scroll to zoom.

**Event inspector** (under the viewer when an event is selected)
* ID, class/subtype, a one-line description, start/end/peak frames and times, duration, evidence level, and whether the
  event is uncertain.
* **Why it fired:** every evidence code explained in words, then the reasons for doubt.
* **Players involved:** role, frames, hidden intervals, and whether the hidden player reappeared (same track / new track
  at the prediction / not within 1 s).
* Raw measurements.
* **Ground-truth comparison:** the outcome (for example `tp`, `fp_gt_overlap_only`, `non_gt_person`), the track → GT
  player mapping, and the matched GT episode with start/end errors.
* **Before / start / peak / end / after** snapshots in one fixed crop.
* **Enlarged players** at the peak frame.
* **Step 1 baseline:** the baseline events that share a track and overlap this event in time.
* **Your review:** correct / incorrect / uncertain plus a note.
  * Saved to `data/processed/video/reviews/<source>/reviews.jsonl`, an append-only file that never changes a run.
  * The latest label wins.

**Events tab:** the filtered table, including GT outcome and review label. Select a row to open the event in the viewer.
There is also a CSV download.

**Results tab:**
* the counts for the selected run;
* event-level and claim-level scores;
* success and failure examples by outcome (select one to inspect it);
* missed GT occlusion episodes (select one to jump there; switch on GT boxes);
* the Step 1 vs v2 table for all development matches;
* the saved reviews.

**Upload (planned) tab:** the design only (see below). Uploading is not available.

### How to inspect one occlusion

1. Choose *soccertrack-117093*, run `v2 · frames 3000–3499 · … · video`.
2. In the sidebar, keep *Probable occlusion* (and *Uncertain*) and set *Evidence level* to `high`.
3. Click an orange bar in the timeline, or press **next event ▶**. The view switches to *Follow selected event*.
4. Step with **◀ −1 / +1 ▶** through the event. Solid boxes are detections; dashed boxes are the engine's predicted
   position of a player who is not detected. Labels show `T##`.
5. Read *Why it fired* and *Players involved*, then check the *Ground-truth comparison* and the five snapshots.
6. Save your verdict under *Your review*.

**Temporary IDs:** `T##` are temporary tracker IDs. They change when the tracker loses a player and are **not**
persistent player identities.

### Playback

Streamlit draws one still image per step. **Play** therefore advances by re-running the viewer every 0.15 s, so at
most about 6 steps/s; use stride 2 or 5 to move faster. It is reliable frame scrubbing, not real-time video.

Measured on this PC (server side, before sending the image to the browser):
* decoding and drawing a full-frame view takes 14 ms/frame (117093) to 23 ms/frame (117092) when stepping sequentially;
* a random jump (seek) takes 0.47–0.62 s.

The browser's display rate was not measured. For smooth playback of a window, open the rendered MP4 (below).

## Readable review video (focus layout)

`python -m football_fatigue.video.occlusion run … --start 3000 --end 3499` (default `--render-style focus`) writes
`annotated.mp4` (1920×1120) and `strips/*.png`. The layout:
* **Top:** the whole panorama with one outline and an `E#` tag per active event; no per-player text.
* **Middle:** a compact legend, including the note that T## IDs are temporary.
* **Bottom:** two enlarged, fixed crops of the most relevant active events (probable occlusion > overlap > loss). They
  show readable `T##` labels, solid observed and dashed predicted boxes, 12-frame trails, and a caption (`E#`, class,
  START/END or progress, and a one-line explanation).

Strips show the before / start / peak / end / after crops of 10 selected events. `--render-style full` reproduces the
Step 1 layout.

```powershell
start data\processed\video\soccertrack-117093\occlusion\0.2.0\occl-soccertrack-117093-3000-3499-v2-ff8815ed\annotated.mp4
```

## Engine profiles and evaluation

See `docs/research/VIDEO_V1_STEP2_EVALUATION.md` for the method, the full-match results and the limitations. In short,
`--profile v2` (the default) adds:
* pair events with group links;
* a play-area filter, whose suppressed events are kept in `suppressed.jsonl` and `suppressed_frames.jsonl`;
* bounded prediction for long gaps.

`--profile step1` reproduces Step 1. To score a saved run:
`python -m football_fatigue.video.occlusion evaluate --data-root … --run-dir <run>`. The result is written to
`<source>/occlusion/evaluation/<run_id>.eval-v1.json`; run folders stay write-once.

## Run folder contract

The dashboard reads any folder that matches this contract, which is also what future uploads must produce:

```
<out_root>/<source_id>/occlusion/<engine version>/<run_id>/
  manifest.json            status=complete, run_id = folder name, source{fps, pts_offset_s, width, height, video_sha256},
                           window{start,end,processing}, profile, config, play_area, replay_run | (uploads) tracks path,
                           video_path_relative | (uploads) video_path_absolute, provenance, stats, import_roots
  events.jsonl             football_fatigue.video.occlusion_event 1.1.0
  frames.jsonl             per-frame event rows (boxes with state visible / hidden_predicted / lost_predicted)
  suppressed*.jsonl        (v2) off-pitch events, never mixed with events
  annotated.mp4, strips/   optional
<out_root>/<source_id>/occlusion/evaluation/<run_id>.eval-v1.json   (development sources only)
<out_root>/reviews/<source_id>/reviews.jsonl                          human labels, append-only
```

## Next milestone design: local video upload (not implemented)

Nothing here processes an arbitrary video yet. The planned workflow:

1. **Upload.** A file picker in the Upload tab, plus a CLI equivalent.
   * The MP4 is *copied* to `data/processed/video/uploads/<upload_id>/source.mp4`; the original is never touched.
   * Record the SHA-256, size, original name and the upload time.
   * Size limit as in `multicam_lab` (4 GB).
   * `upload_id` = date + short hash.
2. **Probe and check.**
   * Use PyAV: frame rate (constant frame rate required, or flagged), resolution, duration and first PTS.
   * Run `verify_reader` (exact decoded-index check against sequential decoding).
   * Optional clip range and preview thumbnails.
3. **Detect (GPU, needs approval per run).**
   * Run the existing detector and tile settings through the existing detection-cache writer into
     `uploads/<id>/detection_cache/`.
   * A job runner like `multicam_lab.jobs` (one job at a time, a status file with stage / progress / ETA / log tail,
     cancel); the dashboard polls it.
4. **Track.** The existing ByteTrack replay with production parameters writes `uploads/<id>/tracks/frames.jsonl`.
5. **Occlusion engine.**
   * The chosen profile; the run is written with `source_id = upload-<id>` and `video_path_absolute`.
   * No play-area filter unless the user draws a pitch outline (a polygon editor on one frame). Otherwise
     `play_area.status = unavailable` is recorded.
6. **Review.**
   * The run appears in the Match / Run picker like any development run.
   * There is no GT, so the Results tab shows counts and human reviews only.

**Dashboard compatibility already in place:**
* runs are discovered by folder contract, not by a fixed list;
* the video path and tracks come from the manifest (`video_path_absolute` is already honoured);
* GT is optional;
* reviews are keyed by source.

**What remains:**
* source registration for uploads (the `SOURCES` table covers development data only);
* the job runner;
* the upload widget;
* the GPU stage, which needs approval;
* a pitch-outline tool.

## Known limitations

* Playback is stepped frame scrubbing (at most about 6 steps/s), not real-time.
* Random seeks take about 0.5–0.6 s.
* Evaluation uses SoccerTrack's template-box GT, which has no referees; see the evaluation report.
* Referees and linesmen on the pitch still create events: they are flagged (`involves_referee_label` when the detector
  labelled them), not suppressed.
* Duplicate tracks on one player create false occlusion pairs; this is not handled yet.
* The evidence level is not a calibrated probability.
* Temporary track IDs are not identities. Persistent identity is a future milestone.
