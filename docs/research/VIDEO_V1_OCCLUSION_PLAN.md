# Session B — V1 implementation plan: visual occlusion events + tile-merge A/B

Status: **plan, awaiting approval** (2026-11-17). Background and evidence: [VIDEO_ENGINE_DECISION_REPORT_v1.md](VIDEO_ENGINE_DECISION_REPORT_v1.md).

Scope:

- **In scope:** image-space occlusion events, the tile-merge NMS comparison, a review UI, and evaluation.
- **Out of scope:** persistent identity, roles, 22-player association, calibration work (existing pitch mappings are read only as an optional sidecar), and Session A's models.

## 0. Isolation and environment

| item | rule |
|---|---|
| Code | `F:\Football-Fatigue-Video`, branch `feature/video-occlusion-events` only |
| Inputs (read-only) | `F:\Football-Fatigue-Project\data` (videos, MOT GT, existing caches), passed as `--data-root`; weights `F:\Football-Fatigue-Project\models\vision\player_detector.pt`, passed as `--weights` and SHA-checked. Nothing is written there |
| Outputs | `F:\Football-Fatigue-Video\data\processed\video\<source>\<stage>\<version>\<run_id>\` (ignored by `/data/`), passed as `--out-root`; write-once runs, each with a `manifest.json` |
| Python | New `F:\Football-Fatigue-Video\.venv` (ignored): Python 3.12 and `pip install -e ".[dev,vision]"`, with torch/torchvision cu126 pinned to the versions verified in the shared env (2.9.1 / 0.24.1). **This installs packages and downloads ~3 GB, so it needs your approval.** No new project dependencies |

## 1. Modules

**New** (`src/football_fatigue/video/`; only files that are implemented):

| file | purpose |
|---|---|
| `schema.py` | Event, manifest and review dataclasses; `SCHEMA_VERSION`; JSON read/write with version check |
| `sources.py` | Source descriptor (match/camera id, video SHA, fps, **container PTS offset**, size); frame-index ↔ PTS mapping |
| `occlusion/geometry.py` | IoU, IoS (intersection over smaller box), horizontal overlap, depth order by foot y, box expansion |
| `occlusion/engine.py` | Per-frame evidence, then group graph, then hysteresis lifecycle and event records |
| `occlusion/reference.py` | SoccerTrack GT projection-occlusion episodes (the evaluation reference only; never an engine input) |
| `occlusion/evaluate.py` | Episode recall, review-based precision, failure lists |
| `occlusion/__main__.py` | CLI: `run`, `reference`, `evaluate`, `render` |
| `merge_ab.py` | Raw per-tile capture (GPU step) and CPU replay of merge variants |
| `render.py` | Annotated MP4/PNG output (OpenCV) |
| `review/server.py` + `review/static/` | Stdlib HTTP review app |

**Reused by import, not copied or modified:**

- `vision.detection_cache.DetectionCacheReader.iter_records`: existing caches;
- `vision.tracker.MultiObjectTracker`: ByteTrack replay with the production parameters, which gives track history, lost and born tracks, and predicted boxes;
- `vision.detector`: tiling and the existing merge policies, i.e. variant A0;
- `vision.mot_eval.load_reference` / `standard_metrics`: GT loading and TrackEval;
- `multicam_lab.media.FrameReader`, `probe_video`, `parse_range`: frame decoding and HTTP range serving.

**Untouched:** `vision/identity*.py`, `association.py`, `handover.py`, `demo/`, all Session A packages.

## 2. Event detection logic

Inputs per frame (native fps):

- person detections from the cache;
- the ByteTrack state replayed from the same cache: matched, lost and born tracks, plus the Kalman-predicted box of each lost track.

**Per-frame evidence**, for every pair of tracks in a grid-bucketed neighbourhood:

| signal | definition |
|---|---|
| `overlap` | IoU, IoS, and horizontal overlap of the two boxes, each widened by 10% |
| `depth` | The near member is the one with the larger box bottom y; the other is the far member |
| `far_degraded` | Far member's confidence falls ≥ 0.15 below its 1 s median, or its height shrinks ≥ 15% |
| `hidden` | A track goes lost while its predicted box has IoS ≥ 0.5 with a visible near box |
| `merged` | A visible box's width or area jumps ≥ 1.4× its 5-frame median in the same frame or ±2 frames as a neighbour goes `hidden` |
| `reappear` | A lost track is re-matched, or a new track is born, within the gap limit next to the occluder |

**Three outputs that must stay distinct** (`event_class`):

| class | rule | claim |
|---|---|---|
| `box_overlap` | IoS ≥ 0.2 for ≥ 2 frames; both boxes detected; no degradation | "Boxes overlap in the image." No claim of hiding |
| `probable_occlusion` | Overlap with IoS ≥ 0.5, plus at least one of `far_degraded`, `hidden` or `merged` | "One player is probably partly or fully hidden." `subtype` is `partial`, `full` or `merged` |
| `detection_loss` | A track is lost with **no** occluder (predicted box IoS < 0.2 with every visible box), or the predicted box leaves the frame | "Detector or tracker lost the player; not an occlusion" (subtype `isolated_miss` or `left_frame`) |

**Lifecycle per connected group** (players linked by any active pair edge):

- `open`: first frame with evidence;
- `active`: n_on = 2 of 3 frames, or immediately for `hidden` / `merged`;
- `close`: g = 5 frames without evidence, or the gap limit (25 frames) for a hidden member that never reappears, which sets outcome `not_reappeared`.

Groups that merge become one event, with a `members` timeline. A group that splits creates linked children (`parent_event_id`). The class is upgraded during the event (`box_overlap` → `probable_occlusion`) and never downgraded.

**Outcome flags** (descriptive only): `reappeared_same_track`, `reappeared_new_track`, `not_reappeared`, `truncated_by_window`. The engine does **not** decide identity.

**Thresholds:**

- The numbers above are starting values, tuned only on the tuning matches and then frozen in `config.json` with a hash before the check matches are scored.
- Every value is reported with a ±1-step sensitivity.

**Not recoverable from existing caches** (documented in the report and shown as missed reference episodes):

- Players removed by the A0 merge NMS or below the 0.1 floor. If such a player never had a track, `hidden` cannot fire, so the occlusion is seen at best as `merged` or not at all.
- Any pixel-based evidence: masks, true depth order, appearance.
- Referee and staff occlusions can be detected but cannot be scored (they are not in GT). They are shown with the label "unscored".

## 3. Tile-merge NMS A/B

**Why a raw capture.** The cache stores boxes *after* the merge, so the merge alternatives cannot be replayed from it. One GPU pass stores the **raw per-tile model outputs** (tile index, tile rectangle, class, confidence, box; floor 0.1). Every variant is then replayed on CPU from the same raw outputs. This gives identical model outputs and isolates the merge policy.

**First, confirm.** Ultralytics skips NMS for end-to-end models (`ultralytics/utils/nms.py` line 71; checked in code, not at runtime). Step 3 records `model.end2end` at run time. If it is false, within-tile NMS exists, and that becomes a fourth variant.

| variant | rule |
|---|---|
| **A0** (current) | `edge_aware` with same-role NMS at IoU 0.5 over all boxes, then cross-role at 0.7 |
| **A1** cross-tile only | Same edge-aware truncation rule. NMS (IoU 0.5) only between boxes from **different tiles**, so boxes from the same tile are never suppressed against each other |
| **A2** cross-tile + overlap zone | As A1, but a pair is a duplicate candidate only if both boxes lie mostly (≥ 0.8) inside the **shared overlap strip** of their two tiles |
| **A3** looser global | A0 with IoU 0.7: a simple control |

**Parity check.** A0 replayed from the raw capture must reproduce the existing cache on the same frames: same box count, and boxes equal within 1e-3 px. If it fails, the A/B stops and the cause is reported.

**Sample.**

- 8 windows of 250 frames: 2 per development match (117092, 117093, 118575, 118576).
- Start frames are drawn by `sha256("v1-mergeab|<match>|<k>")` and declared in the protocol before the GPU run.
- 2,000 frames in total.

**Metrics** (person classes; GT matched 1:1 at IoU 0.5; referees and staff excluded from FP as "unscored"):

| metric | definition |
|---|---|
| Occluded recall | Recall by occlusion bin (0, <0.25, 0.25–0.5, 0.5–0.75, ≥ 0.75) |
| Precision | Matched / (matched + unmatched detections overlapping no GT person by IoU ≥ 0.1 and not in "unscored") |
| **Duplicates** | Extra detections with IoU ≥ 0.5 to a detection already matched to the same GT box, plus pairs of detections that both match one GT box better than any other. Reported per 1,000 detections, split into overlap-strip and non-strip |
| Pair separation | For GT pairs with IoU > 0.3, the share in which both members are detected separately |
| Tracker input | Boxes per frame; ByteTrack replay on each variant: HOTA, IDF1, IDSW and Frag within the windows, and IDSW inside vs outside engine events |
| Occlusion events | Engine recall and precision proxy on each variant (same frozen engine config) |
| Runtime | Merge ms/frame on CPU (the GPU capture is shared by all variants) |

**Decision rule** (scored on all 8 windows; pre-declared):

- An alternative is adopted for regeneration only if, against A0:
  - occluded recall at occlusion ≥ 0.5 improves by ≥ 8 points;
  - precision drops by ≤ 2 points;
  - duplicates rise by ≤ 5 per 1,000;
  - window HOTA does not fall by more than 0.005 on any match.
- Otherwise A0 stays and the result is reported as is.
- Full-cache regeneration is a separate request.

## 4. Videos and evaluation

| use | data |
|---|---|
| Engine tuning | 117093, 118575 (full 4-min caches) |
| Engine check | 117092, 118576 (thresholds frozen first) |
| Merge A/B | 8 windows (§3) |
| Qualitative domain check | Wadi Degla `wadi_degla_moving.mp4` and `wadi_degla_fixed_same_match.mp4`, one 60 s clip each (start chosen by hash and away from the known copied insert at 2,711.9–2,748.8 s), needs a GPU detection run |
| Never | 118577 (spent), 118578, 128058, Tier 2 |

**Honesty about independence.** None of the four development matches is independent:

- the detector merge (`edge_aware`) was selected on 117092 windows;
- ByteTrack parameters were tuned on 117093;
- all four were used in identity milestones.

The check split is therefore a **development-internal check of the new engine thresholds**, reported as such. It is never called validation.

**Reference and review:**

- **Reference episodes:** GT projection occlusion with IoS ≥ 0.5 for ≥ 3 frames, merged across gaps of ≤ 2 frames. These are geometric (template GT boxes), so they are a proxy.
- **"Detectable" subset:** both members detected within ±10 frames of the episode.
- **Precision:** a stratified sample of 150 check-match events (50 per class) is labelled by you in the review UI:
  - `true visual occlusion`;
  - `overlap, nobody hidden`;
  - `not an occlusion / error`;
  - `unsure`.
- **Wadi Degla:** no GT, so no recall or precision is reported. Only event counts, a reviewed sample if you label one, and a failure gallery.

## 5. Event schema (`football_fatigue.video.occlusion_event` 1.0.0, JSONL)

```json
{"schema": "football_fatigue.video.occlusion_event", "schema_version": "1.0.0",
 "event_id": "oe-<run8>-<start_frame>-<hash6>", "run_id": "...",
 "source": {"source_id": "soccertrack-117093", "match_id": "117093", "camera_id": "st-pano-0",
            "video_sha256": "...", "fps": 25.0, "pts_offset_s": 0.979, "width": 4096, "height": 1080},
 "event_class": "probable_occlusion", "subtype": "full",
 "start_frame": 3120, "end_frame": 3134, "start_pts_s": 125.779, "end_pts_s": 126.339,
 "members": [{"track_id": 41, "role": "near", "frames": [3120, 3134]},
             {"track_id": 57, "role": "far", "frames": [3120, 3123], "hidden": [[3124, 3131]]}],
 "parent_event_id": null,
 "measurements": {"max_iou": 0.41, "max_ios": 0.93, "overlap_frames": 15, "hidden_frames": 8,
                  "max_area_ratio": 1.62, "far_conf_drop": 0.31},
 "evidence": ["ios>=0.5", "hidden", "reappeared_same_track"],
 "score": 0.82, "score_method": "rule_v1", "calibrated": false,
 "pixel_xy": [2101.5, 402.0],
 "pitch": {"status": "unavailable", "x_m": null, "y_m": null, "sigma_m": null, "calibration_id": null},
 "outcome": "reappeared_same_track", "truncated": null,
 "frames_ref": "frames.jsonl", "provenance": {"engine_version": "0.1.0", "config_sha256": "...",
   "detection_source": {"kind": "cache", "run_id": "...", "records_sha256": "..."},
   "tracker": {"name": "bytetrack", "params": {...}}, "git_commit": "...", "git_dirty": false}}
```

**Per-frame detail.** `frames.jsonl` holds, per frame of each event: member boxes (observed or predicted), pair IoU and IoS, confidence and evidence flags.

**Run files.** Every run also writes `manifest.json` (input hashes, config, counts, status) and an optional `reviews.jsonl` (append-only; never merged into the events).

**Rules:**

- Readers ignore unknown fields.
- Additive changes are a minor version; any rename or retype is a major version.
- `pitch` is filled only from an existing validated mapping (117093, 118575, 118576). Otherwise it is `"unavailable"`. Events never depend on it.

## 6. Interactive demonstration

**Annotated output** (`render`):

- an MP4 per run window, plus PNG strips (start / peak / end crop) for the gallery;
- boxes show the track id;
- overlap regions are shaded by class (grey `box_overlap`, orange `probable_occlusion`, red outline `detection_loss`);
- a hidden member is drawn dashed at its predicted box;
- a start/end banner shows the event id.

**Review app**, `python -m football_fatigue.video.review` (stdlib server, 127.0.0.1):

- **Video and overlay:** the video plays with a live canvas overlay drawn from `frames.jsonl`, synced on `requestVideoFrameCallback` `mediaTime` with the PTS offset.
- **Timeline:** lanes per event class, plus a **GT reference lane** for SoccerTrack with hits, misses and engine-only events colour-coded. Click to seek; `N`/`P` move to the next/previous event.
- **Event panel:** measurements, evidence, outcome, crop strip, label buttons (written to `reviews.jsonl`), and a note.
- **Gallery tabs:**
  - *Correct detections*: events matched to the reference and/or labelled true;
  - *False positives*: events labelled false or not matched;
  - *Missed events*: reference episodes with no event, split by cause (`not_detected_in_cache`, `merged_never_split`, `below_threshold`).
- **Merge A/B view:** the same window under A0 and the chosen alternative side by side, with recall and duplicate counts.
- **Export:** filtered events as CSV/JSONL.

## 7. Running it locally (after approval and implementation)

```powershell
cd F:\Football-Fatigue-Video
.\.venv\Scripts\python.exe -m football_fatigue.video.occlusion run `
  --data-root F:\Football-Fatigue-Project\data --out-root data\processed\video `
  --source soccertrack-117093 --cache soccertrack-117093-m6-d1fixed-full-merged-20261025
.\.venv\Scripts\python.exe -m football_fatigue.video.occlusion evaluate --out-root data\processed\video --source soccertrack-117093
.\.venv\Scripts\python.exe -m football_fatigue.video.review --data-root F:\Football-Fatigue-Project\data --out-root data\processed\video
# open http://127.0.0.1:5830
```

## 8. Acceptance criteria (frozen before scoring the check matches)

| # | criterion | target |
|---|---|---|
| E1 | Recall of detectable reference episodes with IoS ≥ 0.5, on the check matches | ≥ 0.85 |
| E2 | Reviewed precision of `probable_occlusion` (true visual occlusion) | ≥ 0.75 |
| E3 | Reviewed precision of `box_overlap` (overlap exists) | ≥ 0.90 |
| E4 | `detection_loss` events labelled "actually an occlusion" | ≤ 0.20 |
| E5 | Determinism: two runs give byte-identical outputs | yes |
| E6 | Engine cost on top of the tracker replay | ≤ 5 ms/frame CPU |
| E7 | Review UI: sync check on decoded pixels within ±1 frame; click-to-seek; label write and reload | pass (scripted browser check) |
| E8 | Report shows ≥ 10 correct, ≥ 10 false-positive and ≥ 10 missed examples per class where they exist | yes |
| E9 | Tests: unit tests (geometry, lifecycle incl. 3-player groups and truncation, schema round-trip) plus the full existing suite | pass, numbers reported |
| M | Merge A/B: decision rule in §3 applied as written; result reported whether or not it passes | yes |

If E1–E4 fail, V1 still ships the UI and the report, with the verdict stated. Thresholds are not re-tuned on the check matches.

## 9. Compute

| step | device | estimate |
|---|---|---|
| Venv install | network/disk | ~3 GB download, ~6 GB on disk |
| Engine + tracker replay, 4 matches × ~6,000 frames | CPU | ~2–5 min per match (the probe read all 4 caches in 15 s; the ByteTrack replay dominates) |
| Render MP4 for one 20 s window at 4096×1080 | CPU | ~1–2 min |
| Merge A/B raw capture, 2,000 frames | GPU (4 GB) | ~1 h at 1.77 s/frame (measured earlier for this config); the 8 variant replays take CPU minutes |
| Wadi Degla, 2 × 60 s (3,600 frames, 640×360) | GPU | ~10–20 min (estimate; to be measured on 50 frames first) |
| Human review | you | ~1.5–2 h for 150 events |

## 10. Smallest step with a visual result, and the full sequence

**Step 1 (first deliverable): CPU only, existing cache.**

- Implement `schema`, `sources`, `occlusion/geometry` and `engine` (box_overlap plus the hidden/merged signals) and `render`.
- Run on 117093, frames 3000–3499 (20 s).
- Output: an annotated MP4, ~10 PNG event strips and `events.jsonl`, plus their unit tests.
- About one working session; you can watch the result right away.

**Steps 2–6** follow only after you review Step 1:

| step | content |
|---|---|
| 2 | Reference builder and evaluation; tuning on 117093 and 118575; freeze; score the check matches |
| 3 | **GPU:** merge A/B raw capture (2,000 frames), CPU replays, decision |
| 4 | Review app with timeline, galleries and labels |
| 5 | **GPU:** Wadi Degla 2 × 60 s detection, engine, qualitative gallery |
| 6 | V1 report (short), browser check, full test suite |

## Approval requested

1. **Create `.venv` in this worktree** (package install, ~3 GB). Alternative: run Step 1 with the shared env's interpreter in read-only use, and install later.
2. **Implement Step 1** (CPU, existing 117093 cache, outputs under `F:\Football-Fatigue-Video\data\processed\video\`).
3. Later, separately: the GPU runs in Steps 3 and 5 (~1.5 h in total), each started only after a "go".
