# Session B — playback repair, T0 profile and detection plan (v1, 2026-11-18)

Scope: development matches 117093, 118575 (tune), 117092 and 118576 (check), plus Wadi Degla videos. Everything here
ran on the CPU or in a browser. **No GPU inference, downloads, installs or training were run.** Reserved SoccerTrack
sequences and FIFA held-out data were not touched. GRU C, HGB C, M11 and FIFA code were not modified.

## A. Root cause and fix of `StreamlitWidgetReadyInstantiationError`

* **Root cause.** The frame slider had `key="frame"` and the view radio had `key="zoom_mode"`. After those widgets
  existed, the code assigned to the same keys in three places:
  * the `run_every` fragment play loop (`st.session_state.frame = nxt`, the reported line);
  * `select_event` (timeline, table and inspector);
  * the missed-episode jump.

  Streamlit forbids that. A regression test reproduces it: `test_root_cause_writing_a_widget_key_after_the_widget_exists_raises`.
* **Fix** (`dashboard/app.py`):
  * The state lives in internal keys: `cur`, `selected` and `view_mode`.
  * The widgets use their own keys (`frame_slider`, `view_radio`). They are synced from the internal state *before*
    they are created. Their `on_change` / `on_click` callbacks write only the internal keys.
  * The rerun-based play loop is removed. Streamlit cannot play video in real time: it reruns the whole script per
    frame and measured 5–6 fps.
* **Options compared:**

  | option | verdict |
  |---|---|
  | Repair the Streamlit loop | Still about 5–6 fps; the crash class remains. **Rejected** |
  | Reuse the M7 demo / M10.8 lab player patterns | HTML5 `<video>` as the master clock (M7), stdlib server with a whitelist and range requests (M10.8). **Adopted** |
  | Minimal replacement of the whole dashboard | Would lose the review/results tabs. Not needed |

  The result: a native player (`football_fatigue.video.player`) that the Streamlit Viewer tab embeds as an iframe.
  The player also works on its own. The Streamlit "Frame inspector" stays for exact-frame stills.

## B. GUI: what exists (all verified in a real browser unless marked)

The player at http://127.0.0.1:5831/ provides:

* **Navigation:**
  * video select (4 dev matches and 2 Wadi Degla videos);
  * play/pause, ±1/±25, frame box, speed, loop over the selected event;
  * prev/next event; timeline click to seek or select, and wheel zoom;
  * keyboard shortcuts.
* **Views:** full frame, follow event, follow track, and manual zoom/pan (wheel and drag).
* **Overlays and inspection:**
  * layer toggles, a legend, and a hover tooltip naming each box's layer;
  * B0/T0 buttons, and a B0 | T0 side-by-side compare mode;
  * an event list with filters;
  * a B0 vs T0 metrics table and a provenance panel.
* **Export:** WebM clip export of the canvas, using MediaRecorder.

**Not implemented:**

* Detector variants other than the production cache (none exist yet; see F).
* Upload of new videos ("Upload (planned)" tab).
* Per-tile raw boxes (never stored).

## C. Separated overlay layers

| layer | source | style |
|---|---|---|
| detector boxes (all cached, conf > 0.1) | Session A detection cache (post tile-merge) | green = passed to the tracker under the selected profile; pink = removed by the turf filter; dotted orange = below conf 0.4 |
| detector-only (Wadi) | M10.8 detection windows | lime |
| tracker output `T##` | ByteTrack output = a **measured** box plus a temporary ID | cyan |
| engine predicted | occlusion engine `hidden_predicted` / `lost_predicted` | dashed red, labelled "predicted" |
| ground truth | SoccerTrack annotation | dotted white |

**Layers that cannot be shown, stated rather than invented:**

* **Raw per-tile YOLO and pre-merge boxes:** never stored, so a capture is needed (G1).
* **ByteTrack's internal Kalman predictions:** not exported. So every `T##` box is a detection, and the only
  predicted boxes on screen come from the engine.

## D. T0 profile (approved, implemented)

* **Profile:** `video/tracking.py`.
  * **B0** = production tracker input. It stays the default, and its outputs are unchanged.
  * **T0** = the same weights, NMS, threshold 0.4 and ByteTrack parameters, but the turf filter decision is ignored.
* **T0 tracks:** `data/processed/video/<src>/tracking/T0-tracker-input-v1/`.
  * They are identical to the scored diagnostic replay on every frame of all 4 matches. The check is recorded in the
    manifest.
* **Engine:** `python -m football_fatigue.video.occlusion run … --tracker-input T0` writes
  `occl-<src>-<s>-<e>-v2-T0-ff8815ed`.
* **B0 integrity:** a B0 rerun is byte-identical to the saved runs. It was checked in scratch on all 4 matches, and is
  covered by a test on a 500-frame window.

## E. B0 vs T0 (same evaluator; full matches; `data/processed/video/_comparison/b0-t0-v1/`)

| match | HOTA B0→T0 | IDSW | Frag | events | detector-miss events | PO precision | PO recall |
|---|---|---|---|---|---|---|---|
| 117093 | 0.371→0.423 | 373→348 | 2,692→921 | 3070→1657 | 1574→154 | 0.706→0.699 | 0.648→0.652 |
| 118575 | 0.371→0.409 | 383→266 | 3,256→1,452 | 2948→1526 | 1505→131 | 0.703→0.719 | 0.695→0.721 |
| 117092 | 0.305→0.307 | 509→509 | 4,496→4,149 | 3470→3334 | 915→701 | 0.571→0.566 | 0.682→0.682 |
| 118576 | 0.384→0.419 | 387→350 | 2,399→996 | 2451→1366 | 1116→63 | 0.742→0.752 | 0.767→0.772 |

**Conclusion:**

* T0 clearly improves tracking continuity: HOTA +0.035 to +0.052 and Frag −55 to −66% on three matches, but not on
  117092.
* T0 removes most false detector-miss events.
* Probable-occlusion precision and recall are unchanged within ±0.03. **T0 does not improve occlusion detection.**
* T0 also adds more non-GT persons (mostly referees) to events: probable-occlusion non-GT events +3 to +14.
* 117092 (low light) is limited by model misses, not by the turf filter.
* The check matches are development data, not independent validation.

## F. Detector inventory and comparison plan

| checkpoint | status |
|---|---|
| `Football-Fatigue-Project/models/vision/player_detector.pt` (YOLO26-L, fine-tuned, imgsz 1280, classes ball/player/referee/goalkeeper, SHA-256 `05ee5d6c…087f`) | **production**; benchmarked through the cache (diag-v1, replay-v1) |
| `models/yolov8n/s/m.pt` (COCO) | downloaded earlier (Session A); yolov8n used for the M10.8 Wadi windows; not benchmarked on SoccerTrack GT |
| OSNet `osnet_ain_x1_0` (ReID) | tested in M6.3-B, NO-GO. A ReID model, not a detector |
| RF-DETR (N–L detection, Seg), football fine-tuned community weights | researched only; **not downloaded** |

* **Trackers** are kept separate: ByteTrack B0/T0 (production); Deep-EIoU-style C1 (M9B, NO-GO).
* **Fair comparison design:**
  * Same frames for every variant: the windows in G1/G2 below, plus the turf examples from diag-v1.
  * Same GT and the same evaluator (`detection ab` / diag stages).
  * Metrics:
    * recall and precision at IoU 0.5;
    * recall by visibility (visible, partial, heavy, hidden);
    * duplicates per 100 TP;
    * ms/frame;
    * downstream HOTA, IDSW and Frag through the same T0 replay.
  * Selection on tune matches; check matches are reported, not tuned on.
* The comparison needs the raw capture first, and nothing is promised before it is measured.

## G. GPU / download approval requests (nothing run)

**G1 — raw per-tile capture for the tile-merge NMS A/B (A0–A3).**

* Commands, one per window, with the same data root and out root:

  ```
  python -m football_fatigue.video.detection capture --data-root F:\Football-Fatigue-Project\data --source <src> --start <s> --frames 125 --approved
  ```

* Windows (8 in total):

  | match | role | starts | how chosen |
  |---|---|---|---|
  | 117093 | tune | 1375, 1500 | densest `merge_suspect` |
  | 118575 | tune | 4750, 875 | densest `merge_suspect` |
  | 117092 | check | 1500, 4500 | fixed quarter points |
  | 118576 | check | 1500, 4500 | fixed quarter points |

  Note: I have now seen the check matches' merge-suspect counts, so these windows are stated in advance by a fixed rule
  instead.
* About 1,000 frames × 1.77 s ≈ **30 min GPU**.
* Peak VRAM under 3 GB (batch 2).
* About 50 MB of disk.
* Then the CPU `detection ab`.

**G2 — 117092 low-light test-time scaling pilot.**

* Same weights; 640-px tiles at imgsz 1280.
* Windows 2875 and 4875 × 125 frames (the most missed GT boxes).
* About **30 min GPU**, VRAM ≤ 3.5 GB, about 30 MB of disk.
* The exact CLI flag still has to be added: a `--tile 640 --imgsz 1280` option for `capture` (a CPU code change only).

**G3 — Wadi Degla upscale pilot.**

* The production YOLO26-L on the 4 M10.8 windows (1050, 2400, 3900, 4800 s), 10 s each, every 5th frame at 2×
  upscale.
* About 240 frames, roughly 10 min GPU, VRAM ≤ 3 GB, about 20 MB.
* Detector-only; no accuracy claims without labels.

**G4 (optional, after G1–G2) — RF-DETR-M zero-shot sanity check.**

* `pip install rfdetr` (a package install) plus about a 130 MB download (Apache-2.0).
* About 20 min GPU on the G1 frames.
* Only worth doing if G1–G2 leave a large gap.

## H. Wadi Degla test plan

* **Files** (`data/external/wadi-degla/raw/`, read-only): `wadi_degla_moving.mp4`, `wadi_degla_fixed_same_match.mp4`
  and `wadi_degla_fixed.mp4`.
  * All are H.264 Main, 640×360, 30 fps, about 90 min each.
  * Camera conditions: a moving broadcast-style camera, and fixed wide cameras; players are small.
* **Playable now:** moving and fixed_same_match play in the player, with the M10.8 detector-only windows at 1050, 2400,
  3900 and 4800 s as an overlay.
* **Proposed annotation set:**
  * 4 windows × 10 frames (every 1 s) per camera, about 80 frames;
  * person boxes for players/referees, plus an `occluded` flag;
  * done in the player's manual view.
  * Without these labels, the Wadi results are visual comparisons only: side-by-side clips via Export.
* **Pipelines to compare on the same frames:** YOLOv8n native 640 (M10.8), YOLO26-L native, and YOLO26-L 2× upscale
  (G3).

## I. Measured performance (CPU decode in the browser, GTX 1650 Ti idle)

| video | presented fps at 1× | dropped frames |
|---|---|---|
| 117093 (4096×1080) | 24.87–24.97 | 0 |
| 118575 | 24.88 | 0 |
| 117092 (3840×1906) | 24.88 | 0 |
| 118576 | 24.90 | 0 |
| Wadi (640×360 @ 30) | 29.95–29.99 | 0 |

* Overlay draw time: 0.4–0.8 ms mean, ≤ 1.6 ms max.
* Overlay vs. video lag: ≤ 1 frame (measured by sampling during play).
* First overlay chunk: about 1.5 s.

## J. Tests and browser QA

* **pytest:** `tests/test_video_player.py` (new) together with the existing `test_video_step2.py`,
  `test_video_occlusion.py` and `test_video_detection.py`: 73 passed. The `needs_data` tests were run, using the
  development data root.
* **Real headless Edge** (`python -m football_fatigue.video.player.qa --streamlit-url …`): **59/59 PASS**.
  * Report: `data/processed/video/_qa/player-qa-v1-20261118-061934/qa.json`, with screenshots.
  * **Exact frame:** browser frame = exactly decoded frame at f100, f1585, f3001 and f5900, after pause, and on the
    other 3 matches.
  * **PTS:** the browser-measured first PTS equals the container value (0.979 s / 0.984 s).
  * **Inconclusive frames:** f1 and both Wadi frames are near-static, so neighbouring frames differ by < 1%. They were
    accepted within ±1 and are labelled INCONCLUSIVE.
  * **Player controls:** buttons, frame box, clamping, events, follow views, timeline mouse click, wheel zoom, B0|T0
    compare, the T0 run switch, loop, export, missing/unknown inputs, refusal of reserved matches, no console errors.
  * **Streamlit:**
    * loads with the player embedded;
    * each nav button moves the frame slider by the expected amount with no exception;
    * the Manual-region and Follow-event views work;
    * all 5 tabs open.
* **Not browser-tested:** run/source switching inside Streamlit is covered by AppTest only. Visual quality on a
  physical GPU-decoding desktop browser was not measured; headless Edge decoded in software.

## K. Launch

```powershell
cd F:\Football-Fatigue-Video
$env:PYTHONPATH="F:\Football-Fatigue-Video\src"
F:\Football-Fatigue-Project\.venv\Scripts\python.exe -m football_fatigue.video.dashboard --data-root F:\Football-Fatigue-Project\data
```

* Dashboard: http://127.0.0.1:5830
* Player: http://127.0.0.1:5831/

## L. Remaining issues

* Detection quality on 117092 (low light) is unchanged. It needs G2 or a model-side change.
* Raw tile boxes are needed to measure the effect of the merge NMS (G1).
* Occlusion precision is about 0.57–0.75. T0 did not change it.
* There are no Wadi labels yet.
