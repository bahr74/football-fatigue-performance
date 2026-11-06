# Football Vision Demo (M7-DEMO)

A local, display-only dashboard over **saved** outputs of the CV work (no detection, no training, no GPU). Default match: **SoccerTrack v2 117093**
(4096×1080 panorama, 25 fps, 5 976 frames = 3 min 59 s) — the only development match whose video is in the repository next to its saved
detection/tracking run and team/appearance caches.

## M7.3 additions (2026-10-28), all opt-in
Details are in `data/processed/vision_eval/m73-core-engine-v1/` (`M7.3_EXECUTIVE_SUMMARY.md`, `M7.3_TECHNICAL_REPORT.md`, `commands.md`). Defaults are unchanged.
- **Physical analytics:**
  - **positions:** "continuous (M7.3)" uses the sidecar `data/processed/demo/117093/motion_v1.json` (continuous-inverse positions plus per-sample σ).
  - **Smoothing:** Savitzky–Golay, Hampel+SG, robust polynomial and Kalman RTS (`analytics/motion.py`), plus a "Kalman response" setting.
  - The table has a "± pos (m)" column.
  - The plausibility gate is recomputed for each position source and is unchanged. Real values stay **withheld**: raw/smoothed distance 1.74 against a 1.15 limit.
- **Suggested hand-overs** (Identity review panel, `GET /api/suggestions`, `handover_suggestions_v1.json`):
  - short-gap link candidates from `vision/handover.py`, with tier, evidence and competing candidates;
  - buttons to jump before or after the gap, and accept, reject or unsure (`handover_decision` in the review log).
  - Suggestions never change predictions. Automatic joining was evaluated and rejected (NO-GO).

## M7.2 additions (2026-10-27)
Details are in `data/processed/vision_eval/m72-sprint-v1/` (`M7.2_EXECUTIVE_SUMMARY.md`, `M7.2_TECHNICAL_REPORT.md`).
- **Overlay v5** (`data/processed/demo/117093/overlay_v5.json`) uses an opt-in team assignment (`demo/teams.py`):
  - colour votes within ±1 s inside each tracklet, kept only where they agree with the whole-tracklet vote;
  - touchline people become Unknown.

  Frames with more than 11 eligible Blue dropped from 58.7 % to 15.5 %. Positions and identities are unchanged; v4 columns are kept as `*_v4`.
- **Physical analytics panel.**
  - Coach definitions in km/h, m and s, with presets (`analytics/coach.py`).
  - The server computes everything (`POST /api/analytics`, `demo/physical.py`).
  - Eligibility levels: L1 observed segment, L2 reviewed identity, L3 blocked.
  - Real values are **withheld** while the plausibility gate fails (foot-point jitter). Engineers can see diagnostic values, labelled NOT VALID, in the Advanced view.
  - Exports: summary JSON, segment/sprint CSV, preset JSON, data-quality JSON.
- **Identity review panel** (`demo/review.py`, `GET/POST /api/reviews`).
  - Append-only log at `data/processed/demo/117093/reviews/identity_reviews_v1.jsonl`; it never changes predictions and is not validation evidence.
  - "Reviewed" label mode.
  - Options: `--review-file` and `--preset-dir`.

## Launch
```powershell
cd F:\Football-Fatigue-Project
.\.venv\Scripts\python.exe -m football_fatigue.demo            # opens http://127.0.0.1:5790/ in the default browser
.\.venv\Scripts\python.exe -m football_fatigue.demo --no-browser --port 5790
```
Use Chrome or Edge (H.264 4096×1080 plays natively). Keyboard: Space play/pause, ←/→ ±1 frame, Shift+←/→ ±1 s. The server is the existing playback
server's handler (range-streamed video) restricted to a fixed route whitelist; it does not touch the FIFA board or `/cv` pages.

## M7.1 (2026-10-27): coach dashboard and stable board — current version
The page at `/` is the redesigned coach dashboard. The M7 layout is kept at `/m7`. It serves `data/processed/demo/117093/overlay_v4.json`: the v3 observed rows, unchanged, plus a display-continuity layer from `football_fatigue.demo.continuity`.
- **Gap fill.** Short tracking gaps (≤ 25 frames, same tracker ID and identity tracklet, plausible movement, not crowded for gaps > 12 frames) get dashed, fading *estimated* markers.
- **Lost fade.** A player whose track ends fades out over 6 frames, marked "lost".
- **Never measurements.** Estimated rows are never counted, never analytics-eligible and carry no confidence.
- **Labels.** Number (stable, = tracker ID), Tracker ID and Identity are separate. Identity labels downgrade immediately and upgrade only after 12 frames, so they are never shown as more trusted than the data.
- **Measured effect** (whole clip):
  - Marker flicker: 880.8 → 69.8 per minute with gap fill, 40.4 including the lost fade.
  - Identity-label flicker: 139.8 → 57.2 per minute.
  - Identity statuses are unchanged.
- **Further detail:** coach panels (team overview, selected player, insights, team analytics), the movement-metrics readiness table and the measurements are in `data/processed/vision_eval/m71-sprint-v1/README.md` and `SUMMARY.md` (feature matrix, remaining problems, 3-minute mentor script).
- **Team naming.** Teams are named Blue and White after their kit-colour clusters, not home and away. Blue is over-filled in 59 % of frames, and the page warns about it.

The sections below describe the M7 layout, which is still available at `/m7`.

## What you see
- **Video panel:** the real match video; saved detection boxes; local tracker IDs or global-ID hypotheses (toggle); box style = identity status
  (confirmed solid green, provisional dashed amber, unknown/ambiguous dotted); team colour chip only when team confidence ≥ 0.7.
  Fit / 2× / 4× zoom with follow-selected; safe-segment pixel trail of the selected player; click a box to select.
- **Timeline:** per-≈4 s share of confirmed / provisional / unknown boxes (coverage over time) and detections per frame; click/drag to seek.
- **Board (right), default "Pitch projection (approx.)" (overlay v3, M7 sprint):** foot points mapped to a 105×68 m pitch with a hybrid fisheye + thin-plate-spline
  calibration fitted to the 65 supplied keypoints of 117093.
  - Held-out keypoint error: leave-one-out median 0.45 m, p90 1.07 m; interior (n=11) 0.52 / 1.75 m.
  - Valid for display, team shape and coarse zones; **not** for speed or distance (the p90 fails the 1.0 m gate).
  - Markers 1–5 m outside the pitch are grey. Markers more than 5 m outside (bench, staff) are not drawn, and their number is shown.
  - Nobody is moved or spread.
  - "Image layout" switches back to the old image-space schematic (APPROXIMATE · NON-METRIC, origin top-left, units pixels, projection none).
- **Calibration layer** (checkbox): reprojected pitch markings, keypoints coloured by their leave-one-out error (green ≤ 0.5 m, yellow ≤ 1 m, red > 1 m;
  values printed at 2×/4×) and foot points coloured by pitch validity.
- **People filters:** All detected / Active-player candidates (players + goalkeepers) / Goalkeepers / Officials / Unknown role; colour by team or role.
  Roles combine the detector class, the team cluster and pitch validity. Off-pitch people and the third colour cluster stay *unknown*. There is no cap on people shown.
- **Identity-continuity timeline** for the selected tracker ID: status per observed frame, gaps, safe-segment boundaries, attributable segments; click to seek.
- **Team analytics:** zone occupancy (6×4) per team and team shape (centroid, length, width, visible count, frame coverage) for the whole clip or the last 60 s / 15 s.
  It uses observed, on-pitch, active-role, team-confident samples only, so it describes the visible subset. Readiness badges explain why speed, distance, pitch control, fatigue and ball metrics are unavailable.
- **Panels:** detections, active tracks, confirmed / provisional / unknown / ambiguous counts, trusted share, selected-player provenance (local
  tracker ID, global-ID hypothesis with trust wording, tracklet ID, status, local confidence, link confidence, association status, team, observed vs
  estimated, track span and interruptions, M7-0 safe segment and whether it is attributable), match insights and a confidence histogram.

## Sync fix (2026-10-27) — read this if boxes ever look early or late
v1 of the overlay stored `video.start_time_s = 0.0` instead of the video's first-frame PTS **0.979 s** (the build script's repository path was one level
too high and `run_index.video_start_time_s()` silently returns 0.0 for an unreadable file). Every overlay, board marker and counter was therefore 24 frames
(0.96 s) ahead of the displayed video; the earlier self-test could not see it because it converted time to frame with the same constant. Measured on decoded pixels
(`data/processed/vision_eval/m7demo-sync-v1/`): before 1 of 73 captures showed the claimed frame (all others exactly −24), after 72 of 72 at 0.5×/1×/2×, paused, seeks, steps, start and end.
Three independent safeguards now exist: the build script reads the PTS from the video and fails if it cannot; the server checks the stored start against the container
on every start and overrides a mismatch (`start_time_source`); and the page measures the first presented frame's `mediaTime` itself and uses it if it disagrees (badge shown).

## One master clock
The `<video>` element is the clock. Every redraw (video overlay, board, counters, selected card, timeline) is driven by
`requestVideoFrameCallback`'s presented `mediaTime` → frame `floor((t − start)·25)`, where `start = 0.979 s` is the container's first-frame PTS
(PTS(f) = 0.979 + f/25 verified with PyAV for all frames). Seeking sets `currentTime` to the middle of the target frame; stale frames presented
while a seek is in flight are ignored; nothing is drawn from a different frame than the one on screen. If the video cannot be loaded the page says so
and runs the same views on a stand-in clock (no video pixels).

## Data and how it is produced
The server serves `data/processed/demo/117093/overlay_v3.json` (15 MB), built by `data/processed/vision_eval/m7-sprint-v1/scripts/build_overlay_v3.py`.
It contains exactly the v2 boxes (the script asserts this) plus these columns:
- `pxm`, `pym`: pitch metres.
- `pv`: pitch validity.
- `kpd`: distance to the nearest keypoint.
- `role`: role code.

It also carries the calibration diagnostics. Evaluation and audit are in `data/processed/vision_eval/m7-sprint-v1/README.md`.
`data/processed/demo/117093/overlay_v2.json` (12 MB, columnar; `overlay.json` is the superseded v1, kept, see the sync fix below) is built once by
`data/processed/vision_eval/m7demo-v1/scripts/build_overlay.py`: it re-solves the **frozen** M6.3-A status-cap deferred identity configuration (LOMO
weights, not fitted on 117093) on the cached tracker records, annotates them, runs the M7-0 export for the safe segments, and writes the overlay
(boxes, statuses, confidences, association, team, segment index). Source: `soccertrack-117093-m6-d1fixed-full-merged-20261025-full-stable-replay`. The build
is the same as the M7-0 evaluation (133 259 boxes, 138 attributable segments, 3 722 transitions blocked, 0 bridged). The identity solve is
offline (needs the whole match), so the demo is a replay, not a live system.

## Display-only interpolation (off by default)
"display-interpolated" shows ≤ 5-frame gaps as dashed grey boxes marked `~`, only if both ends share tracklet, local id, global id, identity status,
association status and class. They are `est=1` extras: excluded from every count, from the histogram, from segments and from the selected-player
numbers; observed samples are unchanged. Strict observed-only is the default. Audit on the real overlay: 6 140 estimated boxes, 0 violations.

## Verification that was actually run
- `tests/test_demo_overlay.py` (17): interpolation safeguards (tracklet/global-id/status/association boundaries, long gaps, disabled), column
  consistency, estimates not counted, unknown never confirmed, non-metric labels, missing frames, server whitelist/gzip/range/404, page contents.
- Headless-Chrome smoke test on the real page and video (`data/processed/vision_eval/m7demo-v1/scripts/smoke_test.mjs`, report in
  `data/processed/demo/117093/screenshots/selftest_cdp.json`): 18 checks (including start offset = measured first-frame PTS) — seeking to 0, 1000, 3000, 2999 (back), 10 (far back), 5975, 1500, 1499, 1500:
  video frame = requested frame = rendered frame = board frame and the detection card equals the overlay data; boxes drawn; local/global toggle;
  untrusted ids never shown as confirmed (28 boxes checked); selection card; 3 s of 1× playback advanced 73 frames with 0 dropped; overlay frame equals video
  frame during playback; interpolation off by default. That self-test only checks internal consistency; the pixel-level proof that the displayed video frame equals the claimed frame is `m7demo-sync-v1/scripts/b1_browser_capture.mjs` + `b2_match_captures.py` (repeatable).
- Independent alignment: frame 3000 decoded with OpenCV straight from the file with the saved boxes drawn
  (`screenshots/alignment_cv2_frame3000_crop.png`).

## Limitations (say these first)
- **Approximate pitch positions only.** No distances, speeds, sprints, workload or fatigue anywhere.
  - Calibration: hybrid fisheye + TPS for this one match; held-out keypoint LOO median 0.45 m / p90 1.07 m; interior n=11: 0.52 / 1.75 m.
  - It was chosen among 4 candidates on the same keypoints, so the error figures are mildly optimistic.
  - Players are on average 9.7 m from the nearest keypoint, and the error grows away from keypoints.
  - Team analytics describe visible players only. Roles were checked visually by the assistant, not by a human labeller: the referee class also fires on sideline staff.
- Global IDs are **hypotheses**. Only 12.6 % of boxes (138 segments) are "attributable"; 15.5 % are confirmed; 56.9 % provisional; 27.6 % unknown/ambiguous
  — by design under the status cap. One tracker ID can span several global hypotheses (e.g. local 25 → 4 tracklets / 4 global ids).
- Interior identity swaps are not detectable (M6.3-D NO-GO). One development match, no independent validation; **no claim of whole-match identity reliability.**
- No ball, no jersey numbers, no player names. Team colour is a track-level appearance cluster, shown only at confidence ≥ 0.7.
- Detections come from the saved run; the demo does not run a detector. The identity solve is offline.
- Tested in Chrome (headless). Playback of a 4096×1080 H.264 stream at 2× depends on the laptop's hardware decoder.

## Mentor demo script, short version (2–3 min, M7 sprint)
1. **(20 s)** "Real SoccerTrack match, saved detections and tracks, our identity layer, replayed on the video's own clock. Sync is verified on decoded pixels: 75/75 frames exact at 0.5×, 1× and 2×."
2. **(40 s)** Point to the board.
   - "This is now a real pitch projection. We tested four calibrations on held-out keypoints. A global homography is 6 m off. Our hybrid fisheye + spline is 0.45 m median, 1.07 m p90."
   - Tick **calibration layer** and show the green and yellow keypoints and the reprojected lines.
   - "Good enough for shape and zones, not for speed. We say so on screen."
3. **(30 s)** People filters.
   - Click **Officials** at about 0:47: L74 at the corner flag.
   - Click **Unknown role**: bench, staff and the goalkeeper in the third colour cluster stay unknown rather than guessed. There is no 22-player cap.
4. **(40 s)** Switch to **Global ID hypothesis** and click local 25 at about 2:22.
   - Read the card: confirmed G127, attributable safe segment.
   - Show the continuity timeline: amber and grey early, green later, boundaries marked.
   - "Only 15.5 % of boxes are confirmed. We show uncertainty instead of hiding it."
5. **(30 s)** Team analytics: zone occupancy per team and team length/width with visible counts. Switch the window to the last 60 s.
   - Point to the red badges: speed, distance and sprint detection are built and tested on synthetic data, but switched off here because the calibration p90 and identity coverage fail our own gates.

Do not claim: distances, speeds, fatigue, xG, validated identity reliability, human-validated role precision.

## Mentor demo script (≈ 6 min, M7-DEMO version)
1. (30 s) "Real SoccerTrack match 117093, saved YOLO detections + ByteTrack tracks + our identity layer. Everything is image-space; nothing here is metric." Point to the two amber badges.
2. (60 s) Press Play. Left: boxes follow players; right: the same players on the board. Say: both are driven by the video's own clock; seek on the timeline and both jump together.
3. (60 s) Toggle **Local tracker ID → Global ID hypothesis**. Explain `G7` solid = confirmed, `G7?` = hypothesis (provisional or via an unvalidated link), `?` = not trusted. Show the colour of the box = identity status.
4. (90 s) Click a player: show the provenance card (local ID vs global hypothesis, tracklet, local vs link confidence, association, safe segment). Use **4× + follow selected** and the trail (pixel trail, safe segment only). Pick local 25 near frame 3550 for a confirmed, long segment; contrast with local 74 near frame 3000 (provisional via an unvalidated link).
5. (60 s) Timeline + insights: coverage over time, 15.5 % confirmed / 56.9 % provisional / 27.6 % unknown. "This is the honest state of identity: good local tracking, cautious global identity."
6. (45 s) Tick **display-interpolated**: dashed `~` boxes fill ≤ 5-frame gaps for smooth viewing, never counted, never measurements. Untick to return to observed-only.
7. (45 s) What is not done: calibration (metric board), ball, interior swaps (tested, NO-GO), independent validation. Next: M7-0 export feeding segment-level analytics, calibration keypoints, validation match.
Do not claim: distances, speeds, sprints, fatigue, xG, validated identity reliability.
