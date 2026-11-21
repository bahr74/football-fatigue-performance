# Session B · V1 Step 2 — occlusion event evaluation (development data)

Date: 2026-11-17 · engine 0.2.0 · evaluation `eval-v1` · branch `feature/video-occlusion-events` (uncommitted)

## What was evaluated

* **Inputs (identical for both profiles):** existing detection caches v2 and the production ByteTrack replays of the four
  development matches. No GPU inference, no new detections, and no change to the tracker.
* **Profiles:**
  * `step1`: the Step 1 engine, unchanged. On the Step 1 window its events are byte-identical to the Step 1 run, apart
    from run/version metadata.
  * `v2`: the Step 2 changes listed below.
* **Ground truth (GT):** SoccerTrack MOT boxes.
  * 22 players only, with no referees, staff or spectators.
  * Template-sized boxes projected from positions, so "occlusion" in GT means *geometric projection occlusion*. This is
    a proxy for visual occlusion, not a label of it.
* **Split:**
  * **tune**: 117093 and 118575 (thresholds were looked at and changed here);
  * **check**: 117092 and 118576 (scored once with the frozen `v2` config `ff8815ed`).
  * All four are development data; **the check matches are not independent final validation**.
  * The reserved sequences (118577, 118578, 128058, Tier 2) were not accessed.

## Method (`src/football_fatigue/video/occlusion/evaluate.py`)

* **GT episodes:** formed per GT player pair, with far = the smaller bottom y. There are two tiers:
  * **overlap**: the near box covers ≥ 20% of the far box for ≥ 2 frames;
  * **occlusion**: the near box covers ≥ 50% of the far box for ≥ 3 frames.

  Gaps of ≤ 2 frames are merged.
* **Track → GT identity:** the majority GT match (IoU ≥ 0.3) of the track's own *observed* boxes within ±10 frames of
  the event. Predicted boxes are never used, because they sit on the occluder by construction.
* **Event outcomes:**
  * `tp`: the same GT pair has an episode of the same tier within ±5 frames;
  * `fp_gt_overlap_only`: GT shows only 20–50% cover;
  * `fp_no_gt_overlap`;
  * `fp_duplicate_track`: both tracks map to one GT player;
  * `non_gt_person`: a member is not a GT player. These events are **not scored**, because referees and staff cannot be
    judged against this GT.
* **Event-level P/R:** precision counts events; recall counts GT episodes, over *detectable* episodes (both players
  tracked within ±10 frames).
* **Claim-level precision:** every pair an event asserts is scored on its own. A pair is asserted when a member is hidden
  behind an occluder, or when a strongly overlapped far player is degraded. This metric is independent of how events are
  grouped, so it is the fair before/after comparison: Step 1 chains many pairs into one event, which inflates its
  event-level precision.
* **Detection losses:** scored by whether the GT player is ≥ 50% covered during the gap (`actually_occluded_in_gt`) or
  not (`true_loss`). "loss-not-occl P" = true_loss / (true_loss + actually_occluded_in_gt). It checks that losses are not
  occlusions in disguise.
* **Timing:** the median |start/end error| in frames for `tp` probable occlusions.

## Changes in profile `v2` (the targeted improvements)

| Issue (from Step 1 review) | Change | Effect measured |
|---|---|---|
| Off-pitch staff, bench and spectators | Play-area polygon from SoccerTrack's annotated pitch landmarks (not GT boxes). Tracks whose foot point is > 1.5 box heights outside it in ≥ 70% of frames are off-pitch; their events go to `suppressed.jsonl` (kept for review). | 86 / 40 / 236 / 3 events suppressed (117093 / 118575 / 117092 / 118576). On the tune matches **all 126 suppressed events involved only non-GT people** (checked against GT). |
| Long chained crowd events | One event per player pair, plus a `group_id` that links concurrent pairs sharing a track. A hidden player is re-attributed to a new occluder only when it covers ≥ 50% of the prediction (tuned: was ≥ 20%). | \|end error\| 11–15.5 → 5–8 frames on all four matches. Box-overlap recall ×1.6–2.5. |
| Long-gap extrapolation onto empty grass | Damped prediction: median velocity capped at 0.25 box heights per frame, applied for 5 frames, then held. | Not isolated by an ablation; no measurable change in the scores. |
| Detector misses vs tracker gaps | Kept the Step 1 split; it is now scored. | See the detection-loss section below. |

## Results (full matches)

| match | role | profile | events | PO ev | PO P | PO R | PO F1 | claim P (occl) | \|start err\| | \|end err\| | BO ev | BO P | BO R | claim P (ovl) | occl recall (any) | loss-not-occl P |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 117093 | tune | step1 | 2675 | 242 | 0.733 | 0.706 | 0.719 | 0.666 | 6 | 13 | 159 | 0.848 | 0.144 | 0.911 | 0.740 | 0.988 |
| 117093 | tune | v2 | 3070 | 493 | 0.706 | 0.648 | 0.676 | 0.663 | 6 | 6 | 386 | 0.871 | 0.359 | 0.908 | 0.751 | 0.989 |
| 118575 | tune | step1 | 2755 | 204 | 0.790 | 0.695 | 0.739 | 0.677 | 6 | 11 | 173 | 0.913 | 0.307 | 0.893 | 0.752 | 0.988 |
| 118575 | tune | v2 | 2948 | 299 | 0.703 | 0.695 | 0.699 | 0.679 | 5 | 5 | 309 | 0.888 | 0.482 | 0.893 | 0.771 | 0.989 |
| 117092 | check | step1 | 3362 | 236 | 0.697 | 0.725 | 0.711 | 0.571 | 9 | 14 | 125 | 0.767 | 0.179 | 0.790 | 0.774 | 0.986 |
| 117092 | check | v2 | 3470 | 423 | 0.571 | 0.682 | 0.622 | 0.537 | 7 | 8 | 283 | 0.785 | 0.340 | 0.803 | 0.768 | 0.986 |
| 118576 | check | step1 | 1925 | 208 | 0.894 | 0.804 | 0.846 | 0.708 | 8 | 15.5 | 165 | 0.895 | 0.205 | 0.896 | 0.840 | 0.986 |
| 118576 | check | v2 | 2451 | 458 | 0.742 | 0.767 | 0.754 | 0.687 | 6 | 6 | 448 | 0.870 | 0.479 | 0.904 | 0.903 | 0.989 |

PO = probable occlusion vs detectable GT occlusion episodes; BO = box overlap vs detectable GT overlap episodes; occl
recall (any) = share of detectable GT occlusion episodes matched by an event of either class.

## Reading the results

1. **v2 is a structural, timing and readability improvement, not a precision improvement.**
   * Claim-level occlusion precision is unchanged on the tune matches (0.666 → 0.663, 0.677 → 0.679). On the check
     matches it is slightly lower (0.571 → 0.537, 0.708 → 0.687).
   * Event-level PO precision falls more (for example 0.894 → 0.742 on 118576). That is mostly because pair events no
     longer hide wrong pairs inside one correct chain, so the event-level number overstates the loss.
2. **Clear gains:**
   * end-of-event timing (median |end error| 13–15.5 → 5–8 frames);
   * box-overlap recall (0.14–0.31 → 0.34–0.48);
   * occlusion episode recall (any class) is equal or better on three matches and −0.006 on 117092;
   * crowd events are now reviewable pairs.
3. **The remaining false positives are dominated by two categories** (claim level, v2, all matches):
   * `gt_overlap_only` (GT shows 20–50% cover, the engine says occluded). This is partly the template-box proxy:
     template boxes are narrower than real players.
   * `duplicate_track` (two tracks on one player: duplicate boxes or a re-acquired track). This is a detector/tracker
     artefact the engine currently trusts.

   Genuine "no GT overlap" errors are rare (3–18 per match).
4. **Detection loss ≠ occlusion:** 98.6–98.9% of scored detection-loss events are *not* occlusions in GT. The engine does
   not mistake the clear losses for occlusions.
5. **Detector miss vs tracker gap:**
   * Detector misses are mostly real GT players: 1,175 of 1,574 on 117093 (v2); the rest are non-GT people.
   * Tracker gaps are dominated by non-GT people on 117093 (402 of 508), but much less so on 118575 (403 true losses vs
     250 non-GT).
   * The play-area filter removes the bench and spectators, but not referees and linesmen on the pitch.
6. **Missed GT occlusions** (v2): 43–126 per match (117093 126, 118575 66, 117092 107, 118576 43). Most of them (76–100%
   per match) are "tracked but no event": both players were tracked, but the cached boxes never overlapped enough or the
   hidden player was never lost. The remainder never had a track (26 on 117092). The merge-rule (NMS) comparison is the planned next lever for the first group.

## Limitations

* The GT is a geometric proxy (template boxes, no referees), so precision is a lower bound for visual occlusion and the
  `gt_overlap_only` errors include proxy disagreement.
* The tune/check split is development-internal; nothing here is final validation.
* `non_gt_person` events (referees, linesmen, false detections) are excluded from precision; they are 10–25% of PO events.
* The engine's evidence level is a grade, not a calibrated probability.
* The damped predictor and the play-area filter were not ablated separately (the play-area effect was counted directly).

## Reproduce

```powershell
$env:PYTHONPATH = "F:\Football-Fatigue-Video\src"
$PY = "F:\Football-Fatigue-Project\.venv\Scripts\python.exe"
foreach ($s in "soccertrack-117093","soccertrack-118575","soccertrack-117092","soccertrack-118576") {
  foreach ($p in "step1","v2") { & $PY -m football_fatigue.video.occlusion run --data-root F:\Football-Fatigue-Project\data --source $s --profile $p --no-render }
}
Get-ChildItem data\processed\video\*\occlusion\0.2.0\occl-* -Directory | ForEach-Object {
  & $PY -m football_fatigue.video.occlusion evaluate --data-root F:\Football-Fatigue-Project\data --run-dir $_.FullName }
& $PY -c "from football_fatigue.video.occlusion.report import collect, markdown; print(markdown(collect('data/processed/video')))"
```

Each run takes about 20 s per match on CPU (engine ≈ 0.3–0.4 ms/frame); each evaluation takes about 20 s. Run folders
are write-once.
