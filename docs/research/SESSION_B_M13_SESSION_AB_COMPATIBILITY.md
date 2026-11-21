# Session B M13: compatibility with Session A's tracklet contract (2026-11-21)

**Documentation and one isolated, synthetic-tested builder only. Nothing in Session A was modified, imported or run.**
Session A's side (read-only): `F:/Football-Fatigue-Project/docs/research/m11_session_ab_interface_v1.md` (contract
v1, message `m11-session-ab-candidate-event-v1`) and `m12_session_ab_compatibility_v1.md` (M12: contract unchanged,
synthetic adapter in `m11_explorer/ab_contract.py`; Session A's learned N×M association added in M12).

## Mapping Session B outputs to the contract

| contract field | Session B source today | status |
|---|---|---|
| `timestamp_s` | `pts_offset_s + frame / fps`, the convention the occlusion engine already uses (117093: pts offset 0.979 s, 25 fps) | ready |
| `pixel.x_px, y_px` | foot point = bottom-centre of the tracked box (`bbox_xyxy_px`) | ready |
| `pixel.box_h_px`, `det_conf` | box height; the matched cached detection's confidence | ready |
| `position_status` | `observed` for every exported sample (tracks hold measured cached boxes only; ByteTrack predictions are not exported) | ready |
| `pitch.x_m, y_m, sigma_m` | **none**: no validated pitch calibration with uncertainty for the SoccerTrack panorama (Session A records ~1 m fixed-calibration bias in M10.7; M8.1 stopped at calibration QC) | **blocked** → `pitch = null`, Session A must abstain |
| `camera.calibration` | `available = false` | blocked |
| `incoming[].identity` | temporary ByteTrack ID → `identity_status = local_only`, `global_id = null` | ready (honest) |
| `outgoing_candidates[].identity` | `anonymous`, `global_id = null` (never pre-filled) | ready |
| `window.per_track_hidden_intervals` | gaps between observed samples of one tracklet (`hidden_intervals`) | ready |
| track quality | detection confidence per sample; pipeline-v1 stage codes are available for *evaluation* only (they use GT and must not enter a request) | partial |
| camera motion | SoccerTrack: fixed panorama (none). Wadi Degla moving camera: no GT, no calibration → out of scope | n/a |

## Rules Session B will keep

1. **Raw pixel coordinates never go into `pitch`.** Pixel-to-metre scale varies strongly across the 4096 px
   panorama, so pixels fed to Session A's pitch-trained models would be silently wrong. `ab_tracklets.tracklets()`
   only fills `pitch` through an explicit `calibrate` function that returns an uncertainty.
2. **Only observed samples.** No interpolation, no Kalman predictions and no GTA "connected" segments are exported as
   observations. A GTA link is an identity hypothesis, not a measurement.
3. **Temporary IDs are not identities.** Every Session B ID is `local_only`. An ID change found by GT evaluation is a
   *Session B evaluation fact* and is never sent as an identity label.
4. **Abstention is valid.** With `pitch = null`, Session A's documented answer is `abstain` (no assignment).
5. **Frame rate and time.** 25 fps for SoccerTrack, from the tracker manifest; any resampling is done by Session A
   on `timestamp_s`.

## What is still missing before an integration test (none of it is in M13 scope)

- A calibrated clip with an uncertainty estimate and independent identity ground truth (Session A blocker 1).
- An agreed video event set for fitting abstention/probability (blocker 2).
- Session B's own identity continuity: today 1,890 temporary-ID changes on 4 dev matches (T0), 733 GT-verified swaps.
  Sending these tracklets would hand Session A wrong "incoming" continuity.

Code: `src/football_fatigue/video/ab_tracklets.py` (no Session A import). Test:
`tests/test_video_m13.py::test_ab_tracklets_are_observed_pixels_only_without_calibration` (synthetic; checks that
predictions are skipped, timestamps use the video clock, `pitch` stays null without calibration, and outgoing
candidates are anonymous).
