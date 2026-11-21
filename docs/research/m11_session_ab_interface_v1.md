# Session A / Session B integration contract, v1 (proposal)

**Status:** a design proposal for discussion; nothing is implemented.
- No Session B file was read beyond the published M7-0 identity-export description (`docs/identity_export.md`), and none was changed.
- Every JSON example is synthetic: see `m11_session_ab_interface_v1_examples/`.
- `tests/test_m11_phase2.py` checks that the examples parse and satisfy the invariants of §4.

**Scientific boundary, stated first:**
- The Session A models were trained on **PFF metric pitch coordinates** of FIFA World Cup 2022 tracking, resampled to 15 Hz.
- They were not trained on raw pixel box centres, and they were never evaluated on video-derived positions.
- **They are not plug-and-play with video tracking.** §5 lists what blocks direct use today.

## 1. Pipeline and responsibilities

```
Video frames ─▶ [B] detection + tracking ─▶ [B] candidate-event trigger ─▶ request (§3.1)
                                                                           │
             [A] domain check ─▶ [A] link scoring ─▶ [A] assignment with null / abstention / delay ─▶ result (§3.2)
                                                                           │
                                         [B] apply or ignore ─▶ recovered tracks (identity provenance kept)
```

| Step | Owner | Output |
|---|---|---|
| Detection, local tracking, tracklet ids | B | `frames.jsonl` (existing) |
| Pitch calibration and pixel-to-pitch projection with uncertainty | B (calibration) | `pitch.x_m`, `y_m`, `sigma_m` per sample |
| Event trigger: fragmentation, simultaneous loss, suspected occlusion | B | candidate-event request |
| Candidate construction (which incoming / outgoing / dormant tracks are scored) | B proposes; A may prune | `incoming`, `outgoing_candidates`, `dormant_tracks` |
| Domain check, link scores, assignment, abstention, delay | A | result |
| Whether a result changes a global id | B, under its identity policy | recovered tracks |

**Session A never writes Session B state.** B applies a result only if its own trust rules allow it, for example the M7-0 rule that a
global id is attributable only for `confirmed` samples.

## 2. Terms (aligned with the dataset v2 taxonomy)

| Term | Meaning here |
|---|---|
| Tracklet | A maximal run of one local track id with no identity-relevant break (M7-0 segment semantics) |
| Incoming tracklet | Observed **before** the event; may carry a global id and an identity status |
| Outgoing candidate | Observed **after** the event; **anonymous** by contract, since its id is exactly what is being inferred |
| Dormant track | A global id with no live tracklet, for example after leaving the camera view; it can be re-claimed by an outgoing tracklet |
| Hidden interval | Per-track time without observations. Tracks may disappear and reappear **asynchronously** |
| Trigger evidence | Box overlap, track loss, low detection confidence. **Not proof of visual occlusion** (§5.3) |

## 3. Messages

### 3.1 Candidate-event request (`m11-session-ab-candidate-event-v1`), B → A

**Example:** `01_candidate_event_request.json`

| Field | Required | Notes |
|---|---|---|
| `event_id` | yes | Stable; unique per match. B's id is reused in the result |
| `match` | yes | `match_id`, `period`, `fps`, `clock` (`video_time_s`). Frame indices and timestamps are both carried; A uses timestamps |
| `camera` | yes | `camera_id` and `calibration` {`available`, `calibration_id`, `model`, `reprojection_error_m`, `valid_region`, `time_varying`} |
| `trigger` | yes | `type` ∈ {`track_fragmentation`, `simultaneous_loss`, `id_conflict`, `reentry`, `manual`}, `reasons`, `occlusion_candidate` {box IoU, foot-point px distance, `verified_visual_occlusion`: null unless a human verified it} |
| `window` | yes | `hidden_interval_s` (the union) **and** `per_track_hidden_intervals` (asynchronous loss) |
| `incoming[]` | ≥ 1 | Per tracklet: `tracklet_id`, `identity` {`global_id`, `identity_status` ∈ {`confirmed`, `linked_unvalidated`, `local_only`}, `identity_confidence`, `source`}, `observations` |
| `outgoing_candidates[]` | ≥ 0 | The same, with `identity.global_id = null` and `identity_status = "anonymous"`. **B must not pre-fill an id here** |
| `dormant_tracks[]` | ≥ 0 | `global_id`, `last_seen_s`, `last_pitch_m`, `left_camera_view` |
| `context_players[]` | ≥ 0 | Nearby tracked players (group context, possible distractors) |
| `appearance` | yes | `available` and `reason`. ReID embeddings, when present, are optional extra evidence; A's motion models do not use them |
| `provenance` | yes | Hashes of the tracker output and configuration |

**`observations`** are parallel arrays, one per sample:
- `frame`, `timestamp_s`;
- `pixel` {`x_px`, `y_px` (foot point), `box_h_px`, `det_conf`};
- `pitch` {`x_m`, `y_m`, `sigma_m`}, `null` when uncalibrated;
- `position_status` ∈ {`observed`, `interpolated`, `predicted`}.

**Only `observed` samples may be used as model input.** Interpolated samples inside the hidden interval would leak exactly the
quantity being inferred.

### 3.2 Re-association result (`m11-session-ab-reassociation-result-v1`), A → B

**Examples:** `02_reassociation_response.json` and `03_abstain_and_delayed.json`

| Field | Notes |
|---|---|
| `producer` | Model or baseline version and bundle hash; `trained_domain` stated verbatim |
| `input_contract.domain_check` | `calibration_ok`, `sigma_m_p90`, `inside_trained_envelope`, `reasons`. Outside the envelope, results are advisory only |
| `link_scores[]` | Per (incoming, outgoing) pair: `score_kind` (for example `kalman_bidir_nll`, or a model logit), `value`, `inside_gate` |
| `decision.status` | `assigned` / `partial` / `abstain` |
| `decision.solver` | `hungarian_with_null` (pairs above the null cost are never chosen) |
| `decision.assignments[]` | `incoming`, `outgoing`, the propagated `global_id`, `probability` (null unless calibrated on a separate calibration partition), `margin` |
| `unmatched_incoming[]`, `unmatched_outgoing[]` | With `reason` ∈ {`outside_gate`, `abstained`, `no_candidate`, `dormant`} |
| `decision.abstention` | Rule and reason. Abstention is a first-class output, not an error |
| `decision.delayed` and `delay` | Re-open until a time, waiting for listed evidence (a second outgoing tracklet, OCR, re-entry) |
| `identity_effect` | Whether the result may propagate a global id under B's policy, and why not if not |

## 4. Invariants (checked on the examples by the tests)

1. Assignments are one-to-one, and every id in them appears in `candidates_scored`.
2. Every outgoing candidate in a request is anonymous (`global_id = null`).
3. `margin` equals the cost of the best alternative full assignment minus the cost of the chosen one, from `link_scores`.
4. `probability` is null when `calibrated` is false.
5. An `abstain` result has no assignments and lists every candidate as unmatched.
6. `synthetic_example` is true in every file of the examples folder.

## 5. What prevents direct integration today

### 5.1 Coordinate domain and calibration

- Models consume metric pitch positions in a sample frame, with 15 Hz resampling and ≥ 3 observed points per tracklet in 2 s
  (Stage 3 contract).
- Video gives pixel foot points. Metres per pixel vary strongly across a panorama, and most matches have no pitch calibration
  (`identity_export.md`, "Known limitations").
- Earlier calibration work (M8.1) stopped at calibration QC. M10.7 found a fixed-calibration bias of about 1 m (project memory /
  handoffs). **A calibration with an uncertainty estimate is a hard prerequisite.** Without it, every positional model input is
  undefined.

### 5.2 Noise and domain shift

**Measured on the development data in Phase 2 (Task B):**
- On the PFF tracklets, the label-free Kalman selection prefers a measurement noise of **0.05 m** and white-acceleration noise of
  **8 m/s²**.
- Video foot points projected through a homography carry decimetre-to-metre errors that vary with distance from the camera
  [hypothesis, to be measured].
- M2 scenario testing already showed degradation under added noise and dropout [DOC].
- The provider texture probe showed that models partly exploit ESTIMATED interpolation smoothness [DOC], **which video data will not
  have**.

**Rule:** a request whose `sigma_m` or `per_track_hidden_intervals` fall outside the validated envelope receives
`inside_trained_envelope = false`, and its result is advisory only.

### 5.3 Box overlap is not occlusion

- Overlapping boxes, or a lost track, are consistent with occlusion, detector failure, a player leaving the view, a camera cut or
  tracker fragmentation.
- The interaction catalogue likewise shows that proximity on the pitch is not visibility: close contacts are mostly provider
  ESTIMATED.
- `verified_visual_occlusion` therefore stays `null` unless a human verified it. Phase 2 performed no such verification.

### 5.4 Identity known before and inferred after

| | Known before (incoming) | Inferred after (outgoing) |
|---|---|---|
| Source | B's identity status and confidence (`confirmed` or not) | Session A's assignment |
| Can be wrong? | Yes: an unconfirmed incoming id makes any propagated id unconfirmed | Yes: errors compound with incoming errors |
| Propagation rule | — | Only `confirmed` incoming × an assignment inside the gate × not abstained × inside the envelope. Otherwise suggestion only |

### 5.5 Candidate construction

- B proposes the incoming tracks lost within Δt of each other, plus the outgoing tracklets born within the search radius and time
  window. The radius should grow with time (√t-type growth, or the Kalman covariance), not stay fixed.
- A adds dormant ids whose last position is reachable at a plausible speed (≤ 12 m/s).
- **The candidate set must not be filtered using the eventual answer.**
- N × M with null is the general case; 2 × 2 is a special case. The Phase 2 Hungarian-with-null solver
  (`m11_phase2/baselines.py::assign`) already handles it.

### 5.6 Players leaving the camera view

- The track becomes dormant, with `left_camera_view = true`. Its re-entry is a separate `reentry` event.
- Pure motion association across long absences is not supported by any Session A evidence: Stage 3 gaps are 0.4 to 1.6 s.
- Such events should be `delayed` and resolved with appearance or jersey evidence, or left unresolved.

### 5.7 Asynchronous and partial loss

- Stage 3 hides both players over the **same** interval.
- Video loss is asynchronous: one player may never disappear, and extra or missing detections occur.
- These cases are outside the trained distribution until a dataset v2 or `multiplayer-association-v1` population covers them
  (`m11_dataset_v2_design.md`).

## 6. Minimal path to a first integration test (proposal, needs approval)

1. Choose one video clip with an independent identity ground truth (for example the SoccerTrack v2 local GT [DOC]) and a validated
   calibration.
2. Have B emit requests for every fragmentation event. Score them with **deterministic Kalman / Hungarian baselines first** (no
   domain-transfer claim needed), then with the frozen models labelled exploratory.
3. Report:
   - coverage;
   - abstention rate;
   - assignment accuracy against the video GT;
   - the share of events inside the trained envelope.

   Never report PFF development accuracy as video accuracy.
