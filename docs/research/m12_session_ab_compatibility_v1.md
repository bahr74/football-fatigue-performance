# M12 report 9: Session A / B compatibility (v1)

Status: contract kept unchanged (v1, `m11_session_ab_interface_v1.md`); a synthetic-only contract adapter and tests were added. **No
Session B file was read for this work or modified; no integration was attempted.** Date 2026-11-21.

## What was added

`src/football_fatigue/m11_explorer/ab_contract.py` is the Session A side of the contract, tested only on synthetic requests:

- `request_to_scenario(request)`:
  - accepts a `m11-session-ab-candidate-event-v1` request and checks its schema;
  - **refuses** (`ContractError`) an outgoing candidate whose `global_id` is pre-filled or whose status is not `anonymous`;
  - keeps **only `observed`** pitch samples;
  - computes the domain check (`calibration_ok`, `sigma_m_p90`, `inside_trained_envelope`, `reasons`).
- `respond(request)`:
  - scores with the bidirectional Kalman d2 and solves with Hungarian with null (gate χ²(2, 0.99));
  - computes per-link margins from full alternative assignments;
  - abstains when a margin is < 1 d2 unit (an **uncalibrated placeholder rule**, labelled as such in the message) or when the
    pitch input is unusable;
  - `probability` is always null because nothing is calibrated;
  - writes a `m11-session-ab-reassociation-result-v1` message.
- `check_result(result, request)` verifies:
  - one-to-one assignments;
  - ids inside `candidates_scored`;
  - no probability while uncalibrated;
  - an abstain result lists every candidate as unmatched;
  - every candidate appears exactly once (assigned or unmatched);
  - no global-id propagation outside the trained envelope;
  - the event id is echoed.
- `synthetic_request(level, seed, ...)` builds a synthetic request from `multiplayer.generate` (truth returned separately, never in
  the request). Its incoming tracks carry B-side `interpolated` samples placed 1 km away: if any interpolated sample were used, the
  result would change.

## Tests (`tests/test_m11_nxm.py`, contract part, synthetic only)

| Test | What it guarantees |
|---|---|
| Round trip valid, interpolated ignored | The result passes every invariant; the decision is identical with and without the B-side interpolated samples |
| Pre-filled outgoing id / wrong schema | Refused with `ContractError` |
| Noisy input (σ 0.6 m) | `inside_trained_envelope = false`, no global-id propagation, still a valid message |
| No calibration | `abstain`, every candidate unmatched, valid message |
| Levels 1, 3, 5, 7 | Invariants hold from 2×2 to crowded scenes |

## Interface semantics (what B must provide and what A returns)

| Topic | Contract position |
|---|---|
| **Calibration** | A consumes metric pitch coordinates only. `camera.calibration.available = false` or `pitch = null` gives `abstain`. Calibration with an uncertainty estimate is a hard prerequisite; earlier work found about 1 m fixed-calibration bias (M10.7) |
| **Temporal alignment** | A uses `timestamp_s` (video clock), never frame indices. Pairs where the outgoing track starts before the incoming one ends are infeasible (one player cannot be on two tracks). Asynchronous loss is described per track (`per_track_hidden_intervals`) |
| **Tracklet format** | Parallel arrays per tracklet; only `position_status = observed` samples are model input. Interpolated and predicted samples are ignored (tested) because they would leak the hidden interval |
| **Identity semantics** | Incoming identities are *known before* with B's status; outgoing candidates are *anonymous*. A returns links, not identities; the propagated `global_id` is a suggestion unless the incoming id is `confirmed`, the link is not abstained and the input is inside the envelope |
| **Confidence** | `margin` = cost of the best alternative full assignment minus the chosen one (d2 units). `probability` stays null until fitted on a separate calibration partition of *video* events |
| **Visibility evidence** | Box overlap and track loss are trigger evidence only; `verified_visual_occlusion` is null unless a human verified it. ESTIMATED provider positions are not visibility evidence |
| **Domain shift** | All Session A models were built on provider tracking (centimetre noise, ESTIMATED interpolation). The envelope check flags σ p90 > 0.15 m. The audit (report 2) shows learned two-player models partly read provider interpolation texture, which video will not have. Therefore deterministic motion baselines go first in any integration test |

## Still not compatible (blocking integration)

1. No calibrated video clip with independent identity ground truth has been agreed.
2. The abstention threshold and any probability calibration must be fitted on video events, not on provider data.
3. Learned scorers are trained on provider data and are out of domain for video. Results from them must be labelled exploratory.
4. Long absences (re-entry) are outside motion-only association: they need `delayed` plus appearance or jersey evidence.
