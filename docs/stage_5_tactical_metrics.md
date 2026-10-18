# Stage 5 tactical metrics: definitions and implementation requirements

This document fixes the meaning of the requested Stage 5 names. It does not
claim that a metric is available from the current SoccerTrack video demo.

## Definitions selected for this project

| Name | Meaning | What it measures | Minimum inputs |
| --- | --- | --- | --- |
| PC | **Pitch control** | The probability that each team can reach and control the ball at each point on the pitch, based on player location and movement model. | Calibrated 2D player positions, team labels, pitch dimensions, velocity/response assumptions and a validated frame clock. |
| EPV | **Expected possession value** | Expected goals or other chosen terminal reward that the possessing team will obtain from a game state or action. | A declared target, possession and event definitions, calibrated tracking, plus a train/validation split with enough labelled possessions. |
| OBSO | **Off-Ball Scoring Opportunity** | An off-ball player's scoring opportunity through likely ball arrival/control and scoring potential. | Ball position and possession, player/team identities, calibrated tracking, pitch direction, and a documented scoring/control model. |
| DAS | **Dangerous Accessible Space** | The threat-weighted area that an attacking team can access through physically simulated passes and carries. | Calibrated tracking, team/ball state, pitch orientation, pass-speed assumptions, and an explicitly selected value function. |

## Evidence base

- The pitch-control formulation treats control as a race between attacking and defending players after accounting for player motion; its assumptions must be reported alongside any map.
- EPV is a learned value model, not a generic distance-to-goal score. Its result changes with the reward definition and the data used to fit it.
- OBSO combines off-ball reach/control and scoring potential; it needs ball possession and player identities, which a detection-only video pass does not establish.
- DAS is a recent open method that simulates pass completion and combines accessible space with a value function. The published approach is currently limited to low passes when 3D ball data are unavailable.

Primary sources: [Spearman, Beyond Expected Goals (2018)](https://www.researchgate.net/publication/327139841_Beyond_Expected_Goals), [Fernandez and Bornn, Wide Open Spaces (2018)](https://www.sloansportsconference.com/research-papers/wide-open-spaces-a-statistical-technique-for-measuring-space-creation-in-professional-soccer), [Yamamoto et al., OBSO](https://salabufmg.github.io/assets/pdf/papers/obso.pdf), and [Bischofberger and Baca, DAS (2026)](https://link.springer.com/article/10.1186/s40537-026-01387-8).

## Current module status

The SoccerTrack 117093 CV demo has detector boxes, image-space track IDs and
an image-space ball estimate for 30 frames. Its pitch calibration did not meet
the project validation threshold, and player team/identity and possession are
not inferred. Therefore it deliberately publishes no PC, EPV, OBSO or DAS
number.

## What must happen before an implementation is shown as a result

1. Validate the video-to-pitch mapping with independent landmarks at an agreed
   error threshold. Keep frames that fail the check out of metric computation.
2. Establish stable player team attribution, pitch attacking direction, and a
   ball/possession state with an audited error report.
3. Choose one published formulation per metric, freeze its assumptions and
   parameter values, and record them in every output manifest.
4. For EPV, train only on a defined labelled dataset and evaluate on held-out
   matches. For PC, OBSO and DAS, validate maps or pass predictions against
   independent held-out events.
5. Expose scores with their coverage and uncertainty; never turn unavailable
   inputs into zero-valued tactical claims.
