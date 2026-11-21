# Advisor meeting brief (28 Oct 2026)

**Full report:** [ADVISOR_PROGRESS_REPORT.md](ADVISOR_PROGRESS_REPORT.md)

**Demo:** `.\.venv\Scripts\python.exe -m football_fatigue.demo`, then open http://127.0.0.1:5790/.

## The idea in one sentence
Measure how football players move during a match (from tracking data and from a single panoramic video), show it to coaches on a synchronised tactical board, and, in later project stages, use reliable movement data for performance and fatigue research.

## My responsibility (Task 2: Software Architecture & Tactical Board Streaming)
- The software foundation.
- Data platform for professional tracking data (FIFA World Cup 2022) with a tactical board.
- Video pipeline: detection → tracking → player identity → team/role → pitch metres → tactical board → coach analytics.

**Not my task / not built:** fatigue models, physiological or wearable data, injury risk.

## Completed
- Tracking-data platform: validated loading, event synchronisation, tactical board, movement and spatial summaries.
- Video pipeline on real annotated footage (SoccerTrack v2, 4-minute panoramic clips), evaluated with standard tracking metrics.
- Coach dashboard: synchronised video + board, identity status, team filters, team shape, coach-configurable physical settings, human identity review, exports.
- M7.3 (just finished): diagnosis of implausible speeds; a much better motion filter; a test of automatic gap joining (rejected → review suggestions).

## What I can demonstrate
1. Video with boxes, pixel-synchronised with the top-down board.
2. Identity status on every player (confirmed / provisional / unknown).
3. Team filters and team shape over time.
4. The physical analytics panel: change a sprint threshold and recalculate. The values are **withheld**, and the panel shows why.
5. Suggested identity hand-overs that a reviewer can accept or reject.

## 5 most important measured results
| # | result | type |
|---|---|---|
| 1 | Video and board pixel-synchronised: 75 of 75 test frames exact (was 1 of 73 before a 1-s offset fix) | real footage |
| 2 | Pitch calibration: 6.3 m → **0.45 m** median error (1.07 m at the 90th percentile) on held-out landmarks | real footage, 1 match |
| 3 | Wrong-team assignments: 4–18 % → **0.3–3.3 %** on 4 development matches | development |
| 4 | Confirmed identities are 99.5 % or more correct, but only **13–20 %** of boxes are confirmed. On one independent match the earlier system reached 96–97 % at 25–36 % coverage, **below target** | development / independent |
| 5 | Motion filter: speed error **8.0 → 2.8 km/h**, true sprints detected 3 % → 75 % on simulated runs with real noise; on the real match, implausible peaks 20.5 % → 3 % | synthetic / plausibility |

## 5 most important technical problems
1. **Identity swaps** when players cross. Team-mates look alike at 30–50 px.
2. **Fragmentation:** about half of a player's boxes fall outside their main identity.
3. **Foot-point jitter:** 1 px is about 0.5 m on the far side, which creates false speed.
4. **One calibrated match; short clips** (about 4 minutes): no full-match data.
5. **No independent validation of the current identity system**; staff and referees contaminate detections.

## Why real speed and sprint values are still blocked
- Our safety test has two parts. The first (implausible peaks) now passes; the second (noise in the input positions) fails: 1.74 against a limit of 1.15.
- Smoothing cannot fix input noise. We did **not** relax the threshold, because a lower bar would not make the measurement more accurate.
- Coach sprint thresholds only define what counts as a sprint; they do not make the speed accurate.

## Next steps
1. Measure foot-position noise directly. Pre-declare a revised, evidence-based safety test and check it on unused footage.
2. Collect a first set of human identity reviews with the new suggestion tool.
3. Improve interior calibration; calibrate a second match.
4. Later: ball and possession; integration with fatigue research data; final evaluation on reserved matches.

## Questions for the advisor
1. Is a conservative "withhold until validated" policy acceptable for the thesis, or should we report diagnostic values with clear warnings?
2. Should we spend the two reserved validation matches now on the current identity system, or first improve it?
3. Can the project obtain any reference data (GPS or wearable from a club, or manually labelled foot positions) to validate speed on real footage?
4. How should Task 2 outputs be handed over to the fatigue-research part: which variables, in which format, with which uncertainty fields?
5. Is a manual review step (a human confirming identities) acceptable in the intended workflow?

## 60–90 second speaking script
> "Our project studies football player performance and workload. In the long term this should support fatigue research. My part, Task 2, is the software foundation: turning match data and match video into player positions, identities and a tactical board for coaches.
>
> We built two connected parts. First, a validated platform for professional tracking data with a tactical playback board. Second, a video pipeline. It detects players in panoramic footage, tracks them, estimates who is who, assigns teams, maps them to pitch metres and shows everything on a board synchronised exactly with the video.
>
> We evaluate on a public dataset with annotated 4-minute clips, using standard tracking metrics, and we keep separate matches for independent testing. Synchronisation is pixel-exact. Pitch calibration is about half a metre median error. Team errors dropped to about 3 percent.
>
> The main open problems are player identity when players cross, and noise in the foot position, which inflates speeds. This month we built a filter that cut simulated speed error from 8 to under 3 kilometres per hour. Real speeds are still withheld, because one safety check on input noise fails, and we chose not to relax it.
>
> Next we will measure that noise directly, pre-declare a fair validation test, and start human identity review. We do not claim fatigue measurement; these are workload descriptors that later modules can build on."
