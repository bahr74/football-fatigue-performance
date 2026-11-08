# M1 – Possession module plan

## Goal
A reusable module that takes RAW tracking (ball + players per frame) and outputs, for every frame,
one of four states, with player, team and confidence:
1. DEAD – ball out of play: any PFF OUT (outType T = over a line, W = whistle, H/A = goal), or end of a
   half (END), until the next restart starts; also before each period's kick-off.
   NOT dead: advantage played, ball hits woodwork/corner flag and stays in play (PFF G event).
   Unknown (not dead): videoMissing stretches.
2. CONTROL – a player has the ball (a PFF OTB possession spell, except rebounds).
3. TRAVELLING – ball moving between players after a pass, cross or shot.
4. LOOSE – nobody controls it: after a rebound, block, parry, deflection, challenge or clearance (FIFA calls this "In Contest").
The four states match FIFA's official World Cup possession states (Team A, Team B, In Contest, Out of Play).

## Hard rules
- MODEL INPUT = raw tracking only (balls, homePlayers, awayPlayers), cleaned by OUR smoother.
- NEVER use as input: PFF event tags, game_event / possession_event fields in tracking records,
  ballsSmoothed, homePlayersSmoothed, awayPlayersSmoothed. PFF smoothed data is built from PFF's tags
  (ball pinned to the tagged carrier, removed in dead ball) = leakage.
- PFF tags and PFF smoothed data may ONLY be used to build the answer key and to evaluate.
- Study and tune on DEV_MATCHES only (3812, 3813, 3814, 3815, 3818, 3819, 3820, 3821), rotating which
  matches are studied vs checked. Test matches (the other 56) are run ONCE at the very end.
- Rules must follow PFF definitions (PFF FC Event Data Specification v2.5): control vs rebound (4.6),
  challenges (4.4: duel winner is not the possessor), touches (4.5), game events OTB / OUT (2).
- The module must be standalone: no imports from the rest of this project, input = arrays in the
  tracking contract (frame, time, period, ball x/y/z, players x/y with team and id).
- Do not change existing project code.

## Reference methods
- Link & Hoernig 2017, "Individual ball possession in soccer" (PLOS ONE): possession starts when the
  ball is near the player AND a kick shows as a ball acceleration peak (>= 4 m/s²); ends when the player
  can no longer reach the ball's position predicted 1 s ahead; RTS smoother on the ball.
- Vidal-Codina et al. 2022, "Automatic event detection in football using tracking data": possession
  zone around players (0.5–1 m depending on provider), duel zone 1 m around the ball (two opponents
  inside = duel, overrides possession), possession only counts if the ball changes direction or speed;
  dead-ball restarts from player patterns.
- databallpy (implements Vidal-Codina; defaults 1.5 m, 10°, 5 m/s) – a baseline to beat.

## Time tolerance (fixed in Step 1; also used by Step D scoring)
- PFF tag times vary by up to about ±0.8 s around the physical moment (Step 0 follow-up).
- Answer-key labels use the exact PFF times; they are never shifted.
- Every comparison between labels and tracking (ball near the player, a kick near a release, and Step D frame
  scoring at state changes) uses a tolerance of **±0.3 s (±9 frames at 29.97 fps)**: `m1_labels.TOLERANCE_S`.

## Steps (one at a time; stop and report after each)
0. Smoother – clean the raw ball without tags; score it against PFF's smoothed ball. (done)
1. (A) Answer key – labels v1 for all 64 matches from PFF tags; lock the 16 test matches and the 6 learning folds;
   run C0 (naive) and C1 (paper) on the dev matches with the scoring list of the Approach section.
2. (B) C2 full rules – implement rules A–D; tune every threshold within table T1 by 6-fold CV on the 48 learning matches.
3. (C) C3 gradient boosting – features from the same rules plus a football-logic decoding layer; same folds.
4. (D) Score – C0, C1, C2 and C3 on the same answer key, frames and folds; then the 16 locked test matches, once.

## Approach (decided 2026-11-06)

Contestants – all scored on the same answer key (labels v1), the same frames and the same folds:
- C0 Naive: CONTROL = nearest player within 1.5 m of our smoothed ball; DEAD = our ball missing; TRAVELLING = ball > 5 m/s and nobody within 1.5 m; LOOSE = the rest.
- C1 Paper: Vidal-Codina et al. 2022 as in databallpy, with the published values: possession zone 1.0 m, duel zone 1.0 m, direction change 10°, speed change 5 m/s, spatial threshold 10 cm, plus the restart tolerances in table T2.
- C2 Full rules (below), every threshold tuned within the ranges in table T1.
- C3 Gradient boosting (LightGBM or XGBoost) on features built from the same rules, plus a football-logic decoding layer (an HMM/Viterbi or a state machine). Deep sequence models only later, if C3 plateaus.

Data split:
- 16 matches = final test, locked, run once at the end.
- 48 matches = learning, with 6-fold cross-validation by match (every match is scored by a model that never saw it).
- The 8 DEV_MATCHES stay inside the learning set as the inspection set.
- Choose the 16 test matches at random with a fixed seed, and record the list here.

Inputs (all contestants): our smoothed raw ball (with observed / interpolated / estimated / missing flags) and raw player positions. Never PFF tags or *Smoothed fields.

Scoring (all contestants):
- 4×4 confusion matrix (DEAD, CONTROL, TRAVELLING, LOOSE) on scored frames;
- player accuracy and team accuracy inside CONTROL;
- possession-change F1 within ±0.3 s;
- dead-ball start/end F1 within ±0.3 s and ±2 s;
- a breakdown by situation: restart types, whistles, duels, keeper, one-touch passes, blocks, missing-data stretches.

Source tags:
- [PFF] = PFF Event Data Specification v2.5
- [LH] = Link & Hoernig 2017
- [VC] = Vidal-Codina et al. 2022 (incl. supplementary S4.2)
- [KM] = Khaustov & Mozgovoy 2020
- [MO] = Morra et al. 2020
- [ET] = ETSY, Van Roy et al. 2023
- [EL] = ELASTIC 2025
- [MU] = Müller et al. 2022 (in-game status from player movement)
- [LAW] = IFAB Laws of the Game
- [GUESS] = our starting guess, no published value

Note: the published values come mostly from optical stadium tracking. PFF is broadcast tracking, and even a player in control sits a median of ~1.2 m from the raw ball. So the search ranges below are deliberately wider than the papers.

T1 – C2 thresholds (start value, search range, source)

| Parameter | Start | Range | Source |
|---|---|---|---|
| Control radius (possession zone) | 1.5 m | 0.5–3.0 m | [VC] 0.5–1.5 m (1 m optical), [KM] 0.9–1.2 m, databallpy 1.5 m |
| Touch radius (foot / pass-like) | 2.5 m | 1.0–3.5 m | [ET] ≤ 2.5 m, [EL] 3 m |
| Touch radius (incoming / reception) | 2.0 m | 1.0–3.0 m | [ET] ≤ 2 m |
| Touch radius (bad touch / tackle) | 3.0 m | 1.5–4.0 m | [ET] ≤ 3 m |
| Duel radius | 1.0 m | 0.5–2.0 m | [VC] 1 m |
| Ball height: foot touch | ≤ 1.5 m | 1.0–2.0 m | [ET] |
| Ball height: header | ≥ 1.0 m | 0.8–1.5 m | [ET] |
| Ball height: max for any touch | 3.5 m | 2.5–4.0 m | [EL] 3.5 m, [ET] ≤ 3 m (≤ 4 m for fouls/tackles) |
| Kick acceleration peak | 10 m/s² | 4–20 m/s² | [LH] 4, [EL] 20 |
| Direction change | 12.5° | 5–25° | [KM] 5–25°, [VC] 10° |
| Speed change (ratio) | 1.5× | 1.1–1.5× | [KM] |
| Speed change (absolute) | 5 m/s | 1–5 m/s | [VC] |
| Grace period (new owner confirmed) | 0.5 s | 0.2–2.0 s | [KM] 0.5–2 s |
| Reach-ahead time | 1.0 s | 0.5–1.5 s | [LH] |
| Release distance (ball moved away after a kick = pass) | 5 m | 2–8 m | [EL] |
| Arrival distance (ball came from far = reception) | 5 m | 2–8 m | [EL] |
| Minimum failed-pass length (shorter = tackle / duel) | 2.5 m | 0.9–7.5 m | [KM] |
| Shot vs pass near goal: distance from a post | 5 m | 3–8 m | [KM] |
| Out over a line: ball beyond the line by | 0.3 m | 0.1–1.0 m | [GUESS] (ball radius 0.11 m [LAW]) |
| Out over a line: stays out for | 0.3 s | 0.1–1.0 s | [GUESS] |
| Stoppage context window | ±4 s | 2–8 s | [MU] |
| Stoppage smoothing (median filter) | 3.6 s | 1–5 s | [MU] |
| Minimum stoppage length | 2 s | 1–4 s | [MU] |
| Time tolerance when matching | ±0.3 s | fixed | our Step 0 |

T2 – restart patterns (values from [VC] S4.2 grid search; goal dimensions and 9.15 m from [LAW])

| Restart | Rule |
|---|---|
| Kick-off | All players in their own half (tolerance 3–5 m); at least one player within 2 m of the centre spot |
| Penalty | Keeper within 2 m of the goal line between the posts; taker within 2 m of the penalty spot; all others outside the box and ≥ 9.15 m from the spot (tolerance 4 m) |
| Corner | Ball and taker within 3 m of a corner mark |
| Throw-in | Taker beyond the touchline (tolerance 0.5 m); ball held at about 2 m height |
| Goal kick | Ball and taker inside the goal area (tolerance 0.5 m) |
| Free kick | Ball still elsewhere; taker moves ≤ 1 m before the kick; opponents ≥ 9.15 m away |
| Drop ball | Ball dropped between two opposing players; live once it touches the ground [LAW] |
| Default | A dead-ball restart that matches no pattern is a free kick [VC] |

C2 full rules

A. Ball in play / dead
- A1 Out over a line: the ball is fully past the touchline or goal line (pitch size from metadata) for the T1 time, or disappears at a line → DEAD from the crossing. [VC] [GUESS]
- A2 Goal: crosses the goal line between the posts (7.32 m apart) and under 2.44 m → DEAD. [LAW]
- A3 Whistle (foul, offside, injury). Work BACKWARDS from the restart:
  1. Find the restart with the T2 patterns. A free kick, penalty or drop ball with no line crossing before it means a whistle stoppage.
  2. Search back for the stoppage start = the last moment play was competitive. After that moment:
     - the ball is not contested (no opponent pressing the player who has it; PFF pressure is 5 yd attempt / 3 yd successful [PFF]);
     - most players drop to walking speed;
     - the team shape breaks up.
  3. The ball may still move during the stoppage (players fetching it or kicking it back) – this is NOT a sign of play.
  Use ±4 s context and the stoppage smoothing from T1. [MU] [PFF] [GUESS]
- A4 The restart ends DEAD at the first kick of the placed ball (the drop ball at its first ground contact). [LAW] [VC]
- A5 DEAD before each period's first kick-off and after its end. [PFF]
- A6 Ball missing ≠ dead. Without dead cues, keep the previous state for short gaps, then UNKNOWN with low confidence. [VC: handling tracking errors raised F1 from 85% to 95%]
- A7 Not dead: a woodwork rebound (the ball comes back into play), and advantage (no collective slowdown). [PFF]

B. Touches
- B1 A touch = a local peak of ball acceleration (speed OR direction change, T1 thresholds) AND a player within the touch radius at a local minimum of player–ball distance, within ±0.3 s, AND ball height within the T1 limits (higher for a keeper inside their box). [EL] [ET] [LH] [KM]
- B2 A ball passing close by with no change in its motion = not a touch. [LH] [KM]
- B3 Actor = the nearest player at the touch moment, preferring a player moving towards the ball. If team-mates tie, the previous owner wins. If opponents tie, apply the duel rule C7. [KM] [MO]
- B4 A reception usually shows as a DECELERATION of the ball near the player; a pass as an ACCELERATION away from them. [ET]

C. What each touch means
- C1 Control / reception: after the touch the ball stays within the control radius of the same player for the grace period, or the player can reach the ball's position predicted ahead by the reach-ahead time → CONTROL from the touch. Ball speed must stay low enough for control (crosses passing through must not count). [LH] [KM] [MO]
- C2 Dribble: repeated touches by the same player with the ball moving along with them → CONTROL continues. If the ball leaves the control radius and the SAME player gets it back before anyone else, the whole stretch stays CONTROL by that player ("ball far away", e.g. a knock-on). [KM] [PFF ball carry]
- C3 Release: after a kick the ball moves beyond the release distance → TRAVELLING, team = kicker. A clearance (a long, high kick from the own third with no team-mate in that direction) → LOOSE. [EL] [PFF]
- C4 Interception: an opponent touches and then controls the ball → CONTROL for the new team. An opponent-won "pass" shorter than the minimum failed-pass length is a tackle/duel, not a pass. [KM]
- C5 Block / parry: a touch after which nobody controls the ball and it keeps high speed or bounces off → LOOSE. A deflection (small angle, similar speed) → TRAVELLING continues. A block happens within ~5 yd of the passer (more for fast balls); further away it's an interception. [PFF 4.1, 4.6]
- C6 Keeper: a catch (the ball stops and moves with the keeper at hand height, inside the box) → CONTROL. A parry or punch → LOOSE. [PFF] [ET]
- C7 Duel: ≥ 2 opposing players within the duel radius and no clear controller → LOOSE. An aerial duel (ball above header height, opponents converging under it) → LOOSE. A carrier who keeps the ball through contact or while shielding → CONTROL stays. The duel winner is NOT necessarily the next owner. [VC] [PFF 4.4] [MO]
- C8 Loose ball: ball slow and nobody in control → LOOSE until someone controls it. [PFF] [FIFA "In Contest"]
- C9 Hand-over between team-mates without a kick (PFF "take over") → CONTROL switches. [PFF 4.5]
- C10 One-touch pass: a touch with an immediate release → CONTROL can last 1 frame, then TRAVELLING. [PFF] [GUESS]

D. Consistency
- D1 The owner changes only at a detected touch. [LH]
- D2 A new owner is confirmed only after the grace period. [KM]
- D3 Minimum durations stop flicker. [MU] [GUESS]
- D4 Every frame gets a confidence score.
- D5 Every rule logs its name per frame, so errors can be traced.

C3 features (gradient boosting) = the measurements behind every rule above, PLUS per-player speed and acceleration with context at −4, −2, −0.8, −0.4, +0.4, +0.8, +2, +4 s for stoppage detection. [MU]

## Locked split (v2, redrawn 2026-11-06)
Why redrawn: labels v1 showed that in 18 matches the tracking file puts the tagged ball carrier > 10 m from the ball in more than
10 % of CONTROL frames (tracker identity switches and estimated player tracks, Part 1 of the 2026-11-06 follow-up; no clean fix),
and the ten coordinate-disagreement matches are among them. v1 (seed 20261106) had 5 of them in its test set.
- **Stress set (never used for learning; reported separately at the end):** 3816, 3817, 3826, 3829, 3835, 3837, 3841, 3845, 3846,
  3847, 3852, 3853, 3859, 10502, 10504, 10512, 10514, 10516
- Clean = the other 46. Drawn with `numpy.random.default_rng(20261107)`: 16 test matches from the 38 clean matches outside
  DEV_MATCHES; the other 30 permuted with the same generator and dealt into 5 folds of 6. Stored in `src/m1_possession/splits.py`.
- **Test (locked, run once at the end):** 3822, 3825, 3827, 3828, 3834, 3836, 3842, 3844, 3848, 10503, 10505, 10508, 10509, 10510, 10511, 10513
- Fold 0: 3813, 3815, 3818, 3819, 3849, 3858
- Fold 1: 3820, 3821, 3823, 3831, 10506, 10517
- Fold 2: 3824, 3832, 3833, 3843, 3851, 3855
- Fold 3: 3814, 3839, 3854, 3856, 3857, 10515
- Fold 4: 3812, 3830, 3838, 3840, 3850, 10507
- All 8 dev matches are clean and in the learning set (folds 0: 4, 1: 2, 3: 1, 4: 1).
- Note: the plan's Approach section says 48 learning matches / 6 folds; v2 replaces that with 30 / 5 for the reason above.

## Findings log
(The agent adds a short dated entry here after each step: what was built, key numbers, decisions, open issues.)

**2026-11-05 – Step 0, ball smoother.** Built `src/m1_possession/` (standalone, numpy only: `config.py`, `smoother.py`), tests in
`tests/test_m1_smoother.py` (12 pass), dev evaluation `scripts/m1/evaluate_smoother.py`. Pipeline: drop held repeats
(every 2nd frame of an identical run; 497k of 996k raw samples), drop one-sample spikes > 40 m/s (74), split at remaining
impossible jumps (1,380) and gaps > 0.5 s, then smooth. Recommended: `rts_adaptive` (Kalman + RTS, q = 30, r = 0.25 m, velocity
allowed to jump at kicks found in the track, dv >= 3 m/s). Dev results (8 matches): coverage 66.7 % (raw 66.8, PFF 54.5; 99.9 % of
PFF frames covered); kick preservation 49.6 % vs 28.1 % for plain RTS and 49.9 % for unsmoothed interpolation; jitter 6.5 m/s²
(linear 9.0, Savitzky-Golay 7.8–8.1, PFF 4.5). Distance to PFF smoothed outside OTB spells is about 3.3 m median for every
method including raw: PFF's ball is straight constant-speed lines between tagged events and runs about 6 frames (0.2 s) behind
the raw ball, so it is a weak reference for travelling balls. Never more than one entry in `balls`. Open: the raw ball's speed
ramps up over ~0.4 s before PFF kick eventTime, so kick timing must be learned with a tolerance, not taken from the tag.
**2026-11-05 – Step 0 follow-up (offset, missed kicks, leakage).** Added output flag `estimated` (kept raw update with
ESTIMATED visibility; 14 tests pass); script `scripts/m1/step0_followup.py`. Smoother settings unchanged. Kick timing: the
largest velocity change of our ball within ±1 s is a median 0.067 s (2 frames) before PFF eventTime, the same in all 8 matches
(−0.03 to −0.10 s), but the spread is wide (p5 −0.73, p95 +0.90 s). Kick hit rate 49.6 % at ±3 frames, 74.3 % at ±10, 77.9 % at ±15.
Misses: 54 % have a speed change ≥ 3 m/s but the peak lies outside ±3 frames, 13 % are direction-only, 10 % have no ball, and
raw samples around misses are more often ESTIMATED (39 % vs 24 %). Proof: ballsSmoothed equals the tagged carrier's smoothed
position inside OTB spells (median 0.004 m, 99.6 % within 0.1 m), follows straight lines between carriers between spells (74.6 %
within 0.5 m) and is missing in 99.9 % of dead-ball frames. Leakage test (nearest player ≤ 1.5 m): player correct in OTB spells
75.6 % with ballsSmoothed vs 65.1 % with our ball; "ball missing = dead" F1 0.948 vs 0.775. Decision: ballsSmoothed is tag-derived
and is never model input; it may serve only as a diagnostic upper bound.

**2026-11-05 – Step 1 (A), answer key.** Built `src/m1_labels/` (pure labeller, tag-based; never imported by `m1_possession`,
enforced by `tests/test_m1_labels.py`, 5 tests) and `scripts/m1/build_labels.py`; labels per match in
`data/processed/m1_possession/labels/<match>.npz`. Spell frames come from the tracking overlay start/end_frame for 10,090 of 10,091
spells; aligned startTime/endTime equal it or are one frame earlier in 100 %. Dev totals: DEAD 43.5 %, CONTROL 25.9 %, TRAVELLING
26.4 %, LOOSE 4.2 %, UNKNOWN 0.02 %; score_mask 98.7 %. Ball in play 48.3–64.9 % (3812, 3813, 3814 below 55 %: long whistle and
goal stoppages). CONTROL agrees with the OTB overlay in 96.9–99.7 % of frames both ways (differences: drop-ball setup, rebound /
duel / null-player spells as LOOSE, a few gaps). With our smoothed raw ball: CONTROL player-ball distance median 1.0–1.3 m
(p95 3.5–16.9 m; 325 runs > 10 m for > 0.3 s), speed medians CONTROL 3.3, TRAVELLING 9.2, LOOSE 6.0, DEAD 2.1 m/s; a kick
within ±0.3 s of 73.0 % of PA/CR/SH/CL releases. Set-piece spells start at the release (restart_setup 0 s except drop balls).
Open decisions: TC as LOOSE, BC retained / no-possession-event gaps (UNKNOWN), drop-ball setup, the 325 far-control runs.

**2026-11-06 – Step 1 finished (labels v1, split, C0/C1).** Approach section added; steps list updated; split locked (seed
20261106, `src/m1_possession/splits.py`). Labeller v1: TC P/O/C → LOOSE, TC/BC retained → CONTROL continues, no-possession-event
gaps UNKNOWN (unscored), drop balls live, CONTROL > 10 m from our ball for > 0.3 s keeps its label but is unscored. Labels v1 for
all 64 matches in `labels/v1/` (`scripts/m1/build_labels_v1.py`): DEAD 41.4 %, CONTROL 27.3 %, TRAVELLING 26.8 %, LOOSE 4.5 %,
UNKNOWN 0.04 %; ball in play 58.6 %; 95.3 % of frames scored; CONTROL vs OTB overlay 96.9–99.8 % both ways. Far-control masking
removes 243k frames (4,816 runs); it is worst in the ten coordinate-flagged matches (up to 39 % of CONTROL in 10512), five of which
are in the locked test set. Baselines on the 8 dev matches (`scripts/m1/run_baselines.py`, scoring in `m1_labels/scoring.py`):
C0 frame accuracy 68.2 %, player 69.4 %, team 70.3 %, possession-change F1 0.14, dead start/end F1 (±2 s) 0.18 / 0.29; C1 60.1 %,
player 68.8 %, team 82.2 %, possession-change F1 0.14, dead start/end F1 (±2 s) 0.62 / 0.22. LOOSE is almost never found
(C0 24 %, C1 5 %); whistle stoppages are the main DEAD gap (no whistle detection in either).

**2026-11-07 – Far-control cause, split v2, C1 fair reproduction, C2 built and tuned.** Part 1
(`scripts/m1/far_control_cause.py`, 5 named + 200 random far-control stretches over the 18 flagged matches): player tracking
error 71 (the tagged player is ESTIMATED most of the stretch), identity mix-up 67 (64 = another jersey of the same team at the
ball), ball tracking error 41, time offset 10, mirrored coordinates 0, unclear 16. Roster jerseys equal event jerseys in all 64
matches. Only 10 swap signatures (8 matches, ~3.3k of 243k masked frames) are stable across the match, so there is no clean
deterministic fix; nothing was applied. Part 2: split redrawn (v2, seed 20261107, see "Locked split"): 18 stress matches excluded
from learning, 16 test, 30 learning in 5 folds. Parts 3-4 (`scripts/m1/compare.py`, `report_compare.py`; cross-validated by
match on the 30 learning matches, nothing run on test or stress): frame accuracy C0 0.677, C1-published 0.594, C1-tuned 0.619
(zone 1.5 m, angle 5°, speed 5 m/s, spatial 0.2 m), C2-start 0.639, C2-tuned 0.682; C1 + oracle ball-in-play (not a contestant)
0.817. C2-tuned: player 64.5 %, team 77.2 %, possession-change F1 0.206 (best), dead start/end F1 (±2 s) 0.51 / 0.33. Pass
detection F1 at ±10 / 2 / 0.5 s: C1-published 0.62 / 0.57 / 0.46, C1-tuned 0.67 / 0.63 / 0.58, C2-tuned 0.64 / 0.60 / 0.58.
C2 (`src/m1_possession/rules.py`, 12 synthetic tests) is tuned group-wise (dead → touch → control) with an own evolutionary
search (Optuna not installed); threshold table in the step report. C2's main errors: CONTROL predicted as TRAVELLING (the release
fires during dribbles), and false whistle stoppages from the collective-slowdown rule (CONTROL / TRAVELLING → DEAD). DEAD is the
strongest class (86.8 % recall); LOOSE remains weak (21.5 %).

**2026-11-08 – C2-v2, input upgrades, C3 (cross-validated on the 30 learning matches only; nothing run on test or stress;
inputs tag-free: our smoothed raw ball + raw players).** lightgbm and optuna installed in `.venv` (not yet listed in
`pyproject.toml`). C2-v2 (`rules.py`, `C2Params.v2`; whistle stoppage only if a restart pattern follows within `restart_limit_s`,
two layers, 1 m "clearly closer" margin [coderhema], back-dating [coderhema], ball-missing gap rules [KoichiInoue7]; only the 5 new
parameters tuned per fold with Optuna, `scripts/m1/c2v2.py`): frame accuracy 0.659 vs C2-tuned 0.682, but player 73.4 % vs 64.5 %,
team 85.1 % vs 77.2 %, spell-F1 0.370 vs 0.333, H/A/D macro-F1 0.759 vs 0.758; possession-change F1 0.168 vs 0.206. Tuning
pushes `restart_limit_s` to its 120 s ceiling and the margin down to ~0.3 m. Ball-from-players (`ball_infer.py`,
`scripts/m1/bfp.py`, LightGBM on hidden observed frames, held-out folds): median / p95 error 2.6 / 12.7 m vs last-seen 6.3 /
48.5 m; > 15 s gaps 6.9 / 16.7 m (only 134 air samples). C3 (`c3.py`, `scripts/m1/c3.py`: LightGBM alive/dead + stoppage
reason, state with LOOSE weight, 5-candidate player ranker, 5-state Viterbi with touch / restart / no-direct-team-switch
constraints and back-dating; LightGBM and decoder settings by Optuna nested inside the training folds). With/without at fixed
settings: all upgrades within ±0.1 point (none 0.839, gap features 0.840, smoothing + gap filling + gap features 0.840);
ball-from-players dropped (no gain). Final C3 (smoothing + gap filling + gap features, nested tuning): frame accuracy 0.851,
player 81.5 %, team 90.9 %, H/A/D macro-F1 0.865, spell-F1 0.477, dead start / end F1 (±2 s) 0.79 / 0.60, pass F1 (±2 s) 0.65
(precision 0.87, recall 0.52); possession-change F1 0.146 is below C2-tuned. Main C3 errors: TRAVELLING after passes predicted as
CONTROL (20 % of error frames) or DEAD (18 %, mostly long ball-missing stretches), CONTROL predicted as TRAVELLING (16 %); LOOSE
recall 37.9 %. Known caveat: inner-CV C2-v2 features for fold k use fold-k parameters (tuned on the same 24 matches as the inner
folds), a mild optimistic bias in inner tuning only; outer scores are clean.

**2026-11-08 – Housekeeping and C3 video check list (no training or tuning).** M1 code, tests, scripts and this plan
committed on branch `m1-possession` (local only, not pushed); `pyproject.toml` gained the extra
`m1 = ["lightgbm==4.7.0", "optuna==5.0.0"]`. Several M1 scripts still import `football_fatigue.event_detection.config`
(uncommitted, Milestone 3A) and read its position cache `data/processed/event_detection/cache`. **Scratch space for long runs:**
`m1_possession.config.use_scratch()` (one place; default `D:\m1_tmp`, override with env `M1_TMP`) sets TEMP / TMP / TMPDIR /
JOBLIB_TEMP_FOLDER and Python's tempdir for the process and its workers; `optuna_storage(name)` gives an SQLite study file
there. `bfp.py`, `c2v2.py`, `c3.py`, `compare.py`, `report_c3.py` call it at start. Deleted re-creatable caches
`c3/feat/` (2.8 GB) and `c3/c2cache/` (2.5 GB), both rebuilt by `scripts/m1/c3.py features --config <cfg>`; Z: free 12 → 18 GB.
Video check list `reports/m1_possession/c3_error_videos.md` (`scripts/m1/c3_error_videos.py`, the 2 longest single-type
stretches of C3's 5 biggest error types, video time + film-room links). Pattern: most long stretches have the ball missing
(80–100 %) with no PFF event inside; the two longest "TRAVELLING → DEAD" stretches (25–29 s) and the 67 s "CONTROL → DEAD"
stretch (3843) look like answer-key holds across long ball-missing spells rather than model errors and need a video check.
