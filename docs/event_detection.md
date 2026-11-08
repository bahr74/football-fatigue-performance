# Event detection from positions (Milestone 3A)

Status: **rules detector `rules-0.2` and ML detector `ml-0.3` implemented**, pending independent review. The CV-input
adapter and the robustness test are not started. Package: `football_fatigue.event_detection`.

## Goal

Detect passes (with outcome), crosses, shots, goals, and key passes **from positions and time only**, write them in a
PFF-like event file, and measure accuracy against the PFF event labels. The same detector is meant to run later on positions
from the CV module, so it reads nothing PFF-specific.

## Data: what is used and what is not

| Used | Not used (and why) |
| --- | --- |
| Raw player positions (`homePlayers`/`awayPlayers`, `CoordinatePolicy.RAW`) through `MatchData` | `ballsSmoothed`: PFF smooths the ball *using the event labels* (the ball is placed at the carrier between events), so it would leak the answer |
| Raw ball (`balls`), missing kept as NaN | The `game_event` / `possession_event` overlays inside tracking frames |
| Metadata: fps, team ids, `homeTeamStartLeft(ExtraTime)` for attack direction | Event Data files, except for **scoring** (root version) |

Data handling (all in derived code; `data/raw/` is untouched):

- **Ball cleaning** (`kinematics.py`): raw positions are held for two frames (about 15 Hz); within a run of identical
  samples every second one is a repeat and is not used for fitting (a resting ball keeps half of its samples). A sample that
  implies more than 40 m/s both into and out of it is a spike and is dropped. Counts are written to the run manifest. Nothing
  is interpolated.
- **Exclusions:** periods 3 and 4 of `10506`, `10510`, `10511`, `10517` (suspected side/identity anomaly or no tracking;
  audit §7). Labels outside the tracked time range are counted as `not_scoreable`, never dropped silently.
- **Flagged:** the ten matches whose event coordinates disagree with the supplied tracking (event comparison §3.4) are
  evaluated but marked in the per-match table.
- **Positions cache:** one tracking pass takes about 1.5 minutes per match, so positions are cached as
  `data/processed/event_detection/cache/<match>[_excl3-4].positions.npz` (columnar NumPy arrays; derived data).

## Method (rules)

1. **Touches** (`touches.py`): the ball velocity is fitted on about 0.2 s before and after each frame. A touch is a local
   maximum of the change `|v_after - v_before|` above 3 m/s where some player comes within 3 m of the ball. The nearest
   player is the actor.
2. **Releases** (`classify.py`): a touch after which the ball next reaches a *different* player (or the same player only after
   leaving a 4 m radius), with a release speed of at least 4 m/s. A ball that disappears for more than 1 s ends the play.
3. **Shot:** within 25 m of the attacked goal, and either the ball direction (displacement over 0.4 s, or the fitted
   velocity) crosses the goal line within the posts ± 4.5 m at 6 m/s or more, or the ball is seen at the goal mouth.
4. **Goal** (`rules-0.2`): two independent signs are required. (a) Within 5 s of the shot the ball is seen at the goal mouth
   (beyond the line between the posts and under the bar, or within 1.5 m of the goal frame in the pitch plane), **and**
   (b) play stops for 60 to 300 s (no touches; the pause starts within 15 s of the shot), the length of a celebration and
   kickoff. A pause alone (injury, VAR, time-wasting, a lost ball) or a ball near the goal alone (a save, a near miss) is
   not a goal; pauses over 300 s are other stoppages. Only the last shot before a pause can be the goal. The pause is needed
   because the ball is usually not tracked into the net. `rules-0.1` accepted the pause alone and had goal precision 0.27 on
   the test split.
5. **Cross:** a release from a wide position (|y| ≥ 12 m) within 35 m of the attacked goal line, moving inwards at 8 m/s or
   more, ending in the box area.
6. **Pass:** any other release. Outcome `C` when a teammate touches next, `D` when an opponent does, `O` when the ball is last
   seen outside the pitch; when the ball is lost (missing) without a receiver no pass is claimed.
7. **Key pass** (derived, same rule for labels and detections): the last completed pass or cross of the shooting team before a
   shot, with no opponent action in between, within 10 s.

All thresholds live in `config.DetectorConfig` and are copied into every run manifest. Every detected event carries a
`reason` with the feature values that decided it.

## Method (ML refinement, `ml-0.2`)

The rules keep finding candidate touches, with looser thresholds (`ML_CANDIDATE_OVERRIDES`: velocity change 2 m/s, reach
4 m, peak window ±5 frames), so 93 % of labelled passes and 99 % of labelled shots have a candidate (dev sample). Then:

- **Rows per candidate actor** (`features.actor_rows`): each touch is scored for the nearest player and, when one is in
  reach, the nearest player of the other side. The nearest player is often the blocker or the defender; scoring both lets
  the model attribute the event to the right team, and the features (attack direction, goal distance) follow the actor.
- **Features** (`features.py`, positions only): ball position in the actor's attacking frame, distance and opening angle to
  goal, velocity change, speeds before and after, ball height, flight direction and its projected goal-line crossing, where
  the ball ends up, time and side of the next and previous touches, carry flag, ball gap, out of play, closest approach to
  the goal mouth and the pause that follows, defenders within 3 m and 6 m, nearest opponent and teammate, ball visibility.
- **Type model:** scikit-learn `HistGradientBoostingClassifier` (none, PA, CR, SH, CL), sample weights
  `(n_none / n_class) ** 0.5` for the rare classes. A row is positive only when its actor's team is the labelled team. Per
  touch the actor with the most confident event wins; clearances are learned but not exported.
- **Completion model:** a second classifier predicts whether a pass or cross was completed (PFF `C`). The receiver is the
  next touch's player when that player is a teammate.
- **Goals and key passes** use the same rules as the rules detector (key-pass chains are broken by the opponent's detected
  events, not by every loose candidate touch).
- **Honest scoring:** the 64 matches are dealt into 8 folds by sorted id; each match is scored by the models trained on the
  other 7 folds (recorded per match as `model=fold_k`). A model trained on all matches (`full`) is used for matches outside
  the dataset, such as CV input. Hyperparameters are fixed in code and were not tuned on the scored matches.
- Models and the training table are written to `models/event_detection/ml-0.2/` (ignored by Git) with a manifest listing the
  features, classes, folds, class counts and weights. Training needs the optional `ml` extra (`scikit-learn`).

## Evaluation

- **Reference:** PFF Event Data, root version, types `PA`, `CR`, `SH` (`shotOutcomeType = G` for goals); `CL` (clearance) is
  shown in the confusion table only. PFF labels are human annotations, not physical ground truth.
- **Matching:** per type, one-to-one, greedy by smallest time difference, same period, |Δt| ≤ tolerance; reported at 0.5,
  1 and 2 s (primary 1 s). Also: team, player and completed-or-not agreement on matched passes; a confusion table; timing errors.
- **Split:** thresholds were tuned on the dev matches `3812, 3813, 3814, 3815, 3818, 3819, 3820, 3821` only. The headline
  numbers are from the other 56 matches (test split), which were not looked at during tuning.

## Running it

```powershell
.\.venv\Scripts\python.exe -m football_fatigue.event_detection detect 3812                    # events.json + manifest.json
.\.venv\Scripts\python.exe -m football_fatigue.event_detection evaluate --split test --workers 4
.\.venv\Scripts\python.exe -m football_fatigue.event_detection viewer 3812                    # local HTML replay viewer
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -e ".[dev,ml]"                       # ML extra (scikit-learn)
.\.venv\Scripts\python.exe -m football_fatigue.event_detection train --workers 4              # table + 8 fold models + full model + threshold
.\.venv\Scripts\python.exe -m football_fatigue.event_detection evaluate --detector ml --split test --workers 4
.\.venv\Scripts\python.exe -m football_fatigue.event_detection detect 10517 --detector ml     # PFF-like events of the final
.\.venv\Scripts\python.exe -m football_fatigue.event_detection report --split test            # mentor report (HTML)
```

Detection reads the cached positions, so a match takes seconds once its cache exists (the first run of a match reads the
tracking file once, about 1.5 minutes). Saved runs: `data/processed/event_detection/runs/<match>/<version>/`.

- `detect` writes `data/processed/event_detection/runs/<match>/<version>/events.json` (PFF-like records: `gameId`,
  `eventTime`, `frameNum`, `gameEvents{period, teamId, playerId}`, `possessionEvents{possessionEventType, passOutcomeType,
  receiverPlayerId, shotOutcomeType, ...}`, `ball`, and `detector{reason, confidence, keyPass, playerKey}`) and
  `manifest.json` (source paths, sizes and SHA-256, parameters, ball-cleaning counts, software versions). Existing runs are
  not overwritten without `--overwrite`; output inside the raw root is refused.
- `evaluate` writes `reports/event_detection/<version>/<split>/summary.md`, `summary.json`, `per_match.csv`.
- `viewer` writes `reports/event_detection/<version>/viewer_<match>.html`: detections next to labels (hit, other type,
  missed, false alarm) with a 6 s position replay and the `reason` of each decision. It embeds tracking positions: keep it
  local (`reports/` is ignored by Git) until the licence is verified.

## Results (test split, 56 held-out matches, ±1 s)

Copied from `reports/event_detection/{rules-0.2,ml-0.3}/test/summary.md` (runs of 2026-11-02); the mentor report
`reports/event_detection/mentor_report.html` is generated from the same files (`report --split test`). 1,386 labels outside
the tracked time range were not scored. Rules thresholds were tuned on the dev matches; ML scores are out-of-fold (no match
scored by a model that saw it).

| Event | Labels | Rules P / R / F1 | ML P / R / F1 |
| --- | ---: | --- | --- |
| Pass incl. cross | 59,411 | 0.727 / 0.750 / **0.738** | 0.927 / 0.722 / **0.811** |
| Cross | 2,033 | 0.449 / 0.344 / **0.390** | 0.699 / 0.645 / **0.671** |
| Shot | 1,302 | 0.519 / 0.353 / **0.420** | 0.645 / 0.591 / **0.617** |
| Goal | 161 | 0.659 / 0.360 / **0.466** | 0.741 / 0.534 / **0.621** |
| Key pass | 952 | 0.435 / 0.173 / **0.248** | 0.531 / 0.350 / **0.422** |

On matched events, ML: pass team correct 93.7 % (40,195 / 42,887), player 84.2 %, completed-vs-not 89.1 % (38,231 / 42,887);
shot team 93.4 % (718 / 769), player 74.1 %. Rules: pass team 85.6 %, player 76.5 %, completed-vs-not 78.1 %. Median timing
error 0.10 s for both. The stricter goal rule (`rules-0.2`) raised rules goal precision from 0.272 to 0.659 (recall 0.472 to
0.360, F1 0.345 to 0.466).

ML versions tried on the same split (kept for the record under `reports/event_detection/`): `ml-0.1` (one actor per touch,
argmax) pass 0.823, cross 0.667, shot 0.611, goal 0.481, key pass 0.270, but pass team correct 87 % and shot team 63 %;
`ml-0.2` (both candidate actors, completion model) improved attribution, goals and key passes but lost recall because
events whose team had no player in reach were labelled as no event; `ml-0.3` keeps them and calibrates the event threshold
on the dev matches (0.60).

## Limitations

- Raw ball positions are missing in about a third of frames, update at about 15 Hz, and are noisy; most missed passes have
  no clear ball velocity change or a player more than 3 m from the ball at the labelled time.
- Shots and crosses are hard to separate from passes with geometry alone (similar speeds; many shots are blocked or
  deflected within a metre). Goals depend on the shot being detected first.
- The detector does not label clearances, challenges, carries, or set pieces; clearances are mostly reported as passes.
- Player identity comes from the tracking jersey join (68 % of raw entries have `LOW` jersey confidence), so the actor is
  wrong in a share of correctly timed detections (see the attribute table).
- The ML detector learned PFF tracking and PFF labelling conventions. CV positions will be noisier and differently biased;
  the CV adapter and a robustness test with degraded positions are the next step before any claim about video input.
- About 7 % of labelled passes have no candidate touch at all, which caps pass recall; most remaining pass misses are touches
  the model is unsure about (precision was preferred by the dev-calibrated threshold).
