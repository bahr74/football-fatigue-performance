# M11 Trajectory Recovery Lab: complete explanation of the dashboard

This document explains every screen, button, control, word and technique in the **M11 Trajectory Recovery Lab**. A short
presentation script is in [m11_demo_guide.md](m11_demo_guide.md).

## 0. Starting and stopping

| How | What to do |
|---|---|
| Double-click | `scripts\m11_lab.cmd` (opens the browser; close the black window to stop) |
| PowerShell (repository folder) | `.\.venv\Scripts\python.exe -m football_fatigue.m11_dashboard --port 8610 lab` |
| Address | **http://127.0.0.1:8610** (this computer only) |

Port 8610 is used because Windows reserves ports 8494–8593 on this computer, so the usual Streamlit port 8501 does not start
here. The first time a match is opened, its tracking file is loaded, which takes about 10 s; later loads are cached.

## 1. What the dashboard demonstrates, in one paragraph

Two players come very close to each other in a real FIFA World Cup 2022 match. The dashboard shows both players **before** a
short interval with their identities, hides **both** of them for that interval, and then shows two **anonymous** tracks after it.
The question is which track belongs to which player. Two trained models, **GRU C** and **HGB C**, answer from the trajectories
alone. A fixed **routing policy** decides which answer to trust, or whether to abstain. Finally the dashboard reveals the true
pairing, which comes from the provider's continuous tracking of each player.

The hidden interval is **simulated**. The provider did track both players; the dashboard deletes those positions from the model
input to imitate what a tracker sees when players are lost in a crossing. **It is not a video occlusion**, and no video is used.

## 2. Why a test in this dashboard is a valid, unseen test

- **Data:** 40 World Cup matches. **8 are frozen held-out matches** that are never opened (the code refuses them), and 32 are
  development matches. The dashboard offers only the 32.
- **Unseen-match models:** the 32 matches are split into 8 *folds* of 4. For each fold, one GRU C and one HGB C were trained on
  the other 28 matches: 25 to fit the weights and 3 to decide when to stop training. When you pick a match, the dashboard
  automatically uses the fold whose models **never saw that match**. It is shown as "models: outer fold k (never trained on match
  m)", and the check list in **✅ Is this a valid test?** confirms it.
- **Same files as Milestone 1:** each checkpoint's SHA-256 fingerprint is compared with the one recorded and verified in
  Milestone 1.
- **Labels are not invented:**
  - The pairing comes from each player's own continuous provider track.
  - Every encounter passed the Stage 2 label-quality procedure. It is re-run whenever the gap or its placement is changed; if it
    fails, the scenario is marked **invalid** and never shown to a model.
  - The label is re-checked against the raw provider positions.
- **Real inference:** pressing **Run prediction** executes both models now, on the model-visible tracks only. When the same
  scenario exists in the saved Milestone 2 run, the saved probabilities appear in **Advanced information** only as a cross-check
  (they agree to within 1e-6).
- **What it is not:** it is not the final held-out evaluation (that has not been run), and the occlusion is synthetic.

## 3. Page layout

- **Left sidebar:** the **Mode** switch, and in Research lab mode all scenario controls.
- **Top:** the title and a one-line statement of what is shown.
- **Six tabs:** 🎬 Scenario, 🖼 Example gallery, 📊 Evaluation (Milestone 2), 🧠 How the models work, 📄 Load JSON tracklets,
  ❓ Guide & legend.

### 3.1 Sidebar: Mode

| Option | Meaning |
|---|---|
| **Demo** | A guided 7-step walkthrough with simple controls. Best for presenting. |
| **Research lab** | All controls: choose match, encounter, stress test, custom combinations and seeds. |

### 3.2 Sidebar in Research lab mode

**1 · Match and encounter**

| Control | What it does |
|---|---|
| **Development match** | One of the 32 development matches. The label shows which fold's models will be used ("fold k models"). |
| **Encounter (Milestone 2 selection, 60 per match)** | Close two-player encounters of that match. These are the 60 per match chosen by a fixed random rule in Milestone 2, independent of any model result. The label reads `#n · match clock · crossing type · movement type`. |
| **Seed** + **🎲 Seeded random** | Picks a random match and encounter from the seed number. The same seed always gives the same scenario (reproducible). |
| **🎲 Truly random encounter** | Draws a new seed from the operating system (nobody chose it). The seed is displayed ("Selected by: … seed N") so it can be repeated. |

**2 · Stress test → Variant**

**One factor (Milestone 2):** exactly one property differs from the baseline (0.8 s gap, centred, clean, all observations, 2 s
context), as in the Milestone 2 suite. The saved result of the same scenario is then available as a cross-check.

| Factor | Levels and meaning |
|---|---|
| **baseline** | 0.8 s gap, clean |
| **gap duration** | 0.2, 0.4, 0.6, 1.0, 1.2, 1.6, 2.0, 2.4 s of hidden time |
| **gap placement (shift)** | the hidden interval is moved −0.5 … +0.5 s away from the moment of closest approach |
| **positional noise** | random Gaussian error added to every observed x and y: 1, 2, 5, 10, 25, 50 cm |
| **observation dropout** | `random10/30/50` removes 10/30/50 % of observations at random; `burst0p5` removes one 0.5 s block per track; `rate7p5` keeps every other frame (7.5 Hz) |
| **context length** | only the last / first 1.5, 1.0 or 0.5 s before / after the gap are kept |

**Level** is the value of the chosen factor.

**Custom combination:** any combination of the controls below. Such results are not Milestone 2 comparisons and have no saved
cross-check.

| Control | Meaning |
|---|---|
| **Gap duration (s)** | Length of the hidden interval. |
| **Gap placement: shift from closest approach (s)** | Where the hidden interval sits relative to the closest approach. |
| **Positional noise σ** | Standard deviation of the added position error. |
| **Observation dropout** | One of the dropout modes above. |
| **Context per side (s)** | How much observed trajectory is kept on each side of the gap. |
| **Noise / dropout / order seed** | Fixes the random noise, the random dropout and the random order of Track X / Y. |

Changing any control rebuilds the scenario immediately and clears the previous prediction and reveal.

## 4. 🎬 Scenario tab

### 4.1 Demo mode: the 7 steps

A progress bar shows "Step k of 7". **◀ Back** and **Next ▶** move between steps; the blue box explains the current step.

| Step | What you do and see |
|---|---|
| 1 Select a scenario | **🎲 Pick a random unseen-match scenario**: a truly random development match and encounter at the 0.8 s baseline; the demo then moves to step 2. |
| 2 Watch the approach | Press **▶ Play** under the pitch and watch Player 1 and Player 2 approach each other. |
| 3 The hidden interval | The board opens in the middle of the hidden interval: both players are gone and a brown "SIMULATED OCCLUSION" badge is shown. |
| 4 Run GRU and HGB | **▶ Run prediction** executes both models now. |
| 5 Compare predictions | Read the three cards (GRU C, HGB C, Operational decision). |
| 6 Reveal the truth | **👁 Reveal ground truth**. |
| 7 Try another | **🎲 Another random scenario**, or choose **Gap for this encounter (s)** and press **↻ Same encounter with this gap**: the same two players with a different hidden duration. 0.2 and 2.0 / 2.4 s are outside the validated range, so the policy abstains. |

### 4.2 Header line

"**Match m · encounter `id` · variant · models: outer fold k (never trained on match m)**". The encounter id is a fixed hash of
the encounter; the variant states what was changed from the baseline.

### 4.3 Buttons above the pitch

| Button | What it does |
|---|---|
| **▶ Run prediction** | Executes GRU C and HGB C on the visible tracks now. The caption shows the time of the run and a counter ("live model call #n"), so a new run is visible each time. |
| **👁 Reveal ground truth** | Shows the true pairing. It is disabled after use; a new scenario hides the truth again. |

### 4.4 View and Playback

**View**

| Option | Shows |
|---|---|
| Encounter close-up | the four tracks plus a small margin |
| Surrounding play | about 15 m or more around them |
| Full pitch | the whole 105 × 68 m pitch |

**Playback**

- **Animation:**
  - **▶ Play** and **⏸ Pause** under the pitch;
  - the time slider (labelled in seconds) can be dragged to any moment;
  - it runs in the browser, so it is smooth.
- **Step by step:**
  - **⏮ Start**, **◀ −1 frame**, **+1 frame ▶**, **+0.5 s ⏩**;
  - **Before gap** (the last frame before the hidden interval), **In gap** (its middle), **After gap** (about 1 s after it);
  - the **Time (frame)** slider, where one frame is 1/15 s.

### 4.5 The pitch board: every symbol

| Element | Meaning |
|---|---|
| Green pitch with white lines | Real pitch markings, 105 × 68 m. Coordinates are the provider's, in metres, origin at the centre spot. |
| **Badge, top left** | `t = … s`, time relative to the start of the hidden interval (t = 0 is when both players disappear), and the phase. Blue **BEFORE**: incoming tracks observed. Brown **SIMULATED OCCLUSION**: both players hidden from the model. Orange **AFTER**: anonymous outgoing tracks observed. |
| **Circles, Player 1 (light blue) / Player 2 (purple)** | Incoming tracklets: the players before the gap, identities known. The small dots are the individual observations the model receives; with noise or dropout you see exactly what the model sees. Hollow circle = last seen position. |
| **Diamonds, Track X (orange) / Track Y (yellow)** | Outgoing tracklets: two anonymous tracks after the gap. Their colours are deliberately unrelated to the players' colours, and their order is random, so neither colour nor order gives the answer away. |
| **Dashed yellow circle with brown fill** | The encounter area where the players are hidden; the "HIDDEN for x s" label appears during the gap. |
| **Grey dots** | The other players of the match at the same moment (real positions, not part of the question). |
| **White line(s), after Run prediction** | The operational decision drawn from each player's last position to the track it was assigned to. Solid = accepted decision; dashed = the model's lean when the system abstained. Drawn only after the hidden interval. |
| **After reveal** | Each track takes the colour of its true player and is renamed "Track X = Player k"; dotted lines show where each player really went while hidden. |
| **Legend under the pitch** | Lists every element above. |

### 4.6 Result cards (after Run prediction)

**GRU C card and HGB C card**

- **Bar:** green part = P(MATCHED), red part = P(SWAPPED); the two add up to 1.
- **Model's pairing:** MATCHED or SWAPPED, and the resulting Player → Track links.
- **Confidence:** max(P, 1 − P), between 0.5 (coin flip) and 1, compared with that model's **5 % threshold** (GRU 0.6425,
  HGB 0.7875). "clears the threshold" or "below threshold (would abstain)".
- **Green box "This is the model the routing policy used."** This model's answer is the operational one.
- **Blue box "Not the routed model for this input: shown for comparison only."**
- **Red box "Input outside the supported range: this probability is diagnostic only, not validated."** Shown when the policy does
  not support the input; the number is shown for curiosity only.

**Operational decision card (policy `m11-routing-v1`, unchanged)**

| Result | Meaning |
|---|---|
| **ACCEPTED: MATCHED / SWAPPED** by GRU C or HGB C (primary / fallback route) | The system commits to an answer. |
| **ABSTAIN: low confidence** | The routed model was not confident enough. |
| **ABSTAIN: unsupported input** | Gap below 0.4 s or above 1.6 s, declared noise above 5 cm, or too few points. |

- **Why:** the policy's own reasons, for example "primary: gap_s 0.800 s within the GRU C band (0.75, 0.85) s".
- **Caption:** the gap, the declared noise, the operating point (5 % error), the time and duration of inference (milliseconds,
  CPU), and the **Milestone 2** coverage and realised error measured for the same condition across 1,920 encounters.

### 4.7 Ground-truth box (after reveal)

- **Ground truth:** MATCHED or SWAPPED and the true links. It comes from the provider's continuous tracks, and the label was
  re-verified against raw positions.
- Jersey numbers and team side (home / away) of the two players.
- **GRU C correct ✅ / wrong ❌**, **HGB C correct / wrong**, and the **Operational decision** outcome:

| Outcome | Meaning |
|---|---|
| ✅ accepted and correct | the system committed and was right |
| ❌ accepted but WRONG | the system committed and was wrong |
| ⏸ abstained (low confidence) | the system declined; a caption says whether the abstention *avoided an error* or *withheld a correct answer* |
| 🚫 abstained (unsupported input) | the input was outside the validated range |

### 4.8 ✅ Is this a valid test?

Available before reveal, because it contains no answer. Every row shows ✅ or ❌:

- the match is a development match;
- it is not held-out;
- the fold's models never saw it in training;
- it was not used for early stopping;
- both checkpoints match the verified fingerprints;
- the label was accepted by Stage 2 and re-verified against raw data.

It also states the real **closest distance** of the two players in the raw data (always under 0.5 m), the match clock, how long
the simulated occlusion is, and the **raw source file** (`…/Tracking Data/<match>.jsonl.bz2`, the PFF FIFA World Cup 2022
tracking file, 29.97 frames per second).

### 4.9 📄 Raw provider data of the two players (after reveal)

This is the evidence that the encounter is real and the label is right.

- **Distance chart:** the distance between the two players in the raw file, from 2 s before to 2 s after the hidden interval.
  The brown band is the hidden interval and the dashed line is the 0.5 m encounter radius. The curve dipping below the line shows
  a real close encounter or crossing.
- **Raw table:** every raw frame of both players.

| Column | Meaning |
|---|---|
| `player` | Player 1 or Player 2 |
| `jersey` | shirt number |
| `frame` | the provider's frame number, so it can be found in the original file |
| `video_time_s` | the provider's time |
| `t_rel_s` | time relative to the hidden interval |
| `x_m`, `y_m` | the provider's position in metres |
| `provider_visibility` | **VISIBLE** = the provider saw the player; **ESTIMATED** = the provider filled in the position |
| `provider_confidence` | LOW / MEDIUM / HIGH, as given by the provider |
| `hidden_from_model` | True = removed from the model input |
| `distance_between_players_m` | distance between the two players in that frame |

- **Download these raw rows (CSV):** saves the table.

The raw data is locked until the reveal because it shows where each player went, which is the answer.

### 4.10 🔍 Why could the models fail here? (after reveal)

- **Automatic risk factors** for this scenario:
  - gap different from 0.8 s (GRU C was trained only on 0.8 s);
  - added noise; dropout or irregular sampling; short context;
  - a shift beyond the ±0.3 s used in training;
  - a movement type with high error in Milestone 2;
  - simple physics pointing the wrong way (the players turned, slowed or accelerated while hidden);
  - low model confidence.
- **Physics reference: constant-velocity mismatch (m)**, a simple non-learned reference that is not one of the deployed models.
  - Each player is extended forward and each track backward across the gap at constant speed and direction.
  - The table gives the mismatch in metres for the four links; the pairing with the smaller total wins.
  - When physics is wrong but the models are right, the models learned more than straight-line motion. When everyone is wrong,
    the movement inside the gap was genuinely unpredictable from the outside.
- **Movement type table:** this encounter's descriptors and the Milestone 2 baseline accuracy of encounters with the same
  descriptor (how many there are, GRU accuracy, HGB accuracy, routed error).

| Descriptor | Values |
|---|---|
| paths crossing | local crossing / no crossing / window-only crossing |
| relative direction | same direction / crossing angle / opposite direction / slow player |
| distance trend | converge then diverge / converging, stay close / close then diverge / parallel close |
| largest heading change | straight < 30° / turn 30–90° / sharp > 90° |
| speed change | steady / more than 50 % |
| provider status in the gap | both VISIBLE / mixed / both ESTIMATED |
| teams | same team / opposite teams |

Milestone 2 found that **both-VISIBLE** encounters are clearly harder than provider-ESTIMATED ones (error 14.7 % vs 3.1 % at
the baseline).

### 4.11 Advanced information (match, fold, checkpoints, provenance)

- Match, encounter, period and scenario id.
- The fold's 4 unseen matches.
- The exact parameters; the label source (`stage2_accepted_g0p80` = accepted at 0.8 s; `stage2_label_quality_rerun` = re-checked
  for the new gap or shift).
- The hidden interval [g0, g1] in provider time.
- The policy thresholds.
- The checkpoints used: SHA-256 of the GRU `.pt` and HGB `.joblib` files.
- The model input summary from the adapter: valid points per tracklet, frame mode, observations ignored inside the gap.
- **Cross-check with the saved Milestone 2 run**: saved vs live probability, absolute difference, and tolerance.
- **Download model-visible input (JSON):** exactly what the models see, with no answer.
- **Download ground truth (JSON):** only after reveal.

### 4.12 Scenarios generated this session (Research lab)

The last 12 scenarios: match, encounter, variant, scenario id and whether it was valid.

### 4.13 Invalid scenario message

"**Invalid scenario: not shown to any model**", with the reason, for example
`label_quality_REJECTED:continuous_reference`, which means that at this gap one player's provider track is not continuous enough
to give a trustworthy pairing. The dashboard never invents a label.

## 5. 🖼 Example gallery

Examples come from the saved Milestone 2 run (55,680 scenarios). **Category** offers:

- GRU correct / GRU wrong (routed to GRU and accepted);
- HGB correct / HGB wrong (routed to HGB and accepted);
- GRU and HGB disagree;
- low-confidence abstention that avoided an error;
- low-confidence abstention of a correct answer;
- longer gap (1.6 / 2.0 / 2.4 s);
- noisy trajectories (10 / 25 / 50 cm).

The number in brackets is how many scenarios are in the category.

| Control | What it does |
|---|---|
| **Load #1 … #6** | The fixed examples: the first six of the category in scenario-id (hash) order, not chosen for looking good. |
| **🎲 Random from category** | Any scenario of the category, chosen by a fresh random seed (shown). |

Loading an example switches to Research lab mode and rebuilds the scenario; press **Run prediction** on the Scenario tab to
compute it live. The category name hints at the outcome, so for a fair demonstration use the random buttons on the Scenario tab.

## 6. 📊 Evaluation (Milestone 2)

These are saved results, with no new computation.

- **What kind of evidence is this?** The five kinds:
  - existing checkpoints;
  - unseen-match evaluation;
  - synthetic interruptions;
  - unsupported / out-of-distribution inputs;
  - the frozen held-out evaluation, which has **not** been run.
- **Headline numbers:**
  - Scenarios 55,680, of which Valid 55,287.
  - **Routed coverage** 68.1 %: the share of scenarios where the system committed to an answer.
  - **Realised error (target 5 %)** 8.0 %: the share of committed answers that were wrong.
  - **Accepted wrong** 2,999.
- **Yellow box:** the measured limitation of `m11-routing-v1`. About 8 % of committed answers are wrong across the stress
  suite, not 5 %. The policy cannot see dropout or short context, trusts the declared noise, and its thresholds were set at 0.8 s.
  The policy is shown unchanged.
- **Factor** (gap / gap centre shift / noise / dropout / context): a chart of GRU C (blue) and HGB C (orange) balanced accuracy
  per level, with 95 % confidence bars. Grey ✕ marks levels the policy does not support (diagnostic only). 0.5 = chance.
- **Table columns:**

| Column | Meaning |
|---|---|
| valid | how many of the 1,920 encounters were valid at this level |
| GRU / HGB BA and 95 % CI | balanced accuracy and its match-grouped bootstrap interval |
| log loss | how good the probabilities are; lower is better |
| routed model | which model the policy used |
| coverage, realised error | as in the headline numbers |
| accepted / wrong / abstain low conf. / unsupported | counts of each outcome |
| invalid | variants whose label could not be verified |

- **Movement families (baseline 0.8 s):** accuracy per movement descriptor.
- **Download all saved metrics (JSON).**

**Balanced accuracy** is the average of the accuracy on MATCHED cases and on SWAPPED cases, so it cannot be inflated by
always guessing one answer.

## 7. 🧠 How the models work

**The problem.** Two possible answers; each model outputs **p = P(SWAPPED)**. The models use positions and times only: no video,
appearance, team or jersey.

**Step 1 · Input preparation (both models)**

- 2.0 s before and 2.0 s after the gap, resampled at **15 Hz**, giving 30 points per tracklet. Only observations outside the
  hidden interval are used; missing points are *masked*, never invented.
- Positions are rotated and shifted into an **encounter frame**: origin between the two players at the gap, x-axis along their
  incoming direction. The model therefore learns relative geometry, not pitch location.
- Exchanging Track X and Track Y turns p into exactly 1 − p (**symmetry**).

**Step 2a · GRU C, a recurrent neural network** (diagram in the tab)

- **Input per tracklet:** 30 steps × 4 channels: x / 10 m, y / 10 m, mask (1 = observed), time to the gap edge in s.
- **Two GRU encoders** (*Gated Recurrent Unit*, hidden size 48), one for incoming and one for outgoing tracklets. A GRU reads the
  points one after the other and keeps a memory, learning speed, direction and turning. Its final state becomes a 48-number
  *embedding*.
- **Pair head:** for each Player–Track combination, the inputs [a, b, a·b, |a − b|, gap] (193 numbers) pass through a small
  network (96 → 48 → 1) that gives a compatibility *score*.
- **Output:** logit = score(P1, Y) + score(P2, X) − score(P1, X) − score(P2, Y); p = sigmoid(logit). **43,585** parameters.
- **Training:**
  - binary cross-entropy loss with the AdamW optimiser (learning rate 0.001, weight decay 0.0001), batches of 256, gradient
    clipping 1.0, dropout 0.1;
  - random mirroring and random order exchange as data augmentation;
  - early stopping on 3 training matches (patience 10), at most 160 epochs;
  - **config C:** half of the samples have the gap centred on the closest approach, half shifted (up to ±0.3 s);
  - **every training gap is 0.8 s**, which explains why GRU C fails at other gap lengths.
- **Training-curve chart:** the real per-epoch training loss and early-stopping loss of the exact fold model in use, with the
  best (kept) epoch marked. For some folds the best epoch is near 160: the model was still improving when training stopped.

**Step 2b · HGB C, gradient-boosted decision trees**

- **36 motion features:**
  - 25 *pairing* features, each the cost difference SWAPPED − MATCHED for a motion rule: distances between extrapolated and
    observed positions under nearest-point, constant-velocity and constant-acceleration models (also on a log scale), and
    differences in velocity, speed, heading, acceleration and implied speed;
  - 11 *context* features: gap, separations, speeds and angles.
- **Model:** scikit-learn HistGradientBoostingClassifier, 400 trees, learning rate 0.05, at most 31 leaves, at least 40 samples
  per leaf, L2 regularisation 1.0. Hyperparameters were chosen by inner cross-validation on training matches.
- **Symmetry:** trained on each sample and its mirrored copy; p = (f(x) + 1 − f(swapped x)) / 2.

**Step 3 · Routing and abstention:** as in 4.6.

**Step 4 · Train / test separation:** as in section 2. There are 25,331 training encounters (players within 0.5 m); the labels
come from the provider's continuous tracks.

**Live panels for the current scenario**

- **GRU C input tensors:** choose **Tracklet** (Player 1 / 2, Track X / Y) to see the exact 30 × 4 numbers the network receives.
- **HGB C input:** all 36 feature values, with their names. `perm_*` = cost(SWAPPED) − cost(MATCHED); positive favours MATCHED.

## 8. 📄 Load JSON tracklets

This tab runs the models on **four manually prepared tracklets**: two incoming and two outgoing, each with time in seconds and
x, y in metres on a pitch. The format is `m11-trajectory-input-v1`, the same as **Download model-visible input**.

| Control | What it does |
|---|---|
| **Input: Upload a JSON file** | Choose a file with **Trajectory JSON**. |
| **Input: Use the current scenario (model-visible part)** | Reuses the scenario on the Scenario tab. |
| **Fold models** | For external input, any fold (none of them saw it). For a development match id, the unseen fold is chosen automatically. Held-out match ids are refused. |
| **Declared positional noise σ (m)** | How accurate the coordinates are. Video-derived positions are typically 0.4–1 m off, far beyond the supported 5 cm, so they will be **unsupported**. |
| **Known true pairing** | unknown / MATCHED / SWAPPED; only if a trustworthy pairing exists, for example a careful manual annotation. |
| **▶ Run prediction on this input** | Runs both models and the routing policy. |

Without a true pairing the result is marked **Diagnostic only**. A raw video file (MP4) cannot be analysed here: detection,
tracking and pitch calibration are not part of this tool.

## 9. ❓ Guide & legend

This tab has a short description of the lab, the legend of terms, and the 3–5 minute demonstration outline.

## 10. Glossary

| Term | Meaning |
|---|---|
| Encounter | Two players within 0.5 m of each other in the raw data. |
| Tracklet | A short piece of one player's trajectory (here up to 2 s). |
| Incoming / outgoing tracklet | Before / after the hidden interval. |
| Hidden interval, gap, simulated occlusion | The time during which both players' positions are removed from the model input. |
| MATCHED / SWAPPED | Player 1 → Track X and Player 2 → Track Y / the reverse. |
| P(SWAPPED), probability | The model's output; P(MATCHED) = 1 − P(SWAPPED). |
| Confidence | max(P, 1 − P). |
| Threshold, operating point | The minimum confidence to commit. "5 %" = set so that about 5 % of committed answers were wrong on development data at the 0.8 s baseline. |
| Abstention | The system declines to answer (low confidence or unsupported input). |
| Routing policy `m11-routing-v1` | The fixed rule choosing GRU C, HGB C or abstention. |
| Coverage | Share of inputs where the system commits. |
| Realised error | Share of committed answers that are wrong. |
| Fold, outer fold | One of 8 groups of 4 development matches; its models were trained without them. |
| Unseen-match | Evaluated on a match the model never trained on. |
| Held-out matches | 8 matches frozen for the final evaluation; never used here. |
| Provider | PFF, the source of the World Cup tracking data. VISIBLE / ESTIMATED is the provider's own flag. |
| Seed | A number that fixes a random draw so it can be repeated. |
| Diagnostic only | A number computed for interest outside the validated range; not a validated result. |
| Balanced accuracy, log loss, 95 % CI | See section 6. |
| Constant-velocity reference | Straight-line extrapolation, used only as an explanation, not as a model. |

## 11. Known limitations (state them honestly)

- The occlusion is simulated on real provider data; it is not a video occlusion.
- Development matches only; the final held-out evaluation has not been run.
- GRU C was trained only on 0.8 s gaps and clean data, and its epoch budget was binding in some folds.
- Policy `m11-routing-v1` has about 8 % realised error across the stress suite (target 5 %). It is shown unchanged; a revised
  policy has not been approved.
- All encounters are close (< 0.5 m); clearly separated pairs are not in the data.
- Video-level position error (0.4–1 m) is outside what the models support.
