# Session A: research, code, experiment and dashboard audit (2026-11-19)

**Repository:** `F:\Football-Fatigue-Project` · **Branch:** `feature/m11-identity-finalization` · **HEAD:** `1bd016d8099e32d94807aaca805028e8621007f8`

**Scope:** read-only audit of the M11 two-player trajectory identity-recovery research (Stages 1–4C, Milestones 1–3) and of the
*M11 Trajectory Recovery Lab* dashboard, plus a paper-feasibility assessment and a roadmap. **Nothing was implemented or fixed.**
This file is the only file created in the repository by this audit (see §15).

## Evidence labels

| Label | Meaning |
|---|---|
| **[V]** | Verified on 2026-11-19 from current source, Git or saved development artifacts, with read-only commands |
| **[R]** | Ran on 2026-11-19 (a read-only computation or test, stated explicitly) |
| **[DOC]** | Reported in an existing project document; not recomputed |
| **[CLAIM]** | Historical assistant statement (memory, earlier session transcript); not re-checked |
| **[NV]** | Not verified |
| **[HYP]** | Hypothesis or interpretation by this audit, not an observed fact |
| **[PROP]** | Proposal only; nothing implemented |

**Protected data:** no held-out match (3822, 3824, 3834, 3839, 3844, 10510, 10514, 10517), no official FIFA validation/test data, no
SoccerTrack reserved sequence and nothing in `F:\Football-Fatigue-Video` was opened. All numbers below come from development
artifacts or documents.

---

## 1. Executive summary

1. **What exists.** A complete *development* pipeline for one narrowly defined problem: two players who come within 0.5 m in real
   PFF FIFA World Cup 2022 tracking are hidden for a simulated interval (0.8 s by default); given their two identified *incoming*
   tracklets and two anonymous *outgoing* tracklets, predict MATCHED vs SWAPPED. Two model families (GRU C, HGB C) were trained as
   8 match-grouped outer-fold models each; an abstaining routing policy (`m11-routing-v1`) and an inference package
   (`m11_infer`) exist; a 55,680-scenario stress suite (Milestone 2) and a Streamlit lab (Milestone 3) exist. [V]
2. **Integrity checks pass today.** All 16 checkpoints match their recorded SHA-256; the frozen split hash matches; the saved
   Milestone 2 metrics are the ones the dashboard reads; three of five recorded code fingerprints still match. [R]
3. **What is not done.** No final model trained on all 32 matches; no Stage 5 held-out evaluation; no routing v2; Milestones 1–3
   are **uncommitted**; no real (video) occlusion was ever tested. [V]
4. **Headline development numbers** (unseen-match out-of-fold, synthetic interruption, not independent validation) [V from
   saved metrics]: GRU C balanced accuracy (BA) 0.921 [0.910, 0.932] vs HGB C 0.850 at the 0.8 s baseline (1,920 encounters).
   The routing policy reaches 5.0 % error at the baseline (by construction, see §4.9) but **8.0 %** across the stress suite.
5. **Most important scientific weaknesses:**
   - the 5 % thresholds were fitted on the same development predictions they are later "validated" on;
   - the GRU never saw any gap other than 0.8 s;
   - provider interpolation ("ESTIMATED") texture carries signal that the models exploit;
   - VISIBLE-in-gap pairs are much harder (14.7 % routed error).
6. **Dashboard.** It runs live inference correctly and separates truth from the model input in code and tests. It grew in one
   evening from 5 to 6 tabs and gained three expanders inside an already long Scenario tab, which explains why it feels less clear
   than before. I found **two pre-reveal hint leaks in the Research-lab encounter dropdown**, one hyperparameter misstatement, two
   misleading count statements, an unexposed pair of gallery categories, and several redundancies (§7–§8).
7. **Paper readiness: not ready.** There is a plausible, honest paper here: a benchmark-and-analysis paper on trajectory-only
   two-player re-association with selective prediction. It still needs baselines, a literature review, threshold calibration on
   data separate from evaluation, convergence fixes, statistics and one preregistered held-out evaluation (§11–§12).

---

## 2. Verified current state (Part 1)

### 2.1 Git and filesystem [V]

```text
## feature/m11-identity-finalization            HEAD 1bd016d (2026-11-17 21:13 +0200)
 M src/football_fatigue/m11_dashboard/__main__.py      (+ "lab" subcommand; 4 insertions, 2 deletions)
?? docs/dashboard_explaination1.md
?? docs/handoffs/                                     (Nov 18 handoff; this audit adds one file)
?? docs/m11_demo_guide.md
?? docs/research/m11_milestone1_model_readiness_v1.md
?? docs/research/m11_milestone2_scenario_testing_v1.md
?? docs/research/m11_stage4c_correction_v1.md
?? scripts/m11_lab.cmd
?? src/football_fatigue/m11_benchmark/scenario_eval.py
?? src/football_fatigue/m11_benchmark/scenarios.py
?? src/football_fatigue/m11_dashboard/lab.py
?? src/football_fatigue/m11_dashboard/lab_app.py
?? src/football_fatigue/m11_dashboard/lab_board.py
?? src/football_fatigue/m11_infer/
?? tests/test_m11_infer.py
?? tests/test_m11_lab.py
?? tests/test_m11_scenarios.py
```

- **Recent commits:** `1bd016d` (M11 research, benchmarks, dashboards), `1458e1b`, `8505c5c`, `4653047`, `3ec9441`, `936bc07`,
  `1d020fc`, `2e18001`.
- **Only commit touching `m11_dashboard/`:** `1bd016d`. The Milestone 3 lab files (`lab*.py`) have **no Git history at all**.
- **Modification times:** the lab files were last modified on 2026-11-18 at 01:41 (`lab.py`), 00:29 (`lab_board.py`) and 01:50
  (`lab_app.py`). Nothing in source has changed since the Nov 18 handoff.

### 2.2 Is the Nov 18 handoff still current?

Yes for every fact re-checked: Git state (plus `docs/handoffs/`), checkpoint hashes, split hash, fold map, HGB per-fold
hyperparameters, routing constants and code-fingerprint staleness. This audit adds findings the handoff did not contain:
§7.4 leaks, §8 items 2–9, the held-out exposure note in §3.8, and the threshold-provenance analysis in §4.9.

### 2.3 Inventory of M11 components (Session A scope)

| Component | Location | State |
|---|---|---|
| Stage 1 local states / candidates | `window_intersections/local.py`, `candidates.py` | Implemented, committed [V] |
| Stage 2 candidates + frozen split | `window_intersections/split.py`; `data/processed/m11-stage2-candidates-v1`, `m11-stage2-split-v1` | Done, committed code; split SHA `66b2321a…6c0` [R] |
| Stage 3 supervised samples | `window_intersections/samples.py`; `m11-stage3-samples-v1` | Done, committed [V]; 25,331 rows / 18,777 independent in `g0p80` [R] |
| Stage 4A/4B benchmarks | `m11_benchmark/{data,folds,metrics,models,run,evaluate,robust,gates}.py`; `m11-stage4-benchmarks-v1` | Done, committed [V] |
| Stage 4C robustness / selection | `m11_benchmark/{shiftaug,stage4c,stage4c_eval}.py`; `m11-stage4c-robustness-v1` | Done, committed [V] |
| M1 inference package | `m11_infer/{bundle,adapter,predict,fifa,__main__}.py`; `m11-infer-v1/bundles_v1.json` | Done, **untracked** [V] |
| M2 scenario suite | `m11_benchmark/{scenarios,scenario_eval}.py`; `m11-scenarios-v2` | Done, **untracked**; code fingerprints stale (§9) [R] |
| M3 Trajectory Recovery Lab | `m11_dashboard/{lab,lab_board,lab_app}.py`, `__main__.py` (modified), `scripts/m11_lab.cmd` | Done, **untracked**, awaiting review [V] |
| Tests | `tests/test_m11_infer.py` (18 functions), `test_m11_scenarios.py` (13), `test_m11_lab.py` (18) | Present [V]; run result in §9.6 |
| Final 32-match training | — | **Not started** [V] |
| Stage 5 held-out build/eval | protocol `m11_stage5_final_protocol_PROPOSED_v1.md` | **Proposal only**; builders refuse held-out [V] |
| Routing v2 | M2 report §(routing v2) | **Proposal only** [DOC] |
| Dependency declaration (torch, pandas, pyarrow) | `pyproject.toml` | **Missing** [V, Nov 18; unchanged file] |
| Earlier, separate M11 tracks | `matching/` (RF, Steps 1–8), `m11_recovery/`, `intersections/`, Step 8 dashboard `app.py`, explorer `encounters_app.py` | Committed; historical, different problem definitions (§3.1) |

---

## 3. Research timeline (Part 2)

### 3.1 Pre-history: problem redefinitions before Stage 1 [DOC, CLAIM]

These tracks are relevant because they consumed data and shaped decisions.

| Period | Track | Problem | Outcome |
|---|---|---|---|
| Nov 6 | Steps 1–7 (`matching/`) | Many-to-many tracklet association at provider-natural gaps; RF with 51 features | Frozen RF validated once on 12 official val matches, then **locked official test consumed once**: AUC 0.838, relaxed coverage 0.403 / false association 0.088 |
| Nov 6–8 | Step 8 / 8.1 dashboard (`app.py`) | Visualise the RF on development matches (in-sample) | Six-tab app with ★ Advisor Demo (§7.1) |
| Nov 8–9 | Advisor intersection pipeline, contacts, `m11_recovery` | 4-segment samples; identity-free contacts; GBM / hybrid GRU | Label-leak found: selecting by geometric crossing leaks the label ("assume crossed" ≈ 75 %). The `m11_recovery` validation was run once on the official validation split |
| Nov 9–13 | Occlusion catalogue, exact / spatiotemporal / sync-proximity, window-intersection methods A/B/C, four-method pilot | Defining *what an encounter is* | Exact simultaneity non-generic; 0.25 m proximity adopted then broadened; several concepts superseded |
| Nov 13 | User decisions | VISIBLE and ESTIMATED equally eligible; status never a model input | Basis for Stage 1–2 |

**Consequence [V from docs]:** the official FIFA validation and test splits are consumed, so the only clean evaluation left inside
FIFA is the 8-match internal held-out set.

### 3.2 Stage 1: local states and candidates (2026-11-13) [DOC]

1. **Objective:** define identity-free encounter candidates and per-method crossing states. Pilot on 3812 and 3816 only.
2. **Input / eligibility:** Step 1 caches (`m11-encounters-v2`). An encounter is an identity-free episode with d(t) ≤ 0.50 m,
   merged within 1 s. Crossing cohorts F1–F4 are metadata only.
3. **Samples / labels:** none (descriptive).
4. **Algorithms:** `local.method_states` (PASS / FAIL / UNDET), truncation and occupancy causes, chord-speed plausibility
   (12 m/s), resumable runner; `candidates.py`.
5. **Training / evaluation:** none.
6. **Splits / leakage:** a held-out set was proposed (not frozen).
7. **Artifacts:** `m11-window-local-methods-pilot-v4` (valid); v3 and the v3-aborted directory are kept.
8. **Findings:** 2,328 candidates (425 non-crossing); 395 HIGH at 0.8 s; a VISIBLE ≥ 0.5 s rule would remove 1,918; 1:1
   within-match controls are infeasible.
9. **Limitations:** two matches only.
10. **Decision:** the user made VISIBLE and ESTIMATED equally eligible, and the held-out set was chosen as least exposed.

### 3.3 Stage 2: candidates on 40 matches and frozen split (2026-11-13) [DOC; split hash R]

1. **Objective:** a full candidate catalogue and a frozen internal split.
2. **Input:** the 40 official *development* matches. The held-out set excludes 10 previously modelled matches and takes the first
   8 by `sha256("m11-stage2-heldout-v1|id")`.
3. **Labels:** `candidates.label_quality` decides ACCEPTED / LOWER / REJECTED per gap from the two continuous provider chains.
4. **Algorithms:** `split.py`, `candidates.py`.
5. **Training:** none.
6. **Leakage control:** the split was frozen with SHA `66b2321a…6c0` [R].
7. **Artifacts:** `m11-stage2-candidates-v1` (40 matches).
8. **Findings:** 36,732 candidates; at 0.8 s, 31,219 ACCEPTED / 23 LOWER / 5,490 REJECTED. Train/CV has 25,331 (18,777
   independent); held-out 5,888. 51 % are fully ESTIMATED in the gap.
9. **Limitations:** controls ≈ 0.2:1; near-coincidence 3,502; jump risk 471.
10. **Decision:** build Stage 3.

### 3.4 Stage 3: supervised dataset (2026-11-14) [DOC; counts R]

1. **Objective:** fixed-format supervised samples.
2. **Input:** the 32 train/CV matches only; `samples.py` **refuses** held-out matches [V in source].
3. **Construction:**
   - Grid at 1/15 s with T = 30 points per side.
   - The hidden interval `[g0, g1] = [tc − g/2, tc + g/2]`, with `tc` the closest approach; only observations strictly outside it
     are used.
   - Encounter frame from the incoming tracklets only (modes 0/1/2).
   - Label y: 0 = MATCHED, 1 = SWAPPED. Input order bits are randomised, so the swapped share is 0.499.
   - Labels verified against raw slots for 100,832 / 100,832 samples.
4. **Tables:**

   | Table | Samples |
   |---|---|
   | `g0p80` (primary) | 25,331 (18,777 independent) [R] |
   | `g0p40` | 25,668 |
   | `g1p60` | 24,661 |
   | `g0p80_shift` | 25,172 |

5. **Training:** none.
6. **Leakage:** independence is defined by overlap groups. The Stage 3 *samples* are not independent encounters: 25,331 rows
   contain 18,777 independent ones.
7. **Artifacts:** `m11-stage3-samples-v1` (141 MB).
8. **Findings:** construction succeeds for 100 % of g0p80.
9. **Limitations:** every primary sample has `gap_s = 0.8`.
10. **Decision:** benchmark plan (Stage 4).

### 3.5 Stage 4A / 4B: baselines, tabular ML and neural benchmarks (2026-11-14) [DOC]

1. **Objective:** compare physics baselines, tabular ML and sequence networks under match-grouped CV.
2. **Input:** `g0p80` on 32 matches. Outer folds are 8 groups of 4 matches by `sha256("m11-stage3-folds-v1|m")` rank
   (`m11_benchmark/folds.py:19-23`) [V]. Inner folds come from 3 groups; early stopping uses 3 matches drawn from the training
   partition (`folds.py:31-33`) [V].
3. **Samples:** all rows for training; metrics on the independent subset.
4. **Algorithms:**
   - physics combos (`phys_*`);
   - LR, KNN, Nystroem-SVM, RF, HGB (4-point grid, `models.py:77-83`);
   - MLP, GRU, LSTM;
   - the corrective gap-last GRU / LSTM (`models.py:140-143`).
5. **Procedure:** 3 seeds for the neural models; 80-epoch cap; interim metrics `v1` and final `v2`.
6. **Gates and controls:** synthetic gates; a 20-permutation null.
7. **Artifacts:** `m11-stage4-benchmarks-v1`.
8. **Findings (independent BA):**

   | Model | BA |
   |---|---|
   | `gru_gaplast` | 0.902 ± 0.005 (ECE 0.008) |
   | HGB | 0.870 |
   | SVM | 0.859 |
   | RF / LSTM-gaplast | 0.857 |
   | MLP | 0.845 |
   | LR | 0.837 |
   | KNN | 0.822 |
   | Best physics combo | 0.696 |

9. **Limitations:**
   - the GRU lead vanishes on smoothed or noised data and at 1.6 s;
   - fully VISIBLE gaps are harder (GRU 0.812);
   - the 80-epoch cap binds.
10. **Decision:** run Stage 4C robustness.

### 3.6 Stage 4C: shift robustness and model selection (2026-11-14) [DOC; correction note DOC]

1. **Objective:**
   - robustness to the placement of the gap centre;
   - a longer budget;
   - a texture probe;
   - nested GRU/HGB ensembles;
   - selection of the candidates.
2. **Input:** `g0p80` plus `m11-stage4c-shiftaug-v1`: 150,895 shifted samples (±0.1 / 0.2 / 0.3 s), **all at gap 0.8 s**.
3. **Configs** (`stage4c.py:9-16`) [V]: A (centred); B (uniform over versions); **C** (each epoch: centred with p = 0.5, otherwise
   one valid shifted version; one draw per encounter per epoch); Anoise; Asmooth.
4. **Procedure:** one 160-epoch run per fold records the best early-stopping state at caps 80 / 120 / 160. HGB C uses sample
   weights (0.5 centred, 0.5 / n over the shifted versions; `stage4c_eval.py:311-328`) [V] and **reuses the Stage 4 inner-CV
   hyperparameter choice** (`stage4c_eval.py:341`) [V].
5. **Leakage:** same folds.
6. **Artifacts:** `m11-stage4c-robustness-v1` (`metrics/v1`, `audit/audit_v2.json`).
7. **Findings:**

   | Model | Centred | Shifted | Other |
   |---|---|---|---|
   | GRU C | 0.926 | 0.910 | VISIBLE 0.845 |
   | HGB C | 0.861 | 0.878 | — |

   - A texture-only probe pairs tracklets at BA 0.701 (ESTIMATED 0.739, VISIBLE 0.525).
   - The nested ensemble chose pure GRU in 24/24.
8. **Correction note [DOC]:** GRU C hit the 160-epoch cap in 13 of 24 fold fits, so it is probably under-converged. Config C and
   the 160-epoch budget were chosen after looking at the same OOF results, so they are **selection-optimistic**.
9. **Decision:** GRU C is primary and HGB C is the reference; a Stage 5 protocol was **proposed** (not approved).

### 3.7 Milestones 1–3 (2026-11-17 / 18)

| | M1 Model readiness | M2 Scenario testing | M3 Trajectory Recovery Lab |
|---|---|---|---|
| Objective | Reusable inference + routing + checkpoint verification | Reproducible one-factor stress suite | Interactive live testing / demo |
| Inputs | Stage 4C seed-0 GRU C `c160` + HGB C fold models | 60 encounters per match × 32 matches = 1,920 (seeded, model-independent selection) | M2 generator + M1 predictor |
| Construction | `adapter.build_inputs` re-implements the Stage 3 grid from timestamped tracks | 29 variants per encounter (baseline + 8 gap, 6 shift, 6 noise, 5 dropout, 3 context) = 55,680; 393 invalid (label check re-run) | Same generator; custom combinations allowed |
| Evaluation | Reproduction vs stored OOF: GRU CPU max abs Δp 9.1e-5, HGB 0; adapter rebuild 60/60 [DOC] | Match-grouped bootstrap CIs (1,000) [V in metrics] | 22 tests + browser walkthroughs [CLAIM] |
| Output | `m11-infer-v1/bundles_v1.json` | `m11-scenarios-v2` (v1 aborted, kept) | Streamlit app on port 8610 |
| Status | Done, untracked | Done, untracked | Done, untracked, awaiting review |

### 3.8 Counts that must not be confused [R unless noted]

| Quantity | Value |
|---|---|
| Stage 2 accepted candidates at 0.8 s, train/CV | 25,331 rows (= Stage 3 g0p80 samples) |
| …of which **independent** (overlap-group representatives) | **18,777** |
| Fitting samples per fold, centred (GRU) | 19,419–20,250 (fold 0: 19,783; 1: 20,250; 2: 19,419; 3: 19,833; 4: 19,689; 5: 19,786; 6: 19,491; 7: 19,648) |
| Early-stopping samples per fold | 2,246–2,777 (3 matches) |
| HGB C training rows per fold | fit + early-stop matches (fold 0: 22,200 centred), plus weighted shifted copies, ×2 for the mirrored order |
| Unseen-fold test rows per fold | 2,741–3,437 (independent 2,077–2,517) |
| Shift-augmented variants | 150,895: **generated variants of the same encounters, not new observations** |
| M2 encounters | 1,920 (a subset of the independent g0p80 set) |
| M2 scenarios | 55,680 = 1,920 × 29 **variants**; 55,287 valid. Effective independent units: 1,920 encounters in 32 matches |

**Held-out exposure note [DOC: `m11_stage2_results_v1.md:64`; sync-proximity / window-intersection reports].** The 8 held-out
matches were never used for training, selection or thresholding. Their raw data **was processed** before the split was frozen and
in Stage 2:
- Step 1 caches;
- sync-proximity v2 and window-intersection datasets on all 40 development matches;
- the Stage 2 candidate catalogue, with aggregate held-out counts reported (5,888 candidates).

A paper must describe them as "not used for model development" rather than "never touched".

---

## 4. Data and model audit (Part 3)

### 4.1 Why these architectures [DOC + V]

- **GRU (gap-last):** a learned sequence summary of each tracklet. The first GRU read sequences from the gap outward and underfit
  (BA 0.765). The gap-last variant reads towards the gap (`models.py:141-143,164-165`) and became the best model (0.902).
- **HGB:** the strongest tabular model on hand-engineered motion features. It is deterministic, fast, and transfers better across
  gap lengths.
- **Selection:** both were chosen empirically from the Stage 4 benchmark on the same OOF data that is reported (selection bias,
  §4.8).

### 4.2 Input / output contract [V]

- **Input** (`m11_infer/adapter.py`; `predict.py:159-161`): two incoming and two outgoing `Track(t, x, y)` objects in seconds and
  metres, the hidden interval `g0, g1`, and optionally `noise_sigma_m`.
- **Output:** p = P(SWAPPED), where SWAPPED means in0→out1, in1→out0 (`predict.py:3-4, 43-44`).
- **Decision** (`predict.py:108-113`): `decide()` returns SWAPPED if p > 0.5 (ties → MATCHED) and abstains if max(p, 1 − p) is
  below the threshold.
- **Prediction record** (`predict.py:83-105`): decision, pairing, p, confidence, threshold, support, model, reasons, both models'
  probabilities, fold, checkpoint hashes and an input summary.

### 4.3 GRU C [V]

- **Tensors** (`m11_benchmark/data.py:105-113`): shape (N, 2, T = 30, 4) per side, with channels x / 10 m, y / 10 m, mask and dt
  (s).
  - Masked steps are zero except the mask channel.
  - Incoming sequences are stored reversed (index 0 nearest the gap). The encoder flips every sequence (`models.py:164-165`), so
    the gap-adjacent point is processed last for both sides.
- **Normalisation:** positions are in the encounter frame and divided by 10 m. There is no learned normalisation.
- **Encoder** (`models.py:152-169`): a `GRU(4→48)` per side (separate weights for incoming and outgoing). The summary is the
  output at the last valid (mask) step, followed by `Linear(48,48)+ReLU`.
  - Masked steps are still fed to the GRU as zeros.
- **Pair head** (`models.py:171-187`): for each (i, j), `[a, b, a·b, |a−b|, gap]` (193) → 96 → ReLU → Dropout 0.1 → 48 → ReLU → 1.
  - **Logit = s(0,1) + s(1,0) − s(0,0) − s(1,1)**. This is exactly antisymmetric under swapping either order, so p ↦ 1 − p.
  - **43,585 parameters** [V, Nov 18 recount].
- **Training** (`models.py:206-250`; Stage 4C equivalent in `stage4c.py:120-210`):
  - BCE with logits; AdamW (lr 1e-3, weight decay 1e-4); batch 256; gradient clip 1.0.
  - Per batch (not per sample), each with p = 0.5: reflection y→−y, exchange of the outgoing order (label flip), exchange of the
    incoming order (label flip).
  - Early stopping on the log loss of 3 early-stop matches (improvement > 1e-4, patience 10).
- **Fold training records (seed 0, cap 160)** [R]:

  | Fold | Best epoch | Epochs run | Budget binding? |
  |---|---|---|---|
  | 0 | 159 | 160 | **yes** |
  | 1 | 114 | 124 | no |
  | 2 | 123 | 133 | no |
  | 3 | 157 | 160 | **yes** |
  | 4 | 159 | 160 | **yes** |
  | 5 | 143 | 153 | no |
  | 6 | 136 | 146 | no |
  | 7 | 130 | 140 | no |

- **Gap input:** constant at 0.8 for every training sample, so its weight is never learned. Any other gap is out of distribution
  for the GRU [DOC correction §2, consistent with source].

### 4.4 HGB C and the 36 features [V]

**Features** (`m11_recovery/motion.py:87-165`):
- **25 antisymmetric `perm_*` features.** Each is cost(SWAPPED) − cost(MATCHED) = (c01 + c10) − (c00 + c11); positive values
  favour MATCHED.
  - 8 distance costs `d_*`: `nearest`, `cv_fwd`, `cv_bidir`, `ca_bidir`, `cv1_fwd`, `cv1_bidir`, `ca2_fwd`, `ca2_bidir`.
  - The same 8 on a log scale, `logd_*` = log(d + 0.05).
  - 9 kinematic terms: `dv`, `dspeed`, `dheading`, `dacc`, `implied_minus_speed`, `cv_mid_err_rel`, `ca_fwd`, `ca_bwd`, `cv_bwd`.
  - Fit windows: constant velocity over 0.5 s, constant acceleration over 1.0 s, long constant velocity over 1.0 s, long constant
    acceleration over 2.0 s (`motion.py:24-27`).
- **11 symmetric context features:** `gap_s`, `sep_before_end`, `sep_after_start`, `speed_before_min/max`,
  `speed_after_min/max`, `angle_before`, `angle_after`, `cv_bidir_min_pair`, `cv_bidir_abs_margin`.
- **Note:** `FEATURE_NAMES` is filled lazily on the first call to `features()` (`motion.py:115,163-164`). Importing the module
  alone shows an empty list [R].

**Model:**
- `HistGradientBoostingClassifier(min_samples_leaf=40, l2_regularization=1.0, early_stopping=False)` plus the grid configuration
  (`models.py:82-83,100`).
- Trained on X and its swapped copy with flipped labels. Prediction is p = (f(x) + 1 − f(swap x)) / 2 (`models.py:104-130`);
  the measured symmetry error is 1.1e-16 [R].

**Per-fold hyperparameters** [R from fold records]:

| Folds | learning_rate | max_iter | max_leaf_nodes |
|---|---|---|---|
| **0, 3, 7** | **0.1** | **200** | 31 |
| 1, 2, 4, 5, 6 | 0.05 | 400 | 31 |

**Documentation discrepancy** [V]: `lab_app.py:810` and `docs/dashboard_explaination1.md:399` say "400 trees, learning rate
0.05" for all folds. This is wrong for folds 0, 3 and 7. The Stage 5 proposal (`m11_stage5_final_protocol_PROPOSED_v1.md:34`)
uses 0.05 / 400 for *final* training and says explicitly that this is the inner-CV choice in 5/8 folds, which is consistent.

### 4.5 Augmentation and leakage review

| Risk | Status |
|---|---|
| Match leakage across folds | Prevented by construction: outer folds by match; early-stop matches drawn from the training partition (`folds.py:31-33`); the dashboard always uses the unseen fold (`bundle.fold_for_match`, `bundle.py:101-109`) [V] |
| Overlapping samples from one encounter | Training uses all rows; metrics use the independent subset. Overlaps stay inside one match, so they cannot cross folds [V by construction] |
| Shift augmentation | Variants of the same training encounters; never crosses folds [V] |
| **Provider texture** | ESTIMATED segments are provider interpolations with a distinctive texture; a texture-only probe reaches BA 0.70 [DOC]. **Models partly learn the provider's gap-filling, not just player motion** [HYP, supported by the probe] |
| Label from provider identity | The label is the provider's identity chain, which may itself contain identity errors; 210 encounters with d_min < 0.05 m carry reference ambiguity [DOC] |
| Selection by crossing geometry | Avoided in Stage 2 (identity-free proximity); crossing class is metadata only [DOC] |
| Threshold / model selection on reported data | **Present** (§4.8–4.9) |

### 4.6 Early stopping and convergence

- **Observed:** 3/8 seed-0 bundles are budget-bound, and 13/24 fold fits across seeds [DOC + R].
- **Interpretation [HYP]:** GRU C development figures are a lower bound for this architecture under longer training, but are also
  selection-optimistic.

### 4.7 Calibration, confidence and uncertainty [R from saved M2 metrics]

| Model | ECE | Log loss | Brier | AUC |
|---|---|---|---|---|
| GRU C (0.8 s baseline) | 0.0145 | 0.188 | 0.058 | 0.978 |
| HGB C (0.8 s baseline) | 0.030 | 0.308 | 0.100 | 0.940 |

- Neither model is post-hoc calibrated.
- Off-baseline calibration degrades sharply. Example: GRU at 0.2 s has ECE 0.15 and log loss 0.835 on the gap-common subset.
- **Uncertainty:** there is no epistemic uncertainty (one seed in the bundle; no ensemble).
- **Confidence:** "confidence" is just max(p, 1 − p).

### 4.8 Model selection and selection bias

The following were all chosen by looking at development OOF results on the same 32 matches that are reported:
- the architecture (gap-last);
- config C;
- the 160-epoch cap;
- the primary/fallback roles;
- the routing gap band.

[DOC: correction note; V for the mechanics] The development numbers are therefore optimistic. Only the frozen held-out run can
remove this bias.

### 4.9 Routing, abstention and the 5 % / 2 % operating points

**Rules** (`predict.py:54-80`; constants `predict.py:36-40`) [V]:

1. Fewer than 3 grid points in any tracklet → unsupported.
2. Gap outside [0.4, 1.6] s → unsupported.
3. Declared noise > 5 cm → unsupported.
4. Gap in [0.75, 0.85] s and noise ≤ 2 cm (or undeclared) → GRU C.
5. Otherwise → HGB C.
6. Abstain if the routed model's confidence is below its threshold.

**Provenance of the thresholds** (`bundle.py:36-40, 237-254`; `metrics.py:96-109`) [V]:
- The thresholds are the confidence values at which the *cumulative* error of the pooled seed-0 OOF predictions, sorted by
  confidence, last stays ≤ 5 % (or ≤ 2 %), on the **independent g0p80 subset (centred 0.8 s, clean)** of all 8 folds.

| Model | 5 % | 2 % |
|---|---|---|
| GRU C | 0.6425 | 0.8525 |
| HGB C | 0.7875 | 0.9029 |

Consequences:
1. **The thresholds are fitted in-sample.** The M2 baseline encounters are a subset of the same data, so the routed error of
   exactly **0.050** at the M2 baseline [R] is by construction, not validation.
2. **No margin or confidence bound.** The "last index ≤ target" rule picks the largest coverage that still meets the target. It
   is a point estimate (no Clopper–Pearson / Geifman–El-Yaniv style bound).
3. **No transfer guarantee.** The thresholds were set at 0.8 s centred clean. Nothing guarantees they hold under shift, noise,
   dropout, context loss or other gaps.

**Why about 8 % accepted-decision error under the stress suite** [R: decomposition of saved `metrics_v1.json`]:
- Total: 2,999 wrong / 37,629 accepted = 0.0797.
- GRU-routed variants contribute 2,456 of the 2,999 wrong answers (error 8.4 %); HGB-routed contribute 543 (6.6 %).
- Largest contributors:

  | Variant | Wrong / accepted | Error |
  |---|---|---|
  | `dropout=random50` | 271 / 1,693 | 16.0 % |
  | `context=0.5` | 260 / 1,696 | 15.3 % |
  | `dropout=random30` | 210 / 1,694 | 12.4 % |
  | `noise=0.02` | 176 / 1,730 | 10.2 % |
  | `gap=1.6` | 174 / 1,344 | 12.9 % |
  | `shift=−0.5` | 174 / 1,682 | 10.3 % |

- Mechanisms:
  1. **The router is blind to dropout and context length.** It only sees the gap length, declared noise and a ≥ 3-point minimum,
     so degraded inputs go to the GRU with its 0.8 s clean threshold.
  2. **The GRU band admits 2 cm noise.** Yet the GRU error at 2 cm is already 10.2 %.
  3. **HGB thresholds do not transfer.** They were calibrated at 0.8 s and do not hold at 1.2–1.6 s.
  4. **Equal weighting of variants.** The 8 % is a *design-weighted* average: one-factor variants each count for 1,920. It is not
     the error under any realistic deployment distribution.

  Points 1–3 are [V from code + metrics]; point 4 is a property of the summary statistic [V].

### 4.10 Failure mechanisms and unsupported conditions [R from metrics; DOC]

- **Unsupported conditions:**
  - gap < 0.4 or > 1.6 s;
  - noise > 5 cm;
  - fewer than 3 points;
  - not exactly 2 in / 2 out;
  - asynchronous hiding of the two players;
  - a missing or foreign outgoing track;
  - pairs farther apart than 0.5 m (never sampled);
  - video-scale noise (0.4–1 m);
  - real visual occlusion.
- **Failure mechanisms:**
  - **GRU gap extrapolation:** BA 0.62 at 0.2 s, while HGB reaches 0.94.
  - **Noise sensitivity:** GRU 0.868 at 2 cm and 0.685 at 10 cm.
  - **Dropout and short context.**
  - **HGB collapse under large shift:** 0.751 at −0.5 s and 0.713 at +0.5 s.
  - **Harder subsets (routed error at baseline):**

    | Subset | Routed error |
    |---|---|
    | Both fully VISIBLE in the gap | 14.7 % (n = 181) |
    | Both fully ESTIMATED | 3.1 % |
    | `no_crossing` | 10.5 % (n = 334) |
    | `local_crossing` | 3.4 % |
    | Straight paths | 2.3 % |
    | Sharp turns | 5.9 % |

  - Being confidently wrong (high-confidence accepted errors exist; there is a saved gallery category for them, §7.4).

---

## 5. Experiment inventory

| ID | Output directory | Code | Status | Notes |
|---|---|---|---|---|
| Stage 1 pilot | `m11-window-local-methods-pilot-v4` (+ v3, v3-aborted kept) | `local.py`, `candidates.py` | Done | 2 matches |
| Stage 2 | `m11-stage2-candidates-v1`, `m11-stage2-split-v1` | `candidates.py`, `split.py` | Done | 40 matches incl. held-out (candidate generation only) |
| Stage 3 | `m11-stage3-samples-v1` (+ pilot, pilot-rebuild) | `samples.py` | Done | 4 tables; labels 100,832/100,832 |
| Stage 4A/B | `m11-stage4-benchmarks-v1` (metrics v1 interim, v2 final; gates_FAILED_*, controls_FAILED_* kept) | `m11_benchmark/*` | Done | 3 seeds |
| Stage 4C | `m11-stage4c-robustness-v1` (audit v1 superseded, v2) | `shiftaug.py`, `stage4c*.py` | Done | GRU C/HGB C bundles live here |
| M1 | `m11-infer-v1/bundles_v1.json` | `m11_infer/*` | Done | 16 checkpoints recorded |
| M2 | `m11-scenarios-v2` (v1 = aborted crash run, kept) | `scenarios.py`, `scenario_eval.py` | Done | code fingerprints stale |
| Stage 5 | — | — | **Not run** | protocol PROPOSED |

---

## 6. Dashboard: feature-by-feature inventory (Part 4)

**Source:** `src/football_fatigue/m11_dashboard/lab_app.py` (885 lines), `lab.py` (460), `lab_board.py` (195).

**Launch:** `python -m football_fatigue.m11_dashboard --port 8610 lab` (`__main__.py`, modified).

**Inference:** "Live" means `L.run_inference` → `Predictor.predict` on the public tracklets, every time the button is pressed
(`lab.py:220-235`; `lab_app.py:129-134`). Fold models are cached per fold (`lab_app.py:54-56`) and their hashes are checked at load
(`bundle.py:170-179`).

**Legend for the inventory tables:**
- *Live/Saved*: L = live inference, S = reads saved results, G = regenerates a scenario (no model), — = UI only.
- *Truth risk*: ✔ = cannot reveal the answer; ⚠ = can hint at the answer before reveal; R = reveals truth, after reveal only.
- *Status*: F = functional (by code reading and tests), U = unverified in this audit, Red = redundant.
- *Where*: D = Demo mode, L = Research lab, Both.

### 6.1 Global (page, sidebar mode)

| Visible label | Purpose / actual behaviour | I/O | Live/Saved | Truth | Status | Function | Where |
|---|---|---|---|---|---|---|---|
| Title "M11 Trajectory Recovery Lab" + caption | Scope caveat (simulated, development, held-out not run) | — | — | ✔ | F | page body `lab_app.py:863-868`, `CAVEAT` L24 | Both |
| Sidebar radio **Mode**: "Demo" / "Research lab" | Switches the Scenario tab between the guided flow and the full controls; the other 5 tabs are identical in both modes | → `w_mode` | — | ✔ | F | `sidebar()` L331-338 | Both |
| Sidebar text "Follow the steps on the Scenario tab." | Demo-mode hint | — | — | ✔ | F | L336 | D |

### 6.2 Research-lab sidebar (`sidebar()`, L339-398)

| Visible label | Actual behaviour | Live/Saved | Truth | Status | Where |
|---|---|---|---|---|---|
| **Development match** (selectbox "id · fold f models") | Picks one of 32 train/CV matches; resets the encounter | G | ✔ | F | L |
| **Encounter (Milestone 2 selection, 60 per match)** — options "#n · mm:ss · *crossing* · *movement*" | Picks one of the 60 seeded M2 encounters | G | **⚠: shows `crossing_class` and `fam_heading_relation` before reveal (§7.4)** | F, hint risk | L |
| **Seed** + **🎲 Seeded random** | Reproducible random match and encounter (`random_scenario`) | G | ✔ | F | L |
| **🎲 Truly random encounter** | OS-entropy seed, shown in "Selected by" | G | ✔ | F | L |
| **Variant** radio: "One factor (Milestone 2)" / "Custom combination" | Chooses the M2 one-factor design or a free combination | G | ✔ | F | L |
| **Factor** selectbox (baseline, gap duration, gap placement (shift), positional noise, observation dropout, context length) | M2 factor | G | ✔ | F | L |
| **Level** select-slider | M2 level of that factor | G | ✔ | F | L |
| Custom: **Gap duration (s)** 0.2–2.4, **Gap placement: shift** −0.5…+0.5, **Positional noise σ** 0–50 cm, **Observation dropout**, **Context per side** 0.5–2.0, **Noise / dropout / order seed** | Any combination through `SC.build_custom` (labels re-checked); no saved cross-check | G | ✔ | F | L |
| Caption "Several factors at once are allowed…" | Explains why there is no cross-check | — | ✔ | F | L |

### 6.3 Scenario tab: Research lab (`scenario_tab_lab`, L403-423)

| Visible label | Actual behaviour | Live/Saved | Truth | Status | Function |
|---|---|---|---|---|---|
| Header "Match … · encounter … · variant · models: outer fold k (never trained on match …)" | Context line | — | ✔ | F | `scenario_header` |
| "Selected by: …" | Shows the origin of the selection (seed) | — | ✔ | F | L412 |
| Invalid-scenario error box | Shown when the label check rejects the variant; no model is run | G | ✔ | F | `invalid_block` |
| **▶ Run prediction** | Runs both fold models + routing now | **L** | ✔ | F | `action_row`/`run_prediction` |
| **👁 Reveal ground truth** | Sets `revealed`; disabled after use; **no "hide again"** | — | R | F | `reveal` |
| Caption "Prediction computed at … (live model call #n)" | Proof of liveness | — | ✔ | F | `action_row` |
| **View** radio: Encounter close-up / Surrounding play / Full pitch | Zoom | — | ✔ | F | `playback` |
| **Playback** radio: Animation / Step by step | Plotly animation vs server-side stepping | — | ✔ | F | `playback` |
| Animation: **▶ Play / ⏸ Pause** + time slider (inside the Plotly figure) | Client-side frames at 15 Hz | — | ✔ before reveal (test `test_board_before_reveal_does_not_leak`) | F; styling: buttons render low-contrast (screenshot `shot4_a_home.png`) | `lab_board.board` |
| Step by step: **⏮ Start, ◀ −1 frame, +1 frame ▶, +0.5 s ⏩, Before gap, In gap, After gap**, slider **Time (frame)** | Server-side frame selection | — | ✔ | F | `playback` |
| Pitch board | Incoming circles (Player 1/2), outgoing diamonds (Track X/Y), grey others, brown "HIDDEN" badge, phase badge; after prediction a white line (solid = accepted, dashed = abstained lean) after `g1`; after reveal the tracks are recoloured and the true hidden path is dotted | L result + R | ✔ before reveal; R after | F | `lab_board.board` |
| Model cards **GRU C**, **HGB C** | P(MATCHED)/P(SWAPPED) bar, pairing, confidence vs that model's 5 % threshold, routed / comparison / diagnostic note | L | ✔ | F | `model_card` |
| **Operational decision** card | ACCEPTED / ABSTAIN with reasons, gap, declared noise, runtime, plus "Milestone 2, this condition: coverage …, realised error …" | L + S | ✔ | F | `routing_card`, `measured_limit` |
| **Ground truth** box | Pairing, mapping, jerseys / team sides, per-model correct/wrong, outcome, abstention verdict | — | R | F | `truth_block`, `L.evaluate` |
| Expander **✅ Is this a valid test?** | Table of checks (development match, not held-out, fold unseen, not early-stop, checkpoint SHA ×2, label accepted, label re-verified) + d_min, clock, raw file | — | ✔ | F; two rows are derived from one boolean (§8.6) | `evidence_blocks`, `L.validity` |
| Caption "🔒 Raw provider data … unlock after Reveal" | Pre-reveal lock notice | — | ✔ | F | `evidence_blocks` |
| Expander **📄 Raw provider data of the two players** (expanded in lab) | Distance-vs-time chart (hidden interval shaded, 0.5 m line); raw rows table (frame, time, x, y, VISIBLE/ESTIMATED, confidence, `hidden_from_model`, distance) | Raw file read | R (after reveal only) | F | `L.raw_window` |
| **Download these raw rows (CSV)** | CSV of the table | — | R | F | `evidence_blocks` |
| Expander **🔍 Why could the models fail here?** | Rule-based reasons (gap ≠ 0.8, noise, dropout, short context, shift > 0.3, families with routed error ≥ 8 %, physics disagrees, low confidence); physics constant-velocity table; movement-family table with M2 accuracies | S (families) + computed | R | F | `failure_analysis`, `L.physics`, `L.difficulty` |
| Expander **Advanced information** | Match/fold/unseen list, params, label source, [g0, g1], policy + thresholds, checkpoint hashes, model input summary, **cross-check with saved M2** (saved vs live p, tolerance) | L + S | ✔ | F | `advanced`, `L.saved_crosscheck` |
| **Download model-visible input (JSON)** | `m11-trajectory-input-v1` without truth | — | ✔ (test `test_public_carries_no_truth`) | F | `Public.input_json` |
| **Download ground truth (JSON)** (after reveal) | Pairing, mapping, player identities | — | R | F | `advanced` |
| Expander **Scenarios generated this session** | Last 12 scenarios | — | ✔ | F | L422-423 |

### 6.4 Scenario tab: Demo mode, 7 steps (`scenario_tab_demo`, L446-495)

| Step (progress text) | Visible controls | What happens in code | Notes |
|---|---|---|---|
| 1 Select a scenario | **◀ Back / Next ▶**, info text, **🎲 Pick a random unseen-match scenario** | The random button draws a fresh OS seed and jumps to step 2 | **A default scenario (match 3812, first encounter) is already loaded, so "Next ▶" works without ever picking at random**: the demo then always shows the same encounter (`default_spec`, L74-77; screenshot `shot4_a_home.png`) |
| 2 Watch the approach | Board starts at frame 0 | — | — |
| 3 The hidden interval | Board starts mid-gap | — | — |
| 4 Run GRU and HGB | **▶ Run prediction / 👁 Reveal ground truth** appear | Same functions as the lab | Reveal is already available at step 4, two steps before "Reveal the truth" |
| 5 Compare predictions | Cards visible | — | — |
| 6 Reveal the truth | Truth box; raw data and failure analysis expanders (collapsed) | — | Reveal possible without running a prediction ("No prediction was run") |
| 7 Try another | **🎲 Another random scenario**, **Gap for this encounter (s)** {0.8, 0.4, 0.6, 1.0, 1.2, 1.6, 2.0, 2.4, 0.2}, **↻ Same encounter with this gap** | M2 gap variants of the same encounter; returns to step 2 | Works; gaps 0.2/2.0/2.4 show the unsupported abstention |
| All steps | **Legend and terms** expander; **Advanced information** expander; validity expander | Research-oriented panels (checkpoint JSON, hashes) also appear in Demo | Overcrowding (§7.3) |

### 6.5 Other tabs (identical in both modes)

| Tab / control | Actual behaviour | Live/Saved | Truth | Status | Function | Proposed home |
|---|---|---|---|---|---|---|
| **🖼 Example gallery**: **Category** selectbox (9 categories), table, **Load #1…#6**, **🎲 Random from category** | Loads a saved M2 scenario, regenerates it and **silently switches mode to Research lab** (`ss["w_mode"] = …`, L527) | G (+ S for list) | **⚠ category names state the outcome** (documented in the tab text) | F; 2 saved categories not offered (§7.4) | `gallery_tab` | B |
| **📊 Evaluation (Milestone 2)**: 5 metrics, warning box, **Factor** radio, BA-with-CI chart, table, movement-family table, **Download all saved metrics (JSON)** | Reads `m11-scenarios-v2/summary/metrics_v1.json` | **S** | ✔ | F | `evaluation_tab`, `factor_chart`, `L.factor_table` | D |
| **🧠 How the models work**: problem, Step 1 (input), 2a GRU (graphviz diagram), 2b HGB, 3 routing, 4 separation; live **Tracklet** selectbox with the GRU tensor table; the 36 HGB feature values; **GRU training curve** for the current fold | Static text + live features (no model call) + saved training history | text + computed + S | ✔ | F; **HGB hyperparameter error (L810)**; count wording (§8) | `models_tab`, `L.model_inputs`, `L.training_history` | C |
| **📄 Load JSON tracklets**: **Input** radio (upload / current scenario), **file uploader**, **Fold models** selectbox (external), **Declared positional noise σ (m)**, **Known true pairing**, **▶ Run prediction on this input**, cards + board | Refuses held-out match ids; user-chosen fold for external data | **L** | ✔ (user-supplied truth only) | F | `json_tab`, `L.from_json` | B |
| **❓ Guide & legend** | Caveat, legend, 3–5 minute script | — | ✔ | **Red**: legend duplicated in Demo; script duplicated in `docs/m11_demo_guide.md` | `guide_tab`, `legend` | A |

---

## 7. Dashboard evolution and proposed redesign

### 7.1 Reconstructed evolution

**Evidence:**
- Git (`1bd016d` holds only the older apps);
- the session transcript (Write/Bash events);
- the original `lab_app.py` text recovered from the transcript (695 lines, 2026-11-17 22:07);
- screenshots in the session scratchpad;
- `docs/research/m11_step8_interactive_dashboard.md`.

| # | Date / time | Version | Organisation | Evidence |
|---|---|---|---|---|
| 0 | Nov 6–8 | **Step 8 / 8.1 RF dashboard** (`app.py`, committed) | Six numbered tabs, one job each: **★ Advisor Demo**, **1 · Match Explorer**, **2 · Tactical Replay**, **3 · Identity Recovery**, **4 · Model Explanation**, **5 · Research Results**; filters in the sidebar; 6-step demo | `app.py:399-400`; Step 8 report §7, §14 |
| 0b | Nov 10–11 | **Football Trajectory Explorer** (`encounters_app.py`, committed) | 3 tabs: Trajectory explorer / Dataset overview / Review log | `encounters_app.py:140-151` |
| 1 | Nov 17 22:07 | **Lab v1** (`lab_app.py` first write, 695 lines) | 5 tabs: 🎬 Scenario, 🖼 Example gallery, 📊 Evaluation (Milestone 2), 📄 Load JSON tracklets, ❓ Guide & legend; Mode "**Advisor demo**" / "Research lab" | recovered text; screenshot `shot1_c_pred.png` |
| 1a | Nov 17 22:08–22:31 | Scripted fixes | Seed controls, gap steps, tab rendering, routing-card wording | transcript (4 scripted edits) |
| 2 | Nov 18 ≈01:41–01:43 | **Lab v2** (`patch_lab.py`, `patch_app.py`) | "Advisor demo" → "**Demo**" (user request); new **🧠 How the models work** tab (6 tabs); new `evidence_blocks` in the Scenario tab (validity, raw data, failure analysis) | transcript; screenshots `shot4_*` |
| 3 | Nov 18 01:50 | Layout fix | Failure-analysis tables changed from two columns to stacked (they were truncated, `shot4_c_why.png`) | current code L759-766 |

### 7.2 What changed between v1 and the current version [V by text comparison]

| Area | v1 (Nov 17) | Current |
|---|---|---|
| Tabs | 5 | 6 (+ 🧠 How the models work) |
| Mode name | Advisor demo | Demo |
| Scenario tab sections | header, actions, board, cards, truth, advanced, history | + validity expander, 🔒 caption, raw-data expander (chart + table + CSV), failure-analysis expander (reasons + 2 tables) |
| Functions | 29 | 32 (`evidence_blocks`, `failure_analysis`, `models_tab`) |
| `lab.py` | scenario, inference, gallery, metrics | + `validity`, `raw_window`, `raw_source`, `model_inputs`, `physics`, `training_history`, `difficulty`, `FAMILY_LABELS`, `VIS_TEXT`, `CONF_TEXT` |
| Playback | Plotly animation; screenshot v1 shows axis tick numbers and slider labels colliding with the legend | Axis ticks hidden; legend moved below the slider. Play/Pause buttons still low-contrast in `shot4_a_home.png` |

Nothing was removed. All changes are additions or renames.

### 7.3 Why an earlier version may feel clearer (evidence-based) [HYP grounded in code]

The user's preferred earlier version is not identified in the repository. Both candidates, version 0 and version 1, share the
properties below; the current version does not.

1. **One job per tab.** Version 0 split explore / replay / recover / explain / results into separate numbered tabs. The current
   Scenario tab stacks up to 11 blocks vertically: header, actions, view and playback radios, board, three cards, truth box,
   validity, lock caption, raw data, failure analysis, advanced, and history.
2. **Demo and research overlap.** Demo mode still shows Advanced information (checkpoint JSON, input summary, cross-check), the
   validity table and two post-reveal expanders. The "Demo" is therefore not much lighter than the lab.
3. **Mode is implicit.** Five of six tabs ignore the mode. The gallery silently flips the mode to Research lab.
4. **Explanations are scattered.** Explanations appear in four places (Demo info boxes, Legend expander, Guide tab, How the
   models work tab) with overlapping content.
5. **Version 1 was smaller.** It had five tabs and no expanders inside the Scenario flow; the additions of version 2 roughly
   doubled the post-reveal content.

### 7.4 Dashboard issues found

| # | Issue | Severity | Evidence |
|---|---|---|---|
| D1 | **Pre-reveal hint (Research lab):** the encounter dropdown shows `crossing_class` (whether the two *identity* paths crossed) and `movement` (`fam_heading_relation`, computed from the clean tracks on both sides). With the visible geometry, "crossing" lets a human infer which track continues which player. `lab.difficulty` itself says such descriptors are "reveal only" | High for fair human tests; no effect on models | `lab.py:118-133` vs `lab.py:441-443`; `lab_app.py:349`; test `test_encounter_table_is_public` checks only y / p / outcome columns |
| D2 | **Gallery category names reveal the outcome**; loading also switches the mode silently | Medium | `lab_app.py:516-540` (documented in tab text) |
| D3 | **HGB hyperparameters misstated** ("400 trees, lr 0.05" for all folds) | Medium (scientific accuracy) | `lab_app.py:810`; `dashboard_explaination1.md:399` |
| D4 | **"Training samples: 25,331 two-player encounters"** conflates samples with independent encounters (18,777) and with per-fold training sets (≈ 19.4–20.3 k fitting) | Medium | `lab_app.py:823-824` |
| D5 | **"40 FIFA World Cup 2022 matches"** omits the other 24 (official val/test, consumed by the earlier RF track) | Low | `lab_app.py:819` |
| D6 | **Validity rows 7–8** ("label accepted", "label re-verified") are both `ok=pub.valid`: true, but not two independent checks | Low (overstated rigour) | `lab.py:374-376` |
| D7 | **Demo does not force random selection**: the default 3812 encounter is shown and "Next ▶" is enabled | Medium (demo fairness / repetition) | `lab_app.py:74-77, 455` |
| D8 | **Reveal is available from step 4**, and without running a prediction | Low | `lab_app.py:483-484, 318-326` |
| D9 | **No "hide truth again"** for a scenario (only regeneration) | Low | `action_row` |
| D10 | **Saved gallery categories `hard_parallel_baseline` and `highest_confidence_accepted_errors` are not offered** (metrics has 11, the UI 9) | Low (useful research views hidden) | `lab.py:316-326` vs `metrics_v1.json["examples"]` [R] |
| D11 | **Overall 8 % presented as "the" policy error**; it is a design-weighted average over one-factor variants (§4.9) | Medium (interpretation) | `evaluation_tab` warning |
| D12 | **Play / Pause buttons look disabled** (low contrast) | Low | screenshot `shot4_a_home.png` |
| D13 | **Close-up circle centre uses outgoing start positions** and is drawn from the first frame (it shows where tracks will reappear, not who is who) | Low | `lab_board.py:54-57,171-172` |
| D14 | **The GRU training curve marks `argmin(val loss)`**, while the kept state is the last improvement > 1e-4. These can differ by a few epochs | Low | `lab_app.py:852` vs `models.py:241-248` |
| D15 | **Duplicate explanations** (Legend expander, Guide tab, demo guide, `dashboard_explaination1.md`) | Low (maintenance) | — |
| D16 | **Last `lab_app.py` edit (01:50)** came after the Nov 18 full pytest run started; see §9.6 for today's run | Info | handoff §8 |

**Not found:**
- no case where truth reaches inference (`run_inference` takes `Public` only; test `test_truth_never_reaches_inference`);
- no protected-data path (`fold_for_match` and `MatchData` refuse held-out matches; the JSON tab refuses held-out ids).

### 7.5 Proposed information architecture [PROP; nothing implemented]

| Area | Purpose | Contents |
|---|---|---|
| **A. Advisor-friendly Demo** | 3–5 minute story, minimal controls | Forced random pick (no default); 5 steps (Pick → Watch → Hidden → Predict → Reveal); one compact result card (routed decision + both p); reveal; "same encounter, longer gap" button; one plain-language caveat. No hashes, JSON or tables |
| **B. Research & Diagnostics** | Controlled testing | Match / encounter selection (descriptors hidden until reveal); one-factor or custom variants; seeds; full board with step controls; both model cards + routing; truth; raw provider rows + CSV; failure analysis; gallery (all 11 categories, with neutral names before reveal); JSON tracklet upload; session history |
| **C. Model Internals** | How the models work | Input construction; GRU diagram + live tensors; HGB 36 features + live values; **per-fold hyperparameter table**; training curves; routing rules and threshold provenance |
| **D. Evaluation & Reproducibility** | Evidence and integrity | M2 metrics, charts, family table, metrics download; validity checks; checkpoint hashes; saved-vs-live cross-check; code-fingerprint status; evidence-type statement; held-out status |

**Mapping of every existing feature (nothing disappears):**

| Existing feature | Proposed location |
|---|---|
| Mode radio | Replaced by top-level navigation A / B / C / D |
| Demo 7 steps, progress, Back/Next, info texts | A (condensed to 5 steps; text kept) |
| 🎲 Pick random / Another random / Gap for this encounter / ↻ Same encounter | A (and B) |
| Match, encounter, seed, seeded / truly random | B (encounter descriptors only after reveal: fixes D1) |
| Variant / factor / level / custom sliders / seed | B |
| Run prediction, Reveal, call counter | A (simplified) and B |
| View zoom, Animation / Step by step, step buttons, frame slider | B (A: animation only) |
| Model cards, Operational decision, measured-limit caption | A (compact) and B (full) |
| Ground truth box | A and B |
| ✅ Validity expander | D (summary badge in A and B) |
| 📄 Raw provider data + CSV | B |
| 🔍 Why could the models fail here? | B |
| Advanced information (fold, params, hashes, input summary, cross-check, JSON downloads) | D (input / truth JSON downloads also in B) |
| Scenarios generated this session | B |
| Example gallery | B (neutral names; add the 2 hidden categories) |
| Evaluation (Milestone 2) | D |
| How the models work (static text, live tensors, features, training curve) | C |
| Load JSON tracklets | B |
| Guide & legend | Legend → A (one expander); script → `docs/m11_demo_guide.md` only; glossary → C |

---

## 8. Known bugs and scientific inconsistencies

1. HGB hyperparameter text (D3).
2. Pre-reveal crossing / movement descriptors in the Research-lab dropdown (D1).
3. Sample-versus-encounter wording (D4) and the "40 matches" wording (D5).
4. Threshold fitted in-sample; the M2 baseline 5.0 % is not an independent confirmation (§4.9).
5. The overall 8 % is a design-weighted mixture (D11).
6. Validity-panel rows duplicate one boolean (D6).
7. The M2 report §8 lists only the `scenario_eval.py` docstring edit; the `scenarios.py` refactor (`_build`, `build_custom`,
   `example_sets`) is not mentioned [V Nov 18; unchanged].
8. Torch, pandas and pyarrow are used but not declared in `pyproject.toml` [V Nov 18; file unchanged].
9. **Wording "held-out never read":** the held-out matches were processed in pre-split descriptive stages (§3.8). The correct
   wording is "never used for model development".
10. **`motion.FEATURE_NAMES` is empty until the first feature computation.** This is harmless in the current code, but a trap for
    future scripts.

---

## 9. Reproducibility and correctness audit (Part 5)

### 9.1 Checkpoints [R]

- All 16 checkpoint files (`m11-stage4c-robustness-v1/neural/C/seed0/fold<f>_c160.pt`, `hgb/C/fold<f>.joblib`) match the
  SHA-256 values in `m11-infer-v1/bundles_v1.json`.
- `load_fold_model` refuses a mismatch (`bundle.py:175-179`). The dashboard validity row therefore restates a check that already
  happened at load.

### 9.2 Folds and isolation [V]

- Fold map from `folds.outer_folds` (salt `m11-stage3-folds-v1`). Matches per fold as in the Nov 18 handoff §5.4.
- Early-stop matches per fold [R]:

  | Fold | Early-stop matches |
  |---|---|
  | 0 | 3849, 3840, 10512 |
  | 1 | 3855, 3836, 3852 |
  | 2 | 3826, 3835, 3812 |
  | 3 | 3849, 3836, 10506 |
  | 4 | 10506, 3849, 3818 |
  | 5 | 3851, 3828, 3855 |
  | 6 | 3815, 3851, 3831 |
  | 7 | 3823, 3838, 3812 |

  None is in its own fold's test set.

### 9.3 Raw-data provenance [V in source; DOC]

- PFF FIFA World Cup 2022 tracking `.jsonl.bz2` under `data/raw` (immutable by rule). Step 1 caches were re-verified against the
  raw files with 0 mismatches [DOC].
- The dashboard's raw-row table reads the cached provider arrays (`md.c`). The path is shown as `data/raw/…/<file>`.
- **[NV]:** licence text and file checksums. A search found PFF's public release announcement but no licence text (§11.11).

### 9.4 Label verification [V]

- `scenarios._build` re-runs `candidates.label_quality` for gap / shift variants. Every built variant must pass
  `samples.verify_label` against raw slots (`scenarios.py:291-345`); invalid variants never reach a model.
- Stage 3: 100,832 / 100,832 verified [DOC].

### 9.5 Saved-result preservation and code fingerprints [R]

| `m11-scenarios-v2/run_config.json` `code_sha256` entry | Current file |
|---|---|
| `adapter.py` | **match** |
| `bundle.py` | **match** |
| `predict.py` | **match** |
| `scenario_eval.py` | **differs** (edited after the run) |
| `scenarios.py` | **differs** (refactor after the run) |

- **Integrity:** `metrics_v1.json` and the 32 per-match files were verified unchanged on Nov 18 [V Nov 18]. Today the dashboard
  reads the same metrics (55,680 scenarios / 55,287 valid / error 0.0797) [R].
- **Can results be reproduced with current code?** Partly evidenced:
  - the M3 tests compare live inference with saved M2 probabilities for sampled scenarios (`test_live_inference_matches_saved_milestone2`);
  - regeneration was checked equal to the saved arrays for 60/60 [CLAIM].
  - A full re-run would need a **new output version** (`m11-scenarios-v3`), because the resume guard refuses mismatched code.
  - **Procedure (not run):** run the suite into a new directory with the same suite name and seeds; compare per-scenario
    probabilities and outcomes with v2 (tolerance as in `bundle.TOL`); record the diff. This is CPU-only, ≈ 10 min [DOC: M2 run
    543 s].
- **Live or cached in the dashboard:**

  | Content | Source |
  |---|---|
  | Predictions | **Live** on every button press |
  | Evaluation tab, gallery lists, family statistics, measured-limit captions, saved cross-check | **Saved** |
  | Training curves | Saved fold records |

### 9.6 Tests

**Run on 2026-11-19** [R]: `pytest -p no:cacheprovider -q tests/test_m11_infer.py tests/test_m11_scenarios.py tests/test_m11_lab.py`.
Result: **83 passed in 128 s** (2026-11-19 21:17–21:19 +02:00): 38 M1 + 23 M2 + 22 M3. This run covers the 01:50 `lab_app.py`
edit, which closes item D16. The full suite was **not** run today.

**Earlier runs:**
- full suite 1112 passed / 6 skipped on 2026-11-18 ≈ 01:50 [CLAIM];
- M3 22 tests passed on 2026-11-18 [CLAIM];
- M1 + M2 61 passed on 2026-11-17 [DOC].

**Coverage gaps:**
- no test that encounter descriptors are free of label hints (D1);
- no test of the hyperparameter text;
- no test of Demo step gating (D7 / D8);
- no visual regression tests.

### 9.7 Environment [V Nov 18]

- Python 3.12 venv; torch 2.9.1+cu126, scikit-learn 1.9.1, streamlit 1.67.0, plotly 6.9.0.
- `pyproject` extra `m11dash` = streamlit, plotly, scikit-learn; torch, pandas and pyarrow are undeclared.
- **Version sensitivity:** sklearn joblib checkpoints are pickles tied to the sklearn version. A different sklearn may fail to load
  them or change predictions; the hash check covers the file, not the runtime.

---

## 10. Reproducibility risks (ranked)

1. **Uncommitted M1–M3 code.** A disk loss or a stray `git clean` would remove the inference, scenario and lab code that
   produced the saved results.
2. **Stale code fingerprints for M2.** A reviewer cannot confirm by hash that the current code produced v2.
3. **Pickle / sklearn and torch version coupling.** Versions are not pinned in `pyproject`.
4. **Data under `data/processed` is git-ignored and exists only on this machine.** There is no backup or manifest outside it.
5. **Single-seed bundle.** Seed variance appears in reports, but not in the deployed models or the dashboard.

---

## 11. Paper feasibility assessment (Part 6)

**Literature status: no literature review was performed in this audit.** Four web searches were run; the few sources they returned
are listed in §11.13. All novelty statements below are **[HYP] pending a real review.**

### 11.1 Candidate contribution, honestly scoped

| Element | Likely status | Evidence needed |
|---|---|---|
| Two-player re-association from trajectories alone, using real elite-football tracking with simulated interruptions | Plausibly useful as a **controlled benchmark**; tracking-by-motion and MOT association are established fields | Review of MOT association (Kalman / Hungarian, motion-only trackers), sports MOT benchmarks (e.g. SoccerNet tracking, SportsMOT), trajectory prediction (e.g. Social-LSTM-style models), football tracking-data literature |
| GRU vs feature-engineered HGB | Comparison itself is not novel; the **robustness profile** (GRU wins in-distribution but breaks off-distribution, HGB transfers) is a useful finding | Stronger baselines and convergence fixes |
| Match-grouped OOF evaluation | Standard good practice, not a contribution | — |
| Gap / noise / dropout robustness analysis | A useful contribution if systematic and statistically sound | CIs, multiple seeds, a multiple-comparison policy |
| **Provider-texture finding** (models exploit interpolation texture; VISIBLE-in-gap much harder) | **Potentially the most interesting and most citable result:** a caution for anyone training on provider tracking data | Formal ablation (VISIBLE-only training/evaluation, texture-removal transforms) |
| Selective prediction with abstention | Methods are established (Geifman & El-Yaniv 2017; SelectiveNet 2019); the contribution would be **applied**: the risk-coverage behaviour under distribution shift | Threshold calibration on separate data, risk bounds |
| Explainable failure analysis, reproducibility tooling | Supporting material, not a core contribution | — |

### 11.2 Defensible research questions [PROP]

- **RQ1.** How accurately can two players' identities be re-associated across a short tracking interruption from motion alone, and
  how does accuracy depend on interruption length, positional noise, observation dropout and context length?
- **RQ2.** Do learned sequence models exploit provider-specific interpolation texture, and how much of their advantage over
  feature-based models remains on directly observed (VISIBLE) data?
- **RQ3.** Can confidence-based abstention keep the error of committed re-associations at a target rate under such shifts, and
  which shift indicators must a selective policy observe?

### 11.3 Candidate titles [PROP]

1. "Who Is Who After the Crossing? Trajectory-Only Re-Association of Football Players Across Simulated Tracking Gaps"
2. "Motion Is Not Enough Under Shift: Robustness and Selective Prediction for Two-Player Identity Recovery in Football Tracking Data"
3. "Provider Interpolation Texture Inflates Learned Identity Association in Football Tracking Data"

### 11.4 Hypotheses [PROP]

- **H1.** A sequence model beats engineered-feature boosting at the trained gap; the gap closes or reverses at other gaps and under
  noise.
- **H2.** Performance on VISIBLE-in-gap encounters is substantially lower than on ESTIMATED ones for learned models, more than for
  physics baselines.
- **H3.** A shift-aware selective policy, one that observes gap, density and context, keeps realised risk within a pre-specified
  bound on held-out matches, and a confidence-only policy does not.

### 11.5 Required baselines

| Kind | Baselines |
|---|---|
| Classical | Nearest endpoint; constant-velocity forward / bidirectional; constant acceleration (already in `motion.BASELINES`); Kalman-filter prediction with a Mahalanobis cost + Hungarian assignment; the best physics combination |
| ML | Logistic regression on the perm features; RF; HGB |
| Deep | MLP, GRU-far-end, LSTM-gap-last, GRU-gap-last; optionally a small Transformer/TCN encoder |
| Trivial | Coin; "always MATCHED"; texture-only probe |
| Upper reference | Oracle with the hidden path (sanity only) |

### 11.6 Necessary ablations

- Gap-last vs far-end reading.
- Gap input present vs removed.
- Each augmentation (reflection, order exchange, shift config C).
- Epoch budget with nested selection.
- Encounter-frame vs pitch-frame.
- Context length.
- HGB feature groups (distance costs / log costs / kinematic / context).
- Training on VISIBLE-only or texture-neutralised inputs (smoothing, jitter matched to VISIBLE).
- Variable-gap training.

### 11.7 Robustness experiments

- Development: the existing M2 factors, plus interactions (gap × noise, dropout × context).
- Encounter distance bands: ≤ 0.5 m and the never-tested 0.5–2 m.
- Same-team vs opposite-team.
- Asynchronous hiding.
- Realistic noise models: correlated, not i.i.d.

### 11.8 Statistics

- **Unit of analysis:** the match for CIs (match-grouped bootstrap, as now) and the independent encounter for point estimates.
- **Paired comparisons:** paired bootstrap of BA / log-loss differences over matches.
- **Multiple seeds:** ≥ 3 seeds for every neural result.
- **Pre-registered primary endpoints** and a correction policy for secondary ones.
- **Report** BA, log loss, Brier, ECE, AUC.
- **No conflation:** never count the 55,680 variants as independent.

### 11.9 Calibration and selective-risk evaluation

- Risk-coverage curves and AURC.
- Fit thresholds on **calibration matches separate from the evaluation matches** (nested within development; then frozen for
  held-out).
- Report realised risk with exact binomial CIs at each operating point.
- Consider conformal / risk-controlling thresholds.
- Evaluate per-condition risk, not only the pooled figure.

### 11.10 Independence and leakage checks

- Match-level isolation (done).
- An overlap-group independent subset (done).
- **Texture leakage check** (RQ2).
- **Label noise:** quantify provider identity errors via the d_min < 0.05 m and duplicate-track cases.
- **Selection leakage:** document every development-based choice.
- Report the held-out pre-exposure (§3.8).

### 11.11 Limitations, external validity, ethics, licensing

- **Limitations:**
  - simulated interruptions;
  - one provider and one tournament;
  - elite men's football only;
  - ≤ 0.5 m encounters;
  - two players only;
  - no video;
  - the provider identity is used as truth.
- **Ethics:**
  - The data concern identifiable professional players. Results are about tracking, not individuals; avoid player-level
    performance claims.
  - Nothing is about fatigue (PROJECT_RULES).
- **Licensing:**
  - PFF announced a free public release of the 2022 World Cup dataset (blog post, §11.13).
  - **The licence and redistribution terms were not found and are [NV]**: they must be read before publication.
  - Data cannot be redistributed without them; release code plus derived identifiers only.
  - Cite the dataset as PFF FC requests.

### 11.12 Possible outline [PROP]

1. Introduction (identity switches in sports tracking; why motion-only).
2. Related work (MOT association, sports tracking, trajectory prediction, selective classification).
3. Data and task (PFF WC 2022; encounter definition; simulated interruption; labels; splits; pre-exposure statement).
4. Methods (baselines, HGB features, GRU pair encoder, training, selective policy).
5. Experimental protocol (match-grouped CV, nested selection, preregistered held-out).
6. Results (in-distribution, robustness, texture analysis, selective risk).
7. Failure analysis.
8. Limitations and external validity.
9. Reproducibility statement.

### 11.13 Sources consulted (search results only; not a literature review)

- Geifman & El-Yaniv, *Selective Classification for Deep Neural Networks*, NeurIPS 2017: <https://arxiv.org/abs/1705.08500>
- Geifman & El-Yaniv, *SelectiveNet*, ICML 2019: <https://proceedings.mlr.press/v97/geifman19a.html>
- PFF FC, 2022 World Cup dataset release: <https://www.blog.fc.pff.com/blog/pff-fc-release-2022-world-cup-data>
- SportMamba (team-sports MOT, 2025), arXiv 2506.03335 (title seen only): <https://arxiv.org/pdf/2506.03335>

### 11.14 Is it publishable now?

**No.**

**What supports it today:**
- a clean development pipeline;
- match-grouped evaluation;
- verified labels;
- a large stress suite;
- honest failure documentation.

**Blocking gaps:**
1. no literature review;
2. in-sample thresholds;
3. selection-optimistic, under-converged GRU;
4. no Kalman / assignment baseline under the same protocol;
5. no preregistered held-out result;
6. texture confound not resolved;
7. single provider.

**Claims that must NOT be made:**
- "validated" or "generalises to video";
- "real occlusion";
- "5 % error guaranteed";
- "state of the art";
- anything about fatigue;
- "held-out never accessed" (use "never used for model development");
- counting variants as independent samples.

### 11.15 Preregistered final evaluation [PROP]

Extend `m11_stage5_final_protocol_PROPOSED_v1.md` with:

1. Primary endpoints (BA and log loss at 0.8 s on the independent subset; VISIBLE-only co-primary; risk at the 5 % operating point
   with an exact CI).
2. Thresholds fitted on development calibration folds, then frozen.
3. Final models trained with nested-selected budgets and hashes recorded.
4. Held-out builders in a logged `--final-heldout` mode.
5. One evaluation; all candidates reported.
6. A pre-written analysis script, hashed before the run.
7. A deviation log.

---

## 12. Prioritised roadmap (Part 7)

Legend: **Ess** = essential for a publishable paper; **Opt** = optional polish. "Retrain" = requires new model training.
"Changes results" = alters existing saved results (all proposals write new versions instead).

### Phase A: documentation and dashboard correctness

| Task | Why | Deliverable | Depends on | Risk | Retrain | Changes results | Approval | Class |
|---|---|---|---|---|---|---|---|---|
| A1 Commit M1–M3 (user) | Protect untracked code | Commit(s) | — | none | no | no | **user action** | Ess |
| A2 Fix HGB text (D3), counts (D4, D5) | Accuracy | Edited `lab_app.py`, `dashboard_explaination1.md` | — | low | no | no | yes | Ess |
| A3 Hide pre-reveal descriptors (D1); neutral gallery names (D2) + test | Fair human testing | Code + test | — | low | no | no | yes | Ess |
| A4 Declare torch / pandas / pyarrow; pin versions | Reproducibility | `pyproject.toml` | — | low (no install) | no | no | yes | Ess |
| A5 Erratum note for the M2 report §8, the "held-out never read" wording and the 8 % interpretation | Honesty | Addendum doc | — | none | no | no | yes | Ess |
| A6 Dashboard restructure into A / B / C / D (§7.5) | Usability | Refactored app | A2–A3 | medium (UI regressions) | no | no | yes | Opt |
| A7 Demo fixes D7 / D8 / D12 / D14, gallery extra categories | Polish | Code | A6 | low | no | no | yes | Opt |

### Phase B: reproducibility and baseline verification

| Task | Why | Deliverable | Depends on | Risk | Retrain | Changes results | Approval | Class |
|---|---|---|---|---|---|---|---|---|
| B1 Re-run the M2 suite into `m11-scenarios-v3` with current code; diff vs v2 | Close the fingerprint gap | v3 + diff report | A1 | low (CPU ≈ 10 min) | no | no (new version) | yes | Ess |
| B2 Full test suite run | Fresh evidence | Log | — | none | no | no | no | Ess |
| B3 Environment lock (`pip freeze`) and data manifest of `data/processed` M11 dirs | Reproducibility | Lock + manifest | A4 | none | no | no | yes | Ess |
| B4 Kalman + Hungarian baseline under the same folds | Required baseline | Results + report | — | low | fits only | no | yes | Ess |

### Phase C: controlled development experiments (32 matches only)

| Task | Why | Deliverable | Depends on | Risk | Retrain | Changes results | Approval | Class |
|---|---|---|---|---|---|---|---|---|
| C1 Nested selection of the GRU epoch budget; 3 seeds | Convergence, selection bias | New Stage-4D outputs | B | GPU time | **yes** | no (new dir) | yes | Ess |
| C2 Texture ablations (VISIBLE-only evaluation / training; texture-neutralising transforms) | RQ2 | Report | C1 | medium | **yes** | no | yes | Ess |
| C3 Variable-gap / noise / dropout training | RQ1, robustness | Models + report | C1 | medium | **yes** | no | yes | Ess |
| C4 Selective policy v2: thresholds fitted on calibration folds, shift-aware routing (density, context), per-condition risk | RQ3 | Policy + report | C1–C3 | medium | no | no | yes | Ess |
| C5 Ablations (§11.6) | Paper completeness | Report | C1 | GPU time | **yes** | no | yes | Ess |
| C6 Wider-distance encounters (0.5–2 m) | External validity | Dataset version + results | Stage 2 code | medium | yes | no | yes | Opt |

### Phase D: paper-quality statistics

| Task | Why | Deliverable | Depends on | Retrain | Approval | Class |
|---|---|---|---|---|---|---|
| D1 Analysis plan + scripts (paired match bootstrap, seeds, multiplicity) | Rigour | Hashed scripts | C | no | yes | Ess |
| D2 Literature review with citations | Novelty claims | Related-work draft | — | no | no | Ess |

### Phase E: frozen final evaluation (only after explicit approval)

| Task | Why | Deliverable | Depends on | Retrain | Changes results | Approval | Class |
|---|---|---|---|---|---|---|---|
| E1 Freeze protocol v2 (§11.15) | Preregistration | Signed protocol + hashes | C, D1 | no | no | **yes** | Ess |
| E2 Final training on 32 matches | Final models | Models + hashes | E1 | **yes** | no | **yes** | Ess |
| E3 `--final-heldout` build mode + one evaluation | The only unbiased result | Held-out results | E2 | no | no | **yes (irreversible)** | Ess |

### Phase F: manuscript

| Task | Deliverable | Depends on | Class |
|---|---|---|---|
| F1 Draft (outline §11.12), figures, reproducibility appendix, dataset citation / licence check | Manuscript | E3, D2 | Ess |
| F2 Public code release plan (no data redistribution) | Release checklist | F1, licence | Ess |

---

## 13. Open decisions requiring user approval

1. Which earlier dashboard version you preferred: version 0 (the Step 8 numbered-tab RF dashboard) or version 1 (the Nov 17 lab
   with 5 tabs). This determines A6.
2. Commit M1–M3 now (you perform it).
3. Approve Phase A fixes A2–A5 (no retraining, no result changes).
4. Approve a dependency declaration (no installation).
5. Approve B1 (new `m11-scenarios-v3` directory).
6. Approve GPU retraining for Phase C, with its budget.
7. Choose the paper's primary research question (RQ1, RQ2 or RQ3) to focus Phase C.
8. Stage 5 / final evaluation: protocol v2, final training and the held-out build mode. Later, and only once.

---

## 14. Exact commands to launch the existing dashboard

```powershell
# from F:\Football-Fatigue-Project (options BEFORE the subcommand)
.\.venv\Scripts\python.exe -m football_fatigue.m11_dashboard --port 8610 lab
# or double-click scripts\m11_lab.cmd, then open http://127.0.0.1:8610
# older apps: --port 8503 encounters  (Football Trajectory Explorer);  no subcommand = Step 8 RF dashboard (app.py)
```

Port 8610 is used because Windows reserves 8494–8593 on this machine [CLAIM]. The first match load takes about 10 s.

---

## 15. Final Git status and changes caused by this audit

- **Files created in the repository:** this file only, `docs/handoffs/SESSION_A_RESEARCH_AND_DASHBOARD_AUDIT_2026-11-19.md`.
  It lands inside the already-untracked `docs/handoffs/`, so `git status --short` is unchanged.
- **Files modified:** none. **Git operations:** read-only (`status`, `log`, `diff`, `rev-parse`).
- **Outside the repository:** read-only helper scripts and a recovered copy of the original `lab_app.py` text in the session
  scratchpad.
- **Computations run:** hash checks, saved-metric decomposition, per-fold counts, and the focused pytest run (cache disabled).
  No training, no GPU job, no dashboard server, no protected data.

```text
## feature/m11-identity-finalization   (identical to §2.1)
```
