# Session A: verified current-state master handoff (2026-11-18)

**Repository:** `F:\Football-Fatigue-Project` · **Branch:** `feature/m11-identity-finalization` · **HEAD:** `1bd016d8099e32d94807aaca805028e8621007f8` (2026-11-17)

**Prepared:** 2026-11-18, read-only inspection plus this single new file. No source, test, configuration, checkpoint, dataset or
earlier report was modified for this handoff. No model was trained, no test suite was run for it, and no protected data was opened.

**Missing document:** `Football_Fatigue_Project_Master_Handoff.md` does **not exist** in the repository (searched 2026-11-18).
The nearest equivalents are `docs/PROJECT_MASTER_CONTEXT.md` (last updated 2026-10-28, after M7.3, before M11) and
`docs/research/m11_step1_handoff.md`.

## Evidence labels

| Label | Meaning |
|---|---|
| **[VERIFIED]** | Checked against current source or files during this handoff (2026-11-18), with read-only commands |
| **[DOC]** | Reported by an existing project document; the underlying computation was not repeated |
| **[CLAIM]** | Historical Claude statement in an earlier session or memory, not re-checked now |
| **[NOT VERIFIED]** | Insufficient evidence |
| **[PLANNED]** | Proposed or intended; not implemented |

---

## 1. Executive overview and current status

**Project.** *Football Fatigue & Performance Prediction*, a Biomedical Engineering graduation project. It aims at a modular
football-analytics platform for movement, workload, fatigue-related indicators and possibly injury risk [DOC: README.md].

**The repository owner's responsibility is Task 2: Software Architecture & Tactical Board Streaming** [DOC: README.md,
PROJECT_MASTER_CONTEXT §2]. Fatigue, physiology, wearables and injury risk are **not started**: no data and no models.
Tracking-derived movement metrics are external-load descriptors and **never fatigue measurements** (PROJECT_RULES.md).

**Three workstreams exist inside Task 2:**

| Track | Data | Status |
|---|---|---|
| A. FIFA tracking platform | PFF FIFA World Cup 2022: tracking, events, rosters, metadata; 64 matches; **no video** | Milestones 0–2C and the tactical board implemented; formal reviews pending [DOC] |
| B. Computer-vision pipeline | SoccerTrack v2 panoramic video | M2–M7.3 done; identity and physical metrics blocked by evidence [DOC: PROJECT_MASTER_CONTEXT] |
| **M11. Two-player identity recovery** | FIFA PFF tracking only (trajectories, no video) | **Active.** Stages 1–4C done; Milestones 1–3 done (uncommitted); Stage 5 **not run** |

**M11 status (2026-11-18):**
- Stages 1, 2, 3, 4A, 4B and 4C are complete. The Stage 4C models are GRU C (primary) and HGB C (reference/fallback), as **8 outer-fold models each** [VERIFIED: manifest and files].
- Milestone 1 (inference package `m11_infer`), Milestone 2 (scenario stress suite `m11-scenarios-v2`) and Milestone 3 (Streamlit *M11 Trajectory Recovery Lab*) are implemented and **uncommitted**. Milestone 3 awaits user review.
- **Not done:**
  - Stage 5 (final held-out evaluation);
  - a model trained on all 32 development matches;
  - any held-out shard build [VERIFIED: no `*stage5*` or `*final*` directories in `data/processed`; the Stage 5 protocol is marked PROPOSED].
- **All reported M11 performance is development, unseen-match out-of-fold performance.** It is **not** independent final validation.

---

## 2. Architecture and data flow

### 2.1 Track A: FIFA platform (original system)

[DOC: FIFA_ORIGINAL_SYSTEM_COMPLETE_TECHNICAL_WALKTHROUGH.md §1.3; paths VERIFIED to exist]

```text
data/raw (immutable PFF files)
  └─ inventory.scan (names/sizes only)              src/football_fatigue/inventory.py
      └─ Dataset → metadata, roster, events(explicit EventVersion), streaming tracking frames
                                                    src/football_fatigue/ingestion/{loaders,models,validation}.py
          ├─ synchronize_match (same period, nearest video time, ≤ 1 frame)   src/football_fatigue/synchronization.py
          └─ MatchData (RAW / SMOOTHED / SMOOTHED_THEN_RAW policy, coordinate states)   src/football_fatigue/match_data.py
              ├─ analytics.movement / analytics.spatial                       src/football_fatigue/analytics/
              └─ playback adapter → stdlib HTTP server → board/index.html (Canvas)
                                                    src/football_fatigue/playback/, src/football_fatigue/board/
```

### 2.2 M11 data flow (two-player trajectory association)

```text
data/raw PFF tracking (29.97 fps records; provider updates ≈ 15 Hz)
  └─ Step 1 caches               m11-encounters-v2                  (football_fatigue.encounters, step1_verify)
      └─ Stage 1 states          m11-window-local-methods-pilot-v4  window_intersections/local.py, candidates.py
          └─ Stage 2 candidates  m11-stage2-candidates-v1           + frozen split m11-stage2-split-v1 (split.py)
              └─ Stage 3 samples m11-stage3-samples-v1              window_intersections/samples.py (refuses held-out)
                  ├─ Stage 4A/4B benchmarks  m11-stage4-benchmarks-v1      m11_benchmark/{run,models,robust,evaluate}.py
                  └─ Stage 4C                m11-stage4c-robustness-v1     m11_benchmark/{shiftaug,stage4c,stage4c_eval}.py
                      └─ Milestone 1: m11_infer (bundle, adapter, routing m11-routing-v1)   manifest m11-infer-v1/bundles_v1.json
                          ├─ Milestone 2: scenario suite → m11-scenarios-v2                 m11_benchmark/{scenarios,scenario_eval}.py
                          └─ Milestone 3: Trajectory Recovery Lab (Streamlit)              m11_dashboard/{lab,lab_board,lab_app}.py
```

All `data/processed/...` paths are git-ignored and exist only on this machine.

---

## 3. Timeline and major decisions

Dates are 2026. Sources: git log [VERIFIED], `docs/research/*` reports [DOC], memory [CLAIM].

| Date | Event |
|---|---|
| Oct 15 | Raw FIFA data downloaded; readability check passed (data_policy.md) [DOC] |
| Oct ≈ 16–20 | Track A: M0, M1A inventory, M1B schema audit, M2A ingestion, M2B synchronization, M2C MatchData, tactical board [DOC] |
| Oct 18–28 | Track B: CV M2–M7.3 (SoccerTrack); last update of PROJECT_MASTER_CONTEXT is Oct 28 [DOC] |
| Nov 6 | Commits `2e18001`…`1458e1b` (vision, analytics, demo, multicam, M11 encounter catalogue, identity matching, locked test, docs) [VERIFIED: git log] |
| Nov 6 | **Earlier M11 track (Steps 1–8):** a random-forest matcher was frozen (`m11-frozen-v1`), validated once on 12 official validation matches, then evaluated **once on the locked official test** (12 matches; AUC 0.838). The **official FIFA validation and test splits are consumed** for that model [DOC: m11_step6/7 reports; CLAIM: memory] |
| Nov 8–13 | Advisor-directed intersection, occlusion-catalogue, exact-intersection, sync-proximity and window-intersection studies. Several concepts were superseded (see the reports in `docs/research/`) [DOC] |
| Nov 13 | **User decision:** VISIBLE and ESTIMATED provider samples are equally eligible; status is never a model input. Stage 1 done on 3812 and 3816 [CLAIM: memory; DOC: m11_stage1_candidates_and_states.md] |
| Nov 13 | **Stage 2:** internal held-out split frozen (`split.json` SHA `66b2321a…6c0`); 31,219 candidates accepted at 0.8 s (25,331 train/CV) [DOC: m11_stage2_results_v1.md; split hash VERIFIED] |
| Nov 14 | **Stage 3:** supervised samples `m11-stage3-samples-v1`; labels verified against raw slots, 100,832 / 100,832 [DOC] |
| Nov 14 | **Stage 4A/4B:** baselines, ML and neural benchmarks; best `gru_gaplast` BA 0.902 vs HGB 0.870 [DOC] |
| Nov 14 | **Stage 4C:** shift augmentation and a 160-epoch budget; GRU C centred 0.926 / shifted 0.910; HGB C 0.861 / 0.878; Stage 5 protocol **PROPOSED** [DOC] |
| Nov 17 | Commit `1bd016d` (M11 research, benchmarks, dashboards). Session A starts on branch `feature/m11-identity-finalization`; scope narrowed to the 2-in / 2-out trajectory model [VERIFIED: git; CLAIM: scope] |
| Nov 17 | **Milestone 1** (`m11_infer`) and Stage 4C correction note; **Milestone 2** (`m11-scenarios-v2`) [DOC] |
| Nov 18 | **Milestone 3:** Trajectory Recovery Lab; renamed "Advisor demo" → "Demo" at the user's request; dashboard explanation and demo guide written [VERIFIED: files] |

---

## 4. Repository structure and file/function map

### 4.1 Top level [VERIFIED]

| Path | Role |
|---|---|
| `PROJECT_RULES.md` | Authoritative rules (scientific integrity, Git, data) |
| `CLAUDE.md`, `AGENTS.md` | Agent roles: Claude Code implements; Codex reviews independently. Git is read-only for agents |
| `README.md`, `docs/architecture.md`, `docs/data_policy.md`, `docs/data_contracts.md`, `docs/team_handoff.md` | Track A context |
| `docs/PROJECT_MASTER_CONTEXT.md` | Whole-project context up to 2026-10-28 (Tracks A and B) |
| `docs/FIFA_ORIGINAL_SYSTEM_COMPLETE_TECHNICAL_WALKTHROUGH.md` | 5,497-line static walkthrough of Track A (inspected HEAD `de0687e`, 2026-10-20) |
| `docs/research/` | M11 reports (46 files) |
| `src/football_fatigue/` | Package |
| `tests/` | 68 entries in the directory |
| `data/` (ignored) | `raw/` immutable; `processed/` versioned outputs |
| `wadi-degla-demo-assets.zip` | Unrelated asset bundle; ignored |

### 4.2 Track A entry points [DOC, paths VERIFIED]

| Entry point | Purpose |
|---|---|
| `python -m football_fatigue.inventory [--match ID] [--format json]` | Metadata-only inventory |
| `python -m football_fatigue.ingestion quality <id> --events root` | Quality report |
| `football_fatigue.synchronization.synchronize_match(Dataset, id, EventVersion, spatial=False)` | Event-to-tracking alignment |
| `football_fatigue.match_data.MatchData.open(root, id, EventVersion, policy=CoordinatePolicy.RAW)` | Unified interface |
| `python -m football_fatigue.playback` | Tactical board, port 5780 |
| `python -m football_fatigue.demo` | Track B coach dashboard, port 5790 |

### 4.3 M11 code [VERIFIED: files exist; line counts and roles from source]

**`m11_infer/` (untracked, Milestone 1)**

| File | Contents |
|---|---|
| `bundle.py` | `fold_map`, `fold_matches`, `fold_for_match` (raises for held-out), `heldout_matches`, `train_cv_matches`, `load_fold_model`, `recorded_sha256`, `sha256_file`, `check_bundles`, `pooled_thresholds`, `stored_oof`, `predict_gru`, `predict_hgb`, `write_json_once` |
| `adapter.py` | `Track`, `ModelInput`, `gap_from_observations`, `MIN_POINTS`; timestamped tracks → Stage 3 arrays |
| `predict.py` | `POLICY="m11-routing-v1"`, `GRU_GAP_BAND=(0.75,0.85)`, `GAP_RANGE=(0.4,1.6)`, `GRU_MAX_NOISE_M=0.02`, `MAX_NOISE_M=0.05`; `route`, `decide`, `Prediction`, `Predictor(fold)`, `Predictor.for_match`, `predict_input`, `predict` |
| `fifa.py` | Development-only FIFA scenario source; adapter-from-raw check |
| `__main__.py` | CLI: `check-bundles`, `check-adapter`, `export-fifa`, `predict` |

**`m11_benchmark/` (committed, except two untracked files)**

| File | Contents |
|---|---|
| `data.py`, `folds.py`, `metrics.py`, `models.py` | Model definitions (`make_net`, `NEURAL`, `Tabular`, HGB grid) |
| `run.py`, `evaluate.py`, `robust.py`, `gates.py` | Stage 4A/4B |
| `shiftaug.py`, `stage4c.py`, `stage4c_eval.py` | Stage 4C |
| `scenarios.py`, `scenario_eval.py` | **Untracked**, Milestone 2 |

**`m11_dashboard/`**
- Committed:
  - `app.py`, `data.py`, `inference.py`, `presentation.py`, `research.py`, `pitch.py`: Step 8 RF dashboard;
  - `encounters.py`, `encounters_app.py`: encounter explorer;
  - `__main__.py`: **modified**; adds the `lab` subcommand.
- **Untracked (Milestone 3):** `lab.py` (scenario, truth separation, inference, evaluation, validity and explanation helpers), `lab_board.py` (Plotly pitch animation) and `lab_app.py` (Streamlit app).

**Other M11 packages (committed):**
- `window_intersections/` (`core`, `local`, `candidates`, `split`, `samples`, figures, CLI);
- `m11_recovery/` (earlier GBM / hybrid recovery track);
- `intersections/`, `matching/`, `encounters/`;
- `proximity_*.py`, `step1_verify.py`.

### 4.4 Dependencies [VERIFIED: pyproject.toml and the venv]

- `pyproject.toml` declares `dependencies = []`, plus the extras `dev`, `vision`, `tactical` and `m11dash` (streamlit, plotly, scikit-learn).
- Installed: torch 2.9.1+cu126, scikit-learn 1.9.1, streamlit 1.67.0, plotly 6.9.0.
- **torch, pandas and pyarrow are used by M11 code but are not declared in any extra.** This is a packaging gap; adding them needs user approval.

---

## 5. Dataset contracts, splits and preservation rules

### 5.1 FIFA PFF World Cup 2022 [DOC: data_policy.md, dataset_schema_audit.md]

- **Coverage:** 64 matches (3812–3859, 10502–10517), each with tracking (`.jsonl.bz2`), events, metadata and rosters.
- **Tracking:** 29.97 fps; pitch-centre metres.
- **Events:** three versions (root, March 14 2025, May 1 2025); which one is authoritative is **undecided**.
- **Licensing and checksums:** unverified. There is no video.
- **Rules:** `data/raw/` is immutable. Never hard-code 3812. Missing ≠ zero. Do not interpolate silently.
- **M11 raw fields used** [CLAIM, as used in the M11 code]: x, y, video time, frame number, `visibility` (1 VISIBLE, 2 ESTIMATED), `confidence` (LOW / MEDIUM / HIGH), jersey, side, player id. The provider updates positions at ≈ 15 Hz.

### 5.2 Official FIFA M11 split [VERIFIED: `data/processed/m11-encounters-v1/splits.json`]

| Split | Matches |
|---|---|
| development (40) | 3812, 3814, 3815, 3816, 3818, 3819, 3821, 3822, 3823, 3824, 3826, 3828, 3829, 3830, 3831, 3833, 3834, 3835, 3836, 3837, 3838, 3839, 3840, 3843, 3844, 3847, 3849, 3851, 3852, 3855, 3858, 3859, 10502, 10503, 10506, 10507, 10510, 10512, 10514, 10517 |
| validation (12) | 3817, 3820, 3825, 3827, 3832, 3841, 3842, 3846, 3856, 10508, 10511, 10513 |
| test (12) | 3813, 3845, 3848, 3850, 3853, 3854, 3857, 10504, 10505, 10509, 10515, 10516 |

Validation and test were used **once** in the earlier Step 6 / Step 7 RF track (the locked test is consumed) [DOC]. They are **off-limits** for current work.

### 5.3 Internal held-out split [VERIFIED: `data/processed/m11-stage2-split-v1/split.json`, SHA-256 `66b2321a58f3845194757f943c2d36b92f18695322fb5490917f8a0144ff56c0`]

- **held-out (8, frozen, never read):** 3822, 3824, 3834, 3839, 3844, 10510, 10514, 10517.
- **train/CV (32):** 3812, 3814, 3815, 3816, 3818, 3819, 3821, 3823, 3826, 3828, 3829, 3830, 3831, 3833, 3835, 3836, 3837, 3838, 3840, 3843, 3847, 3849, 3851, 3852, 3855, 3858, 3859, 10502, 10503, 10506, 10507, 10512.
- **Rule:**
  - previously exposed matches were excluded from the held-out pool;
  - the remaining matches were ranked by `sha256("m11-stage2-heldout-v1|id")`;
  - the first 8 became held-out.
- **Access control** [VERIFIED in source; CLAIM that it covers every path]: `bundle.fold_for_match` raises for held-out matches; `samples.py` and the scenario `MatchData` refuse them (`samples.HeldOutRefused`). Protection was **not** tested by opening protected data.

### 5.4 Outer folds (match → fold) [VERIFIED: `bundle.fold_matches`]

| Fold | Unseen test matches |
|---|---|
| 0 | 3815, 3826, 3835, 3852 |
| 1 | 3819, 3838, 3859, 10507 |
| 2 | 3821, 3829, 3831, 3847 |
| 3 | 3816, 3818, 3823, 3830 |
| 4 | 3812, 3833, 3843, 3855 |
| 5 | 3814, 3840, 10503, 10512 |
| 6 | 3828, 3836, 3837, 10506 |
| 7 | 3849, 3851, 3858, 10502 |

Each fold model was trained on the other 28 matches: 25 for fitting and 3 early-stopping matches [DOC].

### 5.5 Other reserved data

- SoccerTrack 118578 and 128058 are reserved and were never accessed; 118577 is spent [DOC: PROJECT_MASTER_CONTEXT §5].
- `F:\Football-Fatigue-Video` is the Session B worktree. Do not touch it.

---

## 6. M11 models and input/output contracts

### 6.1 Problem and label semantics [DOC: Stage 3 report, correction note; VERIFIED in `lab.py` / `predict.py`]

- **Setup:** two players come within ≤ 0.5 m in the raw data (a Stage 2 candidate).
  - A hidden interval `[g0, g1]` is simulated around the closest approach; both players' observations inside it are removed.
  - Input: two **incoming** tracklets (identities known: Player 1, Player 2) and two **outgoing** anonymous tracklets (Track X, Track Y).
- **Labels:** `y = 0` **MATCHED** (P1 → X, P2 → Y) or `y = 1` **SWAPPED** (P1 → Y, P2 → X).
  - The model output is **p = P(SWAPPED)**.
  - `recovery_y_to_stage3 = 1 − y` converts from the older m11_recovery convention.
- **Truth:** the provider's continuous slot tracks, verified against raw slots. **The occlusion is synthetic; the encounter is real.**

### 6.2 Exact gap semantics [DOC: m11_stage4c_correction_v1.md §2]

- **Gap boundaries:** `g0 = tc − g/2`, `g1 = tc + g/2`.
- **Grids:** incoming grid `g0 + k/15`, k = −30 … −1; outgoing grid `g1 + k/15`, k = 1 … 30.
- **Observations used:** only those strictly outside `[g0, g1]`, via interpolation; never extrapolated.
- **Training gaps:** every training sample has `gap_s = 0.8`, so other gap values are extrapolation for the GRU.
- **Unobserved duration** at 0.8 s is 0.80–0.93 s.
- **Frame:** encounter frame built from the incoming tracklets only (modes 0 / 1 / 2).

### 6.3 GRU C (`gru_gaplast`, config C) [VERIFIED: `m11_benchmark/models.py::make_net`, `stage4c.py`; parameter count recomputed]

- **Input:** per side `(2, 30, 4)` = x / 10 m, y / 10 m, mask, dt (s). Incoming sequences are reversed, so the point nearest the gap is last. The gap length is also an input.
- **Encoders:** `GRU(4 → 48)` for incoming and a separate one for outgoing tracklets; the summary is the last valid step, then `Linear(48, 48) + ReLU`.
- **Pair head:** `[a, b, a·b, |a − b|, gap]` (193) → `Linear 96` → ReLU → `Dropout 0.1` → `Linear 48` → ReLU → `Linear 1`.
- **Output:** `logit = s(1,Y) + s(2,X) − s(1,X) − s(2,Y)`, `p = sigmoid`. **43,585 parameters** [VERIFIED].
- **Training:**
  - AdamW, lr 1e-3, weight decay 1e-4, batch 256, BCE, gradient clip 1.0;
  - early stopping on 3 matches with patience 10, at most 160 epochs;
  - augmentation: reflection y → −y and order exchange with label flip;
  - **config C:** each epoch, each encounter is centred with probability 0.5, otherwise a valid shift of ±0.1 / 0.2 / 0.3 s (`m11-stage4c-shiftaug-v1`, 150,895 shifted samples) [DOC].
- **Bundles used:** seed 0, `neural/C/seed0/fold<f>_c160.pt`.

### 6.4 HGB C [VERIFIED: hyperparameters read from the 8 joblib bundles]

- **Features (36):** `perm_*` antisymmetric pair-cost differences (cost(SWAPPED) − cost(MATCHED)) plus context features (`football_fatigue.m11_recovery.motion`).
- **Symmetry:** trained on both outgoing orders; prediction `p = (f(x) + 1 − f(swap x)) / 2`.
- **Hyperparameters:** chosen per fold by inner CV from a 4-configuration grid. **Actual per-fold choice:**
  - folds 1, 2, 4, 5, 6: lr 0.05, max_iter 400;
  - **folds 0, 3, 7: lr 0.1, max_iter 200**;
  - all folds: max_leaf_nodes 31, min_samples_leaf 40, l2 1.0.
- **Known documentation error:** `lab_app.py` (models tab, ≈ line 810) and `docs/dashboard_explaination1.md` (≈ line 399) state "400 trees, learning rate 0.05" for all folds. This is wrong for folds 0, 3 and 7. Not corrected here (documentation-only task).

### 6.5 Inference API and routing `m11-routing-v1` [VERIFIED: `predict.py` constants; DOC: Milestone 1 report]

| Condition | Support | Model | Threshold (5 % / 2 %) |
|---|---|---|---|
| ≥ 3 grid points per tracklet, gap 0.75–0.85 s, noise ≤ 2 cm or undeclared | primary | GRU C | 0.6425 / 0.8525 |
| gap in [0.4, 1.6] s outside the GRU band, or noise in (2, 5] cm | fallback | HGB C | 0.7875 / 0.9029 |
| gap < 0.4 or > 1.6 s, noise > 5 cm, or < 3 points | unsupported | — | ABSTAIN; probabilities are diagnostic only |
| confidence `max(p, 1 − p)` below the threshold | (any) | — | ABSTAIN (`low_confidence`) |

- **Output fields:** `decision`, `predicted_pairing`, `pairs`, `p_swapped`, `confidence`, `threshold`, `support`, `model`, `reasons`, both models' probabilities, fold, checkpoint hashes, input summary.
- **Not supported:**
  - more or fewer than two players;
  - one player hidden while the other is visible;
  - a missing or foreign outgoing track;
  - appearance or team cues;
  - **video-scale noise (0.4–1 m)**;
  - real visual occlusion.

### 6.6 Checkpoints [VERIFIED: SHA-256 recomputed 2026-11-18; all 16 equal the manifest]

- **Manifest:** `data/processed/m11-infer-v1/bundles_v1.json` (SHA-256 `7108d2e5…0991`), schema `m11-infer-bundles-v1`, source `m11-stage4c-robustness-v1`.
- **Recorded check:** `all_reproduced: true`, `thresholds_match: true`.

**SHA-256 prefixes (each checkpoint matches its recorded hash):**

| Fold | GRU C `neural/C/seed0/fold<f>_c160.pt` | HGB C `hgb/C/fold<f>.joblib` |
|---|---|---|
| 0 | b4554cb8e605 | 755f13598090 |
| 1 | 86d411406d8b | 57344ba56049 |
| 2 | 1cdf140b05e5 | 02c04a178d23 |
| 3 | 6be4b900d6c2 | 51d3e2826563 |
| 4 | 7c27d92da6c3 | dbd9029b7a8c |
| 5 | 7b0fce311c43 | 750fab2c8940 |
| 6 | f814ba59e141 | 920291396ab4 |
| 7 | 3be7777945c2 | 1b0380d416ff |

**Reproduction** [DOC: Milestone 1 report]:
- GRU: max |Δp| vs stored OOF 9.1e-5 on CPU (tolerance 2.5e-4); exact on CUDA (tolerance 1e-6).
- HGB: 0.0 (tolerance 1e-12).
- No decision or abstention flips.
- Adapter rebuild from raw for 3812: 60 / 60 byte-identical.

---

## 7. Training, evaluation and reproducibility evidence

All figures below are development out-of-fold results on 32 matches [DOC].

| Item | Result | Source |
|---|---|---|
| Stage 3 dataset | g0p80 25,331 samples (18,777 independent; swapped share 0.499) | `m11_stage3_supervised_dataset_results_v1.md` |
| Stage 4B best | `gru_gaplast` BA 0.902 ± 0.005; HGB 0.870; SVM 0.859; RF 0.857; physics combo 0.696 | `m11_stage4_combined_comparison_v1.md` |
| Stage 4C | GRU C: centred 0.926, shifted 0.910, VISIBLE 0.845. HGB C: 0.861 / 0.878 | `m11_stage4c_robustness_and_model_selection_v1.md` |
| Texture probe | A texture-only model pairs tracklets at BA 0.701 (ESTIMATED 0.739, VISIBLE 0.525), so **provider texture carries signal** | same |
| Epoch-budget correction | GRU C hit the 160-epoch cap in 3 / 8 (seed 0), 3 / 8 (seed 1) and 7 / 8 (seed 2) fold fits; bundles of folds 0, 3 and 4 are budget-bound; **under-converged and selection-optimistic** | `m11_stage4c_correction_v1.md` |
| Milestone 1 fold BA | GRU 0.915–0.938; HGB 0.854–0.868 | `m11_milestone1_model_readiness_v1.md` |

**Milestone 2 (55,680 scenarios, 393 invalid, 5 % operating point)** [DOC: `m11_milestone2_scenario_testing_v1.md`]:

- **Baseline 0.8 s:** GRU BA 0.921, HGB 0.850; routed coverage 0.927, error 0.050.
- **Gap:**

  | Gap (s) | GRU BA | HGB BA | Routed error |
  |---|---|---|---|
  | 0.2 (unsupported) | 0.622 | 0.944 | — |
  | 0.4 | 0.837 | 0.921 | 0.020 (HGB) |
  | 1.6 | 0.729 | 0.798 | 0.130 (HGB) |

- **Noise:** at 2 cm the GRU-routed error is 0.102. At 25–50 cm both models are near chance (HGB 0.73 / 0.66).
- **Dropout:** random 50 % → GRU 0.801, routed error 0.160; the router cannot see dropout.
- **Context:** 0.5 s per side → routed error 0.153.
- **Shift:** HGB collapses at ±0.5 s (0.751 / 0.713); GRU is robust (0.855 / 0.902).
- **Totals:** 34,630 accepted correct, 2,999 accepted wrong, 6,317 low-confidence abstentions, 11,341 unsupported. **Coverage 0.681, realised error 0.080 against the 5 % target.**
- **Hard subsets:** both fully VISIBLE in the gap: error 14.7 % vs 3.1 % for ESTIMATED. Also no-crossing encounters and sharp turns.

**Milestone 2 output integrity** [VERIFIED 2026-11-18]:
- `summary/metrics_v1.json` equals the manifest `metrics_sha256`.
- All 32 `scenarios.parquet` and `arrays.npz` files equal their recorded SHA-256 values.
- **Stale code hashes** in `m11-scenarios-v2/run_config.json`:
  - `scenarios.py` and `scenario_eval.py` differ from the recorded `code_sha256`, because they were edited after the run (a docstring fix, plus a refactor adding `_build` / `build_custom` / `example_sets`);
  - `adapter.py`, `bundle.py` and `predict.py` still match;
  - regeneration was checked equal to the saved arrays (60 / 60) [CLAIM, 2026-11-17 / 18 session];
  - **not overwritten**;
  - a resumed run would refuse, so any rerun needs a new output version.
- **Report inaccuracy:** M2 report §8 mentions only the `scenario_eval.py` docstring edit; the `scenarios.py` refactor is not listed there.

**Earlier aborted run:** `m11-scenarios-v1` crashed on an empty tracklet. It is kept unchanged.

---

## 8. Milestones 1–3: completion and verification matrix

| Item | Implemented | Test evidence | Other evidence |
|---|---|---|---|
| M1 `m11_infer` | Yes, untracked | `tests/test_m11_infer.py` (18 `def test_` functions; 38 collected items per report) | check-bundles run [DOC]; hashes re-verified [VERIFIED] |
| M2 scenario suite | Yes, untracked | `tests/test_m11_scenarios.py` (13 functions; 23 items per report); report: "61 passed" for M2 + M1 tests [DOC] | Outputs and hashes re-verified [VERIFIED] |
| M3 Trajectory Recovery Lab | Yes, untracked | `tests/test_m11_lab.py` (18 functions; **22 passed**, run 2026-11-18 in this session [CLAIM]) | Headless Edge CDP browser walkthroughs of Demo and Research lab, no exceptions, 2026-11-18 [CLAIM] |
| Full suite | — | **1112 passed, 6 skipped** (log `full_tests2.log`, finished 2026-11-18 ≈ 01:50) [CLAIM] | — |

**Caveat:** the last edit to `lab_app.py` (a layout change in the failure-analysis panel, ≈ 01:50) came **after** that pytest run started. It was covered only by the subsequent browser walkthrough, not by pytest. **No tests were run for this handoff.**

**M3 tests cover:**
- truth separation (Public object, figure JSON, colours);
- animation timeline (15 Hz, phases, key frames, links only after `g1`);
- held-out refusal; match → fold mapping and validity checks;
- regeneration determinism;
- live inference equal to saved M2 within tolerance, with the same routed outcome;
- truth is never an inference input;
- raw-window provenance;
- explanation helpers; M2 outputs unchanged;
- three AppTest UI flows (demo reveal, lab regenerate / reset, gallery / models / JSON).

---

## 9. Dashboard operational guide and known problems

**Launch** [VERIFIED: `__main__.py` diff and `scripts/m11_lab.cmd`]:

```powershell
.\.venv\Scripts\python.exe -m football_fatigue.m11_dashboard --port 8610 lab
# or double-click scripts\m11_lab.cmd ; then open http://127.0.0.1:8610
```

Port 8610 is used because Windows reserves 8494–8593 on this machine [CLAIM]. Options go **before** the subcommand. Other subcommands: `encounters` (explorer) and the default (Step 8 RF app).

**Modes:**
- **Demo:** 7 guided steps: pick → approach → hidden interval → run → compare → reveal → try another gap.
- **Research lab:** match / encounter, seeds, one-factor (M2) or custom variants.

**Tabs:** 🎬 Scenario, 🖼 Example gallery, 📊 Evaluation (Milestone 2), 🧠 How the models work, 📄 Load JSON tracklets, ❓ Guide & legend.

**Key features:**
- live GRU C / HGB C inference with the unseen-fold models;
- truth hidden until **Reveal ground truth**;
- a validity panel available before the reveal;
- raw provider rows and a distance chart after the reveal;
- failure analysis with a constant-velocity physics reference and movement-family statistics;
- saved-M2 cross-check;
- JSON input and output.

**Documentation:**
- `docs/dashboard_explaination1.md` (filename spelling as requested by the user): every control and term;
- `docs/m11_demo_guide.md`: 3–5 minute script.

**Known problems:**
1. HGB hyperparameter text is wrong for folds 0, 3 and 7 (§6.4).
2. The policy routes noisy, dropout and short-context inputs to GRU with up to 16 % error; this is shown to users as a limitation.
3. The first match load takes ≈ 10 s.
4. The demo is development data (out-of-fold), not independent validation.
5. Gallery category names reveal the outcome; use the random buttons for fair demos.

---

## 10. Git state [VERIFIED 2026-11-18]

```text
## feature/m11-identity-finalization
 M src/football_fatigue/m11_dashboard/__main__.py
?? docs/dashboard_explaination1.md
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

This handoff adds `?? docs/handoffs/`.

**Latest 8 commits:**

| Commit | Date | Message |
|---|---|---|
| `1bd016d` | 11-17 | Add M11 identity recovery research, benchmarks and dashboards (HEAD) |
| `1458e1b` | 11-06 | chore: Step 1 handoff, ignore demo asset bundle, scrub local username |
| `8505c5c` | 11-06 | docs: project master context, technical walkthrough, advisor reports |
| `4653047` | 11-06 | feat(m11): encounter catalogue, identity matching, locked-test evaluation |
| `3ec9441` | 11-06 | feat(multicam): unified multi-camera lab |
| `936bc07` | 11-06 | feat(demo): coach dashboard demo, overlay continuity, review layer |
| `1d020fc` | 11-06 | feat(analytics): coach analytics, motion, observed-quantity modules |
| `2e18001` | 11-06 | feat(vision): detection cache, identity association, MOT evaluation |

**Committed M11 code (in HEAD):** `m11_benchmark/` (Stage 4A–4C), `m11_dashboard/` (Step 8, encounters), `m11_recovery/`, `window_intersections/`, plus Stage 1–4C reports. Everything from Milestones 1–3 is uncommitted. All Git state changes are performed by the user.

---

## 11. Implemented vs documented vs unverified vs planned

| Status | Items |
|---|---|
| **Implemented and verified now** | 16 fold checkpoints match their manifest hashes; frozen split hash; M2 output hashes; fold mapping; GRU architecture and parameter count; HGB per-fold hyperparameters; routing constants; Git state |
| **Implemented, evidence from earlier runs** [DOC / CLAIM] | M1 reproduction (Δp ≤ 9.1e-5); M2 run and metrics; M3 tests (22) and full suite (1112 / 6 skipped); browser walkthroughs |
| **Documented only** | Track A milestone reviews ("pending independent review"); Stage 4 numbers (not recomputed); earlier RF locked-test result |
| **Not verified** | That every code path refuses held-out data (only the main guards were inspected); licensing; event-version authority; real-world (video) performance; the routing band ±0.05 s around 0.8 s (declared, not measured) |
| **Planned / proposed (needs approval)** | Stage 5 held-out evaluation (`m11_stage5_final_protocol_PROPOSED_v1.md`); final training on all 32 matches (called "Milestone 4" in the reports); routing v2; variable-gap / noise / dropout GRU retraining; VISIBLE-only reporting; declaring torch / pandas / pyarrow dependencies |

---

## 12. Scientific limitations and risks

1. **Synthetic occlusion:** real encounters with hidden intervals cut out artificially. No real camera occlusion was tested.
2. **Development-only evidence:** thresholds, config C, the 160-epoch budget and routing were all chosen on the same 32 matches, so the results are optimistic.
3. **GRU under-converged** (budget binding) and **trained only at gap 0.8 s, on clean data**: fragile to gap change, noise ≥ 2 cm, dropout and short context. It is confidently wrong in some cases (p ≈ 0 / 1).
4. **Provider-texture leakage risk:** ESTIMATED segments carry an interpolation texture that models exploit. VISIBLE-only performance is much lower; it is the better proxy for real tracking.
5. **Policy:** realised error 8.0 % across the stress suite vs a 5 % target. Thresholds do not transfer beyond 0.8 s.
6. **Population:** all encounters are ≤ 0.5 m; separated pairs are untested. Only 2 players, both hidden simultaneously.
7. **Video incompatibility:** video positional error of 0.4–1 m is far outside the supported 5 cm.
8. **Official FIFA validation/test are consumed** by the earlier RF track. The only unbiased option left inside FIFA is the 8-match internal held-out set (single use).
9. Tracking-derived results say **nothing about fatigue**.

---

## 13. Exact stopping point

- Milestone 3 is complete and **awaiting user review**. The user then asked for this verified handoff. It is the only change in this step.
- No milestone after M3 has started: no Stage 5, no final training, no routing v2, no retraining.
- No server is running from this session (the last lab server was stopped by the user).
- Everything from Milestones 1–3 is uncommitted on `feature/m11-identity-finalization`.

---

## 14. Recommended next actions (each requires user approval)

1. **Review and commit** the M1–M3 files and this handoff (the user performs all Git operations). First fix the HGB hyperparameter text in `lab_app.py` and `dashboard_explaination1.md` (§6.4) if desired.
2. **Declare missing dependencies** (torch, pandas, pyarrow) in `pyproject.toml`, without installing anything new.
3. **Decide on routing v2:**
   - measured density / context inputs;
   - HGB at its 2 % threshold for off-band gaps and irregular input;
   - possibly admit gaps of 0.2–0.4 s to HGB.
4. **Decide on final training on all 32 matches** (larger epoch budget chosen by nested selection; optionally variable-gap, noise and dropout augmentation).
5. **Then, and only then, the single Stage 5 held-out run:**
   - approve and freeze the protocol;
   - add a logged `--final-heldout` build mode;
   - evaluate once;
   - report VISIBLE-only figures alongside the overall ones.
6. Any video application needs a noise-robust model and real annotated occlusions, which is a separate milestone.

---

## 15. Handoff for ChatGPT (how to continue without reconstruction)

- **Rules first:** `PROJECT_RULES.md`.
  - Git is read-only for agents.
  - Never modify `data/raw/`.
  - Write new versioned output directories and never overwrite.
  - Report only checks that actually ran.
  - Ask before installs, training, GPU jobs or any protected-data access.
  - One milestone at a time; stop and report after each.
- **Never open:**
  - held-out 3822, 3824, 3834, 3839, 3844, 10510, 10514, 10517;
  - official validation/test (`m11-encounters-v1/splits.json`);
  - SoccerTrack 118578 / 128058;
  - `F:\Football-Fatigue-Video`.
- **The task:** given 2 incoming identified tracklets and 2 outgoing anonymous tracklets around a simulated hidden interval, predict `p = P(SWAPPED)`, route with `m11-routing-v1`, or abstain.
- **Use the existing API:** `from football_fatigue.m11_infer.predict import Predictor`, then `Predictor.for_match(dev_id)` or `Predictor(fold).predict(incoming, outgoing, g0, g1, noise_sigma_m=...)`. Do not reimplement it.
- **Ground truth for claims:**
  - reports in `docs/research/` (Stage 4C report, correction note, Milestone 1 and 2 reports);
  - `data/processed/m11-infer-v1/bundles_v1.json`;
  - `data/processed/m11-scenarios-v2/summary/metrics_v1.json`;
  - this file.
- **Commands:**
  - tests: `.\.venv\Scripts\python.exe -m pytest tests/test_m11_infer.py tests/test_m11_scenarios.py tests/test_m11_lab.py`;
  - full suite: `-m pytest` (≈ 5 min);
  - lab: `-m football_fatigue.m11_dashboard --port 8610 lab`.
- **Phrase every result as:** "development, unseen-match out-of-fold, synthetic interruption of real provider tracking". Never call it validated, real occlusion, video-ready or fatigue-related.
- **Open decisions for the user:**
  - commit M1–M3;
  - routing v2;
  - final 32-match training (budget and augmentation);
  - Stage 5 protocol approval and the held-out build mode;
  - fixing the HGB documentation error.
