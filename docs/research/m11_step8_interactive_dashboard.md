# M11 Step 8 — Interactive identity recovery dashboard

> **Update 2026-11-08 (Step 8.1):** advisor demo tab, clearer replay, prediction and explanation views, and stricter reference-leak
> tests. Presentation only; the model, thresholds and results are unchanged. See section 14.

**Date:** 2026-11-06. **Scope:** an interactive demonstration of the frozen Step 6 model on real FIFA World Cup 2022 RAW tracking. Only
development matches are served.

**What this step is not:**
- It is not an experiment: nothing was trained, tuned, recalibrated or re-selected.
- No threshold, feature, assignment rule or tier definition changed.
- No RAW file or earlier experiment output was written.
- The validation and locked-test splits are not served by the dashboard. The locked test stays consumed (Step 7).

Every agreement figure in this document and in the dashboard is **provider-reference agreement**. The provider's identity labels
are a reference, not independently verified ground truth.

## 1. Summary

**Launch command:**

```powershell
.\.venv\Scripts\python.exe -m football_fatigue.m11_dashboard
```

**What it does:**
- Loads the 10 development matches that have Step 2–3 outputs, chosen from the frozen split and frozen specification (no
  hard-coded IDs).
- Draws a selected encounter on a 105 × 68 m pitch from the RAW tracking cache.
- Runs the checksum-verified frozen model through the unchanged Step 6 code path.
- Shows candidate probabilities and MATCHED / UNCERTAIN / UNMATCHED decisions at both frozen operating points.
- Keeps the provider reference hidden until the user clicks **Reveal Provider Reference**.

**Verification:**
- The per-encounter inference reproduces the Step 6 batch path **exactly** on every instance of all 10 development matches
  (section 8.2).
- It reproduces the **saved** Step 6 frozen predictions exactly (section 8.3).
- Tests: 19 new tests pass, and the full suite passes (section 8.1).

**The most important caveat:** the 10 development matches are the frozen model's *training* matches. High-tier pairs without
ESTIMATED endpoints were fitted on, so agreement seen in the dashboard is in-sample and optimistic. The dashboard says so on every
page. The unbiased numbers are the Step 7 locked-test results, shown in the Research Results tab as historical values.

## 2. Architecture

New package `src/football_fatigue/m11_dashboard/`. It is isolated: it imports `matching`, `encounters`, `ingestion` and
`match_data` read-only, and no existing module was edited.

| file | role |
|---|---|
| `data.py` | Discovers development matches; loads and checksum-verifies task B instances and references; builds the identity-free public context; slices the RAW cache for the pitch; refuses validation and test matches. |
| `inference.py` | Loads and verifies the frozen model. Runs `validate.features_only` → forest → stored Platt calibration → `validate.decide` on one instance. Also: assignment invariant check, Saabas path attribution, batch-reproduction helpers. |
| `pitch.py` | Plotly tactical board: pitch markings, animated frames, play/pause, slider, reference and prediction layers. |
| `research.py` | Reads the saved, checksummed Step 6/7 scoring summaries and the frozen spec's development reference. |
| `app.py` | Streamlit user interface (5 tabs). |
| `__main__.py` | CLI: launches the app, or `verify --match <id>` (prints a reproduction report and writes nothing). |

Dependencies: `streamlit` 1.67.0 and `plotly` 6.9.0 were installed with approval, together with their own requirements (including
`pandas` 3.1.1). They are declared as a new optional extra `m11dash` in `pyproject.toml`. `numpy` 2.5.4 and `scikit-learn` 1.9.1
were checked after the install and are unchanged; they are the versions recorded in the frozen manifest.

## 3. Data flow and what each view may contain

```
m11-encounters-v1/splits.json ─┐
m11-frozen-v1/spec.json ───────┴─► development_matches()   (spec development list ∩ split "development" ∩ outputs present)

m11-matching-v3/<mid>/root.instances_b.jsonl ─► instance  ──► infer_instance() ──► probabilities, decisions
                                       (matcher-visible)          (frozen model, no reference)

m11-matching-v3/<mid>/root.references.jsonl ─┬► public_context()      shown always   (whitelisted keys only)
                                             └► provider_reference()  shown only after "Reveal Provider Reference"

m11-encounters-v2/<mid>/root.cache.npz ─┬► raw_window()      other players' RAW positions (participants excluded in every row)
                                        └► hidden_interval() participants' withheld positions (only after reveal)

RAW metadata (Dataset.metadata) ───────► pitch size, period start times, competition name
m11-validation-v1/scoring, m11-test-v1/scoring, frozen spec ─► Research Results (historical)
```

**The model** receives only the task B instance: anonymous `in-k` / `out-k` tracklets with `t`, `x`, `y`, `visibility` and
`confidence`, and an empty withheld interval. `infer_instance` rejects any instance that contains interval observations, and it
has no reference argument.

**The public context** is copied key by key from the reference record. Keys: `instance_id`, `match_id`, `period`, `tier`,
`below_high_reasons`, `start_frame`, `end_frame`, `start_time_s`, `end_time_s`, `rows`, `encounter_ids`, `step1_categories`,
`discontinuity_evidence` and `n_participants`. To these it adds:
- from the instance: `ambiguity_s`, `gap_bin`, `n_incoming`, `n_outgoing`;
- `min_distance_m`: the Step 1 minimum distance, which is symmetric and carries no mapping;
- `suspicious`;
- `filter_sharp_turn` (see below).

**Kept out of the public context:** `incoming`, `outgoing` (slot, side, jersey, player and team IDs), `expected`,
`without_incoming` / `without_outgoing`, `team_relation`, and the `difficulty` block.

**Why `difficulty` is hidden:** `benchmark.difficulty()` computes `crossing`, `sharp_turn`, `max_heading_change_deg` and
`mean_speed_m_s` *from the expected mapping*. Showing them would hint at the answer.

**The sharp-turn filter:** `filter_sharp_turn` is reference-derived. The sidebar offers it as a selection filter, with a tooltip
saying so. It is never displayed per encounter before reveal and never reaches the model.

**The pitch:**
- **Context layer:** the RAW positions of every *non-participant* slot over the before, withheld and after rows. Participant slots
  are taken from the reference internally and only to exclude them, so the matched players' withheld positions never appear before
  reveal.
- **Withheld positions after reveal:** `hidden_interval` draws them as red dotted, "reference only" paths.

**Names and personal data:** player names are never loaded. After reveal, identities are shown as side and shirt number only
(for example `home #26`); no player ID is displayed.

## 4. Input and output schemas

**Inputs** (all read-only and checksum-verified against their manifests):

| artifact | used fields |
|---|---|
| `m11-frozen-v1/{manifest,spec}.json`, `model.joblib` | model, 51 features, Platt `a`/`b`, `operating_points`, `splits`, `development_reference`; SHA-256 must equal the Step 7 constants (`629cca77…`, `6fa1df5c…`) |
| `m11-encounters-v1/splits.json` | split per match; SHA-256 must equal the spec's `manifest_sha256` |
| `m11-matching-v3/<mid>/root.instances_b.jsonl` | task B instance schema: `instance_id`, `task`, `ambiguity_interval_s`, `incoming[]`/`outgoing[]` tracklets (`tracklet_id`, `t` s relative to the start of the withheld interval, `x`/`y` m, `visibility` 1 VISIBLE / 2 ESTIMATED, `confidence` 1–3), `during` (empty) |
| `m11-matching-v3/<mid>/root.references.jsonl` | public keys above; hidden identity and mapping keys only after reveal |
| `m11-matching-v3/<mid>/root.manifest.json` | must be `m11-matching-v3`, the same match, split `development`, no synthetic swaps, no smoothed coordinates |
| `m11-encounters-v2/<mid>/root.cache.npz` | `x`, `y`, `visibility`, `video_time_s`, `slot_side`, `slot_jersey` (about 100 MB per match in memory; one match cached) |
| `data/raw` match metadata | `pitch_length_m`, `pitch_width_m`, `start_period_{1,2}_s`, `competition_name` |
| `m11-validation-v1/scoring`, `m11-test-v1/scoring` | `summary.json` after `validate.verify_dir` |

**Output of `infer_instance`** (held in memory only):
- `pairs[]`: `incoming`, `outgoing`, `reachable`, calibrated `probability`, `raw_probability`, and the 51 named `features`.
- `decisions[point][]`: exactly `validate.decide`'s records, namely `incoming`, `decision`, `outgoing`, `candidate`,
  `probability`, `margin` and `evidence`.
- `operating_points`, plus the feature matrices.

**Output of `verify`:** JSON on stdout with `instances`, `pairs`, `max_abs_probability_diff`, `decision_mismatches`,
`assignment_problems`, the model and spec SHA-256, and the source checksums. No file is written.

**Coordinates:** source pitch coordinates, never transformed. Metres, origin at the pitch centre, x along the touchlines, y along the
end lines (`match_data.COORDINATE_SYSTEM`); attack direction not normalised. They are plotted as-is with equal axis scales.

**Times:**
- The withheld interval starts at 0 s.
- "Time since period start" = provider video time − `start_period_k_s` from the metadata. The two share one clock: the cache's
  period ranges equal the metadata bounds to the millisecond for 3829 and 3812.

**Replay rate:** the replay steps through every second RAW row (about 15 Hz of the 29.97 Hz data).

## 5. Model artifact requirements

The dashboard refuses to run inference unless all of the following hold:
1. `m11-frozen-v1/manifest.json` verifies `model.joblib` and `spec.json`, through `validate.load_frozen`, which also checks the
   feature list against `FROZEN_FEATURES` and the bundle's operating points against the spec.
2. Both SHA-256 values equal the values recorded at Step 7.
3. The model expects 51 features and the spec has exactly the `relaxed` and `conservative` points.

**Operating points:** read from `spec.json`. That gives relaxed 0.76 / margin 1.0 and conservative 0.87 / margin 1.0. They do not
come from `validate.OPERATING`, whose relaxed 0.75 was resolved to 0.76 at freeze time; a test asserts this.

**Determinism:**
- **The finding:** the frozen forest was pickled with `n_jobs=2`. With two threads, scikit-learn adds the per-tree probabilities
  in a varying order, and repeated predictions differed by up to 2.2 × 10⁻¹⁶.
- **The fix:** the dashboard sets `n_jobs=1` on the loaded forest *in memory*. Repeat runs are then bit-identical; the
  single-threaded result differs from the two-thread result by at most 1.7 × 10⁻¹⁶. The file is not changed.
- **Effect on Steps 6–7:** the same 10⁻¹⁶ jitter existed there. It was invisible because saved probabilities are rounded to 6
  decimals, and it changes no decision.

If verification or inference fails, the app shows an explicit error and **no** prediction. There is no fallback to cached or
invented scores.

## 6. Install and run

```powershell
# dependencies (already installed in .venv for this step)
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -e ".[dev,m11dash]"

# launch (opens http://127.0.0.1:8501; binds to this computer only)
.\.venv\Scripts\python.exe -m football_fatigue.m11_dashboard

# options: --root data/raw  --port 8501  --address 127.0.0.1  --headless
# reproduction check on one development match (prints JSON, writes nothing)
.\.venv\Scripts\python.exe -m football_fatigue.m11_dashboard verify --match 3829

# tests
.\.venv\Scripts\python.exe -m pytest tests/test_m11_dashboard.py
```

The first interactive Streamlit launch on a machine may ask for an e-mail address; press Enter to skip. `--headless` skips the
question and does not open a browser.

## 7. The dashboard, tab by tab

**Sidebar:**
- Selection: match (development only), period, encounter.
- Filters, all computed from existing metadata:
  - 2 vs 3+ incoming tracklets;
  - withheld-gap bins `0-1.5s`, `1.5-2.5s`, `2.5-4s`, `>=4s` (the fixed Step 2 bins; short = first bin, long = last two);
  - reference tier (high / lower / natural suspicious);
  - sharp turn (reference-derived, labelled as such);
  - suspicious discontinuity (natural-suspicious tier or Step 1 discontinuity evidence).
- No filter was omitted; the only compromise is that the sharp-turn filter is reference-derived.

**1 · Match Explorer:**
- the selected encounter's metadata: period and time since period start, provider video time, frame range, players involved,
  tracklet counts, withheld gap, minimum separation, tier and quality flags, Step 1 categories, discontinuity evidence, and how
  many of its pairs were training pairs;
- a glossary of tracking, encounter, tracklet, identity masking, candidate pair, global assignment and abstention;
- a provenance legend (RAW observation, provider-estimated, withheld, model prediction, provider reference);
- a table of the filtered encounters.

**2 · Tactical Replay:**
- an animated Plotly pitch with Play / Pause and a scrub slider inside the board;
- Step ◀ / ▶ and Before / Withheld / After jumps;
- playback speed 0.25×–2×;
- a full-pitch toggle;
- a before / withheld / after timeline bar;
- optional model links;
- a reveal button.

What the board draws:
- filled markers are VISIBLE observations and hollow markers ESTIMATED ones;
- incoming trails stay faded after their last observation;
- during the withheld interval the matched players are not drawn;
- after reveal, the tracklets are recoloured by provider identity and the withheld positions appear as red dotted
  "reference only" paths.

**3 · Identity Recovery:**
- the two operating points and the model checksum;
- **Run Identity Recovery**;
- a probability heatmap (✕ = unreachable);
- per incoming tracklet: best candidate, p, runner-up p, margin, reachable candidates, relaxed and conservative decisions, and
  whether the pair is in-sample.

After **Reveal Provider Reference**:
- the provider identity and continuation per incoming tracklet, with the outcome at each point (agrees / disagrees /
  abstained / UNMATCHED);
- committed links that agree or disagree;
- caveats: tier, natural-suspicious wording, discontinuity evidence, reference-derived crossing, turn and team relation, and the
  in-sample notice;
- the disclaimer.

**4 · Model Explanation:** for a chosen candidate pair:
- six plain-language measurements: spatial proximity, gap duration, velocity continuity, heading consistency, extrapolation error
  and global assignment ambiguity (assignment regret and reachable candidates);
- a Saabas decision-path attribution of the raw forest probability: top 12 features, sums per feature group, and
  bias + contributions = raw probability → calibrated probability;
- the frozen forest's global MDI importance;
- all 51 feature values with their descriptions from `ml.FEATURES` / `ml2.FOOTBALL_FEATURES`.

**5 · Research Results:**
- the locked-test headline metrics, read from `m11-test-v1/scoring/summary.json`: ROC-AUC 0.838 [0.826, 0.848]; relaxed
  coverage 40.3%, false association 8.8%; conservative 27.0%, 4.6%;
- development / validation / test with 95% match-bootstrap intervals;
- the locked-test comparison methods;
- the source paths.

The tab states that these are historical results, not computed from the selected encounter.

## 8. Verification

### 8.1 Tests

`tests/test_m11_dashboard.py`: 19 tests. Synthetic tests use a temporary processed tree. Real-data tests skip when the
git-ignored artifacts are absent.

| requirement | tests |
|---|---|
| development-only selection | `test_development_only_match_selection`, `test_real_development_matches_come_from_the_frozen_split`, `test_split_manifest_change_is_refused` |
| frozen model loading | `test_frozen_model_and_feature_schema` (checksums, wrong-checksum refusal) |
| feature schema | the same (`FROZEN_FEATURES`, 51, no `conf_*`), `test_training_pairs_match_the_label_protocol` |
| exact inference reproduction | `test_exact_reproduction_of_batch_inference_on_development`, `test_exact_reproduction_of_saved_frozen_predictions` |
| threshold selection | `test_operating_points_come_from_the_frozen_spec`, `test_threshold_rule_and_assignment_on_real_encounters` |
| assignment validity | `test_check_assignment_detects_violations`, `test_threshold_rule_and_assignment_on_real_encounters` |
| reference hidden | `test_public_context_holds_no_identity_or_mapping`, `test_real_contexts_hold_no_identity`, `test_inference_accepts_only_task_b_instances`, `test_app_hides_reference_until_revealed_and_writes_nothing` (headless Streamlit `AppTest`: no identity before reveal, before or after running the model; identity after reveal) |
| coordinate mapping | `test_pitch_uses_source_coordinates_unchanged`, `test_coordinate_mapping_from_raw_cache` (the cache's slot × rows equal the instance tracklets exactly) |
| missing artifacts | `test_missing_artifacts_raise_clear_errors` |
| no changes to frozen files | `test_app_hides_reference_until_revealed_and_writes_nothing` (SHA-256 and mtime of `m11-frozen-v1`, the match's `m11-matching-v3` and `m11-encounters-v2` files, before and after a full app session) |
| explanation is exact and read-only | `test_path_contributions_are_exact_and_leave_predictions_unchanged` |
| research values come from artifacts | `test_research_results_read_from_saved_scoring` |

**Results:** `tests/test_m11_dashboard.py`: 19 passed. Full suite: 750 passed, 6 skipped, 1 warning (113 s). The previous suite
was 731 passed; the warning is the existing ByteTrack deprecation in the vision tests.

### 8.2 Full-match reproduction (development)

`python -m football_fatigue.m11_dashboard verify --match <id>` compared `infer_instance` instance by instance with the Step 6 batch
path (`validate.predict_tables`, frozen model only), on every instance of every development match:

| match | instances | pairs | max \|Δp\| (6 d.p.) | decision mismatches | assignment-invariant violations | seconds |
|---|---|---|---|---|---|---|
| 3812 | 845 | 3983 | 0.0 | 0 | 0 | 32.9 |
| 3816 | 774 | 3741 | 0.0 | 0 | 0 | 30.4 |
| 10506 | 781 | 3649 | 0.0 | 0 | 0 | 30.4 |
| 3814 | 628 | 3032 | 0.0 | 0 | 0 | 23.9 |
| 3847 | 891 | 4223 | 0.0 | 0 | 0 | 32.8 |
| 3836 | 791 | 3788 | 0.0 | 0 | 0 | 29.6 |
| 3833 | 686 | 3301 | 0.0 | 0 | 0 | 26.6 |
| 3815 | 873 | 4170 | 0.0 | 0 | 0 | 36.4 |
| 3851 | 832 | 3847 | 0.0 | 0 | 0 | 35.4 |
| 3829 | 823 | 3943 | 0.0 | 0 | 0 | 33.4 |
| **all** | **7924** | **37677** | **0.0** | **0** | **0** | |

Decision records (decision, outgoing, candidate, probability, margin, evidence) are compared as whole dictionaries, at both points.
All ten runs exited with code 0.

### 8.3 Saved predictions

No artifact contains the frozen model's predictions on development matches. The spec's development reference comes from
leave-one-match-out fold models that were not saved. The only saved frozen-model predictions are the Step 6 validation outputs.
So `test_exact_reproduction_of_saved_frozen_predictions`:
1. reads the matcher-visible instances of the first validation match, 3832, through the unchanged `validate.load_visible`;
2. reads the saved `m11-validation-v1/predictions` after `verify_dir`;
3. checks that the dashboard's probabilities (rounded to 6 decimals as saved) and both points' decisions are identical, on 40
   instances.

No reference, scoring or test file is read, and validation data is not shown in the dashboard. Step 7 made the same reproduction
check on 3832. This satisfies acceptance criterion 7 in the strongest form available. **Deviation from the wording of criterion
7:** the saved predictions it reproduces are validation predictions, because no development predictions of the frozen model
were ever saved.

### 8.4 Visual check

I have no browser here, so I redrew Plotly figure data with matplotlib into scratch images. These showed:
- pitch markings in place;
- tracklets at their source coordinates;
- no participant marker in the withheld interval;
- reference paths only after reveal;
- dashed UNCERTAIN links.

The real Streamlit server was launched with the CLI on `127.0.0.1` and answered its health endpoint (HTTP 200). The Streamlit UI
itself was exercised headless with `AppTest`, not viewed in a browser.

## 9. Example walkthrough (development match 3829)

Examples were chosen by fixed rules on public metadata, in time order — never by model output, validation or test results:
- A: the first high-tier 2-player encounter with a withheld gap over 1 s;
- B: the first high-tier encounter with 3+ incoming tracklets;
- C: the first natural-suspicious encounter.

Match 3829 contains 823 encounters:
- by tier: 123 high, 693 lower, 7 natural suspicious;
- by incoming tracklets: 726 with 2, 97 with 3;
- by gap: 243 at 0–1.5 s, 257 at 1.5–2.5 s, 197 at 2.5–4 s, 126 at ≥4 s.

| | A `453e55e4…` | B `0c080ccb…` | C `82a4f751…` |
|---|---|---|---|
| time | P1 00:11.8 | P1 20:14.5 | P1 04:57.0 |
| tracklets in / out | 2 / 2 | 3 / 3 | 2 / 2 |
| withheld gap, min. separation | 1.03 s, 1.09 m | 2.37 s, 0.70 m | 1.17 s, 0.88 m |
| tier | high | high | natural suspicious (implausible jump) |
| model, both points | in-0→out-0 MATCHED p 0.998 (margin 6.1); in-1→out-1 MATCHED p 0.992 (4.8) | all UNCERTAIN: best p 0.56 / 0.66 / 0.55, margins 0.25 / 0.64 / 0.22 < 1.0 | in-0→out-0 MATCHED p 0.947 (2.9); in-1→out-1 MATCHED p 0.909 (2.3) |
| after reveal | both agree with the provider reference; reference-derived: crossing, heading change 69° | each abstained tracklet's best candidate is the provider continuation, but close alternatives made it abstain | both agree with the provider reference; unverified tier, so agreement here establishes nothing about a swap |
| in-sample pairs | 4 of 4 | 9 of 9 | 0 (unverified tier is never trained on) |

A and B are in-sample: the model was fitted on these pairs' provider labels.

## 10. Known limitations

1. **In-sample demonstration.** Development matches are the frozen model's training data. Agreement in the dashboard is optimistic;
   the unbiased figures are the historical locked-test results.
2. **Provider reference, not ground truth.** All agreement is with provider labels. Natural-suspicious encounters may have wrong
   labels, and disagreement there is not a confirmed swap.
3. **Task B is artificial withholding.** The interval is removed on purpose from real trajectories. Proximity is not proof of
   visual occlusion, and no video was checked.
4. **Sharp-turn filter is reference-derived.** Selecting it tells the viewer the provider mapping includes a sharp turn. Crossing,
   heading change and team relation are therefore shown only after reveal.
5. **Explanations are descriptive.** The Saabas attribution is exact and additive for this forest but not causal, and it is biased
   towards features split near the root. MDI importance is global and training-based. No claim is made that a feature *caused* a
   decision.
6. **Replay rate and scope.** The replay uses every second RAW row (about 15 Hz). The ball is not drawn (the Step 1 cache has no
   ball), and attack direction is not normalised.
7. **Memory.** One match's RAW cache (about 100 MB of arrays) is held in memory; switching matches reloads it in about 0.3 s.
8. **Validation instances in one test.** The validation instances are read in one test, only to reproduce saved predictions
   (section 8.3); they are never shown.
9. **First-run prompt.** Streamlit's first-run question needs an interactive terminal or `--headless`.
10. **Streamlit and Plotly versions.** These are new optional dependencies (`m11dash`); the rest of the project does not need them.

## 11. Scientific disclaimers (as shown in the app)

- "Agreement is measured against the data provider's identity labels … **not** verified identity accuracy."
- The withheld interval is "real observations deliberately removed from the model's input", not a real occlusion; "being close is
  not proof of a visual occlusion".
- Natural-suspicious encounters: "disagreement does not establish which side is right. This is not a confirmed identity swap."
- Research Results: historical Step 6/7 values "not computed from the encounter selected above".
- E0-Adapted "is a FOOTBALLTrace-inspired matcher built for this task; the published FOOTBALLTrace method corrects already-labelled
  tracks and was not reproduced or outperformed here."
- The dashboard never applies a correction to provider trajectories.

## 12. Demonstration script for the advisor (about 10 minutes)

1. **Launch** (`python -m football_fatigue.m11_dashboard`). Point out the caption: real World Cup 2022 tracking, frozen model,
   development matches only, and the in-sample note in the sidebar.
2. **Match Explorer.** Open "Key ideas" and read *Encounter*, *Tracklet* and *Identity masking*: the model only sees anonymous
   before and after pieces.
3. **Select example A.** Match 3829, period 1, filters "2 players" and high tier; choose the encounter at 00:11.8. Go to
   **Tactical Replay** and press **Before**, then Play.
   - Two anonymous incoming tracklets approach.
   - At **Withheld** the two players disappear while everyone else keeps moving: their observations were removed from the
     model's input.
   - At **After**, two anonymous outgoing tracklets appear.
4. **Identity Recovery → Run.** Both links are MATCHED with p ≈ 0.99 and wide margins. Switch on "Show model decisions" in the
   replay to see the white links.
5. **Reveal Provider Reference.**
   - Colours now show the provider's identities, and the red dotted paths are what was hidden; the model agrees.
   - Read the warning: this is agreement with the provider, and this match was in the training data.
6. **Model Explanation.**
   - Distance, gap, velocity continuity, heading and extrapolation error for in-0 → out-0.
   - The path-attribution chart shows which measurements moved this forest's score; it is descriptive, not causal.
7. **Example B** (3 or more players, P1 20:14.5).
   - Run: every tracklet is UNCERTAIN because another assignment is almost as good (margin < 1.0).
   - Reveal: the best guesses were in fact the provider's continuations. This is the system abstaining instead of risking a wrong
     link.
8. **Research Results.**
   - On 12 unseen locked-test matches, evaluated once: AUC 0.838; the relaxed point links 40.3% of tracklets with 8.8% disagreeing
     with the provider; the conservative point links 27.0% with 4.6%.
   - Simple rules link more but disagree on 34–47% of their links.
   - These are historical numbers, not from the encounter on screen.

## 13. Files

**Created:**
- `src/football_fatigue/m11_dashboard/{__init__,__main__,data,inference,pitch,research,app}.py`
- `tests/test_m11_dashboard.py`
- this report

**Modified:** `pyproject.toml` (new optional extra `m11dash`).

**Not modified:** all `matching/`, `encounters/`, vision, demo, multicam and playback modules; all data under `data/`.

## 14. Step 8.1: advisor demo and UX improvements (2026-11-08)

Presentation only. The frozen model and its checksums (`629cca77…`, `6fa1df5c…`, re-hashed after the work), the 51-feature schema,
the Platt calibration, the assignment, the operating points (relaxed 0.76 / 1.0, conservative 0.87 / 1.0), the RAW files and every
saved result are unchanged. Nothing was trained, tuned, recomputed or re-selected, and validation and test data are still not served.
Sections 1–13 describe Step 8; this section lists what changed.

### 14.1 Changes

| file | change |
|---|---|
| `presentation.py` (new) | Display-only logic, Streamlit-free: tracklet colours, demo-example rules, decision explanations, candidate chart, 5 feature categories. No function takes a provider reference. |
| `app.py` | New **★ Advisor Demo** tab; reworked views (below); callbacks take explicit arguments; widget keys so examples can be loaded. |
| `pitch.py` | Fixed per-tracklet colours, role symbols, labels, static legend, withheld label on the pitch, labelled model links, close / wide / full zoom, still frames. |
| `research.py` | Also returns the saved outcome counts (`incoming`, `matched`, `correct`, `wrong`) and pair counts behind the rates. Rates are unchanged. |
| `tests/test_m11_dashboard.py` | 10 new tests (14.4). |

**Advisor Demo:** six guided steps: Select → Watch (before / withheld / after stills) → Run model → Associations → Reveal → Understand.
- Examples A, B and C come from `presentation.demo_examples`. It applies the fixed Step 8 rules (section 9) in time order and reads only
  the public keys `tier`, `n_incoming`, `ambiguity_s`, `period`, `start_time_s` and `instance_id`.
- The demo match is 3829 (the documented walkthrough match) when present, otherwise the first development match.
- On 3829 the rules give `453e55e4`, `0c080ccb` and `82a4f751`, the same encounters as section 9.
- The page states that these are development, in-sample examples that explain behaviour, not evidence of generalisation, and that
  general claims rest on the locked test. The final step repeats this, as does the comparison after reveal.

**Decision explanations** (`presentation.explain_decision`) follow the frozen rule (`ml.assign` / `ml.decide`) and use only
`infer_instance` outputs:
1. **Reachability:** no reachable candidate → UNMATCHED.
2. **Global assignment:** threshold-free, so it is the same at both points. A tracklet is assigned exactly when its decision record
   has a margin.
   - An unassigned tracklet is UNCERTAIN, either because its best candidate has p < 0.5 or because that candidate is linked to
     another incoming tracklet (one-to-one).
   - An assigned link may differ from the tracklet's highest-p candidate; this is stated, with the tracklet that took that candidate.
3. **Operating point:** threshold and margin checks, each reported as passed or failed.

The explanation derives the decision these checks imply and flags any difference from the record.
- **Match 3829:** on all 823 encounters (1,765 incoming tracklets) the implied decision equals the frozen record at both points, with
  0 inconsistencies. The assignment (candidate, margin) is identical at both points in all 823.
- **Wording:** p is called a calibrated score of agreement with provider labels, not identity confidence. Explanation p values have
  3 decimals, so 0.998 is not shown as 1.00.

**Prediction screen:**
- a global-assignment table (assigned link, p, margin, highest-p candidate, relaxed / conservative decision, in-sample pair);
- a card per incoming tracklet with candidate bars (assigned bar outlined, unreachable bars grey), lines at the 0.5 assignment floor
  and at 0.76 / 0.87, and the per-point reasons;
- the probability matrix and the decision rule in expanders.

**Tactical replay:**
- **Tracklets:** each has one fixed colour. Incoming tracklets are cool-coloured circles and outgoing ones warm-coloured diamonds; an
  incoming and an outgoing tracklet never share a colour, because the order is random and a shared colour would suggest a pairing.
- **Context:** other players are small grey markers with no team colour; team counts of the others would hint at the participants'
  teams.
- **Withheld interval:** labelled "WITHHELD · hidden from the model" on the pitch, and a timeline bar with the phase durations.
- **Model links:** labelled with p and drawn by kind: solid = MATCHED, dashed = assigned but UNCERTAIN, dotted = best candidate not
  assigned.
- **Legend:** built from static key markers. The first browser check showed it empty at frame 0, because Plotly omits traces with no
  data.
- **Opening frame:** the first frame with an incoming tracklet, not an empty board.
- **Zoom:** close-up (tracklet bounding box + 3 m), wide, or full pitch.

**Model explanation:**
- **Categories:** a display-only mapping of all 51 features into five categories, each feature exactly once:
  - position and distance (5);
  - speed and acceleration (11);
  - direction and turning (9);
  - gap and trajectory continuity (11, including the 4 observation-quality features);
  - assignment ambiguity (15).
- **Per pair:** one headline metric per category, the Saabas contribution summed per category, and each category's real feature
  values and contributions.
- **The decision:** the decision for the pair's incoming tracklet at both points, with its reasons.
- **Unchanged:** the Step 8 `FEATURE_GROUPS` and the attribution itself.

**Match explorer:**
- short metrics (period, clock, tracklets, gap, closest), so nothing is truncated;
- tier badges with explanations of all three tiers;
- technical metadata and the provenance legend moved into expanders.

**Research results:**
- **Splits:** badges explain each split (development = fitted, in-sample; validation = once; locked test = once, the only source of
  general claims).
- **Locked-test cards:** each rate shows its numerator and denominator, for example relaxed coverage 2,240 of 5,555 tracklets and
  false association 198 of 2,240 links.
- **Explanations:** ROC-AUC (11,869 pairs: 5,555 same-player, 6,314 different-player), coverage, false association and the trade-off.
- **Chart:** coverage against false association for the saved locked-test methods (points only, no new curve).
- **Table:** one row per split and point, with counts. All values are read from the saved files.

### 14.2 Reference hiding

`MD.provider_reference` and `D.hidden_interval` are called only inside `if revealed` branches. All pre-reveal views use the instance,
the public context and the model output. New guards:
- **Board figure:** before reveal, no trace name, label, hover text or legend entry contains a reference word or an identity, at
  every zoom, both animated and still.
- **Whole page:** for examples A, B and C, before and after running the model:
  - `provider_reference` and `hidden_interval` are wrapped to count calls, and the count must stay 0;
  - every element proto the page sends is scanned, including Plotly specs, tooltips, help text and table contents. Neither the
    encounter's identity labels nor "Provider identity", "Provider continuation", "REFERENCE ONLY", "crossing = ", "team relation",
    "player_id" or "jersey" may appear;
  - **positive control:** after Reveal the calls happen and the labels appear; after Hide the labels disappear again.

### 14.3 Browser verification

- **How:** no browser tool was connected, so I drove the system's Microsoft Edge in headless mode through the Chrome DevTools
  Protocol (script outside the repository, `websockets` from the existing environment, nothing installed).
- **Walkthrough:** against the real server on `127.0.0.1`: load example A → run → every tab → reveal → replay → example B → run.
  I inspected the screenshots.
- **Defects this found and fixed:**
  - an empty legend at frame 0;
  - a clipped "assigned" label;
  - a "+" rendered as a bullet;
  - misleading ↑ arrows on descriptive metric deltas;
  - a too-wide results table;
  - the legend overlapping the slider.
- **Browser leak scan:** the rendered page text before reveal had no `home #` / `away #` label and no "crossing ="; after reveal it
  did. Streamlit draws tables on a canvas, so this text scan does not see table cells; the `AppTest` test above reads table contents
  directly.

### 14.4 Tests

| test | checks |
|---|---|
| `test_explanation_follows_assignment_threshold_and_margin` | on the real `ml` rule with synthetic probabilities: assigned ≠ best candidate, high p taken by another tracklet, p < 0.5, threshold-only and margin-only failures, unreachable |
| `test_explanations_agree_with_frozen_decisions_on_real_encounters` | 150 real instances: explanation = frozen record at both points; assignment identical at both points |
| `test_tracklet_colours_never_suggest_a_pairing` | incoming and outgoing palettes disjoint |
| `test_demo_examples_use_public_metadata_only`, `test_demo_examples_are_the_documented_development_encounters` | rule-based selection; hidden keys cannot change it; A/B/C = section 9 |
| `test_feature_categories_cover_the_frozen_schema_exactly_once` | 51 features, 5 categories, no duplicates |
| `test_candidate_figure_marks_assignment_and_frozen_thresholds` | assigned bar marked; lines at 0.5 / 0.76 / 0.87 |
| `test_board_hides_reference_until_reveal_and_labels_links` | see 14.2; link labels; withheld label; still frames |
| `test_research_counts_reproduce_the_saved_rates` | counts / denominators equal the saved rates to 1e-4 (test relaxed: 2,240 / 5,555, 198 / 2,240) |
| `test_no_reference_leak_before_reveal_in_any_view` | see 14.2 |

**Results:** `tests/test_m11_dashboard.py` 29 passed, the 19 Step 8 tests unchanged. Full suite: 760 passed, 6 skipped, 1 warning (the
existing ByteTrack deprecation), against 750 before.

### 14.5 Limitations

- The demo is still in-sample: examples A, B and C are training-match encounters. They explain behaviour; only the locked test
  supports general claims.
- Feature categories are a presentation grouping. The Saabas attribution remains descriptive, not causal.
- Label placement on the pitch can still overlap when tracklets end within a metre of each other.
- Browser checks were headless screenshots, not an interactive session. Animation playback timing in a real browser was not timed.
- No video frames and no ball positions; none exist in this data.
