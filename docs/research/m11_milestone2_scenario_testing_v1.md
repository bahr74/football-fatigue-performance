# M11 Milestone 2: new-scenario testing and robustness (`m11-scenarios-v2`)

**Status:** implemented, run and tested.
- No training was run, and no model, threshold, routing policy, dataset or earlier output was changed.
- No held-out match, official FIFA validation/test data or reserved external sequence was read.
- **Data:** the 32 train/CV matches only.
- **Nature of the results:** these are synthetic interruptions of real provider trajectories, scored by fold models that never saw the source match. They are development stress cases. They are **not** independent matches and **not** real occlusion events.
- **Optimism:** the 5 % / 2 % thresholds and the choice of GRU C were made on out-of-fold results from the same 32 matches, so the baseline figures carry that optimism. Every recommendation below that is derived from these results needs confirmation outside this data.

## 1. What was built

| File | Purpose |
|---|---|
| `src/football_fatigue/m11_benchmark/scenarios.py` | Seeded encounter selection, scenario variants (one factor at a time), label revalidation, perturbations, movement families, exact `regenerate()` |
| `src/football_fatigue/m11_benchmark/scenario_eval.py` | Batched scoring with each match's unseen-match fold models; routed decision of `m11-routing-v1`; outcome accounting; write-once outputs; metrics and examples |
| `tests/test_m11_scenarios.py` | 23 tests |

The `m11_infer` modules are reused unchanged: adapter, routing (`route` / `decide`), fold selection and checkpoint loading.

### Population

- Per match: 60 encounters, drawn with a seed derived from `sha256(suite | select | match)`. They are chosen from the **independent** Stage 3 `g0p80` rows, which are Stage 2 candidates accepted at 0.8 s.
- Total: 32 × 60 = **1,920 encounters**.
- Every variant is built per encounter: 1 baseline + 28 stressed = **55,680 scenarios**.
- Limitation: all Stage 2 candidates come within 0.5 m of each other, so **separated pairs (> 0.5 m) are not in the population** and cannot be tested from this source.

### Variants

Baseline: 0.8 s gap, centred on the closest approach, clean provider positions, all observations, 2.0 s context.

| Factor | Levels |
|---|---|
| gap | 0.2, 0.4, 0.6, 1.0, 1.2, 1.6, 2.0, 2.4 s (centred) |
| shift (gap centre vs closest approach) | ±0.1, ±0.3, ±0.5 s; training used up to ±0.3 s |
| noise | i.i.d. Gaussian on every observed x and y: 1, 2, 5, 10, 25, 50 cm |
| dropout | random 10 / 30 / 50 %; one 0.5 s burst per tracklet; 7.5 Hz (every other frame, keeping the one next to the gap) |
| context | 1.5, 1.0, 0.5 s of observations on each side |

The requested 0.8 s gap is the baseline, and the centred / shifted comparison is the shift factor.

### Validity: labels are never fabricated

- **gap and shift variants:** the unchanged Stage 2 `label_quality` procedure is re-run at the new gap and centre. A variant that is not ACCEPTED is recorded as **invalid** with its reasons and is never shown to a model.
- **Every built variant:** checked with `verify_label` against the raw provider positions, and the gap boundaries are re-checked.
- **noise / dropout / context:** these only perturb or remove observations of the accepted baseline geometry.
- **Model input:** only observations strictly outside `[g0, g1]` are passed; there is no interpolation across the gap and no extrapolation.
- **Tracklet order:** drawn from the scenario id, independently of Stage 3.

**Invalid:** 393 of 55,680 (0.7 %), all from label re-validation. The main reason is `continuous_reference`: at longer gaps or larger shifts, one chain no longer has a clean reference.

| Variant | Invalid |
|---|---|
| gap 1.0 | 10 |
| gap 1.2 | 31 |
| gap 1.6 | 54 |
| gap 2.0 | 82 |
| gap 2.4 | 102 |
| shift ±0.1 to ±0.5 | 5–35 per level |

### Determinism

- `scenario_id = sha256(suite | candidate | factor | level)[:16]`.
- All random draws of a scenario (order, dropout, noise) use a generator seeded from its id.
- Encounter selection does not use any model output.

### Routing

- The noise declared to the router equals the noise added by the scenario (0 for clean data).
- Dropout and context are **not** routing inputs in `m11-routing-v1`, so the router cannot see them.
- A test checks that the batched routing equals `Predictor.predict_input`.

### Outcome categories

| Category | Meaning |
|---|---|
| `accepted_correct` / `accepted_wrong` | The routed model decided. |
| `abstain_low_confidence` | The routed model's confidence was below its threshold. |
| `abstain_unsupported` | The input is outside the policy: gap, declared noise, or fewer than 3 grid points. Raw probabilities, where present, are **diagnostic only**. |
| `invalid` | Not a valid scenario; never scored. |

## 2. Commands

```powershell
.\.venv\Scripts\python.exe -m football_fatigue.m11_benchmark.scenario_eval run         # 32 matches, about 9 min on CPU, resumable, write-once
.\.venv\Scripts\python.exe -m football_fatigue.m11_benchmark.scenario_eval summarize   # metrics + examples + manifest (write-once)
.\.venv\Scripts\python.exe -m pytest tests/test_m11_scenarios.py tests/test_m11_infer.py
```

To rebuild one scenario exactly: `scenarios.regenerate(match, candidate_id, factor, level)`.

## 3. Saved outputs (`data/processed/m11-scenarios-v2/`, 301 MB, git-ignored)

| File | Contents |
|---|---|
| `run_config.json` | Suite, factors, policy, thresholds, split SHA, bundles-manifest SHA, code SHAs |
| `matches/<m>/scenarios.parquet` | One row per scenario. Ids, match, fold, all parameters, valid/invalid reason, label source, truth `y`, `g0`/`g1`/`tc`, tracklet order, grid points, `p_gru`, `p_hgb`, routed support/model/decision/confidence/threshold, outcome, abstention reason, families, checkpoint SHAs, policy |
| `matches/<m>/arrays.npz` | Per valid scenario: perturbed observations (concatenated, with offsets), grid tracklets in pitch coordinates with masks, and the clean hidden path of each player over `[g0, g1]` (15 Hz, for the reveal view) |
| `matches/<m>/done.json` | Counts, runtime, file SHAs |
| `summary/metrics_v1.json` | All tables below, reliability bins, and the rule-based example sets |
| `manifest.json` | Matches, per-match records, metrics SHA, `heldout_used: false`, `official_data_used: false` |

`data/processed/m11-scenarios-v1/` is an **aborted first attempt**, kept unchanged and not used: 11 matches before a crash, described in §8.

## 4. Results by factor

**Notation:**
- BA = balanced accuracy; LL = log loss.
- CI = 95 % match-grouped bootstrap (1,000 resamples).
- Δ = GRU − HGB paired BA, with its match-grouped CI.
- "Routed" = the operational decision at the 5 % operating point: coverage, and realised error among accepted predictions.
- *Italic rows* lie outside the supported range of the policy. Their raw BA is **diagnostic only and not validated**. The routed decision there is always ABSTAIN.

### 4.1 Gap (centred)

| Gap s | Valid | GRU BA [CI] | GRU LL | HGB BA [CI] | HGB LL | Δ [CI] | Routed model | Coverage | Realised error |
|---|---|---|---|---|---|---|---|---|---|
| *0.2* | 1920 | *0.622 [0.592, 0.654]* | *0.84* | *0.944 [0.933, 0.954]* | *0.16* | *−0.322* | unsupported | 0 | — |
| 0.4 | 1920 | 0.837 [0.819, 0.853] | 0.41 | 0.921 [0.909, 0.932] | 0.20 | −0.084 [−0.105, −0.063] | HGB | 0.790 | 0.020 |
| 0.6 | 1919 | 0.893 [0.880, 0.907] | 0.28 | 0.886 [0.875, 0.896] | 0.25 | +0.007 [−0.007, 0.023] | HGB | 0.739 | 0.032 |
| **0.8 (baseline)** | 1920 | **0.921 [0.910, 0.932]** | 0.19 | 0.850 [0.837, 0.863] | 0.31 | +0.071 [0.058, 0.085] | **GRU** | **0.927** | **0.050** |
| 1.0 | 1910 | 0.807 [0.790, 0.822] | 0.44 | 0.848 [0.834, 0.861] | 0.33 | −0.041 [−0.056, −0.025] | HGB | 0.702 | 0.061 |
| 1.2 | 1889 | 0.779 [0.762, 0.796] | 0.60 | 0.839 [0.822, 0.853] | 0.37 | −0.060 [−0.076, −0.044] | HGB | 0.710 | 0.090 |
| 1.6 | 1866 | 0.729 [0.714, 0.743] | 0.85 | 0.798 [0.781, 0.815] | 0.49 | −0.069 [−0.086, −0.053] | HGB | 0.720 | **0.130** |
| *2.0* | 1838 | *0.695* | *1.00* | *0.754* | *0.59* | *−0.059* | unsupported | 0 | — |
| *2.4* | 1818 | *0.735* | *0.98* | *0.746* | *0.65* | *−0.011* | unsupported | 0 | — |

The same trend holds on the 1,815 encounters that are valid at **every** gap. For example, at 1.6 s GRU 0.727 and HGB 0.798; at 0.2 s GRU 0.624 and HGB 0.946. So the gap effect is not caused by the change in valid population.

**The GRU is reliable only at the gap it was trained on.** All of its training samples have `gap_s = 0.8`, and it fails badly at 0.2 s, even though a short gap is the easier problem (HGB reaches 0.944 there). HGB degrades smoothly with gap length.

### 4.2 Gap shift (0.8 s gap; centre moved from the closest approach)

| Shift s | GRU BA [CI] | HGB BA [CI] | Δ [CI] | Routed coverage | Realised error |
|---|---|---|---|---|---|
| −0.5 | 0.855 [0.840, 0.867] | 0.751 [0.729, 0.776] | +0.103 [0.081, 0.125] | 0.892 | 0.103 |
| −0.3 | 0.905 [0.892, 0.918] | 0.893 [0.877, 0.908] | +0.012 | 0.909 | 0.064 |
| −0.1 | 0.904 [0.892, 0.915] | 0.872 [0.860, 0.884] | +0.032 | 0.904 | 0.069 |
| 0 | 0.921 | 0.850 | +0.071 | 0.927 | 0.050 |
| +0.1 | 0.897 [0.887, 0.908] | 0.878 [0.865, 0.889] | +0.019 | 0.916 | 0.071 |
| +0.3 | 0.917 [0.905, 0.930] | 0.866 [0.853, 0.878] | +0.051 | 0.912 | 0.058 |
| +0.5 | 0.902 [0.882, 0.920] | 0.713 [0.688, 0.736] | +0.189 [0.162, 0.213] | 0.923 | 0.067 |

- GRU C, trained with shifts, is robust across ±0.5 s.
- HGB collapses at ±0.5 s.
- Every shifted level stays on GRU (gap 0.8). The realised error is 5.8–10.3 %, above the nominal 5 %, because the threshold was set on the centred / independent pool.

### 4.3 Positional noise (0.8 s gap)

| Noise | GRU BA [CI] | GRU LL / ECE | HGB BA [CI] | HGB LL / ECE | Routed model | Coverage | Realised error |
|---|---|---|---|---|---|---|---|
| clean | 0.921 | 0.19 / 0.015 | 0.850 | 0.31 / 0.030 | GRU | 0.927 | 0.050 |
| 1 cm | 0.901 [0.888, 0.913] | 0.23 / 0.026 | 0.856 | 0.31 / 0.023 | GRU | 0.912 | 0.065 |
| 2 cm | 0.868 [0.854, 0.881] | 0.35 / 0.050 | 0.839 | 0.32 / 0.023 | GRU | 0.901 | **0.102** |
| 5 cm | 0.773 [0.755, 0.789] | 0.78 / 0.125 | 0.826 [0.813, 0.839] | 0.35 / 0.030 | HGB | 0.673 | 0.070 |
| *10 cm* | *0.685* | *1.42 / 0.211* | *0.800* | *0.41 / 0.037* | unsupported | 0 | — |
| *25 cm* | *0.621* | *2.26 / 0.287* | *0.731* | *0.53 / 0.079* | unsupported | 0 | — |
| *50 cm* | *0.566* | *3.53 / 0.374* | *0.657* | *0.70 / 0.148* | unsupported | 0 | — |

- The GRU is sensitive to noise: at 2 cm the GRU-routed error already doubles, and the GRU stays over-confident.
- HGB stays calibrated up to 10 cm.
- At **video-level noise (25–50 cm), neither model is usable**: diagnostic HGB BA is 0.66–0.73.

### 4.4 Dropout and irregular sampling (0.8 s gap, clean)

| Dropout | GRU BA [CI] | HGB BA [CI] | Δ [CI] | Routed coverage | Realised error |
|---|---|---|---|---|---|
| none | 0.921 | 0.850 | +0.071 | 0.927 | 0.050 |
| random 10 % | 0.887 [0.873, 0.903] | 0.858 | +0.029 [0.012, 0.047] | 0.910 | 0.082 |
| random 30 % | 0.841 [0.824, 0.857] | 0.846 | −0.006 [−0.024, 0.016] | 0.882 | 0.124 |
| random 50 % | 0.801 [0.786, 0.815] | 0.836 | −0.035 [−0.051, −0.019] | 0.882 | **0.160** |
| 0.5 s burst | 0.906 [0.894, 0.916] | 0.844 | +0.061 | 0.914 | 0.064 |
| 7.5 Hz | 0.909 [0.900, 0.919] | 0.855 | +0.054 | 0.907 | 0.065 |

- Random dropout hurts the GRU. The adapter's linear interpolation between irregular observations differs from the regular grid it was trained on.
- A regular 7.5 Hz rate and a single burst are tolerated.
- The router has no dropout input, so these all stay on GRU, with realised error up to 16 %.
- 5 scenarios had a tracklet with fewer than 3 grid points and were correctly routed as `too_few_valid_points`.

### 4.5 Context length (0.8 s gap, clean)

| Context | GRU BA [CI] | HGB BA [CI] | Δ [CI] | Routed coverage | Realised error |
|---|---|---|---|---|---|
| 2.0 s | 0.921 | 0.850 | +0.071 | 0.927 | 0.050 |
| 1.5 s | 0.908 [0.896, 0.919] | 0.851 | +0.057 | 0.915 | 0.062 |
| 1.0 s | 0.886 [0.871, 0.899] | 0.851 | +0.034 | 0.865 | 0.067 |
| 0.5 s | 0.819 [0.802, 0.835] | 0.839 | −0.019 [−0.038, −0.001] | 0.883 | **0.153** |

### 4.6 Routed totals (all 55,287 valid scenarios, 5 % operating point)

| Outcome | Count |
|---|---|
| accepted correct | 34,630 |
| accepted wrong | 2,999 |
| abstain (low confidence) | 6,317 |
| abstain (unsupported) | 11,341 |

- **By reason for unsupported:** gap 5,576; noise 5,760; too few points 5.
- **Overall:** coverage 0.681, realised error 0.080.
- **Interpretation:** this pool is dominated by deliberate stress, so it is not an estimate for any real deployment.

## 5. Movement families (descriptive; computed from clean provider chains at the baseline geometry, not model inputs)

Each cell shows GRU BA / HGB BA / routed realised error.

| Family (n at baseline) | Baseline 0.8 s | Gap 1.6 s (HGB routed) | Noise 5 cm (HGB routed) |
|---|---|---|---|
| local crossing (1227) | 0.941 / 0.892 / 0.034 | 0.773 / 0.871 / 0.075 | 0.812 / 0.870 / 0.043 |
| no crossing (334) | 0.852 / 0.738 / **0.105** | 0.587 / 0.600 / **0.329** | 0.694 / 0.698 / 0.158 |
| window-only crossing (359) | 0.920 / 0.812 / 0.054 | 0.713 / 0.733 / 0.161 | 0.711 / 0.798 / 0.089 |
| same direction (812) | 0.910 / 0.819 / 0.057 | 0.692 / 0.759 / 0.153 | 0.757 / 0.797 / 0.085 |
| crossing angle (466) | 0.914 / 0.878 / 0.053 | 0.772 / 0.820 / 0.138 | 0.820 / 0.876 / 0.059 |
| opposite direction (98) | 0.925 / 0.892 / 0.053 | 0.766 / 0.851 / 0.140 | 0.844 / 0.910 / 0.038 |
| sharp turn > 90° (725) | 0.908 / 0.823 / 0.059 | 0.684 / 0.726 / **0.215** | 0.760 / 0.808 / 0.079 |
| straight (505) | 0.949 / 0.899 / 0.023 | 0.788 / 0.895 / 0.036 | 0.796 / 0.862 / 0.045 |
| speed change > 50 % (1146) | 0.908 / 0.813 / 0.060 | 0.709 / 0.764 / 0.167 | 0.778 / 0.801 / 0.086 |
| hard parallel (66) | 0.892 / 0.897 / 0.071 | 0.614 / 0.818 / 0.036 | 0.685 / 0.842 / 0.071 |
| same team (598) | 0.935 / 0.869 / 0.038 | 0.752 / 0.833 / 0.096 | 0.783 / 0.838 / 0.074 |
| opposite teams (1322) | 0.915 / 0.841 / 0.056 | 0.718 / 0.782 / 0.145 | 0.768 / 0.821 / 0.068 |
| both fully VISIBLE in gap (181) | 0.818 / 0.795 / **0.147** | 0.683 / 0.701 / 0.221 | 0.696 / 0.807 / 0.103 |
| both fully ESTIMATED in gap (981) | 0.940 / 0.855 / 0.031 | 0.722 / 0.823 / 0.096 | 0.743 / 0.818 / 0.065 |

What the family breakdown shows:
- **The hardest cases are encounters without a clear crossing, sharp turns and large speed changes.**
- **Encounters where both players are VISIBLE inside the gap are clearly harder than ESTIMATED ones** (baseline error 14.7 % vs 3.1 %). This matches the Stage 4C texture finding: performance is probably inflated by provider interpolation inside ESTIMATED stretches. The VISIBLE subset is the closer proxy for real tracking data.
- Per the agreed stance, this is reported but does not block.
- Small cells (hard parallel 66, opposite direction 98) have wide uncertainty; no CIs are computed per family.

## 6. Examples (reproducible, rule-based)

- **Where they are stored:** `summary/metrics_v1.json → examples`.
- **Selection rule:** within each category, scenarios are taken in scenario-id order (a hash), not by result. The one exception is `highest_confidence_accepted_errors`, which is ordered by confidence on purpose.
- **Category sizes:**

| Category | Scenarios |
|---|---|
| GRU correct (primary) | 26,922 |
| GRU wrong (primary) | 2,456 |
| HGB correct (fallback) | 7,708 |
| HGB wrong (fallback) | 543 |
| Models disagree | 10,956 |
| Successful abstentions (the routed model would have been wrong) | 2,454 |
| Abstentions on a correct prediction | 3,863 |
| Long gap ≥ 1.6 s | 5,522 |
| Noise ≥ 10 cm | 5,760 |
| Hard parallel at baseline | 66 |

- **Most confident accepted errors:** the GRU outputs p ≈ 0 or 1 on dropout and 0.5 s-context inputs, sometimes while HGB gives the correct side. For example, scenario `926f838cdb9c3cbc` (3812, random 30 %): p_gru 1.000 against truth MATCHED. Its label passed the raw verification.

## 7. Tests

`tests/test_m11_scenarios.py` has 23 tests:

| Area | What is tested |
|---|---|
| Ids and variants | Deterministic and distinct; one factor at a time |
| Context cut | Respects the gap and the context length |
| Dropout (5 modes) | Seeded; only removes observations; never alters values or adds times; burst length; the observation next to the gap is kept |
| Noise | Seeded; keeps timestamps; correct σ |
| Families | Correct on synthetic geometry |
| Routing | The batched router equals `Predictor.predict_input` in 7 policy regimes; outcome accounting sums |
| Empty tracklet | Unsupported, never filled in |
| Integration | Held-out / unknown matches refused; selection deterministic and independent; regeneration exact; no observation inside `[g0, g1]`; `y` consistent with order; hidden-path length; baseline predictions reproduce the stored Stage 3 OOF under reordering (exact antisymmetry); saved results consistent (no held-out match, fold mapping, outcome invariants, array alignment) |

Checks that ran:
- `pytest tests/test_m11_scenarios.py tests/test_m11_infer.py`: **61 passed.**
- The full-suite result is given in the delivery message.

## 8. Failures and issues

- **First run crashed** (`m11-scenarios-v1`, 11 matches). A dropout perturbation removed every observation of one tracklet, and the adapter correctly refused the empty input.
  - **Fix:** `scenario_eval.model_input` now returns an all-masked input, which is routed as unsupported (`too_few_valid_points`).
  - **Output:** a new version, `m11-scenarios-v2`. The v1 folder is kept unchanged and is unused; it can be deleted on request.
- **Router blind spots:** `m11-routing-v1` cannot see dropout, irregular sampling or short context, and it relies on a declared noise level. Under those conditions the primary GRU route accepts at 6–16 % realised error, against a 5 % target.
- **Thresholds do not transfer:** at the HGB fallback gaps 1.0–1.6 s the realised error at its 5 % threshold is 6–13 %. The thresholds were calibrated at 0.8 s.
- **Code hash recorded in `run_config.json`:** after the run, one docstring line in `scenario_eval.py` was corrected (it named a file that is not written). The `code_sha256` stored for that file is therefore the pre-edit hash. A resumed run would refuse because the hash differs; to rerun, use a new output version.

## 9. Recommendations

These are not implemented and need approval. They are derived from development stress data, so they need independent confirmation before any claim.

1. **Routing v2 (proposal).**
   - Keep GRU primary only for clean, regular, full-context input at 0.75–0.85 s. This needs measured observation density / context inputs, not just declarations.
   - Use HGB at its 2 % threshold (0.9029) for:
     - gap 0.4–1.2 s: realised error 0.6–5.1 %, coverage 0.53–0.62;
     - noise up to 5 cm;
     - irregular or dropout input.
   - Mark gap 1.6 s as low-confidence: HGB at 0.9029 gives an error of 9.2 % even so.
   - Consider admitting gaps of 0.2–0.4 s to HGB (diagnostic 0.944 BA, 0.3 % error at 0.9029). This range is currently unsupported, so admitting it would be a policy change that needs approval.
2. **Do not use either model on video-calibrated tracks** (noise ≈ 0.4–1 m) without a noise-robust model. Both are near chance at 50 cm.
3. **For Milestone 4 (if approved):** train the GRU with a variable gap (`gap_s` as a real input), with noise and dropout augmentation, and with a larger epoch budget. The GRU's narrow gap and noise tolerance is the main weakness found here.
4. **Report VISIBLE-only figures alongside the overall ones** in the dashboard and in the final evaluation.

## 10. Readiness for Milestone 3 (dashboard)

The data needed for the dashboard is ready:
- per-scenario rows with raw coordinates, observation timestamps, grid tracklets, the hidden clean paths, truth, both probabilities, the routed decision, the reason and provenance;
- `regenerate()` for exact reconstruction;
- rule-based example sets.

The existing `m11_dashboard/pitch.py` board can render these directly. Milestone 3 has not been started.
