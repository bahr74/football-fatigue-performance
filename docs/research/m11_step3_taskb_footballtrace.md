# M11 Step 3: task B (withheld interval) and FOOTBALLTrace E0 reproduction (match 3812 pilot)

Status: 2026-11-06. Inputs: the step-1 catalogue `m11-encounters-v2/3812` and the step-2 benchmark design
(`m11_step2_matching_design.md`). Outputs: `data/processed/m11-matching-v3/3812/`. v1 and v2 are unchanged.

**Not done in this step:** other matches, validation or test data, model training, synthetic swaps.

All rates in this report are **agreement with the provider identity reference**, not verified identity accuracy.

## 1. Implementation

| File | Change |
|---|---|
| `src/football_fatigue/matching/footballtrace.py` (new) | Savitzky-Golay filter (numpy, scipy `mode="interp"` behaviour); Algorithm 1 turn cosines; `literal_breakpoint` (rules `min_angle` / `min_cosine`); `literal_correct` (tail exchange); `protocol_l` and `summarize_protocol_l` (E0-Literal, evaluation-side); `adapted` (E0-Adapted, a matcher) |
| `matching/benchmark.py` | `withhold_interval` (task B); schema `m11-matching-v3`; new outputs `instances_b`, `predictions_b`, `footballtrace_literal`; `results_b` and `footballtrace_literal` in the summary |
| `matching/baselines.py` | `e0_adapted` registered; documents what `chain_during` does on task B |
| `matching/evaluation.py` | New breakdowns: gap duration (a priori bins <1.5, 1.5–2.5, 2.5–4, ≥4 s) and sharp turn. Methods missing from a prediction set are skipped |
| `tests/test_matching.py` | 27 tests in total (10 new) |

Eligibility, events, tiers and task A are **unchanged**. Verified against v2:
- task A instances identical (apart from the schema string);
- references identical;
- all 3,380 task A predictions of the four step-2 baselines identical.

## 2. Task B: withheld-interval association

- **Construction:** the task A instances with the same incoming and outgoing tracklets, anonymous IDs and seeded order, but with
  `during` emptied. The matcher knows only the interval bounds (the tracklets' end and start times).
- **Source data:** the RAW cache is untouched.
- **Answers:** references stay evaluation-only.
- **Not an occlusion:** this is a controlled information-loss experiment on real trajectories, not a confirmed occlusion.

**Statistics** (845 instances: 214 high confidence, 628 lower, 3 natural suspicious):

- **Withheld interval:** p10 1.10 s, median 2.04 s, p90 4.57 s, max 5.97 s. Medians by tier: high 1.84 s, lower 2.10 s,
  suspicious 4.30 s.
- **Gap bins:**

  | Tier | <1.5 s | 1.5–2.5 s | 2.5–4 s | ≥4 s |
  |---|---|---|---|---|
  | high confidence | 72 | 77 | 46 | 19 |
  | lower confidence | 157 | 210 | 155 | 106 |
  | natural suspicious | 1 | 0 | 0 | 2 |

- **Checks run:** no task B instance has an interval observation; its tracklets are identical to task A's; boundaries hold
  (incoming t < 0 < interval end < outgoing t); same key whitelist; no forbidden words; methods read only the written instance
  file.

## 3. FOOTBALLTrace: what the paper specifies (read from `data/raw/122268.pdf`)

| Element | Paper | Notes |
|---|---|---|
| Frame rate | 25 fps, panoramic video, DAN tracker (δb = 15 frames) | Tracker-level, not reproducible here |
| Coordinates | Homography to a pitch top view; Table 1 shows X 264–269, Y 201–206; Figure 7 axes 180–350 | Top-view image units. **No scale to metres is given** |
| Candidates | 50-frame (2 s) sliding windows, 50% overlap; distance ≤ τ for ≥ 50% of the window; τ = 4 "for reproducibility" | τ is in top-view units, so τ = 4 ≠ 4 m. Step 1 implements the rule in seconds and metres |
| Confirmation | An operator confirms each candidate on video (211 → 40 events) | No operator and no real swap labels here |
| Analysis window | 5 s around the occlusion event | |
| Pre-processing | SG filter (w, n), then resampling with step s; w = 7, n = 2, s = 7 by majority vote (grid w ∈ {5, 7, 9}, n ∈ {2, 3, 4}, s ∈ {5, 7, 9, 10}) | In samples |
| Breakpoint | v1 = P(i) − P(i−1), v2 = P(i) − P(i+1); "Angle" = dot/(norms) (the cosine); take the index of the minimal value | **Contradiction:** the minimal cosine is the straightest point, while the text and Figure 8 call it the sharp turn (minimal angle = maximal cosine). Both are implemented |
| Correction | Split at each breakpoint; T_before(occluded) + T_after(occluding) and vice versa | **Always** exchanges the tails; the swap decision belongs to the operator |
| Input | ID-labelled tracker trajectories of the occluded and occluding players | Pairwise only |
| Result | 87.5% of 80 confirmed disrupted trajectories corrected | Only on events confirmed as disrupted |

**Missing details:**
- the τ unit scale;
- edge handling of the SG filter (the scipy default is assumed);
- the resampling anchor (index 0 is assumed);
- how the 5 s window is placed (centred on the interval midpoint is assumed);
- what counts as "successfully corrected";
- the treatment of zero-length vectors and of more than two players.

## 4. E0-Literal and E0-Adapted

**E0-Literal (protocol L, evaluation-side).** Algorithm 1 exactly:
- applied to the two *provider-labelled* RAW trajectories of each 2-player instance;
- a 5 s window centred on the interval;
- RAW samples at 29.97 fps, 15 Hz repeats included;
- w = 7, n = 2, s = 7 in samples (s = 0.23 s here, against 0.28 s in the paper);
- both breakpoint rules.

**Incompatible with tasks A and B.** E0-Literal needs labelled trajectories through the encounter, which is exactly the
information both tasks remove, and it has no decision of its own. It is therefore reported as a separate protocol and never
compared in the matcher tables.

**E0-Adapted (`e0_adapted`, task B compatible).** It is FOOTBALLTrace-*inspired*, not the published method.

| Difference from the paper | Justification |
|---|---|
| Update instants are linearly resampled to the paper's 25 Hz clock before SG (a temporary stage; RAW untouched) | FIFA's 15 Hz updates inside a 29.97 fps file; this restores w = 7, s = 7 to the paper's time scale (0.28 s) |
| Each tracklet is smoothed on its own; resampling is anchored at the junction end | Task B has no observations across the gap; anchoring keeps the real end and start points |
| The breakpoint search becomes a cost: the sharpness (1 + cos)/2 of the sharpest turn at the junction of incoming ⊕ outgoing | The paper's premise is that a wrong connection appears as a sharp turn. The assignment, not the always-swap, takes the decision |
| Exact one-to-one assignment with abstention; speed gate 12 m/s; UNCERTAIN when the alternative is within 0.1; neutral cost 0.5 when no turn is defined | No operator. All values set a priori and not tuned |
| N players allowed | The paper handles pairs only |

The SG and cosine stages are scale-invariant, so the coordinate unit matters only for the candidate threshold τ. The benchmark
keeps the user-specified 2 m.

## 5. Results

**Task A vs task B.** `nearest`, `constant_velocity`, `hungarian_cv` and `e0_adapted` never read interval points, so their task A
and task B results are identical. That is verified, and it confirms the information-access design. Only `chain_during` changes.

**Task B, high-confidence tier (214 instances, 439 incoming tracklets):**

| Method | Correct | Wrong | Uncertain | Unmatched | Coverage | False assoc. | Complete |
|---|---|---|---|---|---|---|---|
| nearest | 0.428 | 0.344 | 0.150 | 0.077 | 0.772 | 0.445 | 0.042 |
| constant_velocity | 0.437 | 0.339 | 0.184 | 0.039 | 0.777 | 0.437 | 0.107 |
| hungarian_cv | 0.364 | 0.255 | 0.326 | 0.055 | 0.620 | 0.412 | 0.341 |
| chain_during (no interval) | 0.485 | 0.162 | 0.353 | 0.000 | 0.647 | 0.250 | 0.477 |
| e0_adapted | 0.264 | 0.196 | 0.540 | 0.000 | 0.460 | 0.426 | 0.243 |

For reference, `chain_during` on task A agrees 0.934 with 0 wrong.

**Task B, lower-confidence tier (628 instances):**

| Method | Correct | Wrong | Uncertain | Unmatched | False assoc. | Complete |
|---|---|---|---|---|---|---|
| hungarian_cv | 0.392 | 0.189 | 0.283 | 0.136 | 0.326 | 0.347 |
| chain_during | 0.399 | 0.268 | 0.331 | 0.002 | 0.401 | 0.385 |
| e0_adapted | 0.298 | 0.235 | 0.464 | 0.003 | 0.441 | 0.276 |

**Natural suspicious (3 instances, 7 incoming):** too few to interpret.
- `e0_adapted`: 2 correct, 0 wrong, 5 uncertain.
- `hungarian_cv`: 2 correct, 2 wrong.

**By gap duration, task B, all tiers (complete-assignment accuracy):**

| Method | <1.5 s | 1.5–2.5 s | 2.5–4 s | ≥4 s |
|---|---|---|---|---|
| hungarian_cv | 0.448 | 0.362 | 0.254 | 0.268 |
| chain_during | 0.530 | 0.415 | 0.353 | 0.260 |
| e0_adapted | 0.361 | 0.279 | 0.199 | 0.181 |

The false-association rate rises with the gap for every method. For `e0_adapted` it goes from 0.35 to 0.53.

**By motion difficulty, task B, complete-assignment accuracy:**

| Method | 2 players | 3+ players | Standard | Challenging | Crossing | No crossing | Sharp turn | No sharp turn |
|---|---|---|---|---|---|---|---|---|
| hungarian_cv | 0.373 | 0.095 | 0.538 | 0.301 | 0.384 | 0.319 | 0.215 | 0.497 |
| chain_during | 0.438 | 0.143 | 0.734 | 0.333 | 0.064 | 0.647 | 0.415 | 0.400 |
| e0_adapted | 0.286 | 0.095 | 0.373 | 0.243 | 0.298 | 0.246 | 0.209 | 0.336 |

`chain_during` on task B collapses on crossings (0.064, false association 0.80): with no interval points it links each end to
the nearest start. Sharp turns hurt `e0_adapted` and `hungarian_cv` most.

**E0-Literal (protocol L, 761 two-player pairs):**
- It corrected (exchanged tails in) 761 of 761 pairs under both rules, so reference agreement is **0.0** by construction.
- **Breakpoint localisation:** the share of breakpoints within the interval (± one step) is 0.560 for `min_angle` and 0.565 for
  `min_cosine`. A uniformly random point in the 5 s window gives **0.555**, so **localisation is at chance**.
- The two rules never pick the same breakpoint, but each lands in the interval equally often.

## 6. Failure analysis

**e0_adapted:**
- **`0aeeec7489b2a0a1`** (home #5 / away #21, sharp turn, 1.70 s): the true continuation turns sharply (sharpness 0.998), so the
  method prefers the swapped link.
- **`827f7c331debf4f3`**: same pattern, true sharpness 0.97.
- **`ec35b55bba2f41cd`** (3.57 s interval): the turn is ambiguous over a long gap.
- **Successes** (`237395eb2a7ffaba`, `9b256730c59d77df`): near-straight continuations with sharpness ≈ 0.001.
- **The core problem:** the paper's premise that a sharp turn means a wrong connection is violated in duels, where real players
  turn sharply. In the high tier, 129 of 214 instances have a sharp turn of over 45°.

**hungarian_cv:** sharp-turn cases fail (`55fc6c316e5dd014`, `20ef96ee62350d46`), as in step 2.

**Complementarity:** in the 205 high-tier 2-player instances, e0_adapted is completely correct in 19 where Hungarian is not,
and Hungarian in 40 where e0_adapted is not. Both are correct in 32 and neither in 114.

## 7. Tests

`tests/test_matching.py` has 27 tests. The 10 new ones cover:
- withholding exactly the interval with tracklets and quality unchanged, and boundaries;
- crossing on task B (constant velocity, Hungarian and e0_adapted correct);
- long and short withheld intervals with the gap bins;
- irregular cadence (updates every 2, 3 and 5 frames);
- three players (one-to-one methods have no conflicts) and a missing outgoing tracklet;
- task A unchanged by task B (instances, predictions, manifest);
- SG exactness on polynomials, including edges;
- literal breakpoint rules (`min_cosine` misses a V-turn that `min_angle` finds) and tail exchange;
- protocol L always disagreeing when it corrects;
- e0_adapted having no reference access and being deterministic;
- no mutation of the cache, mask, encounters or task A instances.

FIFA regression and encounters: see the final report (the full command is in the step-1 handoff, section 8).

## 8. Limitations

- **Reference:** provider labels, not ground truth. No real swap labels exist, so E0-Literal's swap-correction ability (87.5% in
  the paper) cannot be measured here. Only its breakpoint localisation can, and it is at chance.
- **Units:** τ (top-view units) has no conversion to metres, so the candidate stage is reproduced in rule only, not in scale.
- **E0-Adapted is a reinterpretation,** with untuned margin and neutral cost. Its weak result must not be read as the paper's
  result.
- **Task B selection:** it inherits the step-2 biases (selection for continuity, de-duplication by time order, the ESTIMATED
  filter). The intervals are proximity episodes, not verified occlusions.
- **Scope:** one match.

## 9. GO / NO-GO

- **NO-GO** for FOOTBALLTrace (literal or adapted) as an identity-recovery method on FIFA RAW data. The literal method has no
  decision without an operator, its breakpoint is at chance, and the adapted version is the weakest one-to-one method on task B.
- **GO** for task B as the primary benchmark. It separates methods, degrades with gap length, and needs no swap labels.
- **Recommended next experiment** (needs approval):
  1. Build step-1 catalogues for 2–3 more development matches, checking the 3816 known swap separately as a natural case.
  2. On task B, test a motion model fitted to football kinematics (non-learned first: physical reachability sets or a
     velocity-and-acceleration prior estimated on development matches only), combined with a turn-aware cost.
  3. Report the same tables, with gap-binned false-association rates as the primary safety metric.
