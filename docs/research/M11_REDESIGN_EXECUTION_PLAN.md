# M11 redesign: execution plan

Date: 2026-11-13. Companion to [M11_VIDEO_GROUNDED_OCCLUSION_RESEARCH.md](M11_VIDEO_GROUNDED_OCCLUSION_RESEARCH.md) (here "R§n" means
section n of that document). This plan only proposes work. **Nothing below has been implemented**, and every phase after Phase 0 needs
approval.

Standing rules, from PROJECT_RULES and earlier M11 practice:

- RAW data and existing versioned outputs are never modified. New outputs go to new versioned directories with manifests and code hashes.
- Development matches only, until a pre-registered one-time evaluation.
- The FIFA test partition stays closed.
- No package installs or large downloads without approval.
- Each phase ends with a report and a stop.

---

## Overview

```text
Phase 0  Evidence audit ────────────────────────────────┐ (mostly done)
Phase 1a SoccerTrack identity-failure study (E1) ──┐    │ CPU, no labels
Phase 1b Wadi Degla W2 annotation pilot ───────────┤    │ human time
Phase 2  Real occlusion / failure patterns ◄───────┘    │
Phase 3  Transferability to FIFA ◄─── Phase 2 + Phase 0 │
Phase 4  Detector experiments on FIFA S40 ◄─── Phase 3 (risk model), can start B/F/C2 volumes in parallel
Phase 5  Candidate scoring + dataset construction ◄─── Phase 4
Phase 6  Identity-association baselines ◄─── Phase 5
Phase 7  Advanced ML/DL ◄─── Phase 6 gate
Phase 8  Fair evaluation, ablations, error analysis ◄─── Phases 6–7
Phase 9  Method selection + one-time validation ◄─── Phase 8 + advisor
```

**Parallel tracks:**

- Phase 1a ∥ Phase 1b.
- Phase 4 volume curves for B, F and C2 ∥ Phases 1–3. These are descriptive and independent of the risk model.
- Phase 5 quality-flag sidecar ∥ Phases 1–3.
- Phase 6 M0 Kalman baseline ∥ Phase 4, on a provisional label-symmetric population. Final numbers wait for Phase 5.

**Strictly sequential:**

- the risk model (Phase 3) before risk-scored datasets (Phase 5);
- the common evaluation population (Phase 5) before any model comparison (Phases 6–8);
- the frozen configuration before any validation run (Phase 9).

---

## Phase 0: evidence and repository audit

| Item | Content |
|---|---|
| Objective | Fix the factual base; close cheap unknowns |
| Done (this session) | Repository and asset audit; D1–D4 diagnostics; literature check (R§0.3, R§1–3, R§11) |
| Remaining steps | 1. Get the PFF definition of `visibility` / `ESTIMATED` / `confidence`. The spec PDF text was not machine-extractable; read it manually or ask PFF/Gradient. 2. Promote the D1–D3 scratch scripts into a versioned `data/processed/m11-redesign-diagnostics-v1/` with a manifest (optional). 3. Run D1 on all 40 development matches (about 15 min on 3 CPUs) to confirm cadence and burst statistics are not match-specific |
| Deliverables | Short addendum to the research doc; diagnostics folder |
| Compute | < 1 h CPU |
| Validation checks | Cadence identical across matches; burst enrichment ≥ 3× in ≥ 90 % of matches |
| Failure condition | Burst enrichment absent in many matches → method F is not general; downgrade to a stratum |
| Progress criterion | Addendum written |
| Advisor decisions | None |

---

## Phase 1a: SoccerTrack identity-failure risk study (Experiment E1, **first experiment**)

| Item | Content |
|---|---|
| Objective | Measure which **camera-agnostic** features predict **real tracker identity failures** on video. Measure how the 0.25 m rule, crossing + τ and image overlap perform as detectors of those failures |
| Inputs | SoccerTrack MOT GT for 117093, 118575, 118576 (calibrated; 117092 pixel-only analysis); cached tracker outputs from M6/M6.3; the M6.3-D event machinery (reference-ID change events); `fit_hybrid(continuous=True)` |
| Steps | 1. Build GT pitch tracks (foot point → metres) at 25 fps; also resample to 15 Hz to mimic FIFA. 2. Generate all reference pair-episodes with d_min ≤ 3 m (unordered pairs, label-symmetric). 3. Label each pair-episode from our tracker against the reference: `switch` (an output track moves from one reference ID to the other within the episode ± 1 s), `fragment` (either reference ID gets a new output track), `clean`. Use the existing evaluation-time IoU matching, flagged as possibly noisy at crowded contacts [R M6.3-D caveat]. 4. Features. Class A: d_min, time below 0.5/1/2 m, relative speed at t_min, crossing angle, heading change ± 1 s, same team, crowding within 2 m, curvature. Class B (reference only): occlusion fraction, perpendicular / line-of-sight components. 5. Models: logistic regression and `HistGradientBoostingClassifier`, leave-one-match-out over the 3 matches, with Platt calibration inside folds. 6. Compare ROC/PR against single rules: d_min ≤ {0.25, 0.5, 1} m; crossing with \|Δt\| ≤ {h, 0.2} s; occlusion ≥ 0.5 |
| Deliverables | `data/processed/vision_eval/m11e1-identity-failure-risk-v1/` (README, results JSON, figures); the risk-function specification |
| Compute | CPU only, < 1 h; no new detections |
| Validation checks | Label-free sanity: failure rate per match; class balance; per-match AUC; bootstrap CIs over episodes clustered by pair |
| Failure conditions | Class-A AUC < 0.65 in LOMO, or no gain over d_min alone (ΔAUC < 0.03 with CI including 0) → camera-agnostic transfer is not supported. Then FIFA candidates stay proximity + F-bursts, with no learned risk |
| Progress criterion | Class-A AUC ≥ 0.70 and better than d_min alone → Phase 3 |
| Advisor decisions | Approve SoccerTrack as the video calibration domain |

**Pre-registered hypotheses** (they follow from R§3.2 / R§4):

- H1: most identity failures occur at d_min > 0.25 m.
- H2: relative speed and crowding add information beyond d_min.
- H3: image overlap (class B) predicts failures better than d_min. This quantifies what FIFA cannot see.

---

## Phase 1b: Wadi Degla W2 annotation pilot (parallel)

| Item | Content |
|---|---|
| Objective | Human-verified visual occlusion and identity outcomes on a second, lower-quality, two-view domain |
| Inputs | `wadi_degla_fixed_same_match.mp4` + `wadi_degla_moving.mp4` (τ = −0.125 s); M10.8 lab review tool; ByteTrack outputs |
| Steps | 1. Freeze the annotation protocol (R§2.3) and stratified random clip sampling (seeded) **before viewing**. 2. Inspection round: 20 clips × 10 s; measure the event rate; refine the label definitions (one revision allowed, documented). 3. Pilot: 60 clips, two annotators, adjudication. 4. Compute κ with bootstrap CIs; the unresolvable share; the tracker outcome rates |
| Deliverables | Label files (append-only, provenance), protocol v1, reliability report |
| Compute | Tracking on 60 × 10 s clips: about 1 h CPU (M10.8: 42–63 s per window). **About 25 person-hours** of annotation |
| Validation checks | κ(event_type) ≥ 0.7; κ(identity_outcome) reported; unresolvable share reported |
| Failure conditions | κ < 0.6 after one revision, or unresolvable > 50 % for same-team identity → Wadi Degla is used for event detection only, not identity |
| Progress criterion | ≥ 150 labelled episodes with κ ≥ 0.7 |
| Advisor decisions | Annotation budget; who annotates; whether measured pitch dimensions can be obtained (the metric scale for W2) |

---

## Phase 2: empirical study of real occlusion and failure patterns

| Item | Content |
|---|---|
| Objective | Characterise real occlusions and identity failures: duration, geometry, relative velocity, team, crowding, partial vs full |
| Inputs | Phase 1a outputs (and 1b when available) |
| Steps | Distributions of visibility-loss duration (the gap model for synthetic FIFA gaps); failure rate vs (d_min, v_rel, angle, team, crowding); partial vs full occlusion; camera-motion effects (Wadi Degla moving camera) |
| Deliverables | Empirical gap-length and failure-condition distributions (JSON), feeding Phases 3 and 5 |
| Compute | Minutes |
| Validation | Agreement between SoccerTrack and Wadi Degla in direction of effects (not magnitudes) |
| Failure condition | Effects contradict each other between domains → report domain dependence; do not transfer magnitudes |
| Progress | Distributions frozen |
| Advisor decisions | None |

---

## Phase 3: transferability analysis to FIFA

| Item | Content |
|---|---|
| Objective | Decide whether the E1 risk function may score FIFA candidates |
| Inputs | E1 model; FIFA S40 proximity v2 + F-bursts; optional BePro XML |
| Steps | 1. **Relative-distance calibration error.** Compare pairwise distances from our mapping with BePro pitch positions (BePro XML, about 216–318 MB per match, **download approval**). Without XML: synthetic perturbation of keypoints (weaker evidence). 2. **Covariate shift.** Domain classifier AUC (SoccerTrack vs FIFA) on class-A features; per-feature histograms; restrict to the overlapping support. 3. **Proxy validation on FIFA.** The risk score must rank *F-bursts with a neighbour* above matched VISIBLE controls. These are independent FIFA-internal events that the risk model never saw. 4. Optional broadcast-camera prior as a sensitivity variable |
| Deliverables | Transfer verdict (GO / CONDITIONAL / NO-GO) with evidence |
| Compute | < 1 h CPU (+ download) |
| Validation | Proxy AUC (bursts vs controls) ≥ 0.70 |
| Failure | Domain AUC > 0.9 with little support overlap, or proxy AUC < 0.6 → do not transfer; use proximity + F-burst tiers only |
| Progress | GO / CONDITIONAL with a documented scope ("ranking only") |
| Advisor decisions | Accept "risk-scored proximity" as the FIFA candidate definition (in place of "occlusion") |

---

## Phase 4: detector experiments on FIFA (S40 development)

| Item | Content |
|---|---|
| Objective | Compare candidate detectors as **detectors**: volume, diversity, quality and, where possible, recall of independent events |
| Detectors and ranges | **A** crossing + τ: τ ∈ {h, 2h, 3h, 0.2, 0.5} s, z ∈ {1, 2, 3}. **B** proximity: R ∈ {0.10, 0.25, 0.50, 0.75, 1.0, 1.5} m (0.75–1.5 are new runs of the existing code under a new version tag). **C2** velocity-adaptive: R0 ∈ {0.25, 0.5}, k ∈ {0, 0.5, 1}, z ∈ {0, 2}. **C3** probabilistic: σ_EST ∈ {1, 2, 4} × measured jitter, P ∈ {0.1, 0.3, 0.5}. **D** risk deciles (if Phase 3 passes). **F** bursts: max length {0.5, 1, 2} s, R_b {0.5, 1, 2} m |
| Steps | 1. Volume curves per detector and parameter, per match. 2. Tier composition (T1–T4). 3. Cluster structure and effective n (repeated pairs within 10 s). 4. Overlap matrix between detectors (Jaccard on pair-time). 5. **Recall of F-bursts-with-neighbour** by each geometric detector: an independent FIFA-internal event set. 6. Compute benchmarking |
| Deliverables | `m11-detector-comparison-v1/` report and tables |
| Compute | B at 1.5 m on S40: about 2–4 h CPU (v2 at 0.25 m was feasible); others minutes to hours |
| Validation | Nesting checks (thresholds nest exactly); determinism; manifest hashes |
| Failure | None scientific; a detector is just characterised |
| Progress | Recommend a generator (expected: B at about 1 m) and a scoring scheme |
| Advisor decisions | Approve the generator radius and the tier scheme |

**Why downstream accuracy does not rank detectors:** each detector produces a population of different intrinsic difficulty.
Higher accuracy on an easier population (for example crossings with the "assume crossed" prior) says nothing about detection quality
[R a1/a3]. Detectors are compared on detection metrics. Models are compared on one common population.

---

## Phase 5: candidate-quality scoring and dataset construction

| Item | Content |
|---|---|
| Objective | Build a large, honest, label-symmetric dataset with tiers, risk and clusters |
| Steps | 1. Quality-flag sidecar per sample: ESTIMATED run length and position; short-burst flag; confidence; jitter; clone; distance to nearest other player. Step 1 is unchanged. 2. Candidate set = generator (Phase 4) ∪ F-bursts; attach risk (if Phase 3 GO), tier and cluster ID. 3. Exclude only clones (bit-identical) and substitution windows; keep everything else, flagged. 4. **Common evaluation population (CEP).** A seeded, detector-independent stratified sample over (tier × risk decile × K × same/opposite team). The CEP is the only population used to compare models. 5. **Synthetic gaps.** Length from the Phase 2 empirical distribution (fallback: FIFA short-burst distribution, median 0.27 s), plus a stress stratum up to 3 s. Placement randomised around t*. Anonymised and shuffled tracklets; 50/50 keep/swap by construction. 6. F-bursts give **natural-gap** samples (no synthetic removal; VISIBLE on both sides) |
| Deliverables | `m11-candidates-v4/` + `m11-recovery-dataset-v2/` with manifests; data card (counts, n_eff, tiers, clusters) |
| Compute | A few CPU hours |
| Validation checks | (a) Permuting player IDs leaves features unchanged (unit test). (b) Within every stratum P(keep) = 0.5 ± sampling error. (c) No cluster split across folds. (d) Counts reported as events and as n_eff |
| Failure | Any leakage test fails → stop and fix |
| Progress | Data card approved |
| Advisor decisions | Accept synthetic gaps with provider labels as the training signal; the gap-length distribution source |

---

## Phase 6: identity-association baselines

| Item | Content |
|---|---|
| Objective | Strong, interpretable reference performance on the CEP |
| Models | **M0:** nearest end-to-start, constant velocity, constant acceleration, **Kalman-CV likelihood + Hungarian assignment** (first baseline). **M0+:** quality-dependent measurement noise (VISIBLE / ESTIMATED), fitted on development only. **M1:** `HistGradientBoostingClassifier` on symmetric pairwise features, including M0's negative log-likelihood (hybrid M5) |
| Protocol | Nested leave-one-match-out over 40 development matches (inner loop for hyperparameters). Grouped by cluster. Permutation-level evaluation with abstention |
| Metrics | Accuracy, swap rate, coverage at FA ≤ 5 % / 10 %, ECE, Brier. By tier, K, gap length, team, risk decile; separately on synthetic-gap and F-burst (natural-gap) samples. Reference baselines: always keep / always swap / random |
| Compute | M0 seconds; M1 minutes per fold on CPU |
| Validation | Learning curves (25 / 50 / 100 % of training matches) for under- and overfitting; train-vs-validation gap |
| Failure | M1 does not beat M0+ on T1 → kinematic learning adds nothing; M0+ is the method |
| Progress | Frozen baseline numbers on the CEP |
| Advisor decisions | None (development only) |

---

## Phase 7: advanced ML/DL (conditional)

| Item | Content |
|---|---|
| Entry gate | Phase 6 done. The data card shows n_eff ≥ about 10⁴ T1 + T2 events |
| M2 | Siamese GRU / 1D-TCN tracklet encoder. Input (T × 6): relative x, y, Δt, velocity, quality mask, gap mask; 2.5 s each side. Pairwise compatibility head plus a permutation softmax for K = 2. Loss: BCE or permutation cross-entropy. Masking for variable length. Early stopping on an inner fold |
| M3 / M4 | Set transformer / GNN over all tracklets in the window, **only** for K > 2 groups, and only if M2 beats M1 by the pre-registered margin |
| Pre-registered margin | M2 must beat M1 by ≥ 1.5 points of permutation accuracy on T1 **and** have no worse ECE, with the grouped-bootstrap CI of the difference excluding 0 |
| Compute | GTX 1650 Ti (4 GB): M2 about 10–60 min per fold. Nested LOMO with 40 matches is expensive: use 8 outer folds of 5 matches. M3/M4 1–3 h per fold |
| Overfitting controls | Grouped folds, weight decay, dropout, small capacity first, learning curves, no early stopping on outer folds |
| Failure | Margin not met → M1 (or M0+) stays. Record a negative result |

---

## Phase 8: fair evaluation, ablations and error analysis

| Item | Content |
|---|---|
| Steps | 1. All models on the identical CEP and folds. 2. Feature-family ablations (distance, velocity, curvature, quality, team, crowding, Kalman NLL). 3. Synthetic gaps vs natural gaps (F-bursts): report the gap between them as a realism check. 4. **Cross-domain check on video:** apply the frozen FIFA-trained model to SoccerTrack reference pitch tracks with the gaps observed from real tracker failures. Its accuracy against the BePro reference is the **only real-identity evaluation** available. 5. Error analysis: inspect 50 random failures and 50 random confident successes per tier |
| Metrics | As in Phase 6; for video, IDF1/HOTA via TrackEval only when re-linking real tracks |
| Failure | Large synthetic-vs-natural or FIFA-vs-video gap → claims are restricted to the synthetic regime |
| Deliverables | Evaluation report with ablation tables and error taxonomy |

---

## Phase 9: final method selection and one-time evaluation

| Item | Content |
|---|---|
| Steps | 1. Freeze: detector, tiers, model, abstention threshold, metrics, success criteria (pre-registration document). 2. Advisor approval of the held-out policy. Recommended option (b): the evidence base is development nested LOMO; the FIFA validation partition (already seen for other tasks) is used **once**, with disclosure. 3. Run once. Report every number by tier with CIs. 4. The FIFA test partition stays closed |
| Success criteria (proposal, to be fixed before the run) | T1 permutation accuracy ≥ M0+ + 3 points; coverage ≥ 0.40 at FA ≤ 5 %; ECE ≤ 0.05; no tier with FA > 10 % at the chosen threshold |
| Advisor decisions | Held-out policy; success criteria; final wording ("risk-scored proximity encounters", not "occlusions") |

---

## Thresholds to investigate (summary)

| Quantity | Range | Justification |
|---|---|---|
| Temporal resolution floor | h = 0.0667 s | D1: 15 Hz updates for all samples |
| Arrival tolerance (A) | h … 0.5 s, z ∈ {1, 2, 3} | Only as a descriptive stratum (redundant with distance) |
| Proximity generator R | 0.10 … 1.5 m | D2: severe image occlusions have median ground distance 1.7–2.9 m, while P(occlusion) is high only below 0.5 m. High recall needs R ≥ about 1 m, with risk scoring |
| Velocity term k | 0, 0.5, 1 (× ‖v_rel‖·h) | Sampling: 1–3 cm at typical relative speeds |
| σ_EST multiplier | 1, 2, 4 × measured jitter | D1: ESTIMATED jitter 8–10 cm median; true error unknown |
| Burst length (F) | ≤ 0.5 / 1 / 2 s | D3: median 0.27 s |
| Burst neighbour radius R_b | 0.5 / 1 / 2 m | D3: enrichment 8× (< 0.5 m), 5× (< 1 m), 3× (< 2 m) |
| Synthetic gap length | empirical (Phase 2), stress up to 3 s | SoccerTrack occlusion median 0.16–0.2 s, p95 about 1.2 s; FIFA bursts median 0.27 s |
| Context window | ± 2.5 s (unchanged) | Advisor framework; sensitivity ± 1.5 s |

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| E1 too small (3 matches × 4 min) | Report per-match results; treat as calibration evidence, not as a final model; extend to more SoccerTrack dev clips only with approval (new detections need about 3 h GPU per match) |
| BePro reference errors | Flag crowded-contact reference flips (M6.3-D caveat); sensitivity analysis excluding events < 25 observations |
| Provider labels are not truth | Primary metrics on T1 and natural gaps; real-identity evaluation only on video |
| Over-promising transfer | Transfer is claimed as ranking only, with proxy validation (Phase 3) |
| Annotator burden | Pilot capped at 60 clips; active learning only after the random sample |

---

## Decision Brief

**What we know [V]:**

- FIFA RAW: frames at exactly 29.97 fps, no gaps; positions update every 2 frames (15 Hz) for VISIBLE and ESTIMATED alike; mm quantisation.
- VISIBLE tracks are smooth: jitter is about 1 cm.
- ESTIMATED tracks jitter about 8× more. They cover more than half of all samples, mostly in multi-second runs, consistent with off-screen players.
- On real camera geometry (SoccerTrack), **97–98 % of severe image occlusions occur at ground distances above 0.25 m** (median 1.7–2.9 m), because occlusion is a thin strip along the camera ray (perpendicular component median 0.15–0.19 m).
- Short provider ESTIMATED bursts (median 0.27 s) occur within 1 m of another player about 5× more often than controls. This is FIFA's best internal proxy of a real visual ambiguity.
- SoccerTrack-v2 in the repository has video, calibration, a persistent identity reference and 514 catalogued real tracker identity changes.
- Wadi Degla has none of these labels and 13–29 px players.

**What we do not know [U]:**

- The absolute accuracy of FIFA positions, especially ESTIMATED ones.
- PFF's definition of `ESTIMATED`.
- The relative-distance accuracy of our SoccerTrack calibration.
- Whether camera-agnostic risk transfers from SoccerTrack to FIFA.
- Human-verified identity outcomes on any video.
- The advisor's acceptance of the redefinition.

**Recommended direction:**

- Separate three concepts: visual occlusion (video), proximity encounter (ground plane), identity ambiguity (joint).
- Calibrate "risky encounter" on **SoccerTrack video with real tracker failures**.
- Transfer only a camera-agnostic risk *ranking* to FIFA.
- Generate FIFA candidates broadly (proximity at about 1 m plus provider visual-loss bursts), tier and score them instead of deleting.
- Formulate recovery as local bipartite assignment with abstention.
- Start with a Kalman + Hungarian baseline, then GBM, and DL only behind a pre-registered margin.

**What can be implemented now (with approval):**

- Phase 0 remainder (D1 on all 40 development matches).
- **E1** (Phase 1a; CPU, existing data).
- The Wadi Degla protocol freeze and the 20-clip inspection round.
- Proximity volume curves at R up to 1.5 m.
- The quality-flag sidecar.

**What should not be implemented yet:**

- Any model training on a new population.
- DL models.
- Re-cleaning Step 1.
- Any use of the FIFA validation or test partitions.
- Any claim that a FIFA event is an occlusion.
- Use of the provider-smoothed stream.
- Large downloads (BePro XML, TeamTrack) without approval.

**The exact first experiment, E1:**

- *Data:* SoccerTrack-v2 development clips 117093, 118575, 118576, CPU only, existing caches.
- *Population:* every reference player pair with a ground distance ≤ 3 m.
- *Label:* whether our tracker swapped or fragmented their identities, against the BePro-derived reference.
- *Model:* camera-agnostic features (d_min, relative speed, crossing angle, duration, team, crowding), logistic regression and gradient boosting, leave-one-match-out.
- *Comparison:* against the 0.25 m rule, crossing + τ, and image overlap.
- *Pass:* class-A AUC ≥ 0.70 and better than d_min alone → proceed to transfer (Phase 3).
- *Fail:* FIFA candidates remain proximity plus burst tiers without learned risk.
- *Runtime:* under 1 hour.
- *Data use:* no new data and no labels.
