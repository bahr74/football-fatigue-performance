# M11: FIFA scientific method decision (candidate definition, thresholds, cleaning, sample design)

Status: **research decision, awaiting approval**. Written 2026-11-13. Nothing in this document was implemented in the repository. All new measurements were made read-only on **development matches only**, using scratch scripts outside the repository.

Evidence labels:

| Label | Meaning |
|---|---|
| **[V]** | Verified in this project, by a computation or file inspection named here |
| **[R]** | Reported in an earlier project report (cited) |
| **[I]** | Inference or estimate, not measured |
| **[U]** | Unresolved or unknown |

Related documents: `M11_VIDEO_GROUNDED_OCCLUSION_RESEARCH.md`, `M11_REDESIGN_EXECUTION_PLAN.md`, E1 results (`data/processed/vision_eval/m11e1-identity-failure-risk-v1/README.md`), and the execution roadmap `M11_FIFA_EXECUTION_ROADMAP.md`.

---

## 1. Executive summary

1. **Stop calling FIFA proximity events "occlusions."** FIFA PFF tracking contains positions only. It cannot show a visual occlusion. What it can show is *geometric proximity of two players in pitch coordinates*. The operational unit should therefore be an **identity-ambiguity candidate**: an unordered pair (or group) of players whose continuous-time synchronized minimum distance d_min is ≤ 2 m, with a complete, valid ±2.5 s context. Candidates are tiered by distance band and data quality.
2. **The distance threshold of the generator should be raised from 0.25 m to 2 m.** Bands (≤ 0.5, 0.5–1, 1–2 m) are kept as strata, not as filters.
   - Video evidence for this: on SoccerTrack, 87 % of real tracker failures and 78 % of switches happened at d_min > 0.25 m. The 0.25 m rule recalled only 13 % of failures, while ≤ 2 m recalled 78 % of failures and 86 % of switches **[V, E1]**.
   - Cost: about 9–10× more episodes and about 6–7× more independent pair clusters **[V, matches 3812 and 3816]**.
3. **The geometric-intersection method (path crossing + arrival-time difference Δt) is kept only as a descriptor**, not as a selection rule or a model input.
   - It is redundant in one direction: |Δt| ≤ 0.0667 s implies d_min ≤ 0.25 m in 90 % of cases.
   - It misses the converse: only 9 % of pairs with d_min ≤ 0.25 m have |Δt| ≤ 0.0667 s, and the median |Δt| for those pairs is 0.86 s **[V, 166,819 crossings]**.
   - Crossing-based selection is also **label-asymmetric**. A crossing is defined by identity-labelled paths, so a model trained on crossings can learn "they crossed, therefore swap." This is shown by a 75 % "assume crossed" prior and 50.7 % transfer to non-crossings **[R]**.
4. **Use piecewise-linear interpolation between native samples, not curves.**
   - At the native 15 Hz update rate, splines gain about 2 mm of median accuracy on VISIBLE samples and are *worse* on ESTIMATED samples (53–73 % of all data).
   - They also raise artificial speed peaks.
   - PCHIP and Akima are acceptable as a sensitivity check only **[V]**.
5. **Cleaning.** Reuse the Step 1 verified data and add a *quality-flag sidecar* in a new versioned output. Use a 5-class anomaly taxonomy, and exclude only corruption. Flagged data is kept and stratified; it is not silently deleted.
6. **Sample design.**
   - Input: K = 2 incoming and 2 outgoing tracklets of (x, y, t_rel, mask), event-centred, on a canonical 15 Hz grid, after a symmetric synthetic gap that contains the closest-approach time t*.
   - Target: the keep/swap permutation, randomised 50/50 by a seeded coin.
   - Label: provider identity continuity under artificial removal.
   - Identity, team, visibility and quality metadata are kept beside the sample and never used as input.
7. **Evaluation.** Use nested grouped cross-validation by match on the 40 development matches. Disclose that validation has already been touched twice; the test set is consumed. The only *clean* final test available is human-labelled Wadi Degla video (about 60 random 10 s fixed-camera clips).
8. **Recommendation: CONDITIONAL GO** for Phase 1 of the roadmap only: the 40-match data audit, the quality sidecar, and the 2 m generator into new versioned outputs. Model training waits for a second approval.

---

## 2. Verified state

### 2.1 Data and splits

| Fact | Status |
|---|---|
| 64 FIFA WC 2022 matches. Split in `data/processed/m11-encounters-v1/splits.json`: 40 development, 12 validation, 12 test | [V] |
| The test set was **consumed** by the M11 Step 7 locked test (AUC 0.838) | [R] |
| The validation set was **used twice**: in Step 6 and in the m11-recovery-v1 one-time validation. It is no longer pristine | [R] |
| Step 1 verification (`data/processed/m11-step1-verification-v1/`): 40 matches, 0 frame gaps, 0 duplicate timestamps, cadence ok in every match | [V] |
| ESTIMATED share is 0.535–0.729 per match | [V] |
| Steps: 81,826,714 over 7,432,290 frames (≈ frames × 22 / 2, i.e. a 2-frame cadence everywhere). The 938 invalid steps are all period changes. There are 385,029 jumps, and 220,209 valid steps have speeds of 12–20 m/s | [V] |
| Frame rate is 29.97 fps. Positions update every 2 frames (0.0667 s) in 99.7–99.8 % of intervals. Coordinates are mm-quantised | [V, 3 matches] |
| Noise (second difference at updates) has a median of 1.1–1.3 cm on VISIBLE samples and 8–10 cm on ESTIMATED samples. ESTIMATED samples are always LOW confidence | [V, 3 matches] |
| The provider-smoothed stream is about 0.7 m from raw at lag 0. It is not a lagged copy and is excluded | [V] |

### 2.2 Proximity v2 reproducibility check

The in-memory proximity code (`intersections.proximity`) reproduces the stored v2 event files **exactly** **[V]**:

| Match | 0 m | 0.10 m | 0.25 m | 0.50 m |
|---|---|---|---|---|
| 3812, stored v2 = in-memory | 18 | 327 | 844 | 1,737 |
| 3816, stored v2 = in-memory | 30 | 357 | 908 | 1,855 |

### 2.3 Video evidence (SoccerTrack E1, pre-registered, development clips only) [V]

- **Population:** 1,115 scorable episodes: 217 switch, 308 fragment and 590 clean.
- **Distance rules:**
  - 87 % of failures and 78 % of switches have d_min > 0.25 m.
  - 40 % of switches have d_min > 1 m.
  - At d_min ≤ 2 m, recall is 0.78 for failures and 0.86 for switches.
- **Model:** LR-A (camera-agnostic, position-only features) reaches AUC 0.806 [0.783, 0.830], against 0.737 for −d_min.
  - The gain comes from crowding and duration (fragmentation).
  - For switch vs clean alone, d_min (0.808) is as good as LR-A (0.801).
- **Reference ambiguity:** 84 % of switch labels are reference-ambiguous. The reference itself is not human-verified.

Image occlusion and ground distance are largely decoupled: 97–98 % of severe image occlusions occur at ground distance > 0.25 m (median 1.7–2.9 m) **[V, earlier research doc]**. This is why a 0.25 m ground threshold cannot represent video failure risk.

---

## 3. Methods considered

| Code | Method | Decision | Reason |
|---|---|---|---|
| A | Path crossing + arrival-time difference \|Δt\| ≤ τ (advisor's intersection method) | **Descriptor and reporting stratum only** | See items A-1 to A-3 below |
| A2 | Method A on spline-smoothed paths | **Rejected** | Curves add nothing at 15 Hz (§5) and create artificial crossings and speed peaks. Their crossing count was not measured **[U]** |
| B | Continuous-time synchronized distance d_min ≤ R (unordered pair) | **Generator** | Label-symmetric; exact on piecewise-linear paths; R is justified by video failures |
| C | Hybrid A ∧ B | **Redundant** | A with small τ is almost a subset of B (§4.3) |
| D | Risk scoring on top of B | **Adopted, symmetric features only** | See items D-1 to D-3 below |
| F | Short ESTIMATED bursts (VISIBLE→ESTIMATED→VISIBLE ≤ 1 s) | **Separate stratum** | These bursts are within 1 m of another player in 32–34 % of cases, against 6–7 % for controls **[V]**. They are a provider-side ambiguity marker |

Method A, item by item:

- **A-1.** It is redundant with B in one direction and misses most close approaches (§4.3).
- **A-2.** Crossing is defined on identity-labelled paths, so selecting on it is label-asymmetric.
- **A-3.** It ignores parallel running and group situations.

Method D, item by item:

- **D-1.** The E1 d_min-curve plus crowding score.
- **D-2.** The uncertainty-aware probability P(d ≤ R) for ESTIMATED samples (§4.4).
- **D-3.** The magnitude of a Kalman two-hypothesis margin |Λ|. It is label-symmetric because it uses the absolute value. It is used only for difficulty stratification and hard-example mining.

---

## 4. Mathematics

### 4.1 Synchronized distance on piecewise-linear paths

Players a and b have native samples. On every common interval [t₀, t₁] where both paths are linear:

- Δp(t) = Δp₀ + Δv · (t − t₀), with Δp = p_a − p_b.
- Minimiser: s* = clamp(−Δp₀·Δv / ‖Δv‖², 0, t₁ − t₀).
- Distance: d_min = ‖Δp₀ + Δv·s*‖.

This is what `intersections.exact` / `intersections.proximity` already compute (`pair_rows`, `exact_border`, `qualify`). It is exact for the linear model, has no sampling bias, and is symmetric in (a, b).

### 4.2 Episodes

An episode is a maximal run of consecutive qualifying common intervals for one unordered pair. The closest-approach time is t*. Episodes of the same pair less than 10 s apart form a **pair cluster**. This is the unit for de-duplication and for bootstrap resampling.

### 4.3 Crossing time difference vs distance

For two straight paths that cross at point X, let Δt be the difference in arrival times at X. Then

- d_min = |Δt| · ‖v_a × v_b‖ / ‖v_a − v_b‖ = |Δt| · v_a v_b sin θ / ‖v_a − v_b‖.

A small Δt therefore forces a small d_min. A small d_min, however, is possible with a large Δt when sin θ is small (shallow angles, parallel running) or when the speeds are low.

The data agree on all 166,819 S10 crossings **[V]**. Spearman correlation between |Δt| and d_min is 0.465 overall and 0.538 when both players are VISIBLE.

| Condition | n | P(d_min ≤ 0.25 m) | P(d_min ≤ 0.5 m) | median d_min |
|---|---|---|---|---|
| \|Δt\| ≤ 0.0667 s | 2,310 | 0.897 | 0.990 | 0.08 m |
| \|Δt\| ≤ 0.2 s | 7,085 | 0.683 | 0.874 | 0.14 m |
| \|Δt\| ≤ 0.5 s | 19,032 | 0.449 | 0.633 | — |

| Condition | n | P(\|Δt\| ≤ 0.0667 s) | P(\|Δt\| ≤ 0.2 s) | median \|Δt\| |
|---|---|---|---|---|
| d_min ≤ 0.10 m | 10,641 | 0.12 | 0.26 | 0.64 s |
| d_min ≤ 0.25 m | 23,278 | 0.089 | 0.208 | 0.86 s |
| d_min ≤ 1.0 m | 60,988 | — | 0.114 | 1.26 s |

Conclusion: B contains almost all of A, and A misses about 90 % of B's close approaches. A ∧ B adds nothing that B lacks.

### 4.4 Uncertainty

Model each position as the true position plus noise with standard deviation σ per axis. The two noises are independent per player. The observed difference vector then has per-axis σ_Δ = √(σ_a² + σ_b²), so the observed distance follows a Rice distribution and P(d ≤ R) is a noncentral chi-square CDF. This gives an uncertainty-aware membership probability that can be stored beside the hard d_min.

Indicative noise levels **[I, from §2.1 and §5]**:

| Class | Basis | σ |
|---|---|---|
| VISIBLE | 1–2 cm (hold-out median 4–7 mm at 0.133 s) | ≈ 1 cm |
| ESTIMATED | Hold-out median 4–5 cm, p90 12–15 cm | ≈ 5–10 cm |

ESTIMATED error is model-based, not white noise, so these values are scales, not calibrated variances **[U]**.

What follows from these scales:

- At R ≥ 0.5 m the uncertainty barely changes membership, so the 2 m generator is robust to it.
- At R = 0.10 m an est/est pair has σ_Δ ≈ 7–14 cm, so the 0.10 m class is mostly noise-defined for ESTIMATED data.

---

## 5. Curved vs piecewise-linear trajectories

Test design: hold-out reconstruction on 2,022 chains (match 3812) and 1,333 chains (match 3816) **[V]**. Every 2nd or 3rd native vertex was kept, the others were predicted, and errors were measured on interior points only.

| Hold-out | Class | Linear | Cubic | PCHIP | Akima |
|---|---|---|---|---|---|
| Keep 1 of 2 (0.133 s), 3812 | VISIBLE median / p90 | 6.6 mm / 29 mm | 4.9 / 36 mm | 4.9 / 28 mm | 4.6 / 27 mm |
| Keep 1 of 2, 3812 | ESTIMATED median / p90 | 47.4 mm / 152 mm | 50.0 / 158 mm | 48.6 / 154 mm | 48.2 / 153 mm |
| Keep 1 of 2, 3816 | VISIBLE median | 6.0 mm | 4.0 | 4.1 | 3.9 |
| Keep 1 of 2, 3816 | ESTIMATED median | 40.7 mm | 43.3 | 42.0 | 41.5 |
| Keep 1 of 3 (0.2 s), 3812 | VISIBLE / ESTIMATED median | 12.1 / 54.9 mm | 10.3 / 57.8 | 9.9 / 56.4 | 9.6 / 56.1 |

Overshoot was measured per interval: the curve's excursion outside the bounding box of the interval's two vertices, on an 11-point grid per interval, over 2,001,696 intervals in 3812 **[V]**.

| Method | Intervals > 1 cm | Intervals > 5 cm | > 20 cm | Max |
|---|---|---|---|---|
| Cubic spline | 130,745 | 7,469 (0.37 %; 98 % of these involve ESTIMATED samples) | 16 | 0.36 m |
| Akima | 34,859 | 848 (0.04 %) | 0 | 0.14 m |
| PCHIP | 0 (monotone by construction) | 0 | 0 | 0 |

*Correction:* an earlier chain-level overshoot count in this task (for example "PCHIP 1,118 chains > 5 cm") came from a sampling-grid misalignment and is **retracted**. The per-interval figures above replace it.

Chains whose maximum speed exceeds 12 m/s, 3812:

| Linear segments | Cubic | PCHIP | Akima |
|---|---|---|---|
| 1,345 / 2,022 | 1,546 | 1,620 | 1,643 |

The linear count reflects raw over-speed steps that are already in the data (Step 1: 220,209 such steps). The curves add more speed peaks. The number of artificial crossings created by curves was not measured **[U]**.

**Decision: piecewise-linear.**

- At 15 Hz the curves' gain is at most 2 mm on VISIBLE samples, which is below the 1–2 cm noise.
- On ESTIMATED samples, which are the majority, every curve is worse.
- Splines add derivative spikes that would contaminate kinematic features.
- PCHIP is the only curve that cannot overshoot. It is allowed as a **sensitivity re-run** of the generator, not as the default.

---

## 6. Thresholds and tolerances

| Parameter | Value | Justification |
|---|---|---|
| Generator radius R_gen | **2.0 m** | Recalls 78 % of video failures and 86 % of switches **[V, E1]**. 0.25 m recalls only 13 % of failures |
| Distance bands (strata) | ≤ 0.5 / 0.5–1 / 1–2 m | Each band is reported separately. 0.10 and 0.25 m are kept as historic sub-bands for continuity with v2 |
| Time tolerance in selection | **none** | d_min is computed in continuous time on synchronized intervals, so no time window is needed |
| Δt descriptor bins | ≤ 0.0667 / ≤ 0.2 / ≤ 0.5 s / > 0.5 s / no crossing | 0.0667 s is one native update interval, the resolution limit. 0.2 s is 3 updates. These are metadata for reporting only |
| Context | ±2.5 s around t* (existing `context_half_s`). A ±4 s extended window for the stress gaps | Inherited from v2/v3. It covers the median ESTIMATED run (3.4–6.4 s) only partly. This is disclosed |
| Synthetic gap | Symmetric, contains t*. Main lengths ≤ 1.2 s, stress lengths ≤ 3 s | Empirical gap lengths of the video trackers. The exact distribution is set in roadmap Phase 7 from SoccerTrack development data **[I]** |
| Cluster gap | 10 s per pair | Matches the existing pair-cluster definition |
| Burst stratum F | VISIBLE→ESTIMATED→VISIBLE, ESTIMATED run ≤ 1 s | §3 |

Volumes per match **[V]**, from the in-memory proximity code:

| R | 3812 episodes | 3812 pair clusters | 3816 episodes | 3816 pair clusters | Median duration (3812 / 3816) |
|---|---|---|---|---|---|
| 0.10 m | 327 | 250 | 357 | 269 | 0.05 / 0.06 s |
| 0.25 m | 844 | 551 | 908 | 589 | 0.10 / 0.12 s |
| 0.50 m | 1,737 | 993 | 1,855 | 1,044 | 0.18 / 0.20 s |
| 1.0 m | 4,053 | 2,091 | 4,179 | 2,071 | 0.34 / 0.38 s |
| 2.0 m | 8,697 | 3,702 | 8,114 | 3,696 | 0.48 / 0.71 s |

Extrapolation to the 40 development matches **[I]**:

- At 0.25 m, v2 found 30,438 episodes **[R]**.
- At 2 m the expected scale is about 270k–310k episodes and about 140k–150k pair clusters, before the context-completeness and quality filters.

---

## 7. Uncertainty handling

1. Store the hard d_min, t*, the band, and the per-player visibility class at t* (VISIBLE/VISIBLE, mixed, ESTIMATED/ESTIMATED).
   - At 2 m in 3812: 4,873 est/est, 2,760 obs/obs and 1,064 mixed **[V]**.
   - ESTIMATED/ESTIMATED is the majority, so excluding it would discard more than half of the data.
2. Store P(d ≤ R) under the σ scales of §4.4 as a *descriptor*, not a filter, until σ is calibrated.
3. The label "provider continuity" is itself uncertain in ESTIMATED stretches, because the provider may have bridged a real ambiguity. Report all metrics per visibility class. The headline metric is computed on obs/obs + mixed; est/est is reported separately.
4. Use cluster bootstrap CIs by (match, pair cluster) throughout.

---

## 8. Cleaning and normalisation

**Strategy:** reuse the Step 1 verified raw cache and loaders unchanged. Write a new, versioned **quality-flag sidecar**. Do not change the existing cleaning pipelines or any versioned dataset.

| Class | Content | Action |
|---|---|---|
| 1. Corruption | Clones and bit-identical positions of two slots (`intersections.detect.duplicate_positions`) | **Exclude** from candidates (it would create a false d = 0) |
| 2. Suspected artifacts | Implausible jumps (3812: 14,658 steps, 3816: 8,285 **[V]**), frame-wide shifts, cloned segments | Flag. Exclude only when the flag falls inside the candidate window |
| 3. Uncertain | ESTIMATED / LOW confidence, sub-metre over-speed (speed > 12 m/s caused by small displacements) | Keep. Stratify |
| 4. Plausible extremes | Fast sprints and sharp turns that are physically possible | Keep |
| 5. High quality | VISIBLE, no flags | Keep. This is tier T1 |

The only new flag is an **acceleration flag**: |a| above a physical limit over 3 native samples. The limit is set in roadmap Phase 2 from the empirical distribution, and the rule is pre-registered there. Substitution windows are excluded (the catalogue already carries a substitution flag).

**Quality tiers for candidates:**

| Tier | Definition |
|---|---|
| T1 | VISIBLE/VISIBLE, no flags |
| T2 | Mixed |
| T3 | ESTIMATED/ESTIMATED |
| T4 | Class 2 artifact present (diagnostic only) |

**Normalisation:**

- Translate to the pair midpoint at t*, in metres. Time is relative to t*.
- Randomised reflection augmentation (x → −x, y → −y), because pitch direction is not a feature.
- Rotation only as an ablation.
- Explicit masks.
- Resample to a **canonical 15 Hz grid** (the FIFA native update rate). Wadi Degla video (25 or 30 fps) is resampled to the same grid.

---

## 9. Size vs quality

- **More data is not better if it is mostly noise.** The trade-off is managed with strata, not cuts:
  - All of 0–2 m goes into the pool.
  - Training-subset ablations of equal size (T1 only vs T1+T2 vs all; ≤ 1 m vs ≤ 2 m) measure whether lower tiers help.
- Independence is per pair cluster, not per episode. Going from 0.25 to 2 m multiplies episodes by about 9–10 but clusters by only about 6–7 **[V]**.
- Reported sample sizes always give both episodes and clusters.
- At 0.25 m, 629 of 844 episodes had a complete context **[V, 3812 v2 summary]**. At 2 m the share is unknown **[U]** and will be measured in Phase 3.

---

## 10. Bias and leakage

| Risk | Control |
|---|---|
| Label-asymmetric selection (crossing) | Crossing is never used for selection or as an input. The generator uses unordered pairs |
| Shortcut "close → swap" | Keep/swap is a seeded 50/50 coin on randomised slot order. Check that a label-shuffle control gives AUC ≈ 0.5 |
| Identity leakage | No IDs, team, jersey, names, appearance or physiology as inputs. Metadata is kept in a separate table. The input schema is asserted in tests |
| Team as an input | Not used. Team is used only for stratification (same team vs opposing team) |
| Near-duplicate episodes | All splits and bootstraps are grouped by match. Clusters are reported |
| Validation reuse | Validation has already been used twice. Any further use is disclosed and needs advisor approval |
| Test set | Consumed. It is not a fresh test and must never be reported as one |
| Dashboard in-sample evidence | The existing dashboards are development-only and in-sample. They are labelled as such |
| Provider label noise in ESTIMATED | Metrics per tier. Headline on T1+T2 |
| Domain shift to video | Video-like noise and gap augmentation (Phase 7). The final verdict is on Wadi Degla human labels only |

---

## 11. Sample schema

```
sample_id, schema_version
inputs (model-visible):
  in_tracks  [2, T_in, 4]   (x_m, y_m, t_rel_s, mask)   randomised slot order
  out_tracks [2, T_out, 4]  (x_m, y_m, t_rel_s, mask)   randomised slot order
target:
  perm ∈ {keep, swap}       seeded coin, 50/50
metadata (never model input):
  match, period, slot_a, slot_b, team_relation, gk_involved,
  d_min_m, t_star_s, band, tier, visibility_class_at_tstar, quality_flags,
  crossing_flag, abs_dt_s (descriptor), P_d_le_R (descriptor),
  cluster_id, burst_F, substitution_window,
  gap_start_s, gap_end_s, gap_mode {symmetric, independent}, seed,
  detector_provenance {proximity_v4, F_burst, ...}, source_hashes
```

---

## 12. Strategy

- **Training pool.** One unified pool with detector-provenance tags. When detectors are compared, they are compared *as detectors* on one **common evaluation population (CEP)**.
- **Model ladder.**
  1. **M0:** nearest neighbour, constant velocity, and Kalman + Hungarian, on position and time only.
  2. **M2:** GRU or TCN on raw (x, y, t, mask).
  3. **M1:** GBM (`sklearn` HistGradientBoosting; no install) on derived kinematics. These features need a written justification.
  4. Transformer or GNN only for K > 2, and only after a pre-registered margin over M2.
- **Evaluation.**
  - Nested grouped cross-validation by match on the 40 development matches.
  - FIFA validation is used once more, with disclosure and only after advisor approval.
  - The final evidence is Wadi Degla W2 human labels: fixed camera, about 60 random 10 s clips, about 25 person-hours **[I]**.

---

## 13. Limitations

- Only matches 3812 and 3816 were measured directly for volumes and interpolation. The 40-match figures are extrapolations **[I]**.
- The video evidence comes from 3 SoccerTrack development clips with a non-human-verified reference. 84 % of switch labels are reference-ambiguous.
- The ESTIMATED noise model is not calibrated **[U]**.
- The ±2.5 s context covers only part of the long ESTIMATED runs.
- Provider continuity is not ground truth, especially in ESTIMATED stretches.
- The validation and test sets are no longer pristine.
- Wadi Degla has about 1 m calibration error and no ground truth. Its human labelling has not started.

---

## 14. Recommendation

**CONDITIONAL GO**, limited to roadmap Phases 1–3:

1. Protocol freeze.
2. 40-development-match audit and quality sidecar.
3. 2 m generator into new versioned outputs, with an exact consistency check against v2 at ≤ 0.5 m.

**Conditions:**

- The advisor accepts that crossing and Δt are descriptors, not the selection rule.
- The advisor accepts that "occlusion" is not used for FIFA candidates.
- Nothing touches validation or test data.

Training (Phases 6–11) needs a second approval after the Phase 3 volume and quality report.
