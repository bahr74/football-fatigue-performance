# M11: Video-grounded occlusion detection and identity recovery (scientific investigation)

Date: 2026-11-13. Branch `feature/wadi-degla-demo`. This is a research and planning document. No pipeline, dataset, model, split or
locked artifact was changed. The only computation was four small read-only diagnostics. Their scripts and JSON outputs are in the session scratchpad, outside the repository (§0.3).

Companion document: [M11_REDESIGN_EXECUTION_PLAN.md](M11_REDESIGN_EXECUTION_PLAN.md).

---

## 0. Method of this investigation

### 0.1 Evidence labels

| Label | Meaning |
|---|---|
| **[V]** | Verified in this session: recomputed from raw/stored data or read from source code |
| **[R]** | Reported by an earlier repository document and not re-checked here |
| **[I]** | Inference: our interpretation or a hypothesis |
| **[U]** | Unresolved: needs new evidence or a decision |

### 0.2 Sources read

- **M11 documents:**
  - `docs/research/M11_ADVISOR_EVIDENCE_HANDOFF.md`;
  - `docs/research/m11_step1_verification_and_event_centred_context.md`;
  - `m11_identity_recovery_report.md`, `m11_advisor_intersection_correction.md`, `m11_sync_proximity_*.md`;
  - the advisor discussion report (`M11_ADVISOR_DISCUSSION_REPORT.md`, 11 Nov; outside the repository).
- **Video and CV documents:**
  - `docs/wadi_degla_cv_handoff.md`;
  - the M10.4–M10.8 Wadi Degla reports under `data/processed/vision_eval/`, with `m108…/FINAL_DECISION.md` and `TRACKING_AND_CROSSVIEW_EVALUATION.md` read in full;
  - `soccertrack-official-repo-audit-v1/SOCCERTRACK_OFFICIAL_REPOSITORY_AUDIT.md`;
  - `m63d-interior-swap-v1/README.md`.
- **Source code:**
  - `vision/pitch_mapping.py` and `vision/pitch_prototype.py`;
  - `ingestion/loaders.py` (raw vs smoothed handling);
  - `docs/dataset_schema_audit.md`.
- **Split file:** `data/processed/m11-encounters-v1/splits.json`: 40 development, 12 validation, 12 test **[V]**.

### 0.3 Diagnostics run (read-only, development data only)

| ID | Script (scratchpad `diag/`) | Data | Purpose |
|---|---|---|---|
| D1 | `fifa_cadence.py` | FIFA RAW 3812, 3816, 10506 (fixed development matches), full re-parse with plain `json` | Frame and update cadence, quantisation, noise, ESTIMATED runs, raw vs provider-smoothed stream |
| D2 | `st_occlusion.py` | SoccerTrack-v2 MOT ground truth for 117092, 117093, 118575, 118576 (the development clips of the CV milestones); calibration through the project's own `fit_hybrid(continuous=True)` | Image-box overlap against ground-plane distance |
| D3 | `fifa_short_est.py` | FIFA RAW 3812, 3816 | Lag of the provider-smoothed stream; whether short VISIBLE→ESTIMATED→VISIBLE bursts happen near other players |
| D4 | inline `cv2` probe | Wadi Degla MP4s; M10.8 cached detections | Real video properties and player size in pixels |

None of these opened a FIFA validation or test match. None opened SoccerTrack 118577, 118578 or 128058 (CV validation) or any Tier-2 file.

---

## 1. What the repository actually contains (verified)

| Asset | Status | Evidence |
|---|---|---|
| FIFA RAW tracking, 64 matches (40 dev / 12 val / 12 test) | present | [V] split file; RAW files under `data/raw/Tracking Data-*/` |
| Step 1 caches, all 40 dev matches | present, independently verified (0 mismatches in 7,432,290 frames) | [R] `m11_step1_verification…md` §2; outputs in `m11-step1-verification-v1` |
| Proximity v2/v3 (S40): 30,438 episodes at 0.25 m, 24,052 complete contexts, 1,766 presentation-ready | present | [R] handoff §7.1 (V'd there on 11 Nov) |
| Locked test (Step 7) | **consumed** | [R] `m11_step7_final_locked_test.md` |
| Validation partition | **seen twice** (Step 6 frozen run; m11-recovery-v1 one-time validation) | [R] handoff §6; memory of the Step 6 / recovery runs |
| Wadi Degla video | 3 files, **no tracking ground truth, no human labels** | [V] D4 (§2.1); M10.8 `FINAL_DECISION.md`: "Human-label accuracy … NOT RUN" |
| **SoccerTrack-v2 video with identity reference, pitch calibration and our own tracker outputs** | present for 4 development clips (117092, 117093, 118575, 118576) | [V] D2; [R] M6–M8.1b reports |
| Existing tracker ID-switch analysis on SoccerTrack | 514 reference-ID change events in 613 tracks (337 cross-team, 165 same-team, 12 GK) | [R] `m63d-interior-swap-v1/README.md` |
| Human annotation tools | Review tool for cross-camera pairs and sync (M10.8 `/review`, `/sync`); **no saved human decisions** | [R] M10.8 final decision |
| Hardware | 8 logical CPUs; NVIDIA GTX 1650 Ti (4 GB); torch 2.9.1+cu126 with CUDA available | [V] |
| ML libraries | numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, OpenCV 5.0.0, torch. **Not installed:** LightGBM, XGBoost, Shapely | [V] |

**The most important repository finding of this investigation [V].** The user's brief frames the choice as Wadi Degla
video versus FIFA positional data. The repository also holds a **third, much stronger video source**: **SoccerTrack-v2**.

- It is a fixed BePro panoramic camera recorded at 25 fps.
- Each clip has per-frame boxes with persistent identities derived from the BePro tracker, plus pitch calibration.
- Our own detector and tracker have already been run on it.
- Real tracker identity switches on these clips have already been catalogued (M6.3-D).

It is the only asset in the repository where **video, pitch coordinates, a persistent identity reference and real tracker
identity failures exist for the same moments.** That changes the recommended starting point (§14).

---

## 2. Wadi Degla video feasibility

### 2.1 Assets [V, D4]

| File | Resolution | Container fps | Frames | Duration | Notes |
|---|---|---|---|---|---|
| `wadi_degla_fixed.mp4` | 640 × 360 | 30.0 | 163,006 | 5,433.5 s | Fixed camera; a **different match** from the other two [R M10.4] |
| `wadi_degla_fixed_same_match.mp4` | 640 × 360 | 30.0 | 164,930 | 5,497.7 s | Fixed camera, same match as `moving`; **25 fps content in a 30 fps file**; 2,711.9–2,748.8 s is inserted footage copied from the moving camera [R M10.8] |
| `wadi_degla_moving.mp4` | 640 × 360 | 30.0 | 164,897 | 5,496.6 s | XbotGo auto-panning camera: pure rotation at a fixed position, no zoom, HFOV about 72° [R M10.4] |

All three are H.264 re-encodes of about 640 × 360 [V], described as Google re-encodes [R M10.4]. Recording metadata such as the original
resolution and the camera mounting height is **not available** [U].

**Player size in pixels** [V, D4: M10.8 cached detections, 4 windows per camera]:

- Fixed camera: median box height **13–16 px**, median width **6–9 px**; 17–23 players per frame.
- Moving camera: median height **21–29 px**, width 10–14 px; 5–18 players per frame.

For comparison, SoccerTrack-v2 ground-truth boxes have a median height of 41–46 px at 4096 × 1080 [V, D2]. Even at that size,
pretrained ReID failed the same-team identity gate (AUC 0.651) [R M6.3-B].

### 2.2 Answers to the ten feasibility questions

| # | Question | Answer | Label |
|---|---|---|---|
| 1 | Which matches and assets? | Two matches. Match W1 has one fixed view. Match W2 has a fixed and a moving view, synchronised (τ = −0.125 s, 95 % CI [−0.178, −0.072]) | [V] files; [R] sync |
| 2 | Camera type | Fixed single camera (two files) and one auto-panning camera; together a 2-view rig for W2 only | [R] M10.4–M10.8 |
| 3 | Frame rates and resolution | 640 × 360. The container says 30 fps; the fixed W2 content is 25 fps | [V]/[R] |
| 4 | Can the pipeline detect, track and recover trajectories around occlusions? | It detects and tracks (ByteTrack is the least fragmented). In 10 s windows: 18–39 tracks against ~11–23 visible people; fragmentation index 1.4–2.7; "new IDs appear after occlusions"; groups and pans are the weak spots. **Recovery across occlusions is not validated.** | [R] M10.8 workstream D |
| 5 | Pixels, pitch, or both? | Both. Pixels are native. Pitch coordinates come from calibration, with no measured pitch dimensions | [R] |
| 6 | Calibration reliability | Fixed camera, refined affine: held-out nearest-neighbour **0.64–0.98 m**, with about **1 m systematic error** [R M10.7]. Moving camera: per-frame rotation, **4.8–7.5 px** median residual [R M10.8]. Metric scale is not fixed until the pitch is measured | [R] |
| 7 | Can visual occlusions be identified and verified manually? | **Yes for the event itself** (two boxes merge or one disappears behind another). A 2-annotator protocol is feasible (§2.3) | [I] |
| 8 | Can the occluded player's identity be verified before and after? | **Only partially.** Jerseys are unreadable at 13–29 px. Kit colour separates teams, but **same-team identity across an occlusion is often not human-verifiable** from one view. The W2 second view sometimes resolves it (when the moving camera covers the event). Expect a sizeable `unresolvable` class | [I], supported by [R] M6.3-B (ReID fails even at 41–46 px) |
| 9 | Annotation effort | About 3–6 minutes per event for two-annotator labelling with the second view; about 10 h for a 100-event pilot (§2.4) | [I] estimate |
| 10 | Partial occlusion, camera motion, exits | Partial: label an ordinal visibility class, not a binary flag. Camera motion: pans create new IDs (M10.8: 8 reappearances with a new ID in pan window B_2006). Exits: label `left_view` as a separate outcome and never as a recovery | [R]/[I] |

**Ground-truth statement.** Wadi Degla gives **no reliable identity ground truth** today. Reliable labels exist only after the
human protocol below, and even then same-team identity will often stay `unresolvable`. The W2 two-view rig is the main tool for
reducing that class.

### 2.3 Annotation protocol (proposed, Wadi Degla W2)

**Unit.** An interaction episode: two or more players whose boxes overlap, or whose foot points come within 1.5 m in the
calibrated plane, within a 10 s clip.

**Labels per episode:**

1. `event_type`: one of:
   - `full_occlusion` (one player not visible for at least 1 frame);
   - `partial_occlusion` (box overlap, both partially visible);
   - `close_no_occlusion` (close but visually separate);
   - `side_by_side` (sustained parallel motion);
   - `camera_overlap` (overlap in image, clearly separate on the pitch);
   - `exit_or_cut` (player leaves the view or the camera pans away);
   - `pan_artifact`.
2. `visibility_loss_frames` for each player (integer frames, 25 fps).
3. `same_team`: yes / no / unknown.
4. `identity_outcome` for each pair (incoming tracklet → outgoing tracklet):
   - `continuous_correct`;
   - `switched` (the tracker swapped);
   - `fragmented` (a new ID was created);
   - `unresolvable_by_human`.
5. `evidence_used`: `single_view`, `second_view`, `motion_only`, `kit_colour`.
6. `annotator_confidence` (1–3).

**Process.**

- Two independent annotators work blind to tracker IDs. They see the raw clip plus the second view, and the tracker overlay only in a final step.
- Disagreements are adjudicated by a third pass.
- Reliability is reported as Cohen's κ on `event_type` and on `identity_outcome`, with bootstrap CIs over clips. The proposed target is κ ≥ 0.7 for `event_type`; below 0.6 the category is merged or redefined.
- Unresolvable labels are reported, never imputed.

**Unbiased selection.** Do not pick clips by eye.

1. Pre-compute label-free strata from the tracker output for the whole match: pan speed (static / slow / fast), crowding (number of
   foot-point pairs < 1.5 m), pitch third, and period.
2. Draw 10 s clip start times uniformly at random within each stratum, with a fixed seed recorded before viewing.
3. Annotate **every** interaction in a drawn clip, not only interesting ones. This gives the denominator needed for precision and recall.

### 2.4 Pilot size

- **Initial inspection:** 20 random clips (200 s), to calibrate the protocol and estimate the event rate. Expected about 2–8 interactions per clip [I].
- **Pilot labelling:** 60 clips → roughly 150–400 episodes. That is enough to estimate a 30 % `switched` or `fragmented` rate to about ±5–8 percentage points, and κ to about ±0.1 [I, binomial arithmetic].
- **Cost:** 60 clips × about 10 min per annotator × 2 annotators, plus adjudication: **about 25 person-hours** [I].

---

## 3. Data diagnostics (new evidence)

### 3.1 D1: FIFA RAW timing, cadence and noise (3812, 3816, 10506) [V]

| Quantity | 3812 | 3816 | 10506 |
|---|---|---|---|
| Unique frames (after collapsing repeated records) | 184,857 (889 repeats) | 198,142 (870) | 233,020 (1,245) |
| Frame interval within a period | 33.37 ms in **100 %** of intervals; frameNum step 1 in 100 % | same | same |
| Frame rate | **29.97 fps** (1/0.033367) | same | same |
| Raw coordinate update interval: 2 frames (0.0667 s) | 2,030,564 of 2,035,033 (99.78 %) | 99.73 % | 99.78 % |
| Update interval: 1 frame / 4 frames / ≥ 6 frames | 4,044 / 397 / 28 | 4,334 / 563 / 15 | 5,080 / 454 / 59 |
| Update interval, VISIBLE vs ESTIMATED | identical: p1 = p99.9 = 0.0667 s for both | same | same |
| Coordinate quantisation | ≤ 3 decimals (mm) | same | same |
| Step length at an update, VISIBLE (median / p99) | 0.125 / 0.75 m | 0.122 / 0.62 m | 0.126 / 0.73 m |
| Second difference at consecutive updates, VISIBLE (median / p95) | **1.3 / 12.6 cm** | 1.2 / 11.3 cm | 1.1 / 9.2 cm |
| Second difference at consecutive updates, ESTIMATED (median / p95) | **9.7 / 47.7 cm** | 8.1 / 32.4 cm | 9.0 / 52.7 cm |
| ESTIMATED share per player slot (median) | 0.55 | 0.64 | 0.57 |
| ESTIMATED run duration (p5 / median / p95) | 0.13 / **3.4** / 42 s | 0.13 / **6.4** / 65 s | 0.33 / **6.3** / 53 s |
| Confidence on ESTIMATED samples | always `LOW` | always `LOW` | always `LOW` |
| `HIGH` share of VISIBLE samples | 74.6 % | 66.1 % | 58.4 % |
| raw − provider-smoothed distance, VISIBLE (median / p95) | **0.74 / 2.86 m** | 0.69 / 2.64 m | 0.77 / 3.27 m |
| raw − provider-smoothed distance, ESTIMATED (median / p95) | 0.29 / 1.97 m | 0.29 / 1.77 m | 0.35 / 2.29 m |

**Interpretation.**

1. **Timing is not a source of uncertainty.**
   - Frames are exactly 29.97 fps with no gaps.
   - Positions update every second frame (15 Hz) for **both** VISIBLE and ESTIMATED samples.
   - The update instant is known to within the 33 ms frame clock, and the sampling interval is h = 0.0667 s.
2. **Interpolation error between updates is negligible for VISIBLE positions.**
   - The chord error of linear interpolation is at most a·h²/8. At a = 5 m/s² this is 2.8 mm.
   - The VISIBLE second difference (median 1.1–1.3 cm) is about what real acceleration alone produces (a·h² ≈ 1.3 cm at 3 m/s²). So VISIBLE tracks are already smooth, probably pre-filtered by the provider [I].
   - **Random jitter on VISIBLE positions is at the millimetre-to-centimetre level. Absolute accuracy against the true foot position is unknown** [U]: no independent reference exists for FIFA.
3. **ESTIMATED positions jitter about 7–8 times more** (median second difference 8–10 cm, p95 32–53 cm). Their absolute error is unknown and is probably the dominant uncertainty [I].
4. **ESTIMATED runs are long** (median 3.4–6.4 s, p95 42–65 s), and they cover more than half of all samples.
   - This fits players being **outside the broadcast camera's view**, not brief occlusions [I]. PFF states that its data comes from TV broadcast video; its documentation found online does not define `ESTIMATED` [U] (see the PFF blog in §11).
   - The provider's visibility flag is therefore mostly an off-screen flag and only rarely a visual-occlusion flag [I]. §3.3 tests the short runs.
5. **The provider-smoothed stream (`homePlayersSmoothed`) is not a lightly denoised raw stream.**
   - It differs from raw by a median of 0.7 m on VISIBLE samples.
   - The project correctly does not use it in M11 (Step 1 uses raw).
   - The smoothed stream is likely non-causal and uses information from other times, so **it must not be used for an online task**.
   - Its exact relation to raw (lag or a different estimator) is examined in D3 (§3.3).

**Consequences for thresholds.**

- Temporal tolerances below h = 0.0667 s are not resolvable. Above about 2h they are a modelling choice, not a measurement limit.
- For VISIBLE pairs, a spatial tolerance of 0.10–0.25 m is **not** forced by noise. It is a definition of "close".
- For ESTIMATED pairs, no tolerance is defensible without an error model, because their absolute error is unmeasured.
- Arrival-time uncertainty at a crossing is σ_t ≈ σ_pos / v. At v = 2 m/s it is about 5 ms for VISIBLE (σ ≈ 1 cm) and about 50 ms for ESTIMATED (σ ≈ 10 cm, jitter only, not bias).

### 3.2 D2: image occlusion against ground distance, SoccerTrack-v2 [V]

**Set-up.**

- The ground-truth boxes are BePro tracker positions projected into the image, with template box sizes [R official-repo audit]. Box overlap here is therefore **geometric projection occlusion under the real camera**, not human-verified visual occlusion.
- Foot points (box bottom centre) were mapped to pitch metres with the project's `fit_hybrid(continuous=True)` mapper. Its held-out error is 0.37–0.65 m median [R M7.1, M8.1].
- Calibration error is spatially smooth, so the error on the **distance between two nearby players** should be much smaller than the absolute error [I; not measured].
- Occlusion fraction = area of the far player's box (smaller foot y) covered by the near player's box.

| | 117093 | 118575 | 118576 | 117092 (pixels only; calibration failed QC) |
|---|---|---|---|---|
| Frames (25 fps) / identities | 5,976 / 22 | 6,000 / 22 | 5,975 / 22 | 6,000 / 22 |
| Median box height | 45.7 px | 44.0 px | 44.9 px | 40.9 px |
| Pair episodes with occlusion ≥ 0.5, per minute | 194.8 | 115.8 | 167.9 | 151.0 |
| Episode length, occlusion ≥ 0.5 (median / p95 frames) | 5 / 29 | 4 / 27 | 4 / 25 | 4 / 24 |
| **Ground distance at severe occlusion (≥ 0.5), median** | **2.30 m** | **2.86 m** | **1.67 m** | – |
| Share of severe-occlusion pair-frames with distance **> 0.25 m** | **97.4 %** | **97.9 %** | **98.1 %** | – |
| … > 1 m / > 2 m | 78.5 % / 56.4 % | 81.6 % / 63.0 % | 70.2 % / 43.6 % | – |
| Line-of-sight component at severe occlusion (median) | 2.29 m | 2.86 m | 1.65 m | – |
| **Perpendicular component** at severe occlusion (median / p95) | **0.18 / 0.39 m** | **0.15 / 0.43 m** | **0.19 / 0.39 m** | – |
| P(occlusion ≥ 0.5 \| d < 0.25 m) | 0.93 | 0.93 | 0.98 | – |
| P(occlusion ≥ 0.5 \| 0.25 ≤ d < 0.5) | 0.43 | 0.80 | 0.61 | – |
| P(occlusion ≥ 0.5 \| 0.5 ≤ d < 1) | 0.20 | 0.36 | 0.23 | – |
| P(occlusion ≥ 0.5 \| 1 ≤ d < 2) | 0.12 | 0.16 | 0.10 | – |
| Pair-frames with d < 0.25 m | 196 | 83 | 115 | – |

The estimated camera ground position is about (51 m, 85–87 m) in the corner-origin frame: behind the halfway line, about 17–19 m
beyond the near touchline. The height estimate from the panorama fit is not reliable (−0.1 to −6.9 m) [V value; I reliability].

**Interpretation.** This is the central quantitative finding of the investigation.

1. **A 0.25 m ground threshold is a high-precision but very low-recall detector of camera occlusion.** Below 0.25 m almost every
   pair is occluded (93–98 %). But fewer than 3 % of severe occlusions happen below 0.25 m.
2. **Visual occlusion is anisotropic.** It depends on the **angular separation as seen from the camera**: the perpendicular
   component is a median 0.15–0.19 m. It hardly depends on depth separation (a median of 1.7–2.9 m along the line of sight).
   Two players 3 m apart along the camera ray occlude each other; two players 0.6 m apart across it do not.
3. **Visual occlusion is frequent and short:** 116–195 severe pair-episodes per minute, median 4–5 frames (0.16–0.2 s).
4. **Consequence for FIFA:** the FIFA source camera (broadcast) is unknown and moves, so the image-space condition **cannot be computed from FIFA tracking** without assuming a camera.

### 3.3 D3: provider-smoothed lag and short ESTIMATED bursts (3812, 3816) [V]

**(a) Is the provider-smoothed stream lagged?**

- Median |raw(t) − smoothed(t + k)| on VISIBLE samples is **minimal at k = 0**: 0.742 m (3812) and 0.688 m (3816).
- It rises symmetrically to about 1.8 m at ±30 frames.
- So it is **not a time-shifted copy**: it is a different estimate at the same instant, about 0.7 m away from raw. Its nature is [U]. It stays excluded from M11.

**(b) Short ESTIMATED bursts.**

- *Burst:* a VISIBLE → ESTIMATED → VISIBLE run of the same player in the same period, lasting ≤ 1 s.
- *Measure:* the minimum distance to any other player within ±3 frames of the burst.
- *Control:* a random fully VISIBLE interval of the same length for the same player.

| | 3812 burst | 3812 control | 3816 burst | 3816 control |
|---|---|---|---|---|
| n | 1,234 | 1,219 | 758 | 734 |
| Burst length (p25 / median / p75) | 0.13 / **0.27** / 0.67 s | – | 0.13 / **0.27** / 0.53 s | – |
| Median minimum distance to another player | **1.69 m** | 4.25 m | **1.55 m** | 4.22 m |
| Share < 0.5 m | **13.7 %** | 1.8 % | **16.0 %** | 1.6 % |
| Share < 1 m | **31.8 %** | 6.5 % | **34.0 %** | 5.9 % |
| Share < 2 m | 54.1 % | 19.9 % | 59.4 % | 18.0 % |

**Interpretation.**

- Short ESTIMATED bursts are **strongly enriched near another player**: about 5× at < 1 m and about 8× at < 0.5 m.
- Their median duration (0.27 s) is close to the SoccerTrack occlusion-episode durations (median 0.16–0.20 s; §3.2).
- This is the strongest evidence available that **the provider's own detector loses players during close interactions**. A short burst bounded by VISIBLE is therefore the **best FIFA-internal proxy of a real visual ambiguity** [I, supported by these data; not video-verified].
- About two thirds of bursts are *not* within 1 m of anyone. Other causes (motion blur, camera cuts, frame edges) are likely [I].
- **This is a new, label-symmetric candidate stratum.** The burst belongs to one player and does not depend on the pairing of labels across the gap. It needs no camera model. It should be a primary stratum in the redesigned FIFA candidate set (§6, §14).

---

## 4. Three different phenomena: mathematics

Notation: player positions p_i(t) ∈ ℝ² on the pitch, velocities v_i(t), camera centre C ∈ ℝ³ (height H), and a camera
projection π: ℝ³ → image.

### 4.1 Ground-plane proximity (what M11 v2 detects)

- Distance: d_ij(t) = ‖p_i(t) − p_j(t)‖.
- Episode: E_R = {t : d_ij(t) ≤ R}.
- This is **symmetric under swapping the two labels**: d_ij does not depend on which incoming track continues as which outgoing track. That property is essential (§8).

### 4.2 Image-space occlusion (camera-dependent)

Approximate each player by a vertical cylinder of radius r ≈ 0.25–0.3 m and height h ≈ 1.8 m.

- **Angular separation.** As seen from C, let θ_ij(t) be the angle between the rays to the two players' axes.
- **Overlap condition.** The silhouettes overlap when θ_ij(t) ≲ (w_i + w_j)/2, where w_k ≈ 2r/‖p_k − C_xy‖ is the angular half-width.
- **Ground-plane equivalent.** With the components d_⊥ (perpendicular to the ray) and d_∥ (along it): **occlusion ⇔ |d_⊥| ≲ 2r ≈ 0.5 m, for any d_∥** up to a limit set by vertical extent and camera height. Approximately, the far player's feet are hidden only while d_∥ < h·D/H, where D is the horizontal distance from the camera.
- **Implication.** Occlusion is a thin strip along the line of sight, not a disc around the player. D2 confirms it: perpendicular p95 is 0.39–0.43 m, while the along-ray median is 1.7–2.9 m.
- **What determines the risk.** For a low camera (small H) the strip is long. For a drone view (large H) it collapses towards a disc. **The detector therefore depends on the camera.**

### 4.3 Tracking-identity ambiguity (what identity recovery needs)

**Two-player case.** Given incoming tracklets I_1, I_2 that end near t_0 and outgoing tracklets O_a, O_b that start near t_1,
identity is ambiguous when the two assignments have similar likelihood.

**Likelihood ratio and swap probability:**

Λ = log [ L(I_1→O_a, I_2→O_b) / L(I_1→O_b, I_2→O_a) ]

P_swap = 1/(1 + e^Λ)

**Gaussian motion model.** With a predictive density from a motion model, for example a Kalman filter with process noise q and
measurement noise σ:

L(I→O) = N(o_start; μ_I(t_1), Σ_I(t_1) + Σ_O)

Σ_I grows roughly as q·Δt³ for a constant-velocity model over a gap Δt = t_1 − t_0.

**Ambiguity grows with:**

- a short separation relative to the prediction spread, ‖μ_1 − μ_2‖ / √|Σ|;
- a longer gap;
- similar velocities. Side-by-side motion is maximally ambiguous; a high-angle crossing at speed is the least ambiguous for a motion model.

**This is the quantity a recovery model must resolve.** It is neither "the two were close" (§4.1) nor "the camera could not see
one of them" (§4.2):

- **Proximity without a gap** is not an identity problem: both players are observed.
- **A gap without proximity** is a re-identification problem.
- **Ambiguity requires both:** missing or merged observations (from occlusion, an off-screen period or a detector merge) **and**
  competing candidates within the predictive spread.

**Synthesis.** Visual occlusion (§4.2) is one **cause** of missing or merged observations. Proximity (§4.1) is a **necessary
condition** for competing candidates. Identity ambiguity is the **joint event**. A detector for training data should therefore
estimate:

risk_ij = P(observation loss or merge | geometry, camera, quality) × P(competing hypothesis | proximity, velocities, gap)

---

## 5. Transferability between Wadi Degla / SoccerTrack and FIFA

| Feature | Wadi Degla / SoccerTrack | FIFA | Class |
|---|---|---|---|
| Ground-plane distance d(t), its minimum, time below R | yes, through calibration (SoccerTrack about 0.4–0.6 m abs.; Wadi Degla about 1 m abs.) | yes (mm) | **A**, transferable if calibration error on *relative* distance is small (to be measured) |
| Relative velocity, crossing angle, heading change, duration | yes (noisier in video) | yes | **A** |
| Same team / opposite team | yes (kit) | yes | **A** |
| Number of other players within radius ρ (crowding) | yes | yes | **A** |
| Trajectory curvature and continuity before/after | yes | yes | **A** |
| Position uncertainty | tracker covariance / detection confidence | provider VISIBLE / ESTIMATED / confidence | **A in concept, B in calibration**: the scales differ, and ESTIMATED is mostly off-screen |
| Image-box overlap, occlusion fraction, depth order | yes | **no** (no camera) | **B** (viewpoint-dependent) |
| Line-of-sight / perpendicular decomposition | yes (camera known) | only with an **assumed** broadcast camera | **B** |
| Visual appearance, kit, pose | yes | no | **C** |
| Actual visual visibility loss of the occluded player | yes (labels) | not directly. **Short VISIBLE→ESTIMATED→VISIBLE bursts (≤ 1 s)** are a proxy enriched about 5× near other players (§3.3); long ESTIMATED runs are not | **C** directly; **proxy is A-like** (it needs validation on video) |
| Provider estimation error | n/a | unknown | **C** |
| Tracker ID-switch outcome | yes (SoccerTrack vs BePro; Wadi Degla after human labels) | no real switches. Provider IDs are continuous by construction; switches can only be synthetic | **C** |

**Transfer strategy.** Learn P(identity failure | class-A features) on video where failures are observed, then apply it to FIFA as
a **risk score**.

- **Where:** SoccerTrack first (reference IDs, real tracker failures), with Wadi Degla as a second domain once labelled.
- **What not to transfer:** class-B features.
- **Optional geometric prior:** a **broadcast-camera prior** (a camera on the halfway line at a typical stand height and distance) may be added to FIFA as a sensitivity variable. It is a hypothesis, not a measurement.

**Validity conditions.** All four are required:

1. Relative-distance error under calibration is shown to be small on SoccerTrack, ideally against BePro pitch positions (download approval needed).
2. Class-A feature distributions overlap between domains. Check with covariate-shift diagnostics: a domain classifier's AUC and per-feature histograms.
3. The risk model is evaluated leave-one-match-out within the video domain.
4. Transfer is claimed only as **ranking of candidates by risk**, never as "this FIFA event was an occlusion".

**Conclusion.** Direct transfer of *visual occlusion labels* to FIFA events is scientifically invalid:

- the matches are different;
- the camera is unknown;
- no FIFA video exists.

Transfer of a **camera-agnostic identity-ambiguity risk function** is plausible and testable.

---

## 6. Candidate-detection methods

Common inputs: Step 1 valid segments (piecewise linear, 15 Hz) with quality flags. Cost is given for S40 (about 3,380 player-minutes
× 231 pairs).

### A. Geometric intersection with temporal tolerance

**Formulation.**

- Segment intersection P = A(u) = B(v).
- Arrival times t_A = t_A0 + u·Δt_A and t_B likewise.
- Accept if |t_A − t_B| ≤ τ.

**Hyperparameters:** τ, plus optional floors on speed and angle.

**Uncertainty.** σ_Δt² = σ_A²/v_A² + σ_B²/v_B² (first order). A defensible rule is |Δt| ≤ max(h, z·σ_Δt).

**Evidence.** It is resolution-limited below h = 0.067 s. Counts are fragile: 284 accepted at (h, z = 2), ranging from 0 to 1,454 across (τ, z) [R].

**Weaknesses.**

- It misses near misses that never cross on the sampled paths.
- **It is not label-symmetric.** Whether paths "cross" depends on the identity labels, so selecting on it leaks the answer: the "assume crossed" prior gives 75 % [R].

**Cost:** O(segment pairs), cheap.

**Suitability for training: poor** as a selection rule; acceptable as a stratum.

### B. Synchronised proximity

**Formulation.** d(t) ≤ R, solved analytically per synchronous interval. Already implemented: `proximity.py`.

**Hyperparameters:** R, minimum duration, merge gap.

**Evidence.**

- Counts at 0 / 0.10 / 0.25 / 0.50 m: 2,750 / 13,915 / 30,438 / 61,106 (S40) [R].
- d_min is near-uniform below 0.5 m, so there is no natural break [R].
- Interpolation error is negligible for VISIBLE positions (§3.1).

**Strengths:**

- label-symmetric;
- exact in continuous time;
- cheap (O(pairs × intervals); already run on S40).

**Weaknesses:**

- isotropic, so it misses camera-occlusion geometry;
- includes side-by-side movement and pass-bys with no identity risk;
- does not use velocity or uncertainty.

**Suitability:** a good **high-recall candidate generator** at a generous R (0.5–1.0 m) when combined with scoring. As a final definition at 0.25 m, the recall loss is large; D2 suggests over 97 % of camera occlusions would be missed.

### C. Hybrid spatiotemporal

| Variant | Formulation | Comment |
|---|---|---|
| C1: distance AND time tolerance | d ≤ R and \|Δt\| ≤ τ at a crossing | **Double counting.** For crossing paths, d_min ≈ v_rel·\|Δt\|·sin(φ)/… : small \|Δt\| and small d_min are the **same** condition, so it adds little information |
| C2: velocity-adaptive radius | d(t) ≤ R0 + k·‖v_rel‖·h + z·σ_ij(t) | Justified: it accounts for sampling and uncertainty. With VISIBLE noise the extra terms are small (about 1–2 cm). With ESTIMATED they matter |
| C3: probabilistic overlap | P(‖p_i − p_j‖ ≤ 2r) with p ~ N(μ, Σ_quality) | A principled way to include ESTIMATED positions. It needs Σ for ESTIMATED, which is unknown (§3.1) |
| C4: nearest approach in space-time | min_t d(t) with the time to minimum and the relative speed there | Equivalent to B plus descriptors. Use as features |
| C5: **learned ambiguity risk** | f(class-A features) → P(identity failure), trained on video (SoccerTrack) | **Recommended scoring layer** (§5). Label-symmetric if its features are symmetric |

**Does combining add information?**

- Distance AND arrival-time difference: **no**, they are largely redundant.
- Distance **with relative velocity, duration and uncertainty**: **yes**, these are distinct dimensions. D2 shows that distance alone has a poor operating curve.

### D. Video-grounded method

**Formulation.**

1. On SoccerTrack, compute from BePro-derived pitch tracks every pair-window with d_min ≤ 3 m (high recall, given §3.2).
2. Label each with the observed **tracker identity outcome** from our tracker against the reference (switch / fragment / continuous), using the existing M6.3-D machinery.
3. Fit a calibrated model of risk on class-A features.
4. Apply it to FIFA as a candidate score.

**Data:** 4 dev clips × 4 min = 16 match-minutes; 514 reference-ID change events [R]. These are small. Use GBM or logistic regression with monotone constraints, not deep learning.

**Weaknesses:**

- domain shift: fixed panorama vs broadcast; university vs World Cup;
- BePro reference errors are unmeasured;
- the tracker's failures depend on our detector.

**Suitability:** the best available **calibration** of what "risky" means, and a defensible basis for a threshold or weight.

### E. Alternatives from the literature

| Method | Idea | Relevance |
|---|---|---|
| FOOTBALLTrace (Mostafa et al., 2023) | Windowed distance ≤ τ (image units) for ≥ 50 % of a 2 s window; operator confirmation; SG smoothing; tail exchange | **Candidate stage** = B with a duration criterion. The correction stage was at chance on FIFA [R Step 3]. Its τ is not in metres |
| ByteTrack (Zhang et al., 2022) | Associate low-score boxes to recover occluded objects | A video-side mechanism, not a FIFA detector |
| Deep-EIoU (Huang et al., 2024) | Expansion IoU without a Kalman filter, for irregular sports motion | Tried on our cached detections, M9 Phase B: −0.053 HOTA vs ByteTrack [R] |
| Graph-based MOT (MPNTrack, Brasó & Leal-Taixé 2020; SUSHI, Cetintas et al. 2023) | Learned association on a graph of detections or tracklets | A model for multi-player association (§7). Not needed for candidate detection |
| Off-screen imputation (Graph Imputer, Omidshafiei et al. 2022) | Bidirectional graph VAE imputes off-screen players, with a camera module simulating visibility | **Highly relevant**: it shows how to simulate broadcast visibility on full tracking, which supports synthetic-gap realism (§9) |

### F. Provider visual-loss bursts (new; from D3)

**Formulation.**

- *Candidate:* a short ESTIMATED burst of player i (VISIBLE before and after, ≤ 1 s, same period).
- *Pairing:* with every player j with d_ij ≤ R_b at some time within the burst ± δ.

**Hyperparameters:** maximum burst length (0.5 / 1 / 2 s), R_b (0.5 / 1 / 2 m), δ (0–0.2 s).

**Evidence:** about 1,000 bursts per match; 32–34 % lie within 1 m of another player, against 6–7 % for controls [V D3].

**Strengths:**

- it is a *real* observation-loss event, the provider's own;
- it is label-symmetric;
- its gap is natural rather than synthetic. Both endpoints are VISIBLE, so before/after positions are observed.

**Weakness:** the cause of a burst is unknown. Ground truth about whether the provider's identity across the burst is correct cannot be had from FIFA.

**Suitability:** the **most realistic FIFA population for evaluating recovery under real gaps**.

**Ranking for M11 candidate extraction:**

1. B at a generous radius, as the **candidate generator** (recall);
2. F, provider visual-loss bursts with a nearby player, as the **realistic-gap stratum**;
3. C5/D, a **video-calibrated, label-symmetric risk score** (ranking and tiers);
4. C2/C3 for quality-aware handling of ESTIMATED positions;
5. A only as a descriptive stratum.

The exact (R = 0) and pure 2D crossing definitions remain rejected [R handoff §4].

---

## 7. Cleaning: restart or extend?

**Assessment.** Step 1 is verified correct for its rules (0 mismatches; 80,086,285 segments, 0 defects) [R]. The new objective
needs **more information**, not different cuts.

| Option | Recommendation | Reason | Cost |
|---|---|---|---|
| Keep Step 1 unchanged | **Yes** (immutable base) | Correct; the hashes underpin v2/v3 | 0 |
| Add quality flags (sidecar) | **Yes** | Per-sample: ESTIMATED run length and position in run (start / middle / end), short-burst flag, `confidence` HIGH/MEDIUM/LOW, local jitter (second difference), clone flag, distance to the nearest other player | CPU hours, a new versioned directory |
| Revise segmentation | **No** | The 0.5 s gap rule never binds; jumps are rightly cut; joining jumps inflates counts (3812: 18,469 → 22,144) [R] | – |
| Selective smoothing | **No for detection**, optional for features | VISIBLE is already smooth; smoothing ESTIMATED hides jitter that is informative about uncertainty | – |
| Treat ESTIMATED probabilistically | **Yes, in scoring** (C3) with a σ estimated from jitter, labelled as a lower bound | The absolute error is unknown | small |
| Quality tiers | **Yes**: T1 VISIBLE/VISIBLE at t* ± 1 s; T2 mixed; T3 ESTIMATED/ESTIMATED; T4 artifact-suspected | Report every metric per tier | small |
| Exclude or down-weight | Exclude only T4 clones (bit-identical positions) from training; down-weight T3 through sample weights, and test the sensitivity | Clones are not physical events [V handoff §2.4] | small |

**Verdict:** **extend, do not restart.** Restarting would cost about a day of CPU time and invalidate verified provenance for no identified gain.

---

## 8. The identity-recovery learning problem

### 8.1 Formulation

| Formulation | Fit | Reason |
|---|---|---|
| 22-way player classification | **Reject** | Identities do not transfer across matches. The model would memorise rosters. The task is relative association |
| Binary pairwise matching (in, out) → same? | Building block | Produces the cost matrix. Alone it ignores the one-to-one constraint |
| **Bipartite assignment over a local interaction group** | **Primary (offline)** | K incoming × K outgoing tracklets (K = 2 at first; K ≤ 4 for groups). Score each pair, then solve with Hungarian or exhaustive search (K! ≤ 24). Evaluate the permutation |
| Permutation prediction (direct) | Equivalent for K = 2; brittle for larger K | – |
| Probabilistic tracking (Kalman / JPDA / MHT) | **Primary baseline and online formulation** | Gives P_swap with uncertainty; the natural abstention signal |
| Metric / contrastive embedding | Later | Needs much more data; useful mainly with appearance |
| Sequence-to-sequence (predict positions in the gap) | Auxiliary | Prediction is not association; it can supply a likelihood |
| GNN over all players in the window | Advanced, for K > 2 and team context | §9 |

**Offline task (main).**

- *Input:* the anonymous incoming tracklets for [t0 − 2.5, t0] s, the anonymous outgoing tracklets for [t1, t1 + 2.5] s, the gap mask, the per-sample quality, team membership, and the other players' positions (context).
- *Output:* a permutation with a calibrated confidence, plus an abstain option.

**Online task (separate).**

- At each time, assign the current observations to existing tracks without future data, deferring decisions up to a latency L.
- Evaluate it separately: online models must never see the outgoing future.
- The provider-smoothed stream must never be used online (§3.1).

### 8.2 Label construction without leakage

- **Labels:** provider slot continuity through a **synthetic** gap. Real FIFA swaps cannot be labelled.
- **The gap must not be centred on t* in a way that reveals the answer.**
  - Randomise its placement and length: draw from the empirical ESTIMATED short-run distribution (§3.1), or from the SoccerTrack visibility-loss distribution (§3.2: median 0.16–0.2 s, p95 about 1.2 s).
  - Also include longer gaps (up to 2–3 s) as a stress stratum.
- **Anonymise and randomise the order of incoming and outgoing tracklets.** The positive class (keep vs swap) must be 50/50 by construction. Report "always keep" and "always swap" baselines.
- **Forbidden features:**
  - slot or player IDs, jersey;
  - `conf_in`/`conf_out`, which carried a provider hint (AUC 0.60) [R Step 5];
  - absolute match time;
  - any feature computed on identity-labelled paths that crosses the gap.
  - The crossing or non-crossing of *labelled* paths is the a1 leak [R].
- **Selection must be label-symmetric.** Candidates come from unordered-position functions (§4.1), never from labelled-path crossings.

### 8.3 Validity of provider identities as labels

- Provider IDs are continuous by construction.
- During long ESTIMATED runs the "truth" is a provider guess. A recovery label there is a label of *provider consistency*, not of real identity [I].
- Mitigation:
  - report results on T1 (both VISIBLE around the gap) as the primary metric;
  - report results on F-bursts (a real provider gap, VISIBLE on both sides) as the realistic-gap metric. Here the label is the provider's own re-acquisition, so the score measures agreement with the provider, not truth;
  - treat T3 as an exploratory stratum;
  - measure real identity accuracy only on video (SoccerTrack against BePro; Wadi Degla against human labels).

### 8.4 Hard cases

| Case | Handling |
|---|---|
| Multi-player groups (K > 2) | Group = connected component of the proximity graph within the gap. Solve the K × K assignment. Report results per K |
| Substitutions | Exclude windows where the slot set changes |
| Missing outgoing tracklet (exit, end of period) | Allow an "unmatched" column with a learned cost. Report it separately |
| Abstention | Abstain if max-permutation probability < threshold. Report a coverage–accuracy curve. Thresholds are fixed on development only |

---

## 9. Models (progressive)

| Stage | Model | Input | Loss / output | Variable length, missing data | Risk | Data need | Cost (local GPU 4 GB / 8 CPU) |
|---|---|---|---|---|---|---|---|
| M0 | Nearest end-to-start; constant velocity; constant acceleration; Kalman (CV) with Mahalanobis + Hungarian | last k samples in / first k out | likelihood → permutation | native | none (no training except q, σ) | none | seconds |
| M0+ | Kalman with **quality-dependent R** (VISIBLE vs ESTIMATED) and RTS smoothing within tracklets | same | same | native | q/R tuned on dev | dev only | seconds |
| M1 | GBM (sklearn `HistGradientBoostingClassifier`; LightGBM not installed) on symmetric pairwise features: Kalman NLL both directions, end-to-start distance, velocity and heading differences, curvature, gap length, quality tier, team flags, crowding, assignment regret | fixed vector | log loss → pairwise P → Hungarian | via engineered features | low to moderate; nested LOMO | about 10³–10⁴ independent episodes | minutes on CPU |
| M2 | Siamese GRU / 1D-TCN encoder (relative coordinates, Δt, quality mask) → compatibility score; in + out | padded (T × 6) sequences with masks | pairwise BCE or a permutation-level softmax over K! | masks; time-aware Δt input | **moderate–high**: correlated windows; needs grouped CV and early stopping | about 10⁴+ episodes | minutes to 1 h on a GTX 1650 Ti |
| M3 | Small transformer over tracklet tokens with set attention (in + out + context players) | tokens (≤ 30) | permutation softmax (Sinkhorn optional) | native masking | high | 10⁴–10⁵ | about 1–3 h |
| M4 | GNN (message passing; MPNTrack / SUSHI-style) over all players in the window | graph of tracklets | edge classification + flow constraint | native | high | as M3 | about 1–3 h |
| M5 | Hybrid: M0 Kalman likelihood as an input feature to M1/M2 | – | – | – | lowest of the learned models | – | – |

**Expectation [I].**

- Previous evidence: GBM 0.764 vs hybrid GRU 0.769 (not significant) on contacts [R]. Gains from DL on pure kinematics are likely small.
- Information-theoretic limit: when two players move identically side by side through a gap, no kinematic model can do better than chance. The **ceiling depends on the population**.
- Recommended order: M0 → M0+ → M1 → M5 → M2. M3/M4 only if M2 beats M1 by a pre-registered margin on development LOMO.

---

## 10. Evaluation and leakage

### 10.1 Separate the two questions

1. **Detector quality.** Recall and precision of candidate events against an *independent* event reference:
   - SoccerTrack tracker failures vs BePro (identity-failure recall);
   - Wadi Degla human labels (occlusion recall);
   - candidate volume and diversity (strata coverage);
   - quality-adjusted yield (T1-equivalent events per match).
2. **Recovery quality.** Association accuracy, swap rate, coverage at a fixed false-association rate, calibration (ECE), on a
   **common evaluation population** that is identical for every detector and model. Recommended: a union of strata, risk-stratified, drawn by a fixed seeded rule independent of any detector.

Training separate models on populations produced by different detectors and comparing their accuracies confounds detector
quality with population difficulty. That is exactly the a1/a3 failure: 79.7 % on crossings, 50.7 % on non-crossings [R].

### 10.2 Metrics

- **Permutation-level:** accuracy, swap rate, coverage@FA ≤ 5 % and 10 %, ECE, Brier score.
- **Breakdowns:** by tier, K, gap length, same vs opposite team, risk decile.
- **Video:** IDF1 and HOTA only where real tracks and a reference exist (SoccerTrack, using TrackEval). These are **not** applicable to FIFA synthetic gaps, which have no detection component.

### 10.3 Leakage controls

| Risk | Control |
|---|---|
| Overlapping windows / repeated encounters | Group by (match, pair, connected episode cluster within 10 s); one evaluation sample per cluster; grouped CV |
| Same match in train and test | Leave-one-match-out within development; match-level splits only |
| Synthetic variants of one event | All variants in the same fold; count independent *events*, not rows |
| IDs, slots | Feature allow-list; an automated test that permuting the IDs leaves the features unchanged |
| Future data online | Separate online pipeline; unit test that outputs at time t are unchanged when data after t is perturbed |
| Selection disclosing the answer | Selection only from label-symmetric functions; test that P(keep) = 0.5 within every stratum |
| Hyperparameter tuning on final data | Nested LOMO on development; frozen config; one-time evaluation |

### 10.4 Honest status of the held-out data

- **Test (12 matches):** consumed by Step 7 for pairwise encounter matching. **Validation (12):** used by Step 6 and by m11-recovery-v1 [R].
- For the **new formulation**, neither is pristine in the strict sense: their trajectories have informed earlier decisions.
- Options for the advisor:
  - (a) declare the validation set as one-time final for the new formulation, with full disclosure;
  - (b) use the 40 development matches only, with nested LOMO, as the primary evidence and the validation set as a single confirmatory run;
  - (c) obtain additional matches (none exist in this release).
- **Recommendation: (b).** The test set stays closed.

---

## 11. Literature (verified)

| Work | Year | Link | Data | Method | Ground truth | Results (as reported) | Limits | Relevance |
|---|---|---|---|---|---|---|---|---|
| Mostafa, Rushdi, Basha, Sayed: *FOOTBALLTrace* (icSPORTS) | 2023 | [doi:10.5220/0012226800003587](https://doi.org/10.5220/0012226800003587) | 3-camera 25 fps panorama | Windowed distance candidates; operator confirmation; SG smoothing; tail exchange | Operator-confirmed events (40) | 87.5 % (70/80 trajectories), including tuning data [V handoff §5] | Image units; small n; no separate held-out figure | Candidate stage = method B + duration; human confirmation step is the model for our annotation |
| Zhang et al.: *ByteTrack* (ECCV) | 2022 | [arXiv:2110.06864](https://arxiv.org/abs/2110.06864) | MOT17/20 | Associates low-score boxes to recover occluded objects | Human boxes | MOT17 test: 80.3 MOTA, 77.3 IDF1, 63.1 HOTA | Pedestrian domain | Our production video tracker |
| Huang et al.: *Deep-EIoU* (WACV workshops) | 2024 | [arXiv:2306.13074](https://arxiv.org/abs/2306.13074) | SportsMOT, SoccerNet-Tracking | Expansion IoU, no Kalman filter, plus ReID | Human boxes | 77.2 HOTA SportsMOT; 85.4 SoccerNet | Broadcast close-ups | Our replay was −0.053 HOTA [R M9] |
| Cui et al.: *SportsMOT* (ICCV) | 2023 | [arXiv:2304.05170](https://arxiv.org/abs/2304.05170) | 240 clips (basketball, volleyball, football) | Dataset + MixSort | Human boxes | IDF1/AssA low while DetA is high, so association is the bottleneck | Broadcast clips | Supports focusing on association |
| Scott et al.: *TeamTrack* (CVPR workshops) | 2024 | [arXiv:2404.13868](https://arxiv.org/abs/2404.13868) | Full-pitch soccer (fisheye and drone), basketball, handball | Dataset; ByteTrack / BoT-SORT baselines; trajectory forecasting | Annotated boxes | 4.37 M boxes | – | **Candidate second video domain with drone (top-down) views**, which isolates camera-geometry effects (§4.2); not downloaded [U] |
| Scott et al.: *SoccerTrack v2* (arXiv) | 2025 | [arXiv:2508.01802](https://arxiv.org/abs/2508.01802) | 10 university matches, BePro panoramic 4K | Dataset for game-state reconstruction | BePro-derived positions and IDs | GS-HOTA baselines | Paper says identities are not in the public release; our local MOT GT has 22 persistent IDs [V] | **Primary video-grounded source** for this redesign |
| Luiten et al.: *HOTA* (IJCV 129) | 2021 | [arXiv:2009.07736](https://arxiv.org/abs/2009.07736) | – | Detection, association and localisation decomposition | – | – | Needs a detection-level GT | Video evaluation only |
| Brasó & Leal-Taixé: *MPNTrack* (CVPR) | 2020 | [arXiv:1912.07515](https://arxiv.org/abs/1912.07515) | MOT15/16/17 | Message-passing network over the min-cost-flow graph | Human boxes | MOT17 about 58.8 MOTA / 61.7 IDF1 | Pedestrians | Template for the M4 GNN |
| Cetintas, Brasó, Leal-Taixé: *SUSHI* (CVPR) | 2023 | [arXiv:2212.03038](https://arxiv.org/abs/2212.03038) | MOT17, MOT20, DanceTrack, BDD | Hierarchical GNN, short + long term | Human boxes | SOTA claims on 4 datasets | Heavy | Long-gap association; reference for M4 |
| Omidshafiei et al.: *Multiagent off-screen behaviour prediction in football* (Sci. Rep.) | 2022 | [doi:10.1038/s41598-022-12547-0](https://www.nature.com/articles/s41598-022-12547-0) | Full tracking + simulated broadcast camera | Graph Imputer (bidirectional graph VAE) | Full tracking | Better than baselines on imputation | Imputation, not association | **Camera-module simulation of visibility** is the template for realistic synthetic gaps on FIFA |
| PFF FC 2022 World Cup dataset (blog) | 2023 | [pff.com blog](https://www.blog.fc.pff.com/blog/pff-fc-release-2022-world-cup-data) | – | States that tracking comes from broadcast video | – | – | `ESTIMATED` is not defined there [U] | Supports reading ESTIMATED as off-screen [I] |

Not verified, so not cited with numbers: GTATrack (SoccerTrack challenge winner). Our earlier audit [R] reports HOTA 0.60, from
its repository, not from a peer-reviewed paper.

**Reproducible findings** are the ones above with stated benchmarks. **Speculative** elements of this redesign:

- the broadcast-camera prior;
- the transfer of the risk function;
- the expected size of DL gains.

---

## 12. Dataset size and quality

**Numbers versus independent instances.**

- 30,438 proximity episodes (S40) are not 30,438 independent labels. Repeated contacts of one pair cluster: ~82 % of 2D events
  lie in pairs that re-cross within 5 s [R].
- Synthetic scenarios multiply rows (×15) without adding independent information.
- **Effective size** n_eff ≈ n / (1 + (m − 1)·ρ) for clusters of size m with intra-cluster correlation ρ. Report n_eff (estimated from grouped-bootstrap variance), not row counts.

**Strategies to grow the useful data while keeping it valid:**

1. **High-recall generation:** proximity at R = 1.0 m (S40), stored with continuous descriptors. Nothing is deleted at this stage.
2. **Scoring instead of deletion:** video-calibrated risk (§6 C5) plus quality tier. Weight training by tier.
3. **Hard negatives:** side-by-side and pass-by pairs at high risk where the kinematics *do* disambiguate. Close-but-easy cases teach calibration.
4. **Controlled synthetic ambiguity:** gaps drawn from empirical visibility-loss distributions; noise drawn from the measured jitter per tier (§3.1). A camera-visibility simulation (Graph Imputer style) is optional.
5. **Physically plausible augmentation:** pitch reflection (x → −x, y → −y, consistently for all players), time reversal (valid for offline association), and rotation of local frames. Never augment across folds.
6. **Active learning for annotation:** select Wadi Degla / SoccerTrack clips where the risk model is most uncertain, *after* the random pilot sample has fixed the denominator.
7. **No weak labels from the model itself on evaluation data.**

---

## 13. Controlled comparison design (summary; details in the execution plan)

| Detector | Search range (evidence) |
|---|---|
| A: crossing + τ | τ ∈ {h, 2h, 3h, 0.2, 0.5} s; z ∈ {1, 2, 3} |
| B: proximity | R ∈ {0.10, 0.25, 0.50, 0.75, 1.00, 1.50} m (nested, already computed up to 0.50) |
| C2: velocity-adaptive | R0 ∈ {0.25, 0.5}, k ∈ {0, 0.5, 1}, z ∈ {0, 2}; σ by tier |
| C3: probabilistic | P threshold ∈ {0.1, 0.3, 0.5}; σ_EST multipliers {1, 2, 4} × jitter |
| D: video-calibrated risk | risk deciles |

**Detector evaluation** (separate from recovery):

- recall of identity failures and visual occlusions on SoccerTrack / Wadi Degla labels;
- volume curves on S40;
- tier composition;
- compute time.

**Recovery evaluation:** a single common population (§10.1); the same models; the same folds.

**Ablations:** drop one feature family at a time (distance, velocity, curvature, quality, team, crowding, Kalman likelihood);
permutation importance computed on held-out folds only.

---

## 14. Final recommendations (answers to the 15 questions)

1. **Start with Wadi Degla video analysis?**
   - **Start with *video*, but SoccerTrack-v2, not Wadi Degla.** SoccerTrack has a persistent-ID reference, pitch calibration, real tracker failures (514 events) and 41–46 px players.
   - Wadi Degla has no ground truth, 13–29 px players and about 1 m calibration error.
   - Wadi Degla is the second domain, for a human-labelled pilot on match W2 (two views), run in parallel.
2. **Smallest useful pilot:**
   - SoccerTrack, 3 calibrated development clips (117093, 118575, 118576), CPU only, no annotation.
   - Join the M6.3-D identity-failure events to BePro-derived pitch features.
   - Estimate P(failure | d_min, v_rel, angle, duration, team, crowding).
   - Compare its ROC with the 0.25 m rule and with crossing + τ.
3. **Can video-confirmed occlusions be related to FIFA reliably?** **Not event-by-event** (different matches, unknown FIFA camera). Only as a **calibrated, camera-agnostic risk function** applied to FIFA. Valid only for ranking, after domain-shift checks.
4. **Most defensible occlusion definition.** Keep three definitions apart:
   - **visual occlusion** (camera-relative; video only);
   - **proximity encounter** (ground plane; label-symmetric);
   - **identity ambiguity** (joint event; §4.3).
   - For FIFA training candidates, use *proximity encounter with a video-calibrated ambiguity risk*, not "occlusion".
   - The closest FIFA analogue of a real occlusion is a **provider visual-loss burst next to another player** (§3.3, method F). It is the preferred realistic-gap evaluation stratum.
5. **Temporal, spatial, or combined thresholds?** Use synchronised spatial proximity as a generous **generator**, plus a learned or Kalman-based **risk** that includes relative velocity and uncertainty. Do not add an arrival-time tolerance on top of distance (redundant, §6 C).
6. **How should uncertainty affect thresholds?**
   - VISIBLE: there is no basis for widening (jitter ~1 cm).
   - ESTIMATED: widen probabilistically (C3), with σ as an explicit, sensitivity-tested assumption.
   - Never let ESTIMATED events into the T1 metric.
7. **Redo Step 1?** **No.** Extend with a quality-flag sidecar and tiers (§7).
8. **Large population without artifacts?** Generate broadly, never delete except clones, score and tier, weight in training, count independent events, and report everything by tier.
9. **Formulation:** offline **local bipartite assignment** (K = 2 → K ≤ 4) with calibrated confidence and abstention. Online is a separate later task.
10. **First baseline:** a **Kalman constant-velocity likelihood + Hungarian assignment**, with quality-dependent measurement noise, on label-symmetric candidates with randomised synthetic gaps. Then a GBM on symmetric features.
11. **Advanced methods worth testing:** a GBM with the Kalman likelihood (hybrid); then a Siamese GRU/TCN. A transformer or GNN only for K > 2 groups, and only if the GRU beats the GBM by a pre-registered margin.
12. **Controls against over/underfitting, bias and leakage:**
    - nested LOMO and grouped clusters;
    - label-symmetric selection with a 50/50 test;
    - ID-permutation invariance test;
    - a common evaluation population;
    - learning curves (data-size ablation) for under- and overfitting;
    - calibration plots;
    - per-tier reporting;
    - a frozen, one-time validation.
13. **Missing evidence:**
    - the absolute accuracy of FIFA positions (VISIBLE and ESTIMATED);
    - the meaning of `ESTIMATED` (PFF documentation);
    - relative-distance accuracy of the SoccerTrack calibration against BePro pitch positions;
    - human-verified identity labels on any video;
    - the FIFA broadcast camera geometry;
    - the advisor's acceptance of the definitions.
14. **Ask the advisor before major implementation:**
    - acceptance of the three-definition separation and of "risk-scored proximity" for FIFA;
    - use of SoccerTrack as the video calibration domain;
    - whether a BePro XML download (about 1 GB for 4 matches) is approved;
    - acceptance of synthetic gaps with provider labels as the FIFA training signal;
    - held-out policy option (b);
    - an annotation budget (about 25 person-hours) for the Wadi Degla W2 pilot.
15. **Single most useful next experiment:** **E1 (SoccerTrack identity-failure risk study)** in the execution plan. It costs no new data and no GPU, and it directly measures which camera-agnostic features predict real identity failures. It also measures how badly the 0.25 m rule performs as a detector of them.

---

## Appendix: diagnostic reproduction

Scripts live in the session scratchpad (`…/scratchpad/diag/`): `fifa_cadence.py`, `st_occlusion.py`, `fifa_short_est.py`. Run them with the
project venv (`python -I` for the FIFA scripts). Outputs: `cadence.json`, `st_all.json`, `short_est.json`. They are not
committed. If they should become project artifacts, move them into a versioned `data/processed/m11-redesign-diagnostics-v1/`
folder with a manifest; that needs approval.
