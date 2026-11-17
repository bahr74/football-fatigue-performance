# M11 identity preservation and recovery at player contacts — technical report

> **Formulation note (added after the advisor's correction, 2026-11-09).** Every result in this report uses the identity-free
> *contact* population (same-time distance ≤ 1 m), not the advisor's exact geometric intersection. They are kept as
> historical comparison artifacts and must not be quoted as advisor-intersection results. See
> [m11_advisor_intersection_correction.md](m11_advisor_intersection_correction.md).

Status: autonomous research run (2026-11-09), uncommitted. Living log and checkpoint: [m11_identity_recovery_progress.md](m11_identity_recovery_progress.md).

**Scope statement.** All results use FIFA World Cup 2022 PFF RAW *provider positions* with **synthetic** gaps (positions removed
for 0.4–2.0 s around a contact) and synthetic Gaussian position noise. The reference label is the provider's identity
continuity, not physical ground truth. No video, visual detection, visual tracking or real occlusion was tested. A geometric
crossing is not a visual occlusion and is never claimed as a confirmed identity swap.

## 1. Data and partitions

- Development (iterative work): 3812, 3814, 3815, 3816, 3829, 3833, 3836, 3847, 3851, 10506 — the ten development matches with
  Step 1 caches (`m11-encounters-v2`).
- Official validation (12 matches; caches `m11-validation-v1/encounters`): evaluated **once** after the model and thresholds were
  frozen. It was examined before in Step 6 of the historical pipeline, so it is not a pristine partition.
- Locked test (12 matches): never read. It was already consumed by Step 7. **No genuinely untouched independent test partition
  remains for this task**; this is reported as a limitation, and no previously examined match was renamed as a test set.
- Preserved unchanged: RAW data, Step 1–8 outputs, `m11-frozen-v1` (checksums verified), v1 and v2 intersection outputs, the
  Step 8 dashboard.

## 2. Stage S1.1 — temporal audit v3 (10 development matches)

`intersections/audit_v3.py`: the v2 audit plus one predefined rule. A speed spike (> 12 m/s, the Step 1 physiological bound)
landing within ±0.5 s of a player's crossing makes the event *uncertain* (reason `speed_spike_near_crossing`). On 3812 the v1
events are reproduced byte for byte, and every v2 difference is explained by the spike rule. Output: `m11-intersections-v3`.

## 3. Stage S1.2 — identity-free contacts (selection-bias correction)

Selecting samples by geometric path intersection leaks the label: whether two paths cross is computed from identity-labelled
tracks. A tracker at a real occlusion does not know whether the players crossed or turned back. Samples are therefore defined by
an identity-free **contact**: two players whose same-time distance falls to ≤ 1.0 m. The episode lasts while the distance stays
≤ 1.5 m, and t* is the time of minimum distance. Exchanging the two players' labels after t* leaves the contacts unchanged
(tested). Crossing geometry is kept as a stratum. The advisor's time-aligned crossings (|Δt| ≤ 0.20 s) are a reported subset.
Output: `m11-contacts-v1` (same audit categories and the lossless ±2.5 s trajectory store).

## 4. Training views and scenarios

- Update instants are resampled onto a common 0.1 s grid around t*; bridging is allowed only for update gaps ≤ 0.5 s. Identical
  grids for both players remove the 15 Hz phase and held-position shortcuts.
- Each scenario removes both players' positions within a gap g ∈ {0.4, 0.8, 1.2, 1.6, 2.0} s centred on t* and adds i.i.d. noise
  σ ∈ {0, 0.15, 0.3} m.
- The order of the before tracks and, independently, of the after tracks comes from a seeded hash. The label is y = 1 when
  {before 0 → after 0, before 1 → after 1} is the provider continuity.
- Normalisation uses the before tracks only. Inputs never include identities, visibility or confidence.

## 5. Models

- **Motion baselines** (8): nearest position; constant velocity forward (0.5 s and 1 s fits); bidirectional bridging to the gap
  centre (CV and CA, short and long spans). Each cost margin is turned into a probability by a logistic fit on the training
  matches; the fit may invert the rule's direction.
- **GBM** (`HistGradientBoostingClassifier`, fixed hyperparameters): 36 features, being antisymmetric permutation differences of
  the baseline costs and kinematics, plus symmetric context. It is trained with after-order swap augmentation and predicts
  symmetrically, so swapping the after tracks gives exactly 1 − p.
- **Hybrid PairNet** (GRU encoders of each track plus physics pair features): logit = s00 + s11 − s01 − s10. This is the
  one-to-one assignment score, which extends to N players with the Hungarian algorithm. Training adds reflection and after-swap
  augmentation, and early-stops on one training match.

## 6. Protocol

Pre-registered in the progress log before any learned-model result:

- leave-one-match-out over the 10 development matches;
- pooled and per-match scores with 95 % match-bootstrap intervals;
- selection: the simplest model not significantly worse in log loss (paired match bootstrap);
- operating thresholds frozen from development out-of-fold predictions;
- one official-validation run.

**Metrics.** Accuracy on all samples; the abstention metrics *coverage*, *false-association rate* (wrong committed / committed)
and *recovery rate* (correct committed / all). Accuracy on committed predictions alone is never reported as accuracy.

## 7. Samples

| | development (10 matches) | official validation (12 matches) |
|---|---|---|
| contacts | 24,685 | 28,541 |
| eligible interactions (clean + uncertain @0.20 s; 998 invalid development artefacts excluded) | 23,687 | 27,181 |
| scenario rows (interaction × 15 scenarios) | 355,305 | 407,712 |
| clean / flagged rows | 75,585 / 279,720 | 85,890 / 321,822 |

Geometry of the development candidates: 46 % no path crossing, 36 % crossing not time-aligned, 18 % advisor aligned crossings
(64,650 rows). Leakage control is match-level: every fold holds out a whole match, so overlapping windows, shared players,
repeated pair clusters and duplicate segments of a match never straddle training and evaluation.

## 8. Development results (leave-one-match-out, pooled out-of-fold, 95 % match-bootstrap CI)

Operating points use thresholds fitted on these pooled predictions (descriptive for every model; frozen only for the
selected model). Coverage / recovery = committed share / correct committed share of all rows.

| model | accuracy [CI] | log loss | ECE | cov @FA1 % | cov / rec @FA5 % | cov / rec @FA10 % |
|---|---|---|---|---|---|---|
| nearest (fitted; direction inverted by the fit) | 0.552 [0.536, 0.569] | 0.675 | 0.008 | 0.000 | 0.000 / 0.000 | 0.018 / 0.016 |
| cv_fwd | 0.565 [0.549, 0.582] | 0.688 | 0.044 | 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| cv_bidir | 0.518 [0.491, 0.538] | 0.694 | 0.017 | 0 | 0 / 0 | 0 / 0 |
| ca_bidir | 0.553 [0.544, 0.561] | 0.687 | 0.026 | 0.000 | 0.000 / 0.000 | 0.001 / 0.001 |
| cv1_fwd | 0.549 [0.528, 0.571] | 0.694 | 0.047 | 0 | 0 / 0 | 0 / 0 |
| cv1_bidir | 0.480 [0.459, 0.501] | 0.692 | 0.044 | 0 | 0 / 0 | 0 / 0 |
| ca2_fwd | 0.565 [0.552, 0.578] | 0.687 | 0.044 | 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| ca2_bidir | 0.526 [0.511, 0.540] | 0.693 | 0.020 | 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| **GBM (selected)** | **0.764 [0.754, 0.773]** | **0.455** | **0.008** | **0.187** | **0.378 / 0.359** | **0.544 / 0.489** |
| hybrid PairNet | 0.769 [0.759, 0.778] | 0.453 | 0.015 | 0.146 | 0.388 / 0.368 | 0.560 / 0.504 |

- **Motion rules alone fail at contacts.** The players are close, often slow and jostling, so extrapolation is ambiguous.
  The raw nearest-position rule agrees with the provider 0.448 of the time; the fit inverts it to 0.552.
- **Selection** (pre-registered rule): GBM vs hybrid log-loss difference +0.002 [−0.002, 0.006], which includes 0, so the
  simpler GBM is selected. The hybrid is 0.5 pp more accurate ([0.2, 0.8] pp) but has lower coverage at the 1 % target.
- **GBM by gap** (accuracy; coverage at FA 5 %): 0.4 s 0.845 / 0.587; 0.8 s 0.776 / 0.388; 1.2 s 0.755 / 0.348;
  1.6 s 0.731 / 0.295; 2.0 s 0.713 / 0.272.
- **GBM by noise**: 0 m 0.805; 0.15 m 0.767; 0.3 m 0.720.
- **GBM per match**: accuracy 0.735–0.783; false association at the pooled 5 % threshold 0.043–0.065.
- **GBM by tier**: clean 0.748, flagged 0.768. Flagged contacts are not harder for the model. Many are ESTIMATED-heavy, and
  the estimated-share = 1 stratum reaches 0.810: provider-interpolated tracks are smoother and easier to continue, which is a
  provider artefact that will not transfer to visual tracking.
- **GBM by geometry**: no crossing 0.768, crossing not aligned 0.767, advisor aligned crossings 0.748.
- **Leak oracle (invalid predictor).** "Assume they crossed" uses the identity-derived crossing flag. It scores 0.813 on the
  advisor subset and 0.668 overall when applied as "anti-nearest when crossed, nearest otherwise". GBM does not use the flag.
- **Feature audit** (permutation importance, fold 3812): the largest drops come from the nearest-position distance features
  (−0.069), then the long-span constant-velocity bridging and separation / gap context. The model learns from motion context
  when the players have passed each other. No identity, time-index or ordering feature exists, and swapping the after tracks
  gives exactly 1 − p (tested).
- **Gap-centre robustness (t\* uses positions inside the gap).** The gap was shifted randomly by up to ±0.4 s, with sample times
  re-centred on the gap, while training used unshifted views. GBM scored 0.761 vs 0.764 (log loss 0.463 vs 0.455). Placing the
  gap at the minimum-distance time is therefore not what drives performance.

## 9. Official validation (frozen models and thresholds, one run)

Selected model `gbm-v1`, trained on all 10 development matches; thresholds frozen from development out-of-fold predictions.

| model | accuracy [CI] | log loss | ECE | FA1 % target: cov / FA | FA5 % target: cov / FA / recovery | FA10 % target: cov / FA |
|---|---|---|---|---|---|---|
| **gbm-v1 (selected)** | **0.767 [0.762, 0.771]** | **0.452** | **0.010** | **0.184 / 0.009** | **0.377 / 0.048 [0.045, 0.052] / 0.359** | **0.545 / 0.099** |
| hybrid-v1 (comparator) | 0.779 [0.775, 0.784] | 0.441 | 0.025 | 0.173 / 0.009 | 0.440 / 0.057 [0.053, 0.061] / 0.415 | 0.611 / 0.106 |
| ca_bidir-v1 (comparator) | 0.556 [0.548, 0.564] | 0.687 | 0.030 | 0.000 / – | 0.000 / – / 0.000 | 0.001 / 0.104 |
| nearest-v1 (addendum, see note) | 0.547 [0.535, 0.558] | 0.678 | – | 0.000 / – | 0.000 / 0.153 / 0.000 | 0.016 / 0.119 |

- **Transfer.** The frozen 5 % operating point transfers: validation false association is 0.048, with the CI reaching 0.052.
  Per match it ranges from 0.037 to 0.059; 3817 (0.059), 3832 (0.055), 3825 (0.054) and 3841 (0.052) exceed 5 %, and none exceeds 10 %.
- **Strata.** By gap: 0.849 / 0.779 / 0.760 / 0.731 / 0.714. By noise: 0.808 / 0.771 / 0.721. Advisor aligned crossings 0.757.
- **Hybrid.** It is better on validation (log loss −0.011 [−0.015, −0.006]), but its frozen 5 % point overshoots (FA 0.057).
  By the pre-registered protocol this does **not** change the selection; it is recorded as evidence for a future protocol.
- **Addendum (disclosed).** The baseline comparator in the main run (`ca_bidir`) was mislabelled as "best by development log
  loss"; that is the fitted `nearest` (0.675). `nearest-v1` was frozen afterwards and evaluated under the separate tag
  `validation/addendum-nearest-baseline`. The selected model's run was not repeated or changed.

## 10. Recovery figures

`data/processed/m11-recovery-v1/figures/e1-baselines-gbm-gbm/{correct,wrong,abstained}.png`: seeded random examples of
each outcome, from development out-of-fold GBM predictions at the 5 % operating point. Several confident "wrong" examples look
like plausible provider identity inconsistencies. They are disagreements with the reference, candidates for human review, not
confirmed provider errors.

## 11. Limitations

1. **No visual tracking was tested.** Gaps and noise are synthetic. Real occlusion gaps are not centred on the
   minimum-distance time and come with detector jitter and box merges. Provider positions inside contacts are partly
   interpolated (ESTIMATED), and the model benefits from that (estimated-share = 1 stratum 0.81).
2. **The label is provider continuity**, which may itself be wrong at contacts. No human annotation exists.
3. **No pristine test.** Validation was examined in Step 6 and the locked test is consumed, so the validation figures are
   the best available out-of-sample estimate but not a pristine one.
4. **Two players only.** The N-player case (group contacts) is not evaluated. The hybrid's pair score extends to the
   Hungarian algorithm; GBM would need pairwise reduction.
5. **Bounds.** Gaps above 2.0 s or below 0.4 s are outside the trained domain; the API abstains on them.
6. **Thresholds.** They come from leave-one-match-out models but are applied to a model refitted on all 10 matches; the
   validation run shows that this transfers (0.048 vs 0.05).

## 12. Readiness for visual tracking

Not ready for production claims; ready for a controlled pilot. The recovery model is a reasonable *association prior* to plug
into the tracker's handover after occlusion: two before tracks and two after tracks in pitch metres, an abstention-aware
decision, ~36 % recovered at 5 % false association with the rest left for review. Before any claim:

1. a test on real tracker gaps in pitch coordinates (calibration quality limits this; see M8.1);
2. human-annotated identity at contacts;
3. a re-check of the operating point on that data.

## 13. Reproduction

```powershell
.\.venv\Scripts\python.exe -m football_fatigue.intersections.audit_v3 --matches 3812 3814 3815 3816 3829 3833 3836 3847 3851 10506
.\.venv\Scripts\python.exe -m football_fatigue.intersections.contacts --matches 3812 3814 3815 3816 3829 3833 3836 3847 3851 10506
.\.venv\Scripts\python.exe -m football_fatigue.m11_recovery.experiments --exp e1-baselines-gbm --models nearest cv_fwd cv_bidir ca_bidir cv1_fwd cv1_bidir ca2_fwd ca2_bidir gbm
.\.venv\Scripts\python.exe -m football_fatigue.m11_recovery.experiments --exp e2-hybrid --models hybrid
.\.venv\Scripts\python.exe -m football_fatigue.m11_recovery.experiments --exp r1-shifted --models gbm --shifted
.\.venv\Scripts\python.exe -m football_fatigue.m11_recovery.final freeze --exp e1-baselines-gbm --model gbm
.\.venv\Scripts\python.exe -m football_fatigue.m11_recovery.final validate --models gbm-v1 hybrid-v1 ca_bidir-v1   # once; refuses to rerun
.\.venv\Scripts\python.exe -m football_fatigue.m11_recovery infer --model gbm-v1 --input tracks.json
.\.venv\Scripts\python.exe -m football_fatigue.m11_recovery dashboard          # http://127.0.0.1:8502
```
