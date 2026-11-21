# Paper experiment map, novelty-gap analysis and advisor memo (v1)

**Date:** 2026-11-20

**Status:**
- Planning material for the advisor discussion.
- **No novelty claim is proven, and no paper scope is chosen.**
- All quantitative statements about this project are development-only (32 train/CV matches).
- No held-out or official evaluation exists for any new result.

**Starting point:**
- the verified literature table in `M11_VIDEO_GROUNDED_OCCLUSION_RESEARCH.md` §11;
- the Phase 2 reports:
  - `m11_phase2_quality_audit_v1.md`
  - `m11_phase2_ambiguity_benchmark_v1.md`
  - `m11_dataset_v2_design.md`
  - `m11_four_model_benchmark_readiness.md`
  - `m11_session_ab_interface_v1.md`

## 1. Literature basis

Bibliographic facts were checked today against arXiv / proceedings pages, or are in the repository's verified table. **Results quoted
from papers are as reported by their authors and were not reproduced here.**

| Work | Verified reference | What it does (as reported) | Limitation relative to this project |
|---|---|---|---|
| SORT (Bewley et al.) | ICIP 2016; arXiv:1602.00763 | Kalman filter + Hungarian assignment on detections | Linear motion; no reasoning about interaction ambiguity |
| OC-SORT (Cao et al.) | CVPR 2023; arXiv:2203.14360 | Observation-centric re-update, momentum and recovery to correct Kalman error accumulated during occlusion | Box / IoU-based. Our `ocm_dir` borrows only the direction-consistency idea and is **not** OC-SORT |
| Multiple Hypothesis Tracking (Reid) | IEEE TAC 24(6):843–854, 1979 | Maintains and prunes data-association hypotheses with Kalman estimates | The template for *delayed* association; not evaluated on football close contacts here |
| GTA, Global Tracklet Association (Sun et al.) | arXiv:2411.08216 (ACCV 2024 workshop) | Appearance-based tracklet splitting and joining as post-processing; reports SportsMOT / SoccerNet HOTA gains | Appearance-driven, box-level; no motion-only association study; no pitch coordinates |
| SoccerNet Game State Reconstruction (Somers et al.) | CVPR 2024 Workshops; arXiv:2404.11335 | Task and dataset: track and identify athletes on a pitch minimap from single-camera video; GS-HOTA metric | Pitch-space end-to-end benchmark; identity via jersey / ReID, not a motion-ambiguity analysis |
| SportsMOT (Cui et al.) | ICCV 2023; arXiv:2304.05170 [repo-verified] | Sports MOT dataset; association is the bottleneck | Pixel space, broadcast clips |
| Deep-EIoU (Huang et al.) | arXiv:2306.13074 [repo-verified] | Expansion IoU without Kalman, plus ReID | Our replay: −0.053 HOTA vs ByteTrack [DOC M9] |
| MPNTrack / SUSHI (Brasó, Cetintas, Leal-Taixé) | arXiv:1912.07515 / 2212.03038 [repo-verified] | Graph neural association | Pedestrian MOT; template for N × M scorers |
| FOOTBALLTrace (Mostafa et al.) | doi:10.5220/0012226800003587 [repo-verified] | Windowed-distance candidates, operator confirmation, tail exchange | Image units, small n; the correction stage was at chance on FIFA in our reproduction [DOC] |
| Graph Imputer (Omidshafiei et al.) | Sci. Rep. 2022 [repo-verified] | Off-screen imputation with a simulated camera | Imputation, not identity association; a template for realistic loss simulation |
| Selective classification (Geifman & El-Yaniv) | NeurIPS 2017; arXiv:1705.08500 | Risk–coverage guarantees by abstention on a held-out calibration set | Provides the statistical framing for abstention / delayed decisions |

**Not verified, so not cited with numbers:**
- GTATrack;
- any claim that motion-only identity re-association in metric football tracking has or has not been published.

**A systematic search for directly competing work (football tracking-data identity re-association) is still required [REC].**

## 2. Evidence already available (development only)

| Evidence | Value | Source |
|---|---|---|
| Deduplicated multi-radius interaction catalogue; reproduces earlier datasets exactly | 45,989 edges at 0.5 m … 216,551 at 2 m | Phase 1, Task A |
| Transitive group inflation quantified | 60 % of connected 3+ groups at 0.5 m never have 3 players at once | Task A §2 |
| Membership sensitivity of small radii | 26 % of 0.5 m edges change under a 5 cm bias | Task A §5.5 |
| Provider status confound | ESTIMATED contacts are smoother (median max accel 5.1 vs 8.8 m/s²) and easier for physics (kf_fwd 0.745 vs 0.558 for mixed) | Tasks A, B |
| Physics difficulty at 0.5 m / 0.8 s | Best deterministic 0.667 BA; 10 % physics-hard; 23 % physics-easy | Task B |
| Gap and centring effects | Best BA 0.86 at 0.4 s, 0.59 at 1.6 s; bidirectional methods collapse when centred | Task B |
| Learned vs physics | GRU 0.900; learned models 87–91 % on physics-hard samples | Task B §6 (historical OOF) |
| Unflagged severe contradictions | 25 suspected provider swaps | Task B §7 |
| Historical model comparison not fair | Budget-bound, untuned networks; boundary-bound grids | Task D |

## 3. Candidate contributions

**Scores:** 1 (low) – 5 (high). They are judgements [HYP], not measurements.

| # | Contribution | Potential novelty | Significance | Feasibility | Effort | Independent validation needed |
|---|---|---|---|---|---|---|
| 1 | **Motion-only identity re-association benchmark** on metric football tracking, with a deduplicated interaction catalogue and physics-difficulty strata | 3 | 4 | **5** | Low–medium (fair re-benchmark + v1/v2 data) | Held-out 8 matches once; competing-work search |
| 2 | **Interaction-aware multi-player (N × M) recovery** with null and distractors | **4** | 4 | 2–3 | High (schema B, per-link scorers, new models) | Held-out, plus ideally video GT |
| 3 | **Ambiguity-aware and delayed re-association** (abstain or defer when physics and learned evidence conflict; MHT-style) | 3–4 | 4 | 3 | Medium (calibration partition, risk–coverage, delay policy) | Held-out; calibration on separate matches |
| 4 | **Provider interpolation and trajectory-reliability analysis** (ESTIMATED texture, clone-like pairs, label swaps) | 3 | 3–4 (data-validity contribution for every user of the dataset) | **4** | Low–medium (human audit of about 125 events is the key cost) | Human audit; ideally a video cross-check |
| 5 | **Selective prediction under distribution shift** (PFF → noisier / video-like inputs) | 2–3 | 4 | 2 | Medium–high (needs a realistic shift: a calibrated video clip) | Video data with identity GT |

### Novelty-gap matrix

| Existing method / work | Limitation | Possible project contribution | Evidence already | Missing experiment | Required baseline | Publication risk |
|---|---|---|---|---|---|---|
| SORT / Kalman + Hungarian | Linear motion; fails at crossings | Quantify where and why it fails in football (Task B strata) | kf_fwd 0.667; physics-hard 10 % | Variable centre / gap; v2 population | Kalman fwd / bidir, CA, nearest (done) | Low novelty if presented as the contribution itself |
| OC-SORT | Box-based; designed for occlusion in video | Test whether observation-centric ideas help in metric space | `ocm_dir` 0.643, the most robust at 1.6 s (0.587) | A faithful OC-SORT adaptation is not done | OC-SORT on video (Session B) | Mis-labelling a partial re-implementation |
| MHT | Hypothesis explosion; tuning | Delayed decision for ambiguous contacts | Physics vs learned disagreement measurable | Delay and abstention policy evaluation | Greedy / Hungarian immediate decision | Medium |
| GTA / ReID trackers | Appearance-dependent; pixel space | Motion-only complement for same-kit or occluded players | — | Integration on a calibrated clip | GTA-style post-processing | High (needs video) |
| SoccerNet GSR | Pitch-space end-to-end; identity via jersey | Motion association as a GSR component | — | GSR evaluation | GSR baseline | High (large engineering) |
| Graph MOT (MPN / SUSHI) | Pedestrian MOT | N × M per-link scorers with group context | Catalogue groups (1,199 simultaneous 3+ at 0.5 m) | Schema B build; models | Hungarian + Kalman with null (implemented) | High effort |
| Provider tracking data papers | ESTIMATED semantics undocumented | Reliability analysis and artefact flags | Task A §4; Task B status strata | Human audit; video cross-check | — | Medium (may be seen as dataset-specific) |
| Selective classification | General ML | Risk–coverage for identity decisions | GRU ECE ~0.008 on OOF [DOC] | Separate calibration matches; shifted inputs | Confidence thresholding | Medium |

## 4. Experiment roadmap (by decision gate; nothing launched)

| Step | Experiment | Serves | Cost | Gate |
|---|---|---|---|---|
| R0 | Human audit: 100 review events + 25 severe failures | 1, 4 (label validity) | About 1–2 reviewer days | — |
| R1 | Fair four-family re-benchmark, centred, priority 1 (Task D) | 1 | About 36 GPU-h + CPU | Approval of protocol and budget |
| R2 | Physics-tier × status × texture-ablation analysis of R1 | 1, 4 | CPU | After R1 |
| R3 | Variable centre / gap / asynchrony evaluation (no new training; Stage 3 tables + v2 views) | 1, 3 | CPU | Dataset v2 pilot gate |
| R4 | Calibration-partition abstention / delay study | 3, 5 | CPU on R1 outputs | After R1 |
| R5 | Dataset v2 build (strategy 3) and re-benchmark | 1, 2 | About 50–300 GPU-h | Dataset v2 gates G-v2-0…5 |
| R6 | Schema B pilot (N × M with null), Kalman / Hungarian first, then per-link scorers | 2 | High | Advisor scope decision |
| R7 | Video integration test on one calibrated clip with identity GT | 5, external validity | High | Session B readiness + calibration QC |
| R8 | **One-time held-out evaluation** of frozen configurations | All | Low | Everything above frozen and pre-registered |

## 5. Advisor discussion memo (one page)

**Where we are:**
- **The catalogue counts are correct and deduplicated.** Interactions are dominated by repeated contacts of the same pairs. Most
  close contacts are provider-ESTIMATED.
- **The 0.5 m population is physically hard.** The best deterministic Kalman / Hungarian baseline reaches 0.667 BA, and the
  historical GRU 0.900.
- **About 10 % of encounters defeat every physical model.**
- **Learned models solve most of those (87–91 %).** We do not yet know whether this reflects motion understanding or provider
  artefacts. This is the central scientific risk.
- **The historical GRU-vs-LSTM comparison is not fair.** Both networks were budget-bound and untuned, and the tabular grids were
  boundary-bound.

**Decisions requested:**
1. **Scope of the first paper.** Our ranking of feasibility × significance is 1 > 4 > 3 > 2 > 5:
   - candidate 1, the motion-only benchmark;
   - with candidate 4, reliability, as a necessary companion section;
   - and candidate 3, abstention / delay, as the methodological extension.

   Multi-player N × M (2) as a follow-up?
2. **Population.** Keep v1 (0.5 m) as the primary, pre-registered population, and add a stratified v2 (strategy 3, weights reported)
   as the generalisation study?
3. **ESTIMATED contacts.** Headline on all data stratified by status, plus a VISIBLE-only reference analysis?
4. **Human audit.** Approve the 125-event audit before any label-quality or artefact claim?
5. **Compute.** Approve about 36 GPU-h for the centred fair re-benchmark first, deferring shift augmentation (about 255 GPU-h)?
6. **Held-out policy.** Confirm the 8 internal held-out matches stay untouched until R8.

**Claims we will not make:**
- visual occlusion from tracking data;
- video readiness of the current models;
- novelty before a systematic search for competing work;
- any result on held-out data.
