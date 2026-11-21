# Dataset v2: scientific design (proposal; nothing built or frozen)

**Date:** 2026-11-20

**Status:** a design for advisor discussion.
- No dataset v2 sample was built, selected or frozen.
- Every example below is synthetic.
- Quantitative inputs come from Phase 1 (catalogue) and Phase 2 Tasks A and B, development matches only.

**Labels:**

| Label | Meaning |
|---|---|
| [V] | Measured |
| [EST] | Estimate |
| [HYP] | Interpretation |
| [REC] | Recommendation |

---

## 1. Why v2 is needed (evidence)

| Limitation of Stage 3 v1 (0.5 m, 0.8 s, 2 × 2, centred, simultaneous) | Evidence |
|---|---|
| Population defined by a noise-level radius | A 5 cm bias changes about 26 % of 0.5 m membership [V, Task A §5.5] |
| Difficulty depends on the centring convention | Bidirectional baselines 0.55 centred vs 0.69 shifted [V, Task B §2] |
| Most physically hard encounters lie outside 0.5 m | About 8 k physics-hard edges at 2 m vs 3.5 k at 0.5 m [EST, Task B §5] |
| Only correct candidates are offered | No distractors, no null; multi-player context is 11–47 % of edges depending on R [V] |
| Hiding is simultaneous and symmetric | Video loss is asynchronous (interface §5.7) |
| ESTIMATED-dominated close contacts | 67 % of 0.5 m edges are both ESTIMATED, and smoother [V] |
| Possible unflagged identity swaps | 25 severe motion contradictions with no label flag [V, Task B §7] |

## 2. Taxonomy: keep these phenomena separate

| Term | Definition in this project | Observable in PFF tracking? | Observable in video? |
|---|---|---|---|
| **Pairwise geometric interaction** | Two players within R metres (edge), or their paths crossing (crossing) | Yes (catalogue) | Only after calibration |
| **Candidate identity ambiguity** | Given what is observed before and after an interval, more than one assignment is plausible under a stated motion model | Yes, model-relative (`cv_margin`, physics tiers) | Yes, model-relative |
| **Synthetic observation loss** | We hide samples that *were* observed in order to create a supervised task | Yes: we create it | Yes: we create it |
| **Real visual occlusion** | A player is not visible in the image because another body or object blocks the line of sight | **No** (ESTIMATED is provider metadata, not verified occlusion) | Only with human annotation |
| **Detector failure** | No detection although the player is visible (blur, scale, threshold) | No | Yes (vs GT) |
| **Track fragmentation** | One player's track is split into several tracklets | Provider chain breaks (label-risk flags) | Yes |
| **Identity switch** | A tracklet continues with the wrong player | Suspected only (jumps, severe motion contradictions) | Yes (vs GT) |
| **Re-identification** | Re-linking a dormant identity to a new tracklet after a long absence or leaving the view | Not studied (gaps ≤ 1.6 s) | Yes |

**Rule:** a dataset or result may only name the phenomenon it actually controls. Synthetic hiding in PFF data is *observation loss*,
not occlusion.

## 3. Candidate schema A: `pairwise-association-v2`

**Purpose:** two incoming and K ≥ 2 outgoing candidates; association with null. A strict generalisation of Stage 3.

### 3.1 Unit and sampling

**Anchor:** a catalogue edge chain at radius R_anchor ≤ 2 m (`edge_chain_id`, a stable hash).

**Per anchor, one or more *views* (variants of the same encounter):**
- gap length g ∈ {0.2, 0.4, 0.8, 1.2, 1.6, 2.4} s;
- gap centre offset δc ∈ [−0.5, 0.5] s relative to t\*;
- per-player asynchrony: start offsets Δ0_A, Δ0_B and end offsets Δ1_A, Δ1_B ∈ [0, 0.6] s;
- partial loss: one player never hidden (p = 0.15);
- distractors: up to K − 2 other players whose first observation after the gap lies within a search radius;
- null: with probability p_null, one true outgoing tracklet is removed (player "leaves view"), so the correct answer is *unmatched*.

**Views are never independent:** `event_id` groups every view of an anchor, and `dependency_id` groups anchors (see §5).

### 3.2 Tensors (model inputs only)

| Field | Shape / type | Notes |
|---|---|---|
| `in_pos` | (2, T_in, 2) float32 | Sample frame (origin and rotation from incoming data only, as in Stage 3) |
| `in_mask` | (2, T_in) bool | |
| `in_t` | (2, T_in) float32 | Seconds relative to that track's own last observation, so asynchronous loss stays honest |
| `out_pos` | (K, T_out, 2) | Outgoing candidates in a **random order** (seeded by `event_id`, view) |
| `out_mask` / `out_t` | (K, T_out) | Relative to the earliest first observation among candidates |
| `cand_valid` | (K,) bool | Padding for a variable K |
| `gap_in` | (2,) float32 | Each incoming track's time since last observation at the decision time |
| `noise_sigma_m` | float32 | Only if noise was injected **and** it would be known at deployment (calibration σ); otherwise absent |

Missing values: masked steps carry zeros plus mask 0. **NaN never enters a tensor.** T_in = T_out = 30 at 15 Hz, the same as v1.

### 3.3 Targets

| Field | Type | Meaning |
|---|---|---|
| `assign` | (2,) int8 | Index of the outgoing candidate for each incoming track, or −1 = null (left view) |
| `y_pair` | int8 | For 2 × 2 views only: Stage 3 convention, 0 MATCHED / 1 SWAPPED |

**Labels are generated from provider chains:**
- Identity = provider slot.
- The chain must be continuous over [first incoming sample, last outgoing sample] for every true player.
- `label_quality` comes from the catalogue `label_risk` reasons.

### 3.4 Metadata (evaluation only; never an input)

| Group | Fields |
|---|---|
| Ids | `match_id`, `period`, `event_id` (= anchor edge-chain id), `view_id`, `dependency_id` (connected group id at δ = 1 s), `pair_period_id` |
| Split | `split` (`train_cv`), `outer_fold` (frozen Stage 4 folds), `role` ∈ {fit, early_stop, calibration, test} per outer fold |
| Geometry | R_anchor, d_min, relative speed, crossing descriptors (`abs_dt_s`, `d_sync_m`), duration |
| Context | Simultaneous group size, sequential partners, distractor count and distances |
| Quality | `status_composition`, ESTIMATED fractions, `reliability_reasons`, `label_risk_reasons`, clone-pair flag, co-motion flag, boundary-uncertain flag |
| Difficulty (label-derived) | `cv_margin_m`, `cv_tier`, physics tier (Task B definition) |
| Provenance | Catalogue run hash, code hash, view seed |

### 3.5 Synthetic example (one view)

```json
{"event_id": "ec-9f1c2a7b", "view_id": 3, "dependency_id": "m3812-p1-g1-0412", "match_id": 3812, "outer_fold": 4,
 "inputs": {"in_pos": "float32[2,30,2]", "in_mask": "bool[2,30]", "in_t": "float32[2,30]",
            "out_pos": "float32[3,30,2]", "out_mask": "bool[3,30]", "out_t": "float32[3,30]", "cand_valid": [true, true, true],
            "gap_in": [0.93, 0.80]},
 "target": {"assign": [2, -1]},
 "meta": {"R_anchor": 1.0, "d_min": 0.62, "gap_s": 0.8, "centre_offset_s": 0.2, "asynchrony_s": [0.13, 0.0],
          "distractors": 1, "null_injected": true, "status_composition": "mixed", "cv_tier": "marginal",
          "label_risk_reasons": "", "clone_pair": false}}
```

## 4. Candidate schema B: `multiplayer-association-v1`

**Purpose:** N incoming × M outgoing tracklets in one group situation, with null and dormant ids. This is the multi-player study.

| Item | Specification |
|---|---|
| Anchor | A *simultaneous* group (δ = 0) with `max_simultaneous_players` ≥ 3 at R_anchor. Connected groups are **not** used as anchors because of transitive inflation (Task A §2) |
| Players | Every player in the simultaneous component at any instant, plus distractors within a radius |
| Hiding | Per-player intervals drawn from an empirical loss model (start/end offsets, durations), with at least two players hidden overlapping |
| Inputs | `in_pos` (N, T, 2), `out_pos` (M, T, 2), masks and times; `dormant` (D, 3): last position and time of ids hidden longer than the window |
| Target | `assign` (N,) ∈ {0..M−1, −1}; one-to-one constraint |
| Evaluation units | Group (`event_id`), dependency id, match |
| Metrics | Exact-assignment rate, pairwise link precision/recall, null precision, IDF1-style per-group identity F1, risk–coverage under abstention |

**Synthetic example:**
```json
{"event_id": "g0-3812-p1-r1p00-0077", "N": 3, "M": 3, "D": 1,
 "target": {"assign": [1, 0, -1]},
 "meta": {"max_simultaneous_players": 3, "n_home": 2, "n_away": 1, "hidden_intervals_s": [[0.0, 0.9], [0.1, 0.8], [0.0, 1.4]],
          "dormant_ids": ["slot-14"], "status_composition": ["both_estimated", "mixed", "mixed"]}}
```

**Counts available [V, Phase 1]:** simultaneous 3+ groups number 1,199 at 0.5 m, 4,313 at 1 m and 12,335 at 2 m. Their connected
groups (independence units) are fewer.

## 5. Leakage and dependency controls

| Variable | At prediction time | Stratified evaluation only | Never |
|---|---|---|---|
| Observed positions before / after, masks, times relative to own track | ✔ | | |
| Gap length per track | ✔ (known when the track reappears) | | |
| Calibration uncertainty σ (video) | ✔ | | |
| Positions inside the hidden interval | | | ✔ |
| Provider ESTIMATED / VISIBLE status | | ✔ | ✔ as input (texture leak; it does not exist in video) |
| Slot, player id, jersey, team, chain id | | ✔ (team relation as a stratum) | ✔ as input |
| t\*, d_min, cv_margin, physics tier, crossing class | | ✔ | ✔ as input (they use the hidden interval or the label) |
| Candidate order | Random (seeded) | | Never sorted by a label-derived key |
| Group membership computed from hidden-interval positions | | ✔ | ✔ as input |

**Grouping:**
- Every view of one anchor shares `event_id`.
- Anchors that share a player within 1 s share `dependency_id`.
- Folds are by **match** (frozen outer folds).
- Inside a fold, early-stop and calibration partitions are also match-disjoint (Task D partition).
- Bootstrap CIs resample matches.
- Repeated pairs: report results also clustered by `pair_period_id`. Task A §3.2: 10 % of pairs hold 40 % of edges.

**Separation of roles per outer fold (Task D proposal, validated):**
- fit 22 matches: tuning by 3 inner folds;
- early-stop 3;
- calibration 3: abstention thresholds and probability calibration;
- test 4.

**The 8 internal held-out matches are touched once, after freezing.**

## 6. Sampling strategies compared

| | 1. Distance-only | 2. Physics-ambiguity-enriched | 3. Stratified mixed with population weights |
|---|---|---|---|
| Rule | All edge chains with d_min ≤ R | Anchors with `cv_tier` ∈ {confusable, marginal} (or physics tier ≠ easy) at R ≤ 2 m | Strata by (R band × cv_tier × status × context); fixed sampling fraction per stratum; weights w = 1 / fraction kept |
| Size at R ≤ 2 m [V edges] | 216,551 edges / 138,950 chains | 46,566 edges (confusable 24,369 + marginal 22,197) | Tunable, for example 40–60 k chains |
| Difficulty coverage | Diluted: confusable share falls 23 % → 11 % from 0.5 to 2 m | High | High, controlled |
| Representativeness | Representative of *the radius population*, which is itself arbitrary | **Biased**: train and test differ from deployment; selection uses the label (cv_margin) | Representative after reweighting; selection effect known exactly |
| Selection bias | Membership noise at small R (26 % per 5 cm) | Strong, and label-dependent (cv_margin uses provider identities) | Present but documented and invertible by weights |
| Leakage risk | Low | **High** if the tier is used at test time or to sort candidates | Low if tiers are metadata only |
| Deployment realism | Low (no trigger defines R in video) | Low (the tier is not observable before the answer) | Medium (needs a deployable trigger mapped to strata) |
| Statistical power for hard cases | Low per row | High | High |

**Recommendation [REC], not frozen: strategy 3.**
- Stratified mixed sampling over R ≤ 2 m.
- Physics-hard and confusable strata are **over-sampled for training**.
- **Evaluation is reported both on (i) the weighted population and (ii) each stratum.** Headline numbers must never come from an
  enriched set without weights.
- The label-derived tier is used *only* to choose sampling fractions and to stratify evaluation. It is never an input, and never a
  test-time filter.

**Deployment trigger.** A separate, label-free trigger must be defined and evaluated against the strata, for example:
- predicted-separation margin from the observed past only;
- the Kalman covariance;
- simultaneous loss in video.

That trigger is what a deployable system would use.

## 7. Realistic noise and loss (to be measured before use)

| Component | Source of the distribution | Status |
|---|---|---|
| Position noise | Phase 2 Kalman selection on PFF tracklets: σ_m 0.05 m [V]. Video calibration error: **unknown**, to be measured on a calibrated clip | Partial |
| Loss durations and asynchrony | Video tracker fragmentation statistics (Session B), or a camera-visibility simulation (Graph Imputer-style [DOC]) | Not measured |
| ESTIMATED texture | Ablate it (smooth / noise both statuses), or restrict to VISIBLE windows for a reference analysis | Ablations exist for GRU / HGB only [DOC] |
| Distractor density | Catalogue simultaneous / sequential context at 1–2 m [V] | Available |

## 8. Decision gates (proposed)

| Gate | Question | Pass criterion | Who decides |
|---|---|---|---|
| G-v2-0 | Is the taxonomy and the leakage table accepted? | Advisor sign-off | Advisor |
| G-v2-1 | Human audit of the 100-event review sample plus the 25 severe failures | Label-swap rate estimated with a CI; artefact strata confirmed or rejected | User + reviewer |
| G-v2-2 | Sampling strategy and stratum fractions fixed **before** building | Written specification with hashes | User + advisor |
| G-v2-3 | Pilot build on 2 development matches | All structural checks; leakage tests (no forbidden field reaches inputs; outgoing order random; view grouping) | Automated |
| G-v2-4 | Full development build (32 matches) | Counts per stratum within ±5 % of the plan; label verification 100 % | Automated + review |
| G-v2-5 | Deterministic baselines on v2 (no training) | Physics tiers reproduce the Task B ordering on the v1-equivalent subset | Automated |
| G-v2-6 | Approval to train | Only after G-v2-0…5 and the Task D protocol | User |

## 9. Open questions for the advisor

1. Is the N × M study (schema B) in scope for the first paper, or a follow-up?
2. Should ESTIMATED-dominated contacts be in the headline population, or only in a stratified secondary analysis?
3. What anchor radius cap should apply: 2 m, or larger, for long-gap and re-identification work?
4. Is a VISIBLE-only reference evaluation required? There are 2,674 independent both-VISIBLE samples at 0.5 m, about 48 k
   both-VISIBLE edges at 2 m.
