# M11 Stage 3: supervised MATCHED/SWAPPED dataset design (v1, design only)

Status: **design for approval. Nothing is implemented, no samples are built, nothing is trained.**

Inputs:
- `data/processed/m11-stage2-candidates-v1`: 40 development matches, with the outcomes, dependency groups and cohort states.
- The frozen split `m11-stage2-split-v1`: 32 train/CV matches and 8 held-out matches. It is never modified.

Measured counts are in `docs/research/m11_stage2_results_v1.md`.

## 1. From an accepted candidate to a sample

There is one sample per (candidate, gap), for every ACCEPTED candidate. LOWER_CONFIDENCE_IDENTITY candidates are built into a separate evaluation-only table and are never used for training (decision D1). REJECTED candidates are never built.

For candidate *c* with minimum time t*, players A and B on their candidate chains (`chain_a`, `chain_b`), and gap length g:

| Element | Interval (relative time τ = t − t*) | Source |
|---|---|---|
| incoming tracklet of A, of B | τ ∈ [−g/2 − C, −g/2) | the candidate chain of that player only |
| hidden gap | τ ∈ [−g/2, +g/2] | removed for both players (never stored in the sample) |
| outgoing tracklet of A, of B | τ ∈ (+g/2, +g/2 + C] | the candidate chain of that player only |

- **Context and gaps.** C is 2.0 s, as in Stage 2. The primary gap is 0.8 s; 0.4 s and 1.6 s are separate sensitivity tables. They are not pooled and not treated as independent data.
- **Continuity is guaranteed.** Stage 2 already requires one valid chain per player to cover [−g/2 − 0.5, +g/2 + 0.5]. Every tracklet therefore has at least 0.5 s of observation next to the gap, and the reference association across the gap is the continuity of that chain. Nothing is interpolated across missing or discontinuous regions.
- **Gap centre.**
  - The primary position is t*. t* is computed from the unordered pair of positions, so it carries no label.
  - A robustness table shifts the gap centre by a seeded offset in {−0.3, …, +0.3} s (step 0.1 s). This mirrors the existing `dataset.shift_for`, so a model cannot learn that "the gap is centred on the closest approach".
  - Shifted samples are built only where the continuity margin still holds for the shifted gap; that check is recomputed.

## 2. Schemas

**Model input (the only arrays a model may see).** Let T = round(C / Δ) points per tracklet.

| Field | Shape / type | Content |
|---|---|---|
| `in_pos` | (2, T, 2) float32 | incoming tracklets in a sample frame (below), metres; the last point is nearest the gap |
| `in_mask` | (2, T) bool | 1 = observed on the candidate chain |
| `in_dt` | (T,) float32 | time of each point relative to the gap start (negative) |
| `out_pos` | (2, T, 2) float32 | outgoing tracklets in the same frame; the first point is nearest the gap |
| `out_mask` | (2, T) bool | 1 = observed |
| `out_dt` | (T,) float32 | time relative to the gap end (positive) |
| `gap_s` | float32 | hidden-gap length |

**Target.** `y` int8: 0 = MATCHED (in0→out0, in1→out1), 1 = SWAPPED (in0→out1, in1→out0). This reverses the existing `m11_recovery` convention (`y = 1` means H0 = matched), so the reused code needs one explicit conversion. It is covered by a test.

**Metadata (never model input; for splits, strata and audits only).**
- Identifiers: `match_id`, `period`, `split`, `candidate_id`, `gap_s`, `shift_s`, `draw`.
- Dependency: `overlap_group`, `independent`.
- Strata: `crossing_class`, the F1–F4 `cohort_*` states, `d_min` band, slower chord-speed band, `team_relation`, `near_coincidence`, `jump_transfer_risk`.
- Status audit: `est_frac_used_{in0,in1}` and `est_frac_in_gap`.
- Ordering: `in_order` and `out_order` (the hash bits).

**Identity columns.** Slot, player id, jersey and team id are not written to the sample table at all. They stay in the Stage 2 candidate table, which the trainer never opens.

## 3. Temporal resampling and validity masks

- **Grid.** The proposed grid is Δ = 1/15 s, the native update rate, giving T = 30 points per tracklet at C = 2 s. The grid is anchored at the gap edges, so the in-grid ends exactly at −g/2 and the out-grid starts at +g/2.
  - Positions come from linear interpolation of the candidate chain's knots, the same geometry Stage 2 validated.
  - The provider's 29.97 fps duplicate frames are not used.
  - The alternative is Δ = 0.1 s (T = 20), which reuses `m11_recovery.dataset` unchanged (decision D2).
- **Masks.** A grid point is masked out when it lies outside the candidate chain's span or the period. These are the only reasons, and both are identity-free given the chain. Masked positions are set to 0 and never NaN.
- **Minimum observed points.** At least 3 per tracklet, as in the existing `motion.usable`. This is guaranteed by the 0.5 s continuity margin (≥ 7 points); a builder assertion checks it.

## 4. Boundary handling

- **Chain ends inside the context.** The tracklet is clipped and the tail is masked. The model never sees the neighbouring chain, so a jump into or out of the chain is never in the input.
- **Period edges.** Stage 2 already rejects windows that leave the period.
- **Frame normalisation.** This is computed from the incoming tracklets only (existing `dataset.frame`):
  - origin: the midpoint of their last observed points;
  - x axis: their summed velocity.

  Outgoing tracks are expressed in that frame, so the normalisation cannot carry the label. Absolute pitch coordinates are not input in v1 (decision D3).

## 5. Ground-truth label

- **Label rule.** The incoming tracklet whose player is P continues as the outgoing tracklet of the same chain. y = 0 if the outgoing order puts P's outgoing tracklet in the same slot index as P's incoming one.
- **What decides the label.** Only the reference chain identity, offline. Never the crossing class, d_min or any F1–F4 state.
- **Label reliability.**
  - Ensured by the Stage 2 ACCEPTED outcome: continuous chain, no identity-risk flag on used samples, no corruption, plausible motion.
  - Near-coincidence (d_min < 0.05 m) and jump-transfer-risk candidates are kept, and metrics are reported with and without them (decision D4).

## 6. Ordering and label balance

- Incoming order and outgoing order each come from an independent bit of `sha256(seed | candidate_id | gap | draw)`. The existing `views.make_sample` does exactly this.
- This gives y balanced in expectation (about 50/50, checked per split). Neither order depends on slot, time or file order.
- The incoming order is randomised too, because `slot_a < slot_b` is an identity artefact.
- Balance between crossing and non-crossing encounters is a separate question (benchmark plan §4). It does not affect y balance.
- **Augmentation at train time only.** Exchanging the outgoing order (y → 1 − y) and reflecting the y axis, as the existing `GBM` and `Hybrid` already do. Predictions are symmetrised over the outgoing order, so p(swap outgoing) = 1 − p.

## 7. Leakage controls

| Risk | Control |
|---|---|
| Identities, team or status as inputs | not written to the sample table; a test asserts the input schema exactly |
| Crossing class used for labels or inputs | label from chain identity only; crossing kept as a metadata stratum |
| Hidden-gap positions | not stored in the sample at all |
| Gap centred on the closest approach | shift-robustness table (§1) |
| Slot / file order | hashed independent orders (§6) |
| Absolute frame or pitch side | sample frame from incoming tracks only (§4) |
| Future information | the outgoing context is the *observable* future after the gap, which is the task definition. Nothing after t* + g/2 + C is used. Normalisation constants (feature scaling, Platt) are fitted inside each training fold only. |
| Dependent samples across folds | folds and the held-out set are by match; overlap groups never cross matches (Stage 2) |
| **Tracking-status signature shortcut** (VISIBLE jitter ~1 cm vs ESTIMATED ~8 cm; a model could pair tracklets by noise texture instead of motion) | Status is never an input, but its texture is in the positions. Report metrics by status combination. Run an ablation that applies a light smoothing or noise injection to all tracks (decision D5). The size of this risk is quantified in the Stage 2 report. |

## 8. Overlapping and dependent samples

- **Folds.** All samples of a match stay in one fold (grouped CV by match), so overlapping samples never straddle train and validation.
- **Training weights.** Optionally weight each sample by 1 / (size of its overlap group), so dense scrambles don't dominate (decision D6). Default: unweighted, with weighted training as an ablation.
- **Reporting.** Metrics on all samples, plus the greedy independent subset (`independent` = True) as the primary unbiased estimate. Confidence intervals by match bootstrap (existing `evaluate.match_bootstrap`).

## 9. The frozen split

- The builder reads `split.json` through `split.load_split`, which verifies the hash. Every sample carries `split`.
- Training and CV code receive only `train_cv` rows. The held-out table is a separate file, opened once by a final-evaluation command after all methodological decisions are frozen, as in the earlier Step 6/7 protocol.
- Thresholds, calibrators and model selection use the train/CV folds only.

## 10. Is the sample size sufficient?

Measured numbers are in the Stage 2 report (primary gap 0.8 s, train/CV matches). With tens of thousands of accepted samples and roughly 70% of them independent, sample size is sufficient for:
- **Tabular models:** LR, KNN, RF and GBM on about 60 engineered features. The existing GBM reached its plateau with about 20k rows.
- **RBF SVM:** with Nyström features or a subsample, because a kernel SVM scales as O(n²) in memory.
- **Small recurrent models:** an MLP, LSTM or GRU with roughly 20–60k parameters, and the existing PairNet GRU at hidden size 64. 10k+ independent sequences with symmetry augmentation are adequate for these sizes. Large sequence models are not justified.

Learning curves (benchmark plan) will test this empirically.

## 11. Efficient builder (no full dataset in memory)

- **Per-match processing**, reusing the Stage 2 runner (`local.run_matches`): partial directory, atomic writes, `match_done.json` hashes, `--resume`, and crash-and-resume tested.
- **Per match:**
  1. Load the Step 1 cache and segments (about 7 s).
  2. Read the accepted candidates.
  3. Build the knot tracks once per chain.
  4. Write a `samples_g{gap}.npz` shard: float32 arrays, about 1,000 samples × 4 tracks × 30 points.
  5. Write a `samples_meta.parquet` shard.
- **Size:** about 1 MB per match per gap and about 0.1 GB in total.
- **Time:** about 10–20 s per match, roughly 10 min for 40 matches.
- **Training** loads only the train/CV shards. The held-out shards stay closed.

## 12. Reuse from `m11_recovery`

| Reuse | Module | Change needed |
|---|---|---|
| Hashed independent orders, `Sample` | `views.make_sample`, `views.Sample` | a new source (Stage 2 chains instead of the v3 store); label convention flip |
| Motion fits, baselines (nearest, CV and CA, forward and bidirectional), pair costs, antisymmetric features | `motion.Kin`, `pair_costs`, `pair_matrices`, `features` | none (they read a `Sample`) |
| Sample frame, sequence tensors | `dataset.frame`, `dataset.tensor`, `model_inputs` | grid length T, mask as a channel |
| Baseline-to-probability, GBM with antisymmetric augmentation, GRU PairNet | `models.Baseline`, `GBM`, `Hybrid` | label convention; a split-aware fit |
| Metrics, abstention, strata, match bootstrap, paired difference | `evaluate` | label convention; new strata keys (cohorts, crossing, status, overlap) |
| Grouped CV loop | `experiments.lomo` | K grouped folds instead of LOMO; train_cv only |

New code:
- A Stage 3 builder module (`window_intersections/samples.py`).
- An `m11_recovery` population adapter for the new shards.
- Tests: schema has no identity columns; label correctness on synthetic encounters; order balance; split protection (the held-out set is not readable by the trainer); the shift-table continuity check; and determinism.
